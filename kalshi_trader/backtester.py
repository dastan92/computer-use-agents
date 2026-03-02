"""Backtesting engine for Kalshi prediction market strategies.

The core idea: take settled markets (where we know the outcome), ask Claude
what probability it would have assigned BEFORE settlement, compare to the
market price at the time, and simulate what trades we would have made.

This measures:
1. Calibration: Is Claude's probability estimation accurate?
2. Edge: Would Claude's edges have been profitable after fees?
3. Contamination risk: Could Claude have "known" outcomes from training data?

v2: Anti-contamination safeguards
- Time window filter: only test markets after Claude's training cutoff
- Blind mode: analyst hides market price to prevent anchoring
- Contamination flags: warns about markets likely in training data
"""

import json
import logging
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .analyst import ClaudeAnalyst
from .config import AppConfig, BacktestConfig
from .models import (
    BacktestResult,
    BacktestTrade,
    Market,
    MarketAnalysis,
    OrderAction,
    Side,
)

logger = logging.getLogger(__name__)

# High-profile events likely in Claude's training data
HIGH_PROFILE_KEYWORDS = [
    "presidential", "super bowl", "world series", "world cup",
    "fed rate", "fomc", "election", "inauguration", "oscar",
    "grammy", "nobel", "state of the union",
]


def contamination_score(market: Market, min_date: datetime) -> float:
    """Estimate how likely Claude saw this outcome in training data.

    Returns 0.0 (safe) to 1.0 (very likely contaminated).
    """
    score = 0.0

    if market.close_time:
        close = market.close_time
        if hasattr(close, "tzinfo") and close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        if hasattr(min_date, "tzinfo") and min_date.tzinfo is None:
            min_date = min_date.replace(tzinfo=timezone.utc)
        if close < min_date:
            score += 0.5
            days_before = (min_date - close).days
            score += min(0.3, days_before / 365 * 0.3)

    title_lower = market.title.lower()
    for keyword in HIGH_PROFILE_KEYWORDS:
        if keyword in title_lower:
            score += 0.2
            break

    if market.volume > 50_000:
        score += 0.1

    return min(1.0, score)


class Backtester:
    """Backtest AI trading strategies against historical Kalshi markets.

    Supports anti-contamination modes:
    - Time filtering: skip markets before Claude's knowledge cutoff
    - Blind analysis: Claude doesn't see market prices (via analyst blind_mode)
    - Contamination scoring: flag and optionally exclude risky markets
    """

    def __init__(self, config: AppConfig, analyst: ClaudeAnalyst):
        self.config = config
        self.bt_config = config.backtest
        self.trading_config = config.trading
        self.analyst = analyst

        self.balance = self.bt_config.initial_balance
        self.trades: list[BacktestTrade] = []
        self.analyses: list[MarketAnalysis] = []
        self.peak_balance = self.balance
        self.max_drawdown = 0.0
        self.skipped_contaminated = 0

    def reset(self):
        """Reset backtester state for a fresh run."""
        self.balance = self.bt_config.initial_balance
        self.trades = []
        self.analyses = []
        self.peak_balance = self.balance
        self.max_drawdown = 0.0
        self.skipped_contaminated = 0

    def _filter_markets(self, markets: list[Market]) -> list[Market]:
        """Filter markets for backtesting, removing contaminated ones.

        Applies:
        1. Must be settled with a result
        2. Time window filter (after min_close_date)
        3. Contamination score filter
        """
        min_date_str = self.bt_config.min_close_date
        min_date = datetime.strptime(min_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        filtered = []
        for m in markets:
            if m.result not in ("yes", "no"):
                continue

            # Time window filter
            if m.close_time:
                close = m.close_time
                if hasattr(close, "tzinfo") and close.tzinfo is None:
                    close = close.replace(tzinfo=timezone.utc)
                if close < min_date:
                    self.skipped_contaminated += 1
                    logger.debug(
                        "Skipping %s (closed %s, before cutoff %s)",
                        m.ticker, close.date(), min_date.date(),
                    )
                    continue

            # Contamination score check
            c_score = contamination_score(m, min_date)
            if c_score > 0.7:
                self.skipped_contaminated += 1
                logger.debug(
                    "Skipping %s (contamination score %.2f)", m.ticker, c_score
                )
                continue

            filtered.append(m)

        logger.info(
            "Filtered %d -> %d markets (skipped %d for contamination risk)",
            len(markets), len(filtered), self.skipped_contaminated,
        )
        return filtered

    def _calculate_position_size(self, edge: float, confidence: float) -> int:
        """Calculate position size using fractional Kelly criterion.

        Kelly fraction = edge / odds
        For binary markets at price p: edge = (ai_prob - p), odds = (1/p - 1)
        We use quarter-Kelly for safety.
        """
        if edge <= 0 or confidence < 0.3:
            return 0

        kelly_pct = edge * confidence * self.trading_config.kelly_fraction
        kelly_pct = min(kelly_pct, self.trading_config.max_position_pct)

        max_dollars = self.balance * kelly_pct
        contracts = max(1, int(max_dollars))

        return contracts

    def _simulate_trade(
        self, market: Market, analysis: MarketAnalysis
    ) -> Optional[BacktestTrade]:
        """Simulate a trade based on analysis of a settled market."""
        edge = analysis.edge
        abs_edge = abs(edge)

        if abs_edge < self.trading_config.min_edge_threshold:
            return None

        if analysis.confidence < 0.3:
            return None

        if edge > 0:
            side = Side.YES
            entry_price = market.yes_ask if market.yes_ask else market.yes_mid
        else:
            side = Side.NO
            entry_price = market.no_ask if market.no_ask else (100 - market.yes_mid)

        entry_price += self.bt_config.slippage_cents

        size = self._calculate_position_size(abs_edge, analysis.confidence)
        if size == 0:
            return None

        cost = (entry_price / 100.0) * size + self.bt_config.commission_per_contract * size
        if cost > self.balance * self.trading_config.max_position_pct:
            size = max(
                1,
                int(
                    (self.balance * self.trading_config.max_position_pct)
                    / (entry_price / 100.0 + self.bt_config.commission_per_contract)
                ),
            )
            cost = (entry_price / 100.0) * size + self.bt_config.commission_per_contract * size

        if cost > self.balance:
            return None

        settled_result = market.result
        if not settled_result:
            return None

        if side == Side.YES:
            exit_price = 100 if settled_result == "yes" else 0
        else:
            exit_price = 100 if settled_result == "no" else 0

        pnl = ((exit_price - entry_price) / 100.0) * size
        pnl -= self.bt_config.commission_per_contract * size * 2

        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        trade = BacktestTrade(
            ticker=market.ticker,
            entry_time=market.close_time or datetime.utcnow(),
            exit_time=market.expiration_time or market.close_time,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            settled_result=settled_result,
        )

        return trade

    def run(
        self,
        markets: list[Market],
        batch_size: int = 5,
        progress_callback=None,
        skip_contamination_filter: bool = False,
    ) -> BacktestResult:
        """Run a backtest over a list of settled markets.

        Args:
            markets: Settled markets with known outcomes
            batch_size: Number of markets to analyze per Claude call
            progress_callback: Optional callback(completed, total) for progress
            skip_contamination_filter: If True, test ALL markets (for comparison)
        """
        self.reset()

        if skip_contamination_filter:
            settled = [m for m in markets if m.result in ("yes", "no")]
            logger.warning("Contamination filter DISABLED - results may be inflated")
        else:
            settled = self._filter_markets(markets)

        if not settled:
            logger.warning("No markets passed filters for backtesting")
            return self._build_result(settled)

        logger.info("Starting backtest with %d markets", len(settled))
        total = len(settled)

        for i in range(0, total, batch_size):
            batch = settled[i : i + batch_size]

            analyses = self.analyst.analyze_markets_batch(batch)
            self.analyses.extend(analyses)

            analysis_map = {a.ticker: a for a in analyses}
            for market in batch:
                analysis = analysis_map.get(market.ticker)
                if not analysis:
                    continue

                trade = self._simulate_trade(market, analysis)
                if trade:
                    self.trades.append(trade)
                    logger.info(
                        "Trade: %s %s @ %.0fc -> %s | PnL: $%.2f | Balance: $%.2f",
                        trade.side.value.upper(),
                        trade.ticker,
                        trade.entry_price,
                        trade.settled_result,
                        trade.pnl,
                        self.balance,
                    )

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return self._build_result(settled)

    def run_contamination_comparison(
        self, markets: list[Market], batch_size: int = 5
    ) -> dict:
        """Run backtest TWICE: with and without contamination filter.

        This directly measures whether Claude's performance is inflated
        by knowledge contamination. If results are similar, contamination
        is not a major factor. If unfiltered is much better, be suspicious.
        """
        print("\n  Running CLEAN backtest (post-cutoff markets only)...")
        clean_result = self.run(markets, batch_size, skip_contamination_filter=False)
        clean_trades = list(self.trades)
        clean_analyses = list(self.analyses)

        print("\n  Running UNFILTERED backtest (all markets)...")
        unfiltered_result = self.run(markets, batch_size, skip_contamination_filter=True)

        contamination_delta = unfiltered_result.win_rate - clean_result.win_rate

        print(f"\n{'=' * 60}")
        print("  CONTAMINATION COMPARISON")
        print(f"{'=' * 60}")
        print(f"  Clean Win Rate:      {clean_result.win_rate:.1%} ({clean_result.total_trades} trades)")
        print(f"  Unfiltered Win Rate: {unfiltered_result.win_rate:.1%} ({unfiltered_result.total_trades} trades)")
        print(f"  Delta:               {contamination_delta:+.1%}")

        if contamination_delta > 0.10:
            print("  WARNING: Unfiltered performs >10% better. Likely contamination.")
            print("  Trust ONLY the clean backtest results.")
        elif contamination_delta > 0.05:
            print("  CAUTION: Some contamination signal detected. Prefer clean results.")
        else:
            print("  GOOD: Minimal contamination signal. Results appear trustworthy.")

        # Restore clean state for reporting
        self.trades = clean_trades
        self.analyses = clean_analyses

        return {
            "clean": clean_result,
            "unfiltered": unfiltered_result,
            "contamination_delta": contamination_delta,
        }

    def _build_result(self, markets: list[Market]) -> BacktestResult:
        """Build the final backtest result summary."""
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)

        sharpe = None
        if len(self.trades) >= 5:
            returns = [t.pnl / self.bt_config.initial_balance for t in self.trades]
            if statistics.stdev(returns) > 0:
                sharpe = (statistics.mean(returns) / statistics.stdev(returns)) * math.sqrt(252)

        avg_edge = 0
        if self.analyses:
            edges = [abs(a.edge) for a in self.analyses if abs(a.edge) > 0]
            avg_edge = statistics.mean(edges) if edges else 0

        start_date = min(
            (m.close_time for m in markets if m.close_time),
            default=datetime.utcnow(),
        )
        end_date = max(
            (m.close_time for m in markets if m.close_time),
            default=datetime.utcnow(),
        )

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.bt_config.initial_balance,
            final_balance=self.balance,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=total_pnl,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=len(winning) / len(self.trades) if self.trades else 0,
            avg_edge=avg_edge,
            trades=self.trades,
        )

    def print_report(self, result: BacktestResult):
        """Print a formatted backtest report."""
        print(f"\n{'=' * 60}")
        print("  BACKTEST REPORT")
        print(f"{'=' * 60}")
        print(f"  Period:           {result.start_date:%Y-%m-%d} to {result.end_date:%Y-%m-%d}")
        print(f"  Initial Balance:  ${result.initial_balance:,.2f}")
        print(f"  Final Balance:    ${result.final_balance:,.2f}")
        print(f"  Total P&L:        ${result.total_pnl:,.2f} ({result.return_pct:+.1f}%)")
        print(f"  Max Drawdown:     {result.max_drawdown:.1%}")
        print(f"-" * 60)
        print(f"  Total Trades:     {result.total_trades}")
        print(f"  Winning Trades:   {result.winning_trades}")
        print(f"  Losing Trades:    {result.losing_trades}")
        print(f"  Win Rate:         {result.win_rate:.1%}")
        print(f"  Average Edge:     {result.avg_edge:.1%}")
        if result.sharpe_ratio is not None:
            print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"-" * 60)

        # Anti-contamination info
        if self.skipped_contaminated > 0:
            print(f"  Contamination:    Skipped {self.skipped_contaminated} risky markets")
        print(f"  Analyst Mode:     {'BLIND (no price)' if self.analyst.blind_mode else 'Standard'}")

        # Claude API usage
        usage = self.analyst.get_usage_stats()
        print(f"  Claude API Calls: {usage['calls']}")
        print(f"  Tokens Used:      {usage['input_tokens'] + usage['output_tokens']:,}")
        print(f"  Est. API Cost:    ${usage['estimated_cost_usd']:.2f}")
        print(f"{'=' * 60}")

        if self.analyses:
            self._print_calibration()
        if self.trades:
            self._print_top_trades()

    def _print_calibration(self):
        """Print calibration analysis."""
        print(f"\n  CALIBRATION ANALYSIS")
        print(f"-" * 60)

        buckets = {
            "0-20%": {"predicted": [], "actual": []},
            "20-40%": {"predicted": [], "actual": []},
            "40-60%": {"predicted": [], "actual": []},
            "60-80%": {"predicted": [], "actual": []},
            "80-100%": {"predicted": [], "actual": []},
        }

        for analysis in self.analyses:
            matching_trades = [t for t in self.trades if t.ticker == analysis.ticker]
            if not matching_trades:
                continue

            trade = matching_trades[0]
            actual = 1.0 if trade.settled_result == "yes" else 0.0
            pred = analysis.ai_probability

            if pred < 0.2:
                key = "0-20%"
            elif pred < 0.4:
                key = "20-40%"
            elif pred < 0.6:
                key = "40-60%"
            elif pred < 0.8:
                key = "60-80%"
            else:
                key = "80-100%"

            buckets[key]["predicted"].append(pred)
            buckets[key]["actual"].append(actual)

        # Calculate Brier score components
        all_pred = []
        all_actual = []

        for bucket, data in buckets.items():
            if data["predicted"]:
                avg_pred = statistics.mean(data["predicted"])
                avg_actual = statistics.mean(data["actual"])
                n = len(data["predicted"])
                calibration_error = abs(avg_pred - avg_actual)
                indicator = "OK" if calibration_error < 0.1 else "DRIFT"
                print(
                    f"  {bucket:>8s}: Predicted={avg_pred:.1%}  "
                    f"Actual={avg_actual:.1%}  (n={n}) [{indicator}]"
                )
                all_pred.extend(data["predicted"])
                all_actual.extend(data["actual"])
            else:
                print(f"  {bucket:>8s}: No data")

        # Overall Brier score
        if all_pred:
            brier = statistics.mean(
                (p - a) ** 2 for p, a in zip(all_pred, all_actual)
            )
            print(f"\n  Brier Score:      {brier:.4f} (lower is better, 0.25 = coin flip)")

    def _print_top_trades(self):
        """Print best and worst trades."""
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl, reverse=True)

        print(f"\n  TOP 5 WINNING TRADES")
        print(f"-" * 60)
        for t in sorted_trades[:5]:
            if t.pnl > 0:
                print(
                    f"  {t.side.value.upper():3s} {t.ticker:30s} "
                    f"Entry={t.entry_price:.0f}c PnL=${t.pnl:+.2f}"
                )

        print(f"\n  TOP 5 LOSING TRADES")
        print(f"-" * 60)
        for t in sorted_trades[-5:]:
            if t.pnl <= 0:
                print(
                    f"  {t.side.value.upper():3s} {t.ticker:30s} "
                    f"Entry={t.entry_price:.0f}c PnL=${t.pnl:+.2f}"
                )

    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "result": result.model_dump(mode="json"),
            "analyses": [a.model_dump(mode="json") for a in self.analyses],
            "usage": self.analyst.get_usage_stats(),
            "config": {
                "blind_mode": self.analyst.blind_mode,
                "anonymize": self.analyst.anonymize,
                "min_close_date": self.bt_config.min_close_date,
                "skipped_contaminated": self.skipped_contaminated,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved backtest results to %s", filepath)
