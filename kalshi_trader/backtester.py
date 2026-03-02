"""Backtesting engine for Kalshi prediction market strategies.

The core idea: take settled markets (where we know the outcome), ask Claude
what probability it would have assigned BEFORE settlement, compare to the
market price at the time, and simulate what trades we would have made.

This measures two things:
1. Calibration: Is Claude's probability estimation accurate?
2. Edge: Would Claude's edges have been profitable after fees?
"""

import json
import logging
import math
import statistics
from datetime import datetime
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


class Backtester:
    """Backtest AI trading strategies against historical Kalshi markets."""

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

    def reset(self):
        """Reset backtester state for a fresh run."""
        self.balance = self.bt_config.initial_balance
        self.trades = []
        self.analyses = []
        self.peak_balance = self.balance
        self.max_drawdown = 0.0

    def _calculate_position_size(self, edge: float, confidence: float) -> int:
        """Calculate position size using fractional Kelly criterion.

        Kelly fraction = edge / odds
        For binary markets at price p: edge = (ai_prob - p), odds = (1/p - 1)
        We use quarter-Kelly for safety.
        """
        if edge <= 0 or confidence < 0.3:
            return 0

        # Kelly sizing
        kelly_pct = edge * confidence * self.trading_config.kelly_fraction

        # Cap at max position size
        kelly_pct = min(kelly_pct, self.trading_config.max_position_pct)

        # Convert to number of contracts (each ~$1 max value)
        max_dollars = self.balance * kelly_pct
        contracts = max(1, int(max_dollars))

        return contracts

    def _simulate_trade(
        self, market: Market, analysis: MarketAnalysis
    ) -> Optional[BacktestTrade]:
        """Simulate a trade based on analysis of a settled market.

        Returns a BacktestTrade if the analysis suggests a trade, None otherwise.
        """
        edge = analysis.edge
        abs_edge = abs(edge)

        # Check minimum edge threshold
        if abs_edge < self.trading_config.min_edge_threshold:
            return None

        # Check confidence threshold
        if analysis.confidence < 0.3:
            return None

        # Determine trade side
        if edge > 0:
            # AI thinks YES is underpriced -> buy YES
            side = Side.YES
            entry_price = market.yes_ask if market.yes_ask else market.yes_mid
        else:
            # AI thinks NO is underpriced -> buy NO
            side = Side.NO
            entry_price = market.no_ask if market.no_ask else (100 - market.yes_mid)

        # Add slippage
        entry_price += self.bt_config.slippage_cents

        # Calculate size
        size = self._calculate_position_size(abs_edge, analysis.confidence)
        if size == 0:
            return None

        # Check if we can afford it
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

        # Determine outcome based on settlement
        settled_result = market.result  # "yes" or "no"
        if not settled_result:
            return None

        # Calculate P&L
        if side == Side.YES:
            if settled_result == "yes":
                exit_price = 100  # YES wins, pays $1
            else:
                exit_price = 0  # YES loses, pays $0
        else:
            if settled_result == "no":
                exit_price = 100  # NO wins, pays $1
            else:
                exit_price = 0  # NO loses, pays $0

        pnl = ((exit_price - entry_price) / 100.0) * size
        pnl -= self.bt_config.commission_per_contract * size * 2  # entry + exit fees

        # Update balance
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
    ) -> BacktestResult:
        """Run a backtest over a list of settled markets.

        Args:
            markets: Settled markets with known outcomes
            batch_size: Number of markets to analyze per Claude call
            progress_callback: Optional callback(completed, total) for progress
        """
        self.reset()

        # Filter to only settled markets with results
        settled = [m for m in markets if m.result in ("yes", "no")]
        if not settled:
            logger.warning("No settled markets found for backtesting")
            return self._build_result(settled)

        logger.info("Starting backtest with %d settled markets", len(settled))
        total = len(settled)

        # Process in batches to reduce API calls
        for i in range(0, total, batch_size):
            batch = settled[i : i + batch_size]

            # Get Claude's analysis for this batch
            analyses = self.analyst.analyze_markets_batch(batch)
            self.analyses.extend(analyses)

            # Simulate trades for each analyzed market
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

    def _build_result(self, markets: list[Market]) -> BacktestResult:
        """Build the final backtest result summary."""
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)

        # Calculate Sharpe ratio if we have enough trades
        sharpe = None
        if len(self.trades) >= 5:
            returns = [t.pnl / self.bt_config.initial_balance for t in self.trades]
            if statistics.stdev(returns) > 0:
                sharpe = (statistics.mean(returns) / statistics.stdev(returns)) * math.sqrt(
                    252
                )  # annualized

        # Average edge on trades taken
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
        print("\n" + "=" * 60)
        print("  BACKTEST REPORT")
        print("=" * 60)
        print(f"  Period:           {result.start_date:%Y-%m-%d} to {result.end_date:%Y-%m-%d}")
        print(f"  Initial Balance:  ${result.initial_balance:,.2f}")
        print(f"  Final Balance:    ${result.final_balance:,.2f}")
        print(f"  Total P&L:        ${result.total_pnl:,.2f} ({result.return_pct:+.1f}%)")
        print(f"  Max Drawdown:     {result.max_drawdown:.1%}")
        print("-" * 60)
        print(f"  Total Trades:     {result.total_trades}")
        print(f"  Winning Trades:   {result.winning_trades}")
        print(f"  Losing Trades:    {result.losing_trades}")
        print(f"  Win Rate:         {result.win_rate:.1%}")
        print(f"  Average Edge:     {result.avg_edge:.1%}")
        if result.sharpe_ratio is not None:
            print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print("-" * 60)

        # Claude API usage
        usage = self.analyst.get_usage_stats()
        print(f"  Claude API Calls: {usage['calls']}")
        print(f"  Tokens Used:      {usage['input_tokens'] + usage['output_tokens']:,}")
        print(f"  Est. API Cost:    ${usage['estimated_cost_usd']:.2f}")
        print("=" * 60)

        # Calibration analysis
        if self.analyses:
            self._print_calibration()

        # Top trades
        if self.trades:
            self._print_top_trades()

    def _print_calibration(self):
        """Print calibration analysis - how well-calibrated are Claude's estimates?"""
        print("\n  CALIBRATION ANALYSIS")
        print("-" * 60)

        # Bucket analyses by AI probability
        buckets = {
            "0-20%": {"predicted": [], "actual": []},
            "20-40%": {"predicted": [], "actual": []},
            "40-60%": {"predicted": [], "actual": []},
            "60-80%": {"predicted": [], "actual": []},
            "80-100%": {"predicted": [], "actual": []},
        }

        for analysis in self.analyses:
            # Find corresponding trade to get actual result
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

        for bucket, data in buckets.items():
            if data["predicted"]:
                avg_pred = statistics.mean(data["predicted"])
                avg_actual = statistics.mean(data["actual"])
                n = len(data["predicted"])
                print(f"  {bucket:>8s}: Predicted={avg_pred:.1%}  Actual={avg_actual:.1%}  (n={n})")
            else:
                print(f"  {bucket:>8s}: No data")

    def _print_top_trades(self):
        """Print best and worst trades."""
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl, reverse=True)

        print("\n  TOP 5 WINNING TRADES")
        print("-" * 60)
        for t in sorted_trades[:5]:
            if t.pnl > 0:
                print(
                    f"  {t.side.value.upper():3s} {t.ticker:30s} "
                    f"Entry={t.entry_price:.0f}c PnL=${t.pnl:+.2f}"
                )

        print("\n  TOP 5 LOSING TRADES")
        print("-" * 60)
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
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved backtest results to %s", filepath)
