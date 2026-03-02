"""Trading strategy framework.

Combines Claude's analysis with position sizing, risk management,
and order execution logic.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .analyst import ClaudeAnalyst
from .config import AppConfig
from .kalshi_client import KalshiClient
from .models import (
    Market,
    MarketAnalysis,
    OrderAction,
    Side,
    TradeSignal,
)

logger = logging.getLogger(__name__)


class TradingStrategy:
    """AI-powered trading strategy for Kalshi prediction markets.

    Flow:
    1. Scan markets for candidates (sufficient volume, time to expiry, etc.)
    2. Send candidates to Claude for probability analysis
    3. Filter for markets where Claude sees an edge
    4. Size positions using fractional Kelly criterion
    5. Generate trade signals
    """

    def __init__(
        self,
        config: AppConfig,
        client: KalshiClient,
        analyst: ClaudeAnalyst,
        tracker=None,
    ):
        self.config = config
        self.client = client
        self.analyst = analyst
        self.tracker = tracker  # optional Tracker for persistent logging
        self.trading_config = config.trading

    def scan_candidates(self) -> list[Market]:
        """Scan for tradeable market candidates."""
        logger.info("Scanning for market candidates...")

        all_markets = []
        cursor = None

        # Fetch open markets
        while len(all_markets) < 200:
            markets, cursor = self.client.get_markets(
                limit=100, cursor=cursor, status="open"
            )
            all_markets.extend(markets)
            if not cursor:
                break

        # Filter candidates
        candidates = []
        now = datetime.now(timezone.utc)

        for m in all_markets:
            # Skip low volume
            if m.volume < self.trading_config.min_volume:
                continue

            # Skip if no bid/ask (illiquid)
            if not m.yes_bid or not m.yes_ask:
                continue

            # Skip extreme prices (very likely or unlikely - small edge)
            if m.yes_mid < 5 or m.yes_mid > 95:
                continue

            # Skip wide spreads (> 10 cents)
            if m.spread > 10:
                continue

            # Check time to expiry
            if m.close_time:
                close_time = m.close_time
                if isinstance(close_time, str):
                    close_time = datetime.fromisoformat(close_time)
                if hasattr(close_time, "tzinfo") and close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                days_to_expiry = (close_time - now).days
                if days_to_expiry > self.trading_config.max_days_to_expiry:
                    continue
                if days_to_expiry < 0:
                    continue

            candidates.append(m)

        logger.info(
            "Found %d candidates from %d total markets",
            len(candidates),
            len(all_markets),
        )
        return candidates

    def analyze_candidates(
        self, candidates: list[Market], max_analyze: int = 20
    ) -> list[MarketAnalysis]:
        """Send candidates to Claude for analysis."""
        # Prioritize by volume (more liquid = better)
        sorted_candidates = sorted(candidates, key=lambda m: m.volume, reverse=True)
        to_analyze = sorted_candidates[:max_analyze]

        logger.info("Analyzing %d markets with Claude...", len(to_analyze))

        # Batch analyze for efficiency
        analyses = self.analyst.analyze_markets_batch(to_analyze)

        # Sort by absolute edge
        analyses.sort(key=lambda a: abs(a.edge), reverse=True)

        return analyses

    def generate_signals(
        self, analyses: list[MarketAnalysis], balance: float
    ) -> list[TradeSignal]:
        """Generate trade signals from analyses.

        Only generates signals where:
        - Edge exceeds minimum threshold
        - Confidence is sufficient
        - Position sizing is valid
        """
        signals = []

        for analysis in analyses:
            # Check edge threshold
            if analysis.abs_edge < self.trading_config.min_edge_threshold:
                continue

            # Check confidence
            if analysis.confidence < 0.3:
                continue

            # Determine side and price
            if analysis.edge > 0:
                side = Side.YES
                target_price = analysis.market_probability * 100  # buy at market
            else:
                side = Side.NO
                target_price = (1 - analysis.market_probability) * 100

            # Kelly position sizing
            kelly_raw = analysis.abs_edge * analysis.confidence
            kelly_fraction = kelly_raw * self.trading_config.kelly_fraction
            kelly_fraction = min(kelly_fraction, self.trading_config.max_position_pct)

            dollars = balance * kelly_fraction
            contracts = max(1, int(dollars / (target_price / 100)))

            # Cap at max position
            max_contracts = int(
                balance * self.trading_config.max_position_pct / (target_price / 100)
            )
            contracts = min(contracts, max(1, max_contracts))

            signal = TradeSignal(
                ticker=analysis.ticker,
                side=side,
                action=OrderAction.BUY,
                target_price=round(target_price),
                size=contracts,
                edge=analysis.edge,
                confidence=analysis.confidence,
                reasoning=analysis.reasoning,
            )
            signals.append(signal)

        # Sort by edge * confidence (expected value)
        signals.sort(key=lambda s: abs(s.edge) * s.confidence, reverse=True)

        # Limit to max concurrent positions
        signals = signals[: self.trading_config.max_concurrent_positions]

        return signals

    def execute_signals(
        self, signals: list[TradeSignal], dry_run: bool = True
    ) -> list[dict]:
        """Execute trade signals. Set dry_run=False for live trading."""
        results = []

        for signal in signals:
            print(
                f"\n{'[DRY RUN] ' if dry_run else ''}Signal: "
                f"{signal.action.value.upper()} {signal.size} "
                f"{signal.side.value.upper()} @ {signal.target_price}c "
                f"on {signal.ticker}"
            )
            print(f"  Edge: {signal.edge:+.1%} | Confidence: {signal.confidence:.1%}")
            print(f"  Reasoning: {signal.reasoning[:120]}...")

            mode = "dry_run" if dry_run else "live"
            fees = self.config.backtest.commission_per_contract * signal.size

            if dry_run:
                results.append(
                    {
                        "signal": signal.model_dump(mode="json"),
                        "status": "dry_run",
                    }
                )
            else:
                try:
                    order = self.client.place_order(
                        ticker=signal.ticker,
                        side=signal.side,
                        action=signal.action,
                        count=signal.size,
                        price=int(signal.target_price),
                    )
                    results.append(
                        {
                            "signal": signal.model_dump(mode="json"),
                            "order": order,
                            "status": "submitted",
                        }
                    )
                    logger.info("Order submitted for %s", signal.ticker)
                except Exception as e:
                    logger.error("Failed to place order for %s: %s", signal.ticker, e)
                    results.append(
                        {
                            "signal": signal.model_dump(mode="json"),
                            "status": "failed",
                            "error": str(e),
                        }
                    )

            # Log trade to persistent tracker
            if self.tracker:
                self.tracker.log_trade(
                    ticker=signal.ticker,
                    side=signal.side.value,
                    action=signal.action.value,
                    price_cents=signal.target_price,
                    contracts=signal.size,
                    fees_usd=fees,
                    mode=mode,
                    ai_probability=None,
                    market_probability=None,
                    edge=signal.edge,
                    confidence=signal.confidence,
                )

        return results

    def run_scan_and_trade(self, dry_run: bool = True) -> list[dict]:
        """Full pipeline: scan -> analyze -> signal -> execute."""
        # Get balance
        balance = self.config.backtest.initial_balance
        if not dry_run:
            try:
                balance = self.client.get_balance()
                logger.info("Account balance: $%.2f", balance)
            except Exception:
                logger.warning("Could not fetch balance, using default")

        # Scan
        candidates = self.scan_candidates()
        if not candidates:
            print("No tradeable candidates found.")
            return []

        # Analyze
        analyses = self.analyze_candidates(candidates)

        # Print analysis summary
        print(f"\n{'=' * 60}")
        print("  MARKET ANALYSIS SUMMARY")
        print(f"{'=' * 60}")
        for a in analyses[:10]:
            direction = "YES ↑" if a.edge > 0 else "NO ↑"
            print(
                f"  {a.ticker:30s} Edge={a.edge:+.1%} "
                f"Conf={a.confidence:.0%} -> {direction}"
            )

        # Generate and execute signals
        signals = self.generate_signals(analyses, balance)
        if not signals:
            print("\nNo signals meet the threshold criteria.")
            return []

        print(f"\n  Generated {len(signals)} trade signals")
        return self.execute_signals(signals, dry_run=dry_run)
