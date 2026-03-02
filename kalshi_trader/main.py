#!/usr/bin/env python3
"""Kalshi AI Trading & Backtesting System - CLI Entry Point.

Usage:
    python -m kalshi_trader scan              # Scan markets and show AI analysis
    python -m kalshi_trader trade             # Dry-run trade signals
    python -m kalshi_trader trade --live      # Live trading (use with caution!)
    python -m kalshi_trader backtest          # Backtest against settled markets
    python -m kalshi_trader backtest --blind  # Backtest with blind mode (anti-anchoring)
    python -m kalshi_trader backtest --check-contamination  # Run contamination comparison
    python -m kalshi_trader collect           # Collect market data for backtesting
    python -m kalshi_trader analyze TICKER    # Deep-analyze a single market
    python -m kalshi_trader costs             # View daily cost & P&L report
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .analyst import ClaudeAnalyst
from .backtester import Backtester
from .config import AppConfig
from .kalshi_client import KalshiClient
from .strategy import TradingStrategy
from .tracker import Tracker


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_scan(args, config: AppConfig):
    """Scan markets and show Claude's analysis."""
    tracker = Tracker()
    client = KalshiClient(config.kalshi)
    analyst = ClaudeAnalyst(config.claude, tracker=tracker)
    strategy = TradingStrategy(config, client, analyst, tracker=tracker)

    candidates = strategy.scan_candidates()
    if not candidates:
        print("No tradeable candidates found.")
        return

    print(f"\nFound {len(candidates)} candidates. Analyzing top {args.limit}...")
    analyses = strategy.analyze_candidates(candidates, max_analyze=args.limit)

    print(f"\n{'=' * 70}")
    print("  MARKET ANALYSIS - Powered by Claude Sonnet 4.6")
    print(f"{'=' * 70}")

    for i, a in enumerate(analyses, 1):
        direction = "BUY YES" if a.edge > 0 else "BUY NO"
        edge_bar = "█" * int(abs(a.edge) * 100)

        print(f"\n  #{i} {a.market_title}")
        print(f"  Ticker:      {a.ticker}")
        print(f"  Market:      {a.market_probability:.1%}")
        print(f"  AI Estimate: {a.ai_probability:.1%}")
        print(f"  Edge:        {a.edge:+.1%} {edge_bar}")
        print(f"  Confidence:  {a.confidence:.0%}")
        print(f"  Signal:      {direction}")
        print(f"  Reasoning:   {a.reasoning[:200]}")
        if a.key_factors:
            print(f"  Key Factors: {', '.join(a.key_factors[:3])}")
        if a.risk_factors:
            print(f"  Risks:       {', '.join(a.risk_factors[:3])}")

    usage = analyst.get_usage_stats()
    print(f"\n  API Cost: ~${usage['estimated_cost_usd']:.3f}")


def cmd_trade(args, config: AppConfig):
    """Generate and optionally execute trade signals."""
    tracker = Tracker()

    # Check daily API budget before proceeding
    if not tracker.check_daily_budget(config.schedule.max_daily_api_cost_usd):
        print(f"Daily API budget exceeded (${config.schedule.max_daily_api_cost_usd:.2f}). Skipping.")
        return

    client = KalshiClient(config.kalshi)
    analyst = ClaudeAnalyst(config.claude, tracker=tracker)
    strategy = TradingStrategy(config, client, analyst, tracker=tracker)

    dry_run = not args.live

    if not dry_run:
        print("\n!!  LIVE TRADING MODE  !!")
        print("This will place REAL orders on Kalshi.")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

    results = strategy.run_scan_and_trade(dry_run=dry_run)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def cmd_backtest(args, config: AppConfig):
    """Run a backtest against settled markets."""
    tracker = Tracker()
    client = KalshiClient(config.kalshi)

    # Configure analyst based on flags
    blind_mode = args.blind or config.backtest.hide_market_price
    anonymize = args.anonymize or config.backtest.anonymize_markets
    analyst = ClaudeAnalyst(config.claude, blind_mode=blind_mode, anonymize=anonymize, tracker=tracker)
    backtester = Backtester(config, analyst)

    # Override min_close_date if specified
    if args.min_date:
        config.backtest.min_close_date = args.min_date

    # Load data from file or fetch from API
    if args.data_file:
        print(f"Loading market data from {args.data_file}...")
        markets = client.load_market_data(args.data_file)
    else:
        print("Fetching settled markets from Kalshi API...")
        markets = client.collect_settled_markets(limit=args.limit)

        if args.save_data:
            save_path = f"backtest_data/markets_{datetime.utcnow():%Y%m%d_%H%M}.json"
            client.save_market_data(markets, save_path)
            print(f"Saved {len(markets)} markets to {save_path}")

    settled = [m for m in markets if m.result in ("yes", "no")]
    print(f"\nLoaded {len(markets)} markets ({len(settled)} settled)")

    if not settled:
        print("No settled markets available. Use 'collect' command first.")
        return

    # Print configuration
    print(f"\n  Configuration:")
    print(f"  Initial Balance:     ${config.backtest.initial_balance:,.2f}")
    print(f"  Kelly Fraction:      {config.trading.kelly_fraction}")
    print(f"  Min Edge Threshold:  {config.trading.min_edge_threshold:.1%}")
    print(f"  Analyst Mode:        {'BLIND (no market price)' if blind_mode else 'Standard'}")
    print(f"  Anonymize Markets:   {'YES' if anonymize else 'No'}")
    print(f"  Min Close Date:      {config.backtest.min_close_date}")

    # Progress bar
    def progress(done, total):
        pct = done / total * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.0f}% ({done}/{total})", end="", flush=True)

    # Run contamination comparison or normal backtest
    if args.check_contamination:
        print(f"\n  Running CONTAMINATION CHECK (two backtests)...")
        comparison = backtester.run_contamination_comparison(
            markets, batch_size=args.batch_size
        )
        result = comparison["clean"]
    else:
        print(f"\n  Running backtest...")
        result = backtester.run(
            markets, batch_size=args.batch_size, progress_callback=progress
        )
        print()  # newline after progress bar

    backtester.print_report(result)

    if args.output:
        backtester.save_results(result, args.output)
        print(f"\nDetailed results saved to {args.output}")


def cmd_collect(args, config: AppConfig):
    """Collect market data for offline backtesting."""
    client = KalshiClient(config.kalshi)

    print(f"Collecting up to {args.limit} settled markets...")
    markets = client.collect_settled_markets(limit=args.limit)

    save_path = args.output or f"backtest_data/markets_{datetime.utcnow():%Y%m%d_%H%M}.json"
    client.save_market_data(markets, save_path)

    settled = [m for m in markets if m.result]
    yes_wins = sum(1 for m in settled if m.result == "yes")

    print(f"\nCollected {len(markets)} markets")
    print(f"  Settled: {len(settled)} (YES: {yes_wins}, NO: {len(settled) - yes_wins})")
    print(f"  Saved to: {save_path}")


def cmd_costs(args, config: AppConfig):
    """View daily cost and P&L report."""
    tracker = Tracker()
    tracker.print_daily_report(days=args.days)


def cmd_analyze(args, config: AppConfig):
    """Deep-analyze a single market."""
    tracker = Tracker()
    client = KalshiClient(config.kalshi)
    analyst = ClaudeAnalyst(config.claude, tracker=tracker)

    print(f"Fetching market {args.ticker}...")
    market = client.get_market(args.ticker)

    print(f"\n  Market: {market.title}")
    print(f"  Status: {market.status}")
    print(f"  Price:  {market.yes_mid}c (implies {market.implied_probability:.1%})")
    print(f"  Volume: {market.volume:,}")
    print(f"  Spread: {market.spread}c")

    print("\nAsking Claude for deep analysis...")
    analysis = analyst.analyze_market(market)

    print(f"\n{'=' * 60}")
    print("  CLAUDE'S ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  AI Probability: {analysis.ai_probability:.1%}")
    print(f"  Market Price:   {analysis.market_probability:.1%}")
    print(f"  Edge:           {analysis.edge:+.1%}")
    print(f"  Confidence:     {analysis.confidence:.0%}")

    if analysis.recommended_side:
        print(f"  Signal:         BUY {analysis.recommended_side.value.upper()}")
    else:
        print("  Signal:         No trade (insufficient edge)")

    print(f"\n  Reasoning:\n  {analysis.reasoning}")

    if analysis.key_factors:
        print(f"\n  Key Factors:")
        for f in analysis.key_factors:
            print(f"    - {f}")

    if analysis.risk_factors:
        print(f"\n  Risk Factors:")
        for f in analysis.risk_factors:
            print(f"    - {f}")

    usage = analyst.get_usage_stats()
    print(f"\n  API Cost: ~${usage['estimated_cost_usd']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi AI Trading & Backtesting System - Powered by Claude Sonnet 4.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kalshi_trader scan                          # Scan and analyze markets
  python -m kalshi_trader scan --limit 5                # Analyze top 5 markets
  python -m kalshi_trader analyze TICKER-HERE           # Deep-analyze one market
  python -m kalshi_trader backtest --limit 50           # Backtest 50 settled markets
  python -m kalshi_trader backtest --blind              # Backtest without price anchoring
  python -m kalshi_trader backtest --check-contamination  # Test for data leakage
  python -m kalshi_trader backtest --min-date 2025-06-01  # Only post-June-2025 markets
  python -m kalshi_trader collect --limit 200           # Collect data for backtesting
  python -m kalshi_trader trade                         # Dry-run trade signals
  python -m kalshi_trader trade --live                  # LIVE trading
  python -m kalshi_trader costs                         # Daily cost & P&L report
  python -m kalshi_trader costs --days 30               # Last 30 days
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan markets and show AI analysis")
    p_scan.add_argument("--limit", type=int, default=10, help="Max markets to analyze")

    # trade
    p_trade = subparsers.add_parser("trade", help="Generate trade signals")
    p_trade.add_argument("--live", action="store_true", help="Execute real trades")
    p_trade.add_argument("-o", "--output", help="Save results to JSON file")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Backtest against settled markets")
    p_bt.add_argument("--limit", type=int, default=50, help="Number of markets to fetch")
    p_bt.add_argument("--data-file", help="Load data from file instead of API")
    p_bt.add_argument("--save-data", action="store_true", help="Save fetched data")
    p_bt.add_argument("--batch-size", type=int, default=5, help="Markets per Claude call")
    p_bt.add_argument("-o", "--output", help="Save results to JSON file")
    # Anti-contamination options
    p_bt.add_argument(
        "--blind", action="store_true",
        help="Hide market prices from Claude (anti-anchoring)"
    )
    p_bt.add_argument(
        "--anonymize", action="store_true",
        help="Strip names/dates from market titles (reduce data leakage)"
    )
    p_bt.add_argument(
        "--min-date", type=str, default=None,
        help="Only backtest markets closed after this date (YYYY-MM-DD). "
             "Default: 2025-04-01 (after Claude's training cutoff)"
    )
    p_bt.add_argument(
        "--check-contamination", action="store_true",
        help="Run two backtests (filtered vs unfiltered) to measure contamination"
    )
    p_bt.add_argument(
        "--no-filter", action="store_true",
        help="Disable contamination filter (test all markets)"
    )

    # collect
    p_collect = subparsers.add_parser("collect", help="Collect market data for backtesting")
    p_collect.add_argument("--limit", type=int, default=200, help="Number of markets")
    p_collect.add_argument("-o", "--output", help="Output file path")

    # costs
    p_costs = subparsers.add_parser("costs", help="View daily cost & P&L report")
    p_costs.add_argument("--days", type=int, default=7, help="Number of days to show")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Deep-analyze a single market")
    p_analyze.add_argument("ticker", help="Market ticker to analyze")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)
    config = AppConfig()

    commands = {
        "scan": cmd_scan,
        "trade": cmd_trade,
        "backtest": cmd_backtest,
        "collect": cmd_collect,
        "costs": cmd_costs,
        "analyze": cmd_analyze,
    }

    commands[args.command](args, config)


if __name__ == "__main__":
    main()
