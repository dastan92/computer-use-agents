#!/usr/bin/env python3
"""Kalshi AI Trading & Backtesting System - CLI Entry Point.

Usage:
    python -m kalshi_trader scan          # Scan markets and show AI analysis
    python -m kalshi_trader trade         # Dry-run trade signals
    python -m kalshi_trader trade --live  # Live trading (use with caution!)
    python -m kalshi_trader backtest      # Backtest against settled markets
    python -m kalshi_trader collect       # Collect market data for backtesting
    python -m kalshi_trader analyze TICKER  # Deep-analyze a single market
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


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_scan(args, config: AppConfig):
    """Scan markets and show Claude's analysis."""
    client = KalshiClient(config.kalshi)
    analyst = ClaudeAnalyst(config.claude)
    strategy = TradingStrategy(config, client, analyst)

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
    client = KalshiClient(config.kalshi)
    analyst = ClaudeAnalyst(config.claude)
    strategy = TradingStrategy(config, client, analyst)

    dry_run = not args.live

    if not dry_run:
        print("\n⚠  LIVE TRADING MODE ⚠")
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
    client = KalshiClient(config.kalshi)
    analyst = ClaudeAnalyst(config.claude)
    backtester = Backtester(config, analyst)

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

    # Run backtest with progress
    def progress(done, total):
        pct = done / total * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.0f}% ({done}/{total})", end="", flush=True)

    print(f"\nRunning backtest on {len(settled)} markets...")
    print(f"  Initial Balance: ${config.backtest.initial_balance:,.2f}")
    print(f"  Kelly Fraction:  {config.trading.kelly_fraction}")
    print(f"  Min Edge:        {config.trading.min_edge_threshold:.1%}")
    print()

    result = backtester.run(settled, batch_size=args.batch_size, progress_callback=progress)
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


def cmd_analyze(args, config: AppConfig):
    """Deep-analyze a single market."""
    client = KalshiClient(config.kalshi)
    analyst = ClaudeAnalyst(config.claude)

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
  python -m kalshi_trader scan                    # Scan and analyze markets
  python -m kalshi_trader scan --limit 5          # Analyze top 5 markets
  python -m kalshi_trader analyze TICKER-HERE     # Deep-analyze one market
  python -m kalshi_trader backtest --limit 50     # Backtest on 50 settled markets
  python -m kalshi_trader backtest --data-file backtest_data/markets.json
  python -m kalshi_trader collect --limit 200     # Collect data for backtesting
  python -m kalshi_trader trade                   # Dry-run trade signals
  python -m kalshi_trader trade --live            # LIVE trading
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
    p_bt.add_argument("--limit", type=int, default=50, help="Number of markets")
    p_bt.add_argument("--data-file", help="Load data from file instead of API")
    p_bt.add_argument("--save-data", action="store_true", help="Save fetched data")
    p_bt.add_argument("--batch-size", type=int, default=5, help="Batch size for analysis")
    p_bt.add_argument("-o", "--output", help="Save results to JSON file")

    # collect
    p_collect = subparsers.add_parser("collect", help="Collect market data for backtesting")
    p_collect.add_argument("--limit", type=int, default=200, help="Number of markets")
    p_collect.add_argument("-o", "--output", help="Output file path")

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
        "analyze": cmd_analyze,
    }

    commands[args.command](args, config)


if __name__ == "__main__":
    main()
