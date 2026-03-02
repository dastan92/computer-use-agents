"""Long-running service runner for Railway / server deployment.

Runs the scan-analyze-trade cycle on a configurable schedule.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone

from .config import AppConfig
from .analyst import ClaudeAnalyst
from .kalshi_client import KalshiClient
from .strategy import TradingStrategy
from .tracker import Tracker

logger = logging.getLogger(__name__)

SCHEDULE_INTERVALS = {
    "continuous": 5 * 60,       # 5 minutes
    "hourly": 60 * 60,          # 1 hour
    "daily": 24 * 60 * 60,      # 24 hours
}

shutdown = False


def handle_signal(signum, frame):
    global shutdown
    logger.info("Received signal %s, shutting down gracefully...", signum)
    shutdown = True


def in_trading_hours(config: AppConfig) -> bool:
    now = datetime.now(timezone.utc)
    return config.schedule.trading_start_hour <= now.hour < config.schedule.trading_end_hour


def run_cycle(config: AppConfig, tracker: Tracker, dry_run: bool):
    """Run one scan-analyze-trade cycle."""
    logger.info("Starting trading cycle at %s", datetime.now(timezone.utc).isoformat())

    if not in_trading_hours(config):
        logger.info("Outside trading hours (%d-%d UTC), skipping.",
                     config.schedule.trading_start_hour, config.schedule.trading_end_hour)
        return

    if not tracker.check_daily_budget(config.schedule.max_daily_api_cost_usd):
        logger.warning("Daily API budget ($%.2f) exceeded, skipping.",
                       config.schedule.max_daily_api_cost_usd)
        return

    try:
        client = KalshiClient(config.kalshi)
        analyst = ClaudeAnalyst(config.claude, tracker=tracker)
        strategy = TradingStrategy(config, client, analyst, tracker=tracker)

        results = strategy.run_scan_and_trade(dry_run=dry_run)
        logger.info("Cycle complete: %d trades executed", len(results))
    except Exception:
        logger.exception("Error during trading cycle")


def main():
    global shutdown

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    config = AppConfig()
    tracker = Tracker()

    mode = config.schedule.mode
    interval = SCHEDULE_INTERVALS.get(mode, SCHEDULE_INTERVALS["daily"])
    dry_run = not config.kalshi.api_key_id  # dry-run if no Kalshi key

    logger.info("Kalshi AI Trader starting")
    logger.info("  Schedule: %s (every %d seconds)", mode, interval)
    logger.info("  Dry run: %s", dry_run)
    logger.info("  Categories: %s", config.schedule.preferred_categories)
    logger.info("  Trading hours: %d-%d UTC", config.schedule.trading_start_hour,
                config.schedule.trading_end_hour)
    logger.info("  Daily API budget: $%.2f", config.schedule.max_daily_api_cost_usd)

    if mode == "manual":
        logger.info("Manual mode — running single cycle and exiting.")
        run_cycle(config, tracker, dry_run)
        return

    while not shutdown:
        run_cycle(config, tracker, dry_run)

        logger.info("Next cycle in %d seconds...", interval)
        # Sleep in small increments so we can catch shutdown signals
        elapsed = 0
        while elapsed < interval and not shutdown:
            time.sleep(min(10, interval - elapsed))
            elapsed += 10

    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
