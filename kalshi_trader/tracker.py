"""Persistent cost and trade tracking via SQLite.

Logs every API call, trade, and daily summary so you always know:
- How much you're spending on Claude API per day
- Trading P&L per day, per market, cumulative
- Win rate trends over time
- Whether the system is worth running
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    date TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    call_type TEXT NOT NULL,  -- 'single_analysis', 'batch_analysis', 'other'
    market_tickers TEXT,      -- comma-separated tickers analyzed
    blind_mode INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    market_title TEXT,
    side TEXT NOT NULL,        -- 'yes' or 'no'
    action TEXT NOT NULL,      -- 'buy' or 'sell'
    price_cents REAL NOT NULL,
    contracts INTEGER NOT NULL,
    cost_usd REAL NOT NULL,    -- total cost of the trade
    fees_usd REAL NOT NULL,
    is_live INTEGER NOT NULL DEFAULT 0,  -- 0=dry_run/backtest, 1=live
    mode TEXT NOT NULL,        -- 'live', 'dry_run', 'backtest'
    ai_probability REAL,
    market_probability REAL,
    edge REAL,
    confidence REAL,
    pnl_usd REAL,             -- NULL until settled
    settled_result TEXT        -- 'yes', 'no', or NULL
);

CREATE TABLE IF NOT EXISTS daily_summary (
    date TEXT PRIMARY KEY,
    api_calls INTEGER NOT NULL DEFAULT 0,
    api_cost_usd REAL NOT NULL DEFAULT 0,
    api_input_tokens INTEGER NOT NULL DEFAULT 0,
    api_output_tokens INTEGER NOT NULL DEFAULT 0,
    trades_placed INTEGER NOT NULL DEFAULT 0,
    trades_won INTEGER NOT NULL DEFAULT 0,
    trades_lost INTEGER NOT NULL DEFAULT 0,
    trading_pnl_usd REAL NOT NULL DEFAULT 0,
    trading_fees_usd REAL NOT NULL DEFAULT 0,
    net_pnl_usd REAL NOT NULL DEFAULT 0,  -- trading_pnl - api_cost - fees
    cumulative_pnl_usd REAL NOT NULL DEFAULT 0,
    balance_usd REAL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_calls_date ON api_calls(date);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
"""


class Tracker:
    """Persistent tracker for API costs, trades, and daily P&L."""

    def __init__(self, db_path: str = "kalshi_trader.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("Tracker initialized: %s", db_path)

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- API Cost Tracking ---

    def log_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        call_type: str = "single_analysis",
        market_tickers: Optional[list[str]] = None,
        blind_mode: bool = False,
    ):
        """Log a single Claude API call."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        tickers_str = ",".join(market_tickers) if market_tickers else ""

        self.conn.execute(
            """INSERT INTO api_calls
               (timestamp, date, model, input_tokens, output_tokens, cost_usd,
                call_type, market_tickers, blind_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now.isoformat(), date_str, model, input_tokens, output_tokens,
             cost_usd, call_type, tickers_str, int(blind_mode)),
        )
        self.conn.commit()
        self._update_daily_api(date_str)

    def _update_daily_api(self, date_str: str):
        """Update daily summary with latest API cost totals."""
        row = self.conn.execute(
            """SELECT COUNT(*) as calls, SUM(cost_usd) as cost,
                      SUM(input_tokens) as inp, SUM(output_tokens) as out
               FROM api_calls WHERE date = ?""",
            (date_str,),
        ).fetchone()

        self.conn.execute(
            """INSERT INTO daily_summary (date, api_calls, api_cost_usd,
                   api_input_tokens, api_output_tokens, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                   api_calls = excluded.api_calls,
                   api_cost_usd = excluded.api_cost_usd,
                   api_input_tokens = excluded.api_input_tokens,
                   api_output_tokens = excluded.api_output_tokens,
                   updated_at = excluded.updated_at""",
            (date_str, row["calls"], row["cost"], row["inp"], row["out"],
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    # --- Trade Tracking ---

    def log_trade(
        self,
        ticker: str,
        side: str,
        action: str,
        price_cents: float,
        contracts: int,
        fees_usd: float,
        mode: str = "dry_run",
        market_title: str = "",
        ai_probability: Optional[float] = None,
        market_probability: Optional[float] = None,
        edge: Optional[float] = None,
        confidence: Optional[float] = None,
        pnl_usd: Optional[float] = None,
        settled_result: Optional[str] = None,
    ):
        """Log a trade (live, dry_run, or backtest)."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        cost_usd = (price_cents / 100.0) * contracts
        is_live = 1 if mode == "live" else 0

        self.conn.execute(
            """INSERT INTO trades
               (timestamp, date, ticker, market_title, side, action,
                price_cents, contracts, cost_usd, fees_usd, is_live, mode,
                ai_probability, market_probability, edge, confidence,
                pnl_usd, settled_result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now.isoformat(), date_str, ticker, market_title, side, action,
             price_cents, contracts, cost_usd, fees_usd, is_live, mode,
             ai_probability, market_probability, edge, confidence,
             pnl_usd, settled_result),
        )
        self.conn.commit()
        self._update_daily_trades(date_str)

    def update_trade_result(self, trade_id: int, pnl_usd: float, settled_result: str):
        """Update a trade with its settlement result."""
        self.conn.execute(
            "UPDATE trades SET pnl_usd = ?, settled_result = ? WHERE id = ?",
            (pnl_usd, settled_result, trade_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT date FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        if row:
            self._update_daily_trades(row["date"])

    def _update_daily_trades(self, date_str: str):
        """Update daily summary with latest trade totals."""
        row = self.conn.execute(
            """SELECT
                   COUNT(*) as placed,
                   SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as won,
                   SUM(CASE WHEN pnl_usd IS NOT NULL AND pnl_usd <= 0 THEN 1 ELSE 0 END) as lost,
                   COALESCE(SUM(pnl_usd), 0) as pnl,
                   COALESCE(SUM(fees_usd), 0) as fees
               FROM trades WHERE date = ?""",
            (date_str,),
        ).fetchone()

        # Get api_cost for net calculation
        api_row = self.conn.execute(
            "SELECT COALESCE(api_cost_usd, 0) as api_cost FROM daily_summary WHERE date = ?",
            (date_str,),
        ).fetchone()
        api_cost = api_row["api_cost"] if api_row else 0

        net_pnl = row["pnl"] - api_cost - row["fees"]

        # Get cumulative P&L
        prev_cumulative = self.conn.execute(
            """SELECT COALESCE(cumulative_pnl_usd, 0) as cum
               FROM daily_summary WHERE date < ? ORDER BY date DESC LIMIT 1""",
            (date_str,),
        ).fetchone()
        prev_cum = prev_cumulative["cum"] if prev_cumulative else 0

        self.conn.execute(
            """INSERT INTO daily_summary (date, trades_placed, trades_won,
                   trades_lost, trading_pnl_usd, trading_fees_usd,
                   net_pnl_usd, cumulative_pnl_usd, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                   trades_placed = excluded.trades_placed,
                   trades_won = excluded.trades_won,
                   trades_lost = excluded.trades_lost,
                   trading_pnl_usd = excluded.trading_pnl_usd,
                   trading_fees_usd = excluded.trading_fees_usd,
                   net_pnl_usd = excluded.net_pnl_usd,
                   cumulative_pnl_usd = excluded.cumulative_pnl_usd,
                   updated_at = excluded.updated_at""",
            (date_str, row["placed"], row["won"], row["lost"],
             row["pnl"], row["fees"], net_pnl, prev_cum + net_pnl,
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    # --- Queries / Reports ---

    def get_today_api_cost(self) -> float:
        """Get total API cost for today."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as cost FROM api_calls WHERE date = ?",
            (date_str,),
        ).fetchone()
        return row["cost"]

    def get_daily_summaries(self, limit: int = 30) -> list[dict]:
        """Get recent daily summaries."""
        rows = self.conn.execute(
            """SELECT * FROM daily_summary ORDER BY date DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_time_stats(self) -> dict:
        """Get all-time aggregate statistics."""
        api = self.conn.execute(
            """SELECT COUNT(*) as calls, COALESCE(SUM(cost_usd), 0) as cost,
                      COALESCE(SUM(input_tokens), 0) as inp,
                      COALESCE(SUM(output_tokens), 0) as out
               FROM api_calls"""
        ).fetchone()

        trades = self.conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as won,
                      SUM(CASE WHEN pnl_usd IS NOT NULL AND pnl_usd <= 0 THEN 1 ELSE 0 END) as lost,
                      COALESCE(SUM(pnl_usd), 0) as pnl,
                      COALESCE(SUM(fees_usd), 0) as fees
               FROM trades WHERE mode = 'live'"""
        ).fetchone()

        live_total = trades["total"] or 0
        win_rate = (trades["won"] or 0) / live_total if live_total > 0 else 0

        return {
            "api_calls": api["calls"],
            "api_cost_usd": api["cost"],
            "api_tokens": api["inp"] + api["out"],
            "live_trades": live_total,
            "live_won": trades["won"] or 0,
            "live_lost": trades["lost"] or 0,
            "win_rate": win_rate,
            "trading_pnl_usd": trades["pnl"],
            "trading_fees_usd": trades["fees"],
            "net_pnl_usd": trades["pnl"] - api["cost"] - trades["fees"],
        }

    def get_trades_for_ticker(self, ticker: str) -> list[dict]:
        """Get all trades for a specific market."""
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE ticker = ? ORDER BY timestamp",
            (ticker,),
        ).fetchall()
        return [dict(r) for r in rows]

    def print_daily_report(self, days: int = 7):
        """Print a formatted daily cost/P&L report."""
        summaries = self.get_daily_summaries(limit=days)

        if not summaries:
            print("  No data yet.")
            return

        print(f"\n{'=' * 75}")
        print("  DAILY COST & P&L REPORT")
        print(f"{'=' * 75}")
        print(f"  {'Date':<12} {'API Cost':>9} {'Trades':>7} {'W/L':>7} "
              f"{'Trade P&L':>10} {'Net P&L':>10} {'Cumulative':>11}")
        print(f"  {'-'*12} {'-'*9} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*11}")

        for s in reversed(summaries):
            wl = f"{s['trades_won'] or 0}/{s['trades_lost'] or 0}"
            print(
                f"  {s['date']:<12} "
                f"${s['api_cost_usd']:>7.2f} "
                f"{s['trades_placed']:>7} "
                f"{wl:>7} "
                f"${s['trading_pnl_usd']:>+9.2f} "
                f"${s['net_pnl_usd']:>+9.2f} "
                f"${s['cumulative_pnl_usd']:>+10.2f}"
            )

        print(f"  {'-'*12} {'-'*9} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*11}")

        # Totals
        stats = self.get_all_time_stats()
        print(f"\n  ALL TIME:")
        print(f"  API Cost:    ${stats['api_cost_usd']:,.2f} ({stats['api_calls']} calls, {stats['api_tokens']:,} tokens)")
        print(f"  Live Trades: {stats['live_trades']} (Win Rate: {stats['win_rate']:.1%})")
        print(f"  Trade P&L:   ${stats['trading_pnl_usd']:+,.2f}")
        print(f"  Fees:        ${stats['trading_fees_usd']:,.2f}")
        print(f"  Net P&L:     ${stats['net_pnl_usd']:+,.2f} (after API costs + fees)")
        print(f"{'=' * 75}")

    def check_daily_budget(self, max_cost: float) -> bool:
        """Check if today's API spend is within budget. Returns True if OK."""
        today_cost = self.get_today_api_cost()
        if today_cost >= max_cost:
            logger.warning(
                "Daily API budget exceeded: $%.2f / $%.2f",
                today_cost, max_cost,
            )
            return False
        return True
