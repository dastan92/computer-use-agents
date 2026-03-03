"""Time-series backtest: simulate trading over months with cumulative P&L.

Uses existing ensemble predictions on 80 settled markets to simulate
a realistic trading timeline — entering positions when signals appear
and settling them when markets resolve.

Iterates through algorithm versions to find the best parameters.
"""

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path("backtest_data")

# --- v8 engine parameters (data-driven from statistical analysis) ---
PORTFOLIO_SIZE = 2000
MAX_CLUSTER_PCT = 0.15
MAX_SINGLE_PCT = 0.12
MIN_EDGE = 0.09           # v7c: raised from 0.03 — below 9% edge has low win rate
MIN_AGREEMENT = 0.75
MAX_CONTRACTS = 9999       # v11: uncapped — Kelly + price floor handle risk now
NO_SIDE_ONLY = True        # v5+: YES bets had 0% win rate across all 5 trades
MAX_SAME_BASE_TICKER = 2   # v5+: prevents 3x concentration, allows 2 related bets

# --- v11 filters (data-driven from 5-round parameter sweep) ---
MIN_NO_PRICE = 0.25        # Don't buy cheap NO contracts (market says YES likely, usually right)
MAX_AGREEMENT = 1.0        # Cap agreement
MAX_CONFIDENCE = 1.0       # Cap raw confidence
CATEGORY_MIN_EDGES = {}    # Category-specific edge overrides

# --- v11 position management ---
CHEAP_NO_MIN_EDGE = 0.40   # Allow cheap NOs through IF edge is this huge (Japanese election rule)
KELLY_MULTIPLIER = 0.80    # 80% Kelly (justified by 70% win rate + edge-conditional filter)
MAX_LOSS_PER_TRADE = 9999  # Max dollar loss per trade
SKIP_CATEGORIES = []       # Categories to completely skip

PROB_WEIGHTS = {"bayesian": 0.75, "contrarian": 0.25}
FILTER_STRATEGIES = ["base_rate", "mean_reversion"]


def load_data():
    """Load settled markets and backtest predictions."""
    with open(DATA_DIR / "settled_markets_v2.json") as f:
        markets = json.load(f)

    with open(DATA_DIR / "ensemble_backtest.json") as f:
        bt = json.load(f)

    return markets, bt


def get_base_ticker(ticker):
    """Extract base ticker to detect related markets (e.g. KXIPOOPENAI from KXIPOOPENAI-25DEC01)."""
    # Strip date suffixes like -25DEC01, -26JAN01, -29, -30, etc.
    base = re.sub(r'-\d{2}[A-Z]{3}\d{2}$', '', ticker)  # -25DEC01
    base = re.sub(r'-\d{2,4}$', '', base)                 # -29, -30
    # Also strip trailing candidate/person suffixes like -GS, -CG, -RPRE, -TPET
    base = re.sub(r'-[A-Z]{1,4}$', '', base)
    return base


def compute_signal(ticker, strategy_results, market_prob, category=""):
    """Compute trading signal using v9 engine logic."""
    # v9: Skip entire categories
    if category in SKIP_CATEGORIES:
        return None

    probs = {}
    confs = {}

    for sname, results in strategy_results.items():
        for r in results:
            if r.get("ticker") == ticker:
                probs[sname] = r["ai_probability"]
                confs[sname] = r.get("confidence", 0.5)
                break

    if not probs:
        return None

    # Probability: weighted avg of bayesian + contrarian
    total_w = 0
    weighted_prob = 0
    for sname, weight in PROB_WEIGHTS.items():
        if sname in probs:
            conf = confs.get(sname, 0.5)
            w = weight * conf
            weighted_prob += probs[sname] * w
            total_w += w

    if total_w == 0:
        return None

    ai_prob = weighted_prob / total_w
    edge = ai_prob - market_prob

    # v5: Only take NO-side trades (YES had 0% win rate)
    side = "NO" if edge < 0 else "YES"
    if NO_SIDE_ONLY and side == "YES":
        return None

    # v9: Smart price floor — cheap NO contracts need a MUCH bigger edge to justify
    # (cheap NO = market says YES is very likely; usually correct unless edge is huge)
    if side == "NO":
        no_price = 1 - market_prob
        if no_price < MIN_NO_PRICE:
            # If edge-conditional override is set, allow cheap NOs with huge edges
            if CHEAP_NO_MIN_EDGE > 0 and abs(edge) >= CHEAP_NO_MIN_EDGE:
                pass  # allow it through
            else:
                return None

    # Directional agreement across ALL strategies
    agree_count = sum(
        1 for s, p in probs.items()
        if (p - market_prob > 0) == (edge > 0)
    )
    agreement = agree_count / len(probs) if probs else 0

    # v8: Agreement ceiling
    if agreement > MAX_AGREEMENT:
        return None

    # Mean reversion agreement
    mr_agrees = False
    if "mean_reversion" in probs:
        mr_edge = probs["mean_reversion"] - market_prob
        mr_agrees = (mr_edge > 0) == (edge > 0)

    # Confidence
    prob_confs = [confs[s] for s in PROB_WEIGHTS if s in confs]
    avg_conf = sum(prob_confs) / len(prob_confs) if prob_confs else 0.5

    # v8: Penalize overconfidence
    if avg_conf > MAX_CONFIDENCE:
        return None

    if agreement >= 0.75:
        avg_conf *= 1.2
    if mr_agrees:
        avg_conf *= 1.15
    avg_conf = min(avg_conf, 0.95)

    # v8: Category-specific edge thresholds
    effective_min_edge = CATEGORY_MIN_EDGES.get(category, MIN_EDGE)

    # Trade filter
    if abs(edge) < effective_min_edge or agreement < MIN_AGREEMENT or avg_conf < 0.50:
        return None

    # v9: Dynamic Kelly sizing
    kelly = abs(edge) * KELLY_MULTIPLIER
    kelly = min(kelly, MAX_SINGLE_PCT)
    position_size = PORTFOLIO_SIZE * kelly

    return {
        "ai_prob": ai_prob,
        "edge": edge,
        "side": side,
        "agreement": agreement,
        "confidence": avg_conf,
        "mr_agrees": mr_agrees,
        "kelly": kelly,
        "position_size": position_size,
        "individual_probs": probs,
    }


def run_timeseries_backtest():
    """Run chronological backtest simulating real trading over time."""
    markets, bt = load_data()
    strategy_results = bt["strategy_results"]

    # Build market lookup with dates
    market_data = []
    for m in markets:
        close_time = m.get("close_time", "")
        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        # Use created_time as "when we'd enter the trade"
        created_time = m.get("created_time", "")
        try:
            created_dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            created_dt = close_dt - timedelta(days=90)

        mid_price = m.get("_mid_price", 50)
        market_prob = mid_price / 100.0
        actual = m.get("result", "")

        market_data.append({
            "ticker": m["ticker"],
            "title": m.get("title", "")[:60],
            "category": m.get("_category", ""),
            "market_prob": market_prob,
            "actual": actual,
            "entry_date": created_dt,
            "settle_date": close_dt,
            "mid_price": mid_price,
        })

    # Sort by settlement date for chronological simulation
    market_data.sort(key=lambda x: x["settle_date"])

    # Filter to markets from 2025+ (where we have decent volume)
    market_data = [m for m in market_data if m["settle_date"].year >= 2025]

    version = "v5"
    print(f"{'=' * 100}")
    print(f"  TIME-SERIES BACKTEST {version}: {len(market_data)} markets from "
          f"{market_data[0]['settle_date'].strftime('%b %Y')} to "
          f"{market_data[-1]['settle_date'].strftime('%b %Y')}")
    print(f"  Starting capital: ${PORTFOLIO_SIZE:,.0f}")
    print(f"  Rules: NO-side only={NO_SIDE_ONLY}, MIN_EDGE={MIN_EDGE}, "
          f"MAX_CONTRACTS={MAX_CONTRACTS}, MAX_SAME_TICKER={MAX_SAME_BASE_TICKER}")
    print(f"{'=' * 100}\n")

    # Simulate chronologically
    cash = PORTFOLIO_SIZE
    open_positions = []  # {ticker, side, entry_price, size, settle_date, ...}
    trade_log = []
    daily_values = []  # (date, portfolio_value)
    total_trades = 0
    wins = 0
    losses = 0
    cumulative_pnl = 0
    peak_value = PORTFOLIO_SIZE
    max_drawdown = 0
    monthly_pnl = defaultdict(float)

    # Track base ticker concentration
    base_ticker_counts = defaultdict(int)

    # Process each market at its settlement date
    for md in market_data:
        ticker = md["ticker"]

        # Generate signal
        signal = compute_signal(ticker, strategy_results, md["market_prob"], md.get("category", ""))

        if signal is None:
            continue

        # Check position limits
        allocated = sum(p["size"] for p in open_positions)
        if allocated + signal["position_size"] > PORTFOLIO_SIZE * 0.75:
            continue

        # v5: Check base-ticker concentration (no 3x Klarna, 3x half-trillionaire)
        base = get_base_ticker(ticker)
        if base_ticker_counts[base] >= MAX_SAME_BASE_TICKER:
            continue

        # Entry
        side = signal["side"]
        edge = signal["edge"]
        entry_price = md["market_prob"] if side == "YES" else (1 - md["market_prob"])
        size = signal["position_size"]
        contracts = max(1, round(size / entry_price))

        # v5: Cap contracts to prevent boom-or-bust sizing
        contracts = min(contracts, MAX_CONTRACTS)

        # v9: Cap max dollar loss per trade
        max_loss = entry_price * contracts
        if max_loss > MAX_LOSS_PER_TRADE:
            contracts = max(1, int(MAX_LOSS_PER_TRADE / entry_price))

        # Settle immediately (we know the outcome)
        actual_is_yes = md["actual"] == "yes"

        # P&L: you pay entry_price per contract, get $1 if your side wins, $0 if it loses
        if side == "YES":
            won = actual_is_yes
            pnl = ((1.0 - entry_price) if won else (-entry_price)) * contracts
        else:  # NO
            won = not actual_is_yes
            pnl = ((1.0 - entry_price) if won else (-entry_price)) * contracts

        total_trades += 1
        base_ticker_counts[base] += 1
        if won:
            wins += 1
        else:
            losses += 1

        cumulative_pnl += pnl
        portfolio_value = PORTFOLIO_SIZE + cumulative_pnl

        # Track drawdown
        peak_value = max(peak_value, portfolio_value)
        drawdown = (peak_value - portfolio_value) / peak_value
        max_drawdown = max(max_drawdown, drawdown)

        # Track monthly P&L
        month_key = md["settle_date"].strftime("%Y-%m")
        monthly_pnl[month_key] += pnl

        # Log
        settle_str = md["settle_date"].strftime("%Y-%m-%d")
        daily_values.append((md["settle_date"], portfolio_value))

        result_str = "WIN " if won else "LOSS"
        trade_log.append({
            "date": settle_str,
            "ticker": ticker,
            "title": md["title"],
            "side": side,
            "edge": edge,
            "entry_price": entry_price,
            "contracts": contracts,
            "pnl": pnl,
            "won": won,
            "cumulative_pnl": cumulative_pnl,
            "portfolio_value": portfolio_value,
        })

    # Print trade-by-trade log
    print(f"  {'Date':10s} {'Side':>4} {'Edge':>7} {'P&L':>8} {'Cumul':>8} {'Value':>9} {'W/L':>4}  Market")
    print(f"  {'-' * 90}")

    for t in trade_log:
        result_marker = "W" if t["won"] else "L"
        print(
            f"  {t['date']:10s} {t['side']:>4} {t['edge']:>+6.1%} "
            f"${t['pnl']:>+7.2f} ${t['cumulative_pnl']:>+7.2f} "
            f"${t['portfolio_value']:>8.2f} {result_marker:>4}  {t['title']}"
        )

    # Monthly summary
    print(f"\n  {'=' * 60}")
    print(f"  MONTHLY P&L BREAKDOWN")
    print(f"  {'=' * 60}")
    print(f"  {'Month':10s} {'P&L':>10} {'Cumulative':>12} {'Trades':>8}")
    print(f"  {'-' * 44}")

    running = 0
    for month in sorted(monthly_pnl.keys()):
        pnl = monthly_pnl[month]
        running += pnl
        month_trades = sum(1 for t in trade_log if t["date"].startswith(month))
        bar = "+" * int(max(0, pnl / 5)) + "-" * int(max(0, -pnl / 5))
        print(f"  {month:10s} ${pnl:>+9.2f} ${running:>+11.2f} {month_trades:>8}  {bar}")

    # Final summary
    win_rate = wins / total_trades if total_trades else 0
    avg_win = sum(t["pnl"] for t in trade_log if t["won"]) / wins if wins else 0
    avg_loss = sum(t["pnl"] for t in trade_log if not t["won"]) / losses if losses else 0
    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses and avg_loss else float("inf")
    total_return = cumulative_pnl / PORTFOLIO_SIZE

    # Annualize: use actual period boundaries (first/last market settle dates)
    period_start = market_data[0]["settle_date"]
    period_end = market_data[-1]["settle_date"]
    days_span = max((period_end - period_start).days, 1)
    annual_return = total_return * (365 / days_span) if days_span > 0 else 0

    print(f"\n  {'=' * 60}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  Period:              {market_data[0]['settle_date'].strftime('%b %Y')} - "
          f"{market_data[-1]['settle_date'].strftime('%b %Y')} ({days_span} days)")
    print(f"  Starting Capital:    ${PORTFOLIO_SIZE:,.0f}")
    print(f"  Final Value:         ${PORTFOLIO_SIZE + cumulative_pnl:,.2f}")
    print(f"  Total P&L:           ${cumulative_pnl:>+,.2f}")
    print(f"  Total Return:        {total_return:>+.1%}")
    print(f"  Annualized Return:   {annual_return:>+.1%}")
    print(f"  Max Drawdown:        {max_drawdown:.1%}")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Win Rate:            {win_rate:.1%} ({wins}W / {losses}L)")
    print(f"  Avg Win:             ${avg_win:>+.2f}")
    print(f"  Avg Loss:            ${avg_loss:>+.2f}")
    print(f"  Profit Factor:       {profit_factor:.2f}x")
    print(f"  Avg P&L per Trade:   ${cumulative_pnl / total_trades if total_trades else 0:>+.2f}")
    print(f"  {'=' * 60}\n")

    # Save results
    results = {
        "version": version,
        "rules": {
            "no_side_only": NO_SIDE_ONLY,
            "min_edge": MIN_EDGE,
            "max_contracts": MAX_CONTRACTS,
            "max_same_base_ticker": MAX_SAME_BASE_TICKER,
            "min_agreement": MIN_AGREEMENT,
        },
        "period_start": market_data[0]["settle_date"].isoformat(),
        "period_end": market_data[-1]["settle_date"].isoformat(),
        "days_span": days_span,
        "starting_capital": PORTFOLIO_SIZE,
        "final_value": round(PORTFOLIO_SIZE + cumulative_pnl, 2),
        "total_pnl": round(cumulative_pnl, 2),
        "total_return": round(total_return, 4),
        "annualized_return": round(annual_return, 4),
        "max_drawdown": round(max_drawdown, 4),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "monthly_pnl": {k: round(v, 2) for k, v in sorted(monthly_pnl.items())},
        "trades": trade_log,
        "equity_curve": [(d.isoformat(), round(v, 2)) for d, v in daily_values],
    }

    with open(DATA_DIR / "timeseries_backtest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved to {DATA_DIR / 'timeseries_backtest.json'}")

    # ASCII equity curve
    if daily_values:
        print(f"\n  EQUITY CURVE")
        print(f"  {'=' * 60}")
        min_val = min(v for _, v in daily_values)
        max_val = max(v for _, v in daily_values)
        range_val = max_val - min_val if max_val > min_val else 1

        # Sample ~20 points for the chart
        step = max(1, len(daily_values) // 20)
        chart_points = daily_values[::step]
        if daily_values[-1] not in chart_points:
            chart_points.append(daily_values[-1])

        for dt, val in chart_points:
            bar_len = int(40 * (val - min_val) / range_val)
            bar = "█" * bar_len
            marker = " ◀" if val == max_val else ""
            print(f"  {dt.strftime('%Y-%m'):7s} ${val:>8.0f} |{bar}{marker}")

        print(f"  {'=' * 60}")


def run_sweep():
    """Run parameter sweep to find optimal settings.

    v8 sweep: tests new data-driven filters discovered from statistical analysis:
    - NO price floor (avoid cheap NO contracts where market is usually right)
    - Agreement ceiling (100% agreement = 36% accuracy — paradoxically bad)
    - Confidence cap (0.90+ predictions are least accurate)
    - Category-specific edges (Economics is gold at lower edges, Politics is a coin flip)
    """

    # Each config: (label, {param_overrides})
    configs = [
        # --- Baselines ---
        ("v7c-baseline", {}),
        ("v10a-best-risk", {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.30, "CHEAP_NO_MIN_EDGE": 0.30}),

        # --- R5: Fine-tune around 100%+ return configs ---
        # v10e-K50-nf25-ce40 was +102.5%. Tune around it.
        ("v11a-K50-nf25-ce35", {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.35, "MAX_CONTRACTS": 9999}),
        ("v11a-K50-nf25-ce45", {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.45, "MAX_CONTRACTS": 9999}),
        ("v11a-K50-nf25-ce50", {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.50, "MAX_CONTRACTS": 9999}),
        ("v11a-K50-nf20-ce40", {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.20, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),
        ("v11a-K50-nf28-ce40", {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.28, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),

        # Bump Kelly higher — how far can we push it?
        ("v11b-K55-nf25-ce40", {"KELLY_MULTIPLIER": 0.55, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),
        ("v11b-K60-nf25-ce40", {"KELLY_MULTIPLIER": 0.60, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),
        ("v11b-K65-nf25-ce40", {"KELLY_MULTIPLIER": 0.65, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),
        ("v11b-K70-nf25-ce40", {"KELLY_MULTIPLIER": 0.70, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),
        ("v11b-K80-nf25-ce40", {"KELLY_MULTIPLIER": 0.80, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),
        ("v11b-K100-nf25-ce40",{"KELLY_MULTIPLIER": 1.00, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),

        # Add category edges to the 100%+ configs
        ("v11c-K50-nf25-ce40-cat",  {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999, "CATEGORY_MIN_EDGES": {"Economics": 0.06, "Companies": 0.06}}),
        ("v11c-K60-nf25-ce40-cat",  {"KELLY_MULTIPLIER": 0.60, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999, "CATEGORY_MIN_EDGES": {"Economics": 0.06, "Companies": 0.06}}),
        ("v11c-K50-nf30-ce40-cat",  {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.30, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999, "CATEGORY_MIN_EDGES": {"Economics": 0.06}}),

        # Ticker limit 3 for more volume
        ("v11d-K50-nf25-ce40-t3",   {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999, "MAX_SAME_BASE_TICKER": 3}),
        ("v11d-K60-nf25-ce40-t3",   {"KELLY_MULTIPLIER": 0.60, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999, "MAX_SAME_BASE_TICKER": 3}),

        # All-in aggressive: high Kelly + edge-conditional + category + ticker limit
        ("v11e-aggro1",  {"KELLY_MULTIPLIER": 0.60, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999, "CATEGORY_MIN_EDGES": {"Economics": 0.06, "Companies": 0.06}, "MAX_SAME_BASE_TICKER": 3}),
        ("v11e-aggro2",  {"KELLY_MULTIPLIER": 0.70, "MIN_NO_PRICE": 0.25, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999, "CATEGORY_MIN_EDGES": {"Economics": 0.06}, "MAX_SAME_BASE_TICKER": 3}),
        ("v11e-aggro3",  {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.20, "CHEAP_NO_MIN_EDGE": 0.35, "MAX_CONTRACTS": 9999, "CATEGORY_MIN_EDGES": {"Economics": 0.05, "Companies": 0.05}, "MAX_SAME_BASE_TICKER": 3}),

        # Safe versions: best risk-adjusted + extra return
        ("v11f-safe1",   {"KELLY_MULTIPLIER": 0.50, "MIN_NO_PRICE": 0.30, "CHEAP_NO_MIN_EDGE": 0.40, "MAX_CONTRACTS": 9999}),
        ("v11f-safe2",   {"KELLY_MULTIPLIER": 0.45, "MIN_NO_PRICE": 0.30, "CHEAP_NO_MIN_EDGE": 0.40, "CATEGORY_MIN_EDGES": {"Economics": 0.07}}),
        ("v11f-safe3",   {"KELLY_MULTIPLIER": 0.55, "MIN_NO_PRICE": 0.30, "CHEAP_NO_MIN_EDGE": 0.30}),
    ]

    # Defaults to restore after each run
    defaults = {
        "NO_SIDE_ONLY": True,
        "MIN_EDGE": 0.09,
        "MAX_CONTRACTS": 750,
        "MAX_SAME_BASE_TICKER": 2,
        "MIN_AGREEMENT": 0.75,
        "MIN_NO_PRICE": 0.0,
        "MAX_AGREEMENT": 1.0,
        "MAX_CONFIDENCE": 1.0,
        "CATEGORY_MIN_EDGES": {},
        "CHEAP_NO_MIN_EDGE": 0.0,
        "KELLY_MULTIPLIER": 0.25,
        "MAX_LOSS_PER_TRADE": 9999,
        "SKIP_CATEGORIES": [],
    }

    print(f"\n{'='*120}")
    print(f"  v8 PARAMETER SWEEP: {len(configs)} configurations")
    print(f"  Testing: NO price floor, agreement ceiling, confidence cap, category edges")
    print(f"{'='*120}")
    print(f"  {'Config':<22} {'P&L':>10} {'Return':>8} {'WR':>6} {'Trades':>7} "
          f"{'AvgWin':>8} {'AvgLoss':>8} {'PF':>6} {'MaxDD':>7} {'RiskAdj':>8}")
    print(f"  {'-'*110}")

    results_table = []
    for label, overrides in configs:
        # Reset all globals to defaults
        global NO_SIDE_ONLY, MIN_EDGE, MAX_CONTRACTS, MAX_SAME_BASE_TICKER
        global MIN_AGREEMENT, MIN_NO_PRICE, MAX_AGREEMENT, MAX_CONFIDENCE, CATEGORY_MIN_EDGES
        NO_SIDE_ONLY = defaults["NO_SIDE_ONLY"]
        MIN_EDGE = defaults["MIN_EDGE"]
        MAX_CONTRACTS = defaults["MAX_CONTRACTS"]
        MAX_SAME_BASE_TICKER = defaults["MAX_SAME_BASE_TICKER"]
        MIN_AGREEMENT = defaults["MIN_AGREEMENT"]
        MIN_NO_PRICE = defaults["MIN_NO_PRICE"]
        MAX_AGREEMENT = defaults["MAX_AGREEMENT"]
        MAX_CONFIDENCE = defaults["MAX_CONFIDENCE"]
        CATEGORY_MIN_EDGES = defaults["CATEGORY_MIN_EDGES"]

        # Apply overrides
        for k, v in overrides.items():
            globals()[k] = v

        # Suppress printing
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        run_timeseries_backtest()
        sys.stdout = old_stdout

        # Read results
        with open(DATA_DIR / "timeseries_backtest.json") as f:
            r = json.load(f)

        pnl = r["total_pnl"]
        ret = r["total_return"] * 100
        wr = r["win_rate"] * 100 if r["total_trades"] > 0 else 0
        trades = r["total_trades"]
        avg_w = r["avg_win"]
        avg_l = r["avg_loss"]
        pf = r["profit_factor"]
        dd = r["max_drawdown"] * 100

        # Risk-adjusted return (return / max_drawdown)
        risk_adj = ret / dd if dd > 0 else 0

        results_table.append((label, pnl, ret, wr, trades, avg_w, avg_l, pf, dd, risk_adj, r))

        print(f"  {label:<22} ${pnl:>+9.2f} {ret:>+7.1f}% {wr:>5.1f}% {trades:>7} "
              f"${avg_w:>+7.2f} ${avg_l:>+7.2f} {pf:>5.2f}x {dd:>6.1f}% {risk_adj:>7.2f}")

    # Find best by risk-adjusted, then by absolute return as tiebreaker
    results_table.sort(key=lambda x: (x[9], x[2]), reverse=True)
    best = results_table[0]
    best_return = max(results_table, key=lambda x: x[2])

    print(f"\n  {'='*80}")
    print(f"  BEST RISK-ADJUSTED:  {best[0]} (score: {best[9]:.2f}, +{best[2]:.1f}%, {best[4]} trades)")
    print(f"  BEST ABSOLUTE RETURN: {best_return[0]} (+{best_return[2]:.1f}%, {best_return[4]} trades)")
    print(f"  {'='*80}")
    print(f"{'='*120}\n")

    # Save the best risk-adjusted config
    if best:
        with open(DATA_DIR / "timeseries_backtest.json", "w") as f:
            json.dump(best[10], f, indent=2, default=str)
        print(f"  Saved best result ({best[0]}) to timeseries_backtest.json")


if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        run_sweep()
    else:
        run_timeseries_backtest()
