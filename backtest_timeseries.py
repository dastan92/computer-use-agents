"""Time-series backtest: simulate trading over months with cumulative P&L.

Uses existing ensemble predictions on 80 settled markets to simulate
a realistic trading timeline — entering positions when signals appear
and settling them when markets resolve.
"""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path("backtest_data")

# --- Use same v4 engine parameters ---
PORTFOLIO_SIZE = 2000
MAX_CLUSTER_PCT = 0.15
MAX_SINGLE_PCT = 0.12
MIN_EDGE = 0.03
MIN_AGREEMENT = 0.75
PROB_WEIGHTS = {"bayesian": 0.75, "contrarian": 0.25}
FILTER_STRATEGIES = ["base_rate", "mean_reversion"]


def load_data():
    """Load settled markets and backtest predictions."""
    with open(DATA_DIR / "settled_markets_v2.json") as f:
        markets = json.load(f)

    with open(DATA_DIR / "ensemble_backtest.json") as f:
        bt = json.load(f)

    return markets, bt


def compute_signal(ticker, strategy_results, market_prob):
    """Compute trading signal using v4 engine logic."""
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

    # Directional agreement across ALL strategies
    direction = 1 if edge > 0 else -1
    agree_count = sum(
        1 for s, p in probs.items()
        if (p - market_prob > 0) == (edge > 0)
    )
    agreement = agree_count / len(probs) if probs else 0

    # Mean reversion agreement
    mr_agrees = False
    if "mean_reversion" in probs:
        mr_edge = probs["mean_reversion"] - market_prob
        mr_agrees = (mr_edge > 0) == (edge > 0)

    # Confidence
    prob_confs = [confs[s] for s in PROB_WEIGHTS if s in confs]
    avg_conf = sum(prob_confs) / len(prob_confs) if prob_confs else 0.5

    if agreement >= 0.75:
        avg_conf *= 1.2
    if mr_agrees:
        avg_conf *= 1.15
    avg_conf = min(avg_conf, 0.95)

    # Trade filter
    if abs(edge) < MIN_EDGE or agreement < MIN_AGREEMENT or avg_conf < 0.50:
        return None

    # Quarter-Kelly sizing
    kelly = abs(edge) * 0.25
    kelly = min(kelly, MAX_SINGLE_PCT)
    position_size = PORTFOLIO_SIZE * kelly

    return {
        "ai_prob": ai_prob,
        "edge": edge,
        "side": "NO" if edge < 0 else "YES",
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

    print(f"{'=' * 100}")
    print(f"  TIME-SERIES BACKTEST: {len(market_data)} markets from "
          f"{market_data[0]['settle_date'].strftime('%b %Y')} to "
          f"{market_data[-1]['settle_date'].strftime('%b %Y')}")
    print(f"  Starting capital: ${PORTFOLIO_SIZE:,.0f}")
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

    # Process each market at its settlement date
    for md in market_data:
        ticker = md["ticker"]

        # Generate signal
        signal = compute_signal(ticker, strategy_results, md["market_prob"])

        if signal is None:
            continue

        # Check position limits
        allocated = sum(p["size"] for p in open_positions)
        if allocated + signal["position_size"] > PORTFOLIO_SIZE * 0.75:
            continue

        # Entry
        side = signal["side"]
        edge = signal["edge"]
        entry_price = md["market_prob"] if side == "YES" else (1 - md["market_prob"])
        size = signal["position_size"]
        contracts = max(1, round(size / entry_price))

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

    # Annualize: figure out time span
    if daily_values:
        first_date = daily_values[0][0]
        last_date = daily_values[-1][0]
        days_span = max((last_date - first_date).days, 1)
        annual_return = total_return * (365 / days_span)
    else:
        days_span = 0
        annual_return = 0

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


if __name__ == "__main__":
    run_timeseries_backtest()
