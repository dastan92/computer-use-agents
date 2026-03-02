"""Backtest the ensemble alpha engine against settled markets.

Compares single-strategy performance vs ensemble to validate
that combining strategies improves accuracy and trade P&L.
"""

import json
import logging
import os
import statistics
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = os.getenv("CLAUDE_MODEL", "claude-4-sonnet-20250514")
CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

DATA_DIR = Path("backtest_data")

# Same prompts as alpha_engine.py
STRATEGIES = {
    "bayesian": {
        "weight": 0.40,
        "prompt": """\
You are an elite prediction market analyst. For each market, estimate P(YES).

USE THE MARKET PRICE as a Bayesian prior. The market aggregates many informed
participants. Only disagree when you have SPECIFIC information the market may be
mispricing. Your estimate should be CLOSE to the market price (within ±10%)
unless you have strong evidence.

RULES:
- Multi-choice: base rate = 1/N candidates, use market for adjustment
- Deadlines: most things don't happen on schedule, discount accordingly
- Binary: status quo usually persists
- Calibration: Would about X out of 100 similar situations resolve YES?

JSON array only: [{"ticker": "...", "ai_probability": 0.X, "confidence": 0.X, "reasoning": "..."}]
"""
    },
    "contrarian": {
        "weight": 0.20,
        "prompt": """\
You are a contrarian analyst searching for MISPRICINGS in prediction markets.

COGNITIVE BIASES TO EXPLOIT:
- Recency bias: market overweighting recent events
- Narrative bias: compelling story ≠ high probability
- Anchoring: market stuck at old price despite new info
- Herding: consensus without independent analysis
- Availability bias: vivid events overweighted vs base rates

DISCIPLINE: Only disagree with SPECIFIC, ARTICULABLE reasons. If you can't
name the exact bias, stay close to market price. Random disagreement is noise.

JSON array only: [{"ticker": "...", "ai_probability": 0.X, "confidence": 0.X, "reasoning": "..."}]
"""
    },
    "base_rate": {
        "weight": 0.25,
        "prompt": """\
You are a reference-class forecaster. For each market, estimate P(YES)
using OUTSIDE VIEW thinking.

METHOD:
1. Identify the reference class (what type of event is this?)
2. Find the base rate for that class
3. Adjust MODESTLY for specific factors (max ±20% from base rate)
4. Give the market price 40% weight in your final answer

Never let narrative evidence move you more than 20% from the base rate.

JSON array only: [{"ticker": "...", "ai_probability": 0.X, "confidence": 0.X, "reasoning": "..."}]
"""
    },
    "mean_reversion": {
        "weight": 0.15,
        "prompt": """\
You are a quantitative analyst focused on EXTREME PRICES in prediction markets.

YOUR JOB:
- For markets priced BELOW 20%: Is the market too pessimistic?
- For markets priced ABOVE 80%: Is the market too optimistic?
- For markets in the middle (20-80%): Stay very close to market price.

Be conservative with middle-priced markets. Your alpha is at the extremes.

JSON array only: [{"ticker": "...", "ai_probability": 0.X, "confidence": 0.X, "reasoning": "..."}]
"""
    },
}


def run_strategy(markets, system_prompt, name):
    """Run one strategy on settled markets."""
    results = []
    batch_size = 8

    for i in range(0, len(markets), batch_size):
        batch = markets[i:i + batch_size]
        lines = []
        for j, m in enumerate(batch, 1):
            mid = m["_mid_price"]
            lines.append(
                f'{j}. Ticker: {m["ticker"]}\n'
                f'   Question: {m["title"]}\n'
                f'   Market Price: {mid}c ({mid/100:.0%})\n'
                f'   Category: {m.get("_category", "")}'
            )

        prompt = (
            "Analyze these prediction markets. For EACH, estimate P(YES).\n\n"
            "IMPORTANT: These are real markets. Estimate the TRUE probability,\n"
            "not what happened. Do NOT try to guess outcomes.\n\n"
            "Markets:\n" + "\n".join(lines) + "\n\nJSON array only."
        )

        try:
            response = CLIENT.messages.create(
                model=MODEL, max_tokens=4096, temperature=0.2,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            parsed = json.loads(text.strip())
            for r in parsed:
                r["_strategy"] = name
            results.extend(parsed)
            logger.info("[%s] batch %d: %d results", name, i // batch_size, len(parsed))

        except Exception as e:
            logger.error("[%s] batch %d failed: %s", name, i // batch_size, e)

        time.sleep(1)

    return results


def evaluate(predictions, markets, label):
    """Evaluate predictions against actual outcomes."""
    market_map = {m["ticker"]: m for m in markets}
    correct = 0
    total = 0
    brier_scores = []
    trade_wins = 0
    trade_total = 0
    trade_pnl = 0.0

    for p in predictions:
        ticker = p.get("ticker", "")
        if ticker not in market_map:
            continue

        m = market_map[ticker]
        actual = 1.0 if m["result"] == "yes" else 0.0
        ai_prob = p.get("ai_probability", 0.5)
        market_prob = m["_mid_price"] / 100.0

        # Accuracy: did AI predict the right side?
        ai_side = 1 if ai_prob >= 0.5 else 0
        if ai_side == actual:
            correct += 1
        total += 1

        # Brier score
        brier = (ai_prob - actual) ** 2
        brier_scores.append(brier)

        # Trade simulation: buy when edge > 3%
        edge = ai_prob - market_prob
        if abs(edge) >= 0.03:
            trade_total += 1
            if edge > 0:  # buy YES
                pnl = (actual - market_prob) * 10  # $10 per trade
                if actual == 1.0:
                    trade_wins += 1
            else:  # buy NO
                pnl = (market_prob - actual) * 10
                if actual == 0.0:
                    trade_wins += 1
            trade_pnl += pnl

    accuracy = correct / total if total else 0
    avg_brier = statistics.mean(brier_scores) if brier_scores else 1.0
    market_brier = statistics.mean(
        (m["_mid_price"] / 100.0 - (1.0 if m["result"] == "yes" else 0.0)) ** 2
        for m in markets if m["ticker"] in {p.get("ticker") for p in predictions}
    )
    trade_wr = trade_wins / trade_total if trade_total else 0

    return {
        "label": label,
        "n": total,
        "accuracy": round(accuracy, 3),
        "brier": round(avg_brier, 4),
        "market_brier": round(market_brier, 4),
        "brier_edge": round(market_brier - avg_brier, 4),
        "trades": trade_total,
        "trade_wr": round(trade_wr, 3),
        "trade_pnl": round(trade_pnl, 2),
    }


def combine_ensemble(strategy_results, markets):
    """Combine strategies into ensemble predictions."""
    combined = {}

    for sname, results in strategy_results.items():
        weight = STRATEGIES[sname]["weight"]
        for r in results:
            ticker = r.get("ticker", "")
            if ticker not in combined:
                combined[ticker] = {"probs": [], "weights": [], "confs": []}
            try:
                prob = float(r["ai_probability"])
                conf = float(r.get("confidence", 0.5))
                combined[ticker]["probs"].append(prob)
                combined[ticker]["weights"].append(weight * conf)
                combined[ticker]["confs"].append(conf)
            except (ValueError, KeyError):
                pass

    predictions = []
    for ticker, data in combined.items():
        if not data["probs"]:
            continue
        total_w = sum(data["weights"])
        if total_w == 0:
            continue
        ai_prob = sum(p * w for p, w in zip(data["probs"], data["weights"])) / total_w
        predictions.append({
            "ticker": ticker,
            "ai_probability": round(ai_prob, 4),
            "confidence": round(statistics.mean(data["confs"]), 3),
            "n_strategies": len(data["probs"]),
        })

    return predictions


def main():
    # Load settled markets
    with open(DATA_DIR / "settled_markets_v2.json") as f:
        markets = json.load(f)

    logger.info("Loaded %d settled markets", len(markets))

    # Run each strategy independently
    strategy_results = {}
    for sname, config in STRATEGIES.items():
        logger.info("=== Running strategy: %s ===", sname)
        strategy_results[sname] = run_strategy(markets, config["prompt"], sname)
        logger.info("Got %d predictions from %s", len(strategy_results[sname]), sname)

    # Evaluate each strategy independently
    print(f"\n{'='*80}")
    print(f"  ENSEMBLE BACKTEST RESULTS (80 settled markets)")
    print(f"{'='*80}\n")

    all_evals = []

    # Single strategies
    print("  INDIVIDUAL STRATEGIES:")
    print(f"  {'Strategy':<20} {'Acc':>6} {'Brier':>7} {'MktBrier':>8} {'Edge':>7} {'Trades':>7} {'WR':>6} {'P&L':>8}")
    print(f"  {'-'*72}")

    for sname, results in strategy_results.items():
        # Convert to prediction format
        preds = [{
            "ticker": r.get("ticker", ""),
            "ai_probability": float(r.get("ai_probability", 0.5)),
        } for r in results]

        ev = evaluate(preds, markets, sname)
        all_evals.append(ev)
        print(f"  {sname:<20} {ev['accuracy']:>5.0%} {ev['brier']:>7.4f} {ev['market_brier']:>8.4f} "
              f"{ev['brier_edge']:>+7.4f} {ev['trades']:>7} {ev['trade_wr']:>5.0%} ${ev['trade_pnl']:>7.2f}")

    # Ensemble combinations
    print(f"\n  ENSEMBLE COMBINATIONS:")
    print(f"  {'Strategy':<20} {'Acc':>6} {'Brier':>7} {'MktBrier':>8} {'Edge':>7} {'Trades':>7} {'WR':>6} {'P&L':>8}")
    print(f"  {'-'*72}")

    # Full ensemble (all 4)
    ensemble_preds = combine_ensemble(strategy_results, markets)
    ev = evaluate(ensemble_preds, markets, "ensemble_4")
    all_evals.append(ev)
    print(f"  {'ensemble_all4':<20} {ev['accuracy']:>5.0%} {ev['brier']:>7.4f} {ev['market_brier']:>8.4f} "
          f"{ev['brier_edge']:>+7.4f} {ev['trades']:>7} {ev['trade_wr']:>5.0%} ${ev['trade_pnl']:>7.2f}")

    # Top 3 (without mean_reversion)
    top3 = {k: v for k, v in strategy_results.items() if k != "mean_reversion"}
    t3_preds = combine_ensemble(top3, markets)
    ev = evaluate(t3_preds, markets, "top3_no_mr")
    all_evals.append(ev)
    print(f"  {'top3_no_meanrev':<20} {ev['accuracy']:>5.0%} {ev['brier']:>7.4f} {ev['market_brier']:>8.4f} "
          f"{ev['brier_edge']:>+7.4f} {ev['trades']:>7} {ev['trade_wr']:>5.0%} ${ev['trade_pnl']:>7.2f}")

    # Bayesian + base_rate only
    bb = {k: v for k, v in strategy_results.items() if k in ("bayesian", "base_rate")}
    bb_preds = combine_ensemble(bb, markets)
    ev = evaluate(bb_preds, markets, "bayes_baserate")
    all_evals.append(ev)
    print(f"  {'bayes+baserate':<20} {ev['accuracy']:>5.0%} {ev['brier']:>7.4f} {ev['market_brier']:>8.4f} "
          f"{ev['brier_edge']:>+7.4f} {ev['trades']:>7} {ev['trade_wr']:>5.0%} ${ev['trade_pnl']:>7.2f}")

    # Bayesian + contrarian
    bc = {k: v for k, v in strategy_results.items() if k in ("bayesian", "contrarian")}
    bc_preds = combine_ensemble(bc, markets)
    ev = evaluate(bc_preds, markets, "bayes_contrarian")
    all_evals.append(ev)
    print(f"  {'bayes+contrarian':<20} {ev['accuracy']:>5.0%} {ev['brier']:>7.4f} {ev['market_brier']:>8.4f} "
          f"{ev['brier_edge']:>+7.4f} {ev['trades']:>7} {ev['trade_wr']:>5.0%} ${ev['trade_pnl']:>7.2f}")

    print(f"\n{'='*80}\n")

    # Save results
    with open(DATA_DIR / "ensemble_backtest.json", "w") as f:
        json.dump({
            "evaluations": all_evals,
            "strategy_results": {
                sname: [{
                    "ticker": r.get("ticker"),
                    "ai_probability": r.get("ai_probability"),
                    "confidence": r.get("confidence"),
                } for r in results]
                for sname, results in strategy_results.items()
            },
            "ensemble_predictions": [{
                "ticker": p["ticker"],
                "ai_probability": p["ai_probability"],
            } for p in ensemble_preds],
        }, f, indent=2)

    logger.info("Results saved to %s", DATA_DIR / "ensemble_backtest.json")


if __name__ == "__main__":
    main()
