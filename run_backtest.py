#!/usr/bin/env python3
"""Standalone backtest runner for rapid iteration.

Usage: python3 run_backtest.py [--sample N] [--blind] [--iteration N]
"""

import json
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic

# --- Config ---
MODEL = "claude-4-sonnet-20250514"
API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATA_FILE = "backtest_data/settled_markets_v2.json"
RESULTS_DIR = "backtest_data/results"
BATCH_SIZE = 5  # markets per Claude call
TEMPERATURE = 0.2

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(api_key=API_KEY)

# --- Prompt Templates (will be iterated on) ---

SYSTEM_PROMPT = """\
You are an expert prediction market analyst and superforecaster.
Your job is to estimate the TRUE probability of events resolving YES.

## Forecasting Method (follow this exact sequence)

1. **Parse the question**: What EXACTLY does this contract ask? What are the resolution criteria?

2. **Base rate**: What is the historical base rate for this TYPE of event?
   Be specific. Don't just guess — think about reference classes.

3. **Update from specifics**: What current information shifts the probability
   from the base rate? List each factor with direction (+/-) and magnitude.

4. **Steelman the opposite**: What would need to be true for the OTHER outcome?
   How plausible is that?

5. **Final calibrated estimate**: Combine into a single probability (0.0-1.0).

## Calibration Rules

- If uncertain, stay CLOSER TO 50%.
- Extreme probabilities (>85% or <15%) require overwhelming evidence.
- Your edge comes from BETTER REASONING, not from being more extreme.
- Think about what you DON'T know as much as what you DO know.

## Output Format

Respond with ONLY a valid JSON array. No markdown code blocks, no extra text.
"""

# Version for each iteration
SYSTEM_PROMPTS = {
    1: SYSTEM_PROMPT,  # baseline
    2: SYSTEM_PROMPT,  # blind mode test
    3: """\
You are an elite superforecaster who consistently ranks in the top 2% of prediction tournaments.
Your job is to estimate the TRUE probability of events resolving YES.

## Your Method

1. **REFERENCE CLASS**: Before ANYTHING else, identify the reference class.
   For "Will X happen by date Y?", ask: historically, how often do events
   like X happen in this timeframe? Use actual statistics if you know them.

2. **INSIDE VIEW**: Now consider the specifics of THIS particular case.
   What makes it more or less likely than the base rate? Be specific and
   quantitative about each update.

3. **PRE-MORTEM**: Imagine it's the future and the event DID happen.
   What's the most plausible story? Now imagine it DIDN'T happen. What's
   the most plausible story for that? Which story is more believable?

4. **SYNTHESIS**: Average your reference class estimate with your
   inside view, weighted by how much unique information the inside view adds.

## Critical Rules

- START from the base rate. Most people start from their gut and adjust
  insufficiently. You should start from data and adjust carefully.
- BEWARE OF NARRATIVE BIAS: a compelling story doesn't make something likely.
  Many compelling stories never happen.
- YOUR DEFAULT SHOULD BE NEAR THE BASE RATE, not near 50%.
  If incumbents win 65% of the time, start at 65%, not 50%.
- For questions about whether something will happen "first" or "next" among
  N options, the base rate is roughly 1/N unless there's strong evidence otherwise.
- For "Will X happen by date Y" questions, consider: how far away is the date?
  The further away, the more can happen, but also the less urgent.

Respond with ONLY a valid JSON array. No markdown, no code blocks.
""",
    4: """\
You are a quantitative prediction analyst. Estimate the probability of events resolving YES.

## Method

For EACH market, think step by step:

1. What TYPE of question is this? (binary outcome, multi-choice "who will...",
   deadline-based "will X happen by Y", threshold "will X exceed Y")

2. For MULTI-CHOICE questions (who will X, which Y):
   - How many realistic candidates are there?
   - Base rate per candidate ≈ 1 / (number of serious candidates)
   - Only deviate significantly if there's a clear frontrunner

3. For DEADLINE questions (will X happen by Y):
   - Has there been any indication this is imminent?
   - Most things that "could" happen by a deadline DON'T unless actively in progress
   - Default to LOW probability unless there's active momentum

4. For BINARY questions (will X happen):
   - What is the status quo? The status quo usually persists.
   - How large a change is required? Larger changes are less likely.

5. FINAL CHECK: Is your probability consistent with your reasoning?
   If you said "this is unlikely" but gave 40%, something is wrong.

Respond with ONLY a valid JSON array. No markdown, no code blocks.
""",
    5: """\
You are a calibrated forecaster. Your goal is ACCURACY, not boldness.

For each market, provide your probability estimate for YES.

KEY PRINCIPLE: Think about what you DON'T know.
- If you're unsure about a topic, stay close to the base rate or 50%
- Only move away from 50% when you have SPECIFIC, CONCRETE evidence
- A 60% prediction you're right about is better than an 80% prediction you're wrong about

For multi-candidate questions ("who will be the next X"):
- This specific candidate winning is usually unlikely (base rate ~1/N candidates)
- Only give >50% if this person is the overwhelming consensus choice

For "will X happen by date Y":
- Most predicted events DON'T happen on schedule
- Default to <30% unless actively underway

IMPORTANT: Events that seem newsworthy or dramatic are NOT more likely.
Availability bias makes us overestimate dramatic events.

Respond with ONLY a valid JSON array. No markdown, no code blocks.
""",
    6: """\
You are a prediction market analyst. Estimate P(YES) for each market.

RULES:
1. Base rate first, always. For N-way choices, start at 1/N.
2. Update conservatively. Each piece of evidence moves the probability by at most 10-15 percentage points.
3. Never go below 2% or above 98% — black swans exist in both directions.
4. For events requiring ACTION (someone doing something): default to LOW probability.
   Inertia is the strongest force in human affairs.
5. For events requiring NO ACTION (status quo continuing): default to HIGH probability.
6. Your confidence should reflect YOUR KNOWLEDGE, not the importance of the event.
   You can be uncertain about important things.

OUTPUT: JSON array only. No markdown.
""",
    7: """\
You are a superforecaster. Estimate P(YES) for each prediction market.

APPROACH:
- Identify the question type: multi-choice, deadline, binary, or threshold
- For multi-choice: P ≈ 1/N for each option, unless clear frontrunner
- For deadlines: most things don't happen on time. P usually < 25% unless imminent
- For binary: what's the base rate for this class of event?
- For thresholds: where is the current value relative to the threshold?

CALIBRATION CHECK:
After forming your estimate, ask: "If I had 100 events like this at this probability,
would about [X] of them actually happen?" Adjust until the answer feels right.

ANTI-BIAS:
- Don't anchor to round numbers (50%, 25%, 75%)
- Don't let the vividness of a scenario affect probability
- Remember: most predictions of change are too confident. Status quo bias exists for a reason.

OUTPUT: JSON array only.
""",
    8: """\
You are predicting binary outcomes for Kalshi prediction markets.
For each market, estimate the probability it resolves YES.

FRAMEWORK:
1. Classification: What kind of prediction is this?
   a) PERSON-SPECIFIC: "Will [person] do/be [X]?" → Consider their track record, incentives, constraints
   b) EVENT-TIMING: "Will [X] happen by [date]?" → Consider base rate of similar events, current trajectory
   c) MULTI-OPTION: "Will [specific option] be chosen?" → Start at 1/N, adjust for evidence
   d) THRESHOLD: "Will [metric] exceed [value]?" → Consider current level, trend, volatility

2. Evidence Quality Assessment:
   - STRONG evidence: Official announcements, verified data, legal rulings → large updates (±20-30%)
   - MODERATE evidence: Expert consensus, polls, trends → medium updates (±10-15%)
   - WEAK evidence: Rumors, speculation, "could happen" reasoning → small updates (±5%)

3. Probability Assignment:
   - Start with base rate
   - Apply evidence-based updates
   - Sanity check: would you bet real money at these odds?

JSON array only. No markdown.
""",
    9: """\
You are an elite prediction market analyst combining quantitative rigor with deep domain knowledge.
For each market, estimate the probability it resolves YES.

## STEP-BY-STEP FRAMEWORK

1. **CLASSIFY** the question type:
   - MULTI-OPTION ("Will [X] be next Y?"): Base rate = 1/N where N = serious candidates.
     Only deviate significantly with strong evidence of a frontrunner.
   - DEADLINE ("Will X happen by Y?"): Most things don't happen on schedule.
     Only give >40% if there's active, concrete momentum toward the event.
   - BINARY ("Will X happen?"): What's the status quo? Status quo usually persists.
   - THRESHOLD ("Will X exceed Y?"): Where's the current value? What's the trend?

2. **ASSESS EVIDENCE** quality:
   - Official data, court rulings, confirmed actions → large update (±20-30%)
   - Expert consensus, strong polls, clear trends → medium update (±10-15%)
   - Speculation, rumors, "could happen" → small update (±5% max)

3. **USE THE MARKET PRICE** as a Bayesian prior. The market aggregates many informed
   participants. Only disagree when you have SPECIFIC information the market may be
   mispricing. If you don't have strong reason to disagree, your estimate should be
   CLOSE to the market price (within ±10%).

4. **CALIBRATION CHECK**: After your estimate, ask: "If I saw 100 similar situations
   at this probability, would about [X] actually happen?" Adjust until it feels right.

## KEY BIASES TO AVOID
- Don't be contrarian for its own sake — the market is usually approximately right
- Don't let narrative vividness affect probability (dramatic ≠ likely)
- Events requiring someone to DO something are less likely than events requiring inaction
- For multi-choice: resist the urge to spread probability too evenly across options

JSON array only. No markdown, no code blocks.
""",
}

# Use the latest by default, or select by iteration
def get_system_prompt(iteration):
    return SYSTEM_PROMPTS.get(iteration, SYSTEM_PROMPTS[max(SYSTEM_PROMPTS.keys())])

def make_batch_prompt(markets, blind=False):
    """Create a batch analysis prompt for multiple markets."""
    lines = []
    for i, m in enumerate(markets, 1):
        title = m["title"]
        subtitle = m.get("subtitle", "")
        close_time = m.get("close_time", "Unknown")
        volume = m.get("volume", 0)

        if blind:
            lines.append(
                f'{i}. Ticker: {m["ticker"]}\n'
                f'   Question: {title}\n'
                f'   {f"Details: {subtitle}" if subtitle else ""}\n'
                f'   Volume: {volume:,} contracts | Closes: {close_time}'
            )
        else:
            # Use mid_price if available, otherwise last_price
            price = m.get("_mid_price", m.get("last_price", 50))
            implied = price / 100.0
            lines.append(
                f'{i}. Ticker: {m["ticker"]}\n'
                f'   Question: {title}\n'
                f'   {f"Details: {subtitle}" if subtitle else ""}\n'
                f'   Market Price: {price}c (implies {implied:.0%})\n'
                f'   Volume: {volume:,} contracts | Closes: {close_time}'
            )

    prompt = f"""Analyze these prediction markets. For EACH one, estimate the probability of YES.

Markets:
{chr(10).join(lines)}

For EACH market, respond with a JSON array:
[
  {{
    "ticker": "<ticker>",
    "ai_probability": <your estimate 0.0-1.0>,
    "confidence": <how confident 0.0-1.0>,
    "reasoning": "<brief: base rate -> updates -> conclusion>"
  }},
  ...
]

IMPORTANT: Output ONLY the JSON array. No markdown, no code blocks."""
    return prompt


def parse_response(text):
    """Extract JSON from Claude's response."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try finding JSON array in text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def run_backtest(markets, blind=False, iteration=1):
    """Run backtest on a set of settled markets."""
    results = []
    total_cost = 0.0
    total_input = 0
    total_output = 0
    failed = 0

    sys_prompt = get_system_prompt(iteration)

    # Process in batches
    batches = [markets[i:i+BATCH_SIZE] for i in range(0, len(markets), BATCH_SIZE)]

    print(f"\n{'='*70}")
    print(f"  BACKTEST ITERATION #{iteration}")
    print(f"  Markets: {len(markets)} | Batches: {len(batches)} | Blind: {blind}")
    print(f"  Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"{'='*70}\n")

    for batch_idx, batch in enumerate(batches):
        prompt = make_batch_prompt(batch, blind=blind)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                temperature=TEMPERATURE,
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            inp = response.usage.input_tokens
            out = response.usage.output_tokens
            cost = inp * 3 / 1_000_000 + out * 15 / 1_000_000
            total_cost += cost
            total_input += inp
            total_output += out

            text = response.content[0].text
            parsed = parse_response(text)

            if parsed is None:
                print(f"  Batch {batch_idx+1}: PARSE FAILED")
                failed += len(batch)
                continue

            # Match predictions to markets
            pred_map = {r["ticker"]: r for r in parsed if "ticker" in r}

            for m in batch:
                ticker = m["ticker"]
                pred = pred_map.get(ticker)

                if not pred:
                    failed += 1
                    continue

                ai_prob = float(pred["ai_probability"])
                actual_result = m.get("result", "")
                actual_yes = 1.0 if actual_result == "yes" else 0.0
                market_prob = (m.get("_mid_price", m.get("last_price", 50)) or 50) / 100.0

                # Did AI predict correctly?
                ai_predicted_yes = ai_prob > 0.5
                actual_was_yes = actual_result == "yes"
                correct = ai_predicted_yes == actual_was_yes

                # Brier score (lower is better, 0 = perfect)
                brier = (ai_prob - actual_yes) ** 2
                market_brier = (market_prob - actual_yes) ** 2

                results.append({
                    "ticker": ticker,
                    "title": m["title"][:100],
                    "category": m.get("_category", ""),
                    "ai_probability": ai_prob,
                    "market_probability": market_prob,
                    "actual_result": actual_result,
                    "actual_yes": actual_yes,
                    "correct": correct,
                    "brier": brier,
                    "market_brier": market_brier,
                    "edge": ai_prob - market_prob,
                    "confidence": float(pred.get("confidence", 0.5)),
                    "reasoning": pred.get("reasoning", ""),
                })

            pct = (batch_idx + 1) / len(batches) * 100
            print(f"  Batch {batch_idx+1}/{len(batches)} ({pct:.0f}%) - cost: ${cost:.4f}")

        except Exception as e:
            print(f"  Batch {batch_idx+1}: ERROR - {e}")
            failed += len(batch)
            time.sleep(2)

    return results, {
        "total_cost": total_cost,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "failed": failed,
    }


def compute_metrics(results):
    """Compute comprehensive backtest metrics."""
    if not results:
        return {}

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    win_rate = correct / n

    # Brier scores
    avg_brier = sum(r["brier"] for r in results) / n
    avg_market_brier = sum(r["market_brier"] for r in results) / n

    # Calibration buckets
    buckets = {
        "0-20%": {"predicted": [], "actual": []},
        "20-40%": {"predicted": [], "actual": []},
        "40-60%": {"predicted": [], "actual": []},
        "60-80%": {"predicted": [], "actual": []},
        "80-100%": {"predicted": [], "actual": []},
    }

    for r in results:
        p = r["ai_probability"]
        if p < 0.2: bucket = "0-20%"
        elif p < 0.4: bucket = "20-40%"
        elif p < 0.6: bucket = "40-60%"
        elif p < 0.8: bucket = "60-80%"
        else: bucket = "80-100%"
        buckets[bucket]["predicted"].append(p)
        buckets[bucket]["actual"].append(r["actual_yes"])

    calibration = {}
    for name, data in buckets.items():
        if data["predicted"]:
            avg_pred = sum(data["predicted"]) / len(data["predicted"])
            avg_actual = sum(data["actual"]) / len(data["actual"])
            calibration[name] = {
                "count": len(data["predicted"]),
                "avg_predicted": round(avg_pred, 3),
                "actual_rate": round(avg_actual, 3),
                "error": round(abs(avg_pred - avg_actual), 3),
            }

    # Simulated trading P&L
    # If we bet on every market where |edge| > 5%, did we make money?
    trades = [r for r in results if abs(r["edge"]) > 0.05]
    trade_wins = 0
    trade_pnl = 0.0
    for t in trades:
        if t["edge"] > 0:  # bet YES
            won = t["actual_result"] == "yes"
        else:  # bet NO
            won = t["actual_result"] == "no"

        if won:
            trade_wins += 1
            # Simplified P&L: win (1 - entry_price), lose entry_price
            entry = t["market_probability"] if t["edge"] > 0 else (1 - t["market_probability"])
            trade_pnl += (1 - entry)
        else:
            entry = t["market_probability"] if t["edge"] > 0 else (1 - t["market_probability"])
            trade_pnl -= entry

    trade_win_rate = trade_wins / len(trades) if trades else 0

    # By category
    by_category = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if r["correct"]:
            by_category[cat]["correct"] += 1

    return {
        "n_markets": n,
        "n_correct": correct,
        "win_rate": round(win_rate, 4),
        "avg_brier_ai": round(avg_brier, 4),
        "avg_brier_market": round(avg_market_brier, 4),
        "brier_improvement": round(avg_market_brier - avg_brier, 4),
        "calibration": calibration,
        "n_trades": len(trades),
        "trade_win_rate": round(trade_win_rate, 4),
        "trade_pnl": round(trade_pnl, 2),
        "by_category": {k: {**v, "win_rate": round(v["correct"]/v["total"], 3)}
                       for k, v in by_category.items() if v["total"] > 0},
    }


def print_report(metrics, api_stats, iteration):
    """Print a formatted report."""
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS - ITERATION #{iteration}")
    print(f"{'='*70}")

    print(f"\n  Overall Accuracy:")
    print(f"    Markets analyzed:  {metrics['n_markets']}")
    print(f"    Correct predictions: {metrics['n_correct']}/{metrics['n_markets']} "
          f"({metrics['win_rate']:.1%})")

    print(f"\n  Brier Scores (lower = better):")
    print(f"    AI:     {metrics['avg_brier_ai']:.4f}")
    print(f"    Market: {metrics['avg_brier_market']:.4f}")
    imp = metrics['brier_improvement']
    print(f"    AI vs Market: {'+' if imp > 0 else ''}{imp:.4f} "
          f"({'AI better' if imp > 0 else 'Market better'})")

    print(f"\n  Calibration (predicted vs actual):")
    for bucket, data in metrics.get("calibration", {}).items():
        if data["count"] > 0:
            bar = "█" * int(data["actual_rate"] * 20)
            print(f"    {bucket:8s}: predicted={data['avg_predicted']:.1%} "
                  f"actual={data['actual_rate']:.1%} "
                  f"err={data['error']:.1%} n={data['count']} {bar}")

    print(f"\n  Simulated Trading (|edge| > 5%):")
    print(f"    Trades taken:    {metrics['n_trades']}")
    print(f"    Trade win rate:  {metrics['trade_win_rate']:.1%}")
    print(f"    Simulated P&L:   ${metrics['trade_pnl']:.2f}")

    print(f"\n  By Category:")
    for cat, data in sorted(metrics.get("by_category", {}).items(),
                           key=lambda x: -x[1]["total"]):
        print(f"    {cat:25s}: {data['correct']}/{data['total']} ({data['win_rate']:.1%})")

    print(f"\n  API Cost: ${api_stats['total_cost']:.4f} "
          f"(in={api_stats['total_input_tokens']:,} out={api_stats['total_output_tokens']:,})")
    if api_stats['failed']:
        print(f"  Failed: {api_stats['failed']} markets")

    print(f"{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50, help="Number of markets to sample")
    parser.add_argument("--blind", action="store_true", help="Hide market prices")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number")
    parser.add_argument("--all", action="store_true", help="Run on all markets")
    args = parser.parse_args()

    # Load data
    with open(DATA_FILE) as f:
        all_markets = json.load(f)

    # Filter to markets with results, reasonable volume, and mid-range prices
    markets = [m for m in all_markets
               if m.get("result") in ("yes", "no")
               and m.get("volume", 0) >= 50
               and m.get("_mid_price", 0) > 0]

    print(f"Loaded {len(all_markets)} markets, {len(markets)} usable (with results + mid-price)")

    if not args.all:
        # Sample deterministically for reproducibility
        import hashlib
        markets.sort(key=lambda m: hashlib.md5(m["ticker"].encode()).hexdigest())
        markets = markets[:args.sample]

    # Run backtest
    results, api_stats = run_backtest(markets, blind=args.blind, iteration=args.iteration)

    # Compute and print metrics
    metrics = compute_metrics(results)
    print_report(metrics, api_stats, args.iteration)

    # Save results
    output = {
        "iteration": args.iteration,
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "blind": args.blind,
            "sample_size": len(markets),
            "batch_size": BATCH_SIZE,
        },
        "metrics": metrics,
        "api_stats": api_stats,
        "predictions": results,
    }

    outfile = f"{RESULTS_DIR}/iteration_{args.iteration}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {outfile}")


if __name__ == "__main__":
    main()
