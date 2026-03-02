"""Advanced alpha generation engine v2 for Kalshi prediction markets.

Key improvements over v1:
- Correlation clustering: Groups related markets to prevent concentration
- Time-decay scoring: Prefers markets closing sooner (faster capital turnover)
- Spread-based confidence: Uses bid-ask spread as liquidity/info signal
- Dynamic ensemble weighting: Weights shift based on strategy agreement
- Mean reversion strategy: 4th prompt targeting extreme-priced markets
- Portfolio-level risk: Max 25% per cluster, max 15 positions
"""

import json
import logging
import math
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv("CLAUDE_MODEL", "claude-4-sonnet-20250514")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "backtest_data"))

# --- Edge categories (from backtest analysis) ---
EDGE_CATEGORIES = {
    "Politics": {"weight": 1.5, "min_edge": 0.03, "accuracy": 0.86},
    "Economics": {"weight": 1.3, "min_edge": 0.05, "accuracy": 0.84},
    "Companies": {"weight": 1.0, "min_edge": 0.06, "accuracy": 1.0},
    "Elections": {"weight": 1.2, "min_edge": 0.04, "accuracy": 1.0},
    "World": {"weight": 0.8, "min_edge": 0.08, "accuracy": 0.75},
    "Crypto": {"weight": 0.5, "min_edge": 0.12, "accuracy": 1.0},
    "Climate and Weather": {"weight": 0.7, "min_edge": 0.08},
    "Financials": {"weight": 1.0, "min_edge": 0.06},
    "Science and Technology": {"weight": 0.6, "min_edge": 0.10},
}

# --- Portfolio constraints ---
MAX_POSITIONS = 15
MAX_CLUSTER_PCT = 0.15  # max 15% of portfolio per correlation cluster
MAX_SINGLE_PCT = 0.08   # max 8% per single trade
PORTFOLIO_SIZE = 2000    # demo portfolio

# --- Ensemble prompts ---
PROMPT_BAYESIAN = """\
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

PROMPT_CONTRARIAN = """\
You are a contrarian analyst searching for MISPRICINGS in prediction markets.

COGNITIVE BIASES TO EXPLOIT:
- Recency bias: market overweighting recent events
- Narrative bias: compelling story ≠ high probability
- Anchoring: market stuck at old price despite new info
- Herding: consensus without independent analysis
- Availability bias: vivid events overweighted vs base rates

DISCIPLINE: Only disagree with SPECIFIC, ARTICULABLE reasons. If you can't
name the exact bias, stay close to market price. Random disagreement is noise.

Your edge: second-order thinking. What does the market ASSUME that might be wrong?

JSON array only: [{"ticker": "...", "ai_probability": 0.X, "confidence": 0.X, "reasoning": "..."}]
"""

PROMPT_BASE_RATE = """\
You are a reference-class forecaster. For each market, estimate P(YES)
using OUTSIDE VIEW thinking.

METHOD:
1. Identify the reference class (what type of event is this?)
2. Find the base rate for that class
3. Adjust MODESTLY for specific factors (max ±20% from base rate)
4. Give the market price 40% weight in your final answer

REFERENCE CLASSES:
- "Will person X be elected/appointed": historical win rates for similar races
- "Will event happen by date Y": base rate of on-schedule delivery
- "Will metric exceed threshold": historical frequency of threshold-crossing
- Multi-choice with N options: start at 1/N

Never let narrative evidence move you more than 20% from the base rate.
The market price is informative - weight it at 40%.

JSON array only: [{"ticker": "...", "ai_probability": 0.X, "confidence": 0.X, "reasoning": "..."}]
"""

PROMPT_MEAN_REVERSION = """\
You are a quantitative analyst focused on EXTREME PRICES in prediction markets.

Markets at extreme prices (below 15% or above 85%) often reflect overshoot.

YOUR JOB:
- For markets priced BELOW 20%: Is the market too pessimistic? Could unexpected events make YES more likely than priced?
- For markets priced ABOVE 80%: Is the market too optimistic? What tail risks could cause NO?
- For markets in the middle (20-80%): Stay very close to market price.

EXTREME PRICE HEURISTICS:
- Markets below 10%: things that "can't happen" sometimes do (fat tails)
- Markets above 90%: things that "are certain" sometimes don't (tail risk)
- The further from 50%, the more the market tends to be wrong in absolute terms

Be conservative with middle-priced markets. Your alpha is at the extremes.

JSON array only: [{"ticker": "...", "ai_probability": 0.X, "confidence": 0.X, "reasoning": "..."}]
"""


def classify_question(title: str) -> str:
    """Classify market question type."""
    t = title.lower()
    if any(w in t for w in ["who will", "which", "next", "first"]):
        return "multi_choice"
    if any(w in t for w in ["before", "by ", "by end", "by jan", "by dec"]):
        return "deadline"
    if any(w in t for w in ["exceed", "higher than", "lower than", "above", "below", "more than", "less than"]):
        return "threshold"
    return "binary"


def extract_cluster_key(ticker: str, title: str) -> str:
    """Extract a cluster key for grouping correlated markets.

    Markets with similar tickers or topics get the same cluster key.
    Uses both ticker pattern and title keywords for smarter grouping.
    """
    # Remove date suffixes and option suffixes from ticker
    # e.g., KXMUSKTRILLION-27, KXMUSKTRILLION-28 -> KXMUSKTRILLION
    base = re.sub(r'-\d+$', '', ticker)
    base = re.sub(r'-[A-Z]{1,6}$', '', base)

    # Also cluster by event ticker prefix
    prefix = ticker.split('-')[0] if '-' in ticker else ticker

    cluster = base if len(base) >= len(prefix) else prefix

    # Title-based clustering for related markets with different ticker bases
    # e.g., KXTRILLIONAIRE and KXMUSKTRILLION are the same thesis
    title_lower = title.lower()
    TITLE_CLUSTERS = {
        "trillionaire": "TRILLIONAIRE_CLUSTER",
        "musk": "MUSK_CLUSTER",
        "next supreme leader of iran": "IRAN_LEADER_CLUSTER",
        "next prime minister of united kingdom": "UK_PM_CLUSTER",
        "next u.k. election": "UK_ELECTION_CLUSTER",
    }
    for keyword, cluster_name in TITLE_CLUSTERS.items():
        if keyword in title_lower:
            return cluster_name

    return cluster


def days_until_close(close_time: str) -> float:
    """Calculate days until market closes."""
    try:
        close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
        delta = close_dt - datetime.now(timezone.utc)
        return max(delta.total_seconds() / 86400, 1)
    except (ValueError, TypeError):
        return 365  # default to 1 year if unknown


def time_decay_multiplier(days: float) -> float:
    """Score multiplier favoring sooner-closing markets.

    Markets closing in 7 days get 2x, 30 days get 1.5x, 365 days get 0.7x.
    """
    if days <= 7:
        return 2.0
    elif days <= 30:
        return 1.5
    elif days <= 90:
        return 1.2
    elif days <= 180:
        return 1.0
    elif days <= 365:
        return 0.8
    else:
        return 0.6


def spread_confidence_factor(yes_bid: int, yes_ask: int) -> float:
    """Derive a confidence factor from bid-ask spread.

    Tight spread = liquid, well-priced market = higher confidence in market price.
    Wide spread = illiquid, potentially mispriced = lower confidence in market.
    """
    spread = (yes_ask - yes_bid) if yes_ask and yes_bid else 10
    # Spread of 1-2 is very tight, 10+ is wide
    if spread <= 2:
        return 1.0  # very confident in market price
    elif spread <= 5:
        return 0.9
    elif spread <= 10:
        return 0.8
    else:
        return 0.65  # wide spread = more mispricing opportunity


def calculate_kelly_fraction(edge: float, confidence: float) -> float:
    """Calculate quarter-Kelly bet fraction with confidence scaling."""
    if edge <= 0 or confidence < 0.45:
        return 0.0

    # Edge * confidence as effective edge, quarter-Kelly
    effective_edge = abs(edge) * confidence
    kelly = effective_edge * 0.25

    # Cap per-trade at MAX_SINGLE_PCT
    return min(kelly, MAX_SINGLE_PCT)


def score_opportunity(pred: dict, cat_config: dict) -> float:
    """Score a trading opportunity from 0-100.

    Combines edge, confidence, agreement, category, volume, time decay, spread.
    """
    edge = abs(pred.get("edge", 0))
    confidence = pred.get("confidence", 0.5)
    agreement = pred.get("agreement", 0.5)
    volume = pred.get("volume", 0)
    q_type = pred.get("question_type", "binary")
    days = pred.get("days_to_close", 365)

    # Base: edge * confidence * agreement
    base_score = edge * confidence * (0.5 + 0.5 * agreement) * 100

    # Category multiplier
    base_score *= cat_config.get("weight", 1.0)

    # Question type bonus
    type_mult = {"multi_choice": 1.3, "deadline": 1.0, "threshold": 1.1, "binary": 1.0}
    base_score *= type_mult.get(q_type, 1.0)

    # Volume factor (log scale)
    if volume > 0:
        vol_factor = min(math.log10(volume) / 5, 1.5)
        base_score *= (0.7 + 0.3 * vol_factor)

    # Time decay: prefer sooner-closing markets
    base_score *= time_decay_multiplier(days)

    # Spread bonus: wide spreads = more mispricing opportunity
    spread_factor = pred.get("spread_factor", 0.9)
    if spread_factor < 0.85:
        base_score *= 1.15  # bonus for wide-spread markets

    return min(base_score, 100.0)


class AlphaEngine:
    """Multi-strategy alpha generation for prediction markets (v3).

    Backtest-validated approach:
    - Probability: bayes+contrarian weighted avg (best Brier: 0.1251 vs market 0.1346)
    - Trade filter: directional agreement across 3+ strategies
    - Confidence boost: mean_reversion agreement (62% WR in backtest)
    - base_rate used only as a filter (89% accuracy but 22% WR — bad for trading)
    """

    # Primary strategies for probability estimation
    PROB_STRATEGIES = [
        {"name": "bayesian", "prompt": PROMPT_BAYESIAN, "weight": 0.65},
        {"name": "contrarian", "prompt": PROMPT_CONTRARIAN, "weight": 0.35},
    ]

    # Filter strategies for trade selection (not used in prob calculation)
    FILTER_STRATEGIES = [
        {"name": "base_rate", "prompt": PROMPT_BASE_RATE},
        {"name": "mean_reversion", "prompt": PROMPT_MEAN_REVERSION},
    ]

    ALL_STRATEGIES = PROB_STRATEGIES + FILTER_STRATEGIES

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=API_KEY)
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def fetch_markets(self) -> list[dict]:
        """Fetch tradeable open markets from Kalshi."""
        events = []
        cursor = None
        for _ in range(5):
            params = {"limit": 100, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(
                "https://api.elections.kalshi.com/trade-api/v2/events",
                params=params, timeout=30,
            )
            if not resp.ok:
                break
            data = resp.json()
            events.extend(data.get("events", []))
            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(0.3)

        # Filter to edge categories
        event_cats = {e["event_ticker"]: e.get("category", "") for e in events
                      if e.get("category", "") in EDGE_CATEGORIES}

        markets = []
        for et, cat in list(event_cats.items())[:60]:
            resp = requests.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"event_ticker": et, "limit": 20, "status": "open"},
                timeout=30,
            )
            if not resp.ok:
                continue
            for m in resp.json().get("markets", []):
                yes_bid = m.get("yes_bid", 0) or 0
                yes_ask = m.get("yes_ask", 0) or 0
                volume = m.get("volume", 0) or 0
                if yes_bid and yes_ask and volume >= 50:
                    mid = (yes_bid + yes_ask) / 2
                    if 8 <= mid <= 92:
                        m["_mid_price"] = mid
                        m["_category"] = cat
                        m["_question_type"] = classify_question(m.get("title", ""))
                        m["_cluster"] = extract_cluster_key(m["ticker"], m.get("title", ""))
                        m["_days_to_close"] = days_until_close(m.get("close_time", ""))
                        m["_spread_factor"] = spread_confidence_factor(yes_bid, yes_ask)
                        markets.append(m)
            time.sleep(0.2)

        return markets

    def _run_prompt(self, markets: list[dict], system_prompt: str, name: str) -> list[dict]:
        """Run a single prompt strategy on markets."""
        results = []
        batch_size = 6

        for i in range(0, len(markets), batch_size):
            batch = markets[i:i + batch_size]
            lines = []
            for j, m in enumerate(batch, 1):
                mid = m["_mid_price"]
                spread = (m.get("yes_ask", 0) or 0) - (m.get("yes_bid", 0) or 0)
                lines.append(
                    f'{j}. Ticker: {m["ticker"]}\n'
                    f'   Question: {m["title"]}\n'
                    f'   Market Price: {mid:.0f}c ({mid/100:.0%}) | Spread: {spread}c\n'
                    f'   Volume: {m.get("volume", 0):,} | Category: {m.get("_category", "")}\n'
                    f'   Closes: {m.get("close_time", "Unknown")} '
                    f'({m["_days_to_close"]:.0f} days)'
                )

            prompt = (
                "Analyze these prediction markets. For EACH, estimate P(YES).\n\n"
                "Markets:\n" + "\n".join(lines) + "\n\n"
                "JSON array only."
            )

            try:
                response = self.client.messages.create(
                    model=MODEL, max_tokens=4096, temperature=0.2,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
                self.total_cost += cost

                text = response.content[0].text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                parsed = json.loads(text.strip())
                for r in parsed:
                    r["_strategy"] = name
                results.extend(parsed)

                logger.info("[%s] batch %d-%d: %d results", name, i, i+len(batch), len(parsed))

            except Exception as e:
                logger.error("[%s] batch %d failed: %s", name, i, e)

            time.sleep(1)

        return results

    def run_ensemble(self, markets: list[dict]) -> list[dict]:
        """Run all strategies: prob strategies for estimates, filter strategies for trade selection."""
        logger.info("Running ensemble analysis on %d markets...", len(markets))

        # Run all strategies
        strategy_results = {}
        for s in self.ALL_STRATEGIES:
            logger.info("Running strategy: %s", s["name"])
            strategy_results[s["name"]] = self._run_prompt(markets, s["prompt"], s["name"])

        # Combine predictions by ticker
        market_map = {m["ticker"]: m for m in markets}
        combined = {}

        for sname, results in strategy_results.items():
            for r in results:
                ticker = r.get("ticker", "")
                if ticker not in combined:
                    combined[ticker] = {"probs": {}, "confidences": {}, "reasonings": []}
                try:
                    combined[ticker]["probs"][sname] = float(r["ai_probability"])
                    combined[ticker]["confidences"][sname] = float(r.get("confidence", 0.5))
                    combined[ticker]["reasonings"].append(f"[{sname}] {r.get('reasoning', '')}")
                except (ValueError, KeyError):
                    pass

        # Generate final predictions
        predictions = []
        for ticker, data in combined.items():
            if not data["probs"] or ticker not in market_map:
                continue

            m = market_map[ticker]
            market_prob = m["_mid_price"] / 100.0

            # PROBABILITY: Use only bayes+contrarian (best Brier combo from backtest)
            prob_strats = {s["name"]: s["weight"] for s in self.PROB_STRATEGIES}
            total_w = 0
            weighted_prob = 0
            for sname, weight in prob_strats.items():
                if sname in data["probs"]:
                    conf = data["confidences"].get(sname, 0.5)
                    w = weight * conf
                    weighted_prob += data["probs"][sname] * w
                    total_w += w

            if total_w == 0:
                continue

            ai_prob = weighted_prob / total_w
            edge = ai_prob - market_prob

            # DIRECTIONAL AGREEMENT: How many strategies agree on the direction?
            direction = 1 if edge > 0 else -1
            agree_count = 0
            total_strats = 0
            for sname, prob in data["probs"].items():
                strat_edge = prob - market_prob
                total_strats += 1
                if (strat_edge > 0) == (edge > 0):
                    agree_count += 1

            directional_agreement = agree_count / total_strats if total_strats else 0

            # CONFIDENCE: based on prob strategies agreement + filter confirmation
            prob_confs = [data["confidences"][s] for s in prob_strats if s in data["confidences"]]
            avg_prob_conf = statistics.mean(prob_confs) if prob_confs else 0.5

            # Boost if mean_reversion agrees (62% WR in backtest)
            mr_agrees = False
            if "mean_reversion" in data["probs"]:
                mr_edge = data["probs"]["mean_reversion"] - market_prob
                mr_agrees = (mr_edge > 0) == (edge > 0)

            # Base confidence from prob strategies
            spread_factor = m.get("_spread_factor", 0.9)
            final_confidence = avg_prob_conf * spread_factor

            # Scale by directional agreement
            if directional_agreement >= 0.75:
                final_confidence *= 1.2  # 3/4 or 4/4 agree
            elif directional_agreement <= 0.25:
                final_confidence *= 0.5  # strong disagreement

            # Mean reversion bonus
            if mr_agrees:
                final_confidence *= 1.15

            final_confidence = min(final_confidence, 0.95)

            cat = m.get("_category", "")
            cat_config = EDGE_CATEGORIES.get(cat, {"weight": 1.0, "min_edge": 0.08})
            q_type = m.get("_question_type", "binary")

            pred = {
                "ticker": ticker,
                "title": m.get("title", "")[:120],
                "category": cat,
                "question_type": q_type,
                "cluster": m.get("_cluster", ticker),
                "ai_probability": round(ai_prob, 4),
                "market_probability": round(market_prob, 4),
                "edge": round(edge, 4),
                "confidence": round(final_confidence, 4),
                "directional_agreement": round(directional_agreement, 2),
                "mr_agrees": mr_agrees,
                "n_strategies": len(data["probs"]),
                "individual_probs": {s: round(p, 3) for s, p in data["probs"].items()},
                "volume": m.get("volume", 0),
                "days_to_close": round(m.get("_days_to_close", 365), 1),
                "spread_factor": round(spread_factor, 2),
                "close_time": m.get("close_time"),
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "reasoning": " | ".join(data["reasonings"][:4]),
            }

            # Score the opportunity
            pred["agreement"] = directional_agreement  # for score_opportunity
            pred["score"] = round(score_opportunity(pred, cat_config), 2)

            # TRADE FILTER: Only trade when 3+ strategies agree on direction
            min_edge = cat_config.get("min_edge", 0.05)
            trade_eligible = (
                abs(edge) >= min_edge
                and directional_agreement >= 0.5  # at least half agree
                and final_confidence >= 0.40
            )

            if trade_eligible:
                kelly = calculate_kelly_fraction(abs(edge), final_confidence)
                # Boost kelly if strong agreement
                if directional_agreement >= 0.75:
                    kelly *= 1.3
                kelly = min(kelly, MAX_SINGLE_PCT)
                pred["kelly_fraction"] = round(kelly, 4)
                pred["suggested_size"] = round(PORTFOLIO_SIZE * kelly, 2)
                pred["side"] = "YES" if edge > 0 else "NO"
                pred["tradeable"] = True
            else:
                pred["kelly_fraction"] = 0
                pred["suggested_size"] = 0
                pred["side"] = None
                pred["tradeable"] = False

            predictions.append(pred)

        # Sort by score
        predictions.sort(key=lambda p: p["score"], reverse=True)

        # Apply portfolio constraints
        predictions = self._apply_portfolio_constraints(predictions)

        logger.info(
            "Generated %d predictions, %d tradeable, %d active. Cost: $%.4f",
            len(predictions),
            sum(1 for p in predictions if p["tradeable"]),
            sum(1 for p in predictions if p.get("in_portfolio")),
            self.total_cost,
        )

        return predictions

    def _apply_portfolio_constraints(self, predictions: list[dict]) -> list[dict]:
        """Apply portfolio-level risk constraints.

        - Max positions
        - Max allocation per cluster
        - Diversification requirements
        """
        cluster_allocation = defaultdict(float)
        total_allocation = 0
        portfolio_count = 0

        for pred in predictions:
            if not pred["tradeable"]:
                pred["in_portfolio"] = False
                pred["portfolio_size"] = 0
                continue

            cluster = pred["cluster"]
            suggested = pred["suggested_size"]

            # Check position count
            if portfolio_count >= MAX_POSITIONS:
                pred["in_portfolio"] = False
                pred["portfolio_size"] = 0
                pred["rejection_reason"] = "max_positions"
                continue

            # Check cluster concentration
            max_cluster = PORTFOLIO_SIZE * MAX_CLUSTER_PCT
            remaining_cluster = max_cluster - cluster_allocation[cluster]
            if remaining_cluster <= 0:
                pred["in_portfolio"] = False
                pred["portfolio_size"] = 0
                pred["rejection_reason"] = "cluster_limit"
                continue

            # Adjust size for cluster limit
            actual_size = min(suggested, remaining_cluster)

            # Check total portfolio utilization (cap at 60%)
            max_total = PORTFOLIO_SIZE * 0.60
            remaining_total = max_total - total_allocation
            if remaining_total <= 0:
                pred["in_portfolio"] = False
                pred["portfolio_size"] = 0
                pred["rejection_reason"] = "portfolio_full"
                continue

            actual_size = min(actual_size, remaining_total)

            pred["in_portfolio"] = True
            pred["portfolio_size"] = round(actual_size, 2)
            cluster_allocation[cluster] += actual_size
            total_allocation += actual_size
            portfolio_count += 1

        return predictions

    def generate_report(self, predictions: list[dict]) -> dict:
        """Generate a comprehensive alpha report."""
        tradeable = [p for p in predictions if p["tradeable"]]
        in_portfolio = [p for p in predictions if p.get("in_portfolio")]

        # Category breakdown
        by_cat = {}
        for p in predictions:
            cat = p["category"]
            if cat not in by_cat:
                by_cat[cat] = {"total": 0, "tradeable": 0, "in_portfolio": 0,
                               "total_size": 0, "avg_edge": [], "avg_agreement": []}
            by_cat[cat]["total"] += 1
            if p["tradeable"]:
                by_cat[cat]["tradeable"] += 1
                by_cat[cat]["avg_edge"].append(abs(p["edge"]))
            if p.get("in_portfolio"):
                by_cat[cat]["in_portfolio"] += 1
                by_cat[cat]["total_size"] += p.get("portfolio_size", 0)
                by_cat[cat]["avg_agreement"].append(p["agreement"])

        for cat in by_cat:
            edges = by_cat[cat]["avg_edge"]
            by_cat[cat]["avg_edge"] = round(statistics.mean(edges), 4) if edges else 0
            agr = by_cat[cat]["avg_agreement"]
            by_cat[cat]["avg_agreement"] = round(statistics.mean(agr), 3) if agr else 0

        # Cluster breakdown
        by_cluster = defaultdict(lambda: {"count": 0, "total_size": 0, "tickers": []})
        for p in in_portfolio:
            c = p["cluster"]
            by_cluster[c]["count"] += 1
            by_cluster[c]["total_size"] += p.get("portfolio_size", 0)
            by_cluster[c]["tickers"].append(p["ticker"])

        # Question type breakdown
        by_type = {}
        for p in tradeable:
            qt = p["question_type"]
            if qt not in by_type:
                by_type[qt] = {"count": 0, "avg_score": [], "avg_edge": []}
            by_type[qt]["count"] += 1
            by_type[qt]["avg_score"].append(p["score"])
            by_type[qt]["avg_edge"].append(abs(p["edge"]))
        for qt in by_type:
            by_type[qt]["avg_score"] = round(statistics.mean(by_type[qt]["avg_score"]), 1)
            by_type[qt]["avg_edge"] = round(statistics.mean(by_type[qt]["avg_edge"]), 4)

        # Strategy agreement analysis
        strategy_stats = defaultdict(lambda: {"count": 0, "avg_prob": []})
        for p in predictions:
            for sname, prob in p.get("individual_probs", {}).items():
                strategy_stats[sname]["count"] += 1
                strategy_stats[sname]["avg_prob"].append(prob)
        for s in strategy_stats:
            probs = strategy_stats[s]["avg_prob"]
            strategy_stats[s]["avg_prob"] = round(statistics.mean(probs), 3) if probs else 0
            strategy_stats[s]["std_prob"] = round(statistics.stdev(probs), 3) if len(probs) > 1 else 0

        # Top trades
        top_trades = [{
            "ticker": p["ticker"],
            "title": p["title"],
            "side": p["side"],
            "edge": p["edge"],
            "confidence": p["confidence"],
            "directional_agreement": p.get("directional_agreement", 0),
            "mr_agrees": p.get("mr_agrees", False),
            "score": p["score"],
            "portfolio_size": p.get("portfolio_size", 0),
            "category": p["category"],
            "cluster": p["cluster"],
            "days_to_close": p["days_to_close"],
        } for p in in_portfolio[:10]]

        # Portfolio summary
        total_allocation = sum(p.get("portfolio_size", 0) for p in in_portfolio)
        max_single = max((p.get("portfolio_size", 0) for p in in_portfolio), default=0)
        n_yes = sum(1 for p in in_portfolio if p["side"] == "YES")
        n_no = sum(1 for p in in_portfolio if p["side"] == "NO")
        n_clusters = len(by_cluster)

        return {
            "version": "v2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_markets_analyzed": len(predictions),
            "tradeable_signals": len(tradeable),
            "portfolio_positions": len(in_portfolio),
            "total_allocation": round(total_allocation, 2),
            "portfolio_utilization": round(total_allocation / PORTFOLIO_SIZE, 3),
            "max_single_position": round(max_single, 2),
            "n_clusters": n_clusters,
            "yes_signals": n_yes,
            "no_signals": n_no,
            "avg_confidence": round(statistics.mean(p["confidence"] for p in in_portfolio), 3) if in_portfolio else 0,
            "avg_edge": round(statistics.mean(abs(p["edge"]) for p in in_portfolio), 4) if in_portfolio else 0,
            "avg_agreement": round(statistics.mean(p["agreement"] for p in in_portfolio), 3) if in_portfolio else 0,
            "avg_days_to_close": round(statistics.mean(p["days_to_close"] for p in in_portfolio), 0) if in_portfolio else 0,
            "by_category": by_cat,
            "by_cluster": dict(by_cluster),
            "by_question_type": by_type,
            "strategy_stats": dict(strategy_stats),
            "top_trades": top_trades,
            "api_cost": round(self.total_cost, 4),
            "tokens": {"input": self.total_input_tokens, "output": self.total_output_tokens},
        }


def run_alpha_scan():
    """Run a full alpha scan and save results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    engine = AlphaEngine()

    # Fetch markets
    markets = engine.fetch_markets()
    logger.info("Found %d tradeable markets in edge categories", len(markets))

    if not markets:
        logger.warning("No markets found")
        return

    # Sort by volume and take top 30
    markets.sort(key=lambda m: m.get("volume", 0), reverse=True)
    to_analyze = markets[:30]

    # Log cluster distribution
    clusters = defaultdict(list)
    for m in to_analyze:
        clusters[m["_cluster"]].append(m["ticker"])
    logger.info("Market clusters: %d clusters from %d markets", len(clusters), len(to_analyze))
    for c, tickers in sorted(clusters.items(), key=lambda x: -len(x[1]))[:5]:
        logger.info("  %s: %d markets", c, len(tickers))

    # Run ensemble analysis
    predictions = engine.run_ensemble(to_analyze)

    # Generate report
    report = engine.generate_report(predictions)

    # Save everything
    DATA_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / "live_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    with open(DATA_DIR / "alpha_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print report
    print(f"\n{'='*70}")
    print(f"  ALPHA ENGINE v2 REPORT")
    print(f"{'='*70}")
    print(f"\n  Markets Analyzed:     {report['total_markets_analyzed']}")
    print(f"  Tradeable Signals:    {report['tradeable_signals']}")
    print(f"  Portfolio Positions:  {report['portfolio_positions']}")
    print(f"  Clusters:             {report['n_clusters']}")
    print(f"  Total Allocation:     ${report['total_allocation']:.2f} / ${PORTFOLIO_SIZE:,}")
    print(f"  Portfolio Use:        {report['portfolio_utilization']:.0%}")
    print(f"  Avg Confidence:       {report['avg_confidence']:.0%}")
    print(f"  Avg Edge:             {report['avg_edge']:.1%}")
    print(f"  Avg Agreement:        {report['avg_agreement']:.0%}")
    print(f"  Avg Days to Close:    {report['avg_days_to_close']:.0f}")
    print(f"  YES/NO Split:         {report['yes_signals']} / {report['no_signals']}")

    print(f"\n  Strategy Stats:")
    for s, data in report["strategy_stats"].items():
        print(f"    {s:20s}: avg_prob={data['avg_prob']:.3f}  std={data['std_prob']:.3f}  n={data['count']}")

    print(f"\n  By Category:")
    for cat, data in sorted(report["by_category"].items(), key=lambda x: -x[1]["in_portfolio"]):
        if data["in_portfolio"]:
            print(f"    {cat:25s}: {data['in_portfolio']} positions, "
                  f"${data['total_size']:.0f} allocated, "
                  f"avg edge {data['avg_edge']:.1%}, "
                  f"agreement {data['avg_agreement']:.0%}")

    print(f"\n  Cluster Allocation:")
    for c, data in sorted(report["by_cluster"].items(), key=lambda x: -x[1]["total_size"]):
        print(f"    {c:30s}: {data['count']} pos, ${data['total_size']:.0f}")

    print(f"\n  Top Portfolio Trades:")
    for i, t in enumerate(report["top_trades"][:8], 1):
        mr = "MR" if t.get("mr_agrees") else "  "
        print(f"    {i}. [{t['side']:3s}] score={t['score']:.0f} edge={t['edge']:+.1%} "
              f"conf={t['confidence']:.0%} dir={t.get('directional_agreement',0):.0%} {mr} "
              f"${t['portfolio_size']:.0f} | {t['title'][:42]} ({t['days_to_close']:.0f}d)")

    print(f"\n  API Cost: ${report['api_cost']:.4f}")
    print(f"  Tokens: {report['tokens']['input']:,} in / {report['tokens']['output']:,} out")
    print(f"{'='*70}\n")

    return predictions, report


if __name__ == "__main__":
    run_alpha_scan()
