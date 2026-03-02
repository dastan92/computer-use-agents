"""Live prediction service - polls Kalshi markets and makes AI predictions.

Runs as a background task, saving predictions to JSON for the dashboard.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv("CLAUDE_MODEL", "claude-4-sonnet-20250514")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "backtest_data"))
PREDICTIONS_FILE = DATA_DIR / "live_predictions.json"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "3600"))  # default 1 hour

GOOD_CATEGORIES = [
    "Economics", "Politics", "Financials", "World",
    "Science and Technology", "Crypto", "Climate and Weather",
    "Companies", "Elections",
]

SYSTEM_PROMPT = """\
You are an elite prediction market analyst combining quantitative rigor with deep domain knowledge.
For each market, estimate the probability it resolves YES.

## FRAMEWORK

1. CLASSIFY the question type (multi-option, deadline, binary, threshold)
2. ASSESS evidence quality (strong/moderate/weak -> proportional updates)
3. USE THE MARKET PRICE as a Bayesian prior. Only disagree with specific evidence.
4. CALIBRATION CHECK: Would about [X] out of 100 similar situations resolve YES?

JSON array only. No markdown.
"""


def fetch_open_markets():
    """Fetch interesting open markets from Kalshi."""
    # First get events with good categories
    events = []
    cursor = None
    for _ in range(3):
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

    good_events = [e for e in events if e.get("category", "") in GOOD_CATEGORIES]
    event_cats = {e["event_ticker"]: e.get("category", "") for e in good_events}

    # Fetch markets for good events
    markets = []
    for e in good_events[:40]:
        et = e["event_ticker"]
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
                if 10 <= mid <= 90:
                    m["_mid_price"] = mid
                    m["_category"] = event_cats.get(et, "")
                    markets.append(m)
        time.sleep(0.2)

    return markets


def analyze_markets(markets, client):
    """Run Claude analysis on markets in batches."""
    predictions = []
    batch_size = 5

    for i in range(0, len(markets), batch_size):
        batch = markets[i:i + batch_size]
        lines = []
        for j, m in enumerate(batch, 1):
            mid = m["_mid_price"]
            lines.append(
                f'{j}. Ticker: {m["ticker"]}\n'
                f'   Question: {m["title"]}\n'
                f'   Market Price: {mid:.0f}c (implies {mid/100:.0%})\n'
                f'   Volume: {m.get("volume", 0):,} | Closes: {m.get("close_time", "Unknown")}'
            )

        prompt = (
            "Analyze these prediction markets. For EACH, estimate P(YES).\n\n"
            "Markets:\n" + "\n".join(lines) + "\n\n"
            "JSON array: [{\"ticker\": \"...\", \"ai_probability\": 0.X, "
            "\"confidence\": 0.X, \"reasoning\": \"...\"}]"
        )

        try:
            response = client.messages.create(
                model=MODEL, max_tokens=4096, temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            results = json.loads(text.strip())
            pred_map = {r["ticker"]: r for r in results if "ticker" in r}

            for m in batch:
                pred = pred_map.get(m["ticker"])
                if not pred:
                    continue
                ai_prob = float(pred["ai_probability"])
                market_prob = m["_mid_price"] / 100.0

                predictions.append({
                    "ticker": m["ticker"],
                    "title": m["title"][:120],
                    "category": m.get("_category", ""),
                    "ai_probability": ai_prob,
                    "market_probability": market_prob,
                    "edge": round(ai_prob - market_prob, 4),
                    "confidence": float(pred.get("confidence", 0.5)),
                    "reasoning": pred.get("reasoning", ""),
                    "volume": m.get("volume", 0),
                    "close_time": m.get("close_time"),
                    "predicted_at": datetime.now(timezone.utc).isoformat(),
                })

            logger.info("Analyzed batch %d-%d: %d predictions",
                       i, i + len(batch), len(predictions))

        except Exception as e:
            logger.error("Batch analysis failed: %s", e)

        time.sleep(1)

    return predictions


def save_predictions(predictions):
    """Save predictions to JSON for dashboard consumption."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by absolute edge
    predictions.sort(key=lambda p: abs(p.get("edge", 0)), reverse=True)

    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info("Saved %d predictions to %s", len(predictions), PREDICTIONS_FILE)


def run_once():
    """Run a single prediction cycle."""
    if not API_KEY:
        logger.error("No ANTHROPIC_API_KEY set")
        return []

    client = anthropic.Anthropic(api_key=API_KEY)

    logger.info("Fetching open markets...")
    markets = fetch_open_markets()
    logger.info("Found %d tradeable markets", len(markets))

    if not markets:
        return []

    # Analyze top 30 by volume
    markets.sort(key=lambda m: m.get("volume", 0), reverse=True)
    to_analyze = markets[:30]

    logger.info("Analyzing %d markets with Claude...", len(to_analyze))
    predictions = analyze_markets(to_analyze, client)

    save_predictions(predictions)
    return predictions


def run_loop():
    """Run prediction loop continuously."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting prediction service (interval=%ds)", POLL_INTERVAL)

    while True:
        try:
            predictions = run_once()
            logger.info("Cycle complete: %d predictions", len(predictions))
        except Exception:
            logger.exception("Prediction cycle failed")

        logger.info("Next cycle in %d seconds...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_loop()
