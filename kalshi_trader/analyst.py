"""Claude Sonnet 4.6 market analyst - the brains of the operation.

Uses Claude to analyze prediction markets and estimate true probabilities.

v2 improvements:
- Anti-anchoring: option to hide market price so Claude reasons independently
- Structured Fermi reasoning: forces base-rate thinking
- Anonymization mode: strips identifiers to reduce training data leakage
- Explicit debiasing instructions
- Confidence calibration nudges
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Optional

import anthropic

from .config import ClaudeConfig
from .models import Market, MarketAnalysis, Side

logger = logging.getLogger(__name__)

# ---- System Prompts ----

SYSTEM_PROMPT = """\
You are an expert prediction market analyst and quantitative forecaster.
Your job is to estimate the TRUE probability of events, then identify where
prediction markets misprice outcomes.

## Your Forecasting Method (follow this exact sequence)

1. **Parse the question**: What EXACTLY does the contract ask? What are the
   precise resolution criteria?

2. **Base rate**: Before considering specifics, what is the historical base
   rate for this TYPE of event? (e.g., "incumbent presidents win re-election
   ~65% of the time", "monthly CPI surprises to the upside ~40% of the time")

3. **Update from specifics**: What current information shifts the probability
   UP or DOWN from the base rate? List each factor with its directional effect.

4. **Consider the other side**: Steelman the opposing view. What would need to
   be true for the OPPOSITE outcome? How likely is that scenario?

5. **Final estimate**: Combine into a single probability. Check: does this
   feel calibrated? When you say 70%, do events like this happen ~70% of the time?

## Calibration Rules

- DO NOT anchor to the market price. Form your estimate INDEPENDENTLY first.
- If you are uncertain, your probability should be CLOSER TO 50%, not further.
- Extreme probabilities (>90% or <10%) require EXTREME evidence. Default to
  more moderate estimates unless the evidence is overwhelming.
- Your edge comes from BETTER REASONING, not from being more extreme.
- A 5% edge on a well-calibrated estimate is more valuable than a 20% edge
  on an overconfident one.

## Output

Respond with ONLY a valid JSON object matching the requested schema.
No markdown, no explanation outside the JSON.
"""

SYSTEM_PROMPT_BLIND = """\
You are an expert prediction market analyst and quantitative forecaster.
Your job is to estimate the TRUE probability of events described below.

IMPORTANT: You are estimating probability WITHOUT seeing the current market
price. This is intentional — it prevents anchoring bias and lets you form
an independent estimate.

## Your Forecasting Method (follow this exact sequence)

1. **Parse the question**: What EXACTLY is being asked?
2. **Base rate**: What is the historical base rate for this TYPE of event?
3. **Update from specifics**: What factors shift probability from the base rate?
4. **Steelman the opposite**: What would make the other outcome happen?
5. **Final calibrated estimate**: Combine into a single probability.

## Calibration Rules

- If uncertain, stay CLOSER TO 50%.
- Extreme probabilities (>90% or <10%) require extreme evidence.
- Better to be slightly wrong but well-calibrated than boldly wrong.

Respond with ONLY a valid JSON object. No markdown, no extra text.
"""

# ---- Analysis Prompts ----

ANALYSIS_PROMPT = """\
Estimate the TRUE probability for this prediction market contract:

**Question**: {title}
{subtitle}

**Market Price**: {market_price} cents ({implied_prob:.1%} implied probability)
**Volume**: {volume:,} contracts | **Open Interest**: {open_interest:,}
**Closes**: {close_time}
**Analysis Date**: {current_date}

Use your structured forecasting method. Respond with this JSON:
{{
    "base_rate": <base rate probability for this type of event, 0.0-1.0>,
    "ai_probability": <your final calibrated estimate, 0.0-1.0>,
    "confidence": <how confident in your estimate, 0.0-1.0>,
    "reasoning": "<step-by-step reasoning: base rate -> updates -> final>",
    "key_factors": ["<factor shifting probability up/down>", ...],
    "risk_factors": ["<what could make you wrong>", ...]
}}
"""

ANALYSIS_PROMPT_BLIND = """\
Estimate the TRUE probability for this prediction market contract:

**Question**: {title}
{subtitle}

**Volume**: {volume:,} contracts traded
**Closes**: {close_time}
**Analysis Date**: {current_date}

NOTE: The current market price is HIDDEN. Estimate the probability purely
from your reasoning about the question. This prevents anchoring bias.

Use your structured forecasting method. Respond with this JSON:
{{
    "base_rate": <base rate probability for this type of event, 0.0-1.0>,
    "ai_probability": <your final calibrated estimate, 0.0-1.0>,
    "confidence": <how confident in your estimate, 0.0-1.0>,
    "reasoning": "<step-by-step reasoning: base rate -> updates -> final>",
    "key_factors": ["<factor shifting probability up/down>", ...],
    "risk_factors": ["<what could make you wrong>", ...]
}}
"""

BATCH_ANALYSIS_PROMPT = """\
Analyze these prediction markets. For EACH one, estimate the TRUE probability.

Markets:
{markets_text}

Analysis Date: {current_date}

For EACH market, follow this process:
1. Identify the base rate for this type of event
2. Update from current specifics
3. Arrive at a calibrated probability

Respond with a JSON array, one object per market:
[
  {{
    "ticker": "<market ticker>",
    "base_rate": <base rate, 0.0-1.0>,
    "ai_probability": <calibrated estimate, 0.0-1.0>,
    "confidence": <confidence, 0.0-1.0>,
    "reasoning": "<brief step-by-step>",
    "key_factors": ["<factor>", ...],
    "risk_factors": ["<risk>", ...]
  }},
  ...
]

Rank by size of edge (difference between your estimate and market price).
"""

BATCH_ANALYSIS_PROMPT_BLIND = """\
Analyze these prediction markets. For EACH one, estimate the TRUE probability.
Market prices are HIDDEN to prevent anchoring bias.

Markets:
{markets_text}

Analysis Date: {current_date}

For EACH market, follow this process:
1. Identify the base rate for this type of event
2. Update from current specifics
3. Arrive at a calibrated probability

Respond with a JSON array, one object per market:
[
  {{
    "ticker": "<market ticker>",
    "base_rate": <base rate, 0.0-1.0>,
    "ai_probability": <calibrated estimate, 0.0-1.0>,
    "confidence": <confidence, 0.0-1.0>,
    "reasoning": "<brief step-by-step>",
    "key_factors": ["<factor>", ...],
    "risk_factors": ["<risk>", ...]
  }},
  ...
]
"""


def _anonymize_title(title: str) -> str:
    """Strip specific names/dates that might trigger training data recall.

    Replaces specific proper nouns with generic labels while keeping the
    structure of the question intact.
    """
    # Replace specific dates with relative references
    title = re.sub(r"\b(January|February|March|April|May|June|July|August|"
                   r"September|October|November|December)\s+\d{1,2},?\s*\d{4}\b",
                   "[DATE]", title)
    title = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "[DATE]", title)

    # Replace dollar amounts
    title = re.sub(r"\$[\d,]+\.?\d*\s*(billion|million|trillion|B|M|T)?",
                   "$[AMOUNT]", title)

    return title


class ClaudeAnalyst:
    """Uses Claude Sonnet 4.6 to analyze prediction markets.

    Supports two modes:
    - Standard: Shows market price (for live trading)
    - Blind: Hides market price (for unbiased backtesting)
    """

    def __init__(self, config: ClaudeConfig, blind_mode: bool = False,
                 anonymize: bool = False):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.blind_mode = blind_mode
        self.anonymize = anonymize
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_BLIND if self.blind_mode else SYSTEM_PROMPT

    def _call_claude(self, user_prompt: str) -> str:
        """Make a call to Claude and track usage."""
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": user_prompt}],
        )

        self.call_count += 1
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        text = response.content[0].text

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return text.strip()

    def _prepare_title(self, market: Market) -> tuple[str, str]:
        """Prepare market title/subtitle, optionally anonymizing."""
        title = market.title
        subtitle = market.subtitle
        if self.anonymize:
            title = _anonymize_title(title)
            subtitle = _anonymize_title(subtitle)
        return title, subtitle

    def analyze_market(self, market: Market) -> MarketAnalysis:
        """Analyze a single market and return probability estimate + reasoning."""
        title, subtitle = self._prepare_title(market)

        if self.blind_mode:
            prompt = ANALYSIS_PROMPT_BLIND.format(
                title=title,
                subtitle=subtitle,
                volume=market.volume,
                open_interest=market.open_interest,
                close_time=market.close_time or "Unknown",
                current_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            )
        else:
            prompt = ANALYSIS_PROMPT.format(
                title=title,
                subtitle=subtitle,
                market_price=market.yes_mid,
                implied_prob=market.implied_probability,
                volume=market.volume,
                open_interest=market.open_interest,
                close_time=market.close_time or "Unknown",
                current_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            )

        try:
            result_text = self._call_claude(prompt)
            result = json.loads(result_text)

            ai_prob = float(result["ai_probability"])
            market_prob = market.implied_probability
            edge = ai_prob - market_prob

            # Determine recommended side
            recommended_side = None
            if abs(edge) > 0.02:
                recommended_side = Side.YES if edge > 0 else Side.NO

            analysis = MarketAnalysis(
                ticker=market.ticker,
                market_title=market.title,
                ai_probability=ai_prob,
                market_probability=market_prob,
                edge=edge,
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                recommended_side=recommended_side,
                key_factors=result.get("key_factors", []),
                risk_factors=result.get("risk_factors", []),
            )

            logger.info(
                "Analyzed %s: AI=%.1f%% Market=%.1f%% Edge=%.1f%% Conf=%.0f%% %s",
                market.ticker,
                ai_prob * 100,
                market_prob * 100,
                edge * 100,
                analysis.confidence * 100,
                "[BLIND]" if self.blind_mode else "",
            )

            return analysis

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse Claude response for %s: %s", market.ticker, e)
            return MarketAnalysis(
                ticker=market.ticker,
                market_title=market.title,
                ai_probability=market.implied_probability,
                market_probability=market.implied_probability,
                edge=0,
                confidence=0,
                reasoning=f"Analysis failed: {e}",
            )

    def analyze_markets_batch(self, markets: list[Market]) -> list[MarketAnalysis]:
        """Analyze multiple markets in a single Claude call for efficiency."""
        if not markets:
            return []

        lines = []
        for i, m in enumerate(markets, 1):
            title, _ = self._prepare_title(m)
            if self.blind_mode:
                lines.append(
                    f"{i}. [{m.ticker}] {title}\n"
                    f"   Volume: {m.volume:,} | Closes: {m.close_time or 'Unknown'}"
                )
            else:
                lines.append(
                    f"{i}. [{m.ticker}] {title}\n"
                    f"   Price: {m.yes_mid}c ({m.implied_probability:.1%}) | "
                    f"Volume: {m.volume:,} | Closes: {m.close_time or 'Unknown'}"
                )

        if self.blind_mode:
            prompt = BATCH_ANALYSIS_PROMPT_BLIND.format(
                markets_text="\n".join(lines),
                current_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            )
        else:
            prompt = BATCH_ANALYSIS_PROMPT.format(
                markets_text="\n".join(lines),
                current_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            )

        try:
            result_text = self._call_claude(prompt)
            results = json.loads(result_text)

            analyses = []
            market_map = {m.ticker: m for m in markets}

            for r in results:
                ticker = r.get("ticker", "")
                market = market_map.get(ticker)
                if not market:
                    continue

                ai_prob = float(r["ai_probability"])
                market_prob = market.implied_probability
                edge = ai_prob - market_prob

                recommended_side = None
                if abs(edge) > 0.02:
                    recommended_side = Side.YES if edge > 0 else Side.NO

                analyses.append(
                    MarketAnalysis(
                        ticker=ticker,
                        market_title=market.title,
                        ai_probability=ai_prob,
                        market_probability=market_prob,
                        edge=edge,
                        confidence=float(r.get("confidence", 0.5)),
                        reasoning=r.get("reasoning", ""),
                        recommended_side=recommended_side,
                        key_factors=r.get("key_factors", []),
                        risk_factors=r.get("risk_factors", []),
                    )
                )

            return analyses

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Batch analysis failed: %s. Falling back to individual.", e)
            return [self.analyze_market(m) for m in markets]

    def get_usage_stats(self) -> dict:
        """Return API usage statistics."""
        return {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": (
                self.total_input_tokens * 3 / 1_000_000
                + self.total_output_tokens * 15 / 1_000_000
            ),
        }
