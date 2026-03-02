"""Claude Sonnet 4.6 market analyst - the brains of the operation.

Uses Claude to analyze prediction markets and estimate true probabilities.
The key insight: prediction markets are efficient but not perfect. Claude can
read the question, consider context, news, base rates, and reasoning to find
edges where the market misprices an outcome.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import anthropic

from .config import ClaudeConfig
from .models import Market, MarketAnalysis, Side

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert prediction market analyst and quantitative trader. Your job is
to estimate the TRUE probability of events described in prediction market contracts,
then compare your estimate to the market price to find trading edges.

You must think like a superforecaster:
- Use base rates and reference classes
- Consider multiple scenarios and weight them
- Account for known unknowns and update priors
- Be well-calibrated: when you say 70%, it should happen ~70% of the time
- Be honest about uncertainty - don't overfit to narratives

When analyzing a market, you will:
1. Understand what the contract is actually asking (exact resolution criteria)
2. Consider the current date and time horizon
3. Think about base rates for similar events
4. Identify key factors that could shift the probability
5. Assess what information the market might be missing
6. Provide a probability estimate with reasoning

CRITICAL: Your probability must be YOUR honest best estimate, not just agreeing
with the market. The whole point is to find mispricings. Be bold but calibrated.

Output your analysis as valid JSON matching the requested schema.
"""

ANALYSIS_PROMPT = """\
Analyze this prediction market and estimate the TRUE probability:

**Market**: {title}
{subtitle}

**Current Market Price**: {market_price} cents (implies {implied_prob:.1%} probability)
**Volume**: {volume:,} contracts traded
**Open Interest**: {open_interest:,}
**Close/Expiration**: {close_time}
**Current Date**: {current_date}

Please analyze this market and respond with a JSON object:
{{
    "ai_probability": <your estimated true probability, 0.0 to 1.0>,
    "confidence": <how confident you are in your estimate, 0.0 to 1.0>,
    "reasoning": "<detailed reasoning for your probability estimate>",
    "key_factors": ["<factor 1>", "<factor 2>", ...],
    "risk_factors": ["<risk 1>", "<risk 2>", ...]
}}

Think step by step. Consider base rates, current events, and what information
might not be priced in. Be specific in your reasoning.
"""

BATCH_ANALYSIS_PROMPT = """\
Analyze these prediction markets and rank them by trading opportunity.
For each, estimate the TRUE probability and identify any edge vs the market price.

Markets to analyze:
{markets_text}

Current Date: {current_date}

Respond with a JSON array of objects, one per market, each with:
{{
    "ticker": "<market ticker>",
    "ai_probability": <true probability estimate, 0.0-1.0>,
    "confidence": <confidence in estimate, 0.0-1.0>,
    "reasoning": "<brief reasoning>",
    "key_factors": ["<factor>", ...],
    "risk_factors": ["<risk>", ...]
}}

Rank by size of edge (difference between your estimate and market price).
Focus on markets where you have genuine insight or see clear mispricings.
"""


class ClaudeAnalyst:
    """Uses Claude Sonnet 4.6 to analyze prediction markets."""

    def __init__(self, config: ClaudeConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _call_claude(self, user_prompt: str) -> str:
        """Make a call to Claude and track usage."""
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=SYSTEM_PROMPT,
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

    def analyze_market(self, market: Market) -> MarketAnalysis:
        """Analyze a single market and return probability estimate + reasoning."""
        prompt = ANALYSIS_PROMPT.format(
            title=market.title,
            subtitle=market.subtitle,
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
            if abs(edge) > 0.02:  # at least 2% edge
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
                "Analyzed %s: AI=%.1f%% Market=%.1f%% Edge=%.1f%% Confidence=%.1f%%",
                market.ticker,
                ai_prob * 100,
                market_prob * 100,
                edge * 100,
                analysis.confidence * 100,
            )

            return analysis

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse Claude response for %s: %s", market.ticker, e)
            # Return neutral analysis on parse failure
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

        # Build market descriptions
        lines = []
        for i, m in enumerate(markets, 1):
            lines.append(
                f"{i}. [{m.ticker}] {m.title}\n"
                f"   Price: {m.yes_mid}c ({m.implied_probability:.1%}) | "
                f"Volume: {m.volume:,} | Closes: {m.close_time or 'Unknown'}"
            )

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
