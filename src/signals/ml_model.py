"""Advanced LLM-based short-term price-move predictor.

Queries the Gemini API with recent price action to predict intraday movement.
Falls back to Claude (Anthropic) when Gemini exhausts its quota or rate-limit.
Only queries the LLM if the technical signal is flashing to save API costs.
"""
from __future__ import annotations

import os
import json
import pandas as pd
from typing import Optional

from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)

# Gemini error substrings that indicate resource exhaustion (quota / rate-limit / overload)
_GEMINI_RESOURCE_ERRORS = (
    "429",
    "quota",
    "rate limit",
    "resource exhausted",
    "503",
    "overloaded",
    "unavailable",
)


def _is_resource_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _GEMINI_RESOURCE_ERRORS)


def _build_prompt(symbol: str, tech_score: float, prompt_data: str) -> str:
    return (
        f"You are an expert quantitative day trader. Analyze the following 5-minute OHLCV chart data for {symbol}.\n"
        f"Technical signal is currently flashing with a score of {tech_score:.2f} (negative is bearish, positive is bullish).\n"
        f"Data (latest at bottom):\n{prompt_data}\n\n"
        "Will the price go UP or DOWN in the next 15 minutes?\n"
        "Respond with ONLY a valid JSON object containing:\n"
        '- "score": A float between -1.0 (strong sell) to 1.0 (strong buy)\n'
        '- "confidence": A float between 0.0 to 1.0\n'
        '- "reason": A brief 1 sentence explanation.\n'
    )


def _parse_llm_response(raw_text: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:-3].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:-3].strip()
    return json.loads(raw_text)


class MLSignal:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.horizon = int(cfg.get("prediction_horizon_minutes", 15))
        self.min_conf = float(cfg.get("min_confidence", 0.55))
        self.tech_threshold = float(cfg.get("tech_threshold", 0.2))

        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.gemini_key:
            log.warning("GEMINI_API_KEY not set. MLSignal will try Claude if ANTHROPIC_API_KEY is set.")
        if not self.anthropic_key:
            log.warning("ANTHROPIC_API_KEY not set. No Claude fallback available.")

    # ---------- private ----------

    def _call_gemini(self, prompt: str) -> dict:
        from google import genai
        client = genai.Client(api_key=self.gemini_key)
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return _parse_llm_response(response.text)

    def _call_claude(self, prompt: str) -> dict:
        import anthropic
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_llm_response(message.content[0].text)

    def _query_llm(self, symbol: str, prompt: str) -> tuple[dict, str]:
        """Return (parsed result, provider_name). Tries Gemini first, falls back to Claude."""
        if self.gemini_key:
            try:
                return self._call_gemini(prompt), "gemini"
            except Exception as e:
                if _is_resource_error(e):
                    log.warning("Gemini resource exhausted for %s (%s); falling back to Claude.", symbol, e)
                else:
                    raise  # non-resource errors bubble up to evaluate()'s handler

        if self.anthropic_key:
            result = self._call_claude(prompt)
            return result, "claude"

        raise RuntimeError("No LLM provider available (both GEMINI_API_KEY and ANTHROPIC_API_KEY are unset).")

    # ---------- public ----------

    def evaluate(self, symbol: str, bars: pd.DataFrame, tech_signal: Optional[Signal] = None) -> Optional[Signal]:
        if tech_signal is None or abs(tech_signal.score) < self.tech_threshold:
            return Signal(
                symbol=symbol,
                source=SignalSource.ML,
                score=0.0,
                confidence=0.0,
                metadata={"reason": "technicals not flashing, skipped to save cost"},
            )

        if not self.gemini_key and not self.anthropic_key:
            return Signal(
                symbol=symbol,
                source=SignalSource.ML,
                score=0.0,
                confidence=0.0,
                metadata={"reason": "no api key"},
            )

        if bars is None or len(bars) < 20:
            return None

        try:
            recent_bars = bars.tail(20).copy()
            recent_bars["SMA_5"] = recent_bars["Close"].rolling(5).mean()
            prompt_data = recent_bars[["Open", "High", "Low", "Close", "Volume", "SMA_5"]].tail(10).to_csv()
            prompt = _build_prompt(symbol, tech_signal.score, prompt_data)

            result, provider = self._query_llm(symbol, prompt)
            score = float(result.get("score", 0.0))
            confidence = float(result.get("confidence", 0.0))

            if confidence < self.min_conf:
                return Signal(
                    symbol=symbol,
                    source=SignalSource.ML,
                    score=0.0,
                    confidence=confidence,
                    metadata={"reason": "below min confidence", "llm_reason": result.get("reason", ""), "provider": provider},
                )

            return Signal(
                symbol=symbol,
                source=SignalSource.ML,
                score=score,
                confidence=confidence,
                metadata={"horizon_min": self.horizon, "llm_reason": result.get("reason", ""), "provider": provider},
            )

        except Exception as e:
            log.warning("LLM prediction failed for %s: %s", symbol, e)
            return None

    @classmethod
    def train_from_history(cls, *args, **kwargs):
        log.warning("train_from_history is deprecated. The bot now uses an online LLM API.")
