"""Advanced LLM-based short-term price-move predictor.

Queries the Gemini API with recent price action to predict intraday movement.
Falls back to Claude (Anthropic) when Gemini exhausts its quota or rate-limit.

Cost-control features
─────────────────────
1. Tech-signal gate   — only calls LLM when |tech_score| >= tech_threshold (0.30).
   Low-conviction bars get score=0 for free.
2. Long cache TTL     — result is reused for `prediction_horizon_minutes` (45 min).
   With 60s polling that is 45 free hits per symbol between API calls.
3. Daily call budget  — hard cap of `max_daily_calls` (80) per calendar day.
   Once exhausted every symbol returns score=0 for the rest of the day.
4. Per-cycle cap      — within a single engine cycle only the top
   `max_calls_per_cycle` (3) symbols by |tech_score| actually call the LLM;
   the rest use their cached value or return score=0.
"""
from __future__ import annotations

import os
import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)

# Gemini error substrings that indicate resource exhaustion / rate-limit
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


def _build_prompt(symbol: str, tech_score: float, prompt_data: str, horizon: int) -> str:
    return (
        f"You are an expert quantitative day trader. Analyse the following 5-minute OHLCV "
        f"chart data for {symbol}.\n"
        f"Technical signal score: {tech_score:.2f} (negative=bearish, positive=bullish).\n"
        f"Data (latest at bottom):\n{prompt_data}\n\n"
        f"Will the price go UP or DOWN in the next {horizon} minutes?\n"
        "Respond with ONLY a valid JSON object with keys:\n"
        '  "score"      — float -1.0 (strong sell) to 1.0 (strong buy)\n'
        '  "confidence" — float 0.0 to 1.0\n'
        '  "reason"     — one sentence explanation\n'
    )


def _parse_llm_response(raw_text: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:-3].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:-3].strip()
    return json.loads(raw_text)


def _null_signal(symbol: str, reason: str) -> Signal:
    return Signal(
        symbol=symbol,
        source=SignalSource.ML,
        score=0.0,
        confidence=0.0,
        metadata={"reason": reason},
    )


class MLSignal:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.horizon = int(cfg.get("prediction_horizon_minutes", 45))
        self.min_conf = float(cfg.get("min_confidence", 0.55))
        self.tech_threshold = float(cfg.get("tech_threshold", 0.30))
        self.max_daily_calls = int(cfg.get("max_daily_calls", 80))
        self.max_calls_per_cycle = int(cfg.get("max_calls_per_cycle", 3))

        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.gemini_key:
            log.warning("GEMINI_API_KEY not set. MLSignal will try Claude if ANTHROPIC_API_KEY is set.")
        if not self.anthropic_key:
            log.warning("ANTHROPIC_API_KEY not set. No Claude fallback available.")

        # Per-symbol result cache: {symbol: (fetched_at, Signal)}
        self._cache: Dict[str, Tuple[datetime, Signal]] = {}

        # Daily call budget tracking
        self._budget_date: Optional[date] = None
        self._calls_today: int = 0

        # Per-cycle state — reset by the engine each cycle via prepare_cycle()
        # Maps symbol -> abs(tech_score) for the current cycle's candidates
        self._cycle_candidates: Dict[str, float] = {}
        self._cycle_calls_remaining: int = self.max_calls_per_cycle

    # ---------- budget helpers ----------

    def _budget_ok(self) -> bool:
        """Return True if we still have budget for today."""
        today = date.today()
        if self._budget_date != today:
            self._budget_date = today
            self._calls_today = 0
        return self._calls_today < self.max_daily_calls

    def _charge_budget(self) -> None:
        self._calls_today += 1
        remaining = self.max_daily_calls - self._calls_today
        log.info(
            "Gemini call charged | used=%d/%d remaining=%d",
            self._calls_today, self.max_daily_calls, remaining,
        )
        if remaining == 10:
            log.warning("Gemini daily budget: only 10 calls left for today.")
        elif remaining == 0:
            log.warning("Gemini daily budget EXHAUSTED — ML signal disabled for rest of day.")

    # ---------- cycle management ----------

    def prepare_cycle(self, candidates: Dict[str, float]) -> None:
        """Called once per engine cycle with {symbol: abs_tech_score} for all
        tickers whose tech score exceeds the threshold.

        Selects the top `max_calls_per_cycle` symbols that don't have a fresh
        cache entry and marks them as eligible for a live LLM call this cycle.
        All others will return their cached value or score=0.
        """
        self._cycle_candidates.clear()
        ttl = timedelta(minutes=self.horizon)
        now = datetime.now()

        # Filter to symbols with stale/missing cache and rank by tech strength.
        stale = {
            sym: score
            for sym, score in candidates.items()
            if sym not in self._cache or (now - self._cache[sym][0]) >= ttl
        }
        top = sorted(stale, key=lambda s: stale[s], reverse=True)[: self.max_calls_per_cycle]
        self._cycle_candidates = {s: stale[s] for s in top}
        self._cycle_calls_remaining = len(self._cycle_candidates)

        if self._cycle_candidates:
            log.info(
                "MLSignal cycle: %d tickers eligible for LLM call this cycle: %s  "
                "(budget %d/%d used today)",
                len(self._cycle_candidates),
                list(self._cycle_candidates.keys()),
                self._calls_today,
                self.max_daily_calls,
            )

    # ---------- LLM callers ----------

    def _call_gemini(self, prompt: str) -> dict:
        from google import genai
        client = genai.Client(api_key=self.gemini_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
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

    def _query_llm(self, symbol: str, prompt: str) -> Tuple[dict, str]:
        """Return (parsed result, provider_name). Gemini first, Claude fallback."""
        if self.gemini_key:
            try:
                result = self._call_gemini(prompt)
                self._charge_budget()
                return result, "gemini"
            except Exception as e:
                if _is_resource_error(e):
                    log.warning(
                        "Gemini resource exhausted for %s (%s); falling back to Claude.",
                        symbol, e,
                    )
                else:
                    raise

        if self.anthropic_key:
            result = self._call_claude(prompt)
            self._charge_budget()
            return result, "claude"

        raise RuntimeError(
            "No LLM provider available (both GEMINI_API_KEY and ANTHROPIC_API_KEY are unset)."
        )

    # ---------- public ----------

    def evaluate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        tech_signal: Optional[Signal] = None,
    ) -> Optional[Signal]:

        # Gate 1: technical signal too weak — free return
        if tech_signal is None or abs(tech_signal.score) < self.tech_threshold:
            return _null_signal(symbol, "tech score below threshold — skipped to save quota")

        # Gate 2: no API keys at all
        if not self.gemini_key and not self.anthropic_key:
            return _null_signal(symbol, "no API key configured")

        # Gate 3: serve from cache if fresh
        ttl = timedelta(minutes=self.horizon)
        cached = self._cache.get(symbol)
        if cached is not None and (datetime.now() - cached[0]) < ttl:
            age_min = int((datetime.now() - cached[0]).total_seconds() // 60)
            log.debug("LLM cache hit for %s (age %dm / TTL %dm).", symbol, age_min, self.horizon)
            return cached[1]

        # Gate 4: daily budget exhausted
        if not self._budget_ok():
            return _null_signal(symbol, "daily Gemini budget exhausted")

        # Gate 5: per-cycle cap — only the pre-selected top-N symbols may call LLM
        if symbol not in self._cycle_candidates:
            # Return stale cache if available, otherwise score=0
            if cached is not None:
                log.debug(
                    "LLM stale-cache for %s (not in cycle top-%d).",
                    symbol, self.max_calls_per_cycle,
                )
                return cached[1]
            return _null_signal(symbol, f"not in top-{self.max_calls_per_cycle} this cycle")

        # All gates passed — make the live LLM call
        if bars is None or len(bars) < 20:
            return None

        try:
            recent = bars.tail(20).copy()
            col = "Close" if "Close" in recent.columns else "close"
            recent["SMA_5"] = recent[col].rolling(5).mean()
            ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "SMA_5"]
                          if c in recent.columns]
            prompt_data = recent[ohlcv_cols].tail(10).to_csv()
            prompt = _build_prompt(symbol, tech_signal.score, prompt_data, self.horizon)

            result, provider = self._query_llm(symbol, prompt)
            score = float(result.get("score", 0.0))
            confidence = float(result.get("confidence", 0.0))
            llm_reason = result.get("reason", "")

            if confidence < self.min_conf:
                signal = _null_signal(symbol, f"low confidence ({confidence:.2f}) — {llm_reason}")
                signal.metadata.update({"provider": provider, "confidence": confidence})
            else:
                signal = Signal(
                    symbol=symbol,
                    source=SignalSource.ML,
                    score=score,
                    confidence=confidence,
                    metadata={
                        "horizon_min": self.horizon,
                        "llm_reason": llm_reason,
                        "provider": provider,
                    },
                )
                log.info(
                    "ML signal %s | score=%+.2f conf=%.2f provider=%s reason=%s",
                    symbol, score, confidence, provider, llm_reason,
                )

            self._cache[symbol] = (datetime.now(), signal)
            # Remove from cycle candidates so a second evaluate() call this cycle
            # won't trigger another API hit.
            self._cycle_candidates.pop(symbol, None)
            return signal

        except Exception as e:
            log.warning("LLM prediction failed for %s: %s", symbol, e)
            return None

    @classmethod
    def train_from_history(cls, *args, **kwargs):
        log.warning("train_from_history is deprecated. The bot now uses an online LLM API.")
