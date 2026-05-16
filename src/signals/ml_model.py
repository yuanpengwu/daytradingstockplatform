"""Advanced LLM-based short-term price-move predictor.

Queries the Gemini API with recent price action to predict intraday movement.
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

class MLSignal:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.horizon = int(cfg.get("prediction_horizon_minutes", 15))
        self.min_conf = float(cfg.get("min_confidence", 0.55))
        self.tech_threshold = float(cfg.get("tech_threshold", 0.2)) # Min tech score to trigger LLM
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            log.warning("GEMINI_API_KEY not found in environment. MLSignal will return neutral.")

    # ---------- public ----------
    def evaluate(self, symbol: str, bars: pd.DataFrame, tech_signal: Optional[Signal] = None) -> Optional[Signal]:
        # Cost-saving measure: Only query if Technicals are flashing
        if tech_signal is None or abs(tech_signal.score) < self.tech_threshold:
            return Signal(
                symbol=symbol,
                source=SignalSource.ML,
                score=0.0,
                confidence=0.0,
                metadata={"reason": "technicals not flashing, skipped to save cost"}
            )
            
        if not self.api_key:
            return Signal(symbol=symbol, source=SignalSource.ML, score=0.0, confidence=0.0, metadata={"reason": "no api key"})
            
        if bars is None or len(bars) < 20:
            return None
            
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            
            # Format recent bars for the prompt (last 20 bars)
            recent_bars = bars.tail(20).copy()
            # Calculate a few simple indicators for context
            recent_bars['SMA_5'] = recent_bars['Close'].rolling(5).mean()
            prompt_data = recent_bars[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5']].tail(10).to_csv()
            
            prompt = f"""You are an expert quantitative day trader. Analyze the following 5-minute OHLCV chart data for {symbol}.
Technical signal is currently flashing with a score of {tech_signal.score:.2f} (negative is bearish, positive is bullish).
Data (latest at bottom):
{prompt_data}

Will the price go UP or DOWN in the next 15 minutes? 
Respond with ONLY a valid JSON object containing:
- "score": A float between -1.0 (strong sell) to 1.0 (strong buy)
- "confidence": A float between 0.0 to 1.0
- "reason": A brief 1 sentence explanation.
"""

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            # Parse JSON (try to strip markdown blocks if model wraps it)
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            result = json.loads(raw_text)
            score = float(result.get("score", 0.0))
            confidence = float(result.get("confidence", 0.0))
            
            if confidence < self.min_conf:
                return Signal(
                    symbol=symbol,
                    source=SignalSource.ML,
                    score=0.0,
                    confidence=confidence,
                    metadata={"reason": "below min confidence", "llm_reason": result.get("reason", "")}
                )
                
            return Signal(
                symbol=symbol,
                source=SignalSource.ML,
                score=score,
                confidence=confidence,
                metadata={"horizon_min": self.horizon, "llm_reason": result.get("reason", "")}
            )
            
        except Exception as e:
            log.warning("LLM prediction failed for %s: %s", symbol, e)
            return None

    @classmethod
    def train_from_history(cls, *args, **kwargs):
        log.warning("train_from_history is deprecated. The bot now uses an online LLM API.")
