"""News-sentiment signal.

Default: VADER (lightweight, no GPU). Optional: FinBERT via transformers
if installed and configured in `signals.sentiment.model = finbert`.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from ..data.news_feed import Headline, NewsFeed
from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)


class SentimentSignal:
    def __init__(self, cfg: dict, news_feed: Optional[NewsFeed] = None):
        self.cfg = cfg
        self.news = news_feed or NewsFeed(
            sources=cfg.get("sources", ("newsapi", "finnhub")),
            max_age_minutes=cfg.get("max_age_minutes", 120),
        )
        self.model_name = cfg.get("model", "vader").lower()
        self._vader = None
        self._finbert = None
        self._init_model()

    def _init_model(self):
        if self.model_name == "finbert":
            try:
                from transformers import pipeline  # type: ignore

                self._finbert = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                )
                log.info("FinBERT sentiment model loaded.")
                return
            except Exception as e:
                log.warning("FinBERT unavailable, falling back to VADER: %s", e)
                self.model_name = "vader"

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
        except ImportError:
            log.warning("vaderSentiment not installed; sentiment will return neutral.")

    def evaluate(self, symbol: str) -> Optional[Signal]:
        headlines = self.news.get_headlines(symbol)
        if not headlines:
            return Signal(
                symbol=symbol,
                source=SignalSource.SENTIMENT,
                score=0.0,
                confidence=0.0,
                metadata={"headline_count": 0},
            )
        scores: List[float] = []
        weights: List[float] = []
        for h in headlines:
            txt = f"{h.title}. {h.summary}".strip()
            if not txt:
                continue
            s = self._score_text(txt)
            scores.append(s)
            # Decay weight with age (linear, capped).
            w = max(0.1, 1.0 - h.age_minutes / self.cfg.get("max_age_minutes", 120))
            weights.append(w)
        if not scores:
            return Signal(symbol=symbol, source=SignalSource.SENTIMENT, score=0.0, confidence=0.0)
        weighted = float(np.average(scores, weights=weights))
        # Confidence: more headlines + tighter agreement => higher.
        n = len(scores)
        agreement = 1.0 - float(np.std(scores))
        confidence = float(np.clip(min(1.0, n / 10.0) * 0.5 + agreement * 0.5, 0.0, 1.0))
        return Signal(
            symbol=symbol,
            source=SignalSource.SENTIMENT,
            score=float(np.clip(weighted, -1.0, 1.0)),
            confidence=confidence,
            metadata={"headline_count": n, "model": self.model_name},
        )

    # ---------- internals ----------
    def _score_text(self, txt: str) -> float:
        if self._finbert is not None:
            try:
                r = self._finbert(txt[:512])[0]
                label = r["label"].lower()
                score = float(r["score"])
                if label == "positive":
                    return score
                if label == "negative":
                    return -score
                return 0.0
            except Exception as e:  # pragma: no cover
                log.warning("FinBERT scoring failed: %s", e)
        if self._vader is not None:
            return float(self._vader.polarity_scores(txt)["compound"])
        return 0.0
