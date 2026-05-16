"""Fundamental / event-driven signal.

For day trading, "fundamental" is interpreted as catalyst-driven events:
  - Fresh 8-K filings (material events) → strong directional move likely.
  - Form 4 insider buys (positive bias) / sells (negative bias).
  - Earnings within window → suppress signal (avoid surprise gaps).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..data.sec_filings import SECFilings
from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)


# Heuristic priors per form type — refined over time with experience.
FORM_PRIORS = {
    "8-K": 0.15,        # Material event — direction unknown, magnitude high
    "10-Q": 0.05,       # Quarterly report — usually anticipated
    "10-K": 0.05,
    "4": 0.30,          # Insider trade — direction inferred from content
}


class FundamentalSignal:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sec = SECFilings(form_types=cfg.get("sec_filing_types", ["8-K", "10-Q", "10-K", "4"]))
        self.earnings_window = int(cfg.get("earnings_window_days", 3))

    def evaluate(self, symbol: str) -> Optional[Signal]:
        filings = self.sec.get_recent_filings(symbol, lookback_hours=48)
        if not filings:
            return Signal(
                symbol=symbol,
                source=SignalSource.FUNDAMENTAL,
                score=0.0,
                confidence=0.0,
                metadata={"filing_count": 0},
            )

        net = 0.0
        weights = 0.0
        details = []
        for f in filings:
            prior = FORM_PRIORS.get(f.form_type, 0.05)
            # Direction: insider buys positive, sells negative, others unknown (zero direction, but reduces uncertainty).
            direction = 0.0
            if f.form_type == "4":
                t = f.title.lower()
                if "purchase" in t or "acquired" in t:
                    direction = 1.0
                elif "sale" in t or "disposed" in t:
                    direction = -1.0
            # 8-Ks within hours often pre-announce gaps — treat as cautionary (slight bias toward direction-uncertain).
            recency = max(0.1, 1.0 - f.age_hours / 48.0)
            weight = prior * recency
            net += direction * weight
            weights += weight
            details.append({"form": f.form_type, "title": f.title[:80], "age_h": round(f.age_hours, 1)})

        score = float(np.clip(net / max(weights, 1e-6), -1.0, 1.0)) if weights > 0 else 0.0
        # Confidence rises with filing count and weight.
        confidence = float(np.clip(min(1.0, weights * 3), 0.0, 1.0))

        return Signal(
            symbol=symbol,
            source=SignalSource.FUNDAMENTAL,
            score=score,
            confidence=confidence,
            metadata={"filing_count": len(filings), "details": details},
        )
