"""Fundamental / event-driven signal.

For day trading, "fundamental" is interpreted as catalyst-driven events:
  - Fresh 8-K filings (material events) → direction inferred from title keywords.
  - Form 4 insider buys (positive bias) / sells (negative bias), amplified by count.
  - Earnings within window → suppress signal (avoid surprise gaps).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.sec_filings import SECFilings
from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)

# Heuristic priors per form type.
FORM_PRIORS = {
    "8-K": 0.30,   # Material event — now directionally scored via keywords
    "10-Q": 0.05,  # Quarterly report — usually anticipated
    "10-K": 0.05,
    "4": 0.30,     # Insider trade — direction inferred from content
}

# 8-K title keywords that imply a directional move.
_8K_BULLISH = frozenset([
    "merger", "acquisition", "acquires", "buyout", "fda approval", "approved", "cleared",
    "partnership", "collaboration", "license agreement", "stock repurchase", "buyback",
    "dividend increase", "raises guidance", "raised guidance", "exceeds expectations",
    "beats estimates", "record revenue", "positive", "breakthrough", "award", "contract",
])
_8K_BEARISH = frozenset([
    "sec investigation", "sec subpoena", "lawsuit", "class action", "restatement",
    "guidance lower", "lowers guidance", "reduces guidance", "misses estimates",
    "below expectations", "recall", "data breach", "cybersecurity incident",
    "regulatory action", "fined", "penalty", "bankruptcy", "going concern",
    "resignation", "terminated", "fraud",
])


def _8k_direction(title: str) -> float:
    t = title.lower()
    bull = sum(1 for kw in _8K_BULLISH if kw in t)
    bear = sum(1 for kw in _8K_BEARISH if kw in t)
    if bull > bear:
        return min(1.0, bull * 0.4)
    if bear > bull:
        return -min(1.0, bear * 0.4)
    return 0.0


def _form4_direction(title: str) -> float:
    t = title.lower()
    if any(kw in t for kw in ("purchase", "acquired", "acquisition")):
        return 1.0
    if any(kw in t for kw in ("sale", "sold", "disposed", "disposition")):
        return -1.0
    return 0.0


class FundamentalSignal:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sec = SECFilings(form_types=cfg.get("sec_filing_types", ["8-K", "10-Q", "10-K", "4"]))
        self.earnings_window_h = int(cfg.get("earnings_window_days", 3)) * 24
        self.lookback_hours = int(cfg.get("lookback_hours", 48))

    def evaluate(self, symbol: str) -> Optional[Signal]:
        filings = self.sec.get_recent_filings(symbol, lookback_hours=self.lookback_hours)
        if not filings:
            return Signal(
                symbol=symbol,
                source=SignalSource.FUNDAMENTAL,
                score=0.0,
                confidence=0.0,
                metadata={"filing_count": 0},
            )

        # Earnings suppression: if a 10-Q/10-K dropped recently the stock just had an
        # earnings event and is prone to unpredictable gaps — return flat signal.
        for f in filings:
            if f.form_type in ("10-Q", "10-K") and f.age_hours < self.earnings_window_h:
                log.info("%s: recent %s filing (%.1fh ago) — suppressing fundamental signal.", symbol, f.form_type, f.age_hours)
                return Signal(
                    symbol=symbol,
                    source=SignalSource.FUNDAMENTAL,
                    score=0.0,
                    confidence=0.1,
                    metadata={"reason": "earnings_suppression", "form": f.form_type},
                )

        net = 0.0
        weights = 0.0
        details = []

        # Cluster Form 4s: multiple insiders moving the same direction amplifies confidence.
        form4_directions = []

        for f in filings:
            prior = FORM_PRIORS.get(f.form_type, 0.05)
            recency = max(0.1, 1.0 - f.age_hours / self.lookback_hours)

            if f.form_type == "4":
                direction = _form4_direction(f.title)
                form4_directions.append(direction)
                weight = prior * recency
            elif f.form_type == "8-K":
                direction = _8k_direction(f.title)
                weight = prior * recency
            else:
                direction = 0.0
                weight = prior * recency

            net += direction * weight
            weights += weight
            details.append({"form": f.form_type, "title": f.title[:80], "age_h": round(f.age_hours, 1), "dir": round(direction, 2)})

        score = float(np.clip(net / max(weights, 1e-6), -1.0, 1.0)) if weights > 0 else 0.0

        # Confidence: base on total weight, amplified when multiple insiders agree.
        insider_agreement_boost = 0.0
        if len(form4_directions) >= 2:
            same_dir = sum(1 for d in form4_directions if d == form4_directions[0] and d != 0)
            if same_dir == len(form4_directions) and same_dir >= 2:
                insider_agreement_boost = min(0.15, same_dir * 0.05)

        confidence = float(np.clip(weights * 2.5 + insider_agreement_boost, 0.0, 1.0))

        return Signal(
            symbol=symbol,
            source=SignalSource.FUNDAMENTAL,
            score=score,
            confidence=confidence,
            metadata={"filing_count": len(filings), "details": details},
        )
