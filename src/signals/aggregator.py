"""Combine sub-signals into a single weighted score per ticker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)


@dataclass
class AggregatedDecision:
    symbol: str
    score: float                  # in [-1, +1]
    confidence: float             # in [0, 1]
    components: Dict[str, float]  # per-source contribution
    enter_long: float = 0.35      # thresholds carried from config
    enter_short: float = -0.35
    min_confidence: float = 0.25  # minimum confidence required to act

    @property
    def action(self) -> str:
        if self.confidence < self.min_confidence:
            return "HOLD"
        if self.score >= self.enter_long:
            return "BUY"
        if self.score <= self.enter_short:
            return "SELL"
        return "HOLD"


class SignalAggregator:
    def __init__(
        self,
        weights: dict,
        enter_long: float = 0.35,
        enter_short: float = -0.35,
        exit_thresh: float = 0.10,
        min_confidence: float = 0.25,
    ):
        # Normalize weights so they sum to 1.
        total = sum(max(0, float(v)) for v in weights.values()) or 1.0
        self.weights = {k: max(0, float(v)) / total for k, v in weights.items()}
        self.enter_long = enter_long
        self.enter_short = enter_short
        self.exit_thresh = exit_thresh
        self.min_confidence = min_confidence

    def aggregate(
        self,
        signals: List[Signal],
        market_multiplier: float = 1.0,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, AggregatedDecision]:
        """Aggregate signals into per-symbol decisions.

        Args:
            signals:          All Signal objects from this cycle.
            market_multiplier: Scalar applied to the raw score (MacroSignal).
            weight_overrides: Optional dict of source_name → new_weight that
                              replaces individual weights for this cycle only
                              (used by regime detection).  Will be renormalised.
        """
        # Build effective weights for this cycle.
        if weight_overrides:
            merged = {**self.weights, **{k: max(0.0, float(v)) for k, v in weight_overrides.items()}}
            total_w = sum(merged.values()) or 1.0
            effective_weights = {k: v / total_w for k, v in merged.items()}
        else:
            effective_weights = self.weights

        by_sym: Dict[str, List[Signal]] = {}
        for s in signals:
            by_sym.setdefault(s.symbol, []).append(s)

        # Sum of all configured weights (already normalised to 1.0 in __init__).
        total_weight = sum(effective_weights.values()) or 1.0

        out: Dict[str, AggregatedDecision] = {}
        for sym, sigs in by_sym.items():
            comp = {}
            num = 0.0
            den = 0.0          # sum of (w * confidence) for each signal
            for s in sigs:
                w = effective_weights.get(s.source.value, 0.0)
                # Effective weight is the configured weight times the signal's own confidence.
                # Neutral / inactive signals (confidence=0) contribute nothing.
                eff = w * s.confidence
                contribution = eff * s.score
                comp[s.source.value] = contribution
                num += contribution
                den += eff

            raw_score = float(num / den) if den > 0 else 0.0
            score = raw_score * market_multiplier

            # Confidence = fraction of the total weight budget that was *opinionated*.
            # Using simple np.mean(confidences) was wrong: it averaged in all the zero-
            # confidence neutral signals (no news, no ORB setup, etc.) and systematically
            # suppressed the aggregate below the min_confidence gate.
            # den/total_weight reads: "X% of all configured signal weight was active and
            # confident", which is the right question to ask before placing a trade.
            confidence = float(np.clip(den / total_weight, 0.0, 1.0))

            out[sym] = AggregatedDecision(
                symbol=sym,
                score=float(np.clip(score, -1.0, 1.0)),
                confidence=confidence,
                components=comp,
                enter_long=self.enter_long,
                enter_short=self.enter_short,
                min_confidence=self.min_confidence,
            )
        return out
