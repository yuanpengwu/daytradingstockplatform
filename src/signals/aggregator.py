"""Combine sub-signals into a single weighted score per ticker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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

    @property
    def action(self) -> str:
        if self.score >= self.enter_long:
            return "BUY"
        if self.score <= self.enter_short:
            return "SELL"
        return "HOLD"


class SignalAggregator:
    def __init__(self, weights: dict, enter_long: float = 0.35, enter_short: float = -0.35, exit_thresh: float = 0.10):
        # Normalize weights so they sum to 1.
        total = sum(max(0, float(v)) for v in weights.values()) or 1.0
        self.weights = {k: max(0, float(v)) / total for k, v in weights.items()}
        self.enter_long = enter_long
        self.enter_short = enter_short
        self.exit_thresh = exit_thresh

    def aggregate(self, signals: List[Signal], market_multiplier: float = 1.0) -> Dict[str, AggregatedDecision]:
        by_sym: Dict[str, List[Signal]] = {}
        for s in signals:
            by_sym.setdefault(s.symbol, []).append(s)

        out: Dict[str, AggregatedDecision] = {}
        for sym, sigs in by_sym.items():
            comp = {}
            num = 0.0
            den = 0.0
            confs = []
            for s in sigs:
                w = self.weights.get(s.source.value, 0.0)
                # Effective weight is the configured weight times the signal's own confidence.
                eff = w * s.confidence
                contribution = eff * s.score
                comp[s.source.value] = contribution
                num += contribution
                den += eff
                confs.append(s.confidence)
            
            raw_score = float(num / den) if den > 0 else 0.0
            score = raw_score * market_multiplier
            
            confidence = float(np.mean(confs)) if confs else 0.0
            out[sym] = AggregatedDecision(
                symbol=sym,
                score=float(np.clip(score, -1.0, 1.0)),
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                components=comp,
                enter_long=self.enter_long,
                enter_short=self.enter_short,
            )
        return out
