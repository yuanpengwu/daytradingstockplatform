"""Combine sub-signals into a single weighted score per ticker.

Dynamic threshold scaling
--------------------------
When signal sources are offline (e.g. Gemini API exhausted → ML = 0, EDGAR
timeouts → fundamental = 0), the confidence budget shrinks and the bot would
never reach the configured enter_long_threshold or min_confidence.

To compensate, SignalAggregator tracks which sources have returned
zero-confidence signals for every ticker across N consecutive cycles
(``dead_signal_cycles`` parameter, default 3).  Once a source is declared
dead, both entry thresholds are scaled down proportionally:

    ratio                 = active_weight / total_configured_weight
    effective_enter_long  = base_enter_long  × ratio
    effective_min_conf    = base_min_confidence × ratio

Example: ML (15 %) + fundamental (10 %) both dead → ratio = 0.75
    score threshold:  0.50 × 0.75 = 0.375
    conf  threshold:  0.55 × 0.75 = 0.4125

When sources recover the thresholds are restored to their configured values
automatically and a single INFO log is emitted.

A floor of _RATIO_FLOOR (0.50) prevents the scaling from becoming
dangerously permissive if many sources are simultaneously down.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np

from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)

# Minimum scaling ratio — thresholds are never reduced below 50 % of their
# configured values even if the majority of signal sources go offline.
_RATIO_FLOOR: float = 0.50


@dataclass
class AggregatedDecision:
    symbol: str
    score: float                  # in [-1, +1]
    confidence: float             # in [0, 1]
    components: Dict[str, float]  # per-source contribution
    enter_long: float = 0.35      # effective threshold (may be scaled down)
    enter_short: float = -0.35    # effective threshold (may be scaled down)
    min_confidence: float = 0.25  # effective threshold (may be scaled down)
    agreement_ok: bool = True     # ML+FinRL agreement gate (False ⇒ force HOLD)

    @property
    def action(self) -> str:
        # ML+FinRL agreement gate — when both learned models are opinionated
        # but disagree on direction, the setup is too conflicted to trade.
        if not self.agreement_ok:
            return "HOLD"
        # Confidence gate comes first — an uncertain signal should never
        # trigger an order regardless of its magnitude.
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
        dead_signal_cycles: int = 3,
        require_ml_finrl_agreement: bool = False,
    ):
        # Normalize weights so they sum to 1.
        total = sum(max(0, float(v)) for v in weights.values()) or 1.0
        self.weights = {k: max(0, float(v)) / total for k, v in weights.items()}

        # Base (unscaled) thresholds from config — never mutated after init.
        self._base_enter_long: float     = enter_long
        self._base_enter_short: float    = enter_short
        self._base_min_confidence: float = min_confidence

        self.exit_thresh: float = exit_thresh

        # When True, an entry is suppressed (forced HOLD) if the ML and FinRL
        # models are BOTH opinionated yet point in opposite directions.  Two
        # independent learned models disagreeing is a strong "stay out" signal.
        self.require_ml_finrl_agreement: bool = require_ml_finrl_agreement

        # ── Dead-signal detection ──────────────────────────────────────────
        # A source is "dead" when it returns confidence = 0 for EVERY ticker
        # across _dead_signal_cycles consecutive cycles.  This catches silent
        # API failures (Gemini 429, EDGAR read-timeout, …) that zero-out an
        # entire signal column without raising an exception.
        #
        # Detection uses the BASE config weights so that regime weight
        # overrides don't amplify or dampen the scaling effect.
        self._dead_signal_cycles: int         = max(1, int(dead_signal_cycles))
        self._silent_cycles: Dict[str, int]   = {}          # source → silent-cycle count
        self._last_dead_sources: FrozenSet[str] = frozenset()  # for change-detection logging
        self._last_ratio: float               = 1.0

    # ------------------------------------------------------------------
    # Public read-only state (status page / external logging)
    # ------------------------------------------------------------------

    @property
    def dead_sources(self) -> FrozenSet[str]:
        """Sources currently considered dead (zero-confidence for N+ cycles)."""
        return frozenset(
            s for s, n in self._silent_cycles.items()
            if n >= self._dead_signal_cycles
        )

    @property
    def threshold_ratio(self) -> float:
        """Active-weight ratio used this cycle (1.0 = all sources alive, no scaling)."""
        return self._last_ratio

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        signals: List[Signal],
        market_multiplier: float = 1.0,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, AggregatedDecision]:
        """Aggregate signals into per-symbol decisions.

        Args:
            signals:           All Signal objects collected this cycle.
            market_multiplier: Scalar applied to the raw score (MacroSignal).
            weight_overrides:  Optional source_name → new_weight dict used by
                               regime detection. Will be renormalised internally.
        """
        # ── Build effective weights (may include regime overrides) ─────────
        if weight_overrides:
            merged  = {**self.weights, **{k: max(0.0, float(v)) for k, v in weight_overrides.items()}}
            total_w = sum(merged.values()) or 1.0
            effective_weights = {k: v / total_w for k, v in merged.items()}
        else:
            effective_weights = self.weights

        # ── Step 1: detect which sources were active this cycle ───────────
        # A source is "active" if at least one ticker returned confidence > 0
        # (meaning the signal engine had real data and formed an opinion).
        active_sources: Set[str] = {
            s.source.value for s in signals if s.confidence > 1e-9
        }

        # Update silence counters for every source that has a configured weight.
        for source in list(self.weights):
            if source in active_sources:
                self._silent_cycles[source] = 0            # back online — reset counter
            else:
                self._silent_cycles[source] = self._silent_cycles.get(source, 0) + 1

        dead = self.dead_sources   # frozenset snapshot for this cycle

        # ── Step 2: compute threshold scaling ratio ───────────────────────
        # Use BASE config weights (not regime-adjusted) so that regime shifts
        # don't change how aggressively we scale for dead signals.
        #
        # ratio = (sum of weights for alive sources) / (total configured weight)
        active_weight_base = sum(w for s, w in self.weights.items() if s not in dead)
        total_weight_base  = sum(self.weights.values()) or 1.0
        raw_ratio = active_weight_base / total_weight_base if total_weight_base > 0 else 1.0
        ratio = float(np.clip(raw_ratio, _RATIO_FLOOR, 1.0))
        self._last_ratio = ratio

        # ── Step 3: derive effective entry thresholds ─────────────────────
        if dead and ratio < 1.0 - 1e-6:
            eff_enter_long  = self._base_enter_long  * ratio
            eff_enter_short = self._base_enter_short * ratio   # stays negative
            eff_min_conf    = self._base_min_confidence * ratio

            # Log only when the set of dead sources changes to avoid every-cycle spam.
            # Use DEBUG in backtest contexts where offline sources are expected.
            if dead != self._last_dead_sources:
                log.debug(
                    "Scaled thresholds active: score=%.3f (was %.3f), "
                    "confidence=%.3f (was %.3f) — dead signals: %s "
                    "(active weight %.0f%%)",
                    eff_enter_long,  self._base_enter_long,
                    eff_min_conf,    self._base_min_confidence,
                    sorted(dead),
                    ratio * 100,
                )
                self._last_dead_sources = dead
        else:
            eff_enter_long  = self._base_enter_long
            eff_enter_short = self._base_enter_short
            eff_min_conf    = self._base_min_confidence

            if self._last_dead_sources:
                # Signals recovered — emit a single recovery log then clear.
                log.info(
                    "Dead-signal scaling lifted — all sources active. "
                    "Thresholds restored: score=%.3f, confidence=%.3f",
                    self._base_enter_long, self._base_min_confidence,
                )
                self._last_dead_sources = frozenset()

        # ── Step 4: aggregate per-symbol ──────────────────────────────────
        by_sym: Dict[str, List[Signal]] = {}
        for s in signals:
            by_sym.setdefault(s.symbol, []).append(s)

        # Sum of all effective weights (already normalised to 1.0 in most cases).
        total_weight_denom = sum(effective_weights.values()) or 1.0

        out: Dict[str, AggregatedDecision] = {}
        for sym, sigs in by_sym.items():
            comp: Dict[str, float] = {}
            num = 0.0
            den = 0.0   # Σ(weight × confidence) across active signals

            for s in sigs:
                w   = effective_weights.get(s.source.value, 0.0)
                eff = w * s.confidence          # zero-confidence signals contribute nothing
                contribution = eff * s.score
                comp[s.source.value] = contribution
                num += contribution
                den += eff

            raw_score  = float(num / den) if den > 0 else 0.0

            # Apply macro multiplier sign-aware:
            #   Long signals  (score ≥ 0): multiply — bull boosts longs, bear dampens.
            #   Short signals (score < 0): divide  — bear amplifies shorts, bull dampens.
            # Without this, a bearish multiplier (0.60) silently suppresses short signals
            # in exactly the conditions where shorts are most valuable.
            if raw_score >= 0:
                score = raw_score * market_multiplier
            else:
                score = raw_score / max(market_multiplier, 0.10)  # avoid div-by-zero

            # Confidence = (weight coverage) × (cross-source agreement factor).
            #
            # Weight coverage: what fraction of the total weight budget was active
            # and opinionated this cycle.  Higher = more signal breadth.
            #
            # Agreement factor: reward when multiple independent sources agree in
            # direction; penalise when they disagree.  Two sources both saying BUY
            # is far more reliable than one saying BUY and one saying SELL.
            #   • All agreeing sources (or only 1 source active) → factor = 1.0
            #   • Majority agree                                  → factor = 0.85
            #   • Split / minority agree                          → factor = 0.65
            #
            # This prevents a single strong sub-signal from masking opposing noise
            # from other sources and still reaching min_confidence.
            weight_coverage = den / total_weight_denom if total_weight_denom > 0 else 0.0

            # Gather sources that have a meaningful weight (≥5%) and actual opinion.
            opinionated = [
                (src, c_val)
                for src, c_val in comp.items()
                if effective_weights.get(src, 0.0) >= 0.05 and abs(c_val) > 1e-4
            ]
            if len(opinionated) >= 2:
                signal_dir = float(np.sign(raw_score)) if raw_score != 0 else 0.0
                n_agree = sum(1 for _, v in opinionated if float(np.sign(v)) == signal_dir)
                agree_ratio = n_agree / len(opinionated)
                if agree_ratio >= 1.0:
                    agree_factor = 1.00   # unanimous
                elif agree_ratio >= 0.67:
                    agree_factor = 0.85   # majority
                else:
                    agree_factor = 0.65   # split / minority — high-noise setup
            else:
                # Single source — apply a mild discount (less information breadth).
                agree_factor = 0.90

            confidence = float(np.clip(weight_coverage * agree_factor, 0.0, 1.0))

            # ── ML + FinRL agreement gate ─────────────────────────────────
            # Find the raw (signed) score of each learned model for this symbol.
            # We read from the signal objects directly (not weighted comp) so a
            # tiny configured weight can't hide a real disagreement.
            agreement_ok = True
            if self.require_ml_finrl_agreement:
                ml_s = next((s for s in sigs
                             if s.source.value == "ml" and s.confidence > 1e-9), None)
                rl_s = next((s for s in sigs
                             if s.source.value == "finrl" and s.confidence > 1e-9), None)
                if ml_s is not None and rl_s is not None:
                    ml_dir = float(np.sign(ml_s.score))
                    rl_dir = float(np.sign(rl_s.score))
                    if ml_dir != 0.0 and rl_dir != 0.0 and ml_dir != rl_dir:
                        agreement_ok = False

            out[sym] = AggregatedDecision(
                symbol=sym,
                score=float(np.clip(score, -1.0, 1.0)),
                confidence=confidence,
                components=comp,
                enter_long=eff_enter_long,
                enter_short=eff_enter_short,
                min_confidence=eff_min_conf,
                agreement_ok=agreement_ok,
            )
        return out
