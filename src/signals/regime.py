"""Market Regime Detector.

Classifies the current broad-market environment into one of four regimes:

  trending_bull    — VIX < 20, SPY above SMA20, positive momentum
                     → full position sizing, default signal weights
  trending_bear    — SPY below SMA20, negative momentum
                     → halve position size, favour mean-reversion signals
  choppy           — mixed signals or SPY straddling SMA20
                     → 70 % position size, suppress ORB/momentum
  high_volatility  — VIX > 25 (elevated fear / gap risk)
                     → halve position size, wider stops implicit

Each regime ships with:
  position_size_mult  — multiplied into RiskManager's max_position_pct
  weight_overrides    — passed to SignalAggregator.aggregate() to replace
                        individual signal weights for this cycle
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regime → behavioural adjustments
# ---------------------------------------------------------------------------
REGIME_ADJUSTMENTS: Dict[str, Dict[str, Any]] = {
    "trending_bull": {
        "position_size_mult": 1.0,
        # No weight overrides — use config defaults
        "weight_overrides": {},
    },
    "trending_bear": {
        "position_size_mult": 0.5,
        "weight_overrides": {
            # ORB breakouts fail in down-trending markets
            "orb":         0.05,
            # Reduce pure momentum weight
            "technical":   0.20,
            # VWAP mean-reversion is more reliable in bear/sideways
            "vwap_bounce": 0.30,
            # Keep sentiment/fundamental/ML roughly same
            "sentiment":   0.15,
            "fundamental": 0.15,
            "ml":          0.15,
        },
    },
    "choppy": {
        "position_size_mult": 0.7,
        "weight_overrides": {
            # ORB ranges break down in chop — many false breakouts
            "orb":         0.05,
            # Momentum signals are noisy in sideways markets
            "technical":   0.15,
            # VWAP bounce is the highest-quality setup in range days
            "vwap_bounce": 0.35,
            "sentiment":   0.15,
            "fundamental": 0.15,
            "ml":          0.15,
        },
    },
    "high_volatility": {
        "position_size_mult": 0.5,
        "weight_overrides": {
            "orb":         0.10,
            "technical":   0.20,
            "vwap_bounce": 0.25,
            "sentiment":   0.15,
            "fundamental": 0.15,
            "ml":          0.15,
        },
    },
}


class RegimeDetector:
    """Detects the current market regime from SPY bars + VIX.

    Results are cached for CACHE_MINUTES so the engine never calls yfinance
    more than once per quarter-hour for the VIX fetch.
    """

    CACHE_MINUTES: int = 15

    def __init__(self) -> None:
        self._last_regime: Optional[str] = None
        self._last_meta: Dict[str, Any] = {}
        self._last_computed: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, spy_bars: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Return (regime_name, metadata_dict).

        Reads a cached result if it is less than CACHE_MINUTES old.
        """
        now = datetime.now()
        if (
            self._last_regime is not None
            and self._last_computed is not None
            and (now - self._last_computed).total_seconds() < self.CACHE_MINUTES * 60
        ):
            return self._last_regime, self._last_meta

        regime, meta = self._compute(spy_bars)
        self._last_regime = regime
        self._last_meta = meta
        self._last_computed = now

        log.info(
            "Market regime → %s | VIX=%.1f SPY_vs_SMA20=%+.2f%% mom5=%+.2f%%",
            regime.upper(),
            meta.get("vix", 0.0),
            meta.get("spy_vs_sma20_pct", 0.0) * 100,
            meta.get("spy_momentum5", 0.0) * 100,
        )
        return regime, meta

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _compute(self, spy_bars: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {}

        if spy_bars is None or len(spy_bars) < 20:
            log.debug("RegimeDetector: insufficient SPY bars, defaulting to 'choppy'.")
            return "choppy", meta

        close = spy_bars["Close"].astype(float)
        current = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])

        spy_vs_sma20 = (current - sma20) / sma20 if sma20 > 0 else 0.0
        momentum5 = (
            (current / float(close.iloc[-6])) - 1.0
            if len(close) > 5 and float(close.iloc[-6]) > 0
            else 0.0
        )

        meta["spy_vs_sma20_pct"] = round(spy_vs_sma20, 5)
        meta["spy_momentum5"] = round(momentum5, 5)
        meta["spy_price"] = round(current, 2)
        meta["spy_sma20"] = round(sma20, 2)

        vix = self._fetch_vix()
        meta["vix"] = round(vix, 1)

        # ── Classification ─────────────────────────────────────────────
        # Thresholds are intentionally generous — we only want to catch
        # clearly trending or clearly high-fear environments.
        if vix > 25:
            regime = "high_volatility"
        elif spy_vs_sma20 > 0.002 and momentum5 > -0.001:
            regime = "trending_bull"
        elif spy_vs_sma20 < -0.002 or momentum5 < -0.004:
            regime = "trending_bear"
        else:
            regime = "choppy"

        return regime, meta

    @staticmethod
    def _fetch_vix() -> float:
        """Fetch the latest VIX close from yfinance. Returns 20.0 on failure."""
        try:
            import yfinance as yf  # lazy import — not needed every cycle

            hist = yf.Ticker("^VIX").history(period="2d", interval="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as exc:
            log.debug("VIX fetch failed (%s) — defaulting to 20.0", exc)
        return 20.0
