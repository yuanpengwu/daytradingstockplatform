"""Market Regime Detector (two implementations).

──────────────────────────────────────────────────────────────────
RegimeDetector  (live-trading, VIX-based)
──────────────────────────────────────────────────────────────────
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

──────────────────────────────────────────────────────────────────
MarketRegimeDetector  (backtesting, ADX + vol-ratio based)
──────────────────────────────────────────────────────────────────
Classifies into:
  TRENDING — strong directional move  → use OLD config (let winners run)
  CHOPPY   — range-bound / sideways   → use BEST config (lock in partials)
  NEUTRAL  — ambiguous                → use BEST config (conservative default)

Algorithm:
  1. ADX (Wilder-smoothed, 14-day):
       ADX > 25 → +2 trending votes
       ADX < 20 → +2 choppy votes
  2. Realised-vol ratio (5d σ / 20d σ):
       ratio > 1.20 → +1 trending vote   (vol expanding = trending)
       ratio < 0.80 → +1 choppy vote     (vol contracting = range-bound)
  Decision: votes_trending ≥ 2 → TRENDING | votes_choppy ≥ 2 → CHOPPY
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
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


# ===========================================================================
# ADX-based regime detector  (used by backtester for strategy routing)
# ===========================================================================

class MarketRegime(Enum):
    TRENDING = "trending"
    CHOPPY   = "choppy"
    NEUTRAL  = "neutral"


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Wilder-smoothed ADX from a daily OHLC DataFrame.

    Columns required (case-insensitive): high, low, close.
    Returns a Series of ADX values (0–100).  First ``2*period`` rows are NaN.
    """
    cols = {c.lower(): c for c in df.columns}
    high  = df[cols["high"]].values.astype(float)
    low   = df[cols["low"]].values.astype(float)
    close = df[cols["close"]].values.astype(float)
    n = len(close)

    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    # Directional Movement
    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up   = high[i]  - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i]  = up
        elif down > up and down > 0:
            minus_dm[i] = down

    def _wilder(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if n < p:
            return out
        out[p - 1] = arr[:p].sum()
        for i in range(p, n):
            out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    atr14      = _wilder(tr, period)
    plus_dm14  = _wilder(plus_dm, period)
    minus_dm14 = _wilder(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di  = np.where(atr14 > 0, 100.0 * plus_dm14  / atr14, 0.0)
        minus_di = np.where(atr14 > 0, 100.0 * minus_dm14 / atr14, 0.0)
        di_sum   = plus_di + minus_di
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    adx_arr = np.full(n, np.nan)
    start = 2 * period - 2
    if n > start:
        adx_arr[start] = np.nanmean(dx[period - 1 : start + 1])
        for i in range(start + 1, n):
            if not np.isnan(adx_arr[i - 1]):
                adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    return pd.Series(adx_arr, index=df.index, name="adx")


def _daily_from_intraday(intraday: pd.DataFrame) -> pd.DataFrame:
    """Resample intraday bars (any interval) to daily OHLCV."""
    df = intraday.copy()
    df.index = pd.DatetimeIndex(df.index)
    cols = {c.lower(): c for c in df.columns}
    agg: dict = {}
    for field, func in [("open", "first"), ("high", "max"), ("low", "min"),
                        ("close", "last"), ("volume", "sum")]:
        if field in cols:
            agg[cols[field]] = func
    daily = df.resample("1D").agg(agg)
    close_col = cols.get("close", "close")
    if close_col in daily.columns:
        daily = daily.dropna(subset=[close_col])
    return daily


class MarketRegimeDetector:
    """Classify the market as TRENDING, CHOPPY, or NEUTRAL using ADX + vol ratio.

    Designed for use in the backtester where VIX data is unavailable.
    Feed it SPY bars (intraday or daily); it resamples to daily internally.

    Parameters
    ----------
    adx_period        : ADX lookback (Wilder smoothing).   Default 14.
    adx_trend_thresh  : ADX ≥ this → 2 trending votes.    Default 25.
    adx_choppy_thresh : ADX ≤ this → 2 choppy votes.      Default 20.
    vol_short         : Short realised-vol window (days).  Default 5.
    vol_long          : Long  realised-vol window (days).  Default 20.
    vol_high_thresh   : vol_ratio ≥ this → 1 trending vote. Default 1.20.
    vol_low_thresh    : vol_ratio ≤ this → 1 choppy vote.   Default 0.80.
    """

    def __init__(
        self,
        adx_period: int          = 14,
        adx_trend_thresh: float  = 25.0,
        adx_choppy_thresh: float = 20.0,
        vol_short: int           = 5,
        vol_long: int            = 20,
        vol_high_thresh: float   = 1.20,
        vol_low_thresh: float    = 0.80,
    ):
        self.adx_period         = adx_period
        self.adx_trend_thresh   = adx_trend_thresh
        self.adx_choppy_thresh  = adx_choppy_thresh
        self.vol_short          = vol_short
        self.vol_long           = vol_long
        self.vol_high_thresh    = vol_high_thresh
        self.vol_low_thresh     = vol_low_thresh

        self._last_regime:         MarketRegime  = MarketRegime.NEUTRAL
        self._last_adx:            Optional[float] = None
        self._last_vol_ratio:      Optional[float] = None
        self._last_votes_trending: int = 0
        self._last_votes_choppy:   int = 0

    # ------------------------------------------------------------------
    def detect(self, spy_bars: pd.DataFrame) -> MarketRegime:
        """Classify today's regime from SPY bars available up to *now*.

        Call once per trading day (or per backtest day) with the bars seen
        so far.  Returns NEUTRAL when there are insufficient bars.
        """
        if spy_bars is None or spy_bars.empty:
            log.warning("MarketRegimeDetector: SPY bars empty — NEUTRAL")
            self._last_regime = MarketRegime.NEUTRAL
            return self._last_regime

        df = spy_bars.copy()
        df.index = pd.DatetimeIndex(df.index)
        # Resample to daily if intraday
        if len(df.index.normalize().unique()) < len(df):
            df = _daily_from_intraday(df)

        min_bars = 2 * self.adx_period + self.vol_long
        if len(df) < min_bars:
            log.debug(
                "MarketRegimeDetector: %d daily bars < %d needed — NEUTRAL",
                len(df), min_bars,
            )
            self._last_regime = MarketRegime.NEUTRAL
            return self._last_regime

        votes_trending = 0
        votes_choppy   = 0

        # ── ADX vote ─────────────────────────────────────────────────
        adx_series = compute_adx(df, period=self.adx_period)
        valid_adx  = adx_series.dropna()
        adx_val    = float(valid_adx.iloc[-1]) if len(valid_adx) > 0 else float("nan")
        self._last_adx = adx_val

        if not np.isnan(adx_val):
            if adx_val >= self.adx_trend_thresh:
                votes_trending += 2
            elif adx_val <= self.adx_choppy_thresh:
                votes_choppy += 2

        # ── Vol-ratio vote ───────────────────────────────────────────
        cols  = {c.lower(): c for c in df.columns}
        close = df[cols.get("close", "close")].astype(float)
        log_ret = np.log(close / close.shift(1)).dropna()

        if len(log_ret) >= self.vol_long:
            vol_s = float(log_ret.iloc[-self.vol_short:].std())
            vol_l = float(log_ret.iloc[-self.vol_long:].std())
            vol_ratio = vol_s / vol_l if vol_l > 1e-9 else 1.0
            self._last_vol_ratio = vol_ratio

            if vol_ratio >= self.vol_high_thresh:
                votes_trending += 1
            elif vol_ratio <= self.vol_low_thresh:
                votes_choppy += 1
        else:
            self._last_vol_ratio = None

        self._last_votes_trending = votes_trending
        self._last_votes_choppy   = votes_choppy

        # ── Decision ─────────────────────────────────────────────────
        if votes_trending >= 2:
            regime = MarketRegime.TRENDING
        elif votes_choppy >= 2:
            regime = MarketRegime.CHOPPY
        else:
            regime = MarketRegime.NEUTRAL

        if regime != self._last_regime:
            log.info(
                "MarketRegime: %s → %s  "
                "(ADX=%.1f, vol_ratio=%s, votes trend=%d choppy=%d)",
                self._last_regime.value, regime.value,
                adx_val if not np.isnan(adx_val) else -1,
                f"{self._last_vol_ratio:.2f}" if self._last_vol_ratio else "n/a",
                votes_trending, votes_choppy,
            )
        self._last_regime = regime
        return regime

    @property
    def status(self) -> dict:
        return {
            "regime":         self._last_regime.value,
            "adx":            self._last_adx,
            "vol_ratio":      self._last_vol_ratio,
            "votes_trending": self._last_votes_trending,
            "votes_choppy":   self._last_votes_choppy,
        }
