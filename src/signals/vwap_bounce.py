"""VWAP Bounce signal.

Detects high-probability mean-reversion setups around the session VWAP:

  1. Price is within `proximity_pct` of VWAP (approaching).
  2. Volume was declining on the approach (exhaustion of the counter-trend move).
  3. The latest bar shows price moving *away* from VWAP with a volume surge
     (institutional buying/selling at VWAP acts as support/resistance).

Bullish bounce: price was below VWAP, pulled back toward it, now reclaims it upward.
Bearish bounce: price was above VWAP, pulled back toward it, now rejects downward.

A secondary "VWAP cross with volume" mode scores momentum when price cleanly
crosses VWAP with above-average volume — a trend continuation signal.
"""
from __future__ import annotations

from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)
NY = ZoneInfo("America/New_York")


def _session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Compute cumulative intraday VWAP using today's bars only."""
    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_ny = idx.tz_convert(NY)
    today = idx_ny[-1].date()
    market_open = pd.Timestamp(today, tz=NY).replace(hour=9, minute=30)
    session_mask = idx_ny >= market_open

    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    vol = bars["Volume"].replace(0, np.nan)

    # Within session: cumulative VWAP; outside: carry forward last known value.
    cum_vp = (tp * vol).where(session_mask, np.nan).fillna(0).cumsum()
    cum_v = vol.where(session_mask, np.nan).fillna(0).cumsum().replace(0, np.nan)
    vwap = (cum_vp / cum_v).ffill()
    return vwap


class VWAPBounceSignal:
    def __init__(self, cfg: dict):
        # Max distance from VWAP (as fraction of price) to consider a "bounce zone".
        self.proximity_pct = float(cfg.get("proximity_pct", 0.003))
        # Volume must drop at least this much on approach bars.
        self.vol_decline_thresh = float(cfg.get("vol_decline_thresh", 0.85))
        # Volume must surge at least this much on the bounce bar.
        self.vol_surge_thresh = float(cfg.get("vol_surge_thresh", 1.3))
        # Lookback bars to assess the approach.
        self.approach_bars = int(cfg.get("approach_bars", 3))

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Optional[Signal]:
        if bars is None or len(bars) < self.approach_bars + 5:
            return None

        vwap = _session_vwap(bars)
        if vwap.isna().all():
            return self._neutral(symbol, "no session VWAP")

        close = bars["Close"].astype(float)
        volume = bars["Volume"].astype(float)
        vwap_now = float(vwap.iloc[-1])
        close_now = float(close.iloc[-1])

        if vwap_now <= 0:
            return self._neutral(symbol, "invalid VWAP")

        deviation = (close_now - vwap_now) / vwap_now

        # ── Mode 1: VWAP Bounce ──────────────────────────────────────────────
        if abs(deviation) <= self.proximity_pct:
            score, confidence, mode = self._bounce_score(
                close, volume, vwap, deviation
            )
            if abs(score) > 0.05:
                return Signal(
                    symbol=symbol,
                    source=SignalSource.VWAP_BOUNCE,
                    score=score,
                    confidence=confidence,
                    metadata={
                        "mode": mode,
                        "vwap": round(vwap_now, 2),
                        "deviation_pct": round(deviation * 100, 3),
                    },
                )

        # ── Mode 2: VWAP Cross with volume ──────────────────────────────────
        cross_score, cross_conf = self._cross_score(close, volume, vwap)
        if abs(cross_score) > 0.05:
            return Signal(
                symbol=symbol,
                source=SignalSource.VWAP_BOUNCE,
                score=cross_score,
                confidence=cross_conf,
                metadata={
                    "mode": "vwap_cross",
                    "vwap": round(vwap_now, 2),
                    "deviation_pct": round(deviation * 100, 3),
                },
            )

        return self._neutral(symbol, "no VWAP setup")

    # ---------- private ----------

    def _bounce_score(
        self,
        close: pd.Series,
        volume: pd.Series,
        vwap: pd.Series,
        deviation: float,
    ) -> tuple[float, float, str]:
        n = self.approach_bars
        approach_vols = volume.iloc[-(n + 1) : -1]
        bounce_vol = float(volume.iloc[-1])
        avg_approach_vol = float(approach_vols.mean()) or 1.0

        # Check for volume exhaustion on approach (declining into VWAP).
        vol_declining = float(approach_vols.iloc[-1]) < avg_approach_vol * self.vol_decline_thresh
        # Check for volume surge on the bounce bar.
        vol_avg20 = float(volume.tail(20).mean()) or 1.0
        vol_surging = bounce_vol > vol_avg20 * self.vol_surge_thresh

        # Price direction on the bounce bar.
        price_change = float(close.iloc[-1] - close.iloc[-2])

        # Is price above or below VWAP and moving away from it?
        above_vwap = deviation > 0
        bouncing_bullish = (not above_vwap) and price_change > 0   # was below, now reclaiming
        bouncing_bearish = above_vwap and price_change < 0          # was above, now rejecting

        if not (bouncing_bullish or bouncing_bearish):
            return 0.0, 0.0, "no_bounce"

        direction = 1.0 if bouncing_bullish else -1.0
        # Base magnitude from how clean the setup is.
        strength = 0.5
        if vol_declining:
            strength += 0.25
        if vol_surging:
            strength += 0.25

        score = float(np.clip(direction * strength, -1.0, 1.0))
        confidence = float(np.clip(0.35 + strength * 0.4, 0.0, 1.0))
        return score, confidence, "vwap_bounce"

    def _cross_score(
        self,
        close: pd.Series,
        volume: pd.Series,
        vwap: pd.Series,
    ) -> tuple[float, float]:
        """Score a clean VWAP cross (price flips side with above-average volume)."""
        if len(close) < 3:
            return 0.0, 0.0

        prev_above = float(close.iloc[-2]) > float(vwap.iloc[-2])
        curr_above = float(close.iloc[-1]) > float(vwap.iloc[-1])

        if prev_above == curr_above:
            return 0.0, 0.0  # no cross

        vol_avg = float(volume.tail(20).mean()) or 1.0
        vol_ratio = float(volume.iloc[-1]) / vol_avg
        if vol_ratio < self.vol_surge_thresh:
            return 0.0, 0.0  # weak cross, ignore

        direction = 1.0 if curr_above else -1.0
        score = float(np.clip(direction * np.tanh(vol_ratio - 1), -1.0, 1.0))
        confidence = float(np.clip(0.4 + min(vol_ratio - 1, 1.0) * 0.3, 0.0, 1.0))
        return score, confidence

    @staticmethod
    def _neutral(symbol: str, reason: str) -> Signal:
        return Signal(
            symbol=symbol,
            source=SignalSource.VWAP_BOUNCE,
            score=0.0,
            confidence=0.0,
            metadata={"reason": reason},
        )
