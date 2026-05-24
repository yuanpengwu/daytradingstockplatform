"""Opening Range Breakout (ORB) signal.

The first `orb_minutes` of the trading session form the Opening Range (OR).
Strategy:
  - Close above OR high + volume surge  → bullish breakout
  - Close below OR low  + volume surge  → bearish breakdown
  - Price inside the range              → neutral (wait)

Signal strength scales with how far price has extended beyond the range
relative to ATR.  A time-decay factor reduces the signal after the first
two hours so late-day noise is suppressed.
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


def _today_session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return only today's intraday bars (9:30 AM ET onward)."""
    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_ny = idx.tz_convert(NY)
    today = idx_ny[-1].date()
    market_open = pd.Timestamp(today, tz=NY).replace(hour=9, minute=30)
    return bars[idx_ny >= market_open]


def _minutes_since_open() -> float:
    now = pd.Timestamp.now(tz=NY)
    open_today = now.normalize().replace(hour=9, minute=30)
    return max(0.0, (now - open_today).total_seconds() / 60)


class ORBSignal:
    def __init__(self, cfg: dict):
        self.orb_minutes = int(cfg.get("orb_minutes", 15))
        self.vol_mult = float(cfg.get("volume_confirmation_mult", 1.3))
        self.decay_start_min = float(cfg.get("decay_start_minutes", 120))
        self.decay_end_min = float(cfg.get("decay_end_minutes", 240))

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Optional[Signal]:
        if bars is None or len(bars) < 10:
            return None

        session = _today_session_bars(bars)
        if session.empty:
            return self._neutral(symbol, "no session bars")

        # Need at least one complete ORB window before signalling.
        orb_bars = session.iloc[: max(1, self.orb_minutes // 5)]
        if len(orb_bars) < max(1, self.orb_minutes // 5):
            return self._neutral(symbol, "orb window not complete")

        orb_high = float(orb_bars["High"].max())
        orb_low = float(orb_bars["Low"].min())
        orb_range = orb_high - orb_low if orb_high > orb_low else 1e-6

        close = float(bars["Close"].iloc[-1])
        atr = self._atr(bars)
        normalizer = max(orb_range, atr, 1e-6)

        # Score: how far beyond the range, normalized.
        if close > orb_high:
            raw = float(np.tanh((close - orb_high) / normalizer * 3))
        elif close < orb_low:
            raw = -float(np.tanh((orb_low - close) / normalizer * 3))
        else:
            return self._neutral(symbol, "price inside opening range")

        # Volume confirmation.
        vol_avg = float(bars["Volume"].tail(20).mean()) or 1.0
        vol_ratio = float(bars["Volume"].iloc[-1]) / vol_avg
        vol_factor = float(np.clip(vol_ratio / self.vol_mult, 0.3, 1.5))

        # Time decay: full strength in first `decay_start_min`, fades to 0 by `decay_end_min`.
        elapsed = _minutes_since_open()
        if elapsed > self.decay_end_min:
            return self._neutral(symbol, "ORB signal expired (end of decay window)")
        decay = 1.0 if elapsed <= self.decay_start_min else (
            1.0 - (elapsed - self.decay_start_min) / (self.decay_end_min - self.decay_start_min)
        )

        score = float(np.clip(raw * vol_factor * decay, -1.0, 1.0))
        confidence = float(np.clip(0.3 + abs(score) * 0.5 + min(vol_ratio - 1, 0.5) * 0.2, 0.0, 1.0))

        return Signal(
            symbol=symbol,
            source=SignalSource.ORB,
            score=score,
            confidence=confidence,
            metadata={
                "orb_high": round(orb_high, 2),
                "orb_low": round(orb_low, 2),
                "orb_range": round(orb_range, 4),
                "vol_ratio": round(vol_ratio, 2),
                "decay": round(decay, 2),
                "elapsed_min": round(elapsed, 1),
            },
        )

    @staticmethod
    def _atr(bars: pd.DataFrame, period: int = 14) -> float:
        high = bars["High"].astype(float)
        low = bars["Low"].astype(float)
        close = bars["Close"].astype(float)
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]) or 1e-6

    @staticmethod
    def _neutral(symbol: str, reason: str) -> Signal:
        return Signal(
            symbol=symbol,
            source=SignalSource.ORB,
            score=0.0,
            confidence=0.0,
            metadata={"reason": reason},
        )
