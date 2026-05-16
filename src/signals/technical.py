"""Technical-indicator signal.

Combines RSI, MACD, Bollinger Bands, EMA cross, and VWAP into a single
[-1, +1] score per ticker.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .base import Signal, SignalSource

log = get_logger(__name__)


class TechnicalSignal:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ---------- public ----------
    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Optional[Signal]:
        if bars is None or bars.empty or len(bars) < 30:
            return None

        close = bars["Close"].astype(float)
        high = bars["High"].astype(float)
        low = bars["Low"].astype(float)
        volume = bars["Volume"].astype(float)

        rsi = self._rsi(close, self.cfg.get("rsi_period", 14))
        macd_hist = self._macd_hist(
            close,
            self.cfg.get("macd_fast", 12),
            self.cfg.get("macd_slow", 26),
            self.cfg.get("macd_signal", 9),
        )
        bb_pct = self._bbands_pct(
            close,
            self.cfg.get("bb_period", 20),
            self.cfg.get("bb_std", 2.0),
        )
        ema_cross = self._ema_cross(
            close,
            self.cfg.get("ema_short", 9),
            self.cfg.get("ema_long", 21),
        )
        vwap_pos = self._vwap_position(high, low, close, volume) if self.cfg.get("use_vwap", True) else 0.0
        atr = self._atr(high, low, close, self.cfg.get("atr_period", 14))

        last = {
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else 50.0,
            "macd_hist": float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0,
            "macd_hist_prev": float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
            "bb_pct": float(bb_pct.iloc[-1]) if not bb_pct.empty else 0.5,
            "ema_cross": float(ema_cross),
            "vwap_pos": float(vwap_pos),
            "atr": float(atr.iloc[-1]) if not atr.empty else 0.0,
        }

        # ----- scoring -----
        scores = []

        # RSI: oversold => bullish, overbought => bearish
        rsi_oversold = self.cfg.get("rsi_oversold", 30)
        rsi_overbought = self.cfg.get("rsi_overbought", 70)
        if last["rsi"] < rsi_oversold:
            scores.append((rsi_oversold - last["rsi"]) / rsi_oversold)
        elif last["rsi"] > rsi_overbought:
            scores.append(-(last["rsi"] - rsi_overbought) / (100 - rsi_overbought))
        else:
            # Neutral zone: small bias toward the side of 50
            scores.append((50 - last["rsi"]) / 100.0)

        # MACD histogram: positive & rising => bullish
        macd_change = last["macd_hist"] - last["macd_hist_prev"]
        macd_score = np.tanh(last["macd_hist"] * 5) * 0.5 + np.tanh(macd_change * 10) * 0.5
        scores.append(float(macd_score))

        # Bollinger %B: <0 below lower band (oversold), >1 above upper (overbought)
        if last["bb_pct"] < 0:
            scores.append(min(1.0, -last["bb_pct"] * 2))
        elif last["bb_pct"] > 1:
            scores.append(-min(1.0, (last["bb_pct"] - 1) * 2))
        else:
            scores.append((0.5 - last["bb_pct"]) * 0.4)

        # EMA cross: +1 if short > long, -1 otherwise, scaled by separation
        scores.append(last["ema_cross"])

        # VWAP position
        scores.append(last["vwap_pos"])

        score = float(np.clip(np.mean(scores), -1.0, 1.0))

        # Confidence rises with magnitude + cross-signal agreement.
        agreement = 1.0 - float(np.std(scores)) if scores else 0.0
        confidence = float(np.clip(0.4 + abs(score) * 0.4 + agreement * 0.2, 0.0, 1.0))

        return Signal(
            symbol=symbol,
            source=SignalSource.TECHNICAL,
            score=score,
            confidence=confidence,
            metadata=last,
        )

    # ---------- indicators ----------
    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50)

    @staticmethod
    def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        ef = close.ewm(span=fast, adjust=False).mean()
        es = close.ewm(span=slow, adjust=False).mean()
        macd = ef - es
        sig = macd.ewm(span=signal, adjust=False).mean()
        return (macd - sig).fillna(0)

    @staticmethod
    def _bbands_pct(close: pd.Series, period: int, std: float) -> pd.Series:
        m = close.rolling(period).mean()
        s = close.rolling(period).std()
        upper = m + std * s
        lower = m - std * s
        rng = (upper - lower).replace(0, np.nan)
        return ((close - lower) / rng).fillna(0.5)

    @staticmethod
    def _ema_cross(close: pd.Series, short: int, long: int) -> float:
        es = close.ewm(span=short, adjust=False).mean()
        el = close.ewm(span=long, adjust=False).mean()
        if len(es) == 0 or len(el) == 0:
            return 0.0
        diff = (es.iloc[-1] - el.iloc[-1]) / el.iloc[-1]
        return float(np.tanh(diff * 50))

    @staticmethod
    def _vwap_position(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series) -> float:
        tp = (high + low + close) / 3.0
        cum_vp = (tp * vol).cumsum()
        cum_v = vol.cumsum().replace(0, np.nan)
        vwap = (cum_vp / cum_v).ffill()
        if vwap.empty or vwap.iloc[-1] == 0:
            return 0.0
        diff = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
        # Above VWAP => bullish bias for momentum / day-trading
        return float(np.tanh(diff * 50))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean().fillna(0)
