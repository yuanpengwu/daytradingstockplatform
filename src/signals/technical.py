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

        price_level = float(close.iloc[-1]) if float(close.iloc[-1]) > 0 else 1.0
        atr_val = float(atr.iloc[-1]) if not atr.empty else 0.0
        atr_pct = atr_val / price_level  # ATR as fraction of price

        # Volume surge: compare latest bar to 20-bar rolling average
        vol_avg = float(volume.tail(20).mean()) if len(volume) >= 2 else 1.0
        vol_ratio = float(volume.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
        price_chg = (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) if len(close) > 1 else 0.0

        # Short-term momentum: return over last 6 bars (~30 min on 5m)
        mom_bars = min(6, len(close) - 1)
        momentum_return = (
            (float(close.iloc[-1]) - float(close.iloc[-mom_bars])) / float(close.iloc[-mom_bars])
            if mom_bars > 0 and float(close.iloc[-mom_bars]) > 0 else 0.0
        )

        last = {
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else 50.0,
            "macd_hist": float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0,
            "macd_hist_prev": float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
            "bb_pct": float(bb_pct.iloc[-1]) if not bb_pct.empty else 0.5,
            "ema_cross": float(ema_cross),
            "vwap_pos": float(vwap_pos),
            "atr": atr_val,
            "atr_pct": round(atr_pct, 5),
            "vol_ratio": round(vol_ratio, 2),
            "momentum_6b": round(momentum_return, 6),
        }

        # ----- scoring -----
        # Seven sub-scores, each in [-1, +1].  Averaged to produce final score.
        #
        # Calibration targets (5-min bars on $100-$500 stocks):
        #   Strong entry signal  → score ~0.40-0.70
        #   Moderate signal      → score ~0.20-0.40
        #   Noisy / sideways     → score <0.15  (below default threshold)
        scores = []

        # 1. RSI — extreme levels fire strongly; neutral zone gives a mild
        #    momentum-following nudge (RSI>50 → slight bull, RSI<50 → slight bear).
        #    Old formula used (50-rsi)/100 which *opposed* momentum in neutral
        #    zone — that suppressed trend-following signals.
        rsi_oversold  = self.cfg.get("rsi_oversold",  30)
        rsi_overbought = self.cfg.get("rsi_overbought", 70)
        if last["rsi"] < rsi_oversold:
            scores.append((rsi_oversold - last["rsi"]) / rsi_oversold)
        elif last["rsi"] > rsi_overbought:
            scores.append(-(last["rsi"] - rsi_overbought) / (100 - rsi_overbought))
        else:
            # Mild trend-following in neutral zone: RSI 60 → +0.06, RSI 40 → −0.06
            scores.append((last["rsi"] - 50) / 50.0 * 0.3)

        # 2. MACD histogram — normalised by price level.
        #    Multiplier 2500/5000 (was 500/1000) gives meaningful scores for
        #    typical intraday MACD magnitudes on $100-$500 stocks:
        #      hist=0.05 on $300 stock → tanh(0.05/300*2500) = tanh(0.42) = 0.40
        macd_change = last["macd_hist"] - last["macd_hist_prev"]
        macd_norm = last["macd_hist"] / price_level
        macd_change_norm = macd_change / price_level
        macd_score = np.tanh(macd_norm * 2500) * 0.5 + np.tanh(macd_change_norm * 5000) * 0.5
        scores.append(float(macd_score))

        # 3. Bollinger %B — mean-reversion signal.
        if last["bb_pct"] < 0:
            scores.append(min(1.0, -last["bb_pct"] * 2))
        elif last["bb_pct"] > 1:
            scores.append(-min(1.0, (last["bb_pct"] - 1) * 2))
        else:
            scores.append((0.5 - last["bb_pct"]) * 0.4)

        # 4. EMA cross — multiplier 250 (was 50); diff=0.1% → tanh(0.5)=0.46.
        scores.append(last["ema_cross"])

        # 5. VWAP position — multiplier 250 (was 50); same calibration as EMA.
        scores.append(last["vwap_pos"])

        # 6. Volume surge in direction of price move.
        vol_direction = float(np.sign(price_chg)) if abs(price_chg) > 0 else 0.0
        vol_score = float(np.tanh(vol_ratio - 1.0)) * vol_direction
        scores.append(vol_score)

        # 7. Short-term price momentum (6-bar return, ~30 min).
        #    0.1% move → tanh(0.001*300)=0.29;  0.5% → tanh(0.005*300)=0.91
        #    Captures slow intraday trends invisible to oscillators.
        momentum_score = float(np.tanh(momentum_return * 300))
        scores.append(momentum_score)

        score = float(np.clip(np.mean(scores), -1.0, 1.0))

        # Confidence: cross-signal agreement + volume surge boost.
        # Baseline raised to 0.35 since we now have 7 well-calibrated signals.
        agreement = max(0.0, 1.0 - float(np.std(scores))) if scores else 0.0
        vol_boost = min(0.08, (vol_ratio - 1.5) * 0.05) if vol_ratio > 1.5 else 0.0
        confidence = float(np.clip(0.35 + abs(score) * 0.40 + agreement * 0.15 + vol_boost, 0.0, 1.0))

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
        return float(np.tanh(diff * 250))

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
        return float(np.tanh(diff * 250))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean().fillna(0)
