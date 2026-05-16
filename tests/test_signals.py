"""Smoke tests for the signal stack — run with `pytest -q`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.aggregator import SignalAggregator
from src.signals.base import Signal, SignalSource
from src.signals.technical import TechnicalSignal


def _fake_bars(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.002, n)
    close = 100 * np.exp(rets.cumsum())
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.integers(1_000, 50_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_technical_signal_runs():
    sig = TechnicalSignal({
        "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_period": 20, "bb_std": 2.0,
        "ema_short": 9, "ema_long": 21,
        "use_vwap": True, "atr_period": 14,
    })
    bars = _fake_bars()
    s = sig.evaluate("TEST", bars)
    assert s is not None
    assert -1.0 <= s.score <= 1.0
    assert 0.0 <= s.confidence <= 1.0
    assert s.source == SignalSource.TECHNICAL


def test_aggregator_weighted():
    agg = SignalAggregator(weights={"technical": 0.5, "sentiment": 0.5})
    sigs = [
        Signal("AAPL", SignalSource.TECHNICAL, score=0.6, confidence=0.8),
        Signal("AAPL", SignalSource.SENTIMENT, score=-0.4, confidence=0.5),
    ]
    out = agg.aggregate(sigs)
    assert "AAPL" in out
    d = out["AAPL"]
    assert -1.0 <= d.score <= 1.0
    assert d.action in {"BUY", "SELL", "HOLD"}


def test_aggregator_empty():
    agg = SignalAggregator(weights={"technical": 1.0})
    assert agg.aggregate([]) == {}
