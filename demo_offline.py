"""Offline end-to-end demo — runs the REAL engine with synthetic data.

This exercises the full pipeline (data -> 4 signal engines -> aggregator ->
risk manager -> trader -> paper broker) without any network access or
external dependencies beyond numpy/pandas/pyyaml.

Use it as a smoke test or to see the engine "run" on a machine that can't
reach live market data.

    python demo_offline.py
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta

# ---- stub yfinance BEFORE importing project modules that import it ----
_yf = types.ModuleType("yfinance")
class _Ticker:
    def __init__(self, *a, **k): pass
    def history(self, *a, **k):
        import pandas as pd
        return pd.DataFrame()
    news = []
_yf.Ticker = _Ticker
_yf.download = lambda *a, **k: None
sys.modules["yfinance"] = _yf

import numpy as np
import pandas as pd

from src.engine import TradingEngine
from src.utils.logger import get_logger

log = get_logger("demo")

TICKERS = ["AAPL", "MSFT", "NVDA"]


def synth_bars(seed: int, n: int = 220, drift: float = 0.0012) -> pd.DataFrame:
    """Generate a synthetic intraday OHLCV series.

    The series dips for the first third (building an oversold RSI) then
    trends up with rising volume — a classic momentum setup that the
    technical engine should score bullish.
    """
    rng = np.random.default_rng(seed)
    ramp = np.concatenate([
        np.full(n // 3, -drift * 1.5),          # early weakness -> oversold
        np.full(n - n // 3, drift * 1.8),       # sustained recovery -> momentum
    ])
    rets = rng.normal(0, 0.0012, n) + ramp
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.0025, n))
    low = close * (1 - rng.uniform(0, 0.0025, n))
    open_ = np.r_[close[0], close[:-1]]
    # Volume rises into the trend (confirms the move)
    vol = (np.linspace(8_000, 60_000, n) * rng.uniform(0.7, 1.3, n)).astype(float)
    idx = pd.date_range(datetime(2024, 6, 3, 9, 30), periods=n, freq="5min")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def main():
    cfg = {
        "broker": {"name": "paper", "starting_cash": 25_000, "slippage_bps": 5},
        "universe": {"tickers": TICKERS},
        "data": {"provider": "yfinance", "bar_interval": "5m", "lookback_days": 30},
        "signals": {
            "weights": {"technical": 0.40, "sentiment": 0.20,
                        "fundamental": 0.15, "ml": 0.25},
            # 0.35 is calibrated for all 4 engines. Offline only the
            # technical engine runs, so we scale the entry threshold down
            # to its weight share (~0.40 of the budget => ~0.12).
            "enter_long_threshold": 0.10,
            "enter_short_threshold": -0.10,
            "exit_threshold": 0.10,
            "technical": {
                "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
                "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                "bb_period": 20, "bb_std": 2.0, "ema_short": 9, "ema_long": 21,
                "use_vwap": True, "atr_period": 14,
            },
            "sentiment": {"sources": [], "max_age_minutes": 120, "model": "vader"},
            "fundamental": {"sec_filing_types": ["8-K"], "earnings_window_days": 3},
            "ml": {"model_path": "models/xgb_intraday.joblib",
                   "prediction_horizon_minutes": 15, "min_confidence": 0.55},
        },
        "risk": {
            "max_position_pct": 0.10, "max_total_exposure_pct": 0.80,
            "max_concurrent_positions": 5, "daily_loss_limit_pct": 0.03,
            "per_trade_stop_loss_pct": 0.02, "take_profit_pct": 0.04,
            "trailing_stop_pct": 0.015, "use_atr_stops": True,
            "shorting_enabled": False, "pdt_protection": False,
            "kelly_fraction": 0.25,
        },
        "schedule": {
            "poll_seconds": 60, "market_open_buffer_minutes": 5,
            "market_close_buffer_minutes": 10, "trade_only_market_hours": False,
        },
        "notifications": {"channels": ["console"]},
    }

    engine = TradingEngine(cfg)

    # ---- wire synthetic data into the live objects (no network) ----
    bars = {t: synth_bars(seed=i) for i, t in enumerate(TICKERS)}

    engine.market.get_bars = lambda sym, force_refresh=False: bars[sym]
    engine.broker.get_last_price = lambda sym: float(bars[sym]["Close"].iloc[-1])
    # News + SEC feeds: skip the network, return nothing.
    engine.sent.news.get_headlines = lambda sym: []
    engine.fund.sec.get_recent_filings = lambda sym, lookback_hours=48: []

    print("=" * 72)
    print("OFFLINE ENGINE DEMO - synthetic data, paper broker, 3 cycles")
    print("Note: only the TECHNICAL engine is fully active offline; sentiment")
    print("needs vaderSentiment, ML needs a trained model, fundamental needs")
    print("network. They return neutral here, so the aggregate = technical.")
    print("=" * 72)

    for cycle in range(1, 4):
        print(f"\n----- CYCLE {cycle} -----")
        engine.run_once()
        eq = engine.broker.get_equity()
        cash = engine.broker.get_cash()
        pos = engine.broker.get_positions()
        print(f"  equity=${eq:,.2f}  cash=${cash:,.2f}  open_positions={len(pos)}")
        for s, p in pos.items():
            print(f"    {s}: qty={p.qty} entry=${p.avg_entry_price:.2f} "
                  f"last=${p.current_price:.2f} pnl={p.unrealized_pnl:+.2f}")
        # advance the synthetic series by one bar so prices move between cycles
        for t in TICKERS:
            df = bars[t]
            last = df.iloc[-1]
            nxt = last.copy()
            bump = 1.015 if t == "NVDA" else 0.985  # NVDA +1.5%/cycle, others -1.5%
            nxt["Close"] = last["Close"] * bump
            nxt["Open"] = last["Close"]
            nxt["High"] = max(nxt["Close"], last["Close"]) * 1.001
            nxt["Low"] = min(nxt["Close"], last["Close"]) * 0.999
            new_idx = df.index[-1] + timedelta(minutes=5)
            bars[t] = pd.concat([df, pd.DataFrame([nxt], index=[new_idx])])

    print("\n" + "=" * 72)
    print(f"FINAL  equity=${engine.broker.get_equity():,.2f}  "
          f"realized_pnl=${engine.broker.realized_pnl:+.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
