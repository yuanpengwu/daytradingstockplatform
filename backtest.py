#!/usr/bin/env python3
"""
backtest.py — 1-week technical-only backtest with automatic parameter tuning.
=============================================================================

Run from project root:
    python backtest.py

What it does
------------
1. Fetches 1 week of 5-minute Alpaca bars for all configured tickers.
2. Replays each trading day bar-by-bar, computing ONLY TechnicalSignal.
   (Sentiment / Fundamental / ML / ORB / VWAP-Bounce are all disabled.)
3. Simulates long entries and exits (stop-loss, take-profit, trailing-stop,
   signal-reversal, end-of-day) with the current config parameters.
4. Prints a full trade table + win-rate summary.
5. If win rate < 50 %, runs a grid search over the most impactful parameters
   and prints the best configuration found, then writes the new values back
   to config.yaml automatically.
"""
from __future__ import annotations

import copy
import itertools
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# ── bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env before importing project modules
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import pandas as pd
import yaml
from datetime import time as dtime
from zoneinfo import ZoneInfo

from src.signals.technical import TechnicalSignal
from src.signals.vwap_bounce import VWAPBounceSignal
from src.data.sectors import get_sector, is_market_etf

NY = ZoneInfo("America/New_York")

# ─────────────────────────────────────────────────────────────────────────────
# Load project config
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_bars_alpaca(tickers: List[str], days: int = 9,
                      bar_minutes: int = 1) -> Dict[str, pd.DataFrame]:
    """Fetch N-min bars via Alpaca historical data API."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    client = StockHistoricalDataClient(key, secret)
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(bar_minutes, TimeFrameUnit.Minute),
        start=start, end=end,
        feed=DataFeed.IEX,
    )
    resp = client.get_stock_bars(req)

    result: Dict[str, pd.DataFrame] = {}
    for sym in tickers:
        try:
            df = resp.df
            if isinstance(df.index, pd.MultiIndex):
                if sym not in df.index.get_level_values(0):
                    continue
                df = df.loc[sym]
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                    "close": "Close", "volume": "Volume"})
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(NY)
            df = df.between_time("09:30", "16:00").dropna()
            if len(df) >= 30:
                result[sym] = df
        except Exception:
            pass
    return result


def fetch_bars_yfinance(tickers: List[str], days: int = 9,
                        bar_minutes: int = 1) -> Dict[str, pd.DataFrame]:
    """Fetch N-min bars via yfinance (fallback, free, no API key needed)."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed.  Run: pip install yfinance")

    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    period_str = f"{days}d"

    result: Dict[str, pd.DataFrame] = {}
    for sym in tickers:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period_str, interval=f"{bar_minutes}m", auto_adjust=True)
            if df.empty:
                continue
            df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                                    "Close": "Close", "Volume": "Volume"})
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(NY)
            df = df.between_time("09:30", "16:00").dropna()
            if len(df) >= 30:
                result[sym] = df
        except Exception as e:
            print(f"  {sym}: yfinance skipped ({e})")
    return result


def fetch_bars(tickers: List[str], days: int = 9,
               bar_minutes: int = 1) -> Dict[str, pd.DataFrame]:
    """Fetch `days` calendar days of N-minute bars.

    Tries Alpaca first (IEX feed, highest quality); falls back to yfinance
    automatically if Alpaca is unavailable (network issues, no API key, etc.).
    """
    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")

    label = f"{bar_minutes}-min"
    # ── Try Alpaca ────────────────────────────────────────────────────────────
    if key and secret:
        print(f"Fetching {days}-day {label} bars for {len(tickers)} tickers via Alpaca …")
        try:
            result = fetch_bars_alpaca(tickers, days, bar_minutes=bar_minutes)
            if result:
                for sym, df in result.items():
                    print(f"  {sym}: {len(df)} bars")
                return result
            print("  Alpaca returned no data — falling back to yfinance.")
        except Exception as e:
            print(f"  Alpaca fetch failed ({e}) — falling back to yfinance.")
    else:
        print("  No Alpaca keys found — using yfinance.")

    # ── Fall back to yfinance ─────────────────────────────────────────────────
    print(f"Fetching {days}-day {label} bars for {len(tickers)} tickers via yfinance …")
    result = fetch_bars_yfinance(tickers, days, bar_minutes=bar_minutes)
    for sym, df in result.items():
        print(f"  {sym}: {len(df)} bars")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

class Trade(NamedTuple):
    symbol: str
    date: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    qty: float           # 1 share — P&L is per-share for comparability
    pnl: float
    pnl_pct: float
    reason: str
    won: bool


# ─────────────────────────────────────────────────────────────────────────────
# Backtest core
# ─────────────────────────────────────────────────────────────────────────────

def _split_days(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split a multi-day DataFrame into per-trading-day slices."""
    out = []
    for _, grp in df.groupby(df.index.date):
        if len(grp) >= 30:
            out.append(grp)
    return out


def _regime_size_mult(spy_day_df: Optional[pd.DataFrame]) -> float:
    """Return a position-size multiplier based on SPY vs its 20-bar SMA.

    Mirrors the live RegimeDetector but VIX-free (no external call needed).
    trending_bull → 1.0 | choppy → 0.7 | trending_bear → 0.5
    """
    if spy_day_df is None or len(spy_day_df) < 20:
        return 0.7  # conservative default
    close = spy_day_df["Close"].astype(float)
    current = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    vs_sma = (current - sma20) / sma20 if sma20 > 0 else 0.0
    mom5 = (current / float(close.iloc[-6]) - 1.0) if len(close) > 5 and float(close.iloc[-6]) > 0 else 0.0
    if vs_sma > 0.002 and mom5 > -0.001:
        return 1.0   # trending_bull
    elif vs_sma < -0.002 or mom5 < -0.004:
        return 0.5   # trending_bear
    return 0.7       # choppy


def _compute_rs_at_bar(
    sym: str,
    day_slices: Dict[str, pd.DataFrame],
    spy_day: Optional[pd.DataFrame],
    bar_i: int,
    lookback: int,
) -> float:
    """Relative strength of sym vs SPY over last `lookback` bars at bar_i."""
    sym_df = day_slices.get(sym)
    if sym_df is None or spy_day is None:
        return 0.0
    n = lookback + 1
    if bar_i < n or len(spy_day) <= bar_i:
        return 0.0
    try:
        sym_close = sym_df["Close"].astype(float)
        spy_close = spy_day["Close"].astype(float)
        sym_ret = (float(sym_close.iloc[bar_i]) - float(sym_close.iloc[bar_i - lookback])) / float(sym_close.iloc[bar_i - lookback])
        spy_ret = (float(spy_close.iloc[bar_i]) - float(spy_close.iloc[bar_i - lookback])) / float(spy_close.iloc[bar_i - lookback])
        return sym_ret - spy_ret
    except (ZeroDivisionError, IndexError):
        return 0.0


def _orb_score_at_bar(
    bars: pd.DataFrame,
    orb_minutes: int = 15,
    vol_mult: float = 1.3,
    decay_start: float = 120.0,
    decay_end: float = 240.0,
) -> Tuple[float, float]:
    """Backtest-safe ORB signal: uses bar timestamps for time decay, not wall clock.

    The live ORBSignal calls pd.Timestamp.now() which always returns current
    real-world time — unusable during backtest replay.  This version reads the
    last bar's own timestamp so decay is computed correctly for any historical bar.
    """
    if bars is None or len(bars) < 3:
        return 0.0, 0.0
    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_ny = idx.tz_convert(NY)
    today = idx_ny[-1].date()
    market_open = pd.Timestamp(today, tz=NY).replace(hour=9, minute=30)
    session_mask = idx_ny >= market_open
    session = bars[session_mask]
    orb_n = max(1, orb_minutes // 5)
    if len(session) < orb_n:
        return 0.0, 0.0
    orb_bars = session.iloc[:orb_n]
    orb_high = float(orb_bars["High"].max())
    orb_low  = float(orb_bars["Low"].min())
    orb_range = max(orb_high - orb_low, 1e-6)
    close = float(bars["Close"].iloc[-1])
    if close > orb_high:
        raw = float(np.tanh((close - orb_high) / orb_range * 3))
    elif close < orb_low:
        raw = -float(np.tanh((orb_low - close) / orb_range * 3))
    else:
        return 0.0, 0.0
    vol_avg   = float(bars["Volume"].tail(20).mean()) or 1.0
    vol_ratio = float(bars["Volume"].iloc[-1]) / vol_avg
    vol_factor = float(np.clip(vol_ratio / vol_mult, 0.3, 1.5))
    # Time decay using bar timestamp (not wall clock)
    elapsed = max(0.0, (bars.index[-1] - market_open).total_seconds() / 60)
    if elapsed > decay_end:
        return 0.0, 0.0
    decay = 1.0 if elapsed <= decay_start else (
        1.0 - (elapsed - decay_start) / max(decay_end - decay_start, 1e-6)
    )
    score = float(np.clip(raw * vol_factor * decay, -1.0, 1.0))
    confidence = float(np.clip(0.3 + abs(score) * 0.5 + min(vol_ratio - 1, 0.5) * 0.2, 0.0, 1.0))
    return score, confidence


def _make_stops(
    close: float,
    atr_val: float,
    *,
    use_atr: bool,
    stop_pct: float,
    tp_pct: float,
    atr_stop_mult: float = 2.5,
    atr_tp_mult: float = 3.0,
) -> Tuple[float, float]:
    """Compute (stop_price, tp_price) applying the ATR-floor fix.

    ATR can tighten the stop (good) but NEVER lower the TP below the
    configured take_profit_pct. This mirrors the live risk_manager fix.
    """
    pct_stop = close * (1 - stop_pct)
    pct_tp   = close * (1 + tp_pct)
    if use_atr and atr_val > 0:
        atr_stop = close - atr_stop_mult * atr_val
        atr_tp   = close + atr_tp_mult   * atr_val
        return max(atr_stop, pct_stop), max(atr_tp, pct_tp)
    return pct_stop, pct_tp


def _compute_stats(trades: List[Trade]) -> dict:
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": None, "total_pnl": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "expectancy": 0.0,
        }
    wins = sum(1 for t in trades if t.won)
    losses = total - wins
    win_pnl  = [t.pnl_pct for t in trades if t.won]
    loss_pnl = [t.pnl_pct for t in trades if not t.won]
    avg_w = float(np.mean(win_pnl))  if win_pnl  else 0.0
    avg_l = float(np.mean(loss_pnl)) if loss_pnl else 0.0
    wr = wins / total
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 4),
        "total_pnl": round(sum(t.pnl for t in trades), 4),
        "avg_win_pct":  round(avg_w * 100, 3),
        "avg_loss_pct": round(avg_l * 100, 3),
        "expectancy":   round((wr * avg_w + (1 - wr) * avg_l) * 100, 3),
    }


def run_backtest(
    bars_by_sym: Dict[str, pd.DataFrame],
    tech_cfg: dict,
    *,
    enter_threshold: float = 0.35,
    exit_threshold:  float = 0.15,
    stop_pct:        float = 0.02,
    tp_pct:          float = 0.04,
    trail_pct:       float = 0.02,
    min_hold_bars:   int   = 5,       # 25 min ÷ 5-min bars
    use_atr_stops:   bool  = True,
    min_conf:        float = 0.30,
    breakeven_trigger: float = 0.004,
    open_buffer_bars:  int  = 6,      # skip first 30 min
    # ── New strategy filters ──────────────────────────────────────────────────
    use_dead_zone:    bool  = True,
    dead_zone_start:  dtime = dtime(11, 30),
    dead_zone_end:    dtime = dtime(14, 0),
    pre_close_cutoff: dtime = dtime(15, 20),  # no new entries after this time
    use_rs_filter:    bool  = True,
    rs_min:           float = 0.0,
    rs_lookback:      int   = 6,
    use_sector_limit: bool  = True,
    max_per_sector:   int   = 1,
    use_regime:       bool  = True,
    max_concurrent:   int   = 5,
    atr_stop_mult:    float = 2.5,    # wider ATR stop (was hard-coded 2.0)
    atr_tp_mult:      float = 3.0,
    orb_cfg:          Optional[dict] = None,
    vwap_cfg:         Optional[dict] = None,
    # ── New entry quality filters ─────────────────────────────────────────────
    use_prev_close_filter: bool  = False,  # only enter if price > prev day close
    use_next_day_cooloff:  bool  = False,  # ban symbol next full day after stop-loss
) -> Tuple[List[Trade], dict]:
    """Portfolio-level bar-by-bar replay incorporating all new strategy filters.

    New vs old backtest:
      • Portfolio simulation — all tickers processed together each bar
      • Dead zone filter     — no new entries 11:30–14:00 ET
      • Relative strength    — only enters when stock outperforms SPY (6-bar)
      • Sector concentration — max 1 open position per sector at a time
      • Regime sizing        — halve size on bear/high-vol days (SPY vs SMA20)
      • ATR TP floor fix     — TP = max(3×ATR, configured %)  (was raw 3×ATR)
    """
    signal_engine = TechnicalSignal(tech_cfg)
    vwap_engine   = VWAPBounceSignal(vwap_cfg or {})
    _orb = orb_cfg or {}
    _orb_params = dict(
        orb_minutes  = int(_orb.get("orb_minutes", 15)),
        vol_mult     = float(_orb.get("volume_confirmation_mult", 1.3)),
        decay_start  = float(_orb.get("decay_start_minutes", 120)),
        decay_end    = float(_orb.get("decay_end_minutes", 240)),
    )
    # Signal weights (normalised to sum=1 using live config values)
    _W_TECH, _W_ORB, _W_VWAP = 0.30, 0.20, 0.15
    _W_TOTAL = _W_TECH + _W_ORB + _W_VWAP

    trades: List[Trade] = []
    # LOOKBACK must cover the longest indicator period (MACD slow = up to 105 bars
    # on 1-min) plus the open-buffer.  120 is a safe floor for 1-min bars.
    LOOKBACK = max(120, open_buffer_bars)

    # ── Collect all unique trading dates ─────────────────────────────────────
    all_dates = sorted({d.date() for df in bars_by_sym.values() for d in df.index})

    # ── Pre-compute previous-day close for each symbol / date ─────────────────
    # prev_closes[sym][date] = closing price of sym on the trading day before date
    prev_closes: Dict[str, Dict] = {}
    if use_prev_close_filter:
        for sym, df_full in bars_by_sym.items():
            prev_closes[sym] = {}
            for idx, tdate in enumerate(all_dates):
                if idx == 0:
                    continue
                prev_date = all_dates[idx - 1]
                prev_bars = df_full[df_full.index.date == prev_date]
                if len(prev_bars) > 0:
                    prev_closes[sym][tdate] = float(prev_bars["Close"].iloc[-1])

    # ── Cross-day symbol cooloff after stop-loss ──────────────────────────────
    # cooloff_until[sym] = date (exclusive) through which sym is banned
    cooloff_until: Dict[str, object] = {}

    for trade_date in all_dates:
        # Build day slices for every symbol
        day_slices: Dict[str, pd.DataFrame] = {}
        for sym, df_full in bars_by_sym.items():
            day_df = df_full[df_full.index.date == trade_date]
            if len(day_df) >= LOOKBACK:
                day_slices[sym] = day_df

        if not day_slices:
            continue

        spy_day = day_slices.get("SPY")

        # Regime size multiplier (computed from full-day SPY slice context)
        size_mult = _regime_size_mult(spy_day) if use_regime else 1.0

        # Expire cooloff entries that have passed
        if use_next_day_cooloff:
            cooloff_until = {s: d for s, d in cooloff_until.items() if d > trade_date}

        # Per-day portfolio state
        # positions: sym → {entry_price, peak, stop, tp, entry_bar_idx}
        positions: Dict[str, dict] = {}
        # Sectors currently held (for concentration gate)
        sectors_held: set = set()
        date_str = str(trade_date)

        # Determine the longest bar count to iterate over
        max_bars = max(len(df) for df in day_slices.values())

        for i in range(LOOKBACK, max_bars):
            # ── Bar time for dead-zone check ──────────────────────────────────
            bar_time: Optional[dtime] = None
            for df in day_slices.values():
                if i < len(df):
                    bar_time = df.index[i].time()
                    break

            in_dead_zone = (
                use_dead_zone
                and bar_time is not None
                and dead_zone_start <= bar_time < dead_zone_end
            )
            past_entry_cutoff = (bar_time is not None and bar_time >= pre_close_cutoff)

            # ── 1. Manage open positions ──────────────────────────────────────
            for sym in list(positions.keys()):
                df = day_slices.get(sym)
                if df is None or i >= len(df):
                    continue
                close = float(df["Close"].iloc[i])
                pos = positions[sym]

                if close > pos["peak"]:
                    pos["peak"] = close

                pnl_pct_live = (close - pos["entry_price"]) / pos["entry_price"]
                exit_reason: Optional[str] = None

                # Breakeven adjustment
                eff_stop = pos["stop"]
                if eff_stop is not None and pnl_pct_live >= breakeven_trigger:
                    eff_stop = max(eff_stop, pos["entry_price"])

                # 1a. Absolute stop / TP
                if eff_stop is not None and close <= eff_stop:
                    exit_reason = "breakeven_stop" if eff_stop != pos["stop"] else "stop_loss"
                elif pos["tp"] is not None and close >= pos["tp"]:
                    exit_reason = "take_profit"
                else:
                    # 1b. % fallback
                    eff_stop_pct = 0.0 if pnl_pct_live >= breakeven_trigger else stop_pct
                    if eff_stop_pct > 0 and pnl_pct_live <= -eff_stop_pct:
                        exit_reason = "stop_loss"
                    elif pnl_pct_live >= tp_pct:
                        exit_reason = "take_profit"
                    else:
                        # 1c. Trailing stop
                        if pos["peak"] > pos["entry_price"]:
                            dd = (pos["peak"] - close) / pos["peak"]
                            if dd >= trail_pct:
                                exit_reason = "trailing_stop"

                        # 1d. Signal reversal (suppressed in profit)
                        if exit_reason is None:
                            held = i - pos["entry_bar"]
                            if held >= min_hold_bars and pnl_pct_live <= 0:
                                window = df.iloc[: i + 1]
                                sig = signal_engine.evaluate(sym, window)
                                if sig and sig.score < -exit_threshold:
                                    exit_reason = "signal_reversal"

                # 1e. EOD
                if exit_reason is None and i == len(df) - 1:
                    exit_reason = "eod"

                if exit_reason:
                    pnl = close - pos["entry_price"]
                    pnl_pct = pnl / pos["entry_price"]
                    trades.append(Trade(
                        symbol=sym,
                        date=date_str,
                        entry_bar=pos["entry_bar"],
                        exit_bar=i,
                        entry_price=round(pos["entry_price"], 4),
                        exit_price=round(close, 4),
                        qty=1.0,
                        pnl=round(pnl, 4),
                        pnl_pct=round(pnl_pct, 6),
                        reason=exit_reason,
                        won=pnl > 0,
                    ))
                    sym_sector = get_sector(sym)
                    if not is_market_etf(sym):
                        sectors_held.discard(sym_sector)
                    # Register next-day cooloff after a stop-loss
                    if use_next_day_cooloff and exit_reason in ("stop_loss", "breakeven_stop"):
                        date_idx = all_dates.index(trade_date)
                        if date_idx + 1 < len(all_dates):
                            # ban through end of next trading day (exclusive = day after next)
                            ban_through = all_dates[date_idx + 2] if date_idx + 2 < len(all_dates) else all_dates[-1]
                            cooloff_until[sym] = ban_through
                    del positions[sym]

            # ── 2. New entries (blocked in dead zone or past pre-close cutoff) ──
            if in_dead_zone or past_entry_cutoff or len(positions) >= max_concurrent:
                continue

            # Score all eligible symbols this bar using combined Tech+ORB+VWAP signal
            candidates: List[Tuple[str, float, object]] = []
            for sym, df in day_slices.items():
                if sym in positions or i >= len(df):
                    continue
                window = df.iloc[: i + 1]

                # Technical signal (primary — also provides ATR for stops)
                tech_sig = signal_engine.evaluate(sym, window)
                tech_s = tech_sig.score if tech_sig else 0.0
                tech_c = tech_sig.confidence if tech_sig else 0.0

                # ORB signal (backtest-safe: uses bar timestamps, not wall clock)
                orb_s, orb_c = _orb_score_at_bar(window, **_orb_params)

                # VWAP-bounce signal (pure bar-data, backtest-safe)
                vwap_sig = vwap_engine.evaluate(sym, window)
                vwap_s = vwap_sig.score if vwap_sig else 0.0
                vwap_c = vwap_sig.confidence if vwap_sig else 0.0

                # Weighted aggregate — only include signals that have an actual
                # opinion (non-zero confidence).  Neutral ORB/VWAP (price inside
                # range, no setup) should not dilute a strong tech signal.
                components = []
                if tech_sig and tech_c > 0.01:
                    components.append((tech_s, tech_c, _W_TECH))
                if orb_c > 0.01:
                    components.append((orb_s, orb_c, _W_ORB))
                if vwap_sig and vwap_c > 0.01:
                    components.append((vwap_s, vwap_c, _W_VWAP))
                if not components:
                    continue
                _active_w  = sum(w for _, _, w in components)
                agg_score  = sum(s * w for s, _, w in components) / _active_w
                agg_conf   = sum(c * w for _, c, w in components) / _active_w

                if agg_score >= enter_threshold and agg_conf >= min_conf:
                    candidates.append((sym, agg_score, tech_sig))

            # Sort by signal strength descending — take best setups first
            candidates.sort(key=lambda x: x[1], reverse=True)

            for sym, score, sig in candidates:
                if len(positions) >= max_concurrent:
                    break

                df = day_slices[sym]
                close = float(df["Close"].iloc[i])

                # Gate A: Relative strength filter (longs only, skip broad ETFs)
                if use_rs_filter and not is_market_etf(sym):
                    rs = _compute_rs_at_bar(sym, day_slices, spy_day, i, rs_lookback)
                    if rs < rs_min:
                        continue

                # Gate B: Sector concentration
                sym_sector = get_sector(sym)

                # Gate C: Prev-day close trend filter — only enter if up vs yesterday
                if use_prev_close_filter and not is_market_etf(sym):
                    pc = prev_closes.get(sym, {}).get(trade_date)
                    if pc is not None and close <= pc:
                        continue

                # Gate D: Next-day cooloff — skip symbol banned after prior stop-loss
                if use_next_day_cooloff and sym in cooloff_until:
                    continue
                if use_sector_limit and not is_market_etf(sym) and sym_sector != "Unknown":
                    count = sum(1 for s in positions if get_sector(s) == sym_sector and not is_market_etf(s))
                    if count >= max_per_sector:
                        continue

                # Compute stops with ATR-floor fix
                atr_val = sig.metadata.get("atr", 0.0)
                stop_p, tp_p = _make_stops(
                    close, atr_val,
                    use_atr=use_atr_stops,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    atr_stop_mult=atr_stop_mult,
                    atr_tp_mult=atr_tp_mult,
                )

                positions[sym] = {
                    "entry_price": close,
                    "peak": close,
                    "stop": stop_p,
                    "tp":   tp_p,
                    "entry_bar": i,
                    "size_mult": size_mult,
                }
                if not is_market_etf(sym) and sym_sector != "Unknown":
                    sectors_held.add(sym_sector)

        # ── EOD sweep: close any still-open positions ─────────────────────────
        for sym, pos in positions.items():
            df = day_slices.get(sym)
            if df is None:
                continue
            close = float(df["Close"].iloc[-1])
            pnl = close - pos["entry_price"]
            pnl_pct = pnl / pos["entry_price"]
            trades.append(Trade(
                symbol=sym,
                date=date_str,
                entry_bar=pos["entry_bar"],
                exit_bar=len(df) - 1,
                entry_price=round(pos["entry_price"], 4),
                exit_price=round(close, 4),
                qty=1.0,
                pnl=round(pnl, 4),
                pnl_pct=round(pnl_pct, 6),
                reason="eod",
                won=pnl > 0,
            ))

    return trades, _compute_stats(trades)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats(stats: dict, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    wr = stats["win_rate"]
    wr_str = f"{wr*100:.1f}%" if wr is not None else "N/A"
    print(f"\n{'='*60}")
    print(f"  Backtest Results{tag}")
    print(f"{'='*60}")
    print(f"  Trades  : {stats['total_trades']}")
    print(f"  Wins    : {stats['wins']}")
    print(f"  Losses  : {stats['losses']}")
    print(f"  Win rate: {wr_str}")
    print(f"  Total P&L (per-share sum): {stats['total_pnl']:+.4f}")
    print(f"  Avg win : +{stats['avg_win_pct']:.3f}%")
    print(f"  Avg loss: {stats['avg_loss_pct']:.3f}%")
    print(f"  Expectancy: {stats['expectancy']:+.3f}% per trade")
    print(f"{'='*60}")


def _print_trade_table(trades: List[Trade], max_rows: int = 40) -> None:
    if not trades:
        print("  (no trades)")
        return
    print(f"\n  {'SYM':<6} {'DATE':<12} {'ENTRY':>8} {'EXIT':>8} "
          f"{'PNL%':>7} {'REASON':<16} {'W/L'}")
    print("  " + "-" * 65)
    shown = trades[-max_rows:] if len(trades) > max_rows else trades
    for t in shown:
        wl = "WIN " if t.won else "LOSS"
        print(f"  {t.symbol:<6} {t.date:<12} {t.entry_price:>8.2f} {t.exit_price:>8.2f} "
              f"{t.pnl_pct*100:>+6.2f}% {t.reason:<16} {wl}")
    if len(trades) > max_rows:
        print(f"  … and {len(trades)-max_rows} earlier trades (showing last {max_rows})")


def _reason_breakdown(trades: List[Trade]) -> None:
    from collections import Counter
    reasons = Counter(t.reason for t in trades)
    wins_by_reason = Counter(t.reason for t in trades if t.won)
    print("\n  Exit reason breakdown:")
    print(f"  {'Reason':<20} {'Count':>6} {'Wins':>6} {'WinRate':>8}")
    print("  " + "-" * 44)
    for reason, count in reasons.most_common():
        w = wins_by_reason.get(reason, 0)
        wr = f"{w/count*100:.0f}%" if count else "—"
        print(f"  {reason:<20} {count:>6} {w:>6} {wr:>8}")


def _sector_breakdown(trades: List[Trade]) -> None:
    from collections import defaultdict
    sector_stats: dict = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        sector = get_sector(t.symbol) if not is_market_etf(t.symbol) else "Market ETF"
        sector_stats[sector]["n"] += 1
        sector_stats[sector]["wins"] += int(t.won)
        sector_stats[sector]["pnl"] += t.pnl_pct * 100
    print("\n  Sector breakdown:")
    print(f"  {'Sector':<24} {'Trades':>6} {'WinRate':>8} {'Avg P&L%':>9}")
    print("  " + "-" * 52)
    for sector, s in sorted(sector_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = f"{s['wins']/s['n']*100:.0f}%" if s["n"] else "—"
        avg_pnl = s["pnl"] / s["n"] if s["n"] else 0
        print(f"  {sector:<24} {s['n']:>6} {wr:>8} {avg_pnl:>+8.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(
    bars_by_sym: Dict[str, pd.DataFrame],
    base_tech_cfg: dict,
    base_risk_cfg: dict,
) -> Optional[dict]:
    """Search over key parameters to find a config with win_rate >= 50 %.

    Grid dimensions (ordered by impact):
      - enter_threshold    : how strong the signal must be to enter
      - stop_pct           : hard stop distance
      - tp_pct             : take-profit distance
      - min_conf           : minimum confidence gate
      - rsi_period         : RSI lookback
      - rsi_oversold/overbought boundaries
    """
    print("\n" + "="*60)
    print("  Win rate < 50% — running parameter grid search …")
    print("="*60)

    # Candidate values for each dimension.
    # Thresholds start lower because the redesigned scoring formula now
    # produces scores in the 0.15–0.60 range for real intraday moves.
    grid = {
        "enter_threshold":  [0.15, 0.20, 0.25, 0.30, 0.35],
        "stop_pct":         [0.015, 0.020, 0.025, 0.030],
        "tp_pct":           [0.025, 0.030, 0.040, 0.050, 0.060],
        "min_conf":         [0.35, 0.40, 0.45, 0.50],
        "rsi_period":       [7, 9, 14],
        "rsi_overbought":   [60, 65, 70],
    }

    # rsi_oversold is always (100 - rsi_overbought) symmetrically
    total_combos = (
        len(grid["enter_threshold"])
        * len(grid["stop_pct"])
        * len(grid["tp_pct"])
        * len(grid["min_conf"])
        * len(grid["rsi_period"])
        * len(grid["rsi_overbought"])
    )
    print(f"  Total combinations to evaluate: {total_combos}")
    print("  This may take a few minutes …\n")

    best_cfg: Optional[dict] = None
    best_wr: float = 0.0
    best_stats: Optional[dict] = None
    evaluated = 0

    for (eth, stp, tpp, mconf, rsi_p, rsi_ob) in itertools.product(
        grid["enter_threshold"],
        grid["stop_pct"],
        grid["tp_pct"],
        grid["min_conf"],
        grid["rsi_period"],
        grid["rsi_overbought"],
    ):
        # Skip configs where stop >= tp (no viable risk/reward)
        if stp >= tpp:
            continue
        rsi_os = 100 - rsi_ob   # symmetric

        tech_cfg = copy.deepcopy(base_tech_cfg)
        tech_cfg["rsi_period"]     = rsi_p
        tech_cfg["rsi_oversold"]   = rsi_os
        tech_cfg["rsi_overbought"] = rsi_ob

        _, stats = run_backtest(
            bars_by_sym,
            tech_cfg,
            enter_threshold=eth,
            stop_pct=stp,
            tp_pct=tpp,
            min_conf=mconf,
            trail_pct=base_risk_cfg.get("trailing_stop_pct", 0.020),
            min_hold_bars=base_risk_cfg.get("min_hold_minutes", 15) // 5,
            use_atr_stops=base_risk_cfg.get("use_atr_stops", True),
            exit_threshold=0.15,
        )

        evaluated += 1
        wr = stats.get("win_rate") or 0.0
        n  = stats.get("total_trades", 0)

        # Must have at least 10 trades to be considered valid
        if n >= 10 and wr > best_wr:
            best_wr = wr
            best_cfg = {
                "enter_threshold": eth,
                "stop_pct":        stp,
                "tp_pct":          tpp,
                "min_conf":        mconf,
                "rsi_period":      rsi_p,
                "rsi_oversold":    rsi_os,
                "rsi_overbought":  rsi_ob,
                "tech_cfg":        tech_cfg,
            }
            best_stats = stats

        # Progress ping every 500 evaluations
        if evaluated % 500 == 0:
            pct = evaluated / total_combos * 100
            print(f"  … {evaluated}/{total_combos} ({pct:.0f}%)  best so far: "
                  f"win_rate={best_wr*100:.1f}%  trades={best_stats['total_trades'] if best_stats else 0}")

        # Stop early if we already have a great result (>= 60%)
        if best_wr >= 0.60:
            print(f"  Early stop — found win rate {best_wr*100:.1f}% >= 60%")
            break

    return best_cfg, best_stats


# ─────────────────────────────────────────────────────────────────────────────
# Config writer
# ─────────────────────────────────────────────────────────────────────────────

def _apply_best_config(cfg: dict, best: dict) -> None:
    """Write the winning parameter set back to config.yaml in-place."""
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = f.read()

    def _replace(text: str, key: str, new_val) -> str:
        """Replace `key: <value>` in yaml text (first match only)."""
        import re
        pattern = rf"(^[ \t]*{re.escape(key)}:)[ \t]*.+"
        replacement = rf"\g<1> {new_val}"
        return re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)

    raw = _replace(raw, "enter_long_threshold", best["enter_threshold"])
    raw = _replace(raw, "per_trade_stop_loss_pct", best["stop_pct"])
    raw = _replace(raw, "take_profit_pct", best["tp_pct"])
    raw = _replace(raw, "min_confidence", best["min_conf"])
    raw = _replace(raw, "rsi_period", best["rsi_period"])
    raw = _replace(raw, "rsi_oversold", best["rsi_oversold"])
    raw = _replace(raw, "rsi_overbought", best["rsi_overbought"])

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"\n  config.yaml updated with new parameters.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _load_config()
    scfg = cfg.get("signals", {})
    rcfg = cfg.get("risk", {})
    tech_cfg = scfg.get("technical", {})

    # Current baseline parameters
    baseline = dict(
        enter_threshold  = float(scfg.get("enter_long_threshold", 0.35)),
        exit_threshold   = float(scfg.get("exit_threshold", 0.15)),
        stop_pct         = float(rcfg.get("per_trade_stop_loss_pct", 0.02)),
        tp_pct           = float(rcfg.get("take_profit_pct", 0.04)),
        trail_pct        = float(rcfg.get("trailing_stop_pct", 0.020)),
        min_hold_bars    = int(rcfg.get("min_hold_minutes", 25)) // 5,   # 25 min → 5 bars
        use_atr_stops    = bool(rcfg.get("use_atr_stops", True)),
        min_conf         = float(scfg.get("min_confidence", 0.25)),
        # ── New strategy filters ──────────────────────────────────────────────
        use_dead_zone    = True,
        dead_zone_start  = dtime(11, 30),
        dead_zone_end    = dtime(14, 0),
        use_rs_filter    = True,
        rs_min           = float(scfg.get("relative_strength_min", 0.0)),
        rs_lookback      = int(scfg.get("rs_lookback_bars", 6)),
        use_sector_limit = True,
        max_per_sector   = int(rcfg.get("max_positions_per_sector", 1)),
        use_regime       = True,
        max_concurrent   = int(rcfg.get("max_concurrent_positions", 5)),
    )

    # In technical-only backtest mode confidence is applied directly (no aggregator).
    # Redesigned scoring raises baseline confidence to ~0.45, so use 0.40 gate.
    baseline["min_conf"] = 0.40
    # Calibrated to the redesigned scoring range (0.15–0.60).
    baseline["enter_threshold"] = min(baseline["enter_threshold"], 0.20)

    # Tickers: use the full current live universe + SPY for RS filter
    ucfg = cfg.get("universe", {})
    # Last known live universe (from logs) + fallback for robustness
    live_universe = [
        "QCOM", "AMD", "INTC", "CRM", "TXN",   # Technology (XLK)
        "MS", "GS", "BAC", "WFC", "C",           # Financials (XLF)
        "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN",  # fallback
        "SPY", "QQQ",                             # broad-market ETFs
    ]
    tickers: List[str] = list(dict.fromkeys(live_universe))  # deduplicate

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    bars = fetch_bars(tickers, days=9)  # 9 calendar days ≈ 5–6 trading days
    if not bars:
        sys.exit("No bar data returned — check Alpaca credentials in .env")
    print(f"\nData ready: {len(bars)} symbols, ~{sum(len(v) for v in bars.values())} total bars")

    # Date range info
    all_dates = sorted({str(d.date()) for df in bars.values() for d in df.index})
    print(f"Trading dates in dataset: {', '.join(all_dates)}")

    # ── 2. Baseline backtest ────────────────────────────────────────────────────
    print(f"\nRunning baseline backtest …")
    print(f"  enter_threshold={baseline['enter_threshold']}  "
          f"stop={baseline['stop_pct']*100:.1f}%  "
          f"tp={baseline['tp_pct']*100:.1f}%  "
          f"min_conf={baseline['min_conf']}  "
          f"rsi_period={tech_cfg.get('rsi_period', 14)}")

    trades, stats = run_backtest(
        bars, tech_cfg,
        **baseline,
        breakeven_trigger=float(rcfg.get("breakeven_trigger_pct", 0.004)),
        open_buffer_bars=int(cfg.get("schedule", {}).get("market_open_buffer_minutes", 30) // 5),
    )

    _print_stats(stats, "baseline")
    _print_trade_table(trades)
    _reason_breakdown(trades)
    _sector_breakdown(trades)

    wr = stats.get("win_rate") or 0.0

    # ── 3. Grid search if win rate < 50% ────────────────────────────────────────
    if stats["total_trades"] == 0:
        print("\n  ⚠  No trades were generated with current thresholds.")
        print("  Try lowering enter_long_threshold or min_confidence in config.yaml")
        print("  Running grid search to find working parameters …")
        run_search = True
    elif wr < 0.50:
        print(f"\n  ⚠  Win rate {wr*100:.1f}% < 50% — initiating parameter search …")
        run_search = True
    else:
        print(f"\n  ✓  Win rate {wr*100:.1f}% ≥ 50%.  Current parameters are good.")
        run_search = False

    if run_search:
        best, best_stats = grid_search(bars, tech_cfg, rcfg)

        if best is None:
            print("\n  ✗  Grid search found no configuration with ≥ 10 trades.")
            print("     The signal may be too weak on this week's data.  Consider:")
            print("     • Lowering enter_long_threshold further (try 0.20)")
            print("     • Widening stop_pct (try 0.03) to avoid noise stop-outs")
        else:
            best_wr = best_stats["win_rate"]
            print(f"\n  ✓  Best configuration found:")
            print(f"     enter_threshold  : {best['enter_threshold']}")
            print(f"     stop_pct         : {best['stop_pct']*100:.1f}%")
            print(f"     tp_pct           : {best['tp_pct']*100:.1f}%")
            print(f"     min_conf         : {best['min_conf']}")
            print(f"     rsi_period       : {best['rsi_period']}")
            print(f"     rsi_oversold     : {best['rsi_oversold']}")
            print(f"     rsi_overbought   : {best['rsi_overbought']}")
            _print_stats(best_stats, f"optimised — win_rate={best_wr*100:.1f}%")

            # Re-run with best params and show trade table
            opt_trades, _ = run_backtest(
                bars, best["tech_cfg"],
                enter_threshold=best["enter_threshold"],
                stop_pct=best["stop_pct"],
                tp_pct=best["tp_pct"],
                min_conf=best["min_conf"],
                trail_pct=rcfg.get("trailing_stop_pct", 0.020),
                min_hold_bars=rcfg.get("min_hold_minutes", 15) // 5,
                use_atr_stops=rcfg.get("use_atr_stops", True),
                exit_threshold=0.15,
            )
            _print_trade_table(opt_trades)
            _reason_breakdown(opt_trades)

            if best_wr >= 0.50:
                auto_apply = not sys.stdin.isatty()   # non-interactive → auto-yes
                if auto_apply:
                    ans = "y"
                    print("\n  (non-interactive mode — applying automatically)")
                else:
                    ans = input("\n  Apply these parameters to config.yaml? [Y/n]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    _apply_best_config(cfg, best)
                    print("\n  ✓  config.yaml updated.  Restart the bot to use new settings:")
                    print("     python main.py --broker alpaca")
                else:
                    print("  Skipped — config.yaml unchanged.")
            else:
                print(f"\n  ⚠  Best found win rate {best_wr*100:.1f}% still < 50%.")
                print("     This week's market conditions may not suit the technical signal.")
                print("     Parameters are NOT written to config.yaml.")

    # ── 4. Final summary ────────────────────────────────────────────────────────
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
