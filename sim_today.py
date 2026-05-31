#!/usr/bin/env python3
"""
sim_today.py — Dead-signal scaling intraday simulation: 2026-05-26
==================================================================

Runs TWO parallel simulations over today's 1-min Alpaca bars:

  A) ORIGINAL  — static thresholds (score ≥ 0.50, conf ≥ 0.55)
                 Represents the bot BEFORE the scaling fix.
  B) SCALED    — dynamic dead-signal scaling active after 3 silent cycles
                 ML (w=0.15) + Sentiment (w=0.10) + Fundamental (w=0.10)
                 are absent → ratio = 0.65
                 • Thresholds:    score ≥ 0.325 / conf ≥ 0.358
                 • Position size: kelly 0.385 (was 0.25), max_pos 15.4% (was 10%)

Signal stack (both modes): Technical + ORB + VWAP Bounce + Macro × Regime
Dead signals (not called): ML, Sentiment, Fundamental

Trading rules (both modes):
  - Open buffer +30 min → first entries at 10:00 AM ET
  - Dead zone 12:00–13:30 ET → no new entries
  - Pre-close cutoff 3:20 PM ET → no new entries after this
  - Close buffer -10 min → flatten all at 3:50 PM ET
  - Persistence: 10 consecutive 1-min bars same direction before entry
  - Daily trend filter: price > session open for longs
  - Stop-loss: 2.5%, Take-profit: 6%, Trailing stop: 3%
  - Position size: min(score × conf × Kelly, max_pos_pct) × equity
  - Max 4 concurrent positions (simplified; live=11)
  - Slippage: 5 bps each way

Usage:
    cd "C:\\Users\\yuanp\\OneDrive\\文档\\Claude\\Projects\\DayTradingBot"
    python sim_today.py
"""
from __future__ import annotations

import os, sys
from dataclasses import dataclass, field
from datetime import date, time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from zoneinfo import ZoneInfo
NY = ZoneInfo("America/New_York")

# ── Monkey-patch ORBSignal to use backtest bar time not real clock ──────────
_BT_BAR_TS: Optional[pd.Timestamp] = None   # updated each bar iteration

def _bt_minutes_since_open() -> float:
    """Replacement for ORBSignal._minutes_since_open() — uses bar time."""
    if _BT_BAR_TS is None:
        return 0.0
    ts_ny = _BT_BAR_TS.tz_convert(NY) if _BT_BAR_TS.tzinfo else _BT_BAR_TS.tz_localize("UTC").tz_convert(NY)
    mkt_open = ts_ny.normalize().replace(hour=9, minute=30)
    return max(0.0, (ts_ny - mkt_open).total_seconds() / 60.0)

import src.signals.orb as _orb_mod
_orb_mod._minutes_since_open = _bt_minutes_since_open

# ── Project imports ─────────────────────────────────────────────────────────
from src.signals.aggregator import SignalAggregator
from src.signals.macro import MacroSignal
from src.signals.orb import ORBSignal
from src.signals.regime import RegimeDetector, REGIME_ADJUSTMENTS
from src.signals.technical import TechnicalSignal
from src.signals.vwap_bounce import VWAPBounceSignal


# ── Configuration ────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
scfg = CFG["signals"]
rcfg = CFG["risk"]

TICKERS = [
    "AMD", "TXN", "QCOM", "INTC",
    "PEG", "EXC", "AEP", "D",
    "TJX", "NKE", "TSLA", "BKNG",
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
    "SPY", "QQQ",
]
TRADE_TICKERS = [t for t in TICKERS if t not in ("SPY", "QQQ")]

STARTING_CASH = 99_000.0
SLIPPAGE      = float(CFG["broker"].get("slippage_bps", 5)) / 10_000
STOP_PCT      = float(rcfg["per_trade_stop_loss_pct"])     # 0.025
TP_PCT        = float(rcfg["take_profit_pct"])             # 0.06
TRAIL_PCT     = float(rcfg["trailing_stop_pct"])           # 0.030
KELLY         = float(rcfg["kelly_fraction"])              # 0.25
MAX_POS_PCT   = float(rcfg["max_position_pct"])            # 0.10
MAX_CONC      = 4                                          # simplified

# Safety ceilings for dead-signal upscaling (mirrors risk_manager.py).
_KELLY_SCALE_CAP   = 0.75
_MAX_POS_SCALE_CAP = 0.20
PERSIST_N     = int(scfg["entry_persistence_bars"])        # 10

BASE_EL   = float(scfg["enter_long_threshold"])            # 0.50
BASE_ES   = float(scfg["enter_short_threshold"])           # -0.50
BASE_EXIT = float(scfg["exit_threshold"])                  # 0.25
BASE_MC   = float(scfg["min_confidence"])                  # 0.55
DEAD_N    = int(scfg["dead_signal_cycles"])                # 3

TODAY         = date(2026, 5, 26)
T_START       = dtime(10,  0)   # 9:30 + 30-min open buffer
T_END         = dtime(15, 50)   # EOD flatten
T_DZ_S        = dtime(12,  0)   # dead zone start
T_DZ_E        = dtime(13, 30)   # dead zone end
T_CUTOFF      = dtime(15, 20)   # no new entries after this


# ── Sim data structures ─────────────────────────────────────────────────────
@dataclass
class Position:
    symbol: str
    qty: int
    entry_ts: pd.Timestamp
    entry_price: float
    stop: float
    tp: float
    trail_high: float

@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    reason: str
    pnl: float = field(init=False)
    pnl_pct: float = field(init=False)

    def __post_init__(self):
        self.pnl = (self.exit_price - self.entry_price) * self.qty
        self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price if self.entry_price else 0.0


class SimState:
    def __init__(self, name: str, agg: SignalAggregator):
        self.name = name
        self.agg = agg
        self.cash = STARTING_CASH
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def get_equity(self, prices: Dict[str, float]) -> float:
        mtm = self.cash
        for sym, pos in self.positions.items():
            mtm += pos.qty * prices.get(sym, pos.entry_price)
        return mtm

    def manage_positions(self, ts: pd.Timestamp, prices: Dict[str, float]) -> None:
        for sym in list(self.positions):
            pos = self.positions[sym]
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            # Update trailing stop
            if price > pos.trail_high:
                pos.trail_high = price
                pos.stop = max(pos.stop, price * (1 - TRAIL_PCT))
            # Check exits
            if price <= pos.stop:
                self._close(sym, ts, price * (1 - SLIPPAGE), "stop_loss")
            elif price >= pos.tp:
                self._close(sym, ts, price * (1 + SLIPPAGE), "take_profit")

    def enter(self, sym: str, ts: pd.Timestamp, price: float,
              score: float, confidence: float,
              signal_ratio: float = 1.0) -> bool:
        """Attempt to enter a long position.

        signal_ratio: active-weight ratio from the aggregator (1.0 = all live).
        When < 1.0, kelly and max_pos are scaled up by 1/ratio to compensate
        for structurally-suppressed confidence (mirrors RiskManager.check_entry).
        """
        if sym in self.positions or len(self.positions) >= MAX_CONC:
            return False
        sig_ratio  = max(1e-6, min(1.0, signal_ratio))
        eff_kelly   = min(KELLY       / sig_ratio, _KELLY_SCALE_CAP)
        eff_max_pos = min(MAX_POS_PCT / sig_ratio, _MAX_POS_SCALE_CAP)
        target_pct  = min(abs(score) * confidence * eff_kelly, eff_max_pos)
        qty = int(self.cash * target_pct / (price * (1 + SLIPPAGE)))
        if qty < 1:
            return False
        fill = price * (1 + SLIPPAGE)
        cost = qty * fill
        if cost > self.cash:
            qty = int(self.cash / fill)
        if qty < 1:
            return False
        self.cash -= qty * fill
        self.positions[sym] = Position(
            symbol=sym, qty=qty, entry_ts=ts, entry_price=fill,
            stop=fill * (1 - STOP_PCT), tp=fill * (1 + TP_PCT),
            trail_high=fill,
        )
        return True

    def flatten_all(self, ts: pd.Timestamp, prices: Dict[str, float]) -> None:
        for sym in list(self.positions):
            price = prices.get(sym, self.positions[sym].entry_price)
            self._close(sym, ts, price * (1 - SLIPPAGE), "eod_flatten")

    def _close(self, sym: str, ts: pd.Timestamp, fill: float, reason: str) -> None:
        pos = self.positions.pop(sym, None)
        if pos is None:
            return
        self.cash += pos.qty * fill
        self.trades.append(Trade(
            symbol=sym, entry_ts=pos.entry_ts, exit_ts=ts,
            entry_price=pos.entry_price, exit_price=fill,
            qty=pos.qty, reason=reason,
        ))

    # ── Reporting ──────────────────────────────────────────────────────────
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    def final_equity(self) -> float:
        return self.cash  # positions are flattened at EOD

    def ret_pct(self) -> float:
        return (self.final_equity() - STARTING_CASH) / STARTING_CASH * 100


# ── Data fetch ───────────────────────────────────────────────────────────────
def fetch_alpaca(tickers, start_str, end_str) -> Dict[str, pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    import datetime as dt

    key    = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not (key and secret):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not in .env")

    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=dt.datetime.fromisoformat(start_str),
        end=dt.datetime.fromisoformat(end_str),
        feed=DataFeed.IEX,
    )
    print(f"  Fetching 1-min bars ({start_str} → {end_str}) for {len(tickers)} tickers...")
    resp = client.get_stock_bars(req)
    df_all = resp.df
    if df_all is None or df_all.empty:
        raise RuntimeError("Alpaca returned no data")

    bars = {}
    got = df_all.index.get_level_values(0).unique().tolist()
    for sym in tickers:
        if sym not in got:
            print(f"    WARNING: no bars for {sym}")
            continue
        sub = df_all.loc[sym].copy()
        sub.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"}, inplace=True)
        bars[sym] = sub[["Open", "High", "Low", "Close", "Volume"]]
    return bars


# ── Helpers ──────────────────────────────────────────────────────────────────
def ts_ny(ts: pd.Timestamp) -> dtime:
    """Bar timestamp → NY time."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(NY).time()


def can_enter(t: dtime) -> bool:
    return T_START <= t <= T_CUTOFF and not (T_DZ_S <= t < T_DZ_E)


def persistence_ok(hist: list, n: int) -> bool:
    """Last n scores all same (positive) direction."""
    return len(hist) >= n and all(s > 0 for s in hist[-n:])


# ── Main simulation ──────────────────────────────────────────────────────────
def run_simulation(bars: Dict[str, pd.DataFrame]) -> None:
    # ── Signal engines ──────────────────────────────────────────────────────
    tech_sig  = TechnicalSignal(scfg["technical"])
    orb_sig   = ORBSignal(scfg["orb"])
    vwap_sig  = VWAPBounceSignal(scfg["vwap_bounce"])
    macro_eng = MacroSignal()
    regime_d  = RegimeDetector()

    # ── Two aggregators ──────────────────────────────────────────────────────
    agg_orig = SignalAggregator(
        weights=scfg["weights"],
        enter_long=BASE_EL, enter_short=BASE_ES,
        exit_thresh=BASE_EXIT, min_confidence=BASE_MC,
        dead_signal_cycles=99999,   # never scales — "original" behavior
    )
    agg_scaled = SignalAggregator(
        weights=scfg["weights"],
        enter_long=BASE_EL, enter_short=BASE_ES,
        exit_thresh=BASE_EXIT, min_confidence=BASE_MC,
        dead_signal_cycles=DEAD_N,  # scales after 3 silent cycles
    )

    state_A = SimState("ORIGINAL (fixed thresholds)", agg_orig)
    state_B = SimState("SCALED   (dead-signal scaling)", agg_scaled)

    # ── Build today's bar timeline ───────────────────────────────────────────
    # We need bars from all tickers to find timestamps where SPY was active.
    spy_idx = bars.get("SPY", pd.DataFrame()).index
    if spy_idx.empty:
        print("ERROR: No SPY bars found — cannot build timeline.")
        return

    # Filter to today's session bars (9:30–16:00 ET)
    spy_ny = spy_idx.tz_convert(NY)
    today_mask = (spy_ny.date == TODAY)
    today_ts = spy_idx[today_mask]
    if today_ts.empty:
        print(f"ERROR: No SPY bars for {TODAY}")
        return

    print(f"\n  Today's bars: {len(today_ts)} 1-min bars "
          f"({ts_ny(today_ts[0])} – {ts_ny(today_ts[-1])} ET)")

    # Shared signal history (scores same in both modes; only thresholds differ)
    signal_hist: Dict[str, list] = {t: [] for t in TRADE_TICKERS}
    day_open:    Dict[str, float] = {}

    # Metadata collection for report
    regime_log = []
    threshold_log = []   # (bar_index, eff_el_scaled, eff_mc_scaled, ratio)
    signal_detail = []   # per-bar (ts, sym, score, conf, action_orig, action_scaled)
    entries_A, entries_B = 0, 0
    blocked_by_conf_A, blocked_by_conf_B = 0, 0
    blocked_by_score_A, blocked_by_score_B = 0, 0
    blocked_by_persist_A, blocked_by_persist_B = 0, 0

    MIN_WARMUP = 120    # need ≥120 bars for MACD-105, EMA-105

    flattened = False
    last_regime = None
    bar_count = 0

    print("\n  Running bar-by-bar simulation...")
    for bar_i, ts in enumerate(today_ts):
        global _BT_BAR_TS
        _BT_BAR_TS = ts

        t = ts_ny(ts)
        in_session = t >= T_START
        at_end     = t >= T_END
        can_enter_now = can_enter(t)

        # ── Compute signals ─────────────────────────────────────────────────
        all_sigs = []
        for sym in TRADE_TICKERS:
            sym_bars = bars.get(sym)
            if sym_bars is None or ts not in sym_bars.index:
                continue
            window = sym_bars.loc[:ts]
            if len(window) < MIN_WARMUP:
                continue

            # Record session open price (first bar of today where we have data)
            if sym not in day_open:
                day_mask = window.index.tz_convert(NY).date == TODAY
                today_w = window[day_mask]
                if not today_w.empty:
                    day_open[sym] = float(today_w.iloc[0]["Open"])

            t_s = tech_sig.evaluate(sym, window)
            o_s = orb_sig.evaluate(sym, window)
            v_s = vwap_sig.evaluate(sym, window)
            for sig in (t_s, o_s, v_s):
                if sig is not None:
                    all_sigs.append(sig)

        if not all_sigs:
            continue

        # ── Macro multiplier ────────────────────────────────────────────────
        spy_w = bars["SPY"].loc[:ts] if "SPY" in bars else pd.DataFrame()
        mkt_mult, _ = macro_eng.evaluate(spy_w) if len(spy_w) >= 20 else (1.0, "")

        # ── Regime detection ────────────────────────────────────────────────
        regime_name, regime_meta = regime_d.detect(spy_w)
        weight_overrides = REGIME_ADJUSTMENTS.get(regime_name, {}).get("weight_overrides", {})
        if regime_name != last_regime and in_session:
            regime_log.append((ts, regime_name, regime_meta))
            last_regime = regime_name

        # ── Aggregate (BOTH modes) ──────────────────────────────────────────
        decs_orig   = agg_orig.aggregate(all_sigs,   mkt_mult, weight_overrides)
        decs_scaled = agg_scaled.aggregate(all_sigs, mkt_mult, weight_overrides)

        # Record threshold state for scaled mode
        ratio = agg_scaled.threshold_ratio
        dead  = agg_scaled.dead_sources
        # Peek at effective thresholds from any decision (they're all the same)
        sample_dec = next(iter(decs_scaled.values()), None)
        eff_el = sample_dec.enter_long if sample_dec else BASE_EL
        eff_mc = sample_dec.min_confidence if sample_dec else BASE_MC
        if in_session:
            threshold_log.append((bar_i, eff_el, eff_mc, ratio, sorted(dead)))
            bar_count += 1

        # ── Update shared signal history ─────────────────────────────────────
        for sym in TRADE_TICKERS:
            d = decs_orig.get(sym) or decs_scaled.get(sym)
            if d:
                h = signal_hist[sym]
                h.append(d.score)
                if len(h) > PERSIST_N * 2:
                    h.pop(0)

        # ── Get current prices ───────────────────────────────────────────────
        prices: Dict[str, float] = {}
        for sym in TRADE_TICKERS:
            sb = bars.get(sym)
            if sb is not None and ts in sb.index:
                prices[sym] = float(sb.at[ts, "Close"])

        # ── EOD flatten ──────────────────────────────────────────────────────
        if at_end and not flattened:
            state_A.flatten_all(ts, prices)
            state_B.flatten_all(ts, prices)
            flattened = True

        # ── Manage open positions ────────────────────────────────────────────
        state_A.manage_positions(ts, prices)
        state_B.manage_positions(ts, prices)

        # ── Equity tracking ──────────────────────────────────────────────────
        eq_A = state_A.get_equity(prices)
        eq_B = state_B.get_equity(prices)
        state_A.equity_curve.append(eq_A)
        state_B.equity_curve.append(eq_B)

        # ── New entries ──────────────────────────────────────────────────────
        if not can_enter_now:
            continue

        for sym in TRADE_TICKERS:
            if sym in state_A.positions and sym in state_B.positions:
                continue
            price = prices.get(sym)
            if not price or price <= 0:
                continue

            d_orig   = decs_orig.get(sym)
            d_scaled = decs_scaled.get(sym)
            if d_orig is None:
                continue

            score = d_orig.score
            conf  = d_orig.confidence    # same in both (only thresholds differ)

            persist = persistence_ok(signal_hist.get(sym, []), PERSIST_N)
            do = day_open.get(sym)
            trend_ok = (do is None) or (price >= do)

            # ── Mode A: original — fixed thresholds, unscaled sizing ──────────
            if sym not in state_A.positions:
                if d_orig.action == "BUY":
                    if persist and trend_ok:
                        # signal_ratio=1.0 → no sizing adjustment
                        if state_A.enter(sym, ts, price, score, conf,
                                         signal_ratio=1.0):
                            entries_A += 1
                    elif not persist:
                        blocked_by_persist_A += 1
                else:
                    if conf < BASE_MC:
                        blocked_by_conf_A += 1
                    elif score < BASE_EL:
                        blocked_by_score_A += 1

            # ── Mode B: scaled — lower thresholds + upscaled sizing ──────────
            if sym not in state_B.positions and d_scaled is not None:
                if d_scaled.action == "BUY":
                    if persist and trend_ok:
                        # Pass the current ratio so SimState.enter() applies the
                        # same kelly / max_pos upscaling as RiskManager.check_entry().
                        if state_B.enter(sym, ts, price, score, conf,
                                         signal_ratio=ratio):
                            entries_B += 1
                    elif not persist:
                        blocked_by_persist_B += 1
                else:
                    if conf < d_scaled.min_confidence:
                        blocked_by_conf_B += 1
                    elif score < d_scaled.enter_long:
                        blocked_by_score_B += 1

    # ── Results ─────────────────────────────────────────────────────────────
    SEP = "=" * 72

    print(f"\n{SEP}")
    print(f"  INTRADAY SIMULATION  |  {TODAY}  |  Tickers: {len(TRADE_TICKERS)}")
    print(SEP)

    # Regime summary
    if regime_log:
        r_ts, r_name, r_meta = regime_log[0]
        print(f"\n  Market regime:  {r_name.upper()}")
        print(f"  SPY vs SMA20:   {r_meta.get('spy_vs_sma20_pct', 0)*100:+.2f}%")
        print(f"  SPY momentum5:  {r_meta.get('spy_momentum5', 0)*100:+.2f}%")
        print(f"  VIX:            {r_meta.get('vix', 20):.1f}")
        if weight_overrides:
            print(f"  Weight override: {weight_overrides}")

    # Threshold progression
    if threshold_log:
        first_dead_set = next((row for row in threshold_log if row[4]), None)

        print(f"\n  Dead-signal scaling:")
        print(f"    Base thresholds:    score ≥ {BASE_EL:.3f}  |  conf ≥ {BASE_MC:.3f}")
        print(f"    Base sizing:        kelly = {KELLY:.3f}  |  max_pos = {MAX_POS_PCT*100:.1f}%")
        if first_dead_set:
            bar_i, el, mc, r, dead_s = first_dead_set
            eff_kelly_rep   = min(KELLY       / r, _KELLY_SCALE_CAP)
            eff_max_pos_rep = min(MAX_POS_PCT / r, _MAX_POS_SCALE_CAP)
            print(f"    Dead sources:       {dead_s}  (detected at session bar {bar_i})")
            print(f"    Active weight:      {r*100:.0f}%  (ratio = {r:.3f})")
            print(f"    Scaled thresholds:  score ≥ {el:.3f}  |  conf ≥ {mc:.3f}")
            print(f"    Scaled sizing:      kelly = {eff_kelly_rep:.3f}  |  max_pos = {eff_max_pos_rep*100:.1f}%")
        else:
            print("    No dead sources detected (all signals active).")

    # Macro multiplier (sample from first in-session bar)
    if mkt_mult != 1.0:
        print(f"\n  Macro multiplier: {mkt_mult:.2f}x (depresses all scores)")

    print(f"\n{'─'*72}")
    print(f"  {'Metric':<30}  {'ORIGINAL':>14}  {'SCALED':>14}")
    print(f"{'─'*72}")

    def row(label, a, b, fmt="{}", suffix=""):
        la = fmt.format(a) + suffix
        lb = fmt.format(b) + suffix
        print(f"  {label:<30}  {la:>14}  {lb:>14}")

    row("Starting equity",    f"${STARTING_CASH:,.0f}", f"${STARTING_CASH:,.0f}")
    row("Ending equity",      f"${state_A.final_equity():,.2f}", f"${state_B.final_equity():,.2f}")
    row("Total return",       f"{state_A.ret_pct():+.3f}%", f"{state_B.ret_pct():+.3f}%")
    row("Total P&L",          f"${state_A.total_pnl():+.2f}", f"${state_B.total_pnl():+.2f}")
    row("Trades entered",     entries_A, entries_B)
    row("Trades closed",      len(state_A.trades), len(state_B.trades))
    row("Win rate",           f"{state_A.win_rate()*100:.0f}%", f"{state_B.win_rate()*100:.0f}%")
    row("Blocked: conf gate", blocked_by_conf_A, blocked_by_conf_B)
    row("Blocked: score gate",blocked_by_score_A, blocked_by_score_B)
    row("Blocked: persistence",blocked_by_persist_A, blocked_by_persist_B)
    print(f"{'─'*72}")

    # Trade details for mode B
    if state_B.trades:
        print(f"\n  SCALED MODE — trade log ({len(state_B.trades)} closed trades):")
        print(f"  {'Sym':<6} {'Entry time':<10} {'Exit time':<10} "
              f"{'Entry':>8} {'Exit':>8} {'Qty':>5} {'P&L':>9} {'Reason':<16}")
        print(f"  {'-'*80}")
        for t in sorted(state_B.trades, key=lambda x: x.entry_ts):
            e_t = t.entry_ts.tz_convert(NY).strftime("%H:%M")
            x_t = t.exit_ts.tz_convert(NY).strftime("%H:%M")
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(f"  {t.symbol:<6} {e_t:<10} {x_t:<10} "
                  f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
                  f"{t.qty:>5} {pnl_sign}${t.pnl:>7.2f}  {t.reason}")
        wins  = [t for t in state_B.trades if t.pnl > 0]
        losses= [t for t in state_B.trades if t.pnl <= 0]
        print(f"\n  Winners: {len(wins)} avg +${np.mean([t.pnl for t in wins]):.2f}" if wins else "  Winners: 0")
        print(f"  Losers:  {len(losses)} avg -${abs(np.mean([t.pnl for t in losses])):.2f}" if losses else "  Losers:  0")

    if state_A.trades:
        print(f"\n  ORIGINAL MODE — trade log ({len(state_A.trades)} closed trades):")
        for t in sorted(state_A.trades, key=lambda x: x.entry_ts):
            e_t = t.entry_ts.tz_convert(NY).strftime("%H:%M")
            x_t = t.exit_ts.tz_convert(NY).strftime("%H:%M")
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(f"  {t.symbol:<6} {e_t:<10} {x_t:<10} "
                  f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
                  f"{t.qty:>5} {pnl_sign}${t.pnl:>7.2f}  {t.reason}")

    # Score distribution snapshot for SCALED mode
    print(f"\n  Top signal scores for SCALED mode at last bar:")
    print(f"  {'Sym':<6} {'Score':>7} {'Conf':>7} {'Action':>8} {'Gate clears'}")
    print(f"  {'-'*55}")
    last_decs = decs_scaled
    sorted_syms = sorted(
        [(sym, d) for sym, d in last_decs.items() if sym in TRADE_TICKERS],
        key=lambda x: x[1].score, reverse=True
    )[:10]
    for sym, d in sorted_syms:
        clears = []
        if d.confidence >= d.min_confidence: clears.append("conf✓")
        if d.score >= d.enter_long:          clears.append("score✓")
        print(f"  {sym:<6} {d.score:>7.3f} {d.confidence:>7.3f} {d.action:>8}  {' '.join(clears) or '—'}")

    print(f"\n{SEP}")
    print("  Key takeaway:")
    if entries_A == 0 and entries_B > 0:
        print(f"  • ORIGINAL mode:  0 entries — min_confidence gate ({BASE_MC:.2f}) is unreachable")
        print(f"    when ML+Sentiment+Fundamental are offline (max achievable conf ≈ 0.37).")
        print(f"  • SCALED mode:   {entries_B} entries — after 3 silent cycles, thresholds")
        first_dead_set_local = next((row for row in threshold_log if row[4]), None)
        if first_dead_set_local:
            _, el, mc, ratio, _ = first_dead_set_local
            print(f"    scaled to score≥{el:.3f} / conf≥{mc:.3f} (ratio={ratio:.2f}),")
        print(f"    unlocking trades that would otherwise be perpetually blocked.")
    elif entries_A == 0 and entries_B == 0:
        print(f"  • Neither mode fired any entries today — market conditions too weak")
        print(f"    (check persistence filter, dead zone coverage, trend filter).")
    elif entries_A > 0:
        print(f"  • Original mode ALSO fired {entries_A} entries — signals were strong today.")
    print(f"{SEP}\n")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*72}")
    print(f"  DayTradingBot — Dead-Signal Scaling Simulation")
    print(f"  Date: {TODAY}  |  Interval: 1-min  |  Tickers: {len(TICKERS)}")
    print(f"{'='*72}")

    # Fetch bars: 10 trading days of history for indicator warmup
    print("\n[1/3] Fetching Alpaca 1-min bars (10-day lookback for indicator warmup)...")
    try:
        bars = fetch_alpaca(
            tickers=TICKERS,
            start_str="2026-05-12T09:00:00",   # ~10 trading days before today
            end_str="2026-05-26T20:00:00",      # today EOD
        )
    except Exception as e:
        print(f"  Alpaca fetch failed: {e}")
        print("  Trying yfinance fallback...")
        import yfinance as yf
        bars = {}
        for sym in TICKERS:
            df = yf.download(sym, start="2026-05-12", end="2026-05-27",
                             interval="1m", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                bars[sym] = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
        if not bars:
            print("  ERROR: Could not fetch any bars. Aborting.")
            return

    print(f"  Loaded bars for: {sorted(bars.keys())}")
    total_bars = sum(len(v) for v in bars.values())
    print(f"  Total bars loaded: {total_bars:,}")

    print("\n[2/3] Running simulation...")
    run_simulation(bars)

    print("[3/3] Done.")


if __name__ == "__main__":
    main()
