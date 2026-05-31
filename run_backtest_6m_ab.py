"""6-month walk-forward backtest — 3-way comparison.

Bar interval  : 5-minute  (~10k bars/symbol, runs in ~8 min)
Lookback      : 180 calendar days  ≈ 6 months  ≈ 130 trading days
Walk-forward  : train first 50 % (≈ 3 months), test last 50 % (≈ 3 months)

Strategy A — BEST (fixed direction labels + dynamic exclusion + tuned exits):
  • LightGBM GPU  + fixed-horizon direction labels  + 14 features
  • Dynamic exclusion (< 25 % win rate, 5 consecutive bad days, min 3 trades/day)
  • No agreement gate
  • min_hold=45  |  max_hold=480  |  EOD@close (5-min buffer)

Strategy B — OLD baseline (nothing new):
  • LightGBM GPU  + fixed-horizon direction labels  + 14 features
  • No dynamic exclusion  |  No agreement gate
  • min_hold=30  |  max_hold=9999 (none)  |  EOD@close (0-min buffer)

Strategy C — TRIPLE-BARRIER (kept for reference, expected to underperform):
  • LightGBM GPU  + triple-barrier labels  + 14 features
  • Dynamic exclusion  |  No agreement gate
  • min_hold=45  |  max_hold=480  |  EOD@close (5-min buffer)
"""
from __future__ import annotations

import copy
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.backtest.backtester import Backtester, BacktestResult
from src.data.market_data import MarketData
from src.signals.finrl_signal import FinRLSignal
from src.signals.ml_model import MLSignal

# ── Parameters ────────────────────────────────────────────────────────────────

LOOKBACK_DAYS = 180        # calendar days (~6 months)
INTERVAL      = "5m"       # 5-min bars — feasible for 6-month window
FINRL_STEPS   = 150_000    # PPO steps (shared between A and B)
SLIPPAGE_BPS  = 5

TICKERS = [
    "CRM", "ADBE", "MSFT", "AVGO", "AMD",
    "VMC", "NUE", "AAPL", "NVDA", "TSLA",
    "META", "GOOGL", "AMZN", "SPY", "QQQ",
]

with open(ROOT / "config.yaml") as f:
    BASE_CFG = yaml.safe_load(f)

STARTING_CASH = float(BASE_CFG["broker"].get("starting_cash", 10_000))

# ── Build old/new configs ─────────────────────────────────────────────────────

def _best_cfg(base: dict) -> dict:
    """Best strategy: all improvements including short selling on the tuned universe.

      • Fixed direction labels + 14 features
      • Dynamic exclusion (win_rate<25%, streak≥5, min 3 trades/day)
      • Partial profit at +1% (50%) and +2.5% (50% of remainder)
      • Trailing stop at 3% below/above high/low watermark after PP1
      • Breakeven stop move at +2.2%
      • Short selling enabled — macro amplifies shorts in bear regimes
      • min_hold=45  |  max_hold=480  |  EOD@close (5-min buffer)
    """
    cfg = copy.deepcopy(base)
    s = cfg["signals"]
    r = cfg["risk"]
    sc = cfg["schedule"]
    s["require_ml_finrl_agreement"]      = False
    s["ml"]["label_method"]              = "fixed"
    s["enter_short_threshold"]           = -0.45
    r["min_hold_minutes"]                = 45
    r["max_hold_minutes"]                = 480
    r["max_symbol_daily_losses"]         = 2
    r["dynamic_exclusion_win_rate"]      = 0.25
    r["dynamic_exclusion_streak_days"]   = 5
    r["partial_profit_1_pct"]            = 0.01
    r["partial_profit_2_pct"]            = 0.025
    r["trailing_stop_pct"]               = 0.03
    r["breakeven_trigger_pct"]           = 0.022
    r["shorting_enabled"]                = False   # disabled until stock-level trend filter added
    sc["no_entry_before_close_minutes"]  = 5
    return cfg


def _old_cfg(base: dict) -> dict:
    """Old baseline — fixed labels, no exclusion, no hold caps, no partial profit, longs only."""
    cfg = copy.deepcopy(base)
    s = cfg["signals"]
    r = cfg["risk"]
    sc = cfg["schedule"]
    s["require_ml_finrl_agreement"]      = False
    s["ml"]["label_method"]              = "fixed"
    r["min_hold_minutes"]                = 30
    r["max_hold_minutes"]                = 9999
    r["max_symbol_daily_losses"]         = 2
    r["dynamic_exclusion_win_rate"]      = 0.33
    r["dynamic_exclusion_streak_days"]   = 9999   # effectively disabled
    r["partial_profit_1_pct"]            = 9999.0  # unreachable
    r["partial_profit_2_pct"]            = 9999.0  # unreachable
    r["breakeven_trigger_pct"]           = 9999.0  # unreachable
    r["shorting_enabled"]                = False   # longs only
    sc["no_entry_before_close_minutes"]  = 0
    return cfg


def _triple_cfg(base: dict) -> dict:
    """Triple-barrier variant — kept for reference."""
    cfg = copy.deepcopy(base)
    s = cfg["signals"]
    r = cfg["risk"]
    sc = cfg["schedule"]
    s["require_ml_finrl_agreement"]      = False
    s["ml"]["label_method"]              = "triple_barrier"
    s["ml"]["tp_barrier_pct"]            = 0.004
    s["ml"]["sl_barrier_pct"]            = 0.003
    r["min_hold_minutes"]                = 45
    r["max_hold_minutes"]                = 480
    r["max_symbol_daily_losses"]         = 2
    r["dynamic_exclusion_win_rate"]      = 0.33
    r["dynamic_exclusion_streak_days"]   = 3
    sc["no_entry_before_close_minutes"]  = 5
    return cfg


CFG_BEST   = _best_cfg(BASE_CFG)
CFG_OLD    = _old_cfg(BASE_CFG)
CFG_TRIPLE = _triple_cfg(BASE_CFG)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _summary(result: BacktestResult, trades) -> dict:
    from collections import defaultdict
    # Group fills by (symbol, entry_time) so each ENTRY counts as one
    # win/loss record even when partial-profit splits it into multiple fills.
    groups: dict = defaultdict(list)
    for t in trades:
        groups[(t.symbol, t.entry_time)].append(t)

    wins, losses = [], []
    for group in groups.values():
        if sum(t.pnl for t in group) > 0:
            wins.append(group)
        else:
            losses.append(group)

    pf = (
        abs(sum(sum(t.pnl for t in g) for g in wins))
        / abs(sum(sum(t.pnl for t in g) for g in losses))
        if losses else 999.0
    )
    avg_w = float(np.mean([sum(t.pnl_pct for t in g) * 100 for g in wins]))   if wins   else 0.0
    avg_l = float(np.mean([sum(t.pnl_pct for t in g) * 100 for g in losses])) if losses else 0.0
    n_entries = len(groups)

    return {
        "net_pnl":    result.ending_cash - result.starting_cash,
        "return_pct": result.total_return * 100,
        "n_trades":   n_entries,
        "n_fills":    len(trades),
        "win_rate":   len(wins) / n_entries * 100 if n_entries else 0.0,
        "avg_win":    avg_w,
        "avg_loss":   avg_l,
        "pf":         pf,
        "sharpe":     result.sharpe,
        "exits":      Counter(t.reason for t in trades),
        "by_sym":     _by_sym(groups),
    }


def _by_sym(groups) -> dict:
    d: dict = {}
    for (sym, _), fills in groups.items():
        pnl = sum(t.pnl for t in fills)
        d.setdefault(sym, []).append(pnl)
    return d


def _print_result(label: str, s: dict, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Net P&L      : ${s['net_pnl']:>+9,.2f}  ({s['return_pct']:+.2f}%)")
    print(f"  Entries      : {s['n_trades']}  ({s.get('n_fills', s['n_trades'])} fills incl. partials)")
    print(f"  Win rate     : {s['win_rate']:.1f}%")
    print(f"  Avg win      : {s['avg_win']:+.2f}%")
    print(f"  Avg loss     : {s['avg_loss']:+.2f}%")
    print(f"  Profit factor: {s['pf']:.2f}x")
    print(f"  Sharpe ratio : {s['sharpe']:.2f}")
    print(f"  Exit reasons :")
    for r, n in s["exits"].most_common():
        pct = n / s["n_trades"] * 100 if s["n_trades"] else 0
        print(f"    {r:<22s} {n:3d} ({pct:.0f}%)")
    print(f"  Per-symbol   :")
    for sym, pnls in sorted(s["by_sym"].items(), key=lambda x: -sum(x[1])):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {sym:<6s} {len(pnls):2d} trades  {w}/{len(pnls)} wins  ${sum(pnls):+.2f}")
    print(bar)


def _print_ab_comparison(new: dict, old: dict) -> None:
    width = 70
    bar   = "=" * width
    dash  = "-" * width
    print(f"\n\n{bar}")
    print(f"  A/B COMPARISON  —  NEW strategy vs OLD strategy")
    print(bar)
    print(f"  {'Metric':<28} {'NEW':>15} {'OLD':>15}")
    print(dash)

    def row(label, nv, ov, fmt="{:.2f}"):
        nf = fmt.format(nv) if nv is not None else "—"
        of = fmt.format(ov) if ov is not None else "—"
        marker = " ✓" if (isinstance(nv, (int, float)) and isinstance(ov, (int, float)) and nv > ov) else ""
        print(f"  {label:<28} {nf:>15} {of:>15}{marker}")

    row("Net P&L ($)",        new["net_pnl"],    old["net_pnl"],    fmt="${:+,.2f}")
    row("Return (%)",         new["return_pct"], old["return_pct"], fmt="{:+.2f}%")
    row("Entries",            new["n_trades"],   old["n_trades"],   fmt="{:d}")
    row("Win rate (%)",       new["win_rate"],   old["win_rate"],   fmt="{:.1f}%")
    row("Avg win (%)",        new["avg_win"],    old["avg_win"],    fmt="{:+.2f}%")
    row("Avg loss (%)",       new["avg_loss"],   old["avg_loss"],   fmt="{:+.2f}%")  # higher (less negative) = better
    row("Profit factor",      new["pf"],         old["pf"],         fmt="{:.2f}x")
    row("Sharpe ratio",       new["sharpe"],     old["sharpe"],     fmt="{:.2f}")
    print(bar)
    print("  ✓ = NEW is better\n")


# ── Main ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  DayTradingBot 6-month A/B backtest")
print(f"  Interval: {INTERVAL}  |  Lookback: {LOOKBACK_DAYS} cal days")
print(f"  Cash: ${STARTING_CASH:,.0f}   Slippage: {SLIPPAGE_BPS} bps")
print(f"{'='*60}\n")

# Step 1 — Fetch data
print(f"Fetching {INTERVAL} bars ({LOOKBACK_DAYS} calendar days) …")
md = MarketData(
    provider=BASE_CFG["data"].get("provider", "alpaca"),
    interval=INTERVAL,
    lookback_days=LOOKBACK_DAYS,
    feed=BASE_CFG["data"].get("feed", "iex"),
)
bars_by_sym: dict = {}
for sym in TICKERS:
    df = md.get_bars(sym)
    if df is not None and not df.empty:
        bars_by_sym[sym] = df
        print(f"  {sym:<6s}  {len(df):6d} bars  "
              f"{df.index[0].strftime('%m/%d/%y')} → {df.index[-1].strftime('%m/%d/%y')}")

# Step 2 — Walk-forward split
print(f"\nWalk-forward split: train on first 50%, test on last 50%")
train_bars, test_bars = {}, {}
for sym, df in bars_by_sym.items():
    n = len(df)
    s = n // 2
    if s >= 200:
        train_bars[sym] = df.iloc[:s]
    if n - s >= 100:
        test_bars[sym]  = df.iloc[s:]

print(f"  Train symbols: {len(train_bars)}   Test symbols: {len(test_bars)}")
if train_bars:
    sample = next(iter(train_bars.values()))
    print(f"  Train period : {sample.index[0].strftime('%m/%d/%y')} → {sample.index[-1].strftime('%m/%d/%y')}")
if test_bars:
    sample = next(iter(test_bars.values()))
    print(f"  Test period  : {sample.index[0].strftime('%m/%d/%y')} → {sample.index[-1].strftime('%m/%d/%y')}")

# Step 3 — Train shared ML model (fixed labels, used by both strategies)
print(f"\nTraining ML model (LightGBM + fixed-horizon direction labels) …")
ml_fixed = MLSignal(CFG_BEST["signals"]["ml"])
ml_fixed.train(train_bars)

# Step 4 — Train shared FinRL model
print(f"Training FinRL PPO agent (shared, {FINRL_STEPS:,} steps) …")
finrl_cfg = dict(BASE_CFG["signals"].get("finrl", {}))
finrl_cfg["total_timesteps"] = FINRL_STEPS
finrl = FinRLSignal(finrl_cfg)
finrl.train(train_bars)

# Step 5 — Run BEST strategy
print(f"\nRunning backtest — BEST (all improvements + short selling) …")
bt_best = Backtester(config=CFG_BEST, starting_cash=STARTING_CASH,
                     slippage_bps=SLIPPAGE_BPS, interval=INTERVAL)
bt_best.ml = ml_fixed; bt_best.finrl = finrl
result_best = bt_best.run(test_bars)
sum_best    = _summary(result_best, result_best.trades)

# Step 6 — Run OLD strategy
print(f"Running backtest — OLD (plain baseline, longs only) …")
bt_old = Backtester(config=CFG_OLD, starting_cash=STARTING_CASH,
                    slippage_bps=SLIPPAGE_BPS, interval=INTERVAL)
bt_old.ml = ml_fixed; bt_old.finrl = finrl
result_old = bt_old.run(test_bars)
sum_old    = _summary(result_old, result_old.trades)

# Step 7 — Print results
_print_result("BEST  (dyn exclusion · partial profit · trailing stop · SHORTS · EOD@close)", sum_best)
_print_result("OLD   (no exclusion  · no partial profit · longs only  · EOD@close)", sum_old)
_print_ab_comparison(sum_best, sum_old)
