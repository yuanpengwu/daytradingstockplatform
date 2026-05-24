#!/usr/bin/env python3
"""
run_backtest_fast.py  — 30-day portfolio backtest with full ensemble signals.

Tests 6 targeted parameter sets instead of a full grid search.
Uses Alpaca IEX data.  Includes Tech + ORB + VWAP-Bounce signals.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import time as dtime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import numpy as np
import yaml

# Import the portfolio backtest engine from backtest.py
from backtest import (
    fetch_bars, run_backtest,
    _print_stats, _print_trade_table, _reason_breakdown, _sector_breakdown,
)

def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scfg     = cfg.get("signals", {})
    rcfg     = cfg.get("risk", {})
    tech_cfg = scfg.get("technical", {})
    orb_cfg  = scfg.get("orb", {})
    vwap_cfg = scfg.get("vwap_bounce", {})

    # Full live universe (Tech XLK + Financials XLF + fallback)
    tickers = [
        "QCOM","AMD","INTC","CRM","TXN",
        "MS","GS","BAC","WFC","C",
        "AAPL","MSFT","NVDA","TSLA","META","GOOGL","AMZN",
        "SPY","QQQ",
    ]

    BAR_MINUTES = 1   # ← 1-min bars (finest resolution Alpaca free tier supports)

    print(f"Fetching 21-day {BAR_MINUTES}-min Alpaca bars (~3 trading weeks) ...")
    bars = fetch_bars(tickers, days=21, bar_minutes=BAR_MINUTES)
    if not bars:
        sys.exit("No data returned.")

    all_dates = sorted({str(d.date()) for df in bars.values() for d in df.index})
    print(f"\nData: {len(bars)} symbols | dates: {', '.join(all_dates)}")
    print(f"Total bars: {sum(len(v) for v in bars.values()):,}\n")

    # ── Common filter settings — mirrors live config (scaled for 1-min bars) ──
    FILTERS = dict(
        use_dead_zone    = True,
        dead_zone_start  = dtime(12,  0),
        dead_zone_end    = dtime(13, 30),
        pre_close_cutoff = dtime(15, 20),
        use_rs_filter    = True,
        rs_min           = scfg.get("relative_strength_min", 0.004),
        rs_lookback      = scfg.get("rs_lookback_bars", 30),    # 30 × 1min = 30 min
        use_sector_limit = True,
        max_per_sector   = rcfg.get("max_positions_per_sector", 2),
        use_regime       = True,
        max_concurrent   = rcfg.get("max_concurrent_positions", 6),
        open_buffer_bars = 30,                                   # 30 × 1min = 30 min open buffer
        use_atr_stops    = True,
        atr_stop_mult    = rcfg.get("atr_stop_multiplier", 4.5),
        atr_tp_mult      = rcfg.get("atr_tp_multiplier", 6.0),
        use_next_day_cooloff  = True,
        use_prev_close_filter = False,
        orb_cfg          = orb_cfg,
        vwap_cfg         = vwap_cfg,
    )

    # ── Optimised param set (Optuna 300-trial result, scaled for 1-min bars) ──
    BASE = dict(
        enter_threshold   = scfg.get("enter_long_threshold", 0.50),
        exit_threshold    = scfg.get("exit_threshold", 0.20),
        min_conf          = scfg.get("min_confidence", 0.60),
        stop_pct          = rcfg.get("per_trade_stop_loss_pct", 0.025),
        tp_pct            = rcfg.get("take_profit_pct", 0.04),
        trail_pct         = rcfg.get("trailing_stop_pct", 0.023),
        breakeven_trigger = rcfg.get("breakeven_trigger_pct", 0.016),
        min_hold_bars     = rcfg.get("min_hold_minutes", 70) // BAR_MINUTES,
    )
    param_sets = [
        {**BASE, "label": "Optimised (1-min bars)"},
        {**BASE, "label": "Enter 0.45",    "enter_threshold": 0.45},
        {**BASE, "label": "Enter 0.55",    "enter_threshold": 0.55},
        {**BASE, "label": "Hold 45 min",   "min_hold_bars": 45},
        {**BASE, "label": "Hold 90 min",   "min_hold_bars": 90},
        {**BASE, "label": "Conf 0.65",     "min_conf": 0.65},
    ]

    best_label = ""
    best_wr    = -1.0
    best_exp   = -999.0
    best_stats = None
    best_trades = None
    best_params = None

    print("=" * 65)
    print(f"  {'Label':<22} {'Trades':>7} {'WinRate':>8} {'AvgWin%':>8} {'AvgLoss%':>9} {'Expect%':>8}")
    print("=" * 65)

    for ps in param_sets:
        label = ps.pop("label")
        trades, stats = run_backtest(bars, tech_cfg, **FILTERS, **ps)
        wr  = stats.get("win_rate") or 0.0
        exp = stats.get("expectancy", -999.0)
        n   = stats["total_trades"]
        aw  = stats["avg_win_pct"]
        al  = stats["avg_loss_pct"]
        print(f"  {label:<22} {n:>7} {wr*100:>7.1f}% {aw:>+7.3f}% {al:>+8.3f}% {exp:>+7.3f}%")

        if n >= 5 and (wr > best_wr or (wr == best_wr and exp > best_exp)):
            best_wr = wr
            best_exp = exp
            best_stats = stats
            best_trades = trades
            best_label = label
            best_params = ps

    print("=" * 65)

    if best_stats is None:
        print("\nNo parameter set generated enough trades.")
        return

    print(f"\n{'='*65}")
    print(f"  BEST: {best_label}")
    print(f"{'='*65}")
    _print_stats(best_stats, best_label)
    _print_trade_table(best_trades, max_rows=60)
    _reason_breakdown(best_trades)
    _sector_breakdown(best_trades)

    # ── Analysis notes ────────────────────────────────────────────────────────
    wr = best_stats.get("win_rate") or 0.0
    exp = best_stats.get("expectancy", 0.0)
    print("\n" + "=" * 65)
    print("  STRATEGY ANALYSIS")
    print("=" * 65)
    if wr >= 0.50:
        print(f"  ✓  Win rate {wr*100:.1f}% — strategy is PROFITABLE on this week.")
    elif wr >= 0.35:
        print(f"  ⚠  Win rate {wr*100:.1f}% — marginal; needs larger wins to break even.")
    else:
        print(f"  ✗  Win rate {wr*100:.1f}% — signal quality is low on this market week.")

    if exp > 0:
        print(f"  ✓  Positive expectancy ({exp:+.3f}% per trade) — edge exists.")
    else:
        print(f"  ✗  Negative expectancy ({exp:+.3f}% per trade) — exits too early or entries too weak.")

    # EOD zero-P&L diagnosis
    zero_eod = [t for t in best_trades if t.reason == "eod" and abs(t.pnl_pct) < 0.0001]
    if zero_eod:
        print(f"\n  NOTE: {len(zero_eod)} EOD trades at ~0.00% P&L — positions entered")
        print(f"        in the last 1-2 bars before close (no time to move).")
        print(f"        Adding a 'no-entry in last 30 min' rule would remove these.")

    stop_n = sum(1 for t in best_trades if t.reason == "stop_loss")
    if stop_n > 0:
        avg_stop_loss = np.mean([t.pnl_pct*100 for t in best_trades if t.reason == "stop_loss"])
        print(f"\n  Stop losses: {stop_n} trades, avg loss {avg_stop_loss:.2f}%")
        print(f"  → Stop may be too tight for current volatility.")

    print("\n  Done.\n")

if __name__ == "__main__":
    main()
