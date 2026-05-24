#!/usr/bin/env python3
"""
run_optimize.py  — Bayesian hyperparameter search to maximise backtest profit.

Uses Optuna (TPE sampler) to intelligently search the parameter space,
evaluating each trial against the full 21-day dataset.

Objective: total_return = expectancy_per_trade × num_trades
  — this rewards both higher win-rate/R:R AND more trade opportunities.
  — a floor of MIN_TRADES is enforced to avoid over-fitted low-sample configs.

Usage:
    python run_optimize.py [--trials N] [--days D] [--apply]

    --trials N   number of Optuna trials (default 300)
    --days   D   trading days of data to fetch (default 21 ≈ 3 weeks)
    --apply      write best parameters to config.yaml after optimising
"""
from __future__ import annotations
import argparse, os, sys, warnings
from datetime import time as dtime
from pathlib import Path

warnings.filterwarnings("ignore")          # suppress optuna/numpy noise

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
import optuna
import yaml
optuna.logging.set_verbosity(optuna.logging.WARNING)

from backtest import fetch_bars, run_backtest

# ── Constants ─────────────────────────────────────────────────────────────────
BAR_MINUTES     = 1       # 1-minute bars (finest resolution on Alpaca free tier)
MIN_TRADES      = 20      # discard configs with fewer trades (higher floor for 1-min)
WARMUP_TRIALS   = 30      # random exploration before TPE kicks in
DEFAULT_TRIALS  = 400     # more trials — larger search space for 1-min params
DEFAULT_DAYS    = 21

# ── Parameter space (scaled for 1-minute bars) ────────────────────────────────
# Key differences vs 5-min space:
#   • min_hold_bars: 15–120 bars = 15–120 min  (was 3–15 bars = 15–75 min on 5m)
#   • atr_stop_mult: wider range — 1-min ATR is ~5× smaller, needs bigger multiplier
#   • atr_tp_mult:   wider range for same reason
# Each entry: (name, type, low, high, step)
PARAM_SPACE = [
    # Entry / exit signal thresholds
    ("enter_threshold",   "float", 0.35,  0.65,  0.05),
    ("exit_threshold",    "float", 0.10,  0.40,  0.05),
    ("min_conf",          "float", 0.40,  0.75,  0.05),

    # Stop / take-profit (% of price)
    ("stop_pct",          "float", 0.005, 0.040, 0.005),
    ("tp_pct",            "float", 0.020, 0.120, 0.010),

    # ATR stop multipliers — wider range for 1-min bars
    ("atr_stop_mult",     "float", 2.0,   15.0,  0.5),
    ("atr_tp_mult",       "float", 3.0,   20.0,  0.5),

    # Trailing stop / breakeven
    ("trail_pct",         "float", 0.005, 0.040, 0.005),
    ("breakeven_trigger", "float", 0.002, 0.025, 0.002),

    # Hold / timing — in 1-min bars: 15 bars = 15 min, 120 bars = 2 hr
    ("min_hold_bars",     "int",   15,    120,   5),

    # Portfolio limits
    ("max_concurrent",    "int",   3,     12,    1),
    ("max_per_sector",    "int",   1,     4,     1),

    # Relative strength filter
    ("rs_min",            "float", -0.002, 0.015, 0.001),
]


def _suggest(trial: optuna.Trial, name: str, ptype: str,
             low, high, step) -> float | int:
    if ptype == "float":
        return trial.suggest_float(name, low, high, step=step)
    elif ptype == "int":
        return trial.suggest_int(name, low, high, step=int(step))
    raise ValueError(f"Unknown param type: {ptype}")


def objective(trial: optuna.Trial, bars: dict, orb_cfg: dict,
              vwap_cfg: dict, tech_cfg: dict) -> float:
    params = {name: _suggest(trial, name, ptype, low, high, step)
              for name, ptype, low, high, step in PARAM_SPACE}

    # Fixed filters that match live config (scaled for 1-min bars)
    filters = dict(
        use_dead_zone     = True,
        dead_zone_start   = dtime(12,  0),
        dead_zone_end     = dtime(13, 30),
        pre_close_cutoff  = dtime(15, 20),
        use_rs_filter     = True,
        use_sector_limit  = True,
        use_regime        = True,
        use_atr_stops     = True,
        open_buffer_bars  = 30,   # 30 × 1min = 30 min open buffer
        orb_cfg           = orb_cfg,
        vwap_cfg          = vwap_cfg,
        use_prev_close_filter = False,
        use_next_day_cooloff  = True,
    )

    try:
        trades, stats = run_backtest(bars, tech_cfg, **filters, **params)
    except Exception:
        return float("-inf")

    n   = stats["total_trades"]
    wr  = stats.get("win_rate") or 0.0
    exp = stats.get("expectancy", -999.0)   # % per trade

    if n < MIN_TRADES:
        # Heavy penalty — return negative proportional to how few trades
        return -10.0 * (MIN_TRADES - n)

    # Objective: total expected return over all trades
    total_return = exp * n

    # Bonus for high WR (tiebreak and overfit-safety)
    bonus = 2.0 * max(0.0, wr - 0.50)

    return total_return + bonus


def _build_tech_cfg() -> dict:
    """Load technical config from config.yaml."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("signals", {}).get("technical", {})


def _load_orb_vwap() -> tuple[dict, dict]:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scfg = cfg.get("signals", {})
    return scfg.get("orb", {}), scfg.get("vwap_bounce", {})


def _apply_to_config(best_params: dict) -> None:
    """Write best params back into config.yaml."""
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Map param names → yaml paths
    scfg = cfg.setdefault("signals", {})
    rcfg = cfg.setdefault("risk", {})

    scfg["enter_long_threshold"]  = best_params["enter_threshold"]
    scfg["exit_threshold"]        = best_params["exit_threshold"]
    scfg["min_confidence"]        = best_params["min_conf"]
    scfg["relative_strength_min"] = best_params["rs_min"]

    rcfg["per_trade_stop_loss_pct"]   = best_params["stop_pct"]
    rcfg["take_profit_pct"]           = best_params["tp_pct"]
    rcfg["atr_stop_multiplier"]       = best_params["atr_stop_mult"]
    rcfg["atr_tp_multiplier"]         = best_params["atr_tp_mult"]
    rcfg["trailing_stop_pct"]         = best_params["trail_pct"]
    rcfg["breakeven_trigger_pct"]     = best_params["breakeven_trigger"]
    rcfg["min_hold_minutes"]          = best_params["min_hold_bars"] * BAR_MINUTES
    rcfg["max_concurrent_positions"]  = best_params["max_concurrent"]
    rcfg["max_positions_per_sector"]  = best_params["max_per_sector"]

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n  config.yaml updated with best parameters.")


def main():
    parser = argparse.ArgumentParser(description="Optimise DayTradingBot parameters via Bayesian search.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    parser.add_argument("--days",   type=int, default=DEFAULT_DAYS,   help="Trading days of history")
    parser.add_argument("--apply",  action="store_true",               help="Apply best params to config.yaml")
    args = parser.parse_args()

    # ── Universe (same as live bot) ───────────────────────────────────────────
    tickers = [
        "QCOM","AMD","INTC","CRM","TXN",
        "MS","GS","BAC","WFC","C",
        "AAPL","MSFT","NVDA","TSLA","META","GOOGL","AMZN",
        "SPY","QQQ",
    ]

    print(f"Fetching {args.days}-day 1-min Alpaca bars ...")
    bars = fetch_bars(tickers, days=args.days, bar_minutes=1)
    if not bars:
        sys.exit("No data returned.")

    all_dates = sorted({str(d.date()) for df in bars.values() for d in df.index})
    total_bars = sum(len(v) for v in bars.values())
    print(f"Data: {len(bars)} symbols | {len(all_dates)} days | {total_bars:,} bars")
    print(f"\nRunning {args.trials} Optuna trials (TPE sampler) ...")
    print(f"Objective: expectancy% × num_trades  (min {MIN_TRADES} trades required)\n")

    orb_cfg, vwap_cfg = _load_orb_vwap()
    tech_cfg = _build_tech_cfg()

    # ── Optuna study ──────────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=WARMUP_TRIALS,
        seed=42,
        multivariate=True,       # model parameter correlations
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    best_so_far = float("-inf")
    checkpoint_interval = max(10, args.trials // 20)   # print every ~5%

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        nonlocal best_so_far
        if trial.value is not None and trial.value > best_so_far:
            best_so_far = trial.value
            v = trial.value
            p = trial.params
            print(
                f"  Trial {trial.number:>4}  obj={v:+.3f}"
                f"  enter={p['enter_threshold']:.2f}"
                f"  conf={p['min_conf']:.2f}"
                f"  stop={p['stop_pct']*100:.1f}%"
                f"  tp={p['tp_pct']*100:.1f}%"
                f"  atr×{p['atr_stop_mult']:.1f}"
                f"  hold={p['min_hold_bars']}b"
                f"  conc={p['max_concurrent']}"
            )
        elif trial.number % checkpoint_interval == 0 and trial.number > 0:
            print(f"  ... trial {trial.number}/{args.trials}  best so far: {best_so_far:+.3f}")

    study.optimize(
        lambda trial: objective(trial, bars, orb_cfg, vwap_cfg, tech_cfg),
        n_trials=args.trials,
        callbacks=[_callback],
        show_progress_bar=False,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*65}")
    print(f"  OPTIMISATION COMPLETE  ({args.trials} trials)")
    print(f"{'='*65}")
    print(f"  Best objective  : {best.value:+.4f}  (expectancy% × trades)")
    print()

    # Re-run best params to get full stats + trade table
    best_params = best.params
    filters = dict(
        use_dead_zone     = True,
        dead_zone_start   = dtime(12,  0),
        dead_zone_end     = dtime(13, 30),
        pre_close_cutoff  = dtime(15, 20),
        use_rs_filter     = True,
        use_sector_limit  = True,
        use_regime        = True,
        use_atr_stops     = True,
        open_buffer_bars  = 30,   # 30 × 1min = 30 min
        orb_cfg           = orb_cfg,
        vwap_cfg          = vwap_cfg,
        use_prev_close_filter = False,
        use_next_day_cooloff  = True,
    )
    trades, stats = run_backtest(bars, tech_cfg, **filters, **best_params)

    n   = stats["total_trades"]
    wr  = stats.get("win_rate") or 0.0
    exp = stats.get("expectancy", 0.0)
    aw  = stats["avg_win_pct"]
    al  = stats["avg_loss_pct"]

    print(f"  Trades    : {n}")
    print(f"  Win rate  : {wr*100:.1f}%")
    print(f"  Avg win   : {aw:+.3f}%")
    print(f"  Avg loss  : {al:+.3f}%")
    print(f"  Expectancy: {exp:+.4f}% per trade")
    print(f"  Total P&L : {exp*n:+.2f}% (sum of expectancies)")
    print()
    print("  Best parameters:")
    print(f"    enter_threshold   = {best_params['enter_threshold']:.2f}")
    print(f"    exit_threshold    = {best_params['exit_threshold']:.2f}")
    print(f"    min_conf          = {best_params['min_conf']:.2f}")
    print(f"    stop_pct          = {best_params['stop_pct']*100:.2f}%")
    print(f"    tp_pct            = {best_params['tp_pct']*100:.2f}%")
    print(f"    atr_stop_mult     = {best_params['atr_stop_mult']:.1f}×")
    print(f"    atr_tp_mult       = {best_params['atr_tp_mult']:.1f}×")
    print(f"    trail_pct         = {best_params['trail_pct']*100:.2f}%")
    print(f"    breakeven_trigger = {best_params['breakeven_trigger']*100:.2f}%")
    print(f"    min_hold_bars     = {best_params['min_hold_bars']} ({best_params['min_hold_bars']*BAR_MINUTES} min)")
    print(f"    max_concurrent    = {best_params['max_concurrent']}")
    print(f"    max_per_sector    = {best_params['max_per_sector']}")
    print(f"    rs_min            = {best_params['rs_min']:.4f}")

    # Exit reason breakdown
    from collections import defaultdict
    reason_stats: dict = defaultdict(lambda: {"n": 0, "wins": 0})
    for t in trades:
        reason_stats[t.reason]["n"]    += 1
        reason_stats[t.reason]["wins"] += int(t.won)
    print()
    print("  Exit breakdown:")
    for reason, s in sorted(reason_stats.items(), key=lambda x: -x[1]["n"]):
        wr_r = s["wins"] / s["n"] if s["n"] else 0.0
        print(f"    {reason:<20} {s['n']:>4} trades  {wr_r*100:>5.1f}% WR")

    # ── Parameter importance ─────────────────────────────────────────────────
    try:
        importance = optuna.importance.get_param_importances(study)
        print()
        print("  Parameter importance (top 5):")
        for i, (k, v) in enumerate(importance.items()):
            if i >= 5:
                break
            print(f"    {k:<24} {v*100:.1f}%")
    except Exception:
        pass

    print(f"\n{'='*65}")

    if args.apply:
        _apply_to_config(best_params)
        print("  Run 'python main.py' to start the live bot with new settings.")
    else:
        print("  Re-run with --apply to write these parameters to config.yaml.")

    print()


if __name__ == "__main__":
    main()
