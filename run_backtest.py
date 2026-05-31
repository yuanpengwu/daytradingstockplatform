"""1-week backtest runner — walk-forward split to avoid ML look-ahead bias."""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.backtest.backtester import Backtester, BacktestResult
from src.data.market_data import MarketData
from src.signals.ml_model import MLSignal
from src.signals.finrl_signal import FinRLSignal
import numpy as np
from collections import Counter

with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

TICKERS = [
    "CRM","ADBE","MSFT","AVGO","AMD",
    "VMC","NUE","AAPL","NVDA","TSLA","META","GOOGL","AMZN",
    "SPY","QQQ",
]
STARTING_CASH = float(config["broker"].get("starting_cash", 10_000))
SLIPPAGE_BPS  = float(config["broker"].get("slippage_bps", 5))

LOOKBACK_DAYS = 40   # ~22 trading days ≈ 1 calendar month

print(f"\n{'='*60}")
print(f"  DayTradingBot 1-month backtest  ({LOOKBACK_DAYS} calendar days)")
print(f"  Cash: ${STARTING_CASH:,.0f}   Slippage: {SLIPPAGE_BPS} bps")
print(f"{'='*60}\n")

print(f"Fetching 1-min bars ({LOOKBACK_DAYS} calendar days) ...")
md = MarketData(
    provider=config["data"].get("provider","alpaca"),
    interval="1m",
    lookback_days=LOOKBACK_DAYS,
    feed=config["data"].get("feed","iex"),
)

bars_by_sym = {}
for sym in TICKERS:
    df = md.get_bars(sym)
    if df is not None and not df.empty:
        bars_by_sym[sym] = df
        print(f"  {sym:6s}  {len(df):5d} bars  "
              f"{df.index[0].strftime('%m/%d')} to {df.index[-1].strftime('%m/%d')}")

print(f"\nWalk-forward split: train on first 50%, test on last 50%")
train_bars, test_bars = {}, {}
for sym, df in bars_by_sym.items():
    n = len(df); s = n // 2
    if s >= 100: train_bars[sym] = df.iloc[:s]
    if n-s >= 50: test_bars[sym]  = df.iloc[s:]

print(f"Training local ML model (LightGBM) on {len(train_bars)} symbols ...")
ml = MLSignal(config["signals"]["ml"])
ml.train(train_bars)

# FinRL: use reduced timesteps for backtest speed (~30s CPU / ~10s GPU)
print(f"Training FinRL PPO agent on {len(train_bars)} symbols ...")
finrl_cfg = dict(config["signals"].get("finrl", {}))
finrl_cfg["total_timesteps"] = 100_000  # more steps for larger 1-month dataset
finrl = FinRLSignal(finrl_cfg)
finrl.train(train_bars)

print(f"Running backtest on {len(test_bars)} symbols ...\n")
bt = Backtester(config=config, starting_cash=STARTING_CASH,
                slippage_bps=SLIPPAGE_BPS, interval="1m")
bt.ml    = ml
bt.finrl = finrl
result = bt.run(test_bars)

wins   = [t for t in result.trades if t.pnl > 0]
losses = [t for t in result.trades if t.pnl <= 0]
pf     = (abs(sum(t.pnl for t in wins)) /
          abs(sum(t.pnl for t in losses))) if losses and any(t.pnl<0 for t in losses) else 999
avg_w  = np.mean([t.pnl_pct*100 for t in wins])   if wins   else 0
avg_l  = np.mean([t.pnl_pct*100 for t in losses]) if losses else 0

print(f"{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Starting cash : ${result.starting_cash:>10,.2f}")
print(f"  Ending cash   : ${result.ending_cash:>10,.2f}")
print(f"  Net P&L       : ${result.ending_cash-result.starting_cash:>+10,.2f}  ({result.total_return*100:+.2f}%)")
print(f"{'─'*60}")
print(f"  Total trades  : {result.num_trades}")
print(f"  Win rate      : {len(wins)}/{result.num_trades}  ({result.win_rate*100:.1f}%)")
print(f"  Avg win       : {avg_w:+.2f}%")
print(f"  Avg loss      : {avg_l:+.2f}%")
print(f"  Profit factor : {pf:.2f}x")
print(f"  Sharpe ratio  : {result.sharpe:.2f}")
print(f"{'─'*60}")
print(f"  Exit reasons  :")
for r, n in Counter(t.reason for t in result.trades).most_common():
    pct = n/result.num_trades*100 if result.num_trades else 0
    print(f"    {r:<22s} {n:3d} ({pct:.0f}%)")
print(f"{'─'*60}")
print(f"  Per-symbol:")
sym_data = {}
for t in result.trades:
    sym_data.setdefault(t.symbol,[]).append(t.pnl)
for sym, pnls in sorted(sym_data.items(), key=lambda x:-sum(x[1])):
    w = sum(1 for p in pnls if p>0)
    print(f"    {sym:<6s} {len(pnls):2d} trades  {w}/{len(pnls)} wins  ${sum(pnls):+.2f}")
print(f"{'='*60}\n")
