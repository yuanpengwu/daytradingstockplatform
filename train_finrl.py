"""Standalone FinRL PPO training script.

Run this once before starting the live bot to pre-train the PPO agent on
recent historical data.  The live engine also auto-trains on startup, but
running this script separately lets you control the training parameters
and see detailed progress logs.

Usage
─────
    python train_finrl.py                   # default 100k steps, auto GPU
    python train_finrl.py --steps 300000    # more steps → better policy
    python train_finrl.py --device cpu      # force CPU

The trained model is saved to models/finrl_ppo.zip and loaded automatically
by the live engine.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import yaml
from src.data.market_data import MarketData
from src.signals.finrl_signal import FinRLSignal

parser = argparse.ArgumentParser(description="Train FinRL PPO agent")
parser.add_argument("--steps", type=int, default=100_000,
                    help="Total PPO training timesteps (default: 100000)")
parser.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Training device (default: auto)")
parser.add_argument("--lookback", type=int, default=30,
                    help="Days of historical bars to train on (default: 30)")
args = parser.parse_args()

with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

TICKERS = [
    "CRM", "ADBE", "MSFT", "AVGO", "AMD",
    "VMC", "NUE", "AAPL", "NVDA", "TSLA", "META", "GOOGL", "AMZN",
    "SPY", "QQQ",
]

print(f"\n{'='*60}")
print(f"  FinRL PPO Training")
print(f"  Tickers : {len(TICKERS)}")
print(f"  Steps   : {args.steps:,}")
print(f"  Device  : {args.device}")
print(f"  Lookback: {args.lookback} days")
print(f"{'='*60}\n")

# Override config with CLI args
cfg = dict(config.get("signals", {}).get("finrl", {}))
cfg["total_timesteps"] = args.steps
cfg["device"] = args.device

print("Fetching 1-min bars …")
d = config.get("data", {})
md = MarketData(
    provider=d.get("provider", "alpaca"),
    interval="1m",
    lookback_days=args.lookback,
    feed=d.get("feed", "iex"),
)

bars_by_sym = {}
for sym in TICKERS:
    df = md.get_bars(sym)
    if df is not None and not df.empty:
        bars_by_sym[sym] = df
        print(f"  {sym:6s}  {len(df):5d} bars  "
              f"{df.index[0].strftime('%m/%d')} → {df.index[-1].strftime('%m/%d')}")

print(f"\nTraining FinRL PPO on {len(bars_by_sym)} symbols …")
finrl = FinRLSignal(cfg)
ok = finrl.train(bars_by_sym)

if ok:
    print(f"\n✓  Model saved to: {finrl.model_path}")
    print(f"   Start the live bot — FinRL signal is ready.\n")
else:
    print(f"\n✗  Training failed. Check logs above.\n")
    sys.exit(1)
