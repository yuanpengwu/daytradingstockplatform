"""Standalone crypto ML training script.

Fetches recent bars for all configured crypto pairs, trains the dedicated
LightGBM model (models/crypto_lgbm.pkl), and prints a summary.

Usage
-----
    python train_crypto_ml.py                   # 10-day lookback (default)
    python train_crypto_ml.py --days 30         # longer history = better model
    python train_crypto_ml.py --interval 1m     # 1-min bars (slower, more precise)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.data.market_data import MarketData
from src.signals.ml_model import MLSignal
from src.utils.logger import get_logger

log = get_logger("train_crypto_ml")

# ── CLI ───────────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="Train the crypto LightGBM model")
_ap.add_argument("--days",     type=int, default=10,  help="Lookback days (default: 10)")
_ap.add_argument("--interval", default="1m",           help="Bar interval (default: 1m)")
_ARGS = _ap.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

ccfg = config.get("crypto", {})
TICKERS = ccfg.get("tickers", ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD"])

# Build crypto ML config (mirrors what CryptoEngine uses)
ml_cfg = dict(ccfg.get("ml", {}))
ml_cfg.setdefault("mode",                       "local")
ml_cfg.setdefault("model_path",                 "models/crypto_lgbm.pkl")
ml_cfg.setdefault("retrain_days",               1)
ml_cfg.setdefault("prediction_horizon_minutes", 30)
ml_cfg.setdefault("label_method",               "fixed")
ml_cfg.setdefault("min_confidence",             float(ccfg.get("min_confidence", 0.45)))
ml_cfg.setdefault("tech_threshold",             0.05)


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  Crypto ML Training")
    print(f"  Pairs    : {', '.join(TICKERS)}")
    print(f"  Lookback : {_ARGS.days} days  |  Interval: {_ARGS.interval}")
    print(f"  Model    : {ml_cfg['model_path']}")
    print(f"{'='*60}\n")

    # ── Fetch bars ────────────────────────────────────────────────────────────
    d = config.get("data", {})
    md = MarketData(
        provider     = d.get("provider", "alpaca"),
        interval     = _ARGS.interval,
        lookback_days= _ARGS.days,
        feed         = d.get("feed", "iex"),
    )

    print("Fetching crypto bars …")
    bars_by_sym = {}
    for sym in TICKERS:
        df = md.get_bars(sym, force_refresh=True)
        if df is not None and not df.empty:
            bars_by_sym[sym] = df
            span = (f"{df.index[0].strftime('%Y-%m-%d')} → "
                    f"{df.index[-1].strftime('%Y-%m-%d')}")
            print(f"  {sym:<10s}  {len(df):6,d} bars  {span}")
        else:
            log.warning("  %s: no data — skipping", sym)

    if not bars_by_sym:
        print("\nNo data fetched. Check ALPACA_API_KEY in .env.")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining crypto LightGBM on {len(bars_by_sym)} pairs …")
    ml = MLSignal(ml_cfg)
    ok = ml.train(bars_by_sym)

    if ok:
        print(f"\n✅  Model saved → {ml_cfg['model_path']}")
        print(f"    Trained at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"    Total bars  : {sum(len(df) for df in bars_by_sym.values()):,}")
    else:
        print("\n❌  Training failed — check logs above for details.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
