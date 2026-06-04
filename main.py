"""Entry point for the live trading bot.

Usage:
    python main.py                                # use config.yaml
    python main.py --config my.yaml
    python main.py --broker paper                 # override broker
    python main.py --once                         # run a single cycle (cron mode)
    python main.py --no-crypto                    # disable crypto engine

Starts two engines in parallel:
  1. TradingEngine  — US stocks, Mon-Fri 9:30 AM – 4:00 PM ET
  2. CryptoEngine   — crypto pairs (BTC/USD etc.), 24 / 7
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.engine import TradingEngine
from src.crypto_engine import CryptoEngine
from src.brokers import get_broker
from src.utils.logger import get_logger


def load_config(path: Path) -> dict:
    if not path.exists():
        example = path.with_suffix(path.suffix + ".example")
        if example.exists():
            print(f"[!] {path} not found. Copy {example.name} -> {path.name} and edit.", file=sys.stderr)
        else:
            print(f"[!] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DayTradingBot")
    parser.add_argument("--config",    default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--broker",    default=None,          help="Override broker name")
    parser.add_argument("--once",      action="store_true",   help="Run one stock cycle then exit")
    parser.add_argument("--no-crypto", action="store_true",   help="Disable crypto engine")
    args = parser.parse_args()

    load_dotenv()
    log = get_logger("main")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    if args.broker:
        cfg["broker"]["name"] = args.broker

    # ── Stock engine (main thread) ────────────────────────────────────────────
    stock_engine = TradingEngine(cfg)

    # ── Crypto engine (background thread, 24/7) ───────────────────────────────
    crypto_cfg = cfg.get("crypto", {})
    crypto_enabled = crypto_cfg.get("enabled", False) and not args.no_crypto

    if crypto_enabled and not args.once:
        broker = stock_engine.broker   # share the same Alpaca account
        crypto_engine = CryptoEngine(broker, cfg)
        crypto_thread = threading.Thread(
            target=crypto_engine.run_forever,
            name="CryptoEngine",
            daemon=True,   # exits automatically when main thread exits
        )
        crypto_thread.start()
        log.info("CryptoEngine started in background thread.")
    else:
        if args.no_crypto:
            log.info("Crypto engine disabled by --no-crypto flag.")
        elif not crypto_enabled:
            log.info("Crypto engine disabled in config (crypto.enabled: false).")

    # ── Run stock engine (blocks until shutdown) ───────────────────────────────
    if args.once:
        stock_engine.run_once()
    else:
        stock_engine.run_forever()


if __name__ == "__main__":
    main()
