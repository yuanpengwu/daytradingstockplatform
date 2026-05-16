"""Entry point for the live trading bot.

Usage:
    python main.py                                # use config.yaml
    python main.py --config my.yaml
    python main.py --broker paper                 # override broker
    python main.py --once                         # run a single cycle (cron mode)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.engine import TradingEngine
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
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--broker", default=None, help="Override broker (paper|alpaca|robinhood)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    args = parser.parse_args()

    load_dotenv()
    log = get_logger("main")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    if args.broker:
        cfg["broker"]["name"] = args.broker

    engine = TradingEngine(cfg)
    if args.once:
        engine.run_once()
    else:
        engine.run_forever()


if __name__ == "__main__":
    main()
