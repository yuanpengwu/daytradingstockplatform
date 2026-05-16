"""Alpaca connection diagnostic.

Runs each step of connecting to Alpaca and reports exactly where it fails.

    .\\venv\\Scripts\\python.exe test_alpaca.py
"""
from __future__ import annotations

import os
import sys


def ok(msg):   print(f"[OK]   {msg}")
def fail(msg): print(f"[FAIL] {msg}")
def info(msg): print(f"       {msg}")


def main():
    print("=" * 60)
    print(" Alpaca connection diagnostic")
    print("=" * 60)

    # 1. Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        ok(".env loaded")
    except Exception as e:
        fail(f"could not load .env: {e}")
        return

    # 2. Check credentials are present
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_API_SECRET", "")
    base = os.getenv("ALPACA_BASE_URL", "")
    if not key or not secret:
        fail("ALPACA_API_KEY / ALPACA_API_SECRET missing from .env")
        info("Open .env and make sure both lines are filled in.")
        return
    ok(f"API key present: {key[:6]}...{key[-4:]}  ({len(key)} chars)")
    ok(f"API secret present: {secret[:4]}...  ({len(secret)} chars)")
    if base:
        ok(f"Base URL: {base}")
        if "paper" not in base:
            info("WARNING: base URL is not the paper endpoint — this would be LIVE money.")
    else:
        info("ALPACA_BASE_URL not set — will default to paper endpoint.")

    # 3. Library installed?
    try:
        from alpaca.trading.client import TradingClient
        ok("alpaca-py library is installed")
    except ImportError:
        fail("alpaca-py is not installed")
        info("Run setup.bat, or: .\\venv\\Scripts\\pip install alpaca-py")
        return

    # 4. Create client + fetch account
    try:
        paper = "paper" in (base or "paper")
        client = TradingClient(key, secret, paper=paper)
        acct = client.get_account()
        ok("Connected to Alpaca and fetched account")
        info(f"Account status : {acct.status}")
        info(f"Equity         : ${float(acct.equity):,.2f}")
        info(f"Cash           : ${float(acct.cash):,.2f}")
        info(f"Buying power   : ${float(acct.buying_power):,.2f}")
    except Exception as e:
        fail(f"Could not connect / fetch account: {type(e).__name__}: {e}")
        info("Common causes:")
        info("  - Keys are wrong, or were regenerated on the Alpaca dashboard")
        info("  - Keys are LIVE keys but base URL is paper (or vice versa)")
        info("  - A firewall / VPN / network is blocking api.alpaca.markets")
        return

    # 5. Market clock
    try:
        clock = client.get_clock()
        state = "OPEN" if clock.is_open else "CLOSED"
        ok(f"Market is currently {state}")
        if not clock.is_open:
            info(f"Next open: {clock.next_open}")
    except Exception as e:
        fail(f"Could not fetch market clock: {e}")

    print("=" * 60)
    print(" If every line above says [OK], Alpaca is connected fine.")
    print("=" * 60)


if __name__ == "__main__":
    main()
