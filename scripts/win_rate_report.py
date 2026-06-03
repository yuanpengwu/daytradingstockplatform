"""
win_rate_report.py — Win-rate analytics across 6 time windows.

Pulls all filled orders from Alpaca, matches buy→sell round-trips (FIFO),
then prints and optionally posts a Discord embed with stats for:
  Day / Week / Month / 3 Months / 6 Months / 1 Year

Usage
-----
  python scripts/win_rate_report.py              # print to console only
  python scripts/win_rate_report.py --discord    # also post to Discord

Called automatically by market_runner.py at end of each trading day.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


# ── Data model ────────────────────────────────────────────────────────────────

class ClosedTrade:
    __slots__ = ("symbol", "entry_price", "exit_price", "qty", "pnl", "pnl_pct", "closed_at")

    def __init__(self, symbol, entry_price, exit_price, qty, closed_at):
        self.symbol      = symbol
        self.entry_price = entry_price
        self.exit_price  = exit_price
        self.qty         = qty
        self.pnl         = (exit_price - entry_price) * qty
        self.pnl_pct     = (exit_price - entry_price) / entry_price if entry_price else 0.0
        self.closed_at   = closed_at   # datetime (UTC-aware)

    @property
    def won(self) -> bool:
        return self.pnl > 0


# ── Alpaca fetch ──────────────────────────────────────────────────────────────

def fetch_closed_trades(days: int = 365) -> List[ClosedTrade]:
    """Pull all filled Alpaca orders and match into round-trip trades (FIFO)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    client = TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=True,
    )

    since = datetime.now(timezone.utc) - timedelta(days=days)
    req   = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=since, limit=500)
    orders = client.get_orders(req)

    # Only filled orders, sorted oldest→newest
    filled = sorted(
        [o for o in orders if o.status.value == "filled"],
        key=lambda o: o.filled_at,
    )

    # FIFO matching: queue of (price, qty, timestamp) per symbol
    buy_queues: dict = defaultdict(list)   # sym → [(price, qty, ts), …]
    trades: List[ClosedTrade] = []

    for o in filled:
        sym   = o.symbol
        price = float(o.filled_avg_price or 0)
        qty   = float(o.filled_qty or 0)
        ts    = o.filled_at

        if o.side.value == "buy":
            buy_queues[sym].append([price, qty, ts])   # mutable so we can reduce qty
        else:
            # Match against oldest buys (FIFO)
            remaining = qty
            while remaining > 1e-6 and buy_queues[sym]:
                buy_price, buy_qty, buy_ts = buy_queues[sym][0]
                matched = min(remaining, buy_qty)
                trades.append(ClosedTrade(sym, buy_price, price, matched, ts))
                buy_queues[sym][0][1] -= matched
                remaining -= matched
                if buy_queues[sym][0][1] < 1e-6:
                    buy_queues[sym].pop(0)

    return trades


# ── Stats per window ──────────────────────────────────────────────────────────

_WINDOWS = [
    ("Day",      1),
    ("Week",     7),
    ("Month",    30),
    ("3 Months", 90),
    ("6 Months", 180),
    ("1 Year",   365),
]


def _stats(trades: List[ClosedTrade], days: int) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    window = [t for t in trades if t.closed_at >= cutoff]
    if not window:
        return dict(n=0, wins=0, losses=0, wr=None, pnl=0.0,
                    avg_win=0.0, avg_loss=0.0, pf=None)
    wins   = [t for t in window if t.won]
    losses = [t for t in window if not t.won]
    pf = (
        abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses))
        if losses and wins else None
    )
    return dict(
        n       = len(window),
        wins    = len(wins),
        losses  = len(losses),
        wr      = len(wins) / len(window) * 100,
        pnl     = sum(t.pnl for t in window),
        avg_win = sum(t.pnl_pct for t in wins)   / len(wins)   * 100 if wins   else 0.0,
        avg_loss= sum(t.pnl_pct for t in losses) / len(losses) * 100 if losses else 0.0,
        pf      = pf,
    )


# ── Console output ─────────────────────────────────────────────────────────────

def print_report(trades: List[ClosedTrade]) -> None:
    today = date.today().isoformat()
    print(f"\n{'='*68}")
    print(f"  Win-Rate Report  —  {today}  ({len(trades)} total closed trades)")
    print(f"{'='*68}")
    header = f"  {'Window':<12} {'Trades':>7} {'Wins':>5} {'Losses':>7} {'WR':>7} {'P&L':>10} {'AvgW':>7} {'AvgL':>7} {'PF':>6}"
    print(header)
    print(f"  {'-'*64}")
    for label, days in _WINDOWS:
        s = _stats(trades, days)
        wr_str  = f"{s['wr']:.1f}%"  if s['wr']  is not None else "  n/a"
        pf_str  = f"{s['pf']:.2f}x" if s['pf']  is not None else "   n/a"
        pnl_str = f"${s['pnl']:+.2f}"
        print(
            f"  {label:<12} {s['n']:>7} {s['wins']:>5} {s['losses']:>7} "
            f"{wr_str:>7} {pnl_str:>10} {s['avg_win']:>+6.2f}% {s['avg_loss']:>+6.2f}% {pf_str:>6}"
        )
    print(f"{'='*68}\n")

    # Top / bottom 5 symbols (all-time)
    by_sym: dict = defaultdict(lambda: [0, 0, 0.0])
    for t in trades:
        by_sym[t.symbol][2] += t.pnl
        if t.won: by_sym[t.symbol][0] += 1
        else:     by_sym[t.symbol][1] += 1

    ranked = sorted(by_sym.items(), key=lambda x: x[1][2], reverse=True)
    print("  Top symbols:")
    for sym, (w, l, pnl) in ranked[:5]:
        n  = w + l
        wr = w / n * 100 if n else 0
        print(f"    {sym:<6}  {n:3d} trades  {wr:5.1f}% WR  ${pnl:+.2f}")
    if len(ranked) > 5:
        print("  Bottom symbols:")
        for sym, (w, l, pnl) in ranked[-5:]:
            n  = w + l
            wr = w / n * 100 if n else 0
            print(f"    {sym:<6}  {n:3d} trades  {wr:5.1f}% WR  ${pnl:+.2f}")
    print()


# ── Discord embed ─────────────────────────────────────────────────────────────

def post_discord(trades: List[ClosedTrade]) -> None:
    import requests

    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("DISCORD_WEBHOOK_URL not set — skipping Discord post.")
        return

    today  = date.today().isoformat()
    fields = []

    for label, days in _WINDOWS:
        s = _stats(trades, days)
        if s["n"] == 0:
            value = "`no trades`"
        else:
            wr_icon = "🟢" if (s["wr"] or 0) >= 50 else "🔴"
            pnl_icon = "▲" if s["pnl"] >= 0 else "▼"
            value = (
                f"{wr_icon} **{s['wr']:.1f}%** WR  "
                f"({s['wins']}W / {s['losses']}L)\n"
                f"{pnl_icon} P&L `{s['pnl']:+.2f}`  "
                f"AvgW `{s['avg_win']:+.2f}%`  "
                f"AvgL `{s['avg_loss']:+.2f}%`"
            )
        fields.append({"name": f"📅 {label}", "value": value, "inline": False})

    # Top 3 and bottom 3 symbols
    by_sym: dict = defaultdict(lambda: [0, 0, 0.0])
    for t in trades:
        by_sym[t.symbol][2] += t.pnl
        if t.won: by_sym[t.symbol][0] += 1
        else:     by_sym[t.symbol][1] += 1

    ranked = sorted(by_sym.items(), key=lambda x: x[1][2], reverse=True)
    if ranked:
        top_lines = []
        for sym, (w, l, pnl) in ranked[:3]:
            n  = w + l
            wr = w / n * 100 if n else 0
            top_lines.append(f"🏆 **{sym}** {n}T {wr:.0f}% `{pnl:+.2f}`")
        bot_lines = []
        for sym, (w, l, pnl) in ranked[-3:]:
            n  = w + l
            wr = w / n * 100 if n else 0
            bot_lines.append(f"💀 **{sym}** {n}T {wr:.0f}% `{pnl:+.2f}`")
        fields.append({
            "name": "Best / Worst Symbols (all-time)",
            "value": "\n".join(top_lines + bot_lines),
            "inline": False,
        })

    day_s   = _stats(trades, 1)
    day_pnl = day_s["pnl"]
    color   = 0x2ECC71 if day_pnl >= 0 else 0xE74C3C

    payload = {
        "embeds": [{
            "title":       f"📊 Win-Rate Report  —  {today}",
            "color":       color,
            "description": f"**{len(trades)} total closed trades** (all-time)",
            "fields":      fields,
            "footer":      {"text": "DayTradingBot • daily summary"},
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }]
    }

    r = requests.post(url, json=payload, timeout=10)
    if r.status_code not in (200, 204):
        print(f"Discord post failed: {r.status_code} {r.text[:200]}")
    else:
        print("Win-rate report posted to Discord.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(send_discord: bool = False) -> None:
    print("Fetching trade history from Alpaca (up to 365 days)…")
    trades = fetch_closed_trades(days=365)
    print(f"  {len(trades)} closed round-trips found.")

    print_report(trades)

    if send_discord:
        post_discord(trades)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DayTradingBot win-rate report")
    ap.add_argument("--discord", action="store_true", help="Post report to Discord")
    args = ap.parse_args()
    main(send_discord=args.discord)
