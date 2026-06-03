"""
engine_status.py — Live terminal dashboard for the DayTradingBot engine.

Shows engine process, market status, open positions, today's P&L,
recent entries/exits, and a live log tail — refreshed every 5 seconds.

Usage
-----
  python scripts/engine_status.py          # refresh every 5s
  python scripts/engine_status.py --once   # print once and exit
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

ET = ZoneInfo("America/New_York")

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"
BG_RED = "\033[41m"

def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"

WIN  = "WIN "
LOSS = "LOSS"

WIDTH = 72

def hline(char="-", width=WIDTH): return char * width
def box_top():    return "+" + hline("=") + "+"
def box_bot():    return "+" + hline("=") + "+"
def box_sep():    return "+" + hline("-") + "+"
def box_row(text, width=WIDTH):
    # strip ANSI for length calculation
    plain = re.sub(r'\033\[[0-9;]*m', '', text)
    pad = width - len(plain)
    return "| " + text + " " * max(0, pad - 2) + " |"


# ── Data helpers ──────────────────────────────────────────────────────────────

def get_engine_pid() -> tuple[int | None, str]:
    """Return (pid, uptime_str) of running main.py, or (None, '')."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "CommandLine like '%main.py%'",
             "get", "ProcessId,CreationDate", "/format:csv"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[2].strip().isdigit():
                pid = int(parts[2].strip())
                raw_dt = parts[1].strip()  # e.g. 20260603080828.000000+000
                try:
                    dt = datetime.strptime(raw_dt[:14], "%Y%m%d%H%M%S")
                    dt = dt.replace(tzinfo=timezone.utc)
                    elapsed = datetime.now(timezone.utc) - dt
                    h, rem = divmod(int(elapsed.total_seconds()), 3600)
                    m = rem // 60
                    uptime = f"{h}h {m}m" if h else f"{m}m"
                    return pid, uptime
                except Exception:
                    return pid, "?"
    except Exception:
        pass
    return None, ""


def get_market_status() -> tuple[str, str]:
    """Return (status_label, et_time_str)."""
    now_et = datetime.now(ET)
    time_str = now_et.strftime("%I:%M:%S %p ET")
    t = now_et.hour * 60 + now_et.minute
    if now_et.weekday() >= 5:
        return "CLOSED (weekend)", time_str
    if t < 9 * 60 + 30:
        opens_in = (9 * 60 + 30) - t
        return f"PRE-MARKET (opens in {opens_in}m)", time_str
    if t <= 16 * 60:
        return "OPEN", time_str
    return "CLOSED (after hours)", time_str


def get_positions():
    """Return list of (symbol, qty, entry, current, unreal_pnl)."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET"),
            paper=True,
        )
        positions = client.get_all_positions()
        acct = client.get_account()
        result = []
        for p in positions:
            result.append((
                p.symbol,
                float(p.qty),
                float(p.avg_entry_price),
                float(p.current_price),
                float(p.unrealized_pl),
                float(p.unrealized_plpc) * 100,
            ))
        return result, float(acct.equity), float(acct.cash)
    except Exception as e:
        return [], 0.0, 0.0


def get_today_trades():
    """Return today's closed round-trips from Alpaca."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from collections import defaultdict

        client = TradingClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET"),
            paper=True,
        )
        # Start of today ET
        now_et = datetime.now(ET)
        day_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_utc = day_start.astimezone(timezone.utc)

        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=day_start_utc,
            limit=200,
        )
        orders = client.get_orders(req)
        filled = sorted(
            [o for o in orders if o.status.value == "filled"],
            key=lambda o: o.filled_at,
        )

        buy_q = defaultdict(list)
        trades = []
        for o in filled:
            sym   = o.symbol
            price = float(o.filled_avg_price or 0)
            qty   = float(o.filled_qty or 0)
            ts    = o.filled_at
            if o.side.value == "buy":
                buy_q[sym].append([price, qty, ts])
            else:
                remaining = qty
                while remaining > 1e-6 and buy_q[sym]:
                    bp, bq, bt = buy_q[sym][0]
                    matched = min(remaining, bq)
                    pnl = (price - bp) * matched
                    trades.append({
                        "sym": sym, "entry": bp, "exit": price,
                        "qty": matched, "pnl": pnl,
                        "pnl_pct": (price - bp) / bp * 100 if bp else 0,
                        "ts": ts, "won": pnl > 0,
                    })
                    buy_q[sym][0][1] -= matched
                    remaining -= matched
                    if buy_q[sym][0][1] < 1e-6:
                        buy_q[sym].pop(0)
        return trades
    except Exception:
        return []


def get_recent_log_events(n: int = 8) -> list[str]:
    """Return last N important log lines (entries, exits, skips, cycles)."""
    log_path = ROOT / "logs" / "bot.log"
    if not log_path.exists():
        return ["(log not found)"]
    try:
        # Read last 500 lines efficiently
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]
        keywords = ("ENTRY", "EXIT", "SKIP", "Engine cycle", "filled",
                    "take_profit", "stop", "signal_reversed", "eod_flatten",
                    "partial_profit", "Error", "WARNING")
        events = [l for l in lines if any(k in l for k in keywords)]
        return events[-n:]
    except Exception:
        return ["(could not read log)"]


# ── Render ─────────────────────────────────────────────────────────────────────

def render():
    pid, uptime   = get_engine_pid()
    mkt_status, et_time = get_market_status()
    positions, equity, cash = get_positions()
    today_trades  = get_today_trades()
    log_events    = get_recent_log_events(8)

    lines = []
    lines.append(box_top())

    # Header
    is_open = mkt_status == "OPEN"
    mkt_col = green(mkt_status) if is_open else yellow(mkt_status)
    header = f"{bold('DayTradingBot')}  │  {et_time}  │  Market {mkt_col}"
    lines.append(box_row(header))
    lines.append(box_sep())

    # Engine status
    if pid:
        eng = f"{green('● RUNNING')}  PID {pid}  Uptime {uptime}"
    else:
        eng = f"{red('● STOPPED')}  — run: python main.py --broker alpaca"
    lines.append(box_row(bold("ENGINE ") + eng))
    lines.append(box_sep())

    # Account
    pnl_color = green if equity >= 100_000 else red
    acct_line = (f"Equity {bold(pnl_color(f'${equity:,.2f}'))}  "
                 f"Cash ${cash:,.2f}")
    lines.append(box_row(acct_line))
    lines.append(box_sep())

    # Open positions
    lines.append(box_row(bold("OPEN POSITIONS")))
    if positions:
        for sym, qty, entry, cur, upnl, upct in sorted(positions, key=lambda x: -abs(x[4])):
            col  = green if upnl >= 0 else red
            sign = "▲" if upnl >= 0 else "▼"
            row = (f"  {bold(sym):<8} {qty:.0f} sh  "
                   f"entry ${entry:.2f}  now ${cur:.2f}  "
                   f"unrealized {col(f'{sign}${upnl:+.2f} ({upct:+.2f}%)')}")
            lines.append(box_row(row))
    else:
        lines.append(box_row(dim("  (no open positions)")))
    lines.append(box_sep())

    # Today's P&L
    wins   = [t for t in today_trades if t["won"]]
    losses = [t for t in today_trades if not t["won"]]
    day_pnl = sum(t["pnl"] for t in today_trades)
    wr = len(wins) / len(today_trades) * 100 if today_trades else 0
    wr_col = green if wr >= 50 else red
    pnl_col = green if day_pnl >= 0 else red
    lines.append(box_row(bold("TODAY'S CLOSED TRADES")))
    if today_trades:
        stats = (f"  {len(today_trades)} trades  "
                 f"{wr_col(f'{wr:.0f}% WR')}  "
                 f"({green(str(len(wins))+'W')} / {red(str(len(losses))+'L')})  "
                 f"P&L {pnl_col(f'${day_pnl:+.2f}')}")
        lines.append(box_row(stats))
        # Last 4 trades
        for t in sorted(today_trades, key=lambda x: x["ts"], reverse=True)[:4]:
            icon = green("[W]") if t["won"] else red("[L]")
            ts   = t["ts"].astimezone(ET).strftime("%H:%M")
            col  = green if t["won"] else red
            pnl_str = f"{t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)"
            row  = (f"  {icon} {ts}  {bold(t['sym']):<6}  "
                    f"${t['entry']:.2f}→${t['exit']:.2f}  "
                    f"{col(pnl_str)}")
            lines.append(box_row(row))
    else:
        lines.append(box_row(dim("  (no closed trades today)")))
    lines.append(box_sep())

    # Recent log events
    lines.append(box_row(bold("LIVE LOG")))
    for raw in log_events:
        # Trim timestamp prefix and truncate
        parts = raw.split(" ", 4)
        if len(parts) >= 5:
            ts_part  = " ".join(parts[:2])
            src_part = parts[3] if len(parts) > 3 else ""
            msg_part = parts[4] if len(parts) > 4 else ""
            # Colour by content
            if "ENTRY" in msg_part or "filled" in msg_part.lower():
                msg_col = cyan(msg_part[:52])
            elif "EXIT" in msg_part or any(x in msg_part for x in
                                           ("take_profit","stop","signal_rev","eod_flatten","partial")):
                pnl_sign = "+" in msg_part
                msg_col = (green if pnl_sign else red)(msg_part[:52])
            elif "SKIP" in msg_part or "WARNING" in msg_part:
                msg_col = yellow(msg_part[:52])
            elif "Error" in msg_part:
                msg_col = red(msg_part[:52])
            else:
                msg_col = dim(msg_part[:52])
            short_ts  = ts_part[11:19]  # HH:MM:SS
            short_src = src_part.split(".")[-1][:14]
            line = f"  {dim(short_ts)}  {dim(short_src):<14}  {msg_col}"
        else:
            line = f"  {dim(raw[:66])}"
        lines.append(box_row(line))

    lines.append(box_bot())

    # Refresh footer
    now_local = datetime.now().strftime("%H:%M:%S")
    lines.append(dim(f"  Refreshed {now_local}  │  Ctrl+C to exit"))

    return "\n".join(lines)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Print once and exit")
    ap.add_argument("--interval", type=int, default=5, help="Refresh seconds")
    args = ap.parse_args()

    # Enable ANSI + UTF-8 on Windows
    if sys.platform == "win32":
        os.system("color")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        while True:
            output = render()
            if not args.once:
                os.system("cls" if sys.platform == "win32" else "clear")
            print(output)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting status monitor.")


if __name__ == "__main__":
    main()
