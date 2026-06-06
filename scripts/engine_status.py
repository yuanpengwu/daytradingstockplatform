"""
engine_status.py — Live terminal dashboard for the DayTradingBot engine.

Displays in one refreshing screen:
  • Engine PID / uptime / git commit / cycle count / time since last update
  • Market status + current ET time
  • Account equity, cash, day P&L
  • Open positions  (crypto: fractional coins; stocks: shares)
  • Signal scores table — score, confidence, action, top-3 contributors
  • Today's closed trades with entry→exit prices and P&L
  • Live log tail — ENTRY / EXIT / SKIP / WARNING lines

Usage
-----
  python scripts/engine_status.py            # refresh every 5s
  python scripts/engine_status.py --once     # print once and exit
  python scripts/engine_status.py --interval 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

ET = ZoneInfo("America/New_York")

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
MAGENTA= "\033[35m"
WHITE  = "\033[97m"

def green(s):   return f"{GREEN}{s}{RESET}"
def red(s):     return f"{RED}{s}{RESET}"
def yellow(s):  return f"{YELLOW}{s}{RESET}"
def cyan(s):    return f"{CYAN}{s}{RESET}"
def magenta(s): return f"{MAGENTA}{s}{RESET}"
def bold(s):    return f"{BOLD}{s}{RESET}"
def dim(s):     return f"{DIM}{s}{RESET}"

WIDTH = 82

def _plain(s: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', s)

def hline(ch="-"): return ch * WIDTH
def box_top():     return "┌" + hline("─") + "┐"
def box_bot():     return "└" + hline("─") + "┘"
def box_sep():     return "├" + hline("─") + "┤"
def box_row(text: str) -> str:
    pad = WIDTH - len(_plain(text))
    return "│ " + text + " " * max(0, pad - 2) + " │"
def box_title(text: str) -> str:
    plain = _plain(text)
    left  = (WIDTH - len(plain) - 2) // 2
    right = WIDTH - len(plain) - 2 - left
    return "├" + "─" * left + " " + text + " " + "─" * right + "┤"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _wmic_pid() -> tuple[int | None, str]:
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "CommandLine like '%main.py%'",
             "get", "ProcessId,CreationDate", "/format:csv"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[2].strip().isdigit():
                pid = int(parts[2].strip())
                raw = parts[1].strip()
                try:
                    dt = datetime.strptime(raw[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                    secs = int((datetime.now(timezone.utc) - dt).total_seconds())
                    h, r2 = divmod(secs, 3600); m = r2 // 60
                    return pid, (f"{h}h {m}m" if h else f"{m}m")
                except Exception:
                    return pid, "?"
    except Exception:
        pass
    return None, ""


def get_engine_pid():
    pid, up = _wmic_pid()
    return pid, up


def get_git_info() -> tuple[str, str]:
    running = "unknown"
    try:
        log_path = ROOT / "logs" / "bot.log"
        if log_path.exists():
            for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
                if "Engine starting" in line and "commit=" in line:
                    m = re.search(r"commit=([a-f0-9]+)", line)
                    if m:
                        running = m.group(1)
                    break
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=ROOT, capture_output=True, text=True, timeout=5)
        latest = r.stdout.strip() if r.returncode == 0 else "unknown"
        return running, latest
    except Exception:
        return running, "unknown"


def get_market_status() -> tuple[str, str]:
    now = datetime.now(ET)
    t = now.hour * 60 + now.minute
    ts = now.strftime("%I:%M:%S %p ET")
    if now.weekday() >= 5:
        return "CLOSED (weekend)", ts
    if t < 570:   # 9:30
        return f"PRE-MARKET (opens in {570 - t}m)", ts
    if t <= 960:  # 16:00
        return "OPEN", ts
    return "CLOSED (after-hours)", ts


def get_status_json() -> dict:
    p = ROOT / "status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_crypto_status_json() -> dict:
    p = ROOT / "crypto_status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_positions_from_alpaca():
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET"),
            paper="paper" in os.getenv("ALPACA_BASE_URL", "paper"),
        )
        positions = client.get_all_positions()
        acct = client.get_account()
        out = []
        for p in positions:
            sym = p.symbol
            # Normalise crypto: SOLUSD → SOL/USD
            if "/" not in sym and sym.upper().endswith("USD") and len(sym) > 4:
                sym = sym[:-3] + "/USD"
            out.append({
                "symbol": sym,
                "qty":       float(p.qty),
                "entry":     float(p.avg_entry_price),
                "current":   float(p.current_price or 0),
                "upnl":      float(p.unrealized_pl or 0),
                "upnl_pct":  float(p.unrealized_plpc or 0) * 100,
                "mv":        float(p.market_value or 0),
                "is_crypto": "/" in sym,
            })
        return out, float(acct.equity), float(acct.cash)
    except Exception:
        return [], 0.0, 0.0


def get_today_trades():
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = TradingClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET"),
            paper="paper" in os.getenv("ALPACA_BASE_URL", "paper"),
        )
        now_et    = datetime.now(ET)
        day_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        orders    = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, after=day_start, limit=200,
        ))
        filled = sorted([o for o in orders if o.status.value == "filled"],
                        key=lambda o: o.filled_at)
        buy_q  = defaultdict(list)
        trades = []
        for o in filled:
            sym   = o.symbol
            if "/" not in sym and sym.upper().endswith("USD") and len(sym) > 4:
                sym = sym[:-3] + "/USD"
            price = float(o.filled_avg_price or 0)
            qty   = float(o.filled_qty or 0)
            if o.side.value == "buy":
                buy_q[sym].append([price, qty, o.filled_at])
            else:
                rem = qty
                while rem > 1e-9 and buy_q[sym]:
                    bp, bq, bt = buy_q[sym][0]
                    matched = min(rem, bq)
                    pnl = (price - bp) * matched
                    trades.append({"sym": sym, "entry": bp, "exit": price,
                                   "qty": matched, "pnl": pnl,
                                   "pnl_pct": (price - bp) / bp * 100 if bp else 0,
                                   "ts": o.filled_at, "won": pnl > 0,
                                   "is_crypto": "/" in sym})
                    buy_q[sym][0][1] -= matched
                    rem -= matched
                    if buy_q[sym][0][1] < 1e-9:
                        buy_q[sym].pop(0)
        return trades
    except Exception:
        return []


def get_log_events(n: int = 14) -> list[str]:
    p = ROOT / "logs" / "bot.log"
    if not p.exists():
        return ["(log not found)"]
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-800:]
        kw = ("ENTRY", "EXIT", "SKIP", "CRYPTO ENTRY", "CRYPTO EXIT",
              "stop_loss", "take_profit", "trailing_stop", "signal_reversed",
              "eod_flatten", "partial_profit", "reconciled",
              "ERROR", "WARNING", "CryptoTrader", "Alpaca order failed",
              "SELL", "BUY", "filled")
        events = [l for l in lines if any(k in l for k in kw)]
        return events[-n:]
    except Exception:
        return ["(could not read log)"]


def _fmt_qty(qty: float, sym: str) -> str:
    if "/" in sym:                    # crypto — fractional
        base = sym.split("/")[0]
        if qty >= 1:
            return f"{qty:.4f} {base}"
        return f"{qty:.6f} {base}"
    return f"{qty:.0f} sh"


def _fmt_price(price: float, sym: str) -> str:
    if "/" in sym and price < 100:
        return f"${price:.4f}"
    return f"${price:.2f}"


def _age(updated_at_str: str) -> str:
    try:
        dt = datetime.fromisoformat(updated_at_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        return f"{secs // 60}m {secs % 60}s ago"
    except Exception:
        return "?"


# ── Renderer ──────────────────────────────────────────────────────────────────

def render() -> str:
    pid, uptime             = get_engine_pid()
    run_commit, lat_commit  = get_git_info()
    mkt_status, et_time     = get_market_status()
    positions, equity, cash = get_positions_from_alpaca()
    today_trades            = get_today_trades()
    log_events              = get_log_events(14)
    st                      = get_status_json()
    cst                     = get_crypto_status_json()

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(box_top())
    is_open = mkt_status == "OPEN"
    mkt_col = green("OPEN") if is_open else yellow(mkt_status)
    lines.append(box_row(
        f"{bold('DayTradingBot')}  │  {bold(et_time)}  │  Market: {mkt_col}"
        + (f"  │  Regime: {bold(st.get('regime','?').upper())}" if st.get("regime") else "")
    ))

    # ── Engine status ─────────────────────────────────────────────────────────
    lines.append(box_sep())
    stale = run_commit != lat_commit and run_commit not in ("unknown", "")
    if pid:
        ver  = yellow(f"{run_commit} ⚠ STALE (latest={lat_commit})") if stale \
               else green(run_commit)
        age  = _age(st.get("updated_at", ""))
        cyc  = st.get("cycle_count", "?")
        lines.append(box_row(
            f"{green('● RUNNING')}  PID {bold(str(pid))}  up {uptime}  "
            f"commit {ver}  │  cycle #{cyc}  last update {age}"
        ))
    else:
        lines.append(box_row(
            f"{red('● STOPPED')}  {dim('run: python main.py')}"
        ))

    # ── Account ───────────────────────────────────────────────────────────────
    lines.append(box_sep())
    day_pnl  = sum(t["pnl"] for t in today_trades)
    live_pnl = st.get("unrealized_pnl", 0.0)
    eq_col   = green if equity > 0 else red
    dp_col   = green if day_pnl >= 0 else red
    lp_col   = green if live_pnl >= 0 else red
    lines.append(box_row(
        f"Equity {bold(eq_col(f'${equity:,.2f}'))}  "
        f"Cash ${cash:,.2f}  │  "
        f"Day P&L {dp_col(f'${day_pnl:+.2f}')}  "
        f"Unrealized {lp_col(f'${live_pnl:+.2f}')}"
    ))

    # ── Open positions ────────────────────────────────────────────────────────
    lines.append(box_title(bold("OPEN POSITIONS")))
    if positions:
        # Separate crypto vs stocks
        stocks = [p for p in positions if not p["is_crypto"]]
        cryptos = [p for p in positions if p["is_crypto"]]
        for p in sorted(stocks + cryptos, key=lambda x: -abs(x["upnl"])):
            col  = green if p["upnl"] >= 0 else red
            sign = "▲" if p["upnl"] >= 0 else "▼"
            tag  = dim("[crypto]") if p["is_crypto"] else dim("[stock] ")
            qty_str   = _fmt_qty(p["qty"], p["symbol"])
            entry_str = _fmt_price(p["entry"], p["symbol"])
            cur_str   = _fmt_price(p["current"], p["symbol"])
            row = (f"  {tag} {bold(p['symbol']):<10}  {qty_str:<18}  "
                   f"entry {entry_str}  now {cur_str}  "
                   f"{col(f'{sign}${p[chr(117)+chr(112)+chr(110)+chr(108)]:+.2f} ({p[chr(117)+chr(112)+chr(110)+chr(108)+chr(95)+chr(112)+chr(99)+chr(116)]:+.2f}%)')}")
            # Fix: use dict keys directly
            upnl_str = col(f"{sign}${p['upnl']:+.2f} ({p['upnl_pct']:+.2f}%)")
            row = (f"  {tag} {bold(p['symbol']):<10}  {qty_str:<18}  "
                   f"entry {entry_str}  now {cur_str}  {upnl_str}")
            lines.append(box_row(row))
    else:
        lines.append(box_row(dim("  (no open positions)")))

    # ── Signal decisions table ────────────────────────────────────────────────
    decisions  = st.get("decisions", [])
    updated_at = st.get("updated_at", "")
    age_str    = _age(updated_at) if updated_at else "?"

    # Determine staleness label
    is_open    = mkt_status == "OPEN"
    if decisions and not is_open:
        score_title = bold("SIGNAL SCORES") + dim(f"  (last market cycle — {age_str} ago)")
    elif decisions:
        score_title = bold("SIGNAL SCORES") + dim(f"  (live — updated {age_str} ago)")
    else:
        score_title = bold("SIGNAL SCORES")
    lines.append(box_title(score_title))

    if not decisions:
        lines.append(box_row(dim("  Waiting for first trading cycle…  "
                                 "(scores appear at market open)")))
    else:
        # Sort: BUY first, then by abs score descending
        action_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
        sorted_dec = sorted(decisions,
                            key=lambda d: (action_order.get(d.get("action", "HOLD"), 2),
                                           -abs(d.get("score", 0))))
        for d in sorted_dec[:18]:
            sym    = d.get("symbol", "?")
            score  = d.get("score", 0.0)
            conf   = d.get("confidence", 0.0)
            action = d.get("action", "HOLD")

            # Prefer raw_scores for bar (full [-1,+1] range); fall back to components
            display_scores = d.get("raw_scores") or d.get("components", {})
            # Top-3 by absolute raw score
            top3 = sorted(display_scores.items(), key=lambda x: -abs(x[1]))[:3]
            top3_str = "  ".join(f"{k}={v:+.3f}" for k, v in top3)

            # Action colour tag
            if action == "BUY":
                act_col = green(f"[{action:<4}]")
            elif action == "SELL":
                act_col = red(f"[{action:<4}]")
            else:
                act_col = dim(f"[{action:<4}]")

            # Score bar using aggregate score
            bar_len  = min(16, int(abs(score) * 16))
            bar_fill = ("█" * bar_len).ljust(16, "░")
            score_col = green if score >= 0 else red
            score_str = score_col(f"{score:+.3f}")
            sign_ch   = "▲" if score >= 0 else "▼"

            row = (f"  {act_col} {bold(sym):<10}  "
                   f"{sign_ch} {score_str}  {dim(bar_fill)}  "
                   f"conf={conf:.0%}  {dim(top3_str)}")
            lines.append(box_row(row))

    # ── Crypto signal scores ──────────────────────────────────────────────────
    crypto_decisions  = cst.get("crypto_decisions", [])
    crypto_updated_at = cst.get("updated_at", "")
    crypto_age        = _age(crypto_updated_at) if crypto_updated_at else "?"

    if crypto_decisions:
        crypto_title = bold("CRYPTO SIGNALS") + dim(f"  (updated {crypto_age} ago — 24/7)")
    else:
        crypto_title = bold("CRYPTO SIGNALS")
    lines.append(box_title(crypto_title))

    if not crypto_decisions:
        lines.append(box_row(dim("  Waiting for first crypto cycle…")))
    else:
        action_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
        for d in sorted(crypto_decisions,
                        key=lambda d: (action_order.get(d.get("action","HOLD"),2),
                                       -abs(d.get("score",0)))):
            sym    = d.get("symbol", "?")
            score  = d.get("score", 0.0)
            conf   = d.get("confidence", 0.0)
            action = d.get("action", "HOLD")
            display = d.get("raw_scores") or d.get("components", {})
            top3    = sorted(display.items(), key=lambda x: -abs(x[1]))[:2]
            top3_str = "  ".join(f"{k}={v:+.3f}" for k, v in top3)

            if action == "BUY":
                act_col = green(f"[{action:<4}]")
            elif action == "SELL":
                act_col = red(f"[{action:<4}]")
            else:
                act_col = dim(f"[{action:<4}]")

            bar_len   = min(16, int(abs(score) * 16))
            bar_fill  = ("█" * bar_len).ljust(16, "░")
            score_col = green if score >= 0 else red
            sign_ch   = "▲" if score >= 0 else "▼"

            # Show whether this pair is currently held
            held = any(p["symbol"] == sym for p in positions if p.get("is_crypto"))
            tag  = cyan(" [HELD]") if held else ""

            row = (f"  {act_col} {bold(sym):<10}{tag}  "
                   f"{sign_ch} {score_col(f'{score:+.3f}')}  {dim(bar_fill)}  "
                   f"conf={conf:.0%}  {dim(top3_str)}")
            lines.append(box_row(row))

    # ── Today's trades ────────────────────────────────────────────────────────
    lines.append(box_title(bold("TODAY'S CLOSED TRADES")))
    if today_trades:
        wins    = [t for t in today_trades if t["won"]]
        losses  = [t for t in today_trades if not t["won"]]
        day_pnl = sum(t["pnl"] for t in today_trades)
        wr      = len(wins) / len(today_trades) * 100
        wr_col  = green if wr >= 50 else red
        dp_col  = green if day_pnl >= 0 else red
        lines.append(box_row(
            f"  {len(today_trades)} trades  "
            f"{wr_col(f'{wr:.0f}% WR')}  "
            f"({green(f'{len(wins)}W')} / {red(f'{len(losses)}L')})  "
            f"P&L {dp_col(f'${day_pnl:+.2f}')}"
        ))
        for t in sorted(today_trades, key=lambda x: x["ts"], reverse=True)[:6]:
            icon = green("[W]") if t["won"] else red("[L]")
            ts   = t["ts"].astimezone(ET).strftime("%H:%M")
            col  = green if t["won"] else red
            sym  = t["sym"]
            ep   = _fmt_price(t["entry"],  sym)
            xp   = _fmt_price(t["exit"],   sym)
            qty  = _fmt_qty(t["qty"], sym)
            row  = (f"  {icon} {ts}  {bold(sym):<10}  "
                    f"{qty:<16}  {ep}→{xp}  "
                    f"{col(f'${t[chr(112)+chr(110)+chr(108)]:+.2f} ({t[chr(112)+chr(110)+chr(108)+chr(95)+chr(112)+chr(99)+chr(116)]:+.2f}%)')}")
            # Fix: use dict keys directly
            pnl_str = col(f"${t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)")
            row = (f"  {icon} {ts}  {bold(sym):<10}  "
                   f"{qty:<16}  {ep}→{xp}  {pnl_str}")
            lines.append(box_row(row))
    else:
        lines.append(box_row(dim("  (no closed trades today)")))

    # ── Live log ──────────────────────────────────────────────────────────────
    lines.append(box_title(bold("LIVE LOG")))
    # Log format: "2026-06-06 10:35:18 INFO    src.module.name | message text"
    _LOG_RE = re.compile(
        r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\w+\s+([\w.]+)\s*\|\s*(.*)'
    )
    for raw in log_events:
        m = _LOG_RE.match(raw)
        if m:
            ts_str    = m.group(1)
            src_str   = m.group(2)
            msg_str   = m.group(3).strip()
            short_ts  = ts_str[11:19]
            short_src = src_str.split(".")[-1][:16]
            msg_trunc = msg_str[:56]
            if any(x in msg_str for x in ("ENTRY", "CRYPTO ENTRY", "filled")):
                msg_col = cyan(msg_trunc)
            elif any(x in msg_str for x in ("EXIT", "take_profit", "stop_loss",
                                             "trailing_stop", "signal_reversed",
                                             "eod_flatten", "partial_profit")):
                msg_col = (green if "+" in msg_str else red)(msg_trunc)
            elif any(x in msg_str for x in ("SKIP", "WARNING", "reconciled")):
                msg_col = yellow(msg_trunc)
            elif "ERROR" in msg_str or "failed" in msg_str.lower():
                msg_col = red(msg_trunc)
            else:
                msg_col = dim(msg_trunc)
            lines.append(box_row(
                f"  {dim(short_ts)}  {dim(f'{short_src:<16}')}  {msg_col}"
            ))
        else:
            # Fallback: show the last 60 chars if regex didn't match
            lines.append(box_row(f"  {dim(raw.strip()[:60])}"))

    lines.append(box_bot())
    lines.append(dim(f"  Refreshed {datetime.now().strftime('%H:%M:%S')}  │  Ctrl+C to exit"))
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once",     action="store_true", help="Print once and exit")
    ap.add_argument("--interval", type=int, default=5,  help="Refresh interval seconds")
    args = ap.parse_args()

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
