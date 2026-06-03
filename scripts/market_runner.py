"""
market_runner.py — Auto-start/stop DayTradingBot on NYSE trading days.

Designed to be triggered every weekday by Windows Task Scheduler or a
Claude Code scheduled task.  It handles:

  - NYSE holiday detection  (US public holidays + exchange closures)
  - DST-aware US/Eastern time  (via zoneinfo — stdlib since Python 3.9)
  - Engine start via scripts/start_bot.ps1
  - Engine stop  via scripts/stop_bot.ps1 at 4:05 PM ET
  - Logging to logs/market_runner.log

Usage
-----
  python scripts/market_runner.py

Schedule
--------
  Trigger Mon–Fri at 20:30 local (Beijing UTC+8) = 8:30 AM EDT.
  The script waits internally if it's started too early.
"""
from __future__ import annotations

import subprocess
import sys
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT   = Path(__file__).resolve().parent.parent
LOGS   = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "market_runner.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("market_runner")

ET = ZoneInfo("America/New_York")   # handles EDT/EST automatically


# ── NYSE holiday list (US federal + exchange-specific) ───────────────────────
# Add years as needed.  Source: NYSE holiday calendar.
_NYSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025,  1,  1),   # New Year's Day
    date(2025,  1, 20),   # MLK Day
    date(2025,  2, 17),   # Presidents' Day
    date(2025,  4, 18),   # Good Friday
    date(2025,  5, 26),   # Memorial Day
    date(2025,  6, 19),   # Juneteenth
    date(2025,  7,  4),   # Independence Day
    date(2025,  9,  1),   # Labor Day
    date(2025, 11, 27),   # Thanksgiving
    date(2025, 12, 25),   # Christmas
    # 2026
    date(2026,  1,  1),   # New Year's Day
    date(2026,  1, 19),   # MLK Day
    date(2026,  2, 16),   # Presidents' Day
    date(2026,  4,  3),   # Good Friday
    date(2026,  5, 25),   # Memorial Day
    date(2026,  6, 19),   # Juneteenth
    date(2026,  7,  3),   # Independence Day (observed)
    date(2026,  9,  7),   # Labor Day
    date(2026, 11, 26),   # Thanksgiving
    date(2026, 12, 25),   # Christmas
}


def et_now() -> datetime:
    return datetime.now(ET)


def is_trading_day(today: date | None = None) -> bool:
    """Return True if NYSE is open on *today* (ET date)."""
    d = today or et_now().date()
    if d.weekday() >= 5:       # Saturday=5, Sunday=6
        return False
    if d in _NYSE_HOLIDAYS:
        return False
    return True


def _wait_until(target_hour: int, target_min: int) -> None:
    """Block until hh:mm ET, polling every 30 s."""
    while True:
        now = et_now()
        target = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        log.info("Market opens at %02d:%02d ET — waiting %.0f min …",
                 target_hour, target_min, remaining / 60)
        time.sleep(min(remaining, 30))


def _run_ps(script_name: str) -> None:
    """Run a PowerShell script from the scripts/ directory."""
    ps_path = ROOT / "scripts" / script_name
    result = subprocess.run(
        ["powershell", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(ps_path)],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        log.info("PS %s stdout: %s", script_name, result.stdout.strip())
    if result.stderr.strip():
        log.warning("PS %s stderr: %s", script_name, result.stderr.strip())


def main() -> None:
    today_et = et_now().date()
    log.info("market_runner started — checking %s", today_et)

    if not is_trading_day(today_et):
        log.info("Not a trading day (%s) — exiting.", today_et)
        return

    # ── Wait until 9:25 AM ET (5 min before open, engine startup buffer) ────
    _wait_until(9, 25)

    # ── Start engine ─────────────────────────────────────────────────────────
    log.info("Starting engine via start_bot.ps1 …")
    _run_ps("start_bot.ps1")
    log.info("Engine start command issued.")

    # ── Wait until 4:05 PM ET (5 min after close) ───────────────────────────
    now = et_now()
    close = now.replace(hour=16, minute=5, second=0, microsecond=0)
    if close <= now:
        log.warning("Already past 4:05 PM ET — stopping engine immediately.")
    else:
        wait_secs = (close - now).total_seconds()
        log.info("Engine will run for %.1f hours (until 4:05 PM ET).", wait_secs / 3600)
        time.sleep(wait_secs)

    # ── Stop engine ───────────────────────────────────────────────────────────
    log.info("Market closed — stopping engine via stop_bot.ps1 …")
    _run_ps("stop_bot.ps1")
    log.info("Engine stopped. market_runner done.")


if __name__ == "__main__":
    main()
