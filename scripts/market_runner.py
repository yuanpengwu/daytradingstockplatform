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


def _git_pull_latest() -> str:
    """Log the current git commit (do NOT reset — local commits may be ahead of origin).

    We previously did `git reset --hard origin/main` which wiped unpushed
    local commits.  Now we just report the HEAD commit so we can verify
    which code is running without destroying anything.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h %s"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        commit_line = result.stdout.strip()
        log.info("Running commit: %s", commit_line)
        return commit_line
    except Exception as e:
        log.warning("git log failed (%s).", e)
        return "unknown"


def _clear_pycache() -> None:
    """Delete all __pycache__ dirs so Python re-compiles from the pulled .py files."""
    count = 0
    for cache_dir in ROOT.rglob("__pycache__"):
        try:
            import shutil
            shutil.rmtree(cache_dir)
            count += 1
        except Exception:
            pass
    log.info("Cleared %d __pycache__ directories.", count)


def _kill_old_engine() -> None:
    """Kill any stale main.py processes from a previous session."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "CommandLine like '%main.py%'",
             "get", "ProcessId", "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
        killed = 0
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[-1].strip().isdigit():
                pid = int(parts[-1].strip())
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=5)
                killed += 1
        if killed:
            log.info("Killed %d stale engine process(es).", killed)
    except Exception as e:
        log.debug("Kill-old-engine: %s", e)


def main() -> None:
    today_et = et_now().date()
    log.info("market_runner started — checking %s", today_et)

    if not is_trading_day(today_et):
        log.info("Not a trading day (%s) — exiting.", today_et)
        return

    # ── Wait until 9:20 AM ET (engine startup + model training buffer) ────────
    _wait_until(9, 20)

    # ── Step 1: Kill any stale engine from a previous session ─────────────────
    _kill_old_engine()
    time.sleep(2)

    # ── Step 2: Pull latest code + clear stale bytecode ───────────────────────
    commit = _git_pull_latest()
    _clear_pycache()
    log.info("Code ready. Commit: %s", commit)

    # ── Step 3: Start engine (foreground process, output to log files) ────────
    log.info("Starting engine …")
    PYTHON = sys.executable
    engine_proc = subprocess.Popen(
        [PYTHON, str(ROOT / "main.py"), "--broker", "alpaca"],
        cwd=str(ROOT),
        stdout=open(ROOT / "logs" / "bot_stdout.log", "a"),
        stderr=open(ROOT / "logs" / "bot_err.log", "a"),
    )
    log.info("Engine started (PID %d).", engine_proc.pid)

    # Note: the status monitor is opened automatically by main.py on startup.
    # No need to open it here — main.py always does it regardless of launch path.

    # ── Step 5: Wait until 4:05 PM ET then stop ───────────────────────────────
    now = et_now()
    close = now.replace(hour=16, minute=5, second=0, microsecond=0)
    if close <= now:
        log.warning("Already past 4:05 PM ET — stopping engine immediately.")
    else:
        wait_secs = (close - now).total_seconds()
        log.info("Engine will run for %.1f hours (until 4:05 PM ET).", wait_secs / 3600)
        # Poll every minute so we can detect early crash
        while et_now() < close:
            if engine_proc.poll() is not None:
                log.warning("Engine process exited early (code %d)!", engine_proc.returncode)
                break
            time.sleep(60)

    # ── Step 6: Stop engine ────────────────────────────────────────────────────
    log.info("Market closed — stopping engine (PID %d) …", engine_proc.pid)
    engine_proc.terminate()
    try:
        engine_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        engine_proc.kill()
    log.info("Engine stopped.")

    # ── EOD win-rate report → Discord ─────────────────────────────────────────
    log.info("Generating EOD win-rate report …")
    try:
        import subprocess as _sp
        _sp.run(
            [sys.executable, str(ROOT / "scripts" / "win_rate_report.py"), "--discord"],
            cwd=str(ROOT),
            timeout=60,
        )
    except Exception as _e:
        log.warning("Win-rate report failed: %s", _e)

    log.info("market_runner done.")


if __name__ == "__main__":
    main()
