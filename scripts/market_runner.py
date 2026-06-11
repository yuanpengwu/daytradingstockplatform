"""
market_runner.py — Auto-start/stop DayTradingBot on NYSE trading days.

Designed to be triggered every weekday by Windows Task Scheduler or a
Claude Code scheduled task.  It handles:

  - NYSE holiday detection  (US public holidays + exchange closures)
  - DST-aware US/Eastern time  (via zoneinfo — stdlib since Python 3.9)
  - Single-instance lock      (prevents duplicate engine launches)
  - Branch verification       (aborts unless working tree is on 'main')
  - Engine start at 9:20 AM ET
  - Engine stop  at 4:05 PM ET
  - Logging to logs/market_runner.log

Usage
-----
  python scripts/market_runner.py
  (or via run_bot.bat which calls this script through the venv)

Schedule
--------
  Trigger Mon–Fri at 20:30 local (Beijing UTC+8) = 8:30 AM EDT.
  The script waits internally if it's started too early.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT   = Path(__file__).resolve().parent.parent
LOGS   = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

# Single-instance lockfile — stores the PID of the running market_runner so
# any second invocation can detect and exit rather than spawn a second engine.
LOCK_FILE = ROOT / ".market_runner.lock"

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


# ── Single-instance lock ──────────────────────────────────────────────────────

def _acquire_lock() -> bool:
    """Acquire the single-instance lock.

    Returns True  if this process successfully took the lock.
    Returns False if another market_runner is still alive (duplicate detected).
    If the lock file exists but the recorded PID is dead, the stale file is
    silently overwritten (handles crash-without-cleanup and machine restarts).
    """
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            # On Windows, tasklist returns the process entry if it is alive.
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            if f'"{pid}"' in result.stdout:
                return False   # process is alive → another instance is running
            log.debug("Stale lock (PID %d is dead) — overwriting.", pid)
        except Exception as exc:
            log.debug("Could not inspect lock file (%s) — overwriting.", exc)

    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    """Remove the lockfile.  Safe to call even if the file is absent."""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception as exc:
        log.debug("Could not remove lock file: %s", exc)


# ── Branch verification ───────────────────────────────────────────────────────

def _verify_main_branch() -> bool:
    """Return True only when the working tree is on the 'main' branch.

    Running on a feature branch is almost always a mistake — it means
    development code hits the live account.  market_runner does NOT auto-
    checkout main to avoid silently discarding in-progress work; the fix is
    manual: `git checkout main` then restart the bot.
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        branch = result.stdout.strip()
        if branch == "main":
            log.info("Branch check passed: on 'main'.")
            return True
        log.error(
            "ABORT: working tree is on branch %r, not 'main'. "
            "Run `git checkout main` and restart market_runner.",
            branch,
        )
        return False
    except Exception as exc:
        log.warning("Branch check failed (%s) — proceeding without verification.", exc)
        return True   # don't abort if git itself is broken; just warn


# ── Git status report ─────────────────────────────────────────────────────────

def _log_git_status() -> str:
    """Fetch remote refs and report current commit vs origin/main.

    Does NOT reset or pull — local uncommitted changes and ahead-of-origin
    commits are preserved.  A warning is logged if origin/main has commits
    that aren't yet merged locally.
    """
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", "main"],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h %s"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        behind_out = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        behind_n = int(behind_out) if behind_out.isdigit() else 0
        if behind_n:
            log.warning(
                "Running commit: %s  (%d commit(s) behind origin/main — "
                "consider `git pull` before tomorrow's session)",
                commit, behind_n,
            )
        else:
            log.info("Running commit: %s  (up to date with origin/main)", commit)
        return commit
    except Exception as exc:
        log.warning("git status check failed (%s) — continuing.", exc)
        return "unknown"


# ── Utilities ─────────────────────────────────────────────────────────────────

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


def _clear_pycache() -> None:
    """Delete all __pycache__ dirs so Python re-compiles from the current .py files."""
    import shutil
    count = 0
    for cache_dir in ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            count += 1
        except Exception:
            pass
    log.info("Cleared %d __pycache__ directories.", count)


def _kill_old_engine() -> None:
    """Kill any stale main.py processes left over from a previous session."""
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
    except Exception as exc:
        log.debug("Kill-old-engine: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    today_et = et_now().date()
    log.info("market_runner started — checking %s", today_et)

    if not is_trading_day(today_et):
        log.info("Not a trading day (%s) — exiting.", today_et)
        return

    # ── Single-instance guard ─────────────────────────────────────────────────
    # If the Task Scheduler fires twice (DST edge-case, wake-from-sleep, double
    # task registration), the second invocation detects the live PID and exits
    # before spawning a second engine.
    if not _acquire_lock():
        log.warning(
            "market_runner is already running (PID recorded in %s). "
            "This duplicate instance is exiting to prevent a second engine launch.",
            LOCK_FILE.name,
        )
        return

    try:
        # ── Wait until 9:20 AM ET (model warm-up buffer before 9:30 open) ─────
        _wait_until(9, 20)

        # ── Branch guard ──────────────────────────────────────────────────────
        if not _verify_main_branch():
            return   # error already logged; lock released by finally

        # ── Step 1: Kill any stale engine from a previous session ─────────────
        _kill_old_engine()
        time.sleep(2)

        # ── Step 2: Report git status + clear stale bytecode ──────────────────
        commit = _log_git_status()
        _clear_pycache()
        log.info("Code ready. Commit: %s", commit)

        # ── Step 3: Start engine ──────────────────────────────────────────────
        log.info("Starting engine …")
        PYTHON = sys.executable
        engine_proc = subprocess.Popen(
            [PYTHON, str(ROOT / "main.py"), "--broker", "alpaca"],
            cwd=str(ROOT),
            stdout=open(ROOT / "logs" / "bot_stdout.log", "a"),
            stderr=open(ROOT / "logs" / "bot_err.log", "a"),
        )
        log.info("Engine started (PID %d).", engine_proc.pid)

        # ── Step 4: Wait until 4:05 PM ET then stop ───────────────────────────
        now   = et_now()
        close = now.replace(hour=16, minute=5, second=0, microsecond=0)
        if close <= now:
            log.warning("Already past 4:05 PM ET — stopping engine immediately.")
        else:
            wait_secs = (close - now).total_seconds()
            log.info("Engine will run for %.1f hours (until 4:05 PM ET).", wait_secs / 3600)
            while et_now() < close:
                if engine_proc.poll() is not None:
                    log.warning("Engine process exited early (code %d)!", engine_proc.returncode)
                    break
                time.sleep(60)

        # ── Step 5: Stop engine ───────────────────────────────────────────────
        log.info("Market closed — stopping engine (PID %d) …", engine_proc.pid)
        engine_proc.terminate()
        try:
            engine_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            engine_proc.kill()
        log.info("Engine stopped.")

        # ── Step 6: EOD win-rate report → Discord ─────────────────────────────
        log.info("Generating EOD win-rate report …")
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "win_rate_report.py"), "--discord"],
                cwd=str(ROOT),
                timeout=60,
            )
        except Exception as exc:
            log.warning("Win-rate report failed: %s", exc)

    finally:
        _release_lock()

    log.info("market_runner done.")


if __name__ == "__main__":
    main()
