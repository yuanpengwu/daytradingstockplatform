@echo off
REM ============================================================
REM  DayTradingBot launcher — used by Windows Task Scheduler.
REM
REM  Routes through scripts\market_runner.py which enforces:
REM    - Single-instance guard  (one engine at a time via PID lockfile)
REM    - Main-branch check      (aborts if not on 'main')
REM    - NYSE holiday gate      (no-op on weekends / holidays)
REM    - Auto-start at 9:20 ET, auto-stop at 4:05 ET
REM
REM  Do NOT bypass market_runner by calling main.py directly —
REM  that skips all safety checks and can launch duplicate engines.
REM ============================================================
cd /d "%~dp0"

if not exist venv (
    echo [ERROR] venv not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting DayTradingBot via market_runner...
venv\Scripts\python.exe scripts\market_runner.py

echo.
echo market_runner exited.
pause
