@echo off
REM ============================================================
REM  DayTradingBot launcher - runs the bot against Alpaca paper.
REM  Use this with Windows Task Scheduler, or just double-click it.
REM  The bot self-gates to market hours (9:30-4:00 ET); outside
REM  the session it simply sleeps, so it's safe to leave running.
REM ============================================================
cd /d "%~dp0"

if not exist venv (
    echo [ERROR] venv not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting DayTradingBot on Alpaca paper account...
echo Press Ctrl+C to stop.
venv\Scripts\python.exe main.py --broker alpaca

REM If the bot exits (error or Ctrl+C), keep the window open so you can read why.
echo.
echo Bot stopped.
pause
