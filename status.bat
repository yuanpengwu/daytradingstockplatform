@echo off
REM ============================================================
REM  DayTradingBot live status dashboard
REM  Double-click this file to open the dashboard in this window.
REM  Refreshes every 5 seconds. Press Ctrl+C to exit.
REM ============================================================
cd /d "%~dp0"

if not exist venv (
    echo [ERROR] venv not found. Run setup.bat first.
    pause
    exit /b 1
)

REM Enable ANSI colours on Windows 10+
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1

echo Starting DayTradingBot status monitor...
venv\Scripts\python.exe scripts\engine_status.py

echo.
echo Dashboard exited.
pause
