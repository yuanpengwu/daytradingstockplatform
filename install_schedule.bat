@echo off
REM ============================================================
REM  Registers (or updates) a Windows scheduled task that runs
REM  the DayTradingBot every weekday morning.
REM  Run this ONCE. To remove it later, run: uninstall_schedule.bat
REM ============================================================
setlocal
cd /d "%~dp0"

set "TASKNAME=DayTradingBot"
set "BOTPATH=%~dp0run_bot.bat"

if not exist "%BOTPATH%" (
    echo [ERROR] run_bot.bat not found next to this script.
    pause
    exit /b 1
)

echo This will schedule the bot to run every weekday (Mon-Fri).
echo.
echo IMPORTANT: enter the time in YOUR computer's local time.
echo   - US Eastern : 09:20
echo   - US Central : 08:20
echo   - US Pacific : 06:20
echo   (a few minutes before the 9:30 ET market open)
echo.
set "STARTTIME=09:20"
set /p STARTTIME="Start time [default 09:20]: "

echo.
echo Creating scheduled task "%TASKNAME%" at %STARTTIME%, Mon-Fri...
schtasks /create /tn "%TASKNAME%" /tr "\"%BOTPATH%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st %STARTTIME% /f

if errorlevel 1 (
    echo.
    echo [ERROR] Could not create the task. You can set it up manually via
    echo         Task Scheduler instead - ask Claude for the step-by-step.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Done. The bot will launch every weekday at %STARTTIME%.
echo.
echo  Useful commands:
echo    schtasks /run    /tn "%TASKNAME%"    (run it now, to test)
echo    schtasks /query  /tn "%TASKNAME%"    (check it exists)
echo    schtasks /delete /tn "%TASKNAME%" /f (remove it)
echo ============================================
pause
