@echo off
REM Removes the DayTradingBot scheduled task.
schtasks /delete /tn "DayTradingBot" /f
echo.
echo Scheduled task removed (if it existed).
pause
