# DayTradingBot — stop script (called by Windows Task Scheduler)
# Gracefully stops the trading bot and dashboard processes.

Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $cmdLine = $_.CommandLine
    if ($cmdLine -like "*main.py*" -or $cmdLine -like "*uvicorn*src.dashboard.api*") {
        Write-Host "Stopping PID $($_.ProcessId): $cmdLine"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "DayTradingBot stopped."
