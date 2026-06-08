# DayTradingBot — watchdog script
# Scheduled every 5 minutes by Task Scheduler.
# If main.py is not running, starts it via start_bot.ps1.

$projectDir = (Resolve-Path "$PSScriptRoot\..").Path
$logFile    = Join-Path $projectDir "logs\watchdog.log"
$timestamp  = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Write-Log($msg) {
    $line = "$timestamp  $msg"
    Add-Content -Path $logFile -Value $line -Encoding UTF8
    Write-Host $line
}

# Check if main.py is already running
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like "*main.py*" }

if ($running) {
    Write-Log "OK — engine running (PID $($running[0].ProcessId)). No action needed."
    exit 0
}

Write-Log "RESTART — main.py not found. Starting engine via start_bot.ps1 ..."

try {
    & "$PSScriptRoot\start_bot.ps1"
    Write-Log "RESTART — start_bot.ps1 completed."
} catch {
    Write-Log "ERROR — start_bot.ps1 failed: $_"
    exit 1
}
