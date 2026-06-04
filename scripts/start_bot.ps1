# DayTradingBot — start script
# Pulls latest code, starts the engine, opens the live status monitor.
# Called by market_runner.py or manually.

$projectDir = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $projectDir

# Load .env so API keys are available to child processes
$envFile = Join-Path $projectDir ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

$python = "C:\Users\yuanp\AppData\Local\Programs\Python\Python311\python.exe"

# ── Start the trading engine (visible in its own terminal window) ─────────────
$engineWindow = Start-Process -FilePath $python `
    -ArgumentList "main.py --broker alpaca" `
    -WorkingDirectory $projectDir `
    -PassThru `
    -WindowStyle Minimized `
    -RedirectStandardOutput (Join-Path $projectDir "logs\bot_stdout.log") `
    -RedirectStandardError  (Join-Path $projectDir "logs\bot_err.log")

Write-Host "Engine started (PID $($engineWindow.Id))"

# Note: main.py opens the status monitor automatically on startup.
# No need to open it here.
Write-Host "Engine started. Status monitor will open automatically. PID: $($engineWindow.Id)"
