# DayTradingBot — start script (called by Windows Task Scheduler)
# Starts the trading bot and the FastAPI dashboard backend.
#
# Uses $PSScriptRoot to locate the project root dynamically — avoids
# encoding problems with Unicode path characters (e.g. Chinese folder names)
# when spawned by Task Scheduler or a child PowerShell process.

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

# Start the FastAPI dashboard backend
Start-Process -FilePath $python `
    -ArgumentList "-m uvicorn src.dashboard.api:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $projectDir `
    -WindowStyle Hidden

# Start the trading bot
Start-Process -FilePath $python `
    -ArgumentList "main.py --broker alpaca" `
    -WorkingDirectory $projectDir `
    -WindowStyle Hidden

Write-Host "DayTradingBot started. Project: $projectDir"
