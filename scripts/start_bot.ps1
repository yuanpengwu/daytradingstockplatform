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

# ── Open the live status monitor in a new terminal window ─────────────────────
# Gives it a title so it's easy to find in the taskbar.
Start-Process -FilePath "powershell" `
    -ArgumentList "-NoExit", "-Command", `
        "`$host.ui.RawUI.WindowTitle = 'DayTradingBot Monitor'; & '$python' '$projectDir\scripts\engine_status.py'" `
    -WindowStyle Normal

Write-Host "Status monitor opened. Engine PID: $($engineWindow.Id)"
