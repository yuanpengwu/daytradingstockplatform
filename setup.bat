@echo off
REM ============================================================
REM  DayTradingBot - one-click setup for Windows
REM ============================================================
cd /d "%~dp0"
echo ============================================
echo  DayTradingBot setup
echo ============================================
echo.

REM --- 1. Check Python is available ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

REM --- 2. Create virtual environment ---
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM --- 3. Activate + upgrade pip ---
call venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip

REM --- 4. Install dependencies ---
echo.
echo Installing dependencies - this may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed. See messages above.
    pause
    exit /b 1
)

REM --- 5. Verify ---
echo.
echo Verifying core packages...
python -c "import numpy, pandas, yfinance, alpaca, streamlit, sklearn, xgboost; print('  All packages import OK')"
if errorlevel 1 (
    echo [WARN] Some packages failed to import - check messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete!
echo.
echo  Next steps (run these in this folder):
echo    venv\Scripts\activate
echo    python demo_offline.py                 (offline sanity check)
echo    python main.py --broker alpaca --once   (one live cycle)
echo    python main.py --broker alpaca          (continuous)
echo ============================================
pause
