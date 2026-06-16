@echo off
REM ============================================================
REM  SportEdge paper-trading runner (Windows).
REM  Double-click to run a paper-trading session over cached
REM  aligned model/market data and print the P&L. No live orders,
REM  no waiting for a live game -- paper trading on demand.
REM ============================================================

REM Run from this script's own folder regardless of where it's launched.
cd /d "%~dp0"

REM src-layout project: make the package importable without installing.
set "PYTHONPATH=src"

REM Pick an interpreter: prefer "python"; fall back to the "py" launcher.
set "PYEXE=python"
where python >nul 2>nul || set "PYEXE=py"

REM Ensure runtime packages are present; install once if not.
%PYEXE% -c "import pandas, pyarrow, rich, pydantic, yaml, dotenv" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages, one moment...
    %PYEXE% -m pip install pandas pyarrow rich pydantic pyyaml python-dotenv tenacity cryptography
)

%PYEXE% scripts\paper_sim.py %*

REM Keep the window open so the P&L stays visible.
echo.
pause
