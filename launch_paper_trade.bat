@echo off
REM ============================================================
REM  SportEdge paper-trading runner (Windows).
REM  Double-click to run a paper-trading session over cached
REM  aligned model/market data and print the P&L. No live orders,
REM  no waiting for a live game -- paper trading on demand.
REM ============================================================

cd /d "%~dp0"
call "%~dp0_sportedge_setup.bat"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

%PYEXE% scripts\paper_sim.py %*

REM Keep the window open so the P&L stays visible.
echo.
pause
