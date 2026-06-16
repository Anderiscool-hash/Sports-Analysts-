@echo off
REM ============================================================
REM  SportEdge - single launcher.
REM  Double-click to choose: watch real live games (paper-track)
REM  or run a paper-trading session over cached data. One window,
REM  one dependency check, both tools.
REM ============================================================

REM Run from this script's own folder regardless of where it's launched.
cd /d "%~dp0"

REM src-layout project: make the package importable without installing.
set "PYTHONPATH=src"

REM Pick an interpreter: prefer "python"; fall back to the "py" launcher.
set "PYEXE=python"
where python >nul 2>nul || set "PYEXE=py"

REM Ensure runtime packages are present once for both tools.
%PYEXE% -c "import pandas, pyarrow, rich, pydantic, yaml, dotenv, tenacity" >nul 2>nul
if errorlevel 1 %PYEXE% -m pip install pandas pyarrow rich pydantic pyyaml python-dotenv tenacity cryptography

:menu
echo.
echo ============================================================
echo   SportEdge
echo ============================================================
echo   [1] Watch live games + paper-track (real games)
echo   [2] Paper-trade simulation over cached data (see PnL now)
echo   [3] Arm: wait for next Kalshi-covered live game, auto-trade
echo   [4] Paper P^&L report
echo   [Q] Quit
echo ============================================================
set "choice="
set /p choice="Choose an option: "

if /i "%choice%"=="1" goto opt_live
if /i "%choice%"=="2" goto opt_sim
if /i "%choice%"=="3" goto opt_arm
if /i "%choice%"=="4" goto opt_report
if /i "%choice%"=="Q" goto end
echo Unrecognized choice: "%choice%"
goto menu

:opt_live
echo.
echo Launching live dashboard. Pick a game; Ctrl-C to stop.
%PYEXE% -m sportedge.live.dashboard
goto after

:opt_sim
echo.
%PYEXE% scripts\paper_sim.py
goto after

:opt_arm
echo.
echo Waiting for a Kalshi-covered live game; auto-trades when one appears. Ctrl-C to stop.
%PYEXE% -m sportedge.live.dashboard --wait-ready
goto after

:opt_report
echo.
%PYEXE% scripts\paper_report.py
goto after

:after
echo.
echo --- done. Returning to menu ---
goto menu

:end
echo.
echo Bye.
