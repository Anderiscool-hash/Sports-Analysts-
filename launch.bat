@echo off
REM ============================================================
REM  SportEdge - single launcher.
REM  Double-click to choose: watch real live games (paper-track)
REM  or run a paper-trading session over cached data. Uses the
REM  project .venv and the current v2 execution/risk stack.
REM ============================================================

cd /d "%~dp0"
call "%~dp0_sportedge_setup.bat"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

:menu
echo.
echo ============================================================
echo   SportEdge
echo ============================================================
echo   [1] Watch live games + paper-track (real games)
echo   [2] Paper-trade simulation over cached data (see PnL now)
echo   [3] Arm: wait for a Kalshi-covered game (uses config safety gates)
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
echo Waiting for a Kalshi-covered live game. Current config controls paper/live mode.
echo Live orders still require confirm_live and every paper/risk gate. Ctrl-C to stop.
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
