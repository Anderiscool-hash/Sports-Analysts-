@echo off
REM ============================================================
REM  SportEdge live game dashboard launcher (Windows).
REM  Double-click this file to open the current dashboard in a console.
REM  Execution mode and safety gates come from config/config.yaml.
REM ============================================================

cd /d "%~dp0"
call "%~dp0_sportedge_setup.bat"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

%PYEXE% -m sportedge.live.dashboard %*

REM Keep the window open if the dashboard exits or errors.
echo.
pause
