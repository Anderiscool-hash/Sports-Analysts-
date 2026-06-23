@echo off
REM SportEdge browser UI. Localhost only; paper tracking is off until enabled.

cd /d "%~dp0"
call "%~dp0_sportedge_setup.bat"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

echo Opening SportEdge at http://127.0.0.1:8765
echo Close this window or press Ctrl-C to stop the UI server.
%PYEXE% -m sportedge.live.web_dashboard %*

echo.
pause
