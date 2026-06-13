@echo off
REM ============================================================
REM  SportEdge live game dashboard launcher (Windows).
REM  Double-click this file to open the dashboard in a console.
REM ============================================================

REM Run from this script's own folder regardless of where it's launched.
cd /d "%~dp0"

REM src-layout project: make the package importable without installing.
set "PYTHONPATH=src"

REM Pick an interpreter: prefer "python" (the one carrying the packages here);
REM fall back to the "py" launcher if "python" isn't on PATH.
set "PYEXE=python"
where python >nul 2>nul || set "PYEXE=py"

REM Ensure the dashboard's runtime packages are present; install once if not.
%PYEXE% -c "import rich, requests, pydantic, yaml, dotenv" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages, one moment...
    %PYEXE% -m pip install rich requests pydantic pyyaml python-dotenv cryptography
)

%PYEXE% -m sportedge.live.dashboard %*

REM Keep the window open if the dashboard exits or errors.
echo.
pause
