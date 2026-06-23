@echo off
REM Shared launcher bootstrap. Call this file; do not run project commands here.

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "PYEXE="

REM Prefer .venv only when its interpreter can actually start. A moved or
REM upgraded Microsoft Store Python can leave behind a crashing venv executable.
if not exist "%VENV_PY%" goto system_python
"%VENV_PY%" --version >nul 2>nul
if not "%ERRORLEVEL%"=="0" goto system_python
set PYEXE="%VENV_PY%"
goto verify

REM Use the known-good system interpreter when .venv is absent or broken.
:system_python
where python >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=python"
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python 3.11 or newer was not found.
        exit /b 1
    )
    set "PYEXE=py -3"
)

%PYEXE% --version >nul 2>nul
if not "%ERRORLEVEL%"=="0" (
    echo ERROR: The selected Python interpreter could not start.
    exit /b 1
)
echo NOTE: .venv is unavailable; using the system Python for this session.

:verify
%PYEXE% -c "import cryptography, pandas, pyarrow, requests, rich, pydantic, yaml, dotenv, tenacity" >nul 2>nul
if "%ERRORLEVEL%"=="0" goto ready

echo Installing or repairing SportEdge dependencies...
%PYEXE% -m pip install -r requirements.txt
if not "%ERRORLEVEL%"=="0" (
    echo ERROR: Dependency installation failed.
    exit /b 1
)

:ready
%PYEXE% -c "from sportedge.config import load_config; assert load_config()" >nul 2>nul
if not "%ERRORLEVEL%"=="0" (
    echo ERROR: SportEdge failed its startup import check.
    exit /b 1
)
exit /b 0
