@echo off
REM ============================================================================
REM  Build the SportEdge Windows desktop .exe.
REM
REM  Plain build (decompilable - fine for a closed beta):
REM      build_exe.bat
REM  Hardened build (obfuscate first with PyArmor - recommended for selling):
REM      build_exe.bat obf
REM
REM  Output: dist\SportEdge\SportEdge.exe  plus  models\  and  config\  beside it.
REM  Ship the WHOLE dist\SportEdge folder. See PACKAGING.md for the full story on
REM  protecting the source (a local .exe alone is NOT secure - move the model to a
REM  server for real protection).
REM ============================================================================
setlocal
cd /d "%~dp0"

echo [1/4] Installing build dependencies...
python -m pip install --quiet --upgrade pyinstaller || goto :fail

set SRC=src\sportedge\app.py
if /I "%~1"=="obf" (
    echo [2/4] Obfuscating with PyArmor...
    python -m pip install --quiet --upgrade pyarmor || goto :fail
    REM Obfuscate the whole package into build\obf, then build from there.
    pyarmor gen -O build\obf -r src\sportedge || goto :fail
    set SRC=build\obf\sportedge\app.py
) else (
    echo [2/4] Skipping obfuscation ^(plain build^).
)

echo [3/4] Packaging with PyInstaller...
pyinstaller --noconfirm --clean --onedir --name SportEdge ^
    --paths src ^
    --collect-submodules sportedge ^
    --collect-submodules sklearn ^
    --collect-submodules scipy ^
    --collect-data sklearn ^
    --hidden-import joblib ^
    "%SRC%" || goto :fail

echo [4/4] Copying runtime assets beside the exe...
if exist models    xcopy /E /I /Y models    "dist\SportEdge\models"  >nul
if exist config    xcopy /E /I /Y config    "dist\SportEdge\config"  >nul

echo.
echo BUILD OK -> dist\SportEdge\SportEdge.exe
echo Ship the entire dist\SportEdge\ folder. Customers place their license at
echo   %%APPDATA%%\SportEdge\license.key  (or set SPORTEDGE_LICENSE).
goto :eof

:fail
echo.
echo BUILD FAILED. See the error above.
exit /b 1
