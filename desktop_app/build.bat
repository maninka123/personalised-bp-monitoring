@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion
title BP Profile Monitor — Build System
color 0F

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                            ║
echo  ║        BP Profile Monitor — Automated Build                ║
echo  ║        Sleep-Aware Blood Pressure Analysis                 ║
echo  ║                                                            ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

set "ROOT=%~dp0"
set "BACKEND=%ROOT%python-backend"
set "PROJECT_ROOT=%ROOT%.."
set "PERSONALISED=%PROJECT_ROOT%\personalised-bp-monitoring"
if exist "%PROJECT_ROOT%\clinical_report_utils.py" set "PERSONALISED=%PROJECT_ROOT%"
set STEPS=6
set ERRORS=0

:: ── Step 1: Check Python ──────────────────────────────────────────
echo  [1/%STEPS%] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo     [ERROR] Python is not installed or not in PATH.
    echo     Please install Python 3.10+ and try again.
    set /a ERRORS+=1
    goto :done
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo     Found: %%v
echo     [OK]
echo.

:: ── Step 2: Check Node.js ─────────────────────────────────────────
echo  [2/%STEPS%] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo     [ERROR] Node.js is not installed or not in PATH.
    echo     Please install Node.js 18+ and try again.
    set /a ERRORS+=1
    goto :done
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do echo     Found Node: %%v
for /f "tokens=*" %%v in ('npm --version 2^>^&1') do echo     Found npm: %%v
echo     [OK]
echo.

:: ── Step 3: Install Python dependencies ───────────────────────────
echo  [3/%STEPS%] Installing Python dependencies...
pip install -r "%BACKEND%\requirements.txt" pyinstaller --quiet 2>nul
if errorlevel 1 (
    echo     [WARN] Some pip packages may have had issues, continuing...
) else (
    echo     [OK]
)
echo.

:: ── Step 4: Bundle Python backend with PyInstaller ────────────────
echo  [4/%STEPS%] Bundling Python backend with PyInstaller...
echo     This may take 1-3 minutes...
cd /d "%BACKEND%"

set "ADD_DATA1=%PERSONALISED%\clinical_report_utils.py;."
set "ADD_DATA2=%PERSONALISED%\sleep_aware_bp_framework.py;."

if exist "%PROJECT_ROOT%\outputs\dryad_thresholds.csv" (
    set "ADD_DATA3=--add-data %PROJECT_ROOT%\outputs\dryad_thresholds.csv;outputs"
) else (
    set "ADD_DATA3="
)

pyinstaller --onefile --noconfirm --clean --name server ^
    --add-data "%ADD_DATA1%" ^
    --add-data "%ADD_DATA2%" ^
    %ADD_DATA3% ^
    server.py >nul 2>&1

if not exist "%BACKEND%\dist\server.exe" (
    echo     [ERROR] PyInstaller build failed.
    echo     Try running manually: cd python-backend ^&^& pyinstaller --onefile server.py
    set /a ERRORS+=1
    goto :step5
) else (
    for %%f in ("%BACKEND%\dist\server.exe") do echo     Built: server.exe (%%~zf bytes^)
    echo     [OK]
)
echo.

:step5
cd /d "%ROOT%"

:: ── Step 5: Install npm dependencies ──────────────────────────────
echo  [5/%STEPS%] Installing npm dependencies...
call npm install --quiet 2>nul
echo     [OK]
echo.

:: ── Step 6: Build Electron installer ──────────────────────────────
echo  [6/%STEPS%] Building Windows installer with electron-builder...
echo     This may take 2-5 minutes...
call npx electron-builder --win 2>nul
if errorlevel 1 (
    echo     [ERROR] electron-builder failed.
    echo     Try running manually: npx electron-builder --win
    set /a ERRORS+=1
) else (
    echo     [OK]
    echo.
    echo  ── Build output ──────────────────────────────────────────────
    if exist "%ROOT%dist" (
        dir /b "%ROOT%dist\*.exe" 2>nul
    )
)
echo.

:done
echo  ══════════════════════════════════════════════════════════════
if %ERRORS% equ 0 (
    echo   BUILD COMPLETE — No errors!
    echo.
    echo   Installer location: %ROOT%dist\
    echo   To run in dev mode:  npm start
) else (
    echo   BUILD FINISHED with %ERRORS% error(s).
    echo   Review the messages above and fix any issues.
)
echo  ══════════════════════════════════════════════════════════════
echo.
pause
