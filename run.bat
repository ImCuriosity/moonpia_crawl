@echo off
REM ===========================================================
REM  Munpia Data Collector - Windows launcher
REM  Just double-click this file.
REM
REM  NOTE: This file is intentionally ASCII-only. Windows reads
REM  .bat files using the console codepage, so non-ASCII text
REM  here corrupts parsing. All Korean UI lives in Python.
REM ===========================================================
setlocal enabledelayedexpansion

chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0"

echo.
echo ===========================================================
echo   Munpia Web Novel Data Collector
echo ===========================================================
echo.

REM ---- 1. Locate Python -------------------------------------
set "PY="
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" >nul 2>&1 <nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" >nul 2>&1 <nul
        if not errorlevel 1 set "PY=python"
    )
)

if not defined PY (
    echo   [ERROR] Python 3.8+ was not found. Please install it first.
    echo.
    echo     1^) https://www.python.org/downloads/
    echo        Check "Add Python to PATH" during installation.
    echo.
    echo     2^) Or run:  winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo   Found %%v

REM ---- 2. Prepare virtual environment ------------------------
set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo   First run - setting up. This takes 1-2 minutes...
    echo.
    REM "<nul" keeps setup commands from consuming the user's stdin,
    REM which would swallow the first answers when input is piped in.
    echo   [1/2] Creating virtual environment...
    %PY% -m venv .venv <nul
    if errorlevel 1 (
        echo   [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   [2/2] Installing packages ^(requests, pandas^)...
    "%VENV_PY%" -m pip install --upgrade pip --quiet <nul
    "%VENV_PY%" -m pip install --quiet requests pandas <nul
    if errorlevel 1 (
        echo   [ERROR] Package installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo   Setup complete.
    echo.
)

REM ---- 3. Run ------------------------------------------------
"%VENV_PY%" -m munpia.wizard
set "EXITCODE=%errorlevel%"

echo.
echo ===========================================================
if not "%EXITCODE%"=="0" echo   Exited with code %EXITCODE%.
echo   Press any key to close this window.
echo ===========================================================
pause >nul
exit /b %EXITCODE%
