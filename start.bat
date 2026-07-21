@echo off
REM Double-click this to restart the chatbot cleanly.
REM Kills any stale server holding port 5000, clears cached bytecode,
REM then starts a fresh one.

cd /d "%~dp0"
title AI Chatbot Server

echo ============================================
echo  AI Chatbot - clean restart
echo ============================================
echo.

echo [1/4] Stopping anything already on port 5000...
set FOUND=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" ^| findstr "LISTENING"') do (
    echo       killing stale process PID %%a
    taskkill /PID %%a /F >nul 2>&1
    set FOUND=1
)
if "%FOUND%"=="0" echo       nothing was running - good
echo.

echo [2/4] Clearing cached bytecode...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo       done
echo.

echo [3/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo       ERROR: 'python' not found on your PATH.
    echo       Install Python 3.10+ or use the full path to python.exe
    echo.
    pause
    exit /b 1
)
python --version
echo.

echo [4/4] Starting server...
echo.
echo       Open:  http://127.0.0.1:5000
echo       Press Ctrl+Shift+R in the browser the first time.
echo       Close this window to stop the server.
echo.
echo --------------------------------------------
python app.py

echo.
echo --------------------------------------------
echo Server stopped.
pause
