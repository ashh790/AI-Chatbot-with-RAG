@echo off
REM Double-click to install everything and verify the setup.
REM Safe to re-run any time.

cd /d "%~dp0"
title AI Chatbot - Setup

echo ============================================================
echo   AI Chatbot - one-time setup
echo ============================================================
echo.

echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo       ERROR: 'python' is not on your PATH.
    echo       Install Python 3.10+ from https://python.org and tick
    echo       "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
python --version
echo.

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo       done
echo.

echo [3/5] Installing core packages...
echo       (flask, openai, dotenv, pypdf, python-docx, requests, ddgs)
python -m pip install --quiet flask python-dotenv openai pypdf python-docx requests ddgs
if errorlevel 1 (
    echo       ERROR: core install failed. Scroll up for the reason.
    pause
    exit /b 1
)
echo       done
echo.

echo [4/5] Installing chromadb for RAG...
echo       This one is large and can take a few minutes. If it fails,
echo       everything except document search still works.
python -m pip install --quiet chromadb
if errorlevel 1 (
    echo       WARNING: chromadb failed to install.
    echo       Chat, tools and web search will still work.
    echo       RAG document search will be disabled.
) else (
    echo       done
)
echo.

echo [5/5] Running diagnostics...
echo ------------------------------------------------------------
python doctor.py
echo ------------------------------------------------------------
echo.

echo Setup finished.
echo.
echo   Next:  double-click start.bat   (or run: python app.py)
echo   Then:  http://127.0.0.1:5000
echo.
echo   To add your own documents for RAG:
echo     1. put PDFs / text files in the docs folder
echo     2. run: python ingest.py
echo.
pause
