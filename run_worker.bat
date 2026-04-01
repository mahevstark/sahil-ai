@echo off
title Sahil AI - Transcription Worker
cd /d "%~dp0"

echo ============================================
echo   Sahil AI - Distributed Transcription
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Please install Python 3.11+ from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Install / update dependencies
echo [Setup] Installing dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo [Setup] Dependencies OK.
echo.

:: Launch worker
python worker.py

echo.
echo Worker stopped.
pause
