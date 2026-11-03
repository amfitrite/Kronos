@echo off
title Kronos Foundation Core - XAUUSD
cd /d "%~dp0"
echo Starting Kronos Foundation Core...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error. Press any key to close...
    pause >nul
)
