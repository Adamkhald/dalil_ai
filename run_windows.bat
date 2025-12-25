@echo off
TITLE Dalil AI Launcher

echo Checking for Python...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.8+ to run Dalil AI.
    pause
    exit /b
)

echo Starting Dalil AI using your system libraries...
echo -----------------------------------------------
python main.py
pause
