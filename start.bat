@echo off
setlocal

cd /d "%~dp0"

echo.
echo ===== Indeterminate Start =====
echo Starting Flask and Streamlit...
echo.

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"

if errorlevel 1 (
    echo.
    echo Start failed. Please open PowerShell in this folder and run:
    echo powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
    echo.
    pause
    exit /b 1
)
