@echo off
:: Indeterminate — Quick Launcher (Flask + Streamlit)
:: Double-click to run

cd /d "%~dp0"

echo.
echo ===== Indeterminate (Flask + Streamlit) =====
echo Launching startup script...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Script failed! Right-click start.ps1 -^> "Run with PowerShell"
    pause
)
