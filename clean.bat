@echo off
:: Indeterminate — Clean Cache & Model Files
:: Double-click to run

cd /d "%~dp0"

echo.
echo ===== Indeterminate - Clean Cache ^& Models =====
echo Launching cleanup script...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0clean.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Script failed! Right-click clean.ps1 -^> "Run with PowerShell"
    pause
)
