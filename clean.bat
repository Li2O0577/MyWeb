@echo off
setlocal

cd /d "%~dp0"

echo.
echo ===== Indeterminate Clean =====
echo Starting cleanup script...
echo.

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0clean.ps1"

if errorlevel 1 (
    echo.
    echo Clean failed. Please open PowerShell in this folder and run:
    echo powershell -NoProfile -ExecutionPolicy Bypass -File .\clean.ps1
    echo.
    pause
    exit /b 1
)
