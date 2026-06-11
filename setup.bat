@echo off
setlocal

cd /d "%~dp0"

echo.
echo ===== Indeterminate Setup =====
echo Starting PowerShell setup script...
echo.

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

if errorlevel 1 (
    echo.
    echo Setup failed. Please open PowerShell in this folder and run:
    echo powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
    echo.
    pause
    exit /b 1
)

echo.
echo Setup finished.
pause
