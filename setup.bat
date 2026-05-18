@echo off
:: Indeterminate — 一键环境配置 (Flask + Streamlit 前后端分离版)
:: 双击此文件即可运行

cd /d "%~dp0"

echo.
echo ===== Indeterminate (Flask + Streamlit) 环境配置 =====
echo 正在启动 PowerShell 配置脚本...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 脚本执行出错！请尝试右键 setup.ps1 → "使用 PowerShell 运行"
    pause
)
