@echo off
:: Indeterminate — 一键环境配置 (批处理启动器)
:: 双击此文件即可运行，自动绕过 PowerShell 执行策略

cd /d "%~dp0"

echo.
echo ===== Indeterminate 环境配置 =====
echo 正在启动 PowerShell 配置脚本...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 脚本执行出错！请尝试右键 setup.ps1 → "使用 PowerShell 运行"
    pause
)
