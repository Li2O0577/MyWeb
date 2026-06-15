param(
    [string]$CondaEnv = "ml0",
    [int]$FlaskPort = 5001,
    [int]$ReactPort = 5173,
    [string]$NodeDir = "C:\Program Files\nodejs"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$npmCmd = Join-Path $NodeDir "npm.cmd"

if (-not (Test-Path $npmCmd)) {
    throw "npm.cmd not found at $npmCmd. Install Node.js or pass -NodeDir."
}

Write-Host "Starting Indeterminate React development mode..." -ForegroundColor Cyan
Write-Host "Flask API:     http://127.0.0.1:$FlaskPort" -ForegroundColor Cyan
Write-Host "React Vite:    http://127.0.0.1:$ReactPort" -ForegroundColor Cyan
Write-Host "Conda env:     $CondaEnv" -ForegroundColor Cyan

$backendCommand = @"
`$Host.UI.RawUI.WindowTitle = 'Indeterminate Flask API :$FlaskPort'
`$env:FLASK_PORT = '$FlaskPort'
`$env:FLASK_HOST = '127.0.0.1'
`$env:FLASK_DEBUG = '1'
Set-Location '$backendDir'
conda run -n '$CondaEnv' python app.py
"@

$frontendCommand = @"
`$Host.UI.RawUI.WindowTitle = 'Indeterminate React Dev :$ReactPort'
`$env:Path = '$NodeDir;' + `$env:Path
Set-Location '$frontendDir'
& '$npmCmd' run dev -- --host 127.0.0.1 --port $ReactPort
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand

Write-Host ""
Write-Host "Development services launched. Close both service windows to stop." -ForegroundColor Green
