param(
    [string]$CondaEnv = "ml0",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 5001
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backendDir = Join-Path $root "backend"
$distIndex = Join-Path $root "frontend\dist\index.html"

if (-not (Test-Path $distIndex)) {
    throw "frontend/dist/index.html not found. Run build-prod.ps1 first."
}

$env:FLASK_HOST = $HostName
$env:FLASK_PORT = "$Port"
$env:FLASK_DEBUG = "0"

Write-Host "Starting Indeterminate production mode..." -ForegroundColor Cyan
Write-Host "URL:       http://$HostName`:$Port" -ForegroundColor Cyan
Write-Host "Conda env: $CondaEnv" -ForegroundColor Cyan

Set-Location $backendDir
conda run -n $CondaEnv python app.py
