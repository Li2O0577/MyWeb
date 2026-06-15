param(
    [switch]$Install,
    [string]$NodeDir = "C:\Program Files\nodejs"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$frontendDir = Join-Path $root "frontend"
$npmCmd = Join-Path $NodeDir "npm.cmd"

if (-not (Test-Path $npmCmd)) {
    throw "npm.cmd not found at $npmCmd. Install Node.js or pass -NodeDir."
}

$env:Path = "$NodeDir;$env:Path"
Set-Location $frontendDir

if ($Install) {
    & $npmCmd install
}

& $npmCmd run build

Write-Host ""
Write-Host "Production frontend built at frontend/dist." -ForegroundColor Green
Write-Host "Run start-prod.ps1 to serve it from Flask." -ForegroundColor Cyan
