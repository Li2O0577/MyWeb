param(
    [switch]$Force,
    [switch]$IncludeRuntimeData
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$targets = @(
    "frontend\dist",
    "frontend\.vite",
    "frontend\.tsbuild",
    "frontend\tsconfig.tsbuildinfo",
    "frontend\tsconfig.node.tsbuildinfo",
    ".pytest_cache"
)

$cacheDirs = Get-ChildItem -Path $root -Recurse -Directory -Force -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { $_.FullName }

$paths = @()
foreach ($target in $targets) {
    $paths += Join-Path $root $target
}
$paths += $cacheDirs

if ($IncludeRuntimeData) {
    $paths += Join-Path $root "backend\sessions"
    $paths += Join-Path $root "backend\models\registry.json"
    $paths += Get-ChildItem -Path (Join-Path $root "backend\models") -Directory -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "__pycache__" } |
        ForEach-Object { $_.FullName }
}

$existing = $paths | Where-Object { Test-Path -LiteralPath $_ } | Sort-Object -Unique

if (-not $existing) {
    Write-Host "Nothing to clean." -ForegroundColor Green
    return
}

Write-Host "Cleanup targets:" -ForegroundColor Cyan
$existing | ForEach-Object { Write-Host "  $_" }

if (-not $Force) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Force to delete these files." -ForegroundColor Yellow
    Write-Host "Add -IncludeRuntimeData only when you intentionally want to remove uploaded sessions and trained model files." -ForegroundColor Yellow
    return
}

foreach ($path in $existing) {
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Cleanup finished." -ForegroundColor Green
