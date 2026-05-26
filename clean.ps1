# =============================================
#  Indeterminate — Clean Cache & Model Files
#  Usage: .\clean.ps1          (interactive)
#         .\clean.ps1 -Force   (skip confirm)
#         .\clean.ps1 -WhatIf  (preview only)
# =============================================

param(
    [switch]$Force,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "Indeterminate Cleaner"

$dryRun = $WhatIf.IsPresent
$skipConfirm = $Force.IsPresent -or $dryRun

function Write-Step { Write-Host "`n$args" -ForegroundColor Yellow }
function Write-Del  { Write-Host "  $args" -ForegroundColor Red }
function Write-Info { Write-Host "  $args" -ForegroundColor DarkGray }
function Write-Warn { Write-Host "  $args" -ForegroundColor Magenta }

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Indeterminate - Clean Cache & Models" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

if ($dryRun) {
    Write-Warn "[Preview mode] No files will be deleted"
}

# collect all items to clean
$items = [System.Collections.ArrayList]::new()

# ── 1. Python __pycache__ ──
Write-Step "[1/6] Scanning __pycache__ directories..."
$pycacheDirs = @(Get-ChildItem -Path $scriptDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue)
if ($pycacheDirs.Count -gt 0) {
    Write-Info "Found $($pycacheDirs.Count) __pycache__ dir(s)"
    foreach ($d in $pycacheDirs) {
        $sz = (Get-ChildItem $d.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        [void]$items.Add([PSCustomObject]@{ Type="__pycache__"; Path=$d.FullName; SizeBytes=$sz })
    }
} else {
    Write-Info "None found"
}

# ── 2. Trained model files ──
Write-Step "[2/6] Scanning trained model files..."
$modelPatterns = @("*.pth", "*.pt", "*.onnx", "*.h5", "*.pb", "*.pkl", "*.joblib", "*.safetensors", "*.ckpt", "*.bin", "*.sav")
$modelFiles = @()
foreach ($pattern in $modelPatterns) {
    $found = @(Get-ChildItem -Path $scriptDir -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue | Where-Object {
        $_.FullName -notmatch '[\\/]\.git[\\/]'
    })
    $modelFiles += $found
}
if ($modelFiles.Count -gt 0) {
    Write-Info "Found $($modelFiles.Count) model file(s)"
    foreach ($f in $modelFiles) {
        [void]$items.Add([PSCustomObject]@{ Type="Model file"; Path=$f.FullName; SizeBytes=$f.Length })
    }
} else {
    Write-Info "None found"
}

# ── 3. Model version directories ──
Write-Step "[3/6] Scanning model version directories..."
$modelBase = Join-Path $scriptDir "backend\models"
if (Test-Path $modelBase) {
    $modelSubDirs = @(Get-ChildItem -Path $modelBase -Directory -ErrorAction SilentlyContinue)
    $foundVersion = $false
    foreach ($subDir in $modelSubDirs) {
        $versionDirs = @(Get-ChildItem -Path $subDir.FullName -Directory -ErrorAction SilentlyContinue)
        foreach ($vd in $versionDirs) {
            $versionFiles = @(Get-ChildItem -Path $vd.FullName -Recurse -File -ErrorAction SilentlyContinue)
            if ($versionFiles.Count -gt 0) {
                if (-not $foundVersion) { $foundVersion = $true }
                Write-Info "Version dir: $($vd.FullName) ($($versionFiles.Count) file(s))"
                foreach ($f in $versionFiles) {
                    [void]$items.Add([PSCustomObject]@{ Type="Model version"; Path=$f.FullName; SizeBytes=$f.Length })
                }
                [void]$items.Add([PSCustomObject]@{ Type="Model version dir"; Path=$vd.FullName; SizeBytes=0 })
            }
        }
    }
    if (-not $foundVersion) {
        Write-Info "None found"
    }
} else {
    Write-Info "models dir not found"
}

# ── 4. Session data ──
Write-Step "[4/6] Scanning session data..."
$sessionsDir = Join-Path $scriptDir "backend\sessions"
if (Test-Path $sessionsDir) {
    $sessionFiles = @(Get-ChildItem -Path $sessionsDir -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match '\.(parquet|meta\.json)$'
    })
    if ($sessionFiles.Count -gt 0) {
        Write-Info "Found $($sessionFiles.Count) session file(s)"
        foreach ($f in $sessionFiles) {
            [void]$items.Add([PSCustomObject]@{ Type="Session"; Path=$f.FullName; SizeBytes=$f.Length })
        }
    } else {
        Write-Info "None found"
    }
} else {
    Write-Info "sessions dir not found"
}

# ── 5. Logs ──
Write-Step "[5/6] Scanning log files..."
$logFound = $false
$logDirs = @(Get-ChildItem -Path $scriptDir -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object {
    ($_.Name -eq "logs" -or $_.Name -eq "log") -and ($_.FullName -notmatch '[\\/]\.git[\\/]')
})
foreach ($ld in $logDirs) {
    $logContents = @(Get-ChildItem -Path $ld.FullName -Recurse -File -ErrorAction SilentlyContinue)
    if ($ld.Name -eq "logs") {
        foreach ($f in $logContents) {
            if (-not $logFound) { $logFound = $true }
            [void]$items.Add([PSCustomObject]@{ Type="Log file"; Path=$f.FullName; SizeBytes=$f.Length })
        }
    } else {
        $logFound = $true
        [void]$items.Add([PSCustomObject]@{ Type="Log dir"; Path=$ld.FullName; SizeBytes=0 })
    }
}
$logFiles = @(Get-ChildItem -Path $scriptDir -Recurse -File -Filter "*.log" -ErrorAction SilentlyContinue | Where-Object {
    $_.FullName -notmatch '[\\/]\.git[\\/]'
})
foreach ($f in $logFiles) {
    if (-not $logFound) { $logFound = $true }
    [void]$items.Add([PSCustomObject]@{ Type="Log file"; Path=$f.FullName; SizeBytes=$f.Length })
}
if (-not $logFound) {
    Write-Info "None found"
} else {
    Write-Info "Done"
}

# ── 6. Temp files ──
Write-Step "[6/6] Scanning temp files..."
$tempFiles = @(Get-ChildItem -Path $scriptDir -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    ($_.Name -match '\.(tmp|temp)$') -and ($_.FullName -notmatch '[\\/]\.git[\\/]')
})
if ($tempFiles.Count -gt 0) {
    Write-Info "Found $($tempFiles.Count) temp file(s)"
    foreach ($f in $tempFiles) {
        [void]$items.Add([PSCustomObject]@{ Type="Temp file"; Path=$f.FullName; SizeBytes=$f.Length })
    }
} else {
    Write-Info "None found"
}

# ── Summary ──
$itemCount = $items.Count
$totalSize = ($items | Measure-Object -Property SizeBytes -Sum).Sum

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Total: $itemCount item(s)" -ForegroundColor White
if ($totalSize -gt 1MB) {
    Write-Host "  Size:  $([math]::Round($totalSize / 1MB, 1)) MB" -ForegroundColor White
} else {
    Write-Host "  Size:  $([math]::Round($totalSize / 1KB, 1)) KB" -ForegroundColor White
}
Write-Host "============================================" -ForegroundColor Cyan

if ($itemCount -eq 0) {
    Write-Host ""
    Write-Host "  Nothing to clean." -ForegroundColor Green
    Write-Host ""
    if (-not $MyInvocation.ExpectingInput) { pause }
    exit 0
}

# ── Confirm ──
if (-not $skipConfirm) {
    Write-Host ""
    $confirm = Read-Host "Confirm delete all items? (y/N)"
    if ($confirm -notmatch '^[yY]') {
        Write-Host "  Cancelled." -ForegroundColor DarkGray
        Write-Host ""
        if (-not $MyInvocation.ExpectingInput) { pause }
        exit 0
    }
}

# ── Delete ──
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Cleaning..." -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan

$deleted = 0
$failed = 0
$errors = @()

# separate items by category
$fileItems = @($items | Where-Object { $_.Type -ne "__pycache__" -and $_.Type -ne "Model version dir" -and $_.Type -ne "Log dir" })
$dirItems = @($items | Where-Object { $_.Type -eq "__pycache__" -or $_.Type -eq "Model version dir" -or $_.Type -eq "Log dir" })

# delete files first
foreach ($item in $fileItems) {
    $shortPath = $item.Path.Replace($scriptDir, ".").Replace("\", "/")
    if ($dryRun) {
        Write-Del "[Preview] Would delete: $shortPath"
        $deleted++
        continue
    }
    try {
        Remove-Item -Path $item.Path -Force -ErrorAction Stop
        Write-Del "Deleted: $shortPath"
        $deleted++
    } catch {
        Write-Warn "Failed: $shortPath"
        $errors += $shortPath
        $failed++
    }
}

# delete directories (de-duplicated)
$dirPaths = @($dirItems | Select-Object -ExpandProperty Path -Unique)
foreach ($dpath in $dirPaths) {
    if (-not (Test-Path $dpath)) { continue }
    $shortPath = $dpath.Replace($scriptDir, ".").Replace("\", "/")
    if ($dryRun) {
        Write-Del "[Preview] Would delete: $shortPath"
        $deleted++
        continue
    }
    try {
        Remove-Item -Path $dpath -Recurse -Force -ErrorAction Stop
        Write-Del "Deleted: $shortPath"
        $deleted++
    } catch {
        Write-Warn "Failed: $shortPath"
        $errors += $shortPath
        $failed++
    }
}

# ── Done ──
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Cleanup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
if (-not $dryRun) {
    Write-Host "  Deleted: $deleted item(s)" -ForegroundColor White
}
if ($failed -gt 0) {
    Write-Host "  Failed: $failed item(s)" -ForegroundColor Red
    foreach ($e in $errors) {
        Write-Host "    - $e" -ForegroundColor Red
    }
}
Write-Host ""

Write-Host "  Kept (registry + config):" -ForegroundColor DarkGray
Write-Host "    backend/models/.gitkeep" -ForegroundColor DarkGray
Write-Host "    backend/models/registry.json" -ForegroundColor DarkGray
Write-Host "    backend/models/registry.py" -ForegroundColor DarkGray
Write-Host "    backend/models/reg_config.json" -ForegroundColor DarkGray
Write-Host ""

if (-not $MyInvocation.ExpectingInput) {
    Write-Host "Press any key to exit..." -ForegroundColor DarkGray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
