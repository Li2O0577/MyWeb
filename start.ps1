# =============================================
#  Indeterminate - Quick Launcher (Flask + Streamlit)
#  Auto-detect Python interpreter, one-click start
# =============================================

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "Indeterminate Launcher"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Indeterminate - Quick Launcher" -ForegroundColor Cyan
Write-Host "  Flask + Streamlit" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ==========================================
# Step 1: Detect available Python interpreters
# ==========================================
Write-Host "[*] Detecting Python interpreters..." -ForegroundColor Yellow
Write-Host ""

$interpreters = @()
$standardPython = $null
$condaCmd = $null
$uvCmd = $null

# --- Standard Python ---
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $v = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $standardPython = $cmd
            break
        }
    } catch {}
}

if ($standardPython) {
    $pyVer = (& $standardPython --version 2>&1).Trim()
    try { $pyPath = (Get-Command $standardPython -ErrorAction Stop).Source }
    catch { $pyPath = $standardPython }
    $interpreters += @{
        Id     = "python"
        Label  = "Standard Python"
        Detail = "$pyVer  |  $pyPath"
    }
}

# --- Conda ---
foreach ($cmd in @("conda", "conda3")) {
    try {
        $v = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $condaCmd = $cmd
            break
        }
    } catch {}
}

if ($condaCmd) {
    $condaVer = (& $condaCmd --version 2>&1).Trim()
    $envLines = & $condaCmd env list 2>&1
    $envNames = @()
    foreach ($line in $envLines) {
        if ($line -match '^\s*(\S+)\s+') {
            $name = $Matches[1]
            if ($name -ne "base" -and $name -ne "#" -and $name -ne "Name") {
                $envNames += $name
            }
        }
    }
    $envListStr = if ($envNames.Count -gt 0) { $envNames -join ", " } else { "(no other envs)" }
    $interpreters += @{
        Id     = "conda"
        Label  = "Conda"
        Detail = "$condaVer  |  envs: $envListStr"
    }
}

# --- uv ---
try {
    $v = & uv --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $uvCmd = "uv"
    }
} catch {}

if ($uvCmd) {
    $uvVer = (& uv --version 2>&1).Trim()
    $interpreters += @{
        Id     = "uv"
        Label  = "uv (Astral)"
        Detail = "$uvVer"
    }
}

# --- No interpreter found ---
if ($interpreters.Count -eq 0) {
    Write-Host "[!] No Python interpreter found!" -ForegroundColor Red
    Write-Host "    Install Python 3.10+ / Conda / uv and add to PATH." -ForegroundColor Yellow
    Write-Host ""
    if (-not $MyInvocation.ExpectingInput) {
        Write-Host "Press any key to exit..." -ForegroundColor DarkGray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
    exit 1
}

# ==========================================
# Step 2: User selects interpreter
# ==========================================
Write-Host "Available interpreters:" -ForegroundColor Cyan
Write-Host ""
for ($i = 0; $i -lt $interpreters.Count; $i++) {
    $num = $i + 1
    Write-Host "  [$num]  $($interpreters[$i].Label)" -ForegroundColor White
    Write-Host "       $($interpreters[$i].Detail)" -ForegroundColor DarkGray
    Write-Host ""
}

if ($interpreters.Count -eq 1) {
    $choice = 1
    Write-Host "Only one interpreter found, auto-selected." -ForegroundColor DarkGray
} else {
    $choice = 0
    while ($choice -lt 1 -or $choice -gt $interpreters.Count) {
        try {
            $input = Read-Host "Choose interpreter (1-$($interpreters.Count))"
            if ($input -match '^\d+$') {
                $choice = [int]$input
            }
        } catch {
            Write-Host ""
            exit 0
        }
    }
}

$selected = $interpreters[$choice - 1]
Write-Host ""
Write-Host "  Selected: $($selected.Label)" -ForegroundColor Green
Write-Host ""

# ==========================================
# Step 3: Input environment name
# ==========================================
$envName = ""
$interpreterId = $selected.Id

if ($interpreterId -eq "conda") {
    $envName = Read-Host "Conda environment name"
    $check = & $condaCmd env list 2>&1 | Select-String -Pattern "^$envName\s"
    if (-not $check) {
        Write-Host "  [!] Environment '$envName' not detected, will try anyway." -ForegroundColor Yellow
    } else {
        Write-Host "  Found: $($check.Line.Trim())" -ForegroundColor Green
    }
}
elseif ($interpreterId -eq "uv") {
    $defaultEnv = ".venv"
    $envName = Read-Host "uv venv directory name (default: $defaultEnv)"
    if ([string]::IsNullOrWhiteSpace($envName)) {
        $envName = $defaultEnv
    }
    $venvPath = Join-Path $scriptDir $envName
    if (-not (Test-Path $venvPath)) {
        Write-Host "  [!] venv directory '$venvPath' not found." -ForegroundColor Yellow
        Write-Host "  Run first: uv venv $envName ; uv pip install -r requirements.txt -r backend/requirements.txt" -ForegroundColor Yellow
    } else {
        Write-Host "  Found: $venvPath" -ForegroundColor Green
    }
}
else {
    # Standard python
    if ($env:VIRTUAL_ENV) {
        Write-Host "  Active venv: $env:VIRTUAL_ENV" -ForegroundColor Magenta
    }
    if ($env:CONDA_DEFAULT_ENV) {
        Write-Host "  Active Conda: $env:CONDA_DEFAULT_ENV" -ForegroundColor Magenta
    }
    $envName = Read-Host "Python path or venv path (leave empty to use system $standardPython)"
}

Write-Host ""

# ==========================================
# Step 4: Build launch commands
# ==========================================
$backendCmd = ""
$frontendCmd = ""

if ($interpreterId -eq "conda") {
    $backendCmd  = "conda run -n `"$envName`" python app.py"
    $frontendCmd = "conda run -n `"$envName`" streamlit run main.py"
}
elseif ($interpreterId -eq "uv") {
    $venvPath = Join-Path $scriptDir $envName
    $pyExe = Join-Path $venvPath "Scripts\python.exe"
    if (Test-Path $pyExe) {
        $backendCmd  = "& '$pyExe' app.py"
        $frontendCmd = "& '$pyExe' -m streamlit run main.py"
    } else {
        $backendCmd  = "uv run python app.py"
        $frontendCmd = "uv run streamlit run main.py"
    }
}
else {
    if ($envName) {
        $backendCmd  = "& '$envName' app.py"
        $frontendCmd = "& '$envName' -m streamlit run main.py"
    } else {
        $backendCmd  = "& '$standardPython' app.py"
        $frontendCmd = "& '$standardPython' -m streamlit run main.py"
    }
}

# ==========================================
# Step 5: Launch services
# ==========================================
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting services..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- Start Flask backend ---
Write-Host "  [1/2] Flask backend (port 5001)..." -ForegroundColor Yellow
$backendDir = Join-Path $scriptDir "backend"

$flaskArgList = @(
    "-NoExit",
    "-Command",
    "`$Host.UI.RawUI.WindowTitle = 'Indeterminate - Flask Backend :5001'; Write-Host '=== Flask Backend (:5001) ===' -ForegroundColor Cyan; Write-Host ''; cd '$backendDir'; $backendCmd"
)

Start-Process powershell -ArgumentList $flaskArgList

# Wait for Flask to initialize
Write-Host "  Waiting for Flask to init (5s)..." -ForegroundColor DarkGray
Start-Sleep -Seconds 5

# --- Start Streamlit frontend ---
Write-Host "  [2/2] Streamlit frontend (port 8501)..." -ForegroundColor Yellow

$streamlitArgList = @(
    "-NoExit",
    "-Command",
    "`$Host.UI.RawUI.WindowTitle = 'Indeterminate - Streamlit Frontend :8501'; Write-Host '=== Streamlit Frontend (:8501) ===' -ForegroundColor Cyan; Write-Host ''; cd '$scriptDir'; `$env:INDETERMINATE_API_BASE = 'http://127.0.0.1:5001/api'; $frontendCmd"
)

Start-Process powershell -ArgumentList $streamlitArgList

# ==========================================
# Done
# ==========================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  All services started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Flask backend:  http://localhost:5001" -ForegroundColor Cyan
Write-Host "  Streamlit:      http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Close both service windows to stop." -ForegroundColor DarkGray
Write-Host ""

if (-not $MyInvocation.ExpectingInput) {
    Write-Host "Press any key to exit..." -ForegroundColor DarkGray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
