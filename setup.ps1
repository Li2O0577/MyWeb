# =============================================
#  Indeterminate — 一键环境配置 (Flask + Streamlit)
#  自动检测 GPU 并安装对应 PyTorch 版本
# =============================================
#  如果无法运行 .ps1，请在 PowerShell 中执行：
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#  或者直接双击运行 setup.bat
# =============================================

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Indeterminate Setup"

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║ Indeterminate 数据分析平台 — 环境配置   ║" -ForegroundColor Cyan
Write-Host "║    Flask + Streamlit 前后端分离版        ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. 检查 Python ──
Write-Host "[1/5] 检查 Python 环境..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $v = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            break
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host "❌ 未找到 Python！请先安装 Python 3.10+ 并添加到 PATH。" -ForegroundColor Red
    Write-Host "   如果使用 Conda，请先执行: conda activate <环境名>" -ForegroundColor Yellow
    Write-Host "   下载地址: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "🐍 当前 Conda 环境: $env:CONDA_DEFAULT_ENV" -ForegroundColor Magenta
}
if ($env:VIRTUAL_ENV) {
    Write-Host "🐍 当前 venv: $env:VIRTUAL_ENV" -ForegroundColor Magenta
}

$pyVer = & $pythonCmd --version 2>&1
Write-Host "✅ 找到 $pyVer" -ForegroundColor Green

# ── 2. 安装前端依赖 (Streamlit) ──
Write-Host ""
Write-Host "[2/5] 安装前端依赖 (Streamlit)..." -ForegroundColor Yellow
Write-Host "    streamlit, pandas, numpy, scikit-learn, plotly, requests, openpyxl"

$frontendPkgs = @("streamlit", "pandas", "numpy", "scikit-learn", "plotly", "requests", "openpyxl")
foreach ($pkg in $frontendPkgs) {
    Write-Host "    → $pkg" -NoNewline
    & $pythonCmd -m pip install -q $pkg 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host " ✅" -ForegroundColor Green } else { Write-Host " ❌" -ForegroundColor Red }
}

# ── 3. 安装后端依赖 (Flask) ──
Write-Host ""
Write-Host "[3/5] 安装后端依赖 (Flask)..." -ForegroundColor Yellow
Write-Host "    flask, flask-cors, torch, requests, pyarrow"

$backendPkgs = @("flask", "flask-cors", "requests", "pyarrow")
foreach ($pkg in $backendPkgs) {
    Write-Host "    → $pkg" -NoNewline
    & $pythonCmd -m pip install -q $pkg 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host " ✅" -ForegroundColor Green } else { Write-Host " ❌" -ForegroundColor Red }
}

# ── 4. 检测 GPU 并安装 PyTorch ──
Write-Host ""
Write-Host "[4/5] 检测 GPU 并安装 PyTorch..." -ForegroundColor Yellow

$hasNvidia = $false
try {
    $nvidiaCheck = nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0 -and $nvidiaCheck) { $hasNvidia = $true }
} catch {}

if ($hasNvidia) {
    $gpuName = ($nvidiaCheck -split ",")[0].Trim()
    $cc = ($nvidiaCheck -split ",")[-1].Trim()
    $ccParts = $cc -split "\."
    $ccMajor = [int]$ccParts[0]
    $ccMinor = [int]$ccParts[1]

    Write-Host "   🎮 检测到 GPU: $gpuName" -ForegroundColor Cyan
    Write-Host "   计算能力: $cc ($ccMajor.$ccMinor)" -ForegroundColor Cyan

    if ($ccMajor -ge 12) {
        Write-Host "   架构: Blackwell → 使用 CUDA 12.8" -ForegroundColor Green
        & $pythonCmd -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    }
    elseif ($ccMajor -eq 8 -or $ccMajor -eq 9) {
        Write-Host "   架构: Ampere / Ada Lovelace → 使用 CUDA 12.1" -ForegroundColor Green
        & $pythonCmd -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
    else {
        Write-Host "   架构: 较旧型号 → 使用 CUDA 11.8" -ForegroundColor Green
        & $pythonCmd -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
}
else {
    Write-Host "   💻 未检测到 NVIDIA GPU → 安装 CPU 版本 PyTorch" -ForegroundColor Yellow
    & $pythonCmd -m pip install -q torch torchvision torchaudio
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ PyTorch 安装失败！请手动安装。" -ForegroundColor Red
    exit 1
}
Write-Host "✅ PyTorch 安装完成" -ForegroundColor Green

# ── 5. 验证安装 ──
Write-Host ""
Write-Host "[5/5] 验证安装..." -ForegroundColor Yellow

$checks = @(
    @{Module="streamlit"; Desc="Streamlit"},
    @{Module="flask"; Desc="Flask"},
    @{Module="flask_cors"; Desc="Flask-CORS"},
    @{Module="pandas"; Desc="pandas"},
    @{Module="numpy"; Desc="numpy"},
    @{Module="sklearn"; Desc="scikit-learn"},
    @{Module="plotly"; Desc="plotly"},
    @{Module="requests"; Desc="requests"},
    @{Module="openpyxl"; Desc="openpyxl"},
    @{Module="pyarrow"; Desc="PyArrow"},
    @{Module="torch"; Desc="PyTorch"}
)

foreach ($c in $checks) {
    $code = "import $($c.Module); print('ok')"
    $result = & $pythonCmd -c $code 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ $($c.Desc)" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $($c.Desc) — 安装可能失败" -ForegroundColor Red
    }
}

$cudaCheck = & $pythonCmd -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaCheck -eq "True") {
    $cudaVer = & $pythonCmd -c "import torch; print(torch.version.cuda)" 2>&1
    Write-Host ""
    Write-Host "🎉 CUDA 可用！版本: $cudaVer" -ForegroundColor Green
}

# ── 完成 ──
Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║         🎉 环境配置完成！                ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  ⚡ 启动后端 (Flask)： " -ForegroundColor Cyan -NoNewline
Write-Host "cd backend && python app.py" -ForegroundColor White
Write-Host "  🎨 启动前端 (Streamlit)： " -ForegroundColor Cyan -NoNewline
Write-Host "cd .. && streamlit run main.py" -ForegroundColor White
Write-Host ""
Write-Host "  Flask 后端: http://localhost:5001" -ForegroundColor Gray
Write-Host "  Streamlit 前端: http://localhost:8501" -ForegroundColor Gray
Write-Host ""

if (-not $MyInvocation.ExpectingInput) {
    Write-Host "按任意键退出..." -ForegroundColor DarkGray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
