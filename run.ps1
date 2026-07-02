<#
.SYNOPSIS
  Launch the AI Project Workflow Intelligence dashboard, with a preflight that
  verifies everything the demo needs.

.DESCRIPTION
  Idempotent and safe to re-run. It checks / sets up, in order:
    1. Python is available
    2. Ollama is installed (and on PATH)
    3. The Ollama server is running (starts it if not)
    4. The required models are pulled (pulls any that are missing)
    5. A Python virtualenv exists with dependencies installed
    6. A .env file exists (copied from .env.example)
  Then it launches the Streamlit dashboard.

.PARAMETER Check
  Run the preflight checks only and exit (does not launch the dashboard).

.PARAMETER Port
  Port for the dashboard (default 8501).

.EXAMPLE
  .\run.ps1            # preflight + launch
  .\run.ps1 -Check     # just verify the environment
#>
param(
    [switch]$Check,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

$Models = @("qwen2.5:14b", "nomic-embed-text")

function Info($m) { Write-Host "[*]  $m" -ForegroundColor Cyan }
function Good($m) { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[!]  $m" -ForegroundColor Yellow }
function Bad($m)  { Write-Host "[X]  $m" -ForegroundColor Red }

# --------------------------------------------------------------------------- #
# 1. Python
# --------------------------------------------------------------------------- #
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { $python = (Get-Command python3 -ErrorAction SilentlyContinue) }
if (-not $python) {
    Bad "Python not found. Install Python 3.10+ from https://python.org and re-run."
    exit 1
}
Good "Python: $((& $python.Source --version) 2>&1)"

# --------------------------------------------------------------------------- #
# 2. Ollama installed (+ on PATH)
# --------------------------------------------------------------------------- #
function Ensure-OllamaOnPath {
    if (Get-Command ollama -ErrorAction SilentlyContinue) { return $true }
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    if (Test-Path (Join-Path $candidate "ollama.exe")) {
        $env:Path += ";$candidate"
        return $true
    }
    return $false
}

if (-not (Ensure-OllamaOnPath)) {
    Bad "Ollama is not installed."
    Warn "Install it with:  winget install --id Ollama.Ollama -e"
    Warn "or download from: https://ollama.com/download"
    exit 1
}
Good "Ollama: $((& ollama --version) 2>&1)"

# --------------------------------------------------------------------------- #
# 3. Ollama server running
# --------------------------------------------------------------------------- #
function Test-OllamaUp {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
        return $true
    } catch { return $false }
}

if (Test-OllamaUp) {
    Good "Ollama server is running"
} else {
    Info "Starting the Ollama server…"
    $appExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $appExe) { Start-Process $appExe } else { Start-Process "ollama" -ArgumentList "serve" }
    $deadline = (Get-Date).AddSeconds(30)
    while (-not (Test-OllamaUp) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 800 }
    if (Test-OllamaUp) { Good "Ollama server started" } else { Bad "Ollama server did not come up in time"; exit 1 }
}

# --------------------------------------------------------------------------- #
# 4. Required models
# --------------------------------------------------------------------------- #
$installed = (& ollama list) 2>&1 | Out-String
foreach ($m in $Models) {
    if ($installed -match [regex]::Escape($m)) {
        Good "model present: $m"
    } else {
        Info "pulling model $m  (first time only; qwen2.5:14b is ~9GB)…"
        & ollama pull $m
        if ($LASTEXITCODE -ne 0) { Bad "failed to pull $m"; exit 1 }
        Good "pulled $m"
    }
}

# --------------------------------------------------------------------------- #
# 5. Python virtualenv + dependencies
# --------------------------------------------------------------------------- #
$venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Info "creating virtualenv (.venv)…"
    & $python.Source -m venv .venv
}
$venvStreamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $venvStreamlit)) {
    Info "installing dependencies (this can take a few minutes)…"
    & $venvPy -m pip install --upgrade pip --quiet
    & $venvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Bad "dependency install failed"; exit 1 }
    Good "dependencies installed"
} else {
    Good "dependencies present (.venv)"
}

# --------------------------------------------------------------------------- #
# 6. .env
# --------------------------------------------------------------------------- #
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Good "created .env from .env.example (defaults to local Ollama)"
} else {
    Good ".env present"
}

Write-Host ""
Good "Preflight complete."

if ($Check) {
    Info "-Check specified; not launching the dashboard."
    exit 0
}

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
Write-Host ""
Info "Launching dashboard at http://localhost:$Port  (Ctrl+C to stop)"
& $venvPy -m streamlit run dashboard/app.py --server.port $Port
