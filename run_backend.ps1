# =============================================================================
# Start ECHO Backend Server - Windows PowerShell
# =============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$BackendPort = if ($env:ECHO_BACKEND_PORT) { $env:ECHO_BACKEND_PORT } else { "8765" }
$BackendHost = if ($env:ECHO_BACKEND_HOST) { $env:ECHO_BACKEND_HOST } else { "127.0.0.1" }

Write-Host "============================================================"
Write-Host "Starting ECHO Backend"
Write-Host "============================================================"
Write-Host ""

# Corporate-network safe: this repo uses installed Chrome via channel='chrome'.
# Do not allow Playwright to attempt a bundled browser download here.
$env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = "1"

function Test-CommandExists {
    param([string]$Command)
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

if (-not (Test-CommandExists $PythonBin)) {
    throw "Python not found: $PythonBin. Install Python 3.10+ and try again."
}

$PyVersion = & $PythonBin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python: $PyVersion"

& $PythonBin -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10+ is required."
}

if (-not $env:VIRTUAL_ENV -and -not $env:CONDA_DEFAULT_ENV) {
    Write-Host "Warning: no Python virtual environment detected."
    Write-Host "The script will use: $PythonBin"
} else {
    Write-Host "Using active Python environment."
}

Write-Host "Checking backend dependencies..."
$depCheck = @'
import fastapi
import uvicorn
import yaml
import pydantic
import requests
import playwright.sync_api
import multipart
'@
$depCheck | & $PythonBin - 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Missing backend dependencies."
    Write-Host "Install them, then rerun:"
    Write-Host "  $PythonBin -m pip install -r backend\requirements.txt"
    exit 1
}
Write-Host "Backend dependencies OK."

Write-Host "Checking Google Chrome..."
$chromePaths = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe"
)
$Chrome = $chromePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Chrome) {
    throw "Google Chrome was not found. This app uses installed Chrome via Playwright channel='chrome'. Install Chrome or ask IT to deploy it."
}
Write-Host "Chrome found: $Chrome"

Write-Host ""
Write-Host "Initializing backend..."
$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
$bootstrap = @'
from echo_api.services.bootstrap import bootstrap_application
from echo_api.db.connection import get_database_file
bootstrap_application()
print(f"SQLite database: {get_database_file()}")
'@
$bootstrap | & $PythonBin -

Write-Host ""
Write-Host "============================================================"
Write-Host "Backend Configuration"
Write-Host "============================================================"
Write-Host "Backend URL:  http://$BackendHost`:$BackendPort"
Write-Host "API Docs:     http://$BackendHost`:$BackendPort/docs"
Write-Host "Health Check: http://$BackendHost`:$BackendPort/api/health"
Write-Host "App Data:     $ProjectRoot\backend\.data"
Write-Host "============================================================"
Write-Host ""
Write-Host "Press Ctrl+C to stop"
Write-Host ""

Set-Location (Join-Path $ProjectRoot "backend")
& $PythonBin -m uvicorn echo_api.main:app `
    --host $BackendHost `
    --port $BackendPort `
    --reload `
    --log-level info `
    --no-access-log
