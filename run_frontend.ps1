# =============================================================================
# Start ECHO Frontend - Windows PowerShell
# =============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$BackendPort = if ($env:ECHO_BACKEND_PORT) { $env:ECHO_BACKEND_PORT } else { "8765" }
$BackendHost = if ($env:ECHO_BACKEND_HOST) { $env:ECHO_BACKEND_HOST } else { "127.0.0.1" }
$FrontendPort = if ($env:ECHO_FRONTEND_PORT) { $env:ECHO_FRONTEND_PORT } else { "5173" }
$BackendUrl = "http://$BackendHost`:$BackendPort"

Write-Host "============================================================"
Write-Host "Starting ECHO Frontend"
Write-Host "============================================================"
Write-Host ""

function Test-CommandExists {
    param([string]$Command)
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

if (-not (Test-CommandExists "npm")) {
    throw "npm not found. Install Node.js/npm and try again."
}

$NodeModules = Join-Path $ProjectRoot "ui\node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "node_modules not found. Installing frontend dependencies..."
    Push-Location (Join-Path $ProjectRoot "ui")
    try {
        npm install
    } finally {
        Pop-Location
    }
    Write-Host "Frontend dependencies installed."
    Write-Host ""
}

Write-Host "Checking backend connection..."
$backendOk = $false
try {
    Invoke-WebRequest -Uri "$BackendUrl/api/health" -UseBasicParsing -TimeoutSec 3 | Out-Null
    $backendOk = $true
} catch {
    $backendOk = $false
}

if ($backendOk) {
    Write-Host "Backend is running at $BackendUrl"
} else {
    Write-Host "Warning: backend is not responding at $BackendUrl"
    Write-Host "Start it with: .\run_backend.ps1"
    Write-Host ""
    $reply = Read-Host "Continue with frontend anyway? (y/N)"
    if ($reply -notmatch '^[Yy]$') {
        exit 1
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Frontend Configuration"
Write-Host "============================================================"
Write-Host "Frontend URL: http://127.0.0.1:$FrontendPort"
Write-Host "Backend URL:  $BackendUrl"
Write-Host "============================================================"
Write-Host ""
Write-Host "Press Ctrl+C to stop"
Write-Host ""

Set-Location (Join-Path $ProjectRoot "ui")
$env:VITE_ECHO_API_BASE_URL = "$BackendUrl/api"
npm run dev -- --host 127.0.0.1 --port $FrontendPort
