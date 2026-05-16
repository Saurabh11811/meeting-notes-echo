$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = if ($env:ECHO_VENV_DIR) { $env:ECHO_VENV_DIR } else { Join-Path $Root ".venv" }

Write-Host "ECHO setup"
Write-Host "=========="

if (-not (Get-Command $PythonBin -ErrorAction SilentlyContinue)) {
    throw "Python not found: $PythonBin. Install Python 3.10+ and rerun this script."
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating Python virtual environment at $VenvDir"
    & $PythonBin -m venv $VenvDir
}

$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
. $Activate

python -m pip install --upgrade pip
python -m pip install -r (Join-Path $Root "requirements.txt")

if ($env:ECHO_INSTALL_PLAYWRIGHT_CHROMIUM -eq "1") {
    python -m playwright install chromium
} else {
    Write-Host "Skipping Playwright Chromium download. Set ECHO_INSTALL_PLAYWRIGHT_CHROMIUM=1 to download it."
}

if (Get-Command npm -ErrorAction SilentlyContinue) {
    $NodeModules = Join-Path $Root "ui\node_modules"
    if (-not (Test-Path $NodeModules)) {
        Push-Location (Join-Path $Root "ui")
        try {
            npm install
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "Frontend dependencies already installed."
    }
} else {
    Write-Host "npm not found. Install Node.js/npm before running the frontend."
}

python (Join-Path $Root "scripts\doctor.py")

Write-Host ""
Write-Host "Run the app:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  .\run_backend.ps1"
Write-Host "  .\run_frontend.ps1"
