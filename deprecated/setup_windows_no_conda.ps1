param(
    [switch]$SkipFfmpegInstall,
    [switch]$UseVenv
)

$ErrorActionPreference = "Stop"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command($Name, $InstallHint) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "$Name was not found. $InstallHint"
    }
    return $cmd.Source
}

Set-Location $PSScriptRoot

Write-Step "Checking Python"
$python = Require-Command "python" "Install Python 3.10+ from https://www.python.org/downloads/ and select 'Add python.exe to PATH'."
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "Python: $python"
Write-Host "Version: $pythonVersion"

if ($UseVenv) {
    Write-Step "Creating/using .venv"
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    . .\.venv\Scripts\Activate.ps1
    Write-Host "Using venv Python: $((Get-Command python).Source)"
}

Write-Step "Upgrading pip"
python -m pip install --upgrade pip

Write-Step "Installing Python dependencies"
python -m pip install -r requirements.txt

Write-Step "Installing Playwright Chromium"
python -m playwright install chromium

Write-Step "Checking ffmpeg"
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue

if ((-not $ffmpeg -or -not $ffprobe) -and -not $SkipFfmpegInstall) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "ffmpeg was not found and winget is unavailable. Install ffmpeg manually, then reopen PowerShell and run: where ffmpeg"
    }

    Write-Step "Installing ffmpeg with winget"
    winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements

    Write-Host ""
    Write-Host "If ffmpeg is still not found below, close this PowerShell window and open a new one so PATH refreshes." -ForegroundColor Yellow
}

Write-Step "Verifying tools from this Python process"
python -c "import shutil, sys; print('python =', sys.executable); print('ffmpeg =', shutil.which('ffmpeg')); print('ffprobe =', shutil.which('ffprobe'))"
python -c "import playwright.sync_api; print('playwright import = OK')"

Write-Step "Setup check complete"
Write-Host "Run the app with:"
Write-Host "python echo.py" -ForegroundColor Green
