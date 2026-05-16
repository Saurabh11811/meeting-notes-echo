#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ECHO_VENV_DIR:-$ROOT/.venv}"

echo "ECHO setup"
echo "=========="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN"
  echo "Install Python 3.10+ and rerun this script."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"

if [ "${ECHO_INSTALL_PLAYWRIGHT_CHROMIUM:-0}" = "1" ]; then
  python -m playwright install chromium
else
  echo "Skipping Playwright Chromium download. Set ECHO_INSTALL_PLAYWRIGHT_CHROMIUM=1 to download it."
fi

if command -v npm >/dev/null 2>&1; then
  if [ ! -d "$ROOT/ui/node_modules" ]; then
    (cd "$ROOT/ui" && npm install)
  else
    echo "Frontend dependencies already installed."
  fi
else
  echo "npm not found. Install Node.js/npm before running the frontend."
fi

python "$ROOT/scripts/doctor.py"

echo ""
echo "Run the app:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  ./run_app.sh"
