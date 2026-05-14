#!/bin/bash
# =============================================================================
# Start ECHO Backend Server
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND_PORT="${ECHO_BACKEND_PORT:-8765}"
BACKEND_HOST="${ECHO_BACKEND_HOST:-127.0.0.1}"

echo "============================================================"
echo "Starting ECHO Backend"
echo "============================================================"
echo ""

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python not found: $PYTHON_BIN"
    echo "Install Python 3.10+ and try again."
    exit 1
fi

PY_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

echo "Python: $PY_VERSION"

if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
    :
else
    echo "Python 3.10+ is required."
    exit 1
fi

if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Warning: no Python virtual environment detected."
    echo "The script will use: $PYTHON_BIN"
else
    echo "Using active Python environment."
fi

echo "Checking backend dependencies..."
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import fastapi
import uvicorn
import yaml
import pydantic
import requests
import playwright.sync_api
import multipart
PY
then
    echo "Missing backend dependencies."
    echo "Install them yourself, then rerun:"
    echo "  $PYTHON_BIN -m pip install -r backend/requirements.txt"
    exit 1
fi
echo "Backend dependencies OK."

echo ""
echo "Initializing backend..."
PYTHONPATH="$PROJECT_ROOT/backend" "$PYTHON_BIN" - <<'PY'
from echo_api.services.bootstrap import bootstrap_application
from echo_api.db.connection import get_database_file
bootstrap_application()
print(f"SQLite database: {get_database_file()}")
PY

echo ""
echo "============================================================"
echo "Backend Configuration"
echo "============================================================"
echo "Backend URL:  http://$BACKEND_HOST:$BACKEND_PORT"
echo "API Docs:     http://$BACKEND_HOST:$BACKEND_PORT/docs"
echo "Health Check: http://$BACKEND_HOST:$BACKEND_PORT/api/health"
echo "App Data:     $PROJECT_ROOT/backend/.data"
echo "============================================================"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PROJECT_ROOT/backend"
PYTHONPATH="$PROJECT_ROOT/backend" "$PYTHON_BIN" -m uvicorn echo_api.main:app \
    --host "$BACKEND_HOST" \
    --port "$BACKEND_PORT" \
    --reload \
    --log-level info \
    --no-access-log
