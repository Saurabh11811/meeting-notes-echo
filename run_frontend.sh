#!/bin/bash
# =============================================================================
# Start ECHO Frontend (React/Vite)
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

BACKEND_PORT="${ECHO_BACKEND_PORT:-8765}"
BACKEND_HOST="${ECHO_BACKEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${ECHO_FRONTEND_PORT:-5173}"
BACKEND_URL="http://$BACKEND_HOST:$BACKEND_PORT"

echo "============================================================"
echo "Starting ECHO Frontend"
echo "============================================================"
echo ""

if ! command -v npm >/dev/null 2>&1; then
    echo "npm not found. Install Node.js/npm and try again."
    exit 1
fi

if [ ! -d "ui/node_modules" ]; then
    echo "node_modules not found. Installing frontend dependencies..."
    (cd ui && npm install)
    echo "Frontend dependencies installed."
    echo ""
fi

echo "Checking backend connection..."
if curl -s "$BACKEND_URL/api/health" >/dev/null 2>&1; then
    echo "Backend is running at $BACKEND_URL"
else
    echo "Warning: backend is not responding at $BACKEND_URL"
    echo "Start it with: ./run_backend.sh"
    echo ""
    read -p "Continue with frontend anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "Frontend Configuration"
echo "============================================================"
echo "Frontend URL: http://127.0.0.1:$FRONTEND_PORT"
echo "Backend URL:  $BACKEND_URL"
echo "============================================================"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PROJECT_ROOT/ui"
export VITE_ECHO_API_BASE_URL="$BACKEND_URL/api"
npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"

