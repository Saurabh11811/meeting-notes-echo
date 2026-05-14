#!/bin/bash
# =============================================================================
# Start ECHO Backend and Frontend Together
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

BACKEND_PORT="${ECHO_BACKEND_PORT:-8765}"
BACKEND_HOST="${ECHO_BACKEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${ECHO_FRONTEND_PORT:-5173}"
SESSION_NAME="echo-dev"
LOG_DIR="$PROJECT_ROOT/logs"

echo "============================================================"
echo "Starting ECHO Full Stack"
echo "============================================================"
echo ""

mkdir -p "$LOG_DIR"

cleanup() {
    echo ""
    echo "Stopping ECHO services..."
    if command -v tmux >/dev/null 2>&1; then
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    else
        if [[ -n "${BACKEND_PID:-}" ]]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
        if [[ -n "${FRONTEND_PID:-}" ]]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
    fi
    echo "Stopped."
}

trap cleanup INT TERM

if command -v tmux >/dev/null 2>&1; then
    echo "Using tmux session '$SESSION_NAME'."
    echo "Detach:   Ctrl+B then D"
    echo "Reattach: tmux attach -t $SESSION_NAME"
    echo ""

    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && ./run_backend.sh"
    tmux split-window -h -t "$SESSION_NAME" "cd '$PROJECT_ROOT' && sleep 4 && ./run_frontend.sh"
    tmux attach -t "$SESSION_NAME"
else
    echo "tmux not found. Running services in background with log files."
    echo "Install tmux for split-screen development."
    echo ""

    ./run_backend.sh > "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!

    echo "Waiting for backend..."
    for _ in {1..30}; do
        if curl -s "http://$BACKEND_HOST:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    ./run_frontend.sh > "$LOG_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!

    echo "============================================================"
    echo "ECHO is starting"
    echo "============================================================"
    echo "Backend:  http://$BACKEND_HOST:$BACKEND_PORT"
    echo "API Docs: http://$BACKEND_HOST:$BACKEND_PORT/docs"
    echo "Frontend: http://127.0.0.1:$FRONTEND_PORT"
    echo ""
    echo "Logs:"
    echo "Backend:  tail -f logs/backend.log"
    echo "Frontend: tail -f logs/frontend.log"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo "============================================================"

    wait
fi

