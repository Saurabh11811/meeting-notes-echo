#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="$ROOT_DIR/backend/.data"
DB_FILE="$DB_DIR/echo.db"

if [[ "${1:-}" != "--yes" ]]; then
  echo "This will delete the local ECHO development database:"
  echo "  $DB_FILE"
  echo
  echo "Your backend/.data/config.yaml file and generated output files are kept."
  echo "Run again with --yes to continue."
  exit 1
fi

rm -f "$DB_FILE" "$DB_FILE-wal" "$DB_FILE-shm"

PYTHONPATH="$ROOT_DIR/backend" python - <<'PY'
from echo_api.db.connection import get_database_file
from echo_api.services.bootstrap import bootstrap_application

bootstrap_application()
print(f"Recreated SQLite database: {get_database_file()}")
PY
