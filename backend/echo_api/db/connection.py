from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from echo_api.core.config import load_app_config
from echo_api.core.paths import database_path


def get_database_file() -> Path:
    config = load_app_config()
    filename = str(config.get("database", {}).get("filename", "echo.db"))
    return database_path(filename)


def connect() -> sqlite3.Connection:
    config = load_app_config()
    conn = sqlite3.connect(get_database_file(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    pragmas = config.get("database", {}).get("pragmas", {})
    conn.execute(f"PRAGMA busy_timeout = {int(pragmas.get('busy_timeout_ms', 5000))}")
    if pragmas.get("foreign_keys", True):
        conn.execute("PRAGMA foreign_keys = ON")
    journal_mode = pragmas.get("journal_mode")
    if journal_mode:
        conn.execute(f"PRAGMA journal_mode = {journal_mode}")
    return conn


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}

