from __future__ import annotations

import os
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    BACKEND_DIR = Path(getattr(sys, "_MEIPASS")) / "backend"
else:
    BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_CONFIG_PATH = BACKEND_DIR / "config" / "default.yaml"


def app_data_dir() -> Path:
    configured = os.getenv("ECHO_APP_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (BACKEND_DIR / ".data").resolve()


def user_config_path() -> Path:
    configured = os.getenv("ECHO_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return app_data_dir() / "config.yaml"


def database_path(filename: str) -> Path:
    db_dir = app_data_dir()
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / filename


def storage_path(name: str) -> Path:
    path = app_data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
