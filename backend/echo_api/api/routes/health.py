from __future__ import annotations

import os
import shutil
import subprocess
import sys
import platform
from pathlib import Path
from fastapi import APIRouter

from echo_api import __version__
from echo_api.core.paths import app_data_dir
from echo_api.db.connection import get_database_file
from echo_api.core.config import load_app_config

router = APIRouter(tags=["health"])


def find_binary(name: str) -> str | None:
    return shutil.which(name)


def playwright_ok() -> bool:
    try:
        import playwright.sync_api  # noqa
        return True
    except Exception:
        return False


def ollama_has_model(tag: str) -> bool:
    if not shutil.which("ollama"):
        return False
    try:
        # Run ollama list and check for the model tag
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if line.strip().startswith(tag):
                return True
        return False
    except Exception:
        return False


def output_writable(path_str: str) -> tuple[bool, str]:
    try:
        p = Path(path_str).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True, str(p.resolve())
    except Exception as e:
        return False, str(e)


@router.get("/health")
def health_check() -> dict:
    config = load_app_config()
    db_file = get_database_file()
    
    # Dependencies
    ffmpeg_path = find_binary("ffmpeg")
    pw_ok = playwright_ok()
    oll_installed = find_binary("ollama") is not None
    
    local_model = config.get("backends", {}).get("local", {}).get("model", "llama3:latest")
    model_ok = ollama_has_model(local_model) if oll_installed else False
    
    # Storage
    output_dir = config.get("storage", {}).get("output_dir", "out")
    write_ok, write_detail = output_writable(output_dir)

    return {
        "status": "ok",
        "service": "echo-api",
        "version": __version__,
        "os": f"{platform.system()} {platform.release()}",
        "python": {
            "version": platform.python_version(),
            "ok": sys.version_info >= (3, 10),
        },
        "database": {
            "path": str(db_file),
            "exists": db_file.exists(),
        },
        "dependencies": {
            "ffmpeg": {
                "installed": ffmpeg_path is not None,
                "path": ffmpeg_path,
            },
            "playwright": {
                "installed": pw_ok,
            },
            "ollama": {
                "installed": oll_installed,
                "model_present": model_ok,
                "model_name": local_model,
            },
            "storage": {
                "writable": write_ok,
                "path": write_detail,
            }
        }
    }
