from __future__ import annotations

import subprocess
import sys
import platform
from pathlib import Path
from fastapi import APIRouter

from echo_api import __version__
from echo_api.db.connection import get_database_file
from echo_api.core.config import load_app_config
from echo_api.core.dependencies import find_binary, find_browser, install_guidance

router = APIRouter(tags=["health"])

def playwright_ok() -> bool:
    try:
        import playwright.sync_api  # noqa
        return True
    except Exception:
        return False


def ollama_has_model(tag: str) -> bool:
    ollama = find_binary("ollama", env_var="ECHO_OLLAMA_PATH")
    if not ollama:
        return False
    try:
        # Run ollama list and check for the model tag
        out = subprocess.check_output([ollama, "list"], text=True, stderr=subprocess.STDOUT, timeout=15)
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
    ffmpeg_path = find_binary("ffmpeg", env_var="ECHO_FFMPEG_PATH")
    ffprobe_path = find_binary("ffprobe", env_var="ECHO_FFPROBE_PATH")
    pw_ok = playwright_ok()
    browser = find_browser()
    ollama_path = find_binary("ollama", env_var="ECHO_OLLAMA_PATH")
    oll_installed = ollama_path is not None
    
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
            "ffprobe": {
                "installed": ffprobe_path is not None,
                "path": ffprobe_path,
            },
            "playwright": {
                "installed": pw_ok,
            },
            "browser": {
                "installed": browser is not None,
                "name": browser[0] if browser else None,
                "path": browser[1] if browser else None,
            },
            "ollama": {
                "installed": oll_installed,
                "path": ollama_path,
                "model_present": model_ok,
                "model_name": local_model,
            },
            "asr": {
                "faster_whisper": module_available("faster_whisper"),
                "transformers": module_available("transformers"),
                "torch": module_available("torch"),
            },
            "storage": {
                "writable": write_ok,
                "path": write_detail,
            }
        },
        "install_guidance": install_guidance(),
    }


def module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False
