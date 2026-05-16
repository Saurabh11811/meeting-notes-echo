from __future__ import annotations

import subprocess
import sys
from typing import Any

from echo_api.core.dependencies import find_binary, find_browser, install_guidance
from echo_api.core.paths import BACKEND_DIR
from echo_api.core.config import deep_merge, load_app_config, masked_config, write_user_config
from echo_api.services.processing_service import load_prompt

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import helpers_azure  # noqa: E402
import helpers_dify  # noqa: E402
import helpers_local  # noqa: E402


def get_settings(*, reveal_secrets: bool = False) -> dict[str, Any]:
    config = load_app_config()
    response = config if reveal_secrets else masked_config(config)
    response.setdefault("summary", {})["effective_prompt"] = load_prompt(config)
    return response


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    current = load_app_config()
    patch = preserve_masked_secrets(current, patch)
    updated = deep_merge(current, patch)
    write_user_config(updated)
    response = masked_config(updated)
    response.setdefault("summary", {})["effective_prompt"] = load_prompt(updated)
    return response


def preserve_masked_secrets(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for backend_name in ("dify", "azure"):
        incoming = (
            patch.get("backends", {})
            .get(backend_name, {})
            .get("api_key")
        )
        if incoming == "********":
            current_secret = (
                current.get("backends", {})
                .get(backend_name, {})
                .get("api_key", "")
            )
            patch.setdefault("backends", {}).setdefault(backend_name, {})["api_key"] = current_secret
    return patch


def test_backend_connection(backend_kind: str, patch: dict[str, Any] | None = None) -> dict[str, Any]:
    current = load_app_config()
    patch = preserve_masked_secrets(current, patch or {})
    config = deep_merge(current, patch)
    kind = (backend_kind or config.get("summary", {}).get("default_backend", "local")).strip()

    if kind not in ("local", "dify", "azure"):
        return {"backend_kind": kind, "ok": False, "message": "Unknown backend provider."}

    if config.get("privacy", {}).get("local_only_mode") and kind != "local":
        return {
            "backend_kind": kind,
            "ok": False,
            "message": "Local-only mode is enabled. Disable it before testing cloud providers.",
        }

    try:
        if kind == "local":
            local = config.get("backends", {}).get("local", {})
            ok, message = helpers_local.test_local_connection(model=local.get("model", "llama3:latest"))
        elif kind == "dify":
            dify = config.get("backends", {}).get("dify", {})
            ok, message = helpers_dify.test_dify_connection(
                api_key=dify.get("api_key", ""),
                app_base=dify.get("base_url", "https://api.dify.ai/v1"),
            )
        else:
            azure = config.get("backends", {}).get("azure", {})
            ok, message = helpers_azure.test_azure_connection(
                endpoint=azure.get("endpoint", ""),
                api_key=azure.get("api_key", ""),
                api_version=azure.get("api_version", "2024-02-15-preview"),
                deployment=azure.get("deployment", ""),
            )
    except Exception as exc:
        ok, message = False, str(exc)

    return {"backend_kind": kind, "ok": bool(ok), "message": message}


def get_setup_status() -> dict[str, Any]:
    config = load_app_config()
    model = str(config.get("backends", {}).get("local", {}).get("model", "llama3:latest"))
    ollama = find_binary("ollama", env_var="ECHO_OLLAMA_PATH")
    browser = find_browser()
    return {
        "ollama": {
            "installed": bool(ollama),
            "path": ollama,
            "model_name": model,
        },
        "browser": {
            "installed": browser is not None,
            "name": browser[0] if browser else None,
            "path": browser[1] if browser else None,
        },
        "ffmpeg": {
            "installed": bool(find_binary("ffmpeg", env_var="ECHO_FFMPEG_PATH")),
            "path": find_binary("ffmpeg", env_var="ECHO_FFMPEG_PATH"),
        },
        "guidance": install_guidance(),
    }


def pull_ollama_model(model: str | None = None) -> dict[str, Any]:
    config = load_app_config()
    model_name = (model or config.get("backends", {}).get("local", {}).get("model") or "llama3:latest").strip()
    if not model_name:
        return {"ok": False, "message": "No Ollama model configured.", "model": model_name}

    ollama = find_binary("ollama", env_var="ECHO_OLLAMA_PATH")
    if not ollama:
        return {
            "ok": False,
            "message": "Ollama is not installed. Download and launch Ollama first, then pull the model.",
            "model": model_name,
        }

    try:
        proc = subprocess.run(
            [ollama, "pull", model_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": f"Timed out while pulling {model_name}. Ollama may still be downloading it.", "model": model_name}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "model": model_name}

    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return {"ok": False, "message": output or f"ollama pull exited with code {proc.returncode}", "model": model_name}
    return {"ok": True, "message": output or f"{model_name} is ready.", "model": model_name}
