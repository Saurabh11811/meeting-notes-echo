from __future__ import annotations

from typing import Any

from echo_api.core.config import deep_merge, load_app_config, masked_config, write_user_config
from echo_api.services.processing_service import load_prompt


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
