from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import yaml

from echo_api.core.paths import DEFAULT_CONFIG_PATH, app_data_dir, user_config_path


Config = dict[str, Any]


def deep_merge(base: Config, override: Config) -> Config:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: Path) -> Config:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_user_config() -> Path:
    app_data_dir().mkdir(parents=True, exist_ok=True)
    target = user_config_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_CONFIG_PATH, target)
    return target


def load_app_config() -> Config:
    default_config = read_yaml(DEFAULT_CONFIG_PATH)
    user_config = read_yaml(ensure_user_config())
    return deep_merge(default_config, user_config)


def write_user_config(config: Config) -> None:
    target = user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def masked_config(config: Config) -> Config:
    masked = copy.deepcopy(config)
    for backend in masked.get("backends", {}).values():
        if isinstance(backend, dict) and backend.get("api_key"):
            backend["api_key"] = "********"
    return masked

