from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from echo_api.services.settings_service import get_settings, get_setup_status, pull_ollama_model, test_backend_connection, update_settings

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsPatch(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class BackendTestRequest(BaseModel):
    backend_kind: str = ""
    values: dict[str, Any] = Field(default_factory=dict)


class OllamaPullRequest(BaseModel):
    model: str | None = None


@router.get("")
def read_settings() -> dict[str, Any]:
    return get_settings(reveal_secrets=False)


@router.put("")
def write_settings(payload: SettingsPatch) -> dict[str, Any]:
    return update_settings(payload.values)


@router.post("/test-backend")
def test_backend(payload: BackendTestRequest) -> dict[str, Any]:
    return test_backend_connection(payload.backend_kind, payload.values)


@router.get("/setup")
def read_setup_status() -> dict[str, Any]:
    return get_setup_status()


@router.post("/ollama/pull-model")
def pull_model(payload: OllamaPullRequest) -> dict[str, Any]:
    return pull_ollama_model(payload.model)
