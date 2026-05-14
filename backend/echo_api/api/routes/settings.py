from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from echo_api.services.settings_service import get_settings, update_settings

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsPatch(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


@router.get("")
def read_settings() -> dict[str, Any]:
    return get_settings(reveal_secrets=False)


@router.put("")
def write_settings(payload: SettingsPatch) -> dict[str, Any]:
    return update_settings(payload.values)

