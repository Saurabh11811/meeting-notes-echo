from __future__ import annotations

from fastapi import APIRouter

from echo_api.services.home_service import get_home_summary

router = APIRouter(prefix="/home", tags=["home"])


@router.get("")
def read_home() -> dict:
    return get_home_summary()

