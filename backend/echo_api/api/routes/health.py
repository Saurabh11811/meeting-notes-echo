from __future__ import annotations

from fastapi import APIRouter

from echo_api import __version__
from echo_api.core.paths import app_data_dir
from echo_api.db.connection import get_database_file

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    db_file = get_database_file()
    return {
        "status": "ok",
        "service": "echo-api",
        "version": __version__,
        "app_data_dir": str(app_data_dir()),
        "database": str(db_file),
        "database_exists": db_file.exists(),
    }

