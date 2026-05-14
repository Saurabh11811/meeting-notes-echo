from __future__ import annotations

from echo_api.core.config import ensure_user_config, load_app_config
from echo_api.core.paths import storage_path
from echo_api.db.connection import db_session
from echo_api.db.schema import initialize_database
from echo_api.db.seed import seed_reference_data


def bootstrap_application() -> None:
    ensure_user_config()
    config = load_app_config()
    storage = config.get("storage", {})
    for key in ("output_dir", "uploads_dir", "exports_dir"):
        if storage.get(key):
            storage_path(str(storage[key]))
    with db_session() as conn:
        initialize_database(conn)
        seed_reference_data(conn)

