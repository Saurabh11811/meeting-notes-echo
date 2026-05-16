from __future__ import annotations

import os

import uvicorn

from echo_api.core.config import load_app_config
from echo_api.main import app


def main() -> None:
    config = load_app_config()
    api = config.get("app", {}).get("api", {})
    host = os.getenv("ECHO_BACKEND_HOST", str(api.get("host", "127.0.0.1")))
    port = int(os.getenv("ECHO_BACKEND_PORT", str(api.get("port", 8765))))
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv("ECHO_BACKEND_LOG_LEVEL", "info"),
        access_log=os.getenv("ECHO_BACKEND_ACCESS_LOG", "0") == "1",
    )


if __name__ == "__main__":
    main()
