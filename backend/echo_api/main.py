from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from echo_api import __version__
from echo_api.api.router import api_router
from echo_api.core.config import load_app_config
from echo_api.services.bootstrap import bootstrap_application


def create_app() -> FastAPI:
    config = load_app_config()
    app_config = config.get("app", {})
    api_config = app_config.get("api", {})

    app = FastAPI(
        title=f"{app_config.get('product_name', 'ECHO')} API",
        version=__version__,
        description=app_config.get("product_subtitle", "Executive Calls, Highlights & Outcomes"),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(api_config.get("cors_origins", [])),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def on_startup() -> None:
        bootstrap_application()

    app.include_router(api_router)
    return app


app = create_app()

