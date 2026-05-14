import uvicorn

from echo_api.core.config import load_app_config


def main() -> None:
    config = load_app_config()
    api = config.get("app", {}).get("api", {})
    uvicorn.run(
        "echo_api.main:app",
        host=str(api.get("host", "127.0.0.1")),
        port=int(api.get("port", 8765)),
        reload=False,
    )


if __name__ == "__main__":
    main()

