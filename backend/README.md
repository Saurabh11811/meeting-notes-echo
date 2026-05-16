# ECHO Backend

Fresh product backend for ECHO. The original Gradio MVP remains untouched at the repository root and should be treated as reference logic while this backend grows into the production API.

## Goals

- Local-first FastAPI service.
- SQLite persistence with lightweight migrations.
- YAML-driven settings with no hardcoded product configuration.
- API contracts that match the generated React UI.
- Future desktop packaging through an Electron shell with the Python backend as a sidecar.

## Development

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
uvicorn echo_api.main:app --reload --port 8765
```

The repo-root `requirements.txt` is the product runtime dependency set. The local `backend/requirements.txt` file delegates to it so backend startup, ASR, browser capture, and LLM provider imports do not drift. Desktop packaging is handled from the repository root through Electron scripts in `package.json`; the backend sidecar entrypoint is `backend/desktop_server.py`.

Run a local environment check from the repository root with:

```bash
python scripts/doctor.py
```

The development database and user config are created under `backend/.data/` by default.

Override paths with:

```bash
export ECHO_APP_DATA_DIR=/path/to/app-data
export ECHO_CONFIG_PATH=/path/to/config.yaml
```

## First APIs

- `GET /api/health`
- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/home`

These are intentionally enough to wire the Home and Settings screens first.
