# Project ECHO – Meeting Notes, Your Way

> **Cross‑platform • Customizable • Controllable meeting summaries**

Project **ECHO** (Executive Calls, Highlights & Outcomes) turns meeting recordings into **ready‑to‑share minutes**.  
Unlike black‑box tools, ECHO is built to be **auditable**, **backend‑agnostic**, and **privacy‑aware**.

---

## ✨ Why ECHO
- **Cross‑platform**: Works with Teams/Stream (URL) *and* raw media files (MP4/MP3/etc.).
- **Configurable**: Prompts live in files; YAML config is explicit and versionable.
- **Backend‑flexible**: Local (Ollama), Azure OpenAI, Dify (native App API), or your choice.
- **Privacy‑first**: On‑device summarization path (Ollama) with no cloud dependency.
- **Portable outputs**: `*_transcript.txt`, `*_summary.txt`, and `.eml` email drafts.

---

## 🧱 Architecture (MVP)
```
Recording (File OR Teams/Stream URL)
        │
        ▼
 Transcript Capture
    ├── Use built-in transcript (if visible)
    └── ASR (Whisper: faster-whisper or HF pipeline)
        │
        ▼
 Summarization (Either of Backend)
    ├── Local (Ollama)
    ├── Azure OpenAI
    └── Dify (native App API)
        │
        ▼
      Outputs
    ├── <name>_transcript.txt
    ├── <name>_summary.txt
    └── <name>.eml
```

---

## 📦 Repo Layout
```
.
├─ echo.py                # One-page Gradio UI (file OR URL)
├─ chat_helper.py         # Teams/Stream transcript capture (browser)
├─ calls_helper.py        # Local ASR (Whisper) for media files
├─ helpers_local.py       # Summaries via Ollama (local)
├─ helpers_azure.py       # Summaries via Azure OpenAI
├─ helpers_dify.py        # Summaries via Dify (native App API)
├─ prompts.py             # Centralized prompt loader (file/inline/default)
├─ config.yaml            # Auto-created on first run; editable
└─ out/                   # Outputs (transcripts, summaries, .eml)
```

---

## 🚀 Quick Start

### Product UI + New Backend

The new product architecture runs a FastAPI backend and the generated React/Vite UI.

First-time setup:

```bash
# macOS / Linux
./install.sh

# Windows PowerShell
.\install.ps1
```

Run diagnostics at any time:

```bash
python scripts/doctor.py
```

Desktop shell development:

```bash
# Terminal 1: frontend dev server
./run_frontend.sh

# Terminal 2: Electron shell
npm run desktop:dev
```

`desktop:dev` starts the FastAPI backend automatically unless one is already running at `http://127.0.0.1:8765/api/health`. Set `ECHO_SKIP_BACKEND_SIDECAR=1` if you want to manage the backend yourself.

Build the Python backend sidecar for the current operating system:

```bash
python -m pip install -r requirements-packaging.txt
npm run backend:build-sidecar
```

Create an unpacked desktop app for local packaging checks:

```bash
npm install
npm run desktop:pack
```

Create native installer artifacts for the current operating system:

```bash
npm run desktop:dist
```

Platform-specific builds:

```bash
npm run desktop:dist:mac     # DMG + zip on macOS
npm run desktop:dist:win     # NSIS .exe + .msi + portable exe on Windows
npm run desktop:dist:linux   # AppImage + .deb on Linux
```

Native installers should be built on the target OS. The GitHub Actions workflow at `.github/workflows/desktop-release.yml` builds macOS, Windows, and Linux artifacts on native runners and uploads them as workflow artifacts. Use `workflow_dispatch` to run it manually; set `package_asr=true` only when you want to experiment with a much larger build that bundles local recording transcription libraries.

The Electron shell loads the React UI in a native desktop window and starts the FastAPI backend as a sidecar. In development it launches `python -m echo_api`; in packaged builds it looks for a PyInstaller-built `echo-api` binary under bundled resources.

Packaged local builds use ad-hoc macOS signing by default (`identity: "-"`) so developers can test a DMG without an Apple Developer certificate. Consumer macOS releases should use Developer ID signing and notarization. Set `ECHO_MAC_SIGN=1` in a properly configured signing environment before running the macOS release build.

By default, the sidecar is a lightweight backend package that supports settings, transcript-first flows, provider checks, and local Ollama summarization. It intentionally excludes `torch`, `faster-whisper`, and `transformers` to keep first installers smaller and more reliable. For local audio/video ASR inside the packaged app, build with `ECHO_PACKAGE_ASR=1` after validating size, startup time, and platform-specific native library behavior.

```bash
# Backend only
./run_backend.sh

# Frontend only
./run_frontend.sh

# Backend + frontend together
./run_app.sh
```

Default URLs:

- Backend: `http://127.0.0.1:8765`
- API docs: `http://127.0.0.1:8765/docs`
- Health check: `http://127.0.0.1:8765/api/health`
- Frontend: `http://127.0.0.1:5173`

The backend script uses your current Python environment and does not create a virtual environment. If dependencies are missing, install the product runtime dependencies with `python -m pip install -r requirements.txt`. The `backend/requirements.txt` file intentionally delegates to the repo-root runtime file so ASR, browser capture, and provider modules stay in one dependency set. Frontend dependencies are installed by `run_frontend.sh` if `ui/node_modules` is missing. Backend development data is stored under `backend/.data/`.

### Product Backend Data

The product backend uses local SQLite by default. The database is generated automatically on backend startup by `backend/echo_api/services/bootstrap.py`, using migrations in `backend/echo_api/db/schema.py` and reference-data setup in `backend/echo_api/db/seed.py`.

- SQLite file: `backend/.data/echo.db`
- User config: `backend/.data/config.yaml`
- Generated transcripts/summaries/email drafts: `backend/.data/out/`
- Uploaded media: `backend/.data/uploads/`

To reset only the development database and recreate an empty schema:

```bash
./reset_dev_db.sh --yes
```

This keeps `backend/.data/config.yaml` and generated output files. To reset manually, stop the backend, delete `backend/.data/echo.db`, `backend/.data/echo.db-wal`, and `backend/.data/echo.db-shm`, then start `./run_backend.sh` again.

### Legacy Gradio MVP

### 1) Requirements
- **Python 3.12+** (3.10+ works; tested on macOS)
- `ffmpeg` installed and on PATH
- Optional: **Ollama** for local LLM
- Optional: Playwright (Chromium) for URL capture

### 2) Install deps
```bash
pip install -r requirements.txt
python -m playwright install chromium
```

> If you don’t use URLs (only local files), you can skip Playwright.

### 3) Run the UI
```bash
python echo.py
```
Then open the browser page it prints (usually `http://127.0.0.1:7860`).

---

## 🧭 Using the App
- **Inputs**: Upload a file **OR** paste a Teams/Stream URL (strict OR).
- **Setup Panel (left)**:
  - Auto environment checks (ffmpeg, Playwright, MPS, Ollama).
  - Choose **summary backend**: Local / Dify / Azure.
  - Enter backend credentials (Azure endpoint, Dify key, etc.).
  - **Prompt**: Provide a **file path** _or_ inline text (falls back to default).
  - Output directory & save toggles (transcript / summary / `.eml`).
  - **Test All Connections** button verifies Local/Dify/Azure in one click.
- **Outputs**: Full **Summary**, **Transcript**, and a **Status** panel with timings & saved paths.

---

## ⚙️ Configuration (`config.yaml`)
Created automatically on first run. Example:
```yaml
summary:
  default_backend: local      # local | dify | azure
  model: llama3:latest        # legacy; used by local if not set under backends.local
  max_chars: 0                # 0 = no trim
  prompt_file: ""             # optional path to a .txt prompt
  prompt_text: ""             # inline fallback if file missing/empty

backends:
  local:
    enabled: true
    name: Local Ollama
    base_url: http://localhost:11434
    model: llama3:latest
  dify:
    enabled: true
    name: Dify
    base_url: "https://api.dify.ai/v1"
    api_key: "<your-dify-app-key>"
  azure:
    enabled: false
    name: Azure OpenAI
    endpoint: "https://<resource>.openai.azure.com"
    api_key: "<your-key>"
    api_version: "2024-02-15-preview"
    deployment: "gpt-4o"     # deployment name (not model id)

output:
  dir: "./out"
  save_raw_transcript: true
  save_summary: true
  save_email_draft: true
  retention_days: 14

asr:
  backend: faster-whisper
  model_size: "small"         # tiny | base | small | medium | large-v3
  language: "en"              # 'auto' to detect
  vad: true
  force_hf: false             # check this to disable faster-whisper
  chunk_length_s: 30
  stride_length_s: 5

privacy:
  local_only_mode: true       # blocks Dify/Azure until the user opts into cloud providers
```

**Prompt resolution order** (via `prompts.py`):
1. If `summary.prompt_file` is set and readable → use its content.
2. Else if `summary.prompt_text` is set → use that.
3. Else → a **default executive-summary** prompt ships with the repo.

---

## 🔌 Backends

### Local (Ollama)
- This is the shipped default backend.
- Set model under `backends.local.model` (e.g., `llama3:latest`).
- The UI checks if Ollama is installed and if model is present.
- Great for **private** / **offline** summarization.

### Azure OpenAI
- Disable `privacy.local_only_mode` before using Azure.
- Provide **endpoint**, **api_key**, **api_version**, and **deployment** (name).
- We call the **Chat Completions** API with your deployment.

### Dify (Native App API)
- Disable `privacy.local_only_mode` before using Dify.
- Provide **base_url** and **app API key** (not OpenAI-compatible key).
- We send the transcript via `/v1/chat-messages` in blocking mode.
- If your Dify app expects a “system prompt”, we **prepend** it to the user message.

---

## 🧠 Transcript Capture (URL)
- We **prefer built-in Teams transcript** when visible.
- If the meeting is DRM‑protected for non‑owners, we **don’t** bypass it.  
  The scraper only reads safe transcripts/captions via DOM or payload sniffing.
- If transcript is not visible, switch to **owner mode** or use **file + ASR**.

---

## 🎙️ ASR (Local Files)
- Uses **faster-whisper** by default (CTranslate2).  
  - On Macs, CPU int8 is typically fastest; MPS is not used by faster‑whisper.
- **HF Whisper pipeline** is the fallback (MPS/CPU/GPU).  
- Toggle **“Force HF”** if faster‑whisper isn’t available or misbehaves.

---

## 🧪 Health Checks
Open **Settings -> System Health** in the UI to verify:
- Local Ollama, including the configured model.
- ffmpeg and ffprobe for audio/video processing.
- Chrome, Edge, or Chromium for meeting link transcript capture.
- Dify and Azure from their provider tabs after local-only mode is disabled.

If Ollama is missing, the app offers a **Download Ollama** action. If Ollama is installed but the configured model is missing, the app offers **Pull model** and runs `ollama pull <model>` for the user.

---

## 🛠️ Troubleshooting
- **Playwright error / no Chromium**: run `python -m playwright install chromium`.
- **Browser capture behind corporate proxy**: install Chrome or Edge and sign in normally. ECHO prefers a system browser instead of forcing a bundled Chromium download.
- **No transcript from URL**: Ensure you’re **owner** or transcript is visible to you.
- **ASR: “no audio stream”**: The file may not have an audio track. Check with `ffprobe`.
- **Azure “deployment not found”**: The **deployment name** is required (not just model id).
- **Ollama model missing**: Use **Settings -> System Health -> Pull model**, or run `ollama pull llama3:latest` manually.

---

## 🗺️ Wishlist and Next steps
- PII redaction + retention policies
- Zoom/Webex/GMeet capture
- Structured MoM templates (RACI, owners/dates)
- Red-team/privacy validation

---

## 🔒 Compliance & Privacy
- The URL capture flow **never attempts to bypass DRM** or protected streams.
- Local mode (Ollama) ensures content **stays on device**.
- You control what to persist via `config.yaml` (transcripts, summaries, email drafts).
