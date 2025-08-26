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
  backend: local              # local | azure | dify
  model: llama3:latest        # legacy; used by local if not set under backends.local
  max_chars: 0                # 0 = no trim
  prompt_file: ""             # optional path to a .txt prompt
  prompt_text: ""             # inline fallback if file missing/empty

backends:
  local:
    model: llama3:latest
  azure:
    endpoint: "https://<resource>.openai.azure.com"
    api_key: "<your-key>"
    api_version: "2024-02-15-preview"
    deployment: "gpt-4o"     # deployment name (not model id)
  dify:
    base_url: "https://api.dify.ai"
    api_key: "<your-dify-app-key>"

output:
  dir: "./out"
  save_raw_transcript: true
  save_summary: true
  save_email_draft: true
  retention_days: 14

asr:
  model_size: "small"         # tiny | base | small | medium | large-v3
  language: "en"              # 'auto' to detect
  vad: false
  force_hf: false             # check this to disable faster-whisper
  chunk_length_s: 30
  stride_length_s: 5
  concurrency: 1
```

**Prompt resolution order** (via `prompts.py`):
1. If `summary.prompt_file` is set and readable → use its content.
2. Else if `summary.prompt_text` is set → use that.
3. Else → a **default executive-summary** prompt ships with the repo.

---

## 🔌 Backends

### Local (Ollama)
- Set model under `backends.local.model` (e.g., `llama3:latest`).
- The UI checks if Ollama is installed and if model is present.
- Great for **private** / **offline** summarization.

### Azure OpenAI
- Provide **endpoint**, **api_key**, **api_version**, and **deployment** (name).
- We call the **Chat Completions** API with your deployment.

### Dify (Native App API)
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
Click **Test All Connections** in the UI to verify:
- Local (Ollama model)
- Dify (native App API)
- Azure OpenAI (deployment reachable)

---

## 🛠️ Troubleshooting
- **Playwright error / no Chromium**: run `python -m playwright install chromium`.
- **No transcript from URL**: Ensure you’re **owner** or transcript is visible to you.
- **ASR: “no audio stream”**: The file may not have an audio track. Check with `ffprobe`.
- **Azure “deployment not found”**: The **deployment name** is required (not just model id).
- **Ollama model missing**: Run `ollama pull llama3:latest` (or your model tag).

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

