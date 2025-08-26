# mvp_single_ui.py  (only showing full file for paste)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meeting Notes — MVP (single page)
- Left-side Setup panel (show/hide via button)
- Strict OR input: either file OR Teams/Stream URL
- Summary (full) → Transcript (full) → Status (saved paths + timings)
- Backends: Local (Ollama), Dify (native), Azure OpenAI
- Force HF (disable faster-whisper) toggle for ASR
- Test All Connections button
- Prompt centralization via prompts.py (file or inline override)
- Settings saved to ./config.yaml (auto-created on first run)
"""
import os, sys, logging, time, platform, shutil, subprocess, yaml
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from typing import Any, Dict, Tuple
import gradio as gr

import chat_helper
import calls_helper
import helpers_local
import helpers_azure
import helpers_dify
import prompts  # NEW

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mvp-ui")

# ---------------- Config -----------------
CONFIG_PATH = Path("./config.yaml")

DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "platform": "teams",
        "owner_mode": True,
        "timezone": "Asia/Kolkata",
    },
    "summary": {
        "backend": "local",               # local | dify | azure
        "model": os.getenv("OLLAMA_MODEL", "llama3:latest"),  # legacy; used by local as fallback
        "max_chars": 0,
        "prompt_file": "",               # NEW: path to prompt file
        "prompt_text": "",               # NEW: inline override
    },
    "backends": {
        "local": {
            "model": os.getenv("OLLAMA_MODEL", "llama3:latest")
        },
        "dify": {
            "base_url": "https://api.dify.ai",
            "api_key": ""
        },
        "azure": {
            "endpoint": "",
            "api_key": "",
            "api_version": "2024-02-15-preview",
            "deployment": ""
        }
    },
    "output": {
        "dir": "./out",
        "save_raw_transcript": True,
        "save_summary": True,
        "save_email_draft": True,
        "retention_days": 14,
    },
    "asr": {
        "preset": "auto",
        "backend": "",
        "model_size": os.getenv("WHISPER_MODEL_SIZE", "small"),
        "language": os.getenv("ASR_LANGUAGE", "en"),
        "vad": os.getenv("USE_VAD", "0") not in ("0", "false", "False"),
        "force_hf": False,
        "chunk_length_s": 30,
        "stride_length_s": 5,
        "concurrency": 1,
    }
}

def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in base.items():
        if isinstance(v, dict):
            out[k] = merge_dicts(v, override.get(k, {})) if isinstance(override.get(k), dict) else v.copy()
        else:
            out[k] = override.get(k, v)
    for k, v in override.items():
        if k not in out:
            out[k] = v
    return out

def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            return merge_dicts(DEFAULT_CONFIG, cfg)
        except Exception as e:
            log.warning(f"Failed to read config.yaml: {e}; using defaults")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

CONFIG = load_config()
if not CONFIG_PATH.exists():
    save_config(CONFIG)

OUT_DIR = Path(CONFIG["output"]["dir"]).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Environment Checks ----------------
def which_ffmpeg() -> str:
    return shutil.which("ffmpeg") or ""

def playwright_ok() -> bool:
    try:
        import playwright.sync_api  # noqa
        return True
    except Exception:
        return False

def torch_mps_ok() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False

def ollama_installed() -> bool:
    return shutil.which("ollama") is not None

def ollama_has_model(tag: str) -> bool:
    if not ollama_installed():
        return False
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if line.strip().startswith(tag):
                return True
        return False
    except Exception:
        return False

def os_label() -> str:
    sysname = platform.system().lower()
    if sysname == "darwin":
        return f"macOS {platform.mac_ver()[0] or ''}".strip()
    if sysname == "windows":
        return f"Windows {platform.release()}"
    return f"Linux {platform.release()}"

def output_writable(path_str: str):
    try:
        p = Path(path_str).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True, str(p.resolve())
    except Exception as e:
        return False, f"{path_str} ({e})"

def render_status(cfg: Dict[str, Any]) -> str:
    py_ok = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
    ff = which_ffmpeg(); ff_ok = bool(ff)
    pw_ok = playwright_ok()
    mps_ok = torch_mps_ok()
    out_ok, out_detail = output_writable(cfg["output"]["dir"])
    model = cfg["backends"]["local"]["model"] or cfg["summary"]["model"]
    oll_ok = ollama_installed()
    mdl_ok = ollama_has_model(model) if oll_ok else False
    mark = lambda b: "✅" if b else "❌"
    return "\n".join([
        f"OS: {os_label()} {mark(True)}",
        f"Python ≥ 3.10 (found {platform.python_version()}) {mark(py_ok)}",
        f"ffmpeg ({ff or 'not found'}) {mark(ff_ok)}",
        f"Playwright Python import {mark(pw_ok)}",
        f"PyTorch MPS available {mark(mps_ok)}",
        f"Output folder writable: {out_detail} {mark(out_ok)}",
        f"Ollama ({'installed' if oll_ok else 'not installed'}) — {model} present {mark(mdl_ok)}",
    ])

# ---------------- Summarization Dispatcher ----------------
def summarize_dispatch(text: str, cfg: Dict[str, Any]) -> str:
    backend = (cfg["summary"]["backend"] or "local").lower()
    system_prompt = prompts.load_prompt(cfg)  # NEW: single-source prompt
    if backend == "local":
        model = cfg["backends"]["local"]["model"] or cfg["summary"]["model"]
        return helpers_local.summarize_local_ollama(text, model, system_prompt=system_prompt)
    elif backend == "azure":
        b = cfg["backends"]["azure"]
        return helpers_azure.summarize_azure_openai(
            text,
            endpoint=b["endpoint"],
            api_key=b["api_key"],
            api_version=b["api_version"],
            deployment=b["deployment"],
            system_prompt=system_prompt,
        )
    elif backend == "dify":
        b = cfg["backends"]["dify"]
        return helpers_dify.summarize_dify_native(
            text,
            api_key=b["api_key"],
            app_base=b.get("base_url", "https://api.dify.ai"),
            system_prompt=system_prompt,
        )
    else:
        return f"[Unsupported backend: {backend}]"

# ---------------- Name inference & Save ----------------
def _sanitize_filename(name: str) -> str:
    bad = r'<>:"/\\|?*'
    for ch in bad: name = name.replace(ch, " ")
    return " ".join(name.split()).strip() or "recording"

def base_from_file(file_path: str) -> str:
    return _sanitize_filename(Path(file_path).stem)

def base_from_url(url: str) -> str:
    try:
        u = urlparse(url); qs = parse_qs(u.query)
        if "id" in qs and qs["id"]:
            decoded = unquote(qs["id"][0]); candidate = Path(decoded).name
        else:
            candidate = unquote(Path(u.path).name or "")
        return _sanitize_filename(Path(candidate).stem if candidate else "recording")
    except Exception:
        return "recording"

def save_transcript(text: str, base: str, cfg: Dict[str, Any]) -> Path | None:
    if not cfg["output"]["save_raw_transcript"]:
        return None
    out_dir = Path(cfg["output"]["dir"]).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{base}_transcript.txt"; p.write_text(text or "", encoding="utf-8"); return p

def save_summary_file(text: str, base: str, cfg: Dict[str, Any]) -> Path | None:
    if not cfg["output"]["save_summary"]:
        return None
    out_dir = Path(cfg["output"]["dir"]).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{base}_summary.txt"; p.write_text(text or "", encoding="utf-8"); return p

def write_eml(summary_text: str, base: str, cfg: Dict[str, Any]) -> Path | None:
    if not cfg["output"]["save_email_draft"]:
        return None
    from email.utils import formatdate
    now = formatdate(localtime=True); subj = f"{base} — Meeting Summary"
    eml = f"""From: "Me" <me@example.com>
To:
Subject: {subj}
Date: {now}
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8

{summary_text}
"""
    out_dir = Path(cfg["output"]["dir"]).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{base}.eml"; p.write_text(eml, encoding="utf-8"); return p

# ---------------- Running-timer rendering ----------------
def fmt_secs(s: float) -> str: return f"{s:0.2f}s"
def render_running(phase: str, t_cap: float, t_sum: float, t_tot: float) -> str:
    return (f"⏳ {phase}\n"
            f"• Capture/Transcribe: {fmt_secs(t_cap)}\n"
            f"• Summarize        : {fmt_secs(t_sum)}\n"
            f"• Total            : {fmt_secs(t_tot)}")
def render_final(t_cap: float, t_sum: float, t_tot: float, paths: Tuple[Path | None, Path | None, Path | None]) -> str:
    tpath, spath, epath = paths
    lines = ["✅ Saved files",
             f"• Transcript → {tpath}" if tpath else "• Transcript → (not saved)",
             f"• Summary   → {spath}" if spath else "• Summary   → (not saved)",
             f"• Email     → {epath}" if epath else "• Email     → (not saved)",
             "", "⏱️ Timings",
             f"• Capture/Transcribe: {fmt_secs(t_cap)}",
             f"• Summarize        : {fmt_secs(t_sum)}",
             f"• Total            : {fmt_secs(t_tot)}"]
    return "\n".join(lines)

# ---------------- Main generator handler ----------------
SUPPORTED_EXTS = sorted(list(calls_helper.SUPPORTED_EXTS))

def process(fileobj, url_text, cfg_state):
    cfg = cfg_state or CONFIG
    url_text = (url_text or "").strip()
    has_file = bool(fileobj); has_url = bool(url_text)

    if has_file and has_url:
        yield ("", "", "Please provide either a file OR a URL, not both."); return
    if not has_file and not has_url:
        yield ("", "", "Please upload a file OR paste a recording URL."); return

    base = base_from_file(fileobj.name if hasattr(fileobj, "name") else str(fileobj)) if has_file else base_from_url(url_text)

    t0 = time.time(); t_cap_start = time.time()
    t_cap = 0.0; t_sum = 0.0

    # Phase 1: Capture/Transcribe
    import concurrent.futures
    def do_capture():
        if has_file:
            return calls_helper.transcribe_one(
                fileobj.name if hasattr(fileobj, "name") else str(fileobj),
                model_size=str(cfg["asr"]["model_size"]),
                language=str(cfg["asr"]["language"]),
                vad=bool(cfg["asr"]["vad"]),
                use_faster=(False if cfg["asr"]["force_hf"] else None),
            )
        full_text, *_ = chat_helper.capture_transcript(
            url=url_text, out_dir="./out_transcript",
            sniff_seconds=6, enforce_pause_ms=800, max_scrolls=320,
            scroll_pause_ms=900, stabilize_rounds=3,
            profile_dir=os.getenv("PW_PROFILE_DIR"), close_browser_after_capture=True,
        )
        return full_text

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(do_capture)
        while True:
            if fut.done(): break
            t_cap = time.time() - t_cap_start; t_tot = time.time() - t0
            running = render_running("Running: Capture/Transcribe…", t_cap, t_sum, t_tot)
            yield (running, running, running); time.sleep(0.5)
        try:
            transcript = fut.result()
        except Exception as e:
            yield ("", "", f"{'Transcription' if has_file else 'Capture'} error: {e}"); return

    if not transcript or not transcript.strip():
        yield ("", "", "No transcript found (owner? transcript visible?)."); return

    t_cap = time.time() - t_cap_start
    t_tot = time.time() - t0
    running = render_running("Starting: Summarize…", t_cap, t_sum, t_tot)
    yield (running, transcript, running)

    # Phase 2: Summarize
    t_sum_start = time.time()
    def do_summary(): return summarize_dispatch(transcript, cfg)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(do_summary)
        while True:
            if fut.done(): break
            t_sum = time.time() - t_sum_start; t_tot = time.time() - t0
            running = render_running("Running: Summarize…", t_cap, t_sum, t_tot)
            yield (running, transcript, running); time.sleep(0.5)
        try:
            summary = fut.result()
        except Exception as e:
            yield ("", transcript, f"Summary error: {e}"); return

    t_sum = time.time() - t_sum_start; t_tot = time.time() - t0

    tpath = save_transcript(transcript, base, cfg)
    spath = save_summary_file(summary, base, cfg)
    epath = write_eml(summary, base, cfg)
    final_status = render_final(t_cap, t_sum, t_tot, (tpath, spath, epath))

    yield (summary, transcript, final_status)

# ---------------- Gradio UI ----------------
with gr.Blocks(
    title="Meeting Notes — MVP (File OR URL)",
    css="""
/* simple sidebar feel */
.sidebar-wrap { border-right: 1px solid #eee; padding-right: 8px; }
"""
) as app:
    gr.Markdown("### Meeting Notes — MVP\nProvide **either** a file **OR** a Teams/Stream URL. "
                "No DRM bypass; only visible transcript or safe caption payloads.")

    cfg_state = gr.State(CONFIG)
    sidebar_visible = gr.State(True)

    # Top bar with toggle
    with gr.Row():
        toggle_btn = gr.Button("☰ Setup")
        gr.Markdown("")  # spacer

    with gr.Row():
        # Left sidebar
        with gr.Column(scale=2, elem_classes=["sidebar-wrap"]) as sidebar_col:
            with gr.Group(visible=True) as setup_group:
                env_status = gr.Textbox(label="Environment Status", value=render_status(CONFIG), lines=10)

                # ---- Summary backend selection ----
                backend_dd = gr.Dropdown(
                    label="Summary Backend",
                    choices=["local","dify","azure"],
                    value=CONFIG["summary"]["backend"]
                )

                # Local
                local_model = gr.Textbox(
                    label="Local (Ollama) model",
                    value=CONFIG["backends"]["local"]["model"],
                    placeholder="e.g., llama3:latest"
                )

                # Dify (Native)
                dify_base = gr.Textbox(label="Dify Base URL", value=CONFIG["backends"]["dify"]["base_url"])
                dify_key  = gr.Textbox(label="Dify API Key", value=CONFIG["backends"]["dify"]["api_key"], type="password")

                # Azure
                az_endpoint   = gr.Textbox(label="Azure OpenAI Endpoint", value=CONFIG["backends"]["azure"]["endpoint"])
                az_key        = gr.Textbox(label="Azure OpenAI API Key", value=CONFIG["backends"]["azure"]["api_key"], type="password")
                az_api_ver    = gr.Textbox(label="Azure API Version", value=CONFIG["backends"]["azure"]["api_version"])
                az_deployment = gr.Textbox(label="Azure Deployment name (not model)", value=CONFIG["backends"]["azure"]["deployment"])

                # Prompt inputs (file or inline)
                gr.Markdown("---")
                prompt_file = gr.Textbox(label="Prompt file path (optional)", value=CONFIG["summary"].get("prompt_file",""), placeholder="./prompt.txt")
                prompt_text = gr.Textbox(label="Inline prompt override (optional)", value=CONFIG["summary"].get("prompt_text",""), lines=6, placeholder="If set, used when file is empty/missing")

                # Output & saving
                gr.Markdown("---")
                outdir_input = gr.Textbox(label="Output folder", value=str(Path(CONFIG["output"]["dir"]).expanduser()))
                with gr.Row():
                    save_transcript_chk = gr.Checkbox(label="Save transcript (.txt)", value=bool(CONFIG["output"]["save_raw_transcript"]))
                    save_summary_chk   = gr.Checkbox(label="Save summary (.txt)", value=bool(CONFIG["output"]["save_summary"]))
                    save_eml_chk       = gr.Checkbox(label="Save email draft (.eml)", value=bool(CONFIG["output"]["save_email_draft"]))

                gr.Markdown("**Advanced ASR (file uploads)**")
                asr_model_size = gr.Dropdown(label="Whisper model size", choices=["tiny","base","small","medium","large-v3"], value=str(CONFIG["asr"]["model_size"]))
                asr_language   = gr.Textbox(label="ASR language (use 'auto' to detect)", value=str(CONFIG["asr"]["language"]))
                asr_vad        = gr.Checkbox(label="Voice Activity Detection (VAD)", value=bool(CONFIG["asr"]["vad"]))
                force_hf_chk   = gr.Checkbox(label="Force HF (disable faster-whisper)", value=bool(CONFIG["asr"]["force_hf"]))

                with gr.Row():
                    save_btn = gr.Button("Save Settings", variant="secondary")
                    test_btn = gr.Button("Test All Connections")
                test_result = gr.Textbox(label="Backend Test Result", lines=6)

        # Main content (right)
        with gr.Column(scale=5):
            with gr.Row():
                file_in = gr.File(label="Upload a file (audio/video)", file_count="single", file_types=SUPPORTED_EXTS, scale=1)
                gr.HTML("<div style='font-weight:600; text-align:center; padding-top:28px;'>— OR —</div>")
                url_in = gr.Textbox(label="Recording URL", placeholder="https://...", scale=2)

            run_btn = gr.Button("Process", variant="primary")

            summary_out    = gr.Textbox(label="Summary", lines=16)
            transcript_out = gr.Textbox(label="Transcript", lines=18)
            status_out     = gr.Textbox(label="Status / Saved Paths / Timings", lines=10)

    # Toggle sidebar visibility
    def on_toggle(vis):
        vis = not bool(vis)
        return vis, gr.update(visible=vis)
    toggle_btn.click(on_toggle, inputs=[sidebar_visible], outputs=[sidebar_visible, setup_group])

    # Save settings
    def on_save(_backend,
                _local_model,
                _dify_base, _dify_key,
                _az_endpoint, _az_key, _az_api_ver, _az_deploy,
                _prompt_file, _prompt_text,
                _outdir, _save_t, _save_s, _save_e,
                _asr_model, _asr_lang, _asr_vad, _asr_force, _cfg):
        new_cfg = merge_dicts(DEFAULT_CONFIG, _cfg or {})
        new_cfg["summary"]["backend"] = (_backend or "local").strip().lower()
        # local
        new_cfg["backends"]["local"]["model"] = (_local_model or "llama3:latest").strip()
        # dify
        new_cfg["backends"]["dify"]["base_url"] = (_dify_base or "https://api.dify.ai").strip()
        new_cfg["backends"]["dify"]["api_key"]  = (_dify_key or "").strip()
        # azure
        new_cfg["backends"]["azure"]["endpoint"]   = (_az_endpoint or "").strip()
        new_cfg["backends"]["azure"]["api_key"]    = (_az_key or "").strip()
        new_cfg["backends"]["azure"]["api_version"]= (_az_api_ver or "2024-02-15-preview").strip()
        new_cfg["backends"]["azure"]["deployment"] = (_az_deploy or "").strip()
        # prompt
        new_cfg["summary"]["prompt_file"] = (_prompt_file or "").strip()
        new_cfg["summary"]["prompt_text"] = (_prompt_text or "").strip()
        # output
        new_cfg["output"]["dir"] = (_outdir or "./out").strip()
        new_cfg["output"]["save_raw_transcript"] = bool(_save_t)
        new_cfg["output"]["save_summary"]        = bool(_save_s)
        new_cfg["output"]["save_email_draft"]    = bool(_save_e)
        # asr
        new_cfg["asr"]["model_size"] = (_asr_model or "small").strip()
        new_cfg["asr"]["language"]   = (_asr_lang or "en").strip()
        new_cfg["asr"]["vad"]        = bool(_asr_vad)
        new_cfg["asr"]["force_hf"]   = bool(_asr_force)

        save_config(new_cfg)
        global CONFIG
        CONFIG = new_cfg
        return (
            CONFIG,
            render_status(CONFIG),
            gr.update(value=str(Path(CONFIG["output"]["dir"]).expanduser()))
        )

    save_btn.click(
        on_save,
        inputs=[
            backend_dd,
            local_model,
            dify_base, dify_key,
            az_endpoint, az_key, az_api_ver, az_deployment,
            prompt_file, prompt_text,
            outdir_input, save_transcript_chk, save_summary_chk, save_eml_chk,
            asr_model_size, asr_language, asr_vad, force_hf_chk, cfg_state
        ],
        outputs=[cfg_state, env_status, outdir_input]
    )

    # Test All Connections
    def on_test(_cfg):
        cfg = _cfg or CONFIG
        lines = []

        # Local
        model = (cfg["backends"]["local"]["model"] or "llama3:latest")
        try:
            resp = helpers_local.summarize_local_ollama("ping", model=model, system_prompt="You are a health check.")
            ok = not resp.lower().startswith("[local summarizer unavailable")
            lines.append(f"Local (Ollama: {model}): " + ("OK ✅" if ok else f"FAIL ❌ — {resp[:120]}"))
        except Exception as e:
            lines.append(f"Local (Ollama: {model}): FAIL ❌ — {e}")

        # Dify
        dify_base = cfg["backends"]["dify"]["base_url"]
        dify_key  = cfg["backends"]["dify"]["api_key"]
        if dify_key:
            ok, msg = helpers_dify.test_dify_connection(api_key=dify_key, app_base=dify_base)
            lines.append(f"Dify ({dify_base}): " + ("OK ✅" if ok else f"FAIL ❌ — {msg}"))
        else:
            lines.append(f"Dify ({dify_base}): SKIP ⚠️ — API key not set")

        # Azure
        az = cfg["backends"]["azure"]
        if az["endpoint"] and az["api_key"] and az["deployment"]:
            ok, msg = helpers_azure.test_azure_connection(
                endpoint=az["endpoint"],
                api_key=az["api_key"],
                api_version=az["api_version"],
                deployment=az["deployment"],
            )
            lines.append(f"Azure (dep: {az['deployment']}): " + ("OK ✅" if ok else f"FAIL ❌ — {msg}"))
        else:
            missing = []
            if not az["endpoint"]:   missing.append("endpoint")
            if not az["api_key"]:    missing.append("api_key")
            if not az["deployment"]: missing.append("deployment")
            miss = ", ".join(missing) or "config"
            lines.append(f"Azure: SKIP ⚠️ — missing {miss}")

        return "\n".join(lines)

    test_btn.click(on_test, inputs=[cfg_state], outputs=[test_result])

    # Main process
    run_btn.click(process, inputs=[file_in, url_in, cfg_state], outputs=[summary_out, transcript_out, status_out])

if __name__ == "__main__":
    app.launch()
