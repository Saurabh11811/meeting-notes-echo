# calls_helper.py
from __future__ import annotations
import os, json, shutil, logging, subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple

try:
    from echo_api.core.dependencies import find_binary
except Exception:
    find_binary = None

hf_pipeline = None
WhisperModel = None
_has_transformers: bool | None = None
_has_faster: bool | None = None

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("calls-helper")

# ---- Corporate TLS support ----
# In many corporate networks, HTTPS traffic is re-signed by a company root CA.
# Python packages such as httpx/huggingface_hub may not trust the Windows
# certificate store by default. If the optional `truststore` package is
# installed, this lets Python use the OS trust store before any model download
# is attempted. Set ECHO_USE_TRUSTSTORE=0 to disable.
if os.getenv("ECHO_USE_TRUSTSTORE", "1") not in ("0", "false", "False"):
    try:
        import truststore  # type: ignore
        truststore.inject_into_ssl()
        log.info("[TLS] truststore enabled; Python will use the OS certificate store")
    except Exception as exc:
        log.debug("[TLS] truststore not enabled: %s", exc)

# ---- Defaults (used if caller doesn't pass params) ----
USE_FASTER = os.getenv("USE_FASTER_WHISPER", "1") not in ("0", "false", "False")
DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # tiny|base|small|medium|large-v3
DEFAULT_LANGUAGE   = os.getenv("ASR_LANGUAGE", "en")           # 'auto' to detect
DEFAULT_VAD        = os.getenv("USE_VAD", "0") not in ("0", "false", "False")

SUPPORTED_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm"
}

def _torch():
    import torch

    return torch


def _has_faster_whisper() -> bool:
    global WhisperModel, _has_faster
    if _has_faster is not None:
        return _has_faster
    try:
        from faster_whisper import WhisperModel as imported_model

        WhisperModel = imported_model
        _has_faster = True
    except Exception:
        _has_faster = False
    return _has_faster


def _has_transformers_pipeline() -> bool:
    global hf_pipeline, _has_transformers
    if _has_transformers is not None:
        return _has_transformers
    try:
        import transformers  # noqa: F401
        from transformers import pipeline as imported_pipeline

        hf_pipeline = imported_pipeline
        _has_transformers = True
    except Exception:
        _has_transformers = False
    return _has_transformers


def _hf_device_and_kwargs():
    torch = _torch()
    if torch.cuda.is_available():
        return 0, {}, "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), {"torch_dtype": torch.float16}, "mps"
    return -1, {}, "cpu"

# ---- ffprobe helpers ----
def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode(errors="ignore"), p.stderr.decode(errors="ignore")

def _ffprobe_json(path: str):
    ffprobe = find_binary("ffprobe", env_var="ECHO_FFPROBE_PATH") if find_binary else shutil.which("ffprobe")
    if not ffprobe:
        return None
    rc, out, _ = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None

def probe_summary(path: str) -> str:
    meta = _ffprobe_json(path)
    if not meta:
        return "ffprobe: unavailable"
    fmt = meta.get("format", {})
    dur = fmt.get("duration")
    size = fmt.get("size")
    streams = meta.get("streams", [])
    aud = [s for s in streams if s.get("codec_type") == "audio"]
    a = aud[0] if aud else {}
    return f"duration={dur}, size_bytes={size}, audio_codec={a.get('codec_name')}, ch={a.get('channels')}, sr={a.get('sample_rate')}"

def _source_has_audio(path: str) -> bool:
    meta = _ffprobe_json(path)
    if not meta:
        return True
    return any(s.get("codec_type") == "audio" for s in meta.get("streams", []))

# ---- Model resolution helpers ----
def _truthy_env(name: str) -> bool:
    return os.getenv(name, "0") not in ("", "0", "false", "False", "no", "No")

def _candidate_model_dirs(model_size: str, backend: str) -> list[Path]:
    """Return local model folders to try before allowing Hugging Face download."""
    env_names = []
    if backend == "faster":
        env_names = ["FASTER_WHISPER_MODEL_PATH", "WHISPER_MODEL_PATH", "ASR_MODEL_PATH"]
    else:
        env_names = ["HF_WHISPER_MODEL_PATH", "WHISPER_MODEL_PATH", "ASR_MODEL_PATH"]

    candidates: list[Path] = []
    for name in env_names:
        val = os.getenv(name)
        if val:
            candidates.append(Path(val).expanduser())

    # Project-local defaults. These let the Windows runner work without any
    # user-specific environment variables once the model is copied into ./models.
    here = Path(__file__).resolve()
    backend_dir = here.parent
    project_root = backend_dir.parent
    if backend == "faster":
        folder = f"faster-whisper-{model_size}"
    else:
        folder = f"whisper-{model_size}"
    candidates.extend([
        project_root / "models" / folder,
        backend_dir / "models" / folder,
        Path.home() / ".cache" / "meeting-notes-echo" / "models" / folder,
    ])
    return candidates

def _find_local_model_dir(model_size: str, backend: str) -> Optional[Path]:
    for candidate in _candidate_model_dirs(model_size, backend):
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None

def _offline_model_error(model_size: str, backend: str) -> RuntimeError:
    tried = "\n  - ".join(str(p) for p in _candidate_model_dirs(model_size, backend))
    if backend == "faster":
        repo = f"Systran/faster-whisper-{model_size}"
        env = "FASTER_WHISPER_MODEL_PATH"
    else:
        repo = f"openai/whisper-{model_size}"
        env = "HF_WHISPER_MODEL_PATH"
    return RuntimeError(
        "ASR model is not available locally and online downloads are disabled or blocked.\n"
        f"Expected backend: {backend}\n"
        f"Expected model: {repo}\n"
        "Download this model on a network that can access Hugging Face, copy the full "
        "model folder to this machine, then either:\n"
        f"  1) set ${env} to that folder, or\n"
        f"  2) place it in one of these locations:\n  - {tried}"
    )

def _resolve_model_ref(model_size: str, backend: str) -> str:
    """Use a local model folder when present; otherwise return the remote id/alias."""
    local_dir = _find_local_model_dir(model_size, backend)
    if local_dir:
        log.info("[%s] using local model folder: %s", backend, local_dir)
        return str(local_dir)

    if _truthy_env("HF_HUB_OFFLINE") or _truthy_env("TRANSFORMERS_OFFLINE") or _truthy_env("ECHO_ASR_OFFLINE"):
        raise _offline_model_error(model_size, backend)

    if backend == "faster":
        log.info("[faster-whisper] local model not found; allowing Hugging Face download for model=%s", model_size)
        return model_size

    model_id = f"openai/whisper-{model_size}"
    log.info("[HF] local model not found; allowing Hugging Face download for %s", model_id)
    return model_id

# ---- ASR backends (parameterized) ----
class _FasterWhisperASR:
    def __init__(self, model_size: str, vad: bool):
        self.model_size = model_size
        self.vad = vad
        torch = _torch()
        device = "cpu"
        compute_type = "int8"
        if torch.cuda.is_available():
            device = "cuda"; compute_type = "float16"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "cpu"; compute_type = "int8"  # faster-whisper has no MPS
        log.info(f"[faster-whisper] model={model_size}, device={device}, compute_type={compute_type}, VAD={'on' if vad else 'off'}")
        model_ref = _resolve_model_ref(model_size, "faster")
        self.model = WhisperModel(model_ref, device=device, compute_type=compute_type)

    def transcribe(self, media_path: str, language: Optional[str]) -> str:
        if not _source_has_audio(media_path):
            raise RuntimeError(f"No audio stream. Probe: {probe_summary(media_path)}")
        segments, _info = self.model.transcribe(
            media_path,
            language=None if (not language or language.lower()=="auto") else language,
            beam_size=1,
            vad_filter=self.vad,
            vad_parameters={"min_silence_duration_ms": 500} if self.vad else None,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join((getattr(seg, "text", "") or "").strip() for seg in segments)
        if not text.strip():
            # This is a valid outcome for silent recordings or files where VAD
            # removes all audio. Return an empty transcript so the job layer can
            # mark it as EMPTY_TRANSCRIPT instead of crashing as an unhandled ASR error.
            log.warning("[faster-whisper] no speech text detected. Probe: %s", probe_summary(media_path))
            return ""
        return text

class _HFWhisperASR:
    def __init__(self, model_size: str):
        if not _has_transformers_pipeline():
            raise RuntimeError("transformers not installed.")
        device, model_kw, device_label = _hf_device_and_kwargs()
        model_ref = _resolve_model_ref(model_size, "hf")
        log.info(f"[HF] loading {model_ref} on {device_label}")
        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_ref,
            device=device,
            model_kwargs=model_kw,
        )

    def transcribe(self, media_path: str, language: Optional[str]) -> str:
        if not _source_has_audio(media_path):
            raise RuntimeError(f"No audio stream. Probe: {probe_summary(media_path)}")
        base_gk = {"task":"transcribe","do_sample":False,"num_beams":1,"temperature":0.0}
        if language and language.lower() != "auto":
            base_gk["language"] = language
        out = self.pipe(media_path, chunk_length_s=30, stride_length_s=5, return_timestamps=False,
                        generate_kwargs=base_gk)
        text = out["text"] if isinstance(out, dict) else str(out)
        if not text.strip():
            log.warning("[HF Whisper] no speech text detected. Probe: %s", probe_summary(media_path))
            return ""
        return text

# ---- Cache ASR instances by backend+settings ----
_ASR_CACHE: Dict[Tuple[str, str, bool], object] = {}

def _get_asr(use_faster: bool, model_size: str, vad: bool):
    faster_available = _has_faster_whisper() if use_faster else False
    key = ("faster" if (use_faster and faster_available) else "hf", model_size, bool(vad))
    if key in _ASR_CACHE:
        return _ASR_CACHE[key]
    if use_faster and faster_available:
        asr = _FasterWhisperASR(model_size=model_size, vad=vad)
    else:
        asr = _HFWhisperASR(model_size=model_size)
    _ASR_CACHE[key] = asr
    return asr

def transcribe_one(
    media_path: str,
    *,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    vad: Optional[bool] = None,
    use_faster: Optional[bool] = None,
) -> str:
    """
    Transcribe a single file with requested settings.
    - model_size: tiny|base|small|medium|large-v3
    - language : e.g., 'en', or 'auto'
    - vad      : True/False
    - use_faster: True=prefer faster-whisper, False=force HF, None=env default
    """
    ms = (model_size or DEFAULT_MODEL_SIZE).strip()
    lang = (language or DEFAULT_LANGUAGE).strip()
    vd = DEFAULT_VAD if vad is None else bool(vad)
    uf = USE_FASTER if use_faster is None else bool(use_faster)
    asr = _get_asr(use_faster=uf, model_size=ms, vad=vd)
    return asr.transcribe(media_path, lang)

__all__ = ["SUPPORTED_EXTS", "probe_summary", "transcribe_one"]
