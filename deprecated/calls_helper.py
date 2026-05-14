# calls_helper.py
from __future__ import annotations
import os, json, shutil, logging, subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple
import torch

# Optional deps
try:
    import transformers  # noqa: F401
    from transformers import pipeline as hf_pipeline
    _has_transformers = True
except Exception:
    _has_transformers = False

try:
    from faster_whisper import WhisperModel
    _has_faster = True
except Exception:
    _has_faster = False

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("calls-helper")

# ---- Defaults (used if caller doesn't pass params) ----
USE_FASTER = os.getenv("USE_FASTER_WHISPER", "1") not in ("0", "false", "False")
DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # tiny|base|small|medium|large-v3
DEFAULT_LANGUAGE   = os.getenv("ASR_LANGUAGE", "en")           # 'auto' to detect
DEFAULT_VAD        = os.getenv("USE_VAD", "0") not in ("0", "false", "False")

SUPPORTED_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm"
}

# ---- Device selection for HF fallback ----
if torch.cuda.is_available():
    _HF_DEVICE = 0
    _MODEL_KW = {}
    _device_label = "cuda:0"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    _HF_DEVICE = torch.device("mps")
    _MODEL_KW = {"torch_dtype": torch.float16}
    _device_label = "mps"
else:
    _HF_DEVICE = -1
    _MODEL_KW = {}
    _device_label = "cpu"
log.info(f"[HF fallback] device: {_device_label}")

# ---- ffprobe helpers ----
def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode(errors="ignore"), p.stderr.decode(errors="ignore")

def _ffprobe_json(path: str):
    if not shutil.which("ffprobe"):
        return None
    rc, out, _ = _run(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams", path])
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

# ---- ASR backends (parameterized) ----
class _FasterWhisperASR:
    def __init__(self, model_size: str, vad: bool):
        self.model_size = model_size
        self.vad = vad
        device = "cpu"
        compute_type = "int8"
        if torch.cuda.is_available():
            device = "cuda"; compute_type = "float16"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "cpu"; compute_type = "int8"  # faster-whisper has no MPS
        log.info(f"[faster-whisper] model={model_size}, device={device}, compute_type={compute_type}, VAD={'on' if vad else 'off'}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

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
            raise RuntimeError("Empty transcript (faster-whisper).")
        return text

class _HFWhisperASR:
    def __init__(self, model_size: str):
        if not _has_transformers:
            raise RuntimeError("transformers not installed.")
        model_id = f"openai/whisper-{model_size}"
        log.info(f"[HF] loading {model_id}")
        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=_HF_DEVICE,
            model_kwargs=_MODEL_KW,
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
            raise RuntimeError("Empty transcript (HF Whisper).")
        return text

# ---- Cache ASR instances by backend+settings ----
_ASR_CACHE: Dict[Tuple[str, str, bool], object] = {}

def _get_asr(use_faster: bool, model_size: str, vad: bool):
    key = ("faster" if (use_faster and _has_faster) else "hf", model_size, bool(vad))
    if key in _ASR_CACHE:
        return _ASR_CACHE[key]
    if use_faster and _has_faster:
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
