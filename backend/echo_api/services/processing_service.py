from __future__ import annotations

import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from echo_api.core.config import load_app_config
from echo_api.core.paths import REPO_ROOT, storage_path
from echo_api.db.connection import db_session

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chat_helper  # noqa: E402
import calls_helper  # noqa: E402
import helpers_azure  # noqa: E402
import helpers_dify  # noqa: E402
import helpers_local  # noqa: E402
import prompts  # noqa: E402


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="echo-worker")
log = logging.getLogger("echo.worker")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def submit_job(job_id: str) -> None:
    log.info("Queued job %s for background processing", job_id)
    _EXECUTOR.submit(process_job, job_id)


def process_job(job_id: str) -> None:
    try:
        job = get_job(job_id)
        if not job:
            log.warning("Job %s not found; skipping", job_id)
            return
        log_event(job_id, job.get("stage", "Queued"), int(job.get("progress", 0)), "Worker picked up the job.")
        source_payload = json.loads(job["source_payload_json"] or "{}")
        source_type = job["source_type"]

        if source_type == "url":
            process_url_job(job, source_payload)
        elif source_type == "upload":
            process_upload_job(job, source_payload)
        elif source_type == "transcript":
            process_transcript_job(job, source_payload)
        else:
            fail_job(
                job_id,
                "UNSUPPORTED_SOURCE",
                "This source type is queued, but automatic processing is not implemented yet.",
            )
    except Exception as exc:
        log.exception("Job %s failed with unhandled error", job_id)
        fail_job(job_id, "PROCESSING_ERROR", str(exc))


def process_url_job(job: dict, source_payload: dict) -> None:
    config = load_app_config()
    url = source_payload.get("source", "")
    if not url:
        fail_job(job["id"], "SOURCE_MISSING", "The meeting link is missing.")
        return

    update_job(job["id"], stage="Reading transcript", progress=10, status="running", message="Opening browser and looking for a visible transcript.")
    update_meeting(job["meeting_id"], status="Reading transcript")

    capture_cfg = config.get("processing", {}).get("transcript_capture", {})
    log.info("Job %s: opening browser for transcript capture", job["id"])
    full_text, *_ = chat_helper.capture_transcript(
        url=url,
        out_dir=str(storage_path("transcript_capture")),
        sniff_seconds=int(capture_cfg.get("sniff_seconds", 6)),
        enforce_pause_ms=int(capture_cfg.get("enforce_pause_ms", 800)),
        max_scrolls=int(capture_cfg.get("max_scrolls", 320)),
        scroll_pause_ms=int(capture_cfg.get("scroll_pause_ms", 900)),
        stabilize_rounds=int(capture_cfg.get("stabilize_rounds", 3)),
        profile_dir=None,
        close_browser_after_capture=False,
        login_wait_seconds=int(config.get("processing", {}).get("login_wait_seconds", 300)),
    )

    if not full_text.strip():
        fail_job(
            job["id"],
            "TRANSCRIPT_NOT_FOUND",
            "We could not find a visible transcript. Upload the recording file or check that you have access.",
        )
        return

    update_job(
        job["id"],
        stage="Transcript captured",
        progress=45,
        status="running",
        message=f"Transcript captured ({len(full_text):,} characters).",
    )
    transcript_id = save_transcript(job["meeting_id"], full_text, "url")
    summarize_and_complete(job, transcript_id, full_text)


def process_transcript_job(job: dict, source_payload: dict) -> None:
    row = None
    with db_session() as conn:
        row = conn.execute(
            "SELECT id, text FROM transcripts WHERE meeting_id = ? ORDER BY created_at DESC LIMIT 1",
            (job["meeting_id"],),
        ).fetchone()
    if not row:
        fail_job(job["id"], "TRANSCRIPT_MISSING", "No transcript was found for this job.")
        return
    update_job(job["id"], stage="Creating MoM", progress=60, status="running", message="Using the pasted transcript.")
    update_meeting(job["meeting_id"], status="Creating MoM")
    summarize_and_complete(job, row["id"], row["text"])


def process_upload_job(job: dict, source_payload: dict) -> None:
    config = load_app_config()
    source = source_payload.get("source", "")
    if not source:
        fail_job(job["id"], "SOURCE_MISSING", "The uploaded recording path is missing.")
        return
    path = Path(source)
    if not path.exists():
        fail_job(job["id"], "UPLOAD_NOT_FOUND", "The uploaded recording could not be found on disk.")
        return

    asr = config.get("asr", {})
    update_job(
        job["id"],
        stage="Transcribing recording",
        progress=20,
        status="running",
        message=f"Transcribing {path.name} with Whisper {asr.get('model_size', 'small')}.",
    )
    update_meeting(job["meeting_id"], status="Transcribing recording")
    transcript = calls_helper.transcribe_one(
        str(path),
        model_size=str(asr.get("model_size", "small")),
        language=str(asr.get("language", "en")),
        vad=bool(asr.get("vad", False)),
        use_faster=False if asr.get("force_hf", False) else None,
    )
    if not transcript.strip():
        fail_job(job["id"], "EMPTY_TRANSCRIPT", "The recording was transcribed, but no speech text was found.")
        return

    update_job(
        job["id"],
        stage="Transcript captured",
        progress=50,
        status="running",
        message=f"Recording transcribed ({len(transcript):,} characters).",
    )
    transcript_id = save_transcript(job["meeting_id"], transcript, "upload")
    summarize_and_complete(job, transcript_id, transcript)


def summarize_and_complete(job: dict, transcript_id: str, transcript: str) -> None:
    config = load_app_config()
    backend_kind = config.get("summary", {}).get("default_backend", "local")
    update_job(
        job["id"],
        stage="Creating MoM",
        progress=65,
        status="running",
        message=f"Summarizing with {backend_kind}.",
    )
    update_meeting(job["meeting_id"], status="Creating MoM")

    summary = summarize_transcript(transcript, config)
    if summary.startswith("[") and "error" in summary.lower():
        fail_job(job["id"], "SUMMARY_ERROR", summary)
        return
    version_number = next_version_number(job["meeting_id"])
    mom_id = str(uuid4())
    now = now_iso()
    title = job.get("title") or get_meeting_title(job["meeting_id"])

    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO mom_versions (
              id, meeting_id, transcript_id, version_number, template_id, title,
              summary, content_markdown, status, backend_kind, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mom_id,
                job["meeting_id"],
                transcript_id,
                version_number,
                None,
                title,
                summary,
                summary,
                "ready_for_review",
                backend_kind,
                now,
            ),
        )
    save_outputs(title, transcript, summary, config)
    update_meeting(job["meeting_id"], status="Ready for review")
    update_job(
        job["id"],
        stage="Ready for review",
        progress=100,
        status="completed",
        error_code="",
        error_message="",
        message="MoM generated and saved. Ready for review.",
    )


def summarize_transcript(transcript: str, config: dict) -> str:
    backend = config.get("summary", {}).get("default_backend", "local")
    system_prompt = load_prompt(config)
    timeout = int(config.get("summary", {}).get("timeout_seconds", 240))

    if backend == "dify":
        dify = config.get("backends", {}).get("dify", {})
        log.info("Summarizing transcript with Dify at %s", dify.get("base_url", ""))
        return helpers_dify.summarize_dify_native(
            transcript,
            api_key=dify.get("api_key", ""),
            app_base=dify.get("base_url", "https://api.dify.ai/v1"),
            system_prompt=system_prompt,
            timeout=timeout,
        )
    if backend == "azure":
        azure = config.get("backends", {}).get("azure", {})
        log.info("Summarizing transcript with Azure deployment %s", azure.get("deployment", ""))
        return helpers_azure.summarize_azure_openai(
            transcript,
            endpoint=azure.get("endpoint", ""),
            api_key=azure.get("api_key", ""),
            api_version=azure.get("api_version", "2024-02-15-preview"),
            deployment=azure.get("deployment", ""),
            system_prompt=system_prompt,
            timeout=timeout,
        )

    local = config.get("backends", {}).get("local", {})
    log.info("Summarizing transcript with local Ollama model %s", local.get("model", "llama3:latest"))
    return helpers_local.summarize_local_ollama(
        transcript,
        model=local.get("model", "llama3:latest"),
        system_prompt=system_prompt,
    )


def load_prompt(config: dict) -> str:
    summary = config.get("summary", {})
    prompt_file = (summary.get("prompt_file") or "").strip()
    if prompt_file:
        path = Path(prompt_file).expanduser()
        if path.exists() and path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass
    prompt_text = (summary.get("prompt_text") or "").strip()
    if prompt_text:
        return prompt_text
    legacy_cfg = {
        "summary": {
            "prompt_file": prompt_file,
            "prompt_text": prompt_text,
        }
    }
    return prompts.load_prompt(legacy_cfg)


def save_transcript(meeting_id: str, text: str, source: str) -> str:
    transcript_id = str(uuid4())
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO transcripts (id, meeting_id, text, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (transcript_id, meeting_id, text, source, now_iso()),
        )
    return transcript_id


def save_outputs(title: str, transcript: str, summary: str, config: dict) -> None:
    storage = config.get("storage", {})
    output_dir = storage_path(str(storage.get("output_dir", "out")))
    base = sanitize_filename(title)
    if storage.get("save_transcript", True):
        (output_dir / f"{base}_transcript.txt").write_text(transcript, encoding="utf-8")
    if storage.get("save_summary", True):
        (output_dir / f"{base}_summary.txt").write_text(summary, encoding="utf-8")
    if storage.get("save_email_draft", True):
        (output_dir / f"{base}.eml").write_text(summary, encoding="utf-8")


def get_job(job_id: str) -> dict | None:
    with db_session() as conn:
        row = conn.execute(
            """
            SELECT q.*, m.title
            FROM queue_jobs q
            LEFT JOIN meetings m ON m.id = q.meeting_id
            WHERE q.id = ?
            """,
            (job_id,),
        ).fetchone()
        return dict(row) if row else None


def next_version_number(meeting_id: str) -> int:
    with db_session() as conn:
        value = conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM mom_versions WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()[0]
        return int(value)


def get_meeting_title(meeting_id: str) -> str:
    with db_session() as conn:
        row = conn.execute("SELECT title FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        return row["title"] if row else "Meeting"


def update_job(
    job_id: str,
    *,
    stage: str,
    progress: int,
    status: str,
    error_code: str | None = None,
    error_message: str | None = None,
    message: str | None = None,
) -> None:
    with db_session() as conn:
        conn.execute(
            """
            UPDATE queue_jobs
            SET stage = ?, progress = ?, status = ?,
                error_code = COALESCE(?, error_code),
                error_message = COALESCE(?, error_message),
                updated_at = ?
            WHERE id = ?
            """,
            (stage, progress, status, error_code, error_message, now_iso(), job_id),
        )
    event_message = message or stage
    log_event(job_id, stage, progress, event_message, "error" if status == "failed" else "info")
    log.info("Job %s: %s (%s%%, %s)", job_id, event_message, progress, status)


def update_meeting(meeting_id: str, *, status: str) -> None:
    with db_session() as conn:
        conn.execute(
            "UPDATE meetings SET status = ?, updated_at = ? WHERE id = ?",
            (status, now_iso(), meeting_id),
        )


def fail_job(job_id: str, error_code: str, error_message: str) -> None:
    job = get_job(job_id)
    if job and job.get("meeting_id"):
        update_meeting(job["meeting_id"], status="Needs attention")
    update_job(
        job_id,
        stage="Needs attention",
        progress=100,
        status="failed",
        error_code=error_code,
        error_message=error_message,
        message=error_message,
    )


def log_event(job_id: str, stage: str, progress: int, message: str, level: str = "info") -> None:
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO queue_job_events (id, job_id, stage, progress, message, level, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid4()), job_id, stage, progress, message, level, now_iso()),
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', " ", value)
    value = " ".join(value.split()).strip()
    return value[:120] or "meeting"
