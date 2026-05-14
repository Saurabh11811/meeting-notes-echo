from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

from fastapi import UploadFile

from echo_api.core.paths import storage_path
from echo_api.db.connection import db_session


def create_jobs(payload: dict) -> list[dict]:
    sources = [source.strip() for source in payload.get("sources", []) if source and source.strip()]
    if not sources:
        raise ValueError("At least one source is required.")

    created: list[dict] = []
    should_process: list[str] = []
    with db_session() as conn:
        for source in sources:
            meeting_id = str(uuid4())
            job_id = str(uuid4())
            title = infer_title(source, payload.get("source_type", "unknown"))
            source_type = payload.get("source_type", "unknown")
            now = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """
                INSERT INTO meetings (
                  id, title, meeting_type, project, host, source_type, source_label,
                  status, confidentiality, tags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meeting_id,
                    title,
                    payload.get("meeting_type") or "Executive",
                    payload.get("project") or "",
                    payload.get("host") or "",
                    source_type,
                    source if source_type != "transcript" else "Pasted transcript",
                    "Queued",
                    payload.get("confidentiality") or "Internal",
                    json.dumps([payload.get("meeting_type") or "Executive"]),
                    now,
                    now,
                ),
            )

            if source_type == "transcript":
                conn.execute(
                    """
                    INSERT INTO transcripts (id, meeting_id, text, source, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid4()), meeting_id, source, "pasted", now),
                )

            stage = "Waiting" if not payload.get("run_now", True) else (
                "Ready to create MoM" if source_type == "transcript" else "Waiting"
            )
            status = "queued"
            conn.execute(
                """
                INSERT INTO queue_jobs (
                  id, meeting_id, job_type, source_type, source_payload_json, stage,
                  progress, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    meeting_id,
                    "generate_mom",
                    source_type,
                    json.dumps(
                        {
                            "title": title,
                            "source": source if source_type != "transcript" else "pasted transcript",
                            "template_name": payload.get("template_name") or "Executive MoM",
                        }
                    ),
                    stage,
                    0,
                    status,
                    now,
                    now,
                ),
            )
            created.append(
                {
                    "id": job_id,
                    "meeting_id": meeting_id,
                    "title": title,
                    "stage": stage,
                    "progress": 0,
                    "status": status,
                }
            )
            if payload.get("run_now", True) and source_type in ("url", "transcript", "upload"):
                should_process.append(job_id)
    if should_process:
        from echo_api.services.processing_service import submit_job

        for job_id in should_process:
            submit_job(job_id)
    return created


async def create_upload_jobs(files: list[UploadFile], payload: dict) -> list[dict]:
    if not files:
        raise ValueError("At least one file is required.")

    upload_dir = storage_path("uploads")
    saved_paths: list[str] = []
    for uploaded in files:
        if not uploaded.filename:
            continue
        safe_name = sanitize_upload_name(uploaded.filename)
        target = upload_dir / f"{uuid4()}_{safe_name}"
        with target.open("wb") as handle:
            shutil.copyfileobj(uploaded.file, handle)
        saved_paths.append(str(target))

    if not saved_paths:
        raise ValueError("No valid upload files were received.")

    return create_jobs(
        {
            **payload,
            "source_type": "upload",
            "sources": saved_paths,
        }
    )


def list_jobs() -> list[dict]:
    with db_session() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, meeting_id, source_type, stage, progress, status, error_message,
                       created_at, updated_at,
                       json_extract(source_payload_json, '$.title') AS title
                FROM queue_jobs
                ORDER BY created_at DESC
                LIMIT 100
                """
            ).fetchall()
        ]


def infer_title(source: str, source_type: str) -> str:
    if source_type == "transcript":
        first_line = next((line.strip() for line in source.splitlines() if line.strip()), "")
        return truncate(first_line or "Pasted Transcript")
    if source_type == "upload":
        stem = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
        if "_" in stem:
            maybe_uuid, rest = stem.split("_", 1)
            if len(maybe_uuid) >= 32:
                stem = rest
        return truncate(stem or "Uploaded Recording")
    try:
        parsed = urlparse(source)
        qs = parse_qs(parsed.query)
        if qs.get("id"):
            return truncate(unquote(qs["id"][0]).rsplit("/", 1)[-1].rsplit(".", 1)[0])
        candidate = unquote(parsed.path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
        return truncate(candidate or parsed.netloc or "Meeting Link")
    except Exception:
        return "Meeting Link"


def truncate(value: str, limit: int = 96) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def sanitize_upload_name(filename: str) -> str:
    name = Path(filename).name
    for char in '<>:"/\\|?*':
        name = name.replace(char, " ")
    return " ".join(name.split()).strip() or "recording"
