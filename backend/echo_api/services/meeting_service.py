from __future__ import annotations

import json
import re
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from echo_api.core.paths import storage_path
from echo_api.db.connection import db_session
from echo_api.services.processing_service import now_iso, start_job


def list_meetings(limit: int = 100) -> list[dict]:
    with db_session() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT m.id, m.title, m.meeting_type, m.project, m.host, m.source_type,
                       m.source_label, m.status, m.confidentiality, m.created_at, m.updated_at,
                       COALESCE((SELECT COUNT(*) FROM decisions d WHERE d.meeting_id = m.id), 0) AS decisions_count,
                       COALESCE((SELECT COUNT(*) FROM action_items a WHERE a.meeting_id = m.id), 0) AS action_items_count,
                       COALESCE((SELECT MAX(version_number) FROM mom_versions mv WHERE mv.meeting_id = m.id), 0) AS mom_version
                FROM meetings m
                ORDER BY m.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        ]


def get_meeting_detail(meeting_id: str) -> dict | None:
    with db_session() as conn:
        meeting = conn.execute(
            """
            SELECT id, title, meeting_type, project, host, source_type, source_label,
                   status, confidentiality, tags_json, created_at, updated_at,
                   COALESCE((SELECT COUNT(*) FROM decisions d WHERE d.meeting_id = meetings.id), 0) AS decisions_count,
                   COALESCE((SELECT COUNT(*) FROM action_items a WHERE a.meeting_id = meetings.id), 0) AS action_items_count,
                   COALESCE((SELECT MAX(version_number) FROM mom_versions mv WHERE mv.meeting_id = meetings.id), 0) AS mom_version
            FROM meetings
            WHERE id = ?
            """,
            (meeting_id,),
        ).fetchone()
        if not meeting:
            return None

        mom_versions = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, version_number, title, summary, content_markdown, status,
                       backend_kind, created_at, approved_at
                FROM mom_versions
                WHERE meeting_id = ?
                ORDER BY version_number DESC
                """,
                (meeting_id,),
            ).fetchall()
        ]

        transcript = conn.execute(
            """
            SELECT id, text, source, language, created_at
            FROM transcripts
            WHERE meeting_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (meeting_id,),
        ).fetchone()

        decisions = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, description, context, owner, source_ref, created_at
                FROM decisions
                WHERE meeting_id = ?
                ORDER BY created_at ASC
                """,
                (meeting_id,),
            ).fetchall()
        ]
        action_items = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, description, owner, due_date, status, project, confidence, source_ref, created_at
                FROM action_items
                WHERE meeting_id = ?
                ORDER BY created_at ASC
                """,
                (meeting_id,),
            ).fetchall()
        ]
        risks = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, description, severity, mitigation, source_ref, created_at
                FROM risks
                WHERE meeting_id = ?
                ORDER BY created_at ASC
                """,
                (meeting_id,),
            ).fetchall()
        ]
        jobs = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, stage, progress, status, error_code, error_message,
                       created_at, updated_at
                FROM queue_jobs
                WHERE meeting_id = ?
                ORDER BY created_at DESC
                """,
                (meeting_id,),
            ).fetchall()
        ]
        for job in jobs:
            job["events"] = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT stage, progress, message, level, created_at
                    FROM queue_job_events
                    WHERE job_id = ?
                    ORDER BY created_at ASC
                    """,
                    (job["id"],),
                ).fetchall()
            ]

    meeting_dict = dict(meeting)
    latest_mom = mom_versions[0] if mom_versions else None
    meeting_dict["meeting_occurred_at"] = infer_meeting_occurred_at(meeting_dict)
    meeting_dict["source_info"] = source_info(meeting_dict)
    meeting_dict["mom_generated_at"] = latest_mom["created_at"] if latest_mom else None

    return {
        "meeting": meeting_dict,
        "latest_mom": latest_mom,
        "mom_versions": mom_versions,
        "transcript": dict(transcript) if transcript else None,
        "decisions": decisions,
        "action_items": action_items,
        "risks": risks,
        "jobs": jobs,
    }


def regenerate_meeting_mom(
    meeting_id: str,
    *,
    template_name: str = "Executive MoM",
    backend_kind: str = "",
    run_now: bool = True,
) -> dict:
    now = now_iso()
    with db_session() as conn:
        meeting = conn.execute(
            "SELECT id, title, meeting_type FROM meetings WHERE id = ?",
            (meeting_id,),
        ).fetchone()
        if not meeting:
            raise ValueError("Meeting not found.")
        transcript = conn.execute(
            "SELECT id FROM transcripts WHERE meeting_id = ? ORDER BY created_at DESC LIMIT 1",
            (meeting_id,),
        ).fetchone()
        if not transcript:
            raise ValueError("This meeting has no saved transcript to regenerate from.")

        resolved_template_name = template_name
        if not resolved_template_name:
            template = conn.execute(
                """
                SELECT name
                FROM templates
                WHERE meeting_type = ?
                ORDER BY is_default DESC, is_locked DESC, name ASC
                LIMIT 1
                """,
                (meeting["meeting_type"],),
            ).fetchone()
            if not template:
                template = conn.execute(
                    "SELECT name FROM templates ORDER BY is_default DESC, is_locked DESC, name ASC LIMIT 1"
                ).fetchone()
            resolved_template_name = template["name"] if template else "Executive MoM"

        job_id = str(uuid4())
        stage = "Ready to create MoM" if run_now else "Waiting"
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
                "regenerate_mom",
                "transcript",
                json.dumps(
                    {
                        "title": meeting["title"],
                        "source": "saved transcript",
                        "template_name": resolved_template_name,
                        "backend_kind": backend_kind,
                        "auto_start": bool(run_now),
                    }
                ),
                stage,
                0,
                "queued",
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE meetings SET status = ?, updated_at = ? WHERE id = ?",
            ("Queued" if not run_now else "Ready to create MoM", now, meeting_id),
        )

    if run_now:
        started = start_job(job_id)
    else:
        started = None
    return {
        "job": {
            "id": job_id,
            "meeting_id": meeting_id,
            "stage": stage,
            "status": "queued",
            "progress": 0,
        },
        "started": started,
    }


def create_mom_export(meeting_id: str, export_type: str) -> Path:
    detail = get_meeting_detail(meeting_id)
    if not detail or not detail.get("latest_mom"):
        raise ValueError("No MoM is available to export.")
    mom = detail["latest_mom"]
    meeting = detail["meeting"]
    content = mom.get("content_markdown") or mom.get("summary") or ""
    title = meeting.get("title") or "Meeting notes"
    export_dir = storage_path("exports")
    stem = safe_filename(f"{title}_v{mom.get('version_number', 1)}")

    if export_type == "email":
        path = export_dir / f"{stem}.eml"
        path.write_text(email_draft(title, content), encoding="utf-8")
        return path
    if export_type == "html":
        path = export_dir / f"{stem}.html"
        path.write_text(render_html_document(title, content), encoding="utf-8")
        return path
    if export_type == "text":
        path = export_dir / f"{stem}.txt"
        path.write_text(markdown_to_plain_text(content), encoding="utf-8")
        return path
    if export_type == "pdf":
        path = export_dir / f"{stem}.pdf"
        write_simple_pdf(path, title, markdown_to_plain_text(content))
        return path
    raise ValueError("Unsupported export type.")


def infer_meeting_occurred_at(meeting: dict) -> str | None:
    candidates = [meeting.get("title", ""), meeting.get("source_label", "")]
    for candidate in candidates:
        match = re.search(r"(20\d{6})[_-](\d{4,6})", candidate or "")
        if not match:
            continue
        date_part, time_part = match.groups()
        if len(time_part) == 4:
            time_part = f"{time_part}00"
        try:
            return datetime.strptime(f"{date_part}{time_part[:6]}", "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            continue

    if meeting.get("source_type") == "upload" and meeting.get("source_label"):
        path = Path(meeting["source_label"])
        if path.exists():
            stat = path.stat()
            created = getattr(stat, "st_birthtime", stat.st_mtime)
            return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
    return None


def source_info(meeting: dict) -> dict:
    source_type = meeting.get("source_type") or "unknown"
    source_label = meeting.get("source_label") or ""
    if source_type == "url":
        return {
            "label": "Meeting link",
            "display": "Open meeting link",
            "href": source_label if source_label.startswith(("http://", "https://")) else "",
            "kind": "external",
        }
    if source_type == "upload":
        path = Path(source_label)
        return {
            "label": "Recording file",
            "display": path.name or "Uploaded recording",
            "href": path.resolve().as_uri() if source_label and path.exists() else "",
            "kind": "file",
        }
    if source_type == "transcript":
        return {"label": "Source", "display": "Pasted transcript", "href": "", "kind": "transcript"}
    return {"label": "Source", "display": source_label or source_type, "href": "", "kind": source_type}


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:120] or "meeting_notes"


def email_draft(title: str, content: str) -> str:
    boundary = f"echo-{safe_filename(title)[:32] or 'mom'}"
    plain = markdown_to_plain_text(content)
    html = render_html_body(content)
    return (
        f"Subject: Meeting notes - {title}\n"
        "MIME-Version: 1.0\n"
        f"Content-Type: multipart/alternative; boundary=\"{boundary}\"\n\n"
        f"--{boundary}\n"
        "Content-Type: text/plain; charset=utf-8\n\n"
        "Hi,\n\nPlease find the meeting notes below.\n\n"
        f"{plain}\n\n"
        f"--{boundary}\n"
        "Content-Type: text/html; charset=utf-8\n\n"
        "<p>Hi,</p><p>Please find the meeting notes below.</p>"
        f"{html}\n"
        f"--{boundary}--\n"
    )


def render_html_document(title: str, content: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title>"
        "<style>"
        "body{font-family:Inter,Arial,sans-serif;line-height:1.55;color:#111318;max-width:920px;margin:40px auto;padding:0 24px;background:#fff;}"
        "h1,h2,h3,h4{line-height:1.25;margin:24px 0 12px;}p{margin:8px 0;}ul,ol{padding-left:24px;}li{margin:4px 0;}"
        "table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px;}th,td{border:1px solid #d0d5dd;padding:8px 10px;text-align:left;vertical-align:top;}th{background:#f1f3f5;font-weight:700;}"
        "blockquote{border-left:3px solid #0f766e;margin:16px 0;padding-left:14px;color:#4b5563;}code{background:#f1f3f5;padding:2px 5px;border-radius:4px;}"
        "</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f"{render_html_body(content)}"
        "</body></html>"
    )


def render_html_body(content: str) -> str:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    html: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            html.append(f"<p>{format_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            html.append("<ul>" + "".join(f"<li>{format_inline(item)}</li>" for item in list_items) + "</ul>")
            list_items.clear()

    def flush_table() -> None:
        if not table_lines:
            return
        rows = [parse_markdown_table_row(row) for row in table_lines if not re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", row)]
        if rows:
            head, body = rows[0], rows[1:]
            html.append("<table><thead><tr>" + "".join(f"<th>{format_inline(cell)}</th>" for cell in head) + "</tr></thead><tbody>")
            for row in body:
                html.append("<tr>" + "".join(f"<td>{format_inline(cell)}</td>" for cell in row) + "</tr>")
            html.append("</tbody></table>")
        table_lines.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            flush_table()
            continue
        if line.startswith("|"):
            flush_paragraph()
            flush_list()
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("### "):
            flush_paragraph()
            flush_list()
            html.append(f"<h3>{format_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            flush_paragraph()
            flush_list()
            html.append(f"<h2>{format_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            flush_paragraph()
            flush_list()
            html.append(f"<h1>{format_inline(line[2:])}</h1>")
        elif line.startswith(("- ", "* ")):
            flush_paragraph()
            list_items.append(line[2:].strip())
        else:
            paragraph.append(line)
    flush_paragraph()
    flush_list()
    flush_table()
    return "\n".join(html)


def parse_markdown_table_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def format_inline(value: str) -> str:
    escaped = escape(value)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def markdown_to_plain_text(content: str) -> str:
    value = content.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"```[\s\S]*?```", "", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"__([^_]+)__", r"\1", value)
    value = re.sub(r"^#+\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s*[-*]\s+", "- ", value, flags=re.MULTILINE)
    return value.strip() + "\n"


def write_simple_pdf(path: Path, title: str, content: str) -> None:
    lines = [title, "", *content.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    pages: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        wrapped = wrap_pdf_line(line, 86) or [""]
        for part in wrapped:
            if len(current) >= 46:
                pages.append(current)
                current = []
            current.append(part)
    if current:
        pages.append(current)
    if not pages:
        pages = [["No content"]]

    objects: list[bytes] = []

    def add_object(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    content_ids: list[int] = []
    page_ids: list[int] = []
    for page in pages:
        stream = pdf_page_stream(page)
        content_ids.append(add_object(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"))
    for content_id in content_ids:
        page_ids.append(
            add_object(
                f"<< /Type /Page /Parent {{PAGES}} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
            )
        )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    fixed_objects = [body.replace(b"{PAGES}", str(pages_id).encode()) for body in objects]
    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for index, body in enumerate(fixed_objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(fixed_objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n \n".encode())
    chunks.append(
        f"trailer\n<< /Size {len(fixed_objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.write_bytes(b"".join(chunks))


def wrap_pdf_line(line: str, limit: int) -> list[str]:
    words = line.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def pdf_page_stream(lines: list[str]) -> bytes:
    body = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
    for line in lines:
        body.append(f"({escape_pdf_text(line)}) Tj")
        body.append("T*")
    body.append("ET")
    return "\n".join(body).encode("latin-1", errors="replace")


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
