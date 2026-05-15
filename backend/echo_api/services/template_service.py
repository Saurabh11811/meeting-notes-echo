from __future__ import annotations

import json
from datetime import datetime, timezone
from sqlite3 import Connection
from uuid import uuid4

from echo_api.core.config import load_app_config
from echo_api.db.connection import db_session


MEETING_TYPE_PRESETS: dict[str, dict[str, object]] = {
    "General": {
        "description": "Balanced notes for meetings that do not need a specialized structure.",
        "sections": ["Summary", "Decisions", "Action Items", "Open Questions", "Next Steps"],
    },
    "Executive": {
        "description": "Board-ready summary with decisions, risks, owners, and next steps.",
        "sections": ["Executive Summary", "Decisions", "Action Items", "Risks", "Full MoM"],
    },
    "Project Review": {
        "description": "Milestones, blockers, owners, dates, and delivery risks.",
        "sections": ["Summary", "Milestones", "Blockers", "Decisions", "Action Items", "Risks"],
    },
    "Client Call": {
        "description": "External-facing recap with commitments, decisions, and follow-ups.",
        "sections": ["Summary", "Client Commitments", "Decisions", "Action Items", "Follow-ups"],
    },
    "Townhall": {
        "description": "Key announcements, Q&A themes, and follow-ups.",
        "sections": ["Announcements", "Q&A Themes", "Employee Concerns", "Follow-ups"],
    },
    "Demo/UAT": {
        "description": "Acceptance criteria, defects, open questions, and sign-off readiness.",
        "sections": ["Demo Summary", "Acceptance Criteria", "Defects", "Open Questions", "Action Items"],
    },
    "Incident": {
        "description": "Timeline, root cause, customer impact, mitigations, and follow-up actions.",
        "sections": ["Timeline", "Root Cause", "Impact", "Mitigations", "Action Items"],
    },
}


def list_templates() -> list[dict]:
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT id, name, meeting_type, description, sections_json, system_prompt,
                   is_default, is_locked, created_at, updated_at
            FROM templates
            ORDER BY is_default DESC, meeting_type ASC, name ASC
            """
        ).fetchall()
        return [normalize_template(row) for row in rows]


def get_template(template_id: str) -> dict | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
        return normalize_template(row) if row else None


def get_template_by_name(template_name: str) -> dict | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM templates WHERE name = ?", (template_name,)).fetchone()
        return normalize_template(row) if row else None


def get_default_template_for_type(meeting_type: str) -> dict | None:
    normalized_type = (meeting_type or "").strip()
    with db_session() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM templates
            WHERE meeting_type = ?
            ORDER BY is_default DESC, is_locked DESC, name ASC
            LIMIT 1
            """,
            (normalized_type,),
        ).fetchone()
        if not row:
            row = conn.execute(
                """
                SELECT *
                FROM templates
                ORDER BY is_default DESC, is_locked DESC, name ASC
                LIMIT 1
                """
            ).fetchone()
        return normalize_template(row) if row else None


def create_template(data: dict) -> dict:
    template_id = str(uuid4())
    now = now_iso()
    sections = data.get("sections") or []
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO templates (
              id, name, meeting_type, description, sections_json, system_prompt,
              is_default, is_locked, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                data["name"],
                data.get("meeting_type") or "General",
                data.get("description") or "",
                json.dumps(sections),
                data.get("system_prompt") or "",
                1 if data.get("is_default") else 0,
                0,
                now,
                now,
            ),
        )
        if data.get("is_default"):
            conn.execute("UPDATE templates SET is_default = 0 WHERE id != ?", (template_id,))
        record_template_version(conn, template_id, "Template created")
    return get_template(template_id)


def update_template(template_id: str, data: dict) -> dict | None:
    with db_session() as conn:
        existing = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
        if not existing:
            return None

        protected_when_locked = {"name"}
        if existing["is_locked"]:
            data = {key: value for key, value in data.items() if key not in protected_when_locked}

        fields: list[str] = []
        params: list[object] = []
        field_map = {
            "name": "name",
            "meeting_type": "meeting_type",
            "description": "description",
            "system_prompt": "system_prompt",
            "is_default": "is_default",
        }
        for key, column in field_map.items():
            if key in data:
                value = data[key]
                if key == "is_default":
                    value = 1 if value else 0
                fields.append(f"{column} = ?")
                params.append(value)

        if "sections" in data:
            fields.append("sections_json = ?")
            params.append(json.dumps(data.get("sections") or []))

        if not fields:
            return get_template(template_id)

        now = now_iso()
        params.extend([now, template_id])
        conn.execute(
            f"UPDATE templates SET {', '.join(fields)}, updated_at = ? WHERE id = ?",
            tuple(params),
        )

        if data.get("is_default"):
            conn.execute("UPDATE templates SET is_default = 0 WHERE id != ?", (template_id,))
        record_template_version(conn, template_id, "Template updated")
    return get_template(template_id)


def duplicate_template(template_id: str) -> dict | None:
    with db_session() as conn:
        existing = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
        if not existing:
            return None

        new_id = str(uuid4())
        now = now_iso()
        new_name = unique_copy_name(conn, existing["name"])
        conn.execute(
            """
            INSERT INTO templates (
              id, name, meeting_type, description, sections_json, system_prompt,
              is_default, is_locked, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                new_name,
                existing["meeting_type"],
                existing["description"],
                existing["sections_json"],
                existing["system_prompt"] if "system_prompt" in existing.keys() else "",
                0,
                0,
                now,
                now,
            ),
        )
        record_template_version(conn, new_id, f"Duplicated from {existing['name']}")
    return get_template(new_id)


def delete_template(template_id: str) -> bool:
    with db_session() as conn:
        existing = conn.execute("SELECT is_locked FROM templates WHERE id = ?", (template_id,)).fetchone()
        if not existing or existing["is_locked"]:
            return False
        conn.execute("DELETE FROM templates WHERE id = ?", (template_id,))
        return True


def get_template_prompt(template_id: str) -> dict | None:
    template = get_template(template_id)
    if not template:
        return None
    from echo_api.services.processing_service import template_prompt_instruction

    config = load_app_config()
    prompt = template_prompt_instruction(template["name"], config)
    return {
        "template_id": template_id,
        "template_name": template["name"],
        "prompt": prompt,
        "effective_prompt": prompt,
    }


def list_template_versions(template_id: str) -> list[dict]:
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT id, template_id, version_number, name, meeting_type, description,
                   sections_json, system_prompt, change_note, created_at
            FROM template_versions
            WHERE template_id = ?
            ORDER BY version_number DESC
            """,
            (template_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["sections"] = json.loads(item.pop("sections_json") or "[]")
        result.append(item)
    return result


def restore_template_version(template_id: str, version_id: str) -> dict | None:
    with db_session() as conn:
        template = conn.execute("SELECT is_locked FROM templates WHERE id = ?", (template_id,)).fetchone()
        version = conn.execute(
            """
            SELECT *
            FROM template_versions
            WHERE id = ? AND template_id = ?
            """,
            (version_id, template_id),
        ).fetchone()
        if not template or not version:
            return None
        if template["is_locked"]:
            conn.execute(
                """
                UPDATE templates
                SET meeting_type = ?, description = ?, sections_json = ?,
                    system_prompt = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    version["meeting_type"],
                    version["description"],
                    version["sections_json"],
                    version["system_prompt"],
                    now_iso(),
                    template_id,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE templates
                SET name = ?, meeting_type = ?, description = ?, sections_json = ?,
                    system_prompt = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    version["name"],
                    version["meeting_type"],
                    version["description"],
                    version["sections_json"],
                    version["system_prompt"],
                    now_iso(),
                    template_id,
                ),
            )
        record_template_version(conn, template_id, f"Restored from v{version['version_number']}")
    return get_template(template_id)


def list_template_presets() -> list[dict]:
    return [
        {
            "meeting_type": meeting_type,
            "description": str(preset["description"]),
            "sections": list(preset["sections"]),
            "system_prompt": build_template_prompt(
                name=f"{meeting_type} MoM",
                description=str(preset["description"]),
                sections=list(preset["sections"]),
            ),
        }
        for meeting_type, preset in MEETING_TYPE_PRESETS.items()
    ]


def build_template_prompt(*, name: str, description: str, sections: list[str]) -> str:
    section_lines = "\n".join(f"- {section}" for section in sections)
    return (
        "You are a professional scribe. Produce detailed, structured, and comprehensive Minutes of Meeting (MoM) from the provided transcript.\n"
        "The MoM must be clear enough for someone who did not attend to understand the discussion, outcomes, decisions, and next steps.\n\n"
        f"Template: {name}\n"
        f"Purpose: {description or 'Create clear, evidence-based meeting notes.'}\n\n"
        "Write in professional Markdown with clear headings and tables where they improve scanability.\n"
        "Use only facts from the transcript. If evidence is missing, write 'Not captured in transcript'.\n"
        "Capture speakers, owners, dates, deadlines, dependencies, risks, decisions, and exact values whenever they appear.\n"
        "Do not invent commitments, attendees, dates, risks, or decisions.\n\n"
        "Required sections:\n"
        f"{section_lines or '- Summary\n- Decisions\n- Action Items\n- Risks\n- Next Steps'}"
    )


def normalize_template(row) -> dict:
    item = dict(row)
    item["sections"] = json.loads(item.pop("sections_json") or "[]")
    item["system_prompt"] = item.get("system_prompt") or ""
    item["is_default"] = bool(item["is_default"])
    item["is_locked"] = bool(item["is_locked"])
    return item


def record_template_version(conn: Connection, template_id: str, change_note: str) -> None:
    template = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
    if not template:
        return
    version_number = int(
        conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM template_versions WHERE template_id = ?",
            (template_id,),
        ).fetchone()[0]
    )
    conn.execute(
        """
        INSERT INTO template_versions (
          id, template_id, version_number, name, meeting_type, description,
          sections_json, system_prompt, change_note, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            template_id,
            version_number,
            template["name"],
            template["meeting_type"],
            template["description"],
            template["sections_json"],
            template["system_prompt"] if "system_prompt" in template.keys() else "",
            change_note,
            now_iso(),
        ),
    )


def unique_copy_name(conn: Connection, base_name: str) -> str:
    new_name = f"{base_name} (Copy)"
    counter = 1
    while conn.execute("SELECT 1 FROM templates WHERE name = ?", (new_name,)).fetchone():
        new_name = f"{base_name} (Copy {counter})"
        counter += 1
    return new_name


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
