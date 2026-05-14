from __future__ import annotations

from echo_api.db.connection import db_session


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

    return {
        "meeting": dict(meeting),
        "latest_mom": mom_versions[0] if mom_versions else None,
        "mom_versions": mom_versions,
        "transcript": dict(transcript) if transcript else None,
        "decisions": decisions,
        "action_items": action_items,
        "risks": risks,
        "jobs": jobs,
    }
