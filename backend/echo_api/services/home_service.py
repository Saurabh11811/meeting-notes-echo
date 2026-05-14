from __future__ import annotations

from echo_api.db.connection import db_session


def get_home_summary() -> dict:
    with db_session() as conn:
        counts = {
            "meetings_processed": conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0],
            "pending_review": conn.execute(
                "SELECT COUNT(*) FROM meetings WHERE status IN ('Ready for review', 'draft')"
            ).fetchone()[0],
            "queue_running": conn.execute(
                "SELECT COUNT(*) FROM queue_jobs WHERE status = 'running'"
            ).fetchone()[0],
            "action_items_open": conn.execute(
                "SELECT COUNT(*) FROM action_items WHERE status != 'Done'"
            ).fetchone()[0],
        }

        active_jobs = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, meeting_id, source_type, stage, progress, status, error_code,
                       error_message, created_at, updated_at,
                       json_extract(source_payload_json, '$.title') AS title,
                       json_extract(source_payload_json, '$.template_name') AS template_name
                FROM queue_jobs
                WHERE status IN ('queued', 'running', 'failed')
                ORDER BY updated_at DESC
                LIMIT 8
                """
            ).fetchall()
        ]
        for job in active_jobs:
            job["events"] = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT stage, progress, message, level, created_at
                    FROM queue_job_events
                    WHERE job_id = ?
                    ORDER BY created_at ASC
                    LIMIT 30
                    """,
                    (job["id"],),
                ).fetchall()
            ]

        recent_meetings = [
            dict(row)
            for row in conn.execute(
                """
                SELECT m.id, m.title, m.meeting_type, m.status, m.source_label, m.created_at, m.updated_at,
                       COALESCE((SELECT COUNT(*) FROM decisions d WHERE d.meeting_id = m.id), 0) AS decisions_count,
                       COALESCE((SELECT COUNT(*) FROM action_items a WHERE a.meeting_id = m.id), 0) AS action_items_count,
                       COALESCE((SELECT MAX(version_number) FROM mom_versions mv WHERE mv.meeting_id = m.id), 0) AS mom_version
                FROM meetings m
                ORDER BY m.updated_at DESC
                LIMIT 8
                """
            ).fetchall()
        ]

        action_items = [
            dict(row)
            for row in conn.execute(
                """
                SELECT a.id, a.description, a.owner, a.due_date, a.status, a.project,
                       m.title AS source_meeting
                FROM action_items a
                JOIN meetings m ON m.id = a.meeting_id
                WHERE a.status != 'Done'
                ORDER BY COALESCE(a.due_date, '9999-12-31') ASC
                LIMIT 6
                """
            ).fetchall()
        ]

    return {
        "counts": counts,
        "active_jobs": active_jobs,
        "ready_for_review": [
            meeting for meeting in recent_meetings if meeting["status"] in ("Ready for review", "draft")
        ],
        "recent_meetings": recent_meetings,
        "open_action_items": action_items,
    }
