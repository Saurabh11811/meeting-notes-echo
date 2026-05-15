from __future__ import annotations

import json
from sqlite3 import Connection
from uuid import uuid4

from echo_api.core.config import load_app_config
from echo_api.services.template_service import build_template_prompt


def seed_reference_data(conn: Connection) -> None:
    seed_templates(conn)
    seed_backend_profiles(conn)


def seed_templates(conn: Connection) -> None:
    existing = conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
    if existing:
        return
    templates = load_app_config().get("templates", {}).get("defaults", [])
    for template in templates:
        conn.execute(
            """
            INSERT INTO templates (
              id, name, meeting_type, description, sections_json, system_prompt, is_default, is_locked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                template["name"],
                template["meeting_type"],
                template.get("description", ""),
                json.dumps(template.get("sections", [])),
                template.get("system_prompt", "") or build_template_prompt(
                    name=template["name"],
                    description=template.get("description", ""),
                    sections=template.get("sections", []),
                ),
                1 if template.get("is_default") else 0,
                1 if template.get("is_locked") else 0,
            ),
        )


def seed_backend_profiles(conn: Connection) -> None:
    existing = conn.execute("SELECT COUNT(*) FROM backend_profiles").fetchone()[0]
    if existing:
        return
    backends = load_app_config().get("backends", {})
    default_backend = load_app_config().get("summary", {}).get("default_backend", "dify")
    for kind, config in backends.items():
        conn.execute(
            """
            INSERT INTO backend_profiles (
              id, kind, name, config_json, is_enabled, is_default
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                kind,
                config.get("name", kind.title()),
                json.dumps(config),
                1 if config.get("enabled") else 0,
                1 if kind == default_backend else 0,
            ),
        )
