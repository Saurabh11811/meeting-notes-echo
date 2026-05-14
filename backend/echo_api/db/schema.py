from __future__ import annotations

from sqlite3 import Connection


CURRENT_SCHEMA_VERSION = 2


def initialize_database(conn: Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        migrate_v1(conn)
        conn.execute("PRAGMA user_version = 1")
        version = 1
    if version < 2:
        migrate_v2(conn)
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")


def migrate_v1(conn: Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
          key TEXT PRIMARY KEY,
          value_json TEXT NOT NULL,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS backend_profiles (
          id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          name TEXT NOT NULL,
          config_json TEXT NOT NULL DEFAULT '{}',
          is_enabled INTEGER NOT NULL DEFAULT 0,
          is_default INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS templates (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL UNIQUE,
          meeting_type TEXT NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          sections_json TEXT NOT NULL DEFAULT '[]',
          is_default INTEGER NOT NULL DEFAULT 0,
          is_locked INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS meetings (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          meeting_type TEXT NOT NULL DEFAULT 'Executive',
          project TEXT NOT NULL DEFAULT '',
          host TEXT NOT NULL DEFAULT '',
          source_type TEXT NOT NULL DEFAULT 'unknown',
          source_label TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'draft',
          confidentiality TEXT NOT NULL DEFAULT 'Internal',
          tags_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_meetings_status ON meetings(status);
        CREATE INDEX IF NOT EXISTS idx_meetings_type ON meetings(meeting_type);
        CREATE INDEX IF NOT EXISTS idx_meetings_created ON meetings(created_at);

        CREATE TABLE IF NOT EXISTS transcripts (
          id TEXT PRIMARY KEY,
          meeting_id TEXT NOT NULL,
          text TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'unknown',
          language TEXT NOT NULL DEFAULT 'en',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS mom_versions (
          id TEXT PRIMARY KEY,
          meeting_id TEXT NOT NULL,
          transcript_id TEXT,
          version_number INTEGER NOT NULL DEFAULT 1,
          template_id TEXT,
          title TEXT NOT NULL,
          summary TEXT NOT NULL DEFAULT '',
          content_markdown TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'draft',
          backend_kind TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          approved_at TEXT,
          FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
          FOREIGN KEY(transcript_id) REFERENCES transcripts(id) ON DELETE SET NULL,
          FOREIGN KEY(template_id) REFERENCES templates(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_mom_versions_meeting ON mom_versions(meeting_id, version_number);

        CREATE TABLE IF NOT EXISTS queue_jobs (
          id TEXT PRIMARY KEY,
          meeting_id TEXT,
          job_type TEXT NOT NULL DEFAULT 'generate_mom',
          source_type TEXT NOT NULL,
          source_payload_json TEXT NOT NULL DEFAULT '{}',
          stage TEXT NOT NULL DEFAULT 'queued',
          progress INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'queued',
          error_code TEXT NOT NULL DEFAULT '',
          error_message TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_queue_jobs_status ON queue_jobs(status, created_at);

        CREATE TABLE IF NOT EXISTS action_items (
          id TEXT PRIMARY KEY,
          meeting_id TEXT NOT NULL,
          mom_version_id TEXT,
          description TEXT NOT NULL,
          owner TEXT NOT NULL DEFAULT '',
          due_date TEXT,
          status TEXT NOT NULL DEFAULT 'Open',
          project TEXT NOT NULL DEFAULT '',
          confidence REAL NOT NULL DEFAULT 0.0,
          source_ref TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
          FOREIGN KEY(mom_version_id) REFERENCES mom_versions(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status, due_date);

        CREATE TABLE IF NOT EXISTS decisions (
          id TEXT PRIMARY KEY,
          meeting_id TEXT NOT NULL,
          mom_version_id TEXT,
          description TEXT NOT NULL,
          context TEXT NOT NULL DEFAULT '',
          owner TEXT NOT NULL DEFAULT '',
          source_ref TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
          FOREIGN KEY(mom_version_id) REFERENCES mom_versions(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS risks (
          id TEXT PRIMARY KEY,
          meeting_id TEXT NOT NULL,
          mom_version_id TEXT,
          description TEXT NOT NULL,
          severity TEXT NOT NULL DEFAULT 'Medium',
          mitigation TEXT NOT NULL DEFAULT '',
          source_ref TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
          FOREIGN KEY(mom_version_id) REFERENCES mom_versions(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS exports (
          id TEXT PRIMARY KEY,
          meeting_id TEXT NOT NULL,
          mom_version_id TEXT,
          export_type TEXT NOT NULL,
          file_path TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
          FOREIGN KEY(mom_version_id) REFERENCES mom_versions(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS audit_events (
          id TEXT PRIMARY KEY,
          actor TEXT NOT NULL DEFAULT 'local-user',
          event_type TEXT NOT NULL,
          entity_type TEXT NOT NULL,
          entity_id TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def migrate_v2(conn: Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS queue_job_events (
          id TEXT PRIMARY KEY,
          job_id TEXT NOT NULL,
          stage TEXT NOT NULL,
          progress INTEGER NOT NULL DEFAULT 0,
          message TEXT NOT NULL,
          level TEXT NOT NULL DEFAULT 'info',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(job_id) REFERENCES queue_jobs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_queue_job_events_job_created
          ON queue_job_events(job_id, created_at);
        """
    )
