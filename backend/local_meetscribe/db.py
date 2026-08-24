from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from local_meetscribe.config import Settings, ensure_runtime_dirs
from local_meetscribe.schemas import JobRecord, JobStatus


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        ensure_runtime_dirs(settings)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.settings.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    source_path TEXT,
                    output_dir TEXT,
                    transcript_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_job(
        self,
        job_id: str,
        source_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> JobRecord:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, status, stage, progress, source_path, output_dir,
                    transcript_path, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "queued",
                    "queued",
                    0.0,
                    str(source_path) if source_path else None,
                    str(output_dir) if output_dir else None,
                    None,
                    None,
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: str | None = None,
        progress: float | None = None,
        source_path: Path | None = None,
        output_dir: Path | None = None,
        transcript_path: Path | None = None,
        error: str | None = None,
    ) -> JobRecord:
        updates: dict[str, object] = {"updated_at": _now()}
        if status is not None:
            updates["status"] = status
        if stage is not None:
            updates["stage"] = stage
        if progress is not None:
            updates["progress"] = max(0.0, min(1.0, progress))
        if source_path is not None:
            updates["source_path"] = str(source_path)
        if output_dir is not None:
            updates["output_dir"] = str(output_dir)
        if transcript_path is not None:
            updates["transcript_path"] = str(transcript_path)
        if error is not None:
            updates["error"] = error

        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values())
        values.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_job(row)

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            status=row["status"],
            stage=row["stage"],
            progress=row["progress"],
            source_path=row["source_path"],
            output_dir=row["output_dir"],
            transcript_path=row["transcript_path"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
