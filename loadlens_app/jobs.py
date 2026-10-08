"""Job state for background tasks (report generation, Confluence publishing).

Jobs live in an in-memory dict for cheap polling and are written through to the
``report_jobs`` table, so a job can be found again after the browser tab was
closed or the process restarted, and the archive can list running/failed jobs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional

from psycopg2 import sql

from loadlens_app.core import _ts_conn
from settings import CONFIG

logger = logging.getLogger(__name__)

JOB_KIND_REPORT = "report"
JOB_KIND_CONFLUENCE = "confluence"
JOB_KIND_FORECAST_CONFLUENCE = "forecast_confluence"

JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_ERROR = "error"
JOB_STATUSES = (JOB_STATUS_RUNNING, JOB_STATUS_DONE, JOB_STATUS_ERROR)

DEFAULT_JOBS_TABLE = "report_jobs"

_JOB_COLUMNS = (
    "job_id", "kind", "run_name", "service", "status", "progress", "message",
    "error", "report_url", "page_url", "page_id", "created_at", "updated_at",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    """Snapshot of one background job."""

    job_id: str
    kind: str
    run_name: str = ""
    service: str = ""
    status: str = JOB_STATUS_RUNNING
    progress: int = 0
    message: str = ""
    error: Optional[str] = None
    report_url: Optional[str] = None
    page_url: Optional[str] = None
    page_id: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "run_name": self.run_name,
            "service": self.service,
            "status": self.status,
            "progress": int(self.progress),
            "message": self.message,
            "error": self.error,
            "report_url": self.report_url,
            "page_url": self.page_url,
            "page_id": self.page_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "JobRecord":
        values = dict(zip(_JOB_COLUMNS, row))
        return cls(
            job_id=str(values["job_id"]),
            kind=str(values["kind"] or ""),
            run_name=str(values["run_name"] or ""),
            service=str(values["service"] or ""),
            status=str(values["status"] or JOB_STATUS_RUNNING),
            progress=int(values["progress"] or 0),
            message=str(values["message"] or ""),
            error=values["error"],
            report_url=values["report_url"],
            page_url=values["page_url"],
            page_id=values["page_id"],
            created_at=values["created_at"] or _utcnow(),
            updated_at=values["updated_at"] or _utcnow(),
        )


def _jobs_table_ref() -> tuple[str, str]:
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return str(cfg.get("schema", "public")), str(cfg.get("jobs_table", DEFAULT_JOBS_TABLE))


class JobStore:
    """In-memory job registry with write-through persistence to TimescaleDB."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._table_ensured = False

    # ---- public API -----------------------------------------------------

    def create(self, kind: str, *, run_name: str = "", service: str = "", message: str = "") -> JobRecord:
        record = JobRecord(
            job_id=uuid.uuid4().hex,
            kind=kind,
            run_name=run_name,
            service=service,
            message=message,
        )
        with self._lock:
            self._jobs[record.job_id] = record
        self._persist(record)
        return record

    def update(self, job_id: str, **fields) -> Optional[JobRecord]:
        """Applies field changes; ``progress`` never decreases."""
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return None
            if "progress" in fields:
                incoming = fields["progress"]
                fields["progress"] = max(current.progress, int(incoming)) if incoming is not None else current.progress
            updated = replace(current, **fields, updated_at=_utcnow())
            self._jobs[job_id] = updated
        self._persist(updated)
        return updated

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            cached = self._jobs.get(job_id)
        if cached is not None:
            return cached
        return self._load_one(job_id)

    def list(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[JobRecord]:
        """Returns jobs with the given statuses, newest first.

        In-memory records win over persisted rows for the same job_id because
        they are always at least as fresh.
        """
        wanted = [s for s in (statuses or []) if s in JOB_STATUSES] or list(JOB_STATUSES)
        merged: Dict[str, JobRecord] = {}
        for record in self._load_many(wanted, limit):
            merged[record.job_id] = record
        with self._lock:
            for record in self._jobs.values():
                if record.status in wanted:
                    merged[record.job_id] = record
        ordered = sorted(merged.values(), key=lambda r: r.updated_at, reverse=True)
        return ordered[: max(1, int(limit))]

    # ---- persistence ----------------------------------------------------

    def _ensure_table(self, conn) -> None:
        if self._table_ensured:
            return
        schema, table = _jobs_table_ref()
        prev_autocommit = getattr(conn, "autocommit", False)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            job_id      TEXT PRIMARY KEY,
                            kind        TEXT NOT NULL,
                            run_name    TEXT,
                            service     TEXT,
                            status      TEXT NOT NULL,
                            progress    INTEGER NOT NULL DEFAULT 0,
                            message     TEXT,
                            error       TEXT,
                            report_url  TEXT,
                            page_url    TEXT,
                            page_id     TEXT,
                            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                        """
                    ).format(sql.Identifier(schema), sql.Identifier(table))
                )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (status, updated_at DESC);").format(
                        sql.Identifier(f"idx_{table}_status_updated"),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )
        finally:
            conn.autocommit = prev_autocommit
        self._table_ensured = True

    def _persist(self, record: JobRecord) -> None:
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.{} (
                                job_id, kind, run_name, service, status, progress, message,
                                error, report_url, page_url, page_id, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (job_id) DO UPDATE SET
                                run_name = EXCLUDED.run_name,
                                service = EXCLUDED.service,
                                status = EXCLUDED.status,
                                progress = EXCLUDED.progress,
                                message = EXCLUDED.message,
                                error = EXCLUDED.error,
                                report_url = EXCLUDED.report_url,
                                page_url = EXCLUDED.page_url,
                                page_id = EXCLUDED.page_id,
                                updated_at = EXCLUDED.updated_at
                            """
                        ).format(sql.Identifier(schema), sql.Identifier(table)),
                        (
                            record.job_id, record.kind, record.run_name, record.service, record.status,
                            int(record.progress), record.message, record.error, record.report_url,
                            record.page_url, record.page_id, record.created_at, record.updated_at,
                        ),
                    )
            finally:
                conn.close()
        except Exception:
            # The job itself keeps running; only the persisted status is stale.
            logger.exception("Failed to persist job %s (%s) to %s.%s", record.job_id, record.kind, schema, table)

    def _load_one(self, job_id: str) -> Optional[JobRecord]:
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT {} FROM {}.{} WHERE job_id = %s").format(
                            sql.SQL(", ").join(sql.Identifier(c) for c in _JOB_COLUMNS),
                            sql.Identifier(schema),
                            sql.Identifier(table),
                        ),
                        (job_id,),
                    )
                    row = cur.fetchone()
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to load job %s from %s.%s", job_id, schema, table)
            return None
        return JobRecord.from_row(row) if row else None

    def _load_many(self, statuses: List[str], limit: int) -> List[JobRecord]:
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "SELECT {} FROM {}.{} WHERE status = ANY(%s) ORDER BY updated_at DESC LIMIT %s"
                        ).format(
                            sql.SQL(", ").join(sql.Identifier(c) for c in _JOB_COLUMNS),
                            sql.Identifier(schema),
                            sql.Identifier(table),
                        ),
                        (list(statuses), max(1, int(limit))),
                    )
                    rows = cur.fetchall() or []
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to list jobs from %s.%s", schema, table)
            return []
        return [JobRecord.from_row(row) for row in rows]


job_store = JobStore()

__all__ = [
    "JOB_KIND_CONFLUENCE",
    "JOB_KIND_FORECAST_CONFLUENCE",
    "JOB_KIND_REPORT",
    "JOB_STATUS_DONE",
    "JOB_STATUS_ERROR",
    "JOB_STATUS_RUNNING",
    "JOB_STATUSES",
    "JobRecord",
    "JobStore",
    "job_store",
]
