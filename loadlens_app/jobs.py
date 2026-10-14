"""Job state for background tasks (report generation, Confluence publishing).

Jobs live in an in-memory dict for cheap polling and are written through to the
``report_jobs`` table, so a job can be found again after the browser tab was
closed or the process restarted, and the archive can list running/failed jobs.

Finished jobs can be dismissed: ``dismissed_at`` hides them from ``list`` while ``get`` still finds them.
Jobs run in threads of the web process, so a row that is still ``running`` from before this process
started belongs to a process that is gone; the first access to the table marks such rows as interrupted.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

import psycopg2
from psycopg2 import sql

from AI.storage_state import is_storage_error
from loadlens_app.core import _ts_conn
from settings import CONFIG

logger = logging.getLogger(__name__)


def _log_store_failure(exc: BaseException, message: str, *args: object) -> None:
    """A storage outage was already logged once by the breaker: one line here, no traceback per call."""
    if is_storage_error(exc):
        logger.warning(message + " (storage unavailable)", *args)
    else:
        logger.error(message, *args, exc_info=(type(exc), exc, exc.__traceback__))

JOB_KIND_REPORT = "report"
JOB_KIND_CONFLUENCE = "confluence"
JOB_KIND_FORECAST_CONFLUENCE = "forecast_confluence"

JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_ERROR = "error"
JOB_STATUSES = (JOB_STATUS_RUNNING, JOB_STATUS_DONE, JOB_STATUS_ERROR)

DEFAULT_JOBS_TABLE = "report_jobs"

# Shown for jobs that were running when the previous application process stopped.
INTERRUPTED_MESSAGE = "Задача прервана: сервер перезапустился во время выполнения"
INTERRUPTED_ERROR = "The application process stopped before the job finished"
# Finished jobs stay in memory this long; afterwards the table is the source of truth.
MEMORY_KEEP_FINISHED = timedelta(hours=1)

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
    # Who started the job; kept in memory only, for the per-user limit of running jobs.
    owner: str = ""
    # Hidden from the job list (in-memory copy of the dismissed_at column).
    dismissed: bool = False

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


_DISMISS_UNSUPPORTED = (
    "Очистка списка задач недоступна: в таблице задач нет колонки dismissed_at, а у приложения нет прав её добавить"
)


def _jobs_table_ref() -> tuple[str, str]:
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return str(cfg.get("schema", "public")), str(cfg.get("jobs_table", DEFAULT_JOBS_TABLE))


class JobStoreError(RuntimeError):
    """The job table cannot do what was asked; the message is safe to show to the user."""


class JobStore:
    """In-memory job registry with write-through persistence to TimescaleDB."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._table_ensured = False
        # Rows still "running" that were last updated before this moment belong to a stopped process.
        self._started_at = _utcnow()
        # False when the table has no dismissed_at column and this role cannot add it.
        self._dismiss_supported = True

    # ---- public API -----------------------------------------------------

    def create(self, kind: str, *, run_name: str = "", service: str = "", message: str = "", owner: str = "") -> JobRecord:
        record = JobRecord(
            job_id=uuid.uuid4().hex,
            kind=kind,
            run_name=run_name,
            service=service,
            message=message,
            owner=owner,
        )
        with self._lock:
            self._evict_finished_locked()
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

    def running_counts(self, owner: str) -> tuple[int, int]:
        """Jobs of this process that are still running: in total and those started by ``owner``."""
        with self._lock:
            running = [record for record in self._jobs.values() if record.status == JOB_STATUS_RUNNING]
        return len(running), sum(1 for record in running if record.owner == owner)

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
            self._evict_finished_locked()
            for record in self._jobs.values():
                if record.dismissed:
                    merged.pop(record.job_id, None)
                elif record.status in wanted:
                    merged[record.job_id] = record
        ordered = sorted(merged.values(), key=lambda r: r.updated_at, reverse=True)
        return ordered[: max(1, int(limit))]

    def dismiss(self, job_ids: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Hides finished jobs from ``list``; returns (dismissed ids, ids skipped because they still run).

        Rows stay in the table and ``get`` still finds them. Unknown and already hidden ids are ignored.
        Raises JobStoreError when the table cannot hold the flag, and psycopg2 errors when it is unreachable.
        """
        wanted = list(dict.fromkeys(str(job_id) for job_id in job_ids if job_id))
        if not wanted:
            return [], []
        with self._lock:
            memory = {job_id: self._jobs[job_id] for job_id in wanted if job_id in self._jobs}
        running_here = {job_id for job_id, record in memory.items() if record.status == JOB_STATUS_RUNNING}
        finished_here = {job_id for job_id, record in memory.items() if record.status != JOB_STATUS_RUNNING and not record.dismissed}
        candidates = [job_id for job_id in wanted if job_id not in running_here]
        hidden, running_in_table = self._dismiss_rows(candidates) if candidates else (set(), set())
        with self._lock:
            for job_id in finished_here:
                record = self._jobs.get(job_id)
                if record is not None:
                    self._jobs[job_id] = replace(record, dismissed=True)
        # The in-memory status is the fresher one when its last write to the table failed.
        dismissed = hidden | finished_here
        skipped = running_here | (running_in_table - finished_here)
        return [job_id for job_id in wanted if job_id in dismissed], [job_id for job_id in wanted if job_id in skipped]

    def _evict_finished_locked(self) -> None:
        """Drops finished jobs older than MEMORY_KEEP_FINISHED from memory; ``get`` reads them from the table."""
        cutoff = _utcnow() - MEMORY_KEEP_FINISHED
        stale = [job_id for job_id, record in self._jobs.items() if record.status != JOB_STATUS_RUNNING and record.updated_at < cutoff]
        for job_id in stale:
            del self._jobs[job_id]

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
                self._ensure_dismissed_column(cur, schema, table)
                self._mark_interrupted(cur, schema, table)
        finally:
            conn.autocommit = prev_autocommit
        self._table_ensured = True

    def _ensure_dismissed_column(self, cur, schema: str, table: str) -> None:
        # Checked first: ALTER TABLE needs ownership even when the column is already there.
        cur.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_schema = %s AND table_name = %s AND column_name = 'dismissed_at'",
            (schema, table),
        )
        if cur.fetchall():
            self._dismiss_supported = True
            return
        try:
            cur.execute(
                sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS dismissed_at TIMESTAMPTZ").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            self._dismiss_supported = True
        except psycopg2.Error:
            self._dismiss_supported = False
            logger.warning(
                "Cannot add dismissed_at to %s.%s (the table belongs to another role?): the job list cannot be cleared",
                schema, table, exc_info=True,
            )

    def _mark_interrupted(self, cur, schema: str, table: str) -> None:
        """Rows left "running" by a process that stopped become errors, so nothing polls them forever."""
        cur.execute(
            sql.SQL(
                "UPDATE {}.{} SET status = %s, message = %s, error = %s, updated_at = %s"
                " WHERE status = %s AND updated_at < %s RETURNING job_id"
            ).format(sql.Identifier(schema), sql.Identifier(table)),
            (JOB_STATUS_ERROR, INTERRUPTED_MESSAGE, INTERRUPTED_ERROR, _utcnow(), JOB_STATUS_RUNNING, self._started_at),
        )
        interrupted = [str(row[0]) for row in (cur.fetchall() or [])]
        if interrupted:
            logger.warning("Marked %d job(s) interrupted by a restart: %s", len(interrupted), ", ".join(interrupted[:20]))

    def _dismiss_rows(self, job_ids: List[str]) -> Tuple[Set[str], Set[str]]:
        """Sets dismissed_at on finished rows; returns (hidden ids, ids whose rows are still running)."""
        schema, table = _jobs_table_ref()
        if not self._dismiss_supported:
            raise JobStoreError(_DISMISS_UNSUPPORTED)
        conn = _ts_conn()
        try:
            self._ensure_table(conn)
            if not self._dismiss_supported:
                raise JobStoreError(_DISMISS_UNSUPPORTED)
            with conn, conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "UPDATE {}.{} SET dismissed_at = %s"
                        " WHERE job_id = ANY(%s) AND status <> %s AND dismissed_at IS NULL RETURNING job_id"
                    ).format(sql.Identifier(schema), sql.Identifier(table)),
                    (_utcnow(), list(job_ids), JOB_STATUS_RUNNING),
                )
                hidden = {str(row[0]) for row in (cur.fetchall() or [])}
                cur.execute(
                    sql.SQL("SELECT job_id FROM {}.{} WHERE job_id = ANY(%s) AND status = %s").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    ),
                    (list(job_ids), JOB_STATUS_RUNNING),
                )
                running = {str(row[0]) for row in (cur.fetchall() or [])}
        finally:
            conn.close()
        return hidden, running

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
        except Exception as exc:
            # The job itself keeps running; only the persisted status is stale.
            _log_store_failure(exc, "Failed to persist job %s (%s) to %s.%s", record.job_id, record.kind, schema, table)

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
        except Exception as exc:
            _log_store_failure(exc, "Failed to load job %s from %s.%s", job_id, schema, table)
            return None
        return JobRecord.from_row(row) if row else None

    def _load_many(self, statuses: List[str], limit: int) -> List[JobRecord]:
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn)
                # Without the dismissed_at column nothing can be hidden, and the filter would fail.
                visible = sql.SQL(" AND dismissed_at IS NULL") if self._dismiss_supported else sql.SQL("")
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "SELECT {} FROM {}.{} WHERE status = ANY(%s){} ORDER BY updated_at DESC LIMIT %s"
                        ).format(
                            sql.SQL(", ").join(sql.Identifier(c) for c in _JOB_COLUMNS),
                            sql.Identifier(schema),
                            sql.Identifier(table),
                            visible,
                        ),
                        (list(statuses), max(1, int(limit))),
                    )
                    rows = cur.fetchall() or []
            finally:
                conn.close()
        except Exception as exc:
            _log_store_failure(exc, "Failed to list jobs from %s.%s", schema, table)
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
    "INTERRUPTED_MESSAGE",
    "JobRecord",
    "JobStore",
    "JobStoreError",
    "job_store",
]
