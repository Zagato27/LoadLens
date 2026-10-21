# SPDX-License-Identifier: Apache-2.0
"""Job state for background tasks (report generation, Confluence publishing).

The ``report_jobs`` table is the source of truth. A job runs either in a thread of the web process
(``runner="thread"``) or in a Celery worker (``runner="celery"``). Thread jobs are also kept in an in-memory
dict of the process running them: cheap polling, and their state survives a failed write to the table.
Celery jobs live only in the table: the worker changes the row through :meth:`JobStore.update`, which works
from any process, and the web process reads it.

Finished jobs can be dismissed: ``dismissed_at`` hides them from ``list`` while ``get`` still finds them.

Jobs that stopped without finishing:

- a thread job dies with its process, so a ``thread`` row still ``running`` from before this process started
  is marked interrupted on the first access of the web process (the Celery worker never does it);
- a Celery job waits in the broker queue first (``started_at`` is empty), then the worker calls
  :meth:`JobStore.mark_started` and refreshes ``updated_at`` every 30 s (:meth:`JobStore.heartbeat`).
  :meth:`JobStore.sweep_stale` turns a started ``celery`` row silent for longer than ``LOADLENS_JOB_STALE_MIN``
  minutes, or one queued for longer than ``LOADLENS_JOB_QUEUE_MAX_MIN`` minutes, into an error; ``list`` and
  ``get`` run it at most every 30 s. The job limits count running rows by the same two rules.

When the role cannot add the ``owner``/``runner``/``started_at`` columns to an older table, the store works as
before them: the job limits count the jobs of this process only, and nothing is swept.

The job list is shared by users with different UI languages, so a message is stored twice: ``message`` holds the
text in the language of the job, ``message_ref`` (JSONB) the dictionary key with its parameters
(:meth:`AI.i18n.Msg.ref`); readers render the ref in their own language (:func:`message_text`). A message without
a key (plain text) stores no ref. Without the ``message_ref`` column (an older table the role may not ALTER)
only the text is kept, as before.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import psycopg2
from psycopg2 import sql

from AI.i18n import LocalizedError, Msg, is_ref, msg, render_ref, t
from AI.storage_state import is_storage_error
from loadlens_app import response_cache
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

# Where a job runs: a thread of the web process or a Celery worker.
JOB_RUNNER_THREAD = "thread"
JOB_RUNNER_CELERY = "celery"
JOB_RUNNERS = (JOB_RUNNER_THREAD, JOB_RUNNER_CELERY)

DEFAULT_JOBS_TABLE = "report_jobs"

# Message keys (locales/<lang>/job.json); INTERRUPTED_MESSAGE, STALE_MESSAGE and QUEUED_MESSAGE give their texts.
# Shown for jobs that were running when the previous application process stopped.
INTERRUPTED_KEY = "job.interrupted"
INTERRUPTED_ERROR = "The application process stopped before the job finished"
# Shown for Celery jobs whose worker went silent (killed, out of memory, host lost).
STALE_KEY = "job.stale"
STALE_ERROR = "No progress or heartbeat from the Celery worker for {minutes:g} min; the job was abandoned"
# Shown for Celery jobs that no worker took from the queue (no worker running, queue blocked).
QUEUED_KEY = "job.queued_too_long"
QUEUED_ERROR = "No Celery worker took the job from the queue within {minutes:g} min; the job was abandoned"
_MESSAGE_CONSTANTS = {"INTERRUPTED_MESSAGE": INTERRUPTED_KEY, "STALE_MESSAGE": STALE_KEY, "QUEUED_MESSAGE": QUEUED_KEY}
# A started job without progress or heartbeat for this many minutes counts as stuck.
STALE_MIN_ENV = "LOADLENS_JOB_STALE_MIN"
DEFAULT_STALE_MIN = 15.0
# A Celery job not started by a worker within this many minutes counts as stuck.
QUEUE_MAX_MIN_ENV = "LOADLENS_JOB_QUEUE_MAX_MIN"
DEFAULT_QUEUE_MAX_MIN = 120.0
MAX_ENV_MINUTES = 7 * 24 * 60.0
# list() and get() sweep stuck Celery jobs at most this often in one process.
SWEEP_INTERVAL_SEC = 30.0
# Finished jobs stay in memory this long; afterwards the table is the source of truth.
MEMORY_KEEP_FINISHED = timedelta(hours=1)

_BASE_COLUMNS = (
    "job_id", "kind", "run_name", "service", "status", "progress", "message",
    "error", "report_url", "page_url", "page_id", "created_at", "updated_at",
)
# Added after the table: an older table gets them through ALTER TABLE when the role may do it.
_TRACKING_COLUMNS = ("owner", "runner", "started_at")
_JOB_COLUMNS = _BASE_COLUMNS + _TRACKING_COLUMNS
# The dictionary ref of ``message`` (JSONB); read and written only when the table has it, always last.
_REF_COLUMN = "message_ref"
_OPTIONAL_COLUMNS = ("dismissed_at",) + _TRACKING_COLUMNS + (_REF_COLUMN,)
# What update() may change in the row of a job that is not in memory.
_UPDATABLE_COLUMNS = frozenset(
    ("run_name", "service", "status", "progress", "message", _REF_COLUMN, "error", "report_url", "page_url", "page_id")
)
# The upsert of a known job keeps these as first written.
_INSERT_ONLY_COLUMNS = frozenset(("job_id", "kind", "created_at"))

_monotonic = time.monotonic


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _env_minutes(name: str, default: float) -> timedelta:
    """Minutes from the environment, read on every call; missing, unparsable or non-positive gives ``default``."""
    raw = (os.getenv(name) or "").strip()
    try:
        minutes = float(raw) if raw else default
    except ValueError:
        minutes = default
    if not math.isfinite(minutes) or minutes <= 0:
        minutes = default
    return timedelta(minutes=min(minutes, MAX_ENV_MINUTES))


def stale_after() -> timedelta:
    """How long a started job may stay silent before it counts as stuck: ``LOADLENS_JOB_STALE_MIN`` (15) minutes."""
    return _env_minutes(STALE_MIN_ENV, DEFAULT_STALE_MIN)


def queue_max_wait() -> timedelta:
    """How long a Celery job may wait for a worker before it counts as stuck: ``LOADLENS_JOB_QUEUE_MAX_MIN`` (120)."""
    return _env_minutes(QUEUE_MAX_MIN_ENV, DEFAULT_QUEUE_MAX_MIN)


def _as_timedelta(value: Union[timedelta, float, None], default: Callable[[], timedelta]) -> timedelta:
    if value is None:
        return default()
    if isinstance(value, timedelta):
        return value
    return timedelta(seconds=float(value))


def _invalidate_caches(previous_status: Optional[str], status: Optional[str]) -> None:
    response_cache.invalidate(response_cache.JOBS)
    if status != previous_status and status in (JOB_STATUS_DONE, JOB_STATUS_ERROR):
        # A finished report job has saved (or failed to save) its rows: the dashboard may have changed.
        response_cache.invalidate(response_cache.DASHBOARD)


# ---- messages ---------------------------------------------------------------------------------------------


def stored_locale() -> str:
    """Language of a text written outside a job (the sweeps, the first message of a new job): the installation
    setting ``appearance.language``. Inside a job, messages are rendered in the job's language (``tasks._run_job``)."""
    from loadlens_app.i18n import report_language

    return report_language()


def job_message(key: str, /, *, locale: Optional[str] = None, **params: Any) -> Msg:
    """:class:`~AI.i18n.Msg` of ``key`` rendered in ``locale`` (default :func:`stored_locale`), for a job row."""
    return Msg(t(key, locale=locale or stored_locale(), **params), key, params)


def _ref(value: Any) -> Optional[Dict[str, Any]]:
    """A stored ``{"key", "params"}`` (JSON text when the driver does not decode jsonb); anything else is no ref."""
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    if isinstance(value, str) and value:
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return dict(value) if is_ref(value) else None


def _ref_json(ref: Optional[Dict[str, Any]]) -> Optional[str]:
    return json.dumps(ref, ensure_ascii=False) if ref else None


def _placeholder(column: str) -> sql.Composable:
    return sql.SQL("%s::jsonb") if column == _REF_COLUMN else sql.Placeholder()


def _column_value(record: "JobRecord", column: str) -> Any:
    if column == "progress":
        return int(record.progress)
    if column == _REF_COLUMN:
        return _ref_json(record.message_ref)
    return getattr(record, column)


def message_fields(message: Any) -> Dict[str, Any]:
    """``message`` and ``message_ref`` of a job row: a :class:`~AI.i18n.Msg` or a localized error keeps its ref,
    plain text has none (so it also clears the ref of the previous message)."""
    if isinstance(message, (Msg, LocalizedError)) and message.key:
        return {"message": str(message), _REF_COLUMN: message.ref()}
    return {"message": "" if message is None else str(message), _REF_COLUMN: None}


def message_text(record: "JobRecord", locale: Optional[str] = None) -> str:
    """The message of a job in ``locale`` (default: the current one): its ref rendered again, else the stored text."""
    text = record.message or ""
    ref = getattr(record, "message_ref", None)
    return render_ref(ref, fallback=text, locale=locale) if ref else text


def failure_message(record: Optional["JobRecord"], default_key: str) -> Msg:
    """Message of a failed job in the current locale: ``job.failed_at_stage`` naming the stage the job was in
    (as its ref when there is one, so a reader sees the stage in their language too), else ``default_key``."""
    if record is not None:
        stage: Any = getattr(record, "message_ref", None) or (record.message or "").strip()
        if stage:
            return msg("job.failed_at_stage", stage=stage)
    return msg(default_key)


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
    # Who started the job, for the per-user limit of running jobs.
    owner: str = ""
    # Where the job runs (JOB_RUNNERS); only thread jobs are kept in memory, by the process running them.
    runner: str = JOB_RUNNER_THREAD
    # When a worker began the job; empty while a Celery job waits in the queue.
    started_at: Optional[datetime] = None
    # Hidden from the job list (in-memory copy of the dismissed_at column).
    dismissed: bool = False
    # Dictionary ref of ``message`` ({"key", "params"}), rendered in the reader's language; None for plain text.
    message_ref: Optional[Dict[str, Any]] = None

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
    def from_row(cls, row: tuple, columns: Optional[Sequence[str]] = None) -> "JobRecord":
        """A row in ``columns`` order (default ``_JOB_COLUMNS``); missing optional columns get their defaults."""
        values = dict(zip(columns or _JOB_COLUMNS, row))
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
            owner=str(values.get("owner") or ""),
            runner=str(values.get("runner") or JOB_RUNNER_THREAD),
            started_at=values.get("started_at"),
            message_ref=_ref(values.get(_REF_COLUMN)),
        )


def _abandoned(record: JobRecord) -> bool:
    """A Celery job still running in the table that sweep_stale() would end: queued or silent for too long."""
    if record.status != JOB_STATUS_RUNNING or record.runner != JOB_RUNNER_CELERY:
        return False
    if record.started_at is None:
        return record.created_at < _utcnow() - queue_max_wait()
    return record.updated_at < _utcnow() - stale_after()


_DISMISS_UNSUPPORTED = "job.dismiss_unsupported"


def _jobs_table_ref() -> tuple[str, str]:
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return str(cfg.get("schema", "public")), str(cfg.get("jobs_table", DEFAULT_JOBS_TABLE))


class JobStoreError(LocalizedError, RuntimeError):
    """The job table cannot do what was asked; the message (a dictionary key) is safe to show to the user."""


class JobStore:
    """Job registry backed by the ``report_jobs`` table; thread jobs of this process are also kept in memory.

    Every method may be called from any thread. A Celery worker calls update(), mark_started(), heartbeat()
    and get(); the rest serve the web process, and only they run the restart recovery of thread rows. Only
    ``dismiss`` raises when the table is unreachable; the other methods log the failure and fall back to memory.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._table_ensured = False
        # Whether the restart recovery of thread rows ran in this process (see _ensure_table).
        self._recovered = False
        # Thread rows still "running" that were last updated before this moment belong to a stopped process.
        self._started_at = _utcnow()
        # False when the table has no dismissed_at column and this role cannot add it.
        self._dismiss_supported = True
        # False when the table has no owner/runner/started_at columns and this role cannot add them.
        self._columns_supported = True
        # False when the table has no message_ref column and this role cannot add it: only texts are stored.
        self._refs_supported = True
        self._untracked_runner_logged = False
        # The first sweep waits one interval: the start-up recovery has just run, and a Celery row has to stay
        # silent for stale_after() before it is swept anyway.
        self._last_sweep = _monotonic()

    # ---- public API -----------------------------------------------------

    def create(
        self,
        kind: str,
        *,
        run_name: str = "",
        service: str = "",
        message: Any = "",
        owner: str = "",
        runner: str = JOB_RUNNER_THREAD,
    ) -> JobRecord:
        """Registers a running job and writes its row.

        ``runner="thread"``: the job starts now in a thread of this process and is kept in memory as well.
        ``runner="celery"``: only the row exists, queued until the worker calls mark_started(); when that write
        fails the job cannot be followed (logged). ``message`` may be a :class:`~AI.i18n.Msg` (its ref is kept).
        """
        if runner not in JOB_RUNNERS:
            raise ValueError(f"Unknown job runner {runner!r}; expected one of: {', '.join(JOB_RUNNERS)}")
        now = _utcnow()
        text = message_fields(message)
        record = JobRecord(
            job_id=uuid.uuid4().hex,
            kind=kind,
            run_name=run_name,
            service=service,
            message=text["message"],
            message_ref=text[_REF_COLUMN],
            created_at=now,
            updated_at=now,
            owner=owner,
            runner=runner,
            started_at=now if runner == JOB_RUNNER_THREAD else None,
        )
        with self._lock:
            self._evict_finished_locked()
            if runner == JOB_RUNNER_THREAD:
                self._jobs[record.job_id] = record
        self._persist(record)
        if runner != JOB_RUNNER_THREAD and not self._columns_supported:
            self._warn_untracked_runner()
        response_cache.invalidate(response_cache.JOBS)
        return record

    def update(self, job_id: str, **fields) -> Optional[JobRecord]:
        """Applies field changes; ``progress`` never decreases.

        A thread job of this process changes in memory and is written through to the table. Any other job
        (a Celery job, changed by the worker or by the web process) is changed in the table by one UPDATE and
        is not kept in memory; fields other than the row's data columns raise TypeError there. Returns the new
        state, or ``None`` when the job is unknown or the table cannot be reached (logged).

        ``message`` may be a :class:`~AI.i18n.Msg` or a localized error: its ref goes to ``message_ref`` unless that
        is given too; a plain text message clears the ref (see :func:`message_fields`).
        """
        if "message" in fields:
            if _REF_COLUMN in fields:
                fields["message"] = "" if fields["message"] is None else str(fields["message"])
            else:
                fields.update(message_fields(fields["message"]))
        if _REF_COLUMN in fields:
            fields[_REF_COLUMN] = _ref(fields[_REF_COLUMN])
        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                if "progress" in fields:
                    incoming = fields["progress"]
                    fields["progress"] = max(current.progress, int(incoming)) if incoming is not None else current.progress
                updated = replace(current, **fields, updated_at=_utcnow())
                self._jobs[job_id] = updated
        if current is None:
            return self._update_row(job_id, fields)
        self._persist(updated)
        _invalidate_caches(current.status, updated.status)
        return updated

    def adopt_thread(self, job_id: str) -> Optional[JobRecord]:
        """Makes the job a thread job of this process: ``runner="thread"`` in its row and a copy in memory.

        For the fallback when a job created for the Celery worker could not be sent to the broker and runs in
        a thread here instead. Returns the record, or ``None`` when the job is neither in memory nor in the
        table (or the table cannot be reached).
        """
        with self._lock:
            current = self._jobs.get(job_id)
        if current is not None and current.runner == JOB_RUNNER_THREAD:
            return current
        if current is None:
            current = self._load_one(job_id, recover=True)
            if current is None:
                return None
        now = _utcnow()
        adopted = replace(current, runner=JOB_RUNNER_THREAD, started_at=current.started_at or now, updated_at=now)
        with self._lock:
            self._jobs[job_id] = adopted
        self._persist(adopted)
        response_cache.invalidate(response_cache.JOBS)
        return adopted

    def mark_started(self, job_id: str) -> Optional[bool]:
        """Records that a worker began the job: ``started_at`` and ``updated_at`` become now.

        The task calls it first thing, then heartbeat() every 30 s. Until then a Celery job counts as queued:
        sweep_stale() gives it LOADLENS_JOB_QUEUE_MAX_MIN instead of LOADLENS_JOB_STALE_MIN. Returns True when
        the job was marked; False when it is not running (finished, swept while queued, unknown): the task must
        not run it; None when the table cannot be reached (logged): run it anyway.
        """
        now = _utcnow()
        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None and current.status == JOB_STATUS_RUNNING:
                current = self._jobs[job_id] = replace(current, started_at=now, updated_at=now)
        if current is not None:
            if current.status != JOB_STATUS_RUNNING:
                return False
            self._persist(current)
            return True
        return self._touch_running(job_id, now, started=True)

    def heartbeat(self, job_id: str) -> Optional[bool]:
        """Marks a running job alive: sets only ``updated_at`` of its row to now.

        Called every 30 s by the task wrapper, from its own thread in whatever process runs the job, so a long
        step without progress (an LLM call) is not taken for a dead worker by sweep_stale() and the job limits.
        The in-memory copy is left alone. Returns True when the row was touched, False when it is not running
        (finished, swept, unknown), None when the table cannot be reached (logged); never raises for those.
        """
        return self._touch_running(job_id, _utcnow(), started=False)

    def running_counts(self, owner: str) -> tuple[int, int]:
        """Running jobs in total and those started by ``owner``, for the job limits.

        Counts the ``running`` rows that sweep_stale() would leave alone (jobs of every process and of the
        Celery workers, including jobs still waiting in the queue: they hold a slot) together with the running
        thread jobs in memory; a job found in both counts once, by its in-memory state. Without the table
        (unreachable, or no owner/runner/started_at columns) only memory counts. Costs one query on the
        ``running`` prefix of the (status, updated_at) index.
        """
        rows = self._running_owners()
        with self._lock:
            memory = dict(self._jobs)
        owners = {job_id: row_owner for job_id, row_owner in rows.items() if job_id not in memory}
        owners.update((record.job_id, record.owner) for record in memory.values() if record.status == JOB_STATUS_RUNNING)
        return len(owners), sum(1 for job_owner in owners.values() if job_owner == owner)

    def sweep_stale(
        self,
        max_age: Union[timedelta, float, None] = None,
        *,
        queue_max_age: Union[timedelta, float, None] = None,
    ) -> List[str]:
        """Turns stuck Celery jobs into errors and returns their ids.

        A started job is stuck when its row was not updated for ``max_age`` (default stale_after()): the worker
        died (killed, out of memory, host lost), while heartbeats keep the rows of live jobs fresh. A job that no
        worker started is stuck once it was created ``queue_max_age`` ago (default queue_max_wait()). Thread jobs
        are left alone. Ages are timedeltas or seconds. Returns [] when the table cannot be reached (logged) or
        has no runner/started_at columns.
        """
        age = _as_timedelta(max_age, stale_after)
        queue_age = _as_timedelta(queue_max_age, queue_max_wait)
        now = _utcnow()
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn)
                if not self._columns_supported:
                    return []
                with conn, conn.cursor() as cur:
                    silent = self._fail_running(
                        cur, schema, table,
                        sql.SQL(" AND runner = %s AND started_at IS NOT NULL AND updated_at < %s"),
                        [JOB_RUNNER_CELERY, now - age],
                        message_key=STALE_KEY,
                        error=STALE_ERROR.format(minutes=round(age.total_seconds() / 60, 1)),
                    )
                    queued = self._fail_running(
                        cur, schema, table,
                        sql.SQL(" AND runner = %s AND started_at IS NULL AND created_at < %s"),
                        [JOB_RUNNER_CELERY, now - queue_age],
                        message_key=QUEUED_KEY,
                        error=QUEUED_ERROR.format(minutes=round(queue_age.total_seconds() / 60, 1)),
                    )
            finally:
                conn.close()
        except Exception as exc:
            _log_store_failure(exc, "Failed to sweep stale jobs in %s.%s", schema, table)
            return []
        if silent:
            logger.warning("Marked %d Celery job(s) abandoned by their worker: %s", len(silent), ", ".join(silent[:20]))
        if queued:
            logger.warning("Marked %d Celery job(s) never taken from the queue: %s", len(queued), ", ".join(queued[:20]))
        swept = silent + queued
        if swept:
            _invalidate_caches(None, JOB_STATUS_ERROR)
        return swept

    def get(self, job_id: str) -> Optional[JobRecord]:
        """The job from memory (thread jobs of this process) or from the table.

        A page may poll only this job, so the row is checked like ``list`` would: a running thread job that is
        not in memory runs the restart recovery (once per process), and a Celery job past its limits triggers
        the sweep (at most every 30 s).
        """
        with self._lock:
            cached = self._jobs.get(job_id)
        if cached is not None:
            return cached
        record = self._load_one(job_id)
        if record is None or record.status != JOB_STATUS_RUNNING:
            return record
        if record.runner == JOB_RUNNER_THREAD and not self._recovered:
            return self._load_one(job_id, recover=True) or record
        if _abandoned(record) and job_id in self._maybe_sweep():
            return self._load_one(job_id) or record
        return record

    def list(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[JobRecord]:
        """Returns jobs with the given statuses, newest first.

        In-memory records of thread jobs win over persisted rows for the same job_id because they are always
        at least as fresh; for a Celery job the row wins. Sweeps stuck Celery jobs first (at most every 30 s).
        """
        wanted = [s for s in (statuses or []) if s in JOB_STATUSES] or list(JOB_STATUSES)
        self._maybe_sweep()
        merged: Dict[str, JobRecord] = {}
        for record in self._load_many(wanted, limit):
            merged[record.job_id] = record
        with self._lock:
            self._evict_finished_locked()
            for record in self._jobs.values():
                if record.runner != JOB_RUNNER_THREAD and record.job_id in merged:
                    continue  # the worker writes the row of a Celery job: no copy here is fresher
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
        response_cache.invalidate(response_cache.JOBS)
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

    def _maybe_sweep(self) -> List[str]:
        """sweep_stale() with the default age, at most once per SWEEP_INTERVAL_SEC in this process."""
        now = _monotonic()
        with self._lock:
            if now - self._last_sweep < SWEEP_INTERVAL_SEC:
                return []
            self._last_sweep = now
        return self.sweep_stale()

    def _warn_untracked_runner(self) -> None:
        if self._untracked_runner_logged:
            return
        self._untracked_runner_logged = True
        schema, table = _jobs_table_ref()
        logger.warning(
            "%s.%s has no owner/runner/started_at columns: Celery jobs are not counted in the job limits, not"
            " swept when their worker dies, and a restart of the web process marks them interrupted",
            schema, table,
        )

    # ---- persistence ----------------------------------------------------

    def _columns(self) -> Tuple[str, ...]:
        """The job columns this table has; known once _ensure_table() ran."""
        columns = _JOB_COLUMNS if self._columns_supported else _BASE_COLUMNS
        return columns + (_REF_COLUMN,) if self._refs_supported else columns

    def _columns_sql(self) -> sql.Composable:
        return sql.SQL(", ").join(sql.Identifier(c) for c in self._columns())

    def _ensure_table(self, conn, *, recover: bool = False) -> None:
        """Creates or upgrades the table once per process; with ``recover`` also runs the restart recovery once.

        Only the paths of a process that runs thread jobs pass ``recover`` (writes of in-memory records, the
        job list, the limits, dismissing, adopting): a Celery worker, which only updates rows, never marks the
        thread jobs of a live web process interrupted.
        """
        if self._table_ensured and (self._recovered or not recover):
            return
        schema, table = _jobs_table_ref()
        prev_autocommit = getattr(conn, "autocommit", False)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                if not self._table_ensured:
                    self._create_table(cur, schema, table)
                    present = self._present_columns(cur, schema, table)
                    self._ensure_dismissed_column(cur, schema, table, present)
                    self._ensure_tracking_columns(cur, schema, table, present)
                    self._ensure_ref_column(cur, schema, table, present)
                    self._table_ensured = True
                if recover and not self._recovered:
                    self._mark_interrupted(cur, schema, table)
                    self._recovered = True
        finally:
            conn.autocommit = prev_autocommit

    @staticmethod
    def _create_table(cur, schema: str, table: str) -> None:
        try:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.{} (
                        job_id       TEXT PRIMARY KEY,
                        kind         TEXT NOT NULL,
                        run_name     TEXT,
                        service      TEXT,
                        status       TEXT NOT NULL,
                        progress     INTEGER NOT NULL DEFAULT 0,
                        message      TEXT,
                        error        TEXT,
                        report_url   TEXT,
                        page_url     TEXT,
                        page_id      TEXT,
                        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                        dismissed_at TIMESTAMPTZ,
                        owner        TEXT NOT NULL DEFAULT '',
                        runner       TEXT NOT NULL DEFAULT 'thread',
                        started_at   TIMESTAMPTZ,
                        message_ref  JSONB
                    );
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            # Serves the lists by status, the running-job count and the sweep (status, then updated_at).
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (status, updated_at DESC);").format(
                    sql.Identifier(f"idx_{table}_status_updated"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
        except psycopg2.Error:
            # PostgreSQL checks privileges before IF NOT EXISTS: a role without CREATE on the schema, or not owning
            # the table, fails here even when the owner made both already. Use the table as it is, if it is there.
            cur.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s", (schema, table)
            )
            if not cur.fetchall():
                raise
            logger.warning(
                "Cannot create %s.%s or its index (missing privileges?): using the existing table",
                schema, table, exc_info=True,
            )

    @staticmethod
    def _present_columns(cur, schema: str, table: str) -> Set[str]:
        # Checked first: ALTER TABLE needs ownership even when the column is already there.
        cur.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_schema = %s AND table_name = %s AND column_name = ANY(%s)",
            (schema, table, list(_OPTIONAL_COLUMNS)),
        )
        return {str(row[0]) for row in (cur.fetchall() or [])}

    def _ensure_dismissed_column(self, cur, schema: str, table: str, present: Set[str]) -> None:
        if "dismissed_at" in present:
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

    def _ensure_tracking_columns(self, cur, schema: str, table: str, present: Set[str]) -> None:
        if all(column in present for column in _TRACKING_COLUMNS):
            self._columns_supported = True
            return
        try:
            # Existing rows were all thread jobs. Constant defaults make this a catalog-only change.
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS owner TEXT NOT NULL DEFAULT '',"
                    " ADD COLUMN IF NOT EXISTS runner TEXT NOT NULL DEFAULT 'thread',"
                    " ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            self._columns_supported = True
        except psycopg2.Error:
            self._columns_supported = False
            logger.warning(
                "Cannot add owner/runner/started_at to %s.%s (the table belongs to another role?): the job limits"
                " count only the jobs of this process and stuck Celery jobs are not swept",
                schema, table, exc_info=True,
            )

    def _ensure_ref_column(self, cur, schema: str, table: str, present: Set[str]) -> None:
        if _REF_COLUMN in present:
            self._refs_supported = True
            return
        try:
            cur.execute(
                sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS message_ref JSONB").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            self._refs_supported = True
        except psycopg2.Error:
            self._refs_supported = False
            logger.warning(
                "Cannot add message_ref to %s.%s (the table belongs to another role?): job messages are shown in the"
                " language they were written in",
                schema, table, exc_info=True,
            )

    def _mark_interrupted(self, cur, schema: str, table: str) -> None:
        """Thread jobs left "running" by a process that stopped become errors, so nothing polls them forever.

        Only ``runner='thread'`` rows: a Celery job outlives a restart of the web process, and sweep_stale()
        handles dead workers. Without the runner column every row is a thread job, as before it existed.
        """
        if self._columns_supported:
            condition, params = sql.SQL(" AND runner = %s AND updated_at < %s"), [JOB_RUNNER_THREAD, self._started_at]
        else:
            condition, params = sql.SQL(" AND updated_at < %s"), [self._started_at]
        interrupted = self._fail_running(
            cur, schema, table, condition, params, message_key=INTERRUPTED_KEY, error=INTERRUPTED_ERROR,
        )
        if interrupted:
            logger.warning("Marked %d job(s) interrupted by a restart: %s", len(interrupted), ", ".join(interrupted[:20]))

    def _fail_running(
        self, cur, schema: str, table: str, condition: sql.Composable, condition_params: list, *,
        message_key: str, error: str,
    ) -> List[str]:
        """Turns running rows that also match ``condition`` into errors; returns their ids.

        The message is ``message_key`` in the installation language, plus its ref when the table has the column.
        The ref is a constant of this module, written as a literal so the statement keeps its parameters.
        """
        text = job_message(message_key)
        ref_sql: sql.Composable = sql.SQL("")
        if self._refs_supported:
            ref_sql = sql.SQL(", message_ref = {}::jsonb").format(sql.Literal(_ref_json(text.ref())))
        cur.execute(
            sql.SQL(
                "UPDATE {}.{} SET status = %s, message = %s, error = %s, updated_at = %s{}"
                " WHERE status = %s{} RETURNING job_id"
            ).format(sql.Identifier(schema), sql.Identifier(table), ref_sql, condition),
            [JOB_STATUS_ERROR, str(text), error, _utcnow(), JOB_STATUS_RUNNING, *condition_params],
        )
        return [str(row[0]) for row in (cur.fetchall() or [])]

    def _dismiss_rows(self, job_ids: List[str]) -> Tuple[Set[str], Set[str]]:
        """Sets dismissed_at on finished rows; returns (hidden ids, ids whose rows are still running)."""
        schema, table = _jobs_table_ref()
        if not self._dismiss_supported:
            raise JobStoreError(_DISMISS_UNSUPPORTED)
        conn = _ts_conn()
        try:
            self._ensure_table(conn, recover=True)
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
        """Writes an in-memory record through to its row (only the process running thread jobs has them)."""
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn, recover=True)
                columns = self._columns()
                overwritten = [c for c in columns if c not in _INSERT_ONLY_COLUMNS]
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("INSERT INTO {}.{} ({}) VALUES ({}) ON CONFLICT (job_id) DO UPDATE SET {}").format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                            sql.SQL(", ").join(_placeholder(c) for c in columns),
                            sql.SQL(", ").join(sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in overwritten),
                        ),
                        tuple(_column_value(record, c) for c in columns),
                    )
            finally:
                conn.close()
        except Exception as exc:
            # The job itself keeps running; only the persisted status is stale.
            _log_store_failure(exc, "Failed to persist job %s (%s) to %s.%s", record.job_id, record.kind, schema, table)

    def _update_row(self, job_id: str, fields: dict) -> Optional[JobRecord]:
        """update() of a job that is not in memory: one UPDATE ... RETURNING, so concurrent updates (threads
        of the worker, the web process) cannot undo each other's changes, and progress never decreases."""
        unknown = sorted(set(fields) - _UPDATABLE_COLUMNS)
        if unknown:
            raise TypeError(f"Cannot update job fields in the table: {', '.join(unknown)}")
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn)
                assignments: List[sql.Composable] = []
                params: list = []
                for name, value in fields.items():
                    if name == "progress":
                        if value is None:
                            continue
                        assignments.append(sql.SQL("progress = GREATEST(progress, %s)"))
                        params.append(int(value))
                    elif name == _REF_COLUMN:
                        if not self._refs_supported:
                            continue  # an older table keeps the text only
                        assignments.append(sql.SQL("message_ref = %s::jsonb"))
                        params.append(_ref_json(value))
                    else:
                        assignments.append(sql.SQL("{} = %s").format(sql.Identifier(name)))
                        params.append(value)
                assignments.append(sql.SQL("updated_at = %s"))
                params.extend((_utcnow(), job_id))
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("UPDATE {}.{} SET {} WHERE job_id = %s RETURNING {}").format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                            sql.SQL(", ").join(assignments),
                            self._columns_sql(),
                        ),
                        params,
                    )
                    row = cur.fetchone()
            finally:
                conn.close()
        except Exception as exc:
            _log_store_failure(exc, "Failed to update job %s in %s.%s", job_id, schema, table)
            return None
        if not row:
            return None
        _invalidate_caches(None, fields.get("status"))
        return JobRecord.from_row(row, self._columns())

    def _touch_running(self, job_id: str, now: datetime, *, started: bool) -> Optional[bool]:
        """Sets ``updated_at`` (and ``started_at`` when ``started``) of a running row: heartbeat(), mark_started()."""
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn)
                if started and self._columns_supported:
                    assignments, params = "started_at = %s, updated_at = %s", [now, now]
                else:
                    assignments, params = "updated_at = %s", [now]
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("UPDATE {}.{} SET {} WHERE job_id = %s AND status = %s RETURNING job_id").format(
                            sql.Identifier(schema), sql.Identifier(table), sql.SQL(assignments)
                        ),
                        [*params, job_id, JOB_STATUS_RUNNING],
                    )
                    touched = bool(cur.fetchall())
            finally:
                conn.close()
        except Exception as exc:
            what = "the start" if started else "a heartbeat"
            _log_store_failure(exc, "Failed to record %s of job %s in %s.%s", what, job_id, schema, table)
            return None
        return touched

    def _running_owners(self) -> Dict[str, str]:
        """``job_id -> owner`` of the running rows sweep_stale() would leave alone; {} without the table or columns."""
        schema, table = _jobs_table_ref()
        now = _utcnow()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn, recover=True)
                if not self._columns_supported:
                    return {}
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "SELECT job_id, owner FROM {}.{} WHERE status = %s"
                            " AND ((started_at IS NOT NULL AND updated_at >= %s)"
                            " OR (started_at IS NULL AND created_at >= %s))"
                        ).format(sql.Identifier(schema), sql.Identifier(table)),
                        (JOB_STATUS_RUNNING, now - stale_after(), now - queue_max_wait()),
                    )
                    rows = cur.fetchall() or []
            finally:
                conn.close()
        except Exception as exc:
            _log_store_failure(exc, "Failed to count running jobs in %s.%s", schema, table)
            return {}
        return {str(row[0]): str(row[1] or "") for row in rows}

    def _load_one(self, job_id: str, *, recover: bool = False) -> Optional[JobRecord]:
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn, recover=recover)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT {} FROM {}.{} WHERE job_id = %s").format(
                            self._columns_sql(),
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
        return JobRecord.from_row(row, self._columns()) if row else None

    def _load_many(self, statuses: List[str], limit: int) -> List[JobRecord]:
        schema, table = _jobs_table_ref()
        try:
            conn = _ts_conn()
            try:
                self._ensure_table(conn, recover=True)
                # Without the dismissed_at column nothing can be hidden, and the filter would fail.
                visible = sql.SQL(" AND dismissed_at IS NULL") if self._dismiss_supported else sql.SQL("")
                with conn, conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "SELECT {} FROM {}.{} WHERE status = ANY(%s){} ORDER BY updated_at DESC LIMIT %s"
                        ).format(
                            self._columns_sql(),
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
        return [JobRecord.from_row(row, self._columns()) for row in rows]


job_store = JobStore()


def __getattr__(name: str) -> str:
    """``INTERRUPTED_MESSAGE``, ``STALE_MESSAGE``, ``QUEUED_MESSAGE``: the text a row ended by the store gets
    (its message key in :func:`stored_locale`); quoted by docs/jobs-runner.md."""
    key = _MESSAGE_CONSTANTS.get(name)
    if key is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return t(key, locale=stored_locale())


__all__ = [
    "DEFAULT_QUEUE_MAX_MIN",
    "DEFAULT_STALE_MIN",
    "INTERRUPTED_KEY",
    "INTERRUPTED_MESSAGE",
    "JOB_KIND_CONFLUENCE",
    "JOB_KIND_FORECAST_CONFLUENCE",
    "JOB_KIND_REPORT",
    "JOB_RUNNER_CELERY",
    "JOB_RUNNER_THREAD",
    "JOB_RUNNERS",
    "JOB_STATUS_DONE",
    "JOB_STATUS_ERROR",
    "JOB_STATUS_RUNNING",
    "JOB_STATUSES",
    "JobRecord",
    "JobStore",
    "JobStoreError",
    "QUEUED_KEY",
    "QUEUED_MESSAGE",
    "QUEUE_MAX_MIN_ENV",
    "STALE_KEY",
    "STALE_MESSAGE",
    "STALE_MIN_ENV",
    "SWEEP_INTERVAL_SEC",
    "failure_message",
    "job_message",
    "job_store",
    "message_fields",
    "message_text",
    "queue_max_wait",
    "stale_after",
    "stored_locale",
]
