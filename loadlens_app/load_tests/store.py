# SPDX-License-Identifier: Apache-2.0
"""Load test runs in the ``load_test_runs`` table (instances in a JSONB column).

The table is the source of truth: the watcher of any process can pick up a run whose watcher stopped
heartbeating, and stop requests from any request handler land in their own columns, so they never race
with the watcher rewriting the instances.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2 import sql

from AI.i18n import is_ref

from .messages import Titles

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "load_test_runs"

# Run statuses.
RUN_STARTING = "starting"
RUN_RUNNING = "running"
RUN_STOPPING = "stopping"
RUN_DRAINING = "draining"
RUN_REPORTING = "reporting"
RUN_DONE = "done"
RUN_FAILED = "failed"
RUN_CANCELLED = "cancelled"
RUN_ACTIVE = (RUN_STARTING, RUN_RUNNING, RUN_STOPPING, RUN_DRAINING, RUN_REPORTING)
RUN_STATUSES = RUN_ACTIVE + (RUN_DONE, RUN_FAILED, RUN_CANCELLED)
# Runs still holding a generator: instances may be running.
RUN_LOADING = (RUN_STARTING, RUN_RUNNING, RUN_STOPPING)

# Instance statuses.
INST_PENDING = "pending"
INST_STARTING = "starting"
INST_RUNNING = "running"
INST_FINISHED = "finished"
INST_CHECKS_FAILED = "checks_failed"
INST_FAILED = "failed"
INST_STOPPED = "stopped"
INST_TIMED_OUT = "timed_out"
INST_LOST = "lost"
INST_CANCELLED = "cancelled"
INST_ACTIVE = (INST_PENDING, INST_STARTING, INST_RUNNING)
INST_TERMINAL = (INST_FINISHED, INST_CHECKS_FAILED, INST_FAILED, INST_STOPPED, INST_TIMED_OUT, INST_LOST, INST_CANCELLED)

# Titles of the statuses in the reader's language (common.lt_run_status.* / common.lt_instance_status.*).
RUN_STATUS_TITLES = Titles({
    RUN_STARTING: "common.lt_run_status.starting",
    RUN_RUNNING: "common.lt_run_status.running",
    RUN_STOPPING: "common.lt_run_status.stopping",
    RUN_DRAINING: "common.lt_run_status.draining",
    RUN_REPORTING: "common.lt_run_status.reporting",
    RUN_DONE: "common.lt_run_status.done",
    RUN_FAILED: "common.lt_run_status.failed",
    RUN_CANCELLED: "common.lt_run_status.cancelled",
})
INSTANCE_STATUS_TITLES = Titles({
    INST_PENDING: "common.lt_instance_status.pending",
    INST_STARTING: "common.lt_instance_status.starting",
    INST_RUNNING: "common.lt_instance_status.running",
    INST_FINISHED: "common.lt_instance_status.finished",
    INST_CHECKS_FAILED: "common.lt_instance_status.checks_failed",
    INST_FAILED: "common.lt_instance_status.failed",
    INST_STOPPED: "common.lt_instance_status.stopped",
    INST_TIMED_OUT: "common.lt_instance_status.timed_out",
    INST_LOST: "common.lt_instance_status.lost",
    INST_CANCELLED: "common.lt_instance_status.cancelled",
})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


_TIME_FIELDS = ("launched_at", "started_at", "finished_at", "stop_sent_at", "seen_at", "status_at", "stop_failed_at")


def _ref(value: Any) -> Optional[dict[str, Any]]:
    """A stored ``{"key", "params"}`` (JSON text in old drivers); anything else is no ref."""
    if isinstance(value, str) and value:
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return dict(value) if is_ref(value) else None


@dataclass
class InstanceRecord:
    index: int
    name: str
    tag: str
    generator: str
    tool: str = ""
    script: str = ""
    params: str = ""
    # Values are kept only until the instance ends; afterwards the keys stay with masked values.
    env: dict[str, str] = field(default_factory=dict)
    start_delay_sec: int = 0
    # The command as shown to users and in the audit log: environment values masked.
    command: str = ""
    status: str = INST_PENDING
    ref: str = ""
    exit_code: Optional[int] = None
    message: str = ""
    # {"key", "params"} of the message: rendered again in the language of each reader (None: show the text).
    message_ref: Optional[dict[str, Any]] = None
    launched_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    stop_sent_at: Optional[datetime] = None
    killed: bool = False
    # Consecutive probes that found the runner gone without an exit code.
    dead_probes: int = 0
    # Last time the generator answered a probe.
    seen_at: Optional[datetime] = None
    # Tail of the tool output saved when the instance stopped (containers and jobs are removed later).
    log_tail: str = ""
    # CI generators: SHA-256 of the per-instance secret the pipeline presents.
    ci_token_hash: str = ""
    # The watcher stopped the instance because the run limit passed (generators without their own timer).
    timed_out: bool = False
    # Last answer of the tool status command (k6: VUs), and when it was asked.
    status_text: str = ""
    status_text_ref: Optional[dict[str, Any]] = None
    status_at: Optional[datetime] = None
    # First failed attempt to deliver the stop: a generator that keeps refusing it is given up on later.
    stop_failed_at: Optional[datetime] = None
    # CI generators: gitlab | jenkins, the branch, the values of the pipeline parameters and extra GitLab
    # variables; ci_secret names the password parameters, masked like environment values.
    ci_kind: str = ""
    ci_ref: str = ""
    ci_params: dict[str, str] = field(default_factory=dict)
    ci_extra: dict[str, str] = field(default_factory=dict)
    ci_secret: list[str] = field(default_factory=list)
    # The pipeline passes LOADLENS_RUN_TAG to the tool (generator setting at launch time).
    ci_tagged: bool = False
    # Page of the pipeline or build.
    external_url: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        for key in _TIME_FIELDS:
            data[key] = _iso(getattr(self, key))
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "InstanceRecord":
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in (data or {}).items() if key in known}
        for key in _TIME_FIELDS:
            values[key] = _parse_dt(values.get(key))
        values["env"] = {str(k): str(v) for k, v in (values.get("env") or {}).items()}
        for key in ("ci_params", "ci_extra"):
            raw = values.get(key)
            values[key] = {str(k): "" if v is None else str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        values["ci_secret"] = [str(name) for name in values.get("ci_secret") or [] if name]
        for key in ("message_ref", "status_text_ref"):
            values[key] = _ref(values.get(key))
        return cls(**values)

    @property
    def active(self) -> bool:
        return self.status in INST_ACTIVE


@dataclass
class RunRecord:
    run_id: str
    run_name: str
    run_tag: str
    service: str
    project_area: str = ""
    test_type: str = ""
    status: str = RUN_STARTING
    message: str = ""
    owner: str = ""
    # Report options: auto_report, use_llm, web_only, max_seconds.
    options: dict[str, Any] = field(default_factory=dict)
    instances: list[InstanceRecord] = field(default_factory=list)
    report_job_id: Optional[str] = None
    report_url: Optional[str] = None
    stop_all_at: Optional[datetime] = None
    stop_instances: list[int] = field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    watcher: Optional[str] = None
    watcher_seen: Optional[datetime] = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    # Events reported by CI jobs: {"index", "event": "started" | "finished", "exit_code", "message", "at"}.
    events: list[dict[str, Any]] = field(default_factory=list)
    # {"key", "params"} of the message: rendered again in the language of each reader (None: show the text).
    message_ref: Optional[dict[str, Any]] = None

    @property
    def active(self) -> bool:
        return self.status in RUN_ACTIVE

    def stop_requested(self, index: int) -> bool:
        return self.stop_all_at is not None or index in self.stop_instances

    def copy(self) -> "RunRecord":
        return copy.deepcopy(self)


_COLUMNS = (
    "run_id", "run_name", "run_tag", "service", "project_area", "test_type", "status", "message", "owner",
    "options", "instances", "report_job_id", "report_url", "stop_all_at", "stop_instances", "started_at",
    "finished_at", "watcher", "watcher_seen", "created_at", "updated_at", "events",
)
# Added after the table: an older table gets it through ALTER TABLE when the role may do it; without it the
# messages are shown as stored, in the language of the installation.
_REF_COLUMN = "message_ref"


def _from_row(row: tuple, columns: tuple[str, ...] = _COLUMNS) -> RunRecord:
    values = dict(zip(columns, row))
    options = values.get("options")
    instances = values.get("instances")
    events = values.get("events")
    if isinstance(events, str):
        events = json.loads(events or "[]")
    if isinstance(options, str):
        options = json.loads(options or "{}")
    if isinstance(instances, str):
        instances = json.loads(instances or "[]")
    return RunRecord(
        run_id=str(values["run_id"]),
        run_name=str(values["run_name"] or ""),
        run_tag=str(values["run_tag"] or ""),
        service=str(values["service"] or ""),
        project_area=str(values["project_area"] or ""),
        test_type=str(values["test_type"] or ""),
        status=str(values["status"] or RUN_STARTING),
        message=str(values["message"] or ""),
        owner=str(values["owner"] or ""),
        options=dict(options or {}),
        instances=[InstanceRecord.from_json(item) for item in (instances or [])],
        report_job_id=values.get("report_job_id"),
        report_url=values.get("report_url"),
        stop_all_at=_parse_dt(values.get("stop_all_at")),
        stop_instances=[int(i) for i in (values.get("stop_instances") or [])],
        started_at=_parse_dt(values.get("started_at")),
        finished_at=_parse_dt(values.get("finished_at")),
        watcher=values.get("watcher"),
        watcher_seen=_parse_dt(values.get("watcher_seen")),
        created_at=_parse_dt(values.get("created_at")) or utcnow(),
        updated_at=_parse_dt(values.get("updated_at")) or utcnow(),
        events=[event for event in (events or []) if isinstance(event, dict)],
        message_ref=_ref(values.get(_REF_COLUMN)),
    )


def _ref_json(ref: Optional[dict[str, Any]]) -> Optional[str]:
    return json.dumps(ref, ensure_ascii=False) if ref else None


class LoadTestStore:
    """Interface shared by the PostgreSQL store and the in-memory one used in tests."""

    def create(self, run: RunRecord) -> None:
        raise NotImplementedError

    def get(self, run_id: str) -> Optional[RunRecord]:
        raise NotImplementedError

    def list(self, limit: int = 50, statuses: Optional[Iterable[str]] = None, services: Optional[list[str]] = None) -> list[RunRecord]:
        raise NotImplementedError

    def claim(self, watcher_id: str, stale_after_sec: int) -> list[RunRecord]:
        """Active runs this watcher owns now: its own, unowned ones and those whose watcher went silent."""
        raise NotImplementedError

    def save_progress(self, run: RunRecord, watcher_id: str) -> bool:
        """Writes what the watcher changed; ``False`` when another watcher took the run over."""
        raise NotImplementedError

    def heartbeat(self, run_id: str, watcher_id: str) -> bool:
        """Renews the claim of a run during a long pass; ``False`` when another watcher took it over."""
        raise NotImplementedError

    def request_stop(self, run_id: str, index: Optional[int] = None) -> bool:
        raise NotImplementedError

    def name_in_use(self, run_name: str, service: str) -> bool:
        raise NotImplementedError

    def record_event(self, run_id: str, event: dict[str, Any]) -> bool:
        """Appends a CI event to an active run; ``False`` when the run is gone or finished."""
        raise NotImplementedError


MAX_EVENTS = 200


class MemoryLoadTestStore(LoadTestStore):
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, run: RunRecord) -> None:
        with self._lock:
            self._runs[run.run_id] = run.copy()

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            run = self._runs.get(run_id)
            return run.copy() if run else None

    def list(self, limit: int = 50, statuses: Optional[Iterable[str]] = None, services: Optional[list[str]] = None) -> list[RunRecord]:
        wanted = set(statuses or RUN_STATUSES)
        with self._lock:
            runs = [r.copy() for r in self._runs.values() if r.status in wanted and (not services or r.service in services)]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[: max(1, int(limit))]

    def claim(self, watcher_id: str, stale_after_sec: int) -> list[RunRecord]:
        now = utcnow()
        claimed: list[RunRecord] = []
        with self._lock:
            for run in self._runs.values():
                if run.status not in RUN_ACTIVE:
                    continue
                stale = run.watcher_seen is None or run.watcher_seen < now - timedelta(seconds=stale_after_sec)
                if run.watcher in (None, watcher_id) or stale:
                    run.watcher = watcher_id
                    run.watcher_seen = now
                    claimed.append(run.copy())
        claimed.sort(key=lambda r: r.created_at)
        return claimed

    def save_progress(self, run: RunRecord, watcher_id: str) -> bool:
        with self._lock:
            current = self._runs.get(run.run_id)
            if current is None or current.watcher != watcher_id:
                return False
            # Stop requests are owned by the request handlers: keep what they wrote meanwhile.
            self._runs[run.run_id] = replace(
                run.copy(),
                stop_all_at=current.stop_all_at,
                stop_instances=list(current.stop_instances),
                events=copy.deepcopy(current.events),
                watcher=watcher_id,
                watcher_seen=utcnow(),
                updated_at=utcnow(),
            )
            return True

    def heartbeat(self, run_id: str, watcher_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.watcher != watcher_id or run.status not in RUN_ACTIVE:
                return False
            run.watcher_seen = utcnow()
            return True

    def request_stop(self, run_id: str, index: Optional[int] = None) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status not in RUN_ACTIVE:
                return False
            if index is None:
                run.stop_all_at = run.stop_all_at or utcnow()
            elif index not in run.stop_instances:
                run.stop_instances.append(index)
            run.updated_at = utcnow()
            return True

    def name_in_use(self, run_name: str, service: str) -> bool:
        with self._lock:
            return any(r.run_name == run_name and r.service == service for r in self._runs.values())

    def record_event(self, run_id: str, event: dict[str, Any]) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status not in RUN_ACTIVE or len(run.events) >= MAX_EVENTS:
                return False
            run.events.append(copy.deepcopy(event))
            return True


def _table_ref() -> tuple[str, str]:
    from settings import CONFIG

    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return str(cfg.get("schema", "public")), str(cfg.get("load_tests_table", DEFAULT_TABLE))


class PgLoadTestStore(LoadTestStore):
    def __init__(self) -> None:
        self._ensured = False
        self._lock = threading.Lock()
        # Whether the table has the message_ref column (False when it is old and the role may not ALTER it).
        self._refs = False

    def _conn(self):
        from loadlens_app.core import _ts_conn

        conn = _ts_conn()
        try:
            self._ensure(conn)
        except Exception:
            conn.close()
            raise
        return conn

    def _ensure(self, conn) -> None:
        if self._ensured:
            return
        with self._lock:
            if self._ensured:
                return
            schema, table = _table_ref()
            prev = getattr(conn, "autocommit", False)
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute(sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            run_id          TEXT PRIMARY KEY,
                            run_name        TEXT NOT NULL,
                            run_tag         TEXT NOT NULL,
                            service         TEXT NOT NULL,
                            project_area    TEXT NOT NULL DEFAULT '',
                            test_type       TEXT NOT NULL DEFAULT '',
                            status          TEXT NOT NULL,
                            message         TEXT NOT NULL DEFAULT '',
                            owner           TEXT NOT NULL DEFAULT '',
                            options         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                            instances       JSONB NOT NULL DEFAULT '[]'::jsonb,
                            report_job_id   TEXT,
                            report_url      TEXT,
                            stop_all_at     TIMESTAMPTZ,
                            stop_instances  INTEGER[] NOT NULL DEFAULT '{{}}',
                            started_at      TIMESTAMPTZ,
                            finished_at     TIMESTAMPTZ,
                            watcher         TEXT,
                            watcher_seen    TIMESTAMPTZ,
                            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                            events          JSONB NOT NULL DEFAULT '[]'::jsonb,
                            message_ref     JSONB
                        )
                        """
                    ).format(sql.Identifier(schema), sql.Identifier(table)))
                    # Tables created before CI generators lack the events column.
                    cur.execute(
                        "SELECT 1 FROM information_schema.columns WHERE table_schema = %s AND table_name = %s AND column_name = 'events'",
                        (schema, table),
                    )
                    if not cur.fetchall():
                        cur.execute(sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS events JSONB NOT NULL DEFAULT '[]'::jsonb").format(
                            sql.Identifier(schema), sql.Identifier(table)))
                    cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (status, created_at DESC)").format(
                        sql.Identifier(f"idx_{table}_status_created"), sql.Identifier(schema), sql.Identifier(table)))
                    cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (service, run_name)").format(
                        sql.Identifier(f"idx_{table}_service_name"), sql.Identifier(schema), sql.Identifier(table)))
                    self._refs = self._ensure_ref_column(cur, schema, table)
            finally:
                conn.autocommit = prev
            self._ensured = True

    @staticmethod
    def _ensure_ref_column(cur, schema: str, table: str) -> bool:
        """Adds ``message_ref`` to a table created before it; ``False`` when the role may not ALTER the table."""
        # Checked first: ALTER TABLE needs ownership even when the column is already there.
        cur.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = %s AND table_name = %s AND column_name = %s",
            (schema, table, _REF_COLUMN),
        )
        if cur.fetchall():
            return True
        try:
            cur.execute(sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS message_ref JSONB").format(
                sql.Identifier(schema), sql.Identifier(table)))
        except psycopg2.Error:
            logger.warning(
                "Cannot add message_ref to %s.%s (the table belongs to another role?): run messages are shown in the"
                " language they were written in", schema, table, exc_info=True,
            )
            return False
        return True

    def _columns(self) -> tuple[str, ...]:
        return _COLUMNS + ((_REF_COLUMN,) if self._refs else ())

    def _row(self, row: tuple) -> RunRecord:
        return _from_row(row, self._columns())

    def _ref(self) -> sql.Composed:
        schema, table = _table_ref()
        return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))

    def _select(self) -> sql.Composed:
        return sql.SQL("SELECT {} FROM {}").format(sql.SQL(", ").join(sql.Identifier(c) for c in self._columns()), self._ref())

    def create(self, run: RunRecord) -> None:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                ref_column, ref_value = (", message_ref", ", %s::jsonb") if self._refs else ("", "")
                params = [
                    run.run_id, run.run_name, run.run_tag, run.service, run.project_area, run.test_type, run.status,
                    run.message, run.owner, json.dumps(run.options), json.dumps([i.to_json() for i in run.instances]),
                    run.created_at, run.updated_at,
                ]
                if self._refs:
                    params.append(_ref_json(run.message_ref))
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (run_id, run_name, run_tag, service, project_area, test_type, status, message, owner,"
                        " options, instances, created_at, updated_at" + ref_column + ") VALUES"
                        " (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s" + ref_value + ")"
                    ).format(self._ref()),
                    tuple(params),
                )
        finally:
            conn.close()

    def get(self, run_id: str) -> Optional[RunRecord]:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql.SQL("{} WHERE run_id = %s").format(self._select()), (run_id,))
                row = cur.fetchone()
        finally:
            conn.close()
        return self._row(row) if row else None

    def list(self, limit: int = 50, statuses: Optional[Iterable[str]] = None, services: Optional[list[str]] = None) -> list[RunRecord]:
        wanted = [s for s in (statuses or RUN_STATUSES) if s in RUN_STATUSES]
        clauses = [sql.SQL("status = ANY(%s)")]
        params: list[Any] = [wanted]
        if services:
            clauses.append(sql.SQL("service = ANY(%s)"))
            params.append(list(services))
        params.append(max(1, int(limit)))
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    sql.SQL("{} WHERE {} ORDER BY created_at DESC LIMIT %s").format(self._select(), sql.SQL(" AND ").join(clauses)),
                    tuple(params),
                )
                rows = cur.fetchall() or []
        finally:
            conn.close()
        return [self._row(row) for row in rows]

    def claim(self, watcher_id: str, stale_after_sec: int) -> list[RunRecord]:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "UPDATE {} SET watcher = %s, watcher_seen = now()"
                        " WHERE status = ANY(%s) AND (watcher IS NULL OR watcher = %s OR watcher_seen IS NULL"
                        " OR watcher_seen < now() - (%s * interval '1 second')) RETURNING run_id"
                    ).format(self._ref()),
                    (watcher_id, list(RUN_ACTIVE), watcher_id, int(stale_after_sec)),
                )
                ids = [str(row[0]) for row in (cur.fetchall() or [])]
                if not ids:
                    return []
                cur.execute(sql.SQL("{} WHERE run_id = ANY(%s) ORDER BY created_at").format(self._select()), (ids,))
                rows = cur.fetchall() or []
        finally:
            conn.close()
        return [self._row(row) for row in rows]

    def save_progress(self, run: RunRecord, watcher_id: str) -> bool:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                ref_set = ", message_ref = %s::jsonb" if self._refs else ""
                params = [
                    run.status, run.message, json.dumps([i.to_json() for i in run.instances]), run.report_job_id,
                    run.report_url, run.started_at, run.finished_at, run.run_name,
                ]
                if self._refs:
                    params.append(_ref_json(run.message_ref))
                cur.execute(
                    sql.SQL(
                        "UPDATE {} SET status = %s, message = %s, instances = %s::jsonb, report_job_id = %s, report_url = %s,"
                        " started_at = %s, finished_at = %s, run_name = %s" + ref_set + ", watcher_seen = now(), updated_at = now()"
                        " WHERE run_id = %s AND watcher = %s"
                    ).format(self._ref()),
                    tuple(params + [run.run_id, watcher_id]),
                )
                return cur.rowcount == 1
        finally:
            conn.close()

    def heartbeat(self, run_id: str, watcher_id: str) -> bool:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    sql.SQL("UPDATE {} SET watcher_seen = now() WHERE run_id = %s AND watcher = %s AND status = ANY(%s)").format(self._ref()),
                    (run_id, watcher_id, list(RUN_ACTIVE)),
                )
                return cur.rowcount == 1
        finally:
            conn.close()

    def request_stop(self, run_id: str, index: Optional[int] = None) -> bool:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                if index is None:
                    cur.execute(
                        sql.SQL("UPDATE {} SET stop_all_at = COALESCE(stop_all_at, now()), updated_at = now()"
                                " WHERE run_id = %s AND status = ANY(%s)").format(self._ref()),
                        (run_id, list(RUN_ACTIVE)),
                    )
                else:
                    cur.execute(
                        sql.SQL("UPDATE {} SET stop_instances = array_append(stop_instances, %s), updated_at = now()"
                                " WHERE run_id = %s AND status = ANY(%s) AND NOT (%s = ANY(stop_instances))").format(self._ref()),
                        (int(index), run_id, list(RUN_ACTIVE), int(index)),
                    )
                    if cur.rowcount == 0:
                        cur.execute(sql.SQL("SELECT 1 FROM {} WHERE run_id = %s AND status = ANY(%s)").format(self._ref()),
                                    (run_id, list(RUN_ACTIVE)))
                        return cur.fetchone() is not None
                return cur.rowcount == 1
        finally:
            conn.close()

    def name_in_use(self, run_name: str, service: str) -> bool:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT 1 FROM {} WHERE run_name = %s AND service = %s LIMIT 1").format(self._ref()),
                            (run_name, service))
                return cur.fetchone() is not None
        finally:
            conn.close()

    def record_event(self, run_id: str, event: dict[str, Any]) -> bool:
        conn = self._conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "UPDATE {} SET events = events || %s::jsonb, updated_at = now()"
                        " WHERE run_id = %s AND status = ANY(%s) AND jsonb_array_length(events) < %s"
                    ).format(self._ref()),
                    (json.dumps([event]), run_id, list(RUN_ACTIVE), MAX_EVENTS),
                )
                return cur.rowcount == 1
        finally:
            conn.close()


_store: LoadTestStore = PgLoadTestStore()


def get_store() -> LoadTestStore:
    return _store


def set_store(store: LoadTestStore) -> LoadTestStore:
    """Replaces the store (tests); returns the previous one."""
    global _store
    previous, _store = _store, store
    return previous


__all__ = [
    "INSTANCE_STATUS_TITLES",
    "INST_ACTIVE",
    "INST_TERMINAL",
    "InstanceRecord",
    "LoadTestStore",
    "MemoryLoadTestStore",
    "PgLoadTestStore",
    "RUN_ACTIVE",
    "RUN_LOADING",
    "RUN_STATUSES",
    "RUN_STATUS_TITLES",
    "RunRecord",
    "get_store",
    "set_store",
    "utcnow",
]
