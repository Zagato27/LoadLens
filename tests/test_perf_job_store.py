"""The report_jobs table as the source of truth: Celery jobs, the sweeper, the job limits across processes.

``JobsTable`` stands in for PostgreSQL. It interprets the statements JobStore sends (the SQL objects are
rendered to text, conditions with AND/OR/parentheses are evaluated per row), so two JobStore instances on one
table behave like the web process and the worker sharing the database. A statement naming a column the table
does not have fails like it would in PostgreSQL.
"""

import logging
import re
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402,F401  (before anything that imports update_page)
from loadlens_app import jobs, response_cache  # noqa: E402
from AI.storage_state import StorageUnavailableError  # noqa: E402

ALL_COLUMNS = jobs._JOB_COLUMNS + ("dismissed_at",)
OLD_COLUMNS = jobs._BASE_COLUMNS  # a table from before dismissed_at, owner, runner and started_at
DEFAULTS = {"progress": 0, "owner": "", "runner": "thread"}
JSON_KEYS = {
    "job_id", "kind", "run_name", "service", "status", "progress", "message", "error", "report_url", "page_url",
    "page_id", "created_at", "updated_at",
}


def utc(minutes_ago: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)


def render(query) -> str:
    if isinstance(query, str):
        return query
    if isinstance(query, sql.Composed):
        return "".join(render(part) for part in query.seq)
    if isinstance(query, sql.SQL):
        return query.string
    if isinstance(query, sql.Identifier):
        return ".".join(query.strings)
    if isinstance(query, sql.Placeholder):
        return "%s"
    raise AssertionError(f"unexpected SQL part: {query!r}")


_TOKEN = re.compile(
    r"\s*(?:(?P<open>\()|(?P<close>\))|(?P<op>AND|OR)\b"
    r"|(?P<atom>\w+ IS NOT NULL|\w+ IS NULL|\w+ = ANY\(%s\)|\w+ (?:=|<>|<=|>=|<|>) %s))"
)
_COMPARE = {
    "=": lambda a, b: a == b,
    "<>": lambda a, b: a != b,
    "<": lambda a, b: a is not None and a < b,
    "<=": lambda a, b: a is not None and a <= b,
    ">": lambda a, b: a is not None and a > b,
    ">=": lambda a, b: a is not None and a >= b,
}


class JobsTable:
    """report_jobs in memory; ``statements`` keeps (text, params) of every executed statement."""

    def __init__(self, columns=ALL_COLUMNS, *, alter_fails=False):
        self.columns = list(columns)
        self.rows = {}
        self.alter_fails = alter_fails
        self.down = False
        self.statements = []

    def connect(self):
        if self.down:
            raise StorageUnavailableError("connection refused")
        return JobsConn(self)

    def add(self, job_id, status="running", *, runner="thread", owner="", created_min_ago=0.0, updated_min_ago=None,
            started=True, **values):
        created = utc(created_min_ago)
        updated = utc(created_min_ago if updated_min_ago is None else updated_min_ago)
        row = {column: DEFAULTS.get(column) for column in self.columns}
        row.update(job_id=job_id, kind="report", run_name=f"run-{job_id}", service="demo", status=status,
                   progress=10, message="stage", created_at=created, updated_at=updated)
        for name, value in (("owner", owner), ("runner", runner), ("started_at", created if started else None)):
            if name in self.columns:
                row[name] = value
        row.update(values)
        self.rows[job_id] = row
        return row

    def texts(self, fragment):
        return [text for text, _ in self.statements if fragment in text]

    def require(self, columns):
        for column in columns:
            if column not in self.columns:
                raise psycopg2.ProgrammingError(f'column "{column}" does not exist')

    # ---- a tiny SQL interpreter ------------------------------------------------------------------------

    def where(self, text, params):
        """Parses a WHERE clause into a row predicate, taking its parameters from ``params`` in order."""
        tokens = []
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if not match:
                raise AssertionError(f"cannot parse condition at: {text[position:]!r}")
            tokens.append((match.lastgroup, match.group(match.lastgroup)))
            position = match.end()

        def atom(source):
            if source.endswith("IS NOT NULL"):
                column = source.split()[0]
                self.require([column])
                return lambda row: row[column] is not None
            if source.endswith("IS NULL"):
                column = source.split()[0]
                self.require([column])
                return lambda row: row[column] is None
            if "= ANY(%s)" in source:
                column, value = source.split()[0], params.pop(0)
                self.require([column])
                return lambda row: row[column] in value
            column, op, _ = source.split()
            value = params.pop(0)
            self.require([column])
            return lambda row: _COMPARE[op](row[column], value)

        def parse_or():
            left = parse_and()
            while tokens and tokens[0] == ("op", "OR"):
                tokens.pop(0)
                right = parse_and()
                left = (lambda a, b: lambda row: a(row) or b(row))(left, right)
            return left

        def parse_and():
            left = parse_term()
            while tokens and tokens[0] == ("op", "AND"):
                tokens.pop(0)
                right = parse_term()
                left = (lambda a, b: lambda row: a(row) and b(row))(left, right)
            return left

        def parse_term():
            kind, value = tokens.pop(0)
            if kind == "open":
                inner = parse_or()
                assert tokens.pop(0)[0] == "close"
                return inner
            assert kind == "atom", value
            return atom(value)

        predicate = parse_or()
        assert not tokens, tokens
        return predicate

    def execute(self, query, params):
        text = " ".join(render(query).split())
        params = list(params or [])
        self.statements.append((text, tuple(params)))
        if text.startswith("CREATE"):
            return []
        if "information_schema.columns" in text:
            return [(name,) for name in params[2] if name in self.columns]
        if text.startswith("ALTER TABLE"):
            if self.alter_fails:
                raise psycopg2.ProgrammingError("must be owner of table report_jobs")
            for name, default in re.findall(r"ADD COLUMN IF NOT EXISTS (\w+) [A-Z]+(?: NOT NULL)?(?: DEFAULT ('[^']*'))?", text):
                if name not in self.columns:
                    self.columns.append(name)
                    for row in self.rows.values():
                        row[name] = default[1:-1] if default else None
            return []
        if text.startswith("INSERT INTO"):
            match = re.match(r"INSERT INTO \S+ \((.*?)\) VALUES \((.*?)\) ON CONFLICT \(job_id\) DO UPDATE SET (.*)$", text)
            columns = [c.strip() for c in match.group(1).split(",")]
            self.require(columns)
            values = dict(zip(columns, params, strict=True))
            row = self.rows.get(values["job_id"])
            if row is None:
                row = self.rows[values["job_id"]] = {column: DEFAULTS.get(column) for column in self.columns}
                row.update(values)
            else:
                for column, source in re.findall(r"(\w+) = EXCLUDED\.(\w+)", match.group(3)):
                    row[column] = values[source]
            return []
        if text.startswith("UPDATE"):
            match = re.match(r"UPDATE \S+ SET (.*?) WHERE (.*?)(?: RETURNING (.*))?$", text)
            assignments = re.findall(r"(\w+) = (GREATEST\(\w+, %s\)|%s)", match.group(1))
            assert match.group(1).count("%s") == len(assignments), match.group(1)
            self.require(column for column, _ in assignments)
            new_values = [params.pop(0) for _ in assignments]
            predicate = self.where(match.group(2), params)
            assert not params, f"unused parameters: {params}"
            returning = [c.strip() for c in (match.group(3) or "").split(",") if c.strip()]
            self.require(returning)
            changed = [row for row in self.rows.values() if predicate(row)]
            for row in changed:
                for (column, expression), value in zip(assignments, new_values):
                    row[column] = max(row[column], value) if expression.startswith("GREATEST") else value
            return [tuple(row[c] for c in returning) for row in changed]
        if text.startswith("SELECT"):
            match = re.match(r"SELECT (.*?) FROM \S+ WHERE (.*?)(?: ORDER BY (\w+) DESC)?( LIMIT %s)?$", text)
            columns = [c.strip() for c in match.group(1).split(",")]
            self.require(columns)
            predicate = self.where(match.group(2), params)
            rows = [row for row in self.rows.values() if predicate(row)]
            if match.group(3):
                rows.sort(key=lambda row: row[match.group(3)], reverse=True)
            if match.group(4):
                rows = rows[: params.pop(0)]
            assert not params, f"unused parameters: {params}"
            return [tuple(row[c] for c in columns) for row in rows]
        raise AssertionError(f"unexpected statement: {text}")


class JobsConn:
    def __init__(self, table):
        self.table = table
        self.autocommit = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return JobsCursor(self.table)

    def close(self):
        pass


class JobsCursor:
    def __init__(self, table):
        self.table = table
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.result = self.table.execute(query, params)

    def fetchone(self):
        return self.result[0] if self.result else None

    def fetchall(self):
        return list(self.result)


# ---- fixtures -------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    """Only the fake job table answers; any other storage access fails the test."""

    def no_db():
        raise AssertionError("these tests must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)
    monkeypatch.delenv(jobs.STALE_MIN_ENV, raising=False)
    monkeypatch.delenv(jobs.QUEUE_MAX_MIN_ENV, raising=False)


@pytest.fixture
def table(monkeypatch):
    fake = JobsTable()
    monkeypatch.setattr(jobs, "_ts_conn", fake.connect)
    return fake


@pytest.fixture
def clock(monkeypatch):
    """The monotonic clock of the sweep throttle; set before the store is created."""
    now = types.SimpleNamespace(value=1000.0)
    monkeypatch.setattr(jobs, "_monotonic", lambda: now.value)
    return now


@pytest.fixture
def wall(monkeypatch):
    """The wall clock of JobStore (timestamps written to rows and the cut-offs compared with them)."""
    now = types.SimpleNamespace(value=datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(jobs, "_utcnow", lambda: now.value)
    return now


@pytest.fixture
def caches(monkeypatch):
    """The response cache of this process, switched on, with a dashboard and a job list stored."""
    monkeypatch.setenv("LOADLENS_RESPONSE_CACHE", "1")
    cache = response_cache.response_cache
    cache.set(("dashboard", ()), b"{}", 60)
    cache.set(("jobs", ("running",), 20, True), b"{}", 60)
    return cache


def started_store(table):
    """A store whose start-up recovery already ran, so rows added afterwards meet only the code under test."""
    store = jobs.JobStore()
    store.list(statuses=["running"])
    return store


# ---- create / update ------------------------------------------------------------------------------------


def test_create_records_owner_and_runner_and_keeps_only_thread_jobs_in_memory(table):
    store = jobs.JobStore()

    thread = store.create(jobs.JOB_KIND_REPORT, run_name="nightly", service="demo", message="Инициализация…", owner="ann")
    celery = store.create(jobs.JOB_KIND_CONFLUENCE, run_name="nightly", service="demo", owner="bob", runner="celery")

    assert thread.job_id in store._jobs and celery.job_id not in store._jobs
    assert (table.rows[thread.job_id]["owner"], table.rows[thread.job_id]["runner"]) == ("ann", "thread")
    assert (table.rows[celery.job_id]["owner"], table.rows[celery.job_id]["runner"]) == ("bob", "celery")
    assert table.rows[thread.job_id]["started_at"] is not None  # a thread starts at once
    assert table.rows[celery.job_id]["started_at"] is None  # queued until a worker takes it
    loaded = store.get(celery.job_id)
    assert (loaded.runner, loaded.owner, loaded.kind) == ("celery", "bob", jobs.JOB_KIND_CONFLUENCE)
    # /jobs and /job_status keep their JSON shape.
    assert set(thread.to_dict()) == JSON_KEYS and set(loaded.to_dict()) == JSON_KEYS

    with pytest.raises(ValueError):
        store.create(jobs.JOB_KIND_REPORT, runner="cron")
    assert len(table.rows) == 2


def test_update_from_another_process_reaches_the_table_and_stays_out_of_memory(table, caches):
    web, worker = jobs.JobStore(), jobs.JobStore()
    job = web.create(jobs.JOB_KIND_REPORT, run_name="nightly", service="demo", message="Инициализация…", runner="celery")
    caches.set(("jobs", ("running",), 20, True), b"{}", 60)

    progressed = worker.update(job.job_id, message="Сбор метрик", progress=40)

    assert progressed.progress == 40 and progressed.message == "Сбор метрик" and progressed.runner == "celery"
    assert job.job_id not in worker._jobs and job.job_id not in web._jobs
    assert caches.get(("jobs", ("running",), 20, True)) is None
    assert table.rows[job.job_id]["progress"] == 40
    # One UPDATE ... RETURNING per call: no read-modify-write that a parallel update could undo.
    statement = table.statements[-1][0]
    assert statement.startswith("UPDATE") and "GREATEST(progress, %s)" in statement and "RETURNING" in statement

    assert worker.update(job.job_id, progress=10, message="late event").progress == 40
    assert worker.update(job.job_id, progress=None, message="no percent").progress == 40
    seen = web.get(job.job_id)
    assert (seen.status, seen.progress, seen.message) == ("running", 40, "no percent")

    assert caches.get(("dashboard", ())) == b"{}"  # progress alone keeps the dashboard
    done = worker.update(job.job_id, status=jobs.JOB_STATUS_DONE, progress=100, message="Готово", page_url="/reports/r1")
    assert done.status == "done" and caches.get(("dashboard", ())) is None
    assert web.get(job.job_id).page_url == "/reports/r1"


def test_update_of_a_table_job_checks_fields_and_survives_an_outage(table):
    store = jobs.JobStore()
    job = store.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", runner="celery")
    executed = len(table.statements)

    for bad in ({"kind": "x"}, {"runner": "thread"}, {"bogus": 1}):
        with pytest.raises(TypeError):
            store.update(job.job_id, **bad)
    assert len(table.statements) == executed

    assert store.update("missing", message="x") is None
    table.down = True
    assert store.update(job.job_id, message="x") is None
    assert store.get(job.job_id) is None


# ---- running_counts --------------------------------------------------------------------------------------


def test_running_counts_reads_live_rows_and_merges_memory(table):
    store = started_store(table)
    table.add("celery-live", runner="celery", owner="ann", created_min_ago=30, updated_min_ago=1)
    table.add("celery-silent", runner="celery", owner="ann", created_min_ago=30, updated_min_ago=20)
    table.add("celery-queued", runner="celery", owner="bob", created_min_ago=60, started=False)
    table.add("celery-queued-long", runner="celery", owner="bob", created_min_ago=130, started=False)
    table.add("celery-done", "done", runner="celery", owner="ann", created_min_ago=3)
    table.add("other-thread", runner="thread", owner="bob", created_min_ago=5, updated_min_ago=2)
    mine = store.create(jobs.JOB_KIND_REPORT, run_name="mine", service="demo", owner="ann")
    finished = store.create(jobs.JOB_KIND_REPORT, run_name="finished", service="demo", owner="ann")
    store.update(finished.job_id, status=jobs.JOB_STATUS_DONE)
    table.rows[finished.job_id]["status"] = "running"  # the final write did not reach the table
    executed = len(table.statements)

    assert store.running_counts("ann") == (4, 2)  # celery-live, mine | celery-queued, other-thread
    assert store.running_counts("bob") == (4, 2)
    assert store.running_counts("eve") == (4, 0)

    counted = table.statements[executed:]
    assert len(counted) == 3 and all(text.startswith("SELECT job_id, owner FROM") for text, _ in counted)
    assert mine.job_id in store._jobs


def test_running_counts_take_their_limits_from_the_environment(table, monkeypatch):
    store = started_store(table)
    table.add("two-minutes", runner="celery", owner="ann", created_min_ago=10, updated_min_ago=2)
    table.add("half-a-minute", runner="celery", owner="ann", created_min_ago=10, updated_min_ago=0.5)
    table.add("queued", runner="celery", owner="ann", created_min_ago=45, started=False)
    assert store.running_counts("ann") == (3, 3)

    monkeypatch.setenv(jobs.STALE_MIN_ENV, "1")
    monkeypatch.setenv(jobs.QUEUE_MAX_MIN_ENV, "30")
    assert store.running_counts("ann") == (1, 1)


def test_running_counts_without_the_table_count_memory(table):
    table.down = True
    store = jobs.JobStore()
    store.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", owner="ann")
    store.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", owner="bob", runner="celery")  # lost: no table

    assert store.running_counts("ann") == (1, 1)


# ---- sweep_stale and restart recovery ------------------------------------------------------------------


def test_sweep_stale_ends_only_stuck_celery_jobs(table, caches):
    store = started_store(table)
    table.add("silent", runner="celery", created_min_ago=40, updated_min_ago=20)
    table.add("queued-long", runner="celery", created_min_ago=130, started=False)
    table.add("queued", runner="celery", created_min_ago=60, started=False)  # silent, but waits in the queue
    table.add("live", runner="celery", created_min_ago=40, updated_min_ago=1)
    table.add("thread-silent", runner="thread", created_min_ago=40, updated_min_ago=20)
    table.add("done-old", "done", runner="celery", created_min_ago=300)
    caches.set(("jobs", ("running",), 20, True), b"{}", 60)

    swept = store.sweep_stale()

    assert sorted(swept) == ["queued-long", "silent"]
    assert table.rows["silent"]["message"] == jobs.STALE_MESSAGE == "Задача прервана: воркер не отвечает"
    assert "heartbeat" in table.rows["silent"]["error"] and "15 min" in table.rows["silent"]["error"]
    assert table.rows["queued-long"]["message"] == jobs.QUEUED_MESSAGE == "Задача прервана: воркер не взял её из очереди"
    assert "queue" in table.rows["queued-long"]["error"] and "120 min" in table.rows["queued-long"]["error"]
    assert {job_id: row["status"] for job_id, row in table.rows.items()} == {
        "silent": "error", "queued-long": "error", "queued": "running", "live": "running",
        "thread-silent": "running", "done-old": "done",
    }
    assert caches.get(("jobs", ("running",), 20, True)) is None and caches.get(("dashboard", ())) is None

    assert store.sweep_stale() == []
    assert store.sweep_stale(timedelta(seconds=30)) == ["live"]
    assert store.sweep_stale(queue_max_age=1800) == ["queued"]
    table.down = True
    assert store.sweep_stale() == []


def test_list_and_get_sweep_at_most_every_30_seconds(table, clock):
    store = started_store(table)  # the first sweep waits one interval after start
    table.add("silent", runner="celery", created_min_ago=40, updated_min_ago=20)
    sweeps = lambda: len(table.texts("started_at IS NOT NULL AND updated_at <"))  # noqa: E731

    clock.value += 10
    assert {r.job_id: r.status for r in store.list(statuses=["running", "error"])} == {"silent": "running"}
    assert sweeps() == 0

    clock.value += 21
    listed = {r.job_id: r for r in store.list(statuses=["running", "error"])}
    assert listed["silent"].status == "error" and listed["silent"].message == jobs.STALE_MESSAGE
    assert sweeps() == 1
    store.list(statuses=["running"])
    assert sweeps() == 1

    # A page polling only /job_status sees a dead worker's job end too.
    table.add("silent-2", runner="celery", created_min_ago=40, updated_min_ago=20)
    clock.value += 19
    assert store.get("silent-2").status == "running" and sweeps() == 1
    clock.value += 12
    assert store.get("silent-2").message == jobs.STALE_MESSAGE and sweeps() == 2
    # A live job does not trigger a sweep, however long since the last one.
    table.add("live", runner="celery", created_min_ago=40, updated_min_ago=1)
    clock.value += 100
    assert store.get("live").status == "running" and sweeps() == 2


def test_restart_recovery_marks_only_thread_rows(table):
    table.add("thread-orphan", runner="thread", created_min_ago=30)
    table.add("celery-running", runner="celery", created_min_ago=30, updated_min_ago=3)
    table.add("celery-queued", runner="celery", created_min_ago=30, started=False)
    store = jobs.JobStore()
    table.add("thread-live", runner="thread", created_min_ago=-1)  # written after this process started

    orphan = store.get("thread-orphan")

    assert orphan.status == "error" and orphan.message == jobs.INTERRUPTED_MESSAGE
    assert [table.rows[j]["status"] for j in ("celery-running", "celery-queued", "thread-live")] == ["running"] * 3
    assert len(table.texts("AND runner = %s AND updated_at < %s")) == 1


# ---- adopt_thread, mark_started, heartbeat --------------------------------------------------------------


def test_adopt_thread_takes_a_job_over_after_a_failed_send(table):
    web = jobs.JobStore()
    job = web.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", owner="ann", runner="celery")

    adopted = web.adopt_thread(job.job_id)

    assert adopted.runner == "thread" and adopted.started_at is not None and web._jobs[job.job_id] == adopted
    assert table.rows[job.job_id]["runner"] == "thread" and table.rows[job.job_id]["started_at"] is not None
    web.update(job.job_id, progress=30, message="Сбор метрик")
    assert web._jobs[job.job_id].progress == 30 and table.rows[job.job_id]["progress"] == 30
    assert web.running_counts("ann") == (1, 1)

    executed = len(table.statements)
    assert web.adopt_thread(job.job_id) is web._jobs[job.job_id] and len(table.statements) == executed
    assert web.adopt_thread("missing") is None


def test_mark_started_and_heartbeat_touch_only_running_rows(table, wall):
    web, worker = jobs.JobStore(), jobs.JobStore()
    job = web.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", message="Инициализация…", runner="celery")
    created = wall.value

    wall.value += timedelta(minutes=3)
    assert worker.mark_started(job.job_id) is True
    row = table.rows[job.job_id]
    assert row["started_at"] == row["updated_at"] == created + timedelta(minutes=3)
    assert (row["message"], row["progress"], row["created_at"]) == ("Инициализация…", 0, created)

    wall.value += timedelta(seconds=30)
    assert worker.heartbeat(job.job_id) is True
    assert row["updated_at"] == created + timedelta(minutes=3, seconds=30)
    assert row["started_at"] == created + timedelta(minutes=3) and row["message"] == "Инициализация…"
    assert job.job_id not in worker._jobs

    worker.update(job.job_id, status=jobs.JOB_STATUS_DONE)
    assert worker.heartbeat(job.job_id) is False and worker.mark_started(job.job_id) is False
    assert worker.mark_started("missing") is False
    table.down = True
    assert worker.heartbeat(job.job_id) is None and worker.mark_started(job.job_id) is None


def test_mark_started_of_a_thread_job_changes_memory(table, wall):
    store = jobs.JobStore()
    job = store.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo")
    wall.value += timedelta(seconds=5)

    assert store.mark_started(job.job_id) is True
    assert store._jobs[job.job_id].started_at == wall.value == table.rows[job.job_id]["started_at"]
    store.update(job.job_id, status=jobs.JOB_STATUS_ERROR)
    assert store.mark_started(job.job_id) is False


def test_a_job_swept_from_the_queue_is_not_started_later(table, wall):
    web, worker = jobs.JobStore(), jobs.JobStore()
    job = web.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", runner="celery")
    wall.value += timedelta(minutes=121)

    assert web.sweep_stale() == [job.job_id]
    assert worker.mark_started(job.job_id) is False  # the task must not run it
    assert web.get(job.job_id).message == jobs.QUEUED_MESSAGE


def test_list_prefers_the_row_of_a_celery_job(table):
    web, worker = jobs.JobStore(), jobs.JobStore()
    job = web.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", runner="celery")
    worker.update(job.job_id, progress=70, message="Анализ ИИ")
    web._jobs[job.job_id] = jobs.JobRecord(job_id=job.job_id, kind="report", runner="celery", updated_at=utc(-5))

    listed = web.list(statuses=["running"])

    assert [(r.job_id, r.progress, r.message) for r in listed] == [(job.job_id, 70, "Анализ ИИ")]


# ---- older tables -----------------------------------------------------------------------------------------


def test_an_older_table_gets_the_new_columns(monkeypatch):
    old = JobsTable(columns=OLD_COLUMNS)
    old.add("finished", "done", created_min_ago=60)
    old.add("orphan", "running", created_min_ago=30)
    monkeypatch.setattr(jobs, "_ts_conn", old.connect)
    store = jobs.JobStore()

    listed = {r.job_id: r for r in store.list(statuses=["running", "done", "error"])}

    assert set(ALL_COLUMNS) <= set(old.columns)
    assert len(old.texts("ALTER TABLE")) == 2
    assert {row["runner"] for row in old.rows.values()} == {"thread"} and {row["owner"] for row in old.rows.values()} == {""}
    assert old.rows["finished"]["started_at"] is None
    assert listed["orphan"].status == "error" and listed["finished"].runner == "thread"

    again = jobs.JobStore()
    again.list(statuses=["running"])
    assert len(old.texts("ALTER TABLE")) == 2  # columns found: nothing to alter


def test_without_alter_rights_the_store_works_as_before(monkeypatch, caplog):
    limited = JobsTable(columns=OLD_COLUMNS + ("dismissed_at",), alter_fails=True)
    limited.add("orphan", "running", created_min_ago=30)
    limited.add("other", "running", created_min_ago=-1)
    monkeypatch.setattr(jobs, "_ts_conn", limited.connect)
    caplog.set_level(logging.WARNING, logger="loadlens_app.jobs")
    store = jobs.JobStore()

    job = store.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", owner="ann")

    assert store._columns_supported is False and store._dismiss_supported is True
    assert "Cannot add owner/runner/started_at" in caplog.text
    assert limited.rows["orphan"]["status"] == "error"  # recovery without the runner filter, as before
    assert limited.rows[job.job_id]["status"] == "running" and "runner" not in limited.rows[job.job_id]
    assert {r.job_id for r in store.list(statuses=["running"])} == {job.job_id, "other"}
    assert store.get("other").runner == "thread"
    store.update(job.job_id, progress=20)
    assert limited.rows[job.job_id]["progress"] == 20

    assert store.running_counts("ann") == (1, 1)  # memory only
    assert not limited.texts("SELECT job_id, owner")
    assert store.sweep_stale() == [] and not limited.texts("started_at IS")
    assert store.heartbeat("other") is True and store.mark_started("other") is True

    store.create(jobs.JOB_KIND_REPORT, run_name="c1", service="demo", runner="celery")
    store.create(jobs.JOB_KIND_REPORT, run_name="c2", service="demo", runner="celery")
    assert caplog.text.count("has no owner/runner/started_at columns") == 1


# ---- configuration -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, minutes",
    [(None, None), ("1", 1), ("0.5", 0.5), (" 30 ", 30), ("abc", None), ("0", None), ("-2", None), ("nan", None),
     ("inf", None), ("1e9", jobs.MAX_ENV_MINUTES)],
)
def test_limits_from_the_environment(monkeypatch, raw, minutes):
    for name in (jobs.STALE_MIN_ENV, jobs.QUEUE_MAX_MIN_ENV):
        if raw is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, raw)
    assert jobs.stale_after() == timedelta(minutes=jobs.DEFAULT_STALE_MIN if minutes is None else minutes)
    assert jobs.queue_max_wait() == timedelta(minutes=jobs.DEFAULT_QUEUE_MAX_MIN if minutes is None else minutes)
    assert (jobs.DEFAULT_STALE_MIN, jobs.DEFAULT_QUEUE_MAX_MIN) == (15, 120)
