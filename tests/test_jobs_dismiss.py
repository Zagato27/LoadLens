"""Clearing the job list: dismissed jobs leave the list, running ones stay, orphans of a restart become errors.

The report_jobs table is simulated by FakeJobsTable, which understands the few statements JobStore sends.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from loadlens_app import jobs  # noqa: E402
from AI.storage_state import StorageUnavailableError  # noqa: E402

COLUMNS = jobs._JOB_COLUMNS


def utc(minutes_ago: float = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)


class FakeJobsTable:
    """report_jobs in memory; ``statements`` keeps the text of every executed statement."""

    def __init__(self, *, has_column=True, alter_fails=False):
        self.rows = {}
        self.has_column = has_column
        self.alter_fails = alter_fails
        self.statements = []

    def add(self, job_id, status, *, minutes_ago=0.0, dismissed=False, kind="report", runner="thread"):
        stamp = utc(minutes_ago)
        self.rows[job_id] = {
            "job_id": job_id, "kind": kind, "run_name": f"run-{job_id}", "service": "demo", "status": status,
            "progress": 50, "message": "stage", "error": None, "report_url": None, "page_url": None, "page_id": None,
            "created_at": stamp, "updated_at": stamp, "dismissed_at": stamp if dismissed else None,
            "owner": "", "runner": runner, "started_at": stamp,
        }

    def connect(self):
        return FakeConn(self)


class FakeConn:
    autocommit = False

    def __init__(self, table):
        self.table = table

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self.table)

    def close(self):
        pass


class FakeCursor:
    def __init__(self, table):
        self.table = table
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = str(query)
        table = self.table
        table.statements.append(text)
        rows = table.rows
        self.result = []
        if "information_schema.columns" in text:
            self.result = ([("dismissed_at",)] if table.has_column else []) + [("owner",), ("runner",), ("started_at",)]
        elif "ALTER TABLE" in text:
            if table.alter_fails:
                raise psycopg2.ProgrammingError("must be owner of table report_jobs")
            table.has_column = True
        elif "CREATE" in text:
            pass
        elif "INSERT INTO" in text:
            values = dict(zip(COLUMNS, params))
            row = rows.setdefault(values["job_id"], {"dismissed_at": None, "owner": "", "runner": "thread", "started_at": None})
            row.update(values)
        elif "SET status = %s, message = %s, error = %s, updated_at = %s" in text:
            status, message, error, now, running, *runner, started = params
            for row in rows.values():
                if row["status"] == running and row["updated_at"] < started and row["runner"] in (runner or [row["runner"]]):
                    row.update(status=status, message=message, error=error, updated_at=now)
                    self.result.append((row["job_id"],))
        elif "SET dismissed_at = %s" in text:
            now, ids, running = params
            for job_id in ids:
                row = rows.get(job_id)
                if row and row["status"] != running and row["dismissed_at"] is None:
                    row["dismissed_at"] = now
                    self.result.append((job_id,))
        elif "SELECT job_id FROM" in text:
            ids, running = params
            self.result = [(job_id,) for job_id in ids if job_id in rows and rows[job_id]["status"] == running]
        elif "WHERE job_id = %s" in text:
            row = rows.get(params[0])
            self.result = [tuple(row[c] for c in COLUMNS)] if row else []
        elif "WHERE status = ANY(%s)" in text:
            statuses, limit = params
            if "dismissed_at" in text and not table.has_column:
                raise psycopg2.ProgrammingError('column "dismissed_at" does not exist')
            visible = [
                row for row in rows.values()
                if row["status"] in statuses and ("dismissed_at IS NULL" not in text or row["dismissed_at"] is None)
            ]
            visible.sort(key=lambda r: r["updated_at"], reverse=True)
            self.result = [tuple(row[c] for c in COLUMNS) for row in visible[:limit]]
        else:
            raise AssertionError(f"unexpected statement: {text}")

    def fetchone(self):
        return self.result[0] if self.result else None

    def fetchall(self):
        return list(self.result)


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    """Only the fake job table answers; any other storage access fails the test."""

    def no_db():
        raise AssertionError("these tests must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)


@pytest.fixture
def table(monkeypatch):
    fake = FakeJobsTable()
    monkeypatch.setattr(jobs, "_ts_conn", fake.connect)
    return fake


def ids(records):
    return [r.job_id for r in records]


# ---- JobStore ------------------------------------------------------------------------------------------


def test_rows_left_running_by_a_stopped_process_become_errors(table):
    table.add("orphan", "running", minutes_ago=30)
    store = jobs.JobStore()
    table.add("neighbour", "running", minutes_ago=-1)  # updated after this process started: left alone

    listed = {r.job_id: r for r in store.list(statuses=["running", "error"])}

    assert listed["orphan"].status == "error"
    assert listed["orphan"].message == jobs.INTERRUPTED_MESSAGE
    assert listed["neighbour"].status == "running"
    assert store.get("orphan").status == "error"


def test_jobs_of_this_process_are_not_marked_interrupted(table):
    store = jobs.JobStore()
    job = store.create(jobs.JOB_KIND_REPORT, run_name="live", service="demo")
    store.list(statuses=["running"])
    store.get(job.job_id)

    assert table.rows[job.job_id]["status"] == "running"
    assert sum("updated_at < %s" in text for text in table.statements) == 1  # one sweep per process


def test_dismiss_hides_finished_jobs_and_keeps_running_ones(table):
    table.add("failed", "error", minutes_ago=5)
    table.add("finished", "done", minutes_ago=4)
    store = jobs.JobStore()
    running = store.create(jobs.JOB_KIND_REPORT, run_name="busy", service="demo")

    dismissed, skipped = store.dismiss(["failed", "finished", running.job_id, "unknown", "failed"])

    assert dismissed == ["failed", "finished"]
    assert skipped == [running.job_id]
    assert ids(store.list(statuses=["running", "error", "done"])) == [running.job_id]
    assert table.rows["failed"]["dismissed_at"] is not None
    assert table.rows[running.job_id]["dismissed_at"] is None
    # Hidden, not deleted: the job page still finds it.
    assert store.get("failed").status == "error"
    # Repeating the request changes nothing.
    assert store.dismiss(["failed"]) == ([], [])


def test_dismiss_reaches_jobs_finished_in_this_process(table):
    store = jobs.JobStore()
    job = store.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo")
    store.update(job.job_id, status=jobs.JOB_STATUS_ERROR, message="Ошибка на этапе «Сбор»")

    assert store.dismiss([job.job_id]) == ([job.job_id], [])
    assert store.list(statuses=["error"]) == []


def test_memory_wins_when_the_table_missed_the_last_update(table):
    store = jobs.JobStore()
    job = store.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo")
    store.update(job.job_id, status=jobs.JOB_STATUS_ERROR)
    table.rows[job.job_id]["status"] = "running"  # the final write did not reach the table

    dismissed, skipped = store.dismiss([job.job_id])

    assert dismissed == [job.job_id] and skipped == []
    assert store.list(statuses=["running", "error"]) == []


def test_without_the_column_the_list_works_and_dismiss_explains(table):
    table.has_column = False
    table.alter_fails = True
    table.add("failed", "error", minutes_ago=5)
    store = jobs.JobStore()

    assert ids(store.list(statuses=["error"])) == ["failed"]
    with pytest.raises(jobs.JobStoreError) as raised:
        store.dismiss(["failed"])
    assert "dismissed_at" in str(raised.value)


def test_existing_column_is_not_altered(table):
    store = jobs.JobStore()
    store.list(statuses=["error"])
    assert not any("ALTER TABLE" in text for text in table.statements)

    missing = FakeJobsTable(has_column=False)
    jobs.JobStore()._ensure_table(missing.connect())
    assert any("ALTER TABLE" in text for text in missing.statements) and missing.has_column


def test_old_finished_jobs_leave_memory_but_stay_reachable(table, monkeypatch):
    store = jobs.JobStore()
    job = store.create(jobs.JOB_KIND_REPORT, run_name="old", service="demo")
    store.update(job.job_id, status=jobs.JOB_STATUS_DONE)
    later = utc(-61)
    monkeypatch.setattr(jobs, "_utcnow", lambda: later)

    store.create(jobs.JOB_KIND_REPORT, run_name="new", service="demo")

    assert job.job_id not in store._jobs
    assert store.get(job.job_id).status == "done"


# ---- HTTP ------------------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, table, tmp_path):
    from app import create_app
    from loadlens_app.auth import InMemoryRepository
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    store = jobs.JobStore()
    monkeypatch.setattr(dashboard, "job_store", store)
    flask_app = create_app(auth_repository=InMemoryRepository())
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_dismiss_endpoint(client, table):
    table.add("failed", "error", minutes_ago=5)
    table.add("busy", "running", minutes_ago=-1)

    response = client.post("/jobs/dismiss", json={"job_ids": ["failed", "busy"]})

    assert response.status_code == 200
    assert response.get_json() == {"dismissed": ["failed"], "skipped_running": ["busy"]}
    listed = client.get("/jobs", query_string={"status": "running,error"}).get_json()["jobs"]
    assert [j["job_id"] for j in listed] == ["busy"]
    assert client.get("/job_status/failed").get_json()["status"] == "error"


@pytest.mark.parametrize(
    "body",
    [None, {}, {"job_ids": []}, {"job_ids": "failed"}, {"job_ids": [1]}, {"job_ids": [" "]}, {"job_ids": ["x" * 65]}, {"job_ids": ["a"] * 101}],
)
def test_dismiss_endpoint_validates_the_body(client, body):
    response = client.post("/jobs/dismiss", json=body) if body is not None else client.post("/jobs/dismiss")
    assert response.status_code == 400 and response.get_json()["error"]


def test_dismiss_endpoint_without_the_column(client, table):
    table.has_column = False
    table.alter_fails = True
    table.add("failed", "error", minutes_ago=5)
    response = client.post("/jobs/dismiss", json={"job_ids": ["failed"]})
    assert response.status_code == 503 and response.get_json()["code"] == "jobs_dismiss_unavailable"


def test_dismiss_endpoint_while_storage_is_down(client, monkeypatch):
    def down():
        raise StorageUnavailableError("connection refused")

    monkeypatch.setattr(jobs, "_ts_conn", down)
    response = client.post("/jobs/dismiss", json={"job_ids": ["failed"]})
    assert response.status_code == 503 and response.get_json()["code"] == "storage_unavailable"


def test_only_engineers_dismiss_and_the_action_is_audited(monkeypatch, table, tmp_path):
    from app import create_app
    from loadlens_app.auth import Actor, InMemoryRepository, passwords, web
    from loadlens_app.blueprints import dashboard

    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    monkeypatch.setattr(passwords, "_method", lambda: "pbkdf2:sha256:1000")
    monkeypatch.setattr(passwords, "_dummy_hash", "")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(dashboard, "job_store", jobs.JobStore())
    repo = InMemoryRepository()
    flask_app = create_app(auth_repository=repo)
    service = flask_app.extensions[web.EXTENSION_KEY]
    for name, role in (("vera", "viewer"), ("emil", "engineer")):
        service.create_user(username=name, role=role, password="Correct-Horse-1", must_change_password=False, actor=Actor(name="test"))
    table.add("failed", "error", minutes_ago=5)

    def signed_in(name):
        client = flask_app.test_client()
        token = client.get("/auth/me").get_json()["csrf_token"]
        reply = client.post("/login", json={"username": name, "password": "Correct-Horse-1"}, headers={"X-CSRF-Token": token})
        return client, reply.get_json()["csrf_token"]

    viewer, viewer_token = signed_in("vera")
    assert viewer.post("/jobs/dismiss", json={"job_ids": ["failed"]}, headers={"X-CSRF-Token": viewer_token}).status_code == 403

    engineer, engineer_token = signed_in("emil")
    no_csrf = engineer.post("/jobs/dismiss", json={"job_ids": ["failed"]})
    assert no_csrf.status_code == 403 and no_csrf.get_json()["code"] == "csrf_failed"
    ok = engineer.post("/jobs/dismiss", json={"job_ids": ["failed"]}, headers={"X-CSRF-Token": engineer_token})
    assert ok.status_code == 200 and ok.get_json()["dismissed"] == ["failed"]

    events = [e for e in repo.list_audit(limit=50) if e.action == "jobs.dismiss"]
    assert len(events) == 1 and events[0].actor_name == "emil" and events[0].details["dismissed"] == 1
