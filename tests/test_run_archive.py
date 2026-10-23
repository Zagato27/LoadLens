# SPDX-License-Identifier: Apache-2.0
"""Load test runs in the archive (plans/run-archive-plan.md).

A finished run no longer holds its name; the archive's Runs tab lists runs without commands for every role;
finished runs are deleted one by one or in bulk, optionally with their reports, found by report id.
"""

import copy
import json
import re
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from loadlens_app import core
from loadlens_app.auth import Actor, Role, policy
from loadlens_app.auth import passwords
from loadlens_app.load_tests import executors, service, store
from loadlens_app.load_tests.store import InstanceRecord, MemoryLoadTestStore, PgLoadTestStore, RunRecord
from tests.i18n_helpers import use_locale
from tests.js_harness import run_js
from tests.test_load_tests_api import FORM, GENERATORS, FakeExecutor

T0 = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
PASSWORD = "Correct-Horse-1"


def _no_db(monkeypatch):
    """Every storage connection fails loudly: these tests use the memory store and fake report lookups."""

    def no_db():
        raise AssertionError("this test must not open a database connection")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)


def _run(run_id, name="nightly", status=store.RUN_DONE, service_id="checkout", owner="eng", minutes=0, **extra):
    created = T0 + timedelta(minutes=minutes)
    inst = InstanceRecord(index=0, name="rest", tag=f"{name}-rest", generator="gen-1", tool="k6", script="secret/path.js",
                          params="--vus 5", env={"TOKEN": "s3cret"}, command="k6 run TOKEN=*** secret/path.js")
    return RunRecord(run_id=run_id, run_name=name, run_tag=name, service=service_id, status=status, owner=owner,
                     instances=[inst], created_at=created, updated_at=created, started_at=created,
                     finished_at=None if status in store.RUN_ACTIVE else created + timedelta(minutes=5), **extra)


# ---- memory store ---------------------------------------------------------------------------------------


def test_a_finished_run_leaves_its_name_free():
    memory = MemoryLoadTestStore()
    memory.create(_run("a", status=store.RUN_DONE))
    memory.create(_run("b", name="soak", status=store.RUN_FAILED))
    assert not memory.name_in_use("nightly", "checkout")
    assert not memory.name_in_use("soak", "checkout")
    memory.create(_run("c", status=store.RUN_RUNNING))
    assert memory.name_in_use("nightly", "checkout")
    assert not memory.name_in_use("nightly", "payments")


def test_memory_list_pages_searches_and_counts():
    memory = MemoryLoadTestStore()
    for i in range(5):
        memory.create(_run(f"r{i}", name=f"Night_{i}", minutes=i))
    memory.create(_run("x", name="soak", minutes=10, service_id="payments"))
    assert [r.run_id for r in memory.list(limit=2)] == ["x", "r4"]
    assert [r.run_id for r in memory.list(limit=2, offset=2)] == ["r3", "r2"]
    assert [r.run_id for r in memory.list(query="night_1")] == ["r1"]
    assert memory.count(query="NIGHT") == 5
    assert memory.count(services=["payments"]) == 1
    assert memory.count(statuses=store.RUN_ACTIVE) == 0


def test_memory_delete_keeps_active_runs_and_reports_what_went():
    memory = MemoryLoadTestStore()
    memory.create(_run("done"))
    memory.create(_run("going", status=store.RUN_REPORTING))
    assert memory.delete(["done", "going", "missing", "done"]) == ["done"]
    assert memory.get("done") is None and memory.get("going") is not None
    assert {r.run_id for r in memory.get_many(["going", "nope"])} == {"going"}


def test_memory_delete_service_only_takes_finished_runs_of_that_service():
    memory = MemoryLoadTestStore()
    memory.create(_run("a"))
    memory.create(_run("b", status=store.RUN_CANCELLED))
    memory.create(_run("c", status=store.RUN_RUNNING))
    memory.create(_run("d", service_id="payments"))
    assert memory.delete_service("checkout") == 2
    assert {r.run_id for r in memory.list()} == {"c", "d"}


# ---- PostgreSQL store -----------------------------------------------------------------------------------


class Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.conn.log.append((" ".join(_plain(query).split()), params))
        self.rowcount = self.conn.rowcount

    def fetchone(self):
        return self.conn.one

    def fetchall(self):
        return self.conn.rows


class Conn:
    def __init__(self, one=None, rows=(), rowcount=0):
        self.log = []
        self.one = one
        self.rows = list(rows)
        self.rowcount = rowcount

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return Cursor(self)

    def close(self):
        pass


@pytest.fixture
def pg(monkeypatch):
    conns = []

    def connect(**kwargs):
        conn = Conn(**kwargs)
        conns.append(conn)
        monkeypatch.setattr(core, "_ts_conn", lambda: conn)
        return conn

    monkeypatch.setattr(store, "_table_ref", lambda: ("public", "load_test_runs"))
    pg_store = PgLoadTestStore()
    pg_store._ensured = True
    return types.SimpleNamespace(store=pg_store, connect=connect)


def _plain(part):
    """Text of a psycopg2 ``sql`` composition without a connection (identifiers in double quotes)."""
    if isinstance(part, store.sql.Identifier):
        return ".".join(f'"{s}"' for s in part.strings)
    if isinstance(part, store.sql.SQL):
        return part.string
    if isinstance(part, store.sql.Composed):
        return "".join(_plain(p) for p in part.seq)
    return str(part)


def test_pg_name_in_use_asks_for_active_runs_only(pg):
    conn = pg.connect(one=None)
    assert pg.store.name_in_use("nightly", "checkout") is False
    text, params = conn.log[-1]
    assert "run_name = %s AND service = %s AND status = ANY(%s)" in text
    assert params == ("nightly", "checkout", list(store.RUN_ACTIVE))


def test_pg_delete_checks_the_status_in_the_statement(pg):
    conn = pg.connect(rows=[("a",)])
    assert pg.store.delete(["a", "b", "a"]) == ["a"]
    text, params = conn.log[-1]
    assert text == 'DELETE FROM "public"."load_test_runs" WHERE run_id = ANY(%s) AND status = ANY(%s) RETURNING run_id'
    assert params == (["a", "b"], list(store.RUN_TERMINAL))
    assert pg.store.delete([]) == []


def test_pg_delete_service_keeps_active_runs(pg):
    conn = pg.connect(rowcount=3)
    assert pg.store.delete_service("checkout") == 3
    text, params = conn.log[-1]
    assert text == 'DELETE FROM "public"."load_test_runs" WHERE service = %s AND status = ANY(%s)'
    assert params == ("checkout", list(store.RUN_TERMINAL))


def test_pg_list_and_count_escape_like_wildcards(pg):
    conn = pg.connect(rows=[], one=(7,))
    pg.store.list(limit=20, offset=40, services=["checkout"], query="50%_x\\")
    text, params = conn.log[-1]
    assert "WHERE status = ANY(%s) AND service = ANY(%s) AND run_name ILIKE %s ORDER BY created_at DESC, run_id DESC LIMIT %s OFFSET %s" in text
    assert params[2] == "%50\\%\\_x\\\\%" and params[3:] == (20, 40)
    assert pg.store.count(query="a") == 7
    text, params = conn.log[-1]
    assert text.startswith('SELECT count(*) FROM "public"."load_test_runs" WHERE status = ANY(%s) AND run_name ILIKE %s')


# ---- HTTP API with sign-in off (administrator) ------------------------------------------------------------


@pytest.fixture
def env(monkeypatch, tmp_path):
    from loadlens_app.blueprints import config_api, dashboard
    from loadlens_app.blueprints import load_tests as blueprint

    _no_db(monkeypatch)
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setitem(core.CONFIG, "load_generators", copy.deepcopy(GENERATORS))
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setitem(core.CONFIG, "limits", {})
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    reports = {}
    deleted_reports = []

    def lookup(ids):
        if reports.get("__fail__"):
            raise RuntimeError("database is down")
        return {i: reports[i] for i in ids if i in reports}

    def delete_report(name):
        if name in reports.get("__undeletable__", ()):
            raise RuntimeError("locked")
        deleted_reports.append(name)

    monkeypatch.setattr(dashboard, "_lookup_report_refs", lookup)
    monkeypatch.setattr(core, "_delete_run_data", delete_report)
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: name == "has-report")
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(service, "_installed_lt_packs", lambda area, service_id: ["lt-k6-prometheus-rw"])
    monkeypatch.setattr(executors, "executor_factory", lambda generator: FakeExecutor())
    monkeypatch.setattr(config_api, "_bootstrap_service_configs", lambda area, service_id: None)
    area_services = []
    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: list(area_services))
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), store=memory, reports=reports,
                                 deleted_reports=deleted_reports, area_services=area_services)


def test_launch_accepts_the_name_of_a_finished_run_but_not_of_a_running_one_or_a_report(env):
    env.store.create(_run("old", name="nightly", status=store.RUN_DONE))
    assert env.client.post("/load_tests", json=FORM).status_code == 200  # FORM is named "nightly"
    second = env.client.post("/load_tests", json=FORM)
    assert second.status_code == 400 and "«nightly»" in second.get_json()["message"] and "уже занято" in second.get_json()["message"]
    taken = env.client.post("/load_tests", json={**FORM, "run_name": "has-report"})
    assert taken.status_code == 400 and "уже занято" in taken.get_json()["message"]


def test_history_lists_runs_without_commands_or_parameters(env):
    env.store.create(_run("r1", minutes=1))
    resp = env.client.get("/load_tests/history")
    assert resp.status_code == 200 and resp.headers["X-Total-Count"] == "1"
    row = resp.get_json()["runs"][0]
    assert set(row) == {
        "run_id", "run_name", "service", "test_type", "status", "status_title", "active", "owner", "created_at",
        "started_at", "finished_at", "instances", "tools", "report", "has_report", "can_delete",
    }
    assert row["instances"] == 1 and row["tools"] == ["k6"] and row["status_title"] == "Завершён"
    assert row["can_delete"] is True and row["report"] is None and row["has_report"] is False
    text = json.dumps(resp.get_json())
    for secret in ("s3cret", "secret/path.js", "--vus", "TOKEN", "gen-1", "10.0.0.5"):
        assert secret not in text


def test_history_filters_pages_and_rejects_unknown_statuses(env):
    for i in range(3):
        env.store.create(_run(f"d{i}", name=f"day-{i}", minutes=i))
    env.store.create(_run("f", name="broken", status=store.RUN_FAILED, minutes=5))
    env.store.create(_run("g", name="going", status=store.RUN_RUNNING, minutes=6, service_id="payments"))
    names = lambda resp: [r["run_name"] for r in resp.get_json()["runs"]]  # noqa: E731
    assert names(env.client.get("/load_tests/history?limit=2&offset=1")) == ["broken", "day-2"]
    assert names(env.client.get("/load_tests/history?status=active")) == ["going"]
    assert names(env.client.get("/load_tests/history?status=failed,done&q=DAY-1")) == ["day-1"]
    assert names(env.client.get("/load_tests/history?service=payments")) == ["going"]
    resp = env.client.get("/load_tests/history?status=done&limit=1")
    assert resp.headers["X-Total-Count"] == "3" and resp.get_json()["total"] == 3
    bad = env.client.get("/load_tests/history?status=paused")
    assert bad.status_code == 400 and "paused" in bad.get_json()["message"]
    assert env.client.get("/load_tests/history?limit=x").status_code == 400
    # A service outside the active area answers an empty page.
    env.area_services.extend(["checkout"])
    assert env.client.get("/load_tests/history?service=payments").get_json() == {"runs": [], "total": 0}
    assert {r["service"] for r in env.client.get("/load_tests/history").get_json()["runs"]} == {"checkout"}


def test_history_finds_the_report_by_id_under_its_current_name(env):
    env.store.create(_run("r1", report_url="/reports/rid-1", minutes=1))
    env.store.create(_run("r2", report_url="/reports/rid-2", minutes=2))
    env.reports["rid-1"] = {"run_id": "rid-1", "run_name": "renamed", "service": "checkout"}
    rows = {r["run_id"]: r for r in env.client.get("/load_tests/history").get_json()["runs"]}
    assert rows["r1"]["report"] == {"url": "/reports/rid-1", "run_name": "renamed"} and rows["r1"]["has_report"] is True
    assert rows["r2"]["report"] is None and rows["r2"]["has_report"] is False
    env.reports["__fail__"] = True
    rows = {r["run_id"]: r for r in env.client.get("/load_tests/history").get_json()["runs"]}
    assert rows["r2"]["has_report"] is None and rows["r2"]["report"] == {"url": "/reports/rid-2", "run_name": ""}


def test_report_id_comes_from_the_report_address_only():
    assert service.report_id_of(_run("a", report_url="/reports/abc%2F1")) == "abc/1"
    assert service.report_id_of(_run("a", report_url="/reports/svc/name")) == ""
    assert service.report_id_of(_run("a", report_url="https://wiki/pages/1")) == ""
    assert service.report_id_of(_run("a")) == ""


def test_bulk_delete_takes_finished_runs_and_explains_the_rest(env):
    env.store.create(_run("done"))
    env.store.create(_run("going", status=store.RUN_RUNNING))
    resp = env.client.post("/load_tests/runs/delete", json={"run_ids": ["done", "going", "missing"]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["deleted"] == [{"run_id": "done", "run_name": "nightly", "service": "checkout"}]
    assert [(s["run_id"], s["code"]) for s in data["skipped"]] == [("going", "active"), ("missing", "not_found")]
    assert "остановите" in data["skipped"][0]["message"]
    assert data["reports_deleted"] == [] and env.deleted_reports == []
    assert env.store.get("done") is None and env.store.get("going") is not None


def test_bulk_delete_validates_the_request(env):
    post = lambda body: env.client.post("/load_tests/runs/delete", json=body)  # noqa: E731
    assert post({"run_ids": []}).status_code == 400
    assert post({"run_ids": "a"}).status_code == 400
    assert post({"run_ids": [1]}).status_code == 400
    assert post({"run_ids": [f"r{i}" for i in range(101)]}).status_code == 400
    assert env.client.post("/load_tests/runs/delete", data="x", content_type="text/plain").status_code == 400


def test_delete_with_reports_removes_each_report_once_under_its_current_name(env):
    env.store.create(_run("a", report_url="/reports/rid-1"))
    env.store.create(_run("b", report_url="/reports/rid-1"))
    env.store.create(_run("c", report_url="/reports/rid-gone"))
    env.store.create(_run("d"))
    env.reports["rid-1"] = {"run_id": "rid-1", "run_name": "renamed", "service": "checkout"}
    data = env.client.post("/load_tests/runs/delete", json={"run_ids": ["a", "b", "c", "d"], "with_reports": True}).get_json()
    assert len(data["deleted"]) == 4 and data["reports_deleted"] == ["renamed"] and data["reports_failed"] == []
    assert env.deleted_reports == ["renamed"]


def test_a_report_that_cannot_be_deleted_is_reported_and_the_run_still_goes(env):
    env.store.create(_run("a", report_url="/reports/rid-1"))
    env.reports.update({"rid-1": {"run_id": "rid-1", "run_name": "locked-report", "service": "checkout"},
                        "__undeletable__": ["locked-report"]})
    data = env.client.delete("/load_tests/runs/a?with_report=1").get_json()
    assert data["status"] == "ok" and data["report_deleted"] is False
    assert data["reports_failed"][0]["run_name"] == "locked-report" and "locked-report" in data["reports_failed"][0]["message"]
    assert env.store.get("a") is None


def test_single_delete_answers_404_409_and_200(env):
    env.store.create(_run("done", report_url="/reports/rid-1"))
    env.store.create(_run("going", status=store.RUN_STOPPING))
    env.reports["rid-1"] = {"run_id": "rid-1", "run_name": "nightly", "service": "checkout"}
    assert env.client.delete("/load_tests/runs/missing").status_code == 404
    going = env.client.delete("/load_tests/runs/going")
    assert going.status_code == 409 and going.get_json()["code"] == "conflict"
    done = env.client.delete("/load_tests/runs/done")
    assert done.status_code == 200 and done.get_json() == {"status": "ok", "run_id": "done", "report_deleted": False, "reports_failed": []}
    # Without with_report the report stays.
    assert env.deleted_reports == []


def test_deleting_a_service_deletes_its_finished_runs_and_survives_a_store_error(monkeypatch):
    monkeypatch.setattr(core, "_load_settings_runtime_data", lambda: {})
    monkeypatch.setattr(core, "_load_metrics_runtime_data", lambda: {})
    monkeypatch.setattr(core, "_ts_conn", lambda: Conn())
    memory = MemoryLoadTestStore()
    memory.create(_run("a"))
    memory.create(_run("b", status=store.RUN_RUNNING))
    monkeypatch.setattr(store, "_store", memory)
    core._delete_service_data("shop", "checkout")
    assert [r.run_id for r in memory.list()] == ["b"]

    class Broken(MemoryLoadTestStore):
        def delete_service(self, service_id):
            raise RuntimeError("database is down")

    monkeypatch.setattr(store, "_store", Broken())
    core._delete_service_data("shop", "checkout")  # does not raise


# ---- roles ------------------------------------------------------------------------------------------------


def test_policy_rules_of_the_runs_tab():
    assert policy.POLICY["load_tests.history"].role == Role.VIEWER
    for endpoint in ("load_tests.delete_run", "load_tests.delete_runs"):
        rule = policy.POLICY[endpoint]
        assert rule.role == Role.ENGINEER and rule.session_only and rule.audit == "load_test.delete"
    # The full list keeps commands and stays with engineers.
    assert policy.POLICY["load_tests.list_runs"].role == Role.ENGINEER


@pytest.fixture
def signed_in(monkeypatch, tmp_path):
    from loadlens_app.auth import InMemoryRepository
    from loadlens_app.auth import web
    from loadlens_app.blueprints import dashboard
    from loadlens_app.blueprints import load_tests as blueprint
    from tests.test_auth import Browser

    _no_db(monkeypatch)
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setitem(core.CONFIG, "load_generators", copy.deepcopy(GENERATORS))
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    monkeypatch.setattr(passwords, "_method", lambda: "pbkdf2:sha256:1000")
    monkeypatch.setattr(passwords, "_dummy_hash", "")
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    reports = {"rid-1": {"run_id": "rid-1", "run_name": "nightly", "service": "checkout"}}
    deleted_reports = []
    monkeypatch.setattr(dashboard, "_lookup_report_refs", lambda ids: {i: reports[i] for i in ids if i in reports})
    monkeypatch.setattr(core, "_delete_run_data", deleted_reports.append)
    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: [])
    from app import create_app

    app = create_app(auth_repository=InMemoryRepository())
    app.config["TESTING"] = True
    auth_service = app.extensions[web.EXTENSION_KEY]

    def login(role):
        name = f"{role}-user"
        auth_service.create_user(username=name, role=Role.parse(role), password=PASSWORD,
                                 must_change_password=False, actor=Actor(name="test"))
        browser = Browser(app)
        assert browser.login(name, PASSWORD).status_code == 200
        browser.username = name
        return browser

    return types.SimpleNamespace(login=login, store=memory, deleted_reports=deleted_reports)


def test_viewers_read_the_runs_tab_but_cannot_delete(signed_in):
    signed_in.store.create(_run("a", owner="someone", report_url="/reports/rid-1"))
    viewer = signed_in.login("viewer")
    rows = viewer.get("/load_tests/history").get_json()["runs"]
    assert rows[0]["can_delete"] is False and rows[0]["report"]["url"] == "/reports/rid-1"
    assert viewer.get("/load_tests/runs").status_code == 403
    assert viewer.post("/load_tests/runs/delete", json={"run_ids": ["a"]}).status_code == 403
    assert viewer.delete("/load_tests/runs/a").status_code == 403
    assert signed_in.store.get("a") is not None


def test_engineers_delete_their_own_runs_and_admins_anyone_s(signed_in):
    engineer = signed_in.login("engineer")
    signed_in.store.create(_run("mine", owner=engineer.username, report_url="/reports/rid-1"))
    signed_in.store.create(_run("theirs", owner="someone"))
    rows = {r["run_id"]: r["can_delete"] for r in engineer.get("/load_tests/history").get_json()["runs"]}
    assert rows == {"mine": True, "theirs": False}
    forbidden = engineer.post("/load_tests/runs/delete", json={"run_ids": ["mine"], "with_reports": True})
    assert forbidden.status_code == 403 and signed_in.store.get("mine") is not None
    data = engineer.post("/load_tests/runs/delete", json={"run_ids": ["mine", "theirs"]}).get_json()
    assert [r["run_id"] for r in data["deleted"]] == ["mine"]
    assert [(s["run_id"], s["code"]) for s in data["skipped"]] == [("theirs", "forbidden")]
    assert engineer.delete("/load_tests/runs/theirs").status_code == 403
    without_csrf = engineer.client.delete("/load_tests/runs/theirs")
    assert without_csrf.status_code in (400, 403) and signed_in.store.get("theirs") is not None

    admin = signed_in.login("admin")
    signed_in.store.create(_run("again", owner="someone", report_url="/reports/rid-1"))
    data = admin.post("/load_tests/runs/delete", json={"run_ids": ["theirs", "again"], "with_reports": True}).get_json()
    assert {r["run_id"] for r in data["deleted"]} == {"theirs", "again"} and data["reports_deleted"] == ["nightly"]
    assert signed_in.deleted_reports == ["nightly"]


# ---- pages ------------------------------------------------------------------------------------------------


def test_archive_page_has_reports_and_runs_tabs(env):
    use_locale(env.client, "en")
    html = env.client.get("/reports").get_data(as_text=True)
    assert re.search(r'<div class="tabs archive-tabs" role="tablist" aria-label="Archive sections">', html)
    assert 'id="tabReports" data-tab="reports" aria-controls="reportsPanel" aria-selected="true">Reports</button>' in html
    assert 'id="tabRuns" data-tab="runs" aria-controls="runsPanel" aria-selected="false" tabindex="-1">Runs</button>' in html
    assert '<section id="runsPanel" role="tabpanel" aria-labelledby="tabRuns" hidden>' in html
    runs_table = re.search(r'<table[^>]*id="ltRunsTable".*?</table>', html, re.S).group(0)
    heads = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", runs_table, re.S)]
    assert heads == ["Select for deletion", "Run", "Service", "Status", "Started", "Instances", "Started by", "Report", "Actions"]
    assert re.search(r'js/archive\.js[^"]*"></script>\s*<script defer src="[^"]*js/archive_runs\.js', html)
    use_locale(env.client, "ru")
    html = env.client.get("/reports").get_data(as_text=True)
    assert ">Запуски</button>" in html and '<span class="archive-filter-key">статус:</span>' in html


def test_launch_page_links_to_the_runs_tab(env):
    html = env.client.get("/launch").get_data(as_text=True)
    assert '<a class="runs-all" href="/reports?tab=runs">Все запуски</a>' in html


# ---- archive_runs.js and archive.js helpers under Node ------------------------------------------------------

JS = ["static/js/common.js", "static/js/ui.js", "static/js/archive.js", "static/js/archive_runs.js"]
RUNS = "LL.archiveRuns._internals"
ARCHIVE = "LL.archive._internals"


def test_runs_tab_state_round_trips_through_the_address():
    state = run_js(JS, f"{RUNS}.parseQuery('?tab=runs&q=%20night%20&service=checkout&status=failed&page=3')")
    assert state == {"page": 3, "q": "night", "service": "checkout", "status": "failed"}
    assert run_js(JS, f"{RUNS}.parseQuery('?status=paused&page=0')") == {"page": 1, "q": "", "service": "", "status": ""}
    assert run_js(JS, f"{RUNS}.buildQuery({RUNS}.parseQuery('?q=a b&status=active&page=2'))") == "?tab=runs&q=a+b&status=active&page=2"
    assert run_js(JS, f"{RUNS}.buildHistoryUrl({{page: 3, q: 'x', service: '', status: 'done'}})") == \
        "/load_tests/history?q=x&status=done&offset=40&limit=20"


def test_runs_tab_texts():
    assert run_js(JS, f"{RUNS}.instancesText({{instances: 2, tools: ['k6', 'JMeter']}})") == {"text": "2 инст.", "note": "k6, JMeter"}
    assert run_js(JS, f"{RUNS}.statusFilterLabel('active')") == "Идут сейчас"
    assert run_js(JS, f"{RUNS}.statusFilterLabel('failed')") == "Не запустился"
    assert run_js(JS, f"{RUNS}.statusTone('failed')") == "fail" and run_js(JS, f"{RUNS}.statusTone('x')") == "na"
    duration = f"{RUNS}.runDuration({{started_at: '2026-10-20T10:00:00Z', finished_at: '2026-10-20T10:05:00Z'}})"
    assert run_js(JS, duration) == run_js(JS, "LL.formatDuration(300000)")
    assert run_js(JS, f"{RUNS}.runDuration({{started_at: '2026-10-20T10:00:00Z', active: false}})") == ""


def test_report_link_of_a_run():
    assert run_js(JS, f"{RUNS}.reportLink({{report: {{url: '/reports/1', run_name: 'r'}}, has_report: true}})") == \
        {"url": "/reports/1", "text": "r", "title": ""}
    unchecked = run_js(JS, f"{RUNS}.reportLink({{report: {{url: '/reports/1', run_name: ''}}, has_report: null}})")
    assert unchecked["text"] == "отчёт" and "не ответила" in unchecked["title"]
    assert run_js(JS, f"{RUNS}.reportLink({{report: null, has_report: false}})") is None


def test_delete_dialog_offers_reports_to_admins_only():
    runs = json.dumps([{"run_name": "a", "has_report": True}, {"run_name": "b", "has_report": False}])
    admin = run_js(JS, f"{RUNS}.deletePlan({runs}, true)")
    assert admin == {"title": "Удалить запуски", "message": admin["message"], "offerReports": True, "reports": 1}
    assert "2" in admin["message"] and "останутся" in admin["message"]
    assert run_js(JS, f"{RUNS}.deletePlan({runs}, false)")["offerReports"] is False
    one = run_js(JS, f"{RUNS}.deletePlan([{{run_name: 'nightly', has_report: false}}], true)", locale="en")
    assert one["title"] == "Delete run" and "“nightly”" in one["message"] and one["offerReports"] is False


def test_delete_summary():
    ok = run_js(JS, f"{RUNS}.deleteSummary({{deleted: [{{}}, {{}}], skipped: [], reports_deleted: ['r'], reports_failed: []}})")
    assert ok == {"text": "Удалено запусков: 2 · отчётов: 1", "tone": "ok"}
    mixed = run_js(JS, f"{RUNS}.deleteSummary({{deleted: [{{}}], skipped: [{{message: 'идёт'}}], reports_deleted: [], reports_failed: []}})")
    assert mixed == {"text": "Удалено запусков: 1 · пропущено: 1. идёт", "tone": "info"}
    none = run_js(JS, f"{RUNS}.deleteSummary({{deleted: [], skipped: [{{message: 'x'}}], reports_deleted: [], reports_failed: []}})")
    assert none["tone"] == "error"


def test_archive_finds_launched_runs_by_report_id_then_by_newest_name():
    runs = json.dumps([
        {"run_id": "new", "service": "s", "run_name": "nightly", "report_url": "/reports/rid-2"},
        {"run_id": "old", "service": "s", "run_name": "nightly", "report_url": "/reports/rid-1"},
        {"run_id": "plain", "service": "s", "run_name": "smoke", "report_url": None},
    ])
    found = run_js(JS, (
        f"(() => {{ const i = {ARCHIVE}.launchedIndex({runs}); return ["
        f"{ARCHIVE}.launchedFor(i, {{run_id: 'rid-1', service: 's', run_name: 'renamed'}}),"
        f"{ARCHIVE}.launchedFor(i, {{run_id: 'other', service: 's', run_name: 'nightly'}}),"
        f"{ARCHIVE}.launchedFor(i, {{run_id: '', service: 's', run_name: 'smoke'}}),"
        f"{ARCHIVE}.launchedFor(i, {{run_id: 'x', service: 's', run_name: 'none'}})]; }})()"
    ))
    assert found == ["old", "new", "plain", None]
    assert run_js(JS, f"{ARCHIVE}.reportIdFromUrl('/reports/a%2Fb')") == "a/b"
    assert run_js(JS, f"{ARCHIVE}.reportIdFromUrl('/reports/svc/name')") == ""
    assert run_js(JS, f"[{ARCHIVE}.tabFromQuery('?tab=runs'), {ARCHIVE}.tabFromQuery('?tab=x'), {ARCHIVE}.tabFromQuery('')]") == \
        ["runs", "reports", "reports"]
