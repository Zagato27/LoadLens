# SPDX-License-Identifier: Apache-2.0
"""Polling and static files (plans/perf-plan.md, agent 6): TTL cache of the polled answers, idle watcher, static URLs.

Offline: the database is a fake that counts statements, the load test store and the job table live in memory.
The response cache is off by default in tests (tests/conftest.py); the route tests here switch it on.
"""

import copy
import hashlib
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before anything that imports update_page)
from loadlens_app import core, jobs, response_cache, static_assets  # noqa: E402
from loadlens_app.auth import Role  # noqa: E402
from loadlens_app.load_tests import executors, service, store, watcher  # noqa: E402
from loadlens_app.load_tests.store import MemoryLoadTestStore  # noqa: E402
from loadlens_app.response_cache import TTLCache  # noqa: E402

TEMPLATES = ROOT_DIR / "templates"
GENERATORS = {
    "gen-1": {
        "title": "Gen 1",
        "type": "ssh",
        "ssh": {"host": "10.0.0.5", "port": 22, "username": "loadlens", "private_key": "-----BEGIN KEY-----\nabc\n-----END KEY-----", "host_key": "ssh-ed25519 AAAA"},
        "work_dir": "/opt/loadlens/runs",
        "scripts_dir": "/opt/tests",
        "tools": {},
    },
}
FORM = {
    "service": "checkout",
    "run_name": "nightly",
    "instances": [{"name": "rest", "generator": "gen-1", "tool": "k6", "script": "a.js", "params": "--vus 5"}],
}


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


# ---- TTLCache -----------------------------------------------------------------------------------


def test_ttl_cache_serves_until_the_ttl_ends():
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    assert cache.get_or_set(("a", 1), 10, compute) == 1
    clock.now += 9.9
    assert cache.get_or_set(("a", 1), 10, compute) == 1
    clock.now += 0.2
    assert cache.get_or_set(("a", 1), 10, compute) == 2
    assert cache.get_or_set(("a", 2), 10, compute) == 3  # another key, its own entry
    assert (cache.hits, cache.misses) == (1, 3)


def test_ttl_cache_invalidates_by_prefix():
    cache = TTLCache(clock=FakeClock())
    for key in (("jobs", "running"), ("jobs", "error"), ("dashboard", ()), ("active", ("svc",), 5)):
        cache.set(key, "x", 60)
    assert cache.invalidate("jobs") == 2
    assert cache.get(("jobs", "running")) is None and cache.get(("dashboard", ())) == "x"
    assert cache.invalidate(("active", ("svc",))) == 1
    assert cache.invalidate(("active", ("other",))) == 0
    cache.set(("jobs", "running"), "y", 60)
    assert cache.invalidate() == 2 and len(cache) == 0


def test_ttl_cache_keeps_at_most_max_entries_and_drops_expired_ones():
    clock = FakeClock()
    cache = TTLCache(max_entries=256, clock=clock)
    for i in range(300):
        cache.set(("k", i), i, 60)
    assert len(cache) == 256
    assert cache.get(("k", 0)) is None and cache.get(("k", 299)) == 299  # the oldest went first
    clock.now += 61
    cache.set(("k", "new"), "fresh", 60)
    assert len(cache) == 1  # the expired ones were dropped by the insert


def test_ttl_cache_skips_values_that_are_not_cacheable_and_errors():
    cache = TTLCache(clock=FakeClock())
    answers = iter([None, "ok", "later"])
    assert cache.get_or_set(("x",), 10, lambda: next(answers), cacheable=lambda v: v is not None) is None
    assert cache.get_or_set(("x",), 10, lambda: next(answers), cacheable=lambda v: v is not None) == "ok"
    assert cache.get_or_set(("x",), 10, lambda: next(answers)) == "ok"

    def boom():
        raise RuntimeError("db down")

    with pytest.raises(RuntimeError):
        cache.get_or_set(("y",), 10, boom)
    assert cache.get_or_set(("y",), 10, lambda: "recovered") == "recovered"


def test_ttl_cache_does_not_store_a_value_computed_across_an_invalidation():
    cache = TTLCache(clock=FakeClock())

    def compute():
        cache.invalidate("jobs")  # a job changed while the old list was being read
        return "stale"

    assert cache.get_or_set(("jobs", 1), 10, compute) == "stale"
    assert cache.get(("jobs", 1)) is None
    assert cache.get_or_set(("jobs", 1), 10, lambda: "fresh") == "fresh"
    assert cache.get(("jobs", 1)) == "fresh"


def test_ttl_cache_computes_a_key_once_for_concurrent_callers():
    cache = TTLCache()
    calls = []
    release = threading.Event()

    def slow():
        calls.append(1)
        release.wait(5)
        return "answer"

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get_or_set(("dash",), 10, slow))) for _ in range(6)]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    release.set()
    for thread in threads:
        thread.join(5)
    assert results == ["answer"] * 6 and len(calls) == 1


# ---- cached routes ------------------------------------------------------------------------------


class CountingCursor:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.db.executed.append((" ".join(str(query).split()), params))

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class CountingDb:
    """Answers every statement with no rows and counts connections and statements."""

    def __init__(self):
        self.connects = 0
        self.executed = []
        self.autocommit = False

    def connect(self):
        self.connects += 1
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return CountingCursor(self)

    def close(self):
        return None

    def rollback(self):
        return None

    def commit(self):
        return None


class FakeExecutor:
    def logs(self, ref, max_bytes=0):
        return ""

    def list_scripts(self, scripts_dir, tool):
        return ["a.js"]


def _no_database(monkeypatch):
    def no_db():
        raise AssertionError("this test must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    import update_page
    from flask import request

    from loadlens_app.blueprints import config_api, dashboard
    from loadlens_app.blueprints import load_tests as blueprint

    monkeypatch.setenv("LOADLENS_RESPONSE_CACHE", "1")
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", runtime, raising=False)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    _no_database(monkeypatch)
    db = CountingDb()
    monkeypatch.setattr(dashboard, "_ts_conn", db.connect)
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    areas = {"a": ["svc-a"], "b": ["svc-b"]}
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: list(areas.get(area or "", [])))
    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: list(areas.get(area or "", [])))
    clock = FakeClock()
    monkeypatch.setattr(response_cache.response_cache, "_clock", clock)

    monkeypatch.setitem(core.CONFIG, "load_generators", copy.deepcopy(GENERATORS))
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setitem(core.CONFIG, "limits", {})
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    monkeypatch.setattr(service, "_installed_lt_packs", lambda area, service_id: ["lt-k6-prometheus-rw"])
    monkeypatch.setattr(executors, "executor_factory", lambda generator: FakeExecutor())
    monkeypatch.setattr(config_api, "_bootstrap_service_configs", lambda area, service_id: None)

    # Who asks: the X-Test-User / X-Test-Admin headers stand in for a signed-in user.
    def owner():
        return request.headers.get("X-Test-User", "local")

    def auth():
        admin = request.headers.get("X-Test-Admin", "1") == "1"
        return types.SimpleNamespace(role=Role.ADMIN if admin else Role.VIEWER, user=None)

    monkeypatch.setattr(blueprint, "_owner", owner)
    monkeypatch.setattr(blueprint, "current_auth", auth)
    monkeypatch.setattr(dashboard, "current_auth", auth)

    # The job table lives in memory: no storage behind the store.
    job_table = jobs.JobStore()
    monkeypatch.setattr(job_table, "_persist", lambda record: None)
    monkeypatch.setattr(job_table, "_load_many", lambda statuses, limit: [])
    monkeypatch.setattr(job_table, "_dismiss_rows", lambda job_ids: (set(), set()))
    monkeypatch.setattr(dashboard, "job_store", job_table)

    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), db=db, clock=clock, store=memory, jobs=job_table)


def _dashboard(env, area=None):
    if area:
        env.client.set_cookie("project_area", area)
    resp = env.client.get("/dashboard_data")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp


def test_second_dashboard_request_within_the_ttl_runs_no_sql(app_env):
    first = _dashboard(app_env)
    assert app_env.db.connects == 1 and app_env.db.executed
    executed = len(app_env.db.executed)

    app_env.clock.now += 9
    second = _dashboard(app_env)
    assert app_env.db.connects == 1 and len(app_env.db.executed) == executed
    assert second.get_data() == first.get_data() and second.mimetype == "application/json"
    assert "no-store" in second.headers.get("Cache-Control", "")  # the request hooks still run on a hit

    app_env.clock.now += 1.5
    _dashboard(app_env)
    assert app_env.db.connects == 2


def test_dashboard_cache_key_follows_the_project_area(app_env):
    _dashboard(app_env, "a")
    _dashboard(app_env, "b")
    assert app_env.db.connects == 2
    _dashboard(app_env, "a")
    _dashboard(app_env, "b")
    assert app_env.db.connects == 2
    seen = " ".join(repr(params) for _, params in app_env.db.executed)
    assert "svc-a" in seen and "svc-b" in seen  # each area ran its own statements


def test_dashboard_errors_are_not_cached(app_env, monkeypatch):
    from loadlens_app.blueprints import dashboard

    attempts = []

    def broken():
        attempts.append(1)
        raise RuntimeError("storage down")

    monkeypatch.setattr(dashboard, "_ts_conn", broken)
    assert app_env.client.get("/dashboard_data").status_code == 500
    assert app_env.client.get("/dashboard_data").status_code == 500
    assert len(attempts) == 2


@pytest.mark.parametrize("action", ["delete", "rename", "demo"])
def test_run_changes_invalidate_the_dashboard(app_env, monkeypatch, action):
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_delete_run_data", lambda name: None)
    monkeypatch.setattr(dashboard, "_rename_run_data", lambda old, new: {"run_name": new})
    monkeypatch.setattr(dashboard, "_lookup_report_ref", lambda **kwargs: None)
    monkeypatch.setattr(dashboard, "seed_demo_run", lambda: {"run_name": "demo-1", "area": ""})
    _dashboard(app_env)
    if action == "delete":
        assert app_env.client.delete("/runs/nightly-1").status_code == 200
    elif action == "rename":
        assert app_env.client.patch("/runs/nightly-1", json={"new_run_name": "nightly-2"}).status_code == 200
    else:
        assert app_env.client.post("/demo/seed").status_code == 200
    _dashboard(app_env)
    assert app_env.db.connects == 2


def test_response_cache_can_be_switched_off(app_env, monkeypatch):
    monkeypatch.setenv("LOADLENS_RESPONSE_CACHE", "0")
    _dashboard(app_env)
    _dashboard(app_env)
    assert app_env.db.connects == 2


def _active(env, user="local", admin="1"):
    resp = env.client.get("/load_tests/active", headers={"X-Test-User": user, "X-Test-Admin": admin})
    assert resp.status_code == 200
    return resp.get_json()


def test_active_runs_are_cached_per_user(app_env):
    created = app_env.client.post("/load_tests", json=FORM, headers={"X-Test-User": "alice", "X-Test-Admin": "0"})
    assert created.status_code == 200, created.get_data(as_text=True)
    run_id = created.get_json()["run_id"]

    assert _active(app_env, "alice", admin="0")["runs"][0]["can_stop"] is True
    assert _active(app_env, "bob", admin="0")["runs"][0]["can_stop"] is False  # not alice's answer from the cache
    assert _active(app_env, "carol", admin="1")["runs"][0]["can_stop"] is True

    # A stored message is its text plus the ref readers render (load_tests/messages.py): change both.
    app_env.store._runs[run_id].message, app_env.store._runs[run_id].message_ref = "changed behind the cache", None
    assert _active(app_env, "alice", admin="0")["runs"][0]["message"] != "changed behind the cache"  # a hit within 3 s
    app_env.clock.now += 3.1
    assert _active(app_env, "alice", admin="0")["runs"][0]["message"] == "changed behind the cache"


def test_creating_stopping_and_ci_events_invalidate_active_runs(app_env, monkeypatch):
    assert _active(app_env) == {"total": 0, "runs": []}
    run_id = app_env.client.post("/load_tests", json=FORM).get_json()["run_id"]
    listed = _active(app_env)
    assert listed["total"] == 1 and listed["runs"][0]["can_stop"]

    assert app_env.client.post(f"/load_tests/runs/{run_id}/stop", json={}).status_code == 200
    assert not _active(app_env)["runs"][0]["can_stop"]  # the stop is requested: no second stop button

    run = app_env.store._runs[run_id]
    monkeypatch.setattr(service, "ci_instance", lambda rid, index, token: (run, run.instances[0], None))
    monkeypatch.setattr(service, "record_ci_event", lambda r, inst, payload: {"event": "finished"})
    assert len(response_cache.response_cache) == 1
    assert app_env.client.post(f"/load_tests/ci/{run_id}/0/events", json={"event": "finished"}).status_code == 200
    assert len(response_cache.response_cache) == 0


def test_bad_query_arguments_bypass_the_cache(app_env):
    assert app_env.client.get("/load_tests/active", query_string={"limit": "many"}).status_code == 400
    assert app_env.client.get("/jobs", query_string={"status": "weird"}).status_code == 400
    assert app_env.client.get("/jobs", query_string={"limit": "x"}).status_code == 400
    assert len(response_cache.response_cache) == 0


def _jobs(env, admin="1", **query):
    resp = env.client.get("/jobs", query_string=query or {"status": "running,error"}, headers={"X-Test-Admin": admin})
    assert resp.status_code == 200
    return resp.get_json()["jobs"]


def test_job_changes_reset_the_job_list(app_env):
    assert _jobs(app_env) == []
    job = app_env.jobs.create(jobs.JOB_KIND_REPORT, run_name="nightly", service="svc-a", message="Старт")
    assert [j["job_id"] for j in _jobs(app_env)] == [job.job_id]  # create() dropped the cached empty list

    app_env.jobs.update(job.job_id, message="Сбор метрик", progress=40)
    assert _jobs(app_env)[0]["message"] == "Сбор метрик"

    response_cache.response_cache.set(("dashboard", ()), b"{}", 60)
    app_env.jobs.update(job.job_id, progress=50)
    assert response_cache.response_cache.get(("dashboard", ())) == b"{}"  # progress alone keeps the dashboard
    app_env.jobs.update(job.job_id, status=jobs.JOB_STATUS_ERROR, message="Ошибка", error="HTTP 502 from grafana")
    assert response_cache.response_cache.get(("dashboard", ())) is None  # the report job ended

    assert _jobs(app_env)[0]["error"] == "HTTP 502 from grafana"
    assert _jobs(app_env, admin="0")[0]["error"] is None  # a viewer never gets the admin's cached answer

    dismissed, _ = app_env.jobs.dismiss([job.job_id])
    assert dismissed == [job.job_id] and _jobs(app_env) == []


def test_job_list_is_cached_within_the_ttl(app_env, monkeypatch):
    reads = []
    original = app_env.jobs.list

    def counting(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(app_env.jobs, "list", counting)
    _jobs(app_env, status="running", limit=10)
    _jobs(app_env, status="running", limit=10)
    assert len(reads) == 1
    _jobs(app_env, status="running", limit=5)
    assert len(reads) == 2
    app_env.clock.now += 3.1
    _jobs(app_env, status="running", limit=10)
    assert len(reads) == 3


# ---- watcher ------------------------------------------------------------------------------------


class RecordingEvent:
    """Stands in for threading.Event: records the timeouts of wait() and never blocks."""

    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.flag = False

    def wait(self, timeout=None):
        self.log.append((self.name, timeout))
        return self.flag

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False

    def is_set(self):
        return self.flag


def _watcher(**kwargs):
    memory = MemoryLoadTestStore()
    return watcher.Watcher(store_getter=lambda: memory, catalog_loader=lambda: None, clock=FakeClock(), **kwargs)


def test_watcher_pauses_15_seconds_without_runs_and_3_with_runs():
    w = _watcher()
    log = []
    w._stop, w._wake = RecordingEvent("stop", log), RecordingEvent("wake", log)
    results = iter([0, 2, 1, 0, RuntimeError("storage down")])

    def tick():
        result = next(results, None)
        if result is None:
            w._stop.set()
            return 0
        if isinstance(result, Exception):
            raise result
        return result

    w.tick = tick
    w._loop()
    assert log[:5] == [
        ("wake", watcher.IDLE_TICK_SEC),  # nothing to do: the long pause, which wake() ends
        ("stop", watcher.TICK_SEC),  # runs handled: the usual pace, wake() does not shorten it
        ("stop", watcher.TICK_SEC),
        ("wake", watcher.IDLE_TICK_SEC),
        ("stop", watcher.TICK_SEC),  # a failed pass retries at the usual pace
    ]
    assert watcher.IDLE_TICK_SEC == 15.0 and watcher.TICK_SEC == 3.0


def test_watcher_with_no_runs_waits_on_the_storage_once_per_idle_pause():
    memory = MemoryLoadTestStore()
    claims = []
    original = memory.claim
    memory.claim = lambda *args, **kwargs: claims.append(1) or original(*args, **kwargs)
    w = watcher.Watcher(store_getter=lambda: memory, catalog_loader=lambda: None, clock=FakeClock())
    log = []
    w._stop, w._wake = RecordingEvent("stop", log), RecordingEvent("wake", log)
    real_pause = w._pause

    def pause(handled):
        real_pause(handled)
        if len(log) == 4:
            w._stop.set()

    w._pause = pause
    w._loop()
    assert len(claims) == 4 and log == [("wake", 15.0)] * 4  # one claim query per 15 s instead of per 3 s


def test_watcher_constructor_stays_compatible():
    assert _watcher()._idle_interval == watcher.IDLE_TICK_SEC
    assert _watcher(interval=0.5)._idle_interval == watcher.IDLE_TICK_SEC
    assert _watcher(interval=30)._idle_interval == 30
    assert _watcher(interval=0.5, idle_interval=2)._idle_interval == 2


def test_wake_ends_the_idle_pause_and_stop_still_works():
    w = _watcher(interval=0.05, idle_interval=30)
    ticks = []
    w.tick = lambda: ticks.append(time.monotonic()) or 0
    assert w.start()
    deadline = time.monotonic() + 5
    while not ticks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(ticks) == 1
    time.sleep(0.1)
    assert len(ticks) == 1  # idle: the next pass would come in 30 s
    w.wake()
    deadline = time.monotonic() + 5
    while len(ticks) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(ticks) == 2
    w.stop()
    w._thread.join(5)
    assert not w.running


def test_ensure_watcher_wakes_a_running_loop(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(start=lambda: calls.append("start") or False, wake=lambda: calls.append("wake"))
    monkeypatch.setattr(watcher, "_watcher", fake)
    monkeypatch.setenv("LOADLENS_LOAD_TEST_WATCHER", "1")
    assert watcher.ensure_watcher() is False
    assert calls == ["start", "wake"]
    monkeypatch.setenv("LOADLENS_LOAD_TEST_WATCHER", "0")
    assert watcher.ensure_watcher() is False and calls == ["start", "wake"]


# ---- static files -------------------------------------------------------------------------------


def test_static_url_is_stable_and_names_the_content():
    static_assets.clear_cache()
    url = static_assets.static_url("css/app.css")
    digest = hashlib.sha1((ROOT_DIR / "static" / "css" / "app.css").read_bytes()).hexdigest()[:10]
    assert url == f"/static/css/app.css?v={digest}"
    assert static_assets.static_url("/static/css/app.css") == url == static_assets.static_url("css/app.css")
    assert static_assets.static_url("js/missing.js") == "/static/js/missing.js"
    assert static_assets.static_url("../app.py") == "/static/../app.py"  # outside static/: no version


def test_static_url_changes_with_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(static_assets, "STATIC_ROOT", tmp_path)
    static_assets.clear_cache()
    asset = tmp_path / "js" / "x.js"
    asset.parent.mkdir()
    asset.write_text("one", encoding="utf-8")
    first = static_assets.static_url("js/x.js")
    asset.write_text("two!", encoding="utf-8")
    assert static_assets.static_url("js/x.js") == first  # computed once per process
    static_assets.clear_cache()  # a new process (deployment)
    second = static_assets.static_url("js/x.js")
    assert second != first and re.fullmatch(r"/static/js/x\.js\?v=[0-9a-f]{10}", second)

    monkeypatch.setenv("LOADLENS_STATIC_NO_CACHE", "1")
    asset.write_text("three, longer", encoding="utf-8")
    assert static_assets.static_url("js/x.js") not in (first, second)  # dev mode follows the file
    assert static_assets.static_max_age() is None
    static_assets.clear_cache()


def test_static_files_are_cacheable_for_a_year(app_env):
    for path in ("/static/css/app.css", "/static/js/settings.js", static_assets.static_url("css/app.css")):
        resp = app_env.client.get(path)
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "public, max-age=31536000"


def test_pages_link_static_files_by_version(app_env):
    html = app_env.client.get("/").get_data(as_text=True)
    links = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert links and all("?v=" in link for link in links)
    assert static_assets.static_url("js/ui.js") in links
    order = [html.index(static_assets.static_url(name)) for name in ("js/ui.js", "js/charts.js", "js/dashboard.js")]
    assert order == sorted(order)  # deferred scripts run in document order
    # The dashboard draws SVG only: Chart.js is not loaded there.
    assert "chart.umd.js" not in html


def test_templates_use_versioned_static_links_and_defer_chart_js():
    chart_tags = []
    for path in TEMPLATES.glob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r'(?:src|href)="/static/', text), f"{path.name}: link /static/ through url_for('static', ...)"
        chart_tags += [(path.name, tag) for tag in re.findall(r"<script[^>]*chart\.umd\.js[^>]*>", text)]
    assert len(chart_tags) == 4
    assert "dashboard.html" not in {name for name, _ in chart_tags}
    assert all(re.search(r"\sdefer[\s>]", tag) for _, tag in chart_tags), chart_tags
