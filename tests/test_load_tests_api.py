"""HTTP API of the launch page, generator settings and the run filter of the load tool packs."""

import copy
import json
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from loadlens_app import core
from loadlens_app.load_tests import executors, live, service, store
from loadlens_app.load_tests.executors import CheckReport
from loadlens_app.load_tests.store import MemoryLoadTestStore

GENERATORS = {
    "gen-1": {
        "title": "Gen 1",
        "type": "ssh",
        "ssh": {"host": "10.0.0.5", "port": 22, "username": "loadlens", "private_key": "-----BEGIN KEY-----\nabc\n-----END KEY-----", "host_key": "ssh-ed25519 AAAA"},
        "work_dir": "/opt/loadlens/runs",
        "scripts_dir": "/opt/tests",
        "tools": {},
    },
    "manual": {"title": "Вручную", "type": "manual"},
}
FORM = {
    "service": "checkout",
    "run_name": "nightly",
    "instances": [{"name": "rest", "generator": "gen-1", "tool": "k6", "script": "a.js", "params": "--vus 5", "env": "TOKEN=s3cret-zz"}],
}


class FakeExecutor:
    def logs(self, ref, max_bytes=0):
        return f"tail of {ref}"

    def list_scripts(self, scripts_dir, tool):
        return ["a.js", "sub/b.js"]


@pytest.fixture
def env(monkeypatch, tmp_path):
    from loadlens_app.blueprints import config_api, dashboard

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setitem(core.CONFIG, "load_generators", copy.deepcopy(GENERATORS))
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setitem(core.CONFIG, "limits", {})
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(service, "_installed_lt_packs", lambda area, service_id: ["lt-k6-prometheus-rw"])
    monkeypatch.setattr(executors, "executor_factory", lambda generator: FakeExecutor())
    monkeypatch.setattr(config_api, "_bootstrap_service_configs", lambda area, service_id: None)
    from loadlens_app.blueprints import load_tests as blueprint

    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: [])
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), store=memory, runtime=runtime)


def test_options_list_generators_without_credentials(env):
    data = env.client.get("/load_tests/options").get_json()
    generators = {g["id"]: g for g in data["generators"]}
    assert generators["gen-1"]["host"] == "10.0.0.5" and "k6" in generators["gen-1"]["tools"]
    assert generators["manual"]["tools"] == []
    assert "BEGIN KEY" not in json.dumps(data)
    assert {t["id"] for t in data["tools"]} >= {"k6", "jmeter", "locust", "gatling", "gatling_maven"}


def test_preview_shows_commands_and_stores_nothing(env):
    resp = env.client.post("/load_tests/preview", json=FORM)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["run_tag"] == "nightly" and "--tag testid=nightly" in data["commands"][0]["command"]
    assert "s3cret-zz" not in json.dumps(data)
    assert env.store.list() == []


def test_create_stop_logs_and_list(env):
    resp = env.client.post("/load_tests", json=FORM)
    assert resp.status_code == 200
    created = resp.get_json()
    run_id = created["run_id"]
    assert created["url"] == f"/launch/{run_id}"

    view = env.client.get(f"/load_tests/runs/{run_id}").get_json()
    assert view["status"] == "starting" and view["instances"][0]["env_keys"] == ["TOKEN"]
    listed = env.client.get("/load_tests/runs", query_string={"active": "1"}).get_json()["runs"]
    assert [run["run_id"] for run in listed] == [run_id]

    assert env.client.get(f"/load_tests/runs/{run_id}/logs", query_string={"index": 0}).get_json()["text"] == ""
    env.store._runs[run_id].instances[0].launched_at = env.store._runs[run_id].created_at
    logs = env.client.get(f"/load_tests/runs/{run_id}/logs", query_string={"index": 0}).get_json()
    assert logs["text"].startswith("tail of /opt/loadlens/runs/")
    assert env.client.get(f"/load_tests/runs/{run_id}/logs", query_string={"index": 5}).status_code == 404

    stopped = env.client.post(f"/load_tests/runs/{run_id}/stop", json={"index": 0})
    assert stopped.status_code == 200 and stopped.get_json()["instances"][0]["stop_requested"]
    assert env.client.post(f"/load_tests/runs/{run_id}/stop", json={"index": "0"}).status_code == 400
    assert env.client.post("/load_tests/runs/missing/stop", json={}).status_code == 404
    assert env.client.get("/load_tests/runs/missing").status_code == 404


def test_create_reports_form_errors(env):
    resp = env.client.post("/load_tests", json={**FORM, "instances": [{"generator": "gen-1", "tool": "k6", "script": "a.js", "params": "'x"}]})
    assert resp.status_code == 400 and "кавычки" in resp.get_json()["message"]
    assert env.client.post("/load_tests", data="not json", content_type="text/plain").status_code == 400


def test_scripts_and_live_endpoints(env, monkeypatch):
    scripts = env.client.get("/load_tests/scripts", query_string={"generator": "gen-1", "tool": "k6"}).get_json()
    assert scripts == {"scripts": ["a.js", "sub/b.js"], "scripts_dir": "/opt/tests", "kind": "file"}
    assert env.client.get("/load_tests/scripts", query_string={"generator": "x", "tool": "k6"}).status_code == 404
    from loadlens_app.blueprints import load_tests as blueprint

    seen = []

    def snapshot(run, tools, window_min=0, since=None, mode=""):
        seen.append((window_min, since, mode))
        return {"ok": True, "series": {}, "run": run.run_id}

    monkeypatch.setattr(blueprint, "live_snapshot", snapshot)
    run_id = env.client.post("/load_tests", json=FORM).get_json()["run_id"]
    assert env.client.get(f"/load_tests/runs/{run_id}/live").get_json()["run"] == run_id
    env.client.get(f"/load_tests/runs/{run_id}/live", query_string={"window": 15, "since": 1760000000500, "mode": "plain"})
    assert seen == [(0, None, ""), (15, 1760000000.5, "plain")]
    assert env.client.get(f"/load_tests/runs/{run_id}/live", query_string={"since": "soon"}).status_code == 400
    assert env.client.get(f"/load_tests/runs/{run_id}/live", query_string={"mode": "all"}).status_code == 400


def test_pages_render(env):
    assert env.client.get("/launch").status_code == 200
    page = env.client.get("/launch/abc").get_data(as_text=True)
    assert 'data-run-id="abc"' in page and "/static/js/launch_run.js" in page


# ---- settings -------------------------------------------------------------------------------------


def test_config_masks_keys_and_restores_them_for_the_same_host(env):
    from loadlens_app.blueprints import config_api

    data = env.client.get("/config").get_json()
    gen = data["load_generators"]["gen-1"]
    assert gen["ssh"]["private_key"] == "***" and "k6" in data["load_tools"]

    edited = copy.deepcopy(data["load_generators"])
    edited["gen-1"]["title"] = "Generator one"
    assert env.client.post("/config", json={"section": "load_generators", "data": edited}).status_code == 200
    saved = json.loads(env.runtime.read_text(encoding="utf-8"))["load_generators"]["gen-1"]
    assert saved["ssh"]["private_key"].startswith("-----BEGIN KEY") and saved["title"] == "Generator one"

    moved = copy.deepcopy(edited)
    moved["gen-1"]["ssh"]["host"] = "attacker.example"
    resp = env.client.post("/config", json={"section": "load_generators", "data": moved})
    assert resp.status_code == 400 and "private_key" in resp.get_json()["error"]

    broken = copy.deepcopy(edited)
    broken["gen-1"]["work_dir"] = "relative"
    resp = env.client.post("/config", json={"section": "load_generators", "data": broken})
    assert resp.status_code == 400 and "абсолютный путь" in resp.get_json()["error"]


def test_saving_tools_keeps_only_differences(env):
    tools = env.client.get("/config").get_json()["load_tools"]
    tools["k6"]["executable"] = "/usr/local/bin/k6"
    del tools["locust"]
    assert env.client.post("/config", json={"section": "load_tools", "data": tools}).status_code == 200
    saved = json.loads(env.runtime.read_text(encoding="utf-8"))["load_tools"]
    assert saved == {"k6": {"executable": "/usr/local/bin/k6"}, "locust": {"enabled": False}}
    bad = env.client.post("/config", json={"section": "load_tools", "data": {"k6": {"command": "{exe} {oops}"}}})
    assert bad.status_code == 400 and "{oops}" in bad.get_json()["error"]


def test_generator_check_runs_on_unsaved_form_values(env, monkeypatch):
    from loadlens_app.blueprints import config_api

    seen = {}

    def fake_check(generator, tools):
        seen["generator"] = generator
        return CheckReport(ok=True, message="Подключение есть", details={"host_key": "ssh-ed25519 BBBB"})

    monkeypatch.setattr(config_api, "check_generator", fake_check)
    form = copy.deepcopy(env.client.get("/config").get_json()["load_generators"])
    form["gen-1"]["ssh"]["username"] = "other"
    resp = env.client.post("/config/test_connection", json={"section": "load_generators", "data": form, "source": "gen-1"})
    body = resp.get_json()
    assert resp.status_code == 200 and body["ok"] and body["details"]["host_key"] == "ssh-ed25519 BBBB"
    assert seen["generator"].ssh.username == "other" and seen["generator"].ssh.private_key.startswith("-----BEGIN KEY")
    # The wizard stores {"at": checked_at, "target": target} as the "verified" mark of the generator.
    details = body["details"]
    assert details["target"] == seen["generator"].check_target() and details["target"].startswith("ssh://other@")
    assert details["checked_at"].endswith("+00:00") and details["problems"] == [] and details["connected"] is False
    form["gen-1"]["verified"] = {"at": details["checked_at"], "target": details["target"]}
    assert env.client.post("/config", json={"section": "load_generators", "data": form}).status_code == 200
    options = env.client.get("/load_tests/options").get_json()
    listed = {gen["id"]: gen for gen in options["generators"]}
    assert listed["gen-1"]["verified"] is True
    missing = env.client.post("/config/test_connection", json={"section": "load_generators", "data": form, "source": "nope"})
    assert missing.status_code == 400


# ---- report hook and query packs ---------------------------------------------------------------------


def test_report_job_passes_the_run_filter_to_update_report(monkeypatch):
    from loadlens_app import jobs
    from loadlens_app.blueprints import dashboard

    calls = []
    monkeypatch.setattr(dashboard, "update_report", lambda *args, **kwargs: calls.append(kwargs) or {"run_id": "x"})
    monkeypatch.setattr(dashboard, "_report_step", lambda area, service_id: "30s")
    monkeypatch.setattr(jobs.job_store, "_persist", lambda record: None)

    def no_db():
        raise RuntimeError("the job stays in memory; the job table is not reachable in this test")

    # running_counts also reads the job table: it must fall back to memory, not reach the configured database.
    monkeypatch.setattr(jobs, "_ts_conn", no_db)
    monkeypatch.setattr(threading, "Thread", lambda target, daemon=True: types.SimpleNamespace(start=target))
    job = dashboard.start_report_job(
        0, 600_000, "checkout", "shop", use_llm=False, save_to_db=False, web_only=True, run_name="nightly",
        test_type="step", owner="eng", run_tag="nightly", run_filter_packs=["lt-k6-influxql"],
    )
    assert calls[0]["run_tag"] == "nightly" and calls[0]["run_filter_packs"] == ["lt-k6-influxql"]
    assert jobs.job_store.get(job.job_id).status == jobs.JOB_STATUS_DONE


def _installed(pack_id, variables=None):
    from AI.query_packs import load_catalog, merge_pack, render_pack

    pack = load_catalog().pack(pack_id)
    block, _ = merge_pack(None, render_pack(pack, variables or {}))
    return block


def test_run_filter_appends_or_replaces_the_pack_variable():
    from AI.query_packs import apply_run_filter

    k6 = _installed("lt-k6-prometheus-rw", {"matchers": 'job="k6"'})
    jmeter = _installed("lt-jmeter-influxql")
    queries = {"lt_framework": k6, "jvm": {"labels": ["x"], "promql_queries": ["up"]}}
    filtered, applied = apply_run_filter(queries, "nightly-1")
    assert applied == ["lt-k6-prometheus-rw"]
    text = " ".join(filtered["lt_framework"]["promql_queries"])
    assert 'job="k6", testid="nightly-1"' in text
    assert filtered["lt_framework"]["labels"] == k6["labels"] and queries["lt_framework"] == k6
    assert filtered["jvm"] == queries["jvm"]

    filtered, applied = apply_run_filter({"lt_framework": jmeter}, "nightly-1")
    assert applied == ["lt-jmeter-influxql"]
    assert all("=~ /^nightly-1$/" in q for q in filtered["lt_framework"]["influxql_queries"] if "application" in q)

    assert apply_run_filter({"lt_framework": jmeter}, "nightly-1", packs=["lt-k6-influxql"]) == ({"lt_framework": jmeter}, [])
    assert apply_run_filter({"lt_framework": jmeter}, "", packs=None)[1] == []


def test_run_filter_leaves_hand_edited_rows_alone():
    from AI.query_packs import apply_run_filter

    block = _installed("lt-k6-prometheus-rw")
    block["promql_queries"][0] = "sum(rate(my_own_metric[1m]))"
    filtered, _ = apply_run_filter({"lt_framework": block}, "t1")
    assert filtered["lt_framework"]["promql_queries"][0] == "sum(rate(my_own_metric[1m]))"
    assert 'testid="t1"' in filtered["lt_framework"]["promql_queries"][1]


def test_pack_run_filter_is_validated():
    from AI.query_packs.models import QueryPack

    base = {
        "schema_version": 1, "id": "lt-x", "version": 1, "title": "x", "category": "load_tool", "domain": "lt_framework",
        "language": "promql", "description": "x",
        "variables": {"m": {"title": "m", "default": ""}},
        "queries": [{"label": "LT: rps", "query": "sum(x{ {{m}} })", "unit": "per_second", "role": "lt.rps_total"}],
    }
    QueryPack(**{**base, "run_filter": {"variable": "m", "value": 'testid="{{run_tag}}"', "join": ", "}})
    for bad in ({"variable": "nope", "value": "{{run_tag}}"}, {"variable": "m", "value": "fixed"}, {"variable": "m", "value": "{{run_tag}}{{m}}"}):
        with pytest.raises(ValueError):
            QueryPack(**{**base, "run_filter": bad})
    with pytest.raises(ValueError, match="load tool packs only"):
        QueryPack(**{**base, "category": "runtime", "domain": "jvm", "queries": [{"label": "x up", "query": "up{ {{m}} }", "unit": "count"}], "run_filter": {"variable": "m", "value": "{{run_tag}}"}})


# ---- dashboard ------------------------------------------------------------------------------------


def test_active_runs_for_the_dashboard(env, monkeypatch):
    briefs = []

    def brief(run, tools, now=None):
        briefs.append(run.run_id)
        return {"ok": True, "message": "", "rps": 12.5, "errors": 0.4, "points": [[1, 2.0], [2, 3.0]]}

    monkeypatch.setattr(live, "live_brief", brief)
    assert env.client.get("/load_tests/active").get_json() == {"total": 0, "runs": []}
    first = env.client.post("/load_tests", json=FORM).get_json()["run_id"]
    second = env.client.post("/load_tests", json={**FORM, "run_name": "second"}).get_json()["run_id"]
    assert env.client.get("/load_tests/active", query_string={"limit": 0}).get_json() == {"total": 2, "runs": []}

    one = env.client.get("/load_tests/active", query_string={"limit": 1, "live": 1}).get_json()
    assert one["total"] == 2 and len(one["runs"]) == 1
    view = one["runs"][0]
    assert view["status"] == "starting" and view["status_title"] and view["loading"] and view["url"] == f"/launch/{view['run_id']}"
    assert view["instances_total"] == 1 and view["instances_running"] == 0 and view["max_seconds"] > 0
    assert view["live"] is None and not briefs  # nothing generates load yet

    run = env.store._runs[first]
    run.status = "running"
    run.instances[0].status = "running"
    run.instances[0].started_at = run.created_at
    runs = {item["run_id"]: item for item in env.client.get("/load_tests/active", query_string={"live": "1"}).get_json()["runs"]}
    assert runs[first]["live"]["rps"] == 12.5 and runs[first]["instances_running"] == 1 and runs[first]["can_stop"]
    assert runs[second]["live"] is None and briefs == [first]
    assert env.client.get("/load_tests/active").get_json()["runs"][0]["live"] is None and briefs == [first]  # live only on request

    env.client.post(f"/load_tests/runs/{first}/stop", json={})
    runs = {item["run_id"]: item for item in env.client.get("/load_tests/active").get_json()["runs"]}
    assert not runs[first]["can_stop"] and runs[second]["can_stop"]  # the stop is already requested

    assert env.client.get("/load_tests/active", query_string={"limit": "many"}).status_code == 400
    assert len(env.client.get("/load_tests/active", query_string={"limit": 500}).get_json()["runs"]) == 2


def test_active_runs_follow_the_project_area(env, monkeypatch):
    from loadlens_app.blueprints import load_tests as blueprint

    env.client.post("/load_tests", json=FORM)
    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: ["payments"])
    assert env.client.get("/load_tests/active").get_json() == {"total": 0, "runs": []}


def test_dashboard_offers_the_stop_to_the_owner_and_admins(env):
    run_id = env.client.post("/load_tests", json=FORM).get_json()["run_id"]
    env.store._runs[run_id].owner = "alice"

    def can_stop(user, admin):
        return service.active_runs(None, 5, user, admin, False)["runs"][0]["can_stop"]

    assert can_stop("alice", False) and can_stop("bob", True)
    assert not can_stop("bob", False) and not can_stop("", False)
    env.store._runs[run_id].status = "reporting"
    assert not can_stop("alice", False)  # the load is over, the report is being built
