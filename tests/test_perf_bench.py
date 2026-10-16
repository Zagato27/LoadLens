"""Bench tooling: the Server-Timing header, the seed builders of tools/seed_bench_data.py and the statistics of
tools/bench_endpoints.py. Offline: no database, no network."""

import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from flask import Flask, jsonify

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402,F401  (before anything that imports update_page)
from loadlens_app.auth import InMemoryRepository, init_auth  # noqa: E402
from loadlens_app.blueprints import dashboard  # noqa: E402

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
SERVER_TIMING = re.compile(r"^app;dur=\d+\.\d$")


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"bench_tool_{name}", ROOT_DIR / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


seed = _load_tool("seed_bench_data")
bench = _load_tool("bench_endpoints")


# ---- Server-Timing ---------------------------------------------------------------------------------------


def _app(auth_enabled: bool) -> Flask:
    flask_app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))
    flask_app.config["TESTING"] = True

    @flask_app.route("/plain")
    def plain():
        return jsonify(ok=True)

    @flask_app.route("/timed")
    def timed():
        response = jsonify(ok=True)
        response.headers["Server-Timing"] = "db;dur=1.5"
        return response

    env = {"LOADLENS_AUTH_ENABLED": "1" if auth_enabled else "0", "LOADLENS_SECRET_KEY": "k" * 32}
    init_auth(flask_app, repository=InMemoryRepository(), config={}, env=env)
    return flask_app


def test_server_timing_header_carries_the_application_time():
    client = _app(auth_enabled=False).test_client()
    response = client.get("/plain")
    assert response.status_code == 200
    assert SERVER_TIMING.match(response.headers["Server-Timing"])
    assert bench.parse_server_timing(response.headers["Server-Timing"]) >= 0.0


def test_server_timing_is_on_public_errors_and_keeps_a_view_metric():
    client = _app(auth_enabled=True).test_client()
    assert SERVER_TIMING.match(client.get("/healthz").headers["Server-Timing"])
    assert SERVER_TIMING.match(client.get("/no-such-page").headers["Server-Timing"])
    unauthorized = client.get("/plain")  # unclassified endpoint behind the login: the gate answers itself
    assert unauthorized.status_code in (302, 401, 403)
    assert SERVER_TIMING.match(unauthorized.headers["Server-Timing"])

    open_app = _app(auth_enabled=False).test_client()
    combined = open_app.get("/timed").headers["Server-Timing"]
    assert combined.startswith("db;dur=1.5, app;dur=")
    assert bench.parse_server_timing(combined) is not None


# ---- seed builders ---------------------------------------------------------------------------------------


def test_run_plan_is_deterministic_for_a_seed():
    first = seed.plan_runs(60, 6, 15, seed=7, now=NOW)
    again = seed.plan_runs(60, 6, 15, seed=7, now=NOW)
    other = seed.plan_runs(60, 6, 15, seed=8, now=NOW)
    assert first == again
    assert [r.run_id for r in first] != [r.run_id for r in other]
    assert [r.run_name for r in first] == [r.run_name for r in other]  # names depend on the position only


def test_run_plan_shapes_names_windows_and_analyses():
    plan = seed.plan_runs(1000, 10, 60, seed=1, now=NOW, days=90)
    assert plan[0].run_name == "bench-svc01-0001" and plan[11].run_name == "bench-svc02-0002"
    assert {r.service for r in plan} == {f"svc{i:02d}" for i in range(1, 11)}
    assert len({r.run_name for r in plan}) == 1000 and len({r.run_id for r in plan}) == 1000
    for run in plan:
        assert run.end - run.start == timedelta(minutes=60)
        assert run.test_type in seed.TEST_TYPES
        first = run.analyses[0]
        assert run.end < first.created_at <= NOW
        assert NOW - first.created_at <= timedelta(days=90)
        assert all(a.created_at >= first.created_at for a in run.analyses)
    reanalysed = sum(len(r.analyses) == 2 for r in plan)
    assert 50 <= reanalysed <= 150  # about 10 %
    styles = {r.rps_style for r in plan}
    assert styles == {"number", "string", "tilde"}


def test_metric_rows_are_deterministic_and_cover_lt_and_jvm():
    run = seed.plan_runs(3, 3, 12, seed=5, now=NOW)[1]
    rows = list(seed.metric_rows(run, series=20, jvm_series=16, seed=5))
    assert rows == list(seed.metric_rows(run, series=20, jvm_series=16, seed=5))
    assert rows != list(seed.metric_rows(run, series=20, jvm_series=16, seed=6))
    assert len(rows) == 12 * (20 + 16)
    assert all(len(row) == len(seed.METRICS_COLUMNS) for row in rows)
    named = [dict(zip(seed.METRICS_COLUMNS, row)) for row in rows]
    assert {r["domain"] for r in named} == {"lt", "jvm"}
    assert {"rps_total", "errors_total", "latency_p95"} <= {r["query_label"] for r in named if r["domain"] == "lt"}
    assert len({(r["domain"], r["query_label"], r["series"]) for r in named}) == 36
    assert {r["run_name"] for r in named} == {run.run_name} and {r["run_id"] for r in named} == {run.run_id}
    assert min(r["time"] for r in named) == run.start and max(r["time"] for r in named) < run.end
    assert {(r["start_ms"], r["end_ms"]) for r in named} == {(run.start_ms, run.end_ms)}


def test_report_rows_follow_the_dashboard_rules():
    assert seed.DASHBOARD_VERDICTS == dashboard.DASHBOARD_VERDICTS
    plan = seed.plan_runs(200, 10, 30, seed=3, now=NOW)
    seen_styles = set()
    for run in plan:
        rows = [dict(zip(seed.REPORT_COLUMNS, row)) for row in seed.report_rows(run, seed=3)]
        finals = [r for r in rows if r["domain"] == "final"]
        assert len(finals) == len(run.analyses)
        assert sorted(r["domain"] for r in rows) == sorted(["final", *seed.OTHER_DOMAINS] * len(run.analyses))
        for row in rows:
            assert row["verdict"] in dashboard.DASHBOARD_VERDICTS and row["project_area"] == "bench"
            assert row["sla_verdict"] in (None, *dashboard.DASHBOARD_VERDICTS)
        for final in finals:
            parsed = final["parsed"].adapted
            section, field = seed.RPS_FIELD_BY_TEST_TYPE[run.test_type]
            level = parsed[section][field]
            seen_styles.add(run.rps_style)
            parsed_number = dashboard._dashboard_number(level)
            if run.rps_style == "tilde":
                assert isinstance(level, str) and level.startswith("~") and parsed_number is None
            else:
                assert parsed_number == pytest.approx(run.max_rps)
            assert json.loads(final["text"]) == parsed
    assert seen_styles == {"number", "string", "tilde"}


def test_report_rows_are_deterministic():
    run = seed.plan_runs(5, 5, 10, seed=2, now=NOW)[4]

    def plain(rows):
        return [tuple(getattr(v, "adapted", v) for v in row) for row in rows]

    assert plain(seed.report_rows(run, seed=2)) == plain(seed.report_rows(run, seed=2))
    assert plain(seed.report_rows(run, seed=2)) != plain(seed.report_rows(run, seed=9))


def test_runtime_json_gets_the_bench_area_and_keeps_others():
    existing = {"llm": {"provider": "x"}, "per_area": {"devstack": {"title": "Dev"}, "bench-pad-07": {"title": "old"}}}
    data = seed.bench_settings_runtime(existing, "bench", 4, pad_kb=40)
    assert data["llm"] == {"provider": "x"} and data["per_area"]["devstack"] == {"title": "Dev"}
    assert list(data["per_area"]["bench"]["services"]) == ["svc01", "svc02", "svc03", "svc04"]
    assert "bench-pad-07" not in data["per_area"]
    assert len(json.dumps(data, ensure_ascii=False, indent=2).encode()) >= 40 * 1024
    assert seed.bench_settings_runtime(data, "bench", 4, pad_kb=40) == data  # re-running replaces, not grows
    bare = seed.bench_settings_runtime(existing, "bench", 2, pad_kb=0)
    assert sorted(bare["per_area"]) == ["bench", "devstack"]

    metrics = seed.bench_metrics_runtime({"devstack": {"services": {}}}, "bench", 2, pad_kb=10)
    assert metrics["devstack"] == {"services": {}}
    assert metrics["bench"]["services"]["svc02"]["metrics"] == []
    assert len(json.dumps(metrics, ensure_ascii=False, indent=2).encode()) >= 10 * 1024


def test_runtime_only_rewrites_the_sandbox_files_without_a_database(tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"devstack": {"title": "Dev"}}}), encoding="utf-8")
    assert seed.main(["--runtime-only", "--runtime-json", str(runtime), "--services", "3", "--runtime-kb", "0", "--metrics-runtime-kb", "0"]) == 0
    data = json.loads(runtime.read_text(encoding="utf-8"))
    assert sorted(data["per_area"]) == ["bench", "devstack"]
    metrics = json.loads((tmp_path / "metrics_config_runtime.json").read_text(encoding="utf-8"))
    assert sorted(metrics["bench"]["services"]) == ["svc01", "svc02", "svc03"]
    with pytest.raises(SystemExit):
        seed.parse_args(["--runtime-only"])


def test_seed_refuses_a_remote_database():
    args = seed.parse_args(["--dsn", "host=db.example.org dbname=x user=y"])
    with pytest.raises(SystemExit):
        seed.connection_params(args)
    local = seed.connection_params(seed.parse_args(["--dbname", "ll_s9"]))
    assert local["host"] == "127.0.0.1" and local["port"] == 55432 and local["dbname"] == "ll_s9"


# ---- bench statistics ------------------------------------------------------------------------------------


def test_percentiles_on_a_known_array():
    values = [float(v) for v in range(10, 0, -1)]  # 10..1, unsorted on purpose
    assert bench.percentile(values, 0.5) == pytest.approx(5.5)
    assert bench.percentile(values, 0.95) == pytest.approx(9.55)
    assert bench.percentile(values, 1.0) == 10.0 and bench.percentile(values, 0.0) == 1.0
    assert bench.percentile([4.0], 0.95) == 4.0 and bench.percentile([], 0.5) is None
    stats = bench.summarize(values)
    assert stats == {"count": 10, "p50": pytest.approx(5.5), "p95": pytest.approx(9.55), "max": 10.0, "mean": pytest.approx(5.5)}
    assert bench.summarize([])["p50"] is None


def test_server_timing_parser():
    assert bench.parse_server_timing("app;dur=12.5") == 12.5
    assert bench.parse_server_timing("db;dur=3, app;dur=7") == 7.0
    assert bench.parse_server_timing('app;desc="flask";dur=4.2') == 4.2
    assert bench.parse_server_timing("db;dur=3") is None and bench.parse_server_timing(None) is None


class FakeTarget:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def get(self, path):
        answer = self.answers[self.calls % len(self.answers)]
        self.calls += 1
        return answer


def test_measure_aggregates_samples_errors_and_sizes():
    target = FakeTarget([(200, 10.0, 8.0, 100), (200, 20.0, 18.0, 300), (500, 30.0, None, 50), (0, 40.0, None, 0)])
    item = bench.measure(target, "/runs", rounds=4, concurrency=2, warmup=0)
    assert target.calls == 8 and item["count"] == 8
    assert item["errors"] == 4 and item["statuses"] == {"200": 4, "500": 2, "0": 2}
    assert item["p50"] == pytest.approx(25.0) and item["max"] == 40.0
    assert item["app_p50"] == pytest.approx(13.0) and item["bytes"] == 300
    assert item["rps"] > 0
    warmed = FakeTarget([(200, 1.0, 1.0, 1)])
    bench.measure(warmed, "/", rounds=1, concurrency=1, warmup=3)
    assert warmed.calls == 4


def test_table_with_comparison():
    after = [{"path": "/runs?limit=20", "p50": 50.0, "p95": 100.0, "max": 120.0, "rps": 18.0, "app_p50": 45.0, "bytes": 9800, "errors": 0}]
    before = [{"path": "/runs?limit=20", "p50": 500.0, "p95": 650.0}]
    table = bench.render_table(after, before)
    lines = table.splitlines()
    assert lines[0].startswith("| Эндпоинт | p50, мс | p95, мс") and "было p95" in lines[0] and "×p95" in lines[0]
    assert "| `/runs?limit=20` | 50.0 | 100.0 | 120.0 | 18.0 | 45.0 | 9.6 КБ | 0 | 500.0 | 650.0 | 10.00× | 6.50× |" == lines[2]
    assert "было" not in bench.render_table(after)


PLAN = [
    "Limit  (cost=294.56..294.57 rows=1 width=727) (actual time=20.251..20.254 rows=1 loops=1)",
    "  Buffers: shared hit=3066 read=4",
    "  ->  Seq Scan on llm_reports  (cost=0.03..292.90 rows=332 width=727) (actual time=0.223..19.870 rows=332 loops=1)",
    "        Buffers: shared hit=3063",
    "Planning:",
    "  Buffers: shared hit=12",
    "Planning Time: 5.432 ms",
    "Execution Time: 17.100 ms",
]


def test_plan_summary():
    assert bench.summarize_plan(PLAN) == {"top": "Limit", "planning_ms": 5.432, "execution_ms": 17.1, "buffers": 3070}
    assert bench.summarize_plan([]) == {"top": "", "planning_ms": None, "execution_ms": None, "buffers": None}
    assert bench.statement_kind("\n  WITH base AS (SELECT 1) SELECT 1") == "WITH"
    assert bench.statement_kind("select 1") == "SELECT" and bench.statement_kind("") == ""


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def mogrify(self, query, params=None):
        return (query % tuple(repr(p) for p in params) if params else query).encode()

    def execute(self, query, params=None):
        self.conn.executed.append(query)

    def fetchone(self):
        return (1,)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.conn.cursor_closed += 1
        return False


class FakeConnection:
    def __init__(self):
        self.executed, self.events, self.cursor_closed, self.autocommit = [], [], 0, False

    def cursor(self):
        return FakeCursor(self)

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, *exc):
        self.events.append("commit" if exc[0] is None else "rollback")
        return False


def test_recording_connection_keeps_statements_and_transaction_semantics():
    inner, log = FakeConnection(), []
    conn = bench._RecordingConnection(inner, log)
    conn.autocommit = True
    assert inner.autocommit is True and conn.autocommit is True
    with conn, conn.cursor() as cur:
        cur.execute("SELECT %s FROM t", ("a",))
        assert cur.fetchone() == (1,)
    assert log == ["SELECT 'a' FROM t"] and inner.executed == ["SELECT %s FROM t"]
    assert inner.events == ["enter", "commit"] and inner.cursor_closed == 1


def test_explain_refuses_a_working_tree():
    with pytest.raises(SystemExit, match="not a sandbox copy"):
        bench._load_sandbox_app(ROOT_DIR, allow_remote=False)


def test_bench_arguments():
    assert bench.parse_args(["--sandbox", "3"]).base == "http://127.0.0.1:5073"
    with pytest.raises(SystemExit):
        bench.parse_args(["--base", "http://loadlens.example.org"])
    assert bench.parse_args(["--base", "http://loadlens.example.org", "--allow-remote"]).base == "http://loadlens.example.org"
    args = bench.parse_args(["--only", "runs"])
    assert bench.selected_endpoints(args) == [p for p in bench.ENDPOINTS if "runs" in p]
    assert len(bench.ENDPOINTS) == 12
