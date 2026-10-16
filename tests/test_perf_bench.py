"""Bench tooling: the Server-Timing header, the seed builders of tools/seed_bench_data.py, the statistics and
the --during-job / --setup-stubs parts of tools/bench_endpoints.py, and the stubs of tools/stubs. Offline: no
database; the only network is servers the tests start on 127.0.0.1 (stubs on port 0), with proxies off."""

import http.client
import importlib.util
import json
import re
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

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


class _ResetCursor:
    """Answers to_regclass by a set of existing tables and 'no foreign rows' to everything else."""

    def __init__(self, existing):
        self.existing = existing
        self.statements = []
        self._answer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.statements.append(statement)
        if statement.startswith("SELECT to_regclass"):
            self._answer = (params[0].split(".", 1)[1] in self.existing,)
        else:
            self._answer = (False,)

    def fetchone(self):
        return self._answer


class _ResetConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


def test_reset_clears_run_index_when_it_exists():
    cur = _ResetCursor({"metrics", "llm_reports", "run_index"})
    conn = _ResetConn(cur)
    seed.reset_bench_rows(conn)
    assert "TRUNCATE public.run_index" in cur.statements and conn.committed


def test_reset_skips_run_index_before_the_app_created_it():
    cur = _ResetCursor({"metrics", "llm_reports"})
    seed.reset_bench_rows(_ResetConn(cur))
    assert "TRUNCATE public.metrics" in cur.statements
    assert not any("run_index" in s and not s.startswith("SELECT to_regclass") for s in cur.statements)


# ---- stubs of tools/stubs ----------------------------------------------------------------------------------


def _load_stub(name: str):
    spec = importlib.util.spec_from_file_location(f"bench_stub_{name}", ROOT_DIR / "tools" / "stubs" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prom_stub = _load_stub("prom_stub")
llm_stub = _load_stub("llm_stub")
confluence_stub = _load_stub("confluence_stub")


@contextmanager
def _serving(module, *argv):
    """A stub on 127.0.0.1 and a free port, stopped when the block ends."""
    server = module.make_server(module.parse_args(["--port", "0", *argv]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _http(base: str, path: str, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    parts = urlsplit(base)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


@pytest.fixture
def local_without_proxy(monkeypatch):
    """requests (used by the application code under test) never sends 127.0.0.1 through a proxy."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


START, END = 1760600000, 1760603600  # 2025-10-16 07:33:20 .. 08:33:20 UTC
RANGE = {"query": "sum by (application, instance) (rate(jvm_gc_pause_seconds_sum[2m]))", "start": str(START), "end": str(END), "step": "15"}


def test_prometheus_stub_series_are_deterministic_and_shaped_like_prometheus():
    data = prom_stub.query_range_payload(RANGE, series=4, steps=4, seed=3, max_points=11000)
    assert data == prom_stub.query_range_payload(dict(RANGE), series=4, steps=4, seed=3, max_points=11000)
    assert data != prom_stub.query_range_payload(RANGE, series=4, steps=4, seed=4, max_points=11000)
    assert data["resultType"] == "matrix" and len(data["result"]) == 4
    for number, series in enumerate(data["result"], start=1):
        assert series["metric"]["application"] == f"application-{number:02d}"
        assert series["metric"]["instance"] == f"instance-{number:02d}"
        stamps = [point[0] for point in series["values"]]
        assert len(stamps) == 241 and stamps[0] == START and stamps[-1] == END
        assert all(b - a == 15 for a, b in zip(stamps, stamps[1:]))
        assert all(isinstance(value, str) and float(value) >= 0 for _ts, value in series["values"])
    values = [float(v) for _ts, v in data["result"][0]["values"]]
    assert 1.6 < (sum(values[-50:]) / 50) / (sum(values[:50]) / 50) < 2.4  # four load steps: x1 .. x2
    same_request = dict(RANGE, start="2025-10-16T07:33:20Z", end="2025-10-16T08:33:20+00:00", step="15s")
    assert prom_stub.query_range_payload(same_request, series=4, steps=4, seed=3, max_points=11000) == data
    flat = prom_stub.query_range_payload(RANGE, series=1, steps=0, seed=3, max_points=11000)["result"][0]["values"]
    assert 0.8 < float(flat[-1][1]) / float(flat[0][1]) < 1.25


def test_prometheus_stub_labels_cover_every_query_of_the_example_and_the_bench_area():
    example = _example_config()
    pairs = [
        (query, keys)
        for block in example["queries"].values()
        for query, keys in zip(block.get("promql_queries") or [], block.get("label_keys_list") or [])
    ]
    pairs += list(zip(bench.LT_PROMQL_QUERIES, bench.LT_LABEL_KEYS))
    assert len(pairs) > 60
    for query, keys in pairs:
        labels = prom_stub.series_labels(query, 2)
        assert all(key in labels[0] for key in keys), (query, keys)
        assert len({tuple(labels[i][key] for key in keys) for i in range(2)}) == (2 if keys else 1)
    assert prom_stub.query_label_names('label_replace(max by (instance) (x), "server", "$1", "instance", "(.*)")') == ["instance", "server"]
    assert prom_stub.query_label_names("histogram_quantile(0.95, sum by (le, application) (rate(x[1m])))") == ["application"]
    percent = prom_stub.query_range_payload(dict(RANGE, query="100 * sum by (node) (rate(cpu[2m]))"), series=6, steps=4, seed=1, max_points=11000)
    assert all(float(v) <= 100.0 for s in percent["result"] for _t, v in s["values"])


def test_prometheus_stub_refuses_what_prometheus_refuses():
    with pytest.raises(prom_stub.BadRequest, match="maximum resolution"):
        prom_stub.query_range_payload(dict(RANGE, start="0", end="1000000", step="1"), series=1, steps=4, seed=1, max_points=11000)
    for bad in (dict(RANGE, query=" "), dict(RANGE, step="0"), dict(RANGE, step="abc"), dict(RANGE, end=str(START - 1)), dict(RANGE, start="yesterday")):
        with pytest.raises(prom_stub.BadRequest):
            prom_stub.query_range_payload(bad, series=1, steps=4, seed=1, max_points=11000)
    assert prom_stub.parse_step("1m") == 60 and prom_stub.parse_step("2.5") == 2.5 and prom_stub.parse_step("500ms") == 0.5
    vector = prom_stub.query_payload({"query": RANGE["query"], "time": str(END)}, series=3, steps=4, seed=1)
    assert vector["resultType"] == "vector" and len(vector["result"]) == 3
    assert vector["result"][0]["value"][0] == END and float(vector["result"][0]["value"][1]) > 0


def test_prometheus_stub_over_http_feeds_the_report_collector(local_without_proxy):
    from AI import pipeline
    from loadlens_app import connection_checks

    with _serving(prom_stub, "--series", "3") as url:
        status, body, _headers = _http(url, "/api/v1/query_range?" + urlencode(RANGE))
        assert status == 200 and json.loads(body)["status"] == "success"
        form = urlencode(RANGE).encode()
        status, body, _headers = _http(url, "/api/v1/query_range", "POST", form, {"Content-Type": "application/x-www-form-urlencoded"})
        assert status == 200 and len(json.loads(body)["data"]["result"]) == 3
        status, body, _headers = _http(url, "/api/v1/query_range?" + urlencode(dict(RANGE, step="-1")))
        assert status == 400 and json.loads(body)["errorType"] == "bad_data"
        assert connection_checks.check_prometheus(url).ok

        frames = pipeline.fetch_and_aggregate_with_label_keys(
            url, START, END,
            [RANGE["query"], "max(k6_vus)"], [["application", "instance"], []],
            step="15s", resample_interval="1min",
            ef_config={"metrics_source": {"type": "prometheus", "prometheus": {"url": url}}},
        )
        assert list(frames[0].columns) == [f"application=application-{i:02d}|instance=instance-{i:02d}" for i in (1, 2, 3)]
        assert len(frames[0]) == 61 and frames[0].notna().all().all()
        assert list(frames[1].columns) == [""]  # no label keys: the series fold into one, as a total does

        stats = json.loads(_http(url, "/_stub/stats")[1])
        assert stats["requests"]["query_range"] == 5 and stats["errors"] == 1 and stats["points"] == 4 * 3 * 241
        assert json.loads(_http(url, "/_stub/reset", "POST")[1]) == {"status": "reset"}
        assert json.loads(_http(url, "/_stub/stats")[1])["requests"] == {}


def _chat(system: str, user: str) -> dict:
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def test_llm_stub_tells_the_requests_apart():
    from AI.scoring import JUDGE_SYSTEM_PROMPT
    from loadlens_app.context_import import SYSTEM_PROMPT as CONTEXT_SYSTEM_PROMPT

    assert llm_stub.classify(_chat(JUDGE_SYSTEM_PROMPT, "[]")) == ("judge", "")
    assert llm_stub.classify(_chat(CONTEXT_SYSTEM_PROMPT, "документ")) == ("context_import", "")
    assert llm_stub.classify(_chat("Отвечай одним словом.", "ping")) == ("ping", "")
    assert llm_stub.classify(_chat("system", 'prompt\n\n{"domain": "kafka", "sections": []}')) == ("analysis", "kafka")
    assert llm_stub.classify(_chat("system", '{"designated_peak_performance": {}, "domains": {"jvm": {"domain": "jvm"}}}')) == ("analysis", "final")
    assert llm_stub.classify(_chat("system", '{"analysis": {}, "unverified_findings": []}')) == ("analysis", "final")
    assert llm_stub.classify({"messages": "broken"}) == ("analysis", "unknown")


def test_llm_stub_answers_pass_the_report_parsers():
    from AI.scoring import JUDGE_SYSTEM_PROMPT, _build_judge_prompt, _parse_judge_scores, parse_llm_analysis_strict
    from AI.verification import verify_analysis
    from loadlens_app.context_import import SYSTEM_PROMPT as CONTEXT_SYSTEM_PROMPT, parse_answer

    for domain in [*llm_stub.DOMAINS, "unknown"]:
        content, kind, seen = llm_stub.answer_text(_chat("system", json.dumps({"domain": domain})))
        parsed = parse_llm_analysis_strict(content)
        assert kind == "analysis" and parsed is not None and parsed.findings, domain
        assert not re.search(r"\d", parsed.findings[0].summary)
        _checked, summary = verify_analysis(parsed, {})
        assert summary.unverified == 0  # qualitative findings: no verification pass is triggered
        assert (parsed.peak_performance is not None) == (domain in ("lt_framework", "final"))
    prompt, _meta = _build_judge_prompt(['{"verdict": "Успешно"}'] * 3, '{"domain": "jvm"}', "jvm")
    content, kind, _domain = llm_stub.answer_text(_chat(JUDGE_SYSTEM_PROMPT, prompt))
    assert kind == "judge" and {0, 1, 2} <= set(_parse_judge_scores(content))
    content, kind, _domain = llm_stub.answer_text(_chat(CONTEXT_SYSTEM_PROMPT, "документ"))
    assert kind == "context_import" and parse_answer(content)["system"]["name"] == "Checkout"


def test_llm_stub_delay_is_deterministic_within_the_jitter():
    assert llm_stub.call_delay(b"x", 2.0, 0.0) == 2.0 and llm_stub.call_delay(b"x", 0.0, 0.5) == 0.0
    delays = [llm_stub.call_delay(str(i).encode(), 2.0, 0.25) for i in range(50)]
    assert all(1.5 <= d <= 2.5 for d in delays) and len(set(delays)) > 10
    assert llm_stub.call_delay(b"same", 2.0, 0.25) == llm_stub.call_delay(b"same", 2.0, 0.25)


def test_llm_stub_over_http_answers_the_provider_code(local_without_proxy):
    from AI.providers import ask_llm_with_text_data
    from AI.scoring import parse_llm_analysis_strict

    with _serving(llm_stub, "--delay", "0", "--fail-every", "3") as url:
        llm_config = {"provider": "openai", "max_attempts": 1, "openai": {"api_base_url": f"{url}/v1", "api_key": "stub", "model": "stub-model", "verify": False}}
        text = ask_llm_with_text_data("Проанализируйте домен", json.dumps({"domain": "jvm", "sections": []}), llm_config=llm_config)
        assert parse_llm_analysis_strict(text).verdict == "Успешно"
        ask_llm_with_text_data("prompt", json.dumps({"domain": "kafka"}), llm_config=llm_config)
        with pytest.raises(Exception):
            ask_llm_with_text_data("prompt", json.dumps({"domain": "kafka"}), llm_config=llm_config)  # third call: 500
        stats = bench._stub_call(url, "/_stub/stats")
        assert stats["calls"] == 3 and stats["by_kind"] == {"analysis": 3} and stats["failed"] == 1
        assert stats["by_domain"] == {"jvm": 1, "kafka": 2} and stats["delay_sec"] == 0.0
        status, body, _headers = _http(url, "/v1/models")
        assert status == 200 and json.loads(body)["data"][0]["id"] == "stub-model"
        assert bench._stub_call(url, "/_stub/reset", "POST") == {"status": "reset"}
    assert bench._stub_call("http://llm.example.org:5090", "/_stub/stats") is None  # never leaves the host


def test_confluence_stub_store_keeps_versions_and_attachments():
    store = confluence_stub.PageStore("PERF")
    status, page = store.create({"title": "Отчёт", "space": {"key": "PERF"}, "ancestors": [{"id": "42"}], "body": {"storage": {"value": "<p>a</p>"}}})
    assert status == 200 and page["id"] == "1001" and page["ancestors"] == [{"id": "42"}] and page["version"] == {"number": 1}
    assert store.create({"title": "Отчёт", "space": {"key": "PERF"}})[0] == 400
    assert store.find("PERF", "Отчёт")[0]["id"] == "1001" and store.find("PERF", "нет") == []
    assert store.update("1001", {"version": {"number": 1}, "body": {"storage": {"value": "x"}}})[0] == 409
    status, page = store.update("1001", {"version": {"number": 2}, "title": "Отчёт", "body": {"storage": {"value": "<p>b</p>"}}})
    assert status == 200 and page["version"]["number"] == 2 and page["body"]["storage"]["value"] == "<p>b</p>"
    status, added = store.add_attachment("1001", "chart.png", 10)
    att_id = added["results"][0]["id"]
    assert status == 200 and store.add_attachment("1001", "chart.png", 10)[0] == 400
    assert store.add_attachment("1001", "chart.png", 12, att_id)[1]["version"]["number"] == 2
    assert [a["title"] for a in store.list_attachments("1001")] == ["chart.png"] and store.list_attachments("7") is None
    assert store.summary()[1]["attachments"] == ["chart.png"]
    assert store.delete(att_id) == 204 and store.delete("1001") == 204 and store.delete("42") == 404
    store.reset()
    assert [p["id"] for p in store.summary()] == ["42"]


def test_confluence_stub_serves_the_publication_client(local_without_proxy):
    from loadlens_app import connection_checks
    from loadlens_app.confluence_export import ConfluenceClient

    with _serving(confluence_stub) as url:
        cfg = {"base_url": url, "username": "reader", "password": "secret", "space_key": "PERF", "parent_page_id": "42"}
        client = ConfluenceClient(cfg)
        assert client.get_page("42")["title"] == "Архитектура Checkout" and client.get_page("999") is None
        page = client.create_page("Отчёт bench", "<p>a</p>")
        assert client.page_url(page) == f"{url}/pages/viewpage.action?pageId={page['id']}"
        assert client.find_page_by_title("Отчёт bench")["id"] == page["id"]
        assert client.update_page(page["id"], "Отчёт bench", "<p>b</p>", 1)["version"]["number"] == 2
        client.upload_attachment(page["id"], "rps.png", b"png")
        existing = client.list_attachments(page["id"])
        client.upload_attachment(page["id"], "rps.png", b"png2", existing[0]["id"])
        client.delete_attachment(existing[0]["id"])
        assert connection_checks.check_confluence(cfg).ok
        assert connection_checks.check_confluence({**cfg, "parent_page_id": ""}).ok  # GET /rest/api/space/PERF
        assert _http(url, "/rest/api/content/42")[0] == 401  # no basic auth
        auth = {"Authorization": confluence_stub.expected_auth("reader", "secret")}
        status, _body, headers = _http(url, "/x/Ab42", headers=auth)
        assert status == 302 and headers["Location"].endswith("pageId=42")
        listed = json.loads(_http(url, "/_stub/pages")[1])["pages"]
        assert [(p["id"], p["version"], p["ancestors"]) for p in listed] == [("42", 1, []), (page["id"], 2, ["42"])]
        stats = json.loads(_http(url, "/_stub/stats")[1])["requests"]
        assert stats["POST create page"] == 1 and stats["POST upload attachment"] == 2 and stats["GET unauthorized"] == 1


# ---- --setup-stubs ---------------------------------------------------------------------------------------


def _example_config() -> dict:
    """CONFIG of settings.example.py (the example file, never a real settings.py)."""
    spec = importlib.util.spec_from_file_location("bench_settings_example", ROOT_DIR / "settings.example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def test_stub_runtime_wires_the_area_and_keeps_the_rest():
    existing = {
        "per_area": {"bench": {"title": "Bench", "services": {"svc01": {}}}, "other": {"title": "O"}},
        "llm": {"provider": "gigachat", "self_consistency_k": 1},
    }
    data = bench.stub_settings_runtime(existing, "bench", app_url="http://127.0.0.1:5071")
    assert data["llm"]["provider"] == "openai" and data["llm"]["self_consistency_k"] == 1
    assert data["llm"]["openai"]["api_base_url"] == "http://127.0.0.1:5090/v1"
    assert data["confluence"]["base_url"] == "http://127.0.0.1:5091" and data["confluence"]["enabled"] is True
    assert data["confluence"]["app_url"] == "http://127.0.0.1:5071"
    assert data["data_sources"]["prometheus"]["prometheus"]["url"] == "http://127.0.0.1:5092"
    assert data["limits"]["llm_reports_per_user_per_day"] is None
    assert data["per_area"]["other"] == {"title": "O"} and data["per_area"]["bench"]["services"] == {"svc01": {}}
    assert data["per_area"]["bench"]["domain_sources"] == {"default": {"source": "prometheus"}}
    assert bench.stub_settings_runtime(data, "bench", app_url="http://127.0.0.1:5071") == data
    assert existing["llm"] == {"provider": "gigachat", "self_consistency_k": 1}
    with pytest.raises(ValueError, match="runtime-only"):
        bench.stub_settings_runtime({"per_area": {}}, "bench", app_url="x")


def test_stub_runtime_turns_every_domain_of_the_example_into_promql_on_the_stub():
    from AI.data_sources import domain_query_language, resolve_domain_source

    example = _example_config()
    data = bench.stub_settings_runtime({"per_area": {"bench": {"services": {}}}}, "bench", app_url="x")
    area = data["per_area"]["bench"]
    cfg = {"data_sources": bench._merged(example["data_sources"], data["data_sources"]), "domain_sources": area["domain_sources"]}
    queries = bench._merged(example["queries"], area["queries"])  # as update_page merges the area over CONFIG
    for domain in ("jvm", "database", "kafka", "microservices", "hard_resources", "lt_framework"):
        resolved = resolve_domain_source(cfg, domain)
        assert resolved.prometheus_url == "http://127.0.0.1:5092"
        assert domain_query_language(resolved, queries[domain]) == "promql"
    lt = queries["lt_framework"]
    assert len(lt["promql_queries"]) == len(lt["labels"]) == len(lt["label_keys_list"]) == len(bench.LT_PROMQL_QUERIES)
    assert lt["label_keys_list"] == example["queries"]["lt_framework"]["label_tag_keys_list"]


def test_setup_stubs_writes_a_sandbox_file_and_refuses_a_working_tree(tmp_path, capsys):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"bench": {"services": {"svc01": {}}}}}), encoding="utf-8")
    assert bench.main(["--setup-stubs", "--runtime-json", str(runtime), "--base", "http://127.0.0.1:5079"]) == 0
    written = json.loads(runtime.read_text(encoding="utf-8"))
    assert written["confluence"]["app_url"] == "http://127.0.0.1:5079" and "Restart" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="working tree"):
        bench.main(["--setup-stubs", "--runtime-json", str(ROOT_DIR / "settings_runtime.json")])
    with pytest.raises(SystemExit, match="runtime-only"):
        empty = tmp_path / "empty" / "settings_runtime.json"
        empty.parent.mkdir()
        bench.main(["--setup-stubs", "--runtime-json", str(empty)])


# ---- --during-job: tables, phases, clients ------------------------------------------------------------------


ROWS = [
    {"path": "/dashboard_data", "idle_p50": 20.0, "idle_p95": 30.0, "job_p50": 25.5, "job_p95": 90.0, "job_max": 120.0,
     "idle_app_p50": 12.0, "job_app_p50": 15.0, "job_count": 150, "errors": 0},
    {"path": "/runs?limit=20", "idle_p50": 120.0, "idle_p95": 140.0, "job_p50": 130.0, "job_p95": 400.0, "job_max": 500.0,
     "idle_app_p50": 110.0, "job_app_p50": 118.0, "job_count": 148, "errors": 2},
    {"path": "/job_status/<id>", "idle_p50": None, "idle_p95": None, "job_p50": 6.0, "job_p95": 30.0, "job_max": 31.0,
     "idle_app_p50": None, "job_app_p50": 0.2, "job_count": 60, "errors": 0},
]


def test_during_job_table_renders_and_parses_back():
    table = bench.render_during_table(ROWS)
    lines = table.splitlines()
    assert lines[0] == "| Эндпоинт | idle p50 | idle p95 | job p50 | job p95 | job/idle p50 | job/idle p95 | app idle p50 | app job p50 | Замеров в job | Ошибки |"
    assert lines[2] == "| `/dashboard_data` | 20.0 | 30.0 | 25.5 | 90.0 | 1.27× | 3.00× | 12.0 | 15.0 | 150 | 0 |"
    assert lines[4] == "| `/job_status/<id>` | — | — | 6.0 | 30.0 | — | — | — | 0.2 | 60 | 0 |"
    text = "# Этап 3\n\n" + bench.render_table([{"path": "/", "p50": 1.0, "p95": 2.0, "max": 3.0, "rps": 1.0, "app_p50": 0.5, "bytes": 10, "errors": 0}])
    text += "\n\nthread:\n\n" + table + "\n\n- Задача `x`: done\n\ncelery:\n\n" + bench.render_during_table(ROWS[:1], before=ROWS) + "\n"
    tables = bench.parse_during_tables(text)
    assert len(tables) == 2
    keys = ("path", "idle_p50", "idle_p95", "job_p50", "job_p95", "idle_app_p50", "job_app_p50", "job_count", "errors")
    assert tables[0] == [{key: row[key] for key in keys} for row in ROWS]
    assert tables[1][0]["before_job_p95"] == 90.0 and tables[1][0]["job_p95"] == 90.0
    compared = bench.render_during_table(ROWS[:2], before=[{"path": "/runs?limit=20", "job_p50": 260.0, "job_p95": 800.0}])
    assert compared.splitlines()[0].endswith("| было job p50 | было job p95 | было/стало job p95 |")
    assert compared.splitlines()[2].endswith("| 0 | — | — | — |")
    assert compared.splitlines()[3].endswith("| 2 | 260.0 | 800.0 | 2.00× |")


def test_during_baseline_comes_from_json_or_markdown(tmp_path):
    md = tmp_path / "baseline.md"
    md.write_text("text\n\n" + bench.render_during_table(ROWS) + "\n\nmore\n\n" + bench.render_during_table(ROWS[:1]) + "\n", encoding="utf-8")
    assert [row["path"] for row in bench.load_during_baseline(str(md))] == [row["path"] for row in ROWS]
    assert len(bench.load_during_baseline(str(md), 2)) == 1
    with pytest.raises(SystemExit):
        bench.load_during_baseline(str(md), 3)
    saved = tmp_path / "thread.json"
    saved.write_text(json.dumps({"rows": ROWS}), encoding="utf-8")
    assert bench.load_during_baseline(str(saved)) == ROWS
    saved.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="during-job"):
        bench.load_during_baseline(str(saved))


def test_phase_stats_rows_and_phases():
    samples = [
        ("/a", 0.5, (200, 10.0, 8.0, 100)), ("/a", 1.5, (200, 20.0, 18.0, 100)), ("/a", 2.5, (500, 30.0, None, 5)),
        ("/b", 1.0, (0, 40.0, None, 0)), ("/b", 9.0, (200, 50.0, 1.0, 1)),
    ]
    window = bench.phase_stats(samples, since=1.0, until=5.0)
    assert window["/a"]["count"] == 2 and window["/a"]["errors"] == 1 and window["/a"]["statuses"] == {"200": 1, "500": 1}
    assert window["/b"]["count"] == 1 and window["/b"]["app_p50"] is None
    idle = bench.phase_stats(samples, until=1.0)
    rows = bench.during_rows(idle, {**window, bench.JOB_STATUS_ROW: {"p50": 5.0, "p95": 6.0, "count": 3, "errors": 0}}, ["/a", "/b"])
    assert [row["path"] for row in rows] == ["/a", "/b", bench.JOB_STATUS_ROW]
    assert rows[0]["idle_p50"] == 10.0 and rows[0]["job_p50"] == 25.0 and rows[0]["errors"] == 1
    assert rows[1]["idle_p50"] is None and rows[1]["errors"] == 1 and rows[2]["idle_p95"] is None

    timeline = [(0.2, "running", 5, "Создание"), (3.0, "running", 35, "Сбор"), (12.5, "running", 52, "Анализ ИИ"),
                (40.0, "running", 83, "Итоговый"), (47.5, "running", 95, "Сохранение"), (48.0, "done", 100, "Готово")]
    assert bench.job_phases(timeline, 48.0) == [("сбор метрик и подготовка", 12.5), ("анализ ИИ по доменам", 27.5), ("итоговый анализ и сохранение", 8.0)]
    assert bench.job_phase_windows(timeline, 48.0)[1] == ("анализ ИИ по доменам", 12.5, 40.0)
    assert bench.job_phases(timeline[:3], None) == [("сбор метрик и подготовка", 12.5), ("анализ ИИ по доменам", None), ("итоговый анализ и сохранение", None)]
    table = bench.render_phase_table([("сбор", window), ("анализ", idle)], ["/a", "/b"]).splitlines()
    assert table[0] == "| Эндпоинт | сбор: p50 | сбор: p95 | сбор: n | анализ: p50 | анализ: p95 | анализ: n |"
    assert table[3] == "| `/b` | 40.0 | 40.0 | 1 | — | — | 0 |"


def test_csrf_token_window_and_arguments():
    from loadlens_app.core import convert_to_timestamp

    page = '<head>\n  <meta charset="utf-8" />\n  <meta name="csrf-token" content="abc-123_x" />\n</head>'
    assert bench.extract_csrf_token(page) == "abc-123_x" and bench.extract_csrf_token('<meta name="csrf-token" content="" />') == ""
    assert bench.extract_csrf_token("<html></html>") == ""
    start, end = bench.report_window(datetime(2026, 10, 16, 15, 42, 37, 120000, tzinfo=timezone(timedelta(hours=3))), 60)
    assert (start, end) == ("2026-10-16T11:42:00+00:00", "2026-10-16T12:42:00+00:00")
    assert convert_to_timestamp(end) - convert_to_timestamp(start) == 3_600_000
    args = bench.parse_args(["--sandbox", "1", "--during-job"])
    assert (args.service, args.window_min, args.test_type, args.keep_report, args.pause_ms) == ("svc01", 60, "step", False, 50.0)
    assert bench.parse_args(["--during-job", "--concurrency", "4"]).pause_ms == 200.0  # same request rate as one client
    assert bench.parse_args(["--during-job", "--concurrency", "4", "--pause-ms", "0"]).pause_ms == 0.0
    assert bench.selected_endpoints(args, bench.DURING_JOB_ENDPOINTS) == list(bench.DURING_JOB_ENDPOINTS)
    assert bench.selected_endpoints(bench.parse_args(["--during-job", "--only", "runs"]), bench.DURING_JOB_ENDPOINTS) == ["/runs?limit=20", "/runs?limit=20&sort=max_rps"]
    for clash in (["--during-job", "--explain"], ["--setup-stubs", "--during-job"], ["--during-job", "--window-min", "0"]):
        with pytest.raises(SystemExit):
            bench.parse_args(clash)


class _CountingClient:
    def __init__(self, target):
        self.paths = []
        self.closed = False
        _CountingClient.made.append(self)

    def get(self, path):
        self.paths.append(path)
        return (200, 1.0, 0.5, 10)

    def close(self):
        self.closed = True


def test_cycle_clients_go_round_the_paths_from_different_offsets():
    _CountingClient.made = []
    samples = bench.cycle_clients(None, ["/a", "/b", "/c"], 2, pause_sec=0, stop=threading.Event(), cycles=3, client_factory=_CountingClient)
    assert len(samples) == 18 and Counter(path for path, _t, _s in samples) == {"/a": 6, "/b": 6, "/c": 6}
    assert sorted(client.paths[0] for client in _CountingClient.made) == ["/a", "/b"]
    assert all(client.closed for client in _CountingClient.made)
    stopped = threading.Event()
    stopped.set()
    assert bench.cycle_clients(None, ["/a"], 3, pause_sec=0, stop=stopped, client_factory=_CountingClient) == []


class _CookieHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    close_after = False

    def do_GET(self):  # noqa: N802
        data = json.dumps({"cookie": self.headers.get("Cookie") or ""}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Set-Cookie", "session=s1; HttpOnly; Path=/")
        self.send_header("Server-Timing", "app;dur=2.5")
        if self.close_after:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("close_after", [False, True])
def test_keep_alive_client_keeps_cookies_and_reconnects(close_after):
    handler = type("Handler", (_CookieHandler,), {"close_after": close_after})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = bench.KeepAliveClient(bench.Target(f"http://127.0.0.1:{server.server_address[1]}", "bench"))
        status, _ms, app, body = client.request("GET", "/one")
        assert status == 200 and app == 2.5 and json.loads(body) == {"cookie": "project_area=bench"}
        assert (client.connection is None) == close_after
        status, _ms, _app, body = client.request("GET", "/two")
        assert json.loads(body) == {"cookie": "project_area=bench; session=s1"}
        assert client.get("/three")[0] == 200
        client.close()
    finally:
        server.shutdown()
        server.server_close()
    assert client.get("/after-shutdown")[0] == 0  # no answer: status 0, connection dropped
    assert client.connection is None


class _FakeApp(BaseHTTPRequestHandler):
    """Just enough of LoadLens for --during-job: a page with a CSRF token, a report job, its report."""

    protocol_version = "HTTP/1.1"
    state: dict = {}

    def _reply(self, status, payload, content_type="application/json", extra=None):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Server-Timing", "app;dur=1.5")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        state = self.state
        if self.path == "/new":
            page = b'<html><head><meta name="csrf-token" content="tok-1" /></head></html>'
            return self._reply(200, page, "text/html", {"Set-Cookie": "session=s1; HttpOnly; Path=/"})
        if self.path.startswith("/job_status/"):
            state["polls"] += 1
            progress = (10, 50, 90, 100)[min(state["polls"], 4) - 1]
            final = state["polls"] >= 4
            return self._reply(200, {
                "job_id": "j1", "status": (state["outcome"] if final else "running"), "progress": progress, "message": "…",
                "page_url": "/reports/r1" if final else None, "run_name": state.get("run_name"),
                "created_at": "2026-10-16T10:00:00+00:00", "updated_at": "2026-10-16T10:00:42.500000+00:00",
                "error": "boom" if final and state["outcome"] == "error" else None,
            })
        if self.path == "/reports/r1":
            return self._reply(200, b"<html></html>", "text/html")
        if self.path.startswith("/llm_reports?"):
            state["llm_reports"] = self.path
            return self._reply(200, [{"domain": "final", "verdict": "Успешно"}, {"domain": "jvm", "verdict": "Успешно"}])
        state["gets"].append(self.path)
        return self._reply(200, {"ok": True})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        self.state["posted"] = (self.path, body, self.headers.get("X-CSRF-Token"), self.headers.get("Cookie"))
        self.state["run_name"] = body.get("run_name")
        return self._reply(200, {"status": "accepted", "job_id": "j1", "run_name": body.get("run_name")})

    def do_DELETE(self):  # noqa: N802
        self.state["deleted"] = (self.path, self.headers.get("X-CSRF-Token"))
        return self._reply(200, {"status": "ok"})

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("outcome", ["done", "error"])
def test_during_job_runs_the_report_and_prints_both_tables(outcome, tmp_path, capsys):
    from loadlens_app.core import convert_to_timestamp

    state = {"polls": 0, "gets": [], "outcome": outcome}
    server = ThreadingHTTPServer(("127.0.0.1", 0), type("App", (_FakeApp,), {"state": state}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "during.json"
    try:
        code = bench.main([
            "--base", f"http://127.0.0.1:{server.server_address[1]}", "--during-job", "--rounds", "2", "--warmup", "0",
            "--pause-ms", "5", "--poll-sec", "0.05", "--llm-stub", "", "--prom-stub", "", "--label", "thread", "--json", str(out),
        ])
    finally:
        server.shutdown()
        server.server_close()
    printed = capsys.readouterr().out
    saved = json.loads(out.read_text(encoding="utf-8"))
    path, body, csrf, cookie = state["posted"]
    assert path == "/create_report" and csrf == "tok-1" and "session=s1" in cookie and "project_area=bench" in cookie
    assert (body["service"], body["project_area"], body["web_only"], body["use_llm"], body["test_type"]) == ("svc01", "bench", True, True, "step")
    assert body["run_name"].startswith("bench-job-") and convert_to_timestamp(body["end"]) - convert_to_timestamp(body["start"]) == 3_600_000
    assert set(bench.DURING_JOB_ENDPOINTS) <= set(state["gets"]) and state["polls"] == 4
    assert [row["path"] for row in saved["rows"]] == [*bench.DURING_JOB_ENDPOINTS, bench.JOB_STATUS_ROW]
    assert all(row["idle_p50"] is not None for row in saved["rows"][:-1])
    job = saved["job"]
    assert job["status"] == outcome and job["label"] == "thread" and job["server_duration_sec"] == 42.5
    assert job["idle_drift"] is not None and saved["idle"]["/"]["count"] == 4  # two rounds before the job, two after
    assert saved["idle_before"]["/"]["count"] == saved["idle_after"]["/"]["count"] == 2
    assert [name for name, _seconds in job["phases"]] == [name for name, _mark in bench.JOB_PHASES]
    tables = bench.parse_during_tables(printed)
    assert len(tables) == 1 and tables[0][-1]["path"] == bench.JOB_STATUS_ROW and tables[0][-1]["job_count"] >= 4
    if outcome == "done":
        assert code == 0 and job["report"]["ok"] and job["report"]["final_verdict"] == "Успешно" and job["report"]["domains"] == 2
        assert state["deleted"] == ("/runs/" + quote(body["run_name"], safe=""), "tok-1")
        assert "удалён (`DELETE /runs` → 200)" in printed and "По фазам задачи" in printed
    else:
        assert code == 1 and "report" not in job and "deleted" not in state
        assert "- Ошибка: boom" in printed


def test_job_summary_lists_duration_phases_stubs_and_report():
    job = {
        "job_id": "abc", "label": "thread", "status": "done", "duration_sec": 47.6, "server_duration_sec": 47.0,
        "phases": [["сбор метрик и подготовка", 16.3], ["анализ ИИ по доменам", 24.2], ["итоговый анализ и сохранение", None]],
        "llm_stub": {"calls": 28, "by_kind": {"analysis": 21, "judge": 7}, "delay_sec": 2.0, "max_in_flight": 2},
        "prometheus_stub": {"requests": {"query_range": 69}, "points": 99774},
        "report": {"run_name": "bench-job-1", "page_url": "/reports/r1", "page_status": 200, "llm_status": 200, "domains": 7, "final_verdict": "Успешно", "deleted": 200},
        "rows": [{"path": "/x", "job_count": 3}, {"path": "/y", "job_count": 50}],
        "requests": 1000, "no_answer": 10,
    }
    assert "Без ответа" not in bench.render_job_summary(job)  # 1 %: within the noise
    job["no_answer"] = 11
    warned = [line for line in bench.render_job_summary(job).splitlines() if line.startswith("- Без ответа")]
    assert warned and warned[0].startswith("- Без ответа (статус 0) 11 из 1000 запросов: скорее всего, исчерпаны временные порты")
    job["no_answer"] = 0
    lines = bench.render_job_summary(job).splitlines()
    assert lines[0] == "- Задача `abc` (thread): done, 47.6 с по опросу `/job_status`, 47.0 с по `created_at`/`updated_at` задачи"
    assert lines[1] == "- Фазы: сбор метрик и подготовка 16.3 с · анализ ИИ по доменам 24.2 с"
    assert lines[2] == "- LLM-заглушка: 28 вызовов (analysis 21, judge 7), задержка 2.0 с на ответ, одновременно до 2"
    assert lines[3] == "- Prometheus-заглушка: 69 запросов `query_range`, 99 774 точек"
    assert lines[4].startswith("- Отчёт `bench-job-1`: `/reports/r1` → 200, `/llm_reports` → 200, доменов 7, итог «Успешно»; удалён")
    assert lines[5] == "- Мало замеров во время задачи (меньше 10): `/x`"
    job["idle_drift"] = 1.1
    assert "- Idle до и после задачи (в таблице оба вместе): p50 после / до = 1.10×" in bench.render_job_summary(job).splitlines()
    job["idle_drift"] = 1.6
    assert "неравномерно" in bench.render_job_summary(job)


def test_idle_drift_is_the_median_ratio_of_p50():
    before = {"/a": {"p50": 10.0}, "/b": {"p50": 20.0}, "/c": {"p50": None}}
    after = {"/a": {"p50": 12.0}, "/b": {"p50": 30.0}, "/c": {"p50": 5.0}, "/d": {"p50": 1.0}}
    assert bench.idle_drift(before, after) == 1.35
    assert bench.idle_drift(before, {}) is None
