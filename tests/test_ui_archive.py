"""Archive API (ui-plan stage 3): /runs normalization, new fields and sorting; /compare_baseline lookup.

The database is faked: RecordingConnection logs every statement and serves scripted fetch results.
"""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI.db_store import BaselineRun  # noqa: E402

END = datetime(2026, 10, 1, 11, 0, tzinfo=timezone.utc)
START = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)


class RecordingCursor:
    def __init__(self, log, fetchone_results, fetchall_results):
        self._log = log
        self._fetchone = fetchone_results
        self._fetchall = fetchall_results

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self._log.append((" ".join(str(query).split()), params))

    def fetchone(self):
        return self._fetchone.pop(0) if self._fetchone else None

    def fetchall(self):
        return self._fetchall.pop(0) if self._fetchall else []


class RecordingConnection:
    def __init__(self, fetchone_results=None, fetchall_results=None):
        self.log = []
        self.closed = False
        self._cursor = RecordingCursor(self.log, list(fetchone_results or []), list(fetchall_results or []))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _isolated_runtime(monkeypatch, tmp_path):
    import update_page

    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    monkeypatch.setattr(core, "_ts_conn", lambda: RecordingConnection())


@pytest.fixture
def client(monkeypatch):
    from app import create_app
    from loadlens_app.blueprints import compare, dashboard

    for module in (dashboard, compare):
        monkeypatch.setattr(module, "_ts_conn", lambda: RecordingConnection())
        monkeypatch.setattr(module, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_ensure_run_index_table", lambda conn, cfg: None)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def use_conn(monkeypatch, module_name, conn):
    from loadlens_app.blueprints import compare, dashboard

    module = {"dashboard": dashboard, "compare": compare}[module_name]
    monkeypatch.setattr(module, "_ts_conn", lambda: conn)
    return conn


# ---- /runs -----------------------------------------------------------------------------------

def test_runs_rows_carry_project_area_max_rps_and_type_key(client, monkeypatch):
    # The last column is the COUNT(*) OVER () total of the single /runs statement.
    rows = [
        ("nightly-1", START, END, "checkout", "Успешно", "Успешно", None, END, "Stability", "rid-1", "devstack", Decimal("1234.5"), 2),
        ("nightly-2", START, END, "checkout", "Провал", None, None, None, "", "", "", None, 2),
    ]
    conn = use_conn(monkeypatch, "dashboard", RecordingConnection(fetchone_results=[(2,)], fetchall_results=[rows]))

    resp = client.get("/runs")

    assert resp.status_code == 200
    first, second = resp.get_json()
    assert first["project_area"] == "devstack"
    assert first["max_rps"] == 1234.5
    assert first["test_type"] == "Stability" and first["test_type_key"] == "soak"
    assert second["max_rps"] is None and second["project_area"] == "" and second["test_type_key"] == ""
    page_sql, _ = conn.log[0]
    assert "project_area, max_rps, COUNT(*) OVER () AS total FROM joined" in page_sql
    assert "->'peak_performance'->>'max_rps'" in page_sql
    assert "->'stability_under_load'->>'actual_rps'" in page_sql
    assert "->'spike_analysis'->>'max_spike_rps'" in page_sql
    assert "::numeric" in page_sql


def test_runs_non_finite_max_rps_becomes_null(client, monkeypatch):
    rows = [("r", START, END, "svc", "Успешно", None, None, END, "step", "", "", Decimal("Infinity"), 1)]
    use_conn(monkeypatch, "dashboard", RecordingConnection(fetchone_results=[(1,)], fetchall_results=[rows]))

    body = client.get("/runs").get_json()

    assert body[0]["max_rps"] is None


@pytest.mark.parametrize(
    ("requested", "bound"),
    [("step", "step"), ("soak", "soak"), ("Stability", "soak"), ("long run", "soak"), ("ступенчатый", "step"),
     ("Burst", "spike"), ("STRESS", "stress"), ("custom-type", "custom-type")],
)
def test_runs_test_type_filter_is_normalized(client, monkeypatch, requested, bound):
    conn = use_conn(monkeypatch, "dashboard", RecordingConnection(fetchone_results=[(0,)], fetchall_results=[[]]))

    assert client.get("/runs", query_string={"test_type": requested}).status_code == 200

    page_sql, page_params = conn.log[0]
    assert "WHERE test_type = %s" in page_sql
    assert page_params[-3] == bound  # followed by offset and limit


def test_runs_test_type_column_maps_aliases_like_the_baseline_lookup(client, monkeypatch):
    from AI.db_store import _test_type_filter_values

    conn = use_conn(monkeypatch, "dashboard", RecordingConnection(fetchone_results=[(0,)], fetchall_results=[[]]))
    client.get("/runs", query_string={"test_type": "soak"})

    count_sql, _ = conn.log[0]
    # Runs without a final report have no type; reports saved without one count as step.
    assert "CASE WHEN f.run_name IS NULL THEN ''" in count_sql
    for key in ("step", "soak", "spike", "stress"):
        values = _test_type_filter_values(key)
        listed = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
        assert f"IN ({listed}) THEN '{key}'" in count_sql
    assert "IN ('', " in count_sql
    assert "COALESCE(f.test_type, '') AS test_type_raw" in count_sql


def test_runs_sorts_by_max_rps_with_nulls_last_and_a_tiebreak(client, monkeypatch):
    conn = use_conn(monkeypatch, "dashboard", RecordingConnection(fetchone_results=[(0,)], fetchall_results=[[]]))

    client.get("/runs", query_string={"sort": "max_rps", "dir": "asc"})

    page_sql, _ = conn.log[0]
    assert "ORDER BY max_rps ASC NULLS LAST, run_name ASC" in page_sql


@pytest.mark.parametrize(
    ("args", "order"),
    [
        ({"sort": "run_name", "dir": "desc"}, "ORDER BY run_name DESC OFFSET"),
        ({"sort": "bogus; DROP TABLE x"}, "ORDER BY end_time DESC, run_name ASC"),
        ({"sort": "verdict", "dir": "sideways"}, "ORDER BY verdict DESC, run_name ASC"),
        ({"sort": "test_type", "dir": "asc"}, "ORDER BY test_type ASC, run_name ASC"),
    ],
)
def test_runs_sort_is_whitelisted(client, monkeypatch, args, order):
    conn = use_conn(monkeypatch, "dashboard", RecordingConnection(fetchone_results=[(0,)], fetchall_results=[[]]))

    assert client.get("/runs", query_string=args).status_code == 200

    page_sql, _ = conn.log[0]
    assert order in page_sql
    assert "DROP" not in page_sql


def test_runs_sql_literal_escapes_quotes_and_percent():
    from loadlens_app.blueprints.dashboard import _runs_sql_literal

    assert _runs_sql_literal("it's 50%") == "'it''s 50%%'"


# ---- /compare_baseline -------------------------------------------------------------------------

def _baseline_calls(monkeypatch, result):
    import AI.db_store as db_store

    calls = []

    def fake_find_previous_run(conn, schema, table, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(db_store, "find_previous_run", fake_find_previous_run)
    return calls


def test_compare_baseline_uses_service_area_type_and_time_of_the_run(client, monkeypatch):
    created = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
    conn = use_conn(monkeypatch, "compare", RecordingConnection(fetchone_results=[("checkout", created, "Stability", "devstack")]))
    previous = BaselineRun(run_name="nightly-0", service="checkout", verdict="Успешно", created_at=END)
    calls = _baseline_calls(monkeypatch, previous)

    resp = client.get("/compare_baseline", query_string={"run_name": "nightly-1", "mode": "previous"})

    assert resp.status_code == 200
    assert resp.get_json() == {
        "run_name": "nightly-0", "service": "checkout", "verdict": "Успешно",
        "created_at": END.isoformat(), "mode": "previous",
    }
    assert calls == [{
        "exclude_run_name": "nightly-1", "mode": "previous", "service": "checkout", "before": created,
        "project_area": "devstack", "test_type": "soak",
    }]
    lookup_sql, lookup_params = conn.log[0]
    assert "WHERE run_name = %s AND domain = 'final'" in lookup_sql
    assert "ORDER BY created_at DESC LIMIT 1" in lookup_sql
    assert lookup_params == ("nightly-1",)
    assert conn.closed


def test_compare_baseline_defaults_to_previous_success_and_the_active_area(client, monkeypatch):
    created = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
    use_conn(monkeypatch, "compare", RecordingConnection(fetchone_results=[("checkout", created, "", "")]))
    calls = _baseline_calls(monkeypatch, BaselineRun("base", "checkout", "Успешно", None))
    client.set_cookie("project_area", "devstack")

    resp = client.get("/compare_baseline", query_string={"run_name": "nightly-1"})

    assert resp.status_code == 200 and resp.get_json()["created_at"] is None
    assert calls[0]["mode"] == "previous_success"
    # The run was saved without an area or a type: the active area applies, the type does not.
    assert calls[0]["project_area"] == "devstack"
    assert calls[0]["test_type"] is None


def test_compare_baseline_looks_up_the_run_within_the_active_area(client, monkeypatch):
    from loadlens_app.blueprints import compare

    conn = use_conn(monkeypatch, "compare", RecordingConnection(fetchone_results=[None]))
    monkeypatch.setattr(compare, "_resolve_services_filter", lambda area: ["checkout", "orders"])
    calls = _baseline_calls(monkeypatch, None)

    resp = client.get("/compare_baseline", query_string={"run_name": "elsewhere"})

    assert resp.status_code == 404
    assert resp.get_json()["mode"] == "previous_success"
    assert "предыдущий успешный прогон" in resp.get_json()["error"]
    lookup_sql, lookup_params = conn.log[0]
    assert "AND service = ANY(%s)" in lookup_sql
    assert lookup_params == ("elsewhere", ["checkout", "orders"])
    assert calls == []


def test_compare_baseline_not_found_names_the_type(client, monkeypatch):
    created = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
    use_conn(monkeypatch, "compare", RecordingConnection(fetchone_results=[("checkout", created, "spike", "")]))
    _baseline_calls(monkeypatch, None)

    resp = client.get("/compare_baseline", query_string={"run_name": "nightly-1", "mode": "previous"})

    assert resp.status_code == 404
    body = resp.get_json()
    assert body["mode"] == "previous"
    assert "предыдущий прогон того же сервиса" in body["error"] and "«spike»" in body["error"]


def test_compare_baseline_validates_params(client):
    assert client.get("/compare_baseline").status_code == 400
    resp = client.get("/compare_baseline", query_string={"run_name": "x", "mode": "best"})
    assert resp.status_code == 400 and "best" in resp.get_json()["error"]


def test_compare_baseline_sql_against_find_previous_run(client, monkeypatch):
    """Without the stub: the second statement is find_previous_run's query with type and area filters."""
    created = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
    conn = use_conn(monkeypatch, "compare", RecordingConnection(fetchone_results=[
        ("checkout", created, "soak", "devstack"),
        ("nightly-0", "checkout", "Успешно", END),
    ]))

    resp = client.get("/compare_baseline", query_string={"run_name": "nightly-1"})

    assert resp.status_code == 200 and resp.get_json()["run_name"] == "nightly-0"
    baseline_sql, params = conn.log[1]
    assert "= 'Успешно'" in baseline_sql
    assert "LOWER(TRIM(COALESCE(r.test_type, ''))) = ANY(%s)" in baseline_sql
    assert "r.project_area = %s" in baseline_sql
    assert params[0] == "checkout" and params[1] == "nightly-1" and params[2] == created
    assert "stability" in params[3] and params[4] == "devstack"
