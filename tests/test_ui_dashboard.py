"""/dashboard_data of the second UI wave: service history, KPI, verdicts per week, SLA of the last run, dedup.

The database is a fake that answers each statement of the endpoint by a recognizable fragment of its SQL,
so the order of the statements does not matter to the tests.
"""

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)  # a Thursday; its week starts on 2026-10-12
WEEKS = ["2026-08-24", "2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28", "2026-10-05", "2026-10-12"]
LAST_CREATED = datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc)
SLA_DETAILS = {
    "checks": [
        {"name": "p95_latency", "passed": True},
        {"name": "error_rate", "passed": False},
        {"name": "target_rps", "passed": True},
        {"name": "cpu_usage", "passed": None},
    ]
}

# Statement fragment -> rows; checked in this order.
ROUTES = (
    ("previous_rps", "AND run_name = %s AND service = %s"),
    ("previous", "r.run_name <> %s"),
    ("last_run", "parsed->'deterministic_sla'->'checks'"),
    ("verdict_counts", "AS cnt"),
    ("runs_total", "COUNT(DISTINCT run_name)"),
    ("services", "PARTITION BY service"),
    ("weeks", "date_trunc('week'"),
    ("recent", "start_ms, end_ms, run_id FROM latest"),
)


class RoutedCursor:
    def __init__(self, db):
        self.db = db
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = " ".join(str(query).split())
        for key, fragment in ROUTES:
            if fragment in text:
                self.db.calls.append((key, text, params))
                result = self.db.results.get(key, [])
                if isinstance(result, Exception):
                    raise result
                self.rows = list(result)
                return
        self.db.calls.append(("unknown", text, params))
        self.rows = []

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class RoutedDb:
    def __init__(self, **results):
        self.results = results
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return RoutedCursor(self)

    def close(self):
        return None

    def rollback(self):
        self.calls.append(("rollback", "", None))

    def call(self, key):
        found = [c for c in self.calls if c[0] == key]
        assert found, f"statement '{key}' was not executed: {[c[0] for c in self.calls]}"
        return found[-1]


def last_row(**overrides):
    row = {
        "run_name": "nightly-3", "service": "checkout", "start_ms": 1_000, "end_ms": 3_601_000,
        "verdict": "Есть риски", "llm_verdict": "Успешно", "sla_verdict": "Есть риски", "created_at": LAST_CREATED,
        "run_id": "rid-3", "test_type": "step", "max_rps": "240.5", "sla_details": SLA_DETAILS, "fallback": None,
    }
    row.update(overrides)
    return tuple(row.values())


def full_db(**overrides):
    results = {
        "last_run": [last_row()],
        "verdict_counts": [("Успешно", 2), ("Есть риски", 1)],
        "runs_total": [(4,)],
        "services": [
            ("checkout", "nightly-3", "Есть риски", LAST_CREATED, "240.5", "step", "rid-3"),
            ("checkout", "soak-1", "Успешно", datetime(2026, 10, 10, tzinfo=timezone.utc), "120", "soak", ""),
            ("checkout", "nightly-2", "Успешно", datetime(2026, 10, 1, tzinfo=timezone.utc), "200", "step", "rid-2"),
            ("billing", "spike-1", "Провал", datetime(2026, 10, 2, tzinfo=timezone.utc), "≈90 rps", "spike", None),
        ],
        "recent": [("nightly-3", "checkout", "Есть риски", LAST_CREATED, "step", 1_000, 3_601_000, "rid-3")],
        "weeks": [
            (date(2026, 10, 12), "Есть риски", 1, 1, 1),
            (date(2026, 10, 5), "Успешно", 2, 0, 2),
            (date(2026, 9, 28), "Провал", 1, 0, 1),
            (date(2026, 9, 7), "что-то новое", 1, 0, 0),
            (date(2026, 6, 1), "Успешно", 5, 0, 0),
        ],
        "previous": [("nightly-2", "checkout", "Успешно", datetime(2026, 10, 1, tzinfo=timezone.utc))],
        "previous_rps": [("rid-2", "200")],
    }
    results.update(overrides)
    return RoutedDb(**results)


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    from app import create_app
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(dashboard, "_dashboard_now", lambda: NOW)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def fetch(client, monkeypatch, db, cookie_area=None):
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_ts_conn", lambda: db)
    if cookie_area:
        client.set_cookie("project_area", cookie_area)
    resp = client.get("/dashboard_data")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()


def test_services_carry_history_newest_first(app_client, monkeypatch):
    data = fetch(app_client, monkeypatch, full_db())
    services = {s["service"]: s for s in data["services"]}
    checkout = services["checkout"]
    assert [h["run_name"] for h in checkout["history"]] == ["nightly-3", "soak-1", "nightly-2"]
    assert checkout["history"][0] == {
        "run_name": "nightly-3", "run_id": "rid-3", "created_at": LAST_CREATED.isoformat(),
        "verdict": "Есть риски", "max_rps": pytest.approx(240.5), "test_type": "step",
    }
    assert checkout["history"][1]["max_rps"] == pytest.approx(120.0)
    assert checkout["recent_verdicts"] == ["Есть риски", "Успешно", "Успешно"]
    assert checkout["run_id"] == "rid-3" and checkout["last_run"] == "nightly-3"
    # Text the LLM wrote instead of a number is no RPS, not a server error.
    assert services["billing"]["max_rps"] is None and services["billing"]["history"][0]["max_rps"] is None
    assert services["billing"]["history"][0]["run_id"] == ""
    assert [s["service"] for s in data["services"]] == ["checkout", "billing"]


def test_runs_are_deduplicated_by_name(app_client, monkeypatch):
    db = full_db()
    fetch(app_client, monkeypatch, db)
    services_sql = db.call("services")[1]
    recent_sql = db.call("recent")[1]
    weeks_sql = db.call("weeks")[1]
    for text in (services_sql, recent_sql, weeks_sql):
        assert "ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC)" in text
    assert "FROM latest WHERE run_rn = 1" in services_sql
    assert "FROM latest WHERE rn = 1 ORDER BY created_at DESC LIMIT %s" in recent_sql
    assert "COALESCE(run_id, '')" in services_sql


def test_kpi_counts_and_verdict_weeks(app_client, monkeypatch):
    db = full_db()
    data = fetch(app_client, monkeypatch, db)
    kpi = data["kpi"]
    assert kpi["runs_7d"] == 1
    assert kpi["runs_30d"] == 4
    assert kpi["risky_30d"] == 2
    assert kpi["risky_share_30d"] == pytest.approx(0.5)
    weeks = data["verdict_weeks"]
    assert [w["week_start"] for w in weeks] == WEEKS
    by_week = {w["week_start"]: w for w in weeks}
    assert by_week["2026-10-12"]["counts"] == {"Успешно": 0, "Есть риски": 1, "Провал": 0, "Недостаточно данных": 0}
    assert by_week["2026-10-05"]["counts"]["Успешно"] == 2 and by_week["2026-10-05"]["total"] == 2
    assert by_week["2026-09-28"]["counts"]["Провал"] == 1
    # An unknown verdict counts as «Недостаточно данных»; a week outside the range is dropped.
    assert by_week["2026-09-07"]["counts"]["Недостаточно данных"] == 1
    assert sum(w["total"] for w in weeks) == 5
    # The statement gets the period bounds: first week start (UTC), then 7 and 30 days back.
    params = db.call("weeks")[2]
    assert params[0] == datetime(2026, 8, 24, tzinfo=timezone.utc)
    assert params[-2] == datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    assert params[-1] == datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    # Compatibility: the latest-slice counts stay.
    assert data["verdict_counts"]["Успешно"] == 2 and data["runs_total"] == 4


def test_last_run_sla_and_kpi_last(app_client, monkeypatch):
    data = fetch(app_client, monkeypatch, full_db())
    last = data["last_run"]
    assert last["sla"] == {"passed": 2, "failed": 1, "total": 3}
    assert last["run_id"] == "rid-3" and last["test_type"] == "step"
    assert last["max_rps"] == pytest.approx(240.5)
    assert last["verdict"] == "Есть риски" and last["llm_verdict"] == "Успешно" and last["sla_verdict"] == "Есть риски"
    assert data["kpi"]["last"] == {"run_name": "nightly-3", "max_rps": pytest.approx(240.5), "test_type": "step"}


def test_last_run_sla_falls_back_to_deterministic_checks(app_client, monkeypatch):
    fallback = json.dumps([{"name": "p95_latency", "passed": True}, {"name": "target_rps", "passed": True}])
    db = full_db(last_run=[last_row(sla_details=json.dumps({"checks": []}), fallback=fallback)])
    data = fetch(app_client, monkeypatch, db)
    assert data["last_run"]["sla"] == {"passed": 2, "failed": 0, "total": 2}

    db = full_db(last_run=[last_row(sla_details=None, fallback=None)])
    assert fetch(app_client, monkeypatch, db)["last_run"]["sla"] == {"passed": 0, "failed": 0, "total": 0}


def test_previous_run_of_the_same_service_and_type(app_client, monkeypatch):
    db = full_db()
    data = fetch(app_client, monkeypatch, db, cookie_area="shop")
    assert data["kpi"]["previous"] == {
        "run_name": "nightly-2", "run_id": "rid-2", "max_rps": pytest.approx(200.0),
        "verdict": "Успешно", "created_at": "2026-10-01T00:00:00+00:00",
    }
    _, text, params = db.call("previous")
    # Any earlier run (not only a successful one), same service, same normalized type, same area.
    assert "'Успешно'" not in text
    assert "LOWER(TRIM(COALESCE(r.test_type, ''))) = ANY(%s)" in text
    assert "r.project_area = %s" in text
    assert params[0] == "checkout" and params[1] == "nightly-3" and params[2] == LAST_CREATED
    assert "step" in params[3] and "" in params[3]
    assert params[4] == "shop"
    assert db.call("previous_rps")[2] == ("nightly-2", "checkout")


def test_no_previous_run(app_client, monkeypatch):
    db = full_db(previous=[])
    data = fetch(app_client, monkeypatch, db)
    assert data["kpi"]["previous"] is None
    assert db.call("previous")
    assert not [c for c in db.calls if c[0] == "previous_rps"]


def test_failed_previous_lookup_keeps_the_dashboard(app_client, monkeypatch):
    db = full_db(previous=RuntimeError("relation is gone"))
    data = fetch(app_client, monkeypatch, db)
    assert data["kpi"]["previous"] is None
    assert data["kpi"]["runs_30d"] == 4 and data["last_run"]["run_name"] == "nightly-3"
    assert db.call("rollback")


def test_empty_database(app_client, monkeypatch):
    db = RoutedDb()
    data = fetch(app_client, monkeypatch, db)
    assert data["last_run"] is None
    assert data["services"] == [] and data["recent_runs"] == []
    assert data["kpi"] == {"runs_7d": 0, "runs_30d": 0, "risky_30d": 0, "risky_share_30d": None, "last": None, "previous": None}
    assert [w["week_start"] for w in data["verdict_weeks"]] == WEEKS
    assert all(w["total"] == 0 for w in data["verdict_weeks"])
    assert not [c for c in db.calls if c[0] in ("previous", "previous_rps")]


def test_area_filter_reaches_every_statement(app_client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: ["checkout", "billing"])
    db = full_db()
    fetch(app_client, monkeypatch, db, cookie_area="shop")
    for key in ("last_run", "verdict_counts", "runs_total", "services", "recent", "weeks"):
        _, text, params = db.call(key)
        assert "service = ANY(%s)" in text, key
        assert ["checkout", "billing"] in list(params), key
    weeks_params = db.call("weeks")[2]
    assert weeks_params[1] == ["checkout", "billing"]


def test_group_service_rows_accepts_rows_without_run_id():
    from loadlens_app.blueprints import dashboard

    rows = [("demo", "run-2", "Провал", LAST_CREATED, "12.5", "", "id-2"), ("demo", "run-1", "Успешно", None, None, "soak")]
    grouped = dashboard._group_service_rows(rows)
    assert len(grouped) == 1
    history = grouped[0]["history"]
    assert [h["run_id"] for h in history] == ["id-2", ""]
    assert history[1]["created_at"] is None and history[1]["max_rps"] is None and history[1]["test_type"] == "soak"
    assert grouped[0]["max_rps"] == pytest.approx(12.5)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("226.5", 226.5), (120, 120.0), (" 7 ", 7.0), ("nan", None), ("inf", None), ("~200", None), ("", None), (None, None), (True, None)],
)
def test_dashboard_number(value, expected):
    from loadlens_app.blueprints import dashboard

    assert dashboard._dashboard_number(value) == expected


def test_periods_are_utc_mondays():
    from loadlens_app.blueprints import dashboard

    periods = dashboard._dashboard_periods(datetime(2026, 10, 12, 0, 30, tzinfo=timezone.utc))
    assert [w.isoformat() for w in periods["weeks"]][-1] == "2026-10-12"
    assert all(w.weekday() == 0 for w in periods["weeks"]) and len(periods["weeks"]) == dashboard.DASHBOARD_VERDICT_WEEKS
    assert periods["since"] <= periods["since_30d"] < periods["since_7d"]
