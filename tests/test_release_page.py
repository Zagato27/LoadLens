# SPDX-License-Identifier: Apache-2.0
"""Release pages (plans/release-page-plan.md): the data of AI/release_data.py, the routes of
loadlens_app/blueprints/release_pages.py, the Jira links, the Confluence page of a release and the schema files.

Offline: SQL is checked by its text on a recording connection; the routes run against stubs of the data functions.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.test_smoke_api import FakeConnection, client, patch_core  # noqa: F401  (loadlens_app before update_page)
from AI import release_data as rd
from AI.db_store import BaselineRun
from AI.i18n import with_locale
from loadlens_app import jira_links
from loadlens_app import release_confluence as rc
from tests.i18n_helpers import CYRILLIC

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
THRESHOLDS = {
    "regression_latency_risk_pct": 10.0, "regression_latency_fail_pct": 25.0,
    "regression_errors_risk_pp": 0.5, "regression_errors_fail_pp": None,
    "regression_capacity_risk_pct": 10.0, "regression_capacity_fail_pct": None,
}
JIRA = {"enabled": True, "base_url": "https://jira.example.com", "project_key": "PROJ"}


def flat(text) -> str:
    return " ".join(str(text).split())


def render(query) -> str:
    from psycopg2 import sql

    if isinstance(query, str):
        return query
    if isinstance(query, sql.Composed):
        return "".join(render(part) for part in query.seq)
    if isinstance(query, sql.Identifier):
        return ".".join('"' + name.replace('"', '""') + '"' for name in query.strings)
    if isinstance(query, sql.SQL):
        return query.string
    raise TypeError(f"unexpected query part {query!r}")


class Recorder:
    """Connection whose cursor records statements and answers them with ``answer(text, params)``."""

    def __init__(self, answer=None):
        self.log = []
        self.answer = answer or (lambda text, params: [])
        self.autocommit = False
        self.commits = 0

    def cursor(self):
        conn = self

        class Cursor:
            rowcount = 1

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, query, params=None):
                text = flat(render(query))
                conn.log.append((text, tuple(params or ())))
                self.rows = conn.answer(text, tuple(params or ()))

            def fetchone(self):
                return self.rows[0] if self.rows else None

            def fetchall(self):
                return list(self.rows)

        return Cursor()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.commits += 1
        return False

    def commit(self):
        self.commits += 1

    def close(self):
        pass


def details(target, p95, errors, ceiling):
    """sla_details of a step test with a confirmed target step and a ceiling."""
    return {
        "checks": [
            {"name": "target_rps", "actual": target, "passed": True},
            {"name": "p95_latency", "actual": p95, "passed": True},
            {"name": "error_rate", "actual": errors, "passed": True},
        ],
        "target_window": {"level": target},
        "stable_window": {"level": ceiling},
        "headroom": {"ceiling_level": ceiling},
        "load_step_table": {"steps": [
            {"rps": target, "latency_p95": p95, "error_rate": errors, "label": "S3", "selected": True},
            {"rps": ceiling, "latency_p95": p95 * 2, "error_rate": errors * 2, "label": "S5", "ceiling": True},
        ]},
    }


def mark(run_name, *, release="25.11", service="api", test_type="step", created=T0, verdict="pass", accepted=True,
         baseline=False, manual=None, details_=None, run_id=None):
    item = {
        "release": release, "run_name": run_name, "run_id": run_id or f"id-{run_name}", "service": service,
        "test_type": test_type, "created_at": created, "verdict": verdict, "manual_verdict": manual, "max_rps": 800.0,
        "accepted": accepted, "baseline": baseline,
    }
    if details_ is not None:
        item["sla_details"] = details_
    return item


# ---- status and the release list -----------------------------------------------------------------------------


def test_release_status_is_incomplete_then_the_worst_verdict():
    assert rd.release_status([], []) == "no_data"
    assert rd.release_status(["pass", "fail"], ["web"]) == "incomplete"
    assert rd.release_status(["pass", "risk", "no_data"]) == "risk"
    assert rd.release_status(["pass", "no_data"]) == "no_data"
    assert rd.release_status(["pass", "fail", "risk"]) == "fail"
    assert rd.release_status(["Успешно"]) == "pass"  # legacy labels map to codes


def test_overview_items_count_verdicts_and_order_newest_first():
    accepted = [
        mark("a1", release="25.10", created=T0 - timedelta(days=30)),
        mark("a2", release="25.11", created=T0, verdict="risk"),
        mark("w2", release="25.11", service="web", created=T0 + timedelta(hours=2), verdict="pass"),
        mark("a2s", release="25.11", test_type="soak", created=T0 + timedelta(hours=1), verdict="fail"),
    ]
    rows = [
        {"release": "25.11", "ticket": "PROJ-7", "updated_at": "2026-11-02T00:00:00+00:00"},
        {"release": "26.01", "ticket": "", "updated_at": "2026-11-05T00:00:00+00:00"},
    ]
    items = rd.overview_items(accepted, rows, ["api", "web", "db"])
    assert [item["release"] for item in items] == ["25.11", "25.10", "26.01"]
    first = items[0]
    assert first["status"] == "incomplete" and first["services_tested"] == 2 and first["services_required"] == 3
    assert first["verdicts"] == {"pass": 1, "risk": 1, "fail": 1, "no_data": 0}
    assert first["ticket"] == "PROJ-7" and first["last_created_at"] == (T0 + timedelta(hours=2)).isoformat()
    assert items[1]["status"] == "incomplete" and items[2] == {
        "release": "26.01", "status": "incomplete", "services_tested": 0, "services_required": 3,
        "verdicts": {"pass": 0, "risk": 0, "fail": 0, "no_data": 0}, "last_created_at": None, "ticket": "",
    }
    complete = rd.overview_items([mark("a1", verdict="risk")], [], ["api"])
    assert complete[0]["status"] == "risk"


# ---- rows of a release ---------------------------------------------------------------------------------------


def _previous(name="a10", release="25.10", created=None):
    return BaselineRun(run_name=name, service="api", verdict="pass", created_at=created or T0 - timedelta(days=30),
                       mode="previous_release", release=release, run_id=f"id-{name}")


def test_rows_pair_the_accepted_run_with_the_previous_release_and_colour_by_thresholds():
    marks = [
        mark("a11", created=T0, verdict="risk"),
        mark("a11-retry", created=T0 - timedelta(hours=3), accepted=False, verdict="fail"),
        mark("w11", service="web", created=T0, baseline=True),
    ]
    current = details(600.0, 450.0, 0.8, 800.0)
    current["regression"] = {"available": True, "baseline_run_name": "a09"}
    found = {"a11": current, "a10": details(600.0, 400.0, 0.5, 900.0)}
    calls = []

    def previous_of(run):
        calls.append(run["run_name"])
        return _previous() if run["service"] == "api" else None

    rows = rd.build_rows(marks, previous_of=previous_of, details=found, thresholds_of=lambda service: THRESHOLDS)
    assert calls == ["a11", "w11"]
    api, web = rows
    assert (api["service"], api["test_type"], api["run"]["run_name"], api["run"]["verdict"]) == ("api", "step", "a11", "risk")
    assert [a["run_name"] for a in api["attempts"]] == ["a11", "a11-retry"] and [a["accepted"] for a in api["attempts"]] == [True, False]
    assert api["previous"] == {"release": "25.10", "run_name": "a10", "run_id": "id-a10",
                               "created_at": (T0 - timedelta(days=30)).isoformat(), "verdict": "pass"}
    deltas = api["deltas"]
    assert deltas["available"] and (deltas["kind"], deltas["level"]) == ("target", 600.0)
    items = {item["metric"]: item for item in deltas["items"]}
    assert (items["p95"]["change"], items["p95"]["unit"], items["p95"]["tone"]) == (12.5, "pct", "warn")
    assert (items["errors"]["change"], items["errors"]["unit"], items["errors"]["tone"]) == (0.3, "pp", "ok")
    assert (items["capacity"]["change"], items["capacity"]["delta"], items["capacity"]["tone"]) == (-11.1, 11.1, "warn")
    assert api["stored_regression"] == {"available": True, "baseline_run_name": "a09"}
    assert api["report_url"] == "/reports/id-a11" and api["compare_url"] == "/compare?run_a=a11&run_b=a10"
    assert api["baseline"] is False
    assert web["previous"] is None and web["deltas"]["available"] is False and web["deltas"]["reason"] == "no_previous"
    assert web["baseline"] is True and web["compare_url"] is None and web["stored_regression"] is None


def test_rows_without_thresholds_are_neutral_and_without_an_accepted_run_have_no_deltas():
    found = {"a11": details(600.0, 450.0, 0.8, 800.0), "a10": details(600.0, 400.0, 0.5, 900.0)}
    rows = rd.build_rows([mark("a11")], previous_of=lambda run: _previous(), details=found,
                         thresholds_of=lambda service: rd_thresholds_off())
    assert {item["tone"] for item in rows[0]["deltas"]["items"]} == {"neutral"}
    assert {item["passed"] for item in rows[0]["deltas"]["items"]} == {None}
    none = rd.build_rows([mark("a11", accepted=False)], previous_of=lambda run: pytest.fail("not asked"), details={},
                         thresholds_of=lambda service: None)
    assert none[0]["run"] is None and none[0]["deltas"]["reason"] == "no_accepted" and none[0]["report_url"] is None


def rd_thresholds_off():
    from AI.regression import thresholds_from_sla

    return thresholds_from_sla({})


def test_release_view_wires_the_lookups(monkeypatch):
    seen = {}
    monkeypatch.setattr(rd, "load_release", lambda conn, schema, name: {"release": name, "ticket": "PROJ-1", "note": "n",
                                                                          "author": "eng", "updated_at": "x"})
    monkeypatch.setattr(rd, "marked_runs", lambda conn, schema, table, **kw: seen.update(marks=kw) or [mark("a11")])

    def previous(conn, schema, table, **kw):
        seen["previous"] = kw
        return _previous()

    monkeypatch.setattr(rd, "previous_release_run", previous)
    monkeypatch.setattr(rd, "final_details", lambda conn, schema, table, names: seen.update(names=sorted(names)) or {})
    view = rd.release_view(object(), "public", "llm_reports", "25.11", services=["api", "web"], required=["api", "web"],
                           thresholds_of=lambda service: None, project_area="demo")
    assert seen["marks"] == {"release": "25.11", "services": ["api", "web"]}
    assert seen["previous"] == {"service": "api", "test_type": "step", "before": T0, "exclude_run_name": "a11",
                                "exclude_release": "25.11", "project_area": "demo"}
    assert seen["names"] == ["a10", "a11"]
    assert (view["ticket"], view["status"], view["untested"]) == ("PROJ-1", "incomplete", ["web"])
    assert view["rows"][0]["deltas"]["reason"] == "no_checks"

    monkeypatch.setattr(rd, "load_release", lambda conn, schema, name: None)
    monkeypatch.setattr(rd, "marked_runs", lambda conn, schema, table, **kw: [])
    assert rd.release_view(object(), "public", "llm_reports", "nope", services=None, required=[], thresholds_of=lambda s: None) is None


def test_marked_runs_sql_filters_and_maps_rows(monkeypatch):
    monkeypatch.setattr(rd, "release_marks_ready", lambda conn, schema: True)
    monkeypatch.setattr(rd, "manual_verdicts_ready", lambda conn, schema: True)
    row = ("25.11", "a11", "", "api", "step", T0, "risk", "fail", 812.5, True, True, {"checks": []})
    conn = Recorder(lambda text, params: [row])
    runs = rd.marked_runs(conn, "public", "llm_reports", release="25.11", services=["api"], service="api",
                          test_type="Ступенчатый", accepted_only=True, limit=12, with_details=True)
    text, params = conn.log[-1]
    assert 'FROM "public"."release_marks" m WHERE m.release <> \'\' AND m.release = %s AND m.accepted' in text
    assert "AND m.service = ANY(%s) AND m.service = %s AND m.test_type = %s" in text
    assert 'FROM "public"."llm_reports" r JOIN marks m ON m.run_name = r.run_name' in text and "manual_verdicts" in text
    assert text.endswith("ORDER BY f.created_at DESC, m.run_name LIMIT %s") and ", f.sla_details FROM marks m" in text
    assert params == ("25.11", ["api"], "api", "step", 12)
    assert runs == [{
        "release": "25.11", "run_name": "a11", "run_id": "", "service": "api", "test_type": "step", "created_at": T0,
        "verdict": "risk", "manual_verdict": "fail", "max_rps": 812.5, "accepted": True, "baseline": True,
        "sla_details": {"checks": []},
    }]
    monkeypatch.setattr(rd, "release_marks_ready", lambda conn, schema: False)
    assert rd.marked_runs(Recorder(), "public", "llm_reports") == []


def test_trend_points_in_report_order_at_the_own_level(monkeypatch):
    seen = {}
    newest_first = [
        mark("a12", release="25.12", created=T0 + timedelta(days=30), verdict="fail", details_=details(600.0, 500.0, 1.0, 700.0)),
        mark("a11", release="25.11", created=T0, baseline=True, details_={"checks": [
            {"name": "p95_latency", "actual": 420.0, "passed": True}, {"name": "error_rate", "actual": 0.4, "passed": True},
        ], "stable_window": {"level": 900.0}, "headroom": {"ceiling_level": 900.0}}),
    ]
    monkeypatch.setattr(rd, "marked_runs", lambda conn, schema, table, **kw: seen.update(kw) or newest_first)
    points = rd.trend_points(object(), "public", "llm_reports", "api", "Ступенчатый", 5)
    assert seen == {"service": "api", "test_type": "step", "accepted_only": True, "limit": 5, "with_details": True}
    assert [p["release"] for p in points] == ["25.11", "25.12"]
    old, new = points
    assert (old["kind"], old["level"], old["p95"], old["errors"], old["capacity"], old["baseline"]) == ("ceiling", 900.0, 420.0, 0.4, 900.0, True)
    assert (new["kind"], new["level"], new["p95"], new["errors"], new["capacity"], new["verdict"]) == ("target", 600.0, 500.0, 1.0, 700.0, "fail")


def test_a_trend_point_without_a_ceiling_takes_the_measured_level():
    failed = details(600.0, 520.0, 0.9, 250.0)
    failed.pop("target_window")
    failed.pop("headroom")
    failed["checks"][1]["passed"] = False
    failed["load_step_table"]["steps"] = [{"rps": 250.0, "latency_p95": 520.0, "error_rate": 0.9, "selected": True}]
    point = rd.trend_point(mark("a12", details_=failed), "step")
    assert (point["kind"], point["level"], point["p95"], point["errors"], point["capacity"]) == ("ceiling", 250.0, 520.0, 0.9, None)
    scenario = rd.trend_point(mark("s1", test_type="scenario", details_=failed), "scenario")
    assert (scenario["kind"], scenario["level"], scenario["p95"]) == (None, None, None)


def test_release_rows_sql_and_publication(monkeypatch):
    conn = Recorder(lambda text, params: [("25.11", "PROJ-3", "note", "eng", T0)])
    assert rd.save_release(conn, "public", "25.11", ticket="PROJ-3", note=None, author="eng") == {
        "release": "25.11", "ticket": "PROJ-3", "note": "note", "author": "eng", "updated_at": T0.isoformat(),
    }
    text, params = conn.log[-1]
    assert text.startswith('INSERT INTO "public"."releases" AS rel') and "COALESCE(%s, rel.note)" in text
    assert params == ("25.11", "PROJ-3", None, "eng", "PROJ-3", None)
    conn = Recorder()
    assert rd.delete_release(conn, "public", "25.11") is True
    assert [entry[0] for entry in conn.log] == [
        'DELETE FROM "public"."releases" WHERE name = %s', 'DELETE FROM "public"."release_publications" WHERE release = %s',
    ]
    conn = Recorder()
    rd._ENSURED.discard("s1")
    rd.ensure_release_tables(conn, "s1")
    assert [text.split(" (")[0] for text, _ in conn.log if text.startswith("CREATE TABLE")] == [
        'CREATE TABLE IF NOT EXISTS "s1"."releases"', 'CREATE TABLE IF NOT EXISTS "s1"."release_publications"',
    ]
    count = len(conn.log)
    rd.ensure_release_tables(conn, "s1")
    assert len(conn.log) == count and conn.autocommit is False and conn.commits == 1


def test_legacy_publication_primary_key_is_upgraded_without_losing_rows(monkeypatch):
    monkeypatch.setattr(rd, "_ENSURED", set())
    primary = ["release"]

    def answer(text, params):
        if "FROM pg_constraint" in text:
            assert params == ("tenant", "release_publications")
            return [("legacy_publications_pk", list(primary))]
        if "DROP CONSTRAINT" in text:
            primary[:] = ["release", "project_area"]
        return []

    conn = Recorder(answer)
    rd.ensure_release_tables(conn, "tenant")
    statements = [text for text, _ in conn.log]
    assert 'ALTER TABLE "tenant"."release_publications" ADD COLUMN IF NOT EXISTS project_area TEXT NOT NULL DEFAULT \'\'' in statements
    assert 'ALTER TABLE "tenant"."release_publications" DROP CONSTRAINT "legacy_publications_pk", ADD PRIMARY KEY (release, project_area)' in statements
    assert not any(text.startswith(("DELETE", "UPDATE", "DROP TABLE")) for text in statements)
    assert conn.commits == 1 and conn.autocommit is False
    # Another worker sees the new key and does not drop/rebuild it.
    rd._ENSURED.clear()
    conn.log.clear()
    rd.ensure_release_tables(conn, "tenant")
    assert not any("DROP CONSTRAINT" in text for text, _ in conn.log)


def test_publication_queries_keep_the_project_area_in_the_key():
    conn = Recorder(lambda text, params: [("9", "https://wiki/9", "LT", "1", T0)])
    assert rd.load_publication(conn, "public", "25.11", "shop")["page_id"] == "9"
    assert conn.log[-1][1] == ("25.11", "shop")
    assert "WHERE release = %s AND project_area = %s" in conn.log[-1][0]
    rd.save_publication(conn, "public", "25.11", "9", "https://wiki/9", "LT", "1", "shop")
    assert "ON CONFLICT (release, project_area) DO UPDATE" in conn.log[-1][0]
    assert conn.log[-1][1] == ("25.11", "shop", "9", "https://wiki/9", "LT", "1")
    rd.load_publication(conn, "public", "25.11")
    assert conn.log[-1][1] == ("25.11", "")  # legacy callers retain the global scope


def test_release_tables_are_in_the_schema_files_and_the_app_role():
    schema = (ROOT / "docker" / "initdb" / "02_schema.sql").read_text(encoding="utf-8")
    for table in ("releases", "release_publications"):
        assert f"CREATE TABLE IF NOT EXISTS public.{table} (" in schema
    assert "PRIMARY KEY (release, project_area)" in schema
    role = (ROOT / "docker" / "initdb" / "03_app_role.sh").read_text(encoding="utf-8")
    tables = role.split("LOADLENS_TABLES=", 1)[1].splitlines()[0]
    assert " releases " in tables and " release_publications " in tables


# ---- Jira ----------------------------------------------------------------------------------------------------


def test_jira_links_encode_the_jql_and_need_an_enabled_section():
    url = jira_links.issues_url(JIRA, 'DEMO 25.11 "hot"')
    assert url == ("https://jira.example.com/issues/?jql=project%20%3D%20PROJ%20AND%20fixVersion%20%3D%20"
                   "%22DEMO%2025.11%20%5C%22hot%5C%22%22")
    assert jira_links.ticket_url(JIRA, "PROJ-12") == "https://jira.example.com/browse/PROJ-12"
    assert jira_links.ticket_url(JIRA, "demo-12") is None and jira_links.ticket_url(JIRA, "") is None
    assert jira_links.links(JIRA, "25.11", "PROJ-1") == {
        "issues_url": "https://jira.example.com/issues/?jql=project%20%3D%20PROJ%20AND%20fixVersion%20%3D%20%2225.11%22",
        "ticket_url": "https://jira.example.com/browse/PROJ-1",
    }
    assert jira_links.links({**JIRA, "enabled": False}, "25.11", "PROJ-1") is None
    assert jira_links.links({**JIRA, "base_url": "https://jira.example.com/"}, "25.11") is None
    assert jira_links.links({}, "25.11") is None


@pytest.mark.parametrize("payload, key", [
    ({"enabled": "yes"}, "server.jira.enabled_bool"),
    ({"base_url": "jira.example.com"}, "server.jira.base_url_invalid"),
    ({"base_url": "https://jira.example.com/"}, "server.jira.base_url_invalid"),
    ({"base_url": "https://jira example.com"}, "server.jira.base_url_invalid"),
    ({"project_key": "demo"}, "server.jira.project_key_invalid"),
    ({"project_key": "1AB"}, "server.jira.project_key_invalid"),
    ({"enabled": True, "base_url": "https://jira.example.com", "project_key": ""}, "server.jira.incomplete"),
])
def test_validate_jira_rejects(payload, key):
    with pytest.raises(jira_links.JiraConfigError) as caught:
        jira_links.validate_jira(payload)
    assert caught.value.key == key


def test_validate_jira_accepts_and_tickets_are_checked():
    jira_links.validate_jira({"enabled": True, "base_url": "https://jira.example.com/jira", "project_key": "DEMO_2"})
    jira_links.validate_jira({"enabled": False, "base_url": "", "project_key": ""})
    assert jira_links.normalize_ticket(" PROJ-42 ") == "PROJ-42" and jira_links.normalize_ticket("") == ""
    assert jira_links.normalize_ticket(None) == ""
    for bad in ("DEMO", "demo-1", "PROJ-1a", 12):
        with pytest.raises(jira_links.TicketError):
            jira_links.normalize_ticket(bad)


def test_config_api_saves_the_jira_section(client, monkeypatch, tmp_path):
    import update_page
    from loadlens_app import core

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setitem(core.CONFIG, "jira", {"enabled": False, "base_url": "https://old.example.com", "project_key": "OLD"})
    bad = client.post("/config", json={"section": "jira", "data": {"enabled": True, "base_url": "https://jira.example.com/", "project_key": "PROJ"}})
    assert bad.status_code == 400 and "«/»" in bad.get_json()["error"]
    unknown = client.post("/config", json={"section": "jira", "data": {**JIRA, "projekt": "x"}})
    assert unknown.status_code == 200 and any("projekt" in item for item in unknown.get_json()["warnings"])
    # A new address of a section without secrets is not a moved destination.
    ok = client.post("/config", json={"section": "jira", "data": JIRA})
    assert ok.status_code == 200, ok.get_json()
    assert client.get("/config").get_json()["jira"] == JIRA
    assert "release_parent_page_id" in (ROOT / "settings.example.py").read_text(encoding="utf-8")


# ---- routes --------------------------------------------------------------------------------------------------


@pytest.fixture
def pages(monkeypatch):
    from loadlens_app.blueprints import release_pages as rp

    state = {"saved": [], "deleted": [], "views": {}, "trend": [], "publication": None, "exists": True}
    monkeypatch.setattr(rp, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(rp, "ensure_release_tables", lambda conn, schema: None)
    monkeypatch.setattr(rp, "area_services", lambda area: ["api", "web"])
    monkeypatch.setattr(rp, "_resolve_services_filter", lambda area: ["api", "web"])
    monkeypatch.setattr(rp, "_active_project_area", lambda: "demo")
    monkeypatch.setattr(rp, "release_overview", lambda conn, schema, table, **kw: state.update(overview=kw) or [{"release": "25.11"}])

    def view(conn, schema, table, name, **kw):
        state["view_args"] = kw
        return state["views"].get(name)

    monkeypatch.setattr(rp, "release_view", view)
    monkeypatch.setattr(rp, "load_publication", lambda conn, schema, name, area="": state.update(publication_area=area) or state["publication"])
    monkeypatch.setattr(rp, "load_release", lambda conn, schema, name: None)
    monkeypatch.setattr(rp, "marked_runs", lambda conn, schema, table, **kw: [{"run_name": "a"}] if state["exists"] else [])

    def save(conn, schema, name, *, ticket=None, note=None, author=""):
        state["saved"].append((name, ticket, note, author))
        return {"release": name, "ticket": ticket or "", "note": note or "", "author": author, "updated_at": None}

    monkeypatch.setattr(rp, "save_release", save)
    monkeypatch.setattr(rp, "delete_release_row", lambda conn, schema, name: state["deleted"].append(name) or name == "25.11")
    monkeypatch.setattr(rp, "trend_points", lambda conn, schema, table, service, test_type, limit: state["trend"].append((service, test_type, limit)) or [])
    monkeypatch.setattr(jira_links, "jira_config", lambda config=None: dict(JIRA))
    return state


def test_release_data_routes(client, pages):
    pages["views"]["25.11"] = {"release": "25.11", "ticket": "PROJ-5", "note": "", "author": "", "updated_at": None,
                               "status": "pass", "rows": [], "untested": []}
    pages["publication"] = {"page_id": "9", "page_url": "https://wiki/9", "updated_at": "t", "space_key": "LT", "parent_page_id": "1"}
    body = client.get("/releases_data/25.11").get_json()
    assert body["jira"] == {"issues_url": jira_links.issues_url(JIRA, "25.11"), "ticket_url": "https://jira.example.com/browse/PROJ-5"}
    assert body["publication"] == {"page_id": "9", "page_url": "https://wiki/9", "updated_at": "t"} and body["services_required"] == 2
    assert pages["publication_area"] == "demo"
    assert pages["view_args"]["services"] == ["api", "web"] and pages["view_args"]["project_area"] == "demo"
    missing = client.get("/releases_data/26.01")
    assert missing.status_code == 404 and missing.get_json()["code"] == "not_found"
    # A name with a slash is a release name too.
    pages["views"]["DEMO 25/11"] = dict(pages["views"]["25.11"], release="DEMO 25/11")
    assert client.get("/releases_data/DEMO%2025/11").get_json()["release"] == "DEMO 25/11"
    assert client.get("/releases_overview").get_json() == [{"release": "25.11"}]
    assert pages["overview"] == {"services": ["api", "web"], "required": ["api", "web"]}


def test_update_and_delete_release(client, pages):
    assert client.put("/releases_data/25.11", json=[]).status_code == 400
    bad = client.put("/releases_data/25.11", json={"ticket": "demo-1"})
    assert bad.status_code == 400 and "PROJ-123" in bad.get_json()["error"]
    long_note = client.put("/releases_data/25.11", json={"note": "x" * 1001})
    assert long_note.status_code == 400 and "1000" in long_note.get_json()["error"]
    assert client.put("/releases_data/25.11", json={"note": 5}).status_code == 400
    ok = client.put("/releases_data/25.11", json={"ticket": " PROJ-9 ", "note": " smoke "})
    assert ok.status_code == 200 and ok.get_json()["jira"]["ticket_url"] == "https://jira.example.com/browse/PROJ-9"
    assert pages["saved"] == [("25.11", "PROJ-9", "smoke", "local")]
    client.put("/releases_data/25.11", json={"note": "only the note"})
    assert pages["saved"][-1] == ("25.11", None, "only the note", "local")
    assert client.delete("/releases_data/25.11").get_json() == {"status": "ok"}
    assert client.delete("/releases_data/26.01").status_code == 404


def test_release_trend_route(client, pages):
    assert client.get("/release_trend").status_code == 400
    assert client.get("/release_trend?service=api&limit=0").status_code == 400
    assert client.get("/release_trend?service=api&limit=51").status_code == 400
    assert client.get("/release_trend?service=api&limit=x").status_code == 400
    body = client.get("/release_trend?service=api&test_type=Стресс").get_json()
    assert body == {"service": "api", "test_type": "stress", "points": []} and pages["trend"] == [("api", "stress", 12)]
    assert client.get("/release_trend?service=other").get_json()["points"] == [] and len(pages["trend"]) == 1


def test_release_confluence_job(client, pages, monkeypatch):
    from loadlens_app.blueprints import release_pages as rp

    calls = []

    def fake_publish(release, area, source_url, progress):
        calls.append((release, area, source_url))
        return {"page_id": "8", "page_url": "https://wiki.local/pages/8", "title": "LoadLens: релиз 25.11"}

    monkeypatch.setattr(rp, "publish_release", fake_publish)
    assert client.get("/release_publications/25.11").get_json() == {"release": "25.11", "page_id": None, "page_url": None, "updated_at": None}
    assert pages["publication_area"] == "demo"
    assert client.post("/releases/25.11/confluence", data="{}").status_code == 415
    resp = client.post("/releases/25.11/confluence", json={})
    assert resp.status_code == 200 and resp.get_json()["status"] == "accepted"
    job = client.get(f"/job_status/{resp.get_json()['job_id']}").get_json()
    assert (job["status"], job["kind"], job["page_url"], job["report_url"]) == (
        "done", "release_confluence", "https://wiki.local/pages/8", "/releases/25.11",
    )
    assert calls == [("25.11", "demo", "http://localhost/releases/25.11")]
    pages["exists"] = False
    assert client.post("/releases/26.01/confluence", json={}).status_code == 404


def test_release_routes_follow_the_roles():
    from loadlens_app.auth import Role, policy

    rules = {name: rule for name, rule in policy.POLICY.items() if name.startswith("release_pages.")}
    assert {name: (rule.role, rule.page, rule.audit, rule.session_only) for name, rule in rules.items()} == {
        "release_pages.releases_page": (Role.VIEWER, True, "", False),
        "release_pages.release_page": (Role.VIEWER, True, "", False),
        "release_pages.releases_overview": (Role.VIEWER, False, "", False),
        "release_pages.get_release": (Role.VIEWER, False, "", False),
        "release_pages.update_release": (Role.ENGINEER, False, "release.update", False),
        "release_pages.delete_release": (Role.ADMIN, False, "release.delete", True),
        "release_pages.release_trend": (Role.VIEWER, False, "", False),
        "release_pages.get_release_confluence": (Role.VIEWER, False, "", False),
        "release_pages.post_release_confluence": (Role.ENGINEER, False, "release.publish", False),
    }


# ---- Confluence page -----------------------------------------------------------------------------------------


def _view():
    found = {"a11": details(600.0, 450.0, 0.8, 800.0), "a10": details(600.0, 400.0, 0.5, 900.0)}
    rows = rd.build_rows(
        [mark("a11", verdict="risk", baseline=True), mark("w11", service="web", test_type="soak")],
        previous_of=lambda run: _previous() if run["service"] == "api" else None,
        details=found, thresholds_of=lambda service: THRESHOLDS,
    )
    return {
        "release": "25.11", "ticket": "PROJ-5", "note": "Hotfix release", "status": "incomplete", "rows": rows,
        "untested": ["db"], "services_required": 3, "publication": None,
        "jira": jira_links.links(JIRA, "25.11", "PROJ-5"),
    }


def test_release_page_body_links_published_pages_and_falls_back_to_loadlens():
    body = rc.release_page_body(
        _view(), source_url="https://ll.local/releases/25.11", base_url="https://ll.local", generated_text="01.11.2026 12:00",
        report_pages={"a11": "https://wiki/report-a11"}, compare_pages={},
        trend_files=[("api · Ступенчатый", "trend-api-step.png")],
    )
    assert "<h2>LoadLens: релиз 25.11</h2>" in body
    assert '<a href="https://wiki/report-a11">a11</a> <em>(эталон)</em>' in body
    assert '<a href="https://ll.local/reports/id-w11">w11</a>' in body
    assert '<a href="https://ll.local/compare?run_a=a11&amp;run_b=a10">25.10 · a10</a>' in body
    assert '<ac:parameter ac:name="colour">Yellow</ac:parameter><ac:parameter ac:name="title">Есть риски</ac:parameter>' in body
    assert "Неполный" in body and "прогнано 2 из 3" in body and "Hotfix release" in body
    assert "+12,5 %" in body and "+0,3 п.п." in body and "−11,1 %" in body and "на 600 RPS" in body
    assert "нет прошлого релиза с этим сервисом" in body
    assert '<a href="https://jira.example.com/browse/PROJ-5">PROJ-5</a>' in body and "Задачи релиза" in body
    assert '<ri:attachment ri:filename="trend-api-step.png"/>' in body and "<li>db</li>" in body
    assert '<ac:structured-macro ac:name="expand">' in body
    linked = rc.release_page_body(
        _view(), source_url="", base_url="https://ll.local", generated_text="x", report_pages={},
        compare_pages={("a11", "a10"): "https://wiki/compare"}, trend_files=[],
    )
    assert '<a href="https://wiki/compare">25.10 · a10</a>' in linked and "Тренд по релизам" not in linked


def test_release_page_body_in_english_has_no_russian():
    with with_locale("en"):
        body = rc.release_page_body(_view(), source_url="https://ll/releases/25.11", base_url="https://ll", generated_text="x",
                                    report_pages={}, compare_pages={}, trend_files=[("api · Step", "trend-api-step.png")])
        title = rc.release_page_title("25.11")
    assert not CYRILLIC.search(body + title), body
    assert "+0.3 pp" in body and "LoadLens: release 25.11" == title


def test_publish_release_uploads_the_trends_and_saves_the_page(monkeypatch):
    from loadlens_app.blueprints import release_pages as rp

    captured = {"progress": []}
    monkeypatch.setattr(rc, "confluence_config", lambda: {
        "enabled": True, "base_url": "https://wiki.local", "username": "u", "password": "p", "space_key": "LT",
        "parent_page_id": "1", "release_parent_page_id": " 7 ",
    })
    monkeypatch.setattr(rp, "release_payload", lambda name, area: _view() if name == "25.11" else None)
    points = [{"release": "25.10", "run_name": "a10", "verdict": "pass", "p95": 400.0, "errors": 0.5, "capacity": 900.0, "baseline": False},
              {"release": "25.11", "run_name": "a11", "verdict": "risk", "p95": 450.0, "errors": None, "capacity": 800.0, "baseline": True}]
    monkeypatch.setattr(rp, "trend_payload", lambda service, test_type, limit, area: points if service == "api" else [])
    monkeypatch.setattr(rc, "load_publications", lambda names: {"a11": {"page_url": "https://wiki/r"}})
    monkeypatch.setattr(rc, "load_compare_publication", lambda a, b: None)
    monkeypatch.setattr(rc, "load_release_publication", lambda name, area="": None)
    monkeypatch.setattr(rc, "save_release_publication", lambda *args: captured.update(saved=args))

    def upsert(client, known, title, body, attachments, progress):
        captured.update(title=title, body=body, attachments=attachments, parent=client.parent_page_id, known=known)
        return "9", "https://wiki.local/pages/9"

    monkeypatch.setattr(rc, "upsert_page", upsert)
    result = rc.publish_release("25.11", "demo", "https://ll/releases/25.11", lambda message, pct=None: captured["progress"].append(str(message)))
    assert result == {"page_id": "9", "page_url": "https://wiki.local/pages/9", "title": "LoadLens: релиз 25.11 [demo]"}
    assert captured["parent"] == "7" and list(captured["attachments"]) == ["trend-api-step.png"]
    assert captured["attachments"]["trend-api-step.png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert captured["saved"] == ("25.11", "9", "https://wiki.local/pages/9", "LT", "7", "demo")
    assert '<a href="https://wiki/r">a11</a>' in captured["body"]
    assert captured["progress"][0] == "Собираем релиз…" and captured["progress"][-1] == "Готово"
    with pytest.raises(rc.ReleasePublishError):
        rc.publish_release("26.01", "demo", "", None)


def test_publishing_the_same_release_in_two_areas_preserves_both_pages(monkeypatch):
    from loadlens_app.blueprints import release_pages as rp

    pages = {("25.11", ""): rc.ReleasePublication("25.11", "legacy", "https://wiki/legacy", None)}
    writes = []
    monkeypatch.setattr(rc, "confluence_config", lambda: {
        "enabled": True, "base_url": "https://wiki.local", "username": "u", "password": "p", "space_key": "LT",
        "parent_page_id": "1",
    })
    monkeypatch.setattr(rp, "release_payload", lambda name, area: {**_view(), "note": area})
    monkeypatch.setattr(rc, "_trend_pictures", lambda rows, area: ([], {}))
    monkeypatch.setattr(rc, "load_publications", lambda names: {})
    monkeypatch.setattr(rc, "_compare_pages", lambda rows: {})
    monkeypatch.setattr(rc, "load_release_publication", lambda name, area="": pages.get((name, area)))

    def save(name, page_id, url, space, parent, area=""):
        pages[(name, area)] = rc.ReleasePublication(name, page_id, url, None)

    def upsert(client, known, title, body, attachments, progress):
        page_id = known or str(len(pages) + 1)
        writes.append((page_id, title, body))
        return page_id, f"https://wiki/{page_id}"

    monkeypatch.setattr(rc, "save_release_publication", save)
    monkeypatch.setattr(rc, "upsert_page", upsert)
    first = rc.publish_release("25.11", "shop")
    second = rc.publish_release("25.11", "billing")
    again = rc.publish_release("25.11", "shop")
    assert first["page_id"] != second["page_id"]
    assert again["page_id"] == first["page_id"]
    assert first["title"] != second["title"]
    assert pages[("25.11", "")].page_id == "legacy"
    assert "[shop]" in writes[0][2] and "[billing]" in writes[1][2]


def test_release_parent_and_delta_texts():
    assert rc._release_parent({"parent_page_id": "1", "release_parent_page_id": ""}) == "1"
    assert rc._release_parent({"parent_page_id": "1", "release_parent_page_id": " 5 "}) == "5"
    assert rc.delta_text({"change": -8.0, "unit": "pct"}) == "−8 %"
    assert rc.delta_text({"change": 0.0, "unit": "pp"}) == "0 п.п."
    assert rc.delta_text({"change": None}) == "—"
    assert rc.level_text({"kind": "ceiling", "level": 540.4}) == "на потолке, 540,4 RPS"
    assert rc.level_text({"kind": None, "level": 5}) == ""
    assert rc._base_url("https://ll/app/releases/25.11") == "https://ll/app"
    assert re.fullmatch(r"trend-api-step\.png", rc.trend_filename("api", "step"))
