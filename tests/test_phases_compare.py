# SPDX-License-Identifier: Apache-2.0
"""Scenario runs compared phase against phase: the /compare_phases API, the facts of the AI explanation, the page."""

from __future__ import annotations

import json

from tests.test_smoke_api import RecordingConnection, client, patch_core  # noqa: F401
from AI.db_store import StoredRunReport, StoredRunRow
from AI.phases import phase_pairs
from loadlens_app import compare_explanation as cx
from loadlens_app.blueprints import compare as compare_bp


def _phase(index, key, minutes, start, *, verdict="pass", rps=50.0, groups=None):
    end = start + minutes
    return {
        "index": index, "number": index, "key": key, "title": key, "label": f"фаза {index} · {key}",
        "start_iso": f"2026-11-10T12:{start:02d}:00+03:00", "end_iso": f"2026-11-10T12:{end:02d}:00+03:00",
        "start_utc": f"2026-11-10 09:{start:02d}:00+00:00", "end_utc": f"2026-11-10 09:{end:02d}:00+00:00",
        "duration_min": float(minutes), "rps_mean": rps, "verdict": verdict, "failed": [] if verdict == "pass" else ["p95_latency"],
        "groups": [{"key": g} for g in (groups or [key])], "idle": False,
    }


SCENARIO_B = {"detected": True, "rps_mean": 40.0, "phases": [
    _phase(1, "Catalog", 10, 0), _phase(2, "Cart", 10, 10), _phase(3, "Results", 5, 20),
]}
SCENARIO_A = {"detected": True, "rps_mean": 38.0, "phases": [
    _phase(1, "Catalog", 10, 0), _phase(2, "Cart", 15, 10, verdict="fail", rps=40.0),
]}


def test_phase_pairs_of_two_runs():
    out = phase_pairs(SCENARIO_A, SCENARIO_B)
    assert [(p["a"] and p["a"]["key"], p["b"] and p["b"]["key"]) for p in out["pairs"]] == [
        ("Catalog", "Catalog"), ("Cart", "Cart"), (None, "Results"),
    ]
    assert out["pairs"][1]["duration_change_pct"] == 50.0 and out["pairs"][1]["rps_change_pct"] == -20.0
    assert out["only_in_a"] == 0 and out["only_in_b"] == 1


def _report(name, scenario, verdict="risk"):
    parsed = {"report_language": "ru", "scenario": scenario}
    row = StoredRunRow(
        domain="final", run_id=f"id-{name}", service="demo", test_type="scenario", project_area="", start_ms=0,
        end_ms=3_600_000, verdict=verdict, sla_verdict=verdict, parsed=parsed, sla_details={}, system_context={}, created_at=None,
    )
    return StoredRunReport(run_name=name, rows={"final": row})


def test_the_explanation_gets_the_phase_table_and_a_flag(monkeypatch):
    reports = {"A": _report("rel-121", SCENARIO_A), "B": _report("rel-120", SCENARIO_B)}
    monkeypatch.setattr(cx, "_load_reports", lambda run_a, run_b: (reports, []))
    monkeypatch.setattr(cx, "llm_choice", lambda _area: (None, "no model"))
    context = cx.build_context("rel-121", "rel-120")
    facts = context.payload["facts"]
    assert context.sides["A"].key_rps == 38.0 and context.sides["A"].key_rps_kind == "phases"
    assert facts["runs"]["A"]["key_rps_kind"] == "средний RPS сценария"
    phases = facts["phases"]
    assert phases["only_in_B"] == 1 and phases["pairs"][1]["A_min"] == 15.0 and phases["pairs"][1]["A_verdict"] == "Провал"
    flags = {note.code: note for note in context.flags}
    assert "phases" in flags and cx.flag_text(flags["phases"]) == "Фазы сценария не совпали: только в A — 0, только в B — 1"
    # A step run against a scenario run has no phase table.
    reports["B"] = None
    assert "phases" not in cx.build_context("rel-121", "rel-120").payload["facts"]


class StrictConnection(RecordingConnection):
    """psycopg2 commits when a ``with conn`` block ends: a connection closed inside the block fails there."""

    closed = False

    def __exit__(self, exc_type, exc, tb):
        assert not self.closed, "the connection was closed inside its with block"
        return False

    def close(self):
        self.closed = True


def test_compare_phases_api(client, monkeypatch):  # noqa: F811
    metrics = [
        ("rel-121", "1", "LT p95", 410.0), ("rel-120", "1", "LT p95", 300.0),
        ("rel-121", "0", "LT errors", 0.5), ("rel-120", "0", "LT errors", 0.5),
    ]
    conn = StrictConnection(fetchall_results=[
        [("rel-121", json.dumps(SCENARIO_A)), ("rel-120", SCENARIO_B)],
        metrics,
    ])
    monkeypatch.setattr(compare_bp, "_ts_conn", lambda: conn)
    monkeypatch.setattr(compare_bp, "_roles", lambda domain: {"LT p95": "lt.p95_total", "LT errors": "lt.error_rate_pct", "LT RPS": "lt.rps_total"})
    monkeypatch.setattr(compare_bp, "_services_clause", lambda alias="": ("", []))
    data = client.get("/compare_phases?run_a=rel-121&run_b=rel-120").get_json()
    assert data["available"] is True and data["only_in_b"] == 1
    assert data["metrics"] == [{"key": "p95", "query_label": "LT p95"}, {"key": "errors", "query_label": "LT errors"}]
    cart = data["pairs"][1]
    assert cart["metrics"]["p95"]["value_a"] == 410.0 and round(cart["metrics"]["p95"]["trend_pct"], 1) == 36.7
    assert cart["metrics"]["p95"]["direction"] == "lower" and cart["a"]["verdict"] == "fail"
    assert data["pairs"][0]["metrics"]["errors"]["trend_pct"] == 0.0 and data["pairs"][2]["a"] is None
    sql_text, params = conn.log[1]
    assert "WITH w(run_name, pair, t0, t1) AS (VALUES" in sql_text and params[-1] == ["LT p95", "LT errors"]
    # Five phase windows: two pairs with both runs, one phase only in B.
    assert params[:4] == ("rel-121", "0", "2026-11-10 09:00:00+00:00", "2026-11-10 09:10:00+00:00") and len(params) == 5 * 4 + 1

    assert conn.closed

    conn = StrictConnection(fetchall_results=[[("rel-121", None), ("rel-120", SCENARIO_B)]])
    monkeypatch.setattr(compare_bp, "_ts_conn", lambda: conn)
    assert client.get("/compare_phases?run_a=rel-121&run_b=rel-120").get_json() == {"available": False, "run_a": "rel-121", "run_b": "rel-120", "pairs": []}
    assert conn.closed and len(conn.log) == 1
    assert client.get("/compare_phases?run_a=x").status_code == 400


def test_the_compare_page_draws_the_phase_table():
    from tests.js_harness import run_js

    pairs = phase_pairs(SCENARIO_A, SCENARIO_B)
    pairs["pairs"][1]["metrics"] = {"p95": {"value_a": 410.0, "value_b": 300.0, "trend_pct": 36.7, "direction": "lower"}}
    data = {"available": True, "metrics": [{"key": "p95", "query_label": "LT p95"}], **pairs}
    html = run_js(["static/js/common.js", "static/js/ui.js", "static/js/compare.js"], f"LL.compare._internals.phasesTableHtml({json.dumps(data, ensure_ascii=False)})")
    assert html.count("<tr") == 4 and html.count('class="is-unpaired"') == 1
    assert "фаза 2 · Cart" in html and "15.00 / 10.00" in html and "p95, мс" in html
    assert "↑ 36.7 % · хуже" in html and "↑ 50.0 % · хуже" in html


def test_the_confluence_comparison_has_the_phase_table(monkeypatch):
    from loadlens_app import compare_confluence as cc

    reports = {"A": _report("rel-121", SCENARIO_A), "B": _report("rel-120", SCENARIO_B)}
    monkeypatch.setattr(cx, "_load_reports", lambda run_a, run_b: (reports, []))
    monkeypatch.setattr(cx, "llm_choice", lambda _area: (None, "no model"))
    monkeypatch.setattr(cx, "_load_cached", lambda run_a, run_b: None)
    body = cc.compare_page_body(cx.explanation_view("rel-121", "rel-120"), "", "2026-11-11 12:00")
    assert "<h3>Фазы сценария</h3>" in body and "фаза 2 · Cart" in body and "15,0 / 10,0 (+50,0 %)" in body
    assert body.count("<tr>") >= 4 and "Провал · Успешно" in body
