# SPDX-License-Identifier: Apache-2.0
"""Dashboard of the instrument redesign (plans/instrument-ui-plan.md, stage 2).

/dashboard_data gives the last run its SLA checks and its load profile (``last_run.checks``, ``last_run.profile``);
dashboard.js builds the status line, the verdict columns of 8 weeks, the figures of the last run, the trend in words
and the profile caption; the template carries the status line, the last-run block and the two tables without boxes.
The database is the fake of tests/test_ui_dashboard.py (statements answered by a fragment of their SQL).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from loadlens_app import core  # noqa: F401  (before anything that imports update_page)
from tests.i18n_helpers import use_locale
from tests.js_harness import run_js
from tests.test_ui_dashboard import app_client, fetch, full_db, last_row  # noqa: F401  (app_client is a fixture)

ROOT = Path(__file__).resolve().parents[1]
NBSP = chr(0xA0)

# ---- /dashboard_data: last_run.checks and last_run.profile -------------------------------------------------------

STEPS = [
    {"index": 1, "label": "ступень 1", "start_iso": "2026-10-15T19:10:00+03:00", "end_iso": "2026-10-15T19:20:00+03:00",
     "plateau_start_iso": "2026-10-15T19:11:00+03:00", "plateau_end_iso": "2026-10-15T19:19:00+03:00", "rps": 50.0,
     "selected": False, "after_drop": False, "dip": False, "p95_ms": 120.0, "stable": True},
    {"index": 2, "label": "ступень 2", "start_iso": "2026-10-15T19:20:00+03:00", "end_iso": "2026-10-15T19:30:00+03:00",
     "plateau_start_iso": "2026-10-15T19:21:00+03:00", "plateau_end_iso": "2026-10-15T19:29:00+03:00", "rps": 100.0,
     "selected": True, "after_drop": False, "dip": False, "p95_ms": 180.0, "stable": True},
    {"index": 3, "label": "ступень 3", "start_iso": "2026-10-15T19:30:00+03:00", "end_iso": "2026-10-15T19:40:00+03:00",
     "rps": 150.0, "selected": False, "p95_ms": 450.0, "stable": False},
]
WINDOW = {"start": "2026-10-15T19:22:00+03:00", "end": "2026-10-15T19:29:00+03:00", "level": 100, "series": "all",
          "degraded_level": 150, "degraded_checks": ["p95_latency"]}
CHECKS = [
    {"name": "target_rps", "threshold": 100, "actual": 100.0, "passed": True, "severity": "critical", "message": "m"},
    {"name": "p95_latency", "threshold": "300", "actual": 450.0, "passed": False, "message": "m", "query_label": "q"},
    {"name": "error_rate", "threshold": 1.0, "actual": 0.4, "passed": True},
    {"name": "cpu_usage", "threshold": 80, "actual": None, "passed": "unknown"},
    {"threshold": 5, "passed": True},
]
DETAILS = {"checks": CHECKS, "summary": "s", "load_step_table": {"steps": STEPS, "rps_query": "q"}, "stable_window": WINDOW}
# 2026-10-15 16:30 UTC: the start of the step after the SLA step.
DEGRADATION_MS = 1792081800000
PARSED_STEP = {"spike_analysis": None, "stability_under_load": None, "test_profile": {"mode": "capacity"}, "saturation": None,
               "load_step_table": None, "deterministic_sla": {"stable_window": None}}


def last_run_row(sla_details=DETAILS, parsed=PARSED_STEP, fallback=None):
    """The last-run row of the endpoint: the columns of tests/test_ui_dashboard.py, the manual verdict, the effective
    verdict and the small ``parsed`` object."""
    return last_row(sla_details=sla_details, fallback=fallback) + (None, None, None, None, "Есть риски", parsed)


def payload(client, monkeypatch, **row):
    return fetch(client, monkeypatch, full_db(last_run=[last_run_row(**row)]))


def test_profile_of_the_last_run(app_client, monkeypatch):
    profile = payload(app_client, monkeypatch)["last_run"]["profile"]
    # Only what the staircase draws; the rest of a step row stays in the report.
    assert [sorted(step) for step in profile["steps"]][0] == sorted(
        ["index", "label", "start_iso", "end_iso", "plateau_start_iso", "plateau_end_iso", "rps", "selected", "after_drop", "dip"]
    )
    assert [step["rps"] for step in profile["steps"]] == [50.0, 100.0, 150.0]
    assert "p95_ms" not in profile["steps"][2] and profile["steps"][1]["selected"] is True
    assert profile["window"] == {"start": WINDOW["start"], "end": WINDOW["end"]}
    # The report markers without step ticks and the SLA band (the window draws it); labels in the UI language.
    assert profile["markers"] == [
        {"kind": "line", "t": DEGRADATION_MS, "to": None, "label": "деградация 19:30", "tone": "fail", "role": "degradation"}
    ]
    assert profile["target"] == pytest.approx(100.0)


def test_profile_markers_follow_the_ui_language(app_client, monkeypatch):
    use_locale(app_client, "en")
    profile = payload(app_client, monkeypatch)["last_run"]["profile"]
    assert [m["label"] for m in profile["markers"]] == ["degradation 19:30"]


def test_profile_of_hold_and_spike_runs(app_client, monkeypatch):
    # A hold window equal to the SLA window is drawn once (by the window).
    hold = dict(PARSED_STEP, stability_under_load={"hold": {"start_iso": WINDOW["start"], "end_iso": WINDOW["end"]}})
    window = {"start": WINDOW["start"], "end": WINDOW["end"], "kind": "hold"}
    profile = payload(app_client, monkeypatch, sla_details=dict(DETAILS, stable_window=window), parsed=hold)["last_run"]["profile"]
    assert profile["markers"] == [] and profile["window"] == {"start": WINDOW["start"], "end": WINDOW["end"]}

    spikes = dict(PARSED_STEP, spike_analysis={"spikes": [
        {"index": 1, "label": "spike A", "start_iso": "2026-10-15T19:24:00+03:00", "end_iso": "2026-10-15T19:26:00+03:00"},
    ]})
    profile = payload(app_client, monkeypatch, parsed=spikes)["last_run"]["profile"]
    assert [(m["kind"], m["role"], m["label"], m["tone"]) for m in profile["markers"]] == [("band", "spike", "spike A", "warn")]


def test_profile_of_an_older_row_comes_from_parsed(app_client, monkeypatch):
    parsed = dict(PARSED_STEP, load_step_table={"steps": STEPS}, deterministic_sla={"stable_window": WINDOW})
    fallback = json.dumps([{"name": "target_rps", "threshold": 120, "actual": 100, "passed": False}])
    last = payload(app_client, monkeypatch, sla_details=None, parsed=json.dumps(parsed), fallback=fallback)["last_run"]
    assert [step["index"] for step in last["profile"]["steps"]] == [1, 2, 3]
    assert last["profile"]["window"]["start"] == WINDOW["start"] and last["profile"]["target"] == pytest.approx(120.0)
    assert last["checks"] == [{"name": "target_rps", "title": "целевой RPS", "threshold": 120.0, "actual": 100.0, "passed": False}]


def test_no_profile_without_steps(app_client, monkeypatch):
    last = payload(app_client, monkeypatch, sla_details={"checks": CHECKS, "load_step_table": {"steps": []}})["last_run"]
    assert last["profile"] is None and len(last["checks"]) == 4
    # The fake rows of the older tests (no parsed column) still answer.
    from tests.test_ui_dashboard import full_db as plain_db

    assert fetch(app_client, monkeypatch, plain_db())["last_run"]["profile"] is None


def test_a_broken_profile_keeps_the_dashboard(app_client, monkeypatch):
    from AI import report_markers

    def boom(*args, **kwargs):
        raise RuntimeError("bad table")

    monkeypatch.setattr(report_markers, "markers_for", boom)
    data = payload(app_client, monkeypatch)
    assert data["last_run"]["profile"] is None and data["last_run"]["run_name"] == "nightly-3"
    assert data["kpi"]["runs_30d"] == 4


def test_checks_of_the_last_run(app_client, monkeypatch):
    checks = payload(app_client, monkeypatch)["last_run"]["checks"]
    assert checks == [
        {"name": "target_rps", "title": "целевой RPS", "threshold": 100.0, "actual": 100.0, "passed": True},
        {"name": "p95_latency", "title": "p95 latency", "threshold": 300.0, "actual": 450.0, "passed": False},
        {"name": "error_rate", "title": "доля ошибок", "threshold": 1.0, "actual": 0.4, "passed": True},
        {"name": "cpu_usage", "title": "CPU", "threshold": 80.0, "actual": None, "passed": None},
    ]
    use_locale(app_client, "en")
    titles = [c["title"] for c in payload(app_client, monkeypatch)["last_run"]["checks"]]
    assert titles == ["target RPS", "p95 latency", "error rate", "CPU"]


def test_the_statement_reads_only_the_marker_keys_of_parsed(app_client, monkeypatch):
    db = full_db(last_run=[last_run_row()])
    fetch(app_client, monkeypatch, db)
    text = db.call("last_run")[1]
    assert "parsed AS parsed_doc" in text
    assert "jsonb_build_object('spike_analysis', l.parsed_doc->'spike_analysis'" in text
    assert "'deterministic_sla', jsonb_build_object('stable_window', l.parsed_doc->'deterministic_sla'->'stable_window')" in text
    assert "l.parsed_doc," not in text  # the whole analysis never leaves the database


# ---- dashboard.js ---------------------------------------------------------------------------------------------------

FILES = ["static/js/common.js", "static/js/ui.js", "static/js/dashboard.js"]
WITH_CHARTS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/dashboard.js"]
I = "LL.dashboard._internals"
NOW = "Date.parse('2026-10-20T12:00:00Z')"


def js(value):
    return json.dumps(value, ensure_ascii=False)


def call(expression, locale="ru", files=FILES):
    return run_js(files, expression, locale=locale)


def run_all(expressions, locale="ru", files=FILES):
    body = ", ".join(f"{js(name)}: ({expr})" for name, expr in expressions.items())
    return run_js(files, "({" + body + "})", locale=locale)


DATA = {
    "kpi": {"runs_7d": 3, "runs_30d": 8, "risky_30d": 2, "risky_share_30d": 0.25},
    "last_run": {"run_name": "r5", "created_at": "2026-10-15T09:55:00Z"},
}


def test_status_line():
    out = run_all({
        "full": f"{I}.statusLine({js(DATA)}, Date.parse('2026-10-15T10:00:00Z'))",
        "just_now": f"{I}.statusLine({js(DATA)}, Date.parse('2026-10-15T09:55:30Z')).updated",
        "empty": f"{I}.statusLine({js({'kpi': {'runs_7d': 0, 'runs_30d': 0, 'risky_30d': 0, 'risky_share_30d': None}, 'last_run': None})})",
        "one": f"{I}.statusLine({js({'kpi': {'runs_7d': 1, 'runs_30d': 5, 'risky_share_30d': 0.4}})}).runs.note",
        "legacy": f"{I}.statusLine({{}})",
    })
    assert out["full"] == {
        "runs": {"value": "3", "note": "прогона за 7 дней", "title": ""},
        "risky": {"value": "8", "note": "за 30 дней", "warn": "2 с рисками или провалами", "title": "2 из 8: риски и провалы"},
        "updated": {"value": "5 мин", "note": "с последнего отчёта", "title": "2026-10-15 09:55 · отчёт «r5»"},
    }
    assert out["just_now"] == {"value": "", "note": "последний отчёт только что", "title": "2026-10-15 09:55 · отчёт «r5»"}
    assert out["empty"]["runs"]["note"] == "прогонов за 7 дней" and out["empty"]["risky"]["warn"] == ""
    assert out["empty"]["updated"] == {"value": "", "note": "отчётов пока нет", "title": ""}
    assert out["one"] == "прогон за 7 дней"
    assert out["legacy"]["runs"]["value"] == "—" and out["legacy"]["risky"]["value"] == "—"


def test_status_line_in_en():
    out = run_all({
        "full": f"{I}.statusLine({js(DATA)}, Date.parse('2026-10-15T10:00:00Z'))",
        "jobs": f"[{I}.jobsStat(null), {I}.jobsStat([]), {I}.jobsStat([{{ run_name: 'nightly', progress: 41.6 }}])]",
        "active": f"[{I}.activeStat(null), {I}.activeStat({{ total: 2, runs: [{{ run_name: 'a', status_title: 'Running' }}] }})]",
    }, "en")
    assert [out["full"][k]["note"] for k in ("runs", "risky", "updated")] == ["runs in 7 days", "in 30 days", "since the last report"]
    assert out["full"]["risky"]["warn"] == "2 at risk or failed" and out["full"]["updated"]["value"] == "5 min"
    assert out["jobs"] == [
        {"value": "—", "note": "report jobs running", "title": "job list unavailable"},
        {"value": "0", "note": "report jobs running", "title": "no active jobs"},
        {"value": "1", "note": "report job running", "title": "nightly · 42%"},
    ]
    assert out["active"] == [
        {"value": "—", "note": "tests running now", "title": "run list unavailable"},
        {"value": "2", "note": "tests running now", "title": "a · Running"},
    ]


def test_jobs_and_active_stats_in_ru():
    out = run_all({
        "jobs": f"[{I}.jobsStat([{{}}]), {I}.jobsStat([{{}}, {{}}]), {I}.jobsStat(Array.from({{ length: 20 }}, () => ({{}})))]",
        "active": f"{I}.activeStat({{ total: 1, runs: [{{ run_name: 'nightly', loading: true, started_at: '2026-10-20T11:48:00Z', max_seconds: 1800 }}] }}, {NOW})",
    })
    assert [(s["value"], s["note"]) for s in out["jobs"]] == [("1", "задача в работе"), ("2", "задачи в работе"), ("20+", "задач в работе")]
    assert out["active"] == {"value": "1", "note": "тест идёт сейчас", "title": "nightly · 12 мин из 30 мин"}


WEEKS = [
    {"week_start": "2026-10-05", "counts": {"pass": 2, "fail": 1}},
    {"week_start": "2026-10-12", "counts": {"risk": 1, "no_data": 3}},
    {"week_start": "2026-10-19", "counts": {}},
]


def test_weeks_svg():
    svg = call(f"{I}.weeksSvg({I}.weeksModel({js(WEEKS)}))")
    assert svg.startswith('<svg class="dash-weeks-svg" viewBox="0 0 236 64" width="236" height="64" aria-hidden="true">')
    rects = re.findall(r'<rect class="seg ([a-z]+)" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"/>', svg)
    # Stacked from the axis (y=48) in verdict order, 8 px per run; an empty week is a 2 px stub.
    assert rects == [
        ("ok", "2", "32", "46.9", "16"), ("fail", "2", "24", "46.9", "8"),
        ("warn", "75.3", "40", "46.9", "8"), ("na", "75.3", "16", "46.9", "24"),
        ("empty", "148.7", "46", "46.9", "2"),
    ]
    assert "<title>Неделя 05.10–11.10\nУспешно: 2\nПровал: 1\nВсего: 3</title>" in svg
    assert '<text x="2" y="62">05.10</text>' in svg and '<text x="195.6" y="62" text-anchor="end">19.10</text>' in svg
    assert call(f"{I}.weeksSvg({I}.weeksModel([]))") == ""


def test_weeks_svg_scales_a_busy_week():
    busy = [{"week_start": "2026-10-12", "counts": {"pass": 20, "risk": 3}}, {"week_start": "2026-10-19", "counts": {"fail": 1}}]
    svg = call(f"{I}.weeksSvg({I}.weeksModel({js(busy)}))")
    heights = [float(h) for h in re.findall(r'class="seg [a-z]+"[^>]*height="([\d.]+)"', svg)]
    assert sum(heights[:2]) == pytest.approx(46, abs=0.2)  # the busiest week fills the plot (2..48)
    assert heights[2] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("when", "ru", "en"),
    [
        ("2026-10-20T09:05:00Z", "в 09:05", "at 09:05"),
        ("2026-10-19T23:00:00Z", "вчера", "yesterday"),
        ("2026-10-17T12:00:00Z", "3 дня назад", "3 days ago"),
        ("2026-10-15T12:00:00Z", "5 дней назад", "5 days ago"),
        ("2026-10-13T08:00:00Z", "неделю назад", "a week ago"),
        ("2026-10-06T08:00:00Z", "2 недели назад", "2 weeks ago"),
        ("2026-09-29T08:00:00Z", "3 недели назад", "3 weeks ago"),
        ("2026-09-20T08:00:00Z", "месяц назад", "a month ago"),
        ("2026-08-01T08:00:00Z", "2 месяца назад", "2 months ago"),
        (None, "", ""),
    ],
)
def test_age_words(when, ru, en):
    expr = f"{I}.ageWords({js(when)}, {NOW})"
    assert call(expr) == ru
    assert call(expr, "en") == en


HISTORY = [
    {"run_name": "r5", "created_at": "2026-10-15T10:00:00Z", "verdict": "risk", "max_rps": 240, "test_type": "step"},
    {"run_name": "r4", "created_at": "2026-10-14T10:00:00Z", "verdict": "pass", "max_rps": 120, "test_type": "soak"},
    {"run_name": "r3", "created_at": "2026-10-13T10:00:00Z", "verdict": "pass", "max_rps": 200, "test_type": "step"},
]


def test_trend_words():
    def history(rps):
        return [dict(HISTORY[0], max_rps=rps)] + HISTORY[1:]

    out = run_all({
        "better": f"{I}.trendWords({js(history(240))}, {NOW})",
        "worse": f"{I}.trendWords({js(history(150))}, {NOW})",
        "flat": f"{I}.trendWords({js(history(200.6))}, {NOW})",
        "small": f"{I}.trendWords({js(history(210.4))}, {NOW}).text",
        "alone": f"{I}.trendWords({js(HISTORY[:2])}, {NOW})",
    })
    assert out["better"]["text"] == f"лучше на 20{NBSP}%" and out["better"]["since"] == "чем неделю назад"
    assert out["better"]["direction"] == "up" and out["better"]["previous"]["run_name"] == "r3"
    assert out["worse"]["text"] == f"хуже на 25{NBSP}%" and out["worse"]["direction"] == "down"
    assert (out["flat"]["text"], out["flat"]["since"]) == ("без изменений", "как неделю назад")
    assert out["small"] == f"лучше на 5,2{NBSP}%"
    assert out["alone"] is None
    en = run_all({
        "better": f"{I}.trendWords({js(history(240))}, {NOW})",
        "small": f"{I}.trendWords({js(history(189.6))}, {NOW})",
        "flat": f"{I}.trendWords({js(history(200))}, {NOW})",
    }, "en")
    assert (en["better"]["text"], en["better"]["since"]) == ("20% better", "than a week ago")
    assert en["small"]["text"] == "5.2% worse"
    assert (en["flat"]["text"], en["flat"]["since"]) == ("no change", "same as a week ago")


def test_service_row_shows_the_trend_in_words():
    svc = {"service": "checkout", "last_run": "r5", "run_id": "id5", "verdict": "risk", "max_rps": 240, "test_type": "step",
           "created_at": "2026-10-15T10:00:00Z", "history": HISTORY}
    row = call(f"{I}.serviceRowHtml({js(svc)}, {NOW})")
    assert '<span class="dash-delta up">лучше на 20' in row and '<span class="sub">чем неделю назад</span>' in row
    assert 'title="К прогону «r3» того же типа: 200.0 RPS"' in row
    assert '<a class="link" href="/compare?run_a=r5&amp;run_b=r3"' in row and 'class="actions nowrap"' in row
    assert '<td class="n">240.0</td>' in row and "dash-spark" in row
    alone = call(f"{I}.serviceRowHtml({js(dict(svc, history=HISTORY[:1]))}, {NOW})")
    assert "dash-trend-words" not in alone


LAST = {
    "run_name": "demo-checkout-step", "service": "demo-checkout", "test_type": "step", "max_rps": 249.3, "verdict": "risk",
    "start_ms": 1792058400000, "end_ms": 1792062000000, "created_at": "2026-10-15T11:02:00Z",
    "sla": {"passed": 2, "failed": 1, "total": 3},
    "checks": [
        {"name": "target_rps", "title": "целевой RPS", "threshold": 200, "actual": 249.3, "passed": True},
        {"name": "p95_latency", "title": "p95 latency", "threshold": 300, "actual": 450, "passed": False},
        {"name": "error_rate", "title": "доля ошибок", "threshold": 1, "actual": 0.4, "passed": True},
    ],
}


def test_last_figures():
    figs = call(f"{I}.lastFigures({js(LAST)}, {{}})")
    assert [f["key"] for f in figs] == ["rps", "p95", "errors", "sla"]
    rps, p95, errors, sla = figs
    assert (rps["value"], rps["unit"], rps["tone"]) == ("249.3", "RPS", "")
    assert (rps["caption"], rps["word"], rps["wordTone"]) == ("Max RPS · цель ≥ 200", "достигнута", "ok")
    assert rps["bar"] == {"fill": 83.1, "tick": 66.7, "tone": "ok"}
    assert (p95["value"], p95["unit"], p95["tone"], p95["caption"], p95["word"]) == ("450", "мс", "fail", "p95 · порог ≤ 300", "нарушен")
    assert p95["bar"] == {"fill": 100, "tick": 66.7, "tone": "fail"} and p95["title"] == "p95 latency"
    assert (errors["value"], errors["unit"], errors["caption"], errors["word"]) == ("0.4", "%", "ошибки · порог ≤ 1", "соблюдён")
    assert errors["bar"]["fill"] == 26.7
    assert (sla["value"], sla["unit"], sla["caption"]) == ("2", "/ 3", "критериев SLA пройдено")
    assert [s["tone"] for s in sla["squares"]] == ["ok", "fail", "ok"] and sla["squares"][1]["title"] == "p95 latency"


def test_last_figures_without_checks_and_unknown_values():
    bare = dict(LAST, checks=[], sla={"passed": 0, "total": 0}, max_rps=None)
    figs = call(f"{I}.lastFigures({js(bare)}, {js({'last': {'max_rps': 90}})})")
    # No p95 and error checks: the test window keeps the block from shrinking to one number; no SLA figure.
    assert [f["key"] for f in figs] == ["rps", "window"]
    assert (figs[0]["value"], figs[0]["caption"], figs[0]["bar"]) == ("90.0", "Max RPS", None)
    assert (figs[1]["value"], figs[1]["caption"], figs[1]["note"]) == ("1 ч 00 мин", "Окно теста", "2026-10-15 10:00 — 11:00")
    unknown = dict(LAST, checks=[{"name": "p95_latency", "threshold": None, "actual": None, "passed": None}], sla={"passed": 1, "failed": 1, "total": 2})
    figs = call(f"{I}.lastFigures({js(unknown)}, {{}})")
    p95 = figs[1]
    assert (p95["value"], p95["unit"], p95["caption"], p95["word"], p95["wordTone"], p95["bar"]) == ("—", "", "p95", "не проверен", "na", None)
    # Squares from the counts when the checks are not judged one by one.
    assert [s["tone"] for s in figs[-1]["squares"]] == ["ok", "fail"]


def test_last_figures_in_en():
    figs = call(f"{I}.lastFigures({js(LAST)}, {{}})", "en")
    assert [f["caption"] for f in figs] == ["Max RPS · target ≥ 200", "p95 · limit ≤ 300", "errors · limit ≤ 1", "SLA criteria passed"]
    assert [f["word"] for f in figs[:3]] == ["reached", "breached", "met"] and figs[1]["unit"] == "ms"


def test_figure_markup():
    html = call(f"{I}.figureHtml({I}.lastFigures({js(LAST)}, {{}})[1])")
    assert html == (
        '<div class="fig dash-fig-p95" title="p95 latency"><div class="v fail">450<small>мс</small></div>'
        '<div class="k fail">p95 · порог ≤ 300 <b>нарушен</b></div>'
        '<div class="fig-bar" aria-hidden="true"><i class="fail" style="width:100%"></i><b style="left:66.7%"></b></div></div>'
    )
    sla = call(f"{I}.figureHtml({I}.lastFigures({js(LAST)}, {{}})[3])")
    assert '<div class="dash-fig-squares"><span class="sq ok" title="целевой RPS"></span><span class="sq fail"' in sla
    escaped = call(f"{I}.figureHtml({{ key: 'x', value: '<b>', unit: '', caption: 'a<i>', title: '\"q\"' }})")
    assert '<div class="v">&lt;b&gt;</div>' in escaped
    assert "&lt;i&gt;" in escaped and 'title="&quot;q&quot;"' in escaped


def test_period_previous_note_and_profile_caption():
    out = run_all({
        "period": f"[{I}.heroPeriod({js(LAST)}, true), {I}.heroPeriod({js(LAST)}, false), {I}.heroPeriod({js(dict(LAST, end_ms=1792058400000 + 86400000))}, true)]",
        "prev": f"{I}.previousNote({js(LAST)}, {js({'previous': {'run_name': 'r3', 'max_rps': 238}})})",
        "prev_none": f"{I}.previousNote({js(LAST)}, {{}})",
        "prev_no_rps": f"{I}.previousNote({js(LAST)}, {js({'previous': {'run_name': 'r3', 'max_rps': None}})})",
        "caption": f"[{I}.profileCaption({{ sla: {{ index: 5, rps: 250, startText: '19:50', endText: '19:54' }} }}),"
                   f" {I}.profileCaption({{ sla: {{ index: null, rps: null, startText: '19:50', endText: '19:54' }} }}),"
                   f" {I}.profileCaption({{ sla: {{ index: 2, rps: 249.5, startText: '19:50', endText: '19:54' }} }}),"
                   f" {I}.profileCaption(null)]",
    })
    assert out["period"] == [
        "Ступенчатый (max perf) · 2026-10-15 10:00–11:00 · 1 ч 00 мин",
        "Ступенчатый (max perf)",
        "Ступенчатый (max perf) · 2026-10-15 10:00 — 2026-10-16 10:00 · 1 д 0 ч",
    ]
    assert out["prev"] == f"предыдущий r3 · Max RPS лучше на 4,7{NBSP}%"
    assert out["prev_none"] == "нет предыдущего прогона этого типа"
    assert out["prev_no_rps"] == "предыдущий r3 · нет данных для сравнения"
    assert out["caption"] == ["SLA · ступень 5 · 250 RPS · 19:50–19:54", "SLA · 19:50–19:54", "SLA · ступень 2 · 249.5 RPS · 19:50–19:54", ""]


# renderHero draws into stub elements kept by id; the result is what it left there.
HERO = (
    "((last, kpi) => { const els = {};"
    " document.getElementById = (id) => els[id] || (els[id] = document.createElement('div'));"
    f" {I}.renderHero(last, kpi);"
    " const box = els.lastProfileBox; const card = els.lastCard;"
    " return { tone: card.className, figs: els.lastKpis.innerHTML, badges: els.lastBadges.innerHTML, period: els.lastPeriod.innerHTML,"
    " prev: els.lastPrevNote.textContent, boxHidden: box.hidden, hasProfile: card.classList.contains('has-profile'),"
    " profile: els.lastProfile.innerHTML, caption: els.lastProfileSla.textContent }; })"
)


def test_render_hero_with_the_load_profile():
    profile = {"steps": STEPS, "window": {"start": WINDOW["start"], "end": WINDOW["end"]}, "target": 100,
               "markers": [{"kind": "line", "t": DEGRADATION_MS, "to": None, "label": "деградация 19:30", "tone": "fail", "role": "degradation"}]}
    out = call(f"{HERO}({js(dict(LAST, profile=profile))}, {{}})", files=WITH_CHARTS)
    assert out["tone"].split() == ["tone-warn", "has-profile"] and out["hasProfile"] and out["boxHidden"] is False
    assert out["profile"].startswith("<svg") and "деградация 19:30" in out["profile"] and "100 · цель" in out["profile"]
    assert out["caption"] == "SLA · ступень 2 · 100 RPS · 19:22–19:29"
    assert out["period"] == (
        '<span class="nowrap">Ступенчатый (max perf)</span> · <span class="nowrap">2026-10-15 10:00–11:00</span>'
        ' · <span class="nowrap">1 ч 00 мин</span>'
    )
    assert out["prev"] == "нет предыдущего прогона этого типа"
    assert ">SLA не задан<" not in out["badges"] and 'class="pill warn pill-lg"' in out["badges"]
    assert out["figs"].count('<div class="fig ') == 4


def test_render_hero_without_a_profile_hides_the_column():
    out = call(f"{HERO}({js(dict(LAST, profile=None, sla={'passed': 0, 'total': 0}, checks=[]))}, {{}})", files=WITH_CHARTS)
    assert out["boxHidden"] is True and not out["hasProfile"] and out["profile"] == "" and out["caption"] == ""
    assert ">SLA не задан</span>" in out["badges"] and "Окно теста" in out["figs"]
    # The window figure carries the window, so the meta line keeps only the test type.
    assert out["period"] == '<span class="nowrap">Ступенчатый (max perf)</span>'
    # Without charts.js (or before it loads) the column stays hidden.
    plain = call(f"{HERO}({js(dict(LAST, profile={'steps': STEPS}))}, {{}})")
    assert plain["boxHidden"] is True and plain["profile"] == ""


# ---- template and styles --------------------------------------------------------------------------------------------


@pytest.fixture
def page_client(monkeypatch, tmp_path):
    import update_page
    from app import create_app

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_dashboard_markup(page_client):
    html = page_client.get("/").get_data(as_text=True)
    main = html[html.index('<main id="main"'):html.index("</main>")]
    # Status line under the title (the kpi ids moved into it) and the verdict columns instead of the tiles and the chart.
    assert 'class="dash-stats dash-tiles dash-tiles-5" role="group" id="dashTiles"' in main
    for item in ("kpiRunsValue", "kpiRiskyNote", "kpiActiveValue", "kpiJobsNote", "kpiUpdatedValue"):
        assert f'id="{item}"' in main
    assert "dash-tile-value" not in main and "<canvas" not in main
    assert 'class="dash-weeks-chart" id="weeksChart" role="img"' in main
    # The last run is the one object with a container; its profile column starts hidden.
    assert 'class="dash-run tone-na" id="lastCard"' in main and 'id="lastProfileBox" hidden' in main
    assert 'class="figs dash-run-figs" id="lastKpis"' in main
    # No card boxes: tables are table.grid (data-table keeps the mobile cards), sections hang on a rule.
    assert 'class="card' not in main
    assert 'class="grid data-table dash-services" id="servicesTable"' in main and 'class="grid data-table dense dash-recent"' in main
    assert '<section class="dash-active" id="activeRuns"' in main and '<section class="dash-jobs" id="jobsCard" hidden>' in main


def test_dashboard_styles_live_in_the_page_section():
    css = (ROOT / "static" / "css" / "app.css").read_text(encoding="utf-8")
    start = css.index("/* ===== page: dashboard ===== */")
    end = css.index("/* ===== end page: dashboard ===== */")
    section = css[start:end]
    for rule in (".dash-page .dash-run.tone-warn", ".dash-page .dash-stats", ".dash-weeks-svg .seg.fail", ".dash-page .dash-trend-words"):
        assert rule in section, rule
    assert "@media (max-width: 720px)" in section
