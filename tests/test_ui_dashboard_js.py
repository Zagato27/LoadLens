# SPDX-License-Identifier: Apache-2.0
"""Pure helpers of static/js/dashboard.js (LL.dashboard._internals), run under Node."""

import json
import re

import pytest

from tests.js_harness import run_js

FILES = ["static/js/common.js", "static/js/ui.js", "static/js/dashboard.js"]
I = "LL.dashboard._internals"


def js(value):
    return json.dumps(value, ensure_ascii=False)


def call(expression):
    return run_js(FILES, expression)


# Verdicts are codes as /dashboard_data sends them; the page shows the Russian labels.
HISTORY = [
    {"run_name": "r5", "run_id": "id5", "created_at": "2026-10-15T10:00:00Z", "verdict": "risk", "max_rps": 240, "test_type": "step"},
    {"run_name": "r4", "run_id": "", "created_at": "2026-10-14T10:00:00Z", "verdict": "pass", "max_rps": 120, "test_type": "soak"},
    {"run_name": "r3", "run_id": "id3", "created_at": "2026-10-13T10:00:00Z", "verdict": "pass", "max_rps": 200, "test_type": "step"},
    {"run_name": "r2", "run_id": "id2", "created_at": "2026-10-12T10:00:00Z", "verdict": "fail", "max_rps": None, "test_type": ""},
    {"run_name": "r1", "run_id": "id1", "created_at": "2026-10-11T10:00:00Z", "verdict": "no_data", "max_rps": 180, "test_type": "step"},
]


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [
        (240, 200, {"pct": 20, "direction": "up", "arrow": "▲", "text": "+20.0%"}),
        (150, 200, {"pct": -25, "direction": "down", "arrow": "▼", "text": "−25.0%"}),
        ("200.6", 200, {"pct": 0.3, "direction": "flat", "arrow": "≈", "text": "0%"}),
        (101, 100, {"pct": 1, "direction": "up", "arrow": "▲", "text": "+1.0%"}),
    ],
)
def test_rps_delta(current, previous, expected):
    assert call(f"{I}.rpsDelta({js(current)}, {js(previous)})") == expected


@pytest.mark.parametrize(("current", "previous"), [(None, 100), (100, None), (100, 0), ("abc", 100), (100, True), ("", 100)])
def test_rps_delta_needs_two_numbers(current, previous):
    assert call(f"{I}.rpsDelta({js(current)}, {js(previous)})") is None


def test_previous_of_same_type_skips_other_types():
    assert call(f"{I}.previousOfSameType({js(HISTORY)}).run_name") == "r3"
    delta = call(f"{I}.historyDelta({js(HISTORY)})")
    assert delta["previous"]["run_name"] == "r3" and delta["direction"] == "up" and delta["pct"] == 20


def test_previous_of_same_type_treats_untyped_runs_as_step():
    history = [dict(HISTORY[3], max_rps=150), HISTORY[1], HISTORY[2]]
    assert call(f"{I}.previousOfSameType({js(history)}).run_name") == "r3"
    assert call(f"{I}.historyDelta({js(history)}).text") == "−25.0%"


def test_history_delta_without_comparable_run():
    assert call(f"{I}.historyDelta({js(HISTORY[:2])})") is None  # only another type before
    assert call(f"{I}.historyDelta({js(HISTORY[:1])})") is None
    assert call(f"{I}.historyDelta([])") is None
    assert call(f"{I}.historyDelta(null)") is None
    # The immediately previous run of the type decides: no RPS there means no delta.
    history = [HISTORY[0], dict(HISTORY[2], max_rps=None), HISTORY[4]]
    assert call(f"{I}.historyDelta({js(history)})") is None


@pytest.mark.parametrize(
    ("sla", "expected"),
    [
        ({"passed": 2, "total": 3}, {"text": "SLA 2/3", "tone": "warn"}),
        ({"passed": 3, "total": 3}, {"text": "SLA 3/3", "tone": "ok"}),
        ({"passed": 0, "total": 2}, {"text": "SLA 0/2", "tone": "fail"}),
        ({"passed": 0, "total": 0}, {"text": "SLA не задан", "tone": "na"}),
        (None, {"text": "SLA не задан", "tone": "na"}),
        ({"passed": 5, "total": 2}, {"text": "SLA 2/2", "tone": "ok"}),
    ],
)
def test_sla_badge(sla, expected):
    badge = call(f"{I}.slaBadge({js(sla)})")
    assert {"text": badge["text"], "tone": badge["tone"]} == expected
    assert badge["title"]


def test_weeks_model():
    weeks = [
        {"week_start": "2026-10-05", "counts": {"pass": 2, "fail": 1}},
        {"week_start": "2026-10-12", "counts": {"risk": 1, "no_data": "3"}},
        {"week_start": "2026-12-28", "counts": {}},
    ]
    model = call(f"{I}.weeksModel({js(weeks)})")
    assert model["labels"] == ["05.10", "12.10", "28.12"]
    assert model["ranges"] == ["05.10–11.10", "12.10–18.10", "28.12–03.01"]
    assert [s["verdict"] for s in model["series"]] == ["pass", "risk", "fail", "no_data"]
    assert [s["label"] for s in model["series"]] == ["Успешно", "Есть риски", "Провал", "Недостаточно данных"]
    assert [s["tone"] for s in model["series"]] == ["ok", "warn", "fail", "na"]
    assert [s["values"] for s in model["series"]] == [[2, 0, 0], [0, 1, 0], [1, 0, 0], [0, 3, 0]]
    assert model["totals"] == [3, 4, 0] and model["total"] == 7
    summary = call(f"{I}.weeksSummary({I}.weeksModel({js(weeks)}))")
    assert "всего 7" in summary and "Успешно — 2" in summary


def test_weeks_model_of_nothing():
    model = call(f"{I}.weeksModel(undefined)")
    assert model == {"labels": [], "ranges": [], "series": model["series"], "totals": [], "total": 0}
    assert all(s["values"] == [] for s in model["series"])


def test_spark_points_geometry():
    points = call(f"{I}.sparkPoints({js(HISTORY)}, {{ width: 100, height: 40, pad: 10 }})")
    # Oldest first: r1 (180), r2 (no RPS), r3 (200), r4 (120), r5 (240).
    assert [p["run"]["run_name"] for p in points] == ["r1", "r2", "r3", "r4", "r5"]
    assert [p["x"] for p in points] == [10, 30, 50, 70, 90]
    ys = {p["run"]["run_name"]: p["y"] for p in points}
    assert ys["r5"] == 10 and ys["r4"] == 30  # max on top, min at the bottom of the plot area
    assert ys["r2"] == 30  # no RPS: the bottom edge
    assert ys["r3"] == pytest.approx(16.7)
    assert points[1]["value"] is None


def test_spark_points_flat_and_single():
    flat = call(f"{I}.sparkPoints({js([dict(HISTORY[0], max_rps=5), dict(HISTORY[2], max_rps=5)])}, {{ width: 100, height: 40, pad: 10 }})")
    assert [p["y"] for p in flat] == [20, 20]
    single = call(f"{I}.sparkPoints({js(HISTORY[:1])}, {{ width: 100, height: 40, pad: 10 }})")
    assert [(p["x"], p["y"]) for p in single] == [(50, 20)]
    assert call(f"{I}.sparkPoints(null)") == []


def test_sparkline_markup():
    svg = call(f"{I}.sparkline({js(HISTORY)})")
    assert svg.startswith('<svg class="dash-spark" viewBox="0 0 110 26"') and 'role="img"' in svg
    # The line joins step runs only and breaks at the step run without RPS: r1 alone (no line), then r3-r5;
    # the soak run r4 between them is a faded square off the line.
    lines = re.findall(r'<polyline class="dash-spark-line" points="([^"]+)"', svg)
    assert lines == ["55,10 106,4"]
    # Verdict squares centered on the points; the newest one is larger.
    dots = re.findall(r'<rect class="(dash-spark-dot [^"]+)" x="([^"]+)" y="([^"]+)" width="(\d+)"', svg)
    assert [d[0] for d in dots] == [
        "dash-spark-dot na", "dash-spark-dot fail empty", "dash-spark-dot ok", "dash-spark-dot ok other", "dash-spark-dot warn",
    ]
    assert [d[3] for d in dots] == ["5", "5", "5", "5", "7"]
    assert (dots[2][1], dots[2][2]) == ("52.5", "7.5") and (dots[4][1], dots[4][2]) == ("102.5", "0.5")
    assert svg.count('class="dash-spark-hit"') == 5 and "<circle" not in svg
    assert "<title>r5 · 2026-10-15 10:00 · 240.0 RPS · Ступенчатый (max perf) · Есть риски</title>" in svg
    assert "RPS нет" in svg
    assert 'aria-label="Max RPS запусков от старых к новым: 180.0, нет данных, 200.0, 120.0, 240.0"' in svg


def test_sparkline_escapes_names_and_handles_empty():
    history = [{"run_name": '<img src=x onerror="1">', "verdict": "pass", "max_rps": 1}]
    svg = call(f"{I}.sparkline({js(history)})")
    assert "<img" not in svg and "&lt;img" in svg
    assert call(f"{I}.sparkline([])") == '<span class="dim">—</span>'


def test_hero_kpis_with_previous_run():
    last = {"run_name": "r5", "test_type": "step", "max_rps": 240, "start_ms": 1792058400000, "end_ms": 1792062000000}
    kpi = {"previous": {"run_name": "r3", "max_rps": 200}}
    tiles = call(f"{I}.heroKpis({js(last)}, {js(kpi)})")
    assert [t["key"] for t in tiles] == ["rps", "window", "type"]
    rps, window, kind = tiles
    assert rps["label"] == "Max RPS" and rps["value"] == "240.0"
    assert rps["delta"]["text"] == "+20.0%" and rps["note"] == "к «r3»" and "r3" in rps["title"]
    assert window["value"] == "1 ч 00 мин" and window["note"] == "2026-10-15 10:00 — 11:00"
    assert kind["value"] == "Ступенчатый (max perf)" and kind["note"] == "RPS: пик производительности"


def test_hero_kpis_without_previous_run_and_other_types():
    last = {"run_name": "s1", "test_type": "soak", "max_rps": None, "start_ms": None, "end_ms": None}
    tiles = call(f"{I}.heroKpis({js(last)}, {js({'last': {'max_rps': 90}, 'previous': None})})")
    rps, window, kind = tiles
    assert rps["label"] == "RPS удержания" and rps["value"] == "90.0" and rps["delta"] is None
    assert rps["note"] == "нет предыдущего прогона этого типа"
    assert window["value"] == "—" and window["note"] == ""
    assert kind["note"] == "RPS: удержано (среднее)"
    no_rps = call(f"{I}.heroKpis({js(dict(last, max_rps=10))}, {js({'previous': {'run_name': 'p', 'max_rps': None}})})")
    assert no_rps[0]["delta"] is None and no_rps[0]["note"] == "к «p»: нет данных для сравнения"


def test_kpi_tiles():
    data = {
        "kpi": {"runs_7d": 3, "runs_30d": 8, "risky_30d": 2, "risky_share_30d": 0.25},
        "last_run": {"run_name": "r5", "created_at": "2026-10-15T09:55:00Z"},
    }
    tiles = call(f"{I}.kpiTiles({js(data)}, Date.parse('2026-10-15T10:00:00Z'))")
    assert tiles["runs"] == {"value": "3", "note": "за 30 дней: 8"}
    assert tiles["risky"] == {"value": "25%", "meter": 0.25, "note": "2 из 8: риски и провалы"}
    assert tiles["updated"]["value"] == "5 мин назад" and tiles["updated"]["note"] == "отчёт «r5»"
    empty = call(f"{I}.kpiTiles({js({'kpi': {'runs_7d': 0, 'runs_30d': 0, 'risky_share_30d': None}, 'last_run': None})})")
    assert empty["runs"]["value"] == "0" and empty["risky"]["value"] == "—" and empty["risky"]["meter"] == 0
    assert empty["updated"] == {"value": "—", "note": "отчётов пока нет", "title": ""}
    legacy = call(f"{I}.kpiTiles({{}})")
    assert legacy["runs"]["value"] == "—"


def test_jobs_tile():
    assert call(f"{I}.jobsTile(null)")["value"] == "—"
    assert call(f"{I}.jobsTile([])") == {"value": "0", "note": "нет активных задач"}
    assert call(f"{I}.jobsTile([{{ run_name: 'nightly', progress: 41.6 }}, {{}}])") == {"value": "2", "note": "nightly · 42%"}
    assert call(f"{I}.jobsTile(Array.from({{ length: 20 }}, () => ({{}})))")["value"] == "20+"


def test_service_history_falls_back_to_verdicts():
    svc = {"service": "a", "last_run": "r2", "run_id": "id2", "verdict": "fail", "max_rps": 10, "test_type": "step", "recent_verdicts": ["fail", "pass"]}
    history = call(f"{I}.serviceHistory({js(svc)})")
    assert history[0]["run_name"] == "r2" and history[0]["max_rps"] == 10
    assert history[1] == {"verdict": "pass", "max_rps": None}
    assert call(f"{I}.serviceHistory({js(dict(svc, history=HISTORY))}).length") == 5


# ---- load tests going on now -----------------------------------------------------------------------

NOW = "Date.parse('2026-10-16T10:30:00Z')"
ACTIVE = {
    "run_id": "abc", "run_name": "nightly", "service": "checkout", "owner": "alice", "status": "running",
    "status_title": "Идёт нагрузка", "loading": True, "started_at": "2026-10-16T10:18:00Z", "max_seconds": 1800,
    "instances_total": 3, "instances_running": 2, "can_stop": True, "url": "/launch/abc",
    "live": {"ok": True, "message": "", "rps": 123.45, "errors": 0.4, "points": [[1, 1], [2, 2]]},
}


def test_elapsed_and_errors_text():
    assert call(f"{I}.elapsedText('2026-10-16T10:18:00Z', 1800, {NOW})") == "12 мин из 30 мин"
    assert call(f"{I}.elapsedText('2026-10-16T10:29:20Z', 0, {NOW})") == "40 с"
    assert call(f"{I}.elapsedText(null, 3600, {NOW})") == "лимит 1 ч 00 мин"
    assert call(f"{I}.elapsedText(null, null, {NOW})") == ""
    assert call(f"[0, 0.4, 3.456, null].map({I}.errorsText)") == ["0.0%", "0.40%", "3.5%", ""]


def test_active_run_summary():
    s = call(f"{I}.activeRunSummary({js(ACTIVE)}, {NOW})")
    assert s["name"] == "nightly" and s["url"] == "/launch/abc" and s["tone"] == "running" and s["canStop"]
    assert s["meta"] == "checkout · alice" and s["progress"] == "инстансы: 2 из 3 · 12 мин из 30 мин"
    assert s["liveText"] == "123.5 RPS · ошибки 0.40%" and s["liveNote"] == "" and len(s["points"]) == 2

    failed = call(f"{I}.activeRunSummary({js(dict(ACTIVE, live={'ok': False, 'message': 'нет данных', 'rps': None}))}, {NOW})")
    assert failed["liveText"] == "" and failed["liveNote"] == "нет данных"
    waiting = call(f"{I}.activeRunSummary({js(dict(ACTIVE, live=None, status='starting', instances_total=1))}, {NOW})")
    assert waiting["liveNote"] == "Метрики появятся после старта нагрузки" and waiting["progress"] == "12 мин из 30 мин"
    reporting = call(f"{I}.activeRunSummary({js(dict(ACTIVE, live=None, loading=False, status='reporting', message='Строится отчёт'))}, {NOW})")
    assert reporting["liveNote"] == "Строится отчёт" and reporting["progress"] == "инстансы: 2 из 3"
    assert call(f"{I}.activeRunSummary({js(dict(ACTIVE, status='stopping'))}, {NOW}).tone") == "warn"
    assert call(f"{I}.activeRunSummary({{}}, {NOW}).name") == "(без названия)"


def test_active_tile():
    assert call(f"{I}.activeTile(null)")["value"] == "—"
    assert call(f"{I}.activeTile({{ total: 0, runs: [] }})") == {"value": "0", "note": "нет идущих прогонов"}
    assert call(f"{I}.activeTile({js({'total': 3, 'runs': [ACTIVE]})}, {NOW})") == {"value": "3", "note": "nightly · 12 мин из 30 мин"}
    assert call(f"{I}.activeTile({{ total: 2, runs: [] }})") == {"value": "2", "note": ""}


def test_live_spark():
    assert call(f"{I}.liveSpark([[0, 1]])") == "" and call(f"{I}.liveSpark([[0, null], [1, 2]])") == ""
    svg = call(f"{I}.liveSpark([[0, 0], [10, 10]])")
    assert 'width="120" height="32"' in svg and 'class="line" points="3.0,29.0 117.0,3.0"' in svg
    assert 'class="area" points="3.0,29.0 117.0,3.0 117.0,29.0 3.0,29.0"' in svg
    flat = call(f"{I}.liveSpark([[0, 5], [10, 5]], 60, 20)")
    assert 'points="3.0,3.0 57.0,3.0"' in flat  # from zero: a steady load is a line at the top


def test_active_row_markup():
    row = call(f"{I}.activeRowHtml({js(dict(ACTIVE, run_name='<b>x</b>'))}, {NOW})")
    assert "&lt;b&gt;x&lt;/b&gt;" in row and "<b>" not in row
    assert 'data-stop-run="abc"' in row and 'href="/launch/abc"' in row and "dash-live-spark" in row
    quiet = call(f"{I}.activeRowHtml({js(dict(ACTIVE, can_stop=False, live=None))}, {NOW})")
    assert "data-stop-run" not in quiet and "Метрики появятся" in quiet


def test_menu_badge_counts_running_tests():
    shown = call(
        "(() => { const badge = { hidden: true, textContent: '', title: '' };"
        " document.getElementById = (id) => (id === 'navLaunchBadge' ? badge : null);"
        " const seen = []; [3, 0, 150, 'x'].forEach((n) => { LL.activeRuns.show(n); seen.push({ ...badge }); }); return seen; })()"
    )
    assert shown[0] == {"hidden": False, "textContent": "3", "title": "Идут прогоны: 3"}
    assert shown[1]["hidden"] and shown[2]["textContent"] == "99+" and shown[3]["hidden"]


def test_manual_verdict_pill_is_marked():
    manual = {"verdict": "risk", "reason": "деградация <стенда>", "author": "ivanov", "changed_at": "2026-10-17T12:00:00Z"}
    html = call(f"LL.dashboard._internals.verdictPill('risk', '', {js(manual)})")
    assert 'class="pill warn"' in html and ">Есть риски</span>" in html and "verdict-manual-mark" in html and "&lt;стенда&gt;" in html
    assert "verdict-manual-mark" not in call("LL.dashboard._internals.verdictPill('fail')")
    # A missing verdict and an old label (an open tab of an older page) still render.
    assert 'class="pill na"' in call("LL.dashboard._internals.verdictPill(null)") and ">Недостаточно данных<" in call("LL.dashboard._internals.verdictPill('')")
    assert 'class="pill fail"' in call("LL.dashboard._internals.verdictPill('Провал')")
