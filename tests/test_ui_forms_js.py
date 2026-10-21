# SPDX-License-Identifier: Apache-2.0
"""Pure helpers of the stage 4 forms (ui-plan section 4) run under node: the service picker, the test
window chips and progress timing of the new report form, the launch form instance cards and the project
readiness checklist."""

import json

import pytest

from tests.js_harness import run_js

BASE = ["static/js/common.js", "static/js/ui.js"]
PICKER = BASE + ["static/js/service_picker.js"]
NEW = BASE + ["static/js/new.js"]
LAUNCH = BASE + ["static/js/launch.js"]
PROJECTS = BASE + ["static/js/projects.js"]

SERVICES = [{"id": "a", "title": "Alpha"}, {"id": "b", "title": "b"}, {"id": "c", "title": "Gamma"}]


def _js(value):
    return json.dumps(value, ensure_ascii=False)


# ---- service picker ----------------------------------------------------------------------------------


def test_recent_services_come_first_with_the_recent_meta():
    options = run_js(PICKER, f"LL.servicePicker._internals.serviceOptions({_js(SERVICES)}, ['c', 'gone', 'a', 'c'], '')")
    assert [(o["value"], o["label"], o["meta"]) for o in options] == [
        ("c", "Gamma", "недавно · c"),
        ("a", "Alpha", "недавно · a"),
        ("b", "b", ""),
    ]


@pytest.mark.parametrize("query, expected", [("gam", ["c"]), ("B", ["b"]), ("A", ["c", "a"]), ("zzz", [])])
def test_service_options_filter_by_id_or_title(query, expected):
    options = run_js(PICKER, f"LL.servicePicker._internals.serviceOptions({_js(SERVICES)}, ['c', 'a'], {_js(query)})")
    assert [o["value"] for o in options] == expected


def test_recent_list_moves_to_front_and_keeps_five():
    internals = "LL.servicePicker._internals"
    assert run_js(PICKER, f"{internals}.withRecent(['a', 'b', 'c', 'd', 'e'], 'c')") == ["c", "a", "b", "d", "e"]
    assert run_js(PICKER, f"{internals}.withRecent(['a', 'b', 'c', 'd', 'e'], 'f')") == ["f", "a", "b", "c", "d"]
    assert run_js(PICKER, f"{internals}.withRecent(null, 'x')") == ["x"]


def test_remember_writes_recent_services_pref():
    value = run_js(
        PICKER,
        "LL.servicePicker.remember('b'); LL.servicePicker.remember('z'); LL.prefs.get('recentServices')",
        local_storage={"loadlens.recentServices": json.dumps(["a", "b"])},
    )
    assert value == ["z", "b", "a"]


def test_exact_match_and_normalization():
    internals = "LL.servicePicker._internals"
    assert run_js(PICKER, f"{internals}.exactMatch({_js(SERVICES)}, ' gamma ').id") == "c"
    assert run_js(PICKER, f"{internals}.exactMatch({_js(SERVICES)}, 'A').id") == "a"
    assert run_js(PICKER, f"{internals}.exactMatch({_js(SERVICES)}, 'gam')") is None
    normalized = run_js(PICKER, f"{internals}.normalizeServices(['x', {{id: 'y'}}, {{id: 'x', title: 'dup'}}, {{title: 'no id'}}, null])")
    assert normalized == [{"id": "x", "title": "x"}, {"id": "y", "title": "y"}]


# ---- new report: test window chips --------------------------------------------------------------------

NEW_INTERNALS = "LL.newReport._internals"


def test_runs_merge_into_overlapping_windows_and_sort_newest_first():
    windows = [
        {"start": "2026-10-15T10:00:00+00:00", "end": "2026-10-15T11:00:00+00:00", "rps_peak": 249.4, "ongoing": False},
        {"start": "2026-10-15T07:00:00+00:00", "end": "2026-10-15T07:30:00+00:00", "rps_peak": 50, "ongoing": False},
    ]
    runs = [
        {"run_name": "release-2.5", "start": "2026-10-15T09:58:00+00:00", "end": "2026-10-15T11:01:00+00:00"},
        {"run_name": "smoke", "start": "2026-10-15T08:00:00+00:00", "end": "2026-10-15T08:10:00+00:00"},
        {"run_name": "broken", "start": "x", "end": "y"},
    ]
    items = run_js(NEW, f"{NEW_INTERNALS}.mergeWindows({_js(windows)}, {_js(runs)})")
    assert [(i["name"], i["peak"]) for i in items] == [("release-2.5", 249.4), ("smoke", None), ("", 50)]
    # The run's exact times replace the stretch's (09:58 and 11:01 UTC).
    assert items[0]["start"] == 1792058280000 and items[0]["end"] == 1792062060000
    assert run_js(NEW, f"{NEW_INTERNALS}.mergeWindows({_js(windows)}, [], 1).length") == 1
    assert run_js(NEW, f"{NEW_INTERNALS}.mergeWindows(null, undefined)") == []


def test_window_label_formats_today_yesterday_and_older_dates():
    now = "Date.parse('2026-10-15T18:00:00Z')"
    label = run_js(NEW, f"{NEW_INTERNALS}.windowLabel({{start: Date.parse('2026-10-15T15:57:00Z'), end: Date.parse('2026-10-15T16:57:00Z'), peak: 249.4, name: '', ongoing: false}}, {now})")
    assert label == "15:57–16:57 (1 ч 00 мин, пик 249 RPS)"
    named = run_js(NEW, f"{NEW_INTERNALS}.windowLabel({{start: Date.parse('2026-10-14T23:50:00Z'), end: Date.parse('2026-10-15T00:40:00Z'), peak: null, name: 'nightly', ongoing: false}}, {now})")
    assert named == "nightly · вчера 23:50–00:40 (50 мин)"
    older = run_js(NEW, f"{NEW_INTERNALS}.windowLabel({{start: Date.parse('2026-10-13T10:00:00Z'), end: Date.parse('2026-10-13T10:20:00Z'), peak: 4.25, name: '', ongoing: true}}, {now})")
    assert older == "13.10 10:00–10:20 (20 мин, пик 4.3 RPS, идёт)"


def test_window_label_shows_the_times_the_chip_fills_in():
    label = run_js(NEW, f"{NEW_INTERNALS}.windowLabel({{start: Date.parse('2026-10-15T16:00:54Z'), end: Date.parse('2026-10-15T17:02:24Z'), peak: 0, name: '', ongoing: false}}, Date.parse('2026-10-15T18:00:00Z'))")
    assert label == "16:00–17:03 (1 ч 01 мин)"


def test_window_label_uses_the_browser_time_zone():
    label = run_js(NEW, f"{NEW_INTERNALS}.windowLabel({{start: Date.parse('2026-10-15T12:00:00Z'), end: Date.parse('2026-10-15T12:30:00Z'), peak: 10, name: '', ongoing: false}}, Date.parse('2026-10-15T13:00:00Z'))", tz="Europe/Moscow")
    assert label.startswith("15:00–15:30")


def test_window_inputs_round_outwards_to_the_minute():
    values = run_js(NEW, f"{NEW_INTERNALS}.windowInputs({{start: Date.parse('2026-10-15T10:00:30Z'), end: Date.parse('2026-10-15T10:59:10Z')}})")
    assert values == {"start": "2026-10-15T10:00", "end": "2026-10-15T11:00"}


# ---- new report: progress timing ----------------------------------------------------------------------


@pytest.mark.parametrize("ms, text", [(0, "0:00"), (135_000, "2:15"), (3_723_000, "1:02:03"), (-5, "0:00"), (None, "0:00")])
def test_format_elapsed(ms, text):
    assert run_js(NEW, f"{NEW_INTERNALS}.formatElapsed({_js(ms)})") == text


def _job(kind, minutes, status="done", start="2026-10-15T10:00:00Z"):
    from datetime import datetime, timedelta

    begin = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return {"kind": kind, "status": status, "created_at": begin.isoformat(), "updated_at": (begin + timedelta(minutes=minutes)).isoformat()}


def test_median_duration_of_the_same_kind():
    jobs = [_job("report", 4), _job("confluence", 30), _job("report", 6), _job("report", 5), _job("report", 100, status="error"), _job("report", 8)]
    assert run_js(NEW, f"{NEW_INTERNALS}.medianDurationMs({_js(jobs)}, 'report')") == 5.5 * 60_000
    assert run_js(NEW, f"{NEW_INTERNALS}.medianDurationMs({_js(jobs[:3])}, 'report')") is None
    # Only the latest ten of the kind count: the list is newest first.
    many = [_job("report", 2)] * 10 + [_job("report", 60)] * 5
    assert run_js(NEW, f"{NEW_INTERNALS}.medianDurationMs({_js(many)}, 'report')") == 2 * 60_000
    assert run_js(NEW, f"{NEW_INTERNALS}.medianDurationMs(null, 'report')") is None


def test_typical_text():
    assert run_js(NEW, f"{NEW_INTERNALS}.typicalText(6.4 * 60000)") == "обычно около 6 мин"
    assert run_js(NEW, f"{NEW_INTERNALS}.typicalText(20000)") == "обычно около 1 мин"
    assert run_js(NEW, f"{NEW_INTERNALS}.typicalText(null)") == ""


# ---- launch form --------------------------------------------------------------------------------------

LAUNCH_INTERNALS = "LL.launch._internals"


def test_instance_summary():
    assert run_js(LAUNCH, f"{LAUNCH_INTERNALS}.instanceSummary(0, {{name: 'rest', generator: 'Gen 1', tool: 'k6'}})") == "Инстанс 1 · rest · Gen 1 · k6"
    assert run_js(LAUNCH, f"{LAUNCH_INTERNALS}.instanceSummary(2, {{name: ' ', generator: 'Вручную', tool: ''}})") == "Инстанс 3 · Вручную"
    assert run_js(LAUNCH, f"{LAUNCH_INTERNALS}.instanceSummary(1)") == "Инстанс 2"


def test_cards_collapse_from_the_third_instance():
    assert run_js(LAUNCH, f"[1, 2, 3, 5].map((n) => {LAUNCH_INTERNALS}.cardsToCollapse(n))") == [[], [], [0, 1], [0, 1, 2, 3]]


def test_commands_go_to_their_cards_by_index():
    commands = [{"index": 1, "name": "b"}, {"index": 0, "name": "a"}, {"index": 7, "name": "lost"}, {"name": "no index"}, {"index": 0, "name": "dup"}]
    result = run_js(LAUNCH, f"{LAUNCH_INTERNALS}.commandsByIndex({_js(commands)}, 2)")
    assert {k: v["name"] for k, v in result["byIndex"].items()} == {"0": "a", "1": "b"}
    assert [c["name"] for c in result["leftovers"]] == ["lost", "no index", "dup"]


def test_ci_secrets_never_leave_the_form():
    values = {"VUS": "10", "API_KEY": "s3cret-zz", "OLD": "***", "DEBUG": True}
    result = run_js(LAUNCH, f"{LAUNCH_INTERNALS}.blankSecrets({_js(values)}, ['API_KEY'])")
    assert result == {"VUS": "10", "API_KEY": "", "OLD": "", "DEBUG": True}
    assert run_js(LAUNCH, f"{LAUNCH_INTERNALS}.blankSecrets(null, null)") == {}


def test_repeat_of_a_ci_instance():
    ci = {"kind": "gitlab", "ref": "main", "params": {"VUS": "10", "API_KEY": "***"}, "extra": {"A": "1", "B": "2"}, "secret": ["API_KEY"]}
    spec = run_js(LAUNCH, f"{LAUNCH_INTERNALS}.ciSpecFromView({_js(ci)})")
    assert spec == {"ci_ref": "main", "ci_params": {"VUS": "10", "API_KEY": ""}, "ci_extra": "A=1\nB=2", "ci_secret": ["API_KEY"]}
    assert run_js(LAUNCH, f"{LAUNCH_INTERNALS}.ciSpecFromView(null)") == {}


def test_ci_field_values():
    expr = (f"[{LAUNCH_INTERNALS}.ciFieldValue({{name: 'V', default: '10'}}, {{}}),"
            f" {LAUNCH_INTERNALS}.ciFieldValue({{name: 'V', default: '10'}}, {{V: '20'}}),"
            f" {LAUNCH_INTERNALS}.ciFieldValue({{name: 'K', secret: true, default: 'x'}}, {{}}),"
            f" {LAUNCH_INTERNALS}.ciFieldValue({{name: 'D', type: 'boolean', default: 'true'}}, {{}}),"
            f" {LAUNCH_INTERNALS}.ciFieldValue({{name: 'D', type: 'boolean', default: true}}, {{D: false}})]")
    assert run_js(LAUNCH, expr) == ["10", "20", "", True, False]


# ---- project readiness checklist ----------------------------------------------------------------------

PROJECTS_INTERNALS = "LL.projects._internals"
FULL = {
    "target_rps": 250, "performance_query": "RPS", "has_system_context": True, "query_counts": {"jvm": 3, "lt_framework": 4},
    "missing_sla": {"forecast": [], "resource_plan": []}, "query_packs": [],
}


def test_readiness_items_link_to_the_settings_fields():
    items = run_js(PROJECTS, f"{PROJECTS_INTERNALS}.readinessItems({{}})")
    assert [(i["key"], i["ok"], i["link"]) for i in items] == [
        ("target_rps", False, "sla:sla-val-target_rps"),
        ("performance_query", False, "sla:sla-query-target_rps"),
        ("system_context", False, "context:contextForm"),
        ("queries", False, "queries:addFromPackBtn"),
        ("forecast", True, "sla:sla-peak-mean_latency_query"),
        ("resource_plan", True, "sla:sla-peak-service_cpu_query"),
    ]
    full = run_js(PROJECTS, f"{PROJECTS_INTERNALS}.readinessItems({_js(FULL)})")
    assert all(i["ok"] for i in full) and full[0]["text"] == "Целевой RPS: 250" and full[3]["text"] == "Запросов: 7, доменов: 2"


def test_missing_sla_reference_links_to_its_own_field():
    readiness = dict(FULL, missing_sla={"forecast": ["max_performance_query", "mean_latency_query"], "resource_plan": ["cpu_query"]})
    items = run_js(PROJECTS, f"{PROJECTS_INTERNALS}.readinessItems({_js(readiness)})")
    forecast, plan = items[4], items[5]
    assert not forecast["ok"] and forecast["link"] == "sla:sla-query-target_rps"
    assert forecast["text"] == "Прогноз: не заданы макс. RPS, среднее время"
    assert not plan["ok"] and plan["link"] == "sla:sla-query-max_cpu_pct"


def test_readiness_list_markup():
    readiness = dict(FULL, target_rps=None, performance_query="", has_system_context=False, query_counts={},
                     query_packs=[{"title": "k6", "outdated": True}])
    html = run_js(PROJECTS, f"{PROJECTS_INTERNALS}.readinessList({{id: 'a\"b', readiness: {_js(readiness)}}})")
    assert "Настроено 2 из 6" in html and 'aria-valuenow="2"' in html
    # Squares instead of a bar: one per point, the points done first; a square before every point.
    assert 'class="readiness-squares" role="progressbar"' in html and "readiness-bar" not in html and "width:" not in html
    assert html.count('<i class="ok"></i>') == 2 and html.count("<i></i>") == 4
    assert html.count('<span class="sq ok" aria-hidden="true"></span>') == 2 and html.count('<span class="sq warn" aria-hidden="true"></span>') == 4
    assert 'href="#sla:sla-val-target_rps"' in html and 'data-link="queries:addFromPackBtn"' in html
    assert 'data-project="a&quot;b"' in html and 'data-action="goto"' in html
    assert html.count("readiness-item is-ok") == 2 and html.count("readiness-item is-todo") == 4
    assert "есть обновления: 1" in html
    complete = run_js(PROJECTS, f"{PROJECTS_INTERNALS}.readinessList({{id: 'p', readiness: {_js(FULL)}}})")
    assert "Настроено 6 из 6" in complete and "readiness is-complete" in complete
