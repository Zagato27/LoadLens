# SPDX-License-Identifier: Apache-2.0
"""Release marks in the UI: the Releases tab of the archive, marks and dialogs (archive.js, reports.js), the baseline
buttons of the compare page (compare.js), the release field of the launch form (launch.js) and the page markup.
Pure functions run under Node (tests/js_harness.py)."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: F401,E402  (before anything that imports update_page)
from tests.js_harness import run_js  # noqa: E402

BASE = ["static/js/common.js", "static/js/ui.js"]
ARCHIVE = BASE + ["static/js/archive.js"]
REPORT = BASE + ["static/js/charts.js", "static/js/reports.js"]
COMPARE = BASE + ["static/js/charts.js", "static/js/compare.js"]
LAUNCH = BASE + ["static/js/launch.js"]


def js(value):
    return json.dumps(value, ensure_ascii=False)


def archive(expression, **kwargs):
    return run_js(ARCHIVE, f"const A = LL.archive._internals; {expression}", **kwargs)


# ---- archive ------------------------------------------------------------------------------------------------


def test_the_releases_tab_comes_from_the_address_and_sorts_by_release():
    assert archive("[A.tabFromQuery('?tab=releases'), A.tabFromQuery('?tab=runs'), A.tabFromQuery('?tab=x')]") == [
        "releases", "runs", "reports",
    ]
    state = archive("A.parseQuery('?tab=releases&release=%2025.11%20&service=checkout')")
    assert (state["tab"], state["release"], state["sort"], state["dir"], state["service"]) == (
        "releases", "25.11", "release", "desc", "checkout",
    )
    # The default sort of the tab is not written; the release filter is.
    assert archive("A.buildQuery(A.parseQuery('?tab=releases&release=25.11'))") == "?tab=releases&release=25.11"
    assert archive("A.buildQuery(A.defaultState('releases'))") == "?tab=releases"
    assert archive("A.buildQuery({ ...A.defaultState('releases'), sort: 'end_time', dir: 'desc' })") == "?tab=releases&sort=end_time"
    # A release in the address of the reports tab is ignored.
    assert archive("A.parseQuery('?release=25.11').release") == ""
    assert archive("A.parseQuery('?tab=releases&release=' + 'x'.repeat(200)).release") == "x" * 80


def test_the_runs_url_of_the_releases_tab_asks_for_marked_runs():
    url = archive("A.buildRunsUrl(A.defaultState('releases'))")
    assert url == "/runs?release=*&offset=0&limit=20&sort=release&dir=desc&include=sla"
    url = archive("A.buildRunsUrl({ ...A.defaultState('releases'), release: '25.11 RC' })")
    assert "release=25.11+RC" in url
    assert "release" not in archive("A.buildRunsUrl(A.defaultState())")


def test_the_release_filter_is_a_chip_of_the_releases_tab():
    chips = archive("A.activeChips({ ...A.defaultState('releases'), release: '25.11' })")
    assert chips == [{"key": "release", "label": "Релиз", "value": "25.11"}]
    assert archive("A.hasFilters({ ...A.defaultState('releases'), release: '25.11' })") is True
    assert archive("A.hasFilters(A.defaultState('releases'))") is False
    assert archive("A.activeChips({ ...A.defaultState(), release: '25.11' })") == []


def test_release_pills_for_the_three_states():
    assert archive("A.releasePills({ run_name: 'r' })") == []
    marked = archive("A.releasePills({ release: '25.11', release_accepted: false, baseline: false })")
    assert marked == [{"kind": "release", "text": "25.11", "title": "Релиз 25.11"}]
    accepted = archive("A.releasePills({ release: '25.11', release_accepted: true, baseline: true })")
    assert [(p["kind"], p["text"]) for p in accepted] == [("accepted", "25.11"), ("baseline", "эталон")]
    assert accepted[0]["title"] == "Принятый прогон релиза 25.11"
    baseline = archive("A.releasePills({ release: '', baseline: true })")
    assert [p["kind"] for p in baseline] == ["baseline"]


def test_compare_with_the_previous_release_and_the_dialog():
    assert archive("A.compareWithPreviousRelease({ run_name: 'night 1/2' })") == \
        "/compare?run_a=night%201%2F2&baseline=previous_release"
    html = archive("A.releaseDialogHtml({ release: '25.11', accepted: true, accepted_manual: true, note: '<b>', "
                   "releases: ['25.10', '25\"11'], name_max: 80, note_max: 1000 })")
    assert 'name="releaseName" maxlength="80"' in html and 'value="25.11"' in html
    assert '<option value="25&quot;11"></option>' in html and "&lt;b&gt;</textarea>" in html
    assert 'name="releaseAccepted" checked' in html and "принятый прогон релиза 25.11" in html
    empty = archive("A.releaseDialogHtml({})")
    assert 'value=""' in empty and "checked" not in empty and "не помечен" in empty
    assert archive("[A.releaseFormProblem(' '), A.releaseFormProblem('x'.repeat(81)), A.releaseFormProblem('25.11')]") == [
        "Укажите имя релиза", "Имя релиза — не длиннее 80 символов", "",
    ]


def test_archive_strings_in_english():
    pills = archive("A.releasePills({ release: '25.11', release_accepted: true, baseline: true })", locale="en")
    assert [p["text"] for p in pills] == ["25.11", "baseline"] and pills[0]["title"] == "Accepted run of release 25.11"
    assert "Accept by hand" in archive("A.releaseDialogHtml({})", locale="en")


# ---- report page --------------------------------------------------------------------------------------------


def report(expression, **kwargs):
    return run_js(REPORT, f"const I = LL.report._internals; {expression}", **kwargs)


def test_the_release_line_of_the_report():
    assert report("I.releaseLineHtml(null)") == '<div class="hero-sub rep-release" data-release-line hidden></div>'
    line = report("I.releaseLineHtml({ release: '25.11', accepted: true, accepted_manual: false, baseline: true, "
                  "author: 'engineer1', updated_at: '2026-11-01T12:00:00Z' })")
    assert "Релиз 25.11" in line and "принятый прогон" in line and "эталон сервиса" in line
    assert 'data-release-action="edit"' in line and 'data-release-action="unpin"' in line
    assert 'title="Изменено: engineer1' in line and 'data-min-role="engineer"' in line
    # Nothing marked: only the buttons, for engineers.
    bare = report("I.releaseLineHtml({ release: '', accepted: false, baseline: false })")
    assert bare.startswith('<div class="hero-sub rep-release" data-release-line data-min-role="engineer">')
    assert 'data-release-action="pin"' in bare and "Релиз " not in bare.split("<span")[0]
    manual = report("I.releaseLineHtml({ release: '25.11', accepted: true, accepted_manual: true })")
    assert "принятый прогон (вручную)" in manual
    dialog = report("I.releaseDialogHtml({ release: '25.11', releases: ['25.10'], accepted_manual: false })")
    assert 'value="25.11"' in dialog and '<option value="25.10"></option>' in dialog and "checked" not in dialog


def test_the_release_line_sits_right_under_the_verdict_word():
    source = (ROOT_DIR / "static" / "js" / "reports.js").read_text(encoding="utf-8")
    hero = source[source.index("function heroHtml("):source.index("function renderHero(")]
    word = hero.index("report.hero.no_ai_tag")
    assert hero.index("releaseLineHtml(reportState.mark)") > word
    assert hero.index("releaseLineHtml(reportState.mark)") < hero.index("summary ? `<p class=\"rep-verdict-why\">")


# ---- compare page -------------------------------------------------------------------------------------------


def compare(expression, **kwargs):
    return run_js(COMPARE, f"const C = LL.compare._internals; {expression}", **kwargs)


def test_compare_reads_the_baseline_mode_of_the_link_and_names_the_found_run():
    assert compare("[C.baselineFromQuery('?run_a=x&baseline=previous_release'), C.baselineFromQuery('?baseline=PINNED'), "
                   "C.baselineFromQuery('?baseline=best'), C.baselineFromQuery('')]") == ["previous_release", "pinned", "", ""]
    notes = compare("[C.baselineNoteText('pinned', { mode: 'pinned' }), "
                    "C.baselineNoteText('pinned', { mode: 'previous_release', release: '25.10' }), "
                    "C.baselineNoteText('previous_release', { mode: 'previous_success' }), "
                    "C.baselineNoteText('previous_success', { mode: 'previous_success' }), "
                    "C.baselineNoteText('previous', {})]")
    assert notes == [
        "B: закреплённый эталон сервиса — можно изменить.",
        "B: принятый прогон релиза 25.10 — можно изменить.",
        "B: эталона и принятых прогонов прошлых релизов нет, взят предыдущий успешный прогон — можно изменить.",
        "B: предыдущий успешный прогон сервиса — можно изменить.",
        "B: предыдущий прогон сервиса — можно изменить.",
    ]


# ---- launch form --------------------------------------------------------------------------------------------


def test_changing_compare_selection_invalidates_results_and_actions():
    result = run_js(COMPARE, """
      (async () => {
        const C = LL.compare._internals;
        const nodes = {};
        document.getElementById = (id) => nodes[id] || (nodes[id] = document.createElement('div'));
        const requests = [];
        window.fetch = async (url, options) => {
          requests.push({ url: String(url), method: (options || {}).method || 'GET' });
          const body = String(url).startsWith('/llm_reports') ? []
            : String(url).startsWith('/compare_explanation') ? { status: 'unavailable' } : {};
          return { ok: true, json: async () => body };
        };
        C.setRun('runA', { value: 'a1', meta: '' });
        C.setRun('runB', { value: 'b1', meta: '' });
        await C.runCompare();
        const visibleBefore = !nodes.verdictCompare.hidden && !nodes.metricsCompare.hidden;
        C.setRun('runA', { value: 'a2', meta: '' });
        await C.runExplanation();
        await C.publishComparison();
        return { visibleBefore, hidden: ['verdictCompare', 'phasesCompare', 'compareExplain', 'metricsCompare']
          .every((id) => nodes[id].hidden), posts: requests.filter((r) => r.method === 'POST') };
      })()
    """)
    assert result == {"visibleBefore": True, "hidden": True, "posts": []}


def test_compare_response_does_not_restore_a_replaced_selection():
    result = run_js(COMPARE, """
      (async () => {
        const C = LL.compare._internals;
        const nodes = {};
        document.getElementById = (id) => nodes[id] || (nodes[id] = document.createElement('div'));
        let resolveRows;
        const rows = new Promise((resolve) => { resolveRows = resolve; });
        window.fetch = async () => ({ ok: true, json: async () => rows });
        C.setRun('runA', { value: 'a1', meta: '' });
        C.setRun('runB', { value: 'b1', meta: '' });
        const pending = C.runCompare();
        C.setRun('runA', { value: 'a2', meta: '' });
        resolveRows([]);
        await pending;
        return { hidden: nodes.verdictCompare.hidden && nodes.metricsCompare.hidden,
          canCompare: !nodes.compareBtn.disabled };
      })()
    """)
    assert result == {"hidden": True, "canCompare": True}


def test_repeat_copies_the_release_into_the_form():
    spec = run_js(LAUNCH, "LL.launch._internals.runToSpec({ service: 's', test_type: 'step', release: '25.11', "
                          "options: { max_seconds: 600 }, instances: [] })")
    assert spec["release"] == "25.11" and spec["max_duration_min"] == 10
    assert run_js(LAUNCH, "LL.launch._internals.runToSpec({ service: 's', instances: [] }).release") == ""
    source = (ROOT_DIR / "static" / "js" / "launch.js").read_text(encoding="utf-8")
    assert "release: $('release').value.trim()," in source
    # The draft keeps the release (only run_name, env and secrets are emptied).
    draft = source[source.index("function saveDraft("):source.index("function loadDraft(")]
    assert "release" not in draft


# ---- page markup --------------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from app import create_app
    from tests.test_smoke_api import FakeConnection
    from loadlens_app.blueprints import compare, dashboard

    for module in (dashboard, compare):
        monkeypatch.setattr(module, "_ts_conn", lambda: FakeConnection())
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_pages_carry_the_release_controls(client):
    html = client.get("/reports").get_data(as_text=True)
    assert ('<button type="button" class="tab" role="tab" id="tabReleases" data-tab="releases" '
            'aria-controls="reportsPanel" aria-selected="false" tabindex="-1">Релизы</button>') in html
    tabs = re.findall(r'id="(tab\w+)" data-tab', html)
    assert tabs == ["tabReports", "tabReleases", "tabRuns"]
    assert '<label class="inp archive-filter" id="releaseFilterBox" hidden>' in html and 'id="releaseFilter"' in html
    compare_html = client.get("/compare").get_data(as_text=True)
    buttons = re.findall(r'id="(baseline\w+Btn)"', compare_html)
    assert buttons == ["baselinePinnedBtn", "baselineReleaseBtn", "baselineSuccessBtn", "baselinePrevBtn"]
    launch_html = client.get("/launch").get_data(as_text=True)
    assert '<input type="text" id="release" maxlength="80"' in launch_html


def test_users_page_names_the_audit_actions():
    source = (ROOT_DIR / "static" / "js" / "users.js").read_text(encoding="utf-8")
    for action, key in (("report.release", "report_release"), ("report.release_reset", "report_release_reset"),
                        ("report.baseline", "report_baseline"), ("report.baseline_reset", "report_baseline_reset")):
        assert f"'{action}': LL.t('users.audit_action.{key}')" in source
    for lang in ("ru", "en"):
        users = json.loads((ROOT_DIR / "locales" / lang / "users.json").read_text(encoding="utf-8"))
        assert users["users.audit_action.report_baseline"]
