# SPDX-License-Identifier: Apache-2.0
"""Stage 5 of plans/ui-plan.md in ui.js: theme modes with "system", empty states, card labels; stage 5 of
plans/instrument-ui-plan.md in charts.js and palette.js: the legend list without a separator, the highlighted step of
the load profile, the «system theme» command of the palette.

Runs under Node through tests/js_harness.py (skipped when node is absent). A setup file replaces
``window.matchMedia`` before ui.js loads, so the OS color scheme can be switched from a test.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tests.js_harness import REPO_ROOT, run_js, theme_css_vars

UI = ["static/js/ui.js"]

# A controllable prefers-color-scheme query: window.__os.setLight(bool) flips it and fires "change".
_MATCH_MEDIA = """
(function () {
  const listeners = [];
  const query = {
    matches: __LIGHT__,
    media: '(prefers-color-scheme: light)',
    addEventListener(type, cb) { if (type === 'change') listeners.push(cb); },
    removeEventListener(type, cb) { const i = listeners.indexOf(cb); if (i >= 0) listeners.splice(i, 1); }
  };
  window.matchMedia = (q) => (String(q).indexOf('prefers-color-scheme: light') >= 0
    ? query
    : { matches: false, media: String(q), addEventListener() {}, removeEventListener() {} });
  window.__os = {
    setLight(on) { query.matches = !!on; listeners.slice().forEach((cb) => cb({ matches: query.matches, media: query.media })); },
    listenerCount: () => listeners.length
  };
})();
"""


@pytest.fixture
def os_scheme(tmp_path):
    """Returns files(light) -> [setup file, ui.js] with the OS scheme initially light or dark."""

    def files(light: bool):
        path = tmp_path / f"match_media_{'light' if light else 'dark'}.js"
        path.write_text(_MATCH_MEDIA.replace("__LIGHT__", "true" if light else "false"), encoding="utf-8")
        return [str(path), *UI]

    return files


# ---- theme modes ------------------------------------------------------------------------------

def test_stored_system_resolves_from_os_on_load(os_scheme):
    result = run_js(os_scheme(True), """
        ({ attr: document.documentElement.getAttribute('data-theme'), current: LL.ui.theme.current(),
           mode: LL.ui.theme.mode(), listeners: __os.listenerCount() })
    """, local_storage={"loadlens.theme": "system"})
    assert result == {"attr": "light", "current": "light", "mode": "system", "listeners": 1}


def test_system_mode_follows_os_switch_and_restyles_charts(os_scheme):
    css = theme_css_vars()
    result = run_js(os_scheme(False), """
        const seen = [];
        LL.ui.theme.onChange((name) => seen.push(name));
        const before = [LL.ui.theme.current(), Chart.defaults.color];
        __os.setLight(true);
        const afterLight = [LL.ui.theme.current(), Chart.defaults.color, document.documentElement.getAttribute('data-theme')];
        __os.setLight(true);
        __os.setLight(false);
        ({ before, afterLight, after: LL.ui.theme.current(), seen, mode: LL.ui.theme.mode() })
    """, local_storage={"loadlens.theme": "system"}, dom_ready=True)
    assert result["before"] == ["dark", css["dark"]["--dim"]]
    # ui.js registers restyleCharts on DOMContentLoaded, so Chart.js defaults follow the OS switch.
    assert result["afterLight"] == ["light", css["light"]["--dim"], "light"]
    assert result["after"] == "dark"
    # A repeated OS event with the same scheme does not notify again.
    assert result["seen"] == ["light", "dark"]
    assert result["mode"] == "system"


def test_explicit_mode_ignores_os_switch(os_scheme):
    result = run_js(os_scheme(False), """
        const seen = [];
        LL.ui.theme.onChange((name) => seen.push(name));
        __os.setLight(true);
        [LL.ui.theme.current(), LL.ui.theme.mode(), seen]
    """, local_storage={"loadlens.theme": "dark"})
    assert result == ["dark", "dark", []]


def test_set_system_persists_mode_and_applies_resolved_theme(os_scheme):
    result = run_js(os_scheme(True), """
        const seen = [];
        LL.ui.theme.onChange((name) => seen.push(name));
        LL.ui.theme.set('system');
        const system = [LL.ui.theme.mode(), LL.ui.theme.current(), __harness.storage.get('loadlens.theme'), LL.prefs.get('theme')];
        LL.ui.theme.set('dark');
        const dark = [LL.ui.theme.mode(), LL.ui.theme.current(), __harness.storage.get('loadlens.theme')];
        __os.setLight(false);
        __os.setLight(true);
        ({ system, dark, seen, final: LL.ui.theme.current() })
    """)
    assert result["system"] == ["system", "light", "system", "system"]
    assert result["dark"] == ["dark", "dark", "dark"]
    # set() always notifies (resolved names); OS switches after an explicit choice do not.
    assert result["seen"] == ["light", "dark"]
    assert result["final"] == "dark"


def test_toggle_from_system_switches_to_explicit_opposite(os_scheme):
    result = run_js(os_scheme(True), """
        LL.ui.theme.toggle();
        [LL.ui.theme.mode(), LL.ui.theme.current(), __harness.storage.get('loadlens.theme')]
    """, local_storage={"loadlens.theme": "system"})
    assert result == ["dark", "dark", "dark"]


def test_unknown_mode_and_no_choice_follow_the_os(os_scheme):
    # "system" is the default (plans/instrument-ui-plan.md): an unknown or missing choice follows the OS.
    assert run_js(UI, "LL.ui.theme.set('purple'); [LL.ui.theme.mode(), LL.ui.theme.current()]") == ["system", "dark"]
    assert run_js(UI, "LL.ui.theme.set('purple'); [LL.ui.theme.mode(), LL.ui.theme.current()]", theme="light") == ["system", "light"]
    assert run_js(os_scheme(True), "LL.ui.theme.set('purple'); [LL.ui.theme.mode(), LL.ui.theme.current(), __harness.storage.get('loadlens.theme')]") == ["system", "light", "system"]
    # Without a stored choice the page rendered dark switches to the OS theme, and back.
    result = run_js(os_scheme(True), """
        const before = [LL.ui.theme.mode(), LL.ui.theme.current(), document.documentElement.getAttribute('data-theme')];
        __os.setLight(false);
        ({ before, after: LL.ui.theme.current(), stored: __harness.storage.get('loadlens.theme') || null })
    """)
    assert result == {"before": ["system", "light", "light"], "after": "dark", "stored": None}
    assert run_js(os_scheme(False), "[LL.ui.theme.mode(), LL.ui.theme.current()]", theme="light") == ["system", "dark"]
    assert run_js(UI, "LL.ui.theme.mode()", local_storage={"loadlens.theme": "bogus"}) == "system"


def test_theme_chosen_in_another_tab_applies(os_scheme):
    result = run_js(os_scheme(True), """
        const seen = [];
        LL.ui.theme.onChange((name) => seen.push(name));
        __harness.fireWindow('storage', { key: 'loadlens.theme', newValue: 'system' });
        const viaSystem = [LL.ui.theme.mode(), LL.ui.theme.current()];
        __harness.fireWindow('storage', { key: 'loadlens.theme', newValue: 'light' });
        const sameTheme = [LL.ui.theme.mode(), LL.ui.theme.current()];
        __harness.fireWindow('storage', { key: 'loadlens.theme', newValue: null });
        ({ viaSystem, sameTheme, cleared: [LL.ui.theme.mode(), LL.ui.theme.current()], seen })
    """, local_storage={"loadlens.theme": "dark"})
    assert result["viaSystem"] == ["system", "light"]
    assert result["sameTheme"] == ["light", "light"]
    # A cleared choice is "system" again: the OS (light here) decides.
    assert result["cleared"] == ["system", "light"]
    # Listeners hear only real changes of the resolved theme.
    assert result["seen"] == ["light"]


def test_theme_works_without_match_media():
    result = run_js(UI, """
        delete window.matchMedia;
        LL.ui.theme.set('system');
        [LL.ui.theme.mode(), LL.ui.theme.current()]
    """)
    assert result == ["system", "dark"]


# ---- removed density setting ------------------------------------------------------------------

def test_density_setting_is_gone_and_its_leftover_value_is_dropped():
    result = run_js(UI, """[
        typeof LL.ui.density,
        __harness.storage.has('loadlens.density'),
        document.documentElement.hasAttribute('data-density'),
        __harness.storage.get('loadlens.theme')
    ]""", local_storage={"loadlens.density": json.dumps("compact"), "loadlens.theme": "light"})
    assert result == ["undefined", False, False, "light"]
    # Blocked storage: the cleanup must not throw.
    assert run_js(UI, "typeof LL.ui.theme.current()", storage_throws=True) == "string"


# ---- empty states -----------------------------------------------------------------------------

def test_empty_row_with_icon():
    html = run_js(UI, """LL.ui.emptyRow(6, 'Нет <отчётов>', { icon: 'trending', actionHtml: '<a class="btn" href="/new">Новый отчёт</a>' })""")
    assert html == (
        '<tr class="empty-row"><td colspan="6"><div class="empty-row-body">'
        '<svg class="icon empty-icon" aria-hidden="true"><use href="#i-trending"/></svg>'
        '<span class="empty-row-text">Нет &lt;отчётов&gt;</span>'
        '<span class="empty-row-action"><a class="btn" href="/new">Новый отчёт</a></span>'
        '</div></td></tr>'
    )
    bare = run_js(UI, "LL.ui.emptyRow(2, 'x', { icon: 'col\"umns<' })")
    assert '<use href="#i-columns"/>' in bare
    assert "empty-row-action" not in bare


def test_icon_and_empty_state_helpers():
    assert run_js(UI, "LL.ui.icon('')") == ""
    assert run_js(UI, "LL.ui.icon('x', 'a b\"')") == '<svg class="icon a b" aria-hidden="true"><use href="#i-x"/></svg>'
    html = run_js(UI, """LL.ui.emptyState({ icon: 'columns', title: 'A & B', text: '<i>t</i>', actionHtml: '<a class="btn" href="/reports">Архив</a>' })""")
    assert html == (
        '<div class="empty-state"><svg class="icon empty-icon" aria-hidden="true"><use href="#i-columns"/></svg>'
        '<h3>A &amp; B</h3><p class="empty-state-text">&lt;i&gt;t&lt;/i&gt;</p>'
        '<div class="empty-state-action"><a class="btn" href="/reports">Архив</a></div></div>'
    )
    assert run_js(UI, "LL.ui.emptyState({ text: 'only' })") == '<div class="empty-state"><p class="empty-state-text">only</p></div>'


# ---- responsive table labels ------------------------------------------------------------------

# Minimal table objects on top of the harness elements: rows/cells arrays and per-cell selectors.
_TABLE_KIT = """
function th(text, span) { const el = document.createElement('th'); el.textContent = text; el.colSpan = span || 1; return el; }
function td(text, opts) {
  const o = opts || {};
  const el = document.createElement('td');
  el.textContent = text;
  el.colSpan = o.colspan || 1;
  if (o.className) el.className = o.className;
  const inputs = o.inputs || [];
  el.querySelectorAll = (sel) => (sel === 'input' ? inputs : []);
  el.querySelector = () => (o.button ? {} : null);
  return el;
}
function tr(cells, className) { const el = document.createElement('tr'); if (className) el.className = className; el.cells = cells; return el; }
function table(headCells, rows) { return { tHead: { rows: [tr(headCells)] }, tBodies: [{ rows }] }; }
function dump(rows) {
  return rows.map((row) => ({ check: row.getAttribute('data-has-check'),
    cells: row.cells.map((c) => [c.getAttribute('data-label'), c.getAttribute('data-cell')]) }));
}
"""


def test_label_table_marks_labels_title_actions_and_check():
    result = run_js(UI, _TABLE_KIT + """
        const rows = [
          tr([td('', { inputs: [{ type: 'checkbox' }] }), td('run-1'), td('svc'), td('Успешно'), td('Открыть', { className: 'actions', button: true })], 'data-row'),
          tr([td('', { inputs: [{ type: 'checkbox' }, { type: 'checkbox' }] }), td('run-2'), td('svc'), td('Провал'), td('')]),
          tr([td('Пусто', { colspan: 5 })], 'empty-row'),
          tr([td(''), td(''), td(''), td(''), td('')], 'skeleton-row')
        ];
        const t = table([th(''), th('  Запуск\\n '), th('Сервис'), th('Вердикт'), th('Действия')], rows);
        LL.ui.labelTable(t);
        LL.ui.labelTable(t);
        dump(rows)
    """)
    assert result[0] == {"check": "1", "cells": [
        [None, "check"], ["Запуск", "primary"], ["Сервис", None], ["Вердикт", None], [None, "actions"],
    ]}
    # Two checkboxes are not a selection cell: the first cell becomes the card title (its column has no
    # title, so no label); a last cell without buttons keeps its label.
    assert result[1] == {"check": None, "cells": [
        [None, "primary"], ["Запуск", None], ["Сервис", None], ["Вердикт", None], ["Действия", None],
    ]}
    # Empty and loading rows stay unlabeled.
    assert result[2] == {"check": None, "cells": [[None, None]]}
    assert result[3] == {"check": None, "cells": [[None, None]] * 5}


def test_label_table_columns_follow_header_positions():
    result = run_js(UI, _TABLE_KIT + """
        const rows = [tr([td('a'), td('b1'), td('b2'), td('c')]), tr([td('x'), td('wide', { colspan: 2 }), td('z')])];
        const t = table([th('A'), th('B', 2), th('C')], rows);
        LL.ui.labelTable(t);
        dump(rows)
    """)
    assert [c[0] for c in result[0]["cells"]] == ["A", "B", "B", "C"]
    assert result[0]["cells"][0][1] == "primary"
    # A spanning cell gets no label; the next cell still lines up with its column.
    assert [c[0] for c in result[1]["cells"]] == ["A", None, "C"]


def test_label_table_updates_when_header_or_row_changes():
    result = run_js(UI, _TABLE_KIT + """
        const first = td('m');
        const value = td('1.0');
        const row = tr([first, value]);
        const head = [th('Метрика'), th('A (p95)')];
        const t = table(head, [row]);
        LL.ui.labelTable(t);
        const before = value.getAttribute('data-label');
        head[1].textContent = 'A (среднее)';
        LL.ui.labelTable(t);
        const afterHead = value.getAttribute('data-label');
        value.colSpan = 2;
        LL.ui.labelTable(t);
        [before, afterHead, value.hasAttribute('data-label'), first.getAttribute('data-cell')]
    """)
    assert result == ["A (p95)", "A (среднее)", False, "primary"]


def test_label_table_ignores_tables_without_header():
    assert run_js(UI, "LL.ui.labelTable({ tHead: null, tBodies: [] }); LL.ui.labelTable(null); 'ok'") == "ok"


def test_dom_ready_without_mutation_observer_or_menu():
    # The harness has no MutationObserver and no #viewMenu: wiring must be a no-op, not an error.
    assert run_js(UI, "LL.ui.theme.set('light'); LL.ui.theme.current()", dom_ready=True) == "light"


# ---- LL.charts.legendChips: the legend list of the report -----------------------------------------------------------

CHARTS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js"]

_LEGEND = r"""(() => {
  const chart = {
    canvas: { width: 400, height: 200 }, ctx: {}, width: 400, currentDevicePixelRatio: 1,
    data: { labels: [1, 2], datasets: [{ label: 'api', data: [10, 30], borderColor: '#111111' }, { label: 'db', data: [5, 7] }] },
    isDatasetVisible: () => true, setDatasetVisibility() {}, update() {},
  };
  const values = (host) => {
    const list = host.children[0];
    return list.children.filter((el) => el.className === 'legend-chip').map((chip) => chip.children[2].textContent);
  };
  const live = (host, api) => {
    // The harness DOM has no selectors: give the list and the chips the lookups showIndex uses.
    const list = host.children[0];
    const chips = list.children.filter((el) => el.className === 'legend-chip');
    list.querySelectorAll = () => chips;
    chips.forEach((chip) => { chip.querySelector = () => chip.children[2]; });
    api.showIndex(1);
    const shown = values(host);
    api.showIndex(-1);
    return [shown, values(host)];
  };
  const plainHost = document.createElement('div');
  const plain = LL.charts.legendChips(chart, plainHost, { separator: '' });
  const dotHost = document.createElement('div');
  const dotted = LL.charts.legendChips(Object.assign({}, chart), dotHost);
  return { plain: values(plainHost), plainLive: live(plainHost, plain), dotted: values(dotHost), dottedLive: live(dotHost, dotted) };
})()"""


def test_legend_chips_render_the_value_without_a_separator():
    out = run_js(CHARTS, _LEGEND, locale="en")
    # The report legend is a list: dash, name, value (no "· " between the name and the value).
    assert out["plain"] == ["20", "6"]
    assert out["plainLive"] == [["30", "7"], ["20", "6"]]
    # Other callers keep the chip text with its separator.
    assert out["dotted"] == ["· 20", "· 6"]
    assert out["dottedLive"] == [["· 30", "· 7"], ["· 20", "· 6"]]


def test_report_and_compare_use_the_legend_list():
    reports = (REPO_ROOT / "static" / "js" / "reports.js").read_text(encoding="utf-8")
    assert "maxVisible: MAX_VISIBLE_SERIES, separator: '' }" in reports
    compare = (REPO_ROOT / "static" / "js" / "compare.js").read_text(encoding="utf-8")
    assert "filename: `compare-${queryLabel}`, separator: '' }" in compare


# ---- LL.charts.stepProfile: the highlighted step ---------------------------------------------------------------------

_MSK = timezone(timedelta(hours=3))
_P0 = datetime(2026, 10, 15, 19, 10, tzinfo=_MSK)


def _steps(levels, minutes=10):
    out = []
    for i, rps in enumerate(levels):
        start = _P0 + timedelta(minutes=minutes * i)
        out.append({"index": i + 1, "rps": rps, "selected": i == len(levels) - 1,
                    "start_iso": start.isoformat(), "end_iso": (start + timedelta(minutes=minutes)).isoformat()})
    return out


def test_step_profile_model_highlights_a_step_by_its_index():
    steps = _steps([50, 100, 150, 200])
    model = run_js(CHARTS, f"""[2, '3', 9, null, {{ t: 1 }}].map((h) => LL.charts.stepProfileModel({{ steps: {json.dumps(steps)}, width: 400, compact: true, highlight: h }}).highlight)""")
    second, third, unknown, none, range_ = model
    assert second == {"index": 2, "x": 100, "w": 100, "y": second["y"], "path": f"M100 {second['y']}H200"}
    assert third["index"] == 3 and third["x"] == 200 and third["w"] == 100
    assert unknown is None and none is None and range_ is None
    assert run_js(CHARTS, f"LL.charts.stepProfileModel({{ steps: {json.dumps(steps)}, width: 400 }}).highlight") is None


def test_step_profile_handle_highlight_redraws_and_survives_updates():
    steps = _steps([50, 100, 150, 200])
    result = run_js(CHARTS, f"""
        const host = document.createElement('div');
        const handle = LL.charts.stepProfile(host, {{ steps: {json.dumps(steps)}, width: 400, compact: true }});
        const before = host.innerHTML;
        const same = handle.highlight(2) === handle;
        const lit = host.innerHTML;
        const model = handle.model.highlight;
        handle.update({{ target: 180 }});
        const kept = handle.model.highlight && handle.model.highlight.index;
        handle.highlight(null);
        const cleared = host.innerHTML;
        handle.highlight(42);
        ({{ before, same, lit, model, kept, cleared, unknown: handle.model.highlight, emptyOk: LL.charts.stepProfile(document.createElement('div'), {{ steps: [] }}).highlight(1).model }})
    """, locale="en")
    assert 'class="pf-hl"' not in result["before"] and 'class="pf-hl-path"' not in result["before"]
    assert result["same"] is True
    lit = result["lit"]
    assert '<rect class="pf-hl" x="100" y="8" width="100"' in lit and f'<path class="pf-hl-path" d="{result["model"]["path"]}"/>' in lit
    # The shade is drawn under the staircase, the level of the step over it.
    assert lit.index('class="pf-hl"') < lit.index('class="pf-path"') < lit.index('class="pf-hl-path"')
    assert 'class="pf-level is-highlighted"' in lit
    assert result["kept"] == 2
    assert "pf-hl" not in result["cleared"] and "is-highlighted" not in result["cleared"]
    assert result["unknown"] is None and result["emptyOk"] is None


def test_report_step_cells_carry_the_step_index_and_highlight_the_profile():
    source = (REPO_ROOT / "static" / "js" / "reports.js").read_text(encoding="utf-8")
    assert 'data-step-index="${escapeHtml(String(step.index))}"' in source
    assert "profile.highlight(stepIndex)" in source and "highlightProfile(null)" in source
    # The step highlight on the charts is neutral; the accent marks the SLA zone only.
    assert "tone: 'muted' }" in source and "dataset.stepTo), tone: 'accent'" not in source


# ---- palette: «Тема как в системе» ---------------------------------------------------------------------------------

PALETTE = ["static/js/common.js", "static/js/ui.js", "static/js/palette.js"]


@pytest.mark.parametrize("stored, offered", [(None, False), ("system", False), ("light", True), ("dark", True), ("purple", False)])
def test_palette_offers_the_system_theme_unless_the_mode_is_system(os_scheme, stored, offered):
    storage = {} if stored is None else {"loadlens.theme": stored}
    files = [*os_scheme(False), "static/js/common.js", "static/js/palette.js"]
    result = run_js(files, """
        const P = LL.palette._internals;
        const keys = () => P.actionItems(P.themeState()).map((a) => a.key);
        const first = { state: P.themeState(), keys: keys() };
        LL.ui.theme.set('light');
        const light = keys();
        LL.ui.theme.set('system');
        ({ first, light, system: keys() })
    """, local_storage=storage)
    assert ("action:theme-system" in result["first"]["keys"]) is offered
    expected_mode = stored if stored in ("system", "light", "dark") else "system"
    assert result["first"]["state"]["themeMode"] == expected_mode
    # An explicit choice made on the page offers the way back; choosing «system» removes the command.
    assert "action:theme-system" in result["light"] and "action:theme-system" not in result["system"]


def test_palette_without_ui_js_offers_the_system_theme():
    state = run_js(["static/js/common.js", "static/js/palette.js"], "LL.palette._internals.themeState()")
    assert state == {"theme": "dark", "themeMode": ""}
    keys = run_js(["static/js/common.js", "static/js/palette.js"], "LL.palette._internals.actionItems(LL.palette._internals.themeState()).map((a) => a.key)")
    assert keys == ["action:theme", "action:theme-system"]


# ---- forecast report: labels of the load scale stay inside it --------------------------------------------------------

_FIT = """
const mk = (cls, rect, span) => {
  const set = new Set(cls.split(' '));
  return {
    classList: { add: (...c) => c.forEach((x) => set.add(x)), remove: (...c) => c.forEach((x) => set.delete(x)), contains: (c) => set.has(c) },
    getBoundingClientRect: () => rect, querySelector: () => span || null, cls: () => Array.from(set).sort().join(' '),
  };
};
const marks = [
  mk('forecast-scale-mark', { left: 300, right: 420 }),
  mk('forecast-scale-mark', { left: -20, right: 60 }),
  mk('forecast-scale-mark', { left: 100, right: 200 }),
  mk('forecast-scale-mark is-end', { left: 300, right: 420 }),
  mk('forecast-scale-target', { left: 0, right: 0 }, { getBoundingClientRect: () => ({ left: 330, right: 410 }) }),
  mk('forecast-scale-mark fit-end', { left: 100, right: 200 }),
];
const scale = { getBoundingClientRect: () => ({ left: 0, right: 362, width: 362 }), querySelectorAll: () => marks };
LL.forecastReport._internals.fitScaleLabels({ querySelector: () => scale });
LL.forecastReport._internals.fitScaleLabels(null);
marks.map((m) => m.cls())
"""


def test_forecast_scale_labels_are_anchored_to_the_edge_they_would_cross():
    files = ["static/js/common.js", "static/js/ui.js", "static/js/forecast_report.js"]
    assert run_js(files, _FIT) == [
        "fit-end forecast-scale-mark",  # would cross the right edge of a 362px scale
        "fit-start forecast-scale-mark",  # would cross the left edge
        "forecast-scale-mark",  # fits centered
        "forecast-scale-mark is-end",  # already anchored by the page
        "fit-end forecast-scale-target",  # the target label turns to the left of its line
        "forecast-scale-mark",  # a stale anchor goes once the label fits (a wider window)
    ]
