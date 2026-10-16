"""Stage 5 of docs/ui-plan.md in ui.js: theme modes with "system", density, empty states, card labels.

Runs under Node through tests/js_harness.py (skipped when node is absent). A setup file replaces
``window.matchMedia`` before ui.js loads, so the OS color scheme can be switched from a test.
"""

from __future__ import annotations

import json

import pytest

from tests.js_harness import run_js, theme_css_vars

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
    assert result["before"] == ["dark", css["dark"]["--chart-text"]]
    # ui.js registers restyleCharts on DOMContentLoaded, so Chart.js defaults follow the OS switch.
    assert result["afterLight"] == ["light", css["light"]["--chart-text"], "light"]
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


def test_unknown_mode_falls_back_to_dark_and_no_choice_keeps_page_theme():
    assert run_js(UI, "LL.ui.theme.set('purple'); [LL.ui.theme.mode(), LL.ui.theme.current()]", theme="light") == ["dark", "dark"]
    # Without a stored choice the mode is the theme the page was rendered with.
    assert run_js(UI, "[LL.ui.theme.mode(), LL.ui.theme.current()]", theme="light") == ["light", "light"]
    assert run_js(UI, "LL.ui.theme.mode()", local_storage={"loadlens.theme": "bogus"}) == "dark"


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
    assert result["cleared"] == ["dark", "dark"]
    # Listeners hear only real changes of the resolved theme.
    assert result["seen"] == ["light", "dark"]


def test_theme_works_without_match_media():
    result = run_js(UI, """
        delete window.matchMedia;
        LL.ui.theme.set('system');
        [LL.ui.theme.mode(), LL.ui.theme.current()]
    """)
    assert result == ["system", "dark"]


# ---- density ----------------------------------------------------------------------------------

def test_density_set_applies_attribute_and_persists():
    result = run_js(UI, """
        const start = LL.ui.density.current();
        LL.ui.density.set('compact');
        const compact = [LL.ui.density.current(), document.documentElement.getAttribute('data-density'), __harness.storage.get('loadlens.density')];
        LL.ui.density.set('weird');
        ({ start, compact, normal: [LL.ui.density.current(), document.documentElement.hasAttribute('data-density'), __harness.storage.get('loadlens.density')] })
    """)
    assert result["start"] == "normal"
    # The JSON-encoded value is what the pre-paint scripts compare with ('"compact"').
    assert result["compact"] == ["compact", "compact", json.dumps("compact")]
    assert result["normal"] == ["normal", False, json.dumps("normal")]


def test_density_from_another_tab_and_without_storage():
    result = run_js(UI, """
        __harness.fireWindow('storage', { key: 'loadlens.density', newValue: '"compact"' });
        const a = LL.ui.density.current();
        __harness.fireWindow('storage', { key: 'loadlens.density', newValue: null });
        [a, LL.ui.density.current()]
    """)
    assert result == ["compact", "normal"]
    assert run_js(UI, "LL.ui.density.set('compact'); LL.ui.density.current()", storage_throws=True) == "compact"


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
