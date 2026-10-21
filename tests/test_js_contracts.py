# SPDX-License-Identifier: Apache-2.0
"""Contracts of the shared front-end helpers (static/js/common.js, ui.js, charts.js), run under Node."""

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from tests.js_harness import REPO_ROOT, JsHarnessError, run_js, theme_css_vars

COMMON = ["static/js/common.js"]
UI = ["static/js/common.js", "static/js/ui.js"]
CHARTS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js"]

MINUTES = json.dumps([f"2026-10-15T10:0{i}:00Z" for i in range(4)])
T0 = 1792058400000  # 2026-10-15T10:00:00Z in epoch ms


def js_str(value):
    return json.dumps(value, ensure_ascii=False)


# ---- LL.charts.indexOfTime ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("'2026-10-15T10:02:00Z'", 2),          # exact ISO
        (str(T0 + 60_000), 1),                  # exact epoch ms
        ("'2026-10-15T10:01:20Z'", 1),          # nearest below
        ("'2026-10-15T10:01:40Z'", 2),          # nearest above
        ("'2026-10-15T10:01:30Z'", 1),          # a tie goes to the earlier label
        ("'2026-10-15T09:59:30Z'", 0),          # first - halfStep is still inside
        ("'2026-10-15T09:59:29Z'", -1),         # before first - halfStep
        ("'2026-10-15T10:03:30Z'", 3),          # last + halfStep is still inside
        ("'2026-10-15T10:03:31Z'", -1),         # after last + halfStep
        ("'not a date'", -1),
        ("null", -1),
    ],
)
def test_index_of_time_iso_labels(value, expected):
    assert run_js(CHARTS, f"LL.charts.indexOfTime({MINUTES}, {value})") == expected


def test_index_of_time_epoch_labels_and_iso_value():
    labels = json.dumps([T0 + i * 60_000 for i in range(5)])
    assert run_js(CHARTS, f"LL.charts.indexOfTime({labels}, '2026-10-15T10:03:10Z')") == 3
    assert run_js(CHARTS, f"LL.charts.indexOfTime({labels}, new Date({T0 + 4 * 60_000}))") == 4


def test_index_of_time_empty_and_non_time_labels():
    assert run_js(CHARTS, f"LL.charts.indexOfTime([], {T0})") == -1
    assert run_js(CHARTS, f"LL.charts.indexOfTime(['a', 'b'], {T0})") == -1
    assert run_js(CHARTS, f"LL.charts.indexOfTime(null, {T0})") == -1


def test_index_of_time_uses_median_spacing_across_gaps():
    labels = json.dumps([T0, T0 + 60_000, T0 + 120_000, T0 + 600_000])
    # Median step is one minute even with a gap, so the half step stays 30 s.
    assert run_js(CHARTS, f"LL.charts.indexOfTime({labels}, {T0 + 630_000})") == 3
    assert run_js(CHARTS, f"LL.charts.indexOfTime({labels}, {T0 + 631_000})") == -1
    # Inside the gap the nearest label wins.
    assert run_js(CHARTS, f"LL.charts.indexOfTime({labels}, {T0 + 400_000})") == 3
    assert run_js(CHARTS, f"LL.charts.indexOfTime({labels}, {T0 + 300_000})") == 2


def test_index_of_time_single_label():
    assert run_js(CHARTS, f"[LL.charts.indexOfTime([{T0}], {T0}), LL.charts.indexOfTime([{T0}], {T0 + 1})]") == [0, -1]


# ---- LL.charts.palette / paletteFor ------------------------------------------------------

HEX = re.compile(r"^#[0-9a-f]{6}$")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_palette_reads_series_vars_of_the_theme(theme):
    css = theme_css_vars()[theme]
    expected = [css[f"--series-{i}"] for i in range(1, 9)]
    colors = run_js(CHARTS, "Array.from({ length: 8 }, (_, i) => LL.charts.palette(i))", theme=theme)
    assert colors == expected
    assert len(set(colors)) == 8
    assert all(HEX.match(c) for c in colors)
    status = {css["--status-ok"], css["--status-warn"], css["--status-fail"], css["--accent"]}
    assert not status & set(colors)


def test_palette_differs_between_themes():
    css = theme_css_vars()
    dark = [css["dark"][f"--series-{i}"] for i in range(1, 9)]
    light = [css["light"][f"--series-{i}"] for i in range(1, 9)]
    assert dark != light


def test_palette_follows_theme_at_call_time():
    css = theme_css_vars()
    before, after = run_js(CHARTS, "const a = LL.charts.palette(0); LL.ui.theme.set('light'); [a, LL.charts.palette(0)]")
    assert before == css["dark"]["--series-1"]
    assert after == css["light"]["--series-1"]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_palette_cycles_with_distinct_lightness(theme):
    colors = run_js(CHARTS, "Array.from({ length: 32 }, (_, i) => LL.charts.palette(i))", theme=theme)
    assert all(HEX.match(c) for c in colors)
    assert len(set(colors)) == 32
    assert colors[8:16] != colors[:8]
    assert run_js(CHARTS, "[LL.charts.palette(9), LL.charts.palette(9)]", theme=theme) == [colors[9], colors[9]]
    assert run_js(CHARTS, "LL.charts.palette(-3)", theme=theme) == colors[0]


def test_palette_falls_back_without_css():
    colors = run_js(CHARTS, "Array.from({ length: 8 }, (_, i) => LL.charts.palette(i))", css_vars={"dark": {f"--series-{i}": "" for i in range(1, 9)}})
    assert len(set(colors)) == 8


def test_palette_for_assigns_in_order_and_is_stable():
    result = run_js(CHARTS, """
        const names = ['svc-b', 'svc-a', 'svc-b', 'svc-c'];
        const one = LL.charts.paletteFor(names);
        const two = LL.charts.paletteFor(names);
        ({ one, two, size: one.size, isMap: one instanceof Map,
           slots: [LL.charts.palette(0), LL.charts.palette(1), LL.charts.palette(2)],
           swapped: LL.charts.paletteFor(['svc-a', 'svc-b']) })
    """)
    assert result["isMap"] is True
    assert result["size"] == 3
    assert result["one"] == result["two"]
    assert list(result["one"]) == ["svc-b", "svc-a", "svc-c"]
    assert list(result["one"].values()) == result["slots"]
    assert result["swapped"] == {"svc-a": result["slots"][0], "svc-b": result["slots"][1]}
    assert run_js(CHARTS, "LL.charts.paletteFor(null).size") == 0


def test_recolor_restores_palette_slots():
    result = run_js(CHARTS, """
        const chart = { data: { datasets: [{ llSeries: 2, borderColor: 'red' }, { borderColor: 'keep' }] } };
        LL.charts.recolor(chart);
        [chart.data.datasets[0].borderColor, chart.data.datasets[0].backgroundColor, chart.data.datasets[1].borderColor, LL.charts.palette(2)]
    """)
    assert result[0] == result[1] == result[3]
    assert result[2] == "keep"


def test_only_the_non_drawing_plugins_are_registered_globally():
    # Zoom selection draws only during a drag; the pointer tracker draws nothing. Everything else is per chart.
    ids = run_js(CHARTS, "__harness.registered.map((p) => p.id)")
    assert sorted(ids) == ["llPointer", "llZoomSelection"]
    assert run_js(CHARTS, "[LL.charts.markers.id, LL.charts.thresholds.id]") == ["llMarkers", "llThresholds"]


# ---- LL.charts.thresholds / markers ------------------------------------------------------

def test_thresholds_extend_the_y_scale_with_headroom():
    result = run_js(CHARTS, """
        const items = [{ value: 300, label: 'p95 ≤ 300 мс', tone: 'fail' }, { value: 'x' }];
        const chart = { config: { options: { plugins: { llThresholds: { items } } } } };
        const y = { id: 'y', min: 0, max: 100, options: {} };
        const fixed = { id: 'y', min: 0, max: 100, options: { max: 100 } };
        const x = { id: 'x', min: 0, max: 10, options: {} };
        const inside = { id: 'y', min: 0, max: 1000, options: {} };
        [y, fixed, x, inside].forEach((scale) => LL.charts.thresholds.afterDataLimits(chart, { scale }, {}));
        [y.max, fixed.max, x.max, inside.max, y.min]
    """)
    assert result[0] == pytest.approx(315)
    assert result[1:] == [100, 10, 1000, 0]


FAKE_CHART = """
    const calls = [];
    const ctx = new Proxy({}, {
      get(target, prop) {
        if (prop in target) return target[prop];
        if (prop === 'measureText') return (text) => ({ width: String(text).length * 6 });
        return (...args) => { calls.push([prop, ...args]); };
      },
      set(target, prop, value) { target[prop] = value; return true; }
    });
    const labels = ['2026-10-15T10:00:00Z', '2026-10-15T10:01:00Z', '2026-10-15T10:02:00Z', '2026-10-15T10:03:00Z'];
    const chart = {
      ctx, canvas: {}, chartArea: { left: 100, right: 400, top: 10, bottom: 210 },
      data: { labels, datasets: [] },
      scales: { x: { getPixelForValue: (v) => 100 + v * 100 }, y: { getPixelForValue: (v) => 210 - v } },
      config: { options: { plugins: { llMarkers: { items: ITEMS, highlight: HIGHLIGHT } } } }
    };
"""


def _marker_calls(items, highlight="null"):
    script = FAKE_CHART.replace("ITEMS", items).replace("HIGHLIGHT", highlight) + """
        LL.charts.markers.beforeDatasetsDraw(chart, {}, {});
        const before = calls.slice();
        calls.length = 0;
        LL.charts.markers.afterDatasetsDraw(chart, {}, {});
        ({ before, after: calls })
    """
    return run_js(CHARTS, script)


def test_markers_draw_lines_bands_and_skip_out_of_range_items():
    calls = _marker_calls("""[
        { t: '2026-10-15T10:01:00Z', kind: 'line', label: 'деградация 10:01', tone: 'fail' },
        { t: '2026-10-15T11:00:00Z', kind: 'line', label: 'вне графика', tone: 'fail' },
        { t: '2026-10-15T10:00:30Z', to: '2026-10-15T10:02:00Z', kind: 'band', label: 'SLA: ступень 2', tone: 'accent' },
        { t: '2026-10-15T09:00:00Z', to: '2026-10-15T09:30:00Z', kind: 'band', tone: 'warn' },
        { t: '2026-10-15T09:58:00Z', to: '2026-10-15T10:01:00Z', kind: 'band', tone: 'warn' },
        { t: '2026-10-15T10:02:00Z', kind: 'tick' }
    ]""")
    bands = [c[1:] for c in calls["before"] if c[0] == "fillRect"]
    assert bands == [[150, 10, 150, 200], [100, 10, 100, 200]]  # second band is clipped at the first label
    moves = [c[1:] for c in calls["after"] if c[0] == "moveTo"]
    assert [200, 10] in moves          # dashed line over the full height
    assert [300, 203] in moves         # tick: a short mark at the x axis
    assert all(m[0] in (200, 300) for m in moves)
    texts = [c[1] for c in calls["after"] if c[0] == "fillText"]
    assert texts == ["SLA: ступень 2", "деградация 10:01"]


def test_marker_labels_stack_when_they_overlap():
    calls = _marker_calls("""[
        { t: '2026-10-15T10:01:00Z', kind: 'line', label: 'первая длинная подпись' },
        { t: '2026-10-15T10:01:10Z', kind: 'line', label: 'вторая' }
    ]""")
    tops = [c[3] for c in calls["after"] if c[0] == "fillText"]
    assert len(tops) == 2 and tops[0] != tops[1]


def test_marker_highlight_band_is_drawn():
    calls = _marker_calls("[]", "{ t: '2026-10-15T10:01:00Z', to: '2026-10-15T10:02:00Z' }")
    assert [c[1:] for c in calls["before"] if c[0] == "fillRect"] == [[200, 10, 100, 200]]


# ---- LL.relativeTime / LL.formatDuration / LL.reportUrl ----------------------------------

NOW = "new Date(2026, 9, 15, 12, 0, 0)"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (f"new Date({NOW}.getTime() - 30000)", "только что"),
        (f"new Date({NOW}.getTime() + 30000)", "только что"),  # small clock skew
        (f"new Date({NOW}.getTime() - 5 * 60000)", "5 мин назад"),
        (f"new Date({NOW}.getTime() - 59 * 60000 - 59000)", "59 мин назад"),
        (f"new Date({NOW}.getTime() - 3 * 3600000 - 1000)", "3 ч назад"),
        (f"new Date({NOW}.getTime() - 23 * 3600000)", "23 ч назад"),
        ("new Date(2026, 9, 14, 9, 5)", "вчера в 09:05"),
        ("new Date(2026, 9, 12, 18, 0)", "3 дн назад"),
        ("new Date(2026, 9, 9, 0, 1)", "6 дн назад"),
        ("new Date(2026, 9, 5, 12, 0)", "2026-10-05 12:00"),
        (f"new Date({NOW}.getTime() + 2 * 3600000)", "2026-10-15 14:00"),
        ("'garbage'", "—"),
        ("null", "—"),
        ("undefined", "—"),
    ],
)
def test_relative_time_branches(value, expected):
    assert run_js(COMMON, f"LL.relativeTime({value}, {NOW})") == expected


def test_relative_time_accepts_iso_and_ms():
    assert run_js(COMMON, "LL.relativeTime('2026-10-15T11:50:00Z', Date.parse('2026-10-15T12:00:00Z'))") == "10 мин назад"
    assert run_js(COMMON, "LL.relativeTime(Date.now() - 120000)") == "2 мин назад"
    assert run_js(COMMON, f"LL.relativeTime({NOW}, 'bad now')") == "—"


@pytest.mark.parametrize(
    ("ms", "expected"),
    [
        ("0", "0 с"),
        ("45000", "45 с"),
        ("59999", "59 с"),
        ("60000", "1 мин"),
        ("59 * 60000", "59 мин"),
        ("65 * 60000", "1 ч 05 мин"),
        ("(23 * 60 + 59) * 60000", "23 ч 59 мин"),
        ("(2 * 24 + 3) * 3600000", "2 д 3 ч"),
        ("24 * 3600000", "1 д 0 ч"),
        ("'90000'", "1 мин"),
        ("-1", "—"),
        ("NaN", "—"),
        ("Infinity", "—"),
        ("null", "—"),
        ("undefined", "—"),
        ("'abc'", "—"),
        ("true", "—"),
    ],
)
def test_format_duration(ms, expected):
    assert run_js(COMMON, f"LL.formatDuration({ms})") == expected


def test_report_url():
    result = run_js(COMMON, """[
        LL.reportUrl({ run_id: 'abc 1', run_name: 'ignored' }),
        LL.reportUrl({ run_name: 'run/1 ü' }),
        LL.reportUrl({ run_id: '', run_name: 'r' }),
        LL.reportUrl({}),
        LL.reportUrl(null),
        LL.reportUrl(undefined)
    ]""")
    assert result == ["/reports/abc%201", "/reports/run%2F1%20%C3%BC", "/reports/r", "/reports", "/reports", "/reports"]


# ---- LL.ui.emptyRow / skeletonRows -------------------------------------------------------

def test_empty_row_escapes_text_and_keeps_action_markup():
    text = '<b>x</b> & "y" \'z\''
    assert run_js(UI, f"LL.ui.emptyRow(3, {js_str(text)})") == (
        '<tr class="empty-row"><td colspan="3">&lt;b&gt;x&lt;/b&gt; &amp; &quot;y&quot; &#39;z&#39;</td></tr>'
    )
    action = '<a class="btn btn-sm" href="/new">Создать отчёт</a>'
    assert run_js(UI, f"LL.ui.emptyRow('2', 'Пусто', {{ actionHtml: {js_str(action)} }})") == (
        f'<tr class="empty-row"><td colspan="2">Пусто {action}</td></tr>'
    )
    assert run_js(UI, "LL.ui.emptyRow(0, null, null)") == '<tr class="empty-row"><td colspan="1"></td></tr>'


def test_skeleton_rows():
    html = run_js(UI, "LL.ui.skeletonRows(3, 2)")
    assert html.count('<tr class="skeleton-row"') == 2
    assert html.count("<td>") == 6
    assert html.count('class="skeleton skeleton-line') == 6
    assert run_js(UI, "LL.ui.skeletonRows(2)").count("<tr") == 5


# ---- LL.prefs -------------------------------------------------------------------------------

def test_prefs_round_trip_json_values():
    result = run_js(UI, """
        LL.prefs.set('recentServices', ['svc-a', 'svc-b']);
        LL.prefs.set('sample', 'compact');
        LL.prefs.set('custom', { n: 1, ok: true });
        ({ recent: LL.prefs.get('recentServices'), sample: LL.prefs.get('sample'), custom: LL.prefs.get('custom'),
           missing: LL.prefs.get('nope', 'fallback'), noFallback: LL.prefs.get('nope') === undefined,
           raw: Object.fromEntries(__harness.storage) })
    """)
    assert result["recent"] == ["svc-a", "svc-b"]
    assert result["sample"] == "compact"
    assert result["custom"] == {"n": 1, "ok": True}
    assert result["missing"] == "fallback"
    assert result["noFallback"] is True
    assert result["raw"] == {
        "loadlens.recentServices": '["svc-a","svc-b"]',
        "loadlens.sample": '"compact"',
        "loadlens.custom": '{"n":1,"ok":true}',
    }


def test_prefs_theme_stays_a_raw_string():
    result = run_js(UI, """
        const seeded = LL.prefs.get('theme');
        LL.prefs.set('theme', 'light');
        const afterSet = __harness.storage.get('loadlens.theme');
        LL.ui.theme.set('dark');
        [seeded, afterSet, __harness.storage.get('loadlens.theme'), LL.prefs.get('theme'), document.documentElement.getAttribute('data-theme')]
    """, local_storage={"loadlens.theme": "dark"})
    assert result == ["dark", "light", "dark", "dark", "dark"]


def test_prefs_null_removes_and_bad_json_falls_back():
    result = run_js(UI, """
        LL.prefs.set('sample', 'compact');
        LL.prefs.set('sample', null);
        [__harness.storage.has('loadlens.sample'), LL.prefs.get('sample', 'normal'), LL.prefs.get('broken', 'fb')]
    """, local_storage={"loadlens.broken": "{not json"})
    assert result == [False, "normal", "fb"]


def test_prefs_swallow_storage_errors():
    result = run_js(UI, """
        LL.prefs.set('sample', 'compact');
        LL.ui.theme.set('light');
        [LL.prefs.get('sample', 'normal'), LL.prefs.get('theme', 'dark'), LL.ui.theme.current()]
    """, storage_throws=True)
    assert result == ["normal", "dark", "light"]


def test_prefs_on_change_and_unsubscribe():
    result = run_js(UI, """
        const seen = [];
        const off = LL.prefs.onChange('sample', (value, key) => seen.push([key, value]));
        LL.prefs.set('sample', 'compact');
        LL.prefs.set('theme', 'light');
        __harness.fireWindow('storage', { key: 'loadlens.sample', newValue: '"normal"' });
        __harness.fireWindow('storage', { key: 'other.sample', newValue: '"x"' });
        off();
        LL.prefs.set('sample', 'compact');
        seen
    """)
    assert result == [["sample", "compact"], ["sample", "normal"]]


def test_theme_on_change_returns_unsubscribe():
    result = run_js(UI, """
        const seen = [];
        const off = LL.ui.theme.onChange((name) => seen.push(name));
        LL.ui.theme.set('light');
        off();
        LL.ui.theme.set('dark');
        seen
    """)
    assert result == ["light"]


# ---- Chart.js defaults ------------------------------------------------------------------------

def test_tooltip_defaults_follow_theme_tokens():
    css = theme_css_vars()
    result = run_js(CHARTS, """
        const t = Chart.defaults.plugins.tooltip;
        const dark = { bg: t.backgroundColor, border: t.borderColor, title: t.titleColor, body: t.bodyColor,
                       radius: t.cornerRadius, point: t.usePointStyle, padding: t.padding };
        LL.ui.theme.set('light');
        LL.ui.restyleCharts();
        ({ dark, light: { bg: t.backgroundColor, title: t.titleColor } })
    """)
    # Tooltips float: the popover surface and the strong hairline (plans/instrument-ui-plan.md 2.3).
    assert result["dark"] == {
        "bg": css["dark"]["--bg-pop"], "border": css["dark"]["--line-2"], "title": css["dark"]["--text"],
        "body": css["dark"]["--text"], "radius": 3, "point": True, "padding": 10,
    }
    assert result["light"] == {"bg": css["light"]["--bg-pop"], "title": css["light"]["--text"]}


def test_chart_defaults_use_the_mono_font():
    result = run_js(CHARTS, "[Chart.defaults.font.family, Chart.defaults.font.size, LL.ui.fonts.mono, LL.ui.fonts.sans]")
    assert result[0] == result[2] and result[0].startswith("'JetBrains Mono'") and result[1] == 11
    assert result[3].startswith("'Golos Text'")


def test_harness_reports_js_errors():
    with pytest.raises(JsHarnessError, match="boom"):
        run_js(COMMON, "throw new Error('boom')")


# ---- LL.charts.stepProfileModel / stepProfile (load profile, plans/instrument-ui-plan.md 2.3) ---------------

MSK = timezone(timedelta(hours=3))
P0 = datetime(2026, 10, 15, 19, 10, tzinfo=MSK)


def ms(moment):
    return int(moment.timestamp() * 1000)


def profile_steps(levels, minutes=10, selected=None, after_drop=()):
    """load_step_table.steps as the report stores them: ISO strings with the +03:00 offset of the report."""
    out = []
    for i, rps in enumerate(levels):
        start = P0 + timedelta(minutes=minutes * i)
        out.append({
            "index": i + 1, "label": f"{rps} RPS", "rps": rps, "selected": i == selected, "after_drop": i in after_drop,
            "start_iso": start.isoformat(), "end_iso": (start + timedelta(minutes=minutes)).isoformat(),
            "plateau_start_iso": (start + timedelta(minutes=1)).isoformat(),
            "plateau_end_iso": (start + timedelta(minutes=minutes - 1)).isoformat(),
        })
    return out


def profile_model(opts, **kwargs):
    return run_js(CHARTS, f"LL.charts.stepProfileModel({json.dumps(opts)})", locale="en", **kwargs)


@pytest.mark.parametrize("opts", [
    None, {}, {"steps": []}, {"steps": "nope"},
    {"steps": [{"index": 1, "rps": 50}]},
    {"steps": [{"start_iso": "soon", "end_iso": "later", "rps": 50}]},
    {"steps": [{"start_iso": P0.isoformat(), "end_iso": P0.isoformat(), "rps": 50}]},
    {"steps": [{"start_iso": P0.isoformat(), "end_iso": (P0 + timedelta(minutes=5)).isoformat(), "rps": None}]},
])
def test_step_profile_model_is_null_without_a_drawable_step(opts):
    assert profile_model(opts) is None


def test_step_profile_empties_and_hides_its_host_without_steps():
    result = run_js(CHARTS, """
        const host = document.createElement('div');
        host.innerHTML = 'old';
        const handle = LL.charts.stepProfile(host, { steps: [] });
        ({ hidden: host.hidden, html: host.innerHTML, model: handle.model, cls: host.className })
    """)
    assert result == {"hidden": True, "html": "", "model": None, "cls": "ll-profile"}
    assert run_js(CHARTS, "LL.charts.stepProfile(null, {})") is None


def test_step_profile_model_of_one_step():
    m = profile_model({"steps": profile_steps([120]), "width": 600, "height": 150})
    assert (m["x0"], m["x1"], m["width"], m["height"], m["compact"]) == (ms(P0), ms(P0 + timedelta(minutes=10)), 600, 150, False)
    # 120 RPS with headroom is 134.4: the scale ends at the round 150; the plot runs from y=10 to y=116.
    assert m["yMax"] == 150 and m["plot"] == {"top": 10, "bottom": 116} and m["maxRps"] == 120
    assert m["path"] == "M0 31.2H600"
    step = m["steps"][0]
    assert (step["x"], step["w"], step["y"], step["rps"], step["index"], step["level"]) == (0, 600, 31.2, 120, 1, None)
    assert m["bands"] == [] and m["lines"] == [] and m["target"] is None and m["sla"] is None
    assert [(g["value"], g["text"]) for g in m["grid"]] == [(50, "50"), (100, "100")]
    # Both ends plus round clock times of the report's offset (+03:00), clear of the end labels.
    assert [(t["x"], t["text"], t["anchor"]) for t in m["ticks"]] == [
        (0, "19:10", "start"), (120, "19:12", "middle"), (240, "19:14", "middle"), (360, "19:16", "middle"),
        (480, "19:18", "middle"), (600, "19:20", "end"),
    ]
    assert m["offsetHours"] == 3 and m["aria"] == "Load profile: 1 step, peak 120 RPS"


def test_step_profile_staircase_runs_to_the_next_step_and_marks_the_after_drop_level():
    steps = profile_steps([50, 100, 185.4], after_drop=(2,))
    steps[1]["end_iso"] = (P0 + timedelta(minutes=15)).isoformat()  # a pause before the third step
    m = profile_model({"steps": list(reversed(steps)), "width": 600, "compact": True, "target": 200})
    # Sorted by start; 200 RPS target with headroom (224) ends the scale at 250. Compact plot: y 8..128.
    assert m["yMax"] == 250 and m["plot"] == {"top": 8, "bottom": 128}
    assert m["path"] == "M0 104H200V80H400V39H600"
    assert [s["level"]["text"] for s in m["steps"]] == ["50", "100", "~185"]
    assert all(s["level"]["anchor"] == "middle" for s in m["steps"])
    assert m["target"] == {"y": 32, "value": 200, "text": "200 · target", "x": 4, "anchor": "start"}
    assert m["grid"] == []
    ru = json.loads((REPO_ROOT / "locales" / "ru" / "charts.json").read_text(encoding="utf-8"))["charts.profile.target"]
    assert run_js(CHARTS, f"LL.charts.stepProfileModel({json.dumps({'steps': steps, 'target': 200})}).target.text") == ru.replace("{value}", "200")


def test_step_profile_sla_zone_from_the_window_the_marker_or_the_selected_step():
    steps = profile_steps([50, 100, 150, 200, 250], selected=4)
    window = {"start": (P0 + timedelta(minutes=41)).isoformat(), "end": (P0 + timedelta(minutes=49)).isoformat(), "level": 250}
    full = profile_model({"steps": steps, "window": window, "width": 1000})
    assert full["bands"] == [{"x": 820, "w": 160, "label": "SLA · step 5", "labelX": 824, "labelY": 170, "tone": "accent", "role": "sla"}]
    assert full["sla"] == {"index": 5, "rps": 250, "start": ms(P0 + timedelta(minutes=41)), "end": ms(P0 + timedelta(minutes=49)),
                           "startText": "19:51", "endText": "19:59"}
    compact = profile_model({"steps": steps, "window": window, "width": 1000, "compact": True})
    assert compact["bands"][0]["label"] == "SLA" and compact["bands"][0]["labelY"] == 122
    # The selected step's plateau when there is no window: the same zone.
    assert profile_model({"steps": steps, "width": 1000})["bands"] == full["bands"]
    # A window reaching outside the run is clipped; one entirely outside draws nothing.
    early = {"start_iso": (P0 - timedelta(minutes=30)).isoformat(), "end_iso": (P0 + timedelta(minutes=5)).isoformat()}
    unselected = profile_steps([50, 100, 150, 200, 250])
    clipped = profile_model({"steps": unselected, "window": early, "width": 1000})
    assert [(b["x"], b["w"], b["label"]) for b in clipped["bands"]] == [(0, 100, "SLA · step 1")]
    assert clipped["sla"]["startText"] == "19:10" and clipped["sla"]["index"] == 1
    outside = {"start": (P0 + timedelta(hours=2)).isoformat(), "end": (P0 + timedelta(hours=3)).isoformat()}
    nothing = profile_model({"steps": unselected, "window": outside, "width": 1000})
    assert nothing["bands"] == [] and nothing["sla"] is None
    # Without a window and a selected step, the SLA band of the report markers (its own label in the full size).
    band = {"kind": "band", "role": "sla", "tone": "accent", "label": "SLA: step 4",
            "t": ms(P0 + timedelta(minutes=31)), "to": (P0 + timedelta(minutes=39)).isoformat()}
    marked = profile_model({"steps": unselected, "markers": [band], "width": 1000})
    assert [(b["x"], b["w"], b["label"]) for b in marked["bands"]] == [(620, 160, "SLA: step 4")]


def test_step_profile_lines_and_bands_of_the_markers():
    steps = profile_steps([50, 100, 150, 200, 250])
    markers = [
        {"kind": "line", "t": ms(P0 + timedelta(minutes=45)), "to": None, "label": "degradation 19:55", "tone": "fail", "role": "degradation"},
        {"kind": "line", "t": (P0 + timedelta(minutes=47)).isoformat(), "label": "RPS drop 19:57", "tone": "odd", "role": "rps_drop"},
        {"kind": "line", "t": ms(P0 + timedelta(minutes=10)), "label": "early", "tone": "warn", "role": "x"},
        {"kind": "line", "t": ms(P0 - timedelta(minutes=5)), "label": "before the run", "tone": "fail", "role": "rps_drop"},
        {"kind": "line", "t": ms(P0 + timedelta(minutes=51)), "label": "after the run", "tone": "fail", "role": "rps_drop"},
        {"kind": "line", "t": "not a time", "label": "broken", "tone": "fail", "role": "rps_drop"},
        {"kind": "tick", "t": ms(P0 + timedelta(minutes=20)), "label": "", "tone": "muted", "role": "step"},
        {"kind": "band", "t": ms(P0 + timedelta(minutes=45)), "to": ms(P0 + timedelta(minutes=70)), "label": "hold", "tone": "warn", "role": "hold"},
        {"kind": "band", "t": ms(P0 + timedelta(hours=2)), "to": ms(P0 + timedelta(hours=3)), "label": "spike", "tone": "warn", "role": "spike"},
    ]
    m = profile_model({"steps": steps, "markers": markers, "width": 1000})
    lines = [(l["x"], l["label"], l["labelX"], l["labelY"], l["anchor"], l["tone"], l["role"]) for l in m["lines"]]
    assert lines == [
        (200, "early", 205, 20, "start", "warn", "x"),
        # Labels near the right edge turn left; a label that would overlap the previous one goes a row down.
        (900, "degradation 19:55", 895, 20, "end", "fail", "degradation"),
        (940, "RPS drop 19:57", 935, 33, "end", "fail", "rps_drop"),
    ]
    # Ticks are not drawn (the staircase shows the step starts); a band reaching past the end is clipped.
    assert [(b["x"], b["w"], b["label"], b["tone"], b["role"]) for b in m["bands"]] == [(900, 100, "hold", "warn", "hold")]


def test_step_profile_clock_follows_the_offset_of_the_steps():
    naive = [{"index": 1, "rps": 10, "start_iso": "2026-10-15T16:10:00", "end_iso": "2026-10-15T16:40:00"}]
    # A naive ISO time is UTC (as in the report markers); without an offset the clock is +03:00.
    assert profile_model({"steps": naive, "width": 400})["ticks"][0]["text"] == "19:10"
    assert profile_model({"steps": naive, "width": 400, "offsetHours": 0})["ticks"][0]["text"] == "16:10"
    utc = [{"index": 1, "rps": 10, "start_iso": "2026-10-15T16:10:00-02:00", "end_iso": "2026-10-15T16:40:00-02:00"}]
    first = profile_model({"steps": utc, "width": 400})
    assert first["offsetHours"] == -2 and first["ticks"][0]["text"] == "16:10" and first["ticks"][-1]["text"] == "16:40"


def test_step_profile_renders_an_escaped_svg_and_redraws():
    steps = profile_steps([50, 100, 185.4], selected=1)
    marker = {"kind": "line", "t": ms(P0 + timedelta(minutes=20)), "label": "<b>&", "tone": "fail", "role": "degradation"}
    result = run_js(CHARTS, f"""
        const host = document.createElement('div');
        const handle = LL.charts.stepProfile(host, {{ steps: {json.dumps(steps)}, markers: [{json.dumps(marker)}], compact: true }});
        const first = {{ html: host.innerHTML, hidden: host.hidden, cls: host.className, width: handle.model.width, sla: handle.model.sla.index }};
        handle.update({{ steps: [] }});
        const emptied = {{ html: host.innerHTML, hidden: host.hidden }};
        handle.update({{ steps: {json.dumps(steps)} }});
        ({{ first, emptied, again: host.hidden, same: host.$llProfile === handle }})
    """, locale="en")
    html = result["first"]["html"]
    assert html.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 150" width="100%" height="150"')
    assert 'role="img" aria-label="Load profile: 3 steps, peak 185 RPS"' in html
    for part in ('class="pf-band tone-accent"', 'class="pf-path" d="M0 ', 'class="pf-axis"', 'class="pf-mark tone-fail"',
                 'class="pf-level is-selected"', '>&lt;b&gt;&amp;</text>', '>SLA</text>', '>19:10</text>', '>19:40</text>'):
        assert part in html, part
    assert "<b>" not in html
    assert result["first"]["hidden"] is False and result["first"]["cls"] == "ll-profile"
    assert result["first"]["width"] == 560 and result["first"]["sla"] == 2
    assert result["emptied"] == {"html": "", "hidden": True}
    assert result["again"] is False and result["same"] is True
