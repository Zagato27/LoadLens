"""Chart hover in static/js/charts.js: the HTML tooltip next to the cursor, the crosshair plugin and
live legend values, run under Node (tests/js_harness.py)."""

from pathlib import Path

from tests.js_harness import run_js

REPO = Path(__file__).resolve().parents[1]
CHARTS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js"]


def test_tooltip_sits_right_and_below_the_pointer_when_it_fits():
    # px, py, width, height, viewport width, viewport height
    assert run_js(CHARTS, "LL.charts.tipPosition(500, 300, 220, 160, 1440, 900)") == {"left": 514, "top": 314}


def test_tooltip_flips_left_near_the_right_edge_and_up_near_the_bottom():
    pos = run_js(CHARTS, "[LL.charts.tipPosition(1300, 300, 220, 160, 1440, 900), LL.charts.tipPosition(500, 820, 220, 160, 1440, 900)]")
    # Left of the pointer: the area right of the cursor stays visible.
    assert pos[0] == {"left": 1300 - 14 - 220, "top": 314}
    # Above the pointer near the bottom of the window.
    assert pos[1] == {"left": 514, "top": 820 - 14 - 160}


def test_tooltip_is_clamped_to_the_viewport():
    pos = run_js(CHARTS, "[LL.charts.tipPosition(100, 100, 220, 900, 1440, 700), LL.charts.tipPosition(10, 10, 600, 100, 390, 844)]")
    assert pos[0]["top"] == 8  # taller than the window: pinned to the top margin
    assert pos[1]["left"] == 8  # wider than a phone screen: pinned to the left margin


def test_html_tooltip_replaces_the_canvas_box_and_keeps_eight_rows_by_default():
    result = run_js(CHARTS, """(() => {
      const values = Array.from({ length: 11 }, (_, i) => i * 10);
      const chart = { canvas: {}, ctx: {}, data: { datasets: values.map((v) => ({ data: [v] })) }, isDatasetVisible: () => true };
      const items = values.map((v, i) => ({ parsed: { y: v }, datasetIndex: i, dataIndex: 0, chart }));
      const tip = LL.charts.htmlTooltip();
      const kept = items.filter((item, i, all) => tip.filter(item, i, all)).sort(tip.itemSort);
      return { enabled: tip.enabled, external: typeof tip.external, kept: kept.map((it) => it.datasetIndex),
               footer: tip.callbacks.footer(kept), five: LL.charts.htmlTooltip({ max: 5 }).filter(items[0], 0, items) };
    })()""")
    assert result["enabled"] is False and result["external"] == "function"
    assert result["kept"] == [10, 9, 8, 7, 6, 5, 4, 3]
    assert result["footer"] == "ещё 3"
    assert result["five"] is False  # the smallest value is not among the five largest


def test_value_at_reads_numbers_and_points():
    values = run_js(CHARTS, """[
      LL.charts.valueAt({ data: [1, 2.5, null] }, 1),
      LL.charts.valueAt({ data: [{ x: 0, y: 7 }] }, 0),
      String(LL.charts.valueAt({ data: [1, null] }, 1)),
      String(LL.charts.valueAt({ data: ['x'] }, 0)),
      String(LL.charts.valueAt(null, 0))
    ]""")
    assert values == [2.5, 7, "NaN", "NaN", "NaN"]


def test_crosshair_feeds_live_values_to_the_legend_once_per_index():
    calls = run_js(CHARTS, """(() => {
      const seen = [];
      let active = [];
      const chart = { getActiveElements: () => active, $llLegend: { showIndex: (i) => seen.push(i) } };
      const hook = LL.charts.crosshair.afterDraw;
      hook(chart);                                   // nothing hovered: averages (-1)
      active = [{ index: 4, datasetIndex: 0, element: { x: 10 } }];
      hook(chart); hook(chart);                      // hovered twice at the same index: one call
      active = [{ index: 5, datasetIndex: 0, element: { x: 12 } }];
      hook(chart);
      active = [];
      hook(chart);                                   // pointer left: back to averages
      return seen;
    })()""")
    assert calls == [-1, 4, 5, -1]


def test_pointer_tracker_is_global_and_crosshair_is_per_chart():
    ids = run_js(CHARTS, "__harness.registered.map((p) => p.id)")
    assert "llPointer" in ids
    assert "llCrosshair" not in ids
    assert run_js(CHARTS, "LL.charts.crosshair.id") == "llCrosshair"


def test_synchronized_charts_get_no_tooltip_and_pages_use_the_cursor_tooltip():
    charts_js = (REPO / "static/js/charts.js").read_text(encoding="utf-8")
    assert "tooltip.setActiveElements(elements" not in charts_js
    # The box follows the pointer inside a time bucket: a positioner returning the event position.
    assert "positioners[POINTER_POSITIONER] = function followPointer" in charts_js
    assert "tip.position = POINTER_POSITIONER" in charts_js
    reports_js = (REPO / "static/js/reports.js").read_text(encoding="utf-8")
    assert "LL.charts.htmlTooltip({ max: TOOLTIP_ROWS })" in reports_js and "LL.charts.crosshair" in reports_js
    compare_js = (REPO / "static/js/compare.js").read_text(encoding="utf-8")
    assert "LL.charts.htmlTooltip({ max: TOOLTIP_MAX })" in compare_js and "plugins: [LL.charts.crosshair]" in compare_js


def test_tooltip_and_live_legend_styles_use_theme_tokens():
    css = (REPO / "static/css/app.css").read_text(encoding="utf-8")
    assert ".ll-chart-tip { position: fixed;" in css and "pointer-events: none" in css
    assert ".legend-chips.is-live .legend-chip-avg" in css
