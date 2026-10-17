"""Report page helpers (static/js/reports.js, LL.charts.topTooltip) and the stage-1 report styles, run under Node."""

import json
import re
from pathlib import Path

import pytest

from tests.js_harness import run_js, theme_css_vars

REPO = Path(__file__).resolve().parents[1]
REPORT_JS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
CHARTS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js"]
I = "LL.report._internals"


def js(value):
    return json.dumps(value, ensure_ascii=False)


def check(name, threshold, actual, passed):
    return {"name": name, "threshold": threshold, "actual": actual, "passed": passed, "message": ""}


# ---- SLA mini bar ---------------------------------------------------------------------------

def test_sla_bar_model():
    bars = run_js(REPORT_JS, f"""[
      {I}.slaBarModel({js(check('p95_latency', 300, 450, False))}),
      {I}.slaBarModel({js(check('p95_latency', 300, 150, True))}),
      {I}.slaBarModel({js(check('target_rps', 200, 249.3, True))}),
      {I}.slaBarModel({js(check('error_rate', 1, 9, None))}),
      {I}.slaBarModel({js(check('error_rate', 0, 1, True))}),
      {I}.slaBarModel({js(check('error_rate', 1, None, None))}),
      {I}.slaBarModel({js(check('error_rate', '', 1, True))})
    ]""")
    assert bars[0] == {"fill": 100, "mark": 66.7, "tone": "fail", "higherBetter": False, "percent": 150}
    assert bars[1]["fill"] == 33.3 and bars[1]["tone"] == "ok"
    assert bars[2] == {"fill": 83.1, "mark": 66.7, "tone": "ok", "higherBetter": True, "percent": 125}
    assert bars[3]["fill"] == 100 and bars[3]["tone"] == "na"  # capped at 1.5x the threshold
    assert bars[4:] == [None, None, None]


# ---- hero KPI tiles -----------------------------------------------------------------------------

def final_row(parsed, checks):
    return {"domain": "final", "parsed": parsed, "sla_details": {"checks": checks}}


CHECKS = [check("target_rps", 200, 249.3, True), check("p95_latency", 300, 450, False), check("error_rate", 1, 0.4, True)]


def kpis(row):
    return run_js(REPORT_JS, f"{I}.heroKpis({js(row)})")


def test_hero_kpis_for_a_step_test():
    tiles = kpis(final_row({"peak_performance": {"max_rps": 249.3, "method": "last_stable_step", "max_time": "18:10"}}, CHECKS))
    assert [(t["key"], t["value"], t["unit"], t["tone"]) for t in tiles] == [
        ("rps", "249.3", " RPS", "ok"), ("latency", "450", " мс", "fail"), ("errors", "0.4", " %", "ok"),
    ]
    assert tiles[0]["note"] == "цель ≥ 200 RPS · достигнута"
    assert tiles[1]["note"] == "порог ≤ 300 мс · нарушен"
    assert tiles[0]["term"] == "stable_max"


def test_hero_kpis_fall_back_and_skip_missing_checks():
    tiles = kpis(final_row({}, [check("p99_latency", 900, 800, None)]))
    assert [(t["key"], t["label"], t["value"], t["tone"], t["note"]) for t in tiles] == [("latency", "p99", "800", "", "порог ≤ 900 мс")]
    assert kpis(final_row({}, [])) == []
    assert kpis(None) == []
    no_peak = kpis(final_row({}, [check("target_rps", 200, None, None)]))
    assert (no_peak[0]["value"], no_peak[0]["unit"]) == ("—", "")


def test_hero_kpis_by_test_type_and_meta_without_duplicates():
    stress = {"saturation": {"saturation_rps": 200, "saturation_step_label": "ступень 4", "failure_mode_title": "рост задержек"}}
    soak = {"stability_under_load": {"target_rps": 180, "actual_rps": 182.4, "hold": {"start_iso": "2026-10-15T15:42:00+03:00", "end_iso": "2026-10-15T16:29:00+03:00", "min_bucket_rps": 171}}}
    spike = {"spike_analysis": {"detected": True, "baseline_rps": 60, "max_spike_rps": 250, "worst_spike_index": 2,
                                "spikes": [{"index": 1, "recovered": True, "recovery_time_sec": 60, "overshoot_p95_pct": 10},
                                           {"index": 2, "recovered": False, "overshoot_p95_pct": 41.5}]}}
    stress_tiles = kpis(final_row(stress, CHECKS))
    assert [(t["label"], t["value"]) for t in stress_tiles] == [("Точка насыщения", "200"), ("p95", "450"), ("Ошибки", "0.4")]
    soak_tiles = kpis(final_row(soak, CHECKS))
    assert soak_tiles[0]["value"] == "182.4"
    assert soak_tiles[0]["note"] == "цель ≥ 180 RPS · удержана · 15:42–16:29"
    spike_tiles = kpis(final_row(spike, CHECKS + [check("recovery", 5, None, False)]))
    assert [(t["key"], t["value"], t["tone"]) for t in spike_tiles] == [
        ("spike", "250", "ok"), ("overshoot", "+41.5 %", "fail"), ("recovery", "не восстановилось: 1 из 2", "fail"),
    ]
    assert spike_tiles[0]["note"] == "фон 60 RPS · всплесков: 2 · цель ≥ 200 RPS · достигнута"
    metas = run_js(REPORT_JS, f"""[
      {I}.heroTypeEntries({js(stress)}, {{}}).map((e) => e.label),
      {I}.heroTypeEntries({js(soak)}, {{}}).map((e) => e.label),
      {I}.heroTypeEntries({js(spike)}, {{}}).map((e) => e.label),
      {I}.heroTypeEntries({{}}, {{ max_rps: 10, max_time: '18:10' }}).map((e) => e.label)
    ]""")
    assert metas == [["Характер деградации", "Ограничивающий ресурс"], ["Мин. интервал RPS"], [], ["Время пика", "Время деградации"]]


# ---- chart data ---------------------------------------------------------------------------------

def test_chart_data_orders_series_hides_the_tail_and_uses_palette_slots():
    points = []
    for i in range(3):
        t = f"2026-10-15T10:0{2 - i}:00+00:00"  # out of order on purpose
        for s in range(14):
            points.append({"t": t, "series": f"s{s:02d}", "value": s + i})
    points.append({"t": "2026-10-15T10:00:00+00:00", "series": None, "value": 1})
    data = run_js(REPORT_JS, f"""(() => {{
      const d = {I}.buildChartData({js(points)});
      return {{ labels: d.labels, names: d.datasets.map((x) => x.label), slots: d.datasets.map((x) => x.llSeries),
                hidden: d.datasets.map((x) => x.hidden), colors: d.datasets.map((x) => x.borderColor),
                palette: d.datasets.map((x, i) => LL.charts.palette(i)), first: d.datasets[0].data }};
    }})()""")
    assert data["labels"] == [f"2026-10-15T10:0{i}:00+00:00" for i in range(3)]
    assert data["names"][:3] == ["s13", "s12", "s11"]
    assert data["slots"] == list(range(15))
    assert data["hidden"] == [False] * 12 + [True] * 3
    assert data["colors"] == data["palette"]
    assert "series" in data["names"]
    assert data["first"] == [15, 14, 13]


def test_utc_moment_and_marker_mode():
    result = run_js(REPORT_JS, f"""[
      {I}.utcMoment('2026-10-15 10:00:00'), {I}.utcMoment('2026-10-15T10:00:00+03:00'), {I}.utcMoment(''), {I}.utcMoment('nope'),
      {I}.markerMode({{ spike_analysis: {{}} }}, {{}}), {I}.markerMode({{}}, {{ kind: 'hold' }}), {I}.markerMode({{ test_profile: {{ mode: 'soak' }} }}, {{}}),
      {I}.markerMode({{ saturation: {{}} }}, {{}})
    ]""", tz="America/New_York")
    assert result == [1792058400000, 1792047600000, None, None, "spike", "hold", "hold", "step"]


def test_chosen_sla_step_prefers_marked_then_window_then_target_level():
    steps = [
        {"index": 1, "start_iso": "2026-10-15T10:00:00Z", "end_iso": "2026-10-15T10:10:00Z", "rps": 100, "stable": True},
        {"index": 2, "start_iso": "2026-10-15T10:10:00Z", "end_iso": "2026-10-15T10:20:00Z", "rps": 200, "stable": False},
        {"index": 3, "start_iso": "2026-10-15T10:20:00Z", "end_iso": "2026-10-15T10:30:00Z", "rps": 200, "stable": True},
    ]
    marked = [dict(s, selected=(s["index"] == 1)) for s in steps]
    result = run_js(REPORT_JS, f"""[
      ({I}.chosenSlaStep({js(marked)}, {{ start: '2026-10-15T10:25:00Z' }}, []) || {{}}).index,
      ({I}.chosenSlaStep({js(steps)}, {{ start: '2026-10-15T10:15:00Z' }}, []) || {{}}).index,
      ({I}.chosenSlaStep({js(steps)}, {{}}, [{{ name: 'target_rps', actual: 200.4 }}]) || {{}}).index,
      {I}.chosenSlaStep({js(steps)}, {{}}, [{{ name: 'target_rps', actual: 150 }}])
    ]""")
    assert result == [1, 2, 3, None]


# ---- LL.charts.topTooltip -------------------------------------------------------------------

def test_top_indices():
    assert run_js(CHARTS, "LL.charts.topIndices([5, null, 9, 'x', 9, 1], 3)") == [0, 2, 4]
    assert run_js(CHARTS, "LL.charts.topIndices([1, 2], 12)") == [0, 1]


def test_top_tooltip_keeps_the_largest_values_sorted_with_a_footer():
    result = run_js(CHARTS, """(() => {
      const values = Array.from({ length: 15 }, (_, i) => (i === 3 ? null : i * 10));
      const chart = { canvas: {}, ctx: {}, data: { datasets: values.map((v) => ({ data: [v] })) }, isDatasetVisible: (i) => i !== 14 };
      const items = values.map((v, i) => ({ parsed: { y: v === null ? NaN : v }, datasetIndex: i, dataIndex: 0, chart }))
        .filter((item) => chart.isDatasetVisible(item.datasetIndex) && Number.isFinite(item.parsed.y));
      const tip = LL.charts.topTooltip({ max: 12 });
      const kept = items.filter((item, i, all) => tip.filter(item, i, all)).sort(tip.itemSort);
      const small = items.slice(0, 3);
      return { kept: kept.map((item) => item.datasetIndex), footer: tip.callbacks.footer(kept),
               smallKept: small.filter((item, i, all) => tip.filter(item, i, all)).length, smallFooter: tip.callbacks.footer(small),
               empty: tip.callbacks.footer([]) };
    })()""")
    assert result["kept"] == [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 2, 1]
    assert result["footer"] == "ещё 1"
    assert result["smallKept"] == 3
    assert result["smallFooter"] == "ещё 10"
    assert result["empty"] == ""


# ---- page wiring and styles -----------------------------------------------------------------

def test_reports_js_has_no_legacy_legend_code_and_registers_the_plugins():
    source = (REPO / "static/js/reports.js").read_text(encoding="utf-8")
    for name in ("syncLegends", "scheduleLegendSync", "attachPanelObservers", "buildLegendFor", "downloadChartWithLegend", "cmp-legend"):
        assert name not in source
    assert "plugins: plugins" in source and "LL.charts.markers, LL.charts.thresholds" in source
    assert "maintainAspectRatio: false" in source and "mode: 'index'" in source
    template = (REPO / "templates/reports.html").read_text(encoding="utf-8")
    assert 'id="printReport"' in template and "onclick" not in template


def test_reports_js_loads_without_a_dom_and_keeps_a_token_background():
    ids = run_js(REPORT_JS, "__harness.registered.map((p) => p.id)")
    assert "cmpBg" in ids
    source = (REPO / "static/js/reports.js").read_text(encoding="utf-8")
    assert "'#151515'" not in source


def _stage1_css():
    css = (REPO / "static/css/app.css").read_text(encoding="utf-8")
    match = re.search(r"/\* ===== ui-plan stage 1: report ===== \*/(.*?)/\* ===== end stage 1 ===== \*/", css, re.S)
    assert match, "stage 1 CSS section is missing"
    return match.group(1)


def test_print_styles_force_the_light_tokens():
    section = _stage1_css()
    media = section[section.index("@media print"):]
    block = re.search(r':root, :root\[data-theme="dark"\] \{(.*?)\n  \}', media, re.S)
    assert block, "print block must override the dark tokens"
    tokens = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", block.group(1)))
    light = theme_css_vars()["light"]
    paper = {"--bg", "--bg-sidebar", "--bg-chart", "--shadow", "--shadow-sm"}
    assert {name for name in tokens if name.startswith("--series-")} == {f"--series-{i}" for i in range(1, 9)}
    for name, value in tokens.items():
        if name in paper:
            continue
        assert value.strip() == light[name], name
    for selector in (".app-sidebar", ".app-topbar", ".page-actions", ".feedback-votes", ".legend-chips-tools", "break-inside: avoid"):
        assert selector in media


@pytest.mark.parametrize("selector", [".report-tabs-bar", ".tabs-sticky-sentinel", ".report-kpis", ".sla-bar", ".report-chart-canvas", ".hero-card-foot"])
def test_stage1_selectors_exist(selector):
    assert selector in _stage1_css()


def test_sticky_tabs_sit_under_the_topbar_on_narrow_screens():
    section = _stage1_css()
    assert re.search(r"\.report-tabs-bar \{[^}]*position: sticky; top: 0;", section)
    assert re.search(r"@media \(max-width: 1023px\) \{ \.report-tabs-bar \{ top: var\(--topbar-h\); \} \}", section)


# ---- manual verdict ---------------------------------------------------------------------------------

MANUAL = {"verdict": "Успешно", "reason": "сбой <генератора>", "author": "ivanov", "changed_at": "2026-10-17T12:00:00+00:00"}


def test_verdict_source_names_the_manual_change_and_the_sla():
    calls = [
        {"manual_verdict": MANUAL, "auto_verdict": "Провал", "sla_verdict": "Провал"},
        {"sla_verdict": "Провал", "llm_verdict": "Провал", "verdict": "Провал"},
        {"sla_verdict": "Есть риски", "llm_verdict": "Провал", "verdict": "Есть риски"},
        {"sla_verdict": None, "llm_verdict": "Недостаточно данных", "verdict": "Недостаточно данных"},
        {"sla_verdict": None, "llm_verdict": "Успешно", "verdict": "Успешно"},
    ]
    texts = run_js(REPORT_JS, f"{js(calls)}.map((row) => {I}.verdictSourceText(row))")
    assert texts[0] == "Изменён вручную: ivanov, 2026-10-17 12:00. Автоматически: Провал."
    assert texts[1] == "Определён по SLA-критериям."  # the stored model verdict equals the SLA one
    assert texts[2] == "Определён по SLA-критериям. Оценка ИИ: Провал."  # a report built before the SLA replaced it
    assert texts[3].startswith("SLA-критерии не заданы, поэтому автоматический вердикт — «Недостаточно данных»")
    assert texts[4] == "SLA-критерии не заданы: вердикт определён оценкой ИИ."


def test_verdict_dialog_and_history_markup():
    state = {
        "verdict": "Успешно", "automatic": "Провал", "manual": MANUAL, "reason_max": 1000,
        "choices": ["Успешно", "Есть риски", "Провал", "Недостаточно данных"],
        "history": [
            {"verdict": None, "previous_verdict": "Есть риски", "auto_verdict": "Провал", "reason": "", "author": "petrov",
             "changed_at": "2026-10-17T13:00:00+00:00"},
            {**MANUAL, "previous_verdict": "Провал", "auto_verdict": "Провал"},
        ],
    }
    html = run_js(REPORT_JS, f"{I}.verdictDialogHtml({js(state)})")
    assert 'value="Успешно" disabled' in html and 'value="Провал">' in html  # the verdict in force cannot be picked
    assert "изменён вручную; автоматический — «Провал»" in html and 'maxlength="1000"' in html
    assert "История изменений (2)" in html and "возвращён автоматический: Провал" in html and "Провал → Успешно" in html
    assert "сбой &lt;генератора&gt;" in html and "<генератора>" not in html
    assert run_js(REPORT_JS, f"{I}.verdictHistoryHtml([])") == ""
    problems = run_js(REPORT_JS, f"""[
      {I}.verdictFormProblem('', 'x', {{}}),
      {I}.verdictFormProblem('Провал', '', {{}}),
      {I}.verdictFormProblem('Провал', 'xxxx', {{ reason_max: 3 }}),
      {I}.verdictFormProblem('Провал', 'ok', {{}})
    ]""")
    assert problems[0] == "Выберите новый вердикт" and problems[1].startswith("Укажите причину")
    assert problems[2] == "Причина — не длиннее 3 символов" and problems[3] == ""


def test_manual_verdict_tooltip_for_the_archive_and_the_dashboard():
    titles = run_js(REPORT_JS, f"[LL.manualVerdictTitle({js(MANUAL)}), LL.manualVerdictTitle(null), LL.manualVerdictTitle({{ verdict: 'Провал' }})]")
    assert titles == ["Изменён вручную: ivanov, 2026-10-17 12:00. Причина: сбой <генератора>", "", "Изменён вручную."]
