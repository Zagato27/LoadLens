# SPDX-License-Identifier: Apache-2.0
"""Report page of the «instrument» redesign (plans/instrument-ui-plan.md, stage 1): the verdict sentence built from
the SLA checks, the verdict block (figures, SLA criteria, verdict line), the meta line, «Ход теста» (load profile and
step row), findings in two columns, the template header and the stage-1 CSS section. JS runs under Node."""

import json
import re
from pathlib import Path

import pytest

from tests.js_harness import run_js

REPO = Path(__file__).resolve().parents[1]
REPORT_JS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
I = "LL.report._internals"


def js(value):
    return json.dumps(value, ensure_ascii=False)


def text(html):
    html = html.replace('<span class="sep" aria-hidden="true">·</span>', ' · ')
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def check(name, threshold, actual, passed, **extra):
    return {"name": name, "threshold": threshold, "actual": actual, "passed": passed, "message": "", **extra}


def row(checks, window=None, parsed=None, **extra):
    sla = {"checks": checks}
    if window is not None:
        sla["stable_window"] = window
    return {"domain": "final", "parsed": parsed or {}, "sla_details": sla, **extra}


MET_BREACH = [check("target_rps", 200, 249.3, True), check("p95_latency", 300, 450, False), check("error_rate", 1, 0.4, True)]
ALL_MET = [check("target_rps", 200, 249.3, True), check("p95_latency", 300, 250, True), check("error_rate", 1, 0.4, True)]
MISSED = [check("target_rps", 200, 180, False), check("p95_latency", 300, 450, False), check("error_rate", 1, 2, False)]


def summaries(rows, locale):
    return run_js(REPORT_JS, f"{js(rows)}.map((r) => {I}.verdictSummaryHtml(r))", locale=locale)


# ---- the verdict sentence ------------------------------------------------------------------------------------------

def test_verdict_sentence_in_russian():
    out = [text(html) for html in summaries([
        row(MET_BREACH), row(ALL_MET), row(MISSED), row([check("p95_latency", 300, 450, False)]),
        row([check("target_rps", 200, 210, True), check("memory_usage", 80, None, None)]),
        row([check("cpu_usage", 80, None, None)]), row(MET_BREACH, {"kind": "hold"}),
        row([check("target_rps", 200, 250, True)], {"kind": "spike"}),
        row([check("recovery", 5, 7, False)]), row([check("recovery", 5, None, False)]),
    ], "ru")]
    assert out == [
        "Целевой RPS достигнут на стабильной ступени, но p95 выше порога: 450 мс при пороге 300 мс; доля ошибок 0.4 %.",
        "Целевой RPS достигнут на стабильной ступени, остальные пороги соблюдены; доля ошибок 0.4 %.",
        "Целевой RPS не достигнут: 180 RPS при цели 200 RPS; кроме того, выше порогов: p95, доля ошибок.",
        "p95 выше порога: 450 мс при пороге 300 мс.",
        "Целевой RPS достигнут на стабильной ступени; нет данных: память.",
        "SLA-критерии проверить не удалось, нет данных: CPU.",
        "Целевой RPS удержан на окне удержания, но p95 выше порога: 450 мс при пороге 300 мс; доля ошибок 0.4 %.",
        "Всплеск достиг целевого RPS.",
        "Восстановление после всплеска дольше предела: 7 мин при пределе 5 мин.",
        "Восстановление после всплеска: порог нарушен.",
    ]


def test_verdict_sentence_in_english_and_figures_in_bold():
    html = summaries([row(MET_BREACH), row(ALL_MET), row(MISSED), row([check("error_rate", 1, 3, False)])], "en")
    assert [text(item) for item in html] == [
        "The target RPS was reached on the stable step, but p95 is above the threshold: 450 ms against 300 ms; error rate 0.4 %.",
        "The target RPS was reached on the stable step, and the other thresholds are met; error rate 0.4 %.",
        "The target RPS was not reached: 180 RPS against the target of 200 RPS; also, above the thresholds: p95, error rate.",
        "Error rate is above the threshold: 3 % against 1 %.",
    ]
    assert "<b>450 ms</b>" in html[0] and "<b>300 ms</b>" in html[0] and "<b>0.4 %</b>" in html[0]


def test_verdict_sentence_reads_the_checks_only_and_never_mentions_ai():
    """Built from sla_details.checks (or the deterministic SLA of parsed): the model's verdict and texts do not count."""
    ai_row = row(MET_BREACH, parsed={"verdict": "pass", "verdict_rationale": "AI says all good"}, llm_verdict="pass")
    legacy = {"domain": "final", "parsed": {"deterministic_sla": {"checks": MET_BREACH}}, "sla_details": {}}
    for locale in ("ru", "en"):
        out = summaries([ai_row, legacy, row(ALL_MET), row(MISSED)], locale)
        assert out[0] == out[1] == summaries([row(MET_BREACH)], locale)[0]
        for html in out:
            assert not re.search(r"\b(ИИ|AI|LLM)\b", text(html))
    assert summaries([row([]), {"domain": "final"}, None], "ru") == ["", "", ""]


def test_verdict_sentence_escapes_check_names():
    html = summaries([row([check("<x>", 1, 2, False)])], "ru")[0]
    assert "<x>" not in html and "&lt;x&gt;" in html


# ---- the verdict block ---------------------------------------------------------------------------------------------

STEPS = [
    {"index": i + 1, "start_iso": f"2026-10-20T15:{14 + 10 * i:02d}:00+03:00", "end_iso": f"2026-10-20T15:{24 + 10 * i:02d}:00+03:00",
     "plateau_start_iso": f"2026-10-20T15:{14 + 10 * i:02d}:00+03:00", "plateau_end_iso": f"2026-10-20T15:{23 + 10 * i:02d}:00+03:00",
     "rps": 50.0 * (i + 1), "latency_p95": 216.07 + 10 * i, "stable": True, "after_drop": False, "dip": i == 1, "selected": i == 3}
    for i in range(4)
] + [{"index": 5, "start_iso": "2026-10-20T15:54:00+03:00", "end_iso": "2026-10-20T16:14:00+03:00", "rps": 185.0, "latency_p95": 430,
      "stable": False, "after_drop": True, "selected": False}]
DEMO = {
    "domain": "final", "service": "demo-checkout", "test_type": "step", "verdict": "risk", "sla_verdict": "risk",
    "start_ms": 1792498440000, "end_ms": 1792502040000, "created_at": "2026-10-20T13:14:47.942685+00:00",
    "parsed": {"report_language": "ru", "peak_performance": {"max_rps": 249.3, "method": "last_stable_step", "max_time": "15:58", "drop_time": "15:59"}},
    "sla_details": {
        "checks": [
            check("target_rps", 200, 249.3, True, query_label="LT: RPS sum by all groups", series="all", message="RPS 249.3 ≥ целевой 200"),
            check("p95_latency", 300, 450, False, query_label="LT: p95 total", series="http_req_duration"),
            check("error_rate", 1, 0.4, True, message="Error rate 0.40% ≤ порог 1.0%"),
        ],
        "summary": "SLA verdict: Есть риски.",
        "stable_window": {"start": "2026-10-20T15:44:00+03:00", "end": "2026-10-20T15:53:00+03:00", "level": 200, "kind": "step"},
        "load_step_table": {"latency_query": "LT: p95 by group", "steps": STEPS},
    },
}


def hero(row_, locale="ru"):
    return run_js(REPORT_JS, f"{I}.heroHtml({js(row_)}, [])", locale=locale)


def test_verdict_block_has_the_rule_word_sentence_and_figures():
    html = hero(DEMO)
    assert '<div class="rep-verdict tone-warn" data-verdict="risk">' in html
    assert '<span class="rep-verdict-word">Есть риски</span>' in html
    assert '<p class="rep-verdict-why">Целевой RPS достигнут на стабильной ступени, но p95 выше порога: <b>450 мс</b>' in html
    assert "Определён по SLA-критериям." in html and "verdict-manual-tag" not in html
    figs = re.findall(r'<div class="fig rep-fig[^"]*" role="listitem" data-kpi="(\w+)">(.*?)</div></div>', html)
    assert [key for key, _ in figs] == ["rps", "latency", "errors"]
    assert '<div class="v">249.3<small>RPS</small></div>' in html and '<div class="v fail">450<small>мс</small></div>' in html
    assert '<div class="k ok"><abbr' in html and "цель ≥ 200 RPS <b>достигнута</b>" in html and "порог ≤ 300 мс <b>нарушен</b>" in html
    assert '<i class="fail" style="width:100%"></i><b style="left:66.7%"></b>' in html  # p95 bar: capped fill, tick at the threshold
    assert '<div class="rep-facts">' in html and "15:58" in html


def test_sla_criteria_without_a_box():
    html = hero(DEMO)
    assert "SLA-критерии · ступень 4 · 15:44–15:53" in html
    rows = re.findall(r'<tr class="is-(\w+)">(.*?)</tr>', html)
    assert [tone for tone, _ in rows] == ["ok", "fail", "ok"]
    assert '<span class="rep-glyph ok" role="img" title="Порог соблюдён" aria-label="Порог соблюдён">✓</span>' in rows[0][1]
    assert '<span class="rep-glyph fail"' in rows[1][1] and "✗</span>" in rows[1][1]
    assert '<td class="n th">≥ 200</td>' in rows[0][1] and '<td class="n th">≤ 300</td>' in rows[1][1]
    # Query and series as the second line; the stored message stays the tooltip; without a source the message shows.
    assert '<td class="name" title="RPS 249.3 ≥ целевой 200">' in rows[0][1] and '<span class="sub m">LT: RPS sum by all groups · all</span>' in rows[0][1]
    assert '<span class="sub m">Error rate 0.40% ≤ порог 1.0%</span>' in rows[2][1]
    assert '<div class="hero-sub rep-sla-note">SLA verdict: Есть риски.</div>' in html
    # Header cells exist for assistive technology only.
    assert '<th><span class="visually-hidden">Статус</span></th>' in html


def test_verdict_line_under_the_sla_table():
    html = hero(DEMO)
    line = re.search(r'<div class="rep-verdict-line" data-min-role="engineer">(.*?)</span></div>', html).group(1)
    assert line.index("Вердикт верен?") < line.index('data-vote="agree"') < line.index('data-verdict-action="change"')
    assert 'data-verdict-action="reset"' not in line
    manual = dict(DEMO, manual_verdict={"verdict": "pass", "reason": "сбой <генератора>", "author": "ivanov", "changed_at": "2026-10-17T12:00:00+00:00"},
                  verdict="pass", auto_verdict="risk")
    html = hero(manual)
    assert '<div class="rep-verdict tone-ok" data-verdict="pass">' in html
    assert '<span class="verdict-manual-tag">вручную</span>' in html and 'data-verdict-action="reset"' in html
    assert "Причина: сбой &lt;генератора&gt;" in html and "<генератора>" not in html


def test_meta_line_with_the_report_language_badge():
    meta = run_js(REPORT_JS, f"{I}.reportMetaHtml({js(DEMO)}, [])", locale="en", tz="Europe/Moscow")
    assert meta.startswith('<div class="rep-meta mono" data-report-meta role="group" aria-label="Run details">')
    plain = text(meta)
    assert plain.startswith("Step (max perf) · demo-checkout · 2026-10-20 15:14–16:14 UTC+03:00 · 1 h 00 min · report built 16:14")
    assert plain.endswith("Report language: Russian")
    assert "<b>15:14–16:14</b>" in meta
    ru = run_js(REPORT_JS, f"{I}.reportMetaHtml({js(DEMO)}, [])", locale="ru", tz="Europe/Moscow")
    assert "Язык отчёта" not in ru and "отчёт создан 16:14" in ru
    # The hero carries the meta line; renderHero moves it under the title.
    assert hero(DEMO).startswith('<div class="rep-meta mono" data-report-meta')


def test_figures_keep_the_tile_fields_and_add_bars():
    tiles = run_js(REPORT_JS, f"{I}.heroKpis({js(DEMO)})")
    assert [(t["key"], t["limit"], t["state"], t["bar"]["tone"]) for t in tiles] == [
        ("rps", "цель ≥ 200 RPS", "достигнута", "ok"), ("latency", "порог ≤ 300 мс", "нарушен", "fail"), ("errors", "порог ≤ 1 %", "соблюдён", "ok"),
    ]
    assert tiles[1]["note"] == "порог ≤ 300 мс · нарушен"
    text_tile = run_js(REPORT_JS, f"{I}.renderHeroKpis([{{ key: 'recovery', label: 'Восстановление', term: '', value: 'не восстановилось', unit: '', note: 'предел 5 мин', tone: 'fail', bar: null }}])")
    assert 'class="fig rep-fig is-text"' in text_tile and "Восстановление · предел 5 мин" in text(text_tile) and "fig-bar" not in text_tile
    assert run_js(REPORT_JS, f"{I}.renderHeroKpis([])") == ""


def test_no_final_row_keeps_the_message():
    assert run_js(REPORT_JS, f"{I}.heroHtml(null)") == (
        '<div class="rep-empty hero-empty">Итоговый отчёт для этого запуска ещё не сформирован. Если генерация идёт, '
        "обновите страницу позже; статус задачи виден в архиве.</div>")


# ---- «Ход теста» ---------------------------------------------------------------------------------------------------

def test_step_row_marks_the_sla_step_and_the_step_after_a_drop():
    out = run_js(REPORT_JS, f"""(() => {{
      const row = {js(DEMO)};
      const table = row.sla_details.load_step_table;
      const linked = {I}.linkSlaToSteps(table, row);
      return {{ html: {I}.renderLoadStepTableHtml(Object.assign({{}}, table, linked)), timeline: {I}.timelineHtml(Object.assign({{}}, table, linked)) }};
    }})()""")
    html = out["html"]
    cells = re.findall(r'<div class="(rep-step[^"]*)" role="listitem"([^>]*)>(.*?)</div>', html)
    assert [c[0] for c in cells] == ["rep-step", "rep-step", "rep-step", "rep-step is-sla", "rep-step is-bad"]
    assert text(cells[0][2]) == "1 · 50 RPS15:14–15:23 · p95 216.07 мс"
    assert text(cells[1][2]).endswith("кратковременная просадка")
    assert '<b>4 · 200 RPS<span class="rep-step-tag">SLA</span></b>' in cells[3][2]
    assert text(cells[4][2]) == "после просадки · 185 RPS15:54–16:14 · p95 430 мс"
    # Each cell carries the interval the charts highlight and grows with the step length (10 vs 20 minutes).
    first = cells[0][1]
    assert 'data-step-from="1792498440000" data-step-to="1792498980000" tabindex="0"' in first and 'style="flex-grow:10"' in first
    assert 'style="flex-grow:20"' in cells[4][1]
    assert '<div class="rep-steprow" role="list" aria-label="Ступени теста">' in html
    assert "запрос «LT: p95 by group», максимум по сериям запроса" in html and "Наведите на ступень" in html
    timeline = out["timeline"]
    assert "<h2>Ход теста</h2>" in timeline and "RPS ступеней · время UTC+3" in timeline
    assert '<div class="rep-profile" data-step-profile></div>' in timeline and "<details" not in timeline


def test_profile_options_come_from_the_markers_and_the_target():
    """The page draws LL.charts.stepProfile from the linked steps, the report markers (SLA band and degradation
    line) and the target RPS; the pure model accepts exactly that input."""
    model = run_js(REPORT_JS, f"""(() => {{
      const row = {js(DEMO)};
      row.sla_details.stable_window.degraded_checks = ['p95_latency'];
      const table = row.sla_details.load_step_table;
      const linked = {I}.linkSlaToSteps(table, row);
      const m = LL.charts.stepProfileModel({{ steps: linked.steps, markers: {I}.reportMarkers(row, table), target: 200, width: 1000 }});
      return {{ bands: m.bands.map((b) => [b.role, b.label]), lines: m.lines.map((l) => l.role), target: m.target.text }};
    }})()""")
    assert model["bands"] == [["sla", "SLA: ступень 4"]]
    assert model["lines"] == ["degradation"]  # the drop at the same minute merges into the degradation line
    assert model["target"] == "200 · цель"


def test_sla_scope_and_clock_offset():
    hold = dict(DEMO, sla_details=dict(DEMO["sla_details"], stable_window={"kind": "hold", "start": "2026-10-20T15:14:00+03:00", "end": "2026-10-20T16:14:00+03:00"}))
    spike = dict(DEMO, sla_details=dict(DEMO["sla_details"], stable_window={"kind": "spike"}))
    bare = {"domain": "final", "sla_details": {"checks": MET_BREACH}}
    out = run_js(REPORT_JS, f"[{js(DEMO)}, {js(hold)}, {js(spike)}, {js(bare)}, null].map((r) => {I}.slaScopeText(r))", locale="en")
    assert out == ["step 4 · 15:44–15:53", "hold window 15:14–16:14", "spike windows", "", ""]
    assert run_js(REPORT_JS, f"[3, -5, 5.5, 0, NaN].map((h) => {I}.utcOffsetText(h))") == ["UTC+3", "UTC-5", "UTC+5:30", "UTC+0", "UTC"]


def test_chart_axis_and_tooltip_follow_the_report_clock_not_the_browser():
    # The browser is in New York (UTC-4 in October), the steps of the report carry +03:00: the axis tick and the
    # tooltip title read 15:44 like the step row and the markers, not 08:44.
    label = "2026-10-20T12:44:00Z"
    out = run_js(REPORT_JS, f"""(() => {{
      {I}.reportState.stepTable = {js(DEMO["sla_details"]["load_step_table"])};
      const steps = [{I}.clockLabel({js(label)}), {I}.dateTimeAt(Date.parse({js(label)}), 3)];
      {I}.reportState.stepTable = {{ steps: [{{ start_iso: '2026-10-20T12:44:00-05:00' }}] }};
      const minus5 = {I}.clockLabel({js(label)});
      {I}.reportState.stepTable = null;
      return steps.concat([minus5, {I}.clockLabel({js(label)}), {I}.clockLabel('not a time')]);
    }})()""", tz="America/New_York")
    # Without a step table the report clock is UTC+3, as for the marker labels.
    assert out == ["15:44", "2026-10-20 15:44 UTC+3", "07:44", "15:44", "not a time"]


# ---- analysis tabs -------------------------------------------------------------------------------------------------

def test_findings_and_recommendations_in_two_columns_without_boxes():
    report = {"verdict": "risk", "findings": [{"id": "f1", "summary": "Latency grows", "severity": "high", "component": "orders-api"}],
              "recommended_actions": [{"summary": "Add a connection", "for_finding_ids": ["f1"]}]}
    html = run_js(REPORT_JS, f"{I}.renderStructuredReportHtml({js(report)}, '', 'final')", locale="en")
    block = html[html.index('<section class="rep-findings">'):]
    assert '<div class="rep-findings-grid"><div class="rep-findings-col lbl">Problems</div><div class="rep-findings-col lbl">How to fix</div>' in block
    assert '<div class="rep-f rep-f-problem" data-col="Problems">' in block and '<div class="rep-f rep-f-action" data-col="How to fix">' in block
    assert block.index("Latency grows") < block.index("Add a connection")
    assert "report-two-col-table" not in block and "report-table-shell" not in block


# ---- template, script and styles ---------------------------------------------------------------------------------

def test_template_header_and_sections():
    template = (REPO / "templates/reports.html").read_text(encoding="utf-8")
    assert "<style" not in template and "onclick" not in template
    actions = template[template.index('<div class="page-actions rep-actions">'):]
    order = [actions.index(f'id="{name}"') for name in ("renameReport", "forecastLink", "printReport", "publishConfluence", "confluencePageLink")]
    assert order == sorted(order)
    assert 'id="publishConfluence" class="btn btn-primary"' in template
    assert '<nav id="reportBreadcrumbs" class="breadcrumbs rep-crumbs"' in template
    assert '<section id="reportSteps" class="rep-timeline" hidden></section>' in template
    assert template.index('id="reportHero"') < template.index('id="reportSteps"') < template.index('id="rep-llm-tabs"')
    # Line endings are not mixed: CRLF in a Windows checkout with core.autocrlf, LF in a Linux checkout (CI).
    assert (REPO / "templates/reports.html").read_bytes().count(b"\r\n") in (0, template.count("\n"))


def test_reports_js_draws_the_profile_and_keeps_tokens():
    source = (REPO / "static/js/reports.js").read_text(encoding="utf-8")
    assert "LL.charts.stepProfile(host," in source and "LL.charts.setHighlight(" in source
    assert "#12151c" not in source and "#888" not in source
    assert "report-collapsible\">',\n      `<summary><span>${escapeHtml(LL.t('report.steps.title'))}" not in source
    # Keys of the sentence are literal: no dynamic report.summary.* keys (tests/test_i18n.py DYNAMIC_PREFIXES).
    assert "'report.summary.' +" not in source and "`report.summary.${" not in source


def _css():
    return (REPO / "static/css/app.css").read_text(encoding="utf-8")


def _section():
    match = re.search(r"/\* ===== page: report ===== \*/(.*?)/\* ===== end page: report ===== \*/", _css(), re.S)
    assert match, "the report section of app.css is missing"
    return match.group(1)


def test_stage1_css_uses_tokens_only_and_covers_the_blocks():
    section = _section()
    for selector in (".rep-verdict", ".rep-verdict-word", ".rep-sla-table", ".rep-verdict-line", ".rep-steprow", ".rep-step.is-sla",
                     ".rep-step.is-bad", ".rep-findings-grid", ".report-chart", "@media print"):
        assert selector in section, selector
    # Tokens only: the print block (light tokens, checked against the light theme in tests/test_ui_report_js.py) holds
    # the only literal colors.
    screen = section[:section.index("@media print")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b(?![\w-])", re.sub(r"#rep-[\w-]+|#renameReport\w*|#pageTitle|#pageTimeRange|#confluenceStatus|#engineer\w+|#publishConfluence", "", screen))
    assert not re.search(r"rgba?\(", screen)
    assert re.search(r"\.rep-verdict \{[^}]*border-top: 3px solid var\(--verdict\)", section)
    assert re.search(r"\.rep-verdict-word \{[^}]*font-size: 34px", section)
    # The chart legend is a list (LL.charts.legendChips, shared with the compare page): a 14x2 dash, no chip box.
    assert re.search(r"\n\.legend-chip-swatch \{[^}]*width: 14px; height: 2px;[^}]*border: 0;", _css())
    assert re.search(r"\n\.legend-chip \{[^}]*border: 0;[^}]*background: none;", _css())
    media = section[section.index("@media print"):]
    assert ".rep-verdict-line" in media and "break-inside: avoid" in media and "print-color-adjust: exact" in media


@pytest.mark.parametrize("locale", ["ru", "en"])
def test_new_keys_exist_in_both_languages(locale):
    data = json.loads((REPO / "locales" / locale / "report.json").read_text(encoding="utf-8"))
    for key in ("report.summary.target_met", "report.summary.but", "report.summary.with_errors", "report.summary.name.error_rate",
                "report.timeline.title", "report.timeline.caption", "report.steps.cell_p95", "report.sla.scope_step", "report.meta.created",
                "report.charts.zoom_hint_short", "report.unit.min"):
        assert data[key], key
