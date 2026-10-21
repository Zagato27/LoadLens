# SPDX-License-Identifier: Apache-2.0
"""Localization of the report page (plans/i18n-plan.md 2.5, row 2): templates/reports.html, static/js/reports.js and
their Python twins AI/report_sections.py (Confluence sections) and AI/report_markers.py (chart markers, thresholds).

Page chrome and everything built at display time (titles, table headers, markers, thresholds, names of stored codes:
test type, focus, notes, failure modes, metrics, SLA criteria) follow the UI locale. Texts stored with a report when it
was built (LLM findings and rationale, SLA check messages and summaries, step and spike labels, notes of reports saved
before the codes) are shown as stored, and a badge names the report language when it differs from the UI language
(``report_language`` of the final row; a report without it is Russian). Without a cookie the Russian output stays as it
was (the existing Russian assertions in test_ui_report_js.py, test_report_markers.py and test_test_types.py).
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from loadlens_app import core  # noqa: F401  (before anything that imports update_page)
from AI import report_markers, report_sections
from AI.i18n import with_locale
from AI.saturation import FAILURE_MODE_TEXTS, SATURATION_NOTE_TEXTS
from AI.spike import SPIKE_NOTE_TEXTS
from AI.stability import HOLD_NOTE_TEXTS
from AI.test_profile import TEST_TYPE_TEXTS
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text
from tests.js_harness import REPO_ROOT, run_js

REPORT_JS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
I = "LL.report._internals"
# The name of a language is shown in that language (common.lang.ru), on purpose.
LANGUAGE_NAMES = ("Русский",)


def js(value):
    return json.dumps(value, ensure_ascii=False)


def stored_texts(value) -> list[str]:
    """Every string of a stored row that holds Cyrillic: texts saved with a ru report, shown as stored."""
    if isinstance(value, str):
        return [value] if CYRILLIC.search(value) else []
    if isinstance(value, dict):
        return [text for item in value.values() for text in stored_texts(item)]
    if isinstance(value, list):
        return [text for item in value for text in stored_texts(item)]
    return []


def plain(value) -> str:
    """Visible text of builder output: HTML strings through visible_text, nested values flattened."""
    if isinstance(value, str):
        return visible_text(value) if "<" in value else value
    if isinstance(value, dict):
        return "\n".join(plain(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(plain(item) for item in value)
    return ""


# ---- rows: a legacy ru report (no report_language, Russian stored texts) and an en report ---------------------------

def _steps(label):
    out = []
    for i in range(5):
        hour = 17 + (26 + 10 * i) // 60
        minute = (26 + 10 * i) % 60
        end_hour = 17 + (36 + 10 * i) // 60
        end_minute = (36 + 10 * i) % 60
        out.append({
            "index": i + 1, "label": f"{label} {i + 1}", "start_iso": f"2026-10-15T{hour:02d}:{minute:02d}:00+03:00",
            "end_iso": f"2026-10-15T{end_hour:02d}:{end_minute:02d}:00+03:00", "stable": i != 2, "after_drop": i == 4, "dip": i == 1,
            "rps": 50.0 * (i + 1), "latency_p95": 200.0 + i, "selected": i == 3,
        })
    return out


def stress_row(lang):
    ru = lang == "ru"
    steps = _steps("ступень" if ru else "step")
    checks = [
        {"name": "target_rps", "threshold": 180, "actual": 200.0, "passed": True,
         "message": "RPS 200.0 ≥ целевой 180; label='RPS total'; series='all'" if ru else "RPS 200.0 ≥ target 180; label='RPS total'; series='all'"},
        {"name": "p95_latency", "threshold": 300, "actual": 280.5, "passed": True,
         "message": "p95 280.50 мс ≤ порог 300 мс (запрос «Latency», серия «POST /x»)" if ru else "p95 280.50 ms ≤ threshold 300 ms (query «Latency», series «POST /x»)"},
        {"name": "error_rate", "threshold": 1.5, "actual": 2.2, "passed": False, "message": "ошибки" if ru else "errors", "query_label": "Errors %"},
        {"name": "memory_usage", "threshold": 80, "actual": None, "passed": None, "message": "", "query_label": "Memory"},
    ]
    parsed = {
        "verdict": "risk",
        "verdict_rationale": "Статус «Есть риски»: задержки растут" if ru else 'Status "Risks found": latency grows',
        "test_profile": {"mode": "stress", "test_type": "stress", "normalized_test_type": "stress"},
        "saturation": {
            "saturation_rps": 200, "failure_mode": "latency+errors", "note_code": "limit_not_reached",
            "saturation_step_label": steps[3]["label"] + " · ≈200 RPS", "degraded_step_label": steps[4]["label"],
            "degraded_checks": ["p95_latency", "error_rate"], "rps_drop_iso": "2026-10-15T18:12:00+03:00",
            "saturation_window": {"start_iso": steps[3]["start_iso"], "end_iso": steps[3]["end_iso"]},
            "limiting_resource": {"kind": "memory", "evidence": "threshold", "value": 91.5, "threshold": 80, "series": "node-1"},
        },
        "peak_performance": {"max_rps": 200, "max_time": "18:00", "drop_time": "18:12"},
        "findings": [{"id": "f1", "summary": "Рост задержек" if ru else "Latency grows", "severity": "high", "component": "db",
                      "evidence_items": [{"metric": "p95", "observed_value": "450", "threshold": "300"}],
                      "verification": {"status": "unverified", "unmatched": ["x"]}}],
        "recommended_actions": [{"summary": "Добавить индекс" if ru else "Add an index", "for_finding_ids": ["f1"]}],
        "verification_summary": {"unverified": 3, "revised_by_model": False},
    }
    if not ru:
        parsed["report_language"] = "en"
    return {
        "domain": "final", "service": "checkout", "test_type": "stress", "verdict": "risk", "start_ms": 1792059960000, "end_ms": 1792063560000,
        "manual_verdict": {"verdict": "pass", "reason": "generator fault", "author": "ivanov", "changed_at": "2026-10-17T12:00:00+00:00"},
        "auto_verdict": "risk",
        "scores": {"run_usage": {"calls": 9, "prompt_tokens": 15300, "completion_tokens": 2100},
                   "judge": {"overall": 0.9, "factual": 0.8, "completeness": 0.7, "specificity": 0.6,
                             "rubric": {"evidence_grounding": 0.5, "issue_coverage": 0.4, "specificity": 0.3, "sla_alignment": 0.2, "actionability": 0.1}},
                   "data_score": 0.8, "confidence": 0.7, "final_score": 0.85,
                   "data_score_details": {"label_grounding": 0.5, "numeric_grounding": 0.6, "numeric_claims_total": 5, "numeric_claims_supported": 3,
                                          "peak_checked": True, "peak_consistency": 0.9, "recommendation_grounding": 0.4},
                   "judge_meta": {"used_safe_path": True, "context_truncated": True, "truncated_candidates": 2}},
        "parsed": parsed,
        "sla_details": {
            "checks": checks, "summary": "SLA verdict: Есть риски." if ru else "SLA verdict: Risks found.",
            "stable_window": {"start": steps[3]["start_iso"], "end": steps[3]["end_iso"], "level": 210.0, "kind": "step", "degraded_checks": ["p95_latency"]},
            "load_step_table": {"rps_query": "RPS total", "latency_query": "Latency", "latency_series": None, "steps": steps},
        },
    }


def soak_report(lang):
    ru = lang == "ru"
    return {
        "test_profile": {"mode": "stability", "test_type": "soak", "normalized_test_type": "soak"},
        "stability_under_load": {
            "target_rps": 180, "actual_rps": 182.4, "sla_summary": "Итог: держит" if ru else "Summary: holds", "trend_threshold_pct": 15,
            "hold": {"start_iso": "2026-10-15T15:42:00+03:00", "end_iso": "2026-10-15T16:29:00+03:00", "duration_min": 47, "min_bucket_rps": 171,
                     "min_bucket_label": "интервал 3" if ru else "interval 3", "drop_pct": 6.2, "early_drop_iso": "2026-10-15T16:20:00+03:00",
                     "note_codes": ["early_drop", "long_ramp_up"]},
            "trends": [{"metric_kind": "error_rate", "title": "доля ошибок" if ru else "error rate", "first": 0.1, "last": 0.4, "change_pct": 300,
                        "slope_per_hour": 0.3, "flagged": True, "label": "Errors %"},
                       {"metric_kind": "memory", "title": "память" if ru else "memory", "first": 50, "last": 52, "change_pct": 4, "slope_per_hour": 2}],
        },
    }


def spike_report(lang):
    ru = lang == "ru"
    return {
        "test_profile": {"mode": "spike", "test_type": "spike", "normalized_test_type": "spike"},
        "spike_analysis": {
            "detected": True, "baseline_rps": 60, "max_spike_rps": 250, "target_rps": 250, "target_reached": False, "worst_spike_index": 2,
            "sla_summary": "Итог SLA: риск" if ru else "SLA summary: risk",
            "spikes": [{"index": 1, "label": "всплеск 1 · ≈250 RPS" if ru else "spike 1 · ≈250 RPS", "start_iso": "2026-10-15T17:38:00+03:00",
                        "end_iso": "2026-10-15T17:42:00+03:00", "duration_sec": 240, "pre": {"p95": 100}, "during": {"p95": 140}, "overshoot_p95_pct": 40,
                        "overshoot_error_rate_pp": 1.5, "recovered": True, "recovery_time_sec": 60, "note_code": "no_latency_after"},
                       {"index": 2, "label": "", "recovered": False, "note_code": "no_latency_query", "overshoot_p95_pct": 55}],
        },
    }


CONTEXT = {
    "system": {"name": "Shop", "domain": "retail", "description": "Checkout", "test_goal": "max RPS"},
    "architecture": {"style": "microservices", "components": [{"name": "api", "role": "gateway", "criticality": "high", "technologies": ["go"]}],
                     "dependencies": [{"from": "api", "to": "db", "kind": "sql", "purpose": "orders"}]},
    "load_model": {"critical_user_flows": [{"name": "buy", "steps": ["cart", "pay"], "success_signals": ["200"]}],
                   "entrypoints": [{"name": "/buy", "kind": "http", "business_priority": "p1"}], "expected_hotspots": ["db"]},
    "operational_context": {"known_constraints": ["c"], "known_risks": ["r"], "normal_degradation_rules": ["n"], "analysis_focus": ["f"]},
}


def builders(row, locale, extra_sections=()):
    """Output of every pure builder of the page for ``row`` (and test-type payloads) in ``locale``."""
    table = row["sla_details"]["load_step_table"]
    sections = [row["parsed"], *extra_sections]
    state = {"verdict": "pass", "automatic": "risk", "manual": row["manual_verdict"], "reason_max": 5,
             "history": [{"verdict": None, "previous_verdict": "risk", "auto_verdict": "fail", "reason": "", "author": "p", "changed_at": "2026-10-17T13:00:00+00:00"},
                         {**row["manual_verdict"], "previous_verdict": "fail", "auto_verdict": "fail"}]}
    expr = f"""(() => {{
      const I = {I};
      const row = {js(row)};
      const table = {js(table)};
      const linked = I.linkSlaToSteps(table, row);
      const sections = {js(sections)};
      return {{
        hero: I.heroHtml(row, [row]),
        noHero: I.heroHtml(null),
        sections: sections.map((s) => I.testTypeSection(s)),
        kpis: sections.map((s) => I.heroKpis(Object.assign({{}}, row, {{ parsed: s }}))).map((tiles) => tiles.map((t) => [t.label, t.value, t.unit, t.note])),
        types: sections.map((s) => I.heroTypeEntries(s, s.peak_performance || {{}}).map((e) => [e.label, e.value, e.note])),
        trends: sections.map((s) => I.renderTrendsTable(s)),
        markers: I.reportMarkers(row, table).map((m) => m.label),
        thresholds: I.slaThresholds(row, table).map((t) => t.label),
        steps: I.renderLoadStepTableHtml(Object.assign({{}}, table, linked)),
        holdNote: I.linkSlaToSteps(table, Object.assign({{}}, row, {{ sla_details: Object.assign({{}}, row.sla_details, {{ stable_window: {{ kind: 'hold', start: table.steps[0].start_iso, end: table.steps[4].end_iso, level: 120 }} }}) }})).slaNote,
        spikeNote: I.linkSlaToSteps(table, Object.assign({{}}, row, {{ sla_details: Object.assign({{}}, row.sla_details, {{ stable_window: {{ kind: 'spike' }} }}) }})).slaNote,
        sla: I.renderSlaChecklist(row),
        emptySla: I.renderSlaChecklist({{ domain: 'final' }}),
        structured: I.renderStructuredReportHtml(row.parsed, I.judgeSectionHtml(row.scores), 'final'),
        domain: I.renderStructuredReportHtml({{ verdict: 'pass', peak_performance: {{ max_rps: 5 }}, findings: [], recommended_actions: [] }}, '', 'lt_framework'),
        failed: I.renderStructuredReportHtml({{ analysis_error: {{ provider: 'openai', max_tokens: 100, output_tokens: 0 }} }}, '', 'final'),
        judges: [{{ judge: {{}}, judge_meta: {{ skipped: true }}, data_score: 0.6, confidence: 0.5 }},
                 {{ judge: {{ overall: 0 }}, judge_meta: {{ error: 'x' }}, data_score: 0.3 }},
                 {{ judge: {{ overall: 0 }}, judge_meta: {{ reason: 'partial_scores' }}, data_score: 0.8 }},
                 {{ judge: {{ overall: 0.5 }}, data_score: 0.9, final_score: 0.2, data_score_details: {{ numeric_claims_total: 0 }} }}].map((s) => I.judgeSectionHtml(s)),
        context: I.systemContextMarkdown({js(CONTEXT)}),
        invalid: I.invalidStructuredResponseMarkdown('{{"verdict": 1'),
        usage: [I.formatUsageLine({{ calls: 9, prompt: 15300, completion: 2100 }}), I.formatUsageLine({{ calls: 2 }})],
        verdicts: [row, {{ sla_verdict: 'fail', llm_verdict: 'pass' }}, {{ sla_verdict: 'risk' }}, {{ verdict: 'no_data' }}, {{ verdict: 'pass' }},
                   {{ manual_verdict: {{ verdict: 'fail' }}, auto_verdict: 'pass' }}].map((r) => I.verdictSourceText(r)),
        dialog: [I.verdictDialogHtml({js(state)}), I.verdictDialogHtml({{ verdict: 'risk', automatic: 'risk' }})],
        problems: [I.verdictFormProblem('', 'x', {{}}), I.verdictFormProblem('fail', '', {{}}), I.verdictFormProblem('fail', 'too long', {{ reason_max: 3 }})],
        stored: I.STORED_TEXTS
      }};
    }})()"""
    return run_js(REPORT_JS, expr, locale=locale)


# ---- dictionaries and the Python tables of stored codes -----------------------------------------------------------

def _python_tables(lang):
    return {
        "typeFocus": {key: texts["focus"] for key, texts in TEST_TYPE_TEXTS[lang].items()},
        "holdNotes": HOLD_NOTE_TEXTS[lang],
        "spikeNotes": SPIKE_NOTE_TEXTS[lang],
        "saturationNotes": SATURATION_NOTE_TEXTS[lang],
        "failureModes": FAILURE_MODE_TEXTS[lang],
    }


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_texts_of_stored_codes_equal_the_python_tables(lang):
    """report.focus.*, report.hold_note.*, report.spike_note.*, report.saturation_note.*, report.failure_mode.* hold
    the texts of AI/test_profile.py, AI/stability.py, AI/spike.py and AI/saturation.py (code 'latency+errors' ->
    key segment 'latency_errors'); the page reads them in the UI locale."""
    entries = json.loads((REPO_ROOT / "locales" / lang / "report.json").read_text(encoding="utf-8"))
    prefixes = {"typeFocus": "report.focus.", "holdNotes": "report.hold_note.", "spikeNotes": "report.spike_note.",
                "saturationNotes": "report.saturation_note.", "failureModes": "report.failure_mode."}
    tables = _python_tables(lang)
    for group, table in tables.items():
        from_dictionary = {key[len(prefixes[group]):]: text for key, text in entries.items() if key.startswith(prefixes[group])}
        assert from_dictionary == {re.sub(r"[^a-z0-9_]+", "_", code): text for code, text in table.items()}, group
    assert run_js(REPORT_JS, f"{I}.STORED_TEXTS", locale=lang) == tables


# ---- the page ------------------------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(dashboard, "_lookup_report_ref",
                        lambda run_id="", run_name="", service="": {"run_id": "abc", "run_name": "nightly", "service": "checkout"} if run_id == "abc" else None)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _page(client, lang):
    use_locale(client, lang)
    response = client.get("/reports/abc")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1).strip()
    main = html[html.index('<main id="main"'):html.index("</main>")]
    return html, title, main[main.index(">") + 1:]


def test_report_page_in_en_has_no_russian_text(client):
    html, title, page = _page(client, "en")
    assert re.search(r'<html lang="en"', html)
    assert title == "Report"
    assert_no_cyrillic(title + "\n" + visible_text(page))
    assert 'aria-label="Section navigation"' in page and '<a href="/reports">Archive</a>' in page
    assert '<h1 id="pageTitle">Test report</h1>' in page and ">Rename</button>" in page
    assert 'placeholder="Report name"' in page and ">Save</button>" in page and ">Cancel</button>" in page
    # Short labels keep the meta line on one row at 1440px; the hint names the PDF option.
    assert '<div id="pageTimeRange">Test time: —</div>' in page and ">Print</button>" in page
    assert ">To Confluence</button>" in page and ">Metric charts</h3>" in page
    assert ">• List</button>" in page and ">1. List</button>" in page
    # The page declares the report and charts namespaces for reports.js and charts.js.
    assert '"report.page.title_run":"Test report {run}"' in html


def test_report_page_stays_russian_without_a_cookie(client):
    html, title, page = _page(client, None)
    assert re.search(r'<html lang="ru"', html) and title == "Отчёт"
    assert 'aria-label="Навигация по разделу"' in page and '<h1 id="pageTitle">Отчет по тесту</h1>' in page
    assert 'placeholder="Название отчёта" aria-label="Новое название отчёта"' in page
    assert 'title="Все вкладки, графики в светлой теме; в диалоге печати можно выбрать «Сохранить как PDF»">Печать</button>' in page
    assert 'class="btn btn-primary" data-min-role="engineer">В Confluence</button>' in page
    assert '<h3 style="margin:0;">Итоги от инженера</h3>' in page and '<h3 class="section-title">Графики метрик</h3>' in page
    assert '"report.page.title_run":"Отчет по тесту {run}"' in html


# ---- JS builders -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("report_lang", ["en", "ru"])
def test_js_builders_in_en_have_no_russian_text(report_lang):
    """An en report and a legacy ru report in the en UI: only texts stored with the ru report stay Russian."""
    row = stress_row(report_lang)
    extra = [soak_report(report_lang), spike_report(report_lang)]
    out = builders(row, "en", extra)
    allow = stored_texts(row) + stored_texts(extra) + list(LANGUAGE_NAMES)
    assert_no_cyrillic(plain(out), allow=allow)
    if report_lang == "en":
        assert stored_texts(row) == [] and "Report language" not in plain(out["hero"])
    else:
        assert "Report language: Russian" in plain(out["hero"])


def test_js_builders_speak_english():
    out = builders(stress_row("en"), "en", [soak_report("en"), spike_report("en")])
    stress, soak, spike = out["sections"]
    assert stress["title"] == "Saturation point"
    assert {r["label"]: r["value"] for r in stress["rows"]}["Degradation pattern"] == "latency and error growth"
    assert {r["label"]: r["value"] for r in stress["rows"]}["Breached criteria"] == "P95 latency, ms, Error rate, %"
    assert {r["label"]: r["value"] for r in stress["rows"]}["Limiting resource"] == "memory: p95 91.5 > threshold 80, series “node-1”"
    soak_rows = {r["label"]: r["value"] for r in soak["rows"]}
    assert soak["title"] == "Stability under load" and soak_rows["Test type"] == "Soak (long-running)"
    assert soak_rows["Hold window"] == "15:42–16:29 (47 min)" and soak_rows["Lowest interval"] == "interval 3: 171 RPS (−6.2 % of the mean)"
    assert soak_rows["Note"] == ("RPS dropped long before the end of the test: the drop is inside the hold window; "
                                 "the ramp-up took more than half of the test: the hold window is the whole test")
    spike_rows = {r["label"]: r["value"] for r in spike["rows"]}
    assert spike_rows["Spike target RPS"] == "250 (not reached)"
    assert spike_rows["spike 1 · ≈250 RPS"].startswith("17:38–17:42 (240 s), p95 100 → 140 (+40 %), errors +1.5 pp, recovered in 60 s")
    assert spike_rows["spike"].endswith("no p95/p99 query is chosen in the SLA: latency growth and recovery are not evaluated")
    assert out["kpis"][0] == [["Saturation point", "200", " RPS", "step 4 · ≈200 RPS"], ["p95", "280.5", " ms", "threshold ≤ 300 ms · met"],
                              ["Errors", "2.2", " %", "threshold ≤ 1.5 % · breached"]]
    assert out["kpis"][1][0] == ["Mean RPS", "182.4", " RPS", "target ≥ 180 RPS · held · 15:42–16:29"]
    assert out["kpis"][2][2][1] == "not recovered: 1 of 2"
    assert out["markers"][0] == "SLA: step 4" and out["thresholds"] == ["target ≥ 180 RPS", "p95 ≤ 300 ms", "errors ≤ 1.5 %", "memory ≤ 80 %"]
    assert out["holdNote"].startswith("SLA checked on the hold window 17:26–18:16, mean RPS 120. Target RPS 200 — passed")
    assert out["spikeNote"].startswith("SLA checked in the spike windows: the worst spike for each threshold.")
    assert "query “Latency”, maximum across the query series." in plain(out["steps"])
    assert out["verdicts"][:3] == ["Changed manually: ivanov, 2026-10-17 12:00. Automatic: Risks found.",
                                   "Set by the SLA criteria. AI assessment: Passed.", "Set by the SLA criteria."]
    assert out["problems"] == ["Choose a new verdict", "Give a reason: it is shown in the report, the archive and Confluence",
                               "The reason must be at most 3 characters"]
    assert out["usage"] == ["LLM calls: 9 · tokens: 15k / 2.1k", "LLM calls: 2"]
    assert "3 findings are not confirmed by data and are not counted in the verdict rationale (the model's verification pass did not run)." in plain(out["hero"])
    assert 'You can set thresholds in <a href="/settings">Settings</a> (the “SLA criteria” section).</div>' in out["emptySla"]


def test_unverified_findings_use_plural_forms():
    def notes(locale):
        expr = f"""[1, 3, 5, 21].map((n) => {I}.heroHtml(Object.assign({js(stress_row("ru"))}, {{
          parsed: Object.assign({js(stress_row("ru")["parsed"])}, {{ verification_summary: {{ unverified: n, revised_by_model: true }} }}) }})))"""
        # The note is the text of a glossary term (its tooltip is the glossary's business).
        return [re.search(r'<div class="hero-sub hero-unverified"><abbr [^>]*>([^<]*)</abbr></div>', html).group(1) for html in run_js(REPORT_JS, expr, locale=locale)]

    ru = notes("ru")
    assert ru == [f"{n} {word} не подтверждены данными и не учтены в обосновании вердикта." for n, word in
                  ((1, "находка"), (3, "находки"), (5, "находок"), (21, "находка"))]
    en = notes("en")
    assert en[0] == "1 finding is not confirmed by data and is not counted in the verdict rationale."
    assert en[1] == "3 findings are not confirmed by data and are not counted in the verdict rationale."


def test_report_language_badge_shows_when_the_report_language_differs():
    en_row, ru_row = stress_row("en"), stress_row("ru")
    expr = f"""[{I}.reportLanguageOf({js(en_row)}), {I}.reportLanguageOf({js(ru_row)}),
                {I}.reportLanguageOf({{ parsed: '{{"report_language": "EN-gb"}}' }}), {I}.reportLanguageOf({{ sla_details: {{ report_language: 'en' }} }}),
                {I}.reportLanguageOf({{ parsed: {{ report_language: 'de' }} }}),
                {I}.reportLanguageBadge({js(en_row)}), {I}.reportLanguageBadge({js(ru_row)})]"""
    ru_ui = run_js(REPORT_JS, expr, locale="ru")
    en_ui = run_js(REPORT_JS, expr, locale="en")
    assert ru_ui[:5] == en_ui[:5] == ["en", "ru", "en", "en", "ru"]
    # The language is named in the UI language (stage 5: common.lang_name.*), so the badge carries no lang attribute.
    badge = re.compile(r'<span class="pill na" data-report-language="(\w+)" title="([^"]*)">([^<]*)</span>')
    assert badge.search(ru_ui[5]).group(1, 3) == ("en", "Язык отчёта: английский") and ru_ui[6] == ""
    assert en_ui[5] == "" and badge.search(en_ui[6]).group(1, 2, 3) == (
        "ru", "AI texts, SLA summaries and step labels are shown in the language the report was built in", "Report language: Russian")
    assert ' lang="' not in en_ui[6] and 'class="pill na"' in en_ui[6]


def test_trend_titles_come_from_the_metric_code():
    """A soak report stores the trend title in its language; the page names the metric in the UI language."""
    html = run_js(REPORT_JS, f"{I}.renderTrendsTable({js(soak_report('ru'))})", locale="en")
    text = visible_text(html)
    assert "error rate" in text and "memory" in text and "доля ошибок" not in text
    assert "Risk: an adverse change of more than 15 %." in text


# ---- Python twins (Confluence) -------------------------------------------------------------------------------------

@pytest.mark.parametrize("report_lang", ["en", "ru"])
def test_python_sections_follow_the_locale(report_lang):
    reports = [stress_row(report_lang)["parsed"], soak_report(report_lang), spike_report(report_lang)]
    with with_locale("en"):
        sections = [report_sections.test_type_section(report) for report in reports]
        markers = report_markers.markers_for(stress_row(report_lang))
    assert [title for title, _rows in sections] == ["Saturation point", "Stability under load", "Load spikes"]
    assert_no_cyrillic(json.dumps([sections, markers], ensure_ascii=False), allow=stored_texts(reports) + stored_texts(stress_row(report_lang)))
    stress = dict(sections[0][1])
    assert stress["Degradation pattern"] == "latency and error growth"
    assert stress["Limiting resource"] == "memory: p95 91.5 > threshold 80, series “node-1”"
    soak = dict(sections[1][1])
    assert soak["Assessment focus"] == TEST_TYPE_TEXTS["en"]["soak"]["focus"]
    assert soak["Trends over the hold window"] == "error rate: +300 % (risk); memory: +4 %"
    spike = dict(sections[2][1])
    assert spike["Test type"] == "Spike" and spike["spike"].endswith(SPIKE_NOTE_TEXTS["en"]["no_latency_query"])


def test_python_sections_stay_russian_by_default():
    title, rows = report_sections.test_type_section(soak_report("ru"))
    assert title == "Стабильность под нагрузкой"
    rows = dict(rows)
    assert rows["Окно удержания"] == "15:42–16:29 (47 мин)"
    assert rows["Минимальный интервал"] == "интервал 3: 171 RPS (−6.2 % от среднего)"
    assert rows["Тренды за окно удержания"] == "доля ошибок: +300 % (риск); память: +4 %"
    stress = dict(report_sections.test_type_section(stress_row("ru")["parsed"])[1])
    assert stress["Ограничивающий ресурс"] == "память: p95 91.5 > порог 80, серия «node-1»"
    with with_locale("en"), with_locale("ru"):
        assert report_sections.test_type_section(soak_report("ru"))[0] == "Стабильность под нагрузкой"


def test_confluence_publishing_in_the_report_language_renders_english_sections():
    """Confluence wraps publishing in with_locale(report_language); the section of an en report is then English."""
    copy_row = copy.deepcopy(stress_row("en"))
    with with_locale("en"):
        title, rows = report_sections.test_type_section(copy_row["parsed"])
        thresholds = [item["label"] for item in report_markers.thresholds_for(copy_row)]
    assert title == "Saturation point" and dict(rows)["Saturation point"] == "200 RPS"
    assert thresholds == ["target ≥ 180 RPS", "p95 ≤ 300 ms", "errors ≤ 1.5 %", "memory ≤ 80 %"]
