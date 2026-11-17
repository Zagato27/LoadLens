# SPDX-License-Identifier: Apache-2.0
"""Regression against the baseline on the report page, in the settings form and in Confluence; parity of check names."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from AI import regression as rg  # noqa: E402
from AI.i18n import with_locale  # noqa: E402
from AI.sla_evaluator import SLA_CHECK_NAMES  # noqa: E402
from loadlens_app import confluence_export  # noqa: E402
from tests.js_harness import run_js  # noqa: E402

REPORT_JS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
SETTINGS_JS = ["static/js/common.js", "static/js/ui.js", "static/js/settings.js"]
I = "LL.report._internals"
CYRILLIC = re.compile("[А-Яа-яЁё]")
LOCALES = ROOT / "locales"


def js(value):
    return json.dumps(value, ensure_ascii=False)


def _messages(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted((LOCALES / lang).glob("*.json")):
        out.update(json.loads(path.read_text(encoding="utf-8")))
    return out


REGRESSION_CHECKS = [
    {"name": "regression_latency", "threshold": 10, "actual": 25.0, "passed": False, "severity": "warning", "category": "regression",
     "unit": "pct", "message": "p95 150 мс против 120 мс в «run-base» на ступени цели ≈600 RPS: +25.0 % (порог риска +10 %)"},
    {"name": "regression_errors", "threshold": 0.5, "actual": 0.1, "passed": True, "severity": "warning", "category": "regression",
     "unit": "pp", "message": "доля ошибок 0.2 % против 0.1 % в «run-base»"},
    {"name": "regression_capacity", "threshold": 10, "actual": -4.0, "passed": True, "severity": "warning", "category": "regression",
     "unit": "pct", "message": "пороги задержек и ошибок выполняются до ≈840 RPS против ≈800 RPS в «run-base»"},
]
BASE_CHECKS = [
    {"name": "target_rps", "threshold": 500, "actual": 840, "passed": True, "message": ""},
    {"name": "p95_latency", "threshold": 500, "actual": 300, "passed": True, "message": "", "query_label": "LT p95"},
]


def _regression(**extra):
    return {
        "available": True, "baseline_run_name": "run-base", "baseline_run_id": "id-base", "baseline_release": "",
        "mode": "pinned", "kind": "target", "level": 600, "items": [], "skipped": [], **extra,
    }


def _final_row(regression=None, checks=None):
    details = {"checks": checks if checks is not None else [*BASE_CHECKS, *REGRESSION_CHECKS], "summary": "SLA verdict: Есть риски."}
    if regression is not None:
        details["regression"] = regression
    return {"domain": "final", "run_name": "run-new", "verdict": "risk", "parsed": {}, "sla_details": details}


# ---- parity of the check names -------------------------------------------------------------------------------------


def test_check_names_match_between_python_js_and_confluence():
    names = run_js(REPORT_JS, f"{I}.SLA_CHECK_NAMES")
    assert names == list(SLA_CHECK_NAMES)
    assert confluence_export.SLA_CHECK_NAMES == SLA_CHECK_NAMES
    assert set(rg.REGRESSION_CHECK_NAMES) <= set(SLA_CHECK_NAMES)


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_every_check_name_has_its_titles(lang):
    messages = _messages(lang)
    for name in SLA_CHECK_NAMES:
        for prefix in ("sla.check_title.", "report.sla_check.", "report.summary.name."):
            assert f"{prefix}{name}" in messages, f"{prefix}{name} is missing in {lang}"


def test_check_labels_and_units():
    ru = run_js(REPORT_JS, f"[{I}.slaCheckLabel('regression_latency'), {I}.checkUnit('regression_errors'), {I}.checkUnit('regression_capacity')]")
    assert ru == ["p95 к базе, рост в %", " п.п.", " %"]
    en = run_js(REPORT_JS, f"[{I}.slaCheckLabel('regression_errors'), {I}.checkUnit('regression_errors')]", locale="en")
    assert en == ["Errors against the baseline, pp growth", " pp"]


def test_bar_of_a_regression_check_draws_an_improvement_as_empty():
    bars = run_js(REPORT_JS, f"[{', '.join(f'{I}.slaBarModel({js(check)})' for check in REGRESSION_CHECKS)}]")
    assert bars[0] == {"fill": 100, "mark": 66.7, "tone": "fail", "higherBetter": False, "percent": 250}
    assert bars[1]["fill"] == 13.3 and bars[1]["tone"] == "ok"
    assert bars[2]["fill"] == 0 and bars[2]["percent"] == 0
    # Other checks below zero still have no bar.
    assert run_js(REPORT_JS, f"{I}.slaBarModel({js({'name': 'p95_latency', 'threshold': 10, 'actual': -1, 'passed': True})})") is None


# ---- the SLA table and the baseline line ------------------------------------------------------------------------------


def test_sla_table_shows_regression_rows_with_units_and_the_baseline():
    html = run_js(REPORT_JS, f"{I}.renderSlaChecklist({js(_final_row(_regression()))})")
    rows = re.findall(r"<tr class=\"is-[a-z]+ is-regression\">(.*?)</tr>", html)
    assert len(rows) == 3
    assert "p95 к базе, рост в %" in rows[0] and "«run-base»" in rows[0]
    assert "≤ 10 %" in rows[0] and ">25 %<" in rows[0]
    assert "≤ 0.5 п.п." in rows[1] and ">0.1 п.п.<" in rows[1]
    assert ">-4 %<" in rows[2]


def test_baseline_line_by_mode():
    def line(regression, locale="ru"):
        return run_js(REPORT_JS, f"{I}.baselineLineHtml({js(_final_row(regression))})", locale=locale)

    pinned = line(_regression())
    assert 'data-baseline="compared"' in pinned and "База:" in pinned and "эталон" in pinned
    assert 'href="/reports/id-base"' in pinned and "/compare?run_a=run-new&amp;run_b=run-base" in pinned
    release = line(_regression(mode="previous_release", baseline_release="25.10"))
    assert "релиз 25.10" in release and "принятый прогон прошлого релиза" in release
    success = line(_regression(mode="previous_success", baseline_run_id=""), "en")
    assert "previous successful run" in success and 'href="/reports/run-base"' in success and not CYRILLIC.search(success)
    skipped = line({**_regression(available=False, reason="level_mismatch")})
    assert 'data-baseline="none"' in skipped and "сравнения нет: прогоны измерены на разной нагрузке" in skipped
    none = line({"available": False, "reason": "no_baseline", "baseline_run_name": ""}, "en")
    assert "no regression check: there is no baseline run" in none and "/compare" not in none
    assert line({"available": False, "reason": "disabled"}) == ""
    assert line(None) == ""


def test_hero_puts_the_baseline_line_under_the_verdict_summary():
    html = run_js(REPORT_JS, f"{I}.heroHtml({js(_final_row(_regression()))}, [])")
    assert html.index("rep-verdict-why") < html.index("rep-baseline") < html.index("rep-verdict-source")
    plain = run_js(REPORT_JS, f"{I}.heroHtml({js(_final_row(None, BASE_CHECKS))}, [])")
    assert "rep-baseline" not in plain


def test_verdict_sentence_names_a_single_regression_breach():
    text = run_js(REPORT_JS, f"{I}.verdictSummaryHtml({js(_final_row(_regression()))})")
    assert "p95 к базе хуже на <b>25 %</b> (порог <b>10 %</b>)" in text
    english = run_js(REPORT_JS, f"{I}.verdictSummaryHtml({js(_final_row(_regression()))})", locale="en")
    assert "p95 against the baseline is worse by <b>25 %</b> (threshold <b>10 %</b>)" in english


# ---- settings ------------------------------------------------------------------------------------------------------


def test_settings_schema_has_the_regression_fields():
    en = run_js(SETTINGS_JS, "LL.settings._internals.schemaTexts()", locale="en")
    ru = run_js(SETTINGS_JS, "LL.settings._internals.schemaTexts()", locale="ru")
    assert "p95 growth for a risk, %" in en and "Load drop for a failure, %" in en
    assert "Рост ошибок для риска, п.п." in ru
    assert not [text for text in en if CYRILLIC.search(text)]


def test_settings_page_has_the_regression_block():
    template = (ROOT / "templates" / "settings.html").read_text(encoding="utf-8")
    assert 'id="slaRegressionFields"' in template and "settings.sla.regression_heading" in template


# ---- Confluence ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_confluence_sla_table_has_regression_rows_and_the_baseline(lang):
    row = _final_row(_regression(mode="previous_release", baseline_release="25.10"))
    with with_locale(lang):
        body = confluence_export.sla_criteria_block(row)
    if lang == "ru":
        assert "p95 к базе, рост в %" in body and "10 %" in body and "0.5 п.п." in body
        assert "База проверки деградации: «run-base» (релиз 25.10, принятый прогон прошлого релиза)" in body
    else:
        assert "p95 against the baseline, % growth" in body and "0.5 pp" in body
        assert "Baseline of the regression check: «run-base» (release 25.10, accepted run of an earlier release)" in body


def test_confluence_baseline_line_variants():
    with with_locale("en"):
        assert confluence_export.regression_baseline_text({}) == ""
        assert confluence_export.regression_baseline_text({"regression": {"available": False, "reason": "disabled"}}) == ""
        assert confluence_export.regression_baseline_text({"regression": {"available": False, "reason": "no_baseline"}}) == (
            "No regression check: there is no baseline run"
        )
        assert confluence_export.regression_baseline_text({"regression": _regression(available=False, reason="other_service", mode="")}) == (
            "Baseline of the regression check: «run-base» — not compared: the baseline run belongs to another service"
        )
    # Phase checks got their titles in Confluence with the shared names.
    with with_locale("ru"):
        assert confluence_export.sla_check_label("phase_duration") != "phase_duration"
