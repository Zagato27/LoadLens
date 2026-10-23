# SPDX-License-Identifier: Apache-2.0
"""Localization of the Confluence pages, their PNG charts and the capacity forecast texts (plans/i18n-plan.md 4.5, 4.6/9).

Everything LoadLens writes into Confluence follows the language of the report (``report_language`` of the stored final
row; a report without it is Russian), never the UI language of whoever publishes it: the report page and its charts
(loadlens_app/confluence_export.py), the forecast page and its pictures (loadlens_app/forecast_confluence.py) and the
texts the template flow puts into a copy of the user's template page (update_page.py, confluence_manager/). The job's
own progress messages are ``msg()`` refs in the job's locale. Forecast server texts (errors, notes, plan messages)
follow the current locale: the request on the forecast page, the report language in a publication.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from flask import Flask

from loadlens_app import core  # noqa: F401  (before anything that imports update_page)
from tests.test_smoke_api import FakeConnection, client, patch_core  # noqa: F401  (fixtures)
from AI.db_store import RunAnalyses, StoredFinalReport
from AI.i18n import get_locale, msg, with_locale
from AI.limit_explanation import ExplanationParseError, parse_explanation
from confluence_manager import update_confluence_template as template
from loadlens_app import confluence_export, forecast_confluence
from loadlens_app import forecast as forecast_service
from loadlens_app import forecast_explanation as explain
from loadlens_app.forecast import ForecastError
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale

CONFLUENCE_CFG = {
    "enabled": True, "base_url": "https://wiki.local", "username": "u", "password": "p",
    "space_key": "LT", "parent_page_id": "1",
}


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    def no_db():
        raise AssertionError("Confluence localization tests must not open database connections")

    monkeypatch.setattr(confluence_export, "_ts_conn", no_db)


@pytest.fixture
def drawn_texts(monkeypatch):
    """Every text drawn on a figure saved while the test runs (titles, axis labels, ticks, legends, annotations)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    seen: list[str] = []
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        seen.extend(item.get_text() for item in self.findobj(Text) if item.get_text())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return seen


# ---- report page ---------------------------------------------------------------------------------------------------

STEP_TABLE = {
    "latency_query": "Latency p95 by request",
    "latency_series": "POST /checkout",
    "steps": [
        {"index": 1, "rps": 100, "latency_p95": 120, "stable": True,
         "start_iso": "2024-03-09T18:50:00+03:00", "end_iso": "2024-03-09T19:05:00+03:00"},
        {"index": 2, "rps": 230, "latency_p95": 180, "selected": True, "stable": False, "dip": True,
         "start_iso": "2024-03-09T19:05:00+03:00", "end_iso": "2024-03-09T19:25:00+03:00"},
        {"index": 3, "rps": 150, "latency_p95": 900, "stable": False, "after_drop": True,
         "start_iso": "2024-03-09T19:25:00+03:00", "end_iso": "2024-03-09T19:40:00+03:00"},
    ],
}


def _rows(lang: str | None) -> list[dict]:
    """A step report as the pipeline stores it; stored texts are English so only LoadLens' own words can be Russian."""
    marker = {"report_language": lang} if lang else {}
    final = {
        "domain": "final", "test_type": "step", "start_ms": 1710000000000, "end_ms": 1710003600000,
        "verdict": "risk", "llm_verdict": "fail", "sla_verdict": "risk", "service": "shop", "run_name": "run-1",
        "parsed": {
            "verdict": "risk",
            "verdict_rationale": "Target RPS reached, p95 grows.",
            "peak_performance": {"max_rps": 226.5, "max_time": "12:10", "drop_time": "12:40"},
            "findings": [
                {"id": "f1", "summary": "CPU near the limit", "severity": "high", "component": "orders",
                 "verification": {"status": "unverified"},
                 "evidence_items": [{"metric": "cpu", "observed_value": 95, "threshold": 80, "note": "steady"},
                                    {"metric": "gc", "observed_value": True}]},
            ],
            "recommended_actions": [
                {"summary": "Raise the CPU limit", "details": "From 2 to 4 cores.", "for_finding_ids": ["f1"]},
                {"summary": "Check the cache", "for_finding_ids": ["f9"]},
            ],
            "affected_components": ["orders"],
            **marker,
        },
        "sla_details": {
            "checks": [
                {"name": "target_rps", "passed": True, "threshold": 200, "actual": 245.678, "message": "RPS on the stable step"},
                {"name": "p95_latency", "passed": False, "threshold": 500, "actual": 812.5, "message": "p95 above the threshold"},
                {"name": "cpu_usage", "passed": None, "threshold": 80, "actual": None},
                {"name": "custom_check", "passed": True, "threshold": 1, "actual": 0.5},
            ],
            "summary": "SLA verdict: Risks found.",
            "stable_window": {"start": "2024-03-09T16:10:00+00:00", "end": "2024-03-09T16:20:00+00:00", "level": 245.68},
            "load_step_table": STEP_TABLE,
            **marker,
        },
        "system_context": {
            "system": {"name": "Shop", "domain": "retail", "test_goal": "peak load", "description": "Online shop"},
            "architecture": {"style": "microservices", "components": [{"name": "orders"}]},
            "load_model": {"critical_user_flows": [{"name": "checkout"}]},
            "operational_context": {"known_risks": ["cold cache"]},
        },
    }
    jvm = {"domain": "jvm", "verdict": "pass", "llm_verdict": "pass", "parsed": {"verdict": "pass", "findings": [{"summary": "Heap is fine"}]}}
    kafka = {"domain": "kafka", "parsed": {"analysis_error": {"provider": "openai", "max_tokens": 4000, "output_tokens": 4000}}}
    custom = {"domain": "custom", "text": ""}
    return [final, jvm, kafka, custom]


def _page(rows: list[dict]) -> str:
    return confluence_export.build_page_body(
        run_name="run-1", service="shop", source_url="http://localhost/reports/rid-1", llm_rows=rows,
        engineer_html="", charts=[{"domain": "jvm", "title": "Heap used", "filename": "jvm-heap.png"}],
    )


def test_an_en_report_page_has_no_russian_words():
    body = _page(_rows("en"))
    assert_no_cyrillic(body)
    for part in (
        "<th>Service</th><td>shop</td>", "<th>Run</th>", "<th>Test time</th>", "<th>Test type</th><td>Step (max perf)</td>",
        "Open the report in LoadLens", "<h2>SLA criteria</h2>", "<h2>Engineer&apos;s summary</h2>", "No engineer comment.",
        "<th>Status</th><th>Criterion</th><th>Threshold</th><th>Actual</th>", 'ac:parameter ac:name="title">Met<',
        ">Breached<", ">No data<", "Target RPS<br />RPS on the stable step", "<td>custom_check</td>",
        "SLA checked on step 2 (19:05–19:25), window 19:10–19:20, window RPS 245.68. The RPS in the step table",
        "Context of the system under test", "System: Shop", "Critical flow: checkout", "Risk: cold cache",
        'title">Test steps<', "<th>Step</th><th>Interval</th><th>RPS</th><th>Response time, p95</th>",
        "2 SLA, short dip", "3, after a drop", "Source: query “Latency p95 by request”, series “POST /checkout”",
        "RPS is the step level.", "<h2>LLM analysis</h2>", 'title">Summary<', "<th>Verdict</th><td>Risks found</td>",
        "<th>LLM verdict</th><td>Failed</td>", "Verdict rationale", "Peak performance", "<th>Max RPS</th><td>226.5</td>",
        "Problems and recommendations", "How to fix", "CPU near the limit (not confirmed by data)",
        "cpu | value: 95 | threshold: 80 | steady", "gc | value: yes", "Severity: high · Component: orders",
        "Additional recommendation", "Affected components", "AI analysis did not run", "openai · limit 4000 · output 4000",
        "<p>No data.</p>", "<h2>Metric charts</h2>",
    ):
        assert part in body, part


def test_a_report_without_language_keeps_its_russian_page_in_any_locale():
    rows = _rows(None)
    russian = _page(rows)
    with with_locale("en"):
        assert _page(rows) == russian
    for part in ("SLA-критерии", "Итоги от инженера", "Анализ LLM", 'title">Итог<', "Соблюдён", "Нарушен", "Графики метрик",
                 "Открыть отчёт в LoadLens", "SLA проверено на ступени 2 (19:05–19:25), окно 19:10–19:20, RPS окна 245.68.",
                 "(не подтверждено данными)", "<th>Вердикт LLM</th><td>Провал</td>", "значение: да"):
        assert part in russian, part


def test_an_en_report_keeps_the_page_english_under_a_russian_scope_and_back():
    rows = _rows("en")
    with with_locale("ru"):
        assert _page(rows) == _page(_rows("en"))
    assert stored_language_of(rows) == "en"


def stored_language_of(rows):
    from loadlens_app.i18n import stored_report_language

    return stored_report_language(rows)


def test_a_manual_verdict_and_the_window_notes_follow_the_locale():
    row = {**_rows("en")[0], "manual_verdict": {"verdict": "pass", "author": "ivanov", "changed_at": "2026-10-17T12:20:00Z", "reason": "generator fault"}}
    with with_locale("en"):
        body = confluence_export.render_llm_domain_body(row)
        hold = confluence_export.sla_window_note({"kind": "hold", "start": "2024-03-09T16:10:00+00:00", "end": "2024-03-09T16:20:00+00:00", "level": 300}, STEP_TABLE)
        spike = confluence_export.sla_window_note({"kind": "spike"}, STEP_TABLE)
        stable = confluence_export.sla_window_note({"level": 300}, {"steps": []})
        empty = confluence_export.sla_criteria_block({"domain": "final"})
    assert "<th>Verdict</th><td>Passed</td>" in body and "<th>Automatic verdict</th><td>Risks found</td>" in body
    assert "ivanov, 2026-10-17 12:20 UTC. Reason: generator fault" in body
    assert hold == "SLA checked on the hold window 19:10–19:20, mean RPS 300."
    assert spike == "SLA checked in the spike windows: the worst spike for each threshold."
    assert stable == "SLA checked on a stable step, window RPS 300."
    assert "automatic verdict is “Insufficient data”" in empty
    assert confluence_export.sla_window_note({"level": 300}, {"steps": []}) == "SLA проверено на стабильной ступени, RPS окна 300."


def test_the_unverified_mark_is_not_added_twice_in_either_language():
    item = {"summary": "CPU high (not confirmed by data)", "verification": {"status": "unverified"}}
    assert "(не подтверждено данными)" not in confluence_export._finding_html(item)
    item = {"summary": "CPU high (не подтверждено данными)", "verification": {"status": "unverified"}}
    with with_locale("en"):
        assert "(not confirmed by data)" not in confluence_export._finding_html(item)


def test_png_captions_follow_the_locale(drawn_texts):
    with with_locale("en"):
        confluence_export.render_series_png([], "RPS total")
    assert {"Time", "No data", "RPS total"} <= set(drawn_texts)
    drawn_texts.clear()
    confluence_export.render_series_png([], "RPS total")
    assert {"Время", "Нет данных"} <= set(drawn_texts)


def _patch_publication(monkeypatch, rows):
    points = [{"t": f"2024-03-09T16:{minute:02d}:00+00:00", "series": "all", "value": 100.0 + minute} for minute in range(0, 50, 5)]
    monkeypatch.setattr(confluence_export, "confluence_config", lambda: dict(CONFLUENCE_CFG))
    monkeypatch.setattr(confluence_export, "load_llm_rows", lambda run: copy.deepcopy(rows))
    monkeypatch.setattr(confluence_export, "load_engineer_html", lambda run: "")
    monkeypatch.setattr(confluence_export, "load_domains_schema", lambda run: {"lt_framework": ["RPS total", "Latency p95 by request"]})
    monkeypatch.setattr(confluence_export, "load_run_series", lambda run, domain, label: points)
    monkeypatch.setattr(confluence_export, "load_publication", lambda run: None)
    monkeypatch.setattr(confluence_export, "save_publication", lambda **kwargs: None)
    captured: dict = {}

    def upsert(confluence_client, known_page_id, title, body, attachments, progress_callback):
        captured.update(title=title, body=body, attachments=attachments)
        return "10", "https://wiki.local/p/10"

    monkeypatch.setattr(confluence_export, "upsert_page", upsert)
    return captured


def test_an_en_report_is_published_in_english_with_english_charts(monkeypatch, drawn_texts):
    captured = _patch_publication(monkeypatch, _rows("en"))
    progress: list = []
    result = confluence_export.publish_report("run-1", progress_callback=lambda message, pct: progress.append(message))
    assert result["page_id"] == "10" and captured["title"] == "LoadLens: run-1 (shop)"
    assert_no_cyrillic(captured["body"])
    assert len(captured["attachments"]) == 2
    assert "Time" in drawn_texts
    assert_no_cyrillic("\n".join(drawn_texts))
    # The job's messages follow the job's locale (ru by default here), as refs the job list renders per reader.
    assert [message.key for message in progress] == [
        "confluence.progress.loading_report", "confluence.progress.preparing_charts", "confluence.progress.done",
    ]
    assert str(progress[0]) == "Загрузка отчёта…"


def test_publishing_under_an_en_cookie_keeps_a_ru_report_russian(monkeypatch, drawn_texts):
    captured = _patch_publication(monkeypatch, _rows(None))
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    progress: list = []
    app = Flask(__name__)
    with app.test_request_context(headers={"Cookie": "loadlens.lang=en"}):
        assert get_locale() == "en"
        confluence_export.publish_report("run-1", progress_callback=lambda message, pct: progress.append(message))
    assert "SLA-критерии" in captured["body"] and "Анализ LLM" in captured["body"]
    assert "Время" in drawn_texts and "Time" not in drawn_texts
    assert str(progress[0]) == "Loading the report…" and progress[0].render("ru") == "Загрузка отчёта…"


def test_publication_errors_render_where_they_are_shown(monkeypatch):
    with pytest.raises(ValueError) as off:
        confluence_export.validate_confluence_config({"enabled": False})
    assert str(off.value) == "Публикация в Confluence выключена: установите confluence.enabled=true"
    with with_locale("en"):
        assert str(off.value) == "Publishing to Confluence is off: set confluence.enabled=true"
    with pytest.raises(ValueError) as missing:
        confluence_export.validate_confluence_config({"enabled": True, "base_url": "https://wiki.local"})
    assert str(missing.value) == "Не заполнены настройки Confluence: username, password, space_key, parent_page_id"
    monkeypatch.setattr(confluence_export, "confluence_config", lambda: dict(CONFLUENCE_CFG))
    monkeypatch.setattr(confluence_export, "load_llm_rows", lambda run: [])
    monkeypatch.setattr(confluence_export, "load_run_meta", lambda run: None)
    with pytest.raises(ValueError) as empty:
        confluence_export.publish_report("run-9")
    assert str(empty.value) == "Для запуска 'run-9' нет данных отчёта"
    with with_locale("en"):
        assert str(empty.value) == "Run 'run-9' has no report data"


# ---- template flow -------------------------------------------------------------------------------------------------


class _FakeConfluence:
    """atlassian.Confluence of the template module: one template page, created pages and updates recorded."""

    pages: dict = {}
    created: list = []
    updates: list = []
    fail_update = False

    def __init__(self, **kwargs):
        pass

    def get_page_by_id(self, page_id, **kwargs):
        if page_id not in self.pages:
            raise RuntimeError("404 page not found")
        return copy.deepcopy(self.pages[page_id])

    def create_page(self, space, title, body, parent_id=None, **kwargs):
        self.created.append(title)
        return {"id": "900"}

    def update_page(self, **kwargs):
        if self.fail_update:
            raise RuntimeError("Attempted to update stale data")
        self.updates.append(kwargs)
        return {"id": kwargs.get("page_id")}


@pytest.fixture
def fake_confluence(monkeypatch):
    class Fake(_FakeConfluence):
        pages = {"200": {"id": "200", "title": "Checkout template", "space": {"key": "LT"}, "version": {"number": 1},
                         "body": {"storage": {"value": "<p>$$final_answer$$</p><p>$$x$$</p>"}}}}
        created: list = []
        updates: list = []
        fail_update = False

    monkeypatch.setattr(template, "Confluence", Fake)
    return Fake


def test_template_updates_return_codes_and_raise_localized_errors(fake_confluence):
    assert template.update_confluence_page("u", "n", "p", "200", "$$x$$", "<b>v</b>") == template.STATUS_OK
    with pytest.raises(template.PlaceholderNotFoundError) as missing:
        template.update_confluence_page("u", "n", "p", "200", "$$y$$", "v")
    assert str(missing.value) == "Плейсхолдер $$y$$ не найден на странице Confluence 200"
    with with_locale("en"):
        assert str(missing.value) == "Placeholder $$y$$ is not on Confluence page 200"
    with pytest.raises(template.TemplateUpdateError):
        template.update_confluence_page("u", "n", "p", "404", "$$x$$", "v")
    fake_confluence.fail_update = True
    with pytest.raises(template.TemplateUpdateError) as failed:
        template.update_confluence_page("u", "n", "p", "200", "$$x$$", "v")
    # update_page's retry loop recognises a version conflict by the client's text, kept as a parameter.
    assert "Attempted to update stale data" in str(failed.value)
    assert template.update_confluence_page_multi("u", "n", "p", "200", {"$$x$$": "v"}) == template.STATUS_UPDATE_FAILED
    fake_confluence.fail_update = False
    assert template.update_confluence_page_multi("u", "n", "p", "200", {"$$x$$": "v"}) == template.STATUS_OK
    assert template.update_confluence_page_multi("u", "n", "p", "200", {"$$zz$$": "v"}) == template.STATUS_NO_REPLACEMENTS
    assert template.update_confluence_page_multi("u", "n", "p", "404", {"$$x$$": "v"}) == template.STATUS_LOAD_FAILED


def test_update_page_does_not_parse_russian_status_texts():
    import update_page

    source = Path(update_page.__file__).read_text(encoding="utf-8")
    assert "Плейсхолдер не найден" not in source and 'startswith("Ошибка")' not in source


def test_the_copied_template_page_title_follows_the_locale(fake_confluence):
    assert template.copy_confluence_page("u", "n", "p", "200", "100") == "900"
    with with_locale("en"):
        template.copy_confluence_page("u", "n", "p", "200", "100")
    ru, en = fake_confluence.created
    assert re.fullmatch(r"Checkout template - отчет \d{4}-\d{2}-\d{2} \d{2}:\d{2}", ru)
    assert re.fullmatch(r"Checkout template - report \d{4}-\d{2}-\d{2} \d{2}:\d{2}", en)


TEMPLATE_REPORT = {
    "verdict": "risk", "verdict_rationale": "p95 grows",
    "peak_performance": {"max_rps": 200, "max_time": "12:00"},
    "findings": [{"id": "f1", "summary": "CPU high", "severity": "high", "component": "orders",
                  "evidence_items": [{"metric": "cpu", "observed_value": 95, "threshold": 80}]}, {"summary": "Heap grows"}],
    "recommended_actions": [{"summary": "Add CPU", "details": "- two cores", "for_finding_ids": ["f1"]}],
    "affected_components": ["orders"],
}


def test_template_renderers_follow_their_language():
    en = template.render_llm_markdown(TEMPLATE_REPORT, domain="final", lang="en") + template.render_llm_html(TEMPLATE_REPORT, lang="en")
    en += json.dumps(template.render_llm_report_placeholders({**TEMPLATE_REPORT, "findings": [], "recommended_actions": []}, lang="en"))
    assert_no_cyrillic(en)
    for part in ("| Parameter | Value |", "| Test verdict | **Risks found** |", "#### Verdict rationale", "#### Peak performance",
                 "**Severity:** `high`", "**How to fix:**", "value: 95 | threshold: 80", "Problems and recommendations",
                 "No significant findings", "Affected components:"):
        assert part in en, part
    ru = template.render_llm_markdown(TEMPLATE_REPORT, domain="final")
    assert "| Вердикт по тесту | **Есть риски** |" in ru and "**Рекомендации по устранению:**" in ru
    with with_locale("en"):
        assert template.render_llm_markdown(TEMPLATE_REPORT, domain="final", lang="ru") == ru


SERVICE_CONFIG = {"page_parent_id": "100", "page_sample_id": "200", "metrics": [], "logs": []}
LEGACY = {"grafana_base_url": "http://grafana.test:3000", "url_basic": "https://wiki.test", "user": "wiki-user", "password": "wiki-pass"}
LLM_RESULTS = {
    "final_parsed": TEMPLATE_REPORT,
    "jvm_parsed": {"verdict": "pass", "findings": []},
    "scores": {
        "final": {"judge": {"overall": 0.8, "factual": 0.7, "completeness": 0.6, "specificity": 0.5}, "data_score": 0.9,
                  "final_score": 0.85, "confidence": 0.7, "judge_meta": {},
                  "data_score_details": {"label_grounding": 0.5, "numeric_claims_total": 3, "numeric_claims_supported": 2,
                                         "numeric_grounding": 0.66, "peak_checked": True, "peak_consistency": 0.9}},
        "jvm": {"judge_meta": {"skipped": True}, "data_score": 0.4, "confidence": 0.5,
                "data_score_details": {"label_grounding": 0.3, "numeric_claims_total": 0}},
    },
}


@pytest.fixture
def template_report(monkeypatch, tmp_path, fake_confluence):
    """update_report of the template flow: fake Confluence, no charts or logs, a fake LLM pipeline."""
    import update_page
    from data_collectors import temp_files

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps(LEGACY), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    monkeypatch.setattr(temp_files, "TEMP_DIR", tmp_path / "temporary_files")
    monkeypatch.setattr(update_page, "_active_metrics_config_now", lambda: {"shop": {"services": {"checkout": SERVICE_CONFIG}}})
    monkeypatch.setattr(update_page, "_resolve_services_for_area", lambda area: ["checkout"])
    replacements: list[dict] = []
    monkeypatch.setattr(
        update_page, "update_confluence_page_multi",
        lambda url, user, password, page_id, values, as_text=False: replacements.append(dict(values)) or template.STATUS_OK,
    )
    results: dict = {"value": LLM_RESULTS}
    monkeypatch.setattr(update_page, "uploadFromLLM", lambda *args, **kwargs: copy.deepcopy(results["value"]))

    def run(report_language=None, value=None):
        if value is not None:
            results["value"] = value
        replacements.clear()
        update_page.update_report(
            1_700_000_000_000, 1_700_000_600_000, "checkout", use_llm=True, save_to_db=False,
            run_name="run-1", project_area="shop", report_language=report_language,
        )
        return replacements[-1]

    return run


def test_the_template_flow_writes_its_texts_in_the_report_language(template_report, fake_confluence):
    en = template_report("en")
    assert fake_confluence.created[-1].startswith("Checkout template - report ")
    assert_no_cyrillic(json.dumps(en, ensure_ascii=False))
    final, jvm = en["$$final_answer$$"], en["$$answer_jvm$$"]
    for part in ("#### Answer assessment", "| Judge's score of the text | 80% |", "| Model confidence | 70% |",
                 "| Numbers in the text that match | 66% (2 of 3 confirmed) |", "| peak_performance match | 90% |",
                 "_The data check is heuristic:"):
        assert part in final, part
    assert "| Judge | did not run: the model gave a single answer |" in jvm
    assert "| Numbers in the text that match | not checked, the findings have no explicit numeric claims |" in jvm
    fallback = template_report("en", {"final": "Plain answer"})
    assert fallback["$$final_answer$$"].startswith("### LLM summary\n\nPlain answer")


def test_the_template_flow_of_a_ru_report_stays_russian(template_report, fake_confluence):
    ru = template_report(None)
    assert " - отчет " in fake_confluence.created[-1]
    final = ru["$$final_answer$$"]
    for part in ("#### Оценка ответа", "| Параметр | Значение |", "| Оценка текста судьей | 80% |",
                 "| Совпадение чисел из текста | 66% (подтверждено 2 из 3) |",
                 "_Проверка по данным остаётся эвристической: она оценивает привязку текста к метрикам, совпадение чисел и "
                 "согласованность peak_performance._"):
        assert part in final, part
    assert "| Судья | не запускался: модель дала один вариант ответа |" in ru["$$answer_jvm$$"]


# ---- capacity forecast ---------------------------------------------------------------------------------------------


def _forecast_env(monkeypatch, lang: str | None):
    from tests.test_smoke_api import _forecast_frames, _patch_forecast

    frames = _forecast_frames()
    index = frames["RPS"].index
    steps = [
        {"label": f"step {number + 1}", "source": "step_detector", "start_iso": index[number * 25].isoformat(),
         "end_iso": (index[number * 25] + pd.Timedelta(minutes=25)).isoformat()}
        for number in range(4)
    ]
    sla = {"checks": [{"name": "target_rps", "threshold": 200}], **({"report_language": lang} if lang else {})}
    report = StoredFinalReport(
        run_id="abc", run_name="nightly", service="demo", test_type="step", project_area="",
        start_ms=int(index[0].timestamp() * 1000), end_ms=int((index[-1] + pd.Timedelta(minutes=1)).timestamp() * 1000),
        context={"load_steps": steps}, sla_details=sla,
    )
    _patch_forecast(monkeypatch, report, frames)
    monkeypatch.setattr(explain, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(explain, "load_final_report", lambda *args: report)
    monkeypatch.setattr(explain, "load_run_analyses", lambda *args: RunAnalyses(findings=[], system_context={}))
    monkeypatch.setattr(explain, "load_forecast_explanation", lambda *args: None)
    monkeypatch.setitem(explain.CONFIG, "llm", {"provider": "openai", "openai": {"model": "test-model"}})
    monkeypatch.setattr(forecast_confluence, "confluence_config", lambda: dict(CONFLUENCE_CFG))
    monkeypatch.setattr(forecast_confluence, "load_forecast_publication", lambda run_id: None)
    monkeypatch.setattr(forecast_confluence, "save_forecast_publication", lambda *args: None)
    captured: dict = {}

    def upsert(confluence_client, known_page_id, title, body, attachments, progress_callback):
        captured.update(title=title, body=body, attachments=attachments)
        return "7", "https://wiki.local/p/7"

    monkeypatch.setattr(forecast_confluence, "upsert_page", upsert)
    return captured


def test_an_en_forecast_is_published_in_english(monkeypatch, drawn_texts):
    captured = _forecast_env(monkeypatch, "en")
    progress: list = []
    forecast_confluence.publish_forecast("abc", 200.0, 20.0, None, "http://localhost/forecasting/abc",
                                         lambda message, pct: progress.append(message))
    assert captured["title"] == "Capacity forecast: nightly (demo)"
    assert_no_cyrillic(captured["body"])
    assert_no_cyrillic("\n".join(drawn_texts))
    for part in ("Target 200 RPS: will hold · confidence", "<th>Forecast parameters</th><td>target 200 RPS, pod CPU ceiling 80%, "
                 "headroom to the limit 20%</td>", "Open the forecast in LoadLens", "Scenarios", "SLA target — 200 RPS",
                 "+25% over SLA", "Missing the service instance CPU query: SLA criteria", "p95 is not assigned in the SLA",
                 "How it is calculated", "Why it hit the limit", "No AI analysis yet. The report has no related findings."):
        assert part in captured["body"], part
    assert re.search(r"<th>Generated</th><td>\d{4}-\d{2}-\d{2} \d{2}:\d{2}</td>", captured["body"])
    assert "Mean response time, ms" in drawn_texts and any(text.startswith("step 1\n") for text in drawn_texts)
    # Progress messages are rendered outside the report scope, in the job's locale (ru here).
    assert [message.key for message in progress] == [
        "confluence.progress.forecast", "confluence.progress.scenarios", "confluence.progress.images", "confluence.progress.done",
    ]
    assert str(progress[0]) == "Расчёт прогноза…"


def test_a_forecast_of_an_old_report_is_published_in_russian_under_en(monkeypatch, drawn_texts):
    captured = _forecast_env(monkeypatch, None)
    with with_locale("en"):
        forecast_confluence.publish_forecast("abc", 200.0, 20.0, None, "", None)
    body = captured["body"]
    assert captured["title"] == "Прогноз мощностей: nightly (demo)"
    for part in ("Цель 200 RPS: выдержит · достоверность", "цель 200 RPS, потолок CPU пода 80 %, запас до предела 20 %",
                 "Не задан запрос CPU экземпляров сервисов: SLA-критерии → Определение максимальной производительности",
                 "Разбор ИИ ещё не делали. Связанных находок в отчёте нет."):
        assert part in body, part
    assert re.search(r"<th>Сформировано</th><td>\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}</td>", body)
    assert "Среднее время ответа, мс" in drawn_texts


def test_forecast_formatting_follows_the_locale():
    assert forecast_confluence._rps(1234.4) == "1\u00a0234 RPS"
    assert [forecast_confluence._pods(n) for n in (1, 3, 5, 21)] == ["1 под", "3 пода", "5 подов", "21 под"]
    assert (forecast_confluence._pct(0.0512), forecast_confluence._ms(1500.0), forecast_confluence._ms(8.5)) == ("5,1 %", "1,5 с", "8,5 мс")
    assert forecast_confluence.verdict_title("scale") == "выдержит после масштабирования"
    with with_locale("en"):
        assert forecast_confluence._rps(1234.4) == "1,234 RPS"
        assert [forecast_confluence._pods(n) for n in (1, 3)] == ["1 pod", "3 pods"]
        assert (forecast_confluence._pct(0.0512), forecast_confluence._ms(1500.0), forecast_confluence._ms(8.5)) == ("5.1%", "1.5 s", "8.5 ms")
        assert [forecast_confluence.verdict_title(code) for code in ("holds", "scale", "tight", "blocked")] == [
            "will hold", "will hold after scaling", "at the limit", "will not hold",
        ]
        assert forecast_confluence.confidence_title("medium") == "medium"


def test_forecast_api_texts_follow_the_ui_language(client, monkeypatch):  # noqa: F811  (fixture)
    _forecast_env(monkeypatch, None)  # a Russian report: the forecast page still speaks the reader's language
    use_locale(client, "en")
    body = client.get("/forecast/abc", query_string={"target_rps": 200}).get_json()
    assert body["resources"]["message"].startswith("Missing the service instance CPU query: SLA criteria")
    assert body["notes"] and not any(CYRILLIC.search(note) for note in body["notes"])
    view = client.get("/forecast/abc/explanation").get_json()
    assert view["status"] == "missing" and not CYRILLIC.search(json.dumps(view, ensure_ascii=False))
    use_locale(client, None)
    body = client.get("/forecast/abc", query_string={"target_rps": 200}).get_json()
    assert body["resources"]["message"].startswith("Не задан запрос CPU экземпляров сервисов")


def test_the_explanation_hash_does_not_depend_on_the_reader_locale(monkeypatch):
    _forecast_env(monkeypatch, None)
    ru = explain._context("abc")
    with with_locale("en"):
        en = explain._context("abc")
    assert en.lang == "ru" and en.inputs_hash == ru.inputs_hash and en.payload == ru.payload
    assert CYRILLIC.search(json.dumps(ru.payload["facts"]["cpu"], ensure_ascii=False))


def test_forecast_errors_render_in_the_locale_where_they_are_shown(monkeypatch):
    error = ForecastError(msg("capacity.error.missing_setting", title=msg("capacity.setting.load_model"), hint=msg("capacity.setting_hint")), 422, "missing_setting")
    assert str(error) == "Не задан параметр «модель нагрузки»: SLA-критерии → Определение максимальной производительности"
    with with_locale("en"):
        assert str(error) == "Missing the “load model” parameter: SLA criteria → Finding the maximum throughput"
    assert str(ForecastError("plain text", 400, "x")) == "plain text"
    with pytest.raises(ExplanationParseError) as parse:
        parse_explanation("no json here")
    assert str(parse.value) == "Ответ модели не содержит JSON: «no json here»"
    with with_locale("en"):
        assert str(parse.value) == "The model answer has no JSON: “no json here”"
    monkeypatch.setattr(forecast_service, "load_final_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(forecast_service, "_ts_conn", lambda: FakeConnection())
    with pytest.raises(ForecastError) as missing:
        forecast_service.build_forecast("nope", 200.0, 20.0)
    assert (missing.value.code, str(missing.value)) == ("not_found", "Отчёт не найден")
    with with_locale("en"):
        assert str(missing.value) == "Report not found"
    assert forecast_service.report_language_of("nope") == "ru"
