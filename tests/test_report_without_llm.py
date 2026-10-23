# SPDX-License-Identifier: Apache-2.0
"""Reports built without the model (``use_llm: false``): the deterministic final row and where it shows.

The pipeline stores the final row alone (SLA verdict, steps, test figures and the run context the forecast reads);
the database writer skips the domain rows; Confluence publishes the verdict and the figures with no analysis section,
and a run with metrics but no stored rows with its charts; the report page marks the verdict and shows the figures
in place of the analysis tabs.
"""

import copy
import json
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_smoke_api import RecordingConnection  # noqa: E402,F401  (initialises the app before update_page)
from tests.test_test_types import END_TS, P95_LABEL, PIPELINE_CASES, RPS_LABEL, START_TS, _pipeline_frames, _stress_steps  # noqa: E402
from tests.test_perf_reports_sql import SaveConnection, SqlStub  # noqa: E402
from tests.js_harness import run_js  # noqa: E402
from AI import db_store  # noqa: E402
from AI import pipeline as pipeline_module  # noqa: E402
from AI.context_pack import format_iso  # noqa: E402
from AI.i18n import t, with_locale  # noqa: E402
from AI.verdicts import DETERMINISTIC, is_deterministic_report, to_code  # noqa: E402
from loadlens_app import confluence_export  # noqa: E402

REPORT_JS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
I = "LL.report._internals"  # noqa: E741
CLOCK = re.compile(r"\d\d:\d\d")


def js(value):
    return json.dumps(value, ensure_ascii=False)


# ---- pipeline -------------------------------------------------------------------------------------------------


def _cfg(kind):
    sla = PIPELINE_CASES[kind][0]
    return {
        "llm": {"include_markdown_tables_in_context": False, "self_consistency_k": 1, "max_domain_workers": 1},
        "default_params": {"step": "1m", "resample_interval": "1T"},
        "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prometheus:9090"}}},
        "domain_sources": {"default": {"source": "prom"}, "lt_framework": {"source": "prom"}},
        "queries": {"lt_framework": {"promql_queries": ["a", "b"], "label_keys_list": [[], []], "labels": [RPS_LABEL, P95_LABEL]}},
        "sla": {"max_performance_query": RPS_LABEL, "p95_query": P95_LABEL, **sla},
    }


def _run(monkeypatch, kind, *, use_llm, save_to_db=True):
    """Runs the pipeline on the frames of ``kind``; returns (result, saved results, contexts the model got)."""
    saved: list = []
    contexts: dict = {}
    monkeypatch.setattr(pipeline_module, "load_prompt", lambda name, lang=None: "prompt")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *a, **k: _pipeline_frames(kind))
    monkeypatch.setattr(pipeline_module, "save_domain_labeled", lambda **kwargs: None)
    monkeypatch.setattr(pipeline_module, "save_llm_results", lambda results, run_meta, storage_cfg: saved.append(copy.deepcopy(results)))
    monkeypatch.setattr(pipeline_module, "_load_baseline_context", lambda *args: {"available": False, "reason": "test"})

    def fake_llm(user_prompt, data_context, k, return_scores, domain_key):
        if not use_llm:
            raise AssertionError("the model is called in a report without it")
        payload = {"verdict": "pass", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}
        contexts[domain_key] = json.loads(data_context)
        return json.dumps(payload), dict(payload), {}

    monkeypatch.setattr(pipeline_module, "llm_two_pass_self_consistency", fake_llm)
    result = pipeline_module.uploadFromLLM(
        start_ts=START_TS, end_ts=END_TS, save_to_db=save_to_db, only_collect=not use_llm, ef_config=_cfg(kind),
        run_meta={"test_type": kind, "run_name": f"run-{kind}", "service": "demo"}, active_domains=["lt_framework"],
    )
    return result, saved, contexts


@pytest.mark.parametrize("kind", list(PIPELINE_CASES))
def test_without_the_model_the_final_row_is_saved(monkeypatch, kind):
    _sla, mode, payload_key, _ctx_key = PIPELINE_CASES[kind]
    result, saved, _ = _run(monkeypatch, kind, use_llm=False)

    assert result == {} and len(saved) == 1
    results = saved[0]
    final = results["final_parsed"]
    assert is_deterministic_report(final)
    assert not {"final", "scores", "jvm", "lt_framework"} & set(results)
    assert final["test_profile"]["mode"] == mode and final["report_language"] == results["report_language"]
    assert final["verdict"] == final["deterministic_sla"]["verdict"] == to_code(results["sla_verdict"])
    assert results["sla_checks"] and results["sla_summary"]
    if payload_key:
        assert payload_key in final
    context = json.loads(results["contexts"]["final"])
    assert context["load_steps"] and context["lt_series_labels"] and "rps_drop_iso" in context
    peak = final["peak_performance"]
    if mode in ("capacity", "stress"):
        assert peak["max_rps"] == pytest.approx(300.0, abs=2.0)
        # The times the model would read off the context: the SLA step and the step above it that broke SLA.
        assert CLOCK.fullmatch(peak["max_time"]) and CLOCK.fullmatch(peak["drop_time"])
        assert peak["max_time"] < peak["drop_time"]
        assert final["load_step_table"]["steps"]
    else:
        assert peak["not_applicable"] is True and "max_time" not in peak


def test_the_final_row_without_the_model_matches_the_report_with_it(monkeypatch):
    _, saved, _ = _run(monkeypatch, "stress", use_llm=False)
    with_model, _, contexts = _run(monkeypatch, "stress", use_llm=True, save_to_db=False)
    deterministic, analysed = saved[0]["final_parsed"], with_model["final_parsed"]

    for key in ("report_language", "test_profile", "deterministic_sla", "load_step_table", "saturation", "verdict"):
        assert deterministic[key] == analysed[key], key
    assert deterministic["peak_performance"]["max_rps"] == analysed["peak_performance"]["max_rps"]
    for key in ("sla_verdict", "sla_checks", "sla_summary", "sla_window", "load_step_table", "system_context"):
        assert saved[0][key] == with_model[key], key
    # One builder: the stored context is the one the model reads.
    assert json.loads(saved[0]["contexts"]["final"]) == contexts["final"]


def test_peak_time_without_an_sla_step_is_the_stable_step_at_the_peak():
    steps = _stress_steps()  # 100, 200, 300, 400 RPS, 15 minutes each from 09:00 UTC
    assert pipeline_module._peak_step_start(steps, 395.0, 3) == format_iso(steps[3].start, 3)
    plateau = [replace(step, plateau_start=step.start + pd.Timedelta(minutes=2)) for step in steps]
    assert pipeline_module._peak_step_start(plateau, 300.0, 3)[11:16] == "12:32"
    assert pipeline_module._peak_step_start(steps, 250.0, 3) is None  # no step within 10 %
    assert pipeline_module._peak_step_start([replace(step, stable=False) for step in steps], 400.0, 3) is None
    assert pipeline_module._peak_step_start(steps, None, 3) is None


def test_without_the_model_and_storage_nothing_is_saved(monkeypatch):
    result, saved, _ = _run(monkeypatch, "step", use_llm=False, save_to_db=False)
    assert result == {} and saved == []


# ---- database writer ------------------------------------------------------------------------------------------


def _write(monkeypatch, results):
    captured: dict = {}
    monkeypatch.setattr(db_store, "execute_batch", lambda cur, sql_text, rows, page_size=100: captured.update(rows=rows))
    monkeypatch.setattr(db_store, "_connect", lambda _cfg: SaveConnection())
    monkeypatch.setattr(db_store, "_ensure_llm_reports_table", lambda *_a, **_k: None)
    monkeypatch.setattr(db_store.sql, "SQL", SqlStub)
    db_store.save_llm_results(
        results,
        {"run_id": "r1", "run_name": "run-1", "service": "demo", "test_type": "step", "start_ms": 1, "end_ms": 2},
        {"schema": "public", "llm_table": "llm_reports"},
    )
    return captured["rows"]


def test_the_writer_stores_the_final_row_alone(monkeypatch):
    rows = _write(monkeypatch, {
        "report_language": "en",
        "final_parsed": {"analysis": DETERMINISTIC, "verdict": "risk", "peak_performance": {"max_rps": 300.0}},
        "contexts": {"final": json.dumps({"load_steps": [{"index": 1}]})},
        "sla_verdict": "risk",
        "sla_checks": [{"name": "p95_latency", "passed": False}],
        "sla_summary": "SLA verdict: At risk.",
        "load_step_table": {"steps": [{"index": 1}]},
        "system_context": {"system": {"name": "shop"}},
    })
    assert [row[6] for row in rows] == ["final"]
    (_run_id, _name, _service, _type, _start, _end, _domain, text, parsed, scores, context, system_context,
     verdict, sla_verdict, sla_details, _area, max_rps) = rows[0]
    assert text is None and scores is None
    assert verdict == sla_verdict == "risk"
    assert parsed.adapted["analysis"] == DETERMINISTIC
    assert context.adapted == {"load_steps": [{"index": 1}]}
    assert system_context.adapted == {"system": {"name": "shop"}}
    assert sla_details.adapted["load_step_table"] == {"steps": [{"index": 1}]}
    assert sla_details.adapted["report_language"] == "en" and sla_details.adapted["checks"]
    assert max_rps == 300.0


def test_the_writer_keeps_every_domain_row_of_a_report_with_the_model(monkeypatch):
    rows = _write(monkeypatch, {"jvm": "text", "jvm_parsed": {"verdict": "pass"}, "final": "{}", "final_parsed": {"verdict": "pass"}})
    assert [row[6] for row in rows] == ["jvm", "database", "kafka", "microservices", "hard_resources", "final"]


def test_deterministic_marker():
    assert is_deterministic_report({"analysis": DETERMINISTIC})
    assert not is_deterministic_report({"verdict": "pass"})
    assert not is_deterministic_report(None) and not is_deterministic_report('{"analysis": "deterministic"}')


# ---- Confluence -----------------------------------------------------------------------------------------------

START_MS = 1710000000000  # 2024-03-09 16:00 UTC
END_MS = START_MS + 50 * 60 * 1000
CHARTS = [{"domain": "lt_framework", "title": "LT RPS", "filename": "lt-rps.png"}]


def _final_row(lang="ru", **extra):
    return {
        "domain": "final", "service": "shop", "run_name": "run-1", "test_type": "step",
        "start_ms": START_MS, "end_ms": END_MS, "text": None,
        "verdict": "risk", "llm_verdict": "risk", "sla_verdict": "risk", "manual_verdict": None,
        "parsed": {
            "analysis": DETERMINISTIC, "verdict": "risk", "report_language": lang,
            "peak_performance": {"max_rps": 300.0, "max_time": "19:30", "drop_time": "19:45", "method": "stable_max (query: LT RPS)"},
        },
        "sla_details": {
            "checks": [{"name": "p95_latency", "threshold": 300, "actual": 420, "passed": False, "message": "p95 420 > 300"}],
            "summary": "SLA verdict: risk.", "report_language": lang,
        },
        "system_context": None,
        **extra,
    }


def _page(rows, **kwargs):
    return confluence_export.build_page_body(
        run_name="run-1", service="shop", source_url="", llm_rows=rows, engineer_html="<p>Комментарий</p>",
        charts=CHARTS, step_table=None, **kwargs,
    )


def test_confluence_page_of_a_report_without_the_model():
    body = _page([_final_row()])
    assert "Анализ LLM" not in body and "ui-tabs" not in body.split("Итоги от инженера")[0]
    assert "Отчёт построен без анализа ИИ." in body
    assert body.index("Вердикт по тесту") < body.index("SLA-критерии") < body.index("Итоги от инженера")
    assert "Есть риски" in body and "Пиковая производительность" in body
    assert "300" in body and "19:30" in body and "19:45" in body
    assert "lt-rps.png" in body


def test_confluence_page_without_the_model_shows_a_manual_verdict_and_the_report_language():
    manual = {"verdict": "pass", "author": "p.ivanov", "changed_at": "2026-10-17T13:00:00+00:00", "reason": "retest ok"}
    body = _page([_final_row(manual_verdict=manual)])
    assert "Изменён вручную" in body and "retest ok" in body and "Автоматический вердикт" in body
    body = _page([_final_row("en")])
    assert "The report was built without the AI analysis." in body and "Test verdict" in body
    assert "LLM analysis" not in body and "Анализ" not in body


def test_confluence_page_of_a_run_without_report_rows():
    with with_locale("en"):
        body = _page([], run_meta={"service": "shop", "start_ms": START_MS, "end_ms": END_MS})
        sla_title, analysis_title = t("report.sla.title"), t("confluence.llm_analysis")
    assert "The report was built without the AI analysis." in body
    assert sla_title not in body and analysis_title not in body
    assert "2024-03-09 16:00 UTC" in body and "lt-rps.png" in body


def _patch_publication(monkeypatch, rows, run_meta=None):
    points = [{"t": f"2024-03-09T16:{minute:02d}:00+00:00", "series": "all", "value": 100.0 + minute} for minute in range(0, 50, 5)]
    monkeypatch.setattr(confluence_export, "confluence_config", lambda: {
        "enabled": True, "base_url": "https://wiki.local", "username": "u", "password": "p", "space_key": "LT", "parent_page_id": "42",
    })
    monkeypatch.setattr(confluence_export, "load_llm_rows", lambda run: copy.deepcopy(rows))
    monkeypatch.setattr(confluence_export, "load_run_meta", lambda run: run_meta)
    monkeypatch.setattr(confluence_export, "load_engineer_html", lambda run: "")
    monkeypatch.setattr(confluence_export, "load_domains_schema", lambda run: {"lt_framework": ["RPS total"]})
    monkeypatch.setattr(confluence_export, "load_run_series", lambda run, domain, label: points)
    monkeypatch.setattr(confluence_export, "load_publication", lambda run: None)
    monkeypatch.setattr(confluence_export, "save_publication", lambda **kwargs: None)
    captured: dict = {}

    def upsert(confluence_client, known_page_id, title, body, attachments, progress_callback):
        captured.update(title=title, body=body, attachments=attachments)
        return "10", "https://wiki.local/p/10"

    monkeypatch.setattr(confluence_export, "upsert_page", upsert)
    return captured


def test_publish_a_report_without_the_model(monkeypatch):
    captured = _patch_publication(monkeypatch, [_final_row()])
    result = confluence_export.publish_report("run-1")
    assert result["page_id"] == "10" and captured["title"] == "LoadLens: run-1 (shop)"
    assert "Вердикт по тесту" in captured["body"] and "Анализ LLM" not in captured["body"]
    assert len(captured["attachments"]) == 1


def test_publish_a_run_without_report_rows(monkeypatch):
    captured = _patch_publication(monkeypatch, [], {"service": "shop", "start_ms": START_MS, "end_ms": END_MS})
    result = confluence_export.publish_report("run-2")
    assert result["title"] == "LoadLens: run-2 (shop)"
    assert "Отчёт построен без анализа ИИ." in captured["body"] and "SLA-критерии" not in captured["body"]
    assert len(captured["attachments"]) == 1


class _MetaConnection:
    """Connection of ``load_run_meta``: answers ``fetchone`` in order; ``fail_first`` makes run_index raise."""

    def __init__(self, answers, fail_first=False):
        self.answers = list(answers)
        self.fail_first = fail_first
        self.queries: list = []
        self.rolled_back = False

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self.queries.append((query, params))
        if self.fail_first and len(self.queries) == 1:
            raise RuntimeError('relation "run_index" does not exist')

    def fetchone(self):
        return self.answers.pop(0)

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


@pytest.mark.parametrize("fail_first, answers, expected", [
    (False, [("shop", datetime(2024, 3, 9, 16, 0, tzinfo=timezone.utc), datetime(2024, 3, 9, 16, 50, tzinfo=timezone.utc))],
     {"service": "shop", "start_ms": START_MS, "end_ms": END_MS}),
    (True, [("shop", datetime(2024, 3, 9, 16, 0, tzinfo=timezone.utc), datetime(2024, 3, 9, 16, 50, tzinfo=timezone.utc))],
     {"service": "shop", "start_ms": START_MS, "end_ms": END_MS}),
    (False, [None, ("", None, None)], None),
])
def test_run_meta_reads_the_run_index_then_the_metrics(monkeypatch, fail_first, answers, expected):
    conn = _MetaConnection(answers, fail_first)
    monkeypatch.setattr(confluence_export, "_ts_conn", lambda: conn)
    assert confluence_export.load_run_meta("run-2") == expected
    assert all(params == ("run-2",) for _query, params in conn.queries)
    assert conn.rolled_back is fail_first


# ---- report page ----------------------------------------------------------------------------------------------


def test_report_page_marks_a_report_without_the_model():
    row = _final_row()
    plain = {**row, "parsed": {"verdict": "risk", "report_language": "ru"}}
    out = run_js(REPORT_JS, f"""(() => {{
      const I = {I};
      const row = {js(row)};
      const plain = {js(plain)};
      return {{
        deterministic: [I.isDeterministicFinal(row), I.isDeterministicFinal(plain), I.isDeterministicFinal(null)],
        hero: I.heroHtml(row, [row]),
        plainHero: I.heroHtml(plain, [plain]),
        summary: I.deterministicSummaryHtml(row),
      }};
    }})()""")
    assert out["deterministic"] == [True, False, False]
    assert "без анализа ИИ" in out["hero"] and "без анализа ИИ" not in out["plainHero"]
    assert "Определён по SLA-критериям." in out["hero"]
    # Votes on the verdict rate the model: none without it.
    assert "Вердикт верен?" not in out["hero"] and "Вердикт верен?" in out["plainHero"]
    assert "Отчёт построен без анализа ИИ" in out["summary"]
    assert "Пиковая производительность" in out["summary"] and "19:30" in out["summary"] and "19:45" in out["summary"]
    assert "ai-badge" not in out["summary"]
