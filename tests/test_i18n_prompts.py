# SPDX-License-Identifier: Apache-2.0
"""Language of a report run: prompts, LLM calls and the saved JSON (plans/i18n-plan.md, section 3.3).

A run is fixed to one language when the job is queued; every prompt file, glue string and default system prompt of
that run is in this language, in the pool threads too. The deterministic texts of the context (SLA messages, step
labels) are translated by their own modules; these tests check the prompts and the plumbing.
"""

import hashlib
import importlib.util
import json
import logging
import re
import shutil
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_smoke_api import FakeConnection, client, patch_core  # noqa: E402,F401  (fixtures)
from tests.test_test_types import END_TS, P95_LABEL, RPS_LABEL, START_TS, _pipeline_frames  # noqa: E402
from AI import pipeline as pipeline_module  # noqa: E402
from AI import prompt_store, providers, scoring  # noqa: E402
from AI.i18n import get_locale, with_locale  # noqa: E402
from AI.prompt_store import load_prompt  # noqa: E402
from loadlens_app import core  # noqa: E402
from loadlens_app.i18n import stored_report_language  # noqa: E402

CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def _load_stub():
    spec = importlib.util.spec_from_file_location("llm_stub_for_runs", ROOT_DIR / "tools" / "stubs" / "llm_stub.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STUB = _load_stub()


@pytest.fixture
def report_language(monkeypatch):
    """Sets ``appearance.language`` (the language of new reports) for the test."""

    def choose(lang: str) -> None:
        monkeypatch.setitem(core.CONFIG, "appearance", {**(core.CONFIG.get("appearance") or {}), "language": lang})

    return choose


# ---- a whole run through the real scoring and provider code ---------------------------------------------------


def _run_pipeline(monkeypatch, lang):
    """uploadFromLLM with two candidates per domain (pool threads), judge, critic and verification pass.

    The provider call itself is the LLM stub (answers in the language of the system prompt). Returns
    ``(result, spy_calls, provider_calls)``.
    """
    cfg = {
        "llm": {"include_markdown_tables_in_context": False, "self_consistency_k": 2, "max_domain_workers": 2,
                "verification_pass": True},
        "default_params": {"step": "1m", "resample_interval": "1T"},
        "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prometheus:9090"}}},
        "domain_sources": {"default": {"source": "prom"}},
        "queries": {
            "lt_framework": {"promql_queries": ["a", "b"], "label_keys_list": [[], []], "labels": [RPS_LABEL, P95_LABEL]},
            "jvm": {"promql_queries": ["c"], "label_keys_list": [[]], "labels": ["Heap"]},
        },
        "sla": {"max_performance_query": RPS_LABEL, "p95_query": P95_LABEL, "target_rps": 250, "max_p95_ms": 300},
    }
    monkeypatch.setitem(providers.CONFIG, "llm", {
        "provider": "openai",
        "openai": {"model": "stub-model", "request_min_interval_sec": 0},
        "self_consistency": {"max_candidates": 2, "parallel_candidates": True, "parallel_critics": True},
    })

    def fetch(_url, _start, _end, queries, *_args, **_kwargs):
        frames = _pipeline_frames("step")
        return frames if len(queries) == 2 else [frames[1].rename(columns={"p95_all": "heap"})]

    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", fetch)

    lock = threading.Lock()
    spy_calls, provider_calls = [], []
    main_thread = threading.get_ident()
    real_ask = providers.ask_llm_with_text_data

    def spy(user_prompt, data_context, *args, **kwargs):
        with lock:
            spy_calls.append({"prompt": user_prompt, "locale": get_locale(), "system": kwargs.get("system_prompt")})
        return real_ask(user_prompt, data_context, *args, **kwargs)

    # The fake provider must not depend on the order of calls: candidates run in pool threads, and a call counter
    # shared across threads made the critic and the verification pass appear only on some machines. Exactly one
    # analysis answer rambles (whichever comes first; the critic turns it into JSON) and every other analysis
    # answer carries an unverified number, so the chosen candidates have one and the verification pass runs.
    rambled = {"done": False}

    def provider_call(provider, messages, pcfg, system_text):
        user = messages[-1]["content"]
        body = {"messages": [{"role": "system", "content": system_text}, {"role": "user", "content": user}]}
        with lock:
            # "in_pool": the call ran in a worker thread, so get_locale() proves the ContextVar crossed the pool.
            # The number of distinct threads is not asserted: an executor reuses an idle thread when the previous
            # call finished before the next submit, which happens under load (the fake provider answers at once).
            provider_calls.append({
                "system": system_text, "locale": get_locale(), "in_pool": threading.get_ident() != main_thread,
            })
        content, kind, _domain = STUB.answer_text(body)
        critic = user.startswith(("You act as a strict report validator", "Вы выступаете как строгий валидатор"))
        if kind == "analysis" and not critic and "unverified_findings" not in user:
            with lock:
                ramble, rambled["done"] = not rambled["done"], True
            if ramble:
                return "the model rambles instead of JSON"  # the critic turns it into JSON
            if "designated_peak_performance" in user:
                answer = json.loads(content)
                answer["findings"].append({**answer["findings"][0], "id": "f2", "summary": "p95 jumped to 987654 ms"})
                return json.dumps(answer, ensure_ascii=False)  # an unverified number: the verification pass runs
        return content

    monkeypatch.setattr(scoring, "ask_llm_with_text_data", spy)
    monkeypatch.setattr(providers, "ask_llm_with_text_data", spy)
    monkeypatch.setattr(providers, "_call_provider", provider_call)
    result = pipeline_module.uploadFromLLM(
        start_ts=START_TS, end_ts=END_TS, save_to_db=False, ef_config=cfg,
        run_meta={"test_type": "step", "run_name": "run-1", "service": "demo"},
        active_domains=["lt_framework", "jvm"], report_language=lang,
    )
    return result, spy_calls, provider_calls


def _prompt_part(prompt: str) -> str:
    """Instructions of a prompt: the judge embeds the data context and the candidates after its own text."""
    for marker in ("Context (JSON):", "Контекст (JSON):"):
        if marker in prompt:
            return prompt.split(marker, 1)[0]
    return prompt


def test_en_run_sends_english_prompts_in_every_llm_call(monkeypatch):
    result, spy_calls, provider_calls = _run_pipeline(monkeypatch, "en")

    assert result["report_language"] == "en" and result["final_parsed"]["report_language"] == "en"
    assert not CYRILLIC.search(result["final_parsed"]["test_profile"]["label"])
    prompts = [call["prompt"] for call in spy_calls]
    assert any(p.startswith("You act as a strict report validator") for p in prompts)  # critic
    assert any(p.startswith("TASK: verification pass") for p in prompts)  # verification pass
    assert any("independent judge" in str(call["system"]) for call in spy_calls)  # judge
    assert any(call["in_pool"] for call in provider_calls)  # candidates ran in pool threads
    assert {call["locale"] for call in spy_calls + provider_calls} == {"en"}
    for call in provider_calls:
        assert not CYRILLIC.search(call["system"]), call["system"][:200]
    for prompt in prompts:
        assert not CYRILLIC.search(_prompt_part(prompt)), CYRILLIC.findall(_prompt_part(prompt))[:20]
    analysis = [p for p in prompts if p.startswith("HOW TO READ THE CONTEXT")]
    assert len(analysis) >= 3 and all("ANALYSIS GUIDANCE" in p and "RESPONSE FORMAT (all text fields in English" in p for p in analysis)
    # Domain prompts list the configured queries of their domain; rows added by hand have no role.
    assert any('DOMAIN METRICS' in p and '- "Heap" — meaning not set; unit not set' in p for p in analysis)
    assert any(f'- "{RPS_LABEL}" — meaning not set' in p and f'- "{P95_LABEL}" —' in p for p in analysis)
    assert not any("sections that may be present" in p for p in analysis)
    scores = result["scores"]
    assert scores["jvm"]["language_match"] == pytest.approx(1.0) and scores["final"]["language_match"] >= 0.5


def test_ru_run_keeps_russian_prompts_everywhere(monkeypatch):
    result, spy_calls, provider_calls = _run_pipeline(monkeypatch, "ru")

    assert result["report_language"] == "ru" and result["final_parsed"]["report_language"] == "ru"
    assert {call["locale"] for call in spy_calls + provider_calls} == {"ru"}
    assert all(CYRILLIC.search(call["system"]) for call in provider_calls)
    analysis = [call["prompt"] for call in spy_calls if call["prompt"].startswith("КАК ЧИТАТЬ КОНТЕКСТ")]
    assert analysis and all("ОРИЕНТИРЫ АНАЛИЗА" in p for p in analysis)
    assert any("МЕТРИКИ ДОМЕНА" in p and "- «Heap» — смысл не задан; единица не задана" in p for p in analysis)
    assert not any("какие секции могут присутствовать" in p for p in analysis)


def test_a_missing_en_prompt_falls_back_to_ru_and_the_run_completes(monkeypatch, tmp_path, caplog):
    root = tmp_path / "prompts"
    shutil.copytree(prompt_store.PROMPTS_ROOT, root)
    (root / "en" / "jvm_prompt.txt").unlink()
    (root / "en" / "judge_rubrics" / "jvm.txt").unlink()
    monkeypatch.setattr(prompt_store, "PROMPTS_ROOT", root)
    prompt_store.reload_prompts()
    try:
        with caplog.at_level(logging.WARNING, logger="AI.prompt_store"):
            result, spy_calls, _provider_calls = _run_pipeline(monkeypatch, "en")
    finally:
        prompt_store.reload_prompts()
    assert result["report_language"] == "en" and result["final_parsed"]["verdict"]
    ru_jvm = (root / "ru" / "jvm_prompt.txt").read_text(encoding="utf-8").strip()
    assert any(ru_jvm in call["prompt"] for call in spy_calls)
    missing = [record.getMessage() for record in caplog.records if "missing for language 'en'" in record.getMessage()]
    assert any("jvm_prompt.txt" in message for message in missing)
    assert any("judge_rubrics/jvm.txt" in message for message in missing)


def test_report_language_falls_back_to_run_meta_and_then_to_the_setting(report_language):
    assert pipeline_module.report_language_of("en", {"report_language": "ru"}) == "en"
    assert pipeline_module.report_language_of(None, {"report_language": "en"}) == "en"
    assert pipeline_module.report_language_of(None, None) == "ru"
    report_language("en")
    assert pipeline_module.report_language_of("de", {}) == "en"


# ---- the job carries the language to update_report and the pipeline -------------------------------------------


def test_report_job_fixes_the_language_of_the_setting_when_it_is_queued(monkeypatch, report_language):
    from loadlens_app import jobs
    from loadlens_app.blueprints import dashboard

    calls = []
    monkeypatch.setattr(dashboard, "update_report", lambda *args, **kwargs: calls.append(kwargs) or {"run_id": "x"})
    monkeypatch.setattr(dashboard, "_report_step", lambda area, service_id: "30s")
    monkeypatch.setattr(jobs.job_store, "_persist", lambda record: None)

    def no_db():
        raise RuntimeError("the job stays in memory in this test")

    monkeypatch.setattr(jobs, "_ts_conn", no_db)
    monkeypatch.setattr(threading, "Thread", lambda target, daemon=True: types.SimpleNamespace(start=target))
    report_language("en")
    dashboard.start_report_job(0, 600_000, "checkout", "shop", use_llm=False, save_to_db=False, web_only=True,
                               run_name="nightly", test_type="step", owner="eng")
    report_language("ru")
    dashboard.start_report_job(0, 600_000, "checkout", "shop", use_llm=False, save_to_db=False, web_only=True,
                               run_name="nightly-2", test_type="step", owner="eng", report_language="en")
    dashboard.start_report_job(0, 600_000, "checkout", "shop", use_llm=False, save_to_db=False, web_only=True,
                               run_name="nightly-3", test_type="step", owner="eng")
    assert [call["report_language"] for call in calls] == ["en", "en", "ru"]


def test_update_report_puts_the_language_into_run_meta_and_sends_only_real_overrides(monkeypatch, tmp_path):
    import update_page

    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(tmp_path / "settings_runtime.json"))
    captured = {}
    ru_default = load_prompt("jvm_prompt.txt", "ru")
    en_default = load_prompt("kafka_prompt.txt", "en")
    monkeypatch.setattr(update_page, "_active_metrics_config_now", lambda: {"shop": {"services": {"checkout": {"page_parent_id": "1", "page_sample_id": "2"}}}})
    monkeypatch.setattr(update_page, "_effective_config_for_scope", lambda area, service: {"queries": {}, "sla": {}})
    monkeypatch.setattr(update_page, "_resolve_services_for_area", lambda area: ["checkout"])
    monkeypatch.setattr(update_page, "_prompts_override_for_area", lambda area: {"jvm": ru_default, "kafka": en_default, "database": "own text", "overall": " "})
    monkeypatch.setattr(update_page, "_service_prompts_override", lambda area, service: {"database": "service text", "jvm": ""})

    def pipeline(start, end, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(update_page, "uploadFromLLM", pipeline)
    update_page.update_report(0, 600_000, "checkout", use_llm=True, web_only=True, project_area="shop", report_language="en")
    assert captured["report_language"] == "en" and captured["run_meta"]["report_language"] == "en"
    assert captured["prompts_override"] == {"database": "service text"}
    with with_locale("en"):
        update_page.update_report(0, 600_000, "checkout", use_llm=True, web_only=True, project_area="shop")
    assert captured["report_language"] == "en"  # no argument: the current locale (the worker setting)
    update_page.update_report(0, 600_000, "checkout", use_llm=True, web_only=True, project_area="shop")
    assert captured["report_language"] == "ru" and captured["run_meta"]["report_language"] == "ru"


def test_report_task_passes_the_language_and_defaults_to_ru_for_old_messages(monkeypatch):
    from loadlens_app import tasks

    calls = []
    fake_dashboard = types.SimpleNamespace(
        update_report=lambda *args, **kwargs: calls.append(kwargs) or {"run_id": "r"},
        _job_failure_message=lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(tasks, "_dashboard", lambda: fake_dashboard)
    locales = []
    monkeypatch.setattr(tasks, "_run_job", lambda task, job_id, work, failure, locale=None: locales.append(locale) or work(lambda *a, **k: None))
    base = dict(start=0, end=1, service="s", project_area="a", use_llm=True, save_to_db=False, web_only=True, run_name="n", test_type="step")
    tasks.report_task.run("job-1", **base)
    tasks.report_task.run("job-2", **base, report_language="en")
    assert [call["report_language"] for call in calls] == ["ru", "en"]
    assert locales == ["ru", "en"]  # the job itself runs in the report language


# ---- settings: prompts in the language of reports --------------------------------------------------------------


def test_prompt_defaults_follow_the_language_parameter_and_the_setting(client, report_language):
    ru = client.get("/prompts/defaults").get_json()
    assert ru["language"] == "ru" and ru["domains"]["jvm"] == load_prompt("jvm_prompt.txt", "ru")
    en = client.get("/prompts/defaults", query_string={"lang": "en"}).get_json()
    assert en["language"] == "en" and en["domains"]["jvm"] == load_prompt("jvm_prompt.txt", "en")
    assert "judge" not in en["domains"] and "critic" not in en["domains"]
    report_language("en")
    assert client.get("/prompts/defaults").get_json()["language"] == "en"
    assert client.get("/prompts/defaults", query_string={"lang": "xx"}).get_json()["language"] == "en"


def test_prompts_report_sources_override_language_and_ignore_copies_of_defaults(client, report_language):
    assert client.post("/prompts", json={"area": "demo", "domain": "jvm", "text": "Свой промпт для JVM"}).status_code == 200
    assert client.post("/prompts", json={"area": "demo", "domain": "kafka", "text": load_prompt("kafka_prompt.txt", "en")}).status_code == 200
    body = client.get("/prompts", query_string={"area": "demo"}).get_json()
    assert body["language"] == "ru"
    assert body["domains"]["jvm"] == "Свой промпт для JVM" and body["sources"]["jvm"] == "area"
    assert body["override_language"] == {"jvm": "ru"}
    # A stored copy of the en default is no override: the default of the current language is shown.
    assert body["sources"]["kafka"] == "default" and body["domains"]["kafka"] == load_prompt("kafka_prompt.txt", "ru")
    assert "judge" not in body["sources"]
    report_language("en")
    body = client.get("/prompts", query_string={"area": "demo"}).get_json()
    assert body["language"] == "en" and body["domains"]["kafka"] == load_prompt("kafka_prompt.txt", "en")
    assert body["domains"]["jvm"] == "Свой промпт для JVM"  # an override keeps its own language


# ---- forecast explanation and context import --------------------------------------------------------------------


def _explanation_env(monkeypatch, stored_language):
    from AI.db_store import RunAnalyses, StoredExplanation, StoredFinalReport
    from loadlens_app import forecast_explanation as explain
    from tests.test_smoke_api import _explained_report, _finding, _forecast_frames, _patch_forecast

    frames = _forecast_frames()
    _patch_forecast(monkeypatch, _explained_report(frames), frames)
    stored, asked = {}, []
    sla_details = {"report_language": stored_language} if stored_language else {}
    final = StoredFinalReport(run_id="abc", run_name="nightly", service="demo", test_type="step", project_area="",
                              start_ms=None, end_ms=None, context={}, sla_details=sla_details)
    monkeypatch.setattr(explain, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(explain, "_last_started", {})
    monkeypatch.setattr(explain, "load_final_report", lambda *args: final)
    monkeypatch.setattr(explain, "load_run_analyses", lambda *args: RunAnalyses(findings=[_finding("jvm:f1", (80, 99), "CPU 95 %")], system_context={}))
    monkeypatch.setattr(explain, "load_forecast_explanation", lambda conn, schema, table, run_id: stored.get(run_id))
    monkeypatch.setattr(explain, "save_forecast_explanation",
                        lambda conn, schema, table, run_id, digest, payload: stored.update({run_id: StoredExplanation(run_id, digest, payload, None)}))
    monkeypatch.setitem(explain.CONFIG, "llm", {"provider": "openai", "openai": {"model": "test-model"}})
    answer = {"headline": "CPU", "cause": "CPU 95 %", "key_facts": [], "scaling": {"verdict": "helps", "reason": "CPU"},
              "evidence": [{"finding_id": "jvm:f1", "role": "CPU"}], "scaling_risks": [], "next_checks": [], "confidence": "medium"}

    def ask(prompt, data, **kwargs):
        asked.append({"prompt": prompt, "data": json.loads(data), "system": kwargs.get("system_prompt"), "locale": get_locale()})
        return json.dumps(answer)

    monkeypatch.setattr(explain, "ask_llm_with_text_data", ask)
    return explain, asked


def test_forecast_explanation_runs_in_the_language_of_its_report(monkeypatch):
    explain, asked = _explanation_env(monkeypatch, "en")
    view = explain.generate_explanation("abc")
    assert view.status == "ready"
    call = asked[0]
    assert call["prompt"] == load_prompt("forecast_limit_prompt.txt", "en") and call["locale"] == "en"
    assert not CYRILLIC.search(call["system"]) and not CYRILLIC.search(call["data"]["facts"]["limit"]["kind"])


def test_forecast_explanation_of_an_old_report_keeps_its_russian_prompt_and_hash(monkeypatch):
    explain, asked = _explanation_env(monkeypatch, None)
    context = explain._context("abc")
    ru_prompt = (prompt_store.PROMPTS_ROOT / "ru" / "forecast_limit_prompt.txt").read_text(encoding="utf-8")
    expected = hashlib.sha256(json.dumps({"prompt": ru_prompt, "payload": context.payload}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    assert context.lang == "ru" and context.inputs_hash == expected
    assert context.payload["facts"]["limit"]["kind"] in {"провалы времени отклика", "потолок модели USL"}
    explain.generate_explanation("abc")
    assert asked[0]["prompt"] == ru_prompt and CYRILLIC.search(asked[0]["system"])


def test_context_import_uses_the_global_language_and_its_delimiters(monkeypatch, report_language):
    from loadlens_app import context_import as ci

    calls = []

    def ask(prompt, document, llm_config=None, **kwargs):
        calls.append({"prompt": prompt, "document": document, "system": kwargs.get("system_prompt")})
        return json.dumps({"system": {"name": "Shop"}})

    monkeypatch.setattr(providers, "ask_llm_with_text_data", ask)
    llm = {"provider": "openai", "openai": {"model": "m"}}
    ci.ask_model("The checkout service creates orders.", llm)
    report_language("en")
    ci.ask_model("The checkout service creates orders.", llm)
    ru, en = calls
    assert ru["document"].startswith("<<<ДОКУМЕНТ\n") and ru["document"].endswith("\nДОКУМЕНТ>>>")
    assert ru["prompt"] == load_prompt("context_import_prompt.txt", "ru") and ru["system"] == ci.SYSTEM_PROMPT
    assert en["document"] == "<<<DOCUMENT\nThe checkout service creates orders.\nDOCUMENT>>>"
    assert en["prompt"] == load_prompt("context_import_prompt.txt", "en")
    assert not CYRILLIC.search(en["prompt"] + en["system"])


# ---- providers, stored reports, parsers ----------------------------------------------------------------------------


def test_default_system_prompts_by_language():
    assert providers.default_system_prompt("ru", force_json=False) == load_prompt("_system.txt", "ru").strip()
    en = providers.default_system_prompt("en")
    assert en.startswith("You are a load testing engineer") and "pass | risk | fail | no_data" in en
    assert not CYRILLIC.search(en)
    with with_locale("en"):
        assert providers.default_system_prompt() == en


def test_perplexity_language_preference_differing_from_the_run_warns_once_per_run(caplog):
    providers.reset_usage()
    with caplog.at_level(logging.WARNING, logger="AI.providers"):
        providers._check_language_preference({"language_preference": "ru"}, "en")
        providers._check_language_preference({"language_preference": "ru"}, "en")
        providers._check_language_preference({"language_preference": "English"}, "en")
        providers._check_language_preference({}, "en")
    assert len([r for r in caplog.records if "language_preference" in r.getMessage()]) == 1
    providers.reset_usage()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="AI.providers"):
        providers._check_language_preference({"language_preference": "ru"}, "en")
    assert len([r for r in caplog.records if "language_preference" in r.getMessage()]) == 1


def test_stored_report_language_reads_the_report_or_defaults_to_ru():
    assert stored_report_language({"verdict": "pass", "report_language": "en"}) == "en"
    assert stored_report_language({"parsed": json.dumps({"report_language": "EN"})}) == "en"
    assert stored_report_language(types.SimpleNamespace(sla_details={"report_language": "en"})) == "en"
    rows = [{"domain": "jvm", "parsed": {}}, {"domain": "final", "parsed": {"report_language": "en"}}]
    assert stored_report_language(rows) == "en"
    assert stored_report_language({"verdict": "Успешно"}) == "ru"
    assert stored_report_language(None) == "ru" and stored_report_language([]) == "ru"


def test_step_references_and_sla_markers_of_both_languages_are_parsed():
    from AI.context_pack import _QUERY_RE, _SERIES_RU_RE
    from AI.verification import _STEP_REFERENCE_RE

    for text in ("всплеск 2", "фон 1", "spike 2", "background 1", "step 3", "ступень 3", "interval 4"):
        assert _STEP_REFERENCE_RE.sub(" ", f"p95 on {text} reached").split() == ["p95", "on", "reached"], text
    assert _QUERY_RE.search("p95 по query «LT p95» выше порога").group(1) == "LT p95"
    assert _QUERY_RE.search("запрос «LT RPS»").group(1) == "LT RPS"
    assert _SERIES_RU_RE.search("series «sum_all»").group(1) == "sum_all"
    assert _SERIES_RU_RE.search("серия «sum_all»").group(1) == "sum_all"


def test_pipeline_glue_strings_follow_the_report_language():
    brief_ru = pipeline_module._system_context_prompt_summary({"system": {"name": "Shop"}}, "ru")
    brief_en = pipeline_module._system_context_prompt_summary({"system": {"name": "Shop"}}, "en")
    assert brief_ru == "- Система: Shop" and brief_en == "- System: Shop"
    assert pipeline_module._system_context_prompt_summary(None, "en") == "System context is not set."
    # The "not set" text of any language adds no reference block.
    assert pipeline_module._augment_prompt_with_system_context("P", "Контекст системы не задан.", "en") == "P"
    assert pipeline_module._augment_prompt_with_system_context("P", "System context is not set.", "ru") == "P"
    assert pipeline_module._augment_prompt_with_system_context("P", brief_en, "en").startswith("REFERENCE CONTEXT")
    from AI.prompt_assembly import build_analysis_guidance

    guidance = build_analysis_guidance({"target_rps": 100, "max_p95_ms": 300}, None, "spike", lang="en")
    assert guidance.startswith("ANALYSIS GUIDANCE") and "[Test profile: spikes (spike)]" in guidance
    assert "300 ms" in guidance and not CYRILLIC.search(guidance)
    assert "No numeric SLA is set" in build_analysis_guidance({}, None, None, lang="en")


def test_baseline_reasons_are_written_in_the_report_language():
    from AI.prompt_store import text

    reason = pipeline_module._load_baseline_context({}, {"service": "s"}, {}, False, 0)["reason"]
    assert reason == "хранилище метрик не используется для этого запуска"
    with with_locale("en"):
        assert pipeline_module._load_baseline_context({}, {"service": "s"}, {}, False, 0)["reason"] == "the metrics storage is not used for this run"
    # The former ru wording is kept as it was built before ("{label}а").
    assert text("baseline.none_success", "ru", scope=text("baseline.scope_area", "ru", name="shop"), type_note=text("baseline.type_note", "ru", type="step")) == (
        "для области «shop» нет более раннего успешный прогона типа «step»"
    )
    assert text("baseline.none_any", "en", scope=text("baseline.scope_service", "en", name="api"), type_note="") == 'the "api" service has no earlier run'
