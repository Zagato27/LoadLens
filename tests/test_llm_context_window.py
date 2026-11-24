# SPDX-License-Identifier: Apache-2.0
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from AI import pipeline, providers, scoring
from AI.context_budget import compact_json, text_size
from AI.i18n import with_locale
from tests.test_report_without_llm import END_TS, START_TS, _cfg, _pipeline_frames
from tests.js_harness import run_js


def profile(window, output=768):
    return {
        "provider": "openai", "max_domain_workers": 5, "self_consistency_k": 3,
        "verification_pass": False,
        "openai": {"api_base_url": "http://127.0.0.1:9/v1", "context_window_tokens": window,
                   "max_concurrent": 1, "max_attempts": 1, "generation": {"max_tokens": output}},
    }


def response(status, data):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode()
    return result


def success(text, finish="stop"):
    return response(200, {"choices": [{"message": {"content": text}, "finish_reason": finish}]})


@pytest.mark.parametrize("window, lang", [(8192, "ru"), (16384, "en")])
def test_six_domain_single_slot_report_keeps_all_candidates_within_window(monkeypatch, window, lang):
    config = _cfg("step")
    config["llm"] = profile(window)
    if window == 8192:
        del config["llm"]["openai"]["max_concurrent"]
    domains = ["jvm", "database", "kafka", "microservices", "hard_resources", "lt_framework"]
    for key in domains[:-1]:
        config["queries"][key] = {"promql_queries": ["x"], "labels": ["Heap"], "label_keys_list": [[]]}
    frames = _pipeline_frames("step")
    monkeypatch.setattr(pipeline, "fetch_and_aggregate_with_label_keys",
                        lambda _url, _start, _end, queries, *args, **kwargs: frames if len(queries) > 1 else frames[:1])
    monkeypatch.setattr(pipeline, "_load_baseline_context", lambda *args: {"available": False})
    monkeypatch.setattr(pipeline, "_load_phase_reference", lambda *args: None)
    lock = threading.Lock()
    active, calls, judges = 0, 0, 0

    def post(_url, *, json: dict, **kwargs):
        nonlocal active, calls, judges
        messages = json["messages"]
        assert sum(text_size(m["content"]) for m in messages) + json["max_tokens"] + 128 <= window
        with lock:
            active += 1
            assert active == 1
            calls += 1
        try:
            time.sleep(0.002)
            if messages[0]["content"] == scoring.judge_system_prompt(lang):
                judges += 1
                assert all(f'"index":{i}' in messages[1]["content"].replace(" ", "") for i in range(3))
                return success('{"scores":[' + ",".join(
                    f'{{"index":{i},"factual":1,"completeness":1,"specificity":1,"overall":1}}' for i in range(3)
                ) + "]}")
            return success('{"verdict":"pass","findings":[],"recommended_actions":[],"verdict_rationale":"' +
                           ("A" * 3000) + '"}')
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(providers.requests, "post", post)
    results = pipeline.uploadFromLLM(START_TS, END_TS, save_to_db=False, ef_config=config,
                                     run_meta={"test_type": "step", "service": "demo"},
                                     active_domains=domains, report_language=lang)
    assert calls == 7 * 4 and judges == 7
    assert results["final_parsed"]["deterministic_sla"]["verdict"]
    assert not results["final_parsed"].get("analysis_error")
    assert all(not results["scores"][key]["judge_meta"].get("unavailable") for key in [*domains, "final"])


def test_oversized_context_is_complete_json_and_retry_rebudgets(monkeypatch):
    payloads = []

    def post(_url, *, json, **kwargs):
        payloads.append(json)
        return success("ok", "length" if len(payloads) == 1 else "stop")

    monkeypatch.setattr(providers.requests, "post", post)
    context = json.dumps({"deterministic_sla": {"verdict": "risk"}, "timeline": list(range(100_000))})
    with providers.llm_scope(profile(8192)):
        assert providers.ask_llm_with_text_data("Use the evidence.", context, system_prompt="system") == "ok"
    assert [p["max_tokens"] for p in payloads] == [768, 1536]
    for payload in payloads:
        user = payload["messages"][1]["content"]
        assert json.loads(user.split("\n\n", 1)[1])["deterministic_sla"]["verdict"] == "risk"
        assert text_size(user) + len("system") + payload["max_tokens"] + 128 <= 8192


def test_impossible_window_and_distinct_context_errors(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return response(400, {"error": {"code": "context_length_exceeded"}})

    monkeypatch.setattr(providers.requests, "post", post)
    with providers.llm_scope(profile(8192, output=8192)), pytest.raises(providers.LLMContextOverflow):
        providers.ask_llm_with_text_data("Task", "{}")
    assert not calls
    with providers.llm_scope(profile(8192)), with_locale("en"):
        _, parsed, _ = scoring.llm_two_pass_self_consistency("Task", "{}", k=1, return_scores=True, domain_key="jvm")
        assert parsed.analysis_error.reason == "context_overflow"
        assert parsed.analysis_error.domain == "jvm" and parsed.analysis_error.stage == "candidate"
        assert "provider rejected" in parsed.analysis_error.message.lower()
    assert len(calls) == 1
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: response(400, {"error": {"code": "unknown_model"}}))
    with providers.llm_scope(profile(8192)), pytest.raises(requests.HTTPError):
        providers.ask_llm_with_text_data("Task", "{}")


def test_judge_timeout_retains_data_selection_and_stage(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ReadTimeout("request timed out")

    monkeypatch.setattr(scoring, "judge_candidates_with_llm", fail)
    parsed = scoring.parse_llm_analysis_strict('{"verdict":"pass","findings":[],"recommended_actions":[]}')
    _, selected, scores = scoring._select_best_candidate([("{}", parsed)] * 3, "{}", domain_key="jvm")
    assert selected is not None
    assert scores["judge_meta"]["unavailable"]
    assert scores["judge_meta"]["stage"] == "judge" and scores["judge_meta"]["failure_reason"] == "timeout"


def test_critic_can_repair_an_invalid_ru_draft_with_a_small_window(monkeypatch):
    sent = []

    def post(_url, *, json, **kwargs):
        messages = json["messages"]
        sent.append(messages[-1]["content"])
        assert sum(text_size(m["content"]) for m in messages) + json["max_tokens"] + 128 <= 8192
        if len(sent) == 1:
            return success("FIRST draft " + "x" * 10_000 + " LAST")
        return success('{"verdict":"pass","findings":[],"recommended_actions":[]}')

    monkeypatch.setattr(providers.requests, "post", post)
    with providers.llm_scope(profile(8192)), with_locale("ru"):
        _, result = scoring.llm_two_pass_self_consistency("Task", '{"latency_ms":250}', k=1, domain_key="jvm")
    assert len(sent) == 2 and "FIRST draft" in sent[1] and "LAST" in sent[1]
    assert '"latency_ms":250' in sent[1]
    assert result.verdict == "pass" and result.analysis_error is None


def test_tight_window_preserves_sla_verdict_or_rejects_request(monkeypatch):
    minimal = '{"deterministic_sla":{"verdict":"fail"}}'
    source = json.dumps({"flag": 1, "deterministic_sla": {"checks": list(range(50)), "verdict": "fail"}})
    assert compact_json(source, len(minimal)) == minimal
    assert compact_json(source, len(minimal) - 1) == ""
    monkeypatch.setattr(providers.requests, "post", lambda *_a, **_k: pytest.fail("over-budget POST"))
    window = 128 + 100 + len("system") + len("Task\n\n") + len(minimal) - 1
    with providers.llm_scope(profile(window, output=100)), pytest.raises(providers.LLMContextOverflow):
        providers.ask_llm_with_text_data("Task", source, system_prompt="system")


def test_backend_and_form_default_concurrency_for_custom_and_official_endpoints():
    urls = ["http://localhost:8090/v1", "https://api.openai.com/v1", "https://api.openai.com.example/v1"]
    expected = [1, 4, 1]
    assert [providers.provider_concurrency("openai", {"api_base_url": url}) for url in urls] == expected
    assert providers.provider_concurrency("openai", {"api_base_url": urls[0], "max_concurrent": 3}) == 3
    assert providers.provider_concurrency("openai", {"api_base_url": urls[1], "max_concurrent": 2}) == 2
    shown = run_js(["static/js/common.js", "static/js/ui.js", "static/js/settings.js"],
                   f"{json.dumps(urls)}.map(api_base_url => Number(LL.settings._internals.openaiConcurrencyDefault({{api_base_url}})))")
    assert shown == expected


def test_profiles_share_slots_and_restore_capacity_after_the_stricter_call(monkeypatch):
    labels = ["first", "second", "strict", "third", "fourth"]
    entered = {name: threading.Event() for name in labels}
    release = {name: threading.Event() for name in labels}
    blocked = {name: threading.Event() for name in labels}
    active = set()
    lock = threading.Lock()

    def send(name, limit):
        cfg = profile(8192)
        cfg["_profile_id"] = "small" if limit == 1 else "large"
        cfg["openai"]["max_concurrent"] = limit
        with providers.llm_scope(cfg), providers.usage_domain(name):
            return providers.ask_llm_with_text_data(name, "")

    def post(_url, *, json, **kwargs):
        name = json["messages"][-1]["content"]
        with lock:
            active.add(name)
            assert len(active) <= (1 if "strict" in active else 2)
        entered[name].set()
        try:
            assert release[name].wait(10)
            return success("ok")
        finally:
            with lock:
                active.remove(name)

    monkeypatch.setattr(providers.requests, "post", post)
    with ThreadPoolExecutor(max_workers=5) as workers:
        try:
            first = workers.submit(send, "first", 2)
            second = workers.submit(send, "second", 2)
            assert entered["first"].wait(5) and entered["second"].wait(5)
            key = providers._connection_key("openai", profile(8192)["openai"])
            condition = providers._connection_slots[key].changed
            wait = condition.wait

            def observe_wait(timeout=None):
                blocked[providers.current_usage_domain()].set()
                return wait(timeout)

            monkeypatch.setattr(condition, "wait", observe_wait)
            strict = workers.submit(send, "strict", 1)
            assert blocked["strict"].wait(5)
            third = workers.submit(send, "third", 2)
            assert blocked["third"].wait(5)
            release["first"].set()
            assert first.result(timeout=5) == "ok"
            assert not entered["strict"].is_set() and not entered["third"].is_set()
            release["second"].set()
            assert second.result(timeout=5) == "ok" and entered["strict"].wait(5)
            fourth = workers.submit(send, "fourth", 2)
            assert blocked["fourth"].wait(5)
            assert not entered["third"].is_set() and not entered["fourth"].is_set()
            release["strict"].set()
            assert strict.result(timeout=5) == "ok"
            assert entered["third"].wait(5) and entered["fourth"].wait(5)
            release["third"].set()
            release["fourth"].set()
            assert third.result(timeout=5) == fourth.result(timeout=5) == "ok"
        finally:
            for event in release.values():
                event.set()


def test_complete_model_failure_keeps_domain_and_final_causes(monkeypatch):
    config = _cfg("step")
    config["llm"] = profile(8192, output=512)
    config["llm"]["self_consistency_k"] = 1
    monkeypatch.setattr(pipeline, "fetch_and_aggregate_with_label_keys", lambda *a, **k: _pipeline_frames("step"))
    monkeypatch.setattr(pipeline, "_load_baseline_context", lambda *a: {"available": False})

    def post(*args, **kwargs):
        if providers.current_usage_domain() == "final":
            return response(400, {"error": {"code": "context_length_exceeded"}})
        raise requests.ReadTimeout("internal sample")

    monkeypatch.setattr(providers.requests, "post", post)
    result = pipeline.uploadFromLLM(START_TS, END_TS, save_to_db=False, ef_config=config,
                                    run_meta={"test_type": "step", "service": "demo"}, active_domains=["lt_framework"])
    error = result["final_parsed"]["analysis_error"]
    assert error["reason"] == "context_overflow"
    assert len(error["domain_failures"]) == 1
    detail = error["domain_failures"][0]
    assert (detail["stage"], detail["domain"], detail["reason"]) == ("candidate", "lt_framework", "timeout")
    assert "internal sample" not in detail["message"]
