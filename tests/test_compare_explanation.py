# SPDX-License-Identifier: Apache-2.0
"""AI explanation of a comparison of two runs (plans/compare-explanation-plan.md).

Direction of the metrics, the outcome and caveat rules, the facts for the model, the cache and its statuses, the
language of run A, the API and the Confluence page. Offline: storage, the model and Confluence are stubs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tests.test_smoke_api import FakeConnection, RecordingConnection, client, patch_core  # noqa: F401
from AI import db_store
from AI.compare_explanation import CompareExplanation, check_comparison
from AI.db_store import StoredPairExplanation, StoredRunReport, StoredRunRow
from AI.i18n import with_locale
from AI.metric_direction import (
    HIGHER_ROLES,
    LOWER_ROLES,
    NEUTRAL_ROLES,
    assess,
    change_pct,
    label_direction,
    metric_direction,
)
from AI.prompt_store import load_prompt
from AI.query_packs.roles import ROLES
from loadlens_app import compare_confluence as cc
from loadlens_app import compare_explanation as cx
from loadlens_app import core
from tests.i18n_helpers import CYRILLIC

CREATED_A = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
CREATED_B = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


# ---- helpers ---------------------------------------------------------------------------------------------------


def _finding(fid, summary, *, severity="high", component="orders-api", verification="verified", start="2026-11-09T12:10:00+00:00"):
    return {
        "id": fid, "summary": summary, "severity": severity, "component": component, "start_time": start,
        "end_time": start, "evidence_summary": "", "verification": {"status": verification},
    }


def _row(domain, name, *, parsed, verdict="pass", service="checkout", test_type="step", minutes=60, created=CREATED_A, system=None):
    return StoredRunRow(
        domain=domain, run_id=f"id-{name}", service=service, test_type=test_type, project_area="",
        start_ms=0, end_ms=minutes * 60_000, verdict=verdict, sla_verdict=verdict, parsed=parsed,
        sla_details={}, system_context=system or {}, created_at=created,
    )


def _report(name, *, verdict="pass", rps=1000.0, kind="peak", service="checkout", test_type="step", minutes=60,
            created=CREATED_A, lang="ru", findings=None, manual=None, domain_verdicts=None, system=None):
    parsed = {"report_language": lang}
    if kind == "peak":
        parsed["peak_performance"] = {"max_rps": rps}
    elif kind == "held":
        parsed["stability_under_load"] = {"actual_rps": rps}
    elif kind == "spike":
        parsed["spike_analysis"] = {"max_spike_rps": rps}
    rows = {"final": _row("final", name, parsed=parsed, verdict=verdict, service=service, test_type=test_type,
                          minutes=minutes, created=created, system=system)}
    for domain, items in (findings or {}).items():
        rows[domain] = _row(domain, name, parsed={"findings": items}, verdict=(domain_verdicts or {}).get(domain, "pass"),
                            service=service, test_type=test_type, created=created)
    return StoredRunReport(run_name=name, rows=rows, manual_verdict=manual)


def _facts(side, report, name=None):
    return cx.run_facts(side, name or (report.run_name if report else side), report)


def _metrics(*rows, roles=None):
    return cx.summarize_metrics(rows, roles or {})


ANSWER = {
    "headline": "A держит больше RPS, чем B",
    "summary": "Максимум вырос с 1000 до 1200 RPS; похоже, помог пул соединений базы.",
    "key_changes": [{"label": "Максимум RPS", "value_a": "1200 RPS", "value_b": "1000 RPS", "assessment": "better"}],
    "causes": ["Вероятно, в B пул соединений исчерпывался"],
    "evidence": [{"finding_id": "B:database:f1", "role": "пул исчерпан в B"}, {"finding_id": "C:jvm:f9", "role": "лишняя ссылка"}],
    "caveats": [],
    "next_checks": ["Повторить тест на 1200 RPS"],
    "confidence": "medium",
}


@pytest.fixture(autouse=True)
def _fresh_slots():
    cx._running.clear()
    cx._last_started.clear()
    yield
    cx._running.clear()
    cx._last_started.clear()


@pytest.fixture
def pair(monkeypatch):
    """Stub storage, roles and model of the explanation; returns the mutable state of the pair."""
    state = {
        "reports": {
            "A": _report("night-2", rps=1200.0, findings={"jvm": [_finding("f1", "GC 120 мс на пике")]}),
            "B": _report("night-1", rps=1000.0, created=CREATED_B, findings={
                "database": [_finding("f1", "Пул соединений исчерпан, ожидание 800 мс", component="postgres")],
                "jvm": [_finding("f2", "CPU 12345 %", verification="unverified")],
            }),
        },
        "rows": [("lt_framework", "RPS", 1200.0, 1000.0), ("lt_framework", "p95", 300.0, 400.0)],
        "stored": {},
        "calls": [],
        "answer": ANSWER,
    }
    monkeypatch.setattr(cx, "_load_reports", lambda run_a, run_b: (state["reports"], state["rows"]))
    monkeypatch.setattr(cx, "query_roles", lambda area, service=None: {"lt_framework": {"RPS": "lt.rps_total", "p95": "lt.p95_total"}})
    monkeypatch.setattr(cx, "_find_area_for_service", lambda service: None)
    monkeypatch.setattr(cx, "_load_cached", lambda run_a, run_b: state["stored"].get((run_a, run_b)))
    monkeypatch.setattr(
        cx, "_save",
        lambda run_a, run_b, digest, payload: state["stored"].update({(run_a, run_b): StoredPairExplanation(run_a, run_b, digest, payload, None)}),
    )
    monkeypatch.setitem(cx.CONFIG, "llm", {"provider": "openai", "openai": {"model": "test-model"}})

    def ask(prompt, data, **kwargs):
        state["calls"].append({"prompt": prompt, "data": json.loads(data), **kwargs})
        return json.dumps(state["answer"], ensure_ascii=False)

    monkeypatch.setattr(cx, "ask_llm_with_text_data", ask)
    return state


# ---- direction of the metrics ----------------------------------------------------------------------------------


def test_every_role_has_exactly_one_direction():
    sets = (HIGHER_ROLES, LOWER_ROLES, NEUTRAL_ROLES)
    assert set().union(*sets) == set(ROLES), "a new role needs a direction in AI/metric_direction.py"
    assert not (HIGHER_ROLES & LOWER_ROLES) and not (HIGHER_ROLES & NEUTRAL_ROLES) and not (LOWER_ROLES & NEUTRAL_ROLES)


@pytest.mark.parametrize(("label", "expected"), [
    ("LT: RPS", "higher"), ("Throughput, req/s", "higher"), ("Requests per time unit", "higher"),
    ("Response time p95", "lower"), ("CPU usage", "lower"), ("Ошибки HTTP", "lower"), ("Requests with errors", "lower"),
    ("VUs", "neutral"), ("", "neutral"),
])
def test_label_direction_follows_the_words_of_the_compare_page(label, expected):
    assert label_direction(label) == expected


def test_the_role_wins_over_the_label():
    assert metric_direction("Hits", "db.cache_hit_pct") == "higher"
    assert metric_direction("Requests", "lt.vus") == "neutral"
    assert metric_direction("Requests", "custom.role") == "higher"  # an unknown role falls back to the words


def test_change_and_assessment():
    assert change_pct(300.0, 270.0) == pytest.approx(11.111, rel=1e-3)  # A relative to B
    assert change_pct(5.0, 0.0) == 500.0 and change_pct(None, 1.0) is None
    assert assess(9.9, "higher", 10.0) == "same"
    assert (assess(10.0, "higher", 10.0), assess(-10.0, "higher", 10.0)) == ("better", "worse")
    assert (assess(-25.0, "lower", 10.0), assess(25.0, "lower", 10.0)) == ("better", "worse")
    assert assess(50.0, "neutral", 10.0) == "unknown" and assess(None, "lower", 10.0) == "unknown"


# ---- facts, outcome, caveats -----------------------------------------------------------------------------------


def test_key_rps_and_its_kind():
    assert cx.key_rps({"peak_performance": {"max_rps": "~1200"}, "stability_under_load": {"actual_rps": 500}}) == (500.0, "held")
    assert cx.key_rps({"spike_analysis": {"max_spike_rps": 900}}) == (900.0, "spike")
    assert cx.key_rps({}) == (None, "")


def test_run_facts_of_a_report_and_of_a_run_without_one():
    facts = _facts("A", _report("night-2", manual="fail", minutes=90, findings={"jvm": [_finding("f1", "x"), _finding("f2", "y", verification="unverified")]}))
    assert (facts.has_report, facts.verdict, facts.sla_verdict, facts.window_min) == (True, "fail", "pass", 90.0)
    assert (facts.key_rps, facts.key_rps_kind, facts.findings_total, facts.type_key) == (1000.0, "peak", 1, "step")
    bare = cx.run_facts("B", "night-1", None)
    assert (bare.has_report, bare.verdict, bare.key_rps) == (False, None, None)
    assert _facts("A", _report("x", test_type="")).type_key == "step"  # saved without a type
    assert _facts("A", _report("x", test_type="Стабильность")).type_key == "soak"


SAME_METRICS = _metrics(("lt_framework", "RPS", 1000.0, 1000.0))


@pytest.mark.parametrize(("a", "b", "metrics", "code", "reason"), [
    (dict(service="checkout"), dict(service="cart"), SAME_METRICS, "incomparable", "scope"),
    (dict(test_type="step"), dict(test_type="soak"), SAME_METRICS, "incomparable", "scope"),
    (dict(verdict="pass"), dict(verdict="fail"), SAME_METRICS, "better", "verdict"),
    (dict(verdict="risk"), dict(verdict="pass"), SAME_METRICS, "worse", "verdict"),
    (dict(manual="pass", verdict="fail"), dict(verdict="risk"), SAME_METRICS, "better", "verdict"),
    (dict(rps=1100.0), dict(rps=1000.0), SAME_METRICS, "better", "rps"),
    (dict(rps=900.0), dict(rps=1000.0), SAME_METRICS, "worse", "rps"),
    # Within the tolerance the metrics decide.
    (dict(rps=1030.0), dict(rps=1000.0), _metrics(("lt", "p95", 300.0, 400.0)), "better", "metrics"),
    # The verdict is unknown: the step is skipped.
    (dict(verdict="no_data", rps=800.0), dict(verdict="pass"), SAME_METRICS, "worse", "rps"),
    # A held load is the load of the test, not a result.
    (dict(kind="held", rps=600.0), dict(kind="held", rps=500.0), _metrics(("lt", "p95", 500.0, 400.0)), "worse", "metrics"),
    (dict(kind="held"), dict(kind="peak", rps=2000.0), SAME_METRICS, "same", "same"),
    (dict(), dict(), _metrics(("lt", "p95", 300.0, 400.0), ("lt", "RPS", 800.0, 1000.0)), "mixed", "metrics"),
    (dict(), dict(), _metrics(("lt", "VUs", 300.0, 100.0)), "same", "same"),
])
def test_outcome_rules(a, b, metrics, code, reason):
    outcome = cx.outcome_of(_facts("A", _report("a", **a)), _facts("B", _report("b", **b)), metrics)
    assert (outcome.code, [note.code for note in outcome.reasons]) == (code, [reason])


def test_outcome_texts():
    outcome = cx.outcome_of(_facts("A", _report("a", verdict="pass")), _facts("B", _report("b", verdict="fail")), SAME_METRICS)
    assert cx.outcome_title(outcome.code) == "A лучше B"
    assert cx.reason_text(outcome.reasons[0]) == "Итоговый вердикт: A — Успешно, B — Провал"
    with with_locale("en"):
        assert cx.reason_text(outcome.reasons[0]) == "Final verdict: A Passed, B Failed"
    rps = cx.outcome_of(_facts("A", _report("a", rps=1100.0)), _facts("B", _report("b", rps=1000.0)), SAME_METRICS)
    assert cx.reason_text(rps.reasons[0]) == "Максимум RPS: A — 1100, B — 1000 (+10,0 %)"


def test_flags():
    a = _facts("A", _report("night-2", service="checkout", test_type="step", minutes=60, created=CREATED_B, kind="held", rps=600.0))
    b = _facts("B", _report("night-1", service="cart", test_type="soak", minutes=120, created=CREATED_A, kind="held", rps=500.0))
    metrics = _metrics(("lt", "RPS", 1.0, None), ("lt", "p95", None, 2.0))
    codes = [note.code for note in cx.flags_of(a, b, metrics)]
    assert codes == ["service", "test_type", "window", "order", "no_findings", "no_findings", "load", "metrics_partial"]
    texts = [cx.flag_text(note) for note in cx.flags_of(a, b, metrics)]
    assert texts[0] == "Прогоны разных сервисов: checkout и cart"
    assert texts[1] == "Разные типы теста: Ступенчатый (max perf) и Долговременный (soak)"
    assert texts[2] == "Окна разной длительности: 60 и 120 мин" and texts[-1] == "Часть метрик есть только в одном прогоне: 2"
    no_report = cx.flags_of(_facts("A", _report("a")), cx.run_facts("B", "night-1", None), SAME_METRICS)
    assert [note.code for note in no_report] == ["no_findings", "no_report"]
    kinds = cx.flags_of(_facts("A", _report("a", kind="held")), _facts("B", _report("b")), SAME_METRICS)
    assert "rps_kind" in [note.code for note in kinds]


def test_summary_of_metrics_and_the_largest_changes():
    rows = [("db", f"query {i}", 100.0 + 20 * i, 100.0) for i in range(1, 8)]  # +20 … +140 %, no direction
    rows += [("lt_framework", "p95", 150.0, 100.0), ("lt_framework", "RPS", 101.0, 100.0), ("jvm", "heap", None, 5.0)]
    summary = cx.summarize_metrics(rows, {"lt_framework": {"p95": "lt.p95_total", "RPS": "lt.rps_total"}})
    assert summary.counts() == {"better": 0, "worse": 1, "same": 1, "unknown": 7}
    assert summary.only_in_b == [("jvm", "heap")] and summary.only_in_a == []
    top = summary.top()
    assert top[0].metric == "p95" and top[0].assessment == "worse"  # assessed changes go first
    assert [change.domain for change in top[1:]] == ["db"] * cx.PER_DOMAIN_CHANGES
    assert top[1].change_pct == pytest.approx(140.0)


def test_findings_given_to_the_model():
    report = _report("a", findings={
        "jvm": [_finding("f1", "low one", severity="low"), _finding("f2", "bad numbers", verification="unverified")],
        "database": [_finding("f1", "critical one", severity="critical")],
    })
    picked = cx.picked_findings(report)
    assert [finding.ref for finding in picked] == ["database:f1", "jvm:f1"]


# ---- context, cache, statuses ----------------------------------------------------------------------------------


def test_project_routing_of_comparison_explanations(pair, monkeypatch):
    from AI.providers import current_llm_config

    def set_area(side, area):
        report = pair["reports"][side]
        report.rows["final"] = replace(report.rows["final"], project_area=area)

    project_llm = {"provider": "openai", "openai": {"model": "project-model"}}
    core.CONFIG_RUNTIME_PATH.write_text(json.dumps({"per_area": {"shop": {"llm": project_llm}, "other": {}}}), encoding="utf-8")
    monkeypatch.setattr(cx, "ask_llm_with_text_data", lambda *args, **kwargs: current_llm_config()["openai"]["model"])
    set_area("A", "shop")
    set_area("B", "shop")
    assert cx._ask_model(cx.build_context("night-2", "night-1")) == "project-model"
    set_area("B", "other")
    assert cx._ask_model(cx.build_context("night-2", "night-1")) == "test-model"


def test_explanation_is_generated_checked_and_cached(pair):
    first = cx.explanation_view("night-2", "night-1").to_dict()
    assert (first["status"], first["model"], first["outcome"]["code"], first["language"]) == ("missing", "test-model", "better", "ru")
    assert first["outcome"]["title"] == "A лучше B"
    assert [item["id"] for item in first["findings"]] == ["A:jvm:f1", "B:database:f1"]
    assert first["metrics"]["counts"] == {"better": 2, "worse": 0, "same": 0, "unknown": 0}
    assert first["flags"] == []

    view = cx.generate_explanation("night-2", "night-1").to_dict()
    assert view["status"] == "ready" and view["explanation"]["confidence"] == "medium"
    assert view["check"]["status"] == "verified" and view["check"]["unknown_findings"] == ["C:jvm:f9"]
    [call] = pair["calls"]
    assert call["prompt"] == load_prompt("compare_prompt.txt", "ru")
    assert call["llm_config"] == {"force_json": True} and "русском" in call["system_prompt"]
    facts = call["data"]["facts"]
    assert facts["outcome"]["result"] == "A лучше B" and facts["runs"]["A"]["key_rps"] == 1200.0
    assert [change["assessment"] for change in facts["metrics"]["top_changes"]] == ["лучше", "лучше"]
    assert [item["id"] for item in call["data"]["findings"]["B"]] == ["B:database:f1"]

    assert cx.explanation_view("night-2", "night-1").status == "ready"
    with pytest.raises(cx.ExplanationError) as again:
        cx.generate_explanation("night-2", "night-1")
    assert again.value.code == "too_often" and again.value.status_code == 429


def test_the_explanation_goes_stale_when_the_facts_change(pair):
    cx.generate_explanation("night-2", "night-1")
    pair["rows"] = [("lt_framework", "RPS", 1300.0, 1000.0)]
    stale = cx.explanation_view("night-2", "night-1")
    assert stale.status == "stale" and stale.explanation is not None
    assert stale.message == "Отчёты или метрики прогонов изменились после разбора — обновите его"
    key = ("night-2", "night-1")
    pair["stored"][key] = StoredPairExplanation(*key, "x", {"explanation": {"headline": "старый формат"}, "check": {}}, None)
    old = cx.explanation_view(*key)
    assert old.status == "stale" and old.explanation is None and "прежнем формате" in old.message


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
def test_throughput_is_checked_against_its_metric_and_run(structured, swapped):
    # The other metric has the opposite pair, so even per-run pooling would accept the wrong numbers.
    facts = {"runs": {"A": {"key_rps": 1200}, "B": {"key_rps": 1000}},
             "metrics": {"top_changes": [{"metric": "Heap", "A": 1000, "B": 1200}]}}
    a, b = (1000, 1200) if swapped else (1200, 1000)
    explanation = CompareExplanation(
        headline="Results", confidence="medium",
        summary="Results" if structured else f"Throughput: A reached {a} RPS and B reached {b} RPS.",
        key_changes=[{"label": "Throughput", "value_a": f"{a} RPS", "value_b": f"{b} RPS"}] if structured else [],
    )
    result = check_comparison(explanation, facts, [])
    assert (result.status, result.claims_matched, result.claims_total) == (
        ("unverified", 0, 2) if swapped else ("verified", 2, 2)
    )
    if swapped:
        assert any("A" in item and "1000" in item for item in result.unmatched)
        assert any("B" in item and "1200" in item for item in result.unmatched)


def test_cached_legacy_checks_are_rechecked_without_a_model(pair):
    pair["answer"] = {**ANSWER, "key_changes": [{"label": "RPS", "value_a": "1000 RPS", "value_b": "1200 RPS"}]}
    generated = cx.generate_explanation("night-2", "night-1")
    assert generated.check.status == "unverified"
    stored = pair["stored"][("night-2", "night-1")].payload
    stored.pop("check_version", None)
    stored["check"].update(status="verified", unmatched=[])
    assert cx.explanation_view("night-2", "night-1").check.status == "unverified"
    assert len(pair["calls"]) == 1


def test_nearby_run_values_cannot_be_swapped_within_the_rounding_tolerance():
    answer = CompareExplanation(headline="Results", summary="101 RPS (A) and 100 RPS (B).", confidence="low")
    facts = {"runs": {"A": {"key_rps": 100}, "B": {"key_rps": 101}}}
    result = check_comparison(answer, facts, [])
    assert (result.status, result.claims_matched, result.claims_total) == ("unverified", 0, 2)


@pytest.mark.parametrize("facts", [
    {"runs": {"A": {"key_rps": 1200}, "B": {"key_rps": 1000}}},
    {"metrics": {"top_changes": [{"metric": "RPS", "A": 1200, "B": 1000}]}},
])
def test_russian_throughput_label_with_grouped_values(facts):
    for a, b, expected in (("1 200 RPS", "1 000 RPS", 2), ("1 000 RPS", "1 200 RPS", 0)):
        answer = CompareExplanation(headline="Results", summary="Results", confidence="low", key_changes=[
            {"label": "Пропускная способность", "value_a": a, "value_b": b},
        ])
        result = check_comparison(answer, facts, [], lang="ru")
        assert (result.claims_matched, result.claims_total) == (expected, 2)
        assert result.status == ("verified" if expected else "unverified")


@pytest.mark.parametrize(("lang", "summary"), [
    ("en", "From 1,200 RPS in A to 1,000 RPS in B."),
    ("ru", "В прогоне A 1200 RPS, в B 1000 RPS."),
])
def test_explicit_run_markers_in_both_report_languages(lang, summary):
    answer = CompareExplanation(headline="Results", summary=summary, confidence="low")
    facts = {"runs": {"A": {"key_rps": 1200}, "B": {"key_rps": 1000}}}
    correct = check_comparison(answer, facts, [], lang=lang)
    assert (correct.status, correct.claims_matched, correct.claims_total) == ("verified", 2, 2)
    facts["runs"]["A"], facts["runs"]["B"] = facts["runs"]["B"], facts["runs"]["A"]
    assert check_comparison(answer, facts, [], lang=lang).claims_matched == 0


def test_percentile_identifiers_and_decimal_commas_in_numeric_changes():
    answer = CompareExplanation(headline="Results", summary="Results", confidence="low", key_changes=[
        {"label": "p95", "value_a": "p95 100,500 ms", "value_b": "p95 80 ms"},
    ])
    facts = {"metrics": {"top_changes": [{"metric": "p95", "A": 100.5, "B": 80}]}}
    result = check_comparison(answer, facts, [], lang="ru")
    assert (result.status, result.claims_matched, result.claims_total) == ("verified", 2, 2)


def test_joint_run_label_does_not_assign_shared_prose_to_the_last_run():
    answer = CompareExplanation(headline="Results", summary="A and B: 1000 RPS.", confidence="low")
    facts = {"runs": {"A": {"key_rps": 1200}, "B": {"key_rps": 1000}}}
    assert check_comparison(answer, facts, []).status == "unverified"


def test_nothing_to_compare(pair):
    pair["reports"] = {"A": None, "B": _report("night-1")}
    pair["rows"] = [("lt", "RPS", 10.0, None)]
    view = cx.explanation_view("night-2", "night-1")
    assert view.status == "unavailable" and view.message.startswith("Сравнивать нечего")
    with pytest.raises(cx.ExplanationError) as error:
        cx.generate_explanation("night-2", "night-1")
    assert (error.value.code, error.value.status_code) == ("no_data", 422)
    # Shared metrics are enough without reports.
    pair["reports"] = {"A": None, "B": None}
    pair["rows"] = [("lt", "RPS", 10.0, 9.0)]
    assert cx.explanation_view("night-2", "night-3").status == "missing"


def test_without_an_llm_the_outcome_is_still_shown(pair, monkeypatch):
    monkeypatch.setitem(cx.CONFIG, "llm", {})
    view = cx.explanation_view("night-2", "night-1").to_dict()
    assert view["status"] == "missing" and view["llm_problem"] and view["outcome"]["code"] == "better"
    with pytest.raises(cx.ExplanationError) as error:
        cx.generate_explanation("night-2", "night-1")
    assert (error.value.code, error.value.status_code) == ("missing_llm", 422)
    assert pair["calls"] == []


def test_a_bad_answer_is_reported_and_not_cached(pair):
    from AI.limit_explanation import ExplanationParseError

    pair["answer"] = {"headline": "без схемы"}
    with pytest.raises(ExplanationParseError):
        cx.generate_explanation("night-2", "night-1")
    assert pair["stored"] == {}


def test_the_language_of_run_a_and_a_hash_independent_of_the_reader(pair):
    pair["reports"] = {
        "A": _report("night-2", lang="en", rps=1200.0, findings={"jvm": [_finding("f1", "GC pauses 120 ms at the peak")]}),
        "B": _report("night-1", lang="ru", created=CREATED_B, findings={"database": [_finding("f1", "Pool exhausted, wait 800 ms")]}),
    }
    pair["answer"] = {**ANSWER, "headline": "A holds more RPS", "summary": "1200 vs 1000 RPS", "causes": [], "next_checks": [],
                      "key_changes": [{"label": "Max RPS", "value_a": "1200", "value_b": "1000", "assessment": "better"}],
                      "evidence": [{"finding_id": "B:database:f1", "role": "pool"}]}
    with with_locale("ru"):
        ru_context = cx.build_context("night-2", "night-1")
    with with_locale("en"):
        en_context = cx.build_context("night-2", "night-1")
    assert ru_context.lang == "en" and ru_context.inputs_hash == en_context.inputs_hash
    expected = hashlib.sha256(json.dumps(
        {"prompt": load_prompt("compare_prompt.txt", "en"), "payload": ru_context.payload}, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    assert ru_context.inputs_hash == expected
    assert not CYRILLIC.search(json.dumps(ru_context.payload, ensure_ascii=False))
    with with_locale("ru"):
        cx.generate_explanation("night-2", "night-1")
    [call] = pair["calls"]
    assert call["prompt"] == load_prompt("compare_prompt.txt", "en") and not CYRILLIC.search(call["system_prompt"])


def test_query_roles_of_the_global_queries(monkeypatch):
    monkeypatch.setitem(cx.CONFIG, "queries", {"lt_framework": {"labels": ["RPS", "gone"], "roles": {"RPS": "lt.rps_total", "x": "lt.vus"}}, "bad": []})
    assert cx.query_roles(None) == {"lt_framework": {"RPS": "lt.rps_total"}}


# ---- storage ---------------------------------------------------------------------------------------------------


def test_load_run_report_reads_the_rows_of_the_latest_report_and_the_manual_verdict(monkeypatch):
    rows = [
        ("final", "r9", "checkout", "step", "shop", 0, 3_600_000, "Провал", None, json.dumps({"peak_performance": {"max_rps": 10}}), {}, {}, CREATED_A),
        ("jvm", "r9", "checkout", "step", "shop", 0, 3_600_000, "pass", "risk", {"findings": [_finding("f1", "x")]}, None, None, CREATED_A),
    ]
    conn = RecordingConnection(fetchall_results=[rows])
    monkeypatch.setattr(db_store, "load_verdict_state", lambda conn, schema, table, name: {"manual": {"verdict": "pass"}})
    report = db_store.load_run_report(conn, "lt", "llm_reports", "night-2")
    sql_text, params = conn.log[0]
    assert params == ("night-2", "night-2") and "DISTINCT ON (r.domain)" in sql_text and "domain <> 'engineer'" in sql_text
    assert report.final.verdict == "fail" and report.rows["jvm"].verdict == "risk" and report.verdict == "pass"
    assert [finding.ref for finding in report.findings()] == ["jvm:f1"]
    assert db_store.load_run_report(RecordingConnection(), "lt", "llm_reports", "none") is None


def test_pair_rows_follow_the_deletion_and_renaming_of_a_run(monkeypatch):
    conn = RecordingConnection()
    monkeypatch.setattr(core, "_ts_conn", lambda: conn)
    core._delete_run_data("night-1")
    deletes = [(text, params) for text, params in conn.log if "DELETE" in text and "run_a" in text]
    assert len(deletes) == 2 and all(params == ("night-1", "night-1") for _, params in deletes)
    assert "compare_explanations" in deletes[0][0] and "compare_publications" in deletes[1][0]

    cur = RecordingConnection()
    db_store.rename_pair_rows(cur.cursor(), "public", "compare_explanations", "old", "new")
    assert [params for _, params in cur.log] == [("new", "old"), ("new", "old")]
    assert "Identifier('run_a')" in cur.log[0][0] and "Identifier('run_b')" in cur.log[1][0]


# ---- API -------------------------------------------------------------------------------------------------------


def test_explanation_api(client, pair):
    assert client.get("/compare_explanation").status_code == 400
    assert client.get("/compare_explanation?run_a=x&run_b=x").get_json()["error"] == "A и B — один и тот же прогон"
    body = client.get("/compare_explanation?run_a=night-2&run_b=night-1").get_json()
    assert body["status"] == "missing" and body["run_a"] == "night-2" and body["outcome"]["code"] == "better"
    assert client.post("/compare_explanation", data="{}").status_code == 415
    assert client.post("/compare_explanation", json=[]).status_code == 400
    assert client.post("/compare_explanation", json={"run_a": "night-2"}).status_code == 400
    generated = client.post("/compare_explanation", json={"run_a": "night-2", "run_b": "night-1"})
    assert generated.status_code == 200 and generated.get_json()["status"] == "ready"
    again = client.post("/compare_explanation", json={"run_a": "night-2", "run_b": "night-1"})
    assert again.status_code == 429 and again.get_json()["code"] == "too_often"
    client.set_cookie("loadlens.lang", "en")
    english = client.get("/compare_explanation?run_a=night-2&run_b=night-1").get_json()
    assert english["outcome"]["title"] == "A is better than B" and english["explanation"]["headline"] == ANSWER["headline"]


def test_confluence_api_runs_a_job(client, monkeypatch):
    from loadlens_app.blueprints import compare as compare_api

    calls = []

    def fake_publish(run_a, run_b, source_url, progress):
        calls.append((run_a, run_b, source_url))
        return {"page_id": "8", "page_url": "https://wiki.local/pages/8", "title": "Сравнение"}

    monkeypatch.setattr(compare_api, "publish_comparison", fake_publish)
    monkeypatch.setattr(compare_api, "load_compare_publication", lambda run_a, run_b: None)
    assert client.get("/compare_confluence?run_a=a&run_b=b").get_json()["page_url"] is None
    assert client.post("/compare_confluence", data="{}").status_code == 415
    assert client.post("/compare_confluence", json={"run_a": "a", "run_b": "a"}).status_code == 400
    resp = client.post("/compare_confluence", json={"run_a": "night 2", "run_b": "night-1"})
    assert resp.status_code == 200
    job = client.get(f"/job_status/{resp.get_json()['job_id']}").get_json()
    assert (job["status"], job["kind"], job["page_url"], job["report_url"]) == (
        "done", "compare_confluence", "https://wiki.local/pages/8", "/compare?run_a=night+2&run_b=night-1",
    )
    assert calls == [("night 2", "night-1", "http://localhost/compare?run_a=night+2&run_b=night-1")]


def test_summaries_send_the_direction_of_the_role(client, monkeypatch):
    from loadlens_app.blueprints import compare as compare_api

    monkeypatch.setattr(compare_api, "query_roles", lambda area, service=None: {"database": {"Hits": "db.cache_hit_pct"}})
    monkeypatch.setattr(compare_api, "_ts_conn", lambda: RecordingConnection(fetchall_results=[[("Hits", 90.0, 99.0), ("Lag", 5.0, 1.0)]]))
    rows = client.get("/compare_summary?run_a=a&run_b=b&domain=database").get_json()
    assert [(row["direction"], round(row["trend_pct"], 1)) for row in rows] == [("higher", -9.1), ("lower", 400.0)]
    monkeypatch.setattr(compare_api, "_ts_conn", lambda: RecordingConnection(fetchall_results=[[("pod-1", 90.0, 100.0)]]))
    body = client.get("/compare_metric_summary?run_a=a&run_b=b&domain=database&query_label=Hits").get_json()
    assert body["direction"] == "higher" and body["rows"][0]["direction"] == "higher"
    assert body["rows"][0]["trend_pct"] == pytest.approx(-10.0)


def test_the_aggregate_of_the_facts_is_the_p95_of_the_page():
    from loadlens_app.blueprints import compare as compare_api

    assert cx.P95_SQL == compare_api.AGGREGATES[cx.AGGREGATE]


# ---- Confluence page -------------------------------------------------------------------------------------------

CONFLUENCE_CFG = {
    "enabled": True, "base_url": "https://wiki.local", "username": "u", "password": "p", "space_key": "LT",
    "parent_page_id": "1", "compare_parent_page_id": "",
}


def _published(monkeypatch, view):
    captured = {"progress": []}

    def upsert(client, known, title, body, attachments, progress):
        captured.update(title=title, body=body, attachments=attachments, parent=client.parent_page_id)
        return "9", "https://wiki.local/pages/9"

    monkeypatch.setattr(cc, "confluence_config", lambda: dict(CONFLUENCE_CFG))
    monkeypatch.setattr(cc, "explanation_view", lambda run_a, run_b: view)
    monkeypatch.setattr(cc, "load_compare_publication", lambda run_a, run_b: None)
    monkeypatch.setattr(cc, "save_compare_publication", lambda *args: captured.update(saved=args[1:3]))
    monkeypatch.setattr(cc, "upsert_page", upsert)
    result = cc.publish_comparison("night-2", "night-1", "http://ll/compare?run_a=night-2&run_b=night-1",
                                   lambda message, pct=None: captured["progress"].append(str(message)))
    return result, captured


def test_compare_page_with_the_explanation(pair, monkeypatch):
    view = cx.generate_explanation("night-2", "night-1")
    result, page = _published(monkeypatch, view)
    assert result == {"page_id": "9", "page_url": "https://wiki.local/pages/9", "title": "Сравнение прогонов: night-2 и night-1"}
    body = page["body"]
    assert page["attachments"] == {} and page["parent"] == "1" and page["saved"] == ("9", "https://wiki.local/pages/9")
    assert body.count('<ac:structured-macro ac:name="tip">') == 1  # the outcome «better» is a tip panel
    for text in ("A лучше B", "Максимум RPS: A — 1200, B — 1000 (+20,0 %)", "Вердикты", "Ключевые изменения метрик",
                 "ИИ-разбор различий", ANSWER["headline"], "Вероятные причины", "Разбор ИИ: test-model",
                 "Открыть сравнение в LoadLens", "Как сравнивается", "1200 RPS (максимум теста)"):
        assert text in body, text
    assert page["progress"][0] == "Сбор сравнения…" and page["progress"][-1] == "Готово"


def test_compare_page_without_an_explanation_lists_the_findings(pair, monkeypatch):
    view = cx.explanation_view("night-2", "night-1")
    _, page = _published(monkeypatch, view)
    body = page["body"]
    assert "Разбор ИИ ещё не делали" in body and "Находки прогона B (night-1)" in body
    assert "Пул соединений исчерпан" in body and "CPU 12345" not in body


def test_an_en_comparison_is_published_in_english_under_a_ru_job(pair, monkeypatch):
    pair["reports"] = {
        "A": _report("night-2", lang="en", rps=1200.0, findings={"jvm": [_finding("f1", "GC pauses 120 ms")]}),
        "B": _report("night-1", lang="en", created=CREATED_B),
    }
    with with_locale("ru"):
        view = cx.explanation_view("night-2", "night-1")
        _, page = _published(monkeypatch, view)
    assert not CYRILLIC.search(page["title"] + page["body"]), page["body"]
    assert "Run comparison: night-2 vs night-1" == page["title"] and "A is better than B" in page["body"]
    assert page["progress"][0] == "Сбор сравнения…"


def test_nothing_to_publish(pair, monkeypatch):
    pair["reports"] = {"A": None, "B": None}
    pair["rows"] = []
    view = cx.explanation_view("night-2", "night-1")
    with pytest.raises(cc.ComparePublishError):
        _published(monkeypatch, view)


def test_the_comparison_parent_page(monkeypatch):
    assert cc._compare_parent({"parent_page_id": "1", "compare_parent_page_id": " 5 "}) == "5"
    assert cc._compare_parent({"parent_page_id": "1", "compare_parent_page_id": ""}) == "1"


def test_compare_page_body_in_english_has_no_russian(pair):
    with with_locale("en"):
        body = cc.compare_page_body(cx.explanation_view("night-2", "night-1"), "", "2026-11-10 12:00")
    assert not CYRILLIC.search(body.replace("Пул соединений исчерпан, ожидание 800 мс", "").replace("GC 120 мс на пике", "")), body
