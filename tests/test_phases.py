# SPDX-License-Identifier: Apache-2.0
"""Scenario test type: phases from the request groups of the load tool, SLA per phase, the reference run."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loadlens_app  # noqa: E402,F401  (before update_page users)
from AI.db_store import find_scenario_reference  # noqa: E402
from AI.phases import (  # noqa: E402
    PhaseSettings,
    align_phases,
    build_scenario_payload,
    compare_phase_lists,
    detect_phases,
    group_path,
    label_key_of,
    phase_load_steps,
    phase_section,
    phase_settings,
    phases_ctx,
    phases_from_labeled,
    reference_from_scenario,
    series_labels,
)
from AI.report_texts import localized_step_label  # noqa: E402
from AI.sla_evaluator import evaluate_sla  # noqa: E402
from tests.test_smoke_api import RecordingConnection  # noqa: E402

START = pd.Timestamp("2026-11-10T09:00:00Z")
START_TS = START.timestamp()


def _frame(plan, extra=None, minutes=None):
    """``plan``: [(group path, minutes, rps)] in order; every group gets 1 RPS elsewhere unless ``extra`` says."""
    total = minutes or sum(m for _, m, _ in plan)
    index = pd.date_range(START, periods=total, freq="1min")
    columns = {}
    row = 0
    for path, length, rps in plan:
        name = f"group={path}|name=req"
        values = columns.setdefault(name, np.full(total, 1.0))
        values[row:row + length] = rps
        row += length
    for name, values in (extra or {}).items():
        columns[name] = np.asarray(values, dtype=float)
    return pd.DataFrame(columns, index=index)


def _detect(frame, **settings):
    end = frame.index[-1].timestamp() + 60
    return detect_phases(frame, start_ts=frame.index[0].timestamp(), end_ts=end, settings=PhaseSettings(**settings), source_label="RPS by group")


# ---- series names -------------------------------------------------------------------------------------------

def test_series_names_give_group_paths():
    assert series_labels("group=::a::b|name=x") == {"group": "::a::b", "name": "x"}
    assert series_labels("plain") == {}
    assert label_key_of(["transaction=Login", "transaction=Pay"]) == "transaction"
    assert label_key_of(["group=::a|name=x"], wanted="name") == "name"
    assert label_key_of(["status=200"]) == "status"
    assert group_path("group=::Checkout::Pay::R__x|name=u", "group") == ("Checkout", "Pay", "R__x")
    assert group_path("transaction=Login", "transaction") == ("Login",)
    # A series without the key is its own group.
    assert group_path("sum_all", "group") == ("sum_all",)
    assert group_path("group=|name=x", "group") == ("?",)


# ---- detection ---------------------------------------------------------------------------------------------

def test_three_groups_in_turn_are_three_phases():
    profile = _detect(_frame([("::Catalog", 10, 50), ("::Cart", 10, 40), ("::Pay", 10, 30)]))
    assert profile.detected and profile.reason is None and profile.level == 1 and profile.label_key == "group"
    assert [p.key for p in profile.phases] == ["Catalog", "Cart", "Pay"]
    assert [p.duration_min for p in profile.phases] == [10.0, 10.0, 10.0]
    assert profile.phases[0].start == START and profile.phases[-1].at_end and profile.phases[0].at_start
    assert profile.phases[1].groups[0]["key"] == "Cart" and profile.phases[1].share > 0.9
    assert profile.phases[0].label() == "фаза 1 · Catalog"


def test_a_wrapper_group_moves_the_level_down_and_names_the_child():
    frame = _frame([("::Scenario::Login::R__token", 6, 30), ("::Scenario::Order::R__create", 6, 30)])
    profile = _detect(frame)
    assert profile.level == 2 and [p.key for p in profile.phases] == ["Scenario::Login", "Scenario::Order"]
    assert [p.title for p in profile.phases] == ["Login · R__token", "Order · R__create"]
    assert _detect(frame, level=3).level == 3


def test_short_segments_join_the_phase_before_them_and_pauses_stay():
    # A 6 min, B 1 min (too short: joins A), A 5 min, C 6 min, 4 idle minutes, D 5 min.
    frame = _frame([("::A", 6, 50), ("::B", 1, 60), ("::A", 5, 50), ("::C", 6, 40), ("::idle", 4, 0), ("::D", 5, 30)])
    frame = frame.drop(columns=["group=::idle|name=req"])
    frame.iloc[18:22] = 0.0
    profile = _detect(frame, min_minutes=2)
    assert [(p.key, p.idle, p.duration_min) for p in profile.phases] == [
        ("A", False, 12.0), ("C", False, 6.0), ("", True, 4.0), ("D", False, 5.0),
    ]
    assert profile.phases[0].absorbed == 1 and profile.phases[2].label() == "пауза 1" and profile.phases[3].label() == "фаза 3 · D"


def test_the_phase_count_is_capped():
    plan = [(f"::G{i}", 3, 50 + i) for i in range(10)]
    profile = _detect(_frame(plan), max_count=4)
    assert len(profile.phases) == 4 and profile.detected


@pytest.mark.parametrize("frame, reason", [
    (_frame([("::Only", 10, 50)]), "one_group"),
    (pd.DataFrame(), "no_data"),
])
def test_without_two_leading_groups_there_are_no_phases(frame, reason):
    profile = _detect(frame) if not frame.empty else detect_phases(frame, start_ts=START_TS, end_ts=START_TS + 600, settings=PhaseSettings())
    assert not profile.detected and profile.reason == reason
    if not frame.empty:
        assert len(profile.phases) == 1 and profile.phases[0].duration_min == 10.0


def test_phases_become_contiguous_load_steps_and_localized_labels():
    profile = _detect(_frame([("::Catalog", 10, 50), ("::Cart", 10, 40)]))
    steps = phase_load_steps(profile, START_TS, 3)
    assert [s.source for s in steps] == ["phase_detector"] * 2 and steps[0].end == steps[1].start
    assert steps[0].plateau_start == steps[0].start and steps[1].minute_to == 20.0
    assert steps[0].rps_level == pytest.approx(51.0)  # the other group keeps 1 RPS
    assert localized_step_label("фаза 2 · Cart · R__x", "en") == "phase 2 · Cart · R__x"
    assert localized_step_label("pause 3", "ru") == "пауза 3"


def test_settings_read_the_sla_block_with_defaults():
    assert phase_settings({}) == PhaseSettings()
    settings = phase_settings({"phase_query": " RPS by group ", "phase_level": "2", "phase_min_minutes": "x", "phase_max_count": 1})
    assert settings.query == "RPS by group" and settings.level == 2 and settings.min_minutes == 2.0 and settings.max_count == 2


def test_the_section_is_the_role_query_or_the_named_one():
    groups = _frame([("::A", 5, 10), ("::B", 5, 10)])
    labeled = [
        {"label": "RPS total", "df": groups.sum(axis=1).to_frame("all")},
        {"label": "RPS by group", "df": groups},
        {"label": "p95 by group", "df": groups},
    ]
    block = {"labels": ["RPS total", "RPS by group", "p95 by group"], "roles": {"RPS by group": "lt.rps_by_request"}}
    assert phase_section(labeled, block, PhaseSettings())["label"] == "RPS by group"
    assert phase_section(labeled, block, PhaseSettings(query="p95 by GROUP"))["label"] == "p95 by group"
    assert phase_section(labeled, {"labels": ["RPS by group"]}, PhaseSettings()) is None


def test_a_role_query_without_rows_means_no_data_not_no_query():
    # The window had no load (wrong date): the query of the role exists but returns nothing.
    block = {"labels": ["RPS by group"], "roles": {"RPS by group": "lt.rps_by_request"}}
    end = START_TS + 600
    empty = [{"label": "RPS by group", "df": pd.DataFrame()}]
    profile = phases_from_labeled(empty, block, {}, START_TS, end)
    assert profile.reason == "no_data" and profile.source_label == "RPS by group"
    # The query failed and is not among the labeled sections at all.
    assert phases_from_labeled([], block, {}, START_TS, end).reason == "no_data"
    assert phases_from_labeled(empty, {"labels": ["RPS by group"]}, {}, START_TS, end).reason == "no_group_query"


# ---- two runs ------------------------------------------------------------------------------------------------

def _phase(index, key, minutes, groups=None, **extra):
    return {"index": index, "number": index, "key": key, "title": key, "label": f"фаза {index} · {key}",
            "duration_min": minutes, "groups": [{"key": g} for g in (groups or [key])], "idle": False, **extra}


def test_phases_align_by_key_then_by_similar_groups():
    reference = [_phase(1, "Delete", 2, ["Delete", "CatalogReset"]), _phase(2, "Indicators", 2), _phase(3, "Results", 3)]
    current = [_phase(1, "CatalogReset", 2, ["CatalogReset", "Delete"]), _phase(2, "Indicators", 3), _phase(3, "Extra", 4)]
    pairs = [(r and r["key"], c and c["key"]) for r, c in align_phases(reference, current)]
    assert pairs == [("Delete", "CatalogReset"), ("Indicators", "Indicators"), ("Results", None), (None, "Extra")]


def test_durations_skip_the_edges_and_changes_within_the_data_step():
    reference = [_phase(1, "A", 10, at_start=True), _phase(2, "B", 3), _phase(3, "C", 10), _phase(4, "D", 8, at_end=True)]
    current = [_phase(1, "A", 20, at_start=True), _phase(2, "B", 5), _phase(3, "C", 15), _phase(4, "D", 2, at_end=True)]
    out = compare_phase_lists(reference, current, min_minutes=2, interval_sec=60)
    assert [d["key"] for d in out["durations"]] == ["B", "C"]
    assert [d["significant"] for d in out["durations"]] == [False, True]
    assert out["worst_duration"]["key"] == "C" and out["worst_duration"]["change_pct"] == 50.0
    assert out["missing"] == [] and out["extra"] == []


# ---- SLA ----------------------------------------------------------------------------------------------------

def _sla_inputs(p95_per_phase, reference=None):
    profile = _detect(_frame([("::Catalog", 10, 50), ("::Cart", 10, 40), ("::Pay", 10, 30)])).to_dict(3)
    index = pd.date_range(START, periods=30, freq="1min")
    p95 = np.repeat(p95_per_phase, 10).astype(float)
    lt_pack = {"phase_profile": profile}
    if reference:
        lt_pack["phase_reference"] = reference
    domain_data = {"lt_framework": {"pack": lt_pack, "labeled": [{"label": "p95", "df": pd.DataFrame({"p95_all": p95}, index=index)}]}}
    sla = {"target_rps": 500, "max_p95_ms": 300, "p95_query": "p95"}
    return domain_data, sla, profile


def test_sla_takes_the_worst_phase_and_ignores_the_target_rps():
    domain_data, sla, _ = _sla_inputs([100, 400, 200])
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "phases"}, lang="ru")
    assert result["test_mode"] == "phases" and result["verdict"] == "fail"
    assert [c["name"] for c in result["checks"]] == ["p95_latency"]
    assert "фазы «фаза 2 · Cart»" in result["checks"][0]["message"]
    assert result["stable_window"]["kind"] == "phase" and result["stable_window"]["title"] == "фаза 2 · Cart"
    assert [(r["key"], r["verdict"], r["worst"]) for r in result["phase_results"]] == [
        ("Catalog", "pass", False), ("Cart", "fail", True), ("Pay", "pass", False),
    ]
    assert "Режим сценария" in result["summary"]
    passed = evaluate_sla(_sla_inputs([100, 100, 100])[0], sla, test_profile={"mode": "phases"}, lang="en")
    assert passed["verdict"] == "pass" and passed["stable_window"] is None and "Scenario mode" in passed["summary"]


def test_missing_phases_and_longer_phases_are_risks():
    reference = {"run_name": "release-1", "phases": [
        _phase(1, "Catalog", 10, at_start=True), _phase(2, "Cart", 4), _phase(3, "Pay", 10), _phase(4, "Results", 3, at_end=True),
    ]}
    domain_data, sla, _ = _sla_inputs([100, 100, 100], reference)
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "phases"}, lang="ru")
    checks = {c["name"]: c for c in result["checks"]}
    assert result["verdict"] == "risk"
    assert checks["phase_completeness"]["passed"] is False and checks["phase_completeness"]["category"] == "secondary"
    assert "release-1" in checks["phase_completeness"]["message"] and "фаза 4 · Results" in checks["phase_completeness"]["message"]
    assert checks["phase_duration"]["passed"] is False and checks["phase_duration"]["actual"] == 150.0
    assert "фаза 2 · Cart" in checks["phase_duration"]["message"] and "4 → 10" in checks["phase_duration"]["message"]


def test_without_phases_the_thresholds_cover_the_whole_window():
    domain_data, sla, _ = _sla_inputs([100, 100, 400])
    domain_data["lt_framework"]["pack"] = {"phase_profile": {"detected": False, "reason": "no_group_query", "phases": []}}
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "phases"}, lang="ru")
    assert result["verdict"] == "fail" and result["stable_window"] is None and "phase_results" in result
    assert "фазы не найдены" in result["summary"]


def _with_p95(domain_data, p95):
    frame = domain_data["lt_framework"]["labeled"][0]["df"]
    domain_data["lt_framework"]["labeled"][0]["df"] = pd.DataFrame({"p95_all": np.asarray(p95, dtype=float)}, index=frame.index)
    return domain_data


def test_one_minute_over_the_threshold_is_a_risk_and_two_minutes_fail():
    domain_data, sla, _ = _sla_inputs([100, 100, 100])
    p95 = [100.0] * 30
    p95[14] = 900.0  # Cart: one minute (its p95 is 540 ms, about that minute)
    result = evaluate_sla(_with_p95(domain_data, p95), sla, test_profile={"mode": "phases"}, lang="ru")
    check = result["checks"][0]
    assert result["verdict"] == "risk" and check["passed"] is False and check["brief"] is True and check["breach_min"] == 1.0
    assert "выше порога 1 мин, это меньше 2 мин: кратковременный всплеск" in check["message"]
    assert [(r["key"], r["verdict"]) for r in result["phase_results"]] == [("Catalog", "pass"), ("Cart", "risk"), ("Pay", "pass")]
    assert result["stable_window"]["title"] == "фаза 2 · Cart"
    # Two minutes over the threshold fail the phase and outrank the higher one-minute excursion.
    p95[24] = p95[25] = 400.0
    result = evaluate_sla(_with_p95(domain_data, p95), sla, test_profile={"mode": "phases"}, lang="ru")
    check = result["checks"][0]
    assert result["verdict"] == "fail" and not check.get("brief") and check["actual"] == 400.0 and check["breach_min"] == 2.0
    assert [r["verdict"] for r in result["phase_results"]] == ["pass", "risk", "fail"]
    assert result["stable_window"]["title"] == "фаза 3 · Pay"
    # The limit is a setting.
    result = evaluate_sla(_with_p95(domain_data, p95), {**sla, "phase_breach_minutes": 3}, test_profile={"mode": "phases"}, lang="en")
    assert result["verdict"] == "risk" and result["checks"][0]["brief"] is True and "a short excursion" in result["checks"][0]["message"]


@pytest.mark.parametrize("latency", [100, 900])
def test_all_excluded_phases_do_not_fall_back_to_whole_run_sla(latency):
    domain_data, sla, profile = _sla_inputs([latency, latency, latency])
    for phase in profile["phases"]:
        phase["excluded"] = True
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "phases"}, lang="en")
    assert result["verdict"] == "no_data"
    assert result["checks"] == []
    assert result.get("stable_window") is None


def test_excluded_phases_are_shown_but_not_judged():
    groups = _frame([("::Delete::R__CatalogReset", 10, 50), ("::Cart::R__add", 10, 40), ("::Pay::R__pay", 10, 30)])
    block = {"labels": ["RPS by group"], "roles": {"RPS by group": "lt.rps_by_request"}}
    labeled = [{"label": "RPS by group", "df": groups}]
    end = START_TS + 1800
    profile = phases_from_labeled(labeled, block, {"phase_exclude": "catalogreset"}, START_TS, end)
    assert [p.excluded for p in profile.phases] == [True, False, False]
    # Only the leading group and its child name a phase: authentication running beside every stage does not.
    auth = phase_settings({"phase_exclude": "authentication"})
    beside = _phase(2, "Cart", 10, ["Cart", "Authentication(AllRequests)"], title="Cart · R__add")
    assert not auth.excludes(beside) and auth.excludes(_phase(3, "Authentication(AllRequests)", 3))
    assert not auth.excludes({"idle": True, "key": "", "title": ""})
    # Not a valid expression: matched literally, no error.
    assert [p.excluded for p in phases_from_labeled(labeled, block, {"phase_exclude": "("}, START_TS, end).phases] == [False] * 3

    sla = {"target_rps": 500, "max_p95_ms": 300, "p95_query": "p95", "phase_exclude": "delete|cleanup"}
    profile_dict = profile.to_dict(3)
    profile_dict["phases"][0]["excluded"] = True
    reference = {"run_name": "release-1", "phases": [
        _phase(1, "Cleanup", 3, at_start=True), _phase(2, "Delete", 7), _phase(3, "Cart", 10), _phase(4, "Pay", 10, at_end=True),
    ]}
    p95 = np.repeat([900.0, 100.0, 100.0], 10)
    domain_data = {"lt_framework": {"pack": {"phase_profile": profile_dict, "phase_reference": reference},
                                    "labeled": [{"label": "p95", "df": pd.DataFrame({"p95_all": p95}, index=groups.index)}]}}
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "phases"}, lang="ru")
    # The 900 ms of the cleanup phase break nothing; the reference phases left out by the setting are not missing.
    assert result["verdict"] == "pass" and [r["key"] for r in result["phase_results"]] == ["Cart", "Pay"]
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["p95_latency"]["passed"] is True and checks["phase_completeness"]["passed"] is True
    assert "Пройдены все 2 фаз" in checks["phase_completeness"]["message"]
    payload = build_scenario_payload(profile_dict, reference, result["phase_results"], settings=phase_settings(sla))
    first = payload["phases"][0]
    assert first["excluded"] is True and "verdict" not in first and payload["worst_phase"] is None
    assert payload["reference"]["missing"] == [] and phases_ctx(payload)[0]["excluded"] is True

    from AI.report_sections import scenario_rows

    rows = dict(scenario_rows({"scenario": payload}))
    assert rows["фаза 1 · Delete · R__CatalogReset"].endswith("· без оценки") and rows["фаза 2 · Cart · R__add"].endswith("· Успешно")


# ---- payload and reference -------------------------------------------------------------------------------------

def test_scenario_payload_carries_verdicts_reference_and_mean_rps():
    domain_data, sla, profile = _sla_inputs([100, 400, 100])
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "phases"}, lang="ru")
    reference = reference_from_scenario("release-1", {"detected": True, "interval_sec": 60, "phases": [
        _phase(1, "Catalog", 10, at_start=True), _phase(2, "Cart", 10), _phase(3, "Pay", 10, at_end=True), _phase(4, "Results", 2),
    ]})
    payload = build_scenario_payload(profile, reference, result["phase_results"], focus="f", settings=phase_settings(sla))
    assert payload["detected"] and payload["phase_count"] == 3 and payload["focus"] == "f"
    assert payload["rps_mean"] == pytest.approx(42.0)  # 52, 42 and 32 RPS for 10 minutes each
    assert payload["worst_phase"]["key"] == "Cart" and payload["worst_phase"]["failed"] == ["p95_latency"]
    assert [p["verdict"] for p in payload["phases"]] == ["pass", "fail", "pass"]
    assert payload["reference"]["available"] and payload["reference"]["run_name"] == "release-1"
    assert [m["key"] for m in payload["reference"]["missing"]] == ["Results"]
    ctx = phases_ctx(payload)
    assert ctx[1]["verdict"] == "fail" and ctx[1]["groups"][0]["key"] == "Cart" and "start_utc" not in ctx[1]
    assert reference_from_scenario("x", {"detected": False, "phases": [_phase(1, "A", 2)]}) is None


@pytest.mark.parametrize("service,expected", [("orders", ["orders"]), ("", ["orders", "billing"])])
def test_phase_reference_stays_in_the_selected_service(monkeypatch, service, expected):
    from AI import pipeline as pipeline_module

    seen = {}
    conn = RecordingConnection()
    monkeypatch.setattr(pipeline_module, "_db_connect", lambda cfg: conn)

    def find(*args, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(pipeline_module, "find_scenario_reference", find)
    result = pipeline_module._load_phase_reference(
        {"schema": "public"},
        {"service": service, "run_name": "current", "area_services": ["orders", "billing"]},
        True, START_TS,
    )
    assert result is None
    assert seen["services"] == expected


def test_reference_is_the_latest_earlier_scenario_run_by_its_window():
    conn = RecordingConnection(fetchone_results=[("release-1", json.dumps({"detected": True, "phases": []}))])
    found = find_scenario_reference(
        conn, "public", "llm_reports", exclude_run_name="release-2", services=["demo", "demo"], before_ms=1_000, project_area="Демо",
    )
    assert found == ("release-1", {"detected": True, "phases": []})
    sql_text, params = conn.log[0]
    assert "parsed->'scenario'" in sql_text and "ORDER BY start_ms DESC" in sql_text and "start_ms < %s" in sql_text
    assert params == (["demo", "demo"], "release-2", 1000, "Демо")
    assert find_scenario_reference(RecordingConnection(), "public", "llm_reports", exclude_run_name="x", services=[], before_ms=1) is None


# ---- synthetic checkout scenario -----------------------------------------------------------------------------


def _checkout_frame(keys, discount_group):
    plan = [("::" + key, 4, 50) for key in keys]
    frame = _frame(plan)
    # Both groups run in the discount phase. Their lead switches between releases, so matching needs the
    # shared group set as well as the phase key. All data are generated here; no private corpus is required.
    companion = "Discount" if discount_group == "ApplyCoupon" else "ApplyCoupon"
    activity = np.ones(len(frame))
    first = keys.index(discount_group) * 4
    activity[first:first + 4] = 40
    frame[f"group=::{companion}|name=req"] = activity
    return frame


def test_two_releases_of_a_synthetic_checkout_scenario_align():
    keys = ["Browse", "SignIn", "LoadCart", "UpdateCart", "ApplyCoupon", "Address", "Delivery", "Payment",
            "Confirm", "Inventory", "Receipt", "SendReceipt", "Finish"]
    newer_keys = ["Discount" if key == "ApplyCoupon" else key for key in keys if key != "SendReceipt"]
    earlier = _detect(_checkout_frame(keys, "ApplyCoupon")).to_dict(3)
    later = _detect(_checkout_frame(newer_keys, "Discount")).to_dict(3)
    assert earlier["detected"] and later["detected"] and earlier["level"] == later["level"] == 1
    assert len(earlier["phases"]) == 13 and len(later["phases"]) == 12
    pairs = align_phases(earlier["phases"], later["phases"])
    matched = [(r, c) for r, c in pairs if r and c]
    assert len(matched) == 12 and sum(1 for r, c in matched if r["key"] == c["key"]) == 11
    out = compare_phase_lists(earlier["phases"], later["phases"], interval_sec=60)
    assert [m["key"] for m in out["missing"]] == ["SendReceipt"] and out["extra"] == []
    # Removing a phase shifts later boundaries without changing the durations of the remaining phases.
    assert out["worst_duration"] is None


# ---- report page and Confluence sections ----------------------------------------------------------------------

def _scenario_final_row(monkeypatch, p95_limit):
    """The final row of a scenario report built by the pipeline (tests/test_test_types.py frames, no model)."""
    from AI import pipeline as pipeline_module
    from tests.test_test_types import END_TS, P95_LABEL, RPS_LABEL, START_TS, _pipeline_frames, pipeline_queries

    monkeypatch.setattr(pipeline_module, "load_prompt", lambda name, lang=None: "prompt")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *a, **k: _pipeline_frames("scenario"))
    payload = {"verdict": "pass", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}
    monkeypatch.setattr(pipeline_module, "llm_two_pass_self_consistency", lambda *a, **k: (json.dumps(payload), dict(payload), {}))
    cfg = {
        "llm": {"include_markdown_tables_in_context": False, "self_consistency_k": 1, "max_domain_workers": 1},
        "default_params": {"step": "1m", "resample_interval": "1T"},
        "data_sources": {"prom": {"title": "P", "type": "prometheus", "prometheus": {"url": "http://p:9090"}}},
        "domain_sources": {"default": {"source": "prom"}, "lt_framework": {"source": "prom"}},
        "queries": {"lt_framework": pipeline_queries("scenario")},
        "sla": {"max_performance_query": RPS_LABEL, "p95_query": P95_LABEL, "max_p95_ms": p95_limit},
    }
    result = pipeline_module.uploadFromLLM(
        start_ts=START_TS, end_ts=END_TS, save_to_db=False, ef_config=cfg,
        run_meta={"test_type": "scenario", "run_name": "r", "service": "demo"}, active_domains=["lt_framework"],
    )
    final = json.loads(json.dumps(result["final_parsed"], ensure_ascii=False))
    sla = final["deterministic_sla"]
    return {
        "domain": "final", "test_type": "scenario", "parsed": final,
        "sla_details": {"checks": sla["checks"], "stable_window": sla["stable_window"], "load_step_table": final["load_step_table"]},
    }


def test_the_report_page_shows_the_phases(monkeypatch):
    from tests.js_harness import run_js

    row = _scenario_final_row(monkeypatch, p95_limit=110)
    table = row["sla_details"]["load_step_table"]
    assert [s["label"] for s in table["steps"]] == ["фаза 1 · Catalog", "фаза 2 · Cart", "фаза 3 · Pay"]
    assert all(s["phase"] and s["phase_verdict"] == "fail" for s in table["steps"]) and table["detection"]["phases"] is True
    files = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
    i = "LL.report._internals"
    out = run_js(files, f"""(() => {{
      const row = {json.dumps(row, ensure_ascii=False)};
      return {{
        section: {i}.testTypeSection(row.parsed),
        kpis: {i}.heroKpis(row).map((k) => [k.key, k.value, k.tone]),
        entries: {i}.heroTypeEntries(row.parsed, {{}}),
        scope: {i}.slaScopeText(row),
        steps: {i}.renderLoadStepTableHtml(Object.assign({{}}, row.sla_details.load_step_table, {{ steps: {i}.linkSlaToSteps(row.sla_details.load_step_table, row).steps }})),
        sentence: {i}.verdictSummaryHtml(row),
        meta: {i}.reportMetaHtml(row, [row]),
        bands: LL.charts.stepProfileModel({{
          steps: {i}.linkSlaToSteps(row.sla_details.load_step_table, row).steps,
          markers: {i}.reportMarkers(row, row.sla_details.load_step_table), width: 1000,
        }}).bands.map((b) => [b.role, b.label]),
      }};
    }})()""")
    rows = {r["label"]: r["value"] for r in out["section"]["rows"]}
    assert out["section"]["title"] == "Сценарий по фазам"
    assert rows["Тип теста"] == "Сценарий" and rows["Фазы"] == "3 (ключ «group», уровень 2)"
    assert rows["Худшая фаза"].startswith("фаза 3 · Pay") and rows["Прошлый сценарный прогон"].startswith("не найден")
    assert rows["фаза 1 · Catalog"].endswith("· Провал")
    assert out["kpis"] == [["phases", "3", ""], ["worst_phase", "фаза 3 · Pay", "fail"], ["latency", "121.47", "fail"]]
    assert out["scope"] == "каждой фазе сценария, худшая — фаза 3 · Pay"
    assert 'class="rep-steprow is-phases"' in out["steps"] and 'rep-step-name">фаза 2 · Cart<' in out["steps"]
    assert out["steps"].count("rep-step is-sla is-bad") == 1 and ">худшая<" in out["steps"]
    # Phase cells keep the number when narrow; the RPS is a part the stylesheet hides then.
    assert '<b>2<span class="rep-step-rps"> · ' in out["steps"]
    # The worst phase is the SLA zone of the profile: one band, named after the phase, not two labels on each other.
    assert out["bands"] == [["sla", "худшая фаза: фаза 3 · Pay"]]
    assert "Фазы найдены по запросу «LT RPS by group»: ключ «group», уровень вложенности 2" in out["steps"]
    assert "p95" in out["sentence"] and "Сценарий" in out["meta"]

    from AI.i18n import with_locale
    from AI.report_sections import test_type_section

    with with_locale("en"):
        title, py_rows = test_type_section(row["parsed"])
    labels = dict(py_rows)
    assert title == "Scenario by phase" and labels["Phases"] == "3 (key «group», level 2)" and labels["Test type"] == "Scenario"
    # Phase labels are stored in the language of the report; the values follow the reader.
    assert labels["фаза 2 · Cart"].endswith("· Failed")


def test_the_page_marks_short_excursions_and_excluded_phases():
    from tests.js_harness import run_js

    check = {"name": "p95_latency", "passed": False, "brief": True, "breach_min": 1, "actual": 540, "threshold": 300,
             "category": "primary", "query_label": "p95", "series": "p95_all", "message": "p95 540 > 300"}
    row = {"domain": "final", "parsed": {}, "sla_details": {"checks": [check], "stable_window": None}}
    step = {"index": 1, "phase": True, "excluded": True, "label": "фаза 1 · Delete", "rps": 50,
            "start_iso": "2026-11-10T12:00:00+03:00", "end_iso": "2026-11-10T12:10:00+03:00"}
    i = "LL.report._internals"
    out = run_js(["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"], f"""({{
      sla: {i}.renderSlaChecklist({json.dumps(row, ensure_ascii=False)}),
      steps: {i}.renderLoadStepTableHtml({{ steps: [{json.dumps(step, ensure_ascii=False)}] }}),
      rows: {i}.testTypeSection({{ test_profile: {{ mode: 'phases' }}, scenario: {{ detected: true, phases: [
        {{ label: 'фаза 1 · Delete', excluded: true, duration_min: 10, rps_mean: 50 }}] }} }}).rows.map((r) => r.value),
    }})""")
    assert '<tr class="is-risk">' in out["sla"] and 'rep-glyph risk' in out["sla"] and ">!</span>" in out["sla"]
    assert "выше порога 1 мин — риск" in out["sla"]
    assert "rep-step is-excluded" in out["steps"] and 'title="фаза 1 · Delete · без оценки"' in out["steps"]
    assert out["rows"][-1].endswith("· без оценки")

    from loadlens_app.confluence_export import sla_criteria_block

    block = sla_criteria_block(row)
    assert "Yellow" in block and "Всплеск" in block and "Red" not in block
    # The compare page and its Confluence page name a phase that was not judged.
    pairs = {"available": True, "metrics": [], "pairs": [{"a": {"label": "фаза 1 · Delete", "excluded": True}, "b": None}]}
    html = run_js(["static/js/common.js", "static/js/ui.js", "static/js/compare.js"], f"LL.compare._internals.phasesTableHtml({json.dumps(pairs, ensure_ascii=False)})")
    assert "фаза 1 · Delete" in html and "без оценки" in html


def test_profile_band_labels_do_not_run_into_each_other():
    from tests.js_harness import run_js

    steps = [
        {"index": i + 1, "start_iso": f"2026-11-10T12:{m:02d}:00+03:00", "end_iso": f"2026-11-10T12:{m + 10:02d}:00+03:00", "rps": 10 * (i + 1)}
        for i, m in enumerate((0, 10, 20))
    ]
    t = lambda m: f"2026-11-10T09:{m:02d}:00Z"  # noqa: E731
    markers = [
        {"kind": "band", "t": t(0), "to": t(10), "label": "первая полоса с очень длинной подписью", "tone": "warn", "role": "spike"},
        {"kind": "band", "t": t(12), "to": t(20), "label": "вторая", "tone": "warn", "role": "spike"},
    ]
    script = f"""LL.charts.stepProfileModel({{ steps: {json.dumps(steps)}, markers: {json.dumps(markers, ensure_ascii=False)},
      width: 300 }}).bands.map((b) => [b.x, b.labelX, b.label])"""
    bands = run_js(["static/js/common.js", "static/js/ui.js", "static/js/charts.js"], script)
    assert [b[2] for b in bands] == ["первая полоса с…", "вторая"]
    assert bands[0][1] + len(bands[0][2]) * 6.7 <= bands[1][1]


# ---- what the model may cite -----------------------------------------------------------------------------------

def test_phase_numbers_are_verified_and_phase_numbers_are_references():
    from AI import scoring
    from AI.scoring import parse_llm_analysis_strict
    from AI.verification import build_reference_catalog, verify_analysis

    ctx = {
        "phases": [{"label": "фаза 3 · Pay", "duration_min": 17.0, "rps_mean": 39.1, "groups": [{"key": "Pay", "share": 0.734}]}],
        "scenario": {"rps_mean": 42.0, "phases": [], "reference": {"durations": [
            {"label": "фаза 3 · Pay", "reference_min": 11.0, "current_min": 17.0, "change_pct": 54.5}]}},
    }
    values = {ref["value"] for ref in build_reference_catalog(ctx)}
    assert {17.0, 39.1, 73.4, 42.0, 11.0, 54.5} <= values
    analysis = parse_llm_analysis_strict(json.dumps({"verdict": "risk", "findings": [{
        "id": "f1", "summary": "Фаза 3 · Pay длилась 17 мин против 11 мин в прошлом прогоне (+54.5 %) при 39.1 RPS",
        "component": "lt",
    }]}, ensure_ascii=False))
    verified, _ = verify_analysis(analysis, ctx)
    # «фаза 3» is a reference to the phase, not a number to check.
    assert verified.findings[0].verification.status == "verified"
    assert verified.findings[0].verification.claims_total == 4
    assert {"фаз", "phase"} <= set(scoring._REF_STEMS["phase"])


def test_confluence_renders_the_scenario_section_and_the_worst_phase(monkeypatch):
    from loadlens_app.confluence_export import render_llm_domain_body, sla_window_note
    from confluence_manager.update_confluence_template import render_llm_markdown

    row = _scenario_final_row(monkeypatch, p95_limit=110)
    body = render_llm_domain_body({"domain": "final", "parsed": row["parsed"]})
    assert "Сценарий по фазам" in body and "фаза 2 · Cart" in body and "Пиковая производительность" not in body
    markdown = render_llm_markdown({**row["parsed"], "findings": [], "recommended_actions": []})
    assert "#### Сценарий по фазам" in markdown and "| Худшая фаза | фаза 3 · Pay" in markdown
    window = row["sla_details"]["stable_window"]
    assert sla_window_note(window, row["sla_details"]["load_step_table"]) == "SLA проверено в каждой фазе, худшая — фаза 3 · Pay."
