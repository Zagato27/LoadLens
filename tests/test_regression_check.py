# SPDX-License-Identifier: Apache-2.0
"""Regression against the baseline run: comparable points, deltas, checks, the verdict, validation and the pipeline."""

import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_smoke_api import RecordingConnection  # noqa: E402,F401  (initialises the app before update_page)
from AI import pipeline as pipeline_module  # noqa: E402
from AI import regression as rg  # noqa: E402
from AI import sla_evaluator as se  # noqa: E402
from AI.context_pack import LoadStep, load_step_report_from_frames  # noqa: E402
from AI.i18n import with_locale  # noqa: E402
from AI.pipeline import _deterministic_sla_context, _load_baseline_sla, _reconcile_sla_for_test_profile  # noqa: E402
from AI.prompt_assembly import sla_guidance_lines  # noqa: E402
from loadlens_app import core  # noqa: E402
from loadlens_app.config_schema import SlaConfigError, validate_sla  # noqa: E402
import update_page  # noqa: E402

CYRILLIC = re.compile("[А-Яа-яЁё]")
THRESHOLDS = {
    "regression_latency_risk_pct": 10.0,
    "regression_latency_fail_pct": None,
    "regression_errors_risk_pp": 0.5,
    "regression_errors_fail_pp": None,
    "regression_capacity_risk_pct": 10.0,
    "regression_capacity_fail_pct": None,
}


def _row(index, rps, p95, errors=None, **flags):
    row = {"index": index, "label": f"Step {index}", "rps": rps, "latency_p95": p95, "stable": True, **flags}
    if errors is not None:
        row["error_rate"] = errors
    return row


def _step_details(*, ceiling=800.0, target=600.0, rows=None, p95=300.0, errors=0.5, label="LT p95"):
    """sla_details of a step test: checks on the ceiling, the target step, the headroom and the step table."""
    checks = [
        {"name": "target_rps", "threshold": 500, "actual": ceiling, "passed": True, "severity": "critical"},
        {"name": "error_rate", "threshold": 1.0, "actual": errors, "passed": True, "category": "primary"},
        {"name": "p95_latency", "threshold": 500, "actual": p95, "passed": True, "category": "primary", "query_label": label},
    ]
    details = {"checks": checks, "summary": "", "stable_window": {"level": ceiling, "start": "s", "end": "e"}}
    if target is not None:
        details["target_window"] = {"level": target, "role": "target"}
        details["headroom"] = {"ceiling_level": ceiling, "verdict_level": target, "target_rps": 500}
    if rows is not None:
        details["load_step_table"] = {"latency_query": label, "steps": rows}
    return details


def _hold_details(level, p95=200.0, errors=0.3, kind="hold"):
    return {
        "checks": [
            {"name": "target_rps", "threshold": 250, "actual": level, "passed": True},
            {"name": "error_rate", "threshold": 1.0, "actual": errors, "passed": True},
            {"name": "p95_latency", "threshold": 500, "actual": p95, "passed": True},
        ],
        "stable_window": {"level": level, "kind": kind},
    }


def _items(comparison):
    return {item["metric"]: item for item in comparison["items"]}


# ---- comparable point ----------------------------------------------------------------------------------------------


def test_target_step_of_both_runs_is_the_comparison_level():
    current = _step_details(rows=[_row(1, 300, 100, 0.1), _row(2, 600, 150, 0.2, selected=True, sla_role="target"), _row(3, 800, 300, 0.5, ceiling=True)])
    baseline = _step_details(ceiling=900, rows=[_row(1, 300, 90, 0.1), _row(2, 601, 120, 0.1, selected=True), _row(3, 900, 280, 0.4)])
    result = rg.compare_stored(current, baseline, "step", THRESHOLDS)
    assert result["available"] and result["kind"] == "target" and result["level"] == 600
    assert result["step_label"] == "Step 2"
    items = _items(result)
    assert items["p95"]["current"] == 150 and items["p95"]["baseline"] == 120
    assert items["p95"]["change"] == 25.0 and items["p95"]["delta"] == 25.0 and items["p95"]["stat"] == "p95"
    assert items["p95"]["passed"] is False and items["p95"]["severity"] == "warning" and items["p95"]["tone"] == "warn"
    assert items["errors"]["change"] == 0.1 and items["errors"]["unit"] == "pp" and items["errors"]["passed"] is True
    # The load reached is the ceiling: a drop is a positive delta.
    assert items["capacity"]["current"] == 800 and items["capacity"]["baseline"] == 900
    assert items["capacity"]["change"] == -11.1 and items["capacity"]["delta"] == 11.1 and items["capacity"]["passed"] is False


def test_target_of_one_run_only_compares_at_the_lower_ceiling():
    current = _step_details(ceiling=800, rows=[_row(1, 400, 100, 0.1), _row(2, 700, 200, 0.3), _row(3, 800, 300, 0.5, ceiling=True)])
    baseline = _step_details(ceiling=700, target=None, rows=[_row(1, 350, 90, 0.1), _row(2, 700, 160, 0.2, selected=True)])
    result = rg.compare_stored(current, baseline, "step", THRESHOLDS)
    assert result["kind"] == "ceiling" and result["level"] == 700
    items = _items(result)
    assert (items["p95"]["current"], items["p95"]["baseline"]) == (200, 160)
    assert items["capacity"]["change"] == pytest.approx(14.3) and items["capacity"]["delta"] == pytest.approx(-14.3)
    assert items["capacity"]["passed"] is True and items["capacity"]["tone"] == "ok"


def test_a_step_row_counts_within_two_percent_of_the_level():
    assert rg.STEP_TOLERANCE_PCT == se.TARGET_STEP_TOLERANCE_PCT
    details = _step_details(ceiling=900, target=None, rows=[_row(1, 612, 140, 0.2), _row(2, 900, 300, 0.5)])
    point = rg.comparison_point(details, "step", 600)
    assert point["p95"] == 140 and point["errors"] == 0.2 and point["sources"]["p95"] == "steps"
    far = _step_details(ceiling=900, target=None, rows=[_row(1, 613, 140, 0.2), _row(2, 900, 300, 0.5)])
    point = rg.comparison_point(far, "step", 600)
    assert point["p95"] is None and point["reasons"] == {"p95": "no_step_at_level", "errors": "no_step_at_level"}


def test_without_a_step_table_the_checks_count_when_the_ceilings_are_within_five_percent():
    current = _step_details(ceiling=800, target=None, p95=300, errors=0.5)
    near = _step_details(ceiling=820, target=None, p95=250, errors=0.4)
    result = rg.compare_stored(current, near, "step", THRESHOLDS)
    items = _items(result)
    assert result["kind"] == "ceiling" and result["level"] == 800
    assert items["p95"]["change"] == 20.0 and items["errors"]["change"] == 0.1
    far = _step_details(ceiling=900, target=None, p95=250, errors=0.4)
    result = rg.compare_stored(current, far, "step", THRESHOLDS)
    assert set(_items(result)) == {"capacity"}
    assert result["skipped"] == [{"metric": "p95", "reason": "level_mismatch"}, {"metric": "errors", "reason": "level_mismatch"}]


def test_old_rows_without_error_rate_take_the_checks_of_both_runs():
    current = _step_details(ceiling=800, target=None, errors=0.5, rows=[_row(1, 400, 100, 0.1), _row(2, 800, 300, 0.6, selected=True)])
    old = _step_details(ceiling=800, target=None, errors=0.4, rows=[_row(1, 400, 100), _row(2, 800, 280, selected=True)])
    items = _items(rg.compare_stored(current, old, "step", THRESHOLDS))
    # p95 comes from the step rows of both; errors from the checks of both (the same statistic on both sides).
    assert (items["p95"]["current"], items["p95"]["baseline"]) == (300, 280)
    assert (items["errors"]["current"], items["errors"]["baseline"]) == (0.5, 0.4)
    # At a target step far below the ceilings an old row without errors has nothing to compare.
    current = _step_details(rows=[_row(1, 600, 150, 0.2, selected=True), _row(2, 800, 300, 0.5)])
    old = _step_details(rows=[_row(1, 600, 120, selected=True), _row(2, 800, 280)])
    result = rg.compare_stored(current, old, "step", THRESHOLDS)
    assert "errors" not in _items(result) and {"metric": "errors", "reason": "level_mismatch"} in result["skipped"]


def test_p99_is_compared_when_neither_run_has_p95():
    def details(p99):
        return {
            "checks": [{"name": "p99_latency", "threshold": 900, "actual": p99, "passed": True, "query_label": "LT p99"}],
            "stable_window": {"level": 500},
        }

    result = rg.compare_stored(details(440), details(400), "step", THRESHOLDS)
    item = _items(result)["p95"]
    assert item["stat"] == "p99" and item["change"] == 10.0 and item["passed"] is True
    checks = rg.regression_checks(result, THRESHOLDS, "en", baseline_run="base", mode="capacity")
    latency = next(check for check in checks if check["name"] == "regression_latency")
    assert latency["message"].startswith("p99 ") and "neither run has p95" in latency["message"]


def test_soak_holds_must_match_within_five_percent():
    current = _hold_details(300, p95=220, errors=0.5)
    result = rg.compare_stored(current, _hold_details(330, p95=200), "soak", THRESHOLDS)
    assert not result["available"] and result["reason"] == "level_mismatch"
    result = rg.compare_stored(current, _hold_details(310, p95=200, errors=0.3), "soak", THRESHOLDS)
    items = _items(result)
    assert result["kind"] == "hold" and result["level"] == 300 and set(items) == {"p95", "errors"}
    assert items["p95"]["change"] == 10.0 and items["p95"]["passed"] is True
    assert {"metric": "capacity", "reason": "not_compared"} in result["skipped"]


def test_spike_level_is_the_load_of_a_spike_test():
    current = _hold_details(360, p95=500, kind="spike")
    result = rg.compare_stored(current, _hold_details(400, p95=400, kind="spike"), "spike", THRESHOLDS)
    items = _items(result)
    assert result["kind"] == "spike" and set(items) == {"capacity"}
    assert items["capacity"]["delta"] == 10.0 and items["capacity"]["passed"] is True
    with_locale_message = rg.regression_checks(result, {**THRESHOLDS, "regression_capacity_risk_pct": 5}, "en", baseline_run="b", mode="spike")
    assert with_locale_message[0]["message"].startswith("spike level")


def test_scenarios_and_runs_without_checks_are_not_compared():
    assert rg.compare_stored(_step_details(), _step_details(), "scenario", THRESHOLDS)["reason"] == "mode_unsupported"
    assert rg.comparison_point(_step_details(), "scenario")["reasons"]["p95"] == "mode_unsupported"
    assert rg.compare_stored({}, _step_details(), "step")["reason"] == "no_checks"
    assert rg.compare_stored(_step_details(), {"checks": []}, "step")["reason"] == "no_baseline_sla"


def test_own_level_is_the_target_step_else_the_ceiling():
    point = rg.comparison_point(_step_details(rows=[_row(1, 600, 150, 0.2, selected=True), _row(2, 800, 300, 0.5)]), "step")
    assert (point["kind"], point["level"], point["p95"], point["errors"], point["capacity"]) == ("target", 600, 150, 0.2, 800)
    point = rg.comparison_point(_step_details(target=None), "stress")
    assert (point["kind"], point["level"], point["p95"], point["capacity"]) == ("ceiling", 800, 300, 800)
    # No step held the latency and error thresholds: no ceiling to compare.
    failed = _step_details(target=None)
    failed["checks"][2]["passed"] = False
    point = rg.comparison_point(failed, "step")
    assert point["capacity"] is None and point["reasons"]["capacity"] == "no_ceiling"
    # Its p95 and errors are still read on the step the checks were measured on (the trend of a release shows them).
    assert (point["kind"], point["level"], point["p95"]) == ("ceiling", 800, 300)
    assert rg.comparison_point(_hold_details(300), "soak")["kind"] == "hold"


# ---- deltas ------------------------------------------------------------------------------------------------------


def test_compare_points_zero_baseline_negative_delta_and_load_drop():
    def point(p95, errors, capacity):
        return {"p95": p95, "p99": None, "errors": errors, "capacity": capacity, "reasons": {}}

    result = rg.compare_points(point(120, 0.4, 900), point(0, 0.0, 1000), THRESHOLDS)
    assert result["skipped"] == [{"metric": "p95", "reason": "zero_baseline"}]
    items = _items(result)
    assert items["errors"]["change"] == 0.4 and items["errors"]["passed"] is True
    assert items["capacity"]["delta"] == 10.0 and items["capacity"]["passed"] is True  # at the threshold
    better = _items(rg.compare_points(point(80, 0.1, 1100), point(100, 0.2, 1000), THRESHOLDS))
    assert better["p95"]["delta"] == -20.0 and better["p95"]["tone"] == "ok" and better["p95"]["passed"] is True
    assert better["capacity"]["change"] == 10.0 and better["capacity"]["delta"] == -10.0


def test_without_thresholds_items_are_neutral():
    current = _step_details(rows=[_row(1, 600, 150, 0.2, selected=True)])
    baseline = _step_details(rows=[_row(1, 600, 100, 0.2, selected=True)])
    for thresholds in (None, {}, {key: None for key in rg.THRESHOLD_KEY_NAMES}):
        result = rg.compare_stored(current, baseline, "step", thresholds)
        assert all(item["passed"] is None and item["severity"] == "" and item["tone"] == "neutral" for item in result["items"])
        assert rg.regression_checks(result, thresholds, "ru", baseline_run="b") == []


def test_tone_of():
    assert rg.tone_of("p95", 30, THRESHOLDS) == "warn"
    assert rg.tone_of("p95", 5, THRESHOLDS) == "ok"
    assert rg.tone_of("p95", 30, {**THRESHOLDS, "regression_latency_fail_pct": 25}) == "fail"
    assert rg.tone_of("errors", None, THRESHOLDS) == "neutral"
    assert rg.tone_of("capacity", 50, {}) == "neutral"
    assert rg.tone_of("capacity", 30, {"regression_capacity_fail_pct": 20}) == "fail"


def test_thresholds_from_sla_reads_numbers_and_switches_off_the_rest():
    parsed = rg.thresholds_from_sla({
        "regression_latency_risk_pct": "15", "regression_latency_fail_pct": "", "regression_errors_risk_pp": "0,5",
        "regression_errors_fail_pp": -1, "regression_capacity_risk_pct": True, "regression_capacity_fail_pct": None,
    })
    assert parsed == {
        "regression_latency_risk_pct": 15.0, "regression_latency_fail_pct": None, "regression_errors_risk_pp": 0.5,
        "regression_errors_fail_pp": None, "regression_capacity_risk_pct": None, "regression_capacity_fail_pct": None,
    }
    assert not rg.thresholds_enabled(rg.thresholds_from_sla({}))
    assert rg.thresholds_enabled(parsed)


# ---- checks and their messages ------------------------------------------------------------------------------------


def _comparison(fail=None):
    current = _step_details(ceiling=800, rows=[_row(1, 600, 150, 0.9, selected=True), _row(2, 800, 300, 0.5)])
    baseline = _step_details(ceiling=1000, rows=[_row(1, 600, 100, 0.2, selected=True), _row(2, 1000, 300, 0.5)])
    thresholds = {**THRESHOLDS, **(fail or {})}
    return rg.compare_stored(current, baseline, "step", thresholds), thresholds


def test_regression_checks_risk_failure_and_severity():
    comparison, thresholds = _comparison({"regression_latency_fail_pct": 40, "regression_capacity_risk_pct": None, "regression_capacity_fail_pct": 15})
    checks = {check["name"]: check for check in rg.regression_checks(comparison, thresholds, "ru", baseline_run="run-base", mode="capacity")}
    assert set(checks) == set(rg.REGRESSION_CHECK_NAMES)
    latency, errors, capacity = checks["regression_latency"], checks["regression_errors"], checks["regression_capacity"]
    assert (latency["actual"], latency["threshold"], latency["passed"], latency["severity"]) == (50.0, 10.0, False, "critical")
    assert latency["fail_threshold"] == 40 and latency["unit"] == "pct" and latency["category"] == "regression"
    assert (latency["current_value"], latency["baseline_value"], latency["baseline_run"], latency["level"]) == (150, 100, "run-base", 600)
    assert (errors["actual"], errors["passed"], errors["severity"], errors["unit"]) == (0.7, False, "warning", "pp")
    # Only a failure threshold: it is the threshold of the check.
    assert (capacity["threshold"], capacity["actual"], capacity["passed"], capacity["severity"]) == (15, 20.0, False, "critical")
    assert "«run-base»" in latency["message"] and "ступени цели ≈600 RPS" in latency["message"]
    assert "+50.0 %" in latency["message"] and "порог риска +10 %, провала +40 %" in latency["message"]
    assert "п.п." in errors["message"] and "−20.0 %" in capacity["message"]


def test_error_regression_uses_unrounded_percentage_points():
    thresholds = {"regression_errors_fail_pp": 0.02}
    result = rg.compare_stored(_hold_details(300, errors=0.04), _hold_details(300, errors=0), "soak", thresholds)
    item = _items(result)["errors"]
    assert item["delta"] == 0.0  # Display precision must not decide the SLA result.
    assert item["raw_delta"] == pytest.approx(0.04)
    assert item["passed"] is False and item["severity"] == "critical" and item["tone"] == "fail"
    check = rg.regression_checks(result, thresholds, "en", baseline_run="base", mode="stability")[0]
    assert check["actual"] == pytest.approx(0.04) and check["actual"] > check["threshold"]
    assert "+0.04 pp" in check["message"]
    assert se._apply_regression("pass", [check]) == "fail"


@pytest.mark.parametrize("metric,current,baseline", [("p95", 110.04, 100), ("capacity", 89.96, 100)])
def test_percentage_regression_just_above_threshold_is_not_rounded_to_pass(metric, current, baseline):
    key = "regression_latency_risk_pct" if metric == "p95" else "regression_capacity_risk_pct"
    thresholds = {key: 10.0}
    compared = rg.compare_points({metric: current}, {metric: baseline}, thresholds)
    item = _items(compared)[metric]
    assert item["delta"] == 10.0 and item["raw_delta"] == pytest.approx(10.04)
    assert item["passed"] is False and item["severity"] == "warning" and item["tone"] == "warn"
    check = rg.regression_checks({**compared, "available": True}, thresholds, "en", baseline_run="base", mode="capacity")[0]
    assert check["actual"] > 10.0 and "10.04 %" in check["message"]
    assert se._apply_regression("pass", [check]) == "risk"
    # Exact equality still passes despite binary floating-point subtraction noise.
    exact = rg.compare_points({"errors": 0.8}, {"errors": 0.7}, {"regression_errors_risk_pp": 0.1})
    assert _items(exact)["errors"]["passed"] is True


def test_regression_messages_follow_the_language():
    comparison, thresholds = _comparison()
    english = rg.regression_checks(comparison, thresholds, "en", baseline_run="run-base", mode="capacity")
    assert english and all(not CYRILLIC.search(check["message"]) for check in english)
    assert "on the target step ≈600 RPS" in english[0]["message"] and "pp" in english[1]["message"]
    russian = rg.regression_checks(comparison, thresholds, "ru", baseline_run="run-base", mode="capacity")
    assert all(CYRILLIC.search(check["message"]) for check in russian)


# ---- the verdict ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "severity", "expected"),
    [
        ("pass", "warning", "risk"), ("pass", "critical", "fail"),
        ("risk", "warning", "risk"), ("risk", "critical", "fail"),
        ("fail", "warning", "fail"), ("fail", "critical", "fail"),
        ("no_data", "warning", "no_data"), ("no_data", "critical", "no_data"),
    ],
)
def test_apply_regression_verdict_matrix(verdict, severity, expected):
    failed = [{"name": "regression_latency", "category": "regression", "passed": False, "severity": severity}]
    assert se._apply_regression(verdict, failed) == expected
    passed = [{"name": "regression_latency", "category": "regression", "passed": True, "severity": severity}]
    assert se._apply_regression(verdict, passed) == verdict
    assert se._apply_regression(verdict, []) == verdict


def test_regression_checks_are_not_primary():
    assert not se._is_primary({"name": "regression_latency", "category": "regression"})
    assert set(rg.REGRESSION_CHECK_NAMES) <= set(se.SLA_CHECK_NAMES)


def _evaluated(mode="capacity", verdict="pass", checks=None, **extra):
    checks = checks if checks is not None else [
        {"name": "target_rps", "threshold": 500, "actual": 800, "passed": True, "severity": "critical", "category": "primary"},
        {"name": "p95_latency", "threshold": 500, "actual": 300, "passed": True, "category": "primary", "query_label": "LT p95"},
        {"name": "error_rate", "threshold": 1, "actual": 0.5, "passed": True, "category": "primary"},
    ]
    result = {
        "verdict": verdict, "checks": checks, "test_mode": mode,
        "summary_parts": se.sla_summary_parts(verdict, checks, mode), "stable_window": {"level": 800},
        "target_window": {"level": 600, "role": "target"}, "headroom": {"ceiling_level": 800, "verdict_level": 600},
        **extra,
    }
    result["summary"] = se.format_sla_summary(result["summary_parts"], "ru")
    return result


STEP_TABLE = {"latency_query": "LT p95", "steps": [_row(1, 600, 150, 0.2, selected=True), _row(2, 800, 300, 0.5, ceiling=True)]}


def _baseline(service="demo", **details):
    sla = _step_details(ceiling=850, rows=[_row(1, 600, 100, 0.1, selected=True), _row(2, 850, 290, 0.4)])
    sla.update(details)
    return {
        "available": True, "run_name": "run-base", "run_id": "id-base", "release": "25.10", "mode": "pinned",
        "service": service, "test_type": "step", "sla_details": sla,
    }


def test_apply_regression_without_thresholds_changes_nothing():
    evaluated = _evaluated()
    for cfg in ({}, {key: None for key in rg.THRESHOLD_KEY_NAMES}, {"regression_latency_risk_pct": ""}):
        result = se.apply_regression(evaluated, _baseline(), cfg, "ru", step_table=STEP_TABLE, service="demo")
        assert result.pop("regression") == {"available": False, "reason": "disabled"}
        assert result == evaluated


def test_apply_regression_adds_checks_verdict_and_summary():
    result = se.apply_regression(_evaluated(), _baseline(), THRESHOLDS, "ru", step_table=STEP_TABLE, service="demo")
    assert result["verdict"] == "risk"
    names = [check["name"] for check in result["checks"]]
    assert names[:3] == ["target_rps", "p95_latency", "error_rate"] and names[3:] == list(rg.REGRESSION_CHECK_NAMES)
    regression = result["regression"]
    assert regression["available"] and regression["baseline_run_name"] == "run-base" and regression["baseline_run_id"] == "id-base"
    assert (regression["baseline_release"], regression["mode"], regression["kind"], regression["level"]) == ("25.10", "pinned", "target", 600)
    part = result["summary_parts"]["regression"]
    assert part["baseline_run"] == "run-base" and [item["name"] for item in part["items"]] == list(rg.REGRESSION_CHECK_NAMES)
    assert result["summary_parts"]["verdict"] == "risk"
    assert result["summary"].startswith("SLA verdict: Есть риски. Пройдено: target_rps, p95_latency, error_rate; ")
    assert result["summary"].endswith("деградация к «run-base» на ступени цели ≈600 RPS: p95 +50.0 %")
    # The summary is rebuilt from its parts, in either language.
    assert se.format_sla_summary(result["summary_parts"], "ru") == result["summary"]
    english = se.format_sla_summary(result["summary_parts"], "en")
    assert english.endswith("regression against «run-base» on the target step ≈600 RPS: p95 +50.0 %") and not CYRILLIC.search(english)
    critical = se.apply_regression(_evaluated(verdict="risk"), _baseline(), {**THRESHOLDS, "regression_latency_fail_pct": 30}, "ru", step_table=STEP_TABLE, service="demo")
    assert critical["verdict"] == "fail"


def test_apply_regression_without_regression_says_so():
    calm = {"latency_query": "LT p95", "steps": [_row(1, 600, 105, 0.1, selected=True), _row(2, 850, 290, 0.4, ceiling=True)]}
    evaluated = _evaluated()
    result = se.apply_regression(evaluated, _baseline(), THRESHOLDS, "en", step_table=calm, service="demo")
    assert result["verdict"] == "pass" and all(check["passed"] for check in result["checks"])
    assert result["summary"].endswith("no regression against «run-base» on the target step ≈600 RPS")


@pytest.mark.parametrize(
    ("baseline", "service", "reason"),
    [
        ({"available": False, "reason": "text"}, "demo", "no_baseline"),
        ({**_baseline(), "sla_details": {"checks": []}}, "demo", "no_baseline_sla"),
        (_baseline(service="other"), "demo", "other_service"),
        (_baseline(), "", "other_service"),
        ({**_baseline(), "test_type": "soak"}, "demo", "other_type"),
    ],
)
def test_apply_regression_skips_with_a_reason(baseline, service, reason):
    evaluated = _evaluated()
    result = se.apply_regression(evaluated, baseline, THRESHOLDS, "ru", step_table=STEP_TABLE, service=service)
    assert result["regression"]["available"] is False and result["regression"]["reason"] == reason
    assert result["verdict"] == evaluated["verdict"] and result["checks"] == evaluated["checks"]
    reason_text = se.t(f"sla.regression.reason.{reason}", locale="ru")
    assert result["summary"] == evaluated["summary"] + "; сравнения с базой нет: " + reason_text


def test_scenario_regression_is_recorded_without_a_summary_item():
    evaluated = _evaluated(mode="phases")
    result = se.apply_regression(evaluated, _baseline(), THRESHOLDS, "ru", service="demo")
    assert result["regression"]["reason"] == "mode_unsupported"
    assert {key: value for key, value in result.items() if key != "regression"} == evaluated


def test_reconcile_keeps_a_critical_regression_and_downgrades_a_warning_one():
    def stability(severity):
        checks = [
            {"name": "target_rps", "threshold": 250, "actual": 300, "passed": True},
            {"name": "p95_latency", "threshold": 500, "actual": 200, "passed": True},
            {"name": "cpu_usage", "threshold": 80, "actual": 95, "passed": False},
            {"name": "regression_latency", "category": "regression", "threshold": 10, "actual": 30, "passed": False, "severity": severity},
        ]
        return {"verdict": "fail", "checks": checks, "test_mode": "stability", "summary_parts": se.sla_summary_parts("fail", checks, "stability")}

    assert _reconcile_sla_for_test_profile(stability("warning"), {"mode": "stability"})["verdict"] == "risk"
    assert _reconcile_sla_for_test_profile(stability("critical"), {"mode": "stability"})["verdict"] == "fail"


def test_soak_with_resources_over_and_a_critical_regression_fails():
    checks = [
        {"name": "target_rps", "threshold": 250, "actual": 300, "passed": True, "severity": "critical"},
        {"name": "error_rate", "threshold": 1, "actual": 0.3, "passed": True, "category": "primary"},
        {"name": "p95_latency", "threshold": 500, "actual": 260, "passed": True, "category": "primary"},
        {"name": "cpu_usage", "threshold": 80, "actual": 95, "passed": False, "category": "secondary"},
    ]
    evaluated = {"verdict": "risk", "checks": checks, "test_mode": "stability", "summary_parts": se.sla_summary_parts("risk", checks, "stability"),
                 "stable_window": {"level": 300, "kind": "hold"}}
    baseline = {**_baseline(), "test_type": "soak", "sla_details": _hold_details(305, p95=200, errors=0.3)}
    for fail, expected in ((None, "risk"), (20, "fail")):
        cfg = {**THRESHOLDS, "regression_latency_fail_pct": fail}
        result = se.apply_regression(evaluated, baseline, cfg, "ru", service="demo")
        reconciled = _reconcile_sla_for_test_profile(result, {"mode": "stability"})
        assert reconciled["verdict"] == expected
        assert reconciled["summary_parts"]["regression"]["items"][0]["change"] == 30.0


def test_deterministic_context_carries_the_regression_brief():
    result = se.apply_regression(_evaluated(), _baseline(), THRESHOLDS, "ru", step_table=STEP_TABLE, service="demo")
    brief = _deterministic_sla_context(result)["regression"]
    assert brief["baseline_run_name"] == "run-base" and brief["kind"] == "target" and brief["level"] == 600
    assert "skipped" not in brief and [item["name"] for item in brief["items"]] == list(rg.REGRESSION_CHECK_NAMES)
    assert "regression_latency" in _deterministic_sla_context(result)["failed_checks"]
    skipped = se.apply_regression(_evaluated(), {"available": False}, THRESHOLDS, "en", service="demo")
    with with_locale("en"):
        assert _deterministic_sla_context(skipped)["regression"] == {
            "available": False, "reason": "no_baseline", "reason_text": "there is no baseline run",
        }
    off = se.apply_regression(_evaluated(), _baseline(), {}, "ru", service="demo")
    assert "regression" not in _deterministic_sla_context(off)


# ---- the step table ------------------------------------------------------------------------------------------------


def test_step_table_rows_carry_the_error_rate_of_the_step():
    index = pd.date_range("2024-01-01T09:00:00Z", periods=40, freq="1min")
    rps = pd.DataFrame({"sum": np.repeat([100.0, 200.0], 20)}, index=index)
    errors = pd.DataFrame({"a": np.repeat([0.1, 0.8], 20), "b": np.repeat([0.2, 0.3], 20)}, index=index)
    steps = [
        LoadStep(index=1, label="Step 1", start=index[0], end=index[19], minute_from=0, minute_to=20, rps_level=100.0,
                 stable=True, source="test", shift_hours=0),
        LoadStep(index=2, label="Step 2", start=index[20], end=index[39], minute_from=20, minute_to=40, rps_level=200.0,
                 stable=True, source="test", shift_hours=0),
    ]
    labeled = [{"label": "LT RPS", "df": rps}, {"label": "LT errors", "df": errors}]
    table = load_step_report_from_frames(steps, labeled, rps_query="LT RPS", error_query="LT errors")
    assert table["error_query"] == "LT errors"
    assert [row["error_rate"] for row in table["steps"]] == [pytest.approx(0.2), pytest.approx(0.8)]
    plain = load_step_report_from_frames(steps, labeled, rps_query="LT RPS")
    assert "error_query" not in plain and all("error_rate" not in row for row in plain["steps"])


# ---- the baseline row ----------------------------------------------------------------------------------------------


def test_load_baseline_sla_reads_the_final_row(monkeypatch):
    from AI.db_store import StoredRunReport, StoredRunRow

    details = _step_details()
    row = StoredRunRow(
        domain="final", run_id="id-base", service="demo", test_type="step", project_area="", start_ms=None, end_ms=None,
        verdict="pass", sla_verdict="pass", parsed={}, sla_details=details, system_context={}, created_at=None,
    )
    calls = []

    class Conn:
        def close(self):
            calls.append("close")

    monkeypatch.setattr(pipeline_module, "_db_connect", lambda cfg: Conn())
    monkeypatch.setattr(pipeline_module, "load_run_report", lambda conn, schema, table, name: calls.append(name) or StoredRunReport(run_name=name, rows={"final": row}))
    ctx = {"available": True, "run_name": "run-base", "mode": "pinned", "release": "25.10"}
    loaded = _load_baseline_sla({"schema": "public"}, ctx, True)
    assert calls == ["run-base", "close"]
    assert loaded == {
        "available": True, "run_name": "run-base", "run_id": "id-base", "release": "25.10", "mode": "pinned",
        "service": "demo", "test_type": "step", "sla_details": details,
    }
    assert _load_baseline_sla({"schema": "public"}, {"available": False}, True)["reason"] == "no_baseline"
    assert _load_baseline_sla({}, ctx, False)["reason"] == "no_baseline"
    monkeypatch.setattr(pipeline_module, "load_run_report", lambda *a: None)
    assert _load_baseline_sla({"schema": "public"}, ctx, True)["reason"] == "no_baseline_sla"


# ---- validation and scopes -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"regression_latency_risk_pct": 10, "regression_latency_fail_pct": 25},
        {"regression_latency_risk_pct": None, "regression_latency_fail_pct": None},
        {"regression_errors_risk_pp": 0, "regression_errors_fail_pp": 0},
        {"regression_capacity_risk_pct": "", "regression_capacity_fail_pct": 5},
        {"regression_errors_fail_pp": 0.2},
    ],
)
def test_validate_sla_accepts_regression_thresholds(payload):
    validate_sla(payload)


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({"regression_latency_risk_pct": -1}, "regression_latency_risk_pct"),
        ({"regression_errors_risk_pp": "a lot"}, "regression_errors_risk_pp"),
        ({"regression_capacity_fail_pct": True}, "regression_capacity_fail_pct"),
        ({"regression_latency_risk_pct": 20, "regression_latency_fail_pct": 10}, "regression_latency_fail_pct"),
    ],
)
def test_validate_sla_rejects_bad_regression_thresholds(payload, needle):
    with pytest.raises(SlaConfigError, match=needle):
        validate_sla(payload)


def test_thresholds_follow_the_area_and_the_service(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"shop": {
        "sla": {"regression_latency_risk_pct": 20},
        "services": {"checkout": {"sla": {"regression_latency_fail_pct": 40, "regression_errors_risk_pp": None}}, "cart": {}},
    }}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    monkeypatch.setitem(core.CONFIG, "sla", {"regression_latency_risk_pct": 10, "regression_errors_risk_pp": 0.5})
    monkeypatch.setitem(core.CONFIG, "per_area", {})
    checkout = rg.thresholds_from_sla(update_page._effective_config_for_scope("shop", "checkout")["sla"])
    cart = rg.thresholds_from_sla(update_page._effective_config_for_scope("shop", "cart")["sla"])
    assert (checkout["regression_latency_risk_pct"], checkout["regression_latency_fail_pct"], checkout["regression_errors_risk_pp"]) == (20, 40, None)
    assert (cart["regression_latency_risk_pct"], cart["regression_latency_fail_pct"], cart["regression_errors_risk_pp"]) == (20, None, 0.5)


def test_guidance_names_the_regression_check_only_with_thresholds():
    sla = {"max_p95_ms": 500, "p95_query": "LT p95"}
    plain = sla_guidance_lines(sla, mode="capacity", lang="en")
    with_regression = sla_guidance_lines({**sla, **THRESHOLDS}, mode="capacity", lang="en")
    assert [line for line in with_regression if line not in plain] == [line for line in with_regression if "deterministic_sla.regression" in line]
    assert len(with_regression) == len(plain) + 1
    assert sla_guidance_lines({**sla, **THRESHOLDS}, mode="phases", lang="en") == sla_guidance_lines(sla, mode="phases", lang="en")
    assert any("deterministic_sla.regression" in line for line in sla_guidance_lines({**sla, **THRESHOLDS}, mode="stability", lang="ru"))


# ---- the pipeline ---------------------------------------------------------------------------------------------------

START_TS = 1704099600.0  # 2024-01-01T09:00:00Z
END_TS = START_TS + 3600
INDEX = pd.date_range("2024-01-01T09:00:00Z", periods=60, freq="1min")
NOISE = np.random.default_rng(11).normal(0, 1.0, 60)


def _frames(p95_scale=1.0, errors_shift=0.0):
    rps = np.repeat([100.0, 200.0, 300.0, 400.0], 15) + NOISE
    p95 = (np.repeat([100.0, 110.0, 130.0, 130.0], 15) + NOISE) * p95_scale
    p95[52:] = 420.0  # latency breaks late on the last step
    errors = np.repeat([0.1, 0.2, 0.3, 0.5], 15) + errors_shift
    return [
        pd.DataFrame({"sum_all": rps}, index=INDEX),
        pd.DataFrame({"p95_all": p95}, index=INDEX),
        pd.DataFrame({"errors_all": errors}, index=INDEX),
    ]


def _pipeline_cfg(sla_extra=None):
    return {
        "llm": {"include_markdown_tables_in_context": False, "self_consistency_k": 1, "max_domain_workers": 1},
        "default_params": {"step": "1m", "resample_interval": "1T"},
        "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prometheus:9090"}}},
        "domain_sources": {"default": {"source": "prom"}, "lt_framework": {"source": "prom"}},
        "queries": {"lt_framework": {
            "promql_queries": ["a", "b", "c"], "label_keys_list": [[], [], []], "labels": ["LT RPS", "LT p95", "LT errors"],
        }},
        "sla": {
            "max_performance_query": "LT RPS", "p95_query": "LT p95", "error_rate_query": "LT errors",
            "target_rps": 250, "max_p95_ms": 300, "max_error_rate_pct": 2, **(sla_extra or {}),
        },
    }


def _run_pipeline(monkeypatch, frames, sla_extra=None, baseline=None, service="demo"):
    monkeypatch.setattr(pipeline_module, "load_prompt", lambda name, lang=None: "prompt")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *a, **k: frames)

    def fake_llm(user_prompt, data_context, k, return_scores, domain_key):
        payload = {"verdict": "Успешно", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}
        return json.dumps(payload, ensure_ascii=False), dict(payload), {}

    monkeypatch.setattr(pipeline_module, "llm_two_pass_self_consistency", fake_llm)
    if baseline is not None:
        monkeypatch.setattr(pipeline_module, "_load_baseline_context", lambda *a, **k: {"available": True, "run_name": baseline["run_name"], "stats": {}})
        monkeypatch.setattr(pipeline_module, "_load_baseline_sla", lambda *a, **k: copy.deepcopy(baseline))
    return pipeline_module.uploadFromLLM(
        start_ts=START_TS, end_ts=END_TS, save_to_db=False, ef_config=_pipeline_cfg(sla_extra),
        run_meta={"test_type": "step", "run_name": "run-new", "service": service}, active_domains=["lt_framework"],
    )


def _stored_details(result):
    """sla_details as AI.db_store.save_llm_results stores them from a pipeline result."""
    return {
        "checks": result["sla_checks"], "summary": result["sla_summary"], "load_step_table": result["load_step_table"],
        "stable_window": result["sla_window"], "target_window": result["sla_target_window"], "headroom": result["sla_headroom"],
    }


def test_pipeline_compares_the_run_with_the_stored_baseline(monkeypatch):
    first = _run_pipeline(monkeypatch, _frames())
    assert first["sla_regression"] == {"available": False, "reason": "disabled"}
    rows = first["load_step_table"]["steps"]
    assert first["load_step_table"]["error_query"] == "LT errors" and all("error_rate" in row for row in rows)
    baseline = {
        "available": True, "run_name": "run-base", "run_id": "id-base", "release": "", "mode": "previous_success",
        "service": "demo", "test_type": "step", "sla_details": _stored_details(first),
    }
    second = _run_pipeline(monkeypatch, _frames(p95_scale=1.3, errors_shift=0.1), THRESHOLDS, baseline)
    regression = second["sla_regression"]
    assert regression["available"] and regression["kind"] == "target" and regression["baseline_run_name"] == "run-base"
    items = {item["metric"]: item for item in regression["items"]}
    assert items["p95"]["change"] == pytest.approx(30.0, abs=1.5) and items["p95"]["passed"] is False
    assert items["errors"]["change"] == pytest.approx(0.1, abs=0.05) and items["errors"]["passed"] is True
    assert items["capacity"]["delta"] == pytest.approx(0.0, abs=1.0)
    assert {check["name"] for check in second["sla_checks"]} >= set(rg.REGRESSION_CHECK_NAMES)
    assert second["sla_verdict"] == "risk" and first["sla_verdict"] == "pass"
    final = second["final_parsed"]
    assert final["deterministic_sla"]["regression"]["baseline_run_name"] == "run-base"
    assert final["verdict"] == "risk"
    assert "деградация к «run-base»" in second["sla_summary"]


def test_pipeline_skips_a_baseline_of_another_service(monkeypatch):
    first = _run_pipeline(monkeypatch, _frames())
    baseline = {"available": True, "run_name": "run-other", "service": "other", "test_type": "step", "sla_details": _stored_details(first)}
    second = _run_pipeline(monkeypatch, _frames(p95_scale=2.0), THRESHOLDS, baseline)
    assert second["sla_regression"]["reason"] == "other_service"
    assert not any(check["name"].startswith("regression_") for check in second["sla_checks"])


@pytest.mark.parametrize("kind", ["step", "stress", "soak", "spike", "scenario"])
def test_mode_verdicts_do_not_change_with_thresholds_and_no_baseline(monkeypatch, kind):
    """Every test type: thresholds without a baseline keep the verdict and the checks; only a summary note is added."""
    from tests.test_test_types import PIPELINE_CASES, _pipeline_frames, pipeline_queries

    sla, mode, _payload_key, _ctx_key = PIPELINE_CASES[kind]
    results = []
    for extra in ({}, THRESHOLDS):
        cfg = _pipeline_cfg()
        cfg["queries"] = {"lt_framework": pipeline_queries(kind)}
        cfg["sla"] = {"max_performance_query": "LT RPS", "p95_query": "LT p95", **sla, **extra}
        monkeypatch.setattr(pipeline_module, "load_prompt", lambda name, lang=None: "prompt")
        monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *a, **k: _pipeline_frames(kind))
        monkeypatch.setattr(
            pipeline_module, "llm_two_pass_self_consistency",
            lambda *a, **k: (json.dumps({"verdict": "Успешно", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}),
                             {"verdict": "Успешно", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}, {}),
        )
        results.append(pipeline_module.uploadFromLLM(
            start_ts=START_TS, end_ts=END_TS, save_to_db=False, ef_config=cfg,
            run_meta={"test_type": kind, "run_name": f"run-{kind}", "service": "demo"}, active_domains=["lt_framework"],
        ))
    plain, thresholded = results
    assert plain["sla_verdict"] == thresholded["sla_verdict"]
    assert plain["sla_checks"] == thresholded["sla_checks"]
    assert plain["sla_regression"] == {"available": False, "reason": "disabled"}
    if mode == "phases":
        assert thresholded["sla_regression"]["reason"] == "mode_unsupported"
        assert thresholded["sla_summary"] == plain["sla_summary"]
    else:
        assert thresholded["sla_regression"]["reason"] == "no_baseline"
        note = "; сравнения с базой нет: нет базового прогона"
        assert note in thresholded["sla_summary"] and thresholded["sla_summary"].replace(note, "") == plain["sla_summary"]
