# SPDX-License-Identifier: Apache-2.0
"""Regression against a baseline run: the values of the run's own SLA checks (p95, error rate, the load reached)
next to the same values of the baseline run at a comparable load.

Pure functions over stored ``sla_details`` (the final row of a report). The SLA evaluation of a new report adds
deterministic ``regression_*`` checks from them (AI/sla_evaluator.apply_regression); the release page computes live
deltas with the same functions (``compare_stored``, ``tone_of``).

The comparable point depends on the mode of the test type (AI/test_profile.py):

- capacity / stress: the level is the target step when both runs confirmed one (the lower of the two levels), else
  the lower of the two ceilings. p95 and errors come from the row of the load-step table at that level (within
  ``STEP_TOLERANCE_PCT``, as the target step of the SLA); a report without such a row (older reports, no steps)
  falls back to the values of its checks, measured on its ceiling, when the ceiling is within
  ``LEVEL_TOLERANCE_PCT`` of the level. The load reached (``capacity``) is the ceiling.
- stability: the level is the mean RPS of the hold window; p95 and errors come from the checks when the two levels
  are within ``LEVEL_TOLERANCE_PCT``; the load is not compared.
- spike: the level of the highest spike; p95 and errors from the checks within the same tolerance; the load is the
  spike level.
- phases: not compared (``mode_unsupported``): scenario runs have their own reference run (AI/phases.py).

``change`` is the signed raw change (p95 and the load in percent of the baseline, errors in percentage points);
``delta`` is the worsening, positive is worse: the change itself for p95 and errors, the drop for the load.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from AI.i18n import t
from AI.metric_direction import change_pct
from AI.test_profile import CAPACITY_LIKE_MODES, MODES, profile_for

REGRESSION_CHECK_NAMES = ("regression_latency", "regression_errors", "regression_capacity")
CATEGORY_REGRESSION = "regression"
# Two runs measured at levels this far apart (in percent of the higher one) are still comparable.
LEVEL_TOLERANCE_PCT = 5.0
# A step row this far from the level (in percent of the level) is the step at that level; the SLA uses the same
# tolerance for the target step (AI/sla_evaluator.TARGET_STEP_TOLERANCE_PCT, a test keeps them equal).
STEP_TOLERANCE_PCT = 2.0

METRICS = ("p95", "errors", "capacity")
CHECK_OF_METRIC: Dict[str, str] = {
    "p95": "regression_latency",
    "errors": "regression_errors",
    "capacity": "regression_capacity",
}
METRIC_OF_CHECK: Dict[str, str] = {name: metric for metric, name in CHECK_OF_METRIC.items()}
UNIT_PCT = "pct"
UNIT_PP = "pp"
UNIT_OF_METRIC: Dict[str, str] = {"p95": UNIT_PCT, "errors": UNIT_PP, "capacity": UNIT_PCT}
# Risk and failure thresholds of every metric in the SLA section (null = off).
THRESHOLD_KEYS: Dict[str, Tuple[str, str]] = {
    "p95": ("regression_latency_risk_pct", "regression_latency_fail_pct"),
    "errors": ("regression_errors_risk_pp", "regression_errors_fail_pp"),
    "capacity": ("regression_capacity_risk_pct", "regression_capacity_fail_pct"),
}
THRESHOLD_KEY_NAMES: Tuple[str, ...] = tuple(key for pair in THRESHOLD_KEYS.values() for key in pair)

SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
TONE_OK = "ok"
TONE_WARN = "warn"
TONE_FAIL = "fail"
TONE_NEUTRAL = "neutral"

KIND_TARGET = "target"
KIND_CEILING = "ceiling"
KIND_HOLD = "hold"
KIND_SPIKE = "spike"

# Why a comparison or one of its metrics is missing (``sla.regression.reason.<code>``).
REASON_DISABLED = "disabled"
REASON_NO_CHECKS = "no_checks"
REASON_MODE_UNSUPPORTED = "mode_unsupported"
REASON_NO_BASELINE = "no_baseline"
REASON_NO_BASELINE_SLA = "no_baseline_sla"
REASON_OTHER_SERVICE = "other_service"
REASON_OTHER_TYPE = "other_type"
REASON_NO_LEVEL = "no_level"
REASON_LEVEL_MISMATCH = "level_mismatch"
REASON_NO_STEP = "no_step_at_level"
REASON_NO_VALUE = "no_value"
REASON_ZERO_BASELINE = "zero_baseline"
REASON_NO_CEILING = "no_ceiling"
REASON_NOT_COMPARED = "not_compared"
REASONS = (
    REASON_DISABLED, REASON_NO_CHECKS, REASON_MODE_UNSUPPORTED, REASON_NO_BASELINE, REASON_NO_BASELINE_SLA,
    REASON_OTHER_SERVICE, REASON_OTHER_TYPE, REASON_NO_LEVEL, REASON_LEVEL_MISMATCH, REASON_NO_STEP, REASON_NO_VALUE,
    REASON_ZERO_BASELINE, REASON_NO_CEILING, REASON_NOT_COMPARED,
)

_PRIMARY_NAMES = ("error_rate", "p95_latency", "p99_latency")
_CHECK_OF_VALUE = {"p95": "p95_latency", "p99": "p99_latency", "errors": "error_rate"}


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _within(a: Optional[float], b: Optional[float], tolerance_pct: float) -> bool:
    """Whether two levels differ by no more than ``tolerance_pct`` of the higher one."""
    if a is None or b is None:
        return False
    top = max(abs(a), abs(b))
    if top < 1e-9:
        return True
    return abs(a - b) <= top * tolerance_pct / 100.0


def thresholds_from_sla(sla_cfg: Any) -> Dict[str, Optional[float]]:
    """The six ``regression_*`` thresholds of an (effective) SLA section; None = off (empty, null, not a number or
    negative)."""
    cfg = sla_cfg if isinstance(sla_cfg, Mapping) else {}
    out: Dict[str, Optional[float]] = {}
    for key in THRESHOLD_KEY_NAMES:
        raw = cfg.get(key)
        if isinstance(raw, str):
            raw = raw.strip().replace(",", ".")
            if raw.lower() in {"", "null", "none"}:
                raw = None
        number = _num(raw)
        out[key] = number if number is not None and number >= 0 else None
    return out


def thresholds_enabled(thresholds: Optional[Mapping[str, Any]]) -> bool:
    """Whether at least one regression threshold is set."""
    if not isinstance(thresholds, Mapping):
        return False
    return any(_num(thresholds.get(key)) is not None for key in THRESHOLD_KEY_NAMES)


def metric_limits(metric: str, thresholds: Optional[Mapping[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """(risk, fail) thresholds of a metric (p95 | errors | capacity); None when off."""
    keys = THRESHOLD_KEYS.get(str(metric or ""))
    if keys is None or not isinstance(thresholds, Mapping):
        return None, None
    risk, fail = (_num(thresholds.get(key)) for key in keys)
    return risk, fail


def _exceeds(value: float, threshold: Optional[float]) -> bool:
    """Compare unrounded measurements without treating floating-point subtraction noise as a breach."""
    return threshold is not None and value > threshold and not math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def tone_of(metric: str, delta: Any, thresholds: Optional[Mapping[str, Any]]) -> str:
    """Tone of a worsening ``delta`` against the thresholds of the metric: ``fail`` above the failure threshold,
    ``warn`` above the risk threshold, else ``ok``; ``neutral`` without a delta or without thresholds for it."""
    value = _num(delta)
    risk, fail = metric_limits(metric, thresholds)
    if value is None or (risk is None and fail is None):
        return TONE_NEUTRAL
    if _exceeds(value, fail):
        return TONE_FAIL
    if _exceeds(value, risk):
        return TONE_WARN
    return TONE_OK


def mode_of_test_type(test_type: Any) -> str:
    """Mode of a stored or normalized test type (a mode name is accepted as is)."""
    raw = str(test_type or "").strip().lower()
    if raw in MODES:
        return raw
    return profile_for(raw).mode


def _checks(details: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for item in details.get("checks") or []:
        if isinstance(item, Mapping):
            out.setdefault(str(item.get("name") or ""), item)
    return out


def _window_level(window: Any) -> Optional[float]:
    return _num(window.get("level")) if isinstance(window, Mapping) else None


def _primary_failed(checks: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(checks.get(name, {}).get("passed") is False for name in _PRIMARY_NAMES)


def _ceiling(details: Mapping[str, Any], checks: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    """Last step on which the latency and error thresholds hold; None when no step holds them."""
    if _primary_failed(checks):
        return None
    headroom = details.get("headroom")
    level = _num(headroom.get("ceiling_level")) if isinstance(headroom, Mapping) else None
    return level if level is not None else _window_level(details.get("stable_window"))


def own_level(details: Any, test_type: Any) -> Tuple[Optional[float], Optional[str]]:
    """The run's own comparison level and its kind: the target step (else the ceiling, else the step its checks were
    measured on when no step held them) of a step or stress test, the hold level of a soak test, the highest spike of
    a spike test; (None, None) for scenarios and without data."""
    details = details if isinstance(details, Mapping) else {}
    mode = mode_of_test_type(test_type)
    checks = _checks(details)
    if mode in CAPACITY_LIKE_MODES:
        target = _window_level(details.get("target_window"))
        if target is not None:
            return target, KIND_TARGET
        # No step held the thresholds: the step the checks were measured on (the run has no ceiling to compare).
        level = _ceiling(details, checks)
        if level is None:
            level = _window_level(details.get("stable_window"))
        return (level, KIND_CEILING) if level is not None else (None, None)
    if mode in ("stability", "spike"):
        level = _window_level(details.get("stable_window"))
        if level is None:
            level = _num(checks.get("target_rps", {}).get("actual"))
        return (level, KIND_HOLD if mode == "stability" else KIND_SPIKE) if level is not None else (None, None)
    return None, None


def comparable_level(a_details: Any, b_details: Any, test_type: Any) -> Tuple[Optional[float], Optional[str]]:
    """Level two runs are compared at, with its kind.

    Step and stress tests: the lower target step when both runs confirmed one, else the lower ceiling (a run where
    no step holds the thresholds takes the step its checks were measured on). Soak and spike tests: the level of the
    first run (the second one must be within ``LEVEL_TOLERANCE_PCT`` of it).
    """
    a = a_details if isinstance(a_details, Mapping) else {}
    b = b_details if isinstance(b_details, Mapping) else {}
    mode = mode_of_test_type(test_type)
    if mode in CAPACITY_LIKE_MODES:
        targets = [_window_level(a.get("target_window")), _window_level(b.get("target_window"))]
        if all(level is not None for level in targets):
            return min(targets), KIND_TARGET  # type: ignore[type-var]
        ceilings = [_ceiling(item, _checks(item)) or _window_level(item.get("stable_window")) for item in (a, b)]
        if all(level is not None for level in ceilings):
            return min(ceilings), KIND_CEILING  # type: ignore[type-var]
        return None, None
    return own_level(a, mode)


def _step_rows(details: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    table = details.get("load_step_table")
    rows = table.get("steps") if isinstance(table, Mapping) else None
    return [row for row in rows or [] if isinstance(row, Mapping) and not row.get("dip") and _num(row.get("rps")) is not None]


def _row_at(rows: Sequence[Mapping[str, Any]], level: float) -> Optional[Mapping[str, Any]]:
    """Row of the step at ``level`` (within ``STEP_TOLERANCE_PCT``): the SLA step or the ceiling first, then the last
    stable step before a drop, then any."""
    span = abs(level) * STEP_TOLERANCE_PCT / 100.0
    matches = [row for row in rows if abs(float(row["rps"]) - level) <= span + 1e-9]
    if not matches:
        return None
    for test in (
        lambda row: bool(row.get("selected")),
        lambda row: bool(row.get("ceiling")),
        lambda row: row.get("stable") is not False and not row.get("after_drop"),
        lambda row: not row.get("after_drop"),
    ):
        picked = [row for row in matches if test(row)]
        if picked:
            return picked[-1]
    return matches[-1]


def _table_latency_stat(details: Mapping[str, Any], checks: Mapping[str, Mapping[str, Any]]) -> str:
    """Which percentile the latency column of the step table holds: the table takes the p95 query, else the p99 one."""
    table = details.get("load_step_table")
    label = str(table.get("latency_query") or "") if isinstance(table, Mapping) else ""
    p95_label = str(checks.get("p95_latency", {}).get("query_label") or "")
    p99_label = str(checks.get("p99_latency", {}).get("query_label") or "")
    if label and p99_label == label and p95_label != label:
        return "p99"
    return "p95"


def _empty_point(mode: str) -> Dict[str, Any]:
    return {
        "kind": None, "level": None, "step_label": "",
        "p95": None, "p99": None, "errors": None, "capacity": None,
        "mode": mode, "sources": {}, "reasons": {}, "check_values": None,
    }


def _check_values(checks: Mapping[str, Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    return {key: _num(checks.get(name, {}).get("actual")) for key, name in _CHECK_OF_VALUE.items()}


def comparison_point(sla_details: Any, test_type: Any, level: Optional[float] = None, *, kind: Optional[str] = None) -> Dict[str, Any]:
    """Values of a run at a comparison level: ``{kind, level, step_label, p95, p99, errors, capacity}``.

    ``level=None`` takes the run's own level (``own_level``: the target step, else the ceiling). ``kind`` names the
    level when the caller chose it (``comparable_level``). Extra keys: ``mode``, ``sources`` (``steps`` or ``checks``
    per value), ``reasons`` (why a value is missing) and ``check_values`` (the values of the checks when they were
    measured within ``LEVEL_TOLERANCE_PCT`` of the level, else None).
    """
    details = sla_details if isinstance(sla_details, Mapping) else {}
    mode = mode_of_test_type(test_type)
    point = _empty_point(mode)
    if mode not in CAPACITY_LIKE_MODES and mode not in ("stability", "spike"):
        point["reasons"] = {metric: REASON_MODE_UNSUPPORTED for metric in METRICS}
        return point
    checks = _checks(details)
    own, own_kind = own_level(details, mode)
    at = _num(level) if level is not None else own
    point["kind"] = (kind or own_kind) if level is not None else own_kind
    if mode in CAPACITY_LIKE_MODES:
        ceiling = _ceiling(details, checks)
        point["capacity"] = ceiling
        if ceiling is None:
            point["reasons"]["capacity"] = REASON_NO_CEILING
        else:
            point["sources"]["capacity"] = "checks"
        point["level"] = at
        if at is None:
            point["reasons"].update({"p95": REASON_NO_LEVEL, "errors": REASON_NO_LEVEL})
            return point
        measured = _window_level(details.get("stable_window"))
        check_values = _check_values(checks) if _within(measured, at, LEVEL_TOLERANCE_PCT) else None
        point["check_values"] = check_values
        rows = _step_rows(details)
        row = _row_at(rows, at) if rows else None
        if row is not None:
            point["step_label"] = str(row.get("label") or "")
            latency = _num(row.get("latency_p95"))
            if latency is not None:
                point[_table_latency_stat(details, checks)] = latency
            errors = _num(row.get("error_rate"))
            if errors is not None:
                point["errors"] = errors
        for key in ("p95", "p99", "errors"):
            if point[key] is not None:
                point["sources"][key] = "steps"
            elif check_values is not None and check_values.get(key) is not None:
                point[key] = check_values[key]
                point["sources"][key] = "checks"
        for metric in ("p95", "errors"):
            if point[metric] is None and not (metric == "p95" and point["p99"] is not None):
                if check_values is not None:
                    point["reasons"][metric] = REASON_NO_VALUE
                elif rows and row is None:
                    point["reasons"][metric] = REASON_NO_STEP
                else:
                    point["reasons"][metric] = REASON_LEVEL_MISMATCH
        return point
    # Soak and spike tests: the checks were measured on the hold window or the worst spike.
    point["level"] = own
    if mode == "spike":
        point["capacity"] = own
        if own is None:
            point["reasons"]["capacity"] = REASON_NO_LEVEL
        else:
            point["sources"]["capacity"] = "checks"
    else:
        point["reasons"]["capacity"] = REASON_NOT_COMPARED
    if own is None:
        point["reasons"].update({"p95": REASON_NO_LEVEL, "errors": REASON_NO_LEVEL})
        return point
    if at is not None and not _within(own, at, LEVEL_TOLERANCE_PCT):
        point["reasons"].update({"p95": REASON_LEVEL_MISMATCH, "errors": REASON_LEVEL_MISMATCH})
        return point
    values = _check_values(checks)
    point["check_values"] = values
    for key in ("p95", "p99", "errors"):
        point[key] = values[key]
        if values[key] is not None:
            point["sources"][key] = "checks"
    if point["p95"] is None and point["p99"] is None:
        point["reasons"]["p95"] = REASON_NO_VALUE
    if point["errors"] is None:
        point["reasons"]["errors"] = REASON_NO_VALUE
    return point


def _aligned(current: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Points whose value of one metric comes from different sources (the step table of one run, the checks of the
    other) take the checks of both when both have them: the same statistic on both sides."""
    cur, base = dict(current), dict(baseline)
    cur_checks, base_checks = cur.get("check_values"), base.get("check_values")
    if not isinstance(cur_checks, Mapping) or not isinstance(base_checks, Mapping):
        return cur, base
    for key in ("p95", "p99", "errors"):
        sources = (cur.get("sources", {}).get(key), base.get("sources", {}).get(key))
        if sources[0] == sources[1] or None in sources:
            continue
        if cur_checks.get(key) is not None and base_checks.get(key) is not None:
            cur[key], base[key] = cur_checks[key], base_checks[key]
    return cur, base


def _round1(value: float) -> float:
    rounded = round(value, 1)
    return 0.0 if rounded == 0 else rounded


def _item(metric: str, current: float, baseline: float, thresholds: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    unit = UNIT_OF_METRIC[metric]
    if unit == UNIT_PP:
        raw_change = current - baseline
    else:
        raw_change = change_pct(current, baseline) or 0.0
    raw_delta = -raw_change if metric == "capacity" else raw_change
    change, delta = _round1(raw_change), _round1(raw_delta)
    risk, fail = metric_limits(metric, thresholds)
    passed: Optional[bool] = None
    severity = ""
    if risk is not None or fail is not None:
        broken_fail = _exceeds(raw_delta, fail)
        passed = not (broken_fail or _exceeds(raw_delta, risk))
        severity = SEVERITY_CRITICAL if broken_fail else SEVERITY_WARNING
    return {
        "name": CHECK_OF_METRIC[metric],
        "metric": metric,
        "current": current,
        "baseline": baseline,
        "change": change,
        "delta": delta,
        "raw_change": raw_change,
        "raw_delta": raw_delta,
        "unit": unit,
        "passed": passed,
        "severity": severity,
        "tone": tone_of(metric, raw_delta, thresholds),
    }


def compare_points(current: Mapping[str, Any], baseline: Mapping[str, Any], thresholds: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Items of the metrics both points have (``{items, skipped}``); latency is p95, else p99 when neither run has
    p95 (``stat`` says which). A zero or negative baseline of a relative metric is skipped (``zero_baseline``)."""
    items: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for metric in METRICS:
        stat = metric
        if metric == "p95" and (_num(current.get("p95")) is None or _num(baseline.get("p95")) is None):
            if _num(current.get("p99")) is not None and _num(baseline.get("p99")) is not None:
                stat = "p99"
        cur_value, base_value = _num(current.get(stat)), _num(baseline.get(stat))
        if cur_value is None or base_value is None:
            reason = (
                (current.get("reasons") or {}).get(metric)
                or (baseline.get("reasons") or {}).get(metric)
                or REASON_NO_VALUE
            )
            skipped.append({"metric": metric, "reason": reason})
            continue
        if UNIT_OF_METRIC[metric] == UNIT_PCT and base_value <= 0:
            skipped.append({"metric": metric, "reason": REASON_ZERO_BASELINE})
            continue
        item = _item(metric, cur_value, base_value, thresholds)
        if metric == "p95":
            item["stat"] = stat
        items.append(item)
    return {"items": items, "skipped": skipped}


def _unavailable(reason: str, **extra: Any) -> Dict[str, Any]:
    return {"available": False, "reason": reason, "kind": None, "level": None, "step_label": "", "items": [], "skipped": [], **extra}


def compare_stored(
    current_details: Any,
    baseline_details: Any,
    test_type: Any,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Comparison of two stored ``sla_details`` of the same service and test type.

    ``{available, reason?, kind, level, step_label, items, skipped}``; ``items`` as ``compare_points`` builds them
    (``passed``/``severity`` from ``thresholds``: None or all off gives ``passed`` None, ``severity`` '', ``tone``
    neutral). ``available`` is False with a ``reason`` for scenarios, without checks on either side and when no
    metric could be compared (the reason of the first skipped one).
    """
    mode = mode_of_test_type(test_type)
    if mode not in CAPACITY_LIKE_MODES and mode not in ("stability", "spike"):
        return _unavailable(REASON_MODE_UNSUPPORTED)
    current = current_details if isinstance(current_details, Mapping) else {}
    baseline = baseline_details if isinstance(baseline_details, Mapping) else {}
    if not _checks(current):
        return _unavailable(REASON_NO_CHECKS)
    if not _checks(baseline):
        return _unavailable(REASON_NO_BASELINE_SLA)
    level, kind = comparable_level(current, baseline, mode)
    if level is None:
        cur_point = comparison_point(current, mode)
        base_point = comparison_point(baseline, mode)
    else:
        cur_point = comparison_point(current, mode, level, kind=kind)
        base_point = comparison_point(baseline, mode, level, kind=kind)
    cur_point, base_point = _aligned(cur_point, base_point)
    compared = compare_points(cur_point, base_point, thresholds)
    out: Dict[str, Any] = {
        "available": bool(compared["items"]),
        "kind": kind,
        "level": round(level, 2) if level is not None else None,
        "step_label": str(cur_point.get("step_label") or ""),
        "items": compared["items"],
        "skipped": compared["skipped"],
    }
    if not compared["items"]:
        out["reason"] = compared["skipped"][0]["reason"] if compared["skipped"] else REASON_NO_VALUE
    return out


def _number(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "—"
    if abs(number) >= 100 or float(number).is_integer():
        return f"{number:.0f}"
    return f"{number:.2f}".rstrip("0").rstrip(".") if abs(number) < 1 else f"{number:.1f}"


def signed_change(value: Any) -> str:
    """A change with its sign and one decimal: «+25.0», «−11.1» (typographic minus)."""
    number = _num(value) or 0.0
    return f"{number:+.1f}".replace("-", "−")


def unit_text(unit: str, lang: Optional[str] = None) -> str:
    """Visible unit of a delta: ``%`` or the percentage points of the language."""
    return t("sla.regression.unit_pp", locale=lang) if unit == UNIT_PP else "%"


def scope_text(kind: Optional[str], level: Any, lang: Optional[str] = None) -> str:
    """Where two runs were compared: «on the target step ≈600 RPS», «on the hold ≈200 RPS» …; '' without a level."""
    number = _num(level)
    if number is None or kind not in (KIND_TARGET, KIND_CEILING, KIND_HOLD, KIND_SPIKE):
        return ""
    return t(f"sla.regression.scope.{kind}", locale=lang, level=f"{number:.0f}")


def _limit_text(metric: str, risk: Optional[float], fail: Optional[float], lang: Optional[str]) -> str:
    unit = unit_text(UNIT_OF_METRIC[metric], lang)
    sign = "−" if metric == "capacity" else "+"

    def fmt(value: float) -> str:
        return f"{sign}{value:g} {unit}"

    if risk is not None and fail is not None:
        return t("sla.regression.limit_both", locale=lang, risk=fmt(risk), fail=fmt(fail))
    if fail is not None:
        return t("sla.regression.limit_fail", locale=lang, fail=fmt(fail))
    return t("sla.regression.limit_risk", locale=lang, risk=fmt(risk or 0.0))


def check_message(item: Mapping[str, Any], *, run: str, kind: Optional[str], level: Any, mode: str, thresholds: Optional[Mapping[str, Any]], lang: Optional[str] = None) -> str:
    """Message of a regression check in ``lang``: both values, the change, where they were compared and the limit."""
    metric = str(item.get("metric") or "")
    risk, fail = metric_limits(metric, thresholds)
    unit = unit_text(str(item.get("unit") or ""), lang)
    params = {
        "current": _number(item.get("current")),
        "baseline": _number(item.get("baseline")),
        "change": signed_change(item.get("change")),
        "unit": unit,
        "run": run,
        "scope": scope_text(kind, level, lang),
        "limit": _limit_text(metric, risk, fail, lang),
    }
    raw_delta = _num(item.get("raw_delta"))
    display_delta = _num(item.get("delta"))
    if raw_delta is not None and display_delta is not None and any(
        _exceeds(raw_delta, limit) != _exceeds(display_delta, limit) for limit in (risk, fail)
    ):
        params["change"] = f"{float(item.get('raw_change', item.get('change'))):+.6g}".replace("-", "−")
    if metric == "p95":
        key = "sla.regression.latency_p99" if item.get("stat") == "p99" else "sla.regression.latency"
        params["ms"] = t("report.unit.ms", locale=lang)
    elif metric == "errors":
        key = "sla.regression.errors"
    else:
        key = "sla.regression.capacity_spike" if mode == "spike" else "sla.regression.capacity"
    return t(key, locale=lang, **params)


def regression_checks(
    comparison: Mapping[str, Any],
    thresholds: Optional[Mapping[str, Any]],
    lang: Optional[str] = None,
    *,
    baseline_run: str = "",
    mode: str = "",
) -> List[Dict[str, Any]]:
    """SLA checks of a comparison (``compare_stored``): one per item with a threshold, ``category = regression``.

    ``threshold`` is the risk threshold (else the failure one), ``actual`` the worsening, ``passed`` whether it stays
    within both, ``severity`` ``critical`` above the failure threshold, else ``warning``. Items without a threshold of
    their metric give no check.
    """
    if not isinstance(comparison, Mapping) or not comparison.get("available"):
        return []
    kind = comparison.get("kind")
    level = comparison.get("level")
    checks: List[Dict[str, Any]] = []
    for item in comparison.get("items") or []:
        if not isinstance(item, Mapping) or item.get("passed") is None:
            continue
        metric = str(item.get("metric") or "")
        risk, fail = metric_limits(metric, thresholds)
        check: Dict[str, Any] = {
            "name": CHECK_OF_METRIC.get(metric, metric),
            "threshold": risk if risk is not None else fail,
            "actual": item.get("raw_delta", item.get("delta")),
            "passed": item.get("passed"),
            "severity": item.get("severity") or SEVERITY_WARNING,
            "message": check_message(item, run=baseline_run, kind=kind, level=level, mode=mode, thresholds=thresholds, lang=lang),
            "category": CATEGORY_REGRESSION,
            "metric": metric,
            "unit": item.get("unit"),
            "change": item.get("change"),
            "current_value": item.get("current"),
            "baseline_value": item.get("baseline"),
            "baseline_run": baseline_run,
            "level": level,
        }
        if fail is not None and risk is not None:
            check["fail_threshold"] = fail
        if item.get("stat"):
            check["stat"] = item.get("stat")
        checks.append(check)
    return checks


def summary_items(regression: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Compact items stored in ``summary_parts.regression`` and the model context."""
    return [
        {key: item.get(key) for key in ("name", "metric", "stat", "change", "delta", "raw_change", "raw_delta", "unit", "passed", "severity") if key in item}
        for item in regression.get("items") or []
        if isinstance(item, Mapping)
    ]


__all__ = [
    "CATEGORY_REGRESSION",
    "CHECK_OF_METRIC",
    "LEVEL_TOLERANCE_PCT",
    "METRICS",
    "METRIC_OF_CHECK",
    "REASONS",
    "REGRESSION_CHECK_NAMES",
    "STEP_TOLERANCE_PCT",
    "THRESHOLD_KEYS",
    "THRESHOLD_KEY_NAMES",
    "UNIT_OF_METRIC",
    "check_message",
    "comparable_level",
    "compare_points",
    "compare_stored",
    "comparison_point",
    "metric_limits",
    "mode_of_test_type",
    "own_level",
    "regression_checks",
    "scope_text",
    "signed_change",
    "summary_items",
    "thresholds_enabled",
    "thresholds_from_sla",
    "tone_of",
    "unit_text",
]
