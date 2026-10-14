"""Stress test result: saturation point, character of degradation and limiting resource.

Stress runs keep the capacity semantics: the saturation point is the last load step where
latency and error SLA still hold (the SLA window). The step above it, if it broke the SLA,
gives the failure mode. The limiting resource is the node CPU or memory query from the SLA
settings that breached its threshold at the degraded step, otherwise the one that grew most
since the first step (at least ``SATURATION_GROWTH_MIN_PCT``).

``SATURATION_GROWTH_MIN_PCT`` is a default and a candidate for a future ``sla.*`` key.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from AI.context_pack import LoadStep, _utc_numeric, format_iso, to_utc
from AI.sla_evaluator import _labeled_frame, _worst_series_p95

SATURATION_GROWTH_MIN_PCT = 15.0

# (kind, sla query key, sla threshold key)
RESOURCE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("cpu", "cpu_query", "max_cpu_pct"),
    ("memory", "memory_query", "max_memory_pct"),
)
LATENCY_CHECKS = frozenset({"p95_latency", "p99_latency"})


def _step_at(steps: Sequence[LoadStep], moment: pd.Timestamp) -> Optional[LoadStep]:
    return next((step for step in reversed(list(steps)) if to_utc(step.start) <= moment <= to_utc(step.end)), None)


def _next_step(steps: Sequence[LoadStep], step: Optional[LoadStep]) -> Optional[LoadStep]:
    if step is None:
        return None
    ordered = sorted(steps, key=lambda item: to_utc(item.start))
    for position, item in enumerate(ordered):
        if item.index == step.index and position + 1 < len(ordered):
            return ordered[position + 1]
    return None


def failure_mode(degraded_checks: Sequence[str], rps_drop: Optional[pd.Timestamp]) -> str:
    names = {str(name) for name in degraded_checks or ()}
    latency = bool(names & LATENCY_CHECKS)
    errors = "error_rate" in names
    if latency and errors:
        return "latency+errors"
    if latency:
        return "latency"
    if errors:
        return "errors"
    if rps_drop is not None:
        return "throughput_drop"
    return "none"


FAILURE_MODE_TITLES = {
    "latency+errors": "рост задержек и ошибок",
    "latency": "рост задержек",
    "errors": "рост ошибок",
    "throughput_drop": "падение пропускной способности",
    "none": "деградация не зафиксирована",
}


def _mean_between(frame: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    if column not in frame.columns:
        return None
    series = _utc_numeric(frame, column)
    chunk = series[(series.index >= start) & (series.index < end)].dropna()
    return float(chunk.mean()) if not chunk.empty else None


def limiting_resource(
    hr_labeled: Sequence[Dict[str, Any]],
    sla_cfg: Mapping[str, Any],
    window: tuple[pd.Timestamp, pd.Timestamp],
    first_window: Optional[tuple[pd.Timestamp, pd.Timestamp]],
) -> Optional[Dict[str, Any]]:
    """Node resource that limited the load: threshold breach first, then the largest growth."""
    breaches: List[Dict[str, Any]] = []
    growths: List[Dict[str, Any]] = []
    for kind, query_key, threshold_key in RESOURCE_SPECS:
        query = str(sla_cfg.get(query_key) or "").strip()
        frame = _labeled_frame(list(hr_labeled or []), query) if query else None
        if frame is None:
            continue
        value, column = _worst_series_p95(frame, window[0], window[1])
        if value is None or column is None:
            continue
        threshold = sla_cfg.get(threshold_key)
        try:
            threshold_value = float(threshold) if threshold is not None and str(threshold).strip() != "" else None
        except (TypeError, ValueError):
            threshold_value = None
        base = _mean_between(frame, column, *first_window) if first_window else None
        current = _mean_between(frame, column, *window)
        growth = round((current / base - 1.0) * 100.0, 1) if base and current is not None and abs(base) > 1e-9 else None
        item = {
            "kind": kind, "label": query, "series": column, "value": round(value, 2),
            "threshold": threshold_value, "growth_pct": growth,
        }
        if threshold_value is not None and threshold_value > 0 and value > threshold_value:
            breaches.append({**item, "evidence": "threshold", "_rank": value / threshold_value})
        elif growth is not None and growth >= SATURATION_GROWTH_MIN_PCT:
            growths.append({**item, "evidence": "growth", "_rank": growth})
    pool = breaches or growths
    if not pool:
        return None
    best = max(pool, key=lambda item: item["_rank"])
    return {key: value for key, value in best.items() if key != "_rank"}


def build_saturation(
    *,
    sla_window: Optional[Mapping[str, Any]],
    sla_checks: Sequence[Mapping[str, Any]],
    load_steps: Sequence[LoadStep],
    rps_drop: Optional[pd.Timestamp],
    hr_labeled: Sequence[Dict[str, Any]],
    sla_cfg: Mapping[str, Any],
    shift_hours: int,
    designated_peak: Optional[float] = None,
) -> Dict[str, Any]:
    """``saturation`` of a stress report."""
    sla_cfg = sla_cfg if isinstance(sla_cfg, Mapping) else {}
    steps = sorted(load_steps or [], key=lambda item: to_utc(item.start))
    window = sla_window if isinstance(sla_window, Mapping) and sla_window.get("start") and sla_window.get("end") else None
    degraded_checks = [str(name) for name in (window or {}).get("degraded_checks") or []]
    failed_now = [
        str(check.get("name")) for check in sla_checks or []
        if isinstance(check, Mapping) and check.get("passed") is False and str(check.get("category") or "primary") == "primary"
    ]
    saturation_step = degraded_step = None
    saturation_window = None
    if window is not None:
        start, end = to_utc(window["start"]), to_utc(window["end"])
        saturation_window = (start, end)
        saturation_step = _step_at(steps, start)
        if degraded_checks:
            degraded_step = _next_step(steps, saturation_step)
    saturation_rps = float(window["level"]) if window is not None and window.get("level") is not None else designated_peak
    mode = failure_mode(degraded_checks or failed_now, rps_drop)
    limit_reached = bool(degraded_checks or failed_now or rps_drop is not None)
    probe = None
    if degraded_step is not None:
        probe = (to_utc(degraded_step.start), to_utc(degraded_step.end))
    elif saturation_window is not None:
        probe = saturation_window
    first_window = (to_utc(steps[0].start), to_utc(steps[0].end)) if steps else None
    resource = limiting_resource(hr_labeled, sla_cfg, probe, first_window) if probe is not None else None
    if not limit_reached:
        note = "Предел не достигнут: ни одна ступень не нарушила SLA по задержкам и ошибкам, RPS не падал."
    elif window is None:
        note = "Ступень, на которой выполнялись пороги SLA, не найдена; точка насыщения — пиковый уровень RPS."
    else:
        note = ""
    return {
        "saturation_rps": round(saturation_rps, 2) if saturation_rps is not None else None,
        "saturation_step_label": saturation_step.label if saturation_step else None,
        "saturation_window": (
            {"start_iso": format_iso(saturation_window[0], shift_hours), "end_iso": format_iso(saturation_window[1], shift_hours)}
            if saturation_window else None
        ),
        "first_degraded_rps": (window or {}).get("degraded_level"),
        "degraded_step_label": degraded_step.label if degraded_step else None,
        "degraded_checks": degraded_checks or failed_now,
        "failure_mode": mode,
        "failure_mode_title": FAILURE_MODE_TITLES[mode],
        "rps_drop_iso": format_iso(rps_drop, shift_hours) if rps_drop is not None else None,
        "limiting_resource": resource,
        "limit_reached": limit_reached,
        "note": note,
    }


__all__ = [
    "FAILURE_MODE_TITLES",
    "SATURATION_GROWTH_MIN_PCT",
    "build_saturation",
    "failure_mode",
    "limiting_resource",
]
