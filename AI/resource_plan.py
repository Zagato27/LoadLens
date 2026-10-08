"""CPU plan of a load-test run: how many instances of each service a target load needs.

The utilization law is fitted per service on stable load steps: CPU of one instance =
base + cost × (test RPS / instances). Series that differ only by ``instance=`` belong to
one service; a node is a service with one instance. The plan assumes the load spreads
evenly over instances and the request mix of the test stays the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from AI.capacity_forecast import ConfidenceLevel, Instability, LoadLimit, Samples, StepPoint, TargetEstimate
from AI.context_pack import service_key

MIN_PLAN_STEPS = 3
CAUSE_CPU_SHARE = 0.85
MAX_CPU_FRACTION = 1.5
CPU_FIT_MIN_R2 = 0.9
CPU_EXTRAPOLATION = 1.5
CEIL_TOLERANCE = 1e-9

SCALE_MARGIN = 1.06
SCALE_SPAN_CAP = 2.0

PlanStatus = Literal["ok", "missing_setting", "missing_series", "bad_unit", "too_few_steps"]
CeilingSource = Literal["request", "sla", "default"]
AnswerVerdict = Literal["holds", "scale", "tight", "blocked"]
SafeSource = Literal["cpu", "limit", "none"]
ZoneTone = Literal["ok", "warn", "fail"]
MarkKind = Literal["safe", "limit", "tested"]
LEVELS: tuple[ConfidenceLevel, ...] = ("high", "medium", "low")


@dataclass(frozen=True)
class CpuCeiling:
    pct: float
    source: CeilingSource


@dataclass(frozen=True)
class StepLoad:
    """Stable load step trimmed to its fit samples, with the mean test RPS there."""

    number: int
    start: pd.Timestamp
    end: pd.Timestamp
    rps: float


@dataclass(frozen=True)
class UtilizationLine:
    """CPU of one instance = base + cost × RPS per instance; ``measured`` is the last stable step."""

    name: str
    instances: int
    base: float
    cost: float
    r2: float
    measured: float


@dataclass(frozen=True)
class ServicePlan:
    name: str
    instances: int
    instances_needed: Optional[int]
    cpu_ceiling: float
    cpu_base: float
    cpu_measured: float
    cpu_at_target: float
    cpu_after: Optional[float]
    saturation_rps: Optional[float]
    reliable: bool


@dataclass(frozen=True)
class NodeLoad:
    name: str
    cpu_measured: float
    cpu_at_target: float


@dataclass(frozen=True)
class ResourcePlan:
    """Instances per service at ``target_rps``. ``limit_cause`` is the service whose CPU explains the observed limit."""

    status: PlanStatus
    message: str
    ceiling_pct: float
    ceiling_source: CeilingSource
    target_rps: float
    last_step: Optional[int]
    steps_used: int
    services: tuple[ServicePlan, ...]
    nodes: tuple[NodeLoad, ...]
    nodes_message: str
    capacity_rps: Optional[float]
    bottleneck: Optional[str]
    scaled_capacity_rps: Optional[float]
    next_bottleneck: Optional[str]
    limit_service: Optional[str]
    limit_cpu: Optional[float]
    limit_cause: Optional[str]


@dataclass(frozen=True)
class SafeLoad:
    rps: Optional[float]
    source: SafeSource
    service: Optional[str]


@dataclass(frozen=True)
class CapacityAnswer:
    verdict: AnswerVerdict
    confidence: ConfidenceLevel
    confidence_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ScaleZone:
    from_rps: float
    to_rps: float
    tone: ZoneTone


@dataclass(frozen=True)
class ScaleMark:
    rps: float
    kind: MarkKind


@dataclass(frozen=True)
class LoadScale:
    """Load axis for readers: zones up to the safe load, up to the limit and past it, and the key loads."""

    end_rps: float
    tested_rps: float
    zones: tuple[ScaleZone, ...]
    marks: tuple[ScaleMark, ...]


def stable_step_loads(samples: Samples, steps: Sequence[StepPoint], instability: Optional[Instability]) -> list[StepLoad]:
    """Steps without stalls before the instability onset, trimmed to their fit samples."""
    onset = instability.onset_step if instability is not None else math.inf
    windows = {window.number: window for window in samples.steps}
    fit = samples.fit_rows
    width = pd.Timedelta(minutes=samples.bin_minutes)
    loads: list[StepLoad] = []
    for point in steps:
        if point.stalls or point.after_drop or point.number >= onset:
            continue
        window = windows[point.number]
        inside = fit.loc[(fit.index >= window.plateau_start) & (fit.index < window.plateau_end)]
        if inside.empty:
            continue
        loads.append(StepLoad(point.number, inside.index.min(), inside.index.max() + width, float(inside["x"].mean())))
    return loads


def utilization_lines(frame: pd.DataFrame, steps: Sequence[StepLoad], scale: float) -> list[UtilizationLine]:
    """One line per service with data on MIN_PLAN_STEPS stable steps; ``scale`` turns values into CPU fractions."""
    groups: dict[str, list[str]] = {}
    for column in frame.columns:
        groups.setdefault(service_key(str(column)), []).append(str(column))
    lines: list[UtilizationLine] = []
    for key, columns in groups.items():
        line = _line(_display_name(key), _step_rows(frame[columns] * scale, steps))
        if line is not None:
            lines.append(line)
    return lines


def _display_name(key: str) -> str:
    values = [part.split("=", 1)[-1] for part in key.split("|") if part]
    return " · ".join(values) if values else key


def _step_rows(frame: pd.DataFrame, steps: Sequence[StepLoad]) -> list[tuple[float, float, int]]:
    """(RPS per instance, CPU of one instance, instances) of every step with data."""
    rows: list[tuple[float, float, int]] = []
    for step in steps:
        chunk = frame.loc[(frame.index >= step.start) & (frame.index < step.end)]
        counts = chunk.notna().sum(axis=1)
        reported = counts > 0
        if not reported.any():
            continue
        instances = max(1, int(round(float(counts[reported].median()))))
        total = float(chunk.loc[reported].sum(axis=1).mean())
        rows.append((step.rps / instances, total / instances, instances))
    return rows


def _line(name: str, rows: list[tuple[float, float, int]]) -> Optional[UtilizationLine]:
    if len(rows) < MIN_PLAN_STEPS:
        return None
    x = np.array([row[0] for row in rows])
    y = np.array([row[1] for row in rows])
    if float(np.ptp(x)) <= 0:
        return None
    cost, base = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (base + cost * x)) ** 2))
    spread = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / spread if spread > 0 else 1.0
    return UtilizationLine(name, rows[-1][2], float(base), float(cost), r2, rows[-1][1])


def cpu_at(line: UtilizationLine, rps: float, instances: int) -> float:
    """CPU of one instance at ``rps``; a service whose CPU does not grow with the load keeps its measured CPU."""
    if line.cost <= 0:
        return line.measured
    return line.base + line.cost * rps / instances


def instances_for(line: UtilizationLine, rps: float, ceiling: float) -> Optional[int]:
    """Instances that keep one instance at ``ceiling`` or below, never fewer than now;
    None when the CPU that does not depend on the load already reaches the ceiling."""
    if line.cost <= 0:
        return line.instances if line.measured < ceiling else None
    if line.base >= ceiling:
        return None
    return max(line.instances, math.ceil(line.cost * rps / (ceiling - line.base) - CEIL_TOLERANCE))


def saturation_rps(line: UtilizationLine, instances: int, ceiling: float) -> Optional[float]:
    """Test RPS at which ``instances`` reach ``ceiling``; None when CPU does not grow with the load and stays below it."""
    if line.cost <= 0:
        return 0.0 if line.measured >= ceiling else None
    return max(0.0, (ceiling - line.base) * instances / line.cost)


def _cpu_capacity(lines: Sequence[UtilizationLine], instances: dict[str, int], ceilings: dict[str, float]) -> tuple[Optional[float], Optional[str]]:
    """Lowest test RPS at which a service reaches its ceiling, and that service."""
    reachable: list[tuple[float, str]] = []
    for line in lines:
        rps = saturation_rps(line, instances[line.name], ceilings[line.name])
        if rps is not None:
            reachable.append((rps, line.name))
    if not reachable:
        return None, None
    rps, name = min(reachable)
    return rps, name


def _limit_load(lines: Sequence[UtilizationLine], rps: Optional[float]) -> tuple[Optional[str], Optional[float]]:
    """Busiest service at ``rps`` and its CPU per instance."""
    if rps is None or not lines:
        return None, None
    cpu, name = max((cpu_at(line, rps, line.instances), line.name) for line in lines)
    return name, cpu


def _service_plan(line: UtilizationLine, rps: float, ceiling: float) -> ServicePlan:
    needed = instances_for(line, rps, ceiling)
    return ServicePlan(
        name=line.name,
        instances=line.instances,
        instances_needed=needed,
        cpu_ceiling=ceiling,
        cpu_base=line.base if line.cost > 0 else line.measured,
        cpu_measured=line.measured,
        cpu_at_target=cpu_at(line, rps, line.instances),
        cpu_after=None if needed is None else cpu_at(line, rps, needed),
        saturation_rps=saturation_rps(line, line.instances, ceiling),
        reliable=line.r2 >= CPU_FIT_MIN_R2,
    )


def build_plan(
    lines: Sequence[UtilizationLine],
    nodes: Sequence[UtilizationLine],
    nodes_message: str,
    steps: Sequence[StepLoad],
    rps: float,
    ceiling: CpuCeiling,
    limit: LoadLimit,
) -> ResourcePlan:
    """Plan for ``rps``. A service whose CPU explains the observed limit keeps that CPU as its ceiling:
    the test showed it breaking there."""
    share = ceiling.pct / 100.0
    limit_service, limit_cpu = _limit_load(lines, limit.rps)
    cause = limit_service if limit_cpu is not None and limit_cpu >= CAUSE_CPU_SHARE else None
    ceilings = {
        line.name: min(share, limit_cpu) if line.name == cause and limit_cpu is not None else share
        for line in lines
    }
    plans = sorted(
        (_service_plan(line, rps, ceilings[line.name]) for line in lines),
        key=lambda plan: plan.cpu_at_target,
        reverse=True,
    )
    capacity, bottleneck = _cpu_capacity(lines, {line.name: line.instances for line in lines}, ceilings)
    scaled = {plan.name: plan.instances_needed or plan.instances for plan in plans}
    scaled_capacity, next_bottleneck = _cpu_capacity(lines, scaled, ceilings)
    node_loads = sorted(
        (NodeLoad(node.name, node.measured, cpu_at(node, rps, node.instances)) for node in nodes),
        key=lambda node: node.cpu_at_target,
        reverse=True,
    )
    return ResourcePlan(
        status="ok",
        message="",
        ceiling_pct=ceiling.pct,
        ceiling_source=ceiling.source,
        target_rps=rps,
        last_step=steps[-1].number if steps else None,
        steps_used=len(steps),
        services=tuple(plans),
        nodes=tuple(node_loads),
        nodes_message=nodes_message,
        capacity_rps=capacity,
        bottleneck=bottleneck,
        scaled_capacity_rps=scaled_capacity,
        next_bottleneck=next_bottleneck,
        limit_service=limit_service,
        limit_cpu=limit_cpu,
        limit_cause=cause,
    )


def unavailable_plan(status: PlanStatus, message: str, ceiling: CpuCeiling, rps: float) -> ResourcePlan:
    return ResourcePlan(
        status=status, message=message, ceiling_pct=ceiling.pct, ceiling_source=ceiling.source, target_rps=rps,
        last_step=None, steps_used=0, services=(), nodes=(), nodes_message="", capacity_rps=None, bottleneck=None,
        scaled_capacity_rps=None, next_bottleneck=None, limit_service=None, limit_cpu=None, limit_cause=None,
    )


def limit_applies(plan: ResourcePlan, limit: LoadLimit) -> bool:
    """The observed limit stays a constraint of its own unless a service's CPU explains it."""
    return limit.rps is not None and not (plan.status == "ok" and plan.limit_cause is not None)


def safe_capacity(plan: ResourcePlan, limit: LoadLimit, headroom_share: float) -> SafeLoad:
    """Current configuration: the lower of the CPU capacity and the limit minus the headroom."""
    candidates: list[tuple[float, SafeSource, Optional[str]]] = []
    if plan.status == "ok" and plan.capacity_rps is not None:
        candidates.append((plan.capacity_rps, "cpu", plan.bottleneck))
    if limit_applies(plan, limit) and limit.rps is not None:
        candidates.append((limit.rps * (1.0 - headroom_share), "limit", None))
    if not candidates:
        return SafeLoad(rps=None, source="none", service=None)
    rps, source, service = min(candidates, key=lambda item: item[0])
    return SafeLoad(rps=rps, source=source, service=service)


def load_scale(safe: SafeLoad, limit: LoadLimit, tested_rps: float, target_rps: Optional[float]) -> LoadScale:
    """Zones and marks of the load axis; the axis reaches past the tested load, at most SCALE_SPAN_CAP times it."""
    references = [value for value in (safe.rps, limit.rps, target_rps) if value is not None]
    end = SCALE_MARGIN * min(max([tested_rps, *references]), SCALE_SPAN_CAP * tested_rps)
    marks = [ScaleMark(tested_rps, "tested")]
    if safe.rps is not None:
        marks.append(ScaleMark(safe.rps, "safe"))
    if limit.rps is not None:
        marks.append(ScaleMark(limit.rps, "limit"))
    if limit.rps is None:
        zones = [ScaleZone(0.0, tested_rps, "ok")]
    else:
        safe_end = min(safe.rps, limit.rps) if safe.rps is not None else limit.rps
        zones = [ScaleZone(0.0, safe_end, "ok"), ScaleZone(safe_end, limit.rps, "warn"), ScaleZone(limit.rps, end, "fail")]
    return LoadScale(
        end_rps=end,
        tested_rps=tested_rps,
        zones=tuple(zone for zone in zones if zone.to_rps > zone.from_rps),
        marks=tuple(sorted(marks, key=lambda mark: mark.rps)),
    )


def _verdict(limit_verdict: str, applies: bool, scaling: bool, stuck: bool) -> AnswerVerdict:
    if stuck or (applies and limit_verdict == "over_limit"):
        return "blocked"
    if scaling:
        return "scale"
    if applies and limit_verdict == "no_headroom":
        return "tight"
    return "holds"


def _plan_reasons(plan: ResourcePlan, tested_rps: float, verdict: AnswerVerdict) -> list[tuple[str, ConfidenceLevel]]:
    """Weak CPU line of a service that decides the answer, CPU extrapolated far, an untested scaled configuration."""
    deciding = [item for item in plan.services if item.name == plan.bottleneck or (item.instances_needed or 0) > item.instances]
    reasons: list[tuple[str, ConfidenceLevel]] = []
    if any(not item.reliable for item in deciding):
        reasons.append(("cpu_fit", "medium"))
    if tested_rps > 0 and plan.target_rps > CPU_EXTRAPOLATION * tested_rps:
        reasons.append(("cpu_extrapolation", "medium"))
    if verdict == "scale":
        reasons.append(("scaling_untested", "medium"))
    return reasons


def capacity_answer(target: TargetEstimate, plan: ResourcePlan, limit: LoadLimit, tested_rps: float) -> CapacityAnswer:
    """Verdict for the target: the CPU plan when it explains the limit, the observed limit otherwise."""
    planned = plan.status == "ok"
    applies = limit_applies(plan, limit)
    scaling = planned and any(item.instances_needed is None or item.instances_needed > item.instances for item in plan.services)
    stuck = planned and any(item.instances_needed is None for item in plan.services)
    verdict = _verdict(target.verdict, applies, scaling, stuck)
    reasons = _plan_reasons(plan, tested_rps, verdict) if planned else []
    levels = [LEVELS.index(level) for _, level in reasons]
    codes = [code for code, _ in reasons]
    if applies or not planned:
        levels.append(LEVELS.index(target.confidence))
        codes = list(target.confidence_reasons) + codes
    return CapacityAnswer(verdict=verdict, confidence=LEVELS[max(levels, default=0)], confidence_reasons=tuple(codes))


__all__ = [
    "MAX_CPU_FRACTION",
    "MIN_PLAN_STEPS",
    "CapacityAnswer",
    "CpuCeiling",
    "LoadScale",
    "NodeLoad",
    "ResourcePlan",
    "SafeLoad",
    "ScaleMark",
    "ScaleZone",
    "ServicePlan",
    "StepLoad",
    "UtilizationLine",
    "build_plan",
    "capacity_answer",
    "cpu_at",
    "instances_for",
    "limit_applies",
    "load_scale",
    "safe_capacity",
    "saturation_rps",
    "stable_step_loads",
    "unavailable_plan",
    "utilization_lines",
]
