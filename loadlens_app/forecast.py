# SPDX-License-Identifier: Apache-2.0
"""Capacity forecast for one stored report: samples, USL fit and the calculator answer.

Texts (errors, notes, plan messages) are :func:`AI.i18n.msg` values rendered in the current locale: the request
locale for the forecast page, the report language when ``forecast_confluence.publish_forecast`` builds the page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg2

from AI.capacity_forecast import (
    LOAD_MODELS,
    LATENCY_UNIT_SECONDS,
    ROLE_DEGRADED,
    ROLE_EDGE,
    ROLE_STALL,
    CapacityModelError,
    CurvePoint,
    Instability,
    KneePoint,
    LoadCoverage,
    LoadLimit,
    ModelContext,
    SamplePoint,
    Samples,
    StepPoint,
    StepWindow,
    TargetEstimate,
    UslFit,
    align_samples,
    concurrency_for,
    curve_upper,
    detect_instability,
    evaluate_target,
    fit_usl,
    knee_point,
    load_coverage,
    load_limit,
    model_curve,
    p95_ratio,
    sample_points,
    step_points,
    step_windows,
)
from AI.context_pack import DEFAULT_ELASTICITY_THRESHOLD, format_iso, to_utc
from AI.db_store import StoredFinalReport, list_final_reports, load_final_report, load_run_frames
from AI.query_packs import ROLES, resolve_sla_refs
from AI.pipeline import _cfg_float, _designated_rps_series, _time_shift_hours
from AI.test_profile import profile_for
from AI.verdicts import to_code
from AI.resource_plan import (
    MAX_CPU_FRACTION,
    MIN_PLAN_STEPS,
    CapacityAnswer,
    CpuCeiling,
    LoadScale,
    ResourcePlan,
    SafeLoad,
    StepLoad,
    UtilizationLine,
    build_plan,
    capacity_answer,
    limit_applies,
    load_scale,
    safe_capacity,
    stable_step_loads,
    unavailable_plan,
    utilization_lines,
)
from AI.sla_evaluator import PRIMARY_SPECS, _safe_float, _worst_series_p95
from AI.i18n import DEFAULT, has_key, msg, render_ref, t
from AI.report_texts import localized_step_label
from loadlens_app.core import _find_area_for_service, _ts_conn
from loadlens_app.errors import log_failure, with_error_id
from loadlens_app.i18n import stored_report_language
from settings import CONFIG
from update_page import _effective_config_for_scope

LT_DOMAIN = "lt_framework"
# Domains of the CPU series follow the query-pack role registry (runtime.cpu, node.cpu_pct).
SERVICES_CPU_DOMAIN = ROLES["runtime.cpu"].domain
NODES_CPU_DOMAIN = ROLES["node.cpu_pct"].domain
NODE_CPU_SCALE = 0.01  # sla.cpu_query is node CPU in percent
DEFAULT_CPU_CEILING_PCT = 80.0
# The SLA checks that decide the report maximum (latency and errors); CPU and memory do not move it.
CEILING_CHECKS = frozenset(spec[0] for spec in PRIMARY_SPECS)


class ForecastError(Exception):
    """Forecast is unavailable; ``code`` lets the page tell setup problems from inapplicable tests.

    ``message`` is usually a :func:`AI.i18n.msg`: ``str()`` renders it again in the locale current where the error
    is shown (the request of the forecast page, the job of a publication).
    """

    def __init__(self, message: str, status_code: int, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code

    def __str__(self) -> str:
        return render_ref(self.message, str(self.message))


def _not_applicable() -> str:
    return msg("capacity.error.not_applicable")


def _missing_series(label: str) -> str:
    return msg("capacity.error.missing_series", label=label, domain=msg("common.domain.lt_framework"))


@dataclass(frozen=True)
class ForecastSettings:
    rps_label: str
    mean_label: str
    vus_label: str
    p95_label: str
    latency_unit: str
    load_model: str
    elasticity_threshold: float
    service_cpu_label: str
    node_cpu_label: str
    sla_cpu_pct: Optional[float]


@dataclass(frozen=True)
class InputSeries:
    rps: pd.Series
    mean: pd.Series
    vus: Optional[pd.Series]
    p95: Optional[pd.Series]
    notes: list[str]


@dataclass(frozen=True)
class SlaCheckName:
    name: str
    title: str


@dataclass(frozen=True)
class SlaCeiling:
    """Report maximum (``sla_details.stable_window``): the last load step on which the SLA latency and error
    thresholds hold. ``end_iso`` ends its step; the samples after it lie above the maximum. ``degraded_*`` are the
    level and the broken checks of the step after it, as the report states them. ``limit_rps`` is the load limit
    it sets for the forecast (None when the window has no samples above the maximum)."""

    rps: Optional[float]
    step_number: Optional[int]
    step_label: str
    end_iso: str
    degraded_rps: Optional[float]
    degraded_checks: list[SlaCheckName]
    limit_rps: Optional[float] = None


@dataclass(frozen=True)
class ForecastReport:
    run_id: str
    run_name: str
    service: str
    load_model: str
    latency_unit: str
    window_start_iso: str
    cutoff_iso: str
    cutoff_kind: str
    bin_minutes: float
    fit: UslFit
    tested_concurrency: float
    tested_rps: float
    peak_rps: float
    samples_used: int
    excluded_edges: int
    excluded_stalls: int
    excluded_degraded: int
    think_time_s: float
    p95_ratio: Optional[float]
    knee: Optional[KneePoint]
    instability: Optional[Instability]
    limit: LoadLimit
    limit_applies: bool
    safe: SafeLoad
    scale: LoadScale
    target: Optional[TargetEstimate]
    answer: Optional[CapacityAnswer]
    resources: ResourcePlan
    points: list[SamplePoint]
    steps: list[StepPoint]
    curve: list[CurvePoint]
    sla_target_rps: Optional[float]
    sla_p95_ms: Optional[float]
    sla_ceiling: Optional[SlaCeiling]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_forecast(
    run_id: str,
    target_rps: Optional[float],
    headroom_pct: float,
    cpu_ceiling_pct: Optional[float] = None,
) -> ForecastReport:
    """Fits the model on the stored samples of a run and evaluates the target load."""
    report = _stored_report(run_id)
    if not profile_for(report.test_type).forecast_allowed:
        raise ForecastError(_not_applicable(), 422, "not_applicable")
    settings = _forecast_settings(_scope_sla(report))
    start, cutoff, cutoff_kind = _window(report)
    inputs = _input_series(_stored_frames(run_id, settings), settings, start, cutoff)
    steps = _report_steps(report)
    sla_ceiling, ceiling_end = _sla_ceiling(report, steps, _time_shift_hours())
    samples = _samples(inputs, settings, start, cutoff, steps, ceiling_end)
    notes = [*inputs.notes, *_ceiling_notes(report, sla_ceiling)]
    ceiling = _cpu_ceiling(cpu_ceiling_pct, settings)
    return _assemble(
        report, settings, samples, start, cutoff, cutoff_kind, notes, target_rps, headroom_pct, ceiling, sla_ceiling,
    )


def _storage() -> dict[str, Any]:
    raw = (CONFIG.get("storage") or {}).get("timescale") or {}
    return raw if isinstance(raw, dict) else {}


def _stored_report(run_id: str) -> StoredFinalReport:
    cfg = _storage()
    conn = _ts_conn()
    try:
        report = load_final_report(conn, str(cfg.get("schema") or "public"), str(cfg.get("llm_table") or "llm_reports"), run_id)
    finally:
        conn.close()
    if report is None:
        raise ForecastError(msg("capacity.error.not_found"), 404, "not_found")
    if report.start_ms is None or report.end_ms is None:
        raise ForecastError(msg("capacity.error.missing_window"), 422, "missing_window")
    return report


def report_language_of(run_id: str) -> str:
    """Language of the stored report of a run: ``ru`` for reports saved before the field or when it cannot be read."""
    cfg = _storage()
    try:
        conn = _ts_conn()
        try:
            report = load_final_report(conn, str(cfg.get("schema") or "public"), str(cfg.get("llm_table") or "llm_reports"), run_id)
        finally:
            conn.close()
    except Exception:
        return DEFAULT
    return stored_report_language(report) if report is not None else DEFAULT


def _scope_sla(report: StoredFinalReport) -> dict[str, Any]:
    area = report.project_area or _find_area_for_service(report.service) or ""
    return _resolved_sla(_effective_config_for_scope(area, report.service or None))


def _resolved_sla(cfg: dict[str, Any]) -> dict[str, Any]:
    """SLA of the scope with empty query references filled from query-pack roles."""
    sla = cfg.get("sla") if isinstance(cfg.get("sla"), dict) else {}
    queries = cfg.get("queries") if isinstance(cfg.get("queries"), dict) else {}
    return resolve_sla_refs(sla, queries)[0]


def _missing_setting(title: str) -> ForecastError:
    """``title`` is a msg of capacity.setting.*."""
    return ForecastError(msg("capacity.error.missing_setting", title=title, hint=msg("capacity.setting_hint")), 422, "missing_setting")


def _required_text(sla: dict[str, Any], key: str, title: str) -> str:
    value = str(sla.get(key) or "").strip()
    if not value:
        raise _missing_setting(title)
    return value


def _forecast_settings(sla: dict[str, Any]) -> ForecastSettings:
    model = str(sla.get("load_model") or "").strip()
    unit = str(sla.get("latency_unit") or "").strip()
    if model not in LOAD_MODELS:
        raise _missing_setting(msg("capacity.setting.load_model"))
    if unit not in LATENCY_UNIT_SECONDS:
        raise _missing_setting(msg("capacity.setting.latency_unit"))
    vus_label = str(sla.get("vus_query") or "").strip()
    if model == "closed" and not vus_label:
        raise _missing_setting(msg("capacity.setting.vus_query"))
    return ForecastSettings(
        rps_label=_required_text(sla, "max_performance_query", msg("capacity.setting.rps_query")),
        mean_label=_required_text(sla, "mean_latency_query", msg("capacity.setting.mean_query")),
        vus_label=vus_label,
        p95_label=str(sla.get("p95_query") or "").strip(),
        latency_unit=unit,
        load_model=model,
        elasticity_threshold=_cfg_float(sla.get("step_elasticity_threshold"), DEFAULT_ELASTICITY_THRESHOLD),
        service_cpu_label=str(sla.get("service_cpu_query") or "").strip(),
        node_cpu_label=str(sla.get("cpu_query") or "").strip(),
        sla_cpu_pct=_positive(_safe_float(sla.get("max_cpu_pct"))),
    )


def _positive(value: Optional[float]) -> Optional[float]:
    return value if value is not None and value > 0 else None


def _cpu_ceiling(requested: Optional[float], settings: ForecastSettings) -> CpuCeiling:
    """The request value, then SLA max_cpu_pct, then DEFAULT_CPU_CEILING_PCT; the page shows the source."""
    if requested is not None:
        return CpuCeiling(requested, "request")
    if settings.sla_cpu_pct is not None:
        return CpuCeiling(settings.sla_cpu_pct, "sla")
    return CpuCeiling(DEFAULT_CPU_CEILING_PCT, "default")


def _window(report: StoredFinalReport) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    start = pd.Timestamp(int(report.start_ms or 0), unit="ms", tz="UTC")
    raw = str((report.context or {}).get("rps_drop_iso") or "").strip()
    if raw:
        return start, to_utc(raw), "rps_drop"
    return start, pd.Timestamp(int(report.end_ms or 0), unit="ms", tz="UTC"), "test_end"


def _stored_frames(run_id: str, settings: ForecastSettings) -> dict[str, pd.DataFrame]:
    labels = [settings.rps_label, settings.mean_label, settings.vus_label, settings.p95_label]
    cfg = _storage()
    conn = _ts_conn()
    try:
        return load_run_frames(
            conn,
            str(cfg.get("schema") or "public"),
            str(cfg.get("table") or "metrics"),
            run_id,
            LT_DOMAIN,
            labels,
        )
    finally:
        conn.close()


def _one_series(frame: Optional[pd.DataFrame], label: str) -> pd.Series:
    if frame is None or frame.empty:
        raise ForecastError(_missing_series(label), 422, "missing_series")
    if len(list(frame.columns)) != 1:
        raise ForecastError(msg("capacity.error.multiple_series", label=label), 422, "multiple_series")
    return frame.iloc[:, 0]


def _rps_series(frames: dict[str, pd.DataFrame], label: str) -> pd.Series:
    frame = frames.get(label)
    series = _designated_rps_series([{"label": label, "df": frame}], label) if frame is not None else None
    if series is None or series.dropna().empty:
        raise ForecastError(_missing_series(label), 422, "missing_series")
    return series


def _optional_vus(frames: dict[str, pd.DataFrame], label: str) -> tuple[Optional[pd.Series], list[str]]:
    if not label:
        return None, []
    frame = frames.get(label)
    if frame is None or frame.empty:
        return None, [msg("capacity.note.vus_missing", label=label)]
    if len(list(frame.columns)) != 1:
        return None, [msg("capacity.note.vus_multiple", label=label)]
    return frame.iloc[:, 0], []


def _p95_series(
    frames: dict[str, pd.DataFrame],
    label: str,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[Optional[pd.Series], list[str]]:
    if not label:
        return None, [msg("capacity.note.p95_unset")]
    frame = frames.get(label)
    if frame is None or frame.empty:
        return None, [msg("capacity.note.p95_missing", label=label)]
    name = _worst_series_p95(frame, start, cutoff)[1]
    if not name or name not in frame.columns:
        return None, [msg("capacity.note.p95_no_points", label=label)]
    return frame[name], []


def _input_series(
    frames: dict[str, pd.DataFrame],
    settings: ForecastSettings,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> InputSeries:
    notes: list[str] = []
    if settings.load_model == "closed":
        vus: Optional[pd.Series] = _one_series(frames.get(settings.vus_label), settings.vus_label)
    else:
        vus, vus_notes = _optional_vus(frames, settings.vus_label)
        notes.extend(vus_notes)
    p95, p95_notes = _p95_series(frames, settings.p95_label, start, cutoff)
    notes.extend(p95_notes)
    mean = _one_series(frames.get(settings.mean_label), settings.mean_label)
    p95, unit_notes = _p95_in_mean_units(mean, p95, settings.latency_unit)
    notes.extend(unit_notes)
    return InputSeries(
        rps=_rps_series(frames, settings.rps_label),
        mean=mean,
        vus=vus,
        p95=p95,
        notes=notes,
    )


def _p95_in_mean_units(mean: pd.Series, p95: Optional[pd.Series], unit: str) -> tuple[Optional[pd.Series], list[str]]:
    """Puts p95 into the mean series unit when one query is stored in ms and the other in seconds."""
    if p95 is None:
        return None, []
    joined = pd.concat({"mean": mean, "p95": p95}, axis=1).dropna()
    joined = joined[(joined["mean"] > 0) & (joined["p95"] > 0)]
    if joined.empty:
        return p95, []
    ratio = float((joined["p95"] / joined["mean"]).median())
    if unit == "s" and ratio >= 50.0:
        return p95 / 1000.0, [msg("capacity.note.p95_ms_to_s")]
    if unit == "ms" and 0 < ratio <= 0.02:
        return p95 * 1000.0, [msg("capacity.note.p95_s_to_ms")]
    return p95, []


def _report_steps(report: StoredFinalReport) -> tuple[StepWindow, ...]:
    """Load steps of the report; their stored labels (in the report's language) are shown in the current locale."""
    steps = report.context.get("load_steps") if isinstance(report.context, dict) else None
    windows = step_windows(steps if isinstance(steps, list) else [])
    return tuple(replace(window, label=localized_step_label(window.label)) for window in windows)


def _timestamp(value: Any) -> Optional[pd.Timestamp]:
    try:
        stamp = to_utc(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(stamp) else stamp


def _check_names(raw: Any) -> list[SlaCheckName]:
    """Broken checks of the step after the maximum with their titles in the current locale."""
    names = [str(item).strip() for item in raw if str(item or "").strip()] if isinstance(raw, list) else []
    return [SlaCheckName(name, t(f"sla.check_title.{name}") if has_key(f"sla.check_title.{name}") else name) for name in names]


def _sla_details(report: StoredFinalReport) -> dict[str, Any]:
    return report.sla_details if isinstance(report.sla_details, dict) else {}


def _broken_on_ceiling(details: dict[str, Any]) -> list[SlaCheckName]:
    """Latency and error checks that fail on the ceiling window itself. The report checks them there, and keeps its
    latest step as the window when no step holds them: then the run has no maximum."""
    checks = details.get("checks")
    if not isinstance(checks, list):
        return []
    return _check_names([
        check.get("name") for check in checks
        if isinstance(check, dict) and str(check.get("name") or "") in CEILING_CHECKS and check.get("passed") is False
    ])


def _sla_ceiling(
    report: StoredFinalReport,
    steps: tuple[StepWindow, ...],
    shift_hours: int,
) -> tuple[Optional[SlaCeiling], Optional[pd.Timestamp]]:
    """Report maximum and the end of its load step (the end of the window itself without report steps).

    The forecast follows the report and does not evaluate the SLA again. (None, None) for a report without a step
    window (no SLA thresholds, an old report), with bounds that cannot be read or when no step holds the latency and
    error thresholds.
    """
    details = _sla_details(report)
    window = details.get("stable_window")
    if not isinstance(window, dict) or str(window.get("kind") or "step") != "step" or _broken_on_ceiling(details):
        return None, None
    start, end = _timestamp(window.get("start")), _timestamp(window.get("end"))
    if start is None or end is None or end <= start:
        return None, None
    step = next((item for item in reversed(steps) if item.start <= start <= item.end), None)
    ceiling_end = step.end if step is not None else end
    ceiling = SlaCeiling(
        rps=_positive(_safe_float(window.get("level"))),
        step_number=step.number if step is not None else None,
        step_label=step.label if step is not None else "",
        end_iso=format_iso(ceiling_end, shift_hours),
        degraded_rps=_positive(_safe_float(window.get("degraded_level"))),
        degraded_checks=_check_names(window.get("degraded_checks")),
    )
    return ceiling, ceiling_end


def _ceiling_notes(report: StoredFinalReport, ceiling: Optional[SlaCeiling]) -> list[str]:
    """Why the steps are judged by stalls only: the report has no maximum, or no step holds its thresholds."""
    if ceiling is not None:
        return []
    broken = _broken_on_ceiling(_sla_details(report))
    if broken:
        return [msg("capacity.note.sla_broken_on_ceiling", checks=", ".join(check.title for check in broken))]
    return [msg("capacity.note.no_sla_ceiling")]


def _samples(
    inputs: InputSeries,
    settings: ForecastSettings,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    steps: tuple[StepWindow, ...],
    ceiling_end: Optional[pd.Timestamp] = None,
) -> Samples:
    try:
        return align_samples(
            inputs.rps,
            inputs.mean,
            unit=settings.latency_unit,
            load_model=settings.load_model,
            start=start,
            cutoff=cutoff,
            vus=inputs.vus,
            p95=inputs.p95,
            steps=steps,
            ceiling_end=ceiling_end,
        )
    except CapacityModelError as exc:
        # The first argument keeps a msg (rendered again where it is shown); a plain text stays as it is.
        raise ForecastError(exc.args[0] if exc.args else str(exc), 422, exc.code) from exc


def _sla_limit_rps(
    sla_ceiling: Optional[SlaCeiling],
    samples: Samples,
    steps: list[StepPoint],
    tested_rps: float,
) -> Optional[float]:
    """Load of the first step above the report maximum, never below the tested load; without load steps the
    level the report names for the step after the maximum. None when the window has no samples above it."""
    if sla_ceiling is None or not bool((samples.frame["role"] == ROLE_DEGRADED).any()):
        return None
    first = next((step for step in steps if step.degraded), None)
    rps = first.rps if first is not None else sla_ceiling.degraded_rps
    return None if rps is None else max(rps, tested_rps)


def _model_context(
    samples: Samples,
    fit: UslFit,
    coverage: LoadCoverage,
    instability: Optional[Instability],
    sla_rps: Optional[float] = None,
) -> ModelContext:
    return ModelContext(
        fit=fit,
        think_time_s=samples.think_time_s,
        p95_ratio=p95_ratio(samples),
        tested_concurrency=coverage.concurrency,
        limit=load_limit(fit, instability, sla_rps),
        unstable=instability is not None,
    )


def _curve_edge(
    model: ModelContext,
    target_rps: Optional[float],
    instability: Optional[Instability],
    sla_rps: Optional[float] = None,
) -> float:
    """Right edge of the model curve; it stops where the stalls began or above the report maximum."""
    target_n = concurrency_for(model.fit, target_rps) if target_rps else None
    observed = [value for value in (instability.onset_rps if instability is not None else None, sla_rps) if value is not None]
    return curve_upper(model.fit, model.tested_concurrency, target_n, min(observed) if observed else None)


def _resource_frames(run_id: str, settings: ForecastSettings) -> dict[tuple[str, str], pd.DataFrame]:
    """CPU series of the service instances and of the nodes, keyed by (domain, label)."""
    wanted = [(SERVICES_CPU_DOMAIN, settings.service_cpu_label), (NODES_CPU_DOMAIN, settings.node_cpu_label)]
    cfg = _storage()
    conn = _ts_conn()
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    try:
        for domain, label in wanted:
            if not label:
                continue
            loaded = load_run_frames(
                conn, str(cfg.get("schema") or "public"), str(cfg.get("table") or "metrics"), run_id, domain, [label],
            )
            if label in loaded:
                frames[(domain, label)] = loaded[label]
    finally:
        conn.close()
    return frames


def _node_lines(
    settings: ForecastSettings,
    frame: Optional[pd.DataFrame],
    loads: list[StepLoad],
) -> tuple[list[UtilizationLine], str]:
    if not settings.node_cpu_label:
        return [], msg("capacity.nodes.no_query")
    if frame is None or frame.empty:
        return [], msg("capacity.nodes.no_series", label=settings.node_cpu_label)
    return utilization_lines(frame, loads, NODE_CPU_SCALE), ""


def _resource_plan(
    run_id: str,
    settings: ForecastSettings,
    samples: Samples,
    steps: list[StepPoint],
    instability: Optional[Instability],
    limit: LoadLimit,
    rps: float,
    ceiling: CpuCeiling,
) -> ResourcePlan:
    """Instances per service for ``rps``; the status tells the page why the plan is missing."""
    label = settings.service_cpu_label
    if not label:
        return unavailable_plan("missing_setting", msg("capacity.plan.no_query", hint=msg("capacity.setting_hint")), ceiling, rps)
    frames = _resource_frames(run_id, settings)
    services = frames.get((SERVICES_CPU_DOMAIN, label))
    if services is None or services.empty:
        return unavailable_plan("missing_series", msg("capacity.plan.no_series", label=label), ceiling, rps)
    peak = float(np.nanmax(services.to_numpy(dtype=float)))
    if peak > MAX_CPU_FRACTION:
        return unavailable_plan("bad_unit", msg("capacity.plan.bad_unit", label=label, peak=f"{peak:.1f}"), ceiling, rps)
    loads = stable_step_loads(samples, steps, instability)
    lines = utilization_lines(services, loads, 1.0) if len(loads) >= MIN_PLAN_STEPS else []
    if not lines:
        message = msg("capacity.plan.too_few_steps", min=MIN_PLAN_STEPS, count=len(loads))
        return unavailable_plan("too_few_steps", message, ceiling, rps)
    nodes, nodes_message = _node_lines(settings, frames.get((NODES_CPU_DOMAIN, settings.node_cpu_label)), loads)
    return build_plan(lines, nodes, nodes_message, loads, rps, ceiling, limit)


def _assemble(
    report: StoredFinalReport,
    settings: ForecastSettings,
    samples: Samples,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    cutoff_kind: str,
    notes: list[str],
    target_rps: Optional[float],
    headroom_pct: float,
    ceiling: CpuCeiling,
    sla_ceiling: Optional[SlaCeiling] = None,
) -> ForecastReport:
    shift = _time_shift_hours()
    fit_rows = samples.fit_rows
    fit = fit_usl(fit_rows["n"].to_numpy(dtype=float), fit_rows["x"].to_numpy(dtype=float))
    instability = detect_instability(samples, shift)
    steps = step_points(samples, cutoff, shift)
    coverage = load_coverage(samples, steps)
    sla_rps = _sla_limit_rps(sla_ceiling, samples, steps, coverage.rps)
    model = _model_context(samples, fit, coverage, instability, sla_rps)
    n_low = float(fit_rows["n"].min())
    upper = _curve_edge(model, target_rps, instability, sla_rps)
    roles = samples.frame["role"]
    target = evaluate_target(model, target_rps, headroom_pct / 100.0) if target_rps else None
    plan = _resource_plan(report.run_id, settings, samples, steps, instability, model.limit, target_rps or coverage.rps, ceiling)
    safe = safe_capacity(plan, model.limit, headroom_pct / 100.0)
    return ForecastReport(
        run_id=report.run_id,
        run_name=report.run_name,
        service=report.service,
        load_model=settings.load_model,
        latency_unit=settings.latency_unit,
        window_start_iso=format_iso(start, shift),
        cutoff_iso=format_iso(cutoff, shift),
        cutoff_kind=cutoff_kind,
        bin_minutes=samples.bin_minutes,
        fit=fit,
        tested_concurrency=coverage.concurrency,
        tested_rps=coverage.rps,
        peak_rps=coverage.peak_rps,
        samples_used=int(len(fit_rows)),
        excluded_edges=int((roles == ROLE_EDGE).sum()),
        excluded_stalls=int((roles == ROLE_STALL).sum()),
        excluded_degraded=int((roles == ROLE_DEGRADED).sum()),
        think_time_s=samples.think_time_s,
        p95_ratio=model.p95_ratio,
        knee=knee_point(fit, samples.think_time_s, settings.elasticity_threshold, n_low, upper),
        instability=instability,
        limit=model.limit,
        limit_applies=limit_applies(plan, model.limit),
        safe=safe,
        scale=load_scale(safe, model.limit, coverage.rps, target_rps),
        target=target,
        answer=capacity_answer(target, plan, model.limit, coverage.rps) if target is not None else None,
        resources=plan,
        points=sample_points(samples, shift),
        steps=steps,
        curve=model_curve(fit, samples.think_time_s, model.p95_ratio, n_low, upper),
        sla_target_rps=_sla_threshold(report, "target_rps"),
        sla_p95_ms=_sla_threshold(report, "p95_latency"),
        sla_ceiling=replace(sla_ceiling, limit_rps=sla_rps) if sla_ceiling is not None else None,
        notes=notes,
    )


def _sla_threshold(report: StoredFinalReport, name: str) -> Optional[float]:
    """Positive threshold of the named SLA check: target_rps is the default goal, p95_latency a chart line."""
    checks = report.sla_details.get("checks") if isinstance(report.sla_details, dict) else None
    if not isinstance(checks, list):
        return None
    for check in checks:
        if not isinstance(check, dict) or check.get("name") != name:
            continue
        try:
            value = float(check.get("threshold"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    return None


@dataclass(frozen=True)
class ForecastAvailability:
    available: bool
    code: str
    message: str


def forecast_availability(test_type: str, series_labels: Optional[list[str]], sla: dict[str, Any]) -> ForecastAvailability:
    """Whether a stored report can be forecast: capacity test, forecast settings, required series."""
    if not profile_for(test_type).forecast_allowed:
        return ForecastAvailability(False, "not_applicable", t("capacity.error.not_applicable"))
    try:
        settings = _forecast_settings(sla)
    except ForecastError as exc:
        return ForecastAvailability(False, "missing_setting", str(exc))
    required = [settings.rps_label, settings.mean_label]
    if settings.load_model == "closed":
        required.append(settings.vus_label)
    if series_labels is None:
        return ForecastAvailability(False, "missing_series", t("capacity.availability.legacy"))
    for label in required:
        if label not in series_labels:
            return ForecastAvailability(False, "missing_series", t("capacity.availability.missing_series", label=label))
    return ForecastAvailability(True, "ok", "")


def _sla_cached(area: str, service: str, cache: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    key = (area, service)
    if key not in cache:
        cache[key] = _resolved_sla(_effective_config_for_scope(area, service or None))
    return cache[key]


@dataclass(frozen=True)
class ForecastListItem:
    run_id: str
    run_name: str
    service: str
    test_type: str
    created_at: Optional[str]
    verdict: str
    load_model: str


@dataclass(frozen=True)
class ForecastCatalog:
    reports: list[ForecastListItem]
    missing_settings: int
    missing_series: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_forecast_reports(services: list[str]) -> ForecastCatalog:
    """Reports of the area that can be forecast, plus counts of capacity reports that cannot."""
    cfg = _storage()
    conn = _ts_conn()
    try:
        rows = list_final_reports(
            conn, str(cfg.get("schema") or "public"), str(cfg.get("llm_table") or "llm_reports"), services,
        )
    finally:
        conn.close()
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    reports: list[ForecastListItem] = []
    missing_settings = 0
    missing_series = 0
    for row in rows:
        area = row.project_area or _find_area_for_service(row.service) or ""
        sla = _sla_cached(area, row.service, cache)
        status = forecast_availability(row.test_type, row.lt_series_labels, sla)
        if status.code == "not_applicable":
            continue
        if status.code == "missing_setting":
            missing_settings += 1
            continue
        if status.code == "missing_series":
            missing_series += 1
            continue
        reports.append(ForecastListItem(
            run_id=row.run_id,
            run_name=row.run_name,
            service=row.service,
            test_type=row.test_type,
            created_at=row.created_at.isoformat() if row.created_at else None,
            verdict=to_code(row.verdict),
            load_model=str(sla.get("load_model") or ""),
        ))
    return ForecastCatalog(reports=reports, missing_settings=missing_settings, missing_series=missing_series)


def _stored_labels(report: StoredFinalReport) -> Optional[list[str]]:
    raw = report.context.get("lt_series_labels") if isinstance(report.context, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    return [str(item) for item in raw]


def forecast_status(run_id: str) -> ForecastAvailability:
    """Availability of one report for the button on the report page."""
    report = _stored_report(run_id)
    return forecast_availability(report.test_type, _stored_labels(report), _scope_sla(report))


def database_unavailable(exc: psycopg2.Error) -> ForecastError:
    """Maps a database failure to the API error the page shows; the details stay in the server log."""
    error_id = log_failure(exc, "api.error.database_unavailable")
    return ForecastError(with_error_id("api.error.database_unavailable", error_id), 502, "database")
