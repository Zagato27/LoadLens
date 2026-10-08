"""Capacity forecast for one stored report: samples, USL fit and the calculator answer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg2

from AI.capacity_forecast import (
    LOAD_MODELS,
    LATENCY_UNIT_SECONDS,
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
from AI.pipeline import _cfg_float, _designated_rps_series, _test_profile_from_type, _time_shift_hours
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
from AI.sla_evaluator import _safe_float, _worst_series_p95
from loadlens_app.core import _find_area_for_service, _ts_conn
from settings import CONFIG
from update_page import _effective_config_for_scope

LT_DOMAIN = "lt_framework"
SERVICES_CPU_DOMAIN = "jvm"
NODES_CPU_DOMAIN = "hard_resources"
NODE_CPU_SCALE = 0.01  # sla.cpu_query is node CPU in percent
DEFAULT_CPU_CEILING_PCT = 80.0
SETTING_HINT = "SLA-критерии → Определение максимальной производительности"


class ForecastError(Exception):
    """Forecast is unavailable; ``code`` lets the page tell setup problems from inapplicable tests."""

    def __init__(self, message: str, status_code: int, code: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


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
    if _test_profile_from_type(report.test_type).get("mode") != "capacity":
        raise ForecastError("Прогноз строится только для тестов на максимальную производительность", 422, "not_applicable")
    settings = _forecast_settings(_scope_sla(report))
    start, cutoff, cutoff_kind = _window(report)
    inputs = _input_series(_stored_frames(run_id, settings), settings, start, cutoff)
    samples = _samples(inputs, settings, start, cutoff, _report_steps(report))
    ceiling = _cpu_ceiling(cpu_ceiling_pct, settings)
    return _assemble(report, settings, samples, start, cutoff, cutoff_kind, inputs.notes, target_rps, headroom_pct, ceiling)


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
        raise ForecastError("Отчёт не найден", 404, "not_found")
    if report.start_ms is None or report.end_ms is None:
        raise ForecastError("У отчёта нет интервала теста", 422, "missing_window")
    return report


def _scope_sla(report: StoredFinalReport) -> dict[str, Any]:
    area = report.project_area or _find_area_for_service(report.service) or ""
    sla = _effective_config_for_scope(area, report.service or None).get("sla") or {}
    return sla if isinstance(sla, dict) else {}


def _missing_setting(title: str) -> ForecastError:
    return ForecastError(f"Не задан {title}: {SETTING_HINT}", 422, "missing_setting")


def _required_text(sla: dict[str, Any], key: str, title: str) -> str:
    value = str(sla.get(key) or "").strip()
    if not value:
        raise _missing_setting(title)
    return value


def _forecast_settings(sla: dict[str, Any]) -> ForecastSettings:
    model = str(sla.get("load_model") or "").strip()
    unit = str(sla.get("latency_unit") or "").strip()
    if model not in LOAD_MODELS:
        raise _missing_setting("параметр «модель нагрузки»")
    if unit not in LATENCY_UNIT_SECONDS:
        raise _missing_setting("параметр «единицы времени отклика»")
    vus_label = str(sla.get("vus_query") or "").strip()
    if model == "closed" and not vus_label:
        raise _missing_setting("запрос числа VU")
    return ForecastSettings(
        rps_label=_required_text(sla, "max_performance_query", "запрос производительности"),
        mean_label=_required_text(sla, "mean_latency_query", "запрос среднего времени отклика"),
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
        raise ForecastError(
            f"В данных отчёта нет ряда «{label}». Добавьте запрос в «Нагрузочный инструмент» и перегенерируйте отчёт",
            422,
            "missing_series",
        )
    if len(list(frame.columns)) != 1:
        raise ForecastError(f"Запрос «{label}» вернул несколько серий. Для прогноза нужна одна серия", 422, "multiple_series")
    return frame.iloc[:, 0]


def _rps_series(frames: dict[str, pd.DataFrame], label: str) -> pd.Series:
    frame = frames.get(label)
    series = _designated_rps_series([{"label": label, "df": frame}], label) if frame is not None else None
    if series is None or series.dropna().empty:
        raise ForecastError(
            f"В данных отчёта нет ряда «{label}». Добавьте запрос в «Нагрузочный инструмент» и перегенерируйте отчёт",
            422,
            "missing_series",
        )
    return series


def _optional_vus(frames: dict[str, pd.DataFrame], label: str) -> tuple[Optional[pd.Series], list[str]]:
    if not label:
        return None, []
    frame = frames.get(label)
    if frame is None or frame.empty:
        return None, [f"Ряд VU «{label}» не найден — на графике только расчётный параллелизм"]
    if len(list(frame.columns)) != 1:
        return None, [f"Запрос VU «{label}» вернул несколько серий — он не используется"]
    return frame.iloc[:, 0], []


def _p95_series(
    frames: dict[str, pd.DataFrame],
    label: str,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[Optional[pd.Series], list[str]]:
    if not label:
        return None, ["p95 не назначен в SLA — прогноз только для среднего времени отклика"]
    frame = frames.get(label)
    if frame is None or frame.empty:
        return None, [f"Ряд p95 «{label}» не найден"]
    name = _worst_series_p95(frame, start, cutoff)[1]
    if not name or name not in frame.columns:
        return None, [f"На участке до падения RPS нет точек запроса «{label}»"]
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
        return p95 / 1000.0, ["p95 в отчёте в миллисекундах, для прогноза переведён в секунды"]
    if unit == "ms" and 0 < ratio <= 0.02:
        return p95 * 1000.0, ["p95 в отчёте в секундах, для прогноза переведён в миллисекунды"]
    return p95, []


def _report_steps(report: StoredFinalReport) -> tuple[StepWindow, ...]:
    steps = report.context.get("load_steps") if isinstance(report.context, dict) else None
    return step_windows(steps if isinstance(steps, list) else [])


def _samples(
    inputs: InputSeries,
    settings: ForecastSettings,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    steps: tuple[StepWindow, ...],
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
        )
    except CapacityModelError as exc:
        raise ForecastError(str(exc), 422, exc.code) from exc


def _model_context(samples: Samples, fit: UslFit, coverage: LoadCoverage, instability: Optional[Instability]) -> ModelContext:
    return ModelContext(
        fit=fit,
        think_time_s=samples.think_time_s,
        p95_ratio=p95_ratio(samples),
        tested_concurrency=coverage.concurrency,
        limit=load_limit(fit, instability),
        unstable=instability is not None,
    )


def _curve_edge(model: ModelContext, target_rps: Optional[float], instability: Optional[Instability]) -> float:
    """Right edge of the model curve; it stops where the stalls began."""
    target_n = concurrency_for(model.fit, target_rps) if target_rps else None
    stop_rps = instability.onset_rps if instability is not None else None
    return curve_upper(model.fit, model.tested_concurrency, target_n, stop_rps)


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
        return [], "Узлы не проверены: в SLA не задан запрос CPU узлов"
    if frame is None or frame.empty:
        return [], f"Узлы не проверены: в данных отчёта нет ряда «{settings.node_cpu_label}»"
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
        return unavailable_plan("missing_setting", f"Не задан запрос CPU экземпляров сервисов: {SETTING_HINT}", ceiling, rps)
    frames = _resource_frames(run_id, settings)
    services = frames.get((SERVICES_CPU_DOMAIN, label))
    if services is None or services.empty:
        message = f"В данных отчёта нет ряда «{label}» домена JVM: добавьте запрос и перегенерируйте отчёт"
        return unavailable_plan("missing_series", message, ceiling, rps)
    peak = float(np.nanmax(services.to_numpy(dtype=float)))
    if peak > MAX_CPU_FRACTION:
        message = f"Запрос «{label}» возвращает значения до {peak:.1f}: нужна загрузка CPU экземпляра в долях от 0 до 1, как у process_cpu_usage"
        return unavailable_plan("bad_unit", message, ceiling, rps)
    loads = stable_step_loads(samples, steps, instability)
    lines = utilization_lines(services, loads, 1.0) if len(loads) >= MIN_PLAN_STEPS else []
    if not lines:
        message = f"Для расчёта подов нужно не меньше {MIN_PLAN_STEPS} ступеней без провалов с данными CPU, в отчёте {len(loads)}"
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
) -> ForecastReport:
    shift = _time_shift_hours()
    fit_rows = samples.fit_rows
    fit = fit_usl(fit_rows["n"].to_numpy(dtype=float), fit_rows["x"].to_numpy(dtype=float))
    instability = detect_instability(samples, shift)
    steps = step_points(samples, cutoff, shift)
    coverage = load_coverage(samples, steps)
    model = _model_context(samples, fit, coverage, instability)
    n_low = float(fit_rows["n"].min())
    upper = _curve_edge(model, target_rps, instability)
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
    if _test_profile_from_type(test_type).get("mode") != "capacity":
        return ForecastAvailability(False, "not_applicable", "Прогноз строится только для тестов на максимальную производительность")
    try:
        settings = _forecast_settings(sla)
    except ForecastError as exc:
        return ForecastAvailability(False, "missing_setting", str(exc))
    required = [settings.rps_label, settings.mean_label]
    if settings.load_model == "closed":
        required.append(settings.vus_label)
    if series_labels is None:
        return ForecastAvailability(
            False, "missing_series",
            "Отчёт создан до появления прогноза. Добавьте запросы и перегенерируйте отчёт",
        )
    for label in required:
        if label not in series_labels:
            return ForecastAvailability(False, "missing_series", f"В данных отчёта нет ряда «{label}»")
    return ForecastAvailability(True, "ok", "")


def _sla_cached(area: str, service: str, cache: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    key = (area, service)
    if key not in cache:
        sla = _effective_config_for_scope(area, service or None).get("sla") or {}
        cache[key] = sla if isinstance(sla, dict) else {}
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
            verdict=row.verdict,
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
    """Maps a database failure to the API error the page shows."""
    return ForecastError(f"База данных недоступна: {exc}", 502, "database")
