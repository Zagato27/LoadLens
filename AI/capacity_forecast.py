"""Universal Scalability Law forecast from one load-test run.

Samples are the stored metric buckets up to the first RPS drop. Concurrency is
the virtual-user count for a closed workload and RPS times mean response time
for an open one. The leading ramp-up, the trailing ramp-down and response-time
stalls inside a load step stay out of the fit. The fit is a coarse grid search
on numpy, without scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from AI.context_pack import format_iso, to_utc
from AI.sla_evaluator import _bin_width

LOAD_MODELS = ("open", "closed")
LATENCY_UNIT_SECONDS = {"ms": 0.001, "s": 1.0}
MIN_SAMPLES = 10
LOW_LOAD_SHARE = 0.5
THINK_TIME_NOISE_SHARE = 0.05
MIN_R2 = 0.8
MEDIUM_EXTRAPOLATION = 1.5
CURVE_POINTS = 120
KNEE_GRID_POINTS = 400
CURVE_SPAN = 1.5
CURVE_MAX_SPAN = 5.0
SIGMA_GRID = np.linspace(0.0, 0.99, 100)
KAPPA_LOG_RANGE = (-6.0, 2.0)  # kappa * n_max**2 on a log scale
KAPPA_GRID_POINTS = 120
ZOOM_POINTS = 21
ZOOM_PASSES = 2
EDGE_RPS_SHARE = 0.8
EDGE_MAX_SAMPLES = 2
STALL_FACTOR = 3.0
STALL_QUANTILE = 0.25
UNSTABLE_STEP_STALLS = 2
UNSTABLE_STEP_SHARE = 0.2
PERIOD_MIN_RUNS = 3
PERIOD_TOLERANCE = 0.25
PERIOD_REGULAR_SHARE = 0.5

SampleRole = Literal["fit", "edge", "stall", "after_cutoff"]
TargetVerdict = Literal["holds", "no_headroom", "over_limit"]
ConfidenceLevel = Literal["high", "medium", "low"]
LimitKind = Literal["instability", "model", "none"]
ROLE_FIT: SampleRole = "fit"
ROLE_EDGE: SampleRole = "edge"
ROLE_STALL: SampleRole = "stall"
ROLE_AFTER_CUTOFF: SampleRole = "after_cutoff"


class CapacityModelError(ValueError):
    """The samples cannot support a forecast; ``code`` names the reason for the UI."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UslFit:
    lam: float
    sigma: float
    kappa: float
    r2: float
    n_peak: Optional[float]
    x_max: Optional[float]


@dataclass(frozen=True)
class StepWindow:
    """Report load step: the whole span for stall checks, the plateau for step averages."""

    number: int
    label: str
    source: str
    start: pd.Timestamp
    end: pd.Timestamp
    plateau_start: pd.Timestamp
    plateau_end: pd.Timestamp


@dataclass(frozen=True)
class Samples:
    """Aligned series. ``role`` keeps ramps, stalls and samples past the cutoff out of the fit;
    ``step`` is the position of the sample's load step in ``steps``, -1 outside every step."""

    frame: pd.DataFrame
    think_time_s: float
    bin_minutes: float
    steps: tuple[StepWindow, ...]

    @property
    def fit_rows(self) -> pd.DataFrame:
        return self.frame.loc[self.frame["role"] == ROLE_FIT]

    @property
    def checked_rows(self) -> pd.DataFrame:
        """Samples of the load window: fit samples and stalls."""
        return self.frame.loc[self.frame["role"].isin((ROLE_FIT, ROLE_STALL))]


@dataclass(frozen=True)
class SamplePoint:
    time_iso: str
    concurrency: float
    rps: float
    response_ms: float
    p95_ms: Optional[float]
    vus: Optional[float]
    role: SampleRole


@dataclass(frozen=True)
class StepPoint:
    number: int
    label: str
    source: str
    start_iso: str
    end_iso: str
    concurrency: float
    rps: float
    response_ms: float
    p95_ms: Optional[float]
    stalls: int
    samples: int
    after_drop: bool


@dataclass(frozen=True)
class CurvePoint:
    concurrency: float
    rps: float
    response_ms: float
    p95_ms: Optional[float]


@dataclass(frozen=True)
class KneePoint:
    concurrency: float
    rps: float
    response_ms: float


@dataclass(frozen=True)
class Instability:
    """Load step where response-time stalls became frequent, with the stalls of the whole window."""

    onset_iso: str
    onset_rps: float
    onset_step: int
    onset_label: str
    stalls: int
    samples: int
    worst_response_ms: float
    worst_p95_ms: Optional[float]
    period_min: Optional[float]


@dataclass(frozen=True)
class LoadLimit:
    """Load the forecast must not cross: the model ceiling or the onset of stalls."""

    rps: Optional[float]
    kind: LimitKind


@dataclass(frozen=True)
class LoadCoverage:
    concurrency: float
    rps: float
    peak_rps: float


@dataclass(frozen=True)
class ModelContext:
    fit: UslFit
    think_time_s: float
    p95_ratio: Optional[float]
    tested_concurrency: float
    limit: LoadLimit
    unstable: bool


@dataclass(frozen=True)
class TargetEstimate:
    rps: float
    verdict: TargetVerdict
    headroom_pct: Optional[float]
    concurrency: Optional[float]
    response_ms: Optional[float]
    p95_ms: Optional[float]
    confidence: ConfidenceLevel
    confidence_reasons: tuple[str, ...]


def usl_throughput(lam: float, sigma: float, kappa: float, n: np.ndarray) -> np.ndarray:
    """USL throughput X(N) = λN / (1 + σ(N-1) + κN(N-1))."""
    return lam * n / (1.0 + sigma * (n - 1.0) + kappa * n * (n - 1.0))


def _named(values: pd.Series, name: str) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    series.name = name
    return series


def _require_closed_vus(frame: pd.DataFrame) -> pd.DataFrame:
    if "vus" not in frame.columns:
        raise CapacityModelError("Для закрытой модели нужен ряд числа VU", "missing_series")
    kept = frame.dropna(subset=["vus"])
    kept = kept.copy()
    kept["n"] = kept["vus"]
    return kept


def step_windows(steps: Sequence[Any]) -> tuple[StepWindow, ...]:
    """Report load steps with parseable bounds; a step without a detector plateau averages its whole span."""
    windows: list[StepWindow] = []
    for position, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        span = _bounds(step, "start_iso", "end_iso")
        if span is None:
            continue
        plateau = _bounds(step, "plateau_start_iso", "plateau_end_iso") or span
        windows.append(StepWindow(
            number=position + 1,
            label=str(step.get("label") or ""),
            source=str(step.get("source") or ""),
            start=span[0],
            end=span[1],
            plateau_start=plateau[0],
            plateau_end=plateau[1],
        ))
    return tuple(windows)


def _bounds(step: dict[str, Any], start_key: str, end_key: str) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    start_raw, end_raw = step.get(start_key), step.get(end_key)
    if not start_raw or not end_raw:
        return None
    try:
        return to_utc(start_raw), to_utc(end_raw)
    except (TypeError, ValueError):
        return None


def align_samples(
    rps: pd.Series,
    mean_latency: pd.Series,
    *,
    unit: str,
    load_model: str,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    vus: Optional[pd.Series] = None,
    p95: Optional[pd.Series] = None,
    steps: Sequence[StepWindow] = (),
) -> Samples:
    """Joins the series on time, converts latency to seconds, computes concurrency and sample roles.

    A sample whose bucket ends after ``cutoff`` is ``after_cutoff``. The leading ramp-up and
    the trailing ramp-down are ``edge``; a response spike inside a load step is ``stall``.
    Runs without load steps have no stall check. Only ``fit`` samples feed the model.
    """
    factor = LATENCY_UNIT_SECONDS[unit]
    parts: dict[str, pd.Series] = {"x": _named(rps, "x"), "r_s": _named(mean_latency, "r_s") * factor}
    if vus is not None:
        parts["vus"] = _named(vus, "vus")
    if p95 is not None:
        parts["p95_s"] = _named(p95, "p95_s") * factor
    frame = pd.concat(parts, axis=1).sort_index()
    frame = frame.dropna(subset=["x", "r_s"])
    frame = frame[(frame["x"] > 0) & (frame["r_s"] > 0) & (frame.index >= start)]
    frame = _require_closed_vus(frame) if load_model == "closed" else frame.assign(n=frame["x"] * frame["r_s"])
    width = _bin_width(pd.DatetimeIndex(frame.index))
    frame = frame.assign(step=_step_positions(pd.DatetimeIndex(frame.index), steps))
    frame = frame.assign(role=_roles(frame, width, cutoff))
    fit = frame.loc[frame["role"] == ROLE_FIT]
    _check_coverage(fit, frame)
    think = _think_time(fit) if load_model == "closed" else 0.0
    return Samples(frame=frame, think_time_s=think, bin_minutes=width.total_seconds() / 60.0, steps=tuple(steps))


def _step_positions(index: pd.DatetimeIndex, steps: Sequence[StepWindow]) -> np.ndarray:
    positions = np.full(len(index), -1, dtype=int)
    for position, step in enumerate(steps):
        positions[(index >= step.start) & (index < step.end)] = position
    return positions


def _roles(frame: pd.DataFrame, width: pd.Timedelta, cutoff: pd.Timestamp) -> pd.Series:
    roles = pd.Series(ROLE_FIT, index=frame.index, dtype=object)
    inside = frame.loc[(frame.index + width) <= cutoff]
    roles.loc[~roles.index.isin(inside.index)] = ROLE_AFTER_CUTOFF
    edge = _edge_mask(inside["x"]).to_numpy()
    roles.loc[inside.index[edge]] = ROLE_EDGE
    steady = inside.loc[~edge]
    stall = _stall_mask(steady).to_numpy()
    roles.loc[steady.index[stall]] = ROLE_STALL
    return roles


def _edge_mask(rps: pd.Series) -> pd.Series:
    """Ramp-up at the start and ramp-down at the end: up to EDGE_MAX_SAMPLES samples
    carrying less than EDGE_RPS_SHARE of their inner neighbour's RPS."""
    values = rps.to_numpy(dtype=float)
    edge = np.zeros(len(values), dtype=bool)
    reach = min(EDGE_MAX_SAMPLES, len(values) - 1)
    for offset in range(reach):
        if values[offset] >= EDGE_RPS_SHARE * values[offset + 1]:
            break
        edge[offset] = True
    for offset in range(reach):
        last = len(values) - 1 - offset
        if values[last] >= EDGE_RPS_SHARE * values[last - 1]:
            break
        edge[last] = True
    return pd.Series(edge, index=rps.index)


def _stall_mask(frame: pd.DataFrame) -> pd.Series:
    """Response above STALL_FACTOR times the lower quartile of the sample's load step."""
    stepped = frame.loc[frame["step"] >= 0]
    if stepped.empty:
        return pd.Series(False, index=frame.index)
    reference = stepped.groupby("step")["r_s"].transform(lambda values: values.quantile(STALL_QUANTILE))
    stall = stepped["r_s"] > STALL_FACTOR * reference
    return stall.reindex(frame.index, fill_value=False).astype(bool)


def _check_coverage(fit: pd.DataFrame, frame: pd.DataFrame) -> None:
    count = int(len(fit))
    if count < MIN_SAMPLES:
        edges = int((frame["role"] == ROLE_EDGE).sum())
        stalls = int((frame["role"] == ROLE_STALL).sum())
        raise CapacityModelError(
            f"Для модели осталось {count} точек до первого падения RPS "
            f"(исключены разгон и спад нагрузки — {edges}, провалы — {stalls}), нужно не меньше {MIN_SAMPLES}",
            "too_few_points",
        )
    low, high = float(fit["n"].min()), float(fit["n"].max())
    if not (high > 0) or low > LOW_LOAD_SHARE * high:
        raise CapacityModelError(
            "Нет точек на низкой нагрузке: модели нужен разгон с малой нагрузки",
            "no_low_load",
        )


def _think_time(fit: pd.DataFrame) -> float:
    raw = float((fit["vus"] / fit["x"] - fit["r_s"]).median())
    response = float(fit["r_s"].median())
    if raw < -THINK_TIME_NOISE_SHARE * response:
        raise CapacityModelError(
            "Число VU меньше параллелизма по закону Литтла (RPS × время отклика): "
            "похоже, сценарий шлёт запросы параллельно. Выберите открытую модель нагрузки.",
            "negative_think_time",
        )
    return max(raw, 0.0)


def _grid_best(n: np.ndarray, x: np.ndarray, sigmas: np.ndarray, kappas: np.ndarray) -> tuple[float, float, float, float]:
    """(sse, lam, sigma, kappa) with the smallest squared error; lam solves the linear least squares."""
    best = (math.inf, 0.0, 0.0, 0.0)
    nn1 = n * (n - 1.0)
    for sigma in sigmas:
        denom = 1.0 + float(sigma) * (n - 1.0) + np.outer(kappas, nn1)
        valid = np.all(denom > 0, axis=1)
        shape = n / np.where(denom > 0, denom, 1.0)
        gg = np.einsum("ij,ij->i", shape, shape)
        lam = np.divide(shape @ x, gg, out=np.zeros_like(gg), where=gg > 0)
        sse = np.sum((x - lam[:, None] * shape) ** 2, axis=1)
        sse = np.where(valid & (gg > 0), sse, math.inf)
        index = int(np.argmin(sse))
        if float(sse[index]) < best[0]:
            best = (float(sse[index]), float(lam[index]), float(sigma), float(kappas[index]))
    return best


def _zoom_grid(
    best: tuple[float, float, float, float],
    sigma_step: float,
    kappa_factor: float,
    n_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma, kappa = best[2], best[3]
    sigmas = np.clip(np.linspace(sigma - sigma_step, sigma + sigma_step, ZOOM_POINTS), 0.0, 0.99)
    center = kappa if kappa > 0 else 1e-6 / n_max ** 2
    kappas = np.geomspace(max(center / kappa_factor, 1e-18), center * kappa_factor, ZOOM_POINTS)
    if kappa == 0.0:
        kappas = np.concatenate(([0.0], kappas))
    return sigmas, kappas


def _ceiling(lam: float, sigma: float, kappa: float) -> tuple[Optional[float], Optional[float]]:
    if kappa > 0 and sigma < 1.0:
        n_peak = math.sqrt((1.0 - sigma) / kappa)
        x_max = float(usl_throughput(lam, sigma, kappa, np.array([n_peak]))[0])
        return n_peak, x_max
    if sigma > 0:
        return None, lam / sigma
    return None, None


def fit_usl(concurrency: np.ndarray, throughput: np.ndarray) -> UslFit:
    """Coarse σ×κ grid, then two zooms around the best cell; κ grid is scaled by n_max**2."""
    n_max = float(np.max(concurrency))
    if n_max <= 0:
        raise CapacityModelError("Параллелизм на всех точках нулевой", "no_low_load")
    span = (KAPPA_LOG_RANGE[1] - KAPPA_LOG_RANGE[0]) / (KAPPA_GRID_POINTS - 1)
    kappas = np.concatenate(([0.0], np.logspace(*KAPPA_LOG_RANGE, KAPPA_GRID_POINTS) / n_max ** 2))
    best = _grid_best(concurrency, throughput, SIGMA_GRID, kappas)
    sigma_step = float(SIGMA_GRID[1] - SIGMA_GRID[0])
    kappa_factor = 10 ** span
    for _ in range(ZOOM_PASSES):
        sigmas, zoom_kappas = _zoom_grid(best, sigma_step, kappa_factor, n_max)
        best = min(best, _grid_best(concurrency, throughput, sigmas, zoom_kappas))
        sigma_step, kappa_factor = sigma_step / 5.0, kappa_factor ** 0.2
    sse, lam, sigma, kappa = best
    sst = float(np.sum((throughput - throughput.mean()) ** 2))
    n_peak, x_max = _ceiling(lam, sigma, kappa)
    return UslFit(lam=lam, sigma=sigma, kappa=kappa, r2=1.0 - sse / sst if sst > 0 else 0.0, n_peak=n_peak, x_max=x_max)


def concurrency_for(fit: UslFit, x_target: float) -> Optional[float]:
    """Concurrency on the rising branch that delivers ``x_target``; None when it is out of reach."""
    a = fit.kappa * x_target
    b = x_target * (fit.sigma - fit.kappa) - fit.lam
    c = x_target * (1.0 - fit.sigma)
    if a == 0.0:
        return c / -b if b < 0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0 or b >= 0:
        return None
    return (-b - math.sqrt(disc)) / (2.0 * a)


def p95_ratio(samples: Samples) -> Optional[float]:
    """Median of p95 / mean on the fit samples."""
    if "p95_s" not in samples.frame.columns:
        return None
    fit = samples.fit_rows
    ratio = (fit["p95_s"] / fit["r_s"]).replace([np.inf, -np.inf], np.nan).dropna()
    positive = ratio[ratio > 0]
    if len(positive) < MIN_SAMPLES:
        return None
    return float(positive.median())


def _response_seconds(fit: UslFit, think_time_s: float, n: np.ndarray, throughput: np.ndarray) -> np.ndarray:
    return n / np.maximum(throughput, 1e-12) - think_time_s


def knee_point(
    fit: UslFit,
    think_time_s: float,
    threshold: float,
    n_lo: float,
    n_hi: float,
) -> Optional[KneePoint]:
    """First point where throughput stops growing or response time outruns it."""
    if n_hi <= n_lo:
        return None
    grid = np.linspace(n_lo, n_hi, KNEE_GRID_POINTS)
    throughput = usl_throughput(fit.lam, fit.sigma, fit.kappa, grid)
    response = _response_seconds(fit, think_time_s, grid, throughput)
    for index in range(1, len(grid)):
        previous_r, current_r = float(response[index - 1]), float(response[index])
        previous_x = float(throughput[index - 1])
        if previous_r <= 0 or current_r <= 0 or previous_x <= 0:
            continue
        growth_x = float(throughput[index]) - previous_x
        if growth_x <= 0 or (current_r - previous_r) / previous_r >= threshold * growth_x / previous_x:
            return KneePoint(float(grid[index]), float(throughput[index]), current_r * 1000.0)
    return None


def curve_upper(fit: UslFit, tested_n: float, target_n: Optional[float], stop_rps: Optional[float] = None) -> float:
    """Right edge of the chart: past the tested range, the target and the peak, capped at 5x and at ``stop_rps``."""
    edges = [CURVE_SPAN * tested_n]
    if target_n is not None:
        edges.append(1.1 * target_n)
    if fit.n_peak is not None:
        edges.append(1.2 * fit.n_peak)
    upper = min(max(edges), CURVE_MAX_SPAN * tested_n)
    stop_n = concurrency_for(fit, stop_rps) if stop_rps is not None else None
    return upper if stop_n is None else min(upper, stop_n)


def model_curve(
    fit: UslFit,
    think_time_s: float,
    ratio: Optional[float],
    n_lo: float,
    n_hi: float,
) -> list[CurvePoint]:
    grid = np.linspace(max(n_lo, 1e-6), max(n_hi, n_lo + 1e-6), CURVE_POINTS)
    throughput = usl_throughput(fit.lam, fit.sigma, fit.kappa, grid)
    response = _response_seconds(fit, think_time_s, grid, throughput)
    points: list[CurvePoint] = []
    for concurrency, rps, seconds in zip(grid, throughput, response):
        millis = max(float(seconds), 0.0) * 1000.0
        p95 = millis * ratio if ratio is not None and seconds > 0 else None
        points.append(CurvePoint(float(concurrency), float(rps), millis, p95))
    return points


def load_limit(fit: UslFit, instability: Optional[Instability]) -> LoadLimit:
    """The lower of the model ceiling and the load where stalls became frequent."""
    limits: list[tuple[float, LimitKind]] = []
    if fit.x_max is not None:
        limits.append((fit.x_max, "model"))
    if instability is not None:
        limits.append((instability.onset_rps, "instability"))
    if not limits:
        return LoadLimit(rps=None, kind="none")
    rps, kind = min(limits, key=lambda item: item[0])
    return LoadLimit(rps=rps, kind=kind)


def _confidence(model: ModelContext, concurrency: Optional[float]) -> tuple[ConfidenceLevel, tuple[str, ...]]:
    """Confidence level with the reasons that lower it: weak fit, stalls, extrapolation."""
    reasons: list[tuple[str, ConfidenceLevel]] = []
    if model.fit.r2 < MIN_R2:
        reasons.append(("low_r2", "low"))
    if model.unstable:
        reasons.append(("instability", "medium"))
    span = concurrency / model.tested_concurrency if concurrency is not None and model.tested_concurrency > 0 else 0.0
    if span > 1.0:
        reasons.append(("extrapolation", "medium" if span <= MEDIUM_EXTRAPOLATION else "low"))
    levels = {level for _, level in reasons}
    level: ConfidenceLevel = "low" if "low" in levels else "medium" if "medium" in levels else "high"
    return level, tuple(code for code, _ in reasons)


def evaluate_target(model: ModelContext, x_target: float, headroom_share: float) -> TargetEstimate:
    """Whether the system holds ``x_target`` and keeps ``headroom_share`` of the limit in reserve."""
    limit = model.limit.rps
    concurrency = concurrency_for(model.fit, x_target)
    headroom = None if limit is None else (limit - x_target) / limit
    if concurrency is None or concurrency <= 0 or (headroom is not None and headroom <= 0):
        level, reasons = _confidence(model, None)
        return TargetEstimate(x_target, "over_limit", None, None, None, None, level, reasons)
    response_s = concurrency / x_target - model.think_time_s
    level, reasons = _confidence(model, concurrency)
    return TargetEstimate(
        rps=x_target,
        verdict="holds" if headroom is None or headroom >= headroom_share else "no_headroom",
        headroom_pct=None if headroom is None else headroom * 100.0,
        concurrency=concurrency,
        response_ms=response_s * 1000.0,
        p95_ms=response_s * model.p95_ratio * 1000.0 if model.p95_ratio is not None else None,
        confidence=level,
        confidence_reasons=reasons,
    )


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _worst_p95_ms(rows: pd.DataFrame) -> Optional[float]:
    if "p95_s" not in rows.columns:
        return None
    worst = _finite(rows["p95_s"].max())
    return None if worst is None else worst * 1000.0


def sample_points(samples: Samples, shift_hours: int) -> list[SamplePoint]:
    """One chart point per aligned sample. Non-finite numbers become ``None``."""
    frame = samples.frame
    points: list[SamplePoint] = []
    for stamp, row in frame.iterrows():
        p95 = _finite(row["p95_s"]) if "p95_s" in frame.columns else None
        vus = _finite(row["vus"]) if "vus" in frame.columns else None
        points.append(SamplePoint(
            time_iso=format_iso(stamp, shift_hours),
            concurrency=_finite(row["n"]) or 0.0,
            rps=_finite(row["x"]) or 0.0,
            response_ms=(_finite(row["r_s"]) or 0.0) * 1000.0,
            p95_ms=None if p95 is None else p95 * 1000.0,
            vus=vus,
            role=row["role"],
        ))
    return points


def step_points(samples: Samples, cutoff: pd.Timestamp, shift_hours: int) -> list[StepPoint]:
    """Plateau averages of each load step over its fit samples and stalls, for chart markers and facts."""
    checked = samples.checked_rows
    points: list[StepPoint] = []
    for step in samples.steps:
        chunk = checked.loc[(checked.index >= step.plateau_start) & (checked.index < step.plateau_end)]
        if chunk.empty:
            continue
        points.append(StepPoint(
            number=step.number,
            label=step.label,
            source=step.source,
            start_iso=format_iso(step.start, shift_hours),
            end_iso=format_iso(step.end, shift_hours),
            concurrency=float(chunk["n"].mean()),
            rps=float(chunk["x"].mean()),
            response_ms=float(chunk["r_s"].mean()) * 1000.0,
            p95_ms=_worst_p95_ms(chunk),
            stalls=int((chunk["role"] == ROLE_STALL).sum()),
            samples=int(len(chunk)),
            after_drop=step.plateau_end > cutoff,
        ))
    return points


def load_coverage(samples: Samples, steps: Sequence[StepPoint]) -> LoadCoverage:
    """Tested range: the highest step median concurrency of fit samples and the best step mean RPS.

    A run without load steps uses its highest fit sample and its best sample instead.
    """
    fit = samples.fit_rows
    stepped = fit.loc[fit["step"] >= 0]
    concurrency = float(stepped.groupby("step")["n"].median().max()) if not stepped.empty else float(fit["n"].max())
    peak = float(samples.checked_rows["x"].max())
    rps = max((step.rps for step in steps if not step.after_drop), default=peak)
    return LoadCoverage(concurrency=concurrency, rps=rps, peak_rps=peak)


def detect_instability(samples: Samples, shift_hours: int) -> Optional[Instability]:
    """First load step with frequent stalls. Its onset load is the median RPS of the step's
    normal samples, or of all its samples when every one of them stalled."""
    checked = samples.checked_rows
    position = _first_unstable_step(checked)
    if position is None:
        return None
    step_rows = checked.loc[checked["step"] == position]
    normal = step_rows.loc[step_rows["role"] == ROLE_FIT]
    stalls = checked.loc[checked["role"] == ROLE_STALL]
    step = samples.steps[position]
    return Instability(
        onset_iso=format_iso(step_rows.loc[step_rows["role"] == ROLE_STALL].index.min(), shift_hours),
        onset_rps=float((normal if not normal.empty else step_rows)["x"].median()),
        onset_step=step.number,
        onset_label=step.label,
        stalls=int(len(stalls)),
        samples=int(len(checked)),
        worst_response_ms=float(stalls["r_s"].max()) * 1000.0,
        worst_p95_ms=_worst_p95_ms(stalls),
        period_min=_stall_period(checked),
    )


def _first_unstable_step(checked: pd.DataFrame) -> Optional[int]:
    """Position of the first step with UNSTABLE_STEP_STALLS stalls making UNSTABLE_STEP_SHARE of its samples."""
    stepped = checked.loc[checked["step"] >= 0]
    if stepped.empty:
        return None
    counts = (stepped["role"] == ROLE_STALL).groupby(stepped["step"]).agg(["sum", "size"])
    frequent = counts[(counts["sum"] >= UNSTABLE_STEP_STALLS) & (counts["sum"] >= UNSTABLE_STEP_SHARE * counts["size"])]
    return int(frequent.index.min()) if not frequent.empty else None


def _stall_period(checked: pd.DataFrame) -> Optional[float]:
    """Median minutes between the starts of stall runs, when most gaps stay close to it."""
    is_stall = (checked["role"] == ROLE_STALL).to_numpy()
    run_start = is_stall & ~np.concatenate(([False], is_stall))[:-1]
    starts = pd.DatetimeIndex(checked.index[run_start])
    if len(starts) < PERIOD_MIN_RUNS:
        return None
    gaps = np.diff(starts.asi8) / 60e9
    median = float(np.median(gaps))
    regular = np.abs(gaps - median) <= PERIOD_TOLERANCE * median
    return median if float(regular.mean()) >= PERIOD_REGULAR_SHARE else None


__all__ = [
    "CURVE_MAX_SPAN",
    "CURVE_SPAN",
    "CapacityModelError",
    "CurvePoint",
    "Instability",
    "KneePoint",
    "LATENCY_UNIT_SECONDS",
    "LOAD_MODELS",
    "LoadCoverage",
    "LoadLimit",
    "MIN_R2",
    "MIN_SAMPLES",
    "ModelContext",
    "ROLE_AFTER_CUTOFF",
    "ROLE_EDGE",
    "ROLE_FIT",
    "ROLE_STALL",
    "SamplePoint",
    "Samples",
    "StepPoint",
    "StepWindow",
    "TargetEstimate",
    "UslFit",
    "align_samples",
    "concurrency_for",
    "curve_upper",
    "detect_instability",
    "evaluate_target",
    "fit_usl",
    "knee_point",
    "load_coverage",
    "load_limit",
    "model_curve",
    "p95_ratio",
    "sample_points",
    "step_points",
    "step_windows",
    "usl_throughput",
]
