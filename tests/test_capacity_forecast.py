"""USL fit, sample alignment and the forecast HTTP API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from AI.capacity_forecast import (
    CapacityModelError,
    LoadLimit,
    ModelContext,
    TargetEstimate,
    align_samples,
    concurrency_for,
    detect_instability,
    evaluate_target,
    fit_usl,
    load_coverage,
    load_limit,
    step_points,
    step_windows,
    usl_throughput,
)
from AI.resource_plan import CpuCeiling, build_plan, capacity_answer, safe_capacity, stable_step_loads, utilization_lines


def _usl_series(noise: float = 0.0, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    concurrency = np.linspace(1.0, 100.0, 100)
    clean = usl_throughput(10.0, 0.01, 0.0002, concurrency)
    if noise:
        clean = clean * (1.0 + np.random.default_rng(seed).normal(0.0, noise, len(clean)))
    return concurrency, clean


def test_fit_recovers_known_usl():
    concurrency, throughput = _usl_series(noise=0.01)
    fit = fit_usl(concurrency, throughput)
    assert fit.sigma == pytest.approx(0.01, abs=0.01)
    assert fit.kappa == pytest.approx(0.0002, rel=0.3)
    assert fit.x_max == pytest.approx(263.6, rel=0.03)
    assert fit.r2 > 0.98


def test_concurrency_for_stays_on_the_rising_branch():
    fit = fit_usl(*_usl_series())
    concurrency = concurrency_for(fit, 200.0)
    assert concurrency is not None and concurrency < (fit.n_peak or 0)
    assert float(usl_throughput(fit.lam, fit.sigma, fit.kappa, np.array([concurrency]))[0]) == pytest.approx(200.0, rel=0.02)
    assert concurrency_for(fit, (fit.x_max or 0) + 50.0) is None


def test_linear_system_has_no_ceiling():
    concurrency = np.linspace(1.0, 40.0, 40)
    fit = fit_usl(concurrency, 5.0 * concurrency)
    assert fit.x_max is None


def test_open_model_uses_little_law_until_cutoff():
    index = pd.date_range("2026-10-01T10:00:00Z", periods=30, freq="1min")
    rps = pd.Series(np.linspace(10.0, 200.0, 30), index=index)
    latency = pd.Series([100.0] * 30, index=index)
    start, cutoff = index[0], index[0] + pd.Timedelta(minutes=20)
    samples = align_samples(rps, latency, unit="ms", load_model="open", start=start, cutoff=cutoff)
    roles = samples.frame["role"]
    assert int((roles == "after_cutoff").sum()) == 10
    assert list(roles.iloc[:2]) == ["edge", "edge"]
    fit = samples.fit_rows
    assert len(fit) == 18
    assert fit["n"].iloc[0] == pytest.approx(float(fit["x"].iloc[0]) * 0.1)


def _unstable_run():
    index = pd.date_range("2026-10-01T10:00:00Z", periods=60, freq="1min")
    rps = np.repeat([100.0, 200.0, 300.0, 400.0, 500.0, 600.0], 10)
    latency = np.full(60, 30.0)
    latency[40:] = np.where(np.arange(20) % 2 == 1, 300.0, 30.0)
    rps[0], latency[0] = 20.0, 1000.0
    rps[-1] = 150.0
    steps = step_windows([
        {
            "label": f"ступень {number + 1}",
            "source": "step_detector",
            "start_iso": index[number * 10].isoformat(),
            "end_iso": (index[number * 10] + pd.Timedelta(minutes=10)).isoformat(),
        }
        for number in range(6)
    ])
    cutoff = index[-1] + pd.Timedelta(minutes=1)
    samples = align_samples(
        pd.Series(rps, index=index), pd.Series(latency, index=index),
        unit="ms", load_model="open", start=index[0], cutoff=cutoff, steps=steps,
    )
    return samples, cutoff


def test_ramps_and_stalls_stay_out_of_the_fit():
    samples, cutoff = _unstable_run()
    roles = samples.frame["role"]
    assert list(roles.iloc[[0, -1]]) == ["edge", "edge"]
    assert int((roles == "stall").sum()) == 9
    instability = detect_instability(samples, 0)
    assert instability is not None
    assert (instability.onset_step, instability.onset_rps, instability.period_min) == (5, 500.0, 2.0)
    fit = fit_usl(samples.fit_rows["n"].to_numpy(), samples.fit_rows["x"].to_numpy())
    limit = load_limit(fit, instability)
    assert (limit.kind, limit.rps) == ("instability", 500.0)
    coverage = load_coverage(samples, step_points(samples, cutoff, 0))
    model = ModelContext(fit, 0.0, None, coverage.concurrency, limit, True)
    held = evaluate_target(model, 300.0, 0.2)
    assert held.verdict == "holds" and "instability" in held.confidence_reasons
    assert evaluate_target(model, 450.0, 0.2).verdict == "no_headroom"
    assert evaluate_target(model, 600.0, 0.2).verdict == "over_limit"


def _over_limit(rps: float) -> TargetEstimate:
    return TargetEstimate(rps, "over_limit", None, None, None, None, "medium", ("instability",))


def test_cpu_plan_scales_the_service_that_explains_the_limit():
    samples, cutoff = _unstable_run()
    index = samples.frame.index
    load = np.repeat([100.0, 200.0, 300.0, 400.0, 500.0, 600.0], 10)
    cpu = pd.DataFrame({
        "application=api|instance=a": 0.1 + 0.0016 * load,
        "application=db|instance=b": np.full(60, 0.2),
        "application=db|instance=c": np.full(60, 0.2),
    }, index=index)
    steps = stable_step_loads(samples, step_points(samples, cutoff, 0), detect_instability(samples, 0))
    assert [step.number for step in steps] == [1, 2, 3, 4]
    lines = utilization_lines(cpu, steps, 1.0)
    limit = LoadLimit(500.0, "instability")
    plan = build_plan(lines, [], "", steps, 600.0, CpuCeiling(80.0, "sla"), limit)
    api = plan.services[0]
    assert (plan.limit_cause, api.name, api.instances, api.instances_needed) == ("api", "api", 1, 2)
    assert api.cpu_after == pytest.approx(0.58)
    assert (plan.capacity_rps, plan.scaled_capacity_rps) == (pytest.approx(437.5), pytest.approx(875.0))
    answer = capacity_answer(_over_limit(600.0), plan, limit, 400.0)
    assert answer.verdict == "scale" and answer.confidence_reasons == ("scaling_untested",)
    safe = safe_capacity(plan, limit, 0.2)
    assert (safe.source, safe.service, safe.rps) == ("cpu", "api", pytest.approx(437.5))
    lower = LoadLimit(300.0, "instability")
    unexplained = build_plan(lines, [], "", steps, 350.0, CpuCeiling(80.0, "sla"), lower)
    assert unexplained.limit_cause is None
    assert capacity_answer(_over_limit(350.0), unexplained, lower, 400.0).verdict == "blocked"


def test_closed_model_rejects_parallel_requests():
    index = pd.date_range("2026-10-01T10:00:00Z", periods=30, freq="1min")
    rps = pd.Series(np.linspace(10.0, 200.0, 30), index=index)
    latency = pd.Series([0.1] * 30, index=index)
    vus = rps * 0.05
    with pytest.raises(CapacityModelError) as exc:
        align_samples(rps, latency, unit="s", load_model="closed", start=index[0], cutoff=index[-1] + pd.Timedelta(minutes=1), vus=vus)
    assert exc.value.code == "negative_think_time"
