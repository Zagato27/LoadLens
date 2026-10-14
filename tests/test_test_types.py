"""Soak, spike and stress analysis: deterministic metrics, SLA policy, rendering and the pipeline."""

import json
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
from AI.context_pack import equal_time_buckets  # noqa: E402
from AI.pipeline import _reconcile_sla_for_test_profile, _sla_step_context  # noqa: E402
from AI.report_sections import test_type_section  # noqa: E402
from AI.saturation import build_saturation, failure_mode, limiting_resource  # noqa: E402
from AI.sla_evaluator import _verdict_for_mode, evaluate_sla  # noqa: E402
from AI.spike import analyze_spikes, build_spike_payload, detect_spikes, spike_load_steps  # noqa: E402
from AI.stability import compute_trends, hold_check, hold_window_from_series  # noqa: E402
from confluence_manager.update_confluence_template import render_llm_markdown  # noqa: E402
from loadlens_app.confluence_export import render_llm_domain_body  # noqa: E402

START_TS = 1704099600.0  # 2024-01-01T09:00:00Z
END_TS = START_TS + 3600
INDEX = pd.date_range("2024-01-01T09:00:00Z", periods=60, freq="1min")
RPS_LABEL = "LT RPS"
P95_LABEL = "LT p95"
NOISE = np.random.default_rng(7).normal(0, 1.0, 60)


def _soak_frames(p95_end=170.0, rps=None):
    rps = rps if rps is not None else np.concatenate([np.linspace(0, 300, 5), np.full(55, 300.0)]) + NOISE
    p95 = np.linspace(100, p95_end, 60) + NOISE
    return pd.Series(rps, index=INDEX), pd.Series(p95, index=INDEX)


def _spike_frames(recover=True):
    rps = np.full(60, 100.0)
    rps[20:25] = 400.0
    p95 = np.full(60, 100.0)
    p95[20:25] = 400.0
    p95[25:27] = 300.0
    if not recover:
        p95[25:] = 300.0
    return pd.Series(rps + NOISE, index=INDEX), pd.Series(p95 + NOISE, index=INDEX)


def _labeled(rps, p95):
    return [
        {"label": RPS_LABEL, "df": rps.to_frame("sum_all")},
        {"label": P95_LABEL, "df": p95.to_frame("p95_all")},
    ]


# ---- soak ---------------------------------------------------------------------

def test_hold_window_excludes_ramp_and_uses_median_level():
    rps, _ = _soak_frames()
    steps = equal_time_buckets(START_TS, END_TS, 3)
    hold = hold_window_from_series(rps, steps, series_name="sum_all", label=RPS_LABEL)
    assert hold is not None and not hold.whole_run and hold.early_drop is None
    assert hold.start == INDEX[4]
    assert hold.mean_rps == pytest.approx(300.0, abs=1.0)
    assert hold.min_bucket_rps == pytest.approx(300.0, abs=1.5)
    data = hold.to_dict(3)
    assert data["start_iso"] == "2024-01-01T12:04:00+03:00" and data["duration_min"] == pytest.approx(56.0)


def test_hold_window_keeps_an_early_collapse_and_fails_the_hold_check():
    rps = np.concatenate([np.linspace(0, 300, 5), np.full(35, 300.0), np.full(20, 120.0)]) + NOISE
    hold = hold_window_from_series(pd.Series(rps, index=INDEX), equal_time_buckets(START_TS, END_TS, 3))
    assert hold.early_drop is not None and hold.end == INDEX[-1] + pd.Timedelta(minutes=1)
    result = hold_check(hold.to_dict(3), 290.0)
    assert result["passed"] is False and result["lowest_ok"] is False


def test_hold_check_fails_on_sustained_bucket_drop_even_when_mean_passes():
    hold = {"actual_rps_mean": 296.0, "min_bucket_rps": 240.0, "min_bucket_label": "интервал 4",
            "start_iso": "2024-01-01T12:04:00+03:00", "end_iso": "2024-01-01T13:00:00+03:00", "duration_min": 56.0}
    result = hold_check(hold, 300.0)
    assert result["mean_ok"] is True and result["lowest_ok"] is False and result["passed"] is False
    assert "интервал 4" in result["message"] and "12:04–13:00" in result["message"]
    assert hold_check({**hold, "min_bucket_rps": 280.0}, 300.0)["passed"] is True


def _soak_domain_data(p95_end=170.0, hold=True):
    rps, p95 = _soak_frames(p95_end)
    pack = {}
    if hold:
        window = hold_window_from_series(rps, equal_time_buckets(START_TS, END_TS, 3), series_name="sum_all", label=RPS_LABEL)
        pack["hold_window"] = window.to_dict(3)
    return {"lt_framework": {"pack": pack, "labeled": _labeled(rps, p95)}}


def test_soak_sla_checks_use_hold_window_and_target_by_mean():
    sla = {"target_rps": 290, "max_p95_ms": 160, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL}
    result = evaluate_sla(_soak_domain_data(), sla, test_profile={"mode": "stability"})
    target, p95 = result["checks"]
    assert target["name"] == "target_rps" and target["passed"] is True
    assert target["actual"] == pytest.approx(300.0, abs=1.0) and "Удержание" in target["message"]
    assert p95["passed"] is False and "окна удержания" in p95["message"]
    assert result["verdict"] == "Провал"  # any primary breach fails a soak test
    assert result["stable_window"]["kind"] == "hold"


def test_soak_without_hold_window_reports_unknown_target_and_uses_whole_run():
    sla = {"target_rps": 290, "max_p95_ms": 500, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL}
    result = evaluate_sla(_soak_domain_data(hold=False), sla, test_profile={"mode": "stability"})
    target, p95 = result["checks"]
    assert target["passed"] is None and "окна удержания" in target["message"]
    assert p95["passed"] is True and "всего окна" in p95["message"]
    assert result["verdict"] == "Успешно"


def test_soak_resource_breach_is_a_risk_not_a_failure():
    data = _soak_domain_data()
    cpu = pd.DataFrame({"node=a": np.full(60, 95.0)}, index=INDEX)
    data["hard_resources"] = {"labeled": [{"label": "CPU", "df": cpu}], "pack": {}}
    sla = {"target_rps": 290, "max_cpu_pct": 80, "cpu_query": "CPU", "max_performance_query": RPS_LABEL}
    result = evaluate_sla(data, sla, test_profile={"mode": "stability"})
    assert [c["name"] for c in result["checks"] if c["passed"] is False] == ["cpu_usage"]
    assert result["verdict"] == "Есть риски"


def test_compute_trends_flags_adverse_drift_only():
    data = _soak_domain_data(p95_end=170.0)
    hold = data["lt_framework"]["pack"]["hold_window"]
    sla = {"max_performance_query": RPS_LABEL, "p95_query": P95_LABEL}
    trends = {t["metric_kind"]: t for t in compute_trends(data, sla, hold, 20.0)}
    assert trends["p95"]["flagged"] is True and trends["p95"]["change_pct"] > 20
    assert trends["p95"]["slope_per_hour"] > 0
    assert trends["rps"]["flagged"] is False
    flat = {t["metric_kind"]: t for t in compute_trends(_soak_domain_data(p95_end=105.0), sla, hold, 20.0)}
    assert flat["p95"]["flagged"] is False


def test_reconcile_downgrades_success_to_risk_for_flagged_trends():
    base = {"verdict": "Успешно", "summary": "SLA verdict: Успешно. Пройдено: target_rps",
            "checks": [{"name": "target_rps", "passed": True}]}
    result = _reconcile_sla_for_test_profile(base, {"mode": "stability"}, flagged_trends=["p95 latency"])
    assert result["verdict"] == "Есть риски"
    assert result["summary"].startswith("SLA verdict: Есть риски.") and "p95 latency" in result["summary"]
    untouched = _reconcile_sla_for_test_profile(base, {"mode": "capacity"}, flagged_trends=["p95 latency"])
    assert untouched["verdict"] == "Успешно"


def test_sla_step_context_ignores_non_step_windows():
    steps = equal_time_buckets(START_TS, END_TS, 3)
    window = {"start": str(INDEX[4]), "end": str(INDEX[-1]), "level": 300.0, "kind": "hold"}
    assert _sla_step_context({"stable_window": window, "checks": []}, steps, 3) is None
    step_window = {key: value for key, value in window.items() if key != "kind"}
    assert _sla_step_context({"stable_window": step_window, "checks": []}, steps, 3) is not None


# ---- spike --------------------------------------------------------------------

def test_detect_spikes_finds_single_spike_over_baseline():
    rps, _ = _spike_frames()
    profile = detect_spikes(rps, series_name="sum_all", label=RPS_LABEL)
    assert profile.baseline_rps == pytest.approx(100.0, abs=1.0)
    assert len(profile.spikes) == 1
    spike = profile.spikes[0]
    assert spike.start == INDEX[20] and spike.end == INDEX[25]
    assert spike.level_rps == pytest.approx(400.0, abs=1.5) and spike.label.startswith("всплеск 1")


def test_detect_spikes_handles_two_spikes_and_ignores_short_blip():
    values = np.full(60, 100.0)
    values[10:14] = 350.0
    values[30] = 500.0  # one-minute blip: shorter than the minimum spike duration after smoothing
    values[45:50] = 420.0
    profile = detect_spikes(pd.Series(values, index=INDEX))
    assert [s.start for s in profile.spikes] == [INDEX[10], INDEX[45]]
    assert [s.index for s in profile.spikes] == [1, 2]


def test_detect_spikes_finds_nothing_in_flat_load():
    profile = detect_spikes(pd.Series(np.full(60, 200.0) + NOISE, index=INDEX))
    assert profile.spikes == () and profile.threshold_rps == pytest.approx(300.0, abs=2.0)
    assert detect_spikes(pd.Series([100.0, 120.0], index=INDEX[:2])) is None


def test_spike_load_steps_cover_window_contiguously_with_labels():
    rps, _ = _spike_frames()
    steps = spike_load_steps(detect_spikes(rps), START_TS, END_TS, 3)
    assert [s.label.split(" · ")[0] for s in steps] == ["фон 1", "всплеск 1", "фон 2"]
    assert steps[0].start == pd.Timestamp(START_TS, unit="s", tz="UTC")
    assert steps[-1].end == pd.Timestamp(END_TS, unit="s", tz="UTC")
    assert all(a.end == b.start for a, b in zip(steps, steps[1:]))
    assert all(s.source == "spike_detector" and s.plateau_start == s.start for s in steps)


def test_spike_analysis_reports_overshoot_and_recovery_time():
    rps, p95 = _spike_frames()
    profile = analyze_spikes(detect_spikes(rps, series_name="sum_all"), _labeled(rps, p95), {"p95_query": P95_LABEL}, END_TS, 3)
    spike = profile["spikes"][0]
    assert profile["detected"] is True
    assert spike["pre"]["p95"] == pytest.approx(100.0, abs=3.0)
    assert spike["overshoot_p95_pct"] == pytest.approx(300.0, abs=10.0)
    assert spike["recovered"] is True and spike["recovery_time_sec"] == pytest.approx(120.0)
    assert spike["rps_recovered"] is True


def test_spike_analysis_marks_no_recovery_when_p95_stays_high():
    rps, p95 = _spike_frames(recover=False)
    profile = analyze_spikes(detect_spikes(rps), _labeled(rps, p95), {"p95_query": P95_LABEL}, END_TS, 3)
    assert profile["spikes"][0]["recovered"] is False and profile["spikes"][0]["recovery_time_sec"] is None


def test_spike_analysis_without_p95_query_leaves_recovery_unknown():
    rps, p95 = _spike_frames()
    profile = analyze_spikes(detect_spikes(rps), _labeled(rps, p95), {}, END_TS, 3)
    spike = profile["spikes"][0]
    assert spike["recovered"] is None and spike["overshoot_p95_pct"] is None and "p95" in spike["note"]


def _spike_domain_data(recover=True):
    rps, p95 = _spike_frames(recover)
    labeled = _labeled(rps, p95)
    profile = analyze_spikes(detect_spikes(rps, series_name="sum_all", label=RPS_LABEL), labeled, {"p95_query": P95_LABEL}, END_TS, 3)
    return {"lt_framework": {"pack": {"spike_profile": profile}, "labeled": labeled}}


def test_spike_sla_checks_use_spike_window():
    sla = {"target_rps": 350, "max_p95_ms": 300, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL}
    result = evaluate_sla(_spike_domain_data(), sla, test_profile={"mode": "spike"})
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["target_rps"]["passed"] is True and "всплеск 1" in checks["target_rps"]["message"]
    assert checks["p95_latency"]["passed"] is False and "окна всплеска «всплеск 1" in checks["p95_latency"]["message"]
    assert checks["recovery"]["passed"] is True and checks["recovery"]["actual"] == pytest.approx(2.0)
    assert result["verdict"] == "Есть риски"  # latency broke during the spike but recovered
    assert result["stable_window"]["kind"] == "spike"


def test_spike_sla_fails_without_recovery():
    sla = {"target_rps": 350, "max_p95_ms": 500, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL}
    result = evaluate_sla(_spike_domain_data(recover=False), sla, test_profile={"mode": "spike"})
    assert {c["name"]: c["passed"] for c in result["checks"]}["recovery"] is False
    assert result["verdict"] == "Провал"


def test_spike_sla_without_detected_spike_is_insufficient_data():
    flat = pd.Series(np.full(60, 100.0) + NOISE, index=INDEX)
    labeled = _labeled(flat, flat)
    profile = analyze_spikes(detect_spikes(flat), labeled, {"p95_query": P95_LABEL}, END_TS, 3)
    sla = {"target_rps": 350, "max_p95_ms": 500, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL}
    result = evaluate_sla({"lt_framework": {"pack": {"spike_profile": profile}, "labeled": labeled}}, sla, test_profile={"mode": "spike"})
    assert profile["detected"] is False and "порога всплеска" in profile["reason"]
    assert result["verdict"] == "Недостаточно данных" and "не обнаружен" in result["summary"]


@pytest.mark.parametrize(
    "checks, verdict",
    [
        ([("target_rps", True, "primary"), ("p95_latency", True, "primary"), ("recovery", True, "primary")], "Успешно"),
        ([("target_rps", True, "primary"), ("p95_latency", False, "primary"), ("recovery", True, "primary")], "Есть риски"),
        ([("target_rps", True, "primary"), ("recovery", False, "primary")], "Провал"),
        ([("target_rps", False, "primary"), ("recovery", True, "primary")], "Провал"),
        ([("target_rps", True, "primary"), ("cpu_usage", False, "secondary")], "Есть риски"),
    ],
)
def test_spike_verdict_policy(checks, verdict):
    items = [{"name": name, "passed": passed, "category": category} for name, passed, category in checks]
    assert _verdict_for_mode("spike", items) == verdict


def test_spike_payload_drops_machine_fields_and_reports_target():
    payload = build_spike_payload(_spike_domain_data()["lt_framework"]["pack"]["spike_profile"], focus="f", target_rps=350.0, sla_summary="s")
    assert payload["target_reached"] is True and payload["worst_spike_index"] == 1
    assert "start_utc" not in payload["spikes"][0]


# ---- stress -------------------------------------------------------------------

def _stress_steps():
    from AI.context_pack import LoadStep

    steps = []
    for i, level in enumerate([100.0, 200.0, 300.0, 400.0]):
        start = INDEX[i * 15]
        end = INDEX[i * 15 + 14] + pd.Timedelta(minutes=1)
        steps.append(LoadStep(index=i + 1, label=f"ступень {i + 1} · ≈{level:.0f} RPS", start=start, end=end,
                              minute_from=i * 15.0, minute_to=i * 15.0 + 15, rps_level=level, stable=True,
                              source="step_detector", shift_hours=3))
    return steps


def _cpu_frame(values):
    return [{"label": "CPU", "df": pd.DataFrame({"node=a": values}, index=INDEX)}]


def test_saturation_from_sla_window_with_degraded_step():
    steps = _stress_steps()
    window = {"start": str(steps[2].start), "end": str(steps[2].end), "level": 300.0,
              "degraded_level": 400.0, "degraded_checks": ["p95_latency"]}
    cpu = _cpu_frame(np.repeat([30.0, 45.0, 60.0, 92.0], 15))
    result = build_saturation(sla_window=window, sla_checks=[], load_steps=steps, rps_drop=None, hr_labeled=cpu,
                              sla_cfg={"cpu_query": "CPU", "max_cpu_pct": 80}, shift_hours=3)
    assert result["saturation_rps"] == pytest.approx(300.0) and result["saturation_step_label"].startswith("ступень 3")
    assert result["degraded_step_label"].startswith("ступень 4") and result["failure_mode"] == "latency"
    assert result["limit_reached"] is True
    resource = result["limiting_resource"]
    assert resource["kind"] == "cpu" and resource["evidence"] == "threshold" and resource["value"] == pytest.approx(92.0)


def test_limiting_resource_prefers_threshold_breach_then_growth():
    steps = _stress_steps()
    probe = (steps[3].start, steps[3].end)
    first = (steps[0].start, steps[0].end)
    grown = _cpu_frame(np.repeat([30.0, 35.0, 40.0, 60.0], 15))
    resource = limiting_resource(grown, {"cpu_query": "CPU", "max_cpu_pct": 80}, probe, first)
    assert resource["evidence"] == "growth" and resource["growth_pct"] == pytest.approx(100.0)
    flat = _cpu_frame(np.full(60, 30.0))
    assert limiting_resource(flat, {"cpu_query": "CPU", "max_cpu_pct": 80}, probe, first) is None


def test_saturation_reports_limit_not_reached_without_degradation():
    steps = _stress_steps()
    window = {"start": str(steps[3].start), "end": str(steps[3].end), "level": 400.0, "degraded_level": None, "degraded_checks": []}
    result = build_saturation(sla_window=window, sla_checks=[{"name": "p95_latency", "passed": True}], load_steps=steps,
                              rps_drop=None, hr_labeled=[], sla_cfg={}, shift_hours=3)
    assert result["limit_reached"] is False and result["failure_mode"] == "none" and "не достигнут" in result["note"]
    assert failure_mode(["p95_latency", "error_rate"], None) == "latency+errors"
    assert failure_mode([], pd.Timestamp("2024-01-01T09:40:00Z")) == "throughput_drop"


def test_stress_sla_keeps_capacity_window_and_reports_mode_stress():
    sla = {"target_rps": 250, "max_performance_query": RPS_LABEL}
    pack = {"load_steps": [s.to_dict() for s in _stress_steps()]}
    result = evaluate_sla({"lt_framework": {"pack": pack, "labeled": []}}, sla, test_profile={"mode": "stress"})
    assert result["test_mode"] == "stress"
    assert result["stable_window"]["level"] == pytest.approx(400.0) and "kind" not in result["stable_window"]
    assert result["verdict"] == "Успешно"


# ---- rendering ----------------------------------------------------------------

def _soak_report():
    return {
        "verdict": "Есть риски",
        "test_profile": {"mode": "stability", "test_type": "soak", "normalized_test_type": "soak", "label_ru": "Долговременный (soak)"},
        "peak_performance": {"not_applicable": True},
        "stability_under_load": {
            "target_rps": 290, "actual_rps": 299.8, "sla_summary": "SLA verdict: Есть риски",
            "hold": {"start_iso": "2024-01-01T12:04:00+03:00", "end_iso": "2024-01-01T13:00:00+03:00", "duration_min": 56,
                     "min_bucket_rps": 299.4, "min_bucket_label": "интервал 3", "drop_pct": 0.1},
            "trends": [{"metric_kind": "p95", "title": "p95 latency", "change_pct": 61.2, "flagged": True}],
        },
    }


def test_test_type_section_rows_per_type():
    title, rows = test_type_section(_soak_report())
    values = dict(rows)
    assert title == "Стабильность под нагрузкой"
    assert values["Окно удержания"] == "12:04–13:00 (56 мин)" and "p95 latency: +61.2 % (риск)" in values["Тренды за окно удержания"]
    spike = {"spike_analysis": build_spike_payload(_spike_domain_data()["lt_framework"]["pack"]["spike_profile"], focus="f", target_rps=350.0, sla_summary="s")}
    title, rows = test_type_section(spike)
    assert title == "Всплески нагрузки" and any(label.startswith("всплеск 1") and "восстановление за 120 с" in value for label, value in rows)
    stress = {"saturation": {"saturation_rps": 300, "failure_mode": "latency", "failure_mode_title": "рост задержек",
                             "limiting_resource": {"kind": "cpu", "value": 92, "threshold": 80, "evidence": "threshold"}},
              "peak_performance": {"max_rps": 300}}
    title, rows = test_type_section(stress)
    assert title == "Точка насыщения" and dict(rows)["Ограничивающий ресурс"].startswith("CPU: p95 92 > порог 80")
    assert test_type_section({"peak_performance": {"max_rps": 300}}) is None


def test_legacy_soak_payload_still_renders():
    title, rows = test_type_section({"test_profile": {"mode": "stability", "test_type": "soak"},
                                     "stability_under_load": {"target_rps": 200, "actual_rps": 210, "sla_summary": "ok"}})
    assert title == "Стабильность под нагрузкой" and dict(rows)["Фактический RPS"] == "210"


def test_confluence_export_and_legacy_markdown_render_test_type_sections():
    final = render_llm_domain_body({"domain": "final", "parsed": _soak_report()})
    assert "Стабильность под нагрузкой" in final and "Пиковая производительность" not in final and "интервал 3" in final
    lt = render_llm_domain_body({"domain": "lt_framework", "parsed": _soak_report()})
    assert "Стабильность под нагрузкой" not in lt
    md = render_llm_markdown({**_soak_report(), "findings": [], "recommended_actions": []})
    assert "#### Стабильность под нагрузкой" in md and "| Окно удержания | 12:04–13:00 (56 мин) |" in md
    stress = render_llm_domain_body({"domain": "final", "parsed": {
        "saturation": {"saturation_rps": 300, "failure_mode_title": "рост задержек"}, "peak_performance": {"max_rps": 300}}})
    assert "Точка насыщения" in stress and "Пиковая производительность" not in stress


# ---- pipeline -----------------------------------------------------------------

def _pipeline_frames(kind):
    if kind in ("step", "stress"):
        rps = np.repeat([100.0, 200.0, 300.0, 400.0], 15) + NOISE
        p95 = np.repeat([100.0, 110.0, 130.0, 130.0], 15) + NOISE
        p95[52:] = 420.0  # latency breaks late on the last step
        return [pd.DataFrame({"sum_all": rps}, index=INDEX), pd.DataFrame({"p95_all": p95}, index=INDEX)]
    rps, p95 = _soak_frames() if kind == "soak" else _spike_frames()
    return [rps.to_frame("sum_all"), p95.to_frame("p95_all")]


PIPELINE_CASES = {
    "step": ({"target_rps": 250, "max_p95_ms": 300}, "capacity", None, "sla_step"),
    "stress": ({"target_rps": 250, "max_p95_ms": 300}, "stress", "saturation", "sla_step"),
    "soak": ({"target_rps": 290, "max_p95_ms": 300}, "stability", "stability_under_load", "hold_window"),
    "spike": ({"target_rps": 350, "max_p95_ms": 500}, "spike", "spike_analysis", "spike_windows"),
}


@pytest.mark.parametrize("kind", list(PIPELINE_CASES))
def test_upload_from_llm_payload_per_test_type(monkeypatch, kind):
    sla, mode, payload_key, ctx_key = PIPELINE_CASES[kind]
    cfg = {
        "llm": {"include_markdown_tables_in_context": False, "self_consistency_k": 1, "max_domain_workers": 1},
        "default_params": {"step": "1m", "resample_interval": "1T"},
        "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prometheus:9090"}}},
        "domain_sources": {"default": {"source": "prom"}, "lt_framework": {"source": "prom"}},
        "queries": {"lt_framework": {"promql_queries": ["a", "b"], "label_keys_list": [[], []], "labels": [RPS_LABEL, P95_LABEL]}},
        "sla": {"max_performance_query": RPS_LABEL, "p95_query": P95_LABEL, **sla},
    }
    captured = {}
    monkeypatch.setattr(pipeline_module, "read_prompt_from_file", lambda filename: "prompt")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *a, **k: _pipeline_frames(kind))

    def fake_llm(user_prompt, data_context, k, return_scores, domain_key):
        payload = {"verdict": "Успешно", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}
        captured[domain_key] = (user_prompt, json.loads(data_context))
        return json.dumps(payload, ensure_ascii=False), dict(payload), {}

    monkeypatch.setattr(pipeline_module, "llm_two_pass_self_consistency", fake_llm)
    result = pipeline_module.uploadFromLLM(
        start_ts=START_TS, end_ts=END_TS, save_to_db=False, ef_config=cfg,
        run_meta={"test_type": kind, "run_name": f"run-{kind}", "service": "demo"}, active_domains=["lt_framework"],
    )
    final = result["final_parsed"]
    lt_prompt, lt_ctx = captured["lt_framework"]
    final_ctx = captured["final"][1]

    assert final["test_profile"]["mode"] == mode and final["deterministic_sla"]["test_mode"] == mode
    assert lt_prompt.count("[Профиль теста") == 1
    assert ctx_key in lt_ctx
    for other in {"sla_step", "hold_window", "spike_windows"} - {ctx_key}:
        assert other not in lt_ctx
    if payload_key:
        assert payload_key in final and payload_key in final_ctx
    peak = final["peak_performance"]
    if mode in ("capacity", "stress"):
        assert peak["max_rps"] == pytest.approx(300.0, abs=2.0)
    else:
        assert peak["not_applicable"] is True and "max_rps" not in peak

    if kind == "soak":
        hold = final["stability_under_load"]["hold"]
        assert hold["actual_rps_mean"] == pytest.approx(300.0, abs=1.0)
        assert "p95" in final["stability_under_load"]["flagged_trends"]
        assert result["sla_verdict"] == "Есть риски"
    if kind == "spike":
        assert final["spike_analysis"]["detected"] is True
        assert [s["label"].split(" · ")[0] for s in final_ctx["load_steps"]] == ["фон 1", "всплеск 1", "фон 2"]
        assert {c["name"] for c in final["deterministic_sla"]["checks"]} >= {"target_rps", "recovery"}
        assert result["sla_verdict"] == "Успешно"
    if kind == "stress":
        saturation = final["saturation"]
        assert saturation["saturation_rps"] == pytest.approx(300.0, abs=2.0)
        assert saturation["failure_mode"] == "latency" and saturation["limit_reached"] is True
