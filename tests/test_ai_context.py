"""Tests for the step-aware LLM context, baseline lookup, prompt assembly and finding verification."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AI import pipeline as pipeline_module
from AI import providers as providers_module
from AI import scoring as scoring_module
from AI.context_pack import (
    LoadStep,
    build_timeline,
    derive_load_steps,
    detect_step_anomalies,
    equal_time_buckets,
    first_rps_drop,
    format_iso,
    load_step_report_from_context,
    load_step_report_from_frames,
    step_table,
)
from AI.db_store import BaselineRun, find_previous_run, load_run_metric_stats
from AI.pipeline import (
    _align_domain_verdict_with_sla,
    _baseline_for_domain,
    _baseline_for_overall,
    _sla_step_context,
    build_context_pack,
)
from AI.prompt_assembly import assemble_prompt, build_analysis_guidance
from AI.scoring import LLMAnalysis, parse_llm_analysis_strict
from AI.sla_evaluator import evaluate_sla, format_sla_rationale
from AI.verification import build_reference_catalog, unverified_findings, verify_analysis
from tests.test_smoke_api import RecordingConnection

PROMPTS_DIR = ROOT_DIR / "AI" / "prompts"
START_TS = 1704103200.0  # 2024-01-01T10:00:00Z
END_TS = START_TS + 40 * 60


def _staircase(levels, minutes_per_step=10, freq_sec=30, noise=0.0):
    """RPS staircase: one level per step, sampled every ``freq_sec`` seconds."""
    samples = minutes_per_step * 60 // freq_sec
    values = np.concatenate([np.full(samples, float(level)) for level in levels])
    if noise:
        values = values + np.linspace(0, noise, values.size)
    index = pd.date_range(datetime.fromtimestamp(START_TS, tz=timezone.utc), periods=values.size, freq=f"{freq_sec}s")
    return pd.Series(values, index=index)


def _segments(levels, minutes_per_step=10):
    start = datetime.fromtimestamp(START_TS, tz=timezone.utc)
    out = []
    for i, level in enumerate(levels):
        seg_start = start + pd.Timedelta(minutes=minutes_per_step * i)
        seg_end = seg_start + pd.Timedelta(minutes=minutes_per_step) - pd.Timedelta(seconds=30)
        out.append({"start": str(seg_start), "end": str(seg_end), "level": float(level), "stable": True})
    return out


# ---- load steps ----------------------------------------------------------------

def test_derive_load_steps_from_detector_segments():
    steps = derive_load_steps(_segments([50, 100, 150, 200]), START_TS, END_TS, shift_hours=3)
    assert [s.rps_level for s in steps] == [50, 100, 150, 200]
    assert steps[0].source == "step_detector"
    assert steps[0].to_dict()["start_iso"] == "2024-01-01T13:00:00+03:00"
    assert steps[-1].to_dict()["end_iso"] == "2024-01-01T13:40:00+03:00"
    assert steps[1].minute_from == pytest.approx(10.0) and steps[1].minute_to == pytest.approx(20.0)
    assert "≈100 RPS" in steps[1].label
    # Contiguous coverage: each step starts where the previous one ends.
    assert all(steps[i].end == steps[i + 1].start for i in range(len(steps) - 1))


def test_derive_load_steps_falls_back_to_equal_buckets():
    steps = derive_load_steps(None, START_TS, END_TS, shift_hours=0)
    assert len(steps) == 4 and steps[0].source == "time_buckets"
    assert steps[0].rps_level is None and "интервал 1 · 10:00–10:10" == steps[0].label
    single = derive_load_steps(_segments([100])[:1], START_TS, END_TS, shift_hours=0)
    assert single[0].source == "time_buckets"
    long_run = equal_time_buckets(START_TS, START_TS + 8 * 3600, shift_hours=0)
    assert len(long_run) == 8


def test_format_iso_handles_naive_and_aware_timestamps():
    assert format_iso(pd.Timestamp("2024-01-01 10:00:00"), 3) == "2024-01-01T13:00:00+03:00"
    assert format_iso(pd.Timestamp("2024-01-01 10:00:00", tz="UTC"), -2) == "2024-01-01T08:00:00-02:00"


# ---- step table and anomalies ----------------------------------------------------

def test_step_table_ranks_series_by_change_and_aggregates_instances():
    steps = derive_load_steps(_segments([50, 100, 150, 200]), START_TS, END_TS, shift_hours=0)
    rps = _staircase([50, 100, 150, 200])
    latency = _staircase([0.10, 0.11, 0.12, 0.45])
    flat_big = _staircase([900, 900, 900, 900])
    df = pd.DataFrame({
        "latency|application=api|instance=a": latency,
        "latency|application=api|instance=b": latency * 1.1,
        "flat|application=big": flat_big,
    })
    sections = step_table([{"label": "p95 latency", "df": df}, {"label": "RPS", "df": rps.to_frame("sum_all")}], steps, top_n=5)
    rows = sections[0]["rows"]
    assert rows[0]["series"] == "latency|application=api|instances=2 (среднее)"
    assert rows[0]["change_vs_first_pct"] == pytest.approx(350.0, abs=1.0)
    assert rows[-1]["series"] == "flat|application=big" and rows[-1]["change_vs_first_pct"] == pytest.approx(0.0)
    assert [s["step"] for s in rows[0]["per_step"]] == [1, 2, 3, 4]
    assert rows[0]["per_step"][3]["samples"] == 20
    assert sections[1]["rows"][0]["per_step"][3]["mean"] == pytest.approx(200.0)


def test_step_stats_use_plateau_and_report_drop_after_it():
    # The last step holds 258 RPS for 6 minutes, then RPS falls to 60 until the test end.
    index = pd.date_range(datetime.fromtimestamp(START_TS, tz=timezone.utc), periods=40, freq="1min")
    rps = pd.Series([100.0] * 20 + [258.0] * 6 + [60.0] * 14, index=index)
    segments = [
        {"start": str(index[0]), "end": str(index[19]), "level": 100.0, "stable": True},
        {"start": str(index[20]), "end": str(index[25]), "level": 258.0, "stable": True, "drop_time": str(index[26])},
    ]
    steps = derive_load_steps(segments, START_TS, START_TS + 40 * 60, shift_hours=3)
    assert steps[-1].to_dict()["drop_iso"] == "2024-01-01T13:26:00+03:00"
    row = step_table([{"label": "RPS", "df": rps.to_frame("all")}], steps)[0]["rows"][0]
    last = row["per_step"][-1]
    assert last["mean"] == pytest.approx(258.0)
    assert last["after_plateau"]["min"] == pytest.approx(60.0)
    assert "after_plateau" not in row["per_step"][0]


def test_segments_shorter_than_min_stable_are_not_load_steps():
    # 15-minute steps with 1-minute ramps, then RPS collapses 258 -> 147 -> 60 within ten minutes.
    # The detector cuts 3-minute segments out of every ramp and out of the collapse.
    values = []
    for previous, level in zip((None, 100, 150, 200), (100, 150, 200, 258)):
        if previous is not None:
            values += list(np.linspace(previous, level, 6)[1:-1])
        values += [float(level)] * 60
    values += list(np.linspace(258, 147, 6)[1:-1]) + [147.0] * 12 + [60.0] * 20
    index = pd.date_range("2026-10-01T10:00:00Z", periods=len(values), freq="15s")
    labeled = [{"label": "RPS", "df": pd.Series(values, index=index).to_frame("all")}]
    cfg = {"use_step_profile": True}
    start_ts, end_ts = index[0].timestamp(), (index[-1] + pd.Timedelta(seconds=15)).timestamp()

    run = pipeline_module._derive_steps_for_run(
        lt_labeled=labeled, perf_label="RPS", stable_cfg=cfg, min_stable_minutes=10.0,
        start_ts=start_ts, end_ts=end_ts, shift_hours=0,
    )
    steps = run.steps
    assert [round(step.rps_level) for step in steps] == [100, 150, 200, 258]
    assert run.rps_drop is not None and steps[-1].start < run.rps_drop <= index[-1]
    assert steps[-1].stable is True and "drop_iso" in steps[-1].to_dict()
    last = step_table(labeled, steps)[0]["rows"][0]["per_step"][-1]
    assert last["mean"] == pytest.approx(258.0)
    assert last["after_plateau"]["min"] == pytest.approx(60.0)

    pack = build_context_pack(labeled, top_n=5, min_stable_minutes=10.0, stable_detection_cfg=cfg, steps=steps, start_ts=start_ts)
    segments = pack["sections"][0]["top_series"][0]["step_segments"]
    assert [round(seg["level"]) for seg in segments] == [100, 150, 200, 258]


def test_labels_with_data_skips_empty_frames():
    index = pd.date_range("2026-10-01T10:00:00Z", periods=3, freq="1min")
    filled = pd.DataFrame({"all": [1.0, 2.0, 3.0]}, index=index)
    empty = pd.DataFrame({"all": [float("nan")] * 3}, index=index)
    labels = pipeline_module._labels_with_data([
        {"label": "RPS", "df": filled},
        {"label": "empty", "df": empty},
        {"label": "blank", "df": pd.DataFrame()},
        {"label": "", "df": filled},
        "not-a-section",
    ])
    assert labels == ["RPS"]


def test_first_rps_drop_is_the_earliest_fall():
    later_drop = "2026-10-01T10:20:00Z"
    earlier_step_down = "2026-10-01T10:12:00Z"
    segments = [
        {"start": "2026-10-01T10:00:00Z", "end": "2026-10-01T10:10:00Z", "level": 100.0, "drop_time": later_drop},
        {"start": earlier_step_down, "end": "2026-10-01T10:18:00Z", "level": 40.0, "after_drop": True},
    ]
    assert first_rps_drop(segments) == pd.Timestamp(earlier_step_down)
    assert first_rps_drop(None) is None
    assert first_rps_drop([]) is None


def test_load_step_report_keeps_plateau_rps_and_latency():
    index = pd.date_range(datetime.fromtimestamp(START_TS, tz=timezone.utc), periods=40, freq="1min")
    rps = pd.Series([100.0] * 20 + [258.0] * 6 + [60.0] * 14, index=index)
    latency = pd.Series([40.0] * 20 + [80.0] * 6 + [900.0] * 14, index=index)
    segments = [
        {"start": str(index[0]), "end": str(index[19]), "level": 100.0, "stable": True},
        {"start": str(index[20]), "end": str(index[25]), "level": 258.0, "stable": True, "drop_time": str(index[26])},
    ]
    steps = derive_load_steps(segments, START_TS, START_TS + 40 * 60, shift_hours=0)
    labeled = [
        {"label": "Transactions per second", "df": rps.to_frame("sum")},
        {"label": "p95 response time", "df": latency.to_frame("p95|tx=pay")},
    ]
    report = load_step_report_from_frames(
        steps,
        labeled,
        rps_query="Transactions per second",
        latency_query="p95 response time",
        selected_level=258,
        selected_start=str(index[20]),
    )
    last = report["steps"][-1]
    assert last["rps"] == pytest.approx(258.0)
    assert last["latency_p95"] == pytest.approx(80.0)
    assert last["selected"] is True
    assert report["steps"][0]["selected"] is False
    assert report["steps"][0]["latency_p95"] == pytest.approx(40.0)
    assert report["latency_series"] == "p95|tx=pay"

    pack = {
        "load_steps": [step.to_dict() for step in steps],
        "step_table": [
            {"label": "Transactions per second", "rows": [{"series": "sum", "overall_max": 258, "per_step": [
                {"step": 1, "mean": 100, "p95": 100},
                {"step": 2, "mean": 147, "p95": 258},
            ]}]},
            {"label": "p95 response time", "rows": [{"series": "p95|tx=pay", "overall_max": 900, "per_step": [
                {"step": 1, "mean": 40, "p95": 40},
                {"step": 2, "mean": 80, "p95": 80},
            ]}]},
        ],
    }
    checks = [
        {"name": "target_rps", "actual": 258, "message": "label='Transactions per second'; series='sum'"},
        {"name": "p95_latency", "actual": 80, "message": "запрос «p95 response time», серия «p95|tx=pay»"},
    ]
    from_context = load_step_report_from_context(pack, checks=checks)
    assert from_context["steps"][-1]["rps"] == pytest.approx(258.0)
    assert from_context["steps"][-1]["latency_p95"] == pytest.approx(80.0)
    assert from_context["steps"][-1]["selected"] is True
    bare = load_step_report_from_context(pack)
    assert bare["steps"][-1]["rps"] == pytest.approx(258.0)
    assert bare["steps"][-1]["latency_p95"] == pytest.approx(80.0)
    assert bare["steps"][-1]["selected"] is False


def test_sla_window_selects_step_when_bucket_rps_differs():
    early = LoadStep(
        1, "интервал 1",
        pd.Timestamp("2026-10-01T20:26:00+03:00"),
        pd.Timestamp("2026-10-01T21:06:00+03:00"),
        0, 40, 703.4, None, "time_buckets", 3,
    )
    late = LoadStep(
        8, "интервал 8",
        pd.Timestamp("2026-10-02T01:06:52+03:00"),
        pd.Timestamp("2026-10-02T01:47:00+03:00"),
        280, 320, 1472.9, None, "time_buckets", 3,
    )
    report = load_step_report_from_frames(
        [early, late],
        [],
        selected_level=1667.5,
        selected_start="2026-10-01 22:27:00+00:00",
    )
    assert report["steps"][0]["selected"] is False
    assert report["steps"][1]["selected"] is True


def test_detect_step_anomalies_elasticity_and_drift():
    steps = derive_load_steps(_segments([100, 200, 300, 400]), START_TS, END_TS, shift_hours=0)
    # Latency doubles between steps 1 and 2 although load grew by 100% -> ratio 1.0 (no anomaly);
    # on step 4 latency is x10 while the load is x4 -> elasticity anomaly.
    latency = _staircase([100, 200, 300, 1000])
    anomalies = detect_step_anomalies(latency, steps)
    kinds = {(a["kind"], a["step_index"]) for a in anomalies}
    assert ("elasticity", 4) in kinds
    elastic = next(a for a in anomalies if a["kind"] == "elasticity" and a["step_index"] == 4)
    assert elastic["load_change_pct"] == pytest.approx(300.0)
    assert elastic["change_pct"] == pytest.approx(900.0)
    assert "ступень 4" in elastic["explanation"]

    drifting = _staircase([100, 100, 100, 100])
    last_step = drifting.index >= steps[3].start
    drifting[last_step] = np.linspace(100, 200, int(last_step.sum()))
    drift = [a for a in detect_step_anomalies(drifting, steps) if a["kind"] == "drift"]
    assert drift and drift[0]["step_index"] == 4 and drift[0]["change_pct"] > 20
    assert detect_step_anomalies(_staircase([100, 100, 100, 100]), steps) == []


def test_detect_step_anomalies_shift_for_time_buckets():
    steps = equal_time_buckets(START_TS, END_TS, shift_hours=0)
    series = _staircase([10, 10, 30, 30])
    anomalies = detect_step_anomalies(series, steps)
    assert anomalies and anomalies[0]["kind"] == "shift" and anomalies[0]["change_pct"] == pytest.approx(200.0)


def test_build_context_pack_with_steps_and_timeline():
    steps = derive_load_steps(_segments([50, 100, 150, 200]), START_TS, END_TS, shift_hours=0)
    lt_labeled = [{"label": "RPS", "df": _staircase([50, 100, 150, 200]).to_frame("sum_all")}]
    # Heap grows x9 on the last step while the load grows x4 -> elasticity anomaly.
    jvm_labeled = [{"label": "Heap used", "df": _staircase([1.0, 1.1, 1.2, 9.0]).to_frame("heap|application=api")}]
    lt_pack = build_context_pack(lt_labeled, top_n=5, steps=steps, start_ts=START_TS)
    jvm_pack = build_context_pack(jvm_labeled, top_n=5, steps=steps, start_ts=START_TS)
    assert lt_pack["load_steps"][0]["index"] == 1 and len(lt_pack["step_table"]) == 1
    top = lt_pack["sections"][0]["top_series"][0]
    shift = pipeline_module._time_shift_hours()
    assert top["max_time_iso"] == format_iso(pd.Timestamp(START_TS + 30 * 60, unit="s", tz="UTC"), shift)
    assert top["max_minute_from_start"] == pytest.approx(30.0)
    assert top["change_pct"] == pytest.approx(300.0)
    assert jvm_pack["sections"][0]["anomalies"][0]["step_anomalies"][0]["kind"] == "elasticity"

    timeline = build_timeline({"jvm": jvm_pack, "lt_framework": lt_pack}, steps)
    assert [row["domain"] for row in timeline["rows"]] == ["lt_framework", "jvm"]
    assert timeline["rows"][0]["mean_per_step"] == pytest.approx([50, 100, 150, 200])
    assert len(timeline["steps"]) == 4


# ---- baseline ---------------------------------------------------------------------

def test_find_previous_run_and_metric_stats_with_scripted_connection():
    created = datetime(2024, 10, 31, 11, 0, tzinfo=timezone.utc)
    conn = RecordingConnection(
        fetchone_results=[("release-2.4", "demo", "Успешно", created)],
        fetchall_results=[[("jvm", "Heap used", "heap|application=api", 1.5, 2.0, 2.5)]],
    )
    previous = find_previous_run(
        conn, "public", "llm_reports", exclude_run_name="nightly-2", service="demo",
        before=datetime(2024, 11, 1, tzinfo=timezone.utc),
    )
    assert previous == BaselineRun(run_name="release-2.4", service="demo", verdict="Успешно", created_at=created)
    sql_text, params = conn.log[0]
    assert "r.service = %s" in sql_text and "= 'Успешно'" in sql_text and "WITH current" not in sql_text
    assert params[0] == "demo" and params[1] == "nightly-2"

    stats = load_run_metric_stats(conn, "public", "metrics", "release-2.4")
    assert stats["jvm"]["Heap used"]["heap|application=api"].p95 == pytest.approx(2.0)
    assert conn.log[1][1] == ("release-2.4",)

    with pytest.raises(ValueError):
        find_previous_run(conn, "public", "llm_reports", exclude_run_name="x", mode="best")


def test_baseline_context_for_domain_and_overall():
    stats = load_run_metric_stats(
        RecordingConnection(fetchall_results=[[("jvm", "Heap used", "heap|application=api", 1.0, 1.5, 2.0)]]),
        "public", "metrics", "release-2.4",
    )
    baseline_ctx = {"available": True, "run_name": "release-2.4", "verdict": "Успешно", "created_at": None, "stats": stats}
    pack = {"sections": [{"label": "Heap used", "top_series": [{"series": "heap|application=api", "mean": 1.5, "max": 3.0}, {"series": "other", "mean": 1.0, "max": 1.0}]}]}
    domain = _baseline_for_domain(baseline_ctx, "jvm", pack)
    row = domain["sections"]["Heap used"]["heap|application=api"]
    assert row["delta_mean_pct"] == pytest.approx(50.0) and row["delta_max_pct"] == pytest.approx(50.0)
    assert "other" not in domain["sections"]["Heap used"]
    overall = _baseline_for_overall(baseline_ctx, {"jvm": {"pack": pack}, "kafka": {"pack": {"sections": []}}})
    assert overall["key_deltas"][0]["series"] == "heap|application=api"
    assert _baseline_for_domain({"available": False, "reason": "нет прогонов"}, "jvm", pack) == {"available": False, "reason": "нет прогонов"}


# ---- prompts ----------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "jvm_prompt.txt", "database_prompt.txt", "kafka_prompt.txt", "microservices_prompt.txt",
    "hard_resources_prompt.txt", "lt_framework_prompt.txt", "application_logs_prompt.txt", "overall_prompt.txt",
])
def test_domain_prompts_have_no_hardcoded_thresholds(name):
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8").lower()
    for forbidden in ("мировой практике", "минимум 5", "рекомендуемые ориентиры", "рекомендуемые sla", "все значения из таблиц"):
        assert forbidden not in text, f"{name} still contains «{forbidden}»"
    assert "ступен" in text or "интервал" in text


def test_build_analysis_guidance_uses_service_sla_and_context_rules():
    sla = {"target_rps": 200, "max_p95_ms": 500, "max_performance_query": "LT RPS", "target_rps_allow_peak_fallback": False}
    context = {"operational_context": {"normal_degradation_rules": ["рост p95 до 10 % на пике"], "analysis_focus": ["checkout"], "known_risks": [], "known_constraints": []}}
    guidance = build_analysis_guidance(sla, context, "step")
    assert "Целевой RPS (target_rps): 200 RPS" in guidance and "p95 latency: 500 мс" in guidance
    assert "«LT RPS»" in guidance and "пиковый max использовать ЗАПРЕЩЕНО" in guidance
    assert "рост p95 до 10 % на пике" in guidance and "checkout" in guidance
    assert "ступенчатый поиск" in guidance

    without_sla = build_analysis_guidance({}, None, "soak")
    assert "Числовых SLA для этого сервиса не задано" in without_sla and "soak" in without_sla
    assert "target_rps" not in without_sla

    prompt = assemble_prompt("domain text", "guide", guidance, "format")
    assert prompt.split("\n\n")[0] == "guide" and prompt.endswith("format") and "domain text" in prompt


def test_domain_guidance_names_the_query_and_hides_stable_max():
    sla = {
        "target_rps": 1000,
        "max_p95_ms": 2000,
        "max_cpu_pct": 90,
        "p95_query": "LT p95",
        "cpu_query": "Nodes CPU",
        "max_performance_query": "LT RPS",
    }
    guidance = build_analysis_guidance(sla, None, "step", domain="microservices")
    assert "2000 мс — проверяется программно по запросу «LT p95»" in guidance
    assert "90 % — проверяется программно по запросу «Nodes CPU»" in guidance
    assert "sla_step" in guidance
    assert "используется stable_max" not in guidance
    assert "используется stable_max" in build_analysis_guidance(sla, None, "step")


def test_context_pack_can_omit_stable_max():
    labeled = [{
        "label": "latency",
        "df": pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")).to_frame("lat"),
    }]
    series = build_context_pack(labeled, top_n=5, include_stable_max=False)["sections"][0]["top_series"][0]
    assert "stable_max" not in series
    assert "step_segments" not in series


def test_sla_step_context_and_domain_verdict_guard():
    step5 = LoadStep(
        index=5, label="интервал 5",
        start=pd.Timestamp("2026-10-01T20:06:30+00:00"),
        end=pd.Timestamp("2026-10-01T20:46:37+00:00"),
        minute_from=0, minute_to=40, rps_level=1227.6, stable=None,
        source="time_buckets", shift_hours=3,
    )
    step6 = LoadStep(
        index=6, label="интервал 6",
        start=pd.Timestamp("2026-10-01T20:46:37+00:00"),
        end=pd.Timestamp("2026-10-01T21:26:45+00:00"),
        minute_from=40, minute_to=80, rps_level=1276.6, stable=None,
        source="time_buckets", shift_hours=3,
    )
    sla_result = {
        "verdict": "Есть риски",
        "stable_window": {
            "start": "2026-10-01T20:06:30+00:00",
            "end": "2026-10-01T20:46:37+00:00",
            "level": 1227.6,
        },
        "checks": [
            {"name": "target_rps", "category": "primary", "threshold": 1000, "actual": 1227.6, "passed": True, "message": "ok"},
            {"name": "p95_latency", "category": "primary", "threshold": 2000, "actual": 187, "passed": True, "message": "ok"},
            {"name": "memory_usage", "category": "secondary", "threshold": 90, "actual": 95, "passed": False, "message": "high"},
        ],
    }
    sla_step = _sla_step_context(sla_result, [step5, step6], 3)
    assert sla_step["step_index"] == 5
    assert sla_step["primary_passed"] is True

    original = {
        "verdict": "Провал",
        "verdict_rationale": "Статус «Провал»: латентность выросла.",
        "findings": [{"id": "f1", "severity": "critical", "start_time": "2026-10-02T00:26:45+03:00", "summary": "после"}],
    }
    aligned = _align_domain_verdict_with_sla(json.dumps(original, ensure_ascii=False), original, sla_step)
    assert aligned is not None
    assert original["verdict"] == "Провал"
    assert json.loads(aligned[0])["verdict"] == "Есть риски"
    assert aligned[1]["verdict"] == "Есть риски"
    assert aligned[1]["verdict_rationale"].startswith("Статус скорректирован по ступени SLA")
    inside = {
        "verdict": "Провал",
        "verdict_rationale": "рано",
        "findings": [{"id": "f1", "severity": "high", "start_time": "2026-10-01T23:10:00+03:00", "summary": "внутри"}],
    }
    assert _align_domain_verdict_with_sla(json.dumps(inside, ensure_ascii=False), inside, sla_step) is None


# ---- verification -----------------------------------------------------------------

def _analysis(findings):
    return parse_llm_analysis_strict(json.dumps({"verdict": "Есть риски", "findings": findings, "recommended_actions": []}, ensure_ascii=False))


def test_verify_analysis_marks_verified_unverified_and_qualitative():
    ctx = {
        "domain": "jvm",
        "load_steps": [{"index": 1, "label": "ступень 1 · ≈100 RPS", "rps_level": 100.0}],
        "sections": [{"label": "Heap used", "top_series": [{"series": "heap|application=api", "mean": 1.5, "min": 1.0, "max": 2.4, "last": 2.4}], "anomalies": []}],
        "step_table": [{"label": "Heap used", "rows": [{"series": "heap|application=api", "change_vs_first_pct": 140.0, "per_step": [{"step": 1, "mean": 1.0, "p95": 1.1, "max": 1.2, "samples": 20}, {"step": 2, "mean": 2.4, "p95": 2.5, "max": 2.6, "samples": 20}]}]}],
        "baseline": {"available": True, "sections": {"Heap used": {"heap|application=api": {"baseline_mean": 1.2, "delta_mean_pct": 25.0}}}},
    }
    analysis = _analysis([
        {"id": "f1", "summary": "Heap used вырос до 2.4 ГБ на ступени 2 (+140 %)", "component": "api", "evidence_items": [{"metric": "heap|application=api", "observed_value": "2.4", "threshold": "не задан", "note": "baseline +25 %"}]},
        {"id": "f2", "summary": "GC pause достиг 620 мс", "component": "api", "evidence_items": [{"metric": "gc pause", "observed_value": "620мс", "threshold": "200мс", "note": ""}]},
        {"id": "f3", "summary": "Потоки стабильны, утечек не видно", "component": "api"},
    ])
    verified, summary = verify_analysis(analysis, ctx)
    statuses = {f.id: f.verification.status for f in verified.findings}
    assert statuses == {"f1": "verified", "f2": "unverified", "f3": "qualitative"}
    assert verified.findings[0].verification.claims_total == 3 and verified.findings[0].verification.claims_matched == 3
    assert verified.findings[1].verification.unmatched == ["620 ms", "200 ms"]
    assert (summary.total, summary.verified, summary.unverified, summary.qualitative) == (3, 1, 1, 1)
    assert verified.verification_summary.unverified == 1 and summary.revised_by_model is False
    assert unverified_findings(verified) == [{"id": "f2", "summary": "GC pause достиг 620 мс", "unmatched_numbers": ["620 ms", "200 ms"]}]
    # Serialized findings carry the flag for the DB and the UI.
    assert verified.dict()["findings"][1]["verification"]["status"] == "unverified"
    assert analysis.findings[0].verification is None  # input is not mutated


def test_reference_catalog_includes_timeline_sla_and_designated_peak():
    ctx = {
        "domains": {"lt_framework": {"sections": [], "step_table": []}},
        "timeline": {"rows": [{"domain": "lt_framework", "label": "RPS", "series": "sum_all", "mean_per_step": [50.0, 100.0]}]},
        "designated_peak_performance": {"series": "sum_all", "stable_max": 226.5},
        "deterministic_sla": {"checks": [{"name": "target_rps", "threshold": 200, "actual": 226.5}]},
    }
    values = {ref["value"] for ref in build_reference_catalog(ctx)}
    assert {50.0, 100.0, 226.5, 200.0} <= values
    analysis = _analysis([{"id": "f1", "summary": "Достигнуто 226.5 RPS при целевых 200", "component": "lt"}])
    verified, _ = verify_analysis(analysis, ctx)
    assert verified.findings[0].verification.status == "verified"


def test_old_reports_without_verification_still_parse():
    parsed = parse_llm_analysis_strict(json.dumps({"verdict": "Успешно", "findings": [{"id": "f1", "summary": "ok"}]}))
    assert isinstance(parsed, LLMAnalysis)
    assert parsed.findings[0].verification is None and parsed.verification_summary is None


# ---- structured output --------------------------------------------------------------

def test_ask_llm_structured_requires_client_support(monkeypatch):
    monkeypatch.setattr(providers_module, "_llm_env_applied", True)
    monkeypatch.setattr(providers_module, "_get_gigachat_client", lambda *_a, **_k: object())
    with pytest.raises(RuntimeError, match="structured_output"):
        providers_module.ask_llm_structured("prompt", "{}", LLMAnalysis)


def test_ask_llm_structured_returns_schema_instance(monkeypatch):
    class FakeRunnable:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages):
            assert len(messages) == 2 and "prompt" in messages[1].content
            return {"verdict": "Успешно", "findings": [], "recommended_actions": []}

    class FakeClient:
        def with_structured_output(self, schema):
            return FakeRunnable(schema)

    monkeypatch.setattr(providers_module, "_llm_env_applied", True)
    monkeypatch.setattr(providers_module, "_get_gigachat_client", lambda *_a, **_k: FakeClient())
    monkeypatch.setattr(providers_module, "_wait_llm_slot", lambda gcfg: None)
    result = providers_module.ask_llm_structured("prompt", "{}", LLMAnalysis)
    assert isinstance(result, LLMAnalysis) and result.verdict == "Успешно"


def test_self_consistency_uses_structured_path_when_enabled(monkeypatch):
    calls = {"structured": 0, "text": 0}

    def fake_structured(user_prompt, data_context, schema, system_prompt=None):
        calls["structured"] += 1
        return schema(verdict="Успешно", findings=[], recommended_actions=[])

    monkeypatch.setattr(scoring_module, "structured_output_enabled", lambda: True)
    monkeypatch.setattr(scoring_module, "ask_llm_structured", fake_structured)
    monkeypatch.setattr(scoring_module, "ask_llm_with_text_data", lambda *a, **k: calls.__setitem__("text", calls["text"] + 1) or "{}")
    monkeypatch.setattr(scoring_module, "judge_candidates_with_llm", lambda *a, **k: ({0: {"overall": 1.0}}, {}))
    text, parsed, scores = scoring_module.llm_two_pass_self_consistency("prompt", json.dumps({"sections": []}), k=1, return_scores=True, domain_key="jvm")
    assert calls == {"structured": 1, "text": 0}
    assert isinstance(parsed, LLMAnalysis) and json.loads(text)["verdict"] == "Успешно"


VALID_ANALYSIS = json.dumps({"verdict": "Успешно", "findings": [], "recommended_actions": []}, ensure_ascii=False)


def test_self_consistency_skips_judge_for_one_candidate(monkeypatch):
    judge_calls = {"n": 0}

    def fake_judge(*_a, **_k):
        judge_calls["n"] += 1
        return {}, {}

    monkeypatch.setattr(scoring_module, "structured_output_enabled", lambda: False)
    monkeypatch.setattr(scoring_module, "ask_llm_with_text_data", lambda *_a, **_k: VALID_ANALYSIS)
    monkeypatch.setattr(scoring_module, "judge_candidates_with_llm", fake_judge)
    _text, parsed, scores = scoring_module.llm_two_pass_self_consistency(
        "prompt", json.dumps({"sections": []}), k=1, return_scores=True, domain_key="jvm"
    )
    assert judge_calls["n"] == 0
    assert isinstance(parsed, LLMAnalysis)
    assert scores["judge_meta"]["skipped"] is True
    assert scores["judge_meta"]["reason"] == "single_candidate"


def test_self_consistency_calls_judge_for_two_candidates(monkeypatch):
    judge_calls = {"n": 0}
    llm_cfg = scoring_module.CONFIG.setdefault("llm", {})
    sc_cfg = llm_cfg.setdefault("self_consistency", {})
    monkeypatch.setitem(llm_cfg, "self_consistency_k", 2)
    monkeypatch.setitem(sc_cfg, "max_candidates", 2)
    monkeypatch.setitem(sc_cfg, "pause_sec_between_calls", 0)

    def fake_judge(texts, *_a, **_k):
        judge_calls["n"] += 1
        assert len(texts) == 2
        return {0: {"overall": 0.4, "factual": 0.4, "completeness": 0.4, "specificity": 0.4}, 1: {"overall": 0.9, "factual": 0.9, "completeness": 0.9, "specificity": 0.9}}, {"domain_key": "jvm"}

    monkeypatch.setattr(scoring_module, "structured_output_enabled", lambda: False)
    monkeypatch.setattr(scoring_module, "ask_llm_with_text_data", lambda *_a, **_k: VALID_ANALYSIS)
    monkeypatch.setattr(scoring_module, "judge_candidates_with_llm", fake_judge)
    _text, parsed, scores = scoring_module.llm_two_pass_self_consistency(
        "prompt", json.dumps({"sections": []}), k=2, return_scores=True, domain_key="jvm"
    )
    assert judge_calls["n"] == 1
    assert isinstance(parsed, LLMAnalysis)
    assert scores["judge_meta"].get("skipped") is not True
    assert scores["selected_index"] == 1


def test_domain_worker_count_caps_by_max_concurrent():
    assert pipeline_module._domain_worker_count({}) == 4
    assert pipeline_module._domain_worker_count({"max_domain_workers": 8, "gigachat": {"max_concurrent": 3}}) == 3
    assert pipeline_module._domain_worker_count({"max_domain_workers": 1, "gigachat": {"max_concurrent": 4}}) == 1


def test_gigachat_usage_is_recorded(monkeypatch):
    providers_module.reset_usage()

    class Resp:
        content = VALID_ANALYSIS
        usage_metadata = {"input_tokens": 12, "output_tokens": 4}

    class Client:
        def invoke(self, _messages):
            return Resp()

    monkeypatch.setattr(providers_module, "_get_gigachat_client", lambda pcfg=None: Client())
    with providers_module.usage_domain("jvm"):
        text = providers_module._gigachat_call([{"role": "user", "content": "user"}], {}, "sys")
    assert "Успешно" in text
    snap = providers_module.snapshot_usage()
    assert snap["jvm"]["calls"] == 1
    assert snap["jvm"]["prompt_tokens"] == 12
    assert snap["jvm"]["completion_tokens"] == 4
    scores = providers_module.attach_usage_to_scores({"jvm": {"data_score": 0.5}, "final": {}})
    assert scores["jvm"]["usage"]["prompt_tokens"] == 12
    assert scores["final"]["run_usage"]["calls"] == 1
    assert scores["final"]["run_usage"]["prompt_tokens"] == 12
    providers_module.reset_usage()


def test_missing_usage_does_not_fail(monkeypatch):
    providers_module.reset_usage()

    class Resp:
        content = "ok"

    class Client:
        def invoke(self, _messages):
            return Resp()

    monkeypatch.setattr(providers_module, "_get_gigachat_client", lambda pcfg=None: Client())
    with providers_module.usage_domain("kafka"):
        assert providers_module._gigachat_call([{"role": "user", "content": "user"}], {}, "sys") == "ok"
    snap = providers_module.snapshot_usage()
    assert snap["kafka"]["calls"] == 1
    assert snap["kafka"]["prompt_tokens"] is None
    assert snap["kafka"]["completion_tokens"] is None
    providers_module.reset_usage()


def test_save_llm_results_writes_context(monkeypatch):
    from AI import db_store

    captured = {}

    def fake_batch(cur, sql_text, rows, page_size=100):
        captured["sql"] = sql_text
        captured["rows"] = rows

    monkeypatch.setattr(db_store, "execute_batch", fake_batch)
    monkeypatch.setattr(db_store, "_connect", lambda _cfg: RecordingConnection())
    monkeypatch.setattr(db_store, "_ensure_llm_reports_table", lambda *_a, **_k: None)

    class SqlStub:
        def format(self, *_a, **_k):
            return self

        def as_string(self, _cur):
            return "INSERT INTO llm_reports (context) VALUES"

    monkeypatch.setattr(db_store.sql, "SQL", lambda _text: SqlStub())

    db_store.save_llm_results(
        {
            "final": '{"verdict":"Успешно"}',
            "final_parsed": {"verdict": "Успешно"},
            "scores": {"final": {"usage": {"calls": 1}}},
            "contexts": {"final": {"domain": "final", "x": 1}},
        },
        {"run_name": "nightly-1"},
        {"schema": "public", "llm_table": "llm_reports"},
    )
    assert "context" in captured["sql"]
    final_row = captured["rows"][-1]
    context_json = final_row[10]
    payload = context_json.adapted if hasattr(context_json, "adapted") else context_json
    assert payload == {"domain": "final", "x": 1}


class _HttpResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text if text else (json.dumps(self._payload) if self._payload else "")
        self.content = self.text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._payload


def _grafana_cfg():
    return {
        "base_url": "http://grafana:3000",
        "verify_ssl": False,
        "auth": {"method": "basic", "username": "a", "password": "b"},
        "prometheus_datasource": {"uid": "prom-uid", "id": 9, "name": "Prom"},
        "influxdb_datasource": {"uid": "influx-uid", "id": 4},
    }


def test_promql_failure_does_not_zero_other_queries(monkeypatch):
    def fake_fetch(_url, _start, _end, query, _step, ef_config=None):
        if query == "bad":
            raise RuntimeError("proxy 403")
        return {"status": "success", "data": {"result": [{"metric": {"app": "a"}, "values": [[1.0, "1.5"]]}]}}

    monkeypatch.setattr(pipeline_module, "fetch_metric_series", fake_fetch)
    dfs = pipeline_module.fetch_and_aggregate_with_label_keys(
        "http://prom", 1.0, 2.0, ["bad", "good"], [["app"], ["app"]], "1m", "1T"
    )
    assert dfs[0].empty
    assert not dfs[1].empty
    assert float(dfs[1].iloc[0, 0]) == 1.5


def test_parse_fills_missing_verdict_rationale_from_findings():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "findings": [
                {
                    "id": "f1",
                    "summary": "p95 вырос до 420 мс на ступени 2",
                    "severity": "high",
                    "component": "orders",
                    "verification": {"status": "verified"},
                },
                {
                    "id": "f2",
                    "summary": "выдуманный p99 999 мс",
                    "severity": "high",
                    "component": "orders",
                    "verification": {"status": "unverified"},
                },
            ],
            "recommended_actions": [],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.verdict_rationale
    assert "Есть риски" in parsed.verdict_rationale
    assert "p95 вырос до 420 мс" in parsed.verdict_rationale
    assert "999" not in parsed.verdict_rationale


def test_sla_rationale_explains_risk_when_target_rps_passed():
    text = format_sla_rationale(
        "Есть риски",
        [
            {
                "name": "target_rps",
                "threshold": 200,
                "actual": 226,
                "passed": True,
                "message": "RPS 226.0 (stable_max) ≥ целевой 200",
            },
            {
                "name": "p95_latency",
                "threshold": 200,
                "actual": 340,
                "passed": False,
                "message": "p95 340 > 200",
            },
        ],
    )
    assert "Статус «Есть риски»" in text
    assert "целевой RPS достигнут" in text
    assert "p95" in text


def test_sla_secondary_metrics_use_p95_of_last_stable_step():
    index = pd.date_range(datetime.fromtimestamp(START_TS, tz=timezone.utc), periods=6, freq="5min")
    # Buckets are 5 minutes. The stable segment ends at minute 15, so the bucket
    # stamped 10:10 already contains the spike at 10:15 and must be excluded.
    cpu = pd.DataFrame({"node=a": [10.0, 50.0, 50.0, 90.0, 90.0, 90.0]}, index=index)
    stable_end = index[3]
    domain_data = {
        "lt_framework": {
            "pack": {
                "sections": [{
                    "label": "LT RPS",
                    "top_series": [{
                        "series": "sum_all",
                        "stable_max": 200.0,
                        "stable_window_start": str(index[0]),
                        "stable_window_end": str(stable_end),
                    }],
                }],
            },
            "labeled": [],
        },
        "hard_resources": {"labeled": [{"label": "Nodes: CPU usage (%) by node", "df": cpu}], "pack": {}},
    }
    sla = {
        "max_cpu_pct": 60,
        "cpu_query": "Nodes: CPU usage (%) by node",
        "max_performance_query": "LT RPS",
    }
    step_result = evaluate_sla(domain_data, sla, test_profile={"mode": "capacity"})
    cpu_check = step_result["checks"][0]
    assert cpu_check["passed"] is True
    assert cpu_check["actual"] == pytest.approx(50.0)
    assert "стабильной ступени ≈200 RPS" in cpu_check["message"]

    soak_result = evaluate_sla(domain_data, sla, test_profile={"mode": "stability"})
    soak_check = soak_result["checks"][0]
    assert soak_check["passed"] is False
    assert soak_check["actual"] == pytest.approx(90.0)
    assert "всего окна" in soak_check["message"]


def test_sla_cuts_off_at_last_report_step_that_holds():
    levels = [703.4, 844.6, 947.6, 1068.9, 1227.6, 1276.6, 1471.7, 1472.9]
    latencies = [100.0, 206.0, 127.0, 158.0, 187.0, 16736.0, 4716.0, 4228.0]
    rps_values: list[float] = []
    latency_values: list[float] = []
    for level, latency in zip(levels, latencies):
        rps_values.extend([level] * 40)
        latency_values.extend([latency] * 40)
    index = pd.date_range("2026-10-01T17:26:00Z", periods=len(rps_values), freq="1min")
    steps = []
    for offset, level in enumerate(levels):
        start = index[offset * 40]
        end = index[(offset + 1) * 40 - 1]
        steps.append({"index": offset + 1, "start_iso": str(start), "end_iso": str(end), "rps_level": level})
    domain_data = {
        "lt_framework": {
            "pack": {
                "load_steps": steps,
                "sections": [{
                    "label": "LT RPS",
                    "top_series": [{
                        "series": "all=",
                        "stable_max": 1667.5,
                        "step_segments": [{
                            "start": str(index[-12]),
                            "end": str(index[-1]),
                            "level": 1667.5,
                            "stable": True,
                        }],
                    }],
                }],
            },
            "labeled": [
                {"label": "LT RPS", "df": pd.DataFrame({"all=": rps_values}, index=index)},
                {"label": "p95", "df": pd.DataFrame({"group=a": latency_values}, index=index)},
            ],
        },
    }
    sla = {"target_rps": 1000, "max_p95_ms": 2000, "p95_query": "p95", "max_performance_query": "LT RPS"}
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "capacity"})
    by_name = {check["name"]: check for check in result["checks"]}
    assert by_name["target_rps"]["actual"] == pytest.approx(1227.6)
    assert by_name["p95_latency"]["passed"] is True
    assert by_name["p95_latency"]["actual"] == pytest.approx(187.0)
    assert result["stable_window"]["degraded_level"] == pytest.approx(1276.6)
    assert "1667" not in by_name["target_rps"]["message"]


def test_sla_window_falls_back_to_last_stable_step_where_latency_holds():
    # Step ≈1198 RPS (min 0-6) is healthy; the later flat step ≈1416 RPS (min 6-12)
    # is already a latency collapse: timeouts keep RPS flat, so RPS alone marks it stable.
    index = pd.date_range(datetime.fromtimestamp(START_TS, tz=timezone.utc), periods=12, freq="1min")
    latency = pd.DataFrame(
        {"group=::ADDRESSById": [60.0, 61.0, 61.0, 57.0, 66.0, 65.0, 461.0, 60001.0, 32731.0, 60003.0, 60001.0, 60001.0]},
        index=index,
    )
    segments = [
        {"start": str(index[0]), "end": str(index[5]), "level": 1198.4, "stable": True},
        {"start": str(index[6]), "end": str(index[11]), "level": 1415.8, "stable": True},
    ]
    domain_data = {
        "lt_framework": {
            "pack": {
                "sections": [{
                    "label": "LT RPS",
                    "top_series": [{"series": "all=", "stable_max": 1415.8, "step_segments": segments}],
                }],
            },
            "labeled": [{"label": "p95", "df": latency}],
        },
    }
    sla = {"target_rps": 1000, "max_p95_ms": 5000, "p95_query": "p95", "max_performance_query": "LT RPS"}
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "capacity"})
    by_name = {check["name"]: check for check in result["checks"]}
    assert result["verdict"] == "Успешно"
    assert by_name["target_rps"]["passed"] is True
    assert by_name["target_rps"]["actual"] == pytest.approx(1198.4)
    assert "≈1416 RPS нарушены: p95_latency" in by_name["target_rps"]["message"]
    assert by_name["p95_latency"]["passed"] is True
    assert by_name["p95_latency"]["actual"] < 100
    assert result["stable_window"]["level"] == pytest.approx(1198.4)
    assert result["stable_window"]["degraded_level"] == pytest.approx(1415.8)

    # A lower target still fails when every stable step breaks the SLA: the latest step is kept.
    latency_all_bad = latency.copy()
    latency_all_bad.iloc[:6] = 9000.0
    domain_data["lt_framework"]["labeled"] = [{"label": "p95", "df": latency_all_bad}]
    result_bad = evaluate_sla(domain_data, sla, test_profile={"mode": "capacity"})
    bad_by_name = {check["name"]: check for check in result_bad["checks"]}
    assert bad_by_name["target_rps"]["actual"] == pytest.approx(1415.8)
    assert bad_by_name["p95_latency"]["passed"] is False


# Balanced detector preset; production feeds it per-minute buckets.
BALANCED_STEP_CFG = {
    "step_detection_resample_sec": 15, "step_detection_smooth_sec": 60, "step_confirm_hold_sec": 180,
    "step_min_step_delta_rps": 8.0, "step_min_step_delta_pct": 0.08, "step_max_cv": 0.10,
    "step_max_slope_rps_per_min": 0.5, "step_max_within_step_drop_pct": 0.08, "step_drop_hold_sec": 120,
}


def _bucket_profile(values, freq="1min"):
    """Step detector with the balanced preset on RPS buckets starting at 10:00 UTC."""
    index = pd.date_range("2026-10-01T10:00:00Z", periods=len(values), freq=freq)
    series = pd.Series([float(v) for v in values], index=index)
    return pipeline_module._find_stable_peak_step_profile(series, min_stable_minutes=5.0, cfg=BALANCED_STEP_CFG), index


def _run_steps(values, index):
    return pipeline_module._derive_steps_for_run(
        lt_labeled=[{"label": "RPS", "df": pd.Series(values, index=index).to_frame("all")}],
        perf_label="RPS", stable_cfg={**BALANCED_STEP_CFG, "use_step_profile": True}, min_stable_minutes=5.0,
        start_ts=index[0].timestamp(), end_ts=(index[-1] + pd.Timedelta(minutes=1)).timestamp(), shift_hours=0,
    )


def test_plateau_after_rps_drop_is_not_stable_max():
    # Per-minute RPS of ODP-87633: steps up to ≈1597, then RPS falls to a flat ≈1400 (timeouts).
    rps = [
        22, 118, 235, 367, 500, 634, 721, 744, 744, 744, 744, 746, 792, 817, 817, 817, 817, 819, 871, 898,
        898, 898, 898, 900, 958, 989, 988, 988, 988, 990, 1055, 1088, 1088, 1088, 1088, 1089, 1162, 1198, 1199, 1198,
        1198, 1200, 1279, 1319, 1319, 1319, 1320, 1322, 1408, 1450, 1451, 1452, 1452, 1454, 1548, 1596, 1597, 1597, 1595, 1599,
        1673, 1450, 1400, 1431, 1398, 1384, 1384, 1751,
    ]
    index = pd.date_range("2026-09-22T15:37:00Z", periods=len(rps), freq="1min")
    profile = pipeline_module._find_stable_peak_step_profile(
        pd.Series([float(v) for v in rps], index=index), min_stable_minutes=5.0, cfg=BALANCED_STEP_CFG,
    )
    segments = profile["step_segments"]
    assert segments[-1]["stable"] is True and segments[-1]["after_drop"] is True
    assert segments[-1]["level"] == pytest.approx(1415.5)
    assert not any(seg["dip"] for seg in segments)
    assert first_rps_drop(segments) == index[61]
    # The ramp-in minute of each step is not instability: the last step before the fall is the maximum.
    last_rising_stable = [seg for seg in segments if seg["stable"] and not seg["after_drop"]][-1]
    assert profile["stable_max"] == pytest.approx(last_rising_stable["level"])
    assert profile["stable_max"] == pytest.approx(1596.5)
    assert profile["stable_max"] > segments[-1]["level"]

    healthy_latency = pd.DataFrame({"group=::ADDRESSById": [60.0] * len(rps)}, index=index)
    domain_data = {
        "lt_framework": {
            "pack": {"sections": [{"label": "LT RPS", "top_series": [{"series": "all=", "stable_max": profile["stable_max"], "step_segments": segments}]}]},
            "labeled": [{"label": "p95", "df": healthy_latency}],
        },
    }
    sla = {"target_rps": 1000, "max_p95_ms": 5000, "p95_query": "p95", "max_performance_query": "LT RPS"}
    result = evaluate_sla(domain_data, sla, test_profile={"mode": "capacity"})
    assert result["stable_window"]["level"] == pytest.approx(last_rising_stable["level"])


@pytest.mark.parametrize("position, value", [(15, 240.0), (25, 240.0), (25, 0.0)], ids=["spike", "dip", "empty-minute"])
def test_one_minute_outliers_are_not_rps_drops(position, value):
    # One odd minute inside a plateau (a burst, a slow minute, a metrics gap read as 0) is noise:
    # it must not mark the following steps "after a drop" nor cut the forecast there.
    rps = [100.0] * 10 + [200.0] * 10 + [300.0] * 10 + [400.0] * 10
    rps[position] = value
    profile, _ = _bucket_profile(rps)
    segments = profile["step_segments"]
    assert profile["stable_max"] == pytest.approx(400.0)
    assert not any(seg["after_drop"] or seg["dip"] or seg["drop_time"] for seg in segments)
    assert first_rps_drop(segments) is None
    if value:
        assert [round(seg["level"]) for seg in segments if seg["stable"]] == [100, 200, 300, 400]


def test_dip_that_cuts_a_plateau_becomes_dip_time():
    # A slow minute right before the next step still cuts that plateau, but RPS climbs back: not a drop.
    rps = [100.0] * 8 + [200.0] * 8 + [300.0] * 8 + [400.0] * 8
    rps[20] = 240.0
    profile, index = _bucket_profile(rps)
    segments = profile["step_segments"]
    cut = [seg for seg in segments if seg["dip_time"]]
    assert len(cut) == 1 and pd.Timestamp(cut[0]["dip_time"]) == index[20]
    assert cut[0]["drop_time"] is None and not cut[0]["dip"]
    assert not any(seg["after_drop"] or seg["dip"] for seg in segments)
    assert first_rps_drop(segments) is None
    assert profile["stable_max"] == pytest.approx(400.0)


def test_recovered_dip_is_flagged_and_never_stable_max():
    # RPS sags to 180 for 8 minutes and returns to 300, then the load goes on to 400.
    rps = [100.0] * 10 + [200.0] * 10 + [300.0] * 10 + [180.0] * 8 + [300.0] * 10 + [400.0] * 10
    profile, index = _bucket_profile(rps)
    segments = profile["step_segments"]
    assert [round(seg["level"]) for seg in segments if seg["dip"]] == [240, 180, 240]
    assert not any(seg["after_drop"] for seg in segments)
    assert profile["stable_max"] == pytest.approx(400.0)

    run = _run_steps(rps, index)
    steps = run.steps
    assert [round(step.rps_level) for step in steps] == [100, 200, 300, 180, 300, 400]
    dip_step = steps[3].to_dict()
    assert steps[3].dip is True and steps[3].stable is False
    assert dip_step["dip"] is True and "after_drop" not in dip_step
    assert steps[2].drop_time is None
    assert run.rps_drop is None


def test_real_drop_after_recovered_dip_is_found():
    # Same run, but after 400 RPS the system gives up: RPS falls to 288 and stays there.
    rps = [100.0] * 10 + [200.0] * 10 + [300.0] * 10 + [180.0] * 8 + [300.0] * 10 + [400.0] * 10 + [288.0] * 8
    profile, index = _bucket_profile(rps)
    assert [round(seg["level"]) for seg in profile["step_segments"] if seg["after_drop"]] == [344, 288]
    assert profile["stable_max"] == pytest.approx(400.0)

    run = _run_steps(rps, index)
    steps = run.steps
    assert run.rps_drop is not None and abs((run.rps_drop - index[58]).total_seconds()) <= 60
    assert steps[3].dip is True and steps[3].after_drop is False
    assert "drop_iso" in steps[-2].to_dict()
    assert steps[-1].to_dict()["after_drop"] is True


def test_one_point_spike_at_five_minute_cadence():
    # 5-minute buckets: one bucket is a whole confirmation window, and it is still not a load step.
    rps = [100.0] * 4 + [200.0] * 4 + [300.0] * 4 + [400.0] * 4
    rps[9] = 360.0
    profile, _ = _bucket_profile(rps, freq="5min")
    segments = profile["step_segments"]
    assert not any(seg["after_drop"] or seg["dip"] or seg["drop_time"] for seg in segments)
    assert first_rps_drop(segments) is None
    assert profile["stable_max"] == pytest.approx(400.0)


@pytest.mark.parametrize(
    "below, run_len, expected",
    [
        ([False, False, True, True, False, False], 2, None),
        ([False, True, True, False, False, True, True], 2, 5),
        ([False, False, True, True, False, True, True], 2, 2),
        ([True, False, True], 1, 2),
        ([False, True, False], 1, None),
        ([], 2, None),
    ],
)
def test_persistent_drop_start_skips_runs_that_recover(below, run_len, expected):
    assert pipeline_module._persistent_drop_start(below, run_len) == expected


def test_plateau_core_trims_ramp_edges_only():
    index = pd.date_range("2026-10-01T10:00:00Z", periods=7, freq="1min")
    plateau = pd.Series([150.0, 200.0, 201.0, 199.0, 200.0, 200.0, 230.0], index=index)
    assert list(pipeline_module._plateau_core(plateau, 200.0, 8.0).index) == list(index[1:6])
    ramp = pd.Series([100.0, 120.0, 200.0, 205.0], index=index[:4])
    assert pipeline_module._plateau_core(ramp, 160.0, 8.0).equals(ramp)


def _decline_segments(levels, down_refs=None, drops=None):
    """Hand-made detector segments; ``down_refs`` and ``drops`` map a position to a step-down reference / drop time."""
    segments = [{"level": float(level), "drop_time": None, "dip_time": None, "after_drop": False, "dip": False} for level in levels]
    for position, reference in (down_refs or {}).items():
        segments[position]["_down_ref"] = float(reference)
    for position, moment in (drops or {}).items():
        segments[position].update(drop_time=moment, _level_ref=segments[position]["level"])
    return segments


def test_classify_declines_marks_recovered_dips_and_unrecovered_drops():
    recovered = _decline_segments([300, 240, 180, 300, 400], down_refs={1: 300})
    pipeline_module._classify_declines(recovered, 8.0, 0.08)
    assert [seg["dip"] for seg in recovered] == [False, True, True, False, False]
    assert not any(seg["after_drop"] for seg in recovered)

    partial = _decline_segments([600, 420, 520], down_refs={1: 600})
    pipeline_module._classify_declines(partial, 8.0, 0.08)
    assert [seg["after_drop"] for seg in partial] == [False, True, True]
    assert not any(seg["dip"] for seg in partial)

    moment = "2026-10-01 10:05:00+00:00"
    dip_inside = _decline_segments([300, 300], drops={0: moment})
    pipeline_module._classify_declines(dip_inside, 8.0, 0.08)
    assert dip_inside[0]["drop_time"] is None and dip_inside[0]["dip_time"] == moment
    assert not any(seg["dip"] or seg["after_drop"] for seg in dip_inside)

    lost = _decline_segments([300, 200], drops={0: moment})
    pipeline_module._classify_declines(lost, 8.0, 0.08)
    assert lost[0]["drop_time"] == moment and lost[0]["after_drop"] is False
    assert lost[1]["after_drop"] is True


def test_first_rps_drop_ignores_recovered_dips():
    segments = [
        {"start": "2026-10-01T10:00:00Z", "end": "2026-10-01T10:09:00Z", "level": 300.0, "stable": True},
        {"start": "2026-10-01T10:10:00Z", "end": "2026-10-01T10:14:00Z", "level": 200.0, "dip": True, "drop_time": "2026-10-01T10:12:00Z"},
        {"start": "2026-10-01T10:15:00Z", "end": "2026-10-01T10:24:00Z", "level": 300.0, "dip_time": "2026-10-01T10:20:00Z"},
    ]
    assert first_rps_drop(segments) is None
    lost = segments + [{"start": "2026-10-01T10:25:00Z", "end": "2026-10-01T10:30:00Z", "level": 150.0, "after_drop": True}]
    assert first_rps_drop(lost) == pd.Timestamp("2026-10-01T10:25:00Z")


def test_step_flags_reach_tail_drop_and_report_rows():
    # A 100 step with a short recovered dip in its tail, a long recovered dip that is a step of
    # its own, a 150 step, then a fall RPS never recovers from (a short tail, then a step).
    start = datetime.fromtimestamp(START_TS, tz=timezone.utc)
    plan = [
        (100.0, 10, {}), (60.0, 3, {"dip": True}), (100.0, 10, {}), (70.0, 7, {"dip": True}),
        (150.0, 10, {}), (40.0, 3, {"after_drop": True}), (30.0, 7, {"after_drop": True}),
    ]
    segments, minute = [], 0
    for level, minutes, flags in plan:
        seg_start = start + pd.Timedelta(minutes=minute)
        segments.append({
            "start": str(seg_start), "end": str(seg_start + pd.Timedelta(minutes=minutes - 1)),
            "level": level, "duration_min": float(minutes), "stable": True, **flags,
        })
        minute += minutes
    steps = derive_load_steps(segments, START_TS, START_TS + minute * 60, shift_hours=3, min_step_minutes=5.0)
    assert [step.rps_level for step in steps] == [100.0, 100.0, 70.0, 150.0, 30.0]
    assert steps[0].drop_time is None
    assert steps[3].drop_time == pd.Timestamp(segments[5]["start"])
    expected = [(False, False), (False, False), (False, True), (False, False), (True, False)]
    assert [(step.after_drop, step.dip) for step in steps] == expected
    assert [step.stable for step in steps] == [True, True, False, True, False]
    from_frames = load_step_report_from_frames(steps, [])["steps"]
    from_context = load_step_report_from_context({"load_steps": [step.to_dict() for step in steps]})["steps"]
    for rows in (from_frames, from_context):
        assert [(row["after_drop"], row["dip"]) for row in rows] == expected


def test_sla_window_skips_recovered_dip_steps():
    # 300 RPS holds, RPS then dips to 180 and recovers, 400 RPS breaks the latency SLA.
    # The SLA step is 300: a dip is not a load level, and it must not fail the 250 RPS target.
    levels, latencies, minutes = [300.0, 180.0, 400.0], [100.0, 100.0, 5000.0], [10, 8, 10]
    rps_values, latency_values, bounds = [], [], []
    for level, latency, length in zip(levels, latencies, minutes):
        rps_values += [level] * length
        latency_values += [latency] * length
    index = pd.date_range("2026-10-01T10:00:00Z", periods=len(rps_values), freq="1min")
    offset = 0
    for length in minutes:
        bounds.append((index[offset], index[offset + length - 1]))
        offset += length
    labeled = [
        {"label": "LT RPS", "df": pd.DataFrame({"all=": rps_values}, index=index)},
        {"label": "p95", "df": pd.DataFrame({"group=a": latency_values}, index=index)},
    ]
    sla = {"target_rps": 250, "max_p95_ms": 2000, "p95_query": "p95", "max_performance_query": "LT RPS"}
    steps = [
        {"index": position + 1, "start_iso": str(st), "end_iso": str(en), "rps_level": level, **({"dip": True} if level == 180.0 else {})}
        for position, (level, (st, en)) in enumerate(zip(levels, bounds))
    ]
    by_steps = evaluate_sla({"lt_framework": {"pack": {"load_steps": steps}, "labeled": labeled}}, sla, test_profile={"mode": "capacity"})
    assert by_steps["stable_window"]["level"] == pytest.approx(300.0)
    assert by_steps["stable_window"]["degraded_level"] == pytest.approx(400.0)
    assert by_steps["verdict"] == "Успешно"

    segments = [
        {"start": str(st), "end": str(en), "level": level, "stable": True, "dip": level == 180.0}
        for level, (st, en) in zip(levels, bounds)
    ]
    pack = {"sections": [{"label": "LT RPS", "top_series": [{"series": "all=", "stable_max": 300.0, "step_segments": segments}]}]}
    by_segments = evaluate_sla({"lt_framework": {"pack": pack, "labeled": labeled}}, sla, test_profile={"mode": "capacity"})
    assert by_segments["stable_window"]["level"] == pytest.approx(300.0)


def test_anthropic_payload_is_shrunk_to_fit_minimax_window():
    from AI.providers import _estimate_tokens, _fit_anthropic_payload

    context = {
        "deterministic_sla": {"verdict": "Успешно"},
        "domains": {"jvm": {"sections": [{"label": "heap", "top_series": [{"series": "a", "mean": 1}]}], "step_table": [{"label": "heap"}]}},
    }
    huge = "инструкция итогового анализа\n\n" + json.dumps(context, ensure_ascii=False) + (" " * 500_000)
    fitted, max_tokens = _fit_anthropic_payload("system", huge, 32768, {"api_base_url": "https://api.minimax.io/anthropic"})
    assert len(fitted) < len(huge)
    assert _estimate_tokens(fitted) + max_tokens < 204_800
    assert "инструкция итогового анализа" in fitted
    assert max_tokens == 32768


def test_fit_to_window_rejects_output_budget_with_no_room_for_input():
    from AI.providers import LLMBudgetError, _fit_to_window

    with pytest.raises(LLMBudgetError):
        _fit_to_window("system", "user", 200_000, 204_800, len)


def test_truncated_answer_is_retried_once_with_double_budget(monkeypatch):
    from AI.providers import LLMOutputTruncated, ask_llm_with_text_data

    calls = []

    def fake(_provider, _messages, pcfg, _system):
        calls.append(pcfg.get("_output_tokens_override"))
        if len(calls) == 1:
            raise LLMOutputTruncated("anthropic", 1000, 1000)
        return '{"verdict":"Успешно"}'

    monkeypatch.setattr(providers_module, "_call_provider", fake)
    monkeypatch.setattr(providers_module, "_wait_llm_slot", lambda *_a, **_k: None)
    assert "Успешно" in ask_llm_with_text_data("prompt", "", system_prompt="ok")
    assert calls == [None, 2000]


def test_second_truncation_is_not_retried_again(monkeypatch):
    from AI.providers import LLMOutputTruncated, ask_llm_with_text_data

    monkeypatch.setattr(providers_module, "_call_provider", lambda *_a, **_k: (_ for _ in ()).throw(LLMOutputTruncated("anthropic", 1000, 1000)))
    monkeypatch.setattr(providers_module, "_wait_llm_slot", lambda *_a, **_k: None)
    with pytest.raises(LLMOutputTruncated):
        ask_llm_with_text_data("prompt", "", system_prompt="ok")


def test_unusable_llm_answer_becomes_analysis_error(monkeypatch):
    from AI.providers import LLMOutputTruncated

    monkeypatch.setattr(scoring_module, "structured_output_enabled", lambda: False)
    monkeypatch.setattr(scoring_module, "ask_llm_with_text_data", lambda *_a, **_k: (_ for _ in ()).throw(LLMOutputTruncated("anthropic", 32768, 32768)))
    _text, parsed = scoring_module.llm_two_pass_self_consistency("prompt", "{}", k=1, domain_key="final")
    assert parsed.analysis_error is not None
    assert parsed.analysis_error.reason == "output_truncated"
    assert parsed.findings == []
