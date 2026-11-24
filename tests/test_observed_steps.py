# SPDX-License-Identifier: Apache-2.0
"""Generator markers through the report pipeline, SLA, charts and export."""

import json

import pandas as pd
import pytest

from tests.test_report_without_llm import START_TS, _cfg, _stub_report_inputs, pipeline_module
from tests.test_report_markers import REPORT_JS, js_markers
from tests.js_harness import run_js
from AI import report_markers
from AI.i18n import LocalizedError
from AI.observed_steps import observed_load_steps, validate_observed_steps
from AI.sla_evaluator import evaluate_sla
from loadlens_app import confluence_export
from confluence_manager.update_confluence_template import render_llm_html, render_llm_markdown


def markers():
    start_ms = int(START_TS * 1000)
    return [{"name": f"stage {i + 1}", "start_ms": start_ms + i * 35000,
             "end_ms": start_ms + (i + 1) * 35000, "planned_rps": (i + 1) * 2} for i in range(8)]


def _report_for_markers(monkeypatch, intervals, rps, p95, minimum):
    start = pd.Timestamp(START_TS, unit="s", tz="UTC")
    index = pd.date_range(start, periods=len(rps), freq="5s")
    frames = [pd.DataFrame({"rps": rps}, index=index), pd.DataFrame({"latency": p95}, index=index)]
    _stub_report_inputs(monkeypatch, "step")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *a, **k: frames)
    saved = {}
    monkeypatch.setattr(pipeline_module, "save_llm_results", lambda results, **kwargs: saved.update(results))
    cfg = _cfg("step")
    cfg["sla"].update(min_stable_minutes=minimum, target_rps=16)
    pipeline_module.uploadFromLLM(
        START_TS, intervals[-1]["end_ms"] / 1000, save_to_db=True, only_collect=True, ef_config=cfg,
        run_meta={"test_type": "step", "run_tag": "demo-01", "observed_steps": intervals},
        active_domains=["lt_framework"], report_language="en",
    )
    return saved


@pytest.mark.parametrize("minimum", [0.5, 5])
def test_generator_steps_drive_report_sla_and_both_charts(monkeypatch, minimum):
    intervals = markers()
    # The last level holds 12 of the planned 16 RPS. The sample at the run end belongs outside it.
    rps = [(i // 7 + 1) * 2 for i in range(49)] + [12] * 7 + [999]
    p95 = [40 + i // 7 for i in range(56)] + [999]
    saved = _report_for_markers(monkeypatch, intervals, rps, p95, minimum)
    table = saved["load_step_table"]
    steps = table["steps"]
    assert table["detection"]["run_tag"] == "demo-01"
    assert [s["label"] for s in steps] == [m["name"] for m in intervals]
    assert [s["latency_p95"] for s in steps] == list(range(40, 48))
    assert [s["rps"] for s in steps] == rps[:56:7]
    assert steps[-1]["planned_rps"] == 16 and steps[-1]["rps"] == 12
    checks = {c["name"]: c for c in saved["sla_checks"]}
    if minimum == 5:
        assert checks["p95_latency"]["actual"] == 47
        assert not any(s["stable"] for s in steps)
        assert checks["target_rps"]["passed"] is None and saved["sla_window"] is None
        assert saved["final_parsed"]["peak_performance"]["max_rps"] is None
    else:
        assert all(s["stable"] for s in steps)
        assert checks["p95_latency"]["actual"] == 46
        assert checks["target_rps"]["passed"] is False and checks["target_rps"]["actual"] == 14
        assert steps[-2]["selected"] and saved["sla_window"]["start"] == steps[-2]["start_iso"]
    row = {"parsed": saved["final_parsed"]}
    annotations = report_markers.markers_for(row)
    assert [m["t"] for m in annotations if m["kind"] == "tick"] == [
        intervals[0]["start_ms"], *[s["end_ms"] for s in intervals]]
    assert js_markers(row) == annotations
    assert "demo-01" in confluence_export.load_step_expand(table)
    assert "stage 8" in confluence_export.load_step_expand(table)
    for render in (render_llm_markdown, render_llm_html):
        exported = render(saved["final_parsed"], domain="final", lang="en")
        assert "demo-01" in exported and "stage 8" in exported and "Planned 16 RPS" in exported


def test_generator_ceiling_requires_configured_latency_data(monkeypatch):
    saved = _report_for_markers(monkeypatch, markers()[:1], [16] * 7, [float("nan")] * 7, 0.5)
    assert saved["load_step_table"]["steps"][0]["rps"] == 16
    assert saved["sla_window"] is None
    assert saved["final_parsed"]["peak_performance"]["max_rps"] is None


def test_degradation_time_uses_the_marked_step_when_load_steps_down(monkeypatch):
    intervals = markers()[:4]
    saved = _report_for_markers(monkeypatch, intervals, [16] * 7 + [6] * 7 + [14] * 7 + [2] * 7,
                                [500] * 7 + [50] * 21, 0.5)
    steps = saved["load_step_table"]["steps"]
    peak = saved["final_parsed"]["peak_performance"]
    assert peak["max_rps"] == 14 and peak["max_time"] == steps[2]["start_iso"][11:16]
    assert peak["drop_time"] == steps[0]["start_iso"][11:16]


@pytest.mark.parametrize("change,key,index", [
    ({"start_ms": -1}, "bounds", 2), ({"end_ms": 10**15}, "bounds", 2),
    ({"start_ms": False}, "bounds", 2), ({"end_ms": 0}, "bounds", 2),
    ({"name": " "}, "name", 2), ({"planned_rps": -1}, "rate", 2),
    ({"start_ms": int(START_TS * 1000)}, "overlap", 2),
])
def test_invalid_step_identifies_its_position(change, key, index):
    intervals = markers()
    intervals[index - 1].update(change)
    with pytest.raises(LocalizedError) as error:
        validate_observed_steps(intervals, intervals[0]["start_ms"], intervals[-1]["end_ms"], "step")
    assert error.value.key.endswith("." + key) and error.value.params["index"] == index


def test_profile_leaves_gaps_between_generator_intervals():
    intervals = markers()[:2]
    intervals[0]["end_ms"] -= 5000
    steps = [{"source": "generator", "rps": i + 1, "index": i + 1,
              "start_iso": pd.Timestamp(s["start_ms"], unit="ms", tz="UTC").isoformat(),
              "end_iso": pd.Timestamp(s["end_ms"], unit="ms", tz="UTC").isoformat()} for i, s in enumerate(intervals)]
    profile = run_js(REPORT_JS, f"LL.charts.stepProfileModel({{steps: {json.dumps(steps)}}})")
    assert profile["steps"][0]["end"] == intervals[0]["end_ms"]
    assert profile["steps"][0]["end"] < profile["steps"][1]["start"]


def test_fractional_boundaries_and_missing_rps_do_not_fall_back_to_run_peaks():
    index = pd.date_range(pd.Timestamp(START_TS, unit="s", tz="UTC"), periods=9, freq="5s")
    marker = markers()[0]
    marker["start_ms"] += 500
    marker["end_ms"] -= 500
    step, = observed_load_steps([marker], None, START_TS, -2, 0.25)
    assert step.to_dict()["start_iso"].endswith(".500000-02:00")
    # Both edge buckets cross the markers. The complete bins have a 2% error rate.
    errors = pd.DataFrame({"errors": [99, 2, 2, 2, 2, 2, 99, 99, 99]}, index=index)
    pack = {"load_steps": [step.to_dict()], "sections": [{"label": "rps", "top_series": [
        {"series": "total", "stable_max": 999, "max": 999}]}]}
    result = evaluate_sla({"lt_framework": {"pack": pack, "labeled": [{"label": "err", "df": errors}]}},
                          {"target_rps": 16, "max_performance_query": "rps", "max_error_rate_pct": 1,
                           "error_rate_query": "err"}, lang="en")
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["target_rps"]["passed"] is None and checks["target_rps"]["actual"] is None
    assert checks["error_rate"]["actual"] == 2 and checks["error_rate"]["passed"] is False
    assert result["verdict"] == "fail" and result["stable_window"] is None
