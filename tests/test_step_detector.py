# SPDX-License-Identifier: Apache-2.0
"""Automatic load step detector (AI/step_detector.py): windows from the cadence, threshold from the plateau noise.

Synthetic series pin the rules; ``tests/fixtures/step_series`` holds RPS series recorded from real runs (times as
offsets, no names) where presets used to merge or split the steps (plans/auto-step-detector-plan.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from AI import pipeline as pipeline_module
from AI import step_detector
from AI.context_pack import first_rps_drop, load_step_report_from_frames
from AI.pipeline import build_context_pack

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "step_series"
T0 = pd.Timestamp("2026-10-01T10:00:00Z")


def _series(values, freq="1min"):
    return pd.Series([float(v) for v in values], index=pd.date_range(T0, periods=len(values), freq=freq))


def _noisy(levels, samples, noise, seed=7, freq="1min"):
    rng = np.random.default_rng(seed)
    values = np.concatenate([np.full(samples, float(level)) for level in levels])
    return _series(values * (1.0 + rng.normal(0.0, noise, len(values))), freq=freq)


def _fixture(name):
    data = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return pd.Series([v for _, v in data["points"]], index=[T0 + pd.Timedelta(seconds=o) for o, _ in data["points"]])


def _minute(moment):
    return (pd.Timestamp(moment) - T0).total_seconds() / 60.0


def _stable_levels(profile):
    return [round(seg["level"]) for seg in profile["step_segments"] if seg["stable"]]


# ---- windows and threshold -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq, minutes, hold",
    [("1min", 5.0, 3), ("3min", 5.0, 2), ("5min", 5.0, 2), ("1min", 10.0, 5), ("15s", 5.0, 10)],
)
def test_the_confirmation_window_follows_the_cadence_and_the_shortest_step(freq, minutes, hold):
    params = step_detector.detector_params(_series([100.0] * 40, freq=freq), minutes)
    assert params is not None and params.hold == hold
    assert params.sample_sec == pd.Timedelta(freq).total_seconds()


def test_the_threshold_comes_from_the_plateau_noise():
    # One plateau with 5 % noise stays one step: the threshold rises with the noise.
    noisy = step_detector.detect_steps(_noisy([1000], 60, 0.05), 5.0)
    assert noisy.params.thr_rel > 0.10
    assert _stable_levels(noisy.profile) == [996]
    # Steps of +10 % with 1 % noise are all found: a quiet plateau keeps the floor of 2.5 %.
    quiet = step_detector.detect_steps(_noisy([1000, 1100, 1210, 1331], 10, 0.01), 5.0)
    assert quiet.params.thr_rel == pytest.approx(step_detector.FLOOR_PCT)
    assert _stable_levels(quiet.profile) == [996, 1097, 1207, 1324]


def test_minute_buckets_are_not_smoothed_so_a_ten_percent_step_is_seen_whole():
    # 3-minute buckets with a partial sample on every transition: the presets smoothed over two samples and saw
    # 9.3-9.8 % instead of 10 %, below the 10 % of "strict", and merged all steps into one.
    values = []
    for previous, level in zip((None, 744.0, 817.0, 898.0), (744.0, 817.0, 898.0, 988.0)):
        if previous is not None:
            values.append(previous + 0.4 * (level - previous))
        values += [level] * 10
    profile = step_detector.find_step_profile(_series(values, freq="3min"), 5.0)
    segments = profile["step_segments"]
    assert [round(seg["level"]) for seg in segments] == [744, 817, 898, 988]
    assert all(seg["stable"] for seg in segments)
    # The ramp sample opens the next step; its plateau starts after it.
    second = segments[1]
    assert _minute(second["start"]) == 30.0 and _minute(second["plateau_start"]) == 33.0


def test_a_series_without_plateaus_has_no_steps():
    # A continuous ramp of two 5-minute samples per load level (the step is shorter than the data can show).
    values = [700.0 + 70.0 * i for i in range(24)]
    found = step_detector.detect_steps(_series(values, freq="5min"), 5.0)
    assert found.profile is None and found.params.reason == step_detector.REASON_NO_PLATEAUS
    run = _run(values, freq="5min")
    assert run.steps[0].source == "time_buckets"
    assert run.detection["reason"] == "no_plateaus" and run.detection["sample_sec"] == 300.0


def test_swinging_rps_is_not_a_stable_step():
    # The last load level never holds: RPS swings between 1757 and 1500 every minute (low cv, no plateau).
    values = [1451.0] * 10 + [1597.0] * 10 + [1757.0 if i % 2 == 0 else 1500.0 for i in range(12)]
    profile = step_detector.find_step_profile(_series(values), 5.0)
    assert profile["stable_max"] == pytest.approx(1597.0)
    tail = profile["step_segments"][-1]
    assert tail["stable"] is False and tail["coherent_share"] < step_detector.COHERENT_SHARE


def test_one_level_run_and_unstable_run_say_why_they_have_no_steps():
    one_level = _run([500.0] * 30)
    assert one_level.steps[0].source == "time_buckets" and one_level.detection["reason"] == "few_steps"
    swinging = _run([900.0 if i % 2 == 0 else 600.0 for i in range(30)])
    assert swinging.steps[0].source == "time_buckets"
    assert swinging.detection["reason"] in ("no_stable_step", "no_plateaus")


def _run(values, freq="1min"):
    index = pd.date_range(T0, periods=len(values), freq=freq)
    step = pd.Timedelta(freq)
    return pipeline_module._derive_steps_for_run(
        lt_labeled=[{"label": "RPS", "df": pd.Series(values, index=index).to_frame("all")}],
        perf_label="RPS", stable_cfg={"use_step_profile": True}, min_stable_minutes=5.0,
        start_ts=index[0].timestamp(), end_ts=(index[-1] + step).timestamp(), shift_hours=0,
    )


# ---- what the report keeps ---------------------------------------------------------------------------------


def test_the_detection_goes_with_the_steps_the_table_and_the_lt_pack():
    values = [100.0] * 10 + [200.0] * 10 + [300.0] * 10
    run = _run(values)
    assert [round(step.rps_level) for step in run.steps] == [100, 200, 300]
    assert run.detection == {
        "method": "auto", "sample_sec": 60.0, "hold_samples": 3, "noise_pct": 0.0, "threshold_pct": 2.5,
        "plateau_share": 1.0, "reason": None, "min_step_minutes": 5.0,
    }
    labeled = [{"label": "RPS", "df": _series(values).to_frame("all")}]
    table = load_step_report_from_frames(run.steps, labeled, rps_query="RPS", detection=run.detection)
    assert table["detection"] == run.detection
    assert "detection" not in load_step_report_from_frames(run.steps, labeled, rps_query="RPS")
    pack = build_context_pack(labeled, top_n=3, stable_detection_cfg={"use_step_profile": True}, step_detection=run.detection)
    assert pack["stable_detection"]["use_step_profile"] is True and pack["stable_detection"]["hold_samples"] == 3


def test_a_detector_that_did_not_run_leaves_no_detection():
    index = pd.date_range(T0, periods=30, freq="1min")
    run = pipeline_module._derive_steps_for_run(
        lt_labeled=[{"label": "RPS", "df": pd.Series([100.0] * 30, index=index).to_frame("all")}],
        perf_label="RPS", stable_cfg={"use_step_profile": False}, min_stable_minutes=5.0,
        start_ts=index[0].timestamp(), end_ts=index[-1].timestamp() + 60, shift_hours=0,
    )
    assert run.detection is None and run.steps[0].source == "time_buckets"


def test_retired_preset_keys_are_listed_and_dropped():
    sla = {"target_rps": 100, "step_detection_preset": "strict", "step_min_step_delta_pct": 0.1, "min_stable_minutes": 5}
    assert step_detector.retired_sla_keys(sla) == ["step_detection_preset", "step_min_step_delta_pct"]
    assert step_detector.without_retired_sla_keys(sla) == {"target_rps": 100, "min_stable_minutes": 5}
    assert step_detector.retired_sla_keys(None) == []


def test_retired_keys_in_settings_are_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(pipeline_module, "_RETIRED_KEYS_WARNED", set())
    sla = {"step_detection_preset": "strict", "step_max_cv": 0.08}
    with caplog.at_level("WARNING", logger=pipeline_module.logger.name):
        pipeline_module._warn_retired_step_keys(sla)
        pipeline_module._warn_retired_step_keys(sla)
    warnings = [record.getMessage() for record in caplog.records if "are ignored" in record.getMessage()]
    assert len(warnings) == 1 and "step_detection_preset" in warnings[0] and "step_max_cv" in warnings[0]


# ---- recorded runs -----------------------------------------------------------------------------------------


def test_ten_steps_at_three_minutes_are_ten_steps():
    # The report of this run had two steps: "strict" (10 %) merged nine steps of +10 % into one of 4.5 hours.
    profile = step_detector.find_step_profile(_fixture("ten_steps_3min"), 5.0)
    assert _stable_levels(profile) == [744, 817, 898, 988, 1088, 1198, 1320, 1452, 1597]
    assert profile["stable_max"] == pytest.approx(1597.3, abs=0.5)
    # The last target (1757) never held: the swinging tail is unstable, the load ends at minute 318.
    drop = first_rps_drop(profile["step_segments"])
    assert drop is not None and 312.0 <= _minute(drop) <= 318.0


def test_short_steps_keep_the_burst_minute_before_the_fall():
    profile = step_detector.find_step_profile(_fixture("short_steps_1min"), 5.0)
    assert _stable_levels(profile)[:9] == [744, 817, 898, 988, 1088, 1198, 1319, 1451, 1597]
    tail = profile["step_segments"][-1]
    assert tail["after_drop"] is True and round(tail["level"]) == 1400
    # Minute 60 is a burst above the last plateau; RPS falls from minute 61.
    assert _minute(first_rps_drop(profile["step_segments"])) == 61.0


def test_an_unstable_last_level_is_not_the_maximum():
    profile = step_detector.find_step_profile(_fixture("unstable_tail_5min"), 5.0)
    assert _stable_levels(profile) == [811, 974, 1168]
    assert profile["step_segments"][-1]["stable"] is False
    assert profile["stable_max"] == pytest.approx(1167.8, abs=0.5)


def test_a_ramp_of_two_samples_per_level_has_no_plateaus():
    found = step_detector.detect_steps(_fixture("no_plateaus_5min"), 5.0)
    assert found.profile is None and found.params.reason == step_detector.REASON_NO_PLATEAUS


def test_steps_of_a_few_rps_are_steps():
    # 2 -> 4 -> 6 -> 7 RPS: the floor of 1 RPS, not a percentage of 8 RPS, decides.
    profile = step_detector.find_step_profile(_fixture("low_rps_1min"), 5.0)
    assert [round(seg["level"]) for seg in profile["step_segments"]] == [2, 4, 6, 7]
    assert _stable_levels(profile) == [7]


def test_a_swing_after_the_steps_does_not_cut_the_run_early():
    profile = step_detector.find_step_profile(_fixture("swinging_tail_3min"), 5.0)
    assert _stable_levels(profile)[:7] == [744, 817, 898, 988, 1088, 1198, 1320]
    assert any(seg["dip"] for seg in profile["step_segments"])
    assert not any(seg["stable"] and round(seg["level"]) >= 1700 for seg in profile["step_segments"])
    # The load ends at minute 315; the presets cut the run 21 minutes before it, the detector at most 9.
    assert _minute(first_rps_drop(profile["step_segments"])) >= 306.0


# ---- the sentence under the step row -----------------------------------------------------------------------

REPORT_JS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
DETECTION = {"method": "auto", "sample_sec": 180.0, "hold_samples": 2, "noise_pct": 0.012, "threshold_pct": 2.5,
             "plateau_share": 0.94, "reason": None, "min_step_minutes": 5.0}


def _step_table(detection):
    steps = [{"index": 1, "label": "Step 1", "start_iso": "2026-10-01T10:00:00+03:00", "end_iso": "2026-10-01T10:30:00+03:00", "rps": 744}]
    table = {"steps": steps, "latency_query": None, "latency_series": None}
    if detection is not None:
        table["detection"] = detection
    return table


@pytest.mark.parametrize(
    "locale, detection, expected",
    [
        ("ru", DETECTION, "Ступени найдены по ряду RPS: шаг данных 3 мин, шум плато 0,01 %, минимальный скачок ступени 2,5 %."),
        ("en", DETECTION, "Steps found from the RPS series: data step 3 min, plateau noise 0.01%, smallest step jump 2.5%."),
        ("ru", {**DETECTION, "sample_sec": 300.0, "reason": "no_plateaus"},
         "В ряде RPS нет плато при шаге данных 5 мин: отчёт разбит на равные интервалы."),
        ("en", {**DETECTION, "reason": "few_steps", "min_step_minutes": 7.5},
         "Fewer than two steps longer than 7.5 min were found: the report is split into equal intervals."),
        ("en", {**DETECTION, "sample_sec": 15.0, "reason": "no_stable_step"},
         "No step held stable for 5 min: the report is split into equal intervals."),
    ],
)
def test_the_report_states_how_the_steps_were_found(locale, detection, expected):
    from tests.js_harness import run_js

    html = run_js(REPORT_JS, f"LL.report._internals.renderLoadStepTableHtml({json.dumps(_step_table(detection))})", locale=locale)
    assert expected in html


def test_reports_saved_before_have_no_detector_sentence():
    from tests.js_harness import run_js

    html = run_js(REPORT_JS, f"LL.report._internals.renderLoadStepTableHtml({json.dumps(_step_table(None))})", locale="en")
    assert "Steps found" not in html and "equal intervals" not in html


@pytest.mark.parametrize(
    "lang, expected",
    [
        ("ru", "Ступени найдены по ряду RPS: шаг данных 3 мин, шум плато 0,01 %, минимальный скачок ступени 2,5 %."),
        ("en", "Steps found from the RPS series: data step 3 min, plateau noise 0.01%, smallest step jump 2.5%."),
    ],
)
def test_confluence_states_how_the_steps_were_found(lang, expected):
    from AI.i18n import with_locale
    from loadlens_app.confluence_export import load_step_expand, step_detection_text

    with with_locale(lang):
        assert step_detection_text(_step_table(DETECTION)) == expected
        assert expected in load_step_expand(_step_table(DETECTION))
        assert step_detection_text(_step_table(None)) == ""
