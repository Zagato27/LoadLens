# SPDX-License-Identifier: Apache-2.0
"""Verdict of step and stress tests on the target step: CPU/memory on the first step at the target, steps above it
are headroom, an optional margin over the target (AI/sla_evaluator.py)."""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AI import sla_evaluator as se  # noqa: E402
from AI.i18n import with_locale  # noqa: E402
from AI.sla_evaluator import evaluate_sla  # noqa: E402

STEP_MINUTES = 10
SLA = {
    "target_rps": 100,
    "max_p95_ms": 1000,
    "p95_query": "p95",
    "max_cpu_pct": 90,
    "cpu_query": "CPU",
    "max_performance_query": "RPS",
}


def _data(levels, latencies, cpus, flags=None):
    """Domain data of a step test: one report step of STEP_MINUTES per level, flat RPS, p95 and CPU per step."""
    rps, latency, cpu, steps = [], [], [], []
    for level, lat, load in zip(levels, latencies, cpus):
        rps += [float(level)] * STEP_MINUTES
        latency += [float(lat)] * STEP_MINUTES
        cpu += [float(load)] * STEP_MINUTES
    index = pd.date_range("2026-10-01T10:00:00Z", periods=len(rps), freq="1min")
    for position, level in enumerate(levels):
        step = {
            "index": position + 1,
            "start_iso": str(index[position * STEP_MINUTES]),
            "end_iso": str(index[(position + 1) * STEP_MINUTES - 1]),
            "rps_level": float(level),
        }
        step.update((flags or {}).get(position, {}))
        steps.append(step)
    return {
        "lt_framework": {
            "pack": {"load_steps": steps},
            "labeled": [
                {"label": "RPS", "df": pd.DataFrame({"all": rps}, index=index)},
                {"label": "p95", "df": pd.DataFrame({"all": latency}, index=index)},
            ],
        },
        "hard_resources": {"labeled": [{"label": "CPU", "df": pd.DataFrame({"node": cpu}, index=index)}]},
    }


def _run(levels, latencies, cpus, sla=None, mode="capacity", flags=None):
    return evaluate_sla(_data(levels, latencies, cpus, flags), sla or SLA, test_profile={"mode": mode})


def _check(result, name):
    return next(check for check in result["checks"] if check["name"] == name)


# ---- the case that started it: more steps never make the verdict worse ---------------------------------------

def test_run_that_goes_further_keeps_its_verdict():
    short = _run([100, 200], [300, 400], [40, 60])
    longer = _run([100, 200, 300], [300, 400, 500], [40, 60, 95])
    broken = _run([100, 200, 300], [300, 400, 5000], [40, 60, 95])
    assert [short["verdict"], longer["verdict"], broken["verdict"]] == ["pass", "pass", "pass"]
    for result in (short, longer, broken):
        assert result["target_window"]["level"] == pytest.approx(100.0)
        assert result["target_window"]["role"] == "target"
        assert _check(result, "cpu_usage")["actual"] == pytest.approx(40.0)
        assert "ступени цели ≈100 RPS" in _check(result, "cpu_usage")["message"]
    assert [r["stable_window"]["level"] for r in (short, longer, broken)] == [200.0, 300.0, 200.0]
    assert longer["headroom"]["above_target_failed"] == ["cpu_usage"]
    assert longer["headroom"]["ceiling_level"] == pytest.approx(300.0) and longer["headroom"]["ratio"] == pytest.approx(3.0)
    assert [c["passed"] for c in longer["headroom"]["above_target_checks"]] == [False]
    assert "выше ступени цели нарушено: cpu_usage" in longer["summary"]
    assert "×3 от цели" in longer["summary"]
    assert short["headroom"]["above_target_failed"] == []
    # The peak, the saturation point and the SLA charts keep reading the ceiling.
    assert broken["stable_window"]["degraded_level"] == pytest.approx(300.0)


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_a_prefix_of_a_run_never_has_a_better_verdict(cut):
    levels, latencies, cpus = [100, 200, 300, 400], [300, 400, 500, 4000], [50, 70, 95, 99]
    rank = {"pass": 0, "risk": 1, "fail": 2, "no_data": 3}
    whole = _run(levels, latencies, cpus)
    prefix = _run(levels[:cut], latencies[:cut], cpus[:cut])
    assert rank[whole["verdict"]] <= rank[prefix["verdict"]]


def test_cpu_over_the_threshold_on_the_target_step_is_a_risk():
    result = _run([100, 200], [300, 400], [95, 97])
    assert result["verdict"] == "risk"
    assert _check(result, "cpu_usage")["passed"] is False and _check(result, "cpu_usage")["actual"] == pytest.approx(95.0)
    # Already broken on the target step: not repeated as a breach above it.
    assert result["headroom"]["above_target_failed"] == []
    assert "Нарушено: cpu_usage" in result["summary"] and "выше ступени цели" not in result["summary"]


# ---- choosing the target step --------------------------------------------------------------------------------

def test_target_between_steps_takes_the_first_step_above_it():
    result = _run([50, 150, 250], [300, 400, 500], [30, 95, 99])
    assert result["target_window"]["level"] == pytest.approx(150.0)
    assert result["verdict"] == "risk"


def test_step_measured_just_under_the_target_is_the_target_step():
    result = _run([99.6, 200], [300, 400], [40, 95])
    assert result["target_window"]["level"] == pytest.approx(99.6)
    assert result["verdict"] == "pass"
    target = _check(result, "target_rps")
    assert target["passed"] is True and target["actual"] == pytest.approx(200.0)
    # Only the target step: the tolerance shows in the sign of the message and of the rationale.
    alone = _run([99.6], [300], [40])
    assert _check(alone, "target_rps")["passed"] is True
    assert "99.6 (stable_max (query: RPS)) ≈ целевой 100" in _check(alone, "target_rps")["message"]
    assert "(99.6 ≈ 100.0)" in se.format_sla_rationale(alone["verdict"], alone["checks"])
    assert _run([97.0], [300], [40])["verdict"] == "fail"


def test_target_step_that_breaks_latency_moves_to_the_next_step_that_holds():
    # Warm-up: p95 is over the threshold on the first step at the target, the next one holds.
    result = _run([100, 200, 300], [1500, 400, 500], [40, 60, 95])
    assert result["verdict"] == "pass"
    assert result["target_window"]["level"] == pytest.approx(200.0)
    assert _check(result, "cpu_usage")["actual"] == pytest.approx(60.0)


def test_no_step_at_the_target_holds_fails():
    result = _run([100, 200], [3000, 4000], [40, 60])
    assert result["verdict"] == "fail"
    assert "target_window" not in result and "headroom" not in result
    target = _check(result, "target_rps")
    # No RPS was reached within the SLA: no actual value, so no reader shows 200 ≥ 100 next to a failure.
    assert target["passed"] is False and target["actual"] is None
    assert "не подтверждён" in target["message"] and "label='RPS'; series='all'" in target["message"]
    assert se.format_sla_rationale(result["verdict"], result["checks"]).startswith("Статус «Провал»: нарушены обязательные SLA-критерии")
    # CPU is checked on the ceiling as before (here the latest step, every step breaks the SLA).
    assert "стабильной ступени ≈200 RPS" in _check(result, "cpu_usage")["message"]


def test_target_not_reached_fails_with_the_ceiling():
    result = _run([50, 80, 120], [300, 400, 5000], [40, 60, 95])
    assert result["verdict"] == "fail"
    target = _check(result, "target_rps")
    assert target["passed"] is False and target["actual"] == pytest.approx(80.0)
    assert "80.0 (stable_max (query: RPS)) < целевой 100" in target["message"]


def test_steps_after_an_rps_drop_are_not_load_levels():
    flags = {2: {"after_drop": True}}
    result = _run([100, 200, 150], [300, 400, 300], [40, 60, 50], flags=flags)
    assert result["stable_window"]["level"] == pytest.approx(200.0)
    assert result["target_window"]["level"] == pytest.approx(100.0)
    # A target only an after-drop step would reach is not reached.
    late = _run([100, 120, 300], [300, 400, 300], [40, 60, 50], sla={**SLA, "target_rps": 250}, flags=flags)
    assert late["verdict"] == "fail" and "target_window" not in late


def test_a_collapse_after_an_rps_drop_is_the_degradation_of_the_ceiling():
    # 100 / 200 / 300 hold, then RPS falls to 150 with p95 far over the threshold (stress collapse).
    result = _run([100, 200, 300, 150], [300, 400, 500, 5000], [40, 60, 70, 99], mode="stress", flags={3: {"after_drop": True}})
    window = result["stable_window"]
    assert window["level"] == pytest.approx(300.0)
    assert window["degraded_level"] == pytest.approx(150.0) and window["degraded_checks"] == ["p95_latency"]
    assert "на ступени ≈150 RPS нарушены: p95_latency" in _check(result, "target_rps")["message"]
    # Every step before the drop breaks the thresholds: the latest of them fails, not the step after the drop.
    broken = _run([100, 200, 150], [3000, 4000, 300], [40, 60, 50], flags={2: {"after_drop": True}})
    assert broken["stable_window"]["level"] == pytest.approx(200.0) and broken["verdict"] == "fail"


# ---- without a target ----------------------------------------------------------------------------------------

def test_without_target_cpu_on_the_ceiling_is_a_risk_and_latency_a_failure():
    sla = {key: value for key, value in SLA.items() if key != "target_rps"}
    assert _run([100, 200], [300, 400], [40, 95], sla=sla)["verdict"] == "risk"
    assert _run([100, 200], [3000, 4000], [40, 60], sla=sla)["verdict"] == "fail"
    passed = _run([100, 200], [300, 400], [40, 60], sla=sla)
    assert passed["verdict"] == "pass" and "target_window" not in passed and "headroom" not in passed


# ---- margin over the target ----------------------------------------------------------------------------------

def test_margin_step_checks_cpu_above_the_target():
    result = _run([100, 160, 300], [300, 400, 500], [40, 70, 99], sla={**SLA, "target_margin_pct": 50})
    assert result["verdict"] == "pass"
    assert [check["name"] for check in result["checks"]][:2] == ["target_rps", "target_margin"]
    margin = _check(result, "target_margin")
    assert margin["passed"] is True and margin["threshold"] == pytest.approx(150.0) and margin["category"] == "secondary"
    assert result["target_window"]["role"] == "margin" and result["target_window"]["margin_pct"] == 50
    assert result["target_window"]["level"] == pytest.approx(160.0)
    assert _check(result, "cpu_usage")["actual"] == pytest.approx(70.0)
    assert "ступени запаса ≈160 RPS (цель +50 %)" in _check(result, "cpu_usage")["message"]
    assert "запас 50 % над целью подтверждён на ступени ≈160 RPS" in result["summary"]


def test_missing_margin_is_a_risk_and_cpu_stays_on_the_target_step():
    result = _run([100, 120], [300, 400], [40, 60], sla={**SLA, "target_margin_pct": 50})
    assert result["verdict"] == "risk"
    margin = _check(result, "target_margin")
    assert margin["passed"] is False and margin["actual"] == pytest.approx(120.0)
    assert "не подтверждён: пороги задержек и ошибок выполняются до ≈120 RPS" in margin["message"]
    assert result["target_window"]["role"] == "target" and result["target_window"]["level"] == pytest.approx(100.0)
    assert se.format_sla_rationale(result["verdict"], result["checks"]).startswith("Статус «Есть риски»: целевой RPS достигнут")


def test_a_small_margin_is_never_confirmed_below_the_target():
    result = _run([99, 300], [300, 400], [40, 60], sla={**SLA, "target_margin_pct": 1})
    assert result["target_window"]["role"] == "margin" and result["target_window"]["level"] == pytest.approx(300.0)
    assert _check(result, "cpu_usage")["actual"] == pytest.approx(60.0)
    alone = _run([99], [300], [40], sla={**SLA, "target_margin_pct": 1})
    assert alone["verdict"] == "risk" and _check(alone, "target_margin")["passed"] is False


def test_a_later_step_at_a_lower_rps_is_no_headroom_above_the_target():
    # A sag too small to count as a drop: the last step that holds is below the target step.
    result = _run([100, 200, 180], [300, 400, 400], [40, 60, 97], sla={**SLA, "target_rps": 200})
    assert result["target_window"]["level"] == pytest.approx(200.0) and result["stable_window"]["level"] == pytest.approx(180.0)
    assert result["verdict"] == "pass"
    assert result["headroom"]["above_target_failed"] == [] and result["headroom"]["above_target_checks"] == []
    assert "выполняются до" not in result["summary"] and "выше ступени цели" not in result["summary"]


def test_margin_off_and_target_missed():
    for value in (None, 0, -5, "", "abc"):
        result = _run([100, 120], [300, 400], [40, 60], sla={**SLA, "target_margin_pct": value})
        assert "target_margin" not in [check["name"] for check in result["checks"]]
    missed = _run([50, 80], [300, 400], [40, 60], sla={**SLA, "target_margin_pct": 20})
    assert missed["verdict"] == "fail" and _check(missed, "target_margin")["passed"] is False


def test_margin_without_steps_compares_the_measured_rps():
    pack = {"sections": [{"label": "RPS", "top_series": [{"series": "all", "max": 130.0}]}]}
    data = {"lt_framework": {"pack": pack, "labeled": []}}
    sla = {"target_rps": 100, "target_margin_pct": 20, "max_performance_query": "RPS"}
    result = evaluate_sla(data, sla, test_profile={"mode": "capacity"})
    margin = _check(result, "target_margin")
    assert margin["passed"] is True and margin["actual"] == pytest.approx(130.0)
    assert "RPS 130.0 ≥ 120 (цель +20 %)" in margin["message"]
    no_rps = evaluate_sla({"lt_framework": {"pack": {"sections": []}, "labeled": []}}, sla, test_profile={"mode": "capacity"})
    assert _check(no_rps, "target_margin")["passed"] is None


# ---- stress, summary parts, languages ------------------------------------------------------------------------

def test_stress_uses_the_same_verdict_and_keeps_the_ceiling():
    result = _run([100, 200, 300], [300, 400, 5000], [40, 95, 99], mode="stress")
    assert result["verdict"] == "pass"
    assert result["stable_window"]["level"] == pytest.approx(200.0)
    assert result["target_window"]["level"] == pytest.approx(100.0)


def test_summary_is_rebuilt_from_its_parts():
    result = _run([100, 200, 300], [300, 400, 500], [40, 60, 95])
    parts = se.summary_parts_from_checks(result)
    assert parts["headroom"]["above_target_failed"] == ["cpu_usage"]
    assert se.format_sla_summary(parts, "ru") == result["summary"]
    english = se.format_sla_summary(parts, "en")
    assert "the target is confirmed on the ≈100 RPS step" in english
    assert "breached above the target step: cpu_usage" in english
    assert not re.search("[А-Яа-яЁё]", english)


def test_english_messages_have_no_cyrillic():
    with with_locale("en"):
        results = [
            _run([100, 200, 300], [300, 400, 500], [40, 60, 95]),
            _run([100, 200], [3000, 4000], [40, 60]),
            _run([100, 160], [300, 400], [40, 70], sla={**SLA, "target_margin_pct": 50}),
            _run([100, 120], [300, 400], [40, 60], sla={**SLA, "target_margin_pct": 50}),
        ]
    texts = [result["summary"] for result in results] + [check["message"] for result in results for check in result["checks"]]
    assert not [text for text in texts if re.search("[А-Яа-яЁё]", text)]
    assert "(target step ≈100 RPS" in _check(results[0], "cpu_usage")["message"]
