# SPDX-License-Identifier: Apache-2.0
"""Deterministic report texts in the run language (plans/i18n-plan.md, 4.6 task 8).

SLA checks and summaries (AI/sla_evaluator.py, ``sla.*``), hold / spike / saturation notes and step labels
(AI/stability.py, AI/spike.py, AI/saturation.py, AI/context_pack.py, ``report.*``), test-type texts
(AI/test_profile.py), domain titles (AI/pipeline.py, ``common.domain.*``) and the demo run (loadlens_app/demo_data.py).

ru stays byte-identical to the texts written before the dictionaries; en has no Cyrillic outside user data.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_smoke_api import RecordingConnection  # noqa: E402,F401  (initialises the app before update_page)
from AI import context_pack as cp  # noqa: E402
from AI import pipeline as pipeline_module  # noqa: E402
from AI import report_markers  # noqa: E402
from AI import saturation as sat  # noqa: E402
from AI import scoring as scoring_module  # noqa: E402
from AI import sla_evaluator as se  # noqa: E402
from AI import spike as sp  # noqa: E402
from AI import stability as st  # noqa: E402
from AI import test_profile as tp  # noqa: E402
from AI.i18n import get_locale, with_locale  # noqa: E402
from AI.verdicts import CODES, LABELS, LEGACY  # noqa: E402
from AI.verification import _STEP_REFERENCE_RE  # noqa: E402
from loadlens_app import core  # noqa: E402
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic  # noqa: E402

START_TS = 1704099600.0  # 2024-01-01T09:00:00Z
END_TS = START_TS + 3600
INDEX = pd.date_range("2024-01-01T09:00:00Z", periods=60, freq="1min")
RPS_LABEL = "LT RPS"
P95_LABEL = "LT p95"
ERR_LABEL = "LT errors"
NOISE = np.random.default_rng(7).normal(0, 1.0, 60)
LANGS = ("ru", "en")


def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in strings(item)]
    return []


def cyrillic(value) -> list[str]:
    return [text for text in strings(value) if CYRILLIC.search(text)]


# ---- fixtures (the same data as tests/test_test_types.py) ------------------------------------------------------

def _labeled(rps, p95, err=None):
    out = [{"label": RPS_LABEL, "df": rps.to_frame("sum_all")}, {"label": P95_LABEL, "df": p95.to_frame("p95_all")}]
    if err is not None:
        out.append({"label": ERR_LABEL, "df": err.to_frame("err")})
    return out


def _segments():
    return [
        {"start": str(INDEX[i * 15]), "end": str(INDEX[i * 15 + 14]), "level": level, "stable": True, "duration_min": 15.0}
        for i, level in enumerate([100.0, 200.0, 300.0, 400.0])
    ]


def _cpu():
    frame = pd.DataFrame({"node=a": np.repeat([30.0, 45.0, 60.0, 92.0], 15), "node=b": np.repeat([20.0, 25.0, 30.0, 35.0], 15)}, index=INDEX)
    return [{"label": "CPU", "df": frame}]


STEP_SLA = {"target_rps": 250, "max_p95_ms": 250, "max_error_rate_pct": 1.0, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL,
            "error_rate_query": ERR_LABEL, "max_cpu_pct": 80, "cpu_query": "CPU", "max_memory_pct": 90, "memory_query": "MEM",
            "max_p99_ms": 900, "p99_query": P95_LABEL}


def step_case():
    rps = pd.Series(np.repeat([100.0, 200.0, 300.0, 400.0], 15) + NOISE, index=INDEX)
    p95 = pd.Series(np.repeat([100.0, 150.0, 210.0, 800.0], 15) + NOISE, index=INDEX)
    err = pd.Series(np.repeat([0.1, 0.2, 0.4, 3.0], 15), index=INDEX)
    steps = cp.derive_load_steps(_segments(), START_TS, END_TS, 3, min_step_minutes=5.0)
    data = {"lt_framework": {"pack": {"load_steps": [s.to_dict() for s in steps]}, "labeled": _labeled(rps, p95, err)},
            "hard_resources": {"labeled": _cpu()}}
    return steps, data


def soak_case():
    rps = pd.Series(np.concatenate([np.linspace(0, 300, 5), np.full(55, 300.0)]) + NOISE, index=INDEX)
    p95 = pd.Series(np.linspace(100, 170, 60) + NOISE, index=INDEX)
    hold = st.hold_window_from_series(rps, cp.equal_time_buckets(START_TS, END_TS, 3), series_name="sum_all", label=RPS_LABEL)
    data = {"lt_framework": {"pack": {"hold_window": hold.to_dict(3)}, "labeled": _labeled(rps, p95)}, "hard_resources": {"labeled": _cpu()}}
    sla = {"target_rps": 290, "max_p95_ms": 160, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL, "max_cpu_pct": 50, "cpu_query": "CPU"}
    return hold, data, sla


def spike_frames(recover=True):
    rps = np.full(60, 100.0)
    rps[20:25] = 400.0
    p95 = np.full(60, 100.0)
    p95[20:25] = 400.0
    p95[25:27] = 300.0
    if not recover:
        p95[25:] = 300.0
    return pd.Series(rps + NOISE, index=INDEX), pd.Series(p95 + NOISE, index=INDEX)


def spike_case(recover=True):
    rps, p95 = spike_frames(recover)
    labeled = _labeled(rps, p95, pd.Series(np.full(60, 0.2), index=INDEX))
    profile = sp.detect_spikes(rps, series_name="sum_all", label=RPS_LABEL)
    analysis = sp.analyze_spikes(profile, labeled, {"p95_query": P95_LABEL, "error_rate_query": ERR_LABEL}, END_TS, 3)
    data = {"lt_framework": {"pack": {"spike_profile": analysis}, "labeled": labeled}, "hard_resources": {"labeled": _cpu()}}
    sla = {"target_rps": 350, "max_p95_ms": 300, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL, "max_cpu_pct": 10, "cpu_query": "CPU"}
    return profile, analysis, data, sla


def _flat_spike_case():
    flat = pd.Series(np.full(60, 100.0) + NOISE, index=INDEX)
    analysis = sp.analyze_spikes(sp.detect_spikes(flat), _labeled(flat, flat), {"p95_query": P95_LABEL}, END_TS, 3)
    sla = {"target_rps": 350, "max_p95_ms": 500, "max_performance_query": RPS_LABEL, "p95_query": P95_LABEL}
    return analysis, se.evaluate_sla({"lt_framework": {"pack": {"spike_profile": analysis}, "labeled": _labeled(flat, flat)}}, sla, test_profile={"mode": "spike"})


def _stress_steps():
    steps = []
    for i, level in enumerate([100.0, 200.0, 300.0, 400.0]):
        steps.append(cp.LoadStep(index=i + 1, label=f"L{i + 1}", start=INDEX[i * 15], end=INDEX[i * 15 + 14] + pd.Timedelta(minutes=1),
                                 minute_from=i * 15.0, minute_to=i * 15.0 + 15, rps_level=level, stable=True, source="step_detector", shift_hours=3))
    return steps


def _checks(result):
    return [[check["name"], check["message"]] for check in result["checks"]]


def _anomalies(steps):
    buckets = cp.equal_time_buckets(START_TS, END_TS, 3)
    out = cp.detect_step_anomalies(pd.Series(np.repeat([10.0, 40.0, 90.0, 200.0], 15), index=INDEX), steps)
    out += cp.detect_step_anomalies(pd.Series(np.concatenate([np.linspace(10, 30, 15), np.full(45, 10.0)]), index=INDEX), steps)
    out += cp.detect_step_anomalies(pd.Series(np.repeat([10.0, 30.0, 10.0, 25.0, 10.0, 10.0], 10), index=INDEX), buckets)
    return [item["explanation"] for item in out]


def _scenarios():
    """name -> texts the modules write, for the current locale."""
    steps, step_data = step_case()
    hold, soak_data, soak_sla = soak_case()
    profile, spike_analysis, spike_data, spike_sla = spike_case()
    _bad_profile, _bad_analysis, bad_data, bad_sla = spike_case(recover=False)
    flat_analysis, flat_result = _flat_spike_case()
    window = {"start": str(_stress_steps()[3].start), "end": str(_stress_steps()[3].end), "level": 400.0, "degraded_level": None, "degraded_checks": []}
    not_reached = sat.build_saturation(sla_window=window, sla_checks=[{"name": "p95_latency", "passed": True}], load_steps=_stress_steps(),
                                       rps_drop=None, hr_labeled=[], sla_cfg={}, shift_hours=3)
    multi = pd.DataFrame({"latency|application=api|instance=a": np.repeat([1.0, 2.0, 3.0, 4.0], 15),
                          "latency|application=api|instance=b": np.repeat([1.5, 2.5, 3.5, 4.5], 15), "single": np.full(60, 5.0)}, index=INDEX)
    early = pd.Series(np.concatenate([np.linspace(0, 300, 5), np.full(35, 300.0), np.full(20, 120.0)]) + NOISE, index=INDEX)
    bare = sp.SpikeProfile(baseline_rps=100.0, threshold_rps=150.0, series="s", label=RPS_LABEL,
                           spikes=(sp.Spike(index=1, start=INDEX[20], end=INDEX[25], level_rps=400.0, peak_rps=410.0),),
                           values=spike_frames()[0][INDEX[20]:INDEX[24]])
    peak_pack = {"sections": [{"label": RPS_LABEL, "top_series": [{"series": "sum_all", "max": 410.0}]}]}
    late = [dict(s, start_iso="2030-01-01T00:00:00+00:00", end_iso="2030-01-01T00:10:00+00:00", rps_level=100.0)
            for s in step_data["lt_framework"]["pack"]["load_steps"]]
    return {
        "step": _checks(se.evaluate_sla(step_data, STEP_SLA, test_profile={"mode": "capacity"})) + [se.evaluate_sla(step_data, STEP_SLA)["summary"]],
        "soak": _checks(se.evaluate_sla(soak_data, soak_sla, test_profile={"mode": "stability"}))
        + [se.evaluate_sla(soak_data, soak_sla, test_profile={"mode": "stability"})["summary"]],
        "spike": _checks(se.evaluate_sla(spike_data, spike_sla, test_profile={"mode": "spike"}))
        + [se.evaluate_sla(spike_data, spike_sla, test_profile={"mode": "spike"})["summary"]],
        "spike_not_recovered": [c for c in _checks(se.evaluate_sla(bad_data, bad_sla, test_profile={"mode": "spike"})) if c[0] == "recovery"],
        "spike_not_detected": [flat_analysis["reason"], flat_result["summary"], [c["message"] for c in flat_result["checks"]]],
        "no_source": [c["message"] for c in se.evaluate_sla({}, {"max_p95_ms": 100, "max_error_rate_pct": 1, "target_rps": 200})["checks"]],
        "no_data": [c["message"] for c in se.evaluate_sla({"lt_framework": {"labeled": []}}, {"max_p95_ms": 100, "p95_query": "missing", "target_rps": 200,
                                                                                               "max_performance_query": RPS_LABEL})["checks"]],
        "empty_window": [c["message"] for c in se.evaluate_sla({"lt_framework": {"pack": {"load_steps": late}, "labeled": step_data["lt_framework"]["labeled"]}},
                                                               STEP_SLA)["checks"]],
        "no_hold": [c["message"] for c in se.evaluate_sla({"lt_framework": {"pack": {}, "labeled": []}}, {**soak_sla, "max_performance_query": ""},
                                                          test_profile={"mode": "stability"})["checks"]],
        "peak": [c["message"] for c in se.evaluate_sla({"lt_framework": {"pack": peak_pack, "labeled": step_data["lt_framework"]["labeled"]}}, STEP_SLA)["checks"]][:1],
        "no_criteria": se.evaluate_sla({}, {})["summary"],
        "no_checks": se.evaluate_sla({}, {"max_cpu_pct": 80})["summary"],
        "hold_check": st.hold_check(hold.to_dict(3), 310.0)["message"],
        "hold_note": st.hold_window_from_series(early, cp.equal_time_buckets(START_TS, END_TS, 3)).to_dict(3)["note"],
        "saturation": [not_reached["note"], not_reached["failure_mode_title"]],
        "step_labels": [s.label for s in steps],
        "bucket_labels": [s.label for s in cp.equal_time_buckets(START_TS, END_TS, 3)],
        "spike_labels": [s.label for s in sp.spike_load_steps(profile, START_TS, END_TS, 3)],
        "spike_labels_no_level": [s.label for s in sp.spike_load_steps(bare, START_TS, END_TS, 3)],
        "spike_note": sp.analyze_spikes(profile, spike_data["lt_framework"]["labeled"], {}, END_TS, 3)["spikes"][0]["note"],
        "anomalies": _anomalies(steps),
        "timeline": cp.build_timeline({}, steps)["note"],
        "aggregated_series": [row["series"] for row in cp.step_table([{"label": "Lat", "df": multi}], steps)[0]["rows"]],
        "step_table_note": cp.load_step_report_from_frames(steps, step_data["lt_framework"]["labeled"], rps_query=RPS_LABEL)["note"],
        "context_label": [row["label"] for row in cp.load_step_report_from_context({"load_steps": [{"index": 2}]})["steps"]],
        "trend_titles": [item["title"] for item in st.compute_trends(soak_data, {**soak_sla, "memory_query": "CPU"}, hold.to_dict(3))],
        "profile": [tp.profile_for("soak").to_dict(get_locale())["label"], tp.profile_for("spike").peak_label(get_locale())],
    }


GOLDEN_RU = {
    "step": [
        ["target_rps", "RPS 300.0 (stable_max (query: LT RPS)) ≥ целевой 250; label='LT RPS'; series='sum_all'; на ступени ≈400 RPS нарушены: error_rate, p95_latency"],
        ["error_rate", "p95 0.40 % ≤ порог 1 % (стабильной ступени ≈300 RPS, запрос «LT errors», серия «err»)"],
        ["p95_latency", "p95 211.15 мс ≤ порог 250 мс (стабильной ступени ≈300 RPS, запрос «LT p95», серия «p95_all»)"],
        ["p99_latency", "p95 211.15 мс ≤ порог 900 мс (стабильной ступени ≈300 RPS, запрос «LT p95», серия «p95_all»)"],
        ["cpu_usage", "p95 60.00 % ≤ порог 80 % (ступени цели ≈300 RPS, запрос «CPU», серия «node=a»)"],
        ["memory_usage", "Нет данных по запросу «MEM»"],
        "SLA verdict: Успешно. Пройдено: target_rps, error_rate, p95_latency, p99_latency, cpu_usage; Нет данных: memory_usage; "
        "цель подтверждена на ступени ≈300 RPS",
    ],
    "soak": [
        ["target_rps", "Удержание: средний RPS 299.8 за окно 12:04–13:00 (56.0 мин) ≥ целевой 290 (допуск 5 %); "
                       "минимальный интервал «интервал 3 · 12:20–12:30» 299.4 RPS (+3 % от целевого, допуск −15 %)"],
        ["p95_latency", "p95 166.85 мс > порог 160 мс (окна удержания ≈300 RPS, запрос «LT p95», серия «p95_all»)"],
        ["cpu_usage", "p95 92.00 % > порог 50 % (окна удержания ≈300 RPS, запрос «CPU», серия «node=a»)"],
        "SLA verdict: Провал. Пройдено: target_rps; Нарушено: p95_latency, cpu_usage; Режим stability: пороги проверены на окне удержания, "
        "target_rps — удержание среднего уровня; CPU/memory трактуются как вторичные риски и сами по себе не переводят тест в «Провал»",
    ],
    "spike": [
        ["target_rps", "Уровень всплеска «всплеск 1 · ≈400 RPS» 399.8 RPS ≥ целевой 350; label='LT RPS'"],
        ["p95_latency", "p95 400.25 мс > порог 300 мс (окна всплеска «всплеск 1 · ≈400 RPS» ≈400 RPS, запрос «LT p95», серия «p95_all»)"],
        ["cpu_usage", "p95 45.00 % > порог 10 % (окна всплеска «всплеск 1 · ≈400 RPS» ≈400 RPS, запрос «CPU», серия «node=a»)"],
        ["recovery", "Задержки вернулись к фону за 2 мин (худший всплеск), предел 15 мин"],
        "SLA verdict: Есть риски. Пройдено: target_rps, recovery; Нарушено: p95_latency, cpu_usage; Режим spike: пороги проверены в окнах всплесков "
        "(худший всплеск), target_rps — уровень всплеска; нарушение target_rps или recovery — «Провал», задержки и ошибки в окне всплеска — риски",
    ],
    "spike_not_recovered": [["recovery", "Задержки не вернулись к фону (±10 %) за 15 мин после: всплеск 1 · ≈400 RPS"]],
    "spike_not_detected": [
        "RPS не поднимался до порога всплеска 150 (фон 100 × 1.5) дольше 60 с",
        "SLA verdict: Недостаточно данных. Пройдено: p95_latency; Нет данных: target_rps, recovery; Режим spike: пороги проверены в окнах всплесков "
        "(худший всплеск), target_rps — уровень всплеска; нарушение target_rps или recovery — «Провал», задержки и ошибки в окне всплеска — риски; "
        "всплеск нагрузки не обнаружен",
        ["RPS не поднимался до порога всплеска 150 (фон 100 × 1.5) дольше 60 с",
         "p95 101.34 мс ≤ порог 500 мс (всего окна, запрос «LT p95», серия «p95_all»)",
         "Всплеск нагрузки не обнаружен: восстановление не оценивалось"],
    ],
    "no_source": ["Нет данных RPS из lt_framework (reason=no_query_configured)", "Источник не задан: выберите запрос в настройках SLA",
                  "Источник не задан: выберите запрос в настройках SLA"],
    "no_data": ["Нет данных RPS из lt_framework (reason=no_sections_in_pack)", "Нет данных по запросу «missing»"],
    "empty_window": ["RPS 100.0 (stable_max (query: LT RPS)) < целевой 250; label='LT RPS'; series='sum_all'",
                     "На стабильной ступени RPS нет точек запроса «LT errors»", "На стабильной ступени RPS нет точек запроса «LT p95»",
                     "На стабильной ступени RPS нет точек запроса «LT p95»", "Нет данных по запросу «CPU»", "Нет данных по запросу «MEM»"],
    "no_hold": ["Нет данных RPS для окна удержания (запрос «не выбран»)", "Нет данных по запросу «LT p95»", "Нет данных по запросу «CPU»"],
    "peak": ["RPS 410.0 (peak_max (query: LT RPS, no stable segments found)) ≥ целевой 250; label='LT RPS'; series='sum_all'"],
    "no_criteria": "SLA-критерии не заданы",
    "no_checks": "SLA verdict: Недостаточно данных. Нет данных: cpu_usage",
    "hold_check": "Удержание: средний RPS 299.8 за окно 12:04–13:00 (56.0 мин) ≥ целевой 310 (допуск 5 %); "
                  "минимальный интервал «интервал 3 · 12:20–12:30» 299.4 RPS (-3 % от целевого, допуск −15 %)",
    "hold_note": "RPS упал задолго до конца теста: падение включено в окно удержания",
    "saturation": ["Предел не достигнут: ни одна ступень не нарушила SLA по задержкам и ошибкам, RPS не падал.", "деградация не зафиксирована"],
    "step_labels": ["ступень 1 · ≈100 RPS", "ступень 2 · ≈200 RPS", "ступень 3 · ≈300 RPS", "ступень 4 · ≈400 RPS"],
    "bucket_labels": ["интервал 1 · 12:00–12:10", "интервал 2 · 12:10–12:20", "интервал 3 · 12:20–12:30",
                      "интервал 4 · 12:30–12:40", "интервал 5 · 12:40–12:50", "интервал 6 · 12:50–13:00"],
    "spike_labels": ["фон 1 · ≈100 RPS", "всплеск 1 · ≈400 RPS", "фон 2 · ≈100 RPS"],
    "spike_labels_no_level": ["фон 1", "всплеск 1 · ≈400 RPS", "фон 2"],
    "spike_note": "запрос p95/p99 не выбран в SLA: рост задержек и восстановление не оцениваются",
    "anomalies": [
        "На ступень 4 · ≈400 RPS: метрика выросла на 1900 % при росте нагрузки на 300 % (в 6.3× быстрее нагрузки) относительно ступень 1 · ≈100 RPS.",
        "На ступень 3 · ≈300 RPS: метрика выросла на 800 % при росте нагрузки на 200 % (в 4.0× быстрее нагрузки) относительно ступень 1 · ≈100 RPS.",
        "На ступень 2 · ≈200 RPS: метрика выросла на 300 % при росте нагрузки на 100 % (в 3.0× быстрее нагрузки) относительно ступень 1 · ≈100 RPS.",
        "Внутри ступень 1 · ≈100 RPS: значение изменилось на +150 % от начала к концу ступени при неизменной нагрузке.",
        "На интервал 2 · 12:10–12:20: среднее значение отличается от первого интервала на +200 %.",
        "На интервал 4 · 12:30–12:40: среднее значение отличается от первого интервала на +150 %.",
    ],
    "timeline": "Значения — средние по плато ступени, без просадки и спада нагрузки после него (они в step_table.per_step.after_plateau). "
                "Уровень нагрузки ступени — steps[].rps_level. Порядок строк: метрики нагрузочного инструмента, затем метрики с наибольшим "
                "относительным изменением.",
    "aggregated_series": ["latency|application=api|instances=2 (среднее)", "single"],
    "step_table_note": "RPS — уровень ступени. Время отклика — p95 на плато, без хвоста после просадки.",
    "context_label": ["ступень 2"],
    "trend_titles": ["RPS", "p95 latency", "CPU", "память"],
    "profile": ["Долговременный (soak)", "Уровень всплеска"],
}


# ---- ru: byte-identical ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scenarios_ru():
    return _scenarios()


@pytest.mark.parametrize("name", sorted(GOLDEN_RU))
def test_ru_texts_are_the_former_texts(scenarios_ru, name):
    assert scenarios_ru[name] == GOLDEN_RU[name]


# The ru templates the SLA rationale and summary were built from before the dictionaries (AI/sla_evaluator.py).
FORMER_HEADS = {
    "pass_target": "Статус «{label}»: целевой RPS достигнут ({actual} ≥ {threshold}), вторичные пороги SLA не нарушены.",
    "pass": "Статус «{label}»: проверяемые SLA-критерии соблюдены.",
    "risk_target": "Статус «{label}»: целевой RPS достигнут, но нарушены вторичные пороги ({failed}).",
    "risk_failed": "Статус «{label}»: нарушены критерии {failed}, без однозначного провала по целевому RPS.",
    "risk": "Статус «{label}»: не все SLA-критерии удалось подтвердить числами.",
    "fail_target": "Статус «{label}»: целевой RPS не достигнут ({actual} < {threshold}).",
    "fail_failed": "Статус «{label}»: нарушены обязательные SLA-критерии ({failed}).",
    "fail": "Статус «{label}»: по SLA целевой результат теста не подтверждён.",
    "no_data": "Статус «{label}»: не хватает метрик, чтобы проверить SLA.",
}
FORMER_TITLES = {"target_rps": "целевой RPS", "error_rate": "доля ошибок", "p95_latency": "p95 latency", "p99_latency": "p99 latency",
                 "cpu_usage": "CPU", "memory_usage": "память", "recovery": "восстановление после всплеска"}
FORMER_MODES = {
    "stability": "Режим stability: пороги проверены на окне удержания, target_rps — удержание среднего уровня; "
                 "CPU/memory трактуются как вторичные риски и сами по себе не переводят тест в «Провал»",
    "spike": "Режим spike: пороги проверены в окнах всплесков (худший всплеск), target_rps — уровень всплеска; "
             "нарушение target_rps или recovery — «Провал», задержки и ошибки в окне всплеска — риски",
}
CHECK_SETS = [
    [],
    [{"name": "target_rps", "passed": True, "actual": 226.5, "threshold": 200, "message": "ok"}],
    [{"name": "target_rps", "passed": True, "actual": 226.5, "threshold": 200}, {"name": "p95_latency", "passed": False, "actual": 340, "threshold": 200}],
    [{"name": "target_rps", "passed": False, "actual": 150, "threshold": 200}, {"name": "cpu_usage", "passed": None}],
    [{"name": "p95_latency", "passed": False}, {"name": "memory_usage", "passed": False}],
    [{"name": "recovery", "passed": None}],
    [{"name": "target_rps", "passed": None, "actual": None, "threshold": 200}],
    [{"name": "error_rate", "passed": True, "actual": 0.1, "threshold": 1}, {"name": "zzz", "passed": True}, {"name": "", "passed": False}],
    [{"name": "p99_latency", "passed": False, "actual": 900.5, "threshold": 500.25}],
]


def _former_title(name):
    return FORMER_TITLES.get(name, name or "критерий")


def _former_rationale(code, checks):
    label = LABELS["ru"][code]
    rps = next((c for c in checks if c.get("name") == "target_rps"), None)
    failed = ", ".join(_former_title(str(c.get("name") or "")) for c in checks if c.get("passed") is False)
    if code == "pass":
        key = "pass_target" if rps and rps.get("passed") is True and rps.get("actual") is not None else "pass"
    elif code == "risk":
        key = "risk_target" if rps and rps.get("passed") is True and failed else ("risk_failed" if failed else "risk")
    elif code == "fail":
        key = "fail_target" if rps and rps.get("passed") is False and rps.get("actual") is not None else ("fail_failed" if failed else "fail")
    else:
        key = "no_data"
    head = FORMER_HEADS[key].format(label=label, failed=failed, actual=(rps or {}).get("actual"), threshold=(rps or {}).get("threshold"))
    lines = []
    for check in checks:
        title, message = _former_title(str(check.get("name") or "")), str(check.get("message") or "").strip()
        if message:
            lines.append(f"- {title}: {message}")
        elif check.get("passed") is True:
            lines.append(f"- {title}: порог соблюдён ({check.get('actual')} при лимите {check.get('threshold')})")
        elif check.get("passed") is False:
            lines.append(f"- {title}: порог нарушен ({check.get('actual')} при лимите {check.get('threshold')})")
        else:
            lines.append(f"- {title}: нет данных для проверки")
    return head + ("\n" + "\n".join(lines) if lines else "")


def _former_summary(code, checks, mode, detected, adjusted):
    items = []
    for word, state in (("Пройдено", True), ("Нарушено", False), ("Нет данных", None)):
        names = [str(c.get("name")) for c in checks if c.get("passed") is state]
        if names:
            items.append(f"{word}: {', '.join(names)}")
    if mode in FORMER_MODES:
        items.append(FORMER_MODES[mode])
    if mode == "spike" and not detected:
        items.append("всплеск нагрузки не обнаружен")
    if adjusted:
        items += [f"Режим {mode}: ресурсные превышения CPU/memory трактуются как риски, но без нарушений target_rps/error_rate/p95/p99 "
                  "не переводят тест в «Провал».", "Пороги SLA выполнены, но за окно удержания найден неблагоприятный тренд: p95, RPS"]
    return f"SLA verdict: {LABELS['ru'][code]}. " + "; ".join(items)


def test_ru_rationale_and_summary_combinations_are_the_former_texts():
    count = 0
    for code, checks in itertools.product(CODES, CHECK_SETS):
        expected = _former_rationale(code, checks)
        assert se.format_sla_rationale(code, checks) == expected
        assert se.format_sla_rationale(code, checks, "ru") == expected
        count += 1
    for label, code in LEGACY.items():
        assert se.format_sla_rationale(label, CHECK_SETS[2]) == _former_rationale(code, CHECK_SETS[2])
    for code, checks, mode, detected, adjusted in itertools.product(CODES, CHECK_SETS, ["capacity", "stress", "stability", "spike", ""],
                                                                    [True, False], [False, True]):
        parts = se.sla_summary_parts(code, checks, mode, detected)
        if adjusted:
            parts["adjustments"] = [{"code": se.ADJUST_RESOURCES, "mode": mode}, {"code": se.ADJUST_TREND, "trends": ["p95", "RPS"]}]
        assert se.format_sla_summary(parts) == _former_summary(code, checks, mode, detected, adjusted)
        count += 1
    assert count == 36 + 720


def test_tables_hold_the_dictionary_texts_in_both_languages():
    """The public tables read the dictionaries: TEST_TYPE_TEXTS, HOLD_NOTE_TEXTS & co. index by language."""
    for lang in LANGS:
        types = tp.TEST_TYPE_TEXTS[lang]
        assert set(types) == set(tp.TEST_TYPES) and all(set(texts) == {"label", "peak_label", "focus"} for texts in types.values())
        assert types["step"]["label"] == tp.test_type_label("step", lang) == tp.profile_for("step").label(lang)
        assert st.HOLD_NOTE_TEXTS[lang]["early_drop"] == st.hold_note_text(["early_drop"], lang)
        assert sp.SPIKE_NOTE_TEXTS[lang]["no_latency_query"] == sp.spike_note_text("no_latency_query", lang)
        assert sat.FAILURE_MODE_TEXTS[lang]["latency+errors"] == sat.failure_mode_title("latency+errors", lang)
        assert sat.SATURATION_NOTE_TEXTS[lang]["no_sla_step"] == sat.saturation_note_text("no_sla_step", lang)
    assert set(tp.TEST_TYPE_TEXTS) == set(LANGS) and "de" not in tp.TEST_TYPE_TEXTS
    # Unknown codes and languages behave as before: the code itself / '' and ru.
    assert st.hold_note_text(["zzz", "early_drop"], "de") == "zzz; RPS упал задолго до конца теста: падение включено в окно удержания"
    assert sp.spike_note_text("zzz", "en") == "zzz" and sat.failure_mode_title("zzz", "en") == "" and tp.test_type_text("x", "label", "en") == ""
    with with_locale("en"):
        assert dict(se.SLA_CHECK_TITLES)["memory_usage"] == "memory" and st.TREND_TITLES["error_rate"] == "error rate"
        assert sat.FAILURE_MODE_TITLES["none"] == "no degradation recorded"
    assert dict(se.SLA_CHECK_TITLES)["memory_usage"] == "память"


# ---- en: no Cyrillic -------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scenarios_en():
    with with_locale("en"):
        return _scenarios()


def test_en_texts_have_no_cyrillic(scenarios_en):
    assert {name: cyrillic(value) for name, value in scenarios_en.items() if cyrillic(value)} == {}


def test_en_labels_and_messages_use_the_english_formats(scenarios_en):
    en = scenarios_en
    assert en["step_labels"][2] == "step 3 · ≈300 RPS" and en["context_label"] == ["step 2"]
    assert en["bucket_labels"][1] == "interval 2 · 12:10–12:20"
    assert en["spike_labels"] == ["background 1 · ≈100 RPS", "spike 1 · ≈400 RPS", "background 2 · ≈100 RPS"]
    assert en["spike_labels_no_level"] == ["background 1", "spike 1 · ≈400 RPS", "background 2"]
    assert not any("baseline" in label for label in en["spike_labels"] + en["step_labels"] + en["bucket_labels"])
    assert en["step"][2] == ["p95_latency", "p95 211.15 ms ≤ threshold 250 ms (stable step ≈300 RPS, query «LT p95», series «p95_all»)"]
    assert en["step"][0][1] == ("RPS 300.0 (stable_max (query: LT RPS)) ≥ target 250; label='LT RPS'; series='sum_all'; "
                                "breached at the ≈400 RPS step: error_rate, p95_latency")
    assert en["step"][-1] == ("SLA verdict: Passed. Passed: target_rps, error_rate, p95_latency, p99_latency, cpu_usage; No data: memory_usage; "
                              "the target is confirmed on the ≈300 RPS step")
    assert en["step"][4] == ["cpu_usage", "p95 60.00 % ≤ threshold 80 % (target step ≈300 RPS, query «CPU», series «node=a»)"]
    assert en["soak"][0][1].startswith("Hold: mean RPS 299.8 over the window 12:04–13:00 (56.0 min) ≥ target 290 (tolerance 5 %); "
                                       "lowest interval «interval 3 · 12:20–12:30» 299.4 RPS (+3 % vs the target")
    assert en["spike"][0][1] == "Spike «spike 1 · ≈400 RPS» level 399.8 RPS ≥ target 350; label='LT RPS'"
    assert en["spike"][-1].endswith('a breach of target_rps or recovery is "Failed", latency and errors in the spike window are risks')
    assert en["spike_not_detected"][1].endswith("; no load spike detected")
    assert en["no_hold"][0] == "No RPS data for the hold window (query «not chosen»)"
    assert en["aggregated_series"] == ["latency|application=api|instances=2 (mean)", "single"]
    assert en["anomalies"][0].startswith("At step 4 · ≈400 RPS: the metric grew by 1900 % while the load grew by 300 % (6.3× faster")
    assert en["trend_titles"] == ["RPS", "p95 latency", "CPU", "memory"]
    assert en["profile"] == ["Soak (long-running)", "Spike level"]
    assert en["hold_note"] == "RPS dropped long before the end of the test: the drop is inside the hold window"
    assert en["saturation"] == ["The limit was not reached: no step broke the latency and error SLA, RPS did not drop.", "no degradation recorded"]


def test_evaluate_sla_lang_argument_wins_over_the_locale():
    steps, data = step_case()
    with with_locale("ru"):
        result = se.evaluate_sla(data, STEP_SLA, lang="en")
    assert cyrillic(result) == [] and result["summary"].startswith("SLA verdict: Passed. ")
    with with_locale("en"):
        result = se.evaluate_sla(data, STEP_SLA, lang="ru")
    assert result["summary"].startswith("SLA verdict: Успешно. ") and "запрос «LT p95»" in result["checks"][2]["message"]
    rationale = se.format_sla_rationale("fail", CHECK_SETS[3], "en")
    assert rationale == ('Status "Failed": the target RPS is not reached (150 < 200).\n'
                         "- target RPS: threshold breached (150 with a limit of 200)\n- CPU: no data to check")


@pytest.mark.parametrize("lang", LANGS)
def test_query_and_series_of_check_messages_are_parsed_in_both_languages(lang):
    """Readers of reports saved before ``query_label``/``series`` parse the check text: «запрос «…»» / «query «…»»."""
    steps, data = step_case()
    with with_locale(lang):
        checks = se.evaluate_sla(data, STEP_SLA)["checks"]
    p95 = next(check for check in checks if check["name"] == "p95_latency")
    assert cp._QUERY_RE.search(p95["message"]).group(1) == P95_LABEL
    assert cp._SERIES_RU_RE.search(p95["message"]).group(1) == "p95_all"
    stripped = [{key: value for key, value in check.items() if key not in {"query_label", "series"}} for check in checks]
    assert cp._check_hint(stripped, "p95_latency") == {"label": P95_LABEL, "series": "p95_all", "actual": p95["actual"]}
    assert cp._check_hint(stripped, "target_rps")["label"] == RPS_LABEL
    assert report_markers._message_hint(p95["message"]) == (P95_LABEL, "p95_all")


def test_english_step_labels_are_references_for_verification_and_scoring():
    for label in ("step 3 · ≈150 RPS", "interval 2 · 10:20–10:30", "background 1 · ≈100 RPS", "spike 1 · ≈480 RPS"):
        assert _STEP_REFERENCE_RE.sub(" ", f"p95 at {label} grew").split()[:2] == ["p95", "at"], label
        assert re.search(r"\d", _STEP_REFERENCE_RE.sub(" ", label.split(" · ")[0])) is None
    # The stems tie a quoted number to the load_steps reference as with «ступень 3 · ≈300 RPS».
    for label in ("step 3 · ≈300 RPS", "ступень 3 · ≈300 RPS"):
        refs = scoring_module._extra_numeric_references({"load_steps": [{"label": label, "rps_level": 300.0}]})
        english = scoring_module._matching_numeric_references("Throughput held at step 3 with 300 requests per second", refs)
        russian = scoring_module._matching_numeric_references("На ступени 3 держалось 300 запросов в секунду", refs)
        assert [ref["value"] for ref in english] == [ref["value"] for ref in russian] == [300.0], label
        assert scoring_module._matching_numeric_references("Heap used is flat", refs) == []


# ---- en report from fixtures (the pipeline) ---------------------------------------------------------------------

def _pipeline_frames(kind):
    if kind in ("step", "stress"):
        rps = np.repeat([100.0, 200.0, 300.0, 400.0], 15) + NOISE
        p95 = np.repeat([100.0, 110.0, 130.0, 130.0], 15) + NOISE
        p95[52:] = 420.0
        return [pd.DataFrame({"sum_all": rps}, index=INDEX), pd.DataFrame({"p95_all": p95}, index=INDEX)]
    if kind == "soak":
        rps = np.concatenate([np.linspace(0, 300, 5), np.full(55, 300.0)]) + NOISE
        p95 = np.linspace(100, 170, 60) + NOISE
        return [pd.DataFrame({"sum_all": rps}, index=INDEX), pd.DataFrame({"p95_all": p95}, index=INDEX)]
    rps, p95 = spike_frames()
    return [rps.to_frame("sum_all"), p95.to_frame("p95_all")]


PIPELINE_SLA = {
    "step": {"target_rps": 250, "max_p95_ms": 300},
    "stress": {"target_rps": 250, "max_p95_ms": 300},
    "soak": {"target_rps": 290, "max_p95_ms": 300},
    "spike": {"target_rps": 350, "max_p95_ms": 500},
}
# Deterministic report fields this task writes (the model texts are stubs, the type payloads are per kind).
REPORT_FIELDS = ("deterministic_sla", "test_profile", "load_step_table", "stability_under_load", "spike_analysis", "saturation", "domain_titles")


def _run_pipeline(monkeypatch, kind, report_language):
    cfg = {
        "llm": {"include_markdown_tables_in_context": False, "self_consistency_k": 1, "max_domain_workers": 1},
        "default_params": {"step": "1m", "resample_interval": "1T"},
        "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prometheus:9090"}}},
        "domain_sources": {"default": {"source": "prom"}, "lt_framework": {"source": "prom"}},
        "queries": {"lt_framework": {"promql_queries": ["a", "b"], "label_keys_list": [[], []], "labels": [RPS_LABEL, P95_LABEL]}},
        "sla": {"max_performance_query": RPS_LABEL, "p95_query": P95_LABEL, **PIPELINE_SLA[kind]},
    }
    captured = {}
    monkeypatch.setattr(pipeline_module, "load_prompt", lambda name, lang=None: "prompt")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *a, **k: _pipeline_frames(kind))

    def fake_llm(user_prompt, data_context, k, return_scores, domain_key):
        payload = {"verdict": "pass", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}
        captured[domain_key] = json.loads(data_context)
        return json.dumps(payload), dict(payload), {}

    monkeypatch.setattr(pipeline_module, "llm_two_pass_self_consistency", fake_llm)
    result = pipeline_module.uploadFromLLM(
        start_ts=START_TS, end_ts=END_TS, save_to_db=False, ef_config=cfg, report_language=report_language,
        run_meta={"test_type": kind, "run_name": f"run-{kind}", "service": "demo"}, active_domains=["lt_framework"],
    )
    return result, captured


@pytest.mark.parametrize("kind", list(PIPELINE_SLA))
def test_en_report_from_fixtures_has_no_cyrillic_in_deterministic_texts(monkeypatch, kind):
    result, captured = _run_pipeline(monkeypatch, kind, "en")
    final = result["final_parsed"]
    assert final["report_language"] == result["report_language"] == "en"
    deterministic = {field: final.get(field) for field in REPORT_FIELDS}
    deterministic.update({key: result.get(key) for key in ("sla_summary", "sla_checks", "load_step_table", "sla_window")})
    assert cyrillic(deterministic) == []
    assert result["sla_summary"].startswith("SLA verdict: ")
    assert final["test_profile"]["label"] == tp.test_type_label(kind, "en")
    labels = [row["label"] for row in (result.get("load_step_table") or {}).get("steps") or []]
    labels += [step["label"] for step in captured["final"].get("load_steps") or []]
    assert labels and all(re.match(r"^(step|interval|background|spike) \d+", label) for label in labels), labels
    # The LLM context of the run: every human text is English; query labels and series are user data (Latin here).
    for domain, context in captured.items():
        assert cyrillic(context) == [], domain


def test_ru_report_from_fixtures_keeps_the_russian_texts(monkeypatch):
    result, captured = _run_pipeline(monkeypatch, "spike", None)
    assert result["report_language"] == "ru"
    assert [step["label"].split(" · ")[0] for step in captured["final"]["load_steps"]] == ["фон 1", "всплеск 1", "фон 2"]
    assert result["sla_summary"].startswith("SLA verdict: Успешно. ")
    assert "Режим spike" in result["sla_summary"]


def test_domain_titles_follow_the_locale():
    assert pipeline_module.DOMAIN_TITLES.get("database") == "База данных" and pipeline_module._domain_title("final") == "Итог"
    with with_locale("en"):
        titles = dict(pipeline_module.DOMAIN_TITLES)
        assert titles["database"] == "Database" and titles["final"] == "Summary"
        assert cyrillic(titles) == [] and pipeline_module._domain_title("unknown") == "unknown"
    assert set(pipeline_module.DOMAIN_TITLES) == set(pipeline_module.DOMAIN_KEYS)


# ---- demo -------------------------------------------------------------------------------------------------------

def _seed_demo(monkeypatch, report_language=None):
    from loadlens_app import demo_data

    saved = {}
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: False)
    monkeypatch.setattr(demo_data, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(demo_data, "save_domain_labeled", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo_data, "save_llm_results", lambda results, run_meta, storage: saved.update(results=results, run_meta=run_meta))
    demo_data.seed_demo_run(now=datetime(2024, 11, 1, 11, 0, tzinfo=timezone.utc), report_language=report_language)
    return saved["results"], saved["run_meta"]


def test_demo_run_in_english_follows_the_report_language_setting(monkeypatch):
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    results, run_meta = _seed_demo(monkeypatch)
    assert results["report_language"] == run_meta["report_language"] == results["final_parsed"]["report_language"] == "en"
    assert results["sla_summary"] == "SLA verdict: Risks found. Passed: target_rps, error_rate; Breached: p95_latency"
    texts = {key: value for key, value in results.items() if key not in {"scores"}}
    assert cyrillic(texts) == []
    assert [step["label"] for step in results["contexts"]["final"]["load_steps"]][0] == "step 1 · ≈50 RPS"
    assert results["load_step_table"]["note"].startswith("RPS is the step level.")
    assert results["system_context"]["system"]["name"] == "Checkout Platform (demo)"


def test_demo_run_stays_russian_by_default(monkeypatch):
    results, run_meta = _seed_demo(monkeypatch)
    assert results["report_language"] == run_meta["report_language"] == "ru"
    assert results["sla_summary"] == "SLA verdict: Есть риски. Пройдено: target_rps, error_rate; Нарушено: p95_latency"
    assert results["lt_framework_parsed"]["findings"][0]["summary"] == "P95 времени ответа превысил 400 мс на ступени 250 RPS"
    assert results["contexts"]["final"]["load_steps"][0]["label"] == "ступень 1 · ≈50 RPS"
    # An explicit language wins over the setting.
    results, _ = _seed_demo(monkeypatch, report_language="en")
    assert results["final_parsed"]["findings"][0]["summary"] == "checkout response time degrades on the 250 RPS step"


def test_demo_texts_have_the_same_keys_in_both_languages():
    from loadlens_app import demo_data

    texts = demo_data._all_demo_texts()
    assert set(texts) == set(LANGS) and set(texts["ru"]) == set(texts["en"])
    assert all(value.strip() for lang in LANGS for value in texts[lang].values())
    assert_no_cyrillic("\n".join(texts["en"].values()))
    with with_locale("en"):
        assert str(demo_data.DemoAlreadyExistsError("demo-run", "svc")) == "Demo run «demo-run» already exists"
    assert str(demo_data.DemoAlreadyExistsError("demo-run", "svc")) == "Демо-прогон «demo-run» уже существует"


@pytest.mark.parametrize(
    "stored, en, ru",
    [
        ("ступень 1 · ≈146 RPS", "step 1 · ≈146 RPS", "ступень 1 · ≈146 RPS"),
        ("ступень 3", "step 3", "ступень 3"),
        ("фон 2 · ≈50 RPS", "background 2 · ≈50 RPS", "фон 2 · ≈50 RPS"),
        ("фон 1", "background 1", "фон 1"),
        ("всплеск 1 · ≈900 RPS", "spike 1 · ≈900 RPS", "всплеск 1 · ≈900 RPS"),
        ("интервал 2 · 10:05–10:10", "interval 2 · 10:05–10:10", "интервал 2 · 10:05–10:10"),
        ("step 4 · ≈293 RPS", "step 4 · ≈293 RPS", "ступень 4 · ≈293 RPS"),
        ("Checkout warm-up", "Checkout warm-up", "Checkout warm-up"),
        ("", "", ""),
    ],
)
def test_stored_step_labels_render_in_the_reader_language(stored, en, ru):
    """Labels written by a report in its language are rendered again from their values (forecast pages)."""
    from AI.i18n import has_key
    from AI.report_texts import STEP_LABEL_KEYS, localized_step_label

    assert all(has_key(f"report.step_label.{key}", lang) for key in STEP_LABEL_KEYS for lang in LANGS)
    assert localized_step_label(stored, "en") == en
    assert localized_step_label(stored, "ru") == ru
    with with_locale("en"):
        assert localized_step_label(stored) == en
