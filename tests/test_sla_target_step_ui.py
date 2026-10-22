# SPDX-License-Identifier: Apache-2.0
"""A step test decided on its target step, as the report shows it: two bands on the charts (target step and ceiling),
the step row tags, the SLA caption, the margin check, Confluence, the dashboard profile and the model context."""

import copy
import json
import re

import loadlens_app  # noqa: F401  (import before update_page-dependent modules)
import pandas as pd
import pytest

from AI import report_markers as rm
from AI.context_pack import LoadStep
from AI.i18n import with_locale
from AI.pipeline import _deterministic_sla_context, _load_step_table_for_report, _sla_step_context
from AI.prompt_assembly import sla_guidance_lines
from AI.sla_evaluator import evaluate_sla
from loadlens_app import confluence_export
from loadlens_app.blueprints.dashboard import _dashboard_profile
from tests.js_harness import run_js
from tests.test_report_markers import REPORT_JS, _steps, js, js_markers, js_thresholds

I = "LL.report._internals"
CHECKS = [
    {"name": "target_rps", "threshold": 100, "actual": 200.0, "passed": True, "query_label": "RPS total", "series": "all",
     "message": "RPS 200.0 ≥ целевой 100; label='RPS total'; series='all'"},
    {"name": "p95_latency", "threshold": 300, "actual": 280.0, "passed": True, "query_label": "Latency", "series": "total", "message": ""},
    {"name": "cpu_usage", "threshold": 90, "actual": 60.0, "passed": True, "query_label": "CPU", "series": "node", "message": ""},
]


def _window(step, level, **extra):
    return {"start": step["start_iso"], "end": step["end_iso"], "level": level, "series": "all", "label": "RPS total", **extra}


def target_row(role="target", same=False, flags=True):
    """Steps 50…250 RPS, target 100: the target (or margin) step 2, the ceiling step 4 (step 2 when ``same``),
    the step after the ceiling broke p95. ``flags``: the step table carries selected/sla_role/ceiling."""
    steps = _steps()
    ceiling_at = 1 if same else 3
    if flags:
        steps[1].update(selected=True, sla_role=role)
        if not same:
            steps[3]["ceiling"] = True
    target = _window(steps[1], 100.0, role=role, **({"margin_pct": 50} if role == "margin" else {}))
    window = _window(steps[ceiling_at], steps[ceiling_at]["rps"], degraded_level=steps[ceiling_at + 1]["rps"], degraded_checks=["p95_latency"])
    return {
        "domain": "final",
        "parsed": {"test_profile": {"mode": "capacity"}},
        "sla_details": {
            "checks": copy.deepcopy(CHECKS), "stable_window": window, "target_window": target,
            "load_step_table": {"rps_query": "RPS total", "rps_series": "all", "latency_query": "Latency", "latency_series": "total", "steps": steps},
        },
    }


ROWS = {
    "target": target_row,
    "margin": lambda: target_row(role="margin"),
    "same_step": lambda: target_row(same=True),
    "no_flags": lambda: target_row(flags=False),
    "only_in_parsed": lambda: _only_in_parsed(target_row()),
}


def _only_in_parsed(row):
    sla = row.pop("sla_details")
    row["parsed"]["deterministic_sla"] = {"checks": sla["checks"], "stable_window": sla["stable_window"], "target_window": sla["target_window"]}
    row["parsed"]["load_step_table"] = sla["load_step_table"]
    return row


def text(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


# ---- chart markers: the JS/Python contract and the two bands ---------------------------------------------------

@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize("name", sorted(ROWS))
def test_js_and_python_build_the_same_markers(name, lang):
    row = ROWS[name]()
    with with_locale(lang):
        assert js_markers(row, locale=lang) == rm.markers_for(row)
        assert js_thresholds(row, locale=lang) == rm.thresholds_for(row)


def test_target_band_ceiling_band_and_degradation_after_the_ceiling():
    row = target_row()
    steps = row["sla_details"]["load_step_table"]["steps"]
    markers = rm.markers_for(row)
    assert [m["role"] for m in markers] == ["sla", "ceiling", "step", "step", "step", "step", "degradation"]
    sla, ceiling = markers[0], markers[1]
    assert (sla["label"], sla["tone"], sla["t"]) == ("Цель: ступень 2", "accent", rm.moment_ms(steps[1]["start_iso"]))
    assert (ceiling["label"], ceiling["tone"], ceiling["t"], ceiling["to"]) == (
        "Максимум: ступень 4", "ok", rm.moment_ms(steps[3]["start_iso"]), rm.moment_ms(steps[3]["end_iso"]))
    assert markers[-1]["t"] == rm.moment_ms(steps[4]["start_iso"])
    # A table without the flags: the steps are found by the window levels, not by the actual target RPS.
    assert [m["label"] for m in rm.markers_for(target_row(flags=False))][:2] == ["Цель: ступень 2", "Максимум: ступень 4"]
    with with_locale("en"):
        labels = [m["label"] for m in rm.markers_for(row)][:2]
        margin = rm.markers_for(target_row(role="margin"))[0]["label"]
    assert labels == ["Target: step 2", "Max: step 4"]
    assert margin == "Margin: step 2"


def test_target_step_that_is_the_ceiling_draws_one_band():
    row = target_row(same=True)
    markers = rm.markers_for(row)
    assert [m["role"] for m in markers if m["kind"] == "band"] == ["sla"]
    assert markers[0]["label"] == "Цель: ступень 2"
    steps = row["sla_details"]["load_step_table"]["steps"]
    assert next(m for m in markers if m["role"] == "degradation")["t"] == rm.moment_ms(steps[2]["start_iso"])


def test_reports_without_a_target_window_keep_their_markers():
    row = target_row()
    row["sla_details"].pop("target_window")
    for step in row["sla_details"]["load_step_table"]["steps"]:
        step.pop("sla_role", None)
        step.pop("ceiling", None)
        step["selected"] = step["index"] == 4
    markers = rm.markers_for(row)
    assert [m["role"] for m in markers if m["kind"] == "band"] == ["sla"] and markers[0]["label"] == "SLA: ступень 4"
    assert js_markers(row) == markers


def test_profile_draws_the_target_zone_and_the_ceiling_band():
    model = run_js(REPORT_JS, f"""(() => {{
      const row = {js(target_row())};
      const table = row.sla_details.load_step_table;
      const linked = {I}.linkSlaToSteps(table, row);
      const m = LL.charts.stepProfileModel({{ steps: linked.steps, markers: {I}.reportMarkers(row, table), target: 100, width: 1000 }});
      return m.bands.map((b) => [b.role, b.label, b.tone]);
    }})()""")
    assert model == [["sla", "Цель: ступень 2", "accent"], ["ceiling", "Максимум: ступень 4", "ok"]]


# ---- step row, SLA caption, margin check ------------------------------------------------------------------------

def _step_row(row, locale="ru"):
    return run_js(REPORT_JS, f"""(() => {{
      const row = {js(row)};
      const table = row.sla_details ? row.sla_details.load_step_table : row.parsed.load_step_table;
      const linked = {I}.linkSlaToSteps(table, row);
      return {{ html: {I}.renderLoadStepTableHtml(Object.assign({{}}, table, linked)), note: linked.slaNote }};
    }})()""", locale=locale)


@pytest.mark.parametrize("name", ["target", "no_flags"])
def test_step_row_tags_the_target_step_and_the_ceiling(name):
    out = _step_row(ROWS[name]())
    cells = re.findall(r'<div class="(rep-step[^"]*)" role="listitem"[^>]*>(.*?)</div>', out["html"])
    assert [c[0] for c in cells] == ["rep-step", "rep-step is-sla", "rep-step", "rep-step is-ceiling", "rep-step"]
    assert '<span class="rep-step-tag">цель</span>' in cells[1][1]
    assert 'class="rep-step-tag is-ceiling"' in cells[3][1] and ">макс.</span>" in cells[3][1]
    assert out["note"].startswith("Цель подтверждена на ступени 2 (17:37–17:45); пороги задержек и ошибок выполняются до ступени 4 (≈200 RPS). ")


def test_step_row_of_a_margin_step_and_in_english():
    out = _step_row(target_row(role="margin"), locale="en")
    assert '<span class="rep-step-tag">margin</span>' in out["html"] and ">max</span>" in out["html"]
    assert out["note"].startswith("The margin over the target is confirmed on step 2 (17:37–17:45); the latency and error thresholds hold up to step 4 (≈200 RPS).")
    same = _step_row(target_row(same=True))
    assert "is-ceiling" not in same["html"] and '<span class="rep-step-tag">цель</span>' in same["html"]


def test_sla_caption_names_the_target_step_and_the_ceiling():
    rows = [target_row(), target_row(role="margin"), target_row(same=True)]
    out = run_js(REPORT_JS, f"{js(rows)}.map((r) => {I}.slaScopeText(r))", locale="ru")
    assert out == [
        "цель — ступень 2 · 17:36–17:46, максимум — ступень 4",
        "запас — ступень 2 · 17:36–17:46, максимум — ступень 4",
        "ступень 2 · 17:36–17:46",
    ]


def test_target_not_confirmed_is_said_so_without_a_figure():
    row = target_row()
    row["sla_details"]["checks"][0].update(passed=False, actual=None)
    out = run_js(REPORT_JS, f"""(() => {{
      const row = {js(row)};
      const tile = {I}.heroKpis(row)[0];
      return {{ sentence: {I}.verdictSummaryHtml(row), state: tile.state }};
    }})()""")
    assert text(out["sentence"]).startswith("Целевой RPS не подтверждён: ни на одной ступени от цели не выполнены пороги задержек и ошибок")
    assert out["state"] == "не подтверждена"


def test_margin_check_is_a_floor_in_the_table_and_the_sentence():
    margin = {"name": "target_margin", "threshold": 150, "actual": 120, "passed": False, "message": "", "category": "secondary"}
    row = target_row()
    row["sla_details"]["checks"].insert(1, margin)
    out = run_js(REPORT_JS, f"""(() => {{
      const row = {js(row)};
      return {{ table: {I}.renderSlaChecklist(row), sentence: {I}.verdictSummaryHtml(row), bar: {I}.slaBarModel({js(margin)}) }};
    }})()""")
    assert "Запас над целевым RPS" in out["table"] and "≥ 150" in text(out["table"])
    assert out["bar"]["higherBetter"] is True
    assert text(out["sentence"]).startswith(
        "Целевой RPS достигнут на стабильной ступени, но запас над целью не подтверждён: пороги выполняются до 120 RPS при требуемых 150 RPS")


# ---- Confluence and the dashboard -------------------------------------------------------------------------------

def test_confluence_names_the_target_step_and_the_ceiling():
    row = target_row()
    table = row["sla_details"]["load_step_table"]
    with with_locale("ru"):
        names = [confluence_export._step_name(step) for step in table["steps"]]
        note = confluence_export.sla_window_note(confluence_export.sla_window_of(row), table, confluence_export.sla_target_window_of(row))
        block = confluence_export.sla_criteria_block(row, table)
    assert names == ["1", "2 цель", "3", "4 макс.", "5"]
    assert note == "Цель подтверждена на ступени 2 (17:37–17:45); пороги задержек и ошибок выполняются до ступени 4 (≈200 RPS)."
    assert note in block
    margin = target_row(role="margin")
    with with_locale("en"):
        assert confluence_export._step_name(margin["sla_details"]["load_step_table"]["steps"][1]) == "2 margin"
        assert confluence_export.sla_check_label("target_margin") == "Margin over the target RPS"
    legacy = {"index": 4, "selected": True}
    assert confluence_export._step_name(legacy) == "4 SLA"


def test_dashboard_profile_shows_the_target_step_and_the_ceiling_band():
    row = target_row()
    profile = _dashboard_profile(json.dumps(row["sla_details"]), json.dumps(row["parsed"]), None)
    steps = row["sla_details"]["load_step_table"]["steps"]
    assert profile["window"] == {"start": steps[1]["start_iso"], "end": steps[1]["end_iso"]}
    assert [(m["role"], m["label"]) for m in profile["markers"] if m["kind"] == "band"] == [("ceiling", "Максимум: ступень 4")]


# ---- pipeline: the model context and the stored step table ------------------------------------------------------

def _load_steps():
    index = pd.date_range("2026-10-01T10:00:00Z", periods=30, freq="1min")
    steps = []
    for position, level in enumerate((100.0, 200.0, 300.0)):
        start, end = index[position * 10], index[position * 10 + 9]
        steps.append(LoadStep(
            index=position + 1, label=f"ступень {position + 1} · ≈{level:.0f} RPS", start=start, end=end,
            minute_from=position * 10.0, minute_to=position * 10.0 + 9, rps_level=level, stable=True, source="test", shift_hours=3,
        ))
    rps = [100.0] * 10 + [200.0] * 10 + [300.0] * 10
    latency = [300.0] * 10 + [400.0] * 10 + [500.0] * 10
    cpu = [40.0] * 10 + [60.0] * 10 + [95.0] * 10
    data = {
        "lt_framework": {
            "pack": {"load_steps": [{"index": s.index, "start_iso": str(s.start), "end_iso": str(s.end), "rps_level": s.rps_level} for s in steps]},
            "labeled": [
                {"label": "RPS", "df": pd.DataFrame({"all": rps}, index=index)},
                {"label": "p95", "df": pd.DataFrame({"all": latency}, index=index)},
            ],
        },
        "hard_resources": {"labeled": [{"label": "CPU", "df": pd.DataFrame({"node": cpu}, index=index)}]},
    }
    return steps, data


SLA = {"target_rps": 100, "max_p95_ms": 1000, "p95_query": "p95", "max_cpu_pct": 90, "cpu_query": "CPU", "max_performance_query": "RPS"}


def test_domain_models_see_the_target_step_with_the_ceiling_next_to_it():
    steps, data = _load_steps()
    result = evaluate_sla(data, SLA, test_profile={"mode": "capacity"})
    step = _sla_step_context(result, steps, 3)
    assert (step["step_index"], step["rps"], step["role"]) == (1, 100.0, "target")
    assert (step["ceiling_step_label"], step["ceiling_rps"]) == ("ступень 3 · ≈300 RPS", 300.0)
    assert step["end_iso"] == "2026-10-01T13:09:00+03:00"
    context = _deterministic_sla_context(result)
    assert context["target_window"]["level"] == 100.0 and context["headroom"]["above_target_failed"] == ["cpu_usage"]
    # Without a target the ceiling stays the SLA step and nothing is added.
    plain = evaluate_sla(data, {k: v for k, v in SLA.items() if k != "target_rps"}, test_profile={"mode": "capacity"})
    plain_step = _sla_step_context(plain, steps, 3)
    assert plain_step["rps"] == 300.0 and "role" not in plain_step and "ceiling_rps" not in plain_step
    assert "target_window" not in _deterministic_sla_context(plain)


def test_stored_step_table_marks_the_target_step_and_the_ceiling():
    steps, data = _load_steps()
    result = evaluate_sla(data, SLA, test_profile={"mode": "capacity"})
    table = _load_step_table_for_report(steps, data["lt_framework"]["labeled"], SLA, result["stable_window"], result["target_window"])
    rows = table["steps"]
    assert [row["selected"] for row in rows] == [True, False, False]
    assert rows[0]["sla_role"] == "target" and rows[2]["ceiling"] is True
    assert "ceiling" not in rows[0] and "sla_role" not in rows[2] and "ceiling" not in rows[1]
    without = _load_step_table_for_report(steps, data["lt_framework"]["labeled"], SLA, result["stable_window"])
    assert [row["selected"] for row in without["steps"]] == [False, False, True]
    assert not any("sla_role" in row or "ceiling" in row for row in without["steps"])


def test_guidance_tells_the_model_that_breaches_above_the_target_are_headroom():
    ru = "\n".join(sla_guidance_lines(SLA, "jvm", "capacity", "ru"))
    en = "\n".join(sla_guidance_lines(SLA, None, "stress", "en"))
    assert "ступени цели" in ru and "запас прочности" in ru
    assert "target step" in en and "headroom" in en
    assert "ступени цели" not in "\n".join(sla_guidance_lines({k: v for k, v in SLA.items() if k != "target_rps"}, "jvm", "capacity", "ru"))
    assert "ступени цели" not in "\n".join(sla_guidance_lines(SLA, "jvm", "stability", "ru"))
