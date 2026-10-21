# SPDX-License-Identifier: Apache-2.0
"""Chart markers, SLA thresholds and series colors: AI/report_markers.py and its JS mirror in
static/js/reports.js must agree in every language (labels are dictionary texts: the UI locale on the page, the
locale of the caller in Python), and the Confluence PNG must draw the same items (ui-plan 1.1, 1.2, 1.7)."""

import io
import json
import random
from datetime import datetime, timedelta, timezone

import loadlens_app  # noqa: F401  (import before update_page-dependent modules)
import pandas as pd
import pytest

from AI import report_markers as rm
from AI import sla_evaluator
from AI.i18n import with_locale
from AI.context_pack import format_iso
from loadlens_app import confluence_export, demo_data
from tests.js_harness import run_js, theme_css_vars

REPORT_JS = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"]
START = datetime(2026, 10, 15, 14, 26, tzinfo=timezone.utc)


def js(value):
    return json.dumps(value, ensure_ascii=False)


def js_markers(row, table=None, locale="ru"):
    return run_js(REPORT_JS, f"LL.report._internals.reportMarkers({js(row)}, {js(table)})", locale=locale)


def js_thresholds(row, table=None, locale="ru"):
    return run_js(REPORT_JS, f"LL.report._internals.slaThresholds({js(row)}, {js(table)})", locale=locale)


# ---- rows ------------------------------------------------------------------------------------

def demo_row():
    """The final row the demo seed stores (loadlens_app/demo_data.seed_demo_run), built without a database."""
    end = START + timedelta(minutes=demo_data.DEMO_DURATION_MIN)
    index = pd.date_range(START, periods=demo_data.DEMO_DURATION_MIN, freq="1min", tz="UTC")
    rng = random.Random(42)
    rps = demo_data._rps_profile(demo_data.DEMO_DURATION_MIN, rng)
    frames = demo_data._lt_frames(index, rps, rng)
    steps, _drop = demo_data._demo_steps(START, end, 3)
    table = demo_data.demo_step_table(steps, frames)
    results = demo_data._demo_results("17:30", "17:31", {})
    row = {
        "domain": "final",
        "test_type": "step",
        "parsed": results["final_parsed"],
        "sla_details": {
            "checks": results["sla_checks"], "summary": results["sla_summary"], "load_step_table": table,
            "stable_window": demo_data.demo_sla_window(steps),
        },
    }
    return json.loads(json.dumps(row))


def _steps(offset="+03:00"):
    out = []
    for i in range(5):
        start = START + timedelta(minutes=10 * i)
        end = start + timedelta(minutes=10)
        shift = timedelta(hours=3) if offset == "+03:00" else timedelta(0)
        fmt = lambda moment: (moment + shift).strftime("%Y-%m-%dT%H:%M:%S") + offset  # noqa: E731
        out.append({
            "index": i + 1, "label": f"ступень {i + 1}", "start_iso": fmt(start), "end_iso": fmt(end),
            "plateau_start_iso": fmt(start + timedelta(minutes=1)), "plateau_end_iso": fmt(end - timedelta(minutes=1)),
            "stable": True, "after_drop": False, "dip": False, "rps": 50.0 * (i + 1), "rps_mean": 50.0 * (i + 1),
            "latency_p95": 200.0 + i, "selected": False,
        })
    return out


def stress_row():
    steps = _steps()
    return {
        "domain": "final",
        "parsed": {
            "test_profile": {"mode": "stress"},
            "saturation": {"rps_drop_iso": "2026-10-15T18:12:00+03:00", "saturation_window": {"start_iso": steps[3]["start_iso"], "end_iso": steps[3]["end_iso"]}},
            "peak_performance": {"max_rps": 200, "max_time": "LLM text", "drop_time": "LLM text"},
        },
        "sla_details": {
            "checks": [
                {"name": "target_rps", "threshold": 180, "actual": 200.0, "passed": True, "message": "RPS 200.0 ≥ целевой 180; label='RPS total'; series='all'"},
                {"name": "p95_latency", "threshold": 300, "actual": 280.5, "passed": True, "query_label": "Latency p95 by request", "series": "POST /checkout",
                 "message": "p95 280.50 мс ≤ порог 300 мс (на ступени, запрос «Old label», серия «x»)"},
                {"name": "error_rate", "threshold": 1.5, "actual": 0.2, "passed": True, "message": "p95 0.20 % ≤ порог 1.5 % (запрос «Errors %»)"},
                {"name": "recovery", "threshold": 5, "actual": 1, "passed": True, "message": "ok"},
                {"name": "memory_usage", "threshold": "", "actual": 50, "passed": None, "message": "запрос «Memory»"},
            ],
            "stable_window": {"start": steps[3]["start_iso"], "end": steps[3]["end_iso"], "level": 200.0, "series": "all", "label": "RPS total",
                              "degraded_level": 250.0, "degraded_checks": ["p95_latency"]},
            "load_step_table": {"rps_query": "RPS total", "rps_series": "all", "latency_query": "Latency p95 by request", "latency_series": None, "steps": steps},
        },
    }


def soak_row():
    return {
        "domain": "final",
        "parsed": {"stability_under_load": {"hold": {"start_iso": "2026-10-15T17:34:00+03:00", "end_iso": "2026-10-15T18:21:00+03:00"}}},
        "sla_details": {"checks": [{"name": "target_rps", "threshold": 100, "actual": 120, "passed": True, "message": "Удержание: средний RPS 120"}],
                        "stable_window": {"start": "2026-10-15 14:34:00+00:00", "end": "2026-10-15 15:21:00+00:00", "level": 120, "label": "RPS total", "series": "all", "kind": "hold"}},
    }


def soak_window_only_row():
    row = soak_row()
    row["parsed"] = {"test_profile": {"mode": "stability"}}
    return row


def spike_row():
    return {
        "domain": "final",
        "parsed": {"spike_analysis": {"detected": True, "spikes": [
            {"index": 1, "label": "всплеск 1", "start_iso": "2026-10-15T17:38:00+03:00", "end_iso": "2026-10-15T17:42:00+03:00"},
            {"index": 2, "label": "", "start_iso": "2026-10-15T18:06:00+03:00", "end_iso": "2026-10-15T18:12:00+03:00"},
            "broken",
            {"index": 4, "label": "без конца", "start_iso": "2026-10-15T18:20:00+03:00"},
        ]}},
        "sla_details": {"checks": [], "stable_window": {"start": "2026-10-15T15:06:00Z", "end": "2026-10-15T15:12:00Z", "kind": "spike", "title": "всплеск 2"}},
    }


def legacy_row():
    """Older report: checks and window only in parsed.deterministic_sla, JSON text columns, naive UTC times,
    an unrecovered RPS drop (after_drop step) and no explicit query fields."""
    steps = _steps(offset="")
    steps[4]["after_drop"] = True
    for step in steps:
        for key in ("start_iso", "end_iso", "plateau_start_iso", "plateau_end_iso"):
            step[key] = step[key].replace("T", " ")
    parsed = {
        "deterministic_sla": {
            "checks": [{"name": "target_rps", "threshold": 140, "actual": 150.2, "passed": True, "message": "RPS 150.2 ≥ целевой 140"},
                       {"name": "p99_latency", "threshold": 900, "actual": 800, "passed": True, "message": "p99 800"}],
            "stable_window": None,
        },
        "load_step_table": {"rps_query": "RPS total", "rps_series": "all", "latency_query": "Latency", "latency_series": "total", "steps": steps},
    }
    return {"domain": "final", "parsed": json.dumps(parsed, ensure_ascii=False), "sla_details": None}


ROWS = {
    "demo": demo_row,
    "stress": stress_row,
    "soak": soak_row,
    "soak_window_only": soak_window_only_row,
    "spike": spike_row,
    "legacy": legacy_row,
    "empty": lambda: {"domain": "final"},
}


# ---- the JS/Python contract --------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize("name", sorted(ROWS))
def test_js_and_python_build_the_same_markers_and_thresholds(name, lang):
    row = ROWS[name]()
    with with_locale(lang):
        assert js_markers(row, locale=lang) == rm.markers_for(row)
        assert js_thresholds(row, locale=lang) == rm.thresholds_for(row)


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_js_and_python_agree_with_a_resolved_step_table(lang):
    row = stress_row()
    table = row["sla_details"].pop("load_step_table")
    with with_locale(lang):
        assert rm.markers_for(row) != rm.markers_for(row, table)
        assert js_markers(row, table, locale=lang) == rm.markers_for(row, table)
        assert js_thresholds(row, table, locale=lang) == rm.thresholds_for(row, table)


def test_marker_and_threshold_labels_follow_the_locale():
    with with_locale("en"):
        markers = {m["role"]: m["label"] for m in rm.markers_for(stress_row())}
        thresholds = {t["name"]: t["label"] for t in rm.thresholds_for(stress_row())}
        hold = rm.markers_for(soak_row())[0]["label"]
        spikes = [m["label"] for m in rm.markers_for(spike_row())]
    assert markers == {"sla": "SLA: step 4", "step": "", "degradation": "degradation 18:06", "rps_drop": "RPS drop 18:12"}
    assert thresholds == {"target_rps": "target ≥ 180 RPS", "p95_latency": "p95 ≤ 300 ms", "error_rate": "errors ≤ 1.5 %"}
    assert hold == "hold window"
    # Spike labels stored with the report stay as stored; a spike without one is named by its index.
    assert spikes == ["всплеск 1", "spike 2"]
    window_only = spike_row()
    window_only["parsed"]["spike_analysis"]["spikes"] = []
    window_only["sla_details"]["stable_window"].pop("title")
    with with_locale("en"):
        assert [m["label"] for m in rm.markers_for(window_only)] == ["spike"]
        assert rm.markers_for(legacy_row())[0]["label"] == "SLA: step 3"


def test_query_and_series_are_parsed_from_ru_and_en_check_texts():
    window = {}
    for message in ("p95 280.50 мс ≤ порог 300 мс (на ступени, запрос «Latency», серия «POST /x»)",
                    "p95 280.50 ms ≤ threshold 300 ms (on the step, query «Latency», series «POST /x»)",
                    "RPS 200.0; label='Latency'; series='POST /x'"):
        check = {"name": "p95_latency", "threshold": 300, "message": message}
        assert rm.check_source(check, window, {}) == ("Latency", "POST /x")
        assert run_js(REPORT_JS, f"LL.report._internals.slaThresholds({js({'sla_details': {'checks': [check]}})}, null)", locale="en") == [
            {"name": "p95_latency", "query_label": "Latency", "series": "POST /x", "value": 300, "label": "p95 ≤ 300 ms", "tone": "fail"}]
    # Explicit fields win over the text.
    explicit = {"name": "p95_latency", "query_label": "Real", "series": "s", "message": "query «Old», series «x»"}
    assert rm.check_source(explicit, window, {}) == ("Real", "s")


def test_demo_row_markers_and_thresholds():
    row = demo_row()
    markers = rm.markers_for(row)
    steps = row["sla_details"]["load_step_table"]["steps"]
    window = row["sla_details"]["stable_window"]
    # The SLA step is the last one (250 RPS): an accent band, then a tick at every later step boundary.
    assert steps[-1]["selected"] is True and not any(step.get("selected") for step in steps[:-1])
    assert markers[0] == {"kind": "band", "t": rm.moment_ms(window["start"]), "to": rm.moment_ms(window["end"]),
                          "label": f"SLA: ступень {len(steps)}", "tone": "accent", "role": "sla"}
    ticks = [m for m in markers if m["role"] == "step"]
    assert [m["t"] for m in ticks] == [rm.moment_ms(step["start_iso"]) for step in steps[1:]]
    assert {m["label"] for m in ticks} == {""}
    assert js_markers(row) == markers
    thresholds = rm.thresholds_for(row)
    # Explicit query labels on the checks win over parsing the message.
    assert [(t["name"], t["query_label"], t["label"], t["tone"]) for t in thresholds] == [
        ("target_rps", demo_data.DEMO_RPS_LABEL, "цель ≥ 200 RPS", "accent"),
        ("p95_latency", demo_data.DEMO_P95_TOTAL_LABEL, "p95 ≤ 300 мс", "fail"),
        ("error_rate", demo_data.DEMO_ERROR_RATE_LABEL, "ошибки ≤ 1 %", "fail"),
    ]


def test_stress_markers_band_ticks_degradation_and_drop():
    markers = rm.markers_for(stress_row())
    by_role = {m["role"]: m for m in markers}
    steps = stress_row()["sla_details"]["load_step_table"]["steps"]
    assert by_role["sla"] == {"kind": "band", "t": rm.moment_ms(steps[3]["start_iso"]), "to": rm.moment_ms(steps[3]["end_iso"]),
                              "label": "SLA: ступень 4", "tone": "accent", "role": "sla"}
    assert [m["t"] for m in markers if m["role"] == "step"] == [rm.moment_ms(s["start_iso"]) for s in steps[1:]]
    assert by_role["degradation"]["t"] == rm.moment_ms(steps[4]["start_iso"])
    assert by_role["degradation"]["label"] == "деградация 18:06"
    assert by_role["degradation"]["tone"] == "fail"
    assert by_role["rps_drop"]["label"] == "падение RPS 18:12"
    assert [m["role"] for m in markers] == ["sla", "step", "step", "step", "step", "degradation", "rps_drop"]


def test_rps_drop_next_to_the_degradation_line_is_not_drawn_twice():
    row = stress_row()
    row["parsed"]["saturation"]["rps_drop_iso"] = "2026-10-15T18:06:30+03:00"
    roles = [m["role"] for m in rm.markers_for(row)]
    assert "degradation" in roles and "rps_drop" not in roles
    assert js_markers(row) == rm.markers_for(row)


def test_llm_peak_times_never_become_markers():
    row = stress_row()
    row["parsed"]["peak_performance"].update({"max_time": "2026-10-15T18:00:00+03:00", "drop_time": "2026-10-15T18:01:00+03:00"})
    assert rm.markers_for(row) == rm.markers_for(stress_row())


def test_soak_spike_and_legacy_markers():
    assert rm.markers_for(soak_row()) == [{"kind": "band", "t": rm.moment_ms("2026-10-15T14:34:00Z"), "to": rm.moment_ms("2026-10-15T15:21:00Z"),
                                           "label": "окно удержания", "tone": "accent", "role": "hold"}]
    assert rm.markers_for(soak_window_only_row()) == rm.markers_for(soak_row())
    spikes = rm.markers_for(spike_row())
    assert [(m["label"], m["tone"]) for m in spikes] == [("всплеск 1", "warn"), ("всплеск 2", "warn")]
    legacy = rm.markers_for(legacy_row())
    # Selected by target RPS (150.2 ≈ step 3), naive times read as UTC, the after_drop step is the RPS drop.
    assert legacy[0]["label"] == "SLA: ступень 3"
    assert legacy[0]["t"] == rm.moment_ms("2026-10-15T14:47:00Z")
    assert legacy[-1]["role"] == "rps_drop" and legacy[-1]["label"] == "падение RPS 18:06"


def test_thresholds_prefer_explicit_fields_then_message_then_window_and_table():
    by_name = {t["name"]: t for t in rm.thresholds_for(stress_row())}
    assert by_name["p95_latency"]["query_label"] == "Latency p95 by request"
    assert by_name["p95_latency"]["series"] == "POST /checkout"
    assert by_name["target_rps"]["query_label"] == "RPS total" and by_name["target_rps"]["series"] == "all"
    assert by_name["error_rate"]["query_label"] == "Errors %" and by_name["error_rate"]["label"] == "ошибки ≤ 1.5 %"
    assert "recovery" not in by_name and "memory_usage" not in by_name
    legacy = {t["name"]: t for t in rm.thresholds_for(legacy_row())}
    assert legacy["target_rps"]["query_label"] == "RPS total"
    assert (legacy["p99_latency"]["query_label"], legacy["p99_latency"]["series"], legacy["p99_latency"]["label"]) == ("Latency", "total", "p99 ≤ 900 мс")


def test_thresholds_for_query_resolves_partial_labels_like_the_evaluator():
    thresholds = [{"query_label": "p95(ms) by group", "value": 300, "label": "p95 ≤ 300 мс", "tone": "fail"}]
    labels = ["LT: http_req_duration p95(ms) by group & name", "LT: RPS"]
    assert rm.thresholds_for_query(thresholds, labels[0], labels) == thresholds
    assert rm.thresholds_for_query(thresholds, labels[0]) == []
    assert rm.resolve_query_label("RPS", ["RPS a", "RPS b"]) == "RPS"
    assert rm.resolve_query_label("lt: rps", labels) == "LT: RPS"
    result = run_js(REPORT_JS, f"LL.report._internals.thresholdsForQuery({js(thresholds)}, {js(labels[0])}, {js(labels)})")
    assert result == [{"value": 300, "label": "p95 ≤ 300 мс", "tone": "fail"}]
    assert run_js(REPORT_JS, f"[LL.report._internals.resolveQueryLabel('RPS', ['RPS a', 'RPS b']), LL.report._internals.resolveQueryLabel('lt: rps', {js(labels)})]") == ["RPS", "LT: RPS"]


# ---- series order and palette ------------------------------------------------------------------

def test_series_palette_matches_the_light_theme_tokens_and_the_js_palette():
    css = run_js(REPORT_JS, "Array.from({ length: 24 }, (_, i) => LL.charts.palette(i))", theme="light")
    assert [rm.series_color(i) for i in range(24)] == css
    assert list(rm.SERIES_PALETTE) == css[:8]


def test_png_tones_match_the_light_theme_tokens():
    light = theme_css_vars()["light"]
    tokens = {"accent": "--accent", "ok": "--status-ok", "warn": "--status-warn", "fail": "--status-fail", "muted": "--text-dim"}
    assert {tone: light[name] for tone, name in tokens.items()} == rm.TONE_COLORS


def test_series_order_matches_the_web_chart():
    values = {"b": [1, 1], "a": [1, None, 1], "big": [10, 20], "neg": [-30, -30], "empty": [None], "text": ["5", "x"]}
    order = rm.order_series(values)
    assert order == ["neg", "big", "text", "a", "b", "empty"]
    assert run_js(REPORT_JS, f"LL.report._internals.orderSeries({js(values)})") == order


def test_png_series_plan_uses_the_web_order_and_colors():
    points = [{"t": f"2026-10-15T10:0{i}:00+00:00", "series": name, "value": value * (i + 1)}
              for i in range(3) for name, value in (("low", 1.0), ("high", 5.0), ("mid", 3.0))]
    labels, ranked = confluence_export.series_plan(points)
    assert labels == sorted({p["t"] for p in points})
    web = run_js(REPORT_JS, f"(() => {{ const d = LL.report._internals.buildChartData({js(points)}); return d.datasets.map((s) => [s.label, s.borderColor, s.llSeries]); }})()", theme="light")
    assert [[name, color] for name, color, _ in ranked] == [[label, color] for label, color, _ in web]
    assert [slot for _, _, slot in web] == [0, 1, 2]


# ---- Confluence PNG ----------------------------------------------------------------------------

def _points(minutes=30):
    return [{"t": (START + timedelta(minutes=i)).isoformat(), "series": name, "value": base + i}
            for i in range(minutes) for name, base in (("POST /checkout", 200.0), ("GET /search", 120.0))]


def test_marker_geometry_places_items_like_the_web_plugin():
    times = [rm.moment_ms((START + timedelta(minutes=i)).isoformat()) for i in range(10)]
    markers = [
        {"kind": "band", "t": times[2] + 30_000, "to": times[5], "label": "SLA: ступень 2", "tone": "accent"},
        {"kind": "line", "t": times[7], "to": None, "label": "деградация", "tone": "fail"},
        {"kind": "tick", "t": times[3], "to": None, "label": "", "tone": "muted"},
        {"kind": "line", "t": times[9] + 31_000, "to": None, "label": "after the end", "tone": "fail"},
        {"kind": "band", "t": times[0] - 600_000, "to": times[0] - 300_000, "label": "before", "tone": "warn"},
        {"kind": "band", "t": times[8], "to": times[9] + 600_000, "label": "clipped", "tone": "warn"},
    ]
    placed = confluence_export.marker_geometry(markers, [float(t) for t in times])
    assert placed == [
        {"kind": "band", "x0": 2.5, "x1": 5.0, "label": "SLA: ступень 2", "tone": "accent"},
        {"kind": "line", "x": 7.0, "label": "деградация", "tone": "fail"},
        {"kind": "tick", "x": 3.0, "label": "", "tone": "muted"},
        {"kind": "band", "x0": 8.0, "x1": 9.0, "label": "clipped", "tone": "warn"},
    ]


def _pixels_near(png, hex_color, tolerance=0.06):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import numpy as np

    image = mpimg.imread(io.BytesIO(png), format="png")[..., :3]
    target = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return int((np.abs(image - target).max(axis=-1) < tolerance).sum())


def test_png_draws_thresholds_markers_and_palette_colors():
    pytest.importorskip("matplotlib")
    points = _points()
    times = [rm.moment_ms(p["t"]) for p in points[::2]]
    plain = confluence_export.render_series_png(points, "p95")
    marked = confluence_export.render_series_png(
        points, "p95",
        markers=[{"kind": "band", "t": times[5], "to": times[12], "label": "SLA: ступень 2", "tone": "accent", "role": "sla"},
                 {"kind": "line", "t": times[20], "to": None, "label": "деградация 18:06", "tone": "fail", "role": "degradation"}],
        thresholds=[{"value": 400, "label": "p95 ≤ 400 мс", "tone": "fail"}],
        clock_offset=3.0,
    )
    assert plain[:8] == b"\x89PNG\r\n\x1a\n" and marked[:8] == b"\x89PNG\r\n\x1a\n"
    fail = rm.TONE_COLORS["fail"]
    assert _pixels_near(plain, fail) == 0
    assert _pixels_near(marked, fail) > 300  # threshold above the data (y axis extended) + degradation line
    # Series colors are the palette slots in web order (largest mean first), not hashes of the names.
    assert _pixels_near(plain, rm.series_color(0)) > 200
    assert _pixels_near(plain, rm.series_color(1)) > 200


def test_publish_report_passes_markers_and_query_thresholds(monkeypatch):
    row = stress_row()
    table = row["sla_details"]["load_step_table"]
    rendered = []

    def fake_render(points, title, **kwargs):
        rendered.append((title, kwargs))
        return b"png"

    monkeypatch.setattr(confluence_export, "_ts_conn", lambda: (_ for _ in ()).throw(AssertionError("no database")))
    monkeypatch.setattr(confluence_export, "confluence_config", lambda: {"enabled": True, "base_url": "https://wiki.local", "username": "u",
                                                                         "password": "p", "space_key": "LT", "parent_page_id": "1"})
    monkeypatch.setattr(confluence_export, "load_llm_rows", lambda run: [dict(row, service="svc", run_name=run)])
    monkeypatch.setattr(confluence_export, "load_engineer_html", lambda run: "")
    monkeypatch.setattr(confluence_export, "load_domains_schema", lambda run: {"lt_framework": ["RPS total", "Latency p95 by request (all)", "Other"]})
    monkeypatch.setattr(confluence_export, "load_run_series", lambda run, domain, ql: _points(3))
    monkeypatch.setattr(confluence_export, "render_series_png", fake_render)
    monkeypatch.setattr(confluence_export, "resolve_load_step_table", lambda run, rows: table)
    monkeypatch.setattr(confluence_export, "load_publication", lambda run: None)
    monkeypatch.setattr(confluence_export, "upsert_page", lambda *args, **kwargs: ("10", "https://wiki.local/p/10"))
    monkeypatch.setattr(confluence_export, "save_publication", lambda **kwargs: None)
    result = confluence_export.publish_report("run-1")
    assert result["page_id"] == "10"
    by_title = {title: kwargs for title, kwargs in rendered}
    expected = rm.markers_for(row, table)
    assert all(kwargs["markers"] == expected for kwargs in by_title.values())
    assert [t["label"] for t in by_title["RPS total"]["thresholds"]] == ["цель ≥ 180 RPS"]
    # «Latency p95 by request» is a unique part of the chart label, as the evaluator matches SLA queries.
    assert [t["label"] for t in by_title["Latency p95 by request (all)"]["thresholds"]] == ["p95 ≤ 300 мс"]
    assert by_title["Other"]["thresholds"] == []
    assert by_title["Other"]["clock_offset"] == 3.0


# ---- explicit query fields of SLA checks ---------------------------------------------------------

def test_sla_checks_carry_query_label_and_series_without_changing_the_message():
    frame = pd.DataFrame({"POST /checkout": [100.0, 200.0, 300.0], "GET /search": [50.0, 60.0, 70.0]},
                         index=pd.date_range("2026-10-15T10:00:00Z", periods=3, freq="1min"))
    labeled = [{"label": "LT: http_req_duration p95(ms) by group & name", "df": frame}]
    check = sla_evaluator._check_by_query_label(
        {"max_p95_ms": 250, "p95_query": "p95(ms) by group"}, name="p95_latency", threshold_key="max_p95_ms",
        query_key="p95_query", labeled=labeled, window=None, unit="мс", category="primary",
    )
    assert check["query_label"] == "LT: http_req_duration p95(ms) by group & name"
    assert check["series"] == "POST /checkout"
    assert "запрос «p95(ms) by group», серия «POST /checkout»" in check["message"]
    missing = sla_evaluator._check_by_query_label(
        {"max_p95_ms": 250, "p95_query": "absent"}, name="p95_latency", threshold_key="max_p95_ms",
        query_key="p95_query", labeled=labeled, window=None, unit="мс", category="primary",
    )
    assert missing["query_label"] == "absent" and "series" not in missing
    plain = sla_evaluator._make_check("recovery", 5, 1, True)
    assert "query_label" not in plain and "series" not in plain


def test_capacity_target_check_names_the_window_query():
    window = sla_evaluator.StableWindow(start="2026-10-15T10:00:00Z", end="2026-10-15T10:10:00Z", level=210.0, series="all", label="RPS total")
    check = sla_evaluator._capacity_target_check({}, 200.0, window, {}, "RPS total")
    assert (check["query_label"], check["series"]) == ("RPS total", "all")
    assert "label='RPS total'; series='all'" in check["message"]
    assert rm.thresholds_for({"sla_details": {"checks": [check]}})[0]["query_label"] == "RPS total"


def test_offset_and_clock_follow_the_step_table():
    steps = [{"start_iso": "2026-10-15T10:00:00+05:30", "end_iso": "2026-10-15T10:10:00+05:30"}]
    assert rm.offset_hours(steps) == 5.5
    assert rm.offset_hours([]) == rm.DEFAULT_OFFSET_HOURS
    assert rm.clock(rm.moment_ms("2026-10-15T04:40:00Z"), 5.5) == "10:10"
    assert rm.moment_ms("2026-10-15 10:00:00") == rm.moment_ms("2026-10-15T10:00:00Z")
    assert rm.moment_ms("garbage") is None and rm.moment_ms(None) is None
    assert run_js(REPORT_JS, "LL.report._internals.utcMoment('2026-10-15 10:00:00')", tz="Europe/Moscow") == rm.moment_ms("2026-10-15T10:00:00Z")
    assert format_iso(START, 3).endswith("+03:00")
