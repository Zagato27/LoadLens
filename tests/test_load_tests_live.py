"""Live charts of a running test: windows, reads of new points only, instances, events, SLA lines, and the
live-only rows that reports do not read."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from loadlens_app.load_tests import live
from loadlens_app.load_tests.store import InstanceRecord, RunRecord
from loadlens_app.load_tests.tools import effective_tools
from tests.js_harness import run_js

T0 = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)
RUN_PAGE = ["static/js/common.js", "static/js/ui.js", "static/js/launch_run.js"]


def _cfg(pack_id="lt-k6-prometheus-rw", sla=None):
    from AI.query_packs import load_catalog, merge_pack, render_pack

    block, _ = merge_pack(None, render_pack(load_catalog().pack(pack_id)))
    return {"queries": {"lt_framework": block}, "sla": sla or {}}


def _inst(index, **extra):
    return InstanceRecord(index=index, name=f"i{index}", tag=f"i{index}", generator="gen", tool="k6", **extra)


def _run(instances, **extra):
    return RunRecord(run_id="r", run_name="nightly", run_tag="nightly", service="checkout", instances=instances,
                     options={"max_seconds": 3600}, started_at=T0, **extra)


@pytest.fixture
def source(monkeypatch):
    """Fake metrics source: remembers the reads, answers by the kind of query."""
    calls = []

    def fetch(cfg_, *, domain, lang, query, start_ts, end_ts, step, label_keys=None):
        calls.append({"query": query, "start": start_ts, "end": end_ts, "step": step, "keys": label_keys})
        points = [(ts, 5.0) for ts in range(int(start_ts) + 10, int(end_ts), 30)]
        if "lt_instance" in (label_keys or []):
            return {"lt_instance=i0": points, "lt_instance=i1": [(ts, 2.0) for ts, _ in points], "lt_instance=": points,
                    "lt_instance=other-run": points}
        return {"series": points}

    from loadlens_app import query_preview

    monkeypatch.setattr(query_preview, "fetch_points", fetch)
    return calls


def test_whole_run_and_last_minutes(monkeypatch, source):
    monkeypatch.setattr(live, "_scope_config", lambda run: _cfg())
    run = _run([_inst(0, launched_at=T0, started_at=T0)])
    now = T0.timestamp() + 7200
    whole = live.live_snapshot(run, effective_tools({}), now=now)
    assert whole["start"] == T0.timestamp() and whole["end"] == now and whole["step_sec"] == 30 and whole["window_min"] == 0
    last = live.live_snapshot(run, effective_tools({}), now=now, window_min=15)
    assert last["start"] == now - 900 and last["step_sec"] == 5
    assert live.live_snapshot(run, effective_tools({}), now=now, window_min=7)["window_min"] == 0  # unknown window: whole run
    fresh = whole["series"]["lt.rps_total"]["points"][-1][0] / 1000
    assert whole["fresh_at"] == fresh and fresh <= now


def test_new_points_only_with_the_step_of_the_whole_window(monkeypatch, source):
    monkeypatch.setattr(live, "_scope_config", lambda run: _cfg())
    run = _run([_inst(0, launched_at=T0, started_at=T0)])
    now = T0.timestamp() + 3600
    since = now - 20
    snapshot = live.live_snapshot(run, effective_tools({}), now=now, since=since, mode="filtered")
    step = snapshot["step_sec"]
    assert step == live.choose_step(T0.timestamp(), now) and snapshot["since"] == since
    # Reads start a couple of steps before `since`: the last bucket may have been filling.
    assert {call["start"] for call in source} == {since - live.OVERLAP_STEPS * step}
    assert {call["step"] for call in source} == {f"{step}s"}


def test_mode_keeps_the_choice_of_the_first_read(monkeypatch, source):
    monkeypatch.setattr(live, "_scope_config", lambda run: _cfg())
    run = _run([_inst(0, launched_at=T0, started_at=T0)])
    plain = live.live_snapshot(run, effective_tools({}), now=T0.timestamp() + 600, since=T0.timestamp() + 580, mode="plain")
    assert plain["mode"] == "plain" and not plain["filtered"]
    assert not any('testid="nightly"' in call["query"] for call in source)
    filtered = live.live_snapshot(run, effective_tools({}), now=T0.timestamp() + 600)
    assert filtered["mode"] == "filtered" and any('testid="nightly"' in call["query"] for call in source)


def test_rps_of_each_instance_by_its_tag(monkeypatch, source):
    monkeypatch.setattr(live, "_scope_config", lambda run: _cfg())
    run = _run([_inst(0, launched_at=T0, started_at=T0), _inst(1, launched_at=T0, started_at=T0)])
    snapshot = live.live_snapshot(run, effective_tools({}), now=T0.timestamp() + 600)
    # Series without the tag or of another run are not instances of this one.
    assert [(item["tag"], item["name"]) for item in snapshot["instances"]] == [("i0", "i0"), ("i1", "i1")]
    assert snapshot["instances"][1]["points"][0][1] == 2.0
    single = live.live_snapshot(_run([_inst(0, launched_at=T0, started_at=T0)]), effective_tools({}), now=T0.timestamp() + 600)
    assert single["instances"] == []


def test_events_and_sla_lines(monkeypatch, source):
    monkeypatch.setattr(live, "_scope_config", lambda run: _cfg(sla={"target_rps": 250, "max_error_rate_pct": 1, "max_p95_ms": 0, "p95_query": "x"}))
    first = _inst(0, launched_at=T0, started_at=T0, finished_at=T0 + timedelta(minutes=10), status="finished")
    second = _inst(1, launched_at=T0, started_at=T0 + timedelta(seconds=30), stop_sent_at=T0 + timedelta(minutes=5))
    snapshot = live.live_snapshot(_run([first, second]), effective_tools({}), now=T0.timestamp() + 900)
    kinds = [(event["kind"], event["label"]) for event in snapshot["events"]]
    assert kinds == [("start", "i0: старт"), ("start", "i1: старт"), ("stop", "i1: остановка"), ("finish", "i0: завершён"), ("limit", "лимит длительности")]
    assert snapshot["events"][-1]["ts"] == int((T0.timestamp() + 3600) * 1000)
    assert snapshot["sla"] == {"target_rps": 250.0, "max_error_rate_pct": 1.0}
    # A stop of the whole run is one mark instead of one per instance.
    stopped = live.run_events(_run([second], stop_all_at=T0 + timedelta(minutes=5)))
    assert [event["label"] for event in stopped if event["kind"] == "stop"] == ["остановка прогона"]


def test_reports_do_not_read_live_only_rows():
    from AI.query_packs import strip_live_rows

    queries = _cfg()["queries"]
    block = queries["lt_framework"]
    assert "lt.rps_by_instance" in block["roles"].values()
    stripped = strip_live_rows(queries)["lt_framework"]
    assert "lt.rps_by_instance" not in stripped["roles"].values()
    assert len(stripped["labels"]) == len(block["labels"]) - 1 == len(stripped["promql_queries"]) == len(stripped["label_keys_list"])
    assert all("lt_instance" not in query for query in stripped["promql_queries"])
    assert "lt.rps_by_instance" in block["roles"].values()  # the original stays as it was
    assert strip_live_rows({"jvm": {"labels": ["a"], "promql_queries": ["up"]}}) == {"jvm": {"labels": ["a"], "promql_queries": ["up"]}}


def test_update_report_strips_live_rows(monkeypatch):
    import update_page

    source = (ROOT_DIR / "update_page.py").read_text(encoding="utf-8")
    # The report reads the service queries only after the live-only rows are gone.
    assert source.index("strip_live_rows(ef_cfg.get(\"queries\")") < source.index("apply_run_filter(ef_cfg.get(\"queries\")")
    assert update_page.strip_live_rows is not None


@pytest.mark.parametrize("pack_id, fragment", [
    ("lt-k6-prometheus-rw", "sum by (lt_instance)"),
    ("lt-k6-prometheus-native", "sum by (lt_instance)"),
    ("lt-k6-influxql", 'GROUP BY "lt_instance"'),
    ("lt-jmeter-influxql", '"lt_instance" fill(null)'),
])
def test_tool_packs_have_rps_by_instance(pack_id, fragment):
    from AI.query_packs import load_catalog, render_pack

    pack = load_catalog().pack(pack_id)
    rows = [query for query in render_pack(pack).queries if query.role == "lt.rps_by_instance"]
    assert len(rows) == 1 and fragment in rows[0].query and list(rows[0].keys) == ["lt_instance"]
    # The run filter applies to it like to the other rows of the pack.
    if pack.run_filter is not None:
        assert "{{" + pack.run_filter.variable + "}}" in next(q.query for q in pack.queries if q.role == "lt.rps_by_instance")


# ---- the page -----------------------------------------------------------------------------------------

INTERNALS = "LL.launchRun._internals"


def test_new_points_replace_the_overlap_and_old_ones_leave_the_window():
    old = [[1000, 1], [2000, 2], [3000, 3], [4000, 4]]
    fresh = [[3000, 30], [5000, 50]]
    assert run_js(RUN_PAGE, f"{INTERNALS}.mergePoints({json.dumps(old)}, {json.dumps(fresh)}, 2000)") == [[2000, 2], [3000, 30], [5000, 50]]
    assert run_js(RUN_PAGE, f"{INTERNALS}.mergePoints({json.dumps(old)}, [], null)") == old
    assert run_js(RUN_PAGE, f"{INTERNALS}.mergePoints(null, {json.dumps(fresh)}, null)") == fresh


def test_freshness_line():
    at = int(datetime(2026, 10, 16, 12, 4, 31).timestamp() * 1000)
    running = run_js(RUN_PAGE, f"{INTERNALS}.freshness({at}, {at + 9000}, true)", tz="UTC")
    assert running == {"text": f"данные на {datetime.fromtimestamp(at / 1000, timezone.utc):%H:%M:%S}, задержка 9 с", "stale": False}
    assert run_js(RUN_PAGE, f"{INTERNALS}.freshness({at}, {at + 61000}, true)", tz="UTC")["stale"] is True
    assert run_js(RUN_PAGE, f"{INTERNALS}.freshness({at}, {at + 61000}, false)", tz="UTC")["text"].startswith("тест завершён")
    assert run_js(RUN_PAGE, f"{INTERNALS}.freshness(null, 0, true)") == {"text": "данных пока нет", "stale": False}


def test_sla_lines_per_chart():
    lines = run_js(RUN_PAGE, f"{INTERNALS}.slaLines({{target_rps: 250, max_error_rate_pct: 1, max_p95_ms: 800}})")
    assert [(line["axis"], line["value"]) for line in lines["rps"]] == [("y", 250), ("y1", 1)]
    assert lines["p95"][0]["label"] == "порог p95 800 мс"
    assert run_js(RUN_PAGE, f"{INTERNALS}.slaLines(null)") == {"rps": [], "p95": []}


def test_far_thresholds_do_not_flatten_the_curve():
    near = run_js(RUN_PAGE, f"{INTERNALS}.axisRange({{beginAtZero: true}}, [{{value: 14}}], 9)")
    assert near["suggestedMax"] == pytest.approx(16.1)
    assert "suggestedMax" not in run_js(RUN_PAGE, f"{INTERNALS}.axisRange({{beginAtZero: true}}, [{{value: 14}}], 0.9)")
    assert "suggestedMax" not in run_js(RUN_PAGE, f"{INTERNALS}.axisRange({{beginAtZero: true}}, [{{value: 14}}], 0)")
    kept = run_js(RUN_PAGE, f"{INTERNALS}.axisRange({{suggestedMax: 5}}, [{{value: 1}}], 5)")
    assert kept["suggestedMax"] == 5


def test_empty_results_are_explained(monkeypatch):
    from loadlens_app import query_preview

    monkeypatch.setattr(live, "_scope_config", lambda run: _cfg("lt-k6-influxql"))
    monkeypatch.setattr(query_preview, "fetch_points", lambda cfg_, **kwargs: {})
    run = _run([_inst(0, launched_at=T0, started_at=T0)])
    snapshot = live.live_snapshot(run, effective_tools({}), now=T0.timestamp() + 120)
    assert snapshot["ok"] and snapshot["series"]["lt.rps_total"]["points"] == []
    assert snapshot["message"].startswith("Запрос «LT (InfluxQL): RPS sum by all groups» не вернул точек за ")
    assert "ни с тегом прогона nightly, ни без него" in snapshot["message"]
    assert snapshot["rps_query"]["label"] == "LT (InfluxQL): RPS sum by all groups" and '"testid" =~ /^.*$/' in snapshot["rps_query"]["query"]
    # Too early to worry, and reads of new points stay quiet.
    assert live.live_snapshot(run, effective_tools({}), now=T0.timestamp() + 20)["message"] == ""
    assert live.live_snapshot(run, effective_tools({}), now=T0.timestamp() + 120, since=T0.timestamp() + 100, mode="filtered")["message"] == ""


def test_dashboard_brief_reads_two_rows_and_caches_them(monkeypatch, source):
    monkeypatch.setattr(live, "_scope_config", lambda run: _cfg())
    live.clear_brief_cache()
    run = _run([_inst(0, launched_at=T0, started_at=T0)])
    now = T0.timestamp() + 3600
    try:
        brief = live.live_brief(run, effective_tools({}), now=now)
        assert brief["ok"] and brief["rps"] == 5.0 and brief["errors"] == 5.0 and len(brief["points"]) > 2
        assert brief["start"] == now - live.BRIEF_WINDOW_MIN * 60 and brief["end"] == now
        reads = len(source)
        # RPS and errors only: no latency, no VUs, no RPS of each instance.
        assert reads == 2 and not any("lt_instance" in (call["keys"] or []) for call in source)
        assert live.live_brief(run, effective_tools({}), now=now + 5) is brief and len(source) == reads
        live.clear_brief_cache()
        live.live_brief(run, effective_tools({}), now=now + 5)
        assert len(source) == 2 * reads
    finally:
        live.clear_brief_cache()
