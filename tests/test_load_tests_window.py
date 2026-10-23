# SPDX-License-Identifier: Apache-2.0
"""The report window of a run follows its load: virtual users and the summed RPS, inside the process time."""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from AI.i18n import render_ref  # noqa: E402
from loadlens_app.load_tests import live, store, watcher  # noqa: E402
from loadlens_app.load_tests.executors import Probe  # noqa: E402
from loadlens_app.load_tests.store import RunRecord  # noqa: E402
from tests.test_load_tests import _inst, _ref, _run, world  # noqa: E402,F401  (fixture)

START, END = 1000.0, 2000.0  # 1000 s: the read step is 5 s
STEP = live.choose_step(START, END)
RUN = RunRecord(run_id="r", run_name="n", run_tag="n", service="s")


def _series(first, last, value=3.0, every=10):
    return [(float(ts), value) for ts in range(first, last + 1, every)]


@pytest.fixture
def source(monkeypatch):
    """A metrics source with a users row and an RPS row; ``points`` per role, ``fail`` roles whose query raises."""
    data = {
        "rows": {role: live.LiveRow(role, f"query of {role}", [], "promql") for role in live.BOUNDS_ROLES},
        "points": {},
        "fail": set(),
        "calls": [],
        "filtered": [],
    }
    monkeypatch.setattr(live, "_scope_config", lambda run: {"queries": {"plain": True}})

    def filtered(cfg, run, packs):
        data["filtered"].append(list(packs))
        return {"filtered": True}

    def block_rows(queries, roles, cfg=None):
        data["queries"] = queries
        return "promql", {role: data["rows"][role] for role in roles if role in data["rows"]}

    def fetch(cfg, lang, row, start, end, step, summed):
        data["calls"].append((row.label, summed, step, start, end))
        if row.label in data["fail"]:
            raise RuntimeError("source down")
        return data["points"].get(row.label, [])

    monkeypatch.setattr(live, "_filtered", filtered)
    monkeypatch.setattr(live, "_block_rows", block_rows)
    monkeypatch.setattr(live, "_fetch", fetch)
    return data


def test_users_keep_the_tail_where_hanging_requests_drop_rps_to_zero(source):
    source["points"] = {
        "lt.rps_total": _series(1100, 1500) + [(float(ts), 0.0) for ts in range(1510, 1900, 10)],
        "lt.vus": _series(1050, 1800, value=50.0),
    }
    window = live.load_bounds(RUN, START, END, [])
    assert window == live.LoadWindow(1050 - STEP, 1800 + STEP, ("lt.vus", "lt.rps_total"))
    # RPS is the sum over the tools, users are not summed; both read the process window with one step.
    assert sorted(call[:2] for call in source["calls"]) == [("lt.rps_total", True), ("lt.vus", False)]
    assert {call[2:] for call in source["calls"]} == {(STEP, START, END)}
    assert source["queries"] == {"plain": True} and source["filtered"] == []


def test_rps_widens_the_window_where_users_are_missing(source):
    source["points"] = {"lt.rps_total": _series(1020, 1900), "lt.vus": _series(1100, 1500)}
    assert live.load_bounds(RUN, START, END, []) == (1020 - STEP, 1900 + STEP, ("lt.vus", "lt.rps_total"))


@pytest.mark.parametrize("rows, points, by", [
    (("lt.vus",), {"lt.vus": _series(1100, 1500)}, ("lt.vus",)),
    (("lt.rps_total",), {"lt.rps_total": _series(1100, 1500)}, ("lt.rps_total",)),
    # A users row without load names only the signal that had it.
    (live.BOUNDS_ROLES, {"lt.vus": [(1100.0, 0.0), (1200.0, 0.0)], "lt.rps_total": _series(1100, 1500)}, ("lt.rps_total",)),
])
def test_the_window_names_the_signals_with_load(source, rows, points, by):
    source["rows"] = {role: row for role, row in source["rows"].items() if role in rows}
    source["points"] = points
    assert live.load_bounds(RUN, START, END, []) == (1100 - STEP, 1500 + STEP, by)


def test_a_failing_query_leaves_its_signal_out(source):
    source["points"] = {"lt.vus": _series(1100, 1500), "lt.rps_total": _series(1050, 1900)}
    source["fail"] = {"lt.rps_total"}
    assert live.load_bounds(RUN, START, END, []) == (1100 - STEP, 1500 + STEP, ("lt.vus",))
    source["fail"] = set(live.BOUNDS_ROLES)
    assert live.load_bounds(RUN, START, END, []) is None


@pytest.mark.parametrize("points", [
    {},
    {"lt.vus": [(1100.0, 5.0)]},
    # One moment seen by both signals is still one moment: noise, not a test.
    {"lt.vus": [(1100.0, 5.0)], "lt.rps_total": [(1100.0, 2.0)]},
    {"lt.vus": [(1100.0, float("nan")), (1200.0, float("inf"))], "lt.rps_total": [(1100.0, None), (1300.0, -1.0)]},
])
def test_fewer_than_two_moments_with_load_say_nothing(source, points):
    source["points"] = points
    assert live.load_bounds(RUN, START, END, []) is None


def test_the_window_stays_inside_the_process_time(source):
    source["points"] = {"lt.vus": [(START, 10.0), (END, 10.0)]}
    assert live.load_bounds(RUN, START, END, []) == (START, END, ("lt.vus",))


def test_no_rows_or_no_config_read_nothing(source, monkeypatch):
    source["rows"] = {}
    assert live.load_bounds(RUN, START, END, []) is None and source["calls"] == []

    def broken(run):
        raise RuntimeError("no config")

    monkeypatch.setattr(live, "_scope_config", broken)
    assert live.load_bounds(RUN, START, END, []) is None


def test_the_report_filter_applies_to_both_signals(source):
    source["points"] = {"lt.vus": _series(1100, 1500)}
    assert live.load_bounds(RUN, START, END, ["lt-k6-influxql"]) is not None
    assert source["filtered"] == [["lt-k6-influxql"]] and source["queries"] == {"filtered": True}


# ---- the watcher ---------------------------------------------------------------------------------


def _ssh_run_to_report(world, monkeypatch, bounds):
    """An SSH k6 instance that runs 600 s; ``bounds(start, end)`` stands for the metrics; the report starts."""
    seen = {}

    def load_bounds(run, start, end, packs):
        seen.update(window=(start, end), packs=list(packs))
        return bounds(start, end)

    monkeypatch.setattr(watcher, "load_bounds", load_bounds)
    run_id = _run(world, [_inst(0)])
    world.watcher.tick()
    t0 = world.clock()
    world.fake.state[_ref(run_id, 0)] = Probe(state="exited", started_at=t0, finished_at=t0 + 600, exit_code=0)
    world.clock.advance(600)
    world.watcher.tick()
    world.clock.advance(61)
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.status == store.RUN_REPORTING
    return t0, run, world.started[0], seen


def test_an_ssh_run_narrows_its_report_window_by_the_load(world, monkeypatch):
    t0, run, report, seen = _ssh_run_to_report(
        world, monkeypatch, lambda start, end: live.LoadWindow(start + 30, end - 45, ("lt.vus", "lt.rps_total")))
    assert seen["window"] == (t0, t0 + 600) and "lt-k6-influxql" in seen["packs"]
    assert (report["start_ms"], report["end_ms"]) == (int((t0 + 30) * 1000), int((t0 + 555) * 1000))
    assert "окно отчёта сужено по пользователям и RPS:" in run.message
    english = render_ref(run.message_ref, locale="en")
    assert english.startswith("Building the report (the report window was narrowed by users and RPS: ")
    # The run keeps the process time; only the report follows the load.
    assert (run.started_at.timestamp(), run.finished_at.timestamp()) == (t0, t0 + 600)


def test_an_ssh_run_without_load_data_reports_the_process_time_without_a_note(world, monkeypatch):
    t0, run, report, _ = _ssh_run_to_report(world, monkeypatch, lambda start, end: None)
    assert (report["start_ms"], report["end_ms"]) == (int(t0 * 1000), int((t0 + 600) * 1000))
    assert run.message == "Строится отчёт"


def test_a_window_that_the_load_does_not_change_has_no_note(world, monkeypatch):
    _, run, _, _ = _ssh_run_to_report(world, monkeypatch, lambda start, end: live.LoadWindow(start, end, ("lt.rps_total",)))
    assert run.message == "Строится отчёт"


def test_a_manual_run_is_narrowed_too(world, monkeypatch):
    monkeypatch.setattr(watcher, "load_bounds", lambda run, start, end, packs: live.LoadWindow(start + 60, end - 60, ("lt.vus",)))
    run_id = _run(world, [_inst(0, generator="manual", tool="")], use_llm=False)
    world.watcher.tick()
    world.clock.advance(900)
    world.store.request_stop(run_id)
    world.watcher.tick()
    world.clock.advance(61)
    world.watcher.tick()
    report = world.started[0]
    assert report["end_ms"] - report["start_ms"] == 780_000
    assert "окно отчёта сужено по пользователям:" in world.store.get(run_id).message
