# SPDX-License-Identifier: Apache-2.0
"""Forms of ui-plan stage 4: the test window hint (GET /lt_windows) and its detection in live.py.

Offline: the metric source is a stub of ``query_preview.fetch_points`` and runs live in a memory store.
"""

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from loadlens_app import core, query_preview
from loadlens_app.auth.models import Role
from loadlens_app.auth.policy import rule_for
from loadlens_app.load_tests import live, store
from loadlens_app.load_tests.store import InstanceRecord, MemoryLoadTestStore, RunRecord

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp()


def _lt_config():
    from AI.query_packs import load_catalog, merge_pack, render_pack

    block, _ = merge_pack(None, render_pack(load_catalog().pack("lt-k6-prometheus-rw")))
    return {"queries": {"lt_framework": block}}


def _series(start, step, values):
    return [(start + i * step, float(v)) for i, v in enumerate(values)]


@pytest.fixture(autouse=True)
def _fresh_cache():
    live.clear_windows_cache()
    yield
    live.clear_windows_cache()


@pytest.fixture
def runs(monkeypatch):
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    return memory


@pytest.fixture
def source(monkeypatch, runs):
    """Stubbed config and metric source; ``source.points`` is what every RPS query returns."""
    state = types.SimpleNamespace(points=[], calls=[], error=None, cfg=_lt_config())
    monkeypatch.setattr(live, "_service_config", lambda area, service: state.cfg)

    def fake_fetch(cfg, *, domain, lang, query, start_ts, end_ts, step, label_keys=None):
        state.calls.append({"domain": domain, "lang": lang, "query": query, "start": start_ts, "end": end_ts, "step": step})
        if state.error:
            raise state.error
        return {"series": list(state.points)}

    monkeypatch.setattr(query_preview, "fetch_points", fake_fetch)
    return state


# ---- segment detection -------------------------------------------------------------------------------


def test_segments_split_on_a_pause_longer_than_three_steps():
    points = _series(0, 60, [0, 5, 7, 0, 0, 6, 0, 0, 0, 0, 9, 11, 0])
    segments = live.activity_segments(points, 60)
    # Two zero steps (a gap of 3 steps) keep one stretch; four zero steps split it.
    assert [(s["start"], s["end"], s["peak"]) for s in segments] == [(60, 300, 7.0), (600, 660, 11.0)]


def test_segments_treat_missing_points_as_a_pause_and_drop_single_points():
    points = [(0, 1.0), (60, 2.0), (500, 3.0), (2000, 4.0), (2060, 4.0), (2120, 5.0)]
    segments = live.activity_segments(points, 60)
    assert [(s["start"], s["end"]) for s in segments] == [(0, 60), (2000, 2120)]
    assert live.activity_segments([], 60) == []
    assert live.activity_segments([(0, float("nan")), (60, -1.0)], 60) == []


def test_hours_are_clamped():
    assert live.clamp_hours(48) == 24
    assert live.clamp_hours(0) == live.WINDOWS_MIN_HOURS
    assert live.clamp_hours("x") == live.WINDOWS_DEFAULT_HOURS
    assert live.clamp_hours(float("inf")) == live.WINDOWS_DEFAULT_HOURS
    assert live.clamp_hours(6) == 6


# ---- load_windows ---------------------------------------------------------------------------------------


def test_load_windows_returns_the_latest_three_stretches_newest_first(source):
    step = live.choose_step(NOW - 6 * 3600, NOW)
    start = NOW - 6 * 3600
    values = []
    for _ in range(4):  # four tests of 10 steps with 10 idle steps in between
        values += [0] * 10 + [100, 200, 250, 240, 230, 200, 150, 120, 110, 100]
    source.points = _series(start, step, values)
    data = live.load_windows("shop", "checkout", 6, now=NOW)
    assert data["ok"] and data["step_sec"] == step == 120 and data["hours"] == 6
    assert len(data["windows"]) == 3
    starts = [w["start"] for w in data["windows"]]
    assert starts == sorted(starts, reverse=True)
    newest = data["windows"][0]
    assert newest["rps_peak"] == 250.0 and newest["duration_sec"] == 9 * step and newest["ongoing"] is False
    assert datetime.fromisoformat(newest["start"]).timestamp() == start + 70 * step
    call = source.calls[0]
    assert call["domain"] == "lt_framework" and call["step"] == f"{step}s"
    assert call["start"] == pytest.approx(NOW - 6 * 3600) and call["end"] == NOW
    assert data["query"]


def test_load_windows_marks_a_stretch_that_reaches_now_as_ongoing(source):
    step = live.choose_step(NOW - 3600, NOW)
    source.points = _series(NOW - 10 * step, step, [5] * 11)
    data = live.load_windows("shop", "checkout", 1, now=NOW)
    assert data["windows"][0]["ongoing"] is True


def test_load_windows_clamps_hours_and_keeps_points_within_limits(source):
    data = live.load_windows("shop", "checkout", 72, now=NOW)
    assert data["hours"] == 24
    call = source.calls[0]
    assert call["start"] == pytest.approx(NOW - 24 * 3600)
    step = int(call["step"].rstrip("s"))
    assert (call["end"] - call["start"]) / step <= 600


def test_load_windows_without_lt_queries_still_lists_runs(source, runs):
    source.cfg = {"queries": {}}
    _add_run(runs, "release-1", NOW - 3 * 3600, NOW - 2 * 3600)
    data = live.load_windows("shop", "checkout", 6, now=NOW)
    assert not data["ok"] and "пакет запросов" in data["message"]
    assert data["windows"] == [] and [r["run_name"] for r in data["runs"]] == ["release-1"]
    assert source.calls == []


def test_load_windows_reports_source_errors_quietly(source):
    source.error = RuntimeError("connection refused")
    data = live.load_windows("shop", "checkout", 6, now=NOW)
    assert not data["ok"] and data["message"] == "Источник метрик недоступен" and data["detail"] == "connection refused"
    assert data["windows"] == []


def test_load_windows_survives_unreadable_settings(source, monkeypatch):
    def broken(area, service):
        raise ValueError("bad runtime file")

    monkeypatch.setattr(live, "_service_config", broken)
    data = live.load_windows("shop", "checkout", 6, now=NOW)
    assert not data["ok"] and data["message"] == "Настройки сервиса не читаются" and data["detail"] == "bad runtime file"


# ---- finished runs ------------------------------------------------------------------------------------


def _ts(value):
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _add_run(memory, name, start, end, service="checkout", status=store.RUN_DONE):
    run = RunRecord(
        run_id=f"id-{name}", run_name=name, run_tag=name, service=service, status=status,
        instances=[InstanceRecord(index=0, name="i0", tag="i0", generator="gen-1", tool="k6",
                                  started_at=_ts(start) if start else None, finished_at=_ts(end) if end else None)],
        started_at=_ts(start) if start else None, finished_at=_ts(end) if end else None,
        created_at=_ts((start or NOW) - 60),
    )
    memory.create(run)
    return run


def test_finished_runs_of_the_service_within_the_window(runs):
    _add_run(runs, "recent", NOW - 2 * 3600, NOW - 3600)
    _add_run(runs, "failed", NOW - 5 * 3600, NOW - 4 * 3600, status=store.RUN_FAILED)
    _add_run(runs, "too-old", NOW - 30 * 3600, NOW - 29 * 3600)
    _add_run(runs, "other", NOW - 2 * 3600, NOW - 3600, service="payments")
    _add_run(runs, "running", NOW - 600, None, status=store.RUN_RUNNING)
    _add_run(runs, "never-started", None, None, status=store.RUN_CANCELLED)
    found = live.finished_runs("checkout", NOW - 6 * 3600, NOW)
    assert [r["run_name"] for r in found] == ["recent", "failed"]
    assert found[0]["duration_sec"] == 3600 and found[0]["status"] == store.RUN_DONE
    assert datetime.fromisoformat(found[0]["start"]).timestamp() == NOW - 2 * 3600


# ---- HTTP endpoint --------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path, source):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    from loadlens_app.blueprints import load_tests as blueprint

    monkeypatch.setattr(blueprint, "_find_area_for_service", lambda service_id: "shop" if service_id == "checkout" else None)
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_lt_windows_endpoint(client, source):
    now = datetime.now(timezone.utc).timestamp()
    step = live.choose_step(now - 6 * 3600, now)
    source.points = _series(now - 3 * 3600, step, [0, 50, 80, 75, 0])
    resp = client.get("/lt_windows?service=checkout&hours=6")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] and data["service"] == "checkout" and data["hours"] == 6
    assert len(data["windows"]) == 1 and data["windows"][0]["rps_peak"] == 80.0
    assert set(data["windows"][0]) == {"start", "end", "duration_sec", "rps_peak", "ongoing"}
    assert data["runs"] == []


def test_lt_windows_endpoint_caches_for_a_minute(client, source):
    client.get("/lt_windows?service=checkout")
    client.get("/lt_windows?service=checkout&hours=6")
    assert len(source.calls) == 1
    client.get("/lt_windows?service=checkout&hours=3")
    assert len(source.calls) == 2
    for key, (at, value) in list(live._windows_cache.items()):  # a minute passes
        live._windows_cache[key] = (at - live.WINDOWS_CACHE_SEC - 1, value)
    client.get("/lt_windows?service=checkout")
    assert len(source.calls) == 3


def test_lt_windows_endpoint_validation(client):
    assert client.get("/lt_windows").status_code == 400
    assert client.get("/lt_windows?service=checkout&hours=abc").status_code == 400
    missing = client.get("/lt_windows?service=nope")
    assert missing.status_code == 404 and "nope" in missing.get_json()["error"]
    assert client.get("/lt_windows?service=checkout&hours=100").get_json()["hours"] == 24


def test_lt_windows_source_error_is_not_an_http_error(client, source):
    source.error = RuntimeError("timeout")
    resp = client.get("/lt_windows?service=checkout")
    assert resp.status_code == 200 and resp.get_json()["ok"] is False


def test_lt_windows_policy_is_engineer_read():
    rule = rule_for("load_tests.lt_windows")
    assert rule.role == Role.ENGINEER and not rule.page and not rule.session_only


def test_report_and_launch_pages_load_the_service_picker(client):
    for url in ("/new", "/launch"):
        html = client.get(url).get_data(as_text=True)
        assert 'id="serviceCombo"' in html and 'type="hidden" id="service"' in html
        assert html.index("/static/js/service_picker.js") < html.index("/static/js/new.js" if url == "/new" else "/static/js/launch.js")
    new_page = client.get("/new").get_data(as_text=True)
    assert 'id="ltWindowsChips"' in new_page and 'id="progressElapsed"' in new_page and 'id="progressTypical"' in new_page
