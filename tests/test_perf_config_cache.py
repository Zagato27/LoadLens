# SPDX-License-Identifier: Apache-2.0
"""Runtime config cache in loadlens_app.core: one stat per call while the files are unchanged.

Every test works on temporary runtime files (core.CONFIG_RUNTIME_PATH, core.METRICS_RUNTIME_PATH and
update_page._SETTINGS_RUNTIME_PATH are patched) and stays offline.
"""

import copy
import json
import os
import pathlib
import sys
import threading
import time
from pathlib import Path

import psycopg2
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before update_page)
from loadlens_app import projects  # noqa: E402
import update_page  # noqa: E402

HOUR_NS = 3600 * 1_000_000_000

SETTINGS = {
    "per_area": {
        "alpha": {"title": "Alpha", "services": {"svc-a": {"title": "Service A", "disabled_domains": ["kafka"]}}},
        "beta": {"title": "Beta", "services": {}},
    },
}
METRICS = {
    "alpha": {"services": {"svc-a": {"page_sample_id": "1", "page_parent_id": "2", "metrics": [], "logs": []}}},
    "gamma": {"services": {"svc-g": {"page_sample_id": "3", "page_parent_id": "4", "metrics": [], "logs": []}}},
}


def _write(path: Path, payload, *, age_ns: int = HOUR_NS) -> None:
    """Writes JSON in place (like the tests and editors do); a positive age moves the mtime into the past."""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if age_ns:
        old = time.time_ns() - age_ns
        os.utime(path, ns=(old, old))


@pytest.fixture
def files(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings_runtime.json"
    metrics_path = tmp_path / "metrics_config_runtime.json"
    _write(settings_path, SETTINGS)
    _write(metrics_path, METRICS)
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", settings_path)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", metrics_path)
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(settings_path))
    monkeypatch.setitem(core.CONFIG, "per_area", {})
    return settings_path, metrics_path


@pytest.fixture
def reads(monkeypatch, files):
    """Counts file reads of the two runtime files, whatever API reads them."""
    watched = {os.fspath(p) for p in files}
    counts = {"n": 0}
    original_bytes = pathlib.Path.read_bytes
    original_text = pathlib.Path.read_text

    def read_bytes(self, *args, **kwargs):
        if os.fspath(self) in watched:
            counts["n"] += 1
        return original_bytes(self, *args, **kwargs)

    def read_text(self, *args, **kwargs):
        if os.fspath(self) in watched:
            counts["n"] += 1
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    return counts


@pytest.fixture
def client(monkeypatch, files):
    def offline():
        raise psycopg2.OperationalError("config cache tests stay offline")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", offline)

    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ---- file cache -------------------------------------------------------------------------------------


def test_second_call_does_not_read_the_file(reads):
    first = core._per_area_config()
    assert reads["n"] == 1
    assert core._per_area_config() is first
    assert core._services_map_for_area("alpha") is first["alpha"]["services"]
    assert core._resolve_services_filter("alpha") == ["svc-a"]
    assert core._list_project_areas() == [{"id": "alpha", "title": "Alpha"}, {"id": "beta", "title": "Beta"}]
    assert core._service_disabled_domains("alpha", "svc-a") == ["kafka"]
    assert reads["n"] == 1


def test_changed_size_is_read_again(files, reads):
    settings_path, _ = files
    assert core._area_entry("beta")["title"] == "Beta"
    changed = copy.deepcopy(SETTINGS)
    changed["per_area"]["beta"]["title"] = "Beta, renamed"
    _write(settings_path, changed)
    assert core._area_entry("beta")["title"] == "Beta, renamed"
    assert reads["n"] == 2


def test_touched_file_is_read_again_but_keeps_the_parsed_object(files, reads):
    settings_path, _ = files
    first = core._per_area_config()
    newer = time.time_ns() - HOUR_NS // 2
    os.utime(settings_path, ns=(newer, newer))
    assert core._per_area_config() is first  # same bytes: no new parse, derived caches stay valid
    assert reads["n"] == 2
    assert core._per_area_config() is first
    assert reads["n"] == 2


@pytest.mark.parametrize("tick_ns", [0, 1_000_000_000])
def test_same_size_rewrite_within_one_mtime_tick_is_noticed(files, tick_ns):
    """Two writes in one mtime tick that keep the size (exact clock, or a 1 s tick as on coarse file
    systems): the stat signature is identical, the recent mtime makes the cache compare the bytes."""
    settings_path, _ = files
    stamp = (time.time_ns() // tick_ns * tick_ns) if tick_ns else 0
    _write(settings_path, {"per_area": {"one": {"title": "AAAA"}}}, age_ns=0)
    stamp = stamp or settings_path.stat().st_mtime_ns
    os.utime(settings_path, ns=(stamp, stamp))
    assert core._area_entry("one")["title"] == "AAAA"
    _write(settings_path, {"per_area": {"one": {"title": "BBBB"}}}, age_ns=0)
    os.utime(settings_path, ns=(stamp, stamp))
    assert core._area_entry("one")["title"] == "BBBB"


def test_save_is_seen_without_any_change_of_the_signature(monkeypatch, files, reads):
    settings_path, _ = files
    frozen = (time.time_ns() - HOUR_NS, 1, 1, 1, 1)
    monkeypatch.setattr(core, "_file_signature", lambda st: frozen)
    assert core._area_entry("alpha")["title"] == "Alpha"
    _write(settings_path, {"per_area": {"alpha": {"title": "edited behind the cache"}}})
    assert core._area_entry("alpha")["title"] == "Alpha"  # the signature is all the cache looks at

    runtime = core._load_settings_runtime_data()
    runtime["per_area"]["alpha"]["title"] = "Saved"
    core._save_settings_runtime_data(runtime)
    count = reads["n"]
    assert core._area_entry("alpha")["title"] == "Saved"
    assert reads["n"] == count  # refreshed by the write itself


def test_writers_get_a_private_copy(files):
    cached = core._per_area_config()
    runtime = core._load_settings_runtime_data()
    assert runtime["per_area"] is not cached
    runtime["per_area"]["alpha"]["title"] = "changed in a copy"
    assert core._per_area_config()["alpha"]["title"] == "Alpha"


def test_broken_json_raises_and_is_not_cached(files, reads):
    settings_path, metrics_path = files
    settings_path.write_text('{"per_area": {"alpha": ', encoding="utf-8")
    with pytest.raises(core.RuntimeConfigError):
        core._per_area_config()
    with pytest.raises(core.RuntimeConfigError):
        core._per_area_config()
    assert reads["n"] == 2  # read again, not served from the cache
    _write(settings_path, SETTINGS)
    assert core._resolve_services_filter("alpha") == ["svc-a"]

    metrics_path.write_text("{broken", encoding="utf-8")
    assert "gamma" not in core._active_metrics_config()  # an unreadable metrics file counts as missing
    _write(metrics_path, METRICS)
    assert core._metrics_services_for_area("gamma") == {"svc-g": METRICS["gamma"]["services"]["svc-g"]}


def test_missing_and_empty_files(files):
    settings_path, metrics_path = files
    settings_path.unlink()
    metrics_path.unlink()
    assert core._per_area_config() == {}
    empty = core._cached_json(settings_path)
    empty["changed"] = True
    assert core._cached_json(settings_path) == {}
    assert core._resolve_services_filter("alpha") == ["alpha"]
    settings_path.write_text("", encoding="utf-8")
    assert core._per_area_config() == {}
    _write(settings_path, SETTINGS, age_ns=0)
    assert core._resolve_services_filter("alpha") == ["svc-a"]


def test_parallel_readers_and_a_writer(files):
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader():
        try:
            for _ in range(50):
                title = core._per_area_config()["alpha"]["title"]
                assert title in {"Alpha", "Alpha 1", "Alpha 2"}
                assert core._resolve_services_filter("alpha") == ["svc-a"]
                assert "gamma" in core._active_metrics_config()
                assert core._find_area_for_service("svc-g") == "gamma"
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            errors.append(exc)

    def writer():
        n = 0
        while not stop.is_set():
            runtime = core._load_settings_runtime_data()
            runtime["per_area"]["alpha"]["title"] = f"Alpha {n % 2 + 1}"
            core._save_settings_runtime_data(runtime)
            n += 1
            stop.wait(0.02)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    background = threading.Thread(target=writer)
    background.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    stop.set()
    background.join(timeout=60)
    assert not errors, errors[:3]
    assert core._per_area_config()["alpha"]["title"] == core._load_settings_runtime_data()["per_area"]["alpha"]["title"]


# ---- _active_metrics_config ---------------------------------------------------------------------------


def test_active_metrics_config_is_computed_once_while_files_are_unchanged(monkeypatch, files):
    calls = {"n": 0}
    original = core._merged_metrics_config

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_merged_metrics_config", counting)
    first = core._active_metrics_config()
    assert core._active_metrics_config() is first
    assert core._metrics_services_for_area("gamma") is first["gamma"]["services"]
    assert core._metrics_service_entry("svc-g") == ("gamma", METRICS["gamma"]["services"]["svc-g"])
    assert calls["n"] == 1

    current = core._load_metrics_runtime_data()
    current["gamma"]["services"]["svc-h"] = {"metrics": []}
    core._save_metrics_runtime_data(current)
    second = core._active_metrics_config()
    assert second is not first and "svc-h" in second["gamma"]["services"]
    assert calls["n"] == 2


def test_active_metrics_config_follows_per_area_changes(files):
    """The grouping by area reads settings_runtime.json, so a change there recomputes it."""
    first = core._active_metrics_config()
    assert "svc-g" in first["gamma"]["services"]
    runtime = core._load_settings_runtime_data()
    runtime["per_area"]["delta"] = {"title": "Delta", "services": {"svc-g": {}}}
    core._save_settings_runtime_data(runtime)
    second = core._active_metrics_config()
    assert second is not first
    assert "delta" in second
    assert core._find_area_for_service("svc-g") == "delta"


def test_active_metrics_config_does_not_share_objects_with_the_file_cache(files):
    metrics = core._active_metrics_config()
    raw = core._metrics_runtime_view()
    assert metrics["alpha"]["services"]["svc-a"] == raw["alpha"]["services"]["svc-a"]
    assert metrics["alpha"]["services"]["svc-a"] is not raw["alpha"]["services"]["svc-a"]


# ---- callers --------------------------------------------------------------------------------------------


def test_bootstrap_of_a_registered_service_reads_and_writes_nothing(monkeypatch, files):
    core._per_area_config()
    core._metrics_runtime_view()

    def forbidden(*args, **kwargs):
        raise AssertionError("a registered service must not reload or rewrite the runtime files")

    monkeypatch.setattr(core, "_read_json_file", forbidden)
    monkeypatch.setattr(core, "_write_json_atomic", forbidden)
    core._bootstrap_service_configs("alpha", "svc-a")


def test_bootstrap_of_a_new_service_still_registers_it(files):
    core._bootstrap_service_configs("beta", "svc-b")
    assert core._services_map_for_area("beta") == {"svc-b": {}}
    assert "svc-b" in core._metrics_runtime_view()["beta"]["services"]
    assert core._find_area_for_service("svc-b") == "beta"


def test_requests_do_not_change_the_cached_configs(client):
    """Audit guard: the read endpoints use the shared caches but never write into them."""
    settings_before = copy.deepcopy(core._settings_runtime_view())
    metrics_before = copy.deepcopy(core._active_metrics_config())
    for url in (
        "/services?area=alpha",
        "/services?area=gamma",
        "/services",
        "/project_areas",
        "/config?area=alpha",
        "/config?area=gamma",
        "/prompts?area=alpha&service=svc-a",
        "/projects",
    ):
        resp = client.get(url)
        assert resp.status_code == 200, (url, resp.get_json())
    assert core._settings_runtime_view() == settings_before
    assert core._active_metrics_config() == metrics_before


def test_direct_file_edits_show_up_in_the_next_request(client, files):
    settings_path, _ = files
    assert [a["id"] for a in client.get("/project_areas").get_json()] == ["alpha", "beta"]
    changed = copy.deepcopy(SETTINGS)
    changed["per_area"]["omega"] = {"title": "Omega", "services": {"svc-o": {}}}
    _write(settings_path, changed, age_ns=0)
    assert [a["id"] for a in client.get("/project_areas").get_json()] == ["alpha", "beta", "omega"]
    services = client.get("/services?area=omega").get_json()["services"]
    assert [s["id"] for s in services] == ["svc-o"]


def test_project_move_copies_the_metrics_fallback(monkeypatch, files):
    """A service known only from metrics_config moves with a copy, not with the cached object."""
    _settings_path, metrics_path = files
    monkeypatch.setitem(core.METRICS_CONFIG, "legacy", {"services": {"svc-l": {"metrics": [{"name": "x"}]}}})
    core._clear_runtime_cache()
    view = core._active_metrics_config()
    snapshot = copy.deepcopy(view)
    projects._move_service_metrics("legacy", "svc-l", "beta")
    moved = json.loads(metrics_path.read_text(encoding="utf-8"))["beta"]["services"]["svc-l"]
    assert moved == {"metrics": [{"name": "x"}]}
    assert view == snapshot  # the object that was cached before the move is untouched
    assert core._metrics_services_for_area("beta")["svc-l"] is not view["legacy"]["services"]["svc-l"]
