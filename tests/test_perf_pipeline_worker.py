"""Report pipeline safe inside a Celery worker: no nested sub-tasks, no web-process state.

Downloads of charts and logs and attachment uploads of the template report run on a local pool of
daemon threads (update_page._TransferPool) instead of Celery tasks the report waited for. The tests
use fake download and upload functions: no Grafana, Loki or Confluence is contacted.
"""

import ast
import inspect
import json
import logging
import os
import sys
import threading
import time
from functools import partial
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before update_page: the package import cycle)
import update_page  # noqa: E402
from data_collectors import temp_files  # noqa: E402

Pool = update_page._TransferPool

LEGACY = {
    "grafana_base_url": "http://grafana.test:3000",
    "grafana_login": "g-user",
    "grafana_pass": "g-pass",
    "url_basic": "https://wiki.test",
    "user": "wiki-user",
    "password": "wiki-pass",
    "loki_url": "http://loki.test/loki/api/v1/query_range",
}
SERVICE_CONFIG = {
    "page_parent_id": "100",
    "page_sample_id": "200",
    "metrics": [
        {"name": "rps", "grafana_url": "/render/d-solo/a?panelId=1"},
        {"name": "cpu", "grafana_url": "/render/d-solo/a?panelId=2"},
        {"name": "mem", "grafana_url": "/render/d-solo/a?panelId=3"},
    ],
    "logs": [
        {"placeholder": "errors", "filter_query": '{app="shop"} |= "ERROR"'},
        {"placeholder": "warns", "filter_query": '{app="shop"} |= "WARN"'},
    ],
}
START_MS, END_MS = 1_700_000_000_000, 1_700_000_600_000
PAGE_ID = "777"


def _image_html(basename: str) -> str:
    return f'<ac:image><ri:attachment ri:filename="{basename}.jpg" /></ac:image>'


def _log_html(basename: str) -> str:
    return (
        '<ac:structured-macro ac:name="view-file" ac:schema-version="1">'
        '<ac:parameter ac:name="name">'
        f'<ri:attachment ri:filename="{basename}.log" />'
        "</ac:parameter>"
        '<ac:parameter ac:name="height">250</ac:parameter>'
        "</ac:structured-macro>"
    )


# --- the pool ---------------------------------------------------------------------------------


def test_pool_defaults_match_the_former_celery_tasks():
    params = inspect.signature(Pool).parameters
    assert update_page._TRANSFER_ATTEMPTS == 3 and params["attempts"].default == 3
    assert update_page._TRANSFER_RETRY_PAUSE_SEC == 2.0 and params["pause"].default == 2.0
    assert update_page._TASK_TIMEOUT == int(os.getenv("CELERY_TASK_TIMEOUT", "600"))


def test_pool_runs_items_in_parallel():
    barrier = threading.Barrier(4, timeout=5)

    def call(i):
        barrier.wait()  # passes only when all four items run at the same time
        return i

    with Pool([partial(call, i) for i in range(4)], workers=4, timeout=10, pause=0) as pool:
        assert [pool.result(i) for i in range(4)] == [0, 1, 2, 3]


def test_pool_never_runs_more_items_than_workers():
    lock = threading.Lock()
    active = [0]
    peak = [0]

    def call():
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return True

    with Pool([call] * 6, workers=2, timeout=10, pause=0) as pool:
        assert all(pool.result(i) for i in range(6))
    assert peak[0] == 2


def test_pool_retries_a_failed_item_up_to_three_times_with_a_pause():
    calls = []

    def flaky():
        calls.append(time.monotonic())
        if len(calls) < 3:
            raise ConnectionError("grafana busy")
        return True

    with Pool([flaky], workers=1, timeout=10, pause=0.05) as pool:
        assert pool.result(0) is True
    assert len(calls) == 3
    assert calls[1] - calls[0] >= 0.045 and calls[2] - calls[1] >= 0.045


def test_pool_gives_up_after_three_attempts_and_raises_the_last_error():
    calls = []

    def broken():
        calls.append(1)
        raise ConnectionError(f"attempt {len(calls)}")

    with Pool([broken, lambda: "next"], workers=1, timeout=10, pause=0) as pool:
        with pytest.raises(ConnectionError, match="attempt 3"):
            pool.result(0)
        assert pool.result(1) == "next"  # a failed item does not stop the others
    assert len(calls) == 3


def test_pool_takes_a_returned_refusal_as_final():
    calls = []

    def refused():
        calls.append(1)
        return False

    with Pool([refused], workers=1, timeout=10, pause=0) as pool:
        assert pool.result(0) is False
    assert calls == [1]


def test_pool_deadline_counts_from_the_start_of_each_item():
    def slow():
        time.sleep(0.2)
        return "ok"

    # One thread: the second item starts at 0.2 s and ends at 0.4 s, past 0.3 s from the pool start.
    with Pool([slow, slow], workers=1, timeout=0.3, pause=0) as pool:
        assert pool.result(0) == "ok"
        assert pool.result(1) == "ok"


def test_pool_abandons_a_hung_item_and_the_queue_keeps_moving():
    release = threading.Event()
    calls = []

    def hung():
        calls.append(1)
        release.wait(10)
        raise ConnectionError("finally failed")  # would be retried if the item were still wanted

    started = time.monotonic()
    pool = Pool([hung, lambda: "behind"], workers=1, timeout=0.3, pause=0, name="hang-test")
    with pool:
        with pytest.raises(TimeoutError):
            pool.result(0)
        with pytest.raises(TimeoutError):
            pool.result(0)  # asking again does not wait again
        # The only thread is stuck; a new one runs the item queued behind it.
        assert pool.result(1) == "behind"
    assert time.monotonic() - started < 2.0

    stuck = [t for t in threading.enumerate() if t.name.startswith("report-hang-test")]
    assert stuck and all(t.daemon for t in stuck)  # never holds up the exit of the process
    release.set()
    for thread in stuck:
        thread.join(5)
    assert calls == [1]  # no new attempt for an abandoned item


def test_pool_stops_retrying_an_item_abandoned_during_the_pause():
    calls = []

    def failing():
        calls.append(1)
        raise ConnectionError("down")

    with Pool([failing], workers=1, timeout=0.2, pause=0.5, name="pause-test") as pool:
        with pytest.raises(TimeoutError):
            pool.result(0)  # gives up while the item sleeps before its second attempt
    time.sleep(0.6)
    assert calls == [1]


def test_pool_keeps_a_system_exit_inside_the_item():
    def exits():
        raise SystemExit(3)  # some collectors call sys.exit()

    with Pool([exits], workers=1, timeout=5, pause=0) as pool:
        with pytest.raises(RuntimeError, match="SystemExit"):
            pool.result(0)


def test_pool_close_drops_items_that_have_not_started():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def first():
        entered.set()
        release.wait(5)
        calls.append("first")
        return True

    pool = Pool([first, lambda: calls.append("second")], workers=1, timeout=10, pause=0)
    assert entered.wait(5)
    pool.close()
    release.set()
    assert pool.result(0) is True  # a running item still finishes
    with pytest.raises(RuntimeError):
        pool.result(1)
    time.sleep(0.1)
    assert calls == ["first"]


def test_pool_without_items_starts_no_thread():
    with Pool([], workers=4, timeout=1) as pool:
        assert pool._spawned == 0


@pytest.mark.parametrize("llm, expected", [
    ({"max_domain_workers": 5}, 5),
    ({"max_domain_workers": "2"}, 2),
    ({"max_domain_workers": 50}, 8),
    ({"max_domain_workers": 0}, 1),
    ({"max_domain_workers": "many"}, 4),
    ({}, 4),
    (None, 4),
])
def test_transfer_workers_follow_max_domain_workers_within_bounds(llm, expected):
    assert update_page._transfer_workers({"llm": llm}) == expected


# --- update_report, template flow -------------------------------------------------------------


@pytest.fixture
def report_env(monkeypatch, tmp_path):
    """update_report of the template flow with fake Confluence, Grafana and Loki."""
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps(LEGACY), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    monkeypatch.setattr(temp_files, "TEMP_DIR", tmp_path / "temporary_files")
    monkeypatch.setattr(update_page, "_TRANSFER_RETRY_PAUSE_SEC", 0)
    monkeypatch.setattr(update_page, "_active_metrics_config_now", lambda: {"shop": {"services": {"checkout": SERVICE_CONFIG}}})
    monkeypatch.setattr(update_page, "_resolve_services_for_area", lambda area: ["checkout"])

    state = {"multi": [], "uploads": [], "downloads": [], "fail": set(), "hang": {}, "threads": set()}
    lock = threading.Lock()

    def no_db():
        raise AssertionError("pipeline worker tests must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)

    def copy_page(url, user, password, sample_id, parent_id):
        state["copy"] = (url, user, password, sample_id, parent_id)
        return PAGE_ID

    def multi(url, user, password, page_id, replacements, as_text=False):
        state["multi"].append({"url": url, "user": user, "page_id": page_id, "replacements": dict(replacements), "as_text": as_text})
        return "ok"

    def fake_image(url, basename, login, password):
        name = basename.split("_")[0]
        with lock:
            state["downloads"].append(("image", name, url, login, password))
            state["threads"].add(threading.get_ident())
        if name in state["hang"]:
            state["hang"][name].wait(10)
            return False
        if name in state["fail"]:
            return False
        # Later items finish first, so completion order differs from the order of the placeholders.
        time.sleep({"rps": 0.06, "cpu": 0.03}.get(name, 0.0))
        Path(temp_files.temp_file_path(basename, ".jpg")).write_bytes(b"\xff\xd8jpeg")
        return True

    def fake_log(loki_url, start_ts, end_ts, filter_query, basename):
        with lock:
            state["downloads"].append(("log", basename, loki_url, start_ts, end_ts, filter_query))
        Path(temp_files.temp_file_path(basename, ".log")).write_text("line\n", encoding="utf-8")
        return True

    def fake_attach(func, url_basic, auth, page_id, file_path):
        time.sleep(0.04 if file_path.endswith("rps_checkout_777.jpg") else 0.0)
        with lock:
            state["uploads"].append((func.__name__, url_basic, auth.username, auth.password, page_id, Path(file_path).name))
        return True

    monkeypatch.setattr(update_page, "copy_confluence_page", copy_page)
    monkeypatch.setattr(update_page, "update_confluence_page_multi", multi)
    monkeypatch.setattr(update_page, "_download_img_with_retry", fake_image)
    monkeypatch.setattr(update_page, "_download_log_with_retry", fake_log)
    monkeypatch.setattr(update_page, "_attach_with_retry", fake_attach)

    def no_llm(*args, **kwargs):
        raise AssertionError("the LLM pipeline is not part of these tests")

    monkeypatch.setattr(update_page, "uploadFromLLM", no_llm)
    return state


def _run_report(progress, **kwargs):
    caller = threading.get_ident()
    callers = set()

    def callback(msg, pct):
        callers.add(threading.get_ident())
        progress.append((msg, pct))

    result = update_page.update_report(
        START_MS, END_MS, "checkout",
        use_llm=False, save_to_db=False, run_name="run-1", project_area="shop",
        progress_callback=callback, **kwargs,
    )
    # The job store is reached only through the callback, and only from the thread of the report.
    assert callers == {caller}
    return result


def test_template_report_keeps_placeholders_order_and_progress(report_env, caplog):
    report_env["fail"].add("mem")
    progress = []
    with caplog.at_level(logging.ERROR, logger="update_page"):
        result = _run_report(progress)

    assert result["page_id"] == PAGE_ID
    assert result["page_url"] == f"{LEGACY['url_basic']}/pages/viewpage.action?pageId={PAGE_ID}"
    assert result["run_name"] == "run-1"
    assert report_env["copy"] == (LEGACY["url_basic"], LEGACY["user"], LEGACY["password"], "200", "100")

    # Charts and logs were downloaded in parallel with the settings of the runtime file.
    images = [d for d in report_env["downloads"] if d[0] == "image"]
    assert {d[1] for d in images} == {"rps", "cpu", "mem"}
    assert all(d[2].startswith(LEGACY["grafana_base_url"] + "/render/") and d[2].endswith(f"&from={START_MS}&to={END_MS}") for d in images)
    assert all(d[3:] == (LEGACY["grafana_login"], LEGACY["grafana_pass"]) for d in images)
    assert len([d for d in images if d[1] == "mem"]) == 3  # a failing download is tried three times
    logs = [d for d in report_env["downloads"] if d[0] == "log"]
    assert sorted(d[1] for d in logs) == ["checkout_errors_777", "checkout_warns_777"]
    assert all(d[2:5] == (LEGACY["loki_url"], START_MS, END_MS) for d in logs)
    assert len(report_env["threads"]) > 1

    # The failed chart is logged and skipped; the others replace their placeholders in config order.
    assert "Ошибка при скачивании metric 'mem'" in caplog.text
    assert len(report_env["multi"]) == 1
    multi = report_env["multi"][0]
    assert (multi["url"], multi["user"], multi["page_id"], multi["as_text"]) == (LEGACY["url_basic"], LEGACY["user"], PAGE_ID, False)
    assert list(multi["replacements"].items()) == [
        ("$$rps$$", _image_html("rps_checkout_777")),
        ("$$cpu$$", _image_html("cpu_checkout_777")),
        ("$$errors$$", _log_html("checkout_errors_777")),
        ("$$warns$$", _log_html("checkout_warns_777")),
    ]
    uploads = sorted(report_env["uploads"], key=lambda u: u[-1])
    assert uploads == sorted([
        ("send_file_to_attachment", LEGACY["url_basic"], LEGACY["user"], LEGACY["password"], PAGE_ID, "rps_checkout_777.jpg"),
        ("send_file_to_attachment", LEGACY["url_basic"], LEGACY["user"], LEGACY["password"], PAGE_ID, "cpu_checkout_777.jpg"),
        ("send_loki_file_to_attachment", LEGACY["url_basic"], LEGACY["user"], LEGACY["password"], PAGE_ID, "checkout_errors_777.log"),
        ("send_loki_file_to_attachment", LEGACY["url_basic"], LEGACY["user"], LEGACY["password"], PAGE_ID, "checkout_warns_777.log"),
    ], key=lambda u: u[-1])

    span = update_page._span_percent
    assert progress == [
        ("Создание отчёта начато…", 5),
        ("Скачивание графиков и логов…", 10),
        ("Скачивание графика «rps» (1/5)", span(10, 30, 0, 5)),
        ("Скачивание графика «cpu» (2/5)", span(10, 30, 1, 5)),
        ("Скачивание графика «mem» (3/5)", span(10, 30, 2, 5)),
        ("Скачивание логов «$$errors$$» (4/5)", span(10, 30, 3, 5)),
        ("Скачивание логов «$$warns$$» (5/5)", span(10, 30, 4, 5)),
        ("Загрузка вложения в Confluence (1/4)", span(30, 40, 0, 4)),
        ("Загрузка вложения в Confluence (2/4)", span(30, 40, 1, 4)),
        ("Загрузка вложения в Confluence (3/4)", span(30, 40, 2, 4)),
        ("Загрузка вложения в Confluence (4/4)", span(30, 40, 3, 4)),
        ("Пропускаем LLM-анализ и сбор доменных данных по запросу пользователя", None),
        ("Обновление данных LLM (одним проходом)...", 85),
        ("ИИ-анализ завершён. Финализация отчёта…", 95),
        ("Отчёт готов ✅", 100),
    ]
    # Temporary files are removed once attached.
    assert list(temp_files.TEMP_DIR.iterdir()) == []


def test_template_report_with_one_worker_uploads_in_placeholder_order(report_env, monkeypatch):
    monkeypatch.setitem(core.CONFIG, "llm", {**core.CONFIG.get("llm", {}), "max_domain_workers": 1})
    _run_report([])
    assert [u[-1] for u in report_env["uploads"]] == [
        "rps_checkout_777.jpg", "cpu_checkout_777.jpg", "mem_checkout_777.jpg",
        "checkout_errors_777.log", "checkout_warns_777.log",
    ]
    assert len(report_env["threads"]) == 1


def test_hung_download_does_not_block_the_report(report_env, monkeypatch, caplog):
    monkeypatch.setattr(update_page, "_TASK_TIMEOUT", 0.5)
    release = threading.Event()
    report_env["hang"]["cpu"] = release
    started = time.monotonic()
    try:
        with caplog.at_level(logging.ERROR, logger="update_page"):
            _run_report([])
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert elapsed < 5
    assert "Ошибка при скачивании metric 'cpu'" in caplog.text
    replaced = list(report_env["multi"][0]["replacements"])
    assert replaced == ["$$rps$$", "$$mem$$", "$$errors$$", "$$warns$$"]


def test_report_reads_template_settings_saved_after_start(report_env, tmp_path):
    """A worker keeps the CONFIG of its start; the template flow reads the runtime file instead."""
    runtime = Path(update_page._SETTINGS_RUNTIME_PATH)
    runtime.write_text(json.dumps({**LEGACY, "user": "renamed", "url_basic": "https://wiki2.test"}), encoding="utf-8")
    result = _run_report([])
    assert report_env["copy"][:2] == ("https://wiki2.test", "renamed")
    assert result["page_url"].startswith("https://wiki2.test/")
    assert {u[2] for u in report_env["uploads"]} == {"renamed"}


# --- worker safety ----------------------------------------------------------------------------


def _forbid_runtime_writes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the report path must not write the runtime config (read-only in the worker)")

    monkeypatch.setattr(core, "_write_json_atomic", refuse)


def test_template_report_runs_like_in_a_worker(report_env, monkeypatch):
    import flask

    _forbid_runtime_writes(monkeypatch)
    seen = {}
    original_copy = update_page.copy_confluence_page

    def copy_page(*args):
        seen["request"] = flask.has_request_context()
        seen["app"] = flask.has_app_context()
        return original_copy(*args)

    monkeypatch.setattr(update_page, "copy_confluence_page", copy_page)
    result = _run_report([])
    assert result["page_id"] == PAGE_ID
    assert seen == {"request": False, "app": False}


def test_web_report_runs_like_in_a_worker(report_env, monkeypatch):
    import flask

    _forbid_runtime_writes(monkeypatch)
    calls = []

    def pipeline(start, end, **kwargs):
        calls.append(kwargs)
        assert not flask.has_request_context() and not flask.has_app_context()
        kwargs["progress_callback"]("Сбор данных…", 20)
        return {}

    monkeypatch.setattr(update_page, "uploadFromLLM", pipeline)
    progress = []
    result = _run_report(progress, web_only=True)
    assert result["page_id"] is None and result["page_url"] == f"/reports/{result['run_id']}"
    assert calls and calls[0]["run_meta"]["area_services"] == ["checkout"]
    assert ("Сбор данных…", 20) in progress and progress[-1] == ("Отчёт (веб) готов ✅", 100)
    assert report_env["downloads"] == [] and report_env["uploads"] == []


def _module_tree(relative: str) -> ast.Module:
    return ast.parse((ROOT_DIR / relative).read_text(encoding="utf-8"), filename=relative)


def _imported_modules(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_update_page_no_longer_imports_celery():
    tree = _module_tree("update_page.py")
    assert not {name for name in _imported_modules(tree) if "celery" in name.lower()}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "celery_app" not in names and "_await_task" not in names
    assert not {"delay", "apply_async"} & called  # no sub-task is queued from the report
    assert not hasattr(update_page, "_await_task")
    for legacy in ("download_grafana_image_task", "download_loki_logs_task", "upload_attachment_task"):
        assert not hasattr(getattr(update_page, legacy), "delay")


# Modules the three background jobs run in a worker: update_report, publish_report, publish_forecast.
WORKER_PATH_MODULES = sorted(
    {
        "update_page.py",
        "loadlens_app/confluence_export.py",
        "loadlens_app/forecast_confluence.py",
        "loadlens_app/forecast.py",
        "loadlens_app/forecast_explanation.py",
    }
    | {str(p.relative_to(ROOT_DIR)).replace("\\", "/") for p in (ROOT_DIR / "AI").glob("*.py")}
    | {str(p.relative_to(ROOT_DIR)).replace("\\", "/") for p in (ROOT_DIR / "confluence_manager").glob("*.py")}
    | {str(p.relative_to(ROOT_DIR)).replace("\\", "/") for p in (ROOT_DIR / "data_collectors").glob("*.py")}
)
RUNTIME_WRITERS = {
    "_save_settings_runtime_data", "_save_metrics_runtime_data", "_write_json_atomic",
    "_bootstrap_service_configs", "_bootstrap_area_defaults", "_bootstrap_metrics_service_config",
    "_bootstrap_service_queries", "migrate_legacy_data_sources",
}


@pytest.mark.parametrize("relative", WORKER_PATH_MODULES)
def test_worker_path_needs_no_request_job_store_or_runtime_writes(relative):
    tree = _module_tree(relative)
    imported = _imported_modules(tree)
    assert not {name for name in imported if name == "flask" or name.startswith("flask.")}, "request state is not available in a worker"
    assert not {name for name in imported if name.startswith("loadlens_app.jobs")}, "progress goes through the callback only"
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    used |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert not used & RUNTIME_WRITERS, "runtime config files are mounted read-only in the worker"


def test_files_of_the_report_do_not_depend_on_the_working_directory():
    # A worker may start in another directory; temporary files (the volume shared with the app in
    # docker), runtime config and prompts are all located from the module files.
    assert temp_files.TEMP_DIR.is_absolute() and temp_files.TEMP_DIR == ROOT_DIR / "data_collectors" / "temporary_files"
    for path in (update_page._SETTINGS_RUNTIME_PATH, update_page._METRICS_RUNTIME_PATH, update_page._PROMPTS_DIR):
        assert os.path.isabs(path) and Path(path).parent.resolve() in {ROOT_DIR.resolve(), (ROOT_DIR / "AI").resolve()}


# --- runtime settings in a worker -------------------------------------------------------------


@pytest.fixture
def config_snapshot():
    saved = dict(update_page.CONFIG)
    yield saved
    update_page.CONFIG.clear()
    update_page.CONFIG.update(saved)


def test_refresh_runtime_config_applies_settings_saved_after_start(monkeypatch, tmp_path, config_snapshot):
    from loadlens_app import confluence_export

    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    assert update_page.CONFIG is core.CONFIG is confluence_export.CONFIG  # one dict for every module

    assert update_page.refresh_runtime_config() is False  # no file: nothing changes
    assert update_page.CONFIG == config_snapshot
    runtime.write_text("{broken", encoding="utf-8")
    assert update_page.refresh_runtime_config() is False
    assert update_page.CONFIG == config_snapshot

    runtime.write_text(json.dumps({
        "confluence": {"space_key": "NEW"},
        "llm": {"provider": "stub"},
        "user": "fresh",
    }), encoding="utf-8")
    assert update_page.refresh_runtime_config() is True
    assert confluence_export.confluence_config()["space_key"] == "NEW"
    assert confluence_export.confluence_config()["base_url"] == config_snapshot["confluence"]["base_url"]  # merged, not replaced
    assert core.CONFIG["llm"]["provider"] == "stub"
    assert core.CONFIG["llm"]["max_domain_workers"] == config_snapshot["llm"]["max_domain_workers"]
    assert core.CONFIG["user"] == "fresh"

    once = json.dumps(update_page.CONFIG, sort_keys=True, default=str)
    assert update_page.refresh_runtime_config() is True
    assert json.dumps(update_page.CONFIG, sort_keys=True, default=str) == once  # idempotent
