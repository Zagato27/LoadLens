"""Report and publication jobs as Celery tasks (loadlens_app.tasks) and the choice of runner (loadlens_app.job_runner).

The web process and the worker are two JobStore instances on one fake report_jobs table (``JobsTable`` from
tests/test_perf_job_store.py). Nothing reaches a broker: ``apply_async`` is replaced by a recorder, a worker
delivery pushes a Celery request (``called_directly=False``) and calls the task body, and the eager test uses
Celery's own in-process execution.
"""

import json
import logging
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from loadlens_app import celery_app as celery_module, job_runner, jobs, tasks  # noqa: E402
from loadlens_app.blueprints import dashboard, forecast as forecast_api  # noqa: E402
from loadlens_app.limits import LimitError  # noqa: E402
from tests.test_perf_job_store import JobsTable  # noqa: E402

import update_page  # noqa: E402

from celery.exceptions import SoftTimeLimitExceeded  # noqa: E402
from kombu.exceptions import OperationalError  # noqa: E402
from redis.exceptions import TimeoutError as RedisTimeoutError  # noqa: E402

REAL_THREAD = threading.Thread
ENV_VARS = ("LOADLENS_JOB_RUNNER", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", "CELERY_TASK_ALWAYS_EAGER")
REPORT_ARGS = dict(
    use_llm=False, save_to_db=False, web_only=True, run_name="nightly", test_type="step", owner="eng",
    run_tag="nightly", run_filter_packs=["lt-k6-influxql"],
)


# ---- fixtures ------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No broker variables from the shell, no storage but the fake job table, no runtime config files."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(celery_module, "_warned", set())

    def no_db():
        raise AssertionError("these tests must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(dashboard, "_report_step", lambda area, service: "30s")
    monkeypatch.setattr(tasks, "FINAL_UPDATE_RETRY_DELAYS_SEC", (0.0, 0.0))


@pytest.fixture
def table(monkeypatch):
    fake = JobsTable()
    monkeypatch.setattr(jobs, "_ts_conn", fake.connect)
    return fake


@pytest.fixture
def web(monkeypatch, table):
    """The job store of the web process, wherever the code looks it up."""
    store = jobs.JobStore()
    for module in (jobs, dashboard, forecast_api):
        monkeypatch.setattr(module, "job_store", store)
    return store


@pytest.fixture
def worker(table):
    """The job store of a Celery worker process on the same table."""
    return jobs.JobStore()


@pytest.fixture
def threads(monkeypatch):
    """threading.Thread replaced like tests/test_smoke_api.py does: start() runs the target at once."""
    started = []

    def fake_thread(target, daemon=True):
        assert daemon is True
        started.append(target)
        return types.SimpleNamespace(start=target)

    monkeypatch.setattr(threading, "Thread", fake_thread)
    return started


@pytest.fixture
def celery_mode(monkeypatch):
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", "celery")


class Broker:
    """Stands in for apply_async of the three tasks; ``error`` makes every send fail with it."""

    def __init__(self, monkeypatch):
        self.sent = []
        self.error = None
        self.on_send = None
        for task in (tasks.report_task, tasks.publish_confluence_task, tasks.publish_forecast_task):
            monkeypatch.setattr(task, "apply_async", self._sender(task.name))

    def _sender(self, name):
        def apply_async(args=None, kwargs=None, **options):
            assert not args
            json.dumps(kwargs)  # the JSON serializer of the Celery app refuses anything else
            self.sent.append(types.SimpleNamespace(name=name, kwargs=kwargs, options=options))
            if self.on_send is not None:
                self.on_send(kwargs)
            if self.error is not None:
                raise self.error

        return apply_async


@pytest.fixture
def broker(monkeypatch):
    return Broker(monkeypatch)


@pytest.fixture
def refreshes(monkeypatch):
    calls = []
    monkeypatch.setattr(update_page, "refresh_runtime_config", lambda: calls.append(True) or True)
    return calls


def deliver(monkeypatch, store, name, kwargs, *, eager=False):
    """Runs a task as the Celery worker would: with a request on its stack, the worker's store, JSON arguments."""
    # The delivery starts a heartbeat thread: under the synchronous Thread stand-in its loop would never return.
    assert isinstance(threading.Thread, type), "do not combine the threads fixture with a worker delivery"
    task = tasks.celery_app.tasks[name]
    with monkeypatch.context() as patch:
        for module in (jobs, dashboard, forecast_api):
            patch.setattr(module, "job_store", store)
        task.push_request(id=kwargs["job_id"], called_directly=False, is_eager=eager)
        try:
            task.run(**json.loads(json.dumps(kwargs)))
        finally:
            task.pop_request()


def start_report(**overrides):
    args = dict(REPORT_ARGS, **overrides)
    return dashboard.start_report_job(0, 600_000, "checkout", "shop", **args)


def fake_update_report(calls, result=None):
    def update_report(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["progress_callback"]("Сбор метрик: JVM (1/6)", 40)
        return result if result is not None else {"run_id": "r-1", "page_url": "/reports/demo/nightly", "page_id": 7, "run_name": "nightly"}

    return update_report


# ---- mode and submit -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, "thread"),
        ({"CELERY_BROKER_URL": "redis://127.0.0.1:16379/4"}, "thread"),  # eager mode is the default
        ({"CELERY_BROKER_URL": "redis://127.0.0.1:16379/4", "CELERY_TASK_ALWAYS_EAGER": "0"}, "celery"),
        ({"CELERY_BROKER_URL": "redis://127.0.0.1:16379/4", "CELERY_TASK_ALWAYS_EAGER": "0", "LOADLENS_JOB_RUNNER": "thread"}, "thread"),
        ({"LOADLENS_JOB_RUNNER": "celery"}, "celery"),
    ],
)
def test_mode_follows_celery_mode_enabled(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert job_runner.mode() == expected


def test_thread_mode_runs_the_task_in_a_daemon_thread(web, threads, monkeypatch, refreshes):
    calls = []
    monkeypatch.setattr(dashboard, "update_report", fake_update_report(calls))

    job = start_report()

    assert len(threads) == 1 and len(calls) == 1
    args, kwargs = calls[0]
    assert args == (0, 600_000, "checkout")
    assert kwargs["project_area"] == "shop" and kwargs["run_tag"] == "nightly" and kwargs["run_filter_packs"] == ["lt-k6-influxql"]
    record = web.get(job.job_id)
    assert (record.runner, record.status, record.progress, record.message) == ("thread", "done", 100, "Готово")
    assert (record.report_url, record.page_url, record.page_id, record.run_name) == ("/reports/demo/nightly", "/reports/r-1", "7", "nightly")
    # A thread of the web process: current settings, nothing to claim, no heartbeat thread.
    assert refreshes == []


def test_a_failing_thread_job_names_the_stage(web, threads, monkeypatch):
    def failing(*args, **kwargs):
        kwargs["progress_callback"]("Сбор метрик: JVM (1/6)", 10)
        raise RuntimeError("Grafana returned 502")

    monkeypatch.setattr(dashboard, "update_report", failing)
    job = start_report()

    record = web.get(job.job_id)
    assert (record.status, record.message, record.error, record.progress) == (
        "error", "Ошибка на этапе «Сбор метрик: JVM (1/6)»", "Grafana returned 502", 10,
    )


def test_celery_mode_sends_the_report_to_the_reports_queue(web, broker, celery_mode, threads, table):
    job = start_report()

    assert threads == []  # nothing runs in the web process
    [sent] = broker.sent
    assert sent.name == "loadlens.report"
    assert sent.options == {"queue": "reports", "task_id": job.job_id}
    assert sent.kwargs == {
        "job_id": job.job_id, "start": 0, "end": 600_000, "service": "checkout", "project_area": "shop",
        "use_llm": False, "save_to_db": False, "web_only": True, "run_name": "nightly", "test_type": "step",
        "run_tag": "nightly", "run_filter_packs": ["lt-k6-influxql"],
    }
    row = table.rows[job.job_id]
    assert (row["runner"], row["status"], row["started_at"], row["owner"]) == ("celery", "running", None, "eng")
    assert job.job_id not in web._jobs  # a Celery job lives only in the table


def test_a_report_delivered_to_the_worker_reaches_done_and_the_web_process_sees_it(
    web, worker, broker, celery_mode, monkeypatch, refreshes, table,
):
    calls = []
    monkeypatch.setattr(dashboard, "update_report", fake_update_report(calls))
    job = start_report()
    assert calls == []
    seen = []
    original_update = worker.update
    monkeypatch.setattr(worker, "update", lambda job_id, **fields: seen.append(dict(fields)) or original_update(job_id, **fields))

    deliver(monkeypatch, worker, "loadlens.report", broker.sent[0].kwargs)

    assert refreshes == [True]  # a worker process re-reads settings_runtime.json first
    assert len(calls) == 1 and calls[0][1]["run_filter_packs"] == ["lt-k6-influxql"]
    assert {"message": "Сбор метрик: JVM (1/6)", "progress": 40} in seen  # progress went to the table
    record = web.get(job.job_id)
    assert (record.runner, record.status, record.progress, record.page_url, record.page_id) == ("celery", "done", 100, "/reports/r-1", "7")
    assert record.started_at is not None
    assert job.job_id not in worker._jobs


def test_a_failing_worker_job_names_the_stage(web, worker, broker, celery_mode, monkeypatch, refreshes):
    def failing(*args, **kwargs):
        kwargs["progress_callback"]("Анализ LLM", 70)
        raise RuntimeError("LLM timed out")

    monkeypatch.setattr(dashboard, "update_report", failing)
    job = start_report()
    deliver(monkeypatch, worker, "loadlens.report", broker.sent[0].kwargs)

    record = web.get(job.job_id)
    assert (record.status, record.message, record.error, record.progress) == ("error", "Ошибка на этапе «Анализ LLM»", "LLM timed out", 70)


def test_eager_celery_runs_the_task_through_celery_in_the_caller(web, celery_mode, monkeypatch, refreshes, table):
    monkeypatch.setattr(tasks.celery_app.conf, "task_always_eager", True)
    calls = []
    monkeypatch.setattr(dashboard, "update_report", fake_update_report(calls))

    job = start_report()

    assert len(calls) == 1
    row = table.rows[job.job_id]
    assert (row["runner"], row["status"], row["progress"]) == ("celery", "done", 100)
    assert row["started_at"] is not None  # claimed like in a worker
    assert refreshes == []  # but this is the web process: its settings are current


# ---- broker failures -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        OperationalError("Error 10061 connecting to 127.0.0.1:16379. No connection could be made."),
        ConnectionRefusedError(10061, "refused"),
        RedisTimeoutError("Timeout reading from socket"),
    ],
    ids=["kombu", "socket", "redis"],
)
def test_a_broker_error_makes_the_web_process_adopt_the_job(web, broker, celery_mode, threads, monkeypatch, caplog, table, error):
    broker.error = error
    monkeypatch.setattr(dashboard, "update_report", fake_update_report([]))

    with caplog.at_level(logging.WARNING, logger="loadlens_app.job_runner"):
        job = start_report()

    assert len(broker.sent) == 1 and len(threads) == 1
    assert "running it in a thread of the web process" in caplog.text
    record = web.get(job.job_id)
    assert (record.runner, record.status, record.progress) == ("thread", "done", 100)
    assert job.job_id in web._jobs
    assert table.rows[job.job_id]["runner"] == "thread"


def test_no_take_over_when_a_worker_started_the_job_despite_the_error(web, worker, broker, celery_mode, threads, caplog):
    broker.error = OperationalError("timed out reading the answer")
    broker.on_send = lambda kwargs: worker.mark_started(kwargs["job_id"])  # the message did get through

    with caplog.at_level(logging.WARNING, logger="loadlens_app.job_runner"):
        job = start_report()

    assert threads == []
    assert "not running it here" in caplog.text
    assert web.get(job.job_id).runner == "celery"


def test_a_late_delivery_of_an_adopted_job_is_not_run_again(web, worker, broker, celery_mode, threads, monkeypatch, refreshes):
    broker.error = OperationalError("timed out")
    calls = []
    monkeypatch.setattr(dashboard, "update_report", fake_update_report(calls))
    job = start_report()
    assert len(calls) == 1 and web.get(job.job_id).status == "done"
    monkeypatch.setattr(threading, "Thread", REAL_THREAD)

    # The broker took the message after all, and a worker gets it now.
    deliver(monkeypatch, worker, "loadlens.report", broker.sent[0].kwargs)
    assert len(calls) == 1


def test_other_errors_of_apply_async_are_not_taken_for_a_broker_outage(web, broker, celery_mode, threads):
    broker.error = TypeError("report_task() got an unexpected keyword argument")
    with pytest.raises(TypeError):
        start_report()
    assert threads == []


# ---- create_job ----------------------------------------------------------------------------------------------


def test_a_celery_job_whose_row_was_not_written_becomes_a_thread_job(web, table, celery_mode, broker, threads, monkeypatch, caplog):
    table.down = True
    monkeypatch.setattr(dashboard, "update_report", fake_update_report([]))

    with caplog.at_level(logging.WARNING, logger="loadlens_app.job_runner"):
        job = start_report()

    assert job.runner == "thread" and job.job_id in web._jobs
    assert broker.sent == [] and len(threads) == 1
    assert "instead of the Celery worker" in caplog.text
    assert web.get(job.job_id).status == "done"  # followed in memory while the table is down


def test_a_celery_row_that_could_not_be_read_back_does_not_hold_a_slot(web, table, celery_mode, broker, threads, monkeypatch):
    real_get = web.get
    misses = []

    def get(job_id):
        if not misses:
            misses.append(job_id)
            return None
        return real_get(job_id)

    monkeypatch.setattr(web, "get", get)
    monkeypatch.setattr(dashboard, "update_report", fake_update_report([]))
    job = start_report()

    [orphan] = misses
    assert orphan != job.job_id
    assert (table.rows[orphan]["runner"], table.rows[orphan]["status"]) == ("celery", "error")
    assert table.rows[orphan]["message"] == job_runner.REPLACED_MESSAGE
    assert table.rows[job.job_id]["runner"] == "thread"
    assert web.running_counts("eng") == (0, 0)


# ---- claiming a delivery -------------------------------------------------------------------------------------


def _queued_report(web, broker):
    job = start_report()
    return job, broker.sent[-1].kwargs


@pytest.mark.parametrize(
    "row_change, reason",
    [
        ({"status": "done"}, "its status is already done"),
        ({"status": "error"}, "its status is already error"),
        ({"started_at": "now", "runner": "thread"}, "by the web process after a failed send"),
        ({"started_at": "now"}, "by another delivery of the task"),
    ],
    ids=["finished", "swept", "adopted", "duplicate"],
)
def test_a_delivery_does_not_run_a_job_it_must_not(web, worker, broker, celery_mode, monkeypatch, table, caplog, refreshes, row_change, reason):
    job, kwargs = _queued_report(web, broker)
    row = table.rows[job.job_id]
    row.update({key: (row["created_at"] if value == "now" else value) for key, value in row_change.items()})
    calls = []
    monkeypatch.setattr(dashboard, "update_report", fake_update_report(calls))

    with caplog.at_level(logging.WARNING, logger="loadlens_app.tasks"):
        deliver(monkeypatch, worker, "loadlens.report", kwargs)

    assert calls == []
    assert reason in caplog.text


def test_a_delivery_of_an_unknown_job_is_skipped(worker, table, monkeypatch, refreshes):
    calls = []
    monkeypatch.setattr(dashboard, "update_report", fake_update_report(calls))
    deliver(monkeypatch, worker, "loadlens.report", dict(
        job_id="gone", start=0, end=1, service="s", project_area="a", use_llm=False, save_to_db=False,
        web_only=True, run_name="n", test_type="", run_tag=None, run_filter_packs=None,
    ))
    assert calls == [] and "gone" not in table.rows


def test_without_the_job_table_the_worker_runs_the_job_and_retries_the_final_state(web, worker, broker, celery_mode, monkeypatch, table, refreshes):
    job, kwargs = _queued_report(web, broker)
    table.down = True
    calls = []
    monkeypatch.setattr(dashboard, "update_report", fake_update_report(calls))
    finals = []
    original_update = worker.update
    monkeypatch.setattr(worker, "update", lambda job_id, **fields: (fields.get("status") and finals.append(fields)) or original_update(job_id, **fields))

    deliver(monkeypatch, worker, "loadlens.report", kwargs)

    assert len(calls) == 1
    assert len(finals) == 1 + len(tasks.FINAL_UPDATE_RETRY_DELAYS_SEC)


def test_the_final_state_reaches_the_table_once_it_answers_again(web, worker, broker, celery_mode, monkeypatch, table, refreshes):
    job, kwargs = _queued_report(web, broker)
    original_update = worker.update
    attempts = []

    def flaky_update(job_id, **fields):
        if fields.get("status"):
            attempts.append(fields["status"])
            if len(attempts) == 1:
                return None  # the table did not take the first write
        return original_update(job_id, **fields)

    monkeypatch.setattr(worker, "update", flaky_update)
    monkeypatch.setattr(dashboard, "update_report", fake_update_report([]))
    deliver(monkeypatch, worker, "loadlens.report", kwargs)

    assert attempts == ["done", "done"]
    assert table.rows[job.job_id]["status"] == "done"


# ---- heartbeat and time limit --------------------------------------------------------------------------------


class RecordingThread(threading.Thread):
    created = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RecordingThread.created.append(self)


def test_the_heartbeat_runs_while_the_worker_works_and_stops_after(web, worker, broker, celery_mode, monkeypatch, table, refreshes):
    job, kwargs = _queued_report(web, broker)
    monkeypatch.setattr(tasks, "HEARTBEAT_INTERVAL_SEC", 0.02)
    RecordingThread.created = []
    monkeypatch.setattr(threading, "Thread", RecordingThread)
    beats = []
    original_heartbeat = worker.heartbeat
    monkeypatch.setattr(worker, "heartbeat", lambda job_id: beats.append(time.monotonic()) or original_heartbeat(job_id))

    def slow_report(*args, **kwargs):
        time.sleep(0.25)  # a long LLM call without progress
        return {"run_id": "r-1"}

    monkeypatch.setattr(dashboard, "update_report", slow_report)
    deliver(monkeypatch, worker, "loadlens.report", kwargs)

    assert len(beats) >= 3
    [heartbeat_thread] = RecordingThread.created
    assert heartbeat_thread.daemon
    heartbeat_thread.join(timeout=2)
    assert not heartbeat_thread.is_alive()
    count = len(beats)
    time.sleep(0.1)
    assert len(beats) == count
    assert table.rows[job.job_id]["status"] == "done"


def test_the_heartbeat_stops_by_itself_once_the_row_is_not_running(worker, table, monkeypatch):
    table.add("swept", "error", runner="celery")
    monkeypatch.setattr(tasks, "HEARTBEAT_INTERVAL_SEC", 0.01)
    RecordingThread.created = []
    monkeypatch.setattr(threading, "Thread", RecordingThread)

    stop = tasks._start_heartbeat(worker, "swept")
    try:
        [heartbeat_thread] = RecordingThread.created
        heartbeat_thread.join(timeout=2)
        assert not heartbeat_thread.is_alive() and not stop.is_set()
        assert len(table.texts("SET updated_at")) == 1  # one heartbeat found the row finished
    finally:
        stop.set()


def test_a_failing_heartbeat_does_not_stop_the_job(web, worker, broker, celery_mode, monkeypatch, table, refreshes):
    job, kwargs = _queued_report(web, broker)
    monkeypatch.setattr(tasks, "HEARTBEAT_INTERVAL_SEC", 0.01)
    beats = []

    def broken_heartbeat(job_id):
        beats.append(job_id)
        raise RuntimeError("pool exhausted")

    monkeypatch.setattr(worker, "heartbeat", broken_heartbeat)

    def slow_report(*args, **kwargs):
        time.sleep(0.1)
        return {"run_id": "r-1"}

    monkeypatch.setattr(dashboard, "update_report", slow_report)
    deliver(monkeypatch, worker, "loadlens.report", kwargs)

    assert len(beats) >= 2
    assert table.rows[job.job_id]["status"] == "done"


def test_soft_time_limit_ends_a_worker_job_with_an_error(web, worker, broker, celery_mode, monkeypatch, table, refreshes):
    job, kwargs = _queued_report(web, broker)

    def too_long(*args, **kwargs):
        kwargs["progress_callback"]("Анализ LLM", 70)
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(dashboard, "update_report", too_long)
    deliver(monkeypatch, worker, "loadlens.report", kwargs)

    row = table.rows[job.job_id]
    assert (row["status"], row["message"]) == ("error", "Задача прервана: превышено время задачи")
    assert "LOADLENS_REPORT_TASK_LIMIT_SEC" in row["error"] and str(tasks.SOFT_TIME_LIMIT_SEC) in row["error"]


def test_soft_time_limit_in_a_thread_job_gives_the_same_error(web, threads, monkeypatch):
    def too_long(*args, **kwargs):
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(dashboard, "update_report", too_long)
    job = start_report()
    record = web.get(job.job_id)
    assert (record.status, record.message) == ("error", tasks.TIME_LIMIT_MESSAGE)


def test_report_tasks_carry_the_time_limits_and_ignore_results():
    soft, hard = celery_module.report_task_time_limits()
    for name in ("loadlens.report", "loadlens.publish_confluence", "loadlens.publish_forecast"):
        task = tasks.celery_app.tasks[name]
        assert (task.soft_time_limit, task.time_limit, task.ignore_result) == (soft, hard, True)
        assert celery_module.celery_app.amqp.router.route({}, name)["queue"].name == "reports"


# ---- limits stay before the job ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["thread", "celery"])
def test_start_report_job_raises_limit_errors_before_creating_a_job(web, table, broker, threads, monkeypatch, mode):
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", mode)
    monkeypatch.setattr(web, "running_counts", lambda owner: (99, 0))
    with pytest.raises(LimitError):
        start_report()
    monkeypatch.setattr(web, "running_counts", lambda owner: (0, 0))
    with pytest.raises(LimitError) as window:
        dashboard.start_report_job(0, 400 * 3_600_000, "checkout", "shop", **REPORT_ARGS)
    assert window.value.code == "window_too_long"
    assert table.rows == {} and web._jobs == {} and broker.sent == [] and threads == []


# ---- publications --------------------------------------------------------------------------------------------


@pytest.fixture
def client(web, monkeypatch):
    from app import create_app
    from loadlens_app.auth import InMemoryRepository

    monkeypatch.setattr(dashboard, "_lookup_report_ref", lambda run_id="", run_name="", service="": {"run_id": "r 1"})
    monkeypatch.setitem(core.CONFIG["confluence"], "app_url", "")  # links follow the request address
    flask_app = create_app(auth_repository=InMemoryRepository())
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_confluence_publication_goes_to_the_worker_with_the_link_built_by_the_route(client, web, worker, broker, celery_mode, monkeypatch, refreshes, table):
    published = []
    monkeypatch.setattr(dashboard, "publish_report", lambda **kwargs: published.append(kwargs) or {"page_id": 42, "page_url": "http://wiki/42"})

    response = client.post("/publish_confluence", json={"run_name": "nightly", "service": "checkout", "source_url": "https://evil.example"})

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"status", "job_id", "message"} and body["status"] == "accepted"
    [sent] = broker.sent
    assert sent.name == "loadlens.publish_confluence" and sent.options["queue"] == "reports"
    assert sent.kwargs == {"job_id": body["job_id"], "run_name": "nightly", "service": "checkout", "source_url": "http://localhost/reports/r%201"}
    assert published == []

    deliver(monkeypatch, worker, sent.name, sent.kwargs)

    assert published[0]["source_url"] == "http://localhost/reports/r%201"
    status = client.get(f"/job_status/{body['job_id']}").get_json()
    assert (status["status"], status["kind"], status["page_url"], status["page_id"]) == ("done", "confluence", "http://wiki/42", "42")


def test_create_report_goes_to_the_worker(client, web, broker, celery_mode, monkeypatch):
    monkeypatch.setattr(dashboard, "_find_area_for_service", lambda service: "demo")
    monkeypatch.setattr(dashboard, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(dashboard, "_metrics_service_entry", lambda service: ("demo", {"metrics": []}))
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda run_name, service: False)

    response = client.post("/create_report", json={"start": "2024-11-01T10:00", "end": "2024-11-01T11:00", "service": "demo", "run_name": "nightly"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "accepted" and body["run_name"] == "nightly"
    [sent] = broker.sent
    assert sent.name == "loadlens.report" and sent.kwargs["job_id"] == body["job_id"]
    assert sent.kwargs["end"] - sent.kwargs["start"] == 3_600_000
    assert (sent.kwargs["service"], sent.kwargs["project_area"], sent.kwargs["use_llm"]) == ("demo", "demo", True)


@pytest.fixture
def forecast_pages(monkeypatch):
    published = []

    def fake_publish(run_id, target, headroom, ceiling, source_url, progress):
        progress("Расчёт прогноза", 30)
        published.append((run_id, target, headroom, ceiling, source_url))
        return {"page_id": "7", "page_url": "http://wiki/7"}

    monkeypatch.setattr(forecast_api, "publish_forecast", fake_publish)
    return published


def test_forecast_publication_route_sends_the_job_to_the_worker(client, web, worker, broker, celery_mode, forecast_pages, monkeypatch, refreshes, table):
    response = client.post("/forecast/abc/confluence", json={"run_name": "nightly", "target_rps": 200, "cpu_ceiling_pct": 70})

    assert response.status_code == 200
    body = response.get_json()
    [sent] = broker.sent
    assert sent.name == "loadlens.publish_forecast" and sent.options["queue"] == "reports"
    assert sent.kwargs["job_id"] == body["job_id"] and sent.kwargs["source_url"] == "http://localhost/forecasting/abc"
    assert table.rows[body["job_id"]]["runner"] == jobs.JOB_RUNNER_CELERY
    assert forecast_pages == []

    deliver(monkeypatch, worker, sent.name, sent.kwargs)

    row = table.rows[body["job_id"]]
    assert (row["status"], row["page_url"]) == ("done", "http://wiki/7")
    assert row["report_url"] == "/forecasting/abc"


def test_a_forecast_thread_job_is_published_in_a_thread_even_in_celery_mode(web, broker, threads, celery_mode, forecast_pages):
    # A job registered with job_store.create is a thread job of this process, whatever the mode says.
    job = web.create(jobs.JOB_KIND_FORECAST_CONFLUENCE, run_name="nightly", owner="eng")
    inputs = forecast_api.ForecastInputs(target_rps=200.0, headroom_pct=20.0, cpu_ceiling_pct=None)
    forecast_api._publish_in_background(job.job_id, "abc", inputs, "http://localhost/forecasting/abc")

    assert len(threads) == 1 and broker.sent == []
    record = web.get(job.job_id)
    assert (record.status, record.page_url, record.page_id) == ("done", "http://wiki/7", "7")
    assert forecast_pages == [("abc", 200.0, 20.0, None, "http://localhost/forecasting/abc")]


def test_a_forecast_job_created_for_the_worker_is_published_there(web, worker, broker, celery_mode, forecast_pages, monkeypatch, refreshes, table):
    inputs = forecast_api.ForecastInputs(target_rps=200.0, headroom_pct=20.0, cpu_ceiling_pct=None)
    celery_job = job_runner.create_job(jobs.JOB_KIND_FORECAST_CONFLUENCE, run_name="nightly", owner="eng")
    forecast_api._publish_in_background(celery_job.job_id, "abc", inputs, "http://localhost/forecasting/abc")
    [sent] = broker.sent
    assert sent.name == "loadlens.publish_forecast" and sent.options["queue"] == "reports"
    assert sent.kwargs == {
        "job_id": celery_job.job_id, "run_id": "abc", "target_rps": 200.0, "headroom_pct": 20.0,
        "cpu_ceiling_pct": None, "source_url": "http://localhost/forecasting/abc",
    }
    deliver(monkeypatch, worker, sent.name, sent.kwargs)
    row = table.rows[celery_job.job_id]
    assert (row["status"], row["page_url"], row["page_id"], row["progress"]) == ("done", "http://wiki/7", "7", 100)
    assert forecast_pages == [("abc", 200.0, 20.0, None, "http://localhost/forecasting/abc")]
    assert refreshes == [True]


def test_a_failed_forecast_publication_in_the_worker_names_the_stage(web, worker, broker, celery_mode, monkeypatch, refreshes, table):
    def failing(run_id, target, headroom, ceiling, source_url, progress):
        progress("Загрузка вложений", 60)
        raise RuntimeError("Confluence returned 500")

    monkeypatch.setattr(forecast_api, "publish_forecast", failing)
    job = job_runner.create_job(jobs.JOB_KIND_FORECAST_CONFLUENCE, run_name="nightly", owner="eng")
    forecast_api._publish_in_background(job.job_id, "abc", forecast_api.ForecastInputs(None, 20.0, None), "http://x/forecasting/abc")
    deliver(monkeypatch, worker, broker.sent[0].name, broker.sent[0].kwargs)

    row = table.rows[job.job_id]
    assert (row["status"], row["message"], row["error"]) == ("error", "Ошибка на этапе «Загрузка вложений»", "Confluence returned 500")


def test_submit_with_a_job_id_uses_the_current_mode(web, broker, threads, monkeypatch):
    task = types.SimpleNamespace(name="loadlens.fake", ran=[])
    task.run = lambda **kwargs: task.ran.append(kwargs)
    task.apply_async = lambda **kwargs: task.ran.append(("sent", kwargs))
    assert job_runner.submit(task, "job-1", x=1) == "thread"
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", "celery")
    assert job_runner.submit(task, "job-2", x=2) == "celery"
    assert task.ran == [
        {"job_id": "job-1", "x": 1},
        ("sent", {"kwargs": {"job_id": "job-2", "x": 2}, "queue": "reports", "task_id": "job-2"}),
    ]
