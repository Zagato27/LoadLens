# SPDX-License-Identifier: Apache-2.0
"""Celery configuration and the compose worker (plans/perf-plan.md, wave B, agent B2).

Offline: nothing here connects to a broker or starts a worker. The checks read the Celery configuration,
the routing table, the job runner mode derived from the environment, and the compose and env templates.
"""

import importlib.util
import logging
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before anything that imports update_page)
from loadlens_app import celery_app as celery_module  # noqa: E402

COMPOSE_PATH = ROOT_DIR / "docker" / "docker-compose.yml"
ENV_EXAMPLE_PATH = ROOT_DIR / "docker" / "env.example"
ENV_VARS = ("LOADLENS_JOB_RUNNER", "CELERY_BROKER_URL", "CELERY_TASK_ALWAYS_EAGER", "LOADLENS_REPORT_TASK_LIMIT_SEC")
NEW_JOB_VARS = {"LOADLENS_JOB_RUNNER": "auto", "LOADLENS_JOB_STALE_MIN": "15", "LOADLENS_REPORT_TASK_LIMIT_SEC": "3600"}
BROKER = "redis://127.0.0.1:16379/2"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """The shell running pytest may carry Celery variables; every test starts without them."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(celery_module, "_warned", set())


@pytest.fixture
def make_app():
    """Builds a fresh Celery app from the current environment and restores the current app afterwards."""
    from celery import _state

    previous = _state.get_current_app()
    made = []

    def make():
        app = celery_module._make_celery()
        made.append(app)
        return app

    yield make
    previous.set_current()
    for app in made:
        app.close()


@pytest.fixture(scope="module")
def compose():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def _env_example() -> dict:
    values = {}
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _load_without_celery(monkeypatch):
    """A separate copy of loadlens_app/celery_app.py executed as if the celery package were missing."""
    monkeypatch.setitem(sys.modules, "celery", None)
    spec = importlib.util.spec_from_file_location("_celery_app_without_celery", celery_module.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Celery configuration -------------------------------------------------------------------------


def test_worker_imports_task_modules_explicitly():
    assert celery_module.CELERY_AVAILABLE is True
    assert list(celery_module.celery_app.conf.include) == ["loadlens_app.tasks"]
    assert celery_module.TASK_MODULES == ("loadlens_app.tasks",)


def test_report_tasks_are_routed_to_the_reports_queue():
    conf = celery_module.celery_app.conf
    assert conf.task_routes == {"loadlens.*": {"queue": "reports"}}
    assert conf.task_default_queue == "celery"

    router = celery_module.celery_app.amqp.router
    for name in ("loadlens.report", "loadlens.publish_confluence", "loadlens.publish_forecast"):
        assert router.route({}, name)["queue"].name == "reports"
    # Download and upload tasks (registered by update_page until it stops using Celery) keep the default queue.
    for name in ("update_page.download_grafana_image_task", "update_page.upload_attachment_task", "loadlens_app.tasks.helper"):
        assert router.route({}, name)["queue"].name == "celery"


def test_acknowledgement_prefetch_and_serialisation():
    conf = celery_module.celery_app.conf
    assert conf.task_acks_late is False
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_serializer == "json"
    assert conf.result_serializer == "json"
    assert list(conf.accept_content) == ["json"]
    # Download and upload limits stay the shared defaults; report tasks set their own.
    assert conf.task_soft_time_limit == 600
    assert conf.task_time_limit == 660


def test_publishing_gives_up_fast_when_the_broker_is_down():
    conf = celery_module.celery_app.conf
    assert conf.task_publish_retry_policy == {"max_retries": 1, "interval_start": 0, "interval_max": 0.5}
    assert conf.broker_connection_timeout == 2
    assert conf.broker_connection_retry_on_startup is True
    assert conf.broker_transport_options == {"socket_connect_timeout": 2, "socket_timeout": 5}

    # The options reach the (lazy, unconnected) broker connection apply_async publishes through.
    connection = celery_module.celery_app.connection_for_write()
    try:
        assert connection.connect_timeout == 2
        assert connection.transport_options["socket_connect_timeout"] == 2
        assert connection.transport_options["socket_timeout"] == 5
    finally:
        connection.release()


def test_eager_mode_is_the_default_and_compose_switches_it_off(monkeypatch, make_app):
    assert make_app().conf.task_always_eager is True
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "0")
    assert make_app().conf.task_always_eager is False
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "1")
    assert make_app().conf.task_always_eager is True


def test_broker_url_falls_back_to_local_redis(monkeypatch, make_app):
    monkeypatch.setenv("CELERY_BROKER_URL", "")
    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
    app = make_app()
    assert app.conf.broker_url == "redis://localhost:6379/0"
    assert app.conf.result_backend == "redis://localhost:6379/0"

    monkeypatch.setenv("CELERY_BROKER_URL", BROKER)
    app = make_app()
    assert app.conf.broker_url == BROKER
    assert app.conf.result_backend == BROKER


def test_report_task_names_follow_the_routing_prefix():
    assert celery_module.TASK_NAME_PREFIX == "loadlens."
    assert celery_module.REPORTS_QUEUE == "reports"
    assert celery_module.DEFAULT_QUEUE == "celery"
    assert f"{celery_module.TASK_NAME_PREFIX}*" in celery_module.celery_app.conf.task_routes


# --- job runner mode ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runner, broker, eager, expected",
    [
        (None, None, None, False),  # development without Redis: threads
        (None, BROKER, None, False),  # eager mode is on unless switched off
        (None, BROKER, "0", True),  # compose: broker set, eager off
        (None, BROKER, "1", False),
        (None, "", "0", False),  # an empty broker URL is not a broker
        (None, "   ", "0", False),
        (None, None, "0", False),
        ("auto", BROKER, "0", True),
        ("AUTO", BROKER, "1", False),
        ("", BROKER, "0", True),
        ("thread", BROKER, "0", False),
        (" Thread ", BROKER, "0", False),
        ("thread", None, None, False),
        ("celery", None, None, True),  # explicit celery wins over the broker and eager checks
        ("celery", BROKER, "0", True),
        ("celery", BROKER, "1", True),
        ("CELERY", None, "0", True),
        ("fork", BROKER, "0", True),  # unknown values behave as auto
        ("fork", BROKER, None, False),
    ],
)
def test_celery_mode_enabled(monkeypatch, runner, broker, eager, expected):
    for name, value in (("LOADLENS_JOB_RUNNER", runner), ("CELERY_BROKER_URL", broker), ("CELERY_TASK_ALWAYS_EAGER", eager)):
        if value is not None:
            monkeypatch.setenv(name, value)
    assert celery_module.celery_mode_enabled() is expected


def test_celery_mode_reads_the_environment_on_every_call(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", BROKER)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "0")
    assert celery_module.celery_mode_enabled() is True
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", "thread")
    assert celery_module.celery_mode_enabled() is False
    monkeypatch.delenv("LOADLENS_JOB_RUNNER")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "1")
    assert celery_module.celery_mode_enabled() is False


@pytest.mark.parametrize(
    "raw, expected",
    [(None, "auto"), ("", "auto"), ("auto", "auto"), ("thread", "thread"), ("celery", "celery"), (" Celery\n", "celery"), ("threads", "auto")],
)
def test_job_runner_setting_normalises_the_value(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("LOADLENS_JOB_RUNNER", raw)
    assert celery_module.job_runner_setting() == expected


def test_unknown_runner_and_eager_celery_are_logged_once(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=celery_module.logger.name)
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", "threads")
    for _ in range(3):
        celery_module.job_runner_setting()
    assert sum("is not one of auto, thread, celery" in r.getMessage() for r in caplog.records) == 1

    caplog.clear()
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", "celery")
    for _ in range(3):
        assert celery_module.celery_mode_enabled() is True
    assert sum("run inline in the caller" in r.getMessage() for r in caplog.records) == 1


def test_without_the_celery_package_jobs_use_threads(monkeypatch):
    monkeypatch.setattr(celery_module, "CELERY_AVAILABLE", False)
    monkeypatch.setenv("CELERY_BROKER_URL", BROKER)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "0")
    for runner in ("auto", "celery", "thread"):
        monkeypatch.setenv("LOADLENS_JOB_RUNNER", runner)
        assert celery_module.celery_mode_enabled() is False


# --- report task time limits ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [(None, (3600, 3660)), ("", (3600, 3660)), ("120", (120, 180)), (" 7200 ", (7200, 7260)), ("0", (3600, 3660)), ("-5", (3600, 3660)), ("1h", (3600, 3660))],
)
def test_report_task_time_limits(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("LOADLENS_REPORT_TASK_LIMIT_SEC", raw)
    assert celery_module.report_task_time_limits() == expected


# --- fallback app without the celery package ------------------------------------------------------


def test_fallback_app_runs_tasks_synchronously(monkeypatch):
    module = _load_without_celery(monkeypatch)
    assert module.CELERY_AVAILABLE is False
    app = module.celery_app
    assert list(app.conf.include) == ["loadlens_app.tasks"]
    assert app.conf.task_routes == {"loadlens.*": {"queue": "reports"}}
    assert app.conf["task_acks_late"] is False

    calls = []

    @app.task(name="loadlens.report", bind=True, ignore_result=True)
    def report(self, job_id, *, service=""):
        calls.append((self.name, job_id, service))
        return f"{job_id}:{service}"

    @app.task
    def plain(value):
        return value * 2

    assert report.name == "loadlens.report"
    assert app.tasks["loadlens.report"] is report
    assert report.apply_async(kwargs={"job_id": "j1", "service": "svc"}, queue="reports").get() == "j1:svc"
    assert report.apply_async(("j2",)).get(timeout=1) == "j2:"
    assert report.delay("j3", service="x").get() == "j3:x"
    assert report.run(job_id="j4") == "j4:"
    assert report("j5") == "j5:"
    assert calls == [("loadlens.report", f"j{n}", s) for n, s in ((1, "svc"), (2, ""), (3, "x"), (4, ""), (5, ""))]
    assert plain.delay(21).get() == 42
    assert plain.__name__ == "plain"


def test_fallback_app_propagates_task_errors(monkeypatch):
    app = _load_without_celery(monkeypatch).celery_app

    @app.task(bind=True)
    def broken(self):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        broken.apply_async()


def test_fallback_mode_is_thread_even_when_celery_is_requested(monkeypatch):
    module = _load_without_celery(monkeypatch)
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", "celery")
    assert module.celery_mode_enabled() is False


# --- docker compose and env template --------------------------------------------------------------


def test_worker_command_consumes_reports_and_default_queues(compose):
    command = compose["services"]["celery_worker"]["command"]
    assert isinstance(command, str)
    assert command.startswith("celery -A loadlens_app.celery_app worker ")
    assert "-Q reports,celery" in command
    queues = set(re.search(r"-Q (\S+)", command).group(1).split(","))
    assert queues == {celery_module.REPORTS_QUEUE, celery_module.DEFAULT_QUEUE}
    assert "--concurrency ${WORKER_CONCURRENCY:-2}" in command
    assert "--max-tasks-per-child 20" in command


def test_worker_memory_and_healthcheck(compose):
    worker = compose["services"]["celery_worker"]
    assert worker["deploy"]["resources"]["limits"]["memory"] == "${WORKER_MEMORY_LIMIT:-2g}"
    test = worker["healthcheck"]["test"]
    assert test[0] == "CMD-SHELL"
    # $$ in compose is a literal $, expanded by the container shell to the worker's node name.
    assert "celery -A loadlens_app.celery_app inspect ping -d celery@$$HOSTNAME" in test[1]


@pytest.mark.parametrize("service", ["app", "celery_worker"])
def test_app_and_worker_receive_the_job_variables(compose, service):
    environment = compose["services"][service]["environment"]
    for name, default in NEW_JOB_VARS.items():
        assert environment[name] == f"${{{name}:-{default}}}"
    assert environment["CELERY_TASK_ALWAYS_EAGER"] == "${CELERY_TASK_ALWAYS_EAGER:-0}"


def test_worker_reads_the_configuration_read_only(compose):
    mounts = [m for m in compose["services"]["celery_worker"]["volumes"] if isinstance(m, dict)]
    targets = {m["target"]: m for m in mounts}
    for target in ("/app/settings.py", "/app/metrics_config.py", "/app/settings_runtime.json", "/app/metrics_config_runtime.json"):
        assert targets[target]["read_only"] is True


def test_env_example_documents_the_job_variables():
    values = _env_example()
    for name, default in NEW_JOB_VARS.items():
        assert values[name] == default
    assert values["WORKER_CONCURRENCY"] == "2"
    assert values["WORKER_MEMORY_LIMIT"] == "2g"
    assert values["CELERY_TASK_ALWAYS_EAGER"] == "0"
    assert int(values["LOADLENS_REPORT_TASK_LIMIT_SEC"]) == celery_module.DEFAULT_REPORT_TASK_LIMIT_SEC


def test_every_compose_variable_of_the_worker_is_in_env_example(compose):
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    block = text.split("  celery_worker:", 1)[1].split("\nvolumes:", 1)[0]
    used = set(re.findall(r"\$\{([A-Z_]+)", block))
    documented = set(_env_example()) | {"CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"}  # commented-out examples
    assert used - documented == set()
