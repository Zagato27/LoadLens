# SPDX-License-Identifier: Apache-2.0
"""Celery application shared by the web process and the worker.

The web process only publishes tasks. The modules listed in ``TASK_MODULES`` are imported by a worker at
start (``include``); the web process imports them only where it submits a task. Task names under
``TASK_NAME_PREFIX`` go to the ``reports`` queue, everything else to the default ``celery`` queue, so the
worker runs with ``-Q reports,celery``.

``celery_mode_enabled()`` decides whether reports and publications go to the worker or stay on a thread of
the web process; ``report_task_time_limits()`` gives the time limits of those tasks.
"""

from __future__ import annotations

import functools
import logging
import os
from types import SimpleNamespace

logger = logging.getLogger(__name__)

try:
    from celery import Celery

    CELERY_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - tests/test_perf_celery_infra.py loads this branch with celery hidden
    CELERY_AVAILABLE = False

    class _SyncAsyncResult:
        """Result of a task the fallback app has already run in the calling thread."""

        def __init__(self, value):
            self._value = value

        def get(self, timeout=None):
            return self._value

    class _SyncTask:
        """Task of the fallback app: ``delay`` and ``apply_async`` run it at once in the calling thread."""

        def __init__(self, func, *, name: str, bind: bool, options: dict):
            functools.update_wrapper(self, func)
            self._func = func
            self.name = name
            self.bind = bind
            self.options = options
            self.request = SimpleNamespace(id=None, retries=0, called_directly=True)

        def run(self, *args, **kwargs):
            if self.bind:
                return self._func(self, *args, **kwargs)
            return self._func(*args, **kwargs)

        __call__ = run

        def delay(self, *args, **kwargs):
            return self.apply_async(args, kwargs)

        def apply_async(self, args=None, kwargs=None, **options):
            return _SyncAsyncResult(self.run(*(args or ()), **(kwargs or {})))

    class _SyncConf(dict):
        """Settings of the fallback app, readable as attributes like Celery's ``app.conf``."""

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name) from None

    class Celery:
        """Stand-in for environments without the celery package: tasks run synchronously."""

        def __init__(self, main=None, *, broker=None, backend=None, include=None, **kwargs):
            self.main = main
            self.conf = _SyncConf(broker_url=broker, result_backend=backend, include=list(include or ()))
            self.tasks: dict = {}

        def task(self, *task_args, **task_kwargs):
            def decorator(func):
                name = task_kwargs.get("name") or f"{func.__module__}.{func.__name__}"
                task = _SyncTask(func, name=name, bind=bool(task_kwargs.get("bind")), options=dict(task_kwargs))
                self.tasks[name] = task
                return task

            if len(task_args) == 1 and callable(task_args[0]) and not task_kwargs:
                return decorator(task_args[0])
            return decorator


#: Queue of the report and publication tasks; the worker consumes it next to DEFAULT_QUEUE.
REPORTS_QUEUE = "reports"
#: Celery's default queue for any task outside TASK_NAME_PREFIX (reports download and upload in their own threads).
DEFAULT_QUEUE = "celery"
#: Task names starting with this prefix are routed to REPORTS_QUEUE.
TASK_NAME_PREFIX = "loadlens."
#: Modules with task definitions a worker imports at start.
TASK_MODULES = ("loadlens_app.tasks",)
#: Values of LOADLENS_JOB_RUNNER.
JOB_RUNNER_MODES = ("auto", "thread", "celery")

#: Broker used when CELERY_BROKER_URL is empty (local development). It does not count as "set".
LOCAL_BROKER_URL = "redis://localhost:6379/0"
#: Seconds to open a broker connection and to wait for an answer of Redis: a broker that is down or hangs
#: fails apply_async within seconds instead of holding a web request.
BROKER_CONNECT_TIMEOUT_SEC = 2
BROKER_SOCKET_TIMEOUT_SEC = 5
#: One retry, at once: the caller falls back to a thread instead of waiting for the broker.
PUBLISH_RETRY_POLICY = {"max_retries": 1, "interval_start": 0, "interval_max": 0.5}

#: Soft time limit of a report or publication task without LOADLENS_REPORT_TASK_LIMIT_SEC, seconds.
DEFAULT_REPORT_TASK_LIMIT_SEC = 3600
#: The hard limit (the worker process is killed) comes this many seconds after the soft one.
REPORT_TASK_HARD_LIMIT_GRACE_SEC = 60

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _broker_url_setting() -> str:
    """CELERY_BROKER_URL as configured; empty when it is not set."""
    return os.getenv("CELERY_BROKER_URL", "").strip()


def _eager_enabled() -> bool:
    """CELERY_TASK_ALWAYS_EAGER, ``1`` unless set: without a broker tasks run in the calling thread."""
    return os.getenv("CELERY_TASK_ALWAYS_EAGER", "1").strip() == "1"


def job_runner_setting() -> str:
    """LOADLENS_JOB_RUNNER as ``auto``, ``thread`` or ``celery``; unset, empty or unknown values mean ``auto``."""
    raw = os.getenv("LOADLENS_JOB_RUNNER", "")
    value = raw.strip().lower() or "auto"
    if value not in JOB_RUNNER_MODES:
        _warn_once(f"runner:{value}", "LOADLENS_JOB_RUNNER=%r is not one of auto, thread, celery; using auto", raw)
        return "auto"
    return value


def celery_mode_enabled() -> bool:
    """True when reports and publications go to the Celery worker, False when they run on a thread.

    - ``LOADLENS_JOB_RUNNER=thread``: False.
    - ``LOADLENS_JOB_RUNNER=celery``: True. With ``CELERY_TASK_ALWAYS_EAGER=1`` the task then runs inline
      in the caller (Celery's eager mode), which is logged once.
    - ``auto`` (default): True only when ``CELERY_BROKER_URL`` is set and ``CELERY_TASK_ALWAYS_EAGER`` is
      not ``1``. Eager mode is the default, so tests and development without Redis use threads.

    Without the celery package there is no worker to send to: always False. The environment is read on
    every call.
    """
    setting = job_runner_setting()
    if setting == "thread":
        return False
    if not CELERY_AVAILABLE:
        if setting == "celery":
            _warn_once("runner:no-celery", "LOADLENS_JOB_RUNNER=celery, but the celery package is not installed; using threads")
        return False
    if setting == "celery":
        if _eager_enabled():
            _warn_once("runner:eager", "LOADLENS_JOB_RUNNER=celery with CELERY_TASK_ALWAYS_EAGER=1: tasks run inline in the caller")
        return True
    return bool(_broker_url_setting()) and not _eager_enabled()


def report_task_time_limits() -> tuple[int, int]:
    """``(soft, hard)`` time limits in seconds of the report and publication tasks.

    The soft limit is LOADLENS_REPORT_TASK_LIMIT_SEC (default 3600): the task gets SoftTimeLimitExceeded
    and can mark its job as failed. The hard limit, 60 s later, kills the worker process. A value that is
    not a positive integer falls back to the default. The prefork pool enforces both; ``--pool=solo`` on
    Windows enforces neither.
    """
    raw = os.getenv("LOADLENS_REPORT_TASK_LIMIT_SEC", "").strip()
    soft = DEFAULT_REPORT_TASK_LIMIT_SEC
    if raw:
        try:
            soft = int(raw)
        except ValueError:
            soft = 0
        if soft <= 0:
            _warn_once(f"limit:{raw}", "LOADLENS_REPORT_TASK_LIMIT_SEC=%r is not a positive integer; using %s", raw, DEFAULT_REPORT_TASK_LIMIT_SEC)
            soft = DEFAULT_REPORT_TASK_LIMIT_SEC
    return soft, soft + REPORT_TASK_HARD_LIMIT_GRACE_SEC


def _make_celery() -> Celery:
    broker_url = _broker_url_setting() or LOCAL_BROKER_URL
    backend_url = os.getenv("CELERY_RESULT_BACKEND", "").strip() or broker_url
    app = Celery("loadlens_app", broker=broker_url, backend=backend_url, include=list(TASK_MODULES))
    app.conf.update(
        task_track_started=True,
        task_always_eager=_eager_enabled(),
        task_eager_propagates=True,
        result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),
        # Only JSON on the wire: a message that reaches Redis can never become a pickle to execute.
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Default limits of any task without its own; report and publication tasks set theirs
        # (report_task_time_limits()).
        task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "600")),
        task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "660")),
        task_default_queue=DEFAULT_QUEUE,
        task_routes={f"{TASK_NAME_PREFIX}*": {"queue": REPORTS_QUEUE}},
        # A task is acknowledged when it starts: a report is never replayed after a worker crash, the
        # job sweeper marks it as failed instead.
        task_acks_late=False,
        worker_prefetch_multiplier=1,
        broker_connection_timeout=BROKER_CONNECT_TIMEOUT_SEC,
        broker_connection_retry_on_startup=True,
        # The Redis transport ignores broker_connection_timeout; without socket timeouts a broker that
        # accepts the connection and never answers blocks apply_async for ever.
        broker_transport_options={
            "socket_connect_timeout": BROKER_CONNECT_TIMEOUT_SEC,
            "socket_timeout": BROKER_SOCKET_TIMEOUT_SEC,
        },
        task_publish_retry_policy=dict(PUBLISH_RETRY_POLICY),
    )
    return app


celery_app = _make_celery()

__all__ = [
    "CELERY_AVAILABLE",
    "DEFAULT_QUEUE",
    "REPORTS_QUEUE",
    "TASK_NAME_PREFIX",
    "celery_app",
    "celery_mode_enabled",
    "job_runner_setting",
    "report_task_time_limits",
]
