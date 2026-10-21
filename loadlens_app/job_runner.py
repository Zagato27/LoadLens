# SPDX-License-Identifier: Apache-2.0
"""Where a report or publication job runs: a thread of the web process or the Celery worker.

The routes register a job with :func:`create_job` and start it with :func:`submit`; the tasks themselves
live in :mod:`loadlens_app.tasks`.

- :func:`mode` names the runner for new jobs: ``celery`` when ``celery_app.celery_mode_enabled()`` (see
  ``LOADLENS_JOB_RUNNER``), otherwise ``thread``, as before the worker took reports.
- :func:`create_job` writes the job for that runner. A Celery job exists only as a row of the job table;
  when that row cannot be written nobody could follow the job, so it becomes a thread job of this process.
- :func:`submit` publishes the task to the ``reports`` queue, or starts it with
  ``threading.Thread(target=..., daemon=True)``. When the broker cannot be reached the job is taken over
  by this process (``JobStore.adopt_thread``) and runs in a thread: a report is never lost to a broker outage.

The job store is looked up as ``jobs.job_store`` on every call, so tests replacing it are honoured.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Union

from AI.i18n import t as translate
from loadlens_app import celery_app as celery_module
from loadlens_app import jobs
from loadlens_app.jobs import JOB_RUNNER_CELERY, JOB_RUNNER_THREAD, JOB_STATUS_ERROR, JobRecord

logger = logging.getLogger(__name__)


def _broker_errors() -> tuple[type[BaseException], ...]:
    """Exceptions of apply_async meaning "the broker cannot take the task": everything from kombu (its
    OperationalError after the one publish retry), plus socket and Redis errors a transport may let through."""
    errors: list[type[BaseException]] = [OSError]
    try:
        from kombu.exceptions import KombuError

        errors.append(KombuError)
    except ModuleNotFoundError:  # pragma: no cover - no celery, so no broker either
        pass
    try:
        from redis.exceptions import RedisError

        errors.append(RedisError)
    except ModuleNotFoundError:  # pragma: no cover - another transport
        pass
    return tuple(errors)


BROKER_ERRORS = _broker_errors()

#: Message key of a Celery row left behind when its job was registered again as a thread job (see create_job);
#: REPLACED_MESSAGE gives its text.
REPLACED_KEY = "job.replaced"
REPLACED_ERROR = "The job row could not be read back after it was written; the job runs as {job_id} in the web process"


def mode() -> str:
    """``"celery"`` when new jobs go to the Celery worker, ``"thread"`` when they run in this process.

    Follows ``celery_app.celery_mode_enabled()``, which reads the environment on every call.
    """
    return JOB_RUNNER_CELERY if celery_module.celery_mode_enabled() else JOB_RUNNER_THREAD


def create_job(kind: str, *, run_name: str = "", service: str = "", message: Any = "", owner: str = "") -> JobRecord:
    """Registers a running job for the runner :func:`mode` picks and returns its record.

    ``message`` may be an :class:`AI.i18n.Msg`: the job keeps its ref (see ``loadlens_app.jobs``).

    A Celery job is only a row of the job table, and ``JobStore.create`` logs instead of raising when that
    write fails. So the row is read back (one indexed query); when it is not there the job is registered
    again as a thread job, which this process also keeps in memory, and its progress stays visible.
    """
    store = jobs.job_store
    runner = mode()
    job = store.create(kind, run_name=run_name, service=service, message=message, owner=owner, runner=runner)
    if runner != JOB_RUNNER_CELERY or store.get(job.job_id) is not None:
        return job
    replacement = store.create(kind, run_name=run_name, service=service, message=message, owner=owner, runner=JOB_RUNNER_THREAD)
    logger.warning(
        "Job %s (%s) is not in the job table after it was written; running it as thread job %s of the web"
        " process instead of the Celery worker",
        job.job_id, kind, replacement.job_id,
    )
    # Should the row exist after all (only reading it failed), it must not hold a job slot while it waits
    # for a worker that never gets the task. Fails quietly when the table is unreachable.
    store.update(
        job.job_id, status=JOB_STATUS_ERROR, message=jobs.job_message(REPLACED_KEY),
        error=REPLACED_ERROR.format(job_id=replacement.job_id),
    )
    return replacement


def submit(task: Any, job: Union[JobRecord, str], **kwargs: Any) -> str:
    """Starts ``task`` for the job and returns the runner that took it, ``"celery"`` or ``"thread"``.

    ``job`` is the record from :func:`create_job`, whose ``runner`` decides, or a job id, for which
    :func:`mode` decides. ``kwargs`` are the task's parameters besides ``job_id``; they must be JSON-safe.

    - ``celery``: ``task.apply_async(kwargs=..., queue="reports", task_id=job_id)``. When that fails with a
      broker error (:data:`BROKER_ERRORS`; the Celery settings give up within seconds) a warning is logged,
      the job is taken over by this process with ``JobStore.adopt_thread`` and started in a thread, unless a
      worker has already started it (the broker took the message but its answer was lost).
    - ``thread``: ``threading.Thread(target=..., daemon=True).start()`` running ``task.run(job_id=..., **kwargs)``.
    """
    if isinstance(job, JobRecord):
        job_id, runner = job.job_id, job.runner
    else:
        job_id, runner = str(job), mode()
    if runner == JOB_RUNNER_CELERY:
        try:
            task.apply_async(kwargs={"job_id": job_id, **kwargs}, queue=celery_module.REPORTS_QUEUE, task_id=job_id)
            return JOB_RUNNER_CELERY
        except BROKER_ERRORS as exc:
            logger.warning(
                "Could not send job %s (%s) to the Celery broker: %s: %s; running it in a thread of the web process",
                job_id, getattr(task, "name", task), type(exc).__name__, exc,
            )
            if not _take_over(job_id):
                return JOB_RUNNER_CELERY
    # A new thread starts with an empty context (no request, no locale): the task opens the job's own locale.
    threading.Thread(target=lambda: task.run(job_id=job_id, **kwargs), daemon=True).start()
    return JOB_RUNNER_THREAD


def _take_over(job_id: str) -> bool:
    """Makes a job that could not be sent to the broker a thread job of this process; False when a worker
    has started it meanwhile and it must not run twice."""
    store = jobs.job_store
    current = store.get(job_id)
    if current is not None and current.runner == JOB_RUNNER_CELERY and current.started_at is not None:
        logger.warning("Job %s was started by a Celery worker although sending it failed; not running it here", job_id)
        return False
    if store.adopt_thread(job_id) is None:
        logger.warning("Job %s is not in the job table; it runs in a thread, but its progress is not visible", job_id)
    return True


def __getattr__(name: str) -> str:
    """``REPLACED_MESSAGE``: the text of :data:`REPLACED_KEY` in the installation language."""
    if name == "REPLACED_MESSAGE":
        return translate(REPLACED_KEY, locale=jobs.stored_locale())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BROKER_ERRORS", "create_job", "mode", "submit"]
