# SPDX-License-Identifier: Apache-2.0
"""Celery tasks of the background jobs: building a report, publishing it to Confluence, publishing a forecast.

:func:`loadlens_app.job_runner.submit` publishes them to the ``reports`` queue, or runs them in a thread of
the web process (``task.run(job_id=..., **kwargs)``). Every parameter besides ``job_id`` is JSON-safe; the
link to LoadLens (``source_url``) is built by the route, which has the request.

The work functions are looked up on the blueprint modules when a task runs (``dashboard.update_report``,
``dashboard.publish_report``, ``forecast.publish_forecast``): that is where the routes see them and where
tests replace them.

Order of calls in a task delivered by Celery:

1. in a worker process (not Celery's eager mode, which runs in the caller): ``update_page.refresh_runtime_config()``,
   since the worker otherwise keeps the settings of its start;
2. claim the job: a job already finished, swept, started by another delivery or taken over by the web
   process is not run again; then ``JobStore.mark_started`` (False: do not run);
3. a heartbeat thread calls ``JobStore.heartbeat`` every 30 s until the task ends;
4. the work, with progress through ``JobStore.update``; ``done`` with the result fields, or ``error`` with
   the failed stage (``_job_failure_message`` / ``_failure_message`` of the blueprint);
   ``SoftTimeLimitExceeded`` gives an error naming the time limit.

In a thread of the web process only step 4 runs: the job was created there as a running thread job, its
settings are current, and thread jobs are not swept.

Step 4 runs in the language of the job (``with_locale``): ``report_language`` of a report job, the installation
setting for a publication (the page itself is written in the language of the stored report). Messages are
:class:`AI.i18n.Msg`: the row keeps the text and its ref, and the job list renders the ref in each reader's
language (``loadlens_app.jobs``).
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from AI.i18n import LocalizedError, Msg, msg, t, with_locale
from loadlens_app import jobs
from loadlens_app.celery_app import celery_app, report_task_time_limits
from loadlens_app.jobs import JOB_STATUS_DONE, JOB_STATUS_ERROR, JOB_STATUS_RUNNING

try:
    from celery.exceptions import SoftTimeLimitExceeded
except ModuleNotFoundError:  # pragma: no cover - the fallback app runs tasks in threads, without time limits

    class SoftTimeLimitExceeded(Exception):
        """Stand-in for Celery's exception when the celery package is missing."""


logger = logging.getLogger(__name__)

#: Seconds between two heartbeats of a job running under Celery (JobStore.sweep_stale waits minutes).
HEARTBEAT_INTERVAL_SEC = 30.0
#: Pauses before the final status of a job is written again when the job table did not take it (Celery only:
#: a thread job keeps its state in memory, a worker job would otherwise end as "the worker does not answer").
FINAL_UPDATE_RETRY_DELAYS_SEC = (1.0, 3.0)

# Message key of a job ended by the time limit; TIME_LIMIT_MESSAGE gives its text (quoted by docs/jobs-runner.md).
TIME_LIMIT_KEY = "job.time_limit"
TIME_LIMIT_ERROR = "The task exceeded its time limit of {seconds} s (LOADLENS_REPORT_TASK_LIMIT_SEC)"

SOFT_TIME_LIMIT_SEC, HARD_TIME_LIMIT_SEC = report_task_time_limits()
_TASK_OPTIONS = dict(bind=True, ignore_result=True, soft_time_limit=SOFT_TIME_LIMIT_SEC, time_limit=HARD_TIME_LIMIT_SEC)

Progress = Callable[..., None]


def _dashboard():
    return importlib.import_module("loadlens_app.blueprints.dashboard")


def _forecast_blueprint():
    return importlib.import_module("loadlens_app.blueprints.forecast")


# ---- the tasks ----------------------------------------------------------------------------------------------


@celery_app.task(name="loadlens.report", **_TASK_OPTIONS)
def report_task(
    self,
    job_id: str,
    *,
    start: int,
    end: int,
    service: str,
    project_area: str,
    use_llm: bool,
    save_to_db: bool,
    web_only: bool,
    run_name: str,
    test_type: str,
    run_tag: Optional[str] = None,
    run_filter_packs: Optional[List[str]] = None,
    report_language: str = "ru",
) -> None:
    """Builds a report (``update_page.update_report``) for the job started by ``dashboard.start_report_job``.

    ``report_language`` is fixed when the job is queued (the worker has its own settings snapshot); messages queued
    by an older web process have no such argument and build a Russian report, as before.
    """
    dashboard = _dashboard()

    def work(progress: Progress) -> Dict[str, Any]:
        res = dashboard.update_report(
            start, end, service,
            use_llm=use_llm,
            save_to_db=save_to_db,
            web_only=web_only,
            run_name=run_name,
            test_type=test_type,
            project_area=project_area,
            progress_callback=progress,
            run_tag=run_tag,
            run_filter_packs=run_filter_packs,
            report_language=report_language,
        )
        result = res if isinstance(res, dict) else {}
        run_id = str(result.get("run_id") or "")
        return dict(
            status=JOB_STATUS_DONE,
            progress=100,
            message=msg("job.done"),
            report_url=result.get("page_url"),
            page_url=f"/reports/{run_id}" if run_id else None,
            page_id=str(result["page_id"]) if result.get("page_id") is not None else None,
            run_name=str(result.get("run_name") or run_name),
        )

    _run_job(self, job_id, work, dashboard._job_failure_message, locale=report_language)


@celery_app.task(name="loadlens.publish_confluence", **_TASK_OPTIONS)
def publish_confluence_task(self, job_id: str, *, run_name: str, service: str, source_url: str) -> None:
    """Publishes a ready web report as a Confluence page (``confluence_export.publish_report``)."""
    dashboard = _dashboard()

    def work(progress: Progress) -> Dict[str, Any]:
        result = dashboard.publish_report(run_name=run_name, service=service, source_url=source_url, progress_callback=progress)
        return dict(
            status=JOB_STATUS_DONE,
            progress=100,
            message=msg("job.done"),
            page_url=result.get("page_url"),
            page_id=str(result["page_id"]) if result.get("page_id") is not None else None,
        )

    _run_job(self, job_id, work, dashboard._job_failure_message)


@celery_app.task(name="loadlens.publish_forecast", **_TASK_OPTIONS)
def publish_forecast_task(
    self,
    job_id: str,
    *,
    run_id: str,
    target_rps: Optional[float],
    headroom_pct: float,
    cpu_ceiling_pct: Optional[float],
    source_url: str,
) -> None:
    """Publishes a capacity forecast as its own Confluence page (``forecast_confluence.publish_forecast``)."""
    forecast = _forecast_blueprint()

    def work(progress: Progress) -> Dict[str, Any]:
        result = forecast.publish_forecast(run_id, target_rps, headroom_pct, cpu_ceiling_pct, source_url, progress)
        return dict(status=JOB_STATUS_DONE, progress=100, message=msg("job.done"), page_url=result["page_url"], page_id=result["page_id"])

    _run_job(self, job_id, work, forecast._failure_message)


# ---- running a job ------------------------------------------------------------------------------------------


def _run_job(
    task: Any,
    job_id: str,
    work: Callable[[Progress], Dict[str, Any]],
    failure_message: Callable[[str], Any],
    *,
    locale: Optional[str] = None,
) -> None:
    """Runs ``work`` for the job and records its outcome; see the module docstring for the order of calls.

    ``locale`` is the language of the job (``None``: the installation setting, read after the settings refresh);
    the work, its messages and the outcome are rendered in it.
    """
    store = jobs.job_store
    request = task.request
    via_celery = not getattr(request, "called_directly", True)
    if via_celery:
        if not getattr(request, "is_eager", False):
            _refresh_runtime_config()
        if not _claim(store, job_id, getattr(task, "name", "")):
            return
    stop_heartbeat = _start_heartbeat(store, job_id) if via_celery else None

    def progress(message: Any, pct: Optional[int] = None) -> None:
        # A Msg (the stages of the pipeline) keeps its ref next to the text; plain text clears the previous ref.
        if isinstance(message, (Msg, LocalizedError)):
            fields = jobs.message_fields(message)
        else:
            fields = {"message": str(message)}
        store.update(job_id, **fields, progress=pct if isinstance(pct, int) else None)

    try:
        with with_locale(locale if locale is not None else jobs.stored_locale()):
            try:
                outcome = work(progress)
            except SoftTimeLimitExceeded:
                logger.warning("Job %s (%s) exceeded its time limit of %s s", job_id, getattr(task, "name", ""), SOFT_TIME_LIMIT_SEC)
                outcome = dict(
                    status=JOB_STATUS_ERROR,
                    **jobs.message_fields(msg(TIME_LIMIT_KEY)),
                    error=TIME_LIMIT_ERROR.format(seconds=SOFT_TIME_LIMIT_SEC),
                )
            except Exception as exc:
                logger.warning("Job %s (%s) failed: %s", job_id, getattr(task, "name", ""), exc, exc_info=True)
                outcome = dict(status=JOB_STATUS_ERROR, **jobs.message_fields(failure_message(job_id)), error=str(exc))
            _finish(store, job_id, outcome, retry=via_celery)
    finally:
        if stop_heartbeat is not None:
            stop_heartbeat.set()


def _refresh_runtime_config() -> None:
    """Applies settings_runtime.json in a worker process; on failure the task goes on with the settings it has."""
    import update_page

    try:
        update_page.refresh_runtime_config()
    except Exception:
        logger.warning("Could not re-read the runtime settings; the job uses the settings the worker started with", exc_info=True)


def _claim(store: Any, job_id: str, task_name: str) -> bool:
    """Whether this delivery of a Celery task may run the job; marks it started when it may.

    The row is read with an UPDATE ... RETURNING (``JobStore.update`` without fields) rather than ``get``:
    ``get`` of a running thread row runs the restart recovery of thread jobs, which a worker must never do.
    A row that is not running, or that was started already (by another delivery of the same message, or
    taken over by the web process after a failed send: ``adopt_thread`` sets ``started_at``), is left alone.
    Without the job table (``None``) the job runs: there is nothing to check against.
    """
    row = store.update(job_id)
    if row is not None and row.status != JOB_STATUS_RUNNING:
        logger.warning("Not running job %s (%s): its status is already %s", job_id, task_name, row.status)
        return False
    if row is not None and row.started_at is not None:
        logger.warning(
            "Not running job %s (%s): it was started at %s (runner %s), by another delivery of the task or by"
            " the web process after a failed send",
            job_id, task_name, row.started_at, row.runner,
        )
        return False
    if store.mark_started(job_id) is False:
        logger.warning("Not running job %s (%s): it is no longer running (finished, swept from the queue or unknown)", job_id, task_name)
        return False
    return True


def _start_heartbeat(store: Any, job_id: str) -> threading.Event:
    """Starts a daemon thread calling ``store.heartbeat(job_id)`` every HEARTBEAT_INTERVAL_SEC; set the
    returned event to stop it. It also stops by itself once the row is no longer running."""
    stop = threading.Event()
    interval = float(HEARTBEAT_INTERVAL_SEC)

    def beat() -> None:
        while not stop.wait(interval):
            try:
                alive = store.heartbeat(job_id)
            except Exception:
                logger.warning("Heartbeat of job %s failed", job_id, exc_info=True)
                continue
            if alive is False:
                return

    threading.Thread(target=beat, daemon=True).start()
    return stop


def _finish(store: Any, job_id: str, outcome: Dict[str, Any], *, retry: bool) -> None:
    """Writes the final state; under Celery it is tried again when the job table did not answer."""
    delays = FINAL_UPDATE_RETRY_DELAYS_SEC if retry else ()
    for delay in (*delays, None):
        if store.update(job_id, **outcome) is not None or delay is None:
            return
        time.sleep(delay)


def __getattr__(name: str) -> str:
    """``TIME_LIMIT_MESSAGE``: the text of :data:`TIME_LIMIT_KEY` in the installation language (docs quote it)."""
    if name == "TIME_LIMIT_MESSAGE":
        return t(TIME_LIMIT_KEY, locale=jobs.stored_locale())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HEARTBEAT_INTERVAL_SEC",
    "SoftTimeLimitExceeded",
    "publish_confluence_task",
    "publish_forecast_task",
    "report_task",
]
