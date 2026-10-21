# SPDX-License-Identifier: Apache-2.0
"""Background loop that starts, watches and stops instances of active runs and then starts their reports.

Every few seconds the watcher claims the active runs (its own, unowned ones and those whose watcher
stopped heartbeating), advances each one and writes it back. All remote work happens here, so a request
handler only validates input and records intent (a new run, a stop request).

Instance lifecycle: pending -> starting -> running -> finished / checks_failed / failed / stopped /
timed_out / lost (cancelled when stopped before its start). A run is running while any instance is
active; then it waits for the metrics to reach the storage (draining), starts the report (reporting)
and ends as done. A run none of whose instances started ends as failed or cancelled.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from AI.i18n import is_ref
from loadlens_app.limits import LimitError, check_job_slot, step_seconds

from .executors import STORED_LOG_BYTES, Executor, GeneratorError, Probe, executor_for
from .generators import Generator
from .live import report_filter, rps_bounds
from .messages import error_param, joined, msg, ref_of
from .service import Catalog, instance_plan, load_catalog
from .store import (
    INST_ACTIVE,
    INST_CANCELLED,
    INST_CHECKS_FAILED,
    INST_FAILED,
    INST_FINISHED,
    INST_LOST,
    INST_PENDING,
    INST_RUNNING,
    INST_STARTING,
    INST_STOPPED,
    INST_TERMINAL,
    INST_TIMED_OUT,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_DRAINING,
    RUN_FAILED,
    RUN_REPORTING,
    RUN_RUNNING,
    RUN_STARTING,
    RUN_STOPPING,
    InstanceRecord,
    LoadTestStore,
    RunRecord,
    get_store,
)
from .tools import MASK, ToolConfigError, ToolInputError, ToolTemplate

logger = logging.getLogger(__name__)

TICK_SEC = 3.0
# A pass that found no run to handle is followed by this longer pause, so an idle watcher does not query the
# storage every few seconds; a new run, a stop request or a page that sees an active run wakes it (wake()).
IDLE_TICK_SEC = 15.0
# A run whose watcher has not renewed its claim for this long is taken over: well above the longest pass,
# which slow generators (SSH commands time out after 30 s) can stretch to minutes.
STALE_AFTER_SEC = 120
# During a pass the claim of a run is renewed before an instance when it is older than this.
HEARTBEAT_EVERY_SEC = 10
DEAD_PROBES_TO_LOST = 2
# A generator that does not answer for longer than the run may last plus this is given up.
UNREACHABLE_GRACE_SEC = 600
SLOW_START_SEC = 120
DRAIN_MIN_SEC = 30
DRAIN_MAX_SEC = 300
CLOCK_SKEW_NOTE_SEC = 30
TIMEOUT_EXIT_CODE = 124
# A failed instance that ran shorter than this never produced load worth a report (bad flag, missing file).
MIN_FAILED_RUN_SEC = 60
ERROR_LOG_INTERVAL_SEC = 60
DEFAULT_OWNER = "load-tests"
# A waiting state the executor calls stuck (image cannot be pulled) fails the instance after this.
STUCK_FAIL_SEC = 180
# How often the tool status command (k6 status) is asked while an instance runs.
STATUS_EVERY_SEC = 15
# Without signals (Kubernetes) the platform kills after the grace period itself: give it time first.
PLATFORM_KILL_EXTRA_SEC = 30
# Generators with their own run timer (SSH timeout, Job deadline, CI) get this long past the run limit
# before the watcher stops the instance itself (a lost CI event, a timer that did not fire).
DEADLINE_SLACK_SEC = 120
# After the last stop step the instance must end within this; otherwise LoadLens closes its window and says
# the process may need a manual stop, so the run, its slots and its report do not hang forever.
STOP_CONFIRM_SEC = 120
# Messages of a stop just sent: (requested by a person, forced by the run duration limit).
_STOP_KEYS = {
    "unsupported": ("lt.watcher.stop.unsupported", "lt.watcher.stop.unsupported_limit"),
    "soft": ("lt.watcher.stop.soft", "lt.watcher.stop.soft_limit"),
    "signal": ("lt.watcher.stop.signal", "lt.watcher.stop.signal_limit"),
    "pipeline": ("lt.watcher.stop.pipeline", "lt.watcher.stop.pipeline_limit"),
    "build": ("lt.watcher.stop.build", "lt.watcher.stop.build_limit"),
    "platform": ("lt.watcher.stop.platform", "lt.watcher.stop.platform_limit"),
}


class RunTakenOver(Exception):
    """Another watcher owns the run now, or its claim could not be renewed: leave it alone in this pass."""


class _Claim:
    """The claim of one run during a pass: renewed while the pass runs long, written before side effects."""

    def __init__(self, store: LoadTestStore, run_id: str, watcher_id: str, renewed_at: float) -> None:
        self.store = store
        self.run_id = run_id
        self.watcher_id = watcher_id
        self.renewed_at = renewed_at

    def renew(self) -> None:
        if time.monotonic() - self.renewed_at < HEARTBEAT_EVERY_SEC:
            return
        try:
            owned = self.store.heartbeat(self.run_id, self.watcher_id)
        except Exception as exc:
            raise RunTakenOver(f"claim not renewed: {exc}") from exc
        if not owned:
            raise RunTakenOver("taken over by another watcher")
        self.renewed_at = time.monotonic()

    def checkpoint(self, run: RunRecord) -> None:
        """Writes the run now: for state others must see before the side effect that follows."""
        try:
            owned = self.store.save_progress(run, self.watcher_id)
        except Exception as exc:
            raise RunTakenOver(f"progress not saved: {exc}") from exc
        if not owned:
            raise RunTakenOver("taken over by another watcher")
        self.renewed_at = time.monotonic()


def _utc(ts: Optional[float]) -> Optional[datetime]:
    return datetime.fromtimestamp(ts, timezone.utc) if ts else None


def _seconds(delta: timedelta) -> float:
    return delta.total_seconds()


def _wipe_ci_secrets(inst: InstanceRecord) -> None:
    """Password parameters of a pipeline are needed for its start only."""
    for name in inst.ci_secret:
        if inst.ci_params.get(name):
            inst.ci_params[name] = MASK


def _wipe_env(inst: InstanceRecord) -> None:
    """Secrets typed into the environment are not kept once the instance stopped."""
    if inst.env:
        inst.env = {name: MASK for name in inst.env}
    _wipe_ci_secrets(inst)


def _say(target: Any, message: Any, limit: Optional[int] = None) -> None:
    """Stores the message of a run or an instance: the text now, and its ref for readers of other languages.

    ``message`` is a dictionary message, a localized error or a ready text (no ref: shown as it is).
    """
    text = str(message)
    target.message = text[:limit] if limit else text
    target.message_ref = ref_of(message)


def _stored(target: Any) -> Any:
    """The stored message of a run or an instance as a message parameter (its ref when it has one)."""
    return target.message_ref if is_ref(target.message_ref) else target.message


def _with_detail(message: Any, detail: Any) -> Any:
    return msg("lt.common.detail", message=message, detail=detail) if detail else message


def _with_skew(message: Any, skew: float) -> Any:
    if abs(skew) <= CLOCK_SKEW_NOTE_SEC:
        return message
    return msg("lt.watcher.with_skew", message=message, seconds=int(skew))


def _past_deadline(inst: InstanceRecord, executor: Executor, max_seconds: int, now: datetime) -> bool:
    """Whether the instance runs past the run limit: at once for generators without their own timer,
    after some slack for those whose timer should have stopped it already."""
    if not max_seconds or inst.started_at is None:
        return False
    slack = DEADLINE_SLACK_SEC if executor.enforces_deadline else 0
    return _seconds(now - inst.started_at) > max_seconds + slack


def summarize_status(output: str) -> str:
    """Short text of a tool status answer: ``vus: "10"`` and ``vus-max: "20"`` of k6 become "VU 10 of 20".

    The k6 summary is a dictionary message (its ref goes to ``status_text_ref``); other output stays as it is.
    """
    values: dict[str, str] = {}
    for line in (output or "").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            values[key.strip().lower()] = value.strip().strip('"')
    if values.get("vus"):
        if values.get("vus-max"):
            text = msg("lt.watcher.status_vus_of", vus=values["vus"], max=values["vus-max"])
        else:
            text = msg("lt.watcher.status_vus", vus=values["vus"])
        if values.get("paused") == "true":
            text = msg("lt.watcher.status_paused", status=text)
        return text
    first = next((line.strip() for line in (output or "").splitlines() if line.strip()), "")
    return first[:120]


def _ran(inst: InstanceRecord) -> bool:
    """Whether the instance produced load: started, and not a failure within its first minute."""
    if inst.started_at is None:
        return False
    if inst.status == INST_FAILED and inst.finished_at is not None:
        return _seconds(inst.finished_at - inst.started_at) >= MIN_FAILED_RUN_SEC
    return True


def _exit_status(code: Optional[int], tool: Optional[ToolTemplate], stop_sent: bool, started: bool, timed_out: bool = False) -> tuple[str, str]:
    """(instance status, message) for an exit code; the message is a dictionary message (a ``str``)."""
    if timed_out:
        return INST_TIMED_OUT, msg("lt.watcher.exit.timed_out")
    if stop_sent:
        if code is not None:
            return INST_STOPPED, msg("lt.watcher.exit.stopped_code", code=code)
        return INST_STOPPED, msg("lt.watcher.exit.stopped")
    if code == TIMEOUT_EXIT_CODE:
        return INST_TIMED_OUT, msg("lt.watcher.exit.timed_out_generator")
    if code is None:
        return INST_FAILED, msg("lt.watcher.exit.no_code")
    # Without a tool template (a CI pipeline) 0 is the only success.
    if code in (tool.success_exit_codes if tool is not None else (0,)):
        return INST_FINISHED, msg("lt.watcher.exit.finished")
    if tool is not None and code in tool.failed_checks_exit_codes:
        return INST_CHECKS_FAILED, msg("lt.watcher.exit.checks_failed", code=code)
    if code == 126 and not started:
        return INST_FAILED, msg("lt.watcher.exit.no_work_dir")
    if code == 127:
        return INST_FAILED, msg("lt.watcher.exit.command_not_found")
    return INST_FAILED, msg("lt.watcher.exit.code", code=code)


class Watcher:
    def __init__(
        self,
        store_getter: Callable[[], LoadTestStore] = get_store,
        catalog_loader: Callable[[], Catalog] = load_catalog,
        executor_getter: Callable[[Generator], Executor] = executor_for,
        report_starter: Optional[Callable[..., object]] = None,
        clock: Callable[[], float] = time.time,
        interval: float = TICK_SEC,
        idle_interval: Optional[float] = None,
    ) -> None:
        self.id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._store_getter = store_getter
        self._catalog_loader = catalog_loader
        self._executor_getter = executor_getter
        self._report_starter = report_starter
        self._clock = clock
        self._interval = interval
        self._idle_interval = max(interval, IDLE_TICK_SEC) if idle_interval is None else idle_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Set by wake() and stop(): ends the idle pause early.
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._tick_lock = threading.Lock()
        self._last_error_log = 0.0

    # ---- thread -----------------------------------------------------------------------------

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._wake.clear()
            self._thread = threading.Thread(target=self._loop, name="load-test-watcher", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Ends the idle pause now (a run was created or seen); the pause between busy passes is kept."""
        self._wake.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            handled: Optional[int] = None
            try:
                handled = self.tick()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_error_log > ERROR_LOG_INTERVAL_SEC:
                    self._last_error_log = now
                    logger.warning("Load test watcher tick failed: %s", exc, exc_info=not _is_storage_error(exc))
            self._pause(handled)

    def _pause(self, handled: Optional[int]) -> None:
        """``interval`` after a pass that handled runs (or failed), ``idle_interval`` after an empty one.

        Only the idle pause listens to wake(): while runs are handled the passes keep their pace however
        often the pages poll.
        """
        if handled == 0:
            self._wake.wait(self._idle_interval)
        else:
            self._stop.wait(self._interval)
        self._wake.clear()

    # ---- one pass -----------------------------------------------------------------------------

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), timezone.utc)

    def tick(self) -> int:
        """Advances every claimed run once; returns how many runs were handled."""
        with self._tick_lock:
            store = self._store_getter()
            runs = store.claim(self.id, STALE_AFTER_SEC)
            if not runs:
                return 0
            claimed_at = time.monotonic()
            catalog = self._catalog_loader()
            for run in runs:
                claim = _Claim(store, run.run_id, self.id, claimed_at)
                try:
                    self.advance(run, catalog, claim)
                except RunTakenOver as exc:
                    logger.info("Load test run %s is left to another watcher: %s", run.run_id, exc)
                    continue
                except Exception:
                    # Instances have their own error handling, so this failed in the run bookkeeping; what the
                    # instances did in this pass (launches, stops) is still written, or it would be repeated.
                    logger.exception("Load test run %s could not be advanced", run.run_id)
                if not store.save_progress(run, self.id):
                    logger.info("Load test run %s was taken over by another watcher", run.run_id)
            return len(runs)

    def advance(self, run: RunRecord, catalog: Catalog, claim: Optional[_Claim] = None) -> RunRecord:
        now = self._now()
        for inst in run.instances:
            if inst.status in INST_TERMINAL:
                continue
            if claim is not None:
                claim.renew()
            try:
                self._advance_instance(run, inst, catalog, now, claim)
            except RunTakenOver:
                raise
            except Exception as exc:
                self._instance_error(run, inst, exc, now)
        self._settle(run, catalog, now)
        return run

    # ---- instances ----------------------------------------------------------------------------

    def _instance_error(self, run: RunRecord, inst: InstanceRecord, exc: Exception, now: datetime) -> None:
        """An unexpected error of one instance: noted on it while the other instances and the run go on."""
        if inst.status == INST_PENDING:
            # Nothing has been started yet (launch errors are handled in _launch): the plan or settings are broken.
            logger.exception("Instance %s of load test run %s could not be started", inst.index, run.run_id)
            inst.status = INST_FAILED
            _say(inst, msg("lt.watcher.not_started", error=error_param(exc)), 500)
            _wipe_env(inst)
            return
        message = msg("lt.watcher.watcher_error", error=error_param(exc))
        if inst.message != str(message)[:500]:
            logger.exception("Instance %s of load test run %s could not be advanced", inst.index, run.run_id)
        _say(inst, message, 500)
        max_seconds = int(run.options.get("max_seconds") or 0)
        since = inst.seen_at or inst.launched_at or now
        if _seconds(now - since) > max_seconds + UNREACHABLE_GRACE_SEC:
            inst.status = INST_LOST
            inst.finished_at = inst.finished_at or now
            _say(inst, msg("lt.watcher.watch_lost", error=error_param(exc)), 500)
            _wipe_env(inst)

    def _advance_manual(self, inst: InstanceRecord, stop: bool, max_seconds: int, now: datetime) -> None:
        if stop:
            inst.status = INST_FINISHED
            _say(inst, msg("lt.watcher.manual_stopped"))
            inst.finished_at = now
            _wipe_env(inst)
        elif max_seconds and inst.started_at and _seconds(now - inst.started_at) > max_seconds:
            # A forgotten manual run would hold its slots forever: the window closes at the run limit.
            inst.status, inst.timed_out = INST_TIMED_OUT, True
            inst.finished_at = inst.started_at + timedelta(seconds=max_seconds)
            _say(inst, msg("lt.watcher.manual_window_closed"))
            _wipe_env(inst)

    def _advance_instance(self, run: RunRecord, inst: InstanceRecord, catalog: Catalog, now: datetime,
                          claim: Optional[_Claim] = None) -> None:
        generator = catalog.generators.get(inst.generator)
        stop = run.stop_requested(inst.index)
        max_seconds = int(run.options.get("max_seconds") or 0)
        if inst.status == INST_PENDING:
            if stop:
                inst.status = INST_CANCELLED
                _say(inst, msg("lt.watcher.cancelled_before_start"))
                _wipe_env(inst)
                return
            if generator is None:
                inst.status = INST_FAILED
                _say(inst, msg("lt.watcher.generator_missing", generator=inst.generator))
                _wipe_env(inst)
                return
            if now < run.created_at + timedelta(seconds=inst.start_delay_sec):
                left = int(_seconds(run.created_at + timedelta(seconds=inst.start_delay_sec) - now))
                _say(inst, msg("lt.watcher.start_in", seconds=left))
                return
            self._launch(run, inst, generator, catalog, now, claim)
            return
        if generator is None:
            inst.status = INST_LOST
            _say(inst, msg("lt.watcher.generator_gone", generator=inst.generator))
            inst.finished_at = inst.finished_at or now
            _wipe_env(inst)
            return
        if generator.is_manual:
            self._advance_manual(inst, stop, max_seconds, now)
            return
        tool = catalog.tools.get(inst.tool)
        executor = self._executor_getter(generator)
        # Docker has no run time limit of its own, and the timers of the others may not fire (a lost CI event).
        if not stop and not inst.timed_out and _past_deadline(inst, executor, max_seconds, now):
            inst.timed_out = True
        if stop or inst.timed_out:
            self._stop_instance(run, inst, generator, executor, tool, now)
        try:
            probe = executor.probe(inst.ref)
        except GeneratorError as exc:
            _say(inst, msg("lt.watcher.no_connection", error=error_param(exc)))
            limit_sec = max_seconds + UNREACHABLE_GRACE_SEC
            since = inst.seen_at or inst.launched_at or now
            if _seconds(now - since) > limit_sec:
                inst.status = INST_LOST
                inst.finished_at = inst.finished_at or now
                _say(inst, msg("lt.watcher.unreachable", error=error_param(exc)))
                _wipe_env(inst)
            else:
                self._give_up_stop(inst, executor, tool, now)
            if inst.status in INST_TERMINAL:
                self._finish_instance(inst, executor)
            return
        inst.seen_at = now
        if probe.ref:
            inst.ref = probe.ref
        if probe.url:
            inst.external_url = probe.url
        self._apply_probe(run, inst, probe, tool, executor, now)
        if inst.status not in INST_TERMINAL:
            self._give_up_stop(inst, executor, tool, now)
        if inst.status in INST_TERMINAL:
            self._finish_instance(inst, executor)
        elif inst.status == INST_RUNNING and inst.stop_sent_at is None and inst.stop_failed_at is None:
            self._ask_status(run, inst, generator, executor, tool, now)

    def _plan(self, run: RunRecord, inst: InstanceRecord, generator: Generator, tool: ToolTemplate, ci_token: str = ""):
        return instance_plan(
            generator, tool, run.run_id, run.run_name, run.run_tag, inst,
            int(run.options.get("max_seconds") or 3600), ci_token=ci_token,
        )

    def _launch(self, run: RunRecord, inst: InstanceRecord, generator: Generator, catalog: Catalog, now: datetime,
                claim: Optional[_Claim] = None) -> None:
        if generator.is_manual:
            inst.status = INST_RUNNING
            inst.launched_at = inst.started_at = now
            _say(inst, msg("lt.watcher.manual_running"))
            _wipe_env(inst)  # LoadLens runs nothing here, the values have no use
            return
        tool = catalog.tools.get(inst.tool)
        if tool is None and not generator.is_ci:
            inst.status = INST_FAILED
            _say(inst, msg("lt.watcher.tool_missing", tool=inst.tool))
            _wipe_env(inst)
            return
        # The per-instance secret only matters when the pipeline can report events to LoadLens.
        token = secrets.token_urlsafe(32) if generator.is_ci and generator.ci and generator.ci.loadlens_url else ""
        try:
            plan = self._plan(run, inst, generator, tool, token)
        except (ToolInputError, ToolConfigError) as exc:
            inst.status = INST_FAILED
            _say(inst, exc)
            _wipe_env(inst)
            return
        except GeneratorError as exc:
            inst.status = INST_FAILED
            _say(inst, msg("lt.ci.params_unreadable", error=error_param(exc)))
            _wipe_env(inst)
            return
        if token:
            inst.ci_token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            if claim is not None:
                # Written before the trigger: the pipeline may report its start before this pass ends.
                claim.checkpoint(run)
        try:
            result = self._executor_getter(generator).launch(plan)
        except Exception as exc:
            if not isinstance(exc, GeneratorError):
                logger.warning("Launch of %s failed unexpectedly", plan.ref, exc_info=True)
            inst.status = INST_FAILED
            _say(inst, msg("lt.watcher.not_started", error=error_param(exc)))
            inst.ci_token_hash = ""
            _wipe_env(inst)
            return
        inst.ref = result.ref
        if not generator.is_ci:
            inst.command = plan.command.shell(mask_env=True)
        inst.status = INST_STARTING
        inst.launched_at = now
        if result.url:
            inst.external_url = result.url
        if result.already_running:
            _say(inst, msg("lt.watcher.already_running"))
        elif generator.is_ci:
            started = msg("lt.watcher.pipeline_started") if generator.ci and generator.ci.kind == "gitlab" \
                else msg("lt.watcher.build_queued")
            _say(inst, _with_detail(started, result.message))
            _wipe_ci_secrets(inst)
        else:
            _say(inst, msg("lt.watcher.launched"))

    def _save_log(self, inst: InstanceRecord, executor: Executor) -> None:
        """Keeps the tail of the output in the run: containers and jobs disappear after they end."""
        try:
            text = executor.logs(inst.ref, STORED_LOG_BYTES)
        except Exception:
            return
        if text:
            inst.log_tail = text[-STORED_LOG_BYTES:]

    def _finish_instance(self, inst: InstanceRecord, executor: Executor) -> None:
        self._save_log(inst, executor)
        try:
            executor.cleanup(inst.ref)
        except Exception as exc:
            logger.info("Clean-up of %s failed: %s", inst.ref, exc)

    def _soft_stop(self, run: RunRecord, inst: InstanceRecord, generator: Generator, executor: Executor,
                   tool: Optional[ToolTemplate]) -> bool:
        """Runs the tool's own stop command (JMeter shutdown); ``False`` falls back to the signal."""
        if tool is None or not tool.stop_command or not executor.supports_control:
            return False
        try:
            plan = self._plan(run, inst, generator, tool)
            code, output = executor.control(inst.ref, plan.stop_command, plan.command.cwd)
        except Exception as exc:
            logger.info("Soft stop of %s failed: %s", inst.ref, exc)
            return False
        return code == 0

    def _stop_instance(self, run: RunRecord, inst: InstanceRecord, generator: Generator, executor: Executor,
                       tool: Optional[ToolTemplate], now: datetime) -> None:
        signal = tool.stop_signal if tool else "TERM"
        grace = tool.grace_seconds if tool else 30
        # The run limit, not a person, stopped it: the second message of each pair says so first.
        variant = 1 if inst.timed_out else 0
        if not getattr(executor, "supports_stop", True):
            if inst.stop_sent_at is None:
                self._save_log(inst, executor)
                inst.stop_sent_at = now
                _say(inst, msg(_STOP_KEYS["unsupported"][variant], grace=grace))
            return
        try:
            if inst.stop_sent_at is None:
                self._save_log(inst, executor)
                soft = self._soft_stop(run, inst, generator, executor, tool)
                if not soft:
                    executor.signal(inst.ref, signal, grace)
                inst.stop_sent_at = now
                if soft:
                    _say(inst, msg(_STOP_KEYS["soft"][variant], grace=grace))
                elif executor.supports_signals:
                    _say(inst, msg(_STOP_KEYS["signal"][variant], signal=signal, grace=grace))
                elif generator.is_ci:
                    what = "pipeline" if generator.ci and generator.ci.kind == "gitlab" else "build"
                    _say(inst, msg(_STOP_KEYS[what][variant], grace=grace))
                else:
                    _say(inst, msg(_STOP_KEYS["platform"][variant], grace=grace))
            else:
                extra = 0 if executor.supports_signals else PLATFORM_KILL_EXTRA_SEC
                if not inst.killed and _seconds(now - inst.stop_sent_at) > grace + extra:
                    executor.kill(inst.ref)
                    inst.killed = True
                    _say(inst, msg("lt.watcher.stop.again") if generator.is_ci else msg("lt.watcher.stop.killed"))
        except Exception as exc:
            if not isinstance(exc, GeneratorError):
                logger.info("Stop of %s failed unexpectedly", inst.ref, exc_info=True)
            inst.stop_failed_at = inst.stop_failed_at or now
            _say(inst, msg("lt.watcher.stop.failed", error=error_param(exc)))

    def _give_up_stop(self, inst: InstanceRecord, executor: Executor, tool: Optional[ToolTemplate], now: datetime) -> None:
        """Closes the window of an instance whose stop the generator does not confirm in time.

        The process may still run on the generator (the message says so), but the run, its slots and its
        report no longer wait for it.
        """
        grace = tool.grace_seconds if tool else 30
        if not getattr(executor, "supports_stop", True):
            anchor, limit = inst.stop_sent_at, grace
            text = msg("lt.watcher.give_up.unsupported")
        elif inst.stop_sent_at is not None:
            extra = 0 if executor.supports_signals else PLATFORM_KILL_EXTRA_SEC
            anchor, limit = inst.stop_sent_at, grace + extra + STOP_CONFIRM_SEC
            text = msg("lt.watcher.give_up.unconfirmed")
        else:
            anchor, limit = inst.stop_failed_at, grace + STOP_CONFIRM_SEC
            text = msg("lt.watcher.give_up.undelivered")
        if anchor is None or _seconds(now - anchor) <= limit:
            return
        inst.status, _ = _exit_status(None, tool, True, inst.started_at is not None, inst.timed_out)
        inst.finished_at = inst.finished_at or now
        _say(inst, text, 500)
        _wipe_env(inst)

    def _ask_status(self, run: RunRecord, inst: InstanceRecord, generator: Generator, executor: Executor,
                    tool: Optional[ToolTemplate], now: datetime) -> None:
        if tool is None or not tool.status_command or not executor.supports_control:
            return
        if inst.status_at is not None and _seconds(now - inst.status_at) < STATUS_EVERY_SEC:
            return
        inst.status_at = now
        try:
            plan = self._plan(run, inst, generator, tool)
            code, output = executor.control(inst.ref, plan.status_command, plan.command.cwd)
        except Exception:
            return
        if code == 0:
            status = summarize_status(output)
            inst.status_text, inst.status_text_ref = str(status), ref_of(status)

    def _apply_probe(self, run: RunRecord, inst: InstanceRecord, probe: Probe, tool: Optional[ToolTemplate],
                     executor: Executor, now: datetime) -> None:
        skew = (probe.remote_now - now.timestamp()) if probe.remote_now else 0.0
        if probe.state == "starting":
            inst.status = INST_STARTING
            waited = _seconds(now - inst.launched_at) if inst.launched_at else 0.0
            max_seconds = int(run.options.get("max_seconds") or 0)
            if (probe.stuck and waited > STUCK_FAIL_SEC) or (max_seconds and waited > max_seconds):
                inst.status = INST_FAILED
                inst.finished_at = now
                _say(inst, msg("lt.watcher.never_started", reason=probe.message or msg("lt.watcher.run_limit_expired")))
                self._save_log(inst, executor)
                try:
                    executor.kill(inst.ref)
                except GeneratorError:
                    pass
                _wipe_env(inst)
                return
            if inst.stop_sent_at is None and inst.stop_failed_at is None:
                if probe.message:
                    _say(inst, probe.message)
                elif waited > SLOW_START_SEC:
                    _say(inst, msg("lt.watcher.slow_start"))
            return
        if probe.state == "running":
            inst.status = INST_RUNNING
            inst.dead_probes = 0
            inst.started_at = inst.started_at or _utc(probe.started_at) or now
            # A stop in progress (or one that did not get through) keeps its message.
            if inst.stop_sent_at is None and inst.stop_failed_at is None:
                _say(inst, _with_skew(msg("lt.watcher.running"), skew))
            return
        if probe.state == "exited":
            started = probe.started_at is not None or inst.started_at is not None
            inst.started_at = inst.started_at or _utc(probe.started_at)
            inst.finished_at = _utc(probe.finished_at) or now
            inst.exit_code = probe.exit_code
            inst.status, message = _exit_status(probe.exit_code, tool, inst.stop_sent_at is not None, started, inst.timed_out)
            if inst.status == INST_FAILED and not started:
                inst.started_at = None
            _say(inst, _with_skew(_with_detail(message, probe.message), skew))
            _wipe_env(inst)
            return
        if probe.state == "dead":
            inst.dead_probes += 1
            if inst.dead_probes >= DEAD_PROBES_TO_LOST:
                inst.status = INST_LOST
                inst.started_at = inst.started_at or _utc(probe.started_at)
                inst.finished_at = now
                _say(inst, msg("lt.watcher.process_vanished"))
                _wipe_env(inst)
            return
        # missing: nothing left on the generator. After a stop that is how a deleted pod or job looks.
        inst.finished_at = inst.finished_at or now
        if inst.stop_sent_at is not None:
            inst.status, message = _exit_status(None, tool, True, inst.started_at is not None, inst.timed_out)
            _say(inst, message)
        else:
            inst.status = INST_LOST
            _say(inst, msg("lt.watcher.instance_vanished"))
        _wipe_env(inst)

    # ---- run --------------------------------------------------------------------------------

    def _settle(self, run: RunRecord, catalog: Catalog, now: datetime) -> None:
        started = [inst.started_at for inst in run.instances if inst.started_at]
        if started:
            run.started_at = min(started)
        active = [inst for inst in run.instances if inst.status in INST_ACTIVE]
        if active:
            if run.stop_all_at is not None:
                run.status = RUN_STOPPING
            elif any(inst.status in (INST_STARTING, INST_RUNNING) for inst in active):
                run.status = RUN_RUNNING
            else:
                run.status = RUN_STARTING
            running = sum(1 for inst in run.instances if inst.status == INST_RUNNING)
            pending = sum(1 for inst in run.instances if inst.status == INST_PENDING)
            if pending:
                _say(run, msg("lt.watcher.run.progress_pending", running=running, total=len(run.instances), pending=pending))
            else:
                _say(run, msg("lt.watcher.run.progress", running=running, total=len(run.instances)))
            return
        if run.status in (RUN_DONE, RUN_FAILED, RUN_CANCELLED):
            return
        ran = [inst for inst in run.instances if _ran(inst)]
        if not ran:
            cancelled = run.stop_all_at is not None or all(inst.status == INST_CANCELLED for inst in run.instances)
            run.status = RUN_CANCELLED if cancelled else RUN_FAILED
            first = next((_stored(inst) for inst in run.instances if inst.status == INST_FAILED and inst.message), "")
            if cancelled:
                _say(run, msg("lt.watcher.run.cancelled_before_start"))
            elif first:
                _say(run, msg("lt.watcher.run.none_started_reason", reason=first))
            else:
                _say(run, msg("lt.watcher.run.none_started"))
            run.finished_at = now
            return
        run.started_at = min(inst.started_at for inst in ran)
        run.finished_at = max((inst.finished_at or now) for inst in ran)
        if not run.options.get("auto_report", True):
            run.status = RUN_DONE
            _say(run, msg("lt.watcher.run.done_no_report"))
            return
        if run.status in (RUN_STARTING, RUN_RUNNING, RUN_STOPPING):
            run.status = RUN_DRAINING
        if run.status == RUN_DRAINING:
            drain = self._drain_seconds(run)
            ready_at = run.finished_at + timedelta(seconds=drain)
            if now < ready_at:
                _say(run, msg("lt.watcher.run.draining", seconds=int(_seconds(ready_at - now)) + 1))
                return
            self._start_report(run, catalog)
            return
        if run.status == RUN_REPORTING:
            self._follow_report(run)

    def _drain_seconds(self, run: RunRecord) -> int:
        try:
            from loadlens_app.blueprints import dashboard

            step = step_seconds(dashboard._report_step(run.project_area or run.service, run.service)) or 60
        except Exception:
            step = 60
        return int(max(DRAIN_MIN_SEC, min(DRAIN_MAX_SEC, 2 * step)))

    def _start_report(self, run: RunRecord, catalog: Catalog) -> None:
        from loadlens_app.blueprints import dashboard
        from loadlens_app.i18n import report_language
        from loadlens_app.jobs import job_store

        owner = run.owner or DEFAULT_OWNER
        try:
            check_job_slot(*job_store.running_counts(owner))
        except LimitError as exc:
            _say(run, msg("lt.watcher.report.waits_slot", error=error_param(exc)))
            return
        assert run.started_at is not None and run.finished_at is not None
        start_ts, end_ts = run.started_at.timestamp(), run.finished_at.timestamp()
        packs, note = report_filter(run, catalog.tools, start_ts, end_ts)
        notes_before: list[Any] = []
        if any(inst.ci_kind and inst.started_at for inst in run.instances):
            # A pipeline's start and end include its preparation: the report window follows the load itself.
            bounds = rps_bounds(run, start_ts, end_ts, packs)
            if bounds is None:
                notes_before.append(msg("lt.watcher.report.note_pipeline_window"))
            elif bounds != (start_ts, end_ts):
                start_ts, end_ts = bounds
                notes_before.append(msg(
                    "lt.watcher.report.note_window_narrowed",
                    start=datetime.fromtimestamp(start_ts).strftime("%H:%M:%S"),
                    end=datetime.fromtimestamp(end_ts).strftime("%H:%M:%S"),
                ))
        report_name = run.run_name
        try:
            if dashboard._run_name_taken(report_name, run.service):
                report_name = f"{run.run_name}-{run.run_id[:4]}"
        except Exception:
            pass
        starter = self._report_starter or dashboard.start_report_job
        use_llm = bool(run.options.get("use_llm", True))
        notes = notes_before + ([note] if note else [])
        kwargs = dict(
            save_to_db=False,
            web_only=True,
            run_name=report_name,
            test_type=run.test_type,
            owner=owner,
            run_tag=run.run_tag if packs else None,
            run_filter_packs=packs,
            # The watcher has no reader: the report is in the language of the installation.
            report_language=report_language(),
        )
        try:
            try:
                job = starter(int(start_ts * 1000), int(end_ts * 1000), run.service, run.project_area, use_llm=use_llm, **kwargs)
            except LimitError as exc:
                if not use_llm or exc.code != "limit_exceeded":
                    raise
                notes.append(msg("lt.watcher.report.note_ai_skipped", error=error_param(exc)))
                job = starter(int(start_ts * 1000), int(end_ts * 1000), run.service, run.project_area, use_llm=False, **kwargs)
        except LimitError as exc:
            run.status = RUN_DONE
            _say(run, msg("lt.watcher.report.not_started", error=error_param(exc)))
            return
        except Exception as exc:
            logger.exception("Report of load test run %s did not start", run.run_id)
            run.status = RUN_DONE
            _say(run, msg("lt.watcher.report.not_started", error=error_param(exc)))
            return
        run.report_job_id = getattr(job, "job_id", None)
        run.run_name = report_name
        run.status = RUN_REPORTING
        if notes:
            _say(run, msg("lt.watcher.report.building_notes", notes=joined(notes)))
        else:
            _say(run, msg("lt.watcher.report.building"))

    def _follow_report(self, run: RunRecord) -> None:
        from loadlens_app.jobs import JOB_STATUS_DONE, JOB_STATUS_ERROR, job_store

        job = job_store.get(run.report_job_id) if run.report_job_id else None
        if job is None:
            run.status = RUN_DONE
            _say(run, msg("lt.watcher.report.job_missing"))
            return
        if job.status == JOB_STATUS_DONE:
            run.status = RUN_DONE
            run.report_url = job.page_url or job.report_url
            _say(run, msg("lt.watcher.report.ready"))
        elif job.status == JOB_STATUS_ERROR:
            run.status = RUN_DONE
            _say(run, msg("lt.watcher.report.failed", error=_job_message(job)))
        else:
            _say(run, msg("lt.watcher.report.progress", stage=_job_message(job), progress=int(job.progress)))


def _job_message(job: Any) -> Any:
    """The message of a report job as a parameter: its ref when the job store keeps one, else the text."""
    ref = getattr(job, "message_ref", None)
    return ref if is_ref(ref) else (job.message or "")


def _is_storage_error(exc: BaseException) -> bool:
    try:
        from AI.storage_state import is_storage_error

        return bool(is_storage_error(exc))
    except Exception:
        return False


_watcher: Optional[Watcher] = None
_watcher_lock = threading.Lock()


def watcher_enabled() -> bool:
    """``LOADLENS_LOAD_TEST_WATCHER=0`` switches the background loop off (tests, read-only replicas)."""
    return os.getenv("LOADLENS_LOAD_TEST_WATCHER", "1").strip() != "0"


def get_watcher() -> Watcher:
    global _watcher
    with _watcher_lock:
        if _watcher is None:
            _watcher = Watcher()
        return _watcher


def ensure_watcher() -> bool:
    """Starts the loop of this process if it is enabled and not running yet, else wakes it from the idle pause.

    Returns True when this call started the loop.
    """
    if not watcher_enabled():
        return False
    watcher = get_watcher()
    if watcher.start():
        return True
    watcher.wake()
    return False


__all__ = ["Watcher", "ensure_watcher", "get_watcher", "watcher_enabled"]
