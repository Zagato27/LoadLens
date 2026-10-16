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
from typing import Callable, Optional

from loadlens_app.limits import LimitError, check_job_slot, step_seconds

from .executors import STORED_LOG_BYTES, Executor, GeneratorError, Probe, executor_for
from .generators import Generator
from .live import report_filter, rps_bounds
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


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _past_deadline(inst: InstanceRecord, executor: Executor, max_seconds: int, now: datetime) -> bool:
    """Whether the instance runs past the run limit: at once for generators without their own timer,
    after some slack for those whose timer should have stopped it already."""
    if not max_seconds or inst.started_at is None:
        return False
    slack = DEADLINE_SLACK_SEC if executor.enforces_deadline else 0
    return _seconds(now - inst.started_at) > max_seconds + slack


def summarize_status(output: str) -> str:
    """Short text of a tool status answer: ``vus: "10"`` and ``vus-max: "20"`` of k6 become «VU 10 из 20»."""
    values: dict[str, str] = {}
    for line in (output or "").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            values[key.strip().lower()] = value.strip().strip('"')
    if values.get("vus"):
        text = f"VU {values['vus']}"
        if values.get("vus-max"):
            text += f" из {values['vus-max']}"
        if values.get("paused") == "true":
            text += ", пауза"
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
    if timed_out:
        return INST_TIMED_OUT, "Остановлен по лимиту длительности прогона"
    if stop_sent:
        return INST_STOPPED, "Остановлен по запросу" + (f" (код выхода {code})" if code is not None else "")
    if code == TIMEOUT_EXIT_CODE:
        return INST_TIMED_OUT, "Остановлен генератором по лимиту длительности прогона"
    if code is None:
        return INST_FAILED, "Код выхода не прочитан"
    # Without a tool template (a CI pipeline) 0 is the only success.
    if code in (tool.success_exit_codes if tool is not None else (0,)):
        return INST_FINISHED, "Завершён"
    if tool is not None and code in tool.failed_checks_exit_codes:
        return INST_CHECKS_FAILED, f"Завершён, инструмент сообщил о непройденных проверках (код {code})"
    if code == 126 and not started:
        return INST_FAILED, "Не найден рабочий каталог инструмента (код 126), см. журнал"
    if code == 127:
        return INST_FAILED, "Команда инструмента не найдена на генераторе (код 127), см. журнал"
    return INST_FAILED, f"Инструмент завершился с кодом {code}, см. журнал"


class Watcher:
    def __init__(
        self,
        store_getter: Callable[[], LoadTestStore] = get_store,
        catalog_loader: Callable[[], Catalog] = load_catalog,
        executor_getter: Callable[[Generator], Executor] = executor_for,
        report_starter: Optional[Callable[..., object]] = None,
        clock: Callable[[], float] = time.time,
        interval: float = TICK_SEC,
    ) -> None:
        self.id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._store_getter = store_getter
        self._catalog_loader = catalog_loader
        self._executor_getter = executor_getter
        self._report_starter = report_starter
        self._clock = clock
        self._interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._tick_lock = threading.Lock()
        self._last_error_log = 0.0

    # ---- thread -----------------------------------------------------------------------------

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="load-test-watcher", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_error_log > ERROR_LOG_INTERVAL_SEC:
                    self._last_error_log = now
                    logger.warning("Load test watcher tick failed: %s", exc, exc_info=not _is_storage_error(exc))
            self._stop.wait(self._interval)

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
            inst.status, inst.message = INST_FAILED, f"Не запустился: {exc}"[:500]
            _wipe_env(inst)
            return
        text = f"Ошибка наблюдателя: {exc}"[:500]
        if inst.message != text:
            logger.exception("Instance %s of load test run %s could not be advanced", inst.index, run.run_id)
        inst.message = text
        max_seconds = int(run.options.get("max_seconds") or 0)
        since = inst.seen_at or inst.launched_at or now
        if _seconds(now - since) > max_seconds + UNREACHABLE_GRACE_SEC:
            inst.status = INST_LOST
            inst.finished_at = inst.finished_at or now
            inst.message = f"Наблюдатель не может следить за инстансом дольше лимита прогона: {exc}"[:500]
            _wipe_env(inst)

    def _advance_manual(self, inst: InstanceRecord, stop: bool, max_seconds: int, now: datetime) -> None:
        if stop:
            inst.status, inst.message = INST_FINISHED, "Остановлен вручную"
            inst.finished_at = now
            _wipe_env(inst)
        elif max_seconds and inst.started_at and _seconds(now - inst.started_at) > max_seconds:
            # A forgotten manual run would hold its slots forever: the window closes at the run limit.
            inst.status, inst.timed_out = INST_TIMED_OUT, True
            inst.finished_at = inst.started_at + timedelta(seconds=max_seconds)
            inst.message = "Окно закрыто по лимиту длительности прогона: остановите инструмент, если он ещё идёт"
            _wipe_env(inst)

    def _advance_instance(self, run: RunRecord, inst: InstanceRecord, catalog: Catalog, now: datetime,
                          claim: Optional[_Claim] = None) -> None:
        generator = catalog.generators.get(inst.generator)
        stop = run.stop_requested(inst.index)
        max_seconds = int(run.options.get("max_seconds") or 0)
        if inst.status == INST_PENDING:
            if stop:
                inst.status, inst.message = INST_CANCELLED, "Отменён до старта"
                _wipe_env(inst)
                return
            if generator is None:
                inst.status, inst.message = INST_FAILED, f"Генератора «{inst.generator}» нет в настройках"
                _wipe_env(inst)
                return
            if now < run.created_at + timedelta(seconds=inst.start_delay_sec):
                left = int(_seconds(run.created_at + timedelta(seconds=inst.start_delay_sec) - now))
                inst.message = f"Старт через {left} с"
                return
            self._launch(run, inst, generator, catalog, now, claim)
            return
        if generator is None:
            inst.status, inst.message = INST_LOST, f"Генератора «{inst.generator}» больше нет в настройках"
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
            inst.message = f"Нет связи с генератором: {exc}"
            limit_sec = max_seconds + UNREACHABLE_GRACE_SEC
            since = inst.seen_at or inst.launched_at or now
            if _seconds(now - since) > limit_sec:
                inst.status = INST_LOST
                inst.finished_at = inst.finished_at or now
                inst.message = f"Генератор не отвечает дольше лимита прогона: {exc}"
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
            inst.message = "Идёт: запустите инструмент сами и остановите прогон кнопкой, когда тест закончится"
            _wipe_env(inst)  # LoadLens runs nothing here, the values have no use
            return
        tool = catalog.tools.get(inst.tool)
        if tool is None and not generator.is_ci:
            inst.status, inst.message = INST_FAILED, f"Инструмента «{inst.tool}» нет в настройках"
            _wipe_env(inst)
            return
        # The per-instance secret only matters when the pipeline can report events to LoadLens.
        token = secrets.token_urlsafe(32) if generator.is_ci and generator.ci and generator.ci.loadlens_url else ""
        try:
            plan = self._plan(run, inst, generator, tool, token)
        except (ToolInputError, ToolConfigError) as exc:
            inst.status, inst.message = INST_FAILED, str(exc)
            _wipe_env(inst)
            return
        except GeneratorError as exc:
            inst.status, inst.message = INST_FAILED, f"Параметры пайплайна не читаются: {exc}"
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
            inst.status, inst.message = INST_FAILED, f"Не запустился: {exc}"
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
            inst.message = "Уже был запущен, подключились к нему"
        elif generator.is_ci:
            inst.message = ("Пайплайн запущен" if generator.ci and generator.ci.kind == "gitlab" else "Сборка поставлена в очередь") \
                + (f": {result.message}" if result.message else "")
            _wipe_ci_secrets(inst)
        else:
            inst.message = "Запущен"

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
        reason = "лимит длительности прогона: " if inst.timed_out else ""
        if not getattr(executor, "supports_stop", True):
            if inst.stop_sent_at is None:
                self._save_log(inst, executor)
                inst.stop_sent_at = now
                inst.message = _capital(f"{reason}LoadLens не может остановить задачу на этом генераторе: "
                                        f"остановите её вручную, ждём завершения до {grace} с")
            return
        try:
            if inst.stop_sent_at is None:
                self._save_log(inst, executor)
                soft = self._soft_stop(run, inst, generator, executor, tool)
                if not soft:
                    executor.signal(inst.ref, signal, grace)
                inst.stop_sent_at = now
                if soft:
                    inst.message = f"{reason}инструменту отправлена команда остановки, ждём завершения до {grace} с"
                elif executor.supports_signals:
                    inst.message = f"{reason}отправлен SIG{signal}, ждём завершения до {grace} с"
                elif generator.is_ci:
                    what = "пайплайну отправлена отмена" if generator.ci and generator.ci.kind == "gitlab" else "сборке отправлена остановка"
                    inst.message = f"{reason}{what}, ждём завершения до {grace} с"
                else:
                    inst.message = f"{reason}процесс получит SIGTERM, через {grace} с — SIGKILL"
                inst.message = _capital(inst.message)
            else:
                extra = 0 if executor.supports_signals else PLATFORM_KILL_EXTRA_SEC
                if not inst.killed and _seconds(now - inst.stop_sent_at) > grace + extra:
                    executor.kill(inst.ref)
                    inst.killed = True
                    inst.message = ("Не завершился сам: остановка отправлена ещё раз" if generator.is_ci
                                    else "Процесс не завершился сам: отправлен SIGKILL")
        except Exception as exc:
            if not isinstance(exc, GeneratorError):
                logger.info("Stop of %s failed unexpectedly", inst.ref, exc_info=True)
            inst.stop_failed_at = inst.stop_failed_at or now
            inst.message = f"Остановка не дошла до генератора: {exc}"

    def _give_up_stop(self, inst: InstanceRecord, executor: Executor, tool: Optional[ToolTemplate], now: datetime) -> None:
        """Closes the window of an instance whose stop the generator does not confirm in time.

        The process may still run on the generator (the message says so), but the run, its slots and its
        report no longer wait for it.
        """
        grace = tool.grace_seconds if tool else 30
        if not getattr(executor, "supports_stop", True):
            anchor, limit = inst.stop_sent_at, grace
            text = "LoadLens не может остановить задачу на этом генераторе и закрыл окно инстанса: остановите её вручную"
        elif inst.stop_sent_at is not None:
            extra = 0 if executor.supports_signals else PLATFORM_KILL_EXTRA_SEC
            anchor, limit = inst.stop_sent_at, grace + extra + STOP_CONFIRM_SEC
            text = "Генератор не подтвердил остановку, LoadLens закрыл окно инстанса: проверьте процесс на генераторе"
        else:
            anchor, limit = inst.stop_failed_at, grace + STOP_CONFIRM_SEC
            text = "Остановка не дошла до генератора, LoadLens закрыл окно инстанса: остановите процесс вручную"
        if anchor is None or _seconds(now - anchor) <= limit:
            return
        inst.status, _ = _exit_status(None, tool, True, inst.started_at is not None, inst.timed_out)
        inst.finished_at = inst.finished_at or now
        inst.message = text[:500]
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
            inst.status_text = summarize_status(output)

    def _apply_probe(self, run: RunRecord, inst: InstanceRecord, probe: Probe, tool: Optional[ToolTemplate],
                     executor: Executor, now: datetime) -> None:
        skew = (probe.remote_now - now.timestamp()) if probe.remote_now else 0.0
        skew_note = f"; часы генератора расходятся с LoadLens на {int(skew)} с" if abs(skew) > CLOCK_SKEW_NOTE_SEC else ""
        if probe.state == "starting":
            inst.status = INST_STARTING
            waited = _seconds(now - inst.launched_at) if inst.launched_at else 0.0
            max_seconds = int(run.options.get("max_seconds") or 0)
            if (probe.stuck and waited > STUCK_FAIL_SEC) or (max_seconds and waited > max_seconds):
                inst.status = INST_FAILED
                inst.finished_at = now
                inst.message = f"Не стартовал: {probe.message or 'истёк лимит прогона'}"
                self._save_log(inst, executor)
                try:
                    executor.kill(inst.ref)
                except GeneratorError:
                    pass
                _wipe_env(inst)
                return
            if inst.stop_sent_at is None and inst.stop_failed_at is None:
                inst.message = probe.message or ("Запускается дольше обычного" if waited > SLOW_START_SEC else inst.message)
            return
        if probe.state == "running":
            inst.status = INST_RUNNING
            inst.dead_probes = 0
            inst.started_at = inst.started_at or _utc(probe.started_at) or now
            # A stop in progress (or one that did not get through) keeps its message.
            if inst.stop_sent_at is None and inst.stop_failed_at is None:
                inst.message = "Идёт" + skew_note
            return
        if probe.state == "exited":
            started = probe.started_at is not None or inst.started_at is not None
            inst.started_at = inst.started_at or _utc(probe.started_at)
            inst.finished_at = _utc(probe.finished_at) or now
            inst.exit_code = probe.exit_code
            inst.status, inst.message = _exit_status(probe.exit_code, tool, inst.stop_sent_at is not None, started, inst.timed_out)
            if inst.status == INST_FAILED and not started:
                inst.started_at = None
            if probe.message:
                inst.message += f": {probe.message}"
            inst.message += skew_note
            _wipe_env(inst)
            return
        if probe.state == "dead":
            inst.dead_probes += 1
            if inst.dead_probes >= DEAD_PROBES_TO_LOST:
                inst.status = INST_LOST
                inst.started_at = inst.started_at or _utc(probe.started_at)
                inst.finished_at = now
                inst.message = "Процесс пропал без кода выхода: генератор перезагружался или процесс убит извне"
                _wipe_env(inst)
            return
        # missing: nothing left on the generator. After a stop that is how a deleted pod or job looks.
        inst.finished_at = inst.finished_at or now
        if inst.stop_sent_at is not None:
            inst.status, inst.message = _exit_status(None, tool, True, inst.started_at is not None, inst.timed_out)
        else:
            inst.status = INST_LOST
            inst.message = "Инстанс на генераторе пропал: удалён каталог, контейнер или задача"
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
            parts = [f"идёт {running} из {len(run.instances)}"]
            if pending:
                parts.append(f"ждут старта {pending}")
            run.message = "; ".join(parts).capitalize()
            return
        if run.status in (RUN_DONE, RUN_FAILED, RUN_CANCELLED):
            return
        ran = [inst for inst in run.instances if _ran(inst)]
        if not ran:
            cancelled = run.stop_all_at is not None or all(inst.status == INST_CANCELLED for inst in run.instances)
            run.status = RUN_CANCELLED if cancelled else RUN_FAILED
            first = next((inst.message for inst in run.instances if inst.status == INST_FAILED and inst.message), "")
            run.message = "Прогон отменён до старта" if cancelled else f"Ни один инстанс не запустился{': ' + first if first else ''}"
            run.finished_at = now
            return
        run.started_at = min(inst.started_at for inst in ran)
        run.finished_at = max((inst.finished_at or now) for inst in ran)
        if not run.options.get("auto_report", True):
            run.status = RUN_DONE
            run.message = "Завершён, отчёт не заказывали"
            return
        if run.status in (RUN_STARTING, RUN_RUNNING, RUN_STOPPING):
            run.status = RUN_DRAINING
        if run.status == RUN_DRAINING:
            drain = self._drain_seconds(run)
            ready_at = run.finished_at + timedelta(seconds=drain)
            if now < ready_at:
                run.message = f"Тест завершён, ждём доставку метрик ещё {int(_seconds(ready_at - now)) + 1} с"
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
        from loadlens_app.jobs import job_store

        owner = run.owner or DEFAULT_OWNER
        try:
            check_job_slot(*job_store.running_counts(owner))
        except LimitError as exc:
            run.message = f"Отчёт ждёт свободного слота: {exc}"
            return
        assert run.started_at is not None and run.finished_at is not None
        start_ts, end_ts = run.started_at.timestamp(), run.finished_at.timestamp()
        packs, note = report_filter(run, catalog.tools, start_ts, end_ts)
        notes_before: list[str] = []
        if any(inst.ci_kind and inst.started_at for inst in run.instances):
            # A pipeline's start and end include its preparation: the report window follows the load itself.
            bounds = rps_bounds(run, start_ts, end_ts, packs)
            if bounds is None:
                notes_before.append("окно отчёта — время пайплайна: данных RPS в нём не найдено")
            elif bounds != (start_ts, end_ts):
                start_ts, end_ts = bounds
                notes_before.append(
                    f"окно отчёта сужено по RPS: {datetime.fromtimestamp(start_ts).strftime('%H:%M:%S')}–"
                    f"{datetime.fromtimestamp(end_ts).strftime('%H:%M:%S')}"
                )
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
        )
        try:
            try:
                job = starter(int(start_ts * 1000), int(end_ts * 1000), run.service, run.project_area, use_llm=use_llm, **kwargs)
            except LimitError as exc:
                if not use_llm or exc.code != "limit_exceeded":
                    raise
                notes.append(f"анализ ИИ пропущен: {exc}")
                job = starter(int(start_ts * 1000), int(end_ts * 1000), run.service, run.project_area, use_llm=False, **kwargs)
        except LimitError as exc:
            run.status = RUN_DONE
            run.message = f"Отчёт не запущен: {exc}"
            return
        except Exception as exc:
            logger.exception("Report of load test run %s did not start", run.run_id)
            run.status = RUN_DONE
            run.message = f"Отчёт не запущен: {exc}"
            return
        run.report_job_id = getattr(job, "job_id", None)
        run.run_name = report_name
        run.status = RUN_REPORTING
        run.message = "Строится отчёт" + (f" ({'; '.join(notes)})" if notes else "")

    def _follow_report(self, run: RunRecord) -> None:
        from loadlens_app.jobs import JOB_STATUS_DONE, JOB_STATUS_ERROR, job_store

        job = job_store.get(run.report_job_id) if run.report_job_id else None
        if job is None:
            run.status = RUN_DONE
            run.message = "Задача отчёта не найдена"
            return
        if job.status == JOB_STATUS_DONE:
            run.status = RUN_DONE
            run.report_url = job.page_url or job.report_url
            run.message = "Отчёт готов"
        elif job.status == JOB_STATUS_ERROR:
            run.status = RUN_DONE
            run.message = f"Отчёт не построен: {job.message}"
        else:
            run.message = f"Строится отчёт: {job.message} ({int(job.progress)}%)"


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
    """Starts the loop of this process if it is enabled and not running yet."""
    if not watcher_enabled():
        return False
    return get_watcher().start()


__all__ = ["Watcher", "ensure_watcher", "get_watcher", "watcher_enabled"]
