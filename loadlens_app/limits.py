# SPDX-License-Identifier: Apache-2.0
"""Limits that keep one user from exhausting the server or the LLM budget.

Values come from ``CONFIG["limits"]`` (see settings.example.py); ``None`` switches a limit off.
Counters live in the process: with several worker processes each one enforces the limits on its own.
"""

from __future__ import annotations

import re
import threading
from datetime import date
from typing import Any, Callable, Optional

from AI.i18n import has_key, t

DEFAULTS: dict[str, Optional[int]] = {
    # Longest test window one report may cover.
    "max_report_window_hours": 168,
    # Window divided by the query step; Prometheus refuses more than 11000 points per series.
    "max_points_per_series": 11000,
    # Background jobs (reports, Confluence publishing) running at the same time.
    "max_parallel_jobs": 4,
    "max_parallel_jobs_per_user": 2,
    # Calls that spend the LLM budget, per user and calendar day.
    "llm_reports_per_user_per_day": 20,
    "llm_explanations_per_user_per_day": 50,
    # System context imported from a document (one model call each).
    "llm_context_imports_per_user_per_day": 20,
    # Load test runs started from the UI that are still active (running, waiting for metrics, reporting).
    "max_parallel_load_tests": 5,
    "max_parallel_load_tests_per_user": 3,
    # Instances in one run and the longest run: the generator stops the tool after this time anyway.
    "max_load_test_instances": 20,
    "max_load_test_hours": 24,
}

_STEP = re.compile(r"^\s*(\d+)\s*(s|sec|second|seconds|m|min|t|minute|minutes|h|hour|hours|d|day|days)\s*$", re.I)
_STEP_SECONDS = {"s": 1, "m": 60, "t": 60, "h": 3600, "d": 86400}


class LimitError(Exception):
    """A request exceeds a limit; the message is safe to show to the user.

    The message is rendered when the error is raised (the locale of the request). ``key``/``params`` keep the
    dictionary key, so code without a request (the load test watcher) can store :meth:`ref` and render it again
    in the reader's locale.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        code: str = "limit_exceeded",
        *,
        key: str = "",
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.key = key
        self.params = dict(params or {})

    @classmethod
    def localized(cls, key: str, status_code: int = 429, code: str = "limit_exceeded", **params: Any) -> "LimitError":
        """The error with the text of ``key`` in the current locale."""
        return cls(t(key, **params), status_code, code, key=key, params=params)

    def ref(self) -> Optional[dict[str, Any]]:
        """``{"key", "params"}`` for :func:`AI.i18n.render_ref`, ``None`` for an error built from plain text."""
        return {"key": self.key, "params": dict(self.params)} if self.key else None


def limit(name: str, config: Optional[dict] = None) -> Optional[int]:
    """Configured value of a limit, the default when it is not set, ``None`` when switched off."""
    if config is None:
        from settings import CONFIG

        config = CONFIG
    section = config.get("limits") if isinstance(config.get("limits"), dict) else {}
    raw = section[name] if name in section else DEFAULTS.get(name)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULTS.get(name)
    return value if value >= 0 else None


def step_seconds(step: Any) -> Optional[int]:
    """``"30s"``, ``"1m"``, ``"10T"``, ``"1h"`` as seconds; ``None`` for anything else."""
    match = _STEP.match(str(step or ""))
    if not match:
        return None
    seconds = int(match.group(1)) * _STEP_SECONDS[match.group(2)[0].lower()]
    return seconds or None


def check_report_window(start_ms: int, end_ms: int, step: Any, config: Optional[dict] = None) -> None:
    """Rejects windows longer than allowed or ones that would return too many points per series."""
    hours = (end_ms - start_ms) / 3_600_000
    max_hours = limit("max_report_window_hours", config)
    if max_hours is not None and hours > max_hours:
        raise LimitError.localized(
            "server.limits.window_too_long", 400, "window_too_long", hours=f"{hours:.1f}", max_hours=max_hours
        )
    seconds = step_seconds(step)
    max_points = limit("max_points_per_series", config)
    if seconds and max_points is not None:
        points = int((end_ms - start_ms) / 1000 / seconds) + 1
        if points > max_points:
            raise LimitError.localized(
                "server.limits.too_many_points", 400, "too_many_points",
                step=str(step), points=points, max_points=max_points,
            )


def check_job_slot(running_total: int, running_for_owner: int, config: Optional[dict] = None) -> None:
    """Refuses a new background job when the process or the user already runs too many."""
    total = limit("max_parallel_jobs", config)
    if total is not None and running_total >= total:
        raise LimitError.localized("server.limits.jobs_total", running=running_total)
    per_owner = limit("max_parallel_jobs_per_user", config)
    if per_owner is not None and running_for_owner >= per_owner:
        raise LimitError.localized("server.limits.jobs_owner", running=running_for_owner)


def check_load_test_slot(running_total: int, running_for_owner: int, config: Optional[dict] = None) -> None:
    """Refuses a new load test run when too many runs are active on the server or for the user."""
    total = limit("max_parallel_load_tests", config)
    if total is not None and running_total >= total:
        raise LimitError.localized("server.limits.load_tests_total", running=running_total, allowed=total)
    per_owner = limit("max_parallel_load_tests_per_user", config)
    if per_owner is not None and running_for_owner >= per_owner:
        raise LimitError.localized("server.limits.load_tests_owner", running=running_for_owner, allowed=per_owner)


class DailyBudget:
    """Counts uses per user and kind within the current calendar day."""

    def __init__(self, today: Callable[[], date] = date.today) -> None:
        self._today = today
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = {}
        self._day: Optional[date] = None

    def consume(self, owner: str, kind: str, allowed: Optional[int], what: str = "server.limits.what.llm_runs") -> None:
        """Counts one use or raises :class:`LimitError` when the day's allowance is spent.

        ``what`` names the counted uses: a dictionary key (rendered in the locale) or plain text.
        """
        if allowed is None:
            return
        with self._lock:
            today = self._today()
            if self._day != today:
                self._day, self._counts = today, {}
            used = self._counts.get((owner, kind), 0)
            if used >= allowed:
                label = {"key": what, "params": {}} if has_key(what, "ru") else what
                raise LimitError.localized("server.limits.daily", what=label, allowed=allowed)
            self._counts[(owner, kind)] = used + 1

    def refund(self, owner: str, kind: str) -> None:
        with self._lock:
            used = self._counts.get((owner, kind), 0)
            if used:
                self._counts[(owner, kind)] = used - 1


llm_budget = DailyBudget()


_LLM_LIMITS = {
    "report": ("llm_reports_per_user_per_day", "server.limits.what.llm_runs"),
    "explanation": ("llm_explanations_per_user_per_day", "server.limits.what.llm_runs"),
    "context_import": ("llm_context_imports_per_user_per_day", "server.limits.what.context_imports"),
}


def consume_llm(owner: str, kind: str, config: Optional[dict] = None) -> None:
    """``kind`` is ``"report"``, ``"explanation"`` or ``"context_import"``."""
    name, what = _LLM_LIMITS[kind]
    llm_budget.consume(owner, kind, limit(name, config), what)


def refund_llm(owner: str, kind: str) -> None:
    """Gives back one use when the model call failed before producing anything."""
    llm_budget.refund(owner, kind)


__all__ = [
    "DEFAULTS",
    "DailyBudget",
    "LimitError",
    "check_job_slot",
    "check_load_test_slot",
    "check_report_window",
    "consume_llm",
    "refund_llm",
    "limit",
    "llm_budget",
    "step_seconds",
]
