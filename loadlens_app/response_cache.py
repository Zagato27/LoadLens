"""Short-lived cache of the JSON answers the pages poll (dashboard, job list, active load test runs).

Every open tab polls ``/dashboard_data``, ``/jobs`` and ``/load_tests/active`` every few seconds; within the
TTL the tabs share one answer instead of each running the same queries. Concurrent misses of one key wait
for the request already computing it (single flight), so N tabs whose polls meet still cost one query.

The cache lives in the process. The TTL bounds how stale an answer can get against changes made elsewhere
(another process, the load test watcher); writes in this process invalidate the affected prefix at once
(:func:`invalidate`). Only ``200`` JSON answers are stored, as bytes: every hit gets a fresh response object
and passes the request hooks (access control, security headers) like an uncached answer.

``LOADLENS_RESPONSE_CACHE=0`` switches the cache off (diagnostics, uncached measurements).
"""

from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

from flask import Response, current_app

# Prefixes (first key element) and TTLs of the cached routes.
DASHBOARD = "dashboard"
JOBS = "jobs"
ACTIVE = "active"
DASHBOARD_TTL_SEC = 10.0
JOBS_TTL_SEC = 3.0
ACTIVE_TTL_SEC = 3.0

MAX_ENTRIES = 256
# How long a request waits for a concurrent one computing the same key before computing it itself.
FOLLOWER_WAIT_SEC = 30.0

_MISSING = object()

Key = Tuple[Hashable, ...]


class TTLCache:
    """Thread-safe ``key -> value`` store with a TTL per entry and invalidation by key prefix.

    Keys are tuples whose first element names the prefix (``("jobs", ...)``). Expired entries are dropped
    when a new one is stored; above ``max_entries`` the oldest entries go first.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES, clock: Callable[[], float] = time.monotonic,
                 follower_wait: float = FOLLOWER_WAIT_SEC) -> None:
        self._max_entries = max(1, int(max_entries))
        self._clock = clock
        self._follower_wait = follower_wait
        self._lock = threading.Lock()
        self._entries: Dict[Key, Tuple[float, Any]] = {}
        self._inflight: Dict[Key, threading.Event] = {}
        # Bumped by invalidate(): a value computed before an invalidation of its prefix is not stored.
        self._generations: Dict[Hashable, int] = {}
        self._epoch = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: Key, default: Any = None) -> Any:
        """The live value of ``key`` or ``default``."""
        with self._lock:
            value = self._live_locked(key)
        return default if value is _MISSING else value

    def set(self, key: Key, value: Any, ttl: float) -> None:
        with self._lock:
            self._store_locked(key, value, ttl)

    def get_or_set(self, key: Key, ttl: float, fn: Callable[[], Any],
                   cacheable: Optional[Callable[[Any], bool]] = None) -> Any:
        """The cached value of ``key``, else ``fn()``, stored for ``ttl`` seconds when ``cacheable(value)``.

        One caller computes a missing key at a time; the others wait for it (at most ``follower_wait``
        seconds) and take its value, or compute their own when it was not cacheable or failed.
        """
        if ttl <= 0:
            return fn()
        flight: Optional[threading.Event] = None
        marks: Tuple[int, int] = (0, 0)
        for attempt in (0, 1):
            with self._lock:
                value = self._live_locked(key)
                if value is not _MISSING:
                    self.hits += 1
                    return value
                running = self._inflight.get(key)
                if running is None:
                    flight = self._inflight[key] = threading.Event()
                    marks = self._marks_locked(key)
                    self.misses += 1
                    break
            if attempt == 0:
                running.wait(self._follower_wait)
        else:
            # The computing request is still busy after the wait: do not queue behind it any longer.
            with self._lock:
                self.misses += 1
            return fn()
        try:
            value = fn()
            if cacheable is None or cacheable(value):
                with self._lock:
                    if self._marks_locked(key) == marks:
                        self._store_locked(key, value, ttl)
            return value
        finally:
            with self._lock:
                if self._inflight.get(key) is flight:
                    del self._inflight[key]
            flight.set()

    def invalidate(self, prefix: Any = None) -> int:
        """Drops the entries of a prefix (``"jobs"`` or a leading part of the key tuple); ``None`` drops all.

        Returns how many entries were dropped. Values being computed right now are not stored afterwards.
        """
        if isinstance(prefix, tuple):
            head = prefix[:1]
            matches = lambda key: key[:len(prefix)] == prefix  # noqa: E731
        else:
            head = (prefix,)
            matches = lambda key: bool(key) and key[0] == prefix  # noqa: E731
        with self._lock:
            if prefix is None or head == ():
                dropped = len(self._entries)
                self._entries.clear()
                self._epoch += 1
                return dropped
            stale = [key for key in self._entries if matches(key)]
            for key in stale:
                del self._entries[key]
            self._generations[head[0]] = self._generations.get(head[0], 0) + 1
            return len(stale)

    def clear(self) -> None:
        """Drops everything and resets the counters (tests)."""
        with self._lock:
            self._entries.clear()
            self._epoch += 1
            self.hits = 0
            self.misses = 0

    # ---- internals ------------------------------------------------------------------------------

    def _marks_locked(self, key: Key) -> Tuple[int, int]:
        return self._epoch, self._generations.get(key[0] if key else None, 0)

    def _live_locked(self, key: Key) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            return _MISSING
        if entry[0] <= self._clock():
            del self._entries[key]
            return _MISSING
        return entry[1]

    def _store_locked(self, key: Key, value: Any, ttl: float) -> None:
        now = self._clock()
        for stale in [k for k, (expires, _) in self._entries.items() if expires <= now]:
            del self._entries[stale]
        self._entries.pop(key, None)
        while len(self._entries) >= self._max_entries:
            del self._entries[next(iter(self._entries))]
        self._entries[key] = (now + ttl, value)


response_cache = TTLCache()


def enabled() -> bool:
    return os.getenv("LOADLENS_RESPONSE_CACHE", "1").strip() != "0"


def invalidate(*prefixes: Any) -> None:
    """Drops the cached answers of these prefixes (``invalidate("jobs", "dashboard")``)."""
    for prefix in prefixes:
        response_cache.invalidate(prefix)


def cached_json(prefix: str, ttl: float, key: Callable[[], Optional[tuple]]):
    """Route decorator: serves ``200`` JSON answers of the view from :data:`response_cache` for ``ttl`` s.

    ``key()`` runs in the request and returns the tuple the answer depends on (area services, query
    arguments, user); ``None`` or an exception skips the cache, so the view answers (and reports bad input)
    itself. The view body is unchanged: its answer is stored as bytes, a hit returns a new response.
    """
    def decorate(view):
        @functools.wraps(view)
        def wrapper(*args, **kwargs):
            if not enabled():
                return view(*args, **kwargs)
            try:
                parts = key()
            except Exception:
                parts = None
            if parts is None:
                return view(*args, **kwargs)
            produced: list = []

            def compute():
                response = current_app.make_response(view(*args, **kwargs))
                produced.append(response)
                if response.status_code == 200 and response.mimetype == "application/json" and not response.direct_passthrough:
                    return response.get_data()
                return None

            body = response_cache.get_or_set((prefix, *parts), ttl, compute, cacheable=lambda data: data is not None)
            if produced:
                return produced[0]
            return Response(body, status=200, mimetype="application/json")

        return wrapper

    return decorate


__all__ = [
    "ACTIVE",
    "ACTIVE_TTL_SEC",
    "DASHBOARD",
    "DASHBOARD_TTL_SEC",
    "JOBS",
    "JOBS_TTL_SEC",
    "MAX_ENTRIES",
    "TTLCache",
    "cached_json",
    "enabled",
    "invalidate",
    "response_cache",
]
