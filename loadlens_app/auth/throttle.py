"""In-process limiter for failed login attempts."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class LoginThrottle:
    """Refuses further attempts for a key after ``max_failures`` within ``window_seconds``.

    State is per process; the per-account lockout stored in the database covers attacks spread
    over several workers or addresses.
    """

    def __init__(
        self,
        *,
        max_failures: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self._max = max(1, int(max_failures))
        self._window = float(window_seconds)
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._failures: dict[str, deque[float]] = {}

    def _prune(self, key: str, now: float) -> deque[float]:
        events = self._failures.get(key)
        if events is None:
            return deque()
        while events and now - events[0] >= self._window:
            events.popleft()
        if not events:
            del self._failures[key]
            return deque()
        return events

    def retry_after(self, key: str) -> float:
        """Seconds until another attempt is allowed; 0 when the key is not blocked."""
        with self._lock:
            now = self._clock()
            events = self._prune(key, now)
            if len(events) < self._max:
                return 0.0
            return max(0.0, self._window - (now - events[0]))

    def register_failure(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            if key not in self._failures and len(self._failures) >= self._max_keys:
                for stale in list(self._failures):
                    self._prune(stale, now)
                if len(self._failures) >= self._max_keys:
                    self._failures.clear()
            self._failures.setdefault(key, deque()).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
