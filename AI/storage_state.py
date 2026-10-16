"""Health of the storage database: a circuit breaker around every psycopg2.connect to storage.timescale.

A failed connection opens the breaker. For ``retry_seconds`` further calls fail at once with
StorageUnavailableError instead of waiting for the network; then one caller probes again while the
others still fail fast. A successful connection closes the breaker. The connect timeout keeps a single
probe short, so pages answer within seconds even when the database host does not respond.

StorageUnavailableError subclasses psycopg2.OperationalError, so existing ``except psycopg2.Error``
handlers keep working; ``is_storage_error`` tells an outage apart from other database errors.

Connections are pooled per process and per set of connection parameters (``pool_max``, ``pool_idle_sec``
in storage.timescale): ``connect()`` hands out a PooledConnection whose ``close()`` gives the session back
after a rollback, so a request reuses a warm session instead of paying for TCP, authentication and a cold
server-side catalog cache. Only new physical connections go through the breaker. See the pool section
below for the rules a caller must keep (no session-level state left on a storage connection).
"""

from __future__ import annotations

import collections
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import psycopg2
from psycopg2 import extensions as _pg_ext

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_SEC = 5
DEFAULT_RETRY_SECONDS = 10.0
# 8 gunicorn threads + the load test watcher + background jobs + a margin.
DEFAULT_POOL_MAX = 12
DEFAULT_POOL_IDLE_SEC = 300.0
MESSAGE_LIMIT = 300

KIND_TITLES: dict[str, str] = {
    "timeout": "Сервер базы не ответил вовремя",
    "refused": "Сервер базы отклонил подключение",
    "auth": "База отклонила логин или пароль",
    "dns": "Имя хоста базы не найдено",
    "database": "На сервере нет указанной базы данных",
    "ssl": "Ошибка TLS при подключении к базе",
    "other": "Не удалось подключиться к базе",
}
KIND_HINTS: dict[str, str] = {
    "timeout": "Проверьте сеть и VPN до сервера базы и что адрес и порт указаны верно.",
    "refused": "База не запущена или слушает другой порт. Проверьте, что сервер PostgreSQL работает.",
    "auth": "Проверьте пользователя и пароль базы в настройках хранилища.",
    "dns": "Проверьте имя хоста базы и DNS.",
    "database": "Проверьте имя базы данных в настройках хранилища.",
    "ssl": "Проверьте режим SSL в настройках хранилища и сертификаты сервера.",
    "other": "Подробности — в журнале приложения.",
}
# libpq and Windows socket messages; the server side may answer in Russian.
_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timeout", ("timeout expired", "timed out", "10060")),
    ("refused", ("connection refused", "10061", "actively refused")),
    ("dns", ("could not translate host name", "name or service not known", "no such host", "unknown host", "nodename nor servname")),
    ("auth", ("password authentication failed", "authentication failed", "no pg_hba.conf entry", 'role "', "проверку подлинности")),
    ("database", ("does not exist", "не существует")),
    ("ssl", ("ssl",)),
)
_SECRET_RE = re.compile(r"(password\s*=\s*)\S+", re.IGNORECASE)


class StorageUnavailableError(psycopg2.OperationalError):
    """The storage database cannot be reached (raised without a network attempt while the breaker is open)."""


def is_storage_error(exc: BaseException) -> bool:
    return isinstance(exc, StorageUnavailableError)


def classify(message: str) -> str:
    text = (message or "").lower()
    for kind, needles in _PATTERNS:
        if any(needle in text for needle in needles):
            return kind
    return "other"


def sanitize(message: str) -> str:
    """One line without passwords; libpq never echoes them, this guards against DSN-style text."""
    text = _SECRET_RE.sub(r"\1***", " ".join(str(message or "").split()))
    return text[:MESSAGE_LIMIT]


@dataclass(frozen=True)
class StorageStatus:
    # None until the first connection attempt of this process.
    available: Optional[bool]
    kind: str
    title: str
    hint: str
    # libpq message and host:port/dbname of the failed attempt: only for administrators.
    message: str
    address: str
    checked_at: Optional[float]
    retry_in: float

    def to_dict(self, details: bool) -> dict[str, Any]:
        out = {
            "available": self.available,
            "kind": self.kind,
            "title": self.title,
            "hint": self.hint,
            "checked_at": self.checked_at,
            "retry_in": round(self.retry_in, 1),
        }
        if details:
            out["message"] = self.message
            out["address"] = self.address
        return out


class StorageBreaker:
    def __init__(self, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._retry_seconds = DEFAULT_RETRY_SECONDS
        self._failed_at: Optional[float] = None
        self._kind = ""
        self._message = ""
        self._address = ""
        self._checked_at: Optional[float] = None
        self._available: Optional[bool] = None
        self._probing = False

    # ---- state -----------------------------------------------------------------------------

    def _wait_left(self) -> float:
        if self._failed_at is None:
            return 0.0
        return max(0.0, self._retry_seconds - (self._clock() - self._failed_at))

    def is_down(self) -> bool:
        with self._lock:
            return self._failed_at is not None

    def status(self) -> StorageStatus:
        with self._lock:
            if self._failed_at is None:
                kind = "ok" if self._available else "unknown"
                return StorageStatus(self._available, kind, "", "", "", self._address, self._checked_at, 0.0)
            return StorageStatus(
                available=False,
                kind=self._kind,
                title=KIND_TITLES.get(self._kind, KIND_TITLES["other"]),
                hint=KIND_HINTS.get(self._kind, KIND_HINTS["other"]),
                message=self._message,
                address=self._address,
                checked_at=self._checked_at,
                retry_in=self._wait_left(),
            )

    def allow_retry_now(self) -> None:
        """Lets the next call try the network at once (after the connection settings were changed)."""
        with self._lock:
            if self._failed_at is not None:
                self._failed_at = self._clock() - self._retry_seconds

    def reset(self) -> None:
        with self._lock:
            self._failed_at = None
            self._kind = self._message = self._address = ""
            self._checked_at = None
            self._available = None
            self._probing = False

    # ---- connecting ----------------------------------------------------------------------------

    def connect(self, opener: Callable[[], Any], *, address: str, retry_seconds: float) -> Any:
        probing = False
        with self._lock:
            self._retry_seconds = retry_seconds
            if self._failed_at is not None:
                if self._probing or self._wait_left() > 0:
                    raise StorageUnavailableError(self._message or KIND_TITLES.get(self._kind, KIND_TITLES["other"]))
                self._probing = probing = True
        try:
            conn = opener()
        except psycopg2.OperationalError as exc:
            message = sanitize(str(exc))
            self._record_failure(classify(message), message, address)
            raise StorageUnavailableError(message) from exc
        finally:
            if probing:
                with self._lock:
                    self._probing = False
        self._record_success(address)
        return conn

    def _record_failure(self, kind: str, message: str, address: str) -> None:
        with self._lock:
            first = self._failed_at is None
            self._failed_at = self._clock()
            self._kind, self._message, self._address = kind, message, address
            self._checked_at = self._wall()
            self._available = False
        # One line per real attempt: at most one per retry window, fast failures are not logged.
        log = logger.error if first else logger.warning
        log("storage unavailable (%s) at %s: %s", kind, address, message)
        # Sessions opened before the outage are most likely dead: do not hand them out after recovery.
        close_pools()

    def _record_success(self, address: str) -> None:
        with self._lock:
            recovered = self._failed_at is not None
            self._failed_at = None
            self._kind = self._message = ""
            self._address = address
            self._checked_at = self._wall()
            self._available = True
        if recovered:
            logger.warning("storage reachable again at %s", address)


BREAKER = StorageBreaker()


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


def connect(params: Mapping[str, Any], cfg: Optional[Mapping[str, Any]] = None) -> Any:
    """A storage connection: a warm session from the pool, or a new one opened through the breaker.

    ``params`` are psycopg2.connect keywords or {"dsn": ...}; ``cfg`` is storage.timescale and gives
    ``connect_timeout_sec`` (default 5), ``retry_seconds`` (default 10), ``pool_max`` (default 12, 0 turns
    the pool off) and ``pool_idle_sec`` (default 300). The caller closes the connection as before; with
    the pool on, ``close()`` returns the session to the pool (see PooledConnection).
    """
    cfg = cfg or {}
    timeout = int(_number(cfg.get("connect_timeout_sec"), DEFAULT_CONNECT_TIMEOUT_SEC, 2, 60))
    retry = _number(cfg.get("retry_seconds"), DEFAULT_RETRY_SECONDS, 1, 300)
    dsn = params.get("dsn")
    if isinstance(dsn, str) and dsn.strip():
        address = "dsn"
        pool_key: tuple = ("dsn", dsn)
        def opener():
            return psycopg2.connect(dsn, connect_timeout=timeout)
    else:
        keywords = {key: value for key, value in params.items() if key != "dsn"}
        address = f"{keywords.get('host') or 'localhost'}:{keywords.get('port') or 5432}/{keywords.get('dbname') or ''}"
        # Port 5432 and "5432" name the same server; None is the same as a missing keyword.
        pool_key = tuple(sorted((name, str(value)) for name, value in keywords.items() if value is not None))
        def opener():
            return psycopg2.connect(connect_timeout=timeout, **keywords)

    def open_new():
        conn = BREAKER.connect(opener, address=address, retry_seconds=retry)
        with _POOL_LOCK:
            _STATS["opened"] += 1
        return conn

    pool_max = int(_number(cfg.get("pool_max"), DEFAULT_POOL_MAX, 0, 100))
    if pool_max <= 0:
        return open_new()
    idle_sec = _number(cfg.get("pool_idle_sec"), DEFAULT_POOL_IDLE_SEC, 5, 3600)
    return _checkout(pool_key, pool_max, idle_sec, open_new, address)


# ---- connection pool ----------------------------------------------------------------------------------
#
# One pool per process and per connection parameters; a LIFO stack keeps the warmest session on top.
# A pooled session must come back exactly as a new one would be:
#
# * close() rolls back (a no-op without an open transaction) and switches autocommit off again; a
#   connection that is closed, broken, busy or failed (transaction status other than IDLE/INTRANS), or
#   carries set_session() characteristics, is closed for real instead.
# * Session-level state does not survive rollback: SET without LOCAL, session advisory locks, LISTEN,
#   temporary tables, prepared statements. The code base does not leave any (2026-10 audit): DDL helpers
#   only toggle autocommit, the auth DDL takes pg_advisory_xact_lock, and the run_id migration releases
#   its pg_try_advisory_lock in ``finally``. That release fails inside an aborted transaction, so an
#   exception that leaves ``with conn:`` marks the session as untrusted and close() drops it. Code that
#   changes session state on purpose ends with ``conn.discard()`` instead of ``conn.close()``.
#
# When the pool is full, a plain connection is opened and really closed by close(): callers never wait.

_POOL_LOCK = threading.Lock()
_STATS = {"opened": 0, "reused": 0, "overflow": 0, "discarded": 0}
_SWEEP_EVERY_SEC = 60.0
_REUSABLE_STATUSES = (_pg_ext.TRANSACTION_STATUS_IDLE, _pg_ext.TRANSACTION_STATUS_INTRANS)
_SESSION_CHARACTERISTICS = ("isolation_level", "readonly", "deferrable")
# Monotonic clock of the pool (tests replace it).
_clock = time.monotonic


class _Pool:
    __slots__ = ("idle", "in_use", "generation", "max_size", "idle_sec", "leaked")

    def __init__(self, max_size: int, idle_sec: float) -> None:
        self.idle: list[tuple[Any, float]] = []  # (session, returned at); the last one is the warmest
        self.in_use = 0  # pooled sessions handed out; in_use + len(idle) <= max_size
        self.generation = 0  # bumped by a flush: sessions handed out before it are closed on return
        self.max_size = max_size
        self.idle_sec = idle_sec
        # One entry per PooledConnection collected without close(); deque.append is safe from __del__.
        self.leaked: collections.deque = collections.deque()


_POOLS: dict[tuple, _Pool] = {}
_last_sweep = 0.0
# Sessions inherited through fork(): closing them would end the parent's sessions, so they are kept.
_FORK_ORPHANS: list[Any] = []


class PooledConnection:
    """A psycopg2 connection lent by the pool; everything except close() goes to the session itself.

    ``with conn:`` keeps psycopg2's meaning (commit or rollback, not close). ``close()`` returns the session
    to the pool and is idempotent; ``discard()`` closes the session for real. After either one the object
    behaves like a closed psycopg2 connection: ``closed`` is 1 and using it raises InterfaceError.
    """

    __slots__ = ("_raw", "_pool", "_generation", "_pooled", "_tainted", "_released", "__weakref__")

    def __init__(self, raw: Any, pool: _Pool, generation: int, pooled: bool) -> None:
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_generation", generation)
        object.__setattr__(self, "_pooled", pooled)
        object.__setattr__(self, "_tainted", False)
        object.__setattr__(self, "_released", False)

    def _session(self) -> Any:
        if object.__getattribute__(self, "_released"):
            raise psycopg2.InterfaceError("connection already closed")
        return object.__getattribute__(self, "_raw")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._session(), name, value)

    @property
    def closed(self) -> int:
        if self._released:
            return 1
        return getattr(self._raw, "closed", 0)

    @property
    def pooled(self) -> bool:
        """False for a connection opened past ``pool_max``: its close() closes it for real."""
        return self._pooled

    def __enter__(self) -> "PooledConnection":
        self._session().__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # The block may have skipped its own cleanup (e.g. an advisory unlock in an aborted transaction).
            object.__setattr__(self, "_tainted", True)
        return self._session().__exit__(exc_type, exc, tb)

    def close(self) -> None:
        if self._released:
            return
        object.__setattr__(self, "_released", True)
        _give_back(self._raw, self._pool, self._generation, self._pooled and not self._tainted, self._pooled)

    def discard(self) -> None:
        """Closes the physical session instead of pooling it (after changing session-level state)."""
        if self._released:
            return
        object.__setattr__(self, "_released", True)
        _give_back(self._raw, self._pool, self._generation, False, self._pooled)

    def __del__(self) -> None:
        # A forgotten close(): the session is closed by psycopg2 with the object, the slot is reclaimed later.
        try:
            if not object.__getattribute__(self, "_released") and object.__getattribute__(self, "_pooled"):
                object.__getattribute__(self, "_pool").leaked.append(1)
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._released else ("pooled" if self._pooled else "unpooled")
        return f"<PooledConnection {state} {self._raw!r}>"


def _poolable(raw: Any) -> bool:
    """Only sessions whose transaction state can be checked are pooled; anything else is returned as is."""
    return callable(getattr(raw, "get_transaction_status", None)) and callable(getattr(raw, "rollback", None))


def _close_quietly(sessions: list) -> None:
    for raw in sessions:
        try:
            raw.close()
        except Exception:
            pass


def _reclaim(pool: _Pool) -> None:
    """Under _POOL_LOCK: frees the slots of connections that were garbage-collected without close()."""
    while pool.leaked:
        pool.leaked.popleft()
        pool.in_use = max(0, pool.in_use - 1)


def _expire(pool: _Pool, now: float, stale: list) -> None:
    """Under _POOL_LOCK: moves sessions idle for longer than ``idle_sec`` (the oldest are first) to ``stale``."""
    while pool.idle and now - pool.idle[0][1] > pool.idle_sec:
        stale.append(pool.idle.pop(0)[0])
        _STATS["discarded"] += 1


def _sweep(now: float, stale: list) -> None:
    """Under _POOL_LOCK, at most once a minute: expires idle sessions of every pool and forgets empty pools
    (left behind when the connection settings change)."""
    global _last_sweep
    if now - _last_sweep < _SWEEP_EVERY_SEC:
        return
    _last_sweep = now
    for key, pool in list(_POOLS.items()):
        _reclaim(pool)
        _expire(pool, now, stale)
        if not pool.idle and pool.in_use == 0:
            del _POOLS[key]


def _flush(pool: _Pool) -> list:
    """Under _POOL_LOCK: takes every idle session out and makes the ones handed out close on return."""
    pool.generation += 1
    stale = [raw for raw, _ in pool.idle]
    pool.idle.clear()
    _STATS["discarded"] += len(stale)
    return stale


def _alive(raw: Any) -> bool:
    """One round trip; used only for a session idle for longer than half of ``pool_idle_sec``."""
    try:
        raw.autocommit = True  # no BEGIN/ROLLBACK around the probe
        try:
            with raw.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            raw.autocommit = False
        return True
    except Exception:
        return False


def _checkout(key: tuple, max_size: int, idle_sec: float, open_new: Callable[[], Any], address: str) -> Any:
    while True:
        stale: list = []
        raw = None
        age = 0.0
        with _POOL_LOCK:
            now = _clock()
            _sweep(now, stale)
            pool = _POOLS.get(key)
            if pool is None:
                pool = _POOLS[key] = _Pool(max_size, idle_sec)
            pool.max_size, pool.idle_sec = max_size, idle_sec
            _reclaim(pool)
            while pool.idle:
                candidate, since = pool.idle.pop()
                if getattr(candidate, "closed", 0) or now - since > idle_sec:
                    stale.append(candidate)
                    _STATS["discarded"] += 1
                    continue
                raw, age = candidate, now - since
                break
            pooled = raw is not None or pool.in_use < max_size
            if pooled:
                pool.in_use += 1
            generation = pool.generation
        _close_quietly(stale)

        if raw is not None:
            if age > idle_sec / 2 and not _alive(raw):
                # The server dropped this session (restart, idle timeout): the others are suspect too.
                with _POOL_LOCK:
                    pool.in_use -= 1
                    stale = _flush(pool)
                    _STATS["discarded"] += 1
                _close_quietly(stale + [raw])
                continue
            with _POOL_LOCK:
                _STATS["reused"] += 1
            return PooledConnection(raw, pool, generation, pooled=True)

        try:
            raw = open_new()
        except BaseException:
            if pooled:
                with _POOL_LOCK:
                    pool.in_use -= 1
            raise
        if not _poolable(raw):
            if pooled:
                with _POOL_LOCK:
                    pool.in_use -= 1
            return raw
        if not pooled:
            with _POOL_LOCK:
                _STATS["overflow"] += 1
            logger.debug("storage pool for %s is full (%d sessions): connection outside the pool", address, max_size)
        return PooledConnection(raw, pool, generation, pooled=pooled)


def _reset_for_reuse(raw: Any) -> tuple[bool, bool]:
    """Puts a returned session back into the state of a new one; returns (reusable, broken)."""
    try:
        if getattr(raw, "closed", 0):
            return False, getattr(raw, "closed", 0) == 2
        status = raw.get_transaction_status()
        if status == _pg_ext.TRANSACTION_STATUS_UNKNOWN:
            return False, True
        if status not in _REUSABLE_STATUSES:
            return False, False
        raw.rollback()
        if raw.autocommit:
            raw.autocommit = False
        if any(getattr(raw, name, None) is not None for name in _SESSION_CHARACTERISTICS):
            return False, False
        return True, False
    except Exception:
        return False, getattr(raw, "closed", 0) == 2


def _give_back(raw: Any, pool: _Pool, generation: int, reuse: bool, pooled: bool) -> None:
    stale: list = []
    broken = False
    if reuse:
        reuse, broken = _reset_for_reuse(raw)
    elif pooled:
        broken = getattr(raw, "closed", 0) == 2
    with _POOL_LOCK:
        if pooled:
            pool.in_use = max(0, pool.in_use - 1)
        if broken:
            # A session broke under its user (server restart, network): the idle ones are suspect too.
            stale = _flush(pool)
        now = _clock()
        _expire(pool, now, stale)
        if reuse and generation == pool.generation and pool.in_use + len(pool.idle) < pool.max_size:
            pool.idle.append((raw, now))
            raw = None
        elif pooled:
            _STATS["discarded"] += 1
    if raw is not None:
        stale.append(raw)
    _close_quietly(stale)


def close_pools() -> None:
    """Closes every idle session; sessions in use are closed when they are returned."""
    with _POOL_LOCK:
        stale = [raw for pool in _POOLS.values() for raw in _flush(pool)]
    _close_quietly(stale)


def reset_pools() -> None:
    """close_pools(), then forgets the pools and zeroes the counters (tests, settings reload)."""
    global _last_sweep
    with _POOL_LOCK:
        stale = [raw for pool in _POOLS.values() for raw in _flush(pool)]
        _POOLS.clear()
        for name in _STATS:
            _STATS[name] = 0
        _last_sweep = 0.0
    _close_quietly(stale)


def pool_stats() -> dict[str, int]:
    """Counters since the start (or reset_pools): physical connections opened, sessions reused, sessions
    handed out now and idle in the pools, connections opened past ``pool_max``, sessions closed by the pool."""
    with _POOL_LOCK:
        for pool in _POOLS.values():
            _reclaim(pool)
        return {
            "opened": _STATS["opened"],
            "reused": _STATS["reused"],
            "in_use": sum(pool.in_use for pool in _POOLS.values()),
            "idle": sum(len(pool.idle) for pool in _POOLS.values()),
            "overflow": _STATS["overflow"],
            "discarded": _STATS["discarded"],
        }


def _forget_pools_after_fork() -> None:
    """In a forked child (Celery prefork, gunicorn --preload) the parent's sessions must not be used or
    closed: both would break them for the parent. They are kept referenced and never touched."""
    global _POOL_LOCK, _POOLS, _last_sweep
    for pool in _POOLS.values():
        _FORK_ORPHANS.extend(raw for raw, _ in pool.idle)
        pool.generation += 1
        pool.idle = []
    _POOL_LOCK = threading.Lock()
    _POOLS = {}
    for name in _STATS:
        _STATS[name] = 0
    _last_sweep = 0.0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pools_after_fork)


def status() -> StorageStatus:
    return BREAKER.status()


def is_down() -> bool:
    return BREAKER.is_down()


__all__ = [
    "BREAKER",
    "KIND_HINTS",
    "KIND_TITLES",
    "PooledConnection",
    "StorageBreaker",
    "StorageStatus",
    "StorageUnavailableError",
    "classify",
    "close_pools",
    "connect",
    "is_down",
    "is_storage_error",
    "pool_stats",
    "reset_pools",
    "sanitize",
    "status",
]
