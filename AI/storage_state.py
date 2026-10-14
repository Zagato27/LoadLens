"""Health of the storage database: a circuit breaker around every psycopg2.connect to storage.timescale.

A failed connection opens the breaker. For ``retry_seconds`` further calls fail at once with
StorageUnavailableError instead of waiting for the network; then one caller probes again while the
others still fail fast. A successful connection closes the breaker. The connect timeout keeps a single
probe short, so pages answer within seconds even when the database host does not respond.

StorageUnavailableError subclasses psycopg2.OperationalError, so existing ``except psycopg2.Error``
handlers keep working; ``is_storage_error`` tells an outage apart from other database errors.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import psycopg2

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_SEC = 5
DEFAULT_RETRY_SECONDS = 10.0
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
    """Opens a psycopg2 connection through the breaker.

    ``params`` are psycopg2.connect keywords or {"dsn": ...}; ``cfg`` is storage.timescale and gives
    ``connect_timeout_sec`` (default 5) and ``retry_seconds`` (default 10).
    """
    cfg = cfg or {}
    timeout = int(_number(cfg.get("connect_timeout_sec"), DEFAULT_CONNECT_TIMEOUT_SEC, 2, 60))
    retry = _number(cfg.get("retry_seconds"), DEFAULT_RETRY_SECONDS, 1, 300)
    dsn = params.get("dsn")
    if isinstance(dsn, str) and dsn.strip():
        address = "dsn"
        def opener():
            return psycopg2.connect(dsn, connect_timeout=timeout)
    else:
        keywords = {key: value for key, value in params.items() if key != "dsn"}
        address = f"{keywords.get('host') or 'localhost'}:{keywords.get('port') or 5432}/{keywords.get('dbname') or ''}"
        def opener():
            return psycopg2.connect(connect_timeout=timeout, **keywords)
    return BREAKER.connect(opener, address=address, retry_seconds=retry)


def status() -> StorageStatus:
    return BREAKER.status()


def is_down() -> bool:
    return BREAKER.is_down()


__all__ = [
    "BREAKER",
    "KIND_HINTS",
    "KIND_TITLES",
    "StorageBreaker",
    "StorageStatus",
    "StorageUnavailableError",
    "classify",
    "connect",
    "is_down",
    "is_storage_error",
    "sanitize",
    "status",
]
