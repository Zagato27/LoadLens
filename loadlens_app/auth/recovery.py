# SPDX-License-Identifier: Apache-2.0
"""Recovery sign-in while the storage database is down.

Users, sessions and the audit log live in the storage database, so an outage would otherwise lock
everybody out of the page where the connection is fixed. While the user store is unreachable the login
form accepts only the recovery administrator: LOADLENS_ADMIN_USER / LOADLENS_ADMIN_PASSWORD, or
``auth.recovery_user`` + ``auth.recovery_password_hash`` (LOADLENS_RECOVERY_USER /
LOADLENS_RECOVERY_PASSWORD_HASH). The session it creates lives in the signed cookie, lasts at most
SESSION_SECONDS, reaches only endpoints marked ``recovery_ok`` and ends as soon as the store answers again.
The audit log is unreachable, so every recovery action is written to the application log instead.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import time
from typing import Any, Callable, MutableMapping, Optional

from . import passwords
from .config import AuthSettings
from .throttle import LoginThrottle

logger = logging.getLogger("loadlens_app.auth.recovery")

SESSION_KEY = "recovery"
SESSION_SECONDS = 30 * 60
ACTOR_NAME = "recovery-admin"
EXTENSION_KEY = "loadlens_recovery_limiter"


def configured(settings: AuthSettings) -> bool:
    return bool(
        (settings.bootstrap_username and settings.bootstrap_password)
        or (settings.recovery_username and settings.recovery_password_hash)
    )


def _same(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def verify(settings: AuthSettings, username: str, password: str) -> bool:
    """Checks the recovery administrator; both sources are evaluated so timing does not tell which exists."""
    name = (username or "").strip().lower()
    secret = password if isinstance(password, str) else ""
    ok = False
    if settings.bootstrap_username and settings.bootstrap_password:
        same_user = _same(name, settings.bootstrap_username.strip().lower())
        same_password = _same(secret, settings.bootstrap_password)
        ok = ok or (same_user and same_password)
    if settings.recovery_username and settings.recovery_password_hash:
        same_user = _same(name, settings.recovery_username.strip().lower())
        hash_ok = passwords.verify_password(settings.recovery_password_hash, secret)
        ok = ok or (same_user and hash_ok)
    else:
        passwords.burn_time(secret)
    return ok


class RecoveryLimiter:
    """Failed recovery logins: per address and name, and for the whole process.

    Stricter than the regular throttle: there is no per-account lockout without the database.
    """

    def __init__(
        self,
        *,
        per_key: int = 5,
        overall: int = 20,
        window_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._keys = LoginThrottle(max_failures=per_key, window_seconds=window_seconds, clock=clock)
        self._all = LoginThrottle(max_failures=overall, window_seconds=window_seconds, clock=clock)

    @staticmethod
    def _key(ip: str, username: str) -> str:
        return f"{ip}|{(username or '').strip().lower()}"

    def retry_after(self, ip: str, username: str) -> float:
        return max(self._keys.retry_after(self._key(ip, username)), self._all.retry_after("*"))

    def failure(self, ip: str, username: str) -> None:
        self._keys.register_failure(self._key(ip, username))
        self._all.register_failure("*")

    def success(self, ip: str, username: str) -> None:
        self._keys.reset(self._key(ip, username))


def establish(session: MutableMapping[str, Any]) -> None:
    """A recovery session: no user id, so nothing in it can be mistaken for a regular sign-in."""
    session.clear()
    session[SESSION_KEY] = 1
    session["iat"] = int(time.time())
    session["csrf"] = secrets.token_urlsafe(32)


def is_recovery(session: MutableMapping[str, Any]) -> bool:
    return bool(session.get(SESSION_KEY))


def expired(session: MutableMapping[str, Any], now: Optional[float] = None) -> bool:
    issued = session.get("iat")
    return not isinstance(issued, int) or (now or time.time()) - issued > SESSION_SECONDS


def log(event: str, *, ip: str = "", **fields: Any) -> None:
    """Application-log line for a recovery event; the audit table is not reachable during an outage."""
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value not in (None, ""))
    logger.warning("recovery %s ip=%s %s", event, ip or "-", details)


__all__ = [
    "ACTOR_NAME",
    "EXTENSION_KEY",
    "RecoveryLimiter",
    "SESSION_KEY",
    "SESSION_SECONDS",
    "configured",
    "establish",
    "expired",
    "is_recovery",
    "log",
    "verify",
]
