"""Auth settings: ``CONFIG["auth"]`` from settings.py, overridden by environment variables.

The section is deliberately not editable through ``POST /config``: a stray edit must not be
able to disable authentication or leak the session key into settings_runtime.json.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .models import Role

logger = logging.getLogger(__name__)

SECRET_KEY_FILE = Path(__file__).resolve().parents[2] / ".loadlens_secret_key"
MIN_SECRET_KEY_LENGTH = 16

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _as_bool(value: Any, default: Optional[bool]) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _as_number(value: Any, default: float, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= minimum else default


def _as_names(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [part for part in value.split(",")]
    if not isinstance(value, (list, tuple)):
        return default
    names = tuple(str(item).strip().lower() for item in value if str(item).strip())
    return names or default


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool = True
    providers: tuple[str, ...] = ("local",)
    api_tokens: bool = True
    # Sliding idle lifetime and absolute cap of a browser session.
    session_hours: float = 12.0
    session_max_hours: float = 168.0
    # None: follow the request scheme (works behind a TLS proxy when trusted_proxies is set).
    cookie_secure: Optional[bool] = None
    trusted_proxies: int = 0
    password_min_length: int = 10
    # Failed logins per client IP and username within the window before attempts are refused.
    login_max_attempts: int = 5
    login_window_seconds: float = 300.0
    # Failed logins on one account (any IP) before it is locked for lockout_minutes.
    max_failed_logins: int = 10
    lockout_minutes: float = 15.0
    user_cache_seconds: float = 10.0
    token_default_days: int = 90
    token_max_days: int = 365
    default_external_role: Role = Role.VIEWER
    secret_key: str = field(default="", repr=False)
    bootstrap_username: str = ""
    bootstrap_password: str = field(default="", repr=False)


def load_auth_settings(config: Optional[Mapping[str, Any]] = None, env: Optional[Mapping[str, str]] = None) -> AuthSettings:
    """Builds settings from ``CONFIG["auth"]`` (may be absent) with environment overrides."""
    section = (config or {}).get("auth") if config else None
    section = section if isinstance(section, Mapping) else {}
    env = os.environ if env is None else env
    defaults = AuthSettings()

    try:
        external_role = Role.parse(section.get("default_external_role", defaults.default_external_role))
    except ValueError:
        external_role = defaults.default_external_role

    enabled = _as_bool(section.get("enabled"), defaults.enabled)
    if env.get("LOADLENS_AUTH_ENABLED", "").strip():
        enabled = _as_bool(env["LOADLENS_AUTH_ENABLED"], enabled)

    cookie_secure = _as_bool(section.get("cookie_secure"), None)
    if env.get("LOADLENS_COOKIE_SECURE", "").strip():
        cookie_secure = _as_bool(env["LOADLENS_COOKIE_SECURE"], cookie_secure)

    proxies = section.get("trusted_proxies", defaults.trusted_proxies)
    if env.get("LOADLENS_TRUSTED_PROXIES", "").strip():
        proxies = env["LOADLENS_TRUSTED_PROXIES"]

    return AuthSettings(
        enabled=bool(enabled),
        providers=_as_names(section.get("providers"), defaults.providers),
        api_tokens=bool(_as_bool(section.get("api_tokens"), defaults.api_tokens)),
        session_hours=_as_number(section.get("session_hours"), defaults.session_hours, 0.1),
        session_max_hours=_as_number(section.get("session_max_hours"), defaults.session_max_hours, 0.1),
        cookie_secure=cookie_secure,
        trusted_proxies=int(_as_number(proxies, defaults.trusted_proxies, 0)),
        password_min_length=int(_as_number(section.get("password_min_length"), defaults.password_min_length, 8)),
        login_max_attempts=int(_as_number(section.get("login_max_attempts"), defaults.login_max_attempts, 1)),
        login_window_seconds=_as_number(section.get("login_window_seconds"), defaults.login_window_seconds, 1),
        max_failed_logins=int(_as_number(section.get("max_failed_logins"), defaults.max_failed_logins, 1)),
        lockout_minutes=_as_number(section.get("lockout_minutes"), defaults.lockout_minutes, 0.1),
        user_cache_seconds=_as_number(section.get("user_cache_seconds"), defaults.user_cache_seconds, 0),
        token_default_days=int(_as_number(section.get("token_default_days"), defaults.token_default_days, 1)),
        token_max_days=int(_as_number(section.get("token_max_days"), defaults.token_max_days, 1)),
        default_external_role=external_role,
        secret_key=str(env.get("LOADLENS_SECRET_KEY") or section.get("secret_key") or "").strip(),
        bootstrap_username=str(env.get("LOADLENS_ADMIN_USER") or "").strip(),
        bootstrap_password=str(env.get("LOADLENS_ADMIN_PASSWORD") or ""),
    )


def resolve_secret_key(settings: AuthSettings, key_file: Optional[Path] = None) -> tuple[str, str]:
    """Returns ``(key, source)``; the source is "config", "file" or "ephemeral".

    Order: LOADLENS_SECRET_KEY / auth.secret_key, then a key file created on first start,
    then a per-process random key (sessions are lost on restart and differ between workers).
    """
    if settings.secret_key:
        if len(settings.secret_key) >= MIN_SECRET_KEY_LENGTH:
            return settings.secret_key, "config"
        logger.warning("auth: secret key is shorter than %d characters and was ignored", MIN_SECRET_KEY_LENGTH)

    path = key_file or SECRET_KEY_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if len(existing) >= MIN_SECRET_KEY_LENGTH:
            return existing, "file"
    except OSError:
        pass
    try:
        generated = secrets.token_hex(32)
        # O_EXCL: when several workers start together, exactly one creates the file and the rest read it.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(generated)
        return generated, "file"
    except FileExistsError:
        try:
            existing = path.read_text(encoding="utf-8").strip()
            if len(existing) >= MIN_SECRET_KEY_LENGTH:
                return existing, "file"
        except OSError:
            pass
    except OSError as exc:
        logger.warning("auth: cannot persist the session key to %s: %s", path, exc)

    logger.warning("auth: using an ephemeral session key; set LOADLENS_SECRET_KEY so sessions survive restarts")
    return secrets.token_hex(32), "ephemeral"
