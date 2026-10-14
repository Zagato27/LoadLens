"""Masking of credentials in configuration payloads exchanged with the browser.

The UI never receives real passwords or tokens: ``GET /config`` replaces them, and passwords inside
URLs such as ``http://user:password@proxy``, with :data:`MASK`. When the UI sends a section back,
the sentinel is substituted with the value stored at the same path.

A stored secret is substituted only while the place it is sent to stays the same. If the address,
port, proxy or TLS settings next to a masked secret change, the secret has to be typed again;
otherwise anyone allowed to edit or test a connection could send a stored password to a host of
their choice.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

MASK = "***"

# Keys whose string values are credentials. ``password_env`` is included because the OpenSearch
# config stores the password itself under that (misleading) name; ``credentials`` is GigaChat's key.
SECRET_KEYS = frozenset({"password", "token", "grafana_pass", "api_key", "password_env", "credentials"})

# Fields that decide where the secrets on the same level, or nested below it, are sent.
DESTINATION_KEYS = frozenset({
    "url", "base_url", "api_base_url", "url_basic", "host", "port",
    "proxies", "verify", "verify_ssl", "sslmode", "ca_bundle",
})
# Secrets of the flat legacy section that belong to one specific address instead.
SPECIFIC_DESTINATIONS = {"grafana_pass": frozenset({"grafana_base_url"})}

_URL_WITH_PASSWORD = re.compile(r"^(?P<head>[a-z][a-z0-9+.-]*://[^/@:\s]*:)(?P<password>[^/@\s]*)(?P<tail>@.*)$", re.I | re.S)


class SecretPlaceholderError(ValueError):
    """Raised when a payload contains the mask sentinel but no stored value exists."""

    def __init__(self, path: str, message: Optional[str] = None):
        super().__init__(
            message
            or f"Секрет «{path}» не задан: замените {MASK} реальным значением или оставьте поле пустым"
        )
        self.path = path


class SecretDestinationChangedError(SecretPlaceholderError):
    """Raised when a masked secret would be sent to another address than the one it was saved for."""

    def __init__(self, path: str):
        super().__init__(
            path,
            f"Секрет «{path}» не подставлен: адрес, порт, прокси или настройки TLS подключения изменились. "
            f"Введите секрет заново вместо {MASK}",
        )


def mask_secrets(tree: Any) -> Any:
    """Returns a copy of ``tree`` with credential values and URL passwords replaced by :data:`MASK`."""
    if isinstance(tree, dict):
        masked: dict = {}
        for key, value in tree.items():
            if key in SECRET_KEYS and isinstance(value, str) and value:
                masked[key] = MASK
            else:
                masked[key] = mask_secrets(value)
        return masked
    if isinstance(tree, list):
        return [mask_secrets(item) for item in tree]
    if isinstance(tree, str):
        return _mask_url_password(tree)
    return tree


def restore_secrets(incoming: Any, current: Any, _path: str = "", _moved: bool = False) -> Any:
    """Replaces :data:`MASK` sentinels in ``incoming`` with values from ``current``.

    Raises:
        SecretPlaceholderError: a sentinel is present but ``current`` has no value there.
        SecretDestinationChangedError: the address the secret is sent to differs from the stored one.
    """
    if isinstance(incoming, str) and _is_masked_url(incoming):
        return _restore_url_password(incoming, current, _path)
    if isinstance(incoming, dict):
        stored = current if isinstance(current, dict) else {}
        moved = _moved or _destination_changed(incoming, stored, DESTINATION_KEYS)
        restored: dict = {}
        for key, value in incoming.items():
            path = f"{_path}.{key}" if _path else str(key)
            if key in SECRET_KEYS and value == MASK:
                stored_value = stored.get(key)
                if not isinstance(stored_value, str) or not stored_value:
                    raise SecretPlaceholderError(path)
                specific = SPECIFIC_DESTINATIONS.get(key)
                if specific:
                    changed = _moved or _destination_changed(incoming, stored, specific)
                else:
                    changed = moved
                if changed:
                    raise SecretDestinationChangedError(path)
                restored[key] = stored_value
            else:
                restored[key] = restore_secrets(value, stored.get(key), path, moved)
        return restored
    if isinstance(incoming, list):
        stored_list = current if isinstance(current, list) else []
        return [
            restore_secrets(item, stored_list[idx] if idx < len(stored_list) else None, f"{_path}[{idx}]", _moved)
            for idx, item in enumerate(incoming)
        ]
    return incoming


def destination_changed(incoming: Any, stored: Any) -> bool:
    """Whether an address, port, proxy or TLS field differs from the stored configuration at any depth."""
    if isinstance(incoming, dict):
        current = stored if isinstance(stored, dict) else {}
        if _destination_changed(incoming, current, DESTINATION_KEYS):
            return True
        return any(destination_changed(value, current.get(key)) for key, value in incoming.items() if isinstance(value, (dict, list)))
    if isinstance(incoming, list):
        current = stored if isinstance(stored, list) else []
        return any(destination_changed(item, current[i] if i < len(current) else None) for i, item in enumerate(incoming))
    return False


def _mask_url_password(value: str) -> str:
    match = _URL_WITH_PASSWORD.match(value)
    if not match or not match.group("password"):
        return value
    return match.group("head") + MASK + match.group("tail")


def _is_masked_url(value: str) -> bool:
    match = _URL_WITH_PASSWORD.match(value)
    return bool(match) and match.group("password") == MASK


def _url_identity(value: str) -> tuple:
    try:
        parts = urlsplit(value.strip())
        return (parts.scheme.lower(), parts.username or "", (parts.hostname or "").lower(), parts.port)
    except ValueError:
        return (value,)


def _restore_url_password(incoming: str, current: Any, path: str) -> str:
    """``scheme://user:***@host`` gets the stored password while scheme, user, host and port stay the same."""
    stored = current if isinstance(current, str) else ""
    stored_match = _URL_WITH_PASSWORD.match(stored)
    if not stored_match or not stored_match.group("password"):
        raise SecretPlaceholderError(path)
    if _url_identity(incoming) != _url_identity(stored):
        raise SecretDestinationChangedError(path)
    match = _URL_WITH_PASSWORD.match(incoming)
    return match.group("head") + stored_match.group("password") + match.group("tail")


def _destination_changed(incoming: dict, stored: dict, keys: Optional[Iterable[str]]) -> bool:
    """Whether any destination field present in ``incoming`` differs from the stored one."""
    for key in keys or ():
        if key in incoming and _normalized(incoming[key]) != _normalized(stored.get(key)):
            return True
    return False


def _normalized(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = _mask_url_password(value.strip())
        lowered = text.lower()
        return lowered if lowered in ("true", "false") else lowered.rstrip("/")
    if isinstance(value, dict):
        return tuple(sorted((str(k), _normalized(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_normalized(item) for item in value)
    return str(value)


__all__ = [
    "MASK",
    "SECRET_KEYS",
    "SecretDestinationChangedError",
    "SecretPlaceholderError",
    "destination_changed",
    "mask_secrets",
    "restore_secrets",
]
