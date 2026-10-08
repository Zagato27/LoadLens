"""Masking of credentials in configuration payloads exchanged with the browser.

The UI never receives real passwords or tokens: ``GET /config`` replaces them
with :data:`MASK`, and when the UI sends a section back the sentinel is
substituted with the currently stored value at the same path.
"""

from __future__ import annotations

from typing import Any

MASK = "***"

# Keys whose string values are credentials. ``password_env`` is included because
# the OpenSearch config stores the password itself under that (misleading) name.
SECRET_KEYS = frozenset({"password", "token", "grafana_pass", "api_key", "password_env"})


class SecretPlaceholderError(ValueError):
    """Raised when a payload contains the mask sentinel but no stored value exists."""

    def __init__(self, path: str):
        super().__init__(
            f"Секрет «{path}» не задан: замените {MASK} реальным значением или оставьте поле пустым"
        )
        self.path = path


def mask_secrets(tree: Any) -> Any:
    """Returns a copy of ``tree`` with credential values replaced by :data:`MASK`."""
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
    return tree


def restore_secrets(incoming: Any, current: Any, _path: str = "") -> Any:
    """Replaces :data:`MASK` sentinels in ``incoming`` with values from ``current``.

    Raises:
        SecretPlaceholderError: a sentinel is present but ``current`` has no value there.
    """
    if isinstance(incoming, dict):
        stored = current if isinstance(current, dict) else {}
        restored: dict = {}
        for key, value in incoming.items():
            path = f"{_path}.{key}" if _path else str(key)
            if key in SECRET_KEYS and value == MASK:
                stored_value = stored.get(key)
                if not isinstance(stored_value, str) or not stored_value:
                    raise SecretPlaceholderError(path)
                restored[key] = stored_value
            else:
                restored[key] = restore_secrets(value, stored.get(key), path)
        return restored
    if isinstance(incoming, list):
        stored_list = current if isinstance(current, list) else []
        return [
            restore_secrets(item, stored_list[idx] if idx < len(stored_list) else None, f"{_path}[{idx}]")
            for idx, item in enumerate(incoming)
        ]
    return incoming


__all__ = ["MASK", "SECRET_KEYS", "SecretPlaceholderError", "mask_secrets", "restore_secrets"]
