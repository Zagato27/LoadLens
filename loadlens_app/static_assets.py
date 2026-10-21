# SPDX-License-Identifier: Apache-2.0
"""Versioned URLs of the static files, so browsers may keep them for a year.

``static_url("js/ui.js")`` gives ``/static/js/ui.js?v=<first 10 hex digits of the sha1 of the file>``: a new
deployment with a changed file changes its URL, and the long ``Cache-Control: max-age`` of the static
endpoint (``SEND_FILE_MAX_AGE_DEFAULT`` in ``app.py``) never serves an old copy. Templates call
``url_for('static', filename=...)``; :func:`add_static_version` (an app ``url_defaults`` hook) adds the
same ``v`` there, so an application built without the hook still renders plain ``/static/...`` links.

The digest is computed once per file and process. ``LOADLENS_STATIC_NO_CACHE=1`` (development) recomputes
it when the file's modification time or size changes and turns the long browser cache off.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

STATIC_ROOT = Path(__file__).resolve().parent.parent / "static"
STATIC_MAX_AGE_SEC = 31536000
VERSION_CHARS = 10
VERSION_ARG = "v"


def dev_mode() -> bool:
    """``LOADLENS_STATIC_NO_CACHE=1``: edited files get a new version without a restart, no long cache."""
    return os.getenv("LOADLENS_STATIC_NO_CACHE", "").strip() == "1"


def _static_root() -> Path:
    try:
        from flask import current_app

        folder = current_app.static_folder
        if folder:
            return Path(folder)
    except RuntimeError:  # outside an application context
        pass
    return STATIC_ROOT


def _resolve(root: Path, path: str) -> Optional[Path]:
    """The file under ``root``; ``None`` for a missing file or a path that leaves the directory."""
    try:
        base = root.resolve()
        target = (base / path).resolve()
    except OSError:
        return None
    if base != target and base not in target.parents:
        return None
    return target if target.is_file() else None


@lru_cache(maxsize=1024)
def _digest(file_path: str, signature: tuple) -> str:
    """sha1 prefix of the file contents; ``signature`` (mtime, size in dev mode) keys the cache."""
    digest = hashlib.sha1()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()[:VERSION_CHARS]


@lru_cache(maxsize=1024)
def _cached_version(root: str, path: str) -> str:
    target = _resolve(Path(root), path)
    return _digest(str(target), ()) if target is not None else ""


def static_version(path: str) -> str:
    """Version of ``static/<path>``; ``""`` when the file does not exist."""
    clean = _clean(path)
    root = _static_root()
    if not dev_mode():
        return _cached_version(str(root), clean)
    target = _resolve(root, clean)
    if target is None:
        return ""
    stat = target.stat()
    return _digest(str(target), (stat.st_mtime_ns, stat.st_size))


def static_url(path: str) -> str:
    """``/static/<path>?v=<version>`` (without ``v`` for a file that does not exist)."""
    clean = _clean(path)
    version = static_version(clean)
    return f"/static/{clean}?{VERSION_ARG}={version}" if version else f"/static/{clean}"


def add_static_version(endpoint: Optional[str], values: dict) -> None:
    """``url_defaults`` hook: ``url_for('static', filename=...)`` gets the ``v`` argument of the file."""
    if endpoint != "static" or VERSION_ARG in values:
        return
    filename = values.get("filename")
    if not filename:
        return
    version = static_version(str(filename))
    if version:
        values[VERSION_ARG] = version


def static_max_age() -> Optional[int]:
    """``SEND_FILE_MAX_AGE_DEFAULT`` of the app: a year, or conditional requests only in dev mode."""
    return None if dev_mode() else STATIC_MAX_AGE_SEC


def clear_cache() -> None:
    """Forgets computed versions (tests)."""
    _cached_version.cache_clear()
    _digest.cache_clear()


def _clean(path: str) -> str:
    clean = str(path or "").replace("\\", "/").lstrip("/")
    return clean[len("static/"):] if clean.startswith("static/") else clean


__all__ = [
    "STATIC_MAX_AGE_SEC",
    "add_static_version",
    "clear_cache",
    "dev_mode",
    "static_max_age",
    "static_url",
    "static_version",
]
