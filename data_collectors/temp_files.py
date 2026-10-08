"""Location and names of the files collectors download before attaching them to Confluence."""

from __future__ import annotations

import os
import re
from pathlib import Path

TEMP_DIR = Path(__file__).resolve().parent / "temporary_files"
_UNSAFE_CHARS = re.compile(r"[^\w.-]+")


def safe_basename(raw: object, fallback: str = "file") -> str:
    """One path component made of letters, digits, ``_``, ``.`` and ``-``.

    Metric, service and placeholder names end up in file names and in the ``ri:filename`` attribute of
    the Confluence page: ``../../x`` must not leave the directory and a quote must not break the markup.
    Idempotent, so a name may pass through it more than once.
    """
    name = _UNSAFE_CHARS.sub("_", str(raw or "")).lstrip(".")
    return name or fallback


def temp_file_path(basename: object, suffix: str) -> str:
    """Absolute path of a temporary file directly inside :data:`TEMP_DIR`, created on demand."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return str(TEMP_DIR / f"{safe_basename(basename)}{suffix}")


def is_temp_file(path: object) -> bool:
    """Whether ``path`` lies directly inside :data:`TEMP_DIR` once ``..`` and links are resolved."""
    try:
        return Path(os.fspath(path)).resolve().parent == TEMP_DIR.resolve()
    except (TypeError, ValueError, OSError):
        return False
