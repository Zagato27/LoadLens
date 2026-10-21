# SPDX-License-Identifier: Apache-2.0
"""Prompt files and the short glue strings of the LLM pipeline, per language (plans/i18n-plan.md, section 3.3).

Layout: ``AI/prompts/<lang>/`` holds the same file names for every language (``*_prompt.txt``, ``_context_guide.txt``,
``_response_format.txt``, ``judge_rubrics/*.txt``, ``profiles/*.txt``, ``_system*.txt``) plus ``_strings.json``: a
flat JSON object of short texts (same keys in every language, ``{param}`` placeholders).

``load_prompt(name, lang)`` reads a file once per change (keyed by language and name). A file missing in ``en`` falls
back to ``ru`` with one warning; a file missing in ``ru`` raises :class:`PromptMissing`. ``text(key, lang)`` does the
same for ``_strings.json``. ``lang=None`` means the current locale (:func:`AI.i18n.get_locale`): the pipeline runs in
``with_locale(report_language)``, so deep code follows the language of the report.

No Flask here: ``AI/`` never imports ``loadlens_app``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from AI.i18n import DEFAULT, SUPPORTED, get_locale, interpolate
from AI.i18n import normalize_lang as _normalize_lang

logger = logging.getLogger(__name__)

PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"
STRINGS_FILE = "_strings.json"

# Prompts the settings page shows and the run may override (domain -> file); judge and critic are read-only.
DOMAIN_PROMPT_FILES: Dict[str, str] = {
    "overall": "overall_prompt.txt",
    "database": "database_prompt.txt",
    "kafka": "kafka_prompt.txt",
    "microservices": "microservices_prompt.txt",
    "jvm": "jvm_prompt.txt",
    "hard_resources": "hard_resources_prompt.txt",
    "lt_framework": "lt_framework_prompt.txt",
    "lt_generators": "lt_generators_prompt.txt",
    "application_logs": "application_logs_prompt.txt",
    "judge": "judge_prompt.txt",
    "critic": "critic_prompt.txt",
}
TEST_PROFILES = ("step", "soak", "spike", "stress", "none")

# Words of a script: letters only, so identifiers with digits or underscores do not count (see _words).
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TECH_TOKEN_RE = re.compile(r"[0-9_./=\\{}\[\]<>|@#$%^&*+]")
_ALL_CAPS_RE = re.compile(r"^[A-Z]{2,}s?$")
# Punctuation around a word (a sentence-final dot included); dots inside a token mark an identifier.
_TOKEN_STRIP = "\"'«»“”„()[]{}<>,;:!?…—–-."


class PromptMissing(FileNotFoundError):
    """A prompt file (or a ``_strings.json`` key) exists in no language, not even in ``ru``."""


def normalize_lang(value: Any) -> Optional[str]:
    """``"ru"``/``"en"`` for a supported code (``"EN"``, ``"en-GB"`` included), else ``None``."""
    return _normalize_lang(value)


def resolve_lang(lang: Any = None) -> str:
    """A supported language: ``lang`` when valid, else the current locale (``with_locale`` scope, then the setting)."""
    return _normalize_lang(lang) or get_locale()


def prompt_dir(lang: Any = None) -> Path:
    """``AI/prompts/<lang>/``."""
    return PROMPTS_ROOT / resolve_lang(lang)


# ---- files -------------------------------------------------------------------------------------------------

_lock = threading.Lock()
# (lang, name) -> (path actually read, mtime_ns, text)
_cache: Dict[Tuple[str, str], Tuple[Path, int, str]] = {}
_strings_cache: Dict[str, Tuple[int, Dict[str, str]]] = {}
_warned: set[Tuple[str, str]] = set()


def _warn_once(lang: str, name: str, what: str) -> None:
    marker = (lang, name)
    with _lock:
        if marker in _warned:
            return
        _warned.add(marker)
    logger.warning("prompts: %s %r is missing for language %r, using the %r one", what, name, lang, DEFAULT)


def _mtime(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _clean_name(name: str) -> str:
    clean = str(name or "").replace("\\", "/").strip("/")
    if not clean or ".." in clean.split("/"):
        raise PromptMissing(f"invalid prompt name {name!r}")
    return clean


def load_prompt(name: str, lang: Any = None) -> str:
    """Text of ``AI/prompts/<lang>/<name>``; the ``ru`` file with one warning when the language has none.

    Read in text mode (universal newlines), cached per (language, name) until the file changes.
    Raises :class:`PromptMissing` when the file is missing in ``ru`` too.
    """
    code = resolve_lang(lang)
    clean = _clean_name(name)
    key = (code, clean)
    with _lock:
        cached = _cache.get(key)
    if cached is not None and _mtime(cached[0]) == cached[1]:
        return cached[2]
    path = PROMPTS_ROOT / code / clean
    mtime = _mtime(path)
    if mtime is None and code != DEFAULT:
        path = PROMPTS_ROOT / DEFAULT / clean
        mtime = _mtime(path)
        if mtime is not None:
            _warn_once(code, clean, "prompt file")
    if mtime is None:
        raise PromptMissing(f"prompt file {clean!r} is missing in AI/prompts/{code} and AI/prompts/{DEFAULT}")
    text = path.read_text(encoding="utf-8")
    with _lock:
        _cache[key] = (path, mtime, text)
    return text


def has_prompt(name: str, lang: Any = None) -> bool:
    """The file exists for the language itself (no fallback)."""
    return (PROMPTS_ROOT / resolve_lang(lang) / _clean_name(name)).is_file()


def prompt_files(lang: Any = None) -> list[str]:
    """Relative names (with ``/``) of every file of the language folder, sorted."""
    folder = prompt_dir(lang)
    if not folder.is_dir():
        return []
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


def reload_prompts() -> None:
    """Forgets the cached files, strings and warnings (tests)."""
    with _lock:
        _cache.clear()
        _strings_cache.clear()
        _warned.clear()


# ---- glue strings ------------------------------------------------------------------------------------------


def _strings(code: str) -> Dict[str, str]:
    path = PROMPTS_ROOT / code / STRINGS_FILE
    mtime = _mtime(path)
    if mtime is None:
        return {}
    with _lock:
        cached = _strings_cache.get(code)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"{path}: a flat JSON object of strings is expected")
    with _lock:
        _strings_cache[code] = (mtime, data)
    return data


def strings(lang: Any = None) -> Dict[str, str]:
    """All keys of ``_strings.json`` of the language (no fallback), for tests and tools."""
    return dict(_strings(resolve_lang(lang)))


def text(key: str, lang: Any = None, /, **params: Any) -> str:
    """Glue string ``key`` of ``_strings.json`` with ``{param}`` filled; ``ru`` with one warning when missing.

    Unknown placeholders stay as they are. Raises :class:`PromptMissing` when ``ru`` has no such key either.
    """
    code = resolve_lang(lang)
    raw = _strings(code).get(key)
    if raw is None and code != DEFAULT:
        raw = _strings(DEFAULT).get(key)
        if raw is not None:
            _warn_once(code, key, "prompt string")
    if raw is None:
        raise PromptMissing(f"prompt string {key!r} is missing in {code} and {DEFAULT}")
    return interpolate(raw, params) if params else raw


def texts_in_all_languages(key: str) -> set[str]:
    """Values of ``key`` in every language that has it (to recognise a stored text of any language)."""
    return {value for value in (_strings(code).get(key) for code in SUPPORTED) if value is not None}


# ---- defaults and overrides --------------------------------------------------------------------------------


def default_prompts(lang: Any = None, domains: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Default text of each editable prompt domain in ``lang`` (``""`` when no file exists)."""
    code = resolve_lang(lang)
    out: Dict[str, str] = {}
    for domain in domains if domains is not None else DOMAIN_PROMPT_FILES:
        fname = DOMAIN_PROMPT_FILES.get(domain)
        if not fname:
            continue
        try:
            out[domain] = load_prompt(fname, code)
        except PromptMissing:
            out[domain] = ""
    return out


def _comparable(value: str) -> str:
    return "\n".join(line.rstrip() for line in str(value).replace("\r\n", "\n").replace("\r", "\n").strip().split("\n"))


def is_default_text(value: Any, name: str) -> bool:
    """``value`` is the default text of prompt ``name`` (domain key or file name) in any language."""
    fname = DOMAIN_PROMPT_FILES.get(name, name)
    probe = _comparable(value)
    for code in SUPPORTED:
        if not has_prompt(fname, code):
            continue
        if _comparable(load_prompt(fname, code)) == probe:
            return True
    return False


def effective_override(value: Any, name: str) -> Optional[str]:
    """The override text to use, or ``None``: an empty text and a copy of a default (any language) are no override.

    ``name`` is a domain key (``jvm``) or a file name (``jvm_prompt.txt``).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    if is_default_text(value, name):
        return None
    return value


# ---- language of a text ------------------------------------------------------------------------------------


def _words(value: Any) -> Tuple[int, int]:
    """(Cyrillic words, Latin words) of natural text: tokens with digits, ``_``, dots or symbols and ALL-CAPS
    abbreviations (CPU, RPS) are identifiers or units and do not count."""
    cyrillic = latin = 0
    for token in str(value or "").split():
        token = token.strip(_TOKEN_STRIP)
        if not token or _TECH_TOKEN_RE.search(token):
            continue
        if _CYRILLIC_RE.search(token):
            cyrillic += 1
        elif _LATIN_RE.search(token) and not _ALL_CAPS_RE.match(token):
            latin += 1
    return cyrillic, latin


def script_share(values: Iterable[Any], lang: Any) -> Optional[float]:
    """Share of the words written in the script of ``lang`` among the natural words of ``values``; ``None``
    when there are no such words."""
    cyrillic = latin = 0
    for value in values:
        c, l = _words(value)
        cyrillic += c
        latin += l
    total = cyrillic + latin
    if not total:
        return None
    own = cyrillic if resolve_lang(lang) == "ru" else latin
    return own / total


def detect_language(value: Any) -> Optional[str]:
    """``ru`` or ``en`` by the share of Cyrillic words in a text (identifiers ignored); ``None`` without words."""
    cyrillic, latin = _words(value)
    if not cyrillic and not latin:
        return None
    return "ru" if cyrillic / (cyrillic + latin) >= 0.3 else "en"


__all__ = [
    "DOMAIN_PROMPT_FILES",
    "PROMPTS_ROOT",
    "PromptMissing",
    "SUPPORTED",
    "TEST_PROFILES",
    "default_prompts",
    "detect_language",
    "effective_override",
    "has_prompt",
    "is_default_text",
    "load_prompt",
    "normalize_lang",
    "prompt_dir",
    "prompt_files",
    "reload_prompts",
    "resolve_lang",
    "script_share",
    "strings",
    "text",
    "texts_in_all_languages",
]
