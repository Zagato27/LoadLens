# SPDX-License-Identifier: Apache-2.0
"""Message catalogs, the active locale and locale-aware formatting (plans/i18n-plan.md, sections 2.3 and 4.3).

No Flask here: ``AI/`` never imports ``loadlens_app``. The Flask layer (``loadlens_app/i18n.py``) registers a
request-aware resolver with :func:`set_locale_resolver`, so code below a request follows the UI language.

Catalogs live in ``locales/<lang>/<ns>.json``: one flat JSON object per namespace whose keys all start with
``<ns>.``. A broken file (bad UTF-8, a BOM, invalid JSON, a duplicate key, a key outside the namespace, a value
that is not a string) raises :class:`CatalogError` on first use instead of rendering half a page.

Locale of a call: explicit ``locale=`` -> :func:`with_locale` (ContextVar) -> registered resolver (the request
cookie, then the ``appearance.language`` setting) -> ``ru``. A ContextVar does not cross into
``ThreadPoolExecutor`` or ``threading.Thread`` by itself: submit the work with ``contextvars.copy_context().run``.

Messages use ``{name}`` placeholders (identifiers only). A placeholder without a parameter stays as it is.
Plural families carry the category as the last key segment: ``.one/.few/.many`` in ru, ``.one/.other`` in en,
with the count passed as ``n``. A missing key falls back to ru, then to the key itself, with one warning per
(locale, key); ``LOADLENS_I18N_STRICT=1`` (the test suite) raises ``KeyError`` instead.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import logging
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

SUPPORTED = ("ru", "en")
DEFAULT = "ru"
# Intl locales of the browser side: en-GB keeps the day-month order of dates.
INTL = {"ru": "ru-RU", "en": "en-GB"}
PLURAL_CATEGORIES = {"ru": ("one", "few", "many"), "en": ("one", "other")}
PLURAL_SUFFIXES = frozenset({"zero", "one", "two", "few", "many", "other"})
STRICT_ENV = "LOADLENS_I18N_STRICT"
LOCALES_DIR = Path(__file__).resolve().parents[1] / "locales"

KEY_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
NAMESPACE_RE = re.compile(r"^[a-z0-9_]+$")
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_DATE_FORMATS = {"ru": "%d.%m.%Y", "en": "%Y-%m-%d"}
_DEV_RECHECK_SEC = 1.0


class CatalogError(ValueError):
    """A dictionary file is broken: fail loudly instead of rendering keys."""


# ---- languages ---------------------------------------------------------------------------------------------


def normalize_lang(value: Any) -> Optional[str]:
    """``"ru"``/``"en"`` for a supported code (``"EN"``, ``"en-GB"``, ``"en_US"`` included), else ``None``."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-").split("-", 1)[0]
    return code if code in SUPPORTED else None


_LOCALE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("loadlens_locale", default=None)
_resolver: Optional[Callable[[], Optional[str]]] = None


def set_locale_resolver(fn: Optional[Callable[[], Optional[str]]]) -> None:
    """Registers the fallback used when no :func:`with_locale` scope is active (``None`` removes it)."""
    global _resolver
    _resolver = fn


def get_locale() -> str:
    """Explicit :func:`with_locale` scope -> registered resolver -> :data:`DEFAULT`."""
    explicit = _LOCALE.get()
    if explicit:
        return explicit
    resolver = _resolver
    if resolver is not None:
        try:
            resolved = normalize_lang(resolver())
        except Exception:  # a resolver must never break rendering
            logger.debug("i18n: locale resolver failed", exc_info=True)
            resolved = None
        if resolved:
            return resolved
    return DEFAULT


@contextmanager
def with_locale(lang: Any) -> Iterator[str]:
    """Scope with a fixed locale (jobs, Confluence, tests). An unknown or empty value means ``ru``."""
    code = normalize_lang(lang) or DEFAULT
    token = _LOCALE.set(code)
    try:
        yield code
    finally:
        _LOCALE.reset(token)


def _resolve(locale: Any) -> str:
    return normalize_lang(locale) or get_locale()


# ---- catalogs ----------------------------------------------------------------------------------------------


class Catalog:
    """Messages of one language: ``namespaces`` maps a namespace to its keys, ``messages`` is the flat view.

    ``serial`` changes with every load, so caches built from a catalog can key on it.
    """

    _serials = itertools.count(1)

    def __init__(self, lang: str, namespaces: dict[str, dict[str, str]]) -> None:
        self.lang = lang
        self.namespaces = namespaces
        self.messages = {key: value for names in namespaces.values() for key, value in names.items()}
        self.serial = next(Catalog._serials)

    def __repr__(self) -> str:
        return f"Catalog({self.lang!r}, {len(self.messages)} messages)"


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CatalogError(f"duplicate key {key!r}")
        out[key] = value
    return out


def read_namespace_file(path: Path) -> dict[str, str]:
    """Validated contents of one ``<ns>.json``; raises :class:`CatalogError` with the file name."""
    namespace = path.stem
    if not NAMESPACE_RE.match(namespace):
        raise CatalogError(f"{path}: namespace file names are lowercase snake_case")
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CatalogError(f"{path}: UTF-8 BOM is not allowed")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError(f"{path}: not valid UTF-8 ({exc})") from exc
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicates)
    except CatalogError as exc:
        raise CatalogError(f"{path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: a namespace file holds one JSON object")
    prefix = namespace + "."
    for key, value in data.items():
        if not key.startswith(prefix):
            raise CatalogError(f"{path}: key {key!r} is outside the namespace {namespace!r}")
        if not isinstance(value, str):
            raise CatalogError(f"{path}: the value of {key!r} must be a string")
    return data


@lru_cache(maxsize=None)
def _load(directory: str, lang: str, signature: Optional[tuple]) -> Catalog:
    folder = Path(directory) / lang
    namespaces: dict[str, dict[str, str]] = {}
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            namespaces[path.stem] = read_namespace_file(path)
    return Catalog(lang, namespaces)


def _dev_mode() -> bool:
    # Same switch as loadlens_app/static_assets.py: edited files are picked up without a restart.
    return os.getenv("LOADLENS_STATIC_NO_CACHE", "").strip() == "1"


_dev_lock = threading.Lock()
_dev_signatures: dict[tuple[str, str], tuple[float, tuple]] = {}


def _signature(directory: Path, lang: str) -> tuple:
    key = (str(directory), lang)
    now = time.monotonic()
    with _dev_lock:
        cached = _dev_signatures.get(key)
        if cached and now - cached[0] < _DEV_RECHECK_SEC:
            return cached[1]
    entries = []
    for path in sorted((directory / lang).glob("*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.name, stat.st_mtime_ns, stat.st_size))
    signature = tuple(entries)
    with _dev_lock:
        _dev_signatures[key] = (now, signature)
    return signature


def catalog(lang: Any) -> Catalog:
    """The loaded catalog of a language (read once per process; per file change in development mode)."""
    code = normalize_lang(lang) or DEFAULT
    directory = LOCALES_DIR
    signature = _signature(directory, code) if _dev_mode() else None
    return _load(str(directory), code, signature)


_warned: set[tuple[str, str]] = set()
_warned_lock = threading.Lock()


def reload_catalogs() -> None:
    """Forgets loaded catalogs and the warnings already given (tests, development)."""
    _load.cache_clear()
    with _dev_lock:
        _dev_signatures.clear()
    with _warned_lock:
        _warned.clear()


def _strict() -> bool:
    return os.getenv(STRICT_ENV, "").strip() == "1"


def _missing(lang: str, key: str) -> None:
    if _strict():
        raise KeyError(f"i18n: no message {key!r} for locale {lang!r}")
    marker = (lang, key)
    with _warned_lock:
        if marker in _warned:
            return
        _warned.add(marker)
    logger.warning("i18n: no message %r for locale %r", key, lang)


def has_key(key: str, locale: Any = None) -> bool:
    """The key (or a plural family with this base) exists in the locale."""
    messages = catalog(_resolve(locale)).messages
    if key in messages:
        return True
    return any(f"{key}.{suffix}" in messages for suffix in PLURAL_SUFFIXES)


def lookup(key: str, locale: Any = None) -> str:
    """Raw message with placeholders: the locale, then ru, then the key itself."""
    lang = _resolve(locale)
    text = catalog(lang).messages.get(key)
    if text is not None:
        return text
    _missing(lang, key)
    if lang != DEFAULT:
        text = catalog(DEFAULT).messages.get(key)
        if text is not None:
            return text
    return key


def plural_category(n: Any, lang: Any = DEFAULT) -> str:
    """CLDR category of ``n`` as used by the dictionaries (same as ``Intl.PluralRules`` in static/js/i18n.js).

    ru: one (1, 21, 101), few (2-4, 22-24, and fractions such as 1.5), many (0, 5-20, 11-14, 25-30).
    en: one (exactly 1), other.
    """
    code = normalize_lang(lang) or DEFAULT
    integer: Optional[int]
    if isinstance(n, int):
        integer = abs(int(n))
    else:
        try:
            number = abs(float(n))
        except (TypeError, ValueError):
            number = math.nan
        if not math.isfinite(number):
            return "other" if code == "en" else "many"
        integer = int(number) if number.is_integer() else None
    if code == "en":
        return "one" if integer == 1 else "other"
    if integer is None:
        return "few"
    if integer % 10 == 1 and integer % 100 != 11:
        return "one"
    if 2 <= integer % 10 <= 4 and not 12 <= integer % 100 <= 14:
        return "few"
    return "many"


def _plural_suffixes(category: str) -> tuple[str, ...]:
    order: list[str] = []
    for suffix in (category, "other", "many"):
        if suffix not in order:
            order.append(suffix)
    return tuple(order)


def lookup_plural(key: str, n: Any, locale: Any = None) -> str:
    """Raw plural form of ``key`` for ``n``: category, ``other``, ``many`` in the locale, then in ru, then the key."""
    lang = _resolve(locale)
    chain = (lang,) if lang == DEFAULT else (lang, DEFAULT)
    for index, code in enumerate(chain):
        messages = catalog(code).messages
        category = plural_category(n, code)
        for suffix in _plural_suffixes(category):
            text = messages.get(f"{key}.{suffix}")
            if text is not None:
                return text
        if index == 0:
            _missing(lang, f"{key}.{category}")
    return key


def number_text(n: Any, locale: Any = None) -> str:
    """``{n}`` of a plural message: integers as digits, fractions with the decimal separator of the locale."""
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        return str(n)
    if isinstance(n, float):
        if n.is_integer():
            return str(int(n))
        text = str(n)
        return text.replace(".", ",") if _resolve(locale) == "ru" else text
    return str(n)


# ---- formatting --------------------------------------------------------------------------------------------


def interpolate(text: str, params: Mapping[str, Any], convert: Callable[[Any], str] = str) -> str:
    """Replaces ``{name}`` with ``convert(params[name])``; unknown or missing names stay as they are."""
    if not params or "{" not in text:
        return text

    def substitute(match: re.Match) -> str:
        name = match.group(1)
        if name in params:
            return convert(params[name])
        return match.group(0)

    return PLACEHOLDER_RE.sub(substitute, text)


def is_ref(value: Any) -> bool:
    """``{"key": "...", "params": {...}}`` as stored next to a fallback text (``message_ref``)."""
    if not isinstance(value, Mapping) or set(value) - {"key", "params"}:
        return False
    key = value.get("key")
    params = value.get("params")
    return isinstance(key, str) and bool(key) and (params is None or isinstance(params, Mapping))


def param_text(value: Any, locale: Any = None) -> str:
    """Text of one parameter in the locale: messages and refs are rendered there, ``None`` is empty."""
    if value is None:
        return ""
    if isinstance(value, Msg):
        return value.render(locale) if value.key else str(value)
    if isinstance(value, LocalizedError):
        return value.render(locale)
    if is_ref(value):
        return render_ref(value, locale=locale)
    return str(value)


def t(key: str, /, *, locale: Any = None, **params: Any) -> str:
    """Plain text of ``key`` in the locale with ``{name}`` parameters (messages and refs rendered too)."""
    lang = _resolve(locale)
    return interpolate(lookup(key, lang), params, lambda value: param_text(value, lang))


def plural(key: str, n: Any, /, *, locale: Any = None, **params: Any) -> str:
    """Plural form of ``key`` for ``n``; ``n`` is also the ``{n}`` parameter unless given explicitly."""
    lang = _resolve(locale)
    params.setdefault("n", number_text(n, lang))
    return interpolate(lookup_plural(key, n, lang), params, lambda value: param_text(value, lang))


def _ref_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in params.items():
        if isinstance(value, (Msg, LocalizedError)):
            out[name] = value.ref()
        elif is_ref(value):
            out[name] = {"key": value["key"], "params": _ref_params(value.get("params") or {})}
        elif value is None or isinstance(value, (str, int, float, bool)):
            out[name] = value
        else:
            out[name] = str(value)
    return out


class Msg(str):
    """A message rendered now in the current locale that remembers its key and parameters.

    It is a ``str`` (logs, JSON and old readers see the text); ``ref()`` gives ``{"key", "params"}`` to store
    next to the text, so a reader can render it again in its own locale with :func:`render_ref`.
    """

    key: str
    params: dict[str, Any]

    def __new__(cls, text: str, key: str = "", params: Optional[Mapping[str, Any]] = None) -> "Msg":
        obj = super().__new__(cls, text)
        obj.key = key
        obj.params = dict(params or {})
        return obj

    def __reduce__(self):
        return (Msg, (str(self), self.key, self.params))

    def __repr__(self) -> str:
        return f"Msg({self.key!r}, {str(self)!r})"

    def ref(self) -> dict[str, Any]:
        return {"key": self.key, "params": _ref_params(self.params)}

    def render(self, locale: Any = None) -> str:
        return t(self.key, locale=locale, **self.params)


def msg(key: str, /, **params: Any) -> Msg:
    """:class:`Msg` of ``key`` rendered in the current locale."""
    return Msg(t(key, **params), key, params)


def render_ref(ref: Any, fallback: str = "", *, locale: Any = None) -> str:
    """Text of a stored ref in the locale; ``fallback`` for no ref, a broken one or a key no catalog knows.

    A ref to a plural family renders with ``plural()`` when its params carry ``n``.
    """
    if isinstance(ref, Msg):
        return ref.render(locale) if ref.key else str(ref)
    if not is_ref(ref):
        return fallback
    key = ref["key"]
    params = dict(ref.get("params") or {})
    lang = _resolve(locale)
    known = (catalog(lang).messages, catalog(DEFAULT).messages)
    if any(key in messages for messages in known):
        raw = lookup(key, lang)
    elif "n" in params and any(f"{key}.{suffix}" in messages for messages in known for suffix in PLURAL_SUFFIXES):
        raw = lookup_plural(key, params["n"], lang)
        params["n"] = number_text(params["n"], lang)
    else:
        marker = (lang, key)
        with _warned_lock:
            first = marker not in _warned
            _warned.add(marker)
        if first:
            logger.warning("i18n: stored ref %r is not in the catalogs, the fallback text is shown", key)
        return fallback or key
    return interpolate(raw, params, lambda value: param_text(value, lang))


def _restore_error(cls: type, key: str, params: dict[str, Any]) -> "LocalizedError":
    error = cls.__new__(cls)
    LocalizedError.__init__(error, key, **params)
    return error


class LocalizedError(Exception):
    """An error whose text is a dictionary key: ``str()`` renders it in the current locale.

    ``raise LocalizedError("api.run_not_found", run=name)``; ``.key``, ``.params`` and ``.ref()`` as in :class:`Msg`.
    """

    def __init__(self, key: str, /, **params: Any) -> None:
        super().__init__(key)
        self.key = key
        self.params = params

    def __str__(self) -> str:
        return self.render()

    def __reduce__(self):
        return (_restore_error, (type(self), self.key, dict(self.params)))

    def render(self, locale: Any = None) -> str:
        return t(self.key, locale=locale, **self.params)

    def ref(self) -> dict[str, Any]:
        return {"key": self.key, "params": _ref_params(self.params)}


def fmt_number(value: Any, digits: int = 1, *, locale: Any = None) -> str:
    """``value`` with ``digits`` decimals: a comma in ru, a point in en; ``—`` for no number."""
    if value is None or isinstance(value, bool):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    text = f"{number:.{max(0, int(digits))}f}"
    return text.replace(".", ",") if _resolve(locale) == "ru" else text


def fmt_date(value: Any, *, locale: Any = None) -> str:
    """Date of a ``date``/``datetime``/ISO string: ``%d.%m.%Y`` in ru, ``%Y-%m-%d`` in en; other values as text."""
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return value
    if isinstance(value, (date, datetime)):
        return value.strftime(_DATE_FORMATS[_resolve(locale)])
    return str(value)


# ---- browser payload ---------------------------------------------------------------------------------------


def client_messages(locale: Any, namespaces: Optional[Iterable[str]] = None) -> dict[str, str]:
    """Messages of the namespaces for the browser: ru values with the locale's values on top, key by key.

    ``None`` takes every namespace of both catalogs; unknown namespaces are skipped.
    """
    lang = normalize_lang(locale) or DEFAULT
    base = catalog(DEFAULT)
    own = catalog(lang)
    if namespaces is None:
        names = sorted(set(base.namespaces) | set(own.namespaces))
    else:
        names = list(dict.fromkeys(namespaces))
    out: dict[str, str] = {}
    for name in names:
        out.update(base.namespaces.get(name, {}))
        if own is not base:
            out.update(own.namespaces.get(name, {}))
    return out
