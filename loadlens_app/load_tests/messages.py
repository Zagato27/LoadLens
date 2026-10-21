# SPDX-License-Identifier: Apache-2.0
"""Texts of the load test feature: messages of the ``lt`` dictionary namespace (plans/i18n-plan.md, section 4).

Code that answers a request renders them in the language of the reader at once (``t``, ``str()`` of an error).
The watcher thread has no reader: what it stores in a run or an instance is the text in the language of the
installation plus ``message_ref`` (``{"key", "params"}``), which the views render again in the language of whoever
reads it (:func:`stored_text`).

Errors of the feature (:class:`LtError` and its subclasses) take a dictionary key with parameters, another localized
error or message (its key is kept), or a ready text: messages of third-party libraries, of other modules and of tests
are shown as they are.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional, Sequence

from AI.i18n import KEY_RE, LocalizedError, Msg, is_ref, lookup, msg, param_text, render_ref, t

# "{text}": a ready text carried as a message (third-party errors, plain strings of callers and tests).
TEXT_KEY = "lt.text"
# Text of a foreign exception kept as a message parameter.
ERROR_TEXT_CHARS = 500


def foreign_ref(value: Any) -> Optional[dict[str, Any]]:
    """The ``ref()`` of an error of another module that keeps its key without being a LocalizedError (LimitError)."""
    if isinstance(value, (Msg, LocalizedError)):
        return None
    method = getattr(value, "ref", None)
    if not callable(method):
        return None
    try:
        ref = method()
    except Exception:
        return None
    return ref if is_ref(ref) else None


def _key_params(message: Any, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(message, LocalizedError):
        return message.key, {**message.params, **params}
    if isinstance(message, Msg) and message.key:
        return message.key, {**message.params, **params}
    ref = foreign_ref(message)
    if ref is not None:
        return ref["key"], {**(ref.get("params") or {}), **params}
    text = "" if message is None else str(message)
    # Messages of the feature have three segments or more (lt.area.name); "lt.rps_total" is a query pack role.
    if text.startswith("lt.") and text.count(".") >= 2 and KEY_RE.match(text):
        return text, dict(params)
    return TEXT_KEY, {"text": text}


class LtError(LocalizedError):
    """An error of the load test feature; ``str()`` renders it in the current locale.

    ``LtError("lt.some.key", name=value)`` is a dictionary message; another :class:`LocalizedError`, a ``Msg`` or an
    error with ``ref()`` (``LimitError``) keeps its key and parameters; any other value is a ready text shown as it is.
    """

    def __init__(self, message: Any = "", /, **params: Any) -> None:
        key, values = _key_params(message, params)
        super().__init__(key, **values)


def error_param(exc: BaseException) -> Any:
    """An exception as a message parameter: a localized one keeps its key, other text is cut to a sane length."""
    if isinstance(exc, LocalizedError):
        return exc
    return foreign_ref(exc) or str(exc)[:ERROR_TEXT_CHARS]


def as_msg(value: Any) -> Any:
    """A localized error as a message that keeps its key (notes cached across requests); other values as they are."""
    if isinstance(value, LocalizedError):
        return Msg(str(value), value.key, value.params)
    return value


def ref_of(message: Any) -> Optional[dict[str, Any]]:
    """``{"key", "params"}`` of a message or a localized error to store next to its text; ``None`` for plain text."""
    if isinstance(message, (Msg, LocalizedError)) and message.key and message.key != TEXT_KEY:
        return message.ref()
    return foreign_ref(message)


def stored_text(text: str, ref: Any) -> str:
    """A stored message in the reader's language: the ref rendered again, else the stored text."""
    return render_ref(ref, fallback=text or "") if ref else (text or "")


def text_of(value: Any) -> str:
    """Text of a message, a localized error or a ref in the current locale; anything else as text."""
    return param_text(value)


def ru_text(key: str) -> str:
    """The ru message of ``key`` as it is (placeholders not filled): defaults that are stored as settings values."""
    return lookup(key, "ru")


def joined(parts: Sequence[Any]) -> Any:
    """Messages joined with "; " as one message whose parts each keep their own key."""
    items = [part for part in parts if part]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return msg("lt.common.joined", first=items[0], rest=joined(items[1:]))


class Titles(Mapping[str, str]):
    """Code -> title in the current locale: ``keys`` are rendered on every read, ``fixed`` titles stay as they are."""

    def __init__(self, keys: Mapping[str, str], fixed: Optional[Mapping[str, str]] = None) -> None:
        self._keys = dict(keys)
        self._fixed = dict(fixed or {})

    def __getitem__(self, code: str) -> str:
        if code in self._keys:
            return t(self._keys[code])
        return self._fixed[code]

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._fixed) + [code for code in self._keys if code not in self._fixed])

    def __len__(self) -> int:
        return len(set(self._fixed) | set(self._keys))

    def key(self, code: str) -> Optional[str]:
        """Dictionary key of the title of ``code`` (``None`` for a fixed title or an unknown code)."""
        return self._keys.get(code)


__all__ = [
    "ERROR_TEXT_CHARS",
    "LtError",
    "TEXT_KEY",
    "Titles",
    "as_msg",
    "error_param",
    "foreign_ref",
    "joined",
    "msg",
    "ref_of",
    "ru_text",
    "stored_text",
    "t",
    "text_of",
]
