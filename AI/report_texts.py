# SPDX-License-Identifier: Apache-2.0
"""Tables of deterministic report texts backed by the dictionaries (``locales/<lang>/*.json``).

The analysis modules keep their public tables (``HOLD_NOTE_TEXTS["en"]["early_drop"]``, ``TEST_TYPE_TEXTS``,
``SLA_CHECK_TITLES``...): readers and tests index them by language or by code. The texts themselves live under
dictionary keys, so a table renders them on access and always shows the loaded catalogs.

- :class:`LanguageTable` — ``table[lang] -> {code: value}`` for every supported language (texts of stored codes);
- :class:`LocaleTable` — ``table[code] -> text`` in the locale active at access time (the ``with_locale`` scope of a
  run, else the request), for names written into a report while it is built.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from AI.i18n import DEFAULT, SUPPORTED, lookup, normalize_lang, t


def text_lang(lang: Any) -> str:
    """Language of an explicit ``lang`` argument: a supported code (``EN``, ``en-GB`` included), else ``ru``."""
    return normalize_lang(lang) or DEFAULT


# Templates of the load-step labels a report stores (AI/context_pack.py, AI/spike.py); the more specific first.
STEP_LABEL_KEYS = ("step_level", "step", "background_level", "background", "spike", "interval")
_STEP_LABEL_PARAMS = {"index": r"\d+", "level": r"-?\d+(?:[.,]\d+)?", "start": r"\d{1,2}:\d{2}", "end": r"\d{1,2}:\d{2}"}


def _step_label_pattern(template: str) -> "re.Pattern[str]":
    parts = re.split(r"\{(\w+)\}", template)
    regex = "".join(
        re.escape(part) if i % 2 == 0 else f"(?P<{part}>{_STEP_LABEL_PARAMS.get(part, '.+?')})" for i, part in enumerate(parts)
    )
    return re.compile(regex)


def localized_step_label(label: Any, lang: Optional[str] = None) -> str:
    """A stored load-step label (``ступень 2 · ≈195 RPS``) in ``lang`` (default: the current locale).

    Reports store step labels in their own language. A label that matches a ``report.step_label.*`` template of a
    supported language is rendered again from the same values, so pages outside the report body show it in the
    reader's language; any other text is returned unchanged. A label already in ``lang`` comes back identical.
    """
    text = str(label or "")
    if not text:
        return text
    for key in STEP_LABEL_KEYS:
        for source in SUPPORTED:
            match = _step_label_pattern(lookup(f"report.step_label.{key}", locale=source)).fullmatch(text)
            if match:
                return t(f"report.step_label.{key}", locale=lang, **match.groupdict())
    return text


class LanguageTable(Mapping):
    """``{lang: {code: render(code, lang)}}`` for the supported languages; values are rendered on access."""

    def __init__(self, codes: Sequence[str], render: Callable[[str, str], Any]) -> None:
        self._codes = tuple(codes)
        self._render = render

    @property
    def codes(self) -> tuple[str, ...]:
        return self._codes

    def __getitem__(self, lang: str) -> dict:
        if lang not in SUPPORTED:
            raise KeyError(lang)
        return {code: self._render(code, lang) for code in self._codes}

    def __iter__(self) -> Iterator[str]:
        return iter(SUPPORTED)

    def __len__(self) -> int:
        return len(SUPPORTED)

    def __repr__(self) -> str:
        return f"LanguageTable({list(self._codes)!r})"


class LocaleTable(Mapping):
    """``{code: render(code)}`` where ``render`` reads the current locale; values are rendered on access."""

    def __init__(self, codes: Sequence[str], render: Callable[[str], str]) -> None:
        self._codes = tuple(codes)
        self._render = render

    def __getitem__(self, code: str) -> str:
        if code not in self._codes:
            raise KeyError(code)
        return self._render(code)

    def __iter__(self) -> Iterator[str]:
        return iter(self._codes)

    def __len__(self) -> int:
        return len(self._codes)

    def __repr__(self) -> str:
        return f"LocaleTable({list(self._codes)!r})"


__all__ = ["LanguageTable", "LocaleTable", "STEP_LABEL_KEYS", "localized_step_label", "text_lang"]
