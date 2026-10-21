# SPDX-License-Identifier: Apache-2.0
"""Helpers for localization tests: switch a test client to a locale and look for Russian text left in a page.

    from tests.i18n_helpers import assert_no_cyrillic, use_locale, visible_text
    use_locale(client, "en")
    html = client.get("/").get_data(as_text=True)
    assert_no_cyrillic(visible_text(html), allow=("Русский",))

The cookie goes the real way to ``current_locale()``; nothing in ``g`` is patched.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Iterable, Optional

from loadlens_app.i18n import COOKIE

CYRILLIC = re.compile("[\u0400-\u04ff]")
# Attributes a user reads or hears (tooltips, placeholders, screen readers, image text).
VISIBLE_ATTRIBUTES = ("title", "placeholder", "aria-label", "alt")
# Language names are shown in their own language (the language menu of every page, locales/allowlist.json "*").
LANGUAGE_NAMES = ("Русский",)


def use_locale(client, code: Optional[str]) -> None:
    """Sets the language cookie of a Flask test client (``None`` removes it)."""
    if code is None:
        client.delete_cookie(COOKIE)
    else:
        client.set_cookie(COOKIE, code)


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "template"):
            self._hidden += 1
        for name, value in attrs:
            if name in VISIBLE_ATTRIBUTES and value:
                self.parts.append(value)

    def handle_startendtag(self, tag, attrs):
        for name, value in attrs:
            if name in VISIBLE_ATTRIBUTES and value:
                self.parts.append(value)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "template") and self._hidden:
            self._hidden -= 1

    def handle_data(self, data):
        if not self._hidden and data.strip():
            self.parts.append(data.strip())


def visible_text(html: str) -> str:
    """Text nodes (``<title>`` included, scripts and styles not) and title/placeholder/aria-label/alt, one per line."""
    parser = _VisibleText()
    parser.feed(html)
    parser.close()
    return "\n".join(parser.parts)


def assert_no_cyrillic(text: str, allow: Iterable[str] = ()) -> None:
    """Fails with the offending lines when ``text`` holds Cyrillic outside the ``allow`` fragments.

    The language names of :data:`LANGUAGE_NAMES` are always allowed.
    """
    cleaned = str(text)
    for fragment in sorted({*allow, *LANGUAGE_NAMES}, key=len, reverse=True):
        cleaned = cleaned.replace(fragment, "")
    offending = [line.strip() for line in cleaned.splitlines() if CYRILLIC.search(line)]
    assert not offending, "Cyrillic text left:\n" + "\n".join(offending[:40])
