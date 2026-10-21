# SPDX-License-Identifier: Apache-2.0
"""Allow-list sanitizer for the engineer's notes, which arrive as HTML from a contenteditable editor.

Only formatting tags survive, attributes are dropped except a link's ``href`` with a safe scheme, and
the content of script-like elements is removed entirely. The output is safe to insert with innerHTML
even if the browser-side sanitizer is missing, and it is what the Confluence export receives.
"""

from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit

ALLOWED_TAGS = frozenset({
    "p", "div", "br", "b", "strong", "i", "em", "u", "s", "ul", "ol", "li",
    "h3", "h4", "blockquote", "pre", "code", "a",
})
VOID_TAGS = frozenset({"br"})
# Elements whose text must not leak into the output either.
DROP_WITH_CONTENT = frozenset({
    "script", "style", "template", "iframe", "object", "noscript", "svg", "math", "title", "head",
    "textarea", "select", "noembed", "noframes", "xmp",
})
SAFE_SCHEMES = frozenset({"http", "https", "mailto"})
MAX_DEPTH = 64


def safe_href(raw: str) -> str:
    """The link target when it is http(s), mailto or a relative address; ``""`` otherwise."""
    cleaned = "".join(ch for ch in raw if ch >= " " and ch != "\x7f").strip()
    if not cleaned:
        return ""
    try:
        scheme = urlsplit(cleaned).scheme.lower()
    except ValueError:
        return ""
    return cleaned if not scheme or scheme in SAFE_SCHEMES else ""


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.open: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if self.skip:
            if name in DROP_WITH_CONTENT:
                self.skip += 1
            return
        if name in DROP_WITH_CONTENT:
            self.skip = 1
            return
        if name not in ALLOWED_TAGS or len(self.open) >= MAX_DEPTH:
            return
        if name in VOID_TAGS:
            self.out.append(f"<{name}>")
            return
        rendered = name
        if name == "a":
            href = safe_href(next((value or "" for key, value in attrs if key.lower() == "href"), ""))
            if href:
                rendered = f'a href="{escape(href, quote=True)}" rel="noopener noreferrer"'
        self.out.append(f"<{rendered}>")
        self.open.append(name)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in VOID_TAGS:
            self.handle_starttag(name, attrs)
        elif name not in DROP_WITH_CONTENT and not self.skip:
            self.handle_starttag(name, attrs)
            self.handle_endtag(name)

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if self.skip:
            if name in DROP_WITH_CONTENT:
                self.skip -= 1
            return
        if name not in self.open:
            return
        while self.open:
            current = self.open.pop()
            self.out.append(f"</{current}>")
            if current == name:
                break

    def handle_data(self, data: str) -> None:
        if not self.skip and data:
            self.out.append(escape(data, quote=False))

    def result(self) -> str:
        self.close()
        while self.open:
            self.out.append(f"</{self.open.pop()}>")
        return "".join(self.out)


def sanitize_html(raw: object) -> str:
    """Formatting-only HTML; comments, scripts, styles, event handlers and unsafe links are removed."""
    text = raw if isinstance(raw, str) else ""
    if not text.strip():
        return ""
    parser = _Sanitizer()
    parser.feed(text)
    return parser.result()


__all__ = ["ALLOWED_TAGS", "safe_href", "sanitize_html"]
