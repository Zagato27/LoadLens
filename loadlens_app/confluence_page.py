# SPDX-License-Identifier: Apache-2.0
"""Text of a Confluence page for the system context import.

The page is read with the saved ``confluence`` connection (the one that publishes reports). A link is
accepted only when it points at that Confluence: same scheme, host, port and context path. Requests are
built from the configured base URL and the page id or title taken from the link, so the stored login
never goes to an address typed by the user.

Supported links: ``/pages/viewpage.action?pageId=…``, ``/spaces/KEY/pages/ID/Title`` (Cloud and newer
Data Center), ``/display/KEY/Title`` and ``/pages/viewpage.action?spaceKey=…&title=…`` (Server and Data
Center), and short links ``/x/CODE``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, unquote_plus, urljoin, urlsplit

from AI.i18n import t
from loadlens_app.context_import import ContextImportError, ExtractedText, _finish

logger = logging.getLogger(__name__)

_PAGE_EXPAND = "body.view,body.storage,space,version"
_DEFAULT_PORTS = {"http": 80, "https": 443}
_BLOCK_TAGS = ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote", "section", "dt", "dd")
_SHORT_LINK = re.compile(r"^/x/([A-Za-z0-9_-]+)/?$")
_SPACE_PAGE = re.compile(r"^/spaces/([^/]+)/pages/(\d+)(?:/.*)?$")
_DISPLAY = re.compile(r"^/display/([^/]+)/([^/?#]+)/?$")


@dataclass(frozen=True)
class PageRef:
    page_id: str = ""
    space_key: str = ""
    title: str = ""
    short_code: str = ""


def _origin(parts) -> tuple[str, str, int]:
    scheme = (parts.scheme or "").lower()
    return scheme, (parts.hostname or "").lower(), parts.port or _DEFAULT_PORTS.get(scheme, 0)


def page_ref(url: str, base_url: str) -> PageRef:
    """Which page a link names; refuses links outside the configured Confluence."""
    link = (url or "").strip()
    if not link:
        raise ContextImportError(t("server.confluence_page.link_required"))
    try:
        parts, base = urlsplit(link), urlsplit(base_url)
    except ValueError as exc:
        raise ContextImportError(t("server.confluence_page.link_invalid")) from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ContextImportError(t("server.confluence_page.link_full"))
    base_path = base.path.rstrip("/")
    if _origin(parts) != _origin(base) or not (parts.path == base_path or parts.path.startswith(base_path + "/")):
        raise ContextImportError(t("server.confluence_page.foreign_host", base=f"{base.scheme}://{base.netloc}{base_path}"))
    path = parts.path[len(base_path):] or "/"
    query = parse_qs(parts.query)
    page_id = (query.get("pageId") or [""])[0].strip()
    if page_id.isdigit():
        return PageRef(page_id=page_id)
    match = _SPACE_PAGE.match(path)
    if match:
        return PageRef(page_id=match.group(2))
    match = _DISPLAY.match(path)
    if match:
        return PageRef(space_key=unquote(match.group(1)), title=unquote_plus(match.group(2)))
    space_key, title = (query.get("spaceKey") or [""])[0].strip(), (query.get("title") or [""])[0].strip()
    if space_key and title:
        return PageRef(space_key=space_key, title=title)
    match = _SHORT_LINK.match(path)
    if match:
        return PageRef(short_code=match.group(1))
    raise ContextImportError(t("server.confluence_page.page_unknown"))


def html_to_text(markup: str) -> str:
    """Readable text of page HTML: blocks on their own lines, list items with dashes, table rows joined by |."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            continue
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n" + "\n".join(rows) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with(soup.new_string("\n"))
    for item in soup.find_all("li"):
        item.insert(0, soup.new_string("- "))
    for block in soup.find_all(_BLOCK_TAGS):
        block.insert_before(soup.new_string("\n"))
        block.insert_after(soup.new_string("\n"))
    return soup.get_text()


def _client():
    from loadlens_app.confluence_export import ConfluenceClient, confluence_config

    cfg = confluence_config()
    base_url = str(cfg.get("base_url") or "").strip()
    if not base_url or not str(cfg.get("username") or "").strip():
        raise ContextImportError(t("server.confluence_page.not_configured"), 422, "missing_confluence")
    return ConfluenceClient(cfg)


def _failure(status: int) -> ContextImportError:
    if status in (401, 403):
        return ContextImportError(t("server.confluence_page.forbidden", status=status), 403, "confluence_forbidden")
    if status == 404:
        return ContextImportError(t("server.confluence_page.not_found"), 404, "not_found")
    return ContextImportError(t("server.confluence_page.http_error", status=status), 502, "confluence_failed")


def _call(client, path: str, action: str, **kwargs: Any):
    try:
        return client._request("GET", path, action, **kwargs)
    except ContextImportError:
        raise
    except Exception as exc:  # network, TLS, timeouts
        logger.warning("context import: Confluence %s failed: %s", action, exc)
        raise ContextImportError(t("server.confluence_page.no_answer", error=str(exc)[:200]), 502, "confluence_failed") from exc


def _resolve_short_link(client, code: str) -> PageRef:
    response = _call(client, f"/x/{code}", "resolve short link", allow_redirects=False)
    location = response.headers.get("Location") or ""
    if response.status_code not in (301, 302, 303, 307, 308) or not location:
        raise _failure(response.status_code if response.status_code >= 400 else 404)
    ref = page_ref(urljoin(client.base_url + "/", location), client.base_url)
    if ref.short_code:
        raise ContextImportError(t("server.confluence_page.short_link_loop"))
    return ref


def _fetch(client, ref: PageRef) -> dict[str, Any]:
    if ref.short_code:
        ref = _resolve_short_link(client, ref.short_code)
    if ref.page_id:
        response = _call(client, f"/rest/api/content/{ref.page_id}", "read page", params={"expand": _PAGE_EXPAND})
        if response.status_code >= 400:
            raise _failure(response.status_code)
        page = response.json()
    else:
        response = _call(
            client, "/rest/api/content", "find page",
            params={"spaceKey": ref.space_key, "title": ref.title, "type": "page", "expand": _PAGE_EXPAND},
        )
        if response.status_code >= 400:
            raise _failure(response.status_code)
        results = (response.json() or {}).get("results") or []
        page = results[0] if results else None
        if page is None:
            raise _failure(404)
    if not isinstance(page, dict):
        raise ContextImportError(t("server.confluence_page.unexpected"), 502, "confluence_failed")
    return page


@dataclass
class ConfluencePage:
    extracted: ExtractedText
    page_id: str
    title: str
    host: str


def read_page(url: str) -> ConfluencePage:
    """Text of the page the link names, read with the saved Confluence connection."""
    client = _client()
    ref = page_ref(url, client.base_url)
    page = _fetch(client, ref)
    body = page.get("body") if isinstance(page.get("body"), dict) else {}
    # The rendered view expands macros such as includes and excerpts; storage is the fallback.
    markup = ""
    for kind in ("view", "storage"):
        node = body.get(kind) if isinstance(body.get(kind), dict) else {}
        if str(node.get("value") or "").strip():
            markup = str(node["value"])
            break
    title = str(page.get("title") or "").strip()
    text = html_to_text(markup)
    extracted = _finish(f"{title}\n\n{text}" if title else text, "confluence", [])
    return ConfluencePage(extracted=extracted, page_id=str(page.get("id") or ref.page_id), title=title, host=urlsplit(client.base_url).netloc)


__all__ = ["ConfluencePage", "PageRef", "html_to_text", "page_ref", "read_page"]
