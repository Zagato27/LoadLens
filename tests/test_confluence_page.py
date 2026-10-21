# SPDX-License-Identifier: Apache-2.0
"""Confluence page as a source of the system context: links, HTML to text, reading with the saved connection."""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import confluence_export  # noqa: E402
from loadlens_app import confluence_page as cp  # noqa: E402
from loadlens_app.context_import import ContextImportError  # noqa: E402

BASE = "https://wiki.example.ru/confluence"
PAGE_HTML = (
    "<h1>Checkout</h1><p>Сервис <strong>checkout-api</strong> оформляет заказы и вызывает <a href='#'>payments</a>.</p>"
    "<ul><li>Java, Spring Boot</li><li>PostgreSQL orders-db</li></ul>"
    "<table><tr><th>Компонент</th><th>Роль</th></tr><tr><td>checkout-api</td><td>оформление</td></tr></table>"
    "<p>Риск: пул соединений мал.<br/>Лимит пула — 20.</p><script>alert(1)</script>"
)


@pytest.mark.parametrize(
    "url, expected",
    [
        (f"{BASE}/pages/viewpage.action?pageId=12345", cp.PageRef(page_id="12345")),
        (f"{BASE}/spaces/SHOP/pages/777/Checkout+architecture", cp.PageRef(page_id="777")),
        (f"{BASE}/display/SHOP/Checkout+architecture", cp.PageRef(space_key="SHOP", title="Checkout architecture")),
        (f"{BASE}/display/SHOP/%D0%90%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%B0", cp.PageRef(space_key="SHOP", title="Архитектура")),
        (f"{BASE}/pages/viewpage.action?spaceKey=SHOP&title=Checkout+architecture", cp.PageRef(space_key="SHOP", title="Checkout architecture")),
        (f"{BASE}/x/AbC_d-1", cp.PageRef(short_code="AbC_d-1")),
        ("https://WIKI.example.ru:443/confluence/pages/viewpage.action?pageId=5#comments", cp.PageRef(page_id="5")),
    ],
)
def test_page_links(url, expected):
    assert cp.page_ref(url, BASE) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/confluence/pages/viewpage.action?pageId=1",
        "http://wiki.example.ru/confluence/pages/viewpage.action?pageId=1",
        "https://wiki.example.ru:8443/confluence/pages/viewpage.action?pageId=1",
        "https://wiki.example.ru/other/pages/viewpage.action?pageId=1",
        "https://wiki.example.ru/confluenceX/pages/viewpage.action?pageId=1",
    ],
)
def test_links_outside_the_configured_confluence_are_refused(url):
    with pytest.raises(ContextImportError, match="Confluence из настроек"):
        cp.page_ref(url, BASE)


@pytest.mark.parametrize("url", ["", "wiki.example.ru/page", f"{BASE}/dashboard.action", f"{BASE}/pages/viewpage.action?pageId=abc"])
def test_links_without_a_page(url):
    with pytest.raises(ContextImportError):
        cp.page_ref(url, BASE)


def test_html_to_text_keeps_structure():
    lines = [line.strip() for line in cp.html_to_text(PAGE_HTML).splitlines() if line.strip()]
    assert lines == [
        "Checkout",
        "Сервис checkout-api оформляет заказы и вызывает payments.",
        "- Java, Spring Boot",
        "- PostgreSQL orders-db",
        "Компонент | Роль",
        "checkout-api | оформление",
        "Риск: пул соединений мал.",
        "Лимит пула — 20.",
    ]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


@pytest.fixture
def confluence(monkeypatch):
    """Saved connection plus a scripted ``ConfluenceClient._request``; records every call."""
    cfg = {"base_url": BASE + "/", "username": "reader", "password": "secret", "verify_ssl": True}
    monkeypatch.setattr(confluence_export, "confluence_config", lambda: dict(cfg))
    state = {"calls": [], "routes": {}}

    def fake_request(self, method, path, action, **kwargs):
        state["calls"].append((method, self.base_url + path, kwargs))
        route = state["routes"].get(path)
        if isinstance(route, Exception):
            raise route
        return route if route is not None else FakeResponse(404, {})

    monkeypatch.setattr(confluence_export.ConfluenceClient, "_request", fake_request)
    state["cfg"] = cfg
    return state


def _page(page_id="12345", title="Архитектура Checkout", view=PAGE_HTML, storage="<p>storage</p>"):
    return {"id": page_id, "title": title, "body": {"view": {"value": view}, "storage": {"value": storage}}}


def test_read_page_by_id_uses_the_rendered_view(confluence):
    confluence["routes"]["/rest/api/content/12345"] = FakeResponse(200, _page())
    page = cp.read_page(f"{BASE}/pages/viewpage.action?pageId=12345")
    assert page.title == "Архитектура Checkout" and page.page_id == "12345" and page.host == "wiki.example.ru"
    assert page.extracted.text.startswith("Архитектура Checkout\n\nCheckout")
    assert "checkout-api | оформление" in page.extracted.text and "alert" not in page.extracted.text
    assert page.extracted.format == "confluence"
    method, url, kwargs = confluence["calls"][0]
    assert url == f"{BASE}/rest/api/content/12345" and "body.view" in kwargs["params"]["expand"]


def test_read_page_by_title_and_storage_fallback(confluence):
    found = _page(page_id="777", view="", storage="<p>" + "Компонент checkout-api пишет заказы в PostgreSQL. " * 3 + "</p>")
    confluence["routes"]["/rest/api/content"] = FakeResponse(200, {"results": [found]})
    page = cp.read_page(f"{BASE}/display/SHOP/Checkout+architecture")
    assert page.page_id == "777" and "пишет заказы в PostgreSQL" in page.extracted.text
    params = confluence["calls"][0][2]["params"]
    assert params["spaceKey"] == "SHOP" and params["title"] == "Checkout architecture"


def test_short_link_is_resolved_on_the_configured_host(confluence):
    confluence["routes"]["/x/AbCd"] = FakeResponse(302, headers={"Location": "/confluence/pages/viewpage.action?pageId=12345"})
    confluence["routes"]["/rest/api/content/12345"] = FakeResponse(200, _page())
    page = cp.read_page(f"{BASE}/x/AbCd")
    assert page.page_id == "12345"
    assert confluence["calls"][0][2] == {"allow_redirects": False}
    # A short link that redirects to another host is refused before anything is read there.
    confluence["calls"].clear()
    confluence["routes"]["/x/Evil"] = FakeResponse(302, headers={"Location": "https://evil.example.com/pages/viewpage.action?pageId=1"})
    with pytest.raises(ContextImportError, match="Confluence из настроек"):
        cp.read_page(f"{BASE}/x/Evil")
    assert len(confluence["calls"]) == 1


@pytest.mark.parametrize(
    "route, status, code",
    [
        (FakeResponse(403, {}), 403, "confluence_forbidden"),
        (FakeResponse(401, {}), 403, "confluence_forbidden"),
        (FakeResponse(404, {}), 404, "not_found"),
        (FakeResponse(500, {}), 502, "confluence_failed"),
        (ConnectionError("connection refused"), 502, "confluence_failed"),
    ],
)
def test_read_page_failures(confluence, route, status, code):
    confluence["routes"]["/rest/api/content/12345"] = route
    with pytest.raises(ContextImportError) as raised:
        cp.read_page(f"{BASE}/pages/viewpage.action?pageId=12345")
    assert (raised.value.status_code, raised.value.code) == (status, code)


def test_unconfigured_confluence(monkeypatch):
    monkeypatch.setattr(confluence_export, "confluence_config", lambda: {"base_url": "", "username": ""})
    with pytest.raises(ContextImportError) as raised:
        cp.read_page(f"{BASE}/pages/viewpage.action?pageId=1")
    assert raised.value.status_code == 422 and raised.value.code == "missing_confluence"


def test_other_host_is_refused_without_any_request(confluence):
    with pytest.raises(ContextImportError):
        cp.read_page("https://evil.example.com/confluence/pages/viewpage.action?pageId=1")
    assert confluence["calls"] == []
