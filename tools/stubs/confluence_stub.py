"""Confluence stub for a LoadLens sandbox on 127.0.0.1:5091: reads, publishes pages, keeps them in memory.

How to run (own console, or in the background; Ctrl+C stops it):

    C:/Python312/python.exe tools/stubs/confluence_stub.py
    C:/Python312/python.exe tools/stubs/confluence_stub.py --delay-ms 100 --log %TEMP%/ll_stubs/confluence.jsonl

Basic auth ``reader`` / ``secret`` (``--user``/``--password``) on every request; anything else gets 401.
Point the sandbox at it in its settings_runtime.json (``tools/bench_endpoints.py --setup-stubs`` writes it):

    "confluence": {"enabled": true, "base_url": "http://127.0.0.1:5091", "username": "reader",
                   "password": "secret", "space_key": "PERF", "parent_page_id": "42", "verify_ssl": false}

What it serves (the REST calls of loadlens_app/confluence_export.py, confluence_page.py and the connection
checks):

* page 42 "Архитектура Checkout" with ``body.view`` and ``body.storage``, short link ``/x/Ab42`` (302 to
  ``/pages/viewpage.action?pageId=42``) - the context import e2e reads it, as with the original
  %TEMP%/ll_ctx_e2e/confluence_stub.py;
* ``GET /rest/api/content/<id>``, ``GET /rest/api/content?spaceKey=&title=`` (search by title),
  ``POST /rest/api/content`` (create, ids from 1001), ``PUT /rest/api/content/<id>`` (the version must be
  the current one + 1, otherwise 409), ``DELETE /rest/api/content/<id>`` (page or attachment);
* ``GET|POST /rest/api/content/<id>/child/attachment`` and ``POST .../child/attachment/<att>/data``
  (multipart upload, ``X-Atlassian-Token: no-check`` required; the file name and the size of the upload
  are kept, not the bytes);
* ``GET /rest/api/space/<key>`` (the "Проверить" button of the Confluence settings).

For checks (B6 e2e): ``GET /_stub/pages`` lists the published pages (id, title, version, ancestors, body
length, attachment names), ``GET /_stub/pages/<id>`` returns one with its storage body, ``GET /_stub/stats``
counts requests by method and path kind, ``POST /_stub/reset`` drops everything but page 42.

``--log FILE`` appends one JSON line per request (method, path, auth ok; no bodies). Listens on
127.0.0.1/localhost only. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import copy
import itertools
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Sequence
from urllib.parse import parse_qs, urlsplit

DEFAULT_PORT = 5091
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SEED_PAGE_ID = "42"
SEED_PAGE_HTML = (
    "<h1>Checkout</h1><p>Сервис <strong>checkout-api</strong> на Java оформляет заказы и синхронно вызывает payments.</p>"
    "<table><tr><th>Компонент</th><th>Хранилище</th></tr><tr><td>checkout-api</td><td>PostgreSQL orders-db</td></tr></table>"
    "<ul><li>Риск: пул соединений к orders-db мал</li></ul>"
)
_FILENAME = re.compile(rb'filename="([^"]*)"')


class PageStore:
    """In-memory pages and attachments with Confluence-like ids and versions (thread-safe)."""

    def __init__(self, space_key: str = "PERF") -> None:
        self.lock = threading.Lock()
        self.space_key = space_key
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.pages: dict[str, dict] = {}
            self.attachments: dict[str, dict] = {}
            self._ids = itertools.count(1001)
            self.pages[SEED_PAGE_ID] = self._page(SEED_PAGE_ID, "Архитектура Checkout", SEED_PAGE_HTML, self.space_key, [], 1)

    @staticmethod
    def _page(page_id: str, title: str, storage: str, space_key: str, ancestors: list, version: int) -> dict:
        return {
            "id": page_id,
            "type": "page",
            "status": "current",
            "title": title,
            "space": {"key": space_key},
            "version": {"number": version},
            "ancestors": [{"id": str(a.get("id"))} for a in ancestors if isinstance(a, dict) and a.get("id")],
            "body": {"storage": {"value": storage, "representation": "storage"},
                     "view": {"value": storage, "representation": "view"}},
            "_links": {"webui": f"/pages/viewpage.action?pageId={page_id}"},
        }

    def get(self, page_id: str) -> Optional[dict]:
        with self.lock:
            page = self.pages.get(str(page_id))
            return copy.deepcopy(page) if page else None

    def find(self, space_key: str, title: str) -> list[dict]:
        with self.lock:
            return [copy.deepcopy(p) for p in self.pages.values()
                    if p["title"] == title and (not space_key or p["space"]["key"] == space_key)]

    def create(self, payload: dict) -> tuple[int, dict]:
        title = str(payload.get("title") or "").strip()
        if not title:
            return 400, {"statusCode": 400, "message": "A page must have a title"}
        space_key = str((payload.get("space") or {}).get("key") or self.space_key)
        storage = str((((payload.get("body") or {}).get("storage") or {}).get("value")) or "")
        with self.lock:
            if any(p["title"] == title and p["space"]["key"] == space_key for p in self.pages.values()):
                return 400, {"statusCode": 400, "message": f"A page with this title already exists: {title}"}
            page_id = str(next(self._ids))
            self.pages[page_id] = self._page(page_id, title, storage, space_key, payload.get("ancestors") or [], 1)
            return 200, copy.deepcopy(self.pages[page_id])

    def update(self, page_id: str, payload: dict) -> tuple[int, dict]:
        with self.lock:
            page = self.pages.get(str(page_id))
            if page is None:
                return 404, {"statusCode": 404, "message": f"No content with id {page_id}"}
            wanted = int(((payload.get("version") or {}).get("number")) or 0)
            current = int(page["version"]["number"])
            if wanted != current + 1:
                return 409, {"statusCode": 409, "message": f"Version must be incremented on update. Current version is: {current}"}
            storage = str((((payload.get("body") or {}).get("storage") or {}).get("value")) or "")
            updated = self._page(page["id"], str(payload.get("title") or page["title"]), storage, page["space"]["key"],
                                 payload.get("ancestors") or page["ancestors"], wanted)
            self.pages[page["id"]] = updated
            return 200, copy.deepcopy(updated)

    def delete(self, content_id: str) -> int:
        with self.lock:
            if self.attachments.pop(str(content_id), None) is not None:
                return 204
            if str(content_id) != SEED_PAGE_ID and self.pages.pop(str(content_id), None) is not None:
                for att_id in [a for a, item in self.attachments.items() if item["page_id"] == str(content_id)]:
                    self.attachments.pop(att_id, None)
                return 204
            return 404

    def list_attachments(self, page_id: str) -> Optional[list[dict]]:
        with self.lock:
            if str(page_id) not in self.pages:
                return None
            return [self._attachment_view(a) for a in self.attachments.values() if a["page_id"] == str(page_id)]

    def add_attachment(self, page_id: str, filename: str, size: int, attachment_id: str = "") -> tuple[int, dict]:
        with self.lock:
            if str(page_id) not in self.pages:
                return 404, {"statusCode": 404, "message": f"No content with id {page_id}"}
            if attachment_id:
                item = self.attachments.get(attachment_id)
                if item is None or item["page_id"] != str(page_id):
                    return 404, {"statusCode": 404, "message": f"No attachment with id {attachment_id}"}
                item.update(size=size, version=item["version"] + 1)
                return 200, self._attachment_view(item)
            if any(a["page_id"] == str(page_id) and a["title"] == filename for a in self.attachments.values()):
                return 400, {"statusCode": 400, "message": f"Cannot add a new attachment with same file name as an existing attachment: {filename}"}
            att_id = f"att{next(self._ids)}"
            self.attachments[att_id] = {"id": att_id, "page_id": str(page_id), "title": filename, "size": size, "version": 1}
            return 200, {"results": [self._attachment_view(self.attachments[att_id])], "size": 1}

    @staticmethod
    def _attachment_view(item: dict) -> dict:
        return {"id": item["id"], "type": "attachment", "title": item["title"], "version": {"number": item["version"]},
                "extensions": {"fileSize": item["size"]}}

    def summary(self) -> list[dict]:
        with self.lock:
            out = []
            for page in self.pages.values():
                out.append({
                    "id": page["id"], "title": page["title"], "space": page["space"]["key"],
                    "version": page["version"]["number"], "ancestors": [a["id"] for a in page["ancestors"]],
                    "body_chars": len(page["body"]["storage"]["value"]),
                    "attachments": sorted(a["title"] for a in self.attachments.values() if a["page_id"] == page["id"]),
                })
            return out


def multipart_filename(body: bytes) -> str:
    match = _FILENAME.search(body or b"")
    return match.group(1).decode("utf-8", "replace") if match else "file"


def expected_auth(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")


# ---- HTTP -------------------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "ConfluenceStub/1"
    options: argparse.Namespace
    store: PageStore
    counters: dict
    counters_lock = threading.Lock()
    log_lock = threading.Lock()

    def _send(self, status: int, payload: Any = None, headers: Optional[dict] = None) -> None:
        data = b"" if status == 204 else json.dumps(payload if payload is not None else {}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if status != 204:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _count(self, kind: str) -> None:
        with self.counters_lock:
            key = f"{self.command} {kind}"
            self.counters[key] = self.counters.get(key, 0) + 1

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _json(self, raw: bytes) -> dict:
        try:
            value = json.loads(raw or b"{}")
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    def _route(self) -> None:
        parts = urlsplit(self.path)
        path = parts.path.rstrip("/") or "/"
        query = {k: v[-1] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
        raw = self._body() if self.command in ("POST", "PUT") else b""
        auth_ok = self.headers.get("Authorization") == expected_auth(self.options.user, self.options.password)
        if self.options.log:
            with self.log_lock, open(self.options.log, "a", encoding="utf-8") as log:
                log.write(json.dumps({"method": self.command, "path": self.path, "auth_ok": auth_ok}, ensure_ascii=False) + "\n")
        if path.startswith("/_stub/"):
            return self._stub(path)
        if self.options.delay_ms > 0:
            time.sleep(self.options.delay_ms / 1000.0)
        if not auth_ok:
            self._count("unauthorized")
            return self._send(401, {"statusCode": 401, "message": "unauthorized"})
        if path == "/x/Ab42":
            self._count("short link")
            return self._send(302, {}, {"Location": f"/pages/viewpage.action?pageId={SEED_PAGE_ID}"})
        space = re.fullmatch(r"/rest/api/space/([^/]+)", path)
        if space and self.command == "GET":
            self._count("space")
            return self._send(200, {"key": space.group(1), "name": f"Stub space {space.group(1)}", "type": "global"})
        if path == "/rest/api/content":
            if self.command == "GET":
                self._count("search")
                results = self.store.find(query.get("spaceKey", ""), query.get("title", ""))
                return self._send(200, {"results": results, "size": len(results), "start": 0, "limit": 25})
            if self.command == "POST":
                self._count("create page")
                return self._send(*self.store.create(self._json(raw)))
        attachment = re.fullmatch(r"/rest/api/content/([^/]+)/child/attachment(?:/([^/]+)/data)?", path)
        if attachment:
            page_id, att_id = attachment.group(1), attachment.group(2) or ""
            if self.command == "GET" and not att_id:
                self._count("list attachments")
                items = self.store.list_attachments(page_id)
                if items is None:
                    return self._send(404, {"statusCode": 404, "message": f"No content with id {page_id}"})
                return self._send(200, {"results": items, "size": len(items)})
            if self.command == "POST":
                if self.headers.get("X-Atlassian-Token") != "no-check":
                    return self._send(403, {"statusCode": 403, "message": "XSRF check failed"})
                self._count("upload attachment")
                return self._send(*self.store.add_attachment(page_id, multipart_filename(raw), len(raw), att_id))
        content = re.fullmatch(r"/rest/api/content/([^/]+)", path)
        if content:
            content_id = content.group(1)
            if self.command == "GET":
                self._count("get page")
                page = self.store.get(content_id)
                return self._send(200, page) if page else self._send(404, {"statusCode": 404, "message": f"No content with id {content_id}"})
            if self.command == "PUT":
                self._count("update page")
                return self._send(*self.store.update(content_id, self._json(raw)))
            if self.command == "DELETE":
                self._count("delete")
                status = self.store.delete(content_id)
                return self._send(status, None if status == 204 else {"statusCode": status, "message": "not found"})
        self._count("unknown")
        return self._send(404, {"statusCode": 404, "message": f"not found: {self.command} {path}"})

    def _stub(self, path: str) -> None:
        if path == "/_stub/pages" and self.command == "GET":
            return self._send(200, {"pages": self.store.summary()})
        one = re.fullmatch(r"/_stub/pages/([^/]+)", path)
        if one and self.command == "GET":
            page = self.store.get(one.group(1))
            return self._send(200, page) if page else self._send(404, {"message": "no such page"})
        if path == "/_stub/stats" and self.command == "GET":
            with self.counters_lock:
                return self._send(200, {"requests": dict(self.counters), "pages": len(self.store.summary())})
        if path == "/_stub/reset" and self.command == "POST":
            self.store.reset()
            with self.counters_lock:
                self.counters.clear()
            return self._send(200, {"status": "reset"})
        return self._send(404, {"message": "unknown stub path"})

    def do_GET(self):  # noqa: N802 (http.server API)
        self._route()

    def do_POST(self):  # noqa: N802 (http.server API)
        self._route()

    def do_PUT(self):  # noqa: N802 (http.server API)
        self._route()

    def do_DELETE(self):  # noqa: N802 (http.server API)
        self._route()

    def log_message(self, *args):
        pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confluence stub for the LoadLens sandbox (see the module docstring).")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (local only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default="reader")
    parser.add_argument("--password", default="secret")
    parser.add_argument("--space", default="PERF", help="space key of the seeded page 42")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="extra latency of every REST call, ms")
    parser.add_argument("--log", help="append one JSON line per request to this file")
    args = parser.parse_args(argv)
    if args.host not in LOCAL_HOSTS:
        parser.error("the stub listens on 127.0.0.1/localhost only")
    return args


def make_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"options": args, "store": PageStore(args.space), "counters": {}})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    return server


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    server = make_server(args)
    print(f"confluence stub on http://{args.host}:{args.port} (basic auth {args.user}/***, space {args.space})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
