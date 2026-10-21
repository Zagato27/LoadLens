# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the Docker Engine and Kubernetes APIs: TLS material from the settings, readable errors."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

import requests

from .base import GeneratorError

DEFAULT_TIMEOUT_SEC = 15
_PEM_DIR: Optional[str] = None
_PEM_LOCK = threading.Lock()


class ApiError(GeneratorError):
    """The API answered with an error status; ``status`` keeps the HTTP code for the callers' decisions."""

    def __init__(self, message: Any, status: int = 0, body: str = "", **params: Any) -> None:
        super().__init__(message, **params)
        self.status = status
        self.body = body


def pem_file(text: str, suffix: str) -> str:
    """Path of a private temporary file holding ``text``; the same content reuses one file per process.

    requests reads certificates and keys from files only. The directory is created with mode 0700 and the
    files with 0600, so other users of the LoadLens host cannot read the key.
    """
    global _PEM_DIR
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    with _PEM_LOCK:
        if _PEM_DIR is None or not os.path.isdir(_PEM_DIR):
            _PEM_DIR = tempfile.mkdtemp(prefix="loadlens-tls-")
            os.chmod(_PEM_DIR, 0o700)
        path = os.path.join(_PEM_DIR, f"{digest}{suffix}")
        if not os.path.exists(path):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text.strip() + "\n")
    return path


def _host(url: str) -> str:
    try:
        parts = urlsplit(url)
        return f"{parts.hostname}:{parts.port}" if parts.port else str(parts.hostname or url)
    except ValueError:
        return url


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        name: str,
        verify: bool = True,
        ca_cert: str = "",
        ca_file: str = "",
        client_cert: str = "",
        client_key: str = "",
        token: str = "",
        auth: Optional[tuple[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self.session = requests.Session()
        if not verify:
            self.session.verify = False
        elif ca_cert.strip():
            self.session.verify = pem_file(ca_cert, ".ca.pem")
        elif ca_file:
            self.session.verify = ca_file
        if client_cert.strip() and client_key.strip():
            self.session.cert = (pem_file(client_cert, ".cert.pem"), pem_file(client_key, ".key.pem"))
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        if auth:
            self.session.auth = auth

    def url_for(self, path: str) -> str:
        """Absolute URL of an API path; absolute URLs (from API answers) must point under the base URL.

        Credentials of the session go with every request, so a URL that an answer names (Jenkins
        ``Location``, ``executable.url``) is never followed to another host or scheme.
        """
        if not path.lower().startswith(("http://", "https://")):
            return f"{self.base_url}{path}"
        if path == self.base_url or path.startswith(self.base_url + "/"):
            return path
        raise GeneratorError("lt.http.foreign_url", name=self.name, host=_host(self.base_url))

    def request(
        self,
        method: str,
        path: str,
        *,
        ok: Iterable[int] = (200, 201, 202, 204),
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> requests.Response:
        url = self.url_for(path)
        try:
            response = self.session.request(method, url, timeout=timeout or self.timeout, **kwargs)
        except requests.exceptions.SSLError as exc:
            raise GeneratorError("lt.http.tls", name=self.name, host=_host(url), error=str(exc)) from exc
        except requests.exceptions.Timeout as exc:
            raise GeneratorError("lt.http.timeout", name=self.name, host=_host(url)) from exc
        except requests.exceptions.ConnectionError as exc:
            raise GeneratorError("lt.http.connect", name=self.name, host=_host(url), error=str(exc)) from exc
        except requests.exceptions.RequestException as exc:
            # Broken chunked answers, bad redirects and the like: still a generator problem, not a watcher crash.
            raise GeneratorError("lt.http.request", name=self.name, host=_host(url), error=str(exc)) from exc
        if response.status_code in tuple(ok):
            return response
        body = response.text[:500]
        key = "lt.http.denied" if response.status_code in (401, 403) else "lt.http.status"
        raise ApiError(key, response.status_code, body, name=self.name, code=response.status_code, detail=_message(body))

    def json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.request(method, path, **kwargs)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise GeneratorError("lt.http.not_json", name=self.name, body=response.text[:200]) from exc


def _message(body: str) -> str:
    """The ``message`` field of a JSON error body, or the body itself."""
    import json

    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:300]
    if isinstance(data, Mapping):
        return str(data.get("message") or data.get("error") or data)[:300]
    return str(data)[:300]


__all__ = ["ApiClient", "ApiError", "pem_file"]
