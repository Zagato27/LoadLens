# SPDX-License-Identifier: Apache-2.0
"""Uniform answers for unexpected failures: details go to the server log, the client gets a short message.

Exception text can carry database hosts, SQL, file paths or bodies of upstream responses, so it is
never sent to the browser. The error id in the message ties what the user sees to the log record.

``message`` arguments are dictionary keys (``api.error.*``): the client gets the text in the locale of the
request, the log line the English one. A text that is not a key is used as is in both places.
"""

from __future__ import annotations

import logging
import uuid

from flask import jsonify

from AI.i18n import has_key, t as translate
from AI.storage_state import is_storage_error

logger = logging.getLogger("loadlens_app.errors")

GENERIC_MESSAGE = "api.error.internal"
STORAGE_MESSAGE = "api.error.storage_unavailable"


def _text(message: str, locale: str | None = None) -> str:
    """The message of a key in ``locale`` (the request's when None); other text as it is."""
    return translate(message, locale=locale) if has_key(message, "ru") else message


def log_failure(exc: BaseException, message: str = GENERIC_MESSAGE) -> str:
    """Logs ``exc`` with its traceback and returns the short id to show to the user."""
    error_id = uuid.uuid4().hex[:8]
    logger.error("%s [error_id=%s]", _text(message, "en"), error_id, exc_info=(type(exc), exc, exc.__traceback__))
    return error_id


def with_error_id(message: str, error_id: str) -> str:
    """``<message>. Error ID: <id>`` in the locale of the request."""
    return translate("api.error.with_id", message=_text(message), error_id=error_id)


def storage_unavailable():
    """503 for a request that needed the storage database while it is down.

    The storage breaker has already logged the cause once, so no traceback is written per request.
    Browser pages get the outage page, everything else a JSON error.
    """
    try:
        from loadlens_app.auth.web import storage_unavailable_for_request

        return storage_unavailable_for_request()
    except Exception:  # no request context or auth not initialised (scripts, some tests)
        logger.debug("storage outage page unavailable", exc_info=True)
        return jsonify({"error": _text(STORAGE_MESSAGE), "code": "storage_unavailable"}), 503


def internal_error(exc: BaseException, message: str = GENERIC_MESSAGE, status: int = 500):
    """A JSON error response that names the failure without its details."""
    if is_storage_error(exc):
        return storage_unavailable()
    error_id = log_failure(exc, message)
    return jsonify({"error": with_error_id(message, error_id), "error_id": error_id}), status


__all__ = [
    "GENERIC_MESSAGE",
    "STORAGE_MESSAGE",
    "internal_error",
    "log_failure",
    "storage_unavailable",
    "with_error_id",
]
