# SPDX-License-Identifier: Apache-2.0
"""Recovery page: fix the storage connection while the database is down.

Reachable by a recovery session (see ``loadlens_app.auth.recovery``) and by administrators. The form
covers only the connection fields of ``storage.timescale``; table names, other sections, connections
and keys stay closed. A masked password is restored only for the stored host, port, database and user
(``restore_secrets``), so the saved password cannot be sent to an address typed into the form.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Optional

from flask import Blueprint, g, jsonify, redirect, render_template, request, session

from AI import storage_state
from AI.i18n import t as translate
from loadlens_app import core
from loadlens_app.auth import recovery
from loadlens_app.auth.web import current_auth, get_service
from loadlens_app.config_secrets import SecretPlaceholderError, mask_secrets, restore_secrets
from loadlens_app.connection_checks import run_check

recovery_bp = Blueprint("recovery", __name__)

SECTION = "storage.timescale"
FORM_FIELDS = ("host", "port", "dbname", "user", "password", "sslmode", "schema", "connect_timeout_sec")
SSL_MODES = ("disable", "allow", "prefer", "require", "verify-ca", "verify-full")
_HOST_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_:[]%")
_NAME_LIMIT = 63


def _stored() -> dict:
    node = (core.CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return copy.deepcopy(node) if isinstance(node, dict) else {}


def _in_recovery() -> bool:
    return current_auth().via == "recovery"


def _storage_available() -> bool:
    """Whether the storage answers now. With sign-in on, the same check the gate uses (the user store)."""
    service = get_service()
    if service.settings.enabled:
        return service.store_available()
    try:
        conn = core._ts_conn()
    except Exception as exc:
        if storage_state.is_storage_error(exc):
            return False
        raise
    try:
        conn.close()
    except Exception:
        pass
    return True


def _form_view(cfg: dict) -> dict:
    view = {key: cfg.get(key) for key in FORM_FIELDS if key in cfg}
    view.setdefault("sslmode", "prefer")
    view.setdefault("schema", "public")
    return mask_secrets(view)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _identifier(value: str) -> bool:
    return (
        0 < len(value) <= _NAME_LIMIT
        and value.isascii()
        and (value[0].isalpha() or value[0] == "_")
        and all(ch.isalnum() or ch == "_" for ch in value)
    )


def _normalized(values: dict) -> tuple[dict, str]:
    """Form values checked and converted; the error text is empty when they are valid."""
    out: dict = {}
    host = _text(values.get("host"))
    if not host:
        return {}, translate("api.recovery.host_required")
    if len(host) > 253 or any(ch not in _HOST_CHARS for ch in host):
        return {}, translate("api.recovery.host_format")
    out["host"] = host
    raw_port = values.get("port")
    port = _as_int(raw_port if raw_port not in (None, "") else 5432)
    if port is None or not 1 <= port <= 65535:
        return {}, translate("api.recovery.port")
    out["port"] = port
    for key, label_key in (("dbname", "api.recovery.field.dbname"), ("user", "api.recovery.field.user")):
        text = _text(values.get(key))
        if not text or len(text) > _NAME_LIMIT or any(ch.isspace() for ch in text):
            return {}, translate("api.recovery.name_format", label=translate(label_key), limit=_NAME_LIMIT)
        out[key] = text
    password = values.get("password")
    if password is not None and not isinstance(password, str):
        return {}, translate("api.recovery.password_string")
    out["password"] = password or ""
    sslmode = (_text(values.get("sslmode")) or "prefer").lower()
    if sslmode not in SSL_MODES:
        return {}, translate("api.recovery.sslmode", allowed=", ".join(SSL_MODES))
    out["sslmode"] = sslmode
    schema = _text(values.get("schema")) or "public"
    if not _identifier(schema):
        return {}, translate("api.recovery.schema")
    out["schema"] = schema
    timeout = values.get("connect_timeout_sec")
    if timeout not in (None, ""):
        seconds = _as_int(timeout)
        if seconds is None or not 2 <= seconds <= 60:
            return {}, translate("api.recovery.timeout")
        out["connect_timeout_sec"] = seconds
    return out, ""


def _candidate(body: object) -> tuple[Optional[dict], str]:
    """The stored section with the form fields applied and a masked password restored."""
    if not isinstance(body, dict):
        return None, translate("api.recovery.body")
    current = _stored()
    merged = {key: current.get(key) for key in FORM_FIELDS if key in current}
    merged.update({key: body[key] for key in FORM_FIELDS if key in body})
    values, error = _normalized(merged)
    if error:
        return None, error
    candidate = copy.deepcopy(current)
    candidate.update(values)
    if "connect_timeout_sec" not in values:
        candidate.pop("connect_timeout_sec", None)
    # The form replaces a connection string: otherwise reports would keep using the old address.
    candidate.pop("dsn", None)
    try:
        candidate = restore_secrets(candidate, current)
    except SecretPlaceholderError as exc:
        return None, str(exc)
    return candidate, ""


def _note(cfg: dict, **fields: Any) -> None:
    """Audit details: where the storage connection was probed or moved to (host, port, database only).

    The after-request hook writes them to the audit table, or to the application log while it is down.
    """
    g.audit_extra = {"section": SECTION, "targets": [f"{cfg.get('host')}:{cfg.get('port')}"], "database": cfg.get("dbname"), **fields}


def _session_left() -> Optional[int]:
    if not _in_recovery():
        return None
    issued = session.get("iat")
    if not isinstance(issued, int):
        return 0
    return max(0, recovery.SESSION_SECONDS - int(time.time() - issued))


def _status_payload(available: bool) -> dict:
    stored = _stored()
    status = storage_state.status().to_dict(details=True)
    status["available"] = available
    return {
        "storage": status,
        "form": _form_view(stored),
        "dsn_configured": bool(_text(stored.get("dsn"))),
        "ssl_modes": list(SSL_MODES),
        "recovery": _in_recovery(),
        "session_seconds_left": _session_left(),
        "connect_timeout_default": storage_state.DEFAULT_CONNECT_TIMEOUT_SEC,
    }


@recovery_bp.route("/recovery", methods=["GET"])
def recovery_page():
    in_recovery = _in_recovery()
    if not in_recovery and _storage_available():
        return redirect("/settings#connections")
    return render_template(
        "recovery.html",
        storage=storage_state.status(),
        in_recovery=in_recovery,
        session_minutes=recovery.SESSION_SECONDS // 60,
        active_nav="",
    )


@recovery_bp.route("/recovery/status", methods=["GET"])
def recovery_status():
    # A recovery session gets here only after the gate found the store down.
    available = False if _in_recovery() else _storage_available()
    return jsonify(_status_payload(available))


@recovery_bp.route("/recovery/storage/check", methods=["POST"])
def storage_check():
    candidate, error = _candidate(request.get_json(silent=True))
    if candidate is None:
        return jsonify({"error": error}), 400
    result = run_check(SECTION, candidate)
    _note(candidate, ok=result.ok)
    return jsonify(result.to_dict())


@recovery_bp.route("/recovery/storage/save", methods=["POST"])
def storage_save():
    candidate, error = _candidate(request.get_json(silent=True))
    if candidate is None:
        return jsonify({"error": error}), 400
    # Imported here: config_api pulls in the settings module and the whole configuration API.
    from loadlens_app.blueprints.config_api import store_section

    store_section(SECTION, candidate, "", "")
    storage_state.BREAKER.allow_retry_now()
    available = _storage_available()
    _note(candidate, available=available)
    if not available:
        return jsonify({"ok": True, "saved": True, **_status_payload(False)})
    if _in_recovery():
        recovery.log("session ended: storage is reachable again", ip=request.remote_addr or "")
        session.clear()
        return jsonify({"ok": True, "saved": True, "available": True, "redirect": "/login?recovered=1"})
    return jsonify({"ok": True, "saved": True, "available": True, "redirect": "/settings#connections"})


__all__ = ["recovery_bp", "FORM_FIELDS", "SSL_MODES"]
