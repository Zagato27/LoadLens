"""Endpoints: login/logout, account (password, API tokens), user administration, audit log."""

from __future__ import annotations

import math
from typing import Any

import psycopg2
from flask import Blueprint, jsonify, make_response, redirect, render_template, request, session

from .models import Role
from .service import AuthError, LoginStatus
from .web import current_actor, current_auth, csrf_token, establish_session, get_service, json_error, safe_next

auth_bp = Blueprint("auth", __name__)

_INVALID_LOGIN = "Неверный логин или пароль, либо учётная запись временно заблокирована"


def _body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@auth_bp.errorhandler(AuthError)
def _auth_error(exc: AuthError):
    return json_error(str(exc), exc.status_code, exc.code)


@auth_bp.errorhandler(psycopg2.Error)
def _database_error(exc: psycopg2.Error):
    return json_error("База данных пользователей недоступна", 503, "auth_unavailable")


def _require_user():
    """The signed-in user, or an error response when authentication is switched off."""
    user = current_auth().user
    if user is None:
        return None, json_error("Недоступно: авторизация отключена", 400, "auth_disabled")
    return user, None


# ---- liveness ----------------------------------------------------------------------------------


@auth_bp.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


# ---- login / logout ------------------------------------------------------------------------------


def _login_page(error: str = "", username: str = "", next_url: str = "/") -> str:
    service = get_service()
    return render_template(
        "login.html",
        error=error,
        username=username,
        next_url=next_url,
        no_users=not service.has_users(),
        providers=[p.label for p in service.providers if p.label],
        active_nav="",
    )


@auth_bp.route("/login", methods=["GET"])
def login_page():
    service = get_service()
    if not service.settings.enabled:
        return redirect("/")
    service.ensure_bootstrap()
    next_url = safe_next(request.args.get("next"))
    if current_auth().authenticated:
        return redirect(next_url)
    return _login_page(next_url=next_url)


@auth_bp.route("/login", methods=["POST"])
def login_submit():
    service = get_service()
    if not service.settings.enabled:
        return redirect("/", 303)
    as_json = request.is_json
    payload = _body() if as_json else request.form
    username = str(payload.get("username") or "")
    password = payload.get("password")
    next_url = safe_next(payload.get("next"))

    service.ensure_bootstrap()
    outcome = service.login(username, password if isinstance(password, str) else "", ip=request.remote_addr or "")
    if outcome.status is LoginStatus.OK:
        user = outcome.user
        establish_session(user)
        target = "/account?force=1" if user.must_change_password else next_url
        if as_json:
            return jsonify({"ok": True, "redirect": target, "user": user.public_dict(), "csrf_token": session["csrf"]})
        return redirect(target, 303)

    if outcome.status is LoginStatus.THROTTLED:
        wait = max(1, math.ceil(outcome.retry_after))
        message = f"Слишком много попыток входа. Повторите через {math.ceil(wait / 60)} мин."
        status, code = 429, "throttled"
    elif outcome.status is LoginStatus.UNAVAILABLE:
        message, status, code, wait = "Сервис авторизации временно недоступен", 503, "auth_unavailable", 0
    else:
        message, status, code, wait = _INVALID_LOGIN, 401, "invalid_credentials", 0

    if as_json:
        response = json_error(message, status, code)
    else:
        response = make_response(_login_page(error=message, username=username, next_url=next_url), status)
    if wait:
        response.headers["Retry-After"] = str(wait)
    return response


@auth_bp.route("/logout", methods=["POST"])
def logout():
    auth = current_auth()
    if auth.user is not None and auth.via == "session":
        get_service().audit("logout", current_actor())
    session.clear()
    if request.is_json:
        return jsonify({"ok": True, "redirect": "/login"})
    return redirect("/login", 303)


@auth_bp.route("/auth/me")
def me():
    """Session state for scripts. Also hands out the CSRF token that unsafe requests must send."""
    service = get_service()
    auth = current_auth()
    enabled = service.settings.enabled
    payload: dict[str, Any] = {
        "enabled": enabled,
        "authenticated": auth.authenticated,
        "role": auth.role.key if (auth.authenticated or not enabled) else None,
        "user": auth.user.public_dict() if auth.user else None,
        "via": auth.via,
    }
    if enabled:
        payload["csrf_token"] = csrf_token()
    return jsonify(payload)


# ---- account ---------------------------------------------------------------------------------------


@auth_bp.route("/account")
def account_page():
    service = get_service()
    auth = current_auth()
    if not service.settings.enabled or auth.user is None:
        return redirect("/")
    settings = service.settings
    return render_template(
        "account.html",
        active_nav="account",
        force_change=bool(request.args.get("force")) or auth.user.must_change_password,
        password_min_length=settings.password_min_length,
        tokens_enabled=settings.api_tokens,
        token_default_days=settings.token_default_days,
        token_max_days=settings.token_max_days,
        roles=[{"key": r.key, "label": r.label} for r in Role if r <= auth.user.role],
        can_see_all_tokens=auth.user.role is Role.ADMIN,
    )


@auth_bp.route("/auth/password", methods=["POST"])
def change_password():
    user, error = _require_user()
    if error:
        return error
    body = _body()
    updated = get_service().change_own_password(
        user, str(body.get("current_password") or ""), str(body.get("new_password") or ""), actor=current_actor()
    )
    # The change bumps the session epoch; keep this browser signed in, everything else is signed out.
    session["epoch"] = updated.session_epoch
    return jsonify({"ok": True})


@auth_bp.route("/auth/tokens", methods=["GET"])
def list_tokens():
    user, error = _require_user()
    if error:
        return error
    service = get_service()
    tokens = service.list_tokens(user, include_all=request.args.get("all") == "1")
    return jsonify(
        {
            "tokens": [t.public_dict() for t in tokens],
            "enabled": service.settings.api_tokens,
            "default_days": service.settings.token_default_days,
            "max_days": service.settings.token_max_days,
        }
    )


@auth_bp.route("/auth/tokens", methods=["POST"])
def create_token():
    user, error = _require_user()
    if error:
        return error
    body = _body()
    days = body.get("expires_in_days")
    token, plain = get_service().create_token(
        user,
        str(body.get("name") or ""),
        actor=current_actor(),
        role=body.get("role") or None,
        expires_in_days=days if days not in (None, "") else None,
    )
    return jsonify({"token": plain, "info": token.public_dict()}), 201


@auth_bp.route("/auth/tokens/<int:token_id>", methods=["DELETE"])
def revoke_token(token_id: int):
    user, error = _require_user()
    if error:
        return error
    get_service().revoke_token(user, token_id, actor=current_actor())
    return jsonify({"ok": True})


# ---- user administration -----------------------------------------------------------------------------


@auth_bp.route("/auth/users", methods=["GET"])
def list_users():
    users = get_service().list_users()
    return jsonify(
        {
            "users": [u.public_dict() for u in users],
            "roles": [{"key": r.key, "label": r.label} for r in Role],
            "current_user_id": current_auth().user.id if current_auth().user else None,
        }
    )


@auth_bp.route("/auth/users", methods=["POST"])
def create_user():
    body = _body()
    user = get_service().create_user(
        username=str(body.get("username") or ""),
        role=body.get("role") or Role.VIEWER.key,
        password=body.get("password"),
        display_name=str(body.get("display_name") or ""),
        email=str(body.get("email") or ""),
        must_change_password=bool(body.get("must_change_password", True)),
        actor=current_actor(),
    )
    return jsonify({"user": user.public_dict()}), 201


@auth_bp.route("/auth/users/<int:user_id>", methods=["PATCH"])
def update_user(user_id: int):
    body = _body()
    changes = {key: body[key] for key in ("display_name", "email", "role", "is_active") if key in body}
    user = get_service().update_user(user_id, actor=current_actor(), **changes)
    return jsonify({"user": user.public_dict()})


@auth_bp.route("/auth/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id: int):
    get_service().delete_user(user_id, actor=current_actor())
    return jsonify({"ok": True})


@auth_bp.route("/auth/users/<int:user_id>/password", methods=["POST"])
def reset_user_password(user_id: int):
    body = _body()
    user = get_service().reset_password(
        user_id,
        body.get("new_password"),
        actor=current_actor(),
        must_change=bool(body.get("must_change", True)),
    )
    return jsonify({"user": user.public_dict()})


@auth_bp.route("/auth/users/<int:user_id>/unlock", methods=["POST"])
def unlock_user(user_id: int):
    user = get_service().unlock_user(user_id, actor=current_actor())
    return jsonify({"user": user.public_dict()})


@auth_bp.route("/auth/audit", methods=["GET"])
def audit_log():
    def _int(name: str, default: int, low: int, high: int) -> int:
        try:
            return min(high, max(low, int(request.args.get(name, default))))
        except (TypeError, ValueError):
            return default

    events = get_service().repo.list_audit(
        limit=_int("limit", 100, 1, 500),
        offset=_int("offset", 0, 0, 1_000_000),
        action=(request.args.get("action") or "").strip(),
        actor=(request.args.get("actor") or "").strip(),
    )
    return jsonify({"events": [e.public_dict() for e in events]})
