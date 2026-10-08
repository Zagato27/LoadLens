"""Flask integration: session cookie, request gate (authentication, roles, CSRF), audit, headers."""

from __future__ import annotations

import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import quote, urlsplit

from flask import Flask, Response, current_app, g, jsonify, redirect, render_template, request, session
from flask.sessions import SecureCookieSessionInterface
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import AuthSettings, load_auth_settings, resolve_secret_key
from .models import Role, User
from .policy import Rule, rule_for, unclassified
from .providers import ProviderContext, build_providers
from .repository import UserRepository
from .service import Actor, AuthService

logger = logging.getLogger(__name__)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
CSRF_HEADER = "X-CSRF-Token"
CSRF_FORM_FIELD = "csrf_token"
EXTENSION_KEY = "loadlens_auth"

# Request fields that identify the target of an audited change. Never add values that may hold secrets.
_AUDIT_BODY_KEYS = ("run_name", "service", "area", "section", "project_id", "service_id", "domain", "new_name", "run_id", "id")
_AUDIT_ARG_KEYS = ("with_data",)


@dataclass(frozen=True)
class AuthContext:
    user: Optional[User]
    role: Role
    via: str  # "session", "token", "disabled" or "anonymous"
    token_id: Optional[int] = None

    @property
    def authenticated(self) -> bool:
        return self.user is not None


_ANONYMOUS = AuthContext(user=None, role=Role.VIEWER, via="anonymous")


def get_service() -> AuthService:
    return current_app.extensions[EXTENSION_KEY]


def current_auth() -> AuthContext:
    return getattr(g, "auth", _ANONYMOUS)


def current_actor() -> Actor:
    auth = current_auth()
    return Actor(user=auth.user, ip=request.remote_addr or "", via=auth.via)


# ---- helpers ---------------------------------------------------------------------------------


def csrf_token() -> str:
    token = session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf"] = token
    return token


def _csrf_valid() -> bool:
    expected = session.get("csrf")
    supplied = request.headers.get(CSRF_HEADER) or request.form.get(CSRF_FORM_FIELD) or ""
    return bool(expected) and bool(supplied) and hmac.compare_digest(str(expected), str(supplied))


def safe_next(target: Optional[str], default: str = "/") -> str:
    """Only same-site relative paths: a login redirect must not be usable to bounce users elsewhere."""
    candidate = (target or "").strip()
    if not candidate.startswith("/") or candidate.startswith("//") or "\\" in candidate:
        return default
    # Browsers drop tabs and newlines inside a URL, so "/\t/host" would turn into "//host".
    if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        return default
    parts = urlsplit(candidate)
    if parts.scheme or parts.netloc:
        return default
    return candidate


def establish_session(user: User) -> None:
    """Starts a fresh session (new id and CSRF token) for the user; call after a successful login."""
    session.clear()
    session.permanent = True
    session["uid"] = user.id
    session["epoch"] = user.session_epoch
    session["iat"] = int(time.time())
    session["csrf"] = secrets.token_urlsafe(32)


def json_error(message: str, status: int, code: str, **extra: Any) -> Response:
    response = jsonify({"error": message, "code": code, **extra})
    response.status_code = status
    return response


def _unauthenticated(rule: Rule, *, bad_token: bool = False) -> Response:
    if rule.page and request.method in ("GET", "HEAD") and not bad_token:
        target = request.full_path.rstrip("?") if request.query_string else request.path
        return redirect(f"/login?next={quote(safe_next(target), safe='')}")
    response = json_error("Требуется вход в систему", 401, "unauthorized", login_url="/login")
    if bad_token:
        response.headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'
    return response


def _forbidden(rule: Rule, message: str = "Недостаточно прав для этого действия", code: str = "forbidden") -> Response:
    if rule.page and request.method in ("GET", "HEAD"):
        html = render_template("forbidden.html", message=message, active_nav="")
        return Response(html, status=403, mimetype="text/html")
    required = rule.role.key if rule.role else None
    return json_error(message, 403, code, required_role=required)


# ---- request gate ----------------------------------------------------------------------------


def _resolve_auth(service: AuthService, settings: AuthSettings) -> tuple[AuthContext, bool]:
    """Returns ``(context, bad_token)``; a presented but invalid bearer token never falls back to the cookie."""
    header = request.headers.get("Authorization", "")
    if header[:7].lower() == "bearer ":
        identity = service.authenticate_token(header[7:].strip())
        if identity is None:
            return _ANONYMOUS, True
        return AuthContext(user=identity.user, role=identity.role, via="token", token_id=identity.token.id), False

    uid, epoch, issued = session.get("uid"), session.get("epoch"), session.get("iat")
    if uid is None:
        return _ANONYMOUS, False
    expired = not isinstance(issued, int) or time.time() - issued > settings.session_max_hours * 3600
    user = None if expired else service.session_user(int(uid), int(epoch or 0))
    if user is None:
        session.clear()
        return _ANONYMOUS, False
    return AuthContext(user=user, role=user.role, via="session"), False


def _gate() -> Optional[Response]:
    service = get_service()
    settings = service.settings
    if not settings.enabled:
        g.auth = AuthContext(user=None, role=Role.ADMIN, via="disabled")
        return None
    g.auth = _ANONYMOUS
    endpoint = request.endpoint
    if endpoint is None:  # unknown URL or method: let Flask answer 404/405
        return None
    rule = rule_for(endpoint)

    try:
        context, bad_token = _resolve_auth(service, settings)
    except Exception:
        logger.exception("auth: failed to resolve the current user")
        if rule.role is None:
            return None
        return json_error("Сервис авторизации временно недоступен", 503, "auth_unavailable")
    g.auth = context

    if rule.role is not None:
        if not context.authenticated:
            return _unauthenticated(rule, bad_token=bad_token)
        if context.user.must_change_password and not rule.password_change_ok:
            if rule.page and request.method in ("GET", "HEAD"):
                return redirect("/account?force=1")
            return json_error("Необходимо сменить пароль", 403, "password_change_required", account_url="/account")

    if request.method not in SAFE_METHODS and rule.csrf and context.via != "token" and not _csrf_valid():
        return json_error("Сессия устарела, обновите страницу и повторите действие", 403, "csrf_failed")

    if rule.role is not None:
        if rule.session_only and context.via == "token":
            return _forbidden(rule, "Это действие доступно только из веб-интерфейса", "session_required")
        if context.role < rule.role:
            return _forbidden(rule)
    return None


# ---- response hooks --------------------------------------------------------------------------


def _audit_details(response: Response) -> dict:
    details: dict[str, str] = {}
    for key, value in (request.view_args or {}).items():
        details[key] = str(value)[:200]
    body = request.get_json(silent=True) if request.is_json else None
    if isinstance(body, dict):
        for key in _AUDIT_BODY_KEYS:
            value = body.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
                details.setdefault(key, str(value)[:200])
    for key in _AUDIT_ARG_KEYS:
        if request.args.get(key):
            details[key] = request.args[key][:50]
    if response.is_json and not response.direct_passthrough:
        payload = response.get_json(silent=True)
        if isinstance(payload, dict) and isinstance(payload.get("job_id"), str):
            details["job_id"] = payload["job_id"]
    return details


def _after_request(response: Response) -> Response:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    endpoint = request.endpoint or ""
    rule = rule_for(endpoint)
    # Pages and data behind the login must not be served from a browser or proxy cache afterwards
    # (for example with the back button after signing out). Public assets stay cacheable.
    sensitive = rule.role is not None or endpoint.startswith("auth.")
    if sensitive and endpoint != "auth.healthz":
        response.headers.setdefault("Cache-Control", "no-store")

    if rule.audit and request.method not in SAFE_METHODS and response.status_code < 400:
        try:
            service = get_service()
            if service.settings.enabled:
                service.audit(rule.audit, current_actor(), details=_audit_details(response))
        except Exception:
            logger.warning("auth: audit hook failed", exc_info=True)
    return response


class _SessionInterface(SecureCookieSessionInterface):
    """Cookie flags come from AuthSettings; ``Secure`` follows the request scheme unless forced."""

    def __init__(self, secure: Optional[bool]) -> None:
        super().__init__()
        self._secure = secure

    def get_cookie_secure(self, app: Flask) -> bool:
        return self._secure if self._secure is not None else bool(request.is_secure)


def _inject_template_context() -> dict:
    auth = current_auth()
    enabled = get_service().settings.enabled
    return {
        "auth_enabled": enabled,
        "current_user": auth.user,
        "role_rank": int(auth.role),
        "role_key": auth.role.key,
        "csrf_token": csrf_token() if enabled else "",
    }


# ---- setup -----------------------------------------------------------------------------------


def _default_repository() -> UserRepository:
    from loadlens_app import core
    from .pg_repository import PostgresRepository

    def schema() -> str:
        cfg = (core.CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
        return str(cfg.get("schema") or "public")

    # Looked up on every call so that tests and runtime reconfiguration of the DB take effect.
    return PostgresRepository(lambda: core._ts_conn(), schema)


def build_service(
    *,
    repository: Optional[UserRepository] = None,
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> AuthService:
    """Settings, repository and providers wired together; shared by the web app and the CLI."""
    if config is None:
        from loadlens_app import core

        config = core.CONFIG
    settings = load_auth_settings(config, env)
    repo = repository or _default_repository()
    providers = build_providers(settings.providers, ProviderContext(repository=repo, settings=settings, config=config))
    return AuthService(settings, repo, providers)


def init_auth(
    app: Flask,
    *,
    repository: Optional[UserRepository] = None,
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    key_file=None,
) -> AuthService:
    """Wires authentication into the app. Safe to call at import time: it never touches the database."""
    from . import views

    service = build_service(repository=repository, config=config, env=env)
    settings = service.settings
    app.extensions[EXTENSION_KEY] = service

    key, source = resolve_secret_key(settings, key_file)
    app.secret_key = key
    app.config.update(
        SESSION_COOKIE_NAME="loadlens_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=int(settings.session_hours * 3600),
        SESSION_REFRESH_EACH_REQUEST=True,
    )
    app.session_interface = _SessionInterface(settings.cookie_secure)
    if settings.trusted_proxies:
        n = settings.trusted_proxies
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=n, x_proto=n, x_host=n)  # type: ignore[method-assign]

    app.register_blueprint(views.auth_bp)
    app.before_request(_gate)
    app.after_request(_after_request)
    app.context_processor(_inject_template_context)

    if settings.enabled:
        logger.info("auth: enabled (providers=%s, session key from %s)", ",".join(settings.providers), source)
        missing = unclassified(name for name in app.view_functions)
        if missing:
            logger.warning("auth: endpoints without an access rule (admin only): %s", ", ".join(missing))
    else:
        logger.warning("auth: DISABLED, every visitor has administrator rights (LOADLENS_AUTH_ENABLED=0)")
    return service
