"""Authentication and user-management logic, independent of Flask."""

from __future__ import annotations

import enum
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from . import passwords
from .config import AuthSettings
from .models import ApiToken, AuditEvent, ExternalIdentity, Role, User, utcnow
from .providers import LOCAL_PROVIDER, AuthProvider, ProviderStatus
from .repository import DuplicateUserError, UserRepository
from .throttle import LoginThrottle
from .tokens import generate_token, hash_token, looks_like_token

logger = logging.getLogger(__name__)

_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{2,63}$")
_MAX_DISPLAY_NAME = 120
_MAX_EMAIL = 254
_MAX_TOKEN_NAME = 80
_TOKEN_TOUCH_INTERVAL = 60.0


class AuthError(Exception):
    """Business-rule failure; the message is safe to show to the user."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, status_code: Optional[int] = None, code: Optional[str] = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class ValidationError(AuthError):
    status_code = 400
    code = "invalid"


class NotFoundError(AuthError):
    status_code = 404
    code = "not_found"


class ConflictError(AuthError):
    status_code = 409
    code = "conflict"


class ForbiddenError(AuthError):
    status_code = 403
    code = "forbidden"


@dataclass(frozen=True)
class Actor:
    """Who performs an operation, for the audit trail."""

    user: Optional[User] = None
    ip: str = ""
    via: str = "session"
    name: str = ""  # used when there is no user (CLI, bootstrap)

    @property
    def label(self) -> str:
        return self.user.username if self.user else self.name


class LoginStatus(enum.Enum):
    OK = "ok"
    INVALID = "invalid"
    THROTTLED = "throttled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class LoginOutcome:
    status: LoginStatus
    user: Optional[User] = None
    retry_after: float = 0.0


@dataclass(frozen=True)
class TokenIdentity:
    user: User
    token: ApiToken
    role: Role  # effective: the lower of the token's and the owner's role


class AuthService:
    def __init__(
        self,
        settings: AuthSettings,
        repository: UserRepository,
        providers: list[AuthProvider],
        *,
        clock: Callable[[], datetime] = utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.repo = repository
        self.providers = providers
        self._clock = clock
        self._monotonic = monotonic
        self._throttle = LoginThrottle(
            max_failures=settings.login_max_attempts, window_seconds=settings.login_window_seconds, clock=monotonic
        )
        self._cache: dict[int, tuple[float, Optional[User]]] = {}
        self._cache_lock = threading.Lock()
        self._token_touched: dict[int, float] = {}
        self._bootstrap_lock = threading.Lock()
        self._bootstrapped = False
        self._users_exist = False

    # ---- audit -------------------------------------------------------------

    def audit(
        self,
        action: str,
        actor: Optional[Actor] = None,
        *,
        target: str = "",
        success: bool = True,
        details: Optional[dict] = None,
    ) -> None:
        """Best effort: a failing audit store must never break the operation being audited."""
        actor = actor or Actor()
        try:
            self.repo.add_audit(
                AuditEvent(
                    action=action,
                    actor_id=actor.user.id if actor.user else None,
                    actor_name=actor.label,
                    target=target,
                    success=success,
                    ip=actor.ip,
                    via=actor.via,
                    details=details or {},
                )
            )
        except Exception:
            logger.warning("auth: failed to write audit event %s", action, exc_info=True)

    # ---- user cache --------------------------------------------------------

    def _cached_user(self, user_id: int) -> Optional[User]:
        ttl = self.settings.user_cache_seconds
        now = self._monotonic()
        if ttl > 0:
            with self._cache_lock:
                hit = self._cache.get(user_id)
            if hit is not None and now - hit[0] < ttl:
                return hit[1]
        user = self.repo.get_user(user_id)
        if ttl > 0:
            with self._cache_lock:
                if len(self._cache) > 1000:
                    self._cache.clear()
                self._cache[user_id] = (now, user)
        return user

    def invalidate_user(self, user_id: Optional[int] = None) -> None:
        """Drops cached user records; other worker processes catch up within ``user_cache_seconds``."""
        with self._cache_lock:
            if user_id is None:
                self._cache.clear()
            else:
                self._cache.pop(user_id, None)

    def session_user(self, user_id: int, epoch: int) -> Optional[User]:
        user = self._cached_user(user_id)
        if user is None or not user.is_active or user.session_epoch != epoch:
            return None
        return user

    # ---- login -------------------------------------------------------------

    def login(self, username: str, password: str, *, ip: str = "") -> LoginOutcome:
        name = (username or "").strip()
        if not name or not isinstance(password, str) or not password or len(name) > 128 or len(password) > 1024:
            return LoginOutcome(LoginStatus.INVALID)
        key = f"{ip}|{name.lower()}"
        wait = self._throttle.retry_after(key)
        if wait > 0:
            return LoginOutcome(LoginStatus.THROTTLED, retry_after=wait)

        unavailable = False
        for provider in self.providers:
            try:
                result = provider.authenticate(name, password)
            except Exception:
                logger.exception("auth: provider %s failed", provider.name)
                unavailable = True
                continue
            if result.status is ProviderStatus.UNAVAILABLE:
                unavailable = True
                continue
            if result.status is not ProviderStatus.OK:
                continue
            try:
                user = result.user or self._provision(provider.name, result.identity)
            except AuthError as exc:
                logger.warning("auth: provider %s identity rejected: %s", provider.name, exc)
                continue
            except Exception:
                logger.exception("auth: provisioning for provider %s failed", provider.name)
                unavailable = True
                continue
            if user is None or not user.is_active:
                continue
            self._finish_login(user, key, ip, provider.name)
            return LoginOutcome(LoginStatus.OK, user=user)

        if unavailable:
            return LoginOutcome(LoginStatus.UNAVAILABLE)
        self._throttle.register_failure(key)
        self.audit("login.failed", Actor(ip=ip, name=name), success=False)
        return LoginOutcome(LoginStatus.INVALID)

    def login_with_identity(self, provider: str, identity: ExternalIdentity, *, ip: str = "") -> User:
        """Entry point for redirect-based providers (OIDC callback): maps the identity and records the login."""
        user = self._provision(provider, identity)
        if not user.is_active:
            raise ForbiddenError("Учётная запись отключена")
        self._finish_login(user, f"{ip}|{identity.username.lower()}", ip, provider)
        return user

    def _finish_login(self, user: User, throttle_key: str, ip: str, provider: str) -> None:
        try:
            self.repo.record_login_success(user.id)
        except Exception:
            logger.warning("auth: failed to record login for user %s", user.id, exc_info=True)
        self._throttle.reset(throttle_key)
        self.invalidate_user(user.id)
        self.audit("login", Actor(user=user, ip=ip), details={"provider": provider})

    def _provision(self, provider: str, identity: Optional[ExternalIdentity]) -> User:
        """Finds or creates the stored user for an externally verified identity."""
        if identity is None:
            raise ValidationError("Провайдер не вернул учётные данные")
        username = self._clean_username(identity.username)
        external_id = (identity.external_id or username).strip().lower()
        user = self.repo.get_by_external(provider, external_id)
        if user is None:
            # Never attach an external identity to an existing account of another provider by name.
            if self.repo.get_by_username(username) is not None:
                raise ConflictError(f"Логин {username!r} уже занят другой учётной записью")
            try:
                user = self.repo.create_user(
                    username=username,
                    role=identity.role or self.settings.default_external_role,
                    display_name=(identity.display_name or "")[:_MAX_DISPLAY_NAME],
                    email=(identity.email or "")[:_MAX_EMAIL],
                    provider=provider,
                    external_id=external_id,
                )
            except DuplicateUserError:
                user = self.repo.get_by_external(provider, external_id)
                if user is None:
                    raise ConflictError(f"Логин {username!r} уже занят другой учётной записью") from None
            else:
                self.audit("user.provision", Actor(name=provider), target=username, details={"role": user.role.key})
            return user

        updates: dict[str, Any] = {}
        if identity.display_name and identity.display_name != user.display_name:
            updates["display_name"] = identity.display_name[:_MAX_DISPLAY_NAME]
        if identity.email and identity.email != user.email:
            updates["email"] = identity.email[:_MAX_EMAIL]
        role_changed = identity.role is not None and identity.role != user.role
        if role_changed:
            updates["role"] = identity.role
        if updates:
            user = self.repo.update_user(user.id, updates, bump_epoch=role_changed) or user
            self.invalidate_user(user.id)
        return user

    # ---- bootstrap ---------------------------------------------------------

    def ensure_bootstrap(self) -> None:
        """Creates the first administrator from LOADLENS_ADMIN_USER / LOADLENS_ADMIN_PASSWORD.

        Runs lazily (on the login page) instead of at startup, so a database that is still coming
        up does not break application start. Does nothing once any user exists.
        """
        cfg = self.settings
        if self._bootstrapped or not cfg.bootstrap_username or not cfg.bootstrap_password:
            return
        with self._bootstrap_lock:
            if self._bootstrapped:
                return
            try:
                if self.repo.count_users() == 0:
                    self.create_user(
                        username=cfg.bootstrap_username,
                        role=Role.ADMIN,
                        password=cfg.bootstrap_password,
                        must_change_password=False,
                        actor=Actor(name="bootstrap"),
                    )
                    logger.info("auth: created the initial administrator %r", cfg.bootstrap_username)
                self._bootstrapped = True
            except AuthError as exc:
                logger.error("auth: LOADLENS_ADMIN_USER/PASSWORD rejected: %s", exc)
                self._bootstrapped = True
            except Exception:
                logger.warning("auth: initial administrator was not created, will retry", exc_info=True)

    def has_users(self) -> bool:
        if self._users_exist:
            return True
        try:
            self._users_exist = self.repo.count_users() > 0
        except Exception:
            return True  # unknown: do not claim that the system is empty
        return self._users_exist

    # ---- user management ---------------------------------------------------

    @staticmethod
    def _clean_username(raw: object) -> str:
        name = str(raw or "").strip()
        if not _USERNAME_RE.match(name):
            raise ValidationError(
                "Логин: от 3 до 64 символов, латиница, цифры и . _ @ -, начинается с буквы или цифры"
            )
        return name

    @staticmethod
    def _clean_text(raw: object, limit: int, label: str) -> str:
        text = str(raw or "").strip()
        if len(text) > limit:
            raise ValidationError(f"{label}: не более {limit} символов")
        return text

    def _clean_email(self, raw: object) -> str:
        email = self._clean_text(raw, _MAX_EMAIL, "Email")
        if email and not re.match(r"^[^@\s]+@[^@\s]+$", email):
            raise ValidationError("Email указан в неверном формате")
        return email

    @staticmethod
    def _parse_role(raw: object) -> Role:
        try:
            return Role.parse(raw)
        except ValueError as exc:
            raise ValidationError(str(exc)) from None

    def _validate_password(self, password: object, username: str) -> str:
        try:
            return passwords.validate_password(password, username=username, min_length=self.settings.password_min_length)
        except passwords.PasswordPolicyError as exc:
            raise ValidationError(str(exc), code="weak_password") from None

    def _require_user(self, user_id: int) -> User:
        user = self.repo.get_user(user_id)
        if user is None:
            raise NotFoundError("Пользователь не найден")
        return user

    def list_users(self) -> list[User]:
        return self.repo.list_users()

    def create_user(
        self,
        *,
        username: str,
        role: object,
        password: str,
        display_name: str = "",
        email: str = "",
        must_change_password: bool = True,
        actor: Optional[Actor] = None,
    ) -> User:
        name = self._clean_username(username)
        parsed_role = self._parse_role(role)
        password = self._validate_password(password, name)
        try:
            user = self.repo.create_user(
                username=name,
                role=parsed_role,
                password_hash=passwords.hash_password(password),
                display_name=self._clean_text(display_name, _MAX_DISPLAY_NAME, "Имя"),
                email=self._clean_email(email),
                provider=LOCAL_PROVIDER,
                must_change_password=must_change_password,
            )
        except DuplicateUserError:
            raise ConflictError("Пользователь с таким логином уже существует") from None
        self.audit("user.create", actor, target=name, details={"role": parsed_role.key})
        return user

    def update_user(
        self,
        user_id: int,
        *,
        actor: Actor,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[object] = None,
        is_active: Optional[bool] = None,
    ) -> User:
        target = self._require_user(user_id)
        fields: dict[str, Any] = {}
        if display_name is not None:
            fields["display_name"] = self._clean_text(display_name, _MAX_DISPLAY_NAME, "Имя")
        if email is not None:
            fields["email"] = self._clean_email(email)
        security_change = False
        if role is not None:
            new_role = self._parse_role(role)
            if new_role != target.role:
                fields["role"] = new_role
                security_change = True
        if is_active is not None and bool(is_active) != target.is_active:
            fields["is_active"] = bool(is_active)
            security_change = True

        if actor.user is not None and actor.user.id == target.id and fields.get("is_active") is False:
            raise ForbiddenError("Нельзя отключить собственную учётную запись")
        loses_admin = target.role is Role.ADMIN and target.is_active and (
            fields.get("role", target.role) is not Role.ADMIN or not fields.get("is_active", target.is_active)
        )
        if loses_admin and self.repo.count_active_admins(exclude_id=target.id) == 0:
            raise ConflictError("В системе должен остаться хотя бы один активный администратор")
        if not fields:
            return target

        updated = self.repo.update_user(target.id, fields, bump_epoch=security_change)
        if updated is None:
            raise NotFoundError("Пользователь не найден")
        self.invalidate_user(target.id)
        details: dict[str, Any] = {"fields": sorted(fields)}
        if "role" in fields:
            details["role"] = fields["role"].key
        if "is_active" in fields:
            details["is_active"] = fields["is_active"]
        self.audit("user.update", actor, target=target.username, details=details)
        return updated

    def delete_user(self, user_id: int, *, actor: Actor) -> None:
        target = self._require_user(user_id)
        if actor.user is not None and actor.user.id == target.id:
            raise ForbiddenError("Нельзя удалить собственную учётную запись")
        if target.role is Role.ADMIN and target.is_active and self.repo.count_active_admins(exclude_id=target.id) == 0:
            raise ConflictError("В системе должен остаться хотя бы один активный администратор")
        self.repo.delete_user(target.id)
        self.invalidate_user(target.id)
        self.audit("user.delete", actor, target=target.username)

    def reset_password(self, user_id: int, new_password: str, *, actor: Actor, must_change: bool = True) -> User:
        target = self._require_user(user_id)
        if target.provider != LOCAL_PROVIDER:
            raise ValidationError("Пароль этой учётной записи управляется внешним провайдером")
        password = self._validate_password(new_password, target.username)
        self.repo.set_password(target.id, passwords.hash_password(password), must_change)
        self.invalidate_user(target.id)
        self.audit("user.password_reset", actor, target=target.username)
        return self._require_user(target.id)

    def change_own_password(self, user: User, current_password: str, new_password: str, *, actor: Actor) -> User:
        fresh = self._require_user(user.id)
        if fresh.provider != LOCAL_PROVIDER or not fresh.password_hash:
            raise ValidationError("Пароль этой учётной записи управляется внешним провайдером")
        now = self._clock()
        if fresh.is_locked(now):
            raise ForbiddenError("Учётная запись временно заблокирована из-за неудачных попыток", code="locked")
        if not passwords.verify_password(fresh.password_hash, current_password):
            # Counts towards the account lockout, so a stolen session cannot be used to guess the password.
            self.repo.register_failed_login(
                fresh.id,
                max_failures=self.settings.max_failed_logins,
                lock_until=now + timedelta(minutes=self.settings.lockout_minutes),
            )
            self.audit("password.change", actor, target=fresh.username, success=False)
            raise ValidationError("Текущий пароль указан неверно", code="wrong_password")
        if current_password == new_password:
            raise ValidationError("Новый пароль должен отличаться от текущего")
        password = self._validate_password(new_password, fresh.username)
        self.repo.set_password(fresh.id, passwords.hash_password(password), False)
        self.invalidate_user(fresh.id)
        self.audit("password.change", actor, target=fresh.username)
        return self._require_user(fresh.id)

    def unlock_user(self, user_id: int, *, actor: Actor) -> User:
        target = self._require_user(user_id)
        self.repo.unlock_user(target.id)
        self.invalidate_user(target.id)
        self.audit("user.unlock", actor, target=target.username)
        return self._require_user(target.id)

    # ---- API tokens --------------------------------------------------------

    def create_token(
        self,
        user: User,
        name: str,
        *,
        actor: Actor,
        role: Optional[object] = None,
        expires_in_days: Optional[int] = None,
    ) -> tuple[ApiToken, str]:
        if not self.settings.api_tokens:
            raise ForbiddenError("API-токены отключены")
        label = self._clean_text(name, _MAX_TOKEN_NAME, "Название")
        if not label:
            raise ValidationError("Укажите название токена")
        token_role = self._parse_role(role) if role else user.role
        if token_role > user.role:
            raise ForbiddenError("Роль токена не может быть выше вашей роли")
        days = self.settings.token_default_days if expires_in_days is None else expires_in_days
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= self.settings.token_max_days:
            raise ValidationError(f"Срок действия токена: от 1 до {self.settings.token_max_days} дней")
        plain, prefix, digest = generate_token()
        token = self.repo.create_token(
            user_id=user.id,
            name=label,
            prefix=prefix,
            token_hash=digest,
            role=token_role,
            expires_at=self._clock() + timedelta(days=days),
        )
        token.owner = user.username
        self.audit("token.create", actor, target=label, details={"token_id": token.id, "role": token_role.key, "days": days})
        return token, plain

    def list_tokens(self, user: User, *, include_all: bool = False) -> list[ApiToken]:
        everyone = include_all and user.role is Role.ADMIN
        return self.repo.list_tokens(None if everyone else user.id)

    def revoke_token(self, user: User, token_id: int, *, actor: Actor) -> None:
        # Administrators may revoke anybody's token; everyone else only their own.
        if not self.repo.revoke_token(token_id, None if user.role is Role.ADMIN else user.id):
            raise NotFoundError("Токен не найден или уже отозван")
        self.audit("token.revoke", actor, details={"token_id": token_id})

    def authenticate_token(self, raw: str) -> Optional[TokenIdentity]:
        if not self.settings.api_tokens or not looks_like_token(raw):
            return None
        token = self.repo.get_token_by_hash(hash_token(raw))
        if token is None or not token.is_usable(self._clock()):
            return None
        user = self._cached_user(token.user_id)
        if user is None or not user.is_active:
            return None
        self._touch_token(token.id)
        return TokenIdentity(user=user, token=token, role=min(token.role, user.role))

    def _touch_token(self, token_id: int) -> None:
        now = self._monotonic()
        last = self._token_touched.get(token_id)
        if last is not None and now - last < _TOKEN_TOUCH_INTERVAL:
            return
        if len(self._token_touched) > 1000:
            self._token_touched.clear()
        self._token_touched[token_id] = now
        try:
            self.repo.touch_token(token_id)
        except Exception:
            logger.debug("auth: failed to update token usage", exc_info=True)
