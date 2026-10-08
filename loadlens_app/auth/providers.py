"""Pluggable authentication providers.

A provider verifies credentials and answers with either a stored ``User`` (the local provider,
which owns the password hashes) or an ``ExternalIdentity`` (LDAP, OIDC, ...). The service then maps
an external identity onto an ``app_users`` row (just-in-time provisioning) and issues the session,
so roles, lockout, tokens and audit behave the same whatever verified the password.

Adding a provider::

    class LdapProvider(AuthProvider):
        name = "ldap"
        label = "Корпоративная учётная запись"

        def authenticate(self, username, password):
            ...  # bind to the directory
            return ProviderResult(ProviderStatus.OK, identity=ExternalIdentity(username=..., role=...))

    register_provider("ldap", lambda ctx: LdapProvider(ctx.config))

and list it in ``auth.providers`` (e.g. ``["ldap", "local"]`` keeps a local break-glass admin).
Redirect-based flows (OIDC) can reuse ``AuthService.login_with_identity`` from their own callback view.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional

from . import passwords
from .config import AuthSettings
from .models import ExternalIdentity, User, utcnow
from .repository import UserRepository

LOCAL_PROVIDER = "local"


class ProviderStatus(enum.Enum):
    OK = "ok"
    INVALID = "invalid"  # credentials rejected
    UNAVAILABLE = "unavailable"  # the provider could not be reached


@dataclass(frozen=True)
class ProviderResult:
    status: ProviderStatus
    user: Optional[User] = None
    identity: Optional[ExternalIdentity] = None


INVALID = ProviderResult(ProviderStatus.INVALID)


class AuthProvider(abc.ABC):
    name: str = ""
    label: str = ""

    @abc.abstractmethod
    def authenticate(self, username: str, password: str) -> ProviderResult: ...


class LocalProvider(AuthProvider):
    """Passwords stored in ``app_users``, with per-account lockout."""

    name = LOCAL_PROVIDER
    label = "Локальная учётная запись"

    def __init__(
        self, repository: UserRepository, settings: AuthSettings, clock: Callable[[], datetime] = utcnow
    ) -> None:
        self._repo = repository
        self._settings = settings
        self._clock = clock

    def authenticate(self, username: str, password: str) -> ProviderResult:
        user = self._repo.get_by_username(username)
        if user is None or user.provider != LOCAL_PROVIDER or not user.password_hash:
            passwords.burn_time(password)
            return INVALID
        now = self._clock()
        if not user.is_active or user.is_locked(now):
            passwords.burn_time(password)
            return INVALID
        if passwords.verify_password(user.password_hash, password):
            return ProviderResult(ProviderStatus.OK, user=user)
        self._repo.register_failed_login(
            user.id,
            max_failures=self._settings.max_failed_logins,
            lock_until=now + timedelta(minutes=self._settings.lockout_minutes),
        )
        return INVALID


@dataclass(frozen=True)
class ProviderContext:
    repository: UserRepository
    settings: AuthSettings
    config: Mapping[str, Any]
    clock: Callable[[], datetime] = utcnow


ProviderFactory = Callable[[ProviderContext], AuthProvider]
_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name.strip().lower()] = factory


def build_providers(names: tuple[str, ...], context: ProviderContext) -> list[AuthProvider]:
    providers: list[AuthProvider] = []
    for name in names:
        factory = _REGISTRY.get(name)
        if factory is None:
            raise ValueError(f"Неизвестный провайдер авторизации: {name!r} (доступны: {', '.join(sorted(_REGISTRY))})")
        providers.append(factory(context))
    return providers


register_provider(LOCAL_PROVIDER, lambda ctx: LocalProvider(ctx.repository, ctx.settings, ctx.clock))
