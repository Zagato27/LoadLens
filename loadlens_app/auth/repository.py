"""Storage interface for users, API tokens and the audit log, plus an in-memory implementation."""

from __future__ import annotations

import abc
import copy
import threading
from datetime import datetime
from typing import Callable, Optional

from .models import ApiToken, AuditEvent, Role, User, utcnow

# Fields update_user() may change; everything else goes through dedicated methods.
UPDATABLE_USER_FIELDS = ("display_name", "email", "role", "is_active")


class DuplicateUserError(Exception):
    """A user with this username (case-insensitive) or external id already exists."""


class UserRepository(abc.ABC):
    # ---- users -----------------------------------------------------------

    @abc.abstractmethod
    def count_users(self) -> int: ...

    @abc.abstractmethod
    def count_active_admins(self, exclude_id: Optional[int] = None) -> int: ...

    @abc.abstractmethod
    def get_user(self, user_id: int) -> Optional[User]: ...

    @abc.abstractmethod
    def get_by_username(self, username: str) -> Optional[User]:
        """Case-insensitive lookup."""

    @abc.abstractmethod
    def get_by_external(self, provider: str, external_id: str) -> Optional[User]: ...

    @abc.abstractmethod
    def list_users(self) -> list[User]: ...

    @abc.abstractmethod
    def create_user(
        self,
        *,
        username: str,
        role: Role,
        password_hash: Optional[str] = None,
        display_name: str = "",
        email: str = "",
        provider: str = "local",
        external_id: Optional[str] = None,
        must_change_password: bool = False,
        is_active: bool = True,
    ) -> User: ...

    @abc.abstractmethod
    def update_user(self, user_id: int, fields: dict, *, bump_epoch: bool = False) -> Optional[User]:
        """Applies UPDATABLE_USER_FIELDS; ``bump_epoch`` invalidates the user's browser sessions."""

    @abc.abstractmethod
    def delete_user(self, user_id: int) -> bool: ...

    @abc.abstractmethod
    def set_password(self, user_id: int, password_hash: str, must_change_password: bool) -> None:
        """Replaces the hash, clears lockout counters and invalidates browser sessions."""

    @abc.abstractmethod
    def register_failed_login(self, user_id: int, *, max_failures: int, lock_until: datetime) -> None:
        """Counts a failure; on reaching ``max_failures`` locks the account and restarts the counter."""

    @abc.abstractmethod
    def record_login_success(self, user_id: int) -> None: ...

    @abc.abstractmethod
    def unlock_user(self, user_id: int) -> None: ...

    # ---- API tokens -------------------------------------------------------

    @abc.abstractmethod
    def create_token(
        self, *, user_id: int, name: str, prefix: str, token_hash: str, role: Role, expires_at: Optional[datetime]
    ) -> ApiToken: ...

    @abc.abstractmethod
    def get_token_by_hash(self, token_hash: str) -> Optional[ApiToken]: ...

    @abc.abstractmethod
    def list_tokens(self, user_id: Optional[int] = None) -> list[ApiToken]:
        """Tokens of one user, or of everybody when ``user_id`` is None; newest first."""

    @abc.abstractmethod
    def revoke_token(self, token_id: int, user_id: Optional[int] = None) -> bool:
        """Revokes an active token; with ``user_id`` only if it belongs to that user."""

    @abc.abstractmethod
    def touch_token(self, token_id: int) -> None: ...

    # ---- audit ------------------------------------------------------------

    @abc.abstractmethod
    def add_audit(self, event: AuditEvent) -> None: ...

    @abc.abstractmethod
    def list_audit(self, *, limit: int = 100, offset: int = 0, action: str = "", actor: str = "") -> list[AuditEvent]:
        """Newest first; ``action`` matches a prefix, ``actor`` an exact username."""


class InMemoryRepository(UserRepository):
    """Thread-safe volatile store used by tests and for embedding without a database."""

    def __init__(self, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._users: dict[int, User] = {}
        self._tokens: dict[int, ApiToken] = {}
        self._audit: list[AuditEvent] = []
        self._next_user = 1
        self._next_token = 1
        self._next_audit = 1

    # ---- users -----------------------------------------------------------

    def count_users(self) -> int:
        with self._lock:
            return len(self._users)

    def count_active_admins(self, exclude_id: Optional[int] = None) -> int:
        with self._lock:
            return sum(
                1 for u in self._users.values() if u.role is Role.ADMIN and u.is_active and u.id != exclude_id
            )

    def get_user(self, user_id: int) -> Optional[User]:
        with self._lock:
            user = self._users.get(user_id)
            return copy.copy(user) if user else None

    def get_by_username(self, username: str) -> Optional[User]:
        wanted = str(username or "").lower()
        with self._lock:
            for user in self._users.values():
                if user.username.lower() == wanted:
                    return copy.copy(user)
        return None

    def get_by_external(self, provider: str, external_id: str) -> Optional[User]:
        with self._lock:
            for user in self._users.values():
                if user.provider == provider and user.external_id == external_id:
                    return copy.copy(user)
        return None

    def list_users(self) -> list[User]:
        with self._lock:
            return [copy.copy(u) for u in sorted(self._users.values(), key=lambda u: u.username.lower())]

    def create_user(
        self,
        *,
        username: str,
        role: Role,
        password_hash: Optional[str] = None,
        display_name: str = "",
        email: str = "",
        provider: str = "local",
        external_id: Optional[str] = None,
        must_change_password: bool = False,
        is_active: bool = True,
    ) -> User:
        with self._lock:
            if self.get_by_username(username) is not None:
                raise DuplicateUserError(username)
            if external_id and self.get_by_external(provider, external_id) is not None:
                raise DuplicateUserError(external_id)
            now = self._clock()
            user = User(
                id=self._next_user,
                username=username,
                role=role,
                display_name=display_name,
                email=email,
                provider=provider,
                external_id=external_id,
                password_hash=password_hash,
                is_active=is_active,
                must_change_password=must_change_password,
                created_at=now,
                updated_at=now,
            )
            self._next_user += 1
            self._users[user.id] = user
            return copy.copy(user)

    def update_user(self, user_id: int, fields: dict, *, bump_epoch: bool = False) -> Optional[User]:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                return None
            for key, value in fields.items():
                if key in UPDATABLE_USER_FIELDS:
                    setattr(user, key, value)
            if bump_epoch:
                user.session_epoch += 1
            user.updated_at = self._clock()
            return copy.copy(user)

    def delete_user(self, user_id: int) -> bool:
        with self._lock:
            removed = self._users.pop(user_id, None)
            for token_id in [t.id for t in self._tokens.values() if t.user_id == user_id]:
                del self._tokens[token_id]
            return removed is not None

    def set_password(self, user_id: int, password_hash: str, must_change_password: bool) -> None:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                return
            user.password_hash = password_hash
            user.must_change_password = must_change_password
            user.session_epoch += 1
            user.failed_logins = 0
            user.locked_until = None
            user.updated_at = self._clock()

    def register_failed_login(self, user_id: int, *, max_failures: int, lock_until: datetime) -> None:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                return
            if user.failed_logins + 1 >= max_failures:
                user.locked_until = lock_until
                user.failed_logins = 0
            else:
                user.failed_logins += 1

    def record_login_success(self, user_id: int) -> None:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                return
            user.failed_logins = 0
            user.locked_until = None
            user.last_login_at = self._clock()

    def unlock_user(self, user_id: int) -> None:
        with self._lock:
            user = self._users.get(user_id)
            if user is not None:
                user.failed_logins = 0
                user.locked_until = None

    # ---- API tokens -------------------------------------------------------

    def create_token(
        self, *, user_id: int, name: str, prefix: str, token_hash: str, role: Role, expires_at: Optional[datetime]
    ) -> ApiToken:
        with self._lock:
            token = ApiToken(
                id=self._next_token,
                user_id=user_id,
                name=name,
                prefix=prefix,
                role=role,
                token_hash=token_hash,
                created_at=self._clock(),
                expires_at=expires_at,
            )
            self._next_token += 1
            self._tokens[token.id] = token
            return self._with_owner(token)

    def _with_owner(self, token: ApiToken) -> ApiToken:
        copied = copy.copy(token)
        owner = self._users.get(token.user_id)
        copied.owner = owner.username if owner else ""
        return copied

    def get_token_by_hash(self, token_hash: str) -> Optional[ApiToken]:
        with self._lock:
            for token in self._tokens.values():
                if token.token_hash == token_hash:
                    return self._with_owner(token)
        return None

    def list_tokens(self, user_id: Optional[int] = None) -> list[ApiToken]:
        with self._lock:
            tokens = [t for t in self._tokens.values() if user_id is None or t.user_id == user_id]
            ordered = sorted(tokens, key=lambda t: t.id, reverse=True)
            return [self._with_owner(t) for t in ordered]

    def revoke_token(self, token_id: int, user_id: Optional[int] = None) -> bool:
        with self._lock:
            token = self._tokens.get(token_id)
            if token is None or token.revoked_at is not None:
                return False
            if user_id is not None and token.user_id != user_id:
                return False
            token.revoked_at = self._clock()
            return True

    def touch_token(self, token_id: int) -> None:
        with self._lock:
            token = self._tokens.get(token_id)
            if token is not None:
                token.last_used_at = self._clock()

    # ---- audit ------------------------------------------------------------

    def add_audit(self, event: AuditEvent) -> None:
        with self._lock:
            stored = copy.copy(event)
            stored.id = self._next_audit
            stored.created_at = self._clock()
            self._next_audit += 1
            self._audit.append(stored)

    def list_audit(self, *, limit: int = 100, offset: int = 0, action: str = "", actor: str = "") -> list[AuditEvent]:
        with self._lock:
            events = [
                e
                for e in reversed(self._audit)
                if (not action or e.action.startswith(action)) and (not actor or e.actor_name == actor)
            ]
            return [copy.copy(e) for e in events[offset : offset + limit]]
