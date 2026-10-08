"""Value objects shared by the auth package."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(enum.IntEnum):
    """Access levels; a higher value includes everything below it."""

    VIEWER = 1
    ENGINEER = 2
    ADMIN = 3

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def label(self) -> str:
        return ROLE_LABELS[self]

    @classmethod
    def parse(cls, value: Any) -> "Role":
        if isinstance(value, Role):
            return value
        text = str(value or "").strip().lower()
        for role in cls:
            if role.key == text:
                return role
        raise ValueError(f"Неизвестная роль: {value!r}")


ROLE_LABELS = {
    Role.VIEWER: "Наблюдатель",
    Role.ENGINEER: "Инженер",
    Role.ADMIN: "Администратор",
}


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


@dataclass
class User:
    id: int
    username: str
    role: Role
    display_name: str = ""
    email: str = ""
    provider: str = "local"
    external_id: Optional[str] = None
    password_hash: Optional[str] = field(default=None, repr=False)
    is_active: bool = True
    must_change_password: bool = False
    session_epoch: int = 0
    failed_logins: int = 0
    locked_until: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def title(self) -> str:
        return self.display_name or self.username

    def is_locked(self, now: Optional[datetime] = None) -> bool:
        return bool(self.locked_until and self.locked_until > (now or utcnow()))

    def public_dict(self, now: Optional[datetime] = None) -> dict:
        """Safe for API responses: never includes the password hash."""
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "email": self.email,
            "role": self.role.key,
            "role_label": self.role.label,
            "provider": self.provider,
            "is_active": self.is_active,
            "must_change_password": self.must_change_password,
            "locked": self.is_locked(now),
            "has_password": bool(self.password_hash),
            "last_login_at": _iso(self.last_login_at),
            "created_at": _iso(self.created_at),
        }


@dataclass
class ApiToken:
    id: int
    user_id: int
    name: str
    prefix: str
    role: Role
    token_hash: str = field(repr=False)
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    owner: str = ""

    def is_usable(self, now: Optional[datetime] = None) -> bool:
        current = now or utcnow()
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > current

    def public_dict(self, now: Optional[datetime] = None) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "prefix": self.prefix,
            "role": self.role.key,
            "owner": self.owner,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "last_used_at": _iso(self.last_used_at),
            "revoked_at": _iso(self.revoked_at),
            "active": self.is_usable(now),
        }


@dataclass
class AuditEvent:
    action: str
    actor_id: Optional[int] = None
    actor_name: str = ""
    target: str = ""
    success: bool = True
    ip: str = ""
    via: str = "session"
    details: dict = field(default_factory=dict)
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": _iso(self.created_at),
            "actor": self.actor_name,
            "actor_id": self.actor_id,
            "action": self.action,
            "target": self.target,
            "success": self.success,
            "ip": self.ip,
            "via": self.via,
            "details": self.details,
        }


@dataclass(frozen=True)
class ExternalIdentity:
    """A user verified by an external provider (LDAP, OIDC, ...)."""

    username: str
    external_id: str = ""
    display_name: str = ""
    email: str = ""
    # Role granted by the identity provider (e.g. from a group mapping); None keeps the stored/default role.
    role: Optional[Role] = None
