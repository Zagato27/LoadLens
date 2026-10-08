"""Authentication and authorization for the web app.

Local users with roles (viewer < engineer < admin), pluggable providers, API tokens, CSRF
protection and an audit log. See ``policy.py`` for the access table and ``providers.py`` for how
to add LDAP/OIDC. Entry point: ``init_auth(app)``.
"""

from .models import ApiToken, ExternalIdentity, Role, User
from .providers import AuthProvider, ProviderResult, ProviderStatus, register_provider
from .repository import InMemoryRepository, UserRepository
from .service import Actor, AuthError, AuthService
from .web import current_actor, current_auth, init_auth

__all__ = [
    "Actor",
    "ApiToken",
    "AuthError",
    "AuthProvider",
    "AuthService",
    "ExternalIdentity",
    "InMemoryRepository",
    "ProviderResult",
    "ProviderStatus",
    "Role",
    "User",
    "UserRepository",
    "current_actor",
    "current_auth",
    "init_auth",
    "register_provider",
]
