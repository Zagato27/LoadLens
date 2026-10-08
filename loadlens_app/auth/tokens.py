"""API token generation and hashing.

Tokens carry 256 bits of randomness, so a plain SHA-256 is enough for storage: only the digest
is kept, the plaintext is shown once at creation.
"""

from __future__ import annotations

import hashlib
import secrets

TOKEN_PREFIX = "ll_"
_DISPLAY_PREFIX_LENGTH = 11
_MAX_TOKEN_LENGTH = 200


def hash_token(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def generate_token() -> tuple[str, str, str]:
    """Returns ``(plaintext, display_prefix, digest)``."""
    plain = f"{TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    return plain, plain[:_DISPLAY_PREFIX_LENGTH], hash_token(plain)


def looks_like_token(value: str) -> bool:
    return value.startswith(TOKEN_PREFIX) and len(TOKEN_PREFIX) < len(value) <= _MAX_TOKEN_LENGTH
