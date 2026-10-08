"""Password hashing and policy."""

from __future__ import annotations

import hashlib
import threading

from werkzeug.security import check_password_hash, generate_password_hash

MAX_PASSWORD_LENGTH = 256
_COMMON_PASSWORDS = frozenset(
    {
        "password",
        "password1",
        "password123",
        "1234567890",
        "12345678910",
        "qwertyuiop",
        "qwerty12345",
        "1q2w3e4r5t",
        "iloveyou123",
        "admin12345",
        "administrator",
        "loadlens123",
        "letmein1234",
    }
)

_dummy_lock = threading.Lock()
_dummy_hash: str = ""


class PasswordPolicyError(ValueError):
    """The password does not satisfy the policy; the message is safe to show to the user."""


def _method() -> str:
    # scrypt is the Werkzeug default; some stripped-down Python builds lack it.
    return "scrypt" if hasattr(hashlib, "scrypt") else "pbkdf2:sha256:600000"


def hash_password(password: str) -> str:
    return generate_password_hash(password, method=_method())


def verify_password(stored_hash: str, password: str) -> bool:
    if not stored_hash or not isinstance(password, str) or len(password) > MAX_PASSWORD_LENGTH * 4:
        return False
    try:
        return bool(check_password_hash(stored_hash, password))
    except (ValueError, TypeError):
        return False


def burn_time(password: str) -> None:
    """Spends the cost of one verification so unknown users cannot be told apart by latency."""
    global _dummy_hash
    with _dummy_lock:
        if not _dummy_hash:
            _dummy_hash = hash_password("loadlens-timing-equalizer")
        stored = _dummy_hash
    verify_password(stored, password if isinstance(password, str) else "")


def validate_password(password: object, *, username: str = "", min_length: int = 10) -> str:
    """Returns the password when acceptable, otherwise raises PasswordPolicyError."""
    if not isinstance(password, str) or not password:
        raise PasswordPolicyError("Укажите пароль")
    if not password.strip():
        raise PasswordPolicyError("Пароль не может состоять только из пробелов")
    if len(password) < min_length:
        raise PasswordPolicyError(f"Пароль должен содержать не менее {min_length} символов")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Пароль не должен быть длиннее {MAX_PASSWORD_LENGTH} символов")
    lowered = password.lower()
    if len(username) >= 3 and username.lower() in lowered:
        raise PasswordPolicyError("Пароль не должен содержать имя пользователя")
    if lowered in _COMMON_PASSWORDS or len(set(password)) == 1:
        raise PasswordPolicyError("Слишком простой пароль, выберите другой")
    return password
