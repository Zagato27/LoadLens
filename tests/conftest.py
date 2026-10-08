"""Shared pytest setup."""

import os

# Sessions need a stable key. Setting it before the app is imported keeps the test run from
# creating .loadlens_secret_key in the repository root.
os.environ.setdefault("LOADLENS_SECRET_KEY", "pytest-secret-key-not-for-production")

import pytest


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch):
    """The API tests call endpoints directly; tests/test_auth*.py switch authentication back on."""
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    for name in ("LOADLENS_ADMIN_USER", "LOADLENS_ADMIN_PASSWORD", "LOADLENS_COOKIE_SECURE", "LOADLENS_TRUSTED_PROXIES"):
        monkeypatch.delenv(name, raising=False)
