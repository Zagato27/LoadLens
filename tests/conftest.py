"""Shared pytest setup."""

import os

# Sessions need a stable key. Setting it before the app is imported keeps the test run from
# creating .loadlens_secret_key in the repository root.
os.environ.setdefault("LOADLENS_SECRET_KEY", "pytest-secret-key-not-for-production")
# The load test watcher polls the storage database in a thread; tests drive it with Watcher.tick().
os.environ["LOADLENS_LOAD_TEST_WATCHER"] = "0"

import pytest


@pytest.fixture(autouse=True)
def _fresh_llm_budget(monkeypatch):
    """Each test starts with an unused daily LLM budget."""
    from loadlens_app import limits

    monkeypatch.setattr(limits, "llm_budget", limits.DailyBudget())
    monkeypatch.delenv("LOADLENS_RUN_MIGRATIONS", raising=False)


@pytest.fixture(autouse=True)
def _storage_breaker_closed():
    """The storage breaker is process state: every test starts and ends with it closed."""
    from AI import storage_state

    storage_state.BREAKER.reset()
    storage_state.reset_pools()  # no pooled (possibly fake) session crosses a test boundary
    yield
    storage_state.BREAKER.reset()
    storage_state.reset_pools()


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch):
    """The API tests call endpoints directly; tests/test_auth*.py switch authentication back on."""
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    for name in ("LOADLENS_ADMIN_USER", "LOADLENS_ADMIN_PASSWORD", "LOADLENS_COOKIE_SECURE", "LOADLENS_TRUSTED_PROXIES"):
        monkeypatch.delenv(name, raising=False)
