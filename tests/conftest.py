"""Shared pytest setup."""

import os

# Sessions need a stable key. Setting it before the app is imported keeps the test run from
# creating .loadlens_secret_key in the repository root.
os.environ.setdefault("LOADLENS_SECRET_KEY", "pytest-secret-key-not-for-production")
# The load test watcher polls the storage database in a thread; tests drive it with Watcher.tick().
os.environ["LOADLENS_LOAD_TEST_WATCHER"] = "0"

import psycopg2
import pytest

# settings.py of a working tree may point at a production database. A connection a test forgot to fake
# fails at once instead of reaching it; tests that need a database replace psycopg2.connect themselves.
_LOCAL_DB_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_real_psycopg2_connect = psycopg2.connect


def _local_only_connect(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
    if dsn or str(kwargs.get("host") or "") not in _LOCAL_DB_HOSTS:
        raise psycopg2.OperationalError("tests must not connect to a database outside 127.0.0.1")
    return _real_psycopg2_connect(dsn, connection_factory=connection_factory, cursor_factory=cursor_factory, **kwargs)


psycopg2.connect = _local_only_connect


@pytest.fixture(autouse=True)
def _fresh_llm_budget(monkeypatch):
    """Each test starts with an unused daily LLM budget."""
    from loadlens_app import limits

    monkeypatch.setattr(limits, "llm_budget", limits.DailyBudget())
    # Start-up migrations are on by default; a test app must not start a thread against the storage database.
    monkeypatch.setenv("LOADLENS_RUN_MIGRATIONS", "0")


@pytest.fixture(autouse=True)
def _runtime_config_cache_cleared():
    """The parsed runtime config files are cached per process: every test starts and ends without them."""
    import sys

    core = sys.modules.get("loadlens_app.core")
    if core is not None:
        core._clear_runtime_cache()
    yield
    core = sys.modules.get("loadlens_app.core")
    if core is not None:
        core._clear_runtime_cache()


@pytest.fixture(autouse=True)
def _response_cache_off_by_default(monkeypatch):
    """Polled answers are cached per process; tests swap the fakes behind an endpoint between two calls.

    The cache starts empty and switched off; tests/test_perf_polling.py switches it on for its own checks.
    """
    from loadlens_app.response_cache import response_cache

    monkeypatch.setenv("LOADLENS_RESPONSE_CACHE", "0")
    response_cache.clear()
    yield
    response_cache.clear()


@pytest.fixture(autouse=True)
def _manual_verdicts_table_known(monkeypatch):
    """As in a running app, where the request path creates it, the report_verdicts table of the public schema is
    known to exist, so fake connections see no extra probe. Tests of an older database clear the set."""
    from AI import db_store

    monkeypatch.setattr(db_store, "_MANUAL_VERDICTS_READY", {"public"})
    monkeypatch.setattr(db_store, "_ENSURED_TABLES", {key for key in db_store._ENSURED_TABLES if key[1] != db_store.REPORT_VERDICTS_TABLE})


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
def _jobs_run_in_threads_by_default(monkeypatch):
    """A shell with CELERY_BROKER_URL and eager mode off must not send test jobs to a real broker.

    Tests of the Celery path set LOADLENS_JOB_RUNNER (and the CELERY_* variables) themselves.
    """
    monkeypatch.setenv("LOADLENS_JOB_RUNNER", "thread")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "1")


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch):
    """The API tests call endpoints directly; tests/test_auth*.py switch authentication back on."""
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    for name in ("LOADLENS_ADMIN_USER", "LOADLENS_ADMIN_PASSWORD", "LOADLENS_COOKIE_SECURE", "LOADLENS_TRUSTED_PROXIES"):
        monkeypatch.delenv(name, raising=False)
