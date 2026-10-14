"""Storage circuit breaker: short timeouts, fast failures while the database is down, one probe at a time."""

import logging
import sys
from pathlib import Path

import psycopg2
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AI import storage_state
from AI.storage_state import StorageBreaker, StorageUnavailableError


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeConn:
    closed = False

    def close(self):
        self.closed = True


REFUSED = 'connection to server at "db.test" (10.0.0.1), port 5432 failed: Connection refused'


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    breaker = StorageBreaker(clock=clock, wall=lambda: 1_700_000_000.0)
    monkeypatch.setattr(storage_state, "BREAKER", breaker)
    return clock


@pytest.fixture
def pg(monkeypatch):
    """psycopg2.connect replaced: records keyword arguments, fails while ``state['down']`` is set."""
    state = {"down": True, "calls": [], "error": REFUSED}

    def fake_connect(*args, **kwargs):
        state["calls"].append((args, kwargs))
        if state["down"]:
            raise psycopg2.OperationalError(state["error"])
        return FakeConn()

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    return state


PARAMS = {"host": "db.test", "port": 5432, "dbname": "lt", "user": "app", "password": "pw", "sslmode": "prefer"}


def test_connect_passes_short_timeout_and_parameters(clock, pg):
    pg["down"] = False
    conn = storage_state.connect(PARAMS, {})
    assert isinstance(conn, FakeConn)
    _, kwargs = pg["calls"][0]
    assert kwargs["connect_timeout"] == storage_state.DEFAULT_CONNECT_TIMEOUT_SEC == 5
    assert kwargs["host"] == "db.test" and kwargs["password"] == "pw"

    storage_state.connect(PARAMS, {"connect_timeout_sec": 600})
    assert pg["calls"][-1][1]["connect_timeout"] == 60  # clamped
    storage_state.connect(PARAMS, {"connect_timeout_sec": "junk"})
    assert pg["calls"][-1][1]["connect_timeout"] == 5


def test_dsn_is_passed_positionally_with_timeout(clock, pg):
    pg["down"] = False
    storage_state.connect({"dsn": "postgresql://app@db.test/lt"}, {"connect_timeout_sec": 3})
    args, kwargs = pg["calls"][0]
    assert args == ("postgresql://app@db.test/lt",) and kwargs == {"connect_timeout": 3}


def test_failure_opens_the_breaker_and_later_calls_fail_fast(clock, pg):
    with pytest.raises(StorageUnavailableError) as first:
        storage_state.connect(PARAMS, {"retry_seconds": 10})
    assert isinstance(first.value, psycopg2.OperationalError)  # old ``except psycopg2.Error`` still catches it
    assert storage_state.is_storage_error(first.value)
    assert storage_state.is_down()
    assert len(pg["calls"]) == 1

    for _ in range(5):
        with pytest.raises(StorageUnavailableError):
            storage_state.connect(PARAMS, {"retry_seconds": 10})
    assert len(pg["calls"]) == 1  # no network attempt inside the window

    clock.now += 10.5
    with pytest.raises(StorageUnavailableError):
        storage_state.connect(PARAMS, {"retry_seconds": 10})
    assert len(pg["calls"]) == 2  # one probe after the pause


def test_success_closes_the_breaker(clock, pg, caplog):
    with pytest.raises(StorageUnavailableError):
        storage_state.connect(PARAMS, {})
    pg["down"] = False
    clock.now += 11
    with caplog.at_level(logging.WARNING, logger="AI.storage_state"):
        storage_state.connect(PARAMS, {})
    assert not storage_state.is_down()
    status = storage_state.status()
    assert status.available is True and status.kind == "ok"
    assert "reachable again" in caplog.text


def test_only_one_caller_probes_while_others_fail_fast(clock, pg):
    with pytest.raises(StorageUnavailableError):
        storage_state.connect(PARAMS, {})
    clock.now += 11
    seen = []

    def slow_probe():
        # Another request arrives while this probe is still waiting for the network.
        with pytest.raises(StorageUnavailableError):
            storage_state.BREAKER.connect(lambda: seen.append("second") or FakeConn(), address="x", retry_seconds=10)
        return FakeConn()

    storage_state.BREAKER.connect(slow_probe, address="db.test:5432/lt", retry_seconds=10)
    assert seen == []
    assert not storage_state.is_down()


def test_allow_retry_now_skips_the_pause(clock, pg):
    with pytest.raises(StorageUnavailableError):
        storage_state.connect(PARAMS, {"retry_seconds": 300})
    storage_state.BREAKER.allow_retry_now()
    pg["down"] = False
    storage_state.connect(PARAMS, {"retry_seconds": 300})
    assert len(pg["calls"]) == 2 and not storage_state.is_down()


def test_retry_window_is_clamped(clock, pg):
    with pytest.raises(StorageUnavailableError):
        storage_state.connect(PARAMS, {"retry_seconds": 0.01})
    assert storage_state.status().retry_in == pytest.approx(1.0)
    clock.now += 1.01
    with pytest.raises(StorageUnavailableError):
        storage_state.connect(PARAMS, {"retry_seconds": 0.01})
    assert len(pg["calls"]) == 2


def test_status_hides_address_and_message_without_details(clock, pg):
    with pytest.raises(StorageUnavailableError):
        storage_state.connect(PARAMS, {})
    status = storage_state.status()
    assert status.kind == "refused" and status.address == "db.test:5432/lt"
    public = status.to_dict(details=False)
    assert "address" not in public and "message" not in public
    assert public["title"] and public["hint"] and public["available"] is False
    full = status.to_dict(details=True)
    assert full["address"] == "db.test:5432/lt" and "Connection refused" in full["message"]


def test_first_failure_is_an_error_then_one_warning_per_attempt(clock, pg, caplog):
    with caplog.at_level(logging.WARNING, logger="AI.storage_state"):
        for _ in range(3):
            with pytest.raises(StorageUnavailableError):
                storage_state.connect(PARAMS, {})
    records = [r for r in caplog.records if r.name == "AI.storage_state"]
    assert [r.levelno for r in records] == [logging.ERROR]


def test_unknown_state_before_the_first_attempt(clock):
    status = storage_state.status()
    assert status.available is None and status.kind == "unknown" and not storage_state.is_down()


@pytest.mark.parametrize(
    "message, kind",
    [
        ('connection to server at "h", port 5432 failed: timeout expired', "timeout"),
        ("could not connect: Connection timed out (0x0000274C/10060)", "timeout"),
        ("Connection refused (0x0000274D/10061)", "refused"),
        ('could not translate host name "nope" to address: Name or service not known', "dns"),
        ('FATAL:  password authentication failed for user "app"', "auth"),
        ('FATAL:  role "ghost" does not exist', "auth"),
        ('FATAL:  database "lt" does not exist', "database"),
        ('ВАЖНО:  база данных "lt" не существует', "database"),
        ("SSL error: certificate verify failed", "ssl"),
        ("something else entirely", "other"),
    ],
)
def test_classify(message, kind):
    assert storage_state.classify(message) == kind


def test_sanitize_masks_passwords_and_flattens():
    text = storage_state.sanitize("bad dsn host=db password=s3cret user=app\n  second line")
    assert "s3cret" not in text and "password=***" in text and "\n" not in text
    assert len(storage_state.sanitize("x" * 5000)) == storage_state.MESSAGE_LIMIT
