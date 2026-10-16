"""Storage connection pool (AI/storage_state.py): one warm session per thread instead of a connect per call.

psycopg2.connect is replaced by a fake whose sessions track transaction status, autocommit, rollbacks and
who is using them; the breaker, the pool and the call sites above them are real.
"""

import gc
import sys
import threading
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import extensions as ext

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import db_store, storage_state  # noqa: E402
from AI.storage_state import PooledConnection, StorageUnavailableError  # noqa: E402

PARAMS = {"host": "db.test", "port": 5432, "dbname": "lt", "user": "app", "password": "pw", "sslmode": "prefer"}
CFG = {"pool_max": 3, "pool_idle_sec": 300}


class FakeCursor:
    def __init__(self, session):
        self.session = session
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        session = self.session
        text = str(query)
        if session.closed:
            raise psycopg2.InterfaceError("connection already closed")
        if session.status == ext.TRANSACTION_STATUS_INERROR:
            raise psycopg2.InternalError("current transaction is aborted")
        session.executed.append((text, session.autocommit))
        if text == "SELECT 1" and session.dead:
            session.closed = 2
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        if not session.autocommit:
            session.status = ext.TRANSACTION_STATUS_INTRANS
        if "FAIL" in text:
            session.status = ext.TRANSACTION_STATUS_INERROR
            raise psycopg2.DataError("boom")
        self.rows = [(1,)]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class FakeSession:
    """Just enough of a psycopg2 connection for the pool."""

    def __init__(self, number):
        self.number = number
        self.closed = 0
        self.autocommit = False
        self.status = ext.TRANSACTION_STATUS_IDLE
        self.rollbacks = 0
        self.commits = 0
        self.executed = []
        self.dead = False
        self.holder = None
        self.cursor_class = FakeCursor
        self.isolation_level = self.readonly = self.deferrable = None

    def get_transaction_status(self):
        return ext.TRANSACTION_STATUS_UNKNOWN if self.closed == 2 else self.status

    def cursor(self, *args, **kwargs):
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        return self.cursor_class(self)

    def rollback(self):
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        self.rollbacks += 1
        self.status = ext.TRANSACTION_STATUS_IDLE

    def commit(self):
        if self.status == ext.TRANSACTION_STATUS_INERROR:
            raise psycopg2.InternalError("current transaction is aborted")
        self.commits += 1
        self.status = ext.TRANSACTION_STATUS_IDLE

    def close(self):
        if not self.closed:
            self.closed = 1

    def __enter__(self):
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False


class Clock:
    def __init__(self):
        self.now = 5000.0

    def __call__(self):
        return self.now


@pytest.fixture
def pg(monkeypatch):
    """psycopg2.connect replaced: every call is a new FakeSession; fails while ``state['down']`` is set."""
    state = {"down": False, "sessions": [], "lock": threading.Lock()}

    def fake_connect(*args, **kwargs):
        if state["down"]:
            raise psycopg2.OperationalError('connection to server at "db.test", port 5432 failed: Connection refused')
        with state["lock"]:
            session = FakeSession(len(state["sessions"]))
            state["sessions"].append(session)
        return session

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    return state


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(storage_state, "_clock", clock)
    return clock


def _connect(cfg=None, params=None):
    return storage_state.connect(params or PARAMS, CFG if cfg is None else cfg)


# ---- reuse -----------------------------------------------------------------------------------------


def test_two_sequential_connects_open_one_session(pg):
    first = _connect()
    assert isinstance(first, PooledConnection) and first.pooled
    session = first._raw
    first.close()
    second = _connect()
    assert second._raw is session
    second.close()

    assert len(pg["sessions"]) == 1
    assert storage_state.pool_stats() == {"opened": 1, "reused": 1, "in_use": 0, "idle": 1, "overflow": 0, "discarded": 0}


def test_close_rolls_back_and_switches_autocommit_off(pg):
    conn = _connect()
    conn.autocommit = True  # what the _ensure_* helpers do
    session = conn._raw
    assert session.autocommit is True and conn.autocommit is True
    conn.close()
    assert session.autocommit is False and session.rollbacks == 1

    conn = _connect()
    assert conn._raw is session
    with conn.cursor() as cur:
        cur.execute("UPDATE t SET x = 1")  # left uncommitted by a careless caller
    assert session.get_transaction_status() == ext.TRANSACTION_STATUS_INTRANS
    conn.close()
    assert session.rollbacks == 2 and session.commits == 0
    assert session.get_transaction_status() == ext.TRANSACTION_STATUS_IDLE
    assert storage_state.pool_stats()["idle"] == 1


def test_with_conn_commits_and_does_not_close(pg):
    conn = _connect()
    session = conn._raw
    with conn, conn.cursor() as cur:
        cur.execute("INSERT INTO t VALUES (1)")
    assert session.commits == 1 and not session.closed and not conn.closed
    with conn, conn.cursor() as cur:  # the same object is usable again, as with psycopg2
        cur.execute("SELECT 1")
    conn.close()
    again = _connect()
    assert again._raw is session and len(pg["sessions"]) == 1
    again.close()


def test_double_close_is_a_noop_and_a_closed_wrapper_refuses_work(pg):
    conn = _connect()
    conn.close()
    conn.close()
    assert conn.closed == 1
    with pytest.raises(psycopg2.InterfaceError):
        conn.cursor()
    with pytest.raises(psycopg2.InterfaceError):
        conn.autocommit = True
    stats = storage_state.pool_stats()
    assert stats["idle"] == 1 and stats["in_use"] == 0
    # The session went back once: the next two callers do not share it.
    a, b = _connect(), _connect()
    assert a._raw is not b._raw
    a.close()
    b.close()


# ---- what is not reused ----------------------------------------------------------------------------


def test_closed_session_is_not_reused(pg):
    conn = _connect()
    session = conn._raw
    conn.close()
    session.closed = 1  # closed while idle in the pool
    fresh = _connect()
    assert fresh._raw is not session and len(pg["sessions"]) == 2
    fresh.close()
    assert storage_state.pool_stats()["discarded"] == 1


def test_failed_transaction_is_closed_not_pooled(pg):
    conn = _connect()
    session = conn._raw
    with pytest.raises(psycopg2.DataError):
        conn.cursor().execute("SELECT FAIL")
    assert session.get_transaction_status() == ext.TRANSACTION_STATUS_INERROR
    conn.close()  # without a rollback
    assert session.closed
    assert storage_state.pool_stats()["idle"] == 0


def test_exception_leaving_with_block_drops_the_session(pg):
    conn = _connect()
    session = conn._raw
    with pytest.raises(ValueError):
        with conn:
            conn.cursor().execute("SELECT 1")
            raise ValueError("handler failed")
    assert session.rollbacks == 1 and not session.closed  # psycopg2 semantics inside the request
    conn.close()
    assert session.closed  # its session state is not trusted any more
    fresh = _connect()
    assert fresh._raw is not session
    fresh.close()


def test_advisory_lock_left_by_a_failed_migration_does_not_reach_the_pool(pg):
    """backfill_run_ids takes a session-level pg_try_advisory_lock; when an UPDATE fails, its unlock fails
    too (aborted transaction). The session holding the lock must be closed, not handed to the next caller."""
    from loadlens_app import migrations

    sessions = []

    def connect():
        conn = _connect()
        conn._raw.cursor_class = MigrationCursor
        sessions.append(conn._raw)
        return conn

    with pytest.raises(psycopg2.Error):
        migrations.backfill_run_ids(connect=connect)
    session = sessions[0]
    assert any("pg_try_advisory_lock" in q for q, _ in session.executed)
    assert not any("pg_advisory_unlock" in q for q, _ in session.executed)  # the unlock never ran
    assert session.closed, "the session still holding the advisory lock went back to the pool"
    assert storage_state.pool_stats()["idle"] == 0


class MigrationCursor(FakeCursor):
    """The statements of backfill_run_ids: the lock is granted, one run lacks an id, its UPDATE fails."""

    def execute(self, query, params=None):
        session = self.session
        text = query if isinstance(query, str) else repr(query)
        if session.status == ext.TRANSACTION_STATUS_INERROR:
            raise psycopg2.InternalError("current transaction is aborted, commands ignored until end of block")
        session.executed.append((text, session.autocommit))
        session.status = ext.TRANSACTION_STATUS_INTRANS
        if "GROUP BY" in text:
            self.rows = [("run-1", "svc", None)]
        elif "UPDATE" in text:
            session.status = ext.TRANSACTION_STATUS_INERROR
            raise psycopg2.DataError("boom")
        else:
            self.rows = [(True,)]


def test_set_session_characteristics_drop_the_session(pg):
    conn = _connect()
    session = conn._raw
    conn.readonly = True  # what set_session(readonly=True) leaves behind
    conn.close()
    assert session.closed and storage_state.pool_stats()["idle"] == 0


def test_discard_closes_the_session(pg):
    conn = _connect()
    session = conn._raw
    conn.discard()
    conn.close()
    assert session.closed and conn.closed == 1
    stats = storage_state.pool_stats()
    assert stats["idle"] == 0 and stats["in_use"] == 0


def test_broken_session_flushes_the_pool(pg):
    a, b = _connect(), _connect()
    idle = b._raw
    b.close()
    a._raw.closed = 2  # the server went away under a request
    a.close()
    assert idle.closed, "idle sessions of a server that dropped one of them are suspect"
    fresh = _connect()
    assert fresh._raw not in (a._raw, idle) and len(pg["sessions"]) == 3
    fresh.close()


# ---- limits, breaker, idle sessions ------------------------------------------------------------------


def test_exhausted_pool_gives_an_unpooled_connection(pg):
    cfg = {"pool_max": 2}
    a, b = _connect(cfg), _connect(cfg)
    extra = _connect(cfg)
    assert a.pooled and b.pooled and not extra.pooled
    extra_session = extra._raw
    extra.close()
    assert extra_session.closed  # closed for real, not added to the pool
    a.close()
    b.close()
    stats = storage_state.pool_stats()
    assert stats == {"opened": 3, "reused": 0, "in_use": 0, "idle": 2, "overflow": 1, "discarded": 0}


def test_breaker_failure_clears_the_pool(pg, monkeypatch):
    idle, busy = _connect(), _connect()
    idle_session, busy_session = idle._raw, busy._raw
    idle.close()
    pg["down"] = True
    with pytest.raises(StorageUnavailableError):
        _connect(params={**PARAMS, "dbname": "other"})  # a new physical connection fails
    assert storage_state.is_down()
    assert idle_session.closed
    busy.close()  # handed out before the failure: closed on return, not pooled
    assert busy_session.closed
    assert storage_state.pool_stats()["idle"] == 0

    # While the breaker is open nothing is handed out; after recovery a fresh session is opened.
    with pytest.raises(StorageUnavailableError):
        _connect()
    pg["down"] = False
    storage_state.BREAKER.allow_retry_now()
    fresh = _connect()
    assert fresh._raw not in (idle_session, busy_session)
    fresh.close()
    assert not storage_state.is_down()


def test_idle_session_is_pinged_after_half_the_idle_time_and_closed_after_it(pg, clock):
    conn = _connect()
    session = conn._raw
    conn.close()

    clock.now += 100  # < pool_idle_sec / 2: no round trip
    conn = _connect()
    assert conn._raw is session and session.executed == []
    conn.close()

    clock.now += 200  # > pool_idle_sec / 2: one SELECT 1 outside a transaction
    conn = _connect()
    assert conn._raw is session and session.executed == [("SELECT 1", True)]
    assert session.autocommit is False and session.status == ext.TRANSACTION_STATUS_IDLE
    conn.close()

    clock.now += 301  # > pool_idle_sec: closed without a round trip
    conn = _connect()
    assert conn._raw is not session and session.closed and len(session.executed) == 1
    conn.close()


def test_failed_ping_discards_the_session_and_opens_a_new_one(pg, clock):
    a, b = _connect(), _connect()
    a_session, b_session = a._raw, b._raw
    a.close()
    b.close()
    b_session.dead = True  # the warmest one is handed out first
    clock.now += 200
    conn = _connect()
    assert conn._raw not in (a_session, b_session)
    assert a_session.closed and b_session.closed  # the rest of the pool is flushed with it
    conn.close()


def test_sessions_of_old_settings_are_closed_by_the_sweep(pg, clock):
    old = _connect(params={**PARAMS, "host": "db.old"})
    old_session = old._raw
    old.close()
    clock.now += 400
    _connect().close()  # any later call sweeps the other pools
    assert old_session.closed
    assert storage_state.pool_stats()["idle"] == 1


def test_forgotten_close_frees_the_slot(pg):
    cfg = {"pool_max": 1}
    conn = _connect(cfg)
    del conn
    gc.collect()
    assert storage_state.pool_stats()["in_use"] == 0
    again = _connect(cfg)
    assert again.pooled
    again.close()


def test_pool_max_zero_turns_the_pool_off(pg):
    conn = _connect({"pool_max": 0})
    assert isinstance(conn, FakeSession)
    conn.close()
    _connect({"pool_max": 0}).close()
    assert len(pg["sessions"]) == 2 and all(s.closed for s in pg["sessions"])


def test_objects_that_are_not_psycopg2_connections_are_returned_as_is(monkeypatch):
    class Bare:
        closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: Bare())
    conn = _connect()
    assert type(conn) is Bare
    conn.close()
    stats = storage_state.pool_stats()
    assert stats["in_use"] == 0 and stats["idle"] == 0 and stats["opened"] == 1


def test_parameters_select_the_pool(pg):
    _connect().close()
    same = _connect(params={**PARAMS, "port": "5432"})  # the same server spelled differently
    assert same._raw is pg["sessions"][0]
    same.close()
    other = _connect(params={**PARAMS, "dbname": "other"})
    assert other._raw is not pg["sessions"][0]
    other.close()
    dsn = storage_state.connect({"dsn": "postgresql://app@db.test/lt"}, CFG)
    assert dsn._raw not in pg["sessions"][:2]
    dsn.close()
    assert len(pg["sessions"]) == 3


def test_forked_child_neither_uses_nor_closes_parent_sessions(pg, monkeypatch):
    monkeypatch.setattr(storage_state, "_FORK_ORPHANS", [])
    conn = _connect()
    parent_session = conn._raw
    conn.close()
    storage_state._forget_pools_after_fork()
    assert not parent_session.closed and parent_session in storage_state._FORK_ORPHANS
    child = _connect()
    assert child._raw is not parent_session
    child.close()


# ---- concurrency and real call sites -----------------------------------------------------------------


@pytest.mark.parametrize("pool_max", [12, 3])
def test_eight_threads_share_the_pool_without_errors(pg, pool_max):
    cfg = {"pool_max": pool_max}
    errors = []
    start = threading.Barrier(8)

    def worker():
        me = threading.get_ident()
        start.wait()
        for _ in range(50):
            try:
                conn = _connect(cfg)
                try:
                    session = conn._raw
                    if session.holder is not None:
                        errors.append(f"session {session.number} shared between threads")
                    session.holder = me
                    with conn, conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    session.holder = None
                finally:
                    conn.close()
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    stats = storage_state.pool_stats()
    assert stats["in_use"] == 0 and stats["idle"] <= pool_max
    assert stats["opened"] - stats["overflow"] <= pool_max
    if pool_max >= 8:
        assert stats["opened"] <= 8 and stats["overflow"] == 0
    assert stats["opened"] + stats["reused"] == 400


def test_core_and_db_store_connections_share_one_session(pg, monkeypatch):
    cfg = {**PARAMS, "schema": "public", "connect_timeout_sec": 5, "pool_max": 4}
    monkeypatch.setitem(core.CONFIG, "storage", {"timescale": cfg})
    conn = core._ts_conn()
    assert isinstance(conn, PooledConnection)
    session = conn._raw
    conn.close()
    other = db_store._connect(cfg)
    assert other._raw is session
    other.close()
    assert storage_state.pool_stats()["opened"] == 1


def test_settings_example_documents_the_pool():
    text = (ROOT_DIR / "settings.example.py").read_text(encoding="utf-8")
    assert '"pool_max": 12' in text and '"pool_idle_sec": 300' in text
    assert storage_state.DEFAULT_POOL_MAX == 12 and storage_state.DEFAULT_POOL_IDLE_SEC == 300
