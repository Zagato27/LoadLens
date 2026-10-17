"""Start-up migrations: the registry table app_migrations, the skip of applied one-off migrations, the retry
while the storage database is unavailable, the environment switch and the CLI."""

import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from AI.storage_state import StorageUnavailableError
from loadlens_app import migrations
from psycopg2 import sql as pgsql


def _render(query) -> str:
    """Text of a psycopg2 sql object without a connection (identifiers quoted, literals repr'd)."""
    if isinstance(query, str):
        return query
    if isinstance(query, pgsql.Composed):
        return "".join(_render(part) for part in query.seq)
    if isinstance(query, pgsql.SQL):
        return query.string
    if isinstance(query, pgsql.Identifier):
        return ".".join(f'"{part}"' for part in query.strings)
    if isinstance(query, pgsql.Literal):
        return repr(query.wrapped)
    return str(query)


class RegistryCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = " ".join(_render(query).split())
        self.conn.log.append((text, params))
        if "CREATE TABLE" in text and self.conn.create_error is not None:
            raise self.conn.create_error
        if text.startswith("INSERT INTO") and self.conn.insert_error is not None:
            raise self.conn.insert_error
        if text.startswith("INSERT INTO"):
            self.conn.applied.add(params[0])

    def fetchall(self):
        return [(name,) for name in sorted(self.conn.applied)]


class RegistryConnection:
    """Fake registry connection: remembers the recorded migrations and every statement."""

    def __init__(self, applied=(), create_error=None, insert_error=None):
        self.applied = set(applied)
        self.create_error = create_error
        self.insert_error = insert_error
        self.log: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.autocommit = False

    def cursor(self):
        return RegistryCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True

    def statements(self, fragment):
        return [entry for entry in self.log if fragment in entry[0]]


@pytest.fixture
def fake_migrations(monkeypatch):
    """Every migration function is replaced by a counter; the registry fake records what ran."""
    monkeypatch.setattr(migrations, "_storage_names", lambda: ("public", "llm_reports"))
    calls = []
    for migration in migrations.MIGRATIONS:
        monkeypatch.setattr(migrations, migration.func, lambda key=migration.key: calls.append(key) or 1)
    return calls


def test_registry_table_is_created_and_every_migration_runs_on_a_fresh_database(fake_migrations):
    registry = RegistryConnection()
    result = migrations.run_all(connect=lambda: registry)

    assert result == {m.key: 1 for m in migrations.MIGRATIONS}
    assert fake_migrations == [m.key for m in migrations.MIGRATIONS]
    create = registry.statements("CREATE TABLE IF NOT EXISTS")[0][0]
    assert '"public"."app_migrations"' in create and "name TEXT PRIMARY KEY" in create and "runs INTEGER" in create
    recorded = [params[0] for text, params in registry.statements("INSERT INTO")]
    assert recorded == [m.key for m in migrations.MIGRATIONS]  # each one right after it ran
    assert "ON CONFLICT (name) DO UPDATE" in registry.statements("INSERT INTO")[0][0]
    assert registry.closed and registry.autocommit is False  # the registry session is given back clean


def test_applied_one_off_migrations_are_skipped_and_catch_ups_still_run(fake_migrations):
    once = [m.key for m in migrations.MIGRATIONS if m.once]
    repeat = [m.key for m in migrations.MIGRATIONS if not m.once]
    registry = RegistryConnection(applied=once)

    result = migrations.run_all(connect=lambda: registry)

    assert fake_migrations == repeat
    assert all(result[key] is None for key in once) and all(result[key] == 1 for key in repeat)
    assert [params[0] for _text, params in registry.statements("INSERT INTO")] == repeat


def test_force_runs_the_recorded_one_off_migrations_again(fake_migrations):
    registry = RegistryConnection(applied=[m.key for m in migrations.MIGRATIONS])
    result = migrations.run_all(force=True, connect=lambda: registry)
    assert fake_migrations == [m.key for m in migrations.MIGRATIONS] and None not in result.values()


def test_without_rights_for_the_registry_everything_runs_and_nothing_is_recorded(fake_migrations, caplog):
    registry = RegistryConnection(create_error=RuntimeError("permission denied for schema public"))
    with caplog.at_level("WARNING"):
        result = migrations.run_all(connect=lambda: registry)
    assert fake_migrations == [m.key for m in migrations.MIGRATIONS] and None not in result.values()
    assert registry.statements("INSERT INTO") == []
    assert "every migration runs at every start" in caplog.text


def test_a_failed_record_does_not_fail_the_run(fake_migrations, caplog):
    registry = RegistryConnection(insert_error=RuntimeError("read-only transaction"))
    with caplog.at_level("WARNING"):
        result = migrations.run_all(connect=lambda: registry)
    assert None not in result.values() and registry.rollbacks == len(migrations.MIGRATIONS)
    assert "could not be recorded" in caplog.text


def test_a_failing_migration_keeps_the_earlier_ones_recorded(fake_migrations, monkeypatch):
    second = migrations.MIGRATIONS[1]
    monkeypatch.setattr(migrations, second.func, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    registry = RegistryConnection()
    with pytest.raises(RuntimeError):
        migrations.run_all(connect=lambda: registry)
    assert [params[0] for _text, params in registry.statements("INSERT INTO")] == [migrations.MIGRATIONS[0].key]
    assert registry.closed


def test_pending_migrations_lists_the_one_off_ones_not_yet_recorded(monkeypatch):
    monkeypatch.setattr(migrations, "_storage_names", lambda: ("public", "llm_reports"))
    once = [m.key for m in migrations.MIGRATIONS if m.once]
    assert migrations.pending_migrations(connect=lambda: RegistryConnection()) == once
    assert migrations.pending_migrations(connect=lambda: RegistryConnection(applied=once[:1])) == once[1:]
    # No registry at all: everything one-off is pending, so the start-up run happens.
    denied = RegistryConnection(create_error=RuntimeError("permission denied"))
    assert migrations.pending_migrations(connect=lambda: denied) == once


@pytest.mark.parametrize("value, expected", [
    (None, True), ("", True), ("1", True), ("true", True), ("yes", True),
    ("0", False), ("false", False), ("No", False), (" off ", False),
])
def test_enabled_by_env(value, expected):
    env = {} if value is None else {"LOADLENS_RUN_MIGRATIONS": value}
    assert migrations.enabled_by_env(env) is expected


def test_retries_while_the_storage_database_is_unavailable(caplog):
    attempts = []
    delays = []

    def run():
        attempts.append(1)
        if len(attempts) < 3:
            raise StorageUnavailableError("connection refused")
        return {"backfill_run_ids": 0, "backfill_run_index": 4, "backfill_max_rps": None, "metrics_run_name_index": 0}

    with caplog.at_level("INFO"):
        result = migrations.run_with_retries(run=run, sleep=delays.append)

    assert result is not None and len(attempts) == 3
    assert delays == [migrations.RETRY_DELAYS_SEC[0], migrations.RETRY_DELAYS_SEC[1]]
    assert "Migrations postponed" in caplog.text
    # A one-off migration ran: said at WARNING, the level that reaches the log without logging configuration.
    applied = [r for r in caplog.records if r.getMessage().startswith("Migrations applied")]
    assert applied and applied[0].levelname == "WARNING"
    assert applied[0].getMessage() == (
        "Migrations applied: backfill_run_ids=0; catch-ups: backfill_run_index=4, metrics_run_name_index=0; already applied: backfill_max_rps"
    )


def test_nothing_new_is_reported_quietly(caplog):
    def run():
        return {"backfill_run_ids": None, "backfill_run_index": 2, "backfill_max_rps": None, "metrics_run_name_index": 0}

    with caplog.at_level("INFO"):
        migrations.run_with_retries(run=run, sleep=lambda s: None)
    [record] = [r for r in caplog.records if r.name == migrations.logger.name]
    assert record.levelname == "INFO"
    assert record.getMessage() == "Migrations checked: nothing new to apply; catch-ups: backfill_run_index=2, metrics_run_name_index=0; already applied: backfill_run_ids, backfill_max_rps"


def test_the_last_retry_delay_repeats_and_attempts_can_be_capped():
    delays = []

    def down():
        raise StorageUnavailableError("down")

    assert migrations.run_with_retries(run=down, sleep=delays.append, max_attempts=8) is None
    assert delays == list(migrations.RETRY_DELAYS_SEC) + [migrations.RETRY_DELAYS_SEC[-1]] * (7 - len(migrations.RETRY_DELAYS_SEC))


def test_a_programming_error_is_logged_once_without_retries(caplog):
    delays = []

    def broken():
        raise KeyError("no such column")

    with caplog.at_level("ERROR"):
        assert migrations.run_with_retries(run=broken, sleep=delays.append) is None
    assert delays == [] and "run `python -m loadlens_app.migrations` manually" in caplog.text


def test_background_start_happens_once_per_process(monkeypatch):
    started = []
    monkeypatch.setattr(migrations, "_started", False)
    monkeypatch.setattr(migrations.threading, "Thread", lambda target, name, daemon: types.SimpleNamespace(start=lambda: started.append((target, name, daemon))))
    assert migrations.start_background_migrations() is True
    assert migrations.start_background_migrations() is False
    assert started == [(migrations.run_with_retries, "loadlens-migrations", True)]


def test_cli_status_lists_applied_and_pending_without_running(monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(migrations, "run_all", lambda **kwargs: ran.append(kwargs))
    monkeypatch.setattr(migrations, "pending_migrations", lambda: ["backfill_max_rps"])
    assert migrations.main(["--status"]) == 2
    out = capsys.readouterr().out
    assert "backfill_run_ids: выполнена" in out and "backfill_max_rps: ожидает" in out
    assert "backfill_run_index: выполняется при каждом старте" in out and ran == []
    monkeypatch.setattr(migrations, "pending_migrations", lambda: [])
    assert migrations.main(["--status"]) == 0


def test_cli_run_reports_skipped_migrations_and_forwards_force(monkeypatch, capsys):
    seen = {}

    def run_all(force=False):
        seen["force"] = force
        return {"backfill_run_ids": None, "backfill_run_index": 3}

    monkeypatch.setattr(migrations, "run_all", run_all)
    assert migrations.main(["--force"]) == 0
    out = capsys.readouterr().out
    assert "backfill_run_ids: пропущена (уже выполнена)" in out and "backfill_run_index: 3" in out and seen["force"] is True


def test_cli_reports_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(migrations, "run_all", lambda force=False: (_ for _ in ()).throw(StorageUnavailableError("down")))
    assert migrations.main([]) == 1
    assert "Ошибка миграции: down" in capsys.readouterr().err
