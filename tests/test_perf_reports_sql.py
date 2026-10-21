# SPDX-License-Identifier: Apache-2.0
"""The ``max_rps`` column of ``llm_reports``: one parsing rule for Python and SQL, the write in
``save_llm_results``, the backfill migration, the read expressions of the dashboard and ``/runs`` and the
DDL (column plus partial indexes of the final rows). Offline: fake connections and cursors only."""

import sys
import types
from decimal import Decimal
from pathlib import Path

import pytest
from psycopg2 import sql

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402,F401  (before anything that imports update_page)
from AI import db_store  # noqa: E402
from loadlens_app import migrations  # noqa: E402
from loadlens_app.blueprints import dashboard  # noqa: E402


def render(query) -> str:
    """Text of a psycopg2 ``sql`` composition without a live connection (identifiers double-quoted)."""
    if isinstance(query, str):
        return query
    if isinstance(query, sql.Composed):
        return "".join(render(part) for part in query.seq)
    if isinstance(query, sql.Identifier):
        return ".".join('"' + name.replace('"', '""') + '"' for name in query.strings)
    if isinstance(query, sql.SQL):
        return query.string
    raise TypeError(f"unexpected query part {query!r}")


def flat(text: str) -> str:
    return " ".join(text.split())


# ---- max_rps_from_parsed: priority of the fields and the number rule ------------------------------------


def test_fields_keep_the_dashboard_priority():
    assert db_store.MAX_RPS_FIELDS == (
        ("peak_performance", "max_rps"),
        ("stability_under_load", "actual_rps"),
        ("spike_analysis", "max_spike_rps"),
    )
    assert dashboard.RUNS_MAX_RPS_FIELDS is db_store.MAX_RPS_FIELDS


def test_first_plain_number_wins_in_field_order():
    parsed = {
        "peak_performance": {"max_rps": 300},
        "stability_under_load": {"actual_rps": 210},
        "spike_analysis": {"max_spike_rps": 900},
    }
    assert db_store.max_rps_from_parsed(parsed) == 300.0
    del parsed["peak_performance"]
    assert db_store.max_rps_from_parsed(parsed) == 210.0
    del parsed["stability_under_load"]
    assert db_store.max_rps_from_parsed(parsed) == 900.0
    del parsed["spike_analysis"]
    assert db_store.max_rps_from_parsed(parsed) is None


def test_text_that_is_not_a_plain_number_falls_through_to_the_next_field():
    parsed = {
        "peak_performance": {"max_rps": "~1200", "not_applicable": False},
        "stability_under_load": {"actual_rps": "1180.5"},
    }
    assert db_store.max_rps_from_parsed(parsed) == 1180.5
    assert db_store.max_rps_from_parsed({"peak_performance": {"max_rps": "~1200"}}) is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (1200, 1200.0),
        (1200.75, 1200.75),
        (-3, -3.0),
        ("1200", 1200.0),
        (" 1200 ", 1200.0),  # TRIM removes spaces
        ("+5", 5.0),
        ("-3.5", -3.5),
        ("1.2e3", 1200.0),
        ("1.2E+3", 1200.0),
        ("12e-1", 1.2),
        ("1e999", None),  # matches the pattern, beyond float: None (JSON has no Infinity), no fall-through
        ("1e400", None),
        (1e20, 1e20),
        (10**400, None),
    ],
)
def test_plain_numbers(value, expected):
    parsed = {"peak_performance": {"max_rps": value}, "stability_under_load": {"actual_rps": 7}}
    result = db_store.max_rps_from_parsed(parsed)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    ["~1200", "≈90 rps", "1,200", "1 200", "1200.", ".5", "1e4000", "1e", "0x10", "nan", "Infinity", "",
     "\t1200", "1200\n", "١٢٠٠", True, False, None, [1200], {"value": 1200}, float("nan"), float("inf")],
)
def test_values_that_are_not_plain_numbers_are_skipped(value):
    parsed = {"peak_performance": {"max_rps": value}, "stability_under_load": {"actual_rps": "42"}}
    assert db_store.max_rps_from_parsed(parsed) == 42.0


@pytest.mark.parametrize("parsed", [None, "", "{}", [], 12, {}, {"peak_performance": "1200"}, {"peak_performance": None}])
def test_parsed_without_a_level_gives_none(parsed):
    assert db_store.max_rps_from_parsed(parsed) is None


# ---- SQL twin and the read expressions ---------------------------------------------------------------------


def test_sql_rule_uses_the_same_fields_and_pattern():
    rule = db_store.max_rps_sql("parsed")
    assert rule.startswith("COALESCE(CASE WHEN TRIM(parsed->'peak_performance'->>'max_rps') ~ ")
    assert rule.index("'peak_performance'") < rule.index("'stability_under_load'") < rule.index("'spike_analysis'")
    assert rule.count(f"'{db_store.MAX_RPS_NUMBER_PATTERN}'") == 3
    assert rule.count("::numeric END") == 3
    assert "%" not in rule  # safe inside statements with psycopg2 parameters


def test_runs_and_dashboard_read_the_column_first():
    runs_sql = dashboard._runs_max_rps_sql("parsed")
    assert runs_sql.startswith("COALESCE(max_rps, ")
    assert runs_sql == f"COALESCE(max_rps, {db_store.max_rps_sql('parsed')})"
    assert dashboard.DASHBOARD_MAX_RPS_SQL == runs_sql
    # The column follows the alias of the parsed column.
    assert dashboard._runs_max_rps_sql("f.parsed").startswith("COALESCE(f.max_rps, COALESCE(CASE WHEN TRIM(f.parsed->")


def test_numeric_values_from_the_column_reach_the_api_as_floats():
    assert dashboard._dashboard_number(Decimal("240.5")) == pytest.approx(240.5)
    assert dashboard._dashboard_number(Decimal("1.2E+3")) == pytest.approx(1200.0)
    assert dashboard._dashboard_number(Decimal("1E+400")) is None
    assert dashboard._runs_float(Decimal("1E+400")) is None


# ---- DDL ----------------------------------------------------------------------------------------------------


class DdlCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = flat(render(query))
        self.conn.statements.append(text)
        if any(fragment in text for fragment in self.conn.fail_on):
            raise RuntimeError("boom")


class DdlConnection:
    def __init__(self, fail_on=()):
        self.statements: list[str] = []
        self.fail_on = tuple(fail_on)
        self.autocommit = False

    def cursor(self):
        return DdlCursor(self)


@pytest.fixture
def fresh_table():
    cfg = {"schema": "public", "llm_table": "llm_reports_perf"}
    db_store._ENSURED_TABLES.discard(("public", "llm_reports_perf"))
    yield cfg
    db_store._ENSURED_TABLES.discard(("public", "llm_reports_perf"))


def test_ensure_adds_the_column_and_both_partial_indexes(fresh_table):
    conn = DdlConnection()
    db_store._ensure_llm_reports_table(conn, fresh_table)
    statements = conn.statements
    assert 'ALTER TABLE "public"."llm_reports_perf" ADD COLUMN IF NOT EXISTS max_rps NUMERIC;' in statements
    assert (
        'CREATE INDEX IF NOT EXISTS "idx_llm_reports_perf_final_created" ON "public"."llm_reports_perf" '
        "(created_at DESC) WHERE domain = 'final';"
    ) in statements
    assert (
        'CREATE INDEX IF NOT EXISTS "idx_llm_reports_perf_final_service_created" ON "public"."llm_reports_perf" '
        "(service, created_at DESC) WHERE domain = 'final';"
    ) in statements
    assert conn.autocommit is False  # restored after the DDL


def test_ensure_runs_every_ddl_statement_even_when_one_fails(fresh_table):
    conn = DdlConnection(fail_on=("max_rps NUMERIC",))
    db_store._ensure_llm_reports_table(conn, fresh_table)
    assert any("_final_created" in s for s in conn.statements)
    assert any("_final_service_created" in s for s in conn.statements)


def test_deployment_schema_has_the_column_and_indexes():
    schema_sql = flat((ROOT_DIR / "docker" / "initdb" / "02_schema.sql").read_text(encoding="utf-8"))
    block = schema_sql[schema_sql.index("CREATE TABLE IF NOT EXISTS public.llm_reports ("):schema_sql.index(
        "CREATE TABLE IF NOT EXISTS public.llm_feedback")]
    assert "max_rps NUMERIC );" in block
    assert "ALTER TABLE public.llm_reports ADD COLUMN IF NOT EXISTS max_rps NUMERIC;" in block
    assert (
        "CREATE INDEX IF NOT EXISTS idx_llm_reports_final_created ON public.llm_reports (created_at DESC) "
        "WHERE domain = 'final';"
    ) in block
    assert (
        "CREATE INDEX IF NOT EXISTS idx_llm_reports_final_service_created ON public.llm_reports "
        "(service, created_at DESC) WHERE domain = 'final';"
    ) in block


# ---- save_llm_results -----------------------------------------------------------------------------------------


class SaveConnection:
    def __init__(self):
        self.committed = False

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        pass


class SqlStub:
    def __init__(self, text):
        self.text = text

    def format(self, *_args, **_kwargs):
        return self

    def as_string(self, _context):
        return self.text


def test_save_llm_results_writes_max_rps_of_the_final_row_only(monkeypatch):
    captured = {}

    def fake_batch(cur, sql_text, rows, page_size=100):
        captured["sql"] = sql_text
        captured["rows"] = rows

    conn = SaveConnection()
    monkeypatch.setattr(db_store, "execute_batch", fake_batch)
    monkeypatch.setattr(db_store, "_connect", lambda _cfg: conn)
    monkeypatch.setattr(db_store, "_ensure_llm_reports_table", lambda *_a, **_k: None)
    monkeypatch.setattr(db_store.sql, "SQL", SqlStub)

    db_store.save_llm_results(
        {
            "jvm": "jvm text",
            "jvm_parsed": {"verdict": "Успешно", "peak_performance": {"max_rps": 999}},
            "final": '{"verdict":"Успешно"}',
            "final_parsed": {
                "verdict": "Успешно",
                "peak_performance": {"max_rps": "~1200"},
                "stability_under_load": {"actual_rps": "1.2e3"},
            },
        },
        {"run_name": "nightly-1", "service": "checkout"},
        {"schema": "public", "llm_table": "llm_reports"},
    )
    assert conn.committed
    text = flat(captured["sql"])
    columns = text[text.index("(") + 1:text.index(")")].split(",")
    assert [c.strip() for c in columns][-1] == "max_rps"
    placeholders = text[text.index("VALUES (") + len("VALUES ("):].split(")")[0].split(",")
    rows = captured["rows"]
    assert all(len(row) == len(columns) == len(placeholders) for row in rows)
    by_domain = {row[6]: row for row in rows}
    assert by_domain["final"][-1] == pytest.approx(1200.0)
    assert by_domain["jvm"][-1] is None  # only final rows carry the load level
    assert all(row[-1] is None for domain, row in by_domain.items() if domain != "final")


def test_save_llm_results_writes_null_when_the_final_analysis_has_no_number(monkeypatch):
    captured = {}
    monkeypatch.setattr(db_store, "execute_batch", lambda cur, s, rows, page_size=100: captured.setdefault("rows", rows))
    monkeypatch.setattr(db_store, "_connect", lambda _cfg: SaveConnection())
    monkeypatch.setattr(db_store, "_ensure_llm_reports_table", lambda *_a, **_k: None)
    monkeypatch.setattr(db_store.sql, "SQL", SqlStub)

    db_store.save_llm_results(
        {"final": "text", "final_parsed": {"verdict": "Провал", "peak_performance": {"max_rps": "~1200"}}},
        {"run_name": "nightly-2"},
        {"schema": "public", "llm_table": "llm_reports"},
    )
    final = [row for row in captured["rows"] if row[6] == "final"][0]
    assert final[-1] is None


# ---- backfill_max_rps ------------------------------------------------------------------------------------------


class MigrationCursor:
    def __init__(self, conn):
        self.conn = conn
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = flat(render(query))
        self.conn.log.append(("execute", text, params))
        if "pg_try_advisory_lock" in text:
            self.row = (self.conn.lock_free,)
        elif "pg_advisory_unlock" in text:
            self.row = (True,)
        elif "UPDATE" in text:
            outcome = self.conn.batches.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.row = outcome
        else:
            self.row = None

    def fetchone(self):
        return self.row


class MigrationConnection:
    def __init__(self, batches, lock_free=True):
        self.batches = list(batches)
        self.lock_free = lock_free
        self.log: list[tuple] = []
        self.closed = False

    def cursor(self):
        return MigrationCursor(self)

    def commit(self):
        self.log.append(("commit", "", None))

    def rollback(self):
        self.log.append(("rollback", "", None))

    def close(self):
        self.closed = True

    def executed(self, fragment):
        return [entry for entry in self.log if entry[0] == "execute" and fragment in entry[1]]


@pytest.fixture
def backfill_env(monkeypatch):
    ensured = []
    monkeypatch.setattr(migrations, "_storage_names", lambda: ("public", "llm_reports"))
    monkeypatch.setattr(db_store, "_ensure_llm_reports_table", lambda conn, cfg: ensured.append(cfg))
    return ensured


def test_backfill_walks_the_final_rows_in_batches(backfill_env):
    conn = MigrationConnection([(5000, 4800), (7400, 2000), (None, 0)])
    filled = migrations.backfill_max_rps(connect=lambda: conn)

    assert filled == 6800
    assert backfill_env == [{"schema": "public", "llm_table": "llm_reports"}]  # column ensured first
    batches = conn.executed("UPDATE")
    assert [params for _, _, params in batches] == [(0, 5000), (5000, 5000), (7400, 5000)]
    statement = batches[0][1]
    rule = db_store.max_rps_sql("parsed")
    assert (
        'WITH batch AS ( SELECT id FROM "public"."llm_reports" WHERE domain = \'final\' AND max_rps IS NULL'
        " AND id > %s ORDER BY id LIMIT %s), filled AS ("
    ) in statement
    assert f'UPDATE "public"."llm_reports" AS r SET max_rps = {rule} FROM batch' in statement
    assert f"WHERE r.id = batch.id AND {rule} IS NOT NULL RETURNING r.id)" in statement
    assert statement.endswith("SELECT (SELECT MAX(id) FROM batch), (SELECT COUNT(*) FROM filled)")
    assert "{1,3}" in statement  # the pattern survives sql.SQL.format
    # Every batch is committed on its own; the lock is taken first and released last.
    assert conn.log[0][0] == "execute" and "pg_try_advisory_lock" in conn.log[0][1]
    update_positions = [i for i, entry in enumerate(conn.log) if entry[0] == "execute" and "UPDATE" in entry[1]]
    for position in update_positions:
        assert conn.log[position + 1][0] == "commit"
    assert "pg_advisory_unlock" in conn.log[-2][1] and conn.log[-1][0] == "commit"
    assert conn.executed("pg_advisory_unlock")[0][2] == (migrations.MAX_RPS_BACKFILL_LOCK,)
    assert conn.closed


def test_backfill_batch_size_is_a_parameter(backfill_env):
    conn = MigrationConnection([(10, 3), (None, 0)])
    assert migrations.backfill_max_rps(connect=lambda: conn, batch_size=10) == 3
    assert [params for _, _, params in conn.executed("UPDATE")] == [(0, 10), (10, 10)]


def test_backfill_skips_when_another_process_holds_the_lock(backfill_env):
    conn = MigrationConnection([], lock_free=False)
    assert migrations.backfill_max_rps(connect=lambda: conn) == 0
    assert conn.executed("UPDATE") == []
    assert conn.executed("pg_advisory_unlock") == []
    assert conn.closed


def test_backfill_failure_rolls_back_and_releases_the_lock(backfill_env):
    conn = MigrationConnection([(5000, 10), RuntimeError("statement timeout")])
    with pytest.raises(RuntimeError):
        migrations.backfill_max_rps(connect=lambda: conn)
    unlock = [i for i, entry in enumerate(conn.log) if entry[0] == "execute" and "pg_advisory_unlock" in entry[1]]
    assert unlock and conn.log[unlock[0] - 1][0] == "rollback"
    assert conn.closed


def test_backfill_lock_differs_from_the_other_migrations():
    locks = [name for name in dir(migrations) if name.endswith("_LOCK") and isinstance(getattr(migrations, name), int)]
    values = [getattr(migrations, name) for name in locks]
    assert "MAX_RPS_BACKFILL_LOCK" in locks and len(values) == len(set(values))
    assert migrations.MAX_RPS_BACKFILL_BATCH == 5000


def test_run_all_includes_the_max_rps_backfill(monkeypatch):
    # Every migration is replaced: a real one would connect to the configured storage database.
    real = [name for name in dir(migrations) if name.startswith(("backfill_", "ensure_")) and callable(getattr(migrations, name))]
    for name in real:
        monkeypatch.setattr(migrations, name, lambda: 0)
    monkeypatch.setattr(migrations, "backfill_run_ids", lambda: 2)
    monkeypatch.setattr(migrations, "backfill_max_rps", lambda: 7)
    # The registry has no rows yet: the fake answers the SELECT with nothing and accepts the INSERTs.
    class RegistryCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query, params=None):
            pass

        def fetchall(self):
            return []

    registry = types.SimpleNamespace(cursor=RegistryCursor, commit=lambda: None, rollback=lambda: None, close=lambda: None, autocommit=False)
    result = migrations.run_all(connect=lambda: registry)
    assert result["backfill_run_ids"] == 2
    assert result["backfill_max_rps"] == 7
