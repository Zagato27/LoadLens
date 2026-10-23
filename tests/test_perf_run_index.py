# SPDX-License-Identifier: Apache-2.0
"""run_index (plans/perf-plan.md, agent 5): the table of runs that /runs and the dashboard counter read instead of
aggregating the metrics hypertable; its upsert on save, the start-up backfill, the metrics run_name index
migration and the delete/rename paths. Offline: every connection is a fake that records the statements."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from psycopg2 import sql

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import db_store  # noqa: E402

START = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
END = datetime(2026, 10, 1, 11, 0, tzinfo=timezone.utc)
CFG = {"schema": "public", "table": "metrics", "llm_table": "llm_reports"}


def render(query) -> str:
    """SQL text of a psycopg2 ``sql`` composition (or a plain string) without a live connection."""
    if isinstance(query, sql.Composed):
        return "".join(render(part) for part in query.seq)
    if isinstance(query, sql.SQL):
        return query.string
    if isinstance(query, sql.Identifier):
        return ".".join('"' + part.replace('"', '""') + '"' for part in query.strings)
    if isinstance(query, sql.Placeholder):
        return "%s"
    return str(query)


def flat(query) -> str:
    return " ".join(render(query).split())


class Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = flat(query)
        self.conn.log.append((text, params, self.conn.autocommit))
        for fragment, result in self.conn.script:
            if fragment in text:
                if isinstance(result, Exception):
                    raise result
                self.rows = list(result)
                self.rowcount = len(self.rows) if fragment.startswith("INSERT") else self.conn.rowcount
                return
        self.rows = []
        self.rowcount = self.conn.rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class Conn:
    """Answers each statement by the first scripted fragment its text contains."""

    def __init__(self, *script, rowcount=0):
        self.script = list(script)
        self.rowcount = rowcount
        self.log = []
        self.autocommit = False
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.commits += 1
        else:
            self.rollbacks += 1
        return False

    def cursor(self):
        return Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True

    def texts(self):
        return [text for text, _params, _autocommit in self.log]


# ---- table, upsert, refresh ---------------------------------------------------------------------------------


def test_ensure_run_index_table_creates_the_table_and_its_index_once(monkeypatch):
    monkeypatch.setattr(db_store, "_ENSURED_TABLES", set())
    conn = Conn()

    db_store._ensure_run_index_table(conn, {"schema": "lt"})
    db_store._ensure_run_index_table(conn, {"schema": "lt"})

    create_table, create_index = conn.texts()
    assert create_table.startswith('CREATE TABLE IF NOT EXISTS "lt"."run_index"')
    for column in ("run_name TEXT PRIMARY KEY", "service TEXT NOT NULL DEFAULT ''", "run_id TEXT NOT NULL DEFAULT ''",
                   "start_time TIMESTAMPTZ NOT NULL", "end_time TIMESTAMPTZ NOT NULL",
                   "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"):
        assert column in create_table
    assert create_index == 'CREATE INDEX IF NOT EXISTS "idx_run_index_service_end" ON "lt"."run_index" (service, end_time DESC);'
    assert all(autocommit for _text, _params, autocommit in conn.log)  # DDL outside the caller's transaction
    assert conn.autocommit is False


def test_upsert_widens_the_span_and_keeps_known_service_and_id():
    conn = Conn()

    assert db_store.upsert_run_index(conn, CFG, "nightly-1", "checkout", "rid-1", START, END) is True

    (text, params, _autocommit), = conn.log
    assert text.startswith('INSERT INTO "public"."run_index" AS ri (run_name, service, run_id, start_time, end_time) VALUES')
    assert "ON CONFLICT (run_name) DO UPDATE SET" in text
    assert "start_time = LEAST(ri.start_time, EXCLUDED.start_time)" in text
    assert "end_time = GREATEST(ri.end_time, EXCLUDED.end_time)" in text
    assert "service = COALESCE(NULLIF(EXCLUDED.service, ''), ri.service)" in text
    assert "run_id = COALESCE(NULLIF(EXCLUDED.run_id, ''), ri.run_id)" in text
    assert "updated_at = now()" in text
    assert params == ("nightly-1", "checkout", "rid-1", START, END)


def test_upsert_skips_a_run_without_a_name_and_stores_missing_fields_as_empty():
    conn = Conn()
    assert db_store.upsert_run_index(conn, CFG, "", "checkout", "rid-1", START, END) is False
    assert db_store.upsert_run_index(conn, CFG, None, "checkout", "rid-1", START, END) is False
    assert conn.log == []

    db_store.upsert_run_index(conn, CFG, "r", None, None, START, END)
    assert conn.log[0][1] == ("r", "", "", START, END)


def test_refresh_builds_a_full_and_an_incremental_pass():
    full = Conn(("INSERT INTO", [(1,)] * 3))
    assert db_store.refresh_run_index(full, CFG) == 3
    (text, params, _), = full.log
    assert text.startswith('INSERT INTO "public"."run_index" AS ri (run_name, service, run_id, start_time, end_time) SELECT')
    assert "COALESCE(MAX(service), ''), COALESCE(MAX(NULLIF(run_id, '')), ''), MIN(\"time\"), MAX(\"time\")" in text
    assert 'FROM "public"."metrics" WHERE run_name IS NOT NULL AND run_name <> \'\' GROUP BY run_name' in text
    assert "LEAST(ri.start_time, EXCLUDED.start_time)" in text and "GREATEST(ri.end_time, EXCLUDED.end_time)" in text
    assert params == ()

    since = END - timedelta(days=1)
    incremental = Conn()
    db_store.refresh_run_index(incremental, {"schema": "lt", "table": "m"}, since=since)
    (text, params, _), = incremental.log
    assert 'FROM "lt"."m" WHERE run_name IS NOT NULL AND run_name <> \'\' AND "time" > %s GROUP BY run_name' in text
    assert text.startswith('INSERT INTO "lt"."run_index"')
    assert params == (since,)


# ---- save_domain_labeled ------------------------------------------------------------------------------------


@pytest.fixture
def saving(monkeypatch):
    """save_domain_labeled against a fake connection; returns (connection, recorded upserts, inserted rows)."""
    conn = Conn()
    upserts = []
    inserted = []
    monkeypatch.setattr(db_store, "_connect", lambda cfg: conn)
    monkeypatch.setattr(db_store, "_ensure_schema_and_table", lambda *args: None)
    monkeypatch.setattr(db_store, "_ensure_run_index_table", lambda *args: None)
    monkeypatch.setattr(db_store.sql.Composed, "as_string", lambda self, context: "INSERT INTO metrics VALUES")
    monkeypatch.setattr(db_store, "execute_batch", lambda cur, text, rows, page_size=500: inserted.extend(rows))
    monkeypatch.setattr(db_store, "upsert_run_index", lambda *args: upserts.append(args) or True)
    return conn, upserts, inserted


def _frames():
    first = pd.DataFrame(
        {"pod-a": [1.0, 2.0, 3.0]},
        index=pd.date_range("2026-10-01T10:00:00Z", periods=3, freq="1min"),
    )
    # The first point has no value: it is not stored, so it does not count for the span either.
    second = pd.DataFrame(
        {"pod-a": [None, 6.0, 7.0], "pod-b": [None, None, 8.0]},
        index=pd.date_range("2026-10-01T09:58:00Z", periods=3, freq="1min"),
    )
    empty = pd.DataFrame({"pod-a": []}, index=pd.DatetimeIndex([], tz="UTC"))
    return [{"df": first, "label": "rps"}, {"df": second, "label": "errors"}, {"df": empty, "label": "none"}]


def test_save_indexes_the_run_with_the_span_of_the_stored_rows(saving):
    conn, upserts, inserted = saving
    run_meta = {"run_name": "nightly-1", "service": "checkout", "run_id": "rid-1"}

    db_store.save_domain_labeled("lt", {"labels": ["rps", "errors", "none"]}, _frames(), run_meta, CFG)

    assert len(inserted) == 6
    (call,) = upserts
    _conn, cfg, run_name, service, run_id, start_time, end_time = call
    assert (cfg, run_name, service, run_id) == (CFG, "nightly-1", "checkout", "rid-1")
    assert start_time == datetime(2026, 10, 1, 9, 59, tzinfo=timezone.utc)
    assert end_time == datetime(2026, 10, 1, 10, 2, tzinfo=timezone.utc)
    assert type(start_time) is datetime and type(end_time) is datetime
    assert min(row[0] for row in inserted) == start_time and max(row[0] for row in inserted) == end_time
    # Same transaction as the inserts, inside a savepoint, then one commit.
    assert conn.texts() == ["SAVEPOINT run_index_upsert", "RELEASE SAVEPOINT run_index_upsert"]
    assert conn.commits == 1 and conn.rollbacks == 0 and conn.closed


def test_save_without_a_run_name_or_rows_does_not_touch_run_index(saving):
    conn, upserts, inserted = saving

    db_store.save_domain_labeled("lt", {}, _frames(), {"run_name": "", "service": "checkout"}, CFG)
    db_store.save_domain_labeled("lt", {}, _frames(), None, CFG)
    db_store.save_domain_labeled("lt", {}, [_frames()[2]], {"run_name": "nightly-1"}, CFG)

    assert upserts == [] and conn.log == []
    assert len(inserted) == 12  # the metrics themselves are still saved


def test_a_failed_index_update_does_not_lose_the_metrics(saving, monkeypatch, caplog):
    conn, _upserts, inserted = saving

    def broken(*args):
        raise RuntimeError("run_index is broken")

    monkeypatch.setattr(db_store, "upsert_run_index", broken)
    db_store.save_domain_labeled("lt", {}, _frames(), {"run_name": "nightly-1"}, CFG)

    assert conn.texts() == ["SAVEPOINT run_index_upsert", "ROLLBACK TO SAVEPOINT run_index_upsert"]
    assert conn.commits == 1 and conn.rollbacks == 0 and len(inserted) == 6
    assert "run_index" in caplog.text and "nightly-1" in caplog.text


# ---- delete, rename, service deletion -------------------------------------------------------------------------


def test_run_tables_include_run_index():
    schema, tables = core._run_tables({"schema": "lt"})
    assert schema == "lt"
    assert tables[0] == "metrics" and "run_index" in tables


def test_deleting_a_run_deletes_its_run_index_row(monkeypatch):
    conn = Conn()
    monkeypatch.setattr(core, "_ts_conn", lambda: conn)

    core._delete_run_data("nightly-1")

    deletes = [(text, params) for text, params, _ in conn.log if text.startswith("DELETE")]
    assert ("DELETE FROM public.run_index WHERE run_name = %s", ("nightly-1",)) in deletes
    assert deletes[0][0] == "DELETE FROM public.metrics WHERE run_name = %s"


def test_deleting_a_service_deletes_its_runs_from_run_index_in_savepoints(monkeypatch):
    from loadlens_app.load_tests import store as lt_store

    monkeypatch.setattr(core, "_load_settings_runtime_data", lambda: {})
    monkeypatch.setattr(core, "_load_metrics_runtime_data", lambda: {})
    # Load test runs live in their own store (tests/test_run_archive.py); keep them off this connection.
    monkeypatch.setattr(lt_store, "_store", lt_store.MemoryLoadTestStore())
    # engineer_reports is missing: its failure is rolled back to its savepoint and the rest still runs.
    conn = Conn(("FROM public.engineer_reports", RuntimeError("relation does not exist")))
    monkeypatch.setattr(core, "_ts_conn", lambda: conn)

    core._delete_service_data("shop", "checkout")

    texts = conn.texts()
    assert "DELETE FROM public.run_index WHERE service = %s" in texts
    assert texts.count("SAVEPOINT service_delete") == 4
    assert texts.count("ROLLBACK TO SAVEPOINT service_delete") == 1
    assert texts.index("DELETE FROM public.metrics WHERE service = %s") < texts.index("DELETE FROM public.run_index WHERE service = %s")
    assert conn.commits == 1


def test_renaming_a_run_moves_its_run_index_row_after_the_name_checks(monkeypatch):
    monkeypatch.setattr(core, "_ensure_schema_and_table", lambda *args: None)
    monkeypatch.setattr(core, "_ensure_llm_reports_table", lambda *args: None)
    monkeypatch.setattr(core, "_ensure_engineer_reports_table", lambda *args: None)
    taken = {"old-run"}

    class RenameCursor(Cursor):
        def execute(self, query, params=None):
            super().execute(query, params)
            if flat(query).startswith("SELECT 1 FROM"):
                self.rows = [(1,)] if params[0] in taken else []

    class RenameConn(Conn):
        def cursor(self):
            return RenameCursor(self)

    conn = RenameConn()
    monkeypatch.setattr(core, "_ts_conn", lambda: conn)
    core._rename_run_data("old-run", "new-run")

    texts = conn.texts()
    probe = 'SELECT 1 FROM "public"."run_index" WHERE run_name = %s LIMIT 1'
    update = 'UPDATE "public"."run_index" SET run_name = %s WHERE run_name = %s'
    assert texts.count(probe) == 2  # the old name exists, the new one is free
    assert update in texts and texts.index(update) > max(i for i, t in enumerate(texts) if t == probe)

    taken.add("new-run")
    conn.log.clear()
    with pytest.raises(FileExistsError):
        core._rename_run_data("old-run", "new-run")
    assert not [t for t in conn.texts() if t.startswith("UPDATE")]


# ---- /runs and /dashboard_data -----------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app.blueprints import dashboard

    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    monkeypatch.setattr(core, "_ts_conn", lambda: Conn())
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: Conn())
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def ensured(monkeypatch):
    from loadlens_app.blueprints import dashboard

    calls = []
    monkeypatch.setattr(dashboard, "_ensure_run_index_table", lambda conn, cfg: calls.append(conn))
    return calls


def use(monkeypatch, conn):
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)
    return conn


def _page_row(name, total, max_rps=None):
    return (name, START, END, "checkout", "Успешно", "Успешно", None, END, "step", f"id-{name}", "shop", max_rps, total)


def test_runs_is_one_statement_over_run_index_with_the_total_in_a_column(client, monkeypatch, ensured):
    rows = [_page_row("b-run", 41, 120), _page_row("a-run", 41)]
    conn = use(monkeypatch, Conn(("FROM joined", rows)))

    resp = client.get("/runs", query_string={"q": "run", "sort": "max_rps", "offset": 20, "limit": 2})

    assert resp.status_code == 200
    assert resp.headers["X-Total-Count"] == "41"
    body = resp.get_json()
    assert [item["run_name"] for item in body] == ["b-run", "a-run"]  # the database order is kept
    assert set(body[0]) == {"run_name", "start_time", "end_time", "service", "verdict", "llm_verdict", "sla_verdict",
                            "report_created_at", "test_type", "test_type_key", "run_id", "project_area", "max_rps",
                            "manual_verdict"}
    assert body[0]["max_rps"] == 120.0 and body[1]["max_rps"] is None and body[0]["run_id"] == "id-b-run"
    assert body[0]["verdict"] == "pass" and body[0]["llm_verdict"] == "pass" and body[0]["sla_verdict"] is None
    assert ensured == [conn]

    (text, params, _), = conn.log
    assert "FROM public.run_index WHERE run_name <> '' AND (run_name ILIKE %s" in text
    assert "public.metrics" not in text and "GROUP BY" not in text
    assert "COUNT(*) OVER () AS total," in text and "manual_changed_at FROM joined" in text
    assert text.endswith("ORDER BY max_rps DESC NULLS LAST, run_name ASC OFFSET %s LIMIT %s")
    assert params == ("%run%", "%run%", 20, 2)


def test_runs_on_an_empty_index_answers_an_empty_list(client, monkeypatch, ensured):
    conn = use(monkeypatch, Conn())

    resp = client.get("/runs")

    assert resp.status_code == 200 and resp.get_json() == []
    assert resp.headers["X-Total-Count"] == "0"
    assert len(conn.log) == 1 and ensured == [conn]


def test_runs_past_the_last_page_still_reports_the_total(client, monkeypatch, ensured):
    # The archive steps back to the last page when a page is empty but the total is not.
    conn = use(monkeypatch, Conn(("SELECT COUNT(*) FROM joined", [(7,)])))

    resp = client.get("/runs", query_string={"offset": 40, "verdict": "Провал"})

    assert resp.status_code == 200 and resp.get_json() == []
    assert resp.headers["X-Total-Count"] == "7"
    page, count = conn.log
    assert "COUNT(*) OVER ()" in page[0]
    assert count[0].endswith("SELECT COUNT(*) FROM joined WHERE verdict = %s")
    assert count[1] == ("fail",)  # the old label of a shared link selects its code


def test_runs_restricts_run_index_and_reports_to_the_area(client, monkeypatch, ensured):
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: ["svc-a", "svc-b"])
    conn = use(monkeypatch, Conn())

    client.get("/runs")

    (text, params, _), = conn.log
    assert "FROM public.run_index WHERE run_name <> '' AND service = ANY(%s)" in text
    assert params == (["svc-a", "svc-b"], ["svc-a", "svc-b"], 0, 20)


def test_runs_answer_500_without_leaking_when_the_query_fails(client, monkeypatch, ensured):
    use(monkeypatch, Conn(("FROM joined", RuntimeError("relation run_index at db.internal"))))

    resp = client.get("/runs")

    assert resp.status_code == 500 and "db.internal" not in resp.get_json()["error"]


def test_dashboard_counts_runs_in_run_index(client, monkeypatch, ensured):
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: ["svc-a"])
    conn = use(monkeypatch, Conn(("COUNT(*) FROM public.run_index", [(12,)])))

    resp = client.get("/dashboard_data")

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.get_json()["runs_total"] == 12
    counted = [(text, params) for text, params, _ in conn.log if "run_index" in text]
    assert counted == [("SELECT COUNT(*) FROM public.run_index WHERE run_name <> '' AND service = ANY(%s)", (["svc-a"],))]
    assert not [text for text in conn.texts() if "metrics" in text]
    assert ensured == [conn]


# ---- migrations ---------------------------------------------------------------------------------------------


@pytest.fixture
def migrations(monkeypatch):
    from loadlens_app import migrations as module

    monkeypatch.setattr(db_store, "_ensure_schema_and_table", lambda *args: None)
    monkeypatch.setattr(db_store, "_ensure_run_index_table", lambda *args: None)
    return module


def test_backfill_indexes_everything_when_run_index_is_empty(migrations):
    conn = Conn(("pg_try_advisory_xact_lock", [(True,)]), ("SELECT MAX(end_time)", [(None,)]), ("INSERT INTO", [(1,)] * 300))

    assert migrations.backfill_run_index(lambda: conn, CFG) == 300

    texts = conn.texts()
    assert texts[0] == "SELECT pg_try_advisory_xact_lock(%s)" and conn.log[0][1] == (migrations.RUN_INDEX_BACKFILL_LOCK,)
    assert texts[1] == 'SELECT MAX(end_time) FROM "public"."run_index"'
    insert = conn.log[2]
    assert insert[0].startswith('INSERT INTO "public"."run_index"') and '"time" > %s' not in insert[0]
    assert conn.commits == 1 and conn.closed


def test_backfill_catches_up_from_a_day_before_the_newest_run(migrations):
    conn = Conn(("pg_try_advisory_xact_lock", [(True,)]), ("SELECT MAX(end_time)", [(END,)]))

    migrations.backfill_run_index(lambda: conn, CFG)

    text, params, _ = conn.log[2]
    assert '"time" > %s' in text and params == (END - timedelta(days=1),)


def test_backfill_skips_while_another_process_holds_the_lock(migrations):
    conn = Conn(("pg_try_advisory_xact_lock", [(False,)]))

    assert migrations.backfill_run_index(lambda: conn, CFG) == 0
    assert not [text for text in conn.texts() if text.startswith("INSERT")]
    assert conn.closed


def test_backfill_prepares_both_tables_first(monkeypatch):
    from loadlens_app import migrations

    prepared = []
    monkeypatch.setattr(db_store, "_ensure_schema_and_table", lambda conn, cfg, schema, table: prepared.append(("metrics", schema, table)))
    monkeypatch.setattr(db_store, "_ensure_run_index_table", lambda conn, cfg: prepared.append(("run_index",)))
    conn = Conn(("pg_try_advisory_xact_lock", [(False,)]))

    migrations.backfill_run_index(lambda: conn, {"schema": "lt", "table": "m"})

    assert prepared == [("metrics", "lt", "m"), ("run_index",)]


def _index_conn(*, table=True, existing=None, timescale=True, hypertable=True, locked=True):
    script = [("pg_try_advisory_lock", [(locked,)]), ("to_regclass", [(table,)]),
              ("FROM pg_index", [existing] if existing is not None else []),
              ("FROM pg_extension", [(timescale,)]), ("timescaledb_information.hypertables", [(hypertable,)])]
    return Conn(*script)


def test_metrics_run_name_index_is_built_per_chunk_on_a_hypertable(migrations):
    conn = _index_conn()

    assert migrations.ensure_metrics_run_name_index(lambda: conn, CFG) == 1

    create = [entry for entry in conn.log if entry[0].startswith("CREATE INDEX")]
    assert create == [(
        'CREATE INDEX IF NOT EXISTS "idx_metrics_run_name" ON "public"."metrics" (run_name, "time")'
        " WITH (timescaledb.transaction_per_chunk)", None, True,
    )]
    assert conn.log[1][1] == ("public", "metrics")
    assert conn.texts()[-1] == "SELECT pg_advisory_unlock(%s)"
    assert all(autocommit for _text, _params, autocommit in conn.log)  # never inside a transaction block
    assert conn.autocommit is False and conn.closed


def test_metrics_run_name_index_on_a_plain_table(migrations):
    conn = _index_conn(timescale=False)

    assert migrations.ensure_metrics_run_name_index(lambda: conn, {"schema": "lt", "table": "m"}) == 1

    assert 'CREATE INDEX IF NOT EXISTS "idx_m_run_name" ON "lt"."m" (run_name, "time")' in conn.texts()
    assert not [text for text in conn.texts() if "hypertables" in text]


def test_metrics_run_name_index_that_exists_is_left_alone(migrations):
    conn = _index_conn(existing=(True,))

    assert migrations.ensure_metrics_run_name_index(lambda: conn, CFG) == 0

    assert not [text for text in conn.texts() if text.startswith(("CREATE", "DROP"))]
    assert conn.texts()[-1] == "SELECT pg_advisory_unlock(%s)"


def test_an_invalid_metrics_run_name_index_is_rebuilt(migrations):
    conn = _index_conn(existing=(False,))

    assert migrations.ensure_metrics_run_name_index(lambda: conn, CFG) == 1

    texts = conn.texts()
    drop = 'DROP INDEX IF EXISTS "public"."idx_metrics_run_name"'
    assert drop in texts and texts.index(drop) < next(i for i, t in enumerate(texts) if t.startswith("CREATE INDEX"))


def test_metrics_run_name_index_skips_a_missing_table_and_a_held_lock(migrations):
    missing = _index_conn(table=False)
    assert migrations.ensure_metrics_run_name_index(lambda: missing, CFG) == 0
    assert not [text for text in missing.texts() if text.startswith("CREATE")]
    assert missing.texts()[-1] == "SELECT pg_advisory_unlock(%s)"

    busy = _index_conn(locked=False)
    assert migrations.ensure_metrics_run_name_index(lambda: busy, CFG) == 0
    assert busy.texts() == ["SELECT pg_try_advisory_lock(%s)"] and busy.closed


def test_run_all_indexes_runs_after_their_ids_and_before_the_index_build():
    from loadlens_app import migrations

    order = [migration.func for migration in migrations.MIGRATIONS]
    ids, run_index, index = (order.index(name) for name in ("backfill_run_ids", "backfill_run_index", "ensure_metrics_run_name_index"))
    assert ids < run_index < index
    by_func = {migration.func: migration for migration in migrations.MIGRATIONS}
    # The index build and the run_index catch-up are cheap checks that repeat at every start.
    assert not by_func["ensure_metrics_run_name_index"].once and not by_func["backfill_run_index"].once


# ---- deployment schema --------------------------------------------------------------------------------------


def test_deployment_schema_creates_run_index_after_metrics():
    schema = (ROOT_DIR / "docker" / "initdb" / "02_schema.sql").read_text(encoding="utf-8")
    flat_schema = " ".join(schema.split())
    assert "CREATE TABLE IF NOT EXISTS public.run_index ( run_name TEXT PRIMARY KEY," in flat_schema
    assert "CREATE INDEX IF NOT EXISTS idx_run_index_service_end ON public.run_index (service, end_time DESC);" in flat_schema
    position = schema.index("public.run_index")
    assert schema.index("idx_metrics_run_name") < position < schema.index("public.llm_reports")


# ---- tools/check_run_index.py -------------------------------------------------------------------------------


def _check_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_run_index_tool", ROOT_DIR / "tools" / "check_run_index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_tool_reports_each_kind_of_difference_and_stays_local():
    tool = _check_tool()
    rows = [
        ("a", "missing", START, None, END, None, "svc", None, "id", None),
        ("b", "span", START, START + timedelta(minutes=1), END, END, "svc", "svc", "id", "id"),
    ]
    text = tool.report({"metrics_runs": 2, "run_index_rows": 1}, rows, 0.25)
    assert "runs in metrics: 2, rows in run_index: 1" in text
    assert "differences: missing 1, stale 0, span 1, fields 0" in text
    assert "  missing a:" in text and "  span    b:" in text
    assert tool.report({"metrics_runs": 0, "run_index_rows": 0}, [], 0).endswith("missing 0, stale 0, span 0, fields 0")

    args = tool.parse_args(["--sandbox", "5"])
    assert tool.connection_params(args)["dbname"] == "ll_s5"
    with pytest.raises(SystemExit):
        tool.connection_params(tool.parse_args(["--dsn", "host=db.internal dbname=lt"]))
