"""One-off data migrations that must not run inside ordinary GET requests.

Run them with ``python -m loadlens_app.migrations``. The app container starts them in the background
when ``LOADLENS_RUN_MIGRATIONS=1`` (docker-compose sets it). Each migration is idempotent, and a
PostgreSQL advisory lock keeps several processes from running the same one at once.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from datetime import timedelta
from typing import Any, Callable, Optional

from psycopg2 import sql

logger = logging.getLogger(__name__)

# Arbitrary constant shared by every process of this app.
RUN_ID_BACKFILL_LOCK = 727_101
RUN_INDEX_BACKFILL_LOCK = 727_105
METRICS_RUN_NAME_INDEX_LOCK = 727_106
METRICS_TABLE = ("public", "metrics")
# How far before the newest indexed run the start-up catch-up looks for late metric rows.
RUN_INDEX_OVERLAP = timedelta(days=1)

_started = False
_start_lock = threading.Lock()


def _storage_names() -> tuple[str, str]:
    from loadlens_app import core

    cfg = (core.CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return str(cfg.get("schema") or "public"), str(cfg.get("llm_table") or "llm_reports")


def backfill_run_ids(connect: Optional[Callable[[], Any]] = None) -> int:
    """Gives every stored run without an id one id shared by all its rows; returns how many runs got one.

    A run is a ``(run_name, service)`` pair. When one of the two tables already has an id for the run,
    that id is reused for the rows that lack it, so report pages and metrics stay linked.
    """
    if connect is None:
        from loadlens_app import core

        connect = core._ts_conn
    schema, llm_table = _storage_names()
    reports = sql.Identifier(schema, llm_table)
    metrics = sql.Identifier(*METRICS_TABLE)
    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (RUN_ID_BACKFILL_LOCK,))
            if not cur.fetchone()[0]:
                logger.info("run_id backfill is already running in another process")
                return 0
            try:
                cur.execute(
                    sql.SQL(
                        "SELECT run_name, service, MAX(run_id) FROM ("
                        " SELECT run_name, COALESCE(service, '') AS service, NULLIF(run_id, '') AS run_id FROM {reports}"
                        " UNION ALL"
                        " SELECT run_name, COALESCE(service, '') AS service, NULLIF(run_id, '') AS run_id FROM {metrics}"
                        ") runs WHERE run_name IS NOT NULL AND run_name <> ''"
                        " GROUP BY run_name, service HAVING bool_or(run_id IS NULL)"
                    ).format(reports=reports, metrics=metrics)
                )
                missing = cur.fetchall()
                for run_name, service, existing in missing:
                    run_id = existing or uuid.uuid4().hex
                    for table in (reports, metrics):
                        cur.execute(
                            sql.SQL(
                                "UPDATE {} SET run_id = %s WHERE run_name = %s AND COALESCE(service, '') = %s"
                                " AND COALESCE(run_id, '') = ''"
                            ).format(table),
                            (run_id, run_name, service),
                        )
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (RUN_ID_BACKFILL_LOCK,))
        if missing:
            logger.info("run_id backfill: %d run(s) got an id", len(missing))
        return len(missing)
    finally:
        conn.close()


def _timescale_cfg() -> dict:
    from loadlens_app import core

    return (core.CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}


def backfill_run_index(connect: Optional[Callable[[], Any]] = None, storage_cfg: Optional[dict] = None) -> int:
    """Fills ``run_index`` from the metrics table; returns the number of run rows written.

    An empty index gets a full pass (one GROUP BY over metrics). Otherwise only metric rows newer than the
    newest indexed run minus ``RUN_INDEX_OVERLAP`` are merged: new rows are indexed when they are saved,
    so this only catches up after a failed save or rows written by another tool.
    """
    from AI.db_store import RUN_INDEX_TABLE, _ensure_run_index_table, _ensure_schema_and_table, refresh_run_index

    if connect is None:
        from loadlens_app import core

        connect = core._ts_conn
    cfg = _timescale_cfg() if storage_cfg is None else storage_cfg
    schema = str(cfg.get("schema") or "public")
    conn = connect()
    try:
        # A fresh database may have neither table yet; both helpers are no-ops on an existing one.
        _ensure_schema_and_table(conn, cfg, schema, str(cfg.get("table") or "metrics"))
        _ensure_run_index_table(conn, cfg)
        with conn, conn.cursor() as cur:
            # Transaction-scoped: released by the commit or rollback, even on a pooled connection.
            cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (RUN_INDEX_BACKFILL_LOCK,))
            if not cur.fetchone()[0]:
                logger.info("run_index backfill is already running in another process")
                return 0
            cur.execute(sql.SQL("SELECT MAX(end_time) FROM {}").format(sql.Identifier(schema, RUN_INDEX_TABLE)))
            latest = cur.fetchone()[0]
            since = latest - RUN_INDEX_OVERLAP if latest is not None else None
            written = refresh_run_index(conn, cfg, since=since)
        if since is None:
            logger.info("run_index backfill: %d run(s) indexed (full pass)", written)
        else:
            logger.info("run_index backfill: %d run(s) merged from metric rows after %s", written, since)
        return written
    finally:
        conn.close()


def ensure_metrics_run_name_index(connect: Optional[Callable[[], Any]] = None, storage_cfg: Optional[dict] = None) -> int:
    """Creates ``idx_<metrics>_run_name ON metrics (run_name, time)`` when it is missing; returns 1 if built.

    docker/initdb/02_schema.sql creates it, the application's table helpers do not, so a database set up
    without that script lacks it, and every lookup, delete or rename by run name reads all chunks.
    Building it on a large hypertable takes a while, hence a start-up migration and never the request
    path: one transaction per chunk, so only the chunk being indexed blocks its writers. An invalid
    index left by an interrupted build is dropped and built again.
    """
    if connect is None:
        from loadlens_app import core

        connect = core._ts_conn
    cfg = _timescale_cfg() if storage_cfg is None else storage_cfg
    schema = str(cfg.get("schema") or "public")
    table = str(cfg.get("table") or "metrics")
    index = f"idx_{table}_run_name"
    conn = connect()
    try:
        # CREATE INDEX ... WITH (timescaledb.transaction_per_chunk) refuses to run inside a transaction block.
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (METRICS_RUN_NAME_INDEX_LOCK,))
            if not cur.fetchone()[0]:
                logger.info("metrics run_name index is being built by another process")
                return 0
            try:
                cur.execute("SELECT to_regclass(format('%%I.%%I', %s, %s)) IS NOT NULL", (schema, table))
                if not cur.fetchone()[0]:
                    return 0
                cur.execute(
                    "SELECT i.indisvalid FROM pg_index i"
                    " JOIN pg_class c ON c.oid = i.indexrelid"
                    " JOIN pg_namespace n ON n.oid = c.relnamespace"
                    " WHERE n.nspname = %s AND c.relname = %s",
                    (schema, index),
                )
                existing = cur.fetchone()
                if existing and existing[0]:
                    return 0
                if existing:
                    logger.warning("Index %s.%s is invalid (interrupted build); rebuilding it", schema, index)
                    cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(schema, index)))
                cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
                )
                hypertable = False
                if cur.fetchone()[0]:
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM timescaledb_information.hypertables"
                        " WHERE hypertable_schema = %s AND hypertable_name = %s)",
                        (schema, table),
                    )
                    hypertable = bool(cur.fetchone()[0])
                statement = sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {} (run_name, "time")').format(
                    sql.Identifier(index), sql.Identifier(schema, table)
                )
                if hypertable:
                    statement = statement + sql.SQL(" WITH (timescaledb.transaction_per_chunk)")
                logger.info("Building index %s.%s on %s.%s (hypertable=%s)", schema, index, schema, table, hypertable)
                started = time.monotonic()
                cur.execute(statement)
                logger.info("Index %s.%s built in %.1f s", schema, index, time.monotonic() - started)
                return 1
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (METRICS_RUN_NAME_INDEX_LOCK,))
    finally:
        try:
            conn.autocommit = False
        except Exception:
            pass
        conn.close()


def run_all() -> dict[str, int]:
    # run_index after the run ids (it copies them) and before the slow index build, so /runs is ready early.
    return {
        "backfill_run_ids": backfill_run_ids(),
        "backfill_run_index": backfill_run_index(),
        "metrics_run_name_index": ensure_metrics_run_name_index(),
    }


def start_background_migrations() -> None:
    """Runs the migrations once per process in a daemon thread; failures are logged, not raised."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True

    def _run() -> None:
        try:
            run_all()
        except Exception:
            logger.exception("Background migrations failed; run `python -m loadlens_app.migrations` manually")

    threading.Thread(target=_run, name="loadlens-migrations", daemon=True).start()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        for name, count in run_all().items():
            print(f"{name}: {count}")
    except Exception as exc:  # database unreachable, bad settings, ...
        print(f"Ошибка миграции: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
