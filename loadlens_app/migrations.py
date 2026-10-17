"""Data migrations that must not run inside ordinary GET requests.

The application checks them at every start (unless ``LOADLENS_RUN_MIGRATIONS=0``) in a background thread:
one-off migrations are recorded in the table ``app_migrations`` of the storage database and skipped once
applied, catch-up migrations are cheap and run every time. So a database that an older version of the
application wrote gets the missing migrations right after the update, whoever deploys it. While the
database is unavailable the thread waits and tries again. ``python -m loadlens_app.migrations`` runs the
same thing by hand (``--status`` only lists, ``--force`` repeats the one-off migrations). Each migration is
idempotent, and a PostgreSQL advisory lock keeps several processes from running the same one at once.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional

import psycopg2
from psycopg2 import sql

from AI import storage_state

logger = logging.getLogger(__name__)

# Arbitrary constant shared by every process of this app.
RUN_ID_BACKFILL_LOCK = 727_101
# Advisory lock of backfill_max_rps (any value not used by another migration).
MAX_RPS_BACKFILL_LOCK = 727_104
# Final rows per UPDATE of backfill_max_rps; each batch is its own transaction.
MAX_RPS_BACKFILL_BATCH = 5000
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


def backfill_max_rps(connect: Optional[Callable[[], Any]] = None, batch_size: int = MAX_RPS_BACKFILL_BATCH) -> int:
    """Fills ``max_rps`` of final rows saved before the column existed; returns how many rows got a value.

    The value comes from ``parsed`` by the rule of ``save_llm_results`` (``AI.db_store.max_rps_sql``). Rows
    whose analysis has no plain number keep NULL; the read queries parse those on the fly. The rows are
    walked by ``id`` in batches of ``batch_size``, one transaction per batch, so the table is never locked
    for the whole pass. A table that has not got the column yet is brought up to date first.
    """
    from AI import db_store

    if connect is None:
        from loadlens_app import core

        connect = core._ts_conn
    schema, llm_table = _storage_names()
    reports = sql.Identifier(schema, llm_table)
    rule = sql.SQL(db_store.max_rps_sql("parsed"))
    statement = sql.SQL(
        "WITH batch AS ("
        " SELECT id FROM {reports} WHERE domain = 'final' AND max_rps IS NULL AND id > %s ORDER BY id LIMIT %s"
        "), filled AS ("
        " UPDATE {reports} AS r SET max_rps = {rule} FROM batch WHERE r.id = batch.id AND {rule} IS NOT NULL"
        " RETURNING r.id"
        ") SELECT (SELECT MAX(id) FROM batch), (SELECT COUNT(*) FROM filled)"
    ).format(reports=reports, rule=rule)
    conn = connect()
    filled = 0
    try:
        # Adds the column (and the indexes) when no request of the new code has touched the table yet.
        db_store._ensure_llm_reports_table(conn, {"schema": schema, "llm_table": llm_table})
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (MAX_RPS_BACKFILL_LOCK,))
            locked = cur.fetchone()[0]
            conn.commit()
            if not locked:
                logger.info("max_rps backfill is already running in another process")
                return 0
            try:
                last_id = 0  # BIGSERIAL ids start at 1; a bigint bound keeps the primary key usable
                while True:
                    cur.execute(statement, (last_id, int(batch_size)))
                    batch_last, batch_filled = cur.fetchone()
                    conn.commit()
                    if batch_last is None:
                        break
                    last_id = batch_last
                    filled += int(batch_filled or 0)
            finally:
                conn.rollback()  # a failed batch leaves an aborted transaction; the unlock needs a fresh one
                cur.execute("SELECT pg_advisory_unlock(%s)", (MAX_RPS_BACKFILL_LOCK,))
                conn.commit()
        if filled:
            logger.info("max_rps backfill: %d final row(s) got a value", filled)
        return filled
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


# ---- registry: which migrations this database already has -------------------------------------------------


@dataclass(frozen=True)
class Migration:
    """One entry of :data:`MIGRATIONS`.

    ``key`` names the migration in the registry table, the logs and the CLI output; ``func`` is the module
    function that runs it (looked up by name at run time). ``once`` migrations are recorded after a successful
    run and skipped from then on; the others are cheap catch-ups that run at every start.
    """

    key: str
    func: str
    once: bool
    title: str


# In this order: run_index copies the run ids, and it fills before the slow index build so /runs is ready early.
MIGRATIONS: tuple[Migration, ...] = (
    Migration("backfill_run_ids", "backfill_run_ids", True, "идентификаторы прогонов, сохранённых до появления run_id"),
    Migration("backfill_run_index", "backfill_run_index", False, "таблица run_index по metrics (полный проход один раз, затем досчёт)"),
    Migration("backfill_max_rps", "backfill_max_rps", True, "колонка max_rps итоговых отчётов, сохранённых до её появления"),
    Migration("metrics_run_name_index", "ensure_metrics_run_name_index", False, "индекс idx_metrics_run_name на таблице metrics"),
)
REGISTRY_TABLE = "app_migrations"
# Waits between attempts while the storage database is unavailable at start-up; the last one repeats.
RETRY_DELAYS_SEC: tuple[int, ...] = (30, 60, 120, 300, 600)


def enabled_by_env(env: Optional[dict] = None) -> bool:
    """Start-up migrations run unless ``LOADLENS_RUN_MIGRATIONS`` says 0, false, no or off."""
    value = (env if env is not None else os.environ).get("LOADLENS_RUN_MIGRATIONS", "")
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _default_connect() -> Callable[[], Any]:
    from loadlens_app import core

    return core._ts_conn


def _registry_ref() -> sql.Composed:
    schema, _llm_table = _storage_names()
    return sql.Identifier(schema, REGISTRY_TABLE)


def _ensure_registry(conn) -> bool:
    """Creates the registry table; False when this role cannot, then nothing is skipped or recorded."""
    previous = getattr(conn, "autocommit", False)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                    " result INTEGER, runs INTEGER NOT NULL DEFAULT 1)"
                ).format(_registry_ref())
            )
        return True
    except Exception as exc:
        logger.warning("Migration registry %s is not available (%s): every migration runs at every start", REGISTRY_TABLE, exc)
        return False
    finally:
        try:
            conn.autocommit = previous
        except Exception:
            pass


def applied_migrations(conn) -> set[str]:
    """Keys recorded in the registry table (the table must exist)."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT name FROM {}").format(_registry_ref()))
        rows = cur.fetchall() or []
    conn.commit()
    return {str(row[0]) for row in rows}


def _record(conn, key: str, result: int) -> None:
    ref = _registry_ref()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "INSERT INTO {} (name, result) VALUES (%s, %s)"
                " ON CONFLICT (name) DO UPDATE SET applied_at = now(), result = EXCLUDED.result, runs = {}.runs + 1"
            ).format(ref, ref),
            (key, int(result or 0)),
        )
    conn.commit()


def pending_migrations(connect: Optional[Callable[[], Any]] = None) -> list[str]:
    """Keys of the one-off migrations this database has not recorded yet (all of them without a registry)."""
    conn = (connect or _default_connect())()
    try:
        if not _ensure_registry(conn):
            return [m.key for m in MIGRATIONS if m.once]
        applied = applied_migrations(conn)
        return [m.key for m in MIGRATIONS if m.once and m.key not in applied]
    finally:
        conn.close()


def run_all(force: bool = False, connect: Optional[Callable[[], Any]] = None) -> dict[str, Optional[int]]:
    """Runs the migrations in order; a one-off migration already in the registry is skipped and reported as None.

    ``connect`` opens the registry connection (the migrations open their own); ``force`` runs the recorded
    one-off migrations again. Each migration is recorded right after it ran, so a failure in a later one does
    not repeat the earlier ones at the next start.
    """
    registry = (connect or _default_connect())()
    results: dict[str, Optional[int]] = {}
    try:
        tracked = _ensure_registry(registry)
        applied = applied_migrations(registry) if tracked else set()
        for migration in MIGRATIONS:
            if migration.once and migration.key in applied and not force:
                results[migration.key] = None
                continue
            count = int(globals()[migration.func]() or 0)  # by name, so tests can replace a migration
            results[migration.key] = count
            if tracked:
                try:
                    _record(registry, migration.key, count)
                except Exception:
                    logger.warning("Migration %s ran but could not be recorded in %s", migration.key, REGISTRY_TABLE, exc_info=True)
                    try:
                        registry.rollback()
                    except Exception:
                        pass
        return results
    finally:
        registry.close()


def _retryable(exc: BaseException) -> bool:
    return storage_state.is_storage_error(exc) or isinstance(exc, psycopg2.OperationalError)


def run_with_retries(
    run: Callable[[], dict[str, Optional[int]]] = run_all,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: Optional[int] = None,
) -> Optional[dict[str, Optional[int]]]:
    """``run()`` until it succeeds; waits :data:`RETRY_DELAYS_SEC` while the storage database is unavailable.

    Any other failure is logged once and ends the attempts (a bug does not deserve a retry loop). Returns the
    results, or None when the attempts ran out or the failure was not about the database.
    """
    attempt = 0
    while True:
        try:
            results = run()
        except Exception as exc:
            if not _retryable(exc):
                logger.exception("Migrations failed; run `python -m loadlens_app.migrations` manually")
                return None
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                logger.error("Migrations postponed %d times, the storage database is still unavailable: %s", attempt, exc)
                return None
            delay = RETRY_DELAYS_SEC[min(attempt - 1, len(RETRY_DELAYS_SEC) - 1)]
            logger.warning("Migrations postponed: the storage database is unavailable (%s); next try in %d s", exc, delay)
            sleep(delay)
            continue
        once_keys = {m.key for m in MIGRATIONS if m.once}
        applied = [f"{key}={count}" for key, count in results.items() if count is not None and key in once_keys]
        catch_ups = [f"{key}={count}" for key, count in results.items() if count is not None and key not in once_keys]
        skipped = [key for key, count in results.items() if count is None]
        tail = (f"; catch-ups: {', '.join(catch_ups)}" if catch_ups else "") + (f"; already applied: {', '.join(skipped)}" if skipped else "")
        if applied:
            # A one-off migration changed this database: worth a line even where INFO is not shown.
            logger.warning("Migrations applied: %s%s", ", ".join(applied), tail)
        else:
            logger.info("Migrations checked: nothing new to apply%s", tail)
        return results


def start_background_migrations() -> bool:
    """Checks and runs the migrations once per process in a daemon thread; False when it already started."""
    global _started
    with _start_lock:
        if _started:
            return False
        _started = True
    threading.Thread(target=run_with_retries, name="loadlens-migrations", daemon=True).start()
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Миграции данных LoadLens: проверяет реестр app_migrations и выполняет недостающие.")
    parser.add_argument("--status", action="store_true", help="только показать, какие миграции выполнены, а какие ожидают")
    parser.add_argument("--force", action="store_true", help="выполнить одноразовые миграции повторно, даже если они записаны как выполненные")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        if args.status:
            pending = set(pending_migrations())
            for migration in MIGRATIONS:
                if not migration.once:
                    state = "выполняется при каждом старте"
                else:
                    state = "ожидает" if migration.key in pending else "выполнена"
                print(f"{migration.key}: {state} — {migration.title}")
            return 2 if pending else 0
        for key, count in run_all(force=args.force).items():
            print(f"{key}: {'пропущена (уже выполнена)' if count is None else count}")
    except Exception as exc:  # database unreachable, bad settings, ...
        print(f"Ошибка миграции: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
