"""One-off data migrations that must not run inside ordinary GET requests.

Run them with ``python -m loadlens_app.migrations``. The app container starts them in the background
when ``LOADLENS_RUN_MIGRATIONS=1`` (docker-compose sets it). Each migration is idempotent, and a
PostgreSQL advisory lock keeps several processes from running the same one at once.
"""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from typing import Any, Callable, Optional

from psycopg2 import sql

logger = logging.getLogger(__name__)

# Arbitrary constant shared by every process of this app.
RUN_ID_BACKFILL_LOCK = 727_101
METRICS_TABLE = ("public", "metrics")

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


def run_all() -> dict[str, int]:
    return {"backfill_run_ids": backfill_run_ids()}


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
