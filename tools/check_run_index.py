# SPDX-License-Identifier: Apache-2.0
"""Compares ``run_index`` with a GROUP BY over the metrics table and prints the differences.

The run list (/runs) and the dashboard run counter read ``run_index``; ``AI.db_store.save_domain_labeled``
keeps it current and ``loadlens_app.migrations.backfill_run_index`` fills it for older data. This script
checks that the two agree (plans/perf-plan.md, agent 5):

    C:/Python312/python.exe tools/check_run_index.py --sandbox 5
    C:/Python312/python.exe tools/check_run_index.py --dsn "host=127.0.0.1 port=55432 dbname=ll_s5 user=ltar password=..."
    C:/Python312/python.exe tools/check_run_index.py --sandbox 5 --rebuild     # rebuild run_index, then check

``--rebuild`` empties ``run_index`` and fills it again from metrics in one transaction (readers see the old
or the new content, never an empty list). Exit code: 0 when everything matches, 1 on differences.

Kinds of differences: ``missing`` (a run in metrics without a row), ``stale`` (a row without metrics),
``span`` (other start/end time) and ``fields`` (other service or run id: the index keeps the last non-empty
value, the GROUP BY takes MAX, so a run saved under several services shows up here without being broken).

Connects only to 127.0.0.1/localhost/::1 unless ``--allow-remote`` is given; never imports ``settings.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
SANDBOX_DB = {"host": "127.0.0.1", "port": 55432, "user": "ltar", "password": "e2e-only-password"}
SHOW = 10

COMPARE_SQL = """
WITH m AS (
  SELECT run_name, COALESCE(MAX(service), '') AS service, COALESCE(MAX(NULLIF(run_id, '')), '') AS run_id,
         MIN("time") AS start_time, MAX("time") AS end_time
  FROM {metrics}
  WHERE run_name IS NOT NULL AND run_name <> ''
  GROUP BY run_name
)
SELECT COALESCE(m.run_name, ri.run_name) AS run_name,
       CASE WHEN ri.run_name IS NULL THEN 'missing'
            WHEN m.run_name IS NULL THEN 'stale'
            WHEN m.start_time <> ri.start_time OR m.end_time <> ri.end_time THEN 'span'
            ELSE 'fields' END AS kind,
       m.start_time, ri.start_time, m.end_time, ri.end_time, m.service, ri.service, m.run_id, ri.run_id
FROM m FULL JOIN {index} ri ON ri.run_name = m.run_name
WHERE m.run_name IS NULL OR ri.run_name IS NULL
   OR m.start_time <> ri.start_time OR m.end_time <> ri.end_time
   OR m.service <> ri.service OR m.run_id <> ri.run_id
ORDER BY 2, 1
"""


def _db_store():
    """AI.db_store without a settings.py: the run_index helpers only use the storage config passed to them."""
    if "settings" not in sys.modules:
        stand_in = types.ModuleType("settings")
        stand_in.CONFIG = {}
        sys.modules["settings"] = stand_in
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from AI import db_store

    return db_store


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compares run_index with the metrics table (see the module docstring).")
    parser.add_argument("--sandbox", type=int, help="sandbox number N: database ll_s<N> on 127.0.0.1:55432")
    parser.add_argument("--dsn", help="libpq connection string (overrides --host/--port/--dbname/--user/--password)")
    parser.add_argument("--host", default=SANDBOX_DB["host"])
    parser.add_argument("--port", type=int, default=SANDBOX_DB["port"])
    parser.add_argument("--dbname")
    parser.add_argument("--user", default=SANDBOX_DB["user"])
    parser.add_argument("--password", default=SANDBOX_DB["password"])
    parser.add_argument("--schema", default="public")
    parser.add_argument("--table", default="metrics", help="metrics table (storage.timescale.table)")
    parser.add_argument("--allow-remote", action="store_true", help="allow a database host other than 127.0.0.1/localhost")
    parser.add_argument("--rebuild", action="store_true", help="refill run_index from metrics before checking")
    args = parser.parse_args(argv)
    if args.sandbox is not None:
        args.dbname = args.dbname or f"ll_s{args.sandbox}"
    if not args.dsn and not args.dbname:
        parser.error("pass --sandbox N, --dsn or --dbname")
    return args


def connection_params(args: argparse.Namespace) -> dict:
    if args.dsn:
        from psycopg2.extensions import parse_dsn

        params = parse_dsn(args.dsn)
    else:
        params = {"host": args.host, "port": args.port, "dbname": args.dbname, "user": args.user, "password": args.password}
    params = {k: v for k, v in params.items() if v not in (None, "")}
    host = str(params.get("host") or "localhost")
    if host not in LOCAL_HOSTS and not args.allow_remote:
        raise SystemExit(f"refusing to connect to {host}: only a local sandbox database is allowed (pass --allow-remote to override)")
    params.setdefault("connect_timeout", 5)
    return params


def rebuild(conn, storage_cfg: dict) -> int:
    """Replaces the content of run_index with a full pass over metrics, in one transaction."""
    db_store = _db_store()
    from psycopg2 import sql

    db_store._ensure_run_index_table(conn, storage_cfg)
    with conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(storage_cfg["schema"], db_store.RUN_INDEX_TABLE)))
        return db_store.refresh_run_index(conn, storage_cfg)


def compare(conn, storage_cfg: dict) -> tuple[dict, list[tuple]]:
    db_store = _db_store()
    from psycopg2 import sql

    metrics = sql.Identifier(storage_cfg["schema"], storage_cfg["table"])
    index = sql.Identifier(storage_cfg["schema"], db_store.RUN_INDEX_TABLE)
    with conn, conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(DISTINCT run_name) FROM {} WHERE run_name IS NOT NULL AND run_name <> ''").format(metrics))
        metric_runs = int(cur.fetchone()[0])
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(index))
        index_rows = int(cur.fetchone()[0])
        cur.execute(sql.SQL(COMPARE_SQL).format(metrics=metrics, index=index))
        differences = cur.fetchall()
    return {"metrics_runs": metric_runs, "run_index_rows": index_rows}, differences


def report(counts: dict, differences: list[tuple], seconds: float) -> str:
    kinds = {kind: 0 for kind in ("missing", "stale", "span", "fields")}
    for row in differences:
        kinds[row[1]] = kinds.get(row[1], 0) + 1
    lines = [
        f"runs in metrics: {counts['metrics_runs']}, rows in run_index: {counts['run_index_rows']} ({seconds:.1f} s)",
        "differences: " + ", ".join(f"{kind} {count}" for kind, count in kinds.items()),
    ]
    for row in differences[:SHOW]:
        name, kind, m_start, i_start, m_end, i_end, m_service, i_service, m_run_id, i_run_id = row
        lines.append(
            f"  {kind:7} {name}: start {m_start} / {i_start}, end {m_end} / {i_end}, "
            f"service {m_service!r} / {i_service!r}, run_id {m_run_id!r} / {i_run_id!r}   (metrics / run_index)"
        )
    if len(differences) > SHOW:
        lines.append(f"  ... and {len(differences) - SHOW} more")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import psycopg2

    storage_cfg = {"schema": args.schema, "table": args.table}
    conn = psycopg2.connect(**connection_params(args))
    try:
        if args.rebuild:
            started = time.perf_counter()
            written = rebuild(conn, storage_cfg)
            print(f"rebuilt run_index: {written} run(s) in {time.perf_counter() - started:.1f} s")
        started = time.perf_counter()
        counts, differences = compare(conn, storage_cfg)
        print(report(counts, differences, time.perf_counter() - started))
    finally:
        conn.close()
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
