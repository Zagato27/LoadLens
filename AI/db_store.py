# SPDX-License-Identifier: Apache-2.0
import logging
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_batch

from AI import storage_state
from AI.i18n import get_locale, t
from AI.scoring import parse_llm_analysis_strict
from AI.verdicts import CODES as VERDICT_CODES, PASS, code_sql, exact_code, normalize_report, optional_code, to_code
from AI.verdicts import label as verdict_label


logger = logging.getLogger(__name__)
_ENSURED_TABLES: set[tuple[str, str]] = set()


def _df_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "series", "value"])

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["time", "series", "value"])

    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")

    work.index.name = "time"
    work_reset = work.reset_index()
    long_df = work_reset.melt(id_vars=["time"], var_name="series", value_name="value")

    long_df["time"] = pd.to_datetime(long_df["time"], utc=True, errors="coerce")
    long_df = long_df.dropna(subset=["time", "value"])
    long_df["series"] = long_df["series"].fillna("").astype(str)
    long_df["value"] = long_df["value"].astype(float)
    return long_df


def _connect(storage_cfg: Dict[str, object]):
    """Connection through the storage breaker (short timeout, fast failures while the database is down)."""
    dsn = storage_cfg.get("dsn")
    if isinstance(dsn, str) and dsn.strip():
        return storage_state.connect({"dsn": dsn}, storage_cfg)

    params = {
        "host": storage_cfg.get("host", "localhost"),
        "port": int(storage_cfg.get("port", 5432)),
        "dbname": storage_cfg.get("dbname", "loadtesting"),
        "user": storage_cfg.get("user"),
        "password": storage_cfg.get("password"),
        "sslmode": storage_cfg.get("sslmode", "prefer"),
    }
    return storage_state.connect(params, storage_cfg)


def _ensure_schema_and_table(conn, storage_cfg: Dict[str, object], schema: str, table: str) -> None:
    key = (schema, table)
    if key in _ENSURED_TABLES:
        return

    create_extension = bool(storage_cfg.get("ensure_extension", False))
    make_hypertable = bool(storage_cfg.get("make_hypertable", True))
    chunk_interval = storage_cfg.get("chunk_interval")

    prev_autocommit = getattr(conn, "autocommit", False)
    conn.autocommit = True  # DDL в отдельной транзакции, чтобы не оставлять соединение в aborted
    try:
        with conn.cursor() as cur:
            if create_extension:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                except Exception as e:
                    logger.warning("Не удалось создать расширение timescaledb (продолжаем): %s", e)

            try:
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(schema))
                )
            except Exception as e:
                logger.warning("Не удалось создать схему %s: %s", schema, e)

            try:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            time        TIMESTAMPTZ NOT NULL,
                            domain      TEXT        NOT NULL,
                            query_label TEXT        NOT NULL,
                            run_id      TEXT,
                            run_name    TEXT,
                            service     TEXT,
                            series      TEXT,
                            value       DOUBLE PRECISION,
                            promql      TEXT,
                            start_ms    BIGINT,
                            end_ms      BIGINT
                        );
                        """
                    ).format(sql.Identifier(schema), sql.Identifier(table))
                )
            except Exception as e:
                logger.warning("Не удалось создать таблицу %s.%s: %s", schema, table, e)

            if make_hypertable:
                try:
                    if chunk_interval:
                        cur.execute(
                            "SELECT create_hypertable(%s, 'time', if_not_exists => TRUE, chunk_time_interval => %s);",
                            (f"{schema}.{table}", chunk_interval),
                        )
                    else:
                        cur.execute(
                            "SELECT create_hypertable(%s, 'time', if_not_exists => TRUE);",
                            (f"{schema}.{table}",),
                        )
                except Exception as e:
                    # Если create_hypertable падает (например, расширение не включено/таблица уже hypertable) — продолжаем
                    logger.debug("create_hypertable skipped: %s", e)

            try:
                index_name = sql.Identifier(f"idx_{table}_run_time")
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (run_id, time);").format(
                        index_name,
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )
            except Exception as e:
                logger.debug("create index skipped: %s", e)
    finally:
        try:
            conn.autocommit = prev_autocommit
        except Exception:
            pass
    
    try:
        conn.commit()
    except Exception:
        # если автокоммит был включён, commit не требуется
        pass
    _ENSURED_TABLES.add(key)


# One row per run with rows in the metrics table: the run list and the run counter read it instead of
# aggregating the metrics hypertable. save_domain_labeled keeps it current in the transaction of the insert;
# loadlens_app.migrations.backfill_run_index fills it for older data and catches up on every start.
RUN_INDEX_TABLE = "run_index"

# Merges a new observation of a run into its row: the time span only grows, and an empty service or run id
# never overwrites a known one.
_RUN_INDEX_ON_CONFLICT = """
    ON CONFLICT (run_name) DO UPDATE SET
        start_time = LEAST(ri.start_time, EXCLUDED.start_time),
        end_time = GREATEST(ri.end_time, EXCLUDED.end_time),
        service = COALESCE(NULLIF(EXCLUDED.service, ''), ri.service),
        run_id = COALESCE(NULLIF(EXCLUDED.run_id, ''), ri.run_id),
        updated_at = now()
"""


def _run_index_ref(storage_cfg: Dict[str, object]) -> sql.Identifier:
    return sql.Identifier(str(storage_cfg.get("schema") or "public"), RUN_INDEX_TABLE)


def _ensure_run_index_table(conn, storage_cfg: Dict[str, object]) -> None:
    """Creates ``<schema>.run_index`` and its (service, end_time) index once per process.

    The table is small (one row per run), so this is cheap enough for the request path.
    """
    schema = str(storage_cfg.get("schema") or "public")
    key = (schema, RUN_INDEX_TABLE)
    if key in _ENSURED_TABLES:
        return
    prev_autocommit = getattr(conn, "autocommit", False)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            run_name    TEXT        PRIMARY KEY,
                            service     TEXT        NOT NULL DEFAULT '',
                            run_id      TEXT        NOT NULL DEFAULT '',
                            start_time  TIMESTAMPTZ NOT NULL,
                            end_time    TIMESTAMPTZ NOT NULL,
                            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                        """
                    ).format(_run_index_ref(storage_cfg))
                )
            except Exception as e:
                logger.warning("Не удалось создать таблицу %s.%s: %s", schema, RUN_INDEX_TABLE, e)
            try:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (service, end_time DESC);").format(
                        sql.Identifier(f"idx_{RUN_INDEX_TABLE}_service_end"), _run_index_ref(storage_cfg)
                    )
                )
            except Exception as e:
                logger.debug("create index skipped: %s", e)
    finally:
        try:
            conn.autocommit = prev_autocommit
        except Exception:
            pass
    _ENSURED_TABLES.add(key)


def upsert_run_index(
    conn,
    storage_cfg: Dict[str, object],
    run_name: Optional[str],
    service: Optional[str],
    run_id: Optional[str],
    start_time: datetime,
    end_time: datetime,
) -> bool:
    """Adds the run to ``run_index`` or widens its time span; returns False for a run without a name.

    Runs in the caller's transaction and does not commit.
    """
    name = str(run_name or "")
    if not name:
        return False
    statement = sql.SQL(
        "INSERT INTO {} AS ri (run_name, service, run_id, start_time, end_time) VALUES (%s, %s, %s, %s, %s)"
    ).format(_run_index_ref(storage_cfg)) + sql.SQL(_RUN_INDEX_ON_CONFLICT)
    with conn.cursor() as cur:
        cur.execute(statement, (name, str(service or ""), str(run_id or ""), start_time, end_time))
    return True


def refresh_run_index(conn, storage_cfg: Dict[str, object], since: Optional[datetime] = None) -> int:
    """Merges the runs of the metrics table into ``run_index``; returns the number of rows written.

    ``since`` limits the pass to metric rows newer than it (TimescaleDB then skips the older chunks);
    None aggregates the whole table. Runs in the caller's transaction and does not commit.
    """
    schema = str(storage_cfg.get("schema") or "public")
    metrics = sql.Identifier(schema, str(storage_cfg.get("table") or "metrics"))
    where = sql.SQL("WHERE run_name IS NOT NULL AND run_name <> ''")
    params: tuple = ()
    if since is not None:
        where = where + sql.SQL(' AND "time" > %s')
        params = (since,)
    statement = sql.SQL(
        "INSERT INTO {index} AS ri (run_name, service, run_id, start_time, end_time)"
        " SELECT run_name, COALESCE(MAX(service), ''), COALESCE(MAX(NULLIF(run_id, '')), ''),"
        ' MIN("time"), MAX("time") FROM {metrics} {where} GROUP BY run_name'
    ).format(index=_run_index_ref(storage_cfg), metrics=metrics, where=where) + sql.SQL(_RUN_INDEX_ON_CONFLICT)
    with conn.cursor() as cur:
        cur.execute(statement, params)
        return int(cur.rowcount or 0)


def _iter_records(
    long_df: pd.DataFrame,
    domain_key: str,
    query_label: str,
    run_meta: Dict[str, object],
    promql_text: str
) -> Iterable[tuple]:
    run_id = str(run_meta.get("run_id") or "")
    run_name = str(run_meta.get("run_name") or "")
    service = str(run_meta.get("service") or "")
    start_ms = int(run_meta.get("start_ms") or 0)
    end_ms = int(run_meta.get("end_ms") or 0)

    for row in long_df.itertuples(index=False):
        yield (
            row.time.to_pydatetime(),
            domain_key,
            query_label,
            run_id,
            run_name,
            service,
            row.series,
            float(row.value),
            promql_text,
            start_ms,
            end_ms,
        )


def save_domain_labeled(
    domain_key: str,
    domain_conf: Dict[str, object],
    labeled_dfs: List[Dict[str, object]],
    run_meta: Optional[Dict[str, object]],
    storage_cfg: Dict[str, object]
) -> None:
    """Сохраняет временные ряды домена в таблицу `metrics`.

    Параметры:
        domain_key (str): Имя домена (`jvm`, `kafka`, ...).
        domain_conf (dict): Конфигурация домена (`labels`, `promql_queries`).
        labeled_dfs (list[dict]): Список DataFrame c их подписями.
        run_meta (dict | None): Метаданные запуска (run_id, service и т.д.).
        storage_cfg (dict): Настройки TimescaleDB (`host`, `schema`, `table`...).

    Побочные эффекты:
        Выполняет INSERT в PostgreSQL/TimescaleDB.

    Исключения:
        Пробрасывает ошибки psycopg2 при недоступности БД.
    """
    if not storage_cfg:
        logger.warning("TimescaleDB конфигурация не задана, пропускаем сохранение домена %s", domain_key)
        return

    schema = storage_cfg.get("schema", "public")
    table = storage_cfg.get("table", "metrics")
    try:
        logger.info(
            "Timescale target: host=%s port=%s db=%s schema=%s table=%s",
            storage_cfg.get("host"), storage_cfg.get("port"), storage_cfg.get("dbname"), schema, table
        )
    except Exception:
        pass

    labels_cfg: List[str] = list(domain_conf.get("labels", [])) if isinstance(domain_conf, dict) else []
    promqls_cfg: List[str] = list(domain_conf.get("promql_queries", [])) if isinstance(domain_conf, dict) else []

    rm = run_meta or {}

    conn = _connect(storage_cfg)
    try:
        _ensure_schema_and_table(conn, storage_cfg, schema, table)
        _ensure_run_index_table(conn, storage_cfg)

        insert_sql = sql.SQL(
            """
            INSERT INTO {}.{} (
                time, domain, query_label, run_id, run_name, service,
                series, value, promql, start_ms, end_ms
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            );
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))

        total_rows = 0
        first_time = last_time = None
        with conn.cursor() as cur:
            for idx, item in enumerate(labeled_dfs):
                df = item.get("df")
                long_df = _df_to_long(df)
                if long_df.empty:
                    continue

                query_label = labels_cfg[idx] if idx < len(labels_cfg) else str(item.get("label") or f"q{idx}")
                promql_text = promqls_cfg[idx] if idx < len(promqls_cfg) else ""

                rows = list(_iter_records(long_df, domain_key, query_label, rm, promql_text))
                if not rows:
                    continue

                execute_batch(
                    cur,
                    insert_sql.as_string(cur),
                    rows,
                    page_size=int(storage_cfg.get("batch_size", 500))
                )
                batch_count = len(rows)
                total_rows += batch_count
                span_start, span_end = long_df["time"].min(), long_df["time"].max()
                first_time = span_start if first_time is None else min(first_time, span_start)
                last_time = span_end if last_time is None else max(last_time, span_end)
                logger.info(
                    "Timescale insert: domain=%s label=%s rows=%d", domain_key, query_label, batch_count
                )

            run_name = str(rm.get("run_name") or "")
            if total_rows and run_name:
                # The run list reads run_index; a failed update there must not lose the metrics themselves.
                cur.execute("SAVEPOINT run_index_upsert")
                try:
                    upsert_run_index(
                        conn, storage_cfg, run_name, rm.get("service"), rm.get("run_id"),
                        first_time.to_pydatetime(), last_time.to_pydatetime(),
                    )
                    cur.execute("RELEASE SAVEPOINT run_index_upsert")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT run_index_upsert")
                    logger.warning(
                        "run_index не обновлён для запуска %s: %s (пересборка: tools/check_run_index.py --rebuild)",
                        run_name, e,
                    )

        conn.commit()
        logger.info("Timescale insert total: domain=%s rows=%d", domain_key, total_rows)
    except Exception as e:
        conn.rollback()
        logger.error("Ошибка сохранения домена %s в TimescaleDB: %s", domain_key, e)
        raise
    finally:
        conn.close()


def _ensure_llm_reports_table(conn, storage_cfg: Dict[str, object]) -> None:
    schema = storage_cfg.get("schema", "public")
    table = storage_cfg.get("llm_table", "llm_reports")
    key = (schema, table)
    if key in _ENSURED_TABLES:
        return
    try:
        ensure_report_verdicts_table(conn, str(schema))
    except Exception as e:
        logger.warning("Не удалось подготовить таблицу ручных вердиктов: %s", e)
    prev_autocommit = getattr(conn, "autocommit", False)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            id          BIGSERIAL PRIMARY KEY,
                            created_at  TIMESTAMPTZ DEFAULT now(),
                            run_id      TEXT,
                            run_name    TEXT,
                            service     TEXT,
                            test_type   TEXT,
                            start_ms    BIGINT,
                            end_ms      BIGINT,
                            domain      TEXT NOT NULL,
                            text        TEXT,
                            parsed      JSONB,
                            scores      JSONB,
                            system_context JSONB
                        );
                        """
                    ).format(sql.Identifier(schema), sql.Identifier(table))
                )
            except Exception as e:
                logger.warning("Не удалось создать таблицу %s.%s: %s", schema, table, e)
            try:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (run_name, created_at DESC);").format(
                        sql.Identifier(f"idx_{table}_run_created"), sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            # Добавляем колонку verdict для стандартизированного вердикта финального домена
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS verdict TEXT;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            # Добавляем колонку test_type, если её ещё нет
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS test_type TEXT;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS sla_verdict TEXT;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS sla_details JSONB;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS system_context JSONB;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS context JSONB;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS project_area TEXT;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            # Load level of a final row (max_rps_from_parsed), so list queries need not parse JSONB per row.
            try:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS max_rps NUMERIC;").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
            # Final rows newest first, overall (dashboard "last run") and per service (baseline lookup).
            try:
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {}.{} (created_at DESC) WHERE domain = 'final';"
                    ).format(sql.Identifier(f"idx_{table}_final_created"), sql.Identifier(schema), sql.Identifier(table))
                )
            except Exception:
                pass
            try:
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {}.{} (service, created_at DESC) WHERE domain = 'final';"
                    ).format(
                        sql.Identifier(f"idx_{table}_final_service_created"), sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
    finally:
        try:
            conn.autocommit = prev_autocommit
        except Exception:
            pass
    _ENSURED_TABLES.add(key)


def _ensure_engineer_reports_table(conn, storage_cfg: Dict[str, object]) -> None:
    """Создаёт таблицу для итогов инженера (если отсутствует).
    Схема: id, created_at, run_id, run_name, service, content_html.
    """
    schema = storage_cfg.get("schema", "public")
    table = storage_cfg.get("engineer_table", "engineer_reports")
    key = (schema, table)
    if key in _ENSURED_TABLES:
        return
    prev_autocommit = getattr(conn, "autocommit", False)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            id           BIGSERIAL PRIMARY KEY,
                            created_at   TIMESTAMPTZ DEFAULT now(),
                            run_id       TEXT,
                            run_name     TEXT NOT NULL,
                            service      TEXT,
                            content_html TEXT
                        );
                        """
                    ).format(sql.Identifier(schema), sql.Identifier(table))
                )
            except Exception as e:
                logger.warning("Не удалось создать таблицу %s.%s (engineer): %s", schema, table, e)
            try:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (run_name, created_at DESC);").format(
                        sql.Identifier(f"idx_{table}_run_created"), sql.Identifier(schema), sql.Identifier(table)
                    )
                )
            except Exception:
                pass
    finally:
        try:
            conn.autocommit = prev_autocommit
        except Exception:
            pass
    _ENSURED_TABLES.add(key)


# ---- manual verdicts ------------------------------------------------------------------------------------

# Verdicts an engineer set by hand, with a reason. Every change is a row: the newest row of a run wins, and a row
# without a verdict returns the automatic one. llm_reports is never changed, so the analysis stays as it was built.
# New rows hold verdict codes (AI.verdicts); rows written before the codes hold Russian labels and are read by
# mapping them to codes. The API accepts a code or a legacy label.
REPORT_VERDICTS_TABLE = "report_verdicts"
MANUAL_VERDICTS = VERDICT_CODES
MANUAL_VERDICT_REASON_MAX = 1000
MANUAL_VERDICT_HISTORY_LIMIT = 50
_MANUAL_VERDICTS_READY: set[str] = set()


def _quoted_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _report_verdicts_ref(schema: str) -> sql.Identifier:
    return sql.Identifier(str(schema or "public"), REPORT_VERDICTS_TABLE)


def ensure_report_verdicts_table(conn, schema: str) -> None:
    """Creates ``<schema>.report_verdicts`` and its (run_name, created_at) index once per process."""
    schema = str(schema or "public")
    key = (schema, REPORT_VERDICTS_TABLE)
    if key in _ENSURED_TABLES:
        return
    created = False
    prev_autocommit = getattr(conn, "autocommit", False)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id               BIGSERIAL   PRIMARY KEY,
                            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                            run_name         TEXT        NOT NULL,
                            run_id           TEXT        NOT NULL DEFAULT '',
                            verdict          TEXT,
                            previous_verdict TEXT        NOT NULL DEFAULT '',
                            auto_verdict     TEXT        NOT NULL DEFAULT '',
                            reason           TEXT        NOT NULL DEFAULT '',
                            author           TEXT        NOT NULL DEFAULT ''
                        );
                        """
                    ).format(_report_verdicts_ref(schema))
                )
                created = True
            except Exception as e:
                logger.warning("Не удалось создать таблицу %s.%s: %s", schema, REPORT_VERDICTS_TABLE, e)
            try:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (run_name, created_at DESC, id DESC);").format(
                        sql.Identifier(f"idx_{REPORT_VERDICTS_TABLE}_run_created"), _report_verdicts_ref(schema)
                    )
                )
            except Exception as e:
                logger.debug("create index skipped: %s", e)
    finally:
        try:
            conn.autocommit = prev_autocommit
        except Exception:
            pass
    _ENSURED_TABLES.add(key)
    if created:
        _MANUAL_VERDICTS_READY.add(schema)


def manual_verdicts_ready(conn, schema: str) -> bool:
    """True when ``<schema>.report_verdicts`` exists: without the table no run has a manual verdict.

    A read-only probe that is safe inside a transaction; a found table is remembered for the process.
    """
    schema = str(schema or "public")
    if schema in _MANUAL_VERDICTS_READY:
        return True
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{_quoted_ident(schema)}.{REPORT_VERDICTS_TABLE}",))
        row = cur.fetchone()
    ready = bool(row and row[0])
    if ready:
        _MANUAL_VERDICTS_READY.add(schema)
    return ready


def manual_verdicts_cte(schema: str, ready: bool) -> str:
    """SQL of ``manual_verdicts AS (...)``: the current manual verdict of each run with its reason, author and
    time (columns run_name, verdict, reason, author, changed_at). Empty when the table does not exist yet.

    Readers join it as ``LEFT JOIN manual_verdicts mv ON mv.run_name = ...`` and take ``effective_verdict_sql``.
    """
    if not ready:
        return (
            "manual_verdicts AS (SELECT NULL::text AS run_name, NULL::text AS verdict, NULL::text AS reason,"
            " NULL::text AS author, NULL::timestamptz AS changed_at WHERE FALSE)"
        )
    table = f"{_quoted_ident(schema or 'public')}.{REPORT_VERDICTS_TABLE}"
    return (
        "manual_verdicts AS (SELECT run_name, verdict, reason, author, changed_at FROM ("
        "SELECT DISTINCT ON (run_name) run_name, verdict, reason, author, created_at AS changed_at"
        f" FROM {table} ORDER BY run_name, created_at DESC, id DESC) latest_manual WHERE verdict IS NOT NULL)"
    )


def effective_verdict_sql(alias: str = "", fallback: bool = True) -> str:
    """Code of the verdict a run shows: the manual one, else the SLA verdict, else the stored verdict of the
    final row; old labels map to codes and a run without any verdict is ``no_data``.

    ``fallback`` is kept for callers; NULL maps to ``no_data`` either way (``AI.verdicts.code_sql``).
    """
    prefix = f"{alias}." if alias else ""
    return code_sql(f"COALESCE(mv.verdict, {prefix}sla_verdict, {prefix}verdict)")


def _code_or_blank(value: Any) -> str:
    """Code of a stored verdict, '' for an empty one (previous/auto verdict of an old history row)."""
    return optional_code(value) or ""


def _manual_entry(row: Sequence[Any]) -> Dict[str, Any]:
    changed_at, verdict, previous, automatic, reason, author = row[:6]
    return {
        "verdict": optional_code(verdict),
        "previous_verdict": _code_or_blank(previous),
        "auto_verdict": _code_or_blank(automatic),
        "reason": str(reason or ""),
        "author": str(author or ""),
        "changed_at": changed_at.isoformat() if isinstance(changed_at, datetime) else (str(changed_at) if changed_at else None),
    }


def load_verdict_state(conn, schema: str, llm_table: str, run_name: str) -> Optional[Dict[str, Any]]:
    """Automatic and current verdict of a run with its manual changes (newest first), or None without a final report.

    ``manual`` is the change in force: None when the run shows its automatic verdict. Every verdict is a code.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT " + code_sql("COALESCE(sla_verdict, verdict)") + ", COALESCE(run_id, '') FROM {} "
                "WHERE run_name = %s AND domain = 'final' ORDER BY created_at DESC LIMIT 1"
            ).format(sql.Identifier(str(schema or "public"), llm_table)),
            (run_name,),
        )
        final = cur.fetchone()
    if not final:
        return None
    automatic = to_code(final[0])
    history: List[Dict[str, Any]] = []
    if manual_verdicts_ready(conn, schema):
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT created_at, verdict, previous_verdict, auto_verdict, reason, author FROM {} "
                    "WHERE run_name = %s ORDER BY created_at DESC, id DESC LIMIT %s"
                ).format(_report_verdicts_ref(schema)),
                (run_name, MANUAL_VERDICT_HISTORY_LIMIT),
            )
            history = [_manual_entry(row) for row in cur.fetchall()]
    manual = history[0] if history and history[0]["verdict"] else None
    return {
        "run_name": run_name,
        "run_id": str(final[1] or ""),
        "automatic": automatic,
        "verdict": manual["verdict"] if manual else automatic,
        "manual": manual,
        "history": history,
    }


def save_manual_verdict(
    conn,
    schema: str,
    llm_table: str,
    run_name: str,
    verdict: Optional[str],
    reason: str,
    author: str,
) -> Dict[str, Any]:
    """Sets the verdict of a run by hand (``verdict`` None returns the automatic one) and returns the new state.

    ``verdict`` is a code of ``MANUAL_VERDICTS`` or a legacy label; the row stores codes in all three verdict
    columns. Raises LookupError without a final report and ValueError for an unknown verdict, a missing
    or too long reason, or a change that changes nothing. Runs in the caller's transaction and does not commit.
    """
    reason = str(reason or "").strip()
    if verdict is not None:
        code = exact_code(verdict)
        if code is None:
            raise ValueError(t("server.verdict.unknown", allowed=", ".join(MANUAL_VERDICTS)))
        verdict = code
    if verdict is not None and not reason:
        raise ValueError(t("server.verdict.reason_required"))
    if len(reason) > MANUAL_VERDICT_REASON_MAX:
        raise ValueError(t("server.verdict.reason_too_long", max=MANUAL_VERDICT_REASON_MAX))
    ensure_report_verdicts_table(conn, schema)
    state = load_verdict_state(conn, schema, llm_table, run_name)
    if state is None:
        raise LookupError(t("server.report.final_not_found", run_name=run_name))
    if verdict is None and state["manual"] is None:
        raise ValueError(t("server.verdict.already_automatic"))
    if verdict is not None and verdict == state["verdict"]:
        raise ValueError(t("server.verdict.unchanged", verdict=verdict_label(verdict, get_locale())))
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "INSERT INTO {} (run_name, run_id, verdict, previous_verdict, auto_verdict, reason, author) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)"
            ).format(_report_verdicts_ref(schema)),
            (run_name, state["run_id"], verdict, state["verdict"], state["automatic"], reason, str(author or "")),
        )
    return load_verdict_state(conn, schema, llm_table, run_name) or state


BASELINE_MODE_PREVIOUS_SUCCESS = "previous_success"
BASELINE_MODE_PREVIOUS = "previous"
BASELINE_MODES = (BASELINE_MODE_PREVIOUS_SUCCESS, BASELINE_MODE_PREVIOUS)


@dataclass(frozen=True)
class BaselineRun:
    run_name: str
    service: str
    verdict: str
    created_at: Optional[datetime]


@dataclass(frozen=True)
class SeriesStats:
    mean: float
    p95: float
    max: float


RunMetricStats = Dict[str, Dict[str, Dict[str, SeriesStats]]]


def _test_type_filter_values(test_type: Optional[str]) -> list[str]:
    """Stored ``test_type`` values that belong to the same normalized type (lowercase)."""
    from AI.test_profile import TEST_TYPE_ALIASES, normalize_test_type

    normalized = normalize_test_type(test_type)
    if not normalized:
        return []
    values = {normalized}
    for alias in TEST_TYPE_ALIASES[normalized]:
        values.add(alias)
        values.add(alias.replace("_", " "))
    if normalized == "step":
        values.add("")
    return sorted(values)


def find_previous_run(
    conn,
    schema: str,
    llm_table: str,
    *,
    exclude_run_name: str,
    mode: str = BASELINE_MODE_PREVIOUS_SUCCESS,
    service: Optional[str] = None,
    services: Optional[Sequence[str]] = None,
    project_area: Optional[str] = None,
    before: Optional[datetime] = None,
    test_type: Optional[str] = None,
) -> Optional[BaselineRun]:
    """Latest earlier run of the same service (and project area, when set) to use as a baseline.

    ``mode=previous_success`` requires the effective verdict ``pass`` (a code or its old label);
    ``mode=previous`` takes any earlier run. ``services`` limits the lookup to the current project area.
    Rows saved before ``project_area`` existed stay eligible when their area is empty.
    ``test_type`` (normalized step/soak/spike/stress) limits the lookup to runs of the same
    type, matching any stored alias; step runs also accept rows saved without a type.
    ``BaselineRun.verdict`` is a code.
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f"Unknown baseline mode '{mode}', expected one of {BASELINE_MODES}")
    manual_cte = sql.SQL(manual_verdicts_cte(schema, manual_verdicts_ready(conn, schema)))
    success_filter = (
        sql.SQL(f"AND {effective_verdict_sql('r', fallback=False)} = '{PASS}'")
        if mode == BASELINE_MODE_PREVIOUS_SUCCESS
        else sql.SQL("")
    )
    columns = sql.SQL(f"r.run_name, r.service, {effective_verdict_sql('r')}, r.created_at")
    table_ref = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(llm_table))
    area = str(project_area or "").strip()
    area_sql = sql.SQL("AND (COALESCE(r.project_area, '') = '' OR r.project_area = %s)") if area else sql.SQL("")
    type_values = _test_type_filter_values(test_type)
    type_sql = sql.SQL("AND LOWER(TRIM(COALESCE(r.test_type, ''))) = ANY(%s)") if type_values else sql.SQL("")
    tail: tuple = ((type_values,) if type_values else ()) + ((area,) if area else ())
    allowed = [str(item).strip() for item in (services or []) if str(item).strip()]
    if service and service not in allowed:
        allowed.append(service)
    if service is None or before is None:
        query = sql.SQL(
            """
            WITH current AS (
              SELECT service, created_at FROM {table}
              WHERE run_name = %s AND domain = 'final'
              ORDER BY created_at DESC LIMIT 1
            ), {manual_cte}
            SELECT {columns}
            FROM {table} r CROSS JOIN current c
            LEFT JOIN manual_verdicts mv ON mv.run_name = r.run_name
            WHERE r.domain = 'final' AND r.service = c.service AND r.run_name <> %s AND r.created_at < c.created_at
            {success_filter} {type_sql} {area_sql}
            ORDER BY r.created_at DESC LIMIT 1
            """
        ).format(
            table=table_ref, manual_cte=manual_cte, columns=columns, success_filter=success_filter,
            type_sql=type_sql, area_sql=area_sql,
        )
        params: tuple = (exclude_run_name, exclude_run_name) + tail
    elif len(allowed) > 1:
        query = sql.SQL(
            """
            WITH {manual_cte}
            SELECT {columns}
            FROM {table} r
            LEFT JOIN manual_verdicts mv ON mv.run_name = r.run_name
            WHERE r.domain = 'final' AND r.service = ANY(%s) AND r.run_name <> %s AND r.created_at < %s
            {success_filter} {type_sql} {area_sql}
            ORDER BY r.created_at DESC LIMIT 1
            """
        ).format(
            table=table_ref, manual_cte=manual_cte, columns=columns, success_filter=success_filter,
            type_sql=type_sql, area_sql=area_sql,
        )
        params = (allowed, exclude_run_name, before) + tail
    else:
        query = sql.SQL(
            """
            WITH {manual_cte}
            SELECT {columns}
            FROM {table} r
            LEFT JOIN manual_verdicts mv ON mv.run_name = r.run_name
            WHERE r.domain = 'final' AND r.service = %s AND r.run_name <> %s AND r.created_at < %s
            {success_filter} {type_sql} {area_sql}
            ORDER BY r.created_at DESC LIMIT 1
            """
        ).format(
            table=table_ref, manual_cte=manual_cte, columns=columns, success_filter=success_filter,
            type_sql=type_sql, area_sql=area_sql,
        )
        params = (service, exclude_run_name, before) + tail
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    if not row:
        return None
    return BaselineRun(
        run_name=str(row[0]),
        service=str(row[1] or ""),
        verdict=to_code(row[2]),
        created_at=row[3],
    )


def load_run_metric_stats(conn, schema: str, metrics_table: str, run_name: str) -> RunMetricStats:
    """Per-series mean / p95 / max of a stored run, grouped as domain → query_label → series."""
    query = sql.SQL(
        """
        SELECT domain, query_label, series,
               AVG(value), percentile_cont(0.95) WITHIN GROUP (ORDER BY value), MAX(value)
        FROM {}.{}
        WHERE run_name = %s AND value IS NOT NULL
        GROUP BY domain, query_label, series
        """
    ).format(sql.Identifier(schema), sql.Identifier(metrics_table))
    with conn.cursor() as cur:
        cur.execute(query, (run_name,))
        rows = cur.fetchall()
    stats: RunMetricStats = {}
    for domain, label, series, mean, p95, max_value in rows:
        stats.setdefault(str(domain), {}).setdefault(str(label), {})[str(series)] = SeriesStats(
            mean=float(mean), p95=float(p95), max=float(max_value),
        )
    return stats


@dataclass(frozen=True)
class StoredFinalReport:
    run_id: str
    run_name: str
    service: str
    test_type: str
    project_area: str
    start_ms: Optional[int]
    end_ms: Optional[int]
    context: Dict[str, Any]
    sla_details: Dict[str, Any] = field(default_factory=dict)


def _millis(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_final_report(conn, schema: str, llm_table: str, run_id: str) -> Optional[StoredFinalReport]:
    """Latest final row of a run by id with its stored LLM context."""
    query = sql.SQL(
        """
        SELECT COALESCE(run_id, ''), run_name, COALESCE(service, ''), COALESCE(test_type, ''),
               COALESCE(project_area, ''), start_ms, end_ms, context, sla_details
        FROM {}.{}
        WHERE run_id = %s AND domain = 'final'
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).format(sql.Identifier(schema), sql.Identifier(llm_table))
    with conn.cursor() as cur:
        cur.execute(query, (run_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return StoredFinalReport(
        run_id=str(row[0] or ""),
        run_name=str(row[1] or ""),
        service=str(row[2] or ""),
        test_type=str(row[3] or ""),
        project_area=str(row[4] or ""),
        start_ms=_millis(row[5]),
        end_ms=_millis(row[6]),
        context=_coerce_json_object(row[7]) or {},
        sla_details=_coerce_json_object(row[8]) or {},
    )


def _series_labels(raw: Any) -> Optional[List[str]]:
    """``lt_series_labels`` from JSON: a list, or ``None`` when the report predates the key."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(raw, list):
        return None
    return [str(item) for item in raw]


@dataclass(frozen=True)
class FinalReportRow:
    run_id: str
    run_name: str
    service: str
    test_type: str
    project_area: str
    created_at: Optional[datetime]
    verdict: str
    lt_series_labels: Optional[List[str]]


def list_final_reports(conn, schema: str, llm_table: str, services: Sequence[str]) -> List[FinalReportRow]:
    """Latest final row per run with an id, newest first; ``services`` restricts to a project area."""
    service_sql = sql.SQL(" AND service = ANY(%s)") if services else sql.SQL("")
    manual_cte = sql.SQL(manual_verdicts_cte(schema, manual_verdicts_ready(conn, schema)))
    query = sql.SQL(
        """
        WITH {manual_cte}
        SELECT latest.run_id, latest.run_name, latest.service, latest.test_type, latest.project_area,
               latest.created_at, {verdict} AS verdict, latest.lt_series_labels
        FROM (
          SELECT COALESCE(run_id, '') AS run_id, run_name, COALESCE(service, '') AS service,
                 COALESCE(test_type, '') AS test_type, COALESCE(project_area, '') AS project_area,
                 created_at, sla_verdict, verdict,
                 context -> 'lt_series_labels' AS lt_series_labels,
                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
          FROM {table}
          WHERE domain = 'final'{service_sql}
        ) latest
        LEFT JOIN manual_verdicts mv ON mv.run_name = latest.run_name
        WHERE latest.rn = 1 AND latest.run_id <> ''
        ORDER BY latest.created_at DESC
        """
    ).format(
        manual_cte=manual_cte, verdict=sql.SQL(effective_verdict_sql("latest")),
        table=sql.Identifier(schema, llm_table), service_sql=service_sql,
    )
    params: tuple = (list(services),) if services else ()
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [
        FinalReportRow(
            run_id=str(row[0] or ""),
            run_name=str(row[1] or ""),
            service=str(row[2] or ""),
            test_type=str(row[3] or ""),
            project_area=str(row[4] or ""),
            created_at=row[5] if isinstance(row[5], datetime) else None,
            verdict=to_code(row[6]),
            lt_series_labels=_series_labels(row[7]),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class StoredFinding:
    """One finding of a stored domain analysis; ``ref`` is ``domain:id``."""

    ref: str
    domain: str
    severity: str
    component: str
    start_time: str
    end_time: str
    summary: str
    evidence_summary: str
    verification: str


@dataclass(frozen=True)
class RunAnalyses:
    findings: List[StoredFinding]
    system_context: Dict[str, Any]


def _stored_finding(domain: str, raw: Dict[str, Any], position: int) -> StoredFinding:
    verification = raw.get("verification") if isinstance(raw.get("verification"), dict) else {}
    finding_id = str(raw.get("id") or f"f{position + 1}")
    return StoredFinding(
        ref=f"{domain}:{finding_id}",
        domain=domain,
        severity=str(raw.get("severity") or ""),
        component=str(raw.get("component") or ""),
        start_time=str(raw.get("start_time") or ""),
        end_time=str(raw.get("end_time") or ""),
        summary=str(raw.get("summary") or ""),
        evidence_summary=str(raw.get("evidence_summary") or ""),
        verification=str(verification.get("status") or ""),
    )


def load_run_analyses(conn, schema: str, llm_table: str, run_id: str) -> RunAnalyses:
    """Findings of the latest domain analyses of a run (final and engineer rows excluded) and its system context."""
    query = sql.SQL(
        """
        SELECT DISTINCT ON (domain) domain, parsed, system_context
        FROM {}
        WHERE run_id = %s AND domain NOT IN ('final', 'engineer')
        ORDER BY domain, created_at DESC
        """
    ).format(sql.Identifier(schema, llm_table))
    with conn.cursor() as cur:
        cur.execute(query, (run_id,))
        rows = cur.fetchall()
    findings: List[StoredFinding] = []
    system_context: Dict[str, Any] = {}
    for domain, parsed, snapshot in rows:
        raw_findings = (_coerce_json_object(parsed) or {}).get("findings")
        for position, raw in enumerate(raw_findings if isinstance(raw_findings, list) else []):
            if isinstance(raw, dict):
                findings.append(_stored_finding(str(domain), raw, position))
        system_context = system_context or (_coerce_json_object(snapshot) or {})
    return RunAnalyses(findings=findings, system_context=system_context)


@dataclass(frozen=True)
class StoredExplanation:
    run_id: str
    inputs_hash: str
    payload: Dict[str, Any]
    created_at: Optional[datetime]


def _ensure_explanations_table(conn, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                run_id      TEXT        PRIMARY KEY,
                inputs_hash TEXT        NOT NULL,
                payload     JSONB       NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        ).format(sql.Identifier(schema, table)))
    conn.commit()


def load_forecast_explanation(conn, schema: str, table: str, run_id: str) -> Optional[StoredExplanation]:
    """Cached LLM explanation of the forecast limit of a run."""
    _ensure_explanations_table(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT run_id, inputs_hash, payload, created_at FROM {} WHERE run_id = %s").format(sql.Identifier(schema, table)),
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return StoredExplanation(
        run_id=str(row[0]),
        inputs_hash=str(row[1]),
        payload=_coerce_json_object(row[2]) or {},
        created_at=row[3] if isinstance(row[3], datetime) else None,
    )


def save_forecast_explanation(conn, schema: str, table: str, run_id: str, inputs_hash: str, payload: Dict[str, Any]) -> None:
    """Replaces the cached explanation of a run."""
    _ensure_explanations_table(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {} (run_id, inputs_hash, payload, created_at) VALUES (%s, %s, %s, now())
                ON CONFLICT (run_id) DO UPDATE SET
                    inputs_hash = EXCLUDED.inputs_hash, payload = EXCLUDED.payload, created_at = now()
                """
            ).format(sql.Identifier(schema, table)),
            (run_id, inputs_hash, Json(payload)),
        )
    conn.commit()


def load_run_frames(
    conn,
    schema: str,
    metrics_table: str,
    run_id: str,
    domain: str,
    labels: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Stored series of a run: one frame per query label, series as columns, UTC index."""
    wanted = [str(label) for label in labels if str(label or "").strip()]
    if not wanted:
        return {}
    query = sql.SQL(
        """
        SELECT query_label, series, time, value
        FROM {}.{}
        WHERE run_id = %s AND domain = %s AND query_label = ANY(%s) AND value IS NOT NULL
        ORDER BY time
        """
    ).format(sql.Identifier(schema), sql.Identifier(metrics_table))
    with conn.cursor() as cur:
        cur.execute(query, (run_id, domain, wanted))
        rows = cur.fetchall()
    if not rows:
        return {}
    frame = pd.DataFrame(rows, columns=["query_label", "series", "time", "value"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frames: Dict[str, pd.DataFrame] = {}
    for label, group in frame.groupby("query_label", sort=False):
        pivoted = group.pivot_table(index="time", columns="series", values="value", aggfunc="mean").sort_index()
        pivoted.columns = [str(column) for column in pivoted.columns]
        frames[str(label)] = pivoted
    return frames


# Final-analysis fields with the run's load level, in priority order: the peak of step/stress runs,
# the held RPS of soak runs, the spike level of spike runs. One rule for Python and SQL below.
MAX_RPS_FIELDS = (
    ("peak_performance", "max_rps"),
    ("stability_under_load", "actual_rps"),
    ("spike_analysis", "max_spike_rps"),
)
# A plain decimal number; the exponent is bounded so the numeric cast cannot overflow. The LLM sometimes
# writes text such as «~1200», which is not a load level and is skipped in favour of the next field.
MAX_RPS_NUMBER_PATTERN = "^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]{1,3})?$"
_MAX_RPS_NUMBER_RE = re.compile(MAX_RPS_NUMBER_PATTERN)


def _max_rps_sql_literal(value: str) -> str:
    """SQL text literal for a trusted constant; ``%`` is doubled for psycopg2 parameter binding."""
    return "'" + str(value).replace("'", "''").replace("%", "%%") + "'"


def _max_rps_json_text(value: Any) -> Optional[str]:
    """Text of a scalar as PostgreSQL ``->>`` prints it, trimmed of spaces like ``TRIM``.

    None for JSON null, booleans, objects and arrays: their text is never a plain number.
    """
    if isinstance(value, str):
        return value.strip(" ")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr of a finite float is a plain or exponent literal; nan/inf (impossible in jsonb) do not match.
        return repr(value)
    return None


def max_rps_from_parsed(parsed: Any) -> Optional[float]:
    """Load level of a parsed final analysis: the first ``MAX_RPS_FIELDS`` value that is a plain number.

    The same rule as ``max_rps_sql``: a field whose text is not a plain number («~1200», «1,2 тыс.»)
    is skipped; the first plain number wins. A number beyond the float range gives None (JSON has no
    Infinity), as the API does for the value the SQL rule returns.
    """
    if not isinstance(parsed, dict):
        return None
    for section, field_name in MAX_RPS_FIELDS:
        block = parsed.get(section)
        if not isinstance(block, dict):
            continue
        text = _max_rps_json_text(block.get(field_name))
        if text is None or not _MAX_RPS_NUMBER_RE.fullmatch(text):
            continue
        number = float(text)
        return number if math.isfinite(number) else None
    return None


def max_rps_sql(parsed_column: str = "parsed") -> str:
    """SQL twin of ``max_rps_from_parsed`` over a JSONB column: a NUMERIC or NULL.

    ``parsed_column`` is a trusted column reference (``parsed``, ``f.parsed``); only constants are inlined.
    """
    parts = []
    for section, field_name in MAX_RPS_FIELDS:
        text = f"TRIM({parsed_column}->{_max_rps_sql_literal(section)}->>{_max_rps_sql_literal(field_name)})"
        parts.append(
            f"CASE WHEN {text} ~ {_max_rps_sql_literal(MAX_RPS_NUMBER_PATTERN)} THEN ({text})::numeric END"
        )
    return f"COALESCE({', '.join(parts)})"


def max_rps_select_sql(parsed_column: str = "parsed") -> str:
    """The stored ``max_rps`` of a final row, else ``max_rps_sql`` over ``parsed_column``.

    ``COALESCE`` evaluates the parse only when the column is NULL, so a filled row never detoasts
    ``parsed``; rows saved before the column existed (or not yet backfilled) still get their value.
    The column takes the qualifier of ``parsed_column`` (``f.parsed`` -> ``f.max_rps``).
    """
    qualifier = parsed_column.rpartition(".")[0]
    column = f"{qualifier}.max_rps" if qualifier else "max_rps"
    return f"COALESCE({column}, {max_rps_sql(parsed_column)})"


def _coerce_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return None


def _finite_json(value: Any) -> Any:
    """Replaces NaN and ±Infinity with None: PostgreSQL rejects these tokens in json and jsonb."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def save_llm_results(
    results: Dict[str, object],
    run_meta: Dict[str, object],
    storage_cfg: Dict[str, object]
) -> None:
    """Сохраняет текст/parsed/scores по доменам и финалу в таблицу `llm_reports`.

    Параметры:
        results (dict): Объединённый ответ `uploadFromLLM`.
        run_meta (dict): Метаданные запуска.
        storage_cfg (dict): Параметры TimescaleDB.

    Побочные эффекты:
        Пишет в таблицу `llm_reports` (создаёт её при необходимости).
    """
    if not storage_cfg:
        logger.warning("TimescaleDB конфигурация не задана, пропускаю сохранение LLM результатов")
        return
    schema = storage_cfg.get("schema", "public")
    table = storage_cfg.get("llm_table", "llm_reports")
    conn = _connect(storage_cfg)
    try:
        _ensure_llm_reports_table(conn, storage_cfg)
        insert_sql = sql.SQL(
            """
            INSERT INTO {}.{} (
                run_id, run_name, service, test_type, start_ms, end_ms, domain,
                text, parsed, scores, context, system_context, verdict, sla_verdict, sla_details, project_area,
                max_rps
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))

        run_id = str((run_meta or {}).get("run_id") or "")
        run_name = str((run_meta or {}).get("run_name") or "")
        service = str((run_meta or {}).get("service") or "")
        test_type = str((run_meta or {}).get("test_type") or "")
        project_area = str((run_meta or {}).get("project_area") or "").strip() or None
        start_ms = int((run_meta or {}).get("start_ms") or 0)
        end_ms = int((run_meta or {}).get("end_ms") or 0)

        rows = []
        scores_all = (results.get("scores", {}) or {})
        contexts_all = results.get("contexts") if isinstance(results.get("contexts"), dict) else {}
        system_context_snapshot = _finite_json(results.get("system_context")) if isinstance(results.get("system_context"), dict) else None
        # Verdicts are stored as codes (AI.verdicts); parsed analyses get codes in ``verdict`` as well.
        run_sla_verdict = optional_code(results.get("sla_verdict")) if results.get("sla_checks") else None
        load_step_table = results.get("load_step_table") if isinstance(results.get("load_step_table"), dict) else None
        run_sla_details = None
        if results.get("sla_checks") or load_step_table:
            run_sla_details = {
                "checks": results.get("sla_checks") or [],
                "summary": results.get("sla_summary") or "",
            }
            if load_step_table:
                run_sla_details["load_step_table"] = load_step_table
            if isinstance(results.get("sla_window"), dict):
                run_sla_details["stable_window"] = results.get("sla_window")
            # Step tests with a target step: the step the verdict was decided on and what the run showed above it.
            if isinstance(results.get("sla_target_window"), dict):
                run_sla_details["target_window"] = results.get("sla_target_window")
            if isinstance(results.get("sla_headroom"), dict):
                run_sla_details["headroom"] = results.get("sla_headroom")
            if results.get("report_language"):
                # Readers of the run row (forecast, SLA block) find the report language without the final JSON.
                run_sla_details["report_language"] = results.get("report_language")
            run_sla_details = _finite_json(run_sla_details)
        domains = [
            ("jvm", results.get("jvm"), results.get("jvm_parsed"), scores_all.get("jvm")),
            ("database", results.get("database"), results.get("database_parsed"), scores_all.get("database")),
            ("kafka", results.get("kafka"), results.get("kafka_parsed"), scores_all.get("kafka")),
            ("microservices", results.get("ms"), results.get("ms_parsed"), scores_all.get("microservices")),
            ("hard_resources", results.get("hard_resources"), results.get("hard_resources_parsed"), scores_all.get("hard_resources")),
            ("final", results.get("final"), results.get("final_parsed"), scores_all.get("final")),
        ]
        if "lt_framework" in results:
            domains.insert(-1, ("lt_framework", results.get("lt_framework"), results.get("lt_framework_parsed"), scores_all.get("lt_framework")))
        if "lt_generators" in results:
            domains.insert(-1, ("lt_generators", results.get("lt_generators"), results.get("lt_generators_parsed"), scores_all.get("lt_generators")))
        if "application_logs" in results:
            domains.insert(-1, ("application_logs", results.get("application_logs"), results.get("application_logs_parsed"), scores_all.get("application_logs")))

        for domain, text_val, parsed_val, scores_val in domains:
            raw_text = str(text_val) if text_val is not None else None
            if parsed_val is None and text_val:
                try:
                    coerced = parse_llm_analysis_strict(raw_text or "")
                    if coerced is not None:
                        parsed_val = coerced.dict()
                except Exception:
                    pass

            verdict_std = None
            if isinstance(parsed_val, dict):
                parsed_val = normalize_report(parsed_val)
                verdict_std = to_code(parsed_val.get("verdict"))
            parsed_val = _finite_json(parsed_val)

            text_str = raw_text
            raw_json_like = bool((raw_text or "").strip().startswith("{") or (raw_text or "").strip().startswith("```json"))
            if parsed_val is not None and raw_json_like:
                try:
                    text_str = json.dumps(parsed_val, ensure_ascii=False, indent=2)
                except Exception:
                    text_str = raw_text
            parsed_json = Json(parsed_val) if parsed_val is not None else None
            scores_json = Json(_finite_json(scores_val)) if scores_val is not None else None
            context_obj = _finite_json(_coerce_json_object(contexts_all.get(domain)))
            context_json = Json(context_obj) if context_obj is not None else None
            system_context_json = Json(system_context_snapshot) if system_context_snapshot is not None else None
            is_final = (domain == "final")
            sla_v = run_sla_verdict if is_final else None
            sla_d = Json(run_sla_details) if (is_final and run_sla_details) else None
            max_rps = max_rps_from_parsed(parsed_val) if is_final else None

            rows.append(
                (
                    run_id, run_name, service, test_type, start_ms, end_ms, domain,
                    text_str, parsed_json, scores_json, context_json, system_context_json,
                    verdict_std, sla_v, sla_d, project_area, max_rps,
                )
            )

        with conn.cursor() as cur:
            execute_batch(cur, insert_sql.as_string(cur), rows, page_size=100)
        conn.commit()
        logger.info("Сохранил LLM результаты в %s.%s: run_name=%s", schema, table, run_name)
    except Exception as e:
        conn.rollback()
        logger.error("Ошибка сохранения LLM результатов: %s", e)
        raise
    finally:
        conn.close()


def _ensure_llm_feedback_table(conn, storage_cfg: Dict[str, object]) -> None:
    """Creates the engineer labeling table used by the report UI and eval harness."""
    schema = storage_cfg.get("schema", "public")
    table = storage_cfg.get("llm_feedback_table", "llm_feedback")
    key = (schema, table)
    if key in _ENSURED_TABLES:
        return
    prev_autocommit = getattr(conn, "autocommit", False)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            id          BIGSERIAL PRIMARY KEY,
                            created_at  TIMESTAMPTZ DEFAULT now(),
                            run_name    TEXT NOT NULL,
                            domain      TEXT NOT NULL,
                            target      TEXT NOT NULL,
                            finding_id  TEXT NOT NULL DEFAULT '',
                            vote        TEXT NOT NULL,
                            comment     TEXT
                        );
                        """
                    ).format(sql.Identifier(schema), sql.Identifier(table))
                )
            except Exception as e:
                logger.warning("Не удалось создать таблицу %s.%s (feedback): %s", schema, table, e)
            try:
                cur.execute(
                    sql.SQL(
                        "CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{} (run_name, domain, target, finding_id);"
                    ).format(
                        sql.Identifier(f"idx_{table}_unique_target"),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )
            except Exception:
                pass
            try:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (run_name, created_at DESC);").format(
                        sql.Identifier(f"idx_{table}_run_created"),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )
            except Exception:
                pass
    finally:
        try:
            conn.autocommit = prev_autocommit
        except Exception:
            pass
    _ENSURED_TABLES.add(key)


FEEDBACK_TARGETS = ("verdict", "finding")
FEEDBACK_VOTES = ("agree", "disagree")


def list_llm_feedback(conn, storage_cfg: Dict[str, object], run_name: str) -> List[Dict[str, Any]]:
    """Returns labeling rows for one run, newest first."""
    schema = str(storage_cfg.get("schema", "public"))
    table = str(storage_cfg.get("llm_feedback_table", "llm_feedback"))
    _ensure_llm_feedback_table(conn, storage_cfg)
    query = sql.SQL(
        """
        SELECT run_name, domain, target, finding_id, vote, comment, created_at
        FROM {}.{}
        WHERE run_name = %s
        ORDER BY created_at DESC, domain, target, finding_id
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))
    with conn.cursor() as cur:
        cur.execute(query, (run_name,))
        rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for run, domain, target, finding_id, vote, comment, created_at in rows:
        out.append({
            "run_name": run,
            "domain": domain,
            "target": target,
            "finding_id": finding_id or "",
            "vote": vote,
            "comment": comment or "",
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
        })
    return out


def upsert_llm_feedback(conn, storage_cfg: Dict[str, object], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stores the latest anonymous vote for a verdict or finding. Last write wins."""
    schema = str(storage_cfg.get("schema", "public"))
    table = str(storage_cfg.get("llm_feedback_table", "llm_feedback"))
    _ensure_llm_feedback_table(conn, storage_cfg)
    run_name = str(payload.get("run_name") or "").strip()
    domain = str(payload.get("domain") or "").strip()
    target = str(payload.get("target") or "").strip()
    vote = str(payload.get("vote") or "").strip()
    finding_id = str(payload.get("finding_id") or "").strip()
    comment = str(payload.get("comment") or "").strip()
    if not run_name:
        raise ValueError(t("api.required.run_name"))
    if not domain:
        raise ValueError(t("server.feedback.domain_required"))
    if target not in FEEDBACK_TARGETS:
        raise ValueError(t("server.feedback.bad_target"))
    if vote not in FEEDBACK_VOTES:
        raise ValueError(t("server.feedback.bad_vote"))
    if target == "verdict":
        finding_id = ""
    elif not finding_id:
        raise ValueError(t("server.feedback.finding_required"))
    if len(comment) > 2000:
        raise ValueError(t("server.feedback.comment_too_long"))
    query = sql.SQL(
        """
        INSERT INTO {}.{} (run_name, domain, target, finding_id, vote, comment, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (run_name, domain, target, finding_id)
        DO UPDATE SET vote = EXCLUDED.vote, comment = EXCLUDED.comment, created_at = now()
        RETURNING run_name, domain, target, finding_id, vote, comment, created_at
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))
    with conn.cursor() as cur:
        cur.execute(query, (run_name, domain, target, finding_id, vote, comment))
        row = cur.fetchone()
    conn.commit()
    created_at = row[6] if row else None
    return {
        "run_name": row[0] if row else run_name,
        "domain": row[1] if row else domain,
        "target": row[2] if row else target,
        "finding_id": (row[3] or "") if row else finding_id,
        "vote": row[4] if row else vote,
        "comment": (row[5] or "") if row else comment,
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
    }

