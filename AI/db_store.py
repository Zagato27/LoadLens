import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_batch

from AI.scoring import parse_llm_analysis_strict


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
    dsn = storage_cfg.get("dsn")
    if isinstance(dsn, str) and dsn.strip():
        return psycopg2.connect(dsn)

    params = {
        "host": storage_cfg.get("host", "localhost"),
        "port": int(storage_cfg.get("port", 5432)),
        "dbname": storage_cfg.get("dbname", "loadtesting"),
        "user": storage_cfg.get("user"),
        "password": storage_cfg.get("password"),
        "sslmode": storage_cfg.get("sslmode", "prefer"),
    }
    return psycopg2.connect(**params)


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
                logger.info(
                    "Timescale insert: domain=%s label=%s rows=%d", domain_key, query_label, batch_count
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


def _standardize_verdict(raw: Optional[str]) -> Optional[str]:
    """Приводит произвольный текст вердикта к одному из стандартных вариантов.
    Варианты: "Успешно", "Есть риски", "Провал", "Недостаточно данных".
    """
    if not isinstance(raw, str) or not raw.strip():
        return "Недостаточно данных"
    s = raw.strip().lower()
    try:
        # Недостаточно данных
        if any(k in s for k in ["insufficient", "нет данных", "недостаточно", "no data", "unknown", "n/a"]):
            return "Недостаточно данных"
        # Провал/критично
        if any(k in s for k in ["fail", "failed", "провал", "критич", "неудовлет", "red", "severe"]):
            return "Провал"
        # Есть риски/деградация/предупреждения
        if any(k in s for k in ["warn", "risk", "risks", "рис", "замеч", "degrad", "degraded", "под вопрос", "нестабиль"]):
            return "Есть риски"
        # Успешно/норма/стабильно
        if any(k in s for k in ["ok", "усп", "норма", "стаб", "успешно", "green", "success", "passed"]):
            return "Успешно"
    except Exception:
        pass
    # По умолчанию не рискуем — считаем как недостаточно данных
    return "Недостаточно данных"


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
) -> Optional[BaselineRun]:
    """Latest earlier run of the same service (and project area, when set) to use as a baseline.

    ``mode=previous_success`` requires the effective verdict «Успешно»; ``mode=previous``
    takes any earlier run. ``services`` limits the lookup to the current project area.
    Rows saved before ``project_area`` existed stay eligible when their area is empty.
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f"Unknown baseline mode '{mode}', expected one of {BASELINE_MODES}")
    success_filter = sql.SQL("AND COALESCE(r.sla_verdict, r.verdict) = 'Успешно'") if mode == BASELINE_MODE_PREVIOUS_SUCCESS else sql.SQL("")
    columns = sql.SQL("r.run_name, r.service, COALESCE(r.sla_verdict, r.verdict, 'Недостаточно данных'), r.created_at")
    table_ref = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(llm_table))
    area = str(project_area or "").strip()
    area_sql = sql.SQL("AND (COALESCE(r.project_area, '') = '' OR r.project_area = %s)") if area else sql.SQL("")
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
            )
            SELECT {columns}
            FROM {table} r, current c
            WHERE r.domain = 'final' AND r.service = c.service AND r.run_name <> %s AND r.created_at < c.created_at
            {success_filter} {area_sql}
            ORDER BY r.created_at DESC LIMIT 1
            """
        ).format(table=table_ref, columns=columns, success_filter=success_filter, area_sql=area_sql)
        params: tuple = (exclude_run_name, exclude_run_name) + ((area,) if area else ())
    elif len(allowed) > 1:
        query = sql.SQL(
            """
            SELECT {columns}
            FROM {table} r
            WHERE r.domain = 'final' AND r.service = ANY(%s) AND r.run_name <> %s AND r.created_at < %s
            {success_filter} {area_sql}
            ORDER BY r.created_at DESC LIMIT 1
            """
        ).format(table=table_ref, columns=columns, success_filter=success_filter, area_sql=area_sql)
        params = (allowed, exclude_run_name, before) + ((area,) if area else ())
    else:
        query = sql.SQL(
            """
            SELECT {columns}
            FROM {table} r
            WHERE r.domain = 'final' AND r.service = %s AND r.run_name <> %s AND r.created_at < %s
            {success_filter} {area_sql}
            ORDER BY r.created_at DESC LIMIT 1
            """
        ).format(table=table_ref, columns=columns, success_filter=success_filter, area_sql=area_sql)
        params = (service, exclude_run_name, before) + ((area,) if area else ())
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    if not row:
        return None
    return BaselineRun(
        run_name=str(row[0]),
        service=str(row[1] or ""),
        verdict=str(row[2] or "Недостаточно данных"),
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
    query = sql.SQL(
        """
        SELECT run_id, run_name, service, test_type, project_area, created_at, verdict, lt_series_labels
        FROM (
          SELECT COALESCE(run_id, '') AS run_id, run_name, COALESCE(service, '') AS service,
                 COALESCE(test_type, '') AS test_type, COALESCE(project_area, '') AS project_area,
                 created_at, COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                 context -> 'lt_series_labels' AS lt_series_labels,
                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
          FROM {table}
          WHERE domain = 'final'{service_sql}
        ) latest
        WHERE rn = 1 AND run_id <> ''
        ORDER BY created_at DESC
        """
    ).format(table=sql.Identifier(schema, llm_table), service_sql=service_sql)
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
            verdict=str(row[6] or "Недостаточно данных"),
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
                text, parsed, scores, context, system_context, verdict, sla_verdict, sla_details, project_area
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
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
        system_context_snapshot = results.get("system_context") if isinstance(results.get("system_context"), dict) else None
        run_sla_verdict = results.get("sla_verdict") if results.get("sla_checks") else None
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
            try:
                if isinstance(parsed_val, dict):
                    verdict_std = _standardize_verdict(parsed_val.get("verdict"))
            except Exception:
                pass

            text_str = raw_text
            raw_json_like = bool((raw_text or "").strip().startswith("{") or (raw_text or "").strip().startswith("```json"))
            if parsed_val is not None and raw_json_like:
                try:
                    text_str = json.dumps(parsed_val, ensure_ascii=False, indent=2)
                except Exception:
                    text_str = raw_text
            parsed_json = Json(parsed_val) if parsed_val is not None else None
            scores_json = Json(scores_val) if scores_val is not None else None
            context_obj = _coerce_json_object(contexts_all.get(domain))
            context_json = Json(context_obj) if context_obj is not None else None
            system_context_json = Json(system_context_snapshot) if system_context_snapshot is not None else None
            is_final = (domain == "final")
            sla_v = run_sla_verdict if is_final else None
            sla_d = Json(run_sla_details) if (is_final and run_sla_details) else None

            rows.append(
                (
                    run_id, run_name, service, test_type, start_ms, end_ms, domain,
                    text_str, parsed_json, scores_json, context_json, system_context_json,
                    verdict_std, sla_v, sla_d, project_area,
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
        raise ValueError("run_name обязателен")
    if not domain:
        raise ValueError("domain обязателен")
    if target not in FEEDBACK_TARGETS:
        raise ValueError("target должен быть verdict или finding")
    if vote not in FEEDBACK_VOTES:
        raise ValueError("vote должен быть agree или disagree")
    if target == "verdict":
        finding_id = ""
    elif not finding_id:
        raise ValueError("finding_id обязателен для находки")
    if len(comment) > 2000:
        raise ValueError("comment слишком длинный")
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

