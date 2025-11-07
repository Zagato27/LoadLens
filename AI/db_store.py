import logging
import json
from typing import Dict, Iterable, List, Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch


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
                            scores      JSONB
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

def save_llm_results(
    results: Dict[str, object],
    run_meta: Dict[str, object],
    storage_cfg: Dict[str, object]
) -> None:
    """Сохраняет текст/parsed/scores по доменам и финалу в обычную таблицу llm_reports."""
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
                run_id, run_name, service, test_type, start_ms, end_ms, domain, text, parsed, scores, verdict
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))

        run_id = str((run_meta or {}).get("run_id") or "")
        run_name = str((run_meta or {}).get("run_name") or "")
        service = str((run_meta or {}).get("service") or "")
        test_type = str((run_meta or {}).get("test_type") or "")
        start_ms = int((run_meta or {}).get("start_ms") or 0)
        end_ms = int((run_meta or {}).get("end_ms") or 0)

        rows = []
        domains = [
            ("jvm", results.get("jvm"), results.get("jvm_parsed"), (results.get("scores", {}) or {}).get("jvm")),
            ("database", results.get("database"), results.get("database_parsed"), (results.get("scores", {}) or {}).get("database")),
            ("kafka", results.get("kafka"), results.get("kafka_parsed"), (results.get("scores", {}) or {}).get("kafka")),
            ("microservices", results.get("ms"), results.get("ms_parsed"), (results.get("scores", {}) or {}).get("microservices")),
            ("hard_resources", results.get("hard_resources"), results.get("hard_resources_parsed"), (results.get("scores", {}) or {}).get("hard_resources")),
            ("final", results.get("final"), results.get("final_parsed"), (results.get("scores", {}) or {}).get("final")),
        ]
        # Дополнительно сохраняем lt_framework, если присутствует
        try:
            if "lt_framework" in results:
                domains.insert(-1, ("lt_framework", results.get("lt_framework"), results.get("lt_framework_parsed"), (results.get("scores", {}) or {}).get("lt_framework")))
        except Exception:
            pass

        # Стандартизированный вердикт рассчитываем один раз для финального домена
        final_parsed = results.get("final_parsed") if isinstance(results, dict) else None
        final_verdict_std = None
        try:
            if isinstance(final_parsed, dict):
                final_verdict_std = _standardize_verdict(final_parsed.get("verdict"))
            else:
                final_verdict_std = _standardize_verdict(None)
        except Exception:
            final_verdict_std = "Недостаточно данных"

        for domain, text_val, parsed_val, scores_val in domains:
            # сохраняем даже пустые тексты, чтобы фиксировать сам факт попытки
            rows.append(
                (
                    run_id, run_name, service, test_type, start_ms, end_ms, domain,
                    str(text_val) if text_val is not None else None,
                    json.dumps(parsed_val) if parsed_val is not None else None,
                    json.dumps(scores_val) if scores_val is not None else None,
                    (final_verdict_std if domain == "final" else None),
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

