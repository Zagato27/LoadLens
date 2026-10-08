"""Blueprint with comparison/reporting endpoints."""

from __future__ import annotations

from typing import Optional

from flask import Blueprint, jsonify, render_template, request

from settings import CONFIG
from loadlens_app.core import (
    _active_project_area,
    _resolve_services_filter,
    _series_key_for,
    _ts_conn,
)

compare_bp = Blueprint("compare", __name__)

# Whitelisted SQL aggregates for summary tables; keys are accepted as the ``agg`` query param.
AGGREGATES = {
    "p95": "percentile_cont(0.95) WITHIN GROUP (ORDER BY value)",
    "avg": "AVG(value)",
    "max": "MAX(value)",
}
DEFAULT_AGGREGATE = "p95"
BASELINE_MODES = ("previous_success", "previous")


def _aggregate_sql() -> tuple[str, str] | tuple[None, str]:
    """Returns (agg_key, sql) for the ``agg`` request param or (None, error message)."""
    agg = (request.args.get("agg") or DEFAULT_AGGREGATE).strip().lower()
    if agg not in AGGREGATES:
        return None, f"Недопустимый агрегат «{agg}». Допустимо: {', '.join(AGGREGATES)}"
    return agg, AGGREGATES[agg]


def _to_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _trend_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = abs(a) if abs(a) > 1e-9 else 1.0
    return ((b - a) / denom) * 100.0


def _services_clause(alias: str = "") -> tuple[str, list]:
    """SQL fragment (with leading AND) restricting metrics rows to the active project area."""
    services_filter = _resolve_services_filter(_active_project_area())
    if not services_filter:
        return "", []
    column = f"{alias}.service" if alias else "service"
    return f" AND {column} = ANY(%s)", [services_filter]


@compare_bp.route("/compare")
def compare_page():
    """Отображает страницу сравнения двух запусков."""
    return render_template("compare.html")


@compare_bp.route("/compare_summary", methods=["GET"])
def compare_summary():
    """Сравнивает агрегат (p95/avg/max) каждой метрики домена между двумя запусками."""
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    if not all([run_a, run_b, domain]):
        return jsonify({"error": "run_a, run_b, domain обязательны"}), 400
    agg, agg_sql = _aggregate_sql()
    if agg is None:
        return jsonify({"error": agg_sql}), 400
    try:
        svc_where, svc_params = _services_clause()
        conn = _ts_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                WITH raw AS (
                  SELECT query_label, run_name, value
                  FROM public.metrics
                  WHERE run_name IN (%s, %s) AND domain = %s{svc_where}
                ), p AS (
                  SELECT query_label, run_name, {agg_sql} AS agg_value
                  FROM raw GROUP BY query_label, run_name
                )
                SELECT query_label,
                       MAX(agg_value) FILTER (WHERE run_name = %s) AS value_a,
                       MAX(agg_value) FILTER (WHERE run_name = %s) AS value_b
                FROM p GROUP BY query_label ORDER BY query_label
                """,
                (run_a, run_b, domain, *svc_params, run_a, run_b),
            )
            rows = cur.fetchall()
        conn.close()
        out = []
        for ql, a, b in rows:
            a_f, b_f = _to_float(a), _to_float(b)
            out.append({"query_label": ql, "value_a": a_f, "value_b": b_f, "trend_pct": _trend_pct(a_f, b_f), "agg": agg})
        return jsonify(out)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@compare_bp.route("/compare_metric_summary", methods=["GET"])
def compare_metric_summary():
    """Сравнивает агрегат (p95/avg/max) по отдельным сериям одной метрики между двумя запусками."""
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    if not all([run_a, run_b, domain, ql]):
        return jsonify({"error": "run_a, run_b, domain, query_label обязательны"}), 400
    agg, agg_sql = _aggregate_sql()
    if agg is None:
        return jsonify({"error": agg_sql}), 400
    try:
        if not series_key or series_key == "auto":
            series_key = _series_key_for(domain, ql, default_key="application")
        svc_where, svc_params = _services_clause("m")
        conn = _ts_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                WITH base AS (
                  SELECT m.series AS series_name, m.run_name, m.value
                  FROM public.metrics m
                  WHERE m.run_name IN (%s, %s) AND m.domain = %s AND m.query_label = %s{svc_where}
                ), p AS (
                  SELECT series_name, run_name, {agg_sql} AS agg_value
                  FROM base GROUP BY series_name, run_name
                )
                SELECT series_name,
                       MAX(agg_value) FILTER (WHERE run_name = %s) AS value_a,
                       MAX(agg_value) FILTER (WHERE run_name = %s) AS value_b
                FROM p GROUP BY series_name ORDER BY series_name
                """,
                (run_a, run_b, domain, ql, *svc_params, run_a, run_b),
            )
            rows = cur.fetchall()
        conn.close()
        out = []
        for s, a, b in rows:
            a_f, b_f = _to_float(a), _to_float(b)
            out.append({"series": s, "value_a": a_f, "value_b": b_f, "trend_pct": _trend_pct(a_f, b_f)})
        return jsonify({"query_label": ql, "series_key": series_key, "agg": agg, "rows": out})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@compare_bp.route("/compare_baseline", methods=["GET"])
def compare_baseline():
    """Ищет предыдущий прогон того же сервиса для сравнения.

    ``mode=previous_success`` (по умолчанию) — последний более ранний прогон с итоговым
    вердиктом «Успешно»; ``mode=previous`` — последний более ранний прогон без учёта вердикта.
    """
    run_name = (request.args.get("run_name") or "").strip()
    mode = (request.args.get("mode") or "previous_success").strip().lower()
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    if mode not in BASELINE_MODES:
        return jsonify({"error": f"Недопустимый mode «{mode}». Допустимо: {', '.join(BASELINE_MODES)}"}), 400
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema = cfg.get("schema", "public")
    table = cfg.get("llm_table", "llm_reports")
    success_filter = "AND COALESCE(r.sla_verdict, r.verdict) = 'Успешно'" if mode == "previous_success" else ""
    try:
        conn = _ts_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                WITH current AS (
                  SELECT service, created_at FROM {schema}.{table}
                  WHERE run_name = %s AND domain = 'final'
                  ORDER BY created_at DESC LIMIT 1
                )
                SELECT r.run_name, r.service, COALESCE(r.sla_verdict, r.verdict, 'Недостаточно данных'), r.created_at
                FROM {schema}.{table} r, current c
                WHERE r.domain = 'final' AND r.service = c.service AND r.run_name <> %s AND r.created_at < c.created_at
                {success_filter}
                ORDER BY r.created_at DESC LIMIT 1
                """,
                (run_name, run_name),
            )
            row = cur.fetchone()
        conn.close()
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500
    if not row:
        label = "предыдущий успешный прогон" if mode == "previous_success" else "предыдущий прогон"
        return jsonify({"error": f"Для «{run_name}» не найден {label} того же сервиса", "mode": mode}), 404
    return jsonify({
        "run_name": row[0],
        "service": row[1],
        "verdict": row[2],
        "created_at": row[3].isoformat() if row[3] else None,
        "mode": mode,
    })


@compare_bp.route("/domains_schema", methods=["GET"])
def domains_schema():
    """Отдаёт домены и query_label, присутствующие в метриках.

    Необязательный ``run_name`` сужает схему до одного запуска, чтобы страница
    отчёта не показывала вкладки доменов без данных.
    """
    run_name = (request.args.get("run_name") or "").strip()
    try:
        svc_where, svc_params = _services_clause()
        conn = _ts_conn()
        with conn, conn.cursor() as cur:
            clauses = []
            params: list = []
            if run_name:
                clauses.append("run_name = %s")
                params.append(run_name)
            if svc_where:
                clauses.append(svc_where[len(" AND "):])
                params.extend(svc_params)
            where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            cur.execute(
                f"""
                SELECT domain, query_label, COUNT(*) AS cnt
                FROM public.metrics
                {where_sql}
                GROUP BY domain, query_label
                ORDER BY domain, query_label
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        conn.close()
        out = {}
        for d, ql, cnt in rows:
            out.setdefault(d, []).append({"query_label": ql, "count": int(cnt)})
        return jsonify(out)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@compare_bp.route("/compare_series", methods=["GET"])
def compare_series():
    """Отдаёт временные ряды выбранного домена/подписей для двух запусков."""
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    align = request.args.get("align", "offset")
    if not all([run_a, run_b, domain, ql]):
        return jsonify({"error": "run_a, run_b, domain, query_label обязательны"}), 400
    try:
        if not series_key or series_key == "auto":
            series_key = _series_key_for(domain, ql, default_key="application")
        svc_where, svc_params = _services_clause("m")
        conn = _ts_conn()
        sql_common = f"""
          SELECT m."time", m.run_name, m.series AS series_name, m.value
          FROM public.metrics m
          WHERE m.run_name IN (%s, %s) AND m.domain = %s AND m.query_label = %s{svc_where}
        """
        base_params = [run_a, run_b, domain, ql, *svc_params]
        with conn, conn.cursor() as cur:
            if align == "absolute":
                cur.execute(
                    f"""
                      WITH base AS ({sql_common})
                      SELECT time_bucket('1 minute'::interval, base."time") AS t,
                             base.run_name, base.series_name, avg(base.value) AS v
                      FROM base GROUP BY 1,2,3 ORDER BY 1,2,3
                    """,
                    tuple(base_params),
                )
                rows = cur.fetchall()
                data = [{"t": r[0].isoformat(), "run_name": r[1], "series": r[2], "value": float(r[3])} for r in rows]
            else:
                bucket_secs = 60
                cur.execute(
                    f"""
                      WITH base AS ({sql_common}),
                      start_ts AS (SELECT run_name, MIN("time") AS t0 FROM base GROUP BY run_name)
                      SELECT
                        FLOOR(EXTRACT(EPOCH FROM (base."time" - st.t0)) / %s)::bigint * %s AS t_offset_sec,
                        base.run_name, base.series_name, avg(base.value) AS v
                      FROM base JOIN start_ts st USING (run_name)
                      GROUP BY 1,2,3 ORDER BY 2,1,3
                    """,
                    tuple(base_params + [bucket_secs, bucket_secs]),
                )
                rows = cur.fetchall()
                data = [{"t_offset_sec": int(r[0]), "run_name": r[1], "series": r[2], "value": float(r[3])} for r in rows]
        conn.close()
        return jsonify({"align": align, "points": data})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@compare_bp.route("/run_series", methods=["GET"])
def run_series():
    """Возвращает временной ряд конкретного запроса в рамках одного запуска."""
    run_name = request.args.get("run_name")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    align = request.args.get("align", "absolute")
    if not all([run_name, domain, ql]):
        return jsonify({"error": "run_name, domain, query_label обязательны"}), 400
    try:
        if not series_key or series_key == "auto":
            series_key = _series_key_for(domain, ql, default_key="application")
        svc_where, svc_params = _services_clause("m")
        conn = _ts_conn()
        sql_common = f"""
          SELECT m."time", m.run_name, m.series AS series_name, m.value
          FROM public.metrics m
          WHERE m.run_name = %s AND m.domain = %s AND m.query_label = %s{svc_where}
        """
        base_params = [run_name, domain, ql, *svc_params]
        with conn, conn.cursor() as cur:
            if align == "absolute":
                cur.execute(
                    f"""
                      WITH base AS ({sql_common})
                      SELECT time_bucket('1 minute'::interval, base."time") AS t,
                             base.series_name, avg(base.value) AS v
                      FROM base GROUP BY 1,2 ORDER BY 1,2
                    """,
                    tuple(base_params),
                )
                rows = cur.fetchall()
                data = [{"t": r[0].isoformat(), "series": r[1], "value": float(r[2])} for r in rows]
            else:
                bucket_secs = 60
                cur.execute(
                    f"""
                      WITH base AS ({sql_common}),
                      start_ts AS (SELECT MIN("time") AS t0 FROM base)
                      SELECT
                        FLOOR(EXTRACT(EPOCH FROM (base."time" - st.t0)) / %s)::bigint * %s AS t_offset_sec,
                        base.series_name, avg(base.value) AS v
                      FROM base, start_ts st
                      GROUP BY 1,2 ORDER BY 1,2
                    """,
                    tuple(base_params + [bucket_secs, bucket_secs]),
                )
                rows = cur.fetchall()
                data = [{"t_offset_sec": int(r[0]), "series": r[1], "value": float(r[2])} for r in rows]
        conn.close()
        return jsonify({"align": align, "points": data})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500
