"""Blueprint with comparison/reporting endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from loadlens_app.core import (
    _active_project_area,
    _resolve_services_filter,
    _series_key_for,
    _ts_conn,
)

compare_bp = Blueprint("compare", __name__)


@compare_bp.route("/compare")
def compare_page():
    """Отображает страницу сравнения двух запусков."""
    return render_template("compare.html")


@compare_bp.route("/compare_summary", methods=["GET"])
def compare_summary():
    """Сравнивает p95-значения выбранного домена между двумя запусками."""
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    if not all([run_a, run_b, domain]):
        return jsonify({"error": "run_a, run_b, domain обязательны"}), 400
    try:
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    """
                    WITH raw AS (
                      SELECT query_label, run_name, value
                      FROM public.metrics
                      WHERE run_name IN (%s, %s)
                        AND domain = %s
                        AND service = ANY(%s)
                    ), p AS (
                      SELECT query_label,
                             run_name,
                             percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
                      FROM raw
                      GROUP BY query_label, run_name
                    )
                    SELECT query_label,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_a,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_b
                    FROM p
                    GROUP BY query_label
                    ORDER BY query_label
                    """,
                    (run_a, run_b, domain, services_filter, run_a, run_b),
                )
            else:
                cur.execute(
                    """
                    WITH raw AS (
                      SELECT query_label, run_name, value
                      FROM public.metrics
                      WHERE run_name IN (%s, %s)
                        AND domain = %s
                    ), p AS (
                      SELECT query_label,
                             run_name,
                             percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
                      FROM raw
                      GROUP BY query_label, run_name
                    )
                    SELECT query_label,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_a,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_b
                    FROM p
                    GROUP BY query_label
                    ORDER BY query_label
                    """,
                    (run_a, run_b, domain, run_a, run_b),
                )
            rows = cur.fetchall()
        conn.close()

        def _safe_float(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        out = []
        for ql, a, b in rows:
            a_f = _safe_float(a)
            b_f = _safe_float(b)
            trend = None
            if a_f is not None and b_f is not None:
                denom = abs(a_f) if abs(a_f) > 1e-9 else 1.0
                trend = ((b_f - a_f) / denom) * 100.0
            out.append({"query_label": ql, "p95_a": a_f, "p95_b": b_f, "trend_pct": trend})
        return jsonify(out)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@compare_bp.route("/compare_metric_summary", methods=["GET"])
def compare_metric_summary():
    """Сравнивает p95 по отдельным сериям (лейблам) внутри запроса."""
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    if not all([run_a, run_b, domain, ql]):
        return jsonify({"error": "run_a, run_b, domain, query_label обязательны"}), 400
    try:
        if not series_key or series_key == "auto":
            series_key = _series_key_for(domain, ql, default_key="application")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        svc_clause = " AND m.service = ANY(%s)" if services_filter else ""
        sql_common = f"""
          SELECT
            regexp_replace(m.series, '.*{series_key}=([^|]+).*', '\\\\1') AS series_name,
            m.run_name,
            m.value
          FROM public.metrics m
          WHERE m.run_name IN (%s, %s)
            AND m.domain = %s
            AND m.query_label = %s
            {svc_clause}
        """
        base_params = [run_a, run_b, domain, ql]
        if services_filter:
            base_params.append(services_filter)
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                  WITH base AS ({sql_common}),
                  p AS (
                    SELECT
                      series_name,
                      run_name,
                      percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
                    FROM base
                    GROUP BY series_name, run_name
                  )
                  SELECT
                    series_name,
                    MAX(p95) FILTER (WHERE run_name = %s) AS p95_a,
                    MAX(p95) FILTER (WHERE run_name = %s) AS p95_b
                  FROM p
                  GROUP BY series_name
                  ORDER BY series_name
                """,
                (*base_params, run_a, run_b),
            )
            rows = cur.fetchall()
        conn.close()

        def _safe_float(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        out = []
        for s, a, b in rows:
            a_f = _safe_float(a)
            b_f = _safe_float(b)
            trend = None
            if a_f is not None and b_f is not None:
                denom = abs(a_f) if abs(a_f) > 1e-9 else 1.0
                trend = ((b_f - a_f) / denom) * 100.0
            out.append({"series": s, "p95_a": a_f, "p95_b": b_f, "trend_pct": trend})
        return jsonify({"query_label": ql, "series_key": series_key, "rows": out})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@compare_bp.route("/domains_schema", methods=["GET"])
def domains_schema():
    """Отдаёт список доменов и связанных query_label, присутствующих в БД."""
    try:
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    """
                    SELECT domain, query_label, COUNT(*) AS cnt
                    FROM public.metrics
                    WHERE service = ANY(%s)
                    GROUP BY domain, query_label
                    ORDER BY domain, query_label
                    """,
                    (services_filter,),
                )
            else:
                cur.execute(
                    """
                    SELECT domain, query_label, COUNT(*) AS cnt
                    FROM public.metrics
                    GROUP BY domain, query_label
                    ORDER BY domain, query_label
                    """
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
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        svc_clause = " AND m.service = ANY(%s)" if services_filter else ""
        sql_common = f"""
          SELECT
            m."time",
            m.run_name,
            regexp_replace(m.series, '.*{series_key}=([^|]+).*', '\\\\1') AS series_name,
            m.value
          FROM public.metrics m
          WHERE m.run_name IN (%s, %s)
            AND m.domain = %s
            AND m.query_label = %s
            {svc_clause}
        """
        base_params = [run_a, run_b, domain, ql]
        if services_filter:
            base_params.append(services_filter)
        with conn, conn.cursor() as cur:
            if align == "absolute":
                cur.execute(
                    f"""
                      WITH base AS ({sql_common})
                      SELECT
                        time_bucket('1 minute'::interval, base."time") AS t,
                        base.run_name,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base
                      GROUP BY 1,2,3
                      ORDER BY 1,2,3
                    """,
                    tuple(base_params),
                )
                rows = cur.fetchall()
                data = [{"t": r[0].isoformat(), "run_name": r[1], "series": r[2], "value": float(r[3])} for r in rows]
            else:
                bucket_secs = 60
                cur.execute(
                    f"""
                      WITH base AS (
                        {sql_common}
                      ), start_ts AS (
                        SELECT run_name, MIN("time") AS t0
                        FROM base GROUP BY run_name
                      )
                      SELECT
                        FLOOR(EXTRACT(EPOCH FROM (base."time" - st.t0)) / %s)::bigint * %s AS t_offset_sec,
                        base.run_name,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base
                      JOIN start_ts st USING (run_name)
                      GROUP BY 1,2,3
                      ORDER BY 2,1,3
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
    """Возвращает временной ряд конкретного запроса в рамках single-run."""
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
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        svc_clause = " AND m.service = ANY(%s)" if services_filter else ""
        sql_common = f"""
          SELECT
            m."time",
            m.run_name,
            regexp_replace(m.series, '.*{series_key}=([^|]+).*', '\\\\1') AS series_name,
            m.value
          FROM public.metrics m
          WHERE m.run_name = %s
            AND m.domain = %s
            AND m.query_label = %s
            {svc_clause}
        """.replace("{series_key}", series_key)
        base_params = [run_name, domain, ql]
        if services_filter:
            base_params.append(services_filter)
        with conn, conn.cursor() as cur:
            if align == "absolute":
                cur.execute(
                    f"""
                      WITH base AS ({sql_common})
                      SELECT
                        time_bucket('1 minute'::interval, base."time") AS t,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base
                      GROUP BY 1,2
                      ORDER BY 1,2
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
                      start_ts AS (
                        SELECT MIN("time") AS t0 FROM base
                      )
                      SELECT
                        FLOOR(EXTRACT(EPOCH FROM (base."time" - st.t0)) / %s)::bigint * %s AS t_offset_sec,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base, start_ts st
                      GROUP BY 1,2
                      ORDER BY 1,2
                    """,
                    tuple(base_params + [bucket_secs, bucket_secs]),
                )
                rows = cur.fetchall()
                data = [{"t_offset_sec": int(r[0]), "series": r[1], "value": float(r[2])} for r in rows]
        conn.close()
        return jsonify({"align": align, "points": data})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


