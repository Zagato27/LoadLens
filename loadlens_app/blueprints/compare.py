# SPDX-License-Identifier: Apache-2.0
"""Blueprint with comparison/reporting endpoints.

A is the assessed run, B the baseline: ``trend_pct`` of the summaries is the change of A relative to B and
``direction`` tells whether a higher value of the metric is better (AI/metric_direction.py).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlencode

import psycopg2
from psycopg2 import sql as psql
from flask import Blueprint, jsonify, render_template, request

from settings import CONFIG
from AI.db_store import (
    BASELINE_MODE_PINNED,
    BASELINE_MODE_PREVIOUS,
    BASELINE_MODE_PREVIOUS_RELEASE,
    BASELINE_MODE_PREVIOUS_SUCCESS,
    BASELINE_MODES,
)
from AI.i18n import t as translate
from AI.limit_explanation import ExplanationParseError
from AI.metric_direction import change_pct, metric_direction
from AI.verdicts import to_code
from loadlens_app.auth import current_auth
from loadlens_app.compare_explanation import CompareView, explanation_view, generate_explanation, query_roles
# publish_comparison runs in loadlens_app.tasks, which looks it up here (tests replace it here).
from loadlens_app.compare_confluence import load_compare_publication, publish_comparison  # noqa: F401
from loadlens_app.errors import internal_error
from loadlens_app.core import (
    _active_project_area,
    _resolve_services_filter,
    _series_key_for,
    _ts_conn,
    public_base_url,
)
from loadlens_app.forecast import database_unavailable
from loadlens_app.forecast_explanation import ExplanationError
from loadlens_app import job_runner, tasks
from loadlens_app.jobs import JOB_KIND_COMPARE_CONFLUENCE, failure_message, job_message, job_store
from loadlens_app.limits import LimitError, check_job_slot

compare_bp = Blueprint("compare", __name__)

# Whitelisted SQL aggregates for summary tables; keys are accepted as the ``agg`` query param.
AGGREGATES = {
    "p95": "percentile_cont(0.95) WITHIN GROUP (ORDER BY value)",
    "avg": "AVG(value)",
    "max": "MAX(value)",
}
DEFAULT_AGGREGATE = "p95"
MAX_RUN_NAME = 200
# Labels of the baseline modes in the "not found" answer of /compare_baseline.
_BASELINE_LABEL_KEYS = {
    BASELINE_MODE_PINNED: "api.compare.pinned",
    BASELINE_MODE_PREVIOUS_RELEASE: "api.compare.previous_release",
    BASELINE_MODE_PREVIOUS_SUCCESS: "api.compare.previous_success",
    BASELINE_MODE_PREVIOUS: "api.compare.previous",
}


def _aggregate_sql() -> tuple[str, str] | tuple[None, str]:
    """Returns (agg_key, sql) for the ``agg`` request param or (None, error message)."""
    agg = (request.args.get("agg") or DEFAULT_AGGREGATE).strip().lower()
    if agg not in AGGREGATES:
        return None, translate("api.compare.bad_aggregate", agg=agg, allowed=", ".join(AGGREGATES))
    return agg, AGGREGATES[agg]


def _to_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _trend_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Change of run A relative to the baseline B in percent."""
    return change_pct(a, b)


def _roles(domain: str) -> Mapping[str, str]:
    """{label: role} of the domain's queries in the active project area (the global queries without one)."""
    return query_roles(_active_project_area()).get(domain) or {}


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
        return jsonify({"error": translate("api.required.compare_runs")}), 400
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
        roles = _roles(domain)
        out = []
        for ql, a, b in rows:
            a_f, b_f = _to_float(a), _to_float(b)
            out.append({
                "query_label": ql,
                "value_a": a_f,
                "value_b": b_f,
                "trend_pct": _trend_pct(a_f, b_f),
                "direction": metric_direction(ql, roles.get(ql)),
                "agg": agg,
            })
        return jsonify(out)
    except Exception as e:  # pragma: no cover
        return internal_error(e)


@compare_bp.route("/compare_metric_summary", methods=["GET"])
def compare_metric_summary():
    """Сравнивает агрегат (p95/avg/max) по отдельным сериям одной метрики между двумя запусками."""
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    if not all([run_a, run_b, domain, ql]):
        return jsonify({"error": translate("api.required.compare_series")}), 400
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
        direction = metric_direction(ql, _roles(domain).get(ql))
        out = []
        for s, a, b in rows:
            a_f, b_f = _to_float(a), _to_float(b)
            out.append({"series": s, "value_a": a_f, "value_b": b_f, "trend_pct": _trend_pct(a_f, b_f), "direction": direction})
        return jsonify({"query_label": ql, "series_key": series_key, "agg": agg, "direction": direction, "rows": out})
    except Exception as e:  # pragma: no cover
        return internal_error(e)


def _baseline_current_run(conn, schema: str, table: str, run_name: str) -> tuple | None:
    """(service, created_at, test_type, project_area) of the run's latest final report.

    Only runs of the active project area are found, like every other compare endpoint.
    """
    svc_where, svc_params = _services_clause()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COALESCE(service, ''), created_at, COALESCE(test_type, ''), COALESCE(project_area, '')
            FROM {schema}.{table}
            WHERE run_name = %s AND domain = 'final'{svc_where}
            ORDER BY created_at DESC LIMIT 1
            """,
            (run_name, *svc_params),
        )
        return cur.fetchone()


@compare_bp.route("/compare_baseline", methods=["GET"])
def compare_baseline():
    """Ищет предыдущий прогон того же сервиса, проектной области и типа теста для сравнения.

    ``mode=previous_success`` (по умолчанию) — последний более ранний прогон с итоговым
    вердиктом ``pass``; ``mode=previous`` — последний более ранний прогон без учёта вердикта;
    ``mode=pinned`` — закреплённый эталон сервиса и типа, иначе как ``previous_release``;
    ``mode=previous_release`` — принятый прогон прошлого релиза, иначе как ``previous_success``
    (``AI.release_marks.find_baseline_run``). В ответе ``mode`` — режим, который нашёл базу, и ``release`` базы.
    Тип теста сравнивается с учётом синонимов (``AI.db_store.find_previous_run``); отчёты,
    сохранённые без области, подходят любой области. ``verdict`` в ответе — код.
    """
    from AI import release_marks
    from AI.test_profile import normalize_test_type

    run_name = (request.args.get("run_name") or "").strip()
    mode = (request.args.get("mode") or "previous_success").strip().lower()
    if not run_name:
        return jsonify({"error": translate("api.required.run_name")}), 400
    if mode not in BASELINE_MODES:
        return jsonify({"error": translate("api.compare.bad_mode", mode=mode, allowed=", ".join(BASELINE_MODES))}), 400
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema = cfg.get("schema", "public")
    table = cfg.get("llm_table", "llm_reports")
    type_key = ""
    previous = None
    try:
        conn = _ts_conn()
        try:
            with conn:
                current = _baseline_current_run(conn, schema, table, run_name)
                if current:
                    service, created_at, test_type, area = (list(current) + [None] * 4)[:4]
                    type_key = normalize_test_type(str(test_type or ""))
                    previous = release_marks.find_baseline_run(
                        conn, schema, table,
                        exclude_run_name=run_name,
                        mode=mode,
                        service=str(service or ""),
                        before=created_at,
                        project_area=str(area or "").strip() or _active_project_area() or None,
                        test_type=type_key or None,
                    )
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover
        return internal_error(e)
    if previous is None:
        label = translate(_BASELINE_LABEL_KEYS[mode])
        message = (
            translate("api.compare.baseline_not_found_type", run_name=run_name, label=label, test_type=type_key)
            if type_key
            else translate("api.compare.baseline_not_found", run_name=run_name, label=label)
        )
        return jsonify({"error": message, "mode": mode}), 404
    return jsonify({
        "run_name": previous.run_name,
        "service": previous.service,
        "verdict": to_code(previous.verdict),
        "created_at": previous.created_at.isoformat() if previous.created_at else None,
        "mode": previous.mode or mode,
        "release": previous.release,
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
        return internal_error(e)


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
        return jsonify({"error": translate("api.required.compare_series")}), 400
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
        return internal_error(e)


@compare_bp.route("/run_series", methods=["GET"])
def run_series():
    """Возвращает временной ряд конкретного запроса в рамках одного запуска."""
    run_name = request.args.get("run_name")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    align = request.args.get("align", "absolute")
    if not all([run_name, domain, ql]):
        return jsonify({"error": translate("api.required.run_series")}), 400
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
        return internal_error(e)


# ---- scenario runs: phase against phase ----------------------------------------------------------------------

# Roles of the per-phase figures, in the order of the table (AI/query_packs/roles.py).
PHASE_METRIC_ROLES = (("p95", "lt.p95_total"), ("errors", "lt.error_rate_pct"))


def _scenario_payloads(cur, run_a: str, run_b: str) -> dict[str, Any]:
    """``scenario`` of the latest final row of each run ({} for a run that is not a scenario)."""
    storage = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    schema = str(storage.get("schema") or "public")
    table = str(storage.get("llm_table") or "llm_reports")
    cur.execute(
        psql.SQL(
            """
            SELECT DISTINCT ON (run_name) run_name, parsed->'scenario'
            FROM {table}
            WHERE run_name IN (%s, %s) AND domain = 'final'
            ORDER BY run_name, created_at DESC
            """
        ).format(table=psql.SQL("{}.{}").format(psql.Identifier(schema), psql.Identifier(table))),
        (run_a, run_b),
    )
    out: dict[str, Any] = {}
    for name, scenario in cur.fetchall():
        if isinstance(scenario, (str, bytes)):
            try:
                scenario = json.loads(scenario)
            except ValueError:
                scenario = None
        out[str(name)] = scenario if isinstance(scenario, dict) else {}
    return out


def _phase_windows(pairs: list[dict], run_a: str, run_b: str) -> list[tuple[str, str, str, str]]:
    """(run, phase key of the pair, start, end) of every phase of both runs; the key is the pair position."""
    windows = []
    for position, pair in enumerate(pairs):
        for side, run in (("a", run_a), ("b", run_b)):
            phase = pair.get(side)
            if phase and phase.get("start_utc") and phase.get("end_utc"):
                windows.append((run, str(position), str(phase["start_utc"]), str(phase["end_utc"])))
    return windows


def _phase_values(cur, pairs: list[dict], run_a: str, run_b: str, wanted: list[str]) -> dict[tuple[str, str, str], float]:
    """{(run, pair position, query label): p95 inside the phase window} of the ``wanted`` lt_framework queries."""
    windows = _phase_windows(pairs, run_a, run_b)
    if not wanted or not windows:
        return {}
    svc_where, svc_params = _services_clause("m")
    rows_sql = ", ".join(["(%s, %s, %s::timestamptz, %s::timestamptz)"] * len(windows))
    cur.execute(
        f"""
        WITH w(run_name, pair, t0, t1) AS (VALUES {rows_sql})
        SELECT w.run_name, w.pair, m.query_label, {AGGREGATES['p95']} AS v
        FROM w JOIN public.metrics m
          ON m.run_name = w.run_name AND m."time" >= w.t0 AND m."time" < w.t1
        WHERE m.domain = 'lt_framework' AND m.query_label = ANY(%s){svc_where}
        GROUP BY w.run_name, w.pair, m.query_label
        """,
        tuple(item for window in windows for item in window) + (wanted, *svc_params),
    )
    values: dict[tuple[str, str, str], float] = {}
    for run, pair, label, value in cur.fetchall():
        number = _to_float(value)
        if number is not None:
            values[(str(run), str(pair), str(label))] = number
    return values


@compare_bp.route("/compare_phases", methods=["GET"])
def compare_phases():
    """Phases of two scenario runs side by side: duration, mean RPS, SLA verdict and the p95 of the latency and error
    queries (roles lt.p95_total, lt.error_rate_pct) inside every phase; A against the baseline B."""
    from AI.phases import phase_pairs

    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    if not run_a or not run_b:
        return jsonify({"error": translate("api.required.compare_runs")}), 400
    try:
        conn = _ts_conn()
        try:
            with conn, conn.cursor() as cur:
                scenarios = _scenario_payloads(cur, run_a, run_b)
                if not scenarios.get(run_a) or not scenarios.get(run_b):
                    matched = None
                else:
                    matched = phase_pairs(scenarios[run_a], scenarios[run_b])
                    roles = _roles("lt_framework")
                    labels = {key: next((label for label, value in roles.items() if value == role), None) for key, role in PHASE_METRIC_ROLES}
                    values = _phase_values(cur, matched["pairs"], run_a, run_b, [label for label in labels.values() if label])
        finally:
            conn.close()
        if matched is None:
            return jsonify({"available": False, "run_a": run_a, "run_b": run_b, "pairs": []})
        pairs = []
        for position, pair in enumerate(matched["pairs"]):
            metrics = {}
            for key, label in labels.items():
                if not label:
                    continue
                a = values.get((run_a, str(position), label))
                b = values.get((run_b, str(position), label))
                metrics[key] = {"value_a": a, "value_b": b, "trend_pct": _trend_pct(a, b),
                                "direction": metric_direction(label, roles.get(label))}
            pairs.append({**pair, "metrics": metrics})
        return jsonify({
            "available": True, "run_a": run_a, "run_b": run_b, "agg": "p95",
            "metrics": [{"key": key, "query_label": label} for key, label in labels.items() if label],
            "pairs": pairs, "only_in_a": matched["only_in_a"], "only_in_b": matched["only_in_b"],
        })
    except Exception as e:  # pragma: no cover
        return internal_error(e)


# ---- AI explanation of the comparison and its Confluence page ------------------------------------------------


class PairError(ValueError):
    """The request does not name two different runs."""


def _pair(source: Mapping[str, Any]) -> tuple[str, str]:
    """``(run_a, run_b)`` of query args or a JSON body: both required, different and not too long."""
    run_a = str(source.get("run_a") or "").strip()
    run_b = str(source.get("run_b") or "").strip()
    if not run_a or not run_b:
        raise PairError(translate("api.required.compare_pair"))
    if len(run_a) > MAX_RUN_NAME or len(run_b) > MAX_RUN_NAME:
        raise PairError(translate("api.compare.bad_run_name", max=MAX_RUN_NAME))
    if run_a == run_b:
        raise PairError(translate("api.compare.same_run"))
    return run_a, run_b


def _owner() -> str:
    user = current_auth().user
    return user.username if user is not None else "local"


def _json_body():
    """The JSON object of a POST, or the error response (415 without JSON, 400 for another value)."""
    if not request.is_json:
        return None, (jsonify({"error": translate("api.common.json_required"), "code": "bad_request"}), 415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": translate("api.common.json_body_object"), "code": "bad_request"}), 400)
    return data, None


def _explanation_response(build: Callable[[], CompareView]):
    try:
        view = build()
    except (ExplanationError, LimitError) as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    except ExplanationParseError as exc:
        return jsonify({"error": str(exc), "code": "bad_answer"}), 502
    except psycopg2.Error as exc:
        failure = database_unavailable(exc)
        return jsonify({"error": str(failure), "code": failure.code}), failure.status_code
    return jsonify(view.to_dict())


@compare_bp.route("/compare_explanation", methods=["GET"])
def get_compare_explanation():
    """Outcome, caveats and key changes of A against B with the cached LLM explanation."""
    try:
        run_a, run_b = _pair(request.args)
    except PairError as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    return _explanation_response(lambda: explanation_view(run_a, run_b))


@compare_bp.route("/compare_explanation", methods=["POST"])
def post_compare_explanation():
    """Asks the LLM to explain the differences; JSON only, so a cross-site form cannot start the call."""
    data, rejected = _json_body()
    if rejected is not None:
        return rejected
    try:
        run_a, run_b = _pair(data)
    except PairError as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    owner = _owner()
    return _explanation_response(lambda: generate_explanation(run_a, run_b, budget_owner=owner))


@compare_bp.route("/compare_confluence", methods=["GET"])
def get_compare_confluence():
    """Confluence page of the comparison, if it was published."""
    try:
        run_a, run_b = _pair(request.args)
    except PairError as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    try:
        publication = load_compare_publication(run_a, run_b)
    except psycopg2.Error as exc:
        failure = database_unavailable(exc)
        return jsonify({"error": str(failure), "code": failure.code}), failure.status_code
    if publication is None:
        return jsonify({"run_a": run_a, "run_b": run_b, "page_id": None, "page_url": None, "updated_at": None})
    return jsonify({
        "run_a": publication.run_a,
        "run_b": publication.run_b,
        "page_id": publication.page_id,
        "page_url": publication.page_url,
        "updated_at": publication.updated_at,
    })


def _failure_message(job_id: str):
    """Failure message (a ``Msg`` in the job's locale) naming the stage of the publication that failed."""
    return failure_message(job_store.get(job_id), "job.compare_failed")


def _compare_url(run_a: str, run_b: str) -> str:
    return "/compare?" + urlencode({"run_a": run_a, "run_b": run_b})


@compare_bp.route("/compare_confluence", methods=["POST"])
def post_compare_confluence():
    """Starts publishing the comparison of A with B as its own Confluence page (a background job)."""
    data, rejected = _json_body()
    if rejected is not None:
        return rejected
    try:
        run_a, run_b = _pair(data)
    except PairError as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    owner = _owner()
    try:
        check_job_slot(*job_store.running_counts(owner))
    except LimitError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    # The runner the job is registered for (Celery worker or a thread here) decides where it is published.
    job = job_runner.create_job(
        JOB_KIND_COMPARE_CONFLUENCE, run_name=run_a, message=job_message("job.publishing_compare"), owner=owner,
    )
    page = _compare_url(run_a, run_b)
    job_store.update(job.job_id, report_url=page)
    job_runner.submit(
        tasks.publish_compare_task, job_store.get(job.job_id) or job.job_id,
        run_a=run_a,
        run_b=run_b,
        source_url=f"{public_base_url()}{page}",
    )
    return jsonify({"status": "accepted", "job_id": job.job_id, "message": translate("api.compare.publish_started")}), 200
