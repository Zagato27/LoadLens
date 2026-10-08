"""Dashboard and report-related routes."""

from __future__ import annotations

import threading
import uuid

from flask import (
    Blueprint,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
)

from AI.db_store import (
    _ensure_engineer_reports_table,
    _ensure_llm_reports_table,
    list_llm_feedback,
    upsert_llm_feedback,
)
from AI.scoring import parse_llm_analysis_strict
from settings import CONFIG
from update_page import default_run_name, update_report

from loadlens_app.confluence_export import load_publication, publish_report
from loadlens_app.core import (
    _active_metrics_config,
    _active_project_area,
    _available_domain_keys,
    _bootstrap_service_configs,
    _delete_run_data,
    _find_area_for_service,
    _list_project_areas,
    _metrics_service_entry,
    _metrics_services_for_area,
    _rename_run_data,
    _resolve_services_filter,
    _services_map_for_area,
    _ts_conn,
    convert_to_timestamp,
    run_name_problem,
)
from loadlens_app.demo_data import DemoAlreadyExistsError, seed_demo_run
from loadlens_app.jobs import (
    JOB_KIND_CONFLUENCE,
    JOB_KIND_REPORT,
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR,
    JOB_STATUSES,
    job_store,
)

dashboard_bp = Blueprint("dashboard", __name__)

JOBS_LIST_MAX_LIMIT = 100
DASHBOARD_RECENT_RUNS = 5
DASHBOARD_VERDICTS_PER_SERVICE = 5
PROJECT_AREA_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def _job_failure_message(job_id: str) -> str:
    """Human-readable failure message naming the phase that was running."""
    current = job_store.get(job_id)
    stage = (current.message or "").strip() if current else ""
    return f"Ошибка на этапе «{stage}»" if stage else "Ошибка выполнения задачи"


def _storage_cfg() -> dict:
    return (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}


def _report_page(service: str = "") -> object:
    """Report page. The project-area cookie follows the service so later API calls match the run."""
    resp = make_response(render_template("reports.html"))
    if not service:
        return resp
    try:
        area = _find_area_for_service(service) or service
        resp.set_cookie("project_area", area, max_age=PROJECT_AREA_COOKIE_MAX_AGE, samesite="Lax")
    except Exception:
        pass
    return resp


def _assign_run_id(conn, schema: str, llm_table: str, run_name: str, service: str) -> str:
    """Gives one id to every stored row of a run that was saved before ids existed."""
    new_id = uuid.uuid4().hex
    service_sql = " AND COALESCE(service, '') = %s" if service else ""
    params: tuple = (new_id, run_name, service) if service else (new_id, run_name)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {schema}.{llm_table} SET run_id = %s "
            f"WHERE run_name = %s AND COALESCE(run_id, '') = ''{service_sql}",
            params,
        )
        cur.execute(
            "UPDATE public.metrics SET run_id = %s "
            f"WHERE run_name = %s AND COALESCE(run_id, '') = ''{service_sql}",
            params,
        )
    return new_id


def _lookup_report_ref(run_id: str = "", run_name: str = "", service: str = "") -> dict | None:
    """Resolves a report to ``{run_id, run_name, service}``, assigning an id when the run has none."""
    run_id = str(run_id or "").strip()
    run_name = str(run_name or "").strip()
    service = str(service or "").strip()
    if not run_id and not run_name:
        return None
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    llm_table = cfg.get("llm_table", "llm_reports")
    conn = _ts_conn()
    try:
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        with conn, conn.cursor() as cur:
            if run_id:
                cur.execute(
                    f"SELECT COALESCE(run_id, ''), run_name, COALESCE(service, '') "
                    f"FROM {schema}.{llm_table} WHERE run_id = %s AND COALESCE(run_id, '') <> '' "
                    f"ORDER BY created_at DESC LIMIT 1",
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        "SELECT COALESCE(run_id, ''), run_name, COALESCE(service, '') "
                        "FROM public.metrics WHERE run_id = %s AND COALESCE(run_id, '') <> '' "
                        "ORDER BY time DESC LIMIT 1",
                        (run_id,),
                    )
                    row = cur.fetchone()
                if row is None:
                    return None
                return {"run_id": row[0], "run_name": row[1], "service": row[2] or ""}

            service_sql = " AND COALESCE(service, '') = %s" if service else ""
            name_params: tuple = (run_name, service) if service else (run_name,)
            cur.execute(
                f"SELECT COALESCE(MAX(NULLIF(run_id, '')), ''), run_name, COALESCE(MAX(service), '') "
                f"FROM {schema}.{llm_table} WHERE run_name = %s{service_sql} GROUP BY run_name",
                name_params,
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "SELECT COALESCE(MAX(NULLIF(run_id, '')), ''), run_name, COALESCE(MAX(service), '') "
                    f"FROM public.metrics WHERE run_name = %s{service_sql} GROUP BY run_name",
                    name_params,
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                return None
            found_id = str(row[0] or "").strip()
            found_service = service or str(row[2] or "")
            if not found_id:
                found_id = _assign_run_id(conn, schema, llm_table, row[1], found_service)
            return {"run_id": found_id, "run_name": row[1], "service": found_service}
    except Exception:
        return None
    finally:
        conn.close()


def _services_clause(services_filter: list[str], column: str = "service") -> tuple[str, list]:
    """SQL fragment (with leading AND) restricting rows to the active project area."""
    if not services_filter:
        return "", []
    return f" AND {column} = ANY(%s)", [services_filter]


@dashboard_bp.route("/")
def home():
    """Рендерит главный дашборд приложения."""
    return render_template("dashboard.html")


@dashboard_bp.route("/reports")
def reports_page():
    """Отображает страницу архива запусков."""
    return render_template("archive.html")


@dashboard_bp.route("/report_ref/<run_id>", methods=["GET"])
def report_ref(run_id: str):
    """Name and service for the report page addressed by ``run_id``."""
    ref = _lookup_report_ref(run_id=run_id)
    if ref is None:
        return jsonify({"error": "Отчёт не найден"}), 404
    return jsonify(ref)


@dashboard_bp.route("/reports/<token>")
def reports_page_run(token: str):
    """Report page by id. A legacy name-only address redirects to the id."""
    by_id = _lookup_report_ref(run_id=token)
    if by_id is not None:
        return _report_page(by_id.get("service") or "")
    by_name = _lookup_report_ref(run_name=token)
    if by_name is not None and by_name.get("run_id"):
        return redirect(f"/reports/{by_name['run_id']}", code=302)
    return render_template("reports.html")


@dashboard_bp.route("/reports/<service>/<run_name>")
def reports_page_service_run(service: str, run_name: str):
    """Legacy address ``/reports/<service>/<run_name>`` redirects to ``/reports/<run_id>``."""
    ref = _lookup_report_ref(run_name=run_name, service=service)
    if ref is not None and ref.get("run_id"):
        return redirect(f"/reports/{ref['run_id']}", code=302)
    return _report_page(service)


@dashboard_bp.route("/new")
def new_report_page():
    """Отображает форму создания нового отчёта."""
    return render_template("index.html")


@dashboard_bp.route("/services", methods=["GET"])
def get_services():
    """Возвращает список сервисов и доступных доменов для выбранной области."""
    area = (request.args.get("area") or "").strip()
    if not area:
        area = _active_project_area() or ""
    services_map = _services_map_for_area(area)
    metrics = _active_metrics_config() or {}
    area_metrics_services = _metrics_services_for_area(area)
    payload = []
    if services_map:
        for sid, meta in services_map.items():
            title = sid
            if isinstance(meta, dict):
                title = (meta.get("title") if isinstance(meta.get("title"), str) else "") or title
                disabled = [d for d in (meta.get("disabled_domains") or []) if isinstance(d, str)]
            else:
                disabled = []
            payload.append({"id": sid, "title": title, "disabled_domains": disabled})
    elif area_metrics_services:
        for sid in area_metrics_services.keys():
            payload.append({"id": sid, "title": sid, "disabled_domains": []})
    elif not area:
        for cfg in metrics.values():
            services = cfg.get("services")
            if isinstance(services, dict):
                for sid in services.keys():
                    payload.append({"id": sid, "title": sid, "disabled_domains": []})
    return jsonify({"area": area, "services": payload, "domains": _available_domain_keys()}), 200


def _group_service_rows(rows: list) -> list[dict]:
    """Groups (service, run_name, verdict, created_at, max_rps, test_type) rows, newest first per service."""
    grouped: dict[str, dict] = {}
    for service, run_name, verdict, created_at, max_rps, test_type in rows:
        entry = grouped.get(service)
        if entry is None:
            entry = {
                "service": service,
                "last_run": run_name,
                "verdict": verdict,
                "created_at": created_at.isoformat() if created_at else None,
                "test_type": test_type or "",
                "max_rps": float(max_rps) if max_rps not in (None, "") else None,
                "recent_verdicts": [],
            }
            grouped[service] = entry
        entry["recent_verdicts"].append(verdict)
    return sorted(grouped.values(), key=lambda e: e["created_at"] or "", reverse=True)


@dashboard_bp.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    """Сводка дашборда: последний запуск, распределение вердиктов, состояние по сервисам, недавние запуски."""
    try:
        cfg = _storage_cfg()
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        services_filter = _resolve_services_filter(_active_project_area())
        svc_where, svc_params = _services_clause(services_filter)
        last_run = None
        verdict_counts = {"Успешно": 0, "Есть риски": 0, "Провал": 0, "Недостаточно данных": 0}
        runs_total = 0
        services: list[dict] = []
        recent_runs: list[dict] = []
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT run_name, service, start_ms, end_ms,
                       COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS effective_verdict,
                       verdict AS llm_verdict, sla_verdict, created_at, COALESCE(run_id, '')
                FROM {schema}.{table}
                WHERE domain = 'final'{svc_where}
                ORDER BY created_at DESC
                LIMIT 1
                """,
                tuple(svc_params),
            )
            r = cur.fetchone()
            if r:
                last_run = {
                    "run_name": r[0],
                    "service": r[1],
                    "start_ms": int(r[2]) if r[2] is not None else None,
                    "end_ms": int(r[3]) if r[3] is not None else None,
                    "verdict": r[4],
                    "llm_verdict": r[5],
                    "sla_verdict": r[6],
                    "created_at": r[7].isoformat() if r[7] else None,
                    "run_id": str(r[8] or "").strip() if len(r) > 8 else "",
                }
            cur.execute(
                f"""
                WITH ranked AS (
                  SELECT run_name,
                         COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS effective_verdict,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                  FROM {schema}.{table}
                  WHERE domain = 'final'{svc_where}
                )
                SELECT effective_verdict AS v, COUNT(*) AS cnt FROM ranked WHERE rn = 1 GROUP BY v
                """,
                tuple(svc_params),
            )
            for v, cnt in cur.fetchall():
                if v in verdict_counts:
                    verdict_counts[v] = int(cnt)
                else:
                    verdict_counts["Недостаточно данных"] += int(cnt)
            cur.execute(
                f"SELECT COUNT(DISTINCT run_name) FROM public.metrics WHERE run_name IS NOT NULL AND run_name <> ''{svc_where}",
                tuple(svc_params),
            )
            total_row = cur.fetchone()
            runs_total = int(total_row[0]) if total_row and total_row[0] is not None else 0
            cur.execute(
                f"""
                WITH ranked AS (
                  SELECT service, run_name,
                         COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                         created_at,
                         parsed->'peak_performance'->>'max_rps' AS max_rps,
                         test_type,
                         ROW_NUMBER() OVER (PARTITION BY service ORDER BY created_at DESC) AS rn
                  FROM {schema}.{table}
                  WHERE domain = 'final' AND service IS NOT NULL AND service <> ''{svc_where}
                )
                SELECT service, run_name, verdict, created_at, max_rps, test_type
                FROM ranked WHERE rn <= %s
                ORDER BY service, rn
                """,
                (*svc_params, DASHBOARD_VERDICTS_PER_SERVICE),
            )
            services = _group_service_rows(cur.fetchall())
            cur.execute(
                f"""
                SELECT run_name, service,
                       COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                       created_at, test_type, start_ms, end_ms, COALESCE(run_id, '')
                FROM {schema}.{table}
                WHERE domain = 'final'{svc_where}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (*svc_params, DASHBOARD_RECENT_RUNS),
            )
            for row in cur.fetchall():
                run_name, service, verdict, created_at, test_type, start_ms, end_ms = row[:7]
                run_id = str(row[7] or "").strip() if len(row) > 7 else ""
                recent_runs.append({
                    "run_name": run_name,
                    "service": service,
                    "verdict": verdict,
                    "created_at": created_at.isoformat() if created_at else None,
                    "test_type": test_type or "",
                    "start_ms": int(start_ms) if start_ms is not None else None,
                    "end_ms": int(end_ms) if end_ms is not None else None,
                    "run_id": run_id,
                })
        conn.close()
        return jsonify({
            "last_run": last_run,
            "verdict_counts": verdict_counts,
            "runs_total": runs_total,
            "services": services,
            "recent_runs": recent_runs,
        })
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/demo/seed", methods=["POST"])
def demo_seed():
    """Создаёт синтетический демо-прогон, чтобы изучить интерфейс без внешних систем."""
    try:
        result = seed_demo_run()
    except DemoAlreadyExistsError as e:
        return jsonify({"status": "exists", "message": str(e), "run_name": e.run_name, "report_url": e.report_url}), 409
    except Exception as e:
        return jsonify({"error": f"Не удалось создать демо-прогон: {e}"}), 500
    resp = jsonify({"status": "ok", **result})
    if result.get("area"):
        # Switch the UI to the demo area so the new run is visible right away.
        resp.set_cookie("project_area", str(result["area"]), max_age=PROJECT_AREA_COOKIE_MAX_AGE, samesite="Lax")
    return resp, 200


@dashboard_bp.route("/llm_reports", methods=["GET"])
def llm_reports():
    """Возвращает сохранённые LLM-ответы и метаданные по конкретному запуску."""
    run_name = request.args.get("run_name")
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = _storage_cfg()
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        services_filter = _resolve_services_filter(_active_project_area())
        select_sql = f"""
            SELECT run_name, service, start_ms, end_ms, domain, verdict, text, parsed, scores,
                   sla_verdict, sla_details, system_context, created_at, test_type
            FROM {schema}.{table}
            WHERE run_name = %s AND domain <> 'engineer'{{svc_where}}
            ORDER BY created_at DESC, domain
        """
        rows = []
        with conn, conn.cursor() as cur:
            if services_filter:
                svc_where, svc_params = _services_clause(services_filter)
                cur.execute(select_sql.format(svc_where=svc_where), (run_name, *svc_params))
                rows = cur.fetchall()
            # Fallback: the run may have been created under another area.
            if not rows:
                cur.execute(select_sql.format(svc_where=""), (run_name,))
                rows = cur.fetchall()
        conn.close()
        data = []
        for r in rows:
            parsed_value = r[7]
            if parsed_value is None and isinstance(r[6], str) and r[6].strip():
                try:
                    recovered = parse_llm_analysis_strict(r[6])
                    if recovered is not None:
                        parsed_value = recovered.dict()
                except Exception:
                    parsed_value = None
            data.append(
                {
                    "run_name": r[0],
                    "service": r[1],
                    "start_ms": int(r[2]) if r[2] is not None else None,
                    "end_ms": int(r[3]) if r[3] is not None else None,
                    "domain": r[4],
                    "verdict": r[9] or r[5],
                    "llm_verdict": r[5],
                    "sla_verdict": r[9],
                    "text": r[6],
                    "parsed": parsed_value,
                    "scores": r[8],
                    "sla_details": r[10],
                    "system_context": r[11],
                    "created_at": r[12].isoformat() if r[12] else None,
                    "test_type": r[13] or "",
                }
            )
        return jsonify(data)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/llm_context", methods=["GET"])
def llm_context():
    """JSON context that was sent to the model. Kept off ``/llm_reports`` because it is large."""
    run_name = request.args.get("run_name")
    domain = (request.args.get("domain") or "").strip()
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        params: list = [run_name]
        domain_sql = ""
        if domain:
            domain_sql = " AND domain = %s"
            params.append(domain)
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT domain, context, created_at
                FROM {schema}.{table}
                WHERE run_name = %s AND domain <> 'engineer'{domain_sql}
                ORDER BY created_at DESC, domain
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        conn.close()
        data = []
        seen = set()
        for row in rows:
            if row[0] in seen:
                continue
            seen.add(row[0])
            data.append({
                "run_name": run_name,
                "domain": row[0],
                "context": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/llm_feedback", methods=["GET"])
def llm_feedback_get():
    run_name = request.args.get("run_name")
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        conn = _ts_conn()
        try:
            data = list_llm_feedback(conn, cfg, run_name)
        finally:
            conn.close()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/llm_feedback", methods=["PUT"])
def llm_feedback_put():
    payload = request.get_json(silent=True) or {}
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        conn = _ts_conn()
        try:
            data = upsert_llm_feedback(conn, cfg, payload)
        finally:
            conn.close()
        return jsonify(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


RUNS_SORT_COLUMNS = {
    "run_name": "run_name",
    "service": "service",
    "start_time": "start_time",
    "end_time": "end_time",
    "verdict": "verdict",
    "test_type": "test_type",
    "report_created_at": "report_created_at",
}


@dashboard_bp.route("/runs", methods=["GET"])
def list_runs():
    """Постраничный список запусков активной области.

    Query params: ``q`` (подстрока run_name/service), ``service``, ``verdict``,
    ``test_type`` (точные совпадения), ``offset``, ``limit``, ``sort``, ``dir``.
    Тело ответа — массив; общее число подходящих запусков в заголовке ``X-Total-Count``.
    """
    try:
        cfg = _storage_cfg()
        schema = cfg.get("schema", "public")
        llm_table = cfg.get("llm_table", "llm_reports")
        services_filter = _resolve_services_filter(_active_project_area())
        q = request.args.get("q", "").strip()
        service_filter = (request.args.get("service") or "").strip()
        verdict_filter = (request.args.get("verdict") or "").strip()
        test_type_filter = (request.args.get("test_type") or "").strip()
        offset = int(request.args.get("offset", "0") or 0)
        limit = int(request.args.get("limit", "20") or 20)
        sort_key = (request.args.get("sort", "end_time") or "end_time").lower()
        sort_sql = RUNS_SORT_COLUMNS.get(sort_key, "end_time")
        dir_sql = "ASC" if (request.args.get("dir", "desc") or "desc").lower() == "asc" else "DESC"
        # Runs without a saved report have no creation time; keep them at the end.
        nulls_sql = " NULLS LAST" if sort_key == "report_created_at" else ""

        svc_where, svc_params = _services_clause(services_filter)
        base_params: list = [*svc_params]
        where_q = ""
        if q:
            where_q = " AND (run_name ILIKE %s OR service ILIKE %s)"
            base_params.extend([f"%{q}%", f"%{q}%"])
        final_params: list = [*svc_params]

        filter_clauses: list[str] = []
        filter_params: list = []
        if service_filter:
            filter_clauses.append("service = %s")
            filter_params.append(service_filter)
        if verdict_filter:
            filter_clauses.append("verdict = %s")
            filter_params.append(verdict_filter)
        if test_type_filter:
            filter_clauses.append("test_type = %s")
            filter_params.append(test_type_filter)
        where_filters = (" WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

        runs_cte = f"""
            WITH base AS (
              SELECT run_name,
                     MIN("time") AS start_time,
                     MAX("time") AS end_time,
                     COALESCE(MAX(service), '') AS service,
                     COALESCE(MAX(NULLIF(run_id, '')), '') AS run_id
              FROM public.metrics
              WHERE run_name IS NOT NULL AND run_name <> ''{svc_where}{where_q}
              GROUP BY run_name
            ), final AS (
              SELECT run_name,
                     COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                     verdict AS llm_verdict,
                     sla_verdict,
                     created_at,
                     test_type,
                     MAX(NULLIF(run_id, '')) OVER (PARTITION BY run_name) AS run_id,
                     ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
              FROM {schema}.{llm_table}
              WHERE domain = 'final'{svc_where}
            ), joined AS (
              SELECT b.run_name, b.start_time, b.end_time, b.service,
                     COALESCE(f.verdict, 'Недостаточно данных') AS verdict,
                     f.llm_verdict, f.sla_verdict,
                     f.created_at AS report_created_at,
                     COALESCE(f.test_type, '') AS test_type,
                     COALESCE(NULLIF(f.run_id, ''), NULLIF(b.run_id, ''), '') AS run_id
              FROM base b
              LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
            )
        """

        conn = _ts_conn()
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        with conn, conn.cursor() as cur:
            cur.execute(
                f"{runs_cte} SELECT COUNT(*) FROM joined{where_filters}",
                (*base_params, *final_params, *filter_params),
            )
            count_row = cur.fetchone()
            total = int(count_row[0]) if count_row and count_row[0] is not None else 0
            cur.execute(
                f"""
                {runs_cte}
                SELECT run_name, start_time, end_time, service,
                       verdict, llm_verdict, sla_verdict, report_created_at, test_type, run_id
                FROM joined
                {where_filters}
                ORDER BY {sort_sql} {dir_sql}{nulls_sql}
                OFFSET %s LIMIT %s
                """,
                (*base_params, *final_params, *filter_params, offset, limit),
            )
            rows = cur.fetchall()
            stored = []
            for r in rows:
                run_id = str(r[9] or "").strip() if len(r) > 9 else ""
                if not run_id and r[0] and len(r) > 9:
                    run_id = _assign_run_id(conn, schema, llm_table, str(r[0]), str(r[3] or ""))
                stored.append((*r[:9], run_id))
        conn.close()
        out = []
        for r in stored:
            out.append(
                {
                    "run_name": r[0],
                    "start_time": r[1].isoformat() if r[1] else None,
                    "end_time": r[2].isoformat() if r[2] else None,
                    "service": r[3],
                    "verdict": r[4],
                    "llm_verdict": r[5],
                    "sla_verdict": r[6],
                    "report_created_at": r[7].isoformat() if r[7] else None,
                    "test_type": r[8] or "",
                    "run_id": r[9] or "",
                }
            )
        response = make_response(jsonify(out))
        response.headers["X-Total-Count"] = str(total)
        return response
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


def _text_field(data: dict, *keys: str) -> str:
    """First non-empty string among ``keys``, stripped; values of other types count as missing."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _run_name_taken(run_name: str, service: str) -> bool:
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    llm_table = cfg.get("llm_table", "llm_reports")
    conn = _ts_conn()
    try:
        with conn, conn.cursor() as cur:
            for table in ("public.metrics", f"{schema}.{llm_table}"):
                try:
                    cur.execute("SAVEPOINT run_name_taken")
                    cur.execute(f"SELECT 1 FROM {table} WHERE run_name = %s AND service = %s LIMIT 1", (run_name, service))
                    row = cur.fetchone()
                    cur.execute("RELEASE SAVEPOINT run_name_taken")
                    if row:
                        return True
                except Exception:
                    cur.execute("ROLLBACK TO SAVEPOINT run_name_taken")
    finally:
        conn.close()
    return False


@dashboard_bp.route("/create_report", methods=["POST"])
def create_report():
    """Создаёт задачу формирования отчёта (Confluence и/или веб)."""
    data = request.json or {}
    if not isinstance(data, dict):
        data = {}
    start_str = data.get("start")
    end_str = data.get("end")
    service = _text_field(data, "service")
    project_area = _text_field(data, "project_area", "area", "projectArea")
    test_type = _text_field(data, "test_type")
    use_llm = bool(data.get("use_llm", True))
    save_to_db = bool(data.get("save_to_db", False))
    web_only = bool(data.get("web_only", False))
    run_name = _text_field(data, "run_name")

    if not all([start_str, end_str, service]):
        return jsonify({"status": "error", "message": "Укажите время начала, окончания и сервис"}), 400

    try:
        start = convert_to_timestamp(start_str)
        end = convert_to_timestamp(end_str)
    except ValueError:
        return jsonify({
            "status": "error",
            "message": "Некорректный формат времени: ожидается ISO 8601, например 2025-01-15T10:00 или 2025-01-15T10:00:00+03:00",
        }), 400
    if end <= start:
        return jsonify({"status": "error", "message": "Время окончания должно быть позже времени начала"}), 400
    run_name_error = run_name_problem(run_name) if run_name else None
    if run_name_error:
        return jsonify({"status": "error", "message": run_name_error}), 400

    # Reports run only for services an administrator has configured, and nothing is written to the
    # runtime configuration before that is checked: otherwise a request could create areas and
    # services that everyone sees, and the service name also ends up in file names of the report.
    service_area = _find_area_for_service(service)
    if not service_area:
        return jsonify({"status": "error", "message": f"Сервис '{service}' не найден в настройках"}), 400
    if project_area and project_area != service_area:
        return jsonify({"status": "error", "message": f"Сервис '{service}' принадлежит другой области"}), 400
    project_area = service_area
    _bootstrap_service_configs(service_area, service)

    _metrics_area, service_metrics_cfg = _metrics_service_entry(service)
    if not service_metrics_cfg:
        return jsonify({"status": "error", "message": f"Конфигурация для сервиса '{service}' не найдена"}), 400

    if run_name:
        try:
            taken = _run_name_taken(run_name, service)
        except Exception:
            taken = False
        if taken:
            return jsonify({"status": "error", "message": f"Запуск '{run_name}' уже существует для сервиса '{service}'. Выберите другое имя."}), 400

    # Name the run up front so the job is identifiable in the archive while it runs.
    run_name = run_name or default_run_name()
    job = job_store.create(JOB_KIND_REPORT, run_name=run_name, service=service, message="Инициализация…")
    job_id = job.job_id

    def _progress_cb(msg: str, pct: int | None = None):
        job_store.update(job_id, message=str(msg), progress=pct if isinstance(pct, int) else None)

    def _runner():
        try:
            res = update_report(
                start, end, service,
                use_llm=use_llm,
                save_to_db=save_to_db,
                web_only=web_only,
                run_name=run_name,
                test_type=test_type,
                project_area=project_area,
                progress_callback=_progress_cb,
            )
            result = res if isinstance(res, dict) else {}
            run_id = str(result.get("run_id") or "")
            job_store.update(
                job_id,
                status=JOB_STATUS_DONE,
                progress=100,
                message="Готово",
                report_url=result.get("page_url"),
                page_url=f"/reports/{run_id}" if run_id else None,
                page_id=str(result["page_id"]) if result.get("page_id") is not None else None,
                run_name=str(result.get("run_name") or run_name),
            )
        except Exception as e:  # pragma: no cover
            job_store.update(job_id, status=JOB_STATUS_ERROR, message=_job_failure_message(job_id), error=str(e))

    threading.Thread(target=_runner, daemon=True).start()
    return jsonify({
        "status": "accepted",
        "job_id": job_id,
        "service": service,
        "project_area": project_area,
        "run_name": run_name,
        "message": "Задача принята. Формирование отчёта началось.",
    }), 200


@dashboard_bp.route("/job_status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    """Возвращает состояние фоновой задачи (доступно и после перезапуска сервера)."""
    job = job_store.get(job_id)
    if job is None:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job.to_dict()), 200


@dashboard_bp.route("/jobs", methods=["GET"])
def list_jobs():
    """Недавние фоновые задачи; по умолчанию — выполняющиеся и завершившиеся ошибкой."""
    raw_statuses = (request.args.get("status") or "running,error").split(",")
    statuses = [s.strip() for s in raw_statuses if s.strip()]
    unknown = [s for s in statuses if s not in JOB_STATUSES]
    if unknown:
        return jsonify({"error": f"Недопустимый статус: {', '.join(unknown)}. Допустимо: {', '.join(JOB_STATUSES)}"}), 400
    try:
        limit = int(request.args.get("limit", "20") or 20)
    except ValueError:
        return jsonify({"error": "limit должен быть целым числом"}), 400
    limit = max(1, min(limit, JOBS_LIST_MAX_LIMIT))
    jobs = job_store.list(statuses=statuses, limit=limit)
    return jsonify({"jobs": [j.to_dict() for j in jobs]}), 200


@dashboard_bp.route("/runs/<run_name>", methods=["DELETE"])
def delete_run(run_name: str):
    """Удаляет все артефакты конкретного запуска."""
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        _delete_run_data(run_name)
        return jsonify({"status": "ok", "message": "Отчёт удалён"})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/runs/<run_name>", methods=["PATCH"])
def rename_run(run_name: str):
    """Переименовывает готовый отчёт во всех таблицах хранения."""
    data = request.get_json(silent=True) or {}
    new_name = (data.get("new_run_name") or data.get("run_name") or data.get("name") or "").strip()
    service = (data.get("service") or "").strip() if isinstance(data.get("service"), str) else ""
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    if not new_name:
        return jsonify({"error": "Новое имя отчёта обязательно"}), 400
    try:
        result = _rename_run_data(run_name, new_name)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except FileExistsError as e:
        return jsonify({"error": str(e)}), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500
    renamed = result["run_name"]
    ref = _lookup_report_ref(run_name=renamed, service=service)
    result["run_id"] = (ref or {}).get("run_id") or ""
    result["page_url"] = f"/reports/{result['run_id']}" if result["run_id"] else (
        f"/reports/{service}/{renamed}" if service else f"/reports/{renamed}"
    )
    return jsonify(result), 200


@dashboard_bp.route("/project_areas", methods=["GET"])
def project_areas():
    """Возвращает краткий список доступных областей проекта."""
    try:
        return jsonify(_list_project_areas())
    except Exception:
        return jsonify([])


@dashboard_bp.route("/current_project_area", methods=["GET"])
def current_project_area():
    """Возвращает активную область из cookie (если есть)."""
    return jsonify({"project_area": _active_project_area()})


@dashboard_bp.route("/project_area", methods=["POST"])
def set_project_area():
    """Устанавливает cookie с выбранной областью для дальнейших запросов UI."""
    data = request.get_json(silent=True) or {}
    name = (data.get("project_area") or data.get("service") or data.get("name") or "").strip()
    resp = jsonify({"status": "ok", "project_area": name})
    try:
        resp.set_cookie("project_area", name, max_age=PROJECT_AREA_COOKIE_MAX_AGE, samesite="Lax")
    except Exception:
        pass
    return resp


@dashboard_bp.route("/engineer_summary", methods=["GET"])
def get_engineer_summary():
    """Возвращает последнюю сохранённую заметку инженера по запуску."""
    run_name = request.args.get("run_name")
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = _storage_cfg()
        schema = cfg.get("schema", "public")
        table = cfg.get("engineer_table", "engineer_reports")
        conn = _ts_conn()
        try:
            _ensure_engineer_reports_table(conn, cfg)
        except Exception:
            pass
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT content_html, created_at
                FROM {schema}.{table}
                WHERE run_name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_name,),
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({"run_name": run_name, "content_html": "", "created_at": None})
        return jsonify({"run_name": run_name, "content_html": row[0] or "", "created_at": row[1].isoformat() if row[1] else None})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/engineer_summary", methods=["POST"])
def post_engineer_summary():
    """Сохраняет новую версию заметки инженера по запуску."""
    data = request.get_json(silent=True) or {}
    run_name = (data.get("run_name") or "").strip()
    content_html = data.get("content_html") if isinstance(data.get("content_html"), str) else data.get("content")
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    if not isinstance(content_html, str):
        content_html = ""
    try:
        cfg = _storage_cfg()
        schema = cfg.get("schema", "public")
        engineer_table = cfg.get("engineer_table", "engineer_reports")
        llm_table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        try:
            _ensure_engineer_reports_table(conn, cfg)
        except Exception:
            pass
        service_val = ""
        with conn, conn.cursor() as cur:
            try:
                cur.execute(
                    f"SELECT service FROM {schema}.{llm_table} WHERE run_name=%s AND domain='final' ORDER BY created_at DESC LIMIT 1",
                    (run_name,),
                )
                r = cur.fetchone()
                if r and r[0]:
                    service_val = str(r[0])
            except Exception:
                service_val = ""
            cur.execute(
                f"""
                INSERT INTO {schema}.{engineer_table}
                  (run_id, run_name, service, content_html)
                VALUES
                  (%s, %s, %s, %s)
                """,
                (None, run_name, service_val, content_html),
            )
        conn.close()
        return jsonify({"status": "ok"})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/confluence_publication", methods=["GET"])
def get_confluence_publication():
    """Возвращает страницу Confluence, куда уже опубликован веб-отчёт (если публиковался)."""
    run_name = (request.args.get("run_name") or "").strip()
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        publication = load_publication(run_name)
        if not publication:
            return jsonify({"run_name": run_name, "page_id": None, "page_url": None})
        return jsonify({"run_name": run_name, **publication})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/publish_confluence", methods=["POST"])
def publish_confluence():
    """Запускает фоновую публикацию готового веб-отчёта страницей Confluence."""
    data = request.get_json(silent=True) or {}
    run_name = (data.get("run_name") or "").strip()
    service = (data.get("service") or "").strip()
    source_url = (data.get("source_url") or "").strip()
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400

    job = job_store.create(JOB_KIND_CONFLUENCE, run_name=run_name, service=service, message="Публикация в Confluence…")
    job_id = job.job_id

    def _progress_cb(msg: str, pct: int | None = None):
        job_store.update(job_id, message=str(msg), progress=pct if isinstance(pct, int) else None)

    def _runner():
        try:
            result = publish_report(run_name=run_name, service=service, source_url=source_url, progress_callback=_progress_cb)
            job_store.update(
                job_id,
                status=JOB_STATUS_DONE,
                progress=100,
                message="Готово",
                page_url=result.get("page_url"),
                page_id=str(result["page_id"]) if result.get("page_id") is not None else None,
            )
        except Exception as e:
            job_store.update(job_id, status=JOB_STATUS_ERROR, message=_job_failure_message(job_id), error=str(e))

    threading.Thread(target=_runner, daemon=True).start()
    return jsonify({"status": "accepted", "job_id": job_id, "message": "Публикация в Confluence началась."}), 200
