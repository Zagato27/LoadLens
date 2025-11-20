"""Dashboard and report-related routes."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime

from flask import (
    Blueprint,
    current_app,
    jsonify,
    make_response,
    render_template,
    request,
    send_file,
)

from AI.db_store import _ensure_engineer_reports_table
from settings import CONFIG
from update_page import update_report

from pathlib import Path

from loadlens_app.core import (
    _active_area_prompts,
    _active_metrics_config,
    _active_project_area,
    _available_domain_keys,
    _bootstrap_service_configs,
    _find_area_for_service,
    _list_project_areas,
    _metrics_service_entry,
    _metrics_services_for_area,
    _resolve_services_filter,
    _services_map_for_area,
    _ts_conn,
    convert_to_timestamp,
)
from loadlens_app.jobs import JOBS, JOBS_LOCK

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def home():
    """Рендерит главный дашборд приложения."""
    return render_template("dashboard.html")


@dashboard_bp.route("/reports")
def reports_page():
    """Отображает страницу архива запусков."""
    return render_template("archive.html")


@dashboard_bp.route("/reports/<run_name>")
def reports_page_run(run_name: str):  # noqa: ARG001
    """Отдаёт страницу отчёта; выбор запуска выполняется на клиенте."""
    return render_template("reports.html")


@dashboard_bp.route("/reports/<service>/<run_name>")
def reports_page_service_run(service: str, run_name: str):  # noqa: ARG001
    """Отдаёт страницу отчёта и фиксирует область сервиса в cookie."""
    resp = make_response(render_template("reports.html"))
    try:
        area = _find_area_for_service(service) or service
        resp.set_cookie("project_area", area, max_age=60 * 60 * 24 * 365, samesite="Lax")
    except Exception:
        pass
    return resp


@dashboard_bp.route("/new")
def new_report_page():
    """Отображает форму создания нового отчёта."""
    return render_template("index.html")


@dashboard_bp.route("/assets/logo.png")
def asset_logo():
    """Отдаёт статичный логотип из каталога шаблонов."""
    logo_path = Path(current_app.root_path) / "templates" / "logo.png"
    return send_file(str(logo_path), mimetype="image/png")


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
            payload.append(
                {
                    "id": sid,
                    "title": title,
                    "disabled_domains": disabled,
                }
            )
    elif area_metrics_services:
        for sid in area_metrics_services.keys():
            payload.append(
                {
                    "id": sid,
                    "title": sid,
                    "disabled_domains": [],
                }
            )
    else:
        for area_name, cfg in metrics.items():
            services = cfg.get("services")
            if isinstance(services, dict):
                for sid in services.keys():
                    payload.append({"id": sid, "title": sid, "disabled_domains": []})
    return jsonify(
        {
            "area": area,
            "services": payload,
            "domains": _available_domain_keys(),
        }
    ), 200


@dashboard_bp.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    """Отдаёт данные для главного дашборда (последний запуск и статистику вердиктов)."""
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        last_run = None
        verdict_counts = {"Успешно": 0, "Есть риски": 0, "Провал": 0, "Недостаточно данных": 0}
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, verdict, created_at
                    FROM {schema}.{table}
                    WHERE domain = 'final' AND service = ANY(%s)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (services_filter,),
                )
            else:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, verdict, created_at
                    FROM {schema}.{table}
                    WHERE domain = 'final'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
            r = cur.fetchone()
            if r:
                last_run = {
                    "run_name": r[0],
                    "service": r[1],
                    "start_ms": int(r[2]) if r[2] is not None else None,
                    "end_ms": int(r[3]) if r[3] is not None else None,
                    "verdict": r[4],
                    "created_at": r[5].isoformat() if r[5] else None,
                }

            if services_filter:
                cur.execute(
                    f"""
                    WITH ranked AS (
                      SELECT run_name, verdict, created_at,
                             ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                      FROM {schema}.{table}
                      WHERE domain = 'final' AND service = ANY(%s)
                    )
                    SELECT COALESCE(verdict, 'Недостаточно данных') AS v, COUNT(*) AS cnt
                    FROM ranked
                    WHERE rn = 1
                    GROUP BY v
                    """,
                    (services_filter,),
                )
            else:
                cur.execute(
                    f"""
                    WITH ranked AS (
                      SELECT run_name, verdict, created_at,
                             ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                      FROM {schema}.{table}
                      WHERE domain = 'final'
                    )
                    SELECT COALESCE(verdict, 'Недостаточно данных') AS v, COUNT(*) AS cnt
                    FROM ranked
                    WHERE rn = 1
                    GROUP BY v
                    """
                )
            rows = cur.fetchall()
            for v, cnt in rows:
                if v in verdict_counts:
                    verdict_counts[v] = int(cnt)
                else:
                    verdict_counts["Недостаточно данных"] += int(cnt)
        conn.close()
        return jsonify({"last_run": last_run, "verdict_counts": verdict_counts})
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/llm_reports", methods=["GET"])
def llm_reports():
    """Возвращает сохранённые LLM-ответы и метаданные по конкретному запуску."""
    run_name = request.args.get("run_name")
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, domain, verdict, text, parsed, scores, created_at
                    FROM {schema}.{table}
                    WHERE run_name = %s AND domain <> 'engineer' AND service = ANY(%s)
                    ORDER BY created_at DESC, domain
                    """,
                    (run_name, services_filter),
                )
            else:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, domain, verdict, text, parsed, scores, created_at
                    FROM {schema}.{table}
                    WHERE run_name = %s AND domain <> 'engineer'
                    ORDER BY created_at DESC, domain
                    """,
                    (run_name,),
                )
            rows = cur.fetchall()
        conn.close()
        data = []
        for r in rows:
            data.append(
                {
                    "run_name": r[0],
                    "service": r[1],
                    "start_ms": int(r[2]) if r[2] is not None else None,
                    "end_ms": int(r[3]) if r[3] is not None else None,
                    "domain": r[4],
                    "verdict": r[5],
                    "text": r[6],
                    "parsed": r[7],
                    "scores": r[8],
                    "created_at": r[9].isoformat() if r[9] else None,
                }
            )
        return jsonify(data)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/runs", methods=["GET"])
def list_runs():
    """Возвращает постраничный список запусков с вердиктами."""
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        llm_table = cfg.get("llm_table", "llm_reports")
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        q = request.args.get("q", "").strip()
        offset = int(request.args.get("offset", "0") or 0)
        limit = int(request.args.get("limit", "20") or 20)
        sort = (request.args.get("sort", "end_time") or "end_time").lower()
        direction = (request.args.get("dir", "desc") or "desc").lower()
        sort_map = {
            "run_name": "run_name",
            "service": "service",
            "start_time": "start_time",
            "end_time": "end_time",
        }
        sort_sql = sort_map.get(sort, "end_time")
        dir_sql = "ASC" if direction == "asc" else "DESC"

        params = []
        where_q = ""
        if q:
            where_q = " AND (run_name ILIKE %s OR service ILIKE %s)"
            like = f"%{q}%"
            params.extend([like, like])

        conn = _ts_conn()
        test_type_supported = True
        with conn, conn.cursor() as cur:
            try:
                if services_filter:
                    cur.execute(
                        f"""
                        WITH base AS (
                          SELECT run_name,
                                 MIN("time") AS start_time,
                                 MAX("time") AS end_time,
                                 COALESCE(MAX(service), '') AS service
                          FROM public.metrics
                          WHERE run_name IS NOT NULL AND run_name <> '' AND service = ANY(%s)
                          {where_q}
                          GROUP BY run_name
                        ), final AS (
                          SELECT run_name, verdict, created_at, test_type,
                                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                          FROM {schema}.{llm_table}
                          WHERE domain = 'final' AND service = ANY(%s)
                        )
                        SELECT b.run_name, b.start_time, b.end_time, b.service,
                               f.verdict, f.created_at AS report_created_at, f.test_type
                        FROM base b
                        LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                        ORDER BY {sort_sql} {dir_sql}
                        OFFSET %s LIMIT %s
                        """,
                        (services_filter, *params, services_filter, offset, limit),
                    )
                else:
                    cur.execute(
                        f"""
                        WITH base AS (
                          SELECT run_name,
                                 MIN("time") AS start_time,
                                 MAX("time") AS end_time,
                                 COALESCE(MAX(service), '') AS service
                          FROM public.metrics
                          WHERE run_name IS NOT NULL AND run_name <> ''
                          {where_q}
                          GROUP BY run_name
                        ), final AS (
                          SELECT run_name, verdict, created_at, test_type,
                                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                          FROM {schema}.{llm_table}
                          WHERE domain = 'final'
                        )
                        SELECT b.run_name, b.start_time, b.end_time, b.service,
                               f.verdict, f.created_at AS report_created_at, f.test_type
                        FROM base b
                        LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                        ORDER BY {sort_sql} {dir_sql}
                        OFFSET %s LIMIT %s
                        """,
                        (*params, offset, limit),
                    )
                rows = cur.fetchall()
            except Exception:
                test_type_supported = False
                try:
                    conn.rollback()
                except Exception:
                    pass
                if services_filter:
                    cur.execute(
                        f"""
                        WITH base AS (
                          SELECT run_name,
                                 MIN("time") AS start_time,
                                 MAX("time") AS end_time,
                                 COALESCE(MAX(service), '') AS service
                          FROM public.metrics
                          WHERE run_name IS NOT NULL AND run_name <> '' AND service = ANY(%s)
                          {where_q}
                          GROUP BY run_name
                        ), final AS (
                          SELECT run_name, verdict, created_at,
                                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                          FROM {schema}.{llm_table}
                          WHERE domain = 'final' AND service = ANY(%s)
                        )
                        SELECT b.run_name, b.start_time, b.end_time, b.service,
                               f.verdict, f.created_at AS report_created_at
                        FROM base b
                        LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                        ORDER BY {sort_sql} {dir_sql}
                        OFFSET %s LIMIT %s
                        """,
                        (services_filter, *params, services_filter, offset, limit),
                    )
                else:
                    cur.execute(
                        f"""
                        WITH base AS (
                          SELECT run_name,
                                 MIN("time") AS start_time,
                                 MAX("time") AS end_time,
                                 COALESCE(MAX(service), '') AS service
                          FROM public.metrics
                          WHERE run_name IS NOT NULL AND run_name <> ''
                          {where_q}
                          GROUP BY run_name
                        ), final AS (
                          SELECT run_name, verdict, created_at,
                                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                          FROM {schema}.{llm_table}
                          WHERE domain = 'final'
                        )
                        SELECT b.run_name, b.start_time, b.end_time, b.service,
                               f.verdict, f.created_at AS report_created_at
                        FROM base b
                        LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                        ORDER BY {sort_sql} {dir_sql}
                        OFFSET %s LIMIT %s
                        """,
                        (*params, offset, limit),
                    )
                rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            item = {
                "run_name": r[0],
                "start_time": r[1].isoformat() if r[1] else None,
                "end_time": r[2].isoformat() if r[2] else None,
                "service": r[3],
                "verdict": r[4],
                "report_created_at": (r[5].isoformat() if r[5] else None),
            }
            item["test_type"] = r[6] or "" if test_type_supported else ""
            out.append(item)
        return jsonify(out)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/create_report", methods=["POST"])
def create_report():
    """Создаёт задачу формирования отчёта (Confluence и/или веб)."""
    data = request.json or {}
    start_str = data.get("start")
    end_str = data.get("end")
    service = data.get("service")
    project_area = (data.get("project_area") or data.get("area") or data.get("projectArea") or "").strip()
    test_type = (data.get("test_type") or "").strip()
    use_llm = bool(data.get("use_llm", True))
    save_to_db = bool(data.get("save_to_db", False))
    web_only = bool(data.get("web_only", False))
    run_name = data.get("run_name") if isinstance(data.get("run_name"), str) else None

    if not all([start_str, end_str, service]):
        return jsonify({"status": "error", "message": "Пожалуйста, укажите время начала, окончания и сервис"}), 400

    try:
        start = convert_to_timestamp(start_str)
        end = convert_to_timestamp(end_str)
    except ValueError:
        return jsonify({"status": "error", "message": "Некорректный формат времени. Используйте формат YYYY-MM-DDTHH:MM"}), 400

    preferred_area = project_area or _find_area_for_service(service) or service
    _bootstrap_service_configs(preferred_area, service)

    metrics_area, service_metrics_cfg = _metrics_service_entry(service)
    if not service_metrics_cfg:
        return jsonify({"status": "error", "message": f"Конфигурация для сервиса '{service}' не найдена"}), 400

    service_area = _find_area_for_service(service)
    if not service_area:
        service_area = metrics_area
    if project_area:
        if not service_area:
            return jsonify({"status": "error", "message": f"Сервис '{service}' не привязан к области '{project_area}'"}), 400
        if service_area != project_area:
            return jsonify({"status": "error", "message": f"Сервис '{service}' принадлежит другой области"}), 400
    else:
        project_area = service_area or ""

    run_name = (run_name or "").strip() if isinstance(run_name, str) else ""
    if run_name:
        try:
            cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
            schema = cfg.get("schema", "public")
            llm_table = cfg.get("llm_table", "llm_reports")
            conn = _ts_conn()
            exists = False
            with conn, conn.cursor() as cur:
                try:
                    cur.execute("SELECT 1 FROM public.metrics WHERE run_name = %s AND service = %s LIMIT 1", (run_name, service))
                    if cur.fetchone():
                        exists = True
                except Exception:
                    exists = bool(exists)
                if not exists:
                    try:
                        cur.execute(f"SELECT 1 FROM {schema}.{llm_table} WHERE run_name = %s AND service = %s LIMIT 1", (run_name, service))
                        if cur.fetchone():
                            exists = True
                    except Exception:
                        pass
            conn.close()
            if exists:
                return jsonify({"status": "error", "message": f"Запуск с именем '{run_name}' уже существует для области '{service}'. Выберите другое имя."}), 400
        except Exception:
            pass

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Инициализация…",
            "report_url": None,
            "error": None,
            "service": service,
            "project_area": project_area,
        }

    def _progress_cb(msg: str, pct: int | None = None):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            if isinstance(pct, int):
                job["progress"] = max(job.get("progress", 0), pct)
            job["message"] = str(msg)

    def _runner():
        try:
            res = update_report(
                start,
                end,
                service,
                use_llm=use_llm,
                save_to_db=save_to_db,
                web_only=web_only,
                run_name=run_name,
                test_type=test_type,
                progress_callback=_progress_cb,
            )
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "done"
                    job["progress"] = max(job.get("progress", 0), 100)
                    job["message"] = "Готово"
                    if isinstance(res, dict) and res.get("page_url"):
                        job["report_url"] = res["page_url"]
        except Exception as e:  # pragma: no cover
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(e)

    threading.Thread(target=_runner, daemon=True).start()
    return jsonify({"status": "accepted", "job_id": job_id, "service": service, "message": "Задача принята. Формирование отчёта началось."}), 200


@dashboard_bp.route("/job_status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    """Возвращает состояние фоновой задачи формирования отчёта."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "not_found"}), 404
        return jsonify(job), 200


@dashboard_bp.route("/runs/<run_name>", methods=["DELETE"])
def delete_run(run_name: str):
    """Удаляет все артефакты конкретного запуска."""
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        metrics_table = cfg.get("table", "metrics")
        llm_table = cfg.get("llm_table", "llm_reports")
        engineer_table = cfg.get("engineer_table", "engineer_reports")
        conn = _ts_conn()
        with conn, conn.cursor() as cur:
            try:
                cur.execute(f"DELETE FROM {schema}.{llm_table} WHERE run_name = %s", (run_name,))
            except Exception:
                pass
            try:
                cur.execute(f"DELETE FROM {schema}.{engineer_table} WHERE run_name = %s", (run_name,))
            except Exception:
                pass
            try:
                cur.execute(f"DELETE FROM {schema}.{metrics_table} WHERE run_name = %s", (run_name,))
            except Exception:
                pass
        conn.close()
        return jsonify({"status": "ok", "message": "Отчёт удалён"})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


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
        resp.set_cookie("project_area", name, max_age=60 * 60 * 24 * 365, samesite="Lax")
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
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
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
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
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


