"""Dashboard and report-related routes."""

from __future__ import annotations

import threading
from urllib.parse import quote

from flask import (
    Blueprint,
    g,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
)

from AI.db_store import (
    MAX_RPS_FIELDS,
    RUN_INDEX_TABLE,
    _ensure_engineer_reports_table,
    _ensure_llm_reports_table,
    _ensure_run_index_table,
    list_llm_feedback,
    max_rps_select_sql,
    upsert_llm_feedback,
)
from AI.scoring import parse_llm_analysis_strict
from settings import CONFIG
from update_page import _effective_config_for_scope, default_run_name, update_report

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
    public_base_url,
    run_name_problem,
)
from loadlens_app.auth import Role, current_auth
from loadlens_app.demo_data import DemoAlreadyExistsError, seed_demo_run
from loadlens_app.errors import internal_error
from loadlens_app.html_sanitizer import sanitize_html
from loadlens_app.limits import LimitError, check_job_slot, check_report_window, consume_llm
from loadlens_app.jobs import (
    JOB_KIND_CONFLUENCE,
    JOB_KIND_REPORT,
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR,
    JOB_STATUSES,
    JobStoreError,
    job_store,
)
from loadlens_app import response_cache

dashboard_bp = Blueprint("dashboard", __name__)

JOBS_LIST_MAX_LIMIT = 100
JOBS_DISMISS_MAX_IDS = 100
JOB_ID_MAX_LENGTH = 64
ENGINEER_NOTE_MAX_CHARS = 200_000
DASHBOARD_RECENT_RUNS = 5
DASHBOARD_VERDICTS_PER_SERVICE = 5
DASHBOARD_VERDICT_WEEKS = 8
DASHBOARD_VERDICTS = ("Успешно", "Есть риски", "Провал", "Недостаточно данных")
DASHBOARD_RISKY_VERDICTS = ("Есть риски", "Провал")
PROJECT_AREA_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def _job_failure_message(job_id: str) -> str:
    """Human-readable failure message naming the phase that was running."""
    current = job_store.get(job_id)
    stage = (current.message or "").strip() if current else ""
    return f"Ошибка на этапе «{stage}»" if stage else "Ошибка выполнения задачи"


def _job_view(job) -> dict:
    """Job state for the client. The raw error (upstream bodies, exception text) is shown to admins only;
    everyone else sees the failed phase in ``message``."""
    data = job.to_dict()
    if data.get("error") and current_auth().role < Role.ADMIN:
        data["error"] = None
    return data


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


def _job_owner() -> str:
    """Key the per-user limits count against."""
    user = current_auth().user
    return user.username if user is not None else "local"


def _limit_error(exc: LimitError):
    return jsonify({"status": "error", "error": str(exc), "message": str(exc), "code": exc.code}), exc.status_code


def _report_step(area: str, service: str) -> str:
    """Query step the report will use; the window limit checks the number of points with it."""
    try:
        cfg = _effective_config_for_scope(area, service)
    except Exception:
        cfg = CONFIG
    return str(((cfg.get("default_params") or {}).get("step")) or "")


def _escape_like(text: str) -> str:
    """A search string matched literally inside ILIKE ... ESCAPE."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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
            # A run saved before ids existed has none until the migration runs; it opens by name.
            found_id = str(row[0] or "").strip()
            found_service = service or str(row[2] or "")
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
    """Name and service for the report page addressed by ``run_id`` (or, for old runs, by name)."""
    ref = _lookup_report_ref(run_id=run_id) or _lookup_report_ref(run_name=run_id)
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
        return redirect(f"/reports/{quote(by_name['run_id'], safe='')}", code=302)
    if by_name is not None:
        return _report_page(by_name.get("service") or "")
    return render_template("reports.html")


@dashboard_bp.route("/reports/<service>/<run_name>")
def reports_page_service_run(service: str, run_name: str):
    """Legacy address ``/reports/<service>/<run_name>`` redirects to ``/reports/<run_id>`` (or the name)."""
    ref = _lookup_report_ref(run_name=run_name, service=service)
    if ref is not None:
        target = ref.get("run_id") or ref.get("run_name") or run_name
        return redirect(f"/reports/{quote(target, safe='')}", code=302)
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
    """Groups (service, run_name, verdict, created_at, max_rps, test_type[, run_id]) rows, newest first per service.

    ``max_rps`` is the peak for step/stress runs, the held RPS for soak and the spike level for spike runs.
    ``history`` keeps every row of the service, newest first; ``recent_verdicts`` (its verdicts) stays for
    older clients.
    """
    grouped: dict[str, dict] = {}
    for row in rows:
        service, run_name, verdict, created_at, max_rps, test_type = row[:6]
        run_id = str(row[6] or "").strip() if len(row) > 6 else ""
        created = created_at.isoformat() if created_at else None
        rps = _dashboard_number(max_rps)
        entry = grouped.get(service)
        if entry is None:
            entry = {
                "service": service,
                "last_run": run_name,
                "run_id": run_id,
                "verdict": verdict,
                "created_at": created,
                "test_type": test_type or "",
                "max_rps": rps,
                "recent_verdicts": [],
                "history": [],
            }
            grouped[service] = entry
        entry["recent_verdicts"].append(verdict)
        entry["history"].append({
            "run_name": run_name,
            "run_id": run_id,
            "created_at": created,
            "verdict": verdict,
            "max_rps": rps,
            "test_type": test_type or "",
        })
    return sorted(grouped.values(), key=lambda e: e["created_at"] or "", reverse=True)


# Peak RPS of a final row: the step/stress peak, the held RPS of a soak run or the spike level.
# The stored column, else the same rule over ``parsed`` (AI.db_store.max_rps_from_parsed / max_rps_sql).
DASHBOARD_MAX_RPS_SQL = max_rps_select_sql("parsed")


def _dashboard_number(value) -> float | None:
    """A finite number from a stored value; the LLM writes these fields, so text that is not a number is None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if abs(number) < float("inf") else None


def _dashboard_now():
    """Current UTC time; the KPI periods and verdict weeks are counted from it."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _dashboard_periods(now) -> dict:
    """Week starts (UTC Mondays, oldest first) of the verdict chart and the lower bounds of the KPI periods."""
    from datetime import datetime, timedelta, timezone

    today = now.astimezone(timezone.utc).date()
    current_week = today - timedelta(days=today.weekday())
    weeks = [current_week - timedelta(weeks=DASHBOARD_VERDICT_WEEKS - 1 - i) for i in range(DASHBOARD_VERDICT_WEEKS)]
    first = datetime(weeks[0].year, weeks[0].month, weeks[0].day, tzinfo=timezone.utc)
    since_7d = now - timedelta(days=7)
    since_30d = now - timedelta(days=30)
    return {"weeks": weeks, "since": min(first, since_30d), "since_7d": since_7d, "since_30d": since_30d}


def _dashboard_date(value):
    """``date`` of a week start returned as a date, a datetime or an ISO string."""
    from datetime import date, datetime

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _dashboard_activity(rows: list, weeks: list) -> tuple[list[dict], dict]:
    """Verdict counts per week and the run counters from (week_start, verdict, runs, runs_7d, runs_30d) rows.

    Every week of ``weeks`` is present, empty ones with zero counts; an unknown verdict counts as
    «Недостаточно данных». ``risky_share_30d`` is the share of «Есть риски» and «Провал» among the runs of
    the last 30 days, None without runs.
    """
    buckets = {week: {v: 0 for v in DASHBOARD_VERDICTS} for week in weeks}
    runs_7d = runs_30d = risky_30d = 0
    for week_start, verdict, runs, last_7d, last_30d in rows:
        label = verdict if verdict in DASHBOARD_VERDICTS else "Недостаточно данных"
        counts = buckets.get(_dashboard_date(week_start))
        if counts is not None:
            counts[label] += int(runs or 0)
        runs_7d += int(last_7d or 0)
        runs_30d += int(last_30d or 0)
        if label in DASHBOARD_RISKY_VERDICTS:
            risky_30d += int(last_30d or 0)
    verdict_weeks = [
        {"week_start": week.isoformat(), "counts": counts, "total": sum(counts.values())}
        for week, counts in buckets.items()
    ]
    kpi = {
        "runs_7d": runs_7d,
        "runs_30d": runs_30d,
        "risky_30d": risky_30d,
        "risky_share_30d": round(risky_30d / runs_30d, 4) if runs_30d else None,
    }
    return verdict_weeks, kpi


def _dashboard_sla_summary(sla_details, fallback_checks) -> dict:
    """``{passed, failed, total}`` of the SLA checks of a final row; ``total`` counts the evaluated checks.

    The checks come from ``sla_details.checks``, else from ``parsed.deterministic_sla.checks`` (older rows).
    """
    import json

    def _load(value):
        if isinstance(value, (str, bytes)):
            try:
                return json.loads(value)
            except ValueError:
                return None
        return value

    details = _load(sla_details)
    checks = details.get("checks") if isinstance(details, dict) else None
    if not (isinstance(checks, list) and checks):
        checks = _load(fallback_checks)
    checks = [c for c in checks if isinstance(c, dict)] if isinstance(checks, list) else []
    passed = sum(1 for c in checks if c.get("passed") is True)
    failed = sum(1 for c in checks if c.get("passed") is False)
    return {"passed": passed, "failed": failed, "total": passed + failed}


def _dashboard_previous_run(conn, schema: str, table: str, last: dict, created_at) -> dict | None:
    """Previous run of the last run's service and test type (the baseline conditions of the report)."""
    from AI.db_store import BASELINE_MODE_PREVIOUS, find_previous_run

    service = str(last.get("service") or "").strip()
    run_name = str(last.get("run_name") or "").strip()
    if not service or not run_name or created_at is None:
        return None
    previous = find_previous_run(
        conn, schema, table,
        exclude_run_name=run_name,
        mode=BASELINE_MODE_PREVIOUS,
        service=service,
        project_area=_active_project_area(),
        before=created_at,
        test_type=last.get("test_type") or None,
    )
    if previous is None:
        return None
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COALESCE(run_id, ''), {DASHBOARD_MAX_RPS_SQL} FROM {schema}.{table} "
            "WHERE domain = 'final' AND run_name = %s AND service = %s ORDER BY created_at DESC LIMIT 1",
            (previous.run_name, previous.service or service),
        )
        row = cur.fetchone()
    return {
        "run_name": previous.run_name,
        "run_id": str(row[0] or "").strip() if row else "",
        "max_rps": _dashboard_number(row[1]) if row and len(row) > 1 else None,
        "verdict": previous.verdict,
        "created_at": previous.created_at.isoformat() if previous.created_at else None,
    }


def _dashboard_cache_key() -> tuple:
    """The dashboard depends on the services of the active area only (not on the user)."""
    return (tuple(_resolve_services_filter(_active_project_area())),)


@dashboard_bp.route("/dashboard_data", methods=["GET"])
@response_cache.cached_json(response_cache.DASHBOARD, response_cache.DASHBOARD_TTL_SEC, _dashboard_cache_key)
def dashboard_data():
    """Сводка дашборда: последний запуск, распределение вердиктов, состояние по сервисам, недавние запуски.

    A run with several final rows (a re-analysis) counts once, by its newest row. ``kpi`` holds the run
    counters of the last 7 and 30 days, the share of risky runs and the max RPS of the last run next to the
    previous run of the same service and test type; ``verdict_weeks`` counts verdicts per UTC week.
    """
    try:
        cfg = _storage_cfg()
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        try:
            _ensure_run_index_table(conn, cfg)
        except Exception:
            pass
        services_filter = _resolve_services_filter(_active_project_area())
        svc_where, svc_params = _services_clause(services_filter)
        periods = _dashboard_periods(_dashboard_now())
        last_run = None
        last_created_at = None
        verdict_counts = {v: 0 for v in DASHBOARD_VERDICTS}
        runs_total = 0
        services: list[dict] = []
        recent_runs: list[dict] = []
        activity_rows: list = []
        previous = None
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT run_name, service, start_ms, end_ms,
                       COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS effective_verdict,
                       verdict AS llm_verdict, sla_verdict, created_at, COALESCE(run_id, ''),
                       COALESCE(test_type, ''), {DASHBOARD_MAX_RPS_SQL} AS max_rps,
                       sla_details, parsed->'deterministic_sla'->'checks'
                FROM {schema}.{table}
                WHERE domain = 'final'{svc_where}
                ORDER BY created_at DESC
                LIMIT 1
                """,
                tuple(svc_params),
            )
            r = cur.fetchone()
            if r:
                last_created_at = r[7]
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
                    "test_type": str(r[9] or "") if len(r) > 9 else "",
                    "max_rps": _dashboard_number(r[10]) if len(r) > 10 else None,
                    "sla": _dashboard_sla_summary(r[11] if len(r) > 11 else None, r[12] if len(r) > 12 else None),
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
            # One row per run (run_index), so a plain count; the metrics hypertable is not touched.
            cur.execute(
                f"SELECT COUNT(*) FROM {schema}.{RUN_INDEX_TABLE} WHERE run_name <> ''{svc_where}",
                tuple(svc_params),
            )
            total_row = cur.fetchone()
            runs_total = int(total_row[0]) if total_row and total_row[0] is not None else 0
            cur.execute(
                f"""
                WITH latest AS (
                  SELECT service, run_name, COALESCE(run_id, '') AS run_id,
                         COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                         created_at,
                         {DASHBOARD_MAX_RPS_SQL} AS max_rps,
                         test_type,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS run_rn
                  FROM {schema}.{table}
                  WHERE domain = 'final' AND service IS NOT NULL AND service <> ''{svc_where}
                ), ranked AS (
                  SELECT service, run_name, run_id, verdict, created_at, max_rps, test_type,
                         ROW_NUMBER() OVER (PARTITION BY service ORDER BY created_at DESC) AS rn
                  FROM latest WHERE run_rn = 1
                )
                SELECT service, run_name, verdict, created_at, max_rps, test_type, run_id
                FROM ranked WHERE rn <= %s
                ORDER BY service, rn
                """,
                (*svc_params, DASHBOARD_VERDICTS_PER_SERVICE),
            )
            services = _group_service_rows(cur.fetchall())
            cur.execute(
                f"""
                WITH latest AS (
                  SELECT run_name, service,
                         COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                         created_at, test_type, start_ms, end_ms, COALESCE(run_id, '') AS run_id,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                  FROM {schema}.{table}
                  WHERE domain = 'final'{svc_where}
                )
                SELECT run_name, service, verdict, created_at, test_type, start_ms, end_ms, run_id
                FROM latest WHERE rn = 1
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
            cur.execute(
                f"""
                WITH latest AS (
                  SELECT created_at,
                         COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                  FROM {schema}.{table}
                  WHERE domain = 'final' AND created_at >= %s{svc_where}
                )
                SELECT (date_trunc('week', created_at AT TIME ZONE 'UTC'))::date AS week_start, verdict,
                       COUNT(*) AS runs,
                       COUNT(*) FILTER (WHERE created_at >= %s) AS runs_7d,
                       COUNT(*) FILTER (WHERE created_at >= %s) AS runs_30d
                FROM latest WHERE rn = 1
                GROUP BY 1, 2
                """,
                (periods["since"], *svc_params, periods["since_7d"], periods["since_30d"]),
            )
            activity_rows = cur.fetchall()
            if last_run is not None:
                try:
                    previous = _dashboard_previous_run(conn, schema, table, last_run, last_created_at)
                except Exception as exc:
                    # The comparison is optional: a failed lookup leaves the rest of the dashboard intact.
                    import logging

                    logging.getLogger(__name__).warning("dashboard: previous run lookup failed: %s", exc)
                    previous = None
                    conn.rollback()
        conn.close()
        verdict_weeks, kpi = _dashboard_activity(activity_rows, periods["weeks"])
        if last_run is not None:
            kpi["last"] = {
                "run_name": last_run["run_name"],
                "max_rps": last_run["max_rps"],
                "test_type": last_run["test_type"],
            }
        else:
            kpi["last"] = None
        kpi["previous"] = previous
        return jsonify({
            "last_run": last_run,
            "verdict_counts": verdict_counts,
            "runs_total": runs_total,
            "services": services,
            "recent_runs": recent_runs,
            "kpi": kpi,
            "verdict_weeks": verdict_weeks,
        })
    except Exception as e:  # pragma: no cover - defensive
        return internal_error(e)


@dashboard_bp.route("/demo/seed", methods=["POST"])
def demo_seed():
    """Создаёт синтетический демо-прогон, чтобы изучить интерфейс без внешних систем."""
    try:
        result = seed_demo_run()
    except DemoAlreadyExistsError as e:
        return jsonify({"status": "exists", "message": str(e), "run_name": e.run_name, "report_url": e.report_url}), 409
    except Exception as e:
        return internal_error(e, "Не удалось создать демо-прогон")
    response_cache.invalidate(response_cache.DASHBOARD)
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
        return internal_error(e)


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
        return internal_error(e)


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
        return internal_error(e)


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
        return internal_error(e)


RUNS_MAX_LIMIT = 500
RUNS_MAX_OFFSET = 1_000_000

RUNS_SORT_COLUMNS = {
    "run_name": "run_name",
    "service": "service",
    "start_time": "start_time",
    "end_time": "end_time",
    "verdict": "verdict",
    "test_type": "test_type",
    "report_created_at": "report_created_at",
    "max_rps": "max_rps",
}
# Sort keys whose column may be NULL (no saved report, no load level); such runs stay at the end.
RUNS_NULLS_LAST_SORTS = frozenset({"report_created_at", "max_rps"})
# Final-analysis fields with the run's load level, in priority order (as on the dashboard):
# the peak for step/stress runs, the held RPS for soak runs, the spike level for spike runs.
# Re-export: the rule lives in AI.db_store (MAX_RPS_FIELDS), shared with save_llm_results.
RUNS_MAX_RPS_FIELDS = MAX_RPS_FIELDS


def _runs_sql_literal(value: str) -> str:
    """SQL text literal for a trusted constant; ``%`` is doubled for psycopg2 parameter binding."""
    return "'" + str(value).replace("'", "''").replace("%", "%%") + "'"


def _runs_test_type_sql(column: str, missing_report_sql: str) -> str:
    """SQL expression mapping a stored test type to step/soak/spike/stress.

    Uses the same alias sets as the baseline lookup (``AI.db_store._test_type_filter_values``), so a
    report typed «stability» is found by the «soak» filter and a report saved without a type counts
    as step. Runs without a final report get ''. Unknown types keep their lowercased stored text.
    Only constants from ``AI.test_profile`` are inlined, never request values.
    """
    from AI.db_store import _test_type_filter_values
    from AI.test_profile import TEST_TYPE_ALIASES

    stored = f"LOWER(TRIM(COALESCE({column}, '')))"
    branches = []
    for key in TEST_TYPE_ALIASES:
        values = ", ".join(_runs_sql_literal(v) for v in _test_type_filter_values(key))
        branches.append(f"WHEN {stored} IN ({values}) THEN {_runs_sql_literal(key)}")
    return f"CASE WHEN {missing_report_sql} THEN '' {' '.join(branches)} ELSE {stored} END"


def _runs_test_type_filter(raw: str) -> str:
    """Value of the normalized ``test_type`` column that the ``test_type`` request param selects."""
    from AI.test_profile import normalize_test_type

    return normalize_test_type(raw) or raw.strip().lower()


def _runs_max_rps_sql(parsed_column: str) -> str:
    """Load level of a final row: the stored ``max_rps`` column, else the first numeric value of
    ``RUNS_MAX_RPS_FIELDS`` in the parsed final analysis, else NULL.

    The column is filled on save and by ``loadlens_app.migrations.backfill_max_rps``; the parse stays
    for rows without it. Text that is not a plain number (the LLM sometimes writes «~1200») is skipped
    instead of failing the cast; the exponent is bounded so the numeric cast cannot overflow.
    """
    return max_rps_select_sql(parsed_column)


def _runs_float(value) -> float | None:
    """Finite float of a numeric DB value, else None (JSON has no Infinity)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _runs_item(row) -> dict:
    """One ``/runs`` item from (run_name, start_time, end_time, service, verdict, llm_verdict,
    sla_verdict, report_created_at, test_type, run_id, project_area, max_rps)."""
    from AI.test_profile import normalize_test_type

    cols = list(row) + [None] * max(0, 12 - len(row))
    test_type = str(cols[8] or "")
    return {
        "run_name": cols[0],
        "start_time": cols[1].isoformat() if cols[1] else None,
        "end_time": cols[2].isoformat() if cols[2] else None,
        "service": cols[3],
        "verdict": cols[4],
        "llm_verdict": cols[5],
        "sla_verdict": cols[6],
        "report_created_at": cols[7].isoformat() if cols[7] else None,
        "test_type": test_type,
        "test_type_key": normalize_test_type(test_type),
        "run_id": str(cols[9] or "").strip(),
        "project_area": str(cols[10] or ""),
        "max_rps": _runs_float(cols[11]) if cols[11] is not None else None,
    }


@dashboard_bp.route("/runs", methods=["GET"])
def list_runs():
    """Постраничный список запусков активной области.

    Query params: ``q`` (подстрока run_name/service), ``service``, ``verdict`` (точные совпадения),
    ``test_type`` (step/soak/spike/stress или любой их синоним), ``offset``, ``limit``, ``sort``
    (включая ``max_rps``), ``dir``. Тело ответа — массив; общее число подходящих запусков
    в заголовке ``X-Total-Count``. ``test_type`` в ответе — сохранённое значение,
    ``test_type_key`` — нормализованный тип; ``max_rps`` и ``project_area`` берутся из итогового отчёта.
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
        try:
            offset = min(RUNS_MAX_OFFSET, max(0, int(request.args.get("offset", "0") or 0)))
            limit = min(RUNS_MAX_LIMIT, max(1, int(request.args.get("limit", "20") or 20)))
        except ValueError:
            return jsonify({"error": "offset и limit должны быть целыми числами"}), 400
        sort_key = (request.args.get("sort", "end_time") or "end_time").lower()
        if sort_key not in RUNS_SORT_COLUMNS:
            sort_key = "end_time"
        sort_sql = RUNS_SORT_COLUMNS[sort_key]
        dir_sql = "ASC" if (request.args.get("dir", "desc") or "desc").lower() == "asc" else "DESC"
        # Runs without a saved report have no creation time or load level; keep them at the end.
        nulls_sql = " NULLS LAST" if sort_key in RUNS_NULLS_LAST_SORTS else ""
        # Equal sort values (same service, verdict...) need a fixed order, or pages overlap.
        tiebreak_sql = "" if sort_sql == "run_name" else ", run_name ASC"

        svc_where, svc_params = _services_clause(services_filter)
        base_params: list = [*svc_params]
        where_q = ""
        if q:
            where_q = " AND (run_name ILIKE %s ESCAPE '\\' OR service ILIKE %s ESCAPE '\\')"
            pattern = f"%{_escape_like(q)}%"
            base_params.extend([pattern, pattern])
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
            # ``test_type`` of ``joined`` is the normalized type, so any stored synonym matches.
            filter_clauses.append("test_type = %s")
            filter_params.append(_runs_test_type_filter(test_type_filter))
        where_filters = (" WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""
        test_type_sql = _runs_test_type_sql("f.test_type", "f.run_name IS NULL")
        max_rps_sql = _runs_max_rps_sql("parsed")

        # ``base`` reads run_index (one row per run with metrics) instead of aggregating the metrics hypertable.
        # ``final`` keeps the newest final row of each run; DISTINCT ON (not a ROW_NUMBER filter) lets the
        # planner estimate one row per run and hash-join it instead of a nested loop over all pairs.
        runs_cte = f"""
            WITH base AS (
              SELECT run_name, start_time, end_time, service, run_id
              FROM {schema}.{RUN_INDEX_TABLE}
              WHERE run_name <> ''{svc_where}{where_q}
            ), final AS (
              SELECT DISTINCT ON (run_name) run_name,
                     COALESCE(sla_verdict, verdict, 'Недостаточно данных') AS verdict,
                     verdict AS llm_verdict,
                     sla_verdict,
                     created_at,
                     test_type,
                     COALESCE(project_area, '') AS project_area,
                     {max_rps_sql} AS max_rps,
                     MAX(NULLIF(run_id, '')) OVER (PARTITION BY run_name) AS run_id
              FROM {schema}.{llm_table}
              WHERE domain = 'final'{svc_where}
              ORDER BY run_name, created_at DESC
            ), joined AS (
              SELECT b.run_name, b.start_time, b.end_time, b.service,
                     COALESCE(f.verdict, 'Недостаточно данных') AS verdict,
                     f.llm_verdict, f.sla_verdict,
                     f.created_at AS report_created_at,
                     COALESCE(f.test_type, '') AS test_type_raw,
                     {test_type_sql} AS test_type,
                     COALESCE(NULLIF(f.run_id, ''), NULLIF(b.run_id, ''), '') AS run_id,
                     COALESCE(f.project_area, '') AS project_area,
                     f.max_rps
              FROM base b
              LEFT JOIN final f ON f.run_name = b.run_name
            )
        """

        conn = _ts_conn()
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        try:
            _ensure_run_index_table(conn, cfg)
        except Exception:
            pass
        with conn, conn.cursor() as cur:
            # One statement: the page plus the number of matching runs (the window runs before OFFSET/LIMIT).
            cur.execute(
                f"""
                {runs_cte}
                SELECT run_name, start_time, end_time, service,
                       verdict, llm_verdict, sla_verdict, report_created_at, test_type_raw, run_id,
                       project_area, max_rps, COUNT(*) OVER () AS total
                FROM joined
                {where_filters}
                ORDER BY {sort_sql} {dir_sql}{nulls_sql}{tiebreak_sql}
                OFFSET %s LIMIT %s
                """,
                (*base_params, *final_params, *filter_params, offset, limit),
            )
            rows = cur.fetchall()
            if rows:
                total = int(rows[0][12] or 0)
            elif offset:
                # A page past the end has no row to carry the total; the archive needs it to step back.
                cur.execute(
                    f"{runs_cte} SELECT COUNT(*) FROM joined{where_filters}",
                    (*base_params, *final_params, *filter_params),
                )
                count_row = cur.fetchone()
                total = int(count_row[0]) if count_row and count_row[0] is not None else 0
            else:
                total = 0
        conn.close()
        out = [_runs_item(r) for r in rows]
        response = make_response(jsonify(out))
        response.headers["X-Total-Count"] = str(total)
        return response
    except Exception as e:  # pragma: no cover
        return internal_error(e)


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

    try:
        project_area = resolve_report_target(service, project_area)
    except ReportRequestError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    if run_name:
        try:
            taken = _run_name_taken(run_name, service)
        except Exception:
            taken = False
        if taken:
            return jsonify({"status": "error", "message": f"Запуск '{run_name}' уже существует для сервиса '{service}'. Выберите другое имя."}), 400

    try:
        job = start_report_job(
            start, end, service, project_area,
            use_llm=use_llm,
            save_to_db=save_to_db,
            web_only=web_only,
            run_name=run_name,
            test_type=test_type,
            owner=_job_owner(),
        )
    except LimitError as exc:
        return _limit_error(exc)
    return jsonify({
        "status": "accepted",
        "job_id": job.job_id,
        "service": service,
        "project_area": project_area,
        "run_name": job.run_name,
        "message": "Задача принята. Формирование отчёта началось.",
    }), 200


class ReportRequestError(ValueError):
    """A report cannot be started for this service; the message is for the user."""


def resolve_report_target(service: str, project_area: str = "") -> str:
    """Area of a configured service, with its configs bootstrapped; raises :class:`ReportRequestError`.

    Reports run only for services an administrator has configured, and nothing is written to the
    runtime configuration before that is checked: otherwise a request could create areas and
    services that everyone sees, and the service name also ends up in file names of the report.
    """
    service_area = _find_area_for_service(service)
    if not service_area:
        raise ReportRequestError(f"Сервис '{service}' не найден в настройках")
    if project_area and project_area != service_area:
        raise ReportRequestError(f"Сервис '{service}' принадлежит другой области")
    _bootstrap_service_configs(service_area, service)
    _metrics_area, service_metrics_cfg = _metrics_service_entry(service)
    if not service_metrics_cfg:
        raise ReportRequestError(f"Конфигурация для сервиса '{service}' не найдена")
    return service_area


def start_report_job(
    start: int,
    end: int,
    service: str,
    project_area: str,
    *,
    use_llm: bool,
    save_to_db: bool,
    web_only: bool,
    run_name: str,
    test_type: str,
    owner: str,
    run_tag: str | None = None,
    run_filter_packs: list[str] | None = None,
):
    """Checks the limits, creates a report job and runs it in a thread; raises :class:`LimitError`.

    Shared by POST /create_report and the load test watcher, which passes ``run_tag`` and the packs its
    tools tagged, so the load tool queries read only the metrics of its run.
    """
    check_report_window(start, end, _report_step(project_area, service))
    check_job_slot(*job_store.running_counts(owner))
    if use_llm:
        consume_llm(owner, "report")

    # Name the run up front so the job is identifiable in the archive while it runs.
    run_name = run_name or default_run_name()
    job = job_store.create(JOB_KIND_REPORT, run_name=run_name, service=service, message="Инициализация…", owner=owner)
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
                run_tag=run_tag,
                run_filter_packs=run_filter_packs,
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
    return job


@dashboard_bp.route("/job_status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    """Возвращает состояние фоновой задачи (доступно и после перезапуска сервера)."""
    job = job_store.get(job_id)
    if job is None:
        return jsonify({"status": "not_found"}), 404
    return jsonify(_job_view(job)), 200


def _jobs_cache_key() -> tuple | None:
    """Statuses, limit and whether raw errors are shown (admins only); ``None`` lets the view reject bad input."""
    raw_statuses = (request.args.get("status") or "running,error").split(",")
    statuses = tuple(s.strip() for s in raw_statuses if s.strip())
    if any(s not in JOB_STATUSES for s in statuses):
        return None
    try:
        limit = int(request.args.get("limit", "20") or 20)
    except ValueError:
        return None
    return statuses, max(1, min(limit, JOBS_LIST_MAX_LIMIT)), current_auth().role >= Role.ADMIN


@dashboard_bp.route("/jobs", methods=["GET"])
@response_cache.cached_json(response_cache.JOBS, response_cache.JOBS_TTL_SEC, _jobs_cache_key)
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
    return jsonify({"jobs": [_job_view(j) for j in jobs]}), 200


@dashboard_bp.route("/jobs/dismiss", methods=["POST"])
def dismiss_jobs():
    """Скрывает завершившиеся задачи из списка; выполняющиеся остаются.

    Тело: ``{"job_ids": [...]}`` — идентификаторы строк, которые видел пользователь. Задачи остаются
    в таблице и открываются по ссылке. Ответ: ``dismissed`` и ``skipped_running`` (списки идентификаторов).
    """
    body = request.get_json(silent=True)
    job_ids = body.get("job_ids") if isinstance(body, dict) else None
    if not isinstance(job_ids, list) or not job_ids:
        return jsonify({"error": "Укажите job_ids — непустой список идентификаторов задач"}), 400
    if len(job_ids) > JOBS_DISMISS_MAX_IDS:
        return jsonify({"error": f"Не больше {JOBS_DISMISS_MAX_IDS} задач за один запрос"}), 400
    if any(not isinstance(job_id, str) or not job_id.strip() or len(job_id) > JOB_ID_MAX_LENGTH for job_id in job_ids):
        return jsonify({"error": "job_ids должен содержать строковые идентификаторы задач"}), 400
    try:
        dismissed, skipped = job_store.dismiss(job_id.strip() for job_id in job_ids)
    except JobStoreError as exc:
        return jsonify({"error": str(exc), "code": "jobs_dismiss_unavailable"}), 503
    except Exception as exc:
        return internal_error(exc, "Не удалось очистить список задач")
    g.audit_extra = {"dismissed": len(dismissed), "skipped_running": len(skipped), "job_ids": dismissed[:20]}
    return jsonify({"dismissed": dismissed, "skipped_running": skipped}), 200


@dashboard_bp.route("/runs/<run_name>", methods=["DELETE"])
def delete_run(run_name: str):
    """Удаляет все артефакты конкретного запуска."""
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        _delete_run_data(run_name)
        response_cache.invalidate(response_cache.DASHBOARD)
        return jsonify({"status": "ok", "message": "Отчёт удалён"})
    except Exception as e:  # pragma: no cover
        return internal_error(e)


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
        return internal_error(e)
    response_cache.invalidate(response_cache.DASHBOARD)
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
        return jsonify({"run_name": run_name, "content_html": sanitize_html(row[0] or ""), "created_at": row[1].isoformat() if row[1] else None})
    except Exception as e:  # pragma: no cover
        return internal_error(e)


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
    if len(content_html) > ENGINEER_NOTE_MAX_CHARS:
        return jsonify({"error": f"Заметка длиннее {ENGINEER_NOTE_MAX_CHARS} символов"}), 400
    # Stored as formatting-only HTML: viewers and the Confluence export get no scripts or forms.
    content_html = sanitize_html(content_html)
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
        return internal_error(e)


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
        return internal_error(e)


@dashboard_bp.route("/publish_confluence", methods=["POST"])
def publish_confluence():
    """Запускает фоновую публикацию готового веб-отчёта страницей Confluence."""
    data = request.get_json(silent=True) or {}
    run_name = (data.get("run_name") or "").strip()
    service = (data.get("service") or "").strip()
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    owner = _job_owner()
    try:
        check_job_slot(*job_store.running_counts(owner))
    except LimitError as exc:
        return _limit_error(exc)
    # The link on the Confluence page is built here, never taken from the request body.
    ref = _lookup_report_ref(run_name=run_name, service=service) or {}
    source_url = f"{public_base_url()}/reports/{quote(ref.get('run_id') or run_name, safe='')}"

    job = job_store.create(JOB_KIND_CONFLUENCE, run_name=run_name, service=service, message="Публикация в Confluence…", owner=owner)
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
