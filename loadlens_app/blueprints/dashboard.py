# SPDX-License-Identifier: Apache-2.0
"""Dashboard and report-related routes."""

from __future__ import annotations

import logging
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
    MANUAL_VERDICT_REASON_MAX,
    MANUAL_VERDICTS,
    MAX_RPS_FIELDS,
    RUN_INDEX_TABLE,
    _ensure_engineer_reports_table,
    _ensure_llm_reports_table,
    _ensure_run_index_table,
    list_llm_feedback,
    load_verdict_state,
    manual_verdicts_cte,
    manual_verdicts_ready,
    max_rps_select_sql,
    save_manual_verdict,
    upsert_llm_feedback,
)
from AI.i18n import t as translate
from AI.scoring import parse_llm_analysis_strict
from AI.verdicts import CODES as VERDICT_CODES, FAIL, RISK, code_sql, exact_code, normalize_report, optional_code, rank_sql, to_code
from settings import CONFIG
# update_report and publish_report run in loadlens_app.tasks, which looks them up here (tests replace them here).
from update_page import _effective_config_for_scope, default_run_name, update_report  # noqa: F401

from loadlens_app import job_runner, tasks
from loadlens_app.confluence_export import load_publication, publish_report  # noqa: F401
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
from loadlens_app.demo_data import DemoAlreadyExistsError, demo_status, seed_demo_run
from loadlens_app.errors import internal_error
from loadlens_app.html_sanitizer import sanitize_html
from loadlens_app.limits import LimitError, check_job_slot, check_report_window, consume_llm
from loadlens_app.jobs import (
    JOB_KIND_CONFLUENCE,
    JOB_KIND_REPORT,
    JOB_STATUSES,
    JobStoreError,
    failure_message,
    job_message,
    job_store,
    message_text,
)
from loadlens_app import response_cache

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)

JOBS_LIST_MAX_LIMIT = 100
JOBS_DISMISS_MAX_IDS = 100
JOB_ID_MAX_LENGTH = 64
ENGINEER_NOTE_MAX_CHARS = 200_000
DASHBOARD_RECENT_RUNS = 5
DASHBOARD_VERDICTS_PER_SERVICE = 5
DASHBOARD_VERDICT_WEEKS = 8
# Verdict codes (AI.verdicts): the keys of verdict_counts and of the weekly counts.
DASHBOARD_VERDICTS = VERDICT_CODES
DASHBOARD_RISKY_VERDICTS = (RISK, FAIL)
PROJECT_AREA_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def _job_failure_message(job_id: str):
    """Failure message (a ``Msg`` in the job's locale) naming the phase that was running."""
    return failure_message(job_store.get(job_id), "job.failed")


def _job_view(job) -> dict:
    """Job state for the client. The raw error (upstream bodies, exception text) is shown to admins only;
    everyone else sees the failed phase in ``message``, rendered in the reader's language from its ref."""
    from loadlens_app.i18n import current_locale

    data = job.to_dict()
    data["message"] = message_text(job, current_locale())
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


def _lookup_report_refs(run_ids: list[str]) -> dict[str, dict]:
    """Existing reports among ``run_ids``: id -> ``{run_id, run_name, service}``, in two queries for any number.

    ``run_index`` answers for reports with metrics, ``llm_reports`` for the rest. A missing table means no
    reports there; a database that does not answer raises, so callers can tell "no report" from "unknown".
    """
    from AI.db_store import RUN_INDEX_TABLE

    ids = list(dict.fromkeys(str(i).strip() for i in run_ids if str(i or "").strip()))
    if not ids:
        return {}
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    llm_table = cfg.get("llm_table", "llm_reports")
    queries = (
        f"SELECT run_id, run_name, COALESCE(service, '') FROM {schema}.{RUN_INDEX_TABLE} WHERE run_id = ANY(%s)",
        f"SELECT DISTINCT ON (run_id) run_id, run_name, COALESCE(service, '') FROM {schema}.{llm_table} "
        f"WHERE run_id = ANY(%s) ORDER BY run_id, created_at DESC",
    )
    found: dict[str, dict] = {}
    conn = _ts_conn()
    try:
        with conn, conn.cursor() as cur:
            for query in queries:
                missing = [i for i in ids if i not in found]
                if not missing:
                    break
                cur.execute("SAVEPOINT report_refs")
                try:
                    cur.execute(query, (missing,))
                    rows = cur.fetchall() or []
                    cur.execute("RELEASE SAVEPOINT report_refs")
                except Exception as exc:
                    cur.execute("ROLLBACK TO SAVEPOINT report_refs")
                    logger.warning("Report lookup by id failed: %s", exc)
                    continue
                for row in rows:
                    report_id, name = str(row[0] or ""), str(row[1] or "")
                    if report_id and name and report_id not in found:
                        found[report_id] = {"run_id": report_id, "run_name": name, "service": str(row[2] or "")}
    finally:
        conn.close()
    return found


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
        return jsonify({"error": translate("api.dashboard.report_not_found")}), 404
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


def _manual_verdict_view(cols) -> dict | None:
    """``{verdict, reason, author, changed_at}`` of joined manual-verdict columns, None without a manual verdict.

    ``verdict`` is a code; rows written before the codes hold a label and are mapped.
    """
    verdict, reason, author, changed_at = (list(cols) + [None] * 4)[:4]
    if not verdict:
        return None
    return {
        "verdict": to_code(verdict),
        "reason": str(reason or ""),
        "author": str(author or ""),
        "changed_at": changed_at.isoformat() if hasattr(changed_at, "isoformat") else (str(changed_at) if changed_at else None),
    }


def _group_service_rows(rows: list) -> list[dict]:
    """Groups (service, run_name, verdict, created_at, max_rps, test_type[, run_id]) rows, newest first per service.

    ``max_rps`` is the peak for step/stress runs, the held RPS for soak and the spike level for spike runs.
    ``history`` keeps every row of the service, newest first; ``recent_verdicts`` (its verdicts) stays for
    older clients.
    """
    grouped: dict[str, dict] = {}
    for row in rows:
        service, run_name, verdict, created_at, max_rps, test_type = row[:6]
        verdict = to_code(verdict)
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

    Counts are keyed by verdict codes; every week of ``weeks`` is present, empty ones with zero counts; an old
    label counts under its code and an unknown verdict as ``no_data``. ``risky_share_30d`` is the share of
    ``risk`` and ``fail`` among the runs of the last 30 days, None without runs.
    """
    buckets = {week: {v: 0 for v in DASHBOARD_VERDICTS} for week in weeks}
    runs_7d = runs_30d = risky_30d = 0
    for week_start, verdict, runs, last_7d, last_30d in rows:
        code = to_code(verdict)
        counts = buckets.get(_dashboard_date(week_start))
        if counts is not None:
            counts[code] += int(runs or 0)
        runs_7d += int(last_7d or 0)
        runs_30d += int(last_30d or 0)
        if code in DASHBOARD_RISKY_VERDICTS:
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
        "verdict": to_code(previous.verdict),
        "created_at": previous.created_at.isoformat() if previous.created_at else None,
    }


# Effective verdict of a final row as a code: the SLA verdict, else the stored one (old labels mapped).
_FINAL_VERDICT_SQL = code_sql("COALESCE(sla_verdict, verdict)")


def _dashboard_cache_key() -> tuple:
    """The dashboard depends on the services of the active area only (not on the user)."""
    return (tuple(_resolve_services_filter(_active_project_area())),)


# Step fields the dashboard load profile draws (LL.charts.stepProfile); the rest of a step row stays in the report.
_DASHBOARD_PROFILE_STEP_FIELDS = (
    "index", "label", "start_iso", "end_iso", "plateau_start_iso", "plateau_end_iso", "rps", "selected", "after_drop", "dip",
)


def _dashboard_profile_parsed_sql(column: str) -> str:
    """The keys of a final row's ``parsed`` the chart markers read (AI/report_markers), as one small JSON object.

    The whole analysis stays in the database: the marker mode (spike, hold, step), the stress RPS drop and the
    load-step table and SLA window of older rows (before ``sla_details`` held them).
    """
    keys = ("spike_analysis", "stability_under_load", "test_profile", "saturation", "load_step_table")
    pairs = ", ".join(f"'{key}', {column}->'{key}'" for key in keys)
    return (
        f"jsonb_build_object({pairs}, "
        f"'deterministic_sla', jsonb_build_object('stable_window', {column}->'deterministic_sla'->'stable_window'))"
    )


def _dashboard_json(value):
    """A JSON column value (a dict or list from the driver, or text); None for broken text."""
    import json

    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _dashboard_checks(sla_details, fallback_checks) -> list[dict]:
    """``[{name, title, threshold, actual, passed}]`` of the SLA checks of a final row (the source of the summary).

    The dashboard draws its figures from them: max RPS against the ``target_rps`` threshold, p95 and the error
    rate against theirs, and one square per check; ``title`` is the check name in the UI language.
    """
    from AI.i18n import has_key

    details = _dashboard_json(sla_details)
    checks = details.get("checks") if isinstance(details, dict) else None
    if not (isinstance(checks, list) and checks):
        checks = _dashboard_json(fallback_checks)
    out = []
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, dict) or not str(check.get("name") or "").strip():
            continue
        passed = check.get("passed")
        name = str(check.get("name")).strip()
        out.append({
            "name": name,
            "title": translate(f"sla.check_title.{name}") if has_key(f"sla.check_title.{name}") else name,
            "threshold": _dashboard_number(check.get("threshold")),
            "actual": _dashboard_number(check.get("actual")),
            "passed": passed if isinstance(passed, bool) else None,
        })
    return out


def _dashboard_profile(sla_details, parsed, fallback_checks) -> dict | None:
    """``{steps, window, markers, target}`` of the last run's load profile; None without a load-step table.

    The steps are ``load_step_table.steps`` saved with the final row, the window its ``target_window`` (a step
    test decided on its target step) or else its ``stable_window``, and the markers those of the report charts
    (AI/report_markers.markers_for: the ceiling, degradation, RPS drop, hold and spike windows) without the step
    ticks and the SLA band, which the window already draws; their labels follow the
    UI language of the request. ``target`` is the threshold of the ``target_rps`` check.
    """
    from AI import report_markers

    details = _dashboard_json(sla_details)
    parsed_doc = _dashboard_json(parsed)
    row = {
        "sla_details": details if isinstance(details, dict) else {},
        "parsed": dict(parsed_doc) if isinstance(parsed_doc, dict) else {},
    }
    fallback = _dashboard_json(fallback_checks)
    if isinstance(fallback, list) and fallback:
        deterministic = row["parsed"].get("deterministic_sla")
        row["parsed"]["deterministic_sla"] = dict(deterministic if isinstance(deterministic, dict) else {}, checks=fallback)
    table = report_markers.embedded_step_table(row) or {}
    steps = [
        {key: step[key] for key in _DASHBOARD_PROFILE_STEP_FIELDS if key in step}
        for step in table.get("steps") or []
        if isinstance(step, dict)
    ]
    if not steps:
        return None
    # A step test decided on its target step shows that step as the SLA zone; its ceiling comes as a marker band.
    window = report_markers.sla_target_window(row) or report_markers.sla_window(row)
    start = report_markers.moment_ms(window.get("start") or window.get("start_iso"))
    end = report_markers.moment_ms(window.get("end") or window.get("end_iso"))
    window_out = {"start": window.get("start") or window.get("start_iso"), "end": window.get("end") or window.get("end_iso")} if (
        start is not None and end is not None and end > start
    ) else None
    markers = []
    for marker in report_markers.markers_for(row):
        if marker.get("kind") == "tick" or (marker.get("kind") == "band" and marker.get("role") == "sla"):
            continue
        if window_out and marker.get("kind") == "band" and (marker.get("t"), marker.get("to")) == (start, end):
            continue  # a hold window equal to the SLA window: drawn once
        markers.append(marker)
    target = next(
        (_dashboard_number(c.get("threshold")) for c in report_markers.sla_checks(row) if str(c.get("name") or "") == "target_rps"),
        None,
    )
    return {"steps": steps, "window": window_out, "markers": markers, "target": target if target and target > 0 else None}


@dashboard_bp.route("/dashboard_data", methods=["GET"])
@response_cache.cached_json(response_cache.DASHBOARD, response_cache.DASHBOARD_TTL_SEC, _dashboard_cache_key)
def dashboard_data():
    """Сводка дашборда: последний запуск, распределение вердиктов, состояние по сервисам, недавние запуски.

    A run with several final rows (a re-analysis) counts once, by its newest row. ``kpi`` holds the run
    counters of the last 7 and 30 days, the share of risky runs and the max RPS of the last run next to the
    previous run of the same service and test type; ``verdict_weeks`` counts verdicts per UTC week.
    ``last_run.checks`` lists its SLA checks and ``last_run.profile`` its load profile (None without steps).
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
        # The manual verdict of a run replaces its automatic one everywhere on the dashboard. Verdicts are
        # codes: old labels in any of the three columns map in SQL, so mixed rows group together.
        manual_cte = manual_verdicts_cte(schema, manual_verdicts_ready(conn, schema))
        final_verdict_sql = _FINAL_VERDICT_SQL
        last_effective_sql = code_sql("COALESCE(mv.verdict, l.effective_verdict)")
        ranked_effective_sql = code_sql("COALESCE(mv.verdict, r.effective_verdict)")
        latest_effective_sql = code_sql("COALESCE(mv.verdict, l.verdict)")
        profile_parsed_sql = _dashboard_profile_parsed_sql("l.parsed_doc")
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
                WITH last AS (
                  SELECT run_name, service, start_ms, end_ms,
                         {final_verdict_sql} AS effective_verdict,
                         verdict AS llm_verdict, sla_verdict, created_at, COALESCE(run_id, '') AS run_id,
                         COALESCE(test_type, '') AS test_type, {DASHBOARD_MAX_RPS_SQL} AS max_rps,
                         sla_details, parsed->'deterministic_sla'->'checks' AS sla_checks, parsed AS parsed_doc
                  FROM {schema}.{table}
                  WHERE domain = 'final'{svc_where}
                  ORDER BY created_at DESC
                  LIMIT 1
                ), {manual_cte}
                SELECT l.run_name, l.service, l.start_ms, l.end_ms,
                       {last_effective_sql}, l.llm_verdict, l.sla_verdict, l.created_at,
                       l.run_id, l.test_type, l.max_rps, l.sla_details, l.sla_checks,
                       mv.verdict, mv.reason, mv.author, mv.changed_at, l.effective_verdict,
                       {profile_parsed_sql}
                FROM last l LEFT JOIN manual_verdicts mv ON mv.run_name = l.run_name
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
                    "verdict": to_code(r[4]),
                    "llm_verdict": optional_code(r[5]),
                    "sla_verdict": optional_code(r[6]),
                    "created_at": r[7].isoformat() if r[7] else None,
                    "run_id": str(r[8] or "").strip() if len(r) > 8 else "",
                    "test_type": str(r[9] or "") if len(r) > 9 else "",
                    "max_rps": _dashboard_number(r[10]) if len(r) > 10 else None,
                    "sla": _dashboard_sla_summary(r[11] if len(r) > 11 else None, r[12] if len(r) > 12 else None),
                    "manual_verdict": _manual_verdict_view(r[13:17]) if len(r) > 16 else None,
                    "auto_verdict": to_code(r[17] if len(r) > 17 else r[4]),
                    "checks": _dashboard_checks(r[11] if len(r) > 11 else None, r[12] if len(r) > 12 else None),
                }
                try:
                    last_run["profile"] = _dashboard_profile(
                        r[11] if len(r) > 11 else None, r[18] if len(r) > 18 else None, r[12] if len(r) > 12 else None,
                    )
                except Exception as exc:
                    # The profile is optional: a malformed step table hides it, the rest of the dashboard stays.
                    logger.warning("dashboard: load profile of %s skipped: %s", last_run["run_name"], exc)
                    last_run["profile"] = None
            cur.execute(
                f"""
                WITH ranked AS (
                  SELECT run_name,
                         {final_verdict_sql} AS effective_verdict,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                  FROM {schema}.{table}
                  WHERE domain = 'final'{svc_where}
                ), {manual_cte}
                SELECT {ranked_effective_sql} AS v, COUNT(*) AS cnt
                FROM ranked r LEFT JOIN manual_verdicts mv ON mv.run_name = r.run_name
                WHERE r.rn = 1 GROUP BY 1
                """,
                tuple(svc_params),
            )
            for v, cnt in cur.fetchall():
                verdict_counts[to_code(v)] += int(cnt)
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
                         {final_verdict_sql} AS verdict,
                         created_at,
                         {DASHBOARD_MAX_RPS_SQL} AS max_rps,
                         test_type,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS run_rn
                  FROM {schema}.{table}
                  WHERE domain = 'final' AND service IS NOT NULL AND service <> ''{svc_where}
                ), {manual_cte}, effective AS (
                  SELECT l.service, l.run_name, l.run_id, {latest_effective_sql} AS verdict, l.created_at,
                         l.max_rps, l.test_type
                  FROM latest l LEFT JOIN manual_verdicts mv ON mv.run_name = l.run_name
                  WHERE l.run_rn = 1
                ), ranked AS (
                  SELECT service, run_name, run_id, verdict, created_at, max_rps, test_type,
                         ROW_NUMBER() OVER (PARTITION BY service ORDER BY created_at DESC) AS rn
                  FROM effective
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
                         {final_verdict_sql} AS verdict,
                         created_at, test_type, start_ms, end_ms, COALESCE(run_id, '') AS run_id,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                  FROM {schema}.{table}
                  WHERE domain = 'final'{svc_where}
                ), {manual_cte}
                SELECT l.run_name, l.service, {latest_effective_sql}, l.created_at, l.test_type,
                       l.start_ms, l.end_ms, l.run_id, mv.verdict AS recent_manual, mv.reason, mv.author, mv.changed_at
                FROM latest l LEFT JOIN manual_verdicts mv ON mv.run_name = l.run_name
                WHERE l.rn = 1
                ORDER BY l.created_at DESC
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
                    "verdict": to_code(verdict),
                    "created_at": created_at.isoformat() if created_at else None,
                    "test_type": test_type or "",
                    "start_ms": int(start_ms) if start_ms is not None else None,
                    "end_ms": int(end_ms) if end_ms is not None else None,
                    "run_id": run_id,
                    "manual_verdict": _manual_verdict_view(row[8:12]) if len(row) > 11 else None,
                })
            cur.execute(
                f"""
                WITH latest AS (
                  SELECT run_name, created_at,
                         {final_verdict_sql} AS verdict,
                         ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                  FROM {schema}.{table}
                  WHERE domain = 'final' AND created_at >= %s{svc_where}
                ), {manual_cte}
                SELECT (date_trunc('week', l.created_at AT TIME ZONE 'UTC'))::date AS week_start,
                       {latest_effective_sql} AS verdict,
                       COUNT(*) AS runs,
                       COUNT(*) FILTER (WHERE l.created_at >= %s) AS runs_7d,
                       COUNT(*) FILTER (WHERE l.created_at >= %s) AS runs_30d
                FROM latest l LEFT JOIN manual_verdicts mv ON mv.run_name = l.run_name
                WHERE l.rn = 1
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
        message = translate("api.dashboard.demo_exists", run_name=e.run_name)
        return jsonify({"status": "exists", "message": message, "run_name": e.run_name, "report_url": e.report_url}), 409
    except Exception as e:
        return internal_error(e, "api.error.demo_failed")
    response_cache.invalidate(response_cache.DASHBOARD)
    resp = jsonify({"status": "ok", **result})
    if result.get("area"):
        # Switch the UI to the demo area so the new run is visible right away.
        resp.set_cookie("project_area", str(result["area"]), max_age=PROJECT_AREA_COOKIE_MAX_AGE, samesite="Lax")
    return resp, 200


@dashboard_bp.route("/demo/status", methods=["GET"])
def demo_status_view():
    """Whether this instance runs with the demo seed (LOADLENS_DEMO_SEED=1) and the demo run is stored.

    The dashboard shows its "demo data" banner only when both hold; a production install answers
    ``{"enabled": false}`` without touching the database.
    """
    return jsonify(demo_status()), 200


@dashboard_bp.route("/llm_reports", methods=["GET"])
def llm_reports():
    """Возвращает сохранённые LLM-ответы и метаданные по конкретному запуску.

    Verdict fields (``verdict``, ``llm_verdict``, ``sla_verdict``, ``auto_verdict``, ``parsed.verdict``,
    ``parsed.deterministic_sla.verdict``, ``manual_verdict.verdict``) are codes; old labels are mapped on read.
    """
    run_name = request.args.get("run_name")
    if not run_name:
        return jsonify({"error": translate("api.required.run_name")}), 400
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
            state = None
            if any(r[4] == "final" for r in rows):
                try:
                    state = load_verdict_state(conn, schema, table, run_name)
                except Exception as exc:  # the report still opens with its automatic verdict
                    logger.warning("llm_reports: manual verdict of %s not read: %s", run_name, exc)
                    conn.rollback()
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
                    "verdict": optional_code(r[9] or r[5]),
                    "llm_verdict": optional_code(r[5]),
                    "sla_verdict": optional_code(r[9]),
                    "text": r[6],
                    "parsed": normalize_report(parsed_value),
                    "scores": r[8],
                    "sla_details": r[10],
                    "system_context": r[11],
                    "created_at": r[12].isoformat() if r[12] else None,
                    "test_type": r[13] or "",
                }
            )
            if r[4] == "final":
                item = data[-1]
                item["auto_verdict"] = item["verdict"]
                item["manual_verdict"] = None
                if state is not None and state["manual"] is not None:
                    item["verdict"] = state["manual"]["verdict"]
                    item["manual_verdict"] = state["manual"]
        return jsonify(data)
    except Exception as e:  # pragma: no cover
        return internal_error(e)


@dashboard_bp.route("/llm_context", methods=["GET"])
def llm_context():
    """JSON context that was sent to the model. Kept off ``/llm_reports`` because it is large."""
    run_name = request.args.get("run_name")
    domain = (request.args.get("domain") or "").strip()
    if not run_name:
        return jsonify({"error": translate("api.required.run_name")}), 400
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
        return jsonify({"error": translate("api.required.run_name")}), 400
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
    # By severity (AI.verdicts.RANK: pass, risk, fail, no_data), not by the text of a label.
    "verdict": "verdict_rank",
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

    cols = list(row) + [None] * max(0, 17 - len(row))
    test_type = str(cols[8] or "")
    return {
        "run_name": cols[0],
        "start_time": cols[1].isoformat() if cols[1] else None,
        "end_time": cols[2].isoformat() if cols[2] else None,
        "service": cols[3],
        "verdict": to_code(cols[4]),
        "llm_verdict": optional_code(cols[5]),
        "sla_verdict": optional_code(cols[6]),
        "report_created_at": cols[7].isoformat() if cols[7] else None,
        "test_type": test_type,
        "test_type_key": normalize_test_type(test_type),
        "run_id": str(cols[9] or "").strip(),
        "project_area": str(cols[10] or ""),
        "max_rps": _runs_float(cols[11]) if cols[11] is not None else None,
        # cols[12] is the total of the page query.
        "manual_verdict": _manual_verdict_view(cols[13:17]),
    }


# ``include`` value of ``/runs`` that adds the SLA summary of every page row (the archive's SLA and p95 columns).
RUNS_INCLUDE_SLA = "sla"


def _runs_include(raw: str) -> set[str]:
    """Known names of the comma-separated ``include`` request param; unknown names are ignored."""
    return {part.strip().lower() for part in str(raw or "").split(",")} & {RUNS_INCLUDE_SLA}


def _runs_sla_checks_sql(schema: str, llm_table: str) -> str:
    """Scalar subquery with the SLA checks of the final row that ``joined`` picked for a run.

    The row is found by run name and creation time (the ``(run_name, created_at)`` index), and only its checks
    leave the database: ``sla_details.checks``, else ``parsed.deterministic_sla.checks`` of rows saved before
    ``sla_details``. Only the outer select of the page uses it, so with ORDER BY ... LIMIT the planner runs it
    for the rows of the page (the jsonb columns never ride through the sorts of all runs).
    """
    return (
        "(SELECT COALESCE(NULLIF(r.sla_details->'checks', '[]'::jsonb), r.parsed->'deterministic_sla'->'checks')"
        f" FROM {schema}.{llm_table} r"
        " WHERE r.domain = 'final' AND r.run_name = joined.run_name AND r.created_at = joined.report_created_at"
        " LIMIT 1)"
    )


def _runs_sla_view(checks) -> dict | None:
    """``{passed, total, p95_ms, p95_passed}`` of a run's SLA checks, None without an evaluated check.

    ``total`` counts the checks with a result (``passed`` true or false), as the report and the dashboard do.
    ``p95_ms`` is the measured value of the first ``p95_latency`` check with a number and ``p95_passed`` its
    result; both None without such a check.
    """
    import json

    if isinstance(checks, (str, bytes)):
        try:
            checks = json.loads(checks)
        except ValueError:
            return None
    items = [c for c in checks if isinstance(c, dict)] if isinstance(checks, list) else []
    passed = sum(1 for c in items if c.get("passed") is True)
    failed = sum(1 for c in items if c.get("passed") is False)
    if not passed + failed:
        return None
    p95_ms = None
    p95_passed = None
    for check in items:
        actual = check.get("actual")
        if check.get("name") != "p95_latency" or isinstance(actual, bool):
            continue
        p95_ms = _runs_float(actual)
        if p95_ms is not None:
            p95_passed = check.get("passed") if isinstance(check.get("passed"), bool) else None
            break
    return {"passed": passed, "total": passed + failed, "p95_ms": p95_ms, "p95_passed": p95_passed}


@dashboard_bp.route("/runs", methods=["GET"])
def list_runs():
    """One page of the runs of the active area.

    Query params: ``q`` (substring of run_name/service), ``service``, ``verdict`` (exact matches; a code
    pass/risk/fail/no_data or an old Russian label), ``test_type`` (step/soak/spike/stress or any synonym),
    ``offset``, ``limit``, ``sort`` (including ``max_rps``; ``verdict`` sorts by severity: pass, risk, fail,
    no_data), ``dir``, ``include`` (``sla``: see below). The body is an array; the number of matching runs is
    in the ``X-Total-Count`` header. Verdicts are codes. ``test_type`` is the stored value, ``test_type_key``
    the normalized type; ``max_rps`` and ``project_area`` come from the final report.

    With ``include=sla`` every item also carries ``sla``: ``{passed, total, p95_ms, p95_passed}`` of the SLA
    checks of the final report (``_runs_sla_view``), null without evaluated checks. The same single statement
    reads it for the rows of the page only.
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
        include_sla = RUNS_INCLUDE_SLA in _runs_include(request.args.get("include", ""))
        try:
            offset = min(RUNS_MAX_OFFSET, max(0, int(request.args.get("offset", "0") or 0)))
            limit = min(RUNS_MAX_LIMIT, max(1, int(request.args.get("limit", "20") or 20)))
        except ValueError:
            return jsonify({"error": translate("api.dashboard.bad_offset_limit")}), 400
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
            # ``verdict`` of ``joined`` is a code; an old label in the URL selects its code, other text nothing.
            filter_clauses.append("verdict = %s")
            filter_params.append(exact_code(verdict_filter) or verdict_filter)
        if test_type_filter:
            # ``test_type`` of ``joined`` is the normalized type, so any stored synonym matches.
            filter_clauses.append("test_type = %s")
            filter_params.append(_runs_test_type_filter(test_type_filter))
        where_filters = (" WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""
        test_type_sql = _runs_test_type_sql("f.test_type", "f.run_name IS NULL")
        max_rps_sql = _runs_max_rps_sql("parsed")
        # Manual verdict of a run with a report, else its SLA or stored verdict; codes, old labels mapped.
        joined_verdict = "COALESCE(CASE WHEN f.run_name IS NOT NULL THEN mv.verdict END, f.verdict)"
        verdict_sql = code_sql(joined_verdict)
        verdict_rank_sql = rank_sql(joined_verdict)

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
                     COALESCE(sla_verdict, verdict) AS verdict,
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
            ), {{manual_cte}}, joined AS (
              SELECT b.run_name, b.start_time, b.end_time, b.service,
                     {verdict_sql} AS verdict,
                     {verdict_rank_sql} AS verdict_rank,
                     CASE WHEN f.run_name IS NOT NULL THEN mv.verdict END AS manual_verdict,
                     mv.reason AS manual_reason, mv.author AS manual_author, mv.changed_at AS manual_changed_at,
                     f.llm_verdict, f.sla_verdict,
                     f.created_at AS report_created_at,
                     COALESCE(f.test_type, '') AS test_type_raw,
                     {test_type_sql} AS test_type,
                     COALESCE(NULLIF(f.run_id, ''), NULLIF(b.run_id, ''), '') AS run_id,
                     COALESCE(f.project_area, '') AS project_area,
                     f.max_rps
              FROM base b
              LEFT JOIN final f ON f.run_name = b.run_name
              LEFT JOIN manual_verdicts mv ON mv.run_name = b.run_name
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
        runs_cte = runs_cte.replace("{manual_cte}", manual_verdicts_cte(schema, manual_verdicts_ready(conn, schema)))
        # Column 17 of a page row with include=sla; the default statement stays as it was.
        sla_select = f", {_runs_sla_checks_sql(schema, llm_table)} AS sla_checks" if include_sla else ""
        with conn, conn.cursor() as cur:
            # One statement: the page plus the number of matching runs (the window runs before OFFSET/LIMIT).
            cur.execute(
                f"""
                {runs_cte}
                SELECT run_name, start_time, end_time, service,
                       verdict, llm_verdict, sla_verdict, report_created_at, test_type_raw, run_id,
                       project_area, max_rps, COUNT(*) OVER () AS total,
                       manual_verdict, manual_reason, manual_author, manual_changed_at{sla_select}
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
        if include_sla:
            for item, row in zip(out, rows):
                item["sla"] = _runs_sla_view(row[17] if len(row) > 17 else None)
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
        return jsonify({"status": "error", "message": translate("api.dashboard.create.missing_fields")}), 400

    try:
        start = convert_to_timestamp(start_str)
        end = convert_to_timestamp(end_str)
    except ValueError:
        return jsonify({
            "status": "error",
            "message": translate("api.dashboard.create.bad_time"),
        }), 400
    if end <= start:
        return jsonify({"status": "error", "message": translate("api.dashboard.create.end_before_start")}), 400
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
            message = translate("api.dashboard.create.run_name_taken", run_name=run_name, service=service)
            return jsonify({"status": "error", "code": "run_name_taken", "message": message}), 400

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
        "message": translate("api.dashboard.create.accepted"),
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
        raise ReportRequestError(translate("api.dashboard.service_not_configured", service=service))
    if project_area and project_area != service_area:
        raise ReportRequestError(translate("api.dashboard.service_other_area", service=service))
    _bootstrap_service_configs(service_area, service)
    _metrics_area, service_metrics_cfg = _metrics_service_entry(service)
    if not service_metrics_cfg:
        raise ReportRequestError(translate("api.dashboard.service_config_missing", service=service))
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
    report_language: str | None = None,
):
    """Checks the limits, creates a report job and starts it; raises :class:`LimitError`.

    The job runs in the Celery worker or in a thread of this process (:mod:`loadlens_app.job_runner`); the
    task is :func:`loadlens_app.tasks.report_task`, which calls ``update_report`` of this module.

    Shared by POST /create_report and the load test watcher, which passes ``run_tag`` and the packs its
    tools tagged, so the load tool queries read only the metrics of its run.

    The report language is fixed here (``appearance.language`` of this process, unless ``report_language`` is
    given): a Celery worker keeps the settings it started with.
    """
    from loadlens_app.i18n import normalize_lang, report_language as current_report_language

    report_lang = normalize_lang(report_language) or current_report_language()
    check_report_window(start, end, _report_step(project_area, service))
    check_job_slot(*job_store.running_counts(owner))
    if use_llm:
        consume_llm(owner, "report")

    # Name the run up front so the job is identifiable in the archive while it runs.
    run_name = run_name or default_run_name()
    job = job_runner.create_job(
        JOB_KIND_REPORT, run_name=run_name, service=service, message=job_message("job.initializing", locale=report_lang), owner=owner,
    )
    job_runner.submit(
        tasks.report_task, job,
        start=start,
        end=end,
        service=service,
        project_area=project_area,
        use_llm=use_llm,
        save_to_db=save_to_db,
        web_only=web_only,
        run_name=run_name,
        test_type=test_type,
        run_tag=run_tag,
        run_filter_packs=list(run_filter_packs) if run_filter_packs is not None else None,
        report_language=report_lang,
    )
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
        message = translate("api.dashboard.jobs.bad_status", unknown=", ".join(unknown), allowed=", ".join(JOB_STATUSES))
        return jsonify({"error": message}), 400
    try:
        limit = int(request.args.get("limit", "20") or 20)
    except ValueError:
        return jsonify({"error": translate("api.dashboard.jobs.bad_limit")}), 400
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
        return jsonify({"error": translate("api.dashboard.jobs.ids_required")}), 400
    if len(job_ids) > JOBS_DISMISS_MAX_IDS:
        return jsonify({"error": translate("api.dashboard.jobs.too_many_ids", max=JOBS_DISMISS_MAX_IDS)}), 400
    if any(not isinstance(job_id, str) or not job_id.strip() or len(job_id) > JOB_ID_MAX_LENGTH for job_id in job_ids):
        return jsonify({"error": translate("api.dashboard.jobs.ids_strings")}), 400
    try:
        dismissed, skipped = job_store.dismiss(job_id.strip() for job_id in job_ids)
    except JobStoreError as exc:
        return jsonify({"error": str(exc), "code": "jobs_dismiss_unavailable"}), 503
    except Exception as exc:
        return internal_error(exc, "api.error.jobs_dismiss_failed")
    g.audit_extra = {"dismissed": len(dismissed), "skipped_running": len(skipped), "job_ids": dismissed[:20]}
    return jsonify({"dismissed": dismissed, "skipped_running": skipped}), 200


@dashboard_bp.route("/runs/<run_name>", methods=["DELETE"])
def delete_run(run_name: str):
    """Удаляет все артефакты конкретного запуска."""
    if not run_name:
        return jsonify({"error": translate("api.required.run_name")}), 400
    try:
        _delete_run_data(run_name)
        response_cache.invalidate(response_cache.DASHBOARD)
        return jsonify({"status": "ok", "message": translate("api.dashboard.run_deleted")})
    except Exception as e:  # pragma: no cover
        return internal_error(e)


@dashboard_bp.route("/runs/<run_name>", methods=["PATCH"])
def rename_run(run_name: str):
    """Переименовывает готовый отчёт во всех таблицах хранения."""
    data = request.get_json(silent=True) or {}
    new_name = (data.get("new_run_name") or data.get("run_name") or data.get("name") or "").strip()
    service = (data.get("service") or "").strip() if isinstance(data.get("service"), str) else ""
    if not run_name:
        return jsonify({"error": translate("api.required.run_name")}), 400
    if not new_name:
        return jsonify({"error": translate("api.dashboard.new_name_required")}), 400
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


def _verdict_author() -> str:
    """Who changed a verdict: the signed-in user; empty when sign-in is off."""
    user = current_auth().user
    return user.username if user is not None else ""


def _verdict_change(run_name: str, verdict: str | None, reason: str):
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    table = cfg.get("llm_table", "llm_reports")
    try:
        conn = _ts_conn()
    except Exception as e:  # pragma: no cover
        return internal_error(e)
    try:
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        with conn:
            state = save_manual_verdict(conn, schema, table, run_name, verdict, reason, _verdict_author())
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # pragma: no cover
        return internal_error(e)
    finally:
        conn.close()
    response_cache.invalidate(response_cache.DASHBOARD)
    history = state.get("history") or []
    previous = history[0]["previous_verdict"] if history else ""
    g.audit_extra = {"run_name": run_name, "verdict": state["verdict"], "previous_verdict": previous,
                     "manual": state["manual"] is not None, "reason": reason[:200]}
    return jsonify(state), 200


@dashboard_bp.route("/runs/<run_name>/verdict", methods=["GET"])
def get_run_verdict(run_name: str):
    """Вердикт прогона: автоматический, действующий и история ручных изменений (новые первыми).

    All verdicts, ``choices`` included, are codes (pass/risk/fail/no_data).
    """
    try:
        cfg = _storage_cfg()
        conn = _ts_conn()
        try:
            try:
                _ensure_llm_reports_table(conn, cfg)
            except Exception:
                pass
            with conn:
                state = load_verdict_state(conn, cfg.get("schema", "public"), cfg.get("llm_table", "llm_reports"), run_name)
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover
        return internal_error(e)
    if state is None:
        return jsonify({"error": translate("server.report.final_not_found", run_name=run_name)}), 404
    state["choices"] = list(MANUAL_VERDICTS)
    state["reason_max"] = MANUAL_VERDICT_REASON_MAX
    return jsonify(state), 200


@dashboard_bp.route("/runs/<run_name>/verdict", methods=["PUT"])
def set_run_verdict(run_name: str):
    """Ставит вердикт прогона вручную: ``{"verdict": "fail", "reason": "..."}``; причина обязательна.

    ``verdict`` is a code (pass/risk/fail/no_data) or a legacy Russian label; the stored value is the code.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": translate("api.common.json_object_expected")}), 400
    verdict = data.get("verdict")
    reason = data.get("reason")
    if not isinstance(verdict, str) or not isinstance(reason, str):
        return jsonify({"error": translate("api.dashboard.verdict_reason_strings")}), 400
    return _verdict_change(run_name, verdict.strip(), reason)


@dashboard_bp.route("/runs/<run_name>/verdict", methods=["DELETE"])
def reset_run_verdict(run_name: str):
    """Возвращает автоматический вердикт; ``{"reason": "..."}`` по желанию."""
    data = request.get_json(silent=True)
    reason = data.get("reason") if isinstance(data, dict) else None
    reason = "" if reason is None else reason
    if not isinstance(reason, str):
        return jsonify({"error": translate("api.dashboard.reason_string")}), 400
    return _verdict_change(run_name, None, reason)


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
        return jsonify({"error": translate("api.required.run_name")}), 400
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
        return jsonify({"error": translate("api.required.run_name")}), 400
    if not isinstance(content_html, str):
        content_html = ""
    if len(content_html) > ENGINEER_NOTE_MAX_CHARS:
        return jsonify({"error": translate("api.dashboard.note_too_long", max=ENGINEER_NOTE_MAX_CHARS)}), 400
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
        return jsonify({"error": translate("api.required.run_name")}), 400
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
        return jsonify({"error": translate("api.required.run_name")}), 400
    owner = _job_owner()
    try:
        check_job_slot(*job_store.running_counts(owner))
    except LimitError as exc:
        return _limit_error(exc)
    # The link on the Confluence page is built here, never taken from the request body.
    ref = _lookup_report_ref(run_name=run_name, service=service) or {}
    source_url = f"{public_base_url()}/reports/{quote(ref.get('run_id') or run_name, safe='')}"

    job = job_runner.create_job(
        JOB_KIND_CONFLUENCE, run_name=run_name, service=service, message=job_message("job.publishing_confluence"), owner=owner,
    )
    # In the worker or in a thread (loadlens_app.tasks.publish_confluence_task calls publish_report of this module).
    job_runner.submit(tasks.publish_confluence_task, job, run_name=run_name, service=service, source_url=source_url)
    return jsonify({"status": "accepted", "job_id": job.job_id, "message": translate("api.dashboard.publish_started")}), 200
