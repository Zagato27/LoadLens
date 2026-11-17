# SPDX-License-Identifier: Apache-2.0
"""Pages and data of releases: the release list, the page of a release, the trend by release and its Confluence page.

The marks of the runs (release, accepted run, baseline) and ``GET /releases_data`` (the names for pickers) belong to
the ``releases`` blueprint (loadlens_app/blueprints/releases.py); this one shows releases as a whole. Everything is
read for the active project area: its services are the required ones (``untested`` lists those without an accepted
run) and only their runs are shown. Data functions live in AI/release_data.py; nothing here is cached.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote

import psycopg2
from flask import Blueprint, abort, g, jsonify, render_template, request

from settings import CONFIG
from AI.i18n import t as translate
from AI.regression import thresholds_from_sla
from AI.release_data import (
    RELEASE_NOTE_MAX,
    TREND_LIMIT,
    TREND_LIMIT_MAX,
    delete_release as delete_release_row,
    ensure_release_tables,
    load_publication,
    load_release,
    marked_runs,
    release_overview,
    release_view,
    save_release,
    trend_points,
)
from AI.release_marks import ReleaseMarkError, mark_test_type, validate_release_name
from loadlens_app import job_runner, jira_links, tasks
from loadlens_app.auth import current_auth
from loadlens_app.core import (
    _active_metrics_config,
    _active_project_area,
    _find_area_for_service,
    _metrics_services_for_area,
    _resolve_services_filter,
    _services_map_for_area,
    _ts_conn,
    public_base_url,
)
from loadlens_app.forecast import database_unavailable
from loadlens_app.jobs import JOB_KIND_RELEASE_CONFLUENCE, failure_message, job_message, job_store
from loadlens_app.limits import LimitError, check_job_slot
# publish_release runs in loadlens_app.tasks, which looks it up here (tests replace it here).
from loadlens_app.release_confluence import publish_release  # noqa: F401

release_pages_bp = Blueprint("release_pages", __name__)

ThresholdsOf = Callable[[str], Optional[Mapping[str, Any]]]


class NotAReleaseName(ValueError):
    """Not a release name (the page and the data answer 404)."""


# ---- scope of the project area -------------------------------------------------------------------------------


def _tables() -> Tuple[str, str]:
    cfg = (CONFIG.get("storage") or {}).get("timescale") or {}
    cfg = cfg if isinstance(cfg, dict) else {}
    return str(cfg.get("schema") or "public"), str(cfg.get("llm_table") or "llm_reports")


def area_services(area: str) -> List[str]:
    """Services of the project area as ``GET /services`` lists them; without an area every configured service."""
    services_map = _services_map_for_area(area)
    if services_map:
        return list(services_map)
    metrics_services = _metrics_services_for_area(area)
    if metrics_services:
        return list(metrics_services)
    if area:
        return []
    names: List[str] = []
    for cfg in (_active_metrics_config() or {}).values():
        services = cfg.get("services") if isinstance(cfg, dict) else None
        if isinstance(services, dict):
            names.extend(str(name) for name in services)
    return list(dict.fromkeys(names))


def release_scope(area: str) -> Tuple[Optional[List[str]], List[str]]:
    """(services whose runs are shown, required services); without an area runs of any service are shown."""
    required = area_services(area)
    if not area:
        return None, required
    return list(_resolve_services_filter(area) or required), required


def thresholds_for(area: str) -> ThresholdsOf:
    """Regression thresholds of a service's effective SLA (``sla.regression_*``), read once per service."""
    cache: Dict[str, Any] = {}

    def of(service: str) -> Optional[Mapping[str, Any]]:
        if service not in cache:
            from update_page import _effective_config_for_scope

            scope = area or _find_area_for_service(service) or ""
            try:
                sla = _effective_config_for_scope(scope, service or None).get("sla")
            except Exception:
                sla = None
            cache[service] = thresholds_from_sla(sla)
        return cache[service]

    return of


def _release_name(raw: str) -> str:
    try:
        return validate_release_name(raw)
    except ReleaseMarkError as exc:
        raise NotAReleaseName(str(exc)) from exc


def _not_found(name: str):
    return jsonify({"error": translate("server.release.not_found", name=name), "code": "not_found"}), 404


def _db_error(exc: psycopg2.Error):
    failure = database_unavailable(exc)
    return jsonify({"error": str(failure), "code": failure.code}), failure.status_code


def release_payload(name: str, area: str) -> Optional[Dict[str, Any]]:
    """The release page data of the area (``GET /releases_data/<name>``), None for an unknown release."""
    schema, llm_table = _tables()
    services, required = release_scope(area)
    conn = _ts_conn()
    try:
        ensure_release_tables(conn, schema)
        view = release_view(
            conn, schema, llm_table, name,
            services=services, required=required, thresholds_of=thresholds_for(area), project_area=area or None,
        )
        publication = load_publication(conn, schema, name, area) if view is not None else None
    finally:
        conn.close()
    if view is None:
        return None
    view["services_required"] = len(required)
    view["jira"] = jira_links.links(jira_links.jira_config(), name, view.get("ticket") or "")
    view["publication"] = (
        {key: publication[key] for key in ("page_id", "page_url", "updated_at")} if publication else None
    )
    return view


def trend_payload(service: str, test_type: str, limit: int, area: str) -> List[Dict[str, Any]]:
    """Points of the trend of a service of the area (none for a service outside it)."""
    services, _required = release_scope(area)
    if services is not None and service not in services:
        return []
    schema, llm_table = _tables()
    conn = _ts_conn()
    try:
        return trend_points(conn, schema, llm_table, service, test_type, limit)
    finally:
        conn.close()


def _release_exists(name: str, area: str) -> bool:
    schema, llm_table = _tables()
    services, _required = release_scope(area)
    conn = _ts_conn()
    try:
        ensure_release_tables(conn, schema)
        if load_release(conn, schema, name) is not None:
            return True
        return bool(marked_runs(conn, schema, llm_table, release=name, services=services))
    finally:
        conn.close()


def _owner() -> str:
    user = current_auth().user
    return user.username if user is not None else "local"


def release_url(name: str) -> str:
    return "/releases/" + quote(name, safe="")


# ---- pages ---------------------------------------------------------------------------------------------------


@release_pages_bp.route("/releases")
def releases_page():
    """The release list with the trend of a chosen service."""
    return render_template("releases.html")


@release_pages_bp.route("/releases/<path:name>")
def release_page(name: str):
    """The page of one release."""
    try:
        release = _release_name(name)
    except NotAReleaseName:
        abort(404)
    return render_template("release.html", release_name=release)


# ---- data ----------------------------------------------------------------------------------------------------


@release_pages_bp.route("/releases_overview", methods=["GET"])
def releases_overview():
    """Releases of the active project area, newest first, with their status and verdict counts."""
    area = _active_project_area() or ""
    services, required = release_scope(area)
    schema, llm_table = _tables()
    try:
        conn = _ts_conn()
        try:
            ensure_release_tables(conn, schema)
            items = release_overview(conn, schema, llm_table, services=services, required=required)
        finally:
            conn.close()
    except psycopg2.Error as exc:
        return _db_error(exc)
    return jsonify(items)


@release_pages_bp.route("/releases_data/<path:name>", methods=["GET"])
def get_release(name: str):
    """A release: attributes, Jira links, status, rows per service and test type, untested services, publication."""
    try:
        release = _release_name(name)
    except NotAReleaseName:
        return _not_found(name)
    try:
        payload = release_payload(release, _active_project_area() or "")
    except psycopg2.Error as exc:
        return _db_error(exc)
    if payload is None:
        return _not_found(release)
    return jsonify(payload)


@release_pages_bp.route("/releases_data/<path:name>", methods=["PUT"])
def update_release(name: str):
    """Sets the Jira ticket and the note of a release: ``{"ticket"?: "PROJ-123", "note"?: "..."}``."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": translate("api.common.json_object_expected"), "code": "bad_request"}), 400
    try:
        release = _release_name(name)
    except NotAReleaseName as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    ticket: Optional[str] = None
    note: Optional[str] = None
    try:
        if "ticket" in data:
            ticket = jira_links.normalize_ticket(data.get("ticket"))
    except jira_links.TicketError as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    if "note" in data:
        raw = data.get("note")
        if raw is not None and not isinstance(raw, str):
            return jsonify({"error": translate("server.release.note_string"), "code": "bad_request"}), 400
        note = (raw or "").strip()
        if len(note) > RELEASE_NOTE_MAX:
            return jsonify({"error": translate("server.release.note_too_long", max=RELEASE_NOTE_MAX), "code": "bad_request"}), 400
    schema, _llm_table = _tables()
    try:
        conn = _ts_conn()
        try:
            ensure_release_tables(conn, schema)
            with conn:
                attributes = save_release(conn, schema, release, ticket=ticket, note=note, author=_owner())
        finally:
            conn.close()
    except psycopg2.Error as exc:
        return _db_error(exc)
    g.audit_extra = {"release": release, "ticket": attributes["ticket"]}
    attributes["jira"] = jira_links.links(jira_links.jira_config(), release, attributes["ticket"])
    return jsonify(attributes)


@release_pages_bp.route("/releases_data/<path:name>", methods=["DELETE"])
def delete_release(name: str):
    """Deletes the attributes and the Confluence page record of a release; the marks of its runs stay."""
    try:
        release = _release_name(name)
    except NotAReleaseName:
        return _not_found(name)
    schema, _llm_table = _tables()
    try:
        conn = _ts_conn()
        try:
            ensure_release_tables(conn, schema)
            with conn:
                deleted = delete_release_row(conn, schema, release)
        finally:
            conn.close()
    except psycopg2.Error as exc:
        return _db_error(exc)
    if not deleted:
        return _not_found(release)
    g.audit_extra = {"release": release}
    return jsonify({"status": "ok"})


@release_pages_bp.route("/release_trend", methods=["GET"])
def release_trend():
    """Accepted runs of a service and test type over the releases: ``?service=&test_type=&limit=12``."""
    service = str(request.args.get("service") or "").strip()
    if not service:
        return jsonify({"error": translate("server.release.service_required"), "code": "bad_request"}), 400
    test_type = mark_test_type(request.args.get("test_type"))
    raw_limit = str(request.args.get("limit") or "").strip()
    try:
        limit = int(raw_limit) if raw_limit else TREND_LIMIT
    except ValueError:
        limit = 0
    if not 1 <= limit <= TREND_LIMIT_MAX:
        return jsonify({"error": translate("server.release.bad_limit", max=TREND_LIMIT_MAX), "code": "bad_request"}), 400
    try:
        points = trend_payload(service, test_type, limit, _active_project_area() or "")
    except psycopg2.Error as exc:
        return _db_error(exc)
    return jsonify({"service": service, "test_type": test_type, "points": points})


# ---- Confluence ----------------------------------------------------------------------------------------------


@release_pages_bp.route("/release_publications/<path:name>", methods=["GET"])
def get_release_confluence(name: str):
    """Confluence page of the release, if it was published."""
    try:
        release = _release_name(name)
    except NotAReleaseName:
        return _not_found(name)
    schema, _llm_table = _tables()
    try:
        conn = _ts_conn()
        try:
            ensure_release_tables(conn, schema)
            publication = load_publication(conn, schema, release, _active_project_area() or "")
        finally:
            conn.close()
    except psycopg2.Error as exc:
        return _db_error(exc)
    if publication is None:
        return jsonify({"release": release, "page_id": None, "page_url": None, "updated_at": None})
    return jsonify({"release": release, **{key: publication[key] for key in ("page_id", "page_url", "updated_at")}})


def _failure_message(job_id: str):
    """Failure message (a ``Msg`` in the job's locale) naming the stage of the publication that failed."""
    return failure_message(job_store.get(job_id), "job.release_failed")


@release_pages_bp.route("/releases/<path:name>/confluence", methods=["POST"])
def post_release_confluence(name: str):
    """Starts publishing the release (its runs of the active project area) as its own Confluence page."""
    if not request.is_json:
        return jsonify({"error": translate("api.common.json_required"), "code": "bad_request"}), 415
    if not isinstance(request.get_json(silent=True), dict):
        return jsonify({"error": translate("api.common.json_body_object"), "code": "bad_request"}), 400
    try:
        release = _release_name(name)
    except NotAReleaseName:
        return _not_found(name)
    area = _active_project_area() or ""
    try:
        if not _release_exists(release, area):
            return _not_found(release)
    except psycopg2.Error as exc:
        return _db_error(exc)
    owner = _owner()
    try:
        check_job_slot(*job_store.running_counts(owner))
    except LimitError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    # The runner the job is registered for (Celery worker or a thread here) decides where it is published.
    job = job_runner.create_job(
        JOB_KIND_RELEASE_CONFLUENCE, run_name=release, message=job_message("job.publishing_release"), owner=owner,
    )
    page = release_url(release)
    job_store.update(job.job_id, report_url=page)
    job_runner.submit(
        tasks.publish_release_task, job_store.get(job.job_id) or job.job_id,
        release=release,
        project_area=area,
        source_url=f"{public_base_url()}{page}",
    )
    return jsonify({"status": "accepted", "job_id": job.job_id, "message": translate("api.release.publish_started")}), 200


__all__ = [
    "area_services",
    "release_pages_bp",
    "release_payload",
    "release_scope",
    "release_url",
    "thresholds_for",
    "trend_payload",
]
