"""HTTP API of the «Проекты» settings section."""

from __future__ import annotations

import psycopg2
from flask import Blueprint, jsonify, request

from loadlens_app.core import _active_project_area
from loadlens_app.projects import (
    ProjectError,
    create_project,
    delete_project,
    list_projects,
    move_service,
    reset_area_section,
    reset_service_section,
    update_project,
)

projects_bp = Blueprint("projects", __name__)


def _body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _text(data: dict, key: str) -> str:
    return str(data.get(key) or "")


def _failure(exc: Exception):
    if isinstance(exc, ProjectError):
        return jsonify({"error": str(exc)}), exc.status_code
    return jsonify({"error": f"База данных недоступна: {exc}"}), 502


@projects_bp.route("/projects", methods=["GET"])
def get_projects():
    """Overview of all projects for the settings page."""
    return jsonify(list_projects(_active_project_area() or "").to_dict())


@projects_bp.route("/projects", methods=["POST"])
def post_project():
    """Creates a project, optionally copying the settings of another one."""
    data = _body()
    try:
        project_id = create_project(
            _text(data, "id"), _text(data, "title"), _text(data, "description"), _text(data, "copy_from").strip()
        )
    except ProjectError as exc:
        return _failure(exc)
    return jsonify({"status": "ok", "id": project_id}), 201


@projects_bp.route("/projects/<project_id>", methods=["PATCH"])
def patch_project(project_id: str):
    """Updates the display name and description."""
    data = _body()
    try:
        update_project(project_id, _text(data, "title"), _text(data, "description"))
    except ProjectError as exc:
        return _failure(exc)
    return jsonify({"status": "ok"})


@projects_bp.route("/projects/<project_id>", methods=["DELETE"])
def remove_project(project_id: str):
    """Deletes the project; ``with_data=1`` also deletes its reports and metrics."""
    with_data = request.args.get("with_data") == "1"
    try:
        services = delete_project(project_id, with_data)
    except (ProjectError, psycopg2.Error) as exc:
        return _failure(exc)
    return jsonify({"status": "ok", "services": services, "with_data": with_data})


@projects_bp.route("/projects/<project_id>/services/<service_id>/move", methods=["POST"])
def post_move_service(project_id: str, service_id: str):
    """Moves a service to another project together with its settings and reports."""
    target = _text(_body(), "target").strip()
    try:
        updated = move_service(project_id, service_id, target)
    except (ProjectError, psycopg2.Error) as exc:
        return _failure(exc)
    return jsonify({"status": "ok", "target": target, "reports_updated": updated})


@projects_bp.route("/projects/<project_id>/overrides/<section>", methods=["DELETE"])
def delete_area_override(project_id: str, section: str):
    """Returns a project section to the global value."""
    try:
        changed = reset_area_section(project_id, section)
    except ProjectError as exc:
        return _failure(exc)
    return jsonify({"status": "ok", "changed": changed})


@projects_bp.route("/projects/<project_id>/services/<service_id>/overrides/<section>", methods=["DELETE"])
def delete_service_override(project_id: str, service_id: str, section: str):
    """Returns a service section to the project value."""
    try:
        changed = reset_service_section(project_id, service_id, section)
    except ProjectError as exc:
        return _failure(exc)
    return jsonify({"status": "ok", "changed": changed})
