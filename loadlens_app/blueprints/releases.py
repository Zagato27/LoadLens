# SPDX-License-Identifier: Apache-2.0
"""Release marks of runs (AI/release_marks.py): the release a run tested, its accepted run and the baseline.

Viewers read the marks, engineers set and remove them. Every answer of a run is its state
``{run_name, run_id, service, test_type, release, accepted, accepted_manual, baseline, note, author, created_at,
updated_at}``; the GET adds the release names of the active area and the input limits.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import psycopg2
from flask import Blueprint, g, jsonify, request

from AI import release_marks
from AI.db_store import _ensure_llm_reports_table
from AI.i18n import t as translate
from settings import CONFIG
from loadlens_app import response_cache
from loadlens_app.auth import current_auth
from loadlens_app.core import _active_project_area, _resolve_services_filter, _ts_conn
from loadlens_app.errors import internal_error

logger = logging.getLogger(__name__)

releases_bp = Blueprint("releases", __name__)


def _storage_cfg() -> dict:
    return (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}


def _author() -> str:
    """Who changed a mark: the signed-in user; empty when sign-in is off."""
    user = current_auth().user
    return user.username if user is not None else ""


def _area_services() -> list[str] | None:
    """Services of the active project area; None means every service (no area chosen)."""
    services = _resolve_services_filter(_active_project_area())
    return list(services) if services else None


def _area_release_names(conn, schema: str, llm_table: str) -> list[str]:
    return [item["release"] for item in release_marks.list_releases(conn, schema, _area_services(), llm_table)]


def _change(action: Callable[[Any, str, str], dict], audit: dict) -> Any:
    """Runs a mark change in one transaction and answers the new state (400, 404 and 409 on refusals)."""
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    llm_table = cfg.get("llm_table", "llm_reports")
    try:
        conn = _ts_conn()
    except Exception as exc:  # pragma: no cover
        return internal_error(exc)
    try:
        try:
            _ensure_llm_reports_table(conn, cfg)
        except Exception:
            pass
        with conn:
            state = action(conn, schema, llm_table)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except psycopg2.IntegrityError as exc:
        # Two changes of the same marks at once: the unique indexes keep one accepted run and one baseline.
        logger.warning("Release mark change refused by the database: %s", exc)
        return jsonify({"error": translate("server.release.conflict"), "code": "conflict"}), 409
    except Exception as exc:
        return internal_error(exc)
    finally:
        conn.close()
    response_cache.invalidate(response_cache.DASHBOARD)
    g.audit_extra = {**audit, "release": state.get("release"), "accepted": state.get("accepted"),
                     "baseline": state.get("baseline")}
    return jsonify(state), 200


def _body() -> dict | None:
    data = request.get_json(silent=True)
    if data is None and not request.get_data():
        return {}
    return data if isinstance(data, dict) else None


@releases_bp.route("/runs/<run_name>/release", methods=["GET"])
def get_run_release(run_name: str):
    """Marks of a run: release, accepted run, baseline; ``releases`` — release names of the active area."""
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    llm_table = cfg.get("llm_table", "llm_reports")
    try:
        conn = _ts_conn()
        try:
            try:
                _ensure_llm_reports_table(conn, cfg)
            except Exception:
                pass
            with conn:
                state = release_marks.mark_state(conn, schema, llm_table, run_name)
                names = _area_release_names(conn, schema, llm_table) if state is not None else []
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover
        return internal_error(exc)
    if state is None:
        return jsonify({"error": translate("server.report.final_not_found", run_name=run_name)}), 404
    state["releases"] = names
    state["name_max"] = release_marks.RELEASE_NAME_MAX
    state["note_max"] = release_marks.RELEASE_NOTE_MAX
    return jsonify(state), 200


@releases_bp.route("/runs/<run_name>/release", methods=["PUT"])
def set_run_release(run_name: str):
    """Marks the run as a test of a release: ``{"release": "25.11", "note"?: "...", "accepted"?: true}``.

    ``accepted: true`` accepts the run by hand; without it the first marked run of the release, service and test
    type is accepted and an automatically accepted one hands over to the newly marked run. ``accepted: false``
    returns this run's manual acceptance to automatic selection; another run accepted by hand stays. A missing ``note``
    keeps the stored one.
    """
    data = _body()
    if data is None:
        return jsonify({"error": translate("api.common.json_object_expected")}), 400
    release = data.get("release")
    note = data.get("note")
    accepted = data.get("accepted")
    if not isinstance(release, str) or (note is not None and not isinstance(note, str)) \
            or (accepted is not None and not isinstance(accepted, bool)):
        return jsonify({"error": translate("api.release.bad_body")}), 400
    author = _author()
    manual = accepted
    return _change(
        lambda conn, schema, llm_table: release_marks.set_release(
            conn, schema, llm_table, run_name, release, note=note, accepted=manual, author=author,
        ),
        {"run_name": run_name, "manual_accept": accepted is True},
    )


@releases_bp.route("/runs/<run_name>/release", methods=["DELETE"])
def reset_run_release(run_name: str):
    """Removes the release mark (the baseline stays); the acceptance moves to the newest run of the release."""
    author = _author()
    return _change(
        lambda conn, schema, llm_table: release_marks.clear_release(conn, schema, run_name, author, llm_table=llm_table),
        {"run_name": run_name},
    )


@releases_bp.route("/runs/<run_name>/baseline", methods=["PUT"])
def set_run_baseline(run_name: str):
    """Pins the run as the baseline of its service and test type: ``{"note"?: "..."}``; the previous one is unpinned."""
    data = _body()
    if data is None:
        return jsonify({"error": translate("api.common.json_object_expected")}), 400
    note = data.get("note")
    if note is not None and not isinstance(note, str):
        return jsonify({"error": translate("api.release.bad_body")}), 400
    author = _author()
    return _change(
        lambda conn, schema, llm_table: release_marks.set_baseline(conn, schema, llm_table, run_name, note=note, author=author),
        {"run_name": run_name},
    )


@releases_bp.route("/runs/<run_name>/baseline", methods=["DELETE"])
def reset_run_baseline(run_name: str):
    """Unpins the baseline of the service (the release mark stays)."""
    author = _author()
    return _change(
        lambda conn, schema, llm_table: release_marks.clear_baseline(conn, schema, run_name, author),
        {"run_name": run_name},
    )


@releases_bp.route("/releases_data", methods=["GET"])
def list_releases():
    """Releases of the active area, newest first: ``[{release, runs, accepted, services, last_created_at}]``."""
    cfg = _storage_cfg()
    try:
        conn = _ts_conn()
        try:
            with conn:
                items = release_marks.list_releases(
                    conn, cfg.get("schema", "public"), _area_services(), cfg.get("llm_table", "llm_reports"),
                )
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover
        return internal_error(exc)
    return jsonify(items), 200
