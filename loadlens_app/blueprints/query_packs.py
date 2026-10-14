"""Query packs in the settings: catalog, preview of the changes, check against the sources, apply.

Every endpoint works on the scope the «Запросы» section edits: global (no area), an area, or a
service of an area. Applying writes ``queries`` (and ``sla`` when the pack fills references) through
the same code path as POST /config and returns the previous values so the page can undo.
"""

from __future__ import annotations

from typing import Any, Mapping

from flask import Blueprint, g, jsonify, request

from loadlens_app.blueprints.config_api import has_own_section, scope_view, store_section
from loadlens_app.core import _list_project_areas, _metrics_services_for_area, _services_map_for_area
from loadlens_app.query_pack_service import PackRequestError, ScopeState, catalog_view, check_pack, detect_packs, plan_apply
from loadlens_app.query_preview import parse_preview_ts

query_packs_bp = Blueprint("query_packs", __name__)


@query_packs_bp.errorhandler(PackRequestError)
def _pack_error(exc: PackRequestError):
    return jsonify({"error": str(exc)}), exc.status_code


def _scope(source: Mapping[str, Any]) -> tuple[str, str]:
    area = str(source.get("area") or "").strip()
    service = str(source.get("service") or "").strip()
    if area and area not in {item["id"] for item in _list_project_areas()}:
        raise PackRequestError(f"Область «{area}» не найдена", 404)
    if service:
        if not area:
            raise PackRequestError("Сервис задаётся вместе с областью")
        known = set(_services_map_for_area(area)) | set(_metrics_services_for_area(area))
        if service not in known:
            raise PackRequestError(f"Сервис «{service}» не найден в области «{area}»", 404)
    return area, service


def _effective_cfg(area: str, service: str) -> dict:
    from update_page import _effective_config_for_scope

    return _effective_config_for_scope(area, service or None)


def _state(area: str, service: str) -> ScopeState:
    return ScopeState(
        queries=scope_view("queries", area, service),
        sla=scope_view("sla", area, service),
        effective_cfg=_effective_cfg(area, service),
    )


def _body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise PackRequestError("Ожидается JSON-объект")
    return body


@query_packs_bp.route("/config/query_packs", methods=["GET"])
def list_query_packs():
    """Catalog with compatibility of every pack with the scope's bindings and installed versions."""
    area, service = _scope(request.args)
    return jsonify(catalog_view(_state(area, service)))


@query_packs_bp.route("/config/query_packs/preview", methods=["POST"])
def preview_query_pack():
    """Rendered queries and what applying would change in queries and SLA; nothing is saved."""
    body = _body()
    area, service = _scope(body)
    return jsonify(plan_apply(_state(area, service), body).to_dict())


@query_packs_bp.route("/config/query_packs/check", methods=["POST"])
def check_query_pack():
    """Runs the selected pack queries against the domain source for the preview window."""
    body = _body()
    area, service = _scope(body)
    try:
        start_ts = parse_preview_ts(body.get("start"))
        end_ts = parse_preview_ts(body.get("end"))
    except ValueError as exc:
        raise PackRequestError("Некорректное окно проверки") from exc
    g.audit_extra = {"pack": str(body.get("pack_id") or "")[:80]}
    return jsonify(check_pack(_state(area, service), body, start_ts, end_ts))


@query_packs_bp.route("/config/query_packs/detect", methods=["POST"])
def detect_query_packs():
    """Reads metric and measurement names from the bound sources and scores every pack against them."""
    body = _body()
    area, service = _scope(body)
    return jsonify(detect_packs(_state(area, service)))


@query_packs_bp.route("/config/query_packs/apply", methods=["POST"])
def apply_query_pack():
    """Merges the pack into the scope's queries and fills SLA references."""
    body = _body()
    area, service = _scope(body)
    state = _state(area, service)
    result = plan_apply(state, body)
    previous = {
        "queries": state.queries,
        "sla": state.sla,
        "own_queries": has_own_section("queries", area, service),
        "own_sla": has_own_section("sla", area, service),
    }
    scope = store_section("queries", result.queries, area, service)
    if result.sla_changed:
        store_section("sla", result.sla, area, service)
    g.audit_extra = {
        "pack": result.pack.id,
        "version": str(result.pack.version),
        "domain": result.pack.domain,
        "mode": result.mode,
        "added": str(len(result.plan["added"])),
        "updated": str(len(result.plan["updated"])),
        "removed": str(len(result.plan["removed"])),
        "sla_fields": ",".join(sorted(result.sla_report.get("changes") or {})),
    }
    return jsonify({
        **result.to_dict(),
        **scope,
        "previous": previous,
        "stored": {"queries": result.queries, "sla": result.sla},
    })


__all__ = ["query_packs_bp"]
