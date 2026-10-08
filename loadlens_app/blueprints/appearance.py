"""Accent color for every page: CSS variables, the logo and a preview for the settings form."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request

from loadlens_app.appearance import AppearanceError, current_accent, palette_payload, parse_accent, recolored_logo, template_appearance

appearance_bp = Blueprint("appearance", __name__)


@appearance_bp.app_context_processor
def _inject_appearance() -> dict[str, dict[str, str]]:
    return {"appearance": template_appearance()}


@appearance_bp.route("/assets/logo.png")
def logo() -> Response | tuple[str, int]:
    """Recolored logo. ``c`` is ``rrggbb``; without it the stored accent is used."""
    raw = request.args.get("c")
    try:
        accent = parse_accent(raw) if raw else current_accent()
    except AppearanceError as exc:
        return str(exc), 400
    return Response(recolored_logo(accent), mimetype="image/png")


@appearance_bp.route("/appearance/palette")
def palette() -> tuple:
    """Shades, CSS and the logo URL for a color the settings page is previewing."""
    try:
        accent = parse_accent(request.args.get("accent"))
    except AppearanceError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(palette_payload(accent))
