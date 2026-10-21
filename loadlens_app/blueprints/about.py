# SPDX-License-Identifier: Apache-2.0
"""What is installed: version, build, license, links and the third-party notices.

Both endpoints are public (loadlens_app/auth/policy.py) and answer while the storage database is down,
so a bug report can always name the version. Nothing here reads the configuration: the payload is built
from loadlens_app/version.py constants and the THIRD_PARTY_NOTICES.md file of the installation.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify

from AI.i18n import t as translate
from loadlens_app.version import BUILD, DOCS_URL, LICENSE, SOURCE_URL, __version__

about_bp = Blueprint("about", __name__)

APP_NAME = "LoadLens"
ROOT_DIR = Path(__file__).resolve().parents[2]
THIRD_PARTY_NOTICES_PATH = ROOT_DIR / "THIRD_PARTY_NOTICES.md"


def about_payload() -> dict[str, str]:
    """The /about document: constants only, no configuration values."""
    return {
        "name": APP_NAME,
        "version": __version__,
        "build": BUILD,
        "license": LICENSE,
        "source_url": SOURCE_URL,
        "docs_url": DOCS_URL,
    }


@about_bp.app_context_processor
def _inject_version() -> dict[str, object]:
    """``app_version`` for the sidebar footer and ``about_info`` for the About section of the settings page."""
    return {"app_version": __version__, "about_info": about_payload()}


@about_bp.route("/about")
def about() -> Response:
    return jsonify(about_payload())


@about_bp.route("/about/third-party")
def third_party() -> Response | tuple[Response, int]:
    """THIRD_PARTY_NOTICES.md as text; the settings page renders it with the vendored marked + DOMPurify."""
    try:
        text = THIRD_PARTY_NOTICES_PATH.read_text(encoding="utf-8")
    except OSError:
        return jsonify({"error": translate("about.third_party.missing")}), 404
    return Response(text, mimetype="text/plain")
