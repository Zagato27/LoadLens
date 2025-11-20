"""Blueprint registration utilities."""

from flask import Flask

from .compare import compare_bp
from .config_api import config_bp
from .dashboard import dashboard_bp
from .settings import settings_bp

BLUEPRINTS = (dashboard_bp, compare_bp, settings_bp, config_bp)


def register_blueprints(app: Flask) -> None:
    """Attach all application blueprints to the Flask app."""
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)


__all__ = ["register_blueprints"]


