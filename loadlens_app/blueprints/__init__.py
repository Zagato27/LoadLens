"""Blueprint registration utilities."""

from flask import Flask

from loadlens_app.core import RuntimeConfigError
from AI.storage_state import StorageUnavailableError
from loadlens_app.errors import internal_error, storage_unavailable

from .appearance import appearance_bp
from .compare import compare_bp
from .config_api import config_bp
from .dashboard import dashboard_bp
from .forecast import forecast_bp
from .load_tests import load_tests_bp
from .projects import projects_bp
from .query_packs import query_packs_bp
from .recovery import recovery_bp
from .settings import settings_bp

BLUEPRINTS = (
    dashboard_bp, compare_bp, settings_bp, config_bp, projects_bp, forecast_bp, appearance_bp, query_packs_bp,
    recovery_bp, load_tests_bp,
)


def register_blueprints(app: Flask) -> None:
    """Attach all application blueprints to the Flask app."""
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    # A damaged runtime settings file stops every change instead of being overwritten with defaults.
    app.register_error_handler(
        RuntimeConfigError,
        lambda exc: internal_error(exc, "Файл runtime-настроек повреждён, изменения не сохранены"),
    )
    # A storage outage that reached Flask uncaught: the outage page or a JSON 503, not a 500.
    app.register_error_handler(StorageUnavailableError, lambda exc: storage_unavailable())


__all__ = ["register_blueprints"]


