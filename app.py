import os

from flask import Flask

from loadlens_app import register_blueprints
from loadlens_app.auth import UserRepository, init_auth
from AI.query_packs import load_catalog
from loadlens_app.data_sources_migration import migrate_legacy_data_sources


def create_app(auth_repository: UserRepository | None = None) -> Flask:
    """Application factory used by tests and WSGI servers.

    ``auth_repository`` replaces the PostgreSQL user store (used by tests).
    """
    migrate_legacy_data_sources()
    # A broken pack file fails start-up with its name instead of surfacing in the settings page.
    load_catalog()
    flask_app = Flask(__name__)
    register_blueprints(flask_app)
    init_auth(flask_app, repository=auth_repository)
    if os.getenv("LOADLENS_RUN_MIGRATIONS", "").strip() == "1":
        # Data migrations (run ids for old runs) belong to start-up, not to GET requests.
        from loadlens_app.migrations import start_background_migrations

        start_background_migrations()
    if os.getenv("LOADLENS_LOAD_TEST_WATCHER", "").strip() == "1":
        # Runs started before a restart are picked up at once, not when someone opens the launch page.
        from loadlens_app.load_tests.watcher import ensure_watcher

        ensure_watcher()
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0")


