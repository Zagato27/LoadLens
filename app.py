# SPDX-License-Identifier: Apache-2.0
import os

from flask import Flask

from loadlens_app import register_blueprints
from loadlens_app.auth import UserRepository, init_auth
from AI.query_packs import load_catalog
from loadlens_app.data_sources_migration import migrate_legacy_data_sources
from loadlens_app.i18n import install as install_i18n
from loadlens_app.static_assets import add_static_version, static_max_age, static_url


def create_app(auth_repository: UserRepository | None = None) -> Flask:
    """Application factory used by tests and WSGI servers.

    ``auth_repository`` replaces the PostgreSQL user store (used by tests).
    """
    migrate_legacy_data_sources()
    # A broken pack file fails start-up with its name instead of surfacing in the settings page.
    load_catalog()
    flask_app = Flask(__name__)
    # Static URLs carry the content hash (?v=), so browsers keep the files for a year without asking again.
    flask_app.config["SEND_FILE_MAX_AGE_DEFAULT"] = static_max_age()
    flask_app.url_defaults(add_static_version)
    flask_app.add_template_global(static_url)
    # Request locale, Jinja t/tn/i18n_script and Vary: Cookie (loadlens_app/i18n.py).
    install_i18n(flask_app)
    register_blueprints(flask_app)
    init_auth(flask_app, repository=auth_repository)
    # Data migrations belong to start-up, not to GET requests: the registry table says which ones this
    # database still lacks, so an update of the application brings them along. LOADLENS_RUN_MIGRATIONS=0 skips.
    from loadlens_app.migrations import enabled_by_env, start_background_migrations

    if enabled_by_env():
        start_background_migrations()
    # The demo stack (docker/docker-compose.demo.yml) sets LOADLENS_DEMO_SEED=1: the demo run and one stub
    # report are written once the database answers, in a background thread; no effect without the variable.
    from loadlens_app.demo_data import demo_seed_enabled, start_background_demo_seed

    if demo_seed_enabled():
        start_background_demo_seed()
    if os.getenv("LOADLENS_LOAD_TEST_WATCHER", "").strip() == "1":
        # Runs started before a restart are picked up at once, not when someone opens the launch page.
        from loadlens_app.load_tests.watcher import ensure_watcher

        ensure_watcher()
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0")


