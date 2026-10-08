from flask import Flask

from loadlens_app import register_blueprints
from loadlens_app.auth import UserRepository, init_auth
from loadlens_app.data_sources_migration import migrate_legacy_data_sources


def create_app(auth_repository: UserRepository | None = None) -> Flask:
    """Application factory used by tests and WSGI servers.

    ``auth_repository`` replaces the PostgreSQL user store (used by tests).
    """
    migrate_legacy_data_sources()
    flask_app = Flask(__name__)
    register_blueprints(flask_app)
    init_auth(flask_app, repository=auth_repository)
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0")


