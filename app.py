from flask import Flask

from loadlens_app import register_blueprints


def create_app() -> Flask:
    """Application factory used by tests and WSGI servers."""
    flask_app = Flask(__name__)
    register_blueprints(flask_app)
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0")


