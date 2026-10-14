"""Recovery mode: the app answers without the storage database and lets a recovery administrator fix it.

psycopg2.connect is replaced by a fake that fails for hosts that are "down"; everything above it is real:
the storage breaker, core._ts_conn, the user repository calls, the gate, the views and the runtime file.
"""

import json
import logging
import sys
import time as real_time
from pathlib import Path

import psycopg2
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import storage_state  # noqa: E402
from loadlens_app.auth import InMemoryRepository, cli, passwords, recovery, web  # noqa: E402

ADMIN = "rescue"
ADMIN_PASSWORD = "Storm-Lantern-2026"
OLD = {"host": "db.old", "port": 5432, "dbname": "lt", "user": "app", "password": "old-secret", "sslmode": "prefer", "schema": "public"}
SAFE = {"GET", "HEAD", "OPTIONS"}


class FakeCursor:
    def __init__(self):
        self.last = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.last = str(query)

    def fetchone(self):
        if "version()" in self.last:
            return ("PostgreSQL 16.4 on x86_64-pc-linux-gnu",)
        return ("2.15.0",)

    def fetchall(self):
        return []


class FakeConn:
    def cursor(self):
        return FakeCursor()

    def close(self):
        pass


class StoreRepository(InMemoryRepository):
    """In-memory users that are only reachable while the storage database is: each call connects first."""

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if not name.startswith("_") and callable(attr):
            def guarded(*args, **kwargs):
                core._ts_conn().close()
                return attr(*args, **kwargs)

            return guarded
        return attr


@pytest.fixture
def pg(monkeypatch):
    """Hosts listed in ``up`` answer; any other host times out."""
    state = {"up": set(), "calls": []}

    def fake_connect(*args, **kwargs):
        host = kwargs.get("host")
        state["calls"].append(host)
        if host not in state["up"]:
            raise psycopg2.OperationalError(f'connection to server at "{host}" (10.0.0.9), port 5432 failed: timeout expired')
        return FakeConn()

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    return state


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setitem(core.CONFIG, "storage", {"timescale": dict(OLD)})
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    for name in ("LOADLENS_RECOVERY_USER", "LOADLENS_RECOVERY_PASSWORD_HASH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(passwords, "_method", lambda: "pbkdf2:sha256:1000")
    monkeypatch.setattr(passwords, "_dummy_hash", "")


class Browser:
    def __init__(self, app):
        self.client = app.test_client()
        self.csrf = ""

    def request(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        if self.csrf and method.upper() not in SAFE:
            headers.setdefault("X-CSRF-Token", self.csrf)
        return self.client.open(url, method=method, headers=headers, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def login(self, username=ADMIN, password=ADMIN_PASSWORD):
        self.csrf = self.get("/auth/me").get_json()["csrf_token"]
        response = self.post("/login", json={"username": username, "password": password})
        if response.status_code == 200:
            self.csrf = response.get_json()["csrf_token"]
        return response


def make_app(monkeypatch, *, bootstrap=False, hashed=False):
    if bootstrap:
        monkeypatch.setenv("LOADLENS_ADMIN_USER", ADMIN)
        monkeypatch.setenv("LOADLENS_ADMIN_PASSWORD", ADMIN_PASSWORD)
    if hashed:
        monkeypatch.setenv("LOADLENS_RECOVERY_USER", ADMIN)
        monkeypatch.setenv("LOADLENS_RECOVERY_PASSWORD_HASH", passwords.hash_password(ADMIN_PASSWORD))
    from app import create_app

    flask_app = create_app(auth_repository=StoreRepository())
    flask_app.config["TESTING"] = True
    return flask_app


def runtime_storage():
    return json.loads(core.CONFIG_RUNTIME_PATH.read_text(encoding="utf-8"))["storage"]["timescale"]


# ---- outage answers -----------------------------------------------------------------------------------


def test_outage_page_for_browsers_and_json_for_the_api(monkeypatch, pg):
    flask_app = make_app(monkeypatch)
    with flask_app.test_client() as client:
        with client.session_transaction() as session:
            session.update(uid=1, epoch=0, iat=int(real_time.time()), csrf="t")
        page = client.get("/")
        assert page.status_code == 503 and page.mimetype == "text/html"
        assert int(page.headers["Retry-After"]) >= 1
        html = page.get_data(as_text=True)
        assert "Хранилище недоступно" in html and "Сервер базы не ответил вовремя" in html
        assert "db.old" not in html  # the address is not shown to an unidentified visitor
        assert "recovery_password_hash" in html  # how to configure a recovery administrator

        api = client.get("/runs")
        assert api.status_code == 503 and api.get_json()["code"] == "storage_unavailable"
    # Only the first request waited for the network; the rest failed fast inside the retry window.
    assert pg["calls"] == ["db.old"]


def test_login_form_explains_the_outage(monkeypatch, pg):
    browser = Browser(make_app(monkeypatch))
    anonymous = browser.get("/")
    assert anonymous.status_code == 302 and anonymous.headers["Location"].startswith("/login")
    login = browser.get("/login")
    html = login.get_data(as_text=True)
    assert login.status_code == 200 and "Хранилище недоступно" in html
    assert 'action="/login"' not in html  # nothing to sign in with
    assert "LOADLENS_RECOVERY_USER" in html

    refused = browser.login()
    assert refused.status_code == 503 and refused.get_json()["code"] == "recovery_not_configured"


def test_login_post_needs_the_csrf_token_during_an_outage(monkeypatch, pg):
    flask_app = make_app(monkeypatch, bootstrap=True)
    client = flask_app.test_client()
    response = client.post("/login", json={"username": ADMIN, "password": ADMIN_PASSWORD})
    assert response.status_code == 403 and response.get_json()["code"] == "csrf_failed"
    with client.session_transaction() as session:
        session.update(uid=1, epoch=0, iat=int(real_time.time()), csrf="t")
    response = client.post("/login", json={"username": ADMIN, "password": ADMIN_PASSWORD})
    assert response.status_code == 403 and response.get_json()["code"] == "csrf_failed"


# ---- recovery sign-in --------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["bootstrap", "hashed"])
def test_recovery_administrator_signs_in_while_the_store_is_down(monkeypatch, pg, caplog, source):
    browser = Browser(make_app(monkeypatch, bootstrap=source == "bootstrap", hashed=source == "hashed"))
    html = browser.get("/login").get_data(as_text=True)
    assert "Войти для восстановления" in html

    wrong = browser.login(password="Wrong-Password-1")
    assert wrong.status_code == 401 and "аварийного администратора" in wrong.get_json()["error"]
    assert browser.login(username="someone-else").status_code == 401

    with caplog.at_level(logging.WARNING, logger="loadlens_app.auth.recovery"):
        ok = browser.login()
    assert ok.status_code == 200
    body = ok.get_json()
    assert body["recovery"] is True and body["redirect"] == "/recovery"
    assert "recovery login" in caplog.text and f"user={ADMIN}" in caplog.text

    assert browser.get("/auth/me").get_json()["via"] == "recovery"
    assert browser.get("/login").headers["Location"] == "/recovery"
    page = browser.get("/recovery")
    assert page.status_code == 200 and "Восстановление подключения" in page.get_data(as_text=True)


def test_recovery_session_reaches_only_the_storage_card(monkeypatch, pg):
    browser = Browser(make_app(monkeypatch, bootstrap=True))
    assert browser.login().status_code == 200

    assert browser.get("/").headers["Location"] == "/recovery"
    assert browser.get("/settings").headers["Location"] == "/recovery"
    for method, url, kwargs in (
        ("GET", "/runs", {}),
        ("GET", "/config", {}),
        ("POST", "/config", {"json": {"section": "storage.timescale", "data": {"host": "evil"}}}),
        ("POST", "/config/test_connection", {"json": {"section": "storage.timescale", "data": {}}}),
        ("GET", "/auth/users", {}),
        ("POST", "/auth/tokens", {"json": {"name": "x"}}),
    ):
        response = browser.request(method, url, **kwargs)
        assert response.status_code == 403, url
        assert response.get_json()["code"] == "recovery_only", url

    status = browser.get("/recovery/status").get_json()
    assert status["recovery"] is True and status["storage"]["available"] is False
    assert status["storage"]["address"] == "db.old:5432/lt"
    assert status["form"]["host"] == "db.old" and status["form"]["password"] == "***"
    assert "table" not in status["form"]
    assert 0 < status["session_seconds_left"] <= recovery.SESSION_SECONDS


def test_recovery_login_is_rate_limited(monkeypatch, pg):
    browser = Browser(make_app(monkeypatch, bootstrap=True))
    for _ in range(5):
        assert browser.login(password="Wrong-Password-1").status_code == 401
    blocked = browser.login()
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) > 0


def test_recovery_credentials_do_nothing_while_the_store_is_up(monkeypatch, pg):
    pg["up"].add("db.old")
    browser = Browser(make_app(monkeypatch, hashed=True))
    html = browser.get("/login").get_data(as_text=True)
    assert "Хранилище недоступно" not in html
    response = browser.login()
    assert response.status_code == 401 and response.get_json()["code"] == "invalid_credentials"


def test_recovery_session_expires(monkeypatch, pg):
    flask_app = make_app(monkeypatch, bootstrap=True)
    browser = Browser(flask_app)
    assert browser.login().status_code == 200
    with browser.client.session_transaction() as session:
        session["iat"] = int(real_time.time()) - recovery.SESSION_SECONDS - 5
    ended = browser.get("/recovery/status")
    assert ended.status_code == 401 and ended.get_json()["code"] == "recovery_ended"
    assert browser.get("/auth/me").get_json()["via"] == "anonymous"


def test_recovery_session_ends_when_the_store_answers_again(monkeypatch, pg, caplog):
    browser = Browser(make_app(monkeypatch, bootstrap=True))
    assert browser.login().status_code == 200
    pg["up"].add("db.old")
    storage_state.BREAKER.allow_retry_now()
    with caplog.at_level(logging.WARNING, logger="loadlens_app.auth.recovery"):
        ended = browser.get("/recovery/status")
    assert ended.status_code == 401 and ended.get_json()["login_url"] == "/login?recovered=1"
    assert "storage is reachable again" in caplog.text
    login = browser.get("/login?recovered=1")
    assert login.status_code == 200 and "Хранилище снова доступно" in login.get_data(as_text=True)
    assert browser.get("/auth/me").get_json()["via"] == "anonymous"


# ---- fixing the connection -------------------------------------------------------------------------------


def test_check_validates_and_keeps_the_saved_password_on_its_own_host(monkeypatch, pg):
    browser = Browser(make_app(monkeypatch, bootstrap=True))
    assert browser.login().status_code == 200

    no_csrf = browser.client.post("/recovery/storage/check", json={"host": "db.new"})
    assert no_csrf.status_code == 403 and no_csrf.get_json()["code"] == "csrf_failed"

    for bad in (
        {"host": "http://db.new/x"},
        {"host": "db new"},
        {"port": 70000},
        {"port": "abc"},
        {"schema": "1abc"},
        {"sslmode": "sometimes"},
        {"dbname": ""},
        {"connect_timeout_sec": 600},
        {"password": 5},
    ):
        response = browser.post("/recovery/storage/check", json=bad)
        assert response.status_code == 400, bad

    # The masked password belongs to db.old: it is not sent to an address typed into the form.
    moved = browser.post("/recovery/storage/check", json={"host": "db.new", "password": "***"})
    assert moved.status_code == 400
    assert "db.new" not in pg["calls"]

    pg["up"].add("db.new")
    checked = browser.post("/recovery/storage/check", json={"host": "db.new", "password": "new-secret"})
    result = checked.get_json()
    assert checked.status_code == 200 and result["ok"] is True and "Подключение успешно" in result["message"]


def test_save_keeps_trying_then_ends_the_session_when_the_store_answers(monkeypatch, pg, caplog):
    browser = Browser(make_app(monkeypatch, bootstrap=True))
    assert browser.login().status_code == 200

    # Same address, only a longer timeout: the saved password stays and the database is still down.
    still_down = browser.post("/recovery/storage/save", json={"connect_timeout_sec": "7", "password": "***"})
    body = still_down.get_json()
    assert still_down.status_code == 200 and body["saved"] is True and body["storage"]["available"] is False
    saved = runtime_storage()
    assert saved["connect_timeout_sec"] == 7 and saved["password"] == "old-secret" and saved["host"] == "db.old"
    assert core.CONFIG["storage"]["timescale"]["connect_timeout_sec"] == 7

    pg["up"].add("db.new")
    with caplog.at_level(logging.WARNING, logger="loadlens_app.auth.recovery"):
        moved = browser.post("/recovery/storage/save", json={"host": "db.new", "port": 6432, "password": "new-secret"})
    body = moved.get_json()
    assert moved.status_code == 200 and body["available"] is True and body["redirect"] == "/login?recovered=1"
    saved = runtime_storage()
    assert (saved["host"], saved["port"], saved["password"]) == ("db.new", 6432, "new-secret")
    assert "audit config.update" in caplog.text and "db.new:6432" in caplog.text
    assert "new-secret" not in caplog.text

    # The recovery session is gone; the regular login works against the restored store.
    assert browser.get("/auth/me").get_json()["via"] == "anonymous"
    assert "Хранилище снова доступно" in browser.get("/login?recovered=1").get_data(as_text=True)
    signed_in = browser.login()  # the bootstrap administrator is created on the restored database
    assert signed_in.status_code == 200 and signed_in.get_json().get("recovery") is None


def test_save_drops_a_connection_string(monkeypatch, pg):
    monkeypatch.setitem(core.CONFIG, "storage", {"timescale": {**OLD, "dsn": "postgresql://app@db.old/lt", "table": "metrics"}})
    browser = Browser(make_app(monkeypatch, bootstrap=True))
    assert browser.login().status_code == 200
    assert browser.get("/recovery/status").get_json()["dsn_configured"] is True
    browser.post("/recovery/storage/save", json={"password": "***"})
    saved = runtime_storage()
    assert "dsn" not in saved and saved["table"] == "metrics"


# ---- authentication switched off ------------------------------------------------------------------------


def test_without_authentication_the_recovery_page_is_open_and_shows_details(monkeypatch, pg):
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    browser = Browser(make_app(monkeypatch))
    runs = browser.get("/runs")
    assert runs.status_code == 503 and runs.get_json()["recovery_url"] == "/recovery"

    page = browser.get("/recovery")
    assert page.status_code == 200
    status = browser.get("/recovery/status").get_json()
    assert status["recovery"] is False and status["storage"]["address"] == "db.old:5432/lt"

    pg["up"].add("db.old")
    saved = browser.post("/recovery/storage/save", json={"password": "***"})
    assert saved.get_json()["redirect"] == "/settings#connections"
    assert browser.get("/recovery").headers["Location"] == "/settings#connections"


def test_admin_pages_show_the_outage_banner(monkeypatch, pg):
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    browser = Browser(make_app(monkeypatch))
    assert browser.get("/runs").status_code == 503  # the breaker learns about the outage
    html = browser.get("/settings").get_data(as_text=True)
    assert 'href="/recovery"' in html and "Хранилище недоступно" in html


# ---- command line ------------------------------------------------------------------------------------------


def test_hash_password_needs_no_database(monkeypatch, capsys):
    def no_service(*args, **kwargs):
        raise AssertionError("hash-password must not build the service")

    monkeypatch.setattr(web, "build_service", no_service)
    monkeypatch.setenv("RECOVERY_PW", ADMIN_PASSWORD)
    assert cli.main(["hash-password", "--password-env", "RECOVERY_PW"]) == 0
    printed = capsys.readouterr()
    hashed = printed.out.strip()
    assert passwords.verify_password(hashed, ADMIN_PASSWORD) and "recovery_password_hash" in printed.err

    monkeypatch.setenv("RECOVERY_PW", "short")
    assert cli.main(["hash-password", "--password-env", "RECOVERY_PW"]) == 1
    monkeypatch.setenv("RECOVERY_PW", f"{ADMIN}-Long-Pass-1")
    assert cli.main(["hash-password", "--username", ADMIN, "--password-env", "RECOVERY_PW"]) == 1
