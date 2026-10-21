# SPDX-License-Identifier: Apache-2.0
"""Server texts in the UI language: API answers, auth, connection checks, storage state, limits and validation.

With the ``loadlens.lang=en`` cookie (or inside ``with_locale("en")``) the answers carry English text without
Cyrillic (user data excepted); without a cookie they stay byte-identical to the Russian texts they replaced.
"""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import pytest
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import i18n as engine  # noqa: E402
from AI import storage_state  # noqa: E402
from AI.i18n import render_ref, with_locale  # noqa: E402
from loadlens_app.auth import Actor, InMemoryRepository, Role, passwords, web  # noqa: E402
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale  # noqa: E402

PASSWORD = "Correct-Horse-1"
OLD_STORAGE = {"host": "db.old", "port": 5432, "dbname": "lt", "user": "app", "password": "old-secret", "sslmode": "prefer", "schema": "public"}

# Modules whose user texts this suite covers; every dictionary key written in them must exist in both languages.
SERVER_FILES = (
    "AI/capacity_forecast.py",
    "AI/data_sources.py",
    "AI/db_store.py",
    "AI/storage_state.py",
    "loadlens_app/appearance.py",
    "loadlens_app/auth/models.py",
    "loadlens_app/auth/passwords.py",
    "loadlens_app/auth/providers.py",
    "loadlens_app/auth/service.py",
    "loadlens_app/auth/views.py",
    "loadlens_app/auth/web.py",
    "loadlens_app/blueprints/__init__.py",
    "loadlens_app/blueprints/appearance.py",
    "loadlens_app/blueprints/compare.py",
    "loadlens_app/blueprints/config_api.py",
    "loadlens_app/blueprints/dashboard.py",
    "loadlens_app/blueprints/forecast.py",
    "loadlens_app/blueprints/projects.py",
    "loadlens_app/blueprints/recovery.py",
    "loadlens_app/blueprints/settings.py",
    "loadlens_app/config_schema.py",
    "loadlens_app/config_secrets.py",
    "loadlens_app/confluence_page.py",
    "loadlens_app/connection_checks.py",
    "loadlens_app/core.py",
    "loadlens_app/errors.py",
    "loadlens_app/forecast.py",
    "loadlens_app/limits.py",
    "loadlens_app/projects.py",
    "loadlens_app/query_preview.py",
)
SERVER_KEY = re.compile(r"^(api|server|auth\.server|common\.role)\.[a-z0-9_]+(\.[a-z0-9_]+)*$")


def json_text(response) -> str:
    return json.dumps(response.get_json(), ensure_ascii=False)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setitem(core.CONFIG, "storage", {"timescale": dict(OLD_STORAGE)})
    for name in ("LOADLENS_RECOVERY_USER", "LOADLENS_RECOVERY_PASSWORD_HASH", "LOADLENS_ADMIN_USER", "LOADLENS_ADMIN_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(passwords, "_method", lambda: "pbkdf2:sha256:1000")
    monkeypatch.setattr(passwords, "_dummy_hash", "")


class _NoRows:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self

    def execute(self, *args, **kwargs):
        return None

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def close(self):
        return None


@pytest.fixture
def client(monkeypatch):
    """The app without sign-in and with an empty fake storage."""
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    from app import create_app
    from loadlens_app.blueprints import compare, dashboard

    for module in (core, compare, dashboard):
        monkeypatch.setattr(module, "_ts_conn", lambda: _NoRows())
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def en(client_):
    use_locale(client_, "en")
    return client_


# ---- dictionaries ----------------------------------------------------------------------------------------------


def _string_constants(path: str) -> set[str]:
    tree = ast.parse((ROOT_DIR / path).read_text(encoding="utf-8"))
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def test_every_server_key_written_in_the_covered_modules_exists_in_both_languages():
    keys = sorted({value for path in SERVER_FILES for value in _string_constants(path) if SERVER_KEY.match(value)})
    assert len(keys) > 250
    missing = [(lang, key) for key in keys for lang in ("ru", "en") if not engine.has_key(key, lang)]
    assert not missing, missing


def test_tables_of_keys_point_at_existing_messages():
    from loadlens_app.auth.models import ROLE_LABEL_KEYS
    from loadlens_app import limits

    tables = [*storage_state.KIND_TITLES.values(), *storage_state.KIND_HINTS.values(), *ROLE_LABEL_KEYS.values()]
    tables += [what for _name, what in limits._LLM_LIMITS.values()]
    for key in tables:
        assert engine.has_key(key, "ru") and engine.has_key(key, "en"), key


def test_server_namespaces_hold_no_cyrillic_in_en():
    for namespace in ("api", "server"):
        values = json.loads((ROOT_DIR / "locales" / "en" / f"{namespace}.json").read_text(encoding="utf-8"))
        assert not [key for key, value in values.items() if CYRILLIC.search(value)]
    auth = json.loads((ROOT_DIR / "locales" / "en" / "auth.json").read_text(encoding="utf-8"))
    assert not [key for key, value in auth.items() if key.startswith("auth.server.") and CYRILLIC.search(value)]


# ---- API answers -------------------------------------------------------------------------------------------------


def test_config_validation_errors_follow_the_cookie(client):
    body = {"section": "queries", "data": {"jvm": "x"}}
    assert client.post("/config", json=body).get_json()["error"] == "jvm: ожидается объект"
    answer = en(client).post("/config", json=body)
    assert answer.status_code == 400 and answer.get_json()["error"] == "jvm: an object is expected"

    sla = client.post("/config", json={"section": "sla", "data": {"latency_unit": "min"}})
    assert sla.status_code == 400 and sla.get_json()["error"] == 'sla.latency_unit: allowed values are ms, s, got "min"'
    for body in ({"section": ""}, {"section": "auth.local", "data": {}}, {"section": "metrics_config", "data": {"services": {}}}):
        answer = client.post("/config", json=body)
        assert answer.status_code in (400, 403)
        assert_no_cyrillic(json_text(answer))


def test_query_preview_rejections_carry_a_code_and_follow_the_cookie(client, monkeypatch):
    empty = client.post("/config/query_preview", json={"domain": "jvm", "lang": "promql", "query": " "})
    assert empty.status_code == 400 and empty.get_json()["message"] == "Запрос пустой" and empty.get_json()["code"] == "empty_query"

    en(client)
    empty = client.post("/config/query_preview", json={"domain": "jvm", "lang": "promql", "query": " "})
    assert empty.status_code == 400 and empty.get_json()["message"] == "The query is empty"
    unknown = client.post("/config/query_preview", json={"domain": "nope", "lang": "promql", "query": "up"})
    assert unknown.status_code == 400 and unknown.get_json() | {"window": ""} == {
        "ok": False, "code": "unknown_domain", "message": 'Unknown domain "nope"', "series_count": 0, "point_count": 0,
        "series": [], "window": "",
    }
    assert unknown.get_json()["window"] == "the last 15 minutes"

    monkeypatch.setitem(core.CONFIG, "data_sources", {
        "influx": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "ops", "token": "t"}},
    })
    monkeypatch.setitem(core.CONFIG, "domain_sources", {"default": {"source": "influx", "bucket": "telegraf"}})
    rejected = client.post("/config/query_preview", json={"domain": "hard_resources", "lang": "promql", "query": "up"})
    assert rejected.status_code == 400 and rejected.get_json()["message"] == "This source runs Flux only"
    assert rejected.get_json()["code"] == "language_rejected"
    window = client.post("/config/query_preview", json={"domain": "jvm", "lang": "promql", "query": "up", "start": "x"})
    assert window.status_code == 400 and window.get_json()["message"] == "Invalid check window"


def test_query_preview_without_series_says_so_in_the_locale(monkeypatch):
    from loadlens_app.query_preview import preview_metric_query

    monkeypatch.setattr("loadlens_app.query_preview.fetch_metric_series", lambda *a, **k: {"status": "success", "data": {"result": []}})
    cfg = {
        "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom"}}},
        "domain_sources": {"default": {"source": "prom"}},
        "default_params": {"step": "30s"},
    }
    window = {"start_ts": 1_760_000_000, "end_ts": 1_760_000_900}
    ru = preview_metric_query(cfg, domain="jvm", lang="promql", query="up", **window)
    assert ru.code == "no_series" and re.fullmatch(r"За окно \d\d\.\d\d \d\d:\d\d–\d\d\.\d\d \d\d:\d\d серий нет", ru.message)
    with with_locale("en"):
        result = preview_metric_query(cfg, domain="jvm", lang="promql", query="up", **window)
    assert result.code == "no_series" and re.fullmatch(r"No series in the window \d\d/\d\d \d\d:\d\d–\d\d/\d\d \d\d:\d\d", result.message)


def test_connection_check_failures_are_english(client):
    en(client)
    unknown = client.post("/config/test_connection", json={"section": "llm", "data": {"provider": "nope"}})
    assert unknown.status_code == 200 and unknown.get_json()["ok"] is False
    assert unknown.get_json()["message"].startswith('Unknown provider "nope". Allowed: ')
    section = client.post("/config/test_connection", json={"section": "bogus", "data": {}})
    assert section.status_code == 400 and section.get_json()["error"].startswith('Checks are not available for section "bogus"')
    source = client.post("/config/test_connection", json={"section": "data_sources", "data": {}, "source": ""})
    assert source.status_code == 400 and source.get_json()["error"] == 'Source "—" not found'
    for answer in (unknown, section, source):
        assert_no_cyrillic(json_text(answer))


def test_connection_check_messages_render_in_the_locale():
    from loadlens_app import connection_checks as checks

    with with_locale("en"):
        assert checks._explain(requests.exceptions.Timeout()) == (
            "No response within 10 s: the server is not responding or the address is wrong"
        )
        assert checks.check_prometheus("").message == "Prometheus URL is not set"
        assert checks.run_check("data_sources", {}).message == "Specify the source to check"
        assert checks.list_llm_models({"provider": "nope"}).message.startswith('Unknown provider "nope"')
        assert checks.check_data_source({"type": ""}).message == (
            "Unknown source type: not set. Allowed: prometheus, grafana_proxy, influxdb"
        )
    assert checks._explain(requests.exceptions.Timeout()) == (
        "Превышено время ожидания ответа (10 с): сервер не отвечает или адрес неверный"
    )
    assert checks.check_data_source({"type": ""}).message == (
        "Неизвестный тип источника: не задан. Допустимо: prometheus, grafana_proxy, influxdb"
    )


def test_report_and_run_endpoints_answer_in_english(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    en(client)
    missing = client.post("/create_report", json={"service": "demo"})
    assert missing.status_code == 400 and missing.get_json()["message"] == "Specify the start time, the end time and the service"
    bad_time = client.post("/create_report", json={"start": "x", "end": "y", "service": "demo"})
    assert bad_time.get_json()["message"].startswith("Invalid time format: ISO 8601 is expected")

    monkeypatch.setattr(dashboard, "_find_area_for_service", lambda service: "demo")
    monkeypatch.setattr(dashboard, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(dashboard, "_metrics_service_entry", lambda service: ("demo", {"metrics": []}))
    monkeypatch.setattr(dashboard, "_report_step", lambda area, service: "1m")
    too_long = client.post("/create_report", json={"start": "2024-11-01T00:00", "end": "2024-11-10T00:00", "service": "demo"})
    assert too_long.status_code == 400 and too_long.get_json()["code"] == "window_too_long"
    assert too_long.get_json()["message"] == "The test window of 216.0 h is longer than the allowed 168 h: shorten the interval"

    assert client.patch("/runs/r1", json={}).get_json()["error"] == "A new report name is required"
    bad_name = client.patch("/runs/r1", json={"new_run_name": "a/b"})
    assert bad_name.status_code == 400 and bad_name.get_json()["error"] == "The new report name contains forbidden characters: / \\ < >"
    assert client.delete("/runs/r1").get_json()["message"] == "Report deleted"
    assert client.get("/compare_baseline").get_json()["error"] == "run_name is required"
    assert client.get("/compare_summary", query_string={"run_a": "a"}).get_json()["error"] == "run_a, run_b and domain are required"
    assert client.get("/jobs", query_string={"status": "nope"}).get_json()["error"].startswith("Invalid status: nope. Allowed: ")
    assert client.get("/report_ref/missing").get_json()["error"] == "Report not found"
    for answer in (missing, bad_time, too_long, bad_name):
        assert_no_cyrillic(json_text(answer))


def test_limit_errors_render_in_the_locale_and_keep_a_ref():
    from loadlens_app.limits import DailyBudget, LimitError, check_report_window

    with with_locale("en"), pytest.raises(LimitError) as window:
        check_report_window(0, 400 * 3_600_000, "30s")
    assert str(window.value) == "The test window of 400.0 h is longer than the allowed 168 h: shorten the interval"
    assert window.value.code == "window_too_long" and window.value.status_code == 400
    assert render_ref(window.value.ref(), locale="ru") == "Окно теста 400.0 ч длиннее допустимых 168 ч: сократите интервал"

    budget = DailyBudget()
    budget.consume("eve", "report", 1)
    with with_locale("en"), pytest.raises(LimitError) as spent:
        budget.consume("eve", "report", 1)
    assert str(spent.value) == "The daily limit of AI analysis runs is used up (1), it resets tomorrow"
    assert render_ref(spent.value.ref(), locale="ru") == "Дневной лимит запусков с анализом ИИ исчерпан (1), он обновится завтра"
    with pytest.raises(LimitError) as ru:
        budget.consume("eve", "report", 1)
    assert str(ru.value) == "Дневной лимит запусков с анализом ИИ исчерпан (1), он обновится завтра"
    assert LimitError("plain text").ref() is None


def test_internal_errors_show_the_locale_and_log_english(client, caplog):
    from loadlens_app.errors import internal_error

    with client.application.test_request_context("/", headers={"Cookie": "loadlens.lang=en"}):
        response, status = internal_error(RuntimeError("boom"), "api.error.demo_failed")
    assert status == 500
    error = response.get_json()["error"]
    assert re.fullmatch(r"Could not create the demo run\. Error ID: [0-9a-f]{8}", error)
    assert "Could not create the demo run [error_id=" in caplog.text
    with client.application.test_request_context("/"):
        response, _status = internal_error(RuntimeError("boom"))
    assert re.fullmatch(r"Внутренняя ошибка сервера\. Код ошибки: [0-9a-f]{8}", response.get_json()["error"])


def test_recovery_form_and_logo_errors_are_english(client):
    en(client)
    check = client.post("/recovery/storage/check", json={"host": ""})
    assert check.status_code == 400 and check.get_json()["error"] == "Specify the database server address"
    port = client.post("/recovery/storage/check", json={"host": "db", "port": "x"})
    assert port.get_json()["error"] == "The port must be a number from 1 to 65535"
    name = client.post("/recovery/storage/check", json={"host": "db", "dbname": "a b"})
    assert name.get_json()["error"] == "Specify the database name: up to 63 characters, no spaces"
    logo = client.get("/assets/logo.png", query_string={"c": "zz"})
    assert logo.status_code == 400 and logo.get_json()["error"] == "Parameter c must be a color in rrggbb format"
    palette = client.get("/appearance/palette", query_string={"accent": "zz"})
    assert palette.get_json()["error"] == 'Accent color "zz" must be in #rrggbb format'


# ---- storage state ---------------------------------------------------------------------------------------------


@pytest.fixture
def storage_down(monkeypatch):
    def fake_connect(*args, **kwargs):
        host = kwargs.get("host")
        raise psycopg2.OperationalError(f'connection to server at "{host}" (10.0.0.9), port 5432 failed: timeout expired')

    monkeypatch.setattr(psycopg2, "connect", fake_connect)


def test_storage_outage_title_and_hint_follow_the_cookie(monkeypatch, storage_down):
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    from app import create_app

    flask_app = create_app()
    flask_app.config["TESTING"] = True
    client_ = flask_app.test_client()
    use_locale(client_, "en")
    runs = client_.get("/runs")
    assert runs.status_code == 503 and runs.get_json()["code"] == "storage_unavailable"
    assert runs.get_json()["error"] == "Storage unavailable: sign-in and report data are temporarily not working"

    status = client_.get("/recovery/status").get_json()["storage"]
    assert status["title"] == "The database server did not respond in time"
    assert status["hint"] == "Check the network and VPN to the database server and that the address and port are correct."
    assert_no_cyrillic(json.dumps(status, ensure_ascii=False))

    use_locale(client_, None)
    status = client_.get("/recovery/status").get_json()["storage"]
    assert status["title"] == "Сервер базы не ответил вовремя"
    assert status["hint"] == "Проверьте сеть и VPN до сервера базы и что адрес и порт указаны верно."


def test_storage_titles_cover_every_kind_in_both_languages():
    for kind in ("timeout", "refused", "auth", "dns", "database", "ssl", "other", "something-new"):
        assert not CYRILLIC.search(storage_state.kind_title(kind, "en") + storage_state.kind_hint(kind, "en"))
        assert CYRILLIC.search(storage_state.kind_title(kind, "ru"))
    assert storage_state.kind_title("other", "ru") == "Не удалось подключиться к базе"


# ---- auth --------------------------------------------------------------------------------------------------------


def _auth_app(monkeypatch):
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    from app import create_app

    flask_app = create_app(auth_repository=InMemoryRepository())
    flask_app.config["TESTING"] = True
    service = flask_app.extensions[web.EXTENSION_KEY]
    for name, role in (("vic", Role.VIEWER), ("ada", Role.ADMIN)):
        service.create_user(username=name, role=role, password=PASSWORD, must_change_password=False, actor=Actor(name="test"))
    return flask_app


def _sign_in(client_, username):
    csrf = client_.get("/auth/me").get_json()["csrf_token"]
    answer = client_.post("/login", json={"username": username, "password": PASSWORD}, headers={"X-CSRF-Token": csrf})
    assert answer.status_code == 200
    return answer.get_json()["csrf_token"]


def test_login_errors_follow_the_cookie(monkeypatch):
    client_ = _auth_app(monkeypatch).test_client()
    csrf = client_.get("/auth/me").get_json()["csrf_token"]
    wrong = {"username": "vic", "password": "nope-nope-nope"}
    ru = client_.post("/login", json=wrong, headers={"X-CSRF-Token": csrf})
    assert ru.status_code == 401
    assert ru.get_json()["error"] == "Неверный логин или пароль, либо учётная запись временно заблокирована"
    use_locale(client_, "en")
    failed = client_.post("/login", json=wrong, headers={"X-CSRF-Token": csrf})
    assert failed.status_code == 401 and failed.get_json()["code"] == "invalid_credentials"
    assert failed.get_json()["error"] == "Invalid username or password, or the account is temporarily locked"
    page = client_.post("/login", data={**wrong, "csrf_token": csrf})
    assert "Invalid username or password, or the account is temporarily locked" in page.get_data(as_text=True)
    unauthenticated = client_.get("/runs")
    assert unauthenticated.status_code == 401 and unauthenticated.get_json()["error"] == "Sign-in required"


def test_forbidden_answers_and_account_errors_are_english(monkeypatch):
    client_ = _auth_app(monkeypatch).test_client()
    use_locale(client_, "en")
    csrf = _sign_in(client_, "vic")
    page = client_.get("/settings")
    assert page.status_code == 403
    assert "You do not have permission for this action" in page.get_data(as_text=True)
    api = client_.post("/config", json={"section": "sla", "data": {}}, headers={"X-CSRF-Token": csrf})
    assert api.status_code == 403 and api.get_json()["error"] == "You do not have permission for this action"
    wrong = client_.post(
        "/auth/password", json={"current_password": "nope", "new_password": "Another-Horse-2"}, headers={"X-CSRF-Token": csrf}
    )
    assert wrong.status_code == 400 and wrong.get_json() == {"error": "The current password is incorrect", "code": "wrong_password"}
    token = client_.post("/auth/tokens", json={"name": ""}, headers={"X-CSRF-Token": csrf})
    assert token.status_code == 400 and token.get_json()["error"] == "Enter a token name"


def test_roles_and_password_policy_are_english_for_the_admin(monkeypatch):
    flask_app = _auth_app(monkeypatch)
    client_ = flask_app.test_client()
    use_locale(client_, "en")
    csrf = _sign_in(client_, "ada")
    users = client_.get("/auth/users").get_json()
    assert [role["label"] for role in users["roles"]] == ["Viewer", "Engineer", "Administrator"]
    assert {user["role_label"] for user in users["users"]} == {"Viewer", "Administrator"}
    weak = client_.post("/auth/users", json={"username": "neo", "role": "viewer", "password": "short"}, headers={"X-CSRF-Token": csrf})
    assert weak.status_code == 400 and weak.get_json()["error"] == "The password must be at least 10 characters long"
    role = client_.post("/auth/users", json={"username": "neo", "role": "boss", "password": PASSWORD}, headers={"X-CSRF-Token": csrf})
    assert role.get_json()["error"] == "Unknown role: 'boss'"
    with flask_app.test_request_context("/login", headers={"Cookie": "loadlens.lang=en"}):
        assert [p.label for p in flask_app.extensions[web.EXTENSION_KEY].providers] == ["Local account"]
    with flask_app.test_request_context("/login"):
        assert [p.label for p in flask_app.extensions[web.EXTENSION_KEY].providers] == ["Локальная учётная запись"]
        assert Role.ADMIN.label == "Администратор"


def test_password_policy_messages_in_both_languages():
    with with_locale("en"), pytest.raises(passwords.PasswordPolicyError, match="^Enter a password$"):
        passwords.validate_password("")
    with pytest.raises(passwords.PasswordPolicyError, match="^Пароль должен содержать не менее 10 символов$"):
        passwords.validate_password("short")


# ---- validation in the modules ---------------------------------------------------------------------------------


def test_data_source_errors_are_english():
    from AI.data_sources import DataSourceConfigError, binding_errors, resolve_domain_source, validate_data_sources

    with with_locale("en"):
        with pytest.raises(DataSourceConfigError) as bad_id:
            validate_data_sources({"Bad ID": {}})
        assert str(bad_id.value).startswith('Source ID "Bad ID" is not valid')
        errors = binding_errors({}, {"default": {"source": "gone"}}, {"demo": {"default": {"source": "gone"}}})
        assert errors == [
            "source gone is in use: global bindings, domain default",
            "source gone is in use: area demo, domain default",
        ]
        with pytest.raises(DataSourceConfigError, match="^The data_sources catalog is not set$"):
            resolve_domain_source({}, "jvm")
    assert binding_errors({}, {}, {"demo": {"default": {"source": "gone"}}}) == ["источник gone используется: область demo, домен default"]


def test_project_page_and_core_errors_are_english():
    from loadlens_app.confluence_page import page_ref
    from loadlens_app.context_import import ContextImportError
    from loadlens_app.projects import ProjectError, _validate_project_id

    with with_locale("en"):
        with pytest.raises(ProjectError, match="^Specify the project ID$"):
            _validate_project_id("")
        with pytest.raises(ProjectError) as bad:
            _validate_project_id("a b")
        assert str(bad.value) == 'ID "a b": up to 40 characters, without spaces, slashes or the signs < >'
        with pytest.raises(ContextImportError, match="^Paste a link to a Confluence page$"):
            page_ref("", "https://wiki.example.com")
        assert core.run_name_problem("a" * 200) == "Report name: at most 180 characters"
    with pytest.raises(ProjectError) as bad:
        _validate_project_id("a b")
    assert str(bad.value) == "Идентификатор «a b»: до 40 символов, без пробелов, слэшей и знаков < >"
    assert core.run_name_problem("a/b") == "Имя отчёта содержит недопустимые символы: / \\ < >"


def test_schema_secrets_verdict_and_capacity_errors_are_english(monkeypatch):
    from AI.capacity_forecast import CapacityModelError, _require_closed_vus, fit_usl
    from AI.db_store import MANUAL_VERDICTS, save_manual_verdict
    from loadlens_app import config_schema
    from loadlens_app.config_secrets import MASK, SecretPlaceholderError

    monkeypatch.setattr(config_schema, "unknown_keys", lambda section, payload: ["sla.bogus"])
    monkeypatch.setattr(config_schema, "known_keys_hint", lambda section, path: ["max_p95_ms"])
    with with_locale("en"):
        assert config_schema.warnings_for("sla", {}) == [
            'Unknown key "sla.bogus", possibly a typo; it will be saved but not used. Known keys: max_p95_ms.'
        ]
        assert str(SecretPlaceholderError("llm.key")) == f'Secret "llm.key" is not set: replace {MASK} with the real value or leave the field empty'
        with pytest.raises(ValueError) as verdict:
            save_manual_verdict(None, "public", "llm_reports", "r1", "bogus", "why", "me")
        assert str(verdict.value) == "The verdict must be one of: " + ", ".join(MANUAL_VERDICTS)
        with pytest.raises(CapacityModelError, match="^The closed model needs the VU series$"):
            _require_closed_vus(pd.DataFrame({"x": [1.0]}))
        with pytest.raises(CapacityModelError, match="^Concurrency is zero at every point$"):
            fit_usl(np.array([0.0]), np.array([0.0]))
    assert config_schema.warnings_for("sla", {}) == [
        "Неизвестный ключ «sla.bogus» — возможно, опечатка; он будет сохранён, но не используется. Известные ключи: max_p95_ms."
    ]
