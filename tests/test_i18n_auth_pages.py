# SPDX-License-Identifier: Apache-2.0
"""en rendering of the pages without a signed-in user: login, recovery, the storage outage page and forbidden.

Texts that come from the server (storage state, login errors, provider labels, the forbidden message) and the
base.html chrome of the forbidden page are translated elsewhere; the tests allow exactly those fragments.
"""

import json
import re
import sys
import time as real_time
from pathlib import Path

import psycopg2
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import i18n as engine  # noqa: E402
from AI import storage_state  # noqa: E402
from loadlens_app.auth import Actor, InMemoryRepository, Role, passwords, web  # noqa: E402
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text  # noqa: E402
from tests.js_harness import REPO_ROOT, run_js  # noqa: E402
from tests.test_recovery_mode import ADMIN, ADMIN_PASSWORD, OLD, StoreRepository  # noqa: E402

PASSWORD = "Correct-Horse-1"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setitem(core.CONFIG, "storage", {"timescale": dict(OLD)})
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    for name in ("LOADLENS_RECOVERY_USER", "LOADLENS_RECOVERY_PASSWORD_HASH", "LOADLENS_ADMIN_USER", "LOADLENS_ADMIN_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(passwords, "_method", lambda: "pbkdf2:sha256:1000")
    monkeypatch.setattr(passwords, "_dummy_hash", "")


@pytest.fixture
def storage_down(monkeypatch):
    """Every storage host times out (the store-backed repository and the breaker see an outage)."""

    def fake_connect(*args, **kwargs):
        host = kwargs.get("host")
        raise psycopg2.OperationalError(f'connection to server at "{host}" (10.0.0.9), port 5432 failed: timeout expired')

    monkeypatch.setattr(psycopg2, "connect", fake_connect)


def make_app(monkeypatch, *, repository=None, bootstrap=False):
    if bootstrap:
        monkeypatch.setenv("LOADLENS_ADMIN_USER", ADMIN)
        monkeypatch.setenv("LOADLENS_ADMIN_PASSWORD", ADMIN_PASSWORD)
    from app import create_app

    flask_app = create_app(auth_repository=repository if repository is not None else InMemoryRepository())
    flask_app.config["TESTING"] = True
    return flask_app


def en_client(flask_app):
    client = flask_app.test_client()
    use_locale(client, "en")
    return client


def html_lang(html):
    return re.search(r'<html lang="([^"]*)"', html).group(1)


def server_storage_texts():
    """What AI/storage_state.py says about the outage (translated by the server side, not by the templates)."""
    status = storage_state.status()
    return tuple(text for text in (status.title, (status.title or "").lower(), status.hint) if text)


def provider_labels(flask_app):
    return tuple(p.label for p in flask_app.extensions[web.EXTENSION_KEY].providers if p.label)


def without(html, pattern):
    """The page without the element that holds a server message."""
    stripped, count = re.subn(pattern, "", html, flags=re.S)
    assert count, f"no match for {pattern!r}"
    return stripped


def assert_english(response, allow=()):
    html = response.get_data(as_text=True)
    assert html_lang(html) == "en"
    assert_no_cyrillic(visible_text(html), allow=allow)
    return html


# ---- login ----------------------------------------------------------------------------------------------


def test_login_page_without_users_is_english(monkeypatch):
    flask_app = make_app(monkeypatch)
    response = en_client(flask_app).get("/login")
    assert response.status_code == 200
    html = assert_english(response, allow=provider_labels(flask_app))
    assert "<title>Sign in — LoadLens</title>" in html
    assert "<strong>No users yet.</strong>" in html
    assert "Or set the <code>LOADLENS_ADMIN_USER</code> and <code>LOADLENS_ADMIN_PASSWORD</code> variables" in html
    assert '<label for="username">Username</label>' in html and '<label for="password">Password</label>' in html
    assert '<button type="submit" class="btn btn-primary btn-lg">Sign in</button>' in html


def test_login_error_and_recovered_notice_are_english(monkeypatch):
    flask_app = make_app(monkeypatch)
    flask_app.extensions[web.EXTENSION_KEY].create_user(
        username="alice", role=Role.VIEWER, password=PASSWORD, must_change_password=False, actor=Actor(name="test")
    )
    client = en_client(flask_app)
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/login").get_data(as_text=True)).group(1)
    failed = client.post("/login", data={"username": "alice", "password": "nope-nope-nope", "csrf_token": csrf})
    assert failed.status_code == 401
    html = failed.get_data(as_text=True)
    assert html_lang(html) == "en"
    # The error text itself comes from the auth views.
    assert_no_cyrillic(visible_text(without(html, r'<div class="auth-alert" role="alert">.*?</div>')), allow=provider_labels(flask_app))

    recovered = client.get("/login?recovered=1")
    html = assert_english(recovered, allow=provider_labels(flask_app))
    assert "The storage is available again. Sign in as usual." in html


@pytest.mark.parametrize("configured", [False, True])
def test_login_during_an_outage_is_english(monkeypatch, storage_down, configured):
    flask_app = make_app(monkeypatch, repository=StoreRepository(), bootstrap=configured)
    response = en_client(flask_app).get("/login")
    assert response.status_code == 200
    html = assert_english(response, allow=server_storage_texts())
    assert "<strong>Storage unavailable: " in html
    assert "Regular sign-in is temporarily unavailable" in html
    if configured:
        assert "Sign in as the recovery administrator. The session lasts up to 30 minutes" in html
        assert ">Sign in for recovery</button>" in html
    else:
        assert "<strong>No recovery administrator is configured.</strong>" in html
        assert "In <code>settings.py</code>, section <code>auth</code>: the <code>recovery_user</code> and" in html
        assert 'action="/login"' not in html


# ---- storage outage page ------------------------------------------------------------------------------------


@pytest.mark.parametrize("configured", [False, True])
def test_storage_unavailable_page_is_english(monkeypatch, storage_down, configured):
    client = en_client(make_app(monkeypatch, repository=StoreRepository(), bootstrap=configured))
    with client.session_transaction() as session:
        session.update(uid=1, epoch=0, iat=int(real_time.time()), csrf="t")
    response = client.get("/")
    assert response.status_code == 503
    html = assert_english(response, allow=server_storage_texts())
    assert "<title>Storage unavailable — LoadLens</title>" in html
    # The number and the unit stay on one line, as in ru.
    assert re.search(r"The page will check the connection again in \d+&nbsp;s\.</p>", html)
    if configured:
        assert ">Sign in as the recovery administrator</a>" in html
        assert "Recovery sign-in opens only the storage connection settings." in html
    else:
        assert "The <code>LOADLENS_RECOVERY_USER</code> and <code>LOADLENS_RECOVERY_PASSWORD_HASH</code> environment" in html


def _outage_page_without_auth(monkeypatch, locale):
    from flask import request

    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    flask_app = make_app(monkeypatch, repository=StoreRepository())
    assert flask_app.test_client().get("/runs").status_code == 503  # the breaker learns about the outage
    # Without sign-in a page reaches the outage page only through a storage error inside a view: render it directly.
    with flask_app.test_request_context("/", headers={"Cookie": f"loadlens.lang={locale}"}):
        return web.storage_unavailable_response(web.rule_for(request.endpoint))


def test_storage_unavailable_page_without_auth_shows_details_in_english(monkeypatch, storage_down):
    response = _outage_page_without_auth(monkeypatch, "en")
    html = assert_english(response, allow=server_storage_texts())
    assert "<dt>Address</dt>" in html and "<dt>Response</dt>" in html
    assert '<a class="btn btn-primary btn-lg" href="/recovery">Fix the connection</a>' in html


def test_storage_unavailable_page_keeps_the_russian_text(monkeypatch, storage_down):
    html = _outage_page_without_auth(monkeypatch, "ru").get_data(as_text=True)
    assert '<html lang="ru"' in html and "<title>Хранилище недоступно — LoadLens</title>" in html
    assert re.search(r"Страница проверит подключение снова через \d+&nbsp;с\.</p>", html)
    assert "<dt>Адрес</dt><dd><code>db.old:5432/lt</code></dd>" in html


# ---- recovery page -----------------------------------------------------------------------------------------


def _recovery_session(monkeypatch, locale):
    flask_app = make_app(monkeypatch, repository=StoreRepository(), bootstrap=True)
    client = flask_app.test_client()
    use_locale(client, locale)
    csrf = client.get("/auth/me").get_json()["csrf_token"]
    signed_in = client.post("/login", json={"username": ADMIN, "password": ADMIN_PASSWORD}, headers={"X-CSRF-Token": csrf})
    assert signed_in.status_code == 200 and signed_in.get_json()["recovery"] is True
    return client


def test_recovery_page_in_a_recovery_session_is_english(monkeypatch, storage_down):
    response = _recovery_session(monkeypatch, "en").get("/recovery")
    assert response.status_code == 200
    html = assert_english(response, allow=server_storage_texts())
    assert "<title>Connection recovery — LoadLens</title>" in html
    assert '<button type="submit" class="btn">Sign out</button>' in html
    assert "The configuration has a <code>dsn</code> connection string." in html
    assert "It lasts up to 30 minutes and ends by itself" in html
    # The page script gets the recovery messages in English.
    assert '"recovery.save.no_answer":"no response"' in html


def test_recovery_page_without_auth_is_english(monkeypatch, storage_down):
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    client = en_client(make_app(monkeypatch, repository=StoreRepository()))
    client.get("/runs")  # the breaker learns about the outage
    response = client.get("/recovery")
    assert response.status_code == 200
    html = assert_english(response, allow=server_storage_texts())
    assert '<a class="btn" href="/settings#connections">All settings</a>' in html
    assert "Changes are written to the audit log, or to the application log while the storage is unavailable." in html


def test_recovery_page_keeps_the_russian_text(monkeypatch, storage_down):
    html = _recovery_session(monkeypatch, "ru").get("/recovery").get_data(as_text=True)
    assert "<title>Восстановление подключения — LoadLens</title>" in html
    assert "Она действует до 30 минут и завершится сама" in html
    assert "В конфигурации задана строка подключения <code>dsn</code>. После сохранения её заменят поля этой формы." in html


@pytest.mark.parametrize(
    "lang, n, text",
    [("ru", 1, "до 1 минуты"), ("ru", 3, "до 3 минут"), ("ru", 30, "до 30 минут"), ("en", 1, "up to 1 minute and"), ("en", 30, "up to 30 minutes")],
)
def test_recovery_session_note_takes_the_plural_of_the_minutes(lang, n, text):
    assert text in engine.plural("recovery.note.session", n, locale=lang)


# ---- forbidden -------------------------------------------------------------------------------------------------


def test_forbidden_page_is_english_in_its_own_part(monkeypatch):
    flask_app = make_app(monkeypatch)
    flask_app.extensions[web.EXTENSION_KEY].create_user(
        username="vic", role=Role.VIEWER, password=PASSWORD, must_change_password=False, actor=Actor(name="test")
    )
    client = en_client(flask_app)
    csrf = client.get("/auth/me").get_json()["csrf_token"]
    assert client.post("/login", json={"username": "vic", "password": PASSWORD}, headers={"X-CSRF-Token": csrf}).status_code == 200
    response = client.get("/settings")
    assert response.status_code == 403
    html = response.get_data(as_text=True)
    assert html_lang(html) == "en" and "<title>Access denied — LoadLens</title>" in html
    # base.html (the top bar) and the reason from the auth gate are translated elsewhere: check the page block only.
    block = re.search(r'<section class="forbidden-page">.*?</section>', html, re.S).group(0)
    assert_no_cyrillic(visible_text(without(block, r'<p class="forbidden-text">.*?</p>')))
    assert "<h1>Access denied</h1>" in block and ">Go to the dashboard</a>" in block


# ---- recovery.js ------------------------------------------------------------------------------------------------

_RECOVERY_DRIVER = r"""
(async () => {
  const ids = ['storageForm', 'formResult', 'checkBtn', 'saveBtn', 'storageState', 'storageDetails', 'storageTitle',
               'storageHint', 'storageAddress', 'storageMessage', 'dsnNote'];
  const nodes = {};
  ids.forEach((id) => { nodes[id] = document.createElement('div'); });
  nodes.storageForm.elements = {};
  ['host', 'port', 'dbname', 'schema', 'user', 'password', 'sslmode', 'connect_timeout_sec'].forEach((name) => {
    nodes.storageForm.elements[name] = { value: '' };
  });
  const results = [];
  Object.defineProperty(nodes.formResult, 'textContent', { get: () => results[results.length - 1] || '', set: (v) => { results.push(v); } });
  document.getElementById = (id) => nodes[id] || null;

  const replies = [];
  window.fetch = async () => {
    const reply = replies.shift();
    if (!reply || reply.fail) throw new Error('network down');
    return { status: reply.status || 200, ok: (reply.status || 200) < 400, json: async () => reply.body || {} };
  };
  let poll = null;
  window.setInterval = (cb) => { poll = cb; return 1; };
  const tick = () => new Promise((resolve) => setTimeout(resolve, 5));
  const titles = [];
  const step = async (reply, action) => { replies.push(reply); action(); await tick(); titles.push(nodes.storageTitle.textContent); };

  replies.push({ fail: true });
  (0, eval)(SOURCE);
  await tick();
  await step({ status: 500 }, () => poll());
  await step({ body: { storage: {}, form: {} } }, () => poll());
  const click = () => nodes.checkBtn.dispatchEvent({ type: 'click' });
  const submit = () => nodes.storageForm.dispatchEvent({ type: 'submit', preventDefault() {} });
  await step({ status: 400 }, click);
  await step({ body: { ok: true, message: 'Connected', elapsed_ms: 42 } }, click);
  await step({ fail: true }, click);
  await step({ status: 400 }, submit);
  await step({ body: { saved: true, available: false, storage: { available: false } } }, submit);
  await step({ body: { saved: true, available: false, storage: { available: false, title: 'Timeout' } } }, submit);
  await step({ fail: true }, submit);
  await step({ body: { saved: true, available: true, redirect: '/login?recovered=1' } }, submit);
  return { results, titles, hint: nodes.storageHint.textContent };
})()
"""


def run_recovery_js(locale):
    source = (REPO_ROOT / "static" / "js" / "recovery.js").read_text(encoding="utf-8")
    return run_js([], _RECOVERY_DRIVER.replace("SOURCE", json.dumps(source)), locale=locale)


def test_recovery_js_messages_in_english():
    out = run_recovery_js("en")
    assert not [text for text in out["results"] + out["titles"] + [out["hint"]] if CYRILLIC.search(text or "")]
    assert out["results"] == [
        "The application server is not responding. Retrying in a few seconds.",
        "Could not get the storage state.",
        "Testing the connection…",
        "The check did not run.",
        "Testing the connection…",
        "Connected (42 ms)",
        "Testing the connection…",
        "The application server is not responding.",
        "Saving and connecting…",
        "The settings were not saved.",
        "Saving and connecting…",
        "Saved, but the storage is still unavailable: no response.",
        "Saving and connecting…",
        "Saved, but the storage is still unavailable: timeout.",
        "Saving and connecting…",
        "The application server is not responding.",
        "Saving and connecting…",
        "Saved. The storage is available, moving on…",
    ]
    assert out["titles"][1] == "The connection has not been checked yet"
    assert out["titles"][-1] == "Storage is available" and out["hint"] == "The connection works."


def test_recovery_js_messages_keep_the_russian_text():
    out = run_recovery_js("ru")
    assert out["results"][0] == "Сервер приложения не отвечает. Повторим через несколько секунд."
    assert out["results"][5] == "Connected (42 мс)"
    assert out["results"][11] == "Сохранено, но хранилище всё ещё недоступно: нет ответа."
    assert out["results"][13] == "Сохранено, но хранилище всё ещё недоступно: timeout."
    assert out["results"][-1] == "Сохранено. Хранилище доступно, переходим дальше…"
    assert out["titles"][1] == "Подключение ещё не проверялось"
    assert out["titles"][-1] == "Хранилище доступно" and out["hint"] == "Подключение работает."
