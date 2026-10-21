# SPDX-License-Identifier: Apache-2.0
"""Localization of the account page and of the user administration (plans/i18n-plan.md 2.5, row 8).

/account renders in en without Russian text (the page part; the sidebar belongs to base.html), role names come
from common.role.* instead of the Russian labels the server sends, and the JS builders of account.js and users.js
speak the UI language. Without a cookie the Russian output stays as it was.
"""

import re

import pytest

from loadlens_app.auth import Role
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text
from tests.js_harness import run_js
from tests.test_auth import Browser, _isolated, add_user, app, logged_in, repo, service  # noqa: F401  (fixtures)

BASE = ["static/js/common.js", "static/js/ui.js"]
USERS_JS = BASE + ["static/js/users.js"]
ACCOUNT_JS = BASE + ["static/js/account.js"]


def _page_part(html):
    """(<title>, the page's own markup): from the page header to </main>; the sidebar and banners are base.html."""
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1).strip()
    return title, html[html.index('<div class="page-header">'):html.index("</main>")]


def _account(browser, url="/account", lang=None):
    use_locale(browser.client, lang)
    response = browser.get(url)
    assert response.status_code == 200
    return response.get_data(as_text=True)


# ---- /account ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role, fact, options",
    [
        ("viewer", "Viewer", ["Viewer"]),
        ("engineer", "Engineer", ["Viewer", "Engineer"]),
        ("admin", "Administrator", ["Viewer", "Engineer", "Administrator"]),
    ],
)
def test_account_page_in_en_has_no_russian_text(logged_in, role, fact, options):
    html = _account(logged_in(role), lang="en")
    assert re.search(r'<html lang="en"', html)
    title, page = _page_part(html)
    assert title == "Profile — LoadLens"
    assert_no_cyrillic(title + "\n" + visible_text(page))
    assert f"<dt>Role</dt><dd>{fact}</dd>" in page
    assert re.findall(r'<option value="\w+"(?: selected)?>([^<]*)</option>', page) == options
    assert "Pass the token in the <code>Authorization: Bearer &lt;token&gt;</code> header." in page
    assert '<div class="field-hint">At least 10 characters, must not contain the username.</div>' in page
    assert 'placeholder="e.g. CI nightly"' in page
    # The page declares its namespace: account.js finds its messages in the inline dictionary.
    assert '"account.tokens.status.active":"active"' in html and '"common.role.viewer":"Viewer"' in html


def test_forced_change_and_external_accounts_in_en(app, service, repo, logged_in):
    add_user(service, "forced", Role.ENGINEER, must_change=True)
    forced = Browser(app)
    use_locale(forced.client, "en")
    assert forced.login("forced").status_code == 200
    title, page = _page_part(_account(forced, "/account?force=1", lang="en"))
    assert "Password change required" in page and 'id="h-tokens"' not in page
    assert_no_cyrillic(title + "\n" + visible_text(page))

    external = logged_in("engineer")
    repo._users[repo.get_by_username(external.username).id].provider = "ldap"
    service.invalidate_user()
    title, page = _page_part(_account(external, lang="en"))
    assert "<dt>Source</dt><dd>ldap</dd>" in page and "managed by an external provider" in page
    assert_no_cyrillic(title + "\n" + visible_text(page))
    title, page = _page_part(_account(external, lang="ru"))
    assert "<dt>Источник</dt><dd>ldap</dd>" in page
    assert "Пароль этой учётной записи управляется внешним провайдером и здесь не меняется." in page


def test_account_page_in_ru_keeps_its_text(logged_in):
    html = _account(logged_in("admin"))
    assert re.search(r'<html lang="ru"', html)
    assert "<title>Профиль — LoadLens</title>" in html
    assert "<dt>Роль</dt><dd>Администратор</dd>" in html and "<dt>Источник</dt><dd>локальная</dd>" in html
    assert '<option value="viewer" selected>Наблюдатель</option>' in html
    assert "Передавайте его в заголовке <code>Authorization: Bearer &lt;токен&gt;</code>. Полное значение" in html
    assert '<div class="field-hint">Не короче 10 символов, без логина внутри.</div>' in html
    assert "<th>Начало токена</th>" in html and '<td colspan="8">Загрузка…</td>' in html


# ---- JS builders -------------------------------------------------------------------------------------------

# Fakes: dialogs/confirms/toasts are recorded in `out`, document.getElementById serves plain objects and fetch
# answers from `responses` (or fails with "boom").
PRELUDE = r"""
const out = [];
window.crypto = { getRandomValues: (a) => { a.fill(7); return a; } };
LL.ui.dialog = (cfg) => { out.push({ dialog: cfg.title, body: cfg.bodyHtml, labels: (cfg.actions || []).map((a) => a.label) }); return Promise.resolve(true); };
LL.ui.confirm = (cfg) => { out.push({ confirm: cfg.title, message: cfg.message, ok: cfg.confirmText }); return Promise.resolve(true); };
LL.ui.toast = (message) => { out.push({ toast: message }); };
const els = {};
['usersBody', 'usersStatus', 'allTokensBody', 'auditBody', 'auditStatus', 'tokensBody'].forEach((id) => {
  els[id] = { innerHTML: '', textContent: '', querySelector: () => null, addEventListener() {} };
});
document.getElementById = (id) => els[id] || null;
let responses = {};
let failing = false;
window.fetch = async (url) => (failing
  ? { ok: false, status: 500, json: async () => ({ error: 'boom' }) }
  : { ok: true, status: 200, json: async () => (responses[url] || {}) });
const ROLES = [{ key: 'viewer', label: 'Наблюдатель' }, { key: 'engineer', label: 'Инженер' }, { key: 'admin', label: 'Администратор' }];
const USERS = [
  { id: 1, username: 'root', display_name: 'Root', email: 'r@x.io', role: 'admin', provider: 'local', is_active: true, locked: false, must_change_password: false },
  { id: 2, username: 'bob', display_name: '', email: '', role: 'engineer', provider: 'local', is_active: true, locked: true, must_change_password: false },
  { id: 3, username: 'eve', display_name: 'Eve', email: '', role: 'viewer', provider: 'ldap', is_active: false, locked: false, must_change_password: false },
  { id: 4, username: 'max', display_name: 'Max', email: '', role: 'viewer', provider: 'local', is_active: true, locked: false, must_change_password: true },
];
const TOKENS = [
  { id: 7, name: 'ci', owner: 'root', prefix: 'll_ab', role: 'engineer', active: true, created_at: '2026-10-01T10:00:00Z', expires_at: '2099-01-01T00:00:00Z' },
  { id: 8, name: 'old', owner: 'bob', prefix: 'll_cd', role: 'viewer', active: false, revoked_at: '2026-10-02T00:00:00Z', expires_at: '2099-01-01T00:00:00Z' },
  { id: 9, name: 'gone', owner: 'eve', prefix: 'll_ef', role: 'admin', active: false, expires_at: '2020-01-01T00:00:00Z' },
];
const EVENTS = [
  { created_at: '2026-10-01T10:00:00Z', actor: 'root', action: 'user.create', success: true, target: 'bob' },
  { created_at: '2026-10-01T11:00:00Z', actor: 'bob', via: 'token', action: 'login.failed', success: false },
];
"""

USERS_RUN = PRELUDE + r"""
(async () => {
  const I = LL.users._internals;
  I.setState({ users: USERS, roles: ROLES, currentUserId: 1 });
  USERS.forEach((u) => out.push(I.userRow(u)));
  TOKENS.forEach((t) => out.push(I.tokenRow(t)));
  EVENTS.forEach((e) => out.push(I.auditRow(e)));
  ['login', 'logout', 'user.provision', 'config.query_preview', 'project.override_delete', 'forecast.publish'].forEach((a) => out.push(I.actionLabel(a)));
  await I.createDialog();
  for (const name of ['edit', 'password', 'unlock', 'toggle', 'delete']) await I.ACTIONS[name](USERS[1]);
  await I.onTokensClick({ target: { closest: () => ({ dataset: { name: 'ci', revoke: '7' } }) } });
  responses = { '/auth/audit?limit=100': { events: EVENTS } };
  await I.loadAudit(); out.push(els.auditStatus.textContent);
  responses = { '/auth/users': { users: [], roles: ROLES }, '/auth/tokens?all=1': { tokens: [] }, '/auth/audit?limit=100': { events: [] } };
  await I.loadUsers(); await I.loadAllTokens(); await I.loadAudit();
  out.push(els.usersBody.innerHTML, els.allTokensBody.innerHTML, els.auditBody.innerHTML);
  failing = true;
  await I.loadUsers(); await I.loadAllTokens(); await I.loadAudit();
  out.push(els.usersBody.innerHTML, els.allTokensBody.innerHTML, els.auditStatus.textContent);
  return out;
})()
"""

ACCOUNT_RUN = PRELUDE + r"""
(async () => {
  const I = LL.account._internals;
  TOKENS.forEach((t) => out.push(I.tokenRow(t)));
  out.push(I.passwordProblem('short', 'short', 12), I.passwordProblem('long-enough-1', 'other', 12), I.passwordProblem('long-enough-1', 'long-enough-1', 12));
  await I.showNewToken('ll_secret');
  responses = { '/auth/tokens': { tokens: [] } };
  await I.loadTokens(); out.push(els.tokensBody.innerHTML);
  failing = true;
  await I.loadTokens(); out.push(els.tokensBody.innerHTML);
  return out;
})()
"""


def _text(values):
    return "\n".join(str(v) for v in values)


@pytest.mark.parametrize("files, script", [(USERS_JS, USERS_RUN), (ACCOUNT_JS, ACCOUNT_RUN)], ids=["users", "account"])
def test_js_builders_in_en_have_no_russian_text(files, script):
    out = run_js(files, script, locale="en")
    assert_no_cyrillic(_text(out))
    assert out  # the run reached the end


def test_users_js_in_en():
    out = run_js(USERS_JS, USERS_RUN, locale="en")
    text = _text(out)
    assert '<span class="pill ok">you</span>' in out[0] and "<td>Administrator</td>" in out[0] and "<td>local</td>" in out[0]
    assert ">Unlock</button>" in out[1] and '<span class="pill warn">locked</span>' in out[1]
    assert '<span class="pill na">disabled</span>' in out[2] and ">Enable</button>" in out[2] and "<td>ldap</td>" in out[2]
    assert '<span class="pill warn">password change pending</span>' in out[3]
    assert [s for s in ("active", "revoked", "expired") if f'">{s}</span>' in text] == ["active", "revoked", "expired"]
    assert "<td>User created</td>" in out[7] and 'Failed sign-in <span class="pill fail">denied</span>' in out[8]
    assert out[9:15] == ["Sign-in", "Sign-out", "User created by provider", "Query preview", "Override reset", "Forecast publication"]
    dialog = out[15]
    assert dialog["dialog"] == "New user" and dialog["labels"] == ["Cancel", "Create"]
    assert '<label>Email <span class="optional">(optional)</span>' in dialog["body"]
    assert '<option value="viewer" selected>Viewer</option>' in dialog["body"]
    assert {"toast": "Saved"} in out and {"toast": "Password changed"} in out and {"toast": "User unlocked"} in out
    reset = next(item for item in out if isinstance(item, dict) and item.get("dialog") == "Reset password")
    assert reset["body"].startswith("<p>New password for <strong>bob</strong>. All of their sessions will be ended.</p>")
    assert reset["labels"] == ["Cancel", "Set password"]
    confirms = [item for item in out if isinstance(item, dict) and "confirm" in item]
    assert [c["confirm"] for c in confirms] == ["Disable user", "Delete user", "Revoke token"]
    assert [c["ok"] for c in confirms] == ["Disable", "Delete", "Revoke"]
    assert confirms[0]["message"].startswith("“bob” will not be able to sign in")
    assert "Latest 2" in out and "Error: boom" in out
    assert any("No users" in str(v) for v in out) and any("Could not load users: boom" in str(v) for v in out)


def test_users_js_in_ru_keeps_its_text():
    out = run_js(USERS_JS, USERS_RUN)
    assert '<span class="pill ok">это вы</span>' in out[0] and "<td>Администратор</td>" in out[0] and "<td>локальная</td>" in out[0]
    assert '>Разблокировать</button>' in out[1] and '>Удалить…</button>' in out[1]
    assert '<span class="pill warn">ждёт смены пароля</span>' in out[3]
    assert 'Неудачный вход <span class="pill fail">отказ</span>' in out[8]
    assert out[9:15] == ["Вход", "Выход", "Пользователь создан провайдером", "Предпросмотр запроса", "Сброс переопределения", "Публикация прогноза"]
    assert out[15]["dialog"] == "Новый пользователь" and out[15]["labels"] == ["Отмена", "Создать"]
    assert '<label>Email <span class="optional">(необязательно)</span>' in out[15]["body"]
    assert '<label>Логин<input id="ufUsername"' in out[15]["body"]
    reset = next(item for item in out if isinstance(item, dict) and item.get("dialog") == "Сброс пароля")
    assert reset["body"].startswith("<p>Новый пароль для <strong>bob</strong>. Все его сессии будут завершены.</p>")
    confirms = [item for item in out if isinstance(item, dict) and "confirm" in item]
    assert confirms[1]["message"] == (
        "Учётная запись «bob» и её API-токены будут удалены безвозвратно. Если нужно лишь запретить вход, используйте «Отключить»."
    )
    assert [c["ok"] for c in confirms] == ["Отключить", "Удалить", "Отозвать"]
    assert "Последние 2" in out and "Ошибка: boom" in out and {"toast": "Сохранено"} in out


def test_account_js_in_en_and_ru():
    en = run_js(ACCOUNT_JS, ACCOUNT_RUN, locale="en")
    assert "<td>Engineer</td>" in en[0] and ">Revoke</button>" in en[0] and '<span class="pill ok">active</span>' in en[0]
    assert '<span class="pill na">revoked</span>' in en[1] and '<span class="pill warn">expired</span>' in en[2]
    assert en[3:6] == ["The password must be at least 12 characters long", "The new password and its repetition do not match", ""]
    assert en[6] == {
        "dialog": "Token created",
        "body": '<p>Copy the token now: it cannot be shown again later.</p><code class="token-secret" id="newTokenValue">ll_secret</code>',
        "labels": ["Copy", "Close"],
    }
    assert "No tokens yet" in en[7] and "Could not load tokens: boom" in en[8]

    ru = run_js(ACCOUNT_JS, ACCOUNT_RUN)
    assert "<td>Инженер</td>" in ru[0] and ">Отозвать</button>" in ru[0] and '<span class="pill ok">действует</span>' in ru[0]
    assert ru[3:6] == ["Пароль должен содержать не менее 12 символов", "Новый пароль и повтор не совпадают", ""]
    assert ru[6]["dialog"] == "Токен создан" and ru[6]["labels"] == ["Копировать", "Закрыть"]
    assert "Токенов пока нет" in ru[7] and "Не удалось загрузить токены: boom" in ru[8]


def test_unknown_role_keeps_the_server_label():
    script = "LL.users._internals.setState({ users: [], roles: [{ key: 'auditor', label: 'Auditor (srv)' }], currentUserId: null });" \
             " [LL.users._internals.roleLabel('auditor'), LL.users._internals.roleLabel('admin'), LL.users._internals.roleLabel('x')]"
    assert run_js(USERS_JS, script, locale="en") == ["Auditor (srv)", "Administrator", "x"]
    assert not CYRILLIC.search(" ".join(run_js(ACCOUNT_JS, "['viewer', 'engineer', 'admin', 'x'].map(LL.account._internals.roleLabel)", locale="en")))


def test_settings_page_delivers_the_users_namespace(logged_in):
    """users.js renders inside the settings page: its messages must be in that page's inline dictionary."""
    admin = logged_in("admin")
    use_locale(admin.client, "en")
    response = admin.get("/settings")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '"users.status.active":"active"' in html and '"common.action.cancel":"Cancel"' in html
    assert "js/users.js" in html
