# SPDX-License-Identifier: Apache-2.0
"""Localization of the page frame (plans/i18n-plan.md 2.5, row 0): base.html and the shared scripts common.js, ui.js,
palette.js and charts.js.

With the ``loadlens.lang=en`` cookie the frame of every page (sidebar, theme and user menus, the storage banner) has
no Russian text, role names come from ``common.role.*``, and the JS helpers speak English (the bilingual search
keywords of the palette excepted). Without a cookie the Russian output stays as it was.
"""

import json
import re

import pytest

from AI import storage_state
from loadlens_app.auth import Role
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text
from tests.js_harness import REPO_ROOT, run_js
from tests.test_auth import Browser, _isolated, add_user, app, logged_in, repo, service  # noqa: F401  (fixtures)

BASE_JS = ["static/js/common.js", "static/js/ui.js"]
SHELL_JS = BASE_JS + ["static/js/palette.js", "static/js/charts.js"]
ADMIN_PAGES = ["/", "/reports", "/compare", "/forecasting", "/launch", "/new", "/settings", "/account"]


def _html_lang(html):
    return re.search(r'<html lang="([^"]*)"', html).group(1)


def _chrome(html):
    """The base.html parts of a page: the sidebar and top bar, the storage banner, and what follows </main>."""
    head = html[html.index("<body"):html.index("<main")]
    banner = re.search(r'<div class="banner banner-fail" id="storageDownBanner".*?</div>\s*</div>', html, re.S)
    tail = html[html.index("</main>"):]
    return "\n".join([head, banner.group(0) if banner else "", tail])


def _without_tokens(html):
    return re.sub(r'(nonce|content)="[^"]*"', "", html)


def _page(browser, url, lang=None, status=200):
    use_locale(browser.client, lang)
    response = browser.get(url)
    assert response.status_code == status, url
    return response.get_data(as_text=True)


# ---- base.html -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("role, label", [("viewer", "Viewer"), ("engineer", "Engineer"), ("admin", "Administrator")])
def test_frame_of_every_page_is_english(logged_in, role, label):
    browser = logged_in(role)
    pages = ADMIN_PAGES if role == "admin" else ["/", "/reports", "/account"]
    for url in pages:
        html = _page(browser, url, "en")
        assert _html_lang(html) == "en", url
        frame = _chrome(html)
        assert_no_cyrillic(visible_text(frame))
        assert f'<span class="user-role">{label}</span>' in frame
        assert f"{browser.username} · {label}</div>" in frame
    assert '<a class="skip-link" href="#main">Skip to content</a>' in frame
    assert '<span class="nav-label">Search</span>' in frame and ">Sign out</button>" in frame
    assert 'href="/account">Profile and API tokens</a>' in frame
    assert '<option value="">All areas</option>' in frame and 'aria-label="Project area"' in frame


def test_nav_labels_and_theme_menu_in_both_languages(logged_in):
    browser = logged_in("admin")
    labels = lambda html: re.findall(r'<span class="nav-label">([^<]*)</span>', html)  # noqa: E731
    en = _page(browser, "/", "en")
    # Stage 5: the theme menu became the "View" menu with a theme group and a language group.
    assert labels(en) == ["Dashboard", "New report", "Run a test", "Archive", "Compare tests", "Forecasting", "Settings", "Search", "View"]
    for text in ("<span>System</span>", "<span>Light</span>", "<span>Dark</span>", 'title="Interface theme and language"',
                 'aria-label="Interface theme"', 'aria-label="Interface language"', "<span>English</span>", "<span>Русский</span>"):
        assert text in en
    ru = _page(browser, "/", "ru")
    assert labels(ru) == ["Дэшборд", "Новый отчёт", "Запуск теста", "Архив", "Сравнение тестов", "Прогнозирование", "Настройки", "Поиск", "Вид"]
    for text in ("<span>Как в системе</span>", "<span>Светлая</span>", "<span>Тёмная</span>", '<a class="skip-link" href="#main">К содержимому</a>',
                 '<span class="user-role">Администратор</span>', 'aria-label="LoadLens — на дэшборд"', 'aria-label="Открыть меню"'):
        assert text in ru
    assert _without_tokens(ru) == _without_tokens(_page(browser, "/"))  # no cookie is ru


def test_forbidden_page_frame_is_english(app, service):
    add_user(service, "vic", Role.VIEWER)
    browser = Browser(app)
    use_locale(browser.client, "en")
    assert browser.login("vic").status_code == 200
    html = _page(browser, "/settings", "en", status=403)
    assert _html_lang(html) == "en"
    assert_no_cyrillic(visible_text(_chrome(html)))
    assert '<span class="user-role">Viewer</span>' in html
    assert 'href="/settings"' not in _chrome(html)  # a viewer gets no link to the settings


def test_storage_banner_passes_the_link_as_a_parameter(logged_in, monkeypatch):
    monkeypatch.setattr(storage_state, "is_down", lambda: True)
    browser = logged_in("admin")
    en = _chrome(_page(browser, "/", "en"))
    assert '<div class="banner-title">Storage unavailable</div>' in en
    assert 'Reports and the archive cannot be loaded right now. <a href="/recovery">Check and fix the connection</a>' in en
    assert_no_cyrillic(visible_text(en))
    ru = _chrome(_page(browser, "/", "ru"))
    assert '<div class="banner-title">Хранилище недоступно</div>' in ru
    assert re.search(r'\n\s+Отчёты и архив сейчас не загружаются\. <a href="/recovery">Проверить и исправить подключение</a>\n', ru.replace("\r\n", "\n"))


def test_pages_with_charts_declare_the_charts_namespace(logged_in):
    declared = re.compile(r"\{%-?\s*set\s+i18n_ns\s*=\s*\[(.*?)\]\s*-?%\}", re.S)
    for page in sorted((REPO_ROOT / "templates").glob("*.html")):
        source = page.read_text(encoding="utf-8")
        if "js/charts.js" not in source:
            continue
        match = declared.search(source)
        assert match and "charts" in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)), f"{page.name} loads charts.js without i18n_ns 'charts'"
    html = _page(logged_in("viewer"), "/", "en")
    assert '"charts.legend.title":"Legend"' in html and '"ui.theme.system":"System"' in html and '"glossary.p95":' in html


# ---- common.js -------------------------------------------------------------------------------------------------

HELPERS = r"""(async () => {
  const NOW = Date.parse('2026-10-15T12:00:00Z');
  const out = {};
  out.rel = [NOW - 1000, NOW - 5 * 60000, NOW - 3 * 3600000, NOW - 26 * 3600000, NOW - 2 * 86400000, NOW - 3 * 86400000].map((v) => LL.relativeTime(v, NOW));
  out.dur = [45000, 59 * 60000, 65 * 60000, 3600000 * 27].map((v) => LL.formatDuration(v));
  out.tables = [Object.assign({}, LL.DOMAIN_TITLES), Object.assign({}, LL.TEST_TYPE_LABELS), Object.assign({}, LL.VERDICT_LABELS)];
  out.keys = [Object.keys(LL.DOMAIN_TITLES), Object.keys(LL.TEST_TYPE_LABELS), Object.keys(LL.VERDICT_LABELS)];
  out.labels = ['pass', 'Провал', 'есть риски', null].map((v) => LL.verdictLabel(v)).concat(['soak', ''].map((k) => LL.testTypeLabel(k)), ['final', 'x'].map((k) => LL.domainTitle(k)));
  out.manual = [LL.manualVerdictTitle({ verdict: 'fail', author: 'ivanov', changed_at: '2026-10-17T12:00:00Z', reason: 'r' }), LL.manualVerdictTitle({ verdict: 'pass' })];
  LL.role = 'admin';
  const btn = LL.jobDismissButton({ job_id: 'j', run_name: 'run-1', status: 'done' }, () => {});
  out.button = [btn.textContent, btn.title, btn.getAttribute('aria-label'), LL.jobDismissButton({ job_id: 'j', status: 'done' }, () => {}).getAttribute('aria-label')];
  const toasts = [];
  LL.ui.toast = (text, opts) => toasts.push([text, opts.tone]);
  const replies = [{ ok: false, status: 500, body: {} }, { ok: true, body: { dismissed: [1, 2] } }, { ok: true, body: { dismissed: [1], skipped_running: [3] } },
                   { ok: true, body: { skipped_running: [3, 4] } }, { ok: true, body: {} }, null];
  for (const reply of replies) {
    window.fetch = async () => { if (!reply) throw new Error('down'); return { ok: reply.ok, status: reply.status || 200, json: async () => reply.body }; };
    await LL.dismissJobs(['a']);
  }
  out.toasts = toasts;
  return out;
})()"""


def test_common_helpers_speak_english():
    out = run_js(BASE_JS, HELPERS, locale="en")
    assert out["rel"] == ["just now", "5 min ago", "3 h ago", "yesterday at 10:00", "2 days ago", "3 days ago"]
    assert out["dur"] == ["45 s", "59 min", "1 h 05 min", "1 d 3 h"]
    assert out["tables"][2] == {"pass": "Passed", "risk": "Risks found", "fail": "Failed", "no_data": "Insufficient data"}
    assert out["tables"][1]["soak"] == "Soak (long-running)" and out["tables"][0]["hard_resources"] == "Node resources"
    assert out["keys"] == [
        ["final", "jvm", "database", "kafka", "microservices", "hard_resources", "lt_framework", "lt_generators", "application_logs"],
        ["step", "soak", "spike", "stress"], ["pass", "risk", "fail", "no_data"],
    ]
    assert out["labels"] == ["Passed", "Failed", "Risks found", "Insufficient data", "Soak (long-running)", "—", "Summary", "x"]
    assert out["manual"] == ["Changed manually: ivanov, 2026-10-17 12:00. Reason: r", "Changed manually."]
    assert out["button"] == ["Hide", "Remove the job from the list. It stays available by its link.", "Hide job run-1", "Hide job"]
    assert out["toasts"] == [
        ["Could not clear the job list (HTTP 500)", "error"], ["Jobs hidden: 2", "ok"],
        ["Jobs hidden: 1; still running and kept in the list: 1", "warn"], ["Still running and kept in the list: 2", "warn"],
        ["These jobs are already hidden", "ok"], ["The server does not respond; the job list was not cleared", "error"],
    ]
    assert not CYRILLIC.search(json.dumps(out["tables"] + [out["rel"], out["dur"], out["manual"], out["button"], out["toasts"]], ensure_ascii=False))


def test_common_helpers_keep_the_russian_output():
    out = run_js(BASE_JS, HELPERS)
    assert out["rel"] == ["только что", "5 мин назад", "3 ч назад", "вчера в 10:00", "2 дн назад", "3 дн назад"]
    assert out["dur"] == ["45 с", "59 мин", "1 ч 05 мин", "1 д 3 ч"]
    assert out["labels"] == ["Успешно", "Провал", "Есть риски", "Недостаточно данных", "Долговременный (soak)", "—", "Итог", "x"]
    assert out["manual"] == ["Изменён вручную: ivanov, 2026-10-17 12:00. Причина: r", "Изменён вручную."]
    assert out["button"] == ["Скрыть", "Убрать задачу из списка. Она останется доступна по ссылке.", "Скрыть задачу run-1", "Скрыть задачу"]
    assert out["toasts"] == [
        ["Не удалось очистить список задач (HTTP 500)", "error"], ["Скрыто задач: 2", "ok"],
        ["Скрыто задач: 1, ещё выполняются и остались в списке: 1", "warn"], ["ещё выполняются и остались в списке: 2", "warn"],
        ["Эти задачи уже скрыты", "ok"], ["Сервер не отвечает, список задач не очищен", "error"],
    ]


def _outage_banner(locale, role):
    """common.js runs again over a fetch that answers 503 storage_unavailable; returns the banner of #main."""
    source = (REPO_ROOT / "static" / "js" / "common.js").read_text(encoding="utf-8")
    expression = f"""(async () => {{
      const main = document.createElement('main');
      document.getElementById = (id) => (id === 'main' ? main : null);
      document.querySelector = (sel) => (sel === 'meta[name="loadlens-role"]' ? {{ getAttribute: () => {json.dumps(role)} }} : null);
      window.fetch = async () => ({{ ok: false, status: 503, clone() {{ return this; }}, json: async () => ({{ code: 'storage_unavailable' }}) }});
      (0, eval)({json.dumps(source)});
      await fetch('/runs');
      await new Promise((resolve) => setTimeout(resolve, 0));
      const inner = main.children[0].children[0];
      return inner.children.map((node) => (node.tagName ? [node.tagName, node.textContent, node.href || ''] : ['#text', node.textContent]));
    }})()"""
    return run_js(BASE_JS, expression, locale=locale)


def test_storage_outage_banner_of_data_calls():
    assert _outage_banner("en", "admin") == [
        ["DIV", "Storage unavailable", ""], ["#text", "Reports and the archive cannot be loaded right now. "],
        ["A", "Check and fix the connection", "/recovery"],
    ]
    assert _outage_banner("en", "viewer") == [
        ["DIV", "Storage unavailable", ""], ["#text", "Reports and the archive cannot be loaded right now. Please tell your LoadLens administrator."],
    ]
    assert _outage_banner("ru", "admin") == [
        ["DIV", "Хранилище недоступно", ""], ["#text", "Отчёты и архив сейчас не загружаются. "],
        ["A", "Проверить и исправить подключение", "/recovery"],
    ]


# ---- ui.js ------------------------------------------------------------------------------------------------------

UI_DRIVER = r"""(() => {
  const made = [];
  const create = document.createElement;
  document.createElement = (tag) => {
    const el = create(tag);
    if (tag === 'dialog') {
      const parts = { '.dialog-title': create('div'), '.dialog-body': create('div'), '.dialog-actions': create('div'), form: create('form') };
      el.querySelector = (sel) => parts[sel] || null;
      el.showModal = () => {};
      el.close = () => {};
      made.push(parts);
    }
    return el;
  };
  LL.ui.confirm({ message: 'x' });
  LL.ui.prompt({ label: 'name' });
  LL.ui.alert('y');
  LL.ui.dialog({ bodyHtml: '' });
  const dialogs = made.map((parts) => [parts['.dialog-title'].textContent].concat(parts['.dialog-actions'].children.map((b) => b.textContent)));
  const value = create('span');
  document.getElementById = (id) => (id === 'viewMenu' ? Object.assign(create('details'), { querySelectorAll: () => [] }) : id === 'viewMenuValue' ? value : null);
  const themes = ['system', 'light', 'dark'].map((mode) => { LL.ui.theme.set(mode); return value.textContent; });
  const toast = LL.ui.toast('hello');
  const host = create('div');
  host.querySelector = () => Object.assign(create('div'), { style: {} });
  LL.ui.combobox(host, { placeholder: 'p' });
  return {
    dialogs, themes, toastClose: toast.children[1].getAttribute('aria-label'), combobox: host.innerHTML,
    glossary: Object.assign({}, LL.GLOSSARY), terms: Object.keys(LL.GLOSSARY).length,
    term: LL.ui.term('p95', 'p95'),
  };
})()"""


def test_ui_components_speak_english():
    out = run_js(BASE_JS, UI_DRIVER, locale="en")
    assert out["dialogs"] == [["Confirmation", "Cancel", "Confirm"], ["", "Cancel", "Save"], ["Message", "Cancel", "Close"], ["", "Close"]]
    assert out["themes"] == ["System", "Light", "Dark"]
    assert out["toastClose"] == "Dismiss notification"
    assert 'aria-label="Clear"' in out["combobox"]
    assert out["terms"] == 27 and out["glossary"]["p95"].startswith("95th percentile")
    assert 'data-tip="95th percentile' in out["term"]
    assert not CYRILLIC.search(json.dumps(out, ensure_ascii=False))


def test_ui_components_keep_the_russian_output():
    out = run_js(BASE_JS, UI_DRIVER)
    assert out["dialogs"] == [["Подтверждение", "Отмена", "Подтвердить"], ["", "Отмена", "Сохранить"], ["Сообщение", "Отмена", "Закрыть"], ["", "Закрыть"]]
    assert out["themes"] == ["Как в системе", "Светлая", "Тёмная"]
    assert out["toastClose"] == "Закрыть уведомление" and 'aria-label="Очистить"' in out["combobox"]
    assert out["glossary"]["verdict_fail"] == "Целевой RPS не достигнут или нарушены критичные пороги SLA."


def test_glossary_terms_have_both_languages():
    ru = json.loads((REPO_ROOT / "locales" / "ru" / "glossary.json").read_text(encoding="utf-8"))
    en = json.loads((REPO_ROOT / "locales" / "en" / "glossary.json").read_text(encoding="utf-8"))
    keys = run_js(BASE_JS, "Object.keys(LL.GLOSSARY)")
    assert sorted(f"glossary.{key}" for key in keys) == sorted(ru) == sorted(en)


# ---- palette.js --------------------------------------------------------------------------------------------------

PALETTE_DRIVER = r"""(() => {
  const P = LL.palette._internals;
  const create = document.createElement;
  let dialog = null;
  const parts = { '.palette-input': create('input'), '.palette-list': create('div'), '[role="status"]': create('div') };
  document.createElement = (tag) => {
    const el = create(tag);
    if (tag === 'dialog') {
      el.querySelector = (sel) => parts[sel] || null;
      el.showModal = () => { el.open = true; };
      dialog = el;
    }
    return el;
  };
  LL.palette.open();
  const groups = ['loading', 'error', 'ready'].map((status) => [P.buildGroups({ runs: { status, items: [] } }, ''), P.buildGroups({ runs: { status, items: [] } }, 'zz')]);
  const actions = P.actionItems({ canCreate: true, theme: 'light', themeMode: 'dark' }).concat(P.actionItems({ theme: 'dark', themeMode: 'system' }));
  const areas = P.areaItems(['demo', { id: 'dev', title: 'Dev' }], 'demo');
  const runs = P.runItems([{ run_name: 'r1', service: 's', end_time: Date.now() - 300000, verdict: 'fail' }], Date.now());
  const visible = (items) => items.map((item) => [item.label, item.meta || '']);
  return {
    dialogLabel: dialog.getAttribute('aria-label'), dialogHtml: dialog.innerHTML, status: parts['[role="status"]'].textContent,
    list: parts['.palette-list'].innerHTML,
    groups: groups.map((pair) => pair.map((list) => list.map((g) => [g.title, g.note]))),
    actions: visible(actions), areas: visible(areas), runs: visible(runs),
    keywords: actions.map((item) => item.keywords).concat(areas.map((item) => item.keywords)),
    empty: P.buildGroups({}, 'zz'),
  };
})()"""


# An explicit theme choice, so the palette also offers "Use the system theme" (the default mode is "system").
DARK_CHOSEN = {"loadlens.theme": "dark"}


def test_palette_speaks_english_and_searches_both_languages():
    out = run_js(SHELL_JS, PALETTE_DRIVER, locale="en", local_storage=DARK_CHOSEN)
    assert out["dialogLabel"] == "Command palette"
    assert 'aria-label="Search pages, reports, areas and actions" placeholder="Page, report or action…"' in out["dialogHtml"]
    assert "> select</span>" in out["dialogHtml"] and "> in a new tab</span>" in out["dialogHtml"]
    assert out["status"] == "Found: 2" and ">Actions</div>" in out["list"]
    assert out["groups"][0] == [[["Recent reports", "Loading reports…"]], [["Reports", "Searching reports…"]]]
    assert out["groups"][1][0] == [["Recent reports", "Could not load reports"]] and out["groups"][2][1] == [["Reports", "No reports found"]]
    assert out["actions"] == [["New report", ""], ["Switch to the dark theme", ""], ["Use the system theme", ""], ["Switch to the light theme", ""]]
    assert out["areas"] == [["All areas", ""], ["demo", "current"], ["Dev", ""]]
    assert out["runs"] == [["r1", "s · 5 min ago · Failed"]]
    visible = {key: value for key, value in out.items() if key != "keywords"}
    assert not CYRILLIC.search(json.dumps(visible, ensure_ascii=False))
    # Search keywords stay bilingual, so a page or action is found by Russian and by English words.
    assert all("area" in kw and CYRILLIC.search(kw) for kw in out["keywords"][-3:])
    assert run_js(SHELL_JS, "LL.palette._internals.filterItems(LL.palette._internals.areaItems(['demo'], ''), 'area').length", locale="en") == 2


def test_palette_keeps_the_russian_output():
    out = run_js(SHELL_JS, PALETTE_DRIVER, local_storage=DARK_CHOSEN)
    assert out["dialogLabel"] == "Командная палитра" and out["status"] == "Найдено: 2"
    assert out["groups"][0] == [[["Недавние отчёты", "Загружаем отчёты…"]], [["Отчёты", "Ищем отчёты…"]]]
    assert out["actions"][0] == ["Новый отчёт", ""] and out["areas"][0] == ["Все области", ""] and out["areas"][1] == ["demo", "текущая"]
    assert "> выбор</span>" in out["dialogHtml"] and "> в новой вкладке</span>" in out["dialogHtml"]


@pytest.mark.parametrize("locale, labels", [
    ("en", ["Dashboard", "Run a test", "Archive", "Compare tests", "Forecasting", "Settings", "Profile and API tokens"]),
    ("ru", ["Дэшборд", "Запуск теста", "Архив", "Сравнение тестов", "Прогнозирование", "Настройки", "Профиль и API-токены"]),
])
def test_pages_are_found_by_their_name_in_either_language(locale, labels):
    hrefs = ["/", "/launch", "/reports", "/compare", "/forecasting", "/settings", "/account"]
    links = [{"href": href, "label": label} for href, label in zip(hrefs, labels)]
    queries = ["дэшборд", "dashboard", "запуск", "run", "архив", "archive", "сравнение", "compare", "прогнозирование",
               "forecasting", "настройки", "settings", "профиль", "profile"]
    found = run_js(SHELL_JS, f"""(() => {{
      const P = LL.palette._internals;
      const pages = P.pageItems({json.dumps(links, ensure_ascii=False)});
      return {json.dumps(queries, ensure_ascii=False)}.map((q) => (P.filterItems(pages, q)[0] || {{}}).href || null);
    }})()""", locale=locale)
    assert found == [href for href in hrefs for _ in (0, 1)]


# ---- charts.js -----------------------------------------------------------------------------------------------------

CHARTS_DRIVER = r"""(() => {
  const chart = {
    canvas: { width: 400, height: 200 }, ctx: {}, width: 400, currentDevicePixelRatio: 1,
    data: { datasets: [{ label: 'api', data: [1200.5, 1300.25], borderColor: '#111111' }, { data: [3] }] },
    isDatasetVisible: () => true, setDatasetVisibility() {}, update() {},
  };
  const host = document.createElement('div');
  LL.charts.legendChips(chart, host);
  const [list, tools] = host.children;
  const chips = list.children.filter((el) => el.className === 'legend-chip').map((chip) => [chip.title, chip.getAttribute('aria-label'), chip.children[1].textContent]);
  const controls = tools.children.map((el) => [el.textContent, el.title || '', el.placeholder || '', el.getAttribute('aria-label') || '']);
  const drawn = [];
  const create = document.createElement;
  document.createElement = (tag) => {
    const el = create(tag);
    if (tag === 'canvas') {
      el.getContext = () => ({ save() {}, restore() {}, scale() {}, fillRect() {}, drawImage() {}, measureText: (t) => ({ width: String(t).length * 6 }), fillText: (t) => drawn.push(t) });
      el.toDataURL = () => 'data:image/png;base64,';
    }
    return el;
  };
  LL.charts.downloadPng(chart, { filename: 'x' });
  return {
    listLabel: list.getAttribute('aria-label'), toolsLabel: tools.getAttribute('aria-label'), chips, controls,
    empty: list.children[list.children.length - 1].textContent, drawn,
    footer: LL.charts.topTooltip({ max: 1 }).callbacks.footer([{ chart: Object.assign({}, chart, { data: { datasets: [{ data: [1] }, { data: [2] }, { data: [3] }] } }), dataIndex: 0 }]),
    avg: [1234.567, 0.012345].map((v) => LL.charts.formatAverage(v)),
  };
})()"""


def test_chart_legend_and_png_speak_english():
    out = run_js(SHELL_JS, CHARTS_DRIVER, locale="en")
    assert out["listLabel"] == "Chart series: click to hide or show" and out["toolsLabel"] == "Legend controls"
    assert out["chips"] == [["api — average 1,250", "api, average 1,250", "api"], ["Series 2 — average 3", "Series 2, average 3", "Series 2"]]
    assert out["controls"] == [
        ["", "", "Find a series", "Find a series by name"], ["All", "Show all series", "", ""], ["None", "Hide all series", "", ""],
        ["PNG", "Download the chart with its legend as PNG", "", ""],
    ]
    assert out["empty"] == "Nothing found" and out["footer"] == "2 more"
    assert out["drawn"] == ["Legend", "api", "Average: 1,250", "", "Average: 3"] and out["avg"] == ["1,235", "0.0123"]
    assert not CYRILLIC.search(json.dumps(out, ensure_ascii=False))


def test_chart_legend_keeps_the_russian_output():
    out = run_js(SHELL_JS, CHARTS_DRIVER)
    nbsp = "\u00a0"
    assert out["chips"][0] == [f"api — среднее 1{nbsp}250", f"api, среднее 1{nbsp}250", "api"]
    assert out["chips"][1][0] == "Серия 2 — среднее 3"
    assert [c[0] for c in out["controls"][1:]] == ["Все", "Ни одной", "PNG"] and out["footer"] == "ещё 2"
    assert out["drawn"][0] == "Легенда" and f"Среднее: 1{nbsp}250" in out["drawn"] and out["avg"] == [f"1{nbsp}235", "0,0123"]
