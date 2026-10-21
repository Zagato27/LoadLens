# SPDX-License-Identifier: Apache-2.0
"""Stage 5 of plans/i18n-plan.md: the report language setting, the interface language switch and its cookie.

- ``appearance.language`` (language of new reports and prompts) is validated, merged with the accent and known;
- the interface language: cookie ``loadlens.lang`` -> ``appearance.language`` -> ``ru``; ``language_switch=false``
  ignores the cookie; ``<html lang>`` follows;
- the switcher: the sidebar "View" menu, the RU · EN links of the standalone pages, static/js/locale.js (cookie and
  localStorage copy), the toast in other tabs and the command palette action;
- a report job keeps the language it was queued with; the demo run follows the setting, not the UI cookie.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_smoke_api import client, patch_core  # noqa: E402,F401  (fixtures; initialise the app before update_page)
from loadlens_app import core  # noqa: E402
from loadlens_app import i18n as web_i18n  # noqa: E402
from tests.i18n_helpers import use_locale  # noqa: E402
from tests.js_harness import run_js  # noqa: E402

TEMPLATES = ROOT_DIR / "templates"
LOCALE_JS = "static/js/locale.js"
SHELL_JS = [LOCALE_JS, "static/js/common.js", "static/js/ui.js"]
PALETTE_JS = ["static/js/common.js", "static/js/ui.js", "static/js/palette.js"]


@pytest.fixture
def appearance(monkeypatch):
    """Sets keys of ``appearance`` for the test (restored afterwards by monkeypatch)."""

    def choose(**values):
        monkeypatch.setitem(core.CONFIG, "appearance", {**(core.CONFIG.get("appearance") or {}), **values})

    return choose


def post_appearance(client, data):
    return client.post("/config", json={"section": "appearance", "data": data})


def html_lang(html: str) -> str:
    return re.search(r'<html lang="([^"]*)"', html).group(1)


def runtime_appearance() -> dict:
    path = Path(core.CONFIG_RUNTIME_PATH)
    return json.loads(path.read_text(encoding="utf-8")).get("appearance", {}) if path.exists() else {}


@pytest.fixture(autouse=True)
def _runtime_in_tmp(monkeypatch, tmp_path):
    import update_page

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")


# ---- the setting ------------------------------------------------------------------------------------------------


def test_appearance_language_is_validated(client):
    resp = post_appearance(client, {"language": "de"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Язык «de» не поддерживается: допустимо ru, en"
    use_locale(client, "en")
    assert post_appearance(client, {"language": "de"}).get_json()["error"] == 'Language "de" is not supported: use ru, en'
    assert post_appearance(client, {"language": 7}).status_code == 400
    assert post_appearance(client, {"language_switch": "no"}).status_code == 400
    assert core.CONFIG["appearance"]["language"] == "ru"  # nothing stored after a refusal
    # A tag of a supported language is stored as its code.
    assert post_appearance(client, {"language": "EN-gb"}).status_code == 200
    assert core.CONFIG["appearance"]["language"] == "en"


def test_appearance_accent_and_language_do_not_overwrite_each_other(client):
    assert post_appearance(client, {"language": "en"}).status_code == 200
    assert post_appearance(client, {"accent": "#00A884"}).status_code == 200
    assert core.CONFIG["appearance"]["language"] == "en" and core.CONFIG["appearance"]["accent"] == "#00A884"
    assert post_appearance(client, {"language": "ru"}).status_code == 200
    assert core.CONFIG["appearance"]["accent"] == "#00A884"
    stored = runtime_appearance()
    assert stored["accent"] == "#00A884" and stored["language"] == "ru"
    body = client.get("/config").get_json()["appearance"]
    assert body["accent"] == "#00A884" and body["language"] == "ru"
    # Both at once, as the settings page sends them.
    assert post_appearance(client, {"accent": "#1d4ed8", "language": "en"}).status_code == 200
    assert core.CONFIG["appearance"]["accent"] == "#1d4ed8" and core.CONFIG["appearance"]["language"] == "en"


def test_appearance_language_is_a_known_key(client):
    resp = post_appearance(client, {"accent": "#6c5ce7", "language": "en", "language_switch": True})
    assert resp.status_code == 200 and resp.get_json()["warnings"] == []
    typo = post_appearance(client, {"langauge": "en"}).get_json()["warnings"]
    assert typo and "langauge" in typo[0]


def test_template_appearance_carries_the_language(appearance):
    from loadlens_app.appearance import template_appearance

    assert template_appearance()["language"] == "ru" and template_appearance()["language_switch"] is True
    appearance(language="EN", language_switch=False)
    values = template_appearance()
    assert values["language"] == "en" and values["language_switch"] is False
    appearance(language="de")
    assert template_appearance()["language"] == "ru"


# ---- the interface language -------------------------------------------------------------------------------------


def test_locale_resolves_cookie_then_global_then_ru(client, appearance):
    assert html_lang(client.get("/").get_data(as_text=True)) == "ru"
    appearance(language="en")
    assert html_lang(client.get("/").get_data(as_text=True)) == "en"  # a new browser follows the setting
    use_locale(client, "ru")
    assert html_lang(client.get("/").get_data(as_text=True)) == "ru"  # the cookie wins
    use_locale(client, 'de"><script>alert(1)</script>')
    html = client.get("/").get_data(as_text=True)
    assert html_lang(html) == "en" and "alert(1)" not in html  # an unknown value is ignored, never echoed
    appearance(language="ru")
    use_locale(client, "EN")
    assert html_lang(client.get("/").get_data(as_text=True)) == "en"


def test_language_switch_false_ignores_the_cookie(client, appearance):
    use_locale(client, "en")
    appearance(language_switch=False)
    html = client.get("/").get_data(as_text=True)
    assert html_lang(html) == "ru" and 'data-view-pref="lang"' not in html
    appearance(language="en", language_switch=False)
    use_locale(client, "ru")
    assert html_lang(client.get("/").get_data(as_text=True)) == "en"
    appearance(language_switch=True)
    assert html_lang(client.get("/").get_data(as_text=True)) == "ru"


def test_current_locale_reads_the_switch_in_a_request(appearance):
    from app import create_app

    flask_app = create_app()
    with flask_app.test_request_context("/", headers={"Cookie": "loadlens.lang=en"}):
        assert web_i18n.current_locale() == "en" and web_i18n.language_switch_enabled()
    appearance(language_switch=False)
    with flask_app.test_request_context("/", headers={"Cookie": "loadlens.lang=en"}):
        assert web_i18n.current_locale() == "ru" and not web_i18n.language_switch_enabled()


def test_locale_dependent_responses_vary_on_cookie(client):
    assert "Cookie" in client.get("/").headers.get("Vary", "")
    assert "Cookie" not in client.get("/static/js/locale.js").headers.get("Vary", "")


def test_view_menu_switcher_in_both_languages(client):
    ru = client.get("/").get_data(as_text=True)
    assert '<span class="nav-label">Вид</span>' in ru and '<div class="menu-group-label" aria-hidden="true">Язык</div>' in ru
    assert re.search(r'aria-checked="true" data-view-pref="lang" data-value="ru" lang="ru">.*?<span>Русский</span>', ru)
    assert re.search(r'aria-checked="false" data-view-pref="lang" data-value="en" lang="en">.*?<span>English</span>', ru)
    assert ru.index("js/i18n.js") < ru.index("js/locale.js") < ru.index("js/ui.js")
    use_locale(client, "en")
    en = client.get("/").get_data(as_text=True)
    assert '<span class="nav-label">View</span>' in en and '<div class="menu-group-label" aria-hidden="true">Language</div>' in en
    assert re.search(r'aria-checked="true" data-view-pref="lang" data-value="en"', en)


def _render_standalone(page: str, lang: str | None, **context):
    from flask import render_template

    from app import create_app

    flask_app = create_app()
    headers = {"Cookie": f"loadlens.lang={lang}"} if lang else {}
    storage = types.SimpleNamespace(title="x", hint="y", address="", message="", available=False)
    with flask_app.test_request_context("/", headers=headers):
        return render_template(page, storage=storage, retry_seconds=15, session_minutes=30, **context)


@pytest.mark.parametrize("page", ["login.html", "recovery.html", "storage_unavailable.html"])
def test_standalone_pages_offer_ru_en_links(page, appearance):
    links = re.compile(r'<a href="#" data-set-locale="(\w+)" lang="\w+" hreflang="\w+" title="([^"]+)"( aria-current="true")?>(\w+)</a>')
    ru = _render_standalone(page, None)
    assert html_lang(ru) == "ru"
    assert links.findall(ru) == [("ru", "Русский", ' aria-current="true"', "RU"), ("en", "English", "", "EN")]
    assert 'aria-label="Язык интерфейса"' in ru and "js/locale.js" in ru
    en = _render_standalone(page, "en")
    assert html_lang(en) == "en" and 'aria-label="Interface language"' in en
    assert links.findall(en) == [("ru", "Русский", "", "RU"), ("en", "English", ' aria-current="true"', "EN")]
    appearance(language_switch=False)
    off = _render_standalone(page, "en")
    assert html_lang(off) == "ru" and "data-set-locale" not in off and "js/locale.js" not in off


# ---- static/js/locale.js and the shell ----------------------------------------------------------------------------


def test_locale_set_writes_cookie_and_mirror():
    out = run_js(SHELL_JS, """
        const before = __harness.reloads();
        const ok = LL.ui.locale.set('en-GB');
        ({ ok, writes: __harness.cookieWrites.slice(), cookie: document.cookie, copy: localStorage.getItem('loadlens.lang'),
           pref: LL.prefs.get('lang'), reloads: __harness.reloads() - before, switching: LL.ui.locale.switching(),
           bad: LL.ui.locale.set('de'), writesAfterBad: __harness.cookieWrites.length })
    """)
    assert out["ok"] is True and out["switching"] == "en"
    assert out["writes"] == ["loadlens.lang=en; Path=/; Max-Age=31536000; SameSite=Lax"]
    assert out["cookie"] == "loadlens.lang=en" and out["copy"] == "en" and out["pref"] == "en"
    assert out["reloads"] == 1
    assert out["bad"] is False and out["writesAfterBad"] == 1


def test_locale_cookie_is_secure_on_https_and_works_without_ui_js():
    out = run_js([LOCALE_JS], """
        location.protocol = 'https:';
        LL.ui.locale.set('ru', { reload: false });
        ({ writes: __harness.cookieWrites, copy: localStorage.getItem('loadlens.lang'), reloads: __harness.reloads(),
           current: LL.ui.locale.current(), normalize: ['EN', 'en_US', 'ru-RU', 'english', '', null].map(LL.ui.locale.normalize) })
    """, locale="en")
    assert out["writes"] == ["loadlens.lang=ru; Path=/; Max-Age=31536000; SameSite=Lax; Secure"]
    assert out["copy"] == "ru" and out["reloads"] == 0 and out["current"] == "en"
    assert out["normalize"] == ["en", "en", "ru", None, None, None]


def test_standalone_links_switch_the_language_on_click():
    out = run_js([LOCALE_JS], """
        const link = (code) => ({ getAttribute: (name) => (name === 'data-set-locale' ? code : null) });
        const click = (code) => {
          const event = { type: 'click', prevented: false, preventDefault() { this.prevented = true; },
                          target: { closest: (sel) => (sel === '[data-set-locale]' && code ? link(code) : null) } };
          document.dispatchEvent(event);
          return event.prevented;
        };
        const results = [click(null), click('ru'), click('en')];
        ({ results, writes: __harness.cookieWrites, reloads: __harness.reloads() })
    """)
    # Outside a link nothing happens; the current language (ru) is not written again; EN switches.
    assert out["results"] == [False, True, True]
    assert out["writes"] == ["loadlens.lang=en; Path=/; Max-Age=31536000; SameSite=Lax"] and out["reloads"] == 1


@pytest.mark.parametrize("locale, text", [
    ("ru", "Язык интерфейса изменён в другой вкладке — обновите страницу"),
    ("en", "The interface language was changed in another tab: reload the page"),
])
def test_locale_change_in_another_tab_asks_to_reload(locale, text):
    out = run_js(SHELL_JS, """
        const toasts = [];
        LL.ui.toast = (message, options) => { toasts.push([message, options && options.timeout]); };
        const other = LL.locale === 'ru' ? 'en' : 'ru';
        __harness.fireWindow('storage', { key: 'loadlens.lang', newValue: LL.locale });  // same language: nothing
        __harness.fireWindow('storage', { key: 'loadlens.lang', newValue: 'de' });       // unsupported: nothing
        __harness.fireWindow('storage', { key: 'loadlens.lang', newValue: other });
        const afterOther = toasts.length;
        LL.ui.locale.set(other, { reload: false });                                    // this tab: no toast
        ({ toasts, afterOther, reloads: __harness.reloads() })
    """, locale=locale)
    assert out["toasts"] == [[text, 0]] and out["afterOther"] == 1 and out["reloads"] == 0


def test_view_menu_value_shows_theme_and_language():
    out = run_js(SHELL_JS, """
        const value = document.createElement('span');
        const items = ['ru', 'en'].map((code) => { const el = document.createElement('button'); el.dataset.value = code; return el; });
        const menu = document.createElement('details');
        menu.querySelectorAll = (sel) => (sel.indexOf('"lang"') >= 0 ? items : []);
        document.getElementById = (id) => (id === 'viewMenu' ? menu : id === 'viewMenuValue' ? value : null);
        LL.ui.theme.set('dark');
        ({ value: value.textContent, checked: items.map((el) => el.getAttribute('aria-checked')) })
    """, locale="en")
    assert out == {"value": "Dark · EN", "checked": ["false", "true"]}


@pytest.mark.parametrize("locale, label, target", [("ru", "Switch to English", "en"), ("en", "Переключить на русский", "ru")])
def test_palette_offers_the_language_action(locale, label, target):
    out = run_js(PALETTE_JS, f"""
        const P = LL.palette._internals;
        ({{ on: P.actionItems({{ canCreate: false, theme: 'dark', themeMode: 'system', languageSwitch: true, locale: '{locale}' }}),
            off: P.actionItems({{ canCreate: false, theme: 'dark', themeMode: 'system', languageSwitch: false, locale: '{locale}' }}) }})
    """, locale=locale)
    action = out["on"][-1]
    assert action["key"] == "action:lang" and action["action"] == "lang"
    assert action["label"] == label and action["lang"] == target
    assert "english" in action["keywords"] and "русский" in action["keywords"]
    assert all(item["key"] != "action:lang" for item in out["off"])


def test_report_language_badge_names_the_language_in_the_ui_language():
    row = {"domain": "final", "parsed": {"report_language": "ru"}}
    en = run_js(["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/reports.js"],
                f"LL.report._internals.reportLanguageBadge({json.dumps(row)})", locale="en")
    assert ">Report language: Russian</span>" in en and "Русский" not in en


# ---- settings page --------------------------------------------------------------------------------------------


def test_settings_page_has_the_report_language_field(client, appearance):
    html = client.get("/settings").get_data(as_text=True)
    assert '<label for="reportLanguage">Язык отчётов и промптов</label>' in html
    assert '<option value="ru" lang="ru" selected>Русский</option><option value="en" lang="en">English</option>' in html
    assert "Готовые отчёты не меняются." in html and 'id="promptLanguageNote"' in html
    appearance(language="en")
    use_locale(client, "ru")
    html = client.get("/settings").get_data(as_text=True)
    assert '<option value="ru" lang="ru">Русский</option><option value="en" lang="en" selected>English</option>' in html
    use_locale(client, "en")
    html = client.get("/settings").get_data(as_text=True)
    assert '<label for="reportLanguage">Report and prompt language</label>' in html


SETTINGS_JS = ["static/js/common.js", "static/js/ui.js", "static/js/settings.js"]


@pytest.mark.parametrize("locale", ["ru", "en"])
def test_prompt_language_note(locale):
    out = run_js(SETTINGS_JS, """
        const N = LL.settings._internals.promptLanguageNote;
        const meta = { language: 'en', sources: { jvm: 'area', kafka: 'default', database: 'service' },
                       overrideLanguage: { jvm: 'ru', database: 'ru' } };
        ({ jvm: N(meta, 'jvm', 'area'), kafka: N(meta, 'kafka', 'area'), db: N(meta, 'database', 'area'),
           same: N({ language: 'ru', sources: { jvm: 'area' }, overrideLanguage: { jvm: 'ru' } }, 'jvm', 'area'),
           enOverride: N({ language: 'ru', sources: { jvm: 'service' }, overrideLanguage: { jvm: 'en' } }, 'jvm', 'service'),
           steps: LL.settings._internals.WIZARD_STEPS.map((s) => s.key) })
    """, locale=locale)
    assert out["kafka"]["show"] is False and out["same"]["show"] is False
    assert out["jvm"]["show"] is True and out["jvm"]["canReset"] is True
    assert out["db"]["show"] is True and out["db"]["canReset"] is False  # a service override seen from the area
    assert out["steps"][0] == "appearance" and out["steps"][1] == "storage"
    if locale == "ru":
        assert out["jvm"]["text"] == "Переопределение на русском, отчёты — на английском; ответ всё равно будет на английском."
        assert out["jvm"]["button"] == "Показать стандартный (en)"
        assert out["enOverride"]["text"] == "Переопределение на английском, отчёты — на русском; ответ всё равно будет на русском."
    else:
        assert out["jvm"]["text"] == "The override is in Russian and reports are in English; the answer will still be in English."
        assert out["jvm"]["button"] == "Show the default (en)" and out["enOverride"]["canReset"] is True


# ---- reports and the demo follow the setting, not the UI cookie ------------------------------------------------


def test_report_job_keeps_the_language_it_was_queued_with(monkeypatch, appearance):
    from loadlens_app import job_runner, jobs, tasks
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_report_step", lambda area, service_id: "30s")
    monkeypatch.setattr(jobs.job_store, "_persist", lambda record: None)

    def no_db():
        raise RuntimeError("the job stays in memory in this test")

    monkeypatch.setattr(jobs, "_ts_conn", no_db)
    submitted = []
    monkeypatch.setattr(job_runner, "submit", lambda task, job, **kwargs: submitted.append((task, job, kwargs)) or "thread")
    appearance(language="en")
    dashboard.start_report_job(0, 600_000, "checkout", "shop", use_llm=False, save_to_db=False, web_only=True,
                               run_name="nightly", test_type="step", owner="eng")
    appearance(language="ru")  # changed before a worker picks the job up
    task, job, kwargs = submitted[0]
    assert task is tasks.report_task and kwargs["report_language"] == "en"

    calls = []
    fake_dashboard = types.SimpleNamespace(
        update_report=lambda *args, **kw: calls.append(kw) or {"run_id": "r"},
        _job_failure_message=lambda *args, **kw: "",
    )
    monkeypatch.setattr(tasks, "_dashboard", lambda: fake_dashboard)
    monkeypatch.setattr(tasks, "_run_job", lambda task_, job_id, work, failure, **options: work(lambda *a, **k: None))
    task.run(getattr(job, "job_id", "job-1"), **kwargs)
    assert [call["report_language"] for call in calls] == ["en"]


def test_demo_seed_follows_report_language(client, monkeypatch, appearance):
    from loadlens_app import demo_data

    saved = {}
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: False)
    monkeypatch.setattr(demo_data, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(demo_data, "save_domain_labeled", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo_data, "save_llm_results", lambda results, run_meta, storage: saved.update(results=results, run_meta=run_meta))
    appearance(language="en")
    use_locale(client, "ru")  # the UI language does not matter
    resp = client.post("/demo/seed")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert saved["run_meta"]["report_language"] == "en" and saved["results"]["report_language"] == "en"
    assert saved["results"]["final_parsed"]["report_language"] == "en"


def test_settings_example_documents_the_keys():
    source = (ROOT_DIR / "settings.example.py").read_text(encoding="utf-8")
    block = re.search(r'"appearance": \{(.*?)\n    \}', source, re.S).group(1)
    assert '"language": "ru"' in block and '"language_switch": True' in block


# ---- leftovers translated in stage 5: POST /prompts, context import, metric discovery, tool defaults ------------


def test_prompt_save_errors_follow_the_request_language(client):
    bad = [({"domain": "jvm", "text": "x"}, "area обязательна", "area is required"),
           ({"area": "demo", "domain": "nope", "text": "x"}, "неверный domain", "invalid domain"),
           ({"area": "demo", "domain": "judge", "text": "x"}, "Редактирование домена запрещено", "This domain cannot be edited"),
           ({"area": "demo", "domain": "jvm", "text": 5}, "text должен быть строкой", "text must be a string")]
    for body, ru, _ in bad:
        resp = client.post("/prompts", json=body)
        assert resp.status_code == 400 and resp.get_json()["error"] == ru
    use_locale(client, "en")
    for body, _, en in bad:
        assert client.post("/prompts", json=body).get_json()["error"] == en


def test_context_import_messages_in_english():
    from AI.i18n import with_locale
    from loadlens_app import context_import as ci

    with with_locale("en"):
        with pytest.raises(ci.ContextImportError) as unsupported:
            ci.extract_text("old.doc", b"x")
        assert str(unsupported.value) == 'The ".doc" format is not supported. Allowed: .docx, .md, .pdf, .txt'
        assert unsupported.value.status_code == 415 and unsupported.value.code == "unsupported"
        with pytest.raises(ci.ContextImportError, match="no extension"):
            ci.extract_text("README", b"x")
        with pytest.raises(ci.ContextImportError, match="^The document has almost no text"):
            ci.text_from_paste("short")
        with pytest.raises(ci.ContextImportError, match="^No LLM provider is configured"):
            ci.llm_for_scope({})
        long = ci.text_from_paste("word " * 30_000)
        assert long.warnings == ["The document is longer than 120,000 characters; only the beginning was read"]
        _, changes = ci.merge_context({}, {"system": {"name": "Shop"}, "operational_context": {"known_risks": ["slow db"]}}, "fill")
        assert [change["title"] for change in changes] == ["System name", "Known risks"]
    # ru stays as it was.
    assert ci.text_from_paste("word " * 30_000).warnings == ["Документ длиннее 120 000 символов, разобрано только начало"]
    _, changes = ci.merge_context({}, {"system": {"name": "Shop"}}, "fill")
    assert changes[0]["title"] == "Название системы"


def test_metric_discovery_errors_keep_the_callers_language_in_the_pool(monkeypatch):
    from AI.i18n import with_locale
    from loadlens_app import metric_discovery as md

    resolved = types.SimpleNamespace(source_id="prom", config={}, source_type="prometheus", prometheus_url="http://prom:9090")
    monkeypatch.setattr(md, "resolve_domain_source", lambda cfg, domain: resolved)
    monkeypatch.setattr(md, "_prometheus_names", lambda url: md._label_values({"status": "error"}))
    assert md.discover({}, ["jvm"])["jvm"].error == "источник не вернул список метрик"
    with with_locale("en"):
        assert md.discover({}, ["jvm"])["jvm"].error == "the source did not return a metric list"


def test_settings_show_built_in_tool_texts_in_the_reader_language(client, monkeypatch):
    from loadlens_app.load_tests.tools import DEFAULT_TOOLS

    monkeypatch.setitem(core.CONFIG, "load_tools", dict(core.CONFIG.get("load_tools") or {}))  # restored after the POST

    ru = client.get("/config").get_json()["load_tools"]
    assert ru["gatling"]["title"] == "Gatling (бандл)" == DEFAULT_TOOLS["gatling"]["title"]
    use_locale(client, "en")
    en = client.get("/config").get_json()["load_tools"]
    assert en["gatling"]["title"] == "Gatling (bundle)" and en["k6"]["help"].startswith("Set the metric output")
    # Sending the shown (English) templates back stores no override for the built-in texts.
    resp = client.post("/config", json={"section": "load_tools", "data": en})
    assert resp.status_code == 200
    assert not any(field in (core.CONFIG.get("load_tools") or {}).get(tool, {})
                   for tool in ("gatling", "k6") for field in ("title", "help"))


def test_new_report_page_names_the_report_language_when_it_differs(client, appearance):
    assert 'id="reportLanguageHint"' not in client.get("/new").get_data(as_text=True)
    use_locale(client, "en")
    html = client.get("/new").get_data(as_text=True)
    assert "The report will be in the installation&#39;s report language, Russian (Settings → Appearance)." in html
    appearance(language="en")
    use_locale(client, "ru")
    html = client.get("/new").get_data(as_text=True)
    assert "Отчёт будет на языке отчётов установки — английский («Настройки → Оформление»)." in html
    use_locale(client, None)
    assert 'id="reportLanguageHint"' not in client.get("/new").get_data(as_text=True)
