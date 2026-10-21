# SPDX-License-Identifier: Apache-2.0
"""English UI of the settings page (plans/i18n-plan.md, section 2.5, row 7).

The page renders its own content (the <title> and <main>; the frame of base.html is checked by its own tests)
without Russian text in en, the field schemas and text builders of settings.js produce English strings, and ru
stays as it was.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from loadlens_app import core  # noqa: E402
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text  # noqa: E402
from tests.js_harness import run_js  # noqa: E402

SETTINGS = ["static/js/common.js", "static/js/ui.js", "static/js/settings.js"]
INTERNALS = "LL.settings._internals"


@pytest.fixture
def client(monkeypatch, tmp_path):
    import update_page
    from app import create_app

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _page(client, lang):
    use_locale(client, lang)
    resp = client.get("/settings")
    assert resp.status_code == 200
    return resp.get_data(as_text=True)


def _own_text(html: str) -> str:
    """Visible text the settings template renders: its <title> and its part of <main>."""
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
    start = html.index('<div class="settings-head">')
    end = html.index("</main>", start)
    return title + "\n" + visible_text(html[start:end])


def _js(value):
    return json.dumps(value, ensure_ascii=False)


def _cyrillic(values):
    return [value for value in values if CYRILLIC.search(json.dumps(value, ensure_ascii=False))]


# ---- the page -----------------------------------------------------------------------------------------


def test_settings_page_in_en_has_no_russian_text(client):
    html = _page(client, "en")
    assert re.search(r'<html lang="en"', html)
    assert "<title>Settings</title>" in html
    text = _own_text(html)
    assert_no_cyrillic(text)
    for phrase in ("First-run setup wizard", "Global settings", "Test connection", "Unsaved changes:", "Reset to violet",
                   "Area (default)", "Advanced parameters (JSON)", "API tokens of all users"):
        assert phrase in text, phrase


def test_settings_page_in_en_keeps_the_markup_inside_sentences(client):
    html = _page(client, "en")
    assert "Passwords and tokens are shown as <code>***</code> and are stored on the server only." in html
    assert "Placeholders such as <code>{answer_jvm}</code> are filled in automatically" in html
    assert "<strong>Viewer</strong> sees reports" in html and "<strong>Administrator</strong> can also" in html
    assert "<code>page_sample_id</code>, <code>page_parent_id</code>, a <code>metrics</code> list" in html
    assert 'Unsaved changes: <strong id="dirtyList"></strong>' in html
    assert "the verdict “Risks found”" in html


def test_settings_page_ru_is_unchanged(client):
    html = _page(client, None)
    assert re.search(r'<html lang="ru"', html) and "<title>Настройки</title>" in html
    for fragment in (
        '<a class="btn btn-sm not-in-wizard" href="/settings?wizard=1">Мастер первичной настройки</a>',
        '<option value="">Область (по умолчанию)</option>',
        "Пароли и токены показываются как <code>***</code> и хранятся только на сервере.",
        "Плейсхолдеры вида <code>{answer_jvm}</code> подставляются автоматически",
        "<strong>Наблюдатель</strong> смотрит отчёты, архив, сравнение и прогнозы.",
        "(ключ <code>services</code>).",
        "нарушения остальных порогов дают вердикт «Есть риски».",
        '<span>Несохранённые изменения: <strong id="dirtyList"></strong></span>',
        '<td colspan="7">Загрузка…</td>',
    ):
        assert fragment in html, fragment


def test_settings_page_sends_its_dictionary(client):
    html = _page(client, "en")
    script = re.search(r"window\.LoadLens\.i18n=(\{.*?\});</script>", html, re.S).group(1)
    data = json.loads(script)
    assert data["locale"] == "en" and data["intl"] == "en-GB"
    assert data["messages"]["settings.card.storage.title"] == "Storage (TimescaleDB)"
    assert "users.title" in data["messages"] or any(key.startswith("users.") for key in data["messages"])


# ---- settings.js -----------------------------------------------------------------------------------------


def test_field_schemas_and_tables_are_english_in_en():
    en = run_js(SETTINGS, f"{INTERNALS}.schemaTexts()", locale="en")
    ru = run_js(SETTINGS, f"{INTERNALS}.schemaTexts()", locale="ru")
    assert len(en) == len(ru) > 400
    assert not _cyrillic(en), _cyrillic(en)[:20]
    assert "Storage (TimescaleDB)" in en and "OpenAI-compatible API" in en and "Perplexity" in en
    assert "Хранилище (TimescaleDB)" in ru and "OpenAI-совместимый API" in ru and "Подставляется вместо {exe}; генератор может задать свой путь." in ru


SUMMARIES = f"""
const I = {INTERNALS};
const badge = (info, type) => {{ const el = I.toolBadge(info, 'k6', type); return el ? el.textContent : null; }};
({{
  gen: [
    I.generatorSummary('m', {{ type: 'manual' }}),
    I.generatorSummary('d', {{ type: 'docker', tools: {{ k6: {{}} }} }}),
    I.generatorSummary('k', {{ type: 'kubernetes', kubernetes: {{ in_cluster: true }} }}),
    I.generatorSummary('g', {{ type: 'gitlab' }}),
    I.generatorSummary('j', {{ type: 'jenkins', jenkins: {{ url: 'https://j', ref: 'dev' }} }}),
    I.generatorSummary('s', {{ type: 'ssh', ssh: {{ host: 'h', port: 2222, username: 'u', host_key: 'AAA' }} }}),
    I.generatorSummary('s', {{ type: 'ssh' }})
  ],
  src: [
    I.sourceSummary('g', {{ type: 'grafana_proxy', grafana: {{ base_url: 'http://g', auth: {{ method: 'basic' }} }} }}),
    I.sourceSummary('g', {{ type: 'grafana_proxy', grafana: {{ auth: {{ method: 'bearer' }} }} }}),
    I.sourceSummary('i', {{ type: 'influxdb', influxdb: {{ url: 'http://i', org: 'acme' }} }})
  ],
  status: [I.generatorStatus({{ type: 'ssh', verified: {{ at: '2026-01-02T03:04:00Z' }} }}), I.generatorStatus({{ type: 'ssh' }})],
  tool: I.toolSummary('k6', {{}}),
  badges: [badge({{ tools_found: ['k6'] }}, 'docker'), badge({{ tools_found: ['k6'] }}, 'ssh'), badge({{ tools_missing: ['k6'] }}, 'docker'), badge({{ tools_missing: ['k6'] }}, 'ssh')],
  host: I.hostKeyMessage({{ ssh: {{ host: 'h', port: 2200 }} }}, {{ host_key_fingerprint: 'SHA256:x' }}),
  options: [I.queryOptions(['a'], 'gone', 'a'), I.queryOptions(['a'], '', '')],
  dead: [I.deadRefText(['a'], 'gone', 'a'), I.deadRefText(['a'], 'gone', '')],
  context: I.CONTEXT_GROUPS.map((g) => I.contextGroupSummary(g, {{ architecture: {{ components: [{{}}] }}, operational_context: {{ known_risks: ['r'] }} }})),
  usage: I.slaUsageText(['max_performance_query', 'memory_query', 'service_cpu_query']),
  diff: [LL.tn('settings.prompts.diff_changed', 1), LL.tn('settings.prompts.diff_changed', 5)]
}})
"""


def test_text_builders_are_english_in_en():
    en = run_js(SETTINGS, SUMMARIES, locale="en")
    assert not _cyrillic(list(en.values())), _cyrillic(list(en.values()))
    assert en["gen"][0] == "id m · LoadLens only marks the start and end of a run"
    assert en["gen"][5] == "id s · u@h:2222 · all tools · host key pinned"
    assert en["src"][1] == "id g · no address · API token sign-in"
    assert en["status"] == [{"text": "checked 02/01/2026, 03:04", "tone": "ok"}, {"text": "not checked", "tone": "warn"}]
    assert en["host"].startswith("The server h:2200 presented the host key SHA256:x. ")
    assert en["options"][0][0] == {"v": "", "l": "auto by role: a"} and en["options"][0][-1] == {"v": "gone", "l": "gone: no such query"}
    assert en["context"][0] == "not filled in" and en["context"][1].startswith("components: 1")
    assert en["context"][3] == "known risks: 1"
    assert en["diff"] == ["1 line differs", "5 lines differ"]


def test_text_builders_keep_the_russian_wording():
    ru = run_js(SETTINGS, SUMMARIES, locale="ru")
    assert ru["gen"][0] == "id m · LoadLens только отмечает начало и конец прогона"
    assert ru["gen"][5] == "id s · u@h:2222 · все инструменты · ключ хоста закреплён"
    assert ru["gen"][6] == "id s · адрес не указан · все инструменты · ключ хоста не закреплён"
    assert ru["src"][0] == "id g · http://g · логин не указан"
    assert ru["status"][0] == {"text": "проверен 02.01.2026, 03:04", "tone": "ok"}
    assert ru["badges"] == ["образ есть", "найден", "нет образа", "не найден"]
    assert ru["host"] == (
        "Сервер h:2200 предъявил ключ хоста SHA256:x. "
        "Сверьте отпечаток с ключом на самом сервере, например командой ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub. "
        "После подтверждения LoadLens будет подключаться только к серверу с этим ключом."
    )
    assert ru["dead"][1] == "Запроса «gone» больше нет: критерий не получит данных. Выберите другой запрос."
    assert ru["context"][3] == "известные риски: 1"
    assert ru["usage"] == "макс. RPS, память, CPU сервисов (прогноз)"
    assert ru["diff"] == ["1 строк отличается", "5 строк отличается"]
