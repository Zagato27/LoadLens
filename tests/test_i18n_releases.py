# SPDX-License-Identifier: Apache-2.0
"""Release pages in both languages (templates/releases.html, release.html) and the pure helpers of
static/js/releases.js under Node: status labels, delta texts and tones, the trend model and the address."""

from __future__ import annotations

import json
import re

import pytest

from loadlens_app import core  # noqa: F401  (before anything that imports update_page)
from tests.i18n_helpers import assert_no_cyrillic, use_locale, visible_text
from tests.js_harness import run_js

NBSP = "\u00a0"  # ru-RU group separator of Intl.NumberFormat
FILES = ["static/js/common.js", "static/js/ui.js", "static/js/charts.js", "static/js/releases.js"]


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


def _page(client, url, lang):
    use_locale(client, lang)
    response = client.get(url)
    assert response.status_code == 200, url
    html = response.get_data(as_text=True)
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1).strip()
    main = html[html.index('<main id="main"'):html.index("</main>")]
    return html, title, main[main.index(">") + 1:]


# ---- templates -----------------------------------------------------------------------------------------------


def test_release_list_in_en_has_no_russian_text(client):
    html, title, page = _page(client, "/releases", "en")
    assert re.search(r'<html lang="en"', html) and title == "Releases"
    assert_no_cyrillic(title + "\n" + visible_text(page))
    assert "<h2>Releases</h2>" in page
    assert re.findall(r"<th>([^<]*)</th>", page) == ["Release", "Status", "Services", "Verdicts", "Latest report", "Ticket"]
    assert 'id="trendService"' in page and 'id="trendCharts"' in page
    # The page declares the releases and charts namespaces and loads Chart.js in the head.
    assert '"releases.status.incomplete":"Incomplete"' in html and '"charts.tooltip.more"' in html
    assert "vendor/chart.umd.js" in html[:html.index("</head>")]
    assert '<a class="nav-btn active" href="/releases" aria-current="page">' in html


def test_release_page_in_en_has_no_russian_text(client):
    html, title, page = _page(client, "/releases/DEMO%2025.11", "en")
    assert title == "Release DEMO 25.11"
    assert_no_cyrillic(title + "\n" + visible_text(page))
    assert '<h2 id="releaseTitle">Release DEMO 25.11</h2>' in page
    headers = re.findall(r"<th(?: class=\"num\")?>([^<]*)</th>", page)
    assert headers == ["Service", "Type", "Accepted run", "Verdict", "Δ p95", "Δ errors", "Δ load", "Previous release"]
    for text in (">Issues in Jira</a>", ">Edit…</button>", ">Publish to Confluence</button>", ">Open in Confluence</a>",
                 "<h3>Not tested</h3>", "<h3>Trend by release</h3>", '<a href="/releases">Releases</a>'):
        assert text in page, text


def test_release_pages_stay_russian_without_a_cookie(client):
    html, title, page = _page(client, "/releases", None)
    assert title == "Релизы" and "<th>Последний отчёт</th>" in page
    html, title, page = _page(client, "/releases/25.11", None)
    assert title == "Релиз 25.11" and ">Опубликовать в Confluence</button>" in page and "<th class=\"num\">Δ нагрузки</th>" in page


def test_a_control_character_is_not_a_release_page(client):
    assert client.get("/releases/a%0Ab").status_code == 404


def test_a_release_named_with_confluence_suffix_renders_its_html_page(client):
    html, title, page = _page(client, "/releases/2026/confluence", "en")
    assert title == "Release 2026/confluence"
    assert '<h2 id="releaseTitle">Release 2026/confluence</h2>' in page
    assert 'id="releaseTable"' in html


# ---- releases.js ---------------------------------------------------------------------------------------------


def test_status_labels_and_delta_texts_in_both_languages():
    expression = """(() => {
      const R = LL.releases._internals;
      return {
        status: ['incomplete', 'risk', 'pass', 'no_data'].map(R.statusLabel),
        classes: ['incomplete', 'fail'].map(R.statusClass),
        deltas: [{ change: 12.5, unit: 'pct' }, { change: 0.3, unit: 'pp' }, { change: -8, unit: 'pct' }, { change: 0, unit: 'pct' },
                 { change: -0.04, unit: 'pp' }, { change: null }, null].map(R.deltaText),
        tones: [{ tone: 'warn' }, { tone: 'fail' }, { tone: 'ok' }, { tone: 'neutral' }, {}, null].map(R.deltaTone),
        levels: [{ kind: 'target', level: 600 }, { kind: 'ceiling', level: 1540.4 }, { kind: 'hold', level: 200 }, { kind: null, level: null }].map(R.levelText),
        title: R.deltaTitle({ metric: 'p95', baseline: 400, current: 450.5 }),
        reasons: ['no_previous', 'no_step_at_level', 'zzz'].map(R.reasonText)
      };
    })()"""
    ru = run_js(FILES, expression)
    assert ru["status"] == ["Неполный", "Есть риски", "Успешно", "Недостаточно данных"]
    assert ru["classes"] == ["rel-incomplete", "fail"]
    assert ru["deltas"] == ["+12,5 %", "+0,3 п.п.", "−8 %", "0 %", "0 п.п.", "—", "—"]
    assert ru["tones"] == ["warn", "fail", "ok", "neutral", "neutral", "neutral"]
    assert ru["levels"] == ["на 600 RPS", f"на потолке, 1{NBSP}540 RPS", "на полке, 200 RPS", ""]
    assert ru["title"] == "400 мс → 451 мс"
    assert ru["reasons"] == ["Нет прошлого релиза с этим сервисом", "Нагрузка прогонов несопоставима", "Сравнить не удалось"]
    en = run_js(FILES, expression, locale="en")
    assert en["status"][:2] == ["Incomplete", "Risks found"]
    assert en["deltas"][:3] == ["+12.5 %", "+0.3 pp", "−8 %"] and en["levels"][1] == "at the ceiling, 1,540 RPS"


def test_trend_model_keeps_the_order_and_the_gaps():
    points = [
        {"release": "25.09", "run_name": "a9", "verdict": "pass", "kind": "target", "level": 600, "p95": 400, "errors": 0.2, "capacity": 900},
        {"release": "25.10", "run_name": "a10", "verdict": "Провал", "p95": None, "errors": "", "capacity": 850, "baseline": True},
        {"release": "", "run_name": "a11", "verdict": "risk", "p95": "420.5", "errors": 0.4},
        None,
    ]
    model = run_js(FILES, f"LL.releases._internals.trendModel({json.dumps(points)})")
    assert model["labels"] == ["25.09", "25.10", "a11"] and model["runs"] == ["a9", "a10", "a11"]
    assert model["verdicts"] == ["pass", "fail", "risk"] and model["baseline"] == [False, True, False]
    assert model["series"] == {"p95": [400, None, 420.5], "errors": [0.2, None, 0.4], "capacity": [900, 850, None]}
    assert model["levels"][0] == {"kind": "target", "level": 600} and model["empty"] is False
    assert run_js(FILES, "LL.releases._internals.trendModel([{release: 'x'}]).empty") is True


def test_address_of_the_pages():
    out = run_js(FILES, """(() => {
      const R = LL.releases._internals;
      return {
        names: ['/releases/DEMO%2025%2F11', '/releases/25.11', '/releases', '/releases/%E0%A4%A'].map(R.releaseNameFromPath),
        url: R.releaseUrl('DEMO 25/11'),
        queries: ['?service=api&type=SOAK', '?type=phases', ''].map(R.parseQuery),
        built: [R.buildQuery({ service: 'api', type: 'stress' }), R.buildQuery({})]
      };
    })()""")
    assert out["names"] == ["DEMO 25/11", "25.11", "", "%E0%A4%A"]
    assert out["url"] == "/releases/DEMO%2025%2F11"
    assert out["queries"] == [{"service": "api", "type": "soak"}, {"service": "", "type": ""}, {"service": "", "type": ""}]
    assert out["built"] == ["?service=api&type=stress", ""]


def test_rows_model_marks_a_report_built_against_another_baseline():
    rows = [
        {
            "service": "api", "test_type": "step", "baseline": True,
            "run": {"run_name": "a11", "run_id": "id11", "verdict": "risk", "manual_verdict": "risk"},
            "previous": {"release": "25.10", "run_name": "a10"},
            "deltas": {"available": True, "kind": "target", "level": 600, "items": [
                {"metric": "p95", "change": 12.5, "unit": "pct", "tone": "warn", "current": 450, "baseline": 400},
                {"metric": "capacity", "change": -11.1, "unit": "pct", "tone": "neutral", "current": 800, "baseline": 900},
            ]},
            "stored_regression": {"baseline_run_name": "a09"},
            "report_url": "/reports/id11", "compare_url": "/compare?run_a=a11&run_b=a10",
            "attempts": [{"run_name": "a11"}, {"run_name": "a11-b"}],
        },
        {"service": "web", "test_type": "soak", "run": None, "deltas": {"available": False, "reason": "no_accepted"}, "attempts": []},
    ]
    models = run_js(FILES, f"LL.releases._internals.rowsModel({json.dumps(rows)})")
    api, web = models
    assert (api["verdict"], api["manual"], api["baseline"], api["level"], api["staleNote"], api["storedBase"]) == (
        "risk", True, True, "на 600 RPS", True, "a09")
    assert api["cells"]["p95"] == {"text": "+12,5 %", "tone": "warn", "title": "400 мс → 450 мс"}
    assert api["cells"]["errors"] == {"text": "—", "tone": "neutral", "title": ""}
    assert api["cells"]["capacity"]["text"] == "−11,1 %" and len(api["attempts"]) == 2
    assert (web["run"], web["available"], web["reason"], web["staleNote"]) == (None, False, "Нет принятого прогона", False)


def test_settings_have_a_jira_card_without_a_check_and_the_release_parent_page():
    from tests.js_harness import REPO_ROOT

    source = (REPO_ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    card = re.search(r"id: 'jira', section: 'jira'[^\n]*", source)
    assert card and "testable: false" in card.group(0) and "scope: 'global'" in card.group(0)
    for path in ("enabled", "base_url", "project_key"):
        assert f"{{ path: '{path}', label: LL.t('settings.field.jira.{path}.label')" in source
    assert "{ path: 'release_parent_page_id', label: LL.t('settings.field.confluence.release_parent_page_id.label')" in source
    texts = run_js([], "['settings.card.jira.title', 'settings.card.jira.desc', 'settings.field.jira.base_url.hint'].map((k) => LL.t(k))", locale="en")
    assert texts[0] == "Jira" and "never calls Jira" in texts[1]
    for text in texts:
        assert_no_cyrillic(text)


def test_verdict_counts_and_release_names_are_escaped():
    html = run_js(FILES, "LL.releases._internals.verdictCountsHtml({pass: 2, fail: 1, risk: 0})")
    assert html == ('<span class="release-verdict-counts"><span class="st ok" title="Успешно">2</span>'
                    '<span class="st fail" title="Провал">1</span></span>')
    assert run_js(FILES, "LL.releases._internals.verdictCountsHtml({})") == '<span class="dim">—</span>'
