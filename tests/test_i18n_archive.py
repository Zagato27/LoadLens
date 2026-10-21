# SPDX-License-Identifier: Apache-2.0
"""English UI of the archive (/reports) and compare (/compare) pages (plans/i18n-plan.md, section 2.5, rows 3-4).

The pages render without Russian text in en, the JS builders of archive.js and compare.js produce English strings
in en and the same Russian strings as before in ru (server data such as run names stays as it is).
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

BASE = ["static/js/common.js", "static/js/ui.js"]
ARCHIVE_JS = BASE + ["static/js/archive.js"]
COMPARE_JS = BASE + ["static/js/charts.js", "static/js/compare.js"]
NBSP = chr(0xA0)


class _NoDb:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        raise AssertionError("the page itself does not query the database")

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app.blueprints import compare, dashboard

    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    for module in (dashboard, compare):
        monkeypatch.setattr(module, "_ts_conn", lambda: _NoDb())
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _html_lang(html: str) -> str:
    return re.search(r'<html lang="([^"]*)"', html).group(1)


def _own_text(html: str) -> str:
    """Visible text of what the page itself renders: its <title> and the <main> block."""
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
    main = re.search(r'<main id="main".*?</main>', html, re.S).group(0)
    return title + "\n" + visible_text(main)


def _js(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def archive(expression, **kwargs):
    return run_js(ARCHIVE_JS, f"const A = LL.archive._internals; {expression}", **kwargs)


def compare(expression, **kwargs):
    return run_js(COMPARE_JS, f"const C = LL.compare._internals; {expression}", **kwargs)


# ---- pages ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("url", "title"), [("/reports", "Report archive"), ("/reports?service=svc", "Report archive"), ("/compare", "Compare tests")])
def test_pages_render_in_english(client, url, title):
    use_locale(client, "en")
    resp = client.get(url)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert _html_lang(html) == "en"
    assert f"<title>{title}</title>" in html
    assert_no_cyrillic(_own_text(html))


def test_english_pages_carry_their_dictionaries(client):
    use_locale(client, "en")
    archive_page = client.get("/reports").get_data(as_text=True)
    assert '"archive.selection.count":"{n} selected"' in archive_page
    assert '"compare.' not in archive_page
    compare_page = client.get("/compare").get_data(as_text=True)
    assert '"compare.hint.pick_both":"Pick both runs."' in compare_page
    assert '"charts.' in compare_page  # charts.js on the compare page keeps its namespace


def test_english_archive_markup(client):
    use_locale(client, "en")
    html = client.get("/reports").get_data(as_text=True)
    assert '<th data-sort="run_name" tabindex="0">Run</th>' in html
    assert '<select id="verdictFilter"><option value="">All</option></select>' in html
    assert 'placeholder="Search by run name or service…"' in html
    # Phone cards read the column names from the header cells, so they follow the language too.
    heads = re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", html, re.S)
    assert [re.sub(r"<[^>]+>", "", h).strip() for h in heads] == [
        "Select for comparison", "Run", "Service", "Test type", "Test date", "Verdict", "SLA", "Max RPS", "p95",
        "Report created", "Actions",
    ]


def test_russian_pages_keep_their_text(client):
    page = client.get("/reports").get_data(as_text=True)
    assert _html_lang(page) == "ru" and "<title>Архив отчётов</title>" in page
    assert '<th data-sort="end_time" tabindex="0" title="Окончание теста и длительность">Дата теста</th>' in page
    assert 'placeholder="Поиск по названию запуска или сервису…" title="Клавиша / — перейти к поиску"' in page
    assert '<button type="button" class="btn btn-primary btn-sm" id="selectionCompare">Сравнить</button>' in page
    page = client.get("/compare").get_data(as_text=True)
    assert "<title>Сравнение тестов</title>" in page
    assert '<span>Прогон A <span class="picker-badge">A</span></span>' in page
    assert '<span class="dim">Агрегат:</span>' in page and 'aria-label="Агрегат"' in page
    assert '<span class="dim" id="compareHint">Выберите оба прогона.</span>' in page


# ---- archive.js ------------------------------------------------------------------------------------------

STATE = {"page": 1, "sort": "end_time", "dir": "desc", "q": "night", "service": "svc", "verdict": "fail", "testType": "soak"}
LABELS = "{ service: (id) => 'svc title', testType: (k) => LL.testTypeLabel(k) }"


def test_archive_builders_in_english():
    chips = archive(f"A.activeChips({_js(STATE)}, {LABELS})", locale="en")
    assert chips == [
        {"key": "q", "label": "Search", "value": "“night”"},
        {"key": "service", "label": "Service", "value": "svc title"},
        {"key": "verdict", "label": "Verdict", "value": "Failed"},
        {"key": "testType", "label": "Test type", "value": "Soak (long-running)"},
    ]
    texts = archive("[0, 1, 2, 3].map((n) => [A.selectionLabel(n), A.selectionHint(n)])", locale="en")
    assert texts == [
        ["0 selected", ""],
        ["1 selected", "Select one more run to compare"],
        ["2 selected", ""],
        ["3 selected", "Exactly two runs can be compared — clear the extra selections"],
    ]
    assert archive("[A.pagerText(2, 57, 20), A.pagerText(1, 0, 0)]", locale="en") == ["Showing 21–40 of 57", "Nothing found"]
    assert archive("A.verdictTitle({ sla_verdict: 'fail', llm_verdict: 'Успешно' })", locale="en") == "By SLA: Failed; AI assessment: Passed"
    assert archive("A.verdictTitle({ sla_verdict: 'fail', llm_verdict: 'fail' })", locale="en") == ""
    job = {"kind": "forecast_confluence", "service": "svc", "status": "running", "progress": 40, "message": "step 2"}
    assert archive(f"A.jobMeta({_js(job)})", locale="en") == "publishing a forecast to Confluence · svc · 40% — step 2"
    rendered = archive(
        "[A.shortTypeLabel('step'), A.timeCell('2026-10-15T09:00:00Z', Date.parse('2026-10-15T10:00:00Z')).text,"
        " A.durationText({ start_time: '2026-10-15T08:00:00Z', end_time: '2026-10-15T09:05:00Z' })]",
        locale="en",
    )
    for text in rendered:
        assert text and not CYRILLIC.search(text), rendered


@pytest.mark.parametrize(
    ("value", "ru", "en"),
    [("95.46", "95,5", "95.5"), ("1234.5", f"1{NBSP}235", "1,235"), ("-1500", f"-1{NBSP}500", "-1,500"), ("12.0", "12", "12"), ("null", "—", "—")],
)
def test_archive_rps_follows_the_language(value, ru, en):
    assert archive(f"A.formatRps({value})") == ru
    assert archive(f"A.formatRps({value})", locale="en") == en


def test_archive_builders_in_russian_are_unchanged():
    assert archive("[A.pagerText(2, 57, 20), A.pagerText(3, 57, 17), A.pagerText(1, 0, 0)]") == [
        "Показано 21–40 из 57", "Показано 41–57 из 57", "Ничего не найдено",
    ]
    assert archive("A.verdictTitle({ sla_verdict: 'risk', llm_verdict: 'pass' })") == "По SLA: Есть риски; оценка ИИ: Успешно"
    manual = {"manual_verdict": {"verdict": "pass", "author": "ivanov", "reason": "ok"}, "sla_verdict": "fail", "llm_verdict": "pass"}
    assert archive(f"A.verdictTitle({_js(manual)})") == "Изменён вручную: ivanov. Причина: ok"
    jobs = [
        {"kind": "confluence", "status": "error", "message": "нет доступа"},
        {"kind": "report", "service": "svc", "status": "running", "progress": "x", "message": ""},
    ]
    assert archive(f"{_js(jobs)}.map(A.jobMeta)") == ["публикация в Confluence · нет доступа", "svc · 0% — "]


# ---- compare.js ------------------------------------------------------------------------------------------

RUNS_A = {
    "final": {
        "domain": "final", "verdict": "pass", "sla_verdict": "risk", "test_type": "step", "start_ms": 0, "end_ms": 3900000,
        "parsed": {"peak_performance": {"max_rps": 250}, "saturation": {"step": 4}, "findings": [{"severity": "High", "summary": "p95 grew", "component": "db"}]},
    },
    "jvm": {"domain": "jvm", "verdict": "fail"},
}
RUNS_B = {"final": {"domain": "final", "verdict": "Есть риски", "test_type": "soak", "parsed": {"stability_under_load": {"actual_rps": 80}}}}

BUILDERS = f"""[
  C.verdictGridHtml({_js(RUNS_A)}, {_js(RUNS_B)}, 'night-1', 'night-0', {{}}),
  C.findingsHtml({_js(RUNS_A['final'])}, 'A · night-1') + C.findingsHtml(null, 'B · night-0'),
  C.trendHtml(12.34, 'rps'), C.trendHtml(-3, 'rps'), C.trendHtml(5, 'latency'), C.trendHtml(7, 'other'), C.trendHtml(0.01, 'rps'),
  C.metricsTableHead('p95'), C.metricsTableHead('avg'), C.metricsTableHead('max'),
  C.metricRowHtml({{ query_label: 'RPS', value_a: 1.5, value_b: 2, trend_pct: 33.3 }}),
  C.chartRowHtml('RPS'),
  C.seriesTableHtml([{{ series: 'pod-1', value_a: 1, value_b: 2, trend_pct: -10 }}], 'latency', 'max'),
  C.noMetricsHtml(), C.startHtml(),
  C.aggNote('p95'), C.aggNote('avg'), C.aggNote('max'),
  C.runHint(null, null), C.runHint({{ value: 'x' }}, {{ value: 'x' }}), C.runHint({{ value: 'x' }}, {{ value: 'y' }}),
  C.keyRpsText({{ parsed: {{ spike_analysis: {{ max_spike_rps: 900 }} }} }}, {{}}),
  C.runMeta({{ service: 'svc', end_time: '2026-10-01T10:00:00Z', test_type: 'soak', verdict: 'fail' }}),
]"""


def test_compare_builders_in_english():
    out = compare(BUILDERS, locale="en")
    for text in out:
        assert text and not CYRILLIC.search(text), text
    grid = out[0]
    assert '<div class="vg-head">Indicator</div>' in grid and "<div>Final verdict</div>" in grid
    assert '<span class="pill ok">Passed</span>' in grid and '<span class="pill warn">Risks found</span>' in grid
    assert "250 (saturation point)" in grid and "80 (held)" in grid
    assert '<span class="trend better" title="An increase counts as an improvement">↑ 12.3 % · better</span>' == out[2]
    assert out[6] == '<span class="trend neutral">no change</span>'
    assert out[8] == '<thead><tr><th>Metric</th><th class="num">A (average)</th><th class="num">B (average)</th><th>Change of B vs A</th></tr></thead>'
    assert 'aria-label="Chart: RPS, runs A and B"' in out[11]
    assert out[-5:-2] == ["Pick both runs.", "The same run is picked twice.", "Press “Compare”."]
    assert out[-2] == "900 (spike)"
    assert out[-1] == "svc · 2026-10-01 10:00 · Soak (long-running) · Failed"


def test_compare_builders_in_russian_are_unchanged():
    out = compare(BUILDERS)
    grid = out[0]
    assert grid.startswith('<div class="vg-head">Показатель</div><div class="vg-head">A · night-1</div><div class="vg-head">B · night-0</div>')
    assert "<div>Итоговый вердикт</div>" in grid and "<div>Вердикт по SLA</div>" in grid and "<div>Длительность окна</div>" in grid
    assert "Макс. RPS / ключевой RPS типа теста" in grid
    assert "<div>250 (точка насыщения)</div>" in grid and "<div>80 (удержано)</div>" in grid
    assert out[1].endswith('<div class="findings-col"><h4>B · night-0</h4><div class="dim">Находок нет или итоговый анализ отсутствует.</div></div>')
    assert out[2] == '<span class="trend better" title="Рост считается улучшением">↑ 12.3 % · лучше</span>'
    assert out[3] == '<span class="trend worse" title="Рост считается улучшением">↓ 3.0 % · хуже</span>'
    assert out[4] == '<span class="trend worse" title="Рост считается ухудшением">↑ 5.0 % · хуже</span>'
    assert out[5] == '<span class="trend neutral" title="Направление не определено по названию метрики">↑ 7.0 %</span>'
    assert out[6] == '<span class="trend neutral">без изменений</span>'
    assert out[7] == '<thead><tr><th>Метрика</th><th class="num">A (p95)</th><th class="num">B (p95)</th><th>Изменение B к A</th></tr></thead>'
    assert 'A (максимум)' in out[9] and "<th>Серия</th>" in out[12] and 'B (максимум)' in out[12]
    assert '<div class="dim">нажмите, чтобы открыть график</div>' in out[10]
    assert '<span class="dim">A — сплошная, B — пунктир; время от старта прогона. Выделите участок мышью, чтобы приблизить</span>' in out[11]
    assert 'aria-label="График: RPS, прогоны A и B"' in out[11] and "Сбросить масштаб</button>" in out[11]
    assert "<h3>Метрик для сравнения нет</h3>" in out[13] and 'data-min-role="engineer">Новый отчёт</a>' in out[13]
    assert '<button type="button" class="btn" id="compareLatestBtn">Сравнить последний прогон</button>' in out[14]
    assert out[15] == "p95 по всем точкам метрики за окно теста: показывает «хвост» без единичных выбросов."
    assert out[18:21] == ["Выберите оба прогона.", "Выбран один и тот же прогон.", "Нажмите «Сравнить»."]
    assert out[-2] == "900 (всплеск)"


def test_compare_builders_escape_server_text():
    html = compare("C.chartRowHtml('CPU \"busy\" <%>') + C.findingsHtml({ parsed: { findings: [{ summary: '<b>x</b>' }] } }, 'A · <run>')", locale="en")
    assert "<b>" not in html and "<run>" not in html
    assert "CPU &quot;busy&quot; &lt;%&gt;" in html and "A · &lt;run&gt;" in html
