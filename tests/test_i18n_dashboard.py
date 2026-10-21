# SPDX-License-Identifier: Apache-2.0
"""English UI of the dashboard, the project cards of the settings page and the forecasting list
(plans/i18n-plan.md, section 2.5, rows 1, 6 and 8).

The dashboard renders in en without Russian text, the builders of dashboard.js, projects.js and forecasting.js speak
the UI language (week counts by plural forms), and without a cookie the Russian output stays as it was. Run names,
services and server messages are data and stay as they came.
"""

from __future__ import annotations

import json
import re

import pytest

from loadlens_app import core  # noqa: F401  (before anything that imports update_page)
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text
from tests.js_harness import run_js

BASE = ["static/js/common.js", "static/js/ui.js"]
DASH = BASE + ["static/js/dashboard.js"]
PROJECTS = BASE + ["static/js/projects.js"]
FORECASTING = BASE + ["static/js/forecasting.js"]
I = "LL.dashboard._internals"
P = "LL.projects._internals"
F = "LL.forecasting._internals"
NOW = "Date.parse('2026-10-16T10:30:00Z')"
NBSP = chr(0xA0)  # setText puts a no-break space into an empty element


def js(value):
    return json.dumps(value, ensure_ascii=False)


def strings(value):
    """Every string inside a JSON-like value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def assert_no_russian(value):
    left = [text for text in strings(value) if CYRILLIC.search(text)]
    assert not left, "Russian text left:\n" + "\n".join(left[:20])


def run_all(files, expressions, locale):
    """Runs {name: expression} in one Node process and returns {name: value}."""
    body = ", ".join(f"{js(name)}: ({expr})" for name, expr in expressions.items())
    return run_js(files, "({" + body + "})", locale=locale)


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
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1).strip()
    main = html[html.index('<main id="main"'):html.index("</main>")]
    return html, title, main[main.index(">") + 1:]


# ---- templates ---------------------------------------------------------------------------------------------


def test_dashboard_in_en_has_no_russian_text(client):
    html, title, page = _page(client, "/", "en")
    assert re.search(r'<html lang="en"', html)
    assert title == "Dashboard"
    assert_no_cyrillic(title + "\n" + visible_text(page))
    assert "<h2>Dashboard</h2>" in page and 'aria-label="Summary"' in page
    assert re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", page) == [
        "Service", "Latest run", "Verdict", "Max RPS", "Trend", '<span class="visually-hidden">Actions</span>',
        "Run", "Service", "Test type", "Verdict", "Report created",
    ]
    assert '<div class="banner-title">Looks like LoadLens is not set up yet</div>' in page
    assert 'href="/settings?wizard=1">Open the wizard</a>' in page and ">Load demo data</button>" in page
    assert 'role="img" aria-label="Run verdicts by week"' in page and "<h3>Recent runs</h3>" in page
    # The page ships the dashboard messages for dashboard.js next to the chart ones.
    assert '"dashboard.weeks.total.other":"{total} in {n} weeks"' in html and '"charts.' in html


def test_dashboard_stays_russian_without_a_cookie(client):
    html, title, page = _page(client, "/", None)
    assert re.search(r'<html lang="ru"', html) and title == "Дэшборд"
    assert re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", page) == [
        "Сервис", "Последний запуск", "Вердикт", "Max RPS", "Тренд", '<span class="visually-hidden">Действия</span>',
        "Запуск", "Сервис", "Тип теста", "Вердикт", "Отчёт создан",
    ]
    assert '<div class="banner-title">Похоже, LoadLens ещё не настроен</div>' in page
    assert "Это займёт несколько минут.</div>" in page and 'id="dashTiles" aria-label="Сводка"' in page
    assert "<p>Три шага, чтобы увидеть первый результат:</p>" in page and ">Весь архив →</a>" in page


def test_settings_and_forecasting_pages_ship_the_messages_of_their_scripts(client):
    html, _, _ = _page(client, "/settings", "en")
    assert '"projects.card.edit":"Edit"' in html
    html, _, _ = _page(client, "/forecasting", "en")
    assert '"forecast.list.hint.other":' in html and '"forecast.list.hint_missing_settings":' in html


# ---- dashboard.js --------------------------------------------------------------------------------------------

HISTORY = [
    {"run_name": "r5", "run_id": "id5", "created_at": "2026-10-15T10:00:00Z", "verdict": "risk", "max_rps": 240, "test_type": "step"},
    {"run_name": "r4", "run_id": "", "created_at": "2026-10-14T10:00:00Z", "verdict": "pass", "max_rps": 120, "test_type": "soak"},
    {"run_name": "r3", "run_id": "id3", "created_at": "2026-10-13T10:00:00Z", "verdict": "pass", "max_rps": 200, "test_type": "step"},
    {"run_name": "r2", "run_id": "id2", "created_at": "2026-10-12T10:00:00Z", "verdict": "fail", "max_rps": None, "test_type": ""},
]
SERVICE = {"service": "checkout", "last_run": "r5", "run_id": "id5", "verdict": "risk", "max_rps": 240, "test_type": "step",
           "created_at": "2026-10-15T10:00:00Z", "history": HISTORY}
WEEKS = [
    {"week_start": "2026-10-05", "counts": {"pass": 2, "fail": 1}},
    {"week_start": "2026-10-12", "counts": {"risk": 1, "no_data": 3}},
]
ACTIVE = {
    "run_id": "abc", "run_name": "nightly", "service": "checkout", "owner": "alice", "status": "running",
    "status_title": "Load running", "loading": True, "started_at": "2026-10-16T10:18:00Z", "max_seconds": 1800,
    "instances_total": 3, "instances_running": 2, "can_stop": True, "url": "/launch/abc",
    "live": {"ok": True, "message": "", "rps": 123.45, "errors": 0.4, "points": [[1, 1], [2, 2]]},
}
JOB = {"run_name": "", "service": "svc", "progress": 41.6, "message": "collecting", "job_id": "j1"}
# renderHero draws into stub elements; the result is the text and markup it left there.
HERO = (
    "((last, kpi, emptyTitle) => { const els = {};"
    " document.getElementById = (id) => els[id] || (els[id] = document.createElement('div'));"
    f" {I}.renderHero(last, kpi, emptyTitle);"
    " return { title: els.lastTitle.textContent, meta: els.lastMeta.textContent, badges: els.lastBadges.innerHTML,"
    " kpis: els.lastKpis.innerHTML }; })"
)
FIVE_MIN_AGO = "new Date(Date.now() - 5 * 60000).toISOString()"


def dashboard_expressions():
    weeks = f"{I}.weeksModel({js(WEEKS)})"
    one_week = f"{I}.weeksModel({js(WEEKS[:1])})"
    last = {"run_name": "r5", "service": "checkout", "test_type": "step", "max_rps": 240, "verdict": "risk",
            "sla": {"passed": 0, "total": 0}, "start_ms": 1792058400000, "end_ms": 1792062000000}
    return {
        "sla": f"[{I}.slaBadge(null), {I}.slaBadge({js({'passed': 2, 'total': 3})})]",
        "labels": f"{weeks}.series.map((s) => s.label)",
        "summary": f"{I}.weeksSummary({weeks})",
        "summary_empty": f"{I}.weeksSummary({I}.weeksModel([{{ week_start: '2026-10-05', counts: {{}} }}]))",
        "totals": f"[{I}.weeksTotalText({weeks}), {I}.weeksTotalText({one_week}), {I}.weeksTotalText({I}.weeksModel([]))]",
        "tooltip": f"[{I}.weeksTooltipTitle({weeks}, [{{ dataIndex: 0 }}]), {I}.weeksTooltipFooter({weeks}, [{{ dataIndex: 0 }}]),"
                   f" {I}.weeksTooltipTitle({weeks}, []), {I}.weeksTooltipFooter(null, [{{ dataIndex: 0 }}])]",
        "sparkline": f"{I}.sparkline({js(HISTORY)})",
        "hero_step": f"{I}.heroKpis({js(last)}, {js({'previous': {'run_name': 'r3', 'max_rps': 200}})})",
        "hero_soak": f"{I}.heroKpis({js({'test_type': 'soak', 'max_rps': 90})}, {{}})",
        "hero_spike": f"{I}.heroKpis({js({'test_type': 'spike', 'max_rps': 90})}, {js({'previous': {'run_name': 'p', 'max_rps': None}})})",
        "hero_stress": f"{I}.heroKpis({js({'test_type': 'stress', 'max_rps': 90})}, {{}})",
        "duration": "LL.formatDuration(3600000)",
        "kpi": f"{I}.kpiTiles({js({'kpi': {'runs_7d': 3, 'runs_30d': 8, 'risky_30d': 2, 'risky_share_30d': 0.25}, 'last_run': {'run_name': 'r5', 'created_at': '2026-10-15T09:55:00Z'}})}, Date.parse('2026-10-15T10:00:00Z'))",
        "kpi_empty": f"{I}.kpiTiles({js({'kpi': {'runs_7d': 0, 'runs_30d': 0, 'risky_share_30d': None}, 'last_run': None})})",
        "jobs": f"[{I}.jobsTile(null), {I}.jobsTile([]), {I}.jobsTile([{{}}])]",
        "elapsed": f"[{I}.elapsedText('2026-10-16T10:18:00Z', 1800, {NOW}), {I}.elapsedText(null, 3600, {NOW})]",
        "active": f"{I}.activeRunSummary({js(ACTIVE)}, {NOW})",
        "active_waiting": f"{I}.activeRunSummary({js(dict(ACTIVE, live=None, status='starting'))}, {NOW})",
        "active_unnamed": f"{I}.activeRunSummary({{}}, {NOW}).name",
        "active_tiles": f"[{I}.activeTile(null), {I}.activeTile({{ total: 0, runs: [] }})]",
        "active_row": f"{I}.activeRowHtml({js(ACTIVE)}, {NOW})",
        "active_more": f"[{I}.activeMoreHtml(2), {I}.activeMoreHtml(0)]",
        "service_row": f"{I}.serviceRowHtml({js(SERVICE)})",
        "service_row_alone": f"{I}.serviceRowHtml({js(dict(SERVICE, history=HISTORY[3:], test_type=''))})",
        "job_row": f"{I}.jobRowHtml({js(JOB)})",
        "manual_pill": f"{I}.verdictPill('risk', '', {js({'verdict': 'risk', 'author': 'ivanov'})})",
        "hero_empty": f"{HERO}(null, null)",
        "hero_no_data": f"{HERO}(null, null, LL.t('dashboard.no_data'))",
        "hero_last": f"{HERO}({{ ...{js(last)}, created_at: {FIVE_MIN_AGO} }}, {{}})",
    }


def test_dashboard_builders_in_en():
    out = run_all(DASH, dashboard_expressions(), "en")
    assert_no_russian(out)
    none, partial = out["sla"]
    assert (none["text"], none["title"]) == ("SLA not set", "SLA criteria were not checked for this run")
    assert (partial["text"], partial["title"]) == ("SLA 2/3", "SLA criteria met: 2 of 3")
    assert out["labels"] == ["Passed", "Risks found", "Failed", "Insufficient data"]
    assert out["summary"] == "Verdicts by week over 2 weeks: 7 in total; Passed — 2, Risks found — 1, Failed — 1, Insufficient data — 3"
    assert out["summary_empty"] == "Verdicts by week: no finished reports in 1 week"
    assert out["totals"] == ["7 in 2 weeks", "3 in 1 week", ""]
    assert out["tooltip"] == ["Week 05.10–11.10", "Total: 3", "", ""]
    assert "<title>r2 · 2026-10-12 10:00 · no RPS · Failed</title>" in out["sparkline"]
    assert 'aria-label="Max RPS of the runs, oldest to newest: no data, 200.0, 120.0, 240.0"' in out["sparkline"]

    rps, window, kind = out["hero_step"]
    assert [rps["label"], window["label"], kind["label"]] == ["Max RPS", "Test window", "Test type"]
    assert rps["note"] == "vs “r3”" and rps["title"] == "Previous run of the same type: r3, 200.0 RPS"
    assert window["value"] == out["duration"] == "1 h 00 min"
    assert kind["value"] == "Step (max perf)" and kind["note"] == "RPS: peak performance"
    assert out["hero_soak"][0]["label"] == "Sustained RPS" and out["hero_soak"][0]["note"] == "no previous run of this type"
    assert out["hero_soak"][2]["note"] == "RPS: sustained (mean)"
    assert out["hero_spike"][0]["label"] == "Spike RPS" and out["hero_spike"][0]["note"] == "vs “p”: no data to compare"
    assert out["hero_spike"][2]["note"] == "RPS: spike level"
    assert out["hero_stress"][0]["label"] == "Saturation RPS" and out["hero_stress"][2]["note"] == "RPS: saturation point"

    assert out["kpi"]["runs"]["note"] == "in 30 days: 8" and out["kpi"]["risky"]["note"] == "2 of 8: risks and failures"
    assert out["kpi"]["updated"]["value"] == "5 min ago" and out["kpi"]["updated"]["note"] == "report “r5”"
    assert out["kpi_empty"]["risky"]["note"] == "no runs in 30 days" and out["kpi_empty"]["updated"]["note"] == "no reports yet"
    assert [tile["note"] for tile in out["jobs"]] == ["job list unavailable", "no active jobs", "(untitled) · 0%"]
    assert out["elapsed"] == ["12 min of 30 min", "limit 1 h 00 min"]
    assert out["active"]["progress"] == "instances: 2 of 3 · 12 min of 30 min" and out["active"]["liveText"] == "123.5 RPS · errors 0.40%"
    assert out["active_waiting"]["liveNote"] == "Metrics will appear once the load starts"
    assert out["active_unnamed"] == "(untitled)"
    assert [tile["note"] for tile in out["active_tiles"]] == ["run list unavailable", "no runs in progress"]
    row = out["active_row"]
    assert 'title="RPS and error rate over the last 15 minutes"' in row and ">Open</a>" in row and ">Stop</button>" in row
    assert out["active_more"] == ['and 2 more — <a href="/launch#runsPanel">all runs</a>', ""]

    service = out["service_row"]
    assert 'title="Against run “r3” of the same type: 200.0 RPS"' in service and ">Compare</a>" in service
    assert 'aria-label="Compare “r5” with the previous run of the same type, “r3”"' in service
    assert 'title="Compare “r5” with the previous run"' in out["service_row_alone"]
    assert ">In progress</span>" in out["job_row"] and ">(untitled)</div>" in out["job_row"] and ">Follow</a>" in out["job_row"]
    assert ">manual</span>" in out["manual_pill"] and ">Risks found</span>" in out["manual_pill"]

    assert out["hero_empty"]["title"] == "No reports yet"
    assert out["hero_empty"]["meta"] == "The outcome of the latest run will appear here: verdict, SLA and Max RPS."
    assert out["hero_no_data"]["title"] == "No data" and out["hero_no_data"]["meta"] == NBSP
    assert out["hero_last"]["meta"] == "checkout · report created 5 min ago"
    assert ">SLA not set</span>" in out["hero_last"]["badges"] and "Test window" in out["hero_last"]["kpis"]


def test_dashboard_builders_keep_russian():
    out = run_all(DASH, dashboard_expressions(), "ru")
    assert out["summary"] == "Вердикты по неделям за 2 нед.: всего 7; Успешно — 2, Есть риски — 1, Провал — 1, Недостаточно данных — 3"
    assert out["summary_empty"] == "Вердикты по неделям: за 1 нед. завершённых отчётов нет"
    assert out["totals"] == ["7 за 2 нед.", "3 за 1 нед.", ""]
    assert out["tooltip"] == ["Неделя 05.10–11.10", "Всего: 3", "", ""]
    assert out["active_more"] == ['и ещё 2 — <a href="/launch#runsPanel">все прогоны</a>', ""]
    assert out["job_row"] == (
        '<div class="job-row"><span class="pill running">В работе</span><div><div class="name">(без названия)</div>'
        '<div class="meta">svc · 42% — collecting</div></div><a href="/new?job=j1" data-min-role="engineer">Следить</a></div>'
    )
    service = out["service_row"]
    assert 'title="К прогону «r3» того же типа: 200.0 RPS"' in service and ">Сравнить</a>" in service
    assert 'aria-label="Сравнить «r5» с предыдущим запуском того же типа «r3»"' in service
    assert out["hero_spike"][0]["note"] == "к «p»: нет данных для сравнения" and out["hero_stress"][2]["note"] == "RPS: точка насыщения"
    assert out["hero_empty"] == {
        "title": "Отчётов пока нет", "meta": "Здесь появится итог последнего запуска: вердикт, SLA и Max RPS.", "badges": "", "kpis": "",
    }
    assert out["hero_last"]["meta"] == "checkout · отчёт создан 5 мин назад" and ">SLA не задан</span>" in out["hero_last"]["badges"]
    assert ">вручную</span>" in out["manual_pill"]


# ---- projects.js (the Projects section of the settings page) ---------------------------------------------------

PROJECT = {
    "id": "nsi", "title": "NSI", "description": "", "reports": 4, "last_report_at": "2026-10-15T10:00:00Z", "last_verdict": "fail",
    "services": [{"id": "s1", "title": "S1", "reports": 4, "last_report_at": None, "own_sections": ["queries"]}],
    "overrides": [
        {"section": "llm", "status": "inherited"},
        {"section": "system_context", "status": "own", "own_domains": ["jvm", "overall"]},
        {"section": "prompts", "status": "same"},
    ],
    "readiness": {
        "target_rps": None, "performance_query": "", "has_system_context": False, "query_counts": {},
        "missing_sla": {"forecast": ["max_performance_query", "mean_latency_query"], "resource_plan": ["cpu_query"]},
        "query_packs": [{"title": "k6", "outdated": True}, {"title": "jvm", "outdated": False}],
    },
}
FULL = {
    "target_rps": 250, "performance_query": "RPS", "has_system_context": True, "query_counts": {"jvm": 3, "lt_framework": 4},
    "missing_sla": {"forecast": [], "resource_plan": []}, "query_packs": [{"title": "k6"}],
}


def projects_expressions():
    return {
        "missing": f"{P}.readinessItems({js(PROJECT['readiness'])}).map((i) => i.text)",
        "full": f"{P}.readinessItems({js(FULL)}).map((i) => i.text)",
        "card": f"{P}.projectCard({js(PROJECT)})",
        "empty_services": f"{P}.servicesTable({{ services: [] }}, () => '')",
        "complete": f"{P}.readinessList({{ id: 'p', readiness: {js(FULL)} }})",
    }


def test_projects_builders_in_en():
    out = run_all(PROJECTS, projects_expressions(), "en")
    assert_no_russian(out)
    assert out["missing"] == [
        "Target RPS not set", "Performance query not selected", "System context is empty", "No queries",
        "Forecast: max RPS, mean response time not set", "Resource plan: node CPU not set",
    ]
    assert out["full"] == [
        "Target RPS: 250", "Performance query selected", "System context filled in", "Queries: 7, domains: 2",
        "Forecast: SLA references set", "Resource plan: SLA references set",
    ]
    card = out["card"]
    assert ">Select</button>" in card and ">Edit</button>" in card and ">Copy</button>" in card and ">Delete…</button>" in card
    assert 'Services: 1 · Reports: 4 · Latest: 2026-10-15 10:00 <span class="pill fail">Failed</span>' in card
    assert '<span class="readiness-count">0 of 6 set up</span>' in card and ">Query packs: 2, updates available: 1</span>" in card
    assert re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", card) == [
        "Service", "Reports", "Latest report", "Own settings", '<span class="visually-hidden">Actions</span>',
    ]
    assert 'title="Revert to the project settings"' in card and ">Queries ×</button>" in card and ">Move…</button>" in card
    assert "<span>What is overridden</span>" in card and ">own sections: 1</small>" in card
    assert ">shared</span>" in card and ">own</span>" in card and ">copy of shared</span>" in card
    assert "System context<div class=\"dim\">JVM, Summary</div>" in card and ">Revert to shared</button>" in card
    assert out["empty_services"] == '<p class="dim">No services.</p>'
    assert "6 of 6 set up" in out["complete"] and ">Query packs: 1</span>" in out["complete"]


def test_projects_builders_keep_russian():
    out = run_all(PROJECTS, projects_expressions(), "ru")
    assert out["missing"][4:] == ["Прогноз: не заданы макс. RPS, среднее время", "План ресурсов: не заданы CPU узлов"]
    card = out["card"]
    assert 'Сервисов: 1 · Отчётов: 4 · Последний: 2026-10-15 10:00 <span class="pill fail">Провал</span>' in card
    assert "<span>Что переопределено</span>" in card and ">своих разделов: 1</small>" in card and ">общие</span>" in card
    assert "Контекст системы<div class=\"dim\">JVM, Итог</div>" in card and ">Удалить…</button>" in card
    assert ">Пакетов запросов: 2, есть обновления: 1</span>" in card and "Настроено 0 из 6" in card


# ---- forecasting.js (the list of /forecasting) -----------------------------------------------------------------

REPORTS = [
    {"run_name": "r1", "service": "api", "test_type": "stress", "created_at": "2026-10-15T10:00:00Z", "verdict": "pass", "load_model": "closed", "run_id": "1"},
    {"run_name": "r2", "service": "api", "test_type": "step", "created_at": None, "verdict": None, "load_model": "open", "run_id": "2"},
]
ROWS = (
    "((reports) => { const body = document.createElement('tbody');"
    " document.getElementById = (id) => (id === 'forecastRows' ? body : null);"
    f" {F}.renderRows(reports);"
    " return { html: body.innerHTML, rows: body.children.map((tr) => tr.children.map((td) => (td.children.length ? td.children[0].textContent : td.textContent))) }; })"
)


def forecasting_expressions():
    return {
        "hints": f"[[0, 0], [1, 0], [0, 3], [1, 2]].map(([a, b]) => {F}.hintText(a, b))",
        "rows": f"{ROWS}({js(REPORTS)})",
        "empty": f"{ROWS}([])",
    }


def test_forecasting_list_in_en():
    out = run_all(FORECASTING, forecasting_expressions(), "en")
    assert_no_russian(out)
    assert out["hints"] == [
        "",
        "1 more step or stress test has no forecast yet: 1 — the data lacks the required series "
        "(add queries to “Load testing tool” and regenerate the report).",
        "3 more step and stress tests have no forecast yet: 3 — the forecast settings are not set (SLA criteria → Capacity forecast).",
        "3 more step and stress tests have no forecast yet: 1 — the data lacks the required series "
        "(add queries to “Load testing tool” and regenerate the report), 2 — the forecast settings are not set "
        "(SLA criteria → Capacity forecast).",
    ]
    assert [row[4:] for row in out["rows"]["rows"]] == [["Passed", "closed"], ["Insufficient data", "open"]]
    assert out["rows"]["rows"][0][2] == "Stress"
    assert "There are no reports to forecast yet." in out["empty"]["html"] and ">New report</a>" in out["empty"]["html"]


def test_forecasting_list_keeps_russian():
    out = run_all(FORECASTING, forecasting_expressions(), "ru")
    assert out["hints"][3] == (
        "Ещё 3 ступенчатых и стресс-тестов пока без прогноза: 1 — нет нужных рядов в данных (добавьте запросы в "
        "«Нагрузочный инструмент» и перегенерируйте отчёт), 2 — не заданы настройки прогноза (SLA-критерии → Прогноз мощностей)."
    )
    assert out["hints"][1].startswith("Ещё 1 ступенчатых и стресс-тестов пока без прогноза: 1 — ")
    assert [row[4:] for row in out["rows"]["rows"]] == [["Успешно", "закрытая"], ["Недостаточно данных", "открытая"]]
    assert "Отчётов, по которым возможен прогноз, пока нет." in out["empty"]["html"] and ">Новый отчёт</a>" in out["empty"]["html"]
