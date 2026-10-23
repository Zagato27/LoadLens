# SPDX-License-Identifier: Apache-2.0
"""Stage 3 of plans/instrument-ui-plan.md: archive, compare and forecasting in the «instrument» look.

/runs?include=sla (SLA n/m and p95 of the page rows in the same statement), the archive markup (filters with the
caption inside, SLA and p95 columns, selection bar, text pager), the pure helpers of archive.js under Node, the
forecasting templates and the scope of their page section of app.css. The database is faked.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from loadlens_app import core  # noqa: E402
from tests.i18n_helpers import use_locale  # noqa: E402
from tests.js_harness import run_js  # noqa: E402
from tests.test_ui_archive import RecordingConnection  # noqa: E402

END = datetime(2026, 10, 1, 11, 0, tzinfo=timezone.utc)
START = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
ARCHIVE_JS = ["static/js/common.js", "static/js/ui.js", "static/js/archive.js"]
NOW = 1792058400000  # 2026-10-15T10:00:00Z
CHECKS = [
    {"name": "target_rps", "actual": 249.3, "passed": True},
    {"name": "p95_latency", "actual": 450.0, "passed": False},
    {"name": "error_rate", "actual": 0.4, "passed": True},
]


@pytest.fixture
def client(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app.blueprints import dashboard

    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    monkeypatch.setattr(core, "_ts_conn", lambda: RecordingConnection())
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: RecordingConnection())
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_ensure_run_index_table", lambda conn, cfg: None)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def use_conn(monkeypatch, conn):
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)
    return conn


def _row(name, total, checks=None, manual=(None, None, None, None)):
    """A page row of /runs: 13 columns, the manual verdict (4) and, with include=sla, the SLA checks."""
    base = (name, START, END, "checkout", "risk", "risk", "risk", END, "step", f"id-{name}", "", 249.3, total)
    return base + tuple(manual) + ((checks,) if checks is not ... else ())


# ---- /runs?include=sla ------------------------------------------------------------------------------------


def test_runs_without_include_keep_the_statement_and_the_keys(client, monkeypatch):
    conn = use_conn(monkeypatch, RecordingConnection(fetchall_results=[[_row("a", 1, checks=...)]]))

    body = client.get("/runs").get_json()

    (text, _params), = conn.log
    assert "sla_checks" not in text and "sla_details" not in text
    assert "manual_author, manual_changed_at FROM joined" in text
    assert "sla" not in body[0]


def test_runs_include_sla_reads_the_checks_of_the_page_rows_in_the_same_statement(client, monkeypatch):
    rows = [
        _row("a", 3, CHECKS),
        _row("b", 3, json.dumps([{"name": "error_rate", "actual": 2, "passed": False}])),
        _row("c", 3, None),
    ]
    conn = use_conn(monkeypatch, RecordingConnection(fetchall_results=[rows]))

    resp = client.get("/runs", query_string={"include": "sla", "q": "x", "offset": 20, "limit": 3})

    assert resp.status_code == 200 and resp.headers["X-Total-Count"] == "3"
    (text, params), = conn.log  # still one statement
    assert ("manual_changed_at, (SELECT COALESCE(NULLIF(r.sla_details->'checks', '[]'::jsonb), "
            "r.parsed->'deterministic_sla'->'checks') FROM public.llm_reports r WHERE r.domain = 'final' "
            "AND r.run_name = joined.run_name AND r.created_at = joined.report_created_at LIMIT 1) AS sla_checks "
            "FROM joined") in text
    assert text.endswith("OFFSET %s LIMIT %s")
    assert params == ("%x%", "%x%", 20, 3)  # no new parameters
    body = resp.get_json()
    assert body[0]["sla"] == {"passed": 2, "total": 3, "p95_ms": 450.0, "p95_passed": False}
    assert body[1]["sla"] == {"passed": 0, "total": 1, "p95_ms": None, "p95_passed": None}
    assert body[2]["sla"] is None
    assert body[0]["manual_verdict"] is None and body[0]["max_rps"] == 249.3


def test_runs_include_ignores_unknown_names():
    from loadlens_app.blueprints.dashboard import _runs_include

    assert _runs_include("SLA, steps,") == {"sla"}
    assert _runs_include("") == set() and _runs_include(None) == set() and _runs_include("sla2") == set()


def test_runs_sla_view_counts_evaluated_checks_and_takes_the_p95():
    from loadlens_app.blueprints.dashboard import _runs_sla_view

    assert _runs_sla_view(CHECKS) == {"passed": 2, "total": 3, "p95_ms": 450.0, "p95_passed": False}
    # Checks without a result do not count; a p95 without a number is skipped for the next one.
    checks = [
        {"name": "target_rps", "actual": None, "passed": None},
        {"name": "p95_latency", "actual": True, "passed": True},
        {"name": "p95_latency", "actual": "312.5", "passed": True},
        "not a check",
    ]
    assert _runs_sla_view(checks) == {"passed": 2, "total": 2, "p95_ms": 312.5, "p95_passed": True}
    assert _runs_sla_view([{"name": "p95_latency", "actual": float("nan"), "passed": False}])["p95_ms"] is None
    assert _runs_sla_view([{"name": "target_rps", "passed": None}]) is None
    assert _runs_sla_view([]) is None and _runs_sla_view(None) is None and _runs_sla_view({"checks": CHECKS}) is None
    assert _runs_sla_view("not json") is None
    assert _runs_sla_view(json.dumps(CHECKS))["total"] == 3


# ---- archive markup -------------------------------------------------------------------------------------


def _main(html):
    return re.search(r'<main id="main".*?</main>', html, re.S).group(0)


def test_archive_markup_has_inline_filter_captions_sla_columns_and_a_text_pager(client):
    use_locale(client, "en")
    main = _main(client.get("/reports").get_data(as_text=True))
    assert 'class="container archive-page"' in main
    # The caption sits inside the field: «Service: All».
    assert re.search(r'<label class="inp archive-filter"><span class="archive-filter-key">Service:</span>\s*'
                     r'<select id="serviceFilter"><option value="">All</option></select>', main)
    assert '<kbd class="archive-kbd" aria-hidden="true">/</kbd>' in main
    assert 'id="archiveCounter"' in main
    assert 'id="searchBtn"' not in main and 'id="clearBtn"' not in main  # the filters apply as they change
    reports_table = re.search(r'<table[^>]*id="runsTable".*?</table>', main, re.S).group(0)
    heads = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", reports_table, re.S)]
    assert heads == ["Select for comparison", "Run", "Service", "Test type", "Test date", "Verdict", "SLA", "Max RPS",
                     "p95", "Report created", "Actions"]
    assert '<th class="n" title="SLA criteria of the final report: met of evaluated">SLA</th>' in main
    # The selection bar sits under the table, before the pager.
    assert main.index('id="selectionBar"') < main.index('id="archivePager"')
    assert '<button type="button" class="link selection-clear" id="selectionClear">Clear selection</button>' in main


def test_archive_markup_in_russian(client):
    main = _main(client.get("/reports").get_data(as_text=True))
    for key in ("сервис:", "вердикт:", "тип:"):
        assert f'<span class="archive-filter-key">{key}</span>' in main
    assert '<option value="">все</option>' in main
    assert 'aria-label="Страницы архива"' in main


# ---- archive.js helpers ---------------------------------------------------------------------------------


def archive(expression, **kwargs):
    return run_js(ARCHIVE_JS, f"const A = LL.archive._internals; {expression}", **kwargs)


def test_archive_sla_and_p95_cells():
    sla = {"passed": 2, "total": 3, "p95_ms": 1240.4, "p95_passed": False}
    assert archive(f"[A.slaText({json.dumps(sla)}), A.slaText(null), A.slaText({{ passed: 0, total: 0 }})]") == ["2/3", "—", "—"]
    texts = archive("[A.p95Text({ p95_ms: 1240.4 }), A.p95Text({ p95_ms: 7.25 }), A.p95Text({ p95_ms: null }), A.p95Text(null)]")
    assert texts == ["1 240", "7,3", "—", "—"]
    assert archive("A.p95Text({ p95_ms: 1240.4 })", locale="en") == "1,240"


def test_archive_short_dates_and_pages_follow_the_language():
    expr = (f"[A.shortDateTime('2026-10-14T09:05:00Z', {NOW}), A.shortDateTime('2025-12-31T23:59:00Z', {NOW}),"
            f" A.shortDateTime('2026-10-15T08:00:00Z', {NOW}, {{ timeToday: true }}), A.shortDateTime('x', {NOW}),"
            " A.pageText(2, 41), A.pageText(1, 0), A.pageCount(40), A.pageCount(41)]")
    assert archive(expr) == ["14.10 09:05", "31.12.2025 23:59", "08:00", "—", "страница 2 из 3", "страница 1 из 1", 2, 3]
    assert archive(expr, locale="en") == ["14/10 09:05", "31/12/2025 23:59", "08:00", "—", "page 2 of 3", "page 1 of 1", 2, 3]


def test_archive_asks_for_the_sla_summary():
    assert archive("A.buildRunsUrl(A.defaultState())").endswith("&include=sla")


# ---- forecasting pages -----------------------------------------------------------------------------------


def test_forecasting_pages_carry_their_scope_and_main_action(client):
    use_locale(client, "en")
    main = _main(client.get("/forecasting").get_data(as_text=True))
    assert 'class="container forecast-page forecast-list-page"' in main
    assert '<table class="data-table forecast-list">' in main
    main = _main(client.get("/forecasting/abc").get_data(as_text=True))
    assert 'class="container forecast-page forecast-report-page"' in main
    assert 'id="forecastConfluence" class="btn btn-primary"' in main


def test_forecasting_list_links_the_run_name():
    files = ["static/js/common.js", "static/js/ui.js", "static/js/forecasting.js"]
    expr = (
        "(() => { const body = document.createElement('tbody');"
        " document.getElementById = (id) => (id === 'forecastRows' ? body : null);"
        " LL.forecasting._internals.renderRows([{ run_name: 'r 1', service: 'api', test_type: 'step', created_at: null,"
        " verdict: 'pass', load_model: 'open', run_id: 'id/1' }]);"
        " const link = body.children[0].children[0].children[0];"
        " return [link.tagName, link.href, link.textContent]; })()"
    )
    assert run_js(files, expr) == ["A", "/forecasting/id%2F1", "r 1"]


# ---- stylesheet -----------------------------------------------------------------------------------------


def _stage3_css():
    css = (ROOT_DIR / "static" / "css" / "app.css").read_text(encoding="utf-8")
    start = css.index("/* ===== page: archive, compare, forecast ===== */")
    end = css.index("/* ===== end page: archive, compare, forecast ===== */")
    return css[start:end]


def _rule(css, selector):
    match = re.search(r"(?:^|\n)" + re.escape(selector) + r" \{([^}]*)\}", css)
    assert match, selector
    return match.group(1)


def test_stage3_rules_stay_on_their_pages():
    css = re.sub(r"/\*.*?\*/", "", _stage3_css(), flags=re.S)
    selectors = []
    for match in re.finditer(r"([^{}]+)\{[^{}]*\}", css):
        head = match.group(1).strip()
        if head.startswith("@media"):
            continue
        head = head.split("@media", 1)[-1].split(")", 1)[-1] if head.startswith("@") else head
        selectors.extend(s.strip() for s in head.split(","))
    assert len(selectors) > 100
    scopes = (".archive-", "table.archive-table", ".compare-page", ".forecast-page", ".forecast-report-page", "table.forecast-list")
    stray = [s for s in selectors if s and not s.startswith(scopes)]
    assert not stray, stray
    # Cards and pills are gone from the pickers, steps and selection bar of these pages.
    css = _stage3_css()
    assert ".compare-page .picker-card { min-width: 0; padding: 2px 0 2px 14px; border: 0;" in css
    step = _rule(css, ".forecast-report-page .forecast-step")
    assert "border-top: 2px solid var(--line-2)" in step and "border-radius" not in step and "background" not in step
    assert "border-top: 1px solid var(--line-2); border-radius: 0; background: var(--bg); box-shadow: none;" in css
    for selector in (".forecast-report-page .forecast-verdict", ".forecast-report-page .forecast-block", ".forecast-report-page .forecast-fact",
                     ".archive-page .filter-chip", ".compare-page .findings-col"):
        rule = _rule(css, selector)
        assert "border-radius: 999px" not in rule and "bg-elev" not in rule and "box-shadow" not in rule, selector
