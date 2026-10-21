# SPDX-License-Identifier: Apache-2.0
"""Stage 4 of the instrument UI (forms, launch, settings, sign-in): markup and styles that carry the new look.

Form sections without boxes, test types as radio rows, test window chips without the accent, instances with a
rule on the left, the run page with figures, text charts and table.grid, the settings menu and wizard as text
with the active item underlined, the readiness checklist on squares, the sign-in card as the only card.
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
from tests.js_harness import run_js  # noqa: E402

BASE = ["static/js/common.js", "static/js/ui.js"]
RUN_PAGE = BASE + ["static/js/launch_run.js"]
PROJECTS = BASE + ["static/js/projects.js"]
STAGE_START = "/* ===== page: forms, settings, sign-in ===== */"
STAGE_END = "/* ===== end page: forms, settings, sign-in ===== */"
NAMESPACES = ("new", "launch", "run", "settings", "projects", "users", "account", "auth", "recovery")


def _read(rel: str) -> str:
    return (ROOT_DIR / rel).read_text(encoding="utf-8")


def _stage_css() -> str:
    css = _read("static/css/app.css")
    assert css.count(STAGE_START) == 1 and css.count(STAGE_END) == 1
    return css[css.index(STAGE_START):css.index(STAGE_END)]


def _rule(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, selector
    return match.group(1)


def _page_style(html: str) -> str:
    return "\n".join(re.findall(r"<style>(.*?)</style>", html, re.S))


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


# ---- shared styles (the forms, settings and sign-in section of app.css) ---------------------------------------


def test_form_sections_lose_their_boxes():
    css = _stage_css()
    section = _rule(css, ".form-section")
    assert "border: 0;" in section and "border-top: 1px solid var(--line)" in section and "background: none" in section
    assert "font-size: 15px" in _rule(css, ".form-section > legend")
    # From 1024px the heading moves into a column on the left of the fields.
    assert re.search(r"@media \(min-width: 1024px\) \{\s*\.form-section \{ padding-left: 252px; \}", css)


def test_window_chips_settings_menu_and_lists_have_no_accent_or_fill():
    css = _stage_css()
    chip = _rule(css, ".lt-chip")
    assert "background: none" in chip and "border: 0" in chip and "var(--mono)" in chip
    active = _rule(css, ".lt-chip.active, .lt-chip[aria-pressed=\"true\"]")
    assert "accent" not in active and "text-decoration-thickness: 2px" in active
    menu = _rule(css, ".settings-menu button.active")
    assert "text-decoration: underline" in menu and "background" not in menu
    assert "background: none" in _rule(css, ".settings-menu button")
    for selector in (".conn-card", ".q-item"):
        rule = _rule(css, selector)
        assert "border-radius" not in rule and "background" not in rule and "border: 1px" not in rule, selector
    assert "border-bottom: 2px solid transparent" in _rule(css, ".stepper .step-item")
    # The accent stays a sample of itself (appearance section); nothing else in the stage paints with it.
    accent_rules = [rule for rule in re.findall(r"([^{}]+)\{([^}]*)\}", css) if "var(--accent" in rule[1]]
    assert [selector.strip() for selector, _ in accent_rules] == [".btn.accent-sample, .btn.accent-sample:hover"]


def test_readiness_squares_and_sign_in_card():
    css = _stage_css()
    assert "background: var(--ok)" in _rule(css, ".readiness-squares > i.ok")
    assert "justify-content: flex-start" in _rule(css, ".auth-brand")
    assert "font-size: 20px" in _rule(css, ".auth-brand .brand")


# ---- new report and launch forms --------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["/new", "/launch"])
def test_form_pages_use_the_form_layout(client, url):
    html = client.get(url).get_data(as_text=True)
    assert '<main id="main" class="container form-page">' in html and "container-narrow" not in html
    assert re.search(r"<h1>[^<]+</h1>\s*<p class=\"form-intro\">", html)
    assert html.count('<fieldset class="form-section">') >= 4
    style = _page_style(html)
    assert "var(--accent" not in style and "border-radius: 10px" not in style and "border-radius: 12px" not in style
    assert "btn-lg" not in html.split('<form', 1)[1].split("</form>", 1)[0]


def test_new_report_type_rows_and_result_links(client):
    html = client.get("/new").get_data(as_text=True)
    style = _page_style(html)
    assert ".type-card::before" in style and ".type-card:has(input:checked)::before" in style
    assert html.count('<input type="radio" name="test_type"') == 4
    assert '<a id="webLink" class="reportLink btn btn-primary" href="#">' in html
    assert ".phase-list li.done::before" in style and "var(--ok)" in style
    script = _read("static/js/new.js")
    assert "'inline-block'" not in script and script.count("style.display = 'inline-flex'") == 3


def test_launch_instances_have_a_rule_on_the_left(client):
    style = _page_style(client.get("/launch").get_data(as_text=True))
    card = _rule(style, ".instance-card")
    assert "border-left: 2px solid var(--line-2)" in card and "border: 1px" not in card
    pre = _rule(style, ".instance-preview pre, .preview-card pre")
    assert "border: 0" in pre and "background: none" in pre and "var(--mono)" in pre


# ---- run page ----------------------------------------------------------------------------------------------


def test_run_page_figures_text_charts_and_grid_table(client):
    html = client.get("/launch/abc").get_data(as_text=True)
    assert '<div class="figs sm run-figs" id="tiles">' in html and 'class="tile"' not in html
    for tile in ("tileRps", "tileErr", "tileP95", "tileVus"):
        assert f'<div class="v" id="{tile}">' in html
    assert '<table class="grid data-table inst-table">' in html and "report-table-shell" not in html
    assert 'class="live-window seg-group"' in html and html.count('class="seg-btn') == 4
    style = _page_style(html)
    assert "border-top: 1px solid var(--line)" in _rule(style, ".chart-card") and "var(--accent" not in style


def test_run_page_canvas_reads_the_mono_font_and_theme_tones():
    script = _read("static/js/launch_run.js")
    assert "'11px sans-serif'" not in script and "ctx.font = `11px ${ui.fonts.mono}`;" in script
    assert "colors.accent" not in script and "INSTANCE_COLORS" not in script and "#e03131" not in script
    lines = run_js(RUN_PAGE, "LL.launchRun._internals.slaLines({target_rps: 250, max_error_rate_pct: 1, max_p95_ms: 800})")
    assert [line["tone"] for line in lines["rps"]] == ["ok", "fail"] and lines["p95"][0]["tone"] == "fail"
    assert all("color" not in line for line in lines["rps"] + lines["p95"])


# ---- settings ----------------------------------------------------------------------------------------------


def test_readiness_meter_is_a_row_of_squares():
    full = {"target_rps": 250, "performance_query": "RPS", "has_system_context": True, "query_counts": {"jvm": 3},
            "missing_sla": {"forecast": [], "resource_plan": []}, "query_packs": []}
    complete = run_js(PROJECTS, f"LL.projects._internals.readinessList({{id: 'p', readiness: {json.dumps(full)}}})")
    assert complete.count('<i class="ok"></i>') == 6 and "<i></i>" not in complete and "<svg" not in complete
    empty = run_js(PROJECTS, "LL.projects._internals.readinessList({id: 'p', readiness: {}})")
    assert empty.count('<i class="ok"></i>') == 2 and empty.count("<i></i>") == 4
    assert 'aria-valuenow="2"' in empty and empty.count('class="sq warn"') == 4


def test_settings_page_heading_area_bar_and_style(client):
    html = client.get("/settings").get_data(as_text=True)
    assert "<h1 style=\"margin:0\">" in html and 'id="areaBar"' in html and 'id="wizardStepper"' in html
    style = _page_style(html)
    assert "var(--accent" not in style and "border-radius: 20px" not in style and "border-radius: 10px" not in style
    assert ".area-bar.is-unknown" in style


def test_wizard_steps_and_unknown_area_in_settings_js():
    script = _read("static/js/settings.js")
    assert '<span class="step-num" aria-hidden="true">' in script and "' aria-current=\"step\"'" in script
    # An area the settings do not know (a stale cookie) gets a warning instead of a 404 from the pack catalog.
    assert "if (areaKnown()) loadPackCatalog()" in script and "LL.t('settings.area.unknown_note', { area: state.area })" in script


def test_area_strings_point_at_the_top_bar():
    for lang in ("ru", "en"):
        settings = json.loads(_read(f"locales/{lang}/settings.json"))
        assert "{area}" in settings["settings.area.unknown_note"]
        for ns in NAMESPACES:
            text = _read(f"locales/{lang}/{ns}.json").lower()
            for phrase in ("боков", "side menu", "side panel", "sidebar"):
                assert phrase not in text, (lang, ns, phrase)


def test_dates_in_user_and_token_tables_are_mono():
    for rel in ("static/js/users.js", "static/js/account.js"):
        script = _read(rel)
        assert '<td class="nowrap">' not in script and '<td class="nowrap m">' in script


# ---- sign-in and error pages ---------------------------------------------------------------------------------


def test_forbidden_page_has_no_card():
    html = _read("templates/forbidden.html")
    assert 'class="card"' not in html and '<section class="forbidden-page">' in html and "<h1>" in html


def test_sign_in_card_is_the_only_card():
    assert _read("templates/login.html").count('class="auth-card"') == 1
    assert 'class="auth-card' not in _read("templates/recovery.html")
