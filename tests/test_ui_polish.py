# SPDX-License-Identifier: Apache-2.0
"""Stage 5 of plans/ui-plan.md and plans/instrument-ui-plan.md outside ui.js: pre-paint scripts, the theme menu
markup, sprite icons, stylesheet rules for reduced motion and phone cards, the page sections and token names of
app.css, and the contrast of text and status colors in both themes. The density setting was removed on 2026-10-16;
the tests check that nothing of it is left.

The pre-paint scripts are cut out of the templates and run under Node (tests/js_harness.py); those
checks are skipped when node is absent.
"""

from __future__ import annotations

import json
import re

import pytest
from jinja2 import Environment, FileSystemLoader

from tests.js_harness import REPO_ROOT, run_js, theme_css_vars

TEMPLATES = REPO_ROOT / "templates"
PRE_PAINT_PAGES = ["base.html", "login.html", "recovery.html", "storage_unavailable.html"]
# Pages without ui.js keep following the OS scheme from the pre-paint script itself.
STANDALONE_PAGES = ["login.html", "recovery.html", "storage_unavailable.html"]
APP_CSS = REPO_ROOT / "static" / "css" / "app.css"


def pre_paint_script(page: str) -> str:
    html = (TEMPLATES / page).read_text(encoding="utf-8")
    match = re.search(r'<script nonce="\{\{ csp_nonce \}\}">(.*?)</script>', html, re.S)
    assert match, f"{page}: no nonce'd pre-paint script"
    return match.group(1)


def app_css() -> str:
    return APP_CSS.read_text(encoding="utf-8")


# Page sections of app.css (plans/instrument-ui-plan.md, stage 5): one per page group, in this order.
PAGE_SECTIONS = ["report", "dashboard", "archive, compare, forecast", "forms, settings, sign-in"]


def run_pre_paint(page: str, *, saved=None, density=None, os_light=False, flip_to=None, storage_throws=False):
    """Runs the page's pre-paint script with a fake prefers-color-scheme query; returns the attributes."""
    storage = {}
    if saved is not None:
        storage["loadlens.theme"] = saved
    if density is not None:
        storage["loadlens.density"] = density
    flip = "" if flip_to is None else f"q.matches = {json.dumps(flip_to)}; (q.listeners || []).forEach((cb) => cb({{ matches: q.matches }}));"
    expression = f"""
        const q = {{ matches: {json.dumps(os_light)}, media: '(prefers-color-scheme: light)', listeners: [],
                    addEventListener(type, cb) {{ if (type === 'change') q.listeners.push(cb); }} }};
        window.matchMedia = () => q;
        {pre_paint_script(page)}
        {flip}
        ({{ theme: document.documentElement.getAttribute('data-theme'),
            density: document.documentElement.getAttribute('data-density'),
            listeners: q.listeners.length }})
    """
    return run_js([], expression, local_storage=storage, storage_throws=storage_throws)


# ---- pre-paint scripts ------------------------------------------------------------------------

@pytest.mark.parametrize("page", PRE_PAINT_PAGES)
def test_pre_paint_resolves_system_theme(page):
    assert run_pre_paint(page, saved="system", os_light=True)["theme"] == "light"
    assert run_pre_paint(page, saved="system", os_light=False)["theme"] == "dark"


@pytest.mark.parametrize("page", PRE_PAINT_PAGES)
def test_pre_paint_explicit_and_missing_theme(page):
    assert run_pre_paint(page, saved="light", os_light=False)["theme"] == "light"
    assert run_pre_paint(page, saved="dark", os_light=True)["theme"] == "dark"
    # No or unknown value: "system" is the default, the OS setting decides (plans/instrument-ui-plan.md).
    assert run_pre_paint(page, os_light=True)["theme"] == "light"
    assert run_pre_paint(page, os_light=False)["theme"] == "dark"
    assert run_pre_paint(page, saved="purple", os_light=True)["theme"] == "light"


@pytest.mark.parametrize("page", PRE_PAINT_PAGES)
def test_pre_paint_ignores_a_leftover_density_value(page):
    # Browsers of people who picked the removed compact density still hold the value.
    assert run_pre_paint(page, density=json.dumps("compact"))["density"] is None
    assert "density" not in pre_paint_script(page)


@pytest.mark.parametrize("page", PRE_PAINT_PAGES)
def test_pre_paint_survives_blocked_storage(page):
    # Without storage there is no stored choice: the OS setting decides (and keeps deciding on standalone pages).
    following = 1 if page in STANDALONE_PAGES else 0
    assert run_pre_paint(page, storage_throws=True) == {"theme": "dark", "density": None, "listeners": following}
    assert run_pre_paint(page, storage_throws=True, os_light=True)["theme"] == "light"


@pytest.mark.parametrize("page", STANDALONE_PAGES)
def test_standalone_pages_follow_os_switch(page):
    result = run_pre_paint(page, saved="system", os_light=False, flip_to=True)
    assert result["theme"] == "light"
    assert result["listeners"] == 1
    # No stored choice means "system" too.
    assert run_pre_paint(page, os_light=False, flip_to=True) == {"theme": "light", "density": None, "listeners": 1}
    # An explicit choice does not subscribe.
    assert run_pre_paint(page, saved="dark", flip_to=True)["listeners"] == 0


def test_base_pre_paint_leaves_live_switching_to_ui_js():
    # ui.js listens to the media query; a second listener here would apply the theme before ui.js
    # notices the change, so charts would never be restyled.
    assert run_pre_paint("base.html", saved="system", os_light=False, flip_to=True) == {"theme": "dark", "density": None, "listeners": 0}
    assert run_pre_paint("base.html", os_light=False, flip_to=True) == {"theme": "dark", "density": None, "listeners": 0}


# ---- templates --------------------------------------------------------------------------------

def test_templates_compile():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    for page in [*PRE_PAINT_PAGES, "compare.html"]:
        env.get_template(page)


@pytest.fixture
def page_client(monkeypatch, tmp_path):
    """A test client of the app (authentication is off in tests): every page renders the base.html frame."""
    from loadlens_app import core
    import update_page
    from app import create_app

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def rendered_view_menu(client, lang=None):
    """(page html, theme menu markup) of the rendered dashboard; ``lang`` sets the language cookie."""
    if lang:
        client.set_cookie("loadlens.lang", lang)
    html = client.get("/").get_data(as_text=True)
    menu = re.search(r'<details class="menu view-menu" id="viewMenu">(.*?)</details>', html, re.S)
    assert menu, "the view menu is missing in the top bar"
    return html, menu.group(1)


def test_view_menu_markup(page_client):
    html, body = rendered_view_menu(page_client)
    assert 'id="viewMenuValue"' in body
    assert 'role="menu"' in body
    # Stage 5 of plans/i18n-plan.md: the menu is "View" with a theme group and a language group.
    assert '<span class="nav-label">Вид</span>' in body and 'title="Тема и язык интерфейса"' in body
    groups = re.findall(r'<div class="menu-group" role="group" aria-label="([^"]+)">', body)
    assert groups == ["Тема интерфейса", "Язык интерфейса"]
    items = re.findall(r'role="menuitemradio" aria-checked="(\w+)" data-view-pref="(\w+)" data-value="(\w+)"', body)
    assert items == [("false", "theme", "system"), ("false", "theme", "light"), ("false", "theme", "dark"),
                     ("true", "lang", "ru"), ("false", "lang", "en")]
    for label in ("Как в системе", "Светлая", "Тёмная", "Русский", "English"):
        assert f"<span>{label}</span>" in body
    assert 'data-value="en" lang="en"' in body and 'data-value="ru" lang="ru"' in body
    source = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    for removed in ("density", "Плотность", "Компактная", "i-rows"):
        assert removed not in source and removed not in body
    # The palette trigger (stage 3) stays above the menu; the old toggle button is gone.
    assert html.index('id="paletteOpen"') < html.index('id="viewMenu"')
    assert 'id="themeToggle"' not in html


def test_view_menu_markup_in_english(page_client):
    _, body = rendered_view_menu(page_client, "en")
    assert '<span class="nav-label">View</span>' in body and 'title="Interface theme and language"' in body
    items = re.findall(r'data-view-pref="theme" data-value="(\w+)">.*?<span>([^<]*)</span>', body)
    assert items == [("system", "System"), ("light", "Light"), ("dark", "Dark")]
    langs = re.findall(r'aria-checked="(\w+)" data-view-pref="lang" data-value="(\w+)"', body)
    assert langs == [("false", "ru"), ("true", "en")]
    # Language names stay in their own language: «Русский» is the only Cyrillic word of the menu.
    assert not re.search("[Ѐ-ӿ]", body.replace("Русский", ""))


def test_view_menu_hides_the_language_group_when_the_switch_is_off(page_client, monkeypatch):
    from loadlens_app import core

    monkeypatch.setitem(core.CONFIG, "appearance", {**core.CONFIG["appearance"], "language_switch": False})
    _, body = rendered_view_menu(page_client, "en")
    assert 'data-view-pref="lang"' not in body and 'data-view-pref="theme"' in body
    assert '<span class="nav-label">Вид</span>' in body  # the cookie is ignored: the global language (ru) is shown


@pytest.mark.parametrize("page", STANDALONE_PAGES)
def test_standalone_pages_offer_language_links(page):
    source = (TEMPLATES / page).read_text(encoding="utf-8")
    assert '{% include "_lang_switch.html" %}' in source
    assert '<html lang="{{ locale }}"' in source


def test_icons_used_by_stage5_exist_in_sprite():
    html = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    sprite = set(re.findall(r'<symbol id="i-([\w-]+)"', html))
    used = set(re.findall(r'<use href="#i-([\w-]+)"/>', html.split('class="icon-sprite"', 1)[1]))
    for name in ("compare.js", "forecasting.js"):
        source = (REPO_ROOT / "static" / "js" / name).read_text(encoding="utf-8")
        used |= set(re.findall(r"icon: '([\w-]+)'", source))
        used |= set(re.findall(r"ui\.icon\('([\w-]+)'", source))
    missing = sorted(used - sprite)
    assert not missing, f"icons missing in the sprite: {missing}"
    assert {"contrast", "monitor", "check", "columns", "trending"} <= used


# ---- stylesheet -------------------------------------------------------------------------------

def _luminance(color: str) -> float:
    value = color.strip().lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_text_dim_meets_aa_on_backgrounds(theme):
    css = theme_css_vars()[theme]
    for background in ("--bg", "--bg-2"):
        ratio = _contrast(css["--dim"], css[background])
        assert ratio >= 4.5, f"{theme} --dim on {background}: {ratio:.2f}"
    # --text-dim is the name AI/report_markers.py and the contract tests read; it is --dim.
    assert css["--text-dim"] == css["--dim"]
    if theme == "light":
        assert _contrast(css["--dim"], css["--bg-pop"]) >= 4.5


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_text_and_status_colors_meet_aa_on_the_page(theme):
    # Status is a colored word (plans/instrument-ui-plan.md 2.1): every tone must read as text on the page and on a
    # hovered row, in both themes ("paper" included).
    css = theme_css_vars()[theme]
    for token in ("--text", "--muted", "--dim", "--ok", "--warn", "--fail", "--na", "--run"):
        for background in ("--bg", "--bg-2"):
            ratio = _contrast(css[token], css[background])
            assert ratio >= 4.5, f"{theme} {token} on {background}: {ratio:.2f}"
    assert [css[f"--status-{tone}"] for tone in ("ok", "warn", "fail")] == [css["--ok"], css["--warn"], css["--fail"]]


def test_css_has_page_sections_and_no_retired_tokens():
    css = app_css()
    positions = [css.index(f"/* ===== page: {name} ===== */") for name in PAGE_SECTIONS]
    assert positions == sorted(positions)
    for name in PAGE_SECTIONS:
        assert css.count(f"/* ===== page: {name} ===== */") == 1 and css.count(f"/* ===== end page: {name} ===== */") == 1
    # The stage markers of the redesign are folded into the page sections.
    assert "ui-plan stage" not in css and "instrument stage" not in css
    # Pre-redesign token names are gone: styles and scripts read the instrument tokens directly.
    retired = ["--bg-elev", "--bg-elev2", "--bg-sidebar", "--bg-chart", "--bg-input", "--bg-hover", "--bg-code", "--border",
               "--border-soft", "--border-strong", "--link", "--danger", "--text-muted", "--chart-grid", "--chart-text",
               "--shadow-sm", "--status-na"]
    for tone in ("ok", "warn", "fail", "na", "run", "info"):
        retired += [f"--{tone}-fg", f"--{tone}-bg", f"--{tone}-border"]
    for name in retired:
        assert not re.search(re.escape(name) + r"(?![\w-])", css), name
    for js in (REPO_ROOT / "static" / "js").glob("*.js"):
        source = js.read_text(encoding="utf-8")
        for name in retired + ["--text-dim", "--status-ok", "--status-warn", "--status-fail"]:
            assert not re.search(re.escape(name) + r"(?![\w-])", source), f"{js.name}: {name}"
    # Cards of the old look: no 8/10/12px radii and no pill radius left.
    assert not re.search(r"border-radius:\s*(8|10|12|999)px", css)


def test_reduced_motion_and_phone_cards():
    css = app_css()
    assert "data-density" not in css
    motion = re.search(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", css, re.S)
    assert motion, "no prefers-reduced-motion block"
    for selector in (".skeleton::after", ".toast", ".topbar-drawer", ".is-highlighted"):
        assert selector in motion.group(1)
    cards = re.search(r"/\* Phone cards:.*?@media \(max-width: 720px\)\s*\{(.*?)\n\}", css, re.S)
    assert cards, "no phone cards block"
    block = cards.group(1)
    assert ".table-wrap > table.data-table:not(.no-cards)" in block
    assert "content: attr(data-label)" in block
    assert re.search(r"> tr\s*\{[^}]*display:\s*grid", block)
    for cell in ('td[data-cell="actions"]', 'td[data-cell="check"]', 'td[data-cell="primary"]'):
        assert cell in block
    # A row is a block under a hairline, not a boxed card.
    row = re.search(r"> tbody > tr \{([^}]*)\}", block).group(1)
    assert "border-bottom: 1px solid var(--line)" in row and "border-radius: 0" in row and "background: none" in row
    # The column title floats in its own column, so a title on two lines makes the cell taller instead of running
    # into the next cell (an absolutely placed title did).
    label = re.search(r"td\[data-label\]::before \{([^}]*)\}", block).group(1)
    assert "float: left" in label and "position: absolute" not in label
    assert re.search(r"td\[data-label\] \{[^}]*display: flow-root", block)
    # Sortable titles above the rows are text, not pills.
    chips = re.search(r"th\[data-sort\] \{([^}]*)\}", block).group(1)
    assert "border: 0" in chips and "background: none" in chips


def test_ui_js_has_no_inline_handlers_or_eval():
    source = (REPO_ROOT / "static" / "js" / "ui.js").read_text(encoding="utf-8")
    assert "eval(" not in source and "new Function" not in source
    assert not re.search(r"\son[a-z]+=", source)
