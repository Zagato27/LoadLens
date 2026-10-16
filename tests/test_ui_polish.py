"""Stage 5 of docs/ui-plan.md outside ui.js: pre-paint scripts, the «Вид» menu markup, sprite icons,
stylesheet rules for density, reduced motion and mobile cards, and the light --text-dim contrast.

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


def stage5_css() -> str:
    css = APP_CSS.read_text(encoding="utf-8")
    match = re.search(r"/\* ===== ui-plan stage 5: polish and mobile ===== \*/(.*?)/\* ===== end stage 5 ===== \*/", css, re.S)
    assert match, "stage 5 section is missing in app.css"
    return match.group(1)


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
    # No or unknown value: the server-rendered data-theme (dark in the harness) stays.
    assert run_pre_paint(page, os_light=True)["theme"] == "dark"
    assert run_pre_paint(page, saved="purple", os_light=True)["theme"] == "dark"


@pytest.mark.parametrize("page", PRE_PAINT_PAGES)
def test_pre_paint_density(page):
    assert run_pre_paint(page, density=json.dumps("compact"))["density"] == "compact"
    assert run_pre_paint(page, density=json.dumps("normal"))["density"] is None
    assert run_pre_paint(page)["density"] is None


@pytest.mark.parametrize("page", PRE_PAINT_PAGES)
def test_pre_paint_survives_blocked_storage(page):
    assert run_pre_paint(page, storage_throws=True) == {"theme": "dark", "density": None, "listeners": 0}


@pytest.mark.parametrize("page", STANDALONE_PAGES)
def test_standalone_pages_follow_os_switch(page):
    result = run_pre_paint(page, saved="system", os_light=False, flip_to=True)
    assert result["theme"] == "light"
    assert result["listeners"] == 1
    # An explicit choice does not subscribe.
    assert run_pre_paint(page, saved="dark", flip_to=True)["listeners"] == 0


def test_base_pre_paint_leaves_live_switching_to_ui_js():
    # ui.js listens to the media query; a second listener here would apply the theme before ui.js
    # notices the change, so charts would never be restyled.
    assert run_pre_paint("base.html", saved="system", os_light=False, flip_to=True) == {"theme": "dark", "density": None, "listeners": 0}


# ---- templates --------------------------------------------------------------------------------

def test_templates_compile():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    for page in [*PRE_PAINT_PAGES, "compare.html"]:
        env.get_template(page)


def test_view_menu_markup():
    html = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    menu = re.search(r'<details class="menu view-menu" id="viewMenu">(.*?)</details>', html, re.S)
    assert menu, "the «Вид» menu is missing in the sidebar foot"
    body = menu.group(1)
    assert 'id="viewMenuValue"' in body
    assert 'role="menu"' in body
    items = re.findall(r'role="menuitemradio" aria-checked="false" data-view-pref="(\w+)" data-value="(\w+)"', body)
    assert items == [("theme", "system"), ("theme", "light"), ("theme", "dark"), ("density", "normal"), ("density", "compact")]
    for label in ("Как в системе", "Светлая", "Тёмная", "Обычная", "Компактная", "Плотность"):
        assert label in body
    # The palette trigger (stage 3) stays above the menu; the old toggle button is gone.
    assert html.index('id="paletteOpen"') < html.index('id="viewMenu"')
    assert 'id="themeToggle"' not in html


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
    for background in ("--bg", "--bg-elev"):
        ratio = _contrast(css["--text-dim"], css[background])
        assert ratio >= 4.5, f"{theme} --text-dim on {background}: {ratio:.2f}"
    if theme == "light":
        assert _contrast(css["--text-dim"], "#ffffff") >= 4.5


def test_stage5_css_rules():
    css = stage5_css()
    compact = css[css.index('[data-density="compact"]'):]
    assert re.search(r':root\[data-density="compact"\]\s*\{\s*--radius-sm:\s*6px;', css)
    assert re.search(r'\.data-table td\s*\{\s*padding:\s*6px', compact)
    assert "height: 32px" in compact
    motion = re.search(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", css, re.S)
    assert motion, "no prefers-reduced-motion block"
    for selector in (".skeleton::after", ".toast", ".app-sidebar"):
        assert selector in motion.group(1)
    cards = re.search(r"@media \(max-width: 720px\)\s*\{(.*?)\n\}", css, re.S)
    assert cards, "no mobile cards block"
    block = cards.group(1)
    assert ".table-wrap > table.data-table:not(.no-cards)" in block
    assert "content: attr(data-label)" in block
    assert re.search(r"> tr\s*\{[^}]*display:\s*grid", block)
    for cell in ('td[data-cell="actions"]', 'td[data-cell="check"]', 'td[data-cell="primary"]'):
        assert cell in block


def test_ui_js_has_no_inline_handlers_or_eval():
    source = (REPO_ROOT / "static" / "js" / "ui.js").read_text(encoding="utf-8")
    assert "eval(" not in source and "new Function" not in source
    assert not re.search(r"\son[a-z]+=", source)
