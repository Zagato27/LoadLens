# SPDX-License-Identifier: Apache-2.0
"""Run pure front-end functions under Node without a browser or a build step.

``run_js(files, expression)`` loads repository JS files into one ``vm`` context with small browser
stubs and returns the JSON-decoded value of ``expression``:

    from tests.js_harness import run_js
    run_js(["static/js/common.js"], "LL.formatDuration(65000)")  # -> "1 мин"

- ``window`` is the context global; ``LL`` is an alias of ``window.LoadLens`` after the files load.
- ``expression`` is evaluated as a script, so several statements are allowed and the value of the
  last one is returned (``LL.prefs.set('a', 1); LL.prefs.get('a')``). Wrap object literals in
  parentheses. A returned promise is awaited. Map and Set values are converted to an object and an
  array; ``undefined`` comes back as ``None``.
- ``getComputedStyle`` reads the CSS variables of ``static/css/app.css`` for the theme in
  ``document.documentElement[data-theme]`` (``theme`` sets the initial value, and the
  ``prefers-color-scheme: light`` query matches when it is ``light``); ``css_vars`` overrides them:
  ``{"dark": {"--series-1": "#000000"}}``. ``var(--name)`` aliases come resolved.
- ``local_storage`` seeds ``localStorage``; ``storage_throws=True`` makes every storage call throw.
- ``dom_ready=True`` fires the ``DOMContentLoaded`` listeners after loading.
- ``window.__harness`` offers ``setCssVar(theme, name, value)``, ``fireWindow(type, props)`` (for
  example a cross-tab ``storage`` event), ``storage`` (the backing Map), ``registered`` (plugins
  passed to ``Chart.register``), ``cookieWrites`` (every string assigned to ``document.cookie``),
  ``cookies`` (name -> value, what ``document.cookie`` returns) and ``reloads()`` (calls of
  ``location.reload``). ``cookies`` seeds ``document.cookie``.
- ``Chart`` is a fake with ``defaults`` (color, borderColor, font, plugins.tooltip, plugins.legend),
  ``instances`` and ``register``; pass ``with_chart=False`` to leave it undefined.

- Localization: ``static/js/i18n.js`` is always loaded first (also for ``run_js([])``) with
  ``LoadLens.i18n = {locale, intl, messages, strict}`` holding every namespace of ``locales/<locale>/``
  (``locale="ru"`` by default; en values fall back to ru per key, like the server). ``messages`` adds or
  replaces keys; strict mode is on, so ``LL.t`` of a missing key throws (``i18n_strict=False`` warns instead).
  ``document.documentElement`` carries ``lang``.

The test is skipped when ``node`` is not on PATH, and fails instead with ``LOADLENS_REQUIRE_NODE=1`` (CI).
The time zone defaults to UTC (``tz``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_CSS = REPO_ROOT / "static" / "css" / "app.css"
I18N_JS = "static/js/i18n.js"
RESULT_MARK = "__LL_RESULT__"


class JsHarnessError(AssertionError):
    """The JS code threw, or node could not run the harness."""


def _block(css: str, selector_pattern: str) -> str:
    match = re.search(selector_pattern + r"\s*\{(.*?)\n\}", css, re.S)
    return match.group(1) if match else ""


def _declarations(block: str) -> dict[str, str]:
    return {name: value.strip() for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", block)}


def i18n_payload(locale: str = "ru", messages: Mapping[str, str] | None = None, *, strict: bool = True) -> dict[str, Any]:
    """``LoadLens.i18n`` as the inline script of base.html sets it, with every namespace of ``locales/<locale>/``.

    en values fall back to ru key by key, like the server; ``messages`` adds or replaces keys.
    """
    from AI import i18n as engine

    code = engine.normalize_lang(locale)
    if code is None:
        raise JsHarnessError(f"unsupported locale: {locale!r}")
    data = engine.client_messages(code, None)
    data.update(messages or {})
    return {"locale": code, "intl": engine.INTL[code], "messages": data, "strict": bool(strict)}


_VAR_REF = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^()]*))?\)")


def _resolve(values: dict[str, str]) -> dict[str, str]:
    """Substitutes ``var(--name)`` references (the aliases of app.css) the way the browser computes them."""
    out = dict(values)
    for _ in range(8):  # aliases are at most a few levels deep
        changed = False
        for name, value in out.items():
            resolved = _VAR_REF.sub(lambda m: out.get(m.group(1), (m.group(2) or "").strip()), value)
            if resolved != value:
                out[name] = resolved
                changed = True
        if not changed:
            break
    return out


@lru_cache(maxsize=1)
def theme_css_vars() -> dict[str, dict[str, str]]:
    """CSS custom properties of both themes from app.css (light inherits what it does not override).

    ``var(--name)`` aliases are resolved per theme, as ``getComputedStyle`` reports them in a browser.
    """
    css = APP_CSS.read_text(encoding="utf-8")
    dark = _declarations(_block(css, r':root,\s*:root\[data-theme="dark"\]'))
    light = {**dark, **_declarations(_block(css, r':root\[data-theme="light"\]'))}
    return {"dark": _resolve(dark), "light": _resolve(light)}


_RUNNER = r"""
const vm = require('vm');
const fs = require('fs');
const P = __PAYLOAD__;

function storageFor(init, throws) {
  const map = new Map(Object.entries(init || {}).map(([k, v]) => [String(k), String(v)]));
  const guard = (fn) => (...args) => { if (throws) throw new Error('storage disabled'); return fn(...args); };
  const api = {
    getItem: guard((k) => (map.has(String(k)) ? map.get(String(k)) : null)),
    setItem: guard((k, v) => { map.set(String(k), String(v)); }),
    removeItem: guard((k) => { map.delete(String(k)); }),
    clear: guard(() => map.clear()),
    key: guard((i) => (Array.from(map.keys())[i] === undefined ? null : Array.from(map.keys())[i])),
  };
  Object.defineProperty(api, 'length', { get: () => map.size });
  return { api, map };
}

function emitter() {
  const listeners = {};
  return {
    addEventListener(type, cb) { (listeners[type] = listeners[type] || []).push(cb); },
    removeEventListener(type, cb) {
      const list = listeners[type] || [];
      const i = list.indexOf(cb);
      if (i >= 0) list.splice(i, 1);
    },
    dispatchEvent(evt) { (listeners[evt.type] || []).slice().forEach((cb) => cb(evt)); return true; },
  };
}

function makeElement(tag) {
  const attrs = new Map();
  const classes = new Set();
  const el = Object.assign(emitter(), {
    tagName: String(tag || 'div').toUpperCase(),
    nodeType: 1,
    children: [],
    parentNode: null,
    dataset: {},
    hidden: false,
    textContent: '',
    innerHTML: '',
    value: '',
    style: { setProperty(k, v) { this[k] = String(v); }, removeProperty(k) { delete this[k]; }, getPropertyValue(k) { return this[k] || ''; } },
    classList: {
      add: (...c) => c.forEach((x) => classes.add(x)),
      remove: (...c) => c.forEach((x) => classes.delete(x)),
      toggle: (c, force) => { const on = force === undefined ? !classes.has(c) : !!force; if (on) classes.add(c); else classes.delete(c); return on; },
      contains: (c) => classes.has(c),
    },
    setAttribute(k, v) { attrs.set(String(k), String(v)); },
    getAttribute(k) { return attrs.has(String(k)) ? attrs.get(String(k)) : null; },
    hasAttribute(k) { return attrs.has(String(k)); },
    removeAttribute(k) { attrs.delete(String(k)); },
    appendChild(child) { el.children.push(child); if (child && typeof child === 'object') child.parentNode = el; return child; },
    prepend(child) { el.children.unshift(child); if (child && typeof child === 'object') child.parentNode = el; },
    remove() {
      const parent = el.parentNode;
      if (parent) { const i = parent.children.indexOf(el); if (i >= 0) parent.children.splice(i, 1); }
      el.parentNode = null;
    },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    closest() { return null; },
    contains(node) { return node === el; },
    focus() {}, blur() {}, click() {},
    getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; },
    getContext() { return null; },
  });
  Object.defineProperty(el, 'className', {
    get: () => Array.from(classes).join(' '),
    set: (v) => { classes.clear(); String(v || '').split(/\s+/).filter(Boolean).forEach((c) => classes.add(c)); },
  });
  return el;
}

const cssVars = { dark: Object.assign({}, P.cssVars.dark), light: Object.assign({}, P.cssVars.light) };
const documentElement = makeElement('html');
documentElement.setAttribute('data-theme', P.theme === 'light' ? 'light' : 'dark');
const document = Object.assign(emitter(), {
  documentElement,
  head: makeElement('head'),
  body: makeElement('body'),
  readyState: 'loading',
  getElementById() { return null; },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  createElement: makeElement,
  createTextNode: (text) => ({ nodeType: 3, textContent: String(text) }),
});
// document.cookie: reading returns "name=value; ..."; writing records the whole string and keeps the pair
// (Max-Age=0 or a past Expires removes it). Attributes are not interpreted otherwise.
const cookieJar = new Map(Object.entries(P.cookies || {}).map(([k, v]) => [String(k), String(v)]));
const cookieWrites = [];
Object.defineProperty(document, 'cookie', {
  configurable: true,
  get() { return Array.from(cookieJar.entries()).map(([k, v]) => `${k}=${v}`).join('; '); },
  set(raw) {
    const text = String(raw);
    cookieWrites.push(text);
    const [pair, ...attrs] = text.split(';');
    const eq = pair.indexOf('=');
    if (eq <= 0) return;
    const name = pair.slice(0, eq).trim();
    const removed = attrs.some((a) => /^\s*max-age\s*=\s*0\s*$/i.test(a));
    if (removed) cookieJar.delete(name); else cookieJar.set(name, pair.slice(eq + 1).trim());
  },
});
let reloadCount = 0;
const local = storageFor(P.localStorage, P.storageThrows);
const session = storageFor({}, P.storageThrows);

function getComputedStyle() {
  const theme = documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  const vars = cssVars[theme];
  return { getPropertyValue: (name) => (Object.prototype.hasOwnProperty.call(vars, name) ? ` ${vars[name]}` : '') };
}

function FakeChart() {}
FakeChart.defaults = { color: '', borderColor: '', font: { family: '' }, plugins: { tooltip: {}, legend: { labels: {} } } };
FakeChart.instances = {};
const registered = [];
FakeChart.register = (...items) => { registered.push(...items); };

const winEvents = emitter();
const sandbox = {
  document,
  localStorage: local.api,
  sessionStorage: session.api,
  navigator: { userAgent: 'node-harness', language: 'ru-RU', clipboard: { writeText: async () => {} } },
  location: {
    href: 'http://127.0.0.1/', origin: 'http://127.0.0.1', protocol: 'http:', host: '127.0.0.1',
    pathname: '/', search: '', hash: '', assign() {}, replace() {}, reload() { reloadCount += 1; },
  },
  history: { replaceState() {}, pushState() {} },
  getComputedStyle,
  // The OS color scheme matches the initial theme: a page rendered light without a stored choice has a light OS.
  matchMedia: (query) => ({ matches: /prefers-color-scheme:\s*light/.test(String(query)) ? P.theme === 'light' : false, media: String(query), addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }),
  requestAnimationFrame: (cb) => setTimeout(() => cb(Date.now()), 0),
  cancelAnimationFrame: (id) => clearTimeout(id),
  fetch: async () => ({ ok: true, status: 200, headers: new Headers(), json: async () => ({}), text: async () => '', clone() { return this; } }),
  addEventListener: winEvents.addEventListener,
  removeEventListener: winEvents.removeEventListener,
  dispatchEvent: winEvents.dispatchEvent,
  devicePixelRatio: 1,
  console, setTimeout, clearTimeout, setInterval, clearInterval, queueMicrotask, structuredClone,
  URL, URLSearchParams, Headers, TextEncoder, TextDecoder,
  Event: class Event { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } },
};
if (P.withChart) sandbox.Chart = FakeChart;
sandbox.window = sandbox;
sandbox.self = sandbox;
sandbox.__harness = {
  setCssVar(theme, name, value) { cssVars[theme === 'light' ? 'light' : 'dark'][name] = String(value); },
  fireWindow(type, props) { winEvents.dispatchEvent(Object.assign({ type }, props || {})); },
  storage: local.map,
  registered,
  cookieWrites,
  cookies: cookieJar,
  reloads: () => reloadCount,
};

function toPlain(key, value) {
  const tag = Object.prototype.toString.call(value);
  if (tag === '[object Map]') return Object.fromEntries(value);
  if (tag === '[object Set]') return Array.from(value);
  return value;
}
function finish(result) {
  let text;
  try { text = JSON.stringify(result, toPlain); } catch (e) { text = JSON.stringify({ ok: false, error: 'result is not JSON-serializable: ' + e.message }); }
  process.stdout.write('\n' + P.mark + text + '\n', () => process.exit(0));
}

// What the inline dictionary script of base.html does, plus the server-rendered <html lang>.
if (P.i18n) {
  sandbox.LoadLens = { i18n: P.i18n };
  documentElement.setAttribute('lang', P.i18n.locale);
}

(async () => {
  try {
    const context = vm.createContext(sandbox);
    for (const file of P.files) {
      vm.runInContext(fs.readFileSync(file, 'utf8'), context, { filename: file });
    }
    if (sandbox.LoadLens && !Object.prototype.hasOwnProperty.call(sandbox, 'LL')) sandbox.LL = sandbox.LoadLens;
    if (P.domReady) {
      document.readyState = 'interactive';
      document.dispatchEvent({ type: 'DOMContentLoaded' });
      document.readyState = 'complete';
    }
    let value = vm.runInContext(P.expression, context, { filename: 'expression.js' });
    if (value && typeof value.then === 'function') value = await value;
    finish({ ok: true, value: value === undefined ? null : value });
  } catch (e) {
    finish({ ok: false, error: String((e && e.stack) || e) });
  }
})();
"""


def run_js(
    files: Iterable[str],
    expression: str,
    *,
    theme: str = "dark",
    css_vars: Mapping[str, Mapping[str, str]] | None = None,
    local_storage: Mapping[str, str] | None = None,
    storage_throws: bool = False,
    dom_ready: bool = False,
    with_chart: bool = True,
    tz: str = "UTC",
    timeout: float = 60,
    locale: str = "ru",
    messages: Mapping[str, str] | None = None,
    i18n_strict: bool = True,
    cookies: Mapping[str, str] | None = None,
) -> Any:
    """Load ``files`` (paths relative to the repository root) and return the value of ``expression``."""
    node = shutil.which("node")
    if not node:
        if os.getenv("LOADLENS_REQUIRE_NODE", "").strip() == "1":
            pytest.fail("node is not installed (LOADLENS_REQUIRE_NODE=1)")
        pytest.skip("node is not installed")
    i18n = i18n_payload(locale, messages, strict=i18n_strict)
    paths = [str((REPO_ROOT / I18N_JS).resolve())]
    for name in files:
        path = (REPO_ROOT / name).resolve()
        if not path.is_file():
            raise JsHarnessError(f"no such file: {name}")
        if str(path) not in paths:  # i18n.js is always loaded first, once
            paths.append(str(path))
    base = theme_css_vars()
    merged = {theme_name: dict(base[theme_name]) for theme_name in ("dark", "light")}
    for theme_name, overrides in (css_vars or {}).items():
        merged.setdefault(theme_name, {}).update(overrides)
    payload = {
        "files": paths,
        "expression": expression,
        "theme": theme,
        "cssVars": merged,
        "localStorage": dict(local_storage or {}),
        "storageThrows": storage_throws,
        "domReady": dom_ready,
        "withChart": with_chart,
        "i18n": i18n,
        "cookies": dict(cookies or {}),
        "mark": RESULT_MARK,
    }
    script = _RUNNER.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    env = {**os.environ, "TZ": tz}
    proc = subprocess.run(
        [node, "-"],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        env=env,
        cwd=str(REPO_ROOT),
    )
    marked = [line for line in proc.stdout.splitlines() if line.startswith(RESULT_MARK)]
    if not marked:
        raise JsHarnessError(f"node exited with {proc.returncode} without a result: {proc.stderr.strip() or proc.stdout.strip()}")
    result = json.loads(marked[-1][len(RESULT_MARK):])
    if not result.get("ok"):
        raise JsHarnessError(result.get("error") or "JS error")
    return result.get("value")
