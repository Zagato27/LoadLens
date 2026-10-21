# SPDX-License-Identifier: Apache-2.0
"""Flask side of the localization: request locale, Jinja helpers and the inline dictionary for the browser.

The engine (catalogs, ``t``, ``plural``, ``msg``, ``with_locale``) is ``AI/i18n.py``; this module re-exports it, so
server code has one import point. :func:`install` wires everything into an app (``app.py``).

The locale is read lazily (:func:`current_locale`), never in a ``before_request`` hook: the auth gate must stay the
first hook and renders ``storage_unavailable.html`` inside itself. Order: a valid ``loadlens.lang`` cookie (unless
``appearance.language_switch`` is false), then the ``appearance.language`` setting (:func:`report_language`), then
``ru``. ``Accept-Language`` is not read.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Iterable, Optional

from flask import Flask, Response, g, has_request_context, request
from markupsafe import Markup, escape

from AI import i18n as engine
from AI.i18n import (  # noqa: F401  (re-exported)
    DEFAULT,
    INTL,
    SUPPORTED,
    LocalizedError,
    Msg,
    fmt_date,
    fmt_number,
    get_locale,
    msg,
    normalize_lang,
    plural,
    render_ref,
    with_locale,
)

# Plain-text translation for server code (the Jinja ``t`` below returns Markup).
translate = engine.t

COOKIE = "loadlens.lang"
# Namespaces every page with JS receives (base.html and recovery.html); a page adds its own with
# {% set i18n_ns = ['dashboard', 'charts'] %}.
BASE_NAMESPACES = ("common", "ui", "glossary", "palette")
EXTENSION_KEY = "loadlens_i18n"

_LOCALE_ATTR = "locale"
_READ_ATTR = "i18n_locale_read"
_JS_ESCAPES = str.maketrans({
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    "'": "\\u0027",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
})


def report_language() -> str:
    """Language of new reports and prompts: ``CONFIG["appearance"]["language"]``, ``ru`` when unset or unknown."""
    from loadlens_app import core

    appearance = core.CONFIG.get("appearance")
    value = appearance.get("language") if isinstance(appearance, dict) else None
    return normalize_lang(value) or DEFAULT


def language_switch_enabled() -> bool:
    """``appearance.language_switch`` (default true); false ignores the cookie: everyone sees ``appearance.language``."""
    from loadlens_app import core

    appearance = core.CONFIG.get("appearance")
    return not (isinstance(appearance, dict) and appearance.get("language_switch") is False)


def _report_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _json_object(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def stored_report_language(obj: Any) -> str:
    """Language of a saved report: ``parsed.report_language``, then ``sla_details.report_language``, then ``ru``.

    ``obj`` is the parsed final JSON, a row (dict or object with ``parsed`` / ``sla_details``, JSON text accepted),
    or a list of rows (the ``final`` row is preferred). Reports saved before the field are Russian.
    """
    if isinstance(obj, (list, tuple)):
        rows = list(obj)
        rows.sort(key=lambda row: 0 if str(_report_field(row, "domain") or "") == "final" else 1)
        for row in rows:
            lang = _stored_language_or_none(row)
            if lang:
                return lang
        return DEFAULT
    return _stored_language_or_none(obj) or DEFAULT


def _stored_language_or_none(obj: Any) -> Optional[str]:
    obj = _json_object(obj)
    if obj is None:
        return None
    direct = normalize_lang(_report_field(obj, "report_language"))
    if direct:
        return direct
    for name in ("parsed", "final_parsed", "sla_details"):
        nested = _json_object(_report_field(obj, name))
        if nested is not None and nested is not obj:
            lang = normalize_lang(_report_field(nested, "report_language"))
            if lang:
                return lang
    return None


def current_locale() -> str:
    """UI language of the request, cached on ``g``; outside a request :func:`AI.i18n.get_locale`.

    Reading it marks the response, so :func:`install` adds ``Vary: Cookie``.
    """
    if not has_request_context():
        return get_locale()
    setattr(g, _READ_ATTR, True)
    cached = g.get(_LOCALE_ATTR)
    if cached:
        return cached
    chosen = normalize_lang(request.cookies.get(COOKIE)) if language_switch_enabled() else None
    lang = chosen or report_language()
    setattr(g, _LOCALE_ATTR, lang)
    return lang


def _resolve_locale() -> str:
    # Registered in the engine: a request follows its cookie, background code the installation setting.
    return current_locale() if has_request_context() else report_language()


# Importing this module is enough (Celery workers never call install()); install() registers it again.
engine.set_locale_resolver(_resolve_locale)


def _markup_param(value: Any, lang: str) -> str:
    if hasattr(value, "__html__"):
        return str(value.__html__())
    return str(escape(engine.param_text(value, lang)))


def t(key: str, /, **params: Any) -> Markup:
    """Jinja ``{{ t('ns.key', name=value) }}``: the message and the parameters are escaped, Markup passes as is."""
    lang = get_locale()
    raw = engine.lookup(key, lang)
    return Markup(engine.interpolate(str(escape(raw)), params, lambda value: _markup_param(value, lang)))


def tn(key: str, n: Any, /, **params: Any) -> Markup:
    """Jinja ``{{ tn('ns.key', count) }}``: plural form for ``n``; ``{n}`` is the count unless given."""
    lang = get_locale()
    params.setdefault("n", engine.number_text(n, lang))
    raw = engine.lookup_plural(key, n, lang)
    return Markup(engine.interpolate(str(escape(raw)), params, lambda value: _markup_param(value, lang)))


def script_json(value: Any) -> str:
    """JSON for an inline ``<script>``: Cyrillic as is, characters that could end the script or a JS string escaped."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).translate(_JS_ESCAPES)


def _namespaces(namespaces: Optional[Iterable[str]]) -> tuple[str, ...]:
    names = list(BASE_NAMESPACES)
    if isinstance(namespaces, str):
        namespaces = [namespaces]
    for name in namespaces or ():
        if isinstance(name, str) and engine.NAMESPACE_RE.match(name) and name not in names:
            names.append(name)
    return tuple(names)


@lru_cache(maxsize=256)
def _payload(lang: str, names: tuple[str, ...], base_serial: int, own_serial: int) -> str:
    # The serials key the cache on the loaded catalogs: a reload builds new fragments.
    data = {"locale": lang, "intl": INTL[lang], "messages": engine.client_messages(lang, names)}
    return script_json(data)


def i18n_script(namespaces: Optional[Iterable[str]] = ()) -> Markup:
    """``<script nonce>`` that sets ``window.LoadLens.i18n = {locale, intl, messages}`` for static/js/i18n.js.

    The base namespaces are always included; unknown names are skipped; en messages fall back to ru per key.
    """
    from loadlens_app.auth.web import csp_nonce

    lang = current_locale()
    names = _namespaces(namespaces)
    payload = _payload(lang, names, engine.catalog(DEFAULT).serial, engine.catalog(lang).serial)
    return Markup(
        f'<script nonce="{escape(csp_nonce())}">'
        f"window.LoadLens=window.LoadLens||{{}};window.LoadLens.i18n={payload};"
        "</script>"
    )


def _template_context() -> dict[str, str]:
    return {"locale": current_locale(), "report_language": report_language()}


def _vary_on_cookie(response: Response) -> Response:
    # The body depends on the language cookie only when the locale was read (pages, localized answers);
    # static files and the logo keep their long cache without Vary.
    if g.get(_READ_ATTR):
        response.vary.add("Cookie")
    return response


def install(app: Flask) -> None:
    """Template context (``locale``, ``report_language``), Jinja globals, the engine resolver and ``Vary: Cookie``."""
    engine.set_locale_resolver(_resolve_locale)
    app.context_processor(_template_context)
    app.add_template_global(t, "t")
    app.add_template_global(tn, "tn")
    app.add_template_global(i18n_script, "i18n_script")
    app.after_request(_vary_on_cookie)
    app.extensions[EXTENSION_KEY] = True
