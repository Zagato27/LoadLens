# SPDX-License-Identifier: Apache-2.0
"""static/js/i18n.js under Node (tests/js_harness.py): LL.t, LL.tn, LL.tHtml, LL.fmt in ru and en, and the plural
rule of the Python engine (AI/i18n.py) against Intl.PluralRules. The harness loads i18n.js first with every
namespace of locales/<locale>/ and strict mode on."""

from __future__ import annotations

import json

import pytest

from AI import i18n
from tests.js_harness import REPO_ROOT, JsHarnessError, run_js

PODS_RU = {"test.pods.one": "{n} под", "test.pods.few": "{n} пода", "test.pods.many": "{n} подов"}
PODS_EN = {"test.pods.one": "{n} pod", "test.pods.other": "{n} pods"}
NBSP = chr(0xA0)
NUMBERS = list(range(0, 201)) + [0.5, 1.5, 2.5, 4.75, 5.5, 11.5, 21.5, 100.1]


def test_t_interpolates_in_both_locales():
    expression = """[
        LL.locale, LL.intl, document.documentElement.getAttribute('lang'),
        LL.t('common.verdict.pass'),
        LL.t('test.greeting', { name: 'Ann', unused: 1 }),
        LL.t('test.greeting'),
        LL.t('test.greeting', { name: null }),
        LL.t('test.two', { a: 0, b: false }),
    ]"""
    ru = run_js([], expression, messages={"test.greeting": "Привет, {name}!", "test.two": "{a}/{b}/{c}"})
    assert ru == ["ru", "ru-RU", "ru", "Успешно", "Привет, Ann!", "Привет, {name}!", "Привет, !", "0/false/{c}"]
    en = run_js([], expression, locale="en", messages={"test.greeting": "Hello, {name}!", "test.two": "{a}/{b}/{c}"})
    assert en == ["en", "en-GB", "en", "Passed", "Hello, Ann!", "Hello, {name}!", "Hello, !", "0/false/{c}"]


def test_en_falls_back_to_ru_per_key_like_the_server():
    from tests.js_harness import i18n_payload

    payload = i18n_payload("en")
    assert payload["messages"] == i18n.client_messages("en", None)
    assert payload["strict"] is True and payload["intl"] == "en-GB"


def test_missing_key_throws_in_strict_mode_and_warns_once_otherwise():
    strict = run_js([], "(() => { try { LL.t('test.nowhere'); return 'no error'; } catch (e) { return e.message; } })()")
    assert "test.nowhere" in strict
    expression = """
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        [LL.t('test.nowhere'), LL.t('test.nowhere'), LL.tn('test.nothing', 3), LL.tHtml('test.<x>'), warnings.length]
    """
    assert run_js([], expression, i18n_strict=False) == ["test.nowhere", "test.nowhere", "test.nothing", "test.&lt;x&gt;", 3]


def test_t_html_escapes_the_message_and_params_but_not_html_wrappers():
    expression = """[
        LL.tHtml('test.link', { name: '<b>Anna</b> & "co"', link: LL.html('<a href="/x">x</a>') }),
        LL.tHtml('test.link', { name: LL.html(LL.tHtml('test.link', { name: '<i>', link: 'y' })), link: '' }),
        LL.t('test.link', { name: '<b>', link: '<a>' }),
    ]"""
    out = run_js([], expression, messages={"test.link": "{name} & {link} 'q'"})
    assert out == [
        "&lt;b&gt;Anna&lt;/b&gt; &amp; &quot;co&quot; &amp; <a href=\"/x\">x</a> &#39;q&#39;",
        "&lt;i&gt; &amp; y &#39;q&#39; &amp;  &#39;q&#39;",
        "<b> & <a> 'q'",
    ]


@pytest.mark.parametrize("locale, messages", [("ru", PODS_RU), ("en", PODS_EN)])
def test_tn_matches_the_python_engine_for_0_to_200_and_fractions(locale, messages, tmp_path, monkeypatch):
    values = run_js([], f"{json.dumps(NUMBERS)}.map((n) => LL.tn('test.pods', n))", locale=locale, messages=messages)
    root = tmp_path / "locales"
    (root / locale).mkdir(parents=True)
    (root / locale / "test.json").write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(i18n, "LOCALES_DIR", root)
    expected = [i18n.plural("test.pods", n, locale=locale) for n in NUMBERS]
    assert values == expected
    if locale == "ru":
        assert values[:6] == ["0 подов", "1 под", "2 пода", "3 пода", "4 пода", "5 подов"]
        assert values[11:15] == ["11 подов", "12 подов", "13 подов", "14 подов"]
        assert values[21:26] == ["21 под", "22 пода", "23 пода", "24 пода", "25 подов"]
        assert values[NUMBERS.index(1.5)] == "1,5 пода"
    else:
        assert values[:3] == ["0 pods", "1 pod", "2 pods"] and values[NUMBERS.index(1.5)] == "1.5 pods"


def test_tn_params_and_fallback_suffixes():
    out = run_js(
        [],
        "[LL.tn('test.pods', 5, { n: 'пять' }), LL.tn('test.only_many', 1), LL.tn('test.with_other', 2), LL.tn('test.pods', 21, { unit: 'x' })]",
        messages={**PODS_RU, "test.only_many.many": "много", "test.with_other.other": "другое"},
    )
    assert out == ["пять подов", "много", "другое", "21 под"]


def test_python_plural_rule_matches_intl_plural_rules():
    expression = f"""(() => {{
        const numbers = {json.dumps(NUMBERS)};
        const ru = new Intl.PluralRules('ru'), en = new Intl.PluralRules('en');
        return {{ ru: numbers.map((n) => ru.select(n)), en: numbers.map((n) => en.select(n)), ll: numbers.map((n) => LL.pluralCategory(n)) }};
    }})()"""
    out = run_js([], expression)
    # Intl gives "other" for ru fractions; the dictionaries have no such form, so both helpers use "few".
    assert [("few" if c == "other" else c) for c in out["ru"]] == [i18n.plural_category(n, "ru") for n in NUMBERS]
    assert out["en"] == [i18n.plural_category(n, "en") for n in NUMBERS]
    assert out["ll"] == [i18n.plural_category(n, "ru") for n in NUMBERS]
    assert run_js([], f"{json.dumps(NUMBERS)}.map((n) => LL.pluralCategory(n))", locale="en") == out["en"]


def test_fmt_number_and_date_time_follow_the_locale():
    expression = """[
        LL.fmt.number(1234.5), LL.fmt.number(0.25, { style: 'percent' }), LL.fmt.number(null), LL.fmt.number('x'), LL.fmt.number('12'),
        LL.fmt.dateTime(Date.UTC(2026, 9, 18, 9, 5)), LL.fmt.dateTime('2026-10-18T09:05:00Z', { hour: '2-digit', minute: '2-digit' }),
        LL.fmt.date(new Date(Date.UTC(2026, 0, 2))), LL.fmt.dateTime('not a date'), LL.fmt.dateTime(null),
    ]"""
    ru = run_js([], expression)
    assert ru == [f"1{NBSP}234,5", f"25{NBSP}%", "—", "—", "12", "18.10.2026, 09:05", "09:05", "02.01.2026", "—", "—"]
    en = run_js([], expression, locale="en")
    assert en == ["1,234.5", "25%", "—", "—", "12", "18/10/2026, 09:05", "09:05", "02/01/2026", "—", "—"]


def test_i18n_js_survives_the_shared_scripts_and_a_page_without_dictionary():
    out = run_js(
        ["static/js/common.js", "static/js/ui.js", "static/js/palette.js"],
        "[LL === window.LoadLens, LL.t('common.verdict.fail'), typeof LL.ui.toast, LL.i18n.locale, LL.locale]",
        locale="en",
    )
    assert out == [True, "Failed", "function", "en", "en"]
    # Without the inline script (a page that forgot it) the helpers still work and fall back to ru formats.
    bare = run_js([], "delete window.LoadLens; " + (REPO_ROOT / "static/js/i18n.js").read_text(encoding="utf-8")
                  + "; [LL === window.LoadLens, window.LoadLens.locale, window.LoadLens.intl, window.LoadLens.t('common.verdict.pass')]",
                  i18n_strict=False)
    assert bare == [False, "ru", "ru-RU", "common.verdict.pass"]


def test_i18n_js_has_no_eval_and_no_cyrillic():
    source = (REPO_ROOT / "static" / "js" / "i18n.js").read_text(encoding="utf-8")
    assert "eval(" not in source and "new Function" not in source and "innerHTML" not in source
    assert not any("Ѐ" <= ch <= "ӿ" for ch in source)


def test_harness_reports_js_errors():
    with pytest.raises(JsHarnessError):
        run_js([], "LL.t('test.nowhere')")


def test_missing_node_skips_unless_required(monkeypatch):
    import tests.js_harness as harness

    monkeypatch.setattr(harness.shutil, "which", lambda name: None)
    monkeypatch.delenv("LOADLENS_REQUIRE_NODE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        harness.run_js([], "1")
    monkeypatch.setenv("LOADLENS_REQUIRE_NODE", "1")
    with pytest.raises(pytest.fail.Exception, match="LOADLENS_REQUIRE_NODE"):
        harness.run_js([], "1")
