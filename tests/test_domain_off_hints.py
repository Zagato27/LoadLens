# SPDX-License-Identifier: Apache-2.0
"""Domains a service does not analyse (the ticks in Settings -> Services) on the queries page of the settings
(static/js/settings.js): the pure helpers under Node (tests/js_harness.py), and the markup, strings and styles that
mark the tabs, explain the shown domain and count the rows at the ticks."""

from __future__ import annotations

import json
import re

from tests.js_harness import REPO_ROOT, run_js

SETTINGS_JS = ["static/js/common.js", "static/js/ui.js", "static/js/settings.js"]
INTERNALS = "LL.settings._internals"

META = {
    "checkout": {"title": "Checkout", "disabled_domains": ["kafka", "jvm"]},
    "billing": {"title": "", "disabled_domains": ["kafka"]},
    "search": {"title": "Search", "disabled_domains": []},
}
IDS = ["billing", "checkout", "search"]


def _note(domain: str, scope: str, meta=META, ids=IDS, locale: str = "ru"):
    expr = f"{INTERNALS}.domainOffNote({json.dumps(domain)}, {json.dumps(scope)}, {json.dumps(meta)}, {json.dumps(ids)})"
    return run_js(SETTINGS_JS, expr, locale=locale)


def test_off_domains_of_a_service_keep_strings_only():
    expr = f"""[
        {INTERNALS}.offDomainsOf({{ disabled_domains: ['kafka', 1, '', null, 'jvm'] }}),
        {INTERNALS}.offDomainsOf({{ disabled_domains: 'kafka' }}),
        {INTERNALS}.offDomainsOf(null),
        {INTERNALS}.offDomainsOf('checkout'),
    ]"""
    assert run_js(SETTINGS_JS, expr) == [["kafka", "jvm"], [], [], []]


def test_services_with_the_domain_off_follow_the_order_of_the_ids():
    expr = f"{INTERNALS}.servicesWithDomainOff('kafka', {json.dumps(META)}, %s)"
    assert run_js(SETTINGS_JS, expr % json.dumps(IDS)) == ["billing", "checkout"]
    assert run_js(SETTINGS_JS, expr % json.dumps(["checkout", "billing", "gone"])) == ["checkout", "billing"]
    assert run_js(SETTINGS_JS, f"{INTERNALS}.servicesWithDomainOff('kafka', null, ['a'])") == []


def test_the_picked_service_names_itself_by_its_title():
    note = _note("kafka", "checkout")
    assert note["services"] == ["checkout"]
    assert note["text"] == ("Домен выключен для сервиса Checkout в разделе «Сервисы»: запросы сохраняются, "
                            "но в отчёты сервиса не попадают.")
    # A service without a title is named by its id; a domain the service reads gives no line.
    assert "сервиса billing " in _note("kafka", "billing")["text"]
    assert _note("database", "checkout") is None
    assert _note("kafka", "search") is None
    assert _note("kafka", "unknown") is None


def test_the_area_scope_lists_the_services_that_skip_the_domain():
    many = _note("kafka", "")
    assert many["services"] == ["billing", "checkout"]
    assert many["text"].startswith("Домен выключен для сервисов billing, Checkout в разделе «Сервисы»: их отчёты")
    one = _note("jvm", "")
    assert one["services"] == ["checkout"]
    assert one["text"].startswith("Домен выключен для сервиса Checkout в разделе «Сервисы»: его отчёты")
    assert _note("database", "") is None
    # Junk in the meta never throws.
    assert _note("kafka", "", meta={"x": "junk", "y": {"disabled_domains": "kafka"}}, ids=["x", "y"]) is None


def test_the_notes_are_english_in_en():
    service = _note("kafka", "checkout", locale="en")["text"]
    assert service == ("The domain is off for the service Checkout in Services: the queries are kept but do not go "
                       "into the reports of the service.")
    assert _note("kafka", "", locale="en")["text"].startswith("The domain is off for the services billing, Checkout")
    assert _note("jvm", "", locale="en")["text"].startswith("The domain is off for the service Checkout in Services")


def test_row_count_reads_the_service_override_over_the_area():
    area = {"kafka": {"labels": ["a", "b", "c"]}, "jvm": {"labels": []}, "database": {"promql_queries": ["q"]}}
    service = {"kafka": {"labels": ["a", "b", "c", "d", "e"]}}
    expr = f"""(() => {{
        const I = {INTERNALS};
        const area = {json.dumps(area)};
        const service = {json.dumps(service)};
        return [
            I.domainRowCount('kafka', area, service),
            I.domainRowCount('kafka', area, undefined),
            I.domainRowCount('jvm', area, service),
            I.domainRowCount('database', area, service),
            I.domainRowCount('application_logs', area, service),
            I.domainRowCount('kafka', null, null),
        ];
    }})()"""
    assert run_js(SETTINGS_JS, expr) == [5, 3, 0, 0, None, None]


def test_the_page_marks_the_tabs_explains_the_domain_and_counts_the_rows():
    source = (REPO_ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    # Tabs: only with a service picked, dimmed and with "off" after the count.
    assert "const off = state.queriesScope ? offDomainsOf(state.servicesMeta[state.queriesScope]) : [];" in source
    assert "tab.classList.add('is-off');" in source and "makeEl('span', 'tab-off', LL.t('settings.queries.tab_off'))" in source
    # The line under the tabs, not in the wizard, with a way to Services.
    assert "const offLine = domainOffLine(domain);" in source
    assert "if (state.wizard) return null;" in source and "() => activateSection('services')" in source
    # Counts at the ticks of Services, and both directions of the refresh.
    assert "state.domainList.map((d) => domainTick(id, d, disabled.includes(d)))" in source
    assert "LL.t('settings.services.domain_count_title', { n: count })" in source
    assert "if ($('queryList')) renderQueriesForm();" in source
    assert "renderServices();  // the row counts at the domain ticks" in source

    css = (REPO_ROOT / "static" / "css" / "app.css").read_text(encoding="utf-8")
    rule = re.search(r"\.struct-tabs \.btn\.is-off \{([^}]*)\}", css)
    assert rule and "var(--na)" in rule.group(1) and "accent" not in rule.group(1)
    assert re.search(r"\.domain-off-note \{[^}]*flex-wrap: wrap", css)

    for lang in ("ru", "en"):
        messages = json.loads((REPO_ROOT / "locales" / lang / "settings.json").read_text(encoding="utf-8"))
        for key in ("queries.tab_off", "queries.tab_off_title", "queries.domain_off_service",
                    "queries.domain_off_services.one", "queries.open_services", "services.domain_count_title"):
            assert f"settings.{key}" in messages, (lang, key)
