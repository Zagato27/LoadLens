# SPDX-License-Identifier: Apache-2.0
"""Several query languages in one domain: a source runs one language, rows of other sources run theirs."""

import copy
import json
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from AI.data_sources import mixed_language_warnings, normalize_binding
from AI.query_packs import (
    SUMMED_RPS_LABEL,
    BlockConflictError,
    DerivedRowError,
    add_summed_rps,
    load_catalog,
    merge_conflict,
    merge_pack,
    render_pack,
    set_row_sources,
)
from loadlens_app import core
from loadlens_app.config_schema import validate_queries

CATALOG = load_catalog()
SOURCES = {
    "graf": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "bearer", "token": "x"}}},
    "vm": {"title": "VictoriaMetrics", "type": "prometheus", "prometheus": {"url": "http://vm:8428"}},
    "vm2": {"title": "VictoriaMetrics 2", "type": "prometheus", "prometheus": {"url": "http://vm2:8428"}},
}
K6_INFLUX = {"source": "graf", "datasource_uid": "influx-uid", "database": "k6"}
BINDINGS = {"default": {"source": "vm"}, "lt_framework": K6_INFLUX}
VM = {"source": "vm"}


def _k6_influx_block():
    block, _ = merge_pack(None, render_pack(CATALOG.packs["lt-k6-influxql"]))
    return block


# ---- merging packs ---------------------------------------------------------------------------------------

def test_a_pack_in_another_language_merges_when_its_rows_read_another_source():
    block = _k6_influx_block()
    rendered = render_pack(CATALOG.packs["lt-gatling-graphite-victoriametrics-labels"])
    # The same Grafana datasource cannot run PromQL next to InfluxQL.
    assert merge_conflict(block, "promql") is not None
    with pytest.raises(BlockConflictError, match="InfluxQL"):
        merge_pack(block, rendered)
    # Read from VictoriaMetrics, the Gatling rows join the k6 rows.
    merged, plan = merge_pack(block, rendered, source=VM)
    assert len(plan.added) == len(rendered.queries) and not plan.removed
    set_row_sources(merged, [query.label for query in rendered.queries], VM)
    validate_queries({"lt_framework": merged})
    assert merge_conflict(merged, "promql", VM) is None
    assert merge_conflict(merged, "influxql", VM) is not None
    assert {entry["id"] for entry in merged["packs"]} == {"lt-k6-influxql", "lt-gatling-graphite-victoriametrics-labels"}


def test_a_pack_row_drops_the_text_of_another_language():
    rendered = render_pack(CATALOG.packs["lt-k6-prometheus-rw"])
    label = rendered.queries[0].label
    block = {"labels": [label], "influxql_queries": ["SELECT 1"], "label_tag_keys_list": [["a"]]}
    merged, plan = merge_pack(block, rendered, source=VM)
    index = merged["labels"].index(label)
    assert merged["influxql_queries"][index] == "" and merged["promql_queries"][index] == rendered.queries[0].query
    assert label in plan.updated


# ---- the summed RPS --------------------------------------------------------------------------------------

def _rps_block(sources):
    block = {"labels": [], "promql_queries": [], "label_keys_list": [], "influxql_queries": [], "label_tag_keys_list": [], "roles": {}}
    for label, (lang, source) in sources.items():
        block["labels"].append(label)
        block["promql_queries"].append(f"sum(rate({label.lower()}[1m]))" if lang == "promql" else "")
        block["influxql_queries"].append("SELECT 1" if lang == "influxql" else "")
        block["label_keys_list"].append([])
        block["label_tag_keys_list"].append([])
        block["roles"][label] = "lt.rps_total"
        if source:
            set_row_sources(block, [label], source)
    return block


def test_summed_rps_takes_promql_rows_of_one_source():
    block = _rps_block({"K6": ("influxql", None), "Gatling": ("promql", VM), "Locust": ("promql", VM)})
    summed, label, parts = add_summed_rps(block)
    assert label == SUMMED_RPS_LABEL and parts == ["Gatling", "Locust"]
    assert summed["sources"][SUMMED_RPS_LABEL] == normalize_binding(VM)
    index = summed["labels"].index(SUMMED_RPS_LABEL)
    assert "gatling" in summed["promql_queries"][index] and summed["influxql_queries"][index] == ""
    validate_queries({"lt_framework": summed})


def test_summed_rps_refuses_parts_from_different_sources_or_languages():
    with pytest.raises(DerivedRowError, match="разные источники"):
        add_summed_rps(_rps_block({"Gatling": ("promql", VM), "Locust": ("promql", {"source": "vm2"})}))
    with pytest.raises(DerivedRowError, match="PromQL"):
        add_summed_rps(_rps_block({"K6": ("influxql", None), "Gatling": ("promql", VM)}))


# ---- warnings --------------------------------------------------------------------------------------------

def test_rows_with_two_languages_are_reported_with_the_language_that_runs():
    block = {"labels": ["Both"], "influxql_queries": ["SELECT 1"], "promql_queries": ["up_both"], "label_keys_list": [[]], "label_tag_keys_list": [[]]}
    cfg = {"data_sources": SOURCES, "domain_sources": BINDINGS}
    assert mixed_language_warnings({"lt_framework": block}, cfg) == [
        "lt_framework: у строки «Both» есть запросы на нескольких языках (influxql, promql); выполняется influxql, остальные лучше удалить"
    ]
    set_row_sources(block, ["Both"], VM)
    assert "выполняется promql" in mixed_language_warnings({"lt_framework": block}, cfg)[0]


# ---- API -------------------------------------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    base = {"queries": {}, "sla": {}, "per_area": {}, "data_sources": SOURCES, "domain_sources": BINDINGS}
    for key, value in base.items():
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(value))
    from loadlens_app.blueprints import query_packs as blueprint

    monkeypatch.setattr(blueprint, "_effective_cfg", lambda area, service: {
        "data_sources": core.CONFIG["data_sources"], "domain_sources": core.CONFIG["domain_sources"], "default_params": {"step": "30s"},
    })
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client())


def _catalog_pack(env, pack_id):
    return next(pack for pack in env.client.get("/config/query_packs").get_json()["packs"] if pack["id"] == pack_id)


def test_k6_in_influxdb_and_gatling_in_victoriametrics_in_one_domain(env):
    assert env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-influxql"}).status_code == 200
    gatling = "lt-gatling-graphite-victoriametrics-labels"
    assert _catalog_pack(env, gatling)["compatibility"]["status"] == "replace_only"
    refused = env.client.post("/config/query_packs/apply", json={"pack_id": gatling})
    assert refused.status_code == 409 and "другой источник" in refused.get_json()["error"]
    applied = env.client.post("/config/query_packs/apply", json={"pack_id": gatling, "binding": VM, "fill_sla": False})
    assert applied.status_code == 200, applied.get_json()
    block = core.CONFIG["queries"]["lt_framework"]
    assert any(text.strip() for text in block["influxql_queries"]) and any(text.strip() for text in block["promql_queries"])
    # Updating the Gatling pack keeps its source and still does not collide with the k6 rows.
    again = env.client.post("/config/query_packs/apply", json={"pack_id": gatling, "fill_sla": False})
    assert again.status_code == 200 and not again.get_json()["plan"]["added"]


def test_saving_rows_in_two_languages_warns(env):
    block = {"labels": ["Both"], "influxql_queries": ["SELECT 1"], "promql_queries": ["up_both"], "label_keys_list": [[]], "label_tag_keys_list": [[]]}
    resp = env.client.post("/config", json={"section": "queries", "data": {"lt_framework": block}})
    assert resp.status_code == 200
    assert any("Both" in warning for warning in resp.get_json()["warnings"])


def test_domain_binding_check_ignores_rows_with_their_own_source(env, monkeypatch):
    from loadlens_app.blueprints import config_api
    from loadlens_app.connection_checks import CheckResult

    languages = []

    def fake_check(resolved, language):
        languages.append((resolved.source_id, language))
        return CheckResult(ok=True, message="ok")

    monkeypatch.setattr(config_api, "check_resolved_source", fake_check)
    block = {"labels": ["K6", "Gatling"], "influxql_queries": ["SELECT 1", ""], "promql_queries": ["", "up_g"],
             "label_keys_list": [[], []], "label_tag_keys_list": [[], []], "sources": {"Gatling": VM}}
    monkeypatch.setitem(core.CONFIG, "queries", {"lt_framework": block})
    resp = env.client.post("/config/test_connection", json={"section": "domain_sources", "domain": "lt_framework", "data": BINDINGS})
    assert resp.status_code == 200, resp.get_json()
    assert languages == [("graf", "influxql")]


def test_an_override_equal_to_the_domain_binding_reads_the_same_source():
    block = _k6_influx_block()
    labels = block["labels"]
    # The k6 rows name the domain's Grafana datasource explicitly.
    set_row_sources(block, labels, K6_INFLUX)
    assert merge_conflict(block, "promql") is None
    assert merge_conflict(block, "promql", domain=K6_INFLUX) is not None
    with pytest.raises(BlockConflictError):
        merge_pack(block, render_pack(CATALOG.packs["lt-k6-prometheus-rw"]), domain=K6_INFLUX)
    parts = _rps_block({"Gatling": ("promql", None), "Locust": ("promql", VM)})
    with pytest.raises(DerivedRowError):
        add_summed_rps(parts)
    summed, _label, names = add_summed_rps(parts, domain=VM)
    assert names == ["Gatling", "Locust"]
