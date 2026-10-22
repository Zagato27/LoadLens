# SPDX-License-Identifier: Apache-2.0
"""Rows of one domain that read different sources: resolution, collection, preview, settings API and packs."""

import copy
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from AI import pipeline as pipeline_module
from AI.data_sources import (
    DataSourceConfigError,
    binding_identity,
    domain_rows_language,
    domain_source_groups,
    normalize_binding,
    normalize_row_sources,
    resolve_domain_source,
    resolve_row_source,
    row_overrides,
    row_source_messages,
)
from AI.query_packs import load_catalog, merge_pack, prune_block, render_pack, set_row_sources
from loadlens_app import core
from loadlens_app.config_schema import QueryConfigError, validate_queries
from loadlens_app.metric_discovery import DomainNames

CATALOG = {
    "prom_apps": {"title": "Prometheus apps", "type": "prometheus", "prometheus": {"url": "http://prom-apps:9090"}},
    "prom_infra": {"title": "Prometheus infra", "type": "prometheus", "prometheus": {"url": "http://prom-infra:9090"}},
    "graf": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "bearer", "token": "x"}}},
    "influx": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "o", "token": "t"}},
}
BINDINGS = {"default": {"source": "prom_apps"}}
INFRA = {"source": "prom_infra"}
GRAFANA_K6 = {"source": "graf", "datasource_uid": "influx-uid", "database": "k6"}


def _cfg(queries=None, bindings=None):
    return {"data_sources": copy.deepcopy(CATALOG), "domain_sources": copy.deepcopy(bindings or BINDINGS), "queries": queries or {}}


def _block(**sources):
    return {
        "labels": ["DB: TPS", "DB host: CPU", "DB host: memory", "DB: empty"],
        "promql_queries": ["rate(tps[1m])", "host_cpu", "host_mem", ""],
        "label_keys_list": [["datname"], ["node"], ["node"], []],
        "sources": sources,
    }


# ---- resolution ----------------------------------------------------------------------------------------

def test_normalized_binding_has_every_field():
    assert normalize_binding({"source": " graf ", "datasource_uid": "u"}) == {
        "source": "graf", "datasource_uid": "u", "datasource_name": "", "database": "", "bucket": "",
    }
    assert binding_identity({"source": "graf"}) == binding_identity(normalize_binding({"source": "graf"}))
    queries = normalize_row_sources({"database": _block(**{"DB host: CPU": INFRA, "DB: TPS": None})})
    assert queries["database"]["sources"]["DB host: CPU"]["bucket"] == ""
    assert queries["database"]["sources"]["DB: TPS"] is None


def test_row_overrides_skip_null_blank_and_unknown_rows():
    block = _block(**{"DB host: CPU": INFRA, "DB host: memory": None, "DB: TPS": {"source": " "}, "gone": INFRA})
    assert row_overrides(block) == {"DB host: CPU": INFRA}


def test_a_row_reads_its_override_and_the_rest_the_domain_binding():
    block = _block(**{"DB host: CPU": INFRA})
    cfg = _cfg({"database": block})
    assert resolve_row_source(cfg, "database", block, "DB host: CPU").prometheus_url == "http://prom-infra:9090"
    assert resolve_row_source(cfg, "database", block, "DB host: CPU").label == "DB host: CPU"
    assert resolve_row_source(cfg, "database", block, "DB: TPS").prometheus_url == "http://prom-apps:9090"
    assert resolve_domain_source(cfg, "database").prometheus_url == "http://prom-apps:9090"


def test_groups_follow_sources_and_skip_rows_without_a_query():
    groups = domain_source_groups(_cfg(), "database", _block(**{"DB host: CPU": INFRA, "DB host: memory": INFRA}))
    assert [(g.resolved.source_id, g.language, g.rows, g.overridden) for g in groups] == [
        ("prom_apps", "promql", (0,), False),
        ("prom_infra", "promql", (1, 2), True),
    ]
    # Without overrides the domain is one group, as before.
    assert [g.rows for g in domain_source_groups(_cfg(), "database", _block())] == [(0, 1, 2)]


def test_a_row_override_must_run_its_language():
    block = _block(**{"DB host: CPU": {"source": "influx", "bucket": "telegraf"}})
    with pytest.raises(DataSourceConfigError, match="DB host: CPU"):
        domain_source_groups(_cfg(), "database", block)


def test_a_broken_override_names_the_row():
    with pytest.raises(DataSourceConfigError, match="DB host: CPU"):
        domain_source_groups(_cfg(), "database", _block(**{"DB host: CPU": {"source": "nope"}}))
    messages = row_source_messages({"database": _block(**{"DB host: CPU": {"source": "graf"}})}, CATALOG)
    assert len(messages) == 1 and "DB host: CPU" in messages[0] and "UID" in messages[0]
    assert row_source_messages({"database": _block(**{"DB host: CPU": GRAFANA_K6})}, CATALOG) == []
    used = row_source_messages({"database": _block(**{"DB host: CPU": INFRA})}, {}, place="запросы области shop", missing_kind="used")
    assert used == ["источник prom_infra используется: запросы области shop, домен database, строка «DB host: CPU»"]


def test_rows_with_their_own_source_do_not_bind_the_domain_language():
    block = {
        "labels": ["LT: RPS (k6)", "LT: RPS (Gatling)"],
        "influxql_queries": ["SELECT 1", ""],
        "promql_queries": ["", "sum(rate(gatling_requests[1m]))"],
        "label_tag_keys_list": [[], []],
        "label_keys_list": [[], []],
        "sources": {"LT: RPS (k6)": GRAFANA_K6},
    }
    resolved = resolve_domain_source(_cfg(), "lt_framework")
    assert domain_rows_language(resolved, block) == "promql"
    groups = domain_source_groups(_cfg(), "lt_framework", block)
    assert [(g.resolved.source_id, g.language, g.rows) for g in groups] == [("graf", "influxql", (0,)), ("prom_apps", "promql", (1,))]


def test_validate_queries_checks_the_shape_of_sources():
    validate_queries({"database": _block(**{"DB host: CPU": INFRA, "DB: TPS": None})})
    for bad in ([], {"DB: TPS": "prom_infra"}, {"DB: TPS": 1}):
        with pytest.raises(QueryConfigError, match="sources"):
            validate_queries({"database": {**_block(), "sources": bad}})


# ---- blocks ----------------------------------------------------------------------------------------------

def test_set_row_sources_and_prune():
    block = _block()
    set_row_sources(block, ["DB host: CPU"], INFRA)
    assert block["sources"] == {"DB host: CPU": normalize_binding(INFRA)}
    # Back to the domain binding: an existing entry becomes an explicit null, a row without one stays out.
    set_row_sources(block, ["DB host: CPU", "DB: TPS"], None)
    assert block["sources"] == {"DB host: CPU": None}
    block["labels"] = block["labels"][:1]
    for key in ("promql_queries", "label_keys_list"):
        block[key] = block[key][:1]
    assert "sources" not in prune_block(block)


def test_replacing_a_domain_with_a_pack_keeps_the_sources_of_its_rows():
    rendered = render_pack(load_catalog().packs["host-node-exporter"], {"host_title": "DB host"})
    label = rendered.queries[0].label
    block, _ = merge_pack(None, rendered, "merge")
    set_row_sources(block, [label], INFRA)
    block["labels"].append("Foreign")
    block["promql_queries"].append("up_foreign")
    block["label_keys_list"].append([])
    block["sources"]["Foreign"] = INFRA
    replaced, plan = merge_pack(block, rendered, "replace")
    assert plan.removed == ["Foreign"]
    assert set(replaced["sources"]) == {label}


# ---- collection ------------------------------------------------------------------------------------------

def test_collection_runs_each_source_and_keeps_label_order(monkeypatch):
    calls = []
    frame = pd.DataFrame({"node=db-1": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1min"))

    def fake_prom(url, start, end, queries, keys, step, resample_interval, ef_config=None):
        calls.append((url, list(queries)))
        if url == "http://prom-infra:9090":
            raise RuntimeError("infra is down")
        return [frame.rename(columns={"node=db-1": query}) for query in queries]

    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", fake_prom)
    block = _block(**{"DB host: memory": INFRA})
    prepared = pipeline_module._prepare_metric_domains(_cfg(), {"database": block}, ["database"])
    frames = pipeline_module._collect_domain_frames(prepared["database"], block, 0, 60, "1m", "1min")
    assert calls == [("http://prom-apps:9090", ["rate(tps[1m])", "host_cpu"]), ("http://prom-infra:9090", ["host_mem"])]
    assert len(frames) == 4
    assert list(frames[0].columns) == ["rate(tps[1m])"] and list(frames[1].columns) == ["host_cpu"]
    # The failed source leaves only its rows empty; the row without a query is empty as before.
    assert frames[2].empty and frames[3].empty


def test_a_report_fails_early_on_a_broken_row_override():
    block = _block(**{"DB host: CPU": {"source": "removed"}})
    with pytest.raises(DataSourceConfigError, match="removed"):
        pipeline_module._prepare_metric_domains(_cfg(), {"database": block}, ["database"])


# ---- preview ---------------------------------------------------------------------------------------------

def _capture_prometheus(monkeypatch):
    from loadlens_app import query_preview

    urls = []

    def fake_fetch(url, start, end, query, step, ef_config=None):
        urls.append(ef_config["metrics_source"]["prometheus"]["url"])
        return {"status": "success", "data": {"result": [{"metric": {"node": "db-1"}, "values": [[1, "1"]]}]}}

    monkeypatch.setattr(query_preview, "fetch_metric_series", fake_fetch)
    return urls


def test_preview_reads_the_row_source(monkeypatch):
    from loadlens_app.query_preview import fetch_points, preview_metric_query

    urls = _capture_prometheus(monkeypatch)
    cfg = _cfg({"database": _block(**{"DB host: CPU": INFRA})})
    assert preview_metric_query(cfg, domain="database", lang="promql", query="host_cpu", label="DB host: CPU").ok
    assert preview_metric_query(cfg, domain="database", lang="promql", query="tps", label="DB: TPS").ok
    assert preview_metric_query(cfg, domain="database", lang="promql", query="x", binding={"source": "prom_infra"}).ok
    fetch_points(cfg, domain="database", lang="promql", query="host_cpu", start_ts=0, end_ts=60, step="30s", label="DB host: CPU")
    assert urls == ["http://prom-infra:9090", "http://prom-apps:9090", "http://prom-infra:9090", "http://prom-infra:9090"]


# ---- settings API ----------------------------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch, tmp_path):
    import update_page

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"shop": {"title": "Shop", "services": {"checkout": {}}}}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    base = {"queries": {}, "sla": {}, "per_area": {}, "data_sources": copy.deepcopy(CATALOG), "domain_sources": copy.deepcopy(BINDINGS)}
    for key, value in base.items():
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(value))
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), runtime=runtime)


def _runtime(env):
    return json.loads(env.runtime.read_text(encoding="utf-8"))


def _post(env, section, data, **scope):
    return env.client.post("/config", json={"section": section, "data": data, **scope})


def test_saving_queries_checks_and_normalizes_row_sources(env):
    bad = _post(env, "queries", {"database": _block(**{"DB host: CPU": {"source": "nope"}})})
    assert bad.status_code == 400 and "DB host: CPU" in bad.get_json()["error"]
    wrong_field = _post(env, "queries", {"database": _block(**{"DB host: CPU": {"source": "prom_infra", "bucket": "b"}})})
    assert wrong_field.status_code == 400 and "bucket" in wrong_field.get_json()["error"]
    language = _post(env, "queries", {"database": _block(**{"DB host: CPU": {"source": "influx", "bucket": "telegraf"}})})
    assert language.status_code == 400 and "flux" in language.get_json()["error"]
    ok = _post(env, "queries", {"database": _block(**{"DB host: CPU": INFRA, "DB: TPS": None, "removed row": INFRA})})
    assert ok.status_code == 200, ok.get_json()
    stored = core.CONFIG["queries"]["database"]["sources"]
    assert stored == {"DB host: CPU": normalize_binding(INFRA), "DB: TPS": None}


def test_a_source_read_by_a_row_cannot_be_removed(env):
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": INFRA})}, area="shop", service="checkout").status_code == 200
    catalog = copy.deepcopy(CATALOG)
    del catalog["prom_infra"]
    resp = _post(env, "data_sources", catalog)
    assert resp.status_code == 400
    error = resp.get_json()["error"]
    assert "prom_infra" in error and "checkout" in error and "DB host: CPU" in error


def test_service_scope_can_return_a_row_to_the_domain_binding(env):
    import update_page

    area_block = _block(**{"DB host: CPU": INFRA})
    assert _post(env, "queries", {"database": area_block}, area="shop").status_code == 200
    assert resolve_row_source(update_page._effective_config_for_scope("shop", "checkout"), "database",
                              update_page._effective_config_for_scope("shop", "checkout")["queries"]["database"], "DB host: CPU").source_id == "prom_infra"
    # The service sets an explicit null: the area's override does not come back through the merge.
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": None})}, area="shop", service="checkout").status_code == 200
    cfg = update_page._effective_config_for_scope("shop", "checkout")
    assert resolve_row_source(cfg, "database", cfg["queries"]["database"], "DB host: CPU").source_id == "prom_apps"
    # Another binding in the service replaces every field of the area's one.
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": GRAFANA_K6})}, area="shop").status_code == 200
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": INFRA})}, area="shop", service="checkout").status_code == 200
    cfg = update_page._effective_config_for_scope("shop", "checkout")
    assert cfg["queries"]["database"]["sources"]["DB host: CPU"] == normalize_binding(INFRA)


def test_query_preview_endpoint_takes_an_unsaved_binding(env, monkeypatch):
    urls = _capture_prometheus(monkeypatch)
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": INFRA})}).status_code == 200
    body = {"domain": "database", "lang": "promql", "query": "host_cpu", "label": "DB host: CPU"}
    assert env.client.post("/config/query_preview", json=body).status_code == 200
    assert env.client.post("/config/query_preview", json={**body, "binding": None}).status_code == 200
    assert env.client.post("/config/query_preview", json={**body, "binding": {"source": "prom_apps"}}).status_code == 200
    assert urls == ["http://prom-infra:9090", "http://prom-apps:9090", "http://prom-apps:9090"]
    bad = env.client.post("/config/query_preview", json={**body, "binding": {"source": "graf"}})
    assert bad.status_code == 400 and "UID" in bad.get_json()["message"]


# ---- packs -----------------------------------------------------------------------------------------------

def _apply(env, **body):
    return env.client.post("/config/query_packs/apply", json={"pack_id": "host-node-exporter", "domain": "database", "variables": {"host_title": "DB host"}, **body})


def _pack_target(env):
    data = env.client.get("/config/query_packs").get_json()
    pack = next(item for item in data["packs"] if item["id"] == "host-node-exporter")
    return pack["targets"]["database"]


def test_pack_rows_read_the_source_picked_at_install(env):
    resp = _apply(env, binding=INFRA)
    assert resp.status_code == 200, resp.get_json()
    block = core.CONFIG["queries"]["database"]
    assert set(block["sources"]) == set(block["labels"]) and all(value == normalize_binding(INFRA) for value in block["sources"].values())
    installed = _pack_target(env)["installed"]
    assert installed["binding"] == normalize_binding(INFRA) and installed["source"] == "Prometheus infra"
    # An update without a binding keeps the rows' source.
    assert _apply(env).status_code == 200
    assert all(value == normalize_binding(INFRA) for value in core.CONFIG["queries"]["database"]["sources"].values())
    # The domain binding itself (or null) returns them to the domain.
    assert _apply(env, binding={"source": "prom_apps"}).status_code == 200
    assert all(value is None for value in core.CONFIG["queries"]["database"]["sources"].values())
    assert _pack_target(env)["installed"]["binding"] is None


def test_pack_binding_is_checked(env):
    bad = _apply(env, binding={"source": "graf"})
    assert bad.status_code == 400 and "UID" in bad.get_json()["error"]
    wrong_language = _apply(env, binding={"source": "influx", "bucket": "telegraf"})
    assert wrong_language.status_code == 409
    assert core.CONFIG["queries"] == {}


def test_pack_check_runs_against_the_picked_source(env, monkeypatch):
    from loadlens_app import query_pack_service
    from loadlens_app.query_preview import QueryPreview

    seen = []

    def fake_preview(cfg, **kwargs):
        seen.append(kwargs.get("binding"))
        return QueryPreview(ok=True, message="ok", series_count=1, point_count=1, series=["node=a"])

    monkeypatch.setattr(query_pack_service, "preview_metric_query", fake_preview)
    resp = env.client.post("/config/query_packs/check", json={"pack_id": "host-node-exporter", "domain": "database", "binding": INFRA})
    assert resp.status_code == 200 and resp.get_json()["source"] == "Prometheus infra"
    assert seen and all(binding == normalize_binding(INFRA) for binding in seen)


def test_detection_offers_a_catalog_source_the_domain_is_not_bound_to():
    from loadlens_app.query_pack_service import ScopeState, detect_packs

    calls = []

    def discover(cfg, domains, extra=None):
        calls.append(sorted(extra or {}))
        out = {domain: DomainNames(domain, "Prometheus apps", "promql", frozenset()) for domain in domains}
        for key, binding in (extra or {}).items():
            names = {"node_cpu_seconds_total", "node_memory_MemAvailable_bytes", "node_uname_info"} if binding["source"] == "prom_infra" else set()
            out[key] = DomainNames(key, binding["source"], "promql", frozenset(names))
        return out

    state = ScopeState(queries={}, sla={}, effective_cfg=_cfg())
    result = detect_packs(state, discover)
    assert calls == [["source:1"]]
    alternative = result["packs"]["host-node-exporter"]["alternative"]
    assert alternative["score"] == 1.0 and alternative["binding"] == normalize_binding(INFRA) and alternative["title"] == "Prometheus infra"
    assert result["sources"]["source:1"]["title"] == "Prometheus infra"
    assert "alternative" not in result["packs"]["jvm-micrometer"]


# ---- live charts and tool hints --------------------------------------------------------------------------

def test_live_rows_carry_their_language_and_source():
    from loadlens_app.load_tests import live

    queries = {"lt_framework": {
        "labels": ["LT: RPS k6", "LT: VUs"],
        "influxql_queries": ["SELECT rps", ""],
        "promql_queries": ["", "sum(vus)"],
        "label_tag_keys_list": [[], []],
        "label_keys_list": [[], []],
        "roles": {"LT: RPS k6": "lt.rps_total", "LT: VUs": "lt.vus"},
        "sources": {"LT: RPS k6": GRAFANA_K6},
    }}
    lang, rows = live._block_rows(queries, ("lt.rps_total", "lt.vus"))
    assert lang == "influxql"
    assert rows["lt.rps_total"] == live.LiveRow("LT: RPS k6", "SELECT rps", [], "influxql", True)
    assert rows["lt.vus"] == live.LiveRow("LT: VUs", "sum(vus)", [], "promql", False)


def test_tool_hint_follows_the_source_of_its_pack_rows():
    from loadlens_app.load_tests.hints import output_hint

    rendered = render_pack(load_catalog().packs["lt-k6-prometheus-rw"])
    block, _ = merge_pack(None, rendered, "merge")
    set_row_sources(block, [query.label for query in rendered.queries], INFRA)
    cfg = _cfg({"lt_framework": block})
    hint = output_hint(cfg, "k6", ["lt-k6-prometheus-rw"])
    assert hint.env["K6_PROMETHEUS_RW_SERVER_URL"].startswith("http://prom-infra:9090")
    assert output_hint(_cfg(), "k6", ["lt-k6-prometheus-rw"]).env["K6_PROMETHEUS_RW_SERVER_URL"].startswith("http://prom-apps:9090")


# ---- review fixes ----------------------------------------------------------------------------------------

def test_an_update_without_a_binding_keeps_the_installation_source_for_new_rows(env):
    first = _apply(env, binding=INFRA, include_indexes=[0])
    assert first.status_code == 200, first.get_json()
    assert len(core.CONFIG["queries"]["database"]["labels"]) == 1
    # An API client updates the installation with every query and no binding: the new rows read INFRA too.
    assert _apply(env).status_code == 200
    block = core.CONFIG["queries"]["database"]
    assert len(block["labels"]) == 7
    assert all(block["sources"].get(label) == normalize_binding(INFRA) for label in block["labels"])


def test_a_narrower_scope_with_its_rows_reads_only_its_own_sources(env):
    import update_page
    from loadlens_app.blueprints.config_api import scope_view

    # The area keeps its own copy of the rows; the global scope adds an override afterwards.
    assert _post(env, "queries", {"database": _block()}, area="shop").status_code == 200
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": INFRA})}).status_code == 200
    cfg = update_page._effective_config_for_scope("shop", None)
    assert resolve_row_source(cfg, "database", cfg["queries"]["database"], "DB host: CPU").source_id == "prom_apps"
    # A domain the area does not list still inherits the global rows and their sources.
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": INFRA}), "kafka": _block(**{"DB host: CPU": INFRA})}).status_code == 200
    cfg = update_page._effective_config_for_scope("shop", None)
    assert row_overrides(cfg["queries"]["kafka"]) == {"DB host: CPU": normalize_binding(INFRA)}
    # The service view and the service's effective config agree.
    assert _post(env, "queries", {"database": _block()}, area="shop", service="checkout").status_code == 200
    assert _post(env, "queries", {"database": _block(**{"DB host: CPU": INFRA})}, area="shop").status_code == 200
    assert "sources" not in scope_view("queries", "shop", "checkout")["database"]
    cfg = update_page._effective_config_for_scope("shop", "checkout")
    assert row_overrides(cfg["queries"]["database"]) == {}


def test_live_rows_with_two_languages_run_the_one_their_source_accepts():
    from loadlens_app.load_tests import live

    queries = {"lt_framework": {
        "labels": ["LT: RPS"],
        "influxql_queries": ["SELECT rps"],
        "promql_queries": ["sum(rps)"],
        "label_tag_keys_list": [[]],
        "label_keys_list": [[]],
        "roles": {"LT: RPS": "lt.rps_total"},
    }}
    assert live._block_rows(queries, ("lt.rps_total",), _cfg())[1]["lt.rps_total"].lang == "promql"
    grafana = _cfg(bindings={"default": GRAFANA_K6})
    assert live._block_rows(queries, ("lt.rps_total",), grafana)[1]["lt.rps_total"].lang == "influxql"
