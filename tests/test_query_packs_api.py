# SPDX-License-Identifier: Apache-2.0
"""Query pack endpoints: catalog, preview, check and apply for global, area and service scopes."""

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
from loadlens_app import core

SOURCES = {
    "prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}},
    "graf": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "basic", "username": "u", "password": "p"}}},
    "influx": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "o", "token": "t"}},
}
BINDINGS = {
    "default": {"source": "prom"},
    "lt_framework": {"source": "graf", "datasource_uid": "influx-uid", "database": "k6"},
    "kafka": {"source": "influx", "bucket": "telegraf"},
}
RPS_LABEL = "LT (InfluxQL): RPS sum by all groups"
VUS_LABEL = "LT (InfluxQL): VUs"


@pytest.fixture
def env(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"shop": {"title": "Shop", "services": {"checkout": {}}}}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    base = {"queries": {}, "sla": {"target_rps": 100}, "per_area": {}, "data_sources": SOURCES, "domain_sources": BINDINGS}
    for key, value in base.items():
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(value))

    from loadlens_app.blueprints import query_packs as blueprint

    # The real resolver reads the runtime file through update_page's own path; the tests keep it in tmp.
    monkeypatch.setattr(blueprint, "_effective_cfg", lambda area, service: {
        "data_sources": core.CONFIG["data_sources"],
        "domain_sources": core.CONFIG["domain_sources"],
        "default_params": {"step": "30s"},
    })
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), runtime=runtime)


def _runtime(env) -> dict:
    return json.loads(env.runtime.read_text(encoding="utf-8"))


def _pack(catalog: dict, pack_id: str) -> dict:
    return next(pack for pack in catalog["packs"] if pack["id"] == pack_id)


def test_catalog_reports_compatibility_and_roles(env):
    resp = env.client.get("/config/query_packs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert _pack(data, "jvm-micrometer")["compatibility"]["status"] == "ok"
    assert _pack(data, "lt-k6-influxql")["compatibility"] == {"status": "ok", "reason": "", "source": "Grafana"}
    kafka = _pack(data, "kafka-exporter")["compatibility"]
    assert kafka["status"] == "incompatible" and "не выполняет PromQL" in kafka["reason"]
    assert _pack(data, "jvm-micrometer")["installed"] is None
    assert data["roles"]["runtime.cpu"]["title"] == "CPU экземпляра сервиса"
    assert any(bundle["id"] == "spring-k8s-k6-influxdb" for bundle in data["bundles"])


def test_preview_writes_nothing(env):
    before = env.runtime.read_text(encoding="utf-8")
    resp = env.client.post("/config/query_packs/preview", json={"pack_id": "jvm-micrometer"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["plan"]["added"]) == 10
    assert data["sla"]["changes"]["service_cpu_query"]["to"] == "JVM: Process CPU usage by (application, instance)"
    assert env.runtime.read_text(encoding="utf-8") == before
    assert core.CONFIG["queries"] == {}


def test_apply_global_scope_then_again_is_unchanged(env):
    resp = env.client.post("/config/query_packs/apply", json={"pack_id": "jvm-micrometer", "variables": {"matchers": 'namespace="shop"'}})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["previous"] == {"queries": {}, "sla": {"target_rps": 100}, "own_queries": True, "own_sla": True}
    jvm = core.CONFIG["queries"]["jvm"]
    assert len(jvm["labels"]) == 10 and 'namespace="shop"' in jvm["promql_queries"][0]
    assert jvm["roles"]["JVM: Process CPU usage by (application, instance)"] == "runtime.cpu"
    assert jvm["packs"] == [{"id": "jvm-micrometer", "version": 1, "variables": {"service_label": "application", "matchers": 'namespace="shop"', "window": "2m"}}]
    assert core.CONFIG["sla"] == {"target_rps": 100, "service_cpu_query": "JVM: Process CPU usage by (application, instance)"}
    assert _runtime(env)["queries"]["jvm"] == jvm

    again = env.client.post("/config/query_packs/apply", json={"pack_id": "jvm-micrometer", "variables": {"matchers": 'namespace="shop"'}}).get_json()
    assert again["plan"]["added"] == [] and len(again["plan"]["unchanged"]) == 10 and again["sla_changed"] is False
    installed = _pack(env.client.get("/config/query_packs").get_json(), "jvm-micrometer")["installed"]
    assert installed["version"] == 1 and installed["outdated"] is False


def test_apply_selected_queries_fills_only_their_sla_fields(env):
    resp = env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-influxql", "include_indexes": [3, 9]})
    assert resp.status_code == 200, resp.get_json()
    assert core.CONFIG["queries"]["lt_framework"]["labels"] == [RPS_LABEL, VUS_LABEL]
    sla = core.CONFIG["sla"]
    assert sla["max_performance_query"] == RPS_LABEL and sla["vus_query"] == VUS_LABEL
    assert sla["load_model"] == "open"
    assert "latency_unit" not in sla and "p95_query" not in sla


def test_apply_with_another_language_needs_replace(env):
    assert env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-influxql"}).status_code == 200
    catalog = env.client.get("/config/query_packs").get_json()
    assert _pack(catalog, "lt-k6-prometheus-rw")["compatibility"]["status"] == "replace_only"
    refused = env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-prometheus-rw"})
    assert refused.status_code == 409 and "InfluxQL" in refused.get_json()["error"]
    replaced = env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-prometheus-rw", "mode": "replace", "overwrite_sla": True})
    assert replaced.status_code == 200
    data = replaced.get_json()
    assert RPS_LABEL in data["plan"]["removed"]
    block = core.CONFIG["queries"]["lt_framework"]
    assert "influxql_queries" not in block and block["packs"][0]["id"] == "lt-k6-prometheus-rw"
    assert core.CONFIG["sla"]["max_performance_query"] == "LT (k6 Prometheus): RPS total"


def test_apply_area_and_service_scopes(env):
    area = env.client.post("/config/query_packs/apply", json={"pack_id": "node-exporter", "area": "shop"})
    assert area.status_code == 200, area.get_json()
    assert area.get_json()["scope"] == "area" and area.get_json()["previous"]["own_queries"] is False
    runtime = _runtime(env)
    assert runtime["per_area"]["shop"]["queries"]["hard_resources"]["labels"][0] == "Nodes: CPU usage (%) by node"
    assert runtime["per_area"]["shop"]["sla"]["cpu_query"] == "Nodes: CPU usage (%) by node"
    assert core.CONFIG["queries"] == {}

    service = env.client.post("/config/query_packs/apply", json={"pack_id": "db-hikaricp", "area": "shop", "service": "checkout"})
    assert service.status_code == 200, service.get_json()
    assert service.get_json()["scope"] == "service"
    stored = _runtime(env)["per_area"]["shop"]["services"]["checkout"]["queries"]
    # The service override starts from the area value, as the settings page saves it.
    assert set(stored) == {"hard_resources", "database"}

    undo = env.client.delete("/projects/shop/services/checkout/overrides/queries")
    assert undo.status_code == 200
    assert "queries" not in _runtime(env)["per_area"]["shop"]["services"]["checkout"]


@pytest.mark.parametrize(
    ("body", "status", "fragment"),
    [
        ({"pack_id": "nope"}, 404, "не найден"),
        ({"pack_id": "jvm-micrometer", "area": "missing"}, 404, "Область"),
        ({"pack_id": "jvm-micrometer", "service": "checkout"}, 400, "вместе с областью"),
        ({"pack_id": "jvm-micrometer", "area": "shop", "service": "other"}, 404, "Сервис"),
        ({"pack_id": "jvm-micrometer", "variables": {"service_label": "a b"}}, 400, "не подходит"),
        ({"pack_id": "jvm-micrometer", "include_indexes": [99]}, 400, "вне пакета"),
        ({"pack_id": "jvm-micrometer", "include_indexes": []}, 400, "ни одного"),
        ({"pack_id": "jvm-micrometer", "mode": "upsert"}, 400, "mode"),
        ({"pack_id": "kafka-exporter"}, 409, "не выполняет"),
    ],
)
def test_apply_rejects_bad_requests(env, body, status, fragment):
    resp = env.client.post("/config/query_packs/apply", json=body)
    assert resp.status_code == status
    assert fragment in resp.get_json()["error"]
    assert core.CONFIG["queries"] == {}


def test_check_runs_selected_queries(env, monkeypatch):
    from loadlens_app import query_pack_service
    from loadlens_app.query_preview import QueryPreview

    calls = []

    def fake_preview(cfg, *, domain, lang, query, label_keys, start_ts, end_ts):
        calls.append((domain, lang, query, tuple(label_keys)))
        if "vus" in query:
            return QueryPreview(ok=False, message="За окно серий нет")
        return QueryPreview(ok=True, message="1 серий", series_count=1, point_count=10, series=["all"])

    monkeypatch.setattr(query_pack_service, "preview_metric_query", fake_preview)
    resp = env.client.post("/config/query_packs/check", json={"pack_id": "lt-k6-influxql", "include_indexes": [3, 9], "start": "2024-11-01T10:00", "end": "2024-11-01T11:00"})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert [r["label"] for r in data["results"]] == [RPS_LABEL, VUS_LABEL]
    assert [r["ok"] for r in data["results"]] == [True, False]
    assert data["ok"] is True and data["missing_required"] == []
    assert data["source"] == "Grafana"
    assert {call[:2] for call in calls} == {("lt_framework", "influxql")}
    assert core.CONFIG["queries"] == {}


def test_post_config_drops_roles_of_removed_rows(env):
    payload = {"jvm": {
        "labels": ["CPU"], "promql_queries": ["process_cpu_usage"], "label_keys_list": [["instance"]],
        "roles": {"CPU": "runtime.cpu", "Removed": "runtime.heap_used"},
        "packs": [{"id": "jvm-micrometer", "version": 1, "variables": {}}],
    }}
    resp = env.client.post("/config", json={"section": "queries", "data": payload})
    assert resp.status_code == 200, resp.get_json()
    jvm = core.CONFIG["queries"]["jvm"]
    assert jvm["roles"] == {"CPU": "runtime.cpu"}
    # No row of the pack is left, so it is no longer listed as installed.
    assert "packs" not in jvm


def _own_lt_block():
    return {"labels": ["My RPS"], "influxql_queries": ['SELECT sum("value") FROM "http_reqs" WHERE $timeFilter'], "label_tag_keys_list": [[]]}


def test_apply_replaces_sla_references_by_default(env):
    core.CONFIG["queries"] = {"lt_framework": _own_lt_block()}
    core.CONFIG["sla"] = {"target_rps": 100, "max_performance_query": "My RPS"}
    resp = env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-influxql"})
    assert resp.status_code == 200, resp.get_json()
    assert core.CONFIG["sla"]["max_performance_query"] == RPS_LABEL
    assert resp.get_json()["sla"]["changes"]["max_performance_query"] == {"from": "My RPS", "to": RPS_LABEL}


def test_apply_can_keep_live_sla_references(env):
    core.CONFIG["queries"] = {"lt_framework": _own_lt_block()}
    core.CONFIG["sla"] = {"target_rps": 100, "max_performance_query": "My RPS"}
    resp = env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-influxql", "overwrite_sla": False})
    assert resp.status_code == 200, resp.get_json()
    assert core.CONFIG["sla"]["max_performance_query"] == "My RPS"
    assert resp.get_json()["sla"]["kept"] == {"max_performance_query": "My RPS"}


def test_apply_with_replace_never_keeps_references_to_removed_queries(env):
    core.CONFIG["queries"] = {"lt_framework": _own_lt_block()}
    core.CONFIG["sla"] = {"target_rps": 100, "max_performance_query": "My RPS"}
    resp = env.client.post("/config/query_packs/apply", json={"pack_id": "lt-k6-influxql", "mode": "replace", "overwrite_sla": False})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert "My RPS" in data["plan"]["removed"]
    assert core.CONFIG["sla"]["max_performance_query"] == RPS_LABEL
    assert data["sla"]["changes"]["max_performance_query"]["reason"] == "missing"
