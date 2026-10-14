"""Query packs, phase 2: updates, domain titles, SLA references from roles, detection by source metrics."""

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
from AI.query_packs import (
    QueryPack,
    domain_title,
    domain_titles,
    load_catalog,
    merge_pack,
    render_pack,
    resolve_sla_refs,
    update_selection,
)
from loadlens_app import core
from loadlens_app.config_schema import QueryConfigError, validate_queries
from loadlens_app.metric_discovery import DomainNames

CATALOG = load_catalog()
DEFAULT_TITLES = {"jvm": "JVM", "lt_framework": "Нагрузочный инструмент", "hard_resources": "Ресурсы узлов"}


# ---- pack versions and updates ----------------------------------------------------------------------

def test_query_since_cannot_exceed_pack_version():
    data = CATALOG.pack("kafka-exporter").model_dump()
    data["queries"][0]["since"] = 2
    with pytest.raises(ValueError, match="newer than the pack version"):
        QueryPack.model_validate(data)


def test_update_selection_keeps_user_choice_and_offers_new_queries():
    pack = CATALOG.pack("http-micrometer")
    assert pack.version == 2
    rendered = render_pack(pack)
    v1_labels = [query.label for query in rendered.queries if query.since == 1]
    new_label = next(query.label for query in rendered.queries if query.since == 2)
    removed = v1_labels[-1]
    block, _ = merge_pack(None, render_pack(pack, include=set(v1_labels) - {removed}))
    selected = {rendered.queries[index].label for index in update_selection(block, rendered, installed_version=1)}
    assert new_label in selected and removed not in selected
    assert set(v1_labels) - {removed} <= selected
    # Opening an up-to-date pack again does not re-add a row the user removed.
    assert new_label not in {rendered.queries[i].label for i in update_selection(block, rendered, installed_version=2)}


# ---- domain titles ------------------------------------------------------------------------------------

def test_domain_titles_follow_installed_packs_and_own_title():
    go_block, _ = merge_pack(None, render_pack(CATALOG.pack("runtime-go")))
    assert domain_title(go_block) == "Go runtime"
    jvm_block, _ = merge_pack(None, render_pack(CATALOG.pack("jvm-micrometer")))
    mixed, _ = merge_pack(jvm_block, render_pack(CATALOG.pack("k8s-workload-cpu")))
    assert domain_title(mixed) == "JVM · Контейнеры"
    assert domain_titles({"jvm": jvm_block}, DEFAULT_TITLES) == {}
    assert domain_titles({"jvm": go_block}, DEFAULT_TITLES) == {"jvm": "Go runtime"}
    own = dict(go_block, title="Payments runtime")
    assert domain_titles({"jvm": own}, DEFAULT_TITLES) == {"jvm": "Payments runtime"}
    with pytest.raises(QueryConfigError, match="title"):
        validate_queries({"jvm": {"labels": [], "title": "x" * 61}})


def test_report_and_confluence_read_stored_titles():
    from AI.pipeline import _report_domain_titles
    from loadlens_app.confluence_export import report_domain_titles

    go_block, _ = merge_pack(None, render_pack(CATALOG.pack("runtime-go")))
    assert _report_domain_titles({"jvm": go_block}) == {"jvm": "Go runtime"}
    rows = [{"domain": "jvm", "parsed": {}}, {"domain": "final", "parsed": json.dumps({"domain_titles": {"jvm": "Go runtime"}})}]
    assert report_domain_titles(rows) == {"jvm": "Go runtime"}
    assert report_domain_titles([{"domain": "final", "parsed": {"verdict": "x"}}]) == {}


# ---- SLA references from roles -------------------------------------------------------------------------

def _with_row(block, label, query):
    """The block with one hand-made PromQL row appended (no role)."""
    block = dict(block)
    block["labels"] = list(block["labels"]) + [label]
    block["promql_queries"] = list(block["promql_queries"]) + [query]
    block["label_keys_list"] = list(block["label_keys_list"]) + [[]]
    return block


def test_resolve_sla_refs_fills_empty_and_dead_fields_and_keeps_live_ones():
    lt = _with_row(merge_pack(None, render_pack(CATALOG.pack("lt-k6-prometheus-rw")))[0], "My RPS", "sum(rate(x[1m]))")
    lt = _with_row(lt, "My mean (s)", "avg(y)")
    queries = {"lt_framework": lt, "hard_resources": merge_pack(None, render_pack(CATALOG.pack("node-exporter")))[0]}

    sla, filled = resolve_sla_refs({"max_performance_query": "My RPS", "latency_unit": "s", "target_rps": 50}, queries)
    assert sla["max_performance_query"] == "My RPS"  # a live explicit reference wins
    assert sla["p95_query"] == "LT (k6 Prometheus): http_req_duration p95(ms) worst"
    assert sla["cpu_query"] == "Nodes: CPU usage (%) by node"
    assert sla["latency_unit"] == "ms"  # every latency reference now comes from the pack
    assert "max_performance_query" not in filled and filled["vus_query"] == "LT (k6 Prometheus): VUs"

    explicit, _ = resolve_sla_refs({"mean_latency_query": "My mean (s)", "latency_unit": "s"}, queries)
    assert explicit["latency_unit"] == "s" and explicit["p95_query"].endswith("worst")

    # A reference to a query that no longer exists is taken from the role like an empty one.
    dead, filled = resolve_sla_refs({"max_performance_query": "Old RPS", "mean_latency_query": "Old mean", "latency_unit": "s"}, queries)
    assert dead["max_performance_query"] == filled["max_performance_query"] == "LT (k6 Prometheus): RPS total"
    assert dead["mean_latency_query"].startswith("LT (k6 Prometheus)") and dead["latency_unit"] == "ms"
    # Without a role query of that kind the dead reference stays as it was.
    orphan, _ = resolve_sla_refs({"cpu_query": "Old CPU"}, {"lt_framework": lt})
    assert orphan["cpu_query"] == "Old CPU"


def test_pipeline_forecast_and_readiness_use_role_refs(monkeypatch):
    from AI.pipeline import _sla_with_role_refs
    from loadlens_app import forecast, projects

    queries = {"jvm": merge_pack(None, render_pack(CATALOG.pack("jvm-micrometer")))[0]}
    assert _sla_with_role_refs({}, queries)["service_cpu_query"] == "JVM: Process CPU usage by (application, instance)"
    assert forecast._resolved_sla({"sla": {}, "queries": queries})["service_cpu_query"].startswith("JVM: Process CPU")
    monkeypatch.setitem(projects.CONFIG, "queries", {})
    monkeypatch.setitem(projects.CONFIG, "sla", {})
    lt = merge_pack(None, render_pack(CATALOG.pack("lt-k6-influxql")))[0]
    readiness = projects._readiness({"queries": {"lt_framework": lt}, "sla": {}})
    assert readiness.performance_query == "LT (InfluxQL): RPS sum by all groups"
    assert readiness.missing_sla["forecast"] == []


# ---- detection ----------------------------------------------------------------------------------------------

def _fake_discover(names_by_domain, errors=None, language="promql"):
    def discover(cfg, domains):
        out = {}
        for domain in domains:
            if domain in (errors or {}):
                out[domain] = DomainNames(domain, "Prometheus", "", error=errors[domain])
            else:
                lang = "influxql" if domain == "lt_framework" else language
                out[domain] = DomainNames(domain, "Prometheus", lang, frozenset(names_by_domain.get(domain, ())))
        return out
    return discover


def test_detect_scores_packs_and_bundles():
    from loadlens_app.query_pack_service import ScopeState, detect_packs

    names = {
        "jvm": {"jvm_memory_used_bytes", "process_cpu_usage", "jvm_gc_pause_seconds_count"},
        "lt_framework": {"http_reqs", "http_req_duration", "http_req_failed", "vus"},
        "hard_resources": {"node_cpu_seconds_total", "node_memory_MemAvailable_bytes"},
    }
    state = ScopeState(queries={}, sla={}, effective_cfg={})
    result = detect_packs(state, _fake_discover(names, errors={"kafka": "источник ответил 503"}))
    packs = result["packs"]
    assert packs["jvm-micrometer"]["score"] == 1.0
    assert packs["jvm-otel"]["found"] == ["jvm_memory_used_bytes"] and packs["jvm-otel"]["score"] < 1
    assert packs["lt-k6-influxql"]["score"] == 1.0
    # A PromQL pack does not match measurement names of an InfluxQL source.
    assert packs["lt-k6-prometheus-rw"]["score"] == 0.0
    assert packs["node-exporter"]["missing"] == ["node_uname_info"]
    assert result["domains"]["kafka"]["error"] == "источник ответил 503"
    bundle = result["bundles"]["spring-k8s-k6-influxdb"]
    assert bundle["total"] == 9 and bundle["matched"] == 2 and 0 < bundle["score"] < 1


def test_discovery_reads_each_source_once(monkeypatch):
    from loadlens_app import metric_discovery

    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "success", "data": ["up", "node_cpu_seconds_total"]}

    def fake_get(url, **_kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(metric_discovery.requests, "get", fake_get)
    monkeypatch.setattr(metric_discovery, "_influxql_measurements", lambda source: frozenset({"http_reqs"}))
    monkeypatch.setattr(metric_discovery, "_grafana_mode", lambda g_cfg: "influxql")
    cfg = {
        "data_sources": {
            "prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}},
            "graf": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://g", "auth": {}}},
        },
        "domain_sources": {"default": {"source": "prom"}, "lt_framework": {"source": "graf", "datasource_uid": "x", "database": "k6"}},
    }
    found = metric_discovery.discover(cfg, ["jvm", "database", "hard_resources", "lt_framework"])
    assert calls == ["http://prom:9090/api/v1/label/__name__/values"]
    assert found["jvm"].names == frozenset({"up", "node_cpu_seconds_total"}) and found["jvm"].language == "promql"
    assert found["lt_framework"].names == frozenset({"http_reqs"}) and found["lt_framework"].source == "Grafana"


def test_discovery_reports_errors_per_domain(monkeypatch):
    from loadlens_app import metric_discovery

    def broken(url, **_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(metric_discovery.requests, "get", broken)
    cfg = {
        "data_sources": {"prom": {"title": "P", "type": "prometheus", "prometheus": {"url": "http://prom"}}},
        "domain_sources": {"default": {"source": "prom"}, "kafka": {"source": "missing"}},
    }
    found = metric_discovery.discover(cfg, ["jvm", "kafka"])
    assert found["jvm"].error == "connection refused" and found["jvm"].source == "P"
    assert "missing" in found["kafka"].error and not found["kafka"].names


# ---- endpoints -------------------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    sources = {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}}}
    for key, value in {"queries": {}, "sla": {}, "per_area": {}, "data_sources": sources, "domain_sources": {"default": {"source": "prom"}}}.items():
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(value))
    from loadlens_app.blueprints import query_packs as blueprint

    monkeypatch.setattr(blueprint, "_effective_cfg", lambda area, service: {
        "data_sources": core.CONFIG["data_sources"], "domain_sources": core.CONFIG["domain_sources"],
    })
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_detect_endpoint(client, monkeypatch):
    from loadlens_app import metric_discovery

    monkeypatch.setattr(metric_discovery, "discover", _fake_discover({"jvm": {"go_goroutines", "go_memstats_heap_alloc_bytes", "process_cpu_seconds_total"}}))
    resp = client.post("/config/query_packs/detect", json={})
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["packs"]["runtime-go"]["score"] == 1.0


def test_catalog_marks_outdated_pack_and_preselects_rows(client):
    assert client.post("/config/query_packs/apply", json={"pack_id": "http-micrometer"}).status_code == 200
    block = core.CONFIG["queries"]["microservices"]
    new_label = "Microservices: p95 request time (sec) by uri (top 20)"
    # Simulate an install of v1: the v2 row is absent and the stored version is 1.
    index = block["labels"].index(new_label)
    for key in ("labels", "promql_queries", "label_keys_list"):
        block[key].pop(index)
    block["roles"].pop(new_label)
    block["packs"][0]["version"] = 1
    data = client.get("/config/query_packs").get_json()
    pack = next(item for item in data["packs"] if item["id"] == "http-micrometer")
    assert pack["installed"]["outdated"] is True and pack["installed"]["new_queries"] == 1
    labels = [query["label"] for query in pack["queries"]]
    assert labels.index(new_label) in pack["installed"]["selected_indexes"]
    assert len(pack["installed"]["selected_indexes"]) == len(labels)
    assert data["roles"]["runtime.cpu"]["sla_field"] == "service_cpu_query"
    assert next(item for item in data["packs"] if item["id"] == "runtime-go")["domain_title"] == "Go runtime"
