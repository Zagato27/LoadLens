"""Catalog, domain bindings and the legacy metrics_source conversion."""

import pandas as pd
import pytest

from AI import pipeline as pipeline_module
from AI.data_sources import (
    DataSourceConfigError,
    binding_errors,
    domain_query_language,
    resolve_domain_source,
    validate_data_sources,
    validate_domain_sources,
)
from loadlens_app.data_sources_migration import AreaLegacy, convert_legacy_sources


def _grafana(url, datasource=None, influx_ds=None, database="", bucket=""):
    grafana = {
        "base_url": url,
        "verify_ssl": False,
        "auth": {"method": "basic", "username": "admin", "password": "admin"},
    }
    if datasource:
        grafana["prometheus_datasource"] = datasource
    if influx_ds:
        grafana["influxdb_datasource"] = influx_ds
    section = {"type": "grafana_proxy", "grafana": grafana}
    influx = {}
    if database:
        influx["database"] = database
    if bucket:
        influx["bucket"] = bucket
    if influx:
        section["influxdb"] = influx
    return section


def _catalog():
    return {
        "grafana": {
            "title": "Grafana",
            "type": "grafana_proxy",
            "grafana": {
                "base_url": "http://grafana:3000",
                "verify_ssl": False,
                "auth": {"method": "basic", "username": "a", "password": "b"},
            },
        },
        "prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}},
    }


def test_same_grafana_becomes_one_source_and_two_bindings():
    metrics = _grafana("http://grafana:3000", {"name": "Prometheus"})
    load = _grafana("http://grafana:3000", {"name": "Prometheus"}, {"name": "InfluxDB-k6"}, database="k6", bucket="k6")
    result = convert_legacy_sources(metrics, load, None, {})
    assert list(result.catalog) == ["grafana"]
    assert result.catalog["grafana"]["grafana"]["base_url"] == "http://grafana:3000"
    assert "prometheus_datasource" not in result.catalog["grafana"]["grafana"]
    assert result.global_bindings["default"] == {"source": "grafana", "datasource_name": "Prometheus"}
    assert result.global_bindings["lt_framework"]["source"] == "grafana"
    assert result.global_bindings["lt_framework"]["datasource_name"] == "InfluxDB-k6"
    assert result.global_bindings["lt_framework"]["database"] == "k6"
    validate_data_sources(result.catalog)
    validate_domain_sources(result.global_bindings, result.catalog)


def test_area_grafana_becomes_second_source():
    metrics = _grafana("http://grafana:3000", {"name": "Prometheus"})
    other = _grafana("http://other:3000", {"uid": "other-prom"})
    result = convert_legacy_sources(metrics, None, None, {"demo": AreaLegacy(other, None, None)})
    assert set(result.catalog) == {"grafana", "grafana-2"}
    assert result.global_bindings["default"]["source"] == "grafana"
    assert "lt_framework" not in result.global_bindings
    assert result.area_bindings["demo"]["default"]["source"] == "grafana-2"
    assert result.area_bindings["demo"]["default"]["datasource_uid"] == "other-prom"


def test_empty_load_source_uses_default_binding():
    result = convert_legacy_sources(_grafana("http://grafana:3000", {"name": "Prometheus"}), {}, None, {})
    assert "lt_framework" not in result.global_bindings


def test_promql_only_load_keeps_prometheus_datasource():
    load = _grafana("http://grafana:3000", {"name": "Prometheus"}, {"name": "InfluxDB-k6"}, database="k6")
    queries = {"promql_queries": ["up"], "influxql_queries": [], "flux_queries": []}
    result = convert_legacy_sources(_grafana("http://grafana:3000", {"name": "Prometheus"}), load, queries, {})
    assert result.global_bindings["lt_framework"]["datasource_name"] == "Prometheus"


def test_numeric_datasource_id_is_not_migrated():
    result = convert_legacy_sources(_grafana("http://grafana:3000", {"id": 7}), None, None, {})
    assert any("id=7" in warning for warning in result.warnings)
    binding = result.global_bindings["default"]
    assert binding == {"source": "grafana"}


def test_unknown_legacy_type_is_skipped():
    result = convert_legacy_sources({"type": "graphite"}, None, None, {})
    assert result.catalog == {}
    assert result.global_bindings == {}
    assert result.warnings


def test_resolved_datasource_comes_only_from_the_binding():
    cfg = {
        "data_sources": _catalog(),
        "domain_sources": {
            "default": {"source": "grafana", "datasource_uid": "prom", "datasource_name": "Prom"},
            "kafka": {"source": "grafana", "datasource_uid": "kafka"},
        },
    }
    kafka = resolve_domain_source(cfg, "kafka")
    datasource = kafka.config["grafana"]["prometheus_datasource"]
    assert datasource == {"id": None, "uid": "kafka", "name": ""}
    assert kafka.config["grafana"]["influxdb_datasource"] == datasource
    assert resolve_domain_source(cfg, "jvm").config["grafana"]["prometheus_datasource"]["uid"] == "prom"
    assert kafka.prometheus_url == ""


def test_prometheus_source_rejects_flux_queries():
    resolved = resolve_domain_source(
        {"data_sources": _catalog(), "domain_sources": {"default": {"source": "prom"}}},
        "hard_resources",
    )
    with pytest.raises(DataSourceConfigError, match="flux"):
        domain_query_language(resolved, {"flux_queries": ['from(bucket: "telegraf")']})


def test_grafana_binding_requires_datasource():
    with pytest.raises(DataSourceConfigError, match="датасорс"):
        validate_domain_sources({"default": {"source": "grafana"}}, _catalog())


def test_removed_source_is_reported_with_its_area():
    errors = binding_errors(_catalog(), {"default": {"source": "prom"}}, {"demo": {"default": {"source": "missing", "datasource_name": "X"}}})
    assert "источник missing используется: область demo, домен default" in errors


def test_hard_resources_collects_flux_from_influx_source(monkeypatch):
    frame = pd.DataFrame({"host=a": [0.2]}, index=pd.date_range("2024-01-01", periods=1, freq="5min"))
    calls = {}

    def fake_influx(**kwargs):
        calls["queries"] = kwargs["flux_queries"]
        calls["bucket"] = kwargs["influx_cfg"].get("bucket")
        return [frame]

    def fake_prom(*_args, **_kwargs):
        raise AssertionError("PromQL не должен вызываться для InfluxDB")

    monkeypatch.setattr(pipeline_module, "fetch_influx_and_aggregate", fake_influx)
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", fake_prom)
    pipeline_module.uploadFromLLM(
        start_ts=1704099600,
        end_ts=1704103200,
        save_to_db=False,
        only_collect=True,
        active_domains=["hard_resources"],
        ef_config={
            "default_params": {"step": "1m", "resample_interval": "5T"},
            "data_sources": {
                "influx": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "ops", "token": "t"}},
            },
            "domain_sources": {"default": {"source": "influx", "bucket": "telegraf"}},
            "queries": {
                "hard_resources": {
                    "flux_queries": ['from(bucket: "{bucket}")'],
                    "label_tag_keys_list": [["host"]],
                    "labels": ["Nodes CPU"],
                },
            },
            "sla": {},
        },
    )
    assert calls["queries"] == ['from(bucket: "{bucket}")']
    assert calls["bucket"] == "telegraf"


def test_preview_rejects_promql_for_influx_domain():
    from loadlens_app.query_preview import preview_metric_query

    result = preview_metric_query(
        {
            "data_sources": {
                "influx": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "ops", "token": "t"}},
            },
            "domain_sources": {"default": {"source": "influx", "bucket": "telegraf"}},
        },
        domain="hard_resources",
        lang="promql",
        query="up",
    )
    assert result.ok is False
    assert result.message == "Этот источник выполняет только Flux"
