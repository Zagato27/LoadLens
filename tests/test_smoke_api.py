import copy
import json
import sys
import threading
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AI import pipeline as pipeline_module
from loadlens_app import core
from update_page import _scale_pipeline_percent, _span_percent
from AI.pipeline import (
    _find_stable_peak_step_profile,
    _has_meaningful_system_context,
    _reconcile_sla_for_test_profile,
    _select_step_profile_candidate,
)
from AI.scoring import parse_llm_analysis_strict


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return None

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return FakeCursor()

    def close(self):
        return None

    def rollback(self):
        return None

    def commit(self):
        return None


class RowsCursor(FakeCursor):
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class RowsConnection(FakeConnection):
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return RowsCursor(self._rows)


class RecordingCursor(FakeCursor):
    """Records executed statements and serves scripted fetch results in order."""

    rowcount = 0

    def __init__(self, log, fetchone_results=None, fetchall_results=None):
        self._log = log
        self._fetchone = list(fetchone_results or [])
        self._fetchall = list(fetchall_results or [])

    def execute(self, query, params=None):
        self._log.append((" ".join(str(query).split()), params))

    def fetchone(self):
        return self._fetchone.pop(0) if self._fetchone else None

    def fetchall(self):
        return self._fetchall.pop(0) if self._fetchall else []


class RecordingConnection(FakeConnection):
    def __init__(self, fetchone_results=None, fetchall_results=None):
        self.log = []
        self._cursor = RecordingCursor(self.log, fetchone_results, fetchall_results)

    def cursor(self):
        return self._cursor


@pytest.fixture(autouse=True)
def patch_core(monkeypatch, tmp_path):
    runtime_path = tmp_path / "settings_runtime.json"
    metrics_runtime_path = tmp_path / "metrics_config_runtime.json"
    original_system_context = copy.deepcopy(core.CONFIG.get("system_context"))
    monkeypatch.setattr(core, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime_path)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", metrics_runtime_path)
    core.CONFIG["system_context"] = copy.deepcopy(original_system_context or {})
    yield
    core.CONFIG["system_context"] = original_system_context


@pytest.fixture
def client(monkeypatch):
    from app import create_app
    from loadlens_app import jobs
    from loadlens_app.blueprints import compare, config_api, dashboard

    def _sync_thread(target, **kwargs):
        target()
        return types.SimpleNamespace(start=lambda: None)

    for module in (dashboard, compare, jobs):
        monkeypatch.setattr(module, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_ensure_engineer_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_metrics_service_entry", lambda service: ("demo", {"page_sample_id": "1", "page_parent_id": "1", "metrics": [], "logs": []}))
    monkeypatch.setattr(dashboard, "_find_area_for_service", lambda service: "demo")
    monkeypatch.setattr(dashboard, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(compare, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(config_api, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(dashboard, "update_report", lambda *args, **kwargs: {"page_id": "1", "page_url": "/reports/demo/test-run", "run_name": "test-run"})
    monkeypatch.setattr(dashboard, "load_publication", lambda run_name: None)
    monkeypatch.setattr(
        dashboard,
        "publish_report",
        lambda **kwargs: {"page_id": "42", "page_url": "https://confluence.example.ru/pages/viewpage.action?pageId=42", "title": "LoadLens: test-run"},
    )
    monkeypatch.setattr(threading, "Thread", lambda target, daemon=True: _sync_thread(target=target))

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ---- config -------------------------------------------------------------------------

def test_get_config_endpoint(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "areas" in data
    assert "system_context" in data
    assert "confluence" in data and "confluence_template" in data
    assert "logs_source" in data
    assert isinstance(data["services"], list)


def test_query_preview_rejects_promql_for_influx_domain(client, monkeypatch):
    monkeypatch.setitem(core.CONFIG, "data_sources", {
        "influx": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "ops", "token": "t"}},
    })
    monkeypatch.setitem(core.CONFIG, "domain_sources", {"default": {"source": "influx", "bucket": "telegraf"}})
    resp = client.post("/config/query_preview", json={"domain": "hard_resources", "lang": "promql", "query": "up"})
    assert resp.status_code == 400
    assert resp.get_json()["message"] == "Этот источник выполняет только Flux"


def test_query_preview_rejects_empty_query(client):
    resp = client.post("/config/query_preview", json={"domain": "jvm", "lang": "promql", "query": "  "})
    assert resp.status_code == 400
    assert resp.get_json()["message"] == "Запрос пустой"


def test_query_preview_counts_prometheus_series(monkeypatch):
    from loadlens_app.query_preview import preview_metric_query

    def fake_fetch(*_args, **_kwargs):
        return {
            "status": "success",
            "data": {"result": [
                {"metric": {"job": "api"}, "values": [[1, "1"], [2, "2"]]},
                {"metric": {"job": "web"}, "values": [[1, "3"]]},
            ]},
        }

    monkeypatch.setattr("loadlens_app.query_preview.fetch_metric_series", fake_fetch)
    result = preview_metric_query(
        {
            "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom"}}},
            "domain_sources": {"default": {"source": "prom"}},
            "default_params": {"step": "30s"},
        },
        domain="jvm",
        lang="promql",
        query="up",
        label_keys=["job"],
    )
    assert result.ok is True
    assert result.series_count == 2
    assert result.point_count == 3
    assert result.series == ["job=api", "job=web"]


def test_query_preview_reports_source_error(monkeypatch):
    from loadlens_app.query_preview import preview_metric_query

    def fake_fetch(*_args, **_kwargs):
        response = requests.Response()
        response.status_code = 400
        response._content = b"bad_data"
        raise requests.HTTPError(response=response)

    monkeypatch.setattr("loadlens_app.query_preview.fetch_metric_series", fake_fetch)
    result = preview_metric_query(
        {
            "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom"}}},
            "domain_sources": {"default": {"source": "prom"}},
            "default_params": {"step": "30s"},
        },
        domain="jvm",
        lang="promql",
        query="up{",
        label_keys=[],
    )
    assert result.ok is False
    assert "400" in result.message
    assert "bad_data" in result.message


def test_get_config_masks_secrets(client, monkeypatch):
    monkeypatch.setitem(core.CONFIG["storage"]["timescale"], "password", "db-secret")
    monkeypatch.setitem(core.CONFIG["confluence"], "password", "cf-secret")
    monkeypatch.setitem(core.CONFIG, "password", "template-secret")
    monkeypatch.setitem(core.CONFIG, "data_sources", {
        "grafana": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "basic", "username": "a", "password": "graf-secret"}}},
    })
    data = client.get("/config").get_json()
    assert data["storage"]["timescale"]["password"] == "***"
    assert data["confluence"]["password"] == "***"
    assert data["confluence_template"]["password"] == "***"
    assert data["data_sources"]["grafana"]["grafana"]["auth"]["password"] == "***"
    dumped = json.dumps(data)
    assert "db-secret" not in dumped and "cf-secret" not in dumped and "template-secret" not in dumped and "graf-secret" not in dumped


def test_unknown_key_check_accepts_keys_shipped_in_example(tmp_path, monkeypatch):
    from loadlens_app import config_schema

    stale = tmp_path / "settings.py"
    stale.write_text("CONFIG = {'sla': {'target_rps': 100}}\n", encoding="utf-8")
    monkeypatch.setattr(config_schema, "SETTINGS_PATH", stale)
    config_schema.default_config.cache_clear()
    try:
        assert config_schema.warnings_for("sla", {"step_detection_enabled": True}) == []
        assert any("sla.max_p95_m" in w for w in config_schema.warnings_for("sla", {"max_p95_m": 1}))
    finally:
        config_schema.default_config.cache_clear()


def test_post_config_restores_secret_and_reports_unknown_keys(client, monkeypatch):
    # POST /config replaces whole section objects; restore them after the test.
    monkeypatch.setitem(core.CONFIG, "confluence", copy.deepcopy(core.CONFIG["confluence"]))
    monkeypatch.setitem(core.CONFIG, "sla", copy.deepcopy(core.CONFIG["sla"]))
    monkeypatch.setitem(core.CONFIG["confluence"], "password", "cf-secret")
    resp = client.post("/config", json={"section": "confluence", "data": {"base_url": "https://wiki.local", "password": "***", "spase_key": "LL"}})
    assert resp.status_code == 200
    body = resp.get_json()
    assert core.CONFIG["confluence"]["password"] == "cf-secret"
    assert core.CONFIG["confluence"]["base_url"] == "https://wiki.local"
    assert any("confluence.spase_key" in w for w in body["warnings"])

    resp = client.post("/config", json={"section": "sla", "data": {"target_rps": 200, "max_p95_m": 300}})
    assert resp.status_code == 200
    assert any("sla.max_p95_m" in w and "max_p95_ms" in w for w in resp.get_json()["warnings"])

    monkeypatch.setitem(core.CONFIG["confluence"], "password", "")
    resp = client.post("/config", json={"section": "confluence", "data": {"password": "***"}})
    assert resp.status_code == 400
    assert "не задан" in resp.get_json()["error"]


def test_post_config_confluence_template_updates_flat_keys(client, monkeypatch):
    monkeypatch.setitem(core.CONFIG, "url_basic", "https://old.local")
    monkeypatch.setitem(core.CONFIG, "password", "old-secret")
    resp = client.post("/config", json={"section": "confluence_template", "data": {"url_basic": "https://new.local", "password": "***", "unknown_key": 1}})
    assert resp.status_code == 200
    assert core.CONFIG["url_basic"] == "https://new.local"
    assert core.CONFIG["password"] == "old-secret"
    assert "unknown_key" not in core.CONFIG


def test_post_config_area_override_is_scoped(client):
    resp = client.post("/config", json={"section": "sla", "area": "demo", "data": {"target_rps": 555}})
    assert resp.status_code == 200
    assert resp.get_json()["scope"] == "area"
    runtime = json.loads(core.CONFIG_RUNTIME_PATH.read_text(encoding="utf-8"))
    assert runtime["per_area"]["demo"]["sla"]["target_rps"] == 555
    assert client.get("/config", query_string={"area": "demo"}).get_json()["sla"]["target_rps"] == 555


def test_test_connection_endpoint(client, monkeypatch):
    from loadlens_app.blueprints import config_api
    from loadlens_app.connection_checks import CheckResult

    calls = {}

    def fake_run_check(section, data):
        calls["section"] = section
        calls["data"] = data
        return CheckResult(ok=True, message="Подключение успешно", details={"version": "16"}, elapsed_ms=12)

    monkeypatch.setattr(config_api, "run_check", fake_run_check)
    monkeypatch.setitem(core.CONFIG["storage"]["timescale"], "password", "db-secret")
    resp = client.post("/config/test_connection", json={"section": "storage.timescale", "data": {"host": "db.local", "password": "***"}})
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    assert calls["section"] == "storage.timescale"
    assert calls["data"]["password"] == "db-secret"

    resp = client.post("/config/test_connection", json={"section": "queries", "data": {}})
    assert resp.status_code == 400


def test_catalog_save_rejects_source_still_in_use(client):
    core.CONFIG_RUNTIME_PATH.write_text(json.dumps({
        "domain_sources": {"default": {"source": "grafana", "datasource_name": "Prom"}},
    }, ensure_ascii=False), encoding="utf-8")
    resp = client.post("/config", json={"section": "data_sources", "data": {
        "other": {"title": "Other", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}},
    }})
    assert resp.status_code == 400
    assert "источник grafana используется: глобальные привязки, домен default" in resp.get_json()["error"]


def test_source_check_restores_secret(client, monkeypatch):
    from loadlens_app.blueprints import config_api
    from loadlens_app.connection_checks import CheckResult

    monkeypatch.setitem(core.CONFIG, "data_sources", {
        "grafana": {
            "title": "Grafana",
            "type": "grafana_proxy",
            "grafana": {"base_url": "http://grafana:3000", "verify_ssl": False, "auth": {"method": "basic", "username": "a", "password": "secret"}},
        },
    })
    calls = {}

    def fake_check(entry):
        calls["entry"] = entry
        return CheckResult(ok=True, message="ok", elapsed_ms=1)

    monkeypatch.setattr(config_api, "check_data_source", fake_check)
    resp = client.post("/config/test_connection", json={
        "section": "data_sources",
        "source": "grafana",
        "data": {"grafana": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "verify_ssl": False, "auth": {"method": "basic", "username": "a", "password": "***"}}}},
    })
    assert resp.status_code == 200
    assert calls["entry"]["grafana"]["auth"]["password"] == "secret"


def test_create_app_migrates_legacy_sources(monkeypatch, tmp_path):
    from app import create_app

    runtime_path = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime_path)
    saved = {key: copy.deepcopy(core.CONFIG.get(key)) for key in ("metrics_source", "lt_metrics_source", "data_sources", "domain_sources")}
    try:
        core.CONFIG.pop("data_sources", None)
        core.CONFIG.pop("domain_sources", None)
        core.CONFIG["metrics_source"] = {
            "type": "grafana_proxy",
            "grafana": {
                "base_url": "http://grafana:3000",
                "verify_ssl": False,
                "auth": {"method": "basic", "username": "admin", "password": "admin"},
                "prometheus_datasource": {"name": "Prometheus"},
            },
        }
        core.CONFIG["lt_metrics_source"] = {}
        runtime_path.write_text(json.dumps({"per_area": {"demo": {"metrics_source": {
            "type": "prometheus",
            "prometheus": {"url": "http://other:9090"},
        }}}}, ensure_ascii=False), encoding="utf-8")
        create_app()
        assert (tmp_path / "settings_runtime.before-data-sources.json").is_file()
        migrated = json.loads(runtime_path.read_text(encoding="utf-8"))
        assert "metrics_source" not in migrated and "lt_metrics_source" not in migrated
        assert "grafana" in migrated["data_sources"] and "prometheus" in migrated["data_sources"]
        assert "metrics_source" not in migrated["per_area"]["demo"]
        assert migrated["per_area"]["demo"]["domain_sources"]["default"]["source"] == "prometheus"
    finally:
        for key, value in saved.items():
            if value is None:
                core.CONFIG.pop(key, None)
            else:
                core.CONFIG[key] = value


def test_anthropic_gateway_lists_models_from_openai_catalog(monkeypatch):
    from loadlens_app.connection_checks import list_llm_models

    calls = []

    class _Response:
        def __init__(self, status, body):
            self.status_code = status
            self.ok = status == 200
            self._body = body

        def json(self):
            return self._body

    def fake_get(url, headers=None, auth=None, timeout=None, verify=True):
        calls.append(url)
        if url.endswith("/anthropic/v1/models"):
            return _Response(404, {})
        return _Response(200, {"data": [{"id": "mimo-v2.5-pro"}, {"id": "mimo-v2.5"}]})

    monkeypatch.setattr("loadlens_app.connection_checks.requests.get", fake_get)
    result = list_llm_models({
        "provider": "anthropic",
        "anthropic": {"api_base_url": "https://api.example.com/anthropic", "api_key": "k"},
    })
    assert result.ok is True
    assert result.models == ["mimo-v2.5", "mimo-v2.5-pro"]
    assert calls == [
        "https://api.example.com/anthropic/v1/models",
        "https://api.example.com/v1/models",
    ]


def test_llm_models_endpoint(client, monkeypatch):
    from loadlens_app.connection_checks import ModelListResult

    monkeypatch.setattr(
        "loadlens_app.connection_checks.list_llm_models",
        lambda _cfg: ModelListResult(ok=True, message="Доступно моделей: 1", models=["MiniMax-M2"], elapsed_ms=5),
    )
    resp = client.post("/config/llm_models", json={"section": "llm", "data": {"provider": "anthropic", "anthropic": {"model": "x"}}})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["models"] == ["MiniMax-M2"]


def test_queries_config_rejects_mismatched_arrays(client):
    resp = client.post("/config", json={"section": "queries", "data": {"jvm": {"labels": ["a"], "promql_queries": ["q1", "q2"], "label_keys_list": [["app"]]}}})
    assert resp.status_code == 400
    assert "разной длины" in resp.get_json()["error"]


def test_mask_and_restore_secrets():
    from loadlens_app.config_secrets import MASK, SecretPlaceholderError, mask_secrets, restore_secrets

    stored = {"grafana": {"auth": {"username": "admin", "password": "pw-1", "token": ""}}, "items": [{"token": "t-1"}]}
    masked = mask_secrets(stored)
    assert masked["grafana"]["auth"]["password"] == MASK
    assert masked["grafana"]["auth"]["username"] == "admin"
    assert masked["grafana"]["auth"]["token"] == ""
    assert masked["items"][0]["token"] == MASK

    incoming = {"grafana": {"auth": {"username": "root", "password": MASK}}, "items": [{"token": MASK}]}
    restored = restore_secrets(incoming, stored)
    assert restored["grafana"]["auth"] == {"username": "root", "password": "pw-1"}
    assert restored["items"][0]["token"] == "t-1"

    with pytest.raises(SecretPlaceholderError) as exc:
        restore_secrets({"confluence": {"password": MASK}}, {"confluence": {}})
    assert "confluence.password" in str(exc.value)


def test_update_system_context_via_config(client):
    payload = {
        "schema_version": 1,
        "system": {
            "name": "Checkout Platform",
            "domain": "e-commerce",
            "description": "Обработка заказов и оплат",
            "test_goal": "Проверить стабильность checkout",
        },
        "architecture": {
            "style": "microservices",
            "components": [
                {"id": "gateway", "name": "API Gateway", "role": "Внешняя точка входа", "criticality": "high", "technologies": ["nginx", "spring"]}
            ],
            "dependencies": [{"from": "gateway", "to": "orders", "kind": "sync_http", "purpose": "Маршрутизация запросов"}],
            "data_stores": [],
        },
        "load_model": {
            "entrypoints": [{"id": "checkout", "name": "POST /checkout", "kind": "http", "business_priority": "high"}],
            "critical_user_flows": [{"id": "checkout", "name": "Оформление заказа", "steps": ["gateway", "orders"], "success_signals": ["2xx"]}],
            "expected_hotspots": ["gateway", "orders"],
        },
        "operational_context": {
            "known_constraints": ["Общий Redis"],
            "known_risks": ["Рост latency при shared DB"],
            "normal_degradation_rules": [],
            "analysis_focus": ["Проверить p95 checkout"],
        },
    }
    resp = client.post("/config", json={"section": "system_context", "data": payload})
    assert resp.status_code == 200

    data = client.get("/config").get_json()
    assert data["system_context"]["system"]["name"] == "Checkout Platform"
    assert data["system_context"]["architecture"]["components"][0]["name"] == "API Gateway"


def test_system_context_can_be_disabled(client):
    payload = {"enabled": False, "schema_version": 1, "system": {"name": "Disabled Context Example"}}
    resp = client.post("/config", json={"section": "system_context", "data": payload})
    assert resp.status_code == 200

    data = client.get("/config").get_json()
    assert data["system_context"]["enabled"] is False
    assert data["system_context"]["system"]["name"] == "Disabled Context Example"
    assert _has_meaningful_system_context(data["system_context"]) is False


# ---- runs, jobs, reports -------------------------------------------------------------

def test_runs_endpoint_smoke(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)
    assert resp.headers["X-Total-Count"] == "0"


def test_runs_applies_filters_and_reports_total(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    end_time = datetime(2024, 11, 1, 11, 0, tzinfo=timezone.utc)
    start_time = datetime(2024, 11, 1, 10, 0, tzinfo=timezone.utc)
    page_rows = [("nightly-1", start_time, end_time, "demo", "Провал", "Есть риски", "Провал", end_time, "step", "rid-1")]
    conn = RecordingConnection(fetchone_results=[(37,)], fetchall_results=[page_rows])
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)

    resp = client.get("/runs", query_string={
        "q": "night", "service": "demo", "verdict": "Провал", "test_type": "step", "offset": 20, "limit": 20,
    })

    assert resp.status_code == 200
    assert resp.headers["X-Total-Count"] == "37"
    body = resp.get_json()
    assert body[0]["run_name"] == "nightly-1"
    assert body[0]["verdict"] == "Провал"
    assert body[0]["test_type"] == "step"
    assert body[0]["run_id"] == "rid-1"

    count_sql, count_params = conn.log[0]
    assert "COUNT(*)" in count_sql
    assert "service = %s AND verdict = %s AND test_type = %s" in count_sql
    assert count_params == ("%night%", "%night%", "demo", "Провал", "step")
    page_sql, page_params = conn.log[1]
    assert "OFFSET %s LIMIT %s" in page_sql
    assert page_params == ("%night%", "%night%", "demo", "Провал", "step", 20, 20)


def test_runs_sorts_by_report_created_at(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    conn = RecordingConnection(fetchone_results=[(0,)], fetchall_results=[[]])
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)

    resp = client.get("/runs", query_string={"sort": "report_created_at", "dir": "desc"})

    assert resp.status_code == 200
    page_sql, _params = conn.log[1]
    assert "ORDER BY report_created_at DESC NULLS LAST" in page_sql


def test_report_address_uses_run_id(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    def lookup(run_id="", run_name="", service=""):
        if run_id == "abc123":
            return {"run_id": "abc123", "run_name": "ODP", "service": "НСИ"}
        if run_name:
            return {"run_id": "abc123", "run_name": run_name, "service": service or "НСИ"}
        return None

    monkeypatch.setattr(dashboard, "_lookup_report_ref", lookup)

    legacy = client.get("/reports/НСИ/ODP")
    assert legacy.status_code == 302
    assert legacy.headers["Location"].endswith("/reports/abc123")

    page = client.get("/reports/abc123")
    assert page.status_code == 200

    ref = client.get("/report_ref/abc123")
    assert ref.status_code == 200
    assert ref.get_json() == {"run_id": "abc123", "run_name": "ODP", "service": "НСИ"}
    assert client.get("/report_ref/missing").status_code == 404


def test_runs_restricts_to_project_area_services(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    conn = RecordingConnection(fetchone_results=[(1,)], fetchall_results=[[]])
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: ["svc-a", "svc-b"])

    resp = client.get("/runs")
    assert resp.status_code == 200
    count_sql, count_params = conn.log[0]
    assert count_sql.count("service = ANY(%s)") == 2
    assert count_params == (["svc-a", "svc-b"], ["svc-a", "svc-b"])


def test_rename_run_endpoint_updates_report_name(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    captured = {}

    def fake_rename(old_name, new_name):
        captured["old_name"] = old_name
        captured["new_name"] = new_name
        return {"status": "ok", "renamed": 3, "run_name": new_name, "old_run_name": old_name}

    monkeypatch.setattr(dashboard, "_rename_run_data", fake_rename)

    resp = client.patch("/runs/old-report", json={"new_run_name": "new-report", "service": "demo"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["run_name"] == "new-report"
    assert body["page_url"] == "/reports/demo/new-report"
    assert captured == {"old_name": "old-report", "new_name": "new-report"}


def test_rename_run_endpoint_rejects_empty_name(client):
    resp = client.patch("/runs/old-report", json={"new_run_name": "   "})
    assert resp.status_code == 400
    assert "Новое имя" in resp.get_json()["error"]


def test_rename_run_endpoint_maps_conflicts(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    def _conflict(_old, _new):
        raise FileExistsError("Отчёт 'new-run' уже существует")

    monkeypatch.setattr(dashboard, "_rename_run_data", _conflict)
    resp = client.patch("/runs/old-run", json={"new_run_name": "new-run"})
    assert resp.status_code == 409
    assert "уже существует" in resp.get_json()["error"]


def test_rename_run_data_updates_tables(monkeypatch):
    store = {"names": {"old-run"}, "updates": []}

    class RenameCursor(FakeCursor):
        def execute(self, query, params=None):
            # Queries are psycopg2 Composed objects; their repr still contains the SQL keywords.
            upper = " ".join(str(query).split()).upper()
            if "SELECT 1 FROM" in upper and params:
                self._hit = params[0] in store["names"]
            elif "UPDATE" in upper and params:
                new_name, old_name = params
                store["updates"].append((old_name, new_name))
                if old_name in store["names"]:
                    store["names"].discard(old_name)
                    store["names"].add(new_name)

        def fetchone(self):
            return (1,) if getattr(self, "_hit", False) else None

    class RenameConnection(FakeConnection):
        def cursor(self):
            return RenameCursor()

    monkeypatch.setattr(core, "_ts_conn", lambda: RenameConnection())
    result = core._rename_run_data("old-run", "new-run")
    assert result["run_name"] == "new-run"
    assert store["names"] == {"new-run"}
    # metrics, llm_reports, engineer_reports, confluence_publications and llm_feedback.
    assert store["updates"] == [("old-run", "new-run")] * 5


def test_rename_run_data_conflict(monkeypatch):
    store = {"names": {"old-run", "taken"}, "updates": []}

    class RenameCursor(FakeCursor):
        def execute(self, query, params=None):
            upper = " ".join(str(query).split()).upper()
            if "SELECT 1 FROM" in upper and params:
                self._hit = params[0] in store["names"]
            elif "UPDATE" in upper and params:
                store["updates"].append(params)

        def fetchone(self):
            return (1,) if getattr(self, "_hit", False) else None

    class RenameConnection(FakeConnection):
        def cursor(self):
            return RenameCursor()

    monkeypatch.setattr(core, "_ts_conn", lambda: RenameConnection())
    with pytest.raises(FileExistsError):
        core._rename_run_data("old-run", "taken")
    assert store["updates"] == []


def test_convert_to_timestamp_accepts_offset_and_legacy_formats():
    # 2025-01-15T10:00:00+03:00 == 2025-01-15T07:00:00Z
    assert core.convert_to_timestamp("2025-01-15T10:00:00+03:00") == 1736924400000
    assert core.convert_to_timestamp("2025-01-15T07:00:00Z") == 1736924400000
    legacy_local = datetime(2025, 1, 15, 10, 0)
    assert core.convert_to_timestamp("2025-01-15T10:00") == int(legacy_local.timestamp() * 1000)
    with pytest.raises(ValueError):
        core.convert_to_timestamp("15.01.2025 10:00")
    with pytest.raises(ValueError):
        core.convert_to_timestamp("")


def test_create_report_smoke(client):
    payload = {
        "start": "2024-11-01T10:00",
        "end": "2024-11-01T11:00",
        "service": "demo",
        "project_area": "demo",
        "use_llm": False,
        "save_to_db": False,
        "web_only": True,
    }
    resp = client.post("/create_report", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "accepted"
    assert "job_id" in data
    assert data["project_area"] == "demo"


def test_create_report_rejects_inverted_window(client):
    resp = client.post("/create_report", json={
        "start": "2024-11-01T11:00:00+03:00",
        "end": "2024-11-01T10:00:00+03:00",
        "service": "demo",
    })
    assert resp.status_code == 400
    assert "позже" in resp.get_json()["message"]


def test_create_report_rejects_service_from_another_area(client):
    resp = client.post("/create_report", json={
        "start": "2024-11-01T10:00",
        "end": "2024-11-01T11:00",
        "service": "demo",
        "project_area": "other-area",
    })
    assert resp.status_code == 400
    assert "другой области" in resp.get_json()["message"]


def _runtime_files_state():
    paths = (core.CONFIG_RUNTIME_PATH, core.METRICS_RUNTIME_PATH)
    return {str(p): (p.read_bytes() if p.exists() else None) for p in paths}


def test_create_report_rejects_unknown_service_without_writing_config(client, monkeypatch):
    """A report request must not create areas or services: they would be visible to everyone."""
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_find_area_for_service", core._find_area_for_service)
    monkeypatch.setattr(dashboard, "_metrics_service_entry", core._metrics_service_entry)
    monkeypatch.setattr(dashboard, "_bootstrap_service_configs", core._bootstrap_service_configs)
    before = _runtime_files_state()
    resp = client.post("/create_report", json={
        "start": "2024-11-01T10:00",
        "end": "2024-11-01T11:00",
        "service": "../../../../tmp/loadlens-probe",
        "project_area": "unknown-area",
    })
    assert resp.status_code == 400
    assert "не найден" in resp.get_json()["message"]
    assert _runtime_files_state() == before


def test_create_report_checks_the_area_before_bootstrapping(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    bootstrapped = []
    monkeypatch.setattr(dashboard, "_bootstrap_service_configs", lambda area, service: bootstrapped.append((area, service)))
    window = {"start": "2024-11-01T10:00", "end": "2024-11-01T11:00", "service": "demo", "web_only": True, "use_llm": False}
    assert client.post("/create_report", json={**window, "project_area": "other-area"}).status_code == 400
    assert bootstrapped == []
    accepted = client.post("/create_report", json=window)
    assert accepted.status_code == 200
    assert accepted.get_json()["project_area"] == "demo"
    assert bootstrapped == [("demo", "demo")]


@pytest.mark.parametrize("run_name", ["r" * 181, "nightly/../../x", "<b>run</b>", "a\\b"])
def test_create_report_rejects_bad_run_names(client, run_name):
    resp = client.post("/create_report", json={
        "start": "2024-11-01T10:00",
        "end": "2024-11-01T11:00",
        "service": "demo",
        "run_name": run_name,
    })
    assert resp.status_code == 400
    assert "Имя отчёта" in resp.get_json()["message"]


def test_create_report_ignores_non_string_fields(client):
    resp = client.post("/create_report", json={
        "start": "2024-11-01T10:00",
        "end": "2024-11-01T11:00",
        "service": ["demo"],
    })
    assert resp.status_code == 400


def test_temporary_file_names_stay_in_their_directory(monkeypatch, tmp_path):
    from data_collectors import temp_files

    monkeypatch.setattr(temp_files, "TEMP_DIR", tmp_path / "temporary_files")
    for raw in ("../../../../tmp/x_logs_123", "..", "a/b\\c", 'cpu"><ac:macro', "ЦПУ узла_svc_1", ""):
        name = temp_files.safe_basename(raw)
        assert name and "/" not in name and "\\" not in name and '"' not in name and "<" not in name
        assert not name.startswith(".")
        assert temp_files.safe_basename(name) == name
        path = Path(temp_files.temp_file_path(raw, ".log"))
        assert path.parent == temp_files.TEMP_DIR
        assert temp_files.is_temp_file(path)
    assert temp_files.safe_basename("ЦПУ узла_svc_1") == "ЦПУ_узла_svc_1"
    assert not temp_files.is_temp_file(temp_files.TEMP_DIR / ".." / "settings.py")
    assert not temp_files.is_temp_file(tmp_path / "elsewhere.log")


def test_create_report_assigns_run_name_and_tracks_job(client):
    resp = client.post("/create_report", json={
        "start": "2024-11-01T10:00:00+03:00",
        "end": "2024-11-01T11:00:00+03:00",
        "service": "demo",
        "use_llm": True,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["run_name"].startswith("run-")

    status = client.get(f"/job_status/{data['job_id']}").get_json()
    assert status["status"] == "done"
    assert status["progress"] == 100
    assert status["report_url"] == "/reports/demo/test-run"
    assert status["run_name"] == "test-run"
    assert status["kind"] == "report"

    listed = client.get("/jobs", query_string={"status": "done", "limit": 5}).get_json()
    assert any(j["job_id"] == data["job_id"] for j in listed["jobs"])


def test_jobs_endpoint_validates_status(client):
    resp = client.get("/jobs", query_string={"status": "weird"})
    assert resp.status_code == 400
    assert "Недопустимый статус" in resp.get_json()["error"]


def test_failed_report_job_names_the_phase(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    def _failing_update_report(*args, **kwargs):
        kwargs["progress_callback"]("Сбор метрик: JVM (1/6)", 10)
        raise RuntimeError("Grafana returned 502")

    monkeypatch.setattr(dashboard, "update_report", _failing_update_report)
    resp = client.post("/create_report", json={
        "start": "2024-11-01T10:00",
        "end": "2024-11-01T11:00",
        "service": "demo",
        "run_name": "broken-run",
    })
    assert resp.status_code == 200
    job = client.get(f"/job_status/{resp.get_json()['job_id']}").get_json()
    assert job["status"] == "error"
    assert job["message"] == "Ошибка на этапе «Сбор метрик: JVM (1/6)»"
    assert "Grafana returned 502" in job["error"]
    assert job["progress"] == 10


def test_job_store_falls_back_to_database_row(monkeypatch):
    from loadlens_app import jobs

    created = datetime(2024, 11, 1, 10, 0, tzinfo=timezone.utc)
    row = (
        "abc123", "report", "nightly", "demo", "running", 47,
        "Анализ ИИ: 6 доменов, ожидание ответов модели", None, None, None, None,
        created, created,
    )
    monkeypatch.setattr(jobs, "_ts_conn", lambda: RecordingConnection(fetchone_results=[row]))
    store = jobs.JobStore()

    record = store.get("abc123")

    assert record is not None
    assert record.status == "running"
    assert record.progress == 47
    assert record.run_name == "nightly"
    assert record.to_dict()["created_at"] == created.isoformat()


def test_job_store_progress_never_decreases(monkeypatch):
    from loadlens_app import jobs

    monkeypatch.setattr(jobs, "_ts_conn", lambda: FakeConnection())
    store = jobs.JobStore()
    job = store.create(jobs.JOB_KIND_REPORT, run_name="r1", service="demo", message="start")
    store.update(job.job_id, progress=40, message="collect")
    store.update(job.job_id, progress=10, message="late event")
    store.update(job.job_id, progress=None, message="no percent")

    current = store.get(job.job_id)
    assert current.progress == 40
    assert current.message == "no percent"
    assert [j.job_id for j in store.list(statuses=["running"])] == [job.job_id]


def test_confluence_template_keeps_pipeline_progress_after_attachments():
    assert _span_percent(10, 30, 0, 4) == 10
    assert _span_percent(10, 30, 4, 4) == 30
    assert _span_percent(30, 40, 1, 2) == 35
    # Pipeline collect (10) stays above the attachment budget (40), and save (95) stays below the final 95.
    assert _scale_pipeline_percent(10) == 45
    assert _scale_pipeline_percent(47) == 65
    assert _scale_pipeline_percent(95) == 91
    assert _scale_pipeline_percent(None) is None


def test_publish_confluence_smoke(client):
    resp = client.post("/publish_confluence", json={"run_name": "test-run", "service": "demo"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "accepted"
    job = client.get(f"/job_status/{data['job_id']}").get_json()
    assert job["status"] == "done"
    assert job["kind"] == "confluence"
    assert job["page_url"]


def test_confluence_publication_empty(client):
    resp = client.get("/confluence_publication", query_string={"run_name": "missing-run"})
    assert resp.status_code == 200
    assert resp.get_json()["page_url"] is None


def test_llm_reports_returns_system_context(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    expected_context = {
        "schema_version": 1,
        "system": {"name": "Checkout Platform", "domain": "e-commerce", "description": "Обработка заказов", "test_goal": "Проверить checkout"},
        "architecture": {"style": "microservices", "components": [], "dependencies": [], "data_stores": []},
        "load_model": {"entrypoints": [], "critical_user_flows": [], "expected_hotspots": []},
        "operational_context": {"known_constraints": [], "known_risks": [], "normal_degradation_rules": [], "analysis_focus": []},
    }
    rows = [
        (
            "test-run",
            "demo",
            1730455200000,
            1730458800000,
            "final",
            "Успешно",
            '{"verdict":"Успешно","verdict_rationale":"Вердикт подтвержден стабильными метриками.\\n- Существенных отклонений не найдено.","findings":[],"recommended_actions":[]}',
            {
                "verdict": "Успешно",
                "verdict_rationale": "Вердикт подтвержден стабильными метриками.\n- Существенных отклонений не найдено.",
                "findings": [],
                "recommended_actions": [],
            },
            {"judge": {"overall": 0.9}},
            "Успешно",
            {"checks": [], "summary": "ok"},
            expected_context,
            datetime.now(timezone.utc),
            "step",
        )
    ]
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: RowsConnection(rows))

    resp = client.get("/llm_reports", query_string={"run_name": "test-run"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert data[0]["domain"] == "final"
    assert data[0]["test_type"] == "step"
    assert "стабильными метриками" in data[0]["parsed"]["verdict_rationale"]
    assert data[0]["system_context"]["system"]["name"] == "Checkout Platform"


# ---- dashboard, demo, compare ----------------------------------------------------------

def test_dashboard_data_groups_services(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    now = datetime(2024, 11, 1, 11, 0, tzinfo=timezone.utc)
    earlier = datetime(2024, 10, 31, 11, 0, tzinfo=timezone.utc)
    conn = RecordingConnection(
        fetchone_results=[("nightly-1", "demo", 1, 2, "Есть риски", "Успешно", "Есть риски", now), (3,)],
        fetchall_results=[
            [("Успешно", 2), ("Провал", 1)],
            [("demo", "nightly-1", "Есть риски", now, "226.5", "step"), ("demo", "old", "Успешно", earlier, None, "soak"), ("billing", "spike", "Провал", earlier, "120", "spike")],
            [("nightly-1", "demo", "Есть риски", now, "step", 1, 2)],
        ],
    )
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: conn)
    data = client.get("/dashboard_data").get_json()
    assert data["runs_total"] == 3
    assert data["verdict_counts"]["Успешно"] == 2
    services = {s["service"]: s for s in data["services"]}
    assert services["demo"]["last_run"] == "nightly-1"
    assert services["demo"]["max_rps"] == pytest.approx(226.5)
    assert services["demo"]["recent_verdicts"] == ["Есть риски", "Успешно"]
    assert services["billing"]["max_rps"] == pytest.approx(120.0)
    assert data["recent_runs"][0]["run_name"] == "nightly-1"


def test_demo_seed_endpoint(client, monkeypatch):
    from loadlens_app.blueprints import dashboard
    from loadlens_app.demo_data import DemoAlreadyExistsError

    monkeypatch.setattr(dashboard, "seed_demo_run", lambda: {"run_name": "demo-checkout-step", "service": "demo-checkout", "area": "demo", "report_url": "/reports/demo-checkout/demo-checkout-step"})
    resp = client.post("/demo/seed")
    assert resp.status_code == 200
    assert resp.get_json()["report_url"] == "/reports/demo-checkout/demo-checkout-step"
    assert "project_area=demo" in resp.headers.get("Set-Cookie", "")

    def _exists():
        raise DemoAlreadyExistsError("demo-checkout-step", "demo-checkout")

    monkeypatch.setattr(dashboard, "seed_demo_run", _exists)
    resp = client.post("/demo/seed")
    assert resp.status_code == 409
    assert resp.get_json()["report_url"] == "/reports/demo-checkout/demo-checkout-step"


def test_seed_demo_run_writes_all_domains(monkeypatch):
    from loadlens_app import demo_data

    saved = {"domains": [], "llm": None, "bootstrap": None}
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: False)
    monkeypatch.setattr(demo_data, "_bootstrap_service_configs", lambda area, service: saved.__setitem__("bootstrap", (area, service)))
    monkeypatch.setattr(demo_data, "save_domain_labeled", lambda domain_key, conf, frames, run_meta, storage: saved["domains"].append((domain_key, [f["label"] for f in frames], len(frames[0]["df"]))))
    monkeypatch.setattr(demo_data, "save_llm_results", lambda results, run_meta, storage: saved.__setitem__("llm", (results, run_meta)))

    result = demo_data.seed_demo_run(now=datetime(2024, 11, 1, 11, 0, tzinfo=timezone.utc))

    assert result["run_name"] == demo_data.DEMO_RUN_NAME
    assert result["area"] == demo_data.DEMO_AREA
    assert saved["bootstrap"] == (demo_data.DEMO_AREA, demo_data.DEMO_SERVICE)
    assert [d[0] for d in saved["domains"]] == ["lt_framework", "jvm", "microservices", "hard_resources"]
    assert all(d[2] == demo_data.DEMO_DURATION_MIN for d in saved["domains"])
    assert "LT (InfluxQL): RPS sum by all groups" in saved["domains"][0][1]
    assert "LT (InfluxQL): http_req_duration mean(seconds)" in saved["domains"][0][1]
    assert "LT (InfluxQL): VUs" in saved["domains"][0][1]
    results, run_meta = saved["llm"]
    assert run_meta["test_type"] == "step" and run_meta["service"] == demo_data.DEMO_SERVICE
    assert results["final_parsed"]["peak_performance"]["max_rps"] == pytest.approx(249.3)
    assert results["sla_verdict"] == "Есть риски"
    assert {c["name"] for c in results["sla_checks"]} == {"target_rps", "p95_latency", "error_rate"}
    assert results["system_context"]["system"]["name"].startswith("Checkout Platform")
    assert len(results["contexts"]["final"]["load_steps"]) == 5
    assert results["contexts"]["final"]["rps_drop_iso"]
    assert "LT (InfluxQL): http_req_duration mean(seconds)" in results["contexts"]["final"]["lt_series_labels"]

    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: True)
    with pytest.raises(demo_data.DemoAlreadyExistsError) as exc:
        demo_data.seed_demo_run()
    assert exc.value.report_url == "/reports/demo-checkout/demo-checkout-step"


def test_demo_run_lookup_on_a_fresh_database(monkeypatch):
    # The first "Загрузить демо-данные" runs before any report table exists.
    import psycopg2.errors
    from loadlens_app import demo_data

    class MissingTableCursor(FakeCursor):
        def execute(self, *args, **kwargs):
            raise psycopg2.errors.UndefinedTable('relation "public.llm_reports" does not exist')

    class FreshDatabase(FakeConnection):
        def cursor(self):
            return MissingTableCursor()

    monkeypatch.setattr(demo_data, "_ts_conn", lambda: FreshDatabase())
    assert demo_data._run_exists(demo_data.DEMO_RUN_NAME) is False


def test_compare_summary_aggregate_whitelist(client, monkeypatch):
    from loadlens_app.blueprints import compare

    conn = RecordingConnection(fetchall_results=[[("LT: RPS", 300.0, 270.0)]])
    monkeypatch.setattr(compare, "_ts_conn", lambda: conn)
    resp = client.get("/compare_summary", query_string={"run_a": "a", "run_b": "b", "domain": "lt_framework", "agg": "max"})
    assert resp.status_code == 200
    row = resp.get_json()[0]
    assert row["agg"] == "max" and row["value_a"] == 300.0 and row["value_b"] == 270.0
    assert row["trend_pct"] == pytest.approx(-10.0)
    assert "MAX(value)" in conn.log[0][0]

    resp = client.get("/compare_summary", query_string={"run_a": "a", "run_b": "b", "domain": "lt_framework", "agg": "median"})
    assert resp.status_code == 400
    assert "median" in resp.get_json()["error"]


def test_compare_baseline(client, monkeypatch):
    from loadlens_app.blueprints import compare

    created = datetime(2024, 10, 31, 11, 0, tzinfo=timezone.utc)
    conn = RecordingConnection(fetchone_results=[("release-2.4", "demo", "Успешно", created)])
    monkeypatch.setattr(compare, "_ts_conn", lambda: conn)
    resp = client.get("/compare_baseline", query_string={"run_name": "nightly-1"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["run_name"] == "release-2.4" and data["mode"] == "previous_success"
    assert "= 'Успешно'" in conn.log[0][0]

    empty = RecordingConnection(fetchone_results=[None])
    monkeypatch.setattr(compare, "_ts_conn", lambda: empty)
    resp = client.get("/compare_baseline", query_string={"run_name": "nightly-1", "mode": "previous"})
    assert resp.status_code == 404
    assert "предыдущий прогон" in resp.get_json()["error"]
    assert "= 'Успешно'" not in empty.log[0][0]

    resp = client.get("/compare_baseline", query_string={"run_name": "nightly-1", "mode": "best"})
    assert resp.status_code == 400


def test_domains_schema_filters_by_run(client, monkeypatch):
    from loadlens_app.blueprints import compare

    conn = RecordingConnection(fetchall_results=[[("jvm", "JVM: Heap used", 12), ("lt_framework", "LT: RPS", 3)]])
    monkeypatch.setattr(compare, "_ts_conn", lambda: conn)

    resp = client.get("/domains_schema", query_string={"run_name": "nightly-1"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data.keys()) == {"jvm", "lt_framework"}
    assert data["jvm"][0]["query_label"] == "JVM: Heap used"
    sql_text, params = conn.log[0]
    assert "WHERE run_name = %s" in sql_text
    assert params == ("nightly-1",)

    conn_all = RecordingConnection(fetchall_results=[[]])
    monkeypatch.setattr(compare, "_ts_conn", lambda: conn_all)
    client.get("/domains_schema")
    sql_all, params_all = conn_all.log[0]
    assert "WHERE" not in sql_all
    assert params_all == ()


# ---- prompts -----------------------------------------------------------------------------

def test_prompt_history_and_defaults(client):
    defaults = client.get("/prompts/defaults").get_json()
    assert "jvm" in defaults["domains"] and "application_logs" in defaults["domains"] and "judge" not in defaults["domains"]

    assert client.post("/prompts", json={"area": "demo", "domain": "jvm", "text": "версия A"}).status_code == 200
    history = client.get("/prompts/history", query_string={"domain": "jvm", "area": "demo"}).get_json()
    assert history["versions"] == []  # the first override replaced the file default, which is not stored as a version

    assert client.post("/prompts", json={"area": "demo", "domain": "jvm", "text": "версия B"}).status_code == 200
    history = client.get("/prompts/history", query_string={"domain": "jvm", "area": "demo"}).get_json()
    assert [v["text"] for v in history["versions"]] == ["версия A"]
    assert history["versions"][0]["saved_at"]

    assert client.post("/prompts", json={"area": "demo", "domain": "jvm", "text": "", "service": "svc"}).status_code == 200
    scoped = client.get("/prompts/history", query_string={"domain": "jvm", "area": "demo", "service": "svc"}).get_json()
    assert scoped["versions"] == []

    assert client.get("/prompts/history", query_string={"domain": "judge", "area": "demo"}).status_code == 400
    assert client.get("/prompts/history", query_string={"domain": "jvm"}).status_code == 400


def test_get_prompts_without_area_returns_file_defaults(client):
    assert client.post("/prompts", json={"area": "demo", "domain": "jvm", "text": "текст области"}).status_code == 200
    global_prompts = client.get("/prompts").get_json()
    assert global_prompts["active_area"] == ""
    assert global_prompts["domains"]["jvm"] != "текст области"
    area_prompts = client.get("/prompts", query_string={"area": "demo"}).get_json()
    assert area_prompts["domains"]["jvm"] == "текст области"


# ---- connection checks -------------------------------------------------------------------

def test_connection_checks_with_mocked_transport(monkeypatch):
    from loadlens_app import connection_checks as cc

    class FakeResponse:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status
            self.content = b"x"

        def raise_for_status(self):
            if self.status_code >= 400:
                err = cc.requests.exceptions.HTTPError(f"HTTP {self.status_code}")
                err.response = self
                raise err

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if url.endswith("/api/v1/status/buildinfo"):
            return FakeResponse({"status": "success", "data": {"version": "2.51.0"}})
        if url.endswith("/api/health"):
            return FakeResponse({"version": "10.4.1"})
        if url.endswith("/api/user"):
            return FakeResponse({"login": "admin"})
        if url.endswith("/health"):
            return FakeResponse({"status": "pass", "version": "2.7"})
        if "/rest/api/space/" in url:
            return FakeResponse({"name": "Load Testing"})
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(cc.requests, "get", fake_get)
    monkeypatch.setattr(cc, "_resolve_grafana_prom_ds_id", lambda g: 7)

    prom = cc.check_data_source({"type": "prometheus", "prometheus": {"url": "http://prom:9090"}})
    assert prom.ok and "2.51.0" in prom.message

    grafana = cc.check_data_source({"type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "basic", "username": "a", "password": "b"}}})
    assert grafana.ok and "авторизация" in grafana.message and "10.4.1" in grafana.message

    grafana_ds = cc.check_grafana({"base_url": "http://grafana:3000", "auth": {"method": "basic", "username": "a", "password": "b"}, "prometheus_datasource": {"name": "Prom"}}, "prometheus")
    assert grafana_ds.ok and "id=7" in grafana_ds.message

    influx = cc.check_data_source({"type": "influxdb", "influxdb": {"url": "http://influx:8086", "token": "t"}})
    assert influx.ok and "pass" in influx.message

    template = cc.check_confluence_template({"url_basic": "https://wiki.local", "space_conf": "LT", "user": "u", "password": "p", "grafana_base_url": "http://grafana:3000"})
    assert template.ok and "Load Testing" in template.message and "10.4.1" in template.message

    monkeypatch.setattr(cc.requests, "post", lambda url, **kwargs: FakeResponse({"count": 1234}))
    opensearch = cc.check_opensearch({"opensearch": {"base_url": "https://os:5601", "index_pattern": "nt_*", "username_env": "u", "password_env": "p"}})
    assert opensearch.ok and "1234" in opensearch.message

    def failing_get(url, **kwargs):
        raise cc.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(cc.requests, "get", failing_get)
    down = cc.check_prometheus("http://prom:9090")
    assert down.ok is False
    assert "Не удалось подключиться" in down.message
    assert down.details["error_type"] == "ConnectionError"

    unknown = cc.check_data_source({"type": "graphite"})
    assert unknown.ok is False and "graphite" in unknown.message


def test_grafana_datasources_endpoint_lists_query_datasources(client, monkeypatch):
    from loadlens_app import connection_checks as cc

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"uid": "loki", "name": "Loki", "type": "loki"},
                {"uid": "k6", "name": "InfluxDB-k6", "type": "influxdb", "database": "k6", "jsonData": {}},
                {"uid": "tg", "name": "Telegraf", "type": "influxdb", "jsonData": {"version": "Flux", "defaultBucket": "telegraf"}},
                {"uid": "prom", "name": "Prometheus", "type": "prometheus", "isDefault": True},
            ]

    monkeypatch.setattr(cc.requests, "get", lambda url, **kwargs: FakeResponse())
    monkeypatch.setitem(core.CONFIG, "data_sources", {
        "grafana": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "basic", "username": "a", "password": "b"}}},
    })
    data = client.get("/config/grafana_datasources", query_string={"source": "grafana"}).get_json()
    assert data["ok"] is True
    items = {item["uid"]: item for item in data["datasources"]}
    assert set(items) == {"k6", "tg", "prom"}
    assert (items["k6"]["mode"], items["k6"]["database"]) == ("influxql", "k6")
    assert (items["tg"]["mode"], items["tg"]["bucket"]) == ("flux", "telegraf")
    assert items["prom"]["mode"] == "promql" and items["prom"]["is_default"] is True
    assert client.get("/config/grafana_datasources", query_string={"source": "missing"}).status_code == 404


def test_check_llm_uses_form_provider_config(monkeypatch):
    from AI import providers
    from loadlens_app import connection_checks as cc

    calls = {}

    def fake_call(provider, messages, pcfg, system_text):
        calls["provider"] = provider
        calls["pcfg"] = pcfg
        calls["system_text"] = system_text
        return "OK"

    monkeypatch.setattr(providers, "_call_provider", fake_call)
    result = cc.check_llm({"provider": "openai", "openai": {"model": "gpt-test", "api_key": "k", "request_timeout_sec": 600}})
    assert result.ok is True and "gpt-test" in result.message and "OK" in result.message
    assert calls["provider"] == "openai"
    assert calls["pcfg"]["model"] == "gpt-test"
    assert calls["pcfg"]["request_timeout_sec"] == cc.LLM_CHECK_TIMEOUT_SEC
    assert calls["pcfg"]["generation"]["max_tokens"] == 16

    unknown = cc.check_llm({"provider": "yandex"})
    assert unknown.ok is False and "yandex" in unknown.message


def test_check_timescale_reports_missing_tables(monkeypatch):
    from loadlens_app import connection_checks as cc

    class Cur(FakeCursor):
        def __init__(self):
            self._q = ""

        def execute(self, query, params=None):
            self._q = str(query)

        def fetchone(self):
            if "version()" in self._q:
                return ("PostgreSQL 16.2 on x86_64",)
            if "pg_extension" in self._q:
                return ("2.14.2",)
            return None

        def fetchall(self):
            return [("metrics",), ("llm_reports",)]

    class Conn(FakeConnection):
        def cursor(self):
            return Cur()

    monkeypatch.setattr(cc.psycopg2, "connect", lambda **kwargs: Conn())
    result = cc.check_timescale({"host": "db", "port": 5432, "dbname": "lt", "user": "u", "password": "p"})
    assert result.ok is True
    assert "PostgreSQL 16.2" in result.message and "TimescaleDB 2.14.2" in result.message
    assert result.details["tables_missing"] == ["engineer_reports"]


# ---- LLM providers and scoring --------------------------------------------------------

def test_ask_llm_uses_provider_override_and_max_attempts(monkeypatch):
    from AI import providers

    calls = {"n": 0}

    def failing_call(provider, messages, pcfg, system_text):
        calls["n"] += 1
        calls["provider"] = provider
        calls["model"] = pcfg.get("model")
        raise RuntimeError("boom")

    monkeypatch.setattr(providers, "_call_provider", failing_call)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        providers.ask_llm_with_text_data("q", "", llm_config={"provider": "gigachat", "gigachat": {"model": "GigaChat-Max"}, "max_attempts": 2})
    assert calls == {"n": 2, "provider": "gigachat", "model": "GigaChat-Max"}


def test_self_consistency_sequential_mode_limits_candidates(monkeypatch):
    from AI import scoring

    monkeypatch.setitem(core.CONFIG["llm"], "self_consistency", {"max_candidates": 2, "parallel_candidates": False, "parallel_critics": False, "pause_sec_between_calls": 0})
    calls = []
    payload = json.dumps({"verdict": "Успешно", "verdict_rationale": "ok", "confidence": 0.9, "findings": [], "recommended_actions": []}, ensure_ascii=False)
    monkeypatch.setattr(scoring, "ask_llm_with_text_data", lambda prompt, ctx: calls.append(prompt) or payload)
    monkeypatch.setattr(scoring, "_select_best_candidate", lambda candidates, ctx, domain_key=None: (candidates[0][0], candidates[0][1], {"final_score": 1.0}))

    text, parsed, scores = scoring.llm_two_pass_self_consistency("prompt", "{}", k=5, return_scores=True, domain_key="jvm")
    assert len(calls) == 2
    assert parsed is not None and parsed.verdict == "Успешно"
    assert scores["final_score"] == 1.0


def test_perplexity_provider_uses_current_sonar_endpoint_and_models():
    from AI.providers import (
        _extract_perplexity_agent_text,
        _perplexity_api_type,
        _perplexity_model,
        _perplexity_url,
        _strip_think,
    )

    assert _perplexity_url({"api_base_url": "https://api.perplexity.ai"}) == "https://api.perplexity.ai/v1/sonar"
    assert _perplexity_url({"api_base_url": "https://api.perplexity.ai", "model": "openai/gpt-5.4"}) == "https://api.perplexity.ai/v1/agent"
    assert _perplexity_url({"api_base_url": "https://api.perplexity.ai/chat/completions"}) == "https://api.perplexity.ai/chat/completions"
    assert _perplexity_api_type({"model": "sonar-pro"}) == "sonar"
    assert _perplexity_api_type({"model": "openai/gpt-5.4"}) == "agent"
    assert _perplexity_model({"model": "sonar-pro"}) == "sonar-pro"
    assert _perplexity_model({"model": "openai/gpt-5.4"}) == "openai/gpt-5.4"
    assert _extract_perplexity_agent_text({"output": [{"content": [{"type": "output_text", "text": "OK"}]}]}) == "OK"
    assert _strip_think("<think>partial reasoning") == ""


def test_parse_llm_analysis_preserves_verdict_rationale():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "verdict_rationale": (
                "Вердикт снижен до уровня риска из-за локальных деградаций в пиковое окно.\n"
                "- p95 вышел за ориентир.\n"
                "- Ошибки выросли одновременно с ростом нагрузки."
            ),
            "confidence": 0.81,
            "findings": [],
            "recommended_actions": [],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.verdict == "Есть риски"
    assert "локальных деградаций" in (parsed.verdict_rationale or "")


def test_parse_llm_analysis_preserves_finding_links():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "verdict_rationale": "Есть локальные деградации.",
            "confidence": 0.77,
            "findings": [
                {"id": "f1", "summary": "P95 latency вырос до 420мс", "severity": "high", "component": "orders", "evidence": "service=orders, 12:10-12:20, peak_time=12:16"},
                {"summary": "Ошибки выросли до 1.2%", "severity": "medium", "component": "orders", "evidence": "service=orders, 12:12-12:18, peak_time=12:17"},
            ],
            "recommended_actions": [
                {
                    "summary": "Проверить пул соединений и таймауты upstream.",
                    "details": "Сверить лимиты пула соединений с пиковым уровнем конкурентности и отдельно проверить таймауты upstream. После корректировки повторить тест и убедиться, что latency и доля timeout-ошибок снизились.",
                    "priority": "high",
                    "affected_components": ["orders"],
                    "for_finding_ids": ["f1"],
                },
                {
                    "summary": "Усилить retry budget и алерты.",
                    "details": "Ограничить агрессивные ретраи, чтобы они не усиливали деградацию, и добавить алерты на ранние признаки роста ошибок.",
                    "priority": "medium",
                    "affected_components": ["orders"],
                },
            ],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.findings[0].id == "f1"
    assert parsed.findings[1].id == "finding_2"
    assert parsed.recommended_actions[0].for_finding_ids == ["f1"]
    assert parsed.recommended_actions[1].for_finding_ids == ["f1", "finding_2"]
    assert "конкурентности" in (parsed.recommended_actions[0].details or "")


def test_parse_llm_analysis_derives_missing_verdict_rationale():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "findings": [{"id": "f1", "summary": "Kafka lag превышает порог", "severity": "high", "component": "kafka"}],
            "recommended_actions": [],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.verdict_rationale
    assert "Kafka lag превышает порог" in parsed.verdict_rationale


def test_parse_llm_analysis_preserves_structured_finding_evidence():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "verdict_rationale": "Есть локальные деградации.",
            "confidence": 0.79,
            "findings": [
                {
                    "id": "f1",
                    "summary": "P95 latency выросла в пиковое окно",
                    "severity": "high",
                    "component": "orders",
                    "start_time": "12:10",
                    "end_time": "12:18",
                    "peak_time": "12:16",
                    "evidence_summary": "Рост latency совпал с ростом нагрузки.",
                    "evidence_items": [{"metric": "P95 latency", "observed_value": "420мс", "threshold": "300мс", "note": "service=orders"}],
                }
            ],
            "recommended_actions": [
                {
                    "summary": "Проверить лимиты и таймауты upstream.",
                    "details": "Проверить лимиты downstream и таймауты клиентских вызовов в окне пика. Затем повторить прогон и убедиться, что p95 и error rate стабилизировались.",
                    "priority": "high",
                    "affected_components": ["orders"],
                    "for_finding_ids": ["f1"],
                }
            ],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.findings[0].start_time == "12:10"
    assert parsed.findings[0].end_time == "12:18"
    assert parsed.findings[0].peak_time == "12:16"
    assert parsed.findings[0].evidence_summary == "Рост latency совпал с ростом нагрузки."
    assert parsed.findings[0].evidence_items[0].metric == "P95 latency"
    assert parsed.findings[0].evidence_items[0].observed_value == "420мс"
    assert parsed.findings[0].evidence_items[0].threshold == "300мс"
    assert "error rate" in (parsed.recommended_actions[0].details or "")


# ---- pipeline ------------------------------------------------------------------------------

def test_select_step_profile_candidate_prefers_last_stable_before_unstable():
    segments = [
        {"start": "2024-01-01T10:00:00", "end": "2024-01-01T10:10:00", "level": 240.0, "stable": True},
        {"start": "2024-01-01T10:10:00", "end": "2024-01-01T10:20:00", "level": 225.0, "stable": True},
        {"start": "2024-01-01T10:20:00", "end": "2024-01-01T10:30:00", "level": 260.0, "stable": False},
    ]
    chosen = _select_step_profile_candidate(segments)
    assert chosen is not None
    assert chosen["level"] == 225.0


def test_step_profile_uses_actual_series_cadence_for_sparse_points():
    idx = pd.date_range("2024-01-01T10:00:00Z", periods=8, freq="5min")
    series = pd.Series([190.0, 191.0, 225.0, 226.0, 225.5, 226.2, 160.0, 150.0], index=idx)
    cfg = {
        "step_detection_resample_sec": 20,
        "step_detection_smooth_sec": 90,
        "step_confirm_hold_sec": 120,
        "step_min_step_delta_rps": 8.0,
        "step_min_step_delta_pct": 0.05,
        "step_max_cv": 0.20,
        "step_max_slope_rps_per_min": 2.0,
        "step_max_within_step_drop_pct": 0.10,
        "step_drop_hold_sec": 90,
    }
    stable = _find_stable_peak_step_profile(series, min_stable_minutes=5.0, cfg=cfg)
    assert stable is not None
    assert stable["stable_max"] == pytest.approx(225.625)
    assert stable["method"] == "step_profile"


def test_reconcile_sla_downgrades_stability_resource_failures_to_risk():
    result = _reconcile_sla_for_test_profile(
        {
            "verdict": "Провал",
            "checks": [
                {"name": "p95_latency", "passed": True},
                {"name": "error_rate", "passed": True},
                {"name": "memory_usage", "passed": False},
            ],
            "summary": "SLA verdict: Провал. Нарушено: memory_usage",
        },
        {"mode": "stability"},
    )
    assert result["verdict"] == "Есть риски"
    assert result["test_mode"] == "stability"
    assert result["checks"][2]["category"] == "secondary"


def test_upload_from_llm_reports_progress_and_passes_deterministic_sla(monkeypatch):
    lt_df = pd.DataFrame(
        {"sum_all": [180.0, 205.0, 226.5]},
        index=pd.date_range("2024-01-01T10:00:00", periods=3, freq="5min"),
    )
    state = {"sla_called": False}
    captured = {}
    ef_cfg = {
        "llm": {"include_markdown_tables_in_context": False, "self_consistency_k": 1, "max_domain_workers": 1},
        "default_params": {"step": "1m", "resample_interval": "5T"},
        "data_sources": {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prometheus:9090"}}},
        "domain_sources": {"default": {"source": "prom"}, "lt_framework": {"source": "prom"}},
        "queries": {
            "lt_framework": {
                "promql_queries": ["sum(rate(http_requests_total[1m]))"],
                "label_keys_list": [[]],
                "labels": ["LT (InfluxQL): RPS sum by all groups"],
            }
        },
        "sla": {"target_rps": 200, "max_performance_query": "LT (InfluxQL): RPS sum by all groups", "target_rps_allow_peak_fallback": True},
    }

    monkeypatch.setattr(pipeline_module, "read_prompt_from_file", lambda filename: "overall prompt" if "overall" in str(filename) else "domain prompt")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *args, **kwargs: [lt_df])
    monkeypatch.setattr(
        pipeline_module,
        "extract_target_rps_from_pack",
        lambda *args, **kwargs: {
            "value": 226.5,
            "method": "stable_max (query: LT (InfluxQL): RPS sum by all groups)",
            "source_label": "LT (InfluxQL): RPS sum by all groups",
            "source_series": "sum_all",
            "reason": None,
        },
    )

    def fake_evaluate_sla(domain_data, sla_cfg, test_profile=None):
        state["sla_called"] = True
        return {
            "verdict": "Успешно",
            "summary": "SLA verdict: Успешно. Пройдено: target_rps",
            "checks": [{"name": "target_rps", "category": "primary", "threshold": 200, "actual": 226.5, "passed": True, "severity": "critical", "message": "RPS 226.5 (stable_max) >= целевой 200"}],
            "stable_window": {
                "start": "2024-01-01T10:00:00+00:00",
                "end": "2024-01-01T10:10:00+00:00",
                "level": 226.5,
                "series": "sum_all",
                "label": "LT (InfluxQL): RPS sum by all groups",
                "degraded_level": None,
                "degraded_checks": [],
            },
        }

    monkeypatch.setattr(pipeline_module, "evaluate_sla", fake_evaluate_sla)

    def fake_llm(user_prompt, data_context, k, return_scores, domain_key):
        payload = {"verdict": "Успешно", "verdict_rationale": "Итог сформирован по данным.", "findings": [], "recommended_actions": [], "peak_performance": {"max_rps": 1}}
        assert state["sla_called"] is True
        if domain_key == "final":
            captured["context"] = json.loads(data_context)
        if domain_key == "lt_framework":
            captured["lt"] = json.loads(data_context)
        return json.dumps(payload, ensure_ascii=False), dict(payload), {}

    monkeypatch.setattr(pipeline_module, "llm_two_pass_self_consistency", fake_llm)

    progress_events = []
    result = pipeline_module.uploadFromLLM(
        start_ts=1704099600,
        end_ts=1704103200,
        save_to_db=False,
        ef_config=ef_cfg,
        active_domains=["lt_framework"],
        progress_callback=lambda msg, pct: progress_events.append((msg, pct)),
    )

    ctx = captured["context"]
    assert captured["lt"]["sla_step"]["rps"] == pytest.approx(226.5)
    assert ctx["designated_peak_performance"]["stable_max"] == pytest.approx(226.5)
    assert ctx["designated_peak_performance"]["series"] == "sum_all"
    assert ctx["deterministic_sla"]["verdict"] == "Успешно"
    assert ctx["deterministic_sla"]["target_rps_check"]["passed"] is True
    assert result["sla_verdict"] == "Успешно"
    assert result["final_parsed"]["peak_performance"]["max_rps"] == pytest.approx(226.5)
    # lt_framework keeps its own peak, other domains never carry one
    assert "peak_performance" in result["lt_framework_parsed"]
    assert result["jvm_parsed"] is None

    messages = [msg for msg, _ in progress_events]
    assert any(msg.startswith("Сбор метрик: Нагрузочный инструмент (1/1)") for msg in messages)
    assert any(msg.startswith("Анализ ИИ: готово 1/1") for msg in messages)
    assert "Проверка SLA-критериев" in messages
    assert "Итоговый анализ ИИ по всем доменам" in messages
    percents = [pct for _, pct in progress_events if pct is not None]
    assert percents == sorted(percents)
    assert percents[0] >= 10 and percents[-1] <= 95


def test_available_domain_keys_adds_application_logs_when_enabled():
    cfg = {"queries": {"jvm": {}, "lt_framework": {}}, "logs_source": {"enabled": True}}
    assert core._available_domain_keys(cfg) == ["jvm", "lt_framework", "application_logs"]
    assert core._available_domain_keys({"queries": {"jvm": {}}, "logs_source": {"enabled": False}}) == ["jvm"]


# ---- Confluence template flow --------------------------------------------------------------

def test_confluence_llm_renderer_matches_structured_ui_format():
    from confluence_manager.update_confluence_template import render_llm_markdown

    md = render_llm_markdown({
        "verdict": "Есть риски",
        "verdict_rationale": "Тест пройден с рисками.\n- Memory выше порога",
        "test_profile": {"mode": "stability", "test_type": "soak", "focus": "Удержание нагрузки"},
        "stability_under_load": {"target_rps": 200, "actual_rps": 210, "sla_summary": "SLA verdict: Есть риски"},
        "findings": [
            {
                "id": "f1",
                "summary": "Memory SLA превышен",
                "severity": "warning",
                "component": "k8s-arg-tr01",
                "evidence_summary": "ArangoDB резервирует память",
                "evidence_items": [{"metric": "Memory", "observed_value": "93.4%", "threshold": "80%"}],
            },
        ],
        "recommended_actions": [
            {"summary": "Проверить OOM/restarts", "details": "Если OOM/restarts нет, считать наблюдением.", "for_finding_ids": ["f1"]},
        ],
    })

    assert "| Вердикт по тесту | **Есть риски** |" in md
    assert "#### Обоснование вердикта" in md
    assert "#### Стабильность под нагрузкой" in md
    assert "#### Проблемы и рекомендации" in md
    assert "Memory SLA превышен" in md
    assert "Проверить OOM/restarts" in md
    assert "<table" not in md
    assert "<div" not in md
    assert "Доверие" not in md
    assert "Затронутые компоненты" not in md


def test_confluence_placeholder_replacement_uses_block_container():
    from confluence_manager.update_confluence_template import _replace_placeholder_storage

    html, replaced = _replace_placeholder_storage(
        "<p>Before</p><p>$$final_answer$$</p><p>After</p>",
        "$$final_answer$$",
        "<table><tbody><tr><td>OK</td></tr></tbody></table>",
    )
    assert replaced is True
    assert "<p><table" not in html
    assert "<table><tbody><tr><td>OK</td></tr></tbody></table>" in html


def _write_project_runtime(per_area: dict, metrics: dict | None = None) -> None:
    core.CONFIG_RUNTIME_PATH.write_text(json.dumps({"per_area": per_area}, ensure_ascii=False), encoding="utf-8")
    core.METRICS_RUNTIME_PATH.write_text(json.dumps(metrics or {}, ensure_ascii=False), encoding="utf-8")


def _patch_projects_db(monkeypatch, conn):
    from loadlens_app import projects as projects_service

    monkeypatch.setattr(projects_service, "_ts_conn", lambda: conn)
    monkeypatch.setattr(projects_service, "_ensure_llm_reports_table", lambda conn, cfg: None)
    return projects_service


def test_projects_overview_shows_reports_readiness_and_overrides(client, monkeypatch):
    _write_project_runtime({
        "alpha": {
            "title": "Альфа",
            "llm": copy.deepcopy(core.CONFIG.get("llm") or {}),
            "sla": {"target_rps": 100, "max_performance_query": "LT RPS"},
            "prompts": {"jvm": "custom jvm"},
            "services": {"svc-a": {"sla": {"target_rps": 50}}},
        },
        "beta": {"services": {}},
    })
    older = datetime(2026, 10, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 10, 2, tzinfo=timezone.utc)
    _patch_projects_db(monkeypatch, RecordingConnection(fetchall_results=[[("svc-a", older, "Успешно"), ("svc-a", newer, "Провал")]]))

    data = client.get("/projects").get_json()
    alpha = next(p for p in data["projects"] if p["id"] == "alpha")
    assert alpha["title"] == "Альфа" and alpha["reports"] == 2 and alpha["last_verdict"] == "Провал"
    assert alpha["readiness"]["target_rps"] == 100 and alpha["readiness"]["performance_query"] == "LT RPS"
    statuses = {o["section"]: o for o in alpha["overrides"]}
    assert statuses["llm"]["status"] == "same"
    assert statuses["sla"]["status"] == "own"
    assert statuses["prompts"]["status"] == "own" and statuses["prompts"]["own_domains"] == ["jvm"]
    assert statuses["queries"]["status"] == "inherited"
    assert alpha["services"][0]["own_sections"] == ["sla"]
    assert next(p for p in data["projects"] if p["id"] == "beta")["reports"] == 0


def test_project_create_copy_edit_and_reset(client):
    _write_project_runtime({"alpha": {"sla": {"target_rps": 100}, "services": {"svc-a": {}}}})
    assert client.post("/projects", json={"id": "gamma", "title": "Гамма", "copy_from": "alpha"}).status_code == 201
    runtime = json.loads(core.CONFIG_RUNTIME_PATH.read_text(encoding="utf-8"))
    assert runtime["per_area"]["gamma"]["sla"] == {"target_rps": 100}
    assert "services" not in runtime["per_area"]["gamma"]
    assert client.post("/projects", json={"id": "gamma"}).status_code == 409
    assert client.post("/projects", json={"id": "bad id"}).status_code == 400
    assert client.patch("/projects/gamma", json={"title": "Гамма 2", "description": "Описание"}).status_code == 200
    assert client.delete("/projects/gamma/overrides/sla").status_code == 200
    assert client.delete("/projects/gamma/overrides/unknown").status_code == 400
    runtime = json.loads(core.CONFIG_RUNTIME_PATH.read_text(encoding="utf-8"))
    assert runtime["per_area"]["gamma"]["title"] == "Гамма 2"
    assert "sla" not in runtime["per_area"]["gamma"]


def test_move_service_between_projects(client, monkeypatch):
    _write_project_runtime(
        {"alpha": {"services": {"svc-a": {"sla": {"target_rps": 50}}}}, "beta": {"services": {}}},
        {"alpha": {"services": {"svc-a": {"metrics": []}}}},
    )
    conn = RecordingConnection()
    _patch_projects_db(monkeypatch, conn)

    assert client.post("/projects/alpha/services/svc-a/move", json={"target": "beta"}).status_code == 200
    runtime = json.loads(core.CONFIG_RUNTIME_PATH.read_text(encoding="utf-8"))
    assert "svc-a" not in runtime["per_area"]["alpha"]["services"]
    assert runtime["per_area"]["beta"]["services"]["svc-a"] == {"sla": {"target_rps": 50}}
    metrics = json.loads(core.METRICS_RUNTIME_PATH.read_text(encoding="utf-8"))
    assert "svc-a" in metrics["beta"]["services"] and "svc-a" not in metrics["alpha"]["services"]
    assert conn.log[-1][1] == ("beta", ["svc-a"])
    assert client.post("/projects/alpha/services/svc-a/move", json={"target": "beta"}).status_code == 404


def test_delete_project_keeps_or_removes_data(client, monkeypatch):
    _write_project_runtime({"alpha": {"services": {"svc-a": {}}}, "beta": {"services": {"svc-b": {}}}})
    conn = RecordingConnection()
    projects_service = _patch_projects_db(monkeypatch, conn)
    deleted = []
    monkeypatch.setattr(projects_service, "_delete_service_data", lambda area, service: deleted.append((area, service)))

    assert client.delete("/projects/alpha?with_data=0").status_code == 200
    assert conn.log[-1][1] == (None, ["svc-a"])
    assert deleted == []
    assert client.delete("/projects/beta?with_data=1").status_code == 200
    assert deleted == [("beta", "svc-b")]
    assert json.loads(core.CONFIG_RUNTIME_PATH.read_text(encoding="utf-8"))["per_area"] == {}


def test_bootstrap_registers_area_without_copying_sections():
    core._bootstrap_service_configs("delta", "svc-d")
    runtime = json.loads(core.CONFIG_RUNTIME_PATH.read_text(encoding="utf-8"))
    assert runtime["per_area"]["delta"] == {"services": {"svc-d": {}}}


def _forecast_report(start, end, test_type="step"):
    from AI.db_store import StoredFinalReport

    return StoredFinalReport(
        run_id="abc",
        run_name="nightly",
        service="demo",
        test_type=test_type,
        project_area="",
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        context={},
        sla_details={"checks": [{"name": "target_rps", "threshold": 200}]},
    )


def _forecast_frames():
    import numpy as np
    from AI.capacity_forecast import usl_throughput

    concurrency = np.linspace(1.0, 100.0, 100)
    throughput = usl_throughput(10.0, 0.01, 0.0002, concurrency)
    index = pd.date_range("2026-10-01T10:00:00Z", periods=len(concurrency), freq="1min")
    return {
        "RPS": pd.DataFrame({"all": throughput}, index=index),
        "mean": pd.DataFrame({"mean": concurrency / throughput * 1000.0}, index=index),
    }


def _patch_forecast(monkeypatch, report, frames):
    from loadlens_app import forecast as service

    monkeypatch.setattr(service, "load_final_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(service, "load_run_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(service, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(service, "_effective_config_for_scope", lambda *args, **kwargs: {
        "sla": {
            "max_performance_query": "RPS",
            "mean_latency_query": "mean",
            "latency_unit": "ms",
            "load_model": "open",
            "p95_query": "",
            "vus_query": "",
        }
    })


def test_forecast_endpoint_fits_known_model(client, monkeypatch):
    frames = _forecast_frames()
    start = frames["RPS"].index[0]
    end = frames["RPS"].index[-1] + pd.Timedelta(minutes=1)
    _patch_forecast(monkeypatch, _forecast_report(start, end), frames)
    resp = client.get("/forecast/abc", query_string={"target_rps": 200, "headroom_pct": 20})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["fit"]["x_max"] == pytest.approx(263.6, rel=0.03)
    assert body["target"]["verdict"] == "holds"
    assert body["limit"]["kind"] == "model"
    assert body["resources"]["status"] == "missing_setting"
    assert body["answer"]["verdict"] == "holds"
    assert body["safe"]["source"] == "limit"
    assert body["safe"]["rps"] == pytest.approx(0.8 * body["fit"]["x_max"])
    assert body["cutoff_kind"] == "test_end"
    assert body["sla_target_rps"] == 200


def test_forecast_endpoint_rejects_bad_input(client, monkeypatch):
    from loadlens_app import forecast as service

    frames = _forecast_frames()
    start = frames["RPS"].index[0]
    end = frames["RPS"].index[-1] + pd.Timedelta(minutes=1)
    monkeypatch.setattr(service, "load_final_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_ts_conn", lambda: FakeConnection())
    assert client.get("/forecast/missing").status_code == 404
    assert client.get("/forecast/abc", query_string={"target_rps": -5}).status_code == 400
    _patch_forecast(monkeypatch, _forecast_report(start, end, "soak"), frames)
    resp = client.get("/forecast/abc", query_string={"target_rps": 200})
    assert resp.status_code == 422 and resp.get_json()["code"] == "not_applicable"
    _patch_forecast(monkeypatch, _forecast_report(start, end), {"RPS": frames["RPS"]})
    resp = client.get("/forecast/abc", query_string={"target_rps": 200})
    assert resp.status_code == 422 and resp.get_json()["code"] == "missing_series"


def test_sla_rejects_unknown_load_model(client):
    resp = client.post("/config", json={"section": "sla", "data": {"load_model": "mixed"}})
    assert resp.status_code == 400
    assert "load_model" in resp.get_json()["error"]


def _forecast_sla():
    return {
        "sla": {
            "max_performance_query": "RPS",
            "mean_latency_query": "mean",
            "latency_unit": "ms",
            "load_model": "open",
            "p95_query": "",
            "vus_query": "",
        }
    }


def test_forecast_catalog_counts_gaps(client, monkeypatch):
    from AI.db_store import FinalReportRow
    from loadlens_app import forecast as service
    from loadlens_app.blueprints import forecast as forecast_api

    created = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = [
        FinalReportRow("ready1", "ready", "svc", "step", "", created, "Успешно", ["RPS", "mean"]),
        FinalReportRow("nomean", "no-mean", "svc", "step", "", created, "Есть риски", ["RPS"]),
        FinalReportRow("old1", "old", "svc", "step", "", created, "Успешно", None),
        FinalReportRow("soak1", "soak", "svc", "soak", "", created, "Успешно", ["RPS", "mean"]),
    ]
    monkeypatch.setattr(service, "list_final_reports", lambda *args, **kwargs: rows)
    monkeypatch.setattr(service, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(service, "_effective_config_for_scope", lambda *args, **kwargs: _forecast_sla())
    monkeypatch.setattr(forecast_api, "_resolve_services_filter", lambda area: [])
    resp = client.get("/forecast_reports")
    assert resp.status_code == 200
    body = resp.get_json()
    assert [item["run_id"] for item in body["reports"]] == ["ready1"]
    assert body["missing_series"] == 2
    assert body["missing_settings"] == 0


def test_forecast_status_for_the_report_button(client, monkeypatch):
    from AI.db_store import StoredFinalReport
    from loadlens_app import forecast as service

    frames = _forecast_frames()
    start = frames["RPS"].index[0]
    end = frames["RPS"].index[-1]
    ready = StoredFinalReport(
        run_id="abc", run_name="nightly", service="demo", test_type="step", project_area="",
        start_ms=int(start.timestamp() * 1000), end_ms=int(end.timestamp() * 1000),
        context={"lt_series_labels": ["RPS", "mean"]}, sla_details={},
    )
    monkeypatch.setattr(service, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(service, "_effective_config_for_scope", lambda *args, **kwargs: _forecast_sla())
    monkeypatch.setattr(service, "load_final_report", lambda *args, **kwargs: ready)
    resp = client.get("/forecast/abc/status")
    assert resp.status_code == 200 and resp.get_json()["available"] is True

    missing = StoredFinalReport(
        run_id="abc", run_name="nightly", service="demo", test_type="step", project_area="",
        start_ms=int(start.timestamp() * 1000), end_ms=int(end.timestamp() * 1000),
        context={}, sla_details={},
    )
    monkeypatch.setattr(service, "load_final_report", lambda *args, **kwargs: missing)
    resp = client.get("/forecast/abc/status")
    assert resp.status_code == 200
    assert resp.get_json()["available"] is False and resp.get_json()["code"] == "missing_series"

    monkeypatch.setattr(service, "load_final_report", lambda *args, **kwargs: None)
    assert client.get("/forecast/missing/status").status_code == 404


def _explained_report(frames):
    from AI.db_store import StoredFinalReport

    index = frames["RPS"].index
    steps = [
        {
            "label": f"ступень {number + 1}",
            "source": "step_detector",
            "start_iso": index[number * 25].isoformat(),
            "end_iso": (index[number * 25] + pd.Timedelta(minutes=25)).isoformat(),
        }
        for number in range(4)
    ]
    return StoredFinalReport(
        run_id="abc", run_name="nightly", service="demo", test_type="step", project_area="",
        start_ms=int(index[0].timestamp() * 1000), end_ms=int((index[-1] + pd.Timedelta(minutes=1)).timestamp() * 1000),
        context={"load_steps": steps}, sla_details={},
    )


def _finding(ref, minutes, summary, component="orders-api", verification="verified"):
    from AI.db_store import StoredFinding

    start = pd.Timestamp("2026-10-01T10:00:00Z") + pd.Timedelta(minutes=minutes[0])
    end = pd.Timestamp("2026-10-01T10:00:00Z") + pd.Timedelta(minutes=minutes[1])
    return StoredFinding(
        ref=ref, domain=ref.split(":")[0], severity="high", component=component, start_time=start.isoformat(),
        end_time=end.isoformat(), summary=summary, evidence_summary="", verification=verification,
    )


def test_forecast_explanation_is_generated_checked_and_cached(client, monkeypatch):
    import json as json_module

    from AI.db_store import RunAnalyses, StoredExplanation
    from loadlens_app import forecast_explanation as explain

    frames = _forecast_frames()
    _patch_forecast(monkeypatch, _explained_report(frames), frames)
    findings = [
        _finding("jvm:f1", (80, 99), "CPU orders-api 95 % у потолка"),
        _finding("jvm:f2", (80, 99), "CPU 12345 %", verification="unverified"),
        _finding("kafka:f1", (0, 20), "Lag 10 сообщений", component="kafka"),
    ]
    stored = {}
    answer = {
        "headline": "Упёрся orders-api: CPU 95 %",
        "cause": "Похоже, упёрлись в потолок модели 264 RPS: CPU orders-api 95 %.",
        "key_facts": [{"label": "CPU пода", "value": "95 %"}],
        "scaling": {"verdict": "helps", "reason": "Нагрузка упирается в CPU одного сервиса"},
        "evidence": [{"finding_id": "jvm:f1", "role": "CPU у предела"}, {"finding_id": "jvm:f9", "role": "лишняя ссылка"}],
        "scaling_risks": [],
        "next_checks": ["Повторить тест"],
        "confidence": "medium",
    }
    monkeypatch.setattr(explain, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(explain, "load_run_analyses", lambda *args: RunAnalyses(findings=findings, system_context={}))
    monkeypatch.setattr(explain, "load_forecast_explanation", lambda conn, schema, table, run_id: stored.get(run_id))
    monkeypatch.setattr(
        explain, "save_forecast_explanation",
        lambda conn, schema, table, run_id, digest, payload: stored.update({run_id: StoredExplanation(run_id, digest, payload, None)}),
    )
    monkeypatch.setitem(explain.CONFIG, "llm", {"provider": "openai", "openai": {"model": "test-model"}})
    monkeypatch.setattr(explain, "ask_llm_with_text_data", lambda *args, **kwargs: json_module.dumps(answer, ensure_ascii=False))

    first = client.get("/forecast/abc/explanation").get_json()
    assert (first["status"], first["model"], [item["ref"] for item in first["findings"]]) == ("missing", "test-model", ["jvm:f1"])
    assert client.post("/forecast/abc/explanation", data="{}").status_code == 415
    generated = client.post("/forecast/abc/explanation", json={})
    assert generated.status_code == 200
    body = generated.get_json()
    assert body["status"] == "ready" and body["explanation"]["confidence"] == "medium"
    assert body["explanation"]["scaling"]["verdict"] == "helps"
    assert (body["check"]["status"], body["check"]["unknown_findings"]) == ("verified", ["jvm:f9"])
    assert client.get("/forecast/abc/explanation").get_json()["status"] == "ready"
    assert client.post("/forecast/abc/explanation", json={}).status_code == 429
    old = StoredExplanation("abc", stored["abc"].inputs_hash, {"explanation": {"cause": "старый формат"}, "check": {}}, None)
    stored["abc"] = old
    assert client.get("/forecast/abc/explanation").get_json()["status"] == "stale"


def test_forecast_confluence_publication_runs_as_a_job(client, monkeypatch):
    from loadlens_app.blueprints import forecast as forecast_api

    calls = []

    def fake_publish(run_id, target, headroom, ceiling, source_url, progress):
        calls.append((run_id, target, headroom, ceiling, source_url))
        return {"page_id": "7", "page_url": "https://wiki.local/pages/7", "title": "Прогноз мощностей"}

    monkeypatch.setattr(forecast_api, "publish_forecast", fake_publish)
    monkeypatch.setattr(forecast_api, "load_forecast_publication", lambda run_id: None)
    assert client.get("/forecast/abc/confluence").get_json()["page_url"] is None
    assert client.post("/forecast/abc/confluence", data="{}").status_code == 415
    assert client.post("/forecast/abc/confluence", json={"target_rps": -1}).status_code == 400
    resp = client.post("/forecast/abc/confluence", json={"run_name": "nightly", "target_rps": 200, "cpu_ceiling_pct": 70})
    assert resp.status_code == 200
    job = client.get(f"/job_status/{resp.get_json()['job_id']}").get_json()
    assert (job["status"], job["kind"], job["page_url"], job["report_url"]) == (
        "done", "forecast_confluence", "https://wiki.local/pages/7", "/forecasting/abc",
    )
    assert calls == [("abc", 200.0, 20.0, 70.0, "http://localhost/forecasting/abc")]


def test_forecast_confluence_page_has_report_sections(monkeypatch):
    from loadlens_app import forecast as service
    from loadlens_app.forecast_confluence import ForecastScenario, forecast_page_body, render_scale_png, render_steps_png
    from loadlens_app.forecast_explanation import ExplanationView

    frames = _forecast_frames()
    start = frames["RPS"].index[0]
    end = frames["RPS"].index[-1] + pd.Timedelta(minutes=1)
    _patch_forecast(monkeypatch, _forecast_report(start, end), frames)
    report = service.build_forecast("abc", 200.0, 20.0)
    view = ExplanationView("missing", "", [], None, None, "", None, "")
    scenario = ForecastScenario("Цель SLA", 200.0, "выдержит", "ничего", "высокая")
    body = forecast_page_body(report, [scenario], view, "http://localhost/forecasting/abc", 20.0, "06.10.2026 01:00")
    for part in ("Прогноз мощностей: nightly (demo)", "Цель 200 RPS: выдержит", "Сценарии", "forecast-scale.png",
                 "forecast-steps.png", "Открыть прогноз в LoadLens", "Разбор ИИ ещё не делали"):
        assert part in body
    assert render_scale_png(report).startswith(b"\x89PNG")
    assert render_steps_png(report).startswith(b"\x89PNG")


def test_forecasting_pages(client):
    listing = client.get("/forecasting")
    assert listing.status_code == 200 and "Прогнозирование" in listing.get_data(as_text=True)
    detail = client.get("/forecasting/abc")
    assert detail.status_code == 200 and "Прогнозирование" in detail.get_data(as_text=True)
    assert client.get("/forecasting/bad id").status_code == 404


def test_appearance_accent_recolors_pages_and_logo(client):
    from loadlens_app.appearance import accent_palette, palette_css
    from loadlens_app.core import CONFIG

    original = copy.deepcopy(CONFIG.get("appearance"))
    try:
        plain = client.get("/assets/logo.png").data
        tinted = client.get("/assets/logo.png?c=00a884").data
        assert plain.startswith(b"\x89PNG") and tinted.startswith(b"\x89PNG")
        assert plain != tinted
        assert client.get("/assets/logo.png?c=xyz").status_code == 400
        assert client.post("/config", json={"section": "appearance", "data": {"accent": "red"}}).status_code == 400
        assert client.post("/config", json={"section": "appearance", "data": {"accent": "#00A884"}}).status_code == 200
        page = client.get("/").get_data(as_text=True)
        assert "logo.png?c=00a884" in page and "--accent:#00a884" in page
        yellow = accent_palette("#ffeb3b")
        assert yellow.light.on_accent == "#1a1f24"
        assert palette_css(accent_palette("#6200ee")) == ""
    finally:
        CONFIG["appearance"] = original


def test_logo_rejects_a_bad_color_without_reflecting_it(client):
    payload = "<img src=x onerror=alert(document.domain)>"
    resp = client.get("/assets/logo.png", query_string={"c": payload})
    body = resp.get_data(as_text=True)
    assert resp.status_code == 400
    assert resp.mimetype == "application/json"
    assert payload not in body and "<" not in body and "onerror" not in body
    assert "error" in resp.get_json()
