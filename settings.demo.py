# SPDX-License-Identifier: Apache-2.0
"""Settings of the LoadLens demo stack (docker/docker-compose.demo.yml).

The demo compose file mounts this file at /app/settings.py of the app container, so the application
imports it as ``settings`` without any change to the code. Everything points at the containers of the
demo stack: the storage database, a Prometheus stub that answers any PromQL query with synthetic
series and an OpenAI-compatible model stub (tools/stubs). No API key and no external system is needed,
and nothing here is a secret: the database credentials are demo-only values from docker/demo.env.

Production installs keep their own settings.py (copied from settings.example.py) and are not touched
by the demo: the stack has its own project name, volumes and container names.

The file starts from settings.example.py, which ships inside the image next to this file, and overrides
only what the demo needs; the rest (query packs, SLA defaults, limits, authentication) stays as in the
example so the demo shows the product as it is installed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().with_name("settings.example.py")
_spec = importlib.util.spec_from_file_location("loadlens_settings_example", _EXAMPLE)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)
CONFIG = _example.CONFIG

# Service names of docker/docker-compose.demo.yml. The DEMO_* variables come from docker/demo.env through
# the compose file; the defaults are the same values, so the file also works outside compose (tests).
_DB_HOST = os.environ.get("DEMO_DB_HOST") or "timescaledb"
_DB_NAME = os.environ.get("DEMO_DB_NAME") or "loadlens_demo"
_DB_USER = os.environ.get("DEMO_DB_USER") or "loadlens"
_DB_PASSWORD = os.environ.get("DEMO_DB_PASSWORD") or "loadlens-demo-password"
_LLM_STUB_URL = os.environ.get("DEMO_LLM_URL") or "http://llm-stub:5090/v1"
_PROM_STUB_URL = os.environ.get("DEMO_PROM_URL") or "http://prom-stub:5092"

# ---- storage: the TimescaleDB container of the demo stack -------------------------------------------------
CONFIG["storage"]["timescale"].update({
    "host": _DB_HOST,
    "port": 5432,
    "dbname": _DB_NAME,
    "user": _DB_USER,
    "password": _DB_PASSWORD,
    "sslmode": "disable",
})

# ---- model: the OpenAI-compatible stub (tools/stubs/llm_stub.py --lang en) --------------------------------
CONFIG["llm"]["provider"] = "openai"
CONFIG["llm"]["openai"].update({
    "api_base_url": _LLM_STUB_URL,
    "model": "stub-model",
    "api_key": "demo-stub-key",
    "max_concurrent": 4,
    "verify": False,
    "proxies": {"https": "", "http": ""},
    "connect_timeout_sec": 10,
    "request_timeout_sec": 120,
})
# The other providers are not used by the demo and must not point anywhere.
for _provider in ("perplexity", "anthropic", "gigachat"):
    CONFIG["llm"][_provider]["api_key"] = ""
    CONFIG["llm"][_provider]["api_base_url"] = ""

# ---- metrics: the Prometheus stub answers every domain ----------------------------------------------------
CONFIG["data_sources"] = {
    "prometheus": {
        "title": "Prometheus (demo stub)",
        "type": "prometheus",
        "prometheus": {"url": _PROM_STUB_URL},
    },
}
CONFIG["domain_sources"] = {"default": {"source": "prometheus"}}
# One point per 15 s, charts and the step detector on 1-minute buckets (a 60-minute demo window).
CONFIG["default_params"] = {"step": "15s", "resample_interval": "1T"}

# The load tool domain keeps the labels (and so the roles and the SLA references) of the example, which
# the seeded demo run uses as well, so the comparison page lines the two demo runs up; the queries are
# PromQL for the stub instead of the InfluxQL of the k6 → InfluxDB pack.
_lt = CONFIG["queries"]["lt_framework"]
_lt.pop("influxql_queries", None)
_lt.pop("label_tag_keys_list", None)
_lt["promql_queries"] = [
    "sum by (group, name) (rate(k6_http_reqs_total[1m]))",
    "sum by (group, check) (rate(k6_checks_total[1m]))",
    "1000 * histogram_quantile(0.95, sum by (le, group, name) (rate(k6_http_req_duration_seconds_bucket[1m])))",
    "sum by (all) (rate(k6_http_reqs_total[1m]))",
    "1000 * histogram_quantile(0.95, sum by (le) (rate(k6_http_req_duration_seconds_bucket[1m])))",
    "1000 * histogram_quantile(0.99, sum by (le) (rate(k6_http_req_duration_seconds_bucket[1m])))",
    "1000 * sum(rate(k6_http_req_duration_seconds_sum[1m])) / sum(rate(k6_http_req_duration_seconds_count[1m]))",
    "100 * sum(rate(k6_http_req_failed_total[1m])) / sum(rate(k6_http_reqs_total[1m]))",
    "sum by (group, name) (rate(k6_http_req_failed_total[1m]))",
    "max(k6_vus)",
    "sum by (lt_instance) (rate(k6_http_reqs_total[1m]))",
]
_lt["label_keys_list"] = [
    ["group", "name"], ["group", "check"], ["group", "name"], ["all"], [], [], [], [], ["group", "name"], [], ["lt_instance"],
]
_lt["packs"] = []
assert len(_lt["labels"]) == len(_lt["promql_queries"]) == len(_lt["label_keys_list"])

# ---- SLA of the demo service: thresholds the synthetic series of the stub reach -----------------------------
# The stub report settles on a stable step of about 290 RPS with p95 around 45 ms, CPU and memory under 60 %.
CONFIG["sla"].update({
    "target_rps": 250,
    "max_p95_ms": 250,
    "max_cpu_pct": 85,
    "max_memory_pct": 90,
})

# ---- the system under test, as an engineer would describe it for the analysis -----------------------------
CONFIG["system_context"] = {
    "enabled": True,
    "schema_version": 1,
    "system": {
        "name": "Demo checkout",
        "domain": "e-commerce",
        "description": "Order checkout of a small web shop: a Spring Boot API in Kubernetes in front of PostgreSQL.",
        "test_goal": "Confirm that checkout holds 300 RPS with p95 under 250 ms.",
    },
    "architecture": {
        "style": "microservices",
        "components": [
            {"id": "orders-api", "name": "orders-api", "role": "creates and pays orders", "criticality": "high", "technologies": ["Java", "Spring Boot"]},
            {"id": "catalog-api", "name": "catalog-api", "role": "product search and prices", "criticality": "medium", "technologies": ["Java"]},
        ],
        "dependencies": [{"from": "orders-api", "to": "orders-db", "kind": "sync", "purpose": "orders and payments"}],
        "data_stores": [{"id": "orders-db", "type": "PostgreSQL", "purpose": "orders and payments", "used_by": ["orders-api"]}],
    },
    "load_model": {
        "entrypoints": [{"id": "checkout", "name": "POST /checkout", "kind": "http", "business_priority": "high"}],
        "critical_user_flows": [{"id": "buy", "name": "Buy a product", "steps": ["catalog-api", "orders-api"], "success_signals": ["2xx"]}],
        "expected_hotspots": ["orders-api"],
    },
    "operational_context": {
        "known_constraints": ["The database connection pool of orders-api holds 20 connections."],
        "known_risks": ["The connection pool of orders-api runs out under peak load."],
        "normal_degradation_rules": [],
        "analysis_focus": ["p95 of POST /checkout"],
    },
}

# ---- no external systems: Confluence, Grafana, Loki and OpenSearch stay switched off ---------------------------
CONFIG.update({"user": "", "password": "", "grafana_login": "", "grafana_pass": "", "url_basic": "", "grafana_base_url": "", "loki_url": ""})
CONFIG["confluence"].update({"enabled": False, "base_url": "", "username": "", "password": "", "app_url": ""})
CONFIG["logs_source"]["enabled"] = False
CONFIG["logs_source"]["opensearch"]["base_url"] = ""

# ---- interface and reports in English; sign-in on (the first administrator comes from docker/demo.env) --------
CONFIG["appearance"]["language"] = "en"
CONFIG["auth"]["enabled"] = True
