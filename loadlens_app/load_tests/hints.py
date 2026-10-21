# SPDX-License-Identifier: Apache-2.0
"""Where a load tool should write its metrics so the service's report sees them.

The suggestion follows the source of the service's ``lt_framework`` domain and its installed query packs:
k6 remote write for Prometheus or VictoriaMetrics, k6 InfluxDB output for InfluxQL behind Grafana, the
Graphite writer of Gatling. Addresses are those LoadLens knows; a generator in another network may need
different ones. Credentials of the source are never put into a suggestion: the engineer adds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

from .messages import t
from .tools import public_url

LT_DOMAIN = "lt_framework"
# The notes are rendered at once, in the language of the request that asks for the suggestion.
NETWORK_NOTE = "lt.hints.network"


@dataclass
class Hint:
    params: str = ""
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"params": self.params, "env": dict(self.env), "notes": list(self.notes), "target": self.target}


def _family(tool_id: str) -> str:
    head = tool_id.split("_")[0].split("-")[0]
    return head if head in ("k6", "jmeter", "gatling", "locust") else ""


def _grafana_influx(config: dict) -> tuple[str, str]:
    """(InfluxDB URL, database) of the bound Grafana datasource; best effort, empty on failure."""
    try:
        import requests

        from AI.pipeline import _datasource_cfg, _grafana_influx_auth

        grafana = config.get("grafana") or {}
        base_url, headers, auth, verify = _grafana_influx_auth(grafana)
        ds = _datasource_cfg(grafana, True)
        uid, name = str(ds.get("uid") or ""), str(ds.get("name") or "")
        if not uid and not name:
            return "", ""
        path = f"uid/{uid}" if uid else f"name/{name}"
        resp = requests.get(f"{base_url}/api/datasources/{path}", headers=headers, auth=auth, timeout=5, verify=verify)
        resp.raise_for_status()
        data = resp.json()
        database = str(data.get("database") or (data.get("jsonData") or {}).get("dbName") or "")
        return str(data.get("url") or ""), database
    except Exception:
        return "", ""


def _host(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def output_hint(cfg: dict, tool_id: str, installed_packs: list[str]) -> Hint:
    from AI.data_sources import DataSourceConfigError, resolve_domain_source

    family = _family(tool_id)
    hint = Hint()
    try:
        resolved = resolve_domain_source(cfg, LT_DOMAIN)
    except DataSourceConfigError as exc:
        hint.notes.append(t("lt.hints.no_source", error=exc))
        return hint
    source = resolved.config if isinstance(resolved.config, dict) else {}
    kind = resolved.source_type
    packs = set(installed_packs)
    if kind == "prometheus":
        url = public_url(resolved.prometheus_url).rstrip("/")
        hint.target = url
        victoria = any("victoriametrics" in pack for pack in packs) or ":8428" in url
        if family == "k6":
            hint.params = "--out experimental-prometheus-rw"
            hint.env["K6_PROMETHEUS_RW_SERVER_URL"] = f"{url}/api/v1/write"
            if "lt-k6-prometheus-native" in packs and "lt-k6-prometheus-rw" not in packs:
                hint.env["K6_PROMETHEUS_RW_TREND_AS_NATIVE_HISTOGRAM"] = "true"
            else:
                hint.env["K6_PROMETHEUS_RW_TREND_STATS"] = "p(95),p(99),avg"
            if not victoria:
                hint.notes.append(t("lt.hints.prometheus_rw"))
        elif family == "gatling":
            host = _host(url) or "victoriametrics"
            hint.env["JAVA_OPTS"] = (
                "-Dgatling.data.writers.0=console -Dgatling.data.writers.1=graphite "
                f"-Dgatling.data.graphite.host={host} -Dgatling.data.graphite.port=2003"
            )
            hint.notes.append(t("lt.hints.gatling_graphite_vm"))
            if "lt-gatling-graphite-victoriametrics-labels" in packs:
                hint.notes.append(t("lt.hints.gatling_relabel"))
        elif family == "locust":
            hint.notes.append(t("lt.hints.locust_exporter"))
        elif family == "jmeter":
            hint.notes.append(t("lt.hints.jmeter_prometheus"))
    elif kind == "grafana_proxy":
        influx_url, database = _grafana_influx(source)
        database = database or str(((source.get("influxdb") or {}) if isinstance(source.get("influxdb"), dict) else {}).get("database") or "k6")
        influx_url = public_url(influx_url or "http://influxdb:8086").rstrip("/")
        hint.target = f"{influx_url}/{database}"
        if family == "k6":
            hint.params = f"--out influxdb={influx_url}/{database}"
            hint.notes.append(t("lt.hints.influx_auth"))
        elif family == "jmeter":
            hint.notes.append(t("lt.hints.jmeter_influx", url=influx_url, database=database, property="${__P(application,app)}"))
        elif family == "gatling":
            host = _host(influx_url) or "influxdb"
            hint.env["JAVA_OPTS"] = (
                "-Dgatling.data.writers.0=console -Dgatling.data.writers.1=graphite "
                f"-Dgatling.data.graphite.host={host} -Dgatling.data.graphite.port=2003"
            )
            hint.notes.append(t("lt.hints.influx_graphite"))
        elif family == "locust":
            hint.notes.append(t("lt.hints.locust_influx"))
    elif kind == "influxdb":
        influx = source.get("influxdb") if isinstance(source.get("influxdb"), dict) else {}
        url = public_url(str(influx.get("url") or "")).rstrip("/")
        hint.target = url
        if family == "k6":
            hint.params = f"--out xk6-influxdb={url}"
            hint.env["K6_INFLUXDB_ORGANIZATION"] = str(influx.get("org") or "")
            hint.env["K6_INFLUXDB_BUCKET"] = str(cfg.get("domain_sources", {}).get(LT_DOMAIN, {}).get("bucket") or "k6")
            hint.notes.append(t("lt.hints.k6_influx2"))
        else:
            hint.notes.append(t("lt.hints.influx2_plugins"))
    else:
        hint.notes.append(t("lt.hints.no_hint", kind=kind or t("lt.hints.kind_unset")))
    if not family:
        hint.notes.append(t("lt.hints.supported_tools"))
    if hint.target:
        hint.notes.append(t(NETWORK_NOTE))
    return hint


__all__ = ["Hint", "output_hint"]
