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

from .tools import public_url

LT_DOMAIN = "lt_framework"
NETWORK_NOTE = "Адреса указаны так, как их видит LoadLens: с генератора они могут быть другими."


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
        hint.notes.append(f"У домена «Нагрузка» не настроен источник: {exc}")
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
                hint.notes.append("Prometheus должен принимать remote write: флаг --web.enable-remote-write-receiver.")
        elif family == "gatling":
            host = _host(url) or "victoriametrics"
            hint.env["JAVA_OPTS"] = (
                "-Dgatling.data.writers.0=console -Dgatling.data.writers.1=graphite "
                f"-Dgatling.data.graphite.host={host} -Dgatling.data.graphite.port=2003"
            )
            hint.notes.append("Порт 2003 — Graphite-приёмник VictoriaMetrics (-graphiteListenAddr); у Prometheus его нет.")
            if "lt-gatling-graphite-victoriametrics-labels" in packs:
                hint.notes.append(
                    "Пакет с relabeling ждёт правил -relabelConfig в VictoriaMetrics (docs/load-tests.md); тег прогона — "
                    "-Dgatling.core.outputDirectoryBaseName в шаблоне Gatling."
                )
        elif family == "locust":
            hint.notes.append("Метрики Locust собирает locust_exporter: добавьте его в scrape Prometheus, флаги запуска не нужны.")
        elif family == "jmeter":
            hint.notes.append("JMeter пишет в Prometheus только через плагин; встроенный Backend Listener пишет в InfluxDB.")
    elif kind == "grafana_proxy":
        influx_url, database = _grafana_influx(source)
        database = database or str(((source.get("influxdb") or {}) if isinstance(source.get("influxdb"), dict) else {}).get("database") or "k6")
        influx_url = public_url(influx_url or "http://influxdb:8086").rstrip("/")
        hint.target = f"{influx_url}/{database}"
        if family == "k6":
            hint.params = f"--out influxdb={influx_url}/{database}"
            hint.notes.append("Если InfluxDB требует входа, задайте K6_INFLUXDB_USERNAME и K6_INFLUXDB_PASSWORD.")
        elif family == "jmeter":
            hint.notes.append(
                f"В Backend Listener (InfluxdbBackendListenerClient) укажите influxdbUrl={influx_url}/write?db={database} "
                "и application=${__P(application,app)}: тогда LoadLens отделит прогон."
            )
        elif family == "gatling":
            host = _host(influx_url) or "influxdb"
            hint.env["JAVA_OPTS"] = (
                "-Dgatling.data.writers.0=console -Dgatling.data.writers.1=graphite "
                f"-Dgatling.data.graphite.host={host} -Dgatling.data.graphite.port=2003"
            )
            hint.notes.append("InfluxDB принимает Graphite только с включённым graphite-входом и шаблоном gatling.*.")
        elif family == "locust":
            hint.notes.append("Для InfluxDB Locust нужен сторонний плагин (locust-plugins); встроенного вывода нет.")
    elif kind == "influxdb":
        influx = source.get("influxdb") if isinstance(source.get("influxdb"), dict) else {}
        url = public_url(str(influx.get("url") or "")).rstrip("/")
        hint.target = url
        if family == "k6":
            hint.params = f"--out xk6-influxdb={url}"
            hint.env["K6_INFLUXDB_ORGANIZATION"] = str(influx.get("org") or "")
            hint.env["K6_INFLUXDB_BUCKET"] = str(cfg.get("domain_sources", {}).get(LT_DOMAIN, {}).get("bucket") or "k6")
            hint.notes.append("Вывод в InfluxDB 2 есть в сборке k6 с расширением xk6-output-influxdb; токен задайте в K6_INFLUXDB_TOKEN.")
        else:
            hint.notes.append("Для InfluxDB 2 этот инструмент пишет через плагины; настройте их по документации инструмента.")
    else:
        hint.notes.append(f"Источник типа {kind or 'не задан'}: подсказки нет.")
    if not family:
        hint.notes.append("Подсказки есть для k6, JMeter, Gatling и Locust.")
    if hint.target:
        hint.notes.append(NETWORK_NOTE)
    return hint


__all__ = ["Hint", "output_hint"]
