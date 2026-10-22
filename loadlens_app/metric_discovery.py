# SPDX-License-Identifier: Apache-2.0
"""Names of metrics (Prometheus) or measurements (InfluxDB) that the source of each domain holds.

Used to recommend query packs: a pack lists the names it needs in ``requires_metrics``. Only names are
read, never values, so the requests stay cheap on large servers. Domains bound to the same source
and datasource share one request.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import pandas as pd
import requests

from AI.data_sources import DataSourceConfigError, ResolvedSource, resolve_binding, resolve_domain_source
from AI.i18n import get_locale, t, with_locale
from AI.pipeline import (
    _PROXY_CLOSED,
    _grafana_influx_auth,
    _grafana_proxy_send,
    _http_error_text,
    _lookup_grafana_datasource,
    fetch_influxql_via_grafana,
)
from loadlens_app.query_preview import _fetch_flux_csv

TIMEOUT_SEC = 20
WORKERS = 4
FLUX_MEASUREMENTS = 'import "influxdata/influxdb/schema"\nschema.measurements(bucket: "{bucket}")'
FLUX_MEASUREMENTS_FALLBACK = (
    'from(bucket: "{bucket}") |> range(start: {start}, stop: {end}) '
    '|> keep(columns: ["_measurement"]) |> group() |> distinct(column: "_measurement")'
)
LABEL_VALUES_PATH = "/api/v1/label/__name__/values"


@dataclass(frozen=True)
class DomainNames:
    domain: str
    source: str
    language: str
    names: frozenset[str] = frozenset()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "language": self.language, "count": len(self.names), "error": self.error}


def _source_title(cfg: Mapping[str, Any], source_id: str) -> str:
    catalog = cfg.get("data_sources") if isinstance(cfg.get("data_sources"), Mapping) else {}
    entry = catalog.get(source_id) if isinstance(catalog, Mapping) else None
    title = entry.get("title") if isinstance(entry, Mapping) else ""
    return str(title or source_id)


def _key(resolved: ResolvedSource) -> tuple:
    grafana = resolved.config.get("grafana") if isinstance(resolved.config.get("grafana"), dict) else {}
    datasource = grafana.get("prometheus_datasource") if isinstance(grafana.get("prometheus_datasource"), dict) else {}
    influx = resolved.config.get("influxdb") if isinstance(resolved.config.get("influxdb"), dict) else {}
    return (
        resolved.source_id,
        str(datasource.get("uid") or ""),
        str(datasource.get("name") or ""),
        str(influx.get("database") or ""),
        str(influx.get("bucket") or ""),
    )


def _grafana_mode(g_cfg: dict) -> str:
    """Query language of the binding's Grafana datasource: promql, influxql or flux."""
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    datasource = g_cfg.get("prometheus_datasource") if isinstance(g_cfg.get("prometheus_datasource"), dict) else {}
    uid = str(datasource.get("uid") or "").strip()
    name = str(datasource.get("name") or "").strip()
    if not uid and not name:
        raise RuntimeError(t("server.discovery.no_datasource"))
    path = f"uid/{quote(uid, safe='')}" if uid else f"name/{quote(name, safe='')}"
    resp = requests.get(f"{base_url}/api/datasources/{path}", headers=headers, auth=auth, timeout=TIMEOUT_SEC, verify=verify)
    resp.raise_for_status()
    payload = resp.json()
    data = payload if isinstance(payload, dict) else {}
    if data.get("type") == "prometheus":
        return "promql"
    if data.get("type") in ("influxdb", "influxdb2"):
        version = str(((data.get("jsonData") or {}).get("version")) or "InfluxQL").lower()
        if version == "flux":
            return "flux"
        if version == "influxql":
            return "influxql"
    raise RuntimeError(t("server.discovery.unsupported_type", type=data.get("type") or "?"))


def _label_values(payload: Any) -> frozenset[str]:
    if not isinstance(payload, dict) or payload.get("status") != "success" or not isinstance(payload.get("data"), list):
        raise RuntimeError(t("server.discovery.no_metric_list"))
    return frozenset(str(name) for name in payload["data"])


def _prometheus_names(url: str) -> frozenset[str]:
    resp = requests.get(f"{url.rstrip('/')}{LABEL_VALUES_PATH}", timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    return _label_values(resp.json())


def _grafana_prometheus_names(g_cfg: dict) -> frozenset[str]:
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    ds = _lookup_grafana_datasource(g_cfg, influx=False)

    def get(url: str):
        return requests.get(url, headers=headers, auth=auth, timeout=TIMEOUT_SEC, verify=verify)

    resp = _grafana_proxy_send(g_cfg, ds, LABEL_VALUES_PATH, get)
    if resp.status_code in _PROXY_CLOSED and ds.uid:
        # Grafana without the legacy datasource proxy still serves the resource API.
        resp = get(f"{base_url}/api/datasources/uid/{quote(ds.uid, safe='')}/resources{LABEL_VALUES_PATH}")
    resp.raise_for_status()
    return _label_values(resp.json())


def _influxql_measurements(source: dict) -> frozenset[str]:
    influx = source.get("influxdb") if isinstance(source.get("influxdb"), dict) else {}
    payload = fetch_influxql_via_grafana(source.get("grafana") or {}, "SHOW MEASUREMENTS", influx.get("database"))
    result = ((payload or {}).get("results") or [{}])[0] if isinstance(payload, dict) else {}
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    names = set()
    for series in (result.get("series") if isinstance(result, dict) else None) or []:
        for row in (series.get("values") if isinstance(series, dict) else None) or []:
            if isinstance(row, list) and row:
                names.add(str(row[0]))
    return frozenset(names)


def _flux_measurements(source: dict) -> frozenset[str]:
    end = time.time()
    try:
        csv_text = _fetch_flux_csv(source, FLUX_MEASUREMENTS, end - 3600, end)
    except requests.HTTPError:
        # Flux of InfluxDB 1.8 has no schema package: read the measurements of the last hour instead.
        csv_text = _fetch_flux_csv(source, FLUX_MEASUREMENTS_FALLBACK, end - 3600, end)
    frame = pd.read_csv(io.StringIO(csv_text), comment="#")
    if "_value" not in frame.columns:
        return frozenset()
    return frozenset(str(value) for value in frame["_value"].dropna())


def _fetch(resolved: ResolvedSource) -> tuple[str, frozenset[str]]:
    source = resolved.config
    if resolved.source_type == "prometheus":
        return "promql", _prometheus_names(resolved.prometheus_url)
    if resolved.source_type == "influxdb":
        return "flux", _flux_measurements(source)
    g_cfg = source.get("grafana") or {}
    language = _grafana_mode(g_cfg)
    if language == "promql":
        return language, _grafana_prometheus_names(g_cfg)
    if language == "influxql":
        return language, _influxql_measurements(source)
    return language, _flux_measurements(source)


def _error_text(exc: Exception) -> str:
    if getattr(getattr(exc, "response", None), "status_code", None) is not None:
        return t("server.discovery.source_answered", error=_http_error_text(exc))
    return str(exc).strip() or exc.__class__.__name__


def discover(
    cfg: Mapping[str, Any],
    domains: Iterable[str],
    extra: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, DomainNames]:
    """Names per domain; a domain without a binding or with an unreachable source gets ``error``.

    ``extra`` adds bindings by key (a source of the catalog, the source of a row): their names come back
    under the same key, so a pack can be matched with a source no domain is bound to.
    """
    resolved: dict[str, ResolvedSource] = {}
    out: dict[str, DomainNames] = {}
    for domain in domains:
        try:
            resolved[domain] = resolve_domain_source(dict(cfg), domain)
        except DataSourceConfigError as exc:
            out[domain] = DomainNames(domain, "", "", error=str(exc))
    for key, binding in (extra or {}).items():
        try:
            resolved[key] = resolve_binding(cfg, key, binding)
        except DataSourceConfigError as exc:
            out[key] = DomainNames(key, "", "", error=str(exc))
    unique: dict[tuple, ResolvedSource] = {}
    for item in resolved.values():
        unique.setdefault(_key(item), item)

    # Error texts are read by the caller of the request: the pool threads do not inherit its locale.
    locale = get_locale()

    def run(item: ResolvedSource) -> tuple[str, frozenset[str], str]:
        with with_locale(locale):
            try:
                language, names = _fetch(item)
                return language, names, ""
            except Exception as exc:  # each source fails on its own; the others are still reported
                return "", frozenset(), _error_text(exc)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fetched = dict(zip(unique, pool.map(run, unique.values())))
    for domain, item in resolved.items():
        language, names, error = fetched[_key(item)]
        out[domain] = DomainNames(domain, _source_title(cfg, item.source_id), language, names, error)
    return out


__all__ = ["DomainNames", "discover"]
