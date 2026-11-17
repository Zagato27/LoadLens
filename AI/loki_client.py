# SPDX-License-Identifier: Apache-2.0
"""Transport to Grafana Loki for ``AI.loki_logs``: through a Grafana datasource or directly.

Every read is a ``query_range`` call and comes back as a Loki API payload (``matrix`` for metric queries,
``streams`` for log queries), whichever way it went:

- Grafana: the datasource proxy (``/api/datasources/proxy/uid/<uid>/loki/api/v1/query_range``, the numeric-id
  route for old Grafana); when Grafana closes the proxy (403/404/405), ``/api/ds/query`` and its frames are
  turned back into the Loki payload. Grafana adds the tenant header and the credentials of its datasource.
- Direct: ``<url>/loki/api/v1/query_range`` with an optional ``X-Scope-OrgID`` and basic auth.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional
from urllib.parse import quote

import requests

from AI.data_sources import resolve_logs_loki
from AI.i18n import t
from AI.loki_logs import DEFAULT_TIMEOUT_SEC, Fetch
from AI.pipeline import (
    _PROXY_CLOSED,
    GrafanaDatasource,
    _epoch_seconds,
    _frame_columns,
    _grafana_datasource_from,
    _grafana_influx_auth,
    _grafana_proxy_send,
    _grafana_uid,
    _is_time_field,
    _prometheus_matrix,
    _result_frames,
)

QUERY_RANGE_PATH = "/loki/api/v1/query_range"
_LINE_FIELDS = ("line", "body")
_SKIP_FIELDS = ("labels", "id", "tsns", "labeltypes")


class LokiQueryError(RuntimeError):
    """Loki or Grafana refused a query; the text carries the status and the answer, which name the LogQL error."""


def _error(resp: Any) -> LokiQueryError:
    body = str(getattr(resp, "text", "") or "").strip().replace("\n", " ")
    try:
        parsed = resp.json()
    except Exception:
        parsed = None
    if isinstance(parsed, Mapping):
        body = str(parsed.get("message") or parsed.get("error") or body)
    return LokiQueryError(t("server.logs.loki_query_failed", status=getattr(resp, "status_code", "?"), error=body[:400]))


def _params(query: str, start_ns: int, end_ns: int, step_seconds: Optional[int], limit: Optional[int], direction: Optional[str]) -> dict:
    params: Dict[str, str] = {"query": query, "start": str(int(start_ns)), "end": str(int(end_ns))}
    if step_seconds:
        params["step"] = str(int(step_seconds))
    if limit:
        params["limit"] = str(int(limit))
    if direction:
        params["direction"] = direction
    return params


def _payload(resp: Any) -> dict:
    if resp.status_code >= 400:
        raise _error(resp)
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("status") not in (None, "success"):
        raise LokiQueryError(t("server.logs.loki_query_failed", status=resp.status_code, error=str(payload)[:400]))
    return payload


# ---- direct -----------------------------------------------------------------------------------------------

def loki_base_url(url: str) -> str:
    """The Loki root of an address that may already end with the API path (the legacy ``loki_url`` did)."""
    base = str(url or "").strip().rstrip("/")
    for suffix in (QUERY_RANGE_PATH, "/loki/api/v1/query", "/loki/api/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def fetch_loki_direct(connection: Mapping[str, Any], *, query: str, start_ns: int, end_ns: int, step_seconds: Optional[int] = None,
                      limit: Optional[int] = None, direction: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT_SEC) -> dict:
    headers = {"X-Scope-OrgID": str(connection.get("org_id"))} if connection.get("org_id") else {}
    username = str(connection.get("username") or "")
    auth = (username, str(connection.get("password") or "")) if username else None
    resp = requests.get(
        loki_base_url(str(connection.get("url") or "")) + QUERY_RANGE_PATH,
        params=_params(query, start_ns, end_ns, step_seconds, limit, direction),
        headers=headers, auth=auth, timeout=timeout, verify=bool(connection.get("verify_ssl", True)),
    )
    return _payload(resp)


# ---- through Grafana --------------------------------------------------------------------------------------

def lookup_loki_datasource(g_cfg: dict) -> GrafanaDatasource:
    """The datasource named by ``loki_datasource`` (uid first, then name); both routes carry id and uid."""
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    ds_cfg = g_cfg.get("loki_datasource") if isinstance(g_cfg.get("loki_datasource"), dict) else {}
    uid = str(ds_cfg.get("uid") or "").strip()
    name = str(ds_cfg.get("name") or "").strip()
    path = f"uid/{quote(uid, safe='')}" if uid else f"name/{quote(name, safe='')}"
    resp = requests.get(f"{base_url}/api/datasources/{path}", headers=headers, auth=auth, timeout=30, verify=verify)
    if resp.status_code >= 400:
        raise _error(resp)
    return _grafana_datasource_from(resp.json(), uid)


def fetch_loki_via_grafana(g_cfg: dict, ds: GrafanaDatasource, *, query: str, start_ns: int, end_ns: int,
                           step_seconds: Optional[int] = None, limit: Optional[int] = None, direction: Optional[str] = None,
                           timeout: int = DEFAULT_TIMEOUT_SEC) -> dict:
    _, headers, auth, verify = _grafana_influx_auth(g_cfg)
    params = _params(query, start_ns, end_ns, step_seconds, limit, direction)
    resp = _grafana_proxy_send(
        g_cfg, ds, QUERY_RANGE_PATH,
        lambda url: requests.get(url, headers=headers, auth=auth, params=params, timeout=timeout, verify=verify),
    )
    if resp is not None and resp.status_code not in _PROXY_CLOSED:
        return _payload(resp)
    return _via_ds_query(g_cfg, ds, query, start_ns, end_ns, step_seconds, limit, direction, timeout)


def _via_ds_query(g_cfg: dict, ds: GrafanaDatasource, query: str, start_ns: int, end_ns: int, step_seconds: Optional[int],
                  limit: Optional[int], direction: Optional[str], timeout: int) -> dict:
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    start_ms, end_ms = int(start_ns) // 1_000_000, int(end_ns) // 1_000_000
    model: Dict[str, Any] = {
        "refId": "A",
        "datasource": {"type": "loki", "uid": _grafana_uid(g_cfg, ds)},
        "expr": query,
        "queryType": "range",
        "legendFormat": "",
    }
    if step_seconds:
        model.update({"step": f"{int(step_seconds)}s", "intervalMs": int(step_seconds) * 1000,
                      "maxDataPoints": max(1, (end_ms - start_ms) // (int(step_seconds) * 1000) + 1)})
    else:
        model.update({"maxLines": int(limit or 1000), "direction": direction or "backward"})
    resp = requests.post(
        f"{base_url}/api/ds/query",
        json={"from": str(start_ms), "to": str(end_ms), "queries": [model]},
        headers={**headers, "Content-Type": "application/json", "Accept": "application/json"},
        auth=auth, timeout=timeout, verify=verify,
    )
    if resp.status_code >= 400:
        raise _error(resp)
    body = resp.json()
    result = ((body.get("results") or {}).get("A") or {}) if isinstance(body, dict) else {}
    if isinstance(result, dict) and result.get("error"):
        raise LokiQueryError(t("server.logs.loki_query_failed", status=resp.status_code, error=str(result["error"])[:400]))
    frames = _result_frames(body if isinstance(body, dict) else {})
    if step_seconds:
        return _prometheus_matrix(frames)
    return streams_from_frames(frames)


def _labels(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return {str(key): str(item) for key, item in value.items()} if isinstance(value, Mapping) else {}


def streams_from_frames(frames: list[dict]) -> dict:
    """Grafana log frames as a Loki ``streams`` payload.

    Grafana 8–10 send one frame with ``labels``, ``Time``/``Line``/``tsNs`` columns, Grafana 11 the data-plane names
    (``timestamp``, ``body``); older backends put the labels on the line field instead of a column.
    """
    result: list[dict] = []
    for frame in frames:
        columns, series = _frame_columns(frame)
        names = [str(column.get("name") or "").lower() for column in columns]
        line_index = next((i for i, name in enumerate(names) if name in _LINE_FIELDS), None)
        if line_index is None:
            line_index = next(
                (i for i, column in enumerate(columns)
                 if names[i] not in _SKIP_FIELDS and not _is_time_field(column) and str(column.get("type") or "").lower() == "string"),
                None,
            )
        if line_index is None:
            continue
        ns_index = next((i for i, name in enumerate(names) if name == "tsns"), None)
        time_index = next((i for i, column in enumerate(columns) if _is_time_field(column)), None)
        labels_index = next((i for i, name in enumerate(names) if name == "labels"), None)
        field_labels = _labels(columns[line_index].get("labels"))
        for row, line in enumerate(series[line_index]):
            stamp_ns = None
            if ns_index is not None and row < len(series[ns_index]):
                try:
                    stamp_ns = int(str(series[ns_index][row]))
                except (TypeError, ValueError):
                    stamp_ns = None
            if stamp_ns is None and time_index is not None and row < len(series[time_index]):
                seconds = _epoch_seconds(series[time_index][row])
                stamp_ns = int(seconds * 1_000_000_000) if seconds is not None else None
            if stamp_ns is None:
                continue
            labels = _labels(series[labels_index][row]) if labels_index is not None and row < len(series[labels_index]) else field_labels
            result.append({"stream": labels, "values": [[str(stamp_ns), "" if line is None else str(line)]]})
    return {"status": "success", "data": {"resultType": "streams", "result": result}}


# ---- entry point ------------------------------------------------------------------------------------------

def loki_fetcher(catalog: object, logs_cfg: Mapping[str, Any], timeout: Optional[int] = None) -> tuple[Fetch, str]:
    """``(fetch, where)``: the read function for ``AI.loki_logs`` and how messages name the source.

    Raises ``DataSourceConfigError`` when the connection is not configured. The Grafana datasource is looked up
    once, on the first read.
    """
    connection = resolve_logs_loki(catalog, logs_cfg)
    loki = logs_cfg.get("loki") if isinstance(logs_cfg.get("loki"), Mapping) else {}
    raw_timeout = loki.get("request_timeout_sec")
    seconds = int(timeout or (raw_timeout if isinstance(raw_timeout, int) and raw_timeout > 0 else DEFAULT_TIMEOUT_SEC))
    if connection["mode"] == "grafana":
        g_cfg = connection["grafana"]
        found: list[GrafanaDatasource] = []

        def via_grafana(**kwargs: Any) -> dict:
            if not found:
                found.append(lookup_loki_datasource(g_cfg))
            return fetch_loki_via_grafana(g_cfg, found[0], timeout=seconds, **kwargs)

        return via_grafana, f"{connection['source_id']} / {connection['datasource']}"

    def direct(**kwargs: Any) -> dict:
        return fetch_loki_direct(connection, timeout=seconds, **kwargs)

    return direct, loki_base_url(connection["url"])


__all__ = [
    "LokiQueryError",
    "fetch_loki_direct",
    "fetch_loki_via_grafana",
    "loki_base_url",
    "loki_fetcher",
    "lookup_loki_datasource",
    "streams_from_frames",
]
