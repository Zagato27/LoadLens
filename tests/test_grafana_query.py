"""Grafana query falls back to /api/ds/query when the datasource proxy answers 404."""

import types

import pytest
import requests

from AI import pipeline
from loadlens_app.query_preview import _source_error


class _Response:
    def __init__(self, status: int, payload: dict, *, url: str = "", method: str = "GET", text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.url = url
        self.request = types.SimpleNamespace(method=method)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error for url: {self.url}", response=self)


def _grafana(kind: str, datasource: dict | None = None) -> dict:
    key = "prometheus_datasource" if kind == "prometheus" else "influxdb_datasource"
    return {
        "base_url": "http://grafana:3000",
        "verify_ssl": False,
        "auth": {"method": "basic", "username": "a", "password": "b"},
        key: datasource if datasource is not None else {"id": 7, "uid": f"{kind}-uid"},
    }


_MATRIX = {"results": {"A": {"frames": [{
    "schema": {"fields": [
        {"name": "Time", "type": "time"},
        {"name": "Value", "type": "number", "labels": {"application": "orders", "instance": "10.0.0.1:8080"}},
    ]},
    "data": {"values": [[1_700_000_000_000, 1_700_000_030_000], [0.42, 0.5]]},
}]}}}
_PROM_OK = {"status": "success", "data": {"resultType": "matrix", "result": []}}


class _Grafana:
    """Fake Grafana: answers by URL path, records every call."""

    def __init__(self, routes: dict[str, tuple[int, dict]]):
        self.routes = routes
        self.calls: list[tuple[str, str]] = []

    def _answer(self, method: str, url: str, kwargs: dict) -> _Response:
        path = url.replace("http://grafana:3000", "")
        self.calls.append((method, path))
        self.last_json = kwargs.get("json")
        status, payload = self.routes.get(path, (404, {"message": "Not found"}))
        return _Response(status, payload, url=url, method=method, text="" if status < 400 else str(payload))

    def get(self, url, **kwargs):
        return self._answer("GET", url, kwargs)

    def post(self, url, **kwargs):
        return self._answer("POST", url, kwargs)

    def paths(self) -> list[str]:
        return [path for _, path in self.calls]


def _install(monkeypatch, grafana: _Grafana) -> None:
    monkeypatch.setattr(pipeline.requests, "get", grafana.get)
    monkeypatch.setattr(pipeline.requests, "post", grafana.post)


def test_prometheus_uses_ds_query_when_proxy_is_closed(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/uid/prometheus-uid": (200, {"id": 7, "uid": "prometheus-uid"}),
        "/api/ds/query": (200, _MATRIX),
    })
    _install(monkeypatch, grafana)
    payload = pipeline.fetch_prometheus_data_via_grafana(
        _grafana("prometheus"), 1_700_000_000, 1_700_000_300, "process_cpu_usage", "30s",
    )
    series = payload["data"]["result"][0]
    assert payload["status"] == "success"
    assert series["metric"] == {"application": "orders", "instance": "10.0.0.1:8080"}
    assert series["values"] == [[1_700_000_000.0, "0.42"], [1_700_000_030.0, "0.5"]]
    assert grafana.last_json["queries"][0]["expr"] == "process_cpu_usage"
    assert grafana.last_json["queries"][0]["datasource"] == {"type": "prometheus", "uid": "prometheus-uid"}


def test_prometheus_proxy_response_is_used_as_is(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/uid/prometheus-uid": (200, {"id": 7, "uid": "prometheus-uid"}),
        "/api/datasources/proxy/uid/prometheus-uid/api/v1/query_range": (200, _PROM_OK),
    })
    _install(monkeypatch, grafana)
    payload = pipeline.fetch_prometheus_data_via_grafana(
        _grafana("prometheus"), 1_700_000_000, 1_700_000_300, "up", "30s",
    )
    assert payload["data"]["result"] == []
    assert "/api/ds/query" not in grafana.paths()


def test_name_only_binding_never_needs_numeric_id_endpoints(monkeypatch):
    # Bindings migrated from the old config name the datasource without a uid. Newer Grafana has
    # no numeric-id endpoints, so the uid must come from the name lookup itself.
    grafana = _Grafana({
        "/api/datasources/name/Prometheus-main": (200, {"id": 12, "uid": "prom-uid"}),
        "/api/ds/query": (200, _MATRIX),
    })
    _install(monkeypatch, grafana)
    payload = pipeline.fetch_prometheus_data_via_grafana(
        _grafana("prometheus", {"id": None, "uid": "", "name": "Prometheus-main"}), 1_700_000_000, 1_700_000_300, "up", "30s",
    )
    assert payload["data"]["result"][0]["metric"]["application"] == "orders"
    assert grafana.last_json["queries"][0]["datasource"]["uid"] == "prom-uid"
    assert "/api/datasources/12" not in grafana.paths()
    assert grafana.paths()[:3] == [
        "/api/datasources/name/Prometheus-main",
        "/api/datasources/proxy/uid/prom-uid/api/v1/query_range",
        "/api/datasources/proxy/12/api/v1/query_range",
    ]


def test_old_grafana_without_uid_proxy_uses_the_id_route(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/name/Prom": (200, {"id": 12, "uid": "prom-uid"}),
        "/api/datasources/proxy/12/api/v1/query_range": (200, _PROM_OK),
    })
    _install(monkeypatch, grafana)
    payload = pipeline.fetch_prometheus_data_via_grafana(
        _grafana("prometheus", {"name": "Prom"}), 1_700_000_000, 1_700_000_300, "up", "30s",
    )
    assert payload["status"] == "success"
    assert "/api/ds/query" not in grafana.paths()


def test_datasource_name_is_url_encoded(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/name/Prom%2FHF%20%231": (200, {"id": 3, "uid": "u3"}),
        "/api/datasources/proxy/uid/u3/api/v1/query_range": (200, _PROM_OK),
    })
    _install(monkeypatch, grafana)
    pipeline.fetch_prometheus_data_via_grafana(_grafana("prometheus", {"name": "Prom/HF #1"}), 1_700_000_000, 1_700_000_300, "up", "30s")
    assert grafana.paths()[0] == "/api/datasources/name/Prom%2FHF%20%231"


def test_legacy_id_only_binding_skips_the_lookup(monkeypatch):
    grafana = _Grafana({"/api/datasources/proxy/7/api/v1/query_range": (200, _PROM_OK)})
    _install(monkeypatch, grafana)
    pipeline.fetch_prometheus_data_via_grafana(_grafana("prometheus", {"id": 7}), 1_700_000_000, 1_700_000_300, "up", "30s")
    assert grafana.paths() == ["/api/datasources/proxy/7/api/v1/query_range"]


def test_fallback_error_names_both_attempts(monkeypatch):
    grafana = _Grafana({"/api/datasources/name/Prom": (200, {"id": 12, "uid": "prom-uid"})})
    _install(monkeypatch, grafana)
    with pytest.raises(pipeline.GrafanaFallbackError) as caught:
        pipeline.fetch_prometheus_data_via_grafana(_grafana("prometheus", {"name": "Prom"}), 1_700_000_000, 1_700_000_300, "up", "30s")
    message = str(caught.value)
    assert "прокси датасорса ответил 404" in message
    assert "404 POST /api/ds/query" in message and "Not found" in message
    assert _source_error(caught.value) == message


def test_lookup_error_names_the_request():
    response = _Response(404, {}, url="http://grafana:3000/api/datasources/name/Prom?x=1", text='{"message":"Data source not found"}')
    message = _source_error(requests.HTTPError(response=response))
    assert message == 'Источник ответил 404 GET /api/datasources/name/Prom: {"message":"Data source not found"}'


def test_resolvers_return_the_grafana_id(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/name/Prom": (200, {"id": 12, "uid": "prom-uid"}),
        "/api/datasources": (200, [{"id": 4, "uid": "i4", "type": "influxdb"}, {"id": 5, "uid": "p5", "type": "prometheus"}]),
    })
    _install(monkeypatch, grafana)
    assert pipeline._resolve_grafana_prom_ds_id(_grafana("prometheus", {"name": "Prom"})) == 12
    assert pipeline._resolve_grafana_prom_ds_id(_grafana("prometheus", {})) == 5
    assert pipeline._resolve_grafana_influx_ds_id(_grafana("influx", {})) == 4


def test_flux_uses_ds_query_when_proxy_is_closed(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/uid/influx-uid": (200, {"id": 7, "uid": "influx-uid"}),
        "/api/ds/query": (200, {"results": {"A": {"frames": [{
            "schema": {"fields": [
                {"name": "_time", "type": "time"},
                {"name": "_value", "type": "number", "labels": {"group": "checkout"}},
            ]},
            "data": {"values": [[1_700_000_000_000], [12.5]]},
        }]}}}),
    })
    _install(monkeypatch, grafana)
    text = pipeline.fetch_influx_data_via_grafana(
        _grafana("influx"), 'from(bucket: "lt")', start_ts=1_700_000_000, end_ts=1_700_000_300,
    )
    assert grafana.last_json["queries"][0]["queryType"] == "flux"
    assert text.splitlines()[0] == "_time,_value,group"
    assert "12.5" in text and "checkout" in text


def test_influxql_name_only_binding_uses_uid_routes(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/name/InfluxDB-k6": (200, {"id": 9, "uid": "k6-uid"}),
        "/api/datasources/proxy/uid/k6-uid/query": (200, {"results": [{"series": []}]}),
    })
    _install(monkeypatch, grafana)
    payload = pipeline.fetch_influxql_via_grafana(
        _grafana("influx", {"name": "InfluxDB-k6"}), "SELECT 1", "k6", start_ts=1_700_000_000, end_ts=1_700_000_300,
    )
    assert payload == {"results": [{"series": []}]}
    assert grafana.paths() == ["/api/datasources/name/InfluxDB-k6", "/api/datasources/proxy/uid/k6-uid/query"]


def test_ds_query_error_is_reported(monkeypatch):
    grafana = _Grafana({
        "/api/datasources/uid/prometheus-uid": (200, {"id": 7, "uid": "prometheus-uid"}),
        "/api/ds/query": (200, {"results": {"A": {"error": "parse error: bad query"}}}),
    })
    _install(monkeypatch, grafana)
    with pytest.raises(RuntimeError, match="bad query"):
        pipeline.fetch_prometheus_data_via_grafana(
            _grafana("prometheus"), 1_700_000_000, 1_700_000_300, "???", "30s",
        )
