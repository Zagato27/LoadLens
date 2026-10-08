"""Runs one metrics query against the configured source and reports whether it returned series."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from AI.data_sources import DataSourceConfigError, resolve_domain_source, source_accepts_language
from AI.pipeline import (
    _convert_pd_offset_to_influx_interval,
    _http_error_text,
    _iso8601_utc,
    _render_influxql_template,
    fetch_influx_data_via_grafana,
    fetch_influxql_via_grafana,
    fetch_metric_series,
)

def parse_preview_ts(value: Any) -> float | None:
    """Unix seconds from an ISO string or a millisecond/second timestamp."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 1e12 else number
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


PREVIEW_WINDOW_SEC = 15 * 60
WINDOW_LABEL = "последние 15 минут"
APP_DOMAINS = frozenset({"jvm", "database", "kafka", "microservices", "hard_resources"})
LT_DOMAIN = "lt_framework"
QUERY_LANGS = frozenset({"promql", "influxql", "flux"})
_SAMPLE_LIMIT = 5


@dataclass
class QueryPreview:
    ok: bool
    message: str
    series_count: int = 0
    point_count: int = 0
    series: list[str] = field(default_factory=list)
    window: str = WINDOW_LABEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "series_count": self.series_count,
            "point_count": self.point_count,
            "series": list(self.series),
            "window": self.window,
        }


def preview_metric_query(
    cfg: dict[str, Any],
    *,
    domain: str,
    lang: str,
    query: str,
    label_keys: list[str] | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> QueryPreview:
    """Executes one query. HTTP and query errors stay in ``message``.

    Without ``start_ts``/``end_ts`` the window is the last 15 minutes.
    """
    text = str(query or "").strip()
    if not text:
        return QueryPreview(ok=False, message="Запрос пустой")
    domain_name = str(domain or "").strip()
    if domain_name not in APP_DOMAINS and domain_name != LT_DOMAIN:
        return QueryPreview(ok=False, message=f"Неизвестный домен «{domain_name}»")
    language = str(lang or "").strip().lower()
    if language not in QUERY_LANGS:
        return QueryPreview(ok=False, message="Укажите язык запроса: promql, influxql или flux")

    try:
        source = _source_for_domain(cfg, domain_name)
    except DataSourceConfigError as exc:
        return QueryPreview(ok=False, message=str(exc))
    rejected = _language_rejection(str(source.get("type") or "").lower(), language)
    if rejected:
        return QueryPreview(ok=False, message=rejected)
    end = float(end_ts) if end_ts is not None else time.time()
    start = float(start_ts) if start_ts is not None else end - PREVIEW_WINDOW_SEC
    if end <= start:
        return QueryPreview(ok=False, message="Окончание окна должно быть позже начала")
    window = _window_label(start, end)
    step = str(((cfg.get("default_params") or {}).get("step")) or "30s")
    keys = [str(item) for item in (label_keys or []) if str(item).strip()]
    try:
        if language == "promql":
            payload = fetch_metric_series(
                "",
                start,
                end,
                text,
                step,
                ef_config={"metrics_source": source, "default_params": cfg.get("default_params") or {}},
            )
            result = _from_prometheus(payload, keys)
        elif language == "influxql":
            result = _run_influxql(source, text, start, end, step, keys)
        else:
            result = _run_flux(source, text, start, end, keys)
    except Exception as exc:
        return QueryPreview(ok=False, message=_source_error(exc), window=window)
    result.window = window
    if result.ok:
        result.message = f"{result.series_count} серий, {result.point_count} точек, {window}"
    elif result.message.startswith("За последние 15 минут") or result.message.startswith("За "):
        result.message = f"За окно {window} серий нет"
    return result


def _window_label(start_ts: float, end_ts: float) -> str:
    start = time.strftime("%d.%m %H:%M", time.localtime(start_ts))
    end = time.strftime("%d.%m %H:%M", time.localtime(end_ts))
    return f"{start}–{end}"


def _language_rejection(source_type: str, language: str) -> str:
    if source_accepts_language(source_type, language):
        return ""
    if source_type == "prometheus":
        return "Этот источник выполняет только PromQL"
    if source_type == "influxdb":
        return "Этот источник выполняет только Flux"
    return f"Источник типа {source_type or 'не задан'} не выполняет {language}"


def _source_for_domain(cfg: dict[str, Any], domain: str) -> dict[str, Any]:
    """Legacy-shaped connection for ``domain`` from the catalog and bindings."""
    config = resolve_domain_source(cfg, domain).config
    return config if isinstance(config, dict) else {}


def _source_error(exc: Exception) -> str:
    """HTTP errors name the request that failed (e.g. the datasource lookup or the query itself)."""
    if getattr(getattr(exc, "response", None), "status_code", None) is not None:
        return f"Источник ответил {_http_error_text(exc)}"
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _finish(names: list[str], point_count: int) -> QueryPreview:
    unique = list(dict.fromkeys(name for name in names if name))
    if not unique or point_count <= 0:
        return QueryPreview(ok=False, message=f"За {WINDOW_LABEL} серий нет", window=WINDOW_LABEL)
    shown = unique[:_SAMPLE_LIMIT]
    return QueryPreview(
        ok=True,
        message=f"{len(unique)} серий, {point_count} точек за {WINDOW_LABEL}",
        series_count=len(unique),
        point_count=point_count,
        series=shown,
        window=WINDOW_LABEL,
    )


def _from_prometheus(payload: dict[str, Any], label_keys: list[str]) -> QueryPreview:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        error = ""
        if isinstance(payload, dict):
            error = str(payload.get("error") or payload.get("errorType") or payload.get("status") or "")
        return QueryPreview(ok=False, message=error or "Prometheus вернул ошибку", window=WINDOW_LABEL)
    result = ((payload.get("data") or {}).get("result") or []) if isinstance(payload.get("data"), dict) else []
    names: list[str] = []
    points = 0
    for series in result:
        if not isinstance(series, dict):
            continue
        metric = series.get("metric") if isinstance(series.get("metric"), dict) else {}
        if label_keys:
            name = "|".join(f"{key}={metric.get(key, 'unknown')}" for key in label_keys)
        else:
            name = str(metric.get("__name__") or "series")
        values = series.get("values") or []
        points += len(values) if isinstance(values, list) else 0
        names.append(name)
    return _finish(names, points)


def _run_influxql(
    source: dict[str, Any],
    query: str,
    start_ts: float,
    end_ts: float,
    step: str,
    label_keys: list[str],
) -> QueryPreview:
    if str(source.get("type") or "").lower() != "grafana_proxy":
        return QueryPreview(ok=False, message="InfluxQL выполняется через источник Grafana proxy", window=WINDOW_LABEL)
    rendered = _render_influxql_template(query, start_ts, end_ts, _convert_pd_offset_to_influx_interval(step))
    influx = source.get("influxdb") if isinstance(source.get("influxdb"), dict) else {}
    payload = fetch_influxql_via_grafana(
        source.get("grafana") or {},
        rendered,
        influx.get("database"),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    result = ((payload or {}).get("results") or [{}])[0] if isinstance(payload, dict) else {}
    if isinstance(result, dict) and result.get("error"):
        return QueryPreview(ok=False, message=str(result.get("error")), window=WINDOW_LABEL)
    series_list = result.get("series") if isinstance(result, dict) else None
    names: list[str] = []
    points = 0
    for series in series_list or []:
        if not isinstance(series, dict):
            continue
        tags = series.get("tags") if isinstance(series.get("tags"), dict) else {}
        if label_keys:
            parts = [f"{key}={tags[key]}" for key in label_keys if key in tags]
            name = "|".join(parts) if parts else str(series.get("name") or "series")
        else:
            name = str(series.get("name") or "series")
        values = series.get("values") or []
        points += len(values) if isinstance(values, list) else 0
        names.append(name)
    return _finish(names, points)


def _run_flux(
    source: dict[str, Any],
    query: str,
    start_ts: float,
    end_ts: float,
    label_keys: list[str],
) -> QueryPreview:
    csv_text = _fetch_flux_csv(source, query, start_ts, end_ts)
    frame = pd.read_csv(io.StringIO(csv_text), comment="#")
    if "_time" not in frame.columns or "_value" not in frame.columns:
        return QueryPreview(ok=False, message=f"За {WINDOW_LABEL} серий нет", window=WINDOW_LABEL)
    frame = frame.dropna(subset=["_time", "_value"])
    if frame.empty:
        return QueryPreview(ok=False, message=f"За {WINDOW_LABEL} серий нет", window=WINDOW_LABEL)
    names = [_flux_series_name(row, label_keys) for _, row in frame.iterrows()]
    return _finish(names, int(frame.shape[0]))


def _flux_series_name(row: pd.Series, label_keys: list[str]) -> str:
    parts = [f"{key}={row[key]}" for key in label_keys if key in row and pd.notnull(row[key])]
    return "|".join(parts) if parts else "series"


def _fetch_flux_csv(source: dict[str, Any], query: str, start_ts: float, end_ts: float) -> str:
    influx = source.get("influxdb") if isinstance(source.get("influxdb"), dict) else {}
    rendered = (
        str(query or "")
        .replace("{bucket}", str(influx.get("bucket") or ""))
        .replace("{start}", _iso8601_utc(start_ts))
        .replace("{end}", _iso8601_utc(end_ts))
    )
    source_type = str(source.get("type") or "").lower()
    if source_type == "grafana_proxy":
        return fetch_influx_data_via_grafana(source.get("grafana") or {}, rendered, start_ts=start_ts, end_ts=end_ts)
    if source_type != "influxdb":
        raise RuntimeError("Flux выполняется для источника InfluxDB или Grafana proxy")
    url = str(influx.get("url") or "").rstrip("/")
    if not url:
        raise RuntimeError("В источнике InfluxDB не задан url")
    response = requests.post(
        f"{url}/api/v2/query",
        params={"org": influx.get("org") or ""},
        headers={
            "Authorization": f"Token {influx.get('token') or ''}",
            "Accept": "application/csv",
            "Content-Type": "application/json",
        },
        json={"query": rendered},
        timeout=60,
    )
    response.raise_for_status()
    return response.text
