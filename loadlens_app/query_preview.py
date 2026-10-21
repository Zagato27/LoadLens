# SPDX-License-Identifier: Apache-2.0
"""Runs one metrics query against the configured source and reports whether it returned series."""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from AI.data_sources import DataSourceConfigError, resolve_domain_source, source_accepts_language
from AI.i18n import get_locale, t
from AI.pipeline import (
    _convert_pd_offset_to_influx_interval,
    _epoch_seconds,
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
APP_DOMAINS = frozenset({"jvm", "database", "kafka", "microservices", "hard_resources", "lt_generators"})
LT_DOMAIN = "lt_framework"
QUERY_LANGS = frozenset({"promql", "influxql", "flux"})
_SAMPLE_LIMIT = 5
# Codes of QueryPreview: the request itself is wrong (the API answers 400), the rest are results.
CODE_OK = "ok"
CODE_EMPTY_QUERY = "empty_query"
CODE_UNKNOWN_DOMAIN = "unknown_domain"
CODE_MISSING_LANGUAGE = "missing_language"
CODE_LANGUAGE_REJECTED = "language_rejected"
CODE_SOURCE_CONFIG = "source_config"
CODE_BAD_WINDOW = "bad_window"
CODE_NO_SERIES = "no_series"
CODE_QUERY_ERROR = "query_error"
CODE_SOURCE_ERROR = "source_error"
CODE_UNSUPPORTED_SOURCE = "unsupported_source"
INPUT_ERROR_CODES = frozenset({CODE_EMPTY_QUERY, CODE_UNKNOWN_DOMAIN, CODE_MISSING_LANGUAGE, CODE_LANGUAGE_REJECTED})


def window_last_15() -> str:
    """Label of the default window (the last 15 minutes) in the current locale."""
    return t("server.preview.window_last_15")


@dataclass
class QueryPreview:
    ok: bool
    message: str
    series_count: int = 0
    point_count: int = 0
    series: list[str] = field(default_factory=list)
    window: str = field(default_factory=window_last_15)
    # What happened, independent of the message language (CODE_*); empty when not set.
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
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
        return QueryPreview(ok=False, message=t("server.preview.empty_query"), code=CODE_EMPTY_QUERY)
    domain_name = str(domain or "").strip()
    if domain_name not in APP_DOMAINS and domain_name != LT_DOMAIN:
        return QueryPreview(ok=False, message=t("server.preview.unknown_domain", domain=domain_name), code=CODE_UNKNOWN_DOMAIN)
    language = str(lang or "").strip().lower()
    if language not in QUERY_LANGS:
        return QueryPreview(ok=False, message=t("server.preview.language_required"), code=CODE_MISSING_LANGUAGE)

    try:
        source = _source_for_domain(cfg, domain_name)
    except DataSourceConfigError as exc:
        return QueryPreview(ok=False, message=str(exc), code=CODE_SOURCE_CONFIG)
    rejected = _language_rejection(str(source.get("type") or "").lower(), language)
    if rejected:
        return QueryPreview(ok=False, message=rejected, code=CODE_LANGUAGE_REJECTED)
    end = float(end_ts) if end_ts is not None else time.time()
    start = float(start_ts) if start_ts is not None else end - PREVIEW_WINDOW_SEC
    if end <= start:
        return QueryPreview(ok=False, message=t("server.preview.bad_window"), code=CODE_BAD_WINDOW)
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
        return QueryPreview(ok=False, message=_source_error(exc), window=window, code=CODE_SOURCE_ERROR)
    result.window = window
    if result.ok:
        result.message = t("server.preview.found", series=result.series_count, points=result.point_count, window=window)
    elif result.code == CODE_NO_SERIES:
        result.message = t("server.preview.no_series_window", window=window)
    return result


Points = dict[str, list[tuple[float, float]]]


def _epoch(raw: Any) -> float | None:
    seconds = _epoch_seconds(raw)
    if seconds is not None:
        return seconds
    try:
        stamp = pd.Timestamp(raw)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return float(stamp.timestamp())


def _number(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def fetch_points(
    cfg: dict[str, Any],
    *,
    domain: str,
    lang: str,
    query: str,
    start_ts: float,
    end_ts: float,
    step: str,
    label_keys: list[str] | None = None,
) -> Points:
    """Series of one query as ``{name: [(unix seconds, value), ...]}``; raises on source and query errors."""
    language = str(lang or "").strip().lower()
    source = _source_for_domain(cfg, domain)
    rejected = _language_rejection(str(source.get("type") or "").lower(), language)
    if rejected:
        raise ValueError(rejected)
    keys = [str(item) for item in (label_keys or []) if str(item).strip()]
    out: Points = {}
    if language == "promql":
        payload = fetch_metric_series(
            "", start_ts, end_ts, query, step,
            ef_config={"metrics_source": source, "default_params": cfg.get("default_params") or {}},
        )
        if not isinstance(payload, dict) or payload.get("status") != "success":
            error = str((payload or {}).get("error") or "") if isinstance(payload, dict) else ""
            raise RuntimeError(error or t("server.preview.prometheus_error"))
        for series in ((payload.get("data") or {}).get("result") or []):
            metric = series.get("metric") if isinstance(series.get("metric"), dict) else {}
            name = "|".join(f"{key}={metric.get(key, '')}" for key in keys) if keys else "series"
            points = out.setdefault(name, [])
            for item in series.get("values") or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    ts, value = _epoch(item[0]), _number(item[1])
                    if ts is not None and value is not None:
                        points.append((ts, value))
        return out
    if language == "influxql":
        if str(source.get("type") or "").lower() != "grafana_proxy":
            raise ValueError(t("server.preview.influxql_grafana_only"))
        interval = step if re.fullmatch(r"\d+[smh]", str(step or "")) else _convert_pd_offset_to_influx_interval(step)
        rendered = _render_influxql_template(query, start_ts, end_ts, interval)
        influx = source.get("influxdb") if isinstance(source.get("influxdb"), dict) else {}
        payload = fetch_influxql_via_grafana(source.get("grafana") or {}, rendered, influx.get("database"), start_ts=start_ts, end_ts=end_ts)
        result = ((payload or {}).get("results") or [{}])[0] if isinstance(payload, dict) else {}
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result.get("error")))
        for series in (result.get("series") if isinstance(result, dict) else None) or []:
            tags = series.get("tags") if isinstance(series.get("tags"), dict) else {}
            name = "|".join(f"{key}={tags.get(key, '')}" for key in keys) if keys else "series"
            points = out.setdefault(name, [])
            for row in series.get("values") or []:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    ts, value = _epoch(row[0]), _number(row[1])
                    if ts is not None and value is not None:
                        points.append((ts, value))
        return out
    if language == "flux":
        csv_text = _fetch_flux_csv(source, query, start_ts, end_ts)
        frame = pd.read_csv(io.StringIO(csv_text), comment="#")
        if "_time" not in frame.columns or "_value" not in frame.columns:
            return out
        for _, row in frame.dropna(subset=["_time", "_value"]).iterrows():
            ts, value = _epoch(row["_time"]), _number(row["_value"])
            if ts is None or value is None:
                continue
            name = _flux_series_name(row, keys) if keys else "series"
            out.setdefault(name, []).append((ts, value))
        return out
    raise ValueError(t("server.preview.language_required"))


def _window_label(start_ts: float, end_ts: float) -> str:
    # Day and month in the order of the locale: 18.10 in ru, 18/10 in en (en-GB).
    pattern = "%d/%m %H:%M" if get_locale() == "en" else "%d.%m %H:%M"
    start = time.strftime(pattern, time.localtime(start_ts))
    end = time.strftime(pattern, time.localtime(end_ts))
    return f"{start}–{end}"


def _language_rejection(source_type: str, language: str) -> str:
    if source_accepts_language(source_type, language):
        return ""
    if source_type == "prometheus":
        return t("server.preview.promql_only")
    if source_type == "influxdb":
        return t("server.preview.flux_only")
    return t("server.preview.language_unsupported", type=source_type or t("server.common.unset"), language=language)


def _source_for_domain(cfg: dict[str, Any], domain: str) -> dict[str, Any]:
    """Legacy-shaped connection for ``domain`` from the catalog and bindings."""
    config = resolve_domain_source(cfg, domain).config
    return config if isinstance(config, dict) else {}


def _source_error(exc: Exception) -> str:
    """HTTP errors name the request that failed (e.g. the datasource lookup or the query itself)."""
    if getattr(getattr(exc, "response", None), "status_code", None) is not None:
        return t("server.preview.source_http_error", error=_http_error_text(exc))
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _finish(names: list[str], point_count: int) -> QueryPreview:
    unique = list(dict.fromkeys(name for name in names if name))
    if not unique or point_count <= 0:
        return _no_series()
    shown = unique[:_SAMPLE_LIMIT]
    window = window_last_15()
    return QueryPreview(
        ok=True,
        message=t("server.preview.found_for", series=len(unique), points=point_count, window=window),
        series_count=len(unique),
        point_count=point_count,
        series=shown,
        window=window,
        code=CODE_OK,
    )


def _no_series() -> QueryPreview:
    window = window_last_15()
    return QueryPreview(ok=False, message=t("server.preview.no_series", window=window), window=window, code=CODE_NO_SERIES)


def _from_prometheus(payload: dict[str, Any], label_keys: list[str]) -> QueryPreview:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        error = ""
        if isinstance(payload, dict):
            error = str(payload.get("error") or payload.get("errorType") or payload.get("status") or "")
        return QueryPreview(ok=False, message=error or t("server.preview.prometheus_error"), code=CODE_QUERY_ERROR)
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
        return QueryPreview(ok=False, message=t("server.preview.influxql_grafana_only"), code=CODE_UNSUPPORTED_SOURCE)
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
        return QueryPreview(ok=False, message=str(result.get("error")), code=CODE_QUERY_ERROR)
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
        return _no_series()
    frame = frame.dropna(subset=["_time", "_value"])
    if frame.empty:
        return _no_series()
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
        raise RuntimeError(t("server.preview.flux_source"))
    url = str(influx.get("url") or "").rstrip("/")
    if not url:
        raise RuntimeError(t("server.preview.influxdb_url_missing"))
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
