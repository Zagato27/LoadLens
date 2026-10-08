import json
from datetime import datetime, timezone
from typing import Any, Dict

from opensearch_error_report import (
    AggregationConfig,
    AppConfig,
    FieldsConfig,
    FiltersConfig,
    OpenSearchConfig,
    ReportConfig,
    aggregate_events,
    fetch_error_events,
    parse_resample_interval_seconds,
    render_markdown,
    to_opensearch_datetime,
)


APPLICATION_LOGS_DOMAIN = "application_logs"


def _cfg_dict(value: object) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _cfg_str(config: Dict[str, Any], key: str, default: str = "") -> str:
    value = config.get(key, default)
    return value if isinstance(value, str) else default


def _cfg_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    return value if isinstance(value, bool) else default


def _cfg_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return value if isinstance(value, int) else default


def _cfg_str_list(config: Dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = config.get(key, default)
    if not isinstance(value, list):
        return list(default)
    return [item for item in value if isinstance(item, str)]


def _positive_cfg_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = _cfg_int(config, key, default)
    return value if value > 0 else default


def _to_utc_datetime(timestamp_sec: float) -> datetime:
    return datetime.fromtimestamp(float(timestamp_sec), tz=timezone.utc)


def build_app_config(logs_cfg: Dict[str, Any], resample_interval: str) -> AppConfig:
    opensearch = _cfg_dict(logs_cfg.get("opensearch"))
    fields = _cfg_dict(logs_cfg.get("fields"))
    filters = _cfg_dict(logs_cfg.get("filters"))
    aggregation = _cfg_dict(logs_cfg.get("aggregation"))
    report = _cfg_dict(logs_cfg.get("report"))

    return AppConfig(
        opensearch=OpenSearchConfig(
            base_url=_cfg_str(opensearch, "base_url").rstrip("/"),
            index_pattern=_cfg_str(opensearch, "index_pattern"),
            username_env=_cfg_str(opensearch, "username_env"),
            password_env=_cfg_str(opensearch, "password_env"),
            verify_ssl=_cfg_bool(opensearch, "verify_ssl", True),
            request_timeout_sec=_cfg_int(opensearch, "request_timeout_sec", 30),
        ),
        fields=FieldsConfig(
            timestamp=_cfg_str(fields, "timestamp", "@timestamp"),
            level=_cfg_str(fields, "level", "level"),
            level_query=_cfg_str(fields, "level_query", _cfg_str(fields, "level", "level")),
            hostname=_cfg_str(fields, "hostname", "hostname"),
            hostname_query=_cfg_str(fields, "hostname_query", _cfg_str(fields, "hostname", "hostname")),
            message=_cfg_str(fields, "message", "message"),
        ),
        filters=FiltersConfig(
            error_levels=_cfg_str_list(filters, "error_levels", ["ERROR"]),
            hostnames=_cfg_str_list(filters, "hostnames", []),
        ),
        aggregation=AggregationConfig(
            resample_interval=resample_interval or _cfg_str(aggregation, "resample_interval", "5T"),
            max_documents=_cfg_int(aggregation, "max_documents", 10000),
            page_size=_cfg_int(aggregation, "page_size", 1000),
            top_error_types_per_service=_cfg_int(aggregation, "top_error_types_per_service", 20),
            samples_per_error_type=_cfg_int(aggregation, "samples_per_error_type", 2),
            max_error_type_length=_cfg_int(aggregation, "max_error_type_length", 300),
        ),
        report=ReportConfig(
            output_path=_cfg_str(report, "output_path", "opensearch_error_report.md"),
            default_last_minutes=_cfg_int(report, "default_last_minutes", 60),
        ),
    )


def _buckets_to_pack(
    app_config: AppConfig,
    buckets: list[Any],
    events_count: int,
    truncated: bool,
    context_limits: Dict[str, int],
) -> Dict[str, Any]:
    intervals: list[Dict[str, Any]] = []
    total_errors = 0
    affected_services: set[str] = set()
    top_error_counter: Dict[str, int] = {}
    service_error_counter: Dict[str, int] = {}

    top_intervals_limit = context_limits["top_intervals"]
    top_services_limit = context_limits["top_services_per_interval"]
    top_types_limit = context_limits["top_error_types_per_service"]
    top_global_types_limit = context_limits["top_global_error_types"]
    samples_limit = context_limits["samples_per_error_type"]
    sample_max_length = context_limits["sample_max_length"]
    error_type_max_length = context_limits["error_type_max_length"]

    for bucket in buckets:
        total_errors += int(bucket.total_errors)

        for hostname, host_bucket in bucket.hosts.items():
            hostname_str = str(hostname)
            affected_services.add(hostname_str)
            service_error_counter[hostname_str] = service_error_counter.get(hostname_str, 0) + int(host_bucket.total_errors)
            for error_type, error_bucket in host_bucket.error_types.items():
                error_type_str = str(error_type)
                top_error_counter[error_type_str] = top_error_counter.get(error_type_str, 0) + int(error_bucket.count)

    selected_buckets = sorted(buckets, key=lambda item: item.total_errors, reverse=True)[:top_intervals_limit]
    selected_buckets = sorted(selected_buckets, key=lambda item: item.start)

    for bucket in selected_buckets:
        interval_services: list[Dict[str, Any]] = []
        hosts = sorted(bucket.hosts.items(), key=lambda item: item[1].total_errors, reverse=True)
        for hostname, host_bucket in hosts[:top_services_limit]:
            error_types = sorted(host_bucket.error_types.items(), key=lambda item: item[1].count, reverse=True)
            service_error_types: list[Dict[str, Any]] = []
            for error_type, error_bucket in error_types[:top_types_limit]:
                error_type_text = str(error_type)[:error_type_max_length]
                samples = [
                    str(sample)[:sample_max_length]
                    for sample in list(error_bucket.samples)[:samples_limit]
                ]
                service_error_types.append(
                    {
                        "type": error_type_text,
                        "count": int(error_bucket.count),
                        "samples": samples,
                    }
                )
            interval_services.append(
                {
                    "hostname": str(hostname),
                    "total_errors": int(host_bucket.total_errors),
                    "error_types": service_error_types,
                    "omitted_error_types": max(0, len(error_types) - top_types_limit),
                }
            )
        intervals.append(
            {
                "start_time": to_opensearch_datetime(bucket.start),
                "end_time": to_opensearch_datetime(bucket.end),
                "total_errors": int(bucket.total_errors),
                "services": interval_services,
                "omitted_services": max(0, len(hosts) - top_services_limit),
            }
        )

    top_error_types = [
        {"type": error_type[:error_type_max_length], "count": count}
        for error_type, count in sorted(top_error_counter.items(), key=lambda item: item[1], reverse=True)[
            :top_global_types_limit
        ]
    ]
    top_services = [
        {"hostname": hostname, "count": count}
        for hostname, count in sorted(service_error_counter.items(), key=lambda item: item[1], reverse=True)[
            :top_services_limit
        ]
    ]

    return {
        "summary": {
            "total_errors": total_errors,
            "fetched_documents": int(events_count),
            "affected_services_count": len(affected_services),
            "affected_services": sorted(affected_services)[:top_services_limit],
            "omitted_affected_services": max(0, len(affected_services) - top_services_limit),
            "truncated": bool(truncated),
            "interval": app_config.aggregation.resample_interval,
            "error_levels": list(app_config.filters.error_levels),
            "total_intervals_with_errors": len(buckets),
            "reported_intervals": len(intervals),
            "omitted_intervals": max(0, len(buckets) - len(intervals)),
        },
        "top_services": top_services,
        "top_error_types": top_error_types,
        "intervals": intervals,
    }


def collect_application_logs(
    start_ts: float,
    end_ts: float,
    logs_cfg: Dict[str, Any],
    resample_interval: str,
) -> Dict[str, Any]:
    app_config = build_app_config(logs_cfg, resample_interval)
    aggregation_cfg = _cfg_dict(logs_cfg.get("aggregation"))
    context_limits = {
        "top_intervals": _positive_cfg_int(aggregation_cfg, "context_top_intervals", 12),
        "top_services_per_interval": _positive_cfg_int(aggregation_cfg, "context_top_services_per_interval", 3),
        "top_error_types_per_service": _positive_cfg_int(aggregation_cfg, "context_top_error_types_per_service", 3),
        "top_global_error_types": _positive_cfg_int(aggregation_cfg, "context_top_global_error_types", 25),
        "samples_per_error_type": _positive_cfg_int(aggregation_cfg, "context_samples_per_error_type", 1),
        "sample_max_length": _positive_cfg_int(aggregation_cfg, "context_sample_max_length", 160),
        "error_type_max_length": _positive_cfg_int(aggregation_cfg, "context_error_type_max_length", 180),
    }
    start = _to_utc_datetime(start_ts)
    end = _to_utc_datetime(end_ts)
    interval_seconds = parse_resample_interval_seconds(app_config.aggregation.resample_interval)
    fetch_result = fetch_error_events(app_config, start, end, debug=False)
    buckets = aggregate_events(
        fetch_result.events,
        interval_seconds=interval_seconds,
        samples_limit=app_config.aggregation.samples_per_error_type,
    )
    markdown = render_markdown(
        config=app_config,
        start=start,
        end=end,
        buckets=buckets,
        events_count=len(fetch_result.events),
        truncated=fetch_result.truncated,
    )
    pack = _buckets_to_pack(
        app_config=app_config,
        buckets=buckets,
        events_count=len(fetch_result.events),
        truncated=fetch_result.truncated,
        context_limits=context_limits,
    )
    ctx = json.dumps(
        {
            "domain": APPLICATION_LOGS_DOMAIN,
            "time_range": {"start": start_ts, "end": end_ts},
            **pack,
        },
        ensure_ascii=False,
    )
    return {"labeled": [], "markdown": markdown, "pack": pack, "ctx": ctx}
