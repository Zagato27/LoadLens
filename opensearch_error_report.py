import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import urllib3


TRACE_ID_JSON_RE = re.compile(
    r"([\"'])(rqUid|rqUID|requestId|traceId|spanId|correlationId)\1\s*:\s*([\"'])([^\"']{8,})\3"
)
TRACE_ID_KEY_VALUE_RE = re.compile(
    r"\b(rqUid|rqUID|requestId|traceId|spanId|correlationId)\s*([=:])\s*[\"']?([A-Za-z0-9._:-]{8,})[\"']?"
)
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
HEX_ID_RE = re.compile(r"\b[0-9a-fA-F]{24,64}\b")
LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class OpenSearchConfig:
    base_url: str
    index_pattern: str
    username_env: str
    password_env: str
    verify_ssl: bool
    request_timeout_sec: int


@dataclass(frozen=True)
class FieldsConfig:
    timestamp: str
    level: str
    level_query: str
    hostname: str
    hostname_query: str
    message: str


@dataclass(frozen=True)
class FiltersConfig:
    error_levels: list[str]
    hostnames: list[str]


@dataclass(frozen=True)
class AggregationConfig:
    resample_interval: str
    max_documents: int
    page_size: int
    top_error_types_per_service: int
    samples_per_error_type: int
    max_error_type_length: int


@dataclass(frozen=True)
class ReportConfig:
    output_path: str
    default_last_minutes: int


@dataclass(frozen=True)
class AppConfig:
    opensearch: OpenSearchConfig
    fields: FieldsConfig
    filters: FiltersConfig
    aggregation: AggregationConfig
    report: ReportConfig


@dataclass(frozen=True)
class ErrorEvent:
    timestamp: datetime
    hostname: str
    message: str
    normalized_message: str


@dataclass
class ErrorTypeBucket:
    count: int
    samples: list[str]


@dataclass
class HostBucket:
    total_errors: int
    error_types: dict[str, ErrorTypeBucket]


@dataclass
class IntervalBucket:
    start: datetime
    end: datetime
    total_errors: int
    hosts: dict[str, HostBucket]


@dataclass(frozen=True)
class FetchResult:
    events: list[ErrorEvent]
    truncated: bool


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{name}' must be an object")
    return value


def _get_str(config: Mapping[str, object], key: str, default: str = "") -> str:
    value = config.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Config value '{key}' must be a string")
    return value


def _get_bool(config: Mapping[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be a boolean")
    return value


def _get_int(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Config value '{key}' must be an integer")
    return value


def _get_str_list(config: Mapping[str, object], key: str, default: Sequence[str]) -> list[str]:
    value = config.get(key, list(default))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Config value '{key}' must be a list of strings")
    return list(value)


def load_config(path: Path) -> AppConfig:
    raw_config = json.loads(path.read_text(encoding="utf-8"))
    root = _require_mapping(raw_config, "root")
    opensearch = _require_mapping(root.get("opensearch"), "opensearch")
    fields = _require_mapping(root.get("fields"), "fields")
    filters = _require_mapping(root.get("filters"), "filters")
    aggregation = _require_mapping(root.get("aggregation"), "aggregation")
    report = _require_mapping(root.get("report"), "report")

    return AppConfig(
        opensearch=OpenSearchConfig(
            base_url=_get_str(opensearch, "base_url").rstrip("/"),
            index_pattern=_get_str(opensearch, "index_pattern"),
            username_env=_get_str(opensearch, "username_env"),
            password_env=_get_str(opensearch, "password_env"),
            verify_ssl=_get_bool(opensearch, "verify_ssl", True),
            request_timeout_sec=_get_int(opensearch, "request_timeout_sec", 30),
        ),
        fields=FieldsConfig(
            timestamp=_get_str(fields, "timestamp", "@timestamp"),
            level=_get_str(fields, "level", "level"),
            level_query=_get_str(fields, "level_query", _get_str(fields, "level", "level")),
            hostname=_get_str(fields, "hostname", "hostname"),
            hostname_query=_get_str(fields, "hostname_query", _get_str(fields, "hostname", "hostname")),
            message=_get_str(fields, "message", "message"),
        ),
        filters=FiltersConfig(
            error_levels=_get_str_list(filters, "error_levels", ["ERROR"]),
            hostnames=_get_str_list(filters, "hostnames", []),
        ),
        aggregation=AggregationConfig(
            resample_interval=_get_str(aggregation, "resample_interval", "5T"),
            max_documents=_get_int(aggregation, "max_documents", 10000),
            page_size=_get_int(aggregation, "page_size", 1000),
            top_error_types_per_service=_get_int(aggregation, "top_error_types_per_service", 20),
            samples_per_error_type=_get_int(aggregation, "samples_per_error_type", 2),
            max_error_type_length=_get_int(aggregation, "max_error_type_length", 300),
        ),
        report=ReportConfig(
            output_path=_get_str(report, "output_path", "opensearch_error_report.md"),
            default_last_minutes=_get_int(report, "default_last_minutes", 60),
        ),
    )


def parse_datetime(value: object) -> datetime:
    if isinstance(value, (int, float)):
        timestamp = float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid datetime value: {value!r}")

    normalized = value.strip()
    normalized = re.sub(r"(\.\d{6})\d+(Z|[+-]\d{2}:\d{2})$", r"\1\2", normalized)
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_opensearch_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_resample_interval_seconds(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]+)\s*", value)
    if not match:
        raise ValueError(f"Unsupported resample_interval: {value!r}")

    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return amount
    if unit in {"t", "m", "min", "mins", "minute", "minutes"}:
        return amount * 60
    if unit in {"h", "hour", "hours"}:
        return amount * 60 * 60
    if unit in {"d", "day", "days"}:
        return amount * 60 * 60 * 24
    raise ValueError(f"Unsupported resample_interval unit: {value!r}")


def floor_datetime(value: datetime, interval_seconds: int) -> datetime:
    epoch_seconds = int(value.timestamp())
    floored_seconds = epoch_seconds - (epoch_seconds % interval_seconds)
    return datetime.fromtimestamp(floored_seconds, tz=timezone.utc)


def get_field_value(source: Mapping[str, object], field_path: str) -> object:
    if field_path in source:
        return source[field_path]

    current: object = source
    for part in field_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def stringify_field(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def normalize_trace_key(value: str) -> str:
    if value == "rqUID":
        return "rqUid"
    return value


def normalize_message(message: str, max_length: int) -> str:
    normalized = TRACE_ID_JSON_RE.sub(
        lambda match: (
            f'{match.group(1)}{normalize_trace_key(match.group(2))}{match.group(1)}: '
            f'"<{normalize_trace_key(match.group(2))}>"'
        ),
        message,
    )
    normalized = TRACE_ID_KEY_VALUE_RE.sub(
        lambda match: f"{normalize_trace_key(match.group(1))}{match.group(2)}<{normalize_trace_key(match.group(1))}>",
        normalized,
    )
    normalized = UUID_RE.sub("<uuid>", normalized)
    normalized = HEX_ID_RE.sub("<hex_id>", normalized)
    normalized = LONG_NUMBER_RE.sub("<id>", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()

    if len(normalized) > max_length:
        return f"{normalized[:max_length].rstrip()}..."
    return normalized or "<empty_message>"


def resolve_credentials(config: OpenSearchConfig) -> tuple[str, str]:
    username = config.username_env.strip()
    password = config.password_env.strip()
    if not username:
        raise ValueError("Config value 'opensearch.username_env' must contain OpenSearch username")
    if not password:
        raise ValueError("Config value 'opensearch.password_env' must contain OpenSearch password")
    return username, password


def build_search_body(
    config: AppConfig,
    start: datetime,
    end: datetime,
    size: int,
    search_after: Sequence[object] | None,
) -> dict[str, object]:
    level_should: list[dict[str, object]] = []
    for level in config.filters.error_levels:
        if config.fields.level_query:
            level_should.append({"term": {config.fields.level_query: level}})
        if config.fields.level:
            level_should.append({"term": {config.fields.level: level}})

    filters: list[dict[str, object]] = [
        {
            "range": {
                config.fields.timestamp: {
                    "gte": to_opensearch_datetime(start),
                    "lt": to_opensearch_datetime(end),
                }
            }
        }
    ]

    if level_should:
        filters.append(
            {
                "bool": {
                    "should": level_should,
                    "minimum_should_match": 1,
                }
            }
        )

    if config.filters.hostnames:
        hostname_should: list[dict[str, object]] = []
        if config.fields.hostname_query:
            hostname_should.append({"terms": {config.fields.hostname_query: config.filters.hostnames}})
        if config.fields.hostname:
            hostname_should.append({"terms": {config.fields.hostname: config.filters.hostnames}})

        if hostname_should:
            filters.append(
                {
                    "bool": {
                        "should": hostname_should,
                        "minimum_should_match": 1,
                    }
                }
            )

    body: dict[str, object] = {
        "size": size,
        "track_total_hits": True,
        "_source": [
            config.fields.timestamp,
            config.fields.level,
            config.fields.hostname,
            config.fields.message,
            "pod",
            "loggerName",
            "mdc",
        ],
        "sort": [
            {config.fields.timestamp: {"order": "asc"}}
        ],
        "query": {
            "bool": {
                "filter": filters,
            }
        },
    }

    if search_after:
        body["search_after"] = list(search_after)

    return body


def extract_hits(response_payload: object) -> list[Mapping[str, object]]:
    root = _require_mapping(response_payload, "OpenSearch response")
    hits_section = _require_mapping(root.get("hits"), "OpenSearch response.hits")
    raw_hits = hits_section.get("hits", [])
    if not isinstance(raw_hits, list):
        raise ValueError("OpenSearch response.hits.hits must be a list")

    hits: list[Mapping[str, object]] = []
    for hit in raw_hits:
        if isinstance(hit, Mapping):
            hits.append(hit)
    return hits


def hit_sort_values(hit: Mapping[str, object]) -> Sequence[object] | None:
    sort_values = hit.get("sort")
    if isinstance(sort_values, Sequence) and not isinstance(sort_values, str):
        return sort_values
    return None


def fetch_error_events(config: AppConfig, start: datetime, end: datetime, debug: bool = False) -> FetchResult:
    username, password = resolve_credentials(config.opensearch)

    if not config.opensearch.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    proxy_url = f"{config.opensearch.base_url}/api/console/proxy"
    proxy_headers = {
        "osd-xsrf": "true",
        "Content-Type": "application/json",
    }

    page_size = max(1, min(config.aggregation.page_size, config.aggregation.max_documents))
    events: list[ErrorEvent] = []
    search_after: Sequence[object] | None = None
    session = requests.Session()
    debug_printed = False

    while len(events) < config.aggregation.max_documents:
        remaining = config.aggregation.max_documents - len(events)
        current_size = min(page_size, remaining)
        body = build_search_body(config, start, end, current_size, search_after)

        response = session.post(
            proxy_url,
            params={
                "path": f"/{config.opensearch.index_pattern}/_search",
                "method": "POST",
                "dataSourceId": "",
            },
            json=body,
            headers=proxy_headers,
            auth=(username, password),
            verify=config.opensearch.verify_ssl,
            timeout=config.opensearch.request_timeout_sec,
        )

        if debug and not debug_printed:
            print("DEBUG status:", response.status_code)
            print("DEBUG url:", response.url)
            print("DEBUG request body:", json.dumps(body, ensure_ascii=False))
            print("DEBUG response preview:", response.text[:3000])
            debug_printed = True

        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenSearch request failed: status={response.status_code}, body={response.text[:1000]}"
            )

        payload = response.json()
        hits = extract_hits(payload)
        if not hits:
            break

        for hit in hits:
            source_raw = hit.get("_source", {})
            if not isinstance(source_raw, Mapping):
                continue

            timestamp_raw = get_field_value(source_raw, config.fields.timestamp)
            if timestamp_raw is None:
                continue

            timestamp = parse_datetime(timestamp_raw)
            hostname = stringify_field(get_field_value(source_raw, config.fields.hostname), "unknown")
            message = stringify_field(get_field_value(source_raw, config.fields.message), "")
            level = stringify_field(get_field_value(source_raw, config.fields.level), "")

            if config.filters.error_levels and level and level not in config.filters.error_levels:
                continue

            events.append(
                ErrorEvent(
                    timestamp=timestamp,
                    hostname=hostname,
                    message=message,
                    normalized_message=normalize_message(
                        message,
                        config.aggregation.max_error_type_length,
                    ),
                )
            )

        search_after = hit_sort_values(hits[-1])
        if len(hits) < current_size or search_after is None:
            break

    return FetchResult(events=events, truncated=len(events) >= config.aggregation.max_documents)


def aggregate_events(events: Sequence[ErrorEvent], interval_seconds: int, samples_limit: int) -> list[IntervalBucket]:
    buckets: dict[datetime, IntervalBucket] = {}
    for event in events:
        interval_start = floor_datetime(event.timestamp, interval_seconds)
        interval_end = interval_start + timedelta(seconds=interval_seconds)
        interval_bucket = buckets.get(interval_start)
        if interval_bucket is None:
            interval_bucket = IntervalBucket(
                start=interval_start,
                end=interval_end,
                total_errors=0,
                hosts={},
            )
            buckets[interval_start] = interval_bucket

        host_bucket = interval_bucket.hosts.get(event.hostname)
        if host_bucket is None:
            host_bucket = HostBucket(total_errors=0, error_types={})
            interval_bucket.hosts[event.hostname] = host_bucket

        error_bucket = host_bucket.error_types.get(event.normalized_message)
        if error_bucket is None:
            error_bucket = ErrorTypeBucket(count=0, samples=[])
            host_bucket.error_types[event.normalized_message] = error_bucket

        interval_bucket.total_errors += 1
        host_bucket.total_errors += 1
        error_bucket.count += 1
        if len(error_bucket.samples) < samples_limit and event.message not in error_bucket.samples:
            error_bucket.samples.append(event.message)

    return [buckets[key] for key in sorted(buckets)]


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def render_markdown(
    config: AppConfig,
    start: datetime,
    end: datetime,
    buckets: Sequence[IntervalBucket],
    events_count: int,
    truncated: bool,
) -> str:
    lines: list[str] = [
        "# OpenSearch Error Logs Report",
        "",
        f"- Period: `{to_opensearch_datetime(start)}` - `{to_opensearch_datetime(end)}`",
        f"- Interval: `{config.aggregation.resample_interval}`",
        f"- Error levels: `{', '.join(config.filters.error_levels)}`",
        f"- Hostname field: `{config.fields.hostname}`",
        f"- Message field: `{config.fields.message}`",
        f"- Fetched ERROR documents: `{events_count}`",
    ]
    if config.filters.hostnames:
        lines.append(f"- Hostname filter: `{', '.join(config.filters.hostnames)}`")
    if truncated:
        lines.append(f"- Warning: result was truncated by `max_documents={config.aggregation.max_documents}`")

    lines.extend(["", "## Summary", ""])
    if not buckets:
        lines.append("No ERROR logs were found for the selected period.")
        return "\n".join(lines) + "\n"

    lines.extend(["| Interval | Total Errors | Affected Services |", "|---|---:|---:|"])
    for bucket in buckets:
        lines.append(
            f"| `{to_opensearch_datetime(bucket.start)} - {to_opensearch_datetime(bucket.end)}` "
            f"| {bucket.total_errors} | {len(bucket.hosts)} |"
        )

    lines.extend(["", "## Details", ""])
    for bucket in buckets:
        lines.extend(
            [
                f"### {to_opensearch_datetime(bucket.start)} - {to_opensearch_datetime(bucket.end)}",
                "",
                f"Total errors: **{bucket.total_errors}**",
                "",
            ]
        )
        hosts = sorted(bucket.hosts.items(), key=lambda item: item[1].total_errors, reverse=True)
        for hostname, host_bucket in hosts:
            lines.extend([f"#### `{markdown_escape(hostname)}`", "", f"Errors: **{host_bucket.total_errors}**", ""])
            lines.extend(["| Count | Normalized Error Type | Samples |", "|---:|---|---|"])
            error_types = sorted(host_bucket.error_types.items(), key=lambda item: item[1].count, reverse=True)
            for error_type, error_bucket in error_types[: config.aggregation.top_error_types_per_service]:
                samples = "<br><br>".join(markdown_escape(sample) for sample in error_bucket.samples)
                lines.append(
                    f"| {error_bucket.count} | `{markdown_escape(error_type)}` | {samples or '-'} |"
                )
            if len(error_types) > config.aggregation.top_error_types_per_service:
                hidden_count = len(error_types) - config.aggregation.top_error_types_per_service
                lines.append(f"| - | Hidden by top limit | {hidden_count} more error types |")
            lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate OpenSearch ERROR logs and write a Markdown report.")
    parser.add_argument(
        "--config",
        default="opensearch/opensearch_logs_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument("--start", default="", help="UTC ISO datetime, for example 2026-05-18T10:00:00Z.")
    parser.add_argument("--end", default="", help="UTC ISO datetime, for example 2026-05-18T11:00:00Z.")
    parser.add_argument("--last-minutes", type=int, default=0, help="Used when --start is omitted.")
    parser.add_argument("--output", default="", help="Markdown output path. Overrides config.report.output_path.")
    parser.add_argument("--debug", action="store_true", help="Print first request and response preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    end = parse_datetime(args.end) if args.end else datetime.now(timezone.utc)
    last_minutes = args.last_minutes if args.last_minutes > 0 else config.report.default_last_minutes
    start = parse_datetime(args.start) if args.start else end - timedelta(minutes=last_minutes)
    if start >= end:
        raise ValueError("--start must be earlier than --end")

    interval_seconds = parse_resample_interval_seconds(config.aggregation.resample_interval)
    fetch_result = fetch_error_events(config, start, end, debug=args.debug)
    buckets = aggregate_events(
        fetch_result.events,
        interval_seconds=interval_seconds,
        samples_limit=config.aggregation.samples_per_error_type,
    )
    markdown = render_markdown(
        config=config,
        start=start,
        end=end,
        buckets=buckets,
        events_count=len(fetch_result.events),
        truncated=fetch_result.truncated,
    )

    output_path = Path(args.output or config.report.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Saved Markdown report to {output_path}")


if __name__ == "__main__":
    main()