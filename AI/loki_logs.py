# SPDX-License-Identifier: Apache-2.0
"""Application error logs from Grafana Loki for the ``application_logs`` domain.

Two kinds of reads per run. A metric LogQL query counts the error lines of every service per aggregation
interval over the whole test window, so the totals are exact. Then log queries read a limited sample of lines for
the busiest intervals and services; the sample gives the error types and the examples. The pack has the shape of
the OpenSearch one (``AI.opensearch_logs.buckets_to_pack``) plus the sample sizes, so the domain prompt, the
report and Confluence read it unchanged.

Only LogQL that Loki 2.x already had is used: a stream selector, line filters, the ``json``, ``logfmt`` and
``regexp`` parsers, label filters, ``count_over_time`` and ``sum by``. Loki 2.x and 3.x answer it alike.

The line format decides how the level is read: a JSON field, a logfmt key, or for plain text (Logback, log4j,
Python logging, with or without ANSI colours) the leftmost level word of the line, which is the one in the header.
``auto`` (the default) reads a few lines that mention an error level and takes the format most of them have.

The transport is passed in as ``fetch`` (see ``AI.loki_client``), so this module has no network code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from AI.i18n import t
from AI.opensearch_logs import APPLICATION_LOGS_DOMAIN, build_app_config, buckets_to_pack, context_limits
from opensearch_error_report import (
    ErrorTypeBucket,
    HostBucket,
    IntervalBucket,
    floor_datetime,
    get_field_value,
    markdown_escape,
    normalize_message,
    parse_resample_interval_seconds,
    render_buckets_markdown,
    stringify_field,
    to_opensearch_datetime,
)

LOGS_TYPE_OPENSEARCH = "opensearch"
LOGS_TYPE_LOKI = "loki"
FORMAT_AUTO = "auto"
FORMAT_JSON = "json"
FORMAT_LOGFMT = "logfmt"
FORMAT_TEXT = "text"
LOG_FORMATS = (FORMAT_AUTO, FORMAT_JSON, FORMAT_LOGFMT, FORMAT_TEXT)
# Lines read to pick the format in ``auto``.
DETECT_LINES = 100
DEFAULT_SERVICE_LABEL = "app"
DEFAULT_LEVEL_FIELD = "level"
DEFAULT_MESSAGE_FIELD = "message"
DEFAULT_LINES_PER_QUERY = 300
# Loki refuses more lines than max_entries_limit_per_query (5000 by default).
MAX_LINES_PER_QUERY = 5000
DEFAULT_TIMEOUT_SEC = 120
# The sample rows of a service: no service label on the stream.
UNLABELED_SERVICE = "unknown"
_LABEL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RE2_SPECIAL = re.compile(r"([\\.^$|?*+()\[\]{}])")
# Level words of text log headers; the configured error levels are added to them.
_TEXT_LEVELS = ("trace", "debug", "info", "notice", "warning", "warn", "error", "fatal", "critical", "severe", "panic")
# The label the regexp parser puts the level of a text line into (a name no log shipper uses).
TEXT_LEVEL_LABEL = "ll_level"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_LEADING_BRACKETS_RE = re.compile(r"^(?:\[[^\]\n]*\]\s*)+")
_EXCEPTION_LINE_RE = re.compile(r"^(?:[\w$]+\.)*[\w$]*(?:Exception|Error)\b")
# An exception class followed by its message; the last one of ``A: B: c.d.PSQLException: msg`` is the cause.
_EXCEPTION_CLASS_RE = re.compile(r"(?:[A-Za-z_$][\w$]*\.)*([A-Z][\w$]*(?:Exception|Error))(?=:)")

# fetch(query=..., start_ns=..., end_ns=..., step_seconds=..., limit=..., direction=...) -> Loki API payload
Fetch = Callable[..., Dict[str, Any]]


class LokiConfigError(ValueError):
    """``logs_source.loki`` cannot be turned into queries; the text is localized."""


def logs_source_type(logs_cfg: Optional[Mapping[str, Any]]) -> str:
    """``loki`` or ``opensearch``; configs written before Loki have no ``type`` and read OpenSearch."""
    value = str((logs_cfg or {}).get("type") or "").strip().lower() if isinstance(logs_cfg, Mapping) else ""
    return LOGS_TYPE_LOKI if value == LOGS_TYPE_LOKI else LOGS_TYPE_OPENSEARCH


@dataclass(frozen=True)
class LokiSettings:
    """What the queries are built from (``logs_source.loki`` with defaults applied).

    With ``line_format`` ``auto`` the filter is the text one until ``resolve_line_format`` reads the lines.
    """

    stream: str
    pipeline: str
    service_label: str
    line_format: str
    level_field: str
    message_field: str
    level_filter: str
    hostnames: tuple[str, ...]
    lines_per_query: int
    timeout_sec: int
    levels: tuple[str, ...] = ("ERROR",)
    custom_filter: bool = False


def _text(block: Mapping[str, Any], key: str, default: str = "") -> str:
    value = block.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _int(block: Mapping[str, Any], key: str, default: int) -> int:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) <= 0:
        return default
    return int(value)


def loki_settings(logs_cfg: Mapping[str, Any]) -> LokiSettings:
    """Settings of the Loki source; raises ``LokiConfigError`` with a localized text."""
    loki = logs_cfg.get("loki") if isinstance(logs_cfg.get("loki"), Mapping) else {}
    filters = logs_cfg.get("filters") if isinstance(logs_cfg.get("filters"), Mapping) else {}
    selector = _text(loki, "selector")
    if not selector:
        raise LokiConfigError(t("server.logs.loki_selector_missing"))
    stream, pipeline = split_selector(selector)
    service_label = _text(loki, "service_label", DEFAULT_SERVICE_LABEL)
    level_field = _text(loki, "level_field", DEFAULT_LEVEL_FIELD)
    for label in (service_label, level_field):
        if not _LABEL_NAME_RE.match(label):
            raise LokiConfigError(t("server.logs.loki_label_invalid", label=label))
    line_format = _text(loki, "format", FORMAT_AUTO).lower()
    if line_format not in LOG_FORMATS:
        raise LokiConfigError(t("server.logs.loki_format_invalid", value=line_format))
    levels = [str(item).strip() for item in (filters.get("error_levels") or []) if str(item).strip()] or ["ERROR"]
    custom_filter = _text(loki, "level_filter")
    level_filter = custom_filter or default_level_filter(line_format, levels, level_field)
    hostnames = tuple(str(item).strip() for item in (filters.get("hostnames") or []) if str(item).strip())
    return LokiSettings(
        stream=stream,
        pipeline=pipeline,
        service_label=service_label,
        line_format=line_format,
        level_field=level_field,
        message_field=_text(loki, "message_field", DEFAULT_MESSAGE_FIELD),
        level_filter=level_filter,
        hostnames=hostnames,
        lines_per_query=min(_int(loki, "lines_per_query", DEFAULT_LINES_PER_QUERY), MAX_LINES_PER_QUERY),
        timeout_sec=_int(loki, "request_timeout_sec", DEFAULT_TIMEOUT_SEC),
        levels=tuple(levels),
        custom_filter=bool(custom_filter),
    )


# ---- LogQL ------------------------------------------------------------------------------------------------

def logql_string(value: str) -> str:
    """A double-quoted LogQL string literal (Go escaping) holding ``value``."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def re2_escape(value: str) -> str:
    """``value`` as a literal inside an RE2 regular expression."""
    return _RE2_SPECIAL.sub(r"\\\1", str(value))


def text_level_regex(levels: Any) -> str:
    """RE2 (and Python) pattern of a level word in a text line: delimited by spaces, brackets, ``|``, ``:`` or
    colour codes, so ``[1;31mERROR[0;39m`` and ``[error]`` match while ``ErrorHandler`` does not.

    The leftmost match is the header's level: a WARN line whose stack trace says ERROR stays a WARN line.
    """
    words = {*_TEXT_LEVELS, *(re2_escape(str(level).lower()) for level in levels)}
    alternatives = "|".join(sorted(words, key=lambda word: (-len(word), word)))
    return (r"(?i)(?:^|[\s\[|(]|\x1b\[[0-9;]*m)"
            f"(?P<{TEXT_LEVEL_LABEL}>{alternatives})"
            r"(?:[\s\]|:)]|\x1b|$)")


def default_level_filter(line_format: str, levels: Any, level_field: str) -> str:
    """Pipeline that keeps the error lines: a cheap case-insensitive line filter first, then the parsed level."""
    words = "|".join(re2_escape(str(level).lower()) for level in levels)
    cheap = f"|~ {logql_string(f'(?i)({words})')}"
    level = logql_string(f"(?i)({words})")
    if line_format in (FORMAT_JSON, FORMAT_LOGFMT):
        return f"{cheap} | {line_format} | {level_field}=~{level}"
    return f"{cheap} | regexp {logql_string(text_level_regex(levels))} | {TEXT_LEVEL_LABEL}=~{level}"


def _line_kind(line: str, level_field: str, logfmt_key: re.Pattern) -> str:
    stripped = line.lstrip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping) and get_field_value(parsed, level_field) is not None:
            return FORMAT_JSON
    if logfmt_key.search(_ANSI_RE.sub("", stripped.split("\n", 1)[0])):
        return FORMAT_LOGFMT
    return FORMAT_TEXT


def detect_line_format(lines: list[str], level_field: str) -> str:
    """``json``, ``logfmt`` or ``text``: the kind most ``lines`` are; text on a tie or without lines.

    A JSON line counts as JSON only when it has the level field (a printed payload is not a log record).
    """
    logfmt_key = re.compile(rf"(?:^|\s){re.escape(level_field)}=")
    votes = {FORMAT_TEXT: 0, FORMAT_JSON: 0, FORMAT_LOGFMT: 0}
    for line in lines:
        votes[_line_kind(line, level_field, logfmt_key)] += 1
    return max(votes, key=lambda kind: votes[kind])


def detect_query(settings: LokiSettings) -> str:
    """Lines of the selector that mention an error level, whatever their format."""
    words = "|".join(re2_escape(level.lower()) for level in settings.levels)
    cheap = f"|~ {logql_string(f'(?i)({words})')}"
    return " ".join(part for part in (_stream(settings, []), settings.pipeline, cheap) if part)


def resolve_line_format(settings: LokiSettings, fetch: "Fetch", start_ns: int, end_ns: int) -> LokiSettings:
    """``auto`` replaced by the format of up to ``DETECT_LINES`` error lines of the window, with its filter.

    A filter written in the settings does not depend on the format, so nothing is read for it.
    """
    if settings.line_format != FORMAT_AUTO or settings.custom_filter:
        return settings
    payload = fetch(query=detect_query(settings), start_ns=start_ns, end_ns=end_ns, step_seconds=None,
                    limit=DETECT_LINES, direction="backward")
    line_format = detect_line_format([line for _, _, line in lines_from_streams(payload)], settings.level_field)
    return replace(settings, line_format=line_format,
                   level_filter=default_level_filter(line_format, settings.levels, settings.level_field))


def split_selector(selector: str) -> tuple[str, str]:
    """``{a="b"} |= "x"`` -> (``a="b"``, ``|= "x"``): the matchers inside the braces and the rest of the pipeline."""
    text = selector.strip()
    if not text.startswith("{"):
        raise LokiConfigError(t("server.logs.loki_selector_invalid", selector=selector))
    quote = ""
    escaped = False
    for index, char in enumerate(text[1:], start=1):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in ('"', "`"):
            quote = char
        elif char == "}":
            return text[1:index].strip().rstrip(",").strip(), text[index + 1:].strip()
    raise LokiConfigError(t("server.logs.loki_selector_invalid", selector=selector))


def _stream(settings: LokiSettings, extra: list[str]) -> str:
    """The selector with ``extra`` matchers; without them the ``filters.hostnames`` limit applies."""
    matchers = [part for part in [settings.stream, *extra] if part]
    if settings.hostnames and not extra:
        alternatives = "|".join(re2_escape(name) for name in settings.hostnames)
        matchers.append(f"{settings.service_label}=~{logql_string(alternatives)}")
    return "{" + ", ".join(matchers) + "}"


def _pipeline(settings: LokiSettings) -> str:
    return " ".join(part for part in (settings.pipeline, settings.level_filter) if part)


def count_query(settings: LokiSettings, interval_seconds: float) -> str:
    """Error lines per service in each ``interval_seconds`` window."""
    inner = " ".join(part for part in (_stream(settings, []), _pipeline(settings)) if part)
    milliseconds = max(1, round(interval_seconds * 1000))
    duration = f"{milliseconds // 1000}s" if milliseconds % 1000 == 0 else f"{milliseconds}ms"
    return f"sum by ({settings.service_label}) (count_over_time({inner} [{duration}]))"


def lines_query(settings: LokiSettings, service: str) -> str:
    """Error lines of one service (``service`` is "" for streams without the service label)."""
    matcher = f"{settings.service_label}={logql_string(service)}"
    return " ".join(part for part in (_stream(settings, [matcher]), _pipeline(settings)) if part)


def total_lines_query(settings: LokiSettings, window_seconds: int) -> str:
    """All lines the selector reads in the window, errors or not (the connection check)."""
    inner = " ".join(part for part in (_stream(settings, []), settings.pipeline) if part)
    return f"sum(count_over_time({inner} [{int(window_seconds)}s]))"


# ---- answers ----------------------------------------------------------------------------------------------

def _result(payload: Any, kind: str) -> list[dict]:
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Mapping):
        return []
    result_type = str(data.get("resultType") or "")
    if result_type and result_type != kind:
        return []
    return [item for item in (data.get("result") or []) if isinstance(item, Mapping)]


def counts_from_matrix(payload: Any, service_label: str, interval_seconds: float, *, bucket_seconds: Optional[int] = None) -> Dict[str, Dict[datetime, int]]:
    """{service: {interval start: error lines}}. A point at ``t`` counts the lines of ``(t - interval, t]``."""
    counts: Dict[str, Dict[datetime, int]] = {}
    for item in _result(payload, "matrix"):
        metric = item.get("metric") if isinstance(item.get("metric"), Mapping) else {}
        service = str(metric.get(service_label) or "")
        for point in item.get("values") or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                seconds = float(point[0])
                value = int(round(float(point[1])))
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            start = floor_datetime(datetime.fromtimestamp(seconds - interval_seconds, tz=timezone.utc), bucket_seconds or int(interval_seconds))
            # One point per interval; a coarser step from the server would leave gaps, never double counts.
            counts.setdefault(service, {})[start] = value
    return counts


def lines_from_streams(payload: Any) -> list[tuple[datetime, dict, str]]:
    """(time, stream labels, line) of a log query answer."""
    out: list[tuple[datetime, dict, str]] = []
    for item in _result(payload, "streams"):
        labels = dict(item.get("stream")) if isinstance(item.get("stream"), Mapping) else {}
        for value in item.get("values") or []:
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            try:
                stamp = datetime.fromtimestamp(int(str(value[0])) / 1e9, tz=timezone.utc)
            except (TypeError, ValueError):
                continue
            out.append((stamp, labels, str(value[1])))
    return out


def message_of(line: str, settings: LokiSettings) -> str:
    """The error text of a line: the message field of a JSON line, the whole line of a JSON line without it,
    otherwise ``text_message``."""
    stripped = line.lstrip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping):
            value = get_field_value(parsed, settings.message_field) if settings.message_field else None
            return stringify_field(value, line) if value is not None else line
    return text_message(line, settings.levels)


def text_message(line: str, levels: Any) -> str:
    """What follows the level word on the first line of a text record, without colour codes and the leading
    ``[thread]`` groups, plus the innermost exception of the exception line under it: the time and the thread would
    make every line its own error type, and wrapper exceptions would push the cause past the context length.
    A line without a level word gives its first line."""
    rows = _ANSI_RE.sub("", line).split("\n")
    head = rows[0].rstrip("\r")
    match = re.search(text_level_regex(levels), head)
    if match:
        rest = head[match.end(TEXT_LEVEL_LABEL):].lstrip(" \t:|]-")
        head = _LEADING_BRACKETS_RE.sub("", rest)
    exception = next((row.strip() for row in rows[1:] if row[:1] not in ("", " ", "\t") and _EXCEPTION_LINE_RE.match(row)), "")
    causes = list(_EXCEPTION_CLASS_RE.finditer(exception))
    if causes:
        exception = exception[causes[-1].start(1):]
    return f"{head.strip()} | {exception}" if exception else head.strip()


# ---- collection -------------------------------------------------------------------------------------------

def _ns(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000_000 + value.microsecond * 1000


def _service_name(raw: str) -> str:
    return raw or UNLABELED_SERVICE


def collect_loki_logs(
    start_ts: float,
    end_ts: float,
    logs_cfg: Dict[str, Any],
    resample_interval: str,
    fetch: Fetch,
) -> Dict[str, Any]:
    """The ``application_logs`` domain from Loki: ``{"labeled", "markdown", "pack", "ctx"}``."""
    settings = loki_settings(logs_cfg)
    app_config = build_app_config(logs_cfg, resample_interval)
    limits = context_limits(logs_cfg)
    interval = parse_resample_interval_seconds(app_config.aggregation.resample_interval)
    step = timedelta(seconds=interval)
    start = datetime.fromtimestamp(float(start_ts), tz=timezone.utc)
    end = datetime.fromtimestamp(float(end_ts), tz=timezone.utc)
    first = floor_datetime(start, interval)
    # Full buckets share one range query. The two clipped edge buckets need their own lookback durations;
    # expanding them to full buckets would count logs outside the test while presenting the totals as exact.
    full_start = first if first == start else first + step
    full_end = floor_datetime(end, interval)
    detected = settings.line_format == FORMAT_AUTO and not settings.custom_filter
    settings = resolve_line_format(settings, fetch, _ns(start), _ns(end))

    counts: Dict[str, Dict[datetime, int]] = {}

    def read_counts(point_start: datetime, point_end: datetime, duration: float) -> None:
        payload = fetch(query=count_query(settings, duration), start_ns=_ns(point_start), end_ns=_ns(point_end),
                        step_seconds=interval, limit=None, direction=None)
        for service, values in counts_from_matrix(payload, settings.service_label, duration, bucket_seconds=interval).items():
            counts.setdefault(service, {}).update(values)

    if full_start + step <= full_end:
        read_counts(full_start + step, full_end, interval)
    if start < min(full_start, end):
        edge_end = min(full_start, end)
        read_counts(edge_end, edge_end, (edge_end - start).total_seconds())
    if full_start <= full_end < end:
        read_counts(end, end, (end - full_end).total_seconds())
    raw_names = {_service_name(raw): raw for raw in counts}
    by_start: Dict[datetime, IntervalBucket] = {}
    for raw, per_interval in counts.items():
        for bucket_start, value in per_interval.items():
            if bucket_start >= end:
                continue
            bucket = by_start.setdefault(bucket_start, IntervalBucket(
                start=max(bucket_start, start), end=min(bucket_start + step, end), total_errors=0, hosts={},
            ))
            bucket.total_errors += value
            bucket.hosts[_service_name(raw)] = HostBucket(total_errors=value, error_types={})
    buckets = [by_start[key] for key in sorted(by_start)]

    # Lines are read only where the model will look: the busiest intervals and their busiest services.
    busiest = sorted(buckets, key=lambda item: item.total_errors, reverse=True)[: limits["top_intervals"]]
    sampled: Dict[tuple[datetime, str], int] = {}
    failed = 0
    for bucket in busiest:
        hosts = sorted(bucket.hosts.items(), key=lambda item: item[1].total_errors, reverse=True)
        for name, host in hosts[: limits["top_services_per_interval"]]:
            try:
                payload = fetch(
                    query=lines_query(settings, raw_names.get(name, name)),
                    start_ns=_ns(bucket.start) + 1, end_ns=_ns(bucket.end),
                    step_seconds=None, limit=settings.lines_per_query, direction="backward",
                )
            except Exception:
                failed += 1
                continue
            lines = lines_from_streams(payload)
            sampled[(bucket.start, name)] = len(lines)
            for _, _, line in lines:
                message = message_of(line, settings)
                key = normalize_message(message, app_config.aggregation.max_error_type_length)
                error_bucket = host.error_types.setdefault(key, ErrorTypeBucket(count=0, samples=[]))
                error_bucket.count += 1
                if len(error_bucket.samples) < app_config.aggregation.samples_per_error_type and message not in error_bucket.samples:
                    error_bucket.samples.append(message)

    total_sampled = sum(sampled.values())
    pack = buckets_to_pack(
        app_config=app_config, buckets=buckets, events_count=total_sampled, truncated=False, context_limits=limits,
    )
    _annotate_samples(pack, sampled, settings, failed)
    if detected:
        pack["summary"]["line_format_detected"] = True
    markdown = _markdown(settings, start, end, app_config.aggregation.resample_interval, buckets, total_sampled, failed,
                         app_config.aggregation.top_error_types_per_service, detected)
    ctx = json.dumps({"domain": APPLICATION_LOGS_DOMAIN, "time_range": {"start": start_ts, "end": end_ts}, **pack}, ensure_ascii=False)
    return {"labeled": [], "markdown": markdown, "pack": pack, "ctx": ctx}


def _annotate_samples(pack: Dict[str, Any], sampled: Dict[tuple[datetime, str], int], settings: LokiSettings, failed: int) -> None:
    """Totals are exact; error types and their counts come from the sampled lines, the pack says how many."""
    summary = pack.setdefault("summary", {})
    summary["source"] = LOGS_TYPE_LOKI
    summary["service_label"] = settings.service_label
    summary["line_format"] = settings.line_format
    summary["error_types_from_sample"] = True
    summary["sampled_lines"] = sum(sampled.values())
    summary["lines_per_query"] = settings.lines_per_query
    if failed:
        summary["sample_queries_failed"] = failed
    for interval in pack.get("intervals") or []:
        for service in interval.get("services") or []:
            key = next((k for k in sampled if to_opensearch_datetime(k[0]) == interval.get("start_time") and k[1] == service.get("hostname")), None)
            lines = sampled.get(key) if key else None
            service["sampled_lines"] = lines if lines is not None else 0
            service["sample_complete"] = lines is not None and lines >= int(service.get("total_errors") or 0)


def _markdown(
    settings: LokiSettings,
    start: datetime,
    end: datetime,
    interval: str,
    buckets: list[IntervalBucket],
    sampled_lines: int,
    failed: int,
    top_error_types: int,
    detected: bool = False,
) -> str:
    stream = _stream(settings, [])
    header = [
        f"- Period: `{to_opensearch_datetime(start)}` - `{to_opensearch_datetime(end)}`",
        f"- Interval: `{interval}`",
        f"- Selector: `{markdown_escape(' '.join(part for part in (stream, settings.pipeline) if part))}`",
        f"- Line format: `{settings.line_format}`" + (" (detected)" if detected else ""),
        f"- Error filter: `{markdown_escape(settings.level_filter)}`",
        f"- Service label: `{settings.service_label}`",
        f"- Error lines (exact): `{sum(bucket.total_errors for bucket in buckets)}`",
        f"- Sampled lines for error types: `{sampled_lines}` (up to {settings.lines_per_query} per interval and service)",
    ]
    if failed:
        header.append(f"- Warning: {failed} sample queries failed")
    return render_buckets_markdown("Loki Error Logs Report", header, buckets, top_error_types)


__all__ = [
    "LOGS_TYPE_LOKI",
    "LOGS_TYPE_OPENSEARCH",
    "LokiConfigError",
    "LokiSettings",
    "collect_loki_logs",
    "count_query",
    "counts_from_matrix",
    "default_level_filter",
    "detect_line_format",
    "detect_query",
    "lines_from_streams",
    "lines_query",
    "logql_string",
    "logs_source_type",
    "loki_settings",
    "message_of",
    "resolve_line_format",
    "split_selector",
    "text_level_regex",
    "text_message",
    "total_lines_query",
]
