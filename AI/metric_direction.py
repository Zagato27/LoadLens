# SPDX-License-Identifier: Apache-2.0
"""Direction of a metric between two runs: whether a higher value is better, worse or neither.

The role of the query (``AI/query_packs/roles.py``) decides first: every role is listed in exactly one of
``HIGHER_ROLES``, ``LOWER_ROLES`` and ``NEUTRAL_ROLES`` (tests/test_metric_direction.py keeps them complete).
A row without a known role falls back to words of its label, the lists the compare page used before the server
sent the direction (static/js/compare.js keeps them for the same fallback).

The change of a metric is the assessed run A relative to the baseline B: ``(A - B) / |B|``.
"""

from __future__ import annotations

from typing import Literal, Optional

Direction = Literal["higher", "lower", "neutral"]
Assessment = Literal["better", "worse", "same", "unknown"]

HIGHER = "higher"
LOWER = "lower"
NEUTRAL = "neutral"
DIRECTIONS: tuple[str, ...] = (HIGHER, LOWER, NEUTRAL)

BETTER = "better"
WORSE = "worse"
SAME = "same"
UNKNOWN = "unknown"

# Throughput and quality the system delivers: more is better.
HIGHER_ROLES: frozenset[str] = frozenset({
    "lt.rps_total", "lt.rps_by_request", "lt.rps_by_instance", "lt.checks_rate", "lt.checks_pct",
    "http.rps", "http.rps_by_route", "grpc.rps",
    "db.tps", "db.queries_rate", "db.cache_hit_pct",
    "kafka.messages_in", "kafka.consumed_rate", "kafka.produced_rate", "kafka.network_idle_pct",
})
# Latencies, errors, saturation and resource use: less is better.
LOWER_ROLES: frozenset[str] = frozenset({
    "lt.p95_total", "lt.p99_total", "lt.p95_by_request", "lt.mean_latency_total", "lt.error_rate_pct",
    "lt.errors_by_request",
    "runtime.cpu", "runtime.heap_used", "runtime.live_data", "runtime.direct_memory", "runtime.rss",
    "runtime.alloc_rate", "runtime.gc_time_pct", "runtime.gc_pause_avg", "runtime.gc_rate", "runtime.gc_collected_rate",
    "runtime.threads", "runtime.threads_busy", "runtime.goroutines", "runtime.open_fds",
    "http.avg_latency", "http.p95", "http.error_rate_pct", "http.avg_latency_by_route", "http.p95_by_route",
    "http.client_avg_latency", "grpc.error_rate_pct", "grpc.p95",
    "db.connections_pct", "db.deadlocks", "db.locks", "db.temp_bytes", "db.max_tx_duration", "db.replication_lag",
    "db.query_avg", "db.slow_queries", "db.threads_running", "db.lock_waits", "db.memory_used", "db.memory_pct",
    "db.evictions", "db.query_p95", "db.queue_length", "db.queue_wait", "db.rejected", "db.tx_aborted", "db.cpu_pct",
    "db.write_stalls", "db.pool_wait_max", "db.pool_active", "db.pool_pending", "db.pool_timeouts",
    "db.pool_acquire_avg", "db.pool_usage_avg",
    "kafka.consumer_lag", "kafka.under_replicated", "kafka.offline_partitions", "kafka.produce_p99",
    "kafka.client_lag_max", "kafka.producer_errors", "kafka.producer_latency_avg", "kafka.listener_avg",
    "node.cpu_pct", "node.mem_pct", "node.load_per_core", "node.disk_util_pct", "node.fs_used_pct",
    "generator.cpu_pct", "generator.mem_pct", "generator.load_per_core",
    "k8s.restarts", "k8s.cpu_throttled_pct", "k8s.memory_pct_of_limit", "k8s.oom_killed",
    "host.cpu_pct", "host.mem_pct", "host.load_per_core", "host.disk_util_pct", "host.fs_used_pct",
})
# Load, limits, sizes and traffic volumes: they follow the test and the setup, not the quality of the system.
NEUTRAL_ROLES: frozenset[str] = frozenset({
    "lt.vus",
    "runtime.heap_max", "runtime.memory_limit",
    "db.connections", "db.rows_read", "db.rows_written", "db.pool_max",
    "kafka.consumer_members", "kafka.bytes_in", "kafka.bytes_out", "kafka.connections", "kafka.brokers",
    "node.net_rx", "node.net_tx", "node.ready", "node.cpu_requests_pct", "node.mem_requests_pct", "node.pods",
    "generator.net_rx", "generator.net_tx", "generator.tcp_established",
    "k8s.replicas_available", "k8s.replicas_desired", "k8s.hpa_replicas",
    "host.net_rx", "host.net_tx",
})

# Words of a label without a role; the Cyrillic stem of "error" matches Russian metric names.
HIGHER_WORDS: tuple[str, ...] = ("rps", "request count", "checks", "throughput", "fetch rate", "requests")
LOWER_WORDS: tuple[str, ...] = (
    "latency", "duration", "time", "p95", "p99", "lag", "cpu", "memory", "mem", "disk", "error", "ошиб", "gc",
    "pause", "heap", "threads",
)


def role_direction(role: Optional[str]) -> Optional[str]:
    """Direction of a known role, None for an empty or unknown one."""
    if not role:
        return None
    if role in HIGHER_ROLES:
        return HIGHER
    if role in LOWER_ROLES:
        return LOWER
    if role in NEUTRAL_ROLES:
        return NEUTRAL
    return None


def label_direction(label: Optional[str]) -> str:
    """Direction from the words of a label; "time" alone does not overrule a throughput word ("response time" stays
    lower, "requests per time unit" stays higher)."""
    text = str(label or "").lower()
    if any(word in text for word in HIGHER_WORDS) and not any(word in text and word != "time" for word in LOWER_WORDS):
        return HIGHER
    if any(word in text for word in LOWER_WORDS):
        return LOWER
    return NEUTRAL


def metric_direction(query_label: Optional[str], role: Optional[str] = None) -> str:
    """Direction of a metric: by its role when the role is known, else by the words of its label."""
    return role_direction(role) or label_direction(query_label)


def change_pct(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Change of ``value`` relative to ``baseline`` in percent; a zero baseline divides by one."""
    if value is None or baseline is None:
        return None
    denominator = abs(baseline) if abs(baseline) > 1e-9 else 1.0
    return (value - baseline) / denominator * 100.0


def assess(change: Optional[float], direction: str, threshold_pct: float) -> str:
    """``same`` under the threshold, ``better``/``worse`` by the direction, ``unknown`` without one."""
    if change is None:
        return UNKNOWN
    if abs(change) < threshold_pct:
        return SAME
    if direction == HIGHER:
        return BETTER if change > 0 else WORSE
    if direction == LOWER:
        return BETTER if change < 0 else WORSE
    return UNKNOWN


__all__ = [
    "BETTER",
    "DIRECTIONS",
    "HIGHER",
    "HIGHER_ROLES",
    "HIGHER_WORDS",
    "LOWER",
    "LOWER_ROLES",
    "LOWER_WORDS",
    "NEUTRAL",
    "NEUTRAL_ROLES",
    "SAME",
    "UNKNOWN",
    "WORSE",
    "assess",
    "change_pct",
    "label_direction",
    "metric_direction",
    "role_direction",
]
