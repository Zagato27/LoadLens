# SPDX-License-Identifier: Apache-2.0
"""Semantic roles of pack queries: what a series means, its unit and which SLA field it can fill.

A role is the stable meaning of a query (``lt.rps_total``, ``runtime.cpu``); the label is how the
series is stored. SLA, the forecast and the cross-domain context find series by label, so a pack
writes the labels of its role queries into the SLA fields listed here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, Optional

from AI.i18n import t

Unit = Literal[
    "fraction",          # 0..1
    "percent",           # 0..100
    "ratio",             # dimensionless, may exceed 1
    "seconds",
    "ms",
    "bytes",
    "bytes_per_second",
    "count",
    "per_second",
]
UNITS: tuple[str, ...] = (
    "fraction", "percent", "ratio", "seconds", "ms", "bytes", "bytes_per_second", "count", "per_second",
)


class LocalizedTitles(Mapping[str, str]):
    """Read-only ``{key: title}`` in a fixed key order; titles come from the dictionary in the current locale.

    Keeps the dict interface (``get``, iteration, ``dict(...)``) for callers that rendered Russian constants.
    """

    def __init__(self, keys: Iterable[str], title: Callable[[str], str]) -> None:
        self._keys = tuple(keys)
        self._title = title

    def __getitem__(self, key: str) -> str:
        if key not in self._keys:
            raise KeyError(key)
        return self._title(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys


def unit_title(unit: str) -> str:
    return t("packs.unit." + unit)


UNIT_TITLES: Mapping[str, str] = LocalizedTitles(UNITS, unit_title)


@dataclass(frozen=True)
class Role:
    name: str
    domain: str
    unit: str
    single_series: bool = False
    sla_field: Optional[str] = None

    @property
    def title(self) -> str:
        """What the series means, in the current locale (``packs.role.<name>``)."""
        return t("packs.role." + self.name)


def _roles(*items: Role) -> dict[str, Role]:
    return {item.name: item for item in items}


LT, JVM, MS, DB, KAFKA, HR = "lt_framework", "jvm", "microservices", "database", "kafka", "hard_resources"
GEN = "lt_generators"

ROLES: dict[str, Role] = _roles(
    # ---- load tool (lt_framework); latencies are milliseconds, the unit of the SLA thresholds ----
    Role("lt.rps_total", LT, "per_second", True, "max_performance_query"),
    Role("lt.rps_by_request", LT, "per_second"),
    Role("lt.p95_total", LT, "ms", True, "p95_query"),
    Role("lt.p99_total", LT, "ms", True, "p99_query"),
    Role("lt.p95_by_request", LT, "ms"),
    Role("lt.mean_latency_total", LT, "ms", True, "mean_latency_query"),
    Role("lt.error_rate_pct", LT, "percent", True, "error_rate_query"),
    Role("lt.errors_by_request", LT, "per_second"),
    Role("lt.checks_rate", LT, "per_second"),
    Role("lt.checks_pct", LT, "percent"),
    Role("lt.vus", LT, "count", True, "vus_query"),
    # Live charts of the Run test page only: the report does not read it (see LIVE_ONLY_ROLES).
    Role("lt.rps_by_instance", LT, "per_second"),
    # ---- application runtime (domain jvm, also Go and Python) ----
    Role("runtime.cpu", JVM, "fraction", sla_field="service_cpu_query"),
    Role("runtime.heap_used", JVM, "bytes"),
    Role("runtime.heap_max", JVM, "bytes"),
    Role("runtime.live_data", JVM, "bytes"),
    Role("runtime.direct_memory", JVM, "bytes"),
    Role("runtime.rss", JVM, "bytes"),
    Role("runtime.memory_limit", JVM, "bytes"),
    Role("runtime.alloc_rate", JVM, "bytes_per_second"),
    Role("runtime.gc_time_pct", JVM, "percent"),
    Role("runtime.gc_pause_avg", JVM, "seconds"),
    Role("runtime.gc_rate", JVM, "per_second"),
    Role("runtime.gc_collected_rate", JVM, "per_second"),
    Role("runtime.threads", JVM, "count"),
    Role("runtime.threads_busy", JVM, "count"),
    Role("runtime.goroutines", JVM, "count"),
    Role("runtime.open_fds", JVM, "count"),
    # ---- HTTP and gRPC servers (microservices); latencies in seconds ----
    Role("http.rps", MS, "per_second"),
    Role("http.avg_latency", MS, "seconds"),
    Role("http.p95", MS, "seconds"),
    Role("http.error_rate_pct", MS, "percent"),
    Role("http.rps_by_route", MS, "per_second"),
    Role("http.avg_latency_by_route", MS, "seconds"),
    Role("http.p95_by_route", MS, "seconds"),
    Role("http.client_avg_latency", MS, "seconds"),
    Role("grpc.rps", MS, "per_second"),
    Role("grpc.error_rate_pct", MS, "percent"),
    Role("grpc.p95", MS, "seconds"),
    # ---- databases and connection pools ----
    Role("db.tps", DB, "per_second"),
    Role("db.connections", DB, "count"),
    Role("db.connections_pct", DB, "percent"),
    Role("db.cache_hit_pct", DB, "percent"),
    Role("db.rows_read", DB, "per_second"),
    Role("db.rows_written", DB, "per_second"),
    Role("db.deadlocks", DB, "per_second"),
    Role("db.locks", DB, "count"),
    Role("db.temp_bytes", DB, "bytes_per_second"),
    Role("db.max_tx_duration", DB, "seconds"),
    Role("db.replication_lag", DB, "seconds"),
    Role("db.queries_rate", DB, "per_second"),
    Role("db.query_avg", DB, "seconds"),
    Role("db.slow_queries", DB, "per_second"),
    Role("db.threads_running", DB, "count"),
    Role("db.lock_waits", DB, "per_second"),
    Role("db.memory_used", DB, "bytes"),
    Role("db.memory_pct", DB, "percent"),
    Role("db.evictions", DB, "per_second"),
    Role("db.query_p95", DB, "seconds"),
    Role("db.queue_length", DB, "count"),
    Role("db.queue_wait", DB, "seconds"),
    Role("db.rejected", DB, "per_second"),
    Role("db.tx_aborted", DB, "per_second"),
    Role("db.cpu_pct", DB, "percent"),
    Role("db.write_stalls", DB, "per_second"),
    Role("db.pool_wait_max", DB, "seconds"),
    Role("db.pool_active", DB, "count"),
    Role("db.pool_pending", DB, "count"),
    Role("db.pool_max", DB, "count"),
    Role("db.pool_timeouts", DB, "per_second"),
    Role("db.pool_acquire_avg", DB, "seconds"),
    Role("db.pool_usage_avg", DB, "seconds"),
    # ---- Kafka ----
    Role("kafka.consumer_lag", KAFKA, "count"),
    Role("kafka.messages_in", KAFKA, "per_second"),
    Role("kafka.consumed_rate", KAFKA, "per_second"),
    Role("kafka.consumer_members", KAFKA, "count"),
    Role("kafka.bytes_in", KAFKA, "bytes_per_second"),
    Role("kafka.bytes_out", KAFKA, "bytes_per_second"),
    Role("kafka.under_replicated", KAFKA, "count"),
    Role("kafka.offline_partitions", KAFKA, "count"),
    Role("kafka.produce_p99", KAFKA, "ms"),
    Role("kafka.network_idle_pct", KAFKA, "percent"),
    Role("kafka.connections", KAFKA, "count"),
    Role("kafka.brokers", KAFKA, "count"),
    Role("kafka.client_lag_max", KAFKA, "count"),
    Role("kafka.produced_rate", KAFKA, "per_second"),
    Role("kafka.producer_errors", KAFKA, "per_second"),
    Role("kafka.producer_latency_avg", KAFKA, "ms"),
    Role("kafka.listener_avg", KAFKA, "seconds"),
    # ---- nodes and Kubernetes (hard_resources); node series are keyed by ``node`` ----
    Role("node.cpu_pct", HR, "percent", sla_field="cpu_query"),
    Role("node.mem_pct", HR, "percent", sla_field="memory_query"),
    Role("node.load_per_core", HR, "ratio"),
    Role("node.disk_util_pct", HR, "percent"),
    Role("node.net_rx", HR, "bytes_per_second"),
    Role("node.net_tx", HR, "bytes_per_second"),
    Role("node.fs_used_pct", HR, "percent"),
    Role("node.ready", HR, "count"),
    Role("node.cpu_requests_pct", HR, "percent"),
    Role("node.mem_requests_pct", HR, "percent"),
    Role("node.pods", HR, "count"),
    # ---- load generators (lt_generators): the hosts that produce the load, not the system under test ----
    Role("generator.cpu_pct", GEN, "percent"),
    Role("generator.mem_pct", GEN, "percent"),
    Role("generator.load_per_core", GEN, "ratio"),
    Role("generator.net_rx", GEN, "bytes_per_second"),
    Role("generator.net_tx", GEN, "bytes_per_second"),
    Role("generator.tcp_established", GEN, "count"),
    Role("k8s.restarts", HR, "count"),
    Role("k8s.cpu_throttled_pct", HR, "percent"),
    Role("k8s.memory_pct_of_limit", HR, "percent"),
    Role("k8s.replicas_available", HR, "count"),
    Role("k8s.replicas_desired", HR, "count"),
    Role("k8s.hpa_replicas", HR, "count"),
    Role("k8s.oom_killed", HR, "count"),
)

# SLA fields a pack may fill, in the order the UI shows them.
SLA_FIELDS: tuple[str, ...] = (
    "max_performance_query", "error_rate_query", "p95_query", "p99_query", "cpu_query", "memory_query",
    "mean_latency_query", "vus_query", "service_cpu_query",
)
# Literal SLA values a pack may set together with its role fields.
SLA_VALUE_CHOICES: dict[str, tuple[str, ...]] = {"latency_unit": ("ms", "s"), "load_model": ("open", "closed")}

# SLA references a feature needs; each must name an existing query of the role's domain.
READINESS_SLA_FIELDS: dict[str, tuple[str, ...]] = {
    "report": ("max_performance_query",),
    "forecast": ("max_performance_query", "mean_latency_query"),
    "resource_plan": ("service_cpu_query", "cpu_query"),
}


# Rows of these roles serve the live charts of a running test; reports drop them (blocks.strip_live_rows).
LIVE_ONLY_ROLES: frozenset[str] = frozenset({"lt.rps_by_instance"})


def role_for_sla_field(field: str) -> Optional[Role]:
    return next((role for role in ROLES.values() if role.sla_field == field), None)


__all__ = [
    "LocalizedTitles",
    "READINESS_SLA_FIELDS",
    "ROLES",
    "Role",
    "SLA_FIELDS",
    "SLA_VALUE_CHOICES",
    "UNITS",
    "UNIT_TITLES",
    "Unit",
    "role_for_sla_field",
    "unit_title",
]
