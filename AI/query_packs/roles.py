"""Semantic roles of pack queries: what a series means, its unit and which SLA field it can fill.

A role is the stable meaning of a query (``lt.rps_total``, ``runtime.cpu``); the label is how the
series is stored. SLA, the forecast and the cross-domain context find series by label, so a pack
writes the labels of its role queries into the SLA fields listed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

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
UNIT_TITLES: dict[str, str] = {
    "fraction": "доля 0..1",
    "percent": "%",
    "ratio": "отношение",
    "seconds": "с",
    "ms": "мс",
    "bytes": "байты",
    "bytes_per_second": "байт/с",
    "count": "шт.",
    "per_second": "в секунду",
}


@dataclass(frozen=True)
class Role:
    name: str
    domain: str
    unit: str
    title: str
    single_series: bool = False
    sla_field: Optional[str] = None


def _roles(*items: Role) -> dict[str, Role]:
    return {item.name: item for item in items}


LT, JVM, MS, DB, KAFKA, HR = "lt_framework", "jvm", "microservices", "database", "kafka", "hard_resources"
GEN = "lt_generators"

ROLES: dict[str, Role] = _roles(
    # ---- load tool (lt_framework); latencies are milliseconds, the unit of the SLA thresholds ----
    Role("lt.rps_total", LT, "per_second", "Общий RPS теста", True, "max_performance_query"),
    Role("lt.rps_by_request", LT, "per_second", "RPS по запросам"),
    Role("lt.p95_total", LT, "ms", "p95 времени ответа, весь тест", True, "p95_query"),
    Role("lt.p99_total", LT, "ms", "p99 времени ответа, весь тест", True, "p99_query"),
    Role("lt.p95_by_request", LT, "ms", "p95 времени ответа по запросам"),
    Role("lt.mean_latency_total", LT, "ms", "Среднее время ответа, весь тест", True, "mean_latency_query"),
    Role("lt.error_rate_pct", LT, "percent", "Доля ошибок, весь тест", True, "error_rate_query"),
    Role("lt.errors_by_request", LT, "per_second", "Ошибки в секунду по запросам"),
    Role("lt.checks_rate", LT, "per_second", "Проверки в секунду"),
    Role("lt.checks_pct", LT, "percent", "Доля успешных проверок"),
    Role("lt.vus", LT, "count", "Виртуальные пользователи", True, "vus_query"),
    # Live charts of the Run test page only: the report does not read it (see LIVE_ONLY_ROLES).
    Role("lt.rps_by_instance", LT, "per_second", "RPS по инстансам прогона LoadLens"),
    # ---- application runtime (domain jvm, also Go and Python) ----
    Role("runtime.cpu", JVM, "fraction", "CPU экземпляра сервиса", sla_field="service_cpu_query"),
    Role("runtime.heap_used", JVM, "bytes", "Занятая куча"),
    Role("runtime.heap_max", JVM, "bytes", "Максимум кучи"),
    Role("runtime.live_data", JVM, "bytes", "Живые данные после GC"),
    Role("runtime.direct_memory", JVM, "bytes", "Direct-буферы"),
    Role("runtime.rss", JVM, "bytes", "Резидентная память процесса"),
    Role("runtime.memory_limit", JVM, "bytes", "Лимит памяти контейнера"),
    Role("runtime.alloc_rate", JVM, "bytes_per_second", "Скорость аллокаций"),
    Role("runtime.gc_time_pct", JVM, "percent", "Доля времени в паузах GC"),
    Role("runtime.gc_pause_avg", JVM, "seconds", "Средняя пауза GC"),
    Role("runtime.gc_rate", JVM, "per_second", "Сборки мусора в секунду"),
    Role("runtime.gc_collected_rate", JVM, "per_second", "Объекты, собранные GC, в секунду"),
    Role("runtime.threads", JVM, "count", "Потоки"),
    Role("runtime.threads_busy", JVM, "count", "Занятые рабочие потоки"),
    Role("runtime.goroutines", JVM, "count", "Горутины"),
    Role("runtime.open_fds", JVM, "count", "Открытые файловые дескрипторы"),
    # ---- HTTP and gRPC servers (microservices); latencies in seconds ----
    Role("http.rps", MS, "per_second", "Входящие запросы в секунду"),
    Role("http.avg_latency", MS, "seconds", "Среднее время обработки"),
    Role("http.p95", MS, "seconds", "p95 времени обработки"),
    Role("http.error_rate_pct", MS, "percent", "Доля ответов 5xx"),
    Role("http.rps_by_route", MS, "per_second", "Запросы в секунду по маршрутам"),
    Role("http.avg_latency_by_route", MS, "seconds", "Среднее время по маршрутам"),
    Role("http.p95_by_route", MS, "seconds", "p95 по маршрутам"),
    Role("http.client_avg_latency", MS, "seconds", "Среднее время исходящих HTTP-вызовов"),
    Role("grpc.rps", MS, "per_second", "gRPC-вызовы в секунду"),
    Role("grpc.error_rate_pct", MS, "percent", "Доля gRPC-ошибок"),
    Role("grpc.p95", MS, "seconds", "p95 обработки gRPC"),
    # ---- databases and connection pools ----
    Role("db.tps", DB, "per_second", "Транзакции в секунду"),
    Role("db.connections", DB, "count", "Соединения"),
    Role("db.connections_pct", DB, "percent", "Соединения, % от max_connections"),
    Role("db.cache_hit_pct", DB, "percent", "Попадания в кэш буферов"),
    Role("db.rows_read", DB, "per_second", "Прочитанные строки в секунду"),
    Role("db.rows_written", DB, "per_second", "Изменённые строки в секунду"),
    Role("db.deadlocks", DB, "per_second", "Дедлоки в секунду"),
    Role("db.locks", DB, "count", "Блокировки"),
    Role("db.temp_bytes", DB, "bytes_per_second", "Запись временных файлов"),
    Role("db.max_tx_duration", DB, "seconds", "Самая долгая транзакция"),
    Role("db.replication_lag", DB, "seconds", "Отставание реплики"),
    Role("db.queries_rate", DB, "per_second", "Запросы или команды в секунду"),
    Role("db.query_avg", DB, "seconds", "Среднее время запроса или команды"),
    Role("db.slow_queries", DB, "per_second", "Медленные запросы в секунду"),
    Role("db.threads_running", DB, "count", "Выполняющиеся запросы"),
    Role("db.lock_waits", DB, "per_second", "Ожидания блокировок в секунду"),
    Role("db.memory_used", DB, "bytes", "Память СУБД"),
    Role("db.memory_pct", DB, "percent", "Память СУБД, % от лимита"),
    Role("db.evictions", DB, "per_second", "Вытеснения ключей в секунду"),
    Role("db.pool_wait_max", DB, "seconds", "Максимальное ожидание соединения"),
    Role("db.pool_active", DB, "count", "Активные соединения пула"),
    Role("db.pool_pending", DB, "count", "Потоки в ожидании соединения"),
    Role("db.pool_max", DB, "count", "Размер пула"),
    Role("db.pool_timeouts", DB, "per_second", "Таймауты получения соединения"),
    Role("db.pool_acquire_avg", DB, "seconds", "Среднее время получения соединения"),
    Role("db.pool_usage_avg", DB, "seconds", "Среднее время удержания соединения"),
    # ---- Kafka ----
    Role("kafka.consumer_lag", KAFKA, "count", "Отставание consumer group"),
    Role("kafka.messages_in", KAFKA, "per_second", "Входящие сообщения в секунду"),
    Role("kafka.consumed_rate", KAFKA, "per_second", "Прочитанные сообщения в секунду"),
    Role("kafka.consumer_members", KAFKA, "count", "Участники consumer group"),
    Role("kafka.bytes_in", KAFKA, "bytes_per_second", "Входящий трафик брокера"),
    Role("kafka.bytes_out", KAFKA, "bytes_per_second", "Исходящий трафик брокера"),
    Role("kafka.under_replicated", KAFKA, "count", "Недореплицированные партиции"),
    Role("kafka.offline_partitions", KAFKA, "count", "Партиции без лидера"),
    Role("kafka.produce_p99", KAFKA, "ms", "p99 обработки Produce на брокере"),
    Role("kafka.network_idle_pct", KAFKA, "percent", "Простой сетевых потоков брокера"),
    Role("kafka.connections", KAFKA, "count", "Соединения с брокером"),
    Role("kafka.brokers", KAFKA, "count", "Брокеры в кластере"),
    Role("kafka.client_lag_max", KAFKA, "count", "Максимальное отставание клиента"),
    Role("kafka.produced_rate", KAFKA, "per_second", "Отправленные записи в секунду"),
    Role("kafka.producer_errors", KAFKA, "per_second", "Ошибки отправки в секунду"),
    Role("kafka.producer_latency_avg", KAFKA, "ms", "Средняя задержка запроса продюсера"),
    Role("kafka.listener_avg", KAFKA, "seconds", "Среднее время обработки в listener"),
    # ---- nodes and Kubernetes (hard_resources); node series are keyed by ``node`` ----
    Role("node.cpu_pct", HR, "percent", "CPU узла", sla_field="cpu_query"),
    Role("node.mem_pct", HR, "percent", "Память узла", sla_field="memory_query"),
    Role("node.load_per_core", HR, "ratio", "load1 на ядро"),
    Role("node.disk_util_pct", HR, "percent", "Загрузка диска"),
    Role("node.net_rx", HR, "bytes_per_second", "Сеть: приём"),
    Role("node.net_tx", HR, "bytes_per_second", "Сеть: передача"),
    Role("node.fs_used_pct", HR, "percent", "Заполнение файловой системы"),
    Role("node.ready", HR, "count", "Узел в состоянии Ready"),
    Role("node.cpu_requests_pct", HR, "percent", "CPU requests, % от allocatable"),
    Role("node.mem_requests_pct", HR, "percent", "Memory requests, % от allocatable"),
    Role("node.pods", HR, "count", "Поды на узле"),
    # ---- load generators (lt_generators): the hosts that produce the load, not the system under test ----
    Role("generator.cpu_pct", GEN, "percent", "CPU генератора нагрузки"),
    Role("generator.mem_pct", GEN, "percent", "Память генератора нагрузки"),
    Role("generator.load_per_core", GEN, "ratio", "load1 на ядро генератора"),
    Role("generator.net_rx", GEN, "bytes_per_second", "Сеть генератора: приём"),
    Role("generator.net_tx", GEN, "bytes_per_second", "Сеть генератора: передача"),
    Role("generator.tcp_established", GEN, "count", "TCP-соединения генератора"),
    Role("k8s.restarts", HR, "count", "Перезапуски контейнеров"),
    Role("k8s.cpu_throttled_pct", HR, "percent", "Троттлинг CPU"),
    Role("k8s.memory_pct_of_limit", HR, "percent", "Память, % от лимита"),
    Role("k8s.replicas_available", HR, "count", "Доступные реплики"),
    Role("k8s.replicas_desired", HR, "count", "Желаемые реплики"),
    Role("k8s.hpa_replicas", HR, "count", "Реплики по HPA"),
    Role("k8s.oom_killed", HR, "count", "Контейнеры, убитые OOM"),
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
    "READINESS_SLA_FIELDS",
    "ROLES",
    "Role",
    "SLA_FIELDS",
    "SLA_VALUE_CHOICES",
    "UNITS",
    "UNIT_TITLES",
    "Unit",
    "role_for_sla_field",
]
