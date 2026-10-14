# План: пакеты запросов под типовые стеки

Цель: администратор выбирает в настройках стек (например, «Spring Boot на Kubernetes + k6 в InfluxDB»),
подставляет 2–3 переменные, нажимает «Проверить» и получает рабочие запросы по всем доменам
плюс заполненные ссылки SLA (`max_performance_query`, `p95_query`, `service_cpu_query` и т. д.).
Онбординг с «недели работы инженера» сокращается до часа.

## Статус реализации (2026-10-13)

Этапы 0–1d (MVP) и фаза 2 сделаны. Отступления от плана ниже.

Фаза 2: автоопределение пакетов по метрикам источников (`loadlens_app/metric_discovery.py`, кнопка «Определить по
источникам», автозапуск в мастере), обновление установленных пакетов (`since` у запроса, кнопка «Обновить»),
ссылки SLA по ролям в пайплайне, прогнозе и готовности проекта, заголовки доменов в отчёте, сравнении и Confluence,
пакеты второй волны и стенд `docker/dev-stack` с прогоном 16 пакетов на живых экспортерах.

| Фаза 2 по плану | Сделано | Почему |
|---|---|---|
| `domain_titles` в переопределениях области | Поле `title` блока домена (наследует область и сервис) и `domain_title` пакета; заголовок сохраняется в итоговом блоке отчёта | Без нового раздела конфигурации; старые отчёты не меняют заголовки задним числом |
| `SERVICES_CPU_DOMAIN` по роли в любом домене | Домены CPU берутся из реестра ролей, ссылка — по роли `runtime.cpu`, если поле SLA пустое | Все пакеты рантайма пишут в `jvm`; поиск по всем доменам не нужен |
| Пакеты `lt-jmeter-prometheus`, `kafka-burrow` | Не сделаны | Имена метрик задаёт пользователь плагина или сторонний экспортер, надёжных значений по умолчанию нет |
| `lt-yandex-tank-influxql` | Сделан без отметки «проверен» | Имена полей InfluxUploader не проверены на живом Tank; описание просит сверить их кнопкой «Проверить» |
| — | Отметка `verified` у пакета | Отделяет 16 пакетов, проверенных на живых экспортерах, от проверенных только синтаксически |

Отступления MVP:

| План | Сделано | Почему |
|---|---|---|
| Пакеты в `loadlens_app/query_packs/*.json` | `AI/query_packs/packs/*.toml` | Пайплайн в `AI/` не импортирует `loadlens_app`; TOML-литералы не требуют экранировать кавычки PromQL и InfluxQL |
| Задержки LT в секундах | В миллисекундах, `latency_unit = "ms"` | Проверка SLA сравнивает p95 из запроса с `max_p95_ms` без пересчёта единиц |
| `roles` как {роль → подпись} | {подпись → роль}, плюс `packs` [{id, version, variables}] | Несколько рядов могут иметь одну роль; переименование строки переносит ключ |
| `unit`/`description` в блоке | Берутся из каталога по установленным пакетам | Конфиг не разрастается; смысл ряда не дублируется |
| `promql-parser` в dev-зависимостях | Свой анализатор подмножества PromQL в `tests/promql_lint.py` | Без сетевой установки; проверены все 166 PromQL-запросов пакетов |
| Шаг мастера блокирует «Далее» до роли `lt.rps_total` с данными | «Далее» доступно, когда у `lt_framework` есть запросы | Проверка источника на шаге мастера не обязательна |
| `node.cpu` через `k8s-nodes` | Узлы через `node-exporter` с hostname из `node_uname_info` (VM и Kubernetes), `k8s-nodes` — готовность и requests | Одни и те же запросы узлов работают на VM и в Kubernetes |

Каталог: 37 пакетов, 5 наборов (`docs/query-packs.md`). Пример конфигурации и демо-прогон переведены на подписи пакета
`lt-k6-influxql` (задержки в мс). 16 пакетов проверены на `docker/dev-stack` (результаты — в его README);
остальные — синтаксически и на контракте рядов и единиц: Kubernetes, JMX, OpenTelemetry, Istio, ingress-nginx,
MySQL, MongoDB, Windows, Flask, Django, HikariCP, JMeter, Gatling и Yandex.Tank нужно прогнать на стенде заказчика.

## 1. Что есть сейчас и что это ограничивает

| Факт | Где | Следствие для пакетов |
|---|---|---|
| Запросы хранятся по доменам в `CONFIG["queries"][domain]` параллельными массивами: `labels`, `promql_queries` + `label_keys_list`, `influxql_queries` / `flux_queries` + `label_tag_keys_list` | `settings.example.py`, `loadlens_app/config_schema.py::validate_queries` | Пакет рендерится в этот же формат; новые ключи блока (`roles`, `packs`) должны проходить `validate_queries` и сохраняться редактором |
| Домены фиксированы: `jvm`, `database`, `kafka`, `microservices`, `hard_resources`, `lt_framework` | `AI/pipeline.py:2730`, `static/js/settings.js:1183` | Пакеты для Go и Python кладут runtime-метрики в домен `jvm` (см. п. 3.4); новые домены не вводим |
| Язык запроса задаётся типом источника, к которому привязан домен: prometheus → PromQL, influxdb → Flux, grafana_proxy → любой | `AI/data_sources.py::domain_query_language`, `source_accepts_language` | Каталог пакетов фильтруется по совместимости с привязкой домена; несовместимый пакет не применяется |
| SLA, прогноз и детектор ступеней находят ряды по тексту подписи (`labels`) | `AI/sla_evaluator.py::_find_section_by_label`, `loadlens_app/forecast.py::_forecast_settings` | Пакет заполняет ссылки SLA своими подписями; подписи становятся стабильным публичным контрактом пакета |
| Пайплайн ищет CPU и heap JVM по подстрокам «Process CPU usage», «Heap used», «Heap max» | `AI/pipeline.py:3036-3051` | Заменить на семантические роли с откатом на подстроки (п. 5.1) |
| CPU сервиса для плана ресурсов берётся только из домена `jvm`, CPU узлов из `hard_resources` | `loadlens_app/forecast.py:75-76` (`SERVICES_CPU_DOMAIN`, `NODES_CPU_DOMAIN`) | Контейнерный CPU для Go/Python тоже кладём в `jvm`; в фазе 2 домен сделать настраиваемым |
| Соглашения по единицам: CPU узла в процентах (`NODE_CPU_SCALE = 0.01`), CPU экземпляра сервиса в долях 0..1, задержки в единице `sla.latency_unit` (одна на `mean_latency_query` и `p95_query`) | `loadlens_app/forecast.py:77`, README «Конфигурация» | Пакет обязан нормализовать единицы сам (см. п. 2.4) |
| Имя серии = `key=value\|key=value`; часть `instance=` отбрасывается, серии одного сервиса группируются по остатку | `AI/context_pack.py::service_key`, `AI/resource_plan.py` | Экземпляр сервиса обязан называться `instance`, узел не должен (иначе все узлы усреднятся в одну серию) |
| InfluxQL-макросы `$timeFilter`, `$__timeFilter`, `$__interval`; Flux-плейсхолдеры `{bucket}`, `{start}`, `{end}` | `AI/pipeline.py:893-895`, `:713` | Шаблонизатор пакетов не должен конфликтовать: используем `{{var}}` |
| Проверка одного запроса уже есть: `POST /config/query_preview` | `loadlens_app/query_preview.py` | Проверка пакета = прогон всех его запросов через `preview_metric_query` |
| Переопределения по области и сервису для `queries` и `sla` | `loadlens_app/blueprints/config_api.py::update_config`, `loadlens_app/projects.py` | Пакет применяется в той же области видимости, что открыта в разделе «Запросы» |
| Каждый эндпоинт обязан иметь правило доступа, иначе падает тест | `loadlens_app/auth/policy.py`, `tests/test_auth.py::test_every_registered_endpoint_has_an_access_rule` | Новые эндпоинты регистрируем как admin + `session_only`, как остальные `config_api.*` |
| Мастер первичной настройки: Хранилище → Источники → Привязка доменов → LLM → SLA → Готово. Шага «Запросы» нет | `static/js/settings.js:2129` | Добавить шаг «Запросы» после привязки доменов |
| Готовность области считает число запросов по доменам | `loadlens_app/projects.py::_readiness` | Дополнить: установленные пакеты и отсутствующие обязательные роли |

Предпосылка: перед началом закоммитить текущее рабочее дерево (60+ изменённых файлов,
включая `config_api.py` и `settings.js`, которые план затрагивает).

## 2. Контракт пакета

### 2.1 Файлы

Пакеты - данные, не код: `loadlens_app/query_packs/<id>.json`, индекс строится при загрузке.
Наборы (bundles) - `loadlens_app/query_packs/bundles.json`: именованные списки пакетов
(«Spring Boot на Kubernetes + k6 (InfluxDB)» = `jvm-micrometer`, `http-micrometer`, `db-hikaricp`,
`k8s-nodes`, `k8s-workload-health`, `lt-k6-influxql`).

### 2.2 Схема пакета

```json
{
  "schema_version": 1,
  "id": "jvm-micrometer",
  "version": 1,
  "title": "JVM: Micrometer (Spring Boot Actuator, Quarkus)",
  "category": "runtime",
  "stack": ["java", "spring-boot", "quarkus"],
  "domain": "jvm",
  "language": "promql",
  "description": "Память, GC, потоки и CPU процесса из Micrometer. Нужен /actuator/prometheus.",
  "requires": {"metrics": ["jvm_memory_used_bytes", "process_cpu_usage"]},
  "variables": {
    "service_label": {"title": "Лейбл с именем сервиса", "default": "application", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"},
    "matchers": {"title": "Дополнительные матчеры", "default": "application!=\"\"", "pattern": "^[^{}]*$"}
  },
  "queries": [
    {
      "role": "runtime.cpu",
      "label": "JVM: Process CPU usage by (application, instance)",
      "query": "sum by ({{service_label}}, instance) (process_cpu_usage{ {{matchers}} })",
      "label_keys": ["{{service_label}}", "instance"],
      "unit": "fraction",
      "required": true,
      "description": "CPU одного экземпляра JVM, доля 0..1"
    }
  ],
  "sla": {"service_cpu_query": "@runtime.cpu"},
  "prompt_addendum": "Метрики Micrometer: jvm_gc_pause_seconds - таймер пауз GC, ..."
}
```

Правила:
- `category`: `runtime | http | database | messaging | infrastructure | load_tool`.
- Шаблонизация только `{{var}}` (регэксп `\{\{\s*[a-z_][a-z0-9_]*\s*\}\}`); двойные фигурные скобки не встречаются
  в PromQL, InfluxQL и Flux, одинарные `{bucket}` и `$timeFilter` остаются нетронутыми. Значения переменных
  проверяются по `pattern` пакета; незакрытые `{{` после рендера - ошибка.
- `label_keys` тоже шаблонизируются (имя лейбла сервиса меняется от стека к стеку).
- `sla` пакета ссылается на роли (`@runtime.cpu`); при применении роли разрешаются в подписи и
  записываются в `sla` области/сервиса. Пустые поля SLA заполняются всегда, непустые - только с флагом `overwrite_sla`.
- `unit` обязателен: `fraction | percent | seconds | ms | bytes | count | per_second`. Для домена `lt_framework`
  все задержки приводятся к секундам, пакет ставит `sla.latency_unit = "s"`.
- Подпись (`label`) - публичный идентификатор ряда: по ней ищут SLA, сравнение прогонов и `metrics.query_label`
  в базе. Подписи пакета не меняются между версиями; новые запросы добавляются с новыми подписями.

### 2.3 Реестр ролей

Единственный источник - `loadlens_app/query_packs/roles.py`: имя роли, домен, единица, одна серия или много,
какое поле SLA заполняет. Тест проверяет, что все роли пакетов есть в реестре.

| Роль | Домен | Единица | Серий | Поле SLA |
|---|---|---|---|---|
| `lt.rps_total` | lt_framework | per_second | одна | `max_performance_query` |
| `lt.rps_by_request` | lt_framework | per_second | много | - |
| `lt.p95_total` | lt_framework | seconds | одна | `p95_query` |
| `lt.p99_total` | lt_framework | seconds | одна | `p99_query` |
| `lt.p95_by_request` | lt_framework | seconds | много | - |
| `lt.mean_latency_total` | lt_framework | seconds | одна | `mean_latency_query` |
| `lt.error_rate_pct` | lt_framework | percent | одна | `error_rate_query` |
| `lt.vus` | lt_framework | count | одна | `vus_query` |
| `lt.checks_pct` | lt_framework | percent | одна/много | - |
| `runtime.cpu` | jvm | fraction | по (service, instance) | `service_cpu_query` |
| `runtime.heap_used`, `runtime.heap_max`, `runtime.rss` | jvm | bytes | по (service, instance) | - |
| `runtime.gc_pause_avg`, `runtime.gc_rate`, `runtime.threads`, `runtime.goroutines`, `runtime.open_fds` | jvm | seconds / per_second / count | по (service, instance) | - |
| `http.rps`, `http.p95`, `http.avg_latency`, `http.error_rate_pct`, `http.rps_by_route`, `http.p95_by_route` | microservices | per_second / seconds / percent | по service | - |
| `db.tps`, `db.connections`, `db.connections_pct`, `db.cache_hit_pct`, `db.locks`, `db.deadlocks`, `db.temp_bytes`, `db.pool_active`, `db.pool_pending`, `db.pool_timeouts`, `db.pool_acquire_avg` | database | ... | - | - |
| `kafka.consumer_lag`, `kafka.messages_in`, `kafka.bytes_in`, `kafka.bytes_out`, `kafka.under_replicated`, `kafka.produce_p99`, `kafka.handler_idle_pct`, `kafka.offline_partitions`, `kafka.client_lag_max`, `kafka.producer_errors` | kafka | ... | - | - |
| `node.cpu_pct` | hard_resources | percent | по node | `cpu_query` |
| `node.mem_pct` | hard_resources | percent | по node | `memory_query` |
| `node.load_per_core`, `node.disk_util_pct`, `node.net_rx`, `node.net_tx`, `node.fs_used_pct` | hard_resources | ... | по node | - |
| `k8s.restarts`, `k8s.cpu_throttled_pct`, `k8s.replicas_available`, `k8s.replicas_desired`, `k8s.hpa_replicas`, `k8s.oom_killed` | hard_resources | ... | по workload | - |

### 2.4 Соглашения по сериям и единицам (проверяются тестами пакетов)

- Узел: ключ подписи `node` (из `instance` через `label_replace(..., "node", "$1", "instance", "(.*)")`), CPU и память в процентах.
- Экземпляр сервиса: ключи `[<service_label>, "instance"]`; для Kubernetes `instance` получается из `pod`
  через `label_replace`. CPU в долях 0..1 (для cAdvisor - доля от лимита контейнера).
- Роли с пометкой «одна серия» обязаны агрегировать без группировки: прогноз (`forecast._one_series`) отвергает
  запрос с несколькими сериями. Для PromQL-агрегата без лейблов имя серии задаётся `label_replace(sum(...), "all", "all", "", "")`
  и ключом `["all"]` (как в примере `LT (InfluxQL): RPS sum by all groups`). Поведение пустого `label_keys` проверить
  в `fetch_and_aggregate_with_label_keys` и зафиксировать тестом.
- Кардинальность: запросы «по маршруту/запросу» оборачиваются в `topk(20, ...)`.
- Окна `rate()` в пакете - переменная `window` со значением по умолчанию `2m` (не меньше двух шагов `default_params.step`).
- InfluxQL: `$timeFilter` и `GROUP BY time(1s)` как в примере; дальше ресемплирует пайплайн.

## 3. Состав пакетов

### 3.1 Фаза 1 (покрывает запрошенный список)

**Инфраструктура**

| id | Домен | Источник метрик | Ключевые запросы (роли) |
|---|---|---|---|
| `node-exporter` | hard_resources | node_exporter | `node.cpu_pct`: `label_replace(100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle", {{matchers}} }[{{window}}]))), "node", "$1", "instance", "(.*)")`; `node.mem_pct`: `100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)`; `node.load_per_core`: `node_load1 / count by (instance) (node_cpu_seconds_total{mode="idle"})`; `node.disk_util_pct`: `rate(node_disk_io_time_seconds_total[w]) * 100`; `node.net_rx/tx`: `rate(node_network_receive_bytes_total{device!~"lo\|veth.*"}[w])`; `node.fs_used_pct` |
| `k8s-nodes` | hard_resources | kube-state-metrics + node_exporter / kubelet | то же для узлов через лейбл `node`; `kube_node_status_condition{condition="Ready",status="true"}`; давление: сумма requests CPU / `kube_node_status_allocatable{resource="cpu"}` |
| `k8s-workload-health` | hard_resources | kube-state-metrics + cAdvisor | `k8s.restarts`: `increase(kube_pod_container_status_restarts_total[5m])` по workload; `k8s.cpu_throttled_pct`: `100 * rate(container_cpu_cfs_throttled_periods_total[w]) / rate(container_cpu_cfs_periods_total[w])`; `k8s.replicas_available` / `k8s.replicas_desired` (`kube_deployment_status_replicas_available`, `kube_deployment_spec_replicas`); `k8s.hpa_replicas`; `k8s.oom_killed` (`kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}`) |
| `k8s-workload-cpu` | jvm | cAdvisor + kube-state-metrics | `runtime.cpu` для любого языка: `label_replace(label_replace(sum by (container, pod) (rate(container_cpu_usage_seconds_total{container!="", container!="POD", {{matchers}} }[w])) / sum by (container, pod) (kube_pod_container_resource_limits{resource="cpu"}), "application", "$1", "container", "(.*)"), "instance", "$1", "pod", "(.*)")`; `runtime.rss`: `container_memory_working_set_bytes`; `runtime.heap_max` ≈ лимит памяти `kube_pod_container_resource_limits{resource="memory"}` |

**Java**

| id | Источник | Ключевые запросы |
|---|---|---|
| `jvm-micrometer` | Spring Boot Actuator, Quarkus, Micrometer registry | `runtime.cpu`: `process_cpu_usage` (доля, уже как надо); `runtime.heap_used`: `jvm_memory_used_bytes{area="heap"}`; `runtime.heap_max`: `sum by (...) (jvm_memory_max_bytes{area="heap"} > 0)` (у некоторых пулов max = -1); `runtime.gc_pause_avg`: `rate(jvm_gc_pause_seconds_sum[w]) / rate(jvm_gc_pause_seconds_count[w])`; `runtime.gc_rate`: `rate(jvm_gc_pause_seconds_count[w])`; `runtime.threads`: `jvm_threads_live_threads`; опционально `tomcat_threads_busy_threads`, `jvm_buffer_memory_used_bytes{id="direct"}` |
| `jvm-jmx-exporter` | Prometheus JMX exporter, client_java simpleclient, Dropwizard | `jvm_memory_bytes_used{area="heap"}`, `jvm_memory_bytes_max{area="heap"}`, `rate(jvm_gc_collection_seconds_sum[w]) / rate(..._count[w])`, `jvm_threads_current`; `runtime.cpu`: `rate(process_cpu_seconds_total[w]) / {{cpu_limit_cores}}` (ядра → доля; переменная `cpu_limit_cores`, по умолчанию 1). Лейбл сервиса по умолчанию `job` |
| `jvm-otel` | OpenTelemetry Java agent 2.x (стабильный semconv) | `jvm_memory_used_bytes{jvm_memory_type="heap"}`, `jvm_memory_limit_bytes{jvm_memory_type="heap"}`, `jvm_gc_duration_seconds_*`, `jvm_thread_count`; `runtime.cpu`: `jvm_cpu_recent_utilization_ratio` (уже доля). Вариант для агента 1.x (`process_runtime_jvm_memory_usage_bytes`, `process_runtime_jvm_cpu_utilization_ratio`) - отдельный пакет `jvm-otel-legacy` |

**Go и Python (домен `jvm`, подписи с префиксом «Go runtime:» / «Python runtime:»)**

| id | Источник | Ключевые запросы |
|---|---|---|
| `runtime-go` | client_golang (коллекторы по умолчанию) | `runtime.cpu`: `rate(process_cpu_seconds_total[w]) / {{cpu_limit_cores}}`; `runtime.heap_used`: `go_memstats_heap_alloc_bytes`; `runtime.heap_max`: `go_memstats_heap_sys_bytes`; `runtime.goroutines`: `go_goroutines`; `runtime.gc_pause_avg`: `rate(go_gc_duration_seconds_sum[w]) / rate(go_gc_duration_seconds_count[w])`; `runtime.gc_rate`; `runtime.rss`: `process_resident_memory_bytes`; `runtime.open_fds` |
| `runtime-python` | prometheus_client (коллекторы по умолчанию) | `runtime.cpu`: `rate(process_cpu_seconds_total[w]) / {{cpu_limit_cores}}`; `runtime.rss`: `process_resident_memory_bytes`; `rate(python_gc_collections_total[w])` по generation; `rate(python_gc_objects_collected_total[w])`; `runtime.open_fds`. Предупреждение в описании: в multiprocess-режиме (gunicorn) `process_*` отсутствуют, тогда брать `k8s-workload-cpu` |

**HTTP-сервисы (домен `microservices`)**

| id | Источник | Ключевые запросы |
|---|---|---|
| `http-micrometer` | `http_server_requests_seconds_*` (Spring MVC/WebFlux, Quarkus) | `http.rps`: `sum by (app) (rate(http_server_requests_seconds_count{...}[1m]))`; `http.p95`: `histogram_quantile(0.95, sum by (le, app) (rate(http_server_requests_seconds_bucket[w])))` (нужен `percentiles-histogram=true`, иначе пакет предлагает `http.avg_latency` = `_sum / _count`); `http.error_rate_pct`: `100 * rate(count{status=~"5.."}) / rate(count)`; `http.rps_by_route`, `http.p95_by_route` по `uri` с `topk(20)`; опционально исходящие `http_client_requests_seconds` |
| `http-otel` | OTel HTTP semconv ≥ 1.23: `http_server_request_duration_seconds_*`, лейблы `http_route`, `http_response_status_code`, сервис из `service_name`/`job` | те же роли; вариант `http-otel-legacy` для `http_server_duration_milliseconds_*` (ms → s) |
| `http-go` | типовые middleware client_golang: `http_request_duration_seconds_*` с `handler`/`code`; gRPC: `grpc_server_handled_total`, `grpc_server_handling_seconds_bucket` | переменные `metric_prefix`, `route_label`, `status_label`; gRPC-ошибки: `grpc_code!="OK"` |
| `http-python-fastapi` | prometheus-fastapi-instrumentator (`http_requests_total`, `http_request_duration_seconds_bucket` по `handler`) | те же роли |
| `http-python-flask` | prometheus-flask-exporter (`flask_http_request_total`, `flask_http_request_duration_seconds_bucket` по `path`) | те же роли |
| `http-python-django` | django-prometheus (`django_http_responses_total_by_status_total`, `django_http_requests_latency_seconds_by_view_method_bucket`) | те же роли |

**Kafka (домен `kafka`)**

| id | Источник | Ключевые запросы |
|---|---|---|
| `kafka-exporter` | danielqsj/kafka_exporter | `kafka.consumer_lag`: `sum by (topic, consumergroup) (kafka_consumergroup_lag{topic!~"__.+"})` (как в примере); `kafka.messages_in`: `sum by (topic) (rate(kafka_topic_partition_current_offset{topic!~"__.+"}[1m]))`; `kafka_consumergroup_members`; `kafka.under_replicated`: `sum by (topic) (kafka_topic_partition_under_replicated_partition)`; `kafka_brokers` |
| `kafka-broker-jmx` | Strimzi metrics / JMX exporter (`kafka-2_0_0.yml`) | `kafka.messages_in`: `rate(kafka_server_brokertopicmetrics_messagesin_total[1m])` по topic; `kafka.bytes_in/out`; `kafka.produce_p99`: `kafka_network_requestmetrics_totaltimems{request="Produce", quantile="0.99"}`; `kafka.under_replicated`: `kafka_server_replicamanager_underreplicatedpartitions`; `kafka.handler_idle_pct`: `100 * rate(kafka_network_requesthandlerpool_requesthandleravgidlepercent_total[w])`; `kafka.offline_partitions`; `kafka_server_socketservermetrics_connection_count` |
| `kafka-client-micrometer` | Spring Kafka / Micrometer binder | `kafka.client_lag_max`: `kafka_consumer_fetch_manager_records_lag_max`; `rate(kafka_consumer_fetch_manager_records_consumed_total[w])`; `rate(kafka_producer_record_send_total[w])`; `kafka.producer_errors`: `rate(kafka_producer_record_error_total[w])`; `kafka_producer_request_latency_avg` |

**Базы данных (домен `database`)**

| id | Источник | Ключевые запросы |
|---|---|---|
| `db-postgres-exporter` | postgres_exporter (переменная `datname`, по умолчанию `.+`) | `db.tps`: `sum by (datname) (rate(pg_stat_database_xact_commit[1m]) + rate(pg_stat_database_xact_rollback[1m]))`; `db.connections`: `sum by (datname, state) (pg_stat_activity_count)`; `db.connections_pct`: `100 * sum(pg_stat_activity_count) / max(pg_settings_max_connections)`; `db.cache_hit_pct`: `100 * rate(blks_hit) / (rate(blks_hit) + rate(blks_read))`; `rate(pg_stat_database_tup_returned/fetched/inserted/updated/deleted)`; `db.deadlocks`: `rate(pg_stat_database_deadlocks[5m])`; `db.locks`: `sum by (mode) (pg_locks_count)`; `db.temp_bytes`: `rate(pg_stat_database_temp_bytes[w])`; опционально `pg_replication_lag_seconds`, `pg_stat_statements` (нужен флаг экспортера) |
| `db-hikaricp` | Micrometer (JDBC-пул Java-сервисов) | `db.pool_active`: `hikaricp_connections_active`; `db.pool_pending`; `hikaricp_connections_max`; `db.pool_timeouts`: `rate(hikaricp_connections_timeout_total[w])`; `db.pool_acquire_avg`: `rate(hikaricp_connections_acquire_seconds_sum[w]) / rate(..._count[w])` |

**Нагрузочные инструменты (домен `lt_framework`)**

| id | Источник | Запросы и SLA |
|---|---|---|
| `lt-k6-influxql` | k6 → InfluxDB 1.x (текущий пример) | 6 запросов примера + `lt.error_rate_pct`: `SELECT mean("value") * 100 FROM "http_req_failed" WHERE $timeFilter GROUP BY time(1s) fill(null)` + `lt.p95_total` (без группировки по group). SLA: `latency_unit = "s"`, `load_model` из переменной (open для arrival-rate executors, closed для ramping-vus) |
| `lt-k6-prometheus-rw` | k6 → Prometheus remote write (`K6_PROMETHEUS_RW_TREND_STATS=p(95),p(99),avg,max`) | `lt.rps_total`: `sum(rate(k6_http_reqs_total[1m]))`; `lt.rps_by_request`: по `name`; `lt.p95_total`: `max(k6_http_req_duration_p95) / 1000`; `lt.mean_latency_total`: `avg(k6_http_req_duration_avg) / 1000`; `lt.error_rate_pct`: `100 * avg(k6_http_req_failed_rate)`; `lt.vus`: `max(k6_vus)`; `lt.checks_pct`: `100 * avg(k6_checks_rate)` |
| `lt-k6-influx2-flux` | xk6-output-influxdb → InfluxDB 2.x | Flux с `{bucket} {start} {end}`: `from(bucket: "{bucket}") \|> range(start: {start}, stop: {end}) \|> filter(fn: (r) => r._measurement == "http_reqs" and r._field == "value") \|> aggregateWindow(every: 1m, fn: sum) \|> group()`; p95 через `quantile(q: 0.95)` и `map` `/ 1000.0`; остальные роли аналогично |
| `lt-jmeter-influxql` | JMeter Backend Listener (InfluxdbBackendListenerClient), measurement `jmeter`, теги `application`, `transaction`, `statut`; поля `count`, `countError`, `avg`, `pct95.0`, `pct99.0`, `maxAT` | переменные `application`, `send_interval_sec` (по умолчанию 5). `lt.rps_total`: `SELECT sum("count") / {{send_interval_sec}} FROM "jmeter" WHERE "transaction" = 'all' AND "statut" = 'all' AND $timeFilter GROUP BY time({{send_interval_sec}}s) fill(null)`; `lt.error_rate_pct`: `sum("countError") / sum("count") * 100`; `lt.p95_total`: `mean("pct95.0") / 1000`; `lt.p95_by_request`: `GROUP BY "transaction"` без `all`/`internal`; `lt.mean_latency_total`: `mean("avg") / 1000`; `lt.vus`: `max("maxAT") WHERE "transaction" = 'internal'`. SLA: `load_model = "closed"`, `latency_unit = "s"` |
| `lt-gatling-influxql` | Gatling → Graphite-листенер InfluxDB (measurement `gatling`, теги `simulation`, `request`, `status`; поля `count`, `mean`, `percentiles95`; пользователи в `gatling.users`, поле `active`) | переменные `measurement`, `users_measurement`, `simulation`. `lt.rps_total`: `sum("count") WHERE "request" = 'allRequests' AND "status" = 'all'` по `time(1s)`; `lt.error_rate_pct`: подзапросы `SELECT sum("ko") / sum("all") * 100 FROM (SELECT "count" AS "ko" ... status = 'ko'), (SELECT "count" AS "all" ... status = 'all')`; `lt.p95_total`: `mean("percentiles95") / 1000`; `lt.mean_latency_total`: `mean("mean") / 1000`; `lt.vus`: `max("active")`. `load_model` переменной (open для `constantUsersPerSec`, closed для `constantConcurrentUsers`). В описании: зависит от graphite-шаблона в `influxdb.conf`, пример шаблона приложить |

### 3.2 Фаза 2

`http-istio` (`istio_requests_total`, `istio_request_duration_milliseconds_bucket`), `http-nginx-ingress`,
`db-mysqld-exporter`, `db-redis-exporter`, `db-pgbouncer`, `db-mongodb-exporter`, `kafka-burrow`,
`lt-locust-prometheus` (locust_exporter), `lt-yandex-tank-influxql`, `lt-jmeter-prometheus` (jmeter-prometheus-plugin,
имена метрик через переменные), `windows-exporter`.

### 3.3 Наборы (bundles)

- «Spring Boot на Kubernetes + k6 (InfluxDB)»: `jvm-micrometer`, `http-micrometer`, `db-hikaricp`, `db-postgres-exporter`, `kafka-exporter`, `k8s-nodes`, `k8s-workload-health`, `lt-k6-influxql`.
- «Spring Boot на VM + JMeter»: `jvm-micrometer`, `http-micrometer`, `db-hikaricp`, `db-postgres-exporter`, `node-exporter`, `lt-jmeter-influxql`.
- «Go на Kubernetes + k6 (Prometheus)»: `runtime-go`, `k8s-workload-cpu`, `http-go`, `db-postgres-exporter`, `k8s-nodes`, `k8s-workload-health`, `lt-k6-prometheus-rw`.
- «Python (FastAPI) на Kubernetes + k6»: `runtime-python`, `k8s-workload-cpu`, `http-python-fastapi`, `db-postgres-exporter`, `k8s-nodes`, `lt-k6-prometheus-rw`.
- «Gatling + JMX exporter»: `jvm-jmx-exporter`, `http-otel`, `node-exporter`, `lt-gatling-influxql`.

### 3.4 Почему Go и Python живут в домене `jvm`

Ключ домена `jvm` прошит в пайплайне (`domain_keys`, файлы промптов, `SERVICES_CPU_DOMAIN`), а план ресурсов
читает CPU сервиса только оттуда. Новый домен означал бы правки в пайплайне, хранилище, UI отчёта, Confluence и промптах.
Компромисс фазы 1: runtime-пакеты других языков пишут в `jvm`, подписи с префиксом языка,
а пакет несёт `prompt_addendum` с описанием метрик (goroutines, RSS вместо heap). В фазе 2 добавить
`domain_titles` в переопределения области (заголовок «Runtime (Go)» вместо «JVM» на отчёте, в сравнении и Confluence).

## 4. Бэкенд

### 4.1 Модуль `loadlens_app/query_packs/`

- `catalog.py`: загрузка и валидация JSON-пакетов при старте (pydantic-модель `QueryPack`, уже есть в зависимостях),
  уникальность `id`, роли из реестра, обязательные поля. Ошибка в файле пакета = ошибка старта приложения с именем файла.
- `render.py`: `render_pack(pack, variables) -> RenderedPack` (список `RenderedQuery(label, query, keys, role, unit, required)`),
  проверка значений переменных по `pattern`, запрет незакрытых `{{`.
- `apply.py`: `merge_into_block(block, rendered, mode)` для `mode = append | replace_domain`
  (`append` пропускает запросы с уже существующей подписью и сообщает об этом), поддерживает `roles`
  (`{role: label}`) и `packs` (`[{id, version, labels}]`) внутри блока домена; `sla_updates(rendered, sla, overwrite)`;
  сама запись - через те же функции, что `update_config` (`_update_area_section`, `_update_area_service_section`,
  глобальный `CONFIG` + runtime JSON). Общие функции записи вынести из `config_api.py` в `core.py`,
  чтобы оба пути были одинаковы.
- `check.py`: `check_pack(cfg, rendered, start, end)` прогоняет каждый запрос через `query_preview.preview_metric_query`
  (пул на 4 потока, таймаут как у preview), результат по запросу: `ok`, `series_count`, `point_count`, `message`;
  итог «обязательные запросы вернули данные».
- `detect.py` (фаза 2): список имён метрик из источника домена (`/api/v1/label/__name__/values` напрямую или через
  Grafana-прокси `/api/datasources/proxy/<id>/...`; `SHOW MEASUREMENTS` для InfluxQL; `schema.measurements()` для Flux),
  сопоставление с `requires.metrics` → рекомендуемые пакеты.
- `__main__.py`: CLI `python -m loadlens_app.query_packs list | render <id> [--var k=v] | export-example`
  (последняя команда генерирует секцию `queries` для `settings.example.py`).

### 4.2 Эндпоинты (blueprint `config_api`, все admin + `session_only`, добавить в `policy.py`)

| Метод и путь | Тело / параметры | Ответ |
|---|---|---|
| `GET /config/query_packs?area=&service=` | - | каталог с `compatible` (язык пакета против привязки домена через `resolve_domain_source` + `source_accepts_language`), `installed` (из `packs` блока), наборы |
| `POST /config/query_packs/render` | `pack_id`, `variables`, `area`, `service` | отрендеренные запросы без выполнения |
| `POST /config/query_packs/check` | то же + `start`, `end` | результат проверки по запросам |
| `POST /config/query_packs/apply` | то же + `mode`, `overwrite_sla` | новый блок домена, изменения SLA, `skipped` (дубли подписей), `previous` (для «Отменить» в UI) |
| `POST /config/query_packs/detect` (фаза 2) | `area` | рекомендуемые пакеты |

Аудит: `config.pack_apply` с `g.audit_extra = {pack, version, domain, mode, scope}`; значения запросов в журнал не пишутся.
Перед записью - `validate_queries` на итоговом блоке и `validate_sla` на итоговом SLA.

### 4.3 Изменения существующего кода

- `config_schema.validate_queries`: принимать `roles` (dict, все подписи существуют, роли из реестра) и `packs` (list).
- `static/js/settings.js`: `QUERY_PARALLEL_KEYS` не трогать (`roles`/`packs` не параллельные массивы);
  при переименовании подписи (`renameSlaLabel`) обновлять и `roles`; при удалении строки убирать её роль.
- `core._series_key_for`, `projects._query_count`: убедиться, что лишние ключи блока не ломают подсчёт (тест).

## 5. Пайплайн и анализ

### 5.1 Фаза 1 (обязательно для Go/Python/k8s-пакетов)

- `AI/pipeline.py:3036-3051`: `aux_resources` для домена microservices выбирать по `roles` блока `jvm`
  (`runtime.cpu`, `runtime.heap_used`, `runtime.heap_max`, `runtime.rss`), откат на подстроки, если ролей нет.
- `AI/context_pack.py`: в секции контекста добавить `unit` и `description` из блока запросов (если есть),
  чтобы модель понимала, что такое `pg_stat_database_blks_hit ratio`. Поля необязательные, старые конфиги без изменений.
- `AI/prompt_assembly.py`: если у области есть `prompt_addendum` от установленных пакетов, добавлять его блоком
  «Особенности метрик стека» после `analysis_guidance`. Хранение: `queries.<domain>.packs[].prompt_addendum`
  (снимок текста при установке, чтобы обновление пакета не меняло промпт втихую).

### 5.2 Фаза 2

- `loadlens_app/forecast.py`: `SERVICES_CPU_DOMAIN` искать по роли `runtime.cpu` в любом домене.
- `AI/sla_evaluator.py`, `forecast._forecast_settings`: при отсутствии подписи из SLA пробовать роль из `roles`.
- `domain_titles` в переопределениях области; отчёт, сравнение, Confluence читают заголовок оттуда.
- Обновление пакетов: сравнение `packs[].version` с каталогом, кнопка «Обновить», diff подписей; удалённые в новой
  версии запросы не трогаются (подписи - идентификаторы рядов в базе).

## 6. Интерфейс

### 6.1 Раздел «Запросы»

- Кнопка «Добавить из пакета» рядом с «Добавить запрос». Открывает панель: категории → пакеты (совместимые с привязкой
  текущего домена первыми, несовместимые серые с причиной «домен привязан к InfluxDB, пакет на PromQL»), наборы сверху.
- Карточка пакета: описание, какой экспортер нужен, переменные формой, список запросов с ролями и единицами.
- «Проверить» → таблица: запрос, ✓/✗, число серий и точек, первые имена серий, сообщение ошибки.
  Окно проверки - то же, что у preview (последние 15 минут или заданное в `previewWindowBar`).
- «Применить»: режим `append` (по умолчанию) или `replace_domain` (с подтверждением и перечнем удаляемых подписей),
  чекбокс «Заполнить ссылки SLA» (для `lt_framework`, `jvm`, `hard_resources`) и «Перезаписать заполненные».
  После применения - тост с «Отменить» (возвращает `previous` через обычный `POST /config`).
- В строке запроса бейдж «пакет jvm-micrometer v1» и роль; роль можно сменить селектом (не обязательно).

### 6.2 Мастер

Новый шаг «Запросы» после «Привязка доменов»: по одному блоку на домен с рекомендованным набором
(фаза 2: подсказка из `detect`), кнопки «Проверить» и «Применить». «Далее» доступно, когда у `lt_framework`
есть роль `lt.rps_total` с данными или шаг пропущен явно. Шаг SLA после этого видит уже заполненные ссылки.

### 6.3 Проекты и готовность

`projects._readiness`: добавить `installed_packs` и `missing_roles` (минимум `lt.rps_total`; для прогноза
`lt.mean_latency_total`, `runtime.cpu`, `node.cpu_pct`). Карточка проекта показывает «Прогноз недоступен: нет роли runtime.cpu»
со ссылкой на раздел «Запросы».

## 7. Тесты

- `tests/test_query_packs.py`: каждый файл пакета проходит схему; рендер с переменными по умолчанию не оставляет `{{`;
  результат проходит `validate_queries`; роли из реестра; единицы ролей совпадают с реестром; LT-пакеты содержат
  `lt.rps_total`, `lt.error_rate_pct`, `lt.mean_latency_total`; роли «одна серия» не содержат `by (` / `GROUP BY "` без агрегата
  (эвристика); подписи уникальны в пределах домена; `append` поверх уже установленного пакета ничего не дублирует.
- PromQL-синтаксис: dev-зависимость `promql-parser` (есть wheels) только в `requirements-dev.txt`; тест пропускается,
  если пакет не установлен. InfluxQL и Flux проверяются регулярками на макросы и баланс кавычек.
- `tests/test_smoke_api.py`: каталог и совместимость; `render`; `check` с подменой `preview_metric_query`;
  `apply` в области и у сервиса, `append`/`replace_domain`, заполнение SLA с `overwrite_sla` и без,
  `previous` в ответе, запись в аудит без текста запросов, отказ для несовместимого языка, запрет по API-токену.
- `tests/test_auth.py`: правила для новых эндпоинтов (тест покрытия упадёт сам, если забыть).
- `tests/test_ai_context.py`: `aux_resources` по ролям и откат на подстроки; `unit`/`description` в секциях контекста.
- Паритет с примером: тест, что `settings.example.py["queries"]["lt_framework"]` равен `render("lt-k6-influxql")`
  с переменными по умолчанию, а `["jvm"]` - подмножество `render("jvm-micrometer")`. Пример обновляется командой `export-example`.
- Ручная проверка на одноразовом стенде (`docker/dev-stack/`, фаза 2): Prometheus + node_exporter + postgres_exporter +
  kafka_exporter + Spring Boot petclinic + k6 с выводом в InfluxDB. Сценарий: мастер → набор «Spring Boot + k6» →
  все обязательные запросы зелёные → демо-прогон k6 → отчёт с SLA и прогнозом.

## 8. Документация

- README: раздел «Пакеты запросов» (что это, как применить, как проверить, как переопределить по сервису),
  обновить «Быстрый старт» (шаг мастера) и «Отладка» (типовые причины «✗ нет данных»: не тот лейбл сервиса,
  не включён percentiles-histogram, нет `K6_PROMETHEUS_RW_TREND_STATS`).
- `docs/query-packs.md` генерируется из каталога: по пакету - экспортер и версия, переменные, таблица запросов с ролями и единицами.
- `AI/README.md`: роли и `prompt_addendum`.

## 9. Этапы и оценка

| Этап | Содержание | Оценка |
|---|---|---|
| 0. Фундамент | схема и загрузчик пакетов, реестр ролей, рендер, `apply`/`check`, эндпоинты, политика, аудит, `validate_queries`, тесты | 4–6 дн. |
| 1a. Пакеты | 25 пакетов фазы 1 и 5 наборов, проверка каждого на реальном экспортере (petclinic, node_exporter, postgres_exporter, kafka_exporter, k6/JMeter/Gatling) | 8–10 дн. |
| 1b. UI | панель в «Запросах», шаг мастера, бейджи, «Отменить», готовность проектов | 3–4 дн. |
| 1c. Пайплайн | `aux_resources` по ролям, `unit`/`description` в контексте, `prompt_addendum` | 1–2 дн. |
| 1d. Документация и пример | README, генерация `docs/query-packs.md`, `export-example`, паритет-тест | 1 дн. |
| 2. Развитие | автоопределение, `SERVICES_CPU_DOMAIN` по роли, `domain_titles`, обновление пакетов, пакеты фазы 2, dev-stack | 6–8 дн. |

MVP (этапы 0–1d): около 4 недель одного разработчика. Порядок внутри MVP: 0 → `lt-k6-influxql` + `jvm-micrometer` +
`node-exporter` (чтобы прогнать сквозной сценарий и демо) → UI → остальные пакеты → пайплайн → документация.

## 10. Риски и решения

- **Подписи - идентификаторы рядов.** `replace_domain` ломает сравнение с прошлыми прогонами и ссылки SLA.
  Решение: по умолчанию `append`, `replace_domain` с перечнем теряемых подписей; подписи пакетов неизменны между версиями.
- **Версии экспортеров.** Имена метрик меняются (OTel semconv 2023, Micrometer `percentiles-histogram`, k6 trend stats).
  Решение: варианты как отдельные пакеты с `requires.metrics`, в фазе 2 автоопределение подсказывает нужный.
- **Единицы и модель нагрузки.** Один `latency_unit` на `mean_latency_query` и `p95_query`; `load_model` зависит от сценария.
  Решение: LT-пакеты нормализуют задержки к секундам, `load_model` - переменная пакета с пояснением.
- **CPU в долях для JMX/Go/Python.** `process_cpu_seconds_total` даёт ядра, не долю.
  Решение: переменная `cpu_limit_cores` с предупреждением в описании; на Kubernetes рекомендовать `k8s-workload-cpu`.
- **Кардинальность.** Запросы по маршрутам ограничены `topk(20)`; `limits.max_points_per_series` уже есть.
- **Агрегаты без лейблов.** Именование серии при пустом `label_keys` не задокументировано.
  Решение: проверить и зафиксировать тестом до написания LT-пакетов (п. 2.4).
- **Шаблоны в тексте запросов.** Значения переменных подставляет только администратор (текст запросов и так admin-only),
  `pattern` защищает от сломанного запроса, не от злоупотребления; это не меняет модель угроз.
- **Домен `jvm` для Go/Python.** Заголовок «JVM» на отчёте вводит в заблуждение до `domain_titles` (фаза 2);
  в фазе 1 префикс подписи и `prompt_addendum` объясняют модели и читателю, что это runtime другого языка.
