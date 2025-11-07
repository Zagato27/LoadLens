# Пример конфигурации (обезличено). Скопируйте в settings.py и заполните свои значения.
# Поддерживаются рантайм‑оверрайды (settings_runtime.json), в т.ч. по проектным областям через блок per_area.
# Ключевые разделы:
#  - llm — провайдер и параметры генерации
#  - default_params — шаг выборки и ресемплирование
#  - metrics_source — метрики тестируемой системы (SUT): Prometheus напрямую или через Grafana‑прокси
#  - lt_metrics_source — метрики инструмента нагрузки (k6/JMeter): Prometheus/InfluxDB напрямую или через Grafana‑прокси
#  - storage.timescale — параметры TimescaleDB (в т.ч. таблица engineer_reports)
#  - queries — набор доменных запросов и подписей (PromQL/Flux/InfluxQL)
#  - per_area (в settings_runtime.json) — переопределения разделов по областям (service)

CONFIG = {
    'user': 'your_login',
    'password': 'your_password',
    'grafana_login': 'admin',
    'grafana_pass': 'admin',
    'url_basic': 'https://confluence.example.com',
    'space_conf': 'SPACE',
    'grafana_base_url': 'http://grafana:3000',
    'loki_url': 'http://loki:3100/loki/api/v1/query_range',

    "llm": {
        # Включать ли markdown‑таблицы в контекст для LLM (увеличивает объём prompt)
        "include_markdown_tables_in_context": False,
        # Провайдер LLM: perplexity | openai | anthropic
        "provider": "openai",
        "perplexity": {
            "api_base_url": "https://api.perplexity.ai",
            "model": "sonar-reasoning-pro",
            "api_key": "",
            "disable_web_search": True,
            "max_concurrent": 2,
            "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 8000, "force_json_in_prompt": True},
            "verify": False,
            "proxies": {"https": "", "http": ""},
            "connect_timeout_sec": 50,
            "request_timeout_sec": 120
        },
        "openai": {
            "api_base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key": "",
            "max_concurrent": 2,
            "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 16000, "force_json_in_prompt": True},
            "verify": True,
            "proxies": {"https": "", "http": ""},
            "connect_timeout_sec": 10,
            "request_timeout_sec": 120
        },
        "anthropic": {
            "api_base_url": "https://api.anthropic.com",
            "model": "claude-3-5-sonnet",
            "api_key": "",
            "max_concurrent": 2,
            "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 120000, "force_json_in_prompt": True},
            "verify": True,
            "proxies": {"https": "", "http": ""},
            "connect_timeout_sec": 10,
            "request_timeout_sec": 120
        }
    },

    # Параметры по умолчанию для построения рядов/агрегаций
    "default_params": {
        # step — шаг выборки (гранулярность измерений, напр. 1m)
        "step": "1m",
        # resample_interval — интервал ресемплинга (напр. 10T ≈ 10 минут)
        "resample_interval": "10T"
    },

    "metrics_source": {
        # Метрики тестируемой системы (SUT)
        # Режим: "grafana_proxy" ИЛИ "prometheus"
        # Для прямого доступа к Prometheus укажите: "type": "prometheus"
        "type": "grafana_proxy",
        # Прямой Prometheus (используется при type="prometheus")
        "prometheus": {
            "url": "http://prometheus:9090"
        },
        # Grafana Proxy (используется при type="grafana_proxy")
        "grafana": {
            "base_url": "http://grafana:3000",
            "verify_ssl": False,
            "auth": {"method": "basic", "username": "admin", "password": "admin", "token": ""},
            "prometheus_datasource": {"id": None, "uid": "your-datasource-uid", "name": "Prometheus"}
        }
    },
    "lt_metrics_source": {
        # Источник метрик инструмента нагрузочного тестирования (lt_framework)
        # Возможные значения: "prometheus" | "grafana_proxy" | "influxdb"
        "type": "prometheus",
        # Прямой Prometheus для lt_framework
        "prometheus": {"url": "http://prometheus:9090"},
        # Grafana‑прокси: можно использовать и Prometheus, и InfluxDB datasource
        "grafana": {
            "base_url": "http://grafana:3000",
            "verify_ssl": False,
            "auth": {"method": "basic", "username": "admin", "password": "admin", "token": ""},
            "prometheus_datasource": {"id": None, "uid": "your-datasource-uid", "name": "Prometheus"},
            # Если используете InfluxDB через Grafana: укажите uid/name соответствующего datasource
            "influxdb_datasource": {"id": None, "uid": "your-influxdb-uid", "name": "InfluxDB-k6"}
        },
        # Прямой InfluxDB: поддерживаются Flux и InfluxQL
        "influxdb": {
            "url": "http://influxdb:8086",
            "org": "your_org",
            # bucket — для Flux запросов; database — для InfluxQL
            "bucket": "your_bucket",
            "database": "k6",
            "token": "your_token"
        }
    },

    "storage": {
        "timescale": {
            "host": "timescaledb",
            "port": 5432,
            "dbname": "loadtesting",
            "user": "app_user",
            "password": "app_password",
            "sslmode": "prefer",
            "schema": "public",
            "table": "metrics",
            "batch_size": 500,
            "make_hypertable": True,
            "ensure_extension": True,
            "chunk_interval": "1 day",
            "llm_table": "llm_reports",
            # Отдельная таблица для «Итогов от инженера»
            "engineer_table": "engineer_reports"
        }
    },

    # Ниже примерная структура запросов — адаптируйте под свои метрики.
    # Для Prometheus используйте promql_queries + label_keys_list.
    # Для InfluxDB (Flux) используйте flux_queries + label_tag_keys_list. Плейсхолдеры: {bucket}, {start}, {end} (ISO8601 UTC).
    # Для InfluxDB (InfluxQL) используйте influxql_queries + label_tag_keys_list. Плейсхолдеры: $timeFilter, $__interval, а также $Group/$Tag/$URL/$Measurement.
    "queries": {
        "jvm": {
            "promql_queries": [
                'sum(jvm_memory_used_bytes{area="heap", application!=""}) by (application, instance)',
                'sum by (application, instance) (process_cpu_usage{application!=""})'
            ],
            "label_keys_list": [["application", "instance"], ["application", "instance"]],
            "labels": [
                "JVM: Heap used (bytes) by (application, instance)",
                "JVM: Process CPU usage by (application, instance)"
            ]
        },
        "database": {
            "promql_queries": [
                'sum by (pod) (rate(db_http_requests_total{job!~".*replica.*"}[1m]))'
            ],
            "label_keys_list": [["pod"]],
            "labels": ["DB: http requests (non-replica)"]
        },
        "kafka": {
            "promql_queries": [
                'sum by (topic, consumergroup) (kafka_consumergroup_lag{topic!~"__.+"})'
            ],
            "label_keys_list": [["topic", "consumergroup"]],
            "labels": ["Kafka: consumergroup lag by topic & group"]
        },
        "microservices": {
            "promql_queries": [
                'sum by (application) (rate(http_server_requests_seconds_count{}[1m]))'
            ],
            "label_keys_list": [["application"]],
            "labels": ["Microservices: request count rate (RPS)"]
        },
        "hard_resources": {
            "promql_queries": [
                'sum(rate(container_cpu_usage_seconds_total{image!=""}[5m])) by (node)'
            ],
            "label_keys_list": [["node"]],
            "labels": ["Nodes: CPU usage by node"]
        },
        "lt_framework": {
            # LT Framework — метрики инструмента нагрузки (пример Prometheus):
            "promql_queries": [
                'sum by (scenario) (rate(lt_requests_total[1m]))',
                'histogram_quantile(0.95, sum(rate(lt_http_req_duration_seconds_bucket[5m])) by (le, scenario))',
                'sum by (scenario, status) (rate(lt_http_requests_total[1m]))'
            ],
            "label_keys_list": [
                ["scenario"],
                ["scenario"],
                ["scenario", "status"]
            ],
            "labels": [
                "LT: Requests per second by scenario",
                "LT: http_req_duration p95 by scenario",
                "LT: http requests by scenario & status"
            ],
            # Пример InfluxDB (Flux). Плейсхолдеры {bucket} {start} {end} будут подставлены автоматически.
            "flux_queries": [
                'from(bucket: "{bucket}") |> range(start: {start}, stop: {end}) |> filter(fn: (r) => r._measurement == "k6" and r._field == "http_reqs") |> aggregateWindow(every: 1m, fn: sum) |> keep(columns: ["_time","_value","scenario"])',
                'from(bucket: "{bucket}") |> range(start: {start}, stop: {end}) |> filter(fn: (r) => r._measurement == "k6" and r._field == "http_req_duration") |> aggregateWindow(every: 1m, fn: mean) |> keep(columns: ["_time","_value","scenario"])'
            ],
            "label_tag_keys_list": [
                ["scenario"],
                ["scenario"]
            ],
            # Пример InfluxDB (InfluxQL) напрямую или через Grafana Proxy — допускаются шаблонные переменные Grafana.
            "influxql_queries": [
                'SELECT sum("value") FROM "http_reqs" WHERE ("group" =~ /^$Group$/ AND "name" =~ /^$Tag$/) AND $timeFilter GROUP BY time(1s), "group"::tag, "name"::tag fill(null)',
                'SELECT sum("value") FROM "checks" WHERE ("group" =~ /^$Group$/) AND $timeFilter GROUP BY time(1s), "check", "group"::tag fill(none)',
                'SELECT percentile("value", 95) FROM /^$Measurement$/ WHERE ("name" =~ /^$URL$/ AND "group" =~ /^$Group$/ AND "name" =~ /^$Tag$/) AND $timeFilter GROUP BY time($__interval), "group", "name"::tag fill(null)'
            ],
            # Для influxql_queries укажите теги, которые попадут в подпись серии (по порядку):
            "label_tag_keys_list": [
                ["group","name"],
                ["group","check"],
                ["group","name"]
            ],
            "labels": [
                "LT (InfluxQL): RPS by group & name",
                "LT (InfluxQL): checks per second by group & check",
                "LT (InfluxQL): http_req_duration p95 by group & name"
            ]
        }
    }
}
