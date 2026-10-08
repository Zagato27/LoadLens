# Пример конфигурации (обезличено). Скопируйте в settings.py и заполните свои значения.
# Поддерживаются рантайм‑оверрайды (settings_runtime.json), в т.ч. по проектным областям через блок per_area.
# Ключевые разделы:
#  - llm — провайдер и параметры генерации
#  - default_params — шаг выборки и ресемплирование
#  - data_sources — каталог подключений (Prometheus, Grafana, InfluxDB); общий для всех областей
#  - domain_sources — какой источник читает домен; область может заменить привязки целиком
#  - logs_source — необязательный домен application_logs: агрегированные ERROR-логи из OpenSearch
#  - confluence — публикация готового веб-отчёта страницей Confluence (кнопка на странице отчёта)
#  - storage.timescale — параметры TimescaleDB (в т.ч. таблицы engineer_reports, report_jobs, confluence_publications)
#  - queries — набор доменных запросов и подписей (PromQL/Flux/InfluxQL)
#  - sla — критерии автоматической оценки результата теста
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
        # Включать ли markdown‑таблицы в отчет (для отладки)
        "include_markdown_tables_in_context": False,
        # Провайдер LLM: perplexity | openai | anthropic | gigachat
        "provider": "openai",
        # Сколько доменов анализировать параллельно и сколько кандидатов генерировать на домен
        "max_domain_workers": 5,
        "self_consistency_k": 3,
        # Троттлинг вызовов внутри одного домена (для провайдеров с жёстким rate limit):
        # последовательный режим и пауза между вызовами
        "self_consistency": {
            "max_candidates": 3,
            "parallel_candidates": True,
            "parallel_critics": True,
            "candidate_workers": 3,
            "critic_workers": 3,
            "pause_sec_between_calls": 0.0
        },
        # Предыдущий прогон того же сервиса в пределах области уходит в контекст как baseline.
        "baseline": {"enabled": True, "mode": "previous_success"},
        # Дополнительный вызов модели: вердикт только по находкам, чьи числа нашлись в метриках.
        "verification_pass": True,
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
            "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 32768, "force_json_in_prompt": True},
            "verify": True,
            "proxies": {"https": "", "http": ""},
            "connect_timeout_sec": 10,
            "request_timeout_sec": 120
        },
        # GigaChat через langchain_gigachat: mTLS‑сертификаты (cert_file/key_file) или api_key
        "gigachat": {
            "api_base_url": "https://gigachat.devices.sberbank.ru/api/v1",
            "model": "GigaChat-Max",
            "cert_file": "AI/GigaChat/client.pem",
            "key_file": "AI/GigaChat/client.key",
            "verify": False,
            "proxies": {"https": "", "http": ""},
            "connect_timeout_sec": 5,
            "request_timeout_sec": 1200,
            "max_concurrent": 4,
            # Число попыток одного вызова; между попытками экспоненциальная пауза
            "max_attempts": 7,
            "request_min_interval_sec": 0.8,
            "rate_limit_cooldown_sec": 15,
            "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 12000, "force_json_in_prompt": True}
        }
    },

    # Параметры по умолчанию для построения рядов/агрегаций
    "default_params": {
        # step — шаг выборки (гранулярность измерений, напр. 1m)
        "step": "1m",
        # resample_interval — интервал ресемплинга (напр. 10T ≈ 10 минут)
        "resample_interval": "10T"
    },

    # SLA‑критерии: главный — target_rps (достигнут на стабильной ступени → тест успешен),
    # остальные пороги вторичные (их нарушение даёт «Есть риски»). None — критерий не проверяется.
    "sla": {
        "target_rps": None,
        "max_error_rate_pct": None,
        "error_rate_query": "",
        "max_p95_ms": None,
        "p95_query": "",
        "max_p99_ms": None,
        "p99_query": "",
        "max_cpu_pct": None,
        "cpu_query": "",
        "max_memory_pct": None,
        "memory_query": "",
        # Подпись запроса lt_framework, по которой считается stable_max
        "max_performance_query": "LT (InfluxQL): RPS sum by all groups",
        "min_stable_minutes": 5.0,
        "target_rps_allow_peak_fallback": True,
        "step_detection_enabled": True,
        "step_detection_preset": "balanced",  # strict | balanced | lenient
        "debug_peak_logging": False,
        # Capacity forecast in the report («Прогноз мощностей»)
        "mean_latency_query": "LT (InfluxQL): http_req_duration mean(seconds)",
        "vus_query": "LT (InfluxQL): VUs",
        "latency_unit": "s",   # ms | s: units of mean_latency_query and p95_query
        "load_model": "open",  # open: concurrency = RPS x mean time; closed: concurrency = VUs
        # jvm query: CPU of one instance as a fraction 0..1, series by (service, instance)
        "service_cpu_query": "JVM: Process CPU usage by (application, instance)",
    },

    # Структурированный контекст тестируемой системы (справочник для анализа ИИ; снимок сохраняется с отчётом)
    "system_context": {
        "enabled": True,
        "schema_version": 1,
        "system": {"name": "", "domain": "", "description": "", "test_goal": ""},
        "architecture": {"style": "microservices", "components": [], "dependencies": [], "data_stores": []},
        "load_model": {"entrypoints": [], "critical_user_flows": [], "expected_hotspots": []},
        "operational_context": {"known_constraints": [], "known_risks": [], "normal_degradation_rules": [], "analysis_focus": []}
    },

    # Необязательный домен application_logs: агрегированные ERROR‑логи из OpenSearch Dashboards
    # (запросы идут через /api/console/proxy). При enabled=true домен добавляется к анализу ИИ.
    "logs_source": {
        "enabled": False,
        "opensearch": {
            "base_url": "https://opensearch-dashboards:5601",
            "index_pattern": "app-logs-*",
            # Логин/пароль хранятся прямо здесь (исторические имена ключей *_env)
            "username_env": "",
            "password_env": "",
            "verify_ssl": False,
            "request_timeout_sec": 300
        },
        "fields": {
            "timestamp": "@timestamp",
            "level": "level",
            "level_query": "level.keyword",
            "hostname": "hostname",
            "hostname_query": "hostname.keyword",
            "message": "message"
        },
        "filters": {"error_levels": ["ERROR"], "hostnames": []},
        "aggregation": {
            "max_documents": 300000,
            "page_size": 1000,
            "top_error_types_per_service": 20,
            "samples_per_error_type": 2,
            "max_error_type_length": 300,
            "context_top_intervals": 12,
            "context_top_services_per_interval": 3,
            "context_top_error_types_per_service": 3,
            "context_top_global_error_types": 25,
            "context_samples_per_error_type": 1,
            "context_sample_max_length": 160,
            "context_error_type_max_length": 180
        },
        "report": {"output_path": "opensearch_error_report.md", "default_last_minutes": 60}
    },

    # Публикация готового веб‑отчёта страницей Confluence (кнопка «Добавить отчёт в Confluence»).
    # Не путать с legacy‑потоком по шаблону, который использует url_basic/space_conf/user/password выше.
    "confluence": {
        "enabled": False,
        "base_url": "https://confluence.example.com",
        "username": "",
        "password": "",
        "space_key": "SPACE",
        "parent_page_id": "",
        # Parent page of forecast reports; empty — the parent page of test reports
        "forecast_parent_page_id": "",
        "verify_ssl": False,
        "request_timeout_sec": 60
    },

    # Accent of the whole install: buttons, logo and the browser tab icon. #rrggbb.
    "appearance": {
        "accent": "#6200ee"
    },

    # Подключения. В источнике только адрес, авторизация и TLS.
    # Датасорс Grafana, база InfluxQL и bucket задаются в domain_sources.
    "data_sources": {
        "grafana": {
            "title": "Grafana",
            "type": "grafana_proxy",
            "grafana": {
                "base_url": "http://grafana:3000",
                "verify_ssl": False,
                "auth": {"method": "basic", "username": "admin", "password": "admin", "token": ""}
            }
        },
        "prometheus": {
            "title": "Prometheus",
            "type": "prometheus",
            "prometheus": {"url": "http://prometheus:9090"}
        },
        "influxdb": {
            "title": "InfluxDB",
            "type": "influxdb",
            "influxdb": {"url": "http://influxdb:8086", "org": "your_org", "token": "your_token"}
        }
    },
    # default — домены без своей строки. lt_framework читает k6 через Grafana.
    # Область может заменить этот объект целиком (per_area.<область>.domain_sources).
    "domain_sources": {
        "default": {"source": "grafana", "datasource_uid": "your-datasource-uid", "datasource_name": "Prometheus"},
        "lt_framework": {
            "source": "grafana",
            "datasource_uid": "your-influxdb-uid",
            "datasource_name": "InfluxDB-k6",
            "database": "k6",
            "bucket": "your_bucket"
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
            "engineer_table": "engineer_reports",
            "llm_feedback_table": "llm_feedback",
            # Статусы фоновых задач (генерация отчётов, публикация в Confluence)
            "jobs_table": "report_jobs",
            # Привязка отчётов к опубликованным страницам Confluence
            "confluence_table": "confluence_publications",
            # Разбор предела прогноза мощностей моделью (кэш на отчёт)
            "forecast_table": "forecast_explanations"
        }
    },

    # Authentication. This section is read-only for the web UI: POST /config refuses it, so edit it
    # here. Environment variables take precedence: LOADLENS_AUTH_ENABLED, LOADLENS_SECRET_KEY,
    # LOADLENS_ADMIN_USER / LOADLENS_ADMIN_PASSWORD (first administrator), LOADLENS_COOKIE_SECURE,
    # LOADLENS_TRUSTED_PROXIES. Users live in the storage.timescale database.
    "auth": {
        "enabled": True,
        # Provider chain, tried in order. "local" keeps accounts in the database; register more with
        # loadlens_app.auth.register_provider (see loadlens_app/auth/providers.py), e.g. ["ldap", "local"].
        "providers": ["local"],
        "api_tokens": True,
        "token_default_days": 90,
        "token_max_days": 365,
        # Browser session: idle lifetime and absolute cap, in hours.
        "session_hours": 12,
        "session_max_hours": 168,
        # None: follow the request scheme. Set True behind HTTPS, and trusted_proxies to the number of
        # reverse proxies in front of the app (so X-Forwarded-For/-Proto are honoured).
        "cookie_secure": None,
        "trusted_proxies": 0,
        "password_min_length": 10,
        # Failed logins per client address and username within the window, then attempts are refused.
        "login_max_attempts": 5,
        "login_window_seconds": 300,
        # Failed logins on one account (from anywhere) before it is locked for lockout_minutes.
        "max_failed_logins": 10,
        "lockout_minutes": 15,
        # Role given to users created on first login by an external provider.
        "default_external_role": "viewer"
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
        # LT Framework. Prometheus example:
        #   sum by (scenario) (rate(lt_requests_total[1m]))
        # Flux example (placeholders {bucket} {start} {end}):
        #   from(bucket: "{bucket}") |> range(start: {start}, stop: {end})
        #   |> filter(fn: (r) => r._measurement == "k6" and r._field == "http_reqs")
        #   |> aggregateWindow(every: 1m, fn: sum)
        "lt_framework": {
            "influxql_queries": [
                'SELECT sum("value") FROM "http_reqs" WHERE $timeFilter GROUP BY time(1s), "group"::tag, "name"::tag fill(null)',
                'SELECT sum("value") FROM "checks" WHERE $timeFilter GROUP BY time(1s), "check", "group"::tag fill(none)',
                'SELECT percentile("value", 95) / 1000 FROM "http_req_duration" WHERE $timeFilter GROUP BY time(1s), "group" fill(null)',
                'SELECT sum("value") as "all" FROM "http_reqs" WHERE $timeFilter GROUP BY "all", time(1s) fill(null)',
                'SELECT mean("value") / 1000 FROM "http_req_duration" WHERE $timeFilter GROUP BY time(1s) fill(null)',
                'SELECT max("value") FROM "vus" WHERE $timeFilter GROUP BY time(1s) fill(null)'
            ],
            "label_tag_keys_list": [
                ["group", "name"],
                ["group", "check"],
                ["group", "name"],
                ["all"],
                [],
                []
            ],
            "labels": [
                "LT (InfluxQL): RPS by group & name",
                "LT (InfluxQL): checks per second by group & check",
                "LT (InfluxQL): http_req_duration p95(seconds) by group & name",
                "LT (InfluxQL): RPS sum by all groups",
                "LT (InfluxQL): http_req_duration mean(seconds)",
                "LT (InfluxQL): VUs"
            ]
        }
    }
}
