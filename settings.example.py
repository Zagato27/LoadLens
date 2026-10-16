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

    # SLA‑критерии. None — критерий не проверяется. Смысл target_rps зависит от типа теста:
    # step/stress — достигнут на стабильной ступени (тест успешен даже при деградации после неё),
    # soak — удерживается весь тест (окно удержания), spike — уровень всплеска.
    # Остальные пороги для step/stress вторичные (их нарушение даёт «Есть риски»).
    "sla": {
        "target_rps": None,
        "max_error_rate_pct": None,
        "error_rate_query": "LT (InfluxQL): error rate (%)",
        "max_p95_ms": None,
        "p95_query": "LT (InfluxQL): http_req_duration p95(ms) total",
        "max_p99_ms": None,
        "p99_query": "LT (InfluxQL): http_req_duration p99(ms) total",
        "max_cpu_pct": None,
        "cpu_query": "Nodes: CPU usage (%) by node",
        "max_memory_pct": None,
        "memory_query": "Nodes: Memory usage (%) by node",
        # Подпись запроса lt_framework, по которой считается stable_max
        "max_performance_query": "LT (InfluxQL): RPS sum by all groups",
        "min_stable_minutes": 5.0,
        "target_rps_allow_peak_fallback": True,
        "step_detection_enabled": True,  # step and stress tests only; soak/spike derive their own steps
        "step_detection_preset": "balanced",  # strict | balanced | lenient
        "debug_peak_logging": False,
        # Capacity forecast in the report («Прогноз мощностей»)
        "mean_latency_query": "LT (InfluxQL): http_req_duration mean(ms)",
        "vus_query": "LT (InfluxQL): VUs",
        "latency_unit": "ms",   # ms | s: units of mean_latency_query and p95_query
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
        "request_timeout_sec": 60,
        # Address of LoadLens for the links on published pages, e.g. "https://loadlens.example.com".
        # Empty: the address the browser used, which follows the Host header of the request.
        "app_url": ""
    },

    # Limits that protect the server and the LLM budget; None switches a limit off.
    # Counters are kept per process: with several worker processes each enforces its own share.
    "limits": {
        "max_report_window_hours": 168,       # longest test window one report may cover
        "max_points_per_series": 11000,       # window / default_params.step; Prometheus refuses more
        "max_parallel_jobs": 4,               # reports and Confluence publications running at once
        "max_parallel_jobs_per_user": 2,
        "llm_reports_per_user_per_day": 20,   # reports with AI analysis
        "llm_explanations_per_user_per_day": 50,  # LLM explanations of the forecast limit
        "llm_context_imports_per_user_per_day": 20,  # system context filled from a document
        "max_parallel_load_tests": 5,         # load test runs started from LoadLens and still active
        "max_parallel_load_tests_per_user": 3,
        "max_load_test_instances": 20,        # instances (tool processes) in one run
        "max_load_test_hours": 24             # the generator stops the tool after this time anyway
    },

    # Accent of the whole install: buttons, logo and the browser tab icon. #rrggbb.
    "appearance": {
        "accent": "#6c5ce7"
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
            # Если база не отвечает: ждать подключения не дольше connect_timeout_sec (2–60 с), затем
            # retry_seconds (1–300 с) отвечать «Хранилище недоступно» без новых попыток.
            "connect_timeout_sec": 5,
            "retry_seconds": 10,
            # Пул соединений процесса: до pool_max (0–100, 0 — без пула) открытых сессий переиспользуются
            # между запросами; сверх лимита открывается обычное соединение. Сессия, простоявшая без дела
            # дольше pool_idle_sec (5–3600 с), закрывается. 12 = 8 потоков gunicorn + наблюдатель + задачи + запас.
            "pool_max": 12,
            "pool_idle_sec": 300,
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
        "default_external_role": "viewer",
        # Recovery administrator: signs in only while the storage database is down and can only fix
        # storage.timescale on /recovery (30-minute session, actions go to the application log).
        # The hash comes from `python -m loadlens_app.auth hash-password`. Environment variables
        # LOADLENS_RECOVERY_USER / LOADLENS_RECOVERY_PASSWORD_HASH take precedence;
        # LOADLENS_ADMIN_USER / LOADLENS_ADMIN_PASSWORD are accepted as well.
        "recovery_user": "",
        "recovery_password_hash": ""
    },

    # Генераторы нагрузки страницы «Запуск теста» (docs/load-tests.md): ssh, docker, kubernetes, gitlab, jenkins,
    # manual. Секреты (private_key, passphrase, password, client_key, token, trigger_token, api_token) удобнее
    # задавать в «Настройки → Подключения → Генераторы нагрузки»: там же «Проверить» закрепляет ключ хоста SSH.
    "load_generators": {
        "manual": {"title": "Вручную", "type": "manual"},
        # "gen-1": {
        #     "title": "Генератор 1", "type": "ssh",
        #     "ssh": {"host": "10.0.0.5", "port": 22, "username": "loadlens", "private_key": "", "host_key": ""},
        #     "work_dir": "/opt/loadlens/runs", "scripts_dir": "/opt/loadlens/tests", "tools": {},
        # },
        # "docker-1": {
        #     "title": "Docker", "type": "docker",
        #     "docker": {"url": "https://gen2:2376", "ca_cert": "", "client_cert": "", "client_key": ""},
        #     "network": "loadtest", "mounts": [{"source": "loadlens-scripts", "target": "/scripts", "read_only": True}],
        #     "scripts_dir": "/scripts",
        # },
        # Пайплайн команды как есть: LoadLens выбирает ветку и значения параметров пайплайна (docs/load-tests.md).
        # "gitlab-1": {
        #     "title": "GitLab CI", "type": "gitlab",
        #     "gitlab": {"url": "https://gitlab.example.ru", "project": "perf/load-tests", "ref": "", "api_token": ""},
        #     "loadlens_url": "https://loadlens.example.ru", "tags_metrics": False,
        # },
        # "jenkins-1": {
        #     "title": "Jenkins", "type": "jenkins",
        #     "jenkins": {"url": "https://jenkins.example.ru", "job": "perf/load-test", "username": "loadlens", "api_token": ""},
        #     "param_options": {"STAND": ["stage", "preprod"]},
        # },
    },
    # Отличия от встроенных шаблонов команд k6, JMeter, Locust и Gatling (loadlens_app/load_tests/tools.py)
    # и свои инструменты; пусто — встроенные шаблоны как есть.
    "load_tools": {},

    # Запросы набора «Spring Boot на Kubernetes + k6 (InfluxDB)» из каталога пакетов (AI/query_packs).
    # Блок и ссылки SLA выше генерирует `python -m AI.query_packs example`; другой стек удобнее выбрать
    # в «Настройки → Запросы → Добавить из пакета». roles и packs — служебные ключи пакетов.
    # Для Prometheus используйте promql_queries + label_keys_list.
    # Для InfluxDB (Flux) используйте flux_queries + label_tag_keys_list. Плейсхолдеры: {bucket}, {start}, {end} (ISO8601 UTC).
    # Для InfluxDB (InfluxQL) используйте influxql_queries + label_tag_keys_list. Плейсхолдеры: $timeFilter, $__interval, а также $Group/$Tag/$URL/$Measurement.
    "queries": {
        "lt_framework": {
            "labels": [
                "LT (InfluxQL): RPS by group & name",
                "LT (InfluxQL): checks per second by group & check",
                "LT (InfluxQL): http_req_duration p95(ms) by group & name",
                "LT (InfluxQL): RPS sum by all groups",
                "LT (InfluxQL): http_req_duration p95(ms) total",
                "LT (InfluxQL): http_req_duration p99(ms) total",
                "LT (InfluxQL): http_req_duration mean(ms)",
                "LT (InfluxQL): error rate (%)",
                "LT (InfluxQL): failed requests per second by group & name",
                "LT (InfluxQL): VUs",
                "LT (InfluxQL): RPS by LoadLens instance"
            ],
            "influxql_queries": [
                'SELECT sum("value") FROM "http_reqs" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time(1s), "group"::tag, "name"::tag fill(null)',
                'SELECT sum("value") FROM "checks" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time(1s), "check", "group"::tag fill(none)',
                'SELECT percentile("value", 95) FROM "http_req_duration" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time($__interval), "group"::tag, "name"::tag fill(null)',
                'SELECT sum("value") as "all" FROM "http_reqs" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY "all", time(1s) fill(null)',
                'SELECT percentile("value", 95) FROM "http_req_duration" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time($__interval) fill(null)',
                'SELECT percentile("value", 99) FROM "http_req_duration" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time($__interval) fill(null)',
                'SELECT mean("value") FROM "http_req_duration" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time($__interval) fill(null)',
                'SELECT mean("value") * 100 FROM "http_req_failed" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time($__interval) fill(null)',
                'SELECT sum("value") FROM "http_req_failed" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time(1s), "group"::tag, "name"::tag fill(null)',
                'SELECT max("value") FROM "vus" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY time(1s) fill(null)',
                'SELECT sum("value") FROM "http_reqs" WHERE "testid" =~ /^.*$/ AND $timeFilter GROUP BY "lt_instance", time(1s) fill(null)'
            ],
            "label_tag_keys_list": [
                ["group", "name"],
                ["group", "check"],
                ["group", "name"],
                ["all"],
                [],
                [],
                [],
                [],
                ["group", "name"],
                [],
                ["lt_instance"]
            ],
            "roles": {
                "LT (InfluxQL): RPS by group & name": "lt.rps_by_request",
                "LT (InfluxQL): checks per second by group & check": "lt.checks_rate",
                "LT (InfluxQL): http_req_duration p95(ms) by group & name": "lt.p95_by_request",
                "LT (InfluxQL): RPS sum by all groups": "lt.rps_total",
                "LT (InfluxQL): http_req_duration p95(ms) total": "lt.p95_total",
                "LT (InfluxQL): http_req_duration p99(ms) total": "lt.p99_total",
                "LT (InfluxQL): http_req_duration mean(ms)": "lt.mean_latency_total",
                "LT (InfluxQL): error rate (%)": "lt.error_rate_pct",
                "LT (InfluxQL): failed requests per second by group & name": "lt.errors_by_request",
                "LT (InfluxQL): VUs": "lt.vus",
                "LT (InfluxQL): RPS by LoadLens instance": "lt.rps_by_instance"
            },
            "packs": [
                {
                    "id": "lt-k6-influxql",
                    "version": 3,
                    "variables": {
                        "testid": ".*",
                        "load_model": "open"
                    }
                }
            ]
        },
        "jvm": {
            "labels": [
                "JVM: Heap used (bytes) by (application, instance)",
                "JVM: Heap max (bytes) by (application, instance)",
                "JVM: Process CPU usage by (application, instance)",
                "JVM: Live data after GC (bytes) by (application, instance)",
                "JVM: GC pause time (%) by (application, instance)",
                "JVM: GC pause avg (s) by (application, instance)",
                "JVM: GC pauses per second by (application, instance)",
                "JVM: Live threads by (application, instance)",
                "JVM: Tomcat busy threads by (application, instance)",
                "JVM: Direct buffers used (bytes) by (application, instance)"
            ],
            "promql_queries": [
                'sum by (application, instance) (jvm_memory_used_bytes{area="heap", application!="",  })',
                'sum by (application, instance) (jvm_memory_max_bytes{area="heap", application!="",  } > 0)',
                'sum by (application, instance) (process_cpu_usage{ application!="",  })',
                'sum by (application, instance) (jvm_gc_live_data_size_bytes{ application!="",  })',
                '100 * sum by (application, instance) (rate(jvm_gc_pause_seconds_sum{ application!="",  }[2m]))',
                'sum by (application, instance) (rate(jvm_gc_pause_seconds_sum{ application!="",  }[2m])) / sum by (application, instance) (rate(jvm_gc_pause_seconds_count{ application!="",  }[2m]))',
                'sum by (application, instance) (rate(jvm_gc_pause_seconds_count{ application!="",  }[2m]))',
                'sum by (application, instance) (jvm_threads_live_threads{ application!="",  })',
                'sum by (application, instance) (tomcat_threads_busy_threads{ application!="",  })',
                'sum by (application, instance) (jvm_buffer_memory_used_bytes{id="direct", application!="",  })'
            ],
            "label_keys_list": [
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"],
                ["application", "instance"]
            ],
            "roles": {
                "JVM: Heap used (bytes) by (application, instance)": "runtime.heap_used",
                "JVM: Heap max (bytes) by (application, instance)": "runtime.heap_max",
                "JVM: Process CPU usage by (application, instance)": "runtime.cpu",
                "JVM: Live data after GC (bytes) by (application, instance)": "runtime.live_data",
                "JVM: GC pause time (%) by (application, instance)": "runtime.gc_time_pct",
                "JVM: GC pause avg (s) by (application, instance)": "runtime.gc_pause_avg",
                "JVM: GC pauses per second by (application, instance)": "runtime.gc_rate",
                "JVM: Live threads by (application, instance)": "runtime.threads",
                "JVM: Tomcat busy threads by (application, instance)": "runtime.threads_busy",
                "JVM: Direct buffers used (bytes) by (application, instance)": "runtime.direct_memory"
            },
            "packs": [
                {
                    "id": "jvm-micrometer",
                    "version": 1,
                    "variables": {
                        "service_label": "application",
                        "matchers": "",
                        "window": "2m"
                    }
                }
            ]
        },
        "microservices": {
            "labels": [
                "Microservices: request count rate (RPS)",
                "Microservices: average request time (sec)",
                "Microservices: p95 request time (sec)",
                "Microservices: 5xx error rate (%)",
                "Microservices: RPS by uri (top 20)",
                "Microservices: average request time (sec) by uri (top 20)",
                "Microservices: p95 request time (sec) by uri (top 20)",
                "Microservices: outgoing HTTP average time (sec) by client"
            ],
            "promql_queries": [
                'sum by (application) (rate(http_server_requests_seconds_count{uri!~"/actuator.*", application!="",  }[1m]))',
                'sum by (application) (rate(http_server_requests_seconds_sum{uri!~"/actuator.*", application!="",  }[1m])) / sum by (application) (rate(http_server_requests_seconds_count{uri!~"/actuator.*", application!="",  }[1m]))',
                'histogram_quantile(0.95, sum by (le, application) (rate(http_server_requests_seconds_bucket{uri!~"/actuator.*", application!="",  }[1m])))',
                '100 * (sum by (application) (rate(http_server_requests_seconds_count{status=~"5..", uri!~"/actuator.*", application!="",  }[1m])) or 0 * sum by (application) (rate(http_server_requests_seconds_count{uri!~"/actuator.*", application!="",  }[1m]))) / sum by (application) (rate(http_server_requests_seconds_count{uri!~"/actuator.*", application!="",  }[1m]))',
                'topk(20, sum by (application, uri) (rate(http_server_requests_seconds_count{uri!~"/actuator.*", application!="",  }[1m])))',
                'topk(20, sum by (application, uri) (rate(http_server_requests_seconds_sum{uri!~"/actuator.*", application!="",  }[1m])) / sum by (application, uri) (rate(http_server_requests_seconds_count{uri!~"/actuator.*", application!="",  }[1m])))',
                'topk(20, histogram_quantile(0.95, sum by (le, application, uri) (rate(http_server_requests_seconds_bucket{uri!~"/actuator.*", application!="",  }[1m]))))',
                'sum by (application, client_name) (rate(http_client_requests_seconds_sum{ application!="",  }[1m])) / sum by (application, client_name) (rate(http_client_requests_seconds_count{ application!="",  }[1m]))'
            ],
            "label_keys_list": [
                ["application"],
                ["application"],
                ["application"],
                ["application"],
                ["application", "uri"],
                ["application", "uri"],
                ["application", "uri"],
                ["application", "client_name"]
            ],
            "roles": {
                "Microservices: request count rate (RPS)": "http.rps",
                "Microservices: average request time (sec)": "http.avg_latency",
                "Microservices: p95 request time (sec)": "http.p95",
                "Microservices: 5xx error rate (%)": "http.error_rate_pct",
                "Microservices: RPS by uri (top 20)": "http.rps_by_route",
                "Microservices: average request time (sec) by uri (top 20)": "http.avg_latency_by_route",
                "Microservices: p95 request time (sec) by uri (top 20)": "http.p95_by_route",
                "Microservices: outgoing HTTP average time (sec) by client": "http.client_avg_latency"
            },
            "packs": [
                {
                    "id": "http-micrometer",
                    "version": 2,
                    "variables": {
                        "service_label": "application",
                        "matchers": "",
                        "window": "1m"
                    }
                }
            ]
        },
        "database": {
            "labels": [
                "DB pool (HikariCP): active connections by (application, pool)",
                "DB pool (HikariCP): max connections by (application, pool)",
                "DB pool (HikariCP): pending threads by (application, pool)",
                "DB pool (HikariCP): timeouts per second by (application, pool)",
                "DB pool (HikariCP): acquire time avg (sec) by (application, pool)",
                "DB pool (HikariCP): usage time avg (sec) by (application, pool)",
                "DB (PostgreSQL): transactions per second by database",
                "DB (PostgreSQL): connections by database & state",
                "DB (PostgreSQL): connections used (% of max_connections) by server",
                "DB (PostgreSQL): cache hit ratio (%) by database",
                "DB (PostgreSQL): rows read per second by database",
                "DB (PostgreSQL): rows written per second by database",
                "DB (PostgreSQL): deadlocks per second by database",
                "DB (PostgreSQL): locks by mode",
                "DB (PostgreSQL): temp files written (bytes/s) by database",
                "DB (PostgreSQL): longest transaction (s) by database",
                "DB (PostgreSQL): replication lag (s) by server"
            ],
            "promql_queries": [
                'sum by (application, pool) (hikaricp_connections_active{ application!="",  })',
                'sum by (application, pool) (hikaricp_connections_max{ application!="",  })',
                'sum by (application, pool) (hikaricp_connections_pending{ application!="",  })',
                'sum by (application, pool) (rate(hikaricp_connections_timeout_total{ application!="",  }[2m]))',
                'sum by (application, pool) (rate(hikaricp_connections_acquire_seconds_sum{ application!="",  }[2m])) / sum by (application, pool) (rate(hikaricp_connections_acquire_seconds_count{ application!="",  }[2m]))',
                'sum by (application, pool) (rate(hikaricp_connections_usage_seconds_sum{ application!="",  }[2m])) / sum by (application, pool) (rate(hikaricp_connections_usage_seconds_count{ application!="",  }[2m]))',
                'sum by (datname) (rate(pg_stat_database_xact_commit{datname=~".+", datname!~"template.*",  }[2m]) + rate(pg_stat_database_xact_rollback{datname=~".+", datname!~"template.*",  }[2m]))',
                'sum by (datname, state) (pg_stat_activity_count{datname=~".+", datname!~"template.*",  })',
                'label_replace(100 * sum by (instance) (pg_stat_activity_count{  }) / max by (instance) (pg_settings_max_connections{  }), "server", "$1", "instance", "(.*)")',
                '100 * sum by (datname) (rate(pg_stat_database_blks_hit{datname=~".+", datname!~"template.*",  }[2m])) / (sum by (datname) (rate(pg_stat_database_blks_hit{datname=~".+", datname!~"template.*",  }[2m])) + sum by (datname) (rate(pg_stat_database_blks_read{datname=~".+", datname!~"template.*",  }[2m])))',
                'sum by (datname) (rate(pg_stat_database_tup_fetched{datname=~".+", datname!~"template.*",  }[2m]))',
                'sum by (datname) (rate(pg_stat_database_tup_inserted{datname=~".+", datname!~"template.*",  }[2m]) + rate(pg_stat_database_tup_updated{datname=~".+", datname!~"template.*",  }[2m]) + rate(pg_stat_database_tup_deleted{datname=~".+", datname!~"template.*",  }[2m]))',
                'sum by (datname) (rate(pg_stat_database_deadlocks{datname=~".+", datname!~"template.*",  }[5m]))',
                'sum by (mode) (pg_locks_count{datname=~".+", datname!~"template.*",  })',
                'sum by (datname) (rate(pg_stat_database_temp_bytes{datname=~".+", datname!~"template.*",  }[2m]))',
                'max by (datname) (pg_stat_activity_max_tx_duration{datname=~".+", datname!~"template.*",  })',
                'label_replace(max by (instance) (pg_replication_lag_seconds{  }), "server", "$1", "instance", "(.*)")'
            ],
            "label_keys_list": [
                ["application", "pool"],
                ["application", "pool"],
                ["application", "pool"],
                ["application", "pool"],
                ["application", "pool"],
                ["application", "pool"],
                ["datname"],
                ["datname", "state"],
                ["server"],
                ["datname"],
                ["datname"],
                ["datname"],
                ["datname"],
                ["mode"],
                ["datname"],
                ["datname"],
                ["server"]
            ],
            "roles": {
                "DB pool (HikariCP): active connections by (application, pool)": "db.pool_active",
                "DB pool (HikariCP): max connections by (application, pool)": "db.pool_max",
                "DB pool (HikariCP): pending threads by (application, pool)": "db.pool_pending",
                "DB pool (HikariCP): timeouts per second by (application, pool)": "db.pool_timeouts",
                "DB pool (HikariCP): acquire time avg (sec) by (application, pool)": "db.pool_acquire_avg",
                "DB pool (HikariCP): usage time avg (sec) by (application, pool)": "db.pool_usage_avg",
                "DB (PostgreSQL): transactions per second by database": "db.tps",
                "DB (PostgreSQL): connections by database & state": "db.connections",
                "DB (PostgreSQL): connections used (% of max_connections) by server": "db.connections_pct",
                "DB (PostgreSQL): cache hit ratio (%) by database": "db.cache_hit_pct",
                "DB (PostgreSQL): rows read per second by database": "db.rows_read",
                "DB (PostgreSQL): rows written per second by database": "db.rows_written",
                "DB (PostgreSQL): deadlocks per second by database": "db.deadlocks",
                "DB (PostgreSQL): locks by mode": "db.locks",
                "DB (PostgreSQL): temp files written (bytes/s) by database": "db.temp_bytes",
                "DB (PostgreSQL): longest transaction (s) by database": "db.max_tx_duration",
                "DB (PostgreSQL): replication lag (s) by server": "db.replication_lag"
            },
            "packs": [
                {
                    "id": "db-hikaricp",
                    "version": 1,
                    "variables": {
                        "service_label": "application",
                        "matchers": "",
                        "window": "2m"
                    }
                },
                {
                    "id": "db-postgres-exporter",
                    "version": 1,
                    "variables": {
                        "datname": ".+",
                        "matchers": "",
                        "window": "2m"
                    }
                }
            ]
        },
        "kafka": {
            "labels": [
                "Kafka: consumergroup lag by topic & group",
                "Kafka: messages in per second by topic",
                "Kafka: consumed messages per second by topic & group",
                "Kafka: consumer group members by group",
                "Kafka: under-replicated partitions by topic",
                "Kafka: brokers"
            ],
            "promql_queries": [
                'sum by (topic, consumergroup) (kafka_consumergroup_lag{topic!~"__.+",  })',
                'sum by (topic) (rate(kafka_topic_partition_current_offset{topic!~"__.+",  }[2m]))',
                'sum by (topic, consumergroup) (rate(kafka_consumergroup_current_offset{topic!~"__.+",  }[2m]))',
                "max by (consumergroup) (kafka_consumergroup_members{  })",
                'sum by (topic) (kafka_topic_partition_under_replicated_partition{topic!~"__.+",  }) > 0',
                'label_replace(max(kafka_brokers{  }), "series", "cluster", "", "")'
            ],
            "label_keys_list": [
                ["topic", "consumergroup"],
                ["topic"],
                ["topic", "consumergroup"],
                ["consumergroup"],
                ["topic"],
                ["series"]
            ],
            "roles": {
                "Kafka: consumergroup lag by topic & group": "kafka.consumer_lag",
                "Kafka: messages in per second by topic": "kafka.messages_in",
                "Kafka: consumed messages per second by topic & group": "kafka.consumed_rate",
                "Kafka: consumer group members by group": "kafka.consumer_members",
                "Kafka: under-replicated partitions by topic": "kafka.under_replicated",
                "Kafka: brokers": "kafka.brokers"
            },
            "packs": [
                {
                    "id": "kafka-exporter",
                    "version": 1,
                    "variables": {
                        "matchers": "",
                        "window": "2m"
                    }
                }
            ]
        },
        "hard_resources": {
            "labels": [
                "Nodes: CPU usage (%) by node",
                "Nodes: Memory usage (%) by node",
                "Nodes: load1 per core by node",
                "Nodes: Disk utilization of the busiest device (%) by node",
                "Nodes: Network receive (bytes/s) by node",
                "Nodes: Network transmit (bytes/s) by node",
                "Nodes: Filesystem used (%) by node",
                "K8s nodes: Ready by node",
                "K8s nodes: CPU requests (% of allocatable) by node",
                "K8s nodes: Memory requests (% of allocatable) by node",
                "K8s nodes: Pods by node",
                "K8s workloads: container restarts (per 5m) by namespace & pod",
                "K8s workloads: CPU throttling (%) by namespace & pod (top 20)",
                "K8s workloads: memory working set (% of limit) by namespace & pod (top 20)",
                "K8s workloads: OOMKilled containers by namespace & pod",
                "K8s workloads: available replicas by namespace & deployment",
                "K8s workloads: desired replicas by namespace & deployment",
                "K8s workloads: HPA current replicas by namespace & hpa"
            ],
            "promql_queries": [
                'label_replace(100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle", job=~".*node.*" }[2m]))) * on (instance) group_left (nodename) max by (instance, nodename) (node_uname_info{ job=~".*node.*" }), "node", "$1", "nodename", "(.*)")',
                'label_replace(100 * (1 - max by (instance) (node_memory_MemAvailable_bytes{ job=~".*node.*" }) / max by (instance) (node_memory_MemTotal_bytes{ job=~".*node.*" })) * on (instance) group_left (nodename) max by (instance, nodename) (node_uname_info{ job=~".*node.*" }), "node", "$1", "nodename", "(.*)")',
                'label_replace(max by (instance) (node_load1{ job=~".*node.*" }) / count by (instance) (node_cpu_seconds_total{mode="idle", job=~".*node.*" }) * on (instance) group_left (nodename) max by (instance, nodename) (node_uname_info{ job=~".*node.*" }), "node", "$1", "nodename", "(.*)")',
                'label_replace(100 * max by (instance) (rate(node_disk_io_time_seconds_total{device!~"(loop|ram|fd|sr|dm-).*", job=~".*node.*" }[2m])) * on (instance) group_left (nodename) max by (instance, nodename) (node_uname_info{ job=~".*node.*" }), "node", "$1", "nodename", "(.*)")',
                'label_replace(sum by (instance) (rate(node_network_receive_bytes_total{device!~"lo|veth.*|docker.*|cali.*|flannel.*|cni.*|virbr.*|br-.*|vxlan.*|tunl.*", job=~".*node.*" }[2m])) * on (instance) group_left (nodename) max by (instance, nodename) (node_uname_info{ job=~".*node.*" }), "node", "$1", "nodename", "(.*)")',
                'label_replace(sum by (instance) (rate(node_network_transmit_bytes_total{device!~"lo|veth.*|docker.*|cali.*|flannel.*|cni.*|virbr.*|br-.*|vxlan.*|tunl.*", job=~".*node.*" }[2m])) * on (instance) group_left (nodename) max by (instance, nodename) (node_uname_info{ job=~".*node.*" }), "node", "$1", "nodename", "(.*)")',
                'label_replace(100 * max by (instance) (1 - node_filesystem_avail_bytes{fstype!~"tmpfs|overlay|squashfs|nsfs|ramfs|fuse.*", job=~".*node.*" } / node_filesystem_size_bytes{fstype!~"tmpfs|overlay|squashfs|nsfs|ramfs|fuse.*", job=~".*node.*" }) * on (instance) group_left (nodename) max by (instance, nodename) (node_uname_info{ job=~".*node.*" }), "node", "$1", "nodename", "(.*)")',
                'max by (node) (kube_node_status_condition{condition="Ready", status="true", job=~".*kube-state-metrics.*" })',
                '100 * sum by (node) (kube_pod_container_resource_requests{resource="cpu", node!="", job=~".*kube-state-metrics.*" }) / sum by (node) (kube_node_status_allocatable{resource="cpu", job=~".*kube-state-metrics.*" })',
                '100 * sum by (node) (kube_pod_container_resource_requests{resource="memory", node!="", job=~".*kube-state-metrics.*" }) / sum by (node) (kube_node_status_allocatable{resource="memory", job=~".*kube-state-metrics.*" })',
                'count by (node) (kube_pod_info{node!="", job=~".*kube-state-metrics.*" })',
                'sum by (namespace, pod) (increase(kube_pod_container_status_restarts_total{namespace=~".+", job=~".*kube-state-metrics.*" }[5m])) > 0',
                'topk(20, 100 * sum by (namespace, pod) (rate(container_cpu_cfs_throttled_periods_total{namespace=~".+", container!="", container!="POD"}[2m])) / sum by (namespace, pod) (rate(container_cpu_cfs_periods_total{namespace=~".+", container!="", container!="POD"}[2m])))',
                'topk(20, 100 * sum by (namespace, pod) (container_memory_working_set_bytes{namespace=~".+", container!="", container!="POD"}) / sum by (namespace, pod) (kube_pod_container_resource_limits{resource="memory", namespace=~".+", job=~".*kube-state-metrics.*" }))',
                'sum by (namespace, pod) (kube_pod_container_status_last_terminated_reason{reason="OOMKilled", namespace=~".+", job=~".*kube-state-metrics.*" }) > 0',
                'sum by (namespace, deployment) (kube_deployment_status_replicas_available{namespace=~".+", job=~".*kube-state-metrics.*" })',
                'sum by (namespace, deployment) (kube_deployment_spec_replicas{namespace=~".+", job=~".*kube-state-metrics.*" })',
                'sum by (namespace, horizontalpodautoscaler) (kube_horizontalpodautoscaler_status_current_replicas{namespace=~".+", job=~".*kube-state-metrics.*" })'
            ],
            "label_keys_list": [
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["node"],
                ["namespace", "pod"],
                ["namespace", "pod"],
                ["namespace", "pod"],
                ["namespace", "pod"],
                ["namespace", "deployment"],
                ["namespace", "deployment"],
                ["namespace", "horizontalpodautoscaler"]
            ],
            "roles": {
                "Nodes: CPU usage (%) by node": "node.cpu_pct",
                "Nodes: Memory usage (%) by node": "node.mem_pct",
                "Nodes: load1 per core by node": "node.load_per_core",
                "Nodes: Disk utilization of the busiest device (%) by node": "node.disk_util_pct",
                "Nodes: Network receive (bytes/s) by node": "node.net_rx",
                "Nodes: Network transmit (bytes/s) by node": "node.net_tx",
                "Nodes: Filesystem used (%) by node": "node.fs_used_pct",
                "K8s nodes: Ready by node": "node.ready",
                "K8s nodes: CPU requests (% of allocatable) by node": "node.cpu_requests_pct",
                "K8s nodes: Memory requests (% of allocatable) by node": "node.mem_requests_pct",
                "K8s nodes: Pods by node": "node.pods",
                "K8s workloads: container restarts (per 5m) by namespace & pod": "k8s.restarts",
                "K8s workloads: CPU throttling (%) by namespace & pod (top 20)": "k8s.cpu_throttled_pct",
                "K8s workloads: memory working set (% of limit) by namespace & pod (top 20)": "k8s.memory_pct_of_limit",
                "K8s workloads: OOMKilled containers by namespace & pod": "k8s.oom_killed",
                "K8s workloads: available replicas by namespace & deployment": "k8s.replicas_available",
                "K8s workloads: desired replicas by namespace & deployment": "k8s.replicas_desired",
                "K8s workloads: HPA current replicas by namespace & hpa": "k8s.hpa_replicas"
            },
            "packs": [
                {
                    "id": "node-exporter",
                    "version": 1,
                    "variables": {
                        "matchers": 'job=~".*node.*"',
                        "window": "2m"
                    }
                },
                {
                    "id": "k8s-nodes",
                    "version": 1,
                    "variables": {
                        "ksm_matchers": 'job=~".*kube-state-metrics.*"'
                    }
                },
                {
                    "id": "k8s-workload-health",
                    "version": 1,
                    "variables": {
                        "namespace": ".+",
                        "ksm_matchers": 'job=~".*kube-state-metrics.*"',
                        "window": "2m"
                    }
                }
            ]
        }
    }
}
