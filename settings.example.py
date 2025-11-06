# Пример конфигурации (обезличено). Скопируйте в settings.py и заполните свои значения.

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
        "include_markdown_tables_in_context": False,
        "provider": "openai",  # perplexity | openai | anthropic
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

    "default_params": {"step": "1m", "resample_interval": "10T"},

    "metrics_source": {
        "type": "grafana_proxy",
        "grafana": {
            "base_url": "http://grafana:3000",
            "verify_ssl": False,
            "auth": {"method": "basic", "username": "admin", "password": "admin", "token": ""},
            "prometheus_datasource": {"id": None, "uid": "your-datasource-uid", "name": "Prometheus"}
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
            "llm_table": "llm_reports"
        }
    },

    # Ниже примерная структура запросов — адаптируйте под свои метрики
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
        }
    }
}


