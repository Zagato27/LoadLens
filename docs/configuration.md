# Configuration

LoadLens is configured by one Python file, `settings.py`, plus two JSON files the UI writes, a small Python file for
the legacy Confluence template report, and environment variables for authentication, background jobs and the
container. Start from [settings.example.py](../settings.example.py): every section is commented, and the queries in
it are the default query-pack bundle. None of the live files is committed: `settings.py`, `metrics_config.py`,
`settings_runtime.json`, `metrics_config_runtime.json`, `docker/.env` and `.loadlens_secret_key` are in `.gitignore`
and `.dockerignore` and are mounted into the containers at run time.

- [Files and precedence](#files-and-precedence)
- [Sections of settings.py](#sections-of-settingspy)
- [Project areas and service overrides](#project-areas-and-service-overrides)
- [Key reference](#key-reference)
- [Environment variables](#environment-variables)
- [metrics_config.py](#metrics_configpy)

## Files and precedence

| File | Role |
|---|---|
| `settings.py` | The base configuration, read at start by the application and the worker |
| `settings_runtime.json` | Everything saved from the settings page: sections, project areas (`per_area`), prompts. Deep-merged over `settings.py`, except `data_sources` and `domain_sources` (also those of an area): the page saves them whole, so their value here replaces the one of `settings.py`. Applied without a restart (the worker re-reads it before every job) |
| `metrics_config.py`, `metrics_config_runtime.json` | Grafana panels, Loki queries and Confluence page ids of the legacy template report, per area and service; the second file is written by "Settings → Visualizations" |
| Environment variables | Authentication, the job runner, migrations, caches and the Docker image; they win over `settings.py` |

The settings page edits every section except `auth` (`POST /config` refuses it; edit the file or use the
environment variables). `GET /config` returns secrets as `***`; a saved form that still contains the mask keeps the
stored value, as long as the address of the connection did not change. Values of `llm`, `logs_source`,
`confluence`, `storage`, `default_params`, `sla`, `appearance` and `limits` are validated against the example on
save; `POST /config` answers with `warnings` for unknown keys.

## Sections of settings.py

| Section | What it sets |
|---|---|
| `data_sources` | The catalogue of connections (`prometheus`, `grafana_proxy`, `influxdb`): address, authentication, TLS. Shared by all areas. See [data-sources.md](data-sources.md) |
| `domain_sources` | Which source each domain reads: `default` and per-domain entries (`jvm`, `database`, `kafka`, `microservices`, `hard_resources`, `lt_framework`, `lt_generators`); `datasource_uid` or `datasource_name` for Grafana, `database` for InfluxQL, `bucket` for Flux |
| `queries` | PromQL, InfluxQL or Flux queries per domain with their labels and series keys; the service keys `roles`, `packs` and the optional domain title are written by the query packs (a role can also be picked in the row editor); `descriptions` maps a row to a note for the AI analysis (up to 500 characters); `sources` maps a row to a source of its own ([data-sources.md](data-sources.md#a-source-per-query-row)) |
| `sla` | Thresholds, the assigned RPS query, the step detector and the series of the capacity forecast |
| `llm` | Provider, models, token limits, parallelism and self-consistency of the AI analysis. See [ai-analysis.md](ai-analysis.md) |
| `default_params` | Sampling step of the queries and the resampling interval before the analysis |
| `logs_source` | OpenSearch Dashboards for the optional `application_logs` domain (off by default) |
| `confluence` | Publication of the web report and the forecast as Confluence pages |
| `limits` | Report window, points per series, parallel jobs, daily LLM budgets, active load tests |
| `load_generators` | Generators of the "Run a test" page: `ssh`, `docker`, `kubernetes`, `gitlab`, `jenkins`, `manual`. See [running-load-tests.md](running-load-tests.md) |
| `load_tools` | Deviations from the built-in command templates of k6, JMeter, Locust and Gatling, and custom tools |
| `system_context` | The structured description of the system under test that goes into the prompts |
| `storage` | `storage.timescale`: the TimescaleDB connection, pool and table names |
| `appearance` | Accent color, the language of reports and prompts, the interface language switch |
| `auth` | Sign-in, sessions, API tokens, lockout, the recovery administrator; not editable from the UI |

The eight top-level keys `url_basic`, `space_conf`, `user`, `password`, `grafana_base_url`, `grafana_login`,
`grafana_pass` and `loki_url` belong to the legacy Confluence template report (Grafana panel renders and Loki logs
on a page template); the settings page shows them as one connection card and `/config` groups them as the
`confluence_template` section.

## Project areas and service overrides

A project area ("Settings → Projects") is a scope with its own services. Without an area selected in the top bar
you edit the global values; with an area selected you edit its overrides, stored under `per_area.<area>` in
`settings_runtime.json`. A service inside an area can override a few sections once more; the context of a service
replaces the context of its area entirely.

| Overridable | Per area | Per service |
|---|---|---|
| `llm`, `domain_sources`, `logs_source`, `default_params` | yes | no |
| `queries`, `sla`, `system_context`, prompts | yes | yes |
| `data_sources`, `storage`, `confluence`, `limits`, `appearance`, `auth` | global only | |

The project card shows which sections an area or a service overrides; `DELETE /projects/<id>/overrides/<section>`
and the matching service endpoint remove an override. Prompts are edited in "Settings → Prompts" with a version
history; an empty text or a copy of the default of either language does not count as an override, so the default
prompt of the report language is used (see [ai-analysis.md](ai-analysis.md#prompts-and-languages)).

## Key reference

### sla

| Key | Meaning |
|---|---|
| `target_rps` | RPS to reach. Step and stress tests: on a step where the latency and error thresholds hold (the target step, 2 % below the target still counts); CPU and memory are checked on that step and the steps above it are headroom that does not affect the verdict; soak: held for the whole hold window (average not more than 5 % below, no interval more than 15 % below); spike: the spike level. `None` disables the check |
| `target_margin_pct` | Step and stress tests: margin over the target in percent. The latency and error thresholds must also hold at `target_rps × (1 + margin / 100)`, otherwise the `target_margin` check fails and the verdict is `risk`; CPU and memory are then checked on that step. `None` or `0` disables it |
| `max_error_rate_pct`, `max_p95_ms`, `max_p99_ms` | Limits of the error ratio and the latency percentiles; `error_rate_query`, `p95_query`, `p99_query` name the series |
| `max_cpu_pct`, `max_memory_pct` | Node limits, read from `cpu_query` and `memory_query` (percent, key `node`). Step and stress tests check them on the target step |
| `max_performance_query` | The total RPS series of the load tool: the step detector, the stable maximum and the live charts work on it |
| `step_detection_enabled`, `step_detection_preset`, `min_stable_minutes` | Step detector (step and stress tests only; soak and spike derive their own windows), preset `strict`, `balanced` or `lenient`, shortest plateau that counts |
| `target_rps_allow_peak_fallback` | When no stable maximum was found, use the peak RPS of the window as the achieved RPS (default `True`) |
| `mean_latency_query`, `vus_query`, `latency_unit`, `load_model` | Series and units of the capacity forecast: `latency_unit` is `ms` or `s`; `load_model` is `open` (concurrency = RPS x mean time) or `closed` (concurrency = VUs) |
| `service_cpu_query` | CPU of one instance as a fraction 0..1, grouped by service and instance: the pod plan of the forecast |
| `debug_peak_logging` | Verbose log of the step detector |

### llm

| Key | Meaning |
|---|---|
| `provider` | `perplexity`, `openai`, `anthropic` or `gigachat`; each has a block of the same name |
| `max_domain_workers` | Domains analysed in parallel |
| `self_consistency_k` | Candidates generated per domain; with `1` the judge is skipped |
| `self_consistency` | Throttling inside a domain: `max_candidates`, `parallel_candidates`, `parallel_critics`, `candidate_workers`, `critic_workers`, `pause_sec_between_calls` |
| `baseline` | `enabled` and `mode` (`previous_success` or `previous`): the previous run of the service in the analysis context |
| `verification_pass` | An extra model call that bases the verdict only on findings whose numbers were found in the metrics |
| `include_markdown_tables_in_context` | Debug: markdown tables of the series in the prompt |

Provider blocks share `api_base_url`, `model`, `api_key`, `max_concurrent`, `generation` (`temperature`, `top_p`,
`max_tokens`, `force_json_in_prompt`), `verify`, `proxies`, `connect_timeout_sec` and `request_timeout_sec`.
Perplexity adds `disable_web_search`; GigaChat uses `cert_file` and `key_file` for mTLS instead of a key and adds
`max_attempts`, `request_min_interval_sec` and `rate_limit_cooldown_sec` for a provider with strict rate limits.
The optional provider key `max_tokens_cap` (or `LOADLENS_LLM_MAX_TOKENS_CAP`) caps `max_tokens`.

### default_params and limits

| Key | Default | Meaning |
|---|---|---|
| `step` | `1m` | Sampling step of the metric queries |
| `resample_interval` | `10T` | Aggregation interval before the analysis (pandas offset) |
| `max_report_window_hours` | 168 | Longest window one report may cover |
| `max_points_per_series` | 11000 | Window divided by `step`; Prometheus refuses more |
| `max_parallel_jobs`, `max_parallel_jobs_per_user` | 4, 2 | Reports and publications running or queued at once, counted in `report_jobs` |
| `llm_reports_per_user_per_day`, `llm_explanations_per_user_per_day`, `llm_context_imports_per_user_per_day` | 20, 50, 20 | Daily budgets of reports with AI analysis, forecast explanations and context imports from documents |
| `max_parallel_load_tests`, `max_parallel_load_tests_per_user`, `max_load_test_instances`, `max_load_test_hours` | 5, 3, 20, 24 | Active load test runs, instances per run, longest run |

`None` switches a limit off. Except the parallel jobs, the counters live in the process: with several gunicorn
processes each one enforces its own share.

### storage.timescale

| Key | Meaning |
|---|---|
| `host`, `port`, `dbname`, `user`, `password`, `sslmode`, `schema` | The connection; the user needs `CREATE` in the schema |
| `connect_timeout_sec`, `retry_seconds` | Wait for a connection at most 5 s; then answer "Storage unavailable" for 10 s without new attempts (recovery mode) |
| `pool_max`, `pool_idle_sec` | Connections the process keeps for reuse (`0` disables the pool) and the idle time after which one is closed |
| `table`, `batch_size`, `make_hypertable`, `ensure_extension`, `chunk_interval` | The metrics hypertable |
| `llm_table`, `engineer_table`, `llm_feedback_table`, `jobs_table`, `confluence_table`, `forecast_table` | Names of the report, summary, feedback, jobs, publication and forecast explanation tables |

### appearance

| Key | Meaning |
|---|---|
| `accent` | Accent color of the installation, `#rrggbb` |
| `language` | Language of new reports, prompts, forecast explanations and document imports: `ru` or `en`. Also the interface language of a browser that has not chosen one. A report job takes the language when it is queued |
| `language_switch` | `false` ignores the `loadlens.lang` cookie, so everyone sees the interface in `language`; `POST /config` applies it without a restart |

### auth

| Key | Meaning |
|---|---|
| `enabled` | Sign-in required (`LOADLENS_AUTH_ENABLED` wins) |
| `providers` | Provider chain tried in order, e.g. `["ldap", "local"]`; `local` keeps the accounts in the database. Register others with `loadlens_app.auth.register_provider` |
| `api_tokens`, `token_default_days`, `token_max_days` | Personal API tokens and their lifetime (at most 365 days) |
| `session_hours`, `session_max_hours` | Idle and absolute lifetime of a browser session |
| `cookie_secure`, `trusted_proxies` | `Secure` cookies behind HTTPS and the number of reverse proxies whose `X-Forwarded-*` headers are trusted |
| `password_min_length`, `login_max_attempts`, `login_window_seconds`, `max_failed_logins`, `lockout_minutes` | Password policy and lockout: 5 failures per address and login in 5 minutes, 10 failures on an account lock it for 15 minutes |
| `default_external_role` | Role of users created on first login by an external provider |
| `recovery_user`, `recovery_password_hash` | The recovery administrator who can sign in while the database is down; the hash comes from `python -m loadlens_app.auth hash-password` |

### confluence

| Key | Meaning |
|---|---|
| `enabled`, `base_url`, `username`, `password`, `verify_ssl`, `request_timeout_sec` | The Confluence connection used to publish reports and forecasts and to read pages for the context import |
| `space_key`, `parent_page_id`, `forecast_parent_page_id` | Where the report pages and the forecast pages are created |
| `app_url` | Address of LoadLens for the links on published pages; empty uses the address the browser used |

### logs_source

`enabled` switches the `application_logs` domain on. `opensearch` holds `base_url`, `index_pattern`,
`username_env`, `password_env` (historical names: the values are the login and the password), `verify_ssl` and
`request_timeout_sec`; `fields` maps the timestamp, level, hostname and message fields; `filters` lists
`error_levels` and `hostnames`; `aggregation` limits the documents read and the number of error types, services and
samples that reach the context; `report` sets the output of the command-line report.

### system_context

A JSON document with `schema_version`, `system` (`name`, `domain`, `description`, `test_goal`), `architecture`
(`style`, `components`, `dependencies`, `data_stores`), `load_model` (`entrypoints`, `critical_user_flows`,
`expected_hotspots`) and `operational_context` (`known_constraints`, `known_risks`, `normal_degradation_rules`,
`analysis_focus`). It is edited in "Settings → System context", can be filled from a PDF, Word, Markdown or text
document or a Confluence page by the model ("Fill from a document", modes "Fill" and "Replace"), and a snapshot is
stored with every report. See [ai-analysis.md](ai-analysis.md#system-context).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LOADLENS_AUTH_ENABLED` | `1` | `0` opens the application to everyone with administrator rights (debugging only) |
| `LOADLENS_SECRET_KEY` | | Signs sessions. Without it a key is generated into `.loadlens_secret_key`, or a temporary one when that file cannot be written |
| `LOADLENS_ADMIN_USER`, `LOADLENS_ADMIN_PASSWORD` | | The first administrator, created on the first visit of `/login` while there are no users; also opens the recovery mode while the database is down |
| `LOADLENS_RECOVERY_USER`, `LOADLENS_RECOVERY_PASSWORD_HASH` | | The recovery administrator (login and scrypt hash) |
| `LOADLENS_COOKIE_SECURE` | request scheme | `1` behind HTTPS: cookies get the `Secure` flag |
| `LOADLENS_TRUSTED_PROXIES` | `0` | Number of reverse proxies in front of the application; then `X-Forwarded-For/-Proto/-Host` are honoured and login throttling counts client addresses |
| `LOADLENS_CSP_REPORT_ONLY` | | `1` switches the Content-Security-Policy header to report-only mode (browser console) |
| `LOADLENS_JOB_RUNNER` | `auto` | Where reports and publications run: `auto` (the worker when a broker is set and eager mode is off), `thread`, `celery` |
| `LOADLENS_JOB_STALE_MIN` | `15` | Minutes without a heartbeat after which a worker job is marked as interrupted |
| `LOADLENS_JOB_QUEUE_MAX_MIN` | `120` | Minutes a job may wait in the queue for a worker |
| `LOADLENS_REPORT_TASK_LIMIT_SEC` | `3600` | Soft time limit of one job in the worker; the process is killed 60 s later |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | | Redis address, with the password in the URL |
| `CELERY_TASK_ALWAYS_EAGER` | `1` in code, `0` in Compose | `1` runs Celery tasks synchronously; for `auto` this means the thread runner |
| `CELERY_TASK_TIMEOUT` | `600` | Seconds for downloading one Grafana panel or Loki log and uploading one Confluence attachment |
| `LOADLENS_RUN_MIGRATIONS` | `1` | Check and run data migrations in the background at every start; `0` leaves them to `python -m loadlens_app.migrations` |
| `LOADLENS_LOAD_TEST_WATCHER` | | `1` starts the watcher of load test runs with the process (Compose); unset, it starts when the launch page is first opened; `0` disables it (tests) |
| `LOADLENS_RESPONSE_CACHE` | `1` | `0` disables the in-process response cache (`/dashboard_data` 10 s, `/jobs` and `/load_tests/active` 3 s) |
| `LOADLENS_STATIC_NO_CACHE` | | `1` while editing the front end: no long browser cache for static files; otherwise they are cached for a year and linked with `?v=<content hash>` |
| `LOADLENS_DEMO_SEED` | | `1` seeds the demo run and builds one stub report at start (the demo stack); idempotent |
| `LOADLENS_BUILD`, `LOADLENS_VERSION` | | Build id and version of the Docker image (`--build-arg`); shown by `GET /about` |
| `LOADLENS_LLM_MAX_TOKENS_CAP`, `LOADLENS_JUDGE_WARN_PROMPT_CHARS` | | Cap of `max_tokens` for every provider; prompt size above which the judge logs a warning (60000) |

The variables of the Compose stack itself (`POSTGRES_*`, `APP_PORT`, `TSDB_PORT`, `TSDB_BIND`, `APP_DB_USER`,
`REDIS_PASSWORD`, `WORKER_CONCURRENCY`, container limits) are described in [deployment.md](deployment.md).
Variables used only by the tests (`LOADLENS_REQUIRE_NODE`, `LOADLENS_I18N_STRICT`, `LOADLENS_TEST_PG_DSN`) are in
[development.md](development.md).

## metrics_config.py

`metrics_config.py` (example: [metrics_config.example.py](../metrics_config.example.py)) feeds the legacy
Confluence template report: for every area and service it names the Confluence template and parent page and the
Grafana panels and Loki queries whose renders go onto the page. The web report and the forecast pages do not use it.

| Field | Meaning |
|---|---|
| `<area>.title`, `<area>.services` | An area with its services; a flat legacy format with `area` inside each service is also accepted |
| `page_sample_id`, `page_parent_id` | The Confluence page used as a template and the parent of the new page |
| `metrics[]` | `name` and `grafana_url`: the `/render/d-solo/...` URL of a panel relative to `grafana_base_url` |
| `logs[]` | `placeholder` in the template and the Loki `filter_query` sent to `loki_url` |

Changes made in "Settings → Visualizations" are written to `metrics_config_runtime.json` and apply at once.
