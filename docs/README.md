# LoadLens documentation

User documentation is in English. The design notes of finished and planned work are in Russian and stay in this
directory as they were written; they are listed at the end of this page.

## User documentation

| Page | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | From the five-minute demo to the first report with your own metrics: requirements, Docker Compose, the first-run wizard, data sources, SLA, the first report and what it contains |
| [configuration.md](configuration.md) | `settings.py` section by section, the runtime files, project area and service overrides, environment variables, `metrics_config.py` |
| [data-sources.md](data-sources.md) | Prometheus, VictoriaMetrics, InfluxDB and Grafana; domains and bindings; series conventions; query packs and bundles; which pack matches which load tool |
| [ai-analysis.md](ai-analysis.md) | The analysis pipeline: context packs, candidates, judge, verification of numbers, verdict codes, test types, prompts and languages, system context, providers |
| [running-load-tests.md](running-load-tests.md) | The "Run a test" page: SSH, Docker, Kubernetes, GitLab CI, Jenkins and manual generators, tool templates, live metrics, run tags, security |
| [api.md](api.md) | The REST API: tokens, roles, every endpoint by area, `/about` and `/healthz` |
| [deployment.md](deployment.md) | Docker Compose in production: environment file, database and roles, Redis and the Celery worker, health checks, migrations, secrets, updates, external database |
| [development.md](development.md) | Tests, the UI sandbox, localization rules and the Cyrillic ratchet, prompts, query packs, SPDX headers, CHANGELOG and releases, CI |
| [query-packs.en.md](query-packs.en.md) | Generated reference of every query pack: variables, queries, roles, units (regenerate with `python -m AI.query_packs docs --lang en docs/query-packs.en.md`; the Russian twin is [query-packs.md](query-packs.md)) |

Related files in the repository root: [README.md](../README.md) (overview and quick start),
[README.ru.md](../README.ru.md) (the Russian README with more screenshots), [CHANGELOG.md](../CHANGELOG.md),
[LICENSE](../LICENSE), [NOTICE](../NOTICE), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md),
[TRADEMARK.md](../TRADEMARK.md). Screenshots and the demo GIF are in [screenshots/](screenshots/).

## Reference pages in Russian

The maintainer notes of the AI package ([AI/README.md](../AI/README.md)), the Docker guide
([docker/DOCKER_SETUP.md](../docker/DOCKER_SETUP.md)) and the live test stand
([docker/dev-stack/README.md](../docker/dev-stack/README.md)) are Russian, as are these two pages:

| File | Title | About |
|---|---|---|
| [load-tests.md](load-tests.md) | «Запуск тестов из LoadLens» | The original of running-load-tests.md: generators, templates, run tags, Gatling relabeling rules, API |
| [jobs-runner.md](jobs-runner.md) | «Фоновые задачи: поток веб-процесса или Celery-воркер» | Background jobs: modes, variables, failure handling, queue inspection, Windows development, the e2e scripts |

Design plans written while the features were built are kept by the maintainer outside the repository; code
comments that mention `plans/...` refer to them.
