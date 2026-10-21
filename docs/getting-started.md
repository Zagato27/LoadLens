# Getting started

From the five-minute demo to the first report built from your own metrics: install LoadLens with Docker Compose,
describe the storage, the metric sources and the SLA in the first-run wizard, and create a report for a test window.
Deployment details (environment file, worker, updates, external database) are in [deployment.md](deployment.md);
every setting is described in [configuration.md](configuration.md).

- [What you need](#what-you-need)
- [Try the demo first](#try-the-demo-first)
- [Install with Docker Compose](#install-with-docker-compose)
- [The first-run wizard](#the-first-run-wizard)
- [Data sources and queries](#data-sources-and-queries)
- [SLA thresholds](#sla-thresholds)
- [The first report](#the-first-report)
- [The interface](#the-interface)
- [What the report contains](#what-the-report-contains)
- [Next steps](#next-steps)

## What you need

| Component | Requirement |
|---|---|
| Runtime | Docker with Compose v2 (recommended), or Python 3.12+ for a local run |
| Storage | TimescaleDB 2.x, or PostgreSQL 14+ with the TimescaleDB extension. The database user needs `CREATE` in the schema; the Compose stack brings its own database |
| Metrics | Prometheus or VictoriaMetrics (PromQL) and/or InfluxDB 1.x (InfluxQL) or 2.x (Flux), directly or through a Grafana 9+ data source proxy |
| LLM | An API key of Perplexity, an OpenAI-compatible API, Anthropic, or GigaChat certificates. Optional: reports can be built without the AI analysis |
| Background jobs | Redis 7+ and a Celery worker (part of the Compose stack). Without them reports run in a thread of the web process, which is enough for development |
| Browser | A current Chrome, Firefox, Edge or Safari; the UI loads no external resources |

## Try the demo first

The demo stack needs Docker only: an LLM stub and a Prometheus stub run inside it, so no API key or external system
is involved, and it never touches `docker/.env` or `settings.py` of a real installation.

```bash
docker compose -f docker/docker-compose.demo.yml --env-file docker/demo.env --profile demo up --build
```

Open http://localhost:5000 and sign in with `admin` / `loadlens-demo` (the values of `docker/demo.env`). Within a
minute or two the dashboard shows the `demo-checkout` service with two runs: `demo-checkout-step`, a seeded run, and
`demo-checkout-step-2`, which the real job pipeline builds against the two stubs. The archive, the comparison and the
capacity forecast are filled, and the dashboard carries a "Demo data" banner. Stop and remove the demo data with the
same command and `down -v`.

## Install with Docker Compose

1. Copy the environment file and fill it in:

   ```bash
   cp docker/env.example docker/.env
   ```

   Set `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, a URL-safe `REDIS_PASSWORD` (`openssl rand -hex 32`),
   a stable `LOADLENS_SECRET_KEY` (another `openssl rand -hex 32`) and the first administrator in `LOADLENS_ADMIN_USER`
   and `LOADLENS_ADMIN_PASSWORD` (at least 10 characters, not containing the login). Every variable is commented in
   [docker/env.example](../docker/env.example).

2. Create the configuration files from the examples (they are mounted into the containers and never baked into the
   image):

   ```bash
   cp settings.example.py settings.py
   cp metrics_config.example.py metrics_config.py
   echo '{}' > settings_runtime.json
   echo '{}' > metrics_config_runtime.json
   ```

   In `settings.py` set `storage.timescale` to the database of the stack: host `timescaledb`, port `5432`, the
   database name, user and password from `docker/.env` (or the unprivileged `APP_DB_USER` / `APP_DB_PASSWORD` role,
   see [deployment.md](deployment.md#the-database)). The rest can wait for the wizard.

3. Start the stack from the `docker/` directory:

   ```bash
   cd docker
   docker compose up -d --build
   ```

   Four containers come up: `app` (the web UI and API), `timescaledb`, `redis` and `celery_worker`, which builds the
   reports and the Confluence publications. Tables are created by the application; `docker/initdb/02_schema.sql`
   prepares them in advance, and data migrations run in the background at every start (table `app_migrations`).

4. Open http://localhost:5000 and sign in with the administrator from `docker/.env`. After the first login remove
   `LOADLENS_ADMIN_PASSWORD` from `.env`.

Without Docker: install `requirements.txt` into Python 3.12+, copy the same example files, create the administrator
with `python -m loadlens_app.auth create-user admin --role admin` and start `python app.py` (a development server on
port 5000; the Docker image runs gunicorn). Without `LOADLENS_SECRET_KEY` a key is generated into
`.loadlens_secret_key`. On a database that already has runs, execute `python -m loadlens_app.migrations` once.

## The first-run wizard

Open `/settings?wizard=1` (also the "First-run setup wizard" button at the top of the settings page). The wizard
walks through the sections of the settings page and checks every connection before you go on; each step can be
skipped and finished later in the regular settings.

| Step | What you set | Check |
|---|---|---|
| Language | The language of reports and prompts (`ru` or `en`). The interface language is a per-browser choice in the View menu of the top bar | |
| Storage | The `storage.timescale` connection | "Test connection" must pass |
| Data sources | The catalogue of connections: Prometheus, VictoriaMetrics, InfluxDB or Grafana, with authentication and TLS | Each connection is tested with the form values, before saving |
| Domain binding | Which source each domain reads (`default` plus per-domain overrides); for Grafana the data source is picked from the list Grafana returns | |
| Queries | "Add from a pack": query packs for your stack. The wizard runs the detection against your sources and sorts packs and bundles by the metrics it found. At least the load tool pack is needed | "Check" runs every query of the pack against the source |
| LLM | Provider, model, API key or certificates, token limits | The provider is called once |
| SLA | Thresholds and the assigned RPS query; can be filled later | |
| Done | "Load demo data" writes a synthetic run so you can look at a report before the first test; "Create the first report" opens the New report page | |

Everything the wizard writes goes to `settings_runtime.json` and applies without a restart. Secrets are shown as
`***` afterwards and are kept when you save a form that still has the mask.

## Data sources and queries

A domain is a group of series the report analyses together: load tool, JVM and runtime, microservices, databases,
Kafka, node resources, load generators and application logs. Every domain reads one source with one query language;
packs fill the queries, the series labels, the roles (which series is "total RPS", "p95", "node CPU") and the SLA
references in one step. [data-sources.md](data-sources.md) explains the catalogue, the bindings, the packs and which
pack matches which load tool; [query-packs.md](query-packs.md) lists every pack with its variables.

Two conventions matter for the SLA and the forecast: node CPU arrives in percent with a `node` series key, the CPU of
one service instance as a fraction 0..1 with an `instance` key, and load tool latencies in milliseconds. The packs
follow them; hand-written queries should too.

## SLA thresholds

"Settings → SLA criteria" (section `sla` of the configuration). Empty thresholds are not checked.

| Field | Meaning |
|---|---|
| `target_rps` | The RPS the test has to reach (step and stress tests: on the last stable step; soak: held for the whole hold window; spike: the spike level) |
| `max_error_rate_pct`, `max_p95_ms`, `max_p99_ms` | Error ratio and latency limits, checked on the same window |
| `max_cpu_pct`, `max_memory_pct` | Node resource limits; a violation gives "Risks found" rather than "Failed" on step and stress tests |
| `max_performance_query` | The assigned total RPS series: the step detector and the stable maximum work on it |
| `step_detection_enabled`, `step_detection_preset`, `min_stable_minutes` | Step detector switch (step and stress tests), preset `strict`, `balanced` or `lenient`, and the shortest plateau that counts as a step |
| `mean_latency_query`, `vus_query`, `load_model`, `latency_unit`, `service_cpu_query` | Series for the capacity forecast and the pod plan |

Query references that are left empty are resolved by role from the installed packs; the form shows such a reference
as "auto by role". Thresholds can differ per project area and per service.

## The first report

"New report" (`/new`):

1. Pick the service (recent ones first) and, if you use several, the project area.
2. Set the test window. Quick presets cover the last hours; after you choose a service the form looks for recent
   load windows in the load tool metrics of the last 6 hours ("Detected runs") and offers them, together with the runs
   started from LoadLens. Tests started from the "Run a test" page get their report automatically for the run window
   ([running-load-tests.md](running-load-tests.md)).
3. Choose the test type: step (maximum performance), soak, spike or stress. It selects the deterministic analysis,
   the SLA window and the instructions for the model.
4. Choose whether to run the AI analysis and whether to publish to Confluence, and start the job.

The job runs in the background with progress per phase; the page keeps the link `/new?job=<id>`, so the tab can be
closed and the job survives a restart of the server. The same report can be requested from CI with an API token
([api.md](api.md)):

```bash
curl -X POST http://localhost:5000/create_report \
  -H "Authorization: Bearer ll_..." -H "Content-Type: application/json" \
  -d '{"service": "checkout", "start": "2026-01-15T10:00:00+03:00", "end": "2026-01-15T11:00:00+03:00", "test_type": "step"}'
```

## The interface

The top bar holds the sections, then the project area (it filters every page), search over pages, reports and areas
(Ctrl+K), the View menu and the user menu; the version of the installation is next to the logo. The View menu
switches the theme (light, dark or system; system by default) and the interface language. Below 1024 px the sections
and the tools open from the menu button, and on phones tables turn into cards. Tables and charts sit on the page
without cards, figures and dates are set in a monospaced face, and a status is a colored square with colored text.

- **Dashboard.** A status line under the title: runs in 7 and 30 days and how many of them were at risk or failed,
  tests running now (engineers and administrators), report jobs in progress and the time since the last report; the
  verdicts by week on the right. The latest run follows, with a rule in its verdict color: the verdict, the figures
  against the SLA thresholds, how many SLA criteria passed and the change of Max RPS against the previous run of
  the same type, and its load profile with the SLA window, the target RPS and, if there was one, the degradation
  line. Below: services with a Max RPS sparkline and the recent runs.
- **Report.** The first screen reads in plain words, for a manager who opens the link for a minute: the verdict, one
  sentence with the figures that explains it ("The target RPS was reached on the stable step, but p95 is above the
  threshold: 450 ms against 300 ms"), three key figures with a bar against the threshold, and the SLA criteria. The
  test timeline below draws the steps; query and series names stay in small print under the criteria.
- **Archive.** Next to the verdict and Max RPS every run shows how many SLA criteria it met out of those evaluated
  (for example 2/3) and the p95 of its SLA check, red when it is over the threshold.

## What the report contains

- **Verdict and key numbers.** The verdict of the SLA check, `pass`, `risk`, `fail` or `no_data` (without SLA criteria
  it is always "Insufficient data"), one sentence with the figures that explains it, the maximum RPS, p95 and the
  error ratio against their thresholds, and the SLA criteria with the fact, the threshold and the status of every
  criterion.
- **Test timeline.** The steps as a staircase with the SLA window, the target RPS and, if there was one, the
  degradation line, and a row of steps below it: the RPS level and p95 of every plateau, the step the SLA was
  checked on, and marks for unstable steps, short dips and the step after a dip. Hovering a step highlights it on the
  timeline and on the charts.
- **Analysis tabs.** The overall conclusion and one tab per domain: verdict rationale, peak performance, findings
  with evidence and recommendations. Findings can be voted "Correct / Wrong"; the "Answer reliability" block shows
  how the answer was chosen and which numbers were confirmed by the data ([ai-analysis.md](ai-analysis.md)).
- **Charts.** Every domain's series with the step zones, the SLA step, the moment the load dropped, hold and spike
  windows and the thresholds of the SLA queries; shared zoom per tab, PNG export, print to PDF.
- **Engineer summary and manual verdict.** An engineer writes the summary, and can override the verdict with a
  mandatory reason; the automatic verdict stays visible and can be restored.
- **Publishing, comparison, forecast.** The report and the forecast go to Confluence as separate pages; the archive
  compares two runs with an automatically chosen baseline; the capacity forecast (step and stress tests) estimates the
  limit, the safe maximum and the number of pods for a target load.

## Next steps

- [configuration.md](configuration.md): every section of `settings.py`, per-area overrides, environment variables.
- [data-sources.md](data-sources.md): sources, bindings, query packs, load tools.
- [running-load-tests.md](running-load-tests.md): start k6, JMeter, Locust or Gatling from LoadLens.
- [deployment.md](deployment.md): worker, updates, external database, health checks.
- [api.md](api.md): the REST API and API tokens.
