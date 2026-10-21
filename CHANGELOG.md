# Changelog

All notable changes to LoadLens are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Release dates are written by `tools/release.py`
(`python tools/release.py --version X.Y.Z`), which also moves the "Unreleased" entries into the new section.

## [Unreleased]

### Added

- **Runs API.** `GET /runs?include=sla` adds `sla` to every run: `passed`, `total`, `p95_ms` and `p95_passed` from the
  SLA checks of its final report, `null` without evaluated checks. Without `include` the response is unchanged.
- **Dashboard API.** `GET /dashboard_data` returns `last_run.checks` (the SLA checks with `name`, `title`,
  `threshold`, `actual`, `passed`) and `last_run.profile` (the load profile: `steps`, `window`, `markers`, `target`;
  `null` without a step table).

### Changed

- **Interface look.** A top bar with text navigation replaces the sidebar: the sections on the left, the project
  area, search (Ctrl+K), the View menu and the user menu on the right, the version next to the logo. Below 1024 px
  the navigation opens from the menu button.
- **Fonts.** Golos Text for text and JetBrains Mono for figures, dates, identifiers and chart axes replace Inter and
  Montserrat; both are bundled under the SIL Open Font License.
- **Themes.** The light and the dark theme are equal, and without a saved choice the theme follows the system. The
  light theme is a paper-like page with status colors picked for contrast.
- **Report.** The first screen says it in plain words: the verdict, one sentence with the figures that explains it,
  three key figures with a bar against their thresholds and the SLA criteria; query and series names move to small
  print. The "Test timeline" draws the steps as a staircase with the SLA window, the target RPS and the degradation
  line, with the row of steps below it.
- **Dashboard.** A status line under the title replaces the KPI tiles; the latest run shows its figures against the
  SLA thresholds, a square per SLA criterion and its load profile.
- **Archive.** New SLA (criteria met out of those evaluated) and p95 columns; a p95 over its threshold is red.
- **Tables, forms and statuses.** Tables and charts sit on the page without cards, separated by hairlines; forms and
  settings are flat; a status is a colored square with colored text instead of a pill. The accent color marks only
  the logo, the focus, the SLA band and the target marks on charts.

## [1.0.0] - unreleased

First public release, Apache License 2.0. LoadLens turns the metrics of a load test into a report with an SLA
verdict, an AI analysis that is checked against the data, and a Confluence page, and it can start the test itself.

### Added

- **Reports and SLA.** Step detection on the assigned RPS series (plateaus, stable maximum, the moment the load
  drops); a programmatic SLA check on the last stable step: target RPS, error ratio, p95/p99, CPU and memory of the
  nodes. A report page with the SLA table, test steps, charts, engineer summary, findings votes and a manual verdict
  override. Archive with search and filters, comparison of two runs with an automatically chosen baseline.
- **Test types.** Dedicated deterministic analysis per type: step tests (stable maximum), soak tests (hold window
  and latency/error/resource trends), spike tests (spike windows, overshoot, recovery time) and stress tests
  (saturation point, degradation pattern, limiting resource). One test profile drives SLA, prompts, forecasting and
  rendering.
- **AI analysis with verification.** Domain reports (JVM, databases, Kafka, microservices, infrastructure, load tool,
  load generators, application logs from OpenSearch) and an overall conclusion with a verdict rationale, findings
  and recommendations. Several candidates per domain, a G-Eval-lite judge and a check of every number in the
  findings against the collected data. Providers: Perplexity, OpenAI-compatible APIs, Anthropic, GigaChat (mTLS).
  Structured system context of the system under test, filled by hand or imported from documents (PDF, Word), and
  the previous run in the analysis context.
- **Data sources and query packs.** Prometheus (PromQL) and InfluxDB (InfluxQL, Flux), directly or through the
  Grafana proxy; a catalogue of sources with domain bindings. Query packs for common stacks (node_exporter,
  kube-state-metrics, JVM, Go, Python, HTTP frameworks, Kafka, PostgreSQL, HikariCP, k6, JMeter, Gatling with
  Graphite or VictoriaMetrics) that are checked against a source and fill the queries of a domain and the SLA
  references in one step.
- **Load-test launching.** The "Run a test" page starts k6, JMeter, Locust and Gatling (or any tool with a command
  template) on SSH, Docker, Kubernetes, GitLab CI and Jenkins generators, or marks the window of a manually started
  test. Several instances per run, delayed start, graceful stop, duration limit, live RPS, errors, p95 and users,
  the tool log tail, presets per service and a report built automatically for the run window. Runs survive a
  restart of LoadLens.
- **Forecasting.** Capacity forecast with the Universal Scalability Law: the limit, the safe maximum, the number of
  pods for a target load, a model-written explanation of the limit, and a Confluence page for the forecast.
- **Confluence export.** The report and the forecast are published as separate Confluence pages that repeat the
  web report: SLA criteria with threshold, fact and status, engineer summary, test steps, domain analysis and charts.
- **Jobs.** Background jobs with phase progress that survive a closed tab and a server restart; an optional Celery
  worker with Redis, a thread runner otherwise.
- **Projects and settings.** Project areas with their own connections, SLA, queries and prompts; per-service
  overrides; a first-run wizard, connection checks, a prompt editor with history, a demo run that needs no external
  systems.
- **Authentication and recovery.** Login with roles (viewer, engineer, administrator), API tokens for CI, an audit
  log, pluggable providers (LDAP, OIDC), CSRF and CSP protection, and a recovery mode that lets an administrator fix
  the storage connection while the database is down.
- **Localization.** Russian and English interface; each user picks the interface language in the browser, the
  language of reports and prompts is a setting of the installation. Reports, prompts, SLA texts, Confluence pages
  and API messages are translated.
- **Appearance.** Dark and light themes, a configurable accent color, no external CDNs.
- **About.** `GET /about` with the version, build and license, an "About" section in the settings with the
  third-party notices, and the version next to the logo.

[Unreleased]: https://github.com/Zagato27/LoadTestingAutoReports/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Zagato27/LoadTestingAutoReports/releases/tag/v1.0.0
