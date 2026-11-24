# Changelog

All notable changes to LoadLens are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Release dates are written by `tools/release.py`
(`python tools/release.py --version X.Y.Z`), which also moves the "Unreleased" entries into the new section.

## [Unreleased]

## [2.1.0] - 2026-11-24

### Added

- Step and stress reports accept named generator intervals with a run tag through the report form or API.
  Charts, step statistics and SLA use the same boundaries. Reports show planned and measured throughput
  separately and apply the configured minimum hold time to capacity checks ([#10](https://github.com/Zagato27/LoadLens/issues/10)).
- Save and switch named LLM connections in **Settings → Connections**. Share a connection from its project
  with **Available to all projects**. Each project can select it independently or follow the shared default.
- Set an authenticated proxy URL for an OpenAI-compatible, Anthropic or Perplexity connection.

### Changed

- The New report form now searches the last seven days of metrics and finished runs when suggesting test
  periods, down from 30 days. Users can still enter older report dates manually.

### Fixed

- The web report and Confluence explain the final verdict using the configured SLA checks and their units.
  The verification count refers to AI findings. Older reports label their saved AI rationale as unchecked
  ([#11](https://github.com/Zagato27/LoadLens/issues/11)).
- The demo starts on single-vCPU hosts and processes manual run start and stop requests with the default
  configuration ([#2](https://github.com/Zagato27/LoadLens/issues/2), [#5](https://github.com/Zagato27/LoadLens/issues/5)).
- Comparison explanations check each value against the named run and metric. Numeric verification retains
  percent units and checks measurements beyond the first twelve
  ([#6](https://github.com/Zagato27/LoadLens/issues/6), [#7](https://github.com/Zagato27/LoadLens/issues/7), [#8](https://github.com/Zagato27/LoadLens/issues/8)).
- OpenAI-compatible analysis and judge requests fit within the model's context window. Concurrent requests
  respect the connection's slot limit. Failed calls identify the affected domain and stage
  ([#3](https://github.com/Zagato27/LoadLens/issues/3), [#4](https://github.com/Zagato27/LoadLens/issues/4)).
- Overlapping reports in the thread runner keep separate LLM usage and cost totals
  ([#9](https://github.com/Zagato27/LoadLens/issues/9)).
- Honor Moonshot's Kimi request parameters in connection checks and reports, and surface API validation errors in connection checks.
- Back off and honor `Retry-After` when an LLM connection returns HTTP 429. Keep SLA results and test figures when model analysis fails.

## [2.0.0] - 2026-11-17

### Upgrade notes

This is a major release: comparison API percentages now evaluate A relative to B, and the automatic step detector
replaces the removed tuning settings. Read the [2.0 upgrade guide](docs/upgrading-to-2.md) before updating API
consumers, configuration and the application/worker. Existing stored reports are not rebuilt automatically.

### Added

- **Release marks, accepted runs and the pinned baseline.** An engineer marks a report as a test of a release
  («Mark release…» in the archive row menu, «Release…» under the verdict of the report page, or the new «Release»
  field of the Run test form, set on the report when it is ready). A release, service and test type has one accepted
  run: the first marked run, then each newly marked one, until a run is accepted by hand («Accept in the release»);
  when the accepted run loses its mark the run with the newest report takes over. Any run can be pinned as the
  baseline of its service and test type. The archive gets a «Releases» tab (`/reports?tab=releases`, filter by
  release, newer releases first), the marks next to the run name and «Compare with the previous release»; the
  Confluence page of a report names its release and baseline. API: `GET/PUT/DELETE /runs/<run_name>/release`,
  `PUT/DELETE /runs/<run_name>/baseline`, `GET /releases_data`, `release` of `GET /runs` and `POST /load_tests`
  ([docs/api.md](docs/api.md#runs-and-verdicts)). Table `release_marks`, created on first use.
- **Release-aware baseline modes.** `llm.baseline.mode` and `GET /compare_baseline` accept `pinned` (the pinned
  baseline, else the accepted run of the previous release, else the previous successful run) and
  `previous_release`; the answer and the model context name the mode that found the run and its release. The
  compare page has «Baseline» and «Previous release» buttons and reads `baseline=<mode>` from its address
  ([docs/ai-analysis.md](docs/ai-analysis.md#the-baseline-run)).

- **Release pages.** «Releases» in the top bar lists the releases of the project area with their status
  (incomplete while a service of the area has no accepted run, else the worst verdict), the services tested, the
  verdicts of the accepted runs and the Jira ticket. The page of a release has one row per service and test type:
  the accepted run with its verdict, the change of p95, errors and the load reached against the accepted run of the
  previous release at a comparable load (coloured by the regression thresholds of the service SLA), a link to that
  comparison, the attempts with «Accept» and «Pin as baseline»; the services not tested yet; and the trend by
  release of a service (three charts: p95, errors, load; the point colour is the verdict, the baseline a diamond).
  API: `GET /releases_overview`, `GET/PUT/DELETE /releases_data/<name>`, `GET /release_trend`.
- **A release as one Confluence page.** «Publish to Confluence» on the release page (background job
  `release_confluence`, Celery task `loadlens.publish_release`) writes the status, the table of services with links
  to their published report and comparison pages (else to LoadLens), the services not tested and a trend picture per
  service. Parent page: `confluence.release_parent_page_id`, else `parent_page_id`.
- **Jira links.** The global settings section `jira` (`enabled`, `base_url`, `project_key`; a card without a
  connection check) adds «Issues in Jira» (`project = KEY AND fixVersion = "<release>"`) and the release ticket to
  the release page and its Confluence page. LoadLens never calls Jira.

- **Scenario test type.** `test_type: scenario` is for one pass of a business process where the load moves from
  stage to stage. The run is split into phases by the request groups of the load tool (k6 groups, JMeter
  transactions, Gatling and Locust requests; query role `lt.rps_by_request`): every interval goes to the group with
  the largest share of RPS, short segments join their neighbours, nested k6 groups pick their level automatically.
  The phases replace the load steps of the report; the SLA is checked in every phase (the worst phase per
  threshold, `target_rps` does not apply) and against the previous scenario run of the service: missing phases
  (`phase_completeness`) and longer or shorter phases (`phase_duration`) are risks. A latency or error threshold
  fails a phase only when it stays broken `phase_breach_minutes` (2); a shorter excursion is a risk. Phases matched
  by `phase_exclude` (preparing and cleaning up test data) are shown but not judged. The report shows the phases
  with their verdicts, the compare page and its Confluence page compare two runs phase against phase
  (`GET /compare_phases`), and the model reads the phases instead of steps
  ([docs/ai-analysis.md](docs/ai-analysis.md#test-types-and-deterministic-analysis)).
- **AI explanation of a comparison.** The compare page says whether run A is better or worse than the baseline B
  (`better`, `worse`, `same`, `mixed`, `incomparable`) from rules: the final verdicts, then the maximum RPS of a step
  test, then the p95 metrics whose change reaches 10 %, with the caveats that make the comparison approximate.
  "Analyse with AI" has the model explain the difference from these facts and the verified findings of both reports;
  its numbers are checked, the answer is cached per pair and written in the language of the report of A
  (`GET`/`POST /compare_explanation`, prompt `compare_prompt.txt`, budget `llm_explanations_per_user_per_day`).
  "Add comparison to Confluence" publishes the outcome, the verdicts, the largest changes and the explanation as its
  own page (`POST /compare_confluence`, job `compare_confluence`, `confluence.compare_parent_page_id`)
  ([docs/ai-analysis.md](docs/ai-analysis.md#other-model-calls-and-limits)).
- **Application logs from Loki.** `logs_source.type: loki` reads the ERROR logs of the test window from Grafana Loki,
  through a Grafana data source of the catalogue or directly (`X-Scope-OrgID`, basic auth). A metric LogQL query gives
  exact error counts per interval and service; limited log queries for the busiest intervals and services give the
  error types and samples (JSON, logfmt or plain lines; the error filter is built from the levels or written in
  LogQL). Works with Loki 2.x and 3.x; "Check" reports the lines and errors of the last 15 minutes
  ([docs/data-sources.md](docs/data-sources.md#application-logs-from-opensearch-or-loki)).
- **Deploy to a systemd installation.** `python tools/deploy.py` updates a git clone run by systemd units over one
  SSH session: checks that the commit is pushed, refuses a server checkout with local changes, installs
  `requirements.txt` only when it changed, restarts the units, waits for them and `/healthz`, and prints their logs
  otherwise; `check` inspects the server read-only and `rollback` returns to the previous commit
  ([docs/deployment.md](docs/deployment.md#updating-a-systemd-installation-from-your-computer)).
- **arm64 image.** Releases publish `ghcr.io/<owner>/loadlens` for linux/arm64 (Apple Silicon, ARM servers) next to
  linux/amd64; both platforms are started once in CI before the push.
- **HTTPS with Caddy.** `docker/docker-compose.https.yml` and `docker/Caddyfile` put Caddy in front of the production
  stack: automatic certificates for a public name or `tls internal`, the app port on `127.0.0.1` only, Secure
  cookies and one trusted proxy ([docs/deployment.md](docs/deployment.md#https-with-caddy)).
- **English query pack catalogue.** `docs/query-packs.en.md`, generated by
  `python -m AI.query_packs docs --lang en docs/query-packs.en.md`; the English docs link to it.
- **Link check.** `.github/workflows/links.yml` checks the external links of every markdown file with lychee on
  documentation changes and weekly.
- **A source per query row.** A row of any domain can read another source than the domain binding
  (`queries.<domain>.sources`): two Prometheus servers, two Grafana data sources, node_exporter of the database hosts
  in an infrastructure Prometheus. The row editor and the pack card have a "Source" field; checks, reports, live
  charts and load tool hints follow it; a catalogue source that a row reads cannot be removed
  ([docs/data-sources.md](docs/data-sources.md#a-source-per-query-row)).
- **Several query languages in one domain.** k6 in InfluxDB behind Grafana and Gatling in VictoriaMetrics fill one
  "Load testing tool" section: every row runs its own language, the report collects a domain in groups per source
  and language, and a source that fails leaves only its own rows empty. The row editor has a language field and marks
  rows with texts in two languages.
- **Hosts of a component.** The packs `host-node-exporter` and `host-windows-exporter` put the CPU, memory, disk and
  network of database, broker or service hosts into the section of the domain picked at install time, up to 10
  installations per domain told apart by the host title. Their `host.*` roles fill no SLA field.
- **Pack detection reads more sources.** "Detect from sources" also reads the direct Prometheus sources of the
  catalogue and the sources rows already read, and offers "Read from it" when a pack's metrics are found there.
- **Margin over the target RPS.** `sla.target_margin_pct` (settings: SLA, step detector block) asks step and stress
  tests to hold the latency and error thresholds at the target plus a margin; the `target_margin` check fails with
  `risk` otherwise, and CPU and memory are then checked on that step
  ([docs/configuration.md](docs/configuration.md#sla)).
- **Domain metrics in the prompt.** Each domain prompt now lists every configured query of the domain with its
  meaning, unit, series keys and note, and marks the queries that returned no data in the run. Queries added by hand
  get a "Role" and a "Note for the AI analysis" in the row editor (`queries.<domain>.descriptions`); the note
  replaces the description of a pack query ([docs/ai-analysis.md](docs/ai-analysis.md#prompts-and-languages)).
- **Runs tab in the archive.** The archive has two tabs, Reports and Runs (`/reports?tab=runs`, also the "All runs"
  link on the Run test page). The Runs tab lists the load test runs of the active area with status, start and
  duration, instances, owner and report, filters by name, service and status, and is open to viewers without
  commands or parameters (`GET /load_tests/history`). Finished runs are deleted from the row menu or with checkboxes:
  an engineer deletes their own, an administrator anyone's and may delete the runs' reports with them
  (`POST /load_tests/runs/delete`, `DELETE /load_tests/runs/<run_id>`, audit `load_test.delete`)
  ([docs/running-load-tests.md](docs/running-load-tests.md#run-history-in-the-archive)).
- **Bulk deletion of queries.** Every row of the query list in the settings has a tick, and the bar above the list
  ticks all shown rows, counts the ticked ones and deletes them after one confirmation that names them and the ones
  the SLA reads. The change reaches the configuration with the Save button, like a single deletion.
- **Off domains in the query list.** With a service picked in Settings → Queries, the tabs of the domains the
  service does not analyse are marked "off", and a line under the tabs says so and opens Services; in the area scope
  the line names the services that skip the domain. The queries stay editable. The domain ticks of a service in
  Services show how many query rows the service reads in each domain.
- **Regression against the baseline in the verdict.** A report compares its p95, error rate and load reached with
  the baseline run of the analysis at a comparable load (the target step of both runs, else the lower ceiling; the
  hold of a soak test; the spike level) and adds the checks `regression_latency`, `regression_errors` and
  `regression_capacity`. Six SLA keys set the risk and failure thresholds (`sla.regression_*`, per area and service;
  the example sets risk at +10 % p95, +0.5 pp errors and −10 % load, failure off): over a risk threshold `pass`
  becomes `risk`, over a failure threshold the verdict becomes `fail`; without thresholds or without a comparable
  baseline the verdict of the test type does not change. The comparison is stored with the report
  (`sla_details.regression`, `deterministic_sla.regression`), the SLA table shows the deltas in % and pp, a line
  under the verdict names the baseline and how it was chosen with a link to the comparison, the Confluence page
  repeats both, and the settings have a "Regression against the baseline" group
  ([docs/ai-analysis.md](docs/ai-analysis.md#regression-against-the-baseline)).

### Changed

- **The test window hint of the New report form looks back 30 days.** When the last six hours hold fewer than
  three load stretches, the form searches the load tool metrics of the last 30 days with a five-minute step and
  restores the exact bounds of every older stretch with a fine query, so a test of last week is offered too.
  Finished LoadLens runs of the same 30 days are listed. `GET /lt_windows` answers with `days` and `since`; a
  failed 30-day search keeps the windows of the last hours and says so under the chips.
- **The default baseline is `pinned`.** Without release marks it is the previous successful run as before; a
  pinned baseline or an accepted run of an earlier release is now taken first. A `mode` set in `settings.py` stays.
- **Accepted runs and baselines are protected from deletion.** `DELETE /runs/<run_name>` answers 409
  `release_protected` for them unless `force=1` (the archive asks a second time); deleting load test runs together
  with their reports keeps such reports and lists them in `reports_failed`.

- **Shared PNG helpers of the Confluence pages.** `figure` and `png` (matplotlib) moved from the forecast page into
  `loadlens_app/confluence_export.py`, which also reads the published report pages of several runs at once
  (`load_publications`); the forecast page uses them unchanged.
- **Error rate per step in the step table.** Every step row of a report stores the error rate of the step
  (`error_rate`, the plateau p95 of `sla.error_rate_query`, the statistic of the SLA check) next to its p95, so a
  step can be compared with the same step of another run.
- **Phase checks have their titles in Confluence.** The SLA table of a Confluence page names
  `phase_completeness` and `phase_duration` like the report page instead of showing the raw names.

- **Compare page: the change column is A against B.** `trend_pct` of `/compare_summary` and
  `/compare_metric_summary` is now the change of the assessed run A relative to the baseline B, `(A − B) / |B|`,
  so "better" and "worse" rate A; before, they rated B against A. The rows also carry `direction`, taken from the
  role of the query (the words of the label remain the fallback).
- **The top bar stays on screen while a page scrolls,** on every width (before, only below 1024px). The chart
  tabs of a report and the section menu of Settings stick below it, and links to an anchor stop below it.
- **A source runs one language, not a domain.** A pack in another language merges into a domain when its rows read
  another source; "Replace only" became "Language conflict" and appears only when rows in another language read the
  same source.
- **Step and stress verdicts are decided on the target step.** CPU and memory are checked on the first step that
  reaches `target_rps` with the latency and error thresholds held, not on the last such step: the steps above the
  target are headroom and never lower the verdict, so a run that goes further gets the same or a better one. No such
  step is `fail`, as is a target step only after-drop steps would reach. Without `target_rps` CPU or memory over the
  threshold is `risk` instead of `fail`. Reports show two bands on the charts and two tags in the step row (target
  step and max), the SLA summary names both, and the model is told that breaches above the target step are headroom
  (`deterministic_sla.target_window`, `headroom`, `sla_step.role`). Reports saved before are shown as they were.
- **Domain prompts no longer list their sections by hand.** The lists named metrics that no pack collects (for
  example `kafka_consumer_fetch_rate` or disk by PVC); the generated domain metrics replace them. A stored copy of an
  old default prompt is still read as the default; a prompt edited in "Settings → Prompts" keeps its own text,
  including the old list, until it is reset.
- **Deleting a service deletes its finished load test runs.** Runs still going stay until they end.
- **The archive finds a run started from LoadLens by its report id.** The "started from LoadLens" mark and "Run
  again" stay on a renamed report, and of two runs with one name the newest is offered.
- **The report window of every run follows the load.** Before, only runs with CI instances were narrowed, and only
  by RPS; the others reported the whole process time with JVM warm-up and a slow stop. Now the report covers the
  span where the users (`lt.vus`) or the total RPS (`lt.rps_total`) are above zero inside the process time, so a
  tail where the stand is down and requests hang until their timeout keeps its users. The run message names the
  signal ([docs/running-load-tests.md](docs/running-load-tests.md#a-run-and-its-instances)).
- **One automatic step detector instead of sensitivity presets.** The detector takes its confirmation window from
  the data step of the RPS series and the minimum step jump from the noise of its plateaus (at least 2.5 % and
  1 RPS), so minute and 5-minute buckets are no longer smoothed into each other. Before, the presets fixed the jump
  in percent: a run of +10 % steps at 3-minute buckets under "strict" (10 %) got two steps instead of ten. A series
  with no plateaus (two samples per load level) falls back to equal intervals, and RPS that swings between two
  levels is no longer a stable step. The step row of the report and of the Confluence page says how the steps were
  found (data step, plateau noise, threshold) or why the report shows equal intervals; the plateau of a step leaves
  out the ramp samples that open it ([docs/configuration.md](docs/configuration.md)).

### Removed

- **`sla.step_detection_preset` and the step detector tuning keys** (`step_detection_resample_sec`,
  `step_detection_smooth_sec`, `step_confirm_hold_sec`, `step_min_step_delta_rps`, `step_min_step_delta_pct`,
  `step_max_cv`, `step_max_slope_rps_per_min`, `step_max_within_step_drop_pct`, `step_drop_hold_sec`) and the
  "Step detector sensitivity" field of the SLA settings. They are ignored with a warning in the log and dropped when
  the SLA block is saved; `min_stable_minutes` ("Min. step duration") stays.

### Fixed

- Update pypdf to 6.21.4 to fix denial-of-service vulnerabilities in malformed PDF parsing used by context imports.
- Loki error totals and sampled lines stay within the requested test window, including partial first/last
  intervals; adjacent sample intervals no longer repeat a boundary event.
- Scenario runs whose phases are all excluded or idle return `no_data` instead of assessing those phases again
  through a whole-window fallback. Scenario references stay within the selected service.
- Regression thresholds use unrounded deltas, so a small real breach is not hidden by display rounding.
- Release Confluence publications are isolated by project area; publishing the same release name in another
  area no longer overwrites the first area's page. Existing unscoped publication records are preserved.
- A failed lookup of release/baseline protection keeps reports intact during load-run deletion and returns
  `reports_failed` with `report_lookup_failed` instead of treating them as unprotected.
- Clearing a manual acceptance checkbox restores automatic selection of the newest release/service/type report.
  An omitted `accepted` field continues to preserve the manual selection.
- Changing a run picker invalidates the displayed comparison and its AI/publication actions; late responses
  from a previous pair cannot restore stale results.
- The release-publication information route is `GET /release_publications/<name>`, separate from the unchanged
  publish POST, so a release name ending in `/confluence` opens its HTML page correctly.
- Release preparation instructions include new files in the review/staging step and use DCO signoff.
- Release notes are emitted as UTF-8 when redirected, including on Windows with a legacy console encoding.
- Documentation, UI hints and tests use synthetic project names, log records and example addresses. The scenario
  alignment fixture no longer depends on a private data corpus. Release screenshots use demo issue keys, and the
  favicon asset uses the project logo.
- **Grafana charts with a multi-value variable.** A chart URL that repeats a variable
  (`var-KAFKA_TOPICS=a&var-KAFKA_TOPICS=b`) reached Grafana with the last value only: adding the render `timeout`
  rebuilt the query through a dictionary. The URL is now sent as configured, with `timeout` appended when missing.
- **Reports without AI analysis are full reports and publish to Confluence.** Before, `use_llm: false` (the "No AI
  analysis" button, which said "Collect metrics only", or an exhausted daily analysis budget) saved the metrics
  only: the report page said the report was not ready, the archive had no verdict, the forecast did not open and
  Confluence refused the run. Now such a report stores its final row with everything computed without the model:
  the SLA verdict and checks, the load steps, the peak RPS with its peak and degradation times or the test-type
  figures, and the run context the forecast reads. The report page marks the verdict "no AI analysis" and shows the
  figures in place of the analysis tabs; the Confluence page heads with the verdict and has no analysis section; the
  verdict can be overridden. Runs built that way before keep their metrics only and publish with their charts and
  the engineer summary ([docs/ai-analysis.md](docs/ai-analysis.md#other-model-calls-and-limits)).
- The capacity forecast no longer shows the steps above the report maximum as green. Before, it judged a step only by
  response time stalls inside it, so a step that was evenly slow past the SLA got "no stalls", entered the model and
  painted the load scale green. Now the forecast reads the maximum of the report (the last step on which the SLA
  latency and error thresholds hold): later steps are red, stay out of the model and the pod plan, and the first of
  them sets the load limit (`limit.kind: sla`). The Confluence page and the AI explanation of the limit follow it
  ([docs/api.md](docs/api.md#capacity-forecast)).
- A finished load test run no longer keeps its name taken. Before, any run of the service with that name, even a
  failed or cancelled one, refused the name forever, and deleting the report from the archive did not free it. Now
  only a running test or an existing report holds a name.
- Saved data sources and domain bindings replace those of `settings.py` instead of being merged into them field by
  field. Before, a binding switched to a direct Prometheus kept the Grafana datasource of `settings.py`, so the next
  save of the bindings card failed with "field datasource_uid is not used for type prometheus", and a source or a
  binding deleted on the page came back after a restart.
- The time axis and the tooltip of the report charts follow the clock of the report, like the step row and the
  markers; in a browser outside the report's time zone the ticks no longer disagree with the marker labels.
- The AI analysis of a run is saved when a PromQL query divides by zero. `+Inf`, `-Inf` and `NaN` samples are
  skipped as missing points, and non-finite numbers in the stored JSON become `null`; before, PostgreSQL rejected
  the results of every domain with `Token "Infinity" is invalid`.

## [1.0.0] - 2026-10-21

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
- **Interface.** An instrument-style layout: a top bar with text navigation, tables and charts on the page
  without cards, statuses as a colored square with text. The report opens with the verdict, one sentence with
  the figures behind it, key figures against their thresholds and a test timeline with the SLA window; the
  dashboard shows a status line, the latest run against its SLA and its load profile; the archive has SLA and
  p95 columns. Golos Text and JetBrains Mono fonts, equal light and dark themes that follow the system, a
  configurable accent color, no external CDNs.
- **About.** `GET /about` with the version, build and license, an "About" section in the settings with the
  third-party notices, and the version next to the logo.
- **API.** The same JSON API as the web UI with API tokens; `GET /runs?include=sla` adds the SLA result of every
  run, `GET /dashboard_data` returns the SLA checks and the load profile of the latest run.

[Unreleased]: https://github.com/Zagato27/LoadLens/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/Zagato27/LoadLens/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/Zagato27/LoadLens/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/Zagato27/LoadLens/releases/tag/v1.0.0
