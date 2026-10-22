# REST API

Everything the web UI does goes through the same JSON API, so scripts and CI pipelines can create reports, start
load tests, read verdicts and publish pages. This page lists the endpoints by area; the access rule of every route
is one table in `loadlens_app/auth/policy.py`, and a test refuses a route without a rule.

- [Authentication and conventions](#authentication-and-conventions)
- [About and health](#about-and-health)
- [Reports and jobs](#reports-and-jobs)
- [Runs and verdicts](#runs-and-verdicts)
- [Comparison and series](#comparison-and-series)
- [Capacity forecast](#capacity-forecast)
- [Load tests (engineer)](#load-tests-engineer)
- [Settings (administrator)](#settings-administrator)
- [Projects, prompts and services](#projects-prompts-and-services)
- [Users, tokens and sessions](#users-tokens-and-sessions)

## Authentication and conventions

- **From scripts and CI**: the header `Authorization: Bearer <token>`. Tokens are created in "Profile and API
  tokens", shown once, stored as hashes, valid for up to 365 days; the role of a token is at most the role of its
  owner, Viewer by default. A token cannot manage users or tokens, and settings, connections, projects, prompts,
  demo data and run deletion are available only from a browser session even with the administrator role
  (`403` with `code: session_required`).
- **From the browser**: the session cookie plus the header `X-CSRF-Token` for `POST`, `PUT`, `PATCH` and `DELETE`
  (`GET /auth/me` returns the token; the UI adds it itself).
- **Errors**: `401 {"error", "code": "unauthorized"}` without a login, `403 {"error", "code": "forbidden",
  "required_role"}` for a missing role, `403 csrf_failed` for a stale page, `503 {"code": "storage_unavailable"}`
  while the database is down, and a generic message with a code for unexpected errors (details in the application
  log).
- **Roles**: Viewer reads dashboards, reports, comparisons and forecasts; Engineer also creates reports, starts and
  stops load tests, publishes to Confluence, writes the engineer summary, votes on findings, renames runs, overrides
  verdicts and asks for forecast explanations; Administrator also edits settings, projects, prompts, users and tokens
  and deletes runs.
- **Verdicts** are codes: `pass`, `risk`, `fail`, `no_data`. Inputs also accept the legacy labels.
- **Language**: messages and errors come in the interface language of the request (cookie `loadlens.lang`, `ru`
  or `en`; `Accept-Language` is not read); responses that depend on it carry `Vary: Cookie`.
- **Project area**: list endpoints answer for the active area (cookie set by `POST /project_area`; `GET
  /project_areas`, `GET /current_project_area`); write endpoints accept `project_area` in the body where it matters.
- **Limits** of the `limits` section (report window, parallel jobs, daily AI budgets, active load tests) answer
  `429` with `code: limit_exceeded` and a message in the interface language.

## About and health

| Endpoint | Access | Returns |
|---|---|---|
| `GET /healthz` | public | `{"status": "ok"}`; the Docker health check |
| `GET /about` | public, works while the database is down | `name`, `version`, `build`, `license`, `source_url`, `docs_url`; no configuration values. Quote `version` and `build` in bug reports |
| `GET /about/third-party` | public | `THIRD_PARTY_NOTICES.md` as text |
| `GET /demo/status` | signed in | Whether the instance runs with the demo seed and the demo run exists |

## Reports and jobs

| Endpoint | Body or parameters | Notes |
|---|---|---|
| `POST /create_report` | `start`, `end` (ISO 8601 with offset), `service`, `project_area?`, `test_type?` (`step`, `soak`, `spike`, `stress`), `use_llm?` (default true), `save_to_db?`, `web_only?`, `run_name?` | Returns `job_id` and the assigned `run_name`. Engineer |
| `GET /job_status/<job_id>` | | Phase, progress, message, error and the link to the result |
| `GET /jobs` | `status=running,error`, `limit` | Recent jobs; cached 3 s |
| `POST /jobs/dismiss` | `{"job_ids": [...]}` | Hides finished jobs from the list; running ones are returned in `skipped_running`. Engineer |
| `GET /llm_reports` | `run_name` | Domain and overall blocks of a report with the SLA details, the step table, the effective verdict and `manual_verdict` |
| `GET /llm_context` | `run_name` | The JSON context that was sent to the model (large, kept off `/llm_reports`) |
| `GET` / `PUT /llm_feedback` | | Votes on findings |
| `GET` / `POST /engineer_summary` | | The engineer summary (sanitized HTML: formatting and http(s)/mailto links only) |
| `GET /confluence_publication`, `POST /publish_confluence` | `run_name` | The published page of a report; publication as a background job. Engineer |
| `GET /report_ref/<run_id>` | | Service and run name of a report by id (old runs by name) |
| `POST /demo/seed` | | Writes the synthetic run `demo-checkout-step` in the area `demo`; `409` when it exists. Administrator |

```bash
curl -s -X POST http://localhost:5000/create_report \
  -H "Authorization: Bearer ll_..." -H "Content-Type: application/json" \
  -d '{"service": "checkout", "start": "2026-01-15T10:00:00+03:00", "end": "2026-01-15T11:00:00+03:00", "test_type": "step"}'
# {"job_id": "...", "run_name": "checkout-2026-01-15-10-00"}
curl -s -H "Authorization: Bearer ll_..." http://localhost:5000/job_status/<job_id>
```

## Runs and verdicts

| Endpoint | Parameters | Notes |
|---|---|---|
| `GET /runs` | `q`, `service`, `verdict`, `test_type`, `offset`, `limit`, `sort`, `dir`, `include` | Runs of the active area; the total is in the header `X-Total-Count`. Sorting by verdict orders by severity: `pass`, `risk`, `fail`, `no_data`. `include=sla` adds the SLA summary of every run (below) |
| `PATCH /runs/<run_name>` | new name | Rename. Engineer |
| `DELETE /runs/<run_name>` | | Deletes all data of the run. Administrator, browser session |
| `GET /runs/<run_name>/verdict` | | The automatic and the effective verdict and the history of manual changes |
| `PUT /runs/<run_name>/verdict` | `{"verdict": "fail", "reason": "..."}` | Manual verdict; the reason is mandatory. Engineer |
| `DELETE /runs/<run_name>/verdict` | | Back to the automatic verdict. Engineer |
| `GET /dashboard_data` | | The dashboard summary; verdict counters are keyed by code; `last_run` carries `checks` and `profile` (below); cached 10 s |
| `GET /services` | | Services and domains of the active area |

`/runs`, `/dashboard_data` and `/llm_reports` return the effective verdict plus `manual_verdict` (`verdict`,
`reason`, `author`, `changed_at`) when one was set. Manual changes are written to the audit log as `report.verdict`
and `report.verdict_reset`.

`include` of `GET /runs` is a comma-separated list; the only known value is `sla`, unknown values are ignored, and
without it the response is unchanged. With `include=sla` every item gets `sla`: `{"passed", "total", "p95_ms",
"p95_passed"}` from the SLA checks of the run's final report, or `null` when no check was evaluated. `total` counts
the checks with a result and `passed` those that passed; `p95_ms` is the measured value of the `p95_latency` check
and `p95_passed` its result, both `null` without such a check. The archive reads it for its SLA and p95 columns.

```bash
curl -s -H "Authorization: Bearer ll_..." "http://localhost:5000/runs?limit=1&include=sla"
# [{"run_name": "demo-checkout-step", "verdict": "risk", ..., "sla": {"passed": 2, "total": 3, "p95_ms": 450.0, "p95_passed": false}}]
```

`last_run` of `GET /dashboard_data` (`null` while there are no reports) also carries:

- `checks`: the SLA checks of the final report, `[{"name", "title", "threshold", "actual", "passed"}]`; `title` is
  the check name in the interface language, `passed` is `true`, `false` or `null` (not evaluated).
- `profile`: the load profile, `null` when the report has no step table. `steps` are the rows of the step table
  (`index`, `label`, `start_iso`, `end_iso`, `plateau_start_iso`, `plateau_end_iso`, `rps`, `selected` for the step
  the SLA was checked on, `after_drop`, `dip`), `window` is `{"start", "end"}` of the SLA window (ISO times) or
  `null`, `markers` are the degradation, RPS drop, hold and spike marks of the report charts (`{"kind", "t", "to",
  "label", "tone", "role"}`, times in epoch milliseconds, labels in the interface language) and `target` is the
  `target_rps` threshold or `null`.

## Comparison and series

| Endpoint | Parameters |
|---|---|
| `GET /compare_summary` | `run_a`, `run_b`, `domain`, `agg` (`p95`, `avg`, `max`): verdicts, findings and metrics per domain |
| `GET /compare_metric_summary` | `run_a`, `run_b`, `query_label`, `agg`: one series of both runs |
| `GET /compare_baseline` | `run_name`, `mode=previous_success|previous`: the baseline run of the same service, type and area |
| `GET /domains_schema`, `GET /run_series`, `GET /compare_series` | Domain schema and time series for the charts |

## Capacity forecast

| Endpoint | Notes |
|---|---|
| `GET /forecast_reports` | Reports a forecast can be built for |
| `GET /forecast/<run_id>/status` | Whether the forecast is possible for this report |
| `GET /forecast/<run_id>?target_rps=&headroom_pct=&cpu_ceiling_pct=` | The USL limit, the safe maximum, the steps and the pod plan |
| `GET` / `POST /forecast/<run_id>/explanation` | The model's explanation of the limit (cached). Engineer for `POST` |
| `GET` / `POST /forecast/<run_id>/confluence` | Publication of the forecast page (background job). Engineer for `POST` |

## Load tests (engineer)

| Endpoint | Body or parameters |
|---|---|
| `GET /load_tests/options` | Generators, tools and limits, without credentials |
| `GET /load_tests/scripts?generator=&tool=` | Scripts of a tool on a generator (SSH, Docker, Kubernetes ConfigMap) |
| `GET /load_tests/hints?service=&tool=` | Metrics output flags and variables for the service's store |
| `POST /load_tests/preview` | Validation of the form and the assembled commands, nothing is started |
| `POST /load_tests` | `service`, `project_area?`, `test_type?`, `run_name?`, `max_duration_min?`, `auto_report?`, `use_llm?`, `instances: [{name, generator, tool, script, params, env, start_delay_sec}]`; a CI instance has `ci_ref`, `ci_params`, `ci_extra` instead of tool, script, params and env. Returns `run_id`, `run_name`, `run_tag`, `warnings`, `url` |
| `GET /load_tests/runs?active=1&limit=`, `GET /load_tests/runs/<run_id>` | Runs of the active area; one run with its instances and `report_url` once the report exists |
| `GET /load_tests/active?limit=5&live=1` | Running runs for the dashboard: `total`, state, instance progress, `can_stop` and, with `live=1`, RPS and error ratio of the last 15 minutes |
| `POST /load_tests/runs/<run_id>/stop` | `{"index": N}` to stop one instance, empty for the whole run |
| `GET /load_tests/runs/<run_id>/logs?index=&bytes=` | Tail of the instance log |
| `GET /load_tests/runs/<run_id>/live?window=&since=&mode=` | Live series by role, per-instance RPS, events, SLA lines, `fresh_at`, `step_sec` |
| `GET /lt_windows?service=&hours=6` | Recent load windows of a service for the New report form (`hours` 0.25 to 24) |
| `GET /load_tests/presets?service=`, `POST /load_tests/presets`, `DELETE /load_tests/presets/<id>` | Presets of the run form (`name`, `service`, `spec`); variable values are not stored |
| `GET /load_tests/ci_params?generator=&ref=`, `GET /load_tests/ci_refs?generator=&search=` | Pipeline parameters and branches of a CI generator |
| `POST /load_tests/ci/<run_id>/<index>/events` | Without a login, with the instance secret in `X-LoadLens-Token`: `{"event": "started"}` or `{"event": "finished", "exit_code": N, "message": "..."}` |

```bash
curl -s -H "Authorization: Bearer $LOADLENS_TOKEN" -H "Content-Type: application/json" \
  -d '{"service":"checkout","test_type":"step","run_name":"release-2.5",
       "instances":[{"name":"rest","generator":"gen-1","tool":"k6","script":"checkout.js",
                     "params":"--out experimental-prometheus-rw",
                     "env":"K6_PROMETHEUS_RW_SERVER_URL=http://prometheus:9090/api/v1/write"}]}' \
  https://loadlens.example.com/load_tests
```

## Settings (administrator)

Browser session only; a token with the administrator role is refused.

| Endpoint | Notes |
|---|---|
| `GET` / `POST /config` | The configuration of the open scope (`area`, `service`); secrets are returned as `***` and restored on save; `POST` answers with `warnings` for unknown keys and refuses the `auth` section |
| `POST /config/test_connection` | Tests a connection with the form values |
| `GET /config/grafana_datasources?source=<id>` | Data sources of a Grafana connection |
| `POST /config/llm_models` | Models of the provider |
| `POST /config/query_preview` | Runs one metric query: `domain`, `lang`, `query`, `label_keys`, `start?`, `end?`, `label?` (the row: its saved source applies), `binding?` (an unsaved source of the row; `null` for the domain binding) |
| `POST /config/system_context/import` | Multipart `file`, `area?`, `service?`, `mode` (`fill`, `replace`), `current?`; or JSON with `text` or `confluence_url`. Returns `context`, `changes`, `source`, `warnings`, `provider`; nothing is saved |
| `GET /config/query_packs?area=&service=` | The pack catalogue with compatibility and installed versions; a pack lists `domains`, `max_instances` and `targets` (compatibility and `instances` per domain; an installation carries its `binding` and `source`) |
| `POST /config/query_packs/detect` | Metric names found at the sources and the share of needed metrics per pack and bundle; `sources` are the extra sources read (catalogue Prometheus sources, sources of rows), and a pack that fits one of them better has `alternative` with its `binding` |
| `POST /config/query_packs/preview`, `.../check`, `.../apply` | Body: `pack_id`, `domain` (required for a pack whose domain is picked), `variables`, `include_indexes`, `binding` (the source of the pack rows; `null` for the domain binding, absent to keep the rows' sources), `mode` (`merge`, `replace`), `fill_sla`, `overwrite_sla`, `area?`, `service?`; `check` also takes `start` and `end`. `apply` returns `previous` for undo and `instance` of a multi-installation pack |
| `POST /config/query_packs/summed_rps` | `area?`, `service?`, `use_for_sla?`: the row that sums the total RPS of several tools |

## Projects, prompts and services

| Endpoint | Notes |
|---|---|
| `GET` / `POST /projects`, `PATCH` / `DELETE /projects/<id>` | Project areas |
| `POST /projects/<id>/services/<sid>/move` | Moves a service to another area |
| `DELETE /projects/<id>/overrides/<section>`, `DELETE /projects/<id>/services/<sid>/overrides/<section>` | Removes an override of an area or a service |
| `GET` / `POST /prompts`, `GET /prompts/defaults?lang=ru|en`, `GET /prompts/history` | Prompt overrides with their sources and language, the defaults, the version history |
| `POST` / `DELETE /service_meta`, `DELETE /service` | Service metadata and removal of a service |
| `GET /project_areas`, `GET /current_project_area`, `POST /project_area` | Areas, the active one, and the cookie that selects it |

## Users, tokens and sessions

| Endpoint | Notes |
|---|---|
| `GET` / `POST /login`, `POST /logout`, `GET /auth/me` | Sign-in form and session state with the CSRF token |
| `POST /auth/password` | Change your own password |
| `GET` / `POST /auth/tokens`, `DELETE /auth/tokens/<id>` | Personal API tokens (browser session only) |
| `GET` / `POST /auth/users`, `PATCH` / `DELETE /auth/users/<id>`, `POST /auth/users/<id>/password`, `POST /auth/users/<id>/unlock` | User management. Administrator |
| `GET /auth/audit` | The audit log: logins and failures, user and token management, settings and project changes, report starts, publications, deletions, load test actions. Administrator |
| `GET /recovery`, `GET /recovery/status`, `POST /recovery/storage/check`, `POST /recovery/storage/save` | Recovery mode while the database is down: the recovery administrator checks and saves the `storage.timescale` connection |

Command-line administration without the API: `python -m loadlens_app.auth create-user <login> --role admin`,
`set-password <login>`, `set-role <login> <role>`, `unlock <login>`, `list-users`, `hash-password`.
