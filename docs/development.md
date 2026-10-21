# Development

How to run the tests, check the UI without touching a real installation, keep the two languages in sync, keep
the license headers and ship a release. The contribution rules (DCO, commit style, pull request checklist) are in
the repository root; this page is the technical companion.

- [Setup](#setup)
- [Tests](#tests)
- [Checking the UI in a sandbox](#checking-the-ui-in-a-sandbox)
- [Localization rules](#localization-rules)
- [Prompts](#prompts)
- [Query packs](#query-packs)
- [License headers and third-party notices](#license-headers-and-third-party-notices)
- [CHANGELOG and releases](#changelog-and-releases)
- [Continuous integration](#continuous-integration)
- [Debugging switches](#debugging-switches)

## Setup

```bash
pip install -r requirements-dev.txt          # requirements.txt plus pytest and pip-audit
cp settings.example.py settings.py           # the tests import the settings module
cp metrics_config.example.py metrics_config.py
```

Python 3.12 and Node 22 (for the front-end tests). Never start the application against a real `settings.py` while
developing: `app.py` builds the application at import time and starts the migration thread against the configured
database. Use the test suite or a sandbox copy with the example settings (below).

## Tests

```bash
python -m pytest -q                          # the whole suite, offline
python -m pytest -q tests/test_docs_links.py # one file
```

- Tests never reach external systems: `tests/conftest.py` fakes the database and HTTP, refuses database
  connections outside `127.0.0.1`, disables migrations and the load test watcher, forces the thread job runner and
  sign-in off, and clears the response cache. Tests that need sign-in or the Celery path set their own variables.
- The repository tests of the user store can also run against a real PostgreSQL in a throwaway container:
  `LOADLENS_TEST_PG_DSN=postgresql://postgres:test@127.0.0.1:55432/postgres`.
- Front-end functions are tested under Node without a browser through `tests/js_harness.py` (`run_js(files,
  expression)` loads `static/js/*.js` into a `vm` context with small DOM, storage, cookie and Chart stubs and the
  dictionaries of the chosen locale). The JS tests are skipped without `node` on PATH and fail instead with
  `LOADLENS_REQUIRE_NODE=1`, which CI sets.
- `LOADLENS_I18N_STRICT=1` makes a missing dictionary key raise instead of falling back; the conftest switches it on.
- Tests that import `update_page` must import `loadlens_app` first, and patch both `core.CONFIG_RUNTIME_PATH` and
  `update_page._SETTINGS_RUNTIME_PATH` when settings are involved, or they read the real runtime file.
- Documentation tests: `tests/test_docs_links.py` (links and language of the English docs, settings keys of
  configuration.md), `tests/test_perf_ops_docs.py` (job variables and defaults in the docs match the code),
  `tests/test_license_files.py`, `tests/test_about.py`.

End-to-end scenarios of the job pipeline (`tools/e2e_jobs/s1..s9`, `bench_modes.py`) need Docker, Playwright and
free local ports; they build their own sandboxes and stubs and are run by hand before a release. The header of each
script explains its invocation.

## Checking the UI in a sandbox

The real `settings.py` of a developer machine may point at production systems, so the UI is never run on it.
Instead:

1. Copy the tracked files (`git ls-files -co --exclude-standard`, so ignored secrets are never copied) into a
   temporary directory and write a `settings.py` from `settings.example.py` with every URL replaced by
   `127.0.0.1:9`, `metrics_config.py` from the example and empty runtime JSON files.
2. Start it from a small runner script outside the sandbox with `LOADLENS_AUTH_ENABLED=0` (or
   `LOADLENS_ADMIN_USER` / `LOADLENS_ADMIN_PASSWORD` when sign-in matters) on a free port of `127.0.0.1`.
3. For data, run `timescale/timescaledb:latest-pg16` in a throwaway container bound to `127.0.0.1` and put only the
   `storage.timescale` fields into the sandbox `settings_runtime.json`; `tools/seed_bench_data.py` fills a `bench`
   area with runs, and `python -m loadlens_app.migrations` from the sandbox directory builds the indexes.
4. For model, Confluence and Prometheus answers use the stubs in `tools/stubs/` (`llm_stub.py`, an OpenAI-compatible
   server that answers in the language of the system prompt, `--lang auto|ru|en`; `confluence_stub.py`;
   `prom_stub.py`, synthetic series for any PromQL query). Point the sandbox at them through its runtime file
   (`llm.provider = "openai"`, `llm.openai.api_base_url = "http://127.0.0.1:5090/v1"`, and so on).
5. Drive the pages with Playwright headless and collect console errors. The CSP has no `'unsafe-eval'`, so
   `page.wait_for_function` fails; poll `locator.inner_text()` instead. Templates are cached by the sandbox server:
   restart it after editing one.

`tools/e2e_jobs/s9_ui_language.py --prepare` automates this for two sandboxes (the working tree and `git archive
HEAD`) and compares every page in both languages; `tools/bench_endpoints.py` measures endpoint latencies.

## Localization rules

Interface and server texts live in `locales/<lang>/<namespace>.json` for `ru` and `en` (one file per namespace:
`common`, `nav`, `dashboard`, `report`, `settings`, `api`, `job`, `sla`, `confluence`, ...). The default language
of the product is `ru`; the interface language is a cookie, the report language a setting
([ai-analysis.md](ai-analysis.md#prompts-and-languages)).

- Every key exists in both languages with the same `{placeholders}`; no empty values, no markup in values (pass
  links as parameters), no Cyrillic in `en` (the language name is the one exception). Keys are
  `namespace.snake_case.path` and describe the meaning, not the wording. Plurals: `.one/.few/.many` for `ru`,
  `.one/.other` for `en`, with `{n}`.
- Canonical file format, enforced by a test: `json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)` plus a
  newline, UTF-8 without BOM, LF. Edit the dictionaries with a script, not by hand in a shell.
- Python: `AI.i18n.t(key, **params)`, `plural`, `msg`, `LocalizedError`, `with_locale`; Flask: `current_locale()`,
  Jinja `t()` / `tn()`; JS: `LL.t`, `LL.tn`, `LL.tHtml`, `LL.fmt`. Logs, CLI output and comments are not localized.
- The ratchet: `tools/i18n_extract.py` counts Cyrillic user-facing literals per file (templates, `static/js`, the
  application packages) against `locales/allowlist.json`; `locales/baseline.json` stores the counts and
  `tests/test_i18n_guard.py` fails when a file's count grows. After translating a file run
  `python tools/i18n_extract.py --write-baseline <files>` (lowering only); `--summary` shows the counts, and the
  baseline is 0 for every file today. Literals that must stay Russian (input synonyms, regexes over stored text) go
  to the allowlist with a reason.

## Prompts

Prompts live in `AI/prompts/ru/` and `AI/prompts/en/` with identical file names; `_strings.json` holds the short
glue strings. Change both folders, keep placeholders, JSON keys and rubric criteria identical, write natural
English. `tests/test_prompt_store.py` checks the parity and the absence of Cyrillic in `en`; `tests/test_i18n_prompts.py`
checks that all model calls of a run use one language. Check English quality by hand on saved runs with
`AI/eval.py` when a real provider is available.

## Query packs

Packs are TOML files in `AI/query_packs/packs/`, bundles in `AI/query_packs/bundles.toml`, roles in
`AI/query_packs/roles.py`. After changing a pack regenerate the catalogue and the example configuration; tests
compare both with the packs and lint the PromQL with `tests/promql_lint.py`:

```bash
python -m AI.query_packs docs docs/query-packs.md
python -m AI.query_packs docs --lang en docs/query-packs.en.md
python -m AI.query_packs example
python -m AI.query_packs render jvm-micrometer --var service_label=app
```

The label of a query is the identifier of its series in the database, the SLA and run comparison: never rename a
label between versions; add new queries with new labels, raise `version` and mark them `since = <version>` so the
update dialog offers them. `verified = true` is set only after `docker/dev-stack/check_packs.py` saw every mandatory
query return data on the live stand ([docker/dev-stack](../docker/dev-stack/README.md)).

## License headers and third-party notices

Every project `.py` file and `static/js/*.js` starts with `SPDX-License-Identifier: Apache-2.0` (after a shebang
and an encoding line, before the module docstring; no copyright line, `NOTICE` carries it). Vendored files,
`locales/` and `tests/fixtures/` are out of scope.

```bash
python tools/license_headers.py --check            # list files without the header, exit 1
python tools/license_headers.py --write [paths]    # add it
python tools/third_party_table.py --check          # the dependency table of THIRD_PARTY_NOTICES.md is current
python tools/third_party_table.py --write          # refresh it from requirements.txt and the installed metadata
```

`tests/test_license_files.py` checks the license files, the vendor directory, the notices and the headers; the
dependency audit workflow additionally refuses GPL, AGPL, SSPL and EUPL packages (LGPL is accepted: psycopg2 and
paramiko stay dynamically linked).

## CHANGELOG and releases

`CHANGELOG.md` follows Keep a Changelog with Semantic Versioning: new entries go under `## [Unreleased]` in the
`Added` / `Changed` / `Fixed` / `Removed` groups, written for users, one bullet per change. The version lives in
`loadlens_app/version.py` (`__version__`, `SOURCE_URL`); `GET /about`, the version label in the top bar and the
release tooling read it.

```bash
python tools/release.py --check                 # version.py and CHANGELOG.md agree (also in tests)
python tools/release.py --version 1.1.0         # bump the version, date the changelog, refresh the notices, check headers
python tools/release.py --notes 1.1.0           # the changelog section of a version (release notes)
```

`--version` moves the Unreleased entries into `## [1.1.0] - <date>`, updates the reference links, regenerates the
dependency table and runs the header check; it never runs git. The owner then reviews the diff, commits, tags
`v1.1.0` and pushes the tag. `.github/workflows/release.yml` runs the suite, verifies that the tag names the same
version (`--check --tag`), builds `ghcr.io/<owner>/loadlens:<tag>` and `:latest` for linux/amd64 and linux/arm64 (a tag with a suffix such as
`v1.1.0-rc1` is a pre-release and does not move `latest`) and creates the GitHub Release with the notes. Repository
renames, tag pushes and image publishing are owner actions.

## Continuous integration

| Workflow | When | What |
|---|---|---|
| `tests.yml` | Every push and pull request | pytest on Ubuntu with Python 3.12 and Node 22, `LOADLENS_REQUIRE_NODE=1`, example configuration |
| `dependency-audit.yml` | Changes of `requirements*.txt` or the notices, weekly | `pip-audit`; `pip-licenses` with the copyleft deny list; every requirement named in `THIRD_PARTY_NOTICES.md` |
| `demo.yml` | Changes of the demo files, weekly, by hand | Starts the demo stack, waits for `/healthz`, signs in, checks the demo run in the archive and the stub report built by the job pipeline |
| `links.yml` | Changes of `*.md`, weekly, by hand | lychee over every markdown file with `.github/lychee.toml`: external links only (local and example hosts are skipped; `tests/test_readme.py` checks the relative ones) |
| `release.yml` | Tags `v*` | Tests, image on GHCR for linux/amd64 and linux/arm64 (arm64 under QEMU; both are started once before the push), GitHub Release from the changelog |

## Debugging switches

| Switch | Effect |
|---|---|
| `LOADLENS_STATIC_NO_CACHE=1` | Static files without the one-year browser cache while editing the front end (otherwise links carry `?v=<content hash>`) |
| `LOADLENS_RESPONSE_CACHE=0` | No in-process response cache (`/dashboard_data`, `/jobs`, `/load_tests/active`) |
| `LOADLENS_CSP_REPORT_ONLY=1` | Content-Security-Policy in report-only mode: violations in the browser console instead of blocked requests |
| `LOADLENS_JOB_RUNNER=thread` | Jobs in the web process, no Redis needed; the default without a broker |
| `Server-Timing` header | `app;dur=...`: time inside the application per request; the gunicorn access log ends with the full request time |
| `sla.debug_peak_logging` | Verbose log of the step detector and the peak selection |
| `python -m loadlens_app.migrations --status` | Which migrations ran and which are pending |
