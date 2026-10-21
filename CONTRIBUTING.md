# Contributing to LoadLens

Thank you for taking the time. Bug reports, query packs, documentation, translations and code are all welcome.
Questions go to GitHub Discussions; bugs and feature requests to GitHub Issues (the templates ask for the
version shown at `/about`). This file explains how to set up the project, what the tests and the i18n rules
expect, and how a pull request gets merged.

## Setup

- Python 3.12 and Node 22 (Node runs the front-end tests through `tests/js_harness.py`; without it they are
  skipped locally and fail in CI).
- Install the dependencies and copy the example configuration; the tests never read your real settings:

```bash
pip install -r requirements-dev.txt
cp settings.example.py settings.py
cp metrics_config.example.py metrics_config.py
```

- `settings.py`, `settings_runtime*.json`, `metrics_config.py`, `metrics_config_runtime.json`, `docker/.env` and
  `.loadlens_secret_key` are ignored by git and must never be committed; keep them out of issues and logs too.
- To run the application itself you need a TimescaleDB: `docker compose -f docker/docker-compose.yml up -d`
  after filling `docker/.env` from `docker/env.example`, or `python app.py` against a local database. The demo
  stack (`docker/docker-compose.demo.yml` with `--env-file docker/demo.env`) needs no configuration at all.

## Tests

```bash
python -m pytest -q                         # the whole suite, offline
LOADLENS_REQUIRE_NODE=1 python -m pytest -q # what CI runs: the JS tests fail instead of being skipped
python -m pytest -q tests/test_i18n.py      # one file while you work
```

- Tests are offline: HTTP and the database are faked, and `tests/conftest.py` refuses any database connection
  outside `127.0.0.1`. Do not add tests that need network access or a running service.
- The user-storage tests can additionally run against a real PostgreSQL with `LOADLENS_TEST_PG_DSN`.
- Every behaviour change comes with a test; a fix for a bug starts with a test that fails before the fix.
- `tests/promql_lint.py` checks the PromQL of the query packs; `python -m AI.query_packs docs docs/query-packs.md`
  and `python -m AI.query_packs example` regenerate the generated files that the tests compare with the packs.

## Localization (i18n)

The UI and the server texts are bilingual, and the tests enforce it:

- Dictionaries live in `locales/<lang>/<namespace>.json` for `ru` and `en`. Every key must exist in both
  languages with the same `{placeholders}`; values must not be empty and must not contain `<` or `>` (pass
  markup as parameters). No Cyrillic in `en` values, except the language name «Русский».
- The canonical file format is enforced by a test: `json.dumps(obj, ensure_ascii=False, indent=2,
  sort_keys=True) + "\n"`, UTF-8 without BOM, LF. Update dictionaries with a small script (load, change, dump)
  rather than by hand.
- Keys are `namespace.section.name` in snake_case and describe the meaning, not the wording
  (`dashboard.kpi.runs_note`, not `dashboard.za_30_dney`). Shared keys go to `common.*` only when two or more
  pages use them in the same meaning. Plurals use the suffixes `.one/.few/.many` (ru) and `.one/.other` (en).
- Templates use `{{ t('ns.key') }}` and `{{ tn('ns.key', n) }}`; JavaScript uses `LL.t()` / `LL.tn()`;
  server code uses `AI.i18n.t()` / `msg()` / `LocalizedError`. Logs, CLI output and comments are not localized.
- Tests run in strict mode (`LOADLENS_I18N_STRICT=1`): a missing key raises instead of falling back.
- `tools/i18n_extract.py` is a ratchet: it counts Russian literals left in templates, `static/js` and the
  server modules against `locales/baseline.json`, and `tests/test_i18n_guard.py` fails when a file's count
  grows. After moving literals into the dictionaries run
  `python tools/i18n_extract.py --write-baseline <files>` to lower the entries of the files you changed
  (`--summary` shows the counts; the goal is 0 everywhere).

## License headers

Every project `.py` file and every `static/js/*.js` file starts with `# SPDX-License-Identifier: Apache-2.0`
(`//` in JavaScript). `python tools/license_headers.py --check` lists the files without it;
`python tools/license_headers.py --write <paths>` adds it. Vendored files in `static/vendor/`, dictionaries and
test fixtures are out of scope. Do not add copyright lines to files: `NOTICE` carries the copyright.

## Developer Certificate of Origin

Contributions are accepted under the [Developer Certificate of Origin 1.1](https://developercertificate.org/):
by signing off a commit you certify that you wrote the change or have the right to submit it under the
Apache License 2.0. Sign every commit with `git commit -s`, which adds a line

```
Signed-off-by: Your Name <your@email.example>
```

with the name and address of your git configuration. A pull request whose commits are not signed off is not
merged; `git commit --amend -s` or `git rebase --signoff` fixes existing commits. There is no CLA.

## Commits and pull requests

- Commit messages in English, imperative mood, one topic per commit: `Add a query pack for the JMeter
  Prometheus listener`, not `fixes` or `WIP`. The subject fits in 72 characters; the body says why, not what.
- A pull request addresses one issue or one change. Describe what changes for the user, link the issue, and
  fill in the checklist of the template:
  - `python -m pytest -q` passes (with Node 22 for the JS tests);
  - new or changed UI and server texts exist in both `locales/ru` and `locales/en`, and
    `python tools/i18n_extract.py --summary` did not grow;
  - new `.py` and `.js` files carry the SPDX header;
  - `CHANGELOG.md` has an entry under `[Unreleased]` for every user-visible change;
  - documentation is updated (`docs/*.md` in English; `README.ru.md` when the Russian README is affected);
  - no secrets, hostnames or internal names in code, fixtures, screenshots or tests;
  - every commit is signed off (DCO).
- Keep the existing line endings of a file you edit; new files are UTF-8 without BOM with LF line endings.
- Code, comments and docstrings are in English. User-visible Russian text is edited only in `locales/ru`.

## Where things are

| Path | What it is |
|---|---|
| `loadlens_app/` | Flask application: blueprints, auth, jobs, load tests, i18n layer |
| `AI/` | Report pipeline, providers, prompts (`AI/prompts/<lang>/`), verdicts, query packs (`AI/query_packs/packs/*.toml`); never imports `loadlens_app` |
| `data_collectors/`, `confluence_manager/` | Metric sources and the Confluence export |
| `templates/`, `static/js/`, `locales/` | Pages, scripts and dictionaries |
| `docs/*.md` | User documentation in English (index: `docs/README.md`) |
| `plans/` | Not in the repository: the maintainer's design notes (Russian). Code comments that mention `plans/...` refer to them; ask in an issue or discussion if you need the background of a decision |
| `docker/` | Compose files, the Dockerfile, the demo stack and a dev stack with live exporters for query packs |
| `tools/` | Release, license headers, i18n ratchet, benchmarks, stubs for the LLM, Prometheus and Confluence |

A good first contribution is a query pack for a stack you run: packs are TOML files with checked queries,
roles and units; `docs/query-packs.md` describes the contract, and `docker/dev-stack` runs the queries against
real exporters. Query labels are identifiers of stored series and SLA references: never rename a label between
pack versions, add new queries with new labels, mark them `since = "<version>"` and bump `version`.

## Releases

Releases are cut by the maintainer:

1. `python tools/release.py --version X.Y.Z` bumps `loadlens_app/version.py`, turns the `[Unreleased]` section
   of `CHANGELOG.md` into `[X.Y.Z] - <date>`, refreshes the Python dependency table in
   `THIRD_PARTY_NOTICES.md` and checks the license headers. The script edits files only.
2. Review the diff, commit, `git tag vX.Y.Z`, `git push origin vX.Y.Z`.
3. `.github/workflows/release.yml` runs the tests, builds `ghcr.io/<owner>/loadlens:vX.Y.Z` (and `:latest` for a
   tag without a suffix), and creates the GitHub Release with the changelog section as its notes.

`python tools/release.py --check` (run by the tests and by CI) verifies that the version and the changelog agree.
Versions follow Semantic Versioning: a breaking change in the configuration, the API or the stored data is a
major version.
