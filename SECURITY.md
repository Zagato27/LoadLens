# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting on this
repository: the "Security" tab, then "Report a vulnerability". The report reaches the maintainer only, and the
discussion stays private until a fix is released.

Include what you can: the version (`GET /about` or Settings → About), how LoadLens is deployed (Docker Compose,
the demo stack, a source checkout), the steps to reproduce, and the impact as you see it. Logs and screenshots
help, with secrets and internal hostnames removed.

What to expect:

- an acknowledgement within 7 days;
- an assessment and, for a confirmed problem, a fix in a patch release of the latest minor version, with credit
  to the reporter in the changelog unless you prefer otherwise;
- a GitHub Security Advisory published together with the fix. Please keep the report private until then; if you
  have not heard back within 14 days, open a short public issue saying only that a private report is waiting.

## Supported versions

| Version | Supported |
|---|---|
| The latest minor release (`1.x`, latest patch) | Yes |
| Older minor releases | No: please upgrade |

## Scope

In scope: authentication and authorization bypass, CSRF or CSP weaknesses, secrets leaking through pages, the
API, logs or the audit log, server-side request forgery through configurable URLs (data sources, Confluence,
LLM endpoints, load generators), injection in queries or commands built from user input, and the Docker image.

Out of scope: anything that requires an administrator account (administrators configure connections and
commands on purpose), denial of service through the limits an administrator chose, and findings in third-party
services LoadLens connects to. Report those upstream.

## Notes for operators

- Never commit `settings.py`, `settings_runtime*.json`, `metrics_config.py`, `metrics_config_runtime.json`,
  `docker/.env` or `.loadlens_secret_key`; they are in `.gitignore` and `.dockerignore`, and Compose mounts them
  at run time.
- LoadLens does not phone home: there is no telemetry, no update check and no external CDN. It connects only to
  the systems you configure.
- Data leaves your installation only to the LLM provider you configure: metric summaries, the description of
  the system under test, documents you import into that description and, if enabled, excerpts of application
  logs. The demo stack uses a local stub instead of a provider.
- Run the application behind HTTPS and set `LOADLENS_COOKIE_SECURE=1`, a stable `LOADLENS_SECRET_KEY`, and
  `LOADLENS_TRUSTED_PROXIES` when a reverse proxy sits in front. Remove `LOADLENS_ADMIN_PASSWORD` from the
  environment after the first sign-in.
- The demo credentials in `docker/demo.env` are public by design; the demo binds to `127.0.0.1` only.
- `pip-audit` runs weekly in CI (`.github/workflows/dependency-audit.yml`); dependency updates with a security
  fix are released as patches.
