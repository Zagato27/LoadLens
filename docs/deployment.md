# Deployment

A production installation of LoadLens is four containers from `docker/docker-compose.yml`: `app` (gunicorn with
the web UI and the API), `timescaledb`, `redis` and `celery_worker`, which builds the reports and the Confluence
publications. The image is built from the repository; tagged releases also publish it as
`ghcr.io/<owner>/loadlens:<version>` (see [development.md](development.md#changelog-and-releases)). The
five-minute demo is a separate stack and is described in [getting-started.md](getting-started.md#try-the-demo-first);
the Russian original of this page is [docker/DOCKER_SETUP.md](../docker/DOCKER_SETUP.md).

- [Files to prepare](#files-to-prepare)
- [Environment file](#environment-file)
- [Start and first login](#start-and-first-login)
- [The database](#the-database)
- [Redis and the Celery worker](#redis-and-the-celery-worker)
- [Health checks and migrations at start](#health-checks-and-migrations-at-start)
- [Secrets and the image](#secrets-and-the-image)
- [Updates](#updates)
- [External database and other topologies](#external-database-and-other-topologies)
- [Database tables](#database-tables)
- [Troubleshooting](#troubleshooting)

## Files to prepare

```bash
cp docker/env.example docker/.env
cp settings.example.py settings.py
[ -f metrics_config.py ] || cp metrics_config.example.py metrics_config.py
[ -f settings_runtime.json ] || echo '{}' > settings_runtime.json
[ -f metrics_config_runtime.json ] || echo '{}' > metrics_config_runtime.json
```

The four configuration files stay in the repository root and are bind-mounted into `app` and `celery_worker`
(read-only in the worker, the runtime JSON files writable in `app`); a missing file stops `docker compose up` with
`bind source path does not exist` instead of silently becoming a directory. `AI/GigaChat/` is mounted as well for the
GigaChat certificates, and `data_collectors/temporary_files` is a shared volume for rendered charts.

In `settings.py` set `storage.timescale`: `host` `timescaledb` (the Compose service name), `port` `5432`, `dbname`,
`user` and `password` as in `docker/.env` (the superuser `POSTGRES_USER`, or better the unprivileged `APP_DB_USER`),
`schema` `public`. The remaining sections can be filled in the UI ([configuration.md](configuration.md)).

## Environment file

`docker/.env` is read by Docker Compose for variable interpolation; `LOADLENS_*` and `CELERY_*` are passed to the
application, `CELERY_*`, `LOADLENS_JOB_*` and `LOADLENS_REPORT_*` also to the worker. Every line is commented in
[docker/env.example](../docker/env.example).

| Variable | Meaning |
|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | The database container; the application reads its own credentials from `settings.py` |
| `APP_DB_USER`, `APP_DB_PASSWORD` | Unprivileged role for the application, created by `docker/initdb/03_app_role.sh` on an empty volume; put the same values into `settings.py`. Empty keeps the superuser |
| `APP_PORT`, `TSDB_PORT`, `TSDB_BIND` | Host port of the application (5000); host port and interface of the database (5432, `127.0.0.1`: not reachable from other hosts) |
| `REDIS_PASSWORD` | Mandatory, URL-safe (`openssl rand -hex 32`); Compose refuses to start without it |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Default `redis://:<REDIS_PASSWORD>@redis:6379/0`; set only for an external Redis, with the password in the URL |
| `CELERY_TASK_ALWAYS_EAGER` | `0`: the worker builds the reports; `1`: they run in a thread of `app` |
| `LOADLENS_JOB_RUNNER`, `LOADLENS_JOB_STALE_MIN`, `LOADLENS_JOB_QUEUE_MAX_MIN`, `LOADLENS_REPORT_TASK_LIMIT_SEC` | Where jobs run and when a stale or queued job is marked as interrupted ([configuration.md](configuration.md#environment-variables)) |
| `LOADLENS_RUN_MIGRATIONS` | Data migrations at every start (`1`) |
| `WORKER_CONCURRENCY` | Jobs the worker runs at once (2); raise `WORKER_MEMORY_LIMIT` with it |
| `APP_CPUS`, `APP_MEMORY_LIMIT`, `WORKER_CPUS`, `WORKER_MEMORY_LIMIT`, `REDIS_MEMORY_LIMIT` | Container limits (2 CPU / 2g, 1 CPU / 2g, 512m) |
| `LOADLENS_AUTH_ENABLED`, `LOADLENS_SECRET_KEY`, `LOADLENS_ADMIN_USER`, `LOADLENS_ADMIN_PASSWORD` | Sign-in, the session key (keep it stable across container re-creations) and the first administrator; remove the password after the first login |
| `LOADLENS_RECOVERY_USER`, `LOADLENS_RECOVERY_PASSWORD_HASH` | The recovery administrator; hash from `docker compose exec app python -m loadlens_app.auth hash-password`, in single quotes in `.env` (Compose cuts an unquoted value at the first `$`) |
| `LOADLENS_COOKIE_SECURE`, `LOADLENS_TRUSTED_PROXIES` | Behind an HTTPS reverse proxy: `1` and the number of proxies |

## Start and first login

```bash
cd docker
docker compose up -d --build
docker compose ps
docker logs ltar_app --tail=100
```

Open `http://localhost:5000/` (port `APP_PORT`). The login page creates the first administrator from
`LOADLENS_ADMIN_USER` / `LOADLENS_ADMIN_PASSWORD` on its first visit while there are no users; without them run
`docker compose exec app python -m loadlens_app.auth create-user admin --role admin`. A forgotten password is reset
with `docker compose exec app python -m loadlens_app.auth set-password <login>`. Other users are added in
"Settings → Users" with a temporary password that must be changed at the first login.

The image runs gunicorn (`GUNICORN_CMD_ARGS` in the Dockerfile: one process with 8 threads, timeout 120 s, access
log with the request time in milliseconds) as the unprivileged user `app` (UID 1000). Put it behind an HTTPS reverse
proxy for production and set `LOADLENS_COOKIE_SECURE=1` and `LOADLENS_TRUSTED_PROXIES=1`. Pages are served with a
Content-Security-Policy that allows scripts and images from the same origin only.

## The database

`timescale/timescaledb:latest-pg16` with the volume `tsdb_data`. On the first start of an empty volume the scripts
of `docker/initdb/` run: `01_timescaledb.sql` enables the extension, `02_schema.sql` prepares the tables and indexes
the application would create anyway, and `03_app_role.sh` creates the unprivileged role of `APP_DB_USER`: no
superuser, CREATEDB, CREATEROLE or REPLICATION rights, owner of the LoadLens tables only (other tables of the
database are untouched; `APP_DB_SCHEMA` and `APP_DB_TABLES` change the scope). The database port is published on
`127.0.0.1` only. Check the extension with `docker exec -it ltar_timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\dx"`.

The role script can also be run later, against an initialized database: `docker compose exec timescaledb sh
/docker-entrypoint-initdb.d/03_app_role.sh`, or from any machine with `psql` and the variables `PGHOST`, `PGPORT`,
`PGPASSWORD`, `POSTGRES_USER`, `POSTGRES_DB`, `APP_DB_USER`, `APP_DB_PASSWORD`. It runs in one transaction with a
5-second lock timeout, is safe to repeat, keeps the previous owner's access through role membership, and needs a
superuser once the `metrics` hypertable has chunks.

## Redis and the Celery worker

- `redis` (`redis:7.4-alpine`, volume `redis_data`) is the broker and result backend; it requires `REDIS_PASSWORD`
  (`--requirepass`), and `redis-cli` inside the container authenticates through `REDISCLI_AUTH`.
- `celery_worker` is a second container from the same image:
  `celery -A loadlens_app.celery_app worker --loglevel=info -Q reports,celery --concurrency ${WORKER_CONCURRENCY:-2} --max-tasks-per-child 20`.
  Reports, report publications and forecast publications use the queue `reports`; a worker process is replaced after
  20 tasks because pandas does not return memory to the system. It reads `settings_runtime.json` before every job,
  so changes made in the UI apply without a restart; after editing `settings.py` or the environment restart both
  `app` and the worker.
- With `CELERY_TASK_ALWAYS_EAGER=0` and `LOADLENS_JOB_RUNNER=auto` (the Compose defaults) `app` only enqueues jobs
  and shows their progress from `report_jobs`; pages stay fast while a report is built. When Redis does not accept a
  job, `app` runs it in a thread and logs `Could not send job ... to the Celery broker`. A worker that stops
  reporting is marked after `LOADLENS_JOB_STALE_MIN` minutes ("worker does not respond"), a job nobody took after
  `LOADLENS_JOB_QUEUE_MAX_MIN`, and a job longer than `LOADLENS_REPORT_TASK_LIMIT_SEC` is interrupted.
- Capacity: the worker runs `WORKER_CONCURRENCY` jobs at once, the application limit `limits.max_parallel_jobs` is 4
  by default; keep the former at least as large, or start another worker from the same image with the same mounts
  (`--scale` does not work because the service has a `container_name`). The provider sees up to
  `WORKER_CONCURRENCY x llm.<provider>.max_concurrent` parallel requests. The clocks of the application and worker
  hosts must agree (NTP).
- Health: `celery inspect ping` once a minute (`docker compose ps` shows `healthy`); `stop_grace_period: 2m` lets
  running jobs finish on `docker compose stop`. Inspect the queue with
  `docker compose exec celery_worker celery -A loadlens_app.celery_app inspect active` and
  `docker exec ltar_redis redis-cli llen reports`; logs with `docker logs -f ltar_celery_worker`.

Modes, failure handling and a Windows development setup are in [jobs-runner.md](jobs-runner.md) (Russian).

## Health checks and migrations at start

- `GET /healthz` is public and is the `HEALTHCHECK` of the image (every 30 s); `GET /about` answers with the
  version and build even while the database is down.
- At every start `app` checks the data migrations in the background against the table `app_migrations` and runs
  the missing ones: ids for reports saved before ids existed, the `run_index` table (about 2 s per 2 million
  `metrics` rows; the archive is empty until it has run once), the `max_rps` column, the index
  `idx_metrics_run_name`. One-off migrations are recorded and not repeated; cheap recounts are repeated; while the
  database is unreachable the attempts continue with a pause of up to 10 minutes. By hand:
  `docker compose exec app python -m loadlens_app.migrations` (`--status` lists, `--force` repeats the one-off
  ones); `LOADLENS_RUN_MIGRATIONS=0` disables the automatic run.
- If metrics were deleted outside the application, `python tools/check_run_index.py --rebuild --dsn "<libpq string>"`
  rebuilds `run_index` and compares it with `metrics` (`--allow-remote` for a database on another host).
- While the database is unreachable, pages show "Storage unavailable" and the API answers `503
  storage_unavailable`; the recovery administrator can sign in on `/login` and fix `storage.timescale` on
  `/recovery` (30-minute session, actions in the application log).

## Secrets and the image

`.dockerignore` excludes `settings.py`, `metrics_config.py`, `settings_runtime*.json`, `metrics_config_runtime.json`,
every `.env`, `.loadlens_secret_key` and `AI/GigaChat/` from the build context, so an image never contains secrets;
the Compose files mount them at run time. `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` ride along in `/app`, and
the image carries the OCI labels `source`, `version` and `licenses`. Build arguments: `LOADLENS_VERSION` and
`LOADLENS_BUILD` (shown by `GET /about`), `APP_UID` / `APP_GID` when the bind-mounted files on a Linux host belong to
another user than UID 1000. On a Linux host `settings_runtime.json`, `metrics_config_runtime.json` and
`data_collectors/temporary_files` must be writable by that UID.

If an installation was built from a version that still copied the configuration into the image, treat every
secret in it as compromised: rotate them, add `REDIS_PASSWORD`, remove password-less `CELERY_*` URLs from `.env`,
rebuild with `docker compose build --pull --no-cache app celery_worker`, verify with
`docker run --rm --entrypoint sh <image> -c 'ls /app/settings.py /app/docker 2>&1'` (every path must be missing), and
prune old images and build cache (`docker image rm`, `docker builder prune -a`), including tags in a registry.

## Updates

```bash
cd docker
git pull
docker compose up -d --build app celery_worker
```

The application and the worker are built from one image and must be updated together: an old worker would execute
the jobs of a new application. Migrations run at the next start (above). Release notes are in
[CHANGELOG.md](../CHANGELOG.md); the running version is shown next to the logo in the top bar, in "Settings → About"
and by `GET /about`.

## External database and other topologies

- **External TimescaleDB.** Point `storage.timescale` at it and do not start the `timescaledb` service (or remove it
  from the Compose file). The database user needs `CREATE` in the schema; the tables are created on first use, or
  prepare them with `docker/initdb/02_schema.sql` and the role script above.
- **External Redis.** Set `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` with the password in the URL.
- **No worker.** `LOADLENS_JOB_RUNNER=thread` or `CELERY_TASK_ALWAYS_EAGER=1` runs reports in a thread of `app`;
  pages are slower while a report is built. Enough for a small team or a trial.
- **Several application processes.** The in-process limits (`limits`, except parallel jobs), the response cache and
  the user cache (10 s) are per process; job state is in the database. Load test runs are watched by one process at
  a time; a silent watcher's runs are taken over after two minutes.
- **Local run without Docker.** `pip install -r requirements.txt`, the same configuration files,
  `python -m loadlens_app.auth create-user admin --role admin`, `python app.py` (the Flask development server on
  port 5000) or gunicorn with the arguments of the Dockerfile.

## Database tables

All tables are created by the application in `storage.timescale.schema`:

| Table | Content |
|---|---|
| `metrics` | Time series of every run (hypertable) |
| `run_index` | Runs for the archive and the dashboard: name, service, window; filled on save and by the migration for old data |
| `llm_reports` | Domain and overall analysis: verdicts, rationales, SLA details, step table, `system_context` snapshot |
| `llm_feedback`, `report_verdicts`, `engineer_reports` | Votes on findings; manual verdicts with author, reason and time (the last row wins); versions of the engineer summary |
| `report_jobs` | Background jobs: state, progress, owner, runner (thread or worker), timestamps |
| `confluence_publications`, `forecast_publications`, `forecast_explanations` | Published report and forecast pages; the cache of forecast explanations |
| `load_test_runs`, `load_test_presets` | Runs started from LoadLens with their instances; presets of the run form |
| `app_users`, `api_tokens`, `audit_log` | Users (scrypt hashes), token hashes, the audit log |
| `app_migrations` | The registry of executed one-off migrations |

## Troubleshooting

- **Redirected to `/login`, no account.** Create the administrator (above).
- **Sessions end after a restart.** Set a permanent `LOADLENS_SECRET_KEY`.
- **"Storage unavailable" / `503 storage_unavailable`.** The application cannot reach `storage.timescale`; the
  reason is in the application log (`AI.storage_state`, one line per attempt). Fix the connection in recovery mode
  or in `settings.py`.
- **A job stays "Running" at 0 % with "Initializing".** No worker took it: it is not running, busy
  (`WORKER_CONCURRENCY`) or listening on the wrong queue (`-Q reports,celery`); after `LOADLENS_JOB_QUEUE_MAX_MIN`
  minutes the job becomes an error.
- **"Job interrupted: the worker does not respond".** The worker stopped updating the job for `LOADLENS_JOB_STALE_MIN`
  minutes: crashed, killed for memory or lost the database. See `docker logs ltar_celery_worker` and start the
  report again.
- **The archive is empty after an update.** The migrations have not finished: watch the application log or run
  `python -m loadlens_app.migrations`.
- **Slow pages.** The header `Server-Timing` (`app;dur=...`) shows the time inside the application and the gunicorn
  access log ends with the full request time in milliseconds; a large gap means the process is busy with something
  else, for example a report in thread mode.
- **Native histograms, missing pack data, source errors.** See [data-sources.md](data-sources.md#troubleshooting).
