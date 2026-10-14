#!/bin/sh
# Creates the unprivileged role the application connects with (APP_DB_USER / APP_DB_PASSWORD) and
# hands it the LoadLens tables: the app adds columns and indexes on its own, which needs ownership.
# The role is not a superuser and cannot create roles or databases. The account that runs the script
# becomes a member of the role, so tools that keep using that account do not lose access.
#
# The postgres image runs this once, on an empty data volume. For an existing compose database:
#   docker compose exec timescaledb sh /docker-entrypoint-initdb.d/03_app_role.sh
# For another server, from any machine with psql. Use a superuser once the metrics table holds data:
# TimescaleDB chunks can change owner only under a superuser. Before that, an account with
# CREATEROLE that owns the database and the LoadLens tables is enough.
#   PGHOST=<host> PGPORT=5432 PGPASSWORD=<password> POSTGRES_USER=<account> POSTGRES_DB=<database> \
#   APP_DB_USER=loadlens_app APP_DB_PASSWORD=<new password> sh docker/initdb/03_app_role.sh
# APP_DB_SCHEMA and APP_DB_TABLES (space separated) override the defaults when settings.py renames
# them. Only the listed tables change owner; everything else in the schema is left alone.
#
# Everything runs in one transaction with a short lock timeout: on any error nothing changes.
# The password reaches psql through the environment, not the command line. The server receives it
# in CREATE/ALTER ROLE, so with log_statement = ddl or all a non-superuser run can leave it in the
# server log; a superuser run switches statement logging off for the transaction.
# The image may source this file instead of executing it, so it never calls exit or set -e.

LOADLENS_TABLES="metrics llm_reports engineer_reports llm_feedback report_jobs confluence_publications forecast_explanations forecast_publications app_users api_tokens audit_log"

if [ -n "${APP_DB_USER:-}" ] && [ -n "${APP_DB_PASSWORD:-}" ]; then
  case "$APP_DB_USER" in
    *[!a-z0-9_]* | [0-9_]* | pg_*)
      echo "03_app_role.sh: APP_DB_USER may contain only lower-case letters, digits and _ and must not start with pg_; role not created" >&2
      false
      ;;
    *)
      export APP_DB_PASSWORD
      psql -v ON_ERROR_STOP=1 --single-transaction --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
        --set=app_user="$APP_DB_USER" --set=app_schema="${APP_DB_SCHEMA:-public}" \
        --set=app_tables="${APP_DB_TABLES:-$LOADLENS_TABLES}" <<'SQL'
\set app_password `printenv APP_DB_PASSWORD`
SET LOCAL lock_timeout = '5s';
SELECT 'SET LOCAL log_statement = ''none''' FROM pg_roles WHERE rolname = current_user AND rolsuper \gexec

SELECT :'app_user' = current_user AS same_account,
       EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user') AS role_exists \gset
\if :same_account
DO $$ BEGIN RAISE EXCEPTION 'APP_DB_USER must differ from the account that runs this script'; END $$;
\endif

\if :role_exists
SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'app_user', :'app_password') \gexec
-- Only the powers the role actually has are revoked: touching SUPERUSER or REPLICATION at all
-- needs a superuser, and a role that has neither should not make the run fail.
SELECT format('ALTER ROLE %I WITH %s', rolname, concat_ws(' ',
         CASE WHEN rolsuper THEN 'NOSUPERUSER' END, CASE WHEN rolcreatedb THEN 'NOCREATEDB' END,
         CASE WHEN rolcreaterole THEN 'NOCREATEROLE' END, CASE WHEN rolreplication THEN 'NOREPLICATION' END,
         CASE WHEN rolbypassrls THEN 'NOBYPASSRLS' END))
  FROM pg_roles
 WHERE rolname = :'app_user' AND (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls) \gexec
\else
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_user', :'app_password') \gexec
\endif

SELECT format('GRANT %I TO CURRENT_USER', :'app_user') \gexec
-- CREATE on the database: the app runs CREATE SCHEMA IF NOT EXISTS, which checks it even for an existing schema.
SELECT format('GRANT CONNECT, CREATE, TEMPORARY ON DATABASE %I TO %I', current_database(), :'app_user') \gexec
SELECT format('GRANT USAGE, CREATE ON SCHEMA %I TO %I', :'app_schema', :'app_user') \gexec

-- TimescaleDB moves the chunks of a hypertable along with it. Outside the superuser the new owner
-- needs CREATE on the chunk schema, which belongs to the extension owner, so say so up front.
SELECT EXISTS (
         SELECT 1
           FROM pg_inherits i
           JOIN pg_class p ON p.oid = i.inhparent
           JOIN pg_namespace pn ON pn.oid = p.relnamespace
           JOIN pg_class c ON c.oid = i.inhrelid
          WHERE pn.nspname = :'app_schema' AND p.relname = ANY (string_to_array(:'app_tables', ' '))
            AND NOT has_schema_privilege(:'app_user', c.relnamespace, 'CREATE'))
       AND NOT (SELECT rolsuper FROM pg_roles WHERE rolname = current_user) AS chunks_need_superuser \gset
\if :chunks_need_superuser
DO $$ BEGIN RAISE EXCEPTION 'The metrics hypertable already has TimescaleDB chunks; only a superuser can hand them to another owner. Run the script as a superuser (POSTGRES_USER=postgres).'; END $$;
\endif

-- Whoever owned the tables so far keeps access through membership, so the account the app used
-- before still works and switching settings.py back stays possible.
SELECT DISTINCT format('GRANT %I TO %I', :'app_user', r.rolname)
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace JOIN pg_roles r ON r.oid = c.relowner
 WHERE n.nspname = :'app_schema' AND c.relkind IN ('r', 'p')
   AND c.relname = ANY (string_to_array(:'app_tables', ' '))
   AND r.rolname NOT IN (:'app_user', current_user) AND NOT r.rolsuper \gexec

SELECT format('ALTER TABLE %I.%I OWNER TO %I', n.nspname, c.relname, :'app_user')
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE n.nspname = :'app_schema' AND c.relkind IN ('r', 'p')
   AND c.relname = ANY (string_to_array(:'app_tables', ' ')) \gexec
SQL
      status=$?
      if [ "$status" -eq 0 ]; then
        echo "03_app_role.sh: role $APP_DB_USER is ready; use it in settings.py storage.timescale.user/password"
      else
        echo "03_app_role.sh: failed with status $status; the transaction was rolled back, nothing changed" >&2
        false
      fi
      ;;
  esac
else
  echo "03_app_role.sh: APP_DB_USER/APP_DB_PASSWORD are not set; the application keeps using $POSTGRES_USER" >&2
fi
