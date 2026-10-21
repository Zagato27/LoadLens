#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Dev load for the db-arangodb pack: document CRUD (implicit transactions), AQL inserts and AQL reads, some of
# them above the slow-query threshold of the dev server. Runs ARANGO_LOAD_SEC seconds (default 600).
set -eu
ENDPOINT="${ARANGO_ENDPOINT:-tcp://arangodb:8529}"
DURATION="${ARANGO_LOAD_SEC:-600}"

until arangosh --server.endpoint "$ENDPOINT" --server.authentication false --quiet \
    --javascript.execute-string 'db._version()' >/dev/null 2>&1; do
  echo "waiting for $ENDPOINT"
  sleep 2
done

bench() {
  arangobench --server.endpoint "$ENDPOINT" --server.authentication false --duration "$DURATION" --quiet "$@" &
}
bench --test-case crud-write-read --collection bench_crud --threads 4 --complexity 4
bench --test-case aqlinsert --collection bench_aql --threads 2
bench --test-case custom-query --collection bench_query --threads 2 \
  --custom-query 'FOR i IN 1..@n COLLECT g = i % 100 WITH COUNT INTO c RETURN {g, c}' \
  --custom-query-bindvars '{"n": 200000}'
wait
