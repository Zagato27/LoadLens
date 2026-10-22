# Data sources, domains and query packs

LoadLens reads the metrics of a test from the stores you already have: Prometheus or VictoriaMetrics (PromQL),
InfluxDB 1.x (InfluxQL) and 2.x (Flux), directly or through a Grafana data source proxy. A *source* is a connection;
a *domain* is a group of series the report analyses together; a *binding* says which source a domain reads; a
*query pack* fills the queries of a domain for a typical exporter, runtime or load tool. The generated reference of
every pack is [query-packs.en.md](query-packs.en.md) (built from the TOML files in `AI/query_packs/packs/`).

- [Sources](#sources)
- [Domains and bindings](#domains-and-bindings)
- [Queries, labels, roles and units](#queries-labels-roles-and-units)
- [Query packs](#query-packs)
- [Load tools and their packs](#load-tools-and-their-packs)
- [Application logs from OpenSearch](#application-logs-from-opensearch)
- [Troubleshooting](#troubleshooting)

## Sources

The catalogue lives in `data_sources` ("Settings → Connections → Data sources") and is shared by all project areas.
A source stores only how to connect; the data source of Grafana, the InfluxQL database and the Flux bucket belong to
the binding. Every connection is tested with the form values before it is saved; passwords and tokens are shown as
`***` afterwards and must be re-entered when the address, port, proxy or TLS settings change.

| `type` | Block | Fields | Query languages |
|---|---|---|---|
| `prometheus` | `prometheus` | `url` (for VictoriaMetrics the Prometheus-compatible endpoint, e.g. `http://victoria:8428`; a cluster uses the `vmselect` path `.../select/0/prometheus`) | PromQL |
| `grafana_proxy` | `grafana` | `base_url`, `verify_ssl`, `auth` with `method` (`basic` or `bearer`), `username`, `password`, `token`. The domain binding picks the data source by `datasource_uid` or `datasource_name` from the list Grafana returns (`GET /config/grafana_datasources?source=<id>`; Prometheus and InfluxDB data sources are listed) | PromQL, InfluxQL, Flux, depending on the data source |
| `influxdb` | `influxdb` | `url`, `org`, `token` (InfluxDB 2.x API) | Flux |

Rules: a direct `prometheus` source runs PromQL only, a direct `influxdb` source runs Flux only, and a Grafana source
runs the language of the chosen data source (InfluxQL for an InfluxDB 1.x data source). A domain whose queries are
written in a language the source does not accept fails with a configuration error that names the domain, the source
and the languages found.

## Domains and bindings

| Domain key | In the UI | What it holds |
|---|---|---|
| `lt_framework` | Load testing tool | RPS, latency percentiles, errors and virtual users of the load tool; the SLA and the step detector work on this domain |
| `jvm` | JVM | Runtime of the service: heap, GC, threads, process CPU; Go, Python and container packs also write here, and the domain title in reports follows the installed packs ("Go runtime", "JVM / Containers") or the `title` field of the domain block |
| `microservices` | Microservices | HTTP and gRPC server metrics: RPS, latency, error ratio, by route; outgoing calls |
| `database` | Database | Databases and connection pools: PostgreSQL, MySQL, MongoDB, Redis, ArangoDB, PgBouncer, HikariCP |
| `kafka` | Kafka | Brokers, topics, consumer lag, clients |
| `hard_resources` | Node resources | CPU, memory, disk, network of the nodes; Kubernetes node and workload health |
| `lt_generators` | Load generator resources | Optional: CPU, memory, network and connections of the load generator hosts, so the report can tell when the generator itself was the bottleneck |
| `application_logs` | Application logs | Optional: aggregated ERROR logs from OpenSearch for the test window (see below) |

`domain_sources` maps domains to sources: the `default` entry serves every domain without a line of its own, and each
domain may name another source. A binding carries `source` and, depending on the source type, `datasource_uid` or
`datasource_name` (Grafana), `database` (InfluxQL) and `bucket` (Flux).

```python
"domain_sources": {
    "default": {"source": "grafana", "datasource_uid": "prom-uid", "datasource_name": "Prometheus"},
    "lt_framework": {"source": "grafana", "datasource_uid": "influx-uid", "datasource_name": "InfluxDB-k6", "database": "k6"},
    "lt_generators": {"source": "prometheus"},
}
```

A project area replaces the whole `domain_sources` object; the catalogue stays global. "Settings → Connections →
Domain binding" shows the effective binding of the open scope. The lists of sources and bindings are also what the
wizard's "Domain binding" step edits.

### A source per query row

The binding of a domain is the default of its rows. A row can read another source: two Prometheus servers, two
Grafana data sources, k6 in InfluxDB and Gatling in VictoriaMetrics in one `lt_framework`, node_exporter of the
database hosts in an infrastructure Prometheus. Open the row in "Settings → Queries" and pick its "Source"; a query
pack takes the source for all its rows in the "Source" field of the pack card. The row is then checked, collected
and drawn on the live charts from that source; the rest of the domain keeps the binding. The domain tab and the
"Domain binding" card show how many rows read a source of their own.

```python
"queries": {
    "database": {
        "labels": ["DB: TPS", "DB host: CPU usage (%) by node"],
        "promql_queries": ["...", "..."],
        "label_keys_list": [["datname"], ["node"]],
        "sources": {"DB host: CPU usage (%) by node": {"source": "prom_infra"}},
    },
}
```

`sources` maps a row label to a binding of the same shape as a `domain_sources` entry. `null` means "the domain
binding": a service or an area writes it to return a row that a wider scope reads elsewhere. Saving the queries
checks every row source against the catalogue and the language of the row, and stores every field of the binding.
A source that a row of any scope reads cannot be removed from the catalogue.

A source runs one language: a direct Prometheus runs PromQL, a direct InfluxDB runs Flux, a Grafana data source runs
the language of its type. A domain may hold rows in several languages as long as each language is read from a source
that runs it. The report collects a domain in groups, one per source and language; a source that does not answer
leaves only its own rows empty.

## Queries, labels, roles and units

`queries.<domain>` holds parallel lists: `labels` (the series names shown in the report and used as identifiers of
the stored series), the queries (`promql_queries`, `influxql_queries`, `flux_queries`) and the label keys that name a
series (`label_keys_list` for PromQL, `label_tag_keys_list` for InfluxQL and Flux). A row runs in the language whose
list holds its text; a row with texts in several languages runs the one its source accepts (InfluxQL, then PromQL,
then Flux for Grafana), and saving warns about it. Two service keys are written by the packs: `roles` maps a label to
its role (`lt.rps_total`, `lt.p95_total`, `node.cpu_pct`, `runtime.cpu`, ...) and `packs` records the installed packs
with their versions and variables. `sources` holds the rows that read their own source (see above).
Flux queries use the placeholders `{bucket}`, `{start}` and `{end}`; InfluxQL queries use `$timeFilter` and
`$__interval` as in Grafana.

The SLA and the forecast rely on three conventions, which the packs follow and hand-written queries should keep:

| Series | Unit and key | Used by |
|---|---|---|
| Node CPU and memory | percent, series key `node` | `max_cpu_pct`, `max_memory_pct`, the forecast |
| CPU of one service instance | fraction 0..1, series keyed by `instance` (and the service label) | The pod plan of the forecast (`service_cpu_query`, role `runtime.cpu`) |
| Load tool latencies | milliseconds, like the `max_p95_ms` / `max_p99_ms` thresholds; `sla.latency_unit` says `ms` or `s` | SLA, the forecast |

An empty SLA reference (`max_performance_query`, `p95_query`, `cpu_query`, ...) is resolved from the roles of the
installed queries; an explicit reference to an existing query wins; a reference to a query that no longer exists is
treated as empty, with a warning in the log and a "no such query" mark in the SLA form. The label of a query is the
identifier of its series in the database, in the SLA and in run comparison: never rename it between pack versions.

## Query packs

A pack is a TOML file with the queries of one domain for one source of metrics: an exporter, a runtime library or a
load tool. It declares variables (the service label, extra matchers, the `rate()` window, the load model), the role,
unit and meaning of every query, which queries are mandatory, the SLA references it fills and a `prompt_addendum`
that tells the model what the stack's metrics mean. Bundles group packs for a stack:

| Bundle | Packs |
|---|---|
| Spring Boot on Kubernetes + k6 (InfluxDB) | `lt-k6-influxql`, `jvm-micrometer`, `http-micrometer`, `db-hikaricp`, `db-postgres-exporter`, `kafka-exporter`, `node-exporter`, `k8s-nodes`, `k8s-workload-health` |
| Spring Boot on virtual machines + JMeter | `lt-jmeter-influxql`, `jvm-micrometer`, `http-micrometer`, `db-hikaricp`, `db-postgres-exporter`, `node-exporter` |
| Java with JMX exporter + Gatling | `lt-gatling-influxql`, `jvm-jmx-exporter`, `http-otel`, `kafka-broker-jmx`, `node-exporter` |
| Go on Kubernetes + k6 (Prometheus remote write) | `lt-k6-prometheus-rw`, `runtime-go`, `http-go`, `db-postgres-exporter`, `node-exporter`, `k8s-nodes`, `k8s-workload-health` |
| Python (FastAPI) on Kubernetes + k6 (Prometheus remote write) | `lt-k6-prometheus-rw`, `k8s-workload-cpu`, `http-python-fastapi`, `db-postgres-exporter`, `node-exporter`, `k8s-nodes`, `k8s-workload-health` |

Beyond the bundles there are packs for JVM (OpenTelemetry agent 1.x and 2.x), HTTP (Django, Flask, OpenTelemetry
before and after semconv 1.23, NGINX ingress, Istio), MySQL, MongoDB, Redis, ArangoDB, PgBouncer, Kafka clients
(Micrometer), windows_exporter, container CPU and memory, Locust, Yandex.Tank, the load generators and the component
hosts; 43 packs in total.

### Hosts of a component

`host-node-exporter` and `host-windows-exporter` put the resources of the hosts a component runs on (a database
server, a Kafka broker, the VM of a service) into that component's section: CPU, memory, load per core, disk, network
and file system fill next to the TPS or the consumer lag. The domain is picked in the pack card ("Database", "Kafka",
"JVM" or "Microservices"). One domain takes up to 10 installations of the pack, told apart by the "Host title"
variable that starts every label ("DB primary", "DB replica", "Kafka brokers"): applying with a new title adds an
installation, applying with an existing one updates it. The roles of these rows (`host.cpu_pct`, `host.mem_pct`, ...)
fill no SLA field: the node CPU and memory SLA and the forecast keep reading `node.*` of "Node resources". Limit the
matchers of `node-exporter` to the application nodes, or the database host is reported in both sections.

Applying a pack ("Settings → Queries → Add from a pack"; the wizard opens the same dialog):

1. The catalogue applies to the scope open in the section: global settings, a project area or a service. "Detect
   from sources" reads the metric or measurement names of each domain's source and sorts packs and bundles by how
   many of the metrics they need were found; in the wizard the detection runs by itself.
2. Fill the variables: the label that carries the service name, extra matchers, the `rate()` window, the load model
   of the scenario.
3. "Check" runs every query against the domain's source for the check window; "Keep only with data" drops the
   optional queries that returned no series.
4. "Show changes" lists the queries and SLA references that will be added, updated or removed; "Apply" saves them.
   The application can be undone from the notification.

Rules of application:

- "Add and update" leaves foreign queries alone and updates the pack's own queries in place (same labels, so the
  series history is kept). The "Source" of the pack card decides which source the pack rows read; by default the
  domain binding. A pack in another language than the rows reading the same source shows "Language conflict": pick
  another source for the pack, or "Replace all queries of the domain".
- SLA references switch to the pack's queries by default; untick "Replace existing references with the pack's
  queries" to keep yours. A reference to a query that disappears is always replaced.
- Load tool packs report latencies in milliseconds and set `latency_unit = "ms"` only when every latency reference
  points at their queries.
- A new pack version is offered by an "Update" button in the Queries section: it marks the queries that already
  exist and the ones added by the new version, keeps the rows you deleted away and reuses the installed variables.
- "Verified on the stand" means every mandatory query returned data on [docker/dev-stack](../docker/dev-stack/README.md)
  with live exporters; the other packs are checked syntactically and for the series and unit contract.

The command line reads the same catalogue: `python -m AI.query_packs list`, `render <pack_id> --var name=value`,
`bundle <bundle_id>`, `docs [--lang en] <file>` (regenerates [query-packs.en.md](query-packs.en.md) and its Russian
twin [query-packs.md](query-packs.md)) and `example` (rewrites the
queries and SLA references of `settings.example.py` from the default bundle). The REST endpoints are listed in
[api.md](api.md#settings-administrator).

## Load tools and their packs

| Tool | Metrics output | Pack | Run tag ([running-load-tests.md](running-load-tests.md#how-the-metrics-of-a-run-are-separated)) |
|---|---|---|---|
| k6 | `--out influxdb` (InfluxDB 1.x) | `lt-k6-influxql` | `--tag testid=<tag>` |
| k6 | InfluxDB 2.x | `lt-k6-influx2-flux` | `--tag testid=<tag>` |
| k6 | `--out experimental-prometheus-rw` with trend stats (`K6_PROMETHEUS_RW_TREND_STATS`) | `lt-k6-prometheus-rw` | `--tag testid=<tag>` |
| k6 | Prometheus remote write with native histograms | `lt-k6-prometheus-native` | `--tag testid=<tag>` |
| JMeter | Backend Listener to InfluxDB 1.x | `lt-jmeter-influxql` | `-Japplication=<tag>` when the listener reads `${__P(application,...)}` |
| Gatling | Graphite writer to InfluxDB 1.x (graphite template `gatling.*.*.*.*`) | `lt-gatling-influxql` | not tagged by default |
| Gatling | Graphite writer to VictoriaMetrics, paths stored as metric names | `lt-gatling-graphite-victoriametrics` | tag replaces the Graphite root prefix |
| Gatling | Graphite writer to VictoriaMetrics with relabeling rules | `lt-gatling-graphite-victoriametrics-labels` | tag replaces the simulation name |
| Locust | `locust_exporter` scraped by Prometheus | `lt-locust-prometheus` | separated by the run window |
| Yandex.Tank | InfluxUploader to InfluxDB 1.x | `lt-yandex-tank-influxql` | separated by the run window |
| Generators | node_exporter on the generator hosts | `lt-generators-node-exporter` (domain `lt_generators`) | |

Several tools in one combined run write to the same store (for example k6 through remote write and Gatling through
Graphite into VictoriaMetrics); with more than one "total RPS" query in the domain, "Total RPS of the tools" in the
Queries section adds a row that sums them and makes it the SLA throughput query (PromQL only).

## Application logs from OpenSearch

`logs_source` switches on the optional `application_logs` domain: LoadLens queries OpenSearch Dashboards through
`/api/console/proxy` for ERROR-level documents of the test window, groups them by service and error type with samples,
and passes the aggregate to the model as one more domain. The block names the index pattern, the field names
(timestamp, level, hostname, message), the levels and hosts to filter and the aggregation limits; the keys are listed
in [configuration.md](configuration.md#logs_source). The same aggregator powers the `opensearch_error_report.py`
command-line report.

## Troubleshooting

- **A pack check returns "no data".** The check window does not overlap a running service, the service label differs
  (`application`, `job`, `app`), or the source lacks the metric. Frequent causes: Micrometer p95 is empty without
  `percentiles-histogram`, k6 statistics are missing in Prometheus without `K6_PROMETHEUS_RW_TREND_STATS`, Python in
  multiprocess mode exposes no `process_*` metrics.
- **HTTP packs are empty although the application exposes histograms.** Prometheus runs with native histograms
  (`--enable-feature=native-histograms`) and does not keep the classic `_bucket`, `_count`, `_sum` series. Set
  `scrape_classic_histograms: true` (Prometheus 2.x) or `always_scrape_classic_histograms: true` (3.x) in the scrape
  config.
- **"Language conflict" or "Incompatible" next to a pack.** Rows in another language already read the source the
  pack would read, or the source does not accept the pack's language. Pick another source in the pack card, change
  the binding or apply the pack with replacement.
- **"Source X is in use: queries of the area …, domain …, row …" when saving the catalogue.** A query row reads the
  source. Return the row to the domain binding (or delete it) in that scope first.
- **A row shows "Two languages".** The row has texts in several languages, left from an earlier language of the
  domain. Open the row and press "Keep only …".
- **The database host is in both "Node resources" and "Database".** The matchers of `node-exporter` also match the
  database host; exclude it there (for example `instance!~"db-.*"`).
- **"Detect from sources" reports an error for a domain.** The message names the source and its answer. A Grafana
  binding needs a selected data source, and the Grafana user needs the permission to read data sources. Detection
  also reads the direct Prometheus sources of the catalogue and the sources rows already read; a pack whose metrics
  are found there gets "Found in …" and a "Read from it" button that sets the pack's source.
- **"The source answered 404" when checking a query.** The message names the failing query. If the data source is
  not found, pick it from the list in the domain binding so that the uid is stored.
- **VictoriaMetrics returns nothing for the last seconds.** It hides the latest 30 s by default
  (`-search.latencyOffset`); build the report of a test that has just finished a minute later.
