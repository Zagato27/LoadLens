# Running load tests from LoadLens

The "Run a test" page (`/launch`) starts load tools on load generators, watches them, shows basic live metrics and
builds the report for the run window when the last instance finishes. LoadLens does not replace the tools or change
the scripts: it assembles the command from the tool template and the parameters the engineer entered and runs it on
the generator over SSH, in a Docker container or as a Kubernetes Job, or starts a GitLab CI pipeline or a Jenkins job
as it is, with the branch and the pipeline parameters chosen in the form. The Russian original of this page is
[load-tests.md](load-tests.md).

- [How it works](#how-it-works)
- [Load generators](#load-generators)
- [Tool templates](#tool-templates)
- [A run and its instances](#a-run-and-its-instances)
- [Presets, repeat and output hints](#presets-repeat-and-output-hints)
- [How the metrics of a run are separated](#how-the-metrics-of-a-run-are-separated)
- [Live metrics and the report](#live-metrics-and-the-report)
- [Security and limits](#security-and-limits)
- [Troubleshooting](#troubleshooting)

## How it works

A run is one or more instances started together, for example a REST scenario in k6 and a Kafka scenario in Gatling
for a combined test. An instance is one tool process on one generator; instances of a run may use different
generators, tools and generator types.

1. The engineer picks the service and the test type and adds instances: generator, tool, script, parameters,
   environment variables and a start delay.
2. LoadLens validates the form, assembles the commands and stores the run in `load_test_runs`.
3. A background watcher starts the instances, polls them every few seconds, sends stop commands and records the start
   and end times. In Docker it starts with the process (`LOADLENS_LOAD_TEST_WATCHER=1`); otherwise it starts when the
   launch page is first opened. Runs survive a restart of LoadLens; with several processes, the runs of a silent
   watcher are taken over by another after two minutes.
4. When the last instance has finished, LoadLens waits for the metrics to reach the store (two sampling steps, 30 s
   to 5 min) and starts the report for the span with load between the first start and the last end.

The test type of the form selects the deterministic analysis of the report, as on the New report page; the load
profile stays in the script.

## Load generators

Generators are configured by an administrator in "Settings → Connections → Load generators" or in the
`load_generators` section. A new generator is added by a four-step wizard: type and name, connection (with "Test
connection"; for SSH the check is mandatory and shows the host key to trust), tools and directories (the check fills
the list of tools found on the generator), limits and saving. A passed check marks the generator `verified` for the
current connection data; the run form flags unverified generators but does not refuse them.

| Type | How an instance runs | Stop | Duration limit |
|---|---|---|---|
| SSH | A background process under GNU `timeout`; state in files of the instance directory | Signal or the tool's stop command, then `SIGKILL` of the process group | `timeout` on the generator |
| Docker | A container through the Docker Engine API | Signal to the container or the tool command through exec | The LoadLens watcher |
| Kubernetes | A Job with `backoffLimit: 0` | Pod deletion with a grace period (SIGTERM, then SIGKILL) | `activeDeadlineSeconds` |
| GitLab CI | A pipeline of the project with the chosen branch and parameters | Pipeline cancellation (needs an API token) | The CI job limit; after the run limit the watcher cancels the pipeline |
| Jenkins | A build of the job with the chosen parameters | Build stop | The job limit; after the run limit the watcher stops the build |
| Manual | Nothing is started; the window is marked with buttons | The "Finish" button | |

Common fields: `title`, `tools` (available tools with their paths or images; empty means all), `max_parallel`
(instances at once, 0 = no limit), `env` (variables for every tool on the generator), `scripts_dir`.

- **SSH** (`ssh` block: `host`, `port`, `username`, `private_key` or `password`, `passphrase`, `host_key`; plus
  `work_dir`). Needs Linux with POSIX `sh`, GNU `timeout`, `nohup` and the tools installed; give LoadLens its own
  user with write access to `work_dir`. The first check shows the fingerprint of the server key; compare it with the
  server (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`) and confirm. Without a pinned key LoadLens does not
  connect, and a changed key is refused with both fingerprints. Each instance gets `<work_dir>/<run_id>/<NN-tag>/`
  (mode 0700) with `output.log`, `exit_code`, `started_at`, `finished_at` and `pid`; the launcher script that carries
  the environment values is created with mode 0600 and deletes itself once the tool has started. Clean `work_dir` on
  a schedule.
- **Docker** (`docker` block: `url`, `ca_cert`, `client_cert`, `client_key`, `verify`; plus `network`, `mounts`,
  `scripts_dir`, `work_dir`, `resources`, `remove_finished`). The Engine API over HTTPS with client certificates
  (`dockerd --tlsverify`) or through a filtering proxy in front of `/var/run/docker.sock` such as
  `tecnativa/docker-socket-proxy` with `CONTAINERS=1 IMAGES=1 EXEC=1 NETWORKS=1 VOLUMES=1 POST=1`; a local socket is
  not supported. Images (defaults `grafana/k6:0.54.0` and `locustio/locust:2.31.8`; JMeter and Gatling images are set
  by the administrator) must already be on the host; the check lists the missing ones.
- **Kubernetes** (`kubernetes` block: `api_url`, `token`, `ca_cert`, or `in_cluster: true`; plus `namespace`,
  `scripts` as `config_map` or `pvc`, `scripts_dir`, `resources`, `node_selector`, `service_account`,
  `image_pull_secret`, `ttl_seconds`). The service account needs a Role in the namespace: `jobs` create/get/list/
  delete, `secrets` create/delete, `pods` list/get/deletecollection, `pods/log` get, `configmaps` get when the scripts
  are in a ConfigMap; the check asks for these rights through SelfSubjectAccessReview. Instance variables go into a
  Secret owned by the Job; the pod gets no service account token and cannot escalate privileges. An image error
  (`ImagePullBackOff`) turns the instance into an error after three minutes.
- **GitLab CI and Jenkins** (`gitlab` block: `url`, `project`, `ref`, `api_token`, `inputs`; `jenkins` block: `url`,
  `job`, `ref`, `username`, `api_token`; plus `loadlens_url`, `tags_metrics`, `param_options`). The pipeline runs as
  it is; the form offers its branch and parameters, read from GitLab (top-level variables with a `description`,
  `spec:inputs`) or Jenkins (string, text, boolean, choice, password; Active Choices only when the script returns a
  plain list). Only values that differ from the defaults are sent; password parameters are never shown or stored.
  The pipeline receives `LOADLENS_RUN_ID`, `LOADLENS_RUN_NAME`, `LOADLENS_RUN_TAG`, `LOADLENS_INSTANCE`,
  `LOADLENS_INSTANCE_INDEX`, `LOADLENS_MAX_SECONDS` and, with `loadlens_url`, `LOADLENS_EVENTS_URL` and
  `LOADLENS_TOKEN` for the `started` / `finished` events (`POST /load_tests/ci/<run_id>/<index>/events` with the
  header `X-LoadLens-Token`). Start and end come from the pipeline, or from the events when they were sent; the
  pipeline time includes its preparation, and the report window is narrowed by users and total RPS as in every run
  (see "Report window" below); without such data the report takes the pipeline time and the run message says so.
  GitLab needs a token with scope `api` and the Developer role (`read_api` only reads parameters); Jenkins needs
  Job/Read, Job/Build, Job/Cancel and, for Active Choices, Job/ExtendedRead.

Configuration examples of every type are in [load-tests.md](load-tests.md) (Russian; the keys are the same) and,
commented out, in [settings.example.py](../settings.example.py).

## Tool templates

Built-in templates: `k6`, `jmeter`, `locust`, `gatling` (bundle up to 3.10) and `gatling_maven` (Maven, including
bundles 3.11+). An administrator changes them in "Settings → Connections → Load testing tools" or in `load_tools`;
only the differences from the built-in templates are stored. A custom tool, for example Yandex.Tank, is a template
of its own.

| Key | Meaning |
|---|---|
| `command` | The command with placeholders; `{params}` is the form parameters as separate words |
| `executable`, `image` | What replaces `{exe}` (a generator may set its own path); the image for Docker and Kubernetes |
| `env`, `cwd` | Variables (a form value with the same name is appended after a space, handy for `JAVA_OPTS`); working directory, by default the directory of the script |
| `script_kind`, `scripts_glob` | `file`, `class` (a Gatling simulation class) or `none`; the masks the form uses to find scripts on the generator |
| `success_exit_codes`, `failed_checks_exit_codes` | Codes of success and of "the tool ran, the thresholds failed" (k6 `99`, Gatling `2`, Locust `1`); the report is built in both cases |
| `stop_signal`, `grace_seconds`, `stop_command` | The stop signal and the wait before `SIGKILL`; a graceful stop command of the tool itself (SSH and Docker) |
| `status_command`, `version_command` | A status command run every 15 s and shown at the instance; the version command of the generator check |
| `run_filter_packs` | Query packs that see the run tag (below) |

Placeholders: `{exe}`, `{exe_dir}`, `{script}`, `{script_dir}`, `{scripts_dir}`, `{run_name}`, `{run_tag}`,
`{instance}`, `{instance_tag}`, `{instance_dir}`, `{control_port}` (a control port per instance), `{params}`. The
template is split into words first and the values are substituted into each word, so a value never becomes several
arguments; form parameters are split by shell rules without expansion of `$`, `;` or backticks, and every argument
reaches the generator single-quoted.

```text
k6:      {exe} run --address 127.0.0.1:{control_port} --tag testid={run_tag} --tag lt_instance={instance_tag} {params} {script}
         status: {exe} status --address 127.0.0.1:{control_port}
JMeter:  {exe} -n -t {script} -j {instance_dir}/jmeter.log -Japplication={run_tag} -Jlt_instance={instance_tag}
            -Jjmeterengine.nongui.port={control_port} -Jjmeterengine.nongui.maxport={control_port} {params}
         graceful stop: {exe_dir}shutdown.sh {control_port}
Locust:  {exe} -f {script} --headless {params}
Gatling: {exe} -rm local -s {script} -rd {run_tag} {params}
Maven:   {exe} gatling:test -Dgatling.simulationClass={script} {params}
```

## A run and its instances

Instance states: waiting to start, starting, running, finished, finished with failed thresholds, error, stopped,
stopped by the duration limit, lost, cancelled. Run states: starting, running, stopping, waiting for metrics,
building the report, finished, failed to start, cancelled.

- **Start delay** postpones an instance relative to the creation of the run.
- **Stop.** "Stop" at an instance or "Stop run" uses the tool's graceful stop when it has one, else the signal of the
  template; a process still alive after `grace_seconds` gets `SIGKILL`. If the generator does not confirm the stop
  within two minutes, LoadLens closes the instance window with a note to check the process by hand.
- **Duration limit** is set in the form and capped by `limits.max_load_test_hours`. Generators with their own timer
  (SSH, Kubernetes, CI) are stopped by the watcher if the instance is still running two minutes after the limit.
- **Log.** The tail of the tool output is shown on the run page; the last 32 KB are stored with the run, so the log
  survives the removal of the container or Job.
- **Report window** follows the load. Its outer limit runs from the first start to the last end of the instances.
  Inside it LoadLens takes the span from the first to the last point where the users (role `lt.vus`) or the total
  RPS (`lt.rps_total`) are above zero, one query step wider on each side; the queries use the run tag filter and
  pauses inside the test stay. JVM warm-up, pipeline preparation and a slow stop are left out, while a tail where the
  stand is down and requests hang until their timeout stays in: RPS falls to zero there, the users still run. The run
  message names the signal that narrowed the window; with no data from either, the window is the process time. The
  run page and the archive always show the process time. An instance that failed in its first minute is left out;
  a run where no instance started finishes without a report.

Several runs on one generator are allowed; the form warns, and `max_parallel` of the generator is the hard cap.

## Run history in the archive

- **Run name.** A name is taken while a run of the service with that name is going, or while the archive holds a
  report with that name. A finished run does not hold its name: after its report is deleted, the name can be used
  again. When the name is taken by the time the report is built, the report gets a suffix (`nightly-1a2b`).
- **The Runs tab** of the archive (`/reports?tab=runs`, the "All runs" link on the Run test page) lists the runs of
  the active area, newest first: status, start and duration, instances and tools, who started it and the report.
  It filters by name, service and status. Viewers see the tab too, without commands or parameters, and without a
  link to the run page.
- **A run and its report are deleted separately.** Deleting a report keeps its run; deleting a run keeps its report.
  The run finds its report by the report id, so renaming the report keeps the link.
- **Deleting runs.** An engineer deletes their own finished runs, an administrator anyone's, from the row menu or
  with checkboxes. Running runs are stopped first. An administrator can tick "Also delete the reports of these
  runs" in the confirmation. Deleting a service also deletes its finished runs.

## Presets, repeat and output hints

- A **preset** is a saved form of a service (instances, parameters, test type, duration, report options), shared by
  the engineers. Values of environment variables are not stored, only their names.
- **Repeat** on the run page and in the archive menu opens the form with the instances and parameters of a past run
  and asks for the variable values again.
- **Metrics output** at an instance adds the flags and variables that send the tool's metrics to the source of the
  service's load tool domain and its installed packs: k6 remote write for Prometheus and VictoriaMetrics, k6
  InfluxDB output for InfluxQL through Grafana, Gatling Graphite output. Addresses are the ones LoadLens sees; source
  credentials are never included.

## How the metrics of a run are separated

The run name becomes a tag of Latin letters, digits, `_` and `-` (for example `release-2-5-step`). Tools that can
tag their metrics get it in the command, and the report and the live charts put it into the pack variable
(`[run_filter]` in the TOML of the pack).

| Tool | Tagging | Packs |
|---|---|---|
| k6 | `--tag testid=<tag>` on every output | `lt-k6-prometheus-rw`, `lt-k6-prometheus-native`, `lt-k6-influxql` |
| JMeter | `-Japplication=<tag>` when the Backend Listener reads `${__P(application,...)}` | `lt-jmeter-influxql` |
| Gatling | Not tagged by default; the tag can replace the Graphite prefix (`-Dgatling.data.graphite.rootPathPrefix={run_tag}`) or the simulation name (`-Dgatling.core.outputDirectoryBaseName={run_tag}`) through `env.JAVA_OPTS` and `run_filter_packs` of the template | `lt-gatling-graphite-victoriametrics`, `lt-gatling-graphite-victoriametrics-labels` |
| Locust | Not tagged | Separated by the run window |
| CI pipeline | Its own command with `LOADLENS_RUN_TAG`; `tags_metrics` on at the generator | The installed packs with a run filter |

When the tagged query finds nothing but the untagged one has data, the tool did not tag its metrics (a JMeter test
plan with a fixed `application`, for instance): the report and the charts read all metrics of the window and say so
in the run message. A CI instance without `tags_metrics` switches the tag filter off for the whole run. The
VictoriaMetrics relabeling rules for Gatling are listed in [load-tests.md](load-tests.md) and in
`docker/dev-stack/gatling/vm-relabel.yml`.

A combined run writes to one store with one query language, with the packs of both tools installed in the load
tool domain; "Total RPS of the tools" in the Queries section adds the row that sums them and makes it the SLA
throughput query (PromQL only), so the steps of the combined test come from the whole load.

## Live metrics and the report

The run page shows RPS, error ratio, p95 and users from the queries with the roles `lt.rps_total`,
`lt.error_rate_pct`, `lt.p95_total` and `lt.vus` of the service's load tool domain: the last values in a row of
figures and three charts. Without an installed pack the charts stay empty and the launch still works.

- While the test runs, the page reads only new points every 5 s and shows "data as of ..., delay N s" above the
  charts; a delay of 5 to 15 s is normal, a minute without new points is highlighted.
- Window: the whole run or the last 15, 30 or 60 minutes, with a sampling step adapted to it.
- In a run of several instances the RPS chart shows the sum and each instance by the `lt_instance` tag (role
  `lt.rps_by_instance`, present in the k6 and JMeter packs).
- Markers: instance starts and ends, stop requests, the moment the duration limit fires; horizontal lines for
  `target_rps`, `max_error_rate_pct` and `max_p95_ms`.

The dashboard shows the running runs of the active area to engineers and administrators ("Running now": state,
service, who started it, instances, time against the limit, RPS of the last 15 minutes with a sparkline, error
ratio, a stop button for your own runs), their number in the status line and a counter next to "Run a test" in the
top bar. The report starts after the metrics delivery pause; if every job slot is busy it waits, and if the daily AI
budget is exhausted it is built without the analysis. The New report form uses the same role `lt.rps_total` to
suggest the windows of tests that were not started from LoadLens (`GET /lt_windows`).

The optional `lt_generators` domain ("Load generator resources", pack `lt-generators-node-exporter` with matchers
of the generator hosts) shows whether the generator itself ran out of CPU, memory, network or connections; it is
analysed as a separate domain and taken into account by the overall verdict.

## Security and limits

- Starting, stopping and viewing runs needs the Engineer role. Scripts are arbitrary code and parameters are free
  text, so an engineer effectively runs code on the generator: make it a dedicated machine, cluster or CI project
  with a least-privilege account, and put only secrets you can trust the engineers with into the generator `env`.
- An instance cannot override generator variables or set variables that change the shell or the runtimes (`PATH`,
  `LD_*`, `BASH_ENV`, `ENV`, `JAVA_TOOL_OPTIONS`, `NODE_OPTIONS`, `PYTHONPATH`); `JAVA_OPTS`, `HEAP`, `JVM_ARGS` and
  the tools' own variables are allowed. Scripts must live in the generator's script directory.
- The Docker API equals root on the host: expose it to LoadLens only, with TLS client certificates or a filtering
  proxy. Kubernetes: a Role on the load namespace only, no API token in the pod. CI: pass secrets as Jenkins
  password parameters or CI/CD variables; the instance secret for events is stored as SHA-256 and lives until the
  instance ends.
- Instance variable values are not shown in the UI or the audit log and are erased when the instance finishes;
  generator secrets are returned as `***` and only re-used while the address, port, CA, `in_cluster` and the pinned
  host key are unchanged. Start, stop, delete and preset actions go to the audit log (`load_test.start`,
  `load_test.stop`, `load_test.delete`, `load_test.preset_save`, `load_test.preset_delete`) with the commands,
  variable values hidden.
- Limits (`limits`): `max_parallel_load_tests`, `max_parallel_load_tests_per_user`, `max_load_test_instances`,
  `max_load_test_hours`.

The endpoints are listed in [api.md](api.md#load-tests-engineer). A stand with an SSH generator (k6, Locust,
Gatling bundle), Docker through a socket proxy and GitLab and Jenkins stubs is in
[docker/dev-stack](../docker/dev-stack/README.md) (profile `generator`).

## Troubleshooting

| What you see | Cause |
|---|---|
| Exit code 127 | Tool not found: check the path in the generator or the template |
| Exit code 126, "no working directory" | The `cwd` of the template does not exist on the generator |
| "Stopped by the duration limit" | The run's duration limit fired |
| "Host key changed" / "Host key not pinned" | The SSH server presented another key: verify it is the same server and pin again; or the generator was saved without a check: click "Check" and save |
| "Image ... is not on the Docker host" | Pull the image on the host in advance |
| "Did not start: ImagePullBackOff" | Kubernetes cannot pull the image: name, tag or registry secret |
| "no rights: ..." in the Kubernetes check | The Role of the service account lacks the listed verbs |
| A CI instance stays "Starting" | The pipeline waits for a runner or the build is queued: open it in GitLab or Jenkins |
| "Pipeline parameters cannot be read" | Check the API token (GitLab: scope `api` or `read_api`), the project or job and the branch |
| "The GitLab project forbids pipeline variables" | Allow them in the CI/CD settings or declare the values as `spec:inputs` |
| "timeout is not GNU coreutils" | A busybox SSH generator: the stop may not reach child processes |
| "Process disappeared without an exit code" | The generator rebooted or the process was killed from outside |
| "Tool metrics without the run tag" | The tool did not tag its metrics, see above |
| No tool metrics in the report | The service has no pack of the tool, or the load tool domain reads another store |
