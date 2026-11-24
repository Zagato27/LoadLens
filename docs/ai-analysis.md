# AI analysis

The `AI/` package collects the metrics of every domain, builds compact context packs, asks the model for several
candidate analyses per domain, scores them with a judge and checks every number in the findings against the data
before the report shows it. The deterministic parts of the report (steps, SLA, hold and spike windows, saturation)
are computed in code; the model explains, it does not decide the verdict. The Russian maintainer notes of the package
are in [AI/README.md](../AI/README.md).

- [Pipeline](#pipeline)
- [Test types and deterministic analysis](#test-types-and-deterministic-analysis)
- [Candidates, judge and verification](#candidates-judge-and-verification)
- [Verdict codes and the result format](#verdict-codes-and-the-result-format)
- [Prompts and languages](#prompts-and-languages)
- [System context](#system-context)
- [Providers](#providers)
- [Other model calls and limits](#other-model-calls-and-limits)
- [Troubleshooting](#troubleshooting)

## Pipeline

1. **Collect.** For the test window the queries of every domain run against its source (PromQL, InfluxQL or Flux,
   directly or through Grafana), the series are stored in TimescaleDB and resampled to `default_params.resample_interval`.
2. **Context packs.** Each domain gets a summary per series: averages, minimums, maximums, last values, the time of
   the peaks and anomaly windows; the `meaning`, `unit` and `description` of a query travel with the series (from
   its pack, or from the role and the note set in "Settings → Queries").
   Domains: JVM and runtime, databases, Kafka, microservices, node resources, load tool, and optionally load
   generators and application logs.
3. **Deterministic analysis.** Step detection on the assigned RPS series, the SLA check, and the analysis of the test
   type (below). Their results go into the prompts and into the report JSON.
4. **Candidates.** `llm.self_consistency_k` candidates per domain, each coerced to strict JSON; domains run in
   parallel (`llm.max_domain_workers`), and domains that failed in the parallel pass are retried one by one with a
   pause (bursts to the provider are the usual cause).
5. **Judge and verification.** A G-Eval-lite judge scores the candidates with a rubric per domain, every number in
   the findings is checked against the collected data, and the best candidate is chosen: first the answer in the
   report language, then by score.
6. **Overall conclusion.** The final prompt sees the domain verdicts, the SLA result and the previous run of the
   service (`llm.baseline`) and produces findings and recommendations. The pipeline generates the final
   verdict rationale from the SLA checks and saves the report and quality metrics (`scores`) to `llm_reports`.

Progress is reported per phase (collecting each domain, each model answer, the SLA check, the final analysis,
saving); the job page shows it with the elapsed and the usual time. The entry point is
`AI.pipeline.uploadFromLLM(start_ts, end_ts, ...)` (`AI.main.uploadFromLLM` is a proxy).

## Test types and deterministic analysis

The test type of a report (`step`, `soak`, `spike`, `stress`, `scenario`; `AI/test_profile.py` maps aliases, including the
Russian names) selects the analysis mode, the SLA window and the profile text added to the prompts.

| Type | Computed in code | SLA window and the meaning of `target_rps` | Verdict rules |
|---|---|---|---|
| Step (`step`) | Steps, the stable maximum (`peak_performance`), the capacity forecast | Two steps. The ceiling (`stable_window`): the last step on which the latency and error thresholds hold. The target step (`target_window`): the first step at `target_rps` (2 % below still counts) on which they hold; CPU and memory are checked on it. With `target_margin_pct` that step is the first one at the target plus the margin | No target step: `fail`; CPU or memory over the threshold on the target step, or the margin not confirmed: `risk`; otherwise `pass`. Breaches above the target step are headroom (`headroom`) and never lower the verdict, so a run that goes further never gets a worse one. Without `target_rps`: latency or errors on the ceiling `fail`, CPU or memory `risk` |
| Soak (`soak`) | The hold window without ramp-up and ramp-down, average and minimum RPS per interval, trends of p95, errors, CPU and memory (`stability_under_load`, `AI/stability.py`) | The hold window; `target_rps` must be held: the average not more than 5 % below, no interval more than 15 % below | Any latency, error or hold violation: `fail`; CPU, memory or an adverse trend: `risk` |
| Spike (`spike`) | Spike windows (RPS above the baseline by 1.5x for longer than a minute), overshoot of p95 and errors against the baseline, recovery time (`spike_analysis`, `AI/spike.py`) | The spike windows, the worst value per threshold; `target_rps` is the spike level; a separate "recovery" criterion | Level not reached or latencies not back to the baseline within 15 min: `fail`; latency and errors inside the spike: `risk`; no spike found: `no_data` |
| Stress (`stress`) | As step, plus the saturation point, the first degraded step, the degradation pattern and the limiting resource (`saturation`, `AI/saturation.py`) | As step | As step |
| Scenario (`scenario`) | Phases from the request groups of the load tool (query role `lt.rps_by_request`): each interval goes to the group with the largest share of RPS, segments shorter than `phase_min_minutes` join their neighbours, the nesting level of k6 groups is chosen automatically; the phases replace the load steps; the phases of the previous scenario run of the service, aligned by their group keys (`scenario`, `AI/phases.py`) | Every phase, the worst phase per threshold; `target_rps` does not apply; `phase_completeness` (phases of the previous run missing in this one) and `phase_duration` (the largest change of a phase duration beyond the data step, against `phase_duration_tolerance_pct`) | Latency or errors over the threshold in a phase: `fail`; CPU, memory, a missing phase or a changed duration: `risk` |

Peak performance is reported only where the total RPS series exists; for soak, spike and scenario it is marked
`not_applicable`. The forecast is built for step and stress tests. It takes the ceiling (`stable_window`) from the
report: steps above it are marked red, stay out of the model and set the load limit, so a run that went past its
SLA maximum is not shown as safe there. The verdict policy per mode is documented in the docstring of
`AI/sla_evaluator.py`.

## The baseline run

The baseline of a report is the run the final prompt and the regression checks compare with. `llm.baseline.mode`
chooses it (`AI/release_marks.find_baseline_run`):

| Mode | Baseline |
|---|---|
| `pinned` (default) | The pinned baseline of the service and test type (archive or report page, «Pin as baseline»), unless it is the run itself; else as `previous_release` |
| `previous_release` | The accepted run of another release of the same service and test type whose report is earlier than the test (the release of the run itself is skipped); else as `previous_success` |
| `previous_success` | The previous run of the service (within the project area) with the effective verdict `pass` |
| `previous` | The previous run of the service, any verdict |

The release modes look in the same service only; without release marks `pinned` is the old `previous_success`. The
context of the model names the baseline (`mode`, `release`, `basis`: the pinned baseline, the accepted run of a
release, or a note that neither was found).

## Regression against the baseline

After the verdict of the mode the report compares its own measured values with the baseline run of the analysis
(`llm.baseline`, the same run the final prompt sees) and adds deterministic checks (`AI/regression.py`). The
thresholds live in the SLA of the service (`regression_*` keys, see the configuration reference); with all six empty
there are no such checks and the verdict is exactly the verdict of the mode.

| Check | Value | Compared at |
|---|---|---|
| `regression_latency` | Growth of p95 in percent of the baseline (p99 when neither run has p95) | Step and stress: the target step of both runs (the lower one), else the lower of the two ceilings, the row of that step in the step table (2 % tolerance); soak: the hold; spike: the highest spike |
| `regression_errors` | Growth of the error rate in percentage points | As `regression_latency` |
| `regression_capacity` | Drop of the load reached in percent: the ceiling of a step or stress test, the spike level of a spike test | Not compared for soak tests |

- Two runs are comparable when their levels differ by no more than 5 %: the hold of a soak test, the spike of a spike
  test, or the ceilings of step reports saved without a step row at that level (older reports; then the values of the
  checks are used). The step table stores the error rate of every step (`error_rate`, the plateau p95 of the
  `error_rate_query` series, the statistic of the SLA check) for this comparison.
- The baseline must be the same service and the same test type; scenario tests are not compared (they have their own
  reference run, `phase_completeness` and `phase_duration`).
- A delta over the risk threshold fails its check with `severity = warning` and turns `pass` into `risk`; over the
  failure threshold `severity = critical` turns `pass` and `risk` into `fail`. `fail` and `no_data` stay. Soak and
  spike tests keep treating a warning-level regression like a CPU or memory overrun.
- The result is fixed when the report is built: `sla_details.regression` keeps the baseline run (name, id, release,
  how it was chosen), the level, every delta and the metrics that could not be compared with the reason. Pinning
  another baseline later does not change the report; the release page shows live deltas with the same functions.
- The report shows the deltas as rows of the SLA table (units `%` and pp) and a line «Baseline: run · how it was
  chosen · Compare» under the verdict summary; without a comparison the line names the reason. The Confluence page
  has the same rows and line.

## Candidates, judge and verification

- **Judge.** A compact form-filling scheme: `scores[].rubric` plus `factual`, `completeness`, `specificity` and
  `overall`. The rubric file is chosen by domain (`AI/prompts/<lang>/judge_rubrics/{jvm,database,kafka,microservices,
  hard_resources,lt_framework,lt_generators,application_logs,final}.txt` with `common.txt`). The final choice is
  hybrid: `0.6 * judge_overall + 0.35 * data_score + 0.05 * confidence`.
- **One candidate.** With `self_consistency_k = 1`, or when the other calls failed, the judge is skipped
  (`judge_meta.skipped = true`); the confidence level shown in the report comes from the data check alone.
- **Judge unavailable.** If the judge fails or scores only some candidates, its absence is not counted as zero:
  candidates are ranked by `(0.35 * data + 0.05 * confidence) / 0.4`, and `judge_meta` carries `unavailable = true`
  with a `reason` (`error`, `no_scores`, `partial_scores`).
- **Verification of numbers.** Every number in the findings is looked up in the collected data; the "Answer
  confidence" block of the report lists which numbers were confirmed and how the answer was chosen. With
  `llm.verification_pass` an extra call can revise an analysis that contains unverified findings.
  The verification count refers to findings. LoadLens builds the final verdict explanation from the
  deterministic SLA checks, including their configured thresholds, units and measured results. Older
  reports identify their saved AI rationale as unchecked.
- **Small context windows.** OpenAI-compatible requests reserve tokens for the reply and chat framing.
  LoadLens estimates input size conservatively using UTF-8 byte length. It uses shorter report instructions
  and reduces JSON evidence when necessary. The judge can use a shorter scoring form while keeping all
  candidate indexes. An infeasible request or a context rejection from the server is recorded in the
  report, and the deterministic SLA remains available.
- **Language match.** The share of words in the expected script is the first key when candidates are compared, so
  a provider that answers in the wrong language loses to one that answers in the report language.
- **Feedback.** Findings can be voted "Correct / Incorrect" on the report page (`llm_feedback`); the votes do not
  change the analysis, they are kept with the report. `AI/eval.py` computes quality metrics over saved runs
  (verified share, language match, report language).

## Verdict codes and the result format

Verdicts are codes everywhere in the data and the API: `pass`, `risk`, `fail`, `no_data` (`AI/verdicts.py`). The UI
labels are Passed / Risks found / Failed / Insufficient data; reports saved earlier with Russian labels are read as
codes without a migration, and the API accepts both codes and legacy labels on input. Each domain block and the
overall block contain:

| Field | Meaning |
|---|---|
| `verdict`, `verdict_rationale` | Verdict code and explanation. In the final report both come from the SLA evaluation. Domain analyses use the model's explanation. |
| `confidence` | The model's confidence, 0..1 |
| `findings` | Strings or objects `{summary, severity, component, evidence}`; `evidence_items[].threshold` is a number with a unit or `null` |
| `recommended_actions`, `affected_components` | Concrete actions; an optional list of components |
| `peak_performance` | `{max_rps, max_time, drop_time, method}` where the total RPS exists; `{not_applicable: true, reason}` for soak and spike |

The overall block adds fields computed by the pipeline, not the model: `report_language` (`ru` or `en`; missing
means `ru`), `test_profile`, `deterministic_sla` (`verdict`, `checks[]`, `stable_window` with its `kind`: step,
`hold` or `spike`; `regression` with the baseline run, the level and the deltas when regression thresholds are set),
`stability_under_load` (soak), `spike_analysis` (spike), `saturation` (stress) and
`domain_titles` for domains whose title comes from the installed packs. The report verdict shown in the UI is the
SLA verdict; the model's verdicts are per domain. An engineer can override the report verdict with a reason
(`PUT /runs/<run_name>/verdict`); the analysis itself does not change.

## Prompts and languages

The language of reports and prompts is the installation setting `appearance.language` (`ru` by default; "Settings →
Appearance" or the first step of the wizard). It is independent of the interface language of the browser. A report
job takes the language when it is queued, so a change of the setting does not affect running jobs, and the whole
run executes inside `AI.i18n.with_locale(report_language)`: candidate, critic, judge and verification prompts, the
default system prompt and the deterministic texts are in one language. The report JSON stores `report_language`.

Prompt files live in `AI/prompts/ru/` and `AI/prompts/en/` with identical names:

| File | Role |
|---|---|
| `<domain>_prompt.txt`, `overall_prompt.txt` | Domain prompts and the final prompt; editable in "Settings → Prompts" |
| `_context_guide.txt`, `_response_format.txt`, `_no_sla.txt` | Shared guidance, the strict JSON format, the note used when no SLA is set |
| `_system.txt`, `_system_json.txt` | The default system prompt and its JSON part |
| `profiles/{step,soak,spike,stress,none}.txt` | The test profile paragraph of the guidance |
| `critic_prompt.txt`, `verify_prompt.txt`, `judge_prompt.txt`, `judge_safe_prompt.txt`, `judge_rubrics/*.txt` | Critic, verification pass, judge (with and without a rubric) and the rubrics |
| `forecast_limit_prompt.txt`, `compare_prompt.txt`, `context_import_prompt.txt` | Explanation of the forecast limit; explanation of a comparison of two runs; import of a document into the system context (not editable in the UI) |
| `_strings.json` | Short glue strings (headings of the guidance block, verdict notes, fallback rationale, document separators), the same keys in both languages |

The guidance block that precedes a domain prompt is built from the settings: the SLA, the rules of the system
context, the **domain metrics**, the `prompt_addendum` notes of the installed packs and the test profile. The domain
metrics list every configured query of the domain with its meaning, unit, series keys and note, and mark the queries
that returned no data in the run, so the domain prompt files do not list the metrics themselves. A query added by
hand gets its meaning and unit from the role picked in its row editor ("Settings → Queries") and a note for the model
from the "Note for the AI analysis" field; the note replaces the description of a pack query with the same name. A
query without a role is listed as "meaning not set", and the model is told not to guess its unit.

The loader `AI/prompt_store.py` caches by language and name; when a file is missing in one language it falls back to
`ru` with one warning. Overrides from "Settings → Prompts" apply per area and per service; an empty text or a copy of
the default of either language is not an override, and neither is a stored copy of an earlier default
(`PREVIOUS_DEFAULTS`: the domain prompts before the domain metrics replaced their lists of sections). An override written in another language is allowed (the editor
warns and offers the default of the report language); the response format in the report language is appended after
the user's text. `GET /prompts` returns `language`, `sources` (`default`, `area`, `service`) and
`override_language`; `GET /prompts/defaults?lang=ru|en` returns the defaults; `GET /prompts/history` the versions.

Editing prompts: change the file in both folders, keep the structure, the placeholders (`{answer_*}`,
`{system_context_brief}`, `{{CANDIDATE}}`, `{{DOMAIN_KEY}}`, `{{JUDGE_RUBRIC}}`, `{{DATA_CONTEXT}}`,
`{{CANDIDATES_JSON}}`), the JSON keys and the rubric criteria identical; write natural English rather than a literal
translation. `tests/test_prompt_store.py` checks the parity of the two folders and `tests/test_i18n_prompts.py` that
every model call of a run uses one language.

## System context

`system_context` is a structured description of the system under test: what the system is and why the test runs
(`system`), components, dependencies and data stores (`architecture`), entry points, critical flows and expected
hotspots (`load_model`), known constraints, risks, normal degradation rules and the desired focus of the analysis
(`operational_context`). A brief of it goes into every prompt and a snapshot is stored with the report. It is set per
area and per service (the service context replaces the area context entirely).

"Fill from a document" in "Settings → System context" accepts a PDF with a text layer, Word (`.docx`), Markdown or
text up to 10 MB, pasted text, or the URL of a Confluence page read with the `confluence` connection (the same host
only; attachments and child pages are not read). The server extracts up to 120 000 characters (PDF up to 200 pages)
and the model of the area's `llm` settings maps them onto the fields in one call. "Fill" only completes empty fields
and adds missing items; "Replace" takes everything from the document. The draft is shown with the changed fields
highlighted and is not saved until you click Save. The audit log records the file name, size, area and service,
never the text. Limit: `limits.llm_context_imports_per_user_per_day`.

## Providers

| `llm.provider` | Notes |
|---|---|
| `openai` | Any OpenAI-compatible chat completions API: `api_base_url`, `model`, `api_key`. Local servers with the same API work the same way; the demo uses a stub |
| `anthropic` | Messages API with `api_base_url`, `model`, `api_key` |
| `perplexity` | `disable_web_search` keeps the analysis on the data; an optional `language_preference` is passed when set and logs one warning per run when it differs from the report language |
| `gigachat` | Through `langchain_gigachat` with mTLS (`cert_file`, `key_file`, mounted from `AI/GigaChat/` in Docker) or an API key; `max_attempts`, `request_min_interval_sec` and `rate_limit_cooldown_sec` handle its rate limits. The model is oriented to Russian: for English reports, candidates in the wrong language are filtered out by the language match |

Common provider keys and throttling (`max_concurrent`, `generation`, timeouts, `llm.self_consistency`) are listed
in [configuration.md](configuration.md#llm). `max_concurrent` applies per process: with a Celery worker the provider
sees up to `WORKER_CONCURRENCY x max_concurrent` parallel requests. "Settings → Connections → LLM" tests the
provider and lists its models (`POST /config/llm_models`). The metrics summaries, the system context and, for the
context import, the document text are sent to the provider you configure; nothing else leaves the installation.

## Other model calls and limits

- **Forecast explanation.** "Explain with AI" on the forecast page links the USL limit to the findings of the report
  (`forecast_limit_prompt.txt`), in the language of the report; the answer is cached per report
  (`forecast_explanations`). Limit: `limits.llm_explanations_per_user_per_day`.
- **Comparison explanation.** On the compare page, A is the assessed run and B the baseline. The program computes
  the outcome (`better`, `worse`, `same`, `mixed`, `incomparable`): the final verdicts first, then the maximum RPS of
  a step test (5 % tolerance), then the p95 metrics whose change of A relative to B reaches 10 %. Whether a change is
  better or worse follows the role of the query (`AI/metric_direction.py`), else the words of its label. Caveats
  (other service or test type, windows of different length, a baseline newer than A, a run without a report or
  without verified findings, other load) are listed beside it. "Analyse with AI" sends these facts and up to 8
  verified findings of each run to the model (`compare_prompt.txt`), which explains the differences without changing
  the outcome. LoadLens checks each value attributed to A or B against that run's value for the named metric.
  The comparison flags swapped values and identifies the affected run and metric. The explanation uses the language of report A and
  is cached per pair (`compare_explanations`); "Add comparison to Confluence" publishes the outcome, the verdicts, the
  largest changes and the cached explanation (or the findings of both runs) as its own page. Limit:
  `limits.llm_explanations_per_user_per_day`.
- **Reports without AI.** `use_llm: false` builds the deterministic report: the SLA verdict and checks, the load
  steps, the peak RPS with its peak and degradation times or the test-type figures (saturation, spikes, stability),
  the charts and the engineer summary. The report page and the Confluence page mark it "no AI analysis" and have
  no analysis tabs; its verdict can be overridden and the forecast opens as for any report.
- **Budgets.** `limits.llm_reports_per_user_per_day` counts reports with analysis; when it is exhausted, a report
  started from the "Run a test" page is built without the analysis.

## Troubleshooting

- **Empty domain blocks.** Check the sources, the test window and `default_params.resample_interval`; the step
  detector needs at least `sla.min_stable_minutes` of a plateau.
- **Slow model answers.** Lower `generation.max_tokens` or the parallelism; for unstable providers use
  `request_min_interval_sec` and `rate_limit_cooldown_sec`.
- **HTTP 429.** LoadLens pauses calls to the same API connection within the worker process and uses `Retry-After`
  when the provider sends it. A wait longer than 60 seconds or a reported quota error ends model analysis for that
  connection in the current report. The report keeps its SLA, figures, charts and any completed domain analyses.
- **The judge did not run.** Expected with one candidate; otherwise look for `judge_meta.unavailable` and its reason
  in `GET /llm_reports`.
- **Mixed languages in a report.** The report language is fixed when the job is queued; a prompt override in the
  other language is appended before the format block. The report page shows a "Report language" badge when the
  stored language differs from the interface language.
