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
   service (`llm.baseline`), and returns the verdict rationale, findings and recommendations. Everything is saved to
   `llm_reports` with the quality metrics (`scores`).

Progress is reported per phase (collecting each domain, each model answer, the SLA check, the final analysis,
saving); the job page shows it with the elapsed and the usual time. The entry point is
`AI.pipeline.uploadFromLLM(start_ts, end_ts, ...)` (`AI.main.uploadFromLLM` is a proxy).

## Test types and deterministic analysis

The test type of a report (`step`, `soak`, `spike`, `stress`; `AI/test_profile.py` maps aliases, including the
Russian names) selects the analysis mode, the SLA window and the profile text added to the prompts.

| Type | Computed in code | SLA window and the meaning of `target_rps` | Verdict rules |
|---|---|---|---|
| Step (`step`) | Steps, the stable maximum (`peak_performance`), the capacity forecast | The last step on which the latency and error thresholds hold; `target_rps` must be reached on it | Target reached: `pass`; violations on that step: `risk` |
| Soak (`soak`) | The hold window without ramp-up and ramp-down, average and minimum RPS per interval, trends of p95, errors, CPU and memory (`stability_under_load`, `AI/stability.py`) | The hold window; `target_rps` must be held: the average not more than 5 % below, no interval more than 15 % below | Any latency, error or hold violation: `fail`; CPU, memory or an adverse trend: `risk` |
| Spike (`spike`) | Spike windows (RPS above the baseline by 1.5x for longer than a minute), overshoot of p95 and errors against the baseline, recovery time (`spike_analysis`, `AI/spike.py`) | The spike windows, the worst value per threshold; `target_rps` is the spike level; a separate "recovery" criterion | Level not reached or latencies not back to the baseline within 15 min: `fail`; latency and errors inside the spike: `risk`; no spike found: `no_data` |
| Stress (`stress`) | As step, plus the saturation point, the first degraded step, the degradation pattern and the limiting resource (`saturation`, `AI/saturation.py`) | As step | As step |

Peak performance is reported only where the total RPS series exists; for soak and spike it is marked
`not_applicable`. The forecast is built for step and stress tests. The verdict policy per mode is documented in the
docstring of `AI/sla_evaluator.py`.

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
  `llm.verification_pass` an extra call rebuilds the verdict from the confirmed findings only.
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
| `verdict`, `verdict_rationale` | The code and a short explanation: a conclusion plus two to four key factors |
| `confidence` | The model's confidence, 0..1 |
| `findings` | Strings or objects `{summary, severity, component, evidence}`; `evidence_items[].threshold` is a number with a unit or `null` |
| `recommended_actions`, `affected_components` | Concrete actions; an optional list of components |
| `peak_performance` | `{max_rps, max_time, drop_time, method}` where the total RPS exists; `{not_applicable: true, reason}` for soak and spike |

The overall block adds fields computed by the pipeline, not the model: `report_language` (`ru` or `en`; missing
means `ru`), `test_profile`, `deterministic_sla` (`verdict`, `checks[]`, `stable_window` with its `kind`: step,
`hold` or `spike`), `stability_under_load` (soak), `spike_analysis` (spike), `saturation` (stress) and
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
| `forecast_limit_prompt.txt`, `context_import_prompt.txt` | Explanation of the forecast limit; import of a document into the system context (not editable in the UI) |
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
- **Reports without AI.** `use_llm: false` builds the deterministic report only: SLA, steps, charts; such reports have
  no analysis tabs and no overall verdict to override.
- **Budgets.** `limits.llm_reports_per_user_per_day` counts reports with analysis; when it is exhausted, a report
  started from the "Run a test" page is built without the analysis.

## Troubleshooting

- **Empty domain blocks.** Check the sources, the test window and `default_params.resample_interval`; the step
  detector needs at least `sla.min_stable_minutes` of a plateau.
- **Slow model answers.** Lower `generation.max_tokens` or the parallelism; for unstable providers use
  `request_min_interval_sec` and `rate_limit_cooldown_sec`.
- **The judge did not run.** Expected with one candidate; otherwise look for `judge_meta.unavailable` and its reason
  in `GET /llm_reports`.
- **Mixed languages in a report.** The report language is fixed when the job is queued; a prompt override in the
  other language is appended before the format block. The report page shows a "Report language" badge when the
  stored language differs from the interface language.
