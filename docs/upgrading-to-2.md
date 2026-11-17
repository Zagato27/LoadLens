# Upgrading to LoadLens 2.0

LoadLens 2.0 adds release tracking and baseline regression, scenario phases, Loki logs and comparison explanations.
It also changes configuration and comparison API behavior from 1.0. Review the differences below before updating
the application and worker together. The full feature list is in [CHANGELOG.md](../CHANGELOG.md).

## Comparison API: A relative to B

The `trend_pct` field of `/compare_summary` and `/compare_metric_summary` now measures the assessed run **A**
against baseline **B**:

```text
trend_pct = (A - B) / abs(B) * 100
```

For example, A = 120 ms and B = 100 ms returns +20%, meaning A is slower. The previous calculation measured B
against A and returned approximately -16.7%. Update API consumers, signs, labels and alert conditions; changing
only the sign of an old value is insufficient because the denominator also changed. The UI applies metric roles
to decide whether an increase is better or worse. Handle unavailable percentages rather than dividing by a zero
baseline yourself.

The release-publication endpoint introduced in development builds is now `GET /release_publications/<name>`.
`POST /releases/<name>/confluence` still publishes a page. The separate GET route leaves release names ending
in `/confluence` available to the HTML release page. Release-mark requests can send `accepted: false` to return
a manually accepted run to automatic selection; omitting the field keeps the current manual choice.

## Automatic step detector

The detector derives its confirmation window from the sampling cadence and its step threshold from plateau
noise. These old SLA settings are ignored with a log warning and removed when the SLA block is saved:

- `step_detection_preset`
- `step_detection_resample_sec`, `step_detection_smooth_sec`
- `step_confirm_hold_sec`, `step_min_step_delta_rps`, `step_min_step_delta_pct`
- `step_max_cv`, `step_max_slope_rps_per_min`, `step_max_within_step_drop_pct`, `step_drop_hold_sec`

`step_detection_enabled` and `min_stable_minutes` remain available. Review a representative report with the
sampling interval used by your installation; the automatically detected plateaus may differ from 1.0.

## Report and baseline behavior

- Step/stress SLA evaluates the target step separately from the capacity ceiling. Resource saturation above an
  achieved target describes headroom instead of automatically failing that target. `sla.target_margin_pct` can
  request additional headroom.
- The default baseline mode is `pinned`: pinned run, otherwise the accepted run of the previous release,
  otherwise the previous successful run. An explicitly configured baseline mode remains in effect.
- Regression checks use the configured `sla.regression_*` thresholds. They do not activate when all thresholds
  are unset. The example configuration enables risk thresholds of +10% latency, +0.5 percentage points errors
  and -10% capacity; review these values before adopting them in an existing service.
- Existing saved verdicts and regression snapshots are not recomputed by an upgrade or a change of baseline.
  Rebuilding a report applies the current algorithms and settings. Reports created with `use_llm=false` now
  contain a full deterministic final result; old metrics-only runs remain readable.
- Accepted runs and pinned baselines are protected from deletion. API clients must handle HTTP 409
  `release_protected` and require a deliberate override before using `force=1`.

## Settings and storage

Runtime `data_sources` and `domain_sources` replace the corresponding base mappings rather than merging deleted
entries back from `settings.py`. Check the saved catalogue and domain bindings after the upgrade. Per-query source
overrides and `null` inheritance are described in [data-sources.md](data-sources.md#a-source-per-query-row).

The application creates the release and comparison tables on first use. Its database role needs the schema/table
permissions documented in [deployment.md](deployment.md). `docker/initdb/02_schema.sql` prepares a new database;
container initialization scripts are not rerun automatically for an existing database volume.

Release publications are identified by release name and project area. If upgrading an intermediate development
build that already has `release_publications`, the lazy migration preserves old rows under the unscoped area
(`''`). A publication from a specific area creates a separate page; an old page is not silently assigned to that
area or overwritten. The old Confluence page itself remains available.

## Update sequence

1. Back up the database and the live configuration files using your installation's existing backup procedure.
2. Let report jobs and load tests finish. Record the running version and image tag or commit.
3. Apply the configuration/API adjustments above, preserving existing secrets and service-specific settings.
4. Update both the application and the Celery worker to the same release, following
   [deployment.md](deployment.md). Do not replace live settings with the example configuration.
5. Check `/healthz`, sign-in, `/about` (2.0.0), existing reports and a representative new report. Verify source
   bindings, the chosen baseline and any external comparison API consumers.

The systemd deploy helper deploys a committed and pushed revision; uncommitted local files do not reach the
server. Its rollback restores the previous code revision, not database/configuration changes. Use the backup
when a full rollback is needed, and validate compatibility before running an older application against upgraded
storage. Preparing a release locally does not publish its image or GitHub release; pushing its version tag starts
the release workflow.
