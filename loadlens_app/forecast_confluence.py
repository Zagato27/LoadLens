# SPDX-License-Identifier: Apache-2.0
"""Capacity forecast of a run as its own Confluence page, apart from the test report page.

The page and its two pictures are written in the language of the run's report (``report_language_of``: ``ru`` for
reports saved before the field), like the report page; the job's progress messages stay in the job's locale.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Sequence

from psycopg2 import sql

from AI.capacity_forecast import StepPoint
from AI.i18n import LocalizedError, fmt_date, fmt_number, get_locale, msg, plural, t, with_locale
from AI.pipeline import _time_shift_hours
from AI.resource_plan import ScaleMark, ScaleZone, ServicePlan
from loadlens_app.confluence_export import (
    ConfluenceClient,
    ProgressCallback,
    attachment_image,
    confluence_config,
    h2,
    h3,
    kv_table,
    li_list,
    p,
    upsert_page,
    validate_confluence_config,
    wrap_expand,
    wrap_macro,
    xml_text,
)
from loadlens_app.core import _ts_conn
from loadlens_app.forecast import ForecastReport, _storage, build_forecast, report_language_of
from loadlens_app.forecast_explanation import ExplanationView, explanation_view

DEFAULT_PUBLICATIONS_TABLE = "forecast_publications"
SCALE_PNG = "forecast-scale.png"
STEPS_PNG = "forecast-steps.png"
RESOURCE_ROWS = 5
RELATED_FINDINGS = 5
SCALE_LABEL_GAP = 0.22
ZONE_TEXT_SHARE = 0.1

# Codes of the calculator answer, confidence, CPU ceiling source and scaling verdict; titles are capacity.<group>.<code>.
VERDICT_PANELS = {"holds": "tip", "scale": "note", "tight": "note", "blocked": "warning"}
SCALING_PANELS = {"helps": "tip", "partly": "note", "no": "warning"}
ZONE_COLORS = {"ok": "#22c55e", "warn": "#eab308", "fail": "#ef4444"}
VERDICT_COLORS = {"holds": "#15803d", "scale": "#a16207", "tight": "#a16207", "blocked": "#b91c1c"}
STEP_COLORS = {"ok": "#22c55e", "fail": "#ef4444", "muted": "#9aa0a6"}
# Digit grouping of whole RPS values: a no-break space in ru, a comma in en.
_GROUP_SEPARATORS = {"ru": "\u00a0", "en": ","}


class ScenarioError(LocalizedError, ValueError):
    """A scenario of the page got no calculator answer."""


@dataclass(frozen=True)
class ForecastScenario:
    title: str
    rps: float
    verdict: str
    action: str
    confidence: str


@dataclass(frozen=True)
class ForecastPublication:
    run_id: str
    page_id: str
    page_url: str
    updated_at: Optional[str]


def _progress(callback: ProgressCallback | None, message: str, pct: int | None = None) -> None:
    if callback:
        callback(message, pct)


def verdict_title(code: str) -> str:
    """Answer of the calculator for a target (holds / scale / tight / blocked) in the current locale."""
    return t(f"capacity.verdict.{code}")


def confidence_title(code: str) -> str:
    return t(f"capacity.confidence.{code}")


def ceiling_source_title(code: str) -> str:
    return t(f"capacity.ceiling_source.{code}")


def scaling_title(code: str) -> str:
    return t(f"capacity.scaling.{code}")


def _grouped(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", _GROUP_SEPARATORS.get(get_locale(), "\u00a0"))


def _rps(value: float) -> str:
    return f"{_grouped(value)} RPS"


def _approx_rps(value: float) -> str:
    digits = max(0, int(math.floor(math.log10(abs(value)))) - 2) if value else 0
    return f"≈ {_rps(round(value, -digits))}"


def _decimal(value: float, digits: int) -> str:
    return fmt_number(value, digits)


def _percent(value: float, digits: int) -> str:
    """``value`` (already in percent) with the percent sign of the locale."""
    return t("capacity.unit.pct", value=_decimal(value, digits))


def _pct(share: float) -> str:
    value = share * 100.0
    return _percent(value, 1 if abs(value) < 10 else 0)


def _ms(value: float) -> str:
    if value >= 1000:
        return t("capacity.unit.seconds", value=_decimal(value / 1000.0, 1))
    return t("capacity.unit.ms", value=_decimal(value, 1 if value < 10 else 0))


def _hhmm(iso: str) -> str:
    return iso[11:16] if len(iso) >= 16 else iso


def _pods(count: int) -> str:
    return plural("capacity.pods", count)


def _step_word(report: ForecastReport) -> str:
    buckets = bool(report.steps) and all(step.source == "time_buckets" for step in report.steps)
    return t("capacity.unit.interval") if buckets else t("capacity.unit.step")


def _scaled(report: ForecastReport) -> list[ServicePlan]:
    return [item for item in report.resources.services if item.instances_needed is None or item.instances_needed > item.instances]


def forecast_page_title(report: ForecastReport) -> str:
    return t("confluence.forecast.page_title", run=report.run_name, service=report.service)


def _scale_action(item: ServicePlan) -> str:
    if item.instances_needed is None:
        return t("confluence.forecast.stuck", name=item.name, cpu=_pct(item.cpu_base))
    params = {"name": item.name, "instances": item.instances, "pods": _pods(item.instances_needed)}
    if item.cpu_after is not None:
        return t("confluence.forecast.add_cpu", after=_pct(item.cpu_after), current=_pct(item.cpu_at_target), **params)
    return t("confluence.forecast.add", **params)


CAUSE_KEYS = {"instability": "confluence.forecast.cause_instability", "sla": "confluence.forecast.cause_sla"}


def _broken_checks(report: ForecastReport) -> str:
    """Titles of the SLA checks the report says broke on the step after its maximum."""
    ceiling = report.sla_ceiling
    return ", ".join(check.title or check.name for check in ceiling.degraded_checks) if ceiling is not None else ""


def _limit_line(report: ForecastReport) -> str:
    limit = report.limit
    plan = report.resources
    if limit.rps is None:
        return t("confluence.forecast.limit_none")
    if not report.limit_applies and plan.limit_cause and plan.limit_cpu is not None:
        key = CAUSE_KEYS.get(limit.kind, "confluence.forecast.cause_model")
        return t(key, rps=_approx_rps(limit.rps), service=plan.limit_cause, cpu=_pct(plan.limit_cpu))
    if plan.status == "ok":
        return t("confluence.forecast.limit_not_cpu", rps=_approx_rps(limit.rps))
    if limit.kind == "instability":
        return t("confluence.forecast.limit_instability", rps=_approx_rps(limit.rps))
    if limit.kind == "sla":
        checks = _broken_checks(report)
        if checks:
            return t("confluence.forecast.limit_sla_checks", rps=_approx_rps(limit.rps), checks=checks)
        return t("confluence.forecast.limit_sla", rps=_approx_rps(limit.rps))
    return t("confluence.forecast.limit_model", rps=_approx_rps(limit.rps))


def _safe_line(report: ForecastReport, headroom_pct: float) -> str:
    safe = report.safe
    if safe.rps is None:
        return ""
    if safe.source == "cpu":
        return t("confluence.forecast.safe_cpu", rps=_rps(math.floor(safe.rps)), service=safe.service)
    return t("confluence.forecast.safe_headroom", headroom=_percent(headroom_pct, 0), rps=_rps(math.floor(safe.rps)))


def _verdict_lines(report: ForecastReport, headroom_pct: float) -> list[str]:
    answer = report.answer
    target = report.target
    plan = report.resources
    lines = [_scale_action(item) for item in _scaled(report)] if answer is not None else []
    if lines and plan.scaled_capacity_rps is not None:
        lines.append(t("confluence.forecast.after_scaling", rps=_approx_rps(plan.scaled_capacity_rps), service=plan.next_bottleneck))
    if answer is not None and answer.verdict == "tight" and target is not None and target.headroom_pct is not None:
        lines.append(t("confluence.forecast.tight", headroom=_percent(target.headroom_pct, 0), required=_percent(headroom_pct, 0)))
    if answer is not None and answer.verdict == "holds" and plan.status == "ok" and plan.services:
        top = plan.services[0]
        lines.append(t("confluence.forecast.busiest", name=top.name, cpu=_pct(top.cpu_at_target), ceiling=_pct(top.cpu_ceiling)))
    lines.extend(line for line in (_limit_line(report), _safe_line(report, headroom_pct)) if line)
    return lines


def _verdict_panel(report: ForecastReport, headroom_pct: float) -> str:
    if report.answer is None or report.target is None:
        return wrap_macro("info", p(t("confluence.forecast.no_target")))
    answer = report.answer
    title = t(
        "confluence.forecast.verdict_title",
        rps=_rps(report.target.rps),
        verdict=verdict_title(answer.verdict),
        confidence=confidence_title(answer.confidence),
    )
    body = f"<p><strong>{xml_text(title)}</strong></p>{li_list(_verdict_lines(report, headroom_pct))}"
    return wrap_macro(VERDICT_PANELS[answer.verdict], body)


def _table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header = "".join(f"<th>{xml_text(cell)}</th>" for cell in head)
    body = "".join("<tr>" + "".join(f"<td>{xml_text(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _scenario_action(report: ForecastReport) -> str:
    answer = report.answer
    if answer is not None and answer.verdict == "blocked" and report.limit_applies and report.limit.rps is not None:
        return t("confluence.forecast.scenario_blocked", rps=_approx_rps(report.limit.rps))
    scaled = _scaled(report)
    if not scaled:
        return t("confluence.forecast.scenario_nothing")
    return "; ".join(
        t("confluence.forecast.scenario_stuck", name=item.name) if item.instances_needed is None
        else f"{item.name} {item.instances} → {item.instances_needed}"
        for item in scaled
    )


def _scenario_targets(report: ForecastReport) -> list[tuple[str, float]]:
    base, key = (
        (report.sla_target_rps, "confluence.forecast.plus_sla") if report.sla_target_rps
        else (report.tested_rps, "confluence.forecast.plus_tested")
    )
    targets: list[tuple[str, float]] = []
    if report.sla_target_rps:
        targets.append((t("confluence.forecast.sla_target"), report.sla_target_rps))
    targets.append((t(key, pct=25), float(round(base * 1.25))))
    targets.append((t(key, pct=50), float(round(base * 1.5))))
    if report.safe.rps is not None:
        targets.append((t("confluence.forecast.safe_max"), float(math.floor(report.safe.rps))))
    return targets


def forecast_scenarios(run_id: str, report: ForecastReport, headroom_pct: float, cpu_ceiling_pct: Optional[float]) -> list[ForecastScenario]:
    """Verdict and what to add for the SLA goal, +25 %, +50 % and the safe load, with the same settings."""
    scenarios: list[ForecastScenario] = []
    for title, rps in _scenario_targets(report):
        scenario = build_forecast(run_id, rps, headroom_pct, cpu_ceiling_pct)
        if scenario.answer is None:
            raise ScenarioError("confluence.forecast.no_verdict", title=title, rps=rps)
        scenarios.append(ForecastScenario(
            title=title,
            rps=rps,
            verdict=verdict_title(scenario.answer.verdict),
            action=_scenario_action(scenario),
            confidence=confidence_title(scenario.answer.confidence),
        ))
    return scenarios


def _scenarios_table(scenarios: Sequence[ForecastScenario]) -> str:
    rows = [(f"{item.title} — {_rps(item.rps)}", item.verdict, item.action, item.confidence) for item in scenarios]
    head = (
        t("confluence.forecast.col_load"),
        t("confluence.forecast.col_verdict"),
        t("confluence.forecast.col_action"),
        t("confluence.forecast.col_confidence"),
    )
    return _table(head, rows)


def _resource_rows(report: ForecastReport) -> list[ServicePlan]:
    scaled = _scaled(report)
    others = [item for item in report.resources.services if item not in scaled]
    return scaled + others[: max(0, RESOURCE_ROWS - len(scaled))]


def _pods_table(report: ForecastReport) -> str:
    plan = report.resources
    if plan.status != "ok":
        return p(plan.message)
    rows = [
        (
            item.name,
            str(item.instances),
            _pct(item.cpu_measured),
            _pct(item.cpu_at_target),
            t("confluence.forecast.not_by_pods") if item.instances_needed is None else str(item.instances_needed),
        )
        for item in _resource_rows(report)
    ]
    head = (
        t("confluence.forecast.service"),
        t("confluence.forecast.col_pods"),
        t("confluence.forecast.col_cpu_step", unit=_step_word(report), number=plan.last_step),
        t("confluence.forecast.col_cpu_target"),
        t("confluence.forecast.col_needed"),
    )
    note = p(t("confluence.forecast.ceiling_note", ceiling=_percent(plan.ceiling_pct, 0), source=ceiling_source_title(plan.ceiling_source)))
    return _table(head, rows) + note


def _check_text(view: ExplanationView) -> str:
    check = view.check
    if check is None:
        return ""
    if check.status == "verified":
        return t("confluence.forecast.check_verified", matched=check.claims_matched, total=check.claims_total)
    if check.status == "unverified":
        return t("confluence.forecast.check_unverified", numbers=", ".join(check.unmatched))
    return t("confluence.forecast.check_none")


def _findings_fallback(view: ExplanationView) -> str:
    reason = f"{view.message}." if view.message else t("confluence.forecast.no_explanation")
    if not view.findings:
        return p(t("confluence.forecast.no_related", reason=reason))
    items = [f"{finding.domain} · {_hhmm(finding.start_time)}–{_hhmm(finding.end_time)}: {finding.summary}" for finding in view.findings[:RELATED_FINDINGS]]
    return p(t("confluence.forecast.related", reason=reason)) + li_list(items)


def _explanation_block(view: ExplanationView) -> str:
    if view.status == "unavailable":
        return p(view.message or t("confluence.forecast.nothing_to_explain"))
    explanation = view.explanation
    if explanation is None:
        return _findings_fallback(view)
    verdict = explanation.scaling.verdict
    parts = [f"<p><strong>{xml_text(explanation.headline)}</strong></p>", p(explanation.cause)]
    if explanation.key_facts:
        parts.append(kv_table([(fact.label, fact.value) for fact in explanation.key_facts]))
    parts.append(wrap_macro(SCALING_PANELS[verdict], p(f"{scaling_title(verdict)}. {explanation.scaling.reason}")))
    if explanation.scaling_risks:
        parts.append(f"<h4>{xml_text(t('confluence.forecast.risks'))}</h4>{li_list(explanation.scaling_risks)}")
    if explanation.next_checks:
        steps = "".join(f"<li>{xml_text(item)}</li>" for item in explanation.next_checks)
        parts.append(f"<h4>{xml_text(t('confluence.forecast.next_checks'))}</h4><ol>{steps}</ol>")
    stale = t("confluence.forecast.stale") if view.status == "stale" else ""
    meta = "; ".join(part for part in (t("confluence.forecast.explained_by", model=view.model), _check_text(view), stale) if part)
    parts.append(p(meta))
    return "".join(parts)


def _method_items(report: ForecastReport) -> list[str]:
    plan = report.resources
    items = [t("confluence.forecast.method_pods")]
    if plan.status == "ok":
        items.append(t("confluence.forecast.method_cpu", count=plan.steps_used))
    usl = {"count": report.samples_used, "minutes": _decimal(report.bin_minutes, 0), "r2": _decimal(report.fit.r2, 2)}
    if report.fit.x_max is not None:
        items.append(t("confluence.forecast.method_usl_ceiling", rps=_approx_rps(report.fit.x_max), **usl))
    else:
        items.append(t("confluence.forecast.method_usl", **usl))
    items.append(t("confluence.forecast.method_excluded", edges=report.excluded_edges, stalls=report.excluded_stalls))
    if report.excluded_degraded:
        items.append(t("confluence.forecast.method_excluded_degraded", count=report.excluded_degraded))
    return items + list(report.notes)


def _scale_caption(report: ForecastReport) -> str:
    if report.limit.rps is None:
        return t("confluence.forecast.caption_no_limit")
    if report.limit.kind == "instability":
        return t("confluence.forecast.caption_instability")
    if report.limit.kind == "sla":
        return t("confluence.forecast.caption_sla")
    return t("confluence.forecast.caption_model")


def forecast_page_body(
    report: ForecastReport,
    scenarios: Sequence[ForecastScenario],
    explanation: ExplanationView,
    source_url: str,
    headroom_pct: float,
    generated_text: str,
) -> str:
    """Storage-format page: verdict, load scale, scenarios, pods, test steps, the limit explanation and the method."""
    plan = report.resources
    target = _rps(report.target.rps) if report.target is not None else t("confluence.forecast.target_unset")
    params = t(
        "confluence.forecast.params",
        target=target,
        ceiling=_percent(plan.ceiling_pct, 0),
        headroom=_percent(headroom_pct, 0),
    )
    parts = [
        h2(forecast_page_title(report)),
        kv_table([
            (t("confluence.forecast.service"), report.service or "—"),
            (t("confluence.forecast.test_window"), f"{report.window_start_iso[:10]} {_hhmm(report.window_start_iso)}–{_hhmm(report.cutoff_iso)}"),
            (t("confluence.forecast.params_title"), params),
            (t("confluence.forecast.generated"), generated_text),
        ]),
    ]
    if source_url:
        parts.append(f'<p><a href="{xml_text(source_url)}">{xml_text(t("confluence.forecast.open_forecast"))}</a></p>')
    parts.extend([
        _verdict_panel(report, headroom_pct),
        h3(t("confluence.forecast.section_scale")), attachment_image(SCALE_PNG), p(_scale_caption(report)),
        h3(t("confluence.forecast.section_scenarios")), _scenarios_table(scenarios),
        h3(t("confluence.forecast.section_pods", rps=_rps(plan.target_rps))), _pods_table(report),
        h3(t("confluence.forecast.section_steps")), attachment_image(STEPS_PNG),
        h3(t("confluence.forecast.section_explanation")), _explanation_block(explanation),
        wrap_expand(t("confluence.forecast.section_method"), li_list(_method_items(report))),
    ])
    return "".join(parts)


def _figure(width: float, height: float) -> tuple[Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    return plt, fig, ax


def _png(plt: Any, fig: Any) -> bytes:
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _zone_text(report: ForecastReport, zone: ScaleZone) -> str:
    if report.limit.rps is None:
        return t("confluence.forecast.zone_no_stalls")
    if zone.tone == "fail":
        if report.limit.kind == "instability":
            return t("confluence.forecast.zone_stalls")
        return t("confluence.forecast.zone_degraded") if report.limit.kind == "sla" else t("confluence.forecast.zone_over_ceiling")
    return t("confluence.forecast.zone_tight") if zone.tone == "warn" else t("confluence.forecast.zone_safe")


LIMIT_MARK_KEYS = {"instability": "confluence.forecast.mark_stalls", "sla": "confluence.forecast.mark_sla"}


def _mark_text(report: ForecastReport, mark: ScaleMark) -> str:
    if mark.kind == "safe":
        return f"{_rps(math.floor(mark.rps))}\n{t('confluence.forecast.mark_safe')}"
    if mark.kind == "limit":
        key = LIMIT_MARK_KEYS.get(report.limit.kind, "confluence.forecast.mark_ceiling")
        return f"{_approx_rps(mark.rps)}\n{t(key)}"
    return f"{_rps(mark.rps)}\n{t('confluence.forecast.mark_tested')}"


def _mark_rows(marks: Sequence[ScaleMark], end: float) -> list[int]:
    last_by_row: list[float] = []
    rows: list[int] = []
    for mark in marks:
        share = mark.rps / end
        row = next((index for index, last in enumerate(last_by_row) if share - last >= SCALE_LABEL_GAP), len(last_by_row))
        if row == len(last_by_row):
            last_by_row.append(share)
        else:
            last_by_row[row] = share
        rows.append(row)
    return rows


def render_scale_png(report: ForecastReport) -> bytes:
    """Load axis picture: zones, the untested part hatched, key loads below and the target above."""
    scale = report.scale
    end = scale.end_rps
    plt, fig, ax = _figure(12.0, 2.6)
    for zone in scale.zones:
        width = zone.to_rps - zone.from_rps
        ax.barh(0, width, left=zone.from_rps, height=0.6, color=ZONE_COLORS[zone.tone], alpha=0.45)
        if width >= ZONE_TEXT_SHARE * end:
            ax.text(zone.from_rps + width / 2, 0, _zone_text(report, zone), ha="center", va="center", fontsize=10, fontweight="bold")
    if scale.tested_rps < end:
        ax.barh(0, end - scale.tested_rps, left=scale.tested_rps, height=0.6, color="none", hatch="///", edgecolor="#9aa0a6", linewidth=0)
    for mark, row in zip(scale.marks, _mark_rows(scale.marks, end)):
        ax.annotate(_mark_text(report, mark), xy=(mark.rps, -0.3), xytext=(mark.rps, -0.55 - 0.75 * row), ha="center", va="top",
                    fontsize=9, arrowprops={"arrowstyle": "-", "color": "#6f7a86", "lw": 0.8})
    if report.target is not None and report.answer is not None:
        x = min(report.target.rps, end)
        color = VERDICT_COLORS[report.answer.verdict]
        ax.plot([x, x], [-0.3, 0.55], linestyle="--", color=color, linewidth=1.6)
        ax.text(x, 0.6, t("confluence.forecast.scale_target", rps=_rps(report.target.rps)), ha="right" if x > 0.85 * end else "left",
                va="bottom", fontsize=10, fontweight="bold", color=color)
    ax.set_xlim(0, end)
    ax.set_ylim(-2.4, 1.0)
    ax.axis("off")
    return _png(plt, fig)


def _step_note(step: StepPoint) -> str:
    if step.after_drop:
        return ""
    if step.degraded:
        return "\n" + t("confluence.forecast.step_degraded")
    return "\n" + t("confluence.forecast.step_stalls", stalls=step.stalls, samples=step.samples) if step.stalls else ""


def render_steps_png(report: ForecastReport) -> bytes:
    """Mean response per load step: green without stalls, red with them or above the report maximum."""
    steps = report.steps
    plt, fig, ax = _figure(12.0, 3.6)
    if not steps:
        ax.text(0.5, 0.5, t("confluence.forecast.steps_none"), ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _png(plt, fig)
    tones = ["muted" if step.after_drop else ("fail" if step.stalls or step.degraded else "ok") for step in steps]
    values = [step.response_ms for step in steps]
    ax.bar(range(len(steps)), values, color=[STEP_COLORS[tone] for tone in tones], alpha=0.8)
    for index, step in enumerate(steps):
        ax.text(index, step.response_ms, _ms(step.response_ms) + _step_note(step), ha="center", va="bottom", fontsize=9)
    word = _step_word(report)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{word} {step.number}\n{_rps(step.rps)}" for step in steps], fontsize=9)
    ax.set_ylabel(t("confluence.forecast.steps_axis"))
    ax.set_ylim(0, max(values) * 1.35)
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _png(plt, fig)


def _publications_ref() -> tuple[str, str]:
    cfg = _storage()
    return str(cfg.get("schema") or "public"), str(cfg.get("forecast_confluence_table") or DEFAULT_PUBLICATIONS_TABLE)


def _ensure_publications_table(conn: Any, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                run_id         TEXT        PRIMARY KEY,
                run_name       TEXT,
                service        TEXT,
                page_id        TEXT        NOT NULL,
                page_url       TEXT,
                space_key      TEXT,
                parent_page_id TEXT,
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        ).format(sql.Identifier(schema, table)))
    conn.commit()


def load_forecast_publication(run_id: str) -> Optional[ForecastPublication]:
    """Confluence page of the forecast report of a run, if it was published."""
    schema, table = _publications_ref()
    conn = _ts_conn()
    try:
        _ensure_publications_table(conn, schema, table)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT page_id, page_url, updated_at FROM {} WHERE run_id = %s").format(sql.Identifier(schema, table)),
                (run_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    updated = row[2].isoformat() if isinstance(row[2], datetime) else None
    return ForecastPublication(run_id=run_id, page_id=str(row[0]), page_url=str(row[1] or ""), updated_at=updated)


def save_forecast_publication(report: ForecastReport, page_id: str, page_url: str, space_key: str, parent_page_id: str) -> None:
    schema, table = _publications_ref()
    conn = _ts_conn()
    try:
        _ensure_publications_table(conn, schema, table)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (run_id, run_name, service, page_id, page_url, space_key, parent_page_id, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (run_id) DO UPDATE SET
                        run_name = EXCLUDED.run_name, service = EXCLUDED.service, page_id = EXCLUDED.page_id,
                        page_url = EXCLUDED.page_url, space_key = EXCLUDED.space_key,
                        parent_page_id = EXCLUDED.parent_page_id, updated_at = now()
                    """
                ).format(sql.Identifier(schema, table)),
                (report.run_id, report.run_name, report.service, page_id, page_url, space_key, parent_page_id),
            )
        conn.commit()
    finally:
        conn.close()


def _forecast_parent(cfg: dict[str, Any]) -> str:
    """confluence.forecast_parent_page_id, or the parent page of test reports when it is empty."""
    own = str(cfg.get("forecast_parent_page_id") or "").strip()
    return own or str(cfg.get("parent_page_id") or "").strip()


def _generated_text() -> str:
    """Local time of the publication: the date in the format of the current locale, then HH:MM."""
    local = datetime.now(timezone.utc) + timedelta(hours=_time_shift_hours())
    return f"{fmt_date(local)} {local.strftime('%H:%M')}"


def publish_forecast(
    run_id: str,
    target_rps: Optional[float],
    headroom_pct: float,
    cpu_ceiling_pct: Optional[float],
    source_url: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    """Builds the forecast with the calculator settings and publishes it as its own Confluence page.

    The texts of the forecast (notes, plan messages, scenario titles) and the page are rendered in the language
    of the run's report; the progress messages are rendered outside that scope, in the job's locale.
    """
    cfg = confluence_config()
    validate_confluence_config(cfg)
    lang = report_language_of(run_id)
    _progress(progress_callback, msg("confluence.progress.forecast"), 10)
    with with_locale(lang):
        report = build_forecast(run_id, target_rps, headroom_pct, cpu_ceiling_pct)
    _progress(progress_callback, msg("confluence.progress.scenarios"), 20)
    with with_locale(lang):
        scenarios = forecast_scenarios(run_id, report, headroom_pct, cpu_ceiling_pct)
        explanation = explanation_view(run_id)
    _progress(progress_callback, msg("confluence.progress.images"), 40)
    with with_locale(lang):
        attachments = {SCALE_PNG: render_scale_png(report), STEPS_PNG: render_steps_png(report)}
        title = forecast_page_title(report)
        body = forecast_page_body(report, scenarios, explanation, source_url, headroom_pct, _generated_text())
    client = ConfluenceClient({**cfg, "parent_page_id": _forecast_parent(cfg)})
    known = load_forecast_publication(run_id)
    page_id, page_url = upsert_page(client, known.page_id if known else "", title, body, attachments, progress_callback)
    save_forecast_publication(report, page_id, page_url, client.space_key, client.parent_page_id)
    _progress(progress_callback, msg("confluence.progress.done"), 100)
    return {"page_id": page_id, "page_url": page_url, "title": title}


__all__ = [
    "ForecastPublication",
    "ForecastScenario",
    "ScenarioError",
    "confidence_title",
    "forecast_page_body",
    "forecast_page_title",
    "forecast_scenarios",
    "load_forecast_publication",
    "publish_forecast",
    "render_scale_png",
    "render_steps_png",
    "verdict_title",
]
