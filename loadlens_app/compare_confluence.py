# SPDX-License-Identifier: Apache-2.0
"""Comparison of two runs as its own Confluence page: outcome, verdicts, key metric changes and the AI explanation.

The page is written in the language of the report of run A (of B when A has none), like the explanation it shows;
the job's progress messages stay in the job's locale. Publishing never asks the model: the page shows the cached
explanation, or the findings of both runs when there is none yet. The page has no pictures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from psycopg2 import sql

from AI.db_store import COMPARE_PUBLICATIONS_TABLE, StoredFinding
from AI.i18n import LocalizedError, fmt_date, fmt_number, get_locale, msg, t, with_locale
from AI.pipeline import _time_shift_hours
from AI.verdicts import label as verdict_label
from loadlens_app.compare_explanation import (
    AGGREGATE,
    scenario_pairs,
    CHANGE_THRESHOLD_PCT,
    KEY_RPS_TOLERANCE_PCT,
    MAX_FINDINGS_PER_RUN,
    SIDES,
    CompareView,
    MetricChange,
    RunFacts,
    assessment_title,
    domain_title,
    domain_verdicts,
    explanation_view,
    flag_text,
    outcome_title,
    reason_text,
    rps_kind_text,
    type_text,
)
from loadlens_app.confluence_export import (
    ConfluenceClient,
    ProgressCallback,
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
from loadlens_app.forecast_confluence import confidence_title
from settings import CONFIG

OUTCOME_PANELS = {"better": "tip", "same": "info", "mixed": "note", "incomparable": "note", "worse": "warning"}
RELATED_FINDINGS = 5


class ComparePublishError(LocalizedError, ValueError):
    """The pair cannot be published (nothing to compare)."""


@dataclass(frozen=True)
class ComparePublication:
    run_a: str
    run_b: str
    page_id: str
    page_url: str
    updated_at: Optional[str]


def _progress(callback: ProgressCallback | None, message: str, pct: int | None = None) -> None:
    if callback:
        callback(message, pct)


def _number(value: Optional[float]) -> str:
    if value is None:
        return "—"
    digits = 0 if abs(value) >= 100 else (2 if abs(value) >= 1 else 4)
    return fmt_number(value, digits)


def _signed_pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return ("+" if value > 0 else "") + t("capacity.unit.pct", value=fmt_number(value, 1))


def _verdict(code: Optional[str]) -> str:
    return verdict_label(code, get_locale()) if code else "—"


def compare_page_title(view: CompareView) -> str:
    context = view.context
    return t("confluence.compare.page_title", a=context.run_a, b=context.run_b)


def _run_line(facts: RunFacts) -> str:
    if not facts.has_report:
        return t("confluence.compare.run_no_report", name=facts.run_name)
    parts = [facts.run_name]
    if facts.created_at:
        parts.append(fmt_date(facts.created_at))
    parts.append(type_text(facts.type_key))
    if facts.window_min:
        parts.append(t("confluence.compare.window_min", minutes=fmt_number(facts.window_min, 0)))
    return " · ".join(parts)


def _key_rps(facts: RunFacts) -> str:
    if facts.key_rps is None:
        return "—"
    kind = rps_kind_text(facts.key_rps_kind)
    value = t("confluence.compare.rps", value=_number(facts.key_rps))
    return f"{value} ({kind})" if kind else value


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "".join(f"<th>{xml_text(cell)}</th>" for cell in head)
    body = "".join("<tr>" + "".join(f"<td>{xml_text(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _outcome_panel(view: CompareView) -> str:
    context = view.context
    outcome = context.outcome
    body = f"<p><strong>{xml_text(outcome_title(outcome.code))}</strong></p>" + li_list([reason_text(note) for note in outcome.reasons])
    if context.flags:
        body += f"<p>{xml_text(t('confluence.compare.flags'))}</p>" + li_list([flag_text(note) for note in context.flags])
    return wrap_macro(OUTCOME_PANELS.get(outcome.code, "info"), body)


def _phases_table(view: CompareView) -> str:
    """Phase against phase of two scenario runs (as the phase table of the page, without the query values)."""
    pairs = scenario_pairs(view.context.reports)
    if not pairs or not pairs.get("pairs"):
        return ""

    def phase(item) -> str:
        if not item:
            return t("compare.phases.absent")
        name = str(item.get("label") or item.get("title") or "—")
        return f"{name} ({t('compare.phases.excluded')})" if item.get("excluded") else name

    def pair_value(a, b, change) -> str:
        text = f"{fmt_number(a, 1) if a is not None else '—'} / {fmt_number(b, 1) if b is not None else '—'}"
        return f"{text} ({'+' if change > 0 else ''}{fmt_number(change, 1)} %)" if change is not None else text

    head = (t("compare.phases.col_phase_a"), t("compare.phases.col_phase_b"), t("compare.phases.col_duration"),
            t("compare.phases.col_rps"), t("compare.phases.col_verdict"))
    rows = []
    for pair in pairs["pairs"]:
        a, b = pair.get("a"), pair.get("b")
        rows.append((
            phase(a), phase(b),
            pair_value(a and a.get("duration_min"), b and b.get("duration_min"), pair.get("duration_change_pct")),
            pair_value(a and a.get("rps_mean"), b and b.get("rps_mean"), pair.get("rps_change_pct")),
            f"{_verdict(a.get('verdict')) if a else '—'} · {_verdict(b.get('verdict')) if b else '—'}",
        ))
    return h3(t("compare.phases.title")) + _table(head, rows)


def _verdicts_table(view: CompareView) -> str:
    context = view.context
    a, b = context.sides["A"], context.sides["B"]
    head = (t("confluence.compare.col_what"), f"A · {a.run_name}", f"B · {b.run_name}")
    rows = [
        (t("confluence.compare.final_verdict"), _verdict(a.verdict), _verdict(b.verdict)),
        (t("confluence.compare.sla_verdict"), _verdict(a.sla_verdict), _verdict(b.sla_verdict)),
        (t("confluence.compare.key_rps"), _key_rps(a), _key_rps(b)),
        (
            t("confluence.compare.test_type"),
            type_text(a.type_key) if a.has_report else "—",
            type_text(b.type_key) if b.has_report else "—",
        ),
        (
            t("confluence.compare.window"),
            t("confluence.compare.window_min", minutes=fmt_number(a.window_min, 0)) if a.window_min else "—",
            t("confluence.compare.window_min", minutes=fmt_number(b.window_min, 0)) if b.window_min else "—",
        ),
    ]
    rows += [(row["domain"], row["A"], row["B"]) for row in domain_verdicts(context.reports, context.titles)]
    return _table(head, rows)


def _change_row(change: MetricChange, titles: dict[str, str]) -> tuple[str, ...]:
    return (
        domain_title(change.domain, titles),
        change.metric,
        _number(change.value_a),
        _number(change.value_b),
        _signed_pct(change.change_pct),
        assessment_title(change.assessment),
    )


def _metrics_block(view: CompareView) -> str:
    context = view.context
    metrics = context.metrics
    counts = metrics.counts()
    summary = p(t(
        "confluence.compare.metrics_counts",
        threshold=fmt_number(CHANGE_THRESHOLD_PCT, 0),
        better=counts["better"],
        worse=counts["worse"],
        same=counts["same"],
        unknown=counts["unknown"],
    ))
    top = metrics.top()
    if not top:
        return summary + p(t("confluence.compare.no_changes"))
    head = (
        t("confluence.compare.col_domain"),
        t("confluence.compare.col_metric"),
        "A",
        "B",
        t("confluence.compare.col_change"),
        t("confluence.compare.col_assessment"),
    )
    return summary + _table(head, [_change_row(change, context.titles) for change in top])


def _check_text(view: CompareView) -> str:
    check = view.check
    if check is None:
        return ""
    if check.status == "verified":
        return t("confluence.forecast.check_verified", matched=check.claims_matched, total=check.claims_total)
    if check.status == "unverified":
        return t("confluence.forecast.check_unverified", numbers=", ".join(check.unmatched))
    return t("confluence.forecast.check_none")


def _finding_line(finding: StoredFinding) -> str:
    component = f" · {finding.component}" if finding.component else ""
    return f"{finding.domain}{component}: {finding.summary}"


def _findings_fallback(view: CompareView) -> str:
    context = view.context
    reason = t("confluence.compare.stale_old") if view.status == "stale" else t("confluence.compare.no_explanation")
    parts = [p(reason)]
    for side in SIDES:
        items = [_finding_line(finding) for s, finding in context.findings if s == side][:RELATED_FINDINGS]
        name = context.sides[side].run_name
        if items:
            parts.append(f"<h4>{xml_text(t('confluence.compare.findings_of', side=side, name=name))}</h4>{li_list(items)}")
    if len(parts) == 1:
        parts.append(p(t("confluence.compare.no_findings")))
    return "".join(parts)


def _list_section(title_key: str, items: Sequence[str], ordered: bool = False) -> str:
    if not items:
        return ""
    if ordered:
        steps = "".join(f"<li>{xml_text(item)}</li>" for item in items)
        return f"<h4>{xml_text(t(title_key))}</h4><ol>{steps}</ol>"
    return f"<h4>{xml_text(t(title_key))}</h4>{li_list(items)}"


def _explanation_block(view: CompareView) -> str:
    explanation = view.explanation
    if explanation is None:
        return _findings_fallback(view)
    parts = [f"<p><strong>{xml_text(explanation.headline)}</strong></p>", p(explanation.summary)]
    if explanation.key_changes:
        head = (t("confluence.compare.col_what"), "A", "B", t("confluence.compare.col_assessment"))
        rows = [(change.label, change.value_a or "—", change.value_b or "—", assessment_title(change.assessment)) for change in explanation.key_changes]
        parts.append(_table(head, rows))
    parts.append(_list_section("confluence.compare.causes", explanation.causes))
    parts.append(_list_section("confluence.compare.caveats", explanation.caveats))
    parts.append(_list_section("confluence.forecast.next_checks", explanation.next_checks, ordered=True))
    stale = t("confluence.compare.stale") if view.status == "stale" else ""
    meta = "; ".join(part for part in (
        t("confluence.forecast.explained_by", model=view.model),
        t("confluence.compare.confidence", level=confidence_title(explanation.confidence)),
        _check_text(view),
        stale,
    ) if part)
    parts.append(p(meta))
    return "".join(part for part in parts if part)


def _method_items() -> list[str]:
    return [
        t("confluence.compare.method_p95"),
        t("confluence.compare.method_threshold", threshold=fmt_number(CHANGE_THRESHOLD_PCT, 0)),
        t("confluence.compare.method_outcome", tolerance=fmt_number(KEY_RPS_TOLERANCE_PCT, 0)),
        t("confluence.compare.method_findings", count=MAX_FINDINGS_PER_RUN),
    ]


def compare_page_body(view: CompareView, source_url: str, generated_text: str) -> str:
    """Storage-format page: parameters, outcome, verdicts, key metric changes, the explanation and the method."""
    context = view.context
    a, b = context.sides["A"], context.sides["B"]
    services = [name for name in dict.fromkeys((a.service, b.service)) if name]
    parts = [
        h2(compare_page_title(view)),
        kv_table([
            (t("confluence.forecast.service"), ", ".join(services) or "—"),
            (t("confluence.compare.run_a"), _run_line(a)),
            (t("confluence.compare.run_b"), _run_line(b)),
            (t("confluence.compare.metrics"), t("confluence.compare.metrics_value", aggregate=AGGREGATE)),
            (t("confluence.forecast.generated"), generated_text),
        ]),
    ]
    if source_url:
        parts.append(f'<p><a href="{xml_text(source_url)}">{xml_text(t("confluence.compare.open_compare"))}</a></p>')
    parts.extend([
        _outcome_panel(view),
        h3(t("confluence.compare.section_verdicts")), _verdicts_table(view),
        _phases_table(view),
        h3(t("confluence.compare.section_metrics")), _metrics_block(view),
        h3(t("confluence.compare.section_explanation")), _explanation_block(view),
        wrap_expand(t("confluence.compare.section_method"), li_list(_method_items())),
    ])
    return "".join(parts)


# ---- publications --------------------------------------------------------------------------------------------


def _publications_ref() -> tuple[str, str]:
    storage = CONFIG.get("storage") if isinstance(CONFIG.get("storage"), dict) else {}
    cfg = storage.get("timescale") if isinstance(storage.get("timescale"), dict) else {}
    return str(cfg.get("schema") or "public"), str(cfg.get("compare_confluence_table") or COMPARE_PUBLICATIONS_TABLE)


def _ensure_publications_table(conn: Any, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                run_a          TEXT        NOT NULL,
                run_b          TEXT        NOT NULL,
                service        TEXT,
                page_id        TEXT        NOT NULL,
                page_url       TEXT,
                space_key      TEXT,
                parent_page_id TEXT,
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (run_a, run_b)
            )
            """
        ).format(sql.Identifier(schema, table)))
    conn.commit()


def load_compare_publication(run_a: str, run_b: str) -> Optional[ComparePublication]:
    """Confluence page of the comparison of A with B, if it was published."""
    schema, table = _publications_ref()
    conn = _ts_conn()
    try:
        _ensure_publications_table(conn, schema, table)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT page_id, page_url, updated_at FROM {} WHERE run_a = %s AND run_b = %s").format(
                    sql.Identifier(schema, table)
                ),
                (run_a, run_b),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    updated = row[2].isoformat() if isinstance(row[2], datetime) else None
    return ComparePublication(run_a=run_a, run_b=run_b, page_id=str(row[0]), page_url=str(row[1] or ""), updated_at=updated)


def save_compare_publication(view: CompareView, page_id: str, page_url: str, space_key: str, parent_page_id: str) -> None:
    schema, table = _publications_ref()
    context = view.context
    service = context.sides["A"].service or context.sides["B"].service
    conn = _ts_conn()
    try:
        _ensure_publications_table(conn, schema, table)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (run_a, run_b, service, page_id, page_url, space_key, parent_page_id, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (run_a, run_b) DO UPDATE SET
                        service = EXCLUDED.service, page_id = EXCLUDED.page_id, page_url = EXCLUDED.page_url,
                        space_key = EXCLUDED.space_key, parent_page_id = EXCLUDED.parent_page_id, updated_at = now()
                    """
                ).format(sql.Identifier(schema, table)),
                (context.run_a, context.run_b, service, page_id, page_url, space_key, parent_page_id),
            )
        conn.commit()
    finally:
        conn.close()


def _compare_parent(cfg: dict[str, Any]) -> str:
    """confluence.compare_parent_page_id, or the parent page of test reports when it is empty."""
    own = str(cfg.get("compare_parent_page_id") or "").strip()
    return own or str(cfg.get("parent_page_id") or "").strip()


def _generated_text() -> str:
    """Local time of the publication: the date in the format of the current locale, then HH:MM."""
    local = datetime.now(timezone.utc) + timedelta(hours=_time_shift_hours())
    return f"{fmt_date(local)} {local.strftime('%H:%M')}"


def publish_comparison(
    run_a: str,
    run_b: str,
    source_url: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    """Publishes the comparison of A with B as its own Confluence page (created once, then updated).

    The page is rendered in the language of the report of A; the progress messages in the job's locale.
    """
    cfg = confluence_config()
    validate_confluence_config(cfg)
    _progress(progress_callback, msg("confluence.progress.compare"), 10)
    view = explanation_view(run_a, run_b)
    if view.status == "unavailable" or view.context is None:
        raise ComparePublishError("compare.explain.nothing")
    with with_locale(view.context.lang):
        title = compare_page_title(view)
        body = compare_page_body(view, source_url, _generated_text())
    client = ConfluenceClient({**cfg, "parent_page_id": _compare_parent(cfg)})
    known = load_compare_publication(run_a, run_b)
    page_id, page_url = upsert_page(client, known.page_id if known else "", title, body, {}, progress_callback)
    save_compare_publication(view, page_id, page_url, client.space_key, client.parent_page_id)
    _progress(progress_callback, msg("confluence.progress.done"), 100)
    return {"page_id": page_id, "page_url": page_url, "title": title}


__all__ = [
    "ComparePublication",
    "ComparePublishError",
    "compare_page_body",
    "compare_page_title",
    "load_compare_publication",
    "publish_comparison",
]
