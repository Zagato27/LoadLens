# SPDX-License-Identifier: Apache-2.0
"""LLM explanation of the differences between two runs of the compare page.

The program computes the facts and the outcome; the model only explains them. A is the assessed run (usually the
newer one), B the baseline; every change is A relative to B. The explanation is built on request, cached per pair
(``compare_explanations``) and goes stale when its inputs change (a new report of either run, other metrics, another
prompt text). Facts, prompt and answer are in the language of the report of A (of B when A has none).

Metrics are compared by p95 over the whole window of each run, as the compare page shows them by default; they are
not limited to the active project area, so the page and the Confluence publication see the same facts.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence

from pydantic import ValidationError

from AI.compare_explanation import CompareExplanation, check_comparison, compare_finding_payload, parse_comparison, side_ref
from AI.db_store import (
    COMPARE_EXPLANATIONS_TABLE,
    MAX_RPS_FIELDS,
    StoredFinding,
    StoredPairExplanation,
    StoredRunReport,
    StoredRunRow,
    load_compare_explanation,
    load_pair_summary,
    load_run_report,
    max_rps_from_parsed,
    save_compare_explanation,
)
from AI.i18n import fmt_number, get_locale, has_key, msg, t, with_locale
from AI.limit_explanation import ExplanationCheck
from AI.metric_direction import BETTER, SAME, UNKNOWN, WORSE, assess, change_pct, metric_direction
from AI.phases import phase_pairs
from AI.prompt_store import load_prompt
from AI.prompt_store import text as prompt_text
from AI.providers import ask_llm_with_text_data, usage_domain
from AI.query_packs.blocks import block_roles
from AI.scoring import VERIFICATION_UNVERIFIED
from AI.test_profile import normalize_test_type, test_type_label
from AI.verdicts import NO_DATA, RANK
from AI.verdicts import label as verdict_label
from loadlens_app.core import _find_area_for_service, _ts_conn
from loadlens_app.forecast_explanation import ExplanationError, LlmChoice, compact_system, llm_choice
from loadlens_app.i18n import stored_report_language
from loadlens_app.limits import consume_llm
from settings import CONFIG

# AI/prompts/<lang>/compare_prompt.txt; the system prompt is ``compare.system`` of _strings.json.
PROMPT_FILE = "compare_prompt.txt"
SIDES = ("A", "B")
AGGREGATE = "p95"
# The p95 aggregate of the compare blueprint (loadlens_app/blueprints/compare.py, AGGREGATES["p95"]).
P95_SQL = "percentile_cont(0.95) WITHIN GROUP (ORDER BY value)"
CHANGE_THRESHOLD_PCT = 10.0
KEY_RPS_TOLERANCE_PCT = 5.0
WINDOW_TOLERANCE_PCT = 30.0
TOP_CHANGES = 15
PER_DOMAIN_CHANGES = 5
MAX_ONLY_IN = 10
MAX_FINDINGS_PER_RUN = 8
MIN_REGENERATE_SEC = 30.0
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
# Order of the domain rows, as in the verdict grid of the page (static/js/compare.js, VERDICT_DOMAINS).
DOMAIN_ORDER = ("lt_framework", "microservices", "jvm", "database", "kafka", "hard_resources", "lt_generators", "application_logs")

OUTCOMES = ("better", "worse", "same", "mixed", "incomparable")
MIXED = "mixed"
INCOMPARABLE = "incomparable"
# Kind of the key load level by the section of the final report it comes from (AI.db_store.MAX_RPS_FIELDS).
KEY_RPS_KINDS = {"peak_performance": "peak", "stability_under_load": "held", "spike_analysis": "spike", "scenario": "phases"}
# Only the peak of a step or stress test is a result; the held, spike and scenario levels are the load the test applied.
RESULT_RPS_KINDS = frozenset({"peak"})

ViewStatus = Literal["unavailable", "missing", "ready", "stale"]


# ---- facts ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunFacts:
    """What the comparison needs of one run; everything is empty for a run without a stored report."""

    side: str
    run_name: str
    has_report: bool = False
    run_id: str = ""
    service: str = ""
    project_area: str = ""
    test_type: str = ""
    created_at: Optional[datetime] = None
    window_min: Optional[float] = None
    verdict: Optional[str] = None
    sla_verdict: Optional[str] = None
    key_rps: Optional[float] = None
    key_rps_kind: str = ""
    findings_total: int = 0

    @property
    def type_key(self) -> str:
        """Normalized test type; a report saved without a type is a step test (as the baseline lookup reads it)."""
        raw = str(self.test_type or "").strip()
        return normalize_test_type(raw) or (raw.lower() if raw else "step")

    @property
    def known_verdict(self) -> Optional[str]:
        return self.verdict if self.verdict and self.verdict != NO_DATA else None


@dataclass(frozen=True)
class MetricChange:
    domain: str
    metric: str
    value_a: Optional[float]
    value_b: Optional[float]
    change_pct: Optional[float]
    direction: str
    assessment: str


@dataclass(frozen=True)
class MetricsSummary:
    """Metrics both runs stored (with the change of A relative to B) and the ones only one run has."""

    changes: list[MetricChange] = field(default_factory=list)
    only_in_a: list[tuple[str, str]] = field(default_factory=list)
    only_in_b: list[tuple[str, str]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {BETTER: 0, WORSE: 0, SAME: 0, UNKNOWN: 0}
        for change in self.changes:
            out[change.assessment] = out.get(change.assessment, 0) + 1
        return out

    def top(self) -> list[MetricChange]:
        """Changes at or above the threshold, the largest first: assessed ones before those without a direction,
        at most PER_DOMAIN_CHANGES per domain and TOP_CHANGES in all."""
        notable = [
            change for change in self.changes
            if change.change_pct is not None and abs(change.change_pct) >= CHANGE_THRESHOLD_PCT
        ]
        notable.sort(key=lambda change: (change.assessment == UNKNOWN, -abs(change.change_pct or 0.0), change.domain, change.metric))
        picked: list[MetricChange] = []
        per_domain: dict[str, int] = {}
        for change in notable:
            if per_domain.get(change.domain, 0) >= PER_DOMAIN_CHANGES:
                continue
            per_domain[change.domain] = per_domain.get(change.domain, 0) + 1
            picked.append(change)
            if len(picked) >= TOP_CHANGES:
                break
        return picked


@dataclass(frozen=True)
class Note:
    """A reason of the outcome or a comparability flag: a code and the parameters of its text."""

    code: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    code: str
    reasons: list[Note]


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def key_rps(parsed: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """Key load level of a final report and its kind (``peak``, ``held``, ``spike``), by the rule of ``max_rps``."""
    for section, _ in MAX_RPS_FIELDS:
        block = parsed.get(section) if isinstance(parsed, Mapping) else None
        if not isinstance(block, Mapping):
            continue
        value = max_rps_from_parsed({section: dict(block)})
        if value is not None:
            return value, KEY_RPS_KINDS[section]
    return None, ""


def _verified(findings: Sequence[StoredFinding]) -> list[StoredFinding]:
    return [finding for finding in findings if finding.verification != VERIFICATION_UNVERIFIED]


def picked_findings(report: Optional[StoredRunReport]) -> list[StoredFinding]:
    """Findings of the domain analyses of a run whose numbers passed the report check, the most severe first."""
    if report is None:
        return []
    ordered = sorted(
        _verified(report.findings()),
        key=lambda finding: (SEVERITY_ORDER.get(finding.severity.lower(), len(SEVERITY_ORDER)), finding.start_time, finding.ref),
    )
    return ordered[:MAX_FINDINGS_PER_RUN]


def run_facts(side: str, run_name: str, report: Optional[StoredRunReport]) -> RunFacts:
    final = report.final if report is not None else None
    if report is None or final is None:
        any_row = next(iter(report.rows.values()), None) if report is not None else None
        return RunFacts(
            side=side,
            run_name=run_name,
            service=any_row.service if any_row else "",
            test_type=any_row.test_type if any_row else "",
            findings_total=len(_verified(report.findings())) if report is not None else 0,
        )
    window = None
    if final.start_ms is not None and final.end_ms is not None and final.end_ms > final.start_ms:
        window = (final.end_ms - final.start_ms) / 60000.0
    value, kind = key_rps(final.parsed)
    return RunFacts(
        side=side,
        run_name=run_name,
        has_report=True,
        run_id=final.run_id,
        service=final.service,
        project_area=final.project_area,
        test_type=final.test_type,
        created_at=final.created_at,
        window_min=window,
        verdict=report.verdict,
        sla_verdict=final.sla_verdict,
        key_rps=value,
        key_rps_kind=kind,
        findings_total=len(_verified(report.findings())),
    )


def summarize_metrics(rows: Sequence[Sequence[Any]], roles: Mapping[str, Mapping[str, str]]) -> MetricsSummary:
    """``rows`` are ``(domain, query_label, value_a, value_b)``; ``roles`` is ``{domain: {label: role}}``."""
    summary = MetricsSummary()
    for row in rows:
        domain, label = str(row[0] or ""), str(row[1] or "")
        value_a, value_b = _float(row[2]), _float(row[3])
        if value_a is None and value_b is None:
            continue
        if value_b is None:
            summary.only_in_a.append((domain, label))
            continue
        if value_a is None:
            summary.only_in_b.append((domain, label))
            continue
        direction = metric_direction(label, (roles.get(domain) or {}).get(label))
        change = change_pct(value_a, value_b)
        summary.changes.append(MetricChange(
            domain=domain,
            metric=label,
            value_a=value_a,
            value_b=value_b,
            change_pct=change,
            direction=direction,
            assessment=assess(change, direction, CHANGE_THRESHOLD_PCT),
        ))
    return summary


def _both(a: Any, b: Any) -> bool:
    return bool(a) and bool(b)


def outcome_of(a: RunFacts, b: RunFacts, metrics: MetricsSummary) -> Outcome:
    """Outcome of A against B; the first rule that applies decides (plans/compare-explanation-plan.md, 3.2)."""
    if (_both(a.service, b.service) and a.service != b.service) or (a.has_report and b.has_report and a.type_key != b.type_key):
        return Outcome(INCOMPARABLE, [Note("scope")])
    verdict_a, verdict_b = a.known_verdict, b.known_verdict
    if verdict_a and verdict_b and RANK[verdict_a] != RANK[verdict_b]:
        code = BETTER if RANK[verdict_a] < RANK[verdict_b] else WORSE
        return Outcome(code, [Note("verdict", {"a": verdict_a, "b": verdict_b})])
    if (
        a.key_rps is not None and b.key_rps is not None
        and a.key_rps_kind == b.key_rps_kind and a.key_rps_kind in RESULT_RPS_KINDS
    ):
        change = change_pct(a.key_rps, b.key_rps)
        if change is not None and abs(change) > KEY_RPS_TOLERANCE_PCT:
            return Outcome(BETTER if change > 0 else WORSE, [Note("rps", {"a": a.key_rps, "b": b.key_rps, "change": change})])
    counts = metrics.counts()
    better, worse = counts[BETTER], counts[WORSE]
    params = {"better": better, "worse": worse, "total": better + worse + counts[SAME], "threshold": CHANGE_THRESHOLD_PCT}
    if better and worse:
        return Outcome(MIXED, [Note("metrics", params)])
    if better or worse:
        return Outcome(BETTER if better else WORSE, [Note("metrics", params)])
    return Outcome(SAME, [Note("same", params)])


def flags_of(a: RunFacts, b: RunFacts, metrics: MetricsSummary) -> list[Note]:
    """What makes the comparison approximate (plans/compare-explanation-plan.md, 3.3)."""
    flags: list[Note] = []
    if _both(a.service, b.service) and a.service != b.service:
        flags.append(Note("service", {"a": a.service, "b": b.service}))
    if a.has_report and b.has_report and a.type_key != b.type_key:
        flags.append(Note("test_type", {"a": a.type_key, "b": b.type_key}))
    if a.window_min and b.window_min and abs(a.window_min - b.window_min) / max(a.window_min, b.window_min) * 100.0 > WINDOW_TOLERANCE_PCT:
        flags.append(Note("window", {"a": a.window_min, "b": b.window_min}))
    if a.created_at and b.created_at and a.created_at < b.created_at:
        flags.append(Note("order"))
    for side in (a, b):
        if not side.has_report:
            flags.append(Note("no_report", {"name": side.run_name}))
        elif not side.findings_total:
            flags.append(Note("no_findings", {"name": side.run_name}))
    if a.key_rps_kind and b.key_rps_kind and a.key_rps_kind != b.key_rps_kind:
        flags.append(Note("rps_kind", {"a": a.key_rps_kind, "b": b.key_rps_kind}))
    elif a.key_rps is not None and b.key_rps is not None and a.key_rps_kind == b.key_rps_kind and a.key_rps_kind not in RESULT_RPS_KINDS:
        change = change_pct(a.key_rps, b.key_rps)
        if change is not None and abs(change) > KEY_RPS_TOLERANCE_PCT:
            flags.append(Note("load", {"a": a.key_rps, "b": b.key_rps}))
    partial = len(metrics.only_in_a) + len(metrics.only_in_b)
    if partial:
        flags.append(Note("metrics_partial", {"count": partial}))
    return flags


# ---- texts -------------------------------------------------------------------------------------------------------


def _number(value: Any, digits: int) -> str:
    return fmt_number(value, digits)


def _rps(value: Any) -> str:
    number = _float(value) or 0.0
    return _number(number, 0 if abs(number) >= 100 else 1)


def _signed_pct(value: Any) -> str:
    number = _float(value) or 0.0
    return ("+" if number > 0 else "") + _number(number, 1)


def _verdict_text(code: Any) -> str:
    return verdict_label(code, get_locale()) if code else "—"


def type_text(key: str) -> str:
    return test_type_label(key, get_locale()) or key


def rps_kind_text(kind: str) -> str:
    return t(f"compare.rps_kind.{kind}") if kind else ""


def note_params(note: Note) -> dict[str, Any]:
    """Parameters of a note as its text shows them, in the current locale."""
    params = dict(note.params)
    if note.code == "verdict":
        params = {"a": _verdict_text(params["a"]), "b": _verdict_text(params["b"])}
    elif note.code == "rps":
        params = {"a": _rps(params["a"]), "b": _rps(params["b"]), "change": _signed_pct(params["change"])}
    elif note.code in ("metrics", "same"):
        params["threshold"] = _number(params["threshold"], 0)
    elif note.code == "test_type":
        params = {"a": type_text(params["a"]), "b": type_text(params["b"])}
    elif note.code == "window":
        params = {"a": _number(params["a"], 0), "b": _number(params["b"], 0)}
    elif note.code == "rps_kind":
        params = {"a": rps_kind_text(params["a"]), "b": rps_kind_text(params["b"])}
    elif note.code == "load":
        params = {"a": _rps(params["a"]), "b": _rps(params["b"])}
    return params


def reason_text(note: Note) -> str:
    return t(f"compare.reason.{note.code}", **note_params(note))


def flag_text(note: Note) -> str:
    return t(f"compare.flag.{note.code}", **note_params(note))


def outcome_title(code: str) -> str:
    return t(f"compare.outcome.{code}")


def assessment_title(code: str) -> str:
    return t(f"compare.assessment.{code}")


def domain_titles(*reports: Optional[StoredRunReport]) -> dict[str, str]:
    """Titles the reports stored for their domains (run A wins)."""
    titles: dict[str, str] = {}
    for report in reversed([item for item in reports if item is not None and item.final is not None]):
        stored = report.final.parsed.get("domain_titles")
        if isinstance(stored, Mapping):
            titles.update({str(key): str(value) for key, value in stored.items() if isinstance(value, str) and value.strip()})
    return titles


def domain_title(domain: str, titles: Mapping[str, str]) -> str:
    if titles.get(domain):
        return titles[domain]
    key = f"common.domain.{domain}"
    return t(key) if has_key(key) else domain


def _ordered_domains(domains: Sequence[str]) -> list[str]:
    known = [domain for domain in DOMAIN_ORDER if domain in domains]
    return known + sorted(domain for domain in domains if domain not in DOMAIN_ORDER)


def _round(value: Optional[float]) -> Optional[float]:
    """Numbers of the facts: whole above 1000, else four significant digits."""
    if value is None:
        return None
    if abs(value) >= 1000:
        return float(round(value))
    return float(f"{value:.4g}")


# ---- context -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompareContext:
    run_a: str
    run_b: str
    sides: dict[str, RunFacts]
    reports: dict[str, Optional[StoredRunReport]]
    metrics: MetricsSummary
    outcome: Outcome
    flags: list[Note]
    findings: list[tuple[str, StoredFinding]]
    titles: dict[str, str]
    payload: dict[str, Any]
    inputs_hash: str
    llm: Optional[LlmChoice]
    llm_problem: str
    lang: str = "ru"

    @property
    def available(self) -> bool:
        """Something to compare: reports of both runs or at least one metric both runs stored."""
        both_reports = all(self.sides[side].has_report for side in SIDES)
        return both_reports or bool(self.metrics.changes)


def _storage_cfg() -> dict[str, Any]:
    storage = CONFIG.get("storage") if isinstance(CONFIG.get("storage"), Mapping) else {}
    timescale = storage.get("timescale") if isinstance(storage.get("timescale"), Mapping) else {}
    return dict(timescale)


def _storage_names() -> tuple[str, str, str]:
    cfg = _storage_cfg()
    return (
        str(cfg.get("schema") or "public"),
        str(cfg.get("llm_table") or "llm_reports"),
        str(cfg.get("compare_table") or COMPARE_EXPLANATIONS_TABLE),
    )


def query_roles(area: Optional[str], service: Optional[str] = None) -> dict[str, dict[str, str]]:
    """``{domain: {label: role}}`` of the queries of a scope; the global queries without an area."""
    try:
        if area:
            from update_page import _effective_config_for_scope

            cfg = _effective_config_for_scope(area, service or None)
        else:
            cfg = CONFIG
        queries = cfg.get("queries") if isinstance(cfg.get("queries"), Mapping) else {}
        return {str(domain): block_roles(block) for domain, block in queries.items() if isinstance(block, Mapping)}
    except Exception:
        return {}


def _scope(sides: Mapping[str, RunFacts]) -> tuple[Optional[str], Optional[str]]:
    """Project area and service whose query roles give the direction of the metrics: those of A, else of B."""
    for side in SIDES:
        facts = sides[side]
        if facts.service:
            return facts.project_area or _find_area_for_service(facts.service), facts.service
    return None, None


def _language(reports: Mapping[str, Optional[StoredRunReport]]) -> str:
    for side in SIDES:
        report = reports.get(side)
        if report is not None and report.final is not None:
            return stored_report_language(report.final)
    return "ru"


def _system(reports: Mapping[str, Optional[StoredRunReport]]) -> dict[str, Any]:
    for side in SIDES:
        report = reports.get(side)
        if report is None:
            continue
        row: Optional[StoredRunRow] = report.final or next(iter(report.rows.values()), None)
        if row is not None and row.system_context:
            return compact_system(row.system_context)
    return {}


def _run_payload(facts: RunFacts) -> dict[str, Any]:
    if not facts.has_report:
        return {"run_name": facts.run_name, "report": t("compare.facts.no_report"), "service": facts.service or None}
    out: dict[str, Any] = {
        "run_name": facts.run_name,
        "service": facts.service or None,
        "test_type": type_text(facts.type_key),
        "report_date": facts.created_at.date().isoformat() if facts.created_at else None,
        "window_min": round(facts.window_min) if facts.window_min else None,
        "verdict": _verdict_text(facts.verdict) if facts.verdict else None,
        "sla_verdict": _verdict_text(facts.sla_verdict) if facts.sla_verdict else None,
        "key_rps": _round(facts.key_rps),
        "key_rps_kind": rps_kind_text(facts.key_rps_kind) or None,
    }
    return {key: value for key, value in out.items() if value is not None}


def domain_verdicts(reports: Mapping[str, Optional[StoredRunReport]], titles: Mapping[str, str]) -> list[dict[str, Any]]:
    domains = sorted({domain for report in reports.values() if report is not None for domain in report.rows if domain != "final"})
    out = []
    for domain in _ordered_domains(domains):
        row = {"domain": domain_title(domain, titles)}
        for side in SIDES:
            report = reports.get(side)
            stored = report.rows.get(domain) if report is not None else None
            row[side] = _verdict_text(stored.verdict) if stored is not None and stored.verdict else "—"
        out.append(row)
    return out


def scenario_of(report: Optional[StoredRunReport]) -> Optional[Mapping[str, Any]]:
    """``scenario`` of the final payload of a scenario run, else None."""
    final = report.final if report is not None else None
    parsed = final.parsed if final is not None and isinstance(final.parsed, Mapping) else {}
    scenario = parsed.get("scenario")
    return scenario if isinstance(scenario, Mapping) and scenario.get("detected") else None


def scenario_pairs(reports: Mapping[str, Optional[StoredRunReport]]) -> Optional[dict[str, Any]]:
    """Phases of two scenario runs aligned (AI.phases.phase_pairs), or None unless both runs are scenarios."""
    scenarios = {side: scenario_of(reports.get(side)) for side in SIDES}
    if not all(scenarios.values()):
        return None
    return phase_pairs(scenarios["A"], scenarios["B"])


def phase_flags(pairs: Optional[Mapping[str, Any]]) -> list[Note]:
    """Phases of one scenario run that the other lacks: the runs did not walk the same scenario."""
    if not pairs or not (pairs.get("only_in_a") or pairs.get("only_in_b")):
        return []
    return [Note("phases", {"a": pairs.get("only_in_a", 0), "b": pairs.get("only_in_b", 0)})]


def _phases_payload(pairs: Mapping[str, Any]) -> dict[str, Any]:
    """The phase table of the model: both labels, durations, mean RPS and verdicts, and the change of A."""
    rows = []
    for pair in pairs.get("pairs") or []:
        a, b = pair.get("a") or {}, pair.get("b") or {}
        rows.append({
            "A": a.get("label") or None, "B": b.get("label") or None,
            "A_min": a.get("duration_min"), "B_min": b.get("duration_min"),
            "duration_change_pct": pair.get("duration_change_pct"),
            "A_rps": _round(_float(a.get("rps_mean"))), "B_rps": _round(_float(b.get("rps_mean"))),
            "rps_change_pct": pair.get("rps_change_pct"),
            "A_verdict": _verdict_text(a.get("verdict")) if a.get("verdict") else None,
            "B_verdict": _verdict_text(b.get("verdict")) if b.get("verdict") else None,
        })
    return {"pairs": rows, "only_in_A": pairs.get("only_in_a", 0), "only_in_B": pairs.get("only_in_b", 0)}


def _only_in(items: Sequence[tuple[str, str]], titles: Mapping[str, str]) -> dict[str, Any]:
    return {"count": len(items), "metrics": [f"{domain_title(domain, titles)}: {label}" for domain, label in items[:MAX_ONLY_IN]]}


def _payload(context_parts: dict[str, Any]) -> dict[str, Any]:
    """Facts for the model in the current locale (the language of the report of A)."""
    sides: Mapping[str, RunFacts] = context_parts["sides"]
    reports = context_parts["reports"]
    metrics: MetricsSummary = context_parts["metrics"]
    outcome: Outcome = context_parts["outcome"]
    titles = context_parts["titles"]
    counts = metrics.counts()
    facts = {
        "runs": {side: _run_payload(sides[side]) for side in SIDES},
        "outcome": {"result": outcome_title(outcome.code), "reasons": [reason_text(note) for note in outcome.reasons]},
        "comparability": [flag_text(note) for note in context_parts["flags"]],
        "domain_verdicts": domain_verdicts(reports, titles),
        "metrics": {
            "aggregate": AGGREGATE,
            "significant_change_pct": CHANGE_THRESHOLD_PCT,
            "better": counts[BETTER],
            "worse": counts[WORSE],
            "unchanged": counts[SAME],
            "direction_unknown": counts[UNKNOWN],
            "top_changes": [
                {
                    "domain": domain_title(change.domain, titles),
                    "metric": change.metric,
                    "A": _round(change.value_a),
                    "B": _round(change.value_b),
                    "change_pct": round(change.change_pct, 1) if change.change_pct is not None else None,
                    "assessment": assessment_title(change.assessment),
                }
                for change in metrics.top()
            ],
            "only_in_A": _only_in(metrics.only_in_a, titles),
            "only_in_B": _only_in(metrics.only_in_b, titles),
        },
    }
    if context_parts.get("phases"):
        facts["phases"] = _phases_payload(context_parts["phases"])
    findings = {side: [compare_finding_payload(s, finding) for s, finding in context_parts["findings"] if s == side] for side in SIDES}
    return {"facts": facts, "findings": findings, "system": _system(reports)}


def _prompt(lang: str = "ru") -> str:
    return load_prompt(PROMPT_FILE, lang)


def _load_reports(run_a: str, run_b: str) -> tuple[dict[str, Optional[StoredRunReport]], list[tuple]]:
    schema, llm_table, _ = _storage_names()
    conn = _ts_conn()
    try:
        reports = {
            "A": load_run_report(conn, schema, llm_table, run_a),
            "B": load_run_report(conn, schema, llm_table, run_b),
        }
        rows = load_pair_summary(conn, run_a, run_b, P95_SQL)
    finally:
        conn.close()
    return reports, rows


def build_context(run_a: str, run_b: str) -> CompareContext:
    """Facts, outcome and flags of the pair; the hash covers the prompt text of the language of A and the payload,
    so the locale of the reader does not change it."""
    reports, rows = _load_reports(run_a, run_b)
    sides = {"A": run_facts("A", run_a, reports["A"]), "B": run_facts("B", run_b, reports["B"])}
    metrics = summarize_metrics(rows, query_roles(*_scope(sides)))
    findings = [(side, finding) for side in SIDES for finding in picked_findings(reports[side])]
    phases = scenario_pairs(reports)
    parts = {
        "sides": sides,
        "reports": reports,
        "metrics": metrics,
        "outcome": outcome_of(sides["A"], sides["B"], metrics),
        "flags": flags_of(sides["A"], sides["B"], metrics) + phase_flags(phases),
        "findings": findings,
        "titles": domain_titles(reports["A"], reports["B"]),
    }
    lang = _language(reports)
    with with_locale(lang):
        payload = _payload({**parts, "phases": phases})
    digest = hashlib.sha256(
        json.dumps({"prompt": _prompt(lang), "payload": payload}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    llm, llm_problem = llm_choice()
    return CompareContext(
        run_a=run_a,
        run_b=run_b,
        payload=payload,
        inputs_hash=digest,
        llm=llm,
        llm_problem=llm_problem,
        lang=lang,
        **parts,
    )


# ---- view --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompareView:
    status: ViewStatus
    message: str
    context: Optional[CompareContext]
    explanation: Optional[CompareExplanation] = None
    check: Optional[ExplanationCheck] = None
    model: str = ""
    generated_at: Optional[str] = None

    def _runs(self) -> dict[str, Any]:
        out = {}
        for side in SIDES:
            facts = self.context.sides[side]
            out[side] = {
                "run_name": facts.run_name,
                "has_report": facts.has_report,
                "service": facts.service,
                "test_type": facts.type_key if facts.has_report else "",
                "verdict": facts.verdict,
                "key_rps": facts.key_rps,
                "key_rps_kind": facts.key_rps_kind,
                "window_min": facts.window_min,
                "created_at": facts.created_at.isoformat() if facts.created_at else None,
            }
        return out

    def _metrics(self) -> dict[str, Any]:
        metrics = self.context.metrics
        titles = self.context.titles
        return {
            "aggregate": AGGREGATE,
            "threshold_pct": CHANGE_THRESHOLD_PCT,
            "counts": metrics.counts(),
            "top_changes": [
                {**asdict(change), "domain_title": domain_title(change.domain, titles)} for change in metrics.top()
            ],
            "only_in_a": len(metrics.only_in_a),
            "only_in_b": len(metrics.only_in_b),
        }

    def to_dict(self) -> dict[str, Any]:
        """Texts of the outcome and the flags in the locale of the request; the explanation in the language of A."""
        out: dict[str, Any] = {
            "status": self.status,
            "message": self.message,
            "explanation": self.explanation.model_dump() if self.explanation is not None else None,
            "check": asdict(self.check) if self.check is not None else None,
            "model": self.model,
            "generated_at": self.generated_at,
        }
        context = self.context
        if context is None:
            return {**out, "run_a": "", "run_b": "", "language": "", "llm_problem": "", "runs": {}, "outcome": None,
                    "flags": [], "metrics": None, "findings": []}
        outcome = context.outcome
        return {
            **out,
            "run_a": context.run_a,
            "run_b": context.run_b,
            "language": context.lang,
            "llm_problem": context.llm_problem,
            "runs": self._runs(),
            "outcome": {"code": outcome.code, "title": outcome_title(outcome.code), "reasons": [reason_text(note) for note in outcome.reasons]},
            "flags": [{"code": note.code, "text": flag_text(note)} for note in context.flags],
            "metrics": self._metrics(),
            "findings": [{"side": side, "id": side_ref(side, finding), **asdict(finding)} for side, finding in context.findings],
        }


def _load_cached(run_a: str, run_b: str) -> Optional[StoredPairExplanation]:
    schema, _, table = _storage_names()
    conn = _ts_conn()
    try:
        return load_compare_explanation(conn, schema, table, run_a, run_b)
    finally:
        conn.close()


def _save(run_a: str, run_b: str, inputs_hash: str, payload: dict[str, Any]) -> None:
    schema, _, table = _storage_names()
    conn = _ts_conn()
    try:
        save_compare_explanation(conn, schema, table, run_a, run_b, inputs_hash, payload)
    finally:
        conn.close()


def _cached_view(status: ViewStatus, context: CompareContext, payload: Mapping[str, Any]) -> CompareView:
    model = str(payload.get("model") or "")
    generated_at = str(payload.get("generated_at") or "") or None
    try:
        explanation = CompareExplanation.model_validate(payload["explanation"])
        check = ExplanationCheck(**payload["check"])
    except (KeyError, TypeError, ValidationError):
        return CompareView("stale", t("compare.explain.old_format"), context, model=model, generated_at=generated_at)
    message = t("compare.explain.stale") if status == "stale" else ""
    return CompareView(status, message, context, explanation, check, model, generated_at)


def explanation_view(run_a: str, run_b: str) -> CompareView:
    """Outcome, flags and key changes of the pair with the cached explanation, if any."""
    context = build_context(run_a, run_b)
    if not context.available:
        return CompareView("unavailable", t("compare.explain.nothing"), context)
    stored = _load_cached(run_a, run_b)
    if stored is None:
        model = context.llm.model if context.llm is not None else ""
        return CompareView("missing", "", context, model=model)
    return _cached_view("ready" if stored.inputs_hash == context.inputs_hash else "stale", context, stored.payload)


# ---- generation --------------------------------------------------------------------------------------------------

_slots_lock = threading.Lock()
_running: set[tuple[str, str]] = set()
_last_started: dict[tuple[str, str], float] = {}


@contextmanager
def _generation_slot(key: tuple[str, str]) -> Iterator[None]:
    """One generation per pair at a time and not more often than MIN_REGENERATE_SEC."""
    now = time.monotonic()
    with _slots_lock:
        if key in _running:
            raise ExplanationError(msg("compare.explain.busy"), 409, "busy")
        last = _last_started.get(key)
        if last is not None and now - last < MIN_REGENERATE_SEC:
            wait = int(MIN_REGENERATE_SEC - (now - last)) + 1
            raise ExplanationError(msg("compare.explain.too_often", wait=wait), 429, "too_often")
        _running.add(key)
        _last_started[key] = now
    try:
        yield
    finally:
        with _slots_lock:
            _running.discard(key)


def _ask_model(context: CompareContext) -> str:
    llm = context.llm
    if llm is None:
        raise ExplanationError(context.llm_problem, 422, "missing_llm")
    try:
        with usage_domain("compare"):
            return ask_llm_with_text_data(
                _prompt(context.lang),
                json.dumps(context.payload, ensure_ascii=False),
                llm_config={"force_json": True},
                system_prompt=prompt_text("compare.system", context.lang),
            )
    except Exception as exc:
        raise ExplanationError(
            msg("compare.explain.llm_failed", provider=llm.provider, model=llm.model, error=str(exc)), 502, "llm_failed"
        ) from exc


def generate_explanation(run_a: str, run_b: str, budget_owner: Optional[str] = None) -> CompareView:
    """Asks the model, checks its numbers and caches the result for the pair.

    With ``budget_owner`` the call counts against that user's daily LLM budget (the kind of the forecast
    explanation: both are short explanations of facts the program computed).
    """
    with _generation_slot((run_a, run_b)):
        context = build_context(run_a, run_b)
        if not context.available:
            raise ExplanationError(msg("compare.explain.nothing"), 422, "no_data")
        if budget_owner is not None and context.llm is not None:
            consume_llm(budget_owner, "explanation")
        with with_locale(context.lang):
            raw = _ask_model(context)
            explanation = parse_comparison(raw)
            check = check_comparison(explanation, context.payload["facts"], context.findings)
        payload = {
            "explanation": explanation.model_dump(),
            "check": asdict(check),
            "model": context.llm.model if context.llm is not None else "",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(run_a, run_b, context.inputs_hash, payload)
        return _cached_view("ready", context, payload)


__all__ = [
    "AGGREGATE",
    "CHANGE_THRESHOLD_PCT",
    "CompareContext",
    "CompareView",
    "MetricChange",
    "MetricsSummary",
    "Note",
    "Outcome",
    "RunFacts",
    "build_context",
    "domain_title",
    "domain_verdicts",
    "explanation_view",
    "flags_of",
    "generate_explanation",
    "key_rps",
    "outcome_of",
    "phase_flags",
    "picked_findings",
    "query_roles",
    "rps_kind_text",
    "run_facts",
    "scenario_of",
    "scenario_pairs",
    "summarize_metrics",
    "type_text",
]
