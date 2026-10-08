"""LLM explanation of the forecast load limit, grounded in the domain findings of the report.

Related findings are picked without the model: they overlap the window where the limit
showed up or name the service whose CPU explains it. The model only links and explains
them; the numbers it cites are checked against the forecast facts and those findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from AI.context_pack import to_utc
from AI.db_store import StoredFinding
from AI.scoring import VERIFICATION_UNVERIFIED, FindingItem, _extract_json_like, _extract_numeric_claims
from AI.verification import verify_finding

MAX_FINDINGS = 12
MAX_EVIDENCE_CHARS = 500
MENTION_WEIGHT = 4
STARTS_INSIDE_WEIGHT = 2
OVERLAP_WEIGHT = 1
SEVERITY_WEIGHT = {"critical": 2, "high": 1}
DIGIT_GROUP_RE = re.compile(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))")
RAW_EXCERPT_CHARS = 200


class ExplanationParseError(ValueError):
    """The model answer is not the expected JSON."""


@dataclass(frozen=True)
class LimitWindow:
    start: pd.Timestamp
    end: pd.Timestamp


class EvidenceLink(BaseModel):
    finding_id: str
    role: str


class KeyFact(BaseModel):
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)


class ScalingAssessment(BaseModel):
    """Whether adding instances of the bottleneck service lifts the limit."""

    verdict: Literal["helps", "partly", "no"]
    reason: str = Field(min_length=1)


class LimitExplanation(BaseModel):
    headline: str = Field(min_length=1)
    cause: str = Field(min_length=1)
    key_facts: list[KeyFact] = Field(default_factory=list)
    scaling: ScalingAssessment
    evidence: list[EvidenceLink] = Field(default_factory=list)
    scaling_risks: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]


@dataclass(frozen=True)
class ExplanationCheck:
    """Numbers of the explanation found in the facts or findings, and evidence ids that do not exist."""

    status: str
    claims_total: int
    claims_matched: int
    unmatched: list[str]
    unknown_findings: list[str]


def _span(finding: StoredFinding) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    if not finding.start_time:
        return None
    try:
        start = to_utc(finding.start_time)
        end = to_utc(finding.end_time) if finding.end_time else start
    except (TypeError, ValueError):
        return None
    if pd.isna(start) or pd.isna(end):
        return None
    return start, max(start, end)


def _relevance(finding: StoredFinding, window: LimitWindow, services: Sequence[str]) -> Optional[int]:
    span = _span(finding)
    overlaps = span is not None and span[0] <= window.end and span[1] >= window.start
    text = f"{finding.component} {finding.summary}"
    mentions = any(name in text for name in services if name)
    if not (overlaps or mentions):
        return None
    starts_inside = span is not None and window.start <= span[0] <= window.end
    return (
        MENTION_WEIGHT * mentions
        + STARTS_INSIDE_WEIGHT * starts_inside
        + OVERLAP_WEIGHT * overlaps
        + SEVERITY_WEIGHT.get(finding.severity, 0)
    )


def related_findings(findings: Sequence[StoredFinding], window: LimitWindow, services: Sequence[str]) -> list[StoredFinding]:
    """Findings that overlap the limit window or name a focus service, the most specific first.

    The best finding of every domain is kept so shared components (database, queues, nodes)
    reach the model; findings whose own numbers failed the report verification are left out.
    """
    scored: list[tuple[int, StoredFinding]] = []
    for finding in findings:
        if finding.verification == VERIFICATION_UNVERIFIED:
            continue
        score = _relevance(finding, window, services)
        if score is not None:
            scored.append((score, finding))
    ordered = [finding for _, finding in sorted(scored, key=lambda pair: (-pair[0], pair[1].start_time, pair[1].ref))]
    picked: dict[str, StoredFinding] = {}
    for finding in ordered:
        picked.setdefault(finding.domain, finding)
    chosen = {finding.ref for finding in picked.values()}
    for finding in ordered:
        if len(chosen) >= MAX_FINDINGS:
            break
        chosen.add(finding.ref)
    return [finding for finding in ordered if finding.ref in chosen]


def finding_payload(finding: StoredFinding) -> dict[str, str]:
    return {
        "id": finding.ref,
        "domain": finding.domain,
        "severity": finding.severity,
        "component": finding.component,
        "start": finding.start_time,
        "end": finding.end_time,
        "summary": finding.summary,
        "evidence": finding.evidence_summary[:MAX_EVIDENCE_CHARS],
    }


def parse_explanation(raw: str) -> LimitExplanation:
    data = _extract_json_like(raw or "")
    if data is None:
        raise ExplanationParseError(f"Ответ модели не содержит JSON: «{(raw or '')[:RAW_EXCERPT_CHARS]}»")
    try:
        return LimitExplanation.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
        raise ExplanationParseError(f"Ответ модели не совпал со схемой разбора, поля: {fields}") from exc


def _grouped(text: str) -> str:
    return DIGIT_GROUP_RE.sub("", text)


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    if isinstance(value, str):
        return [float(claim["value"]) for claim in _extract_numeric_claims(_grouped(value))]
    return []


def _explanation_text(explanation: LimitExplanation) -> str:
    parts = [
        explanation.headline,
        explanation.cause,
        *(f"{fact.label} {fact.value}" for fact in explanation.key_facts),
        explanation.scaling.reason,
        *(link.role for link in explanation.evidence),
        *explanation.scaling_risks,
        *explanation.next_checks,
    ]
    return _grouped(" ".join(parts))


def check_explanation(explanation: LimitExplanation, facts: dict[str, Any], findings: Sequence[StoredFinding]) -> ExplanationCheck:
    """Numbers the model cites must come from the facts or the related findings; evidence ids must exist."""
    values = _numbers(facts) + [number for finding in findings for number in _numbers(f"{finding.summary} {finding.evidence_summary}")]
    refs = [{"value": value, "match_keys": []} for value in values]
    result = verify_finding(FindingItem(summary=_explanation_text(explanation)), refs)
    known = {finding.ref for finding in findings}
    return ExplanationCheck(
        status=result.status,
        claims_total=result.claims_total,
        claims_matched=result.claims_matched,
        unmatched=list(result.unmatched),
        unknown_findings=[link.finding_id for link in explanation.evidence if link.finding_id not in known],
    )


__all__ = [
    "EvidenceLink",
    "ExplanationCheck",
    "ExplanationParseError",
    "KeyFact",
    "LimitExplanation",
    "LimitWindow",
    "ScalingAssessment",
    "check_explanation",
    "finding_payload",
    "parse_explanation",
    "related_findings",
]
