# SPDX-License-Identifier: Apache-2.0
"""LLM explanation of the differences between two runs: the answer schema and the check of its numbers.

The program computes the facts and the outcome of the comparison (loadlens_app/compare_explanation.py); the model
only explains them. Findings reach the model with the side of the run in their id (``A:jvm:f1``, ``B:db:f2``);
the numbers it cites are checked against the facts and those findings, as for the forecast limit
(AI/limit_explanation.py).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from AI.db_store import StoredFinding
from AI.limit_explanation import (
    MAX_EVIDENCE_CHARS,
    RAW_EXCERPT_CHARS,
    EvidenceLink,
    ExplanationCheck,
    ExplanationParseError,
    numbers_in,
    ungrouped,
)
from AI.scoring import FindingItem, _extract_json_like
from AI.verification import verify_finding

SIDES: tuple[str, ...] = ("A", "B")


class KeyChange(BaseModel):
    label: str = Field(min_length=1)
    value_a: str = ""
    value_b: str = ""
    assessment: Literal["better", "worse", "same", "unknown"] = "unknown"


class CompareExplanation(BaseModel):
    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_changes: list[KeyChange] = Field(default_factory=list)
    causes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]


def side_ref(side: str, finding: StoredFinding) -> str:
    """Id of a finding of run A or B in the payload and in the evidence of the answer: ``A:jvm:f1``."""
    return f"{side}:{finding.ref}"


def compare_finding_payload(side: str, finding: StoredFinding) -> dict[str, str]:
    return {
        "id": side_ref(side, finding),
        "domain": finding.domain,
        "severity": finding.severity,
        "component": finding.component,
        "start": finding.start_time,
        "end": finding.end_time,
        "summary": finding.summary,
        "evidence": finding.evidence_summary[:MAX_EVIDENCE_CHARS],
    }


def parse_comparison(raw: str) -> CompareExplanation:
    data = _extract_json_like(raw or "")
    if data is None:
        raise ExplanationParseError("capacity.explain.no_json", excerpt=(raw or "")[:RAW_EXCERPT_CHARS])
    try:
        return CompareExplanation.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
        raise ExplanationParseError("capacity.explain.bad_schema", fields=fields) from exc


def _comparison_text(explanation: CompareExplanation) -> str:
    parts = [
        explanation.headline,
        explanation.summary,
        *(f"{change.label} {change.value_a} {change.value_b}" for change in explanation.key_changes),
        *explanation.causes,
        *(link.role for link in explanation.evidence),
        *explanation.caveats,
        *explanation.next_checks,
    ]
    return ungrouped(" ".join(parts))


def check_comparison(
    explanation: CompareExplanation,
    facts: dict[str, Any],
    findings: Sequence[tuple[str, StoredFinding]],
) -> ExplanationCheck:
    """Numbers the model cites must come from the facts or the findings of both runs; evidence ids must exist.

    ``findings`` are ``(side, finding)`` pairs, the side being ``A`` or ``B``.
    """
    values = numbers_in(facts) + [
        number for _, finding in findings for number in numbers_in(f"{finding.summary} {finding.evidence_summary}")
    ]
    refs = [{"value": value, "match_keys": []} for value in values]
    result = verify_finding(FindingItem(summary=_comparison_text(explanation)), refs)
    known = {side_ref(side, finding) for side, finding in findings}
    return ExplanationCheck(
        status=result.status,
        claims_total=result.claims_total,
        claims_matched=result.claims_matched,
        unmatched=list(result.unmatched),
        unknown_findings=[link.finding_id for link in explanation.evidence if link.finding_id not in known],
    )


__all__ = [
    "SIDES",
    "CompareExplanation",
    "KeyChange",
    "check_comparison",
    "compare_finding_payload",
    "parse_comparison",
    "side_ref",
]
