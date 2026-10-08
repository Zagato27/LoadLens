"""Programmatic check of LLM findings against the collected numbers.

Every number a finding cites (summary, evidence summary, evidence items) is
looked up in the numeric catalog built from the analysis context: series
statistics, the step table, step anomalies, timeline, baseline, the designated
peak and deterministic SLA checks. Findings are marked ``verified``,
``unverified`` (numbers present but mostly not found) or ``qualitative``
(no numbers at all). Unverified findings stay in the report but are excluded
from the verdict rationale by the verification pass of the model.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from AI.scoring import (
    VERIFICATION_QUALITATIVE,
    VERIFICATION_UNVERIFIED,
    VERIFICATION_VERIFIED,
    FindingItem,
    FindingVerification,
    LLMAnalysis,
    VerificationSummary,
    _build_numeric_reference_catalog,
    _claim_value_variants,
    _extra_numeric_references,
    _extract_numeric_claims,
    _extract_sections_from_context,
    _finding_text_blob,
    _matching_numeric_references,
    _safe_float_value,
)

RELATED_TOLERANCE = 0.05
GLOBAL_TOLERANCE = 0.02
# Rounding slack for values the model is likely to round (>= 10): "226 RPS" for 226.5.
ROUNDING_SLACK = 0.5
ROUNDING_SLACK_MIN_VALUE = 10.0
VERIFIED_SHARE = 0.6

# Step / interval / finding identifiers are references, not measurements.
_STEP_REFERENCE_RE = re.compile(r"(ступен[иьяе]|интервал[аеы]?|шаг[аеи]?|step|interval)\s*№?\s*\d+", re.IGNORECASE)
_FINDING_ID_RE = re.compile(r"\bf\d+\b", re.IGNORECASE)
_CYRILLIC_MS_RE = re.compile(r"(\d)\s*мс\b", re.IGNORECASE)
_CYRILLIC_S_RE = re.compile(r"(\d)\s*с\b")


def build_reference_catalog(ctx_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All numbers of the context the model may legitimately cite."""
    return _build_numeric_reference_catalog(_extract_sections_from_context(ctx_obj)) + _extra_numeric_references(ctx_obj)


def _measurement_text(finding: FindingItem) -> str:
    """Finding text with references (step numbers, ids) removed and RU units normalized."""
    blob = _finding_text_blob(finding)
    blob = _STEP_REFERENCE_RE.sub(" ", blob)
    blob = _FINDING_ID_RE.sub(" ", blob)
    blob = _CYRILLIC_MS_RE.sub(r"\1 ms", blob)
    blob = _CYRILLIC_S_RE.sub(r"\1 s", blob)
    return blob


def _claim_matches(claim: Dict[str, Any], refs: Sequence[Dict[str, Any]], tolerance: float) -> bool:
    value = _safe_float_value(claim.get("value"))
    if value is None:
        return False
    variants = _claim_value_variants(value, str(claim.get("unit") or ""))
    for ref in refs:
        ref_value = _safe_float_value(ref.get("value"))
        if ref_value is None:
            continue
        allowed = abs(ref_value) * tolerance
        if abs(ref_value) >= ROUNDING_SLACK_MIN_VALUE:
            allowed = max(ROUNDING_SLACK, allowed)
        if any(abs(variant - ref_value) <= allowed for variant in variants):
            return True
    return False


def _format_claim(claim: Dict[str, Any]) -> str:
    value = float(claim.get("value") or 0.0)
    text = f"{value:g}"
    unit = str(claim.get("unit") or "")
    return f"{text} {unit}".strip()


def verify_finding(finding: FindingItem, refs: Sequence[Dict[str, Any]]) -> FindingVerification:
    """Matches the finding's numbers against related references first, then the whole catalog."""
    blob = _measurement_text(finding)
    claims = _extract_numeric_claims(blob)
    if not claims:
        return FindingVerification(status=VERIFICATION_QUALITATIVE)
    related = _matching_numeric_references(blob, refs)
    matched = 0
    unmatched: List[str] = []
    for claim in claims:
        if _claim_matches(claim, related, RELATED_TOLERANCE) or _claim_matches(claim, refs, GLOBAL_TOLERANCE):
            matched += 1
        else:
            unmatched.append(_format_claim(claim))
    status = VERIFICATION_VERIFIED if matched / len(claims) >= VERIFIED_SHARE else VERIFICATION_UNVERIFIED
    return FindingVerification(status=status, claims_total=len(claims), claims_matched=matched, unmatched=unmatched)


def verify_analysis(parsed: LLMAnalysis, ctx_obj: Dict[str, Any], revised_by_model: bool = False) -> Tuple[LLMAnalysis, VerificationSummary]:
    """Returns a copy of the analysis with per-finding verification and the summary."""
    refs = build_reference_catalog(ctx_obj)
    findings: List[FindingItem] = []
    counts = {VERIFICATION_VERIFIED: 0, VERIFICATION_UNVERIFIED: 0, VERIFICATION_QUALITATIVE: 0}
    for finding in parsed.findings or []:
        verification = verify_finding(finding, refs)
        counts[verification.status] += 1
        findings.append(finding.copy(update={"verification": verification}))
    summary = VerificationSummary(
        total=len(findings),
        verified=counts[VERIFICATION_VERIFIED],
        unverified=counts[VERIFICATION_UNVERIFIED],
        qualitative=counts[VERIFICATION_QUALITATIVE],
        revised_by_model=revised_by_model,
    )
    return parsed.copy(update={"findings": findings, "verification_summary": summary}), summary


def unverified_findings(parsed: LLMAnalysis) -> List[Dict[str, Any]]:
    """Compact list of unverified findings for the verification pass prompt."""
    out: List[Dict[str, Any]] = []
    for finding in parsed.findings or []:
        verification = finding.verification
        if verification is None or verification.status != VERIFICATION_UNVERIFIED:
            continue
        out.append({"id": finding.id, "summary": finding.summary, "unmatched_numbers": list(verification.unmatched)})
    return out


__all__ = [
    "build_reference_catalog",
    "unverified_findings",
    "verify_analysis",
    "verify_finding",
]
