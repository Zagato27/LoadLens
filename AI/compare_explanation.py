# SPDX-License-Identifier: Apache-2.0
"""LLM explanation of the differences between two runs: the answer schema and the check of its numbers.

The program computes the facts and the outcome of the comparison (loadlens_app/compare_explanation.py); the model
only explains them. Findings reach the model with the side of the run in their id (``A:jvm:f1``, ``B:db:f2``);
the numbers it cites are checked against the facts and those findings, as for the forecast limit
(AI/limit_explanation.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence

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
from AI.scoring import (
    VERIFICATION_QUALITATIVE, VERIFICATION_UNVERIFIED, VERIFICATION_VERIFIED,
    FindingItem, _extract_json_like, _extract_numeric_claims,
)
from AI.verification import verify_finding

SIDES: tuple[str, ...] = ("A", "B")
_THROUGHPUT_NAMES = ("rps", "throughput", "req/s", "пропускная способность")
_EN_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?!\d|,\d)")


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


@dataclass(frozen=True)
class _Measure:
    label: str
    aliases: tuple[str, ...]
    values: dict[str, Any]
    qualifier: str = ""


def _measures(facts: dict[str, Any]) -> list[_Measure]:
    runs = facts.get("runs", {})
    result = []
    for field, label, aliases in (
        ("key_rps", "RPS", _THROUGHPUT_NAMES),
        ("window_min", "window_min", ("window", "duration", "min", "minutes", "мин")),
    ):
        values = {side: runs.get(side, {}).get(field) for side in SIDES}
        if any(v is not None for v in values.values()):
            result.append(_Measure(label, aliases, values))
    for row in facts.get("metrics", {}).get("top_changes", []):
        label = row["metric"]
        aliases = _THROUGHPUT_NAMES if label.casefold() in _THROUGHPUT_NAMES else (label,)
        result.append(_Measure(label, aliases, row, row.get("domain", "")))
    for row in facts.get("phases", {}).get("pairs", []):
        for unit in ("min", "rps"):
            aliases = tuple(f"{row[side]} {unit}" for side in SIDES if row.get(side))
            if aliases:
                result.append(_Measure(aliases[0], aliases, {side: row.get(f"{side}_{unit}") for side in SIDES}))
    return result


def _resolve_measure(text: str, measures: list[_Measure]) -> _Measure | None:
    """Pick the most specific matching label/domain. Ambiguous values cannot confirm a claim."""
    words = set(re.findall(r"\w+", text.casefold()))
    candidates: list[tuple[int, _Measure]] = []
    for measure in measures:
        for alias in measure.aliases:
            tokens = set(re.findall(r"\w+", alias.casefold()))
            if tokens and tokens <= words:
                domain = set(re.findall(r"\w+", measure.qualifier.casefold()))
                score = len(tokens) + (len(domain) if domain <= words else 0)
                candidates.append((score, measure))
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    matching = [measure for score, measure in candidates if score == best]
    pairs = {tuple(measure.values.get(side) for side in SIDES) for measure in matching}
    return matching[0] if len(pairs) == 1 else None


def _attributed_parts(text: str) -> Iterator[tuple[str, str, str]]:
    """Split explicit Latin run markers, including both 'A: 1200' and '1200 (A)'."""
    for sentence in re.split(r"[;\n]|[.!?](?!\d)", text):
        parts = re.split(r"\b([AB])\b", sentence)
        if len(parts) == 1:
            yield "", sentence, sentence
        elif len(set(parts[1::2])) > 1 and not any(_extract_numeric_claims(part) for part in parts[2:-1:2]):
            # Both runs named together: the text does not assign each number to a side.
            yield "A/B", sentence, sentence
        elif _extract_numeric_claims(parts[0]) and not _extract_numeric_claims(parts[-1]):
            for before, side in zip(parts[::2], parts[1::2]):
                yield side, before, sentence
            yield "", parts[-1], sentence
        else:
            yield "", parts[0], sentence
            for side, after in zip(parts[1::2], parts[2::2]):
                yield side, after, sentence


def check_comparison(
    explanation: CompareExplanation,
    facts: dict[str, Any],
    findings: Sequence[tuple[str, StoredFinding]],
    *,
    lang: str = "en",
) -> ExplanationCheck:
    """Numbers the model cites must come from the facts or the findings of both runs; evidence ids must exist.

    ``findings`` are ``(side, finding)`` pairs, the side being ``A`` or ``B``.
    """
    def normalize(text: str) -> str:
        text = ungrouped(text)
        if lang == "en":
            return _EN_THOUSANDS.sub(lambda m: m.group().replace(",", ""), text)
        return text

    measures = _measures(facts)
    values = numbers_in(facts) + [
        number for _, finding in findings for number in numbers_in(normalize(f"{finding.summary} {finding.evidence_summary}"))
    ]
    known = {side_ref(side, finding): finding for side, finding in findings}
    total = matched = 0
    unmatched: list[str] = []

    def check(text: str, expected: list[float], label: str = "", other_value: float | None = None) -> None:
        nonlocal total, matched
        refs = [{"value": value, "match_keys": []} for value in expected]
        result = verify_finding(FindingItem(summary=text), refs, other_run_value=other_value)
        total += result.claims_total
        matched += result.claims_matched
        unmatched.extend(f"{label}: {item}" if label else item for item in result.unmatched)

    def check_side(text: str, side: str, measure: _Measure | None, label: str) -> None:
        expected = measure.values.get(side) if measure else None
        other_values = numbers_in(measure.values.get("B" if side == "A" else "A")) if measure else []
        other = other_values[0] if len(other_values) == 1 else None
        check(text, numbers_in(expected), f"{side} · {label}" if label else side, other)

    for change in explanation.key_changes:
        measure = _resolve_measure(change.label, measures)
        for side, text in (("A", change.value_a), ("B", change.value_b)):
            check_side(normalize(text), side, measure, change.label)
    for text in (explanation.headline, explanation.summary, *explanation.causes, *explanation.caveats, *explanation.next_checks):
        for side, part, sentence in _attributed_parts(normalize(text)):
            if side == "A/B":
                check(part, [], side)
            elif side:
                measure = _resolve_measure(part, measures) or _resolve_measure(sentence, measures)
                check_side(part, side, measure, measure.label if measure else "")
            else:
                check(part, values)
    for link in explanation.evidence:
        finding = known.get(link.finding_id)
        expected = numbers_in(normalize(f"{finding.summary} {finding.evidence_summary}")) if finding else []
        check(normalize(link.role), expected, link.finding_id)
    status = VERIFICATION_QUALITATIVE
    if total:
        status = VERIFICATION_UNVERIFIED if unmatched else VERIFICATION_VERIFIED
    return ExplanationCheck(
        status=status,
        claims_total=total,
        claims_matched=matched,
        unmatched=unmatched,
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
