# SPDX-License-Identifier: Apache-2.0
"""Regression checks for numeric evidence shared by reports and explanations."""

import pytest

from AI.limit_explanation import LimitExplanation, check_explanation
from AI.scoring import FindingItem, LLMAnalysis, _extract_numeric_claims, score_candidate_by_data_details
from AI.verification import verify_finding


@pytest.mark.parametrize("text", ["CPU 75%", "CPU 75 %", "CPU 75%.", "CPU 75\u00a0%"])
@pytest.mark.parametrize("source", [0.75, 75.0])
def test_percent_claims_support_both_source_scales(text, source):
    assert _extract_numeric_claims(text) == [{"value": 75.0, "unit": "%"}]
    result = verify_finding(FindingItem(summary=text), [{"value": source, "match_keys": ["cpu"]}])
    assert (result.status, result.claims_matched) == ("verified", 1)


@pytest.mark.parametrize(("count", "matched"), [(12, 12), (13, 12), (13, 13)])
def test_every_distinct_measurement_is_checked(count, matched):
    summary = ", ".join(f"{n} ms" for n in range(1, count + 1))
    refs = [{"value": n, "match_keys": []} for n in range(1, matched + 1)]
    result = verify_finding(FindingItem(summary=summary), refs)
    assert (result.claims_total, result.claims_matched) == (count, matched)
    assert result.unmatched == ([] if count == matched else ["13 ms"])


def test_claim_boundaries_preserve_words_and_measurement_units():
    assert _extract_numeric_claims("p95: 15 ms, 1 s, 2 sec, 75 and 75 %.") == [
        {"value": 15.0, "unit": "ms"}, {"value": 1.0, "unit": "s"},
        {"value": 2.0, "unit": "sec"}, {"value": 75.0, "unit": ""}, {"value": 75.0, "unit": "%"},
    ]


def test_forecast_and_candidate_scoring_count_all_measurements():
    text = ", ".join(f"{n} ms" for n in range(1, 14))
    explanation = LimitExplanation(headline=text, cause="Delay", scaling={"verdict": "no", "reason": "Unknown"}, confidence="low")
    checked = check_explanation(explanation, {"values": list(range(1, 13))}, [])
    assert (checked.claims_matched, checked.claims_total, checked.unmatched) == (12, 13, ["13 ms"])
    candidate = LLMAnalysis(findings=[FindingItem(summary=text)])
    assert score_candidate_by_data_details(candidate, {})["numeric_claims_total"] == 13
