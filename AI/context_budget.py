# SPDX-License-Identifier: Apache-2.0
"""Compact JSON evidence for chat endpoints with a limited context window."""

import json


def text_size(text: str) -> int:
    """Estimate conservatively without assuming the endpoint uses an OpenAI tokenizer."""
    return len(text.encode("utf-8"))


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# Keep decisions and figures ahead of debug data in a reduced context.
_PRIORITY = {
    key: rank for rank, key in enumerate((
        "domain", "verdict", "deterministic_sla", "target_rps_check", "passed",
        "checks", "test_profile", "designated_peak_performance", "time_range",
        "metric", "observed_value", "threshold", "label", "value", "domain_analyses",
        "findings", "summary", "evidence_items", "sections", "top_series", "timeline",
        "load_steps", "domains", "verdict_rationale", "recommended_actions",
    ))
}


def compact_json(text: str, budget: int) -> str:
    """Reduce complete JSON fields and rows. Return empty if no evidence fits."""
    if text_size(text) <= budget:
        return text
    try:
        source = json.loads(text)
    except (ValueError, TypeError):
        return ""

    def fit(value, available):
        encoded = _dump(value)
        if text_size(encoded) <= available:
            return encoded
        if isinstance(value, dict):
            keys = sorted(value, key=lambda key: _PRIORITY.get(key, len(_PRIORITY)))[:32]
            children = [(_dump(key) + ":", value[key]) for key in keys]
            left, right = "{", "}"
        elif isinstance(value, list):
            # Retain the first and last observation in a sampled time series.
            sample = value if len(value) <= 8 else [value[round(i * (len(value) - 1) / 7)] for i in range(8)]
            children = [("", item) for item in sample]
            left, right = "[", "]"
        elif isinstance(value, str) and available >= 32:
            short = value.encode("utf-8")[:available - 8].decode("utf-8", errors="ignore")
            result = _dump(short + "…")
            return result if text_size(result) <= available else ""
        else:
            return ""

        remaining = available - 2
        parts = []
        for index, (prefix, item) in enumerate(children):
            overhead = text_size(prefix) + bool(parts)
            # Reserve space for other signals, redistributing space left by small fields.
            share = (remaining - overhead) // min(4, len(children) - index)
            if share < 1:
                continue
            reduced = fit(item, share)
            if reduced:
                parts.append(prefix + reduced)
                remaining -= overhead + text_size(reduced)
        return left + ",".join(parts) + right if parts else ""

    result = fit(source, budget)
    sla = source.get("deterministic_sla") if isinstance(source, dict) else None
    if isinstance(sla, dict) and "verdict" in sla:
        essential = _dump({"deterministic_sla": {"verdict": sla["verdict"]}})
        if text_size(essential) > budget:
            return ""
        retained = json.loads(result).get("deterministic_sla") if result else None
        if not isinstance(retained, dict) or retained.get("verdict") != sla["verdict"]:
            return essential
    return result
