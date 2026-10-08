"""Offline metrics for stored LLM reports: verification flags, engineer votes, usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _as_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list(raw: Any) -> List[Any]:
    return list(raw) if isinstance(raw, list) else []


def _ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(part) / float(total)


def _text(raw: Any) -> str:
    return str(raw or "").strip()


def _finding_id(finding: Mapping[str, Any], index: int) -> str:
    raw = finding.get("id") or finding.get("finding_id") or finding.get("key") or f"finding_{index + 1}"
    return _text(raw)


def _verification_status(finding: Mapping[str, Any]) -> str:
    block = finding.get("verification") if isinstance(finding.get("verification"), dict) else {}
    status = _text(block.get("status")).lower()
    if status in {"verified", "unverified", "qualitative"}:
        return status
    return "qualitative"


def _action_is_empty(action: Mapping[str, Any]) -> bool:
    details = _text(action.get("details") or action.get("description") or action.get("implementation_details"))
    linked = action.get("for_finding_ids") or action.get("for_findings") or action.get("finding_ids") or action.get("for_finding_id")
    if isinstance(linked, str):
        ids = [linked.strip()] if linked.strip() else []
    elif isinstance(linked, list):
        ids = [_text(item) for item in linked if _text(item)]
    else:
        ids = []
    return (not details) or (not ids)


def _usage_block(scores: Mapping[str, Any]) -> Dict[str, Any]:
    block = scores.get("run_usage") if isinstance(scores.get("run_usage"), dict) else None
    if block is None and isinstance(scores.get("usage"), dict):
        block = scores.get("usage")
    usage = dict(block or {})
    return {
        "calls": int(usage.get("calls") or 0),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_ms": int(usage.get("elapsed_ms") or 0),
        "cost": usage.get("cost"),
    }


def evaluate_analysis(
    parsed: Mapping[str, Any] | None,
    *,
    feedback: Sequence[Mapping[str, Any]] | None = None,
    scores: Mapping[str, Any] | None = None,
    domain: str = "final",
) -> Dict[str, Any]:
    """Computes offline quality/cost metrics for one stored domain answer."""
    payload = _as_dict(parsed)
    findings = [item for item in _as_list(payload.get("findings")) if isinstance(item, dict)]
    actions = [item for item in _as_list(payload.get("recommended_actions") or payload.get("actions")) if isinstance(item, dict)]
    votes = [item for item in (feedback or []) if isinstance(item, dict)]
    domain_key = _text(domain) or "final"

    status_counts = {"verified": 0, "unverified": 0, "qualitative": 0}
    for finding in findings:
        status_counts[_verification_status(finding)] += 1

    finding_ids = {_finding_id(finding, idx) for idx, finding in enumerate(findings)}
    agree = 0
    disagree = 0
    verdict_vote: Optional[str] = None
    for vote in votes:
        if _text(vote.get("domain")) not in {"", domain_key}:
            continue
        target = _text(vote.get("target"))
        mark = _text(vote.get("vote"))
        if target == "verdict" and mark in {"agree", "disagree"}:
            verdict_vote = mark
        elif target == "finding":
            finding_id = _text(vote.get("finding_id"))
            if finding_id and finding_id not in finding_ids and finding_ids:
                continue
            if mark == "agree":
                agree += 1
            elif mark == "disagree":
                disagree += 1

    empty_actions = sum(1 for action in actions if _action_is_empty(action))
    total_findings = len(findings)
    verdict_match: Optional[float]
    if verdict_vote == "agree":
        verdict_match = 1.0
    elif verdict_vote == "disagree":
        verdict_match = 0.0
    else:
        verdict_match = None

    return {
        "domain": domain_key,
        "verdict": payload.get("verdict"),
        "verdict_match": verdict_match,
        "findings_total": total_findings,
        "findings_verified": status_counts["verified"],
        "findings_unverified": status_counts["unverified"],
        "findings_qualitative": status_counts["qualitative"],
        "findings_verified_share": _ratio(status_counts["verified"], total_findings),
        "findings_unverified_share": _ratio(status_counts["unverified"], total_findings),
        "findings_qualitative_share": _ratio(status_counts["qualitative"], total_findings),
        "findings_agree": agree,
        "findings_disagree": disagree,
        "findings_agree_share": _ratio(agree, total_findings),
        "findings_disagree_share": _ratio(disagree, total_findings),
        "actions_total": len(actions),
        "actions_empty": empty_actions,
        "actions_empty_share": _ratio(empty_actions, len(actions)),
        "usage": _usage_block(_as_dict(scores)),
    }


def evaluate_fixture(path: str | Path) -> Dict[str, Any]:
    """Loads a frozen sample (parsed + context + feedback + scores) and scores it."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Eval fixture {path} must be a JSON object")
    return evaluate_analysis(
        payload.get("parsed"),
        feedback=_as_list(payload.get("feedback")),
        scores=_as_dict(payload.get("scores")),
        domain=_text(payload.get("domain")) or "final",
    )


def load_eval_records(
    conn,
    run_name: str,
    *,
    schema: str = "public",
    llm_table: str = "llm_reports",
    feedback_table: str = "llm_feedback",
) -> List[Dict[str, Any]]:
    """Reads stored parsed/context/scores plus feedback for one run (no live LLM)."""
    name = _text(run_name)
    if not name:
        raise ValueError("run_name обязателен")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT domain, parsed, scores, context, created_at
            FROM {schema}.{llm_table}
            WHERE run_name = %s AND domain <> 'engineer'
            ORDER BY created_at DESC, domain
            """,
            (name,),
        )
        report_rows = cur.fetchall()
        cur.execute(
            f"""
            SELECT run_name, domain, target, finding_id, vote, comment
            FROM {schema}.{feedback_table}
            WHERE run_name = %s
            ORDER BY created_at DESC
            """,
            (name,),
        )
        feedback_rows = cur.fetchall()
    votes: List[Dict[str, Any]] = []
    for row in feedback_rows:
        votes.append(
            {
                "run_name": row[0],
                "domain": row[1],
                "target": row[2],
                "finding_id": row[3] or "",
                "vote": row[4],
                "comment": row[5] or "",
            }
        )
    seen: set[str] = set()
    records: List[Dict[str, Any]] = []
    for row in report_rows:
        domain = _text(row[0])
        if not domain or domain in seen:
            continue
        seen.add(domain)
        records.append(
            {
                "domain": domain,
                "parsed": row[1],
                "scores": row[2],
                "context": row[3],
                "feedback": [vote for vote in votes if vote.get("domain") == domain],
            }
        )
    return records


def evaluate_run_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregates per-domain eval metrics; ``final`` is also copied to the top level."""
    domains: Dict[str, Any] = {}
    usage_calls = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    elapsed_ms = 0
    for record in records:
        domain = _text(record.get("domain")) or "unknown"
        metrics = evaluate_analysis(
            record.get("parsed"),
            feedback=_as_list(record.get("feedback")),
            scores=_as_dict(record.get("scores")),
            domain=domain,
        )
        domains[domain] = metrics
        usage = metrics.get("usage") or {}
        usage_calls += int(usage.get("calls") or 0)
        elapsed_ms += int(usage.get("elapsed_ms") or 0)
        if usage.get("prompt_tokens") is not None:
            prompt_tokens = int(prompt_tokens or 0) + int(usage["prompt_tokens"])
        if usage.get("completion_tokens") is not None:
            completion_tokens = int(completion_tokens or 0) + int(usage["completion_tokens"])
    final_metrics = domains.get("final") or {}
    final_usage = _as_dict(final_metrics.get("usage"))
    return {
        "domains": domains,
        "verdict_match": final_metrics.get("verdict_match"),
        "findings_verified_share": final_metrics.get("findings_verified_share"),
        "findings_unverified_share": final_metrics.get("findings_unverified_share"),
        "findings_qualitative_share": final_metrics.get("findings_qualitative_share"),
        "findings_agree_share": final_metrics.get("findings_agree_share"),
        "findings_disagree_share": final_metrics.get("findings_disagree_share"),
        "actions_empty_share": final_metrics.get("actions_empty_share"),
        "usage": {
            "calls": int(final_usage.get("calls") or usage_calls),
            "prompt_tokens": final_usage["prompt_tokens"] if final_usage.get("prompt_tokens") is not None else prompt_tokens,
            "completion_tokens": final_usage["completion_tokens"] if final_usage.get("completion_tokens") is not None else completion_tokens,
            "elapsed_ms": int(final_usage.get("elapsed_ms") or elapsed_ms),
            "cost": final_usage.get("cost"),
        },
    }
