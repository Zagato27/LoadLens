# SPDX-License-Identifier: Apache-2.0
import contextvars
import os
import json
import re
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, ValidationError, root_validator

from AI.providers import (
    LLMBudgetError,
    LLMEmptyAnswer,
    LLMOutputTruncated,
    ask_llm_structured,
    ask_llm_with_text_data,
    structured_output_enabled,
    usage_domain,
)
from settings import CONFIG
from AI.i18n import with_locale
from AI.prompt_store import PromptMissing, load_prompt, resolve_lang, script_share
from AI.prompt_store import text as prompt_text
from AI.verdicts import NO_DATA, PASS, label as verdict_label, to_code


ALLOWED_LEVELS = {"critical", "high", "medium", "low"}
logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


JUDGE_RUBRIC_DIR = "judge_rubrics"
JUDGE_RUBRIC_FILES = {
    "jvm": os.path.join(JUDGE_RUBRIC_DIR, "jvm.txt"),
    "database": os.path.join(JUDGE_RUBRIC_DIR, "database.txt"),
    "kafka": os.path.join(JUDGE_RUBRIC_DIR, "kafka.txt"),
    "microservices": os.path.join(JUDGE_RUBRIC_DIR, "microservices.txt"),
    "hard_resources": os.path.join(JUDGE_RUBRIC_DIR, "hard_resources.txt"),
    "lt_framework": os.path.join(JUDGE_RUBRIC_DIR, "lt_framework.txt"),
    "lt_generators": os.path.join(JUDGE_RUBRIC_DIR, "lt_generators.txt"),
    "application_logs": os.path.join(JUDGE_RUBRIC_DIR, "application_logs.txt"),
    "final": os.path.join(JUDGE_RUBRIC_DIR, "final.txt"),
}
JUDGE_COMMON_RUBRIC_FILE = os.path.join(JUDGE_RUBRIC_DIR, "common.txt")
JUDGE_WARN_PROMPT_CHARS = max(8000, _read_int_env("LOADLENS_JUDGE_WARN_PROMPT_CHARS", 60000))


def judge_system_prompt(lang: Optional[str] = None) -> str:
    """System prompt of the judge in ``lang`` (the current locale by default)."""
    return prompt_text("judge.system", lang)


# The ru judge system prompt (tools/stubs/llm_stub.py and its tests recognise judge calls by it).
JUDGE_SYSTEM_PROMPT = judge_system_prompt("ru")


def _json_loads_lenient(text: str) -> Optional[Any]:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        # Некоторые LLM оставляют реальные переносы строк внутри JSON-строк.
        return json.loads(text, strict=False)
    except Exception:
        return None


def _extract_json_like(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        parsed = _json_loads_lenient(candidate)
        if isinstance(parsed, dict):
            return parsed
    fence = "```"
    if fence in text:
        parts = text.split(fence)
        for i in range(len(parts) - 1):
            block = parts[i + 1]
            if block.strip().startswith("json"):
                block_text = block.strip()[len("json"):].strip()
            else:
                block_text = block
            parsed = _json_loads_lenient(block_text)
            if isinstance(parsed, dict):
                return parsed
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_level(value: Any, default: str = "low") -> str:
    raw = _clean_text(value).lower()
    if raw in ALLOWED_LEVELS:
        return raw
    if any(token in raw for token in ("crit", "critical", "крит")):
        return "critical"
    if any(token in raw for token in ("high", "выс")):
        return "high"
    if any(token in raw for token in ("medium", "med", "сред")):
        return "medium"
    if any(token in raw for token in ("low", "низ")):
        return "low"
    return default


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result: List[str] = []
    for item in items:
        text = _clean_text(item)
        if text:
            result.append(text)
    return result


def _normalize_identifier(value: Any, default: str = "") -> str:
    raw = _clean_text(value).lower()
    if not raw:
        return default
    normalized = re.sub(r"[^a-z0-9_-]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or default


def _normalize_identifier_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result: List[str] = []
    for item in items:
        identifier = _normalize_identifier(item)
        if identifier and identifier not in result:
            result.append(identifier)
    return result


def _finding_ids_by_component(findings: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        identifier = _normalize_identifier(finding.get("id"))
        component = _clean_text(finding.get("component")).lower()
        if not identifier or not component:
            continue
        bucket = mapping.setdefault(component, [])
        if identifier not in bucket:
            bucket.append(identifier)
    return mapping


def _derive_finding_ids_for_action(
    action: Dict[str, Any],
    component_map: Dict[str, List[str]],
) -> List[str]:
    explicit = _normalize_identifier_list(
        action.get("for_finding_ids")
        or action.get("for_findings")
        or action.get("finding_ids")
        or action.get("related_findings")
        or action.get("for_finding_id")
        or action.get("finding_id")
    )
    if explicit:
        return explicit

    components = _normalize_string_list(
        action.get("affected_components")
        or action.get("components")
        or action.get("component")
    )
    derived: List[str] = []
    for component in components:
        for identifier in component_map.get(_clean_text(component).lower(), []):
            if identifier not in derived:
                derived.append(identifier)
    return derived


def _extract_component_from_evidence(evidence: str) -> str:
    text = _clean_text(evidence)
    if not text:
        return "unknown"
    patterns = [
        r"(?:application|service|job|pod|instance)\s*[=:]\s*([A-Za-z0-9._:/-]+)",
        r"([A-Za-z0-9._-]+)\s*(?:CPU|Heap|latency|GC|RPS)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            component = _clean_text(match.group(1))
            if component:
                return component
    return "unknown"


def _normalize_evidence_items(value: Any) -> List[Dict[str, str]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: List[Dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            metric = _clean_text(item.get("metric") or item.get("name") or item.get("label"))
            observed_value = _clean_text(
                item.get("observed_value") or item.get("value") or item.get("actual")
            )
            threshold = _clean_text(item.get("threshold") or item.get("limit") or item.get("baseline"))
            note = _clean_text(item.get("note") or item.get("details") or item.get("evidence"))
        else:
            metric = ""
            observed_value = ""
            threshold = ""
            note = _clean_text(item)
        if not any((metric, observed_value, threshold, note)):
            continue
        normalized.append(
            {
                "metric": metric,
                "observed_value": observed_value,
                "threshold": threshold,
                "note": note,
            }
        )
    return normalized


def _safe_excerpt(text: str, limit: int = 500) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _normalize_unit_score(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(max(0.0, min(default, 1.0)))
    try:
        if isinstance(value, bool):
            num = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            num = float(value)
        else:
            text = _clean_text(value).replace(",", ".")
            percent = text.endswith("%")
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if not match:
                raise ValueError("no numeric score")
            num = float(match.group(0))
            if percent or (1.0 < num <= 100.0):
                num = num / 100.0
        return float(max(0.0, min(num, 1.0)))
    except Exception:
        return float(max(0.0, min(default, 1.0)))


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    text = _clean_text(value).replace(",", ".")
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "нет", "нет данных", "-", "—"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    if text in {"true", "1", "yes", "y", "да", "истина"}:
        return True
    if text in {"false", "0", "no", "n", "нет", "ложь", ""}:
        return False
    return bool(value)


class FindingEvidenceItem(BaseModel):
    metric: str = Field(default="")
    observed_value: Optional[str] = Field(default=None)
    threshold: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)

    @root_validator(pre=True)
    def _normalize_item(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            return {
                "metric": "",
                "observed_value": None,
                "threshold": None,
                "note": _clean_text(values) or None,
            }
        return {
            "metric": _clean_text(values.get("metric") or values.get("name") or values.get("label")),
            "observed_value": _clean_text(
                values.get("observed_value") or values.get("value") or values.get("actual")
            ) or None,
            "threshold": _clean_text(values.get("threshold") or values.get("limit") or values.get("baseline")) or None,
            "note": _clean_text(values.get("note") or values.get("details") or values.get("evidence")) or None,
        }


VERIFICATION_VERIFIED = "verified"
VERIFICATION_UNVERIFIED = "unverified"
VERIFICATION_QUALITATIVE = "qualitative"


class FindingVerification(BaseModel):
    """Result of checking a finding's numbers against the collected metrics."""

    status: str = Field(default=VERIFICATION_QUALITATIVE)
    claims_total: int = Field(default=0)
    claims_matched: int = Field(default=0)
    unmatched: List[str] = Field(default_factory=list)


class FindingItem(BaseModel):
    id: str = Field(default="")
    summary: str = Field(default="")
    severity: Optional[str] = Field(default=None)
    component: Optional[str] = Field(default=None)
    start_time: Optional[str] = Field(default=None)
    end_time: Optional[str] = Field(default=None)
    peak_time: Optional[str] = Field(default=None)
    evidence_summary: Optional[str] = Field(default=None)
    evidence_items: List[FindingEvidenceItem] = Field(default_factory=list)
    evidence: Optional[str] = Field(default=None)
    verification: Optional[FindingVerification] = Field(default=None)

    @root_validator(pre=True)
    def _normalize_finding(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            return {
                "id": "",
                "summary": _clean_text(values),
                "severity": "low",
                "component": "unknown",
                "start_time": None,
                "end_time": None,
                "peak_time": None,
                "evidence_summary": None,
                "evidence_items": [],
                "evidence": "",
            }
        summary = _clean_text(values.get("summary") or values.get("title") or values.get("text"))
        evidence = _clean_text(values.get("evidence") or values.get("evidence_summary") or values.get("details"))
        evidence_items = _normalize_evidence_items(
            values.get("evidence_items") or values.get("evidence_list") or values.get("evidence_rows")
        )
        component = _clean_text(values.get("component")) or _extract_component_from_evidence(evidence)
        values["id"] = _normalize_identifier(values.get("id") or values.get("finding_id") or values.get("key"))
        values["summary"] = summary
        values["severity"] = _normalize_level(values.get("severity"), default="low")
        values["component"] = component or "unknown"
        values["start_time"] = _clean_text(
            values.get("start_time") or values.get("start") or values.get("window_start") or values.get("from")
        ) or None
        values["end_time"] = _clean_text(
            values.get("end_time") or values.get("end") or values.get("window_end") or values.get("to")
        ) or None
        values["peak_time"] = _clean_text(
            values.get("peak_time") or values.get("time_of_peak") or values.get("peak")
        ) or None
        values["evidence_summary"] = evidence or None
        values["evidence_items"] = evidence_items
        values["evidence"] = evidence
        return values


class RecommendedActionItem(BaseModel):
    summary: str = Field(default="")
    details: Optional[str] = Field(default=None)
    priority: Optional[str] = Field(default=None)
    affected_components: List[str] = Field(default_factory=list)
    for_finding_ids: List[str] = Field(default_factory=list)

    @root_validator(pre=True)
    def _normalize_action(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            return {
                "summary": _clean_text(values),
                "details": None,
                "priority": "medium",
                "affected_components": [],
                "for_finding_ids": [],
            }
        summary = _clean_text(values.get("summary") or values.get("action") or values.get("text"))
        details = _clean_text(
            values.get("details")
            or values.get("description")
            or values.get("implementation_details")
            or values.get("how")
            or values.get("rationale")
        )
        steps = _normalize_string_list(values.get("steps") or values.get("implementation_steps"))
        verification = _clean_text(
            values.get("verification") or values.get("validation") or values.get("how_to_verify")
        )
        details_parts: List[str] = []
        if details:
            details_parts.append(details)
        if steps:
            details_parts.append(prompt_text("action.steps") + "\n- " + "\n- ".join(steps))
        if verification:
            details_parts.append(prompt_text("action.verification") + "\n- " + verification)
        values["summary"] = summary
        values["details"] = "\n\n".join(part for part in details_parts if part).strip() or None
        values["priority"] = _normalize_level(values.get("priority"), default="medium")
        values["affected_components"] = _normalize_string_list(
            values.get("affected_components") or values.get("components") or values.get("component")
        )
        values["for_finding_ids"] = _normalize_identifier_list(
            values.get("for_finding_ids")
            or values.get("for_findings")
            or values.get("finding_ids")
            or values.get("related_findings")
            or values.get("for_finding_id")
            or values.get("finding_id")
        )
        return values


class PeakPerformance(BaseModel):
    max_rps: Optional[float] = Field(default=None)
    max_time: Optional[str] = Field(default=None)
    drop_time: Optional[str] = Field(default=None)
    method: Optional[str] = Field(default=None)
    not_applicable: Optional[bool] = Field(default=None)
    reason: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)

    @root_validator(pre=True)
    def _normalize_peak(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            return {}
        values["not_applicable"] = _coerce_bool(values.get("not_applicable") or values.get("notApplicable"))
        values["reason"] = _clean_text(values.get("reason")) or None
        values["note"] = _clean_text(values.get("note")) or None
        values["max_rps"] = _coerce_optional_float(
            values.get("max_rps")
            or values.get("rps")
            or values.get("stable_max")
            or values.get("value")
        )
        values["max_time"] = _clean_text(values.get("max_time") or values.get("time") or values.get("stable_max_time")) or None
        values["drop_time"] = _clean_text(values.get("drop_time") or values.get("degradation_time")) or None
        values["method"] = _clean_text(values.get("method")) or None
        return values


def _derive_verdict_rationale(verdict: str, findings: List[Dict[str, object]], lang: Optional[str] = None) -> Optional[str]:
    """Fallback rationale of a model answer without one, in the language of the report (current locale)."""
    lang = resolve_lang(lang)
    code = to_code(verdict)
    verdict_text = verdict_label(code, lang)
    summaries: List[str] = []
    for item in findings or []:
        if not isinstance(item, dict):
            continue
        verification = item.get("verification") if isinstance(item.get("verification"), dict) else {}
        if str(verification.get("status") or "").lower() == VERIFICATION_UNVERIFIED:
            continue
        summary = _clean_text(item.get("summary") or item.get("title") or item.get("text"))
        if summary:
            summaries.append(summary)
        if len(summaries) >= 3:
            break
    if summaries:
        bullets = "\n".join(f"- {summary}" for summary in summaries)
        return prompt_text("rationale.derived_findings", lang, label=verdict_text, bullets=bullets)
    if code == PASS:
        return prompt_text("rationale.derived_pass", lang, label=verdict_text)
    if code == NO_DATA:
        return prompt_text("rationale.derived_no_data", lang, label=verdict_text)
    return prompt_text("rationale.derived_other", lang, label=verdict_text)


def rationale_from_analysis(parsed: "LLMAnalysis", lang: Optional[str] = None) -> str:
    """Builds a short verdict rationale from confirmed findings when the model omitted one.

    The head is ``rationale.head_<code>`` of AI/prompts/<lang>/_strings.json (``{label}`` is the verdict label).
    """
    lang = resolve_lang(lang)
    code = to_code(getattr(parsed, "verdict", None))
    head = prompt_text(f"rationale.head_{code}", lang, label=verdict_label(code, lang))
    bullets: List[str] = []
    for finding in getattr(parsed, "findings", None) or []:
        verification = getattr(finding, "verification", None)
        status = _clean_text(getattr(verification, "status", None)).lower() if verification is not None else ""
        if status == VERIFICATION_UNVERIFIED:
            continue
        summary = _clean_text(getattr(finding, "summary", None))
        if summary:
            bullets.append(f"- {summary}")
        if len(bullets) >= 4:
            break
    if not bullets:
        return head
    return f"{head}\n" + "\n".join(bullets)


def fill_missing_verdict_rationale(parsed: "LLMAnalysis") -> "LLMAnalysis":
    """Keeps a model-written rationale; otherwise fills one so the status is never unexplained."""
    if _clean_text(getattr(parsed, "verdict_rationale", None)):
        return parsed
    return parsed.copy(update={"verdict_rationale": rationale_from_analysis(parsed)})


class VerificationSummary(BaseModel):
    """Counts of findings per verification status for one LLM answer."""

    total: int = Field(default=0)
    verified: int = Field(default=0)
    unverified: int = Field(default=0)
    qualitative: int = Field(default=0)
    revised_by_model: bool = Field(default=False)


class LLMAnalysisError(BaseModel):
    """Why a domain has no model analysis. Shown instead of a fake verdict."""

    reason: str = Field(default="invalid_json")
    message: str = Field(default="")
    provider: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    output_tokens: Optional[int] = Field(default=None)
    excerpt: Optional[str] = Field(default=None)


class LLMAnalysis(BaseModel):
    # A verdict code (AI.verdicts); the model may answer with a label or a synonym, the validator maps it.
    # The description reaches the function-calling schema of GigaChat structured output.
    verdict: str = Field(default=NO_DATA, description="pass | risk | fail | no_data")
    verdict_rationale: Optional[str] = Field(default=None)
    analysis_error: Optional[LLMAnalysisError] = Field(default=None)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    findings: List[FindingItem] = Field(default_factory=list)
    recommended_actions: List[RecommendedActionItem] = Field(default_factory=list)
    affected_components: Optional[List[str]] = Field(default=None)
    peak_performance: Optional[PeakPerformance] = Field(default=None)
    test_profile: Optional[Dict[str, Any]] = Field(default=None)
    deterministic_sla: Optional[Dict[str, Any]] = Field(default=None)
    stability_under_load: Optional[Dict[str, Any]] = Field(default=None)
    verification_summary: Optional[VerificationSummary] = Field(default=None)

    @root_validator(pre=True)
    def _normalize_fields(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            values = {}
        raw_error = values.get("analysis_error")
        values["analysis_error"] = raw_error if isinstance(raw_error, dict) else None
        values["verdict"] = to_code(_clean_text(values.get("verdict")))
        verdict_rationale = _clean_text(
            values.get("verdict_rationale")
            or values.get("verdict_reason")
            or values.get("rationale")
            or values.get("justification")
        )
        values["verdict_rationale"] = verdict_rationale or None
        conf = values.get("confidence")
        try:
            if conf is not None:
                conf_val = float(conf)
                if conf_val > 1.0 and conf_val <= 100.0:
                    conf_val = conf_val / 100.0
                values["confidence"] = max(0.0, min(conf_val, 1.0))
        except Exception:
            values["confidence"] = None

        findings = values.get("findings")
        if findings is None:
            findings = []
        elif isinstance(findings, (str, dict)):
            findings = [findings]
        elif not isinstance(findings, list):
            findings = []
        norm_findings: List[Dict[str, object]] = []
        for idx, item in enumerate(findings):
            prepared = dict(item) if isinstance(item, dict) else {"summary": _clean_text(item)}
            summary = _clean_text(prepared.get("summary") or prepared.get("title") or prepared.get("text"))
            if not summary:
                continue
            prepared["id"] = _normalize_identifier(
                prepared.get("id") or prepared.get("finding_id") or prepared.get("key"),
                default=f"finding_{len(norm_findings) + 1}",
            )
            norm_findings.append(prepared)
        values["findings"] = norm_findings
        if not values.get("verdict_rationale"):
            values["verdict_rationale"] = _derive_verdict_rationale(values.get("verdict"), norm_findings)

        actions = values.get("recommended_actions") or values.get("actions") or []
        if isinstance(actions, (str, dict)):
            actions = [actions]
        elif not isinstance(actions, list):
            actions = []
        component_map = _finding_ids_by_component(norm_findings)
        norm_actions: List[Dict[str, object]] = []
        for item in actions:
            prepared = dict(item) if isinstance(item, dict) else {"summary": _clean_text(item)}
            summary = _clean_text(prepared.get("summary") or prepared.get("action") or prepared.get("text"))
            if not summary:
                continue
            prepared["summary"] = summary
            prepared["priority"] = _normalize_level(prepared.get("priority"), default="medium")
            prepared["affected_components"] = _normalize_string_list(
                prepared.get("affected_components") or prepared.get("components") or prepared.get("component")
            )
            prepared["for_finding_ids"] = _derive_finding_ids_for_action(prepared, component_map)
            norm_actions.append(prepared)
        values["recommended_actions"] = norm_actions

        affected_components = _normalize_string_list(values.get("affected_components"))
        if not affected_components:
            derived: List[str] = []
            for action in norm_actions:
                derived.extend(_normalize_string_list(action.get("affected_components")))
            for finding in norm_findings:
                component = _clean_text(finding.get("component"))
                if component:
                    derived.append(component)
            if derived:
                affected_components = list(dict.fromkeys(derived))
        values["affected_components"] = affected_components or None
        return values


def _finding_evidence_strings(finding: Any) -> List[str]:
    parts: List[str] = []
    if isinstance(finding, FindingItem):
        evidence_items = getattr(finding, "evidence_items", []) or []
        summary = getattr(finding, "evidence_summary", "")
        legacy = getattr(finding, "evidence", "")
        start_time = getattr(finding, "start_time", "")
        end_time = getattr(finding, "end_time", "")
        peak_time = getattr(finding, "peak_time", "")
        if summary:
            parts.append(str(summary))
        elif legacy:
            parts.append(str(legacy))
        time_bits = [str(x).strip() for x in (start_time, end_time, peak_time) if str(x).strip()]
        if time_bits:
            parts.append(" ".join(time_bits))
        for item in evidence_items:
            if isinstance(item, FindingEvidenceItem):
                for attr in ("metric", "observed_value", "threshold", "note"):
                    val = getattr(item, attr, "")
                    if isinstance(val, str) and val.strip():
                        parts.append(val)
            elif isinstance(item, dict):
                for key in ("metric", "observed_value", "threshold", "note"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        parts.append(val)
    elif isinstance(finding, dict):
        summary = finding.get("evidence_summary") or finding.get("evidence")
        if isinstance(summary, str) and summary.strip():
            parts.append(summary)
        time_bits = [
            str(finding.get(key) or "").strip()
            for key in ("start_time", "end_time", "peak_time")
            if str(finding.get(key) or "").strip()
        ]
        if time_bits:
            parts.append(" ".join(time_bits))
        for item in _normalize_evidence_items(
            finding.get("evidence_items") or finding.get("evidence_list") or finding.get("evidence_rows")
        ):
            for key in ("metric", "observed_value", "threshold", "note"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val)
    return [_clean_text(part) for part in parts if _clean_text(part)]


def _failure_payload(error: Any, raw_text: str) -> Dict[str, Any]:
    excerpt = _safe_excerpt(raw_text) or None
    if isinstance(error, LLMOutputTruncated):
        reason = "output_truncated"
    elif isinstance(error, LLMEmptyAnswer):
        reason = "empty_answer"
    elif isinstance(error, LLMBudgetError):
        reason = "budget"
    elif isinstance(error, BaseException):
        reason = "provider_error"
    elif isinstance(error, dict):
        return {
            "reason": str(error.get("reason") or "invalid_json"),
            "message": str(error.get("message") or prompt_text("analysis_error.invalid_json")),
            "provider": error.get("provider"),
            "max_tokens": error.get("max_tokens"),
            "output_tokens": error.get("output_tokens"),
            "excerpt": error.get("excerpt") or excerpt,
        }
    else:
        reason = "invalid_json"
    message = str(error).strip() if error is not None else ""
    if not message:
        message = prompt_text("analysis_error.invalid_json")
    return {
        "reason": reason,
        "message": message,
        "provider": getattr(error, "provider", None),
        "max_tokens": getattr(error, "max_tokens", None) or None,
        "output_tokens": getattr(error, "output_tokens", None),
        "excerpt": excerpt,
    }


def make_failed_llm_analysis(error: Any, raw_text: str = "") -> LLMAnalysis:
    """Analysis placeholder that reports the failure instead of inventing findings."""
    payload = _failure_payload(error, raw_text)
    return LLMAnalysis.parse_obj(
        {
            "verdict": NO_DATA,
            "verdict_rationale": payload["message"],
            "confidence": 0.0,
            "findings": [],
            "recommended_actions": [],
            "analysis_error": payload,
        }
    )


def parse_llm_analysis_strict(raw_text: str) -> Optional[LLMAnalysis]:
    """Парсит ответ LLM в строгий объект `LLMAnalysis`.

    Параметры:
        raw_text (str): Текст модели (может содержать пояснения/кодовые блоки).

    Возвращает:
        LLMAnalysis | None: Структурированный объект или None при ошибке.

    Исключения:
        Не выбрасывает; ошибки валидации подавляются.
    """
    if not raw_text:
        return None
    maybe_json = _json_loads_lenient(raw_text)
    if not isinstance(maybe_json, dict):
        maybe_json = _extract_json_like(raw_text)
    if maybe_json is None:
        return None
    try:
        return LLMAnalysis.parse_obj(maybe_json)
    except ValidationError as exc:
        try:
            logger.warning("LLM strict validation failed: %s", exc.errors())
        except Exception:
            pass
        return None


class JudgeRubricScores(BaseModel):
    evidence_grounding: float = Field(default=0.0, ge=0.0, le=1.0)
    issue_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    specificity: float = Field(default=0.0, ge=0.0, le=1.0)
    sla_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    actionability: float = Field(default=0.0, ge=0.0, le=1.0)

    @root_validator(pre=True)
    def _normalize_rubric(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            values = {}
        alias_map = {
            "evidence_grounding": ("evidence", "grounding", "groundedness"),
            "issue_coverage": ("coverage", "issues", "completeness"),
            "specificity": ("details",),
            "sla_alignment": ("sla", "verdict_alignment", "risk_alignment"),
            "actionability": ("actions", "recommendations"),
        }
        normalized: Dict[str, float] = {}
        for key, aliases in alias_map.items():
            raw = values.get(key)
            if raw is None:
                for alias in aliases:
                    if alias in values:
                        raw = values.get(alias)
                        break
            normalized[key] = _normalize_unit_score(raw, default=0.0)
        return normalized


def _compute_judge_overall(factual: float, completeness: float, specificity: float) -> float:
    overall = 0.5 * float(factual) + 0.3 * float(completeness) + 0.2 * float(specificity)
    return float(max(0.0, min(overall, 1.0)))


def _compute_judge_axes_from_rubric(rubric: JudgeRubricScores) -> Dict[str, float]:
    factual = 0.65 * rubric.evidence_grounding + 0.35 * rubric.sla_alignment
    completeness = 0.7 * rubric.issue_coverage + 0.3 * rubric.actionability
    specificity = rubric.specificity
    return {
        "factual": float(max(0.0, min(factual, 1.0))),
        "completeness": float(max(0.0, min(completeness, 1.0))),
        "specificity": float(max(0.0, min(specificity, 1.0))),
        "overall": _compute_judge_overall(factual, completeness, specificity),
    }


class JudgeScoreItem(BaseModel):
    index: int
    factual: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    specificity: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    rubric: JudgeRubricScores = Field(default_factory=JudgeRubricScores)

    @root_validator(pre=True)
    def _normalize_score(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            values = {}
        rubric_raw = values.get("rubric")
        if not isinstance(rubric_raw, dict):
            rubric_raw = {
                "evidence_grounding": values.get("evidence_grounding", values.get("factual")),
                "issue_coverage": values.get("issue_coverage", values.get("completeness")),
                "specificity": values.get("rubric_specificity", values.get("specificity")),
                "sla_alignment": values.get("sla_alignment", values.get("factual")),
                "actionability": values.get("actionability", values.get("completeness")),
            }
        rubric = JudgeRubricScores.parse_obj(rubric_raw)
        derived = _compute_judge_axes_from_rubric(rubric)
        factual = _normalize_unit_score(values.get("factual"), default=derived["factual"])
        completeness = _normalize_unit_score(values.get("completeness"), default=derived["completeness"])
        specificity = _normalize_unit_score(values.get("specificity"), default=derived["specificity"])
        overall = _normalize_unit_score(
            values.get("overall"),
            default=_compute_judge_overall(factual, completeness, specificity),
        )
        values["rubric"] = rubric.dict()
        values["factual"] = factual
        values["completeness"] = completeness
        values["specificity"] = specificity
        values["overall"] = overall
        return values


def _normalize_judge_domain_key(domain_key: Optional[str]) -> str:
    raw = _clean_text(domain_key).lower().replace("-", "_")
    if raw in {"overall", "summary", "global"}:
        return "final"
    return raw


def _load_optional_prompt(filename: str, lang: Optional[str] = None) -> Optional[str]:
    """A prompt file of the language (ru when it has none), or None when the file exists nowhere."""
    try:
        text = load_prompt(filename, lang)
    except PromptMissing:
        return None
    return text if text else None


def _load_judge_rubric(domain_key: Optional[str], lang: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    normalized = _normalize_judge_domain_key(domain_key)
    meta: Dict[str, Any] = {
        "domain_key": normalized or None,
        "rubric_domain": normalized or None,
        "used_rubric": False,
        "used_safe_path": False,
        "fallback_reason": None,
    }
    common_text = _load_optional_prompt(JUDGE_COMMON_RUBRIC_FILE, lang)
    domain_file = JUDGE_RUBRIC_FILES.get(normalized or "")
    domain_text = _load_optional_prompt(domain_file, lang) if domain_file else None
    if common_text and domain_text:
        meta["used_rubric"] = True
        return common_text.rstrip() + "\n\n" + domain_text.strip(), meta
    if normalized and normalized not in JUDGE_RUBRIC_FILES:
        meta["fallback_reason"] = f"unknown_domain:{normalized}"
    elif not common_text:
        meta["fallback_reason"] = "missing_common_rubric"
    elif not domain_text:
        meta["fallback_reason"] = f"missing_domain_rubric:{normalized or 'none'}"
    meta["used_safe_path"] = True
    return None, meta


def _compact_jsonish_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = _extract_json_like(raw)
    if parsed is None:
        return raw
    try:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return raw


def _compact_judge_data_context(data_context: str, lang: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    raw = str(data_context or "")
    meta: Dict[str, Any] = {
        "context_chars_raw": len(raw),
        "context_chars": len(raw),
        "context_truncated": False,
        "dropped_context_keys": [],
    }
    if not raw:
        return prompt_text("judge.no_data", lang), meta
    return raw, meta


def _build_judge_prompt(
    candidates_texts: List[str],
    data_context: str,
    domain_key: Optional[str],
    lang: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    lang = resolve_lang(lang)
    template = load_prompt("judge_prompt.txt", lang)
    rubric_text, rubric_meta = _load_judge_rubric(domain_key, lang)
    if "{{JUDGE_RUBRIC}}" in template:
        if rubric_text:
            template = template.replace("{{JUDGE_RUBRIC}}", rubric_text)
        else:
            # No rubric for the domain: the plain factual/completeness/specificity form.
            template = load_prompt("judge_safe_prompt.txt", lang).rstrip("\n")
    normalized_domain = _normalize_judge_domain_key(domain_key) or "final"
    compact_context, context_meta = _compact_judge_data_context(data_context, lang)
    candidates_payload = []
    for idx, text in enumerate(candidates_texts):
        candidates_payload.append({"index": idx, "text": _compact_jsonish_text(text)})
    prompt = template.replace("{{DOMAIN_KEY}}", normalized_domain)
    prompt = prompt.replace("{{CANDIDATES_JSON}}", json.dumps(candidates_payload, ensure_ascii=False))
    prompt = prompt.replace("{{DATA_CONTEXT}}", compact_context or prompt_text("judge.no_data", lang))
    meta = {
        **rubric_meta,
        **context_meta,
        "domain_key": normalized_domain,
        "candidate_count": len(candidates_payload),
        "prompt_chars": len(prompt),
    }
    return prompt, meta


# Keys under which models put the list of per-candidate scores besides the requested "scores".
_JUDGE_LIST_KEYS = ("scores", "candidates", "results", "evaluations", "items")


def _judge_payload(raw_text: str) -> Any:
    """JSON value of a judge answer: the whole text, a fenced block, or an object or list inside prose."""
    text = str(raw_text or "")
    try:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    except Exception:
        pass
    parsed = _json_loads_lenient(text.strip())
    if parsed is not None:
        return parsed
    for block in text.split("```")[1::2]:
        body = block.strip()
        if body.lower().startswith("json"):
            body = body[4:].strip()
        parsed = _json_loads_lenient(body)
        if parsed is not None:
            return parsed
    obj = _extract_json_like(text)
    if obj is not None:
        return obj
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        return _json_loads_lenient(text[start:end + 1])
    return None


def _judge_items(parsed: Any) -> List[Any]:
    """Per-candidate entries of a judge answer in the shapes models actually return."""
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    for key in _JUDGE_LIST_KEYS:
        value = parsed.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            parsed = value  # {"scores": {"0": {...}, "1": {...}}}
            break
    if "index" in parsed:
        return [parsed]
    return [
        {**value, "index": value.get("index", int(str(key).strip()))}
        for key, value in parsed.items()
        if str(key).strip().isdigit() and isinstance(value, dict)
    ]


def _parse_judge_scores(raw_text: str) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for position, item in enumerate(_judge_items(_judge_payload(raw_text))):
        if not isinstance(item, dict):
            continue
        try:
            # Entries without an index are taken in the order of the candidates.
            score_item = JudgeScoreItem.parse_obj({**item, "index": item.get("index", position)})
        except ValidationError:
            continue
        result[int(score_item.index)] = score_item.dict()
    return result


def _align_judge_indexes(scores: Dict[Any, Dict[str, Any]], count: int) -> Dict[int, Dict[str, Any]]:
    """Scores keyed 0..count-1; a judge that numbered the candidates from 1 is shifted back."""
    aligned: Dict[int, Dict[str, Any]] = {}
    for key, value in (scores or {}).items():
        try:
            aligned[int(key)] = value
        except (TypeError, ValueError):
            continue
    if count and 0 not in aligned and set(aligned) == set(range(1, count + 1)):
        aligned = {key - 1: value for key, value in aligned.items()}
    return aligned


def _build_critic_prompt(candidate_text: str, lang: Optional[str] = None) -> str:
    template = load_prompt("critic_prompt.txt", lang)
    return template.replace("{{CANDIDATE}}", candidate_text)


def judge_candidates_with_llm(
    candidates_texts: List[str],
    data_context: str,
    domain_key: Optional[str] = None,
    lang: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Запрашивает у LLM-судьи оценки нескольких кандидатских ответов.

    Параметры:
        candidates_texts (list[str]): JSON-тексты кандидатов.
        data_context (str): Контекст метрик в JSON.
        domain_key (str | None): Ключ домена для выбора judge-рубрики.
        lang (str | None): Язык промпта и рубрик (по умолчанию текущая локаль — язык отчёта).

    Возвращает:
        tuple: `(scores_by_index, judge_meta)`.
    """
    lang = resolve_lang(lang)
    if not candidates_texts:
        return {}, {"domain_key": _normalize_judge_domain_key(domain_key) or None, "empty_candidates": True}
    judge_prompt, meta = _build_judge_prompt(candidates_texts, data_context, domain_key, lang)
    if meta.get("used_safe_path"):
        logger.warning(
            "Judge rubric fallback activated for domain '%s': %s",
            meta.get("domain_key"),
            meta.get("fallback_reason"),
        )
    if meta.get("context_truncated"):
        logger.warning(
            "Judge prompt compacted for domain '%s': context_truncated=%s",
            meta.get("domain_key"),
            meta.get("context_truncated"),
        )
    if int(meta.get("prompt_chars", 0) or 0) > JUDGE_WARN_PROMPT_CHARS:
        logger.warning(
            "Judge prompt for domain '%s' is large: %s chars",
            meta.get("domain_key"),
            meta.get("prompt_chars"),
        )
    started_at = time.perf_counter()
    raw = ask_llm_with_text_data(
        user_prompt=judge_prompt,
        data_context="",
        llm_config={"force_json": True},
        system_prompt=judge_system_prompt(lang),
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    result = _parse_judge_scores(raw)
    meta["elapsed_ms"] = elapsed_ms
    meta["scores_returned"] = len(result)
    if not result:
        logger.warning(
            "Judge returned no valid scores for domain '%s' in %sms. Raw excerpt: %s",
            meta.get("domain_key"),
            elapsed_ms,
            _safe_excerpt(raw, limit=300),
        )
    else:
        logger.info(
            "Judge domain='%s': scores=%s, prompt_chars=%s, context_chars=%s, elapsed_ms=%s, rubric=%s, safe_path=%s",
            meta.get("domain_key"),
            len(result),
            meta.get("prompt_chars"),
            meta.get("context_chars"),
            elapsed_ms,
            meta.get("used_rubric"),
            meta.get("used_safe_path"),
        )
    return result, meta


def _extract_sections_from_context(ctx_obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(ctx_obj, dict):
        return []
    sections: List[Dict[str, Any]] = []
    if isinstance(ctx_obj.get("sections"), list):
        sections.extend(ctx_obj["sections"])
    domains = ctx_obj.get("domains")
    if isinstance(domains, dict):
        for val in domains.values():
            if isinstance(val, dict) and isinstance(val.get("sections"), list):
                sections.extend(val["sections"])
    return sections


def _extract_lt_framework_sections(ctx_obj: Any) -> List[Dict[str, Any]]:
    """Возвращает секции только домена lt_framework из контекста."""
    if not isinstance(ctx_obj, dict):
        return []
    domains = ctx_obj.get("domains")
    if isinstance(domains, dict):
        lt_domain = domains.get("lt_framework")
        if isinstance(lt_domain, dict) and isinstance(lt_domain.get("sections"), list):
            return [sec for sec in lt_domain["sections"] if isinstance(sec, dict)]
        return []
    if str(ctx_obj.get("domain") or "").strip().lower() == "lt_framework":
        sections = ctx_obj.get("sections")
        if isinstance(sections, list):
            return [sec for sec in sections if isinstance(sec, dict)]
    return []


def _collect_label_vocab(sections: List[Dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for section in sections:
        label = section.get("label")
        if label:
            labels.add(str(label).lower())
        for series in section.get("top_series", []) or []:
            series_name = series.get("series")
            if series_name:
                labels.add(str(series_name).lower())
    return labels


def _extract_peak_estimate(sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for section in sections:
        for series in section.get("top_series", []) or []:
            max_val = series.get("max")
            if max_val is None:
                continue
            try:
                max_float = float(max_val)
            except Exception:
                continue
            if best is None or max_float > best.get("max", float("-inf")):
                best = {
                    "max": max_float,
                    "max_time": series.get("max_time"),
                    "series": series.get("series")
                }
    return best


def _finding_matches_labels(finding: Any, labels: set[str]) -> bool:
    if not labels:
        return False
    text_parts: List[str] = []
    try:
        if isinstance(finding, FindingItem):
            for attr in ("summary", "component", "start_time", "end_time", "peak_time", "evidence_summary"):
                text_parts.append(getattr(finding, attr, ""))
            text_parts.extend(_finding_evidence_strings(finding))
        elif isinstance(finding, dict):
            text_parts.extend([
                str(finding.get("summary", "")),
                str(finding.get("component", "")),
                str(finding.get("evidence", "")),
                str(finding.get("evidence_summary", "")),
                str(finding.get("start_time", "")),
                str(finding.get("end_time", "")),
                str(finding.get("peak_time", "")),
            ])
            text_parts.extend(_finding_evidence_strings(finding))
        else:
            text_parts.append(str(finding))
    except Exception:
        text_parts.append(str(finding))

    blob = " ".join([part for part in text_parts if isinstance(part, str)])
    blob_lower = blob.lower()
    return any(label in blob_lower for label in labels if label)


def _finding_text_blob(finding: Any) -> str:
    text_parts: List[str] = []
    try:
        if isinstance(finding, FindingItem):
            for attr in ("summary", "component", "start_time", "end_time", "peak_time", "evidence_summary"):
                text_parts.append(getattr(finding, attr, ""))
            text_parts.extend(_finding_evidence_strings(finding))
        elif isinstance(finding, dict):
            text_parts.extend([
                str(finding.get("summary", "")),
                str(finding.get("component", "")),
                str(finding.get("evidence", "")),
                str(finding.get("evidence_summary", "")),
                str(finding.get("start_time", "")),
                str(finding.get("end_time", "")),
                str(finding.get("peak_time", "")),
            ])
            text_parts.extend(_finding_evidence_strings(finding))
        else:
            text_parts.append(str(finding))
    except Exception:
        text_parts.append(str(finding))
    return _clean_text(" ".join([part for part in text_parts if isinstance(part, str)]))


def _safe_float_value(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _reference_match_keys(section_label: str, series_name: str) -> List[str]:
    short_metric_tokens = {
        "cpu", "mem", "ram", "heap", "gc", "rps", "qps",
        "p95", "p99", "p90", "latency", "error", "errors",
        "lag", "disk", "iops", "tps", "db", "jvm",
    }
    keys: set[str] = set()
    for raw in (section_label, series_name):
        text = _clean_text(raw).lower()
        if not text:
            continue
        keys.add(text)
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9._/-]+", text):
            if (
                token in short_metric_tokens
                or len(token) >= 6
                or any(ch.isdigit() or ch in "._/-" for ch in token)
            ):
                keys.add(token)
    return sorted(keys, key=len, reverse=True)


def _build_numeric_reference_catalog(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for section in sections:
        section_label = _clean_text(section.get("label")).lower()
        for series in section.get("top_series", []) or []:
            if not isinstance(series, dict):
                continue
            series_name = _clean_text(series.get("series")).lower()
            match_keys = _reference_match_keys(section_label, series_name)
            for field in ("mean", "min", "max", "last", "stable_max"):
                num = _safe_float_value(series.get(field))
                if num is None:
                    continue
                refs.append(
                    {
                        "section": section_label,
                        "series": series_name,
                        "field": field,
                        "value": num,
                        "match_keys": match_keys,
                    }
                )
    return refs


def _context_packs(ctx_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Domain pack(s) inside a context: the object itself and every entry of ``domains``."""
    packs: List[Dict[str, Any]] = [ctx_obj]
    domains = ctx_obj.get("domains")
    if isinstance(domains, dict):
        packs.extend(pack for pack in domains.values() if isinstance(pack, dict))
    return packs


# Words a model writes next to a number it quotes, Russian and English stems in one table (no branching by
# the language of the answer): a claim is checked against the references whose stems it mentions.
_REF_STEMS: Dict[str, Tuple[str, ...]] = {
    "load_step": ("rps", "ступен", "интервал", "нагрузк", "step", "interval", "load"),
    "baseline": ("baseline", "предыдущ", "прошл", "previous", "prior"),
    "peak": ("rps", "max_rps", "stable_max", "производительност", "нагрузк", "throughput", "performance", "load"),
    "sla": ("sla", "порог", "threshold", "целев", "target"),
    "sla_step": ("rps", "ступен", "интервал", "нагрузк", "sla", "step", "interval", "load"),
    "hold": ("rps", "удерж", "окно", "интервал", "hold", "нагрузк", "window", "interval", "load"),
    "spike": ("всплеск", "spike", "фон", "восстанов", "overshoot", "p95", "ошиб", "rps", "background", "recover", "error"),
    "saturation": ("насыщ", "saturation", "ступен", "rps", "ресурс", "cpu", "памят", "memory", "saturat", "step", "resource"),
    "trend": ("тренд", "дрейф", "рост", "trend", "drift", "growth"),
}


def _extra_numeric_references(ctx_obj: Any) -> List[Dict[str, Any]]:
    """Numbers the model may cite besides ``top_series`` statistics.

    Covers the step table, step anomalies, load steps, baseline deltas, the
    cross-domain timeline, the designated peak and deterministic SLA checks.
    """
    refs: List[Dict[str, Any]] = []
    if not isinstance(ctx_obj, dict):
        return refs

    def _add(section: Any, series: Any, field: str, value: Any, extra_keys: Tuple[str, ...] = ()) -> None:
        num = _safe_float_value(value)
        if num is None:
            return
        section_label = _clean_text(section).lower()
        series_name = _clean_text(series).lower()
        refs.append({
            "section": section_label,
            "series": series_name,
            "field": field,
            "value": num,
            "match_keys": _reference_match_keys(section_label, series_name) + [k for k in extra_keys if k],
        })

    for pack in _context_packs(ctx_obj):
        for section in pack.get("step_table") or []:
            if not isinstance(section, dict):
                continue
            for row in section.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                _add(section.get("label"), row.get("series"), "change_vs_first_pct", row.get("change_vs_first_pct"))
                for stat in row.get("per_step") or []:
                    if not isinstance(stat, dict):
                        continue
                    for field in ("mean", "p95", "max"):
                        _add(section.get("label"), row.get("series"), f"step{stat.get('step')}_{field}", stat.get(field))
                    tail = stat.get("after_plateau")
                    if isinstance(tail, dict):
                        for field in ("mean", "min", "max"):
                            _add(section.get("label"), row.get("series"), f"step{stat.get('step')}_after_{field}", tail.get(field))
        for section in pack.get("sections") or []:
            if not isinstance(section, dict):
                continue
            for anomaly in section.get("anomalies") or []:
                if not isinstance(anomaly, dict):
                    continue
                for item in anomaly.get("step_anomalies") or []:
                    if isinstance(item, dict):
                        for field in ("value", "reference", "change_pct", "load_change_pct"):
                            _add(section.get("label"), anomaly.get("series"), f"anomaly_{field}", item.get(field))
                for window in anomaly.get("windows") or []:
                    if isinstance(window, dict):
                        for field in ("peak", "mean", "threshold_high"):
                            _add(section.get("label"), anomaly.get("series"), f"window_{field}", window.get(field))
        for step in pack.get("load_steps") or []:
            if isinstance(step, dict):
                _add("load_steps", step.get("label"), "rps_level", step.get("rps_level"), _REF_STEMS["load_step"])
        baseline = pack.get("baseline")
        if isinstance(baseline, dict):
            for label, rows in (baseline.get("sections") or {}).items():
                for series, row in (rows or {}).items():
                    if isinstance(row, dict):
                        for field in ("baseline_mean", "baseline_p95", "baseline_max", "delta_mean_pct", "delta_max_pct"):
                            _add(label, series, field, row.get(field), _REF_STEMS["baseline"])
            for delta in baseline.get("key_deltas") or []:
                if isinstance(delta, dict):
                    for field in ("baseline_mean", "baseline_p95", "baseline_max", "delta_mean_pct", "delta_max_pct"):
                        _add(delta.get("label"), delta.get("series"), field, delta.get(field), _REF_STEMS["baseline"])

    timeline = ctx_obj.get("timeline")
    if isinstance(timeline, dict):
        for row in timeline.get("rows") or []:
            if not isinstance(row, dict):
                continue
            for index, value in enumerate(row.get("mean_per_step") or []):
                _add(row.get("label"), row.get("series"), f"timeline_step{index + 1}", value)
    peak = ctx_obj.get("designated_peak_performance")
    if isinstance(peak, dict):
        _add("designated_peak_performance", peak.get("series") or peak.get("source_label"), "stable_max", peak.get("stable_max"), _REF_STEMS["peak"])
    sla = ctx_obj.get("deterministic_sla")
    if isinstance(sla, dict):
        for check in sla.get("checks") or []:
            if not isinstance(check, dict):
                continue
            name = _clean_text(check.get("name")).lower()
            keys = (name, *_REF_STEMS["sla"])
            _add("deterministic_sla", name, "actual", check.get("actual"), keys)
            _add("deterministic_sla", name, "threshold", check.get("threshold"), keys)
    sla_step = ctx_obj.get("sla_step")
    if isinstance(sla_step, dict):
        _add("sla_step", sla_step.get("step_label"), "rps", sla_step.get("rps"), _REF_STEMS["sla_step"])
        for check in sla_step.get("checks") or []:
            if not isinstance(check, dict):
                continue
            name = _clean_text(check.get("name")).lower()
            keys = (name, *_REF_STEMS["sla"])
            _add("sla_step", name, "actual", check.get("actual"), keys)
            _add("sla_step", name, "threshold", check.get("threshold"), keys)
    _add_test_type_numeric_references(ctx_obj, _add)
    return refs


_HOLD_KEYS = _REF_STEMS["hold"]
_SPIKE_KEYS = _REF_STEMS["spike"]
_SATURATION_KEYS = _REF_STEMS["saturation"]


def _add_test_type_numeric_references(ctx_obj: Dict[str, Any], add: Any) -> None:
    """Registers numbers of the soak hold window and trends, spike analysis and stress saturation via ``add``."""
    holds: List[Any] = [ctx_obj.get("hold_window")]
    spikes: List[Any] = []
    for pack in _context_packs(ctx_obj):
        holds.append(pack.get("hold_window"))
        profile = pack.get("spike_profile")
        if isinstance(profile, dict):
            spikes.extend(profile.get("spikes") or [])
    stability = ctx_obj.get("stability_under_load")
    if isinstance(stability, dict):
        holds.append(stability.get("hold"))
        for trend in stability.get("trends") or []:
            if isinstance(trend, dict):
                keys = (str(trend.get("metric_kind") or ""), str(trend.get("title") or "").lower(), *_REF_STEMS["trend"])
                for field in ("first", "last", "change_pct", "slope_per_hour"):
                    add("stability_under_load", trend.get("series") or trend.get("label"), f"trend_{field}", trend.get(field), keys)
    for hold in holds:
        if isinstance(hold, dict):
            for field in ("level_rps", "actual_rps_mean", "min_bucket_rps", "drop_pct", "duration_min"):
                add("hold_window", hold.get("series") or hold.get("source_label"), field, hold.get(field), _HOLD_KEYS)
    analysis = ctx_obj.get("spike_analysis")
    if isinstance(analysis, dict):
        spikes.extend(analysis.get("spikes") or [])
        for field in ("baseline_rps", "threshold_rps", "max_spike_rps"):
            add("spike_analysis", analysis.get("series"), field, analysis.get(field), _SPIKE_KEYS)
    spikes.extend(item for item in ctx_obj.get("spike_windows") or [] if isinstance(item, dict))
    for spike in spikes:
        if not isinstance(spike, dict):
            continue
        name = spike.get("label")
        for field in ("level_rps", "peak_rps", "duration_sec", "overshoot_p95_pct", "overshoot_error_rate_pp", "recovery_time_sec", "rps_after"):
            add("spike_analysis", name, field, spike.get(field), _SPIKE_KEYS)
        for phase in ("pre", "during"):
            stats = spike.get(phase)
            if isinstance(stats, dict):
                for field in ("p95", "error_rate_pct"):
                    add("spike_analysis", name, f"{phase}_{field}", stats.get(field), _SPIKE_KEYS)
    saturation = ctx_obj.get("saturation")
    if isinstance(saturation, dict):
        for field in ("saturation_rps", "first_degraded_rps"):
            add("saturation", saturation.get("saturation_step_label"), field, saturation.get(field), _SATURATION_KEYS)
        resource = saturation.get("limiting_resource")
        if isinstance(resource, dict):
            for field in ("value", "threshold", "growth_pct"):
                add("saturation", resource.get("series") or resource.get("label"), f"resource_{field}", resource.get(field), _SATURATION_KEYS)


def _matching_numeric_references(text: str, refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blob = _clean_text(text).lower()
    if not blob:
        return []
    matched = [
        ref for ref in refs
        if any(key and key in blob for key in (ref.get("match_keys") or []))
    ]
    return matched


def _extract_numeric_claims(text: str) -> List[Dict[str, Any]]:
    blob = _clean_text(text)
    if not blob:
        return []
    blob = re.sub(
        r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?(?:\.\d+)?(?:Z)?\b",
        " ",
        blob,
    )
    blob = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", blob)
    claims: List[Dict[str, Any]] = []
    seen: set[Tuple[float, str]] = set()
    for match in re.finditer(
        r"(?<![A-Za-zА-Яа-яЁё])(-?\d+(?:[.,]\d+)?)\s*(%|ms|sec|secs|seconds|s|rps|qps)?\b",
        blob,
        flags=re.IGNORECASE,
    ):
        try:
            value = float(str(match.group(1)).replace(",", "."))
        except Exception:
            continue
        unit = str(match.group(2) or "").lower()
        key = (round(value, 6), unit)
        if key in seen:
            continue
        seen.add(key)
        claims.append({"value": value, "unit": unit})
        if len(claims) >= 12:
            break
    return claims


def _claim_value_variants(value: float, unit: str) -> List[float]:
    variants = {float(value)}
    if unit == "%":
        variants.add(float(value) / 100.0)
    if unit == "ms":
        variants.add(float(value) / 1000.0)
    if unit in {"s", "sec", "secs", "seconds"}:
        variants.add(float(value) * 1000.0)
    return [v for v in variants if v == v and abs(v) != float("inf")]


def _best_numeric_match_score(claim: Dict[str, Any], refs: List[Dict[str, Any]]) -> float:
    if not refs:
        return 0.0
    try:
        claim_value = float(claim.get("value"))
    except Exception:
        return 0.0
    unit = str(claim.get("unit") or "").lower()
    variants = _claim_value_variants(claim_value, unit)
    best = 0.0
    for ref in refs:
        ref_value = _safe_float_value(ref.get("value"))
        if ref_value is None:
            continue
        if abs(ref_value) <= 1e-9:
            best = max(best, 1.0 if any(abs(v) <= 1e-9 for v in variants) else 0.0)
            continue
        for variant in variants:
            abs_diff = abs(variant - ref_value)
            if abs_diff <= max(0.5, abs(ref_value) * 0.02):
                closeness = 1.0
            else:
                rel_error = abs_diff / max(abs(ref_value), 1e-9)
                closeness = max(0.0, 1.0 - min(rel_error / 0.25, 1.0))
            best = max(best, closeness)
    return float(max(0.0, min(best, 1.0)))


def score_candidate_by_data_details(parsed: Optional[LLMAnalysis], context_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Подробная эвристическая проверка кандидата по данным.

    Параметры:
        parsed (LLMAnalysis | None): Структурированный ответ.
        context_obj (dict): Контекст с секциями метрик.

    Возвращает:
        dict: Подробности и итоговый балл в диапазоне [0, 1].
    """
    details: Dict[str, Any] = {
        "overall": 0.0,
        "label_grounding": 0.0,
        "numeric_grounding": 0.0,
        "peak_consistency": None,
        "recommendation_grounding": 0.0,
        "structure_bonus": 0.0,
        "findings_total": 0,
        "matched_findings": 0,
        "numeric_claims_total": 0,
        "numeric_claims_supported": 0,
        "peak_checked": False,
        "actions_count": 0,
    }
    if not isinstance(parsed, LLMAnalysis):
        return details
    sections = _extract_sections_from_context(context_obj)
    labels = _collect_label_vocab(sections)
    numeric_refs = _build_numeric_reference_catalog(sections)
    findings = parsed.findings or []
    details["findings_total"] = len(findings)
    label_grounding = 0.0
    if findings:
        matches = sum(1 for f in findings if _finding_matches_labels(f, labels))
        details["matched_findings"] = matches
        label_grounding = matches / max(len(findings), 1)
    details["label_grounding"] = float(max(0.0, min(label_grounding, 1.0)))

    numeric_scores: List[float] = []
    numeric_supported = 0
    numeric_total = 0
    for finding in findings:
        blob = _finding_text_blob(finding)
        claims = _extract_numeric_claims(blob)
        if not claims:
            continue
        numeric_total += len(claims)
        relevant_refs = _matching_numeric_references(blob, numeric_refs)
        for claim in claims:
            best = _best_numeric_match_score(claim, relevant_refs)
            numeric_scores.append(best)
            if best >= 0.85:
                numeric_supported += 1
    numeric_grounding = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    details["numeric_grounding"] = float(max(0.0, min(numeric_grounding, 1.0)))
    details["numeric_claims_total"] = int(numeric_total)
    details["numeric_claims_supported"] = int(numeric_supported)

    peak_sections = _extract_lt_framework_sections(context_obj)
    peak_estimate = _extract_peak_estimate(peak_sections)
    peak = getattr(parsed, "peak_performance", None)
    peak_consistency: Optional[float] = None
    if peak_estimate and peak and getattr(peak, "max_rps", None) is not None:
        details["peak_checked"] = True
        try:
            claimed = float(peak.max_rps)
            actual = float(peak_estimate.get("max", 0.0))
            if actual > 0:
                rel_error = abs(claimed - actual) / max(actual, 1e-9)
                peak_consistency = max(0.0, 1.0 - min(rel_error, 1.0))
        except Exception:
            pass
    details["peak_consistency"] = (
        float(max(0.0, min(peak_consistency, 1.0))) if peak_consistency is not None else None
    )

    actions = parsed.recommended_actions or []
    details["actions_count"] = len(actions)
    action_presence = max(0.0, min(len(actions) / 3.0, 1.0)) if actions else 0.0
    support_anchor = max(
        details["label_grounding"],
        details["numeric_grounding"],
        0.5 * details["peak_consistency"] if isinstance(details["peak_consistency"], (int, float)) else 0.0,
    )
    details["recommendation_grounding"] = float(max(0.0, min(action_presence * support_anchor, 1.0)))

    has_structure = bool(findings or actions or peak)
    details["structure_bonus"] = 0.05 if has_structure else 0.0
    signal_weights: Dict[str, float] = {
        "label_grounding": 0.28,
        "numeric_grounding": 0.47,
        "recommendation_grounding": 0.15,
    }
    if isinstance(details["peak_consistency"], (int, float)):
        # Peak — полезный, но вторичный сигнал: он должен уточнять оценку,
        # а не доминировать над совпадением чисел и привязкой к метрикам.
        signal_weights["peak_consistency"] = 0.10
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in signal_weights.items():
        value = details.get(key)
        if not isinstance(value, (int, float)):
            continue
        weighted_sum += float(value) * float(weight)
        weight_total += float(weight)
    signal_score = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    details["signal_score"] = float(max(0.0, min(signal_score, 1.0)))
    details["peak_weight"] = float(signal_weights.get("peak_consistency", 0.0))
    overall = details["structure_bonus"] + ((1.0 - details["structure_bonus"]) * details["signal_score"])
    details["overall"] = float(max(0.0, min(overall, 1.0)))
    return details


def score_candidate_by_data(parsed: Optional[LLMAnalysis], context_obj: Dict[str, Any]) -> float:
    return float(score_candidate_by_data_details(parsed, context_obj).get("overall", 0.0) or 0.0)


# Weights of the candidate score; without usable judge scores the other two are renormalized.
LANGUAGE_MATCH_MIN = 0.5


def _free_text_values(parsed: Any) -> List[str]:
    """Free text of an answer: rationale, findings and actions, without component and metric names."""
    if parsed is None:
        return []
    data = parsed.dict() if hasattr(parsed, "dict") and not isinstance(parsed, dict) else parsed
    if not isinstance(data, dict):
        return []
    values: List[str] = [str(data.get("verdict_rationale") or "")]
    for finding in data.get("findings") or []:
        if not isinstance(finding, dict):
            values.append(str(finding or ""))
            continue
        values.extend(str(finding.get(key) or "") for key in ("summary", "evidence_summary", "evidence"))
        for item in finding.get("evidence_items") or []:
            if isinstance(item, dict):
                values.extend(str(item.get(key) or "") for key in ("note", "observed_value"))
    for action in data.get("recommended_actions") or []:
        if isinstance(action, dict):
            values.extend(str(action.get(key) or "") for key in ("summary", "details"))
        else:
            values.append(str(action or ""))
    return [value for value in values if value.strip()]


def language_match(parsed: Any, lang: Optional[str] = None) -> float:
    """Share of the words of the answer's free text written in the script of ``lang`` (0..1).

    Identifiers, numbers, units and ALL-CAPS abbreviations do not count; component and metric names are left
    out. An answer without natural words matches (1.0): there is nothing in the wrong language.
    """
    share = script_share(_free_text_values(parsed), lang)
    return 1.0 if share is None else float(share)


JUDGE_WEIGHT = 0.6
DATA_WEIGHT = 0.35
CONFIDENCE_WEIGHT = 0.05


def _candidate_confidence(parsed: Optional[LLMAnalysis]) -> float:
    """The model's own confidence of a parsed answer, clamped to 0..1 (0 when absent)."""
    if not isinstance(parsed, LLMAnalysis) or parsed.confidence is None:
        return 0.0
    try:
        return float(max(0.0, min(float(parsed.confidence), 1.0)))
    except (TypeError, ValueError):
        return 0.0


def _select_best_candidate(
    candidates: List[Tuple[str, Optional[LLMAnalysis]]],
    data_context: str,
    domain_key: Optional[str] = None,
    lang: Optional[str] = None,
) -> Tuple[str, Optional[LLMAnalysis], Dict[str, Any]]:
    """Best candidate: an answer in the report language first (language_match >= 0.5), then by final_score."""
    if not candidates:
        return "", None, {}
    lang = resolve_lang(lang)
    try:
        context_obj = json.loads(data_context) if data_context else {}
    except Exception:
        context_obj = {}
    if len(candidates) == 1:
        # Nothing to choose from, so no judge call. The data check needs no model call and still tells
        # how well the answer is grounded; the report shows its level instead of the judge's.
        text, parsed = candidates[0]
        details = score_candidate_by_data_details(parsed, context_obj)
        return text, parsed, {
            "selected_index": 0,
            "judge_meta": {"skipped": True, "reason": "single_candidate"},
            "data_score": float(details.get("overall", 0.0) or 0.0),
            "data_score_details": dict(details),
            "confidence": _candidate_confidence(parsed),
            "language_match": language_match(parsed, lang),
        }
    try:
        judge_scores, judge_meta = judge_candidates_with_llm(
            [text for (text, _) in candidates],
            data_context,
            domain_key=domain_key,
            lang=lang,
        )
    except Exception as exc:
        logger.warning("Judge failed for domain '%s': %s", domain_key, exc)
        judge_scores, judge_meta = {}, {"domain_key": _normalize_judge_domain_key(domain_key) or None, "error": str(exc)}
    judge_scores = _align_judge_indexes(judge_scores, len(candidates))
    judged = [idx for idx in range(len(candidates)) if idx in judge_scores]
    # A judge that failed or scored only some candidates must not count as zero for the rest:
    # then every candidate is ranked by the data check and its own confidence alone.
    use_judge = len(judged) == len(candidates)
    if not use_judge:
        reason = "error" if (judge_meta or {}).get("error") else ("partial_scores" if judged else "no_scores")
        judge_meta = {**(judge_meta or {}), "unavailable": True, "reason": reason, "scored_candidates": len(judged)}
        logger.warning("Judge gave no usable scores for domain '%s' (%s): candidates ranked by the data check", domain_key, reason)
    scored: List[Tuple[bool, float, int]] = []
    data_score_details_map: Dict[int, Dict[str, Any]] = {}
    language_map: Dict[int, float] = {}
    for idx, (text, parsed) in enumerate(candidates):
        judge_entry = judge_scores.get(idx) or {}
        judge_overall = float(judge_entry.get("overall", 0.0) or 0.0)
        data_score_details = score_candidate_by_data_details(parsed, context_obj)
        data_score_details_map[idx] = data_score_details
        data_score = float(data_score_details.get("overall", 0.0) or 0.0)
        conf = _candidate_confidence(parsed)
        if use_judge:
            final_score = JUDGE_WEIGHT * judge_overall + DATA_WEIGHT * data_score + CONFIDENCE_WEIGHT * conf
        else:
            final_score = (DATA_WEIGHT * data_score + CONFIDENCE_WEIGHT * conf) / (DATA_WEIGHT + CONFIDENCE_WEIGHT)
        language_map[idx] = language_match(parsed, lang)
        scored.append((language_map[idx] >= LANGUAGE_MATCH_MIN, final_score, idx))
    if scored:
        # An answer in the wrong language loses to any answer in the report language.
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_idx = scored[0][2]
        best_text, best_parsed = candidates[best_idx]
        judge_entry = judge_scores.get(best_idx) or {}
        data_score_details_best = data_score_details_map.get(best_idx) or score_candidate_by_data_details(best_parsed, context_obj)
        data_score_best = float(data_score_details_best.get("overall", 0.0) or 0.0)
        conf_best = 0.0
        if isinstance(best_parsed, LLMAnalysis) and best_parsed.confidence is not None:
            try:
                conf_best = float(best_parsed.confidence)
            except Exception:
                conf_best = 0.0
        final_score_best = [s for s in scored if s[2] == best_idx][0][1]
        score_info = {
            "selected_index": best_idx,
            "judge_meta": dict(judge_meta or {}),
            "data_score": float(data_score_best),
            "data_score_details": dict(data_score_details_best or {}),
            "confidence": float(max(0.0, min(conf_best, 1.0))),
            "final_score": float(final_score_best),
            "language_match": float(language_map.get(best_idx, 1.0)),
        }
        if use_judge:
            score_info["judge"] = {
                "overall": float(judge_entry.get("overall", 0.0) or 0.0),
                "factual": float(judge_entry.get("factual", 0.0) or 0.0),
                "completeness": float(judge_entry.get("completeness", 0.0) or 0.0),
                "specificity": float(judge_entry.get("specificity", 0.0) or 0.0),
                "rubric": dict(judge_entry.get("rubric") or {}),
            }
        return best_text, best_parsed, score_info
    # Unreachable with two or more candidates (every one is scored above); kept as a safe default.
    best_text, best_parsed = candidates[0]
    return best_text, best_parsed, {}


def _self_consistency_runtime_cfg() -> Dict[str, Any]:
    """Reads ``llm.self_consistency`` limits used to throttle candidate/critic calls.

    Keys: ``max_candidates`` (cap for ``k``), ``parallel_candidates`` and
    ``parallel_critics`` (run calls concurrently; default is sequential to avoid
    bursts against rate-limited providers), ``candidate_workers``,
    ``critic_workers`` and ``pause_sec_between_calls`` for the sequential mode.
    """
    sc_cfg = ((CONFIG.get("llm", {}) or {}).get("self_consistency", {}) or {})

    def _int(key: str, default: int) -> int:
        try:
            return max(1, int(sc_cfg.get(key, default)))
        except (TypeError, ValueError):
            return default

    try:
        pause_sec = max(0.0, float(sc_cfg.get("pause_sec_between_calls", 0.0)))
    except (TypeError, ValueError):
        pause_sec = 0.0
    return {
        "max_candidates": _int("max_candidates", 3),
        "parallel_candidates": bool(sc_cfg.get("parallel_candidates", True)),
        "parallel_critics": bool(sc_cfg.get("parallel_critics", True)),
        "candidate_workers": _int("candidate_workers", 3),
        "critic_workers": _int("critic_workers", 3),
        "pause_sec_between_calls": pause_sec,
    }


def _ask_llm_for_usage(domain_key: str, user_prompt: str, data_context: str) -> str:
    with usage_domain(domain_key):
        return ask_llm_with_text_data(user_prompt, data_context)


def _run_one_llm_call(domain_key: str, prompt: str, data_context: str) -> object:
    """One candidate call. A provider failure is returned, so the other candidates still run."""
    try:
        return _ask_llm_for_usage(domain_key, prompt, data_context)
    except Exception as exc:
        logger.error("LLM candidate failed for '%s': %s", domain_key, exc)
        return exc


def _run_llm_calls(
    prompts: List[str],
    data_context: str,
    parallel: bool,
    workers: int,
    pause_sec: float,
    domain_key: str = "unscoped",
) -> List[object]:
    """Runs ``ask_llm_with_text_data`` for each prompt either concurrently or one by one.

    Each item is the raw text or the exception that call raised.
    """
    if parallel and len(prompts) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(prompts), max(1, workers))) as executor:
            # The report language (a ContextVar) does not cross into pool threads by itself.
            futures = [
                executor.submit(contextvars.copy_context().run, _run_one_llm_call, domain_key, prompt, data_context)
                for prompt in prompts
            ]
            return [f.result() for f in futures]
    results: List[object] = []
    for idx, prompt in enumerate(prompts):
        results.append(_run_one_llm_call(domain_key, prompt, data_context))
        if idx < len(prompts) - 1 and pause_sec > 0:
            time.sleep(pause_sec)
    return results


def _two_pass_self_consistency(
    user_prompt: str,
    data_context: str,
    k: int = 3,
    return_scores: bool = False,
    domain_key: Optional[str] = None,
    lang: Optional[str] = None,
) -> tuple:
    """Двухпроходный алгоритм self-consistency: генерация k кандидатов + критик.

    Параметры:
        user_prompt (str): Текстовая инструкция.
        data_context (str): JSON с данными.
        k (int): Количество кандидатов (ограничивается ``llm.self_consistency.max_candidates``).
        return_scores (bool): Возвращать ли метрики выбора.
        domain_key (str | None): Домен для выбора judge-рубрики.
        lang (str | None): Язык прогона: промпты критика и судьи, выбор кандидата на нужном языке.
            По умолчанию текущая локаль (пайплайн выполняется в ``with_locale(report_language)``).

    Возвращает:
        tuple: `(best_text, best_parsed)` или `(best_text, best_parsed, scores)`.
    """
    candidates: list[tuple[str, Optional[LLMAnalysis]]] = []
    call_errors: List[BaseException] = []
    runtime_cfg = _self_consistency_runtime_cfg()
    gen_count = min(max(1, int(k)), int(runtime_cfg["max_candidates"]))
    pause_sec = float(runtime_cfg["pause_sec_between_calls"])
    usage_key = domain_key or "unscoped"
    if structured_output_enabled():
        # GigaChat function-calling returns a validated model; the critic pass is not needed.
        for i in range(gen_count):
            try:
                with usage_domain(usage_key):
                    structured = ask_llm_structured(user_prompt, data_context, LLMAnalysis)
            except Exception as exc:
                logger.error("Structured LLM candidate failed for '%s': %s", usage_key, exc)
                call_errors.append(exc)
            else:
                if getattr(structured, "analysis_error", None) is None:
                    candidates.append((json.dumps(structured.dict(), ensure_ascii=False, indent=2), structured))
            if i < gen_count - 1 and pause_sec > 0:
                time.sleep(pause_sec)
        if not candidates and call_errors:
            failed = make_failed_llm_analysis(call_errors[0])
            failed_text = json.dumps(failed.dict(), ensure_ascii=False, indent=2)
            return (failed_text, failed, {}) if return_scores else (failed_text, failed)
        best_text, best_parsed, score_info = _select_best_candidate(candidates, data_context, domain_key=domain_key)
        if best_parsed is not None:
            best_parsed = fill_missing_verdict_rationale(best_parsed)
            best_text = json.dumps(best_parsed.dict(), ensure_ascii=False, indent=2)
        return (best_text, best_parsed, score_info) if return_scores else (best_text, best_parsed)
    raw_results = _run_llm_calls(
        [user_prompt] * gen_count,
        data_context,
        parallel=bool(runtime_cfg["parallel_candidates"]),
        workers=int(runtime_cfg["candidate_workers"]),
        pause_sec=pause_sec,
        domain_key=usage_key,
    )
    need_critics = []
    parsed_or_raw: list[tuple[Optional[LLMAnalysis], str]] = []
    for raw in raw_results:
        if isinstance(raw, BaseException):
            call_errors.append(raw)
            continue
        text = str(raw or "")
        p = parse_llm_analysis_strict(text)
        if p is None:
            need_critics.append(text)
            parsed_or_raw.append((None, text))
        else:
            parsed_or_raw.append((p, text))
    if need_critics:
        critic_prompts = [_build_critic_prompt(r, lang) for r in need_critics]
        critic_results = _run_llm_calls(
            critic_prompts,
            data_context,
            parallel=bool(runtime_cfg["parallel_critics"]),
            workers=int(runtime_cfg["critic_workers"]),
            pause_sec=pause_sec,
            domain_key="critic",
        )
        ci = 0
        for p, raw in parsed_or_raw:
            if p is None:
                crit = critic_results[ci]
                ci += 1
                if isinstance(crit, BaseException):
                    call_errors.append(crit)
                    candidates.append((raw, None))
                    continue
                p2 = parse_llm_analysis_strict(str(crit or ""))
                if p2 is not None:
                    candidates.append((json.dumps(p2.dict(), ensure_ascii=False, indent=2), p2))
                else:
                    candidates.append((raw, None))
            else:
                candidates.append((json.dumps(p.dict(), ensure_ascii=False, indent=2), p))
    else:
        for p, _raw in parsed_or_raw:
            if p is not None:
                candidates.append((json.dumps(p.dict(), ensure_ascii=False, indent=2), p))
    best_text, best_parsed, score_info = _select_best_candidate(candidates, data_context, domain_key=domain_key)
    if best_parsed is None and best_text:
        try:
            mj = _extract_json_like(best_text)
            if mj:
                best_parsed = LLMAnalysis.parse_obj(mj)
                best_text = json.dumps(best_parsed.dict(), ensure_ascii=False, indent=2)
        except Exception:
            pass
    if best_parsed is None:
        failure = call_errors[0] if call_errors else {"reason": "invalid_json", "message": prompt_text("analysis_error.invalid_json", lang)}
        best_parsed = make_failed_llm_analysis(failure, best_text or "")
    if best_parsed is not None:
        best_parsed = fill_missing_verdict_rationale(best_parsed)
        best_text = json.dumps(best_parsed.dict(), ensure_ascii=False, indent=2)
    if return_scores:
        return best_text, best_parsed, score_info
    return best_text, best_parsed


def llm_two_pass_self_consistency(
    user_prompt: str,
    data_context: str,
    k: int = 3,
    return_scores: bool = False,
    domain_key: Optional[str] = None,
    lang: Optional[str] = None,
) -> tuple:
    """k candidates + critic + judge (see :func:`_two_pass_self_consistency`) in the language ``lang``.

    ``lang`` defaults to the current locale: the pipeline runs in ``with_locale(report_language)``. The whole call,
    pool threads included, runs in that locale, so every prompt and the default system prompt use one language.
    """
    code = resolve_lang(lang)
    with with_locale(code):
        return _two_pass_self_consistency(user_prompt, data_context, k, return_scores, domain_key, code)
