import os
import json
import re
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, ValidationError, root_validator

from AI.providers import ask_llm_with_text_data


PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_PROMPT_CACHE: Dict[str, str] = {}
ALLOWED_VERDICTS = {"Успешно", "Есть риски", "Провал", "Недостаточно данных"}
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
    "final": os.path.join(JUDGE_RUBRIC_DIR, "final.txt"),
}
JUDGE_COMMON_RUBRIC_FILE = os.path.join(JUDGE_RUBRIC_DIR, "common.txt")
JUDGE_WARN_PROMPT_CHARS = max(8000, _read_int_env("LOADLENS_JUDGE_WARN_PROMPT_CHARS", 60000))


def read_prompt_from_file(filename: str) -> str:
    """Читает промпт из файла в UTF-8.

    Параметры:
        filename (str): Путь к файлу.

    Возвращает:
        str: Текст промпта.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


CRITIC_PROMPT_FALLBACK = (
    "Вы выступаете как строгий валидатор отчёта. Отвечайте на русском языке. "
    "Перефразируйте все ТЕКСТОВЫЕ поля на русский язык (verdict, verdict_rationale, findings.summary, findings.evidence_summary, findings.evidence_items.note, recommended_actions, affected_components). "
    "Ключи JSON и значения поля severity оставьте на английском согласно схеме. "
    "Ниже дан проект ответа. Исправьте/нормализуйте его до СТРОГОГО JSON со схемой: "
    "{verdict, verdict_rationale?, confidence, findings[], recommended_actions[]}. "
    "Каждый элемент findings обязан содержать id, severity (critical|high|medium|low) и component. "
    "Каждый элемент recommended_actions должен быть объектом {summary, details, priority (critical|high|medium|low), affected_components[], for_finding_ids[]}. "
    "id у findings должен быть коротким ASCII-идентификатором вроде f1, f2. "
    "Поле for_finding_ids обязано ссылаться на один или несколько id из findings и указывать, к каким проблемам относится рекомендация. "
    "Если verdict_rationale присутствует, сохраните его как короткое обоснование вердикта: сначала 1 краткий вывод, затем 2-4 ключевых фактора списком, без дублирования всего findings. "
    "Для findings используйте фиксированные поля: start_time, end_time, peak_time, evidence_summary, evidence_items[]. "
    "Каждый элемент evidence_items должен быть объектом {metric, observed_value, threshold, note}. "
    "Если component не указан — извлеките его из evidence_summary/evidence по лейблам application|service|job|pod|instance, иначе 'unknown'. "
    "Если severity отсутствует — используйте 'low'. Поле peak_performance допускается только для lt_framework/overall; "
    "в остальных доменах удаляйте его. Если test_profile.mode='stability' или peak_performance.not_applicable=true, "
    "не требуйте max_rps и не добавляйте его. "
    "Поле verdict ДОЛЖНО быть одним из: 'Успешно' | 'Есть риски' | 'Провал' | 'Недостаточно данных'. Синонимы нормализуйте к ближайшему значению. "
    "Никакого текста вне JSON. Если данных недостаточно — верните verdict='Недостаточно данных'.\n\nПроект ответа:\n{{CANDIDATE}}"
)


JUDGE_PROMPT_FALLBACK = (
    "Вы выступаете как независимый арбитр отчётов по нагрузочному тестированию. "
    "У вас есть агрегированные данные теста и несколько кандидатов ответов модели (каждый в JSON). "
    "Для каждого кандидата оцените три аспекта (0..1) и общий балл: factual, completeness, specificity. "
    "Рассчитайте overall = 0.5*factual + 0.3*completeness + 0.2*specificity. "
    "Ответьте СТРОГО JSON формата {\"scores\": [{\"index\": int, \"factual\": float, \"completeness\": float, \"specificity\": float, \"overall\": float}, ...]}. "
    "Если данных недостаточно для оценки, укажите 0. Контекст приведён ниже.\n\nКонтекст:\n{{DATA_CONTEXT}}\n\nКандидаты:\n{{CANDIDATES_JSON}}"
)


JUDGE_SYSTEM_PROMPT = (
    "Вы опытный инженер по нагрузочному тестированию и выступаете независимым судьёй. "
    "Используйте предоставленный контекст метрик, чтобы беспристрастно оценить кандидатов. "
    "Верните только JSON согласно запросу."
)


def _get_prompt_template(filename: str, fallback: str) -> str:
    cache_key = filename
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        text = read_prompt_from_file(path)
    except Exception:
        text = fallback
    _PROMPT_CACHE[cache_key] = text
    return text


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


def _normalize_verdict(value: Any) -> str:
    raw = _clean_text(value).lower()
    if not raw:
        return "Недостаточно данных"
    if raw in {v.lower() for v in ALLOWED_VERDICTS}:
        for verdict in ALLOWED_VERDICTS:
            if raw == verdict.lower():
                return verdict
    if any(token in raw for token in ("ok", "success", "passed", "green", "усп", "норма", "стаб")):
        return "Успешно"
    if any(token in raw for token in ("warn", "risk", "рис", "degrad", "предуп")):
        return "Есть риски"
    if any(token in raw for token in ("fail", "error", "red", "провал", "крит")):
        return "Провал"
    return "Недостаточно данных"


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
            details_parts.append("Шаги:\n- " + "\n- ".join(steps))
        if verification:
            details_parts.append("Проверка:\n- " + verification)
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


def _derive_verdict_rationale(verdict: str, findings: List[Dict[str, object]]) -> Optional[str]:
    verdict_text = _clean_text(verdict) or "Недостаточно данных"
    summaries: List[str] = []
    for item in findings or []:
        if not isinstance(item, dict):
            continue
        summary = _clean_text(item.get("summary") or item.get("title") or item.get("text"))
        if summary:
            summaries.append(summary)
        if len(summaries) >= 3:
            break
    if summaries:
        bullets = "\n".join(f"- {summary}" for summary in summaries)
        return f"Вердикт «{verdict_text}» сформирован по ключевым наблюдениям отчёта.\n{bullets}"
    if verdict_text == "Успешно":
        return "Вердикт «Успешно» сформирован потому, что существенных отклонений в доступных данных не обнаружено."
    if verdict_text == "Недостаточно данных":
        return "Вердикт «Недостаточно данных» сформирован потому, что доступных метрик недостаточно для уверенной оценки."
    return f"Вердикт «{verdict_text}» сформирован по доступным метрикам и выявленным рискам."


class LLMAnalysis(BaseModel):
    verdict: str = Field(default="нет данных")
    verdict_rationale: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    findings: List[FindingItem] = Field(default_factory=list)
    recommended_actions: List[RecommendedActionItem] = Field(default_factory=list)
    affected_components: Optional[List[str]] = Field(default=None)
    peak_performance: Optional[PeakPerformance] = Field(default=None)
    test_profile: Optional[Dict[str, Any]] = Field(default=None)
    deterministic_sla: Optional[Dict[str, Any]] = Field(default=None)
    stability_under_load: Optional[Dict[str, Any]] = Field(default=None)

    @root_validator(pre=True)
    def _normalize_fields(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(values, dict):
            values = {}
        values["verdict"] = _normalize_verdict(values.get("verdict"))
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


def make_invalid_llm_analysis(raw_text: str, reason: str = "invalid_json") -> LLMAnalysis:
    return LLMAnalysis.parse_obj(
        {
            "verdict": "Недостаточно данных",
            "verdict_rationale": (
                "Вердикт вынесен как \"Недостаточно данных\", потому что ответ LLM не прошёл "
                "строгую JSON-валидацию.\n"
                "- Структура ответа не соответствовала ожидаемой схеме.\n"
                "- Без корректного JSON нельзя безопасно извлечь факты и рекомендации."
            ),
            "confidence": 0.0,
            "findings": [
                {
                    "id": "finding_1",
                    "summary": "Ответ LLM не прошёл проверку структуры JSON и был заменён безопасным сообщением.",
                    "severity": "medium",
                    "component": "llm_output",
                    "evidence_summary": f"reason={reason}; excerpt={_safe_excerpt(raw_text)}",
                    "evidence": f"reason={reason}; excerpt={_safe_excerpt(raw_text)}",
                }
            ],
            "recommended_actions": [
                {
                    "summary": "Проверьте prompt, лимит токенов и повторите генерацию отчёта.",
                    "details": "Убедитесь, что prompt требует строгий JSON и что лимит токенов достаточен для полного ответа. После корректировки повторите генерацию и проверьте, что ответ проходит JSON-валидацию.",
                    "priority": "medium",
                    "affected_components": ["llm_output"],
                    "for_finding_ids": ["finding_1"],
                }
            ],
            "affected_components": ["llm_output"],
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


def _load_optional_prompt(filename: str) -> Optional[str]:
    path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        text = _get_prompt_template(filename, "")
    except Exception:
        return None
    return text if text else None


def _load_judge_rubric(domain_key: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    normalized = _normalize_judge_domain_key(domain_key)
    meta: Dict[str, Any] = {
        "domain_key": normalized or None,
        "rubric_domain": normalized or None,
        "used_rubric": False,
        "used_safe_path": False,
        "fallback_reason": None,
    }
    common_text = _load_optional_prompt(JUDGE_COMMON_RUBRIC_FILE)
    domain_file = JUDGE_RUBRIC_FILES.get(normalized or "")
    domain_text = _load_optional_prompt(domain_file) if domain_file else None
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


def _compact_judge_data_context(data_context: str) -> Tuple[str, Dict[str, Any]]:
    raw = str(data_context or "")
    meta: Dict[str, Any] = {
        "context_chars_raw": len(raw),
        "context_chars": len(raw),
        "context_truncated": False,
        "dropped_context_keys": [],
    }
    if not raw:
        return "нет данных", meta
    return raw, meta


def _build_judge_prompt(
    candidates_texts: List[str],
    data_context: str,
    domain_key: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    template = _get_prompt_template("judge_prompt.txt", JUDGE_PROMPT_FALLBACK)
    rubric_text, rubric_meta = _load_judge_rubric(domain_key)
    if "{{JUDGE_RUBRIC}}" in template:
        if rubric_text:
            template = template.replace("{{JUDGE_RUBRIC}}", rubric_text)
        else:
            template = JUDGE_PROMPT_FALLBACK
    normalized_domain = _normalize_judge_domain_key(domain_key) or "final"
    compact_context, context_meta = _compact_judge_data_context(data_context)
    candidates_payload = []
    for idx, text in enumerate(candidates_texts):
        candidates_payload.append({"index": idx, "text": _compact_jsonish_text(text)})
    prompt_text = template.replace("{{DOMAIN_KEY}}", normalized_domain)
    prompt_text = prompt_text.replace("{{CANDIDATES_JSON}}", json.dumps(candidates_payload, ensure_ascii=False))
    prompt_text = prompt_text.replace("{{DATA_CONTEXT}}", compact_context or "нет данных")
    meta = {
        **rubric_meta,
        **context_meta,
        "domain_key": normalized_domain,
        "candidate_count": len(candidates_payload),
        "prompt_chars": len(prompt_text),
    }
    return prompt_text, meta


def _parse_judge_scores(raw_text: str) -> Dict[int, Dict[str, Any]]:
    parsed = None
    try:
        parsed = _extract_json_like(raw_text) or json.loads(raw_text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        scores = parsed
    elif isinstance(parsed, dict):
        scores = parsed.get("scores")
    else:
        scores = None
    if not isinstance(scores, list):
        return {}
    result: Dict[int, Dict[str, Any]] = {}
    for item in scores:
        try:
            score_item = JudgeScoreItem.parse_obj(item)
        except ValidationError:
            continue
        result[int(score_item.index)] = score_item.dict()
    return result


def _build_critic_prompt(candidate_text: str) -> str:
    template = _get_prompt_template("critic_prompt.txt", CRITIC_PROMPT_FALLBACK)
    return template.replace("{{CANDIDATE}}", candidate_text)


def _choose_best_candidate(candidates: list) -> tuple[str, Optional[LLMAnalysis]]:
    if not candidates:
        return "", None
    from collections import Counter
    parsed_list = [p for (_, p) in candidates if p is not None]
    if not parsed_list:
        return candidates[0]
    verdicts = [p.verdict for p in parsed_list if p.verdict]
    majority_verdict = Counter(verdicts).most_common(1)[0][0] if verdicts else None

    def conf_val(p: Optional[LLMAnalysis]) -> float:
        if p is None or p.confidence is None:
            return 0.0
        try:
            return float(p.confidence)
        except Exception:
            return 0.0

    filtered = [(t, p) for (t, p) in candidates if p is not None and p.verdict == majority_verdict] if majority_verdict else []
    pool = filtered if filtered else candidates

    def _extract_text_for_lang_score(p: Optional[LLMAnalysis]) -> str:
        if p is None:
            return ""
        parts: list[str] = []
        try:
            if getattr(p, "verdict", None):
                parts.append(str(p.verdict))
            if getattr(p, "verdict_rationale", None):
                parts.append(str(p.verdict_rationale))
            for f in (p.findings or []):
                if isinstance(f, FindingItem):
                    for attr in ("summary", "component", "start_time", "end_time", "peak_time", "evidence_summary"):
                        val = getattr(f, attr, "")
                        if isinstance(val, str) and val.strip():
                            parts.append(val)
                    parts.extend(_finding_evidence_strings(f))
                elif isinstance(f, dict):
                    for key in ("summary", "evidence", "evidence_summary", "component", "start_time", "end_time", "peak_time"):
                        val = f.get(key)
                        if isinstance(val, str) and val.strip():
                            parts.append(val)
                    parts.extend(_finding_evidence_strings(f))
                else:
                    s = str(f).strip()
                    if s:
                        parts.append(s)
            for a in (p.recommended_actions or []):
                if isinstance(a, RecommendedActionItem):
                    if a.summary:
                        parts.append(str(a.summary))
                    if getattr(a, "details", None):
                        parts.append(str(a.details))
                    if getattr(a, "affected_components", None):
                        parts.extend([str(x) for x in a.affected_components if str(x).strip()])
                elif isinstance(a, dict):
                    summary = a.get("summary")
                    if isinstance(summary, str) and summary.strip():
                        parts.append(summary)
                    details = a.get("details")
                    if isinstance(details, str) and details.strip():
                        parts.append(details)
                else:
                    s = str(a).strip()
                    if s:
                        parts.append(s)
            if getattr(p, "affected_components", None):
                parts.extend([str(x) for x in p.affected_components if str(x).strip()])
        except Exception:
            pass
        return " \n".join(parts)

    def _russian_ratio(text: str) -> float:
        if not isinstance(text, str) or not text:
            return 0.0
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
        if not letters:
            return 0.0
        cyr = re.findall(r"[А-Яа-яЁё]", text)
        return float(len(cyr)) / float(len(letters))

    def lang_score(p: Optional[LLMAnalysis]) -> float:
        try:
            blob = _extract_text_for_lang_score(p)
            return _russian_ratio(blob)
        except Exception:
            return 0.0

    best = max(pool, key=lambda tp: (lang_score(tp[1]), conf_val(tp[1])))
    return best


def judge_candidates_with_llm(
    candidates_texts: List[str],
    data_context: str,
    domain_key: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Запрашивает у LLM-судьи оценки нескольких кандидатских ответов.

    Параметры:
        candidates_texts (list[str]): JSON-тексты кандидатов.
        data_context (str): Контекст метрик в JSON.
        domain_key (str | None): Ключ домена для выбора judge-рубрики.

    Возвращает:
        tuple: `(scores_by_index, judge_meta)`.
    """
    if not candidates_texts:
        return {}, {"domain_key": _normalize_judge_domain_key(domain_key) or None, "empty_candidates": True}
    prompt_text, meta = _build_judge_prompt(candidates_texts, data_context, domain_key)
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
        user_prompt=prompt_text,
        data_context="",
        llm_config={"force_json": True},
        system_prompt=JUDGE_SYSTEM_PROMPT
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


def _select_best_candidate(
    candidates: List[Tuple[str, Optional[LLMAnalysis]]],
    data_context: str,
    domain_key: Optional[str] = None,
) -> Tuple[str, Optional[LLMAnalysis], Dict[str, Any]]:
    if not candidates:
        return "", None, {}
    try:
        context_obj = json.loads(data_context) if data_context else {}
    except Exception:
        context_obj = {}
    try:
        judge_scores, judge_meta = judge_candidates_with_llm(
            [text for (text, _) in candidates],
            data_context,
            domain_key=domain_key,
        )
    except Exception as exc:
        logger.warning("Judge failed for domain '%s': %s", domain_key, exc)
        judge_scores, judge_meta = {}, {"domain_key": _normalize_judge_domain_key(domain_key) or None, "error": str(exc)}
    scored: List[Tuple[float, int]] = []
    data_score_details_map: Dict[int, Dict[str, Any]] = {}
    for idx, (text, parsed) in enumerate(candidates):
        judge_entry = judge_scores.get(idx) or judge_scores.get(str(idx)) or {}
        judge_overall = float(judge_entry.get("overall", 0.0) or 0.0)
        data_score_details = score_candidate_by_data_details(parsed, context_obj)
        data_score_details_map[idx] = data_score_details
        data_score = float(data_score_details.get("overall", 0.0) or 0.0)
        conf = 0.0
        if isinstance(parsed, LLMAnalysis) and parsed.confidence is not None:
            try:
                conf = float(parsed.confidence)
            except Exception:
                conf = 0.0
        final_score = 0.6 * judge_overall + 0.35 * data_score + 0.05 * max(0.0, min(conf, 1.0))
        scored.append((final_score, idx))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_idx = scored[0][1]
        best_text, best_parsed = candidates[best_idx]
        judge_entry = judge_scores.get(best_idx) or judge_scores.get(str(best_idx)) or {}
        data_score_details_best = data_score_details_map.get(best_idx) or score_candidate_by_data_details(best_parsed, context_obj)
        data_score_best = float(data_score_details_best.get("overall", 0.0) or 0.0)
        conf_best = 0.0
        if isinstance(best_parsed, LLMAnalysis) and best_parsed.confidence is not None:
            try:
                conf_best = float(best_parsed.confidence)
            except Exception:
                conf_best = 0.0
        final_score_best = [s for s in scored if s[1] == best_idx][0][0]
        score_info = {
            "selected_index": best_idx,
            "judge": {
                "overall": float(judge_entry.get("overall", 0.0) or 0.0),
                "factual": float(judge_entry.get("factual", 0.0) or 0.0),
                "completeness": float(judge_entry.get("completeness", 0.0) or 0.0),
                "specificity": float(judge_entry.get("specificity", 0.0) or 0.0),
                "rubric": dict(judge_entry.get("rubric") or {}),
            },
            "judge_meta": dict(judge_meta or {}),
            "data_score": float(data_score_best),
            "data_score_details": dict(data_score_details_best or {}),
            "confidence": float(max(0.0, min(conf_best, 1.0))),
            "final_score": float(final_score_best),
        }
        return best_text, best_parsed, score_info
    best_text, best_parsed = _choose_best_candidate(candidates)
    return best_text, best_parsed, {}


def llm_two_pass_self_consistency(
    user_prompt: str,
    data_context: str,
    k: int = 3,
    return_scores: bool = False,
    domain_key: Optional[str] = None,
) -> tuple:
    """Двухпроходный алгоритм self-consistency: генерация k кандидатов + критик.

    Параметры:
        user_prompt (str): Текстовая инструкция.
        data_context (str): JSON с данными.
        k (int): Количество кандидатов.
        return_scores (bool): Возвращать ли метрики выбора.
        domain_key (str | None): Домен для выбора judge-рубрики.

    Возвращает:
        tuple: `(best_text, best_parsed)` или `(best_text, best_parsed, scores)`.
    """
    candidates: list[tuple[str, Optional[LLMAnalysis]]] = []
    gen_count = max(1, int(k))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=gen_count) as executor:
        futures = [executor.submit(ask_llm_with_text_data, user_prompt, data_context) for _ in range(gen_count)]
        raw_results = [f.result() for f in futures]
    need_critics = []
    parsed_or_raw: list[tuple[Optional[LLMAnalysis], str]] = []
    for raw in raw_results:
        p = parse_llm_analysis_strict(raw)
        if p is None:
            need_critics.append(raw)
            parsed_or_raw.append((None, raw))
        else:
            parsed_or_raw.append((p, raw))
    if need_critics:
        from concurrent.futures import ThreadPoolExecutor
        critic_prompts = [_build_critic_prompt(r) for r in need_critics]
        with ThreadPoolExecutor(max_workers=len(need_critics)) as executor:
            critic_results = [executor.submit(ask_llm_with_text_data, cp, data_context).result() for cp in critic_prompts]
        ci = 0
        for p, raw in parsed_or_raw:
            if p is None:
                crit = critic_results[ci]
                ci += 1
                p2 = parse_llm_analysis_strict(crit)
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
        best_parsed = make_invalid_llm_analysis(best_text or "", reason="failed_strict_validation")
        best_text = json.dumps(best_parsed.dict(), ensure_ascii=False, indent=2)
    if return_scores:
        return best_text, best_parsed, score_info
    return best_text, best_parsed


