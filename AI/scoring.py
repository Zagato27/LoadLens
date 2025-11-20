import os
import json
import re
from typing import List, Dict, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, ValidationError, root_validator

from AI.providers import ask_llm_with_text_data


PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_PROMPT_CACHE: Dict[str, str] = {}


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
    "Перефразируйте все ТЕКСТОВЫЕ поля на русский язык (verdict, findings.summary, findings.evidence, recommended_actions, affected_components). "
    "Ключи JSON и значения поля severity оставьте на английском согласно схеме. "
    "Ниже дан проект ответа. Исправьте/нормализуйте его до СТРОГОГО JSON со схемой: "
    "{verdict, confidence, findings[], recommended_actions[]}. Каждый элемент findings обязан содержать severity (critical|high|medium|low) и component. "
    "Если component не указан — извлеките его из evidence по лейблам application|service|job|pod|instance, иначе 'unknown'. "
    "Если severity отсутствует — используйте 'low'. Дополнительно допускается поле peak_performance: {max_rps, max_time, drop_time, method}. "
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
        try:
            return json.loads(candidate)
        except Exception:
            pass
    fence = "```"
    if fence in text:
        parts = text.split(fence)
        for i in range(len(parts) - 1):
            block = parts[i + 1]
            if block.strip().startswith("json"):
                block_text = block.strip()[len("json"):].strip()
            else:
                block_text = block
            try:
                return json.loads(block_text)
            except Exception:
                continue
    return None


class FindingItem(BaseModel):
    summary: str = Field(default="")
    severity: Optional[str] = Field(default=None)
    component: Optional[str] = Field(default=None)
    evidence: Optional[str] = Field(default=None)


class PeakPerformance(BaseModel):
    max_rps: Optional[float] = Field(default=None)
    max_time: Optional[str] = Field(default=None)
    drop_time: Optional[str] = Field(default=None)
    method: Optional[str] = Field(default=None)


class LLMAnalysis(BaseModel):
    verdict: str = Field(default="нет данных")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    findings: List[Union[str, FindingItem]] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    affected_components: Optional[List[str]] = Field(default=None)
    peak_performance: Optional[PeakPerformance] = Field(default=None)

    @root_validator(pre=True)
    def _normalize_fields(cls, values: Dict[str, object]) -> Dict[str, object]:
        actions = values.get("recommended_actions") or values.get("actions") or []
        values["recommended_actions"] = actions
        findings = values.get("findings")
        if findings is None:
            values["findings"] = []
        return values


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
    try:
        maybe_json = json.loads(raw_text)
    except Exception:
        maybe_json = _extract_json_like(raw_text)
    if maybe_json is None:
        return None
    try:
        return LLMAnalysis.parse_obj(maybe_json)
    except ValidationError:
        return None


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
            for f in (p.findings or []):
                if isinstance(f, dict):
                    for key in ("summary", "evidence", "component"):
                        val = f.get(key)
                        if isinstance(val, str) and val.strip():
                            parts.append(val)
                else:
                    s = str(f).strip()
                    if s:
                        parts.append(s)
            for a in (p.recommended_actions or []):
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


def judge_candidates_with_llm(candidates_texts: List[str], data_context: str) -> Dict[int, Dict[str, float]]:
    """Запрашивает у LLM-судьи оценки нескольких кандидатских ответов.

    Параметры:
        candidates_texts (list[str]): JSON-тексты кандидатов.
        data_context (str): Контекст метрик в JSON.

    Возвращает:
        dict: Карта `{index: {"factual": float, ...}}`.
    """
    if not candidates_texts:
        return {}
    template = _get_prompt_template("judge_prompt.txt", JUDGE_PROMPT_FALLBACK)
    candidates_payload = [{"index": idx, "text": text} for idx, text in enumerate(candidates_texts)]
    prompt_text = template.replace("{{CANDIDATES_JSON}}", json.dumps(candidates_payload, ensure_ascii=False))
    prompt_text = prompt_text.replace("{{DATA_CONTEXT}}", data_context or "нет данных")
    raw = ask_llm_with_text_data(
        user_prompt=prompt_text,
        data_context="",
        llm_config={"force_json": True},
        system_prompt=JUDGE_SYSTEM_PROMPT
    )
    parsed = None
    try:
        parsed = _extract_json_like(raw) or json.loads(raw)
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return {}
    scores = parsed.get("scores")
    if not isinstance(scores, list):
        return {}
    result: Dict[int, Dict[str, float]] = {}
    for item in scores:
        if not isinstance(item, dict):
            continue
        idx_raw = item.get("index")
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        result[idx] = {
            "factual": float(item.get("factual", 0.0) or 0.0),
            "completeness": float(item.get("completeness", 0.0) or 0.0),
            "specificity": float(item.get("specificity", 0.0) or 0.0),
            "overall": float(item.get("overall", 0.0) or 0.0),
        }
    return result


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
            text_parts.extend([
                getattr(finding, "summary", ""),
                getattr(finding, "component", ""),
                getattr(finding, "evidence", ""),
            ])
        elif isinstance(finding, dict):
            text_parts.extend([
                str(finding.get("summary", "")),
                str(finding.get("component", "")),
                str(finding.get("evidence", "")),
            ])
        else:
            text_parts.append(str(finding))
    except Exception:
        text_parts.append(str(finding))

    blob = " ".join([part for part in text_parts if isinstance(part, str)])
    blob_lower = blob.lower()
    return any(label in blob_lower for label in labels if label)


def score_candidate_by_data(parsed: Optional[LLMAnalysis], context_obj: Dict[str, Any]) -> float:
    """Вычисляет эвристический балл кандидата на основе данных и peak_performance.

    Параметры:
        parsed (LLMAnalysis | None): Структурированный ответ.
        context_obj (dict): Контекст с секциями метрик.

    Возвращает:
        float: Балл в диапазоне [0, 1].
    """
    if not isinstance(parsed, LLMAnalysis):
        return 0.0
    sections = _extract_sections_from_context(context_obj)
    labels = _collect_label_vocab(sections)
    score = 0.15
    findings = parsed.findings or []
    if findings:
        matches = sum(1 for f in findings if _finding_matches_labels(f, labels))
        coverage = matches / max(len(findings), 1)
        score += 0.35 * max(0.0, min(coverage, 1.0))
    peak_estimate = _extract_peak_estimate(sections)
    peak = getattr(parsed, "peak_performance", None)
    if peak_estimate and peak and getattr(peak, "max_rps", None) is not None:
        try:
            claimed = float(peak.max_rps)
            actual = float(peak_estimate.get("max", 0.0))
            if actual > 0:
                rel_error = abs(claimed - actual) / max(actual, 1e-9)
                score += 0.35 * max(0.0, 1.0 - min(rel_error, 1.0))
        except Exception:
            pass
    actions = parsed.recommended_actions or []
    if actions:
        score += 0.15 * max(0.0, min(len(actions) / 3.0, 1.0))
    return max(0.0, min(score, 1.0))


def _select_best_candidate(
    candidates: List[Tuple[str, Optional[LLMAnalysis]]],
    data_context: str
) -> Tuple[str, Optional[LLMAnalysis], Dict[str, Any]]:
    if not candidates:
        return "", None, {}
    try:
        context_obj = json.loads(data_context) if data_context else {}
    except Exception:
        context_obj = {}
    try:
        judge_scores = judge_candidates_with_llm([text for (text, _) in candidates], data_context)
    except Exception:
        judge_scores = {}
    scored: List[Tuple[float, int]] = []
    for idx, (text, parsed) in enumerate(candidates):
        judge_entry = judge_scores.get(idx) or judge_scores.get(str(idx)) or {}
        judge_overall = float(judge_entry.get("overall", 0.0) or 0.0)
        data_score = score_candidate_by_data(parsed, context_obj)
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
        data_score_best = score_candidate_by_data(best_parsed, context_obj)
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
            },
            "data_score": float(data_score_best),
            "confidence": float(max(0.0, min(conf_best, 1.0))),
            "final_score": float(final_score_best),
        }
        return best_text, best_parsed, score_info
    best_text, best_parsed = _choose_best_candidate(candidates)
    return best_text, best_parsed, {}


def llm_two_pass_self_consistency(user_prompt: str, data_context: str, k: int = 3, return_scores: bool = False) -> tuple:
    """Двухпроходный алгоритм self-consistency: генерация k кандидатов + критик.

    Параметры:
        user_prompt (str): Текстовая инструкция.
        data_context (str): JSON с данными.
        k (int): Количество кандидатов.
        return_scores (bool): Возвращать ли метрики выбора.

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
    best_text, best_parsed, score_info = _select_best_candidate(candidates, data_context)
    if best_parsed is None and best_text:
        try:
            mj = _extract_json_like(best_text)
            if mj:
                best_parsed = LLMAnalysis.parse_obj(mj)
                best_text = json.dumps(best_parsed.dict(), ensure_ascii=False, indent=2)
        except Exception:
            pass
    if return_scores:
        return best_text, best_parsed, score_info
    return best_text, best_parsed


