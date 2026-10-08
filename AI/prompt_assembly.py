"""Assembles the final LLM prompt: context guide + analysis guidance + domain text + response format.

Numeric thresholds are never hard-coded in prompt files; ``build_analysis_guidance``
derives them from the service SLA, the system context rules and the test type so
the model judges the run against the customer's criteria only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SLA_FIELDS: List[tuple[str, str, str]] = [
    ("target_rps", "Целевой RPS (target_rps)", " RPS"),
    ("max_error_rate_pct", "Максимальная доля ошибок", " %"),
    ("max_p95_ms", "Максимальный p95 latency", " мс"),
    ("max_p99_ms", "Максимальный p99 latency", " мс"),
    ("max_cpu_pct", "Максимальная загрузка CPU", " %"),
    ("max_memory_pct", "Максимальная загрузка памяти", " %"),
]
SLA_QUERY_KEYS: Dict[str, str] = {
    "target_rps": "max_performance_query",
    "max_error_rate_pct": "error_rate_query",
    "max_p95_ms": "p95_query",
    "max_p99_ms": "p99_query",
    "max_cpu_pct": "cpu_query",
    "max_memory_pct": "memory_query",
}
PEAK_DOMAINS = frozenset({"lt_framework"})

TEST_TYPE_ALIASES: Dict[str, tuple[str, ...]] = {
    "step": ("step", "ступенчатый", "поиск максимальной производительности", "max"),
    "soak": (
        "soak", "stability", "stable", "endurance", "long", "long_run", "longevity",
        "reliability", "долговременный", "стабильность", "стаб", "длительный",
    ),
    "spike": ("spike", "всплеск", "всплески"),
    "stress": ("stress", "стресс"),
}

TEST_TYPE_OVERLAYS: Dict[str, str] = {
    "step": (
        "[Профиль теста: ступенчатый поиск максимальной производительности]\n"
        "- Единица анализа — ступень нагрузки (load_steps / step_table). Сравнивай метрики между ступенями,\n"
        "  а не только начало и конец теста.\n"
        "- Максимальная производительность — последняя стабильная ступень (stable_max в lt_framework),\n"
        "  а не кратковременный пик max. Для итога значение уже посчитано в designated_peak_performance.\n"
        "- Интересует: на какой ступени началась деградация (рост latency/ошибок быстрее нагрузки),\n"
        "  какие ресурсы упёрлись в предел, что произошло у порога."
    ),
    "soak": (
        "[Профиль теста: долговременная стабильность (soak/stability)]\n"
        "- Это тест удержания заданной нагрузки, а не поиск максимальной производительности.\n"
        "  Не определяй peak_performance и не делай вывод о максимальном RPS.\n"
        "- Нагрузка постоянна, поэтому load_steps — равные интервалы времени. Ищи дрейф: монотонный рост\n"
        "  latency, памяти, GC, очередей от интервала к интервалу (anomalies.kind = drift/shift).\n"
        "- Рост метрики при неизменной нагрузке — признак утечки или накопления; отмечай его явно.\n"
        "- Если нужно упомянуть производительность, формулируй это как стабильность под заданной нагрузкой."
    ),
    "spike": (
        "[Профиль теста: всплески (spike)]\n"
        "- Интересует реакция на резкое изменение нагрузки: перерегулирование latency, ошибки в окне\n"
        "  всплеска и время возврата к исходному уровню.\n"
        "- Сравнивай интервал всплеска с интервалами до и после него."
    ),
    "stress": (
        "[Профиль теста: стресс]\n"
        "- Ищи точку насыщения: ступень, после которой пропускная способность перестаёт расти, а latency\n"
        "  и ошибки растут. Точка насыщения — последняя стабильная ступень, не кратковременный пик.\n"
        "- Назови ограничивающий ресурс (если он виден в данных) и характер деградации."
    ),
}

NO_SLA_GUIDANCE = (
    "Числовых SLA для этого сервиса не задано. Не придумывай нормативы («общепринятые», «рекомендуемые»,\n"
    "«по мировой практике» пороги использовать ЗАПРЕЩЕНО). Оценивай относительное изменение метрик по\n"
    "ступеням (step_table.change_vs_first_pct, per_step), аномалии (anomalies) и отличие от baseline.\n"
    "Severity назначай по величине и устойчивости изменения: рост быстрее нагрузки или рост при неизменной\n"
    "нагрузке — риск; отсутствие изменений при росте нагрузки — норма."
)


def _cfg_number(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None and str(value).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _cfg_flag(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_test_type(test_type: Optional[str]) -> str:
    """Maps user-facing test type names (RU/EN) to step/soak/spike/stress or ''."""
    raw = (test_type or "").strip().lower()
    for key, aliases in TEST_TYPE_ALIASES.items():
        if raw in aliases:
            return key
    return ""


def sla_guidance_lines(sla_cfg: Dict[str, Any], domain: Optional[str] = None) -> List[str]:
    """SLA thresholds of the service as prompt lines; empty when no numeric SLA is set."""
    lines: List[str] = []
    for key, label, unit in SLA_FIELDS:
        value = _cfg_number(sla_cfg.get(key))
        if value is None:
            continue
        query = str(sla_cfg.get(SLA_QUERY_KEYS[key]) or "").strip()
        source = (
            f"проверяется программно по запросу «{query}»"
            if query
            else "запрос не выбран, программно не проверяется"
        )
        lines.append(f"- {label}: {value:g}{unit} — {source}")
    if not lines:
        return []
    lines.append(
        "- Каждый порог относится только к своему запросу. Не сравнивай с ним другие метрики: "
        "среднее время обработки сервиса — не p95 нагрузочного инструмента, CPU процесса — не CPU узла."
    )
    if domain is None or domain in PEAK_DOMAINS:
        perf_query = str(sla_cfg.get("max_performance_query") or "").strip()
        if perf_query:
            lines.append(f"- Метрика максимальной производительности: «{perf_query}» (используется stable_max)")
        stable_min = _cfg_number(sla_cfg.get("min_stable_minutes"))
        lines.append(f"- Минимальная длительность стабильной ступени: {stable_min if stable_min is not None else 5.0:g} мин")
        allow_fallback = _cfg_flag(sla_cfg.get("target_rps_allow_peak_fallback"), default=True)
        lines.append(
            "- Если stable_max отсутствует, для target_rps "
            + ("разрешено использовать пиковый max" if allow_fallback else "пиковый max использовать ЗАПРЕЩЕНО")
        )
        lines.append(
            "- Если target_rps достигнут (stable_max >= target_rps), критерий target_rps выполнен, даже если после него была деградация."
        )
    else:
        lines.append(
            "- Целевой RPS и максимальная производительность проверены программно на ступени из sla_step. "
            "По метрикам этого домена их не оценивай и RPS сервисов с target_rps не сравнивай."
        )
    lines.append(
        "- Единственный источник числовых порогов — этот блок и deterministic_sla в контексте. Не используй нормативы, которых здесь нет."
    )
    return lines


def _string_list(container: Any, key: str) -> List[str]:
    values = container.get(key) if isinstance(container, dict) and isinstance(container.get(key), list) else []
    return [str(v).strip() for v in values if str(v).strip()]


def system_context_rules(system_context: Optional[Dict[str, Any]]) -> List[str]:
    """Customer rules from system_context.operational_context that affect severity and focus."""
    operational = (system_context or {}).get("operational_context") if isinstance(system_context, dict) else None
    lines: List[str] = []
    for rule in _string_list(operational, "normal_degradation_rules"):
        lines.append(f"- Считается нормой (не понижай вердикт за это): {rule}")
    for item in _string_list(operational, "analysis_focus"):
        lines.append(f"- Фокус анализа заказчика: {item}")
    for risk in _string_list(operational, "known_risks"):
        lines.append(f"- Известный риск (проверь по данным): {risk}")
    for constraint in _string_list(operational, "known_constraints"):
        lines.append(f"- Известное ограничение: {constraint}")
    return lines


def build_analysis_guidance(
    sla_cfg: Optional[Dict[str, Any]],
    system_context: Optional[Dict[str, Any]],
    test_type: Optional[str],
    domain: Optional[str] = None,
) -> str:
    """Prompt block with the customer's criteria: SLA, system-context rules and the test profile."""
    sections: List[str] = ["ОРИЕНТИРЫ АНАЛИЗА (сформированы автоматически из настроек сервиса):"]
    sla_lines = sla_guidance_lines(sla_cfg if isinstance(sla_cfg, dict) else {}, domain)
    sections.append("SLA сервиса:\n" + ("\n".join(sla_lines) if sla_lines else NO_SLA_GUIDANCE))
    rules = system_context_rules(system_context)
    if rules:
        sections.append("Правила заказчика из описания системы:\n" + "\n".join(rules))
    overlay = TEST_TYPE_OVERLAYS.get(normalize_test_type(test_type))
    if overlay:
        sections.append(overlay)
    else:
        sections.append("[Профиль теста не указан] Ступени в load_steps — равные интервалы времени; сравнивай метрики между интервалами.")
    return "\n\n".join(sections)


def assemble_prompt(domain_prompt: str, context_guide: str, analysis_guidance: str, response_format: str) -> str:
    """Final prompt text sent with the JSON context; runtime overrides affect ``domain_prompt`` only."""
    parts = [context_guide.strip(), analysis_guidance.strip(), (domain_prompt or "").strip(), response_format.strip()]
    return "\n\n".join(part for part in parts if part)


__all__ = [
    "TEST_TYPE_OVERLAYS",
    "assemble_prompt",
    "build_analysis_guidance",
    "normalize_test_type",
    "sla_guidance_lines",
    "system_context_rules",
]
