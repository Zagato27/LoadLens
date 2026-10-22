# SPDX-License-Identifier: Apache-2.0
"""Assembles the final LLM prompt: context guide + analysis guidance + domain text + response format.

Numeric thresholds are never hard-coded in prompt files; ``build_analysis_guidance``
derives them from the service SLA, the system context rules and the test type so
the model judges the run against the customer's criteria only.

Texts come from ``AI/prompts/<lang>/`` (``profiles/*.txt``, ``_no_sla.txt`` and ``guidance.*`` keys of
``_strings.json``); ``lang=None`` means the current locale, which is the report language inside a run.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from AI.prompt_store import load_prompt, resolve_lang, text
from AI.test_profile import TEST_TYPE_ALIASES, is_capacity_like, normalize_test_type, profile_for

# SLA fields shown in the guidance, in this order; labels and units are ``guidance.sla_field.<key>`` /
# ``guidance.sla_unit.<key>`` of _strings.json.
SLA_FIELDS: List[str] = ["target_rps", "max_error_rate_pct", "max_p95_ms", "max_p99_ms", "max_cpu_pct", "max_memory_pct"]
SLA_QUERY_KEYS: Dict[str, str] = {
    "target_rps": "max_performance_query",
    "max_error_rate_pct": "error_rate_query",
    "max_p95_ms": "p95_query",
    "max_p99_ms": "p99_query",
    "max_cpu_pct": "cpu_query",
    "max_memory_pct": "memory_query",
}
PEAK_DOMAINS = frozenset({"lt_framework"})
TEST_TYPE_PROFILES = ("step", "soak", "spike", "stress")


def test_type_overlay(test_type: Optional[str], lang: Optional[str] = None) -> str:
    """Profile block of a normalized test type (``profiles/<type>.txt``); the ``none`` block for other types."""
    name = test_type if test_type in TEST_TYPE_PROFILES else "none"
    return load_prompt(f"profiles/{name}.txt", lang).strip()


def no_sla_guidance(lang: Optional[str] = None) -> str:
    return load_prompt("_no_sla.txt", lang).strip()


# Kept for callers of the ru tables; the run uses test_type_overlay(test_type, lang).
TEST_TYPE_OVERLAYS: Dict[str, str] = {name: test_type_overlay(name, "ru") for name in TEST_TYPE_PROFILES}


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


def sla_guidance_lines(
    sla_cfg: Dict[str, Any],
    domain: Optional[str] = None,
    mode: str = "capacity",
    lang: Optional[str] = None,
) -> List[str]:
    """SLA thresholds of the service as prompt lines; empty when no numeric SLA is set.

    ``mode`` is the test-profile mode: the stable_max / «degradation after the target is fine»
    rules apply to step and stress tests only.
    """
    code = resolve_lang(lang)
    lines: List[str] = []
    for key in SLA_FIELDS:
        value = _cfg_number(sla_cfg.get(key))
        if value is None:
            continue
        query = str(sla_cfg.get(SLA_QUERY_KEYS[key]) or "").strip()
        source = text("guidance.sla_source_query", code, query=query) if query else text("guidance.sla_source_none", code)
        lines.append(text(
            "guidance.sla_line", code,
            label=text(f"guidance.sla_field.{key}", code),
            value=f"{value:g}",
            unit=text(f"guidance.sla_unit.{key}", code),
            source=source,
        ))
    if not lines:
        return []
    lines.append(text("guidance.sla_scope", code))
    if not is_capacity_like(mode):
        lines.extend(_non_capacity_target_lines(sla_cfg, domain, mode, code))
    elif domain is None or domain in PEAK_DOMAINS:
        perf_query = str(sla_cfg.get("max_performance_query") or "").strip()
        if perf_query:
            lines.append(text("guidance.peak_query", code, query=perf_query))
        stable_min = _cfg_number(sla_cfg.get("min_stable_minutes"))
        lines.append(text("guidance.min_stable", code, minutes=f"{stable_min if stable_min is not None else 5.0:g}"))
        allow_fallback = _cfg_flag(sla_cfg.get("target_rps_allow_peak_fallback"), default=True)
        lines.append(text("guidance.peak_fallback_allowed" if allow_fallback else "guidance.peak_fallback_forbidden", code))
        lines.append(text("guidance.target_reached", code))
    else:
        lines.append(text("guidance.domain_sla_step", code))
    lines.append(text("guidance.only_source", code))
    return lines


def _non_capacity_target_lines(sla_cfg: Dict[str, Any], domain: Optional[str], mode: str, lang: str) -> List[str]:
    """How target_rps and the SLA window are checked for soak and spike tests."""
    peak_domain = domain is None or domain in PEAK_DOMAINS
    if mode == "stability":
        if peak_domain:
            perf_query = str(sla_cfg.get("max_performance_query") or "").strip()
            source = text("guidance.soak_query", lang, query=perf_query) if perf_query else ""
            return [text("guidance.soak_target", lang, source=source), text("guidance.soak_thresholds", lang)]
        return [text("guidance.soak_domain", lang)]
    if mode == "spike":
        if peak_domain:
            return [text("guidance.spike_target", lang), text("guidance.spike_thresholds", lang)]
        return [text("guidance.spike_domain", lang)]
    return []


def _string_list(container: Any, key: str) -> List[str]:
    values = container.get(key) if isinstance(container, dict) and isinstance(container.get(key), list) else []
    return [str(v).strip() for v in values if str(v).strip()]


def system_context_rules(system_context: Optional[Dict[str, Any]], lang: Optional[str] = None) -> List[str]:
    """Customer rules from system_context.operational_context that affect severity and focus."""
    code = resolve_lang(lang)
    operational = (system_context or {}).get("operational_context") if isinstance(system_context, dict) else None
    lines: List[str] = []
    for list_key, string_key in (
        ("normal_degradation_rules", "guidance.rule_normal"),
        ("analysis_focus", "guidance.rule_focus"),
        ("known_risks", "guidance.rule_risk"),
        ("known_constraints", "guidance.rule_constraint"),
    ):
        for rule in _string_list(operational, list_key):
            lines.append(text(string_key, code, rule=rule))
    return lines


def metric_inventory_lines(
    items: Optional[Sequence[Mapping[str, Any]]],
    labels_with_data: Optional[Iterable[str]] = None,
    lang: Optional[str] = None,
) -> List[str]:
    """Lines of the «domain metrics» block: one per configured query, in the order of the block.

    ``items`` are ``AI.query_packs.series_inventory`` items (label, meaning, unit_title, keys, description).
    ``labels_with_data`` are the labels that returned values in this run; ``None`` means unknown, so no row is
    marked as empty.
    """
    code = resolve_lang(lang)
    with_data = None if labels_with_data is None else {str(label) for label in labels_with_data}
    lines: List[str] = []
    incomplete = False
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        meaning = " ".join(str(item.get("meaning") or "").split())
        unit = str(item.get("unit_title") or "").strip()
        keys = [str(key).strip() for key in (item.get("keys") or []) if str(key).strip()]
        description = " ".join(str(item.get("description") or "").split())
        facts = [meaning or text("guidance.metric_no_role", code)]
        facts.append(text("guidance.metric_unit", code, unit=unit) if unit else text("guidance.metric_no_unit", code))
        if keys:
            facts.append(text("guidance.metric_keys", code, keys=", ".join(keys)))
        if with_data is not None and label not in with_data:
            facts.append(text("guidance.metric_no_data", code))
        if description:
            facts.append(text("guidance.metric_description", code, text=description))
        incomplete = incomplete or not meaning or not unit
        lines.append(text("guidance.metric_line", code, label=label, facts="; ".join(facts)))
    if lines and incomplete:
        lines.append(text("guidance.metric_unknown", code))
    return lines


def build_analysis_guidance(
    sla_cfg: Optional[Dict[str, Any]],
    system_context: Optional[Dict[str, Any]],
    test_type: Optional[str],
    domain: Optional[str] = None,
    stack_notes: Optional[Sequence[Tuple[str, str]]] = None,
    lang: Optional[str] = None,
    metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    labels_with_data: Optional[Iterable[str]] = None,
) -> str:
    """Prompt block with the customer's criteria: SLA, system-context rules, the domain metrics and the test profile.

    ``stack_notes`` are (pack title, text) of the installed query packs that explain their metrics.
    ``metrics`` are the configured queries of the domain (``AI.query_packs.series_inventory``) and
    ``labels_with_data`` the labels that returned values in this run (see :func:`metric_inventory_lines`).
    """
    code = resolve_lang(lang)
    sections: List[str] = [text("guidance.header", code)]
    profile = profile_for(test_type)
    sla_lines = sla_guidance_lines(sla_cfg if isinstance(sla_cfg, dict) else {}, domain, profile.mode, code)
    sections.append(text("guidance.sla_title", code) + "\n" + ("\n".join(sla_lines) if sla_lines else no_sla_guidance(code)))
    rules = system_context_rules(system_context, code)
    if rules:
        sections.append(text("guidance.rules_title", code) + "\n" + "\n".join(rules))
    metric_lines = metric_inventory_lines(metrics, labels_with_data, code)
    if metric_lines:
        sections.append(text("guidance.metrics_title", code) + "\n" + "\n".join(metric_lines))
    notes = [f"- {title}: {' '.join(str(note).split())}" for title, note in (stack_notes or []) if str(note).strip()]
    if notes:
        sections.append(text("guidance.stack_title", code) + "\n" + "\n".join(notes))
    sections.append(test_type_overlay(profile.test_type, code))
    return "\n\n".join(sections)


def assemble_prompt(domain_prompt: str, context_guide: str, analysis_guidance: str, response_format: str) -> str:
    """Final prompt text sent with the JSON context; runtime overrides affect ``domain_prompt`` only."""
    parts = [context_guide.strip(), analysis_guidance.strip(), (domain_prompt or "").strip(), response_format.strip()]
    return "\n\n".join(part for part in parts if part)


__all__ = [
    "TEST_TYPE_ALIASES",
    "TEST_TYPE_OVERLAYS",
    "assemble_prompt",
    "build_analysis_guidance",
    "metric_inventory_lines",
    "no_sla_guidance",
    "normalize_test_type",
    "sla_guidance_lines",
    "system_context_rules",
    "test_type_overlay",
]
