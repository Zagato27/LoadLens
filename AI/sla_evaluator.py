"""Программная оценка теста по SLA-критериям.

Принцип:
- target_rps — главный критерий. Если достигнут (stable_max >= target_rps),
  тест считается успешным ДАЖЕ если система деградировала после этого.
- Остальные пороги (error_rate, p95, p99, cpu, memory) — вторичные.
  Их нарушение при достигнутом target_rps приводит к «Есть риски», не «Провал».
- Для stability/soak тестов без target_rps ресурсные превышения (CPU/memory)
  являются рисками, а не основанием для «Провал» без нарушений latency/errors.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _keyword_matches(label: str, keyword: str) -> bool:
    """Сопоставляет keyword с label как substring или regex."""
    kw = str(keyword or "").strip().lower()
    if not kw:
        return False
    has_regex_meta = any(ch in kw for ch in (".", "*", "+", "?", "[", "]", "(", ")", "|", "^", "$", "\\"))
    if has_regex_meta:
        try:
            return re.search(kw, label) is not None
        except re.error:
            return kw in label
    return kw in label


def _safe_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _find_section_by_label(
    sections: List[Dict[str, Any]],
    target_label: str,
) -> Optional[Dict[str, Any]]:
    """Возвращает единственную секцию по label (exact -> unique partial)."""
    tl = str(target_label or "").strip().lower()
    if not tl:
        return None
    exact = [
        s for s in sections
        if isinstance(s, dict) and str(s.get("label") or "").strip().lower() == tl
    ]
    if len(exact) == 1:
        return exact[0]
    partial = [
        s for s in sections
        if isinstance(s, dict) and tl in str(s.get("label") or "").strip().lower()
    ]
    if len(partial) == 1:
        return partial[0]
    if len(exact) > 1 or len(partial) > 1:
        logger.warning("SLA label ambiguous: '%s', matches=%d", target_label, len(exact) or len(partial))
    return None


def _best_value_in_section(
    section: Dict[str, Any],
    field: str,
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Возвращает наибольшее значение поля в секции и серию-источник."""
    best: Optional[float] = None
    best_series: Optional[str] = None
    best_method: Optional[str] = None
    for series in (section.get("top_series") or []):
        if not isinstance(series, dict):
            continue
        val = _safe_float(series.get(field))
        if val is not None and (best is None or val > best):
            best = val
            best_series = str(series.get("series") or "")
            best_method = str(series.get("stable_method") or "") if field == "stable_max" else None
    return best, best_series, best_method


def extract_target_rps_from_pack(
    pack: Dict[str, Any],
    target_label: Optional[str] = None,
    allow_peak_fallback: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """Извлекает значение target_rps из lt_framework pack.

    Источник определяется строго по `target_label`.
    При отсутствии `stable_max` может (опционально) использовать `max` из той же секции.
    """
    out = {
        "value": None,
        "method": None,
        "source_label": None,
        "source_series": None,
        "source_stable_method": None,
        "reason": None,
    }

    target = str(target_label or "").strip()
    if not target:
        out["reason"] = "no_query_configured"
        if debug:
            logger.info("[RPS_DEBUG][extract] no_query_configured")
        return out

    sections = pack.get("sections") or []
    if not isinstance(sections, list) or not sections:
        out["reason"] = "no_sections_in_pack"
        if debug:
            logger.info("[RPS_DEBUG][extract] no_sections_in_pack")
        return out

    if debug:
        labels = [str(s.get("label") or "") for s in sections if isinstance(s, dict)]
        logger.info(
            "[RPS_DEBUG][extract] target='%s' allow_peak_fallback=%s sections=%d labels=%s",
            target, bool(allow_peak_fallback), len(labels), labels,
        )

    primary = _find_section_by_label(sections, target)
    if primary is None:
        out["reason"] = "label_not_found_or_ambiguous"
        if debug:
            logger.info("[RPS_DEBUG][extract] label_not_found_or_ambiguous target='%s'", target)
        return out

    source_label = str(primary.get("label") or "")
    out["source_label"] = source_label

    if debug:
        series_dump = []
        for series in (primary.get("top_series") or []):
            if not isinstance(series, dict):
                continue
            series_dump.append(
                {
                    "series": series.get("series"),
                    "max": series.get("max"),
                    "stable_max": series.get("stable_max"),
                    "stable_duration_min": series.get("stable_duration_min"),
                    "stable_method": series.get("stable_method"),
                }
            )
        logger.info("[RPS_DEBUG][extract] primary_label='%s' series=%s", source_label, series_dump)

    stable_val, stable_series, stable_method = _best_value_in_section(primary, "stable_max")
    if stable_val is not None:
        out["value"] = stable_val
        out["source_series"] = stable_series
        out["source_stable_method"] = stable_method
        out["method"] = f"stable_max (query: {source_label})"
        if debug:
            logger.info(
                "[RPS_DEBUG][extract] selected stable_max value=%s label='%s' series='%s' method='%s'",
                stable_val, source_label, stable_series, stable_method,
            )
        return out

    if _safe_bool(allow_peak_fallback, default=True):
        peak_val, peak_series, _ = _best_value_in_section(primary, "max")
        if peak_val is not None:
            out["value"] = peak_val
            out["source_series"] = peak_series
            out["method"] = f"peak_max (query: {source_label}, no stable segments found)"
            out["reason"] = "stable_missing_peak_used"
            if debug:
                logger.info(
                    "[RPS_DEBUG][extract] selected peak_max value=%s label='%s' series='%s'",
                    peak_val, source_label, peak_series,
                )
            return out

    out["reason"] = "stable_missing"
    if debug:
        logger.info("[RPS_DEBUG][extract] stable_missing label='%s'", source_label)
    return out


def _extract_metric_from_pack(
    pack: Dict[str, Any],
    label_keywords: List[str],
    field: str = "max",
) -> Optional[float]:
    """Извлекает числовое значение из top_series по ключевым словам в label."""
    best: Optional[float] = None
    for section in (pack.get("sections") or []):
        label = str(section.get("label") or "").lower()
        if not any(_keyword_matches(label, kw) for kw in label_keywords):
            continue
        for series in (section.get("top_series") or []):
            val = _safe_float(series.get(field))
            if val is not None and (best is None or val > best):
                best = val
    return best


def _make_check(
    name: str,
    threshold: Any,
    actual: Any,
    passed: Optional[bool],
    severity: str = "warning",
    message: str = "",
    category: str = "primary",
) -> Dict[str, Any]:
    return {
        "name": name,
        "threshold": threshold,
        "actual": actual,
        "passed": passed,
        "severity": severity,
        "message": message,
        "category": category,
    }


def _test_mode_from_profile(test_profile: Optional[Dict[str, Any]]) -> str:
    if not isinstance(test_profile, dict):
        return "capacity"
    mode = str(test_profile.get("mode") or "").strip().lower()
    if mode in {"stability", "soak", "endurance"}:
        return "stability"
    return "capacity"


def evaluate_sla(
    domain_data: Dict[str, Dict[str, Any]],
    sla_config: Dict[str, Any],
    test_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Оценивает результаты теста по SLA-критериям."""
    sla_config = sla_config if isinstance(sla_config, dict) else {}
    test_mode = _test_mode_from_profile(test_profile or sla_config.get("test_profile"))
    if not sla_config or not any(
        _safe_float(sla_config.get(k)) is not None
        for k in ("target_rps", "max_error_rate_pct", "max_p95_ms", "max_p99_ms", "max_cpu_pct", "max_memory_pct")
    ):
        return {
            "verdict": "Недостаточно данных",
            "checks": [],
            "summary": "SLA-критерии не заданы",
            "test_mode": test_mode,
        }

    checks: List[Dict[str, Any]] = []
    lt_pack = (domain_data.get("lt_framework") or {}).get("pack") or {}
    hr_pack = (domain_data.get("hard_resources") or {}).get("pack") or {}
    ms_pack = (domain_data.get("microservices") or {}).get("pack") or {}

    target_rps = _safe_float(sla_config.get("target_rps"))
    perf_query = str(sla_config.get("max_performance_query") or "").strip() or None
    if target_rps is not None:
        allow_peak_fallback = _safe_bool(sla_config.get("target_rps_allow_peak_fallback"), default=True)
        debug_peak_logging = _safe_bool(sla_config.get("debug_peak_logging"), default=False)
        rps_pick = extract_target_rps_from_pack(
            lt_pack,
            target_label=perf_query,
            allow_peak_fallback=allow_peak_fallback,
            debug=debug_peak_logging,
        )
        actual_rps = _safe_float(rps_pick.get("value"))
        method = str(rps_pick.get("method") or "unknown")
        source_label = rps_pick.get("source_label")
        source_series = rps_pick.get("source_series")
        reason = str(rps_pick.get("reason") or "")

        if actual_rps is not None:
            passed = actual_rps >= target_rps
            checks.append(_make_check(
                name="target_rps",
                threshold=target_rps,
                actual=round(actual_rps, 2),
                passed=passed,
                severity="critical",
                message=(
                    f"RPS {actual_rps:.1f} ({method}) "
                    f"{'≥' if passed else '<'} целевой {target_rps:.0f}"
                    + (f"; label='{source_label}'" if source_label else "")
                    + (f"; series='{source_series}'" if source_series else "")
                ),
            ))
        else:
            checks.append(_make_check(
                name="target_rps",
                threshold=target_rps,
                actual=None,
                passed=None,
                severity="critical",
                message=f"Нет данных RPS из lt_framework (reason={reason or method})",
            ))

    max_err = _safe_float(sla_config.get("max_error_rate_pct"))
    if max_err is not None:
        err_pct = _extract_metric_from_pack(
            lt_pack, ["error", "ошибк", "fail"], field="max"
        )
        if err_pct is not None:
            passed = err_pct <= max_err
            checks.append(_make_check(
                name="error_rate",
                threshold=max_err,
                actual=round(err_pct, 3),
                passed=passed,
                severity="warning",
                message=f"Error rate {err_pct:.2f}% {'≤' if passed else '>'} порог {max_err}%",
                category="primary",
            ))

    max_p95 = _safe_float(sla_config.get("max_p95_ms"))
    if max_p95 is not None:
        p95_val = _extract_metric_from_pack(
            lt_pack, ["p95", "percentile.*95", "duration.*p95", "95"], field="max"
        )
        if p95_val is None:
            p95_val = _extract_metric_from_pack(
                ms_pack, ["p95", "average request time"], field="max"
            )
            if p95_val is not None:
                p95_val *= 1000
        if p95_val is not None:
            passed = p95_val <= max_p95
            checks.append(_make_check(
                name="p95_latency",
                threshold=max_p95,
                actual=round(p95_val, 2),
                passed=passed,
                severity="warning",
                message=f"P95 latency {p95_val:.1f} ms {'≤' if passed else '>'} порог {max_p95} ms",
                category="primary",
            ))

    max_p99 = _safe_float(sla_config.get("max_p99_ms"))
    if max_p99 is not None:
        p99_val = _extract_metric_from_pack(
            lt_pack, ["p99", "percentile.*99", "duration.*p99", "99"], field="max"
        )
        if p99_val is not None:
            passed = p99_val <= max_p99
            checks.append(_make_check(
                name="p99_latency",
                threshold=max_p99,
                actual=round(p99_val, 2),
                passed=passed,
                severity="warning",
                message=f"P99 latency {p99_val:.1f} ms {'≤' if passed else '>'} порог {max_p99} ms",
                category="primary",
            ))

    max_cpu = _safe_float(sla_config.get("max_cpu_pct"))
    if max_cpu is not None:
        cpu_val = _extract_metric_from_pack(hr_pack, ["cpu"], field="max")
        if cpu_val is not None:
            passed = cpu_val <= max_cpu
            checks.append(_make_check(
                name="cpu_usage",
                threshold=max_cpu,
                actual=round(cpu_val, 2),
                passed=passed,
                severity="warning",
                message=f"CPU usage {cpu_val:.1f}% {'≤' if passed else '>'} порог {max_cpu}%",
                category="secondary",
            ))

    max_mem = _safe_float(sla_config.get("max_memory_pct"))
    if max_mem is not None:
        mem_val = _extract_metric_from_pack(hr_pack, ["memory", "mem"], field="max")
        if mem_val is not None:
            passed = mem_val <= max_mem
            checks.append(_make_check(
                name="memory_usage",
                threshold=max_mem,
                actual=round(mem_val, 2),
                passed=passed,
                severity="warning",
                message=f"Memory usage {mem_val:.1f}% {'≤' if passed else '>'} порог {max_mem}%",
                category="secondary",
            ))

    if not checks:
        return {
            "verdict": "Недостаточно данных",
            "checks": [],
            "summary": "Ни один SLA-критерий не удалось проверить (нет подходящих метрик)",
            "test_mode": test_mode,
        }

    rps_check = next((c for c in checks if c["name"] == "target_rps"), None)
    secondary = [c for c in checks if c["name"] != "target_rps"]

    rps_reached = rps_check["passed"] if rps_check else None
    secondary_failures = [c for c in secondary if c["passed"] is False]
    primary_failures = [
        c for c in checks
        if c["passed"] is False and str(c.get("category") or "primary") == "primary"
    ]
    evaluable = [c for c in checks if c["passed"] is not None]

    if test_mode == "stability" and rps_check is None:
        if primary_failures:
            verdict = "Провал"
        elif secondary_failures:
            verdict = "Есть риски"
        elif evaluable and all(c["passed"] is True for c in evaluable):
            verdict = "Успешно"
        else:
            verdict = "Недостаточно данных"
    elif rps_reached is True:
        if secondary_failures:
            verdict = "Есть риски"
        else:
            verdict = "Успешно"
    elif rps_reached is False:
        verdict = "Провал"
    elif rps_reached is None and rps_check is not None:
        if test_mode == "stability" and primary_failures:
            verdict = "Провал"
        elif test_mode == "stability" and secondary_failures:
            verdict = "Есть риски"
        elif all(c["passed"] is True for c in evaluable):
            verdict = "Есть риски"
        elif any(c["passed"] is False for c in evaluable):
            verdict = "Провал"
        else:
            verdict = "Недостаточно данных"
    else:
        if all(c["passed"] is True for c in evaluable) and evaluable:
            verdict = "Успешно"
        elif any(c["passed"] is False for c in evaluable):
            verdict = "Провал"
        else:
            verdict = "Недостаточно данных"

    passed_names = [c["name"] for c in checks if c["passed"] is True]
    failed_names = [c["name"] for c in checks if c["passed"] is False]
    unknown_names = [c["name"] for c in checks if c["passed"] is None]
    parts = []
    if passed_names:
        parts.append(f"Пройдено: {', '.join(passed_names)}")
    if failed_names:
        parts.append(f"Нарушено: {', '.join(failed_names)}")
    if unknown_names:
        parts.append(f"Нет данных: {', '.join(unknown_names)}")
    if test_mode == "stability":
        parts.append(
            "Режим stability: CPU/memory трактуются как вторичные риски и сами по себе не переводят тест в «Провал»"
        )
    summary = f"SLA verdict: {verdict}. " + "; ".join(parts)

    logger.info(
        "SLA evaluation: verdict=%s, mode=%s, checks=%d, passed=%d, failed=%d",
        verdict, test_mode, len(checks), len(passed_names), len(failed_names)
    )

    return {
        "verdict": verdict,
        "checks": checks,
        "summary": summary,
        "test_mode": test_mode,
    }
