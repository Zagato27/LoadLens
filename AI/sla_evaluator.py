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
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PRIMARY_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("error_rate", "max_error_rate_pct", "error_rate_query", "%"),
    ("p95_latency", "max_p95_ms", "p95_query", "мс"),
    ("p99_latency", "max_p99_ms", "p99_query", "мс"),
)
SECONDARY_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("cpu_usage", "max_cpu_pct", "cpu_query", "%"),
    ("memory_usage", "max_memory_pct", "memory_query", "%"),
)


@dataclass(frozen=True)
class StableWindow:
    """RPS plateau the SLA is evaluated on; ``level`` is its stable_max."""

    start: str
    end: str
    level: float
    series: str
    label: str
    degraded_level: Optional[float] = None
    degraded_checks: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "level": round(self.level, 2),
            "series": self.series,
            "label": self.label,
            "degraded_level": round(self.degraded_level, 2) if self.degraded_level is not None else None,
            "degraded_checks": list(self.degraded_checks),
        }

SLA_CHECK_TITLES = {
    "target_rps": "целевой RPS",
    "error_rate": "доля ошибок",
    "p95_latency": "p95 latency",
    "p99_latency": "p99 latency",
    "cpu_usage": "CPU",
    "memory_usage": "память",
}


def _check_title(name: str) -> str:
    key = str(name or "").strip()
    return SLA_CHECK_TITLES.get(key, key or "критерий")


def _format_check_line(check: Dict[str, Any]) -> str:
    title = _check_title(str(check.get("name") or ""))
    message = str(check.get("message") or "").strip()
    if message:
        return f"- {title}: {message}"
    passed = check.get("passed")
    actual = check.get("actual")
    threshold = check.get("threshold")
    if passed is True:
        return f"- {title}: порог соблюдён ({actual} при лимите {threshold})"
    if passed is False:
        return f"- {title}: порог нарушен ({actual} при лимите {threshold})"
    return f"- {title}: нет данных для проверки"


def format_sla_rationale(verdict: str, checks: List[Dict[str, Any]]) -> str:
    """Human-readable explanation of why the SLA verdict was chosen."""
    rps = next((item for item in checks if item.get("name") == "target_rps"), None)
    failed = [item for item in checks if item.get("passed") is False]
    failed_titles = ", ".join(_check_title(str(item.get("name") or "")) for item in failed)
    if verdict == "Успешно":
        if rps and rps.get("passed") is True and rps.get("actual") is not None:
            head = (
                f"Статус «Успешно»: целевой RPS достигнут "
                f"({rps['actual']} ≥ {rps['threshold']}), вторичные пороги SLA не нарушены."
            )
        else:
            head = "Статус «Успешно»: проверяемые SLA-критерии соблюдены."
    elif verdict == "Есть риски":
        if rps and rps.get("passed") is True and failed:
            head = (
                f"Статус «Есть риски»: целевой RPS достигнут, "
                f"но нарушены вторичные пороги ({failed_titles})."
            )
        elif failed:
            head = (
                f"Статус «Есть риски»: нарушены критерии {failed_titles}, "
                "без однозначного провала по целевому RPS."
            )
        else:
            head = "Статус «Есть риски»: не все SLA-критерии удалось подтвердить числами."
    elif verdict == "Провал":
        if rps and rps.get("passed") is False and rps.get("actual") is not None:
            head = (
                f"Статус «Провал»: целевой RPS не достигнут "
                f"({rps['actual']} < {rps['threshold']})."
            )
        elif failed:
            head = f"Статус «Провал»: нарушены обязательные SLA-критерии ({failed_titles})."
        else:
            head = "Статус «Провал»: по SLA целевой результат теста не подтверждён."
    else:
        head = "Статус «Недостаточно данных»: не хватает метрик, чтобы проверить SLA."
    lines = [_format_check_line(item) for item in checks]
    if not lines:
        return head
    return head + "\n" + "\n".join(lines)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


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


def _segment_end_sort_key(value: Any) -> float:
    try:
        return float(pd.Timestamp(value).value)
    except (TypeError, ValueError):
        return float("-inf")


def _stable_windows_from_rps(pack: Dict[str, Any], query_label: Optional[str]) -> List[StableWindow]:
    """Stable segments of the designated RPS series, latest first.

    Load steps are stretched to the next segment or the end of the test, so they
    still contain the tail after the plateau. Detector segments stop at the drop.
    """
    section = _find_section_by_label(list(pack.get("sections") or []), str(query_label or ""))
    if not isinstance(section, dict):
        return []
    chosen: Optional[Dict[str, Any]] = None
    for series in section.get("top_series") or []:
        if not isinstance(series, dict) or series.get("stable_max") is None:
            continue
        if chosen is None or float(series.get("stable_max") or 0) > float(chosen.get("stable_max") or 0):
            chosen = series
    if chosen is None:
        return []
    label = str(section.get("label") or query_label or "")
    name = str(chosen.get("series") or "")
    windows = [
        StableWindow(start=str(seg["start"]), end=str(seg["end"]), level=float(seg.get("level") or 0.0), series=name, label=label)
        for seg in (chosen.get("step_segments") or [])
        if isinstance(seg, dict) and seg.get("stable") is True and not seg.get("after_drop") and not seg.get("dip")
        and seg.get("start") and seg.get("end")
    ]
    if not windows and chosen.get("stable_window_start") and chosen.get("stable_window_end"):
        windows = [StableWindow(
            start=str(chosen["stable_window_start"]), end=str(chosen["stable_window_end"]),
            level=float(chosen["stable_max"]), series=name, label=label,
        )]
    return sorted(windows, key=lambda w: _segment_end_sort_key(w.end), reverse=True)


def _checks_for_window(
    sla_config: Dict[str, Any],
    specs: tuple[tuple[str, str, str, str], ...],
    labeled: Any,
    window: Optional[StableWindow],
    category: str,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for name, threshold_key, query_key, unit in specs:
        check = _check_by_query_label(
            sla_config, name=name, threshold_key=threshold_key, query_key=query_key,
            labeled=labeled, window=window, unit=unit, category=category,
        )
        if check is not None:
            checks.append(check)
    return checks


def _peak_column(frame: Optional[pd.DataFrame]) -> Optional[str]:
    if frame is None or frame.empty:
        return None
    best_name: Optional[str] = None
    best_max: Optional[float] = None
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        current = float(values.max())
        if best_max is None or current > best_max:
            best_name, best_max = str(column), current
    return best_name


def _mean_between(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    chunk = series[(series.index >= start) & (series.index <= end)].dropna()
    if chunk.empty:
        return None
    return float(chunk.mean())


def _windows_from_load_steps(
    pack: Dict[str, Any],
    query_label: Optional[str],
    labeled: Any,
) -> List[StableWindow]:
    """Report steps, latest first. Their RPS is what the step table shows.

    A recovered RPS dip is skipped: it is not a load level the system was tested at.
    """
    raw = [
        item for item in (pack.get("load_steps") or [])
        if isinstance(item, dict) and item.get("start_iso") and item.get("end_iso") and not item.get("dip")
    ]
    if not raw:
        return []
    label = str(query_label or "")
    frame = _labeled_frame(labeled, label)
    column = _peak_column(frame)
    series = None
    if frame is not None and column is not None:
        values = pd.to_numeric(frame[column], errors="coerce")
        index = pd.DatetimeIndex(frame.index)
        values.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        series = values
    windows: List[StableWindow] = []
    for step in raw:
        level = _safe_float(step.get("rps_level"))
        start = _as_utc(step.get("start_iso"))
        end = _as_utc(step.get("end_iso"))
        if level is None and series is not None and start is not None and end is not None:
            level = _mean_between(series, start, end)
        if level is None or start is None or end is None:
            continue
        windows.append(StableWindow(
            start=str(step["start_iso"]),
            end=str(step["end_iso"]),
            level=level,
            series=column or str(step.get("series") or ""),
            label=label,
        ))
    return sorted(windows, key=lambda item: _segment_end_sort_key(item.end), reverse=True)


def _select_sla_window(
    sla_config: Dict[str, Any],
    lt_pack: Dict[str, Any],
    lt_labeled: Any,
) -> tuple[Optional[StableWindow], List[Dict[str, Any]]]:
    """Latest report step on which latency and error SLA still hold, with its checks.

    The step table is the source of the steps. A later flat RPS stretch can
    continue through a latency collapse, so the search walks from the last step
    backwards and stops at the last one that still passes. Detector segments are
    used only when the report has no load steps. When every step breaks the SLA
    the latest one is kept and fails.
    """
    query = sla_config.get("max_performance_query")
    candidates = _windows_from_load_steps(lt_pack, query, lt_labeled)
    if not candidates:
        candidates = _stable_windows_from_rps(lt_pack, query)
    if not candidates:
        return None, _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, None, "primary")
    latest_checks: List[Dict[str, Any]] = []
    degraded: Optional[StableWindow] = None
    degraded_names: tuple[str, ...] = ()
    for window in candidates:
        checks = _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, window, "primary")
        if window is candidates[0]:
            latest_checks = checks
        failed = tuple(str(c["name"]) for c in checks if c.get("passed") is False)
        if not failed:
            if degraded is None:
                return window, checks
            return replace(window, degraded_level=degraded.level, degraded_checks=degraded_names), checks
        degraded, degraded_names = window, failed
    return candidates[0], latest_checks


def _as_utc(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value, unit="s", tz="UTC") if isinstance(value, (int, float)) else pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _bin_width(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(0)
    deltas = pd.Series(index).diff().dropna()
    if deltas.empty:
        return pd.Timedelta(0)
    width = deltas.median()
    if pd.isna(width) or width <= pd.Timedelta(0):
        return pd.Timedelta(0)
    return width


def _worst_series_p95(df: pd.DataFrame, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> tuple[Optional[float], Optional[str]]:
    """Highest 95th percentile among series, optionally inside the stable window.

    A resampled point is stamped at the start of its bucket but already averages
    the whole bucket. Buckets that begin inside the window and end after it are
    dropped, otherwise the spike after the plateau leaks into the percentile.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    work = df.copy()
    idx = pd.DatetimeIndex(work.index)
    work.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if start is not None and end is not None:
        width = _bin_width(work.index)
        bin_end = work.index + width
        work = work[(work.index >= start) & (bin_end <= end)]
    best: Optional[float] = None
    best_name: Optional[str] = None
    for col in work.columns:
        values = pd.to_numeric(work[col], errors="coerce").dropna()
        if values.empty:
            continue
        current = float(values.quantile(0.95))
        if best is None or current > best:
            best, best_name = current, str(col)
    return best, best_name


def _labeled_frame(labeled: Any, query: str) -> Optional[pd.DataFrame]:
    if not isinstance(labeled, list):
        return None
    section = _find_section_by_label([item for item in labeled if isinstance(item, dict)], query)
    df = section.get("df") if isinstance(section, dict) else None
    return df if isinstance(df, pd.DataFrame) else None


def _check_by_query_label(
    sla_config: Dict[str, Any],
    *,
    name: str,
    threshold_key: str,
    query_key: str,
    labeled: Any,
    window: Optional[StableWindow],
    unit: str,
    category: str,
) -> Optional[Dict[str, Any]]:
    """Checks one SLA threshold against the 95th percentile of a named query.

    On a step test the window is a stable RPS segment closed before the drop.
    Soak tests and runs without such a segment use the whole test window. A
    resampled bucket that runs past the segment end is excluded. Among several
    series the worst percentile is compared with the threshold.
    """
    threshold = _safe_float(sla_config.get(threshold_key))
    if threshold is None:
        return None
    query = str(sla_config.get(query_key) or "").strip()
    if not query:
        return _make_check(
            name=name,
            threshold=threshold,
            actual=None,
            passed=None,
            message="Источник не задан: выберите запрос в настройках SLA",
            category=category,
        )
    frame = _labeled_frame(labeled, query)
    if frame is None:
        return _make_check(
            name=name,
            threshold=threshold,
            actual=None,
            passed=None,
            message=f"Нет данных по запросу «{query}»",
            category=category,
        )
    start = _as_utc(window.start) if window else None
    end = _as_utc(window.end) if window else None
    scope = "всего окна"
    actual, series = (None, None)
    if window is not None and start is not None and end is not None and end > start:
        actual, series = _worst_series_p95(frame, start, end)
        scope = f"стабильной ступени ≈{window.level:.0f} RPS"
    elif window is None:
        actual, series = _worst_series_p95(frame)
    if actual is None:
        empty_message = (
            f"На стабильной ступени RPS нет точек запроса «{query}»"
            if window is not None
            else f"В запросе «{query}» нет чисел"
        )
        return _make_check(
            name=name,
            threshold=threshold,
            actual=None,
            passed=None,
            message=empty_message,
            category=category,
        )
    passed = actual <= threshold
    series_note = f", серия «{series}»" if series else ""
    return _make_check(
        name=name,
        threshold=threshold,
        actual=round(actual, 3),
        passed=passed,
        message=(
            f"p95 {actual:.2f} {unit} {'≤' if passed else '>'} порог {threshold:g} {unit} "
            f"({scope}, запрос «{query}»{series_note})"
        ),
        category=category,
    )


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
    lt_payload = domain_data.get("lt_framework") or {}
    hr_payload = domain_data.get("hard_resources") or {}
    lt_pack = lt_payload.get("pack") or {}
    lt_labeled = lt_payload.get("labeled")
    perf_query = str(sla_config.get("max_performance_query") or "").strip() or None
    if test_mode == "stability":
        window, primary_checks = None, _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, None, "primary")
    else:
        window, primary_checks = _select_sla_window(sla_config, lt_pack, lt_labeled)

    target_rps = _safe_float(sla_config.get("target_rps"))
    if target_rps is not None and window is not None:
        passed = window.level >= target_rps
        degraded_note = ""
        if window.degraded_level is not None:
            degraded_note = (
                f"; на ступени ≈{window.degraded_level:.0f} RPS нарушены: "
                f"{', '.join(window.degraded_checks)}"
            )
        checks.append(_make_check(
            name="target_rps",
            threshold=target_rps,
            actual=round(window.level, 2),
            passed=passed,
            severity="critical",
            message=(
                f"RPS {window.level:.1f} (stable_max (query: {window.label})) "
                f"{'≥' if passed else '<'} целевой {target_rps:.0f}"
                f"; label='{window.label}'; series='{window.series}'{degraded_note}"
            ),
        ))
    elif target_rps is not None:
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

    checks.extend(primary_checks)
    checks.extend(_checks_for_window(sla_config, SECONDARY_SPECS, hr_payload.get("labeled"), window, "secondary"))

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
        "stable_window": window.to_dict() if window is not None else None,
    }
