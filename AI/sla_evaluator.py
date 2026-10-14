"""Программная оценка теста по SLA-критериям.

Окно проверки и правила вердикта зависят от режима профиля теста (AI/test_profile.py):

- capacity / stress (step, stress): окно — последняя ступень, где пороги latency/errors
  выполнены. target_rps — главный критерий: если достигнут (stable_max >= target_rps),
  тест успешен ДАЖЕ если система деградировала после этого; нарушения остальных порогов
  на этой ступени дают «Есть риски».
- stability (soak): окно — окно удержания нагрузки без разгона (``lt_pack["hold_window"]``,
  иначе весь тест). target_rps — уровень, который нужно удерживать: средний RPS окна не ниже
  target на 5 %, ни один интервал не ниже на 15 %. Любое нарушение primary-порога → «Провал»,
  CPU/memory → «Есть риски».
- spike: окна — найденные всплески (``lt_pack["spike_profile"]``), по каждому порогу берётся
  худший всплеск. target_rps сравнивается с уровнем всплеска; отдельный критерий recovery —
  возврат задержек к фону. target/recovery нарушены → «Провал», latency/errors в окне
  всплеска или CPU/memory → «Есть риски». Без всплеска → «Недостаточно данных».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import pandas as pd

from AI.stability import hold_check
from AI.test_profile import MODES, mode_of

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
    """Window the SLA is evaluated on: a load step (``kind="step"``), the soak hold window
    (``"hold"``) or a spike (``"spike"``); ``level`` is its RPS."""

    start: str
    end: str
    level: float
    series: str
    label: str
    degraded_level: Optional[float] = None
    degraded_checks: tuple[str, ...] = ()
    kind: str = "step"
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "level": round(self.level, 2),
            "series": self.series,
            "label": self.label,
            "degraded_level": round(self.degraded_level, 2) if self.degraded_level is not None else None,
            "degraded_checks": list(self.degraded_checks),
        }
        if self.kind != "step":
            out["kind"] = self.kind
            out["title"] = self.title
        return out


def _scope_text(window: Optional[StableWindow]) -> str:
    if window is None:
        return "всего окна"
    if window.kind == "hold":
        return f"окна удержания ≈{window.level:.0f} RPS"
    if window.kind == "spike":
        name = f" «{window.title}»" if window.title else ""
        return f"окна всплеска{name} ≈{window.level:.0f} RPS"
    return f"стабильной ступени ≈{window.level:.0f} RPS"


def _empty_scope_text(window: StableWindow) -> str:
    if window.kind == "hold":
        return "В окне удержания нет точек запроса"
    if window.kind == "spike":
        return "В окне всплеска нет точек запроса"
    return "На стабильной ступени RPS нет точек запроса"

SLA_CHECK_TITLES = {
    "target_rps": "целевой RPS",
    "error_rate": "доля ошибок",
    "p95_latency": "p95 latency",
    "p99_latency": "p99 latency",
    "cpu_usage": "CPU",
    "memory_usage": "память",
    "recovery": "восстановление после всплеска",
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
    scope = _scope_text(window)
    actual, series = (None, None)
    if window is not None and start is not None and end is not None and end > start:
        actual, series = _worst_series_p95(frame, start, end)
    elif window is None:
        actual, series = _worst_series_p95(frame)
    if actual is None:
        empty_message = (
            f"{_empty_scope_text(window)} «{query}»"
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
    """Profile mode: capacity | stability | spike | stress (old reports: capacity | stability)."""
    if not isinstance(test_profile, dict):
        return "capacity"
    mode = mode_of(test_profile)
    return mode if mode in MODES else "capacity"


def _capacity_target_check(
    sla_config: Dict[str, Any],
    target_rps: float,
    window: Optional[StableWindow],
    lt_pack: Dict[str, Any],
    perf_query: Optional[str],
) -> Dict[str, Any]:
    """target_rps on the SLA step; without steps — stable_max or the peak of the designated query."""
    if window is not None:
        passed = window.level >= target_rps
        degraded_note = ""
        if window.degraded_level is not None:
            degraded_note = (
                f"; на ступени ≈{window.degraded_level:.0f} RPS нарушены: "
                f"{', '.join(window.degraded_checks)}"
            )
        return _make_check(
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
        )
    rps_pick = extract_target_rps_from_pack(
        lt_pack,
        target_label=perf_query,
        allow_peak_fallback=_safe_bool(sla_config.get("target_rps_allow_peak_fallback"), default=True),
        debug=_safe_bool(sla_config.get("debug_peak_logging"), default=False),
    )
    actual_rps = _safe_float(rps_pick.get("value"))
    method = str(rps_pick.get("method") or "unknown")
    if actual_rps is None:
        reason = str(rps_pick.get("reason") or "")
        return _make_check(
            name="target_rps", threshold=target_rps, actual=None, passed=None, severity="critical",
            message=f"Нет данных RPS из lt_framework (reason={reason or method})",
        )
    passed = actual_rps >= target_rps
    source_label = rps_pick.get("source_label")
    source_series = rps_pick.get("source_series")
    return _make_check(
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
    )


def _hold_window(lt_pack: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[StableWindow]]:
    hold = lt_pack.get("hold_window") if isinstance(lt_pack, dict) else None
    if not isinstance(hold, dict) or not hold.get("start_utc") or not hold.get("end_utc"):
        return None, None
    level = _safe_float(hold.get("actual_rps_mean"))
    if level is None:
        level = _safe_float(hold.get("level_rps")) or 0.0
    window = StableWindow(
        start=str(hold["start_utc"]), end=str(hold["end_utc"]), level=level,
        series=str(hold.get("series") or ""), label=str(hold.get("source_label") or ""),
        kind="hold", title="окно удержания",
    )
    return hold, window


def _stability_checks(
    sla_config: Dict[str, Any],
    target_rps: Optional[float],
    lt_pack: Dict[str, Any],
    lt_labeled: Any,
    hr_labeled: Any,
) -> tuple[List[Dict[str, Any]], Optional[StableWindow]]:
    hold, window = _hold_window(lt_pack)
    checks: List[Dict[str, Any]] = []
    if target_rps is not None:
        if hold is None:
            query = str(sla_config.get("max_performance_query") or "").strip()
            checks.append(_make_check(
                name="target_rps", threshold=target_rps, actual=None, passed=None, severity="critical",
                message=f"Нет данных RPS для окна удержания (запрос «{query or 'не выбран'}»)",
            ))
        else:
            result = hold_check(hold, target_rps)
            checks.append(_make_check(
                name="target_rps", threshold=target_rps, actual=result["actual"], passed=result["passed"],
                severity="critical", message=result["message"],
            ))
    checks.extend(_checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, window, "primary"))
    checks.extend(_checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, window, "secondary"))
    return checks, window


def _worst_check(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The check of the worst window: a failed one with the highest value, else the highest value."""
    if not candidates:
        return None
    failed = [c for c in candidates if c.get("passed") is False and c.get("actual") is not None]
    if failed:
        return max(failed, key=lambda c: float(c["actual"]))
    measured = [c for c in candidates if c.get("actual") is not None]
    if measured:
        return max(measured, key=lambda c: float(c["actual"]))
    return candidates[0]


def _checks_over_windows(
    sla_config: Dict[str, Any],
    specs: tuple[tuple[str, str, str, str], ...],
    labeled: Any,
    windows: List[StableWindow],
    category: str,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for spec in specs:
        per_window: List[Dict[str, Any]] = []
        for window in windows:
            per_window.extend(_checks_for_window(sla_config, (spec,), labeled, window, category))
        worst = _worst_check(per_window)
        if worst is not None:
            checks.append(worst)
    return checks


def _spike_windows(lt_pack: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[StableWindow]]:
    profile = lt_pack.get("spike_profile") if isinstance(lt_pack, dict) else None
    profile = profile if isinstance(profile, dict) else {}
    spikes = [
        item for item in (profile.get("spikes") or [])
        if isinstance(item, dict) and item.get("start_utc") and item.get("end_utc")
    ]
    windows = [
        StableWindow(
            start=str(item["start_utc"]), end=str(item["end_utc"]), level=float(item.get("level_rps") or 0.0),
            series=str(profile.get("series") or ""), label=str(profile.get("source_label") or ""),
            kind="spike", title=str(item.get("label") or f"всплеск {item.get('index')}"),
        )
        for item in spikes
    ]
    return profile, spikes, windows


def _recovery_check(profile: Dict[str, Any], spikes: List[Dict[str, Any]], latency_configured: bool) -> Optional[Dict[str, Any]]:
    limit = _safe_float(profile.get("recovery_max_window_min"))
    if not profile.get("detected") or not spikes:
        if not profile:
            return None
        return _make_check(
            name="recovery", threshold=limit, actual=None, passed=None, severity="critical",
            message="Всплеск нагрузки не обнаружен: восстановление не оценивалось",
        )
    states = [item.get("recovered") for item in spikes]
    if all(state is None for state in states) and not latency_configured:
        return None
    times = [float(item["recovery_time_sec"]) for item in spikes if item.get("recovery_time_sec") is not None]
    actual = round(max(times) / 60.0, 2) if times else None
    tolerance = _safe_float(profile.get("recovery_tolerance_pct"))
    not_recovered = [str(item.get("label")) for item in spikes if item.get("recovered") is False]
    if not_recovered:
        passed: Optional[bool] = False
        message = (
            f"Задержки не вернулись к фону (±{tolerance:g} %) за {limit:g} мин после: {', '.join(not_recovered)}"
            if tolerance is not None and limit is not None
            else f"Задержки не вернулись к фону после: {', '.join(not_recovered)}"
        )
    elif all(state is True for state in states):
        passed = True
        message = f"Задержки вернулись к фону за {actual:g} мин (худший всплеск), предел {limit:g} мин"
    else:
        passed = None
        message = "Восстановление оценено не для всех всплесков: после всплеска нет точек задержки"
    return _make_check(
        name="recovery", threshold=limit, actual=actual, passed=passed, severity="critical", message=message,
    )


def _spike_checks(
    sla_config: Dict[str, Any],
    target_rps: Optional[float],
    lt_pack: Dict[str, Any],
    lt_labeled: Any,
    hr_labeled: Any,
) -> tuple[List[Dict[str, Any]], Optional[StableWindow], bool]:
    profile, spikes, windows = _spike_windows(lt_pack)
    detected = bool(profile.get("detected")) and bool(windows)
    checks: List[Dict[str, Any]] = []
    top = max(windows, key=lambda w: w.level) if windows else None
    if target_rps is not None:
        if top is None:
            checks.append(_make_check(
                name="target_rps", threshold=target_rps, actual=None, passed=None, severity="critical",
                message=str(profile.get("reason") or "Всплеск нагрузки не обнаружен"),
            ))
        else:
            passed = top.level >= target_rps
            checks.append(_make_check(
                name="target_rps", threshold=target_rps, actual=round(top.level, 2), passed=passed,
                severity="critical",
                message=(
                    f"Уровень всплеска «{top.title}» {top.level:.1f} RPS "
                    f"{'≥' if passed else '<'} целевой {target_rps:.0f}; label='{top.label}'"
                ),
            ))
    if windows:
        checks.extend(_checks_over_windows(sla_config, PRIMARY_SPECS, lt_labeled, windows, "primary"))
        checks.extend(_checks_over_windows(sla_config, SECONDARY_SPECS, hr_labeled, windows, "secondary"))
    else:
        checks.extend(_checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, None, "primary"))
        checks.extend(_checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, None, "secondary"))
    latency_configured = bool(str(sla_config.get("p95_query") or sla_config.get("p99_query") or "").strip())
    recovery = _recovery_check(profile, spikes, latency_configured)
    if recovery is not None:
        checks.append(recovery)
    return checks, top, detected


def _capacity_verdict(checks: List[Dict[str, Any]]) -> str:
    rps_check = next((c for c in checks if c["name"] == "target_rps"), None)
    others = [c for c in checks if c["name"] != "target_rps"]
    evaluable = [c for c in checks if c["passed"] is not None]
    rps_reached = rps_check["passed"] if rps_check else None
    if rps_reached is True:
        return "Есть риски" if any(c["passed"] is False for c in others) else "Успешно"
    if rps_reached is False:
        return "Провал"
    if rps_check is not None:
        if all(c["passed"] is True for c in evaluable):
            return "Есть риски"
        if any(c["passed"] is False for c in evaluable):
            return "Провал"
        return "Недостаточно данных"
    if evaluable and all(c["passed"] is True for c in evaluable):
        return "Успешно"
    if any(c["passed"] is False for c in evaluable):
        return "Провал"
    return "Недостаточно данных"


def _is_primary(check: Dict[str, Any]) -> bool:
    return str(check.get("category") or "primary") == "primary"


def _verdict_for_mode(mode: str, checks: List[Dict[str, Any]], detected: bool = True) -> str:
    """Deterministic verdict by the policy of the test-profile mode (see module docstring)."""
    if mode in {"capacity", "stress"}:
        return _capacity_verdict(checks)
    evaluable = [c for c in checks if c["passed"] is not None]
    failed = [c for c in checks if c["passed"] is False]
    all_passed = bool(evaluable) and all(c["passed"] is True for c in evaluable)
    if mode == "spike":
        if not detected:
            return "Недостаточно данных"
        if any(c["name"] in {"target_rps", "recovery"} for c in failed):
            return "Провал"
        if failed:
            return "Есть риски"
        return "Успешно" if all_passed else "Недостаточно данных"
    if any(_is_primary(c) for c in failed):
        return "Провал"
    if failed:
        return "Есть риски"
    return "Успешно" if all_passed else "Недостаточно данных"


MODE_SUMMARY_NOTES = {
    "stability": (
        "Режим stability: пороги проверены на окне удержания, target_rps — удержание среднего уровня; "
        "CPU/memory трактуются как вторичные риски и сами по себе не переводят тест в «Провал»"
    ),
    "spike": (
        "Режим spike: пороги проверены в окнах всплесков (худший всплеск), target_rps — уровень всплеска; "
        "нарушение target_rps или recovery — «Провал», задержки и ошибки в окне всплеска — риски"
    ),
}


def evaluate_sla(
    domain_data: Dict[str, Dict[str, Any]],
    sla_config: Dict[str, Any],
    test_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Оценивает результаты теста по SLA-критериям с учётом режима профиля теста."""
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

    lt_payload = domain_data.get("lt_framework") or {}
    hr_payload = domain_data.get("hard_resources") or {}
    lt_pack = lt_payload.get("pack") or {}
    lt_labeled = lt_payload.get("labeled")
    hr_labeled = hr_payload.get("labeled")
    perf_query = str(sla_config.get("max_performance_query") or "").strip() or None
    target_rps = _safe_float(sla_config.get("target_rps"))
    detected = True

    if test_mode == "stability":
        checks, window = _stability_checks(sla_config, target_rps, lt_pack, lt_labeled, hr_labeled)
    elif test_mode == "spike":
        checks, window, detected = _spike_checks(sla_config, target_rps, lt_pack, lt_labeled, hr_labeled)
    else:
        window, primary_checks = _select_sla_window(sla_config, lt_pack, lt_labeled)
        checks = []
        if target_rps is not None:
            checks.append(_capacity_target_check(sla_config, target_rps, window, lt_pack, perf_query))
        checks.extend(primary_checks)
        checks.extend(_checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, window, "secondary"))

    if not checks:
        return {
            "verdict": "Недостаточно данных",
            "checks": [],
            "summary": "Ни один SLA-критерий не удалось проверить (нет подходящих метрик)",
            "test_mode": test_mode,
        }

    verdict = _verdict_for_mode(test_mode, checks, detected)

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
    if test_mode in MODE_SUMMARY_NOTES:
        parts.append(MODE_SUMMARY_NOTES[test_mode])
    if test_mode == "spike" and not detected:
        parts.append("всплеск нагрузки не обнаружен")
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
