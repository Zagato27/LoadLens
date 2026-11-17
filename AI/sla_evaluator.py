# SPDX-License-Identifier: Apache-2.0
"""Программная оценка теста по SLA-критериям.

Окно проверки и правила вердикта зависят от режима профиля теста (AI/test_profile.py):

- capacity / stress (step, stress): two windows. The ceiling (``stable_window``) is the last
  step on which the latency/error thresholds hold: the maximum performance, checked with the
  primary thresholds. The target step (``target_window``) is the first step at ``target_rps``
  or above (``TARGET_STEP_TOLERANCE_PCT`` below it still counts) on which they hold; CPU/memory
  are checked on it, so steps above the target are headroom and a run that goes further never
  gets a worse verdict. No such step: ``fail``. CPU/memory over the threshold on the target
  step: ``risk``. With ``target_margin_pct`` the CPU/memory step is the first one at
  ``target × (1 + margin)`` and the ``target_margin`` check fails (``risk``) without it.
  Without ``target_rps``: primary violations on the ceiling ``fail``, CPU/memory ``risk``.
- stability (soak): окно — окно удержания нагрузки без разгона (``lt_pack["hold_window"]``,
  иначе весь тест). target_rps — уровень, который нужно удерживать: средний RPS окна не ниже
  target на 5 %, ни один интервал не ниже на 15 %. Любое нарушение primary-порога → ``fail``,
  CPU/memory → ``risk``.
- spike: окна — найденные всплески (``lt_pack["spike_profile"]``), по каждому порогу берётся
  худший всплеск. target_rps сравнивается с уровнем всплеска; отдельный критерий recovery —
  возврат задержек к фону. target/recovery нарушены → ``fail``, latency/errors в окне
  всплеска или CPU/memory → ``risk``. Без всплеска → ``no_data``.
- phases (scenario): the windows are the phases of the run (``lt_pack["phase_profile"]``, AI/phases.py), each
  threshold takes the worst phase; ``target_rps`` does not apply. Against the phases of the previous scenario run
  (``lt_pack["phase_reference"]``) two secondary checks: ``phase_completeness`` (every earlier phase ran) and
  ``phase_duration`` (the largest significant change of a phase duration within the tolerance). Latency/errors
  ``fail``, CPU/memory and the phase checks ``risk``. Without phases the thresholds are checked on the whole window.

Regression against the baseline run (``apply_regression``, AI/regression.py) comes after the mode verdict and never
changes it for a report without ``regression_*`` thresholds or without a comparable baseline: the ``regression_*``
checks (``category = regression``) compare p95, errors and the load reached with the baseline at a comparable load.
A breached one with ``severity = warning`` turns ``pass`` into ``risk``, a ``critical`` one turns ``pass`` and
``risk`` into ``fail``; ``fail`` and ``no_data`` stay.

The verdict is a code (AI/verdicts.py). The summary and the rationale are narrative written in the language
of the report (``format_sla_summary``, ``format_sla_rationale``); the summary is rebuilt from its parts
(``summary_parts``) when the pipeline adjusts the verdict, so its text is never patched. Check messages, window
titles and the summary use the ``sla.*`` dictionary keys in the language of the run (``evaluate_sla(lang=...)``,
else the current locale). Messages keep the ``запрос «…»`` / ``query «…»`` and ``серия «…»`` / ``series «…»``
structure: readers of reports saved before ``query_label``/``series`` parse it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from AI.i18n import get_locale, normalize_lang, t, with_locale
from AI.regression import (
    CATEGORY_REGRESSION,
    REASON_DISABLED,
    REASON_MODE_UNSUPPORTED,
    REASON_NO_BASELINE,
    REASON_NO_BASELINE_SLA,
    REASON_NO_CHECKS,
    REASON_NO_VALUE,
    REASON_OTHER_SERVICE,
    REASON_OTHER_TYPE,
    REASONS as REGRESSION_REASONS,
    REGRESSION_CHECK_NAMES,
    SEVERITY_CRITICAL,
    compare_stored,
    metric_limits as regression_metric_limits,
    mode_of_test_type,
    regression_checks,
    scope_text as regression_scope_text,
    signed_change,
    summary_items as regression_summary_items,
    thresholds_enabled,
    thresholds_from_sla,
    unit_text as regression_unit_text,
)
from AI.report_texts import LocaleTable, text_lang
from AI.stability import hold_check
from AI.test_profile import MODES, mode_of, normalize_test_type
from AI.verdicts import FAIL, NO_DATA, PASS, RISK, label as verdict_label, to_code

logger = logging.getLogger(__name__)

PRIMARY_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("error_rate", "max_error_rate_pct", "error_rate_query", "%"),
    ("p95_latency", "max_p95_ms", "p95_query", "ms"),
    ("p99_latency", "max_p99_ms", "p99_query", "ms"),
)
SECONDARY_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("cpu_usage", "max_cpu_pct", "cpu_query", "%"),
    ("memory_usage", "max_memory_pct", "memory_query", "%"),
)
# A step measured this far below the target (or the margin level) still counts as reaching it:
# a "100 RPS" step measured at 99.6 RPS is the target step.
TARGET_STEP_TOLERANCE_PCT = 2.0
# Roles of a step window: the step the target (or the margin over it) was confirmed on.
ROLE_TARGET = "target"
ROLE_MARGIN = "margin"


@dataclass(frozen=True)
class StableWindow:
    """Window the SLA is evaluated on: a load step (``kind="step"``), the soak hold window
    (``"hold"``), a spike (``"spike"``) or a scenario phase (``"phase"``); ``level`` is its RPS. ``role`` marks the target step
    (``"target"``) or the margin step (``"margin"``, with ``margin_pct``) of a step test. ``after_drop``
    marks a report step after an RPS drop: evidence of degradation, never a ceiling or a target step."""

    start: str
    end: str
    level: float
    series: str
    label: str
    degraded_level: Optional[float] = None
    degraded_checks: tuple[str, ...] = ()
    kind: str = "step"
    title: str = ""
    role: str = ""
    margin_pct: Optional[float] = None
    after_drop: bool = False

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
        if self.role:
            out["role"] = self.role
        if self.margin_pct is not None:
            out["margin_pct"] = self.margin_pct
        return out


def _unit_text(unit: str) -> str:
    return t("report.unit.ms") if unit == "ms" else unit


def _scope_text(window: Optional[StableWindow]) -> str:
    if window is None:
        return t("sla.scope.whole")
    level = f"{window.level:.0f}"
    if window.kind == "hold":
        return t("sla.scope.hold", level=level)
    if window.kind == "spike":
        if window.title:
            return t("sla.scope.spike_named", title=window.title, level=level)
        return t("sla.scope.spike", level=level)
    if window.kind == "phase":
        return t("sla.scope.phase", title=window.title, level=level)
    if window.role == ROLE_TARGET:
        return t("sla.scope.target_step", level=level)
    if window.role == ROLE_MARGIN:
        return t("sla.scope.margin_step", level=level, margin=f"{window.margin_pct or 0:g}")
    return t("sla.scope.step", level=level)


def _empty_scope_text(window: StableWindow, query: str) -> str:
    if window.kind == "hold":
        return t("sla.check.empty_hold", query=query)
    if window.kind == "spike":
        return t("sla.check.empty_spike", query=query)
    if window.kind == "phase":
        return t("sla.check.empty_phase", query=query)
    return t("sla.check.empty_step", query=query)


SLA_CHECK_NAMES = (
    "target_rps", "target_margin", "error_rate", "p95_latency", "p99_latency", "cpu_usage", "memory_usage", "recovery",
    "phase_completeness", "phase_duration", *REGRESSION_CHECK_NAMES,
)
# Names of the checks (sla.check_title.*) in the language of the report being built.
SLA_CHECK_TITLES = LocaleTable(SLA_CHECK_NAMES, lambda name: t(f"sla.check_title.{name}"))


def _check_title(name: str, lang: Optional[str] = None) -> str:
    key = str(name or "").strip()
    if key in SLA_CHECK_NAMES:
        return t(f"sla.check_title.{key}", locale=lang)
    return key or t("sla.check_title_fallback", locale=lang)


def _format_check_line(check: Dict[str, Any], lang: Optional[str] = None) -> str:
    title = _check_title(str(check.get("name") or ""), lang)
    message = str(check.get("message") or "").strip()
    if message:
        return f"- {title}: {message}"
    passed = check.get("passed")
    actual = str(check.get("actual"))
    threshold = str(check.get("threshold"))
    if passed is True:
        return t("sla.line.passed", locale=lang, title=title, actual=actual, threshold=threshold)
    if passed is False:
        return t("sla.line.failed", locale=lang, title=title, actual=actual, threshold=threshold)
    return t("sla.line.no_data", locale=lang, title=title)


# Codes of the adjustments a pipeline makes to the SLA verdict (``summary_parts["adjustments"]``).
ADJUST_RESOURCES = "resources_are_risks"
ADJUST_TREND = "adverse_trend"
# Parts of the summary by mode: modes with a note of their own and the names of the summary lists.
_MODE_NOTES = ("stability", "spike", "phases")
_SUMMARY_LISTS = ("passed", "failed", "unknown")
# Fields of ``headroom`` the summary is rebuilt from.
_HEADROOM_SUMMARY_KEYS = ("role", "verdict_level", "margin_pct", "ceiling_level", "ratio", "above_target_failed")


def _lang_of(lang: Any) -> str:
    """Language of a narrative: an explicit ``lang`` (unknown codes: ru), else the current locale."""
    return text_lang(lang) if lang else get_locale()


def format_sla_rationale(verdict: str, checks: List[Dict[str, Any]], lang: Optional[str] = None) -> str:
    """Human-readable explanation of why the SLA verdict (a code or an old label) was chosen, in ``lang``
    (default: the current locale)."""
    lang = _lang_of(lang)
    code = to_code(verdict)
    title = verdict_label(code, lang)
    rps = next((item for item in checks if item.get("name") == "target_rps"), None)
    failed = [item for item in checks if item.get("passed") is False]
    failed_titles = ", ".join(_check_title(str(item.get("name") or ""), lang) for item in failed)

    def head(key: str, **params: Any) -> str:
        return t(f"sla.rationale.{key}", locale=lang, label=title, **params)

    if code == PASS:
        if rps and rps.get("passed") is True and rps.get("actual") is not None:
            # Within the tolerance of the target step the RPS may be a little under the target.
            below = _safe_float(rps["actual"]) is not None and _safe_float(rps.get("threshold")) is not None and (
                float(rps["actual"]) < float(rps["threshold"])
            )
            text = head("pass_target", actual=str(rps["actual"]), op="≈" if below else "≥", threshold=str(rps["threshold"]))
        else:
            text = head("pass")
    elif code == RISK:
        if rps and rps.get("passed") is True and failed:
            text = head("risk_target", failed=failed_titles)
        elif failed:
            text = head("risk_failed", failed=failed_titles)
        else:
            text = head("risk")
    elif code == FAIL:
        if rps and rps.get("passed") is False and rps.get("actual") is not None:
            text = head("fail_target", actual=str(rps["actual"]), op="<", threshold=str(rps["threshold"]))
        elif failed:
            text = head("fail_failed", failed=failed_titles)
        else:
            text = head("fail")
    else:
        text = head("no_data")
    lines = [_format_check_line(item, lang) for item in checks]
    if not lines:
        return text
    return text + "\n" + "\n".join(lines)


def sla_summary_parts(
    verdict: Any,
    checks: List[Dict[str, Any]],
    test_mode: str,
    detected: bool = True,
    headroom: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Parts of the SLA summary: the verdict code, check names by state, the mode, the adjustments and, for a
    step test with a target step, its ``headroom`` (target step, ceiling, breaches above the target)."""
    parts: Dict[str, Any] = {
        "verdict": to_code(verdict),
        "passed": [str(c.get("name")) for c in checks if c.get("passed") is True],
        "failed": [str(c.get("name")) for c in checks if c.get("passed") is False],
        "unknown": [str(c.get("name")) for c in checks if c.get("passed") is None],
        "test_mode": str(test_mode or ""),
        "spike_detected": bool(detected),
        "adjustments": [],
    }
    if isinstance(headroom, Mapping):
        parts["headroom"] = {key: headroom.get(key) for key in _HEADROOM_SUMMARY_KEYS}
    return parts


def summary_parts_from_checks(result: Mapping[str, Any]) -> Dict[str, Any]:
    """``summary_parts`` of an SLA result, rebuilt from its checks when it has none (results built elsewhere)."""
    parts = result.get("summary_parts")
    if isinstance(parts, Mapping):
        return {**parts, "adjustments": [dict(item) for item in parts.get("adjustments") or [] if isinstance(item, Mapping)]}
    checks = [item for item in result.get("checks") or [] if isinstance(item, Mapping)]
    return sla_summary_parts(result.get("verdict"), checks, str(result.get("test_mode") or ""))


def _adjustment_text(item: Mapping[str, Any], lang: str, fail_label: str) -> str:
    code = str(item.get("code") or "")
    if code == ADJUST_RESOURCES:
        return t("sla.summary.adjust_resources", locale=lang, mode=str(item.get("mode") or ""), fail=fail_label)
    if code == ADJUST_TREND:
        return t("sla.summary.adjust_trend", locale=lang, trends=", ".join(str(trend) for trend in item.get("trends") or []))
    return str(item.get("text") or code)


def _number_label(value: Any) -> str:
    number = _safe_float(value)
    return f"{number:.0f}" if number is not None else ""


def _headroom_items(headroom: Mapping[str, Any], lang: str) -> List[str]:
    """Summary items of a step test with a target step: where CPU/memory were checked, how far the latency and
    error thresholds held and what broke above the target step (headroom, not the verdict)."""
    items: List[str] = []
    level = _number_label(headroom.get("verdict_level"))
    if level:
        if headroom.get("role") == ROLE_MARGIN:
            margin = _safe_float(headroom.get("margin_pct"))
            items.append(t("sla.summary.margin_step", locale=lang, level=level, margin=f"{margin or 0:g}"))
        else:
            items.append(t("sla.summary.target_step", locale=lang, level=level))
    ceiling = _number_label(headroom.get("ceiling_level"))
    ratio = _safe_float(headroom.get("ratio"))
    above_level = (_safe_float(headroom.get("ceiling_level")) or 0.0) > (_safe_float(headroom.get("verdict_level")) or 0.0)
    if ceiling and ceiling != level and above_level:
        if ratio is not None:
            items.append(t("sla.summary.ceiling_ratio", locale=lang, level=ceiling, ratio=f"{ratio:g}"))
        else:
            items.append(t("sla.summary.ceiling", locale=lang, level=ceiling))
    above = [str(name) for name in headroom.get("above_target_failed") or []]
    if above:
        items.append(t("sla.summary.above_target", locale=lang, names=", ".join(above)))
    return items


def _regression_reason_text(reason: Any, lang: str) -> str:
    code = str(reason or "")
    return t(f"sla.regression.reason.{code}" if code in REGRESSION_REASONS else "sla.regression.reason.no_value", locale=lang)


def _regression_summary_text(regression: Mapping[str, Any], lang: str) -> str:
    """Summary item of the regression part: the breached deltas against the baseline, «no regression», or why there
    was nothing to compare (only when a regression threshold is set: ``summary_parts`` has no part otherwise)."""
    if regression.get("reason"):
        return t("sla.summary.regression_skipped", locale=lang, reason=_regression_reason_text(regression.get("reason"), lang))
    run = str(regression.get("baseline_run") or "")
    scope = regression_scope_text(regression.get("kind"), regression.get("level"), lang)
    scope_part = f" {scope}" if scope else ""
    items = [item for item in regression.get("items") or [] if isinstance(item, Mapping)]
    failed = [item for item in items if item.get("passed") is False]
    if failed:
        names = ", ".join(
            "{name} {change} {unit}".format(
                name=t(f"sla.regression.short.{_regression_short_key(item)}", locale=lang),
                change=signed_change(item.get("change")),
                unit=regression_unit_text(str(item.get("unit") or ""), lang),
            )
            for item in failed
        )
        return t("sla.summary.regression", locale=lang, run=run, scope=scope_part, items=names)
    if any(item.get("passed") is True for item in items):
        return t("sla.summary.regression_ok", locale=lang, run=run, scope=scope_part)
    return ""


def _regression_short_key(item: Mapping[str, Any]) -> str:
    metric = str(item.get("metric") or "")
    if metric == "p95" and item.get("stat") == "p99":
        return "p99"
    return metric if metric in ("p95", "errors", "capacity") else "p95"


def format_sla_summary(parts: Mapping[str, Any], lang: Optional[str] = None) -> str:
    """Summary 'SLA verdict: <label>. Passed: …; Breached: …; <target step and headroom>; <mode note>;
    <adjustments>' in ``lang`` (default: the current locale)."""
    lang = _lang_of(lang)
    fail_label = verdict_label(FAIL, lang)
    items: List[str] = []
    for key in _SUMMARY_LISTS:
        names = [str(name) for name in parts.get(key) or []]
        if names:
            items.append(t(f"sla.summary.{key}", locale=lang, names=", ".join(names)))
    headroom = parts.get("headroom")
    if isinstance(headroom, Mapping):
        items.extend(_headroom_items(headroom, lang))
    mode = str(parts.get("test_mode") or "")
    if mode in _MODE_NOTES:
        items.append(t(f"sla.summary.mode_{mode}", locale=lang, fail=fail_label))
    if mode == "spike" and parts.get("spike_detected") is False:
        items.append(t("sla.summary.no_spike", locale=lang))
    if mode == "phases" and parts.get("spike_detected") is False:
        items.append(t("sla.summary.no_phases", locale=lang))
    regression = parts.get("regression")
    if isinstance(regression, Mapping):
        text = _regression_summary_text(regression, lang)
        if text:
            items.append(text)
    items.extend(_adjustment_text(item, lang, fail_label) for item in parts.get("adjustments") or [] if isinstance(item, Mapping))
    head = t("sla.summary.head", locale=lang, label=verdict_label(parts.get("verdict"), lang))
    return head + " " + "; ".join(items)


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

    A recovered RPS dip is skipped: it is not a load level the system was tested at. Steps after an
    RPS drop are kept with ``after_drop``: the thresholds they break describe the degradation, but they
    are never the ceiling or the target step.
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
            after_drop=bool(step.get("after_drop")),
        ))
    return sorted(windows, key=lambda item: _segment_end_sort_key(item.end), reverse=True)


def _step_candidates(sla_config: Dict[str, Any], lt_pack: Dict[str, Any], lt_labeled: Any) -> List[StableWindow]:
    """Load steps the SLA may be checked on, latest first: the report steps, else the detector segments."""
    query = sla_config.get("max_performance_query")
    candidates = _windows_from_load_steps(lt_pack, query, lt_labeled)
    if not candidates:
        candidates = _stable_windows_from_rps(lt_pack, query)
    return candidates


def _select_sla_window(
    sla_config: Dict[str, Any],
    candidates: List[StableWindow],
    lt_labeled: Any,
) -> tuple[Optional[StableWindow], List[Dict[str, Any]]]:
    """Latest report step on which latency and error SLA still hold (the ceiling), with its checks.

    The step table is the source of the steps. A later flat RPS stretch can
    continue through a latency collapse, so the search walks from the last step
    backwards and stops at the last one that still passes. A step after an RPS drop
    that breaks the thresholds is the degradation of the ceiling; one that holds
    them is passed over. Detector segments are used only when the report has no
    load steps. When every step breaks the SLA the latest one (before a drop, when
    there is one) is kept and fails.
    """
    if not candidates:
        return None, _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, None, "primary")
    by_window: List[tuple[StableWindow, List[Dict[str, Any]]]] = []
    degraded: Optional[StableWindow] = None
    degraded_names: tuple[str, ...] = ()
    for window in candidates:
        checks = _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, window, "primary")
        by_window.append((window, checks))
        failed = tuple(str(c["name"]) for c in checks if c.get("passed") is False)
        if not failed:
            if window.after_drop:
                continue
            if degraded is None:
                return window, checks
            return replace(window, degraded_level=degraded.level, degraded_checks=degraded_names), checks
        degraded, degraded_names = window, failed
    fallback = next(((window, checks) for window, checks in by_window if not window.after_drop), by_window[0])
    return fallback


def _reach_floor(level: float) -> float:
    """Lowest measured step RPS that still reaches ``level`` (``TARGET_STEP_TOLERANCE_PCT`` below it)."""
    return level * (1.0 - TARGET_STEP_TOLERANCE_PCT / 100.0)


def _select_target_window(
    sla_config: Dict[str, Any],
    candidates: List[StableWindow],
    lt_labeled: Any,
    level: float,
    min_level: float = 0.0,
) -> Optional[StableWindow]:
    """First step in time at ``level`` RPS or above (within the tolerance, never below ``min_level``) on which
    latency and error SLA hold.

    A step at the level that breaks them does not end the search: the next one that holds is taken, as the
    ceiling search does. None when no step at the level holds.
    """
    floor = max(_reach_floor(level), min_level)
    for window in sorted(candidates, key=lambda item: _segment_end_sort_key(item.end)):
        if window.level < floor or window.after_drop:
            continue
        checks = _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, window, "primary")
        if not any(check.get("passed") is False for check in checks):
            return window
    return None


def _same_window(a: Optional[StableWindow], b: Optional[StableWindow]) -> bool:
    return a is not None and b is not None and (a.start, a.end) == (b.start, b.end)


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


def _breach_minutes(df: pd.DataFrame, series: Optional[str], start: pd.Timestamp, end: pd.Timestamp, threshold: float) -> Optional[float]:
    """How long ``series`` stays above ``threshold`` inside the window: its points over it times the data interval
    (the same buckets ``_worst_series_p95`` reads)."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex) or series is None:
        return None
    column = next((col for col in df.columns if str(col) == series), None)
    if column is None:
        return None
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    width = _bin_width(idx)
    values = pd.Series(pd.to_numeric(df[column], errors="coerce").to_numpy(), index=idx)
    values = values[(values.index >= start) & (values.index + width <= end)].dropna()
    step = width.total_seconds() / 60.0 if width > pd.Timedelta(0) else 1.0
    return round(int((values > threshold).sum()) * step, 2)


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
            message=t("sla.check.no_source"),
            category=category,
        )
    frame = _labeled_frame(labeled, query)
    # The configured query may be a unique part of the label; the chart is named by the full label.
    matched = _find_section_by_label([item for item in labeled if isinstance(item, dict)], query) if isinstance(labeled, list) else None
    source_label = str(matched.get("label") or query) if isinstance(matched, dict) else query
    if frame is None:
        return _make_check(
            name=name,
            threshold=threshold,
            actual=None,
            passed=None,
            message=t("sla.check.no_query_data", query=query),
            category=category,
            query_label=source_label,
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
        empty_message = _empty_scope_text(window, query) if window is not None else t("sla.check.no_numbers", query=query)
        return _make_check(
            name=name,
            threshold=threshold,
            actual=None,
            passed=None,
            message=empty_message,
            category=category,
            query_label=source_label,
        )
    passed = actual <= threshold
    series_note = t("sla.check.series_note", series=series) if series else ""
    check = _make_check(
        name=name,
        threshold=threshold,
        actual=round(actual, 3),
        passed=passed,
        message=t(
            "sla.check.p95", actual=f"{actual:.2f}", unit=_unit_text(unit), op="≤" if passed else ">",
            threshold=f"{threshold:g}", scope=scope, query=query, series_note=series_note,
        ),
        category=category,
        query_label=source_label,
        series=series,
    )
    if not passed and window is not None and window.kind == "phase" and start is not None and end is not None:
        # A phase is a few minutes long, so its 95th percentile is about its worst minute: how long the threshold
        # stayed broken decides between a failure and a short excursion (``_phase_checks``).
        check["breach_min"] = _breach_minutes(frame, series, start, end, threshold)
    return check


def _make_check(
    name: str,
    threshold: Any,
    actual: Any,
    passed: Optional[bool],
    severity: str = "warning",
    message: str = "",
    category: str = "primary",
    query_label: Optional[str] = None,
    series: Optional[str] = None,
) -> Dict[str, Any]:
    """One SLA check. ``query_label``/``series`` name the chart the value was measured on; they are
    stored only when known (the report page draws the threshold on that chart). ``message`` keeps its
    format: older readers still parse the query and the series out of it."""
    check: Dict[str, Any] = {
        "name": name,
        "threshold": threshold,
        "actual": actual,
        "passed": passed,
        "severity": severity,
        "message": message,
        "category": category,
    }
    if query_label:
        check["query_label"] = str(query_label)
    if series:
        check["series"] = str(series)
    return check


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
    target_window: Optional[StableWindow] = None,
) -> Dict[str, Any]:
    """target_rps of a step test: reached when there is a target step (``target_window``); the actual value is
    the ceiling ``window``. Without steps — stable_max or the peak of the designated query."""
    if window is not None:
        passed = target_window is not None
        actual: Optional[float] = max(window.level, target_window.level) if target_window is not None else window.level
        degraded_note = ""
        if window.degraded_level is not None:
            degraded_note = t("sla.target.degraded", level=f"{window.degraded_level:.0f}", checks=", ".join(window.degraded_checks))
        confirmed_note = ""
        if target_window is not None and not _same_window(target_window, window):
            confirmed_note = t("sla.target.confirmed", level=f"{target_window.level:.0f}")
        if not passed and window.level >= _reach_floor(target_rps):
            # Steps at the target were run, but none of them (none at all) holds the latency and error thresholds:
            # no RPS was reached within the SLA, so the check has no actual value.
            message = t(
                "sla.target.not_confirmed", level=f"{window.level:.0f}", label=window.label, target=f"{target_rps:.0f}",
                series=window.series, degraded=degraded_note,
            )
            actual = None
        else:
            op = ("≥" if actual >= target_rps else "≈") if passed else "<"
            message = t(
                "sla.target.step", level=f"{actual:.1f}", label=window.label, op=op, target=f"{target_rps:.0f}",
                series=window.series, degraded=degraded_note, confirmed=confirmed_note,
            )
        return _make_check(
            name="target_rps",
            threshold=target_rps,
            actual=round(actual, 2) if actual is not None else None,
            passed=passed,
            severity="critical",
            message=message,
            query_label=window.label,
            series=window.series,
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
            message=t("sla.target.no_rps", reason=reason or method),
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
            t("sla.target.measured", actual=f"{actual_rps:.1f}", method=method, op="≥" if passed else "<", target=f"{target_rps:.0f}")
            + (f"; label='{source_label}'" if source_label else "")
            + (f"; series='{source_series}'" if source_series else "")
        ),
        query_label=source_label,
        series=source_series,
    )


def _margin_pct(sla_config: Dict[str, Any]) -> Optional[float]:
    """``target_margin_pct`` when it is a positive number, else None (the margin check is off)."""
    value = _safe_float(sla_config.get("target_margin_pct"))
    return value if value is not None and value > 0 else None


def _margin_check(
    target_rps: float,
    margin_pct: float,
    window: Optional[StableWindow],
    margin_window: Optional[StableWindow],
    target_check: Mapping[str, Any],
) -> Dict[str, Any]:
    """``target_margin``: the latency and error thresholds hold at ``target × (1 + margin)`` too.

    On a step test it passes when there is a margin step (``margin_window``); the actual value is the ceiling.
    Without steps the RPS of the target check is compared with the margin level.
    """
    level = target_rps * (1.0 + margin_pct / 100.0)
    margin = f"{margin_pct:g}"
    query_label = target_check.get("query_label")
    series = target_check.get("series")
    if window is not None:
        passed = margin_window is not None
        actual = max(window.level, margin_window.level) if margin_window is not None else window.level
        if passed:
            message = t("sla.margin.confirmed", margin=margin, level=f"{level:.0f}", step=f"{margin_window.level:.0f}")
        else:
            message = t("sla.margin.missed", margin=margin, level=f"{level:.0f}", ceiling=f"{window.level:.0f}")
    else:
        actual = _safe_float(target_check.get("actual"))
        if actual is None:
            return _make_check(
                name="target_margin", threshold=round(level, 2), actual=None, passed=None, category="secondary",
                message=t("sla.margin.no_rps", margin=margin), query_label=query_label, series=series,
            )
        passed = actual >= max(_reach_floor(level), target_rps)
        message = t(
            "sla.margin.measured", actual=f"{actual:.1f}", op="≥" if actual >= level else ("≈" if passed else "<"),
            level=f"{level:.0f}", margin=margin,
        )
    return _make_check(
        name="target_margin",
        threshold=round(level, 2),
        actual=round(actual, 2),
        passed=passed,
        message=message,
        category="secondary",
        query_label=query_label,
        series=series,
    )


def _headroom(
    sla_config: Dict[str, Any],
    target_rps: float,
    margin_pct: Optional[float],
    window: StableWindow,
    verdict_window: StableWindow,
    hr_labeled: Any,
    verdict_checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """What a step test showed above the step the verdict was decided on: the ceiling (last step on which the
    latency and error thresholds hold), its ratio to the target and the CPU/memory checks on the ceiling.

    Those checks are information only: a breach above the target step is headroom used up, not a risk.
    ``above_target_failed`` names the ones that hold on the verdict step (``verdict_checks``) and break above it.
    """
    above: List[Dict[str, Any]] = []
    # A later step at a lower RPS (a sag too small to count as a drop) is no load above the verdict step.
    if not _same_window(window, verdict_window) and window.level > verdict_window.level:
        above = _checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, window, "secondary")
    failed_on_step = {str(check.get("name")) for check in verdict_checks if check.get("passed") is False}
    peak = max(window.level, verdict_window.level)
    return {
        "role": verdict_window.role,
        "verdict_level": round(verdict_window.level, 2),
        "target_rps": target_rps,
        "margin_pct": margin_pct,
        "ceiling_level": round(window.level, 2),
        "ratio": round(peak / target_rps, 2) if target_rps > 0 else None,
        "above_target_failed": [
            str(check["name"]) for check in above if check.get("passed") is False and str(check["name"]) not in failed_on_step
        ],
        "above_target_checks": above,
    }


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
        kind="hold", title=t("sla.window.hold"),
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
                message=t("sla.target.hold_no_data", query=query or t("sla.target.query_not_chosen")),
                query_label=query,
            ))
        else:
            result = hold_check(hold, target_rps)
            checks.append(_make_check(
                name="target_rps", threshold=target_rps, actual=result["actual"], passed=result["passed"],
                severity="critical", message=result["message"],
                query_label=window.label if window else None, series=window.series if window else None,
            ))
    checks.extend(_checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, window, "primary"))
    checks.extend(_checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, window, "secondary"))
    return checks, window


def _worst_check(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The check of the worst window: a failed one with the highest value (a sustained failure before a short
    excursion of a phase), else the highest value."""
    if not candidates:
        return None
    failed = [c for c in candidates if c.get("passed") is False and c.get("actual") is not None]
    sustained = [c for c in failed if not c.get("brief")]
    if sustained or failed:
        return max(sustained or failed, key=lambda c: float(c["actual"]))
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
            kind="spike", title=str(item.get("label") or t("sla.window.spike", index=str(item.get("index")))),
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
            message=t("sla.recovery.no_spike"),
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
            t("sla.recovery.not_recovered_limit", tolerance=f"{tolerance:g}", limit=f"{limit:g}", spikes=", ".join(not_recovered))
            if tolerance is not None and limit is not None
            else t("sla.recovery.not_recovered", spikes=", ".join(not_recovered))
        )
    elif all(state is True for state in states):
        passed = True
        message = t("sla.recovery.recovered", actual=f"{actual:g}", limit=f"{limit:g}")
    else:
        passed = None
        message = t("sla.recovery.partial")
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
                message=str(profile.get("reason") or t("sla.spike.not_detected")),
            ))
        else:
            passed = top.level >= target_rps
            checks.append(_make_check(
                name="target_rps", threshold=target_rps, actual=round(top.level, 2), passed=passed,
                severity="critical",
                message=t(
                    "sla.target.spike", title=top.title, level=f"{top.level:.1f}", op="≥" if passed else "<",
                    target=f"{target_rps:.0f}", label=top.label,
                ),
                query_label=top.label, series=top.series,
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


def _phase_windows(lt_pack: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[StableWindow]]:
    profile = lt_pack.get("phase_profile") if isinstance(lt_pack, dict) else None
    profile = profile if isinstance(profile, dict) else {}
    phases = [
        item for item in (profile.get("phases") or [])
        if isinstance(item, dict) and not item.get("idle") and not item.get("excluded")
        and item.get("start_utc") and item.get("end_utc")
    ]
    windows = [
        StableWindow(
            start=str(item["start_utc"]), end=str(item["end_utc"]), level=float(item.get("rps_mean") or 0.0),
            series=str(profile.get("label_key") or ""), label=str(profile.get("source_label") or ""),
            kind="phase", title=str(item.get("label") or item.get("title") or ""),
        )
        for item in phases
    ]
    return profile, phases, windows


def _phase_verdict(checks: List[Dict[str, Any]]) -> str:
    failed = [c for c in checks if c.get("passed") is False]
    if any(_fails_run(c) for c in failed):
        return FAIL
    if failed:
        return RISK
    return PASS if any(c.get("passed") is True for c in checks) else NO_DATA


def _phase_reference_checks(profile: Dict[str, Any], phases: List[Dict[str, Any]], lt_pack: Dict[str, Any], sla_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Completeness and phase durations against the phases of the previous scenario run (secondary)."""
    from AI.phases import compare_phase_lists, phase_settings

    reference = lt_pack.get("phase_reference") if isinstance(lt_pack, dict) else None
    if not isinstance(reference, dict) or not reference.get("phases") or not profile.get("detected"):
        return []
    settings = phase_settings(sla_config)
    run = str(reference.get("run_name") or "")
    comparison = compare_phase_lists(reference["phases"], phases, settings.min_minutes, int(profile.get("interval_sec") or 60), settings)
    missing = comparison["missing"]
    ref_count = sum(
        1 for item in reference["phases"]
        if isinstance(item, dict) and not item.get("idle") and not item.get("excluded") and not settings.excludes(item)
    )
    checks = [_make_check(
        name="phase_completeness", threshold=0, actual=len(missing), passed=not missing, severity="warning",
        category="secondary",
        message=(
            t("sla.phases.missing", run=run, phases=", ".join(str(item.get("label") or item.get("title")) for item in missing))
            if missing else t("sla.phases.complete", run=run, count=ref_count)
        ),
    )]
    worst = comparison["worst_duration"]
    tolerance = settings.duration_tolerance_pct
    if worst is None:
        if comparison["durations"]:
            checks.append(_make_check(
                name="phase_duration", threshold=tolerance, actual=0.0, passed=True, category="secondary",
                message=t("sla.phases.duration_same", run=run),
            ))
        return checks
    change = float(worst["change_pct"])
    checks.append(_make_check(
        name="phase_duration", threshold=tolerance, actual=round(abs(change), 1), passed=abs(change) <= tolerance,
        severity="warning", category="secondary",
        message=t(
            "sla.phases.duration", title=str(worst.get("label") or worst.get("title") or ""),
            reference=f"{float(worst['reference_min']):g}", current=f"{float(worst['current_min']):g}",
            change=f"{change:+.0f}", tolerance=f"{tolerance:g}", run=run,
        ),
    ))
    return checks


def _phase_checks(
    sla_config: Dict[str, Any],
    lt_pack: Dict[str, Any],
    lt_labeled: Any,
    hr_labeled: Any,
) -> tuple[List[Dict[str, Any]], Optional[StableWindow], bool, List[Dict[str, Any]]]:
    """Checks of a scenario run: every threshold on its worst phase, then the phase checks against the reference.

    Returns the checks, the window of the worst failed check (the phase the verdict points at; None when every
    phase holds), whether phases were found and the verdict of every phase (``phase_results``).
    """
    from AI.phases import phase_settings

    settings = phase_settings(sla_config)
    profile, phases, windows = _phase_windows(lt_pack)
    detected = bool(profile.get("detected")) and bool(windows)
    if not windows:
        if profile.get("phases"):
            # Detected phases may all be excluded (or idle). Do not evaluate those same windows again via
            # the missing-phase fallback: an exclusion must never turn into a failing whole-run check.
            return [], None, bool(profile.get("detected")), []
        checks = _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, None, "primary")
        checks.extend(_checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, None, "secondary"))
        return checks, None, False, []
    per_phase: List[tuple[Dict[str, Any], StableWindow, List[Dict[str, Any]]]] = []
    window_of: Dict[int, StableWindow] = {}
    for phase, window in zip(phases, windows):
        phase_checks = _checks_for_window(sla_config, PRIMARY_SPECS, lt_labeled, window, "primary")
        phase_checks.extend(_checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, window, "secondary"))
        for check in phase_checks:
            _mark_brief(check, settings.breach_minutes)
        per_phase.append((phase, window, phase_checks))
        for check in phase_checks:
            window_of[id(check)] = window
    checks: List[Dict[str, Any]] = []
    for name, *_ in (*PRIMARY_SPECS, *SECONDARY_SPECS):
        worst = _worst_check([check for _, _, items in per_phase for check in items if check["name"] == name])
        if worst is not None:
            checks.append(worst)
    failed = [c for c in checks if c.get("passed") is False]
    pick = (
        next((c for c in failed if _fails_run(c)), None)
        or next((c for c in failed if _is_primary(c)), None)
        or (failed[0] if failed else None)
    )
    window = window_of.get(id(pick)) if pick is not None else None
    results = [
        {
            "index": phase.get("index"),
            "label": phase.get("label"),
            "key": phase.get("key"),
            "verdict": _phase_verdict(items),
            "failed": [str(c["name"]) for c in items if c.get("passed") is False],
            "worst": window is not None and phase_window is window,
        }
        for phase, phase_window, items in per_phase
    ]
    checks.extend(_phase_reference_checks(profile, phases, lt_pack, sla_config))
    return checks, window, detected, results


def _capacity_verdict(checks: List[Dict[str, Any]]) -> str:
    rps_check = next((c for c in checks if c["name"] == "target_rps"), None)
    others = [c for c in checks if c["name"] != "target_rps"]
    evaluable = [c for c in checks if c["passed"] is not None]
    rps_reached = rps_check["passed"] if rps_check else None
    if rps_reached is True:
        return RISK if any(c["passed"] is False for c in others) else PASS
    if rps_reached is False:
        return FAIL
    if rps_check is not None:
        if all(c["passed"] is True for c in evaluable):
            return RISK
        if any(c["passed"] is False for c in evaluable):
            return FAIL
        return NO_DATA
    # Without a target the latency and error thresholds decide the run; CPU/memory over the threshold is a risk.
    if any(c["passed"] is False and _is_primary(c) for c in evaluable):
        return FAIL
    if any(c["passed"] is False for c in evaluable):
        return RISK
    if evaluable:
        return PASS
    return NO_DATA


def _is_primary(check: Mapping[str, Any]) -> bool:
    return str(check.get("category") or "primary") == "primary"


def _fails_run(check: Mapping[str, Any]) -> bool:
    """A broken primary threshold fails the run, unless it is a short excursion inside a phase (a risk)."""
    return _is_primary(check) and not check.get("brief")


def _mark_brief(check: Dict[str, Any], breach_minutes: float) -> None:
    """Marks a failed primary check of a phase whose threshold stayed broken shorter than ``breach_minutes``."""
    breach = check.get("breach_min")
    if check.get("passed") is not False or not _is_primary(check) or breach is None or breach >= breach_minutes:
        return
    check["brief"] = True
    check["message"] = f"{check.get('message') or ''} {t('sla.phases.brief', minutes=f'{float(breach):g}', limit=f'{breach_minutes:g}')}".strip()


def _verdict_for_mode(mode: str, checks: List[Dict[str, Any]], detected: bool = True) -> str:
    """Deterministic verdict code by the policy of the test-profile mode (see module docstring)."""
    if mode in {"capacity", "stress"}:
        return _capacity_verdict(checks)
    evaluable = [c for c in checks if c["passed"] is not None]
    failed = [c for c in checks if c["passed"] is False]
    all_passed = bool(evaluable) and all(c["passed"] is True for c in evaluable)
    if mode == "spike":
        if not detected:
            return NO_DATA
        if any(c["name"] in {"target_rps", "recovery"} for c in failed):
            return FAIL
        if failed:
            return RISK
        return PASS if all_passed else NO_DATA
    if any(_fails_run(c) for c in failed):
        return FAIL
    if failed:
        return RISK
    return PASS if all_passed else NO_DATA


def evaluate_sla(
    domain_data: Dict[str, Dict[str, Any]],
    sla_config: Dict[str, Any],
    test_profile: Optional[Dict[str, Any]] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Оценивает результаты теста по SLA-критериям с учётом режима профиля теста.

    ``verdict`` is a code; check messages and ``summary`` are written in ``lang`` (default: the current locale)
    and ``summary_parts`` lets the pipeline rebuild the summary.
    """
    lang = normalize_lang(lang) or get_locale()
    with with_locale(lang):
        return _evaluate_sla(domain_data, sla_config, test_profile, lang)


def _evaluate_sla(
    domain_data: Dict[str, Dict[str, Any]],
    sla_config: Dict[str, Any],
    test_profile: Optional[Dict[str, Any]],
    lang: str,
) -> Dict[str, Any]:
    sla_config = sla_config if isinstance(sla_config, dict) else {}
    test_mode = _test_mode_from_profile(test_profile or sla_config.get("test_profile"))
    if not sla_config or not any(
        _safe_float(sla_config.get(k)) is not None
        for k in ("target_rps", "max_error_rate_pct", "max_p95_ms", "max_p99_ms", "max_cpu_pct", "max_memory_pct")
    ):
        return {
            "verdict": NO_DATA,
            "checks": [],
            "summary": t("sla.summary.no_criteria"),
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
    verdict_window: Optional[StableWindow] = None
    headroom: Optional[Dict[str, Any]] = None
    phase_results: Optional[List[Dict[str, Any]]] = None

    if test_mode == "stability":
        checks, window = _stability_checks(sla_config, target_rps, lt_pack, lt_labeled, hr_labeled)
    elif test_mode == "spike":
        checks, window, detected = _spike_checks(sla_config, target_rps, lt_pack, lt_labeled, hr_labeled)
    elif test_mode == "phases":
        checks, window, detected, phase_results = _phase_checks(sla_config, lt_pack, lt_labeled, hr_labeled)
    else:
        checks, window, verdict_window, headroom = _capacity_checks(
            sla_config, target_rps, perf_query, lt_pack, lt_labeled, hr_labeled,
        )

    if not checks:
        return {
            "verdict": NO_DATA,
            "checks": [],
            "summary": t("sla.summary.no_checks"),
            "test_mode": test_mode,
        }

    verdict = _verdict_for_mode(test_mode, checks, detected)
    parts = sla_summary_parts(verdict, checks, test_mode, detected, headroom)
    summary = format_sla_summary(parts, lang)

    logger.info(
        "SLA evaluation: verdict=%s, mode=%s, checks=%d, passed=%d, failed=%d, target_step=%s",
        verdict, test_mode, len(checks), len(parts["passed"]), len(parts["failed"]),
        verdict_window.level if verdict_window is not None else None,
    )

    result: Dict[str, Any] = {
        "verdict": verdict,
        "checks": checks,
        "summary": summary,
        "summary_parts": parts,
        "test_mode": test_mode,
        "stable_window": window.to_dict() if window is not None else None,
    }
    if verdict_window is not None:
        result["target_window"] = verdict_window.to_dict()
    if headroom is not None:
        result["headroom"] = headroom
    if phase_results is not None:
        result["phase_results"] = phase_results
    return result


def _capacity_checks(
    sla_config: Dict[str, Any],
    target_rps: Optional[float],
    perf_query: Optional[str],
    lt_pack: Dict[str, Any],
    lt_labeled: Any,
    hr_labeled: Any,
) -> tuple[List[Dict[str, Any]], Optional[StableWindow], Optional[StableWindow], Optional[Dict[str, Any]]]:
    """Checks of a step or stress test, the ceiling window, the step the verdict is decided on and the headroom.

    The latency and error checks come from the ceiling. CPU/memory are checked on the margin step, else the
    target step, else the ceiling (no target, or the target is not reached).
    """
    candidates = _step_candidates(sla_config, lt_pack, lt_labeled)
    window, primary_checks = _select_sla_window(sla_config, candidates, lt_labeled)
    margin_pct = _margin_pct(sla_config) if target_rps is not None else None
    target_window: Optional[StableWindow] = None
    margin_window: Optional[StableWindow] = None
    if target_rps is not None and window is not None:
        found = _select_target_window(sla_config, candidates, lt_labeled, target_rps)
        if found is not None:
            target_window = replace(found, role=ROLE_TARGET)
            if margin_pct is not None:
                # A margin smaller than the tolerance must still be shown above the target itself.
                above = _select_target_window(
                    sla_config, candidates, lt_labeled, target_rps * (1.0 + margin_pct / 100.0), min_level=target_rps,
                )
                if above is not None:
                    margin_window = replace(above, role=ROLE_MARGIN, margin_pct=margin_pct)
    verdict_window = margin_window or target_window
    checks: List[Dict[str, Any]] = []
    if target_rps is not None:
        target_check = _capacity_target_check(sla_config, target_rps, window, lt_pack, perf_query, target_window=target_window)
        checks.append(target_check)
        if margin_pct is not None:
            checks.append(_margin_check(target_rps, margin_pct, window, margin_window, target_check))
    checks.extend(primary_checks)
    secondary = _checks_for_window(sla_config, SECONDARY_SPECS, hr_labeled, verdict_window or window, "secondary")
    checks.extend(secondary)
    headroom = None
    if target_rps is not None and window is not None and verdict_window is not None:
        headroom = _headroom(sla_config, target_rps, margin_pct, window, verdict_window, hr_labeled, secondary)
    return checks, window, verdict_window, headroom


def _apply_regression(verdict: Any, checks: List[Mapping[str, Any]]) -> str:
    """Verdict code after the regression checks: a breached ``warning`` one turns ``pass`` into ``risk``, a breached
    ``critical`` one turns ``pass`` and ``risk`` into ``fail``; ``fail`` and ``no_data`` stay."""
    code = to_code(verdict)
    if code in (FAIL, NO_DATA):
        return code
    failed = [
        check for check in checks
        if isinstance(check, Mapping) and str(check.get("category") or "") == CATEGORY_REGRESSION and check.get("passed") is False
    ]
    if any(str(check.get("severity") or "") == SEVERITY_CRITICAL for check in failed):
        return FAIL
    if failed and code == PASS:
        return RISK
    return code


# Reasons a comparison is recorded in sla_details.regression without a summary item: scenario runs have their own
# reference run, a run without checks has no summary to extend.
_SILENT_REGRESSION_REASONS = (REASON_MODE_UNSUPPORTED, REASON_NO_CHECKS)


def _type_mode(test_type: Any) -> str:
    return mode_of_test_type(normalize_test_type(str(test_type or "")) or test_type)


def apply_regression(
    result: Optional[Mapping[str, Any]],
    baseline: Optional[Mapping[str, Any]],
    sla_config: Optional[Mapping[str, Any]],
    lang: Optional[str] = None,
    *,
    step_table: Optional[Mapping[str, Any]] = None,
    service: str = "",
) -> Dict[str, Any]:
    """Adds the regression against the baseline run to an SLA result (``evaluate_sla``) and recomputes its verdict.

    ``baseline``: ``{available, run_name, run_id, release, mode, service, test_type, sla_details}`` (the baseline of
    the model context and its stored final row). ``step_table`` is the load-step table of the run (the source of p95
    and errors per step). Without ``regression_*`` thresholds the result only gets ``regression = {available: false,
    reason: disabled}``: checks, verdict and summary stay as they are. Otherwise the ``regression_*`` checks are
    appended, the verdict goes through ``_apply_regression`` and ``summary_parts.regression`` rebuilds the summary in
    ``lang``. ``regression`` records the comparison (or the reason there was none) for ``sla_details``.
    """
    out = dict(result or {})
    thresholds = thresholds_from_sla(sla_config)
    if not thresholds_enabled(thresholds):
        out["regression"] = {"available": False, "reason": REASON_DISABLED}
        return out
    checks = [dict(item) for item in out.get("checks") or [] if isinstance(item, Mapping)]
    if not checks or not isinstance(out.get("summary_parts"), Mapping):
        out["regression"] = {"available": False, "reason": REASON_NO_CHECKS}
        return out
    lang = normalize_lang(lang) or get_locale()
    base = baseline if isinstance(baseline, Mapping) else {}
    base_details = base.get("sla_details") if isinstance(base.get("sla_details"), Mapping) else {}
    mode = str(out.get("test_mode") or "capacity")
    record: Dict[str, Any] = {
        "available": False,
        "baseline_run_name": str(base.get("run_name") or ""),
        "baseline_run_id": str(base.get("run_id") or ""),
        "baseline_release": str(base.get("release") or ""),
        "mode": str(base.get("mode") or ""),
    }
    reason = ""
    comparison: Optional[Dict[str, Any]] = None
    if mode == "phases":
        reason = REASON_MODE_UNSUPPORTED
    elif not base.get("available") or not record["baseline_run_name"]:
        reason = REASON_NO_BASELINE
    elif not base_details.get("checks"):
        reason = REASON_NO_BASELINE_SLA
    elif not str(service or "").strip() or str(base.get("service") or "").strip() != str(service).strip():
        reason = REASON_OTHER_SERVICE
    elif _type_mode(base.get("test_type")) != mode:
        reason = REASON_OTHER_TYPE
    else:
        current: Dict[str, Any] = {key: out.get(key) for key in ("stable_window", "target_window", "headroom")}
        current["checks"] = checks
        if isinstance(step_table, Mapping):
            current["load_step_table"] = step_table
        comparison = compare_stored(current, base_details, mode, thresholds)
        if not comparison.get("available"):
            reason = str(comparison.get("reason") or "")
    new_checks: List[Dict[str, Any]] = []
    if comparison is not None:
        record.update({key: comparison.get(key) for key in ("level", "kind", "step_label", "items", "skipped")})
        if not reason:
            new_checks = regression_checks(comparison, thresholds, lang, baseline_run=record["baseline_run_name"], mode=mode)
            if not new_checks:
                # Thresholds are set only for metrics that could not be compared.
                limited = [
                    item for item in comparison.get("skipped") or []
                    if any(limit is not None for limit in regression_metric_limits(str(item.get("metric") or ""), thresholds))
                ]
                reason = str(limited[0].get("reason") or "") if limited else REASON_NO_VALUE
    record["available"] = not reason
    if reason:
        record["reason"] = reason
    out["regression"] = record
    if reason in _SILENT_REGRESSION_REASONS:
        return out
    parts = summary_parts_from_checks(out)
    if reason:
        parts["regression"] = {"baseline_run": record["baseline_run_name"], "reason": reason}
    else:
        parts["regression"] = {
            "baseline_run": record["baseline_run_name"],
            "baseline_release": record["baseline_release"],
            "mode": record["mode"],
            "level": record.get("level"),
            "kind": record.get("kind"),
            "items": regression_summary_items(comparison or {}),
        }
    verdict = _apply_regression(out.get("verdict"), new_checks)
    parts["verdict"] = verdict
    out["checks"] = [*checks, *new_checks]
    out["verdict"] = verdict
    out["summary_parts"] = parts
    out["summary"] = format_sla_summary(parts, lang)
    if verdict != to_code(result.get("verdict") if isinstance(result, Mapping) else None):
        logger.info("SLA verdict after the regression against '%s': %s", record["baseline_run_name"], verdict)
    return out
