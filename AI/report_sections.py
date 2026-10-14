"""Test-type section of a final report as plain (label, text) rows for Confluence renderers.

The web page (static/js/reports.js) renders the same sections in JavaScript; keep both in step.

- soak   → «Стабильность под нагрузкой» from ``stability_under_load`` (v1 payloads have only
  target_rps / actual_rps / sla_summary / focus and still render);
- spike  → «Всплески нагрузки» from ``spike_analysis``;
- stress → «Точка насыщения» from ``saturation`` (plus peak times from ``peak_performance``).

Old reports without a test-type payload return None, and the caller falls back to the peak table.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from AI.test_profile import mode_of

Rows = List[Tuple[str, str]]

TREND_TITLES = {"rps": "RPS", "p95": "p95 latency", "error_rate": "доля ошибок", "cpu": "CPU", "memory": "память"}
RESOURCE_TITLES = {"cpu": "CPU", "memory": "память"}


def _text(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _num(value: Any, digits: int = 1) -> Optional[str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{number:.0f}"


def _hhmm(iso: Any) -> str:
    text = str(iso or "")
    return text[11:16] if len(text) >= 16 else text


def _signed(value: Any, unit: str) -> Optional[str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:+.1f}".rstrip("0").rstrip(".") + unit


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def report_mode(report: Mapping[str, Any]) -> str:
    """Mode of a stored final report: payload keys first, then ``test_profile``."""
    if isinstance(report.get("spike_analysis"), Mapping):
        return "spike"
    if isinstance(report.get("saturation"), Mapping):
        return "stress"
    profile = _mapping(report.get("test_profile"))
    peak = _mapping(report.get("peak_performance") or report.get("peak_perfomance"))
    if isinstance(report.get("stability_under_load"), Mapping):
        return "stability"
    mode = mode_of(profile) if profile else "capacity"
    if mode == "capacity" and (peak.get("not_applicable") or peak.get("notApplicable")):
        return "stability"
    return mode


def _type_label(report: Mapping[str, Any], fallback: str) -> str:
    profile = _mapping(report.get("test_profile"))
    return _text(profile.get("label_ru") or profile.get("test_type"), fallback)


def stability_rows(report: Mapping[str, Any]) -> Rows:
    stability = _mapping(report.get("stability_under_load"))
    profile = _mapping(report.get("test_profile"))
    rows: Rows = [
        ("Тип теста", _text(profile.get("test_type"), "stability/soak")),
        ("Фокус оценки", _text(stability.get("focus") or profile.get("focus"), "Удержание нагрузки без накопления деградации")),
        ("Целевой RPS", _text(stability.get("target_rps"))),
        ("Фактический RPS", _text(stability.get("actual_rps"))),
    ]
    hold = _mapping(stability.get("hold"))
    if hold:
        window = f"{_hhmm(hold.get('start_iso'))}–{_hhmm(hold.get('end_iso'))}"
        duration = _num(hold.get("duration_min"))
        rows.append(("Окно удержания", f"{window}" + (f" ({duration} мин)" if duration else "")))
        lowest = _num(hold.get("min_bucket_rps"))
        if lowest is not None:
            drop = _num(hold.get("drop_pct"))
            rows.append((
                "Минимальный интервал",
                f"{_text(hold.get('min_bucket_label'), 'интервал')}: {lowest} RPS" + (f" (−{drop} % от среднего)" if drop else ""),
            ))
        if hold.get("early_drop_iso"):
            rows.append(("Падение RPS до конца теста", _hhmm(hold.get("early_drop_iso"))))
        if hold.get("note"):
            rows.append(("Примечание", _text(hold.get("note"))))
    trends = [item for item in stability.get("trends") or [] if isinstance(item, Mapping)]
    if trends:
        parts = []
        for trend in trends:
            change = _signed(trend.get("change_pct"), " %")
            if change is None:
                continue
            title = trend.get("title") or TREND_TITLES.get(str(trend.get("metric_kind")), trend.get("metric_kind"))
            parts.append(f"{title}: {change}" + (" (риск)" if trend.get("flagged") else ""))
        if parts:
            rows.append(("Тренды за окно удержания", "; ".join(parts)))
    rows.append(("Итог SLA", _text(stability.get("sla_summary"))))
    return rows


def spike_rows(report: Mapping[str, Any]) -> Rows:
    analysis = _mapping(report.get("spike_analysis"))
    rows: Rows = [("Тип теста", _type_label(report, "spike"))]
    baseline = _num(analysis.get("baseline_rps"))
    if baseline is not None:
        rows.append(("Фоновая нагрузка", f"{baseline} RPS"))
    if not analysis.get("detected"):
        rows.append(("Всплеск", _text(analysis.get("reason"), "не обнаружен")))
        rows.append(("Итог SLA", _text(analysis.get("sla_summary"))))
        return rows
    target = _num(analysis.get("target_rps"))
    if target is not None:
        reached = analysis.get("target_reached")
        rows.append(("Целевой RPS всплеска", f"{target}" + ("" if reached is None else (" — достигнут" if reached else " — не достигнут"))))
    for spike in analysis.get("spikes") or []:
        if not isinstance(spike, Mapping):
            continue
        pre = _mapping(spike.get("pre"))
        during = _mapping(spike.get("during"))
        bits = [f"{_hhmm(spike.get('start_iso'))}–{_hhmm(spike.get('end_iso'))} ({_num(spike.get('duration_sec'), 0)} с)"]
        if pre.get("p95") is not None and during.get("p95") is not None:
            overshoot = _signed(spike.get("overshoot_p95_pct"), " %")
            bits.append(f"p95 {_num(pre.get('p95'), 3)} → {_num(during.get('p95'), 3)}" + (f" ({overshoot})" if overshoot else ""))
        errors = _signed(spike.get("overshoot_error_rate_pp"), " п.п.")
        if errors:
            bits.append(f"ошибки {errors}")
        if spike.get("recovered") is True:
            bits.append(f"восстановление за {_num(spike.get('recovery_time_sec'), 0)} с")
        elif spike.get("recovered") is False:
            bits.append("задержки не вернулись к фону")
        if spike.get("note"):
            bits.append(str(spike.get("note")))
        rows.append((_text(spike.get("label"), "всплеск"), ", ".join(bits)))
    rows.append(("Итог SLA", _text(analysis.get("sla_summary"))))
    return rows


def _resource_text(resource: Mapping[str, Any]) -> str:
    title = RESOURCE_TITLES.get(str(resource.get("kind")), _text(resource.get("kind")))
    value = _num(resource.get("value"))
    if resource.get("evidence") == "threshold":
        series = _text(resource.get("series"), "")
        return f"{title}: p95 {value} > порог {_num(resource.get('threshold'))}" + (f", серия «{series}»" if series else "")
    growth = _signed(resource.get("growth_pct"), " %")
    return f"{title}: рост {growth} от первой ступени (p95 {value})"


def saturation_rows(report: Mapping[str, Any]) -> Rows:
    saturation = _mapping(report.get("saturation"))
    peak = _mapping(report.get("peak_performance") or report.get("peak_perfomance"))
    rows: Rows = [
        ("Точка насыщения", f"{_text(saturation.get('saturation_rps') or peak.get('max_rps'))} RPS"),
    ]
    if saturation.get("saturation_step_label"):
        rows.append(("Ступень насыщения", _text(saturation.get("saturation_step_label"))))
    if peak.get("max_time"):
        rows.append(("Время пиковой производительности", _text(peak.get("max_time"))))
    if saturation.get("degraded_step_label") or saturation.get("first_degraded_rps") is not None:
        label = saturation.get("degraded_step_label") or f"≈{_num(saturation.get('first_degraded_rps'), 0)} RPS"
        rows.append(("Первая деградировавшая ступень", _text(label)))
    checks = [str(item) for item in saturation.get("degraded_checks") or []]
    if checks:
        rows.append(("Нарушенные критерии", ", ".join(checks)))
    rows.append(("Характер деградации", _text(saturation.get("failure_mode_title") or saturation.get("failure_mode"))))
    resource = _mapping(saturation.get("limiting_resource"))
    rows.append(("Ограничивающий ресурс", _resource_text(resource) if resource else "не определён"))
    if peak.get("drop_time") or saturation.get("rps_drop_iso"):
        rows.append(("Время деградации", _text(peak.get("drop_time") or saturation.get("rps_drop_iso"))))
    if saturation.get("note"):
        rows.append(("Примечание", _text(saturation.get("note"))))
    return rows


def test_type_section(report: Any) -> Optional[Tuple[str, Rows]]:
    """(title, rows) of the test-type section, or None for step reports (peak table instead)."""
    report = report if isinstance(report, Mapping) else {}
    mode = report_mode(report)
    if mode == "stability":
        return "Стабильность под нагрузкой", stability_rows(report)
    if mode == "spike":
        return "Всплески нагрузки", spike_rows(report)
    if mode == "stress" and isinstance(report.get("saturation"), Mapping):
        return "Точка насыщения", saturation_rows(report)
    return None


# Imported into test modules under its own name; keep pytest from collecting it.
test_type_section.__test__ = False  # type: ignore[attr-defined]

__all__ = ["report_mode", "saturation_rows", "spike_rows", "stability_rows", "test_type_section"]
