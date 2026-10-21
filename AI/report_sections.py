# SPDX-License-Identifier: Apache-2.0
"""Test-type section of a final report as plain (label, text) rows for Confluence renderers.

The web page (static/js/reports.js) renders the same sections in JavaScript; keep both in step.

- soak   → «Стабильность под нагрузкой» from ``stability_under_load`` (v1 payloads have only
  target_rps / actual_rps / sla_summary / focus and still render);
- spike  → «Всплески нагрузки» from ``spike_analysis``;
- stress → «Точка насыщения» from ``saturation`` (plus peak times from ``peak_performance``).

Labels, titles and the texts of stored codes (test type, focus, notes, failure mode, metric names) are
dictionary texts (``report.*``) in the current locale of ``AI.i18n``: the request locale on the web,
``with_locale(report_language)`` when Confluence publishes. Texts stored with the report (``sla_summary``,
step and spike labels, notes of reports saved before the codes) are shown as they are.

Old reports without a test-type payload return None, and the caller falls back to the peak table.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from AI.i18n import get_locale, t
from AI.saturation import failure_mode_title, saturation_note_text
from AI.spike import spike_note_text
from AI.stability import hold_note_text
from AI.test_profile import mode_of, stored_type_label, test_type_text

Rows = List[Tuple[str, str]]

# Metric kinds with a name in the dictionary (report.metric.<kind>): trend rows and the limiting resource.
METRIC_KINDS = ("rps", "p95", "error_rate", "cpu", "memory")
RESOURCE_KINDS = ("cpu", "memory")


def metric_title(kind: Any) -> str:
    """Name of a metric kind in the current locale; '' for an unknown kind."""
    key = str(kind or "")
    return t(f"report.metric.{key}") if key in METRIC_KINDS else ""


def _resource_title(kind: Any) -> str:
    return metric_title(kind) if str(kind or "") in RESOURCE_KINDS else ""


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
    return _text(stored_type_label(profile, get_locale()) or profile.get("test_type"), fallback)


def _type_focus(report: Mapping[str, Any], stored: Any) -> str:
    """Focus of the test type by ``normalized_test_type``; the saved text for a report without a known type."""
    profile = _mapping(report.get("test_profile"))
    return test_type_text(profile.get("normalized_test_type"), "focus", get_locale()) or _text(stored, "")


def _note(stored: Mapping[str, Any], text_of_code: Any) -> str:
    """Note of a payload: rendered from its code, else the text saved by older reports."""
    code = stored.get("note_codes") if "note_codes" in stored else stored.get("note_code")
    rendered = text_of_code(code, get_locale()) if code else ""
    return rendered or _text(stored.get("note"), "")


def stability_rows(report: Mapping[str, Any]) -> Rows:
    stability = _mapping(report.get("stability_under_load"))
    profile = _mapping(report.get("test_profile"))
    rows: Rows = [
        (t("report.section.test_type"), _text(profile.get("test_type"), "stability/soak")),
        (t("report.section.focus"), _text(_type_focus(report, stability.get("focus") or profile.get("focus")), t("report.section.focus_default_soak"))),
        (t("report.section.target_rps"), _text(stability.get("target_rps"))),
        (t("report.section.actual_rps"), _text(stability.get("actual_rps"))),
    ]
    hold = _mapping(stability.get("hold"))
    if hold:
        window = f"{_hhmm(hold.get('start_iso'))}–{_hhmm(hold.get('end_iso'))}"
        duration = _num(hold.get("duration_min"))
        rows.append((t("report.section.hold_window"), t("report.section.hold_window_minutes", span=window, minutes=duration) if duration else window))
        lowest = _num(hold.get("min_bucket_rps"))
        if lowest is not None:
            drop = _num(hold.get("drop_pct"))
            label = _text(hold.get("min_bucket_label"), t("report.section.interval"))
            rows.append((
                t("report.section.min_interval"),
                t("report.section.min_interval_drop", label=label, rps=lowest, drop=drop) if drop else f"{label}: {lowest} RPS",
            ))
        if hold.get("early_drop_iso"):
            rows.append((t("report.section.early_drop"), _hhmm(hold.get("early_drop_iso"))))
        hold_note = _note(hold, hold_note_text)
        if hold_note:
            rows.append((t("report.section.note"), hold_note))
    trends = [item for item in stability.get("trends") or [] if isinstance(item, Mapping)]
    if trends:
        parts = []
        for trend in trends:
            change = _signed(trend.get("change_pct"), " %")
            if change is None:
                continue
            # The metric code first: the stored title is in the language the report was built in.
            title = metric_title(trend.get("metric_kind")) or trend.get("title") or trend.get("metric_kind")
            parts.append(t("report.section.trend_change_risk", title=f"{title}", change=change) if trend.get("flagged") else f"{title}: {change}")
        if parts:
            rows.append((t("report.trends.title"), "; ".join(parts)))
    rows.append((t("report.section.sla_summary"), _text(stability.get("sla_summary"))))
    return rows


def spike_rows(report: Mapping[str, Any]) -> Rows:
    analysis = _mapping(report.get("spike_analysis"))
    rows: Rows = [(t("report.section.test_type"), _type_label(report, "spike"))]
    baseline = _num(analysis.get("baseline_rps"))
    if baseline is not None:
        rows.append((t("report.section.baseline"), f"{baseline} RPS"))
    if not analysis.get("detected"):
        rows.append((t("report.section.spike"), _text(analysis.get("reason"), t("report.section.spike_not_detected"))))
        rows.append((t("report.section.sla_summary"), _text(analysis.get("sla_summary"))))
        return rows
    target = _num(analysis.get("target_rps"))
    if target is not None:
        reached = analysis.get("target_reached")
        if reached is None:
            value = f"{target}"
        else:
            value = t("report.section.target_reached", rps=target) if reached else t("report.section.target_missed", rps=target)
        rows.append((t("report.section.spike_target_rps"), value))
    for spike in analysis.get("spikes") or []:
        if not isinstance(spike, Mapping):
            continue
        pre = _mapping(spike.get("pre"))
        during = _mapping(spike.get("during"))
        span = f"{_hhmm(spike.get('start_iso'))}–{_hhmm(spike.get('end_iso'))}"
        bits = [t("report.section.spike_window", span=span, seconds=f"{_num(spike.get('duration_sec'), 0)}")]
        if pre.get("p95") is not None and during.get("p95") is not None:
            overshoot = _signed(spike.get("overshoot_p95_pct"), " %")
            bits.append(f"p95 {_num(pre.get('p95'), 3)} → {_num(during.get('p95'), 3)}" + (f" ({overshoot})" if overshoot else ""))
        errors = _signed(spike.get("overshoot_error_rate_pp"), " " + t("report.unit.pp"))
        if errors:
            bits.append(t("report.section.spike_errors", value=errors))
        if spike.get("recovered") is True:
            bits.append(t("report.section.spike_recovered", seconds=f"{_num(spike.get('recovery_time_sec'), 0)}"))
        elif spike.get("recovered") is False:
            bits.append(t("report.section.spike_not_recovered"))
        spike_note = _note(spike, spike_note_text)
        if spike_note:
            bits.append(spike_note)
        rows.append((_text(spike.get("label"), t("report.section.spike_fallback")), ", ".join(bits)))
    rows.append((t("report.section.sla_summary"), _text(analysis.get("sla_summary"))))
    return rows


def _resource_text(resource: Mapping[str, Any]) -> str:
    title = _resource_title(resource.get("kind")) or _text(resource.get("kind"))
    value = f"{_num(resource.get('value'))}"
    if resource.get("evidence") == "threshold":
        series = _text(resource.get("series"), "")
        threshold = f"{_num(resource.get('threshold'))}"
        if series:
            return t("report.resource.threshold_series", title=title, value=value, op=">", threshold=threshold, series=series)
        return t("report.resource.threshold", title=title, value=value, op=">", threshold=threshold)
    growth = f"{_signed(resource.get('growth_pct'), ' %')}"
    return t("report.resource.growth", title=title, growth=growth, value=value)


def saturation_rows(report: Mapping[str, Any]) -> Rows:
    saturation = _mapping(report.get("saturation"))
    peak = _mapping(report.get("peak_performance") or report.get("peak_perfomance"))
    rows: Rows = [
        (t("report.section.saturation_point"), f"{_text(saturation.get('saturation_rps') or peak.get('max_rps'))} RPS"),
    ]
    if saturation.get("saturation_step_label"):
        rows.append((t("report.section.saturation_step"), _text(saturation.get("saturation_step_label"))))
    if peak.get("max_time"):
        rows.append((t("report.section.peak_time"), _text(peak.get("max_time"))))
    if saturation.get("degraded_step_label") or saturation.get("first_degraded_rps") is not None:
        label = saturation.get("degraded_step_label") or f"≈{_num(saturation.get('first_degraded_rps'), 0)} RPS"
        rows.append((t("report.section.first_degraded_step"), _text(label)))
    checks = [str(item) for item in saturation.get("degraded_checks") or []]
    if checks:
        rows.append((t("report.section.broken_checks"), ", ".join(checks)))
    mode_title = failure_mode_title(saturation.get("failure_mode"), get_locale())
    rows.append((t("report.section.failure_mode"), _text(mode_title or saturation.get("failure_mode_title") or saturation.get("failure_mode"))))
    resource = _mapping(saturation.get("limiting_resource"))
    rows.append((t("report.section.limiting_resource"), _resource_text(resource) if resource else t("report.section.not_determined")))
    if peak.get("drop_time") or saturation.get("rps_drop_iso"):
        rows.append((t("report.section.drop_time"), _text(peak.get("drop_time") or saturation.get("rps_drop_iso"))))
    saturation_note = _note(saturation, saturation_note_text)
    if saturation_note:
        rows.append((t("report.section.note"), saturation_note))
    return rows


def test_type_section(report: Any) -> Optional[Tuple[str, Rows]]:
    """(title, rows) of the test-type section, or None for step reports (peak table instead)."""
    report = report if isinstance(report, Mapping) else {}
    mode = report_mode(report)
    if mode == "stability":
        return t("report.section.title_stability"), stability_rows(report)
    if mode == "spike":
        return t("report.section.title_spike"), spike_rows(report)
    if mode == "stress" and isinstance(report.get("saturation"), Mapping):
        return t("report.section.title_saturation"), saturation_rows(report)
    return None


# Imported into test modules under its own name; keep pytest from collecting it.
test_type_section.__test__ = False  # type: ignore[attr-defined]

__all__ = ["metric_title", "report_mode", "saturation_rows", "spike_rows", "stability_rows", "test_type_section"]
