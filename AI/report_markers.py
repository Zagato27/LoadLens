"""Chart markers, SLA thresholds and series colors of a report, shared by the web page and Confluence.

The report page computes the same lists in ``static/js/reports.js`` (``reportMarkers``,
``slaThresholds``, ``orderSeries``); ``tests/test_report_markers.py`` runs both sides on the demo
row and on synthetic stress, soak and spike rows and requires identical output. Change both sides
together.

Times come only from deterministic data: the load-step table, the SLA window and the test-type
payloads (``saturation``, ``stability_under_load``, ``spike_analysis``). ``peak_performance.max_time``
and ``drop_time`` are LLM text and are never used.

Marker::

    {"kind": "band" | "line" | "tick", "t": epoch ms, "to": epoch ms | None,
     "label": str, "tone": "accent" | "ok" | "warn" | "fail" | "muted", "role": str}

Threshold::

    {"name": check name, "query_label": str, "series": str, "value": float, "label": str, "tone": str}
"""

from __future__ import annotations

import bisect
import colorsys
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Light-theme series palette, copied from static/css/app.css (:root[data-theme="light"] --series-1..8).
# Confluence pages are light, so PNG charts use these values; the web page reads the same tokens
# through LL.charts.palette, so a series gets the same slot color in both.
SERIES_PALETTE: Tuple[str, ...] = ("#026fd7", "#ad3d9a", "#706d06", "#0d8ca3", "#6f3680", "#7a4306", "#8d6cc2", "#035397")
# Light-theme marker and threshold tones (app.css: --accent, --status-ok/warn/fail, --text-dim).
TONE_COLORS: Dict[str, str] = {
    "accent": "#6c5ce7",
    "ok": "#16a34a",
    "warn": "#d97706",
    "fail": "#dc2626",
    "muted": "#666e7c",
}
DEFAULT_OFFSET_HOURS = 3.0
# An RPS drop closer than this to the degradation line is the same event and gets no second line.
DROP_MERGE_MS = 60_000

THRESHOLD_SPECS: Dict[str, Tuple[str, str]] = {
    "target_rps": ("цель ≥ {} RPS", "accent"),
    "p95_latency": ("p95 ≤ {} мс", "fail"),
    "p99_latency": ("p99 ≤ {} мс", "fail"),
    "error_rate": ("ошибки ≤ {} %", "fail"),
    "cpu_usage": ("CPU ≤ {} %", "fail"),
    "memory_usage": ("память ≤ {} %", "fail"),
}
_LABEL_RE = re.compile(r"label='([^']*)'")
_QUERY_RE = re.compile(r"запрос «([^»]*)»")
_SERIES_EN_RE = re.compile(r"series='([^']*)'")
_SERIES_RU_RE = re.compile(r"серия «([^»]*)»")
_OFFSET_RE = re.compile(r"([+-])(\d{2}):(\d{2})$")


# ---- row access ---------------------------------------------------------------------------

def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def sla_checks(final_row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """``sla_details.checks``, or the checks of older reports in ``parsed.deterministic_sla``."""
    checks = _as_dict(final_row.get("sla_details")).get("checks")
    if not (isinstance(checks, list) and checks):
        checks = _as_dict(_as_dict(final_row.get("parsed")).get("deterministic_sla")).get("checks")
    return [item for item in checks if isinstance(item, Mapping)] if isinstance(checks, list) else []


def sla_window(final_row: Mapping[str, Any]) -> Dict[str, Any]:
    window = _as_dict(final_row.get("sla_details")).get("stable_window")
    if not isinstance(window, Mapping):
        window = _as_dict(_as_dict(final_row.get("parsed")).get("deterministic_sla")).get("stable_window")
    return dict(window) if isinstance(window, Mapping) else {}


def embedded_step_table(final_row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    for source in ("sla_details", "parsed"):
        table = _as_dict(final_row.get(source)).get("load_step_table")
        if isinstance(table, Mapping) and isinstance(table.get("steps"), list) and table.get("steps"):
            return dict(table)
    return None


def _table_of(final_row: Mapping[str, Any], step_table: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(step_table, Mapping) and isinstance(step_table.get("steps"), list) and step_table.get("steps"):
        return dict(step_table)
    return embedded_step_table(final_row) or {}


def _steps_of(table: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [step for step in (table.get("steps") or []) if isinstance(step, Mapping)]


# ---- time ---------------------------------------------------------------------------------

def moment_ms(value: Any) -> Optional[int]:
    """Epoch milliseconds of an ISO string; a string without a zone is UTC."""
    text = _text(value).replace(" ", "T", 1)
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(round(moment.timestamp() * 1000))


def offset_hours(steps: Sequence[Mapping[str, Any]]) -> float:
    """Report clock offset from the step table ISO strings (they carry it), else the default."""
    sample = next((_text(step.get("end_iso") or step.get("start_iso")) for step in steps if step.get("end_iso") or step.get("start_iso")), "")
    match = _OFFSET_RE.search(sample)
    if not match:
        return DEFAULT_OFFSET_HOURS
    sign = -1 if match.group(1) == "-" else 1
    return sign * (int(match.group(2)) + int(match.group(3)) / 60)


def clock(ms: int, offset: float) -> str:
    moment = datetime.fromtimestamp(ms / 1000, tz=timezone.utc) + timedelta(hours=offset)
    return moment.strftime("%H:%M")


# ---- numbers ------------------------------------------------------------------------------

def _finite(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def number_text(value: Any) -> str:
    """Number as the report page shows it: integers as is, otherwise up to two decimals."""
    number = _finite(value)
    if number is None:
        return "—"
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


# ---- markers ------------------------------------------------------------------------------

def _marker(kind: str, t: int, to: Optional[int], label: str, tone: str, role: str) -> Dict[str, Any]:
    return {"kind": kind, "t": t, "to": to, "label": label, "tone": tone, "role": role}


def _band(start: Any, end: Any, label: str, tone: str, role: str) -> Optional[Dict[str, Any]]:
    a, b = moment_ms(start), moment_ms(end)
    if a is None or b is None:
        return None
    if b < a:
        a, b = b, a
    return _marker("band", a, b, label, tone, role)


def marker_mode(parsed: Mapping[str, Any], window: Mapping[str, Any]) -> str:
    """``spike``, ``hold`` (soak) or ``step`` (capacity and stress)."""
    kind = _text(window.get("kind")).lower()
    if isinstance(parsed.get("spike_analysis"), Mapping) or kind == "spike":
        return "spike"
    profile = _as_dict(parsed.get("test_profile"))
    if (
        isinstance(parsed.get("stability_under_load"), Mapping)
        or kind == "hold"
        or _text(profile.get("mode")).lower() in {"stability", "soak"}
    ):
        return "hold"
    return "step"


def _step_range(step: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    return (
        moment_ms(step.get("plateau_start_iso") or step.get("start_iso")),
        moment_ms(step.get("plateau_end_iso") or step.get("end_iso")),
    )


def _covers(step: Mapping[str, Any], ms: Optional[int]) -> bool:
    if ms is None:
        return False
    start, end = _step_range(step)
    return start is not None and end is not None and start <= ms <= end


def _check(checks: Sequence[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    return dict(next((item for item in checks if _text(item.get("name")) == name), {}))


def chosen_step(steps: Sequence[Mapping[str, Any]], window: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Step the SLA was checked on: the marked one, the one under the window start, else by target RPS."""
    selected = next((step for step in steps if step.get("selected")), None)
    if selected is not None:
        return dict(selected)
    start = moment_ms(window.get("start") or window.get("start_iso"))
    covered = next((step for step in steps if _covers(step, start)), None)
    if covered is not None:
        return dict(covered)
    target = _finite(_check(checks, "target_rps").get("actual"))
    if target is None:
        return None
    matches = [step for step in steps if _finite(step.get("rps")) is not None and abs(float(step["rps"]) - target) <= 0.51]
    stable = [step for step in matches if step.get("stable") is not False]
    pool = stable or matches
    return dict(pool[-1]) if pool else None


def _position(steps: Sequence[Mapping[str, Any]], step: Optional[Mapping[str, Any]]) -> int:
    if step is None:
        return -1
    for i, item in enumerate(steps):
        if item.get("index") == step.get("index") and item.get("start_iso") == step.get("start_iso"):
            return i
    return -1


def _step_markers(parsed: Mapping[str, Any], window: Mapping[str, Any], checks: Sequence[Mapping[str, Any]], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = offset_hours(steps)
    saturation = parsed.get("saturation") if isinstance(parsed.get("saturation"), Mapping) else None
    chosen = chosen_step(steps, window, checks)
    kind = _text(window.get("kind")).lower() or "step"
    label = f"SLA: ступень {_text(chosen.get('index'))}" if chosen and _text(chosen.get("index")) else "SLA: окно"
    band = None
    if window and kind == "step":
        band = _band(window.get("start") or window.get("start_iso"), window.get("end") or window.get("end_iso"), label, "accent", "sla")
    if band is None and chosen is not None:
        band = _band(chosen.get("plateau_start_iso") or chosen.get("start_iso"), chosen.get("plateau_end_iso") or chosen.get("end_iso"), label, "accent", "sla")
    if band is None and saturation is not None:
        sat_window = _as_dict(saturation.get("saturation_window"))
        band = _band(sat_window.get("start_iso"), sat_window.get("end_iso"), label, "accent", "sla")
    if band is not None:
        out.append(band)
    for step in steps[1:]:
        t = moment_ms(step.get("start_iso"))
        if t is not None:
            out.append(_marker("tick", t, None, "", "muted", "step"))
    degradation_ms: Optional[int] = None
    if window and kind == "step" and (window.get("degraded_level") is not None or window.get("degraded_checks")):
        position = _position(steps, chosen)
        following = steps[position + 1] if 0 <= position < len(steps) - 1 else None
        degradation_ms = moment_ms(following.get("start_iso")) if following else moment_ms(window.get("end") or window.get("end_iso"))
        if degradation_ms is not None:
            out.append(_marker("line", degradation_ms, None, f"деградация {clock(degradation_ms, offset)}", "fail", "degradation"))
    drop_ms = moment_ms(saturation.get("rps_drop_iso")) if saturation is not None else None
    if drop_ms is None:
        after_drop = next((step for step in steps if step.get("after_drop") is True), None)
        drop_ms = moment_ms(after_drop.get("start_iso")) if after_drop else None
    if drop_ms is not None and (degradation_ms is None or abs(drop_ms - degradation_ms) > DROP_MERGE_MS):
        out.append(_marker("line", drop_ms, None, f"падение RPS {clock(drop_ms, offset)}", "fail", "rps_drop"))
    return out


def _hold_markers(parsed: Mapping[str, Any], window: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hold = _as_dict(_as_dict(parsed.get("stability_under_load")).get("hold"))
    band = _band(hold.get("start_iso"), hold.get("end_iso"), "окно удержания", "accent", "hold")
    if band is None and _text(window.get("kind")).lower() == "hold":
        band = _band(window.get("start") or window.get("start_iso"), window.get("end") or window.get("end_iso"), "окно удержания", "accent", "hold")
    return [band] if band else []


def _spike_markers(parsed: Mapping[str, Any], window: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    spikes = _as_dict(parsed.get("spike_analysis")).get("spikes")
    for position, spike in enumerate(spikes if isinstance(spikes, list) else []):
        if not isinstance(spike, Mapping):
            continue
        label = _text(spike.get("label")) or f"всплеск {_text(spike.get('index')) or position + 1}"
        band = _band(spike.get("start_iso"), spike.get("end_iso"), label, "warn", "spike")
        if band:
            out.append(band)
    if not out and _text(window.get("kind")).lower() == "spike":
        band = _band(window.get("start") or window.get("start_iso"), window.get("end") or window.get("end_iso"), _text(window.get("title")) or "всплеск", "warn", "spike")
        if band:
            out.append(band)
    return out


def markers_for(final_row: Mapping[str, Any], step_table: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Markers drawn on every chart of the report.

    Step and stress tests: the SLA step as a band («SLA: ступень N»), step boundaries as ticks,
    the start of the step after the SLA step when it broke the thresholds («деградация HH:MM») and
    the RPS drop (stress ``rps_drop_iso`` or the first ``after_drop`` step, «падение RPS HH:MM»).
    Soak: the hold window. Spike: every spike window. ``step_table`` is the resolved load-step
    table (``resolve_load_step_table``); without it the table saved with the row is used.
    """
    row = final_row if isinstance(final_row, Mapping) else {}
    parsed = _as_dict(row.get("parsed"))
    window = sla_window(row)
    mode = marker_mode(parsed, window)
    if mode == "spike":
        return _spike_markers(parsed, window)
    if mode == "hold":
        return _hold_markers(parsed, window)
    steps = _steps_of(_table_of(row, step_table))
    return _step_markers(parsed, window, sla_checks(row), steps)


# ---- thresholds ---------------------------------------------------------------------------

def _message_hint(message: str) -> Tuple[str, str]:
    label = _LABEL_RE.search(message) or _QUERY_RE.search(message)
    series = _SERIES_EN_RE.search(message) or _SERIES_RU_RE.search(message)
    return (label.group(1) if label else ""), (series.group(1) if series else "")


def check_source(check: Mapping[str, Any], window: Mapping[str, Any], table: Mapping[str, Any]) -> Tuple[str, str]:
    """Query label and series a check was measured on.

    Explicit ``query_label``/``series`` fields first (AI/sla_evaluator), then the ``message`` text,
    then the SLA window (target RPS) and the step table (RPS and latency queries).
    """
    name = _text(check.get("name"))
    label, series = _text(check.get("query_label")), _text(check.get("series"))
    if not label:
        hint_label, hint_series = _message_hint(_text(check.get("message")))
        label, series = hint_label, series or hint_series
    if not label and name == "target_rps":
        label = _text(window.get("label")) or _text(table.get("rps_query"))
        series = series or (_text(window.get("series")) if _text(window.get("label")) else _text(table.get("rps_series")))
    if not label and name in {"p95_latency", "p99_latency"}:
        label = _text(table.get("latency_query"))
        series = series or _text(table.get("latency_series"))
    return label, series


def thresholds_for(final_row: Mapping[str, Any], step_table: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """SLA thresholds with the query they belong to; a chart draws those of its own query."""
    row = final_row if isinstance(final_row, Mapping) else {}
    window = sla_window(row)
    table = _table_of(row, step_table)
    out: List[Dict[str, Any]] = []
    for check in sla_checks(row):
        name = _text(check.get("name"))
        spec = THRESHOLD_SPECS.get(name)
        value = _finite(check.get("threshold"))
        if spec is None or value is None:
            continue
        label, series = check_source(check, window, table)
        if not label:
            continue
        out.append({
            "name": name,
            "query_label": label,
            "series": series,
            "value": value,
            "label": spec[0].format(number_text(value)),
            "tone": spec[1],
        })
    return out


def resolve_query_label(label: str, all_labels: Optional[Iterable[str]]) -> str:
    """Chart label a check label refers to: the exact one, else the only label containing it
    (case-insensitive, as AI/sla_evaluator finds the SLA query), else the label itself."""
    text = _text(label)
    labels = [_text(item) for item in (all_labels or []) if _text(item)]
    if not text or not labels:
        return text
    lower = text.lower()
    exact = [item for item in labels if item.lower() == lower]
    if len(exact) == 1:
        return exact[0]
    partial = [item for item in labels if lower in item.lower()]
    return partial[0] if len(partial) == 1 else text


def thresholds_for_query(
    thresholds: Iterable[Mapping[str, Any]],
    query_label: str,
    all_labels: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Thresholds drawn on the chart of ``query_label``; ``all_labels`` are the run's chart labels."""
    target = _text(query_label)
    labels = list(all_labels or [])
    return [dict(item) for item in thresholds if target and resolve_query_label(_text(item.get("query_label")), labels) == target]


# ---- series colors ------------------------------------------------------------------------

def _js_round(value: float) -> int:
    return int(math.floor(value + 0.5))


def series_color(index: int) -> str:
    """Color of series slot ``index``; mirrors LL.charts.palette on the light theme, cycles past 8."""
    n = max(0, int(index or 0))
    slot, cycle = n % len(SERIES_PALETTE), n // len(SERIES_PALETTE)
    base = SERIES_PALETTE[slot]
    if not cycle:
        return base
    r, g, b = (int(base[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    step = (cycle + 1) // 2
    toward = 0.12 if cycle % 2 == 1 else 0.78
    lightness += (toward - lightness) * (1 - 0.65 ** step)
    channels = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#" + "".join(f"{max(0, min(255, _js_round(c * 255))):02x}" for c in channels)


def order_series(values_by_series: Mapping[str, Iterable[Any]]) -> List[str]:
    """Series in color order: by the absolute mean, largest first, then by name (as the web page)."""
    def mean(values: Iterable[Any]) -> float:
        # A plain left-to-right sum, as in JS (sum() compensates rounding since Python 3.12).
        total, count = 0.0, 0
        for value in values:
            number = _finite(value)
            if number is not None:
                total += number
                count += 1
        return abs(total / count) if count else -1.0
    return sorted(values_by_series, key=lambda name: (-mean(values_by_series[name]), str(name)))


# ---- category-axis placement --------------------------------------------------------------

def fractional_index(times: Sequence[float], value: float) -> float:
    """Position of a time on a category axis of sorted bucket times (as LL.charts places markers)."""
    n = len(times)
    if n == 1:
        return 0.0
    if value <= times[0]:
        return (value - times[0]) / ((times[1] - times[0]) or 1)
    if value >= times[-1]:
        return n - 1 + (value - times[-1]) / ((times[-1] - times[-2]) or 1)
    i = bisect.bisect_right(times, value) - 1
    return i + (value - times[i]) / ((times[i + 1] - times[i]) or 1)


__all__ = [
    "DEFAULT_OFFSET_HOURS",
    "SERIES_PALETTE",
    "THRESHOLD_SPECS",
    "TONE_COLORS",
    "check_source",
    "chosen_step",
    "clock",
    "embedded_step_table",
    "fractional_index",
    "marker_mode",
    "markers_for",
    "moment_ms",
    "number_text",
    "offset_hours",
    "order_series",
    "resolve_query_label",
    "series_color",
    "sla_checks",
    "sla_window",
    "thresholds_for",
    "thresholds_for_query",
]
