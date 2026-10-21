# SPDX-License-Identifier: Apache-2.0
"""Soak (stability) analysis: the window the target load was held in and metric trends over it.

The hold window is the run without the ramp-up and the final ramp-down: from the first to
the last point where the designated RPS (``sla.max_performance_query``) stays at
``HOLD_LEVEL_FRACTION`` of its median. A fall long before the end of the test is not a
ramp-down, so such a tail stays inside the window and fails the hold check.

The module constants are defaults; they are candidates for future ``sla.*`` keys
(``sla.hold_tolerance_pct``, ``sla.hold_drop_tolerance_pct``) once the config needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from AI.context_pack import DEFAULT_DRIFT_PCT, LoadStep, _series_from_labeled, format_iso, to_utc
from AI.i18n import get_locale, t
from AI.report_texts import LanguageTable, LocaleTable, text_lang

HOLD_LEVEL_FRACTION = 0.9
HOLD_MIN_SHARE = 0.5
HOLD_TAIL_ALLOWANCE_MIN = 3.0
HOLD_TAIL_SHARE = 0.05
HOLD_TOLERANCE_PCT = 5.0
HOLD_DROP_TOLERANCE_PCT = 15.0
TREND_MIN_SAMPLES = 8
SMOOTH_POINTS = 3

# (domain, sla query key, metric kind, statistic for the worst series, adverse direction)
TREND_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    ("lt_framework", "max_performance_query", "rps", "max", "down"),
    ("lt_framework", "p95_query", "p95", "p95", "up"),
    ("lt_framework", "error_rate_query", "error_rate", "p95", "up"),
    ("hard_resources", "cpu_query", "cpu", "p95", "up"),
    ("hard_resources", "memory_query", "memory", "p95", "up"),
)
TREND_KINDS = ("rps", "p95", "error_rate", "cpu", "memory")
# Metric names of the trends (report.metric.*) in the language of the report being built.
TREND_TITLES = LocaleTable(TREND_KINDS, lambda kind: t(f"report.metric.{kind}"))

# Notes on how the hold window was chosen, by code (``hold.note_codes``, texts under report.hold_note.*); readers
# render them in their language and fall back to the stored ``hold.note`` text of reports saved before the codes.
HOLD_NOTE_CODES = ("no_steady_level", "early_drop", "long_ramp_up")
HOLD_NOTE_TEXTS = LanguageTable(HOLD_NOTE_CODES, lambda code, lang: t(f"report.hold_note.{code}", locale=lang))


def hold_note_text(codes: Sequence[str], lang: str = "ru") -> str:
    """Text of hold-window note codes in ``lang`` (unknown languages: ru), joined with «; »."""
    lang = text_lang(lang)
    return "; ".join(
        t(f"report.hold_note.{code}", locale=lang) if str(code) in HOLD_NOTE_CODES else str(code) for code in codes or []
    )


def _bin_width(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(0)
    width = pd.Series(index).diff().dropna().median()
    return width if pd.notna(width) and width > pd.Timedelta(0) else pd.Timedelta(0)


def _mean_between(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[Optional[float], int]:
    chunk = series[(series.index >= start) & (series.index < end)].dropna()
    return (float(chunk.mean()), int(chunk.shape[0])) if not chunk.empty else (None, 0)


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


@dataclass(frozen=True)
class HoldWindow:
    """Part of a soak run where the target load was held."""

    start: pd.Timestamp
    end: pd.Timestamp
    level_rps: float
    mean_rps: float
    min_bucket_rps: Optional[float]
    min_bucket_label: Optional[str]
    series: str
    label: str
    samples: int
    whole_run: bool = False
    early_drop: Optional[pd.Timestamp] = None
    note: str = ""
    note_codes: tuple[str, ...] = ()

    @property
    def drop_pct(self) -> Optional[float]:
        """How far the weakest interval fell below the window mean, %."""
        if self.min_bucket_rps is None or self.mean_rps <= 0:
            return None
        return round(max(0.0, (1.0 - self.min_bucket_rps / self.mean_rps) * 100.0), 1)

    def to_dict(self, shift_hours: int) -> Dict[str, Any]:
        return {
            "start_iso": format_iso(self.start, shift_hours),
            "end_iso": format_iso(self.end, shift_hours),
            "start_utc": str(self.start),
            "end_utc": str(self.end),
            "duration_min": round((self.end - self.start).total_seconds() / 60.0, 1),
            "level_rps": _round(self.level_rps),
            "actual_rps_mean": _round(self.mean_rps),
            "min_bucket_rps": _round(self.min_bucket_rps),
            "min_bucket_label": self.min_bucket_label,
            "drop_pct": self.drop_pct,
            "series": self.series,
            "source_label": self.label,
            "samples": self.samples,
            "whole_run": self.whole_run,
            "early_drop_iso": format_iso(self.early_drop, shift_hours) if self.early_drop is not None else None,
            "note": self.note,
            "note_codes": list(self.note_codes),
        }


def _bucket_stats(
    values: pd.Series,
    steps: Sequence[LoadStep],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[Optional[float], Optional[str]]:
    """Lowest mean RPS among the report intervals clipped to the hold window."""
    lowest: Optional[float] = None
    lowest_label: Optional[str] = None
    for step in steps:
        left = max(to_utc(step.start), start)
        right = min(to_utc(step.end), end)
        if right <= left:
            continue
        mean, samples = _mean_between(values, left, right)
        if mean is None or samples < 2:
            continue
        if lowest is None or mean < lowest:
            lowest, lowest_label = mean, step.label
    return lowest, lowest_label


def hold_window_from_series(
    series: Optional[pd.Series],
    steps: Sequence[LoadStep],
    *,
    series_name: str = "",
    label: str = "",
) -> Optional[HoldWindow]:
    """Hold window of a soak run from the designated RPS series (UTC index)."""
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.shape[0] < 2:
        return None
    level = float(values.median())
    if level <= 0:
        return None
    width = _bin_width(pd.DatetimeIndex(values.index))
    run_start, run_last = values.index[0], values.index[-1]
    run_end = run_last + width
    total = run_end - run_start
    smooth = values.rolling(SMOOTH_POINTS, center=True, min_periods=1).median()
    held = smooth[smooth >= level * HOLD_LEVEL_FRACTION]
    start, end = run_start, run_end
    whole_run, early_drop, codes = False, None, []
    if held.empty:
        whole_run = True
        codes.append("no_steady_level")
    else:
        start, last_held = held.index[0], held.index[-1]
        end = last_held + width
        allowance = max(pd.Timedelta(minutes=HOLD_TAIL_ALLOWANCE_MIN), total * HOLD_TAIL_SHARE)
        if run_end - end > allowance:
            early_drop = end
            end = run_end
            codes.append("early_drop")
        if total > pd.Timedelta(0) and (end - start) < total * HOLD_MIN_SHARE:
            start, end, whole_run = run_start, run_end, True
            codes.append("long_ramp_up")
    mean, samples = _mean_between(values, start, end)
    if mean is None:
        return None
    lowest, lowest_label = _bucket_stats(values, steps, start, end)
    return HoldWindow(
        start=start, end=end, level_rps=level, mean_rps=mean,
        min_bucket_rps=lowest, min_bucket_label=lowest_label,
        series=series_name, label=label, samples=samples,
        whole_run=whole_run, early_drop=early_drop, note=hold_note_text(codes, get_locale()), note_codes=tuple(codes),
    )


def hold_window_from_labeled(
    lt_labeled: Sequence[Dict[str, Any]],
    perf_query: str,
    steps: Sequence[LoadStep],
) -> Optional[HoldWindow]:
    """Hold window from the lt_framework frames and ``sla.max_performance_query``."""
    if not perf_query:
        return None
    column, series, label = _series_from_labeled(list(lt_labeled or []), perf_query, "max")
    return hold_window_from_series(series, steps, series_name=column, label=label or perf_query)


def hold_check(hold: Optional[Mapping[str, Any]], target_rps: float) -> Dict[str, Any]:
    """Whether the hold window kept the target load: mean within 5 %, no interval below 15 %."""
    hold = hold if isinstance(hold, Mapping) else {}
    mean = hold.get("actual_rps_mean")
    lowest = hold.get("min_bucket_rps")
    mean_floor = target_rps * (1.0 - HOLD_TOLERANCE_PCT / 100.0)
    lowest_floor = target_rps * (1.0 - HOLD_DROP_TOLERANCE_PCT / 100.0)
    mean_ok = mean is not None and float(mean) >= mean_floor
    lowest_ok = lowest is None or float(lowest) >= lowest_floor
    passed: Optional[bool] = (mean_ok and lowest_ok) if mean is not None else None
    if mean is None:
        message = t("sla.hold.no_data")
    else:
        window = f"{str(hold.get('start_iso') or '')[11:16]}–{str(hold.get('end_iso') or '')[11:16]}"
        message = t(
            "sla.hold.mean", mean=f"{float(mean):.1f}", window=window, minutes=str(hold.get("duration_min")),
            op="≥" if mean_ok else "<", target=f"{target_rps:g}", tolerance=f"{HOLD_TOLERANCE_PCT:g}",
        )
        if lowest is not None:
            label = hold.get("min_bucket_label") or t("sla.hold.interval_fallback")
            below = (1.0 - float(lowest) / target_rps) * 100.0 if target_rps > 0 else 0.0
            message += t(
                "sla.hold.lowest", label=str(label), rps=f"{float(lowest):.1f}", change=f"{-below:+.0f}",
                tolerance=f"{HOLD_DROP_TOLERANCE_PCT:g}",
            )
    return {
        "passed": passed,
        "mean_ok": mean_ok,
        "lowest_ok": lowest_ok,
        "actual": _round(mean),
        "message": message,
        "tolerance_pct": HOLD_TOLERANCE_PCT,
        "drop_tolerance_pct": HOLD_DROP_TOLERANCE_PCT,
    }


def _trend(series: pd.Series, adverse: str, drift_pct: float) -> Optional[Dict[str, Any]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    count = int(values.shape[0])
    if count < 2:
        return None
    quarter = max(1, count // 4)
    first = float(values.iloc[:quarter].mean())
    last = float(values.iloc[-quarter:].mean())
    change_pct = (last / first - 1.0) * 100.0 if abs(first) > 1e-9 else None
    hours = (values.index - values.index[0]).total_seconds() / 3600.0
    slope = float(np.polyfit(np.asarray(hours, dtype=float), values.to_numpy(dtype=float), 1)[0]) if hours[-1] > 0 else None
    direction = "flat" if change_pct is None or abs(change_pct) < 1e-9 else ("up" if change_pct > 0 else "down")
    adverse_change = change_pct is not None and (
        change_pct > drift_pct if adverse == "up" else change_pct < -drift_pct
    )
    return {
        "first": _round(first, 3),
        "last": _round(last, 3),
        "change_pct": _round(change_pct, 1),
        "slope_per_hour": _round(slope, 3),
        "direction": direction,
        "adverse_direction": adverse,
        "samples": count,
        "flagged": bool(adverse_change and count >= TREND_MIN_SAMPLES),
    }


def compute_trends(
    domain_data: Mapping[str, Any],
    sla_cfg: Mapping[str, Any],
    hold: Optional[Mapping[str, Any]],
    drift_pct: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Change of RPS, p95, errors, CPU and memory from the start to the end of the hold window."""
    threshold = float(drift_pct) if drift_pct is not None else DEFAULT_DRIFT_PCT
    start = to_utc(hold["start_utc"]) if isinstance(hold, Mapping) and hold.get("start_utc") else None
    end = to_utc(hold["end_utc"]) if isinstance(hold, Mapping) and hold.get("end_utc") else None
    trends: List[Dict[str, Any]] = []
    for domain, query_key, kind, stat, adverse in TREND_SPECS:
        query = str((sla_cfg or {}).get(query_key) or "").strip()
        payload = domain_data.get(domain) if isinstance(domain_data, Mapping) else None
        labeled = payload.get("labeled") if isinstance(payload, Mapping) else None
        if not query or not isinstance(labeled, list):
            continue
        column, series, label = _series_from_labeled(labeled, query, stat)
        if series is None:
            continue
        if start is not None and end is not None:
            series = series[(series.index >= start) & (series.index < end)]
        trend = _trend(series, adverse, threshold)
        if trend is None:
            continue
        trends.append({
            "domain": domain,
            "metric_kind": kind,
            "title": TREND_TITLES.get(kind, kind),
            "label": label or query,
            "series": column,
            **trend,
        })
    return trends


def build_stability_payload(
    *,
    focus: str,
    hold: Optional[Mapping[str, Any]],
    trends: Sequence[Mapping[str, Any]],
    target_check: Optional[Mapping[str, Any]],
    sla_summary: Optional[str],
    drift_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """``stability_under_load`` of the final report; keeps the v1 keys target_rps/actual_rps."""
    hold_out: Optional[Dict[str, Any]] = None
    if isinstance(hold, Mapping):
        hold_out = {key: value for key, value in hold.items() if key not in {"start_utc", "end_utc"}}
        hold_out["tolerance_pct"] = HOLD_TOLERANCE_PCT
        hold_out["drop_tolerance_pct"] = HOLD_DROP_TOLERANCE_PCT
        hold_out["passed"] = (target_check or {}).get("passed") if target_check else None
    target = (target_check or {}).get("threshold") if target_check else None
    return {
        "mode": "stability",
        "focus": focus,
        "target_rps": target,
        "actual_rps": (hold or {}).get("actual_rps_mean") if isinstance(hold, Mapping) else (target_check or {}).get("actual"),
        "sla_summary": sla_summary,
        "hold": hold_out,
        "trend_threshold_pct": float(drift_pct) if drift_pct is not None else DEFAULT_DRIFT_PCT,
        "trends": [dict(item) for item in trends],
        "flagged_trends": [str(item.get("metric_kind")) for item in trends if item.get("flagged")],
    }


__all__ = [
    "HOLD_DROP_TOLERANCE_PCT",
    "HOLD_LEVEL_FRACTION",
    "HOLD_NOTE_CODES",
    "HOLD_NOTE_TEXTS",
    "HOLD_TOLERANCE_PCT",
    "HoldWindow",
    "TREND_KINDS",
    "TREND_SPECS",
    "TREND_TITLES",
    "build_stability_payload",
    "compute_trends",
    "hold_check",
    "hold_note_text",
    "hold_window_from_labeled",
    "hold_window_from_series",
]
