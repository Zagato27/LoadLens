# SPDX-License-Identifier: Apache-2.0
"""Spike test analysis: bursts of the designated RPS, overshoot of latency and errors, recovery.

Detection works on the designated RPS series (``sla.max_performance_query``). The background
level is the median of the run, so spikes must take less than half of the test. A spike is a
contiguous stretch where the smoothed RPS stays at or above
``max(background × SPIKE_RATIO, background + SPIKE_MIN_DELTA_RPS)`` for at least
``SPIKE_MIN_DURATION_SEC`` and ``SPIKE_MIN_POINTS`` samples; short gaps inside a burst are merged.

For every spike the latency p95 and the error rate inside the burst are compared with the
background right before it, and recovery is the first moment after the burst when latency is
back within ``RECOVERY_TOLERANCE_PCT`` of that background (confirmed by the next point).

The module constants are defaults and candidates for future ``sla.*`` keys
(``sla.spike_ratio``, ``sla.recovery_tolerance_pct``, ``sla.recovery_max_window_min``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from AI.context_pack import LoadStep, _series_from_labeled, _utc_numeric, format_iso, minutes_from, to_utc
from AI.i18n import get_locale, t
from AI.report_texts import LanguageTable, text_lang
from AI.sla_evaluator import _labeled_frame, _worst_series_p95

SPIKE_RATIO = 1.5
SPIKE_MIN_DELTA_RPS = 10.0
SPIKE_MIN_DURATION_SEC = 60.0
SPIKE_MIN_POINTS = 2  # a single sample is an outlier, not a burst
SPIKE_MERGE_GAP_SEC = 30.0
SPIKE_SMOOTH_SEC = 45.0
PRE_SPIKE_WINDOW_MIN = 5.0
RECOVERY_TOLERANCE_PCT = 10.0
RECOVERY_MAX_WINDOW_MIN = 15.0
RPS_RECOVERY_WINDOW_MIN = 2.0
MAX_SPIKES = 5

# Notes of a spike by code (``spike.note_code``, texts under report.spike_note.*); readers render them in their
# language and fall back to the stored ``note`` text of reports saved before the codes.
SPIKE_NOTE_CODES = ("no_latency_after", "no_latency_query")
SPIKE_NOTE_TEXTS = LanguageTable(SPIKE_NOTE_CODES, lambda code, lang: t(f"report.spike_note.{code}", locale=lang))


def spike_note_text(code: Optional[str], lang: str = "ru") -> str:
    """Text of a spike note code in ``lang`` (unknown languages: ru); '' without a code."""
    if not code:
        return ""
    if str(code) not in SPIKE_NOTE_CODES:
        return str(code)
    return t(f"report.spike_note.{code}", locale=text_lang(lang))


@dataclass(frozen=True)
class Spike:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    level_rps: float
    peak_rps: float

    @property
    def duration_sec(self) -> float:
        return round((self.end - self.start).total_seconds(), 1)

    @property
    def label(self) -> str:
        """«всплеск 1 · ≈480 RPS» / «spike 1 · ≈480 RPS» in the language of the report being built."""
        return t("report.step_label.spike", index=self.index, level=f"{self.level_rps:.0f}")


@dataclass(frozen=True)
class SpikeProfile:
    baseline_rps: float
    threshold_rps: float
    series: str
    label: str
    spikes: tuple[Spike, ...]
    values: pd.Series = field(repr=False, compare=False, default_factory=pd.Series)


def _bin_width(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(0)
    width = pd.Series(index).diff().dropna().median()
    return width if pd.notna(width) and width > pd.Timedelta(0) else pd.Timedelta(0)


def _runs(mask: pd.Series) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous True stretches as (first point, last point)."""
    runs: List[tuple[pd.Timestamp, pd.Timestamp]] = []
    first: Optional[pd.Timestamp] = None
    previous: Optional[pd.Timestamp] = None
    for moment, flag in mask.items():
        if flag and first is None:
            first = moment
        if not flag and first is not None:
            runs.append((first, previous))
            first = None
        previous = moment
    if first is not None:
        runs.append((first, previous))
    return runs


def detect_spikes(series: Optional[pd.Series], *, series_name: str = "", label: str = "") -> Optional[SpikeProfile]:
    """Spikes of the designated RPS series (UTC index); None when the series is too short."""
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.shape[0] < 4:
        return None
    width = _bin_width(pd.DatetimeIndex(values.index))
    width_sec = width.total_seconds() or 1.0
    points = max(1, int(round(SPIKE_SMOOTH_SEC / width_sec)))
    points = points if points % 2 == 1 else points + 1
    smooth = values.rolling(points, center=True, min_periods=1).median() if points > 1 else values
    baseline = float(values.median())
    threshold = max(baseline * SPIKE_RATIO, baseline + SPIKE_MIN_DELTA_RPS)
    merged: List[List[pd.Timestamp]] = []
    for first, last in _runs(smooth >= threshold):
        if merged and (first - merged[-1][1]).total_seconds() <= SPIKE_MERGE_GAP_SEC + width_sec:
            merged[-1][1] = last
        else:
            merged.append([first, last])
    candidates = []
    for first, last in merged:
        end = last + width
        chunk = values[(values.index >= first) & (values.index <= last)]
        if (end - first).total_seconds() < SPIKE_MIN_DURATION_SEC or chunk.shape[0] < SPIKE_MIN_POINTS:
            continue
        candidates.append((first, end, float(chunk.median()), float(chunk.max())))
    candidates = sorted(sorted(candidates, key=lambda c: c[3], reverse=True)[:MAX_SPIKES], key=lambda c: c[0])
    spikes = tuple(
        Spike(index=i + 1, start=first, end=end, level_rps=level, peak_rps=peak)
        for i, (first, end, level, peak) in enumerate(candidates)
    )
    return SpikeProfile(
        baseline_rps=baseline, threshold_rps=threshold, series=series_name, label=label,
        spikes=spikes, values=values,
    )


def detect_spikes_from_labeled(lt_labeled: Sequence[Dict[str, Any]], perf_query: str) -> Optional[SpikeProfile]:
    if not perf_query:
        return None
    column, series, label = _series_from_labeled(list(lt_labeled or []), perf_query, "max")
    return detect_spikes(series, series_name=column, label=label or perf_query)


def _median_between(values: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    chunk = values[(values.index >= start) & (values.index < end)].dropna()
    return float(chunk.median()) if not chunk.empty else None


def spike_load_steps(profile: SpikeProfile, start_ts: float, end_ts: float, shift_hours: int) -> List[LoadStep]:
    """Contiguous background / spike steps covering the run; empty when no spike was found."""
    if not profile.spikes:
        return []
    run_start = pd.Timestamp(start_ts, unit="s", tz="UTC")
    run_end = pd.Timestamp(end_ts, unit="s", tz="UTC")
    bounds: List[tuple[str, pd.Timestamp, pd.Timestamp, Optional[Spike]]] = []
    cursor = run_start
    for spike in profile.spikes:
        if spike.start > cursor:
            bounds.append(("background", cursor, spike.start, None))
        bounds.append(("spike", max(spike.start, cursor), min(spike.end, run_end), spike))
        cursor = min(spike.end, run_end)
    if run_end > cursor:
        bounds.append(("background", cursor, run_end, None))
    steps: List[LoadStep] = []
    background_no = 0
    for kind, start, end, spike in bounds:
        if end <= start:
            continue
        if spike is not None:
            level: Optional[float] = spike.level_rps
            label = spike.label
        else:
            background_no += 1
            level = _median_between(profile.values, start, end)
            label = (
                t("report.step_label.background_level", index=background_no, level=f"{level:.0f}")
                if level is not None else t("report.step_label.background", index=background_no)
            )
        steps.append(LoadStep(
            index=len(steps) + 1, label=label, start=start, end=end,
            minute_from=minutes_from(start, start_ts), minute_to=minutes_from(end, start_ts),
            rps_level=level, stable=None, source="spike_detector", shift_hours=shift_hours,
            plateau_start=start, plateau_end=end,
        ))
    return steps


def _latency_query(sla_cfg: Mapping[str, Any]) -> str:
    return str(sla_cfg.get("p95_query") or "").strip() or str(sla_cfg.get("p99_query") or "").strip()


def _window_p95(frame: Optional[pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> tuple[Optional[float], Optional[str]]:
    if frame is None or end <= start:
        return None, None
    return _worst_series_p95(frame, start, end)


def _pct_change(current: Optional[float], base: Optional[float]) -> Optional[float]:
    if current is None or base is None or abs(base) < 1e-9:
        return None
    return round((current / base - 1.0) * 100.0, 1)


def _recovery(
    frame: Optional[pd.DataFrame],
    column: Optional[str],
    pre_p95: Optional[float],
    spike_end: pd.Timestamp,
    limit: pd.Timestamp,
) -> tuple[Optional[bool], Optional[float], Optional[pd.Timestamp]]:
    """First confirmed point after the spike with latency back near the background."""
    if frame is None or column is None or column not in frame.columns or pre_p95 is None:
        return None, None, None
    series = _utc_numeric(frame, column).dropna()
    after = series[(series.index >= spike_end) & (series.index <= limit)]
    if after.empty:
        return None, None, None
    ceiling = pre_p95 * (1.0 + RECOVERY_TOLERANCE_PCT / 100.0)
    ok = (after <= ceiling).tolist()
    for i, flag in enumerate(ok):
        if flag and (i + 1 >= len(ok) or ok[i + 1]):
            moment = after.index[i]
            return True, round(max(0.0, (moment - spike_end).total_seconds()), 1), moment
    return False, None, None


def analyze_spikes(
    profile: Optional[SpikeProfile],
    lt_labeled: Sequence[Dict[str, Any]],
    sla_cfg: Mapping[str, Any],
    run_end_ts: float,
    shift_hours: int,
) -> Dict[str, Any]:
    """Pack entry ``spike_profile``: per-spike overshoot and recovery; ``detected=False`` without spikes."""
    sla_cfg = sla_cfg if isinstance(sla_cfg, Mapping) else {}
    if profile is None:
        return {"detected": False, "reason": t("report.analysis.spike_no_series"), "spikes": []}
    common = {
        "baseline_rps": round(profile.baseline_rps, 2),
        "threshold_rps": round(profile.threshold_rps, 2),
        "spike_ratio": SPIKE_RATIO,
        "series": profile.series,
        "source_label": profile.label,
        "recovery_tolerance_pct": RECOVERY_TOLERANCE_PCT,
        "recovery_max_window_min": RECOVERY_MAX_WINDOW_MIN,
    }
    if not profile.spikes:
        return {
            "detected": False,
            "reason": t(
                "report.analysis.spike_below_threshold", threshold=f"{profile.threshold_rps:.0f}",
                background=f"{profile.baseline_rps:.0f}", ratio=f"{SPIKE_RATIO:g}", seconds=f"{SPIKE_MIN_DURATION_SEC:.0f}",
            ),
            "spikes": [],
            **common,
        }
    latency_query = _latency_query(sla_cfg)
    latency = _labeled_frame(list(lt_labeled or []), latency_query) if latency_query else None
    error_query = str(sla_cfg.get("error_rate_query") or "").strip()
    errors = _labeled_frame(list(lt_labeled or []), error_query) if error_query else None
    run_end = pd.Timestamp(run_end_ts, unit="s", tz="UTC")
    run_start = profile.values.index[0]
    out: List[Dict[str, Any]] = []
    for position, spike in enumerate(profile.spikes):
        previous_end = profile.spikes[position - 1].end if position else run_start
        next_start = profile.spikes[position + 1].start if position + 1 < len(profile.spikes) else run_end
        lookback = max(spike.end - spike.start, pd.Timedelta(minutes=PRE_SPIKE_WINDOW_MIN))
        pre_start = max(spike.start - lookback, previous_end)
        pre_p95, _ = _window_p95(latency, pre_start, spike.start)
        during_p95, worst_column = _window_p95(latency, spike.start, spike.end)
        pre_err, _ = _window_p95(errors, pre_start, spike.start)
        during_err, _ = _window_p95(errors, spike.start, spike.end)
        limit = min(spike.end + pd.Timedelta(minutes=RECOVERY_MAX_WINDOW_MIN), next_start)
        recovered, recovery_sec, recovered_at = _recovery(latency, worst_column, pre_p95, spike.end, limit)
        rps_after = _median_between(profile.values, spike.end, min(spike.end + pd.Timedelta(minutes=RPS_RECOVERY_WINDOW_MIN), next_start))
        rps_recovered = (
            abs(rps_after - profile.baseline_rps) <= max(profile.baseline_rps * RECOVERY_TOLERANCE_PCT / 100.0, 1.0)
            if rps_after is not None else None
        )
        note_code = ""
        if recovered is None and latency is not None and pre_p95 is not None:
            note_code = "no_latency_after"
        elif latency is None:
            note_code = "no_latency_query"
        out.append({
            "index": spike.index,
            "label": spike.label,
            "start_iso": format_iso(spike.start, shift_hours),
            "end_iso": format_iso(spike.end, shift_hours),
            "start_utc": str(spike.start),
            "end_utc": str(spike.end),
            "duration_sec": spike.duration_sec,
            "level_rps": round(spike.level_rps, 2),
            "peak_rps": round(spike.peak_rps, 2),
            "pre": {
                "start_iso": format_iso(pre_start, shift_hours),
                "end_iso": format_iso(spike.start, shift_hours),
                "p95": round(pre_p95, 3) if pre_p95 is not None else None,
                "error_rate_pct": round(pre_err, 3) if pre_err is not None else None,
            },
            "during": {
                "p95": round(during_p95, 3) if during_p95 is not None else None,
                "error_rate_pct": round(during_err, 3) if during_err is not None else None,
            },
            "overshoot_p95_pct": _pct_change(during_p95, pre_p95),
            "overshoot_error_rate_pp": round(during_err - pre_err, 3) if during_err is not None and pre_err is not None else None,
            "recovered": recovered,
            "recovery_time_sec": recovery_sec,
            "recovered_at_iso": format_iso(recovered_at, shift_hours) if recovered_at is not None else None,
            "rps_after": round(rps_after, 2) if rps_after is not None else None,
            "rps_recovered": rps_recovered,
            "latency_query": latency_query or None,
            "note": spike_note_text(note_code, get_locale()),
            "note_code": note_code or None,
        })
    return {"detected": True, "spikes": out, **common}


def build_spike_payload(
    spike_profile: Optional[Mapping[str, Any]],
    *,
    focus: str,
    target_rps: Optional[float],
    sla_summary: Optional[str],
) -> Dict[str, Any]:
    """``spike_analysis`` of the final report (pack entry without machine-only UTC fields)."""
    profile = dict(spike_profile or {"detected": False, "spikes": []})
    spikes = [
        {key: value for key, value in item.items() if key not in {"start_utc", "end_utc"}}
        for item in profile.get("spikes") or [] if isinstance(item, Mapping)
    ]
    worst = max(spikes, key=lambda s: (s.get("overshoot_p95_pct") is not None, s.get("overshoot_p95_pct") or 0.0), default=None)
    top_level = max((float(s.get("level_rps") or 0.0) for s in spikes), default=None)
    profile.update({
        "spikes": spikes,
        "focus": focus,
        "target_rps": target_rps,
        "max_spike_rps": top_level,
        "target_reached": (top_level >= float(target_rps)) if target_rps is not None and top_level is not None else None,
        "worst_spike_index": worst.get("index") if worst else None,
        "sla_summary": sla_summary,
    })
    return profile


def spike_windows_ctx(spike_profile: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Short list of spike windows for domain contexts."""
    return [
        {key: item.get(key) for key in ("index", "label", "start_iso", "end_iso", "level_rps", "recovered", "recovery_time_sec")}
        for item in (spike_profile or {}).get("spikes") or [] if isinstance(item, Mapping)
    ]


__all__ = [
    "RECOVERY_MAX_WINDOW_MIN",
    "RECOVERY_TOLERANCE_PCT",
    "SPIKE_NOTE_CODES",
    "SPIKE_NOTE_TEXTS",
    "SPIKE_RATIO",
    "Spike",
    "SpikeProfile",
    "analyze_spikes",
    "build_spike_payload",
    "detect_spikes",
    "detect_spikes_from_labeled",
    "spike_load_steps",
    "spike_note_text",
    "spike_windows_ctx",
]
