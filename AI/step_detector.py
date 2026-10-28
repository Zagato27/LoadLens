# SPDX-License-Identifier: Apache-2.0
"""Load steps of an RPS series, without settings.

The detector reads its windows from the cadence of the series and its jump threshold from the noise of the
plateaus, so the same code serves raw 15-second points, minute buckets and 5-minute buckets:

- the confirmation window (``hold``) covers half the shortest step the user expects (``min_stable_minutes``),
  at least two samples;
- a first pass with the floor threshold (``FLOOR_PCT``) finds the plateaus, their residuals give the relative
  noise ``sigma``; the final threshold is ``NOISE_SIGMAS * sigma / sqrt(hold)``, never below the floor;
- a series where most samples are not on a plateau (ramps of two samples per step) has no steps
  (``REASON_NO_PLATEAUS``): the report falls back to equal time buckets and says why.

Segments keep the meaning the report relies on: a persistent drop inside a step (``drop_time``), declines RPS
recovered from (``dip``) or never recovered from (``after_drop``), stable steps and the stable maximum.

Plans: plans/auto-step-detector-plan.md (verified on a corpus of real runs in plans/step-detector-corpus/).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Raw points finer than this are averaged into FINE_RESAMPLE_SEC buckets first (the windows below stay small).
FINE_RESAMPLE_SEC = 15
# A rolling median over SMOOTH_SEC smooths only fine cadences; at minute buckets it would blend the samples
# on both sides of a jump and hide steps that are just above the threshold.
SMOOTH_MAX_SAMPLE_SEC = 20.0
SMOOTH_SEC = 60.0
# A level change below FLOOR_PCT of the level, or below FLOOR_RPS, is never a load step.
FLOOR_PCT = 0.025
FLOOR_RPS = 1.0
# The jump threshold is NOISE_SIGMAS noise widths of a hold-window median; fewer sigmas split noisy plateaus.
NOISE_SIGMAS = 6.0
MAD_TO_SIGMA = 1.4826
MIN_NOISE_RESIDUALS = 6
# Segments of PLATEAU_MIN_SAMPLES+ samples are plateaus; below PLATEAU_SHARE of the samples on plateaus the
# series is a continuous ramp (or two samples per step) and has no load steps.
PLATEAU_MIN_SAMPLES = 3
PLATEAU_SHARE = 0.5
# Stable step: coefficient of variation, slope as a share of the level per minute, and the share of the plateau
# core within the threshold of the level (an RPS that swings between two levels has a low cv but no plateau).
MAX_CV = 0.10
SLOPE_SHARE_PER_MIN = 0.01
COHERENT_SHARE = 0.7
# A drop inside a step: RPS below MAX_DROP_PCT of the step start for DROP_HOLD_SEC and not recovered.
MAX_DROP_PCT = 0.08
DROP_HOLD_SEC = 120.0
# Two adjacent unstable segments closer than this many thresholds are one unstable segment.
MERGE_UNSTABLE_FACTOR = 2.0

REASON_NO_PLATEAUS = "no_plateaus"
METHOD = "step_profile"

# ``sla`` keys of the former presets and their tuning: no longer read, dropped when the SLA block is saved.
RETIRED_SLA_KEYS = (
    "step_detection_preset",
    "step_detection_resample_sec",
    "step_detection_smooth_sec",
    "step_confirm_hold_sec",
    "step_min_step_delta_rps",
    "step_min_step_delta_pct",
    "step_max_cv",
    "step_max_slope_rps_per_min",
    "step_max_within_step_drop_pct",
    "step_drop_hold_sec",
)


def retired_sla_keys(sla: Any) -> List[str]:
    """Keys of ``RETIRED_SLA_KEYS`` present in an ``sla`` block."""
    return [key for key in RETIRED_SLA_KEYS if isinstance(sla, dict) and key in sla]


def without_retired_sla_keys(sla: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of an ``sla`` block without the retired detector keys."""
    return {key: value for key, value in sla.items() if key not in RETIRED_SLA_KEYS}


@dataclass(frozen=True)
class DetectorParams:
    """What the detector derived from the series: the report states it next to the steps."""

    sample_sec: float
    hold: int
    sigma_rel: float
    thr_rel: float
    plateau_share: float
    reason: Optional[str] = None

    def threshold(self, level: float) -> float:
        return _step_delta_threshold(level, FLOOR_RPS, self.thr_rel)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "auto",
            "sample_sec": round(self.sample_sec, 1),
            "hold_samples": self.hold,
            "noise_pct": round(self.sigma_rel * 100.0, 3),
            "threshold_pct": round(self.thr_rel * 100.0, 2),
            "plateau_share": round(self.plateau_share, 2),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StepDetection:
    """``profile`` is the stable maximum with all segments (None without a stable step); ``params`` always set."""

    profile: Optional[Dict[str, object]]
    params: Optional[DetectorParams]


# ---- helpers shared with the segment statistics -----------------------------------------------------------


def _longest_true_run(values: List[bool]) -> int:
    best = 0
    cur = 0
    for v in values:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _first_true_run_start(values: List[bool], run_len: int) -> Optional[int]:
    need = max(1, int(run_len))
    cur = 0
    for idx, v in enumerate(values):
        if v:
            cur += 1
            if cur >= need:
                return idx - need + 1
        else:
            cur = 0
    return None


def _persistent_drop_start(values: List[bool], run_len: int) -> Optional[int]:
    """Start of the first run of ``run_len`` low samples that is not followed by ``run_len`` recovered ones.

    A low run that RPS climbs back from inside the same segment is a transient dip, not a drop.
    """
    need = max(1, int(run_len))
    start: Optional[int] = None
    idx = 0
    while idx < len(values):
        end = idx
        while end < len(values) and values[end] == values[idx]:
            end += 1
        if values[idx] and end - idx >= need and start is None:
            start = idx
        elif not values[idx] and start is not None and end - idx >= need:
            start = None
        idx = end
    return start


def _slope_rps_per_min(series_vals: pd.Series) -> float:
    if series_vals is None or len(series_vals) < 2:
        return 0.0
    y = np.asarray(series_vals.values, dtype=float)
    if isinstance(getattr(series_vals, "index", None), pd.DatetimeIndex):
        x = (series_vals.index - series_vals.index[0]).total_seconds().to_numpy(dtype=float) / 60.0
    else:
        x = np.arange(len(y), dtype=float)
    dx = x - x.mean()
    den = float(np.sum(dx * dx))
    if den <= 0:
        return 0.0
    return float(np.sum(dx * (y - y.mean())) / den)


def _segment_end_sort_key(value: Any) -> float:
    try:
        return float(pd.Timestamp(value).value)
    except Exception:
        return float("-inf")


def _step_delta_threshold(level: float, min_delta_rps: float, min_delta_pct: float) -> float:
    """Smallest RPS change that counts as a new load level near ``level``."""
    return max(float(min_delta_rps), abs(float(level)) * float(min_delta_pct))


def _plateau_core(seg: pd.Series, level: float, tolerance: float) -> pd.Series:
    """Segment without the leading/trailing samples farther than ``tolerance`` from ``level``.

    At coarse cadence the first samples of a plateau are still the ramp into it and would fail
    the slope check of an otherwise flat step. The whole segment is returned when trimming
    would leave less than half of it: then it is not a plateau with ramp edges.
    """
    values = seg.values
    lo, hi = 0, len(values)
    while lo < hi and abs(float(values[lo]) - level) > tolerance:
        lo += 1
    while hi > lo and abs(float(values[hi - 1]) - level) > tolerance:
        hi -= 1
    if hi - lo < max(2, (len(values) + 1) // 2):
        return seg
    return seg.iloc[lo:hi]


def _mark_decline(members: List[Dict[str, object]], opener: Optional[Dict[str, object]], recovered: bool) -> None:
    """Flags a closed decline: ``dip`` when RPS came back to the level before it, ``after_drop`` otherwise.

    The drops of a recovered decline become ``dip_time``: RPS fell there but was not lost.
    """
    for seg in members:
        seg["dip" if recovered else "after_drop"] = True
    if not recovered:
        return
    for seg in members + ([opener] if opener is not None else []):
        if seg.get("drop_time") is not None:
            seg["dip_time"] = seg["drop_time"]
            seg["drop_time"] = None


def _classify_declines(segments: List[Dict[str, object]], min_delta_rps: float, min_delta_pct: float) -> None:
    """Sets ``after_drop`` / ``dip`` from the outcome of each RPS decline instead of a flag that never resets.

    A decline starts at a step down (reference: the level before it) or at a persistent drop
    inside a segment (reference: the segment's starting level). It is recovered when a later
    segment climbs back within the detector's step threshold of the reference. Segments of a
    recovered decline are ``dip``; segments of a decline RPS never recovered from are
    ``after_drop``. The segment where an in-segment drop starts keeps its plateau before the
    drop, so it is not part of the decline.
    """
    members: List[Dict[str, object]] = []
    opener: Optional[Dict[str, object]] = None
    reference: Optional[float] = None
    for seg in segments:
        down_ref = seg.get("_down_ref")
        if down_ref is not None:
            reference = float(down_ref) if reference is None else max(reference, float(down_ref))
            members.append(seg)
        elif reference is not None:
            if float(seg.get("level", 0.0)) >= reference - _step_delta_threshold(reference, min_delta_rps, min_delta_pct):
                _mark_decline(members, opener, recovered=True)
                members, opener, reference = [], None, None
            else:
                members.append(seg)
        if seg.get("drop_time") is not None:
            level_ref = float(seg.get("_level_ref", seg.get("level", 0.0)))
            if reference is None:
                reference, opener = level_ref, seg
            else:
                reference = max(reference, level_ref)
    if reference is not None:
        _mark_decline(members, opener, recovered=False)


def _select_step_profile_candidate(segments: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Last stable step before the load first dropped.

    A flat plateau after RPS fell (timeouts, ramp-down) is degradation, not a
    load step, so segments flagged ``after_drop`` are never the stable maximum.
    A recovered dip is not a load level either.
    """
    rising = [seg for seg in segments if not bool(seg.get("after_drop")) and not bool(seg.get("dip"))]
    stable_segments = [seg for seg in rising if bool(seg.get("stable"))]
    if not stable_segments:
        return None
    terminal_stable_segments: List[Dict[str, object]] = []
    for idx, seg in enumerate(rising):
        if not bool(seg.get("stable")):
            continue
        next_seg = rising[idx + 1] if idx + 1 < len(rising) else None
        if next_seg is not None and bool(next_seg.get("stable")):
            continue
        terminal_stable_segments.append(seg)
    if terminal_stable_segments:
        return max(
            terminal_stable_segments,
            key=lambda seg: (_segment_end_sort_key(seg.get("end")), float(seg.get("level", 0.0))),
        )
    return max(
        stable_segments,
        key=lambda seg: (float(seg.get("level", 0.0)), _segment_end_sort_key(seg.get("end"))),
    )


# ---- detection --------------------------------------------------------------------------------------------


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _scan_step_boundaries(values: np.ndarray, hold: int, threshold) -> tuple[List[int], Dict[int, float]]:
    """Positions where the load level changes (with 0 and the length) and the level before every step down.

    ``prev`` is the median of up to two windows of the current segment, counted from its first sample on the
    level; ``next`` the median of the window from ``i``. A change counts when it reaches ``threshold(prev)`` and
    the window after the next one still differs from ``prev`` (a spike or dip of one window is not a step).
    The boundary is the first sample that left the old level (a ramp sample opens the new step); the level of the
    new step is read from its first sample on the new level, so a partial ramp sample does not shift ``prev``
    and make a second boundary.
    """
    n = len(values)
    boundaries = [0]
    down_refs: Dict[int, float] = {}
    level_from = 0
    i = hold
    while i < n - hold:
        prev_vals = values[max(level_from, i - 2 * hold):i]
        if len(prev_vals) < hold:
            i += 1
            continue
        prev_level = _median(prev_vals)
        next_level = _median(values[i:i + hold])
        thr = threshold(prev_level)
        if abs(next_level - prev_level) < thr:
            i += 1
            continue
        after_vals = values[i + hold:i + 2 * hold]
        if len(after_vals) >= 2 and abs(_median(after_vals) - prev_level) < thr:
            i += 1
            continue
        new_level = _median(values[i + 1:i + 1 + 2 * hold]) if i + 1 < n else next_level
        search_from = i
        if new_level < prev_level and float(values[i]) - prev_level > 0.5 * thr:
            # A burst right before a fall still belongs to the old level: the fall starts after it.
            search_from = i + 1
        boundary = next(
            (j for j in range(search_from, min(n, i + 2 * hold)) if abs(float(values[j]) - prev_level) > 0.5 * thr),
            search_from,
        )
        new_tol = 0.5 * threshold(new_level)
        level_from = next(
            (j for j in range(boundary, min(n, boundary + 2 * hold)) if abs(float(values[j]) - new_level) <= new_tol),
            boundary,
        )
        boundaries.append(boundary)
        if new_level < prev_level:
            down_refs[boundary] = prev_level
        i = level_from + hold
    boundaries.append(n)
    return boundaries, down_refs


def _prepared(series: pd.Series) -> Optional[tuple[pd.Series, float]]:
    """Sorted series without gaps, fine cadences resampled and smoothed, with its sample spacing in seconds."""
    s = series.dropna()
    try:
        s = s.sort_index()
    except Exception:
        pass
    if s.empty or len(s) < 3 or not isinstance(s.index, pd.DatetimeIndex):
        return None
    sample_sec = _sample_sec(s)
    if sample_sec < FINE_RESAMPLE_SEC:
        s = s.resample(f"{FINE_RESAMPLE_SEC}s").mean().dropna()
        sample_sec = _sample_sec(s)
    if sample_sec <= SMOOTH_MAX_SAMPLE_SEC:
        # Centered: a trailing median lags half a window, and the plateau would take in the first samples
        # of the ramp after it.
        window = max(2, int(round(SMOOTH_SEC / sample_sec)))
        s = s.rolling(window=window, min_periods=max(2, window // 2), center=True).median().dropna()
    if len(s) < 3:
        return None
    return s, sample_sec


def _sample_sec(s: pd.Series) -> float:
    deltas = s.index.to_series().diff().dropna()
    if deltas.empty:
        return 60.0
    return max(1.0, float(deltas.median().total_seconds()))


def _floor_params(sample_sec: float, min_stable_minutes: float) -> DetectorParams:
    hold = max(2, int(math.ceil((float(min_stable_minutes) * 60.0 / 2.0) / sample_sec)))
    return DetectorParams(sample_sec=sample_sec, hold=hold, sigma_rel=0.0, thr_rel=FLOOR_PCT, plateau_share=0.0)


def _first_pass(s: pd.Series, params: DetectorParams) -> tuple[float, float]:
    """Share of the samples on plateaus and the relative noise of the plateau cores, with the floor threshold."""
    boundaries, _ = _scan_step_boundaries(s.values, params.hold, params.threshold)
    inside = 0
    residuals: List[float] = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b - a < PLATEAU_MIN_SAMPLES:
            continue
        inside += b - a
        seg = s.iloc[a:b]
        level = float(seg.median())
        if level <= 0:
            continue
        core = _plateau_core(seg, level, 0.5 * params.threshold(level))
        if len(core) >= PLATEAU_MIN_SAMPLES:
            residuals.extend(np.abs(core.to_numpy(dtype=float) - level) / level)
    share = inside / max(len(s), 1)
    sigma = MAD_TO_SIGMA * float(np.median(residuals)) if len(residuals) >= MIN_NOISE_RESIDUALS else 0.0
    return share, sigma


def detector_params(series: pd.Series, min_stable_minutes: float) -> Optional[DetectorParams]:
    """Windows and threshold the detector uses for ``series``; None for a series too short to look at."""
    prepared = _prepared(series)
    if prepared is None:
        return None
    s, sample_sec = prepared
    return _params_for(s, sample_sec, min_stable_minutes)


def _params_for(s: pd.Series, sample_sec: float, min_stable_minutes: float) -> DetectorParams:
    floor = _floor_params(sample_sec, min_stable_minutes)
    share, sigma = _first_pass(s, floor)
    thr_rel = max(NOISE_SIGMAS * sigma / math.sqrt(floor.hold), FLOOR_PCT)
    return DetectorParams(
        sample_sec=sample_sec,
        hold=floor.hold,
        sigma_rel=sigma,
        thr_rel=thr_rel,
        plateau_share=share,
        reason=REASON_NO_PLATEAUS if share < PLATEAU_SHARE else None,
    )


def _segment(seg: pd.Series, params: DetectorParams, min_stable_minutes: float, down_ref: Optional[float]) -> Optional[Dict[str, object]]:
    """Statistics of one segment: drop inside it, level, plateau core, stability."""
    drop_run = max(1, int(round(DROP_HOLD_SEC / params.sample_sec)))
    base = seg.iloc[:max(2, min(len(seg), params.hold))]
    level_ref = float(base.median())
    below = [bool(v < level_ref * (1.0 - MAX_DROP_PCT)) for v in seg.values]
    drop_idx = _persistent_drop_start(below, drop_run)
    eval_seg, drop_time = seg, None
    if drop_idx is not None and drop_idx >= 2:
        eval_seg, drop_time = seg.iloc[:drop_idx], seg.index[drop_idx]
    if len(eval_seg) < 2:
        return None
    duration_min = ((eval_seg.index[-1] - eval_seg.index[0]).total_seconds() + params.sample_sec) / 60.0
    level = float(eval_seg.median())
    thr = params.threshold(level)
    core = _plateau_core(eval_seg, level, 0.5 * thr)
    core_values = core.to_numpy(dtype=float)
    core_mean = float(core_values.mean())
    cv = float(core_values.std()) / max(abs(core_mean), 1e-9)
    slope = _slope_rps_per_min(core)
    slope_limit = abs(level) * SLOPE_SHARE_PER_MIN
    coherent = float(np.mean(np.abs(core_values - level) <= thr))
    stable = duration_min >= float(min_stable_minutes) and cv <= MAX_CV and abs(slope) <= slope_limit and coherent >= COHERENT_SHARE
    return {
        "start": eval_seg.index[0],
        "end": eval_seg.index[-1],
        # The plateau starts after the ramp samples that open the segment (they stay in the step, not in its stats).
        "plateau_start": core.index[0],
        "level": level,
        "duration_min": float(max(duration_min, 0.0)),
        "cv": cv,
        "slope_rps_per_min": slope,
        "slope_limit_rps_per_min": float(slope_limit),
        "coherent_share": coherent,
        "longest_drop_min": float(_longest_true_run(below) * (params.sample_sec / 60.0)),
        "drop_time": str(drop_time) if drop_time is not None else None,
        "dip_time": None,
        "had_drop": _first_true_run_start(below, drop_run) is not None,
        "after_drop": False,
        "dip": False,
        "stable": bool(stable),
        "_down_ref": down_ref,
        "_level_ref": level_ref,
        "_slice": seg,
    }


def _mergeable(a: Dict[str, object], b: Dict[str, object], params: DetectorParams) -> bool:
    """Neighbours at one level, or two unstable neighbours close to each other, with no drop between them.

    A small step down between two unstable segments merges too: RPS that swings between two levels is one
    unstable stretch, not a drop (a drop would cut the report window before the real end of the load).
    """
    if a.get("drop_time") is not None:
        return False
    gap = abs(float(b["level"]) - float(a["level"]))
    thr = params.threshold(float(a["level"]))
    if gap < thr:
        return True
    return not a["stable"] and not b["stable"] and gap < MERGE_UNSTABLE_FACTOR * thr


def _merge(segments: List[Dict[str, object]], params: DetectorParams, min_stable_minutes: float) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for seg in segments:
        if out and _mergeable(out[-1], seg, params):
            merged = _segment(pd.concat([out[-1]["_slice"], seg["_slice"]]), params, min_stable_minutes, out[-1]["_down_ref"])
            if merged is not None:
                out[-1] = merged
                continue
        out.append(seg)
    return out


def _exported(seg: Dict[str, object]) -> Dict[str, object]:
    return {
        "start": str(seg["start"]),
        "end": str(seg["end"]),
        "plateau_start": str(seg.get("plateau_start", seg["start"])),
        "level": float(seg.get("level", 0.0)),
        "duration_min": float(seg.get("duration_min", 0.0)),
        "cv": float(seg.get("cv", 0.0)),
        "slope_rps_per_min": float(seg.get("slope_rps_per_min", 0.0)),
        "slope_limit_rps_per_min": float(seg.get("slope_limit_rps_per_min", 0.0)),
        "coherent_share": float(seg.get("coherent_share", 0.0)),
        "longest_drop_min": float(seg.get("longest_drop_min", 0.0)),
        "had_drop": bool(seg.get("had_drop", False)),
        "drop_time": seg.get("drop_time"),
        "dip_time": seg.get("dip_time"),
        "after_drop": bool(seg.get("after_drop", False)),
        "dip": bool(seg.get("dip", False)),
        "stable": bool(seg.get("stable", False)),
    }


def detect_steps(series: pd.Series, min_stable_minutes: float) -> StepDetection:
    """Segments of ``series`` and the stable maximum among them, with the parameters the detector derived."""
    prepared = _prepared(series)
    if prepared is None:
        return StepDetection(None, None)
    s, sample_sec = prepared
    params = _params_for(s, sample_sec, min_stable_minutes)
    if params.reason is not None:
        return StepDetection(None, params)
    boundaries, down_refs = _scan_step_boundaries(s.values, params.hold, params.threshold)
    segments: List[Dict[str, object]] = []
    pending_down_ref: Optional[float] = None
    for a, b in zip(boundaries, boundaries[1:]):
        if a in down_refs:
            pending_down_ref = down_refs[a] if pending_down_ref is None else max(pending_down_ref, down_refs[a])
        if b - a < 2:
            continue
        seg = _segment(s.iloc[a:b], params, min_stable_minutes, pending_down_ref)
        if seg is None:
            continue
        segments.append(seg)
        pending_down_ref = None
    segments = _merge(segments, params, min_stable_minutes)
    _classify_declines(segments, FLOOR_RPS, params.thr_rel)
    chosen = _select_step_profile_candidate(segments)
    if not chosen:
        return StepDetection(None, params)
    return StepDetection(
        {
            "stable_max": float(chosen.get("level", 0.0)),
            "stable_max_time": str(chosen.get("end")),
            "stable_duration_min": float(chosen.get("duration_min", min_stable_minutes)),
            "stable_start_time": str(chosen.get("start")),
            "method": METHOD,
            "step_segments": [_exported(seg) for seg in segments],
        },
        params,
    )


def find_step_profile(series: pd.Series, min_stable_minutes: float) -> Optional[Dict[str, object]]:
    """Stable maximum and segments of ``series``; None when it has no stable step."""
    return detect_steps(series, min_stable_minutes).profile


__all__ = [
    "COHERENT_SHARE",
    "FLOOR_PCT",
    "FLOOR_RPS",
    "MAX_CV",
    "REASON_NO_PLATEAUS",
    "RETIRED_SLA_KEYS",
    "DetectorParams",
    "StepDetection",
    "detect_steps",
    "detector_params",
    "find_step_profile",
    "retired_sla_keys",
    "without_retired_sla_keys",
]
