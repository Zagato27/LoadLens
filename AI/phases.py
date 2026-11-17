# SPDX-License-Identifier: Apache-2.0
"""Phases of a scenario run: the stretches where one group of requests of the load tool carries the load.

A scenario test walks a business process from start to end (catalogue → cart → payment → delivery),
so the load moves from one part of the system to another and RPS levels mean nothing on their own. The
phases are found in the RPS-by-request series of the load tool (role ``lt.rps_by_request``):

1. every series name (``group=::Checkout::Pay|name=…`` for k6, ``transaction=…`` for JMeter, ``request=…`` for
   Gatling, ``name=…`` for Locust) gives a group path; k6 nests groups with ``::``;
2. the nesting level is the smallest one on which at least two groups lead in turn (a k6 script wrapped in one
   ``::Scenario`` group goes one level deeper);
3. every interval is given to the group with the largest share of its RPS; runs of one group are segments;
4. segments shorter than ``phase_min_minutes`` join the one before them (the first one joins the next);
   at most ``phase_max_count`` phases stay, the shortest joining the neighbour that carries more of it;
5. a phase is named by its group and the busiest group under it; its ``key`` (the group path) aligns the
   phases of two runs.

The phases replace the load steps of the run (as spike windows do in a spike test), so the step table, the
anomalies inside a step, the chart zones and the report step table work on them unchanged. Nothing here knows a
particular scenario: the names come from the script.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from AI.context_pack import LoadStep, format_iso, minutes_from
from AI.i18n import t

PHASE_ROLE = "lt.rps_by_request"
# Keys of a series name that hold the group or the request of the load tool, in the order they are tried.
GROUP_KEYS = ("group", "transaction", "scenario", "request", "name", "label")
NEST_SEPARATOR = "::"
DEFAULT_MIN_PHASE_MINUTES = 2.0
DEFAULT_MAX_PHASES = 16
DEFAULT_DURATION_TOLERANCE_PCT = 25.0
# A latency or error threshold fails a phase only when it is broken this long; a shorter excursion is a risk.
DEFAULT_BREACH_MINUTES = 2.0
# Groups listed per phase and the smallest share a listed group has.
MAX_GROUPS_PER_PHASE = 3
MIN_GROUP_SHARE = 0.05
# Two unmatched phases of two runs are the same phase when their groups overlap this much (Jaccard).
MATCH_SIMILARITY = 0.5
IDLE = ""

REASON_NO_QUERY = "no_group_query"
REASON_NO_DATA = "no_data"
REASON_ONE_GROUP = "one_group"


@dataclass(frozen=True)
class PhaseSettings:
    """``sla.phase_*`` with defaults applied."""

    query: str = ""
    label_key: str = ""
    level: int = 0
    min_minutes: float = DEFAULT_MIN_PHASE_MINUTES
    max_count: int = DEFAULT_MAX_PHASES
    duration_tolerance_pct: float = DEFAULT_DURATION_TOLERANCE_PCT
    # Regular expression (case-insensitive) over the name of a phase: matching phases are not judged.
    exclude: str = ""
    breach_minutes: float = DEFAULT_BREACH_MINUTES

    def excludes(self, phase: Any) -> bool:
        """Whether ``phase`` (a ``Phase`` or its dict) is left out of the SLA by ``exclude``."""
        if not self.exclude:
            return False
        pattern = _exclude_pattern(self.exclude)
        return any(pattern.search(name) for name in _phase_names(phase))


@lru_cache(maxsize=32)
def _exclude_pattern(text: str) -> "re.Pattern[str]":
    """``sla.phase_exclude`` compiled; text that is not a valid expression is matched literally."""
    try:
        return re.compile(text, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(text), re.IGNORECASE)


def _phase_names(phase: Any) -> List[str]:
    """The names an exclusion is matched against: the leading group of the phase (its key) and its title (that
    group and its busiest child). Other groups of the phase are left out: authentication runs beside every stage,
    and ``Authentication`` must not exclude them all."""
    get = phase.get if isinstance(phase, Mapping) else (lambda key: getattr(phase, key, None))
    if get("idle"):
        return []
    return [name for name in (str(get("key") or ""), str(get("title") or "")) if name]


def _number(value: Any, default: float, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or number < minimum:
        return default
    return number


def phase_settings(sla_cfg: Optional[Mapping[str, Any]]) -> PhaseSettings:
    cfg = sla_cfg if isinstance(sla_cfg, Mapping) else {}
    return PhaseSettings(
        query=str(cfg.get("phase_query") or "").strip(),
        label_key=str(cfg.get("phase_label_key") or "").strip(),
        level=int(_number(cfg.get("phase_level"), 0, 0)),
        min_minutes=_number(cfg.get("phase_min_minutes"), DEFAULT_MIN_PHASE_MINUTES, 0),
        max_count=max(2, int(_number(cfg.get("phase_max_count"), DEFAULT_MAX_PHASES, 1))),
        duration_tolerance_pct=_number(cfg.get("phase_duration_tolerance_pct"), DEFAULT_DURATION_TOLERANCE_PCT, 0),
        exclude=str(cfg.get("phase_exclude") or "").strip(),
        breach_minutes=_number(cfg.get("phase_breach_minutes"), DEFAULT_BREACH_MINUTES, 0),
    )


# ---- series names -----------------------------------------------------------------------------------------

def series_labels(name: Any) -> Dict[str, str]:
    """``group=::a::b|name=x`` -> ``{"group": "::a::b", "name": "x"}``; a name without ``=`` gives ``{}``."""
    out: Dict[str, str] = {}
    for part in str(name or "").split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def label_key_of(names: Sequence[Any], wanted: str = "") -> str:
    """The key the groups are read from: ``wanted`` when the series have it, else the first of ``GROUP_KEYS``
    they have, else their first key ('' for series without keys)."""
    keys: List[str] = []
    for name in names:
        for key in series_labels(name):
            if key not in keys:
                keys.append(key)
    if wanted and wanted in keys:
        return wanted
    return next((key for key in GROUP_KEYS if key in keys), keys[0] if keys else "")


def group_path(name: Any, label_key: str) -> tuple[str, ...]:
    """Group path of a series: the parts of its ``label_key`` value split by ``::`` (k6 nests groups)."""
    labels = series_labels(name)
    value = labels.get(label_key) if label_key else None
    if value is None:
        value = str(name or "")
    parts = tuple(part.strip() for part in value.split(NEST_SEPARATOR) if part.strip())
    return parts or (value.strip() or "?",)


def _key(path: tuple[str, ...], level: int) -> str:
    return NEST_SEPARATOR.join(path[:level])


# ---- detection --------------------------------------------------------------------------------------------

@dataclass
class _Segment:
    key: str
    first: int
    last: int  # inclusive row
    absorbed: int = 0

    @property
    def rows(self) -> int:
        return self.last - self.first + 1


@dataclass
class Phase:
    index: int
    number: int
    key: str
    title: str
    start: pd.Timestamp
    end: pd.Timestamp
    rps_mean: Optional[float]
    share: Optional[float]
    groups: List[Dict[str, Any]]
    idle: bool
    absorbed: int
    at_start: bool
    at_end: bool
    excluded: bool = False  # left out of the SLA by ``sla.phase_exclude`` (e.g. the test-data cleanup)

    @property
    def duration_min(self) -> float:
        return round((self.end - self.start).total_seconds() / 60.0, 1)

    def label(self) -> str:
        if self.idle:
            return t("report.step_label.pause", index=self.number)
        return t("report.step_label.phase", index=self.number, title=self.title)

    def to_dict(self, shift_hours: int) -> Dict[str, Any]:
        return {
            "index": self.index,
            "number": self.number,
            "key": self.key,
            "title": self.title,
            "label": self.label(),
            "start_iso": format_iso(self.start, shift_hours),
            "end_iso": format_iso(self.end, shift_hours),
            "start_utc": str(self.start),
            "end_utc": str(self.end),
            "duration_min": self.duration_min,
            "rps_mean": round(self.rps_mean, 2) if self.rps_mean is not None else None,
            "share": round(self.share, 3) if self.share is not None else None,
            "groups": self.groups,
            "idle": self.idle,
            "absorbed": self.absorbed,
            "at_start": self.at_start,
            "at_end": self.at_end,
            "excluded": self.excluded,
        }


@dataclass
class PhaseProfile:
    detected: bool
    source_label: str
    label_key: str
    level: int
    interval_sec: int
    min_phase_min: float
    reason: Optional[str]
    phases: List[Phase] = field(default_factory=list)

    def to_dict(self, shift_hours: int) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "source_label": self.source_label,
            "label_key": self.label_key,
            "level": self.level,
            "interval_sec": self.interval_sec,
            "min_phase_min": self.min_phase_min,
            "reason": self.reason,
            "phases": [phase.to_dict(shift_hours) for phase in self.phases],
        }


def _utc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.apply(pd.to_numeric, errors="coerce")
    index = pd.DatetimeIndex(work.index)
    work.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    work = work.sort_index()
    return work.loc[:, work.notna().any(axis=0)]


def _grouped(frame: pd.DataFrame, paths: Mapping[Any, tuple[str, ...]], level: int) -> pd.DataFrame:
    keys = [_key(paths[column], level) for column in frame.columns]
    return frame.fillna(0.0).T.groupby(keys).sum().T


def _dominant(grouped: pd.DataFrame, total: pd.Series) -> pd.Series:
    leader = grouped.idxmax(axis=1)
    return leader.where(total > 0, IDLE)


def _segments(dominant: Sequence[str]) -> List[_Segment]:
    out: List[_Segment] = []
    for row, key in enumerate(dominant):
        if out and out[-1].key == key:
            out[-1].last = row
        else:
            out.append(_Segment(key=key, first=row, last=row))
    return out


def _coalesce(segments: List[_Segment]) -> List[_Segment]:
    out: List[_Segment] = []
    for seg in segments:
        if out and out[-1].key == seg.key:
            out[-1].last = seg.last
            out[-1].absorbed += seg.absorbed
        else:
            out.append(_Segment(seg.key, seg.first, seg.last, seg.absorbed))
    return out


def _merge(segments: List[_Segment], position: int, into: int) -> List[_Segment]:
    """``segments[position]`` joins its neighbour ``segments[into]`` (which keeps its key)."""
    target = segments[into]
    gone = segments[position]
    target.first = min(target.first, gone.first)
    target.last = max(target.last, gone.last)
    target.absorbed += 1 + gone.absorbed
    return _coalesce([seg for i, seg in enumerate(segments) if i != position])


def _absorb_short(segments: List[_Segment], min_rows: int) -> List[_Segment]:
    segments = _coalesce(segments)
    while len(segments) > 1:
        position = next((i for i, seg in enumerate(segments) if seg.rows < min_rows), None)
        if position is None:
            break
        segments = _merge(segments, position, position - 1 if position > 0 else 1)
    return segments


def _cap(segments: List[_Segment], max_count: int, share: pd.DataFrame) -> List[_Segment]:
    while len(segments) > max_count:
        position = min(range(len(segments)), key=lambda i: (segments[i].rows, i))
        neighbours = [i for i in (position - 1, position + 1) if 0 <= i < len(segments)]
        rows = share.iloc[segments[position].first:segments[position].last + 1]

        def carried(i: int) -> float:
            key = segments[i].key
            return float(rows[key].mean()) if key in rows.columns else 0.0

        into = max(neighbours, key=lambda i: (carried(i), -abs(i - position)))
        segments = _merge(segments, position, into)
    return segments


def _choose_level(frame: pd.DataFrame, paths: Mapping[Any, tuple[str, ...]], total: pd.Series, wanted: int) -> tuple[int, bool]:
    """Nesting level and whether at least two groups lead on it."""
    depth = max((len(path) for path in paths.values()), default=1)
    if wanted > 0:
        level = min(wanted, depth)
        leaders = {key for key in _dominant(_grouped(frame, paths, level), total) if key != IDLE}
        return level, len(leaders) >= 2
    for level in range(1, depth + 1):
        leaders = {key for key in _dominant(_grouped(frame, paths, level), total) if key != IDLE}
        if len(leaders) >= 2:
            return level, True
    return 1, False


def _groups_of(rows: pd.DataFrame, frame_rows: pd.DataFrame, paths: Mapping[Any, tuple[str, ...]], level: int) -> List[Dict[str, Any]]:
    """Up to ``MAX_GROUPS_PER_PHASE`` groups of a phase by share, each with its busiest child group."""
    means = rows.mean(axis=0).sort_values(ascending=False)
    out: List[Dict[str, Any]] = []
    for key, value in means.items():
        if len(out) >= MAX_GROUPS_PER_PHASE or value < MIN_GROUP_SHARE:
            break
        children: Dict[str, float] = {}
        for column in frame_rows.columns:
            path = paths[column]
            if _key(path, level) == key and len(path) > level:
                children[path[level]] = children.get(path[level], 0.0) + float(frame_rows[column].fillna(0.0).sum())
        child = max(children, key=children.get) if children else None
        out.append({"key": str(key), "child": child, "share": round(float(value), 3)})
    return out


def _title(key: str, groups: List[Dict[str, Any]]) -> str:
    name = key.split(NEST_SEPARATOR)[-1] if key else ""
    lead = next((group for group in groups if group["key"] == key), None)
    child = lead.get("child") if lead else None
    return f"{name} · {child}" if child else name


def detect_phases(
    frame: Optional[pd.DataFrame],
    *,
    start_ts: float,
    end_ts: float,
    settings: PhaseSettings,
    source_label: str = "",
) -> PhaseProfile:
    """Phases of the run from the RPS-by-request frame (index: time, one column per series)."""
    window_start = pd.Timestamp(start_ts, unit="s", tz="UTC")
    window_end = pd.Timestamp(end_ts, unit="s", tz="UTC")
    empty = PhaseProfile(False, source_label, "", 1, 60, settings.min_minutes, REASON_NO_DATA)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return empty
    work = _utc_frame(frame)
    work = work[(work.index >= window_start) & (work.index < window_end)]
    if work.empty or work.shape[1] == 0:
        return empty
    label_key = label_key_of(list(work.columns), settings.label_key)
    paths = {column: group_path(column, label_key) for column in work.columns}
    total = work.fillna(0.0).sum(axis=1)
    diffs = pd.Series(work.index).diff().dropna().dt.total_seconds()
    interval = int(diffs.median()) if not diffs.empty and diffs.median() > 0 else 60
    level, several = _choose_level(work, paths, total, settings.level)
    grouped = _grouped(work, paths, level)
    share = grouped.div(total.where(total > 0), axis=0).fillna(0.0)
    if not several:
        key = str(grouped.sum(axis=0).idxmax()) if float(total.sum()) > 0 else IDLE
        segments = [_Segment(key=key, first=0, last=len(work) - 1)]
        reason: Optional[str] = REASON_ONE_GROUP if key != IDLE else REASON_NO_DATA
    else:
        segments = _segments(list(_dominant(grouped, total)))
        segments = _absorb_short(segments, max(1, math.ceil(settings.min_minutes * 60.0 / interval)))
        segments = _cap(segments, settings.max_count, share)
        reason = None
    phases: List[Phase] = []
    numbers = {"phase": 0, "pause": 0}
    for position, seg in enumerate(segments):
        start = window_start if position == 0 else work.index[seg.first]
        end = window_end if position == len(segments) - 1 else work.index[segments[position + 1].first]
        rows = slice(seg.first, seg.last + 1)
        idle = seg.key == IDLE
        numbers["pause" if idle else "phase"] += 1
        groups = [] if idle else _groups_of(share.iloc[rows], work.iloc[rows], paths, level)
        rps = total.iloc[rows]
        phases.append(Phase(
            index=position + 1,
            number=numbers["pause" if idle else "phase"],
            key=seg.key,
            title="" if idle else _title(seg.key, groups),
            start=start,
            end=end,
            rps_mean=float(rps.mean()) if not rps.empty else None,
            share=None if idle else float(share[seg.key].iloc[rows].mean()),
            groups=groups,
            idle=idle,
            absorbed=seg.absorbed,
            at_start=position == 0,
            at_end=position == len(segments) - 1,
        ))
    detected = several and sum(1 for phase in phases if not phase.idle) >= 2
    if several and not detected:
        reason = REASON_ONE_GROUP
    return PhaseProfile(detected, source_label, label_key, level, interval, settings.min_minutes, reason, phases)


def phase_section(
    lt_labeled: Sequence[Mapping[str, Any]],
    queries_block: Optional[Mapping[str, Any]],
    settings: PhaseSettings,
) -> Optional[Mapping[str, Any]]:
    """The labeled lt_framework section the phases are read from: ``sla.phase_query``, else the query of role
    ``lt.rps_by_request`` with the most series. A role query without rows is still returned (empty), so the
    profile says "no data in the window" instead of "no query"."""
    sections = [item for item in lt_labeled or [] if isinstance(item, Mapping) and isinstance(item.get("df"), pd.DataFrame)]
    if settings.query:
        wanted = settings.query.lower()
        exact = [item for item in sections if str(item.get("label") or "").strip().lower() == wanted]
        partial = [item for item in sections if wanted in str(item.get("label") or "").strip().lower()]
        found = exact or (partial if len(partial) == 1 else [])
        if found:
            return found[0]
    labels = set(_role_labels(queries_block))
    named = [item for item in sections if str(item.get("label") or "") in labels]
    candidates = [item for item in named if not item["df"].empty]
    if candidates:
        return max(candidates, key=lambda item: item["df"].shape[1])
    return named[0] if named else None


def _role_labels(queries_block: Optional[Mapping[str, Any]]) -> List[str]:
    from AI.query_packs.blocks import labels_for_role

    return list(labels_for_role(queries_block, PHASE_ROLE)) if isinstance(queries_block, Mapping) else []


def phases_from_labeled(
    lt_labeled: Sequence[Mapping[str, Any]],
    queries_block: Optional[Mapping[str, Any]],
    sla_cfg: Optional[Mapping[str, Any]],
    start_ts: float,
    end_ts: float,
) -> PhaseProfile:
    settings = phase_settings(sla_cfg)
    section = phase_section(lt_labeled, queries_block, settings)
    if section is None:
        # A configured role query that failed or returned nothing is missing data, not a missing query.
        labels = _role_labels(queries_block)
        reason = REASON_NO_DATA if labels else REASON_NO_QUERY
        return PhaseProfile(False, labels[0] if labels else "", "", 1, 60, settings.min_minutes, reason)
    profile = detect_phases(section["df"], start_ts=start_ts, end_ts=end_ts, settings=settings, source_label=str(section.get("label") or ""))
    for phase in profile.phases:
        phase.excluded = settings.excludes(phase)
    return profile


def phase_load_steps(profile: PhaseProfile, start_ts: float, shift_hours: int) -> List[LoadStep]:
    """The phases as contiguous load steps (``source="phase_detector"``); a phase is its own plateau."""
    return [
        LoadStep(
            index=phase.index, label=phase.label(), start=phase.start, end=phase.end,
            minute_from=minutes_from(phase.start, start_ts), minute_to=minutes_from(phase.end, start_ts),
            rps_level=phase.rps_mean, stable=None, source="phase_detector", shift_hours=shift_hours,
            plateau_start=phase.start, plateau_end=phase.end,
        )
        for phase in profile.phases
    ]


def phases_ctx(profile: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compact phases for the domain contexts: what ran when, how loaded it was and its SLA verdict."""
    phases = (profile or {}).get("phases") if isinstance(profile, Mapping) else None
    keys = ("index", "label", "key", "start_iso", "end_iso", "duration_min", "rps_mean", "idle", "excluded", "verdict", "failed")
    out = []
    for phase in phases or []:
        if isinstance(phase, Mapping):
            item = {key: phase.get(key) for key in keys}
            item["groups"] = [
                {"key": group.get("key"), "child": group.get("child"), "share": group.get("share")}
                for group in phase.get("groups") or [] if isinstance(group, Mapping)
            ]
            out.append(item)
    return out


# ---- two runs ---------------------------------------------------------------------------------------------

def _active(phases: Any) -> List[Mapping[str, Any]]:
    return [phase for phase in phases or [] if isinstance(phase, Mapping) and not phase.get("idle")]


def _group_keys(phase: Mapping[str, Any]) -> set:
    return {str(group.get("key")) for group in phase.get("groups") or [] if isinstance(group, Mapping)} or {str(phase.get("key"))}


def _similar(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    left, right = _group_keys(a), _group_keys(b)
    return bool(left | right) and len(left & right) / len(left | right) >= MATCH_SIMILARITY


def _lcs_pairs(left: List[str], right: List[str]) -> List[tuple[int, int]]:
    rows, cols = len(left), len(right)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            table[i][j] = table[i + 1][j + 1] + 1 if left[i] == right[j] else max(table[i + 1][j], table[i][j + 1])
    pairs: List[tuple[int, int]] = []
    i = j = 0
    while i < rows and j < cols:
        if left[i] == right[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def align_phases(reference: Any, current: Any) -> List[tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]]:
    """Pairs ``(reference phase, current phase)`` in time order; ``None`` where a phase has no pair.

    Phases are matched by key in order (longest common subsequence); between two matches the remaining phases pair
    in order when their groups overlap (``MATCH_SIMILARITY``). Pauses are left out.
    """
    ref, cur = _active(reference), _active(current)
    anchors = _lcs_pairs([str(p.get("key")) for p in ref], [str(p.get("key")) for p in cur])
    out: List[tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]] = []
    prev_i = prev_j = -1
    for i, j in [*anchors, (len(ref), len(cur))]:
        gap_ref, gap_cur = ref[prev_i + 1:i], cur[prev_j + 1:j]
        next_cur = 0
        for r in gap_ref:
            match = next((m for m in range(next_cur, len(gap_cur)) if _similar(r, gap_cur[m])), None)
            if match is None:
                out.append((r, None))
                continue
            out.extend((None, gap_cur[m]) for m in range(next_cur, match))
            out.append((r, gap_cur[match]))
            next_cur = match + 1
        out.extend((None, gap_cur[m]) for m in range(next_cur, len(gap_cur)))
        if i < len(ref) and j < len(cur):
            out.append((ref[i], cur[j]))
        prev_i, prev_j = i, j
    return out


def _duration_comparable(phase: Mapping[str, Any], min_minutes: float) -> bool:
    duration = phase.get("duration_min")
    return (
        not phase.get("at_start") and not phase.get("at_end")
        and isinstance(duration, (int, float)) and duration >= max(min_minutes, 0.1)
    )


def compare_phase_lists(
    reference: Any,
    current: Any,
    min_minutes: float = DEFAULT_MIN_PHASE_MINUTES,
    interval_sec: int = 60,
    settings: Optional[PhaseSettings] = None,
) -> Dict[str, Any]:
    """Missing and extra phases of ``current`` against ``reference`` and the change of every phase duration.

    Phases left out of the SLA (``excluded``, or matched by ``settings.exclude`` in a reference stored before the
    setting) are left out here too: a cleanup step that took longer is no risk of the system.

    Phases at either edge of a run are left out of the durations: the test window cuts them. A phase boundary is
    known to one data interval, so a change of up to two intervals is not ``significant`` (2 → 3 minutes on
    minute data is +50 % of nothing); ``worst_duration`` is the largest significant change.
    """
    pairs = align_phases(_judged(reference, settings), _judged(current, settings))
    floor_min = 2.0 * max(int(interval_sec or 60), 1) / 60.0
    durations = []
    for ref, cur in pairs:
        if ref is None or cur is None:
            continue
        if not (_duration_comparable(ref, min_minutes) and _duration_comparable(cur, min_minutes)):
            continue
        change = (float(cur["duration_min"]) / float(ref["duration_min"]) - 1.0) * 100.0
        durations.append({
            "key": cur.get("key"), "label": cur.get("label"), "title": cur.get("title"),
            "reference_min": ref.get("duration_min"), "current_min": cur.get("duration_min"),
            "change_pct": round(change, 1),
            "significant": abs(float(cur["duration_min"]) - float(ref["duration_min"])) > floor_min + 1e-9,
        })
    significant = [item for item in durations if item["significant"]]
    return {
        "pairs": [
            {"reference": _brief(ref), "current": _brief(cur)} for ref, cur in pairs
        ],
        "missing": [_brief(ref) for ref, cur in pairs if cur is None],
        "extra": [_brief(cur) for ref, cur in pairs if ref is None],
        "durations": durations,
        "worst_duration": max(significant, key=lambda item: abs(item["change_pct"])) if significant else None,
    }


def _judged(phases: Any, settings: Optional[PhaseSettings]) -> List[Mapping[str, Any]]:
    return [
        phase for phase in phases or []
        if isinstance(phase, Mapping) and not phase.get("excluded") and not (settings is not None and settings.excludes(phase))
    ]


def _brief(phase: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if phase is None:
        return None
    return {key: phase.get(key) for key in ("index", "number", "key", "title", "label", "start_iso", "end_iso", "duration_min", "rps_mean")}


def _change(value: Any, baseline: Any) -> Optional[float]:
    try:
        current, base = float(value), float(baseline)
    except (TypeError, ValueError):
        return None
    return round((current / base - 1.0) * 100.0, 1) if abs(base) > 1e-9 else None


def _side(phase: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if phase is None:
        return None
    keys = ("index", "number", "key", "title", "label", "start_iso", "end_iso", "start_utc", "end_utc", "duration_min",
            "rps_mean", "excluded", "verdict", "failed")
    return {key: phase.get(key) for key in keys}


def phase_pairs(scenario_a: Optional[Mapping[str, Any]], scenario_b: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Phases of run A (the assessed one) against run B (the baseline), aligned in time order.

    Every pair holds both phases (``None`` where one run lacks it) and the change of A against B of the duration
    and the mean RPS; ``only_in_a`` / ``only_in_b`` count the phases without a pair.
    """
    phases_a = (scenario_a or {}).get("phases") if isinstance(scenario_a, Mapping) else None
    phases_b = (scenario_b or {}).get("phases") if isinstance(scenario_b, Mapping) else None
    pairs = []
    for ref, cur in align_phases(phases_b, phases_a):
        pairs.append({
            "a": _side(cur),
            "b": _side(ref),
            "duration_change_pct": _change(cur.get("duration_min"), ref.get("duration_min")) if cur and ref else None,
            "rps_change_pct": _change(cur.get("rps_mean"), ref.get("rps_mean")) if cur and ref else None,
        })
    return {
        "pairs": pairs,
        "only_in_a": sum(1 for pair in pairs if pair["b"] is None),
        "only_in_b": sum(1 for pair in pairs if pair["a"] is None),
    }


def reference_from_scenario(run_name: str, scenario: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The phases of an earlier scenario run (its stored ``scenario`` payload) as the reference of this one."""
    phases = [phase for phase in scenario.get("phases") or [] if isinstance(phase, Mapping)]
    if not scenario.get("detected") or not phases:
        return None
    keys = ("index", "number", "key", "title", "label", "start_iso", "end_iso", "duration_min", "rps_mean", "groups",
            "idle", "at_start", "at_end", "excluded", "verdict")
    return {
        "run_name": run_name,
        "interval_sec": scenario.get("interval_sec"),
        "phases": [{key: phase.get(key) for key in keys} for phase in phases],
    }


def _weighted_rps(phases: List[Mapping[str, Any]]) -> Optional[float]:
    total = sum(float(p.get("duration_min") or 0) for p in phases if p.get("rps_mean") is not None)
    if total <= 0:
        return None
    return round(sum(float(p["rps_mean"]) * float(p.get("duration_min") or 0) for p in phases if p.get("rps_mean") is not None) / total, 2)


def build_scenario_payload(
    profile: Optional[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
    phase_results: Optional[Sequence[Mapping[str, Any]]],
    *,
    focus: str = "",
    settings: Optional[PhaseSettings] = None,
) -> Dict[str, Any]:
    """``scenario`` of the final payload: the phases with their SLA verdicts and the comparison with the reference.

    ``rps_mean`` is the mean RPS of the phases weighted by their duration (the load level of the run on the
    dashboard); ``worst_phase`` is the phase the SLA verdict points at (its worst failed check), else the first
    phase that failed, else the first one at risk.
    """
    profile = profile if isinstance(profile, Mapping) else {}
    settings = settings or PhaseSettings()
    results = {int(item.get("index")): item for item in phase_results or [] if isinstance(item, Mapping) and item.get("index") is not None}
    phases: List[Dict[str, Any]] = []
    for phase in profile.get("phases") or []:
        if not isinstance(phase, Mapping):
            continue
        item = dict(phase)
        result = results.get(int(phase.get("index") or 0))
        if result is not None:
            item["verdict"] = result.get("verdict")
            item["failed"] = list(result.get("failed") or [])
            if result.get("worst"):
                item["worst"] = True
        phases.append(item)
    active = [phase for phase in phases if not phase.get("idle")]
    worst = (
        next((p for p in active if p.get("worst")), None)
        or next((p for p in active if p.get("verdict") == "fail"), None)
        or next((p for p in active if p.get("verdict") == "risk"), None)
    )
    out: Dict[str, Any] = {
        "focus": focus,
        "detected": bool(profile.get("detected")),
        "reason": profile.get("reason"),
        "source_label": profile.get("source_label"),
        "label_key": profile.get("label_key"),
        "level": profile.get("level"),
        "interval_sec": profile.get("interval_sec"),
        "min_phase_min": profile.get("min_phase_min"),
        "phase_count": len(active),
        "rps_mean": _weighted_rps(active),
        "phases": phases,
        "worst_phase": _brief(worst) | {"verdict": worst.get("verdict"), "failed": worst.get("failed")} if worst else None,
        "reference": {"available": False},
    }
    ref_phases = (reference or {}).get("phases") if isinstance(reference, Mapping) else None
    if ref_phases and out["detected"]:
        comparison = compare_phase_lists(ref_phases, phases, settings.min_minutes, int(profile.get("interval_sec") or 60), settings)
        out["reference"] = {
            "available": True,
            "run_name": reference.get("run_name"),
            "phase_count": len(_active(ref_phases)),
            "missing": comparison["missing"],
            "extra": comparison["extra"],
            "durations": comparison["durations"],
            "worst_duration": comparison["worst_duration"],
            "duration_tolerance_pct": settings.duration_tolerance_pct,
        }
    return out


__all__ = [
    "DEFAULT_BREACH_MINUTES",
    "DEFAULT_DURATION_TOLERANCE_PCT",
    "DEFAULT_MAX_PHASES",
    "DEFAULT_MIN_PHASE_MINUTES",
    "PHASE_ROLE",
    "Phase",
    "PhaseProfile",
    "PhaseSettings",
    "align_phases",
    "build_scenario_payload",
    "compare_phase_lists",
    "detect_phases",
    "group_path",
    "label_key_of",
    "phase_load_steps",
    "phase_pairs",
    "phase_section",
    "phase_settings",
    "phases_ctx",
    "phases_from_labeled",
    "reference_from_scenario",
    "series_labels",
]
