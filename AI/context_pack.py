"""Load-step aware context for the LLM.

Turns raw time series into compact, step-aligned facts the model can reason
about: load steps (from the step detector or equal time buckets), per-step
statistics for every metric, step-relative anomalies and a cross-domain
timeline. All timestamps are exposed both as ISO strings (with the configured
display offset) and as minutes from the test start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

MIN_TIME_BUCKETS = 4
MAX_TIME_BUCKETS = 8
TARGET_BUCKET_MINUTES = 10.0
MIN_STEPS_FROM_DETECTOR = 2

DEFAULT_ELASTICITY_THRESHOLD = 2.0
DEFAULT_DRIFT_PCT = 20.0
DEFAULT_SHIFT_PCT = 50.0
MIN_METRIC_GROWTH_PCT = 15.0
MIN_LOAD_GROWTH_PCT = 5.0
MIN_SAMPLES_FOR_DRIFT = 8
MAX_ANOMALIES_PER_SERIES = 3

TIMELINE_DOMAIN_PRIORITY = ["lt_framework", "microservices", "jvm", "database", "kafka", "hard_resources", "application_logs"]


# ---- time helpers -------------------------------------------------------------

def to_utc(value: Any) -> pd.Timestamp:
    """Normalizes naive (assumed UTC) and tz-aware timestamps to UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def format_iso(value: Any, shift_hours: int) -> str:
    """ISO 8601 with the configured display offset, e.g. 2024-11-01T13:42:00+03:00."""
    shifted = to_utc(value) + timedelta(hours=int(shift_hours))
    sign = "+" if shift_hours >= 0 else "-"
    hours = abs(int(shift_hours))
    return f"{shifted.strftime('%Y-%m-%dT%H:%M:%S')}{sign}{hours:02d}:00"


def minutes_from(value: Any, start_ts: float) -> float:
    start = pd.Timestamp(start_ts, unit="s", tz="UTC")
    return round((to_utc(value) - start).total_seconds() / 60.0, 1)


def time_fields(value: Any, start_ts: float, shift_hours: int) -> Dict[str, Any]:
    """Absolute and relative representation of one timestamp for the LLM."""
    return {"iso": format_iso(value, shift_hours), "minute_from_start": minutes_from(value, start_ts)}


def _utc_index(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(work.index))
    work.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return work.sort_index()


# ---- load steps ---------------------------------------------------------------

@dataclass
class LoadStep:
    index: int
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    minute_from: float
    minute_to: float
    rps_level: Optional[float]
    stable: Optional[bool]
    source: str
    shift_hours: int
    # Detector plateau inside the step; the rest of the step is ramp, drop or test end.
    plateau_start: Optional[pd.Timestamp] = None
    plateau_end: Optional[pd.Timestamp] = None
    drop_time: Optional[pd.Timestamp] = None
    # after_drop: the step follows an RPS fall that never recovered; dip: the step is a fall RPS recovered from.
    after_drop: bool = False
    dip: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": self.index,
            "label": self.label,
            "start_iso": format_iso(self.start, self.shift_hours),
            "end_iso": format_iso(self.end, self.shift_hours),
            "minute_from": self.minute_from,
            "minute_to": self.minute_to,
            "start_unix": float(self.start.timestamp()),
            "end_unix": float(self.end.timestamp()),
            "rps_level": round(self.rps_level, 1) if self.rps_level is not None else None,
            "stable": self.stable,
            "source": self.source,
        }
        if self.plateau_start is not None and self.plateau_end is not None:
            out["plateau_start_iso"] = format_iso(self.plateau_start, self.shift_hours)
            out["plateau_end_iso"] = format_iso(self.plateau_end, self.shift_hours)
        if self.drop_time is not None:
            out["drop_iso"] = format_iso(self.drop_time, self.shift_hours)
        if self.after_drop:
            out["after_drop"] = True
        if self.dip:
            out["dip"] = True
        return out


def _hhmm(value: Any, shift_hours: int) -> str:
    return (to_utc(value) + timedelta(hours=int(shift_hours))).strftime("%H:%M")


def equal_time_buckets(start_ts: float, end_ts: float, shift_hours: int) -> List[LoadStep]:
    """Splits the window into 4..8 equal intervals (~10 minutes each when possible)."""
    start = pd.Timestamp(start_ts, unit="s", tz="UTC")
    end = pd.Timestamp(end_ts, unit="s", tz="UTC")
    duration_min = max((end - start).total_seconds() / 60.0, 1.0)
    count = int(min(MAX_TIME_BUCKETS, max(MIN_TIME_BUCKETS, round(duration_min / TARGET_BUCKET_MINUTES))))
    width = (end - start) / count
    steps: List[LoadStep] = []
    for i in range(count):
        st = start + width * i
        en = end if i == count - 1 else start + width * (i + 1)
        steps.append(LoadStep(
            index=i + 1,
            label=f"интервал {i + 1} · {_hhmm(st, shift_hours)}–{_hhmm(en, shift_hours)}",
            start=st, end=en,
            minute_from=minutes_from(st, start_ts), minute_to=minutes_from(en, start_ts),
            rps_level=None, stable=None, source="time_buckets", shift_hours=shift_hours,
        ))
    return steps


def is_load_step_segment(seg: Dict[str, Any], min_step_minutes: float) -> bool:
    """A detector segment is a load step only when its plateau lasted ``min_step_minutes``.

    The detector also cuts segments of one confirmation window out of ramps and out
    of the decline after an RPS drop. They are transitions, not load levels.
    """
    minutes = seg.get("duration_min")
    if minutes is None and seg.get("start") is not None and seg.get("end") is not None:
        minutes = (to_utc(seg["end"]) - to_utc(seg["start"])).total_seconds() / 60.0
    return minutes is None or float(minutes) >= float(min_step_minutes)


def _parse_segments(step_segments: Optional[Sequence[Dict[str, Any]]], min_step_minutes: float) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for seg in step_segments or []:
        if not isinstance(seg, dict) or seg.get("start") is None:
            continue
        try:
            parsed.append({
                "start": to_utc(seg.get("start")),
                "end": to_utc(seg.get("end")) if seg.get("end") is not None else None,
                "drop": to_utc(seg.get("drop_time")) if seg.get("drop_time") else None,
                "level": float(seg.get("level")) if seg.get("level") is not None else None,
                "stable": (
                    bool(seg.get("stable")) and not bool(seg.get("after_drop")) and not bool(seg.get("dip"))
                ) if seg.get("stable") is not None else None,
                "after_drop": bool(seg.get("after_drop")),
                "dip": bool(seg.get("dip")),
                "is_step": is_load_step_segment(seg, min_step_minutes),
            })
        except (TypeError, ValueError):
            continue
    return sorted(parsed, key=lambda s: s["start"])


def first_rps_drop(step_segments: Optional[Sequence[Dict[str, Any]]]) -> Optional[pd.Timestamp]:
    """Earliest fall of the designated RPS that it never recovered from.

    It is a drop inside a detector segment or the start of the first segment after an
    unrecovered decline; a recovered dip (``dip``) is not a drop. Every detector segment
    counts, not only load steps, so it also works when the report fell back to equal time buckets.
    """
    parsed = _parse_segments(step_segments, 0.0)
    moments = [seg["drop"] for seg in parsed if seg["drop"] is not None and not seg["dip"]]
    moments += [seg["start"] for seg in parsed if seg["after_drop"]]
    return min(moments) if moments else None


def _tail_drop(seg: Dict[str, Any], tail: Sequence[Dict[str, Any]]) -> Optional[pd.Timestamp]:
    """Drop inside the plateau, otherwise where the merged tail fell for good.

    A tail segment counts when it follows an unrecovered decline below the step level or has a
    drop of its own; a recovered dip in the tail is not a drop.
    """
    if seg["drop"] is not None:
        return seg["drop"]
    for item in tail:
        if item["after_drop"] and seg["level"] is not None and item["level"] is not None and item["level"] < seg["level"]:
            return item["start"]
        if item["drop"] is not None and not item["dip"]:
            return item["drop"]
    return None


def _detector_step(
    index: int,
    seg: Dict[str, Any],
    tail: Sequence[Dict[str, Any]],
    span: tuple[pd.Timestamp, pd.Timestamp],
    start_ts: float,
    shift_hours: int,
) -> LoadStep:
    st, en = span
    level = seg["level"]
    level_text = f" · ≈{level:.0f} RPS" if level is not None else ""
    plateau_end = min(seg["end"], en) if seg["end"] is not None else None
    return LoadStep(
        index=index,
        label=f"ступень {index}{level_text}",
        start=st, end=en,
        minute_from=minutes_from(st, start_ts), minute_to=minutes_from(en, start_ts),
        rps_level=level, stable=seg["stable"], source="step_detector", shift_hours=shift_hours,
        plateau_start=max(seg["start"], st) if plateau_end is not None else None,
        plateau_end=plateau_end,
        drop_time=_tail_drop(seg, tail),
        after_drop=seg["after_drop"],
        dip=seg["dip"],
    )


def derive_load_steps(
    step_segments: Optional[Sequence[Dict[str, Any]]],
    start_ts: float,
    end_ts: float,
    shift_hours: int,
    min_step_minutes: float = 0.0,
) -> List[LoadStep]:
    """Builds contiguous load steps from detector segments; falls back to equal buckets.

    Segments come from the step detector for the designated RPS series (keys
    ``start``, ``end``, ``level``, ``stable``, ``duration_min``, ``drop_time``, ``after_drop``,
    ``dip``). Only segments held for
    ``min_step_minutes`` become steps; a shorter one joins the step before it, or the
    first step when it precedes all of them. Steps are contiguous so every sample of
    the window belongs to exactly one step.
    """
    parsed = _parse_segments(step_segments, min_step_minutes)
    positions = [i for i, seg in enumerate(parsed) if seg["is_step"]]
    if len(positions) < MIN_STEPS_FROM_DETECTOR:
        return equal_time_buckets(start_ts, end_ts, shift_hours)
    window_start = pd.Timestamp(start_ts, unit="s", tz="UTC")
    window_end = pd.Timestamp(end_ts, unit="s", tz="UTC")
    steps: List[LoadStep] = []
    for k, pos in enumerate(positions):
        next_pos = positions[k + 1] if k + 1 < len(positions) else len(parsed)
        st = window_start if k == 0 else parsed[pos]["start"]
        en = parsed[next_pos]["start"] if next_pos < len(parsed) else window_end
        if en <= st:
            continue
        tail = parsed[pos + 1:next_pos]
        steps.append(_detector_step(len(steps) + 1, parsed[pos], tail, (st, en), start_ts, shift_hours))
    return steps if len(steps) >= MIN_STEPS_FROM_DETECTOR else equal_time_buckets(start_ts, end_ts, shift_hours)


def _slice_step(series: pd.Series, step: LoadStep, is_last: bool) -> pd.Series:
    mask = (series.index >= step.start) & ((series.index <= step.end) if is_last else (series.index < step.end))
    return series[mask].dropna()


def _slice_plateau(series: pd.Series, step: LoadStep, is_last: bool) -> pd.Series:
    """Samples of the step plateau; the whole step when the detector gave no plateau."""
    if step.plateau_start is None or step.plateau_end is None:
        return _slice_step(series, step, is_last)
    mask = (series.index >= step.plateau_start) & (series.index <= step.plateau_end)
    return series[mask].dropna()


def _after_plateau_stats(series: pd.Series, step: LoadStep, is_last: bool) -> Optional[Dict[str, Any]]:
    """Values after the plateau when RPS dropped inside the step or the test ended after it."""
    if step.plateau_end is None or (step.drop_time is None and not is_last):
        return None
    full = _slice_step(series, step, is_last)
    tail = full[full.index > step.plateau_end]
    if tail.empty:
        return None
    return {
        "mean": round(float(tail.mean()), 4),
        "min": round(float(tail.min()), 4),
        "max": round(float(tail.max()), 4),
        "samples": int(tail.shape[0]),
    }


def _step_stats(series: pd.Series, steps: Sequence[LoadStep]) -> List[Optional[Dict[str, Any]]]:
    """Per-step statistics over the plateau, so a drop after it is not mixed into the step value."""
    stats: List[Optional[Dict[str, Any]]] = []
    for i, step in enumerate(steps):
        is_last = i == len(steps) - 1
        chunk = _slice_plateau(series, step, is_last)
        if chunk.empty:
            stats.append(None)
            continue
        item: Dict[str, Any] = {
            "step": step.index,
            "mean": round(float(chunk.mean()), 4),
            "p95": round(float(chunk.quantile(0.95)), 4),
            "max": round(float(chunk.max()), 4),
            "samples": int(chunk.shape[0]),
        }
        tail = _after_plateau_stats(series, step, is_last)
        if tail is not None:
            item["after_plateau"] = tail
        stats.append(item)
    return stats


def _change_pct(first: Optional[float], last: Optional[float]) -> Optional[float]:
    if first is None or last is None or abs(first) < 1e-9:
        return None
    return round((last / first - 1.0) * 100.0, 1)


def service_key(column: str) -> str:
    """Series name without its ``instance=`` part: series of one service share the key."""
    parts = [p for p in str(column).split("|") if p]
    base_parts = [p for p in parts if not p.startswith("instance=")]
    return "|".join(base_parts) if len(base_parts) != len(parts) else str(column)


def _aggregate_instances(df: pd.DataFrame) -> pd.DataFrame:
    """Averages series that differ only by ``instance=`` into one column per application."""
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        groups.setdefault(service_key(str(col)), []).append(str(col))
    out: Dict[str, pd.Series] = {}
    for key, cols in groups.items():
        if len(cols) == 1:
            out[cols[0]] = df[cols[0]]
        else:
            out[f"{key}|instances={len(cols)} (среднее)"] = df[cols].mean(axis=1)
    return pd.DataFrame(out, index=df.index)


def step_table(labeled_dfs: Sequence[Dict[str, Any]], steps: Sequence[LoadStep], top_n: int = 10) -> List[Dict[str, Any]]:
    """Per-section rows of per-step statistics, ranked by relative change across steps."""
    sections: List[Dict[str, Any]] = []
    for item in labeled_dfs:
        label = str(item.get("label") or "?")
        df = item.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            sections.append({"label": label, "rows": []})
            continue
        work = _aggregate_instances(_utc_index(df))
        rows: List[Dict[str, Any]] = []
        for col in work.columns:
            series = pd.to_numeric(work[col], errors="coerce")
            if series.dropna().empty:
                continue
            per_step = _step_stats(series, steps)
            present = [s for s in per_step if s]
            if not present:
                continue
            change = _change_pct(present[0]["mean"], present[-1]["mean"])
            rows.append({
                "series": str(col),
                "change_vs_first_pct": change,
                "overall_max": round(float(series.max()), 4),
                "per_step": per_step,
            })
        rows.sort(key=lambda r: (abs(r["change_vs_first_pct"]) if r["change_vs_first_pct"] is not None else -1.0, r["overall_max"]), reverse=True)
        sections.append({"label": label, "rows": rows[: max(int(top_n), 1)]})
    return sections


# ---- anomalies ----------------------------------------------------------------

def _anomaly(kind: str, step: LoadStep, value: float, reference: float, change_pct: float, explanation: str, load_change_pct: Optional[float] = None) -> Dict[str, Any]:
    out = {
        "kind": kind,
        "step_index": step.index,
        "step_label": step.label,
        "start_iso": format_iso(step.start, step.shift_hours),
        "end_iso": format_iso(step.end, step.shift_hours),
        "value": round(value, 4),
        "reference": round(reference, 4),
        "change_pct": round(change_pct, 1),
        "explanation": explanation,
    }
    if load_change_pct is not None:
        out["load_change_pct"] = round(load_change_pct, 1)
    return out


def detect_step_anomalies(col_series: pd.Series, steps: Sequence[LoadStep], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Step-relative anomalies instead of a global mean+2σ threshold.

    - elasticity: the metric grows much faster than the load between steps;
    - drift: the metric changes noticeably inside a single step;
    - shift: for equal time buckets (no load levels), a bucket deviates from the first one.
    """
    options = cfg or {}
    elasticity_threshold = float(options.get("step_elasticity_threshold", DEFAULT_ELASTICITY_THRESHOLD))
    drift_pct = float(options.get("step_drift_pct", DEFAULT_DRIFT_PCT))
    shift_pct = float(options.get("step_shift_pct", DEFAULT_SHIFT_PCT))
    if not steps or not isinstance(col_series.index, pd.DatetimeIndex):
        return []
    series = pd.to_numeric(col_series, errors="coerce")
    idx = pd.DatetimeIndex(series.index)
    series.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    series = series.sort_index()
    stats = _step_stats(series, steps)
    anomalies: List[Dict[str, Any]] = []

    base = next((s for s in stats if s), None)
    base_step = next((steps[i] for i, s in enumerate(stats) if s), None)
    has_levels = all(step.rps_level is not None for step in steps)
    if base and base_step is not None and abs(base["mean"]) > 1e-9:
        for i, current in enumerate(stats):
            step = steps[i]
            if current is None or step.index == base_step.index:
                continue
            metric_growth = (current["mean"] / base["mean"] - 1.0) * 100.0
            if has_levels and base_step.rps_level and base_step.rps_level > 0:
                load_growth = (float(step.rps_level) / float(base_step.rps_level) - 1.0) * 100.0
                if load_growth > MIN_LOAD_GROWTH_PCT and metric_growth > MIN_METRIC_GROWTH_PCT and metric_growth / load_growth > elasticity_threshold:
                    anomalies.append(_anomaly(
                        "elasticity", step, current["mean"], base["mean"], metric_growth,
                        f"На {step.label}: метрика выросла на {metric_growth:.0f} % при росте нагрузки на {load_growth:.0f} % (в {metric_growth / load_growth:.1f}× быстрее нагрузки) относительно {base_step.label}.",
                        load_change_pct=load_growth,
                    ))
            elif not has_levels and abs(metric_growth) > shift_pct:
                anomalies.append(_anomaly(
                    "shift", step, current["mean"], base["mean"], metric_growth,
                    f"На {step.label}: среднее значение отличается от первого интервала на {metric_growth:+.0f} %.",
                ))

    for i, step in enumerate(steps):
        chunk = _slice_plateau(series, step, is_last=(i == len(steps) - 1))
        if chunk.shape[0] < MIN_SAMPLES_FOR_DRIFT:
            continue
        quarter = max(1, chunk.shape[0] // 4)
        first = float(chunk.iloc[:quarter].mean())
        last = float(chunk.iloc[-quarter:].mean())
        if abs(first) < 1e-9:
            continue
        drift = (last / first - 1.0) * 100.0
        if abs(drift) > drift_pct:
            anomalies.append(_anomaly(
                "drift", step, last, first, drift,
                f"Внутри {step.label}: значение изменилось на {drift:+.0f} % от начала к концу ступени при неизменной нагрузке.",
            ))

    anomalies.sort(key=lambda a: abs(a["change_pct"]), reverse=True)
    return anomalies[:MAX_ANOMALIES_PER_SERIES]


# ---- timeline -----------------------------------------------------------------

def build_timeline(domain_packs: Dict[str, Dict[str, Any]], steps: Sequence[LoadStep], max_rows: int = 15) -> Dict[str, Any]:
    """Cross-domain matrix «top series of each section × load step» (mean per step)."""
    candidates: List[Dict[str, Any]] = []
    ordered_domains = [d for d in TIMELINE_DOMAIN_PRIORITY if d in domain_packs] + [d for d in domain_packs if d not in TIMELINE_DOMAIN_PRIORITY]
    for domain in ordered_domains:
        pack = domain_packs.get(domain) or {}
        for section in pack.get("step_table") or []:
            rows = section.get("rows") or []
            if not rows:
                continue
            top = rows[0]
            candidates.append({
                "domain": domain,
                "label": section.get("label"),
                "series": top.get("series"),
                "change_vs_first_pct": top.get("change_vs_first_pct"),
                "mean_per_step": [(s["mean"] if s else None) for s in top.get("per_step") or []],
            })
    lt_rows = [c for c in candidates if c["domain"] == "lt_framework"]
    other_rows = sorted(
        [c for c in candidates if c["domain"] != "lt_framework"],
        key=lambda c: abs(c["change_vs_first_pct"]) if c["change_vs_first_pct"] is not None else -1.0,
        reverse=True,
    )
    rows = (lt_rows + other_rows)[: max(int(max_rows), 1)]
    return {
        "steps": [{"index": s.index, "label": s.label, "rps_level": round(s.rps_level, 1) if s.rps_level is not None else None} for s in steps],
        "rows": rows,
        "note": (
            "Значения — средние по плато ступени, без просадки и спада нагрузки после него "
            "(они в step_table.per_step.after_plateau). Уровень нагрузки ступени — steps[].rps_level. "
            "Порядок строк: метрики нагрузочного инструмента, затем метрики с наибольшим относительным изменением."
        ),
    }


# ---- report step table --------------------------------------------------------

_LABEL_RE = re.compile(r"label='([^']*)'")
_SERIES_EN_RE = re.compile(r"series='([^']*)'")
_QUERY_RE = re.compile(r"запрос «([^»]*)»")
_SERIES_RU_RE = re.compile(r"серия «([^»]*)»")
_RPS_HINTS = ("rps", "throughput", "нагруз")
_LATENCY_HINTS = ("p95", "latency", "response", "отклик")
_STEP_NOTE = "RPS — уровень ступени. Время отклика — p95 на плато, без хвоста после просадки."


def _round_num(value: Any, digits: int) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits)


def _match_label(items: Sequence[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    target = str(query or "").strip().lower()
    if not target:
        return None
    rows = [item for item in items if isinstance(item, dict)]
    exact = [item for item in rows if str(item.get("label") or "").strip().lower() == target]
    if len(exact) == 1:
        return exact[0]
    partial = [item for item in rows if target in str(item.get("label") or "").strip().lower()]
    return partial[0] if len(partial) == 1 else None


def _best_column(frame: pd.DataFrame, stat: str) -> Optional[Any]:
    best_name: Optional[Any] = None
    best_value: Optional[float] = None
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        current = float(values.max() if stat == "max" else values.quantile(0.95))
        if best_value is None or current > best_value:
            best_value, best_name = current, column
    return best_name


def _utc_numeric(frame: pd.DataFrame, column: Any) -> pd.Series:
    series = pd.to_numeric(frame[column], errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(series.index))
    series.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return series.sort_index()


def _worst_p95_by_step(frame: pd.DataFrame, steps: Sequence[LoadStep]) -> List[Optional[float]]:
    """Highest plateau p95 among series of one query, separately for each step."""
    per_column = [_step_stats(_utc_numeric(frame, column), steps) for column in frame.columns]
    worst: List[Optional[float]] = []
    for index in range(len(steps)):
        values = [
            _round_num((stats[index] or {}).get("p95"), 2)
            for stats in per_column
            if stats[index]
        ]
        present = [value for value in values if value is not None]
        worst.append(max(present) if present else None)
    return worst


def _series_from_labeled(
    labeled: Sequence[Dict[str, Any]],
    query: str,
    stat: str,
) -> tuple[str, Optional[pd.Series], str]:
    section = _match_label(labeled, query)
    frame = section.get("df") if isinstance(section, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "", None, ""
    column = _best_column(frame, stat)
    if column is None:
        return "", None, str(section.get("label") or "")
    return str(column), _utc_numeric(frame, column), str(section.get("label") or "")


def _covers_start(row: Dict[str, Any], start: Optional[str]) -> bool:
    if not start:
        return False
    try:
        moment = to_utc(start)
        left = to_utc(row.get("plateau_start_iso") or row.get("start_iso"))
        right = to_utc(row.get("plateau_end_iso") or row.get("end_iso"))
    except (TypeError, ValueError):
        return False
    return left <= moment <= right


def _select_step(rows: Sequence[Dict[str, Any]], level: Any, start: Optional[str]) -> None:
    target = _round_num(level, 4)
    matches = [
        row for row in rows
        if target is not None and _round_num(row.get("rps"), 4) is not None and abs(float(row["rps"]) - target) <= 0.51
    ]
    covered = [row for row in matches if _covers_start(row, start)]
    # Equal time buckets can average far from the short SLA window inside them.
    containing = [row for row in rows if _covers_start(row, start)]
    pool = covered or containing or [row for row in matches if row.get("stable") is not False] or matches
    if pool:
        pool[-1]["selected"] = True


def _empty_step_report() -> Dict[str, Any]:
    return {
        "rps_query": None,
        "rps_series": None,
        "latency_query": None,
        "latency_series": None,
        "note": _STEP_NOTE,
        "steps": [],
    }


def load_step_report_from_frames(
    steps: Sequence[LoadStep],
    labeled: Sequence[Dict[str, Any]],
    *,
    rps_query: str = "",
    latency_query: str = "",
    selected_level: Any = None,
    selected_start: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-step RPS level and latency p95 on the plateau, for the report table."""
    if not steps:
        return _empty_step_report()
    rps_name, rps_series, rps_label = _series_from_labeled(labeled, rps_query, "max")
    latency_section = _match_label(labeled, latency_query)
    latency_frame = latency_section.get("df") if isinstance(latency_section, dict) else None
    lat_label = str(latency_section.get("label") or "") if isinstance(latency_section, dict) else ""
    lat_name = ""
    lat_p95: List[Optional[float]] = [None] * len(steps)
    same_query = bool(lat_label) and lat_label == rps_label
    if isinstance(latency_frame, pd.DataFrame) and not latency_frame.empty and not same_query:
        if len(list(latency_frame.columns)) == 1:
            lat_name = str(latency_frame.columns[0])
        lat_p95 = _worst_p95_by_step(latency_frame, steps)
    elif same_query:
        lat_label = ""
    rps_stats = _step_stats(rps_series, steps) if rps_series is not None else [None] * len(steps)
    rows: List[Dict[str, Any]] = []
    for step, rps, latency_p95 in zip(steps, rps_stats, lat_p95):
        payload = step.to_dict()
        level = payload.get("rps_level")
        mean = _round_num((rps or {}).get("mean"), 1)
        rows.append({
            "index": payload["index"],
            "label": payload["label"],
            "start_iso": payload.get("start_iso"),
            "end_iso": payload.get("end_iso"),
            "plateau_start_iso": payload.get("plateau_start_iso"),
            "plateau_end_iso": payload.get("plateau_end_iso"),
            "stable": payload.get("stable"),
            "after_drop": step.after_drop,
            "dip": step.dip,
            "rps": level if level is not None else mean,
            "rps_mean": mean,
            "latency_p95": latency_p95,
            "selected": False,
        })
    _select_step(rows, selected_level, selected_start)
    return {
        "rps_query": rps_label or None,
        "rps_series": rps_name or None,
        "latency_query": lat_label or None,
        "latency_series": lat_name or None,
        "note": _STEP_NOTE,
        "steps": rows,
    }


def _check_hint(checks: Optional[Sequence[Dict[str, Any]]], name: str) -> Dict[str, Any]:
    item = next((c for c in checks or [] if isinstance(c, dict) and str(c.get("name") or "") == name), None)
    if item is None:
        return {}
    message = str(item.get("message") or "")
    label_match = _LABEL_RE.search(message) or _QUERY_RE.search(message)
    series_match = _SERIES_EN_RE.search(message) or _SERIES_RU_RE.search(message)
    hint: Dict[str, Any] = {}
    if label_match:
        hint["label"] = label_match.group(1)
    if series_match:
        hint["series"] = series_match.group(1)
    if item.get("actual") is not None:
        hint["actual"] = item.get("actual")
    return hint


def _step_sections(pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    direct = pack.get("step_table")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    domains = pack.get("domains") if isinstance(pack.get("domains"), dict) else {}
    lt = domains.get("lt_framework") if isinstance(domains.get("lt_framework"), dict) else {}
    nested = lt.get("step_table")
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, dict)]
    return []


def _is_aggregate_label(label: str) -> bool:
    text = str(label or "").lower()
    return any(word in text for word in ("sum", "all groups", "total", "общ", "итого"))


def _section_peak(section: Dict[str, Any]) -> float:
    peaks = [
        float(row.get("overall_max") or 0)
        for row in (section.get("rows") or [])
        if isinstance(row, dict)
    ]
    return max(peaks) if peaks else 0.0


def _section_by_keywords(
    sections: Sequence[Dict[str, Any]],
    keywords: Sequence[str],
    skip_label: str = "",
) -> Optional[Dict[str, Any]]:
    skip = skip_label.strip().lower()
    matched = [
        section for section in sections
        if str(section.get("label") or "").strip().lower() != skip
        and any(word in str(section.get("label") or "").lower() for word in keywords)
    ]
    if not matched:
        return None
    aggregate = [section for section in matched if _is_aggregate_label(str(section.get("label") or ""))]
    pool = aggregate or matched
    return max(pool, key=_section_peak)


def _pick_context_row(
    sections: Sequence[Dict[str, Any]],
    hint: Dict[str, Any],
    keywords: Sequence[str],
    skip_label: str = "",
) -> tuple[Optional[Dict[str, Any]], str]:
    section = _match_label(sections, str(hint.get("label") or ""))
    if section is None or str(section.get("label") or "").strip().lower() == skip_label.strip().lower():
        section = _section_by_keywords(sections, keywords, skip_label)
    if section is None:
        return None, ""
    rows = [row for row in (section.get("rows") or []) if isinstance(row, dict)]
    series = str(hint.get("series") or "")
    chosen = next((row for row in rows if series and str(row.get("series") or "") == series), None)
    if chosen is None and rows:
        chosen = max(rows, key=lambda row: float(row.get("overall_max") or 0))
    return chosen, str(section.get("label") or "")


def _context_stat(row: Optional[Dict[str, Any]], step_index: int, field: str) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    raw = list(row.get("per_step") or [])
    found = next(
        (item for item in raw if isinstance(item, dict) and item.get("step") == step_index),
        None,
    )
    if found is None and 1 <= step_index <= len(raw):
        positional = raw[step_index - 1]
        if isinstance(positional, dict) and positional.get("step") in (None, step_index):
            found = positional
    if found is None:
        return None
    return _round_num(found.get(field), 4)


def load_step_report_from_context(
    pack: Dict[str, Any],
    *,
    checks: Optional[Sequence[Dict[str, Any]]] = None,
    selected_level: Any = None,
    selected_start: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Rebuilds the report step table from a stored lt_framework or final context."""
    raw_steps = pack.get("load_steps") if isinstance(pack, dict) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    sections = _step_sections(pack)
    rps_hint = _check_hint(checks, "target_rps")
    latency_hint = _check_hint(checks, "p95_latency") or _check_hint(checks, "p99_latency")
    rps_row, rps_label = _pick_context_row(sections, rps_hint, _RPS_HINTS)
    latency_section = _match_label(sections, str(latency_hint.get("label") or ""))
    if (
        latency_section is None
        or str(latency_section.get("label") or "").strip().lower() == str(rps_label or "").strip().lower()
    ):
        latency_section = _section_by_keywords(sections, _LATENCY_HINTS, rps_label)
    latency_label = str((latency_section or {}).get("label") or "")
    latency_rows = [row for row in ((latency_section or {}).get("rows") or []) if isinstance(row, dict)]
    named_series = str(latency_hint.get("series") or "")
    named_row = next(
        (row for row in latency_rows if named_series and str(row.get("series") or "") == named_series),
        None,
    )
    rows: List[Dict[str, Any]] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        index = int(step.get("index") or len(rows) + 1)
        level = _round_num(step.get("rps_level"), 1)
        mean = _round_num(_context_stat(rps_row, index, "mean"), 1)
        if named_row is not None:
            latency_p95 = _round_num(_context_stat(named_row, index, "p95"), 2)
        else:
            present = [value for value in (_context_stat(row, index, "p95") for row in latency_rows) if value is not None]
            latency_p95 = _round_num(max(present), 2) if present else None
        rows.append({
            "index": index,
            "label": str(step.get("label") or f"ступень {index}"),
            "start_iso": step.get("start_iso"),
            "end_iso": step.get("end_iso"),
            "plateau_start_iso": step.get("plateau_start_iso"),
            "plateau_end_iso": step.get("plateau_end_iso"),
            "stable": step.get("stable"),
            "after_drop": bool(step.get("after_drop")),
            "dip": bool(step.get("dip")),
            "rps": level if level is not None else mean,
            "rps_mean": mean,
            "latency_p95": latency_p95,
            "selected": False,
        })
    if not rows:
        return None
    level = selected_level if selected_level is not None else rps_hint.get("actual")
    _select_step(rows, level, selected_start)
    return {
        "rps_query": rps_label or None,
        "rps_series": str((rps_row or {}).get("series") or "") or None,
        "latency_query": latency_label or None,
        "latency_series": named_series if named_row is not None else None,
        "note": _STEP_NOTE,
        "steps": rows,
    }


__all__ = [
    "LoadStep",
    "build_timeline",
    "derive_load_steps",
    "detect_step_anomalies",
    "equal_time_buckets",
    "first_rps_drop",
    "format_iso",
    "is_load_step_segment",
    "load_step_report_from_context",
    "load_step_report_from_frames",
    "minutes_from",
    "service_key",
    "step_table",
    "time_fields",
    "to_utc",
]
