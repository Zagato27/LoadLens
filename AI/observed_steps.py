# SPDX-License-Identifier: Apache-2.0
"""Run-scoped load step boundaries supplied by a generator."""

import math
import sys
from typing import Any

import pandas as pd

from AI.context_pack import LoadStep, bucket_width, minutes_from, observed_bucket_slice
from AI.i18n import LocalizedError
from AI.step_detector import observed_plateau_stable
from AI.test_profile import is_capacity_like, profile_for


def validate_observed_steps(raw: Any, start_ms: int, end_ms: int, test_type: str) -> list[dict]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 512:
        raise LocalizedError("api.dashboard.observed_steps.list")
    if not is_capacity_like(profile_for(test_type).mode):
        raise LocalizedError("api.dashboard.observed_steps.mode")
    previous_end = start_ms
    intervals = []
    for index, item in enumerate(raw, 1):
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip() or len(name) > 200 or not name.isprintable():
            raise LocalizedError("api.dashboard.observed_steps.name", index=index)
        left, right = item.get("start_ms"), item.get("end_ms")
        if type(left) is not int or type(right) is not int or not start_ms <= left < right <= end_ms:
            raise LocalizedError("api.dashboard.observed_steps.bounds", index=index)
        if left < previous_end:
            raise LocalizedError("api.dashboard.observed_steps.overlap", index=index)
        interval = {"name": name.strip(), "start_ms": left, "end_ms": right}
        if "planned_rps" in item:
            rate = item["planned_rps"]
            if type(rate) not in (int, float) or not 0 <= rate <= sys.float_info.max:
                raise LocalizedError("api.dashboard.observed_steps.rate", index=index)
            interval["planned_rps"] = rate
        intervals.append(interval)
        previous_end = right
    return intervals


def observed_load_steps(intervals: list[dict], rps: pd.Series | None, start_ts: float,
                        shift_hours: int, min_stable_minutes: float) -> list[LoadStep]:
    series = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    if rps is not None:
        series = pd.to_numeric(rps, errors="coerce").replace([math.inf, -math.inf], float("nan"))
        series.index = pd.to_datetime(series.index, utc=True)
        series = series.sort_index()
    width = bucket_width(series.index)
    steps = []
    for index, item in enumerate(intervals, 1):
        start = pd.Timestamp(item["start_ms"], unit="ms", tz="UTC")
        end = pd.Timestamp(item["end_ms"], unit="ms", tz="UTC")
        samples = series.iloc[observed_bucket_slice(series.index, start, end, width)].dropna()
        level = float(samples.mean()) if not samples.empty else None
        covered = not samples.empty and samples.index[0] - start < width and end - samples.index[-1] < 2 * width
        steps.append(LoadStep(
            index=index, label=item["name"], start=start, end=end, minute_from=minutes_from(start, start_ts),
            minute_to=minutes_from(end, start_ts), rps_level=level if level is not None and math.isfinite(level) else None,
            stable=covered and observed_plateau_stable(samples, width, min_stable_minutes),
            source="generator", shift_hours=shift_hours, planned_rps=item.get("planned_rps"),
        ))
    return steps
