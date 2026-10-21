# SPDX-License-Identifier: Apache-2.0
"""Basic metrics of a running test: RPS, errors, p95 and virtual users from the load tool queries.

The queries are the rows of the ``lt_framework`` domain that carry these roles (installed query packs),
filtered by the run tag where the tool tagged its metrics. If the filtered RPS query is empty while the
unfiltered one has data, the tool did not tag this run (for example a JMeter plan with a fixed
``application``), and the panel and the report read the whole window instead.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from AI.i18n import get_locale

from .messages import msg, ru_text, t
from .store import INSTANCE_STATUS_TITLES, RUN_CANCELLED, RUN_DONE, RUN_FAILED, RunRecord
from .tools import ToolTemplate

LT_DOMAIN = "lt_framework"
LIVE_ROLES = ("lt.rps_total", "lt.error_rate_pct", "lt.p95_total", "lt.vus")
# Live rows of one snapshot read at the same time (four roles plus the instances row).
LIVE_FETCH_WORKERS = 5
SUMMED_ROLES = frozenset({"lt.rps_total"})
INSTANCE_ROLE = "lt.rps_by_instance"
# Windows of the live charts in minutes; 0 is the whole run.
LIVE_WINDOWS = (0, 15, 30, 60)
# Points re-read before ``since``: the newest bucket may still have been filling at the previous read.
OVERLAP_STEPS = 2
SLA_LINES = ("target_rps", "max_error_rate_pct", "max_p95_ms")
# A test that runs this long without a single RPS point is worth an explanation, not just empty charts.
NO_DATA_AFTER_SEC = 45
MAX_POINTS = 600
NICE_STEPS = (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)
# Texts of a snapshot are rendered in the language of the request (the caches below are per language).
UNFILTERED_KEY = "lt.live.unfiltered"
# The ru text of the note, for callers that compare with it.
UNFILTERED_NOTE = ru_text(UNFILTERED_KEY)

# Test window hint of the new report form (GET /lt_windows).
WINDOWS_DEFAULT_HOURS = 6.0
WINDOWS_MIN_HOURS = 0.25
WINDOWS_MAX_HOURS = 24.0
WINDOWS_GAP_STEPS = 3  # positive RPS points further apart than this many steps belong to different tests
WINDOWS_KEEP = 3
WINDOWS_RUNS_KEEP = 5
WINDOWS_CACHE_SEC = 60.0
WINDOWS_CACHE_MAX = 128
_windows_cache: dict[tuple[str, str, float, str], tuple[float, dict[str, Any]]] = {}
_windows_lock = threading.Lock()


def installed_filter_packs(cfg: Optional[dict]) -> list[str]:
    """Installed load tool packs of the service that can filter by the run tag."""
    from AI.query_packs import load_catalog as load_packs

    block = ((cfg or {}).get("queries") or {}).get(LT_DOMAIN) if isinstance((cfg or {}).get("queries"), dict) else None
    entries = block.get("packs") if isinstance(block, dict) else None
    ids = [str(entry.get("id")) for entry in entries or [] if isinstance(entry, dict) and entry.get("id")]
    if not ids:
        return []
    catalog = load_packs()
    return [pack_id for pack_id in ids if getattr(catalog.pack(pack_id), "run_filter", None) is not None]


def run_filter_packs(run: RunRecord, tools: dict[str, ToolTemplate], cfg: Optional[dict] = None) -> list[str]:
    """Packs whose metrics the instances of this run tagged with the run tag.

    A tool tags them when its template lists the packs. A CI pipeline tags them only when its generator says
    so (the pipeline passes LOADLENS_RUN_TAG to the tool); then every installed pack with a run filter counts.
    A pipeline that does not tag its metrics turns the filter off for the run: it would hide them.
    """
    packs: list[str] = []
    tagged_ci = False
    for inst in run.instances:
        if inst.launched_at is None:
            continue
        if inst.ci_kind:
            if not inst.ci_tagged:
                return []
            tagged_ci = True
            continue
        tool = tools.get(inst.tool) if inst.tool else None
        if tool is not None:
            packs.extend(tool.run_filter_packs)
    if tagged_ci:
        try:
            packs.extend(installed_filter_packs(cfg if cfg is not None else _scope_config(run)))
        except Exception:
            pass
    return list(dict.fromkeys(packs))


def choose_step(start_ts: float, end_ts: float) -> int:
    span = max(1.0, end_ts - start_ts)
    wanted = span / 240
    return next((step for step in NICE_STEPS if step >= wanted), NICE_STEPS[-1])


def _scope_config(run: RunRecord) -> dict:
    from update_page import _effective_config_for_scope

    return _effective_config_for_scope(run.project_area or run.service, run.service)


def _block_rows(queries: dict, roles: tuple[str, ...]) -> tuple[str, dict[str, tuple[str, str, list[str]]]]:
    """(language, {role: (label, query, keys)}) of the lt_framework block."""
    from AI.query_packs import filled_languages, labels_for_role
    from AI.query_packs.blocks import LANG_FIELDS

    block = queries.get(LT_DOMAIN) if isinstance(queries, dict) else None
    if not isinstance(block, dict):
        return "", {}
    languages = filled_languages(block)
    if not languages:
        return "", {}
    lang = languages[0]
    qkey, kkey = LANG_FIELDS[lang]
    labels = list(block.get("labels") or [])
    texts = list(block.get(qkey) or [])
    keys = list(block.get(kkey) or [])
    from AI.query_packs import derived_rows

    rows: dict[str, tuple[str, str, list[str]]] = {}
    summed = [label for label in derived_rows(block) if label in labels]
    for role in roles:
        candidates = labels_for_role(block, role)
        if role == "lt.rps_total" and summed and lang == "promql":
            # A complex run: the sum of every tool's RPS is the load, not the first tool alone.
            candidates = summed + candidates
        for label in candidates:
            index = labels.index(label)
            text = str(texts[index] if index < len(texts) else "").strip()
            if text:
                row_keys = keys[index] if index < len(keys) and isinstance(keys[index], list) else []
                rows[role] = (label, text, [str(k) for k in row_keys])
                break
    return lang, rows


def _combine(series: dict[str, list[tuple[float, float]]], summed: bool) -> list[tuple[float, float]]:
    merged: dict[float, float] = {}
    for points in series.values():
        for ts, value in points:
            if ts in merged:
                merged[ts] = merged[ts] + value if summed else max(merged[ts], value)
            else:
                merged[ts] = value
    return sorted(merged.items())


def _downsample(points: list[tuple[float, float]], step: int) -> list[tuple[float, float]]:
    """Averages points into ``step``-second buckets so a long run stays light for the browser."""
    if len(points) <= MAX_POINTS:
        return points
    buckets: dict[int, list[float]] = {}
    for ts, value in points:
        buckets.setdefault(int(ts // step) * step, []).append(value)
    return [(float(ts), sum(values) / len(values)) for ts, values in sorted(buckets.items())]


def _run_all(jobs: Iterable[Callable[[], Any]]) -> list[Any]:
    """Results of the jobs in their order, an exception object where a job raised; the jobs run at once.

    The live rows are independent HTTP queries to the metrics source (Prometheus or Grafana proxy), each of
    them 0.2–3 s: read one after another they made the first charts wait for their sum.
    """
    from concurrent.futures import ThreadPoolExecutor

    pending = list(jobs)
    if not pending:
        return []

    def call(job: Callable[[], Any]) -> Any:
        try:
            return job()
        except Exception as exc:  # reported per row, the other rows still arrive
            return exc

    if len(pending) == 1:
        return [call(pending[0])]
    with ThreadPoolExecutor(max_workers=min(len(pending), LIVE_FETCH_WORKERS), thread_name_prefix="live-fetch") as pool:
        return list(pool.map(call, pending))


def _fetch(cfg: dict, lang: str, row: tuple[str, str, list[str]], start: float, end: float, step: int, summed: bool):
    from loadlens_app.query_preview import fetch_points

    label, text, keys = row
    series = fetch_points(cfg, domain=LT_DOMAIN, lang=lang, query=text, start_ts=start, end_ts=end, step=f"{step}s", label_keys=keys)
    return _downsample(_combine(series, summed), step)


def _filtered(cfg: dict, run: RunRecord, packs: list[str]) -> dict:
    from AI.query_packs import apply_run_filter

    queries, _ = apply_run_filter(cfg.get("queries") or {}, run.run_tag, packs)
    return queries


def _window(run: RunRecord, now: Optional[float]) -> tuple[float, float]:
    current = now if now is not None else time.time()
    begin = run.started_at or min((i.launched_at for i in run.instances if i.launched_at), default=None) or run.created_at
    finish = run.finished_at
    start = begin.timestamp() if isinstance(begin, datetime) else current - 900
    end = finish.timestamp() if isinstance(finish, datetime) else current
    return start, max(end, start + 60)


def _ms(value: Optional[datetime]) -> Optional[int]:
    return int(value.timestamp() * 1000) if isinstance(value, datetime) else None


def run_events(run: RunRecord) -> list[dict[str, Any]]:
    """Moments to mark on the charts: starts and ends of instances, stop requests and the run limit."""
    events: list[dict[str, Any]] = []
    stop_all = _ms(run.stop_all_at)
    for inst in run.instances:
        if inst.started_at:
            events.append({"ts": _ms(inst.started_at), "kind": "start", "label": t("lt.live.event.start", name=inst.name)})
        if inst.stop_sent_at and stop_all is None:
            events.append({"ts": _ms(inst.stop_sent_at), "kind": "stop", "label": t("lt.live.event.stop", name=inst.name)})
        if inst.finished_at:
            title = INSTANCE_STATUS_TITLES.get(inst.status) or t("lt.live.event.finished")
            status = f"{title[:1].lower()}{title[1:]}"
            events.append({"ts": _ms(inst.finished_at), "kind": "finish", "label": t("lt.live.event.finish", name=inst.name, status=status)})
    if stop_all is not None:
        events.append({"ts": stop_all, "kind": "stop", "label": t("lt.live.event.run_stop")})
    begin = run.started_at or min((i.started_at for i in run.instances if i.started_at), default=None)
    max_seconds = int(run.options.get("max_seconds") or 0)
    if isinstance(begin, datetime) and max_seconds:
        events.append({"ts": int((begin.timestamp() + max_seconds) * 1000), "kind": "limit", "label": t("lt.live.event.limit")})
    return sorted(events, key=lambda event: event["ts"])


def sla_lines(cfg: dict) -> dict[str, float]:
    """Thresholds of the service SLA drawn on the charts: target RPS, error rate and p95."""
    sla = cfg.get("sla") if isinstance(cfg.get("sla"), dict) else {}
    out: dict[str, float] = {}
    for key in SLA_LINES:
        value = sla.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            out[key] = float(value)
    return out


def _clock(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _points(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[int(ts * 1000), round(value, 4)] for ts, value in points]


def _instance_series(run: RunRecord, cfg: dict, lang: str, row: tuple[str, str, list[str]], start: float, end: float,
                     step: int) -> list[dict[str, Any]]:
    """RPS of every instance by its ``lt_instance`` tag; series without the tag are not instances of this run."""
    from loadlens_app.query_preview import fetch_points

    label, text, keys = row
    names = {inst.tag: inst.name for inst in run.instances}
    series = fetch_points(cfg, domain=LT_DOMAIN, lang=lang, query=text, start_ts=start, end_ts=end, step=f"{step}s", label_keys=keys)
    out = []
    for key, points in sorted(series.items()):
        tag = key.split("=", 1)[1] if "=" in key else key
        if tag not in names:
            continue
        out.append({"tag": tag, "name": names[tag], "points": _points(_downsample(sorted(points), step))})
    return out


def live_snapshot(run: RunRecord, tools: dict[str, ToolTemplate], now: Optional[float] = None, *, window_min: int = 0,
                  since: Optional[float] = None, mode: str = "", roles: tuple[str, ...] = LIVE_ROLES,
                  with_instances: bool = True) -> dict[str, Any]:
    """Points of the live roles of a run.

    ``window_min``: the last minutes of the run (0 — from the first start to the last finish or now).
    ``since`` (unix seconds): only points from then on, a couple of steps earlier, for the page that already
    has the rest; the step stays the one of the whole window so the points fit together. ``mode`` keeps the
    choice of the first read: ``filtered`` by the run tag or ``plain`` (the tag found no data).
    """
    current = now if now is not None else time.time()
    run_start, end = _window(run, current)
    window_min = window_min if window_min in LIVE_WINDOWS else 0
    start = max(run_start, end - window_min * 60) if window_min else run_start
    step = choose_step(start, end)
    fetch_start = max(start, since - OVERLAP_STEPS * step) if since else start
    base: dict[str, Any] = {
        "ok": False, "message": "", "series": {}, "instances": [], "start": start, "end": end, "step_sec": step,
        "window_min": window_min, "since": since, "events": run_events(run), "sla": {}, "now": current,
    }
    try:
        cfg = _scope_config(run)
    except Exception as exc:
        return {**base, "message": t("lt.live.settings_unreadable", error=exc)}
    base["sla"] = sla_lines(cfg)
    packs = run_filter_packs(run, tools, cfg)
    filtered = bool(packs) and mode != "plain"
    wanted = tuple(role for role in roles if role in LIVE_ROLES) or LIVE_ROLES
    roles = wanted + ((INSTANCE_ROLE,) if with_instances else ())
    lang, rows = _block_rows(_filtered(cfg, run, packs) if filtered else (cfg.get("queries") or {}), roles)
    if not any(role in rows for role in wanted):
        return {**base, "message": t("lt.live.no_queries")}

    def read(rows_: dict, lang_: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        """Reads every live row at once: each is one HTTP query to the source, and the page waits for all."""
        jobs: list[tuple[str, tuple[str, str, list[str]], Callable[[], Any]]] = []
        for role in wanted:
            row = rows_.get(role)
            if row is not None:
                jobs.append((role, row, lambda row=row, role=role: _fetch(cfg, lang_, row, fetch_start, end, step, role in SUMMED_ROLES)))
        if INSTANCE_ROLE in rows_ and len(run.instances) > 1:
            row = rows_[INSTANCE_ROLE]
            jobs.append((INSTANCE_ROLE, row, lambda row=row: _instance_series(run, cfg, lang_, row, fetch_start, end, step)))
        series: dict[str, Any] = {}
        instances: list[dict[str, Any]] = []
        errors: list[str] = []
        outcomes = _run_all(job for _role, _row, job in jobs)
        for (role, row, _job), outcome in zip(jobs, outcomes):
            if isinstance(outcome, BaseException):
                errors.append(f"{row[0]}: {outcome}")
            elif role == INSTANCE_ROLE:
                instances = outcome
            else:
                series[role] = {"label": row[0], "points": _points(outcome)}
        return series, instances, errors

    series, instances, errors = read(rows, lang)
    out: dict[str, Any] = {**base, "ok": True, "series": series, "instances": instances, "filtered": filtered,
                           "mode": "filtered" if filtered else "plain"}
    rps = series.get("lt.rps_total")
    used_rows = rows
    both_empty = False
    # A full read decides whether the tool tagged this run; reads of new points keep that choice.
    if filtered and not since and not mode and rps is not None and not rps["points"]:
        plain_lang, plain_rows = _block_rows(cfg.get("queries") or {}, roles)
        plain = plain_rows.get("lt.rps_total")
        try:
            plain_points = _fetch(cfg, plain_lang, plain, fetch_start, end, step, True) if plain else []
        except Exception:
            plain_points = []
        if plain_points:
            series, instances, errors = read(plain_rows, plain_lang)
            used_rows = plain_rows
            out.update(series=series, instances=instances, filtered=False, mode="plain", message=t(UNFILTERED_KEY))
        else:
            # Nothing either way: show the query as the service has it, the one to compare with Grafana.
            both_empty = True
            used_rows = plain_rows if plain else rows
    rps_row = used_rows.get("lt.rps_total")
    # The query actually read, to compare with the panel in Grafana when the charts stay empty.
    out["rps_query"] = {"label": rps_row[0], "query": rps_row[1]} if rps_row else None
    last = (out["series"].get("lt.rps_total") or {}).get("points") or []
    out["fresh_at"] = last[-1][0] / 1000 if last else None
    started = any(inst.started_at for inst in run.instances)
    if rps_row and not since and not last and not errors and started and current - run_start > NO_DATA_AFTER_SEC:
        if both_empty:
            key = "lt.live.no_points_both"
        elif out["mode"] == "filtered":
            key = "lt.live.no_points_tagged"
        else:
            key = "lt.live.no_points"
        out["message"] = t(key, query=rps_row[0], start=_clock(fetch_start), end=_clock(end), tag=run.run_tag)
    if errors and not out["series"]:
        out["ok"] = False
        out["message"] = "; ".join(errors)[:600]
    elif errors:
        out["message"] = (out["message"] + "; " if out["message"] else "") + "; ".join(errors)[:400]
    return out


BRIEF_ROLES = ("lt.rps_total", "lt.error_rate_pct")
BRIEF_WINDOW_MIN = 15
BRIEF_CACHE_SEC = 10.0
_brief_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_brief_lock = threading.Lock()


def live_brief(run: RunRecord, tools: dict[str, ToolTemplate], now: Optional[float] = None) -> dict[str, Any]:
    """RPS and errors of the last 15 minutes for the dashboard, cached for a few seconds per run and language.

    Several open dashboards ask every 15 s; one read of the metrics storage serves them all.
    """
    current = time.monotonic()
    cache_key = (run.run_id, get_locale())
    with _brief_lock:
        hit = _brief_cache.get(cache_key)
        if hit is not None and current - hit[0] < BRIEF_CACHE_SEC:
            return hit[1]
    snapshot = live_snapshot(run, tools, now, window_min=BRIEF_WINDOW_MIN, roles=BRIEF_ROLES, with_instances=False)
    series = snapshot.get("series") or {}
    rps = (series.get("lt.rps_total") or {}).get("points") or []
    errors = (series.get("lt.error_rate_pct") or {}).get("points") or []
    brief = {
        "ok": bool(snapshot.get("ok")),
        "message": snapshot.get("message") or "",
        "rps": rps[-1][1] if rps else None,
        "errors": errors[-1][1] if errors else None,
        "points": rps,
        "fresh_at": snapshot.get("fresh_at"),
        "start": snapshot.get("start"),
        "end": snapshot.get("end"),
    }
    with _brief_lock:
        if len(_brief_cache) >= 256:
            _brief_cache.clear()
        _brief_cache[cache_key] = (current, brief)
    return brief


def clear_brief_cache() -> None:
    with _brief_lock:
        _brief_cache.clear()


def report_filter(run: RunRecord, tools: dict[str, ToolTemplate], start: float, end: float) -> tuple[list[str], str]:
    """Packs to filter in the report and a note for the run (a dictionary message); unfiltered when the tag found no data."""
    try:
        cfg: Optional[dict] = _scope_config(run)
    except Exception:
        cfg = None
    packs = run_filter_packs(run, tools, cfg or {})
    if not packs or cfg is None:
        return packs, ""
    try:
        step = choose_step(start, end)
        lang, rows = _block_rows(_filtered(cfg, run, packs), ("lt.rps_total",))
        row = rows.get("lt.rps_total")
        if row is None or _fetch(cfg, lang, row, start, end, step, True):
            return packs, ""
        _, plain_rows = _block_rows(cfg.get("queries") or {}, ("lt.rps_total",))
        plain = plain_rows.get("lt.rps_total")
        if plain is not None and _fetch(cfg, lang, plain, start, end, step, True):
            return [], msg("lt.live.unfiltered_report")
    except Exception:
        return packs, ""
    return packs, ""


def trim_bounds(points: list[tuple[float, float]], start: float, end: float, step: float) -> Optional[tuple[float, float]]:
    """From the first to the last point with load (above zero), one step wider each side, inside start..end.

    Pauses in the middle stay in the window; fewer than two points with load say nothing about it (None).
    """
    loaded = sorted(ts for ts, value in points if isinstance(value, (int, float)) and math.isfinite(value) and value > 0)
    if len(loaded) < 2:
        return None
    return max(start, loaded[0] - step), min(end, loaded[-1] + step)


def rps_bounds(run: RunRecord, start: float, end: float, packs: list[str]) -> Optional[tuple[float, float]]:
    """The window with load by the summed RPS of the run (the report filter applied); None when unknown."""
    try:
        cfg = _scope_config(run)
        queries = _filtered(cfg, run, packs) if packs else (cfg.get("queries") or {})
        lang, rows = _block_rows(queries, ("lt.rps_total",))
        row = rows.get("lt.rps_total")
        if row is None:
            return None
        step = choose_step(start, end)
        points = _fetch(cfg, lang, row, start, end, step, True)
    except Exception:
        return None
    return trim_bounds(points, start, end, step)


# ---- test window hint ------------------------------------------------------------------------------


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _service_config(area: str, service: str) -> dict:
    from update_page import _effective_config_for_scope

    return _effective_config_for_scope(area or service, service)


def clamp_hours(hours: Any) -> float:
    """Look-back of the window hint in hours, kept within 15 minutes and 24 hours."""
    try:
        value = float(hours)
    except (TypeError, ValueError):
        return WINDOWS_DEFAULT_HOURS
    if not math.isfinite(value):
        return WINDOWS_DEFAULT_HOURS
    return min(WINDOWS_MAX_HOURS, max(WINDOWS_MIN_HOURS, value))


def activity_segments(points: list[tuple[float, float]], step: float, gap_steps: int = WINDOWS_GAP_STEPS) -> list[dict[str, float]]:
    """Stretches of ``points`` with a value above zero, oldest first.

    Zero values and missing points are pauses; positive points more than ``gap_steps`` steps apart start a
    new stretch. A single point is noise, not a test, and is dropped.
    """
    segments: list[dict[str, float]] = []
    current: Optional[dict[str, float]] = None
    limit = gap_steps * float(step) + 1e-6
    for ts, value in sorted(points):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            continue
        if current is not None and ts - current["end"] <= limit:
            current["end"] = ts
            current["peak"] = max(current["peak"], value)
            current["points"] += 1
        else:
            current = {"start": ts, "end": ts, "peak": value, "points": 1}
            segments.append(current)
    return [seg for seg in segments if seg["points"] >= 2]


def _window_step(start: float, end: float, cfg: dict) -> int:
    from loadlens_app.limits import limit

    step = choose_step(start, end)
    try:
        max_points = limit("max_points_per_series", cfg)
    except Exception:
        max_points = None
    if max_points:
        step = max(step, math.ceil((end - start) / max_points))
    return step


def finished_runs(service: str, start: float, end: float, keep: int = WINDOWS_RUNS_KEEP) -> list[dict[str, Any]]:
    """Finished load test runs of the service that overlap ``start``..``end``, newest first."""
    from .store import get_store

    out: list[dict[str, Any]] = []
    for run in get_store().list(limit=50, statuses=(RUN_DONE, RUN_FAILED, RUN_CANCELLED), services=[service]):
        begin = run.started_at or min((i.started_at for i in run.instances if i.started_at), default=None)
        finish = run.finished_at
        if not isinstance(begin, datetime) or not isinstance(finish, datetime):
            continue
        b, f = begin.timestamp(), finish.timestamp()
        if f <= b or f < start or b > end:
            continue
        out.append({
            "run_id": run.run_id,
            "run_name": run.run_name,
            "status": run.status,
            "start": _iso(b),
            "end": _iso(f),
            "duration_sec": int(round(f - b)),
        })
    out.sort(key=lambda item: item["start"], reverse=True)
    return out[:keep]


def load_windows(area: str, service: str, hours: Any = WINDOWS_DEFAULT_HOURS, now: Optional[float] = None) -> dict[str, Any]:
    """Recent load of a service: stretches with RPS above zero and finished load test runs.

    The RPS comes from the ``lt.rps_total`` row of the service's ``lt_framework`` queries (the summed row
    of a complex run on PromQL), over the last ``hours`` with :func:`choose_step`. The three latest
    stretches are returned newest first as ``{start, end, duration_sec, rps_peak, ongoing}``. Source errors
    do not fail the call: ``ok`` is false, ``message`` says why in a few words and ``detail`` carries the
    error text; the runs are still listed.
    """
    span_hours = clamp_hours(hours)
    end = now if now is not None else time.time()
    start = end - span_hours * 3600
    out: dict[str, Any] = {
        "ok": True, "message": "", "detail": "", "service": service, "hours": span_hours,
        "start": _iso(start), "end": _iso(end), "step_sec": None, "query": "", "windows": [], "runs": [],
    }
    try:
        out["runs"] = finished_runs(service, start, end)
    except Exception:
        out["runs"] = []
    try:
        cfg = _service_config(area, service)
    except Exception as exc:
        out.update(ok=False, message=t("lt.live.settings_unreadable_short"), detail=str(exc)[:300])
        return out
    lang, rows = _block_rows(cfg.get("queries") or {}, ("lt.rps_total",))
    row = rows.get("lt.rps_total")
    if row is None:
        out.update(ok=False, message=t("lt.live.no_rps_query"))
        return out
    step = _window_step(start, end, cfg)
    out["step_sec"] = step
    out["query"] = row[0]
    try:
        points = _fetch(cfg, lang, row, start, end, step, True)
    except Exception as exc:
        out.update(ok=False, message=t("lt.live.source_unavailable"), detail=str(exc)[:300])
        return out
    segments = activity_segments(points, step)
    windows = []
    for seg in reversed(segments[-WINDOWS_KEEP:]):
        windows.append({
            "start": _iso(seg["start"]),
            "end": _iso(seg["end"]),
            "duration_sec": int(round(seg["end"] - seg["start"])),
            "rps_peak": round(seg["peak"], 1),
            "ongoing": seg["end"] >= end - WINDOWS_GAP_STEPS * step,
        })
    out["windows"] = windows
    return out


def cached_load_windows(area: str, service: str, hours: Any = WINDOWS_DEFAULT_HOURS) -> dict[str, Any]:
    """:func:`load_windows` cached for a minute per (area, service, hours, language): forms ask on every service pick."""
    key = (area or "", service, clamp_hours(hours), get_locale())
    current = time.monotonic()
    with _windows_lock:
        hit = _windows_cache.get(key)
        if hit is not None and current - hit[0] < WINDOWS_CACHE_SEC:
            return hit[1]
    result = load_windows(area, service, key[2])
    with _windows_lock:
        if len(_windows_cache) >= WINDOWS_CACHE_MAX:
            for stale in [k for k, (at, _) in _windows_cache.items() if current - at >= WINDOWS_CACHE_SEC] or list(_windows_cache)[:1]:
                _windows_cache.pop(stale, None)
        _windows_cache[key] = (current, result)
    return result


def clear_windows_cache() -> None:
    with _windows_lock:
        _windows_cache.clear()


__all__ = [
    "LIVE_ROLES",
    "UNFILTERED_NOTE",
    "activity_segments",
    "cached_load_windows",
    "choose_step",
    "clamp_hours",
    "clear_windows_cache",
    "finished_runs",
    "live_brief",
    "live_snapshot",
    "run_events",
    "sla_lines",
    "load_windows",
    "installed_filter_packs",
    "report_filter",
    "rps_bounds",
    "run_filter_packs",
    "trim_bounds",
]
