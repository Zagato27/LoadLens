# SPDX-License-Identifier: Apache-2.0
"""Synthetic demo run so the product can be explored without Grafana or an LLM provider.

The run is written through the same storage functions the real pipeline uses
(`save_domain_labeled`, `save_llm_results`), so every page (archive, report,
compare, dashboard) shows it exactly like a real report. The demo service is
registered in its own project area (:data:`DEMO_AREA`) so area-scoped pages can
find it.

The run is written in the report language (``appearance.language``, :func:`loadlens_app.i18n.report_language`):
the model texts come from ``demo_texts.json`` (ru and en), the deterministic texts (step labels, SLA summary)
from the dictionaries, and the report JSON carries ``report_language`` like a real run.

The demo stack (docker/docker-compose.demo.yml) sets ``LOADLENS_DEMO_SEED=1``: app.py then calls
:func:`start_background_demo_seed` on start, which seeds this run once and builds a second report for the
same service through the job pipeline with the Prometheus and model stubs, so the archive, the dashboard
history, the comparison and the forecast pages have content from the first minute. Without the variable
nothing happens; with it the seed is idempotent and safe on every container start.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import psycopg2
import psycopg2.errors

from AI import storage_state
from AI.capacity_forecast import UslFit, concurrency_for
from AI.context_pack import derive_load_steps, first_rps_drop, format_iso, load_step_report_from_frames
from AI.db_store import save_domain_labeled, save_llm_results
from AI.i18n import get_locale, normalize_lang, with_locale
from AI.sla_evaluator import format_sla_summary, sla_summary_parts
from AI.verdicts import NO_DATA, PASS, RISK
from loadlens_app.core import _bootstrap_service_configs, _ts_conn
from loadlens_app.i18n import report_language as configured_report_language
from settings import CONFIG

logger = logging.getLogger(__name__)

DEMO_AREA = "demo"
DEMO_RUN_NAME = "demo-checkout-step"
DEMO_SERVICE = "demo-checkout"
DEMO_TEST_TYPE = "step"
DEMO_DURATION_MIN = 60
DEMO_RPS_LABEL = "LT (InfluxQL): RPS sum by all groups"
DEMO_P95_BY_REQUEST_LABEL = "LT (InfluxQL): http_req_duration p95(ms) by group & name"
DEMO_P95_TOTAL_LABEL = "LT (InfluxQL): http_req_duration p95(ms) total"
DEMO_ERROR_RATE_LABEL = "LT (InfluxQL): error rate (%)"
_STEP_LEVELS = [50.0, 100.0, 150.0, 200.0, 250.0]
_STEP_MINUTES = 10
_DROP_MINUTE = (len(_STEP_LEVELS) - 1) * _STEP_MINUTES + 5
_DEGRADED_LEVEL = 180.0
_DEMO_USL = (10.0, 0.01, 0.0002)
_DEMO_THINK_S = 0.05
_DEGRADED_RESPONSE_S = 0.9
DEMO_TEXTS_PATH = Path(__file__).with_name("demo_texts.json")


@lru_cache(maxsize=1)
def _all_demo_texts() -> Dict[str, Dict[str, str]]:
    return json.loads(DEMO_TEXTS_PATH.read_text(encoding="utf-8"))


def demo_texts(lang: Optional[str] = None) -> Dict[str, str]:
    """Texts of the demo run in ``lang`` (default: the current locale; unknown languages: ru)."""
    texts = _all_demo_texts()
    return texts.get(normalize_lang(lang) or get_locale()) or texts["ru"]


class DemoAlreadyExistsError(RuntimeError):
    def __init__(self, run_name: str, service: str):
        # Raised inside the /demo/seed request: the message follows the UI language.
        super().__init__(demo_texts()["exists"].replace("{run_name}", run_name))
        self.run_name = run_name
        self.service = service
        self.report_url = f"/reports/{service}/{run_name}"


def demo_report_url() -> str:
    return f"/reports/{DEMO_SERVICE}/{DEMO_RUN_NAME}"


def _storage_cfg() -> dict:
    return (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}


def _run_exists(run_name: str) -> bool:
    cfg = _storage_cfg()
    schema = cfg.get("schema", "public")
    llm_table = cfg.get("llm_table", "llm_reports")
    conn = _ts_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {schema}.{llm_table} WHERE run_name = %s LIMIT 1", (run_name,))
            return cur.fetchone() is not None
    except psycopg2.errors.UndefinedTable:
        # A fresh database: the report tables are created by the seed itself.
        return False
    finally:
        conn.close()


def _rps_profile(minutes: int, rng: random.Random) -> List[float]:
    """Staircase load: 5 steps of 10 minutes, the last one degrades under overload."""
    values: List[float] = []
    for minute in range(minutes):
        step_idx = min(minute // _STEP_MINUTES, len(_STEP_LEVELS) - 1)
        level = _STEP_LEVELS[step_idx]
        if minute >= _DROP_MINUTE:
            level = _DEGRADED_LEVEL
        values.append(max(0.0, level + rng.gauss(0, level * 0.02)))
    return values


def _series_frame(index: pd.DatetimeIndex, columns: Dict[str, List[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=index)


def _demo_response(rps_value: float, minute: int, rng: random.Random) -> float:
    """Mean response of the demo USL before the drop, then a collapsed value."""
    if minute >= _DROP_MINUTE or rps_value <= 0:
        base = _DEGRADED_RESPONSE_S
    else:
        fit = UslFit(lam=_DEMO_USL[0], sigma=_DEMO_USL[1], kappa=_DEMO_USL[2], r2=1.0, n_peak=None, x_max=None)
        concurrency = concurrency_for(fit, rps_value)
        base = _DEGRADED_RESPONSE_S if concurrency is None else concurrency / rps_value
    return max(0.01, base * (1.0 + rng.uniform(-0.03, 0.03)))


def _lt_frames(index: pd.DatetimeIndex, rps: List[float], rng: random.Random) -> List[dict]:
    checkout = [v * 0.6 for v in rps]
    search = [v * 0.4 for v in rps]
    p95 = [0.18 + (v / 250.0) * 0.12 + (0.15 if v > 230 else 0.0) + rng.gauss(0, 0.01) for v in rps]
    checks = [v * 0.996 for v in rps]
    forecast_rng = random.Random(7)
    response = [_demo_response(value, minute, forecast_rng) for minute, value in enumerate(rps)]
    vus = [value * (seconds + _DEMO_THINK_S) for value, seconds in zip(rps, response)]
    # Labels and units follow the lt-k6-influxql query pack: latencies in milliseconds.
    p95_ms = [v * 1000.0 for v in p95]
    error_pct = [0.2 + v / 250.0 * 0.05 + (0.25 if v > 230 else 0.0) for v in rps]
    return [
        {"label": "LT (InfluxQL): RPS by group & name", "df": _series_frame(index, {"group=checkout|name=POST /checkout": checkout, "group=search|name=GET /search": search})},
        {"label": "LT (InfluxQL): checks per second by group & check", "df": _series_frame(index, {"group=checkout|check=status is 200": [c * 0.6 for c in checks], "group=search|check=status is 200": [c * 0.4 for c in checks]})},
        {"label": DEMO_P95_BY_REQUEST_LABEL, "df": _series_frame(index, {"group=checkout|name=POST /checkout": p95_ms, "group=search|name=GET /search": [v * 0.7 for v in p95_ms]})},
        {"label": DEMO_RPS_LABEL, "df": _series_frame(index, {"all": rps})},
        {"label": DEMO_P95_TOTAL_LABEL, "df": _series_frame(index, {"http_req_duration": p95_ms})},
        {"label": "LT (InfluxQL): http_req_duration mean(ms)", "df": _series_frame(index, {"http_req_duration": [s * 1000.0 for s in response]})},
        {"label": DEMO_ERROR_RATE_LABEL, "df": _series_frame(index, {"http_req_failed": error_pct})},
        {"label": "LT (InfluxQL): VUs", "df": _series_frame(index, {"vus": vus})},
    ]


def _jvm_frames(index: pd.DatetimeIndex, rps: List[float], rng: random.Random) -> List[dict]:
    heap_max = 2.0 * 1024 ** 3
    heap_used = []
    current = 0.35 * heap_max
    for v in rps:
        current += v * 900_000 + rng.gauss(0, 20_000_000)
        if current > 0.85 * heap_max:
            current = 0.4 * heap_max
        heap_used.append(current)
    cpu = [min(0.95, 0.12 + v / 250.0 * 0.6 + rng.gauss(0, 0.02)) for v in rps]
    return [
        {"label": "JVM: Heap used (bytes) by (application, instance)", "df": _series_frame(index, {"application=orders-api|instance=orders-api-0": heap_used})},
        {"label": "JVM: Heap max (bytes) by (application, instance)", "df": _series_frame(index, {"application=orders-api|instance=orders-api-0": [heap_max] * len(rps)})},
        {"label": "JVM: Process CPU usage by (application, instance)", "df": _series_frame(index, {"application=orders-api|instance=orders-api-0": cpu})},
    ]


def _microservices_frames(index: pd.DatetimeIndex, rps: List[float], rng: random.Random) -> List[dict]:
    latency = [0.09 + v / 250.0 * 0.08 + (0.12 if v > 230 else 0.0) + rng.gauss(0, 0.005) for v in rps]
    return [
        {"label": "Microservices: average request time (sec)", "df": _series_frame(index, {"application=orders-api": latency, "application=catalog-api": [v * 0.6 for v in latency]})},
        {"label": "Microservices: request count rate (RPS)", "df": _series_frame(index, {"application=orders-api": [v * 0.6 for v in rps], "application=catalog-api": [v * 0.4 for v in rps]})},
    ]


def _hard_resources_frames(index: pd.DatetimeIndex, rps: List[float], rng: random.Random) -> List[dict]:
    cpu = [min(96.0, 18.0 + v / 250.0 * 55.0 + rng.gauss(0, 2.0)) for v in rps]
    mem = [52.0 + v / 250.0 * 14.0 + rng.gauss(0, 0.8) for v in rps]
    return [
        {"label": "Nodes: CPU usage (%) by node", "df": _series_frame(index, {"node=worker-1": cpu, "node=worker-2": [c * 0.9 for c in cpu]})},
        {"label": "Nodes: Memory usage (%) by node", "df": _series_frame(index, {"node=worker-1": mem, "node=worker-2": [m * 0.95 for m in mem]})},
    ]


def _domain_conf(frames: List[dict]) -> dict:
    return {"labels": [f["label"] for f in frames], "promql_queries": ["demo" for _ in frames]}


def _scores() -> dict:
    return {
        "selected_index": 0,
        "judge": {"overall": 0.86, "factual": 0.9, "completeness": 0.8, "specificity": 0.85, "rubric": {"evidence_grounding": 0.9, "issue_coverage": 0.8, "specificity": 0.85, "sla_alignment": 0.9, "actionability": 0.8}},
        "data_score": 0.78,
        "confidence": 0.82,
        "final_score": 0.83,
    }


def _analysis(verdict: str, rationale: str, findings: List[dict], actions: List[dict], peak: dict | None = None) -> dict:
    payload = {"verdict": verdict, "verdict_rationale": rationale, "confidence": 0.82, "findings": findings, "recommended_actions": actions}
    if peak:
        payload["peak_performance"] = peak
    return payload


def _demo_results(peak_time: str, drop_time: str, system_context: dict, lang: Optional[str] = None) -> dict:
    """Model answers, SLA checks and summary of the demo in ``lang`` (default: the current locale)."""
    lang = normalize_lang(lang) or get_locale()
    tx = demo_texts(lang)
    peak = {"max_rps": 249.3, "max_time": peak_time, "drop_time": drop_time, "method": "last_stable_step"}
    lt = _analysis(
        RISK,
        tx["lt_rationale"],
        [{"id": "f1", "summary": tx["lt_finding"], "severity": "high", "component": "checkout", "start_time": peak_time, "end_time": drop_time, "peak_time": drop_time,
          "evidence_summary": tx["lt_evidence"], "evidence_items": [{"metric": "http_req_duration p95", "observed_value": tx["seconds_p95"], "threshold": tx["seconds_p95_threshold"], "note": "group=checkout"}]}],
        [{"summary": tx["lt_action"], "details": tx["lt_action_details"], "priority": "high", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
        peak,
    )
    jvm = _analysis(
        PASS,
        tx["jvm_rationale"],
        [{"id": "f1", "summary": tx["jvm_finding"], "severity": "low", "component": "orders-api", "evidence_summary": tx["jvm_evidence"]}],
        [{"summary": tx["jvm_action"], "details": tx["jvm_action_details"], "priority": "low", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
    )
    ms = _analysis(
        RISK,
        tx["ms_rationale"],
        [{"id": "f1", "summary": tx["ms_finding"], "severity": "medium", "component": "orders-api", "evidence_summary": tx["ms_evidence"]}],
        [{"summary": tx["ms_action"], "details": tx["ms_action_details"], "priority": "medium", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
    )
    hr = _analysis(
        PASS,
        tx["hr_rationale"],
        [], [{"summary": tx["hr_action"], "details": tx["hr_action_details"], "priority": "low", "affected_components": ["worker-1"], "for_finding_ids": []}],
    )
    no_data = _analysis(NO_DATA, tx["no_data_rationale"], [], [])
    final = _analysis(
        RISK,
        tx["final_rationale"],
        [{"id": "f1", "summary": tx["final_finding"], "severity": "high", "component": "orders-api", "start_time": peak_time, "end_time": drop_time, "peak_time": drop_time,
          "evidence_summary": tx["final_evidence"], "evidence_items": [{"metric": "http_req_duration p95", "observed_value": tx["seconds_p95"], "threshold": tx["seconds_p95_threshold"], "note": "group=checkout"}]}],
        [{"summary": tx["final_action"], "details": tx["final_action_details"], "priority": "high", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
        peak,
    )
    final["report_language"] = lang
    # query_label/series name the chart each value was measured on, as AI/sla_evaluator._make_check stores them.
    checks = [
        {"name": "target_rps", "threshold": 200, "actual": 249.3, "passed": True, "severity": "critical", "message": tx["check_target_rps"],
         "query_label": DEMO_RPS_LABEL, "series": "all"},
        {"name": "p95_latency", "threshold": 300, "actual": 450.0, "passed": False, "severity": "warning", "message": tx["check_p95_latency"],
         "query_label": DEMO_P95_TOTAL_LABEL, "series": "http_req_duration"},
        {"name": "error_rate", "threshold": 1.0, "actual": 0.4, "passed": True, "severity": "warning", "message": tx["check_error_rate"],
         "query_label": DEMO_ERROR_RATE_LABEL, "series": "http_req_failed"},
    ]
    return {
        "report_language": lang,
        "jvm": "", "database": "", "kafka": "", "ms": "", "hard_resources": "", "lt_framework": "", "final": "",
        "jvm_parsed": jvm, "database_parsed": no_data, "kafka_parsed": no_data, "ms_parsed": ms,
        "hard_resources_parsed": hr, "lt_framework_parsed": lt, "final_parsed": final,
        "scores": {"jvm": _scores(), "database": {}, "kafka": {}, "microservices": _scores(), "hard_resources": _scores(), "lt_framework": _scores(), "final": _scores()},
        "sla_verdict": RISK,
        "sla_checks": checks,
        # The same text the evaluator writes for these checks (step test, in the report language).
        "sla_summary": format_sla_summary(sla_summary_parts(RISK, checks, "capacity"), lang),
        "system_context": system_context,
    }


def _demo_steps(start: datetime, end: datetime, shift_hours: int):
    """Five detector steps. The last plateau ends one minute before RPS falls."""
    segments = []
    for index, level in enumerate(_STEP_LEVELS):
        step_start = start + timedelta(minutes=index * _STEP_MINUTES)
        if index < len(_STEP_LEVELS) - 1:
            step_end = start + timedelta(minutes=(index + 1) * _STEP_MINUTES - 1)
            segments.append({"start": step_start.isoformat(), "end": step_end.isoformat(), "level": level, "stable": True, "duration_min": float(_STEP_MINUTES)})
            continue
        step_end = start + timedelta(minutes=_DROP_MINUTE - 1)
        drop = start + timedelta(minutes=_DROP_MINUTE)
        segments.append({
            "start": step_start.isoformat(),
            "end": step_end.isoformat(),
            "level": level,
            "stable": True,
            "duration_min": float(_DROP_MINUTE - index * _STEP_MINUTES),
            "drop_time": drop.isoformat(),
        })
    steps = derive_load_steps(segments, start.timestamp(), end.timestamp(), shift_hours, min_step_minutes=0.0)
    return steps, first_rps_drop(segments)


def demo_sla_window(steps) -> dict:
    """SLA window of the demo: the plateau of the last step (250 RPS), where the target is reached and
    p95 is breached. The tail after the RPS drop is not part of the window."""
    last = steps[-1].to_dict()
    return {
        "start": last["start_iso"],
        "end": last.get("plateau_end_iso") or last["end_iso"],
        "level": _STEP_LEVELS[-1],
        "series": "all",
        "label": DEMO_RPS_LABEL,
        "degraded_level": None,
        "degraded_checks": [],
    }


def demo_step_table(steps, lt_frames: List[dict]) -> dict:
    """Step table of the demo with the SLA step selected."""
    window = demo_sla_window(steps)
    return load_step_report_from_frames(
        steps,
        lt_frames,
        rps_query=DEMO_RPS_LABEL,
        latency_query=DEMO_P95_BY_REQUEST_LABEL,
        selected_level=window["level"],
        selected_start=window["start"],
    )


def _demo_system_context(lang: Optional[str] = None) -> dict:
    tx = demo_texts(lang)
    return {
        "enabled": True,
        "schema_version": 1,
        "system": {"name": tx["system_name"], "domain": "e-commerce", "description": tx["system_description"], "test_goal": tx["system_test_goal"]},
        "architecture": {"style": "microservices", "components": [{"id": "orders-api", "name": "orders-api", "role": tx["role_orders"], "criticality": "high", "technologies": ["Java", "Spring"]}, {"id": "catalog-api", "name": "catalog-api", "role": tx["role_catalog"], "criticality": "medium", "technologies": ["Java"]}], "dependencies": [{"from": "orders-api", "to": "PostgreSQL", "kind": "sync", "purpose": tx["purpose_orders"]}], "data_stores": [{"id": "orders-db", "type": "PostgreSQL", "purpose": tx["purpose_orders"], "used_by": ["orders-api"]}]},
        "load_model": {"entrypoints": [{"id": "checkout", "name": "POST /checkout", "kind": "http", "business_priority": "high"}], "critical_user_flows": [{"id": "buy", "name": tx["flow_buy"], "steps": ["catalog-api", "orders-api"], "success_signals": ["2xx"]}], "expected_hotspots": ["orders-api"]},
        "operational_context": {"known_constraints": [], "known_risks": [tx["risk_pool"]], "normal_degradation_rules": [], "analysis_focus": ["p95 checkout"]},
    }


def seed_demo_run(now: datetime | None = None, report_language: str | None = None) -> dict:
    """Creates the demo run in ``report_language`` (default: the ``appearance.language`` setting).
    Raises DemoAlreadyExistsError if it is already present."""
    if _run_exists(DEMO_RUN_NAME):
        raise DemoAlreadyExistsError(DEMO_RUN_NAME, DEMO_SERVICE)
    lang = normalize_lang(report_language) or configured_report_language()
    with with_locale(lang):
        return _seed_demo_run(now, lang)


def _seed_demo_run(now: datetime | None, lang: str) -> dict:
    _bootstrap_service_configs(DEMO_AREA, DEMO_SERVICE)
    rng = random.Random(42)
    end = (now or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=DEMO_DURATION_MIN)
    index = pd.date_range(start, periods=DEMO_DURATION_MIN, freq="1min", tz="UTC")
    rps = _rps_profile(DEMO_DURATION_MIN, rng)
    run_meta = {
        "run_id": f"demo-{int(end.timestamp())}",
        "run_name": DEMO_RUN_NAME,
        "service": DEMO_SERVICE,
        "test_type": DEMO_TEST_TYPE,
        "report_language": lang,
        "start_ms": int(start.timestamp() * 1000),
        "end_ms": int(end.timestamp() * 1000),
    }
    storage_cfg = _storage_cfg()
    domains = {
        "lt_framework": _lt_frames(index, rps, rng),
        "jvm": _jvm_frames(index, rps, rng),
        "microservices": _microservices_frames(index, rps, rng),
        "hard_resources": _hard_resources_frames(index, rps, rng),
    }
    for domain_key, frames in domains.items():
        save_domain_labeled(domain_key, _domain_conf(frames), frames, run_meta, storage_cfg)

    local_shift = timedelta(hours=int((CONFIG.get("llm") or {}).get("time_shift_hours", 3)))
    shift_hours = int((CONFIG.get("llm") or {}).get("time_shift_hours", 3))
    peak_time = (start + timedelta(minutes=(len(_STEP_LEVELS) - 1) * _STEP_MINUTES + 4) + local_shift).strftime("%H:%M")
    drop_time = (start + timedelta(minutes=_DROP_MINUTE) + local_shift).strftime("%H:%M")
    steps, drop = _demo_steps(start, end, shift_hours)
    results = _demo_results(peak_time, drop_time, _demo_system_context(lang), lang)
    results["contexts"] = {
        "final": {
            "load_steps": [step.to_dict() for step in steps],
            "rps_drop_iso": format_iso(drop, shift_hours) if drop is not None else None,
            "lt_series_labels": [frame["label"] for frame in domains["lt_framework"]],
        }
    }
    results["load_step_table"] = demo_step_table(steps, domains["lt_framework"])
    results["sla_window"] = demo_sla_window(steps)
    save_llm_results(results, run_meta, storage_cfg)
    return {
        "run_name": DEMO_RUN_NAME,
        "service": DEMO_SERVICE,
        "area": DEMO_AREA,
        "report_url": f"/reports/{run_meta['run_id']}",
        "points": len(index),
        "stable_max": round(max(_STEP_LEVELS) * (1 - 0.003), 1),
    }


# ---- demo stack: automatic seed on start (LOADLENS_DEMO_SEED=1) ---------------------------------------------

DEMO_SEED_ENV = "LOADLENS_DEMO_SEED"
#: Second run of the demo service, built through the job pipeline with the stubs of docker/docker-compose.demo.yml.
DEMO_REPORT_RUN_NAME = "demo-checkout-step-2"
DEMO_REPORT_WINDOW_MIN = DEMO_DURATION_MIN
DEMO_REPORT_OWNER = "demo"
#: The seeded run ends this long before the stub report starts: the dashboard shows it as the previous run.
DEMO_SEED_AGE = timedelta(days=1)
# Waits between attempts while the storage database is still starting; the last one repeats.
SEED_RETRY_DELAYS_SEC: tuple[int, ...] = (5, 10, 15, 30)
SEED_MAX_ATTEMPTS = 40

_seed_started = False
_seed_start_lock = threading.Lock()


def demo_seed_enabled(env: Optional[dict] = None) -> bool:
    """True when ``LOADLENS_DEMO_SEED`` is 1, true, yes or on (the demo compose file sets it)."""
    value = (env if env is not None else os.environ).get(DEMO_SEED_ENV, "")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def demo_status() -> dict:
    """What the dashboard banner needs: whether the demo seed is on and whether the demo run is stored.

    The database is read only when the seed is on; ``exists`` is None when it cannot be read.
    """
    enabled = demo_seed_enabled()
    exists: Optional[bool] = None
    if enabled:
        try:
            exists = _run_exists(DEMO_RUN_NAME)
        except Exception as exc:
            logger.debug("demo status: the demo run could not be looked up: %s", exc)
    return {"enabled": enabled, "exists": exists, "run_name": DEMO_RUN_NAME, "report_url": demo_report_url()}


def _stub_report_pending(dashboard, job_store, running_status: str) -> bool:
    """True when the stub report is neither stored nor being built right now."""
    if dashboard._run_name_taken(DEMO_REPORT_RUN_NAME, DEMO_SERVICE):
        return False
    running = job_store.list(statuses=[running_status], limit=50)
    return not any(job.run_name == DEMO_REPORT_RUN_NAME and job.service == DEMO_SERVICE for job in running)


def start_demo_report(now: Optional[datetime] = None) -> bool:
    """Queues the stub report of the demo service like POST /create_report does; False when it exists already.

    The job runs where every report runs (:mod:`loadlens_app.job_runner`): in the demo stack that is a thread of
    the app container, with the Prometheus stub as the metric source and the model stub as the provider. The
    window is the last :data:`DEMO_REPORT_WINDOW_MIN` minutes; the stub answers any window with the same shape.
    """
    # Lazy: the dashboard blueprint imports this module.
    from loadlens_app.blueprints import dashboard
    from loadlens_app.jobs import JOB_STATUS_RUNNING, job_store

    if not _stub_report_pending(dashboard, job_store, JOB_STATUS_RUNNING):
        return False
    end = (now or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=DEMO_REPORT_WINDOW_MIN)
    area = dashboard.resolve_report_target(DEMO_SERVICE, DEMO_AREA)
    dashboard.start_report_job(
        int(start.timestamp() * 1000), int(end.timestamp() * 1000), DEMO_SERVICE, area,
        use_llm=True, save_to_db=True, web_only=True,
        run_name=DEMO_REPORT_RUN_NAME, test_type=DEMO_TEST_TYPE, owner=DEMO_REPORT_OWNER,
    )
    return True


def ensure_demo(now: Optional[datetime] = None) -> dict:
    """Seeds the demo run and queues the stub report when they are missing; a no-op otherwise.

    Safe on every start: an existing demo run is left alone (only its project area is registered again in the
    runtime settings, which a re-created container has lost), and the stub report is queued once. Returns
    ``{"seeded": bool, "report_started": bool}``. Database errors propagate to the caller, who retries.
    """
    now = now or datetime.now(timezone.utc)
    seeded = False
    if _run_exists(DEMO_RUN_NAME):
        _bootstrap_service_configs(DEMO_AREA, DEMO_SERVICE)
    else:
        seed_demo_run(now - DEMO_SEED_AGE)
        seeded = True
    report_started = start_demo_report(now)
    if seeded or report_started:
        from loadlens_app import response_cache

        response_cache.invalidate(response_cache.DASHBOARD)
    return {"seeded": seeded, "report_started": report_started}


def _retryable(exc: BaseException) -> bool:
    return storage_state.is_storage_error(exc) or isinstance(exc, psycopg2.OperationalError)


def run_demo_seed_with_retries(
    run: Callable[[], dict] = ensure_demo,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: Optional[int] = SEED_MAX_ATTEMPTS,
) -> Optional[dict]:
    """``run()`` until it succeeds; waits :data:`SEED_RETRY_DELAYS_SEC` while the storage database is unavailable.

    Any other failure is logged once and ends the attempts (POST /demo/seed repeats the seed by hand). Returns
    what ``run()`` returned, or None when the attempts ran out or the failure was not about the database.
    """
    attempt = 0
    while True:
        try:
            result = run()
        except DemoAlreadyExistsError:
            # Another process of the same database seeded it between the check and the write.
            result = {"seeded": False, "report_started": False}
        except Exception as exc:
            if not _retryable(exc):
                logger.exception("Demo seed failed; POST /demo/seed repeats it")
                return None
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                logger.error("Demo seed postponed %d times, the storage database is still unavailable: %s", attempt, exc)
                return None
            delay = SEED_RETRY_DELAYS_SEC[min(attempt - 1, len(SEED_RETRY_DELAYS_SEC) - 1)]
            logger.warning("Demo seed postponed: the storage database is unavailable (%s); next try in %d s", exc, delay)
            sleep(delay)
            continue
        # Writing the demo data is worth a line even where INFO is not shown (as the migrations do).
        log = logger.warning if result.get("seeded") or result.get("report_started") else logger.info
        log(
            "Demo seed: run %s %s, stub report %s %s",
            DEMO_RUN_NAME, "written" if result.get("seeded") else "already stored",
            DEMO_REPORT_RUN_NAME, "queued" if result.get("report_started") else "already stored or running",
        )
        return result


def start_background_demo_seed() -> bool:
    """Runs :func:`run_demo_seed_with_retries` once per process in a daemon thread; False when already started."""
    global _seed_started
    with _seed_start_lock:
        if _seed_started:
            return False
        _seed_started = True
    threading.Thread(target=run_demo_seed_with_retries, name="loadlens-demo-seed", daemon=True).start()
    return True


__all__ = [
    "DEMO_AREA", "DEMO_REPORT_RUN_NAME", "DEMO_RUN_NAME", "DEMO_SEED_ENV", "DEMO_SERVICE", "DemoAlreadyExistsError",
    "demo_report_url", "demo_seed_enabled", "demo_status", "ensure_demo", "run_demo_seed_with_retries", "seed_demo_run",
    "start_background_demo_seed", "start_demo_report",
]
