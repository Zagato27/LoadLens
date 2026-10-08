"""Synthetic demo run so the product can be explored without Grafana or an LLM provider.

The run is written through the same storage functions the real pipeline uses
(`save_domain_labeled`, `save_llm_results`), so every page (archive, report,
compare, dashboard) shows it exactly like a real report. The demo service is
registered in its own project area (:data:`DEMO_AREA`) so area-scoped pages can
find it.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import psycopg2.errors

from AI.capacity_forecast import UslFit, concurrency_for
from AI.context_pack import derive_load_steps, first_rps_drop, format_iso, load_step_report_from_frames
from AI.db_store import save_domain_labeled, save_llm_results
from loadlens_app.core import _bootstrap_service_configs, _ts_conn
from settings import CONFIG

DEMO_AREA = "demo"
DEMO_RUN_NAME = "demo-checkout-step"
DEMO_SERVICE = "demo-checkout"
DEMO_TEST_TYPE = "step"
DEMO_DURATION_MIN = 60
_STEP_LEVELS = [50.0, 100.0, 150.0, 200.0, 250.0]
_STEP_MINUTES = 10
_DROP_MINUTE = (len(_STEP_LEVELS) - 1) * _STEP_MINUTES + 5
_DEGRADED_LEVEL = 180.0
_DEMO_USL = (10.0, 0.01, 0.0002)
_DEMO_THINK_S = 0.05
_DEGRADED_RESPONSE_S = 0.9


class DemoAlreadyExistsError(RuntimeError):
    def __init__(self, run_name: str, service: str):
        super().__init__(f"Демо-прогон «{run_name}» уже существует")
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
    return [
        {"label": "LT (InfluxQL): RPS by group & name", "df": _series_frame(index, {"group=checkout|name=POST /checkout": checkout, "group=search|name=GET /search": search})},
        {"label": "LT (InfluxQL): checks per second by group & check", "df": _series_frame(index, {"group=checkout|check=status is 200": [c * 0.6 for c in checks], "group=search|check=status is 200": [c * 0.4 for c in checks]})},
        {"label": "LT (InfluxQL): http_req_duration p95(seconds) by group & name", "df": _series_frame(index, {"group=checkout|name=POST /checkout": p95, "group=search|name=GET /search": [v * 0.7 for v in p95]})},
        {"label": "LT (InfluxQL): RPS sum by all groups", "df": _series_frame(index, {"all": rps})},
        {"label": "LT (InfluxQL): http_req_duration mean(seconds)", "df": _series_frame(index, {"http_req_duration": response})},
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


def _demo_results(peak_time: str, drop_time: str, system_context: dict) -> dict:
    peak = {"max_rps": 249.3, "max_time": peak_time, "drop_time": drop_time, "method": "last_stable_step"}
    lt = _analysis(
        "Есть риски",
        "Стабильная ступень 250 RPS удержана 5 минут, затем началась деградация.\n- p95 вырос с 0.25 до 0.45 с на последней ступени.\n- Ошибок менее 0.5 %, проверки проходят.",
        [{"id": "f1", "summary": "P95 времени ответа превысил 400 мс на ступени 250 RPS", "severity": "high", "component": "checkout", "start_time": peak_time, "end_time": drop_time, "peak_time": drop_time,
          "evidence_summary": "Рост p95 совпал с последней ступенью нагрузки.", "evidence_items": [{"metric": "http_req_duration p95", "observed_value": "0.45 с", "threshold": "0.30 с", "note": "group=checkout"}]}],
        [{"summary": "Проверить пул соединений orders-api к БД", "details": "Сверить лимиты пула с пиковой конкурентностью; повторить прогон после увеличения пула.", "priority": "high", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
        peak,
    )
    jvm = _analysis(
        "Успешно",
        "Heap стабилен, GC не влияет на латентность.\n- Heap used колеблется 35–85 % от максимума без роста базовой линии.\n- CPU процесса до 70 %.",
        [{"id": "f1", "summary": "CPU процесса orders-api достигает 70 % на ступени 250 RPS", "severity": "low", "component": "orders-api", "evidence_summary": "process_cpu_usage 0.70 в 10:45–10:55"}],
        [{"summary": "Запас по CPU достаточен, действий не требуется", "details": "При планах роста нагрузки выше 300 RPS добавить реплику.", "priority": "low", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
    )
    ms = _analysis(
        "Есть риски",
        "Среднее время ответа orders-api выросло вдвое на последней ступени.\n- catalog-api стабилен.",
        [{"id": "f1", "summary": "orders-api: среднее время ответа 0.29 с при 250 RPS против 0.10 с на ступени 50 RPS", "severity": "medium", "component": "orders-api", "evidence_summary": "average request time 0.29 с"}],
        [{"summary": "Профилировать обработчик POST /checkout", "details": "Найти блокирующие вызовы к БД и кэшу; цель — p95 < 300 мс на 250 RPS.", "priority": "medium", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
    )
    hr = _analysis(
        "Успешно",
        "Ресурсы узлов не являются узким местом.\n- CPU узлов до 73 %, память до 66 %.",
        [], [{"summary": "Ресурсов достаточно", "details": "Мониторить CPU worker-1 при росте нагрузки.", "priority": "low", "affected_components": ["worker-1"], "for_finding_ids": []}],
    )
    no_data = _analysis("Недостаточно данных", "В демонстрационном прогоне метрики этого домена не собирались.", [], [])
    final = _analysis(
        "Есть риски",
        "Целевой RPS достигнут на стабильной ступени, но нарушен порог p95.\n- stable_max = 249 RPS ≥ цель 200.\n- p95 0.45 с превышает порог 0.30 с на последней ступени.\n- Ошибок менее 0.5 %.",
        [{"id": "f1", "summary": "Деградация времени ответа checkout на ступени 250 RPS", "severity": "high", "component": "orders-api", "start_time": peak_time, "end_time": drop_time, "peak_time": drop_time,
          "evidence_summary": "p95 0.45 с, среднее время ответа 0.29 с, CPU процесса 70 %", "evidence_items": [{"metric": "http_req_duration p95", "observed_value": "0.45 с", "threshold": "0.30 с", "note": "group=checkout"}]}],
        [{"summary": "Увеличить пул соединений orders-api и повторить ступень 250 RPS", "details": "После изменения повторить step-тест и убедиться, что p95 остаётся ниже 300 мс на протяжении 10 минут.", "priority": "high", "affected_components": ["orders-api"], "for_finding_ids": ["f1"]}],
        peak,
    )
    checks = [
        {"name": "target_rps", "threshold": 200, "actual": 249.3, "passed": True, "severity": "critical", "message": "RPS 249.3 (stable_max) ≥ целевой 200"},
        {"name": "p95_latency", "threshold": 300, "actual": 450.0, "passed": False, "severity": "warning", "message": "P95 latency 450.0 ms > порог 300 ms"},
        {"name": "error_rate", "threshold": 1.0, "actual": 0.4, "passed": True, "severity": "warning", "message": "Error rate 0.40% ≤ порог 1.0%"},
    ]
    return {
        "jvm": "", "database": "", "kafka": "", "ms": "", "hard_resources": "", "lt_framework": "", "final": "",
        "jvm_parsed": jvm, "database_parsed": no_data, "kafka_parsed": no_data, "ms_parsed": ms,
        "hard_resources_parsed": hr, "lt_framework_parsed": lt, "final_parsed": final,
        "scores": {"jvm": _scores(), "database": {}, "kafka": {}, "microservices": _scores(), "hard_resources": _scores(), "lt_framework": _scores(), "final": _scores()},
        "sla_verdict": "Есть риски",
        "sla_checks": checks,
        "sla_summary": "SLA verdict: Есть риски. Пройдено: target_rps, error_rate; Нарушено: p95_latency",
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


def _demo_system_context() -> dict:
    return {
        "enabled": True,
        "schema_version": 1,
        "system": {"name": "Checkout Platform (демо)", "domain": "e-commerce", "description": "Оформление заказов и оплат. Демонстрационные данные LoadLens.", "test_goal": "Найти максимальную стабильную нагрузку на checkout"},
        "architecture": {"style": "microservices", "components": [{"id": "orders-api", "name": "orders-api", "role": "оформление заказа", "criticality": "high", "technologies": ["Java", "Spring"]}, {"id": "catalog-api", "name": "catalog-api", "role": "каталог", "criticality": "medium", "technologies": ["Java"]}], "dependencies": [{"from": "orders-api", "to": "PostgreSQL", "kind": "sync", "purpose": "заказы"}], "data_stores": [{"id": "orders-db", "type": "PostgreSQL", "purpose": "заказы", "used_by": ["orders-api"]}]},
        "load_model": {"entrypoints": [{"id": "checkout", "name": "POST /checkout", "kind": "http", "business_priority": "high"}], "critical_user_flows": [{"id": "buy", "name": "Покупка", "steps": ["catalog-api", "orders-api"], "success_signals": ["2xx"]}], "expected_hotspots": ["orders-api"]},
        "operational_context": {"known_constraints": [], "known_risks": ["Пул соединений orders-api ограничен 20"], "normal_degradation_rules": [], "analysis_focus": ["p95 checkout"]},
    }


def seed_demo_run(now: datetime | None = None) -> dict:
    """Creates the demo run. Raises DemoAlreadyExistsError if it is already present."""
    if _run_exists(DEMO_RUN_NAME):
        raise DemoAlreadyExistsError(DEMO_RUN_NAME, DEMO_SERVICE)
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
    results = _demo_results(peak_time, drop_time, _demo_system_context())
    results["contexts"] = {
        "final": {
            "load_steps": [step.to_dict() for step in steps],
            "rps_drop_iso": format_iso(drop, shift_hours) if drop is not None else None,
            "lt_series_labels": [frame["label"] for frame in domains["lt_framework"]],
        }
    }
    results["load_step_table"] = load_step_report_from_frames(
        steps,
        domains["lt_framework"],
        rps_query="LT (InfluxQL): RPS sum by all groups",
        latency_query="LT (InfluxQL): http_req_duration p95(seconds) by group & name",
    )
    save_llm_results(results, run_meta, storage_cfg)
    return {
        "run_name": DEMO_RUN_NAME,
        "service": DEMO_SERVICE,
        "area": DEMO_AREA,
        "report_url": f"/reports/{run_meta['run_id']}",
        "points": len(index),
        "stable_max": round(max(_STEP_LEVELS) * (1 - 0.003), 1),
    }


__all__ = ["DEMO_AREA", "DEMO_RUN_NAME", "DEMO_SERVICE", "DemoAlreadyExistsError", "demo_report_url", "seed_demo_run"]
