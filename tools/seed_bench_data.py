"""Synthetic data for the LoadLens performance bench (see docs/perf-baseline.md).

One command per sandbox (made by make_sandbox.py <N>: database ll_s<N> on 127.0.0.1:55432, app copy in
%TEMP%/ll_ui_s<N>/app):

    C:/Python312/python.exe tools/seed_bench_data.py --sandbox 1 --reset
    C:/Python312/python.exe tools/seed_bench_data.py --sandbox 1 --reset --runs 1000      # the large variant

Any other local database: ``--dsn "host=127.0.0.1 port=55432 dbname=ll_s1 user=ltar password=..."`` (or
``--host/--port/--dbname/--user/--password``) plus ``--runtime-json <sandbox>/settings_runtime.json``.

What it writes, straight into the database with psycopg2 (not through the application):

* ``public.metrics`` (created by ``AI.db_store._ensure_schema_and_table``, after the deployment schema
  ``docker/initdb/02_schema.sql``): for every run ``--minutes`` one-minute points of ``--series`` series of
  the ``lt`` domain (query labels ``rps_total``, ``errors_total``, ``latency_p95``, ...) and ``--jvm-series``
  series of the ``jvm`` domain. Run names are ``bench-<svc>-<k>``, services ``svc01..svcNN``, run ids are
  uuids. Defaults (300 runs x 60 min x 36 series) give about 650 thousand rows; ``--runs 1000`` about 2.2
  million. Inserted with ``execute_values`` in pages of 5000 rows.
* ``public.llm_reports`` (created by ``AI.db_store._ensure_llm_reports_table``): one ``final`` row per run with
  the dashboard verdicts, ``test_type`` step/soak/spike/stress and a ``parsed`` load level in the field of
  its test type (some as text such as ``~1200``, which the SQL parser must skip), ``created_at`` spread over
  the last ``--days`` days, plus ``jvm``, ``kafka`` and ``hard_resources`` rows with text. 10 % of the runs
  were analysed twice (a second set of rows, later ``created_at``), as happens in production.
* the sandbox ``settings_runtime.json`` (``--runtime-json``, a file in %TEMP%, not a secret): area ``bench``
  with services ``svc01..svcNN`` in ``per_area``, so the area filter of the pages sees them. Inert
  ``bench-pad-*`` areas bring the file to ``--runtime-kb`` (227 KB, the size of the production file on
  2026-10-16); the sandbox ``metrics_config_runtime.json`` next to it gets the bench services and is padded
  the same way to ``--metrics-runtime-kb`` (54 KB). ``0`` turns the padding off.

Idempotent: ``--reset`` removes the rows with ``run_name LIKE 'bench-%'`` first (TRUNCATE when the table
holds nothing else); without it the script refuses to add a second copy. Ends with ``VACUUM ANALYZE``.
The same ``--seed`` and ``--now`` give the same rows.

Safety: connects only to 127.0.0.1/localhost/::1 unless ``--allow-remote`` is given, and never imports a
``settings.py`` (AI.db_store gets an empty stand-in, the table helpers do not read it).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]

# Same values as loadlens_app.blueprints.dashboard.DASHBOARD_VERDICTS (tests/test_perf_bench.py checks it).
DASHBOARD_VERDICTS = ("Успешно", "Есть риски", "Провал", "Недостаточно данных")
VERDICT_WEIGHTS = (0.55, 0.25, 0.12, 0.08)
TEST_TYPES = ("step", "soak", "spike", "stress")
# Where the analysis keeps the load level of each test type (the order of RUNS_MAX_RPS_FIELDS in dashboard.py).
RPS_FIELD_BY_TEST_TYPE = {
    "step": ("peak_performance", "max_rps"),
    "stress": ("peak_performance", "max_rps"),
    "soak": ("stability_under_load", "actual_rps"),
    "spike": ("spike_analysis", "max_spike_rps"),
}
LT_LABELS = ("rps_total", "errors_total", "latency_p95", "latency_p99", "active_users")
JVM_LABELS = (
    "jvm_heap_used", "jvm_heap_max", "jvm_gc_pause_ms", "jvm_threads_live",
    "jvm_cpu_usage", "jvm_classes_loaded", "jvm_nonheap_used", "jvm_gc_count",
)
OTHER_DOMAINS = ("jvm", "kafka", "hard_resources")
RUN_PREFIX = "bench-"
REANALYSIS_SHARE = 0.10
PAGE_SIZE = 5000
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SANDBOX_DB = {"host": "127.0.0.1", "port": 55432, "user": "ltar", "password": "e2e-only-password"}
# Mirrors storage.timescale of settings.example.py: what the app passes to the table helpers.
STORAGE_CFG = {
    "schema": "public",
    "table": "metrics",
    "llm_table": "llm_reports",
    "make_hypertable": True,
    "ensure_extension": True,
    "chunk_interval": "1 day",
}
METRICS_COLUMNS = (
    "time", "domain", "query_label", "run_id", "run_name", "service",
    "series", "value", "promql", "start_ms", "end_ms",
)
REPORT_COLUMNS = (
    "created_at", "run_id", "run_name", "service", "test_type", "start_ms", "end_ms", "domain",
    "text", "parsed", "scores", "context", "system_context", "verdict", "sla_verdict", "sla_details", "project_area",
)


# ---- pure builders (deterministic for a given seed and now) -----------------------------------------


@dataclass(frozen=True)
class Analysis:
    """One saved analysis of a run: the rows written by one save_llm_results call."""

    created_at: datetime
    verdict: str
    sla_verdict: str | None


@dataclass(frozen=True)
class RunSpec:
    index: int
    run_name: str
    run_id: str
    service: str
    start: datetime
    end: datetime
    test_type: str
    max_rps: float
    rps_style: str  # "number", "string" (plain number as text) or "tilde" ("~1200", not a number)
    analyses: tuple[Analysis, ...] = field(default_factory=tuple)

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end.timestamp() * 1000)


def service_ids(services: int) -> list[str]:
    return [f"svc{i:02d}" for i in range(1, services + 1)]


def _minute(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def plan_runs(runs: int, services: int, minutes: int, seed: int, now: datetime, days: int = 90) -> list[RunSpec]:
    """Run list: names, ids, services (round robin), time windows, test types, verdicts and load levels."""
    rng = random.Random(seed)
    svc = service_ids(services)
    now = _minute(now.astimezone(timezone.utc))
    out: list[RunSpec] = []
    for index in range(runs):
        service = svc[index % services]
        run_name = f"{RUN_PREFIX}{service}-{index // services + 1:04d}"
        run_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        created = now - timedelta(minutes=rng.randint(60, days * 24 * 60))
        end = _minute(created - timedelta(minutes=rng.randint(2, 30)))
        start = end - timedelta(minutes=minutes)
        test_type = rng.choice(TEST_TYPES)
        max_rps = round(rng.uniform(50.0, 2500.0), 1)
        style_roll = rng.random()
        rps_style = "number" if style_roll < 0.70 else ("string" if style_roll < 0.85 else "tilde")
        analyses = [_analysis(rng, created)]
        if rng.random() < REANALYSIS_SHARE:
            later = min(now - timedelta(minutes=1), created + timedelta(minutes=rng.randint(30, 48 * 60)))
            analyses.append(_analysis(rng, later))
        out.append(RunSpec(index, run_name, run_id, service, start, end, test_type, max_rps, rps_style, tuple(analyses)))
    return out


def _analysis(rng: random.Random, created_at: datetime) -> Analysis:
    verdict = rng.choices(DASHBOARD_VERDICTS, weights=VERDICT_WEIGHTS)[0]
    sla_verdict = None if rng.random() < 0.3 else rng.choices(DASHBOARD_VERDICTS, weights=VERDICT_WEIGHTS)[0]
    return Analysis(created_at, verdict, sla_verdict)


def _promql(domain: str, label: str, service: str) -> str:
    return f'sum by (series) (rate(bench_{domain}_{label}{{service="{service}",namespace="bench"}}[1m]))'


def _load_profile(test_type: str, minute: int, minutes: int) -> float:
    """Share of the peak load at a minute of the run."""
    position = minute / max(1, minutes - 1)
    if test_type == "step":
        return 0.2 + 0.8 * math.floor(position * 5) / 4 if position < 1 else 1.0
    if test_type == "stress":
        return 0.1 + 0.9 * position
    if test_type == "spike":
        return 1.0 if 0.45 <= position <= 0.55 else 0.35
    return 0.9  # soak


def metric_series(series: int, jvm_series: int) -> list[tuple[str, str, str]]:
    """``(domain, query_label, series)`` of every stored series of a run."""
    out = []
    for j in range(series):
        out.append(("lt", LT_LABELS[j % len(LT_LABELS)], f"tx{j // len(LT_LABELS) + 1:02d}"))
    for j in range(jvm_series):
        out.append(("jvm", JVM_LABELS[j % len(JVM_LABELS)], f"pod-{j // len(JVM_LABELS) + 1}"))
    return out


def _metric_value(label: str, load: float, peak: float, rng: random.Random) -> float:
    noise = 1.0 + rng.uniform(-0.05, 0.05)
    if label == "rps_total":
        return peak * load * noise / 4.0
    if label == "errors_total":
        return max(0.0, peak * load * 0.002 * noise - 0.1)
    if label == "latency_p95":
        return (120.0 + 380.0 * load ** 3) * noise
    if label == "latency_p99":
        return (200.0 + 700.0 * load ** 3) * noise
    if label == "active_users":
        return peak * load * 0.4 * noise
    if label.startswith("jvm_cpu"):
        return min(0.98, 0.1 + 0.7 * load * noise)
    if label.startswith("jvm_heap") or label == "jvm_nonheap_used":
        return (0.3 + 0.5 * load) * 2.0 * 1024 ** 3 * noise
    return (10.0 + 90.0 * load) * noise


def metric_rows(run: RunSpec, series: int, jvm_series: int, seed: int) -> Iterator[tuple]:
    """Rows of ``public.metrics`` for one run, in the column order of METRICS_COLUMNS."""
    rng = random.Random(f"{seed}:{run.run_name}")
    layout = [(domain, label, name, _promql(domain, label, run.service)) for domain, label, name in metric_series(series, jvm_series)]
    minutes = int((run.end - run.start).total_seconds() // 60)
    start_ms, end_ms = run.start_ms, run.end_ms
    for minute in range(minutes):
        moment = run.start + timedelta(minutes=minute)
        load = _load_profile(run.test_type, minute, minutes)
        for domain, label, name, promql in layout:
            value = round(_metric_value(label, load, run.max_rps, rng), 4)
            yield (moment, domain, label, run.run_id, run.run_name, run.service, name, value, promql, start_ms, end_ms)


def _rps_value(run: RunSpec) -> Any:
    if run.rps_style == "string":
        return f"{run.max_rps}"
    if run.rps_style == "tilde":
        return f"~{int(round(run.max_rps, -1))}"
    return run.max_rps


_SENTENCES = (
    "Нагрузка {rps} RPS удерживалась {mins} минут без роста ошибок.",
    "p95 времени ответа вырос с {lo} до {hi} мс на последней ступени.",
    "Пул соединений к базе заполнялся до {pct} % на пике.",
    "GC паузы не превышали {gc} мс, базовая линия heap стабильна.",
    "Лаг консьюмеров Kafka оставался ниже {lag} сообщений.",
    "CPU узлов достигал {pct} %, память до {mem} %.",
)


def _paragraph(rng: random.Random, sentences: int) -> str:
    parts = []
    for _ in range(sentences):
        template = rng.choice(_SENTENCES)
        parts.append(template.format(
            rps=rng.randint(50, 2500), mins=rng.randint(5, 60), lo=rng.randint(80, 200), hi=rng.randint(200, 900),
            pct=rng.randint(30, 99), gc=rng.randint(5, 300), lag=rng.randint(10, 5000), mem=rng.randint(30, 90),
        ))
    return " ".join(parts)


def _findings(rng: random.Random, count: int, component: str) -> list[dict]:
    return [
        {
            "id": f"f{i + 1}",
            "summary": _paragraph(rng, 1),
            "severity": rng.choice(("low", "medium", "high")),
            "component": component,
            "evidence_summary": _paragraph(rng, 2),
            "evidence_items": [
                {"metric": rng.choice(LT_LABELS), "observed_value": f"{rng.randint(100, 900)} мс", "threshold": "300 мс", "note": "bench"},
            ],
        }
        for i in range(count)
    ]


def _actions(rng: random.Random, count: int, component: str) -> list[dict]:
    return [
        {
            "summary": _paragraph(rng, 1),
            "details": _paragraph(rng, 3),
            "priority": rng.choice(("low", "medium", "high")),
            "affected_components": [component],
            "for_finding_ids": [f"f{i + 1}"],
        }
        for i in range(count)
    ]


def _scores(rng: random.Random) -> dict:
    overall = round(rng.uniform(0.5, 0.95), 2)
    return {"selected_index": 0, "judge": {"overall": overall, "factual": overall, "completeness": overall}, "final_score": overall}


def _sla_checks(rng: random.Random, run: RunSpec) -> list[dict]:
    return [
        {"name": "target_rps", "threshold": 200, "actual": run.max_rps, "passed": run.max_rps >= 200, "severity": "critical",
         "message": f"RPS {run.max_rps} (stable_max) против цели 200", "query_label": "rps_total", "series": "tx01"},
        {"name": "p95_latency", "threshold": 300, "actual": float(rng.randint(100, 900)), "passed": rng.random() < 0.6,
         "severity": "warning", "message": "P95 latency против порога 300 ms", "query_label": "latency_p95", "series": "tx01"},
        {"name": "error_rate", "threshold": 1.0, "actual": round(rng.uniform(0, 2), 2), "passed": rng.random() < 0.8,
         "severity": "warning", "message": "Error rate против порога 1.0%", "query_label": "errors_total", "series": "tx01"},
    ]


def _system_context(run: RunSpec) -> dict:
    return {
        "system": {"name": f"Bench system {run.service}", "domain": "bench", "description": "Синтетические данные бенча LoadLens.",
                   "test_goal": "Найти максимальную стабильную нагрузку"},
        "components": [{"name": f"{run.service}-api", "kind": "service", "notes": "Java, Spring Boot, 4 реплики"},
                       {"name": f"{run.service}-db", "kind": "database", "notes": "PostgreSQL, пул 50 соединений"}],
        "slo": [{"name": "p95", "value": "300 ms"}, {"name": "errors", "value": "1 %"}],
    }


def report_rows(run: RunSpec, seed: int, area: str = "bench") -> list[tuple]:
    """Rows of ``public.llm_reports`` for one run (every analysis: three domains, then final)."""
    from psycopg2.extras import Json

    rng = random.Random(f"{seed}:{run.run_name}:reports")
    system_context = _system_context(run)
    rows: list[tuple] = []
    for analysis in run.analyses:
        for domain in OTHER_DOMAINS:
            verdict = rng.choices(DASHBOARD_VERDICTS, weights=VERDICT_WEIGHTS)[0]
            parsed = {
                "verdict": verdict, "verdict_rationale": _paragraph(rng, 3), "confidence": round(rng.uniform(0.5, 0.95), 2),
                "findings": _findings(rng, 2, f"{run.service}-api"), "recommended_actions": _actions(rng, 1, f"{run.service}-api"),
            }
            context = {"domain": domain, "summary": _paragraph(rng, 20), "series": [f"{label}:tx01" for label in LT_LABELS]}
            rows.append((
                analysis.created_at, run.run_id, run.run_name, run.service, run.test_type, run.start_ms, run.end_ms, domain,
                _paragraph(rng, 40), Json(parsed), Json(_scores(rng)), Json(context), Json(system_context),
                verdict, None, None, area,
            ))
        section, rps_field = RPS_FIELD_BY_TEST_TYPE[run.test_type]
        level = {rps_field: _rps_value(run), "method": "last_stable_step"}
        if section == "peak_performance":
            level.update({"max_time": run.end.isoformat(), "drop_time": run.end.isoformat()})
        parsed = {
            "verdict": analysis.verdict, "verdict_rationale": _paragraph(rng, 4), "confidence": round(rng.uniform(0.5, 0.95), 2),
            "findings": _findings(rng, 3, f"{run.service}-api"), "recommended_actions": _actions(rng, 2, f"{run.service}-api"),
            section: level,
        }
        checks = _sla_checks(rng, run)
        sla_details = {"checks": checks, "summary": f"SLA verdict: {analysis.sla_verdict}"} if analysis.sla_verdict else None
        rows.append((
            analysis.created_at, run.run_id, run.run_name, run.service, run.test_type, run.start_ms, run.end_ms, "final",
            json.dumps(parsed, ensure_ascii=False, indent=2), Json(parsed), Json(_scores(rng)), None, Json(system_context),
            analysis.verdict, analysis.sla_verdict, Json(sla_details) if sla_details else None, area,
        ))
    return rows


# ---- runtime JSON of the sandbox --------------------------------------------------------------------


def _pad_queries(rng: random.Random, service: str) -> dict:
    out = {}
    for domain in ("jvm", "database", "kafka", "microservices", "hard_resources"):
        labels = [f"{domain.upper()}: series {i} by (application, instance)" for i in range(8)]
        out[domain] = {
            "labels": labels,
            "promql_queries": [f'sum by (application, instance) (rate({domain}_metric_{i}_total{{application="{service}",namespace="bench-{rng.randint(1, 9)}"}}[1m]))' for i in range(8)],
            "label_keys_list": [["application", "instance"] for _ in labels],
        }
    return out


def _pad_metrics(rng: random.Random, service: str) -> dict:
    return {
        "page_sample_id": str(rng.randint(10_000_000, 99_999_999)),
        "page_parent_id": str(rng.randint(10_000_000, 99_999_999)),
        "metrics": [
            {"name": f"panel_{i}", "grafana_url": f"/render/d-solo/BENCHPAD/bench?orgId=1&var-service={service}&panelId={i}&width=1000&height=500&tz=Europe%2FMoscow"}
            for i in range(6)
        ],
        "logs": [{"name": "errors", "loki_query": f'{{app="{service}"}} |= "ERROR"'}],
    }


def _pad_until(target: dict, area: str, kb: int, build) -> None:
    """Adds inert ``<area>-pad-NN`` areas (8 services each) until ``target`` serialises to ``kb`` kilobytes."""
    rng = random.Random(f"pad:{area}")
    n = 0
    while kb > 0 and len(json.dumps(target, ensure_ascii=False, indent=2).encode("utf-8")) < kb * 1024:
        pad_area = f"{area}-pad-{n // 8 + 1:02d}"
        service = f"{area}-pad-svc{n + 1:03d}"
        build(target, pad_area, service, rng)
        n += 1


def bench_settings_runtime(existing: dict, area: str, services: int, pad_kb: int) -> dict:
    """settings_runtime.json with the bench area (and its padding) replaced; other areas are kept."""
    data = json.loads(json.dumps(existing or {}))
    per_area = data.get("per_area") if isinstance(data.get("per_area"), dict) else {}
    per_area = {k: v for k, v in per_area.items() if k != area and not k.startswith(f"{area}-pad-")}
    per_area[area] = {
        "title": "Bench",
        "services": {sid: {"title": f"Bench service {sid[3:]}"} for sid in service_ids(services)},
    }
    data["per_area"] = per_area

    def build(target, pad_area, service, rng):
        entry = target["per_area"].setdefault(pad_area, {"title": f"Bench padding {pad_area[-2:]}", "services": {}})
        entry["services"][service] = {
            "title": service,
            "disabled_domains": ["kafka"],
            "queries": _pad_queries(rng, service),
            "sla": {"p95_ms": rng.randint(200, 900), "error_rate_pct": 1.0, "target_rps": rng.randint(100, 3000)},
        }

    _pad_until(data, area, pad_kb, build)
    return data


def bench_metrics_runtime(existing: dict, area: str, services: int, pad_kb: int) -> dict:
    """metrics_config_runtime.json with the bench services (empty metrics) and inert padding areas."""
    data = {k: v for k, v in json.loads(json.dumps(existing or {})).items() if k != area and not k.startswith(f"{area}-pad-")}
    data[area] = {"services": {sid: {"page_sample_id": "", "page_parent_id": "", "metrics": [], "logs": []} for sid in service_ids(services)}}

    def build(target, pad_area, service, rng):
        target.setdefault(pad_area, {"services": {}})["services"][service] = _pad_metrics(rng, service)

    _pad_until(data, area, pad_kb, build)
    return data


def _rewrite_json(path: Path, update) -> int:
    existing = json.loads(path.read_text(encoding="utf-8-sig") or "{}") if path.exists() else {}
    payload = update(existing)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return len(text.encode("utf-8"))


# ---- database ---------------------------------------------------------------------------------------


def _db_store():
    """AI.db_store without a settings.py: the table helpers only use the storage config passed to them."""
    if "settings" not in sys.modules:
        stand_in = types.ModuleType("settings")
        stand_in.CONFIG = {}
        sys.modules["settings"] = stand_in
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from AI import db_store

    return db_store


def connection_params(args: argparse.Namespace) -> dict:
    if args.dsn:
        from psycopg2.extensions import parse_dsn

        params = parse_dsn(args.dsn)
    else:
        params = {"host": args.host, "port": args.port, "dbname": args.dbname, "user": args.user, "password": args.password}
    params = {k: v for k, v in params.items() if v not in (None, "")}
    host = str(params.get("host") or "localhost")
    if host not in LOCAL_HOSTS and not args.allow_remote:
        raise SystemExit(f"refusing to write to {host}: only a local sandbox database is allowed (pass --allow-remote to override)")
    params.setdefault("connect_timeout", 5)
    return params


def _table_has_foreign_rows(cur, table: str) -> bool:
    cur.execute(f"SELECT EXISTS (SELECT 1 FROM public.{table} WHERE run_name IS NULL OR run_name NOT LIKE %s)", (RUN_PREFIX + "%",))
    return bool(cur.fetchone()[0])


def _bench_rows_exist(cur) -> bool:
    cur.execute("SELECT EXISTS (SELECT 1 FROM public.llm_reports WHERE run_name LIKE %s)", (RUN_PREFIX + "%",))
    found = bool(cur.fetchone()[0])
    cur.execute("SELECT EXISTS (SELECT 1 FROM public.metrics WHERE run_name LIKE %s)", (RUN_PREFIX + "%",))
    return found or bool(cur.fetchone()[0])


def prepare_tables(conn, schema_sql: Path | None) -> None:
    db_store = _db_store()
    if schema_sql is not None:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(schema_sql.read_text(encoding="utf-8"))
        conn.autocommit = False
    db_store._ensure_schema_and_table(conn, STORAGE_CFG, "public", "metrics")
    db_store._ensure_llm_reports_table(conn, STORAGE_CFG)


def reset_bench_rows(conn) -> None:
    with conn.cursor() as cur:
        for table in ("metrics", "llm_reports"):
            if _table_has_foreign_rows(cur, table):
                cur.execute(f"DELETE FROM public.{table} WHERE run_name LIKE %s", (RUN_PREFIX + "%",))
            else:
                cur.execute(f"TRUNCATE public.{table}")
    conn.commit()


def _insert(cur, table: str, columns: Sequence[str], rows: Iterable[tuple], page_size: int) -> int:
    """Multi-row INSERTs of ``page_size`` rows (execute_values: about 10x faster than execute_batch here)."""
    from psycopg2.extras import execute_values

    statement = f"INSERT INTO public.{table} ({', '.join(columns)}) VALUES %s"
    buffer: list[tuple] = []
    total = 0
    for row in rows:
        buffer.append(row)
        if len(buffer) >= page_size:
            execute_values(cur, statement, buffer, page_size=page_size)
            total += len(buffer)
            buffer = []
    if buffer:
        execute_values(cur, statement, buffer, page_size=page_size)
        total += len(buffer)
    return total


def seed_database(conn, plan: list[RunSpec], args: argparse.Namespace) -> tuple[int, int]:
    """Inserts the plan in ten slices, one transaction each, with a progress line per slice."""
    metrics_total = reports_total = 0
    started = time.perf_counter()
    step = max(1, len(plan) // 10)
    with conn.cursor() as cur:
        for first in range(0, len(plan), step):
            chunk = plan[first:first + step]
            metrics = (row for run in chunk for row in metric_rows(run, args.series, args.jvm_series, args.seed))
            reports = (row for run in chunk for row in report_rows(run, args.seed, args.area))
            metrics_total += _insert(cur, "metrics", METRICS_COLUMNS, metrics, PAGE_SIZE)
            reports_total += _insert(cur, "llm_reports", REPORT_COLUMNS, reports, 200)
            conn.commit()
            done = first + len(chunk)
            print(f"  {done}/{len(plan)} runs, {metrics_total} metric rows, {time.perf_counter() - started:.0f} s", file=sys.stderr)
    return metrics_total, reports_total


def vacuum_analyze(conn) -> None:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("VACUUM ANALYZE public.metrics")
        cur.execute("VACUUM ANALYZE public.llm_reports")
    conn.autocommit = False


def table_counts(conn) -> dict:
    with conn.cursor() as cur:
        out = {}
        for table in ("metrics", "llm_reports"):
            cur.execute(f"SELECT count(*), count(DISTINCT run_name) FROM public.{table}")
            rows, runs = cur.fetchone()
            out[table] = {"rows": int(rows), "runs": int(runs)}
        cur.execute("SELECT pg_size_pretty(hypertable_size('public.metrics')), pg_size_pretty(pg_total_relation_size('public.llm_reports'))")
        out["metrics"]["size"], out["llm_reports"]["size"] = cur.fetchone()
    conn.commit()
    return out


# ---- command line -----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seeds a sandbox TimescaleDB with synthetic bench runs (see the module docstring).")
    target = parser.add_argument_group("target")
    target.add_argument("--sandbox", type=int, help="sandbox number N: database ll_s<N> on 127.0.0.1:55432 and %%TEMP%%/ll_ui_s<N>/app runtime files")
    target.add_argument("--dsn", help="libpq connection string (overrides --host/--port/--dbname/--user/--password)")
    target.add_argument("--host", default=SANDBOX_DB["host"])
    target.add_argument("--port", type=int, default=SANDBOX_DB["port"])
    target.add_argument("--dbname")
    target.add_argument("--user", default=SANDBOX_DB["user"])
    target.add_argument("--password", default=SANDBOX_DB["password"])
    target.add_argument("--allow-remote", action="store_true", help="allow a database host other than 127.0.0.1/localhost")
    target.add_argument("--runtime-json", type=Path, help="settings_runtime.json of the sandbox (gets the bench area)")
    target.add_argument("--metrics-runtime-json", type=Path, help="metrics_config_runtime.json of the sandbox (default: next to --runtime-json)")
    target.add_argument("--schema-sql", type=Path, default=ROOT_DIR / "docker" / "initdb" / "02_schema.sql",
                        help="deployment schema applied first (default docker/initdb/02_schema.sql)")
    target.add_argument("--no-schema-sql", action="store_true", help="only the tables the application creates itself")
    data = parser.add_argument_group("data")
    data.add_argument("--runs", type=int, default=300)
    data.add_argument("--services", type=int, default=10)
    data.add_argument("--minutes", type=int, default=60, help="one-minute points per run")
    data.add_argument("--series", type=int, default=20, help="series of the lt domain per run")
    data.add_argument("--jvm-series", type=int, default=16, help="series of the jvm domain per run")
    data.add_argument("--days", type=int, default=90, help="reports are spread over the last N days")
    data.add_argument("--seed", type=int, default=1)
    data.add_argument("--now", help="ISO time the data is anchored to (default: now, UTC)")
    data.add_argument("--area", default="bench", help="project area of the bench services")
    data.add_argument("--runtime-kb", type=int, default=227, help="pad settings_runtime.json to this size (0: no padding)")
    data.add_argument("--metrics-runtime-kb", type=int, default=54, help="pad metrics_config_runtime.json to this size (0: no padding)")
    data.add_argument("--reset", action="store_true", help="remove earlier bench rows first")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.services < 1 or args.minutes < 1 or args.series < 0 or args.jvm_series < 0:
        parser.error("--runs, --services and --minutes must be positive, --series and --jvm-series not negative")
    if args.sandbox is not None:
        args.dbname = args.dbname or f"ll_s{args.sandbox}"
        sandbox_app = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / f"ll_ui_s{args.sandbox}" / "app"
        if args.runtime_json is None:
            if not sandbox_app.is_dir():
                parser.error(f"{sandbox_app} not found: run make_sandbox.py {args.sandbox} first or pass --runtime-json")
            args.runtime_json = sandbox_app / "settings_runtime.json"
    if not args.dsn and not args.dbname:
        parser.error("pass --sandbox N, --dsn or --dbname")
    if args.runtime_json is not None and args.metrics_runtime_json is None:
        args.metrics_runtime_json = args.runtime_json.with_name("metrics_config_runtime.json")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import psycopg2

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    params = connection_params(args)
    print(f"seed: {args.runs} runs x {args.minutes} min x {args.series + args.jvm_series} series -> "
          f"{params.get('host')}:{params.get('port')}/{params.get('dbname')}", file=sys.stderr)
    started = time.perf_counter()
    conn = psycopg2.connect(**params)
    try:
        prepare_tables(conn, None if args.no_schema_sql else args.schema_sql)
        with conn.cursor() as cur:
            exists = _bench_rows_exist(cur)
        conn.commit()
        if exists and not args.reset:
            print("bench rows already exist: pass --reset to replace them", file=sys.stderr)
            return 2
        if args.reset:
            reset_bench_rows(conn)
        plan = plan_runs(args.runs, args.services, args.minutes, args.seed, now, args.days)
        metrics_rows, report_rows_count = seed_database(conn, plan, args)
        vacuum_analyze(conn)
        counts = table_counts(conn)
    finally:
        conn.close()
    if args.runtime_json is not None:
        size = _rewrite_json(args.runtime_json, lambda data: bench_settings_runtime(data, args.area, args.services, args.runtime_kb))
        print(f"runtime: area '{args.area}' with {args.services} services -> {args.runtime_json} ({size // 1024} KB)", file=sys.stderr)
        if args.metrics_runtime_json is not None:
            size = _rewrite_json(args.metrics_runtime_json, lambda data: bench_metrics_runtime(data, args.area, args.services, args.metrics_runtime_kb))
            print(f"runtime: metrics services -> {args.metrics_runtime_json} ({size // 1024} KB)", file=sys.stderr)
    print(json.dumps({
        "inserted": {"metrics": metrics_rows, "llm_reports": report_rows_count},
        "tables": counts,
        "seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
