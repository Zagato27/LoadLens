"""Latency bench of the LoadLens endpoints that the pages poll (see docs/perf-baseline.md).

One command against sandbox N (``run_app.py`` of make_sandbox.py listens on 5070+N, data from
tools/seed_bench_data.py):

    C:/Python312/python.exe tools/bench_endpoints.py --sandbox 1 --json before.json
    C:/Python312/python.exe tools/bench_endpoints.py --sandbox 1 --concurrency 4 --json before-c4.json
    C:/Python312/python.exe tools/bench_endpoints.py --sandbox 1 --compare before.json --json after.json

``--base http://127.0.0.1:5071`` is the same as ``--sandbox 1``. The cookie ``project_area`` is set to
``--area`` (``bench``), so the pages see the seeded services.

For every endpoint: ``--warmup`` sequential requests that are not counted, then ``--concurrency`` clients
send ``--rounds`` requests each at the same time (a new connection per request, the body is read in
full). The table shows p50/p95/max of the client time, the throughput, p50 of the ``Server-Timing``
``app;dur`` value (time inside the application), the response size and the number of failed requests
(status >= 400 or no answer). ``--compare old.json`` adds the old p50/p95 and the ratio old/new
(above 1 means faster now). ``--json`` saves everything for a later ``--compare``.

Only GET requests; only 127.0.0.1/localhost unless ``--allow-remote`` (then ``--token`` or the
LOADLENS_BENCH_TOKEN variable sends an API token when sign-in is on). Standard library only.

``--explain`` does no timing: it imports the app of the sandbox copy (``%TEMP%/ll_ui_s<N>/app`` or
``--app-dir``; a git working tree is refused, so a real settings.py is never loaded), calls each endpoint
with the Flask test client, records every SQL statement and runs ``EXPLAIN (ANALYZE, BUFFERS)`` on each
SELECT in the sandbox database. It prints planning/execution time and buffers per statement (``--plans``:
the full plans) and the cost of opening a connection:

    C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox 1 --explain --only runs --plans

In Git Bash a path argument such as ``--endpoint /runs`` is rewritten to a Windows path; prefix the
command with ``MSYS_NO_PATHCONV=1`` or use ``--only``.

``--during-job`` measures the endpoints while a report is being built (stage 3 of docs/perf-plan.md, section
11.1; the full sandbox recipe with the stubs of tools/stubs is in docs/perf-baseline.md, "Этап 3"):

    C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox 1 --setup-stubs       # once per sandbox reset
    C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox 1 --during-job --label thread --json thread.json
    C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox 1 --during-job --label celery --compare thread.json

First an idle round: ``--concurrency`` clients (each keeps its connection where the server allows it, as a
browser tab; the Werkzeug dev server closes every one) go ``--rounds`` times round DURING_JOB_ENDPOINTS
with ``--pause-ms`` after each request. Then
``POST /create_report`` (``web_only``, ``use_llm``, service ``--service`` of the area, a ``--window-min``
window ending now, run name ``bench-job-<time>``) and the same clients keep going until ``/job_status`` (polled
every ``--poll-sec``, a row of its own) says done or error. Printed: the idle / during-job table (p50/p95,
``job/idle`` above 1 = slower while the report runs, ``Server-Timing`` p50, number of samples), the job
duration (client and server side), its phases (collection, domain analyses, final analysis and saving, from
the progress marks of AI/pipeline.py), the calls the LLM and Prometheus stubs counted, whether the report
opens (``/reports/<run_id>`` and ``/llm_reports``), then p50/p95 per phase. The run is deleted afterwards
unless ``--keep-report``. ``--compare`` takes an earlier ``--json`` of this mode or a markdown file with such
tables (``--compare-table K``), e.g. thread mode against celery mode. Exit code 1 when the job did not reach
``done`` or its report does not open. Which runner built the report (thread or Celery) is a setting of the
server, not of the bench: pass it as ``--label``.

``--setup-stubs`` writes the stub keys into the sandbox ``settings_runtime.json`` (after
``tools/seed_bench_data.py --runtime-only``, which recreates the bench area): LLM, Confluence, the stub
Prometheus as the data source of the area, PromQL queries for lt_framework, no daily LLM budget. A working
tree (a directory with ``.git``) is refused.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import re
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import quote, urlencode, urlsplit

ENDPOINTS = (
    "/",
    "/dashboard_data",
    "/runs?limit=20",
    "/runs?limit=20&q=svc0",
    "/runs?limit=20&sort=max_rps",
    "/load_tests/active?limit=5&live=1",
    "/load_tests/active?limit=0",
    "/jobs?status=running&limit=10",
    "/services",
    "/project_areas",
    "/static/js/settings.js",
    "/static/css/app.css",
)
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_SERVER_TIMING_APP = re.compile(r"(?:^|,)\s*app\s*;(?:[^,]*;)?\s*dur=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


# ---- statistics and formatting (pure) ----------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (numpy's default): ``q`` in [0, 1]; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, q))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: Sequence[float]) -> dict:
    """``count``, ``p50``, ``p95``, ``max`` and ``mean`` of the samples (milliseconds)."""
    if not samples:
        return {"count": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "max": max(samples),
        "mean": sum(samples) / len(samples),
    }


def parse_server_timing(header: Optional[str]) -> Optional[float]:
    """``dur`` of the ``app`` metric of a Server-Timing header, in milliseconds."""
    match = _SERVER_TIMING_APP.search(header or "")
    return float(match.group(1)) if match else None


def _ms(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}"


def _ratio(before: Optional[float], after: Optional[float]) -> str:
    if not before or not after:
        return "—"
    return f"{before / after:.2f}×"


def _size(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value / 1024:.1f} КБ" if value >= 1024 else f"{int(value)} Б"


def render_table(results: Sequence[dict], before: Optional[Sequence[dict]] = None) -> str:
    """Markdown table of the results; with ``before`` the old p50/p95 and the ratios old/new."""
    head = ["Эндпоинт", "p50, мс", "p95, мс", "max, мс", "req/s", "app p50, мс", "Размер", "Ошибки"]
    old = {item["path"]: item for item in (before or [])}
    if before is not None:
        head += ["было p50", "было p95", "×p50", "×p95"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for item in results:
        row = [
            f"`{item['path']}`", _ms(item["p50"]), _ms(item["p95"]), _ms(item["max"]),
            "—" if item.get("rps") is None else f"{item['rps']:.1f}",
            _ms(item.get("app_p50")), _size(item.get("bytes")), str(item.get("errors", 0)),
        ]
        if before is not None:
            prev = old.get(item["path"]) or {}
            row += [_ms(prev.get("p50")), _ms(prev.get("p95")), _ratio(prev.get("p50"), item["p50"]), _ratio(prev.get("p95"), item["p95"])]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---- HTTP ----------------------------------------------------------------------------------------------


class Target:
    def __init__(self, base: str, area: str, token: str = "", timeout: float = 60.0):
        parts = urlsplit(base)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"--base must be an http(s) address, got {base!r}")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.prefix = parts.path.rstrip("/")
        self.timeout = timeout
        self.area = area
        self.headers = {"Accept": "*/*", "User-Agent": "loadlens-bench/1"}
        if area:
            self.headers["Cookie"] = f"project_area={area}"
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def get(self, path: str) -> tuple[int, float, Optional[float], int]:
        """``(status, client ms, app ms, body bytes)``; status 0 when there is no answer."""
        connection_class = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        started = time.perf_counter()
        connection = connection_class(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("GET", self.prefix + path, headers=self.headers)
            response = connection.getresponse()
            body = response.read()
            elapsed = (time.perf_counter() - started) * 1000.0
            return response.status, elapsed, parse_server_timing(response.getheader("Server-Timing")), len(body)
        except (OSError, http.client.HTTPException):
            return 0, (time.perf_counter() - started) * 1000.0, None, 0
        finally:
            connection.close()


def measure(target: Target, path: str, rounds: int, concurrency: int, warmup: int) -> dict:
    """Warm-up requests, then ``concurrency`` clients with ``rounds`` requests each, started together."""
    for _ in range(warmup):
        target.get(path)
    start_line = threading.Barrier(concurrency)

    def client(_: int) -> list[tuple[int, float, Optional[float], int]]:
        start_line.wait()
        return [target.get(path) for _ in range(rounds)]

    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = [sample for batch in pool.map(client, range(concurrency)) for sample in batch]
    wall = time.perf_counter() - began

    ok = [s for s in samples if 0 < s[0] < 400]
    stats = summarize([s[1] for s in samples])
    app = summarize([s[2] for s in samples if s[2] is not None])
    sizes = sorted(s[3] for s in ok)
    return {
        "path": path,
        **stats,
        "rps": len(samples) / wall if wall > 0 else None,
        "app_p50": app["p50"],
        "app_p95": app["p95"],
        "bytes": sizes[len(sizes) // 2] if sizes else None,
        "errors": len(samples) - len(ok),
        "statuses": dict(Counter(str(s[0]) for s in samples)),
    }


# ---- --explain: EXPLAIN (ANALYZE, BUFFERS) of the SQL behind each endpoint ------------------------------

_PLAN_TIME = re.compile(r"^(Planning|Execution) Time: ([0-9.]+) ms$")
_PLAN_BUFFERS = re.compile(r"Buffers: shared(?: hit=(\d+))?(?: read=(\d+))?")


def summarize_plan(lines: Sequence[str]) -> dict:
    """Top node, planning/execution time and shared buffers (hit + read) of the top node of a plan."""
    out: dict[str, Any] = {"top": "", "planning_ms": None, "execution_ms": None, "buffers": None}
    if lines:
        out["top"] = lines[0].split("  (", 1)[0].strip()
    for line in lines:
        timing = _PLAN_TIME.match(line.strip())
        if timing:
            out["planning_ms" if timing.group(1) == "Planning" else "execution_ms"] = float(timing.group(2))
        elif out["buffers"] is None:
            buffers = _PLAN_BUFFERS.search(line)
            if buffers:
                out["buffers"] = int(buffers.group(1) or 0) + int(buffers.group(2) or 0)
    return out


def statement_kind(sql_text: str) -> str:
    words = sql_text.split(None, 1)
    return words[0].upper() if words else ""


class _RecordingCursor:
    """Cursor proxy that keeps the text of every statement, with the parameters inlined by psycopg2."""

    def __init__(self, cursor, log: list):
        self._cursor = cursor
        self._log = log

    def execute(self, query, params=None):
        self._log.append(self._cursor.mogrify(query, params).decode("utf-8", "replace"))
        return self._cursor.execute(query, params)

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __iter__(self):
        return iter(self._cursor)

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cursor.__exit__(*exc)


class _RecordingConnection:
    """Connection proxy: ``with conn`` still means commit/rollback, cursors record their statements."""

    def __init__(self, connection, log: list):
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_log", log)

    def cursor(self, *args, **kwargs):
        return _RecordingCursor(self._connection.cursor(*args, **kwargs), self._log)

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def __setattr__(self, name, value):
        setattr(self._connection, name, value)

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, *exc):
        return self._connection.__exit__(*exc)


def _load_sandbox_app(app_dir: Path, allow_remote: bool):
    """The Flask app of a sandbox copy (sign-in off); refuses working trees and non-local databases."""
    if (app_dir / ".git").exists():
        raise SystemExit(f"{app_dir} is a git working tree, not a sandbox copy: --explain imports its settings.py")
    if not (app_dir / "app.py").is_file() or not (app_dir / "settings.py").is_file():
        raise SystemExit(f"{app_dir}: no app.py/settings.py; run make_sandbox.py first or pass --app-dir")
    os.environ["LOADLENS_AUTH_ENABLED"] = "0"
    os.environ.setdefault("LOADLENS_SECRET_KEY", "bench-explain-only-secret-key")
    os.environ["LOADLENS_LOAD_TEST_WATCHER"] = "0"
    os.environ.pop("LOADLENS_RUN_MIGRATIONS", None)
    os.chdir(app_dir)
    sys.path.insert(0, str(app_dir))
    import settings  # the sandbox copy

    storage = ((getattr(settings, "CONFIG", {}) or {}).get("storage") or {}).get("timescale") or {}
    if str(storage.get("host") or "") not in LOCAL_HOSTS and not allow_remote:
        raise SystemExit("refusing: storage.timescale.host of the sandbox is not local (pass --allow-remote)")
    from app import app as flask_app
    from loadlens_app import core

    return flask_app, core


def explain_endpoints(args: argparse.Namespace) -> int:
    """Calls every endpoint in-process and runs EXPLAIN (ANALYZE, BUFFERS) on each SELECT it executed."""
    if args.app_dir:
        app_dir = Path(args.app_dir)
    elif args.sandbox is not None:
        app_dir = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / f"ll_ui_s{args.sandbox}" / "app"
    else:
        raise SystemExit("--explain needs --sandbox N or --app-dir")
    json_path = os.path.abspath(args.json_path) if args.json_path else None  # the app import changes the cwd
    flask_app, core = _load_sandbox_app(app_dir.resolve(), args.allow_remote)
    original = core._ts_conn
    log: list[str] = []

    def recording():
        return _RecordingConnection(original(), log)

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and getattr(module, "_ts_conn", None) is original:
            module._ts_conn = recording

    connect_ms = []
    for _ in range(5):
        began = time.perf_counter()
        original().close()
        connect_ms.append((time.perf_counter() - began) * 1000.0)
    explain_conn = original()
    round_trip = []
    with explain_conn.cursor() as cur:
        for _ in range(30):
            began = time.perf_counter()
            cur.execute("SELECT 1")
            cur.fetchone()
            round_trip.append((time.perf_counter() - began) * 1000.0)
    explain_conn.rollback()
    print(f"database: connect p50 {_ms(percentile(connect_ms, 0.5))} ms, round trip of SELECT 1 p50 {_ms(percentile(round_trip, 0.5))} ms\n")

    client = flask_app.test_client()
    if args.area:
        client.set_cookie("project_area", args.area)
    report = []
    for path in selected_endpoints(args):
        client.get(path)  # first call creates tables and fills process caches
        log.clear()
        status = client.get(path).status_code
        statements = [s.strip() for s in log if statement_kind(s.strip()) in ("SELECT", "WITH")]
        print(f"### `{path}` — {status}, {len(statements)} SELECT\n")
        if not statements:
            continue
        print("| # | Запрос | Верхний узел | Planning, мс | Execution, мс | Buffers |")
        print("|---|---|---|---|---|---|")
        plans = []
        for number, statement in enumerate(statements, start=1):
            with explain_conn.cursor() as cur:
                cur.execute("EXPLAIN (ANALYZE, BUFFERS) " + statement)
                plan = [row[0] for row in cur.fetchall()]
            explain_conn.rollback()
            item = summarize_plan(plan)
            plans.append({"sql": statement, "plan": plan, **item})
            short = " ".join(statement.split())[:90].replace("|", "\\|")
            print(f"| {number} | `{short}` | {item['top']} | {_ms(item['planning_ms'])} | {_ms(item['execution_ms'])} | {item['buffers'] or '—'} |")
        print()
        if args.plans:
            for number, item in enumerate(plans, start=1):
                print(f"{path} #{number}\n```\n{item['sql']}\n\n" + "\n".join(item["plan"]) + "\n```\n")
        report.append({"path": path, "status": status, "statements": plans})
    explain_conn.close()
    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump({"connect_ms": connect_ms, "round_trip_ms": round_trip, "endpoints": report}, handle, ensure_ascii=False, indent=2)
    return 0


# ---- --setup-stubs: point a sandbox at the stubs of tools/stubs ----------------------------------------------

STUB_HOST = "127.0.0.1"
STUB_PORTS = {"llm": 5090, "confluence": 5091, "prometheus": 5092}
# lt_framework of settings.example.py reads k6 from InfluxDB through Grafana (InfluxQL). The Prometheus stub
# speaks PromQL, so the bench area keeps the same 11 labels (and so the same roles and SLA references) and
# gets PromQL queries in the same order. tests/test_perf_bench.py checks the counts against the example.
LT_PROMQL_QUERIES = (
    "sum by (group, name) (rate(k6_http_reqs_total[1m]))",
    "sum by (group, check) (rate(k6_checks_total[1m]))",
    "1000 * histogram_quantile(0.95, sum by (le, group, name) (rate(k6_http_req_duration_seconds_bucket[1m])))",
    "sum by (all) (rate(k6_http_reqs_total[1m]))",
    "1000 * histogram_quantile(0.95, sum by (le) (rate(k6_http_req_duration_seconds_bucket[1m])))",
    "1000 * histogram_quantile(0.99, sum by (le) (rate(k6_http_req_duration_seconds_bucket[1m])))",
    "1000 * sum(rate(k6_http_req_duration_seconds_sum[1m])) / sum(rate(k6_http_req_duration_seconds_count[1m]))",
    "100 * sum(rate(k6_http_req_failed_total[1m])) / sum(rate(k6_http_reqs_total[1m]))",
    "sum by (group, name) (rate(k6_http_req_failed_total[1m]))",
    "max(k6_vus)",
    "sum by (lt_instance) (rate(k6_http_reqs_total[1m]))",
)
LT_LABEL_KEYS = (
    ("group", "name"), ("group", "check"), ("group", "name"), ("all",), (), (), (), (), ("group", "name"), (), ("lt_instance",),
)


def _merged(base: Any, override: dict) -> dict:
    """``base`` (a dict or anything) deep-merged with ``override``; lists and scalars are replaced."""
    out = dict(base) if isinstance(base, dict) else {}
    for key, value in override.items():
        out[key] = _merged(out.get(key), value) if isinstance(value, dict) else value
    return out


def stub_settings_runtime(existing: dict, area: str, app_url: str, host: str = STUB_HOST, ports: Optional[dict] = None) -> dict:
    """settings_runtime.json of a sandbox wired to the LLM, Confluence and Prometheus stubs of tools/stubs.

    Global keys (``llm``, ``confluence``, ``data_sources``, ``limits``) are merged into CONFIG when the app
    starts, so the sandbox must be restarted afterwards. The area replaces its domain bindings with the stub
    Prometheus (an area binding object replaces the global one entirely) and swaps its lt_framework queries
    for PromQL. Idempotent; other keys and areas are kept. ``ValueError`` when the area is missing.
    """
    ports = {**STUB_PORTS, **(ports or {})}
    data = json.loads(json.dumps(existing or {}))
    per_area = data.get("per_area")
    if not isinstance(per_area, dict) or not isinstance(per_area.get(area), dict):
        raise ValueError(f"area {area!r} is not in the runtime file: run tools/seed_bench_data.py --runtime-only first")
    data["llm"] = _merged(data.get("llm"), {
        "provider": "openai",
        "openai": {
            "api_base_url": f"http://{host}:{ports['llm']}/v1", "api_key": "stub", "model": "stub-model",
            "verify": False, "proxies": {"http": "", "https": ""}, "request_timeout_sec": 300,
        },
    })
    data["confluence"] = _merged(data.get("confluence"), {
        "enabled": True, "base_url": f"http://{host}:{ports['confluence']}", "username": "reader", "password": "secret",
        "space_key": "PERF", "parent_page_id": "42", "forecast_parent_page_id": "42", "verify_ssl": False,
        "app_url": app_url,
    })
    data["data_sources"] = _merged(data.get("data_sources"), {
        "prometheus": {"title": "Prometheus (заглушка)", "type": "prometheus", "prometheus": {"url": f"http://{host}:{ports['prometheus']}"}},
    })
    # Every bench report uses the LLM; the daily budget of the single sign-in-off user would run out.
    data["limits"] = _merged(data.get("limits"), {"llm_reports_per_user_per_day": None})
    entry = per_area[area]
    entry["domain_sources"] = {"default": {"source": "prometheus"}}
    entry["queries"] = _merged(entry.get("queries"), {"lt_framework": {
        "influxql_queries": [],
        "promql_queries": list(LT_PROMQL_QUERIES),
        "label_keys_list": [list(keys) for keys in LT_LABEL_KEYS],
    }})
    return data


def _sandbox_app_dir(sandbox: int) -> Path:
    return Path(os.environ.get("TEMP") or tempfile.gettempdir()) / f"ll_ui_s{sandbox}" / "app"


def _write_json_atomic(path: Path, payload: dict) -> int:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return len(text.encode("utf-8"))


def setup_stubs(args: argparse.Namespace) -> int:
    """Writes the stub configuration into the sandbox settings_runtime.json (never into a working tree)."""
    if args.runtime_json:
        path = Path(args.runtime_json)
    elif args.sandbox is not None:
        path = _sandbox_app_dir(args.sandbox) / "settings_runtime.json"
    else:
        raise SystemExit("--setup-stubs needs --sandbox N or --runtime-json")
    if (path.parent / ".git").exists():
        raise SystemExit(f"{path.parent} is a git working tree, not a sandbox copy: its runtime file is not touched")
    if not path.parent.is_dir():
        raise SystemExit(f"{path.parent} not found: run make_sandbox.py first")
    existing = json.loads(path.read_text(encoding="utf-8-sig") or "{}") if path.exists() else {}
    try:
        data = stub_settings_runtime(existing, args.area, app_url=args.base)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    size = _write_json_atomic(path, data)
    print(f"stubs: {path} ({size // 1024} KB): llm -> {data['llm']['openai']['api_base_url']}, "
          f"confluence -> {data['confluence']['base_url']}, prometheus -> {data['data_sources']['prometheus']['prometheus']['url']}, "
          f"area '{args.area}' bound to the stub Prometheus. Restart the sandbox server.")
    return 0


# ---- --during-job: latency while a report is being built ------------------------------------------------------

# The pages a user keeps open while a report runs; the polls of 11.7 of docs/perf-plan.md are among them.
DURING_JOB_ENDPOINTS = (
    "/",
    "/dashboard_data",
    "/runs?limit=20",
    "/runs?limit=20&sort=max_rps",
    "/load_tests/active?limit=5&live=1",
    "/jobs?status=running&limit=10",
    "/services",
    "/static/css/app.css",
)
JOB_STATUS_ROW = "/job_status/<id>"
# Progress marks of AI/pipeline.py: collection runs 10-45 %, domain analyses 47-80 %, the final analysis from 83 %.
JOB_PHASES = (("сбор метрик и подготовка", 47), ("анализ ИИ по доменам", 83), ("итоговый анализ и сохранение", None))
DURING_COLUMNS = (
    ("Эндпоинт", "path"),
    ("idle p50", "idle_p50"),
    ("idle p95", "idle_p95"),
    ("job p50", "job_p50"),
    ("job p95", "job_p95"),
    ("job/idle p50", "ratio_p50"),
    ("job/idle p95", "ratio_p95"),
    ("app idle p50", "idle_app_p50"),
    ("app job p50", "job_app_p50"),
    ("Замеров в job", "job_count"),
    ("Ошибки", "errors"),
)
COMPARE_COLUMNS = (("было job p50", "before_job_p50"), ("было job p95", "before_job_p95"), ("было/стало job p95", "ratio_before"))
_CSRF_META = re.compile(r"""<meta\s+name=["']csrf-token["']\s+content=["']([^"']*)["']""", re.IGNORECASE)
_INT_FIELDS = frozenset({"job_count", "errors"})


def extract_csrf_token(html: str) -> str:
    """Token of ``<meta name="csrf-token">`` of a page ('' when sign-in is off or there is no tag)."""
    match = _CSRF_META.search(html or "")
    return match.group(1) if match else ""


def report_window(now: datetime, minutes: int) -> tuple[str, str]:
    """ISO start/end (UTC, whole minutes) of a ``minutes`` long window that ends at the current minute."""
    end = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=minutes)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


class KeepAliveClient:
    """A client that keeps its HTTP/1.1 connection while the server allows it (gunicorn does, as for a
    browser tab), reopens it otherwise or after an error, and keeps cookies.

    The Werkzeug dev server of the sandbox answers every request with ``Connection: close``, so there
    each request still opens a connection: keep ``--pause-ms`` so that a job of several minutes does not
    exhaust the ephemeral ports of a busy Windows host (TIME_WAIT lasts two minutes).
    """

    def __init__(self, target: Target):
        self.target = target
        self.connection: Optional[http.client.HTTPConnection] = None
        self.cookies: dict[str, str] = {"project_area": target.area} if target.area else {}

    def _open(self) -> http.client.HTTPConnection:
        if self.connection is None:
            connection_class = http.client.HTTPSConnection if self.target.scheme == "https" else http.client.HTTPConnection
            self.connection = connection_class(self.target.host, self.target.port, timeout=self.target.timeout)
        return self.connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def request(self, method: str, path: str, body: Any = None, headers: Optional[dict] = None) -> tuple[int, float, Optional[float], bytes]:
        """``(status, client ms, app ms, body)``; status 0 when there is no answer."""
        merged = {k: v for k, v in self.target.headers.items() if k != "Cookie"}
        if self.cookies:
            merged["Cookie"] = "; ".join(f"{name}={value}" for name, value in self.cookies.items())
        merged.update(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            merged["Content-Type"] = "application/json"
        started = time.perf_counter()
        try:
            connection = self._open()
            connection.request(method, self.target.prefix + path, body=data, headers=merged)
            response = connection.getresponse()
            payload = response.read()
            elapsed = (time.perf_counter() - started) * 1000.0
            for header in response.msg.get_all("Set-Cookie") or []:
                name, _, rest = header.partition("=")
                if name.strip():
                    self.cookies[name.strip()] = rest.split(";", 1)[0]
            if response.will_close:
                self.close()
            return response.status, elapsed, parse_server_timing(response.getheader("Server-Timing")), payload
        except (OSError, http.client.HTTPException):
            self.close()
            return 0, (time.perf_counter() - started) * 1000.0, None, b""

    def get(self, path: str) -> tuple[int, float, Optional[float], int]:
        status, elapsed, app, payload = self.request("GET", path)
        return status, elapsed, app, len(payload)


def cycle_clients(
    target: Target,
    paths: Sequence[str],
    concurrency: int,
    *,
    pause_sec: float,
    stop: threading.Event,
    cycles: Optional[int] = None,
    client_factory=KeepAliveClient,
) -> list[tuple[str, float, tuple]]:
    """``concurrency`` keep-alive clients go round the ``paths`` (each from another offset) until ``stop`` is
    set or each made ``cycles`` rounds; returns ``(path, monotonic start, sample)`` for every request."""
    samples: list[tuple[str, float, tuple]] = []
    lock = threading.Lock()

    def client(number: int) -> None:
        http_client = client_factory(target)
        try:
            done = 0
            while not stop.is_set() and (cycles is None or done < cycles):
                for k in range(len(paths)):
                    if stop.is_set():
                        break
                    path = paths[(k + number) % len(paths)]
                    began = time.monotonic()
                    sample = http_client.get(path)
                    with lock:
                        samples.append((path, began, sample))
                    if pause_sec > 0:
                        stop.wait(pause_sec)
                done += 1
        finally:
            http_client.close()

    threads = [threading.Thread(target=client, args=(n,), daemon=True) for n in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return samples


def phase_stats(samples: Sequence[tuple[str, float, tuple]], since: float = float("-inf"), until: float = float("inf")) -> dict[str, dict]:
    """Per path: client p50/p95/max, ``Server-Timing`` p50/p95, errors and statuses of the requests started
    in ``[since, until)``."""
    by_path: dict[str, list[tuple]] = {}
    for path, began, sample in samples:
        if since <= began < until:
            by_path.setdefault(path, []).append(sample)
    out = {}
    for path, items in by_path.items():
        ok = [s for s in items if 0 < s[0] < 400]
        app = summarize([s[2] for s in items if s[2] is not None])
        out[path] = {
            **summarize([s[1] for s in items]),
            "app_p50": app["p50"],
            "app_p95": app["p95"],
            "errors": len(items) - len(ok),
            "statuses": dict(Counter(str(s[0]) for s in items)),
        }
    return out


def during_rows(idle: dict[str, dict], during: dict[str, dict], paths: Sequence[str]) -> list[dict]:
    """Rows of the idle / during-job table: the measured paths, then the job status poll."""
    rows = []
    for path in [*paths, *[p for p in during if p not in paths]]:
        before, now = idle.get(path) or {}, during.get(path) or {}
        rows.append({
            "path": path,
            "idle_p50": before.get("p50"),
            "idle_p95": before.get("p95"),
            "job_p50": now.get("p50"),
            "job_p95": now.get("p95"),
            "job_max": now.get("max"),
            "idle_app_p50": before.get("app_p50"),
            "job_app_p50": now.get("app_p50"),
            "job_count": now.get("count", 0),
            "errors": int(before.get("errors", 0)) + int(now.get("errors", 0)),
        })
    return rows


def _cell(key: str, row: dict) -> str:
    if key == "path":
        return f"`{row['path']}`"
    if key == "ratio_p50":
        return _ratio(row.get("job_p50"), row.get("idle_p50"))
    if key == "ratio_p95":
        return _ratio(row.get("job_p95"), row.get("idle_p95"))
    if key == "ratio_before":
        return _ratio(row.get("before_job_p95"), row.get("job_p95"))
    if key in _INT_FIELDS:
        return str(int(row.get(key) or 0))
    return _ms(row.get(key))


def render_during_table(rows: Sequence[dict], before: Optional[Sequence[dict]] = None) -> str:
    """Markdown idle / during-job table; ``job/idle`` above 1 means slower while the report runs. With
    ``before`` (rows of an earlier run, e.g. thread mode) its job p50/p95 and the ratio before/now."""
    columns = list(DURING_COLUMNS)
    if before is not None:
        columns += list(COMPARE_COLUMNS)
        old = {item["path"]: item for item in before}
        rows = [{**row, "before_job_p50": (old.get(row["path"]) or {}).get("job_p50"), "before_job_p95": (old.get(row["path"]) or {}).get("job_p95")} for row in rows]
    lines = ["| " + " | ".join(title for title, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(key, row) for _, key in columns) + " |")
    return "\n".join(lines)


def _parse_cell(key: str, text: str) -> Any:
    text = text.strip()
    if key == "path":
        return text.strip("`")
    if text in ("—", "-", ""):
        return 0 if key in _INT_FIELDS else None
    if key in _INT_FIELDS:
        return int(text)
    return float(text.rstrip("×"))


def parse_during_tables(text: str) -> list[list[dict]]:
    """Every idle / during-job table of a markdown text (as ``render_during_table`` prints them), in order.

    Rows come back with the keys of ``during_rows`` (ratios dropped, they are derived); a table copied into
    docs/perf-baseline.md can be the ``--compare`` baseline of a later run.
    """
    titles = {title: key for title, key in (*DURING_COLUMNS, *COMPARE_COLUMNS)}
    tables: list[list[dict]] = []
    keys: Optional[list[Optional[str]]] = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            keys = None
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if keys is None:
            if cells and cells[0] == DURING_COLUMNS[0][0] and len(cells) > 1 and cells[1] == DURING_COLUMNS[1][0]:
                keys = [titles.get(cell) for cell in cells]
                tables.append([])
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        row: dict[str, Any] = {}
        for key, cell in zip(keys, cells):
            if key and not key.startswith("ratio"):
                row[key] = _parse_cell(key, cell)
        tables[-1].append(row)
    return tables


def job_phase_windows(timeline: Sequence[tuple[float, str, int, str]], finished: Optional[float]) -> list[tuple[str, Optional[float], Optional[float]]]:
    """``(phase, begin, end)`` in seconds since the job started, from its ``(seconds, status, progress, message)``
    polls: a phase ends at the first poll that shows the progress mark of the next one."""

    def first_at(percent: int) -> Optional[float]:
        return next((t for t, _status, progress, _msg in timeline if progress >= percent), None)

    marks: list[Optional[float]] = [0.0] + [first_at(threshold) for _name, threshold in JOB_PHASES if threshold is not None] + [finished]
    return [(name, begin, end) for (name, _threshold), begin, end in zip(JOB_PHASES, marks, marks[1:])]


def job_phases(timeline: Sequence[tuple[float, str, int, str]], finished: Optional[float]) -> list[tuple[str, Optional[float]]]:
    """Seconds per phase of a report job (None when a mark was not seen)."""
    return [
        (name, None if begin is None or end is None else max(0.0, end - begin))
        for name, begin, end in job_phase_windows(timeline, finished)
    ]


def render_phase_table(per_phase: Sequence[tuple[str, dict[str, dict]]], paths: Sequence[str]) -> str:
    """p50/p95 of every endpoint within each phase of the job: where the slowdown comes from."""
    head = ["Эндпоинт"]
    for name, _stats in per_phase:
        head += [f"{name}: p50", f"{name}: p95", f"{name}: n"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for path in paths:
        row = [f"`{path}`"]
        for _name, stats in per_phase:
            item = stats.get(path) or {}
            row += [_ms(item.get("p50")), _ms(item.get("p95")), str(item.get("count", 0))]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _stub_call(url: str, path: str, method: str = "GET", timeout: float = 3.0) -> Optional[dict]:
    """JSON of a stub control endpoint, or None when the stub is not there (local stubs only)."""
    if not url or (urlsplit(url).hostname or "") not in LOCAL_HOSTS:
        return None
    request = urllib.request.Request(url.rstrip("/") + path, method=method, data=b"" if method == "POST" else None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy of the environment
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except (OSError, ValueError):
        return None


def _stub_root(url: str) -> str:
    parts = urlsplit(url or "")
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else ""


class JobWatcher(threading.Thread):
    """Polls ``/job_status/<id>`` like the "Новый отчёт" page does, until the job is done, failed or late."""

    def __init__(self, target: Target, job_id: str, poll_sec: float, timeout_sec: float, started: float):
        super().__init__(daemon=True)
        self.target = target
        self.path = f"/job_status/{quote(job_id, safe='')}"
        self.poll_sec = poll_sec
        self.deadline = started + timeout_sec
        self.started = started
        self.timeline: list[tuple[float, str, int, str]] = []
        self.samples: list[tuple[str, float, tuple]] = []
        self.final: Optional[dict] = None
        self.finished: Optional[float] = None
        self.timed_out = False

    def run(self) -> None:
        client = KeepAliveClient(self.target)
        try:
            while True:
                began = time.monotonic()
                status, elapsed, app, payload = client.request("GET", self.path)
                self.samples.append((JOB_STATUS_ROW, began, (status, elapsed, app, len(payload))))
                if status == 200:
                    try:
                        job = json.loads(payload)
                    except ValueError:
                        job = {}
                    entry = (round(time.monotonic() - self.started, 2), str(job.get("status") or ""), int(job.get("progress") or 0), str(job.get("message") or ""))
                    if not self.timeline or self.timeline[-1][1:] != entry[1:]:
                        self.timeline.append(entry)
                    if entry[1] in ("done", "error"):
                        self.final, self.finished = job, time.monotonic()
                        return
                elif status == 404:
                    self.final, self.finished = {"status": "not_found"}, time.monotonic()
                    return
                if time.monotonic() > self.deadline:
                    self.timed_out = True
                    return
                time.sleep(self.poll_sec)
        finally:
            client.close()


def _server_duration(job: dict) -> Optional[float]:
    try:
        created = datetime.fromisoformat(str(job.get("created_at")))
        updated = datetime.fromisoformat(str(job.get("updated_at")))
    except (TypeError, ValueError):
        return None
    return (updated - created).total_seconds()


def render_job_summary(job: dict) -> str:
    """Markdown list under the table: job, duration, phases, stub calls, report check."""
    lines = []
    status = job.get("status") or "?"
    duration = job.get("duration_sec")
    server = job.get("server_duration_sec")
    lines.append(
        f"- Задача `{job.get('job_id')}` ({job.get('label') or 'режим не указан'}): {status}"
        + (f", {duration:.1f} с по опросу `/job_status`" if duration is not None else "")
        + (f", {server:.1f} с по `created_at`/`updated_at` задачи" if server is not None else "")
        + (" — превышен `--job-timeout`" if job.get("timed_out") else "")
    )
    phases = [f"{name} {seconds:.1f} с" for name, seconds in job.get("phases") or [] if seconds is not None]
    if phases:
        lines.append("- Фазы: " + " · ".join(phases))
    llm = job.get("llm_stub")
    if isinstance(llm, dict):
        kinds = ", ".join(f"{kind} {count}" for kind, count in sorted((llm.get("by_kind") or {}).items()))
        lines.append(
            f"- LLM-заглушка: {llm.get('calls', 0)} вызовов ({kinds or '—'}), задержка {llm.get('delay_sec', 0)} с на ответ, "
            f"одновременно до {llm.get('max_in_flight', 0)}"
        )
    prom = job.get("prometheus_stub")
    if isinstance(prom, dict):
        requests_by_kind = prom.get("requests") or {}
        points = f"{int(prom.get('points') or 0):,}".replace(",", " ")
        lines.append(f"- Prometheus-заглушка: {requests_by_kind.get('query_range', 0)} запросов `query_range`, {points} точек")
    report = job.get("report") or {}
    if report:
        verdict = report.get("final_verdict")
        lines.append(
            f"- Отчёт `{report.get('run_name')}`: `{report.get('page_url')}` → {report.get('page_status')}, "
            f"`/llm_reports` → {report.get('llm_status')}, доменов {report.get('domains', 0)}"
            + (f", итог «{verdict}»" if verdict else "")
            + (f"; удалён (`DELETE /runs` → {report['deleted']})" if report.get("deleted") is not None else "; оставлен (`--keep-report`)")
        )
    if job.get("error"):
        lines.append(f"- Ошибка: {job['error']}")
    few = [row["path"] for row in job.get("rows") or [] if 0 < int(row.get("job_count") or 0) < 10]
    if few:
        lines.append("- Мало замеров во время задачи (меньше 10): " + ", ".join(f"`{p}`" for p in few))
    return "\n".join(lines)


def _check_report(client: KeepAliveClient, job: dict, run_name: str, csrf: str, keep: bool) -> dict:
    """The report of a finished job opens (page and saved analyses); then it is deleted unless ``keep``."""
    page_url = str(job.get("page_url") or job.get("report_url") or "")
    out: dict[str, Any] = {"run_name": str(job.get("run_name") or run_name), "page_url": page_url}
    if page_url:
        out["page_status"] = client.request("GET", page_url)[0]
    status, _ms_, _app, payload = client.request("GET", "/llm_reports?" + urlencode({"run_name": out["run_name"]}))
    out["llm_status"] = status
    try:
        rows = json.loads(payload) if status == 200 else []
    except ValueError:
        rows = []
    rows = rows if isinstance(rows, list) else []
    out["domains"] = len({row.get("domain") for row in rows if isinstance(row, dict)})
    final = next((row for row in rows if isinstance(row, dict) and row.get("domain") == "final"), None)
    out["final_verdict"] = final.get("verdict") if final else None
    out["ok"] = out.get("page_status") == 200 and status == 200 and final is not None
    if not keep:
        headers = {"X-CSRF-Token": csrf} if csrf else {}
        out["deleted"] = client.request("DELETE", "/runs/" + quote(out["run_name"], safe=""), headers=headers)[0]
    return out


def during_job(args: argparse.Namespace) -> int:
    """Idle round, then a report job with the same requests running alongside it; prints both."""
    target = Target(args.base, args.area, args.token, args.timeout)
    paths = selected_endpoints(args, DURING_JOB_ENDPOINTS)
    pause = max(0.0, args.pause_ms / 1000.0)
    before = load_during_baseline(args.compare, args.compare_table) if args.compare else None
    llm_root, prom_root = _stub_root(args.llm_stub), _stub_root(args.prom_stub)
    for root in (llm_root, prom_root):
        _stub_call(root, "/_stub/reset", "POST")

    print(f"idle: {len(paths)} endpoints x {args.rounds} rounds x {args.concurrency} client(s)", file=sys.stderr)
    never = threading.Event()
    if args.warmup:
        cycle_clients(target, paths, args.concurrency, pause_sec=pause, stop=never, cycles=args.warmup)
    idle = phase_stats(cycle_clients(target, paths, args.concurrency, pause_sec=pause, stop=never, cycles=args.rounds))

    session = KeepAliveClient(target)
    page_status, _ms_, _app, page = session.request("GET", "/new")
    csrf = extract_csrf_token(page.decode("utf-8", "replace"))
    start_iso, end_iso = report_window(datetime.now(timezone.utc), args.window_min)
    run_name = args.run_name or datetime.now().strftime("bench-job-%Y%m%d-%H%M%S")
    payload = {
        "start": start_iso, "end": end_iso, "service": args.service, "project_area": args.area,
        "web_only": True, "use_llm": not args.no_llm, "test_type": args.test_type, "run_name": run_name,
    }
    status, _ms_, _app, body = session.request("POST", "/create_report", body=payload, headers={"X-CSRF-Token": csrf} if csrf else {})
    try:
        accepted = json.loads(body)
    except ValueError:
        accepted = {}
    if status != 200 or not accepted.get("job_id"):
        session.close()
        raise SystemExit(f"POST /create_report -> {status} (page /new -> {page_status}): {body[:300].decode('utf-8', 'replace')}")
    job_started = time.monotonic()
    job_id = str(accepted["job_id"])
    print(f"job {job_id} accepted: {args.service} {start_iso}..{end_iso}, run {run_name}", file=sys.stderr)

    watcher = JobWatcher(target, job_id, args.poll_sec, args.job_timeout, job_started)
    stop = threading.Event()
    load: list = []
    runner = threading.Thread(target=lambda: load.extend(cycle_clients(target, paths, args.concurrency, pause_sec=pause, stop=stop)), daemon=True)
    watcher.start()
    runner.start()
    watcher.join()
    stop.set()
    runner.join()
    finished = watcher.finished if watcher.finished is not None else time.monotonic()
    all_samples = [*load, *watcher.samples]
    during = phase_stats(all_samples, since=job_started, until=finished)
    windows = job_phase_windows(watcher.timeline, finished - job_started if watcher.finished is not None else None)
    per_phase = [
        (name, phase_stats(all_samples, since=job_started + begin, until=job_started + end))
        for name, begin, end in windows
        if begin is not None and end is not None
    ]

    final = watcher.final or {}
    job: dict[str, Any] = {
        "job_id": job_id,
        "label": args.label,
        "status": final.get("status") or ("timeout" if watcher.timed_out else "unknown"),
        "duration_sec": round(finished - job_started, 2),
        "server_duration_sec": _server_duration(final),
        "timed_out": watcher.timed_out,
        "timeline": watcher.timeline,
        "phases": job_phases(watcher.timeline, finished - job_started if watcher.finished is not None else None),
        "llm_stub": _stub_call(llm_root, "/_stub/stats"),
        "prometheus_stub": _stub_call(prom_root, "/_stub/stats"),
        "window": [start_iso, end_iso],
        "service": args.service,
        "run_name": run_name,
        "message": final.get("message"),
        "error": final.get("error") if final.get("status") == "error" else None,
    }
    if final.get("status") == "done":
        job["report"] = _check_report(session, final, run_name, csrf, args.keep_report)
    session.close()
    rows = during_rows(idle, during, paths)
    job["rows"] = rows

    print(f"\nduring-job bench {args.base} area={args.area or '-'} rounds={args.rounds} concurrency={args.concurrency} "
          f"pause={args.pause_ms:g} ms label={args.label or '-'}\n")
    print(render_during_table(rows, before))
    print()
    print(render_job_summary(job))
    if per_phase:
        print("\nПо фазам задачи (мс):\n")
        print(render_phase_table(per_phase, [row["path"] for row in rows]))
    if args.json_path:
        meta = {
            "mode": "during_job", "base": args.base, "area": args.area, "rounds": args.rounds, "concurrency": args.concurrency,
            "warmup": args.warmup, "pause_ms": args.pause_ms, "label": args.label, "window_min": args.window_min,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
        }
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump({"meta": meta, "idle": idle, "during": during, "per_phase": dict(per_phase), "rows": rows, "job": job},
                      handle, ensure_ascii=False, indent=2)
    report_ok = (job.get("report") or {}).get("ok", False)
    return 0 if job["status"] == "done" and report_ok else 1


def load_during_baseline(path: str, table: int = 1) -> list[dict]:
    """Rows of an earlier during-job run: its ``--json`` file or the ``table``-th table of a markdown file."""
    text = Path(path).read_text(encoding="utf-8")
    if path.lower().endswith((".md", ".markdown", ".txt")):
        tables = parse_during_tables(text)
        if not 1 <= table <= len(tables):
            raise SystemExit(f"{path}: {len(tables)} during-job table(s), asked for #{table}")
        return tables[table - 1]
    data = json.loads(text)
    if not isinstance(data.get("rows"), list):
        raise SystemExit(f"{path}: not a --during-job result (no 'rows')")
    return data["rows"]


# ---- command line --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency bench of the LoadLens endpoints (see the module docstring).")
    parser.add_argument("--base", default="http://127.0.0.1:5071", help="address of the application")
    parser.add_argument("--sandbox", type=int, help="sandbox number N: same as --base http://127.0.0.1:<5070+N>")
    parser.add_argument("--area", default="bench", help="project area cookie ('' to send none)")
    parser.add_argument("--rounds", type=int, default=20, help="requests per client and endpoint")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel clients per endpoint")
    parser.add_argument("--warmup", type=int, default=2, help="uncounted requests per endpoint before measuring")
    parser.add_argument("--endpoint", action="append", dest="endpoints", help="measure this path instead of the default list (repeatable)")
    parser.add_argument("--only", help="measure only the default paths that contain this text")
    parser.add_argument("--json", dest="json_path", help="save the results to this file")
    parser.add_argument("--compare", help="results of an earlier run (--json) to compare with")
    parser.add_argument("--label", default="", help="free text saved in the JSON meta (branch, change)")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    parser.add_argument("--token", default=os.environ.get("LOADLENS_BENCH_TOKEN", ""), help="API token when sign-in is on")
    parser.add_argument("--allow-remote", action="store_true", help="allow a host other than 127.0.0.1/localhost")
    parser.add_argument("--explain", action="store_true", help="instead of timing: EXPLAIN (ANALYZE, BUFFERS) the SQL of each endpoint (in-process, sandbox copy only)")
    parser.add_argument("--app-dir", help="sandbox app directory for --explain (default %%TEMP%%/ll_ui_s<N>/app)")
    parser.add_argument("--plans", action="store_true", help="with --explain: print the full plans too")
    stubs = parser.add_argument_group("--setup-stubs: wire a sandbox to tools/stubs")
    stubs.add_argument("--setup-stubs", action="store_true", help="write the LLM/Confluence/Prometheus stub keys into the sandbox settings_runtime.json and exit")
    stubs.add_argument("--runtime-json", help="settings_runtime.json to change (default %%TEMP%%/ll_ui_s<N>/app/settings_runtime.json)")
    job = parser.add_argument_group("--during-job: latency while a report is being built")
    job.add_argument("--during-job", action="store_true", help="idle round, then POST /create_report and the same requests until the job ends")
    job.add_argument("--service", default="svc01", help="service of the report (a seeded bench service)")
    job.add_argument("--window-min", type=int, default=60, help="report window, minutes up to now")
    job.add_argument("--test-type", default="step", help="test_type of the report")
    job.add_argument("--run-name", default="", help="run name of the report (default bench-job-<time>)")
    job.add_argument("--no-llm", action="store_true", help="use_llm false: collection and saving only")
    job.add_argument("--keep-report", action="store_true", help="keep the report run (default: DELETE /runs/<name> after the check)")
    job.add_argument("--pause-ms", type=float, default=50.0, help="pause of every client after each request, ms")
    job.add_argument("--poll-sec", type=float, default=1.0, help="interval of the /job_status polls")
    job.add_argument("--job-timeout", type=float, default=900.0, help="give up waiting for the job after this many seconds")
    job.add_argument("--llm-stub", default=f"http://{STUB_HOST}:{STUB_PORTS['llm']}", help="LLM stub to count calls on ('' to skip)")
    job.add_argument("--prom-stub", default=f"http://{STUB_HOST}:{STUB_PORTS['prometheus']}", help="Prometheus stub to count queries on ('' to skip)")
    job.add_argument("--compare-table", type=int, default=1, help="with --compare FILE.md: which during-job table of the file (1-based)")
    args = parser.parse_args(argv)
    if args.sandbox is not None:
        args.base = f"http://127.0.0.1:{5070 + args.sandbox}"
    if args.rounds < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("--rounds and --concurrency must be positive, --warmup not negative")
    if args.window_min < 1 or args.job_timeout <= 0 or args.poll_sec <= 0 or args.pause_ms < 0:
        parser.error("--window-min, --job-timeout and --poll-sec must be positive, --pause-ms not negative")
    if sum(map(bool, (args.explain, args.setup_stubs, args.during_job))) > 1:
        parser.error("--explain, --setup-stubs and --during-job are separate modes")
    host = urlsplit(args.base).hostname or ""
    if host not in LOCAL_HOSTS and not args.allow_remote:
        parser.error(f"refusing to load {host}: pass --allow-remote to measure a non-local instance")
    return args


def selected_endpoints(args: argparse.Namespace, default: Sequence[str] = ENDPOINTS) -> list[str]:
    paths: Iterable[str] = args.endpoints or default
    if args.only:
        paths = [p for p in paths if args.only in p]
    return list(paths)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.explain:
        return explain_endpoints(args)
    if args.setup_stubs:
        return setup_stubs(args)
    if args.during_job:
        return during_job(args)
    target = Target(args.base, args.area, args.token, args.timeout)
    before = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            before = json.load(handle).get("results") or []
    meta: dict[str, Any] = {
        "base": args.base,
        "area": args.area,
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "label": args.label,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    results = []
    for path in selected_endpoints(args):
        item = measure(target, path, args.rounds, args.concurrency, args.warmup)
        results.append(item)
        print(f"  {path}: p50 {_ms(item['p50'])} ms, p95 {_ms(item['p95'])} ms, errors {item['errors']}", file=sys.stderr)
    print(f"\nbench {args.base} area={args.area or '-'} rounds={args.rounds} concurrency={args.concurrency} {meta['started_at']}\n")
    print(render_table(results, before))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump({"meta": meta, "results": results}, handle, ensure_ascii=False, indent=2)
    return 1 if any(item["errors"] for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
