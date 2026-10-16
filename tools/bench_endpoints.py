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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import urlsplit

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
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump({"connect_ms": connect_ms, "round_trip_ms": round_trip, "endpoints": report}, handle, ensure_ascii=False, indent=2)
    return 0


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
    args = parser.parse_args(argv)
    if args.sandbox is not None:
        args.base = f"http://127.0.0.1:{5070 + args.sandbox}"
    if args.rounds < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("--rounds and --concurrency must be positive, --warmup not negative")
    host = urlsplit(args.base).hostname or ""
    if host not in LOCAL_HOSTS and not args.allow_remote:
        parser.error(f"refusing to load {host}: pass --allow-remote to measure a non-local instance")
    return args


def selected_endpoints(args: argparse.Namespace) -> list[str]:
    paths: Iterable[str] = args.endpoints or ENDPOINTS
    if args.only:
        paths = [p for p in paths if args.only in p]
    return list(paths)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.explain:
        return explain_endpoints(args)
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
