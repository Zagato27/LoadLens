# SPDX-License-Identifier: Apache-2.0
"""Prometheus stub for a LoadLens sandbox: synthetic series for any PromQL query on 127.0.0.1:5092.

How to run (own console, or in the background; Ctrl+C stops it):

    C:/Python312/python.exe tools/stubs/prom_stub.py
    C:/Python312/python.exe tools/stubs/prom_stub.py --series 8 --steps 4 --delay-ms 20 --seed 7

Point the sandbox at it in its settings_runtime.json (``tools/bench_endpoints.py --setup-stubs`` writes
this and the rest of the stub configuration, see plans/perf-baseline.md, "Этап 3"):

    "data_sources": {"prometheus": {"title": "Prometheus", "type": "prometheus",
                                    "prometheus": {"url": "http://127.0.0.1:5092"}}},
    "domain_sources": {"default": {"source": "prometheus"}}

Endpoints (GET, or POST with a form body like the real API):

* ``/api/v1/query_range?query=&start=&end=&step=`` - ``resultType: matrix``: ``--series`` series per query,
  one point per ``step`` from ``start`` to ``end`` (unix seconds or RFC 3339; step in seconds or ``15s``,
  ``1m``). More than ``--max-points`` points per series is refused with 400, as Prometheus does at 11 000.
* ``/api/v1/query?query=&time=`` - ``resultType: vector``, one sample per series.
* ``/api/v1/status/buildinfo`` - the connection check of "Настройки → Источники данных".
* ``/api/v1/labels``, ``/api/v1/label/<name>/values`` - metric discovery of the settings page.
* ``/-/healthy``, ``/-/ready``.
* ``/_stub/stats`` (GET) - request counters; ``/_stub/reset`` (POST) sets them to zero. Used by
  ``tools/bench_endpoints.py --during-job`` to report how many queries a report made.

The data: every series carries the labels named in the query (``by (...)`` clauses and the target label of
``label_replace``) plus a fixed set (application, instance, pool, datname, node, namespace, pod, topic, ...),
valued ``<label>-<NN>``, so any ``label_keys_list`` of the query packs finds its keys. Values follow a load
profile of ``--steps`` equal steps across the requested window (``--steps 0``: flat) with 3 % noise; queries
that look like percentages (``100 *``, ``(%)``, ``ratio``) stay within 0..100. Everything is a pure function
of (``--seed``, query, series number, window, step): the same request always gets the same answer.

Listens on 127.0.0.1/localhost; ``--host 0.0.0.0`` is for a container of the demo stack
(docker/docker-compose.demo.yml). Standard library only. ``--log FILE`` appends one JSON line
per request (off by default: a report makes about a hundred queries).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Sequence
from urllib.parse import parse_qs, urlsplit

DEFAULT_PORT = 5092
DEFAULT_SERIES = 6
DEFAULT_STEPS = 4
DEFAULT_MAX_POINTS = 11000
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# Inside a container (docker/docker-compose.demo.yml) the stub must answer the other containers.
CONTAINER_HOSTS = frozenset({"0.0.0.0", "::"})
VERSION = "2.53.0-loadlens-stub"
# Labels every series carries, whatever the query: the keys of the label_keys_list of the bundled query packs.
BASE_LABELS = (
    "application", "instance", "pool", "datname", "state", "server", "mode", "topic", "consumergroup",
    "node", "nodename", "namespace", "pod", "deployment", "horizontalpodautoscaler", "uri", "client_name",
    "service", "series", "job", "group", "name", "check", "lt_instance",
)
NOISE = 0.03
_BY_CLAUSE = re.compile(r"\b(?:by|without)\s*\(([^)]*)\)", re.IGNORECASE)
_LABEL_REPLACE_TARGET = re.compile(r"label_replace\s*\(.*?,\s*\"([A-Za-z_][A-Za-z0-9_]*)\"\s*,", re.IGNORECASE | re.DOTALL)
_LABEL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PERCENT_HINT = re.compile(r"100\s*\*|\(%\)|percent|_pct\b|ratio|utili[sz]ation", re.IGNORECASE)
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)(ms|s|m|h|d|w|y)$")
_DURATION_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
# Labels that would make no sense on an aggregated result.
_DROPPED_LABELS = frozenset({"le", "__name__"})


class BadRequest(ValueError):
    """A request Prometheus would answer with 400 bad_data."""


# ---- pure functions: parsing and data -------------------------------------------------------------------


def parse_time(raw: Optional[str], default: Optional[float] = None) -> float:
    """Unix seconds from a float string or an RFC 3339 timestamp."""
    if raw is None or str(raw).strip() == "":
        if default is None:
            raise BadRequest("missing time parameter")
        return float(default)
    text = str(raw).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BadRequest(f'cannot parse "{text}" to a valid timestamp') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def parse_step(raw: Optional[str]) -> float:
    """Step in seconds from ``15``, ``15.5`` or a Prometheus duration such as ``15s``/``1m``."""
    text = str(raw or "").strip()
    if not text:
        raise BadRequest("missing step parameter")
    try:
        value = float(text)
    except ValueError:
        match = _DURATION.match(text)
        if not match:
            raise BadRequest(f'cannot parse "{text}" to a valid duration') from None
        value = float(match.group(1)) * _DURATION_SECONDS[match.group(2)]
    if value <= 0:
        raise BadRequest("zero or negative query resolution step widths are not accepted. Try a positive integer")
    return value


def query_label_names(query: str) -> list[str]:
    """Label names the query groups by or creates, in order of appearance (``le`` dropped)."""
    names: list[str] = []
    for clause in _BY_CLAUSE.findall(query or ""):
        names.extend(part.strip() for part in clause.split(","))
    names.extend(_LABEL_REPLACE_TARGET.findall(query or ""))
    out: list[str] = []
    for name in names:
        if _LABEL_NAME.match(name) and name not in _DROPPED_LABELS and name not in out:
            out.append(name)
    return out


def series_labels(query: str, count: int) -> list[dict[str, str]]:
    """``count`` label sets: the labels of the query plus BASE_LABELS, valued ``<label>-<NN>``."""
    names = list(dict.fromkeys([*query_label_names(query), *BASE_LABELS]))
    return [{name: f"{name}-{index + 1:02d}" for name in names} for index in range(max(0, count))]


def _rng(seed: int, *parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in (seed, *parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def is_percent_query(query: str) -> bool:
    return bool(_PERCENT_HINT.search(query or ""))


def load_level(ts: float, start: float, end: float, steps: int) -> float:
    """Load multiplier of the step profile at ``ts``: 1.0 on the first step, 2.0 on the last one."""
    if steps <= 1 or end <= start:
        return 1.0
    position = min(max((ts - start) / (end - start), 0.0), 1.0)
    step = min(int(position * steps), steps - 1)
    return 1.0 + step / (steps - 1)


def timestamps(start: float, end: float, step: float) -> list[float]:
    """Points of a range query, aligned to ``start`` as Prometheus does (``end`` included when on the grid)."""
    if end < start:
        raise BadRequest("end timestamp must not be before start time")
    count = int(math.floor((end - start) / step + 1e-9)) + 1
    return [round(start + i * step, 3) for i in range(count)]


def point_count(start: float, end: float, step: float) -> int:
    return int(math.floor((end - start) / step + 1e-9)) + 1 if end >= start else 0


def series_values(query: str, index: int, points: Sequence[float], *, seed: int, steps: int, window: tuple[float, float]) -> list[float]:
    """Deterministic values of series ``index``: a base level, the step profile and 3 % noise."""
    rng = _rng(seed, query, index)
    percent = is_percent_query(query)
    base = rng.uniform(8.0, 35.0) if percent else 10 ** rng.uniform(0.0, 3.0)
    start, end = window
    noise_rng = _rng(seed, query, index, start, end)
    values = []
    for ts in points:
        noise = noise_rng.gauss(0.0, NOISE)
        value = base * load_level(ts, start, end, steps) * (1.0 + noise)
        if percent:
            value = min(100.0, value)
        values.append(round(max(0.0, value), 4))
    return values


def _sample(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def query_range_payload(params: dict[str, str], *, series: int, steps: int, seed: int, max_points: int) -> dict:
    """``data`` of /api/v1/query_range; raises :class:`BadRequest` like Prometheus does."""
    query = str(params.get("query") or "").strip()
    if not query:
        raise BadRequest('invalid parameter "query": empty query')
    start = parse_time(params.get("start"))
    end = parse_time(params.get("end"))
    step = parse_step(params.get("step"))
    if end < start:
        raise BadRequest('invalid parameter "end": end timestamp must not be before start time')
    if point_count(start, end, step) > max_points:
        raise BadRequest(
            f"exceeded maximum resolution of {max_points:,} points per timeseries. "
            "Try decreasing the query resolution (?step=XX)"
        )
    points = timestamps(start, end, step)
    result = []
    for index, labels in enumerate(series_labels(query, series)):
        values = series_values(query, index, points, seed=seed, steps=steps, window=(start, end))
        result.append({"metric": labels, "values": [[ts, _sample(v)] for ts, v in zip(points, values)]})
    return {"resultType": "matrix", "result": result}


def query_payload(params: dict[str, str], *, series: int, steps: int, seed: int, now: Optional[float] = None) -> dict:
    """``data`` of /api/v1/query: one sample per series at ``time`` (default: now)."""
    query = str(params.get("query") or "").strip()
    if not query:
        raise BadRequest('invalid parameter "query": empty query')
    at = parse_time(params.get("time"), default=now if now is not None else time.time())
    result = []
    for index, labels in enumerate(series_labels(query, series)):
        value = series_values(query, index, [at], seed=seed, steps=steps, window=(at - 3600.0, at))[0]
        result.append({"metric": labels, "value": [round(at, 3), _sample(value)]})
    return {"resultType": "vector", "result": result}


def label_values(name: str, series: int) -> list[str]:
    if name == "__name__":
        return sorted({
            "http_server_requests_seconds_count", "http_server_requests_seconds_sum", "jvm_memory_used_bytes",
            "jvm_memory_max_bytes", "process_cpu_usage", "hikaricp_connections_active", "kafka_consumergroup_lag",
            "node_cpu_seconds_total", "node_memory_MemAvailable_bytes", "container_cpu_usage_seconds_total",
            "k6_http_reqs_total", "k6_http_req_duration_p95",
        })
    return [f"{name}-{i + 1:02d}" for i in range(series)]


# ---- HTTP -------------------------------------------------------------------------------------------------


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.counts: dict[str, int] = {}
            self.points = 0
            self.errors = 0
            self.started = time.time()

    def add(self, kind: str, points: int = 0, error: bool = False) -> None:
        with self.lock:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            self.points += points
            self.errors += int(error)

    def snapshot(self) -> dict:
        with self.lock:
            return {"requests": dict(self.counts), "points": self.points, "errors": self.errors, "since": self.started}


class Handler(BaseHTTPRequestHandler):
    server_version = "PrometheusStub/1"
    options: argparse.Namespace
    stats: Stats
    log_lock = threading.Lock()

    def _params(self) -> dict[str, str]:
        parts = urlsplit(self.path)
        raw = parse_qs(parts.query, keep_blank_values=True)
        if self.command == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
            if "json" not in str(self.headers.get("Content-Type") or ""):
                for key, values in parse_qs(body, keep_blank_values=True).items():
                    raw.setdefault(key, values)
        return {key: values[-1] for key, values in raw.items()}

    def _send(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _log(self, record: dict) -> None:
        if not self.options.log:
            return
        with self.log_lock, open(self.options.log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _handle(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        params = self._params()
        opts = self.options
        if opts.delay_ms > 0 and path.startswith("/api/v1/query"):
            time.sleep(opts.delay_ms / 1000.0)
        try:
            if path == "/api/v1/query_range":
                data = query_range_payload(params, series=opts.series, steps=opts.steps, seed=opts.seed, max_points=opts.max_points)
                self.stats.add("query_range", points=sum(len(s["values"]) for s in data["result"]))
                self._log({"path": path, "query": params.get("query"), "start": params.get("start"), "end": params.get("end"), "step": params.get("step")})
                return self._send(200, {"status": "success", "data": data})
            if path == "/api/v1/query":
                data = query_payload(params, series=opts.series, steps=opts.steps, seed=opts.seed)
                self.stats.add("query", points=len(data["result"]))
                self._log({"path": path, "query": params.get("query"), "time": params.get("time")})
                return self._send(200, {"status": "success", "data": data})
        except BadRequest as exc:
            self.stats.add(path.rsplit("/", 1)[-1], error=True)
            return self._send(400, {"status": "error", "errorType": "bad_data", "error": str(exc)})
        if path == "/api/v1/status/buildinfo":
            self.stats.add("buildinfo")
            return self._send(200, {"status": "success", "data": {
                "version": VERSION, "revision": "stub", "branch": "stub", "buildUser": "loadlens", "goVersion": "go1.22",
            }})
        if path == "/api/v1/labels":
            self.stats.add("labels")
            return self._send(200, {"status": "success", "data": sorted({"__name__", *BASE_LABELS})})
        match = re.fullmatch(r"/api/v1/label/([^/]+)/values", path)
        if match:
            self.stats.add("label_values")
            return self._send(200, {"status": "success", "data": label_values(match.group(1), opts.series)})
        if path in ("/-/healthy", "/-/ready"):
            return self._send(200, {"status": "success"})
        if path == "/_stub/stats":
            return self._send(200, self.stats.snapshot())
        if path == "/_stub/reset" and self.command == "POST":
            self.stats.reset()
            return self._send(200, {"status": "reset"})
        return self._send(404, {"status": "error", "errorType": "not_found", "error": f"unknown path {path}"})

    def do_GET(self):  # noqa: N802 (http.server API)
        self._handle()

    def do_POST(self):  # noqa: N802 (http.server API)
        self._handle()

    def log_message(self, *args):
        pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prometheus stub for the LoadLens sandbox (see the module docstring).")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on: 127.0.0.1/localhost, or 0.0.0.0 inside a container")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--series", type=int, default=DEFAULT_SERIES, help="series per query")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="load steps across the window (0 or 1: flat)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delay-ms", type=float, default=0.0, help="extra latency of every query, ms")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="points per series above which a range query gets 400")
    parser.add_argument("--log", help="append one JSON line per query to this file")
    args = parser.parse_args(argv)
    if args.host not in LOCAL_HOSTS and args.host not in CONTAINER_HOSTS:
        parser.error("the stub listens on 127.0.0.1/localhost, or on 0.0.0.0 inside a container")
    if args.series < 0 or args.max_points < 1:
        parser.error("--series must not be negative, --max-points must be positive")
    return args


def make_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"options": args, "stats": Stats()})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    return server


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    server = make_server(args)
    print(f"prometheus stub on http://{args.host}:{args.port} (series={args.series}, steps={args.steps}, seed={args.seed}, delay={args.delay_ms} ms)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
