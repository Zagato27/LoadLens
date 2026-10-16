"""Scenario 9: endpoint latency while a report is built, in a thread of the web process versus in the worker.

How to run (sandbox prepared as in docs/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/bench_modes.py --sandbox 6 --repeats 2
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/bench_modes.py --sandbox 6 --repeats 2 --concurrency 4 --rounds 10
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/bench_modes.py --summarize %TEMP%/ll_e2e_jobs/bench-c1

For every repeat it runs ``tools/bench_endpoints.py --during-job`` twice, back to back: ``thread`` (server with
LOADLENS_JOB_RUNNER=thread, no worker) and ``celery`` (server with LOADLENS_JOB_RUNNER=celery plus a solo
worker), each with a freshly started server; the stubs stay up (LLM --delay 2, Prometheus defaults: 6 series,
4 steps), as in the thread baseline of docs/perf-baseline.md. Between runs it pauses --pause-sec seconds and
prints the TIME_WAIT count (the Werkzeug server closes every connection; ephemeral ports run short). The bench
connects with blocking sockets (see BLOCKING_CONNECT; --timeout-connect runs it unchanged).

The JSON of every run goes to --out (default %TEMP%/ll_e2e_jobs/bench-c<C>); the summary (also written as
summary.md there) has, per mode, the median over the repeats of idle and during-job p50/p95, of the
collection phase p50/p95 and of the job duration, and checks the criteria of docs/perf-plan.md 11.7: in
celery mode job p95 <= 1.3 x idle p95 for /dashboard_data, /runs, /jobs and /load_tests/active, and the job
in the worker at most 10 % slower than in a thread. Exit code 0 when every run finished its job.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from e2e_common import REPO, Sandbox, Stack, log, temp_dir

MODES = ("thread", "celery")
CRITERIA_PATHS = (
    "/dashboard_data",
    "/runs?limit=20",
    "/runs?limit=20&sort=max_rps",
    "/load_tests/active?limit=5&live=1",
    "/jobs?status=running&limit=10",
)
P95_LIMIT = 1.3
DURATION_LIMIT = 1.10


def time_wait_count() -> int:
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, errors="replace").stdout
    return sum(1 for line in out.splitlines() if "TIME_WAIT" in line)


# Runs tools/bench_endpoints.py with blocking TCP connects. Python connects a socket that has a timeout
# without blocking and then waits in select(); on a Windows host with thousands of open connections that
# wait alone took 15-25 ms on loopback (a blocking connect: 2 ms), which hid the server-side differences.
BLOCKING_CONNECT = """
import runpy, socket, sys
_create = socket.create_connection
def _blocking(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):
    sock = _create(address, None, source_address)
    if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
        sock.settimeout(timeout)
    return sock
socket.create_connection = _blocking
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_bench(sb: Sandbox, mode: str, out: Path, args: argparse.Namespace) -> int:
    bench = [
        str(REPO / "tools" / "bench_endpoints.py"), "--sandbox", str(sb.n), "--during-job",
        "--label", mode, "--json", str(out), "--concurrency", str(args.concurrency), "--rounds", str(args.rounds),
    ]
    cmd = [sys.executable, "-X", "utf8", *(["-c", BLOCKING_CONNECT] if not args.timeout_connect else []), *bench]
    log("bench: " + " ".join(bench) + (" (timeout connect)" if args.timeout_connect else " (blocking connect)"))
    with open(out.with_suffix(".txt"), "w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT).returncode


def _median(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return round(a / b, 2) if a is not None and b else None


def _fmt(value, suffix: str = "") -> str:
    return "—" if value is None else f"{value:g}{suffix}"


def summarize(out_dir: Path) -> tuple[str, bool]:
    """Markdown summary of the runs in ``out_dir`` and whether the 11.7 criteria hold."""
    runs: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for path in sorted(out_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        mode = (data.get("meta") or {}).get("label")
        if mode in runs and (data.get("job") or {}).get("status") == "done":
            data["_file"] = path.name
            runs[mode].append(data)
    lines = [f"Прогонов: thread {len(runs['thread'])}, celery {len(runs['celery'])} (только задачи `done`); медиана по повторам, мс.", ""]
    head = "| Эндпоинт | Режим | idle p50 | idle p95 | job p50 | job p95 | job/idle p95 | сбор p50 | сбор p95 | сбор/idle p95 | job/idle p95 по повторам |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    paths = [row["path"] for row in (runs["thread"] or runs["celery"] or [{"rows": []}])[0]["rows"]]
    verdicts = []
    for path in paths:
        for mode in MODES:
            rows = [next((r for r in data["rows"] if r["path"] == path), {}) for data in runs[mode]]
            collect = [next(iter((data.get("per_phase") or {}).values()), {}).get(path, {}) for data in runs[mode]]
            idle50, idle95 = _median(r.get("idle_p50") for r in rows), _median(r.get("idle_p95") for r in rows)
            job50, job95 = _median(r.get("job_p50") for r in rows), _median(r.get("job_p95") for r in rows)
            c50, c95 = _median(c.get("p50") for c in collect), _median(c.get("p95") for c in collect)
            run_ratios = [_ratio(r.get("job_p95"), r.get("idle_p95")) for r in rows]
            per_run = ", ".join(_fmt(value, "×") for value in run_ratios)
            lines.append(
                f"| `{path}` | {mode} | {_fmt(idle50)} | {_fmt(idle95)} | {_fmt(job50)} | {_fmt(job95)} | {_fmt(_ratio(job95, idle95), '×')} "
                f"| {_fmt(c50)} | {_fmt(c95)} | {_fmt(_ratio(c95, idle95), '×')} | {per_run} |"
            )
            if mode == "celery" and path in CRITERIA_PATHS:
                # Each run compares the job with the idle rounds of the same server minutes later or earlier;
                # the median over the runs keeps one noisy idle round from deciding.
                ratio = _median(run_ratios)
                verdicts.append((path, ratio, ratio is not None and ratio <= P95_LIMIT))
    lines.append("")
    durations = {}
    for mode in MODES:
        jobs = [data["job"] for data in runs[mode]]
        durations[mode] = _median(j.get("duration_sec") for j in jobs)
        phases = {}
        for job in jobs:
            for name, seconds in job.get("phases") or []:
                phases.setdefault(name, []).append(seconds)
        phase_text = " · ".join(f"{name} {_fmt(_median(values))} с" for name, values in phases.items())
        each = ", ".join(_fmt(j.get("duration_sec")) for j in jobs)
        drift = ", ".join(_fmt(j.get("idle_drift"), "×") for j in jobs)
        lost = sum(int(j.get("no_answer") or 0) for j in jobs)
        lines.append(f"- {mode}: задача {_fmt(durations[mode])} с (по повторам {each}); фазы {phase_text}; idle после/до {drift}; без ответа {lost}")
    duration_ratio = _ratio(durations.get("celery"), durations.get("thread"))
    duration_ok = duration_ratio is not None and duration_ratio <= DURATION_LIMIT
    lines.append("")
    lines.append(f"Критерии 11.7 (режим celery, медиана по прогонам job p95 / idle p95 ≤ {P95_LIMIT}):")
    for path, ratio, ok in verdicts:
        lines.append(f"- `{path}`: {_fmt(ratio, '×')} — {'да' if ok else 'нет'}")
    lines.append(f"- длительность задачи celery / thread: {_fmt(duration_ratio, '×')} (≤ {DURATION_LIMIT}×) — {'да' if duration_ok else 'нет'}")
    return "\n".join(lines), all(ok for _p, _r, ok in verdicts) and duration_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sandbox", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--modes", default="thread,celery", help="comma-separated, run in this order every repeat")
    parser.add_argument("--pause-sec", type=float, default=90, help="pause between runs (TIME_WAIT lasts 2 minutes)")
    parser.add_argument("--timeout-connect", action="store_true",
                        help="let the bench connect as it does by itself (non-blocking connect + select, see BLOCKING_CONNECT)")
    parser.add_argument("--out", help="directory for the JSON of the runs")
    parser.add_argument("--summarize", metavar="DIR", help="only print the summary of the runs in DIR")
    args = parser.parse_args()
    if args.summarize:
        text, ok = summarize(Path(args.summarize))
        print(text)
        return 0

    sb = Sandbox(args.sandbox)
    out_dir = Path(args.out) if args.out else temp_dir() / "ll_e2e_jobs" / f"bench-c{args.concurrency}"
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    failures = 0
    with Stack(sb, f"bench-c{args.concurrency}", str(out_dir / "logs")) as stack:
        stack.stubs(llm_delay=2.0)
        first = True
        for repeat in range(1, args.repeats + 1):
            for mode in modes:
                if not first:
                    log(f"pause {args.pause_sec:.0f} s; TIME_WAIT now {time_wait_count()}")
                    time.sleep(args.pause_sec)
                first = False
                log(f"repeat {repeat}, mode {mode}: TIME_WAIT {time_wait_count()}")
                server = stack.server(mode)
                worker = stack.worker() if mode == "celery" else None
                stamp = time.strftime("%H%M%S")
                code = run_bench(sb, mode, out_dir / f"{mode}-c{args.concurrency}-r{repeat}-{stamp}.json", args)
                failures += code != 0
                log(f"bench exit code {code}")
                if worker is not None:
                    stack.stop(worker)
                stack.stop(server)
    text, ok = summarize(out_dir)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    log(f"criteria of 11.7 {'met' if ok else 'NOT met'}; runs in {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
