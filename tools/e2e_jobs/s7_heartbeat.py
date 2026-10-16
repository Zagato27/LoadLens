"""Scenario 7: the heartbeat keeps a job with long silent LLM calls alive (LOADLENS_JOB_STALE_MIN=1).

How to run (sandbox prepared as in docs/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N
and no LLM stub on 5090: this check starts its own with a slow model):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s7_heartbeat.py --sandbox 6 --llm-delay 40

Starts the stubs (the LLM stub answers in --llm-delay seconds), the server with LOADLENS_JOB_RUNNER=celery
and LOADLENS_JOB_STALE_MIN=1, and a solo worker; starts a report and, until it ends, reads its row every 2 s
and polls GET /job_status every 5 s and /jobs every 10 s (both run the sweep, like open pages). Checks that

* there was a stretch without any progress change longer than the 1-minute threshold (the check means
  something), and still the job was never swept;
* updated_at moved without a progress change (heartbeats) and never stood still for 60 s or more;
* the job ends "done".

With 28 LLM calls, two at a time, the job takes about 14 x --llm-delay seconds (about 10 minutes at 40 s).
The run is deleted at the end. Exit code 0 on pass.
"""

from __future__ import annotations

import time
from datetime import datetime

from e2e_common import Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, log

CHECK = "s7_heartbeat"
STALE_SEC = 60.0


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--llm-delay", type=float, default=40.0)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()

    sb = Sandbox(args.sandbox)
    results: dict = {}
    evidence: dict = {}
    with Stack(sb, CHECK, args.log_dir) as stack:
        stack.stubs(llm_delay=args.llm_delay)
        stack.server("celery", LOADLENS_JOB_STALE_MIN=1)
        worker = stack.worker()
        client = Client(sb)
        accepted = client.create_report()
        job_id, run_name = accepted["job_id"], accepted["run_name"]
        began = time.monotonic()
        timeline: list[tuple[str, int, str]] = []  # (updated_at, progress, message) as seen in the row
        last_poll = last_list = 0.0
        row: dict = {}
        while time.monotonic() - began < args.timeout:
            row = job_row(sb, job_id) or {}
            entry = (row.get("updated_at"), row.get("progress"), row.get("message"))
            if not timeline or timeline[-1] != entry:
                timeline.append(entry)
                kind = "progress" if len(timeline) == 1 or timeline[-2][1:] != entry[1:] else "heartbeat"
                log(f"  row {kind}: {entry[0]} {entry[1]}% {entry[2]}")
            if row.get("status") != "running":
                break
            now = time.monotonic()
            if now - last_poll >= 5:
                client.job(job_id)
                last_poll = now
            if now - last_list >= 10:
                client.get("/jobs?status=running,error&limit=10")
                last_list = now
            time.sleep(2)

        stamps = [datetime.fromisoformat(u) for u, _p, _m in timeline if u]
        update_gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        progress_points = [datetime.fromisoformat(u) for i, (u, p, m) in enumerate(timeline) if i == 0 or timeline[i - 1][1:] != (p, m)]
        progress_points.append(stamps[-1] if stamps else None)
        progress_gaps = [(b - a).total_seconds() for a, b in zip(progress_points, progress_points[1:]) if a and b]
        heartbeats = sum(1 for i in range(1, len(timeline)) if timeline[i][1:] == timeline[i - 1][1:])
        expect(results, "silent stretch longer than the threshold", max(progress_gaps or [0]) > STALE_SEC,
               f"longest stretch without progress {max(progress_gaps or [0]):.0f} s (threshold {STALE_SEC:.0f} s)")
        expect(results, "heartbeats kept updated_at fresh", heartbeats >= 2 and max(update_gaps or [STALE_SEC]) < STALE_SEC,
               f"{heartbeats} heartbeat-only updates, longest gap between updates {max(update_gaps or [0]):.1f} s")
        expect(results, "job done, never swept", row.get("status") == "done",
               {k: row.get(k) for k in ("status", "runner", "message", "created_at", "started_at", "updated_at")})
        expect(results, "worker heartbeat errors", not worker.log_lines(r"Heartbeat of job .* failed"), "none in the worker log")
        evidence.update(job_id=job_id, run_name=run_name, llm_delay=args.llm_delay, timeline=timeline,
                        progress_gaps_sec=[round(g, 1) for g in progress_gaps], update_gaps_sec=[round(g, 1) for g in update_gaps],
                        logs=str(stack.log_dir))
        if not args.keep_report:
            evidence["deleted_run"] = client.delete_run(run_name)
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
