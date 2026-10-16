"""Scenario 8: jobs wait in the queue for a free worker; a job no worker takes is marked after the limit.

How to run (sandbox prepared as in docs/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s8_queue_wait.py --sandbox 6

Part A, a busy worker. Server with LOADLENS_JOB_RUNNER=celery and one solo worker (one job at a time). Two
reports are started back to back. Checks that while the first runs, the second is "running" in /job_status
with started_at NULL in its row (it waits in the broker or as the worker's one prefetched message), then the
worker starts it after the first ends and it reaches "done".

Part B, no worker. The worker is stopped and the server restarted with LOADLENS_JOB_QUEUE_MAX_MIN=1. A report
is started; checks that it stays queued (started_at NULL) and within about 1.5 minutes becomes "error" with
«Задача прервана: воркер не взял её из очереди». Then a worker is started again: it receives the old message
and skips it ("Not running job ...") instead of building the report. All runs are deleted at the end.
Exit code 0 on pass.
"""

from __future__ import annotations

import time

from e2e_common import (
    Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, log, redis_queue_length, seconds_between, wait_until,
)

CHECK = "s8_queue_wait"
QUEUED_MESSAGE = "Задача прервана: воркер не взял её из очереди"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-wait", type=float, default=150, help="seconds until the queued job must be marked (part B)")
    args = parser.parse_args()

    sb = Sandbox(args.sandbox)
    results: dict = {}
    evidence: dict = {}
    with Stack(sb, CHECK, args.log_dir) as stack:
        stack.stubs(llm_delay=2.0)
        server = stack.server("celery")
        worker = stack.worker()
        client = Client(sb)
        runs: list[str] = []

        # ---- part A: the second job waits for the busy solo worker -----------------------------------------
        first = client.create_report()
        second = client.create_report()
        runs += [first["run_name"], second["run_name"]]
        samples = []
        while True:
            row1, row2 = job_row(sb, first["job_id"]) or {}, job_row(sb, second["job_id"]) or {}
            status2 = client.job(second["job_id"]) or {}
            samples.append({
                "first": (row1.get("status"), row1.get("progress")), "second_status_api": status2.get("status"),
                "second_started_at": row2.get("started_at"), "reports_queue_len": redis_queue_length(sb),
            })
            if row1.get("status") != "running":
                break
            time.sleep(3)
        waiting = [s for s in samples if s["first"][0] == "running"]
        expect(results, "second job waits (running, started_at NULL) while the first runs",
               bool(waiting) and all(s["second_status_api"] == "running" and s["second_started_at"] is None for s in waiting),
               {"samples_while_first_runs": len(waiting), "example": waiting[len(waiting) // 2] if waiting else None})
        final2 = client.wait_job(second["job_id"], args.timeout)
        row1, row2 = job_row(sb, first["job_id"]) or {}, job_row(sb, second["job_id"]) or {}
        expect(results, "second job starts after the first and finishes",
               final2.get("status") == "done" and row2.get("started_at") and row2["started_at"] >= row1["updated_at"],
               {"first_done_at": row1.get("updated_at"), "second_created_at": row2.get("created_at"),
                "second_started_at": row2.get("started_at"), "second_queue_wait_sec": seconds_between(row2.get("created_at"), row2.get("started_at")),
                "second_status": final2.get("status")})
        evidence["part_a"] = {"first": row1, "second": row2}

        # ---- part B: no worker, the queued job is given up after LOADLENS_JOB_QUEUE_MAX_MIN -------------------
        stack.stop(worker)
        stack.stop(server)
        stack.server("celery", LOADLENS_JOB_QUEUE_MAX_MIN=1)
        queued = client.create_report()
        runs.append(queued["run_name"])
        created_at = time.monotonic()
        time.sleep(5)
        row = job_row(sb, queued["job_id"]) or {}
        expect(results, "job queued without a worker", row.get("status") == "running" and row.get("started_at") is None and row.get("runner") == "celery",
               {"row": {k: row.get(k) for k in ("status", "runner", "started_at")}, "reports_queue_len": redis_queue_length(sb)})
        final = client.wait_job(queued["job_id"], args.max_wait + 30, poll=5)
        after = round(time.monotonic() - created_at, 1)
        expect(results, "queued job marked after the limit",
               final.get("status") == "error" and final.get("message") == QUEUED_MESSAGE and after <= args.max_wait,
               {"after_create_sec": after, "message": final.get("message"), "error": final.get("error")})
        again = stack.worker()
        skipped = wait_until(lambda: again.log_lines(rf"Not running job {queued['job_id']}"), 60, what="the worker to skip the stale message")
        expect(results, "a new worker skips the stale message", bool(skipped) and (job_row(sb, queued["job_id"]) or {}).get("status") == "error", skipped[:1])
        evidence["part_b"] = {"row": job_row(sb, queued["job_id"]), "worker_log": again.log_lines(rf"{queued['job_id']}")[:4]}
        evidence["logs"] = str(stack.log_dir)
        if not args.keep_report:
            evidence["deleted_runs"] = {name: client.delete_run(name) for name in runs}
    log("done")
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
