"""Scenario 5: with the broker down a new job runs in a thread of the web process; after it is back, in the worker.

How to run (sandbox prepared as in docs/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N;
Redis is the local throwaway container of the sandboxes, by default ll-perf-redis):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s5_redis_down.py --sandbox 6 --redis-container ll-perf-redis

Starts the stubs (if they are not running), the server with LOADLENS_JOB_RUNNER=celery and a solo worker,
then ``docker stop <container>``. Checks that

* POST /create_report still answers (after the broker timeouts, a few seconds), the job row has runner=thread,
  the server log has "Could not send job <id> ... running it in a thread of the web process", and the job
  ends "done";

then ``docker start <container>`` (also when the check fails) and checks that the next report goes to the
worker again: row runner=celery with started_at, "loadlens.report[<id>] received" in the worker log, "done".
Only the container named by --redis-container is stopped and started. Both runs are deleted at the end.
Exit code 0 on pass.
"""

from __future__ import annotations

import subprocess
import time

from e2e_common import Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, log, redis_ping, seconds_between, wait_until

CHECK = "s5_redis_down"


def docker(action: str, container: str) -> str:
    out = subprocess.run(["docker", action, container], capture_output=True, text=True, timeout=120)
    log(f"docker {action} {container}: {out.returncode} {out.stdout.strip()} {out.stderr.strip()}")
    return out.stdout.strip()


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--redis-container", default="ll-perf-redis", help="local Redis container to stop and start")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    sb = Sandbox(args.sandbox)
    results: dict = {}
    evidence: dict = {}
    if not redis_ping():
        raise SystemExit("Redis on 127.0.0.1:16379 does not answer: start it first")
    with Stack(sb, CHECK, args.log_dir) as stack:
        stack.stubs(llm_delay=2.0)
        server = stack.server("celery")
        worker = stack.worker()
        client = Client(sb)
        client.refresh_csrf()
        runs = []
        try:
            docker("stop", args.redis_container)
            wait_until(lambda: not redis_ping(), 30, what="Redis to stop")
            accepted = client.create_report()
            runs.append(accepted["run_name"])
            job_id = accepted["job_id"]
            final = client.wait_job(job_id, args.timeout)
            row = job_row(sb, job_id) or {}
            warning = server.log_lines(rf"Could not send job {job_id}")
            expect(results, "create_report answers without the broker", accepted["_request_sec"] < 15, f"{accepted['_request_sec']} s")
            expect(results, "warning in the server log", bool(warning), warning[:1])
            expect(results, "job ran in a thread and finished",
                   row.get("runner") == "thread" and final.get("status") == "done",
                   {k: row.get(k) for k in ("status", "runner", "created_at", "started_at", "updated_at", "message")})
            expect(results, "worker did not get it", not worker.log_lines(rf"loadlens\.report\[{job_id}\]"), "no delivery in the worker log")
            evidence["thread_job"] = {"job_id": job_id, "row": row, "request_sec": accepted["_request_sec"],
                                      "job_sec": seconds_between(row.get("created_at"), row.get("updated_at"))}
        finally:
            docker("start", args.redis_container)
            wait_until(redis_ping, 60, what="Redis to answer again")
        # The worker reconnects by itself; give it a moment before the next job.
        try:
            wait_until(lambda: len(worker.log_lines(r"Connected to redis")) >= 2, 60, what="the worker to reconnect")
        except TimeoutError:
            log("the worker log shows no reconnect yet; trying anyway")
        accepted = client.create_report()
        runs.append(accepted["run_name"])
        job_id = accepted["job_id"]
        final = client.wait_job(job_id, args.timeout)
        row = job_row(sb, job_id) or {}
        received = worker.log_lines(rf"loadlens\.report\[{job_id}\] received")
        expect(results, "after Redis is back the worker builds the report",
               row.get("runner") == "celery" and bool(row.get("started_at")) and final.get("status") == "done" and bool(received),
               {k: row.get(k) for k in ("status", "runner", "created_at", "started_at", "updated_at")})
        evidence["celery_job"] = {"job_id": job_id, "row": row, "request_sec": accepted["_request_sec"], "worker_log": received[:1],
                                  "job_sec": seconds_between(row.get("created_at"), row.get("updated_at"))}
        evidence["worker_reconnect"] = worker.log_lines(r"Connected to redis|consumer: Cannot connect|connection to broker lost")[:6]
        evidence["logs"] = str(stack.log_dir)
        if not args.keep_report:
            evidence["deleted_runs"] = {name: client.delete_run(name) for name in runs}
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
