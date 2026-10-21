# SPDX-License-Identifier: Apache-2.0
"""Scenario 6: a restart of the web server does not interrupt a job the worker is building.

How to run (sandbox prepared as in plans/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s6_server_restart.py --sandbox 6

Starts the stubs (if they are not running), the server with LOADLENS_JOB_RUNNER=celery and a solo worker,
starts a report and, once the worker is in the AI phase (progress >= --kill-at), kills the server process
(TerminateProcess, like a crash or a container restart) and starts it again with the same environment.
Checks that the job row keeps changing while the server is down (the worker writes it), that after the
restart GET /job_status shows the job "running" and then "done" (not «Задача прервана: сервер
перезапустился…»: the restart recovery only touches runner=thread rows), and that the row has runner=celery.
The run is deleted at the end. Exit code 0 on pass.
"""

from __future__ import annotations

import time

from e2e_common import Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, log, seconds_between

CHECK = "s6_server_restart"
INTERRUPTED = "Задача прервана: сервер перезапустился во время выполнения"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--kill-at", type=int, default=47)
    parser.add_argument("--down-sec", type=float, default=15, help="seconds the server stays down")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    sb = Sandbox(args.sandbox)
    results: dict = {}
    evidence: dict = {}
    with Stack(sb, CHECK, args.log_dir) as stack:
        stack.stubs(llm_delay=2.0)
        server = stack.server("celery")
        worker = stack.worker()
        client = Client(sb)
        accepted = client.create_report()
        job_id, run_name = accepted["job_id"], accepted["run_name"]
        client.wait_job(job_id, 180, until=lambda j: j.get("status") != "running" or int(j.get("progress") or 0) >= args.kill_at)
        before = job_row(sb, job_id) or {}
        server.kill()
        down_at = time.monotonic()
        rows_while_down = []
        while time.monotonic() - down_at < args.down_sec:
            time.sleep(1)
            row = job_row(sb, job_id) or {}
            rows_while_down.append((row.get("progress"), row.get("message"), row.get("updated_at")))
        stack.server("celery")
        log(f"server restarted after {time.monotonic() - down_at:.1f} s")
        after_restart = client.job(job_id) or {}
        expect(results, "job still running after the restart", after_restart.get("status") in ("running", "done") and after_restart.get("message") != INTERRUPTED,
               {k: after_restart.get(k) for k in ("status", "progress", "message")})
        final = client.wait_job(job_id, args.timeout)
        row = job_row(sb, job_id) or {}
        moved = len({r[2] for r in rows_while_down} | {before.get("updated_at")}) > 1
        expect(results, "worker kept writing the row", moved,
               {"before_kill": {k: before.get(k) for k in ("progress", "message", "updated_at")}, "while_down": rows_while_down[-2:]})
        expect(results, "job done, not marked interrupted", final.get("status") == "done" and row.get("runner") == "celery",
               {k: row.get(k) for k in ("status", "runner", "message", "created_at", "started_at", "updated_at")})
        succeeded = worker.log_lines(rf"loadlens\.report\[{job_id}\] succeeded")
        expect(results, "worker finished the task", bool(succeeded), succeeded[:1])
        evidence.update(job_id=job_id, run_name=run_name, row=row, job_sec=seconds_between(row.get("created_at"), row.get("updated_at")),
                        logs=str(stack.log_dir))
        if not args.keep_report:
            evidence["deleted_run"] = client.delete_run(run_name)
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
