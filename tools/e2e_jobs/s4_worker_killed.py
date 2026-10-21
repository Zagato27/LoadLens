# SPDX-License-Identifier: Apache-2.0
"""Scenario 4: the worker dies in the middle of a report; the job becomes an error, the page stops polling.

How to run (sandbox prepared as in plans/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s4_worker_killed.py --sandbox 6

Starts the stubs (if they are not running), the server with LOADLENS_JOB_RUNNER=celery and
LOADLENS_JOB_STALE_MIN=1, and a solo worker. Starts a report, waits until the worker reports the AI phase
(progress >= --kill-at), opens /new?job=<id> in headless Chromium and kills the worker process
(TerminateProcess of its python.exe: no warm shutdown, no final status). Checks that

* within --max-wait seconds (default 150) GET /job_status says "error" with «Задача прервана: воркер не
  отвечает» (sweep: updated_at older than 1 min, at most every 30 s);
* the page shows the error and sends no /job_status request for it during the next 15 s;
* /jobs?status=running no longer lists the job;
* a worker started again does not receive the killed task (task_acks_late=False: no replay).

The partial run is deleted at the end. Exit code 0 on pass.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from e2e_common import Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, log

CHECK = "s4_worker_killed"
STALE_MESSAGE = "Задача прервана: воркер не отвечает"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--kill-at", type=int, default=47, help="kill the worker once the job progress reaches this")
    parser.add_argument("--max-wait", type=float, default=150, help="seconds from the kill to the error")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    sb = Sandbox(args.sandbox)
    results: dict = {}
    evidence: dict = {}
    with Stack(sb, CHECK, args.log_dir) as stack:
        stack.stubs(llm_delay=2.0)
        server = stack.server("celery", LOADLENS_JOB_STALE_MIN=1)
        worker = stack.worker()
        client = Client(sb)
        accepted = client.create_report()
        job_id, run_name = accepted["job_id"], accepted["run_name"]
        client.wait_job(job_id, 180, until=lambda j: j.get("status") != "running" or int(j.get("progress") or 0) >= args.kill_at)
        before = job_row(sb, job_id) or {}
        expect(results, "job runs in the worker", before.get("runner") == "celery" and bool(before.get("started_at")) and before.get("status") == "running",
               {k: before.get(k) for k in ("status", "runner", "progress", "message", "started_at", "updated_at")})

        requests_seen: list[float] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(locale="ru-RU")
            context.add_cookies([{"name": "project_area", "value": "bench", "domain": "127.0.0.1", "path": "/"}])
            page = context.new_page()
            # Playwright's sync API delivers events only inside its own calls: wait with page.wait_for_timeout,
            # never time.sleep, so that the request times below are real.
            page.on("request", lambda r: requests_seen.append(time.monotonic()) if f"/job_status/{job_id}" in r.url else None)
            opened_at = time.monotonic()
            page.goto(f"{sb.base}/new?job={job_id}", wait_until="domcontentloaded")
            while len(requests_seen) < 2 and time.monotonic() - opened_at < 30:
                page.wait_for_timeout(250)
            expect(results, "page polls the running job", len(requests_seen) >= 2,
                   f"{len(requests_seen)} request(s) in {time.monotonic() - opened_at:.1f} s")

            killed_at = time.monotonic()
            killed_wall = datetime.now(timezone.utc)
            worker_pid = worker.pid
            worker.kill()
            evidence["killed"] = {"pid": worker_pid, "at": killed_wall.isoformat(timespec="seconds"), "row_updated_at": before.get("updated_at")}

            # Only the page polls now (GET /job_status, as a user watching the job): its polls run the sweep.
            shown_at = None
            while shown_at is None and time.monotonic() - killed_at < args.max_wait + 30:
                page.wait_for_timeout(1000)
                if STALE_MESSAGE in page.locator("#progressMessage").inner_text():
                    shown_at = time.monotonic()
            final = client.job(job_id) or {}
            error_after = round((shown_at or time.monotonic()) - killed_at, 1)
            log(f"  page shows the error {error_after} s after the kill")
            expect(results, "job swept as 'worker does not answer'",
                   final.get("status") == "error" and final.get("message") == STALE_MESSAGE and error_after <= args.max_wait,
                   {"after_kill_sec": error_after, "status": final.get("status"), "message": final.get("message"), "error": final.get("error")})

            polls_before = len(requests_seen)
            page.wait_for_timeout(15000)
            later = [t for t in requests_seen if shown_at is not None and t > shown_at + 0.5]
            expect(results, "page stopped polling", shown_at is not None and not later,
                   {"page_message": page.locator("#progressMessage").inner_text(), "polls_after_error_in_15_s": len(later),
                    "polls_until_error": polls_before})
            browser.close()

        running = client.get("/jobs?status=running&limit=20").json().get("jobs", [])
        expect(results, "not listed as running", all(j.get("job_id") != job_id for j in running), f"{len(running)} running job(s)")
        sweep_lines = server.log_lines(r"abandoned by their worker")
        evidence["server_log"] = sweep_lines[:3]

        again = stack.worker()
        time.sleep(15)
        replayed = again.log_lines(rf"loadlens\.report\[{job_id}\]")
        expect(results, "killed task not replayed", not replayed, replayed[:2] or "worker restarted, no delivery of the killed task in 15 s")
        evidence.update(job_id=job_id, run_name=run_name, row=job_row(sb, job_id), logs=str(stack.log_dir))
        if not args.keep_report:
            evidence["deleted_run"] = client.delete_run(run_name)
    log(f"evidence: {evidence.get('killed')}")
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
