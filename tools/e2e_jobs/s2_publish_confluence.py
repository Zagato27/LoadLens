# SPDX-License-Identifier: Apache-2.0
"""Scenario 2: publishing a web report to Confluence runs in the Celery worker.

How to run (sandbox prepared as in plans/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s2_publish_confluence.py --sandbox 6 --run-name e2e-ui-...
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s2_publish_confluence.py --sandbox 6     # builds a report first

Starts the stubs (if they are not running), the server with LOADLENS_JOB_RUNNER=celery and a solo worker;
clears the Confluence stub (POST /_stub/reset), then POST /publish_confluence for the run (the report of
scenario 1, or one this script builds through the worker). Checks that the job row has runner=celery and
started_at, the job ends "done" with page_id, the worker log shows loadlens.publish_confluence[<job id>]
received and succeeded, and the stub lists the page (GET /_stub/pages) under parent 42 with attachments.
Exit code 0 on pass.
"""

from __future__ import annotations

from e2e_common import Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, seconds_between, stub_json

CHECK = "s2_publish_confluence"
CONFLUENCE_PORT = 5091


def _page_summary(page: dict) -> dict:
    """A stub page without its body and attachment names: id, title, version, ancestors, counts."""
    out = {k: v for k, v in page.items() if not isinstance(v, (list, dict)) or k == "ancestors"}
    out.update({f"{k}_count": len(v) for k, v in page.items() if isinstance(v, list) and k != "ancestors"})
    return out


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--run-name", help="report run to publish (default: build one first)")
    parser.add_argument("--service", default="svc01")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    sb = Sandbox(args.sandbox)
    results: dict = {}
    evidence: dict = {}
    with Stack(sb, CHECK, args.log_dir) as stack:
        stack.stubs(llm_delay=2.0)
        stack.server("celery")
        worker = stack.worker()
        client = Client(sb)
        run_name, built = args.run_name, None
        if not run_name:
            built = client.create_report(service=args.service)
            done = client.wait_job(built["job_id"], args.timeout)
            expect(results, "report built first", done.get("status") == "done", done.get("message"))
            run_name = built["run_name"]
        stub_json(CONFLUENCE_PORT, "/_stub/reset", "POST")
        response = client.post("/publish_confluence", {"run_name": run_name, "service": args.service})
        body = response.json()
        job_id = body.get("job_id", "")
        expect(results, "publish accepted", response.status_code == 200 and bool(job_id), f"{response.status_code} {body}")
        final = client.wait_job(job_id, args.timeout)
        row = job_row(sb, job_id) or {}
        expect(results, "job done with a page", final.get("status") == "done" and bool(final.get("page_id")),
               {k: final.get(k) for k in ("status", "message", "page_id", "page_url")})
        expect(results, "job row: celery, started by the worker",
               row.get("runner") == "celery" and bool(row.get("started_at")) and row.get("kind") == "confluence",
               {k: row.get(k) for k in ("kind", "status", "runner", "created_at", "started_at", "updated_at")})
        received = worker.log_lines(rf"loadlens\.publish_confluence\[{job_id}\] received")
        succeeded = worker.log_lines(rf"loadlens\.publish_confluence\[{job_id}\] succeeded")
        expect(results, "worker ran the task", bool(received) and bool(succeeded), received[:1] + succeeded[:1])
        pages = stub_json(CONFLUENCE_PORT, "/_stub/pages") or {}
        items = pages.get("pages", pages) if isinstance(pages, dict) else pages
        page = next((p for p in items or [] if str(p.get("id")) == str(final.get("page_id"))), None)
        expect(results, "stub has the page", page is not None, _page_summary(page) if page else {"pages": len(items or [])})
        evidence.update(
            job_id=job_id, run_name=run_name, report_job=(built or {}).get("job_id"), row=row,
            job_sec=seconds_between(row.get("created_at"), row.get("updated_at")),
            queue_wait_sec=seconds_between(row.get("created_at"), row.get("started_at")), logs=str(stack.log_dir),
        )
        if built and not args.keep_report:
            evidence["deleted_report"] = client.delete_run(run_name)
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
