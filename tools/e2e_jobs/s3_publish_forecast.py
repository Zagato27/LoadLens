"""Scenario 3: publishing a capacity forecast to Confluence runs in the Celery worker.

How to run (sandbox prepared as in docs/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s3_publish_forecast.py --sandbox 6 --run-id <run_id>
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s3_publish_forecast.py --sandbox 6     # builds a report first

Starts the stubs (if they are not running), the server with LOADLENS_JOB_RUNNER=celery and a solo worker;
reads GET /forecast/<run_id>/status, clears the Confluence stub, then POST /forecast/<run_id>/confluence with
headroom 30 %. Checks that the job (kind forecast_confluence) has runner=celery and started_at in its row,
ends "done" with page_id, the worker log shows loadlens.publish_forecast[<job id>] received and succeeded,
the stub lists the page and GET /forecast/<run_id>/confluence returns it. Exit code 0 on pass.
"""

from __future__ import annotations

from urllib.parse import quote

from e2e_common import Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, seconds_between, stub_json

CHECK = "s3_publish_forecast"
CONFLUENCE_PORT = 5091


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="report to forecast (default: build one first)")
    parser.add_argument("--service", default="svc01")
    parser.add_argument("--headroom", type=float, default=30.0)
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
        run_id, built = args.run_id, None
        if not run_id:
            built = client.create_report(service=args.service)
            done = client.wait_job(built["job_id"], args.timeout)
            expect(results, "report built first", done.get("status") == "done", done.get("message"))
            run_id = str(done.get("page_url") or "").rsplit("/", 1)[-1]
        status = client.get(f"/forecast/{quote(run_id, safe='')}/status")
        evidence["forecast_status"] = {"http": status.status_code, "body": status.json() if status.ok else status.text[:300]}
        stub_json(CONFLUENCE_PORT, "/_stub/reset", "POST")
        response = client.post(f"/forecast/{quote(run_id, safe='')}/confluence", {"headroom_pct": args.headroom})
        body = response.json()
        job_id = body.get("job_id", "")
        expect(results, "publish accepted", response.status_code == 200 and bool(job_id), f"{response.status_code} {body}")
        final = client.wait_job(job_id, args.timeout)
        row = job_row(sb, job_id) or {}
        expect(results, "job done with a page", final.get("status") == "done" and bool(final.get("page_id")),
               {k: final.get(k) for k in ("status", "message", "error", "page_id", "page_url")})
        expect(results, "job row: celery, started by the worker",
               row.get("runner") == "celery" and bool(row.get("started_at")) and row.get("kind") == "forecast_confluence",
               {k: row.get(k) for k in ("kind", "status", "runner", "created_at", "started_at", "updated_at")})
        received = worker.log_lines(rf"loadlens\.publish_forecast\[{job_id}\] received")
        succeeded = worker.log_lines(rf"loadlens\.publish_forecast\[{job_id}\] succeeded")
        expect(results, "worker ran the task", bool(received) and bool(succeeded), received[:1] + succeeded[:1])
        pages = (stub_json(CONFLUENCE_PORT, "/_stub/pages") or {}).get("pages") or []
        page = next((p for p in pages if str(p.get("id")) == str(final.get("page_id"))), None)
        expect(results, "stub has the page", page is not None,
               {k: page.get(k) for k in ("id", "title", "version", "ancestors")} if page else {"pages": len(pages)})
        publication = client.get(f"/forecast/{quote(run_id, safe='')}/confluence").json()
        expect(results, "forecast publication recorded", str(publication.get("page_id")) == str(final.get("page_id")), publication)
        evidence.update(
            job_id=job_id, run_id=run_id, report_job=(built or {}).get("job_id"), row=row,
            job_sec=seconds_between(row.get("created_at"), row.get("updated_at")),
            queue_wait_sec=seconds_between(row.get("created_at"), row.get("started_at")), logs=str(stack.log_dir),
        )
        if built and not args.keep_report:
            evidence["deleted_report"] = client.delete_run(built["run_name"])
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
