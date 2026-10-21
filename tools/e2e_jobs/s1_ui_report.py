# SPDX-License-Identifier: Apache-2.0
"""Scenario 1: a report started from the "Новый отчёт" page is built by the Celery worker.

How to run (sandbox prepared as in plans/perf-baseline.md, "Этап 3"; nothing listening on 127.0.0.1:5070+N):

    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s1_ui_report.py --sandbox 6

Starts the stubs (if they are not running), the sandbox server with LOADLENS_JOB_RUNNER=celery and a solo
worker, then drives headless Chromium (Playwright): area "bench", service svc01, "Последний час", test type
"step", AI analysis on, web report. Checks that

* POST /create_report answers at once and the job row has runner=celery and gets started_at from the worker;
* the progress card shows several intermediate percentages while the job runs (progress written by the
  worker reaches the page) and ends with "Отчёт готов.";
* the worker log has "Task loadlens.report[<job id>] received" and "... succeeded";
* "Открыть отчёт" opens /reports/<run_id> with the run name in the title; no console errors except
  GET /auth/users 503 (sign-in is off in the sandbox).

The report run is kept (scenarios 2 and 3 publish it); its name and run_id are printed. Exit code 0 on pass.
"""

from __future__ import annotations

import time

from e2e_common import Client, Sandbox, Stack, all_ok, base_parser, expect, finish, job_row, log, seconds_between

CHECK = "s1_ui_report"


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for the job")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    sb = Sandbox(args.sandbox)
    results: dict = {}
    evidence: dict = {}
    with Stack(sb, CHECK, args.log_dir) as stack:
        stack.stubs(llm_delay=2.0)
        server = stack.server("celery")
        worker = stack.worker()
        console_errors: list[str] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            context = browser.new_context(locale="ru-RU")
            context.add_cookies([{"name": "project_area", "value": "bench", "domain": "127.0.0.1", "path": "/"}])
            page = context.new_page()
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.goto(sb.base + "/new", wait_until="domcontentloaded")
            service_input = page.locator("#serviceInput")
            for _ in range(60):
                if service_input.is_enabled():
                    break
                time.sleep(0.5)
            service_input.click()
            service_input.fill("svc01")
            time.sleep(0.6)  # the combobox filters after a debounce
            page.locator("[role=option]", has_text="svc01").first.click()
            time.sleep(0.3)
            if page.locator("#serviceCombo.open").count():
                service_input.press("Escape")
            chosen = page.locator("#service").input_value()
            expect(results, "service picked", chosen == "svc01", chosen)
            page.locator('#rangePresets .seg-btn[data-minutes="60"]').click()
            page.locator("label.type-card", has=page.locator('input[name="test_type"][value="step"]')).click()
            run_name = time.strftime("e2e-ui-%Y%m%d-%H%M%S")
            page.locator("#run_name").fill(run_name)
            with page.expect_response(lambda r: r.url.endswith("/create_report")) as created:
                began = time.perf_counter()
                page.locator("#createBtn").click()
            response = created.value
            request_sec = round(time.perf_counter() - began, 2)
            accepted = response.json()
            job_id = accepted.get("job_id", "")
            expect(results, "create_report accepted", response.status == 200 and bool(job_id), f"{response.status}, {request_sec} s, job {job_id}")

            seen: list[tuple[float, str, str]] = []
            started = time.perf_counter()
            final_message = ""
            while time.perf_counter() - started < args.timeout:
                pct = page.locator("#progressPct").inner_text()
                message = page.locator("#progressMessage").inner_text()
                if not seen or seen[-1][1:] != (pct, message):
                    seen.append((round(time.perf_counter() - started, 1), pct, message))
                    log(f"  page: {pct} {message}")
                if message in ("Отчёт готов.",) or page.locator("#progressBarFill.error").count():
                    final_message = message
                    break
                time.sleep(1.0)
            percents = sorted({int(p.rstrip("%")) for _t, p, _m in seen if p.rstrip("%").isdigit()})
            intermediate = [p for p in percents if 0 < p < 100]
            expect(results, "page shows progress", len(intermediate) >= 3, f"percentages seen {percents}")
            expect(results, "page shows done", final_message == "Отчёт готов.", final_message or seen[-1:])

            row = job_row(sb, job_id) or {}
            expect(results, "job row: celery, started by the worker, done",
                   row.get("runner") == "celery" and row.get("started_at") and row.get("status") == "done",
                   {k: row.get(k) for k in ("status", "runner", "created_at", "started_at", "updated_at", "page_url")})
            received = worker.log_lines(rf"loadlens\.report\[{job_id}\] received")
            succeeded = worker.log_lines(rf"loadlens\.report\[{job_id}\] succeeded")
            expect(results, "worker ran the task", bool(received) and bool(succeeded), received[:1] + succeeded[:1])
            not_in_server = not server.log_lines(rf"Could not send job {job_id}")
            expect(results, "no fallback to a thread", not_in_server, "no 'Could not send job' in the server log")

            web_link = page.locator("#webLink")
            href = web_link.get_attribute("href") or ""
            report = context.new_page()
            report.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            opened = report.goto(sb.base + href, wait_until="domcontentloaded") if href.startswith("/reports/") else None
            title = ""
            for _ in range(40):
                title = report.locator("#pageTitle").inner_text()
                if run_name in title:
                    break
                time.sleep(0.5)
            time_range = report.locator("#pageTimeRange").inner_text()
            expect(results, "report opens", opened is not None and opened.status == 200 and run_name in title,
                   {"href": href, "status": opened.status if opened else None, "title": title, "range": time_range})
            browser.close()
        noise = [e for e in console_errors if "auth/users" not in e and "503" not in e]
        expect(results, "no console errors", not noise, console_errors[:5])
        evidence.update(
            job_id=job_id, run_name=run_name, run_id=href.rsplit("/", 1)[-1] if href else None,
            create_report_sec=request_sec, queue_wait_sec=seconds_between(row.get("created_at"), row.get("started_at")),
            job_sec=seconds_between(row.get("created_at"), row.get("updated_at")), page_progress=seen, row=row,
            logs=str(stack.log_dir),
        )
    return finish(CHECK, all_ok(results), {"results": results, **evidence}, args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
