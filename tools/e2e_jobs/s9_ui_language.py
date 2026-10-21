# SPDX-License-Identifier: Apache-2.0
"""Scenario 9: the interface, the reports and the publications in Russian and English (plans/i18n-plan.md, 6.4).

Two sandboxes on 127.0.0.1, both with settings generated from their own settings.example.py (every URL
points at 127.0.0.1:9, the storage at the local TimescaleDB):

* NEW, sandbox 9 (port 5079, database ll_s9): a copy of the working tree (``git ls-files -co
  --exclude-standard``: ignored secrets are never copied), stubs LLM 5090, Confluence 5091, Prometheus 5092;
* OLD, sandbox 19 (port 5089, database ll_s19): ``git archive <--old-ref, default HEAD>``, its own copies of the
  stubs on 5093, 5094 and 5095. OLD only serves the Russian reference rendering.

How to run (Docker image timescale/timescaledb:latest-pg16; nothing listening on 5079/5089/5090-5095):

    docker run -d --name ll-i18n-e2e-db -p 127.0.0.1:55432:5432 -e POSTGRES_USER=ltar ^
        -e POSTGRES_PASSWORD=e2e-only-password timescale/timescaledb:latest-pg16
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s9_ui_language.py --prepare     # sandboxes and OLD data, ~1 min
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s9_ui_language.py               # everything, ~25 min
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s9_ui_language.py --pages dashboard,archive --no-matrix
    C:/Python312/python.exe -X utf8 tools/e2e_jobs/s9_ui_language.py --modes auth,down,down_auth
    docker rm -f ll-i18n-e2e-db                                                     # when done

``--prepare`` builds both trees, seeds OLD (tools/seed_bench_data.py of OLD: 24 runs of the "bench" area,
services svc01..svc03; SLA thresholds and a manual load generator in its runtime file), adds the demo run and
one Russian report built by the job pipeline with the LLM stub, then clones the OLD database into the NEW one
(``CREATE DATABASE ll_s9 TEMPLATE ll_s19``): NEW starts from data written by the old code, as after an upgrade,
and runs its start-up migrations on it. Re-run ``--prepare`` after code changes (it drops both databases).

The check runs NEW and OLD in thread mode (no worker), first with sign-in off, then on (bootstrap administrator and
a viewer), then with the storage down (port 9), with and without sign-in (``--modes`` picks some):

1. ru, NEW against OLD on the same data: the visible text of <main> and the sidebar and document.title of every
   page, report pages with every analysis and chart tab (times, dates, numbers, ids and run names normalized).
   Allowed differences only (EXPECTED_RU_DIFF): the «Вид» menu, the language field of «Оформление» and the first
   wizard step, the report-language hint of /new, the RU · EN links of the pages without the sidebar. The archive
   sorts the verdict by severity (``/runs?sort=verdict``), the one intended change.
2. NEW re-seeds the demo run and builds a Russian report, compared with OLD's; then «Язык отчётов и промптов» =
   English is saved on the settings page and NEW builds an English report: JSON with ``report_language``, verdict
   codes and no Cyrillic, English prompts in the LLM stub log, job messages in the reader's language (/job_status
   and the archive jobs strip), both publications in the Confluence stub (en without Cyrillic, ru with the same
   structure as OLD's publication of its report). The setting goes back to Russian afterwards.
3. Switching the language: the sidebar «Вид» menu (cookie, localStorage copy, reload, toast in another tab), the
   command palette action, the RU · EN links on /login.
4. The matrix on NEW (unless ``--no-matrix``): every page of PAGES x {ru (no cookie), en (cookie loadlens.lang=en)}
   x {dark, light} x {1440x900, 390x844}: HTTP status, ``<html lang>``, console errors (known: GET /auth/users 503
   with sign-in off, the 503s of an outage), Cyrillic in en (innerText, title, title/placeholder/aria-label/alt,
   options; allowed: «Русский», the prompt text of the editor in the report language and the stored texts of a
   Russian report under its "Report language: Russian" badge), ``scrollWidth <= innerWidth`` (more than OLD has at
   390 px). Screenshots of the viewport go to the output directory.

Waiting uses ``locator.inner_text()`` polls: the CSP forbids eval, so ``wait_for_function`` fails.
Everything goes to %TEMP%/ll_e2e_jobs/s9-<time>/ (logs, LLM stub requests, screenshots, normalized ru texts and
diffs, Confluence bodies, evidence.json); only 127.0.0.1 is contacted. The two reports NEW builds are deleted at the
end unless ``--keep-report``; the planned load-test runs (one per sandbox and check) stay, there is no API to delete
them. Exit code 0 when every expectation holds.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import e2e_common
from e2e_common import (
    Client, Proc, Sandbox, Stack, all_ok, base_parser, expect, finish, log, port_open, stub_json, temp_dir, wait_until,
)

CHECK = "s9_ui_language"
REPO = e2e_common.REPO
NEW_N, OLD_N = 9, 19
STUB_PORTS = {
    "new": {"llm": 5090, "confluence": 5091, "prom": 5092},
    "old": {"llm": 5093, "confluence": 5094, "prom": 5095},
}
STUB_SCRIPTS = {"llm": "llm_stub.py", "confluence": "confluence_stub.py", "prom": "prom_stub.py"}
# Never copied into a sandbox and never read: the sandbox gets example-based files instead.
SECRET_FILES = (
    "settings.py", "settings_runtime.json", "metrics_config.py", "metrics_config_runtime.json",
    ".loadlens_secret_key", "docker/.env",
)
AREA = "bench"
SERVICE = "svc01"
ADMIN_USER, ADMIN_PASSWORD = "e2e-admin", "Sandbox-only-Pass-7731"
VIEWER_USER, VIEWER_PASSWORD = "e2e-viewer", "Sandbox-only-Pass-5519"
STATE_FILE = "s9_state.json"
CYRILLIC = re.compile("[\u0400-\u04ff]")
SANDBOX_SECRET_KEY = e2e_common.SANDBOX_SECRET_KEY


# ---- sandboxes ------------------------------------------------------------------------------------------


def sandbox_root(n: int) -> Path:
    return temp_dir() / f"ll_ui_s{n}"


def _run(args: list[str], cwd: Path, what: str, env: Optional[dict] = None) -> str:
    proc = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if proc.returncode != 0:
        raise SystemExit(f"{what} failed ({proc.returncode}): {proc.stderr.strip()[-2000:] or proc.stdout.strip()[-2000:]}")
    return proc.stdout


def _assert_no_secrets(app_dir: Path) -> None:
    present = [name for name in SECRET_FILES if (app_dir / name).exists()]
    if present:
        raise SystemExit(f"refusing to continue: {present} found in {app_dir} (secrets must never be copied)")


def copy_working_tree(app_dir: Path) -> int:
    """Tracked and untracked files of the working tree, never ignored ones (the secrets are ignored)."""
    listed = _run(["git", "ls-files", "-co", "--exclude-standard"], REPO, "git ls-files").splitlines()
    copied = 0
    for rel in listed:
        if rel in SECRET_FILES:
            raise SystemExit(f"refusing: {rel} is listed by git")
        src = REPO / rel
        if not src.is_file() or rel.startswith("docker/dev-stack/"):
            continue
        dst = app_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def extract_ref(app_dir: Path, ref: str) -> int:
    """``git archive <ref>`` (committed files only) extracted into ``app_dir``."""
    tar_path = app_dir.parent / "old_tree.tar"
    _run(["git", "archive", "--format=tar", "-o", str(tar_path), ref], REPO, "git archive")
    with tarfile.open(tar_path) as archive:
        members = [m for m in archive.getmembers() if not m.name.startswith("docker/dev-stack/")]
        names = {m.name for m in members}
        for name in SECRET_FILES:
            if name in names:
                raise SystemExit(f"refusing: {name} is committed in {ref}")
        archive.extractall(app_dir, members=members, filter="data")
    tar_path.unlink()
    return sum(1 for m in members if m.isfile())


def write_sandbox_settings(app_dir: Path, n: int, db_port: int) -> None:
    """settings.py from the tree's own example: every URL -> 127.0.0.1:9, the storage -> the local database."""
    _assert_no_secrets(app_dir)
    example = (app_dir / "settings.example.py").read_text(encoding="utf-8")
    example = re.sub(r"https?://[A-Za-z0-9_.:\-]+", "http://127.0.0.1:9", example)
    example += f'''

# ---- s9 sandbox overrides: local throwaway database only, no external systems ----
CONFIG["storage"]["timescale"].update({{
    "host": "127.0.0.1", "port": {db_port}, "dbname": "ll_s{n}", "user": "{e2e_common.DB['user']}",
    "password": "{e2e_common.DB['password']}", "sslmode": "disable",
}})
CONFIG["default_params"] = {{"step": "15s", "resample_interval": "1T"}}
'''
    hosts = set(re.findall(r"https?://([^/\"'\s:]+)", example))
    if hosts - {"127.0.0.1"}:
        raise SystemExit(f"sandbox settings would contact {sorted(hosts - {'127.0.0.1'})}")
    (app_dir / "settings.py").write_text(example, encoding="utf-8")
    shutil.copy2(app_dir / "metrics_config.example.py", app_dir / "metrics_config.py")
    (app_dir / "settings_runtime.json").write_text("{}", encoding="utf-8")
    (app_dir / "metrics_config_runtime.json").write_text("{}", encoding="utf-8")


RUNNER = '''"""Starts s9 sandbox {n} on 127.0.0.1:{port} (written by tools/e2e_jobs/s9_ui_language.py).

Environment: LOADLENS_AUTH_ENABLED (default 0), LL_E2E_STORAGE_PORT (another port of the storage database,
e.g. 9 to simulate an outage).
"""
import os
import sys
from pathlib import Path

SANDBOX = Path(r"{app_dir}")
os.environ.setdefault("LOADLENS_AUTH_ENABLED", "0")
os.environ.setdefault("LOADLENS_SECRET_KEY", "{secret}")
os.chdir(SANDBOX)
sys.path.insert(0, str(SANDBOX))
import settings  # noqa: E402  (the example-based settings of the sandbox)

if os.environ.get("LL_E2E_STORAGE_PORT"):
    settings.CONFIG["storage"]["timescale"]["port"] = int(os.environ["LL_E2E_STORAGE_PORT"])
from app import app  # noqa: E402

app.run(host="127.0.0.1", port={port}, threaded=True, use_reloader=False)
'''


def write_runner(n: int) -> None:
    root = sandbox_root(n)
    text = RUNNER.format(n=n, port=5070 + n, app_dir=root / "app", secret=SANDBOX_SECRET_KEY)
    (root / "run_app.py").write_text(text, encoding="utf-8")


def build_sandbox(kind: str, n: int, ref: str, db_port: int) -> Path:
    root = sandbox_root(n)
    app_dir = root / "app"
    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)
    count = copy_working_tree(app_dir) if kind == "new" else extract_ref(app_dir, ref)
    _assert_no_secrets(app_dir)
    write_sandbox_settings(app_dir, n, db_port)
    write_runner(n)
    log(f"{kind} sandbox {n}: {count} files in {app_dir}")
    return app_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig") or "{}") if path.exists() else {}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def wire_runtime(app_dir: Path, ports: dict, app_port: int) -> None:
    """Stub keys of the tree's own bench_endpoints.py plus the s9 extras: SLA thresholds, a manual generator."""
    _run([
        sys.executable, "-X", "utf8", str(app_dir / "tools" / "bench_endpoints.py"), "--setup-stubs",
        "--runtime-json", str(app_dir / "settings_runtime.json"), "--area", AREA, "--base", f"http://127.0.0.1:{app_port}",
        "--llm-stub", f"http://127.0.0.1:{ports['llm']}", "--confluence-stub", f"http://127.0.0.1:{ports['confluence']}",
        "--prom-stub", f"http://127.0.0.1:{ports['prom']}",
    ], app_dir, "bench_endpoints.py --setup-stubs")
    path = app_dir / "settings_runtime.json"
    data = _read_json(path)
    # Configuration data shows up as is on the pages: keep it Latin so it does not look like a missing translation.
    data["data_sources"]["prometheus"]["title"] = "Prometheus stub"
    data["per_area"][AREA]["sla"] = {"target_rps": 100, "max_error_rate_pct": 5.0, "max_p95_ms": 2000, "max_cpu_pct": 90}
    data["load_generators"] = {"manual": {"title": "Manual", "type": "manual"}}
    _write_json(path, data)


def rewire_ports(app_dir: Path, ports: dict, app_port: int) -> None:
    """Points a runtime file copied from the other sandbox at this sandbox's stubs."""
    path = app_dir / "settings_runtime.json"
    data = _read_json(path)
    data["llm"]["openai"]["api_base_url"] = f"http://127.0.0.1:{ports['llm']}/v1"
    data["confluence"]["base_url"] = f"http://127.0.0.1:{ports['confluence']}"
    data["confluence"]["app_url"] = f"http://127.0.0.1:{app_port}"
    data["data_sources"]["prometheus"]["prometheus"]["url"] = f"http://127.0.0.1:{ports['prom']}"
    _write_json(path, data)


# ---- database ---------------------------------------------------------------------------------------------


def _admin_connection():
    import psycopg2

    connection = psycopg2.connect(dbname="postgres", connect_timeout=5, **e2e_common.DB)
    connection.autocommit = True
    return connection


def _terminate(cursor, dbname: str) -> None:
    cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()", (dbname,))


def recreate_database(dbname: str, template: Optional[str] = None) -> None:
    connection = _admin_connection()
    try:
        with connection.cursor() as cursor:
            _terminate(cursor, dbname)
            cursor.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
            if template:
                _terminate(cursor, template)
                cursor.execute(f'CREATE DATABASE "{dbname}" TEMPLATE "{template}"')
            else:
                cursor.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        connection.close()
    log(f"database {dbname} created" + (f" from {template}" if template else ""))


# ---- processes ----------------------------------------------------------------------------------------------


def start_stubs(stack: Stack, kind: str, app_dir: Path, llm_log: Optional[Path] = None, llm_delay: float = 0.3) -> None:
    """The stubs of a sandbox tree on its own ports (OLD answers like the old code expects)."""
    for name, port in STUB_PORTS[kind].items():
        if port_open(port):
            raise SystemExit(f"port {port} is busy: stop whatever listens there (netstat -ano | grep :{port})")
        args = [sys.executable, "-X", "utf8", str(app_dir / "tools" / "stubs" / STUB_SCRIPTS[name]), "--port", str(port)]
        if name == "llm":
            args += ["--delay", str(llm_delay)]
            if llm_log:
                args += ["--log", str(llm_log)]
        proc = Proc(f"{kind}-{name}-stub", args, app_dir, dict(os.environ), stack._log_path(f"{name}-stub")).start()
        stack.procs.append(proc)
        wait_until(lambda: port_open(port), 30, what=f"the {kind} {name} stub on {port}")


def start_server(stack: Stack, mode: str = "off", **env: Any) -> Proc:
    """``mode``: off (sign-in off), auth (sign-in on, bootstrap admin), down (storage on port 9), down_auth."""
    extra: dict = dict(env)
    if mode in ("auth", "down_auth"):
        extra.update(LOADLENS_AUTH_ENABLED="1", LOADLENS_ADMIN_USER=ADMIN_USER, LOADLENS_ADMIN_PASSWORD=ADMIN_PASSWORD)
    if mode in ("down", "down_auth"):
        extra.update(LL_E2E_STORAGE_PORT="9")
    proc = stack.server("thread", **extra)
    if mode in ("off", "auth"):
        time.sleep(2.0)  # start-up migrations run in a background thread
    return proc


def stop_server(stack: Stack, proc: Optional[Proc]) -> None:
    if proc is not None:
        proc.kill()
        if proc in stack.procs:
            stack.procs.remove(proc)
        wait_until(lambda: not port_open(stack.sb.port), 20, what=f"port {stack.sb.port} to close")


# ---- HTTP helpers -----------------------------------------------------------------------------------------


def make_client(sb: Sandbox, lang: Optional[str] = None, area: str = AREA) -> Client:
    client = Client(sb, area=area)
    if lang:
        client.session.cookies.set("loadlens.lang", lang, domain="127.0.0.1")
    return client


_CSRF_INPUT = re.compile(r"""name=["']csrf_token["']\s+value=["']([^"']*)["']""")


def login_csrf(page: str) -> str:
    """CSRF token of the sign-in form (hidden input) or of a page (meta)."""
    match = _CSRF_INPUT.search(page) or e2e_common._CSRF_META.search(page)
    return match.group(1) if match else ""


def login(client: Client, username: str, password: str) -> dict:
    """Signs the session in (JSON form of POST /login); keeps the CSRF token of the new session."""
    token = login_csrf(client.session.get(client.sb.base + "/login", timeout=30).text)
    headers = {"X-CSRF-Token": token} if token else {}
    response = client.session.post(client.sb.base + "/login", json={"username": username, "password": password}, headers=headers, timeout=30)
    body = response.json() if response.headers.get("Content-Type", "").startswith("application/json") else {}
    if response.status_code != 200 or not body.get("ok"):
        raise SystemExit(f"login {username} -> {response.status_code}: {response.text[:300]}")
    client.csrf = body.get("csrf_token") or ""
    return body


def report_ref(client: Client, run_name: str) -> dict:
    response = client.get("/report_ref/" + quote(run_name, safe=""))
    return response.json() if response.status_code == 200 else {}


def build_report(client: Client, run_name: str, timeout: float, on_poll: Optional[Callable[[dict], None]] = None) -> dict:
    """A web report of svc01 (step, 60 min, AI on) through the job pipeline; returns the final job state."""
    accepted = client.create_report(service=SERVICE, window_min=60, test_type="step", run_name=run_name)
    final = client.wait_job(accepted["job_id"], timeout, on_poll=on_poll)
    final["job_id"] = accepted["job_id"]
    final["run_name"] = run_name
    final.update({k: v for k, v in report_ref(client, run_name).items() if k in ("run_id", "service")})
    return final


def load_state() -> dict:
    path = sandbox_root(NEW_N) / STATE_FILE
    if not path.is_file():
        raise SystemExit(f"{path} not found: run this script with --prepare first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---- --prepare ----------------------------------------------------------------------------------------------


def prepare(args) -> int:
    db_port = e2e_common.DB["port"]
    if not port_open(db_port):
        raise SystemExit(f"no database on 127.0.0.1:{db_port}: start the container first (see the header of this script)")
    out_dir = Path(args.log_dir) if args.log_dir else temp_dir() / "ll_e2e_jobs" / f"s9-prepare-{datetime.now():%Y%m%d-%H%M%S}"
    old_app = build_sandbox("old", OLD_N, args.old_ref, db_port)
    new_app = build_sandbox("new", NEW_N, "", db_port)
    old_commit = _run(["git", "rev-parse", args.old_ref], REPO, "git rev-parse").strip()
    recreate_database(f"ll_s{NEW_N}")
    recreate_database(f"ll_s{OLD_N}")

    dsn = " ".join(f"{k}={v}" for k, v in {**e2e_common.DB, "dbname": f"ll_s{OLD_N}"}.items())
    seeded = _run([
        sys.executable, "-X", "utf8", str(old_app / "tools" / "seed_bench_data.py"), "--dsn", dsn,
        "--runtime-json", str(old_app / "settings_runtime.json"), "--runs", "24", "--services", "3",
        "--runtime-kb", "0", "--metrics-runtime-kb", "0",
    ], old_app, "seed_bench_data.py (OLD)")
    log(f"OLD seeded: {seeded.strip()[-300:]}")
    wire_runtime(old_app, STUB_PORTS["old"], 5070 + OLD_N)

    state: dict = {"old_ref": args.old_ref, "old_commit": old_commit, "prepared_at": datetime.now().isoformat(timespec="seconds")}
    old_sb = Sandbox(OLD_N)
    with Stack(old_sb, CHECK, str(out_dir / "old")) as stack:
        start_stubs(stack, "old", old_app)
        server = start_server(stack)
        client = make_client(old_sb)
        demo = client.post("/demo/seed", {})
        if demo.status_code != 200:
            raise SystemExit(f"POST /demo/seed (OLD) -> {demo.status_code}: {demo.text[:300]}")
        state["demo"] = {k: demo.json().get(k) for k in ("run_name", "service", "area", "report_url")}
        state["demo"]["run_id"] = str(state["demo"]["report_url"] or "").rsplit("/", 1)[-1]
        client = make_client(old_sb)  # the demo answer switched the area cookie to "demo"
        report = build_report(client, "e2e-s9-ru-old", args.timeout)
        if report.get("status") != "done" or not report.get("run_id"):
            raise SystemExit(f"OLD report failed: {report}")
        state["old_report"] = {k: report.get(k) for k in ("run_name", "run_id", "service", "job_id")}
        stop_server(stack, server)
    recreate_database(f"ll_s{NEW_N}", template=f"ll_s{OLD_N}")
    for name in ("settings_runtime.json", "metrics_config_runtime.json"):
        shutil.copy2(old_app / name, new_app / name)
    rewire_ports(new_app, STUB_PORTS["new"], 5070 + NEW_N)
    (sandbox_root(NEW_N) / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"prepared: {json.dumps(state, ensure_ascii=False)}")
    return 0


# ---- pages ----------------------------------------------------------------------------------------------------

SETTINGS_SECTIONS = ("projects", "connections", "sla", "context", "queries", "prompts", "services", "metrics", "appearance", "users")
THEMES = ("dark", "light")
VIEWPORTS = ((1440, 900), (390, 844))


class Page:
    """A page of the matrix. Paths use {placeholders} of the run registry; ``old_path`` None: no OLD comparison."""

    def __init__(self, key: str, path: str, *, mode: str = "off", old_path: Optional[str] = "", tabs: bool = False,
                 stored: Optional[str] = None, status: int = 200, user: Optional[str] = None, compare_phase: str = "same",
                 new_action: Optional[str] = None):
        self.key, self.path, self.mode, self.tabs, self.stored, self.status, self.user = key, path, mode, tabs, stored, status, user
        # Name of an ACTIONS entry run on NEW pages before the capture (OLD pages are captured as loaded).
        self.new_action = new_action
        self.old_path = path if old_path == "" else old_path
        # same: compared while NEW and OLD hold the same data; late: compared after NEW built its own reports.
        self.compare_phase = compare_phase


PAGES = [
    Page("dashboard", "/"),
    Page("archive", "/reports"),
    Page("report_legacy_ru", "/reports/{legacy_ru_id}", tabs=True, stored="legacy_ru"),
    Page("report_demo", "/reports/{demo_id}", old_path="/reports/{old_demo_id}", tabs=True, stored="demo", compare_phase="late"),
    Page("report_ru", "/reports/{ru_id}", old_path="/reports/{legacy_ru_id}", tabs=True, stored="ru", compare_phase="late"),
    Page("report_en", "/reports/{en_id}", old_path=None, tabs=True, stored="en"),
    Page("new", "/new"),
    Page("compare", "/compare?run_a={legacy_ru_name}&run_b={bench_name}"),
    Page("forecasting", "/forecasting"),
    Page("forecast_ru", "/forecasting/{legacy_ru_id}", stored="legacy_ru"),
    Page("forecast_en", "/forecasting/{en_id}", old_path=None, stored="en"),
    Page("launch", "/launch"),
    Page("launch_run", "/launch/{lt_id}", old_path="/launch/{old_lt_id}"),
    *[Page(f"settings_{name}", f"/settings#{name}") for name in SETTINGS_SECTIONS],
    # The wizard of NEW starts with the language step (no OLD counterpart); its next step is OLD's first one.
    Page("settings_wizard", "/settings?wizard=1", old_path=None),
    Page("settings_wizard_storage", "/settings?wizard=1", new_action="wizard_next"),
    Page("login", "/login", mode="auth"),
    Page("account", "/account", mode="auth", user="admin"),
    Page("forbidden", "/settings", mode="auth", user="viewer", status=403),
    Page("recovery", "/recovery", mode="down"),
    Page("dashboard_storage_down", "/", mode="down"),
    Page("storage_unavailable", "/reports/{legacy_ru_id}", mode="down", status=503),
    Page("login_storage_down", "/login", mode="down_auth"),
]

# Console errors that are not defects (plans/i18n-plan.md, 6.4): GET /auth/users answers 503 with sign-in off; an
# outage answers 503 on purpose; the forbidden page itself is a 403.
KNOWN_CONSOLE = {
    "off": (r"/auth/users",),
    "auth": (),
    "down": (r"status of 503",),
    "down_auth": (r"status of 503",),
}

# The only differences of the Russian pages allowed by plans/i18n-plan.md (status section): the sidebar «Вид»
# menu with its «Язык» group, the language field of «Оформление» and its wizard step, the report-language hint
# of /new, the RU · EN links of the pages without the sidebar. Each entry: (page key or "*", regex a changed line
# must match).
EXPECTED_RU_DIFF = (
    ("*", r"^(Тема|Вид)$"),
    ("*", r"^(Тёмная|Светлая|Системная)( · RU)?$"),
    ("settings_appearance", r"Язык отчётов и промптов|язык.*(отчёт|промпт)|^Русский$|^English$|^Цвет акцента$"),
    ("settings_appearance", r"^Действует на новые отчёты|^Акцентный цвет"),
    ("settings_wizard_storage", r"^(✓|#)\. Язык$"),
    ("new", r"(Язык|язык) отчёт"),
    # The RU · EN links of the standalone pages (the «Язык» group of pages without the sidebar).
    ("login", r"^(RU|EN|·)$"),
    ("recovery", r"^(RU|EN|·)$"),
    ("storage_unavailable", r"^(RU|EN|·)$"),
    ("login_storage_down", r"^(RU|EN|·)$"),
)

EXTRACT_JS = """() => {
  const attrs = [];
  document.querySelectorAll('[title],[placeholder],[aria-label],[alt]').forEach((n) => {
    ['title', 'placeholder', 'aria-label', 'alt'].forEach((a) => { const v = n.getAttribute(a); if (v) attrs.push(a + ': ' + v); });
  });
  const options = Array.from(document.querySelectorAll('option')).map((o) => 'option: ' + o.textContent);
  const W = window.innerWidth;
  const wide = [];
  document.querySelectorAll('body *').forEach((n) => {
    const r = n.getBoundingClientRect();
    if (r.width && r.right > W + 1 && wide.length < 12) {
      wide.push(n.tagName.toLowerCase() + (n.id ? '#' + n.id : '') + (typeof n.className === 'string' && n.className ? '.' + n.className.trim().split(/\\s+/).join('.') : '') + ' right=' + Math.round(r.right));
    }
  });
  const main = document.querySelector('main') || document.querySelector('.auth-card') || document.body;
  const side = document.querySelector('.app-sidebar');
  const ace = Array.from(document.querySelectorAll('.ace_editor')).map((n) => n.innerText).join('\\n');
  const badge = document.querySelector('[data-report-language]');
  const badgeVisible = !!(badge && badge.offsetParent !== null);
  return {
    lang: document.documentElement.lang, title: document.title, main: main.innerText, side: side ? side.innerText : '',
    body: document.body.innerText, attrs, options, ace, scrollWidth: document.documentElement.scrollWidth, innerWidth: W, wide,
    badge: badgeVisible ? badge.getAttribute('data-report-language') + ': ' + badge.innerText : '',
  };
}"""


def fill(path: str, runs: dict) -> Optional[str]:
    try:
        return path.format(**runs)
    except KeyError:
        return None


def settle(page, timeout: float = 20.0, quiet: float = 1.5) -> None:
    """Waits until the visible text stops changing for ``quiet`` seconds and no skeleton is shown (no eval: CSP)."""
    deadline = time.monotonic() + timeout
    last, since = None, time.monotonic()
    page.wait_for_timeout(500)
    while time.monotonic() < deadline:
        try:
            text = page.locator("body").inner_text(timeout=5000)
            busy = page.locator(".skeleton:visible, .chart-pending:visible").count()
        except Exception:  # noqa: BLE001 - navigation in progress
            text, busy = None, 1
        if text != last or busy:
            last, since = text, time.monotonic()
        elif time.monotonic() - since >= quiet:
            return
        page.wait_for_timeout(300)


def capture(page) -> dict:
    return page.evaluate(EXTRACT_JS)


def tab_states(page, wide_only: bool = False) -> list[dict]:
    """Captures of a report page: as loaded, then with every analysis tab and every chart tab active."""
    states = [capture(page)]
    for selector in ("#rep-llm-tabs .tabs .tab", "#rep-domain-tabs .tabs .tab"):
        count = page.locator(selector).count()
        for index in range(1, count):
            try:
                page.locator(selector).nth(index).click(timeout=5000)
            except Exception:  # noqa: BLE001 - a hidden tab (narrow screen): scroll and retry once
                page.locator(selector).nth(index).scroll_into_view_if_needed(timeout=5000)
                page.locator(selector).nth(index).click(timeout=5000)
            settle(page, timeout=12, quiet=0.8)
            states.append(capture(page))
    return states


# ---- texts ------------------------------------------------------------------------------------------------------

CYR_RUN = re.compile("[Ѐ-ӿ](?:[Ѐ-ӿ\\s\\-–—,.:;!?()«»\"'%/0-9≈·]*[Ѐ-ӿ])?")
_SPACES = re.compile(r"\s+")


def collapse(text: str) -> str:
    return _SPACES.sub(" ", text or "").strip()


def strings_of(value: Any) -> list[str]:
    """Every string inside a JSON value, JSON-encoded strings included."""
    out: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            out.append(item)
            stripped = item.strip()
            if stripped[:1] in "{[":
                try:
                    stack.append(json.loads(stripped))
                except ValueError:
                    pass
    return out


def stored_texts(client: Client, run: dict) -> str:
    """Cyrillic texts a saved report may show as is on an English page: its rows of /llm_reports."""
    response = client.get(f"/llm_reports?run_name={quote(run['run_name'])}&service={quote(run['service'])}")
    rows = response.json() if response.status_code == 200 else []
    return "\n".join(collapse(s) for s in strings_of(rows) if CYRILLIC.search(s))


def cyrillic_findings(capture_: dict, allowed_sources: list[str]) -> tuple[list[str], list[str]]:
    """(not allowed, allowed) Cyrillic lines of an English page capture."""
    lines = capture_["body"].splitlines() + capture_["attrs"] + capture_["options"] + [capture_["title"]]
    sources = [collapse(s) for s in allowed_sources if s]
    bad, allowed = [], []
    for line in dict.fromkeys(collapse(line) for line in lines):
        runs = CYR_RUN.findall(line)
        if not runs:
            continue
        if all(run == "Русский" or any(run in source for source in sources) for run in runs):
            allowed.append(line)
        else:
            bad.append(line)
    return bad, allowed


_NORMALIZERS = (
    (re.compile(r"\b[0-9a-f]{32}\b"), "<ID>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<ID>"),
    (re.compile(r"\bdemo-\d{9,}\b"), "demo-<ID>"),
    (re.compile(r"\b[0-9a-f]{10,}\b"), "<HASH>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{1,2}:\d{2}(:\d{2})?)?"), "<DATE>"),
    (re.compile(r"\d{1,2}\.\d{1,2}\.\d{2,4}(,? \d{1,2}:\d{2}(:\d{2})?)?"), "<DATE>"),
    (re.compile(r"\d{1,2}:\d{2}(:\d{2})?"), "<TIME>"),
    (re.compile(r"(только что|вчера|позавчера|сегодня)( в <TIME>)?"), "<REL>"),
    (re.compile(r"\d+(?:[\u00a0\u202f ]\d{3})*(?:[.,]\d+)?"), "#"),
    # Relative times depend on the second of the visit ("только что" on one side, "1 мин назад" on the other).
    (re.compile(r"# (сек|с|мин|ч|дн\w*|д)\.? назад"), "<REL>"),
)


def normalize(text: str, aliases: dict) -> list[str]:
    """Lines of a page text with run names, ids, dates, times and numbers replaced by placeholders."""
    for real, alias in sorted(aliases.items(), key=lambda item: -len(item[0])):
        if real:
            text = text.replace(real, alias)
    out = []
    for line in text.splitlines():
        line = collapse(line)
        for rx, repl in _NORMALIZERS:
            line = rx.sub(repl, line)
        if line:
            out.append(line)
    return out


def ru_text(states: list[dict]) -> str:
    first = states[0]
    parts = [f"TITLE: {first['title']}", "SIDEBAR:", first["side"], "MAIN:"]
    parts += [state["main"] for state in states]
    return "\n".join(parts)


def unexpected_diff(key: str, old_lines: list[str], new_lines: list[str]) -> tuple[list[str], list[str]]:
    """(unexpected, expected) changed lines between OLD and NEW (``-``/``+`` prefixed)."""
    patterns = [re.compile(rx) for keys, rx in EXPECTED_RU_DIFF if keys == "*" or keys == key]
    unexpected, expected = [], []
    for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        body = line[1:]
        (expected if any(p.search(body) for p in patterns) else unexpected).append(line)
    return unexpected, expected


# ---- browser ---------------------------------------------------------------------------------------------------


class Browser:
    """One Chromium for the check; contexts per language, theme and viewport; console errors per page."""

    def __init__(self, pw, out: Path, headed: bool = False):
        self.browser = pw.chromium.launch(headless=not headed)
        self.out = out
        (out / "shots").mkdir(parents=True, exist_ok=True)

    def context(self, sb: Sandbox, lang: Optional[str], theme: str = "dark", size: tuple = (1440, 900), area: str = AREA):
        browser_locale = "en-US" if lang == "en" else "ru-RU"
        ctx = self.browser.new_context(viewport={"width": size[0], "height": size[1]}, locale=browser_locale)
        cookies = [{"name": "project_area", "value": area, "url": sb.base}]
        if lang:
            cookies.append({"name": "loadlens.lang", "value": lang, "url": sb.base})
        ctx.add_cookies(cookies)
        ctx.add_init_script(f"try {{ localStorage.setItem('loadlens.theme', '{theme}'); }} catch (e) {{}}")
        return ctx

    @staticmethod
    def watch(page, errors: list) -> None:
        def on_console(msg):
            if msg.type == "error":
                errors.append(f"{msg.text} @ {(msg.location or {}).get('url', '')}")
        page.on("console", on_console)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))

    def open(self, ctx, url: str, errors: list, timeout: float = 25.0):
        page = ctx.new_page()
        self.watch(page, errors)
        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        settle(page, timeout=timeout)
        return page, response

    def close(self) -> None:
        self.browser.close()


def sign_in(ctx, sb: Sandbox, username: str, password: str) -> None:
    token = login_csrf(ctx.request.get(sb.base + "/login").text())
    headers = {"Content-Type": "application/json", **({"X-CSRF-Token": token} if token else {})}
    response = ctx.request.post(sb.base + "/login", data=json.dumps({"username": username, "password": password}), headers=headers)
    if response.status != 200:
        raise SystemExit(f"browser sign-in {username} -> {response.status}: {response.text()[:200]}")


# ---- the check ------------------------------------------------------------------------------------------------


class Run:
    """State of one check run: results, evidence, output directory, runs of both sandboxes."""

    def __init__(self, args, out: Path):
        self.args, self.out = args, out
        self.results: dict = {}
        self.evidence: dict = {"matrix": {}, "ru_compare": {}, "scenarios": {}, "overflow_old": {}}
        self.runs: dict = {}
        self.aliases: dict = {}
        self.stored: dict = {}
        self.new_sb, self.old_sb = Sandbox(NEW_N), Sandbox(OLD_N)
        self.use_old = not args.no_old
        self.only = set(filter(None, (args.pages or "").split(",")))

    def wanted(self, page: Page) -> bool:
        return not self.only or page.key in self.only

    def write(self, rel: str, text: str) -> Path:
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def expect(self, name: str, ok: bool, detail: Any = None) -> bool:
        return expect(self.results, name, ok, detail)


def visit(run: Run, browser: Browser, sb: Sandbox, page_spec: Page, path: str, lang: Optional[str], theme: str, size: tuple,
          tabs: bool, shot: Optional[str] = None, action: Optional[str] = None) -> dict:
    """Opens one page in a fresh context; returns the captures, console errors, status and overflow."""
    ctx = browser.context(sb, lang, theme, size, area=page_spec_area(page_spec))
    errors: list = []
    try:
        if page_spec.user:
            user, password = (ADMIN_USER, ADMIN_PASSWORD) if page_spec.user == "admin" else (VIEWER_USER, VIEWER_PASSWORD)
            sign_in(ctx, sb, user, password)
        page, response = browser.open(ctx, sb.base + path, errors)
        if action:
            ACTIONS[action](page)
        states = tab_states(page) if tabs else [capture(page)]
        if shot:
            page.screenshot(path=str(browser.out / "shots" / shot))
        status = response.status if response else None
    finally:
        ctx.close()
    known = [re.compile(rx) for rx in KNOWN_CONSOLE.get(page_spec.mode, ())]
    if page_spec.status >= 400:
        known.append(re.compile(rf"status of {page_spec.status}"))
    noise = [e for e in errors if not any(rx.search(e) for rx in known)]
    overflow = max(state["scrollWidth"] - state["innerWidth"] for state in states)
    wide = next((state["wide"] for state in states if state["scrollWidth"] > state["innerWidth"]), [])
    return {"states": states, "errors": errors, "noise": noise, "status": status, "overflow": overflow, "wide": wide}


def _wizard_next(page) -> None:
    page.locator("#wizardNextBtn").click()
    settle(page, timeout=10)


ACTIONS: dict[str, Callable] = {"wizard_next": _wizard_next}


def page_spec_area(page_spec: Page) -> str:
    return "demo" if page_spec.stored == "demo" else AREA


def compare_ru(run: Run, browser: Browser, page_spec: Page, phase: str) -> None:
    """Russian NEW against OLD for one page: visible text after normalization; overflow of OLD at 390 px."""
    if page_spec.compare_phase != phase or page_spec.old_path is None or not run.use_old or not run.wanted(page_spec):
        return
    new_path, old_path = fill(page_spec.path, run.runs), fill(page_spec.old_path, run.runs)
    if new_path is None or old_path is None:
        run.expect(f"ru compare {page_spec.key}: paths", False, "a run of the page is missing")
        return
    new = visit(run, browser, run.new_sb, page_spec, new_path, None, "dark", VIEWPORTS[0], page_spec.tabs, action=page_spec.new_action)
    old = visit(run, browser, run.old_sb, page_spec, old_path, None, "dark", VIEWPORTS[0], page_spec.tabs)
    new_lines, old_lines = normalize(ru_text(new["states"]), run.aliases), normalize(ru_text(old["states"]), run.aliases)
    run.write(f"ru/{page_spec.key}.new.txt", "\n".join(new_lines) + "\n")
    run.write(f"ru/{page_spec.key}.old.txt", "\n".join(old_lines) + "\n")
    unexpected, expected_lines = unexpected_diff(page_spec.key, old_lines, new_lines)
    if unexpected or expected_lines:
        run.write(f"ru/{page_spec.key}.diff", "\n".join(difflib.unified_diff(old_lines, new_lines, "old", "new", lineterm="")) + "\n")
    run.evidence["ru_compare"][page_spec.key] = {"expected": expected_lines, "unexpected": unexpected[:40],
                                                 "lang_new": new["states"][0]["lang"], "status": [old["status"], new["status"]]}
    run.expect(f"ru {page_spec.key}: NEW renders like OLD", not unexpected and new["status"] == old["status"]
               and new["states"][0]["lang"] == "ru", {"unexpected": unexpected[:8], "expected": len(expected_lines),
                                                      "status": [old["status"], new["status"]]})
    narrow = visit(run, browser, run.old_sb, page_spec, old_path, None, "dark", VIEWPORTS[1], page_spec.tabs)
    run.evidence["overflow_old"][page_spec.key] = {"overflow": narrow["overflow"], "wide": narrow["wide"]}


def matrix(run: Run, browser: Browser, pages: list[Page]) -> None:
    """Every page x {ru, en} x {dark, light} x {1440, 390} on NEW."""
    for page_spec in pages:
        if not run.wanted(page_spec):
            continue
        path = fill(page_spec.path, run.runs)
        if path is None:
            run.expect(f"matrix {page_spec.key}", False, "a run of the page is missing")
            continue
        allowed = [run.stored.get(page_spec.stored, "")] if page_spec.stored else []
        cells = {}
        for lang in ("ru", "en"):
            for theme in THEMES:
                for size in VIEWPORTS:
                    cell = f"{lang}-{theme}-{size[0]}"
                    tabs = page_spec.tabs and theme == "dark"
                    result = visit(run, browser, run.new_sb, page_spec, path, None if lang == "ru" else "en", theme, size, tabs,
                                   shot=f"{page_spec.key}_{cell}.png", action=page_spec.new_action)
                    first = result["states"][0]
                    problems = []
                    if result["status"] != page_spec.status:
                        problems.append(f"status {result['status']}")
                    if first["lang"] != lang:
                        problems.append(f"lang {first['lang']}")
                    if result["noise"]:
                        problems.append(f"console {result['noise'][:3]}")
                    bad, ok_lines = [], []
                    if lang == "en":
                        sources = allowed + [state["ace"] for state in result["states"]]
                        for state in result["states"]:
                            b, a = cyrillic_findings(state, sources)
                            bad += [x for x in b if x not in bad]
                            ok_lines += [x for x in a if x not in ok_lines]
                        stored_shown = [x for x in ok_lines if x != "Русский" and not x.endswith(": Русский")]
                        if stored_shown and page_spec.stored and not any(s["badge"] for s in result["states"]):
                            # Stored Russian texts on an English page need the "Report language: Russian" badge.
                            if page_spec.key.startswith("report_"):
                                problems.append("stored Russian texts without the report-language badge")
                            else:
                                bad += [f"(stored, no badge) {x}" for x in stored_shown]
                        if bad:
                            problems.append(f"cyrillic {bad[:4]}")
                    pre_existing = run.evidence["overflow_old"].get(page_spec.key, {}).get("overflow", 0) if size[0] < 500 else 0
                    if result["overflow"] > 0 and result["overflow"] > pre_existing + 2:
                        problems.append(f"horizontal overflow {result['overflow']} px {result['wide'][:3]}")
                    cells[cell] = {"ok": not problems, "problems": problems, "cyrillic_allowed": ok_lines[:20],
                                   "overflow": result["overflow"], "pre_existing_overflow": pre_existing,
                                   "badge": next((s["badge"] for s in result["states"] if s["badge"]), "")}
                    log(f"    {page_spec.key} {cell}: {'ok' if not problems else problems}")
        run.evidence["matrix"][page_spec.key] = cells
        failed = {cell: value["problems"] for cell, value in cells.items() if not value["ok"]}
        run.expect(f"matrix {page_spec.key}", not failed, failed or f"{len(cells)} cells ok")


# ---- scenarios ------------------------------------------------------------------------------------------------


def poll_text(locator, predicate: Callable[[str], bool], timeout: float) -> str:
    """Polls ``locator.inner_text()`` (wait_for_function needs eval, which the CSP forbids)."""
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        try:
            text = locator.inner_text(timeout=2000)
        except Exception:  # noqa: BLE001 - not attached yet / navigating
            text = ""
        if predicate(text):
            return text
        locator.page.wait_for_timeout(300)
    return text


def poll_lang(page, lang: str, timeout: float = 15.0) -> str:
    deadline = time.monotonic() + timeout
    value = ""
    while time.monotonic() < deadline:
        try:
            value = page.evaluate("document.documentElement.lang")
        except Exception:  # noqa: BLE001 - the page is reloading
            value = ""
        if value == lang:
            return value
        page.wait_for_timeout(300)
    return value


def lang_cookie(ctx) -> Optional[str]:
    return next((c["value"] for c in ctx.cookies() if c["name"] == "loadlens.lang"), None)


def create_lt_run(client: Client, name: str) -> str:
    """A planned load-test run on the manual generator (nothing runs anywhere; the watcher is off)."""
    response = client.post("/load_tests", {"service": SERVICE, "test_type": "step", "run_name": name,
                                           "instances": [{"name": "by hand", "generator": "manual"}]})
    body = response.json() if response.headers.get("Content-Type", "").startswith("application/json") else {}
    if response.status_code != 200 or not body.get("run_id"):
        raise SystemExit(f"POST /load_tests -> {response.status_code}: {response.text[:300]}")
    return body["run_id"]


def set_report_language(client: Client, lang: str) -> int:
    return client.post("/config", {"section": "appearance", "data": {"language": lang}}).status_code


VERDICT_RANK = {"pass": 0, "risk": 1, "fail": 2, "no_data": 3}


def scenario_verdict_sort(run: Run) -> None:
    """The archive sorts the verdict by severity (pass, risk, fail, no_data): the one intended change of the ru
    archive (plans/i18n-plan.md, section 7, question 3). OLD sorted by the text of the Russian label."""
    rows = make_client(run.new_sb).get("/runs?sort=verdict&dir=asc&limit=100").json()
    codes = [str(r.get("verdict")) for r in rows if r.get("verdict")]
    ranks = [VERDICT_RANK.get(code, 9) for code in codes]
    evidence = {"new": list(dict.fromkeys(codes))}
    if run.use_old:
        old_rows = make_client(run.old_sb).get("/runs?sort=verdict&dir=asc&limit=100").json()
        evidence["old"] = list(dict.fromkeys(str(r.get("verdict")) for r in old_rows if r.get("verdict")))
    run.evidence["scenarios"]["verdict_sort"] = evidence
    run.expect("archive: verdict sort by severity", bool(codes) and ranks == sorted(ranks) and set(codes) <= set(VERDICT_RANK), evidence)


def scenario_switch_menu(run: Run, browser: Browser) -> None:
    """Sidebar «Вид» menu: the cookie and its localStorage copy are written, the page reloads in English, another
    tab gets the toast; then back to Russian."""
    sb = run.new_sb
    ctx = browser.context(sb, None)
    errors: list = []
    try:
        first, _ = browser.open(ctx, sb.base + "/", errors)
        other, _ = browser.open(ctx, sb.base + "/reports", errors)
        before = first.evaluate("document.documentElement.lang")
        first.bring_to_front()
        first.locator("#viewMenu > summary").click()
        first.locator('#viewMenu [data-view-pref="lang"][data-value="en"]').click()
        after = poll_lang(first, "en")
        settle(first, timeout=10)
        stored = first.evaluate("localStorage.getItem('loadlens.lang')")
        value = poll_text(first.locator("#viewMenuValue"), lambda t: "EN" in t, 5)
        toast = poll_text(other.locator(".toast-host"), lambda t: bool(t.strip()), 8)
        run.evidence["scenarios"]["switch_menu"] = {"before": before, "after": after, "cookie": lang_cookie(ctx), "local_storage": stored,
                                                     "menu_value": value, "other_tab_toast": toast, "errors": errors}
        run.expect("switch: «Вид» menu sets the cookie and reloads in English",
                   before == "ru" and after == "en" and lang_cookie(ctx) == "en" and stored == "en" and "EN" in value,
                   {"before": before, "after": after, "cookie": lang_cookie(ctx), "local_storage": stored, "menu_value": value})
        run.expect("switch: another tab gets the toast", bool(toast.strip()), toast)
        first.locator("#viewMenu > summary").click()
        first.locator('#viewMenu [data-view-pref="lang"][data-value="ru"]').click()
        back = poll_lang(first, "ru")
        run.expect("switch: back to Russian from the menu", back == "ru" and lang_cookie(ctx) == "ru", {"lang": back, "cookie": lang_cookie(ctx)})
        run.expect("switch: no console errors", not [e for e in errors if "/auth/users" not in e], errors[:5])
    finally:
        ctx.close()


def scenario_palette(run: Run, browser: Browser) -> None:
    """Command palette action: «Switch to English» on a Russian page, «Переключить на русский» on an English one."""
    sb = run.new_sb
    ctx = browser.context(sb, None)
    errors: list = []
    try:
        page, _ = browser.open(ctx, sb.base + "/", errors)
        seen = {}
        for query, target in (("English", "en"), ("русский", "ru")):
            page.keyboard.press("Control+k")
            page.wait_for_timeout(400)
            page.keyboard.type(query)
            page.wait_for_timeout(600)
            listed = poll_text(page.locator("#paletteList"), lambda t: bool(t.strip()), 5)
            option = page.locator("#paletteList .palette-option", has_text=re.compile("English|русский", re.IGNORECASE)).first
            label = option.inner_text(timeout=3000) if option.count() else ""
            if option.count():
                option.click()
            else:
                page.keyboard.press("Enter")
            got = poll_lang(page, target)
            settle(page, timeout=10)
            seen[target] = {"query": query, "option": label, "listed": listed[:200], "lang": got, "cookie": lang_cookie(ctx)}
        run.evidence["scenarios"]["palette"] = seen
        run.expect("switch: command palette action both ways",
                   seen["en"]["lang"] == "en" and seen["ru"]["lang"] == "ru" and seen["ru"]["cookie"] == "ru",
                   {k: {x: v[x] for x in ("option", "lang", "cookie")} for k, v in seen.items()})
    finally:
        ctx.close()


def scenario_login_links(run: Run, browser: Browser) -> None:
    """RU · EN links of /login (sign-in on): the click writes the cookie and the page reloads in that language."""
    sb = run.new_sb
    ctx = browser.context(sb, None)
    errors: list = []
    try:
        page, _ = browser.open(ctx, sb.base + "/login", errors)
        before = page.evaluate("document.documentElement.lang")
        links = page.locator(".lang-switch").inner_text()
        page.locator('.lang-switch [data-set-locale="en"]').click()
        after = poll_lang(page, "en")
        settle(page, timeout=8)
        title = page.title()
        page.locator('.lang-switch [data-set-locale="ru"]').click()
        back = poll_lang(page, "ru")
        run.evidence["scenarios"]["login_links"] = {"links": links, "before": before, "after": after, "title_en": title, "back": back}
        run.expect("switch: RU · EN links on /login", before == "ru" and after == "en" and back == "ru" and lang_cookie(ctx) == "ru"
                   and not CYRILLIC.search(title), {"links": links, "before": before, "after": after, "back": back, "title_en": title})
    finally:
        ctx.close()


def scenario_settings_language(run: Run, browser: Browser) -> None:
    """«Язык отчётов и промптов» = English saved on the settings page of an English UI."""
    sb = run.new_sb
    ctx = browser.context(sb, "en")
    errors: list = []
    try:
        page, _ = browser.open(ctx, sb.base + "/settings#appearance", errors)
        poll_text(page.locator("#sec-appearance"), lambda t: "Report and prompt language" in t, 10)
        page.locator("#reportLanguage").select_option("en")
        page.locator("#appearanceSave").click()
        status = poll_text(page.locator("#appearanceStatus"), lambda t: "Saved" in t, 15)
    finally:
        ctx.close()
    config = make_client(sb).get("/config").json()
    language = (config.get("appearance") or {}).get("language") if isinstance(config, dict) else None
    run.expect("settings: report language saved as en", "Saved" in status and language == "en", {"status": status, "appearance.language": language})


def _job_messages_ok(pairs: list) -> tuple[bool, dict]:
    running = [(p, en, ru) for p, en, ru in pairs if en and ru]
    en_bad = sorted({en for _p, en, _ru in running if CYRILLIC.search(en)})
    ru_cyr = sorted({ru for _p, _en, ru in running if CYRILLIC.search(ru)})
    distinct = sorted({en for _p, en, _ru in running})
    return (not en_bad and len(ru_cyr) >= 3 and len(distinct) >= 3), {"en_with_cyrillic": en_bad, "ru_samples": ru_cyr[:6], "en_samples": distinct[:8]}


def scenario_en_report(run: Run, browser: Browser, stamp: str) -> Optional[dict]:
    """The English report: job messages per reader, JSON, the LLM prompts, the publication."""
    sb = run.new_sb
    en, ru = make_client(sb, "en"), make_client(sb, "ru")
    log_path = run.out / "llm-requests-new.jsonl"
    llm_before = len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.exists() else 0
    errors: list = []
    ctx_en, ctx_ru = browser.context(sb, "en"), browser.context(sb, "ru")
    try:
        name = f"e2e-s9-en-{stamp}"
        accepted = en.create_report(service=SERVICE, window_min=60, test_type="step", run_name=name)
        job_id = accepted["job_id"]
        # The archive polls its jobs strip only when a job runs at page load: open it after the start.
        strip_en, _ = browser.open(ctx_en, sb.base + "/reports", errors, timeout=6)
        strip_ru, _ = browser.open(ctx_ru, sb.base + "/reports", errors, timeout=6)
        pairs, strips = [], []
        deadline = time.monotonic() + run.args.timeout
        final: dict = {}
        while time.monotonic() < deadline:
            a, b = en.job(job_id) or {}, ru.job(job_id) or {}
            pairs.append((a.get("progress"), a.get("message"), b.get("message")))
            try:
                strips.append((strip_en.locator("#jobsStripList").inner_text(timeout=2000), strip_ru.locator("#jobsStripList").inner_text(timeout=2000)))
            except Exception:  # noqa: BLE001 - the strip re-renders
                pass
            if a.get("status") != "running":
                final = a
                break
            strip_en.wait_for_timeout(700)
    finally:
        ctx_en.close()
        ctx_ru.close()
    run.evidence["scenarios"]["en_job"] = {"pairs": pairs, "strips": [s for s in strips if s[0] or s[1]][:12]}
    run.expect("en report: job done", final.get("status") == "done", {k: final.get(k) for k in ("status", "message", "error")})
    ok, detail = _job_messages_ok(pairs)
    run.expect("jobs: /job_status messages follow the reader (en without Cyrillic, ru in Russian)", ok, detail)
    with_job = [(e, r) for e, r in strips if name in e and name in r]
    strip_ok = bool(with_job) and not any(CYRILLIC.search(e.replace(name, "")) for e, _r in with_job) and any(CYRILLIC.search(r) for _e, r in with_job)
    run.expect("jobs: the archive jobs strip shows English for en and Russian for ru", strip_ok,
               {"samples": with_job[-2:] or strips[-2:]})
    if final.get("status") != "done":
        return None
    ref = report_ref(en, name)
    report = {"run_name": name, "service": SERVICE, "run_id": ref.get("run_id"), "job_id": job_id}
    rows = en.get(f"/llm_reports?run_name={quote(name)}&service={SERVICE}").json()
    final_row = next((r for r in rows if r.get("domain") == "final"), {}) if isinstance(rows, list) else {}
    parsed = final_row.get("parsed") if isinstance(final_row.get("parsed"), dict) else json.loads(final_row.get("parsed") or "{}")
    verdicts = sorted({str(r.get(k)) for r in rows for k in ("verdict", "llm_verdict", "sla_verdict") if r.get(k)})
    cyr = sorted({collapse(s)[:160] for s in strings_of(rows) if CYRILLIC.search(s)})
    sla_texts = [c.get("message") for c in (final_row.get("sla_details") or {}).get("checks", [])] + [(final_row.get("sla_details") or {}).get("summary")]
    run.evidence["scenarios"]["en_report_json"] = {"report_language": parsed.get("report_language"), "verdicts": verdicts,
                                                   "sla_texts": sla_texts, "cyrillic": cyr[:20]}
    run.expect("en report: JSON report_language en, verdict codes, no Cyrillic",
               parsed.get("report_language") == "en" and set(verdicts) <= {"pass", "risk", "fail", "no_data"} and bool(verdicts) and not cyr,
               {"report_language": parsed.get("report_language"), "verdicts": verdicts, "cyrillic": cyr[:5], "sla": sla_texts[:2]})
    lines = log_path.read_text(encoding="utf-8").splitlines()[llm_before:] if log_path.exists() else []
    calls = [json.loads(line) for line in lines if line.strip()]
    system_cyr = [c["n"] for c in calls for m in c["body"].get("messages", []) if m.get("role") == "system" and CYRILLIC.search(m.get("content") or "")]
    user_cyr = sorted({run_[:60] for c in calls for m in c["body"].get("messages", []) if m.get("role") != "system"
                       for run_ in CYR_RUN.findall(m.get("content") or "")})
    run.evidence["scenarios"]["en_prompts"] = {"calls": len(calls), "system_with_cyrillic": system_cyr, "user_cyrillic_runs": user_cyr[:30]}
    run.expect("en report: the LLM stub got English prompts", bool(calls) and not system_cyr and not user_cyr,
               {"calls": len(calls), "kinds": sorted({c.get("kind") for c in calls}), "system_with_cyrillic": system_cyr[:5], "user_cyrillic": user_cyr[:5]})
    return report


def publish(client: Client, run_: dict, port: int, timeout: float, attempts: int = 2) -> Optional[dict]:
    """Publishes a report to a freshly reset Confluence stub; returns the stub page with its body.

    One retry: the stub (http.server, small listen backlog) sometimes drops a connection of the parallel attachment
    uploads on this machine, and the job then fails with a connect timeout; that is not what this check is about.
    """
    for attempt in range(1, attempts + 1):
        stub_json(port, "/_stub/reset", "POST")
        response = client.post("/publish_confluence", {"run_name": run_["run_name"], "service": run_["service"]})
        job_id = (response.json() or {}).get("job_id") if response.status_code == 200 else None
        if not job_id:
            log(f"publish {run_['run_name']} -> {response.status_code}: {response.text[:200]}")
            return None
        final = client.wait_job(job_id, timeout)
        if final.get("status") == "done" and final.get("page_id"):
            return stub_json(port, f"/_stub/pages/{final['page_id']}")
        log(f"publish {run_['run_name']} attempt {attempt}: {final.get('status')} {str(final.get('error') or final.get('message'))[:200]}")
        time.sleep(15)
    return None


def page_body(page: Optional[dict]) -> str:
    return ((page or {}).get("body") or {}).get("storage", {}).get("value", "")


def body_lines(body: str, aliases: dict) -> list[str]:
    text = re.sub(r">\s*<", ">\n<", body)
    return normalize(text, aliases)


def scenario_publications(run: Run, ru_report: dict, en_report: Optional[dict]) -> None:
    """en page body without Cyrillic; ru page of NEW with the structure of OLD's publication of its ru report."""
    timeout = run.args.timeout
    if en_report:
        page = publish(make_client(run.new_sb, "en"), en_report, STUB_PORTS["new"]["confluence"], timeout)
        body = page_body(page)
        text = html_text(body)
        cyr = sorted({m[:80] for m in CYR_RUN.findall(text)})
        run.write("confluence/en.html", body)
        run.expect("confluence: en page without Cyrillic", bool(body) and not cyr, {"title": (page or {}).get("title"), "cyrillic": cyr[:8], "chars": len(body)})
    new_page = publish(make_client(run.new_sb), ru_report, STUB_PORTS["new"]["confluence"], timeout)
    if not run.use_old:
        run.expect("confluence: ru page published", bool(page_body(new_page)), (new_page or {}).get("title"))
        return
    old_page = publish(make_client(run.old_sb), run.runs["legacy_ru"], STUB_PORTS["old"]["confluence"], timeout)
    new_lines, old_lines = body_lines(page_body(new_page), run.aliases), body_lines(page_body(old_page), run.aliases)
    run.write("confluence/ru.new.html", page_body(new_page))
    run.write("confluence/ru.old.html", page_body(old_page))
    diff = [line for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0) if not line.startswith(("---", "+++", "@@"))]
    if diff:
        run.write("confluence/ru.diff", "\n".join(diff) + "\n")
    run.expect("confluence: ru page of NEW has the structure of OLD's", bool(new_lines) and bool(old_lines) and not diff,
               {"lines": [len(old_lines), len(new_lines)], "diff": diff[:10], "titles": [(old_page or {}).get("title"), (new_page or {}).get("title")]})


def html_text(body: str) -> str:
    import html as html_lib

    return html_lib.unescape(re.sub(r"<[^>]+>", " ", body or ""))


def reseed_demo(run: Run) -> None:
    """NEW writes its own demo run (the cloned one was written by OLD)."""
    client = make_client(run.new_sb, area="demo")
    client.delete_run("demo-checkout-step")
    response = client.post("/demo/seed", {})
    body = response.json() if response.status_code == 200 else {}
    run.runs["demo_id"] = str(body.get("report_url") or "").rsplit("/", 1)[-1] or run.runs.get("demo_id")
    run.expect("demo: NEW seeds its own demo run", response.status_code == 200 and bool(body.get("report_url")), {"status": response.status_code, "url": body.get("report_url")})


def setup_users(sb: Sandbox) -> None:
    """Bootstrap administrator signs in; a viewer is created for the forbidden page (sign-in on)."""
    admin = make_client(sb)
    login(admin, ADMIN_USER, ADMIN_PASSWORD)
    response = admin.session.post(sb.base + "/auth/users", json={"username": VIEWER_USER, "password": VIEWER_PASSWORD, "role": "viewer",
                                                                  "must_change_password": False}, headers={"X-CSRF-Token": admin.csrf}, timeout=30)
    if response.status_code not in (201, 409) and "exists" not in response.text:
        raise SystemExit(f"POST /auth/users -> {response.status_code}: {response.text[:300]}")


def mode_pages(mode: str) -> list[Page]:
    return [p for p in PAGES if p.mode == mode]


def check(args) -> int:
    from playwright.sync_api import sync_playwright

    state = load_state()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.log_dir) if args.log_dir else temp_dir() / "ll_e2e_jobs" / f"s9-{stamp}"
    run = Run(args, out)
    legacy = state["old_report"]
    run.runs.update(legacy_ru=legacy, legacy_ru_id=legacy["run_id"], legacy_ru_name=legacy["run_name"],
                    old_demo_id=state["demo"]["run_id"], demo_id=state["demo"]["run_id"])
    run.aliases.update({legacy["run_name"]: "<RU-REPORT>"})
    log(f"output: {out}")
    with Stack(run.new_sb, CHECK, str(out / "new")) as new_stack, Stack(run.old_sb, CHECK, str(out / "old")) as old_stack, \
            sync_playwright() as pw:
        browser = Browser(pw, out, args.headed)
        start_stubs(new_stack, "new", run.new_sb.app_dir, llm_log=out / "llm-requests-new.jsonl")
        if run.use_old:
            start_stubs(old_stack, "old", run.old_sb.app_dir)

        # ---- sign-in off ---------------------------------------------------------------------------------
        modes = set(filter(None, (args.modes or "").split(","))) or {"off", "auth", "down", "down_auth"}
        if "off" in modes:
            new_srv = start_server(new_stack)
            old_srv = start_server(old_stack) if run.use_old else None
            new_client = make_client(run.new_sb)
            set_report_language(new_client, "ru")  # an interrupted earlier run may have left English
            demo_ref = report_ref(make_client(run.new_sb, area="demo"), "demo-checkout-step")
            run.runs["demo_id"] = demo_ref.get("run_id") or run.runs["demo_id"]
            bench = [r for r in new_client.get(f"/runs?service={SERVICE}&limit=50").json() if str(r.get("run_name", "")).startswith("bench-")]
            run.runs["bench_name"] = bench[0]["run_name"] if bench else ""
            run.runs["lt_id"] = create_lt_run(new_client, f"e2e-s9-lt-{stamp}")
            if run.use_old:
                run.runs["old_lt_id"] = create_lt_run(make_client(run.old_sb), f"e2e-s9-lt-{stamp}")
            run.stored["legacy_ru"] = stored_texts(new_client, legacy)
            log("ru: NEW against OLD on the same data")
            for page_spec in mode_pages("off"):
                compare_ru(run, browser, page_spec, "same")
            scenario_verdict_sort(run)

            reseed_demo(run)
            run.stored["demo"] = stored_texts(make_client(run.new_sb, area="demo"),
                                              {"run_name": "demo-checkout-step", "service": state["demo"]["service"]})
            ru_name = f"e2e-s9-ru-{stamp}"
            built = build_report(new_client, ru_name, args.timeout)
            run.expect("ru report: built by NEW", built.get("status") == "done" and bool(built.get("run_id")), {k: built.get(k) for k in ("status", "message", "run_id")})
            ru_report = {"run_name": ru_name, "service": SERVICE, "run_id": built.get("run_id")}
            run.runs.update(ru=ru_report, ru_id=built.get("run_id"))
            run.aliases[ru_name] = "<RU-REPORT>"
            run.stored["ru"] = stored_texts(new_client, ru_report)
            rows = new_client.get(f"/llm_reports?run_name={quote(ru_name)}&service={SERVICE}").json()
            final_row = next((r for r in rows if r.get("domain") == "final"), {})
            parsed = final_row.get("parsed") if isinstance(final_row.get("parsed"), dict) else {}
            run.expect("ru report: JSON report_language ru and verdict codes", parsed.get("report_language") == "ru"
                       and str(final_row.get("verdict")) in ("pass", "risk", "fail", "no_data"),
                       {"report_language": parsed.get("report_language"), "verdict": final_row.get("verdict")})
            log("ru: NEW's own reports against OLD's")
            for page_spec in mode_pages("off"):
                compare_ru(run, browser, page_spec, "late")

            log("en: report language, English report, publications")
            scenario_settings_language(run, browser)
            en_report = scenario_en_report(run, browser, stamp)
            set_report_language(new_client, "ru")
            if en_report:
                run.runs.update(en=en_report, en_id=en_report.get("run_id"))
                run.aliases[en_report["run_name"]] = "<EN-REPORT>"
            scenario_publications(run, ru_report, en_report)
            log("switching the language")
            scenario_switch_menu(run, browser)
            scenario_palette(run, browser)
            if not args.no_matrix:
                log("matrix: sign-in off")
                matrix(run, browser, mode_pages("off"))
            if not args.keep_report:
                for report in (ru_report, en_report):
                    if report:
                        run.evidence.setdefault("deleted", {})[report["run_name"]] = new_client.delete_run(report["run_name"])
            stop_server(new_stack, new_srv)
            if old_srv:
                stop_server(old_stack, old_srv)

        # ---- sign-in on, storage down ---------------------------------------------------------------------
        for mode in ("auth", "down", "down_auth"):
            pages = [p for p in mode_pages(mode) if run.wanted(p)]
            if mode not in modes or (not pages and mode != "auth"):
                continue
            log(f"mode {mode}")
            new_srv = start_server(new_stack, mode)
            old_srv = start_server(old_stack, mode) if run.use_old else None
            if mode == "auth":
                setup_users(run.new_sb)
                if run.use_old:
                    setup_users(run.old_sb)
                scenario_login_links(run, browser)
            for page_spec in pages:
                compare_ru(run, browser, page_spec, "same")
            if not args.no_matrix:
                matrix(run, browser, pages)
            stop_server(new_stack, new_srv)
            if old_srv:
                stop_server(old_stack, old_srv)
        browser.close()
    run.evidence["runs"] = {k: v for k, v in run.runs.items() if not isinstance(v, dict)}
    run.evidence["output"] = str(out)
    run.write("evidence.json", json.dumps({"results": run.results, **run.evidence}, ensure_ascii=False, indent=2, default=str))
    summary = {"results": {k: v for k, v in run.results.items() if not v["ok"]} or "all ok",
               "matrix_cells": sum(len(v) for v in run.evidence["matrix"].values()), "output": str(out)}
    return finish(CHECK, all_ok(run.results), summary, args.json_path)


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.set_defaults(sandbox=NEW_N)
    # The sandbox numbers are fixed (NEW_N, OLD_N): --sandbox of the shared parser does not apply here.
    parser._option_string_actions["--sandbox"].help = argparse.SUPPRESS
    parser.add_argument("--prepare", action="store_true", help="build the NEW and OLD sandboxes and the OLD data, then exit")
    parser.add_argument("--old-ref", default="HEAD", help="git revision of the OLD sandbox (default HEAD)")
    parser.add_argument("--db-port", type=int, default=e2e_common.DB["port"], help="port of the local TimescaleDB")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for a report job")
    parser.add_argument("--pages", help="comma-separated page keys to check (default: all; see PAGES)")
    parser.add_argument("--no-old", action="store_true", help="skip OLD: no Russian comparison, no OLD servers")
    parser.add_argument("--no-matrix", action="store_true", help="skip the page matrix (scenarios and the ru comparison only)")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    parser.add_argument("--modes", help="comma-separated server modes to run: off, auth, down, down_auth (default: all)")
    args = parser.parse_args()
    e2e_common.DB["port"] = args.db_port
    if args.prepare:
        return prepare(args)
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
