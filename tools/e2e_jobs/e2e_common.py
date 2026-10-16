"""Shared helpers of the end-to-end checks of background jobs in tools/e2e_jobs/ (not a check itself).

Every check runs against a sandbox made by make_sandbox.py <N> (docs/perf-plan.md, section 1): the app copy
in %TEMP%/ll_ui_s<N>/app, its runner %TEMP%/ll_ui_s<N>/run_app.py on 127.0.0.1:<5070+N>, the database
ll_s<N> on 127.0.0.1:55432 and the Redis database <N> on 127.0.0.1:16379. Prepare it once as described in
docs/perf-baseline.md, "Этап 3" (seed, migrations, --runtime-only, --setup-stubs).

The checks start what they need themselves and stop it at the end:

* the sandbox server (run_app.py) in mode ``thread`` or ``celery`` (environment of the child process only);
* a Celery worker from the sandbox app directory: ``python -m celery -A loadlens_app.celery_app worker
  --pool=solo -Q reports,celery`` (on Windows only the solo pool works: one job at a time);
* the stubs of tools/stubs/ (LLM 5090, Confluence 5091, Prometheus 5092) unless they already listen.

Logs of the started processes go to %TEMP%/ll_e2e_jobs/<check>-<time>/ (never into the repository). Only
127.0.0.1 is contacted: HTTP to the sandbox and the stubs, SQL to the sandbox database (read-only), Redis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import quote, urlsplit

import requests

REPO = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO / "tools" / "stubs"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
DB = {"host": "127.0.0.1", "port": 55432, "user": "ltar", "password": "e2e-only-password"}
REDIS_PORT = 16379
SANDBOX_SECRET_KEY = "ui-sandbox-only-secret-key"
# Variables of the job runner a child process must not inherit from the shell that runs a check.
RUNNER_ENV_PREFIXES = ("CELERY_", "LOADLENS_JOB_", "LOADLENS_REPORT_", "LOADLENS_RESPONSE_CACHE")
STUBS = {
    "llm": ("llm_stub.py", 5090),
    "confluence": ("confluence_stub.py", 5091),
    "prom": ("prom_stub.py", 5092),
}
_CSRF_META = re.compile(r"""<meta\s+name=["']csrf-token["']\s+content=["']([^"']*)["']""", re.IGNORECASE)


def temp_dir() -> Path:
    return Path(os.environ.get("TEMP") or tempfile.gettempdir())


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_until(predicate: Callable[[], Any], timeout: float, interval: float = 0.5, what: str = "condition") -> Any:
    """Calls ``predicate`` until it returns something truthy; raises TimeoutError after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(interval)


class Sandbox:
    """Addresses and paths of sandbox ``n``."""

    def __init__(self, n: int):
        self.n = n
        self.port = 5070 + n
        self.base = f"http://127.0.0.1:{self.port}"
        self.root = temp_dir() / f"ll_ui_s{n}"
        self.app_dir = self.root / "app"
        self.runner = self.root / "run_app.py"
        self.redis_url = f"redis://127.0.0.1:{REDIS_PORT}/{n}"
        self.dbname = f"ll_s{n}"
        if not self.runner.is_file() or not (self.app_dir / "settings.py").is_file():
            raise SystemExit(f"sandbox {n} not found: run make_sandbox.py {n} and the preparation of docs/perf-baseline.md first")

    def env(self, mode: str, **extra: Any) -> dict:
        """Environment of a server or worker of this sandbox: ``mode`` is ``thread`` or ``celery``."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(RUNNER_ENV_PREFIXES)}
        env.update(LOADLENS_AUTH_ENABLED="0", LOADLENS_SECRET_KEY=SANDBOX_SECRET_KEY, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        if mode == "celery":
            env.update(
                CELERY_BROKER_URL=self.redis_url,
                CELERY_RESULT_BACKEND=self.redis_url,
                CELERY_TASK_ALWAYS_EAGER="0",
                LOADLENS_JOB_RUNNER="celery",
            )
        elif mode == "thread":
            env["LOADLENS_JOB_RUNNER"] = "thread"
        else:
            raise ValueError(f"unknown mode {mode!r}")
        env.update({key: str(value) for key, value in extra.items()})
        return env


class Proc:
    """A child process with its output in a log file; ``kill`` ends it at once (TerminateProcess on Windows)."""

    def __init__(self, name: str, args: list[str], cwd: Path, env: dict, log_path: Path):
        self.name, self.args, self.cwd, self.env, self.log_path = name, args, cwd, env, log_path
        self.popen: Optional[subprocess.Popen] = None

    def start(self) -> "Proc":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.log_path, "ab")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.popen = subprocess.Popen(self.args, cwd=str(self.cwd), env=self.env, stdout=handle, stderr=subprocess.STDOUT, creationflags=flags)
        handle.close()
        log(f"started {self.name}: pid {self.popen.pid}, log {self.log_path}")
        return self

    @property
    def pid(self) -> Optional[int]:
        return self.popen.pid if self.popen else None

    def alive(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

    def kill(self) -> None:
        if self.alive():
            self.popen.kill()
            self.popen.wait(timeout=30)
            log(f"stopped {self.name} (pid {self.popen.pid})")

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def log_lines(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return [line.strip() for line in self.log_text().splitlines() if rx.search(line)]


class Stack:
    """Processes of one check; ``with Stack(...) as stack:`` stops everything it started on exit."""

    def __init__(self, sandbox: Sandbox, check: str, log_dir: Optional[str] = None):
        self.sb = sandbox
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(log_dir) if log_dir else temp_dir() / "ll_e2e_jobs" / f"{check}-{stamp}"
        self.procs: list[Proc] = []
        self._count: dict[str, int] = {}

    def __enter__(self) -> "Stack":
        return self

    def __exit__(self, *exc: Any) -> None:
        for proc in reversed(self.procs):
            try:
                proc.kill()
            except Exception as error:  # noqa: BLE001 - keep stopping the rest
                log(f"could not stop {proc.name}: {error}")

    def _log_path(self, name: str) -> Path:
        self._count[name] = self._count.get(name, 0) + 1
        suffix = "" if self._count[name] == 1 else f"-{self._count[name]}"
        return self.log_dir / f"{name}{suffix}.log"

    def server(self, mode: str, **env: Any) -> Proc:
        """Starts the sandbox server and waits until /healthz answers."""
        if port_open(self.sb.port):
            raise SystemExit(f"port {self.sb.port} is busy: stop the sandbox server first (netstat -ano | grep :{self.sb.port})")
        proc = Proc("server", [sys.executable, "-X", "utf8", str(self.sb.runner)], self.sb.app_dir, self.sb.env(mode, **env), self._log_path("server")).start()
        self.procs.append(proc)

        def ready() -> bool:
            if not proc.alive():
                raise SystemExit(f"the server exited, see {proc.log_path}")
            try:
                return requests.get(self.sb.base + "/healthz", timeout=2).status_code == 200
            except requests.RequestException:
                return False

        wait_until(ready, 90, what="the sandbox server")
        return proc

    def worker(self, **env: Any) -> Proc:
        """Starts a solo Celery worker from the sandbox app directory and waits until it is ready."""
        args = [
            sys.executable, "-X", "utf8", "-m", "celery", "-A", "loadlens_app.celery_app", "worker",
            "--loglevel=info", "--pool=solo", "-Q", "reports,celery", "-n", f"e2e-s{self.sb.n}@%h",
        ]
        proc = Proc("worker", args, self.sb.app_dir, self.sb.env("celery", **env), self._log_path("worker")).start()
        self.procs.append(proc)

        def ready() -> bool:
            if not proc.alive():
                raise SystemExit(f"the worker exited, see {proc.log_path}")
            return " ready." in proc.log_text()

        wait_until(ready, 90, what="the Celery worker")
        return proc

    def stubs(self, llm_delay: float = 2.0, **llm_args: Any) -> None:
        """Starts the stubs that do not listen yet. A running LLM stub must have the wanted delay."""
        for name, (script, port) in STUBS.items():
            if port_open(port):
                if name == "llm":
                    stats = stub_json(port, "/_stub/stats") or {}
                    if abs(float(stats.get("delay_sec") or 0) - llm_delay) > 1e-6:
                        raise SystemExit(f"the LLM stub on {port} answers in {stats.get('delay_sec')} s, this check needs {llm_delay} s: stop it")
                continue
            args = [sys.executable, "-X", "utf8", str(STUBS_DIR / script)]
            if name == "llm":
                args += ["--delay", str(llm_delay)]
                for key, value in llm_args.items():
                    args += [f"--{key.replace('_', '-')}", str(value)]
            proc = Proc(f"{name}-stub", args, REPO, dict(os.environ), self._log_path(f"{name}-stub")).start()
            self.procs.append(proc)
            wait_until(lambda: port_open(port), 30, what=f"the {name} stub on {port}")

    def stop(self, proc: Proc) -> None:
        proc.kill()


def stub_json(port: int, path: str, method: str = "GET") -> Optional[Any]:
    try:
        response = requests.request(method, f"http://127.0.0.1:{port}{path}", timeout=5, proxies={"http": "", "https": ""})
        return response.json()
    except (requests.RequestException, ValueError):
        return None


# ---- HTTP client of the sandbox ------------------------------------------------------------------------


class Client:
    """requests session of the sandbox: area cookie, CSRF token of /new, JSON helpers."""

    def __init__(self, sandbox: Sandbox, area: str = "bench"):
        if (urlsplit(sandbox.base).hostname or "") not in LOCAL_HOSTS:
            raise SystemExit("the e2e checks talk to 127.0.0.1 only")
        self.sb = sandbox
        self.session = requests.Session()
        self.session.trust_env = False  # never through a proxy of the environment
        if area:
            self.session.cookies.set("project_area", area, domain="127.0.0.1")
        self.area = area
        self.csrf = ""

    def refresh_csrf(self) -> None:
        page = self.session.get(self.sb.base + "/new", timeout=30).text
        match = _CSRF_META.search(page)
        self.csrf = match.group(1) if match else ""

    def _headers(self) -> dict:
        return {"X-CSRF-Token": self.csrf} if self.csrf else {}

    def get(self, path: str, timeout: float = 30) -> requests.Response:
        return self.session.get(self.sb.base + path, timeout=timeout)

    def post(self, path: str, payload: dict, timeout: float = 60) -> requests.Response:
        if not self.csrf:
            self.refresh_csrf()
        return self.session.post(self.sb.base + path, json=payload, headers=self._headers(), timeout=timeout)

    def delete(self, path: str, timeout: float = 60) -> requests.Response:
        if not self.csrf:
            self.refresh_csrf()
        return self.session.delete(self.sb.base + path, headers=self._headers(), timeout=timeout)

    def create_report(self, service: str = "svc01", window_min: int = 60, test_type: str = "step", run_name: str = "", use_llm: bool = True) -> dict:
        """POST /create_report like the "Новый отчёт" page (web report); returns the accepted JSON with the timing."""
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = end - timedelta(minutes=window_min)
        payload = {
            "start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"),
            "service": service, "project_area": self.area, "test_type": test_type, "use_llm": use_llm,
            "save_to_db": True, "web_only": True,
            "run_name": run_name or datetime.now().strftime("e2e-%Y%m%d-%H%M%S-%f")[:-3],
        }
        began = time.perf_counter()
        response = self.post("/create_report", payload)
        elapsed = time.perf_counter() - began
        body = response.json() if response.headers.get("Content-Type", "").startswith("application/json") else {}
        if response.status_code != 200 or not body.get("job_id"):
            raise SystemExit(f"POST /create_report -> {response.status_code}: {response.text[:300]}")
        body["_request_sec"] = round(elapsed, 3)
        log(f"report job {body['job_id']} accepted in {elapsed:.2f} s (run {body.get('run_name')})")
        return body

    def job(self, job_id: str) -> Optional[dict]:
        response = self.get("/job_status/" + quote(job_id, safe=""))
        return response.json() if response.status_code == 200 else None

    def wait_job(self, job_id: str, timeout: float, poll: float = 1.5, until: Optional[Callable[[dict], bool]] = None,
                 on_poll: Optional[Callable[[dict], None]] = None) -> dict:
        """Polls /job_status like the UI until ``until(job)`` (default: not running); returns the last state."""
        until = until or (lambda job: job.get("status") != "running")
        deadline = time.monotonic() + timeout
        last: dict = {}
        while True:
            job = self.job(job_id) or {"status": "not_found"}
            if on_poll:
                on_poll(job)
            if (job.get("status"), job.get("progress"), job.get("message")) != (last.get("status"), last.get("progress"), last.get("message")):
                log(f"  job {job_id[:8]}: {job.get('status')} {job.get('progress')}% {job.get('message')}")
            last = job
            if job.get("status") == "not_found" or until(job):
                return job
            if time.monotonic() > deadline:
                raise TimeoutError(f"job {job_id} still {job.get('status')} after {timeout:.0f} s")
            time.sleep(poll)

    def delete_run(self, run_name: str) -> int:
        return self.delete("/runs/" + quote(run_name, safe="")).status_code


# ---- database and broker ---------------------------------------------------------------------------------

JOB_COLUMNS = (
    "job_id", "kind", "run_name", "service", "status", "progress", "message", "error", "page_url", "page_id",
    "owner", "runner", "created_at", "started_at", "updated_at",
)


def db_connect(sandbox: Sandbox):
    import psycopg2

    return psycopg2.connect(dbname=sandbox.dbname, connect_timeout=5, **DB)


def job_row(sandbox: Sandbox, job_id: str) -> Optional[dict]:
    """The row of report_jobs (read-only), with the times as ISO strings."""
    connection = db_connect(sandbox)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT {', '.join(JOB_COLUMNS)} FROM report_jobs WHERE job_id = %s", (job_id,))
            row = cursor.fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    out = dict(zip(JOB_COLUMNS, row))
    for key in ("created_at", "started_at", "updated_at"):
        if out[key] is not None:
            out[key] = out[key].astimezone(timezone.utc).isoformat(timespec="milliseconds")
    return out


def seconds_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if not a or not b:
        return None
    return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 2)


def redis_queue_length(sandbox: Sandbox, queue: str = "reports") -> Optional[int]:
    try:
        import redis

        return int(redis.Redis.from_url(sandbox.redis_url, socket_timeout=2, socket_connect_timeout=2).llen(queue))
    except Exception:  # noqa: BLE001 - only evidence
        return None


def redis_ping(timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", REDIS_PORT), timeout=timeout) as sock:
            sock.sendall(b"PING\r\n")
            return sock.recv(16).startswith(b"+PONG")
    except OSError:
        return False


# ---- running a check -----------------------------------------------------------------------------------


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sandbox", type=int, default=6, help="sandbox number N (port 5070+N, database ll_s<N>, Redis db N)")
    parser.add_argument("--log-dir", help="directory for the process logs (default %%TEMP%%/ll_e2e_jobs/<check>-<time>)")
    parser.add_argument("--json", dest="json_path", help="also write the evidence to this JSON file")
    parser.add_argument("--keep-report", action="store_true", help="do not delete the report runs the check created")
    return parser


def finish(check: str, passed: bool, evidence: dict, json_path: Optional[str] = None) -> int:
    """Prints the verdict and the evidence; the exit code is 0 when the check passed."""
    evidence = {"check": check, "passed": passed, **evidence}
    text = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
    print(text)
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(text, encoding="utf-8")
    log(f"{check}: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def expect(results: dict, name: str, ok: bool, detail: Any = None) -> bool:
    """Records one expectation of a check and logs it."""
    results[name] = {"ok": bool(ok), "detail": detail}
    log(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return bool(ok)


def all_ok(results: dict) -> bool:
    return all(item["ok"] for item in results.values())


__all__: Iterable[str] = [
    "Client", "Proc", "Sandbox", "Stack", "all_ok", "base_parser", "db_connect", "expect", "finish", "job_row",
    "log", "port_open", "redis_ping", "redis_queue_length", "seconds_between", "stub_json", "wait_until",
]
