"""GitLab API stub for the CI generator of the Run test page. Dev stack only.

Runs on the host (127.0.0.1:5095) and answers the GitLab API calls LoadLens makes: version, project, branches,
the pipeline parameters (GraphQL ciConfigVariables, CI lint with the merged YAML, the .gitlab-ci.yml file with
spec:inputs), creating a pipeline with variables and inputs (or by the trigger token), its status, cancel and
the job log. A pipeline is one k6 container of the dev stack (stub_job.py).

The behaviour follows the version in STUB_GITLAB_VERSION (default 17.11.0), so LoadLens can be checked
against older GitLab too:
  - < 15.3: no ciConfigVariables in GraphQL (LoadLens falls back to the lint API);
  - 15.3–15.6: ciConfigVariables(sha:) without valueOptions;
  - >= 15.7: ciConfigVariables(ref:) with valueOptions;
  - >= 17.11: spec:inputs in the configuration and in the API that creates a pipeline.
GraphQL answers null to the first question of a ref, as GitLab does while it computes the variables.
STUB_FORBID_VARIABLES=1 refuses pipeline variables like a project with "no one allowed" (GitLab 17.7+).

    python docker/dev-stack/ci/gitlab_stub.py

Generator in LoadLens: type gitlab, url http://127.0.0.1:5095, project perf/load, API token api-secret
(trigger token trig-secret without one), LoadLens URL for events http://host.docker.internal:<port of LoadLens>.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import stub_job

TRIGGER_TOKEN = "trig-secret"
API_TOKEN = "api-secret"
PORT = 5095
VERSION = os.environ.get("STUB_GITLAB_VERSION", "17.11.0")
FORBID_VARIABLES = os.environ.get("STUB_FORBID_VARIABLES") == "1"
BRANCHES = ["main", "release"]
pipelines = {}
asked = set()
lock = threading.Lock()
counter = [100]


def version() -> tuple:
    return tuple(int(part) for part in VERSION.split("-")[0].split(".")[:2])


def config(ref: str) -> str:
    """.gitlab-ci.yml of the branch: spec:inputs only where GitLab knows them, options since 15.7."""
    options = version() >= (15, 7)
    header = """spec:
  inputs:
    environment:
      default: stage
      options: [stage, preprod]
      description: Стенд
    users:
      type: number
      default: 2
      description: Виртуальные пользователи k6
    send_events:
      type: boolean
      default: false
      description: Присылать started и finished в LoadLens
    loadlens_run_tag:
      default: ""
---
""" if version() >= (17, 11) else ""
    duration = "45s" if ref == "release" else "60s"
    return header + f"""variables:
  PROFILE:
    value: load
{'    options: [smoke, load, stress]' if options else ''}
    description: Профиль нагрузки (RPS 2, 5 или 10)
  DURATION:
    value: {duration}
    description: Длительность k6
  TAG_METRICS:
    value: "true"
{'    options: ["true", "false"]' if options else ''}
    description: Помечать метрики тегом LOADLENS_RUN_TAG
  SETUP_SEC:
    value: "20"
    description: Подготовка перед тестом, с
  INTERNAL_ONLY: not-shown
load-test:
  image: grafana/k6:0.54.0
  script:
    - k6 run /scripts/k6/load.js
"""


def described_variables(ref: str, with_options: bool) -> list:
    import yaml

    docs = [doc for doc in yaml.safe_load_all(config(ref)) if doc]
    variables = next(doc["variables"] for doc in docs if "variables" in doc)
    out = []
    for key, value in variables.items():
        if isinstance(value, dict) and value.get("description"):
            item = {"key": key, "value": str(value.get("value", "")), "description": value["description"]}
            if with_options:
                item["valueOptions"] = [str(option) for option in value.get("options") or []]
            out.append(item)
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_pipeline(pid: int, values: dict) -> None:
    name = f"ll-ci-gl-{pid}"
    time.sleep(2)  # waiting for a runner
    with lock:
        if pipelines[pid]["status"] == "canceled":
            return
        pipelines[pid].update(status="running", started_at=now_iso())
    code = stub_job.start(name, values).wait()
    trace = stub_job.logs(name)
    stub_job.remove(name)
    with lock:
        p = pipelines[pid]
        p["trace"] = trace
        p["finished_at"] = now_iso()
        if p["status"] != "canceled":
            p["status"] = "success" if code == 0 else "failed"


def create(ref: str, variables: dict, inputs: dict) -> dict:
    with lock:
        counter[0] += 1
        pid = counter[0]
        pipelines[pid] = {"status": "created", "ref": ref, "variables": variables, "inputs": inputs,
                          "created_at": now_iso(), "started_at": None, "finished_at": None, "trace": ""}
    # Inputs win over variables of the same meaning, as in a real pipeline that maps them into the job.
    threading.Thread(target=run_pipeline, args=(pid, {**variables, **inputs}), daemon=True).start()
    return view(pid)


def view(pid: int) -> dict:
    p = pipelines[pid]
    return {"id": pid, "status": p["status"], "ref": p["ref"], "web_url": f"http://127.0.0.1:{PORT}/perf/load/-/pipelines/{pid}",
            "created_at": p["created_at"], "started_at": p["started_at"], "finished_at": p["finished_at"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(time.strftime("%H:%M:%S"), self.command, self.path.split("?")[0], fmt % args, flush=True)

    def _send(self, status, payload, text=False):
        body = payload.encode() if text else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8" if text else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        bearer = self.headers.get("Authorization", "")
        return self.headers.get("PRIVATE-TOKEN") == API_TOKEN or bearer == f"Bearer {API_TOKEN}"

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode()
        if "json" in (self.headers.get("Content-Type") or ""):
            return json.loads(raw or "{}")
        form = parse_qs(raw)
        return {key: values[0] for key, values in form.items()}

    def do_GET(self):
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        parts = [unquote(part) for part in url.path.strip("/").split("/")]
        if url.path == "/api/v4/version":
            return self._send(200, {"version": VERSION}) if self._authorized() else self._send(401, {"message": "401 Unauthorized"})
        if not self._authorized():
            return self._send(401, {"message": "401 Unauthorized"})
        if parts[:3] != ["api", "v4", "projects"] or len(parts) < 4 or parts[3] not in ("perf/load", "1"):
            return self._send(404, {"message": "404 Project Not Found"})
        rest = parts[4:]
        if not rest:
            return self._send(200, {"id": 1, "path_with_namespace": "perf/load", "default_branch": "main", "ci_config_path": ""})
        if rest == ["repository", "branches"]:
            search = query.get("search", "")
            return self._send(200, [{"name": name} for name in BRANCHES if search in name])
        if rest[:2] == ["repository", "branches"] and len(rest) == 3:
            return self._send(200, {"name": rest[2], "commit": {"id": f"sha-{rest[2]}"}}) if rest[2] in BRANCHES else self._send(404, {})
        if rest[:2] == ["repository", "files"] and rest[-1] == "raw":
            ref = query.get("ref", "main")
            return self._send(200, config(ref), text=True) if ref in BRANCHES else self._send(404, {"message": "404 File Not Found"})
        if rest == ["ci", "lint"]:
            if version() < (13, 0):
                return self._send(404, {"message": "404"})
            import yaml

            docs = [doc for doc in yaml.safe_load_all(config(query.get("content_ref") or query.get("ref") or "main")) if doc]
            merged = yaml.safe_dump(next(doc for doc in docs if "variables" in doc), allow_unicode=True)
            return self._send(200, {"valid": True, "errors": [], "merged_yaml": merged})
        if rest[0] == "pipelines" and len(rest) == 2:
            pid = int(rest[1])
            return self._send(200, view(pid)) if pid in pipelines else self._send(404, {"message": "404"})
        if rest[0] == "pipelines" and len(rest) == 3 and rest[2] == "jobs":
            return self._send(200, [{"id": int(rest[1]), "name": "load-test"}])
        if rest[0] == "jobs" and len(rest) == 3 and rest[2] == "trace":
            pid = int(rest[1])
            p = pipelines.get(pid)
            if p and p["status"] == "running":
                return self._send(200, stub_job.logs(f"ll-ci-gl-{pid}"), text=True)
            return self._send(200, (p or {}).get("trace", ""), text=True)
        return self._send(404, {"message": "404"})

    def _graphql(self, body):
        query = body.get("query", "")
        variables = body.get("variables") or {}
        if version() < (15, 3):
            return self._send(200, {"errors": [{"message": "Field 'ciConfigVariables' doesn't exist on type 'Project'"}]})
        by_ref = version() >= (15, 7)
        if ("ciConfigVariables(ref:" in query) != by_ref:
            argument = "ref" if "ciConfigVariables(ref:" in query else "sha"
            return self._send(200, {"errors": [{"message": f"Field 'ciConfigVariables' doesn't accept argument '{argument}'"}]})
        if "valueOptions" in query and not by_ref:
            return self._send(200, {"errors": [{"message": "Field 'valueOptions' doesn't exist on type 'CiConfigVariable'"}]})
        value = str(variables.get("value") or "")
        ref = value[len("sha-"):] if value.startswith("sha-") else value
        key = (ref, by_ref)
        with lock:
            first = key not in asked
            asked.add(key)
        if first:
            return self._send(200, {"data": {"project": {"ciConfigVariables": None}}})
        return self._send(200, {"data": {"project": {"ciConfigVariables": described_variables(ref, "valueOptions" in query)}}})

    def do_POST(self):
        url = urlparse(self.path)
        parts = [unquote(part) for part in url.path.strip("/").split("/")]
        body = self._body()
        if url.path == "/api/graphql":
            return self._graphql(body) if self._authorized() else self._send(401, {"message": "401"})
        rest = parts[4:]
        if rest == ["trigger", "pipeline"]:
            if body.get("token") != TRIGGER_TOKEN:
                return self._send(404, {"message": "404 Not Found"})
            variables = body.get("variables") if isinstance(body.get("variables"), dict) else \
                {k[len("variables["):-1]: v for k, v in body.items() if k.startswith("variables[")}
            if FORBID_VARIABLES and variables:
                return self._send(400, {"message": {"base": ["Insufficient permissions to set pipeline variables"]}})
            inputs = body.get("inputs") if version() >= (17, 11) and isinstance(body.get("inputs"), dict) else {}
            return self._send(201, create(body.get("ref", "main"), variables, inputs))
        if not self._authorized():
            return self._send(401, {"message": "401 Unauthorized"})
        if rest == ["pipeline"]:
            ref = body.get("ref") or "main"
            if ref not in BRANCHES:
                return self._send(400, {"message": {"base": ["Reference not found"]}})
            variables = {item["key"]: item["value"] for item in body.get("variables") or []}
            if FORBID_VARIABLES and variables:
                return self._send(400, {"message": {"base": ["Insufficient permissions to set pipeline variables"]}})
            # Older GitLab ignores the parameter it does not know.
            inputs = body.get("inputs") if version() >= (17, 11) and isinstance(body.get("inputs"), dict) else {}
            return self._send(201, create(ref, variables, inputs))
        if rest[0] == "pipelines" and len(rest) == 3 and rest[2] == "cancel":
            pid = int(rest[1])
            with lock:
                if pid not in pipelines:
                    return self._send(404, {"message": "404"})
                pipelines[pid]["status"] = "canceled"
            stub_job.kill(f"ll-ci-gl-{pid}")
            return self._send(200, view(pid))
        return self._send(404, {"message": "404"})


if __name__ == "__main__":
    print(f"GitLab stub {VERSION} on 127.0.0.1:{PORT}" + (" (pipeline variables forbidden)" if FORBID_VARIABLES else ""), flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
