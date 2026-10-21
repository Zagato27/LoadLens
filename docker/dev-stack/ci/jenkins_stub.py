# SPDX-License-Identifier: Apache-2.0
"""Jenkins API stub for the CI generator of the Run test page. Dev stack only.

Runs on the host (127.0.0.1:5096) with basic auth bot / JT and two jobs:
  - perf/load-test — a freestyle job with string, choice, boolean, password, text, file and Active Choices
    parameters (STAND returns a literal list in config.xml, HOST is computed from STAND);
  - perf/load-mb — a multibranch job with the branches main and release (a pipeline job each).
A build waits in the queue for two seconds and then runs one k6 container of the dev stack (stub_job.py).
Queue and build URLs name "http://jenkins.internal:8080", like a Jenkins whose "Jenkins URL" setting is
another host: LoadLens must keep to its own address.

    python docker/dev-stack/ci/jenkins_stub.py

Generator in LoadLens: type jenkins, url http://127.0.0.1:5096, job perf/load-test (or perf/load-mb), user bot,
API token JT, LoadLens URL for events http://host.docker.internal:<port of LoadLens>.
"""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import stub_job

PORT = 5096
PUBLIC = "http://jenkins.internal:8080"
AUTH = "Basic " + base64.b64encode(b"bot:JT").decode()
FREESTYLE = ["perf", "load-test"]
MULTIBRANCH = ["perf", "load-mb"]
BRANCHES = ["main", "release"]

PARAMETERS = [
    {"_class": "hudson.model.ChoiceParameterDefinition", "name": "PROFILE", "type": "ChoiceParameterDefinition",
     "description": "Профиль нагрузки (RPS 2, 5 или 10)", "choices": ["load", "smoke", "stress"], "defaultParameterValue": {"value": "load"}},
    {"_class": "hudson.model.StringParameterDefinition", "name": "VUS", "type": "StringParameterDefinition",
     "description": "Виртуальные пользователи k6", "defaultParameterValue": {"value": "2"}},
    {"_class": "hudson.model.StringParameterDefinition", "name": "DURATION", "type": "StringParameterDefinition",
     "defaultParameterValue": {"value": "60s"}},
    {"_class": "hudson.model.BooleanParameterDefinition", "name": "TAG_METRICS", "type": "BooleanParameterDefinition",
     "description": "Помечать метрики тегом LOADLENS_RUN_TAG", "defaultParameterValue": {"value": True}},
    {"_class": "hudson.model.BooleanParameterDefinition", "name": "SEND_EVENTS", "type": "BooleanParameterDefinition",
     "description": "Присылать started и finished в LoadLens", "defaultParameterValue": {"value": False}},
    {"_class": "hudson.model.PasswordParameterDefinition", "name": "API_KEY", "type": "PasswordParameterDefinition",
     "description": "Ключ тестируемого API", "defaultParameterValue": {"value": "<concealed>"}},
    {"_class": "hudson.model.TextParameterDefinition", "name": "NOTES", "type": "TextParameterDefinition",
     "defaultParameterValue": {"value": ""}},
    {"_class": "org.biouno.unochoice.ChoiceParameter", "name": "STAND", "type": "ChoiceParameter", "description": "Стенд"},
    {"_class": "org.biouno.unochoice.CascadeChoiceParameter", "name": "HOST", "type": "CascadeChoiceParameter"},
    {"_class": "hudson.model.FileParameterDefinition", "name": "DATA", "type": "FileParameterDefinition"},
    {"_class": "hudson.model.StringParameterDefinition", "name": "SETUP_SEC", "type": "StringParameterDefinition",
     "defaultParameterValue": {"value": "20"}},
    {"_class": "hudson.model.StringParameterDefinition", "name": "LOADLENS_RUN_TAG", "type": "StringParameterDefinition",
     "defaultParameterValue": {"value": ""}},
    {"_class": "hudson.model.StringParameterDefinition", "name": "LOADLENS_EVENTS_URL", "type": "StringParameterDefinition",
     "defaultParameterValue": {"value": ""}},
    {"_class": "hudson.model.PasswordParameterDefinition", "name": "LOADLENS_TOKEN", "type": "PasswordParameterDefinition",
     "defaultParameterValue": {"value": "<concealed>"}},
]
CONFIG_XML = """<?xml version='1.1' encoding='UTF-8'?>
<project><properties><hudson.model.ParametersDefinitionProperty><parameterDefinitions>
<org.biouno.unochoice.ChoiceParameter plugin="uno-choice@2.8.3"><name>STAND</name><choiceType>PT_SINGLE_SELECT</choiceType>
<script class="org.biouno.unochoice.model.GroovyScript"><secureScript plugin="script-security@1335"><script>return ['stage:selected', 'preprod', 'prod:disabled']</script><sandbox>true</sandbox></secureScript></script>
</org.biouno.unochoice.ChoiceParameter>
<org.biouno.unochoice.CascadeChoiceParameter plugin="uno-choice@2.8.3"><name>HOST</name><choiceType>PT_CHECKBOX</choiceType><referencedParameters>STAND</referencedParameters>
<script class="org.biouno.unochoice.model.GroovyScript"><secureScript><script>return STAND == 'stage' ? ['st-1', 'st-2'] : ['pp-1']</script></secureScript></script>
</org.biouno.unochoice.CascadeChoiceParameter>
</parameterDefinitions></hudson.model.ParametersDefinitionProperty></properties></project>"""

queue = {}
builds = {}
lock = threading.Lock()
counter = [500]


def job_of(parts):
    """(job key, rest of the path) of /job/perf/job/load-test/... or /job/perf/job/load-mb/job/main/..."""
    names = []
    i = 0
    while i + 1 < len(parts) and parts[i] == "job":
        names.append(parts[i + 1])
        i += 2
    return "/".join(names), parts[i:]


def job_json(job):
    if job == "/".join(MULTIBRANCH):
        return {"_class": "org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject", "fullName": job,
                "jobs": [{"name": name} for name in BRANCHES]}
    if job == "/".join(FREESTYLE):
        cls = "hudson.model.FreeStyleProject"
    elif job.startswith("/".join(MULTIBRANCH) + "/") and job.rsplit("/", 1)[-1] in BRANCHES:
        cls = "org.jenkinsci.plugins.workflow.job.WorkflowJob"
    else:
        return None
    return {"_class": cls, "fullName": job, "property": [{"_class": "hudson.model.ParametersDefinitionProperty", "parameterDefinitions": PARAMETERS}]}


def run_build(number, values):
    name = f"ll-ci-jk-{number}"
    with lock:
        build = builds[number]
        if build["result"] == "ABORTED":
            return
        build["timestamp"] = int(time.time() * 1000)
    code = stub_job.start(name, values).wait()
    log = stub_job.logs(name)
    stub_job.remove(name)
    with lock:
        build = builds[number]
        build["log"] = log
        build["building"] = False
        build["duration"] = int(time.time() * 1000) - build["timestamp"]
        if build["result"] != "ABORTED":
            build["result"] = "SUCCESS" if code == 0 else "FAILURE"


def schedule(item):
    time.sleep(2)
    with lock:
        entry = queue[item]
        if entry.get("cancelled"):
            return
        counter[0] += 1
        number = counter[0]
        entry["number"] = number
        builds[number] = {"job": entry["job"], "building": True, "result": None, "timestamp": 0, "duration": 0, "log": ""}
    run_build(number, entry["values"])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(time.strftime("%H:%M:%S"), self.command, self.path.split("?")[0], fmt % args, flush=True)

    def _send(self, status, payload=None, text=None, headers=None):
        body = text.encode() if text is not None else json.dumps(payload if payload is not None else {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8" if text is not None else "application/json")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parts(self):
        url = urlparse(self.path)
        return [unquote(part) for part in url.path.strip("/").split("/") if part], {k: v[0] for k, v in parse_qs(url.query).items()}

    def do_GET(self):
        if self.headers.get("Authorization") != AUTH:
            return self._send(401, {"message": "Unauthorized"})
        parts, _query = self._parts()
        if parts[:2] == ["queue", "item"] and len(parts) >= 3:
            entry = queue.get(int(parts[2]))
            if entry is None:
                return self._send(404)
            if entry.get("cancelled"):
                return self._send(200, {"cancelled": True, "executable": None})
            number = entry.get("number")
            executable = {"number": number, "url": f"{PUBLIC}/job/{'/job/'.join(entry['job'].split('/'))}/{number}/"} if number else None
            return self._send(200, {"cancelled": False, "executable": executable})
        job, rest = job_of(parts)
        if rest and rest[0].isdigit():
            build = builds.get(int(rest[0]))
            if build is None or build["job"] != job:
                return self._send(404)
            if rest[1:] == ["api", "json"]:
                return self._send(200, {"building": build["building"], "result": build["result"], "timestamp": build["timestamp"], "duration": build["duration"]})
            if rest[1:] == ["consoleText"]:
                text = stub_job.logs(f"ll-ci-jk-{rest[0]}") if build["building"] else build["log"]
                return self._send(200, text=text)
            return self._send(404)
        data = job_json(job)
        if data is None:
            return self._send(404)
        if rest == ["api", "json"]:
            return self._send(200, data)
        if rest == ["config.xml"] and "property" in data:
            return self._send(200, text=CONFIG_XML)
        return self._send(404)

    def do_POST(self):
        if self.headers.get("Authorization") != AUTH:
            return self._send(401, {"message": "Unauthorized"})
        parts, query = self._parts()
        length = int(self.headers.get("Content-Length") or 0)
        form = {key: values[0] for key, values in parse_qs(self.rfile.read(length).decode()).items()}
        if parts == ["queue", "cancelItem"]:
            with lock:
                entry = queue.get(int(query.get("id", "0")))
                if entry is not None and not entry.get("number"):
                    entry["cancelled"] = True
            return self._send(204, text="")
        job, rest = job_of(parts)
        if rest and rest[0].isdigit() and rest[1:] == ["stop"]:
            number = int(rest[0])
            with lock:
                if number in builds and builds[number]["building"]:
                    builds[number]["result"] = "ABORTED"
            stub_job.kill(f"ll-ci-jk-{number}")
            return self._send(200, text="")
        data = job_json(job)
        if data is None or "property" not in data or rest not in (["buildWithParameters"], ["build"]):
            return self._send(404)
        defaults = {p["name"]: (p.get("defaultParameterValue") or {}).get("value") for p in PARAMETERS}
        defaults = {k: ("" if v == "<concealed>" else v) for k, v in defaults.items()}
        values = {**defaults, **{k: v for k, v in form.items() if k in defaults}}  # Jenkins drops undeclared parameters
        with lock:
            counter[0] += 1
            item = counter[0]
            queue[item] = {"job": job, "values": values}
        threading.Thread(target=schedule, args=(item,), daemon=True).start()
        return self._send(201, text="", headers={"Location": f"{PUBLIC}/queue/item/{item}/"})


if __name__ == "__main__":
    print(f"Jenkins stub on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
