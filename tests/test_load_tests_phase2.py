"""Phase 2 of the Run test page: Docker, Kubernetes and CI generators, soft stop, presets, hints, summed RPS."""

import copy
import hashlib
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from loadlens_app import core
from loadlens_app.load_tests import executors, presets, service, store, watcher
from loadlens_app.load_tests.executors import (
    ApiError,
    CiExecutor,
    DockerExecutor,
    GeneratorError,
    KubernetesExecutor,
    LaunchPlan,
    LaunchResult,
    Probe,
    container_name,
    demux,
    job_name,
)
from loadlens_app.load_tests.generators import GeneratorConfigError, configured_generators, parse_generator
from loadlens_app.load_tests.store import InstanceRecord, MemoryLoadTestStore, RunRecord
from loadlens_app.load_tests.tools import CommandLine, RenderContext, control_port, effective_tools, exe_dir, render_aux_command

DOCKER = {
    "title": "Docker", "type": "docker",
    "docker": {"url": "https://gen2:2376", "client_cert": "CERT", "client_key": "KEY", "ca_cert": "CA"},
    "network": "loadtest", "mounts": [{"source": "scripts", "target": "/scripts", "read_only": True}],
    "scripts_dir": "/scripts", "resources": {"cpus": 2, "memory_mb": 1024}, "tools": {},
}
K8S = {
    "title": "K8s", "type": "kubernetes",
    "kubernetes": {"api_url": "https://k8s:6443", "token": "TOKEN", "ca_cert": "CA"},
    "namespace": "loadtest", "scripts": {"config_map": "scripts"}, "node_selector": {"role": "loadgen"},
    "resources": {"cpu": "2", "memory": "2Gi"},
}
GITLAB = {
    "title": "GitLab", "type": "gitlab",
    "gitlab": {"url": "https://gitlab.example", "project": "perf/load", "ref": "main", "trigger_token": "TRIG", "api_token": "API"},
    "loadlens_url": "https://loadlens.example",
}
JENKINS = {
    "title": "Jenkins", "type": "jenkins",
    "jenkins": {"url": "https://jenkins.example", "job": "perf/load-test", "username": "bot", "api_token": "JT"},
    "loadlens_url": "https://loadlens.example",
}


class FakeApi:
    """Stands in for ApiClient: answers by (method, path prefix), records the calls."""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []

    def _answer(self, method, path):
        for (m, prefix), answer in self.routes.items():
            if m == method and path.startswith(prefix):
                return answer(path) if callable(answer) else answer
        return (404, {"message": "no such route"}, b"", {})

    def request(self, method, path, ok=(200, 201, 202, 204), timeout=None, **kwargs):
        self.calls.append((method, path, kwargs))
        status, data, content, headers = (list(self._answer(method, path)) + [None, None, None])[:4]
        if status not in tuple(ok):
            raise ApiError(f"HTTP {status}", status, json.dumps(data))
        return types.SimpleNamespace(
            status_code=status, content=content or json.dumps(data or {}).encode(), text=(content or b"").decode() or json.dumps(data or {}),
            headers=headers or {}, json=lambda: data,
        )

    def json(self, method, path, **kwargs):
        response = self.request(method, path, **kwargs)
        return response.json() or {}


def _plan(**overrides):
    values = dict(ref="loadlens-run1-01-rest", command=CommandLine(argv=("k6", "run", "/scripts/a.js"), env={"TOKEN": "s3cret-zz"}, cwd="/scripts"),
                  max_seconds=600, run_id="run1", run_name="nightly", run_tag="nightly", instance_index=0, instance_name="rest",
                  instance_tag="rest", tool_id="k6", image="grafana/k6:0.54.0", grace_seconds=20)
    values.update(overrides)
    return LaunchPlan(**values)


# ---- generators and templates -------------------------------------------------------------------


def test_new_generator_types_parse():
    gens = configured_generators({"load_generators": {"docker": DOCKER, "k8s": K8S, "gitlab": GITLAB, "jenkins": JENKINS}})
    assert set(gens) == {"docker", "k8s", "gitlab", "jenkins"}
    docker = gens["docker"]
    assert docker.docker.mounts[0].target == "/scripts" and docker.work_dir == "/tmp" and docker.docker.cpus == 2
    tools = effective_tools({})
    assert docker.offers("k6", tools["k6"]) and not docker.offers("jmeter", tools["jmeter"])  # JMeter has no default image
    assert gens["k8s"].scripts_dir == "/scripts" and gens["k8s"].kubernetes.node_selector == {"role": "loadgen"}
    assert gens["gitlab"].is_ci and gens["gitlab"].work_dir == "." and gens["gitlab"].ci.ref == "main"
    public = docker.to_public(tools)
    assert "KEY" not in json.dumps(public) and public["lists_scripts"] and public["host"] == "https://gen2:2376"


@pytest.mark.parametrize("raw, fragment", [
    ({**DOCKER, "docker": {"url": "gen2:2376"}}, "Docker API"),
    ({**DOCKER, "docker": {"url": "https://h", "client_cert": "C"}}, "вместе"),
    ({**DOCKER, "mounts": [{"source": "x y", "target": "/s"}]}, "монтирования"),
    ({**K8S, "kubernetes": {"api_url": "https://k8s:6443"}}, "токен"),
    ({**K8S, "namespace": "Bad_NS"}, "пространство имён"),
    ({**K8S, "scripts": {"config_map": "a", "pvc": "b"}}, "либо"),
    ({**K8S, "resources": {"cpu": "two"}}, "resources.cpu"),
    ({**GITLAB, "loadlens_url": "ftp://loadlens"}, "LoadLens"),
    ({**GITLAB, "gitlab": {"url": "https://g", "project": "p"}}, "триггера"),
    ({**JENKINS, "jenkins": {"url": "https://j", "job": "x"}}, "API-токен"),
])
def test_broken_new_generators_name_the_problem(raw, fragment):
    with pytest.raises(GeneratorConfigError) as error:
        parse_generator("g", raw)
    assert fragment in str(error.value)


def test_control_port_and_exe_dir_and_aux_commands():
    port = control_port("run1", 0)
    assert 20000 <= port < 40000 and port == control_port("run1", 0) and port != control_port("run1", 1)
    assert exe_dir("/opt/jmeter/bin/jmeter") == "/opt/jmeter/bin/" and exe_dir("jmeter") == ""
    jmeter = effective_tools({})["jmeter"]
    ctx = RenderContext(exe="/opt/jmeter/bin/jmeter", exe_dir="/opt/jmeter/bin/", control_port="23456")
    assert render_aux_command(jmeter.stop_command, ctx) == ("/opt/jmeter/bin/shutdown.sh", "23456")
    k6 = effective_tools({})["k6"]
    assert render_aux_command(k6.status_command, RenderContext(exe="k6", control_port="23456"))[-1] == "127.0.0.1:23456"


# ---- Docker ---------------------------------------------------------------------------------------


def test_docker_executor_creates_starts_and_reads_state():
    gen = parse_generator("docker", DOCKER)
    created = {}

    def create(path):
        return (201, {"Id": "abc123"}, None, None)

    api = FakeApi({
        ("POST", "/containers/create"): create,
        ("POST", "/containers/loadlens-run1-01-rest/start"): (204, {}),
        ("GET", "/containers/loadlens-run1-01-rest/json"): (200, {"State": {"Status": "exited", "ExitCode": 99, "StartedAt": "2026-10-14T10:00:00.123456789Z", "FinishedAt": "2026-10-14T10:05:00Z"}}),
        ("POST", "/containers/loadlens-run1-01-rest/kill"): (204, {}),
        ("DELETE", "/containers/loadlens-run1-01-rest"): (204, {}),
        ("GET", "/containers/loadlens-run1-01-rest/logs"): (200, None, b"\x01\x00\x00\x00\x00\x00\x00\x06hello\n\x02\x00\x00\x00\x00\x00\x00\x04err\n"),
    })
    executor = DockerExecutor(gen, api)
    assert executor.launch(_plan()) == LaunchResult(ref="loadlens-run1-01-rest")
    method, path, kwargs = api.calls[0]
    body = kwargs["json"]
    assert kwargs["params"] == {"name": "loadlens-run1-01-rest"} and body["Entrypoint"] == [""]
    assert body["Cmd"] == ["k6", "run", "/scripts/a.js"] and body["Env"] == ["TOKEN=s3cret-zz"]
    assert body["HostConfig"]["NetworkMode"] == "loadtest" and body["HostConfig"]["NanoCpus"] == 2_000_000_000
    assert body["HostConfig"]["Mounts"] == [{"Type": "volume", "Source": "scripts", "Target": "/scripts", "ReadOnly": True}]
    assert body["Labels"]["loadlens.run_id"] == "run1"
    probe = executor.probe("loadlens-run1-01-rest")
    assert probe.state == "exited" and probe.exit_code == 99 and probe.finished_at - probe.started_at == pytest.approx(299.877, abs=0.01)
    assert executor.logs("loadlens-run1-01-rest") == "hello\nerr\n"
    executor.signal("loadlens-run1-01-rest", "TERM")
    assert api.calls[-1][2]["params"] == {"signal": "SIGTERM"}
    executor.cleanup("loadlens-run1-01-rest")
    assert api.calls[-1][0] == "DELETE"


def test_docker_launch_handles_missing_image_and_existing_container():
    gen = parse_generator("docker", DOCKER)
    api = FakeApi({("POST", "/containers/create"): (404, {"message": "No such image"})})
    with pytest.raises(GeneratorError, match="docker pull"):
        DockerExecutor(gen, api).launch(_plan())
    api = FakeApi({
        ("POST", "/containers/create"): (409, {"message": "Conflict"}),
        ("GET", "/containers/loadlens-run1-01-rest/json"): (200, {"State": {"Status": "running"}}),
    })
    assert DockerExecutor(gen, api).launch(_plan()).already_running
    assert DockerExecutor(gen, FakeApi()).probe("gone").state == "missing"


def test_docker_names_and_demux():
    assert container_name("0123456789abcdef", 1, "rest api") == "loadlens-0123456789ab-02-rest-api"
    assert demux(b"plain tty output") == "plain tty output"


# ---- Kubernetes -----------------------------------------------------------------------------------


def test_kubernetes_job_manifest_and_secret():
    gen = parse_generator("k8s", K8S)
    api = FakeApi({
        ("POST", "/api/v1/namespaces/loadtest/secrets"): (201, {}),
        ("POST", "/apis/batch/v1/namespaces/loadtest/jobs"): (201, {"metadata": {"uid": "u-1"}}),
    })
    executor = KubernetesExecutor(gen, api)
    ref = job_name("run1", 0, "Rest_API")
    assert ref == "ll-run1-01-rest-api"
    executor.launch(_plan(ref=ref))
    # The Job first, then its Secret owned by it in the same request: no Secret without a Job.
    assert [call[:2] for call in api.calls] == [("POST", "/apis/batch/v1/namespaces/loadtest/jobs"), ("POST", "/api/v1/namespaces/loadtest/secrets")]
    secret = api.calls[1][2]["json"]
    assert secret["stringData"] == {"TOKEN": "s3cret-zz"} and secret["metadata"]["name"] == f"{ref}-env"
    owner = secret["metadata"]["ownerReferences"][0]
    assert owner["uid"] == "u-1" and owner["name"] == ref and owner["kind"] == "Job"
    job = api.calls[0][2]["json"]
    spec = job["spec"]
    pod = spec["template"]["spec"]
    container = pod["containers"][0]
    assert spec["backoffLimit"] == 0 and spec["activeDeadlineSeconds"] == 600
    assert container["command"] == ["k6", "run", "/scripts/a.js"] and "env" not in container
    assert container["envFrom"] == [{"secretRef": {"name": f"{ref}-env"}}]
    assert container["resources"]["limits"] == {"cpu": "2", "memory": "2Gi"}
    assert pod["nodeSelector"] == {"role": "loadgen"} and pod["volumes"][0]["configMap"]["name"] == "scripts"
    assert pod["automountServiceAccountToken"] is False and pod["terminationGracePeriodSeconds"] == 20


def test_kubernetes_secret_failure_removes_the_job_and_dollars_stay_literal():
    gen = parse_generator("k8s", K8S)
    api = FakeApi({
        ("POST", "/apis/batch/v1/namespaces/loadtest/jobs"): (201, {"metadata": {"uid": "u-1"}}),
        ("POST", "/api/v1/namespaces/loadtest/secrets"): (403, {"message": "forbidden"}),
        ("DELETE", "/apis/batch/v1/namespaces/loadtest/jobs/"): (200, {}),
    })
    command = CommandLine(argv=("k6", "run", "-e", "X=$(TOKEN)", "/scripts/a.js"), env={"TOKEN": "s3cret-zz"}, cwd="/scripts")
    with pytest.raises(ApiError):
        KubernetesExecutor(gen, api).launch(_plan(ref="ll-run1-01-rest", command=command))
    assert api.calls[-1][:2] == ("DELETE", "/apis/batch/v1/namespaces/loadtest/jobs/ll-run1-01-rest")
    # Kubernetes would expand $(TOKEN) from the environment into the visible spec; $$ keeps it literal.
    assert api.calls[0][2]["json"]["spec"]["template"]["spec"]["containers"][0]["command"][3] == "X=$$(TOKEN)"


def _k8s_probe(job, pods):
    gen = parse_generator("k8s", K8S)
    api = FakeApi({
        ("GET", "/apis/batch/v1/namespaces/loadtest/jobs/j"): (200, job) if job is not None else (404, {}),
        ("GET", "/api/v1/namespaces/loadtest/pods"): (200, {"items": pods}),
    })
    return KubernetesExecutor(gen, api).probe("j")


def test_kubernetes_probe_states():
    assert _k8s_probe(None, []).state == "missing"
    waiting = _k8s_probe({"status": {}}, [{"metadata": {"name": "p"}, "status": {"containerStatuses": [{"state": {"waiting": {"reason": "ImagePullBackOff", "message": "pull failed"}}}]}}])
    assert waiting.state == "starting" and waiting.stuck and "ImagePullBackOff" in waiting.message
    running = _k8s_probe({"status": {}}, [{"metadata": {"name": "p"}, "status": {"containerStatuses": [{"state": {"running": {"startedAt": "2026-10-14T10:00:00Z"}}}]}}])
    assert running.state == "running" and running.started_at
    done = _k8s_probe({"status": {}}, [{"metadata": {"name": "p"}, "status": {"containerStatuses": [{"state": {"terminated": {"exitCode": 2, "startedAt": "2026-10-14T10:00:00Z", "finishedAt": "2026-10-14T10:01:00Z"}}}]}}])
    assert done.state == "exited" and done.exit_code == 2 and done.finished_at - done.started_at == 60
    deadline = _k8s_probe({"status": {"conditions": [{"type": "Failed", "status": "True", "reason": "DeadlineExceeded"}]}}, [])
    assert deadline.exit_code == 124


def test_kubernetes_lists_scripts_from_the_config_map():
    gen = parse_generator("k8s", K8S)
    api = FakeApi({("GET", "/api/v1/namespaces/loadtest/configmaps/scripts"): (200, {"data": {"a.js": "", "b.py": "", "c.ts": ""}})})
    assert KubernetesExecutor(gen, api).list_scripts("/scripts", effective_tools({})["k6"]) == ["a.js", "c.ts"]


# ---- CI ---------------------------------------------------------------------------------------------


class SessionApi(FakeApi):
    """FakeApi with the session attribute CiExecutor writes the GitLab token into."""

    def __init__(self, routes=None):
        super().__init__(routes)
        self.session = types.SimpleNamespace(headers={})


def test_gitlab_check_without_api_token_still_reaches_gitlab(monkeypatch):
    from loadlens_app.load_tests.executors import ci

    api = SessionApi({("GET", "/api/v4/version"): (401, {"message": "401 Unauthorized"})})
    monkeypatch.setattr(ci, "ApiClient", lambda *args, **kwargs: api)
    raw = copy.deepcopy(GITLAB)
    raw["gitlab"]["api_token"] = ""
    raw["loadlens_url"] = "http://127.0.0.1:5000"
    report = ci.check_ci_generator(parse_generator("gitlab", raw), effective_tools({}))
    assert report.ok and report.connected and api.calls[0][:2] == ("GET", "/api/v4/version")
    assert any("API-токен не задан" in problem for problem in report.problems)
    assert any("локальный" in problem for problem in report.problems)
    down = SessionApi()
    down.request = lambda *args, **kwargs: (_ for _ in ()).throw(GeneratorError("GitLab: не удалось подключиться к gitlab.example"))
    monkeypatch.setattr(ci, "ApiClient", lambda *args, **kwargs: down)
    failed = ci.check_ci_generator(parse_generator("gitlab", raw), effective_tools({}))
    assert not failed.ok and not failed.connected


def test_docker_check_names_missing_images_separately(monkeypatch):
    from loadlens_app.load_tests.executors import docker

    api = FakeApi({
        ("GET", "/_ping"): (200, {}),
        ("GET", "/version"): (200, {"Version": "27.1", "Os": "linux", "Arch": "amd64"}),
        ("GET", "/images/grafana/k6:0.54.0/json"): (200, {}),
        ("GET", "/networks/loadtest"): (200, {}),
        ("GET", "/volumes/scripts"): (200, {}),
    })
    monkeypatch.setattr(docker, "docker_api", lambda generator: api)
    report = docker.check_docker_generator(parse_generator("docker", DOCKER), effective_tools({}))
    details = report.to_details()
    assert report.ok and report.connected and details["tools_found"] == ["k6"] and details["images"] == {"k6": "grafana/k6:0.54.0"}
    assert "locust" in details["tools_missing"] and any("docker pull locustio/locust" in problem for problem in report.problems)
    assert details["system"].startswith("Docker 27.1")


def test_jenkins_urls_from_answers_stay_on_the_configured_address():
    from loadlens_app.load_tests.executors.ci import jenkins_path
    from loadlens_app.load_tests.executors.http import ApiClient

    # Jenkins names URLs by its own "Jenkins URL" setting: another host, same paths.
    assert jenkins_path("https://evil.example/queue/item/5/", "https://jenkins.example") == "/queue/item/5"
    assert jenkins_path("http://ci.internal/jenkins/job/a/3/", "https://ci.example/jenkins") == "/job/a/3"
    client = ApiClient("https://jenkins.example", name="Jenkins", auth=("bot", "JT"))
    assert client.url_for("/queue/item/5/api/json") == "https://jenkins.example/queue/item/5/api/json"
    assert client.url_for("https://jenkins.example/job/a/1/api/json") == "https://jenkins.example/job/a/1/api/json"
    for foreign in ("https://evil.example/api/json", "https://jenkins.example.evil/api/json", "http://jenkins.example/api/json"):
        with pytest.raises(GeneratorError, match="вне"):
            client.url_for(foreign)


def test_api_client_wraps_every_requests_error():
    import requests

    from loadlens_app.load_tests.executors.http import ApiClient

    client = ApiClient("https://docker.example:2376", name="Docker API")

    def broken(*args, **kwargs):
        raise requests.exceptions.ChunkedEncodingError("connection broken: IncompleteRead")

    client.session.request = broken
    with pytest.raises(GeneratorError, match="не удался"):
        client.request("GET", "/version")


# ---- watcher ----------------------------------------------------------------------------------------


class FakeExecutor:
    def __init__(self, enforces=True, signals=True, control=True):
        self.enforces_deadline = enforces
        self.supports_signals = signals
        self.supports_control = control
        self.state = {}
        self.calls = []
        self.control_answers = {}
        self.plans = {}

    def launch(self, plan):
        self.plans[plan.ref] = plan
        self.state[plan.ref] = Probe(state="starting")
        return LaunchResult(ref=plan.ref)

    def probe(self, ref):
        return self.state[ref]

    def signal(self, ref, sig, grace_seconds=30):
        self.calls.append(("signal", sig))

    def kill(self, ref):
        self.calls.append(("kill",))

    def logs(self, ref, max_bytes=0):
        return "tail of the log"

    def cleanup(self, ref):
        self.calls.append(("cleanup",))

    def control(self, ref, argv, cwd=""):
        self.calls.append(("control", tuple(argv)))
        return self.control_answers.get(argv[-1], (0, ""))

    def list_scripts(self, scripts_dir, tool):
        return None


class Clock:
    def __init__(self):
        self.now = 1_700_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def world(monkeypatch):
    from loadlens_app.blueprints import dashboard

    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    monkeypatch.setattr(dashboard, "_report_step", lambda area, service_id: "30s")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    monkeypatch.setattr(watcher, "report_filter", lambda run, tool_set, start, end: ([], ""))
    monkeypatch.setattr(watcher, "rps_bounds", lambda run, start, end, packs: None)
    from loadlens_app.load_tests import ci_params

    monkeypatch.setattr(ci_params, "cached_params", lambda generator, ref="": ci_params.CiParams(ref=ref or "main", source="none"))
    gens = {"gen-1": {"title": "Gen", "type": "ssh", "ssh": {"host": "h", "username": "u", "password": "p", "host_key": "ssh-ed25519 AAAA"},
                      "work_dir": "/w", "scripts_dir": "/s"},
            "docker": DOCKER, "k8s": K8S, "gitlab": GITLAB}
    catalog = service.Catalog(tools=effective_tools({}), generators=configured_generators({"load_generators": gens}))
    fakes = {}

    def executor_for(gen):
        return fakes.setdefault(gen.id, FakeExecutor(enforces=gen.type != "docker", signals=gen.type != "kubernetes",
                                                     control=gen.type in ("ssh", "docker")))

    clock = Clock()
    w = watcher.Watcher(store_getter=lambda: memory, catalog_loader=lambda: catalog, executor_getter=executor_for,
                        report_starter=lambda *a, **k: types.SimpleNamespace(job_id="job"), clock=clock)
    return types.SimpleNamespace(store=memory, fakes=fakes, clock=clock, watcher=w)


def _run(world, inst, **options):
    run = RunRecord(run_id="r1", run_name="n", run_tag="n", service="s", project_area="a",
                    options={"auto_report": False, "use_llm": False, "max_seconds": 600, **options},
                    instances=[inst], created_at=datetime.fromtimestamp(world.clock(), timezone.utc))
    world.store.create(run)
    return run


def test_docker_deadline_is_enforced_by_the_watcher(world):
    _run(world, InstanceRecord(index=0, name="i", tag="i", generator="docker", tool="k6", script="a.js"))
    world.watcher.tick()
    fake = world.fakes["docker"]
    ref = next(iter(fake.state))
    assert fake.plans[ref].image == "grafana/k6:0.54.0"
    t0 = world.clock()
    fake.state[ref] = Probe(state="running", started_at=t0)
    world.watcher.tick()
    world.clock.now += 601
    world.watcher.tick()
    assert ("signal", "TERM") in fake.calls
    fake.state[ref] = Probe(state="exited", started_at=t0, finished_at=t0 + 605, exit_code=105)
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert inst.status == store.INST_TIMED_OUT and inst.log_tail == "tail of the log" and ("cleanup",) in fake.calls


def test_soft_stop_uses_the_tool_command_and_status_is_asked(world):
    _run(world, InstanceRecord(index=0, name="i", tag="i", generator="gen-1", tool="jmeter", script="t.jmx"))
    world.watcher.tick()
    fake = world.fakes["gen-1"]
    ref = next(iter(fake.state))
    fake.state[ref] = Probe(state="running", started_at=world.clock())
    world.watcher.tick()
    world.store.request_stop("r1")
    world.watcher.tick()
    controls = [call for call in fake.calls if call[0] == "control"]
    assert controls and controls[-1][1][0].endswith("shutdown.sh") and ("signal", "TERM") not in fake.calls
    assert "команда остановки" in world.store.get("r1").instances[0].message


def test_soft_stop_falls_back_to_the_signal(world):
    _run(world, InstanceRecord(index=0, name="i", tag="i", generator="gen-1", tool="jmeter", script="t.jmx"))
    world.watcher.tick()
    fake = world.fakes["gen-1"]
    ref = next(iter(fake.state))
    port = str(control_port("r1", 0))
    fake.control_answers[port] = (1, "shutdown.sh: not found")
    fake.state[ref] = Probe(state="running", started_at=world.clock())
    world.watcher.tick()
    world.store.request_stop("r1")
    world.watcher.tick()
    assert ("signal", "TERM") in fake.calls


def test_k6_status_is_shown(world):
    _run(world, InstanceRecord(index=0, name="i", tag="i", generator="gen-1", tool="k6", script="a.js"))
    world.watcher.tick()
    fake = world.fakes["gen-1"]
    ref = next(iter(fake.state))
    fake.control_answers[f"127.0.0.1:{control_port('r1', 0)}"] = (0, 'paused: "false"\nvus: "12"\nvus-max: "20"\n')
    fake.state[ref] = Probe(state="running", started_at=world.clock())
    world.watcher.tick()
    assert world.store.get("r1").instances[0].status_text == "VU 12 из 20"


def test_kubernetes_stop_deletes_and_missing_job_counts_as_stopped(world):
    _run(world, InstanceRecord(index=0, name="i", tag="i", generator="k8s", tool="k6", script="a.js"))
    world.watcher.tick()
    fake = world.fakes["k8s"]
    ref = next(iter(fake.state))
    fake.state[ref] = Probe(state="running", started_at=world.clock())
    world.watcher.tick()
    world.store.request_stop("r1")
    world.watcher.tick()
    assert ("signal", "TERM") in fake.calls
    assert "SIGTERM" in world.store.get("r1").instances[0].message
    fake.state[ref] = Probe(state="missing")
    world.watcher.tick()
    assert world.store.get("r1").instances[0].status == store.INST_STOPPED


def test_stuck_start_fails_the_instance(world):
    _run(world, InstanceRecord(index=0, name="i", tag="i", generator="k8s", tool="k6", script="a.js"))
    world.watcher.tick()
    fake = world.fakes["k8s"]
    ref = next(iter(fake.state))
    fake.state[ref] = Probe(state="starting", message="ImagePullBackOff: no such image", stuck=True)
    world.watcher.tick()
    assert world.store.get("r1").instances[0].status == store.INST_STARTING
    world.clock.now += watcher.STUCK_FAIL_SEC + 1
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert inst.status == store.INST_FAILED and "ImagePullBackOff" in inst.message and ("kill",) in fake.calls


def test_ci_launch_stores_only_the_token_hash(world):
    _run(world, InstanceRecord(index=0, name="i", tag="i", generator="gitlab", ci_kind="gitlab", ci_ref="main",
                               ci_params={"VUS": "5"}))
    world.watcher.tick()
    fake = world.fakes["gitlab"]
    plan = next(iter(fake.plans.values()))
    inst = world.store.get("r1").instances[0]
    assert plan.ci_token and inst.ci_token_hash == hashlib.sha256(plan.ci_token.encode()).hexdigest()
    assert plan.ci_ref == "main" and plan.ci_variables == {"VUS": "5"} and plan.ci_auto["LOADLENS_TOKEN"] == plan.ci_token
    assert plan.ci_auto["LOADLENS_EVENTS_URL"] == "https://loadlens.example/load_tests/ci/r1/0/events"
    assert "Пайплайн запущен" in inst.message


# ---- review fixes: errors, deadlines, stops that never end, claims -----------------------------------------


def _run_many(world, instances, **options):
    run = RunRecord(run_id="r1", run_name="n", run_tag="n", service="s", project_area="a",
                    options={"auto_report": False, "use_llm": False, "max_seconds": 600, **options},
                    instances=instances, created_at=datetime.fromtimestamp(world.clock(), timezone.utc))
    world.store.create(run)
    return run


def _running(world, gen_id):
    fake = world.fakes[gen_id]
    ref = next(iter(fake.state))
    fake.state[ref] = Probe(state="running", started_at=world.clock())
    world.watcher.tick()
    return fake, ref


class BrokenLaunch(FakeExecutor):
    def launch(self, plan):
        raise OSError("No such file or directory: /tmp/loadlens-tls-x/ca.pem")


def test_an_error_in_one_instance_keeps_the_progress_of_the_others(world):
    world.fakes["docker"] = BrokenLaunch(enforces=False)
    _run_many(world, [
        InstanceRecord(index=0, name="ci", tag="ci", generator="gitlab", tool="k6", script="load/a.js"),
        InstanceRecord(index=1, name="dk", tag="dk", generator="docker", tool="k6", script="a.js", env={"TOKEN": "s3cret-zz"}),
    ])
    for _ in range(3):
        world.watcher.tick()
    # One pipeline, not one per tick: the CI launch is saved although the Docker instance failed.
    assert len(world.fakes["gitlab"].plans) == 1
    ci, dk = world.store.get("r1").instances
    assert ci.status == store.INST_STARTING
    assert dk.status == store.INST_FAILED and "Не запустился" in dk.message and dk.env == {"TOKEN": "***"}


class BrokenProbe(FakeExecutor):
    broken = False

    def probe(self, ref):
        if self.broken:
            raise RuntimeError("unexpected answer")
        return super().probe(ref)


def test_a_failing_probe_does_not_resend_the_stop(world):
    fake = world.fakes["gen-1"] = BrokenProbe()
    _run_many(world, [InstanceRecord(index=0, name="i", tag="i", generator="gen-1", tool="k6", script="a.js")])
    world.watcher.tick()
    _running(world, "gen-1")
    fake.broken = True
    world.store.request_stop("r1")
    for _ in range(3):
        world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert fake.calls.count(("signal", "TERM")) == 1 and inst.stop_sent_at is not None
    assert inst.message.startswith("Ошибка наблюдателя")
    world.clock.now += 600 + watcher.UNREACHABLE_GRACE_SEC + 1
    world.watcher.tick()
    assert world.store.get("r1").instances[0].status == store.INST_LOST


def test_ci_without_finished_event_is_stopped_after_the_limit_and_closed_if_it_cannot_stop(world):
    fake = world.fakes["gitlab"] = FakeExecutor(control=False)
    fake.supports_stop = False  # GitLab without an API token
    _run_many(world, [InstanceRecord(index=0, name="i", tag="i", generator="gitlab", tool="k6", script="load/a.js")])
    world.watcher.tick()
    _running(world, "gitlab")
    world.clock.now += 600 + watcher.DEADLINE_SLACK_SEC - 5
    world.watcher.tick()
    assert world.store.get("r1").instances[0].stop_sent_at is None  # its own timer still has time
    world.clock.now += 10
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert inst.timed_out and inst.stop_sent_at is not None and "вручную" in inst.message
    assert not [call for call in fake.calls if call[0] in ("signal", "kill")]
    world.clock.now += effective_tools({})["k6"].grace_seconds + 1
    world.watcher.tick()
    run = world.store.get("r1")
    assert run.instances[0].status == store.INST_TIMED_OUT and "закрыл окно" in run.instances[0].message
    assert run.status == store.RUN_DONE


def test_a_stop_the_generator_never_confirms_is_given_up(world):
    _run_many(world, [InstanceRecord(index=0, name="i", tag="i", generator="gen-1", tool="k6", script="a.js")])
    world.watcher.tick()
    fake, ref = _running(world, "gen-1")
    world.store.request_stop("r1")
    world.watcher.tick()
    grace = effective_tools({})["k6"].grace_seconds
    world.clock.now += grace + 1
    world.watcher.tick()
    assert ("kill",) in fake.calls and world.store.get("r1").instances[0].status == store.INST_RUNNING
    world.clock.now += watcher.STOP_CONFIRM_SEC
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert inst.status == store.INST_STOPPED and "не подтвердил" in inst.message and ("cleanup",) in fake.calls


class RefusesStop(FakeExecutor):
    def signal(self, ref, sig, grace_seconds=30):
        raise GeneratorError("Jenkins отказал в доступе (HTTP 403)")


def test_a_stop_the_generator_keeps_refusing_is_given_up(world):
    fake = world.fakes["gen-1"] = RefusesStop(control=False)
    _run_many(world, [InstanceRecord(index=0, name="i", tag="i", generator="gen-1", tool="k6", script="a.js")])
    world.watcher.tick()
    _running(world, "gen-1")
    world.store.request_stop("r1")
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert inst.stop_sent_at is None and inst.stop_failed_at is not None and "не дошла" in inst.message
    world.clock.now += effective_tools({})["k6"].grace_seconds + watcher.STOP_CONFIRM_SEC + 1
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert inst.status == store.INST_STOPPED and "вручную" in inst.message


def test_ci_token_hash_is_stored_before_the_pipeline_starts(world):
    seen = {}

    class Checks(FakeExecutor):
        def launch(self, plan):
            seen["stored"] = world.store.get("r1").instances[0].ci_token_hash
            seen["expected"] = hashlib.sha256(plan.ci_token.encode()).hexdigest()
            return super().launch(plan)

    world.fakes["gitlab"] = Checks(control=False)
    _run_many(world, [InstanceRecord(index=0, name="i", tag="i", generator="gitlab", tool="k6", script="load/a.js")])
    world.watcher.tick()
    assert seen["stored"] == seen["expected"] != ""


def test_a_run_taken_over_mid_pass_is_left_alone(world):
    _run_many(world, [InstanceRecord(index=0, name="i", tag="i", generator="gen-1", tool="k6", script="a.js")])
    run = world.store.get("r1")
    catalog = world.watcher._catalog_loader()
    # The claim is old enough to be renewed, and the store says another watcher owns the run now.
    claim = watcher._Claim(world.store, "r1", "watcher-b", renewed_at=-1e9)
    with pytest.raises(watcher.RunTakenOver):
        world.watcher.advance(run, catalog, claim)
    assert "gen-1" not in world.fakes or not world.fakes["gen-1"].plans
    world.store.claim("watcher-b", watcher.STALE_AFTER_SEC)
    assert world.store.heartbeat("r1", "watcher-b") and not world.store.heartbeat("r1", world.watcher.id)
    world.watcher.tick()  # the run is fresh with watcher-b: this watcher does not touch it
    assert "gen-1" not in world.fakes or not world.fakes["gen-1"].plans


def test_stale_window_is_well_above_a_slow_pass():
    assert watcher.STALE_AFTER_SEC >= 4 * 30 and watcher.HEARTBEAT_EVERY_SEC < watcher.STALE_AFTER_SEC / 4


# ---- CI endpoints, presets, hints ------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    from loadlens_app.blueprints import dashboard
    from loadlens_app.blueprints import load_tests as blueprint

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setitem(core.CONFIG, "load_generators", {"gitlab": GITLAB})
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    monkeypatch.setattr(presets, "_store", presets.MemoryPresetStore())
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: [])
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(http=app.test_client(), store=memory)


def _ci_run(store_, token, status=store.INST_STARTING):
    inst = InstanceRecord(index=0, name="rest", tag="rest", generator="gitlab", ci_kind="gitlab", ci_ref="main",
                          ci_params={"VUS": "3"}, status=status, ci_token_hash=hashlib.sha256(token.encode()).hexdigest())
    run = RunRecord(run_id="run1", run_name="nightly", run_tag="nightly", service="checkout", status=store.RUN_RUNNING,
                    options={"max_seconds": 900}, instances=[inst])
    store_.create(run)


def test_ci_events_need_the_instance_token(client):
    _ci_run(client.store, "good-token")
    events = "/load_tests/ci/run1/0/events"
    assert client.http.post(events, json={"event": "started"}).status_code == 404
    assert client.http.post(events, json={"event": "started"}, headers={"X-LoadLens-Token": "wrong"}).status_code == 404
    assert client.http.post(events, json={"event": "started"}, headers={"X-LoadLens-Token": "good-token"}).status_code == 200
    assert client.http.post(events, json={"event": "finished", "exit_code": "1"}, headers={"X-LoadLens-Token": "good-token"}).status_code == 400
    assert client.http.post(events, json={"event": "finished", "exit_code": 0}, headers={"X-LoadLens-Token": "good-token"}).status_code == 200
    assert [e["event"] for e in client.store.get("run1").events] == ["started", "finished"]
    # The plan endpoint of the earlier design is gone: the pipeline holds its own command.
    assert client.http.get("/load_tests/ci/run1/0/plan", headers={"X-LoadLens-Token": "good-token"}).status_code == 404


def test_ci_events_refuse_finished_instances(client):
    _ci_run(client.store, "tok", status=store.INST_FINISHED)
    assert client.http.post("/load_tests/ci/run1/0/events", json={"event": "started"}, headers={"X-LoadLens-Token": "tok"}).status_code == 410


def test_a_preset_is_replaced_only_by_its_author_or_an_admin(monkeypatch):
    monkeypatch.setattr(presets, "_store", presets.MemoryPresetStore())
    form = {"instances": [{"generator": "gitlab", "tool": "k6", "script": "a.js"}]}
    mine = presets.save_preset("Smoke", "checkout", "shop", form, "alice")
    with pytest.raises(presets.PresetError) as error:
        presets.save_preset("Smoke", "checkout", "shop", {**form, "test_type": "soak"}, "mallory")
    assert error.value.status == 403
    assert presets.get_presets().get(mine.preset_id).owner == "alice"
    assert presets.save_preset("Smoke", "checkout", "shop", form, "alice").preset_id == mine.preset_id
    assert presets.save_preset("Smoke", "checkout", "shop", form, "root", is_admin=True).owner == "root"


def test_preset_table_error_is_not_hidden_by_the_closed_connection(monkeypatch):
    class Closed(Exception):
        pass

    class Conn:
        closed = False
        _autocommit = False

        @property
        def autocommit(self):
            return self._autocommit

        @autocommit.setter
        def autocommit(self, value):
            if self.closed:
                raise Closed("connection already closed")
            self._autocommit = value

        def cursor(self):
            conn = self

            class Cursor:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, *args):
                    raise PermissionError("permission denied for schema public")

            return Cursor()

        def close(self):
            self.closed = True

    conn = Conn()
    monkeypatch.setattr(core, "_ts_conn", lambda: conn)
    monkeypatch.setattr(presets, "_table_ref", lambda: presets.sql.Identifier("load_test_presets"))
    with pytest.raises(PermissionError):
        presets.PgPresetStore()._conn()
    assert conn.closed


def test_presets_keep_no_environment_values(client):
    form = {"test_type": "step", "instances": [{"name": "rest", "generator": "gitlab", "tool": "k6", "script": "a.js", "env": "TOKEN=s3cret-zz\nB=2"}]}
    saved = client.http.post("/load_tests/presets", json={"name": "Smoke", "service": "checkout", "spec": form}).get_json()
    assert saved["spec"]["instances"][0]["env"] == "TOKEN=\nB=" and "s3cret-zz" not in json.dumps(saved)
    again = client.http.post("/load_tests/presets", json={"name": "Smoke", "service": "checkout", "spec": {**form, "test_type": "soak"}}).get_json()
    assert again["preset_id"] == saved["preset_id"] and again["spec"]["test_type"] == "soak"
    listed = client.http.get("/load_tests/presets", query_string={"service": "checkout"}).get_json()["presets"]
    assert [p["name"] for p in listed] == ["Smoke"]
    assert client.http.post("/load_tests/presets", json={"name": "", "service": "checkout", "spec": form}).status_code == 400
    assert client.http.delete(f"/load_tests/presets/{saved['preset_id']}").status_code == 200
    assert client.http.get("/load_tests/presets").get_json()["presets"] == []


def test_credentials_in_urls_are_not_shown():
    from loadlens_app.load_tests.tools import public_url

    assert public_url("https://bot:pa55@vm.example:8428/") == "https://vm.example:8428/"
    assert public_url("http://vm:8428") == "http://vm:8428" and public_url("") == ""
    gen = parse_generator("docker", {**DOCKER, "docker": {"url": "https://user:pw@gen2:2376"}})
    assert gen.address() == "https://gen2:2376"
    raw = copy.deepcopy(JENKINS)
    raw["jenkins"]["url"] = "https://bot:JT@jenkins.example"
    assert "JT" not in parse_generator("jenkins", raw).address()


def test_output_hints_follow_the_domain_source():
    from loadlens_app.load_tests.hints import output_hint

    prom = {"data_sources": {"vm": {"title": "VM", "type": "prometheus", "prometheus": {"url": "http://vm:8428"}}},
            "domain_sources": {"default": {"source": "vm"}}}
    hint = output_hint(prom, "k6", ["lt-k6-prometheus-rw"])
    assert hint.params == "--out experimental-prometheus-rw" and hint.env["K6_PROMETHEUS_RW_SERVER_URL"] == "http://vm:8428/api/v1/write"
    gatling = output_hint(prom, "gatling", [])
    assert "graphite.host=vm" in gatling.env["JAVA_OPTS"]
    assert not any("relabel" in note for note in gatling.notes)
    labels = output_hint(prom, "gatling", ["lt-gatling-graphite-victoriametrics-labels"])
    assert any("relabelConfig" in note and "outputDirectoryBaseName" in note for note in labels.notes)
    native = output_hint(prom, "k6", ["lt-k6-prometheus-native"])
    assert native.env.get("K6_PROMETHEUS_RW_TREND_AS_NATIVE_HISTOGRAM") == "true"
    influx = {"data_sources": {"i2": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "perf", "token": "SECRET"}}},
              "domain_sources": {"default": {"source": "i2"}, "lt_framework": {"source": "i2", "bucket": "k6"}}}
    flux = output_hint(influx, "k6", [])
    assert flux.env["K6_INFLUXDB_ORGANIZATION"] == "perf" and "SECRET" not in json.dumps(flux.to_dict())


# ---- summed RPS and the generators domain --------------------------------------------------------------


def test_summed_rps_row_follows_its_parts():
    from AI.query_packs import add_summed_rps, apply_run_filter, load_catalog, merge_pack, prune_block, render_pack
    from tests.promql_lint import check_promql

    catalog = load_catalog()
    block, _ = merge_pack(None, render_pack(catalog.pack("lt-k6-prometheus-rw")))
    with pytest.raises(ValueError):
        add_summed_rps(block)
    block, _ = merge_pack(block, render_pack(catalog.pack("lt-gatling-graphite-victoriametrics")))
    summed, label, parts = add_summed_rps(block)
    index = summed["labels"].index(label)
    check_promql(summed["promql_queries"][index])
    assert len(parts) == 2 and summed["label_keys_list"][index] == ["series"]
    filtered, _ = apply_run_filter({"lt_framework": summed}, "t1", ["lt-k6-prometheus-rw"])
    assert 'testid="t1"' in filtered["lt_framework"]["promql_queries"][index]
    again, _, _ = add_summed_rps(summed)
    assert again["labels"].count(label) == 1
    gone = copy.deepcopy(summed)
    keep = [i for i, name in enumerate(gone["labels"]) if name != parts[1]]
    for key in ("labels", "promql_queries", "label_keys_list"):
        gone[key] = [gone[key][i] for i in keep]
    assert "derived" not in prune_block(gone)


def test_generators_domain_is_a_known_binding_and_pack_domain():
    from AI.data_sources import BINDING_DOMAINS
    from AI.query_packs import load_catalog

    assert "lt_generators" in BINDING_DOMAINS
    pack = load_catalog().pack("lt-generators-node-exporter")
    assert pack.domain == "lt_generators" and {q.role for q in pack.queries} >= {"generator.cpu_pct", "generator.mem_pct"}
    prompt = (ROOT_DIR / "AI" / "prompts" / "lt_generators_prompt.txt").read_text(encoding="utf-8")
    assert "ступен" in prompt and "генератор" in prompt
    overall = (ROOT_DIR / "AI" / "prompts" / "overall_prompt.txt").read_text(encoding="utf-8")
    assert "{answer_lt_generators}" in overall
