# SPDX-License-Identifier: Apache-2.0
"""Load tests started from LoadLens: tool templates, generators, remote scripts, store and watcher."""

import copy
import shlex
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from loadlens_app.limits import LimitError
from loadlens_app.load_tests import executors, live, service, store, tools, watcher
from loadlens_app.load_tests.executors import GeneratorError, LaunchPlan, LaunchResult, Probe
from loadlens_app.load_tests.generators import GeneratorConfigError, configured_generators, parse_generator
from loadlens_app.load_tests.store import InstanceRecord, MemoryLoadTestStore, RunRecord
from loadlens_app.load_tests.tools import (
    DEFAULT_TOOLS,
    CommandLine,
    RenderContext,
    ToolConfigError,
    ToolInputError,
    effective_tools,
    make_tag,
    parse_env,
    parse_params,
    parse_tool,
    render_command,
    resolve_script,
    tools_overrides,
    validate_tools_section,
)

SSH_GENERATOR = {
    "title": "Gen 1",
    "type": "ssh",
    "ssh": {"host": "10.0.0.5", "port": 22, "username": "loadlens", "password": "pw", "host_key": "ssh-ed25519 AAAA"},
    "work_dir": "/opt/loadlens/runs",
    "scripts_dir": "/opt/tests",
    "tools": {},
    "max_parallel": 0,
    "env": {"JAVA_OPTS": "-Xmx2g"},
}
GENERATORS = {"gen-1": SSH_GENERATOR, "manual": {"title": "Вручную", "type": "manual"}}


def _ctx(**overrides):
    values = dict(
        exe="k6", script="/opt/tests/a b.js", script_dir="/opt/tests", scripts_dir="/opt/tests",
        run_name="release 2.5", run_tag="release-2-5", instance="rest", instance_tag="rest",
        instance_dir="/opt/loadlens/runs/r1/01-rest",
    )
    values.update(overrides)
    return RenderContext(**values)


# ---- tool templates -----------------------------------------------------------------------------


def test_default_tools_parse_and_k6_tags_the_run():
    parsed = effective_tools({})
    assert set(parsed) == set(DEFAULT_TOOLS)
    k6 = parsed["k6"]
    command = render_command(k6, _ctx(control_port="23456"), parse_params("--vus 10 --duration 5m"))
    assert command.argv == (
        "k6", "run", "--address", "127.0.0.1:23456", "--tag", "testid=release-2-5", "--tag", "lt_instance=rest",
        "--vus", "10", "--duration", "5m", "/opt/tests/a b.js",
    )
    assert command.cwd == "/opt/tests"
    assert "lt-k6-prometheus-rw" in k6.run_filter_packs and k6.failed_checks_exit_codes == (99,)


def test_user_input_never_reaches_the_shell_unquoted():
    k6 = effective_tools({})["k6"]
    params = parse_params("-e 'A=b c;rm -rf /' $(id) `uname` \"x y\"")
    assert params == ["-e", "A=b c;rm -rf /", "$(id)", "`uname`", "x y"]
    command = render_command(k6, _ctx(), params, {"TOKEN": "s3cr$t'q"})
    line = command.shell()
    assert shlex.split(line) == ["env", "TOKEN=s3cr$t'q", *command.argv]
    assert "'$(id)'" in line and "'A=b c;rm -rf /'" in line
    assert "s3cr" not in command.shell(mask_env=True) and "TOKEN=***" in command.shell(mask_env=True)


@pytest.mark.parametrize("text", ["'unclosed", "a\x00b"])
def test_bad_params_are_rejected(text):
    with pytest.raises(ToolInputError):
        parse_params(text)


def test_template_env_gets_the_user_value_appended():
    tool = parse_tool("g", {"command": "{exe} {params}", "executable": "gatling.sh", "env": {"JAVA_OPTS": "-Dprefix={run_tag}"}})
    command = render_command(tool, _ctx(exe="gatling.sh"), [], {"JAVA_OPTS": "-Xmx4g", "OTHER": "1"})
    assert command.env == {"JAVA_OPTS": "-Dprefix=release-2-5 -Xmx4g", "OTHER": "1"}


@pytest.mark.parametrize("raw, fragment", [
    ({"command": ""}, "команда не задана"),
    ({"command": "{exe} {nope}", "executable": "x"}, "{nope}"),
    ({"command": "{exe} --opts={params}", "executable": "x"}, "{params}"),
    ({"command": "{exe} {params} {params}", "executable": "x"}, "{params}"),
    ({"command": "{exe} run"}, "executable"),
    ({"command": "x", "script_kind": "dir"}, "script_kind"),
    ({"command": "x", "stop_signal": "KILL"}, "stop_signal"),
    ({"command": "x", "success_exit_codes": [300]}, "0–255"),
    ({"command": "x", "scripts_glob": ["../*.js"]}, "маска"),
    ({"command": "x", "env": {"BAD-NAME": "1"}}, "BAD-NAME"),
    ({"command": "x", "cwd": "{params}"}, "{params}"),
])
def test_broken_templates_name_the_problem(raw, fragment):
    with pytest.raises(ToolConfigError) as error:
        parse_tool("t", raw)
    assert fragment in str(error.value)


def test_env_text_and_tags():
    assert parse_env("A=1\n# comment\n\nB = two=2\n") == {"A": "1", "B": " two=2"}
    assert parse_env({"X": None}) == {"X": ""}
    with pytest.raises(ToolInputError):
        parse_env("1BAD=x")
    with pytest.raises(ToolInputError):
        parse_env("NOEQUALS")
    assert make_tag("release 2.5 / step", "x") == "release-2-5-step"
    assert make_tag("прогон", "run-1234") == "run-1234"
    assert len(make_tag("a" * 200, "x")) <= tools.MAX_TAG_CHARS


def test_scripts_resolve_against_the_generator_directory():
    k6 = effective_tools({})["k6"]
    assert resolve_script(k6, "sub/../checkout.js", "/opt/tests") == ("/opt/tests/checkout.js", "/opt/tests")
    assert resolve_script(k6, "/opt/tests/sub/x.js", "/opt/tests") == ("/opt/tests/sub/x.js", "/opt/tests/sub")
    assert resolve_script(k6, "/srv/x.js", "") == ("/srv/x.js", "/srv")
    # With a scripts directory the script stays inside it.
    for outside in ("/srv/x.js", "../x.js", "sub/../../etc/passwd", "/opt/tests-other/x.js"):
        with pytest.raises(ToolInputError, match="вне каталога"):
            resolve_script(k6, outside, "/opt/tests")
    # CI checkouts: relative paths, but not above the checkout.
    assert resolve_script(k6, "k6/load.js", "", allow_relative=True) == ("k6/load.js", "k6")
    with pytest.raises(ToolInputError, match="вне каталога"):
        resolve_script(k6, "../secrets/x.js", "", allow_relative=True)
    with pytest.raises(ToolInputError):
        resolve_script(k6, "x.js", "")
    with pytest.raises(ToolInputError):
        resolve_script(k6, "", "/opt/tests")
    gatling = effective_tools({})["gatling"]
    assert resolve_script(gatling, "acme.CheckoutSimulation", "/opt/g") == ("acme.CheckoutSimulation", "/opt/g")
    with pytest.raises(ToolInputError):
        resolve_script(gatling, "acme.Checkout; rm", "/opt/g")
    assert tools.class_name_from_path("./src/test/java/acme/CheckoutSimulation.java") == "acme.CheckoutSimulation"
    assert tools.class_name_from_path("user-files/simulations/devstack/CheckoutSimulation.scala") == "devstack.CheckoutSimulation"


def test_saved_tools_keep_only_differences_and_disable_removed_defaults():
    edited = copy.deepcopy({key: value for key, value in DEFAULT_TOOLS.items() if key != "locust"})
    edited["k6"]["executable"] = "/usr/local/bin/k6"
    edited["custom"] = {"title": "Tank", "command": "yandex-tank -c {script}", "script_kind": "file"}
    overrides = tools_overrides(edited)
    assert overrides == {
        "k6": {"executable": "/usr/local/bin/k6"},
        "custom": edited["custom"],
        "locust": {"enabled": False},
    }
    effective = effective_tools({"load_tools": overrides})
    assert "locust" not in effective and effective["k6"].executable == "/usr/local/bin/k6" and "custom" in effective
    assert effective["k6"].command == DEFAULT_TOOLS["k6"]["command"]
    with pytest.raises(ToolConfigError):
        validate_tools_section({"k6": {"command": "{exe} {params} {params}"}})
    with pytest.raises(ToolConfigError):
        validate_tools_section({"k6": "k6 run"})


# ---- generators -------------------------------------------------------------------------------


def test_generators_parse_and_hide_credentials():
    generators = configured_generators({"load_generators": {**GENERATORS, "broken": {"type": "ssh"}}})
    assert set(generators) == {"gen-1", "manual"}
    gen = generators["gen-1"]
    public = gen.to_public(effective_tools({}))
    assert public["host"] == "10.0.0.5" and set(public["tools"]) == set(DEFAULT_TOOLS)
    assert "pw" not in str(public) and "password" not in str(public)
    assert generators["manual"].is_manual and generators["manual"].offers("anything")


@pytest.mark.parametrize("patch, fragment", [
    ({"ssh": {"host": "", "username": "u", "password": "p"}}, "адрес"),
    ({"ssh": {"host": "h", "username": "", "password": "p"}}, "пользовател"),
    ({"ssh": {"host": "h", "username": "u"}}, "ключ или пароль"),
    ({"work_dir": "runs"}, "абсолютный путь"),
    ({"ssh": {"host": "h", "username": "u", "password": "p", "port": 70000}}, "порт"),
    ({"type": "podman"}, "тип"),
    ({"max_parallel": -1}, "max_parallel"),
])
def test_broken_generators_name_the_problem(patch, fragment):
    raw = {**copy.deepcopy(SSH_GENERATOR), **patch}
    with pytest.raises(GeneratorConfigError) as error:
        parse_generator("gen-1", raw)
    assert fragment in str(error.value)


def test_generator_tool_list_limits_the_offer():
    gen = parse_generator("g", {**SSH_GENERATOR, "tools": {"k6": {"executable": "/usr/local/bin/k6"}, "gatling": {"scripts_dir": "/opt/gatling/user-files/simulations"}}})
    k6 = effective_tools({})["k6"]
    assert gen.offers("k6") and not gen.offers("jmeter")
    assert gen.executable(k6) == "/usr/local/bin/k6"
    assert gen.scripts_dir_for("gatling") == "/opt/gatling/user-files/simulations" and gen.scripts_dir_for("k6") == "/opt/tests"


# ---- remote scripts -------------------------------------------------------------------------------


def test_probe_output_maps_to_states():
    assert executors.parse_probe("missing=1\nnow=10").state == "missing"
    assert executors.parse_probe("alive=1\nnow=10").state == "starting"
    running = executors.parse_probe("started_at=100\nalive=1\nnow=110")
    assert running.state == "running" and running.started_at == 100 and running.remote_now == 110
    exited = executors.parse_probe("started_at=100\nfinished_at=160\nexit_code=99\nalive=0\nnow=161")
    assert (exited.state, exited.exit_code, exited.finished_at) == ("exited", 99, 160)
    assert executors.parse_probe("started_at=100\nalive=0").state == "dead"


def test_script_lists_turn_into_paths_or_classes():
    assert executors.parse_script_list("__missing__\n", "file") is None
    assert executors.parse_script_list("./b.js\n./a/c.js\n", "file") == ["a/c.js", "b.js"]
    assert executors.parse_script_list("./src/test/java/acme/ASimulation.java\n", "class") == ["acme.ASimulation"]


class FakeShell:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def run(self, script, stdin=None, timeout=30):
        self.calls.append((script, stdin))
        return self.replies.pop(0) if self.replies else (0, "", "")


def test_ssh_executor_uploads_the_runner_and_reads_the_answer():
    gen = parse_generator("gen-1", SSH_GENERATOR)
    k6 = effective_tools({})["k6"]
    command = render_command(k6, _ctx(), ["--vus", "5"])
    shell = FakeShell([(0, "launched\n", ""), (0, "already\n", ""), (0, "oops", ""), (3, "", "mkdir: denied")])
    executor = executors.SshExecutor(gen, shell)
    ref = "/opt/loadlens/runs/r1/01-rest"
    plan = LaunchPlan(ref=ref, command=command, max_seconds=600)
    assert executor.launch(plan) == LaunchResult(ref=ref, already_running=False)
    script, stdin = shell.calls[0]
    assert "nohup sh ./runner.sh" in script and "runner.pid" in script
    runner = stdin.decode()
    assert "timeout -s TERM -k 60 600 k6 run" in runner and "'--vus' 5".replace("'--vus'", "--vus") in runner
    assert executor.launch(plan).already_running
    with pytest.raises(GeneratorError):
        executor.launch(plan)
    with pytest.raises(GeneratorError) as error:
        executor.launch(plan)
    assert "mkdir: denied" in str(error.value)


def test_runner_quotes_paths_and_reports_a_missing_working_directory():
    command = render_command(effective_tools({})["k6"], _ctx(script_dir="/opt/my tests"), [])
    runner = executors.runner_script("/opt/runs/r1/01-a", command, 60)
    assert "cd '/opt/my tests'" in runner and "finish 126" in runner and "D=/opt/runs/r1/01-a" in runner
    assert executors.signal_script("/opt/runs/r 1", "TERM").count("'/opt/runs/r 1'") == 1
    assert 'kill -s KILL -- "-$p"' in executors.kill_script("/x")


def test_runner_keeps_environment_values_private():
    command = CommandLine(argv=("k6", "run", "/opt/tests/a.js"), env={"TOKEN": "s3cret zz"}, cwd="/opt/tests")
    runner = executors.runner_script("/opt/runs/r1/01-a", command, 60)
    # Exported inside the runner (not `env NAME=value` arguments visible in ps), the file private and removed.
    assert "export TOKEN='s3cret zz'" in runner and "env 'TOKEN" not in runner
    assert runner.splitlines()[2] == "umask 077" and runner.count('rm -f "$D/runner.sh"') == 2
    assert runner.index("export TOKEN") < runner.index("timeout -s TERM") < runner.rindex('rm -f "$D/runner.sh"')
    assert "timeout -s TERM -k 60 60 k6 run /opt/tests/a.js" in runner
    assert executors.launch_script("/opt/runs/r1/01-a").splitlines()[0] == "umask 077"


def test_instance_env_cannot_replace_generator_or_runtime_variables():
    from loadlens_app.load_tests.tools import check_instance_env

    check_instance_env({"K6_VUS": "5", "JAVA_OPTS": "-Xmx1g", "HEAP": "-Xmx2g"}, {})
    for name in ("PATH", "LD_PRELOAD", "LD_ANYTHING", "BASH_ENV", "JAVA_TOOL_OPTIONS", "NODE_OPTIONS", "PYTHONPATH", "LOADLENS_TOKEN"):
        with pytest.raises(ToolInputError, match=name):
            check_instance_env({name: "x"}, {})
    with pytest.raises(ToolInputError, match="настройках генератора"):
        check_instance_env({"K6_PROMETHEUS_RW_SERVER_URL": "http://elsewhere"}, {"K6_PROMETHEUS_RW_SERVER_URL": "http://vm", "K6_PROMETHEUS_RW_PASSWORD": "pw"})


def test_plan_rejects_overrides_of_generator_env_and_keeps_list_params_whole(planner):
    with pytest.raises(service.LaunchError, match="JAVA_OPTS"):
        service.plan_run(_form(instances=[{"name": "g", "generator": "gen-1", "tool": "k6", "script": "a.js", "env": "JAVA_OPTS=-Xmx9g"}]), "eng")
    planned = service.plan_run(_form(instances=[{"name": "r", "generator": "gen-1", "tool": "k6", "script": "a.js",
                                                 "params": ["--tag", "team=perf lab", "--vus", "5"]}]), "eng")
    inst = planned.run.instances[0]
    assert inst.params == "--tag 'team=perf lab' --vus 5" and "'team=perf lab'" in inst.command


def test_manual_instances_keep_no_environment(planner):
    planned = service.plan_run(_form(instances=[{"name": "by hand", "generator": "manual", "env": "TOKEN=s3cret-zz"}]), "eng")
    assert planned.run.instances[0].env == {}


def test_manual_window_closes_at_the_run_limit(world):
    run_id = _run(world, [_inst(0, generator="manual", tool="", env={"TOKEN": "s3cret-zz"})], auto_report=False)
    world.watcher.tick()
    inst = world.store.get(run_id).instances[0]
    assert inst.status == store.INST_RUNNING and inst.env == {"TOKEN": "***"}
    world.clock.advance(3601)
    world.watcher.tick()
    inst = world.store.get(run_id).instances[0]
    assert inst.status == store.INST_TIMED_OUT and inst.finished_at - inst.started_at == timedelta(seconds=3600)


def test_changed_or_cleared_host_key_is_a_new_destination_for_ssh_secrets():
    from loadlens_app.config_secrets import MASK, SecretDestinationChangedError, restore_secrets

    stored = {"gen-1": {"type": "ssh", "ssh": {"host": "10.0.0.5", "password": "pw", "host_key": "ssh-ed25519 AAAA"}}}
    for key in ("ssh-ed25519 BBBB", ""):
        incoming = {"gen-1": {"type": "ssh", "ssh": {"host": "10.0.0.5", "password": MASK, "host_key": key}}}
        with pytest.raises(SecretDestinationChangedError):
            restore_secrets(incoming, stored)
    # Pinning the key the first check found keeps the stored password.
    unpinned = {"gen-1": {"type": "ssh", "ssh": {"host": "10.0.0.5", "password": "pw", "host_key": ""}}}
    incoming = {"gen-1": {"type": "ssh", "ssh": {"host": "10.0.0.5", "password": MASK, "host_key": "ssh-ed25519 AAAA"}}}
    assert restore_secrets(incoming, unpinned)["gen-1"]["ssh"]["password"] == "pw"


def test_check_details_have_the_same_keys_for_every_type():
    details = executors.CheckReport(ok=False, message="m", details={"system": "Linux", "host_key": "k"},
                                    connected=True, problems=["нет nohup"]).to_details()
    assert details["system"] == "Linux" and details["host_key"] == "k" and details["connected"] is True
    assert details["problems"] == ["нет nohup"]
    assert {key: details[key] for key in ("tools_found", "tools_missing", "versions", "images")} == {
        "tools_found": [], "tools_missing": [], "versions": {}, "images": {}}
    assert executors.CheckReport(ok=True, message="").to_details()["connected"] is False


def test_verification_mark_follows_the_generator_endpoint():
    gen = parse_generator("gen-1", SSH_GENERATOR)
    target = gen.check_target()
    assert target == "ssh://loadlens@10.0.0.5:22" and not gen.is_verified
    marked = parse_generator("gen-1", {**SSH_GENERATOR, "verified": {"at": "2026-10-15T10:00:00+00:00", "target": target}})
    assert marked.is_verified and marked.to_public({})["verified"] and marked.to_public({})["verified_at"].startswith("2026-10-15")
    moved = copy.deepcopy(SSH_GENERATOR)
    moved["ssh"]["host"] = "10.0.0.6"
    stale = parse_generator("gen-1", {**moved, "verified": {"at": "2026-10-15T10:00:00+00:00", "target": target}})
    assert not stale.is_verified and stale.to_public({})["verified_at"] == ""
    assert parse_generator("manual", {"title": "M", "type": "manual"}).is_verified
    # Directories, tools and limits are not the endpoint: editing them keeps the mark.
    tuned = parse_generator("gen-1", {**SSH_GENERATOR, "scripts_dir": "/srv/t", "max_parallel": 3,
                                      "verified": {"at": "2026-10-15T10:00:00+00:00", "target": target}})
    assert tuned.is_verified


def test_check_report_lists_problems():
    gen = parse_generator("gen-1", {**SSH_GENERATOR, "tools": {"k6": {}, "jmeter": {}}})
    tool_set = effective_tools({})
    values = executors.parse_key_values("uname=Linux x86_64\ntimeout=gnu\nnohup=ok\nwork_dir=ok\nscripts_dir=ok\ntool:k6=found\ntool:jmeter=missing\n")
    ok, problems, found = executors._check_message(values, gen, tool_set)
    assert ok and found == ["k6"] and len(problems) == 1 and "JMeter" in problems[0]
    values["timeout"] = "missing"
    assert executors._check_message(values, gen, tool_set)[0] is False
    script = executors.check_script(gen, tool_set)
    assert "tool:k6=found" in script and "tool:locust" not in script


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("timeout") or not shutil.which("sh"), reason="needs a POSIX shell with GNU timeout")
def test_runner_protocol_on_a_local_shell(tmp_path):
    """The real scripts against a local sh: start, exit code, stop by signal."""

    class LocalShell:
        def run(self, script, stdin=None, timeout=30):
            done = subprocess.run(["sh", "-c", script], input=stdin, capture_output=True, timeout=timeout)
            return done.returncode, done.stdout.decode(), done.stderr.decode()

    gen = parse_generator("local", {**SSH_GENERATOR, "work_dir": str(tmp_path)})
    tool = parse_tool("sh", {"command": "sh -c {params}", "cwd": "/"})
    executor = executors.SshExecutor(gen, LocalShell())

    def wait_for(ref, state, limit=10.0):
        deadline = time.time() + limit
        while time.time() < deadline:
            probe = executor.probe(ref)
            if probe.state == state:
                return probe
            time.sleep(0.1)
        raise AssertionError(f"{ref} did not reach {state}: {executor.probe(ref)}")

    def launch(ref, argv, max_seconds):
        executor.launch(LaunchPlan(ref=ref, command=render_command(tool, _ctx(exe="sh"), argv), max_seconds=max_seconds))

    quick = str(tmp_path / "runs" / "r" / "01-quick")
    launch(quick, ["echo hello; exit 3"], 30)
    probe = wait_for(quick, "exited")
    assert probe.exit_code == 3 and probe.started_at and probe.finished_at
    assert "hello" in executor.logs(quick)

    slow = str(tmp_path / "runs" / "r" / "02-slow")
    launch(slow, ["sleep 30"], 60)
    wait_for(slow, "running")
    executor.signal(slow, "TERM")
    assert wait_for(slow, "exited").exit_code != 0


# ---- watcher --------------------------------------------------------------------------------------


class FakeExecutor:
    """Remote state per ref, driven by the test."""

    enforces_deadline = True
    supports_signals = True
    supports_control = True

    def __init__(self):
        self.launched = {}
        self.plans = {}
        self.state = {}
        self.signals = []
        self.killed = []
        self.cleaned = []
        self.controls = []
        self.control_answer = (0, "")
        self.fail_launch = set()
        self.unreachable = False

    def launch(self, plan):
        if plan.ref in self.fail_launch:
            raise GeneratorError("ssh refused")
        self.launched[plan.ref] = (plan.command, plan.max_seconds)
        self.plans[plan.ref] = plan
        self.state[plan.ref] = Probe(state="starting")
        return LaunchResult(ref=plan.ref)

    def probe(self, ref):
        if self.unreachable:
            raise GeneratorError("timeout")
        return self.state[ref]

    def signal(self, ref, sig, grace_seconds=30):
        self.signals.append((ref, sig))

    def kill(self, ref):
        self.killed.append(ref)

    def logs(self, ref, max_bytes=0):
        return f"log of {ref}"

    def cleanup(self, ref):
        self.cleaned.append(ref)

    def control(self, ref, argv, cwd=""):
        self.controls.append((ref, tuple(argv)))
        return self.control_answer

    def list_scripts(self, scripts_dir, tool):
        return None


class Clock:
    def __init__(self, start=1_700_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def world(monkeypatch):
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    fake = FakeExecutor()
    clock = Clock()
    catalog = service.Catalog(tools=effective_tools({}), generators=configured_generators({"load_generators": GENERATORS}))
    started = []

    def starter(start_ms, end_ms, service_id, area, **kwargs):
        if kwargs.get("use_llm") and getattr(starter, "llm_exhausted", False):
            raise LimitError("Дневной лимит запусков с анализом ИИ исчерпан (1), он обновится завтра")
        started.append({"start_ms": start_ms, "end_ms": end_ms, "service": service_id, "area": area, **kwargs})
        return types.SimpleNamespace(job_id=f"job-{len(started)}")

    from loadlens_app.blueprints import dashboard
    from loadlens_app import jobs

    monkeypatch.setattr(dashboard, "_report_step", lambda area, service_id: "30s")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    monkeypatch.setattr(jobs.job_store, "running_counts", lambda owner: (0, 0))

    def no_db():
        raise RuntimeError("watcher tests keep report jobs in memory, the job table is not reachable")

    # A report job of the fake starter is not in memory: looking it up must not reach the configured database.
    monkeypatch.setattr(jobs, "_ts_conn", no_db)
    monkeypatch.setattr(watcher, "report_filter", lambda run, tool_set, start, end: (live.run_filter_packs(run, tool_set), ""))
    w = watcher.Watcher(
        store_getter=lambda: memory,
        catalog_loader=lambda: catalog,
        executor_getter=lambda gen: fake,
        report_starter=starter,
        clock=clock,
    )
    return types.SimpleNamespace(store=memory, fake=fake, clock=clock, watcher=w, started=started, starter=starter, catalog=catalog)


def _run(world, instances, **options):
    created = datetime.fromtimestamp(world.clock(), timezone.utc)
    run = RunRecord(
        run_id="r" + str(len(world.store.list(limit=1000)) + 1),
        run_name="release-2.5",
        run_tag="release-2-5",
        service="checkout",
        project_area="shop",
        test_type="step",
        owner="eng",
        options={"auto_report": True, "use_llm": True, "max_seconds": 3600, **options},
        instances=instances,
        created_at=created,
    )
    world.store.create(run)
    return run.run_id


def _inst(index, generator="gen-1", tool="k6", **extra):
    return InstanceRecord(index=index, name=f"i{index}", tag=f"i{index}", generator=generator, tool=tool, script="a.js", params="--vus 1", **extra)


def _ref(run_id, index):
    return f"/opt/loadlens/runs/{run_id}/{index + 1:02d}-i{index}"


def test_watcher_runs_an_instance_through_to_the_report(world):
    from loadlens_app import jobs

    run_id = _run(world, [_inst(0, env={"TOKEN": "secret"})])
    world.watcher.tick()
    ref = _ref(run_id, 0)
    command, max_seconds = world.fake.launched[ref]
    assert max_seconds == 3600 and command.env["TOKEN"] == "secret" and command.env["JAVA_OPTS"] == "-Xmx2g"
    run = world.store.get(run_id)
    assert run.status == store.RUN_RUNNING and run.instances[0].status == store.INST_STARTING
    assert "secret" not in run.instances[0].command

    t0 = world.clock()
    world.fake.state[ref] = Probe(state="running", started_at=t0, remote_now=t0)
    world.clock.advance(3)
    world.watcher.tick()
    assert world.store.get(run_id).instances[0].status == store.INST_RUNNING

    world.fake.state[ref] = Probe(state="exited", started_at=t0, finished_at=t0 + 600, exit_code=99, remote_now=t0 + 601)
    world.clock.advance(600)
    world.watcher.tick()
    run = world.store.get(run_id)
    inst = run.instances[0]
    assert inst.status == store.INST_CHECKS_FAILED and inst.exit_code == 99 and inst.env == {"TOKEN": "***"}
    assert run.status == store.RUN_DRAINING and not world.started

    world.clock.advance(61)
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.status == store.RUN_REPORTING and run.report_job_id == "job-1"
    report = world.started[0]
    assert (report["start_ms"], report["end_ms"]) == (int(t0 * 1000), int((t0 + 600) * 1000))
    assert report["run_tag"] == "release-2-5" and "lt-k6-influxql" in report["run_filter_packs"]
    assert report["use_llm"] and report["web_only"] and report["test_type"] == "step" and report["area"] == "shop"

    jobs.job_store._jobs["job-1"] = jobs.JobRecord(job_id="job-1", kind="report", status=jobs.JOB_STATUS_DONE, page_url="/reports/abc")
    try:
        world.watcher.tick()
    finally:
        jobs.job_store._jobs.pop("job-1", None)
    run = world.store.get(run_id)
    assert run.status == store.RUN_DONE and run.report_url == "/reports/abc"
    assert world.watcher.tick() == 0


def test_stop_sends_the_signal_then_kills_after_the_grace_period(world):
    run_id = _run(world, [_inst(0)])
    world.watcher.tick()
    ref = _ref(run_id, 0)
    t0 = world.clock()
    world.fake.state[ref] = Probe(state="running", started_at=t0)
    world.watcher.tick()
    world.store.request_stop(run_id)
    world.watcher.tick()
    assert world.fake.signals == [(ref, "TERM")] and world.store.get(run_id).status == store.RUN_STOPPING
    world.clock.advance(10)
    world.watcher.tick()
    assert world.fake.killed == []
    world.clock.advance(25)
    world.watcher.tick()
    assert world.fake.killed == [ref]
    world.fake.state[ref] = Probe(state="exited", started_at=t0, finished_at=t0 + 40, exit_code=137)
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.instances[0].status == store.INST_STOPPED and run.status == store.RUN_DRAINING


def test_stopping_one_instance_leaves_the_others_running(world):
    run_id = _run(world, [_inst(0), _inst(1, start_delay_sec=600)])
    world.watcher.tick()
    assert _ref(run_id, 1) not in world.fake.launched
    assert world.store.get(run_id).instances[1].message.startswith("Старт через")
    world.store.request_stop(run_id, 1)
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.instances[1].status == store.INST_CANCELLED and run.instances[0].status == store.INST_STARTING
    assert run.status == store.RUN_RUNNING and world.fake.signals == []


def test_delayed_instance_starts_after_its_delay(world):
    run_id = _run(world, [_inst(0), _inst(1, start_delay_sec=120)])
    world.watcher.tick()
    assert list(world.fake.launched) == [_ref(run_id, 0)]
    world.clock.advance(121)
    world.watcher.tick()
    assert _ref(run_id, 1) in world.fake.launched


def test_run_fails_when_nothing_starts(world):
    run_id = _run(world, [_inst(0), _inst(1)])
    world.fake.fail_launch = {_ref(run_id, 0), _ref(run_id, 1)}
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.status == store.RUN_FAILED and "ssh refused" in run.message and not world.started
    assert all(inst.status == store.INST_FAILED for inst in run.instances)


def test_quick_failure_is_not_part_of_the_report_window(world):
    run_id = _run(world, [_inst(0), _inst(1)])
    world.watcher.tick()
    t0 = world.clock()
    world.fake.state[_ref(run_id, 0)] = Probe(state="exited", started_at=t0, finished_at=t0 + 1, exit_code=127)
    world.fake.state[_ref(run_id, 1)] = Probe(state="exited", started_at=t0 + 5, finished_at=t0 + 300, exit_code=0)
    world.clock.advance(400)
    world.watcher.tick()
    world.watcher.tick()
    report = world.started[0]
    assert report["start_ms"] == int((t0 + 5) * 1000)
    assert world.store.get(run_id).instances[0].message.startswith("Команда инструмента не найдена")


def test_manual_generator_marks_the_window_only(world):
    run_id = _run(world, [_inst(0, generator="manual", tool="")], use_llm=False)
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.instances[0].status == store.INST_RUNNING and world.fake.launched == {}
    world.clock.advance(900)
    world.store.request_stop(run_id)
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.instances[0].status == store.INST_FINISHED and run.status == store.RUN_DRAINING
    world.clock.advance(61)
    world.watcher.tick()
    report = world.started[0]
    assert report["end_ms"] - report["start_ms"] == 900_000 and report["run_tag"] is None and not report["use_llm"]


def test_runner_that_vanishes_is_lost_after_two_probes(world):
    run_id = _run(world, [_inst(0)], auto_report=False)
    world.watcher.tick()
    ref = _ref(run_id, 0)
    world.fake.state[ref] = Probe(state="dead", started_at=world.clock())
    world.watcher.tick()
    assert world.store.get(run_id).instances[0].status == store.INST_STARTING
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.instances[0].status == store.INST_LOST and run.status == store.RUN_DONE and not world.started


def test_unreachable_generator_keeps_the_instance_until_the_limit(world):
    run_id = _run(world, [_inst(0)], max_seconds=60)
    world.watcher.tick()
    world.fake.unreachable = True
    world.clock.advance(30)
    world.watcher.tick()
    inst = world.store.get(run_id).instances[0]
    assert inst.status == store.INST_STARTING and "Нет связи" in inst.message
    world.clock.advance(60 + watcher.UNREACHABLE_GRACE_SEC)
    world.watcher.tick()
    assert world.store.get(run_id).instances[0].status == store.INST_LOST


def test_another_watcher_takes_over_a_silent_one(world):
    run_id = _run(world, [_inst(0)])
    world.watcher.tick()
    other = watcher.Watcher(
        store_getter=lambda: world.store,
        catalog_loader=lambda: world.catalog,
        executor_getter=lambda gen: world.fake,
        report_starter=world.starter,
        clock=world.clock,
    )
    assert other.tick() == 0
    world.store._runs[run_id].watcher_seen = datetime.now(timezone.utc) - timedelta(seconds=watcher.STALE_AFTER_SEC + 5)
    t0 = world.clock()
    world.fake.state[_ref(run_id, 0)] = Probe(state="running", started_at=t0)
    assert other.tick() == 1
    run = world.store.get(run_id)
    assert run.watcher == other.id and run.instances[0].status == store.INST_RUNNING
    assert world.fake.launched.keys() == {_ref(run_id, 0)}


def test_report_without_llm_when_the_daily_budget_is_spent(world):
    world.starter.llm_exhausted = True
    run_id = _run(world, [_inst(0)])
    world.watcher.tick()
    t0 = world.clock()
    world.fake.state[_ref(run_id, 0)] = Probe(state="exited", started_at=t0, finished_at=t0 + 120, exit_code=0)
    world.clock.advance(300)
    world.watcher.tick()  # the instance ends and the drain time is already over: the report starts at once
    run = world.store.get(run_id)
    assert run.status == store.RUN_REPORTING and "анализ ИИ пропущен" in run.message
    assert world.started[0]["use_llm"] is False


def test_report_waits_for_a_free_job_slot(world, monkeypatch):
    from loadlens_app import jobs

    run_id = _run(world, [_inst(0)])
    world.watcher.tick()
    t0 = world.clock()
    world.fake.state[_ref(run_id, 0)] = Probe(state="exited", started_at=t0, finished_at=t0 + 60, exit_code=0)
    world.clock.advance(300)
    monkeypatch.setattr(jobs.job_store, "running_counts", lambda owner: (99, 0))
    world.watcher.tick()
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.status == store.RUN_DRAINING and "слота" in run.message and not world.started
    monkeypatch.setattr(jobs.job_store, "running_counts", lambda owner: (0, 0))
    world.watcher.tick()
    assert world.store.get(run_id).status == store.RUN_REPORTING


def test_memory_store_keeps_stop_requests_written_meanwhile():
    memory = MemoryLoadTestStore()
    memory.create(RunRecord(run_id="r1", run_name="a", run_tag="a", service="s", instances=[_inst(0)]))
    claimed = memory.claim("w1", 30)[0]
    memory.request_stop("r1", 0)
    claimed.message = "progress"
    assert memory.save_progress(claimed, "w1")
    saved = memory.get("r1")
    assert saved.message == "progress" and saved.stop_instances == [0]
    assert not memory.save_progress(claimed, "w2")
    assert memory.name_in_use("a", "s") and not memory.name_in_use("a", "other")


def test_instance_json_round_trip():
    inst = _inst(2, env={"A": "1"}, started_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    again = InstanceRecord.from_json(inst.to_json())
    assert again == inst


# ---- planning a run ---------------------------------------------------------------------------------


@pytest.fixture
def planner(monkeypatch):
    from loadlens_app import core
    from loadlens_app.blueprints import dashboard

    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    monkeypatch.setitem(core.CONFIG, "load_generators", copy.deepcopy(GENERATORS))
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setitem(core.CONFIG, "limits", {})
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop" if service_id == "checkout" else (_ for _ in ()).throw(dashboard.ReportRequestError("Сервис не найден")))
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: name == "taken")
    monkeypatch.setattr(service, "_installed_lt_packs", lambda area, service_id: ["lt-k6-prometheus-rw"])
    return memory


def _form(**overrides):
    form = {
        "service": "checkout",
        "test_type": "step",
        "run_name": "release 2.5",
        "instances": [
            {"name": "rest", "generator": "gen-1", "tool": "k6", "script": "checkout.js", "params": "--vus 10", "env": "TOKEN=s3cret-zz"},
            {"name": "kafka", "generator": "gen-1", "tool": "gatling", "script": "acme.KafkaSimulation", "start_delay_sec": 30},
        ],
    }
    form.update(overrides)
    return form


def test_plan_builds_commands_and_warns(planner):
    planned = service.plan_run(_form(), "eng")
    run = planned.run
    assert run.run_tag == "release-2-5" and run.project_area == "shop" and run.options["max_seconds"] == 24 * 3600
    rest, kafka = run.instances
    assert rest.ref == f"/opt/loadlens/runs/{run.run_id}/01-rest" and rest.env == {"TOKEN": "s3cret-zz"}
    assert "TOKEN=***" in rest.command and "s3cret-zz" not in rest.command
    assert "/opt/gatling/bin/gatling.sh -rm local -s acme.KafkaSimulation" in kafka.command and kafka.start_delay_sec == 30
    assert any("Gatling" in w and "пакет запросов" in w for w in planned.warnings)
    assert any("отделяются" in w and "Gatling" in w for w in planned.warnings)


@pytest.mark.parametrize("patch, fragment", [
    ({"service": ""}, "сервис"),
    ({"service": "other"}, "не найден"),
    ({"run_name": "taken"}, "уже занято"),
    ({"run_name": "a/b"}, "недопустимые"),
    ({"test_type": "endurance"}, "Тип теста"),
    ({"instances": []}, "инстанс"),
    ({"instances": [{"generator": "nope", "tool": "k6", "script": "a.js"}]}, "генератор"),
    ({"instances": [{"generator": "gen-1", "tool": "nope", "script": "a.js"}]}, "инструмент"),
    ({"instances": [{"generator": "gen-1", "tool": "k6", "script": ""}]}, "скрипт"),
    ({"instances": [{"generator": "gen-1", "tool": "k6", "script": "a.js", "params": "'x"}]}, "кавычки"),
    ({"instances": [{"generator": "gen-1", "tool": "k6", "script": "a.js", "env": "1X=2"}]}, "1X"),
    ({"instances": [{"generator": "gen-1", "tool": "k6", "script": "a.js", "start_delay_sec": -5}]}, "задержка"),
    ({"max_duration_min": 999999}, "длительность"),
])
def test_plan_rejects_bad_forms(planner, patch, fragment):
    with pytest.raises(service.LaunchError) as error:
        service.plan_run(_form(**patch), "eng")
    assert fragment.lower() in str(error.value).lower()


def test_plan_enforces_run_and_generator_limits(planner, monkeypatch):
    from loadlens_app import core

    monkeypatch.setitem(core.CONFIG, "limits", {"max_parallel_load_tests_per_user": 1})
    planner.create(RunRecord(run_id="busy", run_name="x", run_tag="x", service="checkout", owner="eng", status=store.RUN_RUNNING, instances=[_inst(0, status=store.INST_RUNNING)]))
    with pytest.raises(service.LaunchError) as error:
        service.plan_run(_form(), "eng")
    assert error.value.status == 429
    planned = service.plan_run(_form(), "other")
    assert any("уже идёт прогон «x»" in w for w in planned.warnings)

    generators = copy.deepcopy(GENERATORS)
    generators["gen-1"]["max_parallel"] = 2
    monkeypatch.setitem(core.CONFIG, "load_generators", generators)
    with pytest.raises(service.LaunchError) as error:
        service.plan_run(_form(), "other")
    assert error.value.code == "generator_busy"


def test_manual_instance_needs_no_tool(planner):
    planned = service.plan_run(_form(instances=[{"name": "by hand", "generator": "manual"}]), "eng")
    inst = planned.run.instances[0]
    assert inst.ref.startswith("manual:") and inst.command == "" and planned.commands[0]["manual"]


def test_run_view_hides_environment_values(planner):
    planned = service.create_run(_form(), "eng")
    view = service.run_view(planner.get(planned.run.run_id), with_report=False)
    assert view["instances"][0]["env_keys"] == ["TOKEN"] and "s3cret-zz" not in str(view)
    assert view["status"] == store.RUN_STARTING and view["active"] and view["loading"]


def test_stop_request_validation(planner):
    planned = service.create_run(_form(), "eng")
    run_id = planned.run.run_id
    with pytest.raises(service.LaunchError):
        service.request_stop("missing")
    with pytest.raises(service.LaunchError):
        service.request_stop(run_id, 7)
    assert service.request_stop(run_id, 1).stop_instances == [1]
    planner._runs[run_id].status = store.RUN_DONE
    with pytest.raises(service.LaunchError) as error:
        service.request_stop(run_id)
    assert error.value.status == 409


# ---- live metrics ------------------------------------------------------------------------------------


def test_live_snapshot_reads_roles_and_falls_back_without_the_tag(monkeypatch):
    from AI.query_packs import load_catalog, merge_pack, render_pack

    pack = load_catalog().pack("lt-k6-prometheus-rw")
    block, _ = merge_pack(None, render_pack(pack))
    cfg = {"queries": {"lt_framework": block}}
    monkeypatch.setattr(live, "_scope_config", lambda run: cfg)
    seen = []

    def fake_fetch(cfg_, *, domain, lang, query, start_ts, end_ts, step, label_keys=None):
        seen.append(query)
        if 'testid="release-2-5"' in query:
            return {}
        return {"series": [(start_ts + 10, 5.0), (start_ts + 20, 7.0)]}

    from loadlens_app import query_preview

    monkeypatch.setattr(query_preview, "fetch_points", fake_fetch)
    t0 = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
    run = RunRecord(run_id="r", run_name="release-2.5", run_tag="release-2-5", service="checkout",
                    instances=[_inst(0, launched_at=t0, started_at=t0)], started_at=t0)
    snapshot = live.live_snapshot(run, effective_tools({}), now=t0.timestamp() + 600)
    assert snapshot["ok"] and snapshot["filtered"] is False and snapshot["message"] == live.UNFILTERED_NOTE
    assert snapshot["series"]["lt.rps_total"]["points"][-1][1] == 7.0
    assert any('testid="release-2-5"' in q for q in seen)
    assert set(snapshot["series"]) == set(live.LIVE_ROLES)


def test_live_snapshot_without_lt_queries(monkeypatch):
    monkeypatch.setattr(live, "_scope_config", lambda run: {"queries": {}})
    run = RunRecord(run_id="r", run_name="a", run_tag="a", service="s")
    snapshot = live.live_snapshot(run, effective_tools({}))
    assert not snapshot["ok"] and "пакет запросов" in snapshot["message"]
