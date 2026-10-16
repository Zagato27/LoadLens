"""CI generators run the team's own pipeline: parameters read from GitLab and Jenkins, checked values, launch,
status and the report window cut by the RPS data."""

import copy
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
from loadlens_app.load_tests import ci_params, live, presets, service, store, watcher
from loadlens_app.load_tests.ci_params import CiParam, CiParamError, CiParams
from loadlens_app.load_tests.executors import ApiError, CiExecutor, GeneratorError, LaunchPlan, Probe
from loadlens_app.load_tests.generators import GeneratorConfigError, configured_generators, parse_generator
from loadlens_app.load_tests.store import InstanceRecord, MemoryLoadTestStore, RunRecord
from loadlens_app.load_tests.tools import CommandLine, effective_tools

GITLAB = {
    "title": "GitLab", "type": "gitlab",
    "gitlab": {"url": "https://gitlab.example", "project": "perf/load", "api_token": "API"},
    "loadlens_url": "https://loadlens.example",
}
JENKINS = {
    "title": "Jenkins", "type": "jenkins",
    "jenkins": {"url": "https://jenkins.example", "job": "perf/load-test", "username": "bot", "api_token": "JT"},
}
CONFIG_YAML = """spec:
  inputs:
    environment:
      default: stage
      options: [stage, preprod]
      description: Стенд
    users:
      type: number
      default: 10
    debug:
      type: boolean
      default: false
    tags:
      type: array
      default: [smoke]
    loadlens_run_tag:
      default: ""
    build_id:
      description: Обязательный, без значения по умолчанию
---
include:
  - local: templates/k6.yml
variables:
  PROFILE:
    value: load
    options: [smoke, load, stress]
    description: Профиль нагрузки
  DURATION:
    value: 10m
    description: Длительность
  INTERNAL: plain-without-description
  LOADLENS_RUN_TAG:
    value: ""
    description: Тег прогона от LoadLens
load:
  script:
    - k6 run --tag testid=$LOADLENS_RUN_TAG k6/load.js
  after_script: !reference [.cleanup, script]
"""


class FakeApi:
    """Stands in for ApiClient: answers by (method, path prefix), records the calls."""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []
        self.session = types.SimpleNamespace(headers={})

    def _answer(self, method, path, kwargs):
        for (m, prefix), answer in self.routes.items():
            if m == method and path.startswith(prefix):
                return answer(path, kwargs) if callable(answer) else answer
        return (404, {"message": "404 Not Found"}, b"", {})

    def request(self, method, path, ok=(200, 201, 202, 204), timeout=None, **kwargs):
        self.calls.append((method, path, kwargs))
        status, data, content, headers = (list(self._answer(method, path, kwargs)) + [None, None, None])[:4]
        if status not in tuple(ok):
            raise ApiError(f"HTTP {status}: {data}", status, json.dumps(data))
        text = content.decode() if isinstance(content, bytes) and content else json.dumps(data if data is not None else {})
        return types.SimpleNamespace(status_code=status, content=text.encode(), text=text, headers=headers or {}, json=lambda: data)

    def json(self, method, path, **kwargs):
        response = self.request(method, path, **kwargs)
        return response.json() if response.json() is not None else {}


def _gitlab_routes(graphql, version="17.11.2-ee"):
    return {
        ("GET", "/api/v4/version"): (200, {"version": version}),
        ("GET", "/api/v4/projects/perf%2Fload/repository/branches/"): (200, {"name": "main", "commit": {"id": "abc123"}}),
        ("GET", "/api/v4/projects/perf%2Fload/repository/files/.gitlab-ci.yml/raw"): (200, None, CONFIG_YAML.encode()),
        ("GET", "/api/v4/projects/perf%2Fload/ci/lint"): (404, {}),
        ("GET", "/api/v4/projects/perf%2Fload/repository/branches"): (200, [{"name": "main"}, {"name": "feature/x"}]),
        ("GET", "/api/v4/projects/perf%2Fload"): (200, {"path_with_namespace": "perf/load", "default_branch": "main", "ci_config_path": ""}),
        ("POST", "/api/graphql"): graphql,
    }


def _gen(raw=GITLAB, gid="gitlab"):
    return parse_generator(gid, copy.deepcopy(raw))


# ---- generator settings ----------------------------------------------------------------------------------


def test_ci_generator_settings():
    gitlab = _gen()
    assert gitlab.ci.ref == "" and gitlab.ci.trigger_token == "" and gitlab.tools == {} and not gitlab.offers("k6")
    public = gitlab.to_public(effective_tools({}))
    assert public["tools"] == [] and public["ci"] == {"kind": "gitlab", "ref": "", "reads_params": True, "extra_vars": True,
                                                      "tags_metrics": False, "events": True}
    trigger_only = copy.deepcopy(GITLAB)
    trigger_only["gitlab"] = {"url": "https://g", "project": "p", "trigger_token": "T"}
    assert _gen(trigger_only).ci.ref == "main" and not _gen(trigger_only).to_public({})["ci"]["reads_params"]
    no_events = {**GITLAB, "loadlens_url": "", "param_options": {"ENV": "stage, preprod"}, "tags_metrics": True}
    parsed = _gen(no_events)
    assert parsed.ci.loadlens_url == "" and parsed.ci.param_options == {"ENV": ("stage", "preprod")} and parsed.ci.tags_metrics
    for broken, fragment in (
        ({**GITLAB, "gitlab": {"url": "https://g", "project": "p"}}, "API-токен"),
        ({**GITLAB, "gitlab": {**GITLAB["gitlab"], "inputs": "maybe"}}, "inputs"),
        ({**GITLAB, "param_options": {"ENV": ""}}, "вариант"),
        ({**JENKINS, "jenkins": {**JENKINS["jenkins"], "ref": "a b"}}, "ветка"),
    ):
        with pytest.raises(GeneratorConfigError, match=fragment):
            parse_generator("g", broken)


# ---- reading GitLab parameters --------------------------------------------------------------------------


def test_gitlab_graphql_variables_are_asked_again_while_gitlab_computes_them():
    answers = iter([
        (200, {"data": {"project": {"ciConfigVariables": None}}}),
        (200, {"data": {"project": {"ciConfigVariables": [
            {"key": "PROFILE", "value": "load", "valueOptions": ["smoke", "load", "stress"], "description": "Профиль нагрузки"},
            {"key": "LOADLENS_RUN_TAG", "value": "", "valueOptions": [], "description": "Тег"},
        ]}}}),
    ])
    api = FakeApi(_gitlab_routes(lambda path, kwargs: next(answers)))
    slept = []
    params = ci_params.read_gitlab(api, _gen().ci, "", sleep=slept.append)
    assert params.source == "graphql" and params.ref == "main" and slept == [ci_params.GRAPHQL_RETRY_SEC]
    profile = params.get("PROFILE")
    assert profile.type == "choice" and profile.options == ("smoke", "load", "stress") and profile.default == "load"
    assert params.get("LOADLENS_RUN_TAG") is None and "LOADLENS_RUN_TAG" in params.auto
    # spec:inputs come from the file in any case; GitLab 17.11 takes them through the API.
    assert params.inputs_api and params.get("environment").options == ("stage", "preprod")
    assert params.get("loadlens_run_tag") is None and params.auto_inputs == ["loadlens_run_tag"]
    graphql = [call for call in api.calls if call[1] == "/api/graphql"]
    assert graphql[0][2]["headers"] == {"Authorization": "Bearer API"} and graphql[0][2]["json"]["variables"] == {"path": "perf/load", "value": "main"}


def test_gitlab_graphql_falls_back_to_older_schemas():
    seen = []

    def graphql(path, kwargs):
        query = kwargs["json"]["query"]
        seen.append(query)
        if "valueOptions" in query or "ref:" in query:
            return (200, {"errors": [{"message": "Field 'ciConfigVariables' doesn't accept argument 'ref'"}]})
        return (200, {"data": {"project": {"ciConfigVariables": [{"key": "DURATION", "value": "10m", "description": "Длительность"}]}}})

    params = ci_params.read_gitlab(FakeApi(_gitlab_routes(graphql, version="15.4.0")), _gen().ci, "main", sleep=lambda s: None)
    # ref + options, ref, sha + options, then sha without options answered (GitLab 15.x).
    assert params.source == "graphql" and len(seen) == 4 and "sha:" in seen[-1] and "valueOptions" not in seen[-1]
    assert [param.name for param in params.params if param.kind == "variable"] == ["DURATION"]
    assert not params.inputs_api and any("не принимает входные параметры" in note for note in params.notes)


def test_gitlab_without_graphql_reads_the_lint_merge_and_then_the_file():
    routes = _gitlab_routes((404, {"message": "404 Not Found"}), version="14.10.0")
    routes[("GET", "/api/v4/projects/perf%2Fload/ci/lint")] = (200, {"valid": True, "merged_yaml": "variables:\n  FROM_INCLUDE:\n    value: x\n    description: Из include\n"})
    params = ci_params.read_gitlab(FakeApi(routes), _gen().ci, "main", sleep=lambda s: None)
    assert params.source == "lint" and [p.name for p in params.params if p.kind == "variable"] == ["FROM_INCLUDE"]
    routes[("GET", "/api/v4/projects/perf%2Fload/ci/lint")] = (404, {})
    params = ci_params.read_gitlab(FakeApi(routes), _gen().ci, "main", sleep=lambda s: None)
    # The file: described variables only (as the Run pipeline form), GitLab tags do not break the YAML.
    assert params.source == "file" and [p.name for p in params.params if p.kind == "variable"] == ["PROFILE", "DURATION"]
    inputs = {p.name: p for p in params.params if p.kind == "input"}
    assert inputs["users"].type == "number" and inputs["users"].default == 10 and not inputs["users"].required
    assert inputs["debug"].type == "boolean" and inputs["tags"].type == "array" and inputs["build_id"].required
    assert inputs["build_id"].to_public()["default"] == ""


def test_gitlab_without_api_token_reads_nothing():
    raw = copy.deepcopy(GITLAB)
    raw["gitlab"] = {"url": "https://g", "project": "p", "trigger_token": "T", "ref": "release"}
    api = FakeApi()
    params = ci_params.read_gitlab(api, _gen(raw).ci, "")
    assert params.source == "none" and params.ref == "release" and api.calls == [] and "API-токен" in params.notes[0]
    assert ci_params.gitlab_refs(api, _gen(raw).ci) == ([], "release")


def test_inputs_through_the_api_by_version_and_setting():
    cfg = _gen().ci
    assert not ci_params.inputs_supported(cfg, (17, 10)) and ci_params.inputs_supported(cfg, (17, 11)) and ci_params.inputs_supported(cfg, (18, 0))
    on = _gen({**GITLAB, "gitlab": {**GITLAB["gitlab"], "inputs": "on"}}).ci
    off = _gen({**GITLAB, "gitlab": {**GITLAB["gitlab"], "inputs": "off"}}).ci
    assert ci_params.inputs_supported(on, (16, 0)) and not ci_params.inputs_supported(off, (18, 0))
    assert ci_params.parse_version("16.11.3-ee") == (16, 11, 3) and ci_params.parse_version("") == ()


def test_gitlab_branches():
    names, default = ci_params.gitlab_refs(FakeApi(_gitlab_routes((404, {}))), _gen().ci, "fea")
    assert names == ["main", "feature/x"] and default == "main"


# ---- reading Jenkins parameters ---------------------------------------------------------------------------

JENKINS_DEFINITIONS = {"_class": "hudson.model.FreeStyleProject", "property": [{"_class": "hudson.model.ParametersDefinitionProperty", "parameterDefinitions": [
    {"_class": "hudson.model.StringParameterDefinition", "name": "VUS", "type": "StringParameterDefinition", "description": "<b>Пользователи</b>", "defaultParameterValue": {"value": "10"}},
    {"_class": "hudson.model.ChoiceParameterDefinition", "name": "PROFILE", "type": "ChoiceParameterDefinition", "choices": ["smoke", "load"], "defaultParameterValue": {"value": "smoke"}},
    {"_class": "hudson.model.BooleanParameterDefinition", "name": "DEBUG", "type": "BooleanParameterDefinition", "defaultParameterValue": {"value": False}},
    {"_class": "hudson.model.PasswordParameterDefinition", "name": "API_KEY", "type": "PasswordParameterDefinition", "defaultParameterValue": {"value": "<concealed>"}},
    {"_class": "hudson.model.TextParameterDefinition", "name": "NOTES", "type": "TextParameterDefinition", "defaultParameterValue": {"value": ""}},
    {"_class": "org.biouno.unochoice.ChoiceParameter", "name": "STAND", "type": "ChoiceParameter"},
    {"_class": "org.biouno.unochoice.CascadeChoiceParameter", "name": "HOST", "type": "CascadeChoiceParameter"},
    {"_class": "hudson.model.FileParameterDefinition", "name": "DATA", "type": "FileParameterDefinition"},
    {"_class": "hudson.model.StringParameterDefinition", "name": "LOADLENS_RUN_TAG", "type": "StringParameterDefinition", "defaultParameterValue": {"value": ""}},
]}]}
JENKINS_CONFIG = """<?xml version='1.1' encoding='UTF-8'?>
<project><properties><hudson.model.ParametersDefinitionProperty><parameterDefinitions>
<org.biouno.unochoice.ChoiceParameter plugin="uno-choice@2.8"><name>STAND</name><choiceType>PT_SINGLE_SELECT</choiceType>
<script class="org.biouno.unochoice.model.GroovyScript"><secureScript plugin="script-security"><script>return ['stage', 'preprod:selected', 'prod:disabled']</script></secureScript></script></org.biouno.unochoice.ChoiceParameter>
<org.biouno.unochoice.CascadeChoiceParameter><name>HOST</name><choiceType>PT_CHECKBOX</choiceType>
<script class="org.biouno.unochoice.model.GroovyScript"><secureScript><script>return hosts[STAND]</script></secureScript></script></org.biouno.unochoice.CascadeChoiceParameter>
</parameterDefinitions></hudson.model.ParametersDefinitionProperty></properties></project>"""


def _jenkins_routes(definitions=JENKINS_DEFINITIONS):
    return {
        ("GET", "/job/perf/job/load-test/config.xml"): (200, None, JENKINS_CONFIG.encode()),
        ("GET", "/job/perf/job/load-test/api/json"): lambda path, kwargs: (200, definitions if "property" in kwargs["params"]["tree"] else {"_class": definitions["_class"]}),
    }


def test_jenkins_parameters_of_every_type():
    params = ci_params.read_jenkins(FakeApi(_jenkins_routes()), _gen(JENKINS, "jenkins").ci)
    by = {param.name: param for param in params.params}
    assert params.source == "jenkins" and params.parameterized and params.ref == "" and params.auto == ["LOADLENS_RUN_TAG"]
    assert by["VUS"].type == "string" and by["VUS"].default == "10" and by["VUS"].description == "Пользователи"
    assert by["PROFILE"].options == ("smoke", "load") and by["DEBUG"].type == "boolean" and by["DEBUG"].default is False
    assert by["API_KEY"].secret and by["API_KEY"].to_public()["default"] == "" and by["NOTES"].type == "text"
    # Active Choices: a literal list is read from config.xml; a computed one stays free text.
    assert by["STAND"].options == ("stage", "preprod") and by["STAND"].default == "preprod" and "Active Choices" in by["STAND"].note
    assert by["HOST"].type == "string" and "по другим параметрам" in by["HOST"].note
    assert "DATA" not in by and any("Файловый параметр DATA" in note for note in params.notes)


def test_active_choices_options_from_the_generator_settings():
    raw = {**JENKINS, "param_options": {"HOST": ["gen-1", "gen-2"]}}
    params = ci_params.read_jenkins(FakeApi(_jenkins_routes()), _gen(raw, "jenkins").ci)
    host = params.get("HOST")
    assert host.type == "choice" and host.options == ("gen-1", "gen-2") and host.default == "gen-1" and "настроек генератора" in host.note


def test_literal_choices():
    assert ci_params.literal_choices("return ['a', \"b\"]") == (["a", "b"], "")
    assert ci_params.literal_choices("['x:selected', 'y']") == (["x", "y"], "x")
    assert ci_params.literal_choices("def a = 1\nreturn [a]") is None and ci_params.literal_choices("return []") is None


def test_jenkins_multibranch_branches():
    routes = {
        ("GET", "/job/perf/job/load-test/job/release/api/json"): (200, {"_class": "org.jenkinsci.plugins.workflow.job.WorkflowJob", "property": []}),
        ("GET", "/job/perf/job/load-test/api/json"): (200, {"_class": "org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject",
                                                             "jobs": [{"name": "main"}, {"name": "release"}]}),
    }
    params = ci_params.read_jenkins(FakeApi(routes), _gen(JENKINS, "jenkins").ci, "release")
    assert params.ref == "release" and params.refs == ["main", "release"] and not params.parameterized
    assert params.job_path == "/job/perf/job/load-test/job/release" and any("после первой сборки" in note for note in params.notes)
    assert ci_params.jenkins_refs(FakeApi(routes), _gen(JENKINS, "jenkins").ci) == (["main", "release"], "main")


# ---- values ----------------------------------------------------------------------------------------------


def _params():
    return CiParams(params=[
        CiParam("PROFILE", "variable", "choice", default="load", options=("smoke", "load")),
        CiParam("DURATION", "variable", default="10m"),
        CiParam("users", "input", "number", default=10),
        CiParam("debug", "input", "boolean", default=False),
        CiParam("tags", "input", "array", default=["smoke"]),
        CiParam("build_id", "input", required=True),
        CiParam("version", "input", regex="^v[0-9]+$", default="v1"),
    ], source="graphql", inputs_api=True, auto=["LOADLENS_RUN_TAG", "loadlens_run_tag"], auto_inputs=["loadlens_run_tag"], ref="main")


@pytest.mark.parametrize("values, extra, fragment", [
    ({"NOPE": "1", "build_id": "7"}, {}, "нет параметра NOPE"),
    ({"LOADLENS_RUN_TAG": "x"}, {}, "LoadLens заполняет сам"),
    ({"PROFILE": "huge", "build_id": "7"}, {}, "нет среди вариантов"),
    ({"users": "many", "build_id": "7"}, {}, "ожидается число"),
    ({"debug": "maybe", "build_id": "7"}, {}, "да или нет"),
    ({"version": "1.0", "build_id": "7"}, {}, "шаблон"),
    ({}, {}, "build_id обязателен"),
    ({"build_id": "7"}, {"PROFILE": "x"}, "параметр пайплайна"),
    ({"build_id": "7"}, {"LOADLENS_TOKEN": "x"}, "LoadLens заполняет сам"),
    ({"DURATION": "1m\n2m", "build_id": "7"}, {}, "одну строку"),
])
def test_values_are_checked_against_the_pipeline(values, extra, fragment):
    with pytest.raises(CiParamError, match=fragment):
        ci_params.check_values(_params(), values, extra, "gitlab")


def test_only_changed_values_go_to_gitlab_and_inputs_keep_their_types():
    values, extra = ci_params.check_values(_params(), {"PROFILE": "load", "DURATION": "5m", "users": "25", "debug": True,
                                                       "tags": "smoke, api", "build_id": "7", "version": ""}, {"EXTRA": "1"}, "gitlab")
    assert values["debug"] == "true" and values["tags"] == "smoke, api"
    auto = {"LOADLENS_RUN_TAG": "nightly", "LOADLENS_RUN_ID": "r1"}
    split = ci_params.launch_values(_params(), values, extra, "gitlab", auto)
    # PROFILE keeps its default and is not sent: projects that forbid pipeline variables still start.
    assert split["variables"] == {"DURATION": "5m", "EXTRA": "1"}
    assert split["inputs"] == {"users": 25, "debug": True, "tags": ["smoke", "api"], "build_id": "7", "loadlens_run_tag": "nightly"}
    assert split["auto"] == auto
    no_api = replace_params(inputs_api=False)
    assert ci_params.launch_values(no_api, values, extra, "gitlab", auto)["inputs"] == {}


def replace_params(**changes):
    params = _params()
    for key, value in changes.items():
        setattr(params, key, value)
    return params


def test_jenkins_values_and_free_values_without_definitions():
    jenkins = CiParams(params=[CiParam("VUS", "parameter", default="10"), CiParam("API_KEY", "parameter", "password", secret=True)],
                       source="jenkins", auto=["LOADLENS_RUN_TAG"], parameterized=True)
    values, _ = ci_params.check_values(jenkins, {"VUS": "10", "API_KEY": ""}, {}, "jenkins")
    split = ci_params.launch_values(jenkins, values, {}, "jenkins", {"LOADLENS_RUN_TAG": "t", "LOADLENS_RUN_ID": "r"})
    # The default VUS stays as written, an empty password keeps the CI's secret, declared LOADLENS_* are filled.
    assert split == {"parameters": {"LOADLENS_RUN_TAG": "t"}}
    with pytest.raises(CiParamError, match="только параметры"):
        ci_params.check_values(jenkins, {}, {"X": "1"}, "jenkins")
    free = CiParams(source="none")
    values, extra = ci_params.check_values(free, {"PROFILE": "load"}, {"B": "2"}, "gitlab")
    assert ci_params.launch_values(free, values, extra, "gitlab", {})["variables"] == {"PROFILE": "load", "B": "2"}
    with pytest.raises(CiParamError, match="Недопустимое имя"):
        ci_params.check_values(free, {"bad name": "1"}, {}, "gitlab")


def test_describe_masks_secrets():
    line = ci_params.describe("gitlab", "perf/load", "main", {"VUS": "10", "KEY": "s3cret-zz"}, {"EXTRA": "1"}, ["KEY"])
    assert line == "GitLab perf/load @ main: VUS=10, KEY=***, EXTRA=1"
    assert ci_params.describe("jenkins", "perf/job", "", {}, {}, []) == "Jenkins perf/job"


# ---- launch and status -----------------------------------------------------------------------------------


def _ci_plan(**changes):
    values = dict(ref="r", command=CommandLine(argv=(), env={}, cwd=""), max_seconds=600, run_id="run1", instance_index=0,
                  ci_ref="main", ci_variables={"DURATION": "5m"}, ci_inputs={"users": 25}, ci_auto={"LOADLENS_RUN_TAG": "nightly"})
    values.update(changes)
    return LaunchPlan(**values)


def test_gitlab_pipeline_with_variables_and_inputs():
    api = FakeApi({("POST", "/api/v4/projects/perf%2Fload/pipeline"): (201, {"id": 77, "web_url": "https://bot:x@gitlab.example/perf/load/-/pipelines/77"})})
    result = CiExecutor(_gen(), events=lambda r, i: [], api=api).launch(_ci_plan())
    body = api.calls[0][2]["json"]
    assert body["ref"] == "main" and body["inputs"] == {"users": 25}
    assert body["variables"] == [{"key": "LOADLENS_RUN_TAG", "value": "nightly"}, {"key": "DURATION", "value": "5m"}]
    assert result.ref == "gitlab:run1:0:77" and result.url == "https://gitlab.example/perf/load/-/pipelines/77"


def test_gitlab_project_that_forbids_pipeline_variables():
    def create(path, kwargs):
        if kwargs["json"]["variables"]:
            return (400, {"message": {"base": ["Insufficient permissions to set pipeline variables"]}})
        return (201, {"id": 78, "web_url": ""})

    api = FakeApi({("POST", "/api/v4/projects/perf%2Fload/pipeline"): create})
    executor = CiExecutor(_gen(), events=lambda r, i: [], api=api)
    result = executor.launch(_ci_plan(ci_variables={}))
    assert result.ref.endswith(":78") and "LOADLENS_* не переданы" in result.message and api.calls[-1][2]["json"]["inputs"] == {"users": 25}
    with pytest.raises(GeneratorError, match="запрещает переменные"):
        executor.launch(_ci_plan())


def test_gitlab_trigger_token_only():
    raw = copy.deepcopy(GITLAB)
    raw["gitlab"] = {"url": "https://gitlab.example", "project": "perf/load", "trigger_token": "TRIG"}
    api = FakeApi({("POST", "/api/v4/projects/perf%2Fload/trigger/pipeline"): (201, {"id": 5})})
    executor = CiExecutor(_gen(raw), events=lambda r, i: [], api=api)
    executor.launch(_ci_plan(ci_inputs={}))
    form = api.calls[0][2]["data"]
    assert form["token"] == "TRIG" and form["ref"] == "main" and form["variables[DURATION]"] == "5m" and form["variables[LOADLENS_RUN_TAG]"] == "nightly"
    executor.launch(_ci_plan())
    assert api.calls[-1][2]["json"]["inputs"] == {"users": 25} and api.calls[-1][2]["json"]["variables"]["DURATION"] == "5m"
    # Without the API only the events say anything: the window counts from the launch.
    probe = executor.probe("gitlab:run1:0:5")
    assert probe.state == "running" and probe.started_at is None and "Без API-токена" in probe.message
    assert not executor.supports_stop


def test_gitlab_status_times_and_page():
    pipeline = {"status": "pending", "web_url": "https://gitlab.example/perf/load/-/pipelines/77"}
    api = FakeApi({("GET", "/api/v4/projects/perf%2Fload/pipelines/77"): lambda path, kwargs: (200, dict(pipeline))})
    events = []
    executor = CiExecutor(_gen(), events=lambda r, i: events, api=api)
    first = executor.probe("gitlab:run1:0:77")
    assert first.state == "starting" and "pending" in first.message and first.url.endswith("/77")
    pipeline.update(status="running", started_at="2026-10-15T10:00:00.000Z")
    running = executor.probe("gitlab:run1:0:77")
    assert running.state == "running" and running.started_at == datetime(2026, 10, 15, 10, tzinfo=timezone.utc).timestamp()
    pipeline.update(status="success", finished_at="2026-10-15T10:12:00Z")
    done = executor.probe("gitlab:run1:0:77")
    assert done.state == "exited" and done.exit_code == 0 and done.finished_at - done.started_at == 720
    pipeline.update(status="canceled")
    assert executor.probe("gitlab:run1:0:77").exit_code is None
    # Events of the pipeline are more exact than the API times.
    events.extend([{"event": "started", "at": "2026-10-15T10:02:00+00:00"}, {"event": "finished", "exit_code": 99, "at": "2026-10-15T10:10:00+00:00"}])
    exact = executor.probe("gitlab:run1:0:77")
    assert exact.exit_code == 99 and exact.finished_at - exact.started_at == 480


def test_jenkins_build_with_parameters_or_plain_build():
    location = {"Location": "https://jenkins.example/queue/item/5/"}
    api = FakeApi({
        ("POST", "/job/perf/job/load-test/buildWithParameters"): (201, {}, b"", location),
        ("POST", "/job/perf/job/load-test/job/release/build"): (201, {}, b"", location),
        ("GET", "/queue/item/5/api/json"): (200, {"executable": {"url": "https://jenkins.example/job/perf/job/load-test/12/"}}),
        ("GET", "/job/perf/job/load-test/12/api/json"): (200, {"building": False, "result": "SUCCESS", "timestamp": 1_760_000_000_000, "duration": 600_000}),
    })
    executor = CiExecutor(_gen(JENKINS, "jenkins"), events=lambda r, i: [], api=api)
    result = executor.launch(_ci_plan(ci_ref="", ci_parameters={"VUS": "20", "LOADLENS_RUN_TAG": "nightly"}))
    assert api.calls[0][2]["data"] == {"VUS": "20", "LOADLENS_RUN_TAG": "nightly"} and result.ref == "jenkins:run1:0:queue:/queue/item/5"
    probe = executor.probe(result.ref)
    assert probe.state == "exited" and probe.exit_code == 0 and probe.started_at == 1_760_000_000 and probe.finished_at == 1_760_000_600
    assert probe.ref == "jenkins:run1:0:/job/perf/job/load-test/12" and probe.url == "https://jenkins.example/job/perf/job/load-test/12"
    executor.launch(_ci_plan(ci_ref="release", ci_parameterized=False))
    assert api.calls[-1][1] == "/job/perf/job/load-test/job/release/build"


# ---- check of the settings page ----------------------------------------------------------------------------


def test_ci_check_lists_the_pipeline_parameters(monkeypatch):
    from loadlens_app.load_tests.executors import ci

    answers = iter([(200, {"data": {"project": {"ciConfigVariables": [{"key": "PROFILE", "value": "load", "valueOptions": [], "description": "p"}]}}})])
    api = FakeApi(_gitlab_routes(lambda path, kwargs: next(answers)))
    monkeypatch.setattr(ci, "ApiClient", lambda *args, **kwargs: api)
    report = ci.check_ci_generator(_gen(), {})
    details = report.to_details()
    assert report.ok and report.connected and details["system"] == "GitLab 17.11.2-ee" and details["param_source"] == "graphql"
    assert [param["name"] for param in details["params"]][:2] == ["PROFILE", "environment"] and details["inputs_api"]
    assert "параметры: PROFILE" in report.message


# ---- the form, the watcher, the report window -------------------------------------------------------------------


@pytest.fixture
def planner(monkeypatch):
    from loadlens_app.blueprints import dashboard

    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    monkeypatch.setitem(core.CONFIG, "load_generators", {"gitlab": GITLAB, "jenkins": JENKINS})
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setitem(core.CONFIG, "limits", {})
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    monkeypatch.setattr(service, "_installed_lt_packs", lambda area, service_id: [])
    secret = CiParam("API_KEY", "variable", "password", secret=True)
    params = CiParams(params=_params().params + [secret], source="graphql", inputs_api=True, ref="main",
                      auto=["loadlens_run_tag"], auto_inputs=["loadlens_run_tag"])
    asked = []

    def cached(generator, ref=""):
        asked.append((generator.id, ref))
        if generator.id == "jenkins":
            return CiParams(params=[CiParam("VUS", "parameter", default="10")], source="jenkins", parameterized=True)
        return params

    monkeypatch.setattr(ci_params, "cached_params", cached)
    return types.SimpleNamespace(store=memory, asked=asked)


def _form(*instances):
    return {"service": "checkout", "run_name": "ci-run", "instances": list(instances)}


def test_plan_of_a_ci_instance(planner):
    planned = service.plan_run(_form(
        {"name": "rest", "generator": "gitlab", "ci_ref": "main", "ci_params": {"build_id": "7", "users": 30, "API_KEY": "s3cret-zz"},
         "ci_extra": "EXTRA=1", "tool": "k6", "script": "ignored.js", "env": "IGNORED=1"},
        {"name": "jmeter", "generator": "jenkins", "ci_params": {"VUS": "50"}},
    ), "eng")
    gitlab, jenkins = planned.run.instances
    assert gitlab.ci_kind == "gitlab" and gitlab.tool == "" and gitlab.env == {} and gitlab.ci_ref == "main"
    assert gitlab.ci_params == {"build_id": "7", "users": "30", "API_KEY": "s3cret-zz"} and gitlab.ci_extra == {"EXTRA": "1"}
    assert gitlab.ci_secret == ["API_KEY"] and "API_KEY=***" in gitlab.command and "s3cret-zz" not in gitlab.command
    assert gitlab.command.startswith("GitLab perf/load @ main:") and jenkins.command == "Jenkins perf/load-test: VUS=50"
    assert planned.commands[0]["ci"] == {"ref": "main", "kind": "gitlab"} and planned.commands[0]["command"] == gitlab.command
    assert any("пайплайна «GitLab»" in warning for warning in planned.warnings)
    view = service.instance_view(planned.run, gitlab, service.load_catalog())
    assert view["ci"]["params"]["API_KEY"] == "***" and view["tool_title"] == "Пайплайн GitLab"


@pytest.mark.parametrize("instance, fragment", [
    ({"generator": "gitlab", "ci_params": {"PROFILE": "huge", "build_id": "1"}}, "нет среди вариантов"),
    ({"generator": "gitlab", "ci_params": {}}, "build_id обязателен"),
    ({"generator": "gitlab", "ci_ref": "a b", "ci_params": {"build_id": "1"}}, "ветка"),
    ({"generator": "jenkins", "ci_extra": "X=1"}, "только параметры"),
    ({"generator": "gitlab", "ci_params": ["x"]}, "объект"),
])
def test_bad_ci_instances(planner, instance, fragment):
    with pytest.raises(service.LaunchError, match=fragment) as error:
        service.plan_run(_form(instance), "eng")
    assert str(error.value).startswith("Инстанс 1:")


def test_unreachable_ci_is_a_gateway_error(planner, monkeypatch):
    def down(generator, ref=""):
        raise GeneratorError("GitLab: не удалось подключиться к gitlab.example")

    monkeypatch.setattr(ci_params, "cached_params", down)
    with pytest.raises(service.LaunchError) as error:
        service.plan_run(_form({"generator": "gitlab"}), "eng")
    assert error.value.status == 502 and "не читаются" in str(error.value)


def test_presets_keep_ci_values_but_not_passwords():
    spec = presets.clean_spec({"instances": [{"generator": "gitlab", "ci_ref": "main", "ci_params": {"VUS": "10", "API_KEY": "s3cret-zz", "OLD": "***", "DEBUG": True},
                                              "ci_secret": ["API_KEY"], "ci_extra": "EXTRA=1"}]})
    item = spec["instances"][0]
    assert item["ci_params"] == {"VUS": "10", "API_KEY": "", "OLD": "", "DEBUG": True} and item["ci_ref"] == "main" and item["ci_extra"] == "EXTRA=1"


@pytest.fixture
def world(monkeypatch):
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    catalog = service.Catalog(tools=effective_tools({}), generators=configured_generators({"load_generators": {"gitlab": GITLAB}}))
    params = CiParams(params=[CiParam("API_KEY", "variable", "password", secret=True)], source="graphql", ref="main")
    monkeypatch.setattr(ci_params, "cached_params", lambda generator, ref="": params)
    launched = []

    class Fake:
        enforces_deadline, supports_signals, supports_control = True, False, False

        def __init__(self):
            self.state = Probe(state="starting")

        def launch(self, plan):
            launched.append(plan)
            return types.SimpleNamespace(ref="gitlab:r1:0:77", already_running=False, url="https://gitlab.example/p/-/pipelines/77", message="")

        def probe(self, ref):
            return self.state

        def signal(self, ref, sig, grace_seconds=30):
            launched.append(("signal", ref))

        def kill(self, ref):
            launched.append(("kill", ref))

        def logs(self, ref, max_bytes=0):
            return "job log"

        def cleanup(self, ref):
            pass

    fake = Fake()
    clock = types.SimpleNamespace(now=1_760_000_000.0)
    started_reports = []
    w = watcher.Watcher(store_getter=lambda: memory, catalog_loader=lambda: catalog, executor_getter=lambda gen: fake,
                        report_starter=lambda *a, **k: started_reports.append((a, k)) or types.SimpleNamespace(job_id="job"),
                        clock=lambda: clock.now)
    monkeypatch.setattr(watcher, "report_filter", lambda run, tools, start, end: ([], ""))
    from loadlens_app.blueprints import dashboard

    monkeypatch.setattr(dashboard, "_report_step", lambda area, service_id: "30s")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    from loadlens_app import jobs

    monkeypatch.setattr(jobs.job_store, "running_counts", lambda owner: (0, 0))
    return types.SimpleNamespace(store=memory, fake=fake, launched=launched, clock=clock, watcher=w, reports=started_reports)


def test_ci_instance_runs_through_to_a_report_with_a_cut_window(world, monkeypatch):
    run = RunRecord(run_id="r1", run_name="n", run_tag="n", service="s", project_area="a",
                    options={"auto_report": True, "use_llm": False, "max_seconds": 3600},
                    instances=[InstanceRecord(index=0, name="ci", tag="ci", generator="gitlab", ci_kind="gitlab", ci_ref="main",
                                              ci_params={"API_KEY": "s3cret-zz"}, ci_secret=["API_KEY"])],
                    created_at=datetime.fromtimestamp(world.clock.now, timezone.utc))
    world.store.create(run)
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert world.launched[0].ci_variables == {"API_KEY": "s3cret-zz"} and inst.status == store.INST_STARTING
    # The password is needed for the start only.
    assert inst.ci_params == {"API_KEY": "***"} and inst.external_url.endswith("/77") and inst.message == "Пайплайн запущен"
    t0 = world.clock.now
    world.fake.state = Probe(state="running", started_at=t0)
    world.watcher.tick()
    world.fake.state = Probe(state="exited", started_at=t0, finished_at=t0 + 900, exit_code=0)
    world.clock.now = t0 + 900
    world.watcher.tick()
    inst = world.store.get("r1").instances[0]
    assert inst.status == store.INST_FINISHED and inst.log_tail == "job log"
    seen = {}

    def bounds(run, start, end, packs):
        seen["window"] = (start, end)
        return start + 120, end - 60

    monkeypatch.setattr(watcher, "rps_bounds", bounds)
    world.clock.now += 400
    world.watcher.tick()
    (args, kwargs), = world.reports
    assert seen["window"] == (t0, t0 + 900) and (args[0], args[1]) == (int((t0 + 120) * 1000), int((t0 + 840) * 1000))
    assert "окно отчёта сужено по RPS" in world.store.get("r1").message


def test_stop_of_a_ci_instance_speaks_of_the_pipeline(world):
    run = RunRecord(run_id="r1", run_name="n", run_tag="n", service="s", project_area="a",
                    options={"auto_report": False, "max_seconds": 3600},
                    instances=[InstanceRecord(index=0, name="ci", tag="ci", generator="gitlab", ci_kind="gitlab", ci_ref="main")],
                    created_at=datetime.fromtimestamp(world.clock.now, timezone.utc))
    world.store.create(run)
    world.watcher.tick()
    world.fake.state = Probe(state="running", started_at=world.clock.now)
    world.watcher.tick()
    world.store.request_stop("r1")
    world.watcher.tick()
    assert world.store.get("r1").instances[0].message.startswith("Пайплайну отправлена отмена")
    world.clock.now += 61
    world.watcher.tick()
    assert ("kill", "gitlab:r1:0:77") in world.launched
    assert world.store.get("r1").instances[0].message == "Не завершился сам: остановка отправлена ещё раз"


def test_trim_bounds_and_the_filter_of_ci_runs(monkeypatch):
    points = [(100.0, 0.0), (105.0, 3.0), (110.0, 0.0), (115.0, 4.0), (120.0, 0.0)]
    assert live.trim_bounds(points, 90.0, 200.0, 5) == (100.0, 120.0)
    assert live.trim_bounds([(100.0, 2.0)], 90.0, 200.0, 5) is None and live.trim_bounds([], 90.0, 200.0, 5) is None
    launched = datetime(2026, 10, 15, tzinfo=timezone.utc)
    ci = InstanceRecord(index=0, name="ci", tag="ci", generator="gitlab", ci_kind="gitlab", launched_at=launched)
    k6 = InstanceRecord(index=1, name="k6", tag="k6", generator="gen", tool="k6", launched_at=launched)
    tools = effective_tools({})
    monkeypatch.setattr(live, "installed_filter_packs", lambda cfg: ["lt-k6-prometheus-rw", "lt-jmeter-influxql"])
    # A pipeline that does not tag its metrics turns the filter off: a filter would hide them.
    assert live.run_filter_packs(RunRecord(run_id="r", run_name="n", run_tag="n", service="s", instances=[ci, k6]), tools, {}) == []
    ci.ci_tagged = True
    packs = live.run_filter_packs(RunRecord(run_id="r", run_name="n", run_tag="n", service="s", instances=[ci, k6]), tools, {})
    assert set(packs) == set(tools["k6"].run_filter_packs) | {"lt-k6-prometheus-rw", "lt-jmeter-influxql"}


def test_installed_filter_packs_reads_the_service_queries():
    cfg = {"queries": {"lt_framework": {"packs": [{"id": "lt-k6-prometheus-rw"}, {"id": "lt-generators-node-exporter"}, {"id": "nope"}]}}}
    assert live.installed_filter_packs(cfg) == ["lt-k6-prometheus-rw"]


# ---- endpoints ---------------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setitem(core.CONFIG, "load_generators", {"gitlab": GITLAB, "ssh": {"title": "M", "type": "manual"}})
    monkeypatch.setattr(store, "_store", MemoryLoadTestStore())
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_ci_params_and_refs_endpoints(client, monkeypatch):
    monkeypatch.setattr(ci_params, "cached_params", lambda generator, ref="": CiParams(params=[CiParam("VUS", "variable", default="10")],
                                                                                      source="graphql", ref=ref or "main"))
    monkeypatch.setattr(ci_params, "read_refs", lambda generator, search="": (["main", "release"], "main"))
    body = client.get("/load_tests/ci_params?generator=gitlab&ref=release").get_json()
    assert body["ref"] == "release" and body["params"][0]["name"] == "VUS" and body["source"] == "graphql"
    assert client.get("/load_tests/ci_refs?generator=gitlab").get_json() == {"refs": ["main", "release"], "default": "main"}
    assert client.get("/load_tests/ci_params?generator=ssh").status_code == 404
    assert client.get("/load_tests/ci_params?generator=gitlab&ref=a%20b").status_code == 400

    def down(generator, ref=""):
        raise GeneratorError("GitLab отказал в доступе (HTTP 401)")

    monkeypatch.setattr(ci_params, "cached_params", down)
    failed = client.get("/load_tests/ci_params?generator=gitlab")
    assert failed.status_code == 502 and "HTTP 401" in failed.get_json()["error"]


def test_cached_params_reads_once_a_minute(monkeypatch):
    ci_params.clear_cache()
    calls = []
    monkeypatch.setattr(ci_params, "read_params", lambda generator, ref="": calls.append(ref) or CiParams(ref=ref))
    gen = _gen()
    assert ci_params.cached_params(gen, "main").ref == "main" and ci_params.cached_params(gen, "main").ref == "main"
    ci_params.cached_params(gen, "release")
    assert calls == ["main", "release"]
    ci_params.clear_cache()
