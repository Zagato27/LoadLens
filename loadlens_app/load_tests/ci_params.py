# SPDX-License-Identifier: Apache-2.0
"""Launch parameters of the team's own CI pipelines: read from GitLab or Jenkins, checked before a run.

A CI generator runs the load test pipeline as it is: the pipeline holds the tool, the scripts and their
settings, and LoadLens only picks the branch and the values of the parameters the pipeline declares.

GitLab has no single API for that, and the ways changed between versions, so the reader tries them in order
and keeps the first that answers:

1. GraphQL ``ciConfigVariables`` — the source of the "Run pipeline" form: top-level variables with a
   ``description`` (prefilled variables), with ``valueOptions`` on newer versions, by ``ref`` or by ``sha`` on
   older ones. GitLab computes it in the background and answers ``null`` at first, so it is asked again.
2. The CI lint API with the merged YAML (includes resolved).
3. The CI configuration file itself (no includes).

``spec:inputs`` of the configuration header always come from the file (the merged YAML drops the header).
Whether the API accepts ``inputs`` when creating a pipeline depends on the version (``inputs_mode`` of the
generator overrides the guess).

Jenkins lists the parameter definitions of a job (string, text, boolean, choice, password, ...). Active
Choices parameters compute their options with a Groovy script on the build page: a script that only returns
a literal list is read from ``config.xml``; otherwise the options come from the generator settings or the
value is typed in.

Names starting with ``LOADLENS_`` (``loadlens_`` for inputs) are filled by LoadLens itself.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional
from urllib.parse import quote

import yaml

from .generators import CiSettings, Generator
from .messages import as_msg, msg, text_of
from .tools import ENV_NAME_RE, ToolInputError

AUTO_PREFIX = "LOADLENS_"
# Since this version (GitLab 17.11) the API that creates a pipeline takes spec:inputs; "inputs_mode" overrides.
GITLAB_INPUTS_SINCE = (17, 11)
GRAPHQL_ATTEMPTS = 4
GRAPHQL_RETRY_SEC = 1.0
CACHE_SEC = 60.0
CACHE_MAX = 64
MAX_PARAMS = 100
MAX_VALUE_CHARS = 4000
MAX_CONFIG_BYTES = 1_000_000
ACTIVE_CHOICES = ("ChoiceParameter", "CascadeChoiceParameter", "DynamicReferenceParameter")
_MULTI_CHOICE = ("PT_CHECKBOX", "PT_MULTI_SELECT")
TYPES = ("string", "text", "choice", "boolean", "number", "array", "password")

Sleep = Callable[[float], None]


class CiParamError(ToolInputError):
    """A value does not fit the parameters of the pipeline; the message is for the user."""


@dataclass
class CiParam:
    name: str
    # variable (GitLab prefilled variable), input (GitLab spec:inputs) or parameter (Jenkins)
    kind: str
    type: str = "string"
    description: str = ""
    default: Any = ""
    options: tuple[str, ...] = ()
    required: bool = False
    secret: bool = False
    multiple: bool = False
    regex: str = ""
    # A dictionary message (kept in the cache with its key): rendered in the reader's language by to_public().
    note: str = ""

    def to_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "type": self.type,
            "description": self.description,
            # A password default is the CI's secret: never sent to the browser.
            "default": "" if self.secret else default_text(self.default),
            "options": list(self.options),
            "required": self.required,
            "secret": self.secret,
            "multiple": self.multiple,
            "regex": self.regex,
            "note": text_of(self.note),
        }


@dataclass
class CiParams:
    """What the form shows for a CI instance: the parameters, the branch they were read for, and how."""

    params: list[CiParam] = field(default_factory=list)
    ref: str = ""
    refs: list[str] = field(default_factory=list)
    # graphql | lint | file | jenkins | none (nothing can be read: free-form values only)
    source: str = "none"
    # Dictionary messages (the parameters are cached for a minute): rendered in the reader's language on output.
    notes: list[str] = field(default_factory=list)
    # GitLab: spec:inputs go to the API (otherwise they are shown with their defaults only).
    inputs_api: bool = False
    # Jenkins: the job declares parameters (buildWithParameters) or none (build).
    parameterized: bool = False
    # Names LoadLens fills itself: declared LOADLENS_* variables or parameters, and loadlens_* inputs.
    auto: list[str] = field(default_factory=list)
    auto_inputs: list[str] = field(default_factory=list)
    version: str = ""
    # Jenkins multibranch: the path of the branch job.
    job_path: str = ""

    def get(self, name: str) -> Optional[CiParam]:
        return next((param for param in self.params if param.name == name), None)

    def to_public(self) -> dict[str, Any]:
        return {
            "params": [param.to_public() for param in self.params],
            "ref": self.ref,
            "refs": list(self.refs),
            "source": self.source,
            "notes": [text_of(note) for note in self.notes],
            "inputs_api": self.inputs_api,
            "auto": list(self.auto),
            "version": self.version,
        }


def is_auto(name: str) -> bool:
    return str(name).upper().startswith(AUTO_PREFIX)


def default_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "" if value is None else str(value)


def parse_version(text: str) -> tuple[int, ...]:
    """``16.11.3-ee`` -> (16, 11, 3)."""
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(text or ""))
    return tuple(int(part) for part in match.groups() if part is not None) if match else ()


def inputs_supported(cfg: CiSettings, version: tuple[int, ...]) -> bool:
    if cfg.inputs_mode == "on":
        return True
    if cfg.inputs_mode == "off":
        return False
    return bool(version) and version >= GITLAB_INPUTS_SINCE


# ---- GitLab ------------------------------------------------------------------------------------------


class _GitlabLoader(yaml.SafeLoader):
    """Safe YAML that keeps GitLab tags (!reference, !include...) as plain values instead of failing."""


def _any_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_GitlabLoader.add_multi_constructor("!", _any_tag)


def gitlab_documents(text: str) -> list[Any]:
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise CiParamError("lt.ci.config_too_big")
    try:
        return [doc for doc in yaml.load_all(text, Loader=_GitlabLoader) if doc is not None]
    except yaml.YAMLError as exc:
        raise CiParamError("lt.ci.yaml_error", error=str(exc)[:200]) from exc


def _variable_param(name: str, value: Any, description: str, options: Any) -> CiParam:
    opts = tuple(str(option) for option in (options or []) if option is not None)
    return CiParam(name=str(name), kind="variable", type="choice" if opts else "string", description=str(description or ""),
                   default=default_text(value), options=opts)


def yaml_variables(documents: list[Any]) -> list[CiParam]:
    """Top-level variables with a description: the ones GitLab shows in its "Run pipeline" form."""
    out: list[CiParam] = []
    for doc in documents:
        raw = doc.get("variables") if isinstance(doc, Mapping) else None
        if not isinstance(raw, Mapping):
            continue
        for name, value in raw.items():
            if isinstance(value, Mapping) and value.get("description"):
                out.append(_variable_param(name, value.get("value", ""), value.get("description"), value.get("options")))
    return out


def yaml_inputs(documents: list[Any]) -> list[CiParam]:
    """``spec:inputs`` of the configuration header."""
    header = next((doc for doc in documents if isinstance(doc, Mapping) and isinstance(doc.get("spec"), Mapping)), None)
    inputs = header["spec"].get("inputs") if header else None
    if not isinstance(inputs, Mapping):
        return []
    out: list[CiParam] = []
    for name, raw in inputs.items():
        item = raw if isinstance(raw, Mapping) else {}
        kind = str(item.get("type") or "string")
        options = tuple(default_text(option) for option in (item.get("options") or []))
        ptype = "choice" if options else (kind if kind in ("string", "number", "boolean", "array") else "string")
        out.append(CiParam(
            name=str(name), kind="input", type=ptype, description=str(item.get("description") or ""),
            default=item.get("default", ""), options=options, required="default" not in item,
            regex=str(item.get("regex") or ""),
        ))
    return out


def _graphql_variables(api, cfg: CiSettings, full_path: str, ref: str, sha: Callable[[], str], sleep: Sleep) -> Optional[list[CiParam]]:
    """Prefilled variables from GraphQL, or ``None`` when this GitLab cannot tell (old schema, no access)."""
    from .executors.http import ApiError

    headers = {"Authorization": f"Bearer {cfg.api_token}"}
    for argument, with_options in (("ref", True), ("ref", False), ("sha", True), ("sha", False)):
        value = ref if argument == "ref" else sha()
        if not value:
            continue
        fields = "key value description" + (" valueOptions" if with_options else "")
        query = f"query($path: ID!, $value: String!) {{ project(fullPath: $path) {{ ciConfigVariables({argument}: $value) {{ {fields} }} }} }}"
        for _attempt in range(GRAPHQL_ATTEMPTS):
            try:
                answer = api.json("POST", "/api/graphql", json={"query": query, "variables": {"path": full_path, "value": value}},
                                  headers=headers)
            except ApiError:
                return None
            if answer.get("errors"):
                break  # this GitLab has another schema: the next variant
            project = (answer.get("data") or {}).get("project")
            if project is None:
                return None
            items = project.get("ciConfigVariables")
            if items is None:
                sleep(GRAPHQL_RETRY_SEC)  # computed in the background: asked again
                continue
            return [_variable_param(item.get("key"), item.get("value"), item.get("description"), item.get("valueOptions"))
                    for item in items if isinstance(item, Mapping) and item.get("key")]
        else:
            return None
    return None


def _lint_variables(api, project_path: str, ref: str) -> Optional[list[CiParam]]:
    from .executors.base import GeneratorError

    try:
        answer = api.json("GET", f"{project_path}/ci/lint", params={"content_ref": ref, "include_merged_yaml": "true"})
    except GeneratorError:
        return None
    merged = answer.get("merged_yaml") if isinstance(answer, Mapping) else None
    if not merged:
        return None
    try:
        return yaml_variables(gitlab_documents(str(merged)))
    except CiParamError:
        return None


def _config_file(api, project_path: str, project: Mapping[str, Any], ref: str) -> Optional[str]:
    from .executors.base import GeneratorError

    path = str(project.get("ci_config_path") or "") or ".gitlab-ci.yml"
    if "@" in path or "://" in path:
        return None  # the configuration lives in another project
    try:
        response = api.request("GET", f"{project_path}/repository/files/{quote(path, safe='')}/raw", params={"ref": ref})
    except GeneratorError:
        return None
    return response.text


def read_gitlab(api, cfg: CiSettings, ref: str = "", sleep: Sleep = time.sleep) -> CiParams:
    from .executors.base import GeneratorError

    project_path = f"/api/v4/projects/{quote(cfg.project, safe='')}"
    if not cfg.api_token:
        return CiParams(ref=ref or cfg.ref or "main", source="none", notes=[msg("lt.ci.note.no_api_token")])
    version_text = str((api.json("GET", "/api/v4/version") or {}).get("version") or "")
    version = parse_version(version_text)
    project = api.json("GET", project_path)
    full_path = str(project.get("path_with_namespace") or cfg.project)
    ref = ref or cfg.ref or str(project.get("default_branch") or "main")
    sha_cache: dict[str, str] = {}

    def sha() -> str:
        if "sha" not in sha_cache:
            try:
                branch = api.json("GET", f"{project_path}/repository/branches/{quote(ref, safe='')}")
                sha_cache["sha"] = str(((branch or {}).get("commit") or {}).get("id") or "")
            except GeneratorError:
                sha_cache["sha"] = ""
        return sha_cache["sha"]

    notes: list[str] = []
    variables = _graphql_variables(api, cfg, full_path, ref, sha, sleep)
    source = "graphql"
    if variables is None:
        variables, source = _lint_variables(api, project_path, ref), "lint"
    text = _config_file(api, project_path, project, ref)
    documents: list[Any] = []
    if text is not None:
        try:
            documents = gitlab_documents(text)
        except CiParamError as exc:
            notes.append(as_msg(exc))
    if variables is None:
        variables, source = yaml_variables(documents), "file"
        if text is None:
            source = "none"
            notes.append(msg("lt.ci.note.config_unreadable"))
    inputs = yaml_inputs(documents)
    supported = inputs_supported(cfg, version)
    if inputs and not supported:
        notes.append(msg("lt.ci.note.no_inputs", version=version_text or "?"))
    params, auto = _split_auto(variables + inputs)
    params = _apply_options(params, cfg)
    return CiParams(params=params[:MAX_PARAMS], ref=ref, source=source, notes=notes, inputs_api=supported,
                    auto=auto, auto_inputs=[param.name for param in inputs if is_auto(param.name)],
                    version=version_text, parameterized=True)


def gitlab_refs(api, cfg: CiSettings, search: str = "") -> tuple[list[str], str]:
    project_path = f"/api/v4/projects/{quote(cfg.project, safe='')}"
    if not cfg.api_token:
        return [], cfg.ref or "main"
    project = api.json("GET", project_path)
    params: dict[str, Any] = {"per_page": 100}
    if search:
        params["search"] = search
    branches = api.json("GET", f"{project_path}/repository/branches", params=params) or []
    names = [str(item.get("name")) for item in branches if isinstance(item, Mapping) and item.get("name")]
    return names, cfg.ref or str(project.get("default_branch") or "main")


# ---- Jenkins -----------------------------------------------------------------------------------------


def _jenkins_job_path(job: str) -> str:
    return "".join(f"/job/{quote(part, safe='')}" for part in job.split("/") if part)


def _is_multibranch(info: Mapping[str, Any]) -> bool:
    cls = str(info.get("_class") or "")
    return "MultiBranch" in cls or "OrganizationFolder" in cls


def literal_choices(script: str) -> Optional[tuple[list[str], str]]:
    """Options of an Active Choices script that only returns a literal list: (options, default).

    ``return ['a', 'b:selected', 'c:disabled']`` -> (['a', 'b'], 'b'); anything else -> None.
    """
    match = re.fullmatch(r"\s*(?:return\s+)?\[(.*)\]\s*;?\s*", script or "", re.S)
    if not match:
        return None
    body = match.group(1)
    items = re.findall(r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\"""", body)
    rest = re.sub(r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\"""", "", body)
    if rest.replace(",", "").strip():
        return None  # computed values, not a literal list
    options: list[str] = []
    default = ""
    for single, double in items:
        text = single or double
        if text.endswith(":disabled"):
            continue
        if text.endswith(":selected"):
            text = text[: -len(":selected")]
            default = default or text
        options.append(text)
    return (options, default) if options else None


def active_choices(config_xml: str) -> dict[str, dict[str, Any]]:
    """Active Choices parameters of a job's config.xml: name -> {options, default, multiple}."""
    out: dict[str, dict[str, Any]] = {}
    if not config_xml or len(config_xml.encode("utf-8")) > MAX_CONFIG_BYTES:
        return out
    try:
        root = ElementTree.fromstring(config_xml)
    except ElementTree.ParseError:
        return out
    for element in root.iter():
        if not str(element.tag).startswith("org.biouno.unochoice."):
            continue
        name = (element.findtext("name") or "").strip()
        script = element.findtext("script/secureScript/script")
        if script is None:
            script = element.findtext("script/script")
        parsed = literal_choices(script or "")
        if name and parsed:
            out[name] = {"options": parsed[0], "default": parsed[1],
                         "multiple": (element.findtext("choiceType") or "") in _MULTI_CHOICE}
    return out


def _jenkins_param(raw: Mapping[str, Any], scripted: Mapping[str, dict[str, Any]]) -> Optional[CiParam]:
    name = str(raw.get("name") or "")
    if not name:
        return None
    cls = str(raw.get("_class") or "")
    kind = str(raw.get("type") or cls.rsplit(".", 1)[-1])
    default = (raw.get("defaultParameterValue") or {}).get("value") if isinstance(raw.get("defaultParameterValue"), Mapping) else None
    description = re.sub(r"<[^>]+>", "", str(raw.get("description") or "")).strip()
    param = CiParam(name=name, kind="parameter", description=description, default=default_text(default))
    if kind == "BooleanParameterDefinition":
        param.type, param.default = "boolean", bool(default)
    elif kind == "ChoiceParameterDefinition":
        choices = tuple(str(choice) for choice in raw.get("choices") or [])
        param.type, param.options = "choice", choices
        param.default = param.default or (choices[0] if choices else "")
    elif kind == "PasswordParameterDefinition":
        param.type, param.secret, param.default = "password", True, ""
    elif kind == "TextParameterDefinition":
        param.type = "text"
    elif kind == "StringParameterDefinition":
        param.type = "string"
    elif kind in ACTIVE_CHOICES or "unochoice" in cls:
        found = scripted.get(name)
        if found:
            param.type, param.options, param.multiple = "choice", tuple(found["options"]), bool(found["multiple"])
            param.default = found["default"] or (found["options"][0] if found["options"] and not found["multiple"] else "")
            param.note = msg("lt.ci.note.active_choices_literal")
        else:
            param.type = "string"
            param.note = msg("lt.ci.note.active_choices_cascade" if kind == "CascadeChoiceParameter"
                             else "lt.ci.note.active_choices_script")
    elif kind == "FileParameterDefinition":
        return None
    else:
        param.note = msg("lt.ci.note.type_as_string", kind=kind)
    return param


def read_jenkins(api, cfg: CiSettings, ref: str = "") -> CiParams:
    from .executors.base import GeneratorError

    base = _jenkins_job_path(cfg.job)
    info = api.json("GET", f"{base}/api/json", params={"tree": "_class,jobs[name]"})
    job, refs, notes = base, [], []
    if _is_multibranch(info):
        refs = [str(item.get("name")) for item in info.get("jobs") or [] if isinstance(item, Mapping) and item.get("name")]
        wanted = ref or cfg.ref
        ref = wanted if wanted in refs else ("main" if "main" in refs else (refs[0] if refs else ""))
        if wanted and wanted != ref:
            notes.append(msg("lt.ci.note.branch_missing", wanted=wanted, ref=ref))
        if not ref:
            return CiParams(source="jenkins", notes=[msg("lt.ci.note.no_branches")], job_path=base)
        job = f"{base}/job/{quote(ref, safe='')}"
    else:
        ref = ""
    tree = "_class,property[_class,parameterDefinitions[_class,name,type,description,defaultParameterValue[value],choices]]"
    data = api.json("GET", f"{job}/api/json", params={"tree": tree})
    raw = [item for prop in data.get("property") or [] if isinstance(prop, Mapping)
           for item in prop.get("parameterDefinitions") or [] if isinstance(item, Mapping)]
    scripted: dict[str, dict[str, Any]] = {}
    if any(str(item.get("type") or "") in ACTIVE_CHOICES or "unochoice" in str(item.get("_class") or "") for item in raw):
        try:
            scripted = active_choices(api.request("GET", f"{job}/config.xml").text)
        except GeneratorError:
            notes.append(msg("lt.ci.note.config_xml"))
    params: list[CiParam] = []
    for item in raw:
        param = _jenkins_param(item, scripted)
        if param is None:
            if str(item.get("type") or "") == "FileParameterDefinition":
                notes.append(msg("lt.ci.note.file_param", name=item.get("name")))
            continue
        params.append(param)
    if not raw and "WorkflowJob" in str(data.get("_class") or ""):
        notes.append(msg("lt.ci.note.no_params"))
    params, auto = _split_auto(params)
    params = _apply_options(params, cfg)
    return CiParams(params=params[:MAX_PARAMS], ref=ref, refs=refs, source="jenkins", notes=notes,
                    parameterized=bool(raw), auto=auto, job_path=job)


def jenkins_refs(api, cfg: CiSettings) -> tuple[list[str], str]:
    info = api.json("GET", f"{_jenkins_job_path(cfg.job)}/api/json", params={"tree": "_class,jobs[name]"})
    if not _is_multibranch(info):
        return [], ""
    refs = [str(item.get("name")) for item in info.get("jobs") or [] if isinstance(item, Mapping) and item.get("name")]
    return refs, cfg.ref if cfg.ref in refs else ("main" if "main" in refs else (refs[0] if refs else ""))


# ---- shared ------------------------------------------------------------------------------------------


def _split_auto(params: list[CiParam]) -> tuple[list[CiParam], list[str]]:
    shown = [param for param in params if not is_auto(param.name)]
    auto = [param.name for param in params if is_auto(param.name)]
    return shown, auto


def _apply_options(params: list[CiParam], cfg: CiSettings) -> list[CiParam]:
    """Options from the generator settings turn a free value into a choice (Active Choices, plain strings)."""
    out = []
    for param in params:
        options = cfg.param_options.get(param.name)
        if options and param.type not in ("boolean", "password"):
            default = default_text(param.default)
            param = replace(param, type="choice", options=tuple(options), default=default if default in options else options[0],
                            note=msg("lt.ci.note.options_from_settings"))
        out.append(param)
    return out


def read_params(generator: Generator, ref: str = "", api=None, sleep: Sleep = time.sleep) -> CiParams:
    assert generator.ci is not None
    if api is None:
        from .executors.ci import ci_api

        api = ci_api(generator)
    if generator.ci.kind == "gitlab":
        return read_gitlab(api, generator.ci, ref, sleep)
    return read_jenkins(api, generator.ci, ref)


def read_refs(generator: Generator, search: str = "", api=None) -> tuple[list[str], str]:
    assert generator.ci is not None
    if api is None:
        from .executors.ci import ci_api

        api = ci_api(generator)
    if generator.ci.kind == "gitlab":
        return gitlab_refs(api, generator.ci, search)
    return jenkins_refs(api, generator.ci)


_cache: dict[tuple, tuple[float, CiParams]] = {}
_cache_lock = threading.Lock()


def _identity(generator: Generator) -> tuple:
    cfg = generator.ci
    assert cfg is not None
    options = tuple(sorted((name, tuple(values)) for name, values in cfg.param_options.items()))
    return (generator.id, cfg.kind, cfg.url, cfg.project, cfg.job, cfg.ref, cfg.api_token, cfg.username, cfg.inputs_mode, options)


def cached_params(generator: Generator, ref: str = "") -> CiParams:
    """:func:`read_params` cached for a minute: the form asks on every generator and branch pick."""
    key = _identity(generator) + (ref,)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < CACHE_SEC:
            return hit[1]
    result = read_params(generator, ref)
    with _cache_lock:
        if len(_cache) >= CACHE_MAX:
            _cache.clear()
        _cache[key] = (now, result)
    return result


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ---- values ------------------------------------------------------------------------------------------


def _bool_text(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return "true"
    if text in ("false", "0", "no", "off", ""):
        return "false"
    return None


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return "" if value is None else str(value)


def check_value(param: CiParam, raw: Any) -> str:
    """The form value as text, checked against the parameter; raises :class:`CiParamError`."""
    text = _value_text(raw)
    if len(text) > MAX_VALUE_CHARS or "\x00" in text:
        raise CiParamError("lt.ci.value.too_long", name=param.name, max=MAX_VALUE_CHARS)
    if param.type == "boolean":
        value = _bool_text(raw)
        if value is None:
            raise CiParamError("lt.ci.value.boolean", name=param.name)
        return value
    if param.type in ("string", "choice", "number", "array", "password") and "\n" in text:
        raise CiParamError("lt.ci.value.one_line", name=param.name)
    if param.type == "number" and text != "":
        try:
            number = float(text)
        except ValueError as exc:
            raise CiParamError("lt.ci.value.number", name=param.name) from exc
        if not math.isfinite(number):
            raise CiParamError("lt.ci.value.number", name=param.name)
    if param.type == "array" and text.startswith("["):
        try:
            if not isinstance(json.loads(text), list):
                raise ValueError
        except ValueError as exc:
            raise CiParamError("lt.ci.value.array", name=param.name) from exc
    if param.options and text != "":
        chosen = [item.strip() for item in text.split(",")] if param.multiple else [text]
        unknown = [item for item in chosen if item not in param.options]
        if unknown:
            raise CiParamError("lt.ci.value.not_an_option", name=param.name, value=unknown[0][:60],
                               options=", ".join(param.options[:10]))
    if param.regex and text != "":
        try:
            if not re.fullmatch(param.regex.strip("/"), text):
                raise CiParamError("lt.ci.value.pattern", name=param.name, regex=param.regex)
        except re.error:
            pass
    return text


def check_values(params: CiParams, values: Mapping[str, Any], extra: Mapping[str, str], kind: str) -> tuple[dict[str, str], dict[str, str]]:
    """(parameter values, extra GitLab variables) of the form, checked; raises :class:`CiParamError`."""
    if not isinstance(values, Mapping):
        raise CiParamError("lt.ci.values_object")
    if len(values) > MAX_PARAMS:
        raise CiParamError("lt.ci.too_many", max=MAX_PARAMS)
    out: dict[str, str] = {}
    free = params.source == "none"
    for name, raw in values.items():
        name = str(name)
        if is_auto(name):
            raise CiParamError("lt.ci.auto_param", name=name)
        param = params.get(name)
        if param is None:
            if not free:
                raise CiParamError("lt.ci.unknown_param", name=name)
            if not ENV_NAME_RE.fullmatch(name):
                raise CiParamError("lt.ci.bad_variable", name=name[:40])
            param = CiParam(name=name, kind="variable")
        out[name] = check_value(param, raw)
    for param in params.params:
        if param.required and out.get(param.name, "") == "" and default_text(param.default) == "":
            raise CiParamError("lt.ci.required", name=param.name)
    clean_extra: dict[str, str] = {}
    for name, value in (extra or {}).items():
        name = str(name)
        if kind != "gitlab":
            raise CiParamError("lt.ci.jenkins_declared_only")
        if is_auto(name):
            raise CiParamError("lt.ci.auto_param", name=name)
        if params.get(name) is not None:
            raise CiParamError("lt.ci.extra_is_param", name=name)
        clean_extra[name] = str(value)
    return out, clean_extra


def _typed_input(param: CiParam, text: str) -> Any:
    if param.type == "boolean":
        return text == "true"
    if param.type == "number":
        number = float(text)
        return int(number) if number.is_integer() else number
    if param.type == "array":
        if text.startswith("["):
            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]
    return text


def launch_values(params: CiParams, values: Mapping[str, str], extra: Mapping[str, str], kind: str,
                  auto: Mapping[str, str]) -> dict[str, Any]:
    """What goes to the CI: only values that differ from the pipeline's defaults (the rest stays as written).

    GitLab: ``variables`` (prefilled variables that changed, extra variables), ``inputs`` (typed, when the API
    takes them; ``loadlens_*`` inputs the pipeline declares are filled), ``auto`` (LOADLENS_* variables).
    Jenkins: ``parameters`` (changed values plus the declared LOADLENS_* parameters).
    """
    changed: dict[str, str] = {}
    for name, text in values.items():
        param = params.get(name)
        if param is None:
            changed[name] = text
            continue
        if param.secret and text == "":
            continue  # an empty password keeps the CI's own secret
        if text != default_text(param.default) or param.secret:
            changed[name] = text
    upper = {key.upper(): value for key, value in auto.items()}
    if kind == "jenkins":
        parameters = dict(changed)
        parameters.update({name: upper[name.upper()] for name in params.auto if name.upper() in upper})
        return {"parameters": parameters}
    variables: dict[str, str] = {}
    inputs: dict[str, Any] = {}
    for name, text in changed.items():
        param = params.get(name)
        if param is not None and param.kind == "input":
            # An empty input keeps its default; inputs go only where the API takes them.
            if params.inputs_api and text != "":
                inputs[name] = _typed_input(param, text)
        else:
            variables[name] = text
    variables.update(extra)
    if params.inputs_api:
        inputs.update({name: upper[name.upper()] for name in params.auto_inputs if name.upper() in upper})
    return {"variables": variables, "inputs": inputs, "auto": dict(auto)}


def describe(kind: str, target: str, ref: str, values: Mapping[str, str], extra: Mapping[str, str], secret: list[str]) -> str:
    """One line for the run page and the audit log: «GitLab perf/load @ main: VUS=10, TOKEN=***»."""
    shown = [f"{name}={'***' if name in secret else value}" for name, value in values.items()]
    shown += [f"{name}={value}" for name, value in extra.items()]
    head = f"{'GitLab' if kind == 'gitlab' else 'Jenkins'} {target}{f' @ {ref}' if ref else ''}"
    return f"{head}: {', '.join(shown)}" if shown else head


__all__ = [
    "AUTO_PREFIX",
    "CiParam",
    "CiParamError",
    "CiParams",
    "active_choices",
    "cached_params",
    "check_value",
    "check_values",
    "clear_cache",
    "describe",
    "gitlab_documents",
    "inputs_supported",
    "is_auto",
    "launch_values",
    "literal_choices",
    "parse_version",
    "read_gitlab",
    "read_jenkins",
    "read_params",
    "read_refs",
    "yaml_inputs",
    "yaml_variables",
]
