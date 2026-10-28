# SPDX-License-Identifier: Apache-2.0
"""Load generators from ``CONFIG["load_generators"]``: where an instance runs and how LoadLens reaches it.

Types:
  ssh         LoadLens connects over SSH (key or password, pinned host key) and starts the tool there.
  docker      a container per instance through the Docker Engine API (TLS, or a socket proxy).
  kubernetes  a Job per instance through the Kubernetes API (token, or in-cluster service account).
  gitlab      a pipeline per instance through a GitLab trigger token; the job fetches its plan from LoadLens.
  jenkins     a build per instance through Jenkins ``buildWithParameters``; same plan and events as GitLab.
  manual      nothing is started: the engineer runs the tool by hand and presses start and stop in LoadLens,
              which only records the window of the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .messages import LtError, Titles, msg
from .tools import ENV_NAME_RE, ToolTemplate, is_absolute_path, public_url

GENERATOR_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
GENERATOR_TYPES = ("ssh", "docker", "kubernetes", "gitlab", "jenkins", "manual")
# Product names stay as they are; "manual" is a dictionary message in the reader's language.
GENERATOR_TYPE_TITLES = Titles(
    {"manual": "lt.generator_type.manual"},
    fixed={"ssh": "SSH", "docker": "Docker", "kubernetes": "Kubernetes", "gitlab": "GitLab CI", "jenkins": "Jenkins"},
)
CONTAINER_TYPES = ("docker", "kubernetes")
CI_TYPES = ("gitlab", "jenkins")
DEFAULT_SSH_PORT = 22
MAX_PARALLEL_CAP = 100
DNS_LABEL_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")
VOLUME_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
QUANTITY_RE = re.compile(r"^[0-9]+(\.[0-9]+)?(m|k|Ki|M|Mi|G|Gi|T|Ti)?$")
LABEL_KEY_RE = re.compile(r"^([a-z0-9]([-a-z0-9.]*[a-z0-9])?/)?[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?$")
LABEL_VALUE_RE = re.compile(r"^([A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?)?$")


class GeneratorConfigError(LtError, ValueError):
    """A generator in the settings is not usable; the message names it."""


@dataclass(frozen=True)
class SshSettings:
    host: str
    port: int = DEFAULT_SSH_PORT
    username: str = ""
    private_key: str = ""
    passphrase: str = ""
    password: str = ""
    # "ssh-ed25519 AAAA..." of the server, pinned by the connection check in the settings.
    host_key: str = ""
    connect_timeout: int = 10

    def identity(self) -> tuple:
        """What a cached connection depends on."""
        return (self.host, self.port, self.username, self.private_key, self.passphrase, self.password, self.host_key)


@dataclass(frozen=True)
class Mount:
    source: str  # volume name, or an absolute path on the Docker host
    target: str
    read_only: bool = True

    @property
    def is_bind(self) -> bool:
        return self.source.startswith("/")


@dataclass(frozen=True)
class DockerSettings:
    url: str
    ca_cert: str = ""
    client_cert: str = ""
    client_key: str = ""
    verify: bool = True
    network: str = ""
    mounts: tuple[Mount, ...] = ()
    cpus: float = 0.0
    memory_mb: int = 0
    # Containers are removed once their log is saved in the run; off keeps them for `docker logs`.
    remove_finished: bool = True

    def identity(self) -> tuple:
        return (self.url, self.ca_cert, self.client_cert, self.client_key, self.verify)


@dataclass(frozen=True)
class KubernetesSettings:
    api_url: str = ""
    token: str = ""
    ca_cert: str = ""
    verify: bool = True
    in_cluster: bool = False
    namespace: str = "default"
    scripts_config_map: str = ""
    scripts_pvc: str = ""
    node_selector: dict[str, str] = field(default_factory=dict)
    cpu: str = ""
    memory: str = ""
    service_account: str = ""
    image_pull_secret: str = ""
    ttl_seconds: int = 3600

    def identity(self) -> tuple:
        return (self.api_url, self.token, self.ca_cert, self.verify, self.in_cluster)


@dataclass(frozen=True)
class CiSettings:
    """A CI that runs the team's own load test pipeline as it is (see load_tests/ci_params.py).

    GitLab: project id or path, the default branch of runs ("" — the project's default branch), an API token
    (api scope: reads parameters, starts, watches and cancels pipelines) and/or a pipeline trigger token
    (starts only). Jenkins: job path (folder/job; a multibranch job takes the branch as ``ref``), user and
    API token.
    """

    kind: str  # gitlab | jenkins
    url: str
    project: str = ""
    ref: str = ""
    trigger_token: str = ""
    job: str = ""
    username: str = ""
    api_token: str = ""
    verify: bool = True
    # Address of LoadLens for the optional events of the pipeline (started, finished).
    loadlens_url: str = ""
    # Options of parameters the CI cannot list itself (Active Choices scripts): parameter name -> options.
    param_options: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # GitLab spec:inputs through the API: "auto" (by GitLab version), "on" or "off".
    inputs_mode: str = "auto"
    # The pipeline passes LOADLENS_RUN_TAG to the tool, so run metrics can be told apart by the tag.
    tags_metrics: bool = False

    def identity(self) -> tuple:
        """What a cached HTTP client depends on."""
        return (self.kind, self.url, self.username, self.api_token, self.verify)


@dataclass(frozen=True)
class GeneratorTool:
    executable: str = ""
    scripts_dir: str = ""
    image: str = ""


@dataclass(frozen=True)
class Generator:
    id: str
    title: str
    type: str
    ssh: Optional[SshSettings] = None
    docker: Optional[DockerSettings] = None
    kubernetes: Optional[KubernetesSettings] = None
    ci: Optional[CiSettings] = None
    work_dir: str = ""
    scripts_dir: str = ""
    # Tools this generator offers, with optional executable, scripts directory and image overrides. Empty: all tools.
    tools: dict[str, GeneratorTool] = field(default_factory=dict)
    # Instances running on it at once; 0 — no limit.
    max_parallel: int = 0
    env: dict[str, str] = field(default_factory=dict)
    # Last successful connection check of the settings page and what it was made against (check_target()).
    verified_at: str = ""
    verified_target: str = ""

    @property
    def is_manual(self) -> bool:
        return self.type == "manual"

    @property
    def is_container(self) -> bool:
        return self.type in CONTAINER_TYPES

    @property
    def is_ci(self) -> bool:
        return self.type in CI_TYPES

    def image_for(self, tool: ToolTemplate) -> str:
        own = self.tools.get(tool.id)
        return (own.image if own and own.image else "") or tool.image

    def offers(self, tool_id: str, tool: Optional[ToolTemplate] = None) -> bool:
        if self.is_manual:
            return True
        if self.is_ci:
            return False  # the pipeline chooses its tool
        if self.tools and tool_id not in self.tools:
            return False
        if self.is_container and tool is not None:
            return bool(self.image_for(tool))
        return True

    def executable(self, tool: ToolTemplate) -> str:
        own = self.tools.get(tool.id)
        return (own.executable if own and own.executable else "") or tool.executable

    def scripts_dir_for(self, tool_id: str) -> str:
        own = self.tools.get(tool_id)
        return (own.scripts_dir if own and own.scripts_dir else "") or self.scripts_dir

    def address(self) -> str:
        """Where the generator is, for the launch form (no credentials)."""
        if self.ssh:
            return self.ssh.host
        if self.docker:
            return public_url(self.docker.url)
        if self.kubernetes:
            return f"{'in-cluster' if self.kubernetes.in_cluster else public_url(self.kubernetes.api_url)} / {self.kubernetes.namespace}"
        if self.ci:
            return public_url(self.ci.url)
        return ""

    def check_target(self) -> str:
        """What a connection check vouches for: the endpoint and identity LoadLens connects to.

        A check stays valid while this does not change; credentials in URLs are left out.
        """
        if self.ssh:
            return f"ssh://{self.ssh.username}@{self.ssh.host}:{self.ssh.port}"
        if self.docker:
            return f"docker:{public_url(self.docker.url).rstrip('/')}"
        if self.kubernetes:
            where = "in-cluster" if self.kubernetes.in_cluster else public_url(self.kubernetes.api_url).rstrip("/")
            return f"kubernetes:{where}/{self.kubernetes.namespace}"
        if self.ci:
            return f"{self.ci.kind}:{public_url(self.ci.url).rstrip('/')}/{self.ci.project or self.ci.job}"
        return self.type

    @property
    def is_verified(self) -> bool:
        """A manual generator needs no check; the others need one made against their current endpoint."""
        return self.is_manual or (bool(self.verified_at) and self.verified_target == self.check_target())

    def to_public(self, tools: Mapping[str, ToolTemplate]) -> dict[str, Any]:
        """For the launch form: no credentials."""
        offered = [tool_id for tool_id, tool in tools.items() if self.offers(tool_id, tool)] if not self.is_manual else []
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "type_title": GENERATOR_TYPE_TITLES.get(self.type, self.type),
            "host": self.address(),
            "tools": offered,
            "max_parallel": self.max_parallel,
            "lists_scripts": self.type == "ssh" or self.type == "docker" or bool(self.kubernetes and self.kubernetes.scripts_config_map),
            "relative_scripts": self.is_ci,
            "verified": self.is_verified,
            "verified_at": self.verified_at if self.is_verified else "",
            "ci": {
                "kind": self.ci.kind,
                "ref": self.ci.ref,
                # Parameters can be read: Jenkins always, GitLab with an API token.
                "reads_params": self.ci.kind == "jenkins" or bool(self.ci.api_token),
                "extra_vars": self.ci.kind == "gitlab",
                "tags_metrics": self.ci.tags_metrics,
                "events": bool(self.ci.loadlens_url),
            } if self.ci else None,
        }


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    return "" if value is None else str(value).strip()


def _flag(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _port(value: Any, where: Any) -> int:
    if value in (None, ""):
        return DEFAULT_SSH_PORT
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise GeneratorConfigError("lt.gen_config.port_number", where=where) from exc
    if not 1 <= port <= 65535:
        raise GeneratorConfigError("lt.gen_config.port_range", where=where)
    return port


def _dir(value: str, where: Any, label: Any, required: bool, default: str = "") -> str:
    """``label`` names the directory (a dictionary message) in the errors."""
    if not value:
        if required and not default:
            raise GeneratorConfigError("lt.gen_config.dir_required", where=where, label=label)
        return default
    if not is_absolute_path(value) or "\n" in value or "\x00" in value:
        raise GeneratorConfigError("lt.gen_config.dir_absolute", where=where, label=label)
    return value.rstrip("/") or "/"


def _url(value: str, where: Any, label: Any, schemes: tuple[str, ...] = ("http", "https")) -> str:
    if not re.match(rf"^({'|'.join(schemes)})://[^\s/?#]+", value or ""):
        raise GeneratorConfigError("lt.gen_config.url", where=where, label=label, scheme=schemes[-1])
    return value.rstrip("/")


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, Mapping) else {}


def _parse_ssh(raw: Mapping[str, Any], where: str) -> SshSettings:
    ssh_raw = _section(raw, "ssh")
    host = _text(ssh_raw, "host")
    if not host or any(ch.isspace() for ch in host) or host.startswith("-"):
        raise GeneratorConfigError("lt.gen_config.ssh_host", where=where)
    username = _text(ssh_raw, "username")
    if not username:
        raise GeneratorConfigError("lt.gen_config.ssh_user", where=where)
    private_key = str(ssh_raw.get("private_key") or "").strip()
    password = str(ssh_raw.get("password") or "")
    if not private_key and not password:
        raise GeneratorConfigError("lt.gen_config.ssh_credentials", where=where)
    try:
        timeout = int(ssh_raw.get("connect_timeout") or 10)
    except (TypeError, ValueError):
        timeout = 10
    return SshSettings(
        host=host,
        port=_port(ssh_raw.get("port"), where),
        username=username,
        private_key=private_key,
        passphrase=str(ssh_raw.get("passphrase") or ""),
        password=password,
        host_key=" ".join(str(ssh_raw.get("host_key") or "").split()[:2]),
        connect_timeout=max(2, min(timeout, 60)),
    )


def _parse_docker(raw: Mapping[str, Any], where: str) -> DockerSettings:
    docker_raw = _section(raw, "docker")
    url = _url(_text(docker_raw, "url"), where, msg("lt.gen_config.label.docker_url"))
    client_cert = str(docker_raw.get("client_cert") or "").strip()
    client_key = str(docker_raw.get("client_key") or "").strip()
    if bool(client_cert) != bool(client_key):
        raise GeneratorConfigError("lt.gen_config.client_cert_pair", where=where)
    network = _text(raw, "network")
    if network and not VOLUME_NAME_RE.fullmatch(network):
        raise GeneratorConfigError("lt.gen_config.bad_network", where=where, network=network)
    mounts: list[Mount] = []
    raw_mounts = raw.get("mounts") or []
    if not isinstance(raw_mounts, list):
        raise GeneratorConfigError("lt.gen_config.mounts_list", where=where, shape="{source, target, read_only}")
    for item in raw_mounts:
        if not isinstance(item, Mapping):
            raise GeneratorConfigError("lt.gen_config.mounts_list", where=where, shape="{source, target, read_only}")
        source, target = _text(item, "source"), _text(item, "target")
        if not (VOLUME_NAME_RE.fullmatch(source) or is_absolute_path(source)) or "\n" in source:
            raise GeneratorConfigError("lt.gen_config.mount_source", where=where, source=source)
        if not is_absolute_path(target):
            raise GeneratorConfigError("lt.gen_config.mount_target", where=where, target=target)
        mounts.append(Mount(source=source, target=target.rstrip("/") or "/", read_only=_flag(item, "read_only", True)))
    resources = _section(raw, "resources")
    try:
        cpus = float(resources.get("cpus") or 0)
        memory_mb = int(resources.get("memory_mb") or 0)
    except (TypeError, ValueError) as exc:
        raise GeneratorConfigError("lt.gen_config.docker_resources", where=where) from exc
    if cpus < 0 or memory_mb < 0:
        raise GeneratorConfigError("lt.gen_config.negative_resources", where=where)
    return DockerSettings(
        url=url,
        ca_cert=str(docker_raw.get("ca_cert") or "").strip(),
        client_cert=client_cert,
        client_key=client_key,
        verify=_flag(docker_raw, "verify", True),
        network=network,
        mounts=tuple(mounts),
        cpus=cpus,
        memory_mb=memory_mb,
        remove_finished=_flag(raw, "remove_finished", True),
    )


def _parse_kubernetes(raw: Mapping[str, Any], where: str) -> KubernetesSettings:
    k8s_raw = _section(raw, "kubernetes")
    in_cluster = _flag(k8s_raw, "in_cluster", False)
    api_url = _text(k8s_raw, "api_url")
    token = str(k8s_raw.get("token") or "").strip()
    if not in_cluster:
        api_url = _url(api_url, where, msg("lt.gen_config.label.k8s_url"), ("https", "http"))
        if not token:
            raise GeneratorConfigError("lt.gen_config.k8s_token", where=where)
    namespace = _text(raw, "namespace") or "default"
    if not DNS_LABEL_RE.fullmatch(namespace):
        raise GeneratorConfigError("lt.gen_config.bad_namespace", where=where, namespace=namespace)
    scripts = _section(raw, "scripts")
    config_map, pvc = _text(scripts, "config_map"), _text(scripts, "pvc")
    for value, label in ((config_map, "lt.gen_config.label.scripts_config_map"), (pvc, "lt.gen_config.label.scripts_pvc"),
                         (_text(raw, "service_account"), "lt.gen_config.label.service_account"),
                         (_text(raw, "image_pull_secret"), "lt.gen_config.label.pull_secret")):
        if value and not re.fullmatch(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$", value):
            raise GeneratorConfigError("lt.gen_config.bad_k8s_name", where=where, label=msg(label), value=value)
    if config_map and pvc:
        raise GeneratorConfigError("lt.gen_config.scripts_source", where=where)
    selector_raw = raw.get("node_selector") or {}
    if not isinstance(selector_raw, Mapping):
        raise GeneratorConfigError("lt.gen_config.node_selector_object", where=where)
    selector: dict[str, str] = {}
    for key, value in selector_raw.items():
        if not LABEL_KEY_RE.fullmatch(str(key)) or not LABEL_VALUE_RE.fullmatch(str(value)):
            raise GeneratorConfigError("lt.gen_config.bad_node_label", where=where, label=f"{key}={value}")
        selector[str(key)] = str(value)
    resources = _section(raw, "resources")
    cpu, memory = _text(resources, "cpu"), _text(resources, "memory")
    for value, label in ((cpu, "cpu"), (memory, "memory")):
        if value and not QUANTITY_RE.fullmatch(value):
            raise GeneratorConfigError("lt.gen_config.k8s_quantity", where=where, name=label)
    try:
        ttl = int(raw.get("ttl_seconds") or 3600)
    except (TypeError, ValueError) as exc:
        raise GeneratorConfigError("lt.gen_config.ttl_int", where=where) from exc
    return KubernetesSettings(
        api_url=api_url,
        token=token,
        ca_cert=str(k8s_raw.get("ca_cert") or "").strip(),
        verify=_flag(k8s_raw, "verify", True),
        in_cluster=in_cluster,
        namespace=namespace,
        scripts_config_map=config_map,
        scripts_pvc=pvc,
        node_selector=selector,
        cpu=cpu,
        memory=memory,
        service_account=_text(raw, "service_account"),
        image_pull_secret=_text(raw, "image_pull_secret"),
        ttl_seconds=max(60, min(ttl, 7 * 24 * 3600)),
    )


PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,99}$")
MAX_PARAM_OPTIONS = 200


def _parse_param_options(raw: Mapping[str, Any], where: str) -> dict[str, tuple[str, ...]]:
    """``{"ENV": ["stage", "preprod"]}`` or ``{"ENV": "stage, preprod"}`` (the settings form sends text)."""
    value = raw.get("param_options")
    if value in (None, "", {}):
        return {}
    if not isinstance(value, Mapping):
        raise GeneratorConfigError("lt.gen_config.param_options", where=where)
    out: dict[str, tuple[str, ...]] = {}
    for name, options in value.items():
        if isinstance(options, str):
            items = [item.strip() for item in options.split(",")]
        elif isinstance(options, (list, tuple)):
            items = [str(item).strip() for item in options]
        else:
            raise GeneratorConfigError("lt.gen_config.options_type", where=where, name=name)
        items = [item for item in items if item]
        if not PARAM_NAME_RE.fullmatch(str(name)) or not items:
            raise GeneratorConfigError("lt.gen_config.options_empty", where=where, name=name)
        if any(len(item) > 200 or "\n" in item for item in items):
            raise GeneratorConfigError("lt.gen_config.options_long", where=where, name=name)
        out[str(name)] = tuple(dict.fromkeys(items))[:MAX_PARAM_OPTIONS]
    return out


def _parse_ci(raw: Mapping[str, Any], kind: str, where: str) -> CiSettings:
    ci_raw = _section(raw, kind)
    url = _url(_text(ci_raw, "url"), where, msg("lt.gen_config.label.ci_url", system=GENERATOR_TYPE_TITLES[kind]))
    loadlens_text = _text(raw, "loadlens_url")
    loadlens_url = _url(loadlens_text, where, msg("lt.gen_config.label.loadlens_url")) if loadlens_text else ""
    ref = _text(ci_raw, "ref")
    if any(ch.isspace() for ch in ref) or len(ref) > 200:
        raise GeneratorConfigError("lt.gen_config.bad_ref", where=where, ref=ref[:40])
    common: dict[str, Any] = dict(
        verify=_flag(ci_raw, "verify", True),
        loadlens_url=loadlens_url,
        param_options=_parse_param_options(raw, where),
        tags_metrics=_flag(raw, "tags_metrics", False),
    )
    if kind == "gitlab":
        project = _text(ci_raw, "project")
        if not project or any(ch.isspace() for ch in project):
            raise GeneratorConfigError("lt.gen_config.gitlab_project", where=where)
        trigger = str(ci_raw.get("trigger_token") or "").strip()
        api_token = str(ci_raw.get("api_token") or "").strip()
        if not trigger and not api_token:
            raise GeneratorConfigError("lt.gen_config.gitlab_token", where=where)
        inputs_mode = _text(ci_raw, "inputs") or "auto"
        if inputs_mode not in ("auto", "on", "off"):
            raise GeneratorConfigError("lt.gen_config.gitlab_inputs", where=where)
        # The trigger API needs a branch; with an API token "" means the project's default branch.
        return CiSettings(kind=kind, url=url, project=project, ref=ref or ("" if api_token else "main"),
                          trigger_token=trigger, api_token=api_token, inputs_mode=inputs_mode, **common)
    job = _text(ci_raw, "job").strip("/")
    if not job or not re.fullmatch(r"[A-Za-z0-9._ -]+(/[A-Za-z0-9._ -]+)*", job):
        raise GeneratorConfigError("lt.gen_config.jenkins_job", where=where)
    username = _text(ci_raw, "username")
    api_token = str(ci_raw.get("api_token") or "").strip()
    if not username or not api_token:
        raise GeneratorConfigError("lt.gen_config.jenkins_credentials", where=where)
    return CiSettings(kind=kind, url=url, job=job, ref=ref, username=username, api_token=api_token, **common)


def parse_generator(generator_id: str, raw: Mapping[str, Any]) -> Generator:
    where = msg("lt.gen_config.where", generator=generator_id)
    if not GENERATOR_ID_RE.fullmatch(generator_id):
        raise GeneratorConfigError("lt.config.bad_id", where=where)
    if not isinstance(raw, Mapping):
        raise GeneratorConfigError("lt.config.object_expected", where=where)
    gtype = _text(raw, "type") or "ssh"
    if gtype not in GENERATOR_TYPES:
        raise GeneratorConfigError("lt.gen_config.type_one_of", where=where, types=", ".join(GENERATOR_TYPES))
    title = (_text(raw, "title") or generator_id)[:80]
    try:
        max_parallel = int(raw.get("max_parallel") or 0)
    except (TypeError, ValueError) as exc:
        raise GeneratorConfigError("lt.gen_config.max_parallel_int", where=where) from exc
    if not 0 <= max_parallel <= MAX_PARALLEL_CAP:
        raise GeneratorConfigError("lt.gen_config.max_parallel_range", where=where, max=MAX_PARALLEL_CAP)
    verified = raw.get("verified") if isinstance(raw.get("verified"), Mapping) else {}
    verified_at, verified_target = _text(verified, "at")[:40], _text(verified, "target")[:500]
    if gtype == "manual":
        return Generator(id=generator_id, title=title, type=gtype, max_parallel=max_parallel)

    tools_raw = raw.get("tools") if isinstance(raw.get("tools"), Mapping) else {}
    tools: dict[str, GeneratorTool] = {}
    for tool_id, entry in tools_raw.items():
        entry = entry if isinstance(entry, Mapping) else {"executable": entry} if isinstance(entry, str) else {}
        executable, image = _text(entry, "executable"), _text(entry, "image")
        if "\n" in executable or "\x00" in executable or any(ch.isspace() for ch in image):
            raise GeneratorConfigError("lt.gen_config.bad_tool_entry", where=where, tool=tool_id)
        tools[str(tool_id)] = GeneratorTool(
            executable=executable,
            scripts_dir=_dir(_text(entry, "scripts_dir"), where, msg("lt.gen_config.label.tool_scripts_dir", tool=tool_id),
                             required=False)
            if gtype not in CI_TYPES else _text(entry, "scripts_dir"),
            image=image,
        )
    env_raw = raw.get("env") if isinstance(raw.get("env"), Mapping) else {}
    env: dict[str, str] = {}
    for name, value in env_raw.items():
        if not ENV_NAME_RE.fullmatch(str(name)):
            raise GeneratorConfigError("lt.config.bad_env_name", where=where, name=name)
        env[str(name)] = "" if value is None else str(value)
    common: dict[str, Any] = dict(id=generator_id, title=title, type=gtype, tools=tools, max_parallel=max_parallel, env=env,
                                  verified_at=verified_at, verified_target=verified_target)
    if gtype == "ssh":
        return Generator(
            **common,
            ssh=_parse_ssh(raw, where),
            work_dir=_dir(_text(raw, "work_dir"), where, msg("lt.gen_config.label.work_dir"), required=True),
            scripts_dir=_dir(_text(raw, "scripts_dir"), where, msg("lt.gen_config.label.scripts_dir"), required=False),
        )
    if gtype == "docker":
        return Generator(
            **common,
            docker=_parse_docker(raw, where),
            work_dir=_dir(_text(raw, "work_dir"), where, msg("lt.gen_config.label.container_work_dir"), required=False,
                          default="/tmp"),
            scripts_dir=_dir(_text(raw, "scripts_dir"), where, msg("lt.gen_config.label.container_scripts_dir"), required=False),
        )
    if gtype == "kubernetes":
        k8s = _parse_kubernetes(raw, where)
        default_scripts = "/scripts" if (k8s.scripts_config_map or k8s.scripts_pvc) else ""
        return Generator(
            **common,
            kubernetes=k8s,
            work_dir=_dir(_text(raw, "work_dir"), where, msg("lt.gen_config.label.container_work_dir"), required=False,
                          default="/tmp"),
            scripts_dir=_dir(_text(raw, "scripts_dir"), where, msg("lt.gen_config.label.container_scripts_dir"), required=False,
                             default=default_scripts),
        )
    # CI: the pipeline holds the tool, the scripts and their settings.
    common["tools"] = {}
    return Generator(**common, ci=_parse_ci(raw, gtype, where), work_dir=".")


def configured_generators(config: Mapping[str, Any]) -> dict[str, Generator]:
    """Usable generators; a broken entry is skipped (saving the settings reports it)."""
    raw = config.get("load_generators") if isinstance(config.get("load_generators"), Mapping) else {}
    out: dict[str, Generator] = {}
    for generator_id, entry in raw.items():
        try:
            out[str(generator_id)] = parse_generator(str(generator_id), entry)
        except GeneratorConfigError:
            continue
    return out


def validate_generators_section(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise GeneratorConfigError("lt.gen_config.section_object")
    for generator_id, entry in payload.items():
        parse_generator(str(generator_id), entry)


__all__ = [
    "CI_TYPES",
    "CONTAINER_TYPES",
    "CiSettings",
    "DockerSettings",
    "GENERATOR_TYPES",
    "GENERATOR_TYPE_TITLES",
    "Generator",
    "GeneratorConfigError",
    "GeneratorTool",
    "KubernetesSettings",
    "Mount",
    "SshSettings",
    "configured_generators",
    "parse_generator",
    "validate_generators_section",
]
