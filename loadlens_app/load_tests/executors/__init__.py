# SPDX-License-Identifier: Apache-2.0
"""Executors: starting, watching and stopping one instance on a load generator of any type."""

from __future__ import annotations

import threading
from typing import Callable, Mapping

from ..generators import Generator
from ..messages import t
from ..tools import ToolTemplate
from .base import (
    LOG_TAIL_BYTES,
    STORED_LOG_BYTES,
    CheckReport,
    Executor,
    GeneratorError,
    LaunchPlan,
    LaunchResult,
    Probe,
)
from .ci import TOKEN_HEADER, CiExecutor, check_ci_generator, ci_api, events_url, jenkins_path, pipeline_variables
from .docker import DockerExecutor, check_docker_generator, container_name, demux, docker_api
from .http import ApiClient, ApiError
from .kubernetes import KubernetesExecutor, check_kubernetes_generator, job_name, kubernetes_api
from .manual import ManualExecutor
from .ssh import (
    KILL_AFTER_SECONDS,
    RemoteShell,
    SshExecutor,
    SshShell,
    _check_message,
    check_script,
    check_ssh_generator,
    control_script,
    instance_dir,
    kill_script,
    launch_script,
    list_scripts_script,
    load_private_key,
    logs_script,
    parse_host_key,
    parse_key_values,
    parse_probe,
    parse_script_list,
    probe_script,
    runner_script,
    shell_for,
    signal_script,
)

_CLIENTS: dict[str, tuple[tuple, ApiClient]] = {}
_CLIENTS_LOCK = threading.Lock()


def _cached_client(generator: Generator, identity: tuple, build: Callable[[Generator], ApiClient]) -> ApiClient:
    """One HTTP session per generator (connection reuse); a settings change builds a new one."""
    with _CLIENTS_LOCK:
        cached = _CLIENTS.get(generator.id)
        if cached and cached[0] == identity:
            return cached[1]
        client = build(generator)
        _CLIENTS[generator.id] = (identity, client)
        return client


def default_executor_for(generator: Generator) -> Executor:
    if generator.is_manual:
        return ManualExecutor()
    if generator.type == "ssh":
        return SshExecutor(generator, shell_for(generator))
    if generator.type == "docker" and generator.docker is not None:
        return DockerExecutor(generator, _cached_client(generator, ("docker",) + generator.docker.identity(), docker_api))
    if generator.type == "kubernetes" and generator.kubernetes is not None:
        identity = ("k8s",) + generator.kubernetes.identity()
        return KubernetesExecutor(generator, _cached_client(generator, identity, kubernetes_api))
    if generator.is_ci and generator.ci is not None:
        return CiExecutor(generator, api=_cached_client(generator, ("ci",) + generator.ci.identity(), ci_api))
    raise GeneratorError("lt.executor.type_unsupported", generator=generator.title, type=generator.type)


# Replaced in tests with a factory of fakes.
executor_factory: Callable[[Generator], Executor] = default_executor_for


def executor_for(generator: Generator) -> Executor:
    return executor_factory(generator)


def check_generator(generator: Generator, tools: Mapping[str, ToolTemplate]) -> CheckReport:
    """Connection check of the settings page, by generator type (texts in the language of the request)."""
    if generator.is_manual:
        return CheckReport(ok=True, message=t("lt.check.manual"))
    if generator.type == "ssh":
        return check_ssh_generator(generator, tools)
    if generator.type == "docker":
        return check_docker_generator(generator, tools)
    if generator.type == "kubernetes":
        return check_kubernetes_generator(generator, tools)
    if generator.is_ci:
        return check_ci_generator(generator, tools)
    return CheckReport(ok=False, message=t("lt.check.type_unsupported", type=generator.type))


__all__ = [
    "ApiClient",
    "ApiError",
    "CheckReport",
    "CiExecutor",
    "DockerExecutor",
    "Executor",
    "GeneratorError",
    "KILL_AFTER_SECONDS",
    "KubernetesExecutor",
    "LOG_TAIL_BYTES",
    "LaunchPlan",
    "LaunchResult",
    "ManualExecutor",
    "Probe",
    "RemoteShell",
    "STORED_LOG_BYTES",
    "SshExecutor",
    "SshShell",
    "TOKEN_HEADER",
    "check_generator",
    "check_script",
    "container_name",
    "control_script",
    "default_executor_for",
    "demux",
    "events_url",
    "executor_for",
    "instance_dir",
    "jenkins_path",
    "job_name",
    "kill_script",
    "launch_script",
    "list_scripts_script",
    "load_private_key",
    "logs_script",
    "parse_host_key",
    "parse_key_values",
    "parse_probe",
    "parse_script_list",
    "pipeline_variables",
    "probe_script",
    "runner_script",
    "signal_script",
]
