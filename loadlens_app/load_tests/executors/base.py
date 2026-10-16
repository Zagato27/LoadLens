"""Types shared by the executors: what to launch, what a probe says, what a generator check found."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..tools import CommandLine, ToolTemplate

LOG_TAIL_BYTES = 64 * 1024
# Kept in the run record when an instance ends, so the log survives the removal of its container or job.
STORED_LOG_BYTES = 32 * 1024
SCRIPT_LIST_LIMIT = 500
SKIPPED_DIRS = ("node_modules", ".git", "target", "build", ".venv", "venv", "__pycache__", "results")


class GeneratorError(RuntimeError):
    """The generator cannot be reached or refused an operation; the message is for the user."""


@dataclass(frozen=True)
class LaunchPlan:
    """Everything an executor needs to start one instance."""

    ref: str
    command: CommandLine
    max_seconds: int
    run_id: str = ""
    run_name: str = ""
    run_tag: str = ""
    instance_index: int = 0
    instance_name: str = ""
    instance_tag: str = ""
    tool_id: str = ""
    image: str = ""
    grace_seconds: int = 30
    # Tool commands that talk to the running process (soft stop, status); argv, run where the tool runs.
    stop_command: tuple[str, ...] = ()
    status_command: tuple[str, ...] = ()
    # CI generators run the team's pipeline as it is: the branch, the values for it and the per-instance
    # secret the pipeline may present to report events. variables (GitLab) and parameters (Jenkins) are
    # name -> text; inputs (GitLab spec:inputs) keep their types.
    ci_token: str = ""
    ci_ref: str = ""
    ci_variables: dict[str, str] = field(default_factory=dict)
    ci_inputs: dict[str, Any] = field(default_factory=dict)
    ci_parameters: dict[str, str] = field(default_factory=dict)
    # LOADLENS_* values: sent as variables where the CI accepts them, dropped where it refuses them.
    ci_auto: dict[str, str] = field(default_factory=dict)
    # Jenkins: the job declares parameters (buildWithParameters) or none (build).
    ci_parameterized: bool = True


@dataclass(frozen=True)
class LaunchResult:
    ref: str
    already_running: bool = False
    # Where people can see it: the GitLab pipeline or the Jenkins build page.
    url: str = ""
    message: str = ""


@dataclass(frozen=True)
class Probe:
    """State of one instance as its executor sees it.

    ``state``: ``starting`` (accepted, the tool has not started yet), ``running``, ``exited`` (exit code
    known, may be ``None`` when the platform does not report one), ``dead`` (gone without an exit code)
    or ``missing`` (nothing left on the generator). ``message`` explains waiting states, e.g. an image
    that cannot be pulled.
    """

    state: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    remote_now: Optional[float] = None
    message: str = ""
    # A waiting state that will not resolve by itself (bad image name, missing secret).
    stuck: bool = False
    # A new reference to keep instead of the launch one (a Jenkins build replaces its queue item).
    ref: str = ""
    # Page of the pipeline or build once it is known.
    url: str = ""


@dataclass
class CheckReport:
    """Result of a generator connection check.

    ``message`` is the one-line text of the settings card; the setup wizard reads the structured part:
    ``connected`` (the generator answered and took the credentials, even when ``ok`` is false because of a
    directory, a tool or a permission) and ``problems`` one by one.
    """

    ok: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    connected: bool = False
    problems: list[str] = field(default_factory=list)

    def to_details(self) -> dict[str, Any]:
        """Details with the same keys for every generator type."""
        out: dict[str, Any] = {"system": "", "tools_found": [], "tools_missing": [], "versions": {}, "images": {}}
        out.update(self.details)
        out["connected"] = bool(self.connected)
        out["problems"] = [str(problem) for problem in self.problems]
        return out


class Executor(Protocol):
    # The generator stops the tool after max_seconds by itself (otherwise the watcher enforces it).
    enforces_deadline: bool
    # The executor can send an arbitrary signal; otherwise stopping removes the process with its grace period.
    supports_signals: bool
    # Tool control commands (stop_command, status_command) can run next to the tool.
    supports_control: bool
    # Optional, True when absent: whether the executor can stop the instance at all (a GitLab generator
    # without an API token cannot cancel its pipeline). The watcher then closes the instance window itself.

    def launch(self, plan: LaunchPlan) -> LaunchResult: ...

    def probe(self, ref: str) -> Probe: ...

    def signal(self, ref: str, signal: str, grace_seconds: int = 30) -> None: ...

    def kill(self, ref: str) -> None: ...

    def logs(self, ref: str, max_bytes: int = LOG_TAIL_BYTES) -> str: ...

    def cleanup(self, ref: str) -> None: ...

    def control(self, ref: str, argv: tuple[str, ...], cwd: str = "") -> tuple[int, str]: ...

    def list_scripts(self, scripts_dir: str, tool: ToolTemplate) -> Optional[list[str]]: ...


__all__ = [
    "CheckReport",
    "Executor",
    "GeneratorError",
    "LOG_TAIL_BYTES",
    "LaunchPlan",
    "LaunchResult",
    "Probe",
    "SCRIPT_LIST_LIMIT",
    "SKIPPED_DIRS",
    "STORED_LOG_BYTES",
]
