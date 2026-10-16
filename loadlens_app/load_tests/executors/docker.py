"""Docker generators: one container per instance through the Docker Engine API.

LoadLens talks HTTP(S) to the engine: ``https://host:2376`` with client certificates, or a socket proxy
such as tecnativa/docker-socket-proxy in front of ``/var/run/docker.sock`` (containers, images, exec and
networks allowed). Access to the Docker API is equivalent to root on the Docker host, so it belongs to a
dedicated load generation host and must be reachable only from LoadLens.

A container is named after the run and the instance, so starting it twice finds the first one. Docker has
no run time limit: the watcher stops a container that runs longer than the run allows, and a label keeps
the deadline for later clean-up.
"""

from __future__ import annotations

import posixpath
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ..generators import Generator
from ..tools import ToolTemplate, render_version_command
from .base import LOG_TAIL_BYTES, CheckReport, GeneratorError, LaunchPlan, LaunchResult, Probe, SKIPPED_DIRS, SCRIPT_LIST_LIMIT
from .http import ApiClient, ApiError

API_PREFIX = "/v1.41"
NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
LABEL_PREFIX = "loadlens."
HELPER_TIMEOUT_SEC = 60
LOG_TAIL_LINES = 3000


def container_name(run_id: str, index: int, tag: str) -> str:
    name = f"loadlens-{run_id[:12]}-{index + 1:02d}-{NAME_SANITIZE_RE.sub('-', tag).strip('-')}"
    return name[:120].rstrip("-.")


def _docker_time(value: Any) -> Optional[float]:
    """``2026-10-14T14:05:27.123456789Z`` as Unix seconds; Docker writes ``0001-01-01...`` for "never"."""
    text = str(value or "")
    if not text or text.startswith("0001-"):
        return None
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})?$", text)
    if not match:
        return None
    fraction = (match.group(2) or ".0")[:7]
    zone = match.group(3) or "Z"
    stamp = datetime.fromisoformat(f"{match.group(1)}{fraction}{'+00:00' if zone == 'Z' else zone}")
    return stamp.astimezone(timezone.utc).timestamp()


def demux(raw: bytes) -> str:
    """Text of a Docker log or exec stream: 8-byte frame headers (stream, size) unless the container has a TTY."""
    if not raw:
        return ""
    if raw[0] not in (0, 1, 2) or len(raw) < 8 or raw[1:4] != b"\x00\x00\x00":
        return raw.decode("utf-8", "replace")
    out = bytearray()
    pos = 0
    while pos + 8 <= len(raw):
        size = int.from_bytes(raw[pos + 4:pos + 8], "big")
        out.extend(raw[pos + 8:pos + 8 + size])
        pos += 8 + size
    return out.decode("utf-8", "replace")


def docker_api(generator: Generator) -> ApiClient:
    if generator.docker is None:
        raise GeneratorError(f"Генератор «{generator.title}» не работает через Docker")
    cfg = generator.docker
    return ApiClient(
        cfg.url + API_PREFIX,
        name="Docker API",
        verify=cfg.verify,
        ca_cert=cfg.ca_cert,
        client_cert=cfg.client_cert,
        client_key=cfg.client_key,
    )


def _mounts(generator: Generator) -> list[dict[str, Any]]:
    assert generator.docker is not None
    return [
        {"Type": "bind" if mount.is_bind else "volume", "Source": mount.source, "Target": mount.target, "ReadOnly": mount.read_only}
        for mount in generator.docker.mounts
    ]


class DockerExecutor:
    enforces_deadline = False
    supports_signals = True
    supports_control = True

    def __init__(self, generator: Generator, api: ApiClient) -> None:
        self.generator = generator
        self.api = api

    def _host_config(self) -> dict[str, Any]:
        cfg = self.generator.docker
        assert cfg is not None
        host: dict[str, Any] = {"Init": True, "Mounts": _mounts(self.generator)}
        if cfg.network:
            host["NetworkMode"] = cfg.network
        if cfg.cpus:
            host["NanoCpus"] = int(cfg.cpus * 1_000_000_000)
        if cfg.memory_mb:
            host["Memory"] = int(cfg.memory_mb) * 1024 * 1024
        return host

    def launch(self, plan: LaunchPlan) -> LaunchResult:
        if not plan.image:
            raise GeneratorError(f"Для инструмента {plan.tool_id} не задан образ контейнера")
        body = {
            "Image": plan.image,
            # [""] resets the image entrypoint: argv[0] of the command is the tool itself.
            "Entrypoint": [""],
            "Cmd": list(plan.command.argv),
            "Env": [f"{name}={value}" for name, value in plan.command.env.items()],
            "WorkingDir": plan.command.cwd,
            "Labels": {
                f"{LABEL_PREFIX}managed": "true",
                f"{LABEL_PREFIX}run_id": plan.run_id,
                f"{LABEL_PREFIX}instance": str(plan.instance_index),
                f"{LABEL_PREFIX}tool": plan.tool_id,
                f"{LABEL_PREFIX}deadline": str(int(time.time()) + int(plan.max_seconds)),
            },
            "HostConfig": self._host_config(),
        }
        already = False
        try:
            self.api.request("POST", "/containers/create", params={"name": plan.ref}, json=body, ok=(201,))
        except ApiError as exc:
            if exc.status == 404:
                raise GeneratorError(f"Образа {plan.image} нет на Docker-хосте: скачайте его заранее (docker pull {plan.image})") from exc
            if exc.status != 409:
                raise
            state = self.api.json("GET", f"/containers/{plan.ref}/json").get("State") or {}
            already = state.get("Status") != "created"
            if already:
                return LaunchResult(ref=plan.ref, already_running=True)
        self.api.request("POST", f"/containers/{plan.ref}/start", ok=(204, 304))
        return LaunchResult(ref=plan.ref, already_running=already)

    def probe(self, ref: str) -> Probe:
        try:
            data = self.api.json("GET", f"/containers/{ref}/json")
        except ApiError as exc:
            if exc.status == 404:
                return Probe(state="missing", remote_now=time.time())
            raise
        state = data.get("State") or {}
        status = str(state.get("Status") or "")
        started = _docker_time(state.get("StartedAt"))
        if status == "created":
            return Probe(state="starting", message="Контейнер создан, ещё не запущен")
        if status in ("running", "paused", "restarting"):
            return Probe(state="running", started_at=started)
        message = "Контейнер убит по нехватке памяти (OOM)" if state.get("OOMKilled") else str(state.get("Error") or "")
        code = state.get("ExitCode")
        return Probe(
            state="exited",
            started_at=started,
            finished_at=_docker_time(state.get("FinishedAt")),
            exit_code=int(code) if isinstance(code, int) else None,
            message=message,
        )

    def signal(self, ref: str, signal: str, grace_seconds: int = 30) -> None:
        try:
            self.api.request("POST", f"/containers/{ref}/kill", params={"signal": f"SIG{signal}"}, ok=(204,))
        except ApiError as exc:
            if exc.status not in (404, 409):  # gone or not running: nothing to stop
                raise

    def kill(self, ref: str) -> None:
        self.signal(ref, "KILL")

    def logs(self, ref: str, max_bytes: int = LOG_TAIL_BYTES) -> str:
        try:
            response = self.api.request(
                "GET", f"/containers/{ref}/logs", params={"stdout": 1, "stderr": 1, "tail": LOG_TAIL_LINES}, timeout=30,
            )
        except ApiError as exc:
            if exc.status == 404:
                return ""
            raise
        text = demux(response.content)
        return text[-max_bytes:]

    def cleanup(self, ref: str) -> None:
        cfg = self.generator.docker
        if cfg is None or not cfg.remove_finished:
            return
        try:
            self.api.request("DELETE", f"/containers/{ref}", params={"force": 1}, ok=(204,))
        except ApiError as exc:
            if exc.status != 404:
                raise

    def control(self, ref: str, argv: tuple[str, ...], cwd: str = "") -> tuple[int, str]:
        body: dict[str, Any] = {"Cmd": list(argv), "AttachStdout": True, "AttachStderr": True}
        if cwd:
            body["WorkingDir"] = cwd
        exec_id = self.api.json("POST", f"/containers/{ref}/exec", json=body, ok=(201,)).get("Id")
        response = self.api.request("POST", f"/exec/{exec_id}/start", json={"Detach": False, "Tty": False}, ok=(200,), timeout=60)
        info = self.api.json("GET", f"/exec/{exec_id}/json")
        code = info.get("ExitCode")
        return (int(code) if isinstance(code, int) else 1), demux(response.content).strip()

    def run_helper(self, image: str, script: str, timeout: int = HELPER_TIMEOUT_SEC) -> tuple[int, str]:
        """Runs ``sh -c script`` in a short-lived container with the generator's mounts; returns (code, output)."""
        name = f"loadlens-helper-{secrets.token_hex(6)}"
        body = {
            "Image": image,
            "Entrypoint": [""],
            "Cmd": ["sh", "-c", script],
            "Labels": {f"{LABEL_PREFIX}managed": "true", f"{LABEL_PREFIX}helper": "true"},
            "HostConfig": {"Mounts": _mounts(self.generator)},
        }
        try:
            self.api.request("POST", "/containers/create", params={"name": name}, json=body, ok=(201,))
        except ApiError as exc:
            if exc.status == 404:
                raise GeneratorError(f"Образа {image} нет на Docker-хосте") from exc
            raise
        try:
            self.api.request("POST", f"/containers/{name}/start", ok=(204, 304))
            result = self.api.json("POST", f"/containers/{name}/wait", ok=(200,), timeout=timeout)
            output = self.logs(name, LOG_TAIL_BYTES)
            return int(result.get("StatusCode") or 0), output
        finally:
            try:
                self.api.request("DELETE", f"/containers/{name}", params={"force": 1}, ok=(204, 404))
            except GeneratorError:
                pass

    def list_scripts(self, scripts_dir: str, tool: ToolTemplate) -> Optional[list[str]]:
        from .ssh import list_scripts_script, parse_script_list

        image = self.generator.image_for(tool)
        if not image:
            raise GeneratorError(f"Для инструмента {tool.title} не задан образ контейнера")
        code, out = self.run_helper(image, list_scripts_script(scripts_dir, list(tool.scripts_glob)))
        return parse_script_list(out, tool.script_kind)


def check_docker_generator(generator: Generator, tools: Mapping[str, ToolTemplate]) -> CheckReport:
    """Ping, engine version, images of the offered tools, the network and named volumes."""
    assert generator.docker is not None
    api = docker_api(generator)
    cfg = generator.docker
    try:
        api.request("GET", "/_ping")
        version = api.json("GET", "/version")
    except GeneratorError as exc:
        return CheckReport(ok=False, message=str(exc))
    problems: list[str] = []
    found: list[str] = []
    missing: list[str] = []
    images: dict[str, str] = {}
    for tool_id, tool in tools.items():
        if generator.tools and tool_id not in generator.tools:
            continue
        image = generator.image_for(tool)
        if not image:
            if tool_id in generator.tools:
                problems.append(f"для {tool.title} не задан образ")
            continue
        try:
            api.request("GET", f"/images/{image}/json")
            found.append(tool_id)
            images[tool_id] = image
        except ApiError as exc:
            if exc.status != 404:
                return CheckReport(ok=False, message=str(exc), connected=True, problems=[str(exc)])
            missing.append(tool_id)
            problems.append(f"нет образа {image} ({tool.title}): docker pull {image} на Docker-хосте")
    if cfg.network:
        try:
            api.request("GET", f"/networks/{cfg.network}")
        except ApiError as exc:
            problems.append(f"нет сети {cfg.network}" if exc.status == 404 else str(exc))
    for mount in cfg.mounts:
        if mount.is_bind:
            continue
        try:
            api.request("GET", f"/volumes/{mount.source}")
        except ApiError as exc:
            problems.append(f"нет тома {mount.source}: Docker создаст пустой" if exc.status == 404 else str(exc))
    plain = cfg.url.startswith("http://") and not re.match(r"^http://(127\.0\.0\.1|localhost|\[::1\])[:/]", cfg.url)
    if plain:
        problems.append("Docker API без TLS: доступ к нему равен правам root на хосте, закройте его сетью")
    titles = [tools[tool_id].title for tool_id in found]
    parts = [f"Docker {version.get('Version', '?')} ({version.get('Os', '?')}/{version.get('Arch', '?')})"]
    parts.append(f"образы есть: {', '.join(titles)}" if titles else "образов инструментов нет")
    if problems:
        parts.append(f"проблемы: {'; '.join(problems)}")
    fatal = not found and bool(generator.tools)
    return CheckReport(
        ok=not fatal,
        message="; ".join(parts),
        details={"system": f"Docker {version.get('Version', '?')} {version.get('Os', '')}/{version.get('Arch', '')}".strip(),
                 "tools_found": found, "tools_missing": missing, "images": images, "api_version": version.get("ApiVersion", "")},
        connected=True,
        problems=problems,
    )


__all__ = ["DockerExecutor", "check_docker_generator", "container_name", "demux", "docker_api"]
