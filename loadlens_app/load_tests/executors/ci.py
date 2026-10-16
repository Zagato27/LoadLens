"""CI generators: the team's own GitLab pipeline or Jenkins job, one run per instance.

The pipeline holds the tool, the scripts and their settings; LoadLens starts it with the branch and the
parameter values chosen in the launch form (see ``load_tests/ci_params.py``), follows its status and log
through the CI API and cancels it on stop. Next to the values LoadLens passes its own ``LOADLENS_*``
variables (GitLab) or the ``LOADLENS_*`` parameters the job declares (Jenkins): the run tag, which the
pipeline may give to the tool so that run metrics can be told apart, and, when the generator has the
address of LoadLens, where and with which per-instance secret the pipeline may report ``started`` and
``finished`` for an exact window. Neither is required: the report window is cut by the RPS data anyway.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Mapping, Optional
from urllib.parse import quote, urlsplit

from ..generators import Generator
from ..tools import ToolTemplate, public_url
from .base import LOG_TAIL_BYTES, CheckReport, GeneratorError, LaunchPlan, LaunchResult, Probe
from .http import ApiClient, ApiError

TOKEN_HEADER = "X-LoadLens-Token"
GITLAB_WAITING = frozenset({"created", "waiting_for_resource", "preparing", "pending", "scheduled", "manual"})
GITLAB_RUNNING = frozenset({"running", "canceling"})

EventsReader = Callable[[str, int], list[dict[str, Any]]]


def events_url(base: str, run_id: str, index: int) -> str:
    return f"{base.rstrip('/')}/load_tests/ci/{run_id}/{index}/events"


def pipeline_variables(generator: Generator, run_id: str, run_name: str, run_tag: str, instance_name: str,
                       index: int, max_seconds: int, token: str = "") -> dict[str, str]:
    """LOADLENS_* values of an instance; the events address and secret only when LoadLens can be reached."""
    assert generator.ci is not None
    values = {
        "LOADLENS_RUN_ID": run_id,
        "LOADLENS_RUN_NAME": run_name,
        "LOADLENS_RUN_TAG": run_tag,
        "LOADLENS_INSTANCE": instance_name,
        "LOADLENS_INSTANCE_INDEX": str(index),
        "LOADLENS_MAX_SECONDS": str(int(max_seconds)),
    }
    if generator.ci.loadlens_url and token:
        values["LOADLENS_EVENTS_URL"] = events_url(generator.ci.loadlens_url, run_id, index)
        values["LOADLENS_TOKEN"] = token
    return values


def _store_events(run_id: str, index: int) -> list[dict[str, Any]]:
    from ..store import get_store

    run = get_store().get(run_id)
    if run is None:
        return []
    return [event for event in run.events if int(event.get("index", -1)) == index]


def _time(value: Any) -> Optional[float]:
    """ISO time of GitLab or of an event (``2026-10-15T10:00:00.123Z``) as a timestamp."""
    text = str(value or "")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _jenkins_job_path(job: str) -> str:
    return "".join(f"/job/{quote(part, safe='')}" for part in job.split("/") if part)


def jenkins_path(url: str, base_url: str) -> str:
    """Path of a URL Jenkins named (queue item, build) relative to the configured Jenkins address.

    Jenkins builds these URLs from its own "Jenkins URL" setting, which may name another host than the
    address LoadLens uses; only the path is kept, so the requests (with credentials) stay on that address.
    """
    text = str(url or "").strip()
    if text.lower().startswith(("http://", "https://")):
        text = urlsplit(text).path
    base_path = urlsplit(base_url).path.rstrip("/")
    if base_path and (text == base_path or text.startswith(base_path + "/")):
        text = text[len(base_path):]
    return "/" + text.strip("/")


def _variables_refused(exc: ApiError) -> bool:
    """GitLab 17.7+ may forbid pipeline variables for a project ("Insufficient permissions to set pipeline variables")."""
    text = f"{exc} {exc.body}".lower()
    return exc.status in (400, 403) and "pipeline variables" in text


class CiExecutor:
    enforces_deadline = True  # CI jobs have their own timeout; the watcher also cancels after the run limit
    supports_signals = False
    supports_control = False

    @property
    def supports_stop(self) -> bool:
        """GitLab pipelines can be cancelled only with an API token; Jenkins always uses its user and token."""
        return self.cfg.kind != "gitlab" or bool(self.cfg.api_token)

    def __init__(self, generator: Generator, events: Optional[EventsReader] = None, api: Optional[ApiClient] = None) -> None:
        assert generator.ci is not None
        self.generator = generator
        self.cfg = generator.ci
        self.events = events or _store_events
        if api is not None:
            self.api = api
        elif self.cfg.kind == "jenkins":
            self.api = ApiClient(self.cfg.url, name="Jenkins", verify=self.cfg.verify, auth=(self.cfg.username, self.cfg.api_token))
        else:
            self.api = ApiClient(self.cfg.url, name="GitLab", verify=self.cfg.verify)
            if self.cfg.api_token:
                self.api.session.headers["PRIVATE-TOKEN"] = self.cfg.api_token

    # ---- refs: "<kind>:<run_id>:<index>:<pipeline id, or queue:/build path>" ----

    @staticmethod
    def parse_ref(ref: str) -> tuple[str, str, int, str]:
        kind, run_id, index, rest = (ref.split(":", 3) + ["", "", "", ""])[:4]
        try:
            return kind, run_id, int(index), rest
        except ValueError as exc:
            raise GeneratorError(f"Непонятная ссылка на задачу CI: {ref}") from exc

    @property
    def _project(self) -> str:
        return f"/api/v4/projects/{quote(self.cfg.project, safe='')}"

    def _path(self, url: str) -> str:
        return jenkins_path(url, self.cfg.url)

    def _page(self, path: str) -> str:
        return f"{public_url(self.cfg.url).rstrip('/')}{path}"

    # ---- launch ----

    def launch(self, plan: LaunchPlan) -> LaunchResult:
        if self.cfg.kind == "gitlab":
            return self._launch_gitlab(plan)
        return self._launch_jenkins(plan)

    def _create_pipeline(self, plan: LaunchPlan, ref: str, variables: dict[str, str]) -> Mapping[str, Any]:
        if self.cfg.api_token:
            body: dict[str, Any] = {"ref": ref, "variables": [{"key": key, "value": value} for key, value in variables.items()]}
            if plan.ci_inputs:
                body["inputs"] = plan.ci_inputs
            return self.api.json("POST", f"{self._project}/pipeline", json=body, ok=(200, 201))
        if plan.ci_inputs:
            # Inputs keep their types only in JSON; GitLab versions that take inputs also take JSON here.
            body = {"token": self.cfg.trigger_token, "ref": ref, "variables": variables, "inputs": plan.ci_inputs}
            return self.api.json("POST", f"{self._project}/trigger/pipeline", json=body, ok=(200, 201))
        form = {"token": self.cfg.trigger_token, "ref": ref}
        form.update({f"variables[{name}]": value for name, value in variables.items()})
        return self.api.json("POST", f"{self._project}/trigger/pipeline", data=form, ok=(200, 201))

    def _launch_gitlab(self, plan: LaunchPlan) -> LaunchResult:
        ref = plan.ci_ref or self.cfg.ref or "main"
        variables = {**plan.ci_auto, **plan.ci_variables}
        message = ""
        try:
            pipeline = self._create_pipeline(plan, ref, variables)
        except ApiError as exc:
            if not _variables_refused(exc):
                raise
            if plan.ci_variables:
                raise GeneratorError(
                    "Проект GitLab запрещает переменные пайплайна: задайте значения входными параметрами (spec:inputs) "
                    "или разрешите переменные в настройках CI/CD проекта"
                ) from exc
            pipeline = self._create_pipeline(plan, ref, {})
            message = "проект запрещает переменные пайплайна: LOADLENS_* не переданы"
        pipeline_id = pipeline.get("id")
        if not pipeline_id:
            raise GeneratorError(f"GitLab не вернул номер пайплайна: {str(pipeline)[:200]}")
        url = str(pipeline.get("web_url") or "")
        return LaunchResult(ref=f"gitlab:{plan.run_id}:{plan.instance_index}:{pipeline_id}",
                            url=public_url(url) if url else "", message=message)

    def _job(self, ref: str) -> str:
        """Path of the job; a branch of a multibranch job is a job of its own."""
        base = _jenkins_job_path(self.cfg.job)
        return f"{base}/job/{quote(ref, safe='')}" if ref else base

    def _launch_jenkins(self, plan: LaunchPlan) -> LaunchResult:
        job = self._job(plan.ci_ref)
        if plan.ci_parameterized:
            # Form body, not the query string: values must not end up in access logs of Jenkins or proxies.
            response = self.api.request("POST", f"{job}/buildWithParameters", data=dict(plan.ci_parameters), ok=(200, 201))
        else:
            response = self.api.request("POST", f"{job}/build", ok=(200, 201))
        location = response.headers.get("Location", "")
        if not location:
            raise GeneratorError("Jenkins не вернул ссылку на очередь сборки")
        return LaunchResult(ref=f"jenkins:{plan.run_id}:{plan.instance_index}:queue:{self._path(location)}")

    # ---- status ----

    def _jenkins_build(self, rest: str) -> tuple[str, Optional[dict[str, Any]]]:
        """(build path or the queue path, build JSON when the build exists)."""
        if rest.startswith("queue:"):
            queue = self._path(rest[len("queue:"):])
            item = self.api.json("GET", f"{queue}/api/json")
            executable = item.get("executable") or {}
            if not executable.get("url"):
                if item.get("cancelled"):
                    return queue, {"result": "ABORTED", "building": False}
                return queue, None
            rest = str(executable["url"])
        build = self._path(rest)
        return build, self.api.json("GET", f"{build}/api/json")

    def probe(self, ref: str) -> Probe:
        kind, run_id, index, rest = self.parse_ref(ref)
        events = self.events(run_id, index)
        started = next((e for e in events if e.get("event") == "started"), None)
        finished = next((e for e in reversed(events) if e.get("event") == "finished"), None)
        event_started = _time(started.get("at")) if started else None
        if finished:
            code = finished.get("exit_code")
            return Probe(state="exited", started_at=event_started, finished_at=_time(finished.get("at")),
                         exit_code=int(code) if isinstance(code, int) else None, message=str(finished.get("message") or ""))
        if kind == "gitlab":
            return self._probe_gitlab(rest, event_started)
        return self._probe_jenkins(run_id, index, rest, event_started)

    def _probe_gitlab(self, pipeline_id: str, event_started: Optional[float]) -> Probe:
        if not self.cfg.api_token:
            # Without the API only the events of the pipeline say anything: the window counts from the launch.
            return Probe(state="running", started_at=event_started,
                         message="" if event_started else "Без API-токена LoadLens не видит статус пайплайна: окно закроется "
                                                          "событием пайплайна, кнопкой «Остановить» или по лимиту длительности")
        pipeline = self.api.json("GET", f"{self._project}/pipelines/{pipeline_id}")
        status = str(pipeline.get("status") or "")
        url = public_url(str(pipeline.get("web_url") or ""))
        started_at = event_started or _time(pipeline.get("started_at"))
        finished_at = _time(pipeline.get("finished_at"))
        if status == "success":
            return Probe(state="exited", started_at=started_at, finished_at=finished_at, exit_code=0, url=url)
        if status == "failed":
            return Probe(state="exited", started_at=started_at, finished_at=finished_at, exit_code=1,
                         message="Пайплайн GitLab упал", url=url)
        if status in ("canceled", "skipped"):
            return Probe(state="exited", started_at=started_at, finished_at=finished_at, exit_code=None,
                         message=f"Пайплайн GitLab: {status}", url=url)
        if status in GITLAB_RUNNING:
            return Probe(state="running", started_at=started_at, url=url)
        label = f"Пайплайн GitLab: {status}" if status else "Пайплайн GitLab создаётся"
        return Probe(state="starting", message=label, url=url)

    def _probe_jenkins(self, run_id: str, index: int, rest: str, event_started: Optional[float]) -> Probe:
        build_path, build = self._jenkins_build(rest)
        new_ref = ""
        url = ""
        if build is not None and "/queue/" not in build_path:
            url = self._page(build_path)
            if rest.startswith("queue:"):
                # The queue item expires minutes after the build starts: keep the build path instead.
                new_ref = f"jenkins:{run_id}:{index}:{build_path}"
        if build is None:
            return Probe(state="starting", message="Сборка Jenkins в очереди", ref=new_ref)
        timestamp = build.get("timestamp")
        started_at = event_started or (float(timestamp) / 1000 if isinstance(timestamp, (int, float)) and timestamp else None)
        if not build.get("building", True):
            result = str(build.get("result") or "")
            code = 0 if result == "SUCCESS" else (None if result == "ABORTED" else 1)
            duration = build.get("duration")
            finished_at = (float(timestamp) + float(duration)) / 1000 if isinstance(timestamp, (int, float)) and timestamp \
                and isinstance(duration, (int, float)) and duration else None
            return Probe(state="exited", started_at=started_at, finished_at=finished_at, exit_code=code,
                         message=f"Сборка Jenkins: {result}", ref=new_ref, url=url)
        return Probe(state="running", started_at=started_at, ref=new_ref, url=url)

    # ---- stop and logs ----

    def signal(self, ref: str, signal: str, grace_seconds: int = 30) -> None:
        kind, run_id, index, rest = self.parse_ref(ref)
        if kind == "gitlab":
            if not self.cfg.api_token:
                raise GeneratorError("Остановка недоступна: у генератора GitLab не задан API-токен")
            try:
                self.api.request("POST", f"{self._project}/pipelines/{rest}/cancel", ok=(200, 201))
            except ApiError as exc:
                if exc.status != 404:
                    raise
            return
        if rest.startswith("queue:"):
            queue = self._path(rest[len("queue:"):])
            item_id = queue.rsplit("/", 1)[-1]
            self.api.request("POST", "/queue/cancelItem", params={"id": item_id}, ok=(200, 204, 302))
            return
        self.api.request("POST", f"{self._path(rest)}/stop", ok=(200, 302))

    def kill(self, ref: str) -> None:
        self.signal(ref, "KILL")

    def logs(self, ref: str, max_bytes: int = LOG_TAIL_BYTES) -> str:
        kind, run_id, index, rest = self.parse_ref(ref)
        try:
            if kind == "gitlab":
                if not self.cfg.api_token:
                    return ""
                jobs = self.api.json("GET", f"{self._project}/pipelines/{rest}/jobs") or []
                if not jobs:
                    return ""
                trace = self.api.request("GET", f"{self._project}/jobs/{jobs[0].get('id')}/trace")
                return trace.text[-max_bytes:]
            if rest.startswith("queue:"):
                return ""
            return self.api.request("GET", f"{self._path(rest)}/consoleText").text[-max_bytes:]
        except ApiError as exc:
            if exc.status in (401, 403, 404):
                return ""
            raise

    def cleanup(self, ref: str) -> None:
        return None

    def control(self, ref: str, argv: tuple[str, ...], cwd: str = "") -> tuple[int, str]:
        return 1, "Команды управления инструментом в CI не поддерживаются"

    def list_scripts(self, scripts_dir: str, tool: ToolTemplate) -> Optional[list[str]]:
        return None


_LOCAL_HOST_RE = re.compile(r"^https?://(127\.\d+\.\d+\.\d+|localhost|\[::1\]|0\.0\.0\.0)(:\d+)?(/|$)", re.I)


def _loadlens_url_problems(cfg) -> list[str]:
    if cfg.loadlens_url and _LOCAL_HOST_RE.match(cfg.loadlens_url):
        return [f"адрес LoadLens {cfg.loadlens_url} — локальный: события пайплайна с раннера CI до него не дойдут"]
    return []


def check_ci_generator(generator: Generator, tools: Mapping[str, ToolTemplate]) -> CheckReport:
    """Reaches the CI and the project or job and reads the parameters of the pipeline."""
    from ..ci_params import CiParamError, read_params

    assert generator.ci is not None
    cfg = generator.ci
    executor = CiExecutor(generator, events=lambda run_id, index: [])
    problems = _loadlens_url_problems(cfg)
    details: dict[str, Any] = {"system": "GitLab" if cfg.kind == "gitlab" else "Jenkins"}
    try:
        if cfg.kind == "gitlab":
            if not cfg.api_token:
                # Without a token GitLab still answers (401): that is enough to know the address is right.
                executor.api.request("GET", "/api/v4/version", ok=(200, 401, 403))
                problems.insert(0, "API-токен не задан: LoadLens запустит пайплайн триггером, но не увидит его параметры, "
                                   "статус и журнал и не сможет его отменить")
                return CheckReport(ok=True, message="; ".join(["GitLab отвечает", *problems]), details=details,
                                   connected=True, problems=problems)
            project = executor.api.json("GET", executor._project)
            head = f"проект {project.get('path_with_namespace', cfg.project)}"
        else:
            job = executor.api.json("GET", f"{_jenkins_job_path(cfg.job)}/api/json", params={"tree": "_class,fullName"})
            head = f"задача {job.get('fullName') or cfg.job}"
    except GeneratorError as exc:
        return CheckReport(ok=False, message=str(exc))
    try:
        params = read_params(generator, api=executor.api)
    except (GeneratorError, CiParamError) as exc:
        problems.append(f"параметры пайплайна не читаются: {exc}")
        return CheckReport(ok=False, message="; ".join([head, *problems]), details=details, connected=True, problems=problems)
    problems.extend(params.notes)
    if params.version:
        details["system"] = f"GitLab {params.version}"
    details.update({
        "params": [param.to_public() for param in params.params],
        "param_source": params.source,
        "auto": params.auto,
        "inputs_api": params.inputs_api,
        "ref": params.ref,
    })
    names = ", ".join(param.name for param in params.params[:12]) or "нет"
    message = f"{details['system']}, {head}{f', ветка {params.ref}' if params.ref else ''}; параметры: {names}"
    return CheckReport(ok=True, message="; ".join([message, *problems]), details=details, connected=True, problems=problems)


__all__ = ["CiExecutor", "TOKEN_HEADER", "check_ci_generator", "events_url", "jenkins_path", "pipeline_variables"]
