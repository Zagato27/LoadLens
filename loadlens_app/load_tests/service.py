# SPDX-License-Identifier: Apache-2.0
"""Load test runs for the HTTP layer: validating the launch form, building commands, views, stop and logs."""

from __future__ import annotations

import logging
import shlex
import uuid
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional
from urllib.parse import quote, unquote

from loadlens_app.limits import LimitError, check_load_test_slot, limit

from . import ci_params
from .executors import (
    LOG_TAIL_BYTES,
    GeneratorError,
    LaunchPlan,
    container_name,
    executor_for,
    instance_dir,
    job_name,
    pipeline_variables,
)
from .generators import Generator, configured_generators
from .messages import LtError, Titles, msg, ref_of, stored_text, t
from .store import (
    INSTANCE_STATUS_TITLES,
    INST_ACTIVE,
    RUN_ACTIVE,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_FAILED,
    RUN_LOADING,
    RUN_STATUS_TITLES,
    InstanceRecord,
    RunRecord,
    get_store,
)
from .tools import (
    CommandLine,
    RenderContext,
    control_port,
    exe_dir,
    render_aux_command,
    ToolInputError,
    ToolTemplate,
    check_instance_env,
    effective_tools,
    make_tag,
    parse_env,
    parse_params,
    render_command,
    resolve_script,
)

logger = logging.getLogger(__name__)

TEST_TYPES = ("step", "soak", "spike", "stress")
MAX_NAME_CHARS = 60
MAX_DELAY_SEC = 24 * 3600
TOOL_FAMILIES = ("k6", "jmeter", "gatling", "locust")
LT_DOMAIN = "lt_framework"
CI_TITLES = Titles({"gitlab": "lt.ci.title.gitlab", "jenkins": "lt.ci.title.jenkins"})
MAX_REF_CHARS = 200


class LaunchError(LtError, ValueError):
    """The launch form cannot be accepted; the message is for the user (rendered in the reader's language).

    ``message`` is an ``lt.*`` dictionary key with ``params``, another localized error or a ready text.
    ``instance`` is the number (from 1) of the form instance the error is about: the launch page opens that card
    by this field, not by the text of the message.
    """

    def __init__(self, message: Any, status: int = 400, code: str = "invalid", instance: Optional[int] = None,
                 **params: Any) -> None:
        super().__init__(message, **params)
        self.status = status
        self.code = code
        self.instance = instance


@dataclass(frozen=True)
class Catalog:
    tools: dict[str, ToolTemplate]
    generators: dict[str, Generator]


def _config() -> dict:
    from settings import CONFIG

    return CONFIG


def load_catalog(config: Optional[Mapping[str, Any]] = None) -> Catalog:
    cfg = config if config is not None else _config()
    return Catalog(tools=effective_tools(cfg), generators=configured_generators(cfg))


def _text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    return value.strip() if isinstance(value, str) else ""


def _int(value: Any, default: int, label: Any, low: int, high: int) -> int:
    """``label`` names the field (a dictionary message) in the errors."""
    if value in (None, ""):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise LaunchError("lt.launch.int_expected", field=label) from exc
    if not low <= number <= high:
        raise LaunchError("lt.launch.int_range", field=label, low=low, high=high)
    return number


def _instance_label(index: int) -> Any:
    """"Instance N" of the form (N from 1) as a message parameter."""
    return msg("lt.launch.instance", n=index + 1)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def tool_family(tool_id: str) -> str:
    head = tool_id.split("_")[0].split("-")[0]
    return head if head in TOOL_FAMILIES else ""


def instance_ref(generator: Generator, run_id: str, inst: InstanceRecord) -> tuple[str, str]:
    """(launch reference, value of ``{instance_dir}``) for the generator type."""
    if generator.type == "ssh":
        ref = instance_dir(generator, run_id, inst.index, inst.tag)
        return ref, ref
    if generator.type == "docker":
        return container_name(run_id, inst.index, inst.tag), generator.work_dir or "/tmp"
    if generator.type == "kubernetes":
        return job_name(run_id, inst.index, inst.tag), generator.work_dir or "/tmp"
    if generator.is_ci:
        return f"{run_id}:{inst.index}", "."
    return f"manual:{run_id}:{inst.index}", ""


def ci_launch_plan(generator: Generator, ref: str, run_id: str, run_name: str, run_tag: str, inst: InstanceRecord,
                   max_seconds: int, ci_token: str = "") -> LaunchPlan:
    """The pipeline run of a CI instance: branch, parameter values checked against the pipeline, LOADLENS_*."""
    assert generator.ci is not None
    params = ci_params.cached_params(generator, inst.ci_ref)
    values, extra = ci_params.check_values(params, inst.ci_params, inst.ci_extra, generator.ci.kind)
    auto = pipeline_variables(generator, run_id, run_name, run_tag, inst.name, inst.index, max_seconds, ci_token)
    split = ci_params.launch_values(params, values, extra, generator.ci.kind, auto)
    return LaunchPlan(
        ref=ref,
        command=CommandLine(argv=(), env={}, cwd=""),
        max_seconds=int(max_seconds),
        run_id=run_id,
        run_name=run_name,
        run_tag=run_tag,
        instance_index=inst.index,
        instance_name=inst.name,
        instance_tag=inst.tag,
        ci_token=ci_token,
        ci_ref=params.ref or inst.ci_ref,
        ci_variables=split.get("variables", {}),
        ci_inputs=split.get("inputs", {}),
        ci_parameters=split.get("parameters", {}),
        ci_auto=split.get("auto", {}),
        ci_parameterized=params.parameterized,
    )


def instance_plan(
    generator: Generator,
    tool: Optional[ToolTemplate],
    run_id: str,
    run_name: str,
    run_tag: str,
    inst: InstanceRecord,
    max_seconds: int = 3600,
    ci_token: str = "",
) -> LaunchPlan:
    """Everything needed to start the instance; raises :class:`ToolInputError` for bad input.

    CI instances read the pipeline parameters (cached) and may raise :class:`GeneratorError` as well.
    """
    ref, idir = instance_ref(generator, run_id, inst)
    if generator.is_ci:
        return ci_launch_plan(generator, ref, run_id, run_name, run_tag, inst, max_seconds, ci_token)
    if tool is None:
        raise ToolInputError("lt.launch.choose_tool")
    scripts_dir = generator.scripts_dir_for(tool.id)
    script, script_dir = resolve_script(tool, inst.script, scripts_dir, allow_relative=generator.is_ci)
    exe = generator.executable(tool)
    ctx = RenderContext(
        exe=exe,
        exe_dir=exe_dir(exe),
        control_port=str(control_port(run_id, inst.index)),
        script=script,
        script_dir=script_dir,
        scripts_dir=scripts_dir,
        run_name=run_name,
        run_tag=run_tag,
        instance=inst.name,
        instance_tag=inst.tag,
        instance_dir=idir,
    )
    # The generator's variables win: runs created before the form checked for overrides keep them too.
    env = {**inst.env, **generator.env}
    command = render_command(tool, ctx, parse_params(inst.params), env)
    image = generator.image_for(tool) if generator.is_container else ""
    if generator.is_container and not image:
        raise ToolInputError("lt.launch.no_generator_image", tool=tool.display_title, generator=generator.title)
    return LaunchPlan(
        ref=ref,
        command=command,
        max_seconds=int(max_seconds),
        run_id=run_id,
        run_name=run_name,
        run_tag=run_tag,
        instance_index=inst.index,
        instance_name=inst.name,
        instance_tag=inst.tag,
        tool_id=tool.id,
        image=image,
        grace_seconds=tool.grace_seconds,
        stop_command=render_aux_command(tool.stop_command, ctx),
        status_command=render_aux_command(tool.status_command, ctx),
        ci_token=ci_token,
    )


def instance_command(
    generator: Generator,
    tool: ToolTemplate,
    run_id: str,
    run_name: str,
    run_tag: str,
    inst: InstanceRecord,
) -> tuple[CommandLine, str]:
    """Command and launch reference of an instance (the launch form and older callers)."""
    plan = instance_plan(generator, tool, run_id, run_name, run_tag, inst)
    return plan.command, plan.ref


def _busy_generators(runs: list[RunRecord], exclude_run: str = "") -> dict[str, list[RunRecord]]:
    """Generator id -> runs with instances still holding it."""
    busy: dict[str, list[RunRecord]] = {}
    for run in runs:
        if run.run_id == exclude_run:
            continue
        for inst in run.instances:
            if inst.status in INST_ACTIVE:
                entries = busy.setdefault(inst.generator, [])
                if run not in entries:
                    entries.append(run)
    return busy


def _active_runs() -> list[RunRecord]:
    return get_store().list(limit=1000, statuses=RUN_ACTIVE)


def _installed_lt_packs(area: str, service: str) -> list[str]:
    try:
        from update_page import _effective_config_for_scope

        cfg = _effective_config_for_scope(area, service)
    except Exception:
        return []
    block = ((cfg.get("queries") or {}).get(LT_DOMAIN)) if isinstance(cfg.get("queries"), dict) else None
    packs = block.get("packs") if isinstance(block, dict) else None
    return [str(entry.get("id")) for entry in packs or [] if isinstance(entry, dict) and entry.get("id")]


def _pack_stacks(pack_ids: list[str]) -> set[str]:
    from AI.query_packs import load_catalog as load_packs

    stacks: set[str] = set()
    catalog = load_packs()
    for pack_id in pack_ids:
        pack = catalog.pack(pack_id)
        if pack is not None:
            stacks.update(pack.stack)
    return stacks


def _plan_ci_instance(raw: Mapping[str, Any], label: Any, index: int, name: str, tag: str, generator: Generator,
                      run_id: str, run_name: str, run_tag: str) -> InstanceRecord:
    """A CI instance of the launch form: branch and parameter values checked against the pipeline."""
    assert generator.ci is not None
    ci_ref = _text(raw, "ci_ref")
    if len(ci_ref) > MAX_REF_CHARS or any(ch.isspace() for ch in ci_ref):
        raise LaunchError("lt.launch.bad_ref", label=label, ref=ci_ref[:40])
    values = raw.get("ci_params") or {}
    if not isinstance(values, Mapping):
        raise LaunchError("lt.launch.ci_params_object", label=label)
    try:
        extra = parse_env(raw.get("ci_extra")) if generator.ci.kind == "gitlab" else {}
        if raw.get("ci_extra") and generator.ci.kind != "gitlab":
            raise ci_params.CiParamError("lt.ci.jenkins_declared_only")
        params = ci_params.cached_params(generator, ci_ref)
        checked, extra = ci_params.check_values(params, values, extra, generator.ci.kind)
    except ToolInputError as exc:
        raise LaunchError("lt.launch.instance_error", label=label, error=exc) from exc
    except GeneratorError as exc:
        raise LaunchError("lt.launch.instance_ci_params_unreadable", status=502, code="generator_unreachable",
                          label=label, error=exc) from exc
    inst = InstanceRecord(
        index=index,
        name=name,
        tag=tag,
        generator=generator.id,
        start_delay_sec=_int(raw.get("start_delay_sec"), 0, msg("lt.launch.field.start_delay", label=label), 0, MAX_DELAY_SEC),
        ci_kind=generator.ci.kind,
        ci_ref=params.ref or ci_ref,
        ci_params=checked,
        ci_extra=extra,
        ci_secret=[param.name for param in params.params if param.secret and param.name in checked],
        ci_tagged=generator.ci.tags_metrics,
    )
    try:
        launch = instance_plan(generator, None, run_id, run_name, run_tag, inst)
    except ToolInputError as exc:
        raise LaunchError("lt.launch.instance_error", label=label, error=exc) from exc
    except GeneratorError as exc:
        raise LaunchError("lt.launch.instance_error", status=502, code="generator_unreachable", label=label, error=exc) from exc
    inst.ref = launch.ref
    inst.command = ci_params.describe(generator.ci.kind, generator.ci.project or generator.ci.job, inst.ci_ref,
                                      inst.ci_params, inst.ci_extra, inst.ci_secret)
    return inst


@dataclass
class PlannedRun:
    run: RunRecord
    commands: list[dict[str, Any]]
    warnings: list[str]


def plan_run(data: Mapping[str, Any], owner: str, catalog: Optional[Catalog] = None) -> PlannedRun:
    """Validates the launch form and builds the run without storing it; raises :class:`LaunchError`."""
    from loadlens_app.blueprints import dashboard
    from loadlens_app.core import run_name_problem
    from update_page import default_run_name

    if not isinstance(data, Mapping):
        raise LaunchError("lt.launch.json_object_expected")
    cat = catalog or load_catalog()
    service = _text(data, "service")
    if not service:
        raise LaunchError("lt.launch.choose_service")
    try:
        area = dashboard.resolve_report_target(service, _text(data, "project_area"))
    except dashboard.ReportRequestError as exc:
        raise LaunchError(exc) from exc
    test_type = _text(data, "test_type")
    if test_type and test_type not in TEST_TYPES:
        raise LaunchError("lt.launch.test_type_one_of", types=", ".join(TEST_TYPES))

    run_name = _text(data, "run_name")
    if run_name:
        problem = run_name_problem(run_name)
        if problem:
            raise LaunchError(problem)
        try:
            taken = dashboard._run_name_taken(run_name, service) or get_store().name_in_use(run_name, service)
        except Exception:
            taken = False
        if taken:
            raise LaunchError("lt.launch.run_name_taken", run=run_name, service=service)
    else:
        run_name = default_run_name()

    raw_instances = data.get("instances")
    max_instances = limit("max_load_test_instances") or 20
    if not isinstance(raw_instances, list) or not raw_instances:
        raise LaunchError("lt.launch.add_instance")
    if len(raw_instances) > max_instances:
        raise LaunchError("lt.launch.max_instances", max=max_instances)

    max_hours = limit("max_load_test_hours") or 24
    max_minutes = _int(data.get("max_duration_min"), max_hours * 60, msg("lt.launch.field.max_duration"), 1, max_hours * 60)
    options = {
        "auto_report": _bool(data.get("auto_report"), True),
        "use_llm": _bool(data.get("use_llm"), True),
        "max_seconds": max_minutes * 60,
    }

    run_id = uuid.uuid4().hex
    run_tag = make_tag(run_name, f"run-{run_id[:8]}")
    instances: list[InstanceRecord] = []
    commands: list[dict[str, Any]] = []
    # Shown with this answer only: rendered now, in the language of the request.
    warnings: list[str] = []
    used_tags: set[str] = set()
    for index, raw in enumerate(raw_instances):
        try:
            label = _instance_label(index)
            if not isinstance(raw, Mapping):
                raise LaunchError("lt.launch.instance_object", label=label)
            name = (_text(raw, "name") or f"i{index + 1}")[:MAX_NAME_CHARS]
            tag = make_tag(name, f"i{index + 1}")
            while tag in used_tags:
                tag = f"{tag}-{index + 1}"
            used_tags.add(tag)
            generator = cat.generators.get(_text(raw, "generator"))
            if generator is None:
                raise LaunchError("lt.launch.choose_generator", label=label)
            if generator.is_ci:
                inst = _plan_ci_instance(raw, label, index, name, tag, generator, run_id, run_name, run_tag)
                instances.append(inst)
                commands.append({
                    "index": index,
                    "name": name,
                    "generator": generator.id,
                    "generator_title": generator.title,
                    "tool": "",
                    "command": inst.command,
                    "cwd": "",
                    "manual": False,
                    "generator_type": generator.type,
                    "image": "",
                    "ci": {"ref": inst.ci_ref, "kind": inst.ci_kind},
                })
                continue
            tool_id = _text(raw, "tool")
            tool = cat.tools.get(tool_id) if tool_id else None
            if not generator.is_manual:
                if tool is None:
                    raise LaunchError("lt.launch.instance_choose_tool", label=label)
                if not generator.offers(tool.id, tool):
                    if generator.is_container and not generator.image_for(tool):
                        raise LaunchError("lt.launch.generator_no_image", label=label, generator=generator.title,
                                          tool=tool.display_title)
                    raise LaunchError("lt.launch.generator_no_tool", label=label, generator=generator.title,
                                      tool=tool.display_title)
            try:
                # A manual instance runs nothing through LoadLens: its environment would only be stored.
                env = {} if generator.is_manual else parse_env(raw.get("env"))
                check_instance_env(env, generator.env)
                params = raw.get("params")
                params_text = params if isinstance(params, str) else shlex.join(str(p) for p in params) if isinstance(params, list) else ""
                parse_params(params_text)
            except ToolInputError as exc:
                raise LaunchError("lt.launch.instance_error", label=label, error=exc) from exc
            inst = InstanceRecord(
                index=index,
                name=name,
                tag=tag,
                generator=generator.id,
                tool=tool.id if tool else "",
                script=_text(raw, "script"),
                params=params_text.strip(),
                env=env,
                start_delay_sec=_int(raw.get("start_delay_sec"), 0, msg("lt.launch.field.start_delay", label=label), 0,
                                     MAX_DELAY_SEC),
            )
            if generator.is_manual:
                inst.ref = f"manual:{run_id}:{index}"
            else:
                try:
                    launch = instance_plan(generator, tool, run_id, run_name, run_tag, inst)
                except ToolInputError as exc:
                    raise LaunchError("lt.launch.instance_error", label=label, error=exc) from exc
                command = launch.command
                inst.ref = launch.ref
                inst.command = command.shell(mask_env=True)
            instances.append(inst)
            commands.append({
                "index": index,
                "name": name,
                "generator": generator.id,
                "generator_title": generator.title,
                "tool": inst.tool,
                "command": inst.command,
                "cwd": "" if generator.is_manual else command.cwd,
                "manual": generator.is_manual,
                "generator_type": generator.type,
                "image": "" if generator.is_manual or not generator.is_container else launch.image,
            })
        except LaunchError as exc:
            if exc.instance is None:
                exc.instance = index + 1
            raise

    active = _active_runs()
    total = len(active)
    mine = sum(1 for run in active if run.owner == owner)
    try:
        check_load_test_slot(total, mine)
    except LimitError as exc:
        raise LaunchError(exc, status=429, code="limit_exceeded") from exc

    busy = _busy_generators(active)
    for generator_id in dict.fromkeys(inst.generator for inst in instances):
        generator = cat.generators[generator_id]
        new_here = sum(1 for inst in instances if inst.generator == generator_id)
        running_here = sum(
            1 for run in busy.get(generator_id, []) for inst in run.instances
            if inst.generator == generator_id and inst.status in INST_ACTIVE
        )
        if generator.max_parallel and running_here + new_here > generator.max_parallel:
            raise LaunchError(
                "lt.launch.generator_busy",
                status=429,
                code="generator_busy",
                generator=generator.title,
                max=generator.max_parallel,
                running=running_here,
            )
        if busy.get(generator_id):
            names = ", ".join(t("lt.launch.busy_run", run=run.run_name, owner=run.owner or t("lt.launch.no_owner"))
                              for run in busy[generator_id][:3])
            warnings.append(t("lt.launch.warn_generator_shared", generator=generator.title, runs=names))

    installed = _installed_lt_packs(area, service)
    stacks = _pack_stacks(installed)
    filtered_tools = [inst for inst in instances if inst.tool]
    for tool_id in dict.fromkeys(inst.tool for inst in filtered_tools):
        family = tool_family(tool_id)
        if family and family not in stacks:
            warnings.append(t("lt.launch.warn_no_pack", tool=cat.tools[tool_id].display_title))
    if len(instances) > 1:
        untagged = [cat.tools[i].display_title for i in dict.fromkeys(inst.tool for inst in filtered_tools)
                    if not set(cat.tools[i].run_filter_packs) & set(installed)]
        untagged += [t("lt.launch.untagged_pipeline", generator=cat.generators[inst.generator].title) for inst in instances
                     if inst.ci_kind and not inst.ci_tagged]
        if untagged:
            warnings.append(t("lt.launch.warn_untagged", tools=", ".join(untagged)))

    # Stored: readers in other languages render the ref.
    waiting = msg("lt.launch.waiting_launch")
    run = RunRecord(
        run_id=run_id,
        run_name=run_name,
        run_tag=run_tag,
        service=service,
        project_area=area,
        test_type=test_type,
        owner=owner,
        options=options,
        instances=instances,
        message=str(waiting),
        message_ref=ref_of(waiting),
    )
    return PlannedRun(run=run, commands=commands, warnings=warnings)


def create_run(data: Mapping[str, Any], owner: str) -> PlannedRun:
    planned = plan_run(data, owner)
    get_store().create(planned.run)
    from .watcher import ensure_watcher

    ensure_watcher()
    return planned


def request_stop(run_id: str, index: Optional[int] = None) -> RunRecord:
    store = get_store()
    run = store.get(run_id)
    if run is None:
        raise LaunchError("lt.launch.run_not_found", status=404, code="not_found")
    if index is not None and not any(inst.index == index for inst in run.instances):
        raise LaunchError("lt.launch.instance_not_found", status=404, code="not_found")
    if not run.active:
        raise LaunchError("lt.launch.run_finished", status=409, code="finished")
    store.request_stop(run_id, index)
    from .watcher import ensure_watcher

    ensure_watcher()
    return store.get(run_id) or run


def _service_area(service: str, project_area: str = "") -> str:
    from loadlens_app.blueprints import dashboard

    try:
        return dashboard.resolve_report_target(service, project_area)
    except dashboard.ReportRequestError as exc:
        raise LaunchError(exc) from exc


def output_hints(service: str, project_area: str, tool_id: str) -> dict[str, Any]:
    """Flags and variables that point the tool's metrics at the source of the service's load domain."""
    from update_page import _effective_config_for_scope

    from .hints import output_hint

    if not service:
        raise LaunchError("lt.launch.choose_service")
    area = _service_area(service, project_area)
    if tool_id not in load_catalog().tools:
        raise LaunchError("lt.launch.tool_not_found", status=404, code="not_found")
    cfg = _effective_config_for_scope(area, service)
    return output_hint(cfg, tool_id, _installed_lt_packs(area, service)).to_dict()


def list_presets(services: Optional[list[str]] = None) -> list[dict[str, Any]]:
    from .presets import get_presets

    return [preset.to_view() for preset in get_presets().list(services)]


def save_preset(data: Mapping[str, Any], owner: str, is_admin: bool = False) -> dict[str, Any]:
    from .presets import PresetError, save_preset as store_preset

    if not isinstance(data, Mapping):
        raise LaunchError("lt.launch.json_object_expected")
    service = _text(data, "service")
    if not service:
        raise LaunchError("lt.launch.choose_service")
    area = _service_area(service, _text(data, "project_area"))
    try:
        return store_preset(_text(data, "name"), service, area, data.get("spec") or {}, owner, is_admin=is_admin).to_view()
    except PresetError as exc:
        raise LaunchError(exc, status=exc.status) from exc


def delete_preset(preset_id: str, owner: str, is_admin: bool) -> None:
    from .presets import get_presets

    store = get_presets()
    preset = store.get(preset_id)
    if preset is None:
        raise LaunchError("lt.launch.preset_not_found", status=404, code="not_found")
    if preset.owner and preset.owner != owner and not is_admin:
        raise LaunchError("lt.launch.preset_delete_forbidden", status=403, code="forbidden")
    store.delete(preset_id)


CI_EVENTS = ("started", "finished")
CI_MESSAGE_CHARS = 500


def ci_instance(run_id: str, index: int, token: str) -> tuple[RunRecord, InstanceRecord, Generator]:
    """The instance a CI pipeline reports about, after checking its per-instance token.

    Unknown runs, instances without a token and wrong tokens all answer 404, so the endpoint does not
    reveal which runs exist; a finished instance answers 410.
    """
    import hashlib
    import hmac

    run = get_store().get(run_id)
    inst = next((item for item in run.instances if item.index == index), None) if run else None
    if run is None or inst is None or not inst.ci_token_hash or not token:
        raise LaunchError("lt.launch.instance_not_found", status=404, code="not_found")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(digest, inst.ci_token_hash):
        raise LaunchError("lt.launch.instance_not_found", status=404, code="not_found")
    if inst.status not in INST_ACTIVE:
        raise LaunchError("lt.launch.instance_finished", status=410, code="finished")
    generator = load_catalog().generators.get(inst.generator)
    if generator is None or not generator.is_ci:
        raise LaunchError("lt.launch.instance_generator_removed", status=409, code="config_changed")
    return run, inst, generator


def _ci_generator(generator_id: str) -> Generator:
    generator = load_catalog().generators.get(generator_id)
    if generator is None or not generator.is_ci:
        raise LaunchError("lt.launch.ci_generator_not_found", status=404, code="not_found")
    return generator


def ci_params_view(generator_id: str, ref: str = "") -> dict[str, Any]:
    """Parameters of the pipeline for the launch form (cached for a minute)."""
    generator = _ci_generator(generator_id)
    ref = (ref or "").strip()
    if len(ref) > MAX_REF_CHARS or any(ch.isspace() for ch in ref):
        raise LaunchError("lt.launch.bad_branch")
    try:
        return ci_params.cached_params(generator, ref).to_public()
    except (GeneratorError, ci_params.CiParamError) as exc:
        raise LaunchError("lt.ci.params_unreadable", status=502, code="generator_unreachable", error=exc) from exc


def ci_refs_view(generator_id: str, search: str = "") -> dict[str, Any]:
    """Branches of the GitLab project or the Jenkins multibranch job."""
    generator = _ci_generator(generator_id)
    try:
        refs, default = ci_params.read_refs(generator, (search or "").strip()[:100])
    except GeneratorError as exc:
        raise LaunchError("lt.launch.refs_unreadable", status=502, code="generator_unreachable", error=exc) from exc
    return {"refs": refs, "default": default}


def record_ci_event(run: RunRecord, inst: InstanceRecord, payload: Any) -> dict[str, Any]:
    from .store import utcnow

    if not isinstance(payload, Mapping) or payload.get("event") not in CI_EVENTS:
        raise LaunchError("lt.launch.event_one_of", events=", ".join(CI_EVENTS))
    code = payload.get("exit_code")
    if code is not None and (isinstance(code, bool) or not isinstance(code, int) or not -255 <= code <= 255):
        raise LaunchError("lt.launch.exit_code_int")
    event = {
        "index": inst.index,
        "event": payload["event"],
        "exit_code": code,
        "message": str(payload.get("message") or "")[:CI_MESSAGE_CHARS],
        "at": utcnow().isoformat(),
    }
    if not get_store().record_event(run.run_id, event):
        raise LaunchError("lt.launch.events_closed", status=410, code="finished")
    from .watcher import ensure_watcher

    ensure_watcher()
    return event


def instance_logs(run: RunRecord, index: int, max_bytes: int = LOG_TAIL_BYTES) -> str:
    inst = next((item for item in run.instances if item.index == index), None)
    if inst is None:
        raise LaunchError("lt.launch.instance_not_found", status=404, code="not_found")
    if not inst.ref or inst.launched_at is None:
        return ""
    generator = load_catalog().generators.get(inst.generator)
    if generator is None or generator.is_manual:
        return inst.log_tail
    try:
        text = executor_for(generator).logs(inst.ref, max_bytes)
    except GeneratorError as exc:
        if inst.log_tail:
            return inst.log_tail
        raise LaunchError(exc, status=502, code="generator_unreachable") from exc
    # A removed container or an expired job: the tail saved when the instance stopped.
    return text or inst.log_tail


def _iso(value) -> Optional[str]:
    return value.isoformat() if value else None


def _tool_title(inst: InstanceRecord, cat: Catalog) -> str:
    tool = cat.tools.get(inst.tool) if inst.tool else None
    if tool:
        return tool.display_title
    return CI_TITLES.get(inst.ci_kind, inst.tool) if inst.ci_kind else inst.tool


def instance_view(run: RunRecord, inst: InstanceRecord, cat: Catalog) -> dict[str, Any]:
    generator = cat.generators.get(inst.generator)
    ci = None
    if inst.ci_kind:
        ci = {
            "kind": inst.ci_kind,
            "ref": inst.ci_ref,
            "params": {name: ("***" if name in inst.ci_secret else value) for name, value in inst.ci_params.items()},
            "extra": dict(inst.ci_extra),
            "secret": list(inst.ci_secret),
            "tagged": inst.ci_tagged,
        }
    return {
        "index": inst.index,
        "name": inst.name,
        "tag": inst.tag,
        "generator": inst.generator,
        "generator_title": generator.title if generator else inst.generator,
        "manual": bool(generator and generator.is_manual),
        "tool": inst.tool,
        "tool_title": _tool_title(inst, cat),
        "script": inst.script,
        "params": inst.params,
        "env_keys": sorted(inst.env),
        "start_delay_sec": inst.start_delay_sec,
        "command": inst.command,
        "status": inst.status,
        "status_title": INSTANCE_STATUS_TITLES.get(inst.status, inst.status),
        "active": inst.status in INST_ACTIVE,
        "exit_code": inst.exit_code,
        # Written by the watcher: the ref is rendered in the reader's language, the text is the fallback.
        "message": stored_text(inst.message, inst.message_ref),
        "launched_at": _iso(inst.launched_at),
        "started_at": _iso(inst.started_at),
        "finished_at": _iso(inst.finished_at),
        "stop_requested": run.stop_requested(inst.index),
        "has_logs": bool(inst.launched_at and ((generator and not generator.is_manual) or inst.log_tail)),
        "generator_type": generator.type if generator else "",
        "status_text": stored_text(inst.status_text, inst.status_text_ref),
        "ci": ci,
        "external_url": inst.external_url,
    }


def run_view(run: RunRecord, cat: Optional[Catalog] = None, with_report: bool = True) -> dict[str, Any]:
    cat = cat or load_catalog()
    report: Optional[dict[str, Any]] = None
    if with_report and run.report_job_id:
        from loadlens_app.jobs import job_store

        job = job_store.get(run.report_job_id)
        if job is not None:
            report = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": int(job.progress),
                # The job store may keep a ref of the stage (rendered in the reader's language).
                "message": stored_text(job.message, getattr(job, "message_ref", None)),
                "page_url": job.page_url,
                "run_name": job.run_name,
            }
    return {
        "run_id": run.run_id,
        "run_name": run.run_name,
        "run_tag": run.run_tag,
        "service": run.service,
        "project_area": run.project_area,
        "test_type": run.test_type,
        "status": run.status,
        "status_title": RUN_STATUS_TITLES.get(run.status, run.status),
        "active": run.active,
        "loading": run.status in RUN_LOADING,
        "message": stored_text(run.message, run.message_ref),
        "owner": run.owner,
        "options": {
            "auto_report": bool(run.options.get("auto_report", True)),
            "use_llm": bool(run.options.get("use_llm", True)),
            "max_seconds": int(run.options.get("max_seconds") or 0),
        },
        "report_job_id": run.report_job_id,
        "report_url": run.report_url or (report or {}).get("page_url"),
        "report": report,
        "stop_requested": run.stop_all_at is not None,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "created_at": _iso(run.created_at),
        "updated_at": _iso(run.updated_at),
        "instances": [instance_view(run, inst, cat) for inst in run.instances],
    }


# ---- run history of the archive ----------------------------------------------------------------

HISTORY_MAX_LIMIT = 100
DELETE_MAX_RUNS = 100
# Status filter of the archive tab: request codes -> run statuses.
HISTORY_STATUS_GROUPS: dict[str, tuple[str, ...]] = {
    "active": RUN_ACTIVE,
    RUN_DONE: (RUN_DONE,),
    RUN_FAILED: (RUN_FAILED,),
    RUN_CANCELLED: (RUN_CANCELLED,),
}
_REPORT_PATH = "/reports/"
_SKIP_KEYS = {
    "not_found": "lt.history.skip.not_found",
    "active": "lt.history.skip.active",
    "forbidden": "lt.history.skip.forbidden",
}


def history_statuses(raw: Any) -> Optional[tuple[str, ...]]:
    """Run statuses of the ``status=`` filter (codes joined by commas); ``None`` when there is no filter."""
    codes = [code.strip() for code in str(raw or "").split(",") if code.strip()]
    if not codes:
        return None
    unknown = [code for code in codes if code not in HISTORY_STATUS_GROUPS]
    if unknown:
        raise LaunchError("lt.history.bad_status", status=400, code="bad_request",
                          value=unknown[0][:40], allowed=", ".join(HISTORY_STATUS_GROUPS))
    statuses: list[str] = []
    for code in codes:
        statuses.extend(HISTORY_STATUS_GROUPS[code])
    return tuple(dict.fromkeys(statuses))


def report_id_of(run: RunRecord) -> str:
    """Id of the run's report from its address ``/reports/<id>``; '' when there is none or it is elsewhere.

    The id survives renaming the report, so a run finds its report even under a new name, and two runs that
    share a name never share a report.
    """
    url = str(run.report_url or "")
    if not url.startswith(_REPORT_PATH):
        return ""
    token = url[len(_REPORT_PATH):].split("?", 1)[0].split("#", 1)[0]
    return unquote(token) if token and "/" not in token else ""


def _report_refs(report_ids: list[str]) -> Optional[dict[str, dict[str, str]]]:
    """Existing reports among ``report_ids`` (id -> ``{run_id, run_name, service}``); ``None`` when unknown."""
    if not report_ids:
        return {}
    from loadlens_app.blueprints import dashboard

    try:
        return dashboard._lookup_report_refs(report_ids)
    except Exception as exc:
        logger.warning("Could not look up the reports of load test runs: %s", exc)
        return None


def can_delete(run: RunRecord, user: str, can_write: bool, is_admin: bool) -> bool:
    """Engineers delete their own finished runs, an administrator anyone's."""
    return can_write and not run.active and (is_admin or not run.owner or run.owner == user)


def history_view(run: RunRecord, cat: Catalog, reports: Optional[dict[str, dict[str, str]]], user: str,
                 can_write: bool, is_admin: bool) -> dict[str, Any]:
    """A run for the archive tab: who, what and when, without commands, scripts, parameters or CI values.

    Viewers read this list, so nothing here may name a generator host or carry a test credential.
    """
    tools: list[str] = []
    for inst in run.instances:
        title = _tool_title(inst, cat)
        if title and title not in tools:
            tools.append(title)
    report: Optional[dict[str, str]] = None
    has_report: Optional[bool] = False
    report_id = report_id_of(run)
    if report_id and reports is not None:
        ref = reports.get(report_id)
        has_report = ref is not None
        if ref is not None:
            report = {"url": _REPORT_PATH + quote(report_id, safe=""), "run_name": str(ref.get("run_name") or "")}
    elif run.report_url:
        # The report database did not answer, or the report lives elsewhere: the stored address, unchecked.
        has_report = None
        report = {"url": str(run.report_url), "run_name": ""}
    return {
        "run_id": run.run_id,
        "run_name": run.run_name,
        "service": run.service,
        "test_type": run.test_type,
        "status": run.status,
        "status_title": RUN_STATUS_TITLES.get(run.status, run.status),
        "active": run.active,
        "owner": run.owner,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "instances": len(run.instances),
        "tools": tools,
        "report": report,
        "has_report": has_report,
        "can_delete": can_delete(run, user, can_write, is_admin),
    }


def history_page(services: Optional[list[str]], statuses: Optional[tuple[str, ...]], query: str, offset: int,
                 limit: int, user: str, can_write: bool, is_admin: bool) -> dict[str, Any]:
    """One page of the archive tab, newest first, and how many runs match the filters."""
    store = get_store()
    limit = max(1, min(int(limit), HISTORY_MAX_LIMIT))
    runs = store.list(limit=limit, statuses=statuses, services=services or None, offset=max(0, int(offset)), query=query)
    total = store.count(statuses=statuses, services=services or None, query=query)
    if not runs:
        return {"runs": [], "total": total}
    reports = _report_refs([rid for rid in dict.fromkeys(report_id_of(run) for run in runs) if rid])
    cat = load_catalog()
    return {"runs": [history_view(run, cat, reports, user, can_write, is_admin) for run in runs], "total": total}


def _skipped(run_id: str, code: str, run: Optional[RunRecord] = None) -> dict[str, str]:
    name = run.run_name if run is not None else ""
    return {"run_id": run_id, "run_name": name, "code": code, "message": t(_SKIP_KEYS[code], run=name or run_id)}


def _delete_reports(runs: list[RunRecord]) -> tuple[list[str], list[dict[str, str]]]:
    """Deletes the reports of deleted runs, found by report id under their current name."""
    from loadlens_app.core import _delete_run_data

    ids = {run.run_id: report_id_of(run) for run in runs}
    refs = _report_refs([rid for rid in dict.fromkeys(ids.values()) if rid])
    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    for run in runs:
        report_id = ids[run.run_id]
        if not report_id:
            continue
        if refs is None:
            failed.append({"run_id": run.run_id, "run_name": "", "message": t("lt.history.report_lookup_failed")})
            continue
        ref = refs.get(report_id)
        if ref is None or ref.get("run_name") in deleted:
            continue  # already gone: nothing to delete
        name = str(ref.get("run_name") or "")
        try:
            _delete_run_data(name)
        except Exception as exc:
            logger.warning("Could not delete report '%s' of load test run %s: %s", name, run.run_id, exc)
            failed.append({"run_id": run.run_id, "run_name": name, "message": t("lt.history.report_delete_failed", run=name)})
            continue
        deleted.append(name)
    return deleted, failed


def delete_runs(run_ids: Any, owner: str, is_admin: bool, with_reports: bool = False) -> dict[str, Any]:
    """Deletes finished runs; the rest are skipped with a reason. ``with_reports`` (administrators) removes their reports too.

    A run and its report are separate: without ``with_reports`` the report stays in the archive.
    """
    if not isinstance(run_ids, list) or not all(isinstance(i, str) and i.strip() for i in run_ids) or not run_ids:
        raise LaunchError("lt.history.ids_required", status=400, code="bad_request")
    ids = list(dict.fromkeys(i.strip() for i in run_ids))
    if len(ids) > DELETE_MAX_RUNS:
        raise LaunchError("lt.history.too_many", status=400, code="bad_request", max=DELETE_MAX_RUNS)
    if with_reports and not is_admin:
        raise LaunchError("lt.history.reports_forbidden", status=403, code="forbidden")
    store = get_store()
    found = {run.run_id: run for run in store.get_many(ids)}
    allowed: list[RunRecord] = []
    skipped: list[dict[str, str]] = []
    for run_id in ids:
        run = found.get(run_id)
        if run is None:
            skipped.append(_skipped(run_id, "not_found"))
        elif run.active:
            skipped.append(_skipped(run_id, "active", run))
        elif not can_delete(run, owner, True, is_admin):
            skipped.append(_skipped(run_id, "forbidden", run))
        else:
            allowed.append(run)
    gone = set(store.delete([run.run_id for run in allowed])) if allowed else set()
    deleted = [run for run in allowed if run.run_id in gone]
    # Deleted by someone else meanwhile.
    skipped += [_skipped(run.run_id, "not_found", run) for run in allowed if run.run_id not in gone]
    reports_deleted, reports_failed = _delete_reports(deleted) if with_reports and deleted else ([], [])
    return {
        "deleted": [{"run_id": run.run_id, "run_name": run.run_name, "service": run.service} for run in deleted],
        "skipped": skipped,
        "reports_deleted": reports_deleted,
        "reports_failed": reports_failed,
    }


ACTIVE_MAX_LIMIT = 20


def active_runs(services: Optional[list[str]], limit: int, user: str, is_admin: bool, with_live: bool) -> dict[str, Any]:
    """Runs going on now for the dashboard: the newest ``limit`` of them in brief and how many there are.

    ``with_live`` adds RPS and errors of the last 15 minutes of the runs that generate load. A user may
    stop his own runs from the dashboard (an admin any run); the run page keeps the usual rule.
    """
    from .live import live_brief

    runs = get_store().list(limit=200, statuses=RUN_ACTIVE, services=services or None)
    limit = max(0, min(int(limit), ACTIVE_MAX_LIMIT))
    tools = load_catalog().tools if runs and limit and with_live else {}
    out: list[dict[str, Any]] = []
    for run in runs[:limit]:
        loading = run.status in RUN_LOADING
        view: dict[str, Any] = {
            "run_id": run.run_id,
            "run_name": run.run_name,
            "service": run.service,
            "status": run.status,
            "status_title": RUN_STATUS_TITLES.get(run.status, run.status),
            "message": stored_text(run.message, run.message_ref),
            "owner": run.owner,
            "loading": loading,
            "created_at": _iso(run.created_at),
            "started_at": _iso(run.started_at),
            "max_seconds": int(run.options.get("max_seconds") or 0),
            "instances_total": len(run.instances),
            "instances_running": sum(1 for inst in run.instances if inst.status == "running"),
            "can_stop": loading and run.stop_all_at is None and (is_admin or (bool(user) and run.owner == user)),
            "url": f"/launch/{run.run_id}",
            "live": None,
        }
        if with_live and loading and any(inst.started_at for inst in run.instances):
            try:
                view["live"] = live_brief(run, tools)
            except Exception as exc:  # the dashboard shows the run without numbers
                view["live"] = {"ok": False, "message": str(exc)[:200], "rps": None, "errors": None, "points": []}
        out.append(view)
    return {"total": len(runs), "runs": out}


def launch_options() -> dict[str, Any]:
    cat = load_catalog()
    try:
        busy = _busy_generators(_active_runs())
    except Exception:
        busy = {}
    generators = []
    for generator in cat.generators.values():
        item = generator.to_public(cat.tools)
        item["busy_runs"] = [run.run_name for run in busy.get(generator.id, [])]
        generators.append(item)
    max_hours = limit("max_load_test_hours") or 24
    return {
        "tools": [tool.to_public() for tool in cat.tools.values()],
        "generators": generators,
        "test_types": list(TEST_TYPES),
        "limits": {
            "max_instances": limit("max_load_test_instances") or 20,
            "max_duration_min": max_hours * 60,
        },
    }


def list_scripts(generator_id: str, tool_id: str) -> dict[str, Any]:
    cat = load_catalog()
    generator = cat.generators.get(generator_id)
    tool = cat.tools.get(tool_id)
    if generator is None or tool is None:
        raise LaunchError("lt.launch.generator_or_tool_not_found", status=404, code="not_found")
    if generator.is_manual or tool.script_kind == "none":
        return {"scripts": [], "scripts_dir": "", "kind": tool.script_kind}
    scripts_dir = generator.scripts_dir_for(tool.id)
    if generator.is_ci:
        return {"scripts": [], "scripts_dir": scripts_dir, "kind": tool.script_kind, "message": t("lt.scripts.ci_relative")}
    if generator.type == "kubernetes" and not (generator.kubernetes and generator.kubernetes.scripts_config_map):
        return {"scripts": [], "scripts_dir": scripts_dir, "kind": tool.script_kind, "message": t("lt.scripts.k8s_manual")}
    if not scripts_dir:
        return {"scripts": [], "scripts_dir": "", "kind": tool.script_kind, "message": t("lt.scripts.no_dir")}
    executor = executor_for(generator)
    try:
        found = executor.list_scripts(scripts_dir, tool)
    except GeneratorError as exc:
        raise LaunchError(exc, status=502, code="generator_unreachable") from exc
    if found is None:
        return {"scripts": [], "scripts_dir": scripts_dir, "kind": tool.script_kind,
                "message": t("lt.scripts.dir_missing", dir=scripts_dir)}
    return {"scripts": found, "scripts_dir": scripts_dir, "kind": tool.script_kind}


__all__ = [
    "Catalog",
    "LaunchError",
    "PlannedRun",
    "create_run",
    "instance_command",
    "instance_plan",
    "instance_ref",
    "instance_logs",
    "active_runs",
    "launch_options",
    "list_scripts",
    "load_catalog",
    "ci_params_view",
    "ci_refs_view",
    "plan_run",
    "request_stop",
    "run_view",
]
