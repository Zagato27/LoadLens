"""Command templates of load tools and the command line of one test instance.

A template is a command with placeholders, for example ``{exe} run --tag testid={run_tag} {params} {script}``.
It is split into arguments first and the placeholders are substituted inside each argument afterwards,
so a value never becomes more than one argument. ``{params}`` stands alone and expands into the
parameters the user typed, split like a shell would split them; every argument is quoted when the
command line is built, so nothing the user types is interpreted by the shell on the generator.

Defaults for k6, JMeter, Locust and Gatling live here; ``CONFIG["load_tools"]`` overrides them per tool
(only the keys it sets) and may add tools of its own.
"""

from __future__ import annotations

import copy
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
PLACEHOLDERS = frozenset({
    "exe", "exe_dir", "script", "script_dir", "scripts_dir", "run_name", "run_tag",
    "instance", "instance_tag", "instance_dir", "control_port", "params",
})
CONTROL_PORT_BASE = 20000
CONTROL_PORT_SPAN = 20000
SCRIPT_KINDS = ("file", "class", "none")
STOP_SIGNALS = ("TERM", "INT", "HUP", "QUIT")
TOOL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
GLOB_RE = re.compile(r"^[A-Za-z0-9_.*?-]{1,60}$")
CLASS_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(\.[A-Za-z_$][A-Za-z0-9_$]*)*$")
TAG_RE = re.compile(r"[^A-Za-z0-9_-]+")
# Characters that never belong in a script path or a parameter: they would only hide what is run.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")

MAX_PARAMS_CHARS = 4000
MAX_ARGS = 200
MAX_ENV_VARS = 50
MAX_ENV_VALUE_CHARS = 4000
MAX_SCRIPT_CHARS = 500
MAX_TAG_CHARS = 60
MASK = "***"

# Source roots stripped from a simulation path to get the class name.
_CLASS_SOURCE_ROOTS = (
    "src/test/java/", "src/test/scala/", "src/test/kotlin/",
    "src/gatling/java/", "src/gatling/scala/", "src/gatling/kotlin/",
    "user-files/simulations/", "simulations/",
)
_CLASS_SUFFIXES = (".java", ".scala", ".kt")

DEFAULT_TOOLS: dict[str, dict[str, Any]] = {
    "k6": {
        "title": "k6",
        "executable": "k6",
        "command": "{exe} run --address 127.0.0.1:{control_port} --tag testid={run_tag} --tag lt_instance={instance_tag} {params} {script}",
        "env": {},
        "cwd": "{script_dir}",
        "script_kind": "file",
        "scripts_glob": ["*.js", "*.ts"],
        "success_exit_codes": [0],
        "failed_checks_exit_codes": [99],
        "stop_signal": "TERM",
        "grace_seconds": 30,
        "version_command": "{exe} version",
        "status_command": "{exe} status --address 127.0.0.1:{control_port}",
        "image": "grafana/k6:0.54.0",
        "run_filter_packs": ["lt-k6-prometheus-rw", "lt-k6-prometheus-native", "lt-k6-influxql"],
        "help": (
            "Вывод метрик задайте в параметрах: --out experimental-prometheus-rw (адрес — в переменной "
            "K6_PROMETHEUS_RW_SERVER_URL) или --out influxdb=http://…/k6. Тег testid с именем прогона "
            "LoadLens добавляет сам. Код 99 — пороги thresholds не пройдены, отчёт всё равно строится."
        ),
    },
    "jmeter": {
        "title": "JMeter",
        "executable": "jmeter",
        "command": (
            "{exe} -n -t {script} -j {instance_dir}/jmeter.log -Japplication={run_tag} -Jlt_instance={instance_tag}"
            " -Jjmeterengine.nongui.port={control_port} -Jjmeterengine.nongui.maxport={control_port} {params}"
        ),
        "env": {},
        "cwd": "{script_dir}",
        "script_kind": "file",
        "scripts_glob": ["*.jmx"],
        "success_exit_codes": [0],
        "failed_checks_exit_codes": [],
        "stop_signal": "TERM",
        "grace_seconds": 60,
        "version_command": "",
        # Shutdown lets the samplers finish and the Backend Listener flush; SIGTERM would cut the last batch.
        "stop_command": "{exe_dir}shutdown.sh {control_port}",
        "run_filter_packs": ["lt-jmeter-influxql"],
        "help": (
            "Режим без GUI. Чтобы LoadLens отделил метрики прогона, поле application в Backend Listener "
            "должно читать свойство: ${__P(application,my-app)}. Свойства теста передавайте как -Jимя=значение."
        ),
    },
    "locust": {
        "title": "Locust",
        "executable": "locust",
        "command": "{exe} -f {script} --headless {params}",
        "env": {},
        "cwd": "{script_dir}",
        "script_kind": "file",
        "scripts_glob": ["locustfile*.py", "*locust*.py"],
        "success_exit_codes": [0],
        "failed_checks_exit_codes": [1],
        "stop_signal": "TERM",
        "grace_seconds": 30,
        "version_command": "{exe} --version",
        "image": "locustio/locust:2.31.8",
        "help": (
            "Задайте --users, --spawn-rate и --run-time, иначе тест идёт до остановки или лимита длительности. "
            "Метрики Locust приходят через locust_exporter и отделяются только окном прогона."
        ),
    },
    "gatling": {
        "title": "Gatling (бандл)",
        "executable": "/opt/gatling/bin/gatling.sh",
        "command": "{exe} -rm local -s {script} -rd {run_tag} {params}",
        "env": {},
        "cwd": "{instance_dir}",
        "script_kind": "class",
        "scripts_glob": ["*Simulation.java", "*Simulation.scala", "*Simulation.kt"],
        "success_exit_codes": [0],
        "failed_checks_exit_codes": [2],
        "stop_signal": "TERM",
        "grace_seconds": 20,
        "version_command": "",
        "help": (
            "Бандл Gatling до 3.10 включительно; каталог скриптов — user-files/simulations бандла. JVM-свойства "
            "симуляции передавайте в переменной JAVA_OPTS. Метрики Gatling отделяются окном прогона; как "
            "отделить их тегом при записи в VictoriaMetrics — в docs/load-tests.md. Код 2 — не прошли assertions."
        ),
    },
    "gatling_maven": {
        "title": "Gatling (Maven)",
        "executable": "./mvnw",
        "command": "{exe} gatling:test -Dgatling.simulationClass={script} {params}",
        "env": {},
        "cwd": "{scripts_dir}",
        "script_kind": "class",
        "scripts_glob": ["*Simulation.java", "*Simulation.scala", "*Simulation.kt"],
        "success_exit_codes": [0],
        "failed_checks_exit_codes": [],
        "stop_signal": "TERM",
        "grace_seconds": 30,
        "version_command": "",
        "help": (
            "Проект с gatling-maven-plugin, так устроены и бандлы Gatling 3.11+. Каталог скриптов — корень "
            "проекта с pom.xml и mvnw. Свойства симуляции передавайте как -Dимя=значение."
        ),
    },
}


class ToolConfigError(ValueError):
    """A tool template in the settings is not usable; the message names the tool and the field."""


class ToolInputError(ValueError):
    """Parameters, environment or script of an instance cannot be used; the message is for the user."""


@dataclass(frozen=True)
class ToolTemplate:
    id: str
    title: str
    executable: str
    command: str
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "{script_dir}"
    script_kind: str = "file"
    scripts_glob: tuple[str, ...] = ()
    success_exit_codes: tuple[int, ...] = (0,)
    failed_checks_exit_codes: tuple[int, ...] = ()
    stop_signal: str = "TERM"
    grace_seconds: int = 30
    version_command: str = ""
    # Query packs whose run filter matches how this command tags the run (see RunFilter in AI/query_packs).
    run_filter_packs: tuple[str, ...] = ()
    # Container image for Docker and Kubernetes generators.
    image: str = ""
    # Commands that talk to the running tool where it runs: a soft stop instead of the signal, and a status.
    stop_command: str = ""
    status_command: str = ""
    help: str = ""

    def to_public(self) -> dict[str, Any]:
        """What the launch form needs: no defaults of other tools, nothing secret."""
        return {
            "id": self.id,
            "title": self.title,
            "script_kind": self.script_kind,
            "scripts_glob": list(self.scripts_glob),
            "command": self.command,
            "env_keys": sorted(self.env),
            "success_exit_codes": list(self.success_exit_codes),
            "failed_checks_exit_codes": list(self.failed_checks_exit_codes),
            "stop_signal": self.stop_signal,
            "grace_seconds": self.grace_seconds,
            "run_filter_packs": list(self.run_filter_packs),
            "image": self.image,
            "soft_stop": bool(self.stop_command),
            "help": self.help,
        }


@dataclass(frozen=True)
class RenderContext:
    exe: str
    exe_dir: str = ""
    control_port: str = ""
    script: str = ""
    script_dir: str = ""
    scripts_dir: str = ""
    run_name: str = ""
    run_tag: str = ""
    instance: str = ""
    instance_tag: str = ""
    instance_dir: str = ""

    def values(self) -> dict[str, str]:
        return {
            "exe": self.exe,
            "exe_dir": self.exe_dir,
            "control_port": self.control_port,
            "script": self.script,
            "script_dir": self.script_dir,
            "scripts_dir": self.scripts_dir,
            "run_name": self.run_name,
            "run_tag": self.run_tag,
            "instance": self.instance,
            "instance_tag": self.instance_tag,
            "instance_dir": self.instance_dir,
        }


@dataclass(frozen=True)
class CommandLine:
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: str

    def shell(self, mask_env: bool = False) -> str:
        """The command as one quoted shell line; ``mask_env`` hides variable values (logs, audit, UI)."""
        parts: list[str] = []
        if self.env:
            parts.append("env")
            for name in sorted(self.env):
                value = MASK if mask_env else self.env[name]
                parts.append(shlex.quote(f"{name}={value}"))
        parts.extend(shlex.quote(arg) for arg in self.argv)
        return " ".join(parts)


# ---- configuration -----------------------------------------------------------------------------


def _int_list(value: Any, where: str) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ToolConfigError(f"{where}: ожидается список кодов выхода")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, str)) or not str(item).strip().lstrip("-").isdigit():
            raise ToolConfigError(f"{where}: «{item}» — не код выхода")
        code = int(item)
        if not 0 <= code <= 255:
            raise ToolConfigError(f"{where}: код выхода {code} вне диапазона 0–255")
        out.append(code)
    return tuple(dict.fromkeys(out))


def _template_names(text: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(text or ""))


def _check_placeholders(text: str, where: str, allow_params: bool) -> None:
    unknown = _template_names(text) - PLACEHOLDERS
    if unknown:
        raise ToolConfigError(f"{where}: неизвестные подстановки {', '.join('{' + n + '}' for n in sorted(unknown))}")
    if not allow_params and "params" in _template_names(text):
        raise ToolConfigError(f"{where}: {{params}} допустим только в команде")


def parse_tool(tool_id: str, raw: Mapping[str, Any]) -> ToolTemplate:
    """Validated template; raises :class:`ToolConfigError` with the tool and field in the message."""
    where = f"Инструмент «{tool_id}»"
    if not TOOL_ID_RE.fullmatch(tool_id):
        raise ToolConfigError(f"{where}: идентификатор — латиница в нижнем регистре, цифры, дефис и подчёркивание")
    if not isinstance(raw, Mapping):
        raise ToolConfigError(f"{where}: ожидается объект")
    command = str(raw.get("command") or "").strip()
    if not command:
        raise ToolConfigError(f"{where}: команда не задана")
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ToolConfigError(f"{where}: команда не разбирается на аргументы ({exc})") from exc
    if not tokens:
        raise ToolConfigError(f"{where}: команда не задана")
    _check_placeholders(command, f"{where}, команда", allow_params=True)
    params_tokens = [token for token in tokens if "{params}" in token]
    if len(params_tokens) > 1 or (params_tokens and params_tokens[0] != "{params}"):
        raise ToolConfigError(f"{where}: {{params}} должен стоять в команде один раз отдельным аргументом")
    env_raw = raw.get("env") or {}
    if not isinstance(env_raw, Mapping):
        raise ToolConfigError(f"{where}: env — объект «переменная: значение»")
    env: dict[str, str] = {}
    for name, value in env_raw.items():
        if not ENV_NAME_RE.fullmatch(str(name)):
            raise ToolConfigError(f"{where}: недопустимое имя переменной окружения «{name}»")
        text = "" if value is None else str(value)
        _check_placeholders(text, f"{where}, env.{name}", allow_params=False)
        env[str(name)] = text
    cwd = str(raw.get("cwd") or "{script_dir}").strip()
    _check_placeholders(cwd, f"{where}, cwd", allow_params=False)
    kind = str(raw.get("script_kind") or "file")
    if kind not in SCRIPT_KINDS:
        raise ToolConfigError(f"{where}: script_kind — одно из {', '.join(SCRIPT_KINDS)}")
    globs_raw = raw.get("scripts_glob") or []
    if isinstance(globs_raw, str):
        globs_raw = [globs_raw]
    if not isinstance(globs_raw, (list, tuple)) or len(globs_raw) > 10:
        raise ToolConfigError(f"{where}: scripts_glob — список до 10 масок")
    globs = tuple(str(item) for item in globs_raw)
    for pattern in globs:
        if not GLOB_RE.fullmatch(pattern):
            raise ToolConfigError(f"{where}: маска «{pattern}» — только буквы, цифры, точка, дефис, * и ?")
    signal = str(raw.get("stop_signal") or "TERM").upper().removeprefix("SIG")
    if signal not in STOP_SIGNALS:
        raise ToolConfigError(f"{where}: stop_signal — одно из {', '.join(STOP_SIGNALS)}")
    try:
        grace = int(raw.get("grace_seconds", 30))
    except (TypeError, ValueError) as exc:
        raise ToolConfigError(f"{where}: grace_seconds — целое число секунд") from exc
    if not 1 <= grace <= 600:
        raise ToolConfigError(f"{where}: grace_seconds — от 1 до 600")
    aux: dict[str, str] = {}
    for key in ("version_command", "stop_command", "status_command"):
        text = str(raw.get(key) or "").strip()
        if text:
            _check_placeholders(text, f"{where}, {key}", allow_params=False)
            try:
                shlex.split(text, posix=True)
            except ValueError as exc:
                raise ToolConfigError(f"{where}: {key} не разбирается ({exc})") from exc
        aux[key] = text
    image = str(raw.get("image") or "").strip()
    if any(ch.isspace() for ch in image):
        raise ToolConfigError(f"{where}: недопустимое имя образа «{image}»")
    packs_raw = raw.get("run_filter_packs") or []
    if not isinstance(packs_raw, (list, tuple)) or not all(isinstance(item, str) for item in packs_raw):
        raise ToolConfigError(f"{where}: run_filter_packs — список идентификаторов пакетов запросов")
    executable = str(raw.get("executable") or "").strip()
    if "{exe}" in command and not executable:
        raise ToolConfigError(f"{where}: команда использует {{exe}}, а executable не задан")
    return ToolTemplate(
        id=tool_id,
        title=str(raw.get("title") or tool_id).strip()[:80] or tool_id,
        executable=executable,
        command=command,
        env=env,
        cwd=cwd,
        script_kind=kind,
        scripts_glob=globs,
        success_exit_codes=_int_list(raw.get("success_exit_codes", [0]), f"{where}, success_exit_codes") or (0,),
        failed_checks_exit_codes=_int_list(raw.get("failed_checks_exit_codes"), f"{where}, failed_checks_exit_codes"),
        stop_signal=signal,
        grace_seconds=grace,
        version_command=aux["version_command"],
        stop_command=aux["stop_command"],
        status_command=aux["status_command"],
        image=image,
        run_filter_packs=tuple(dict.fromkeys(item.strip() for item in packs_raw if item.strip())),
        help=str(raw.get("help") or "").strip()[:2000],
    )


def merged_tool_configs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Defaults with ``CONFIG["load_tools"]`` applied key by key; ``"enabled": false`` hides a tool."""
    merged = copy.deepcopy(DEFAULT_TOOLS)
    custom = config.get("load_tools") if isinstance(config.get("load_tools"), Mapping) else {}
    for tool_id, raw in custom.items():
        if not isinstance(raw, Mapping):
            continue
        base = merged.get(str(tool_id), {})
        merged[str(tool_id)] = {**base, **copy.deepcopy(dict(raw))}
    return {tool_id: raw for tool_id, raw in merged.items() if raw.get("enabled", True) is not False}


def effective_tools(config: Mapping[str, Any]) -> dict[str, ToolTemplate]:
    """Usable templates; a broken one is left out (the settings page reports it on save)."""
    out: dict[str, ToolTemplate] = {}
    for tool_id, raw in merged_tool_configs(config).items():
        try:
            out[tool_id] = parse_tool(tool_id, raw)
        except ToolConfigError:
            continue
    return out


def validate_tools_section(payload: Mapping[str, Any]) -> None:
    """Raises :class:`ToolConfigError` for the first tool of ``load_tools`` that cannot be used."""
    if not isinstance(payload, Mapping):
        raise ToolConfigError("load_tools — объект «инструмент: шаблон»")
    for tool_id, raw in payload.items():
        if not isinstance(raw, Mapping):
            raise ToolConfigError(f"Инструмент «{tool_id}»: ожидается объект")
    for tool_id, raw in merged_tool_configs({"load_tools": payload}).items():
        parse_tool(tool_id, raw)


def tools_overrides(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """What the edited templates change against the built-in ones; a missing built-in tool is disabled.

    The settings page sends the effective templates back. Storing only the differences keeps later
    fixes of the built-in templates working for keys nobody changed.
    """
    out: dict[str, dict[str, Any]] = {}
    for tool_id, raw in payload.items():
        if not isinstance(raw, Mapping):
            continue
        base = DEFAULT_TOOLS.get(str(tool_id))
        if base is None:
            out[str(tool_id)] = copy.deepcopy(dict(raw))
            continue
        diff = {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if base.get(key) != value and not (key == "enabled" and value is True and "enabled" not in base)
        }
        if diff:
            out[str(tool_id)] = diff
    for tool_id in DEFAULT_TOOLS:
        if tool_id not in payload:
            out[tool_id] = {"enabled": False}
    return out


# ---- user input -------------------------------------------------------------------------------


def make_tag(text: str, fallback: str) -> str:
    """Letters, digits, ``_`` and ``-`` only: safe as a k6 tag, a Graphite prefix and a regex literal."""
    tag = TAG_RE.sub("-", str(text or "")).strip("-_")[:MAX_TAG_CHARS].strip("-_")
    return tag or fallback


def parse_params(text: Any) -> list[str]:
    """Arguments from what the user typed, split with shell quoting rules (quotes group, nothing expands)."""
    if text is None:
        return []
    if isinstance(text, list):
        args = [str(item) for item in text]
    else:
        raw = str(text)
        if len(raw) > MAX_PARAMS_CHARS:
            raise ToolInputError(f"Параметры длиннее {MAX_PARAMS_CHARS} символов")
        try:
            args = shlex.split(raw.replace("\r\n", "\n").replace("\n", " "), posix=True)
        except ValueError as exc:
            raise ToolInputError(f"Параметры не разбираются: проверьте парные кавычки ({exc})") from exc
    if len(args) > MAX_ARGS:
        raise ToolInputError(f"Больше {MAX_ARGS} аргументов в параметрах")
    for arg in args:
        if _CONTROL_RE.search(arg):
            raise ToolInputError("Параметры содержат управляющие символы")
    return args


def parse_env(value: Any) -> dict[str, str]:
    """``{"NAME": "value"}`` or text with one ``NAME=value`` per line (empty lines and ``#`` comments skipped)."""
    if value in (None, ""):
        return {}
    pairs: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        pairs = [(str(k), "" if v is None else str(v)) for k, v in value.items()]
    elif isinstance(value, str):
        for line in value.replace("\r\n", "\n").split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name, sep, rest = stripped.partition("=")
            if not sep:
                raise ToolInputError(f"Строка окружения «{stripped[:40]}»: ожидается ИМЯ=значение")
            pairs.append((name.strip(), rest))
    else:
        raise ToolInputError("Окружение — строки ИМЯ=значение")
    if len(pairs) > MAX_ENV_VARS:
        raise ToolInputError(f"Больше {MAX_ENV_VARS} переменных окружения")
    env: dict[str, str] = {}
    for name, val in pairs:
        if not ENV_NAME_RE.fullmatch(name):
            raise ToolInputError(f"Недопустимое имя переменной окружения «{name[:40]}»")
        if len(val) > MAX_ENV_VALUE_CHARS or _CONTROL_RE.search(val):
            raise ToolInputError(f"Значение переменной {name} слишком длинное или содержит управляющие символы")
        env[name] = val
    return env


# Variables that change how the shell, the dynamic loader or the runtimes start rather than configure the
# tool; an instance may not set them (JAVA_OPTS, HEAP, JVM_ARGS and the tools' own variables stay allowed).
RESERVED_ENV_NAMES = frozenset({
    "PATH", "HOME", "SHELL", "IFS", "ENV", "BASH_ENV", "SHELLOPTS", "BASHOPTS", "PS4", "PROMPT_COMMAND",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
    "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS",
    "NODE_OPTIONS", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PERL5OPT", "PERL5LIB", "RUBYOPT",
    "LOADLENS_TOKEN",
})


def check_instance_env(env: Mapping[str, str], generator_env: Mapping[str, str]) -> None:
    """Rejects variables an instance may not set: reserved names and those the generator settings fix.

    The generator's variables often carry credentials (remote write password); letting an instance replace
    the address next to them would send those credentials elsewhere.
    """
    reserved = sorted(name for name in env if name.upper() in RESERVED_ENV_NAMES or name.startswith("LD_"))
    if reserved:
        raise ToolInputError(f"Переменные {', '.join(reserved)} нельзя задавать у инстанса")
    fixed = sorted(name for name in env if name in generator_env)
    if fixed:
        raise ToolInputError(f"Переменные {', '.join(fixed)} заданы в настройках генератора, у инстанса их не переопределить")


def public_url(url: str) -> str:
    """The URL without ``user:password@``: for what engineers see (launch form, metric output suggestions)."""
    from urllib.parse import urlsplit, urlunsplit

    text = str(url or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if "@" not in parts.netloc:
        return text
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[-1]))


def is_absolute_path(path: str) -> bool:
    return str(path or "").startswith("/")


def resolve_script(tool: ToolTemplate, script: Any, scripts_dir: str, allow_relative: bool = False) -> tuple[str, str]:
    """(value for ``{script}``, value for ``{script_dir}``) or :class:`ToolInputError`.

    ``allow_relative`` keeps relative paths relative (CI jobs run in their checkout). With a scripts
    directory the script must lie inside it; without one (CI checkout) it may not climb out with ``..``.
    """
    text = str(script or "").strip()
    if tool.script_kind == "none":
        return "", scripts_dir
    if not text:
        raise ToolInputError(f"{tool.title}: укажите {'класс симуляции' if tool.script_kind == 'class' else 'скрипт'}")
    if len(text) > MAX_SCRIPT_CHARS or _CONTROL_RE.search(text):
        raise ToolInputError(f"{tool.title}: недопустимое имя скрипта")
    if tool.script_kind == "class":
        if not CLASS_RE.fullmatch(text):
            raise ToolInputError(f"{tool.title}: «{text}» — не имя класса (например computerdatabase.BasicSimulation)")
        return text, scripts_dir
    path = posixpath.normpath(text if is_absolute_path(text) else posixpath.join(scripts_dir or "", text))
    if scripts_dir:
        root = posixpath.normpath(scripts_dir)
        inside = path == root or path.startswith(root.rstrip("/") + "/")
        if not inside:
            raise ToolInputError(f"{tool.title}: скрипт «{text}» вне каталога скриптов генератора {scripts_dir}")
    if not is_absolute_path(path):
        if not allow_relative:
            raise ToolInputError(f"{tool.title}: каталог скриптов генератора не задан, укажите абсолютный путь к скрипту")
        if path == ".." or path.startswith("../"):
            raise ToolInputError(f"{tool.title}: скрипт «{text}» вне каталога задачи")
        return path, posixpath.dirname(path) or "."
    return path, posixpath.dirname(path) or "/"


def control_port(run_id: str, index: int) -> int:
    """A port for the tool's control API, stable for the instance and unlikely to clash with another run."""
    import hashlib

    digest = hashlib.sha1(f"{run_id}:{index}".encode()).digest()
    return CONTROL_PORT_BASE + int.from_bytes(digest[:4], "big") % CONTROL_PORT_SPAN


def exe_dir(exe: str) -> str:
    """Directory of the executable with a trailing slash, or empty for a name looked up in PATH."""
    head = posixpath.dirname(exe or "")
    return f"{head}/" if head else ""


def class_name_from_path(path: str) -> str:
    """``src/test/java/acme/CheckoutSimulation.java`` -> ``acme.CheckoutSimulation``."""
    text = str(path or "").strip().lstrip("./")
    for root in _CLASS_SOURCE_ROOTS:
        index = text.find(root)
        if index >= 0:
            text = text[index + len(root):]
            break
    for suffix in _CLASS_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.replace("/", ".")


# ---- rendering ---------------------------------------------------------------------------------


def _substitute(text: str, values: Mapping[str, str]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            raise ToolConfigError(f"неизвестная подстановка {{{name}}}")
        return values[name]

    return PLACEHOLDER_RE.sub(replace, text)


def render_command(
    tool: ToolTemplate,
    ctx: RenderContext,
    params: list[str],
    user_env: Optional[Mapping[str, str]] = None,
) -> CommandLine:
    """argv, environment and working directory of one instance.

    A variable set both by the template and by the user gets the user value appended after a space,
    which is what ``JAVA_OPTS``-like variables need.
    """
    values = ctx.values()
    argv: list[str] = []
    for token in shlex.split(tool.command, posix=True):
        if token == "{params}":
            argv.extend(params)
            continue
        argv.append(_substitute(token, values))
    env = {name: _substitute(value, values) for name, value in tool.env.items()}
    for name, value in (user_env or {}).items():
        env[name] = f"{env[name]} {value}".strip() if env.get(name) else value
    cwd = _substitute(tool.cwd, values) or ctx.instance_dir
    return CommandLine(argv=tuple(argv), env=env, cwd=cwd)


def render_version_command(tool: ToolTemplate, exe: str) -> list[str]:
    if not tool.version_command:
        return []
    return [_substitute(token, {"exe": exe, "exe_dir": exe_dir(exe)}) for token in shlex.split(tool.version_command, posix=True)]


def render_aux_command(text: str, ctx: RenderContext) -> tuple[str, ...]:
    """A stop or status command of the template with the instance's values."""
    if not text:
        return ()
    values = ctx.values()
    return tuple(_substitute(token, values) for token in shlex.split(text, posix=True))


__all__ = [
    "CLASS_RE",
    "CommandLine",
    "DEFAULT_TOOLS",
    "PLACEHOLDERS",
    "RenderContext",
    "SCRIPT_KINDS",
    "STOP_SIGNALS",
    "ToolConfigError",
    "ToolInputError",
    "ToolTemplate",
    "class_name_from_path",
    "control_port",
    "effective_tools",
    "exe_dir",
    "is_absolute_path",
    "make_tag",
    "merged_tool_configs",
    "parse_env",
    "parse_params",
    "parse_tool",
    "render_aux_command",
    "render_command",
    "render_version_command",
    "resolve_script",
    "tools_overrides",
    "validate_tools_section",
]
