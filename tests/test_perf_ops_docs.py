"""The background job documentation matches the code (docs/perf-plan.md, wave B, agent B6).

Offline: reads the documentation, the compose files and the Python sources; imports the job modules only
to read their defaults, messages and registered task names. Nothing connects to a broker or a database.

- Every ``LOADLENS_JOB_*`` / ``LOADLENS_REPORT_*`` variable named in the documentation is read by the code,
  and every one the code reads is documented (docs/jobs-runner.md, README.md, docker/env.example) and
  passed to both the app and the worker by docker-compose.yml.
- The defaults in the table of docs/jobs-runner.md, in docker-compose.yml and in docker/env.example are
  the defaults of the code.
- Task names, queue names, user-facing job messages and the broker warning quoted in the docs exist.
- Links into docs/jobs-runner.md point at existing headings.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before anything that imports update_page)
from loadlens_app import celery_app as celery_module  # noqa: E402
from loadlens_app import jobs, tasks  # noqa: E402

yaml = pytest.importorskip("yaml")

JOBS_DOC = ROOT_DIR / "docs" / "jobs-runner.md"
README = ROOT_DIR / "README.md"
DOCKER_SETUP = ROOT_DIR / "docker" / "DOCKER_SETUP.md"
ENV_EXAMPLE = ROOT_DIR / "docker" / "env.example"
COMPOSE = ROOT_DIR / "docker" / "docker-compose.yml"
JOB_RUNNER_SOURCE = ROOT_DIR / "loadlens_app" / "job_runner.py"
DOCS = (JOBS_DOC, README, DOCKER_SETUP, ENV_EXAMPLE)
# Directories whose Python files are not the application: tests, developer tools, the test stand, environments.
NOT_APP_DIRS = {"tests", "tools", "dev-stack", ".venv", "venv", "node_modules", ".git", ".claude", "__pycache__"}
VARIABLE = re.compile(r"\bLOADLENS_(?:JOB|REPORT)_[A-Z0-9_]*[A-Z0-9]\b")
DOC_TABLE_ROW = re.compile(r"^\|\s*`(LOADLENS_(?:JOB|REPORT)_[A-Z0-9_]+)`\s*\|\s*`([^`]+)`\s*\|", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _app_python_files():
    for path in ROOT_DIR.rglob("*.py"):
        if not NOT_APP_DIRS.intersection(path.relative_to(ROOT_DIR).parts):
            yield path


def _code_variables() -> set[str]:
    """Job variables named in string literals of the application code (os.getenv calls and *_ENV constants)."""
    found: set[str] = set()
    for path in _app_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and VARIABLE.fullmatch(node.value):
                found.add(node.value)
    return found


def _doc_variables(path: Path) -> set[str]:
    return set(VARIABLE.findall(_read(path)))


def _code_defaults(monkeypatch) -> dict[str, str]:
    """What the code uses when the variable is not set, as the docs write it."""
    for name in ("LOADLENS_JOB_RUNNER", "LOADLENS_JOB_STALE_MIN", "LOADLENS_JOB_QUEUE_MAX_MIN", "LOADLENS_REPORT_TASK_LIMIT_SEC"):
        monkeypatch.delenv(name, raising=False)
    return {
        "LOADLENS_JOB_RUNNER": celery_module.job_runner_setting(),
        "LOADLENS_JOB_STALE_MIN": f"{jobs.stale_after().total_seconds() / 60:g}",
        "LOADLENS_JOB_QUEUE_MAX_MIN": f"{jobs.queue_max_wait().total_seconds() / 60:g}",
        "LOADLENS_REPORT_TASK_LIMIT_SEC": str(celery_module.report_task_time_limits()[0]),
    }


def _compose_services() -> dict:
    return yaml.safe_load(_read(COMPOSE))["services"]


def _slug(heading: str) -> str:
    """GitHub's anchor of a markdown heading: lower case, punctuation dropped, spaces to dashes."""
    text = heading.strip().lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch in " -_")
    return text.replace(" ", "-")


# ---- variables -----------------------------------------------------------------------------------------------


def test_the_code_reads_job_variables():
    assert {"LOADLENS_JOB_RUNNER", "LOADLENS_JOB_STALE_MIN", "LOADLENS_JOB_QUEUE_MAX_MIN", "LOADLENS_REPORT_TASK_LIMIT_SEC"} <= _code_variables()


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_job_variables_are_read_by_the_code(doc):
    unknown = _doc_variables(doc) - _code_variables()
    assert not unknown, f"{doc.name} names variables the code does not read: {sorted(unknown)}"


@pytest.mark.parametrize("doc", (JOBS_DOC, README, ENV_EXAMPLE), ids=lambda p: p.name)
def test_job_variables_of_the_code_are_documented(doc):
    missing = _code_variables() - _doc_variables(doc)
    assert not missing, f"{doc.name} does not describe {sorted(missing)}"


def test_compose_passes_job_variables_to_app_and_worker():
    services = _compose_services()
    for service in ("app", "celery_worker"):
        environment = services[service]["environment"]
        missing = _code_variables() - set(environment)
        assert not missing, f"docker-compose.yml does not pass {sorted(missing)} to {service}"


def test_documented_defaults_match_the_code(monkeypatch):
    defaults = _code_defaults(monkeypatch)
    table = dict(DOC_TABLE_ROW.findall(_read(JOBS_DOC)))
    assert set(table) == set(defaults), "the variables table of docs/jobs-runner.md lists other variables"
    assert table == defaults


def test_compose_and_env_example_defaults_match_the_code(monkeypatch):
    defaults = _code_defaults(monkeypatch)
    pattern = re.compile(r"^\$\{(\w+):-([^}]*)\}$")
    for service in ("app", "celery_worker"):
        environment = _compose_services()[service]["environment"]
        for name, expected in defaults.items():
            match = pattern.match(str(environment[name]))
            assert match and match.group(1) == name and match.group(2) == expected, f"{service}.{name}: {environment[name]}"
    example = dict(re.findall(r"^(LOADLENS_(?:JOB|REPORT)_\w+)=(.*)$", _read(ENV_EXAMPLE), re.MULTILINE))
    assert example == defaults


# ---- names and messages quoted in the docs -------------------------------------------------------------------


def test_documented_task_names_are_the_registered_tasks():
    registered = {name for name in celery_module.celery_app.tasks.keys() if name.startswith(celery_module.TASK_NAME_PREFIX)}
    documented = set(re.findall(r"\bloadlens\.[a-z_]+\b", _read(JOBS_DOC)))
    assert registered == {tasks.report_task.name, tasks.publish_confluence_task.name, tasks.publish_forecast_task.name}
    assert documented == registered


def test_documented_worker_command_uses_the_queues_of_the_code():
    queues = f"-Q {celery_module.REPORTS_QUEUE},{celery_module.DEFAULT_QUEUE}"
    assert queues in _compose_services()["celery_worker"]["command"]
    for doc in (JOBS_DOC, DOCKER_SETUP):
        assert queues in _read(doc), f"{doc.name} shows a worker command without {queues}"
    assert f"llen {celery_module.REPORTS_QUEUE}" in _read(JOBS_DOC)


@pytest.mark.parametrize(
    "message",
    (jobs.STALE_MESSAGE, jobs.QUEUED_MESSAGE, jobs.INTERRUPTED_MESSAGE, tasks.TIME_LIMIT_MESSAGE),
    ids=("stale", "queued", "interrupted", "time-limit"),
)
def test_job_messages_quoted_in_the_docs_are_the_messages_of_the_code(message):
    assert message in _read(JOBS_DOC)


def test_broker_warning_quoted_in_the_docs_is_logged_by_the_code():
    assert "Could not send job %s (%s) to the Celery broker" in _read(JOB_RUNNER_SOURCE)
    quoted = "Could not send job <id> (loadlens.report) to the Celery broker"
    assert quoted in _read(JOBS_DOC)
    for doc in (README, DOCKER_SETUP):
        assert "Could not send job" in _read(doc)


def test_documented_heartbeat_and_sweep_intervals():
    text = _read(JOBS_DOC)
    assert tasks.HEARTBEAT_INTERVAL_SEC == 30.0 and "раз в 30 с — heartbeat" in text
    assert jobs.SWEEP_INTERVAL_SEC == 30.0 and "не чаще раза в 30 с" in text
    assert tasks.FINAL_UPDATE_RETRY_DELAYS_SEC == (1.0, 3.0) and "через 1 и 3 с" in text
    assert celery_module.REPORT_TASK_HARD_LIMIT_GRACE_SEC == 60 and "через 60 с после него" in text


# ---- links ---------------------------------------------------------------------------------------------------


def test_links_into_the_jobs_doc_point_at_its_headings():
    text = _read(JOBS_DOC)
    anchors = {_slug(line.lstrip("#")) for line in text.splitlines() if line.startswith("#")}
    links = set(re.findall(r"\]\(#([^)]+)\)", text))
    for doc in (README, DOCKER_SETUP):
        links |= set(re.findall(r"jobs-runner\.md#([^)]+)\)", _read(doc)))
        assert "jobs-runner.md" in _read(doc), f"{doc.name} does not link docs/jobs-runner.md"
    assert links, "expected links into docs/jobs-runner.md"
    assert links <= anchors, f"broken anchors: {sorted(links - anchors)}"


def test_e2e_scripts_listed_in_the_jobs_doc_exist():
    listed = set(re.findall(r"`((?:s\d_\w+|bench_modes)\.py)`", _read(JOBS_DOC)))
    present = {path.name for path in (ROOT_DIR / "tools" / "e2e_jobs").glob("*.py") if path.name != "e2e_common.py"}
    assert listed == present
