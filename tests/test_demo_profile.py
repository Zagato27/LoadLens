# SPDX-License-Identifier: Apache-2.0
"""The demo stack (plans/open-source-plan.md, section 4): docker/docker-compose.demo.yml, docker/demo.env,
settings.demo.py, the automatic seed behind LOADLENS_DEMO_SEED and the "demo data" banner of the dashboard.

Offline: the compose file and the settings are parsed, never started; the seed hook is exercised with fakes.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from loadlens_app import core  # noqa: F401  (before anything that imports update_page)
from tests.i18n_helpers import assert_no_cyrillic, use_locale, visible_text

ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT_DIR / "docker" / "docker-compose.demo.yml"
ENV_PATH = ROOT_DIR / "docker" / "demo.env"
SETTINGS_PATH = ROOT_DIR / "settings.demo.py"
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "demo.yml"
DEMO_COMMAND = "docker compose -f docker/docker-compose.demo.yml --env-file docker/demo.env --profile demo up"
#: Hosts the demo may talk to: the services of the compose file and the loopback.
DEMO_HOSTS = {"timescaledb", "llm-stub", "prom-stub", "127.0.0.1", "localhost"}
URL_HOST = re.compile(r"https?://([^/\"'\s:]+)")
# Variables the application and the authentication read from the container environment.
APP_ENV_VARS = {
    "LOADLENS_AUTH_ENABLED", "LOADLENS_SECRET_KEY", "LOADLENS_ADMIN_USER", "LOADLENS_ADMIN_PASSWORD",
    "LOADLENS_RUN_MIGRATIONS", "LOADLENS_LOAD_TEST_WATCHER", "LOADLENS_JOB_RUNNER", "LOADLENS_DEMO_SEED",
    "CELERY_TASK_ALWAYS_EAGER",
}


def _env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


@pytest.fixture(scope="module")
def compose():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def demo_env():
    return _env_file(ENV_PATH)


@pytest.fixture(scope="module")
def demo_config():
    """CONFIG of settings.demo.py loaded under another module name (the real ``settings`` stays untouched)."""
    spec = importlib.util.spec_from_file_location("loadlens_settings_demo", SETTINGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


# ---- settings.demo.py -----------------------------------------------------------------------------------------


def test_demo_settings_point_only_at_the_demo_stack(demo_config):
    hosts = {match for text in _strings(demo_config) for match in URL_HOST.findall(text)}
    assert hosts and hosts <= DEMO_HOSTS, sorted(hosts - DEMO_HOSTS)
    storage = demo_config["storage"]["timescale"]
    assert storage["host"] == "timescaledb" and storage["port"] == 5432
    assert demo_config["llm"]["provider"] == "openai"
    assert demo_config["llm"]["openai"]["api_base_url"] == "http://llm-stub:5090/v1"
    assert demo_config["data_sources"] == {"prometheus": {"title": "Prometheus (demo stub)", "type": "prometheus", "prometheus": {"url": "http://prom-stub:5092"}}}
    assert demo_config["domain_sources"] == {"default": {"source": "prometheus"}}
    assert demo_config["appearance"]["language"] == "en"
    assert demo_config["auth"]["enabled"] is True
    assert demo_config["confluence"]["enabled"] is False and demo_config["logs_source"]["enabled"] is False
    for provider in ("perplexity", "anthropic", "gigachat"):
        assert demo_config["llm"][provider]["api_key"] == "" and demo_config["llm"][provider]["api_base_url"] == ""


def test_demo_settings_keep_the_example_labels_with_promql_queries(demo_config):
    """The seeded demo run uses the labels of the example; the stub report must share them (comparison page)."""
    spec = importlib.util.spec_from_file_location("loadlens_settings_example_for_demo", ROOT_DIR / "settings.example.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    lt = demo_config["queries"]["lt_framework"]
    assert lt["labels"] == example.CONFIG["queries"]["lt_framework"]["labels"]
    assert len(lt["promql_queries"]) == len(lt["labels"]) == len(lt["label_keys_list"])
    assert "influxql_queries" not in lt and "label_tag_keys_list" not in lt
    sla = demo_config["sla"]
    assert sla["max_performance_query"] in lt["labels"] and sla["p95_query"] in lt["labels"]
    assert sla["target_rps"] and sla["max_p95_ms"]
    for domain in ("jvm", "microservices", "database", "kafka", "hard_resources"):
        assert demo_config["queries"][domain]["promql_queries"]


def test_demo_settings_read_the_demo_env_variables(monkeypatch):
    monkeypatch.setenv("DEMO_DB_PASSWORD", "another-demo-password")
    monkeypatch.setenv("DEMO_DB_NAME", "other_db")
    spec = importlib.util.spec_from_file_location("loadlens_settings_demo_env", SETTINGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    storage = module.CONFIG["storage"]["timescale"]
    assert storage["password"] == "another-demo-password" and storage["dbname"] == "other_db"


# ---- docker/demo.env and docker/docker-compose.demo.yml -------------------------------------------------------


def test_demo_env_has_no_empty_values_and_no_real_looking_secrets(demo_env):
    assert demo_env, "docker/demo.env is empty"
    assert all(value for value in demo_env.values()), [k for k, v in demo_env.items() if not v]
    assert demo_env["APP_PORT"] == "5000"
    assert len(demo_env["DEMO_ADMIN_PASSWORD"]) >= 10 and "admin" not in demo_env["DEMO_ADMIN_PASSWORD"].lower()
    for key in ("DEMO_DB_PASSWORD", "DEMO_ADMIN_PASSWORD", "DEMO_SECRET_KEY"):
        value = demo_env[key]
        assert "demo" in value.lower(), key  # obviously a demo value
        assert not re.fullmatch(r"[0-9a-f]{32,}", value), f"{key} looks like a generated secret"
    header = ENV_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("#")


def test_demo_env_matches_the_compose_defaults(compose, demo_env):
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    for name, value in demo_env.items():
        defaults = set(re.findall(r"\$\{" + re.escape(name) + r":-([^}]*)\}", text))
        assert defaults == {value}, f"{name}: compose defaults {defaults} differ from demo.env {value!r}"
    # Every interpolated variable of the compose file has a default and is documented in demo.env.
    variables = set(re.findall(r"\$\{([A-Z_]+)(:-[^}]*)?\}", text))
    assert all(default for _name, default in variables), sorted(name for name, default in variables if not default)
    assert {name for name, _default in variables} == set(demo_env)


def test_demo_compose_is_a_separate_stack(compose):
    assert compose["name"] == "loadlens-demo"
    services = compose["services"]
    assert set(services) == {"timescaledb", "llm-stub", "prom-stub", "app"}
    for name, service in services.items():
        assert service["profiles"] == ["demo"], name
        assert service["container_name"].startswith("loadlens_demo_"), name
        for dependency in service.get("depends_on", {}):
            assert dependency in services, (name, dependency)
    assert services["app"]["ports"] == ["127.0.0.1:${APP_PORT:-5000}:5000"]
    assert "ports" not in services["timescaledb"]
    assert set(compose["volumes"]) == {"tsdb_data"}


def test_demo_compose_app_uses_the_demo_settings_and_the_thread_runner(compose):
    app = compose["services"]["app"]
    mounts = {m["target"]: m for m in app["volumes"] if isinstance(m, dict)}
    assert mounts["/app/settings.py"]["source"] == "../settings.demo.py" and mounts["/app/settings.py"]["read_only"] is True
    assert mounts["/app/metrics_config.py"]["source"] == "../metrics_config.example.py"
    assert "/app/settings_runtime.json" not in mounts and "/app/metrics_config_runtime.json" not in mounts
    env = app["environment"]
    assert env["LOADLENS_DEMO_SEED"] == "1"
    assert env["LOADLENS_JOB_RUNNER"] == "thread" and env["CELERY_TASK_ALWAYS_EAGER"] == "1"
    assert env["LOADLENS_AUTH_ENABLED"] == "1"
    assert env["LOADLENS_ADMIN_USER"] == "${DEMO_ADMIN_USER:-admin}"
    assert env["DEMO_DB_HOST"] == "timescaledb" and env["DEMO_LLM_URL"] == "http://llm-stub:5090/v1"
    assert APP_ENV_VARS <= set(env)
    assert not (set(env) - APP_ENV_VARS - {name for name in env if name.startswith("DEMO_")})


def test_demo_compose_env_names_match_what_the_application_reads(compose):
    from loadlens_app.auth import config as auth_config
    from loadlens_app import celery_app, demo_data, migrations

    read_by_app = set()
    for module in (auth_config, celery_app, demo_data, migrations):
        read_by_app |= set(re.findall(r"\b(LOADLENS_[A-Z_]+|CELERY_[A-Z_]+)\b", Path(module.__file__).read_text(encoding="utf-8")))
    read_by_app |= set(re.findall(r"\b(LOADLENS_[A-Z_]+)\b", (ROOT_DIR / "app.py").read_text(encoding="utf-8")))
    env = compose["services"]["app"]["environment"]
    unknown = {name for name in env if name.startswith(("LOADLENS_", "CELERY_"))} - read_by_app
    assert unknown == set(), unknown
    assert demo_data.DEMO_SEED_ENV in env


def test_demo_stubs_listen_on_every_interface_of_their_container(compose):
    assert compose["services"]["llm-stub"]["command"] == "python tools/stubs/llm_stub.py --host 0.0.0.0 --port 5090 --lang en --delay ${DEMO_LLM_DELAY_SEC:-1}"
    assert compose["services"]["prom-stub"]["command"].startswith("python tools/stubs/prom_stub.py --host 0.0.0.0 --port 5092")
    for name in ("llm-stub", "prom-stub"):
        assert compose["services"][name]["healthcheck"]["test"][0] == "CMD"


@pytest.mark.parametrize("stub", ["llm_stub", "prom_stub"])
def test_stubs_accept_a_container_host_and_refuse_others(stub):
    spec = importlib.util.spec_from_file_location(f"demo_{stub}", ROOT_DIR / "tools" / "stubs" / f"{stub}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.parse_args(["--host", "0.0.0.0", "--port", "0"]).host == "0.0.0.0"
    assert module.parse_args(["--port", "0"]).host == "127.0.0.1"
    with pytest.raises(SystemExit):
        module.parse_args(["--host", "10.0.0.5", "--port", "0"])


def test_demo_workflow_runs_the_documented_command():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert DEMO_COMMAND in text and "--env-file docker/demo.env" in text
    assert "down -v" in text and "/healthz" in text and "/runs" in text
    assert "timeout-minutes: 15" in text
    assert "docker compose config" not in text  # would print the interpolated environment


def test_docker_setup_documents_the_demo_command():
    text = (ROOT_DIR / "docker" / "DOCKER_SETUP.md").read_text(encoding="utf-8")
    assert DEMO_COMMAND in text and "settings.demo.py" in text and "demo.env" in text


# ---- the seed hook ----------------------------------------------------------------------------------------------


def test_demo_seed_is_off_without_the_variable(monkeypatch):
    from loadlens_app import demo_data

    monkeypatch.delenv(demo_data.DEMO_SEED_ENV, raising=False)
    assert demo_data.demo_seed_enabled() is False
    assert demo_data.demo_seed_enabled({}) is False
    for value in ("1", "true", "YES", " on "):
        assert demo_data.demo_seed_enabled({demo_data.DEMO_SEED_ENV: value}) is True
    for value in ("0", "false", "", "no"):
        assert demo_data.demo_seed_enabled({demo_data.DEMO_SEED_ENV: value}) is False


def test_create_app_starts_the_seed_only_with_the_variable(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app import demo_data

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    started = []
    monkeypatch.setattr(demo_data, "start_background_demo_seed", lambda: started.append(True) or True)
    monkeypatch.delenv(demo_data.DEMO_SEED_ENV, raising=False)
    create_app()
    assert started == []
    monkeypatch.setenv(demo_data.DEMO_SEED_ENV, "1")
    create_app()
    assert started == [True]


def test_start_background_demo_seed_runs_once_per_process(monkeypatch):
    from loadlens_app import demo_data

    monkeypatch.setattr(demo_data, "_seed_started", False)
    threads = []

    class FakeThread:
        def __init__(self, target, name, daemon):
            threads.append((target, name, daemon))

        def start(self):
            pass

    monkeypatch.setattr(demo_data.threading, "Thread", FakeThread)
    assert demo_data.start_background_demo_seed() is True
    assert demo_data.start_background_demo_seed() is False
    assert threads == [(demo_data.run_demo_seed_with_retries, "loadlens-demo-seed", True)]


def test_ensure_demo_is_idempotent(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from loadlens_app import demo_data

    now = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    calls = {"seed": [], "bootstrap": [], "report": [], "invalidate": 0}
    monkeypatch.setattr(demo_data, "seed_demo_run", lambda when=None, report_language=None: calls["seed"].append(when) or {})
    monkeypatch.setattr(demo_data, "_bootstrap_service_configs", lambda area, service: calls["bootstrap"].append((area, service)))
    monkeypatch.setattr(demo_data, "start_demo_report", lambda when=None: calls["report"].append(when) or True)
    from loadlens_app import response_cache

    monkeypatch.setattr(response_cache, "invalidate", lambda *prefixes: calls.__setitem__("invalidate", calls["invalidate"] + 1))

    # First start: nothing stored yet.
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: False)
    assert demo_data.ensure_demo(now) == {"seeded": True, "report_started": True}
    assert calls["seed"] == [now - timedelta(days=1)] and calls["report"] == [now] and calls["bootstrap"] == []
    assert calls["invalidate"] == 1

    # Every later start: the run exists, its area is registered again, nothing else is written.
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: True)
    monkeypatch.setattr(demo_data, "start_demo_report", lambda when=None: False)
    assert demo_data.ensure_demo(now) == {"seeded": False, "report_started": False}
    assert calls["seed"] == [now - timedelta(days=1)] and calls["bootstrap"] == [("demo", "demo-checkout")]
    assert calls["invalidate"] == 1


def test_start_demo_report_queues_the_stub_report_once(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from loadlens_app import demo_data
    from loadlens_app.blueprints import dashboard
    from loadlens_app.jobs import JOB_STATUS_RUNNING, job_store

    now = datetime(2026, 10, 19, 12, 0, 30, tzinfo=timezone.utc)
    started = []
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda run_name, service: False)
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service, area: area)
    monkeypatch.setattr(dashboard, "start_report_job", lambda *args, **kwargs: started.append((args, kwargs)) or SimpleNamespace(job_id="j1"))
    monkeypatch.setattr(job_store, "list", lambda statuses=None, limit=20: [])
    assert demo_data.start_demo_report(now) is True
    (start, end, service, area), kwargs = started[0]
    assert (end - start) == demo_data.DEMO_REPORT_WINDOW_MIN * 60_000 and end == 1_792_411_200_000
    assert service == "demo-checkout" and area == "demo"
    assert kwargs == {"use_llm": True, "save_to_db": True, "web_only": True, "run_name": "demo-checkout-step-2", "test_type": "step", "owner": "demo"}

    # Stored already, or being built right now: nothing is queued.
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda run_name, service: True)
    assert demo_data.start_demo_report(now) is False
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda run_name, service: False)
    running = SimpleNamespace(run_name="demo-checkout-step-2", service="demo-checkout", status=JOB_STATUS_RUNNING)
    monkeypatch.setattr(job_store, "list", lambda statuses=None, limit=20: [running])
    assert demo_data.start_demo_report(now) is False
    assert len(started) == 1


def test_run_demo_seed_with_retries_waits_for_the_database():
    from AI.storage_state import StorageUnavailableError
    from loadlens_app import demo_data

    attempts = []
    sleeps = []

    def run():
        attempts.append(1)
        if len(attempts) < 3:
            raise StorageUnavailableError("db starting")
        return {"seeded": True, "report_started": True}

    assert demo_data.run_demo_seed_with_retries(run, sleeps.append) == {"seeded": True, "report_started": True}
    assert sleeps == [5, 10]

    # A bug is not retried; a race with another seeder is not an error.
    def broken():
        raise ValueError("bad config")

    assert demo_data.run_demo_seed_with_retries(broken, sleeps.append) is None
    assert sleeps == [5, 10]

    def exists():
        raise demo_data.DemoAlreadyExistsError("demo-checkout-step", "demo-checkout")

    assert demo_data.run_demo_seed_with_retries(exists, sleeps.append) == {"seeded": False, "report_started": False}
    assert demo_data.run_demo_seed_with_retries(lambda: (_ for _ in ()).throw(StorageUnavailableError("x")), sleeps.append, max_attempts=2) is None


# ---- the banner ---------------------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    import update_page
    from app import create_app

    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_banner_keys_exist_in_both_languages():
    keys = {"dashboard.demo_banner.title", "dashboard.demo_banner.text", "dashboard.demo_banner.open_archive", "dashboard.demo_banner.dismiss"}
    for lang in ("ru", "en"):
        data = json.loads((ROOT_DIR / "locales" / lang / "dashboard.json").read_text(encoding="utf-8"))
        assert keys <= set(data), lang
        assert all(data[key] for key in keys)


def test_demo_banner_renders_in_english_without_cyrillic(client):
    use_locale(client, "en")
    html = client.get("/").get_data(as_text=True)
    banner = re.search(r'<section id="demoBanner" class="banner" hidden>(.*?)</section>', html, re.S)
    assert banner, "the demo banner is missing from the dashboard"
    text = visible_text(banner.group(1))
    assert_no_cyrillic(text)
    assert "Demo data" in text and "sample runs" in text and "Got it" in text
    assert 'id="demoBannerDismiss"' in banner.group(1) and 'href="/reports"' in banner.group(1)


def test_demo_banner_stays_russian_without_a_cookie(client):
    html = client.get("/").get_data(as_text=True)
    assert '<div class="banner-title">Демо-данные</div>' in html


def test_demo_status_endpoint(client, monkeypatch):
    from loadlens_app import demo_data

    monkeypatch.delenv(demo_data.DEMO_SEED_ENV, raising=False)
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: (_ for _ in ()).throw(AssertionError("no database read without the variable")))
    response = client.get("/demo/status")
    assert response.status_code == 200
    assert response.get_json() == {"enabled": False, "exists": None, "run_name": "demo-checkout-step", "report_url": "/reports/demo-checkout/demo-checkout-step"}

    monkeypatch.setenv(demo_data.DEMO_SEED_ENV, "1")
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: run_name == "demo-checkout-step")
    assert client.get("/demo/status").get_json()["exists"] is True
    monkeypatch.setattr(demo_data, "_run_exists", lambda run_name: (_ for _ in ()).throw(RuntimeError("down")))
    assert client.get("/demo/status").get_json() == {"enabled": True, "exists": None, "run_name": "demo-checkout-step", "report_url": "/reports/demo-checkout/demo-checkout-step"}


def test_demo_status_is_readable_by_every_signed_in_role():
    from loadlens_app.auth import policy
    from loadlens_app.auth.models import Role

    rule = policy.rule_for("dashboard.demo_status_view")
    assert rule.role is Role.VIEWER and not rule.page and not rule.session_only
