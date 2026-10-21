# SPDX-License-Identifier: Apache-2.0
"""Localization of the load test backend (namespace ``lt``): request texts, stored watcher messages, status titles."""

import ast
import copy
import json
import re
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from AI.i18n import Msg, msg, with_locale
from loadlens_app import core
from loadlens_app.load_tests import ci_params, executors, live, service, store, watcher
from loadlens_app.load_tests.executors import GeneratorError, LaunchResult, Probe
from loadlens_app.load_tests.generators import GENERATOR_TYPE_TITLES, GeneratorConfigError, configured_generators, parse_generator
from loadlens_app.load_tests.messages import LtError, TEXT_KEY, stored_text
from loadlens_app.load_tests.store import InstanceRecord, MemoryLoadTestStore, RunRecord
from loadlens_app.load_tests.tools import DEFAULT_TOOLS, ToolInputError, effective_tools, tools_overrides
from tests.i18n_helpers import CYRILLIC, use_locale

LOCALES = ROOT_DIR / "locales"
KEY = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
SCANNED = sorted((ROOT_DIR / "loadlens_app" / "load_tests").rglob("*.py")) + [ROOT_DIR / "loadlens_app" / "blueprints" / "load_tests.py"]

GENERATORS = {
    "gen-1": {
        "title": "Gen 1",
        "type": "ssh",
        "ssh": {"host": "10.0.0.5", "port": 22, "username": "loadlens", "private_key": "-----BEGIN KEY-----\nabc\n-----END KEY-----",
                "host_key": "ssh-ed25519 AAAA"},
        "work_dir": "/opt/loadlens/runs",
        "scripts_dir": "/opt/tests",
        "tools": {},
    },
    "hand": {"title": "By hand", "type": "manual"},
}
FORM = {
    "service": "checkout",
    "run_name": "nightly",
    "instances": [{"name": "rest", "generator": "gen-1", "tool": "k6", "script": "a.js", "params": "--vus 5"}],
}


def _messages(lang):
    out = {}
    for path in sorted((LOCALES / lang).glob("*.json")):
        out.update(json.loads(path.read_text(encoding="utf-8")))
    return out


def _no_cyrillic(value):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    assert not CYRILLIC.search(text), text


# ---- dictionary coverage ----------------------------------------------------------------------------------


def test_every_lt_key_constant_of_the_feature_exists_in_both_languages():
    """Keys reach the errors and the watcher as plain strings (``GeneratorError("lt....")``), not only via t().

    Two-segment ``lt.*`` constants are query pack roles (``lt.rps_total``), except the text carrier ``lt.text``.
    """
    ru, en = _messages("ru"), _messages("en")
    found = set()
    for path in SCANNED:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and KEY.match(node.value) \
                    and node.value.startswith(("lt.", "common.lt_")) and (node.value.count(".") >= 2 or node.value == TEXT_KEY):
                found.add(node.value)
    assert len(found) > 300
    missing = sorted(key for key in found if key not in ru or key not in en)
    assert not missing, missing


def test_status_titles_follow_the_locale():
    with with_locale("en"):
        assert store.RUN_STATUS_TITLES["running"] == "Running"
        assert store.RUN_STATUS_TITLES.get("failed") == "Failed to start"
        assert store.INSTANCE_STATUS_TITLES["timed_out"] == "Stopped by the duration limit"
        assert store.RUN_STATUS_TITLES.get("nope", "nope") == "nope"
        assert GENERATOR_TYPE_TITLES["manual"] == "Manual" and GENERATOR_TYPE_TITLES["gitlab"] == "GitLab CI"
        assert service.CI_TITLES["gitlab"] == "GitLab pipeline"
    assert store.RUN_STATUS_TITLES["running"] == "Идёт" and store.INSTANCE_STATUS_TITLES["pending"] == "Ожидает старта"
    assert GENERATOR_TYPE_TITLES["manual"] == "Вручную"
    assert set(store.RUN_STATUS_TITLES) == set(store.RUN_STATUSES)


# ---- errors -------------------------------------------------------------------------------------------------


def test_launch_error_renders_in_the_language_of_the_reader():
    error = service.LaunchError("lt.launch.instance_error", label=msg("lt.launch.instance", n=2),
                                error=ToolInputError("lt.tool_input.params_control"), instance=2)
    with with_locale("en"):
        assert str(error) == "Instance 2: The parameters contain control characters"
    assert str(error) == "Инстанс 2: Параметры содержат управляющие символы"
    assert (error.status, error.code, error.instance) == (400, "invalid", 2)
    assert error.ref()["key"] == "lt.launch.instance_error"
    # A ready text (tests, third-party messages) is shown as it is; another localized error keeps its key.
    assert str(service.LaunchError("as is")) == "as is" and service.LaunchError("as is").key == TEXT_KEY
    wrapped = service.LaunchError(GeneratorError("lt.ssh.auth_failed"), status=502)
    with with_locale("en"):
        assert str(wrapped) == "SSH refused the login: check the user, key or password"


def test_errors_of_other_modules_keep_their_key():
    class ForeignError(Exception):
        def ref(self):
            return {"key": "lt.launch.run_finished", "params": {}}

    with with_locale("en"):
        assert str(service.LaunchError(ForeignError("Прогон уже завершён"), status=429)) == "The run has already finished"
    assert str(LtError(ValueError("plain"))) == "plain"


def test_settings_errors_in_english():
    with with_locale("en"):
        with pytest.raises(GeneratorConfigError) as caught:
            parse_generator("g1", {"type": "ssh", "ssh": {"host": "h", "username": "u", "password": "p", "port": 70000},
                                   "work_dir": "/w"})
        assert str(caught.value) == "Generator “g1”: the port is outside 1–65535"
        with pytest.raises(GeneratorConfigError) as caught:
            parse_generator("g1", {"type": "docker", "docker": {"url": "ftp://x"}})
        assert str(caught.value) == "Generator “g1”: the Docker API address must look like https://host:port"
    with pytest.raises(GeneratorConfigError, match="порт вне диапазона"):
        parse_generator("g1", {"type": "ssh", "ssh": {"host": "h", "username": "u", "password": "p", "port": 70000}, "work_dir": "/w"})


# ---- built-in tool texts ------------------------------------------------------------------------------------


def test_built_in_tool_texts_follow_the_reader_and_settings_text_stays():
    tools = effective_tools({"load_tools": {"k6": {"help": "Our own help"}}})
    with with_locale("en"):
        assert tools["gatling"].display_title == "Gatling (bundle)"
        _no_cyrillic(tools["gatling"].to_public())
        assert tools["k6"].display_help == "Our own help"
    assert tools["gatling"].display_title == "Gatling (бандл)" == DEFAULT_TOOLS["gatling"]["title"]
    # A form that sends a built-in text back in English stores no override for it.
    payload = copy.deepcopy(DEFAULT_TOOLS)
    with with_locale("en"):
        payload["jmeter"]["help"] = effective_tools({})["jmeter"].display_help
    payload["jmeter"]["grace_seconds"] = 90
    assert tools_overrides(payload) == {"jmeter": {"grace_seconds": 90}}


# ---- watcher messages ---------------------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc).timestamp()

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Executor:
    enforces_deadline = True
    supports_signals = True
    supports_control = False

    def __init__(self):
        self.fail = None

    def launch(self, plan):
        if self.fail:
            raise self.fail
        return LaunchResult(ref=plan.ref)

    def probe(self, ref):
        return Probe(state="running")

    def logs(self, ref, max_bytes=0):
        return ""

    def cleanup(self, ref):
        return None


@pytest.fixture
def world(monkeypatch):
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    fake = _Executor()
    clock = _Clock()
    catalog = service.Catalog(tools=effective_tools({}), generators=configured_generators({"load_generators": GENERATORS}))
    w = watcher.Watcher(store_getter=lambda: memory, catalog_loader=lambda: catalog, executor_getter=lambda gen: fake,
                        report_starter=lambda *a, **k: types.SimpleNamespace(job_id="job"), clock=clock)
    return types.SimpleNamespace(store=memory, fake=fake, clock=clock, watcher=w, catalog=catalog)


def _run(world, instances, **options):
    run = RunRecord(
        run_id=f"r{len(world.store.list(limit=100)) + 1}", run_name="nightly", run_tag="nightly", service="checkout",
        project_area="shop", owner="eng", options={"auto_report": False, "use_llm": False, "max_seconds": 3600, **options},
        instances=instances, created_at=datetime.fromtimestamp(world.clock(), timezone.utc),
    )
    world.store.create(run)
    return run.run_id


def test_watcher_messages_are_stored_with_refs_and_rendered_per_reader(world, monkeypatch):
    # The watcher has no reader: it writes in the language of the installation (en here) and keeps the refs.
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    run_id = _run(world, [InstanceRecord(index=0, name="hand", tag="hand", generator="hand")])
    world.watcher.tick()
    run = world.store.get(run_id)
    inst = run.instances[0]
    assert inst.message.startswith("Running: start the tool yourself")
    assert inst.message_ref == {"key": "lt.watcher.manual_running", "params": {}}
    assert run.message == "1 of 1 running"
    assert run.message_ref == {"key": "lt.watcher.run.progress", "params": {"running": 1, "total": 1}}

    with with_locale("ru"):
        view = service.run_view(run, world.catalog, with_report=False)
    assert view["message"] == "Идёт 1 из 1" and view["status_title"] == "Идёт"
    assert view["instances"][0]["message"].startswith("Идёт: запустите инструмент сами")
    assert view["instances"][0]["status_title"] == "Идёт"
    with with_locale("en"):
        view = service.run_view(run, world.catalog, with_report=False)
    assert view["message"] == "1 of 1 running" and view["instances"][0]["status_title"] == "Running"

    world.store.request_stop(run_id)
    world.watcher.tick()
    run = world.store.get(run_id)
    assert run.status == store.RUN_DONE and run.message == "Finished, no report was requested"
    with with_locale("ru"):
        view = service.run_view(run, world.catalog, with_report=False)
    assert view["message"] == "Завершён, отчёт не заказывали" and view["instances"][0]["message"] == "Остановлен вручную"


def test_a_generator_error_is_kept_as_a_nested_ref(world):
    world.fake.fail = GeneratorError("lt.ssh.auth_failed")
    run_id = _run(world, [InstanceRecord(index=0, name="rest", tag="rest", generator="gen-1", tool="k6", script="/opt/tests/a.js")])
    world.watcher.tick()
    run = world.store.get(run_id)
    inst = run.instances[0]
    assert inst.status == store.INST_FAILED
    assert inst.message == "Не запустился: SSH отказал во входе: проверьте пользователя, ключ или пароль"
    assert inst.message_ref == {"key": "lt.watcher.not_started", "params": {"error": {"key": "lt.ssh.auth_failed", "params": {}}}}
    assert run.status == store.RUN_FAILED and run.message.startswith("Ни один инстанс не запустился: Не запустился")
    with with_locale("en"):
        view = service.run_view(run, world.catalog, with_report=False)
    assert view["instances"][0]["message"] == "Did not start: SSH refused the login: check the user, key or password"
    assert view["message"] == "No instance started: Did not start: SSH refused the login: check the user, key or password"
    assert view["status_title"] == "Failed to start"

    # Third-party text goes into the parameters as it is.
    world.fake.fail = GeneratorError("connection reset by peer")
    run_id = _run(world, [InstanceRecord(index=0, name="rest", tag="rest", generator="gen-1", tool="k6", script="/opt/tests/a.js")])
    world.watcher.tick()
    inst = world.store.get(run_id).instances[0]
    with with_locale("en"):
        assert stored_text(inst.message, inst.message_ref) == "Did not start: connection reset by peer"


def test_status_text_of_k6_is_a_message():
    status = watcher.summarize_status('vus: "12"\nvus-max: "20"\npaused: "true"')
    assert status == "VU 12 из 20, пауза" and isinstance(status, Msg)
    with with_locale("en"):
        assert status.render() == "VU 12 of 20, paused"
    assert watcher.summarize_status("custom line\nmore") == "custom line"


def test_messages_without_ref_are_shown_as_stored(world):
    run = RunRecord(run_id="old", run_name="old", run_tag="old", service="checkout", message="Строится отчёт",
                    instances=[InstanceRecord(index=0, name="a", tag="a", generator="hand", message="Запущен")])
    with with_locale("en"):
        view = service.run_view(run, world.catalog, with_report=False)
    assert view["message"] == "Строится отчёт" and view["instances"][0]["message"] == "Запущен"


# ---- store --------------------------------------------------------------------------------------------------


def test_refs_survive_the_json_of_instances_and_the_row_of_runs():
    ref = {"key": "lt.watcher.start_in", "params": {"seconds": 5}}
    inst = InstanceRecord(index=0, name="a", tag="a", generator="g", message="Старт через 5 с", message_ref=ref)
    back = InstanceRecord.from_json(json.loads(json.dumps(inst.to_json())))
    assert back.message_ref == ref and back.status_text_ref is None
    assert InstanceRecord.from_json({**inst.to_json(), "message_ref": {"bad": 1}}).message_ref is None
    assert InstanceRecord.from_json({k: v for k, v in inst.to_json().items() if k != "message_ref"}).message_ref is None

    columns = store._COLUMNS + (store._REF_COLUMN,)
    row = {name: None for name in columns}
    row.update(run_id="r", run_name="n", run_tag="t", service="s", status="done", message="Отчёт готов",
               message_ref=json.dumps({"key": "lt.watcher.report.ready", "params": {}}))
    record = store._from_row(tuple(row[name] for name in columns), columns)
    assert record.message_ref == {"key": "lt.watcher.report.ready", "params": {}}
    assert store._from_row(tuple(row[name] for name in store._COLUMNS)).message_ref is None


class _Cursor:
    def __init__(self, present, alter_fails):
        self.present = present
        self.alter_fails = alter_fails
        self.sql = []
        self._rows = []

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else repr(query)
        self.sql.append(text)
        if "information_schema.columns" in text and "message_ref" in str(params):
            self._rows = [(1,)] if self.present else []
        elif "ADD COLUMN IF NOT EXISTS message_ref" in text and self.alter_fails:
            import psycopg2

            raise psycopg2.ProgrammingError("must be owner of table load_test_runs")
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


@pytest.mark.parametrize("present, alter_fails, expected, altered", [
    (True, False, True, False),
    (False, False, True, True),
    (False, True, False, True),
])
def test_the_ref_column_is_added_when_the_role_may(present, alter_fails, expected, altered):
    cursor = _Cursor(present, alter_fails)
    assert store.PgLoadTestStore._ensure_ref_column(cursor, "public", "load_test_runs") is expected
    assert any("ADD COLUMN IF NOT EXISTS message_ref" in sql for sql in cursor.sql) is altered
    pg = store.PgLoadTestStore()
    pg._refs = expected
    assert ("message_ref" in pg._columns()) is expected


# ---- request texts ------------------------------------------------------------------------------------------


def test_chart_events_and_ci_notes_in_the_reader_language():
    started = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)
    run = RunRecord(run_id="r", run_name="n", run_tag="t", service="s", started_at=started, stop_all_at=started + timedelta(minutes=5),
                    options={"max_seconds": 600},
                    instances=[InstanceRecord(index=0, name="i0", tag="i0", generator="g", status="finished", started_at=started,
                                              finished_at=started + timedelta(minutes=4))])
    with with_locale("en"):
        labels = [event["label"] for event in live.run_events(run)]
    assert labels == ["i0: start", "i0: finished", "run stop", "duration limit"]

    # Parameters are cached for a minute across readers: the notes keep their keys.
    with with_locale("ru"):
        params = ci_params.CiParams(notes=[msg("lt.ci.note.no_api_token")],
                                    params=[ci_params.CiParam(name="ENV", kind="parameter", note=msg("lt.ci.note.options_from_settings"))])
    with with_locale("en"):
        public = params.to_public()
    assert public["notes"][0].startswith("The GitLab API token is not set") and public["params"][0]["note"] == "Options from the generator settings"


def test_load_windows_messages_in_english(monkeypatch):
    monkeypatch.setattr(live, "finished_runs", lambda service_id, start, end: [])
    monkeypatch.setattr(live, "_service_config", lambda area, service_id: (_ for _ in ()).throw(RuntimeError("boom")))
    with with_locale("en"):
        data = live.load_windows("shop", "checkout", 6)
    assert data["message"] == "The service settings cannot be read" and data["detail"] == "boom"


@pytest.fixture
def env(monkeypatch, tmp_path):
    from loadlens_app.blueprints import config_api, dashboard
    from loadlens_app.blueprints import load_tests as blueprint

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setitem(core.CONFIG, "load_generators", copy.deepcopy(GENERATORS))
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setitem(core.CONFIG, "limits", {})
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: False)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(service, "_installed_lt_packs", lambda area, service_id: ["lt-k6-prometheus-rw"])
    monkeypatch.setattr(executors, "executor_factory", lambda generator: _Executor())
    monkeypatch.setattr(config_api, "_bootstrap_service_configs", lambda area, service_id: None)
    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(blueprint, "_find_area_for_service", lambda service_id: "")
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), store=memory)


def test_api_answers_in_english(env):
    use_locale(env.client, "en")
    options = env.client.get("/load_tests/options").get_json()
    tools = {tool["id"]: tool for tool in options["tools"]}
    assert tools["gatling"]["title"] == "Gatling (bundle)"
    assert {g["id"]: g["type_title"] for g in options["generators"]} == {"gen-1": "SSH", "hand": "Manual"}
    _no_cyrillic(options)

    bad = {**FORM, "instances": [{"generator": "gen-1", "tool": "k6", "script": "a.js", "params": "'x"}]}
    resp = env.client.post("/load_tests/preview", json=bad)
    data = resp.get_json()
    assert resp.status_code == 400 and data["instance"] == 1
    assert data["message"].startswith("Instance 1: The parameters cannot be parsed: check that the quotes are paired")
    assert env.client.post("/load_tests/preview", json={**FORM, "service": ""}).get_json()["error"] == "Choose a service"
    assert env.client.get("/load_tests/runs/nope").get_json()["error"] == "Run not found"
    assert env.client.get("/load_tests/runs", query_string={"limit": "x"}).get_json()["error"] == "limit must be a whole number"
    assert env.client.get("/lt_windows").get_json()["error"] == "Specify a service"
    assert env.client.get("/lt_windows", query_string={"service": "nope"}).get_json()["error"] == \
        "Service “nope” is not found in the settings"
    assert env.client.get("/load_tests/scripts", query_string={"generator": "x", "tool": "k6"}).get_json()["error"] == \
        "Generator or tool not found"

    # A message written at launch keeps its ref: each reader sees it in the own language.
    created = env.client.post("/load_tests", json=FORM).get_json()
    view = env.client.get(f"/load_tests/runs/{created['run_id']}").get_json()
    assert view["message"] == "Waiting to launch" and view["status_title"] == "Starting"
    assert view["instances"][0]["status_title"] == "Waiting to start"
    _no_cyrillic(view)
    use_locale(env.client, None)
    view = env.client.get(f"/load_tests/runs/{created['run_id']}").get_json()
    assert view["message"] == "Ожидает запуска" and view["status_title"] == "Запускается"


def test_active_runs_are_cached_per_language(env, monkeypatch):
    monkeypatch.setenv("LOADLENS_RESPONSE_CACHE", "1")
    env.client.post("/load_tests", json=FORM)
    use_locale(env.client, "en")
    assert env.client.get("/load_tests/active").get_json()["runs"][0]["status_title"] == "Starting"
    use_locale(env.client, "ru")
    assert env.client.get("/load_tests/active").get_json()["runs"][0]["status_title"] == "Запускается"
