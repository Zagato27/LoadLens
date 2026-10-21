# SPDX-License-Identifier: Apache-2.0
"""English UI of the new report, launch and run pages (plans/i18n-plan.md, section 2.5, row 5).

The pages render without Russian text in en (their own content: the <title> and <main>; the frame of base.html
is checked by its own tests), the JS builders of the pages produce English strings, ru stays as it was, and the
server answers carry the fields the pages react to instead of parsing Russian messages.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from loadlens_app import core  # noqa: E402
from loadlens_app.load_tests import executors, service, store  # noqa: E402
from loadlens_app.load_tests.store import MemoryLoadTestStore  # noqa: E402
from tests.i18n_helpers import CYRILLIC, assert_no_cyrillic, use_locale, visible_text  # noqa: E402
from tests.js_harness import run_js  # noqa: E402

BASE = ["static/js/common.js", "static/js/ui.js"]
PICKER = BASE + ["static/js/service_picker.js"]
NEW = BASE + ["static/js/new.js"]
LAUNCH = BASE + ["static/js/launch.js"]
RUN_PAGE = BASE + ["static/js/launch_run.js"]
GENERATORS = {
    "gen-1": {
        "title": "Gen 1",
        "type": "ssh",
        "ssh": {"host": "10.0.0.5", "port": 22, "username": "loadlens", "private_key": "-----BEGIN KEY-----\nabc\n-----END KEY-----", "host_key": "ssh-ed25519 AAAA"},
        "work_dir": "/opt/loadlens/runs",
        "scripts_dir": "/opt/tests",
        "tools": {},
    },
}
FORM = {
    "service": "checkout",
    "run_name": "nightly",
    "instances": [{"name": "rest", "generator": "gen-1", "tool": "k6", "script": "a.js", "params": "--vus 5"}],
}


class _FakeExecutor:
    def logs(self, ref, max_bytes=0):
        return ""

    def list_scripts(self, scripts_dir, tool):
        return []


@pytest.fixture
def env(monkeypatch, tmp_path):
    import update_page
    from app import create_app
    from loadlens_app.blueprints import config_api, dashboard
    from loadlens_app.blueprints import load_tests as blueprint

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", runtime)
    monkeypatch.setitem(core.CONFIG, "load_generators", copy.deepcopy(GENERATORS))
    monkeypatch.setitem(core.CONFIG, "load_tools", {})
    monkeypatch.setitem(core.CONFIG, "limits", {})
    memory = MemoryLoadTestStore()
    monkeypatch.setattr(store, "_store", memory)
    taken = set()
    monkeypatch.setattr(dashboard, "resolve_report_target", lambda service_id, area="": "shop")
    monkeypatch.setattr(dashboard, "_run_name_taken", lambda name, service_id: name in taken)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(blueprint, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(service, "_installed_lt_packs", lambda area, service_id: ["lt-k6-prometheus-rw"])
    monkeypatch.setattr(executors, "executor_factory", lambda generator: _FakeExecutor())
    monkeypatch.setattr(config_api, "_bootstrap_service_configs", lambda area, service_id: None)
    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), store=memory, taken=taken)


def _html_lang(html: str) -> str:
    return re.search(r'<html lang="([^"]*)"', html).group(1)


def _own_text(html: str) -> str:
    """Visible text of what the page itself renders: its <title> and the <main> block."""
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
    main = re.search(r'<main id="main".*?</main>', html, re.S).group(0)
    return title + "\n" + visible_text(main)


def _js(value):
    return json.dumps(value, ensure_ascii=False)


# ---- pages ------------------------------------------------------------------------------------------


def test_pages_render_in_english(env):
    run_id = env.client.post("/load_tests", json=FORM).get_json()["run_id"]
    use_locale(env.client, "en")
    for url, title in (("/new", "New report"), ("/launch", "Run a test"), (f"/launch/{run_id}", "Test run")):
        resp = env.client.get(url)
        assert resp.status_code == 200, url
        html = resp.get_data(as_text=True)
        assert _html_lang(html) == "en", url
        assert f"<title>{title}</title>" in html, url
        assert_no_cyrillic(_own_text(html))


def test_page_dictionaries_carry_the_page_namespaces(env):
    use_locale(env.client, "en")
    new_page = env.client.get("/new").get_data(as_text=True)
    assert '"new.form.submit":"Create report"' in new_page and '"picker.placeholder":"Start typing a service name"' in new_page
    launch_page = env.client.get("/launch").get_data(as_text=True)
    assert '"launch.instance.title":"Instance {n}"' in launch_page and '"picker.recent":"recent"' in launch_page
    run_page = env.client.get("/launch/abc").get_data(as_text=True)
    assert '"run.fresh.no_data":"no data yet"' in run_page


def test_run_page_spinner_text_follows_the_locale(env):
    use_locale(env.client, "en")
    page = env.client.get("/launch/abc").get_data(as_text=True)
    assert page.count('data-loading-text="Loading metrics…"') == 3
    assert "content: attr(data-loading-text)" in page
    style = next(block for block in re.findall(r"<style>.*?</style>", page, re.S) if "live-spin" in block)
    assert not CYRILLIC.search(style)


def test_russian_pages_keep_their_text(env):
    new_page = env.client.get("/new").get_data(as_text=True)
    assert "<title>Создание отчёта</title>" in new_page and _html_lang(new_page) == "ru"
    assert '<span class="type-title">Ступенчатый (max perf)</span>' in new_page
    assert 'placeholder="Напр.: release-2.5-step; иначе run-ГГГГММДД-ЧЧММСС"' in new_page
    launch_page = env.client.get("/launch").get_data(as_text=True)
    assert '<option value="soak">Долговременный (soak)</option>' in launch_page
    assert "Администратор добавляет их в настройках, раздел «Подключения» → «Генераторы нагрузки»." in launch_page
    run_page = env.client.get("/launch/abc").get_data(as_text=True)
    assert "<th>Генератор и инструмент</th>" in run_page and 'data-loading-text="Загружаем метрики…"' in run_page


# ---- structured server answers ------------------------------------------------------------------------


def test_taken_run_name_comes_with_a_code(env):
    env.taken.add("nightly")
    resp = env.client.post("/create_report", json={"start": "2026-10-15T10:00", "end": "2026-10-15T11:00", "service": "checkout", "run_name": "nightly"})
    data = resp.get_json()
    assert resp.status_code == 400 and data["code"] == "run_name_taken"
    assert "уже существует" in data["message"]  # the text itself is unchanged
    assert run_js(NEW, f"LL.newReport._internals.runNameTaken({_js(data)})") is True
    assert run_js(NEW, "LL.newReport._internals.runNameTaken({message: 'Запуск уже существует'})") is False


def test_instance_errors_name_the_instance(env):
    bad = {**FORM, "instances": FORM["instances"] + [{"generator": "gen-1", "tool": "k6", "script": "b.js", "params": "'x"}]}
    data = env.client.post("/load_tests/preview", json=bad).get_json()
    assert data["instance"] == 2 and data["message"].startswith("Инстанс 2:")
    missing = env.client.post("/load_tests/preview", json={**FORM, "instances": [FORM["instances"][0], {"generator": "nope"}]}).get_json()
    assert missing["instance"] == 2
    # Errors of the form as a whole name no instance.
    assert "instance" not in env.client.post("/load_tests/preview", json={**FORM, "service": ""}).get_json()
    assert run_js(LAUNCH, f"LL.launch._internals.errorInstance({{data: {_js(data)}}})") == 2
    assert run_js(LAUNCH, "LL.launch._internals.errorInstance(new Error('Инстанс 2: x'))") is None
    assert run_js(LAUNCH, "LL.launch._internals.errorInstance({data: {instance: 0}})") is None


def test_plan_run_numbers_the_instance_of_an_error(env):
    with pytest.raises(service.LaunchError) as caught:
        service.plan_run({"service": "s", "instances": [None]}, "me", service.Catalog(tools={}, generators={}))
    assert caught.value.instance == 1
    assert service.LaunchError("x").instance is None


# ---- JS builders in en --------------------------------------------------------------------------------


def test_service_picker_in_english():
    services = [{"id": "a", "title": "Alpha"}, {"id": "b", "title": "b"}]
    options = run_js(PICKER, f"LL.servicePicker._internals.serviceOptions({_js(services)}, ['a'], '')", locale="en")
    assert [o["meta"] for o in options] == ["recent · a", ""]
    assert run_js(PICKER, f"LL.servicePicker._internals.serviceOptions({_js(services)}, ['a'], '')[0].meta") == "недавно · a"


def test_new_report_builders_in_english():
    internals = "LL.newReport._internals"
    now = "Date.parse('2026-10-15T18:00:00Z')"
    # The duration comes from common.js (LL.formatDuration), translated with the frame: a fixed stand-in here.
    stub = "LL.formatDuration = () => '1 h 00 min';"
    label = run_js(NEW, f"{stub} {internals}.windowLabel({{start: Date.parse('2026-10-13T10:00:00Z'), end: Date.parse('2026-10-13T11:00:00Z'), peak: 4.25, name: 'nightly', ongoing: true}}, {now})", locale="en")
    assert label == "nightly · 13/10 10:00–11:00 (1 h 00 min, peak 4.3 RPS, in progress)"
    assert run_js(NEW, f"{internals}.dayText(Date.parse('2026-10-14T12:00:00Z'), {now})", locale="en") == "yesterday"
    assert run_js(NEW, f"{internals}.typicalText(6.4 * 60000)", locale="en") == "usually about 6 min"
    assert run_js(NEW, f"['running', 'error', 'done', 'odd'].map({internals}.jobStatusLabel)", locale="en") == ["Running", "Error", "Done", "odd"]
    job = {"job_id": "j1", "kind": "confluence", "status": "running", "progress": 40, "message": "msg", "service": "checkout", "run_name": "", "page_url": "/x"}
    row = run_js(NEW, f"const r = {internals}.renderJobRow({_js(job)}); r.children.map((c) => [c.textContent].concat(c.children.map((x) => x.textContent)))", locale="en")
    assert row == [["Running"], ["", "(untitled)", "Confluence publication · checkout · 40% — msg"], ["", "Open page"]]
    for text in (label, row):
        assert_no_cyrillic(json.dumps(text, ensure_ascii=False))


def test_new_report_builders_keep_russian():
    internals = "LL.newReport._internals"
    assert run_js(NEW, f"{internals}.dayText(Date.parse('2026-10-13T12:00:00Z'), Date.parse('2026-10-15T18:00:00Z'))") == "13.10"
    job = {"job_id": "j1", "kind": "forecast_confluence", "status": "running", "progress": 5, "message": "m", "run_name": "r", "page_url": "/x"}
    row = run_js(NEW, f"const r = {internals}.renderJobRow({_js(job)}); r.children.map((c) => [c.textContent].concat(c.children.map((x) => x.textContent)))")
    assert row == [["В работе"], ["", "r", "публикация прогноза в Confluence · 5% — m"], ["", "Открыть страницу"]]


def test_launch_builders_in_english():
    internals = "LL.launch._internals"
    assert run_js(LAUNCH, f"{internals}.instanceSummary(0, {{name: 'rest', generator: 'Gen 1', tool: 'k6'}})", locale="en") == "Instance 1 · rest · Gen 1 · k6"
    assert run_js(LAUNCH, f"[{internals}.ciKindTitle('gitlab'), {internals}.ciKindTitle('jenkins'), {internals}.ciKindTitle('x')]", locale="en") == ["GitLab pipeline", "Jenkins build", ""]
    setup = (
        f"const s = {internals}.state;"
        " s.generators = {g1: {title: 'Gen 1', host: 'h1', type: 'ssh', type_title: 'SSH', verified: false, busy_runs: ['nightly'], tools: ['k6']},"
        " m: {title: 'Manual', type: 'manual', type_title: 'Manual', tools: []}};"
        " s.generatorOrder = ['g1', 'm']; s.tools = {k6: {title: 'k6'}}; s.toolOrder = ['k6'];"
    )
    html = run_js(LAUNCH, f"{setup} [{internals}.generatorOptions('g1'), {internals}.toolOptions('m', '')]", locale="en")
    assert "Gen 1 (h1) · SSH · not checked — running: nightly" in html[0]
    assert html[1].startswith('<option value="">Not set</option>')
    assert_no_cyrillic("\n".join(html))
    ru_html = run_js(LAUNCH, f"{setup} {internals}.generatorOptions('g1')")
    assert "Gen 1 (h1) · SSH · не проверен — идёт: nightly" in ru_html


def test_run_page_builders_in_english():
    internals = "LL.launchRun._internals"
    lines = run_js(RUN_PAGE, f"{internals}.slaLines({{target_rps: 2500, max_error_rate_pct: 1, max_p95_ms: 800}})", locale="en")
    assert [line["label"] for line in lines["rps"] + lines["p95"]] == ["RPS target 2,500", "error threshold 1%", "p95 threshold 800 ms"]
    assert run_js(RUN_PAGE, f"{internals}.freshness(null, 0, true)", locale="en") == {"text": "no data yet", "stale": False}
    finished = run_js(RUN_PAGE, f"{internals}.freshness(Date.parse('2026-10-16T12:04:31Z'), 0, false)", locale="en")
    assert finished["text"] == "test finished, data up to 12:04:31"
    durations = run_js(RUN_PAGE, f"[{internals}.duration('2026-10-16T10:00:00Z', '2026-10-16T11:05:00Z'), {internals}.duration('2026-10-16T10:00:00Z', '2026-10-16T10:02:03Z'), {internals}.duration('2026-10-16T10:00:00Z', '2026-10-16T10:00:09Z')]", locale="en")
    assert durations == ["1 h 5 min", "2 min 3 s", "9 s"]
    assert run_js(RUN_PAGE, f"{internals}.fmtNumber(1234.5, 1)", locale="en") == "1,234.5"


def test_run_page_builders_keep_russian():
    internals = "LL.launchRun._internals"
    durations = run_js(RUN_PAGE, f"[{internals}.duration('2026-10-16T10:00:00Z', '2026-10-16T11:05:00Z'), {internals}.duration('2026-10-16T10:00:00Z', '2026-10-16T10:02:03Z')]")
    assert durations == ["1 ч 5 мин", "2 мин 3 с"]
    # Same digits and separators as the former toLocaleString('ru-RU').
    same = run_js(RUN_PAGE, f"[1234.5, 0.123, 99999].every((x) => {internals}.fmtNumber(x, 2) === Number(x).toLocaleString('ru-RU', {{maximumFractionDigits: 2}}))")
    assert same is True
    assert run_js(RUN_PAGE, f"{internals}.fmtNumber(null, 1)") == "—"
