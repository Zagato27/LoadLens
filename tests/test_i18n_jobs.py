# SPDX-License-Identifier: Apache-2.0
"""Job messages in the reader's language (plans/i18n-plan.md, section 4.4).

A job row keeps the text of its message in the job's language and, in ``message_ref``, the dictionary key with
its parameters; ``/jobs`` and ``/job_status`` render the ref in the language of whoever reads the list. Rows
written before the ref existed show their stored text. Tasks run in the language of the job (``report_language``
of a report, ``ru`` for a message queued without it), threads started inside a job keep that language, and the
cached job list never hands one reader's language to another.
"""

import json
import re
import sys
import threading
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before anything that imports update_page)
from AI import i18n as engine  # noqa: E402
from AI import pipeline  # noqa: E402
from AI.i18n import Msg, get_locale, msg, plural, render_ref, with_locale  # noqa: E402
from loadlens_app import job_runner, jobs, response_cache, tasks  # noqa: E402
from loadlens_app.blueprints import dashboard, forecast as forecast_api  # noqa: E402
from tests.i18n_helpers import CYRILLIC, use_locale  # noqa: E402
from tests.test_perf_job_store import JobsTable  # noqa: E402

import update_page  # noqa: E402

# Modules whose job messages and errors this suite covers.
JOB_FILES = (
    "AI/pipeline.py",
    "loadlens_app/blueprints/dashboard.py",
    "loadlens_app/blueprints/forecast.py",
    "loadlens_app/job_runner.py",
    "loadlens_app/jobs.py",
    "loadlens_app/tasks.py",
    "update_page.py",
)
JOB_KEY = re.compile(r"^job\.[a-z0-9_]+(\.[a-z0-9_]+)*$")

COLLECT_RU = "Сбор метрик: База данных (1/6)"
COLLECT_EN = "Collecting metrics: Database (1/6)"


def collect_message() -> Msg:
    return msg("job.progress.collect", domain=msg("common.domain.database"), index=1, total=6)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No storage but the fakes of a test, no runtime config files, jobs in threads."""
    def no_db():
        raise AssertionError("these tests must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)
    runtime = tmp_path / "settings_runtime.json"
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", runtime, raising=False)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(dashboard, "_report_step", lambda area, service: "30s")
    monkeypatch.setattr(tasks, "FINAL_UPDATE_RETRY_DELAYS_SEC", (0.0, 0.0))


@pytest.fixture
def table(monkeypatch):
    fake = JobsTable()
    monkeypatch.setattr(jobs, "_ts_conn", fake.connect)
    return fake


@pytest.fixture
def memory_store(monkeypatch):
    """A job store without a table, used wherever the code looks it up."""
    store = jobs.JobStore()
    monkeypatch.setattr(store, "_persist", lambda record: None)
    monkeypatch.setattr(store, "_load_many", lambda statuses, limit: [])
    monkeypatch.setattr(store, "_load_one", lambda job_id, recover=False: None)
    for module in (jobs, dashboard, forecast_api):
        monkeypatch.setattr(module, "job_store", store)
    return store


@pytest.fixture
def threads(monkeypatch):
    """threading.Thread of job_runner replaced: start() runs the target at once (as tests/test_perf_tasks.py)."""
    started = []

    class Immediate:
        def __init__(self, target, daemon=True):
            started.append(target)
            self.start = target

    monkeypatch.setattr(job_runner.threading, "Thread", Immediate)
    return started


@pytest.fixture
def client(monkeypatch, memory_store):
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    from app import create_app

    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def run_report_task(job_id: str, **overrides) -> None:
    """The report task as a thread of the web process runs it (``task.run``), without the Celery request."""
    kwargs = dict(
        start=0, end=600_000, service="checkout", project_area="shop", use_llm=False, save_to_db=False,
        web_only=True, run_name="nightly", test_type="step",
    )
    kwargs.update(overrides)
    tasks.report_task.run(job_id=job_id, **kwargs)


# ---- dictionaries and constants ----------------------------------------------------------------------------------


def _string_constants(path: str) -> set[str]:
    import ast

    tree = ast.parse((ROOT_DIR / path).read_text(encoding="utf-8"))
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def test_every_job_key_written_in_the_covered_modules_exists_in_both_languages():
    keys = sorted({value for path in JOB_FILES for value in _string_constants(path) if JOB_KEY.match(value)})
    assert len(keys) > 40
    missing = [(lang, key) for key in keys for lang in ("ru", "en") if not engine.has_key(key, lang)]
    assert not missing, missing


def test_the_job_namespace_holds_no_cyrillic_in_en():
    values = json.loads((ROOT_DIR / "locales" / "en" / "job.json").read_text(encoding="utf-8"))
    assert values and not [key for key, value in values.items() if CYRILLIC.search(value)]


def test_message_constants_follow_the_installation_language(monkeypatch):
    assert jobs.STALE_MESSAGE == "Задача прервана: воркер не отвечает"
    assert jobs.QUEUED_MESSAGE == "Задача прервана: воркер не взял её из очереди"
    assert jobs.INTERRUPTED_MESSAGE == "Задача прервана: сервер перезапустился во время выполнения"
    assert tasks.TIME_LIMIT_MESSAGE == "Задача прервана: превышено время задачи"
    assert job_runner.REPLACED_MESSAGE == "Задача перезапущена в веб-процессе"
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    assert jobs.STALE_MESSAGE == "Job interrupted: the worker does not respond"
    assert not CYRILLIC.search(tasks.TIME_LIMIT_MESSAGE + job_runner.REPLACED_MESSAGE)
    with pytest.raises(AttributeError):
        jobs.NO_SUCH_MESSAGE  # noqa: B018


# ---- the job table -----------------------------------------------------------------------------------------------


def test_a_message_is_stored_as_text_and_ref_and_plain_text_clears_the_ref(table):
    web, worker = jobs.JobStore(), jobs.JobStore()
    job = web.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", message=jobs.job_message("job.initializing"), runner="celery")
    row = table.rows[job.job_id]
    assert row["message"] == "Инициализация…"
    assert json.loads(row["message_ref"]) == {"key": "job.initializing", "params": {}}

    worker.update(job.job_id, message=collect_message(), progress=10)
    row = table.rows[job.job_id]
    assert row["message"] == COLLECT_RU
    assert json.loads(row["message_ref"]) == {
        "key": "job.progress.collect",
        "params": {"domain": {"key": "common.domain.database", "params": {}}, "index": 1, "total": 6},
    }
    seen = web.get(job.job_id)
    assert (seen.message, jobs.message_text(seen, "en"), jobs.message_text(seen, "ru")) == (COLLECT_RU, COLLECT_EN, COLLECT_RU)
    assert set(seen.to_dict()) == set(jobs.JobRecord(job_id="x", kind="report").to_dict())  # the wire shape is unchanged

    worker.update(job.job_id, message="plain text", progress=20)
    assert table.rows[job.job_id]["message_ref"] is None
    assert jobs.message_text(web.get(job.job_id), "en") == "plain text"


def test_swept_rows_get_the_ref_of_their_message(table):
    table.add("silent", "running", runner="celery", created_min_ago=60, updated_min_ago=30)
    store = jobs.JobStore()

    assert store.sweep_stale() == ["silent"]
    row = table.rows["silent"]
    assert row["message"] == "Задача прервана: воркер не отвечает"
    assert json.loads(row["message_ref"]) == {"key": "job.stale", "params": {}}
    assert jobs.message_text(store.get("silent"), "en") == "Job interrupted: the worker does not respond"


def test_without_the_ref_column_only_the_text_is_kept(monkeypatch, caplog):
    old = JobsTable(columns=jobs._JOB_COLUMNS + ("dismissed_at",), alter_fails=True)
    monkeypatch.setattr(jobs, "_ts_conn", old.connect)
    web, worker = jobs.JobStore(), jobs.JobStore()
    job = web.create(jobs.JOB_KIND_REPORT, run_name="r", service="demo", runner="celery")
    worker.update(job.job_id, message=collect_message(), progress=10)

    assert "message_ref" not in old.columns and "Cannot add message_ref" in caplog.text
    seen = web.get(job.job_id)
    assert seen.message_ref is None and jobs.message_text(seen, "en") == COLLECT_RU


def test_dismissing_without_the_column_explains_in_the_reader_language():
    error = jobs.JobStoreError("job.dismiss_unsupported")
    assert isinstance(error, RuntimeError) and "dismissed_at" in str(error) and CYRILLIC.search(str(error))
    with with_locale("en"):
        assert str(error).startswith("Clearing the job list is unavailable") and not CYRILLIC.search(str(error))


# ---- the job list ------------------------------------------------------------------------------------------------


def test_job_list_and_status_render_the_ref_in_the_reader_language(client, memory_store):
    job = memory_store.create(jobs.JOB_KIND_REPORT, run_name="nightly", service="demo", message=jobs.job_message("job.initializing"))
    memory_store.update(job.job_id, message=collect_message(), progress=10)
    # A row written before message_ref existed: its stored text is shown to everyone.
    memory_store._jobs["old"] = jobs.JobRecord(job_id="old", kind="report", message="Сбор метрик: JVM (1/6)")

    ru = {item["job_id"]: item["message"] for item in client.get("/jobs?status=running").get_json()["jobs"]}
    use_locale(client, "en")
    en = {item["job_id"]: item["message"] for item in client.get("/jobs?status=running").get_json()["jobs"]}

    assert (ru[job.job_id], en[job.job_id]) == (COLLECT_RU, COLLECT_EN)
    assert ru["old"] == en["old"] == "Сбор метрик: JVM (1/6)"
    assert client.get(f"/job_status/{job.job_id}").get_json()["message"] == COLLECT_EN
    use_locale(client, "ru")
    assert client.get(f"/job_status/{job.job_id}").get_json()["message"] == COLLECT_RU


def test_cached_job_lists_are_kept_per_language(client, memory_store, monkeypatch):
    monkeypatch.setenv("LOADLENS_RESPONSE_CACHE", "1")
    memory_store.create(jobs.JOB_KIND_REPORT, run_name="nightly", service="demo", message=collect_message())

    first_ru = client.get("/jobs?status=running").get_json()["jobs"][0]["message"]
    use_locale(client, "en")
    first_en = client.get("/jobs?status=running").get_json()["jobs"][0]["message"]
    use_locale(client, "ru")
    again_ru = client.get("/jobs?status=running").get_json()["jobs"][0]["message"]

    assert (first_ru, first_en, again_ru) == (COLLECT_RU, COLLECT_EN, COLLECT_RU)
    keys = [key for key in response_cache.response_cache._entries if key[0] == response_cache.JOBS]
    assert sorted(key[-1] for key in keys) == ["en", "ru"]
    assert response_cache.response_cache.hits >= 1  # the second ru request came from the cache
    jobs_dropped = response_cache.response_cache.invalidate(response_cache.JOBS)
    assert jobs_dropped == 2  # a change drops the list of every language


# ---- tasks -------------------------------------------------------------------------------------------------------


def test_a_failed_job_names_its_stage_and_both_render_in_the_reader_language(client, memory_store, monkeypatch, threads):
    def failing(*args, **kwargs):
        kwargs["progress_callback"](collect_message(), 10)
        raise RuntimeError("Grafana returned 502")

    monkeypatch.setattr(dashboard, "update_report", failing)
    job = dashboard.start_report_job(0, 600_000, "checkout", "shop", use_llm=False, save_to_db=False, web_only=True,
                                     run_name="nightly", test_type="step", owner="eng")

    record = memory_store.get(job.job_id)
    assert (record.status, record.message, record.error) == ("error", f"Ошибка на этапе «{COLLECT_RU}»", "Grafana returned 502")
    assert record.message_ref == {
        "key": "job.failed_at_stage",
        "params": {"stage": {"key": "job.progress.collect", "params": {
            "domain": {"key": "common.domain.database", "params": {}}, "index": 1, "total": 6}}},
    }
    assert render_ref(record.message_ref, locale="en") == f"Failed at the stage “{COLLECT_EN}”"
    use_locale(client, "en")
    assert client.get(f"/job_status/{job.job_id}").get_json()["message"] == f"Failed at the stage “{COLLECT_EN}”"


def test_a_stage_without_a_ref_is_quoted_as_it_was_stored():
    record = jobs.JobRecord(job_id="j", kind="report", message="Анализ LLM")
    failed = jobs.failure_message(record, "job.failed")
    assert failed == "Ошибка на этапе «Анализ LLM»"
    assert failed.render("en") == "Failed at the stage “Анализ LLM”"  # the stage is user-visible stored text
    assert jobs.failure_message(None, "job.forecast_failed") == "Ошибка публикации прогноза"
    assert dashboard._job_failure_message("missing") == "Ошибка выполнения задачи"


def test_a_report_job_runs_in_its_report_language(memory_store, monkeypatch, threads):
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    seen = []

    def update_report(*args, **kwargs):
        seen.append((get_locale(), kwargs["report_language"]))
        kwargs["progress_callback"](msg("job.progress.sla"), 42)
        return {"run_id": "r-1", "page_url": None, "page_id": None, "run_name": "nightly"}

    monkeypatch.setattr(dashboard, "update_report", update_report)
    job = dashboard.start_report_job(0, 600_000, "checkout", "shop", use_llm=False, save_to_db=False, web_only=True,
                                     run_name="nightly", test_type="step", owner="eng")

    record = memory_store.get(job.job_id)
    assert seen == [("en", "en")]
    assert (record.status, record.message, record.message_ref) == ("done", "Done", {"key": "job.done", "params": {}})
    assert jobs.message_text(record, "ru") == "Готово"


def test_a_queued_report_without_report_language_runs_in_ru(memory_store, monkeypatch):
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")  # the worker's own setting does not count
    seen = []

    def update_report(*args, **kwargs):
        seen.append((get_locale(), kwargs["report_language"], str(msg("job.progress.sla"))))
        return {"run_id": "r-1", "page_url": None, "page_id": None, "run_name": "nightly"}

    monkeypatch.setattr(dashboard, "update_report", update_report)
    job = memory_store.create(jobs.JOB_KIND_REPORT, run_name="nightly", service="checkout")
    run_report_task(job.job_id)  # a message queued by an older web process: no report_language

    assert seen == [("ru", "ru", "Проверка SLA-критериев")]
    assert memory_store.get(job.job_id).message == "Готово"
    assert get_locale() == "en"  # the job's scope ended with the task


def test_a_publication_job_runs_in_the_installation_language(memory_store, monkeypatch, threads):
    seen = []

    def publish_report(run_name, service, source_url, progress_callback):
        seen.append(get_locale())
        progress_callback(msg("confluence.progress.loading_report"), 5)  # the Confluence publisher reports Msg stages
        raise RuntimeError("Confluence returned 500")

    monkeypatch.setattr(dashboard, "publish_report", publish_report)
    monkeypatch.setitem(core.CONFIG["appearance"], "language", "en")
    job = job_runner.create_job(jobs.JOB_KIND_CONFLUENCE, run_name="nightly", message=jobs.job_message("job.publishing_confluence"))
    assert job.message == "Publishing to Confluence…"
    job_runner.submit(tasks.publish_confluence_task, job, run_name="nightly", service="demo", source_url="http://x/reports/r")

    record = memory_store.get(job.job_id)
    assert seen == ["en"]
    assert record.message == "Failed at the stage “Loading the report…”"
    assert record.message_ref["params"]["stage"] == {"key": "confluence.progress.loading_report", "params": {}}
    assert render_ref(record.message_ref, locale="ru") == "Ошибка на этапе «Загрузка отчёта…»"


def test_a_soft_time_limit_stores_the_ref_of_its_message(memory_store, monkeypatch):
    def too_long(*args, **kwargs):
        raise tasks.SoftTimeLimitExceeded()

    monkeypatch.setattr(dashboard, "update_report", too_long)
    job = memory_store.create(jobs.JOB_KIND_REPORT, run_name="nightly", service="checkout")
    run_report_task(job.job_id, report_language="ru")

    record = memory_store.get(job.job_id)
    assert (record.message, record.message_ref) == (tasks.TIME_LIMIT_MESSAGE, {"key": "job.time_limit", "params": {}})
    assert jobs.message_text(record, "en") == "Job interrupted: the job ran out of time"


# ---- threads and errors of a job ---------------------------------------------------------------------------------


def test_transfer_threads_keep_the_job_locale():
    seen = []

    def call():
        seen.append((threading.current_thread().name, get_locale()))
        return True

    assert get_locale() == "ru"
    with with_locale("en"):
        with update_page._TransferPool([call, call], workers=2, timeout=10, attempts=1, pause=0, name="i18n") as pool:
            assert pool.result(0) is True and pool.result(1) is True
    assert sorted(lang for _name, lang in seen) == ["en", "en"]
    assert all(name.startswith("report-i18n-") for name, _lang in seen)


def test_report_build_errors_render_where_they_are_shown():
    timeout = update_page._TransferTimeoutError("job.error.transfer_timeout", seconds="5")
    missing = update_page.ServiceConfigError("job.error.service_config_missing", service="checkout", area="shop")
    image = update_page.ReportBuildError("job.error.image_download", name="rps_checkout")
    assert isinstance(timeout, TimeoutError) and isinstance(missing, ValueError) and isinstance(image, RuntimeError)
    assert str(timeout) == "не завершилось за 5 с"
    assert str(missing) == "Конфигурация для сервиса 'checkout' в области 'shop' не найдена."
    assert str(image) == "Не удалось загрузить изображение rps_checkout"
    with with_locale("en"):
        assert str(timeout) == "did not finish within 5 s"
        assert str(missing) == "No configuration found for the service 'checkout' in the area 'shop'."
        assert str(image) == "Could not download the image rps_checkout"


def test_grafana_errors_keep_their_types_and_follow_the_locale():
    fallback = pipeline.GrafanaFallbackError("job.error.grafana_fallback", status=404, error="404 POST /api/ds/query")
    no_uid = pipeline.GrafanaError("job.error.grafana_no_uid", id=7)
    shape = pipeline.CollectionShapeError("job.error.queries_labels_mismatch")
    assert isinstance(fallback, RuntimeError) and isinstance(no_uid, RuntimeError) and isinstance(shape, ValueError)
    assert "прокси датасорса ответил 404" in str(fallback) and str(no_uid) == "У Grafana datasource id=7 нет uid"
    with with_locale("en"):
        assert str(fallback) == (
            "Grafana: the data source proxy answered 404, and the fallback request through /api/ds/query failed:"
            " 404 POST /api/ds/query"
        )
        assert not CYRILLIC.search(str(no_uid) + str(shape))


def test_pipeline_stage_messages_render_again_in_another_language():
    with with_locale("ru"):
        start = Msg(plural("job.progress.llm_start", 6), "job.progress.llm_start", {"n": 6})
        one = Msg(plural("job.progress.llm_start", 1), "job.progress.llm_start", {"n": 1})
        done = msg("job.progress.llm_done", done=2, total=6, domain=msg("common.domain.final"))
    assert start == "Анализ ИИ: 6 доменов, ожидание ответов модели"  # the text of the old code, as it was
    assert one == "Анализ ИИ: 1 доменов, ожидание ответов модели"
    assert done == "Анализ ИИ: готово 2/6 (Итог)"
    assert render_ref(start.ref(), locale="en") == "AI analysis: 6 domains, waiting for the model"
    assert render_ref(one.ref(), locale="en") == "AI analysis: 1 domain, waiting for the model"
    assert render_ref(done.ref(), locale="en") == "AI analysis: 2/6 done (Summary)"
