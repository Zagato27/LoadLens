import ast
import collections
import copy
import json
import json as _json
import logging
import os
import threading
import time
import traceback  # для детального вывода ошибок (опционально)
import uuid
from datetime import datetime  # если ещё не импортирован
from functools import partial
from typing import Callable, Sequence

from urllib.parse import urlsplit

from requests.auth import HTTPBasicAuth

from AI.main import uploadFromLLM
from AI.query_packs import apply_run_filter, strip_live_rows
from confluence_manager.update_confluence_template import (
    copy_confluence_page,
    render_llm_markdown,
    render_llm_report_placeholders,
    update_confluence_page,
    update_confluence_page_multi,
)
from data_collectors.grafana_collector import downloadImagesLogin, send_file_to_attachment
from data_collectors.loki_collector import fetch_loki_logs, send_loki_file_to_attachment
from data_collectors.temp_files import is_temp_file, safe_basename, temp_file_path
from loadlens_app.core import (
    _active_system_context as _core_active_system_context,
    _normalize_system_context,
    _resolve_services_for_area,
)
from metrics_config import METRICS_CONFIG  # Базовая конфигурация метрик
from settings import CONFIG  # Импорт базовой конфигурации

# Поддержка runtime-оверрайда metrics_config
_METRICS_RUNTIME_PATH = os.path.join(os.path.dirname(__file__), 'metrics_config_runtime.json')

logger = logging.getLogger(__name__)
# Deadline of one chart or log download and of one attachment upload, in seconds. The variable keeps
# its old name from the time these ran as Celery tasks, so existing deployments keep their value.
_TASK_TIMEOUT = int(os.getenv("CELERY_TASK_TIMEOUT", "600"))
# Downloads and uploads of the template report: tries per item, the pause between them, threads.
_TRANSFER_ATTEMPTS = 3
_TRANSFER_RETRY_PAUSE_SEC = 2.0
_TRANSFER_WORKERS_DEFAULT = 4
_TRANSFER_WORKERS_MAX = 8

def _deep_merge_dicts(_a: dict, _b: dict) -> dict:
    out = dict(_a or {})
    for k, v in (_b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out.get(k) or {}, v)
        else:
            out[k] = v
    return out

def _active_metrics_config_now() -> dict:
    try:
        from metrics_config import METRICS_CONFIG as BASE
    except Exception:
        BASE = {}
    try:
        if os.path.exists(_METRICS_RUNTIME_PATH):
            with open(_METRICS_RUNTIME_PATH, 'r', encoding='utf-8') as _f:
                _override = _json.load(_f)
            if isinstance(_override, dict):
                return _normalize_metrics_config(_deep_merge_dicts(BASE, _override))
    except Exception:
        return _normalize_metrics_config(BASE)
    return _normalize_metrics_config(BASE)


def _service_area_map() -> dict:
    mapping: dict[str, str] = {}
    per_area = _per_area_data()
    for area, cfg in per_area.items():
        services = cfg.get('services')
        if isinstance(services, dict):
            for sid in services.keys():
                mapping[sid] = area
    return mapping


def _normalize_metrics_config(raw: dict | None) -> dict:
    normalized: dict[str, dict] = {}
    service_to_area = _service_area_map()

    def ensure(area_name: str) -> dict:
        if area_name not in normalized or not isinstance(normalized[area_name], dict):
            normalized[area_name] = {"services": {}}
        if 'services' not in normalized[area_name] or not isinstance(normalized[area_name]['services'], dict):
            normalized[area_name]['services'] = {}
        return normalized[area_name]

    for area_name in _per_area_data().keys():
        ensure(area_name)

    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            if isinstance(value.get('services'), dict):
                entry = ensure(key)
                entry_services = entry.get('services', {})
                entry_services.update(value.get('services') or {})
                entry['services'] = entry_services
                for meta_key, meta_val in value.items():
                    if meta_key != 'services':
                        entry[meta_key] = meta_val
                continue
            target_area = service_to_area.get(key) or value.get('area') or key
            entry = ensure(target_area)
            entry['services'][key] = value
    return normalized


def _metrics_service_entry(metrics_cfg: dict, service_name: str | None, area_name: str | None = None) -> dict:
    """Возвращает конфигурацию метрик/графиков для конкретного сервиса.

    Приоритет:
    1) Явно указанная область (project_area), если там есть service.
    2) Любая другая область, где встречается этот service (для обратной совместимости).
    """
    if not service_name:
        return {}

    # 1) Сначала пробуем явную область
    if area_name and isinstance(metrics_cfg.get(area_name), dict):
        area_cfg = metrics_cfg.get(area_name) or {}
        services = area_cfg.get("services")
        if isinstance(services, dict) and service_name in services:
            return services.get(service_name) or {}

    # 2) Фолбэк: поиск по всем областям (старое поведение)
    for area_cfg in (metrics_cfg or {}).values():
        services = (area_cfg or {}).get("services")
        if isinstance(services, dict) and service_name in services:
            return services.get(service_name) or {}

    # 3) Наследие: конфиги верхнего уровня
    legacy = (metrics_cfg or {}).get(service_name)
    if isinstance(legacy, dict):
        services = legacy.get("services")
        if isinstance(services, dict) and service_name in services:
            return services.get(service_name) or {}
    return {}

_SETTINGS_RUNTIME_PATH = os.path.join(os.path.dirname(__file__), 'settings_runtime.json')

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'AI', 'prompts')
_PROMPT_DOMAIN_FILES = {
    'overall': 'overall_prompt.txt',
    'jvm': 'jvm_prompt.txt',
    'database': 'database_prompt.txt',
    'kafka': 'kafka_prompt.txt',
    'microservices': 'microservices_prompt.txt',
    'hard_resources': 'hard_resources_prompt.txt',
    'lt_framework': 'lt_framework_prompt.txt',
    'lt_generators': 'lt_generators_prompt.txt',
    'application_logs': 'application_logs_prompt.txt',
}


def default_run_name(now: datetime | None = None) -> str:
    """Run name used when the user leaves the name empty."""
    return (now or datetime.now()).strftime("run-%Y%m%d-%H%M%S")


def _load_default_prompts() -> dict:
    prompts: dict[str, str] = {}
    for domain, fname in _PROMPT_DOMAIN_FILES.items():
        path = os.path.join(_PROMPTS_DIR, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompts[domain] = f.read()
        except Exception:
            prompts[domain] = ''
    return prompts


def _load_settings_runtime_data() -> dict:
    try:
        if os.path.exists(_SETTINGS_RUNTIME_PATH):
            with open(_SETTINGS_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                data = _json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _per_area_data() -> dict:
    data = _load_settings_runtime_data()
    per_area = data.get('per_area') if isinstance(data, dict) else {}
    return per_area if isinstance(per_area, dict) else {}


def _find_area_for_service(service_name: str | None) -> str | None:
    if not service_name:
        return None
    per_area = _per_area_data()
    for area, cfg in per_area.items():
        services = cfg.get('services')
        if isinstance(services, dict) and service_name in services:
            return area
    return None


def _service_prompts_override(area_name: str | None, service_name: str | None) -> dict:
    if not area_name or not service_name:
        return {}
    per_area = _per_area_data()
    area_cfg = per_area.get(area_name)
    if not isinstance(area_cfg, dict):
        return {}
    services = area_cfg.get('services')
    if not isinstance(services, dict):
        return {}
    svc_entry = services.get(service_name)
    if not isinstance(svc_entry, dict):
        return {}
    prompts = svc_entry.get('prompts')
    return prompts if isinstance(prompts, dict) else {}


def _service_disabled_domains(area_name: str | None, service_name: str | None) -> list[str]:
    if not area_name or not service_name:
        return []
    per_area = _per_area_data()
    area_cfg = per_area.get(area_name)
    if not isinstance(area_cfg, dict):
        return []
    services = area_cfg.get('services')
    if not isinstance(services, dict):
        return []
    svc_entry = services.get(service_name)
    if not isinstance(svc_entry, dict):
        return []
    disabled = svc_entry.get('disabled_domains')
    return [d for d in (disabled or []) if isinstance(d, str)]


def _prompt_templates_for_scope(area_name: str | None, service_name: str | None) -> dict:
    templates = _load_default_prompts()
    area_prompts = _prompts_override_for_area(area_name)
    service_prompts = _service_prompts_override(area_name, service_name)
    out: dict[str, str] = {}
    for domain in _PROMPT_DOMAIN_FILES.keys():
        if isinstance(service_prompts, dict) and isinstance(service_prompts.get(domain), str) and service_prompts.get(domain).strip():
            out[domain] = service_prompts.get(domain, '')
        elif isinstance(area_prompts, dict) and isinstance(area_prompts.get(domain), str) and area_prompts.get(domain).strip():
            out[domain] = area_prompts.get(domain, '')
        else:
            out[domain] = templates.get(domain, '')
    return out


def _domain_keys_from_config(cfg: dict) -> list[str]:
    queries = (cfg.get('queries') or {})
    preferred_order = ['jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework', 'lt_generators']
    keys = [k for k in preferred_order if k in queries]
    logs_cfg = cfg.get('logs_source') if isinstance(cfg.get('logs_source'), dict) else {}
    if logs_cfg.get('enabled'):
        keys.append('application_logs')
    for key in queries.keys():
        if key not in keys:
            keys.append(key)
    return keys


def _transfer_workers(cfg: dict | None = None) -> int:
    """Threads for the downloads and uploads of one report: ``llm.max_domain_workers`` or 4, within 1..8."""
    llm_cfg = (cfg if isinstance(cfg, dict) else CONFIG).get("llm")
    raw = llm_cfg.get("max_domain_workers") if isinstance(llm_cfg, dict) else None
    try:
        count = int(float(raw))
    except (TypeError, ValueError):
        count = _TRANSFER_WORKERS_DEFAULT
    return max(1, min(count, _TRANSFER_WORKERS_MAX))


class _TransferPool:
    """Runs the downloads or the uploads of one report on a few daemon threads.

    It replaces nested Celery tasks: a report running inside a worker that waits for sub-tasks queued
    on the same worker pool deadlocks once the pool is busy (concurrency 1), and the hop through the
    broker bought nothing. Items start in their order on up to ``workers`` threads. Each item gets up
    to ``attempts`` calls, ``pause`` seconds apart: an exception means "try again", a returned value
    is final (False marks a refused item). :meth:`result` waits at most ``timeout`` seconds counted
    from the moment the item started.

    A thread cannot be killed, so an item past its deadline is abandoned instead: its thread is a
    daemon and never holds up the exit of the process, it makes no further attempts and ends once the
    call returns, and a new thread takes its place, so the items queued behind it still run.
    """

    def __init__(
        self,
        calls: Sequence[Callable[[], object]],
        *,
        workers: int,
        timeout: float,
        attempts: int = _TRANSFER_ATTEMPTS,
        pause: float = _TRANSFER_RETRY_PAUSE_SEC,
        name: str = "transfer",
    ):
        self._calls = list(calls)
        self._timeout = max(0.0, float(timeout))
        self._attempts = max(1, int(attempts))
        self._pause = max(0.0, float(pause))
        self._name = name
        self._cond = threading.Condition()
        self._pending = collections.deque(range(len(self._calls)))
        self._started: list[float | None] = [None] * len(self._calls)
        self._outcome: dict[int, tuple[object, BaseException | None]] = {}
        self._abandoned: set[int] = set()
        self._spawned = 0
        with self._cond:
            for _ in range(min(max(1, int(workers)), len(self._calls))):
                self._spawn()

    def __enter__(self) -> "_TransferPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Drops the items that have not started; the running ones finish on their own."""
        with self._cond:
            self._pending.clear()
            self._cond.notify_all()

    def result(self, index: int) -> object:
        """Value of item ``index``; raises its last error, or :class:`TimeoutError` past its deadline."""
        with self._cond:
            while index not in self._outcome:
                if index in self._abandoned:
                    raise TimeoutError(f"не завершилось за {self._timeout:g} с")
                started = self._started[index]
                if started is None:
                    if index not in self._pending:
                        raise RuntimeError("отменено: отчёт перестал ждать")
                    self._cond.wait(1.0)
                    continue
                remaining = started + self._timeout - time.monotonic()
                if remaining <= 0:
                    self._abandoned.add(index)
                    if self._pending:
                        self._spawn()
                    raise TimeoutError(f"не завершилось за {self._timeout:g} с")
                self._cond.wait(remaining)
            value, error = self._outcome[index]
        if error is not None:
            raise error
        return value

    def _spawn(self) -> None:
        """Starts one more thread; the caller holds the lock."""
        self._spawned += 1
        threading.Thread(target=self._work, name=f"report-{self._name}-{self._spawned}", daemon=True).start()

    def _work(self) -> None:
        while True:
            with self._cond:
                if not self._pending:
                    return
                index = self._pending.popleft()
                self._started[index] = time.monotonic()
                self._cond.notify_all()
            value: object = None
            error: BaseException | None = None
            try:
                value = self._attempt(index)
            except BaseException as exc:  # recorded for result(); the thread itself must not die here
                error = exc
            with self._cond:
                abandoned = index in self._abandoned
                if not abandoned:
                    self._outcome[index] = (value, error)
                self._cond.notify_all()
            if abandoned:
                return  # a new thread already took this one's place

    def _attempt(self, index: int) -> object:
        call = self._calls[index]
        attempt = 1
        while True:
            try:
                return call()
            except Exception as exc:
                with self._cond:
                    given_up = index in self._abandoned
                if attempt >= self._attempts or given_up:
                    raise
                logger.warning(
                    "%s: попытка %s/%s не удалась: %s. Повтор через %s сек.",
                    self._name,
                    attempt,
                    self._attempts,
                    exc,
                    self._pause,
                )
                time.sleep(self._pause)
                attempt += 1


def _span_percent(start: int, end: int, done: int, total: int) -> int:
    """Position ``done`` of ``total`` inside the percent range ``start``..``end``."""
    if total <= 0:
        return start
    ratio = max(0, min(int(done), int(total))) / int(total)
    return start + int(round((end - start) * ratio))


def _scale_pipeline_percent(percent: int | None) -> int | None:
    """Maps the pipeline's 0..100 into 40..94, after charts and attachments."""
    if percent is None:
        return None
    scaled = 40 + int(round(max(0, min(100, int(percent))) * 0.54))
    return min(scaled, 94)


def _download_img_with_retry(image_url: str, file_basename: str, username: str, password: str, max_attempts: int = 3) -> bool:
    for attempt in range(max_attempts):
        delay_sec = min(5 * (attempt + 1), 20) if "/render/" in image_url else (attempt + 1)
        try:
            downloadImagesLogin(image_url, file_basename, username, password)
            path = temp_file_path(file_basename, ".jpg")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            logger.warning("Файл не создан или пуст: %s", path)
        except Exception as exc:
            if attempt >= max_attempts - 1:
                logger.warning(
                    "Попытка загрузки изображения не удалась (%s/%s): %s. Повторы исчерпаны.",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                continue
            logger.warning(
                "Попытка загрузки изображения не удалась (%s/%s): %s. Повтор через %s сек.",
                attempt + 1,
                max_attempts,
                exc,
                delay_sec,
            )
            time.sleep(delay_sec)
            continue
        if attempt < max_attempts - 1:
            time.sleep(delay_sec)
    return False


def _download_log_with_retry(loki_url: str, start_ts: int, end_ts: int, filter_query: str, file_basename: str, max_attempts: int = 3) -> bool:
    for attempt in range(max_attempts):
        try:
            path = fetch_loki_logs(loki_url, start_ts, end_ts, filter_query, file_basename)
            if isinstance(path, str) and os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            logger.warning("Лог-файл не создан или пуст: %s.log", file_basename)
        except Exception as exc:
            logger.warning(
                "Попытка получения логов не удалась (%s/%s): %s",
                attempt + 1,
                max_attempts,
                exc,
            )
        time.sleep(1 * (attempt + 1))
    return False


def _attach_with_retry(func, *args, max_attempts: int = 3, **kwargs) -> bool:
    for attempt in range(max_attempts):
        try:
            resp = func(*args, **kwargs)
            code = getattr(resp, "status_code", None) if resp is not None else None
            if code in (200, 201):
                return True
            if code == 409:
                logger.info("Вложение уже существует, считаю успехом.")
                return True
            logger.warning("Ошибка загрузки вложения (attempt %s): status=%s", attempt + 1, code)
        except Exception as exc:
            logger.warning("Попытка загрузки вложения не удалась (%s): %s", attempt + 1, exc)
        time.sleep(1 * (attempt + 1))
    return False


_LEGACY_KEYS = ("user", "password", "grafana_login", "grafana_pass", "url_basic", "grafana_base_url", "loki_url")


def _legacy_settings() -> dict:
    """Confluence, Grafana and Loki settings of the template flow, read where the report runs.

    settings_runtime.json wins over CONFIG, so a Celery worker, which never sees the in-memory CONFIG
    updates of the web process, still uses the values saved after it started.
    """
    runtime = _load_settings_runtime_data()
    return {key: runtime.get(key, CONFIG.get(key)) for key in _LEGACY_KEYS}


def refresh_runtime_config() -> bool:
    """Applies settings_runtime.json onto the shared CONFIG again, as loadlens_app.core does at import.

    The web process keeps CONFIG current by itself: the settings pages write the runtime file and
    change CONFIG in memory. A Celery worker reads the file once, when it starts, and its child
    processes inherit that snapshot, so a task there would use the settings of that moment (LLM
    provider, Confluence, storage, global queries and SLA). Tasks running in a worker call this first.
    A key deleted from the runtime file keeps its last value until the worker restarts.

    Returns False, leaving CONFIG as it is, when the file is missing, empty or unreadable.
    """
    runtime = _load_settings_runtime_data()
    if not runtime:
        return False
    CONFIG.update(_deep_merge_dicts(CONFIG, runtime))
    return True


def _same_origin(url: str, base: str) -> bool:
    try:
        left, right = urlsplit(str(url)), urlsplit(str(base))
    except ValueError:
        return False
    return bool(right.netloc) and (left.scheme.lower(), left.netloc.lower()) == (right.scheme.lower(), right.netloc.lower())


def download_grafana_image(image_url: str, file_basename: str, settings: dict | None = None) -> bool:
    """Downloads one chart into the temporary files; False when the URL is refused.

    Raises when the download fails, so :class:`_TransferPool` tries again. ``settings`` is the result
    of :func:`_legacy_settings`, read once per report; without it the function reads it itself.
    """
    settings = settings if settings is not None else _legacy_settings()
    # The URL is built from the metrics config: the Grafana login is only ever sent to the configured Grafana.
    if not _same_origin(image_url, settings.get("grafana_base_url") or ""):
        logger.warning("Адрес графика не совпадает с grafana_base_url, загрузка отклонена")
        return False
    if _download_img_with_retry(image_url, file_basename, settings.get("grafana_login") or "", settings.get("grafana_pass") or ""):
        return True
    raise RuntimeError(f"Не удалось загрузить изображение {file_basename}")


def download_loki_logs(start_ts: int, end_ts: int, filter_query: str, file_basename: str, settings: dict | None = None) -> bool:
    """Downloads the logs of one placeholder into the temporary files; raises when that fails."""
    settings = settings if settings is not None else _legacy_settings()
    loki_url = settings.get("loki_url") or ""
    if _download_log_with_retry(loki_url, start_ts, end_ts, filter_query, file_basename):
        return True
    raise RuntimeError(f"Не удалось получить логи {file_basename}")


def upload_attachment(kind: str, page_id: str, file_path: str, settings: dict | None = None) -> bool:
    """Attaches one downloaded file to the page; False when the path is refused, raises when the upload fails."""
    # Upload only files the report downloaded itself.
    if not is_temp_file(file_path):
        logger.warning("Вложение вне каталога временных файлов отклонено: %s", file_path)
        return False
    settings = settings if settings is not None else _legacy_settings()
    url_basic = settings.get("url_basic") or ""
    auth = HTTPBasicAuth(settings.get("user") or "", settings.get("password") or "")
    if kind == "grafana":
        ok = _attach_with_retry(send_file_to_attachment, url_basic, auth, page_id, file_path)
    else:
        ok = _attach_with_retry(send_loki_file_to_attachment, url_basic, auth, page_id, file_path)
    if ok:
        return True
    raise RuntimeError(f"Не удалось загрузить вложение {file_path}")


# Names of the former Celery tasks, still called by tests/test_smoke_api.py. Plain functions now.
download_grafana_image_task = download_grafana_image
download_loki_logs_task = download_loki_logs
upload_attachment_task = upload_attachment


def _load_area_overrides(area_name: str) -> dict:
    try:
        if not area_name:
            return {}
        if os.path.exists(_SETTINGS_RUNTIME_PATH):
            with open(_SETTINGS_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                rt = _json.load(f)
            per_area = (rt.get('per_area') or {}) if isinstance(rt, dict) else {}
            area = (per_area.get(area_name) or {}) if isinstance(per_area, dict) else {}
            return area if isinstance(area, dict) else {}
    except Exception:
        return {}
    return {}

def _service_entry(area_name: str | None, service_name: str | None) -> dict:
    if not area_name or not service_name:
        return {}
    per_area = _per_area_data()
    area_cfg = per_area.get(area_name)
    if not isinstance(area_cfg, dict):
        return {}
    services = area_cfg.get('services')
    if not isinstance(services, dict):
        return {}
    svc_entry = services.get(service_name)
    return svc_entry if isinstance(svc_entry, dict) else {}


# Only these sections are edited per service in the settings UI. Older versions copied the
# other sections onto each service; merging those copies would freeze a snapshot and ignore
# later area changes (notably default_params.resample_interval).
_SERVICE_CONFIG_OVERRIDES = frozenset({"queries", "sla", "system_context"})


def _effective_config_for_scope(area_name: str, service_name: str | None = None) -> dict:
    area = _load_area_overrides(area_name)
    service_entry = _service_entry(area_name, service_name) if service_name else {}
    eff = dict(CONFIG)
    for key in ("llm", "logs_source", "default_params", "queries", "sla", "system_context"):
        base = (CONFIG.get(key) or {}) if isinstance(CONFIG.get(key), dict) else (CONFIG.get(key) if key == 'queries' else {})
        over_area = (area.get(key) or {}) if isinstance(area.get(key), dict) else {}
        over_service = (service_entry.get(key) or {}) if key in _SERVICE_CONFIG_OVERRIDES and isinstance(service_entry.get(key), dict) else {}
        eff[key] = _deep_merge_dicts(_deep_merge_dicts(base, over_area), over_service)
    # The catalog is global. An area that sets domain_sources replaces the global bindings entirely.
    eff["data_sources"] = copy.deepcopy(CONFIG.get("data_sources") or {})
    area_bindings = area.get("domain_sources")
    if isinstance(area_bindings, dict):
        eff["domain_sources"] = copy.deepcopy(area_bindings)
    else:
        eff["domain_sources"] = copy.deepcopy(CONFIG.get("domain_sources") or {})
    return eff

def _prompts_override_for_area(area_name: str | None) -> dict:
    if not area_name:
        return {}
    area = _load_area_overrides(area_name)
    prompts = area.get('prompts') if isinstance(area, dict) else None
    return prompts if isinstance(prompts, dict) else {}


def update_report(start, end, service, use_llm: bool = True, save_to_db: bool = False, web_only: bool = False, run_name: str | None = None, test_type: str | None = None, project_area: str | None = None, progress_callback=None, run_tag: str | None = None, run_filter_packs: list[str] | None = None):
    """Формирует отчёт по нагрузочному тесту (Confluence и/или веб-интерфейс).

    Параметры:
        start (int): Начало интервала в миллисекундах UNIX.
        end (int): Конец интервала в миллисекундах UNIX.
        service (str): Идентификатор сервиса из конфигурации.
        use_llm (bool): Включать ли аналитический блок LLM.
        save_to_db (bool): Сохранять ли промежуточные данные в TimescaleDB.
        web_only (bool): При True генерирует только веб-отчёт (без Confluence).
        run_name (str | None): Пользовательское имя запуска.
        test_type (str | None): Профиль теста (step/soak/...).
        progress_callback (callable | None): Колбэк прогресса (msg, percent).
        run_tag (str | None): Тег прогона, запущенного из LoadLens: запросы пакетов из
            ``run_filter_packs`` читают только его метрики (см. RunFilter в AI/query_packs).

    Возвращает:
        dict: Информация о созданной странице (`page_id`, `page_url`, `run_name`).

    Побочные эффекты:
        - Сетевые запросы к Grafana, Loki, Confluence, LLM-провайдеру.
        - Создание/обновление страниц Confluence, загрузка вложений.
        - Опциональная запись в TimescaleDB.

    Исключения:
        Пробрасывает ошибки работы с Confluence/Grafana/LLM при критических сбоях.
    """
    # Confluence, Grafana and Loki settings of the template flow, as saved now: the report may run in a
    # Celery worker whose CONFIG is the snapshot taken when the worker started.
    legacy_settings = _legacy_settings()
    user = legacy_settings.get("user")
    password = legacy_settings.get("password")
    url_basic = legacy_settings.get("url_basic")
    grafana_base_url = legacy_settings.get("grafana_base_url")

    
    # Определяем область проекта (area) для сервиса
    # В приоритете явно переданный project_area из UI, затем маппинг service->area, затем сам service.
    area_name = (project_area or "").strip() or _find_area_for_service(service) or service

    # Получаем конфигурацию сервиса и проверяем наличие метрик (горячее чтение runtime)
    _mc = _active_metrics_config_now()
    service_config = _metrics_service_entry(_mc, service, area_name)
    if not service_config:
        raise ValueError(f"Конфигурация для сервиса '{service}' в области '{area_name}' не найдена.")
    ef_cfg = _effective_config_for_scope(area_name, service)
    # Rows for the live charts of a running test (RPS by LoadLens instance) are not report data.
    ef_cfg["queries"] = strip_live_rows(ef_cfg.get("queries") or {})
    if run_tag:
        ef_cfg["queries"], _filtered_packs = apply_run_filter(ef_cfg.get("queries") or {}, run_tag, run_filter_packs)
        if _filtered_packs:
            logger.info("Run tag %s applied to query packs: %s", run_tag, ", ".join(_filtered_packs))
    base_prompt_templates = _prompt_templates_for_scope(area_name, service)
    domain_keys = _domain_keys_from_config(ef_cfg)
    disabled_domains = _service_disabled_domains(area_name, service)
    active_domains = [d for d in domain_keys if d not in disabled_domains]
    scoped_system_context = ef_cfg.get("system_context") if isinstance(ef_cfg.get("system_context"), dict) else None
    system_context_snapshot = (
        _normalize_system_context(scoped_system_context)
        if scoped_system_context is not None
        else _core_active_system_context()
    )

    # Режим «только веб»: не создаём страницу Confluence, не скачиваем изображения из Grafana
    if web_only:
        def _progress(msg: str, pct: int | None = None):
            if pct is not None:
                logger.info("[progress] %s %s%%", msg, pct)
            else:
                logger.info("[progress] %s", msg)
            try:
                if callable(progress_callback):
                    progress_callback(msg, pct)
            except Exception:
                logger.exception("Ошибка колбэка прогресса")

        _progress("Создание веб-отчёта (без Confluence) начато…", 5)

        # Гарантируем сохранение данных в БД для веб-страниц /reports и графиков
        save_to_db_effective = True if web_only else bool(save_to_db)

        run_meta = {
            "run_id": uuid.uuid4().hex,
            "run_name": (run_name or "").strip() or default_run_name(),
            "service": service,
            "test_type": (test_type or '').strip(),
            "start_ms": start,
            "end_ms": end,
            "project_area": area_name,
            "area_services": _resolve_services_for_area(area_name) or [service],
        }

        results = None
        if use_llm or save_to_db_effective:
            # The pipeline reports its own phases (collect 10-40 %, LLM 47-80 %, save 95 %).
            results = uploadFromLLM(
                start/1000,
                end/1000,
                save_to_db=save_to_db_effective,
                run_meta=run_meta,
                only_collect=not use_llm,
                ef_config=ef_cfg,
                prompts_override=base_prompt_templates,
                active_domains=active_domains,
                system_context=system_context_snapshot,
                progress_callback=_progress,
            )
        else:
            _progress("Пропускаем LLM-анализ и сбор доменных данных по запросу пользователя")

        _progress("Финализация веб-отчёта…", 98)
        page_url = f"/reports/{run_meta['run_id']}"
        _progress("Отчёт (веб) готов ✅", 100)
        return {"page_id": None, "page_url": page_url, "run_name": run_meta["run_name"], "run_id": run_meta["run_id"]}

    # Получаем `page_sample_id` и `page_parent_id` из конфигурации сервиса
    page_parent_id = service_config["page_parent_id"]
    page_sample_id = service_config["page_sample_id"]
    copy_page_id = copy_confluence_page(url_basic, user, password, page_sample_id, page_parent_id)
    page_url = f"{url_basic.rstrip('/')}/pages/viewpage.action?pageId={copy_page_id}"

    # Прогресс
    def _progress(msg: str, pct: int | None = None):
        if pct is not None:
            logger.info("[progress] %s %s%%", msg, pct)
        else:
            logger.info("[progress] %s", msg)
        try:
            if callable(progress_callback):
                progress_callback(msg, pct)
        except Exception:
            logger.exception("Ошибка колбэка прогресса")

    _progress("Создание отчёта начато…", 5)
   

    # Список задач для обновлений
    tasks = []
    
                
    # Добавим функцию обновления с повторными попытками
    def update_with_retry(url, username, password, page_id, data_to_find, replace_text, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                res = update_confluence_page(url, username, password, page_id, data_to_find, replace_text)
                # Обработка текстовых ошибок из update_confluence_page
                if isinstance(res, str) and (res.startswith("Ошибка") or res == "Плейсхолдер не найден"):
                    raise RuntimeError(res)
                return res
            except Exception as e:
                if ("Attempted to update stale data" in str(e) or "conflict" in str(e).lower()) and attempt < max_attempts-1:
                    logger.warning("Попытка %s не удалась, повторяем через 1 секунду...", attempt + 1)
                    time.sleep(1)
                elif attempt < max_attempts-1:
                    logger.warning("Попытка %s не удалась: %s. Повтор через 1 секунду...", attempt + 1, e)
                    time.sleep(1)
                else:
                    raise e

    # Двухфазный процесс: 1) скачать всё 2) одним проходом вложить и заменить
    _progress("Скачивание графиков и логов…", 10)
    
    # Сформируем задания на скачивание графиков и логов
    metric_items = []  # элементы: {name, placeholder, file_basename, file_path}
    log_items = []     # элементы: {placeholder, file_basename, file_path}

    for metric in service_config["metrics"]:
        name = metric["name"]
        grafana_url = f"{grafana_base_url}{metric['grafana_url']}&from={start}&to={end}"
        # One safe path component: it names the file and goes verbatim into ri:filename below.
        file_basename = safe_basename(f"{name}_{service}_{copy_page_id}")
        file_path = temp_file_path(file_basename, ".jpg")
        metric_items.append({
            "name": name,
            "placeholder": f"$${name}$$",
            "grafana_url": grafana_url,
            "file_basename": file_basename,
            "file_path": file_path,
        })

    for log in service_config.get("logs", []):
        placeholder = log["placeholder"]
        file_basename = safe_basename(f"{service}_{placeholder}_{copy_page_id}")
        file_path = temp_file_path(file_basename, ".log")
        log_items.append({
            "placeholder": f"$${placeholder}$$",
            "filter_query": log["filter_query"],
            "file_basename": file_basename,
            "file_path": file_path,
        })

    # Downloads and uploads run on a local pool of threads, never as Celery sub-tasks: see _TransferPool.
    transfer_workers = _transfer_workers(ef_cfg)
    download_jobs: list[tuple[str, dict, Callable[[], object]]] = []
    for m in metric_items:
        download_jobs.append(
            ("metric", m, partial(download_grafana_image, m["grafana_url"], m["file_basename"], settings=legacy_settings))
        )
    for l in log_items:
        download_jobs.append(
            ("log", l, partial(download_loki_logs, start, end, l["filter_query"], l["file_basename"], settings=legacy_settings))
        )
    download_total = len(download_jobs)
    with _TransferPool(
        [call for _kind, _item, call in download_jobs],
        workers=transfer_workers,
        timeout=_TASK_TIMEOUT,
        pause=_TRANSFER_RETRY_PAUSE_SEC,
        name="download",
    ) as downloads:
        for index, (kind, item, _call) in enumerate(download_jobs, start=1):
            identifier = item.get("name") if kind == "metric" else item.get("placeholder")
            kind_label = "графика" if kind == "metric" else "логов"
            _progress(
                f"Скачивание {kind_label} «{identifier}» ({index}/{download_total})",
                _span_percent(10, 30, index - 1, download_total),
            )
            try:
                downloads.result(index - 1)
            except Exception as exc:
                logger.error("Ошибка при скачивании %s '%s': %s", kind, identifier, exc)

    replacements_pending = {}
    attachment_jobs: list[tuple[tuple[str, str], Callable[[], object]]] = []

    for m in metric_items:
        if os.path.exists(m["file_path"]) and os.path.getsize(m["file_path"]) > 0:
            attachment_jobs.append(
                (
                    (m["placeholder"], f'<ac:image><ri:attachment ri:filename="{m["file_basename"]}.jpg" /></ac:image>'),
                    partial(upload_attachment, "grafana", copy_page_id, m["file_path"], settings=legacy_settings),
                )
            )
        else:
            logger.warning("Не найден файл графика: %s", m["file_path"])

    for l in log_items:
        if os.path.exists(l["file_path"]) and os.path.getsize(l["file_path"]) > 0:
            attachment_jobs.append(
                (
                    (
                        l["placeholder"],
                        (
                            f'<ac:structured-macro ac:name="view-file" ac:schema-version="1">'
                            f'<ac:parameter ac:name="name">'
                            f'<ri:attachment ri:filename="{l["file_basename"]}.log" />'
                            f'</ac:parameter>'
                            f'<ac:parameter ac:name="height">250</ac:parameter>'
                            f'</ac:structured-macro>'
                        ),
                    ),
                    partial(upload_attachment, "logs", copy_page_id, l["file_path"], settings=legacy_settings),
                )
            )
        else:
            logger.warning("Не найден файл логов: %s", l["file_path"])

    success_placeholders: set[str] = set()
    attachment_total = len(attachment_jobs)
    with _TransferPool(
        [call for _replacement, call in attachment_jobs],
        workers=transfer_workers,
        timeout=_TASK_TIMEOUT,
        pause=_TRANSFER_RETRY_PAUSE_SEC,
        name="upload",
    ) as uploads:
        for index, ((placeholder, html), _call) in enumerate(attachment_jobs, start=1):
            _progress(
                f"Загрузка вложения в Confluence ({index}/{attachment_total})",
                _span_percent(30, 40, index - 1, attachment_total),
            )
            ok = False
            try:
                ok = bool(uploads.result(index - 1))
            except Exception as exc:
                logger.error("Ошибка при загрузке вложения: %s", exc)
            if ok:
                success_placeholders.add(placeholder)
                replacements_pending[placeholder] = html

    # Убираем временные файлы
    for m in metric_items:
        try:
            if os.path.exists(m["file_path"]):
                os.remove(m["file_path"])
        except Exception:
            pass
    for l in log_items:
        try:
            if os.path.exists(l["file_path"]):
                os.remove(l["file_path"])
        except Exception:
            pass

    # Одно мульти-обновление всех плейсхолдеров (только успешно загруженные)
    try:
        if replacements_pending:
            update_confluence_page_multi(url_basic, user, password, copy_page_id, replacements_pending)
        else:
            logger.warning("Нет успешных вложений для подстановки плейсхолдеров")
    except Exception as e:
        logger.error("Ошибка при мульти-обновлении плейсхолдеров (графики/логи): %s", e)

    def _pipeline_progress(msg: str, pct: int | None = None):
        _progress(msg, _scale_pipeline_percent(pct))

    # Same in-process pipeline as the web report, so the progress bar keeps moving.
    results = None
    run_meta = {
        "run_id": uuid.uuid4().hex,
        "run_name": (run_name or "").strip() or default_run_name(),
        "service": service,
        "test_type": (test_type or "").strip(),
        "start_ms": start,
        "end_ms": end,
        "project_area": area_name,
        "area_services": _resolve_services_for_area(area_name) or [service],
    }
    if use_llm or save_to_db:
        results = uploadFromLLM(
            start / 1000,
            end / 1000,
            save_to_db=bool(save_to_db),
            run_meta=run_meta,
            only_collect=not use_llm,
            ef_config=ef_cfg,
            prompts_override=base_prompt_templates,
            active_domains=active_domains,
            system_context=system_context_snapshot,
            progress_callback=_pipeline_progress,
        )
    else:
        _progress("Пропускаем LLM-анализ и сбор доменных данных по запросу пользователя")

    # Мульти-обновление плейсхолдеров LLM за один проход
    try:
        _progress("Обновление данных LLM (одним проходом)...", 85)
        llm_replacements = {}
        # Подставляем только те плейсхолдеры, для которых есть данные
        def add_if_present(placeholder: str, key: str):
            if not results:
                return
            val = results.get(key)
            if isinstance(val, str) and val.strip():
                llm_replacements[placeholder] = val

        add_if_present("$$answer_jvm$$", "jvm")
        add_if_present("$$answer_database$$", "database")
        add_if_present("$$answer_kafka$$", "kafka")
        add_if_present("$$answer_ms$$", "ms")
        add_if_present("$$answer_hard_resources$$", "hard_resources")
        add_if_present("$$answer_lt_framework$$", "lt_framework")

        # Добавляем финальный плейсхолдер только как $$final_answer$$
        final_struct = (results or {}).get("final_parsed")
        if isinstance(final_struct, dict) and final_struct:
            md = render_llm_markdown(final_struct, domain="final")
            if md.strip():
                llm_replacements["$$final_answer$$"] = md
        else:
            # Фолбэк: если нет структурированного ответа, отдаем текст как markdown-блок
            final_text = (results or {}).get("final")
            if isinstance(final_text, str) and final_text.strip():
                md_fallback = f"### Итог LLM\n\n{final_text}"
                llm_replacements["$$final_answer$$"] = md_fallback

        # Доменные секции в человекочитаемом markdown при наличии parsed
        def _inject_confidence(rep: dict | None, domain_key: str) -> dict | None:
            if not isinstance(rep, dict):
                return rep
            try:
                if rep.get("confidence") in (None, "", "null"):
                    c = ((results or {}).get("scores", {}) or {}).get(domain_key, {}) or {}
                    cval = c.get("confidence")
                    if isinstance(cval, (int, float)):
                        rep = {**rep, "confidence": float(cval)}
            except Exception:
                pass
            return rep

        jvm_struct = _inject_confidence((results or {}).get("jvm_parsed"), "jvm")
        if isinstance(jvm_struct, dict) and jvm_struct:
            md = render_llm_markdown(jvm_struct, domain="jvm")
            if md.strip():
                llm_replacements["$$answer_jvm$$"] = md

        db_struct = _inject_confidence((results or {}).get("database_parsed"), "database")
        if isinstance(db_struct, dict) and db_struct:
            md = render_llm_markdown(db_struct, domain="database")
            if md.strip():
                llm_replacements["$$answer_database$$"] = md

        kafka_struct = _inject_confidence((results or {}).get("kafka_parsed"), "kafka")
        if isinstance(kafka_struct, dict) and kafka_struct:
            md = render_llm_markdown(kafka_struct, domain="kafka")
            if md.strip():
                llm_replacements["$$answer_kafka$$"] = md

        ms_struct = _inject_confidence((results or {}).get("ms_parsed"), "microservices")
        if isinstance(ms_struct, dict) and ms_struct:
            md = render_llm_markdown(ms_struct, domain="microservices")
            if md.strip():
                llm_replacements["$$answer_ms$$"] = md

        hr_struct = _inject_confidence((results or {}).get("hard_resources_parsed"), "hard_resources")
        if isinstance(hr_struct, dict) and hr_struct:
            md = render_llm_markdown(hr_struct, domain="hard_resources")
            if md.strip():
                llm_replacements["$$answer_hard_resources$$"] = md

        def _try_json_to_markdown(raw_text: str, domain: str | None) -> str | None:
            if not isinstance(raw_text, str) or not raw_text.strip():
                return None
            try:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    return None
                candidate = raw_text[start:end + 1]
                parsed = None
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    try:
                        parsed = ast.literal_eval(candidate)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    md = render_llm_markdown(parsed, domain=domain)
                    return md.strip() or None
            except Exception:
                return None
            return None

        def _ensure_markdown_for_placeholder(ph: str):
            val = llm_replacements.get(ph)
            if not isinstance(val, str):
                return
            md = _try_json_to_markdown(val, {
                "$$answer_jvm$$": "jvm",
                "$$answer_database$$": "database",
                "$$answer_kafka$$": "kafka",
                "$$answer_ms$$": "microservices",
                "$$answer_hard_resources$$": "hard_resources",
            }.get(ph))
            if md:
                llm_replacements[ph] = md

        for ph in ("$$answer_jvm$$", "$$answer_database$$", "$$answer_kafka$$", "$$answer_ms$$", "$$answer_hard_resources$$"):
            _ensure_markdown_for_placeholder(ph)

        # Добавим данные судьи/программной оценки в раздел доверия (для всех доменов и финала)
        try:
            scores = (results or {}).get("scores", {})

            def _pct(x: float | None) -> str:
                try:
                    if x is None:
                        return "—"
                    return f"{int(float(x)*100)}%"
                except Exception:
                    return "—"

            def _append_judge(ph: str, domain_key: str):
                val = llm_replacements.get(ph)
                if not isinstance(val, str) or not val.strip():
                    return
                s = (scores or {}).get(domain_key) or {}
                judge = (s or {}).get("judge") or {}
                overall = judge.get("overall")
                factual = judge.get("factual")
                completeness = judge.get("completeness")
                specificity = judge.get("specificity")
                data_score = (s or {}).get("data_score")
                data_score_details = (s or {}).get("data_score_details") or {}
                final_score = (s or {}).get("final_score")
                conf = (s or {}).get("confidence")
                judge_meta = (s or {}).get("judge_meta") or {}
                judge_skipped = bool(judge_meta.get("skipped"))
                if judge_meta.get("unavailable"):
                    # The judge ran but gave no usable scores; the result rests on the data check.
                    rows = [
                        ("Судья", "не дал оценку"),
                        ("Эвристическая проверка по данным", _pct(data_score)),
                        ("Итог без судьи", _pct(final_score)),
                    ]
                    if isinstance(conf, (int, float)):
                        rows.append(("Уверенность модели", _pct(float(conf))))
                elif judge_skipped:
                    # One answer from the model: nothing to choose from, so only the data check ran.
                    rows = [
                        ("Судья", "не запускался: модель дала один вариант ответа"),
                        ("Эвристическая проверка по данным", _pct(data_score)),
                    ]
                    if isinstance(conf, (int, float)):
                        rows.append(("Уверенность модели", _pct(float(conf))))
                else:
                    rows = [
                        ("Оценка текста судьей", _pct(overall)),
                        ("Эвристическая проверка по данным", _pct(data_score)),
                        ("Итог для выбора кандидата", _pct(final_score)),
                        ("Точность относительно данных", _pct(factual)),
                        ("Полнота покрытия важных наблюдений", _pct(completeness)),
                        ("Конкретика по метрикам и компонентам", _pct(specificity)),
                    ]
                    if isinstance(conf, (int, float)):
                        rows.insert(3, ("Уверенность модели", _pct(float(conf))))
                if isinstance(data_score_details, dict) and data_score_details:
                    rows.append(("Привязка выводов к метрикам и сериям", _pct(data_score_details.get("label_grounding"))))
                    claims_total = int(data_score_details.get("numeric_claims_total") or 0)
                    claims_supported = int(data_score_details.get("numeric_claims_supported") or 0)
                    if claims_total > 0:
                        rows.append(
                            (
                                "Совпадение чисел из текста",
                                f"{_pct(data_score_details.get('numeric_grounding'))} "
                                f"(подтверждено {claims_supported} из {claims_total})",
                            )
                        )
                    else:
                        rows.append(("Совпадение чисел из текста", "не проверялось, в findings не найдено явных числовых утверждений"))
                    if bool(data_score_details.get("peak_checked")):
                        rows.append(("Совпадение peak_performance", _pct(data_score_details.get("peak_consistency"))))
                    else:
                        rows.append(("Совпадение peak_performance", "не проверялось"))
                def _md_cell(x: object) -> str:
                    return str(x).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip() or "—"

                judge_lines = [
                    "",
                    "#### Оценка ответа",
                    "| Параметр | Значение |",
                    "|---|---:|",
                ]
                judge_lines.extend(f"| {_md_cell(label)} | {_md_cell(value)} |" for label, value in rows)
                judge_lines.extend([
                    "",
                    "_Проверка по данным остаётся эвристической: она оценивает привязку текста к метрикам, "
                    "совпадение чисел и согласованность peak_performance._",
                ])
                llm_replacements[ph] = val.rstrip() + "\n\n" + "\n".join(judge_lines)

            _append_judge("$$answer_jvm$$", "jvm")
            _append_judge("$$answer_database$$", "database")
            _append_judge("$$answer_kafka$$", "kafka")
            _append_judge("$$answer_ms$$", "microservices")
            _append_judge("$$answer_hard_resources$$", "hard_resources")
            _append_judge("$$answer_lt_framework$$", "lt_framework")
            _append_judge("$$final_answer$$", "final")
        except Exception as e:
            logger.warning("Не удалось добавить оценки судьи: %s", e)

        if llm_replacements:
            # Model output is text: it is encoded for the template, never inserted as storage markup.
            update_confluence_page_multi(url_basic, user, password, copy_page_id, llm_replacements, as_text=True)
        _progress("ИИ-анализ завершён. Финализация отчёта…", 95)
        logger.info("✓ Плейсхолдеры LLM обновлены за один проход")
    except Exception as e:
        logger.error("Ошибка при мульти-обновлении данных LLM: %s", e)

    _progress("Отчёт готов ✅", 100)
    return {
        "page_id": copy_page_id,
        "page_url": page_url,
        "run_name": run_meta["run_name"],
        "run_id": run_meta["run_id"],
    }
