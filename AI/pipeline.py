import copy
import os
import json
import logging
import re
from datetime import datetime
from typing import Any, List, Dict, Optional
from urllib.parse import quote

import requests
import io
import pandas as pd

from settings import CONFIG
from AI.scoring import llm_two_pass_self_consistency
from AI.db_store import save_domain_labeled, save_llm_results
from AI.sla_evaluator import evaluate_sla, extract_target_rps_from_pack


logger = logging.getLogger(__name__)


def _configure_logging():
    level_name = (CONFIG.get("logging", {}).get("level") if CONFIG.get("logging") else "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger().setLevel(level)


def _time_shift_hours() -> int:
    """Смещение времени для человекочитаемых таблиц/меток LLM (часы).

    По умолчанию используется +3 часа (МСК). Можно переопределить в settings.py:

        CONFIG["llm"]["time_shift_hours"] = 0  # либо другое целое число
    """
    try:
        llm_cfg = CONFIG.get("llm") or {}
        val = llm_cfg.get("time_shift_hours", 3)
        return int(val)
    except Exception:
        return 3


def _has_meaningful_system_context(payload) -> bool:
    if isinstance(payload, str):
        return bool(payload.strip())
    if isinstance(payload, list):
        return any(_has_meaningful_system_context(item) for item in payload)
    if isinstance(payload, dict):
        if payload.get("enabled") is False:
            return False
        for key, value in payload.items():
            if key in {"schema_version", "enabled"}:
                continue
            if _has_meaningful_system_context(value):
                return True
    return False


def _system_context_prompt_summary(system_context: dict | None) -> str:
    if not isinstance(system_context, dict) or not _has_meaningful_system_context(system_context):
        return "Контекст системы не задан."

    lines: list[str] = []
    system = system_context.get("system") if isinstance(system_context.get("system"), dict) else {}
    architecture = (
        system_context.get("architecture")
        if isinstance(system_context.get("architecture"), dict)
        else {}
    )
    load_model = (
        system_context.get("load_model")
        if isinstance(system_context.get("load_model"), dict)
        else {}
    )
    operational_context = (
        system_context.get("operational_context")
        if isinstance(system_context.get("operational_context"), dict)
        else {}
    )

    system_name = str(system.get("name") or "").strip()
    system_domain = str(system.get("domain") or "").strip()
    system_desc = str(system.get("description") or "").strip()
    test_goal = str(system.get("test_goal") or "").strip()
    if system_name:
        lines.append(f"- Система: {system_name}")
    if system_domain:
        lines.append(f"- Домен: {system_domain}")
    if system_desc:
        lines.append(f"- Описание: {system_desc}")
    if test_goal:
        lines.append(f"- Цель теста: {test_goal}")

    style = str(architecture.get("style") or "").strip()
    if style:
        lines.append(f"- Архитектурный стиль: {style}")

    components = []
    for item in architecture.get("components") if isinstance(architecture.get("components"), list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or "").strip()
        role = str(item.get("role") or "").strip()
        if name and role:
            components.append(f"{name} ({role})")
        elif name:
            components.append(name)
    if components:
        preview = ", ".join(components[:6])
        if len(components) > 6:
            preview += f" и еще {len(components) - 6}"
        lines.append(f"- Ключевые компоненты: {preview}")

    flows = []
    for item in load_model.get("critical_user_flows") if isinstance(load_model.get("critical_user_flows"), list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or "").strip()
        steps = item.get("steps") if isinstance(item.get("steps"), list) else []
        steps_preview = " -> ".join(str(step).strip() for step in steps if str(step).strip())
        if name and steps_preview:
            flows.append(f"{name}: {steps_preview}")
        elif name:
            flows.append(name)
    if flows:
        preview = "; ".join(flows[:4])
        if len(flows) > 4:
            preview += f"; и еще {len(flows) - 4}"
        lines.append(f"- Критичные потоки: {preview}")

    hotspots = [
        str(item).strip()
        for item in (load_model.get("expected_hotspots") if isinstance(load_model.get("expected_hotspots"), list) else [])
        if str(item).strip()
    ]
    if hotspots:
        lines.append(f"- Ожидаемые точки нагрузки: {', '.join(hotspots[:8])}")

    focus = [
        str(item).strip()
        for item in (operational_context.get("analysis_focus") if isinstance(operational_context.get("analysis_focus"), list) else [])
        if str(item).strip()
    ]
    if focus:
        lines.append(f"- Фокус анализа: {'; '.join(focus[:6])}")

    risks = [
        str(item).strip()
        for item in (operational_context.get("known_risks") if isinstance(operational_context.get("known_risks"), list) else [])
        if str(item).strip()
    ]
    if risks:
        lines.append(f"- Известные риски: {'; '.join(risks[:4])}")

    return "\n".join(lines) if lines else "Контекст системы не задан."


def _augment_prompt_with_system_context(prompt: str, system_context_brief: str) -> str:
    brief = (system_context_brief or "").strip()
    if not brief or brief == "Контекст системы не задан.":
        return prompt
    prefix = (
        "СПРАВОЧНЫЙ КОНТЕКСТ ТЕСТИРУЕМОЙ СИСТЕМЫ:\n"
        f"{brief}\n\n"
        "Используй этот контекст только как справочную информацию. "
        "Не подменяй им факты из метрик и не делай выводов, которые не подтверждаются данными."
    )
    return f"{prefix}\n\n{prompt}" if prompt else prefix


def _deterministic_sla_context(sla_result: Dict[str, Any] | None) -> Dict[str, Any]:
    result = sla_result if isinstance(sla_result, dict) else {}
    raw_checks = result.get("checks") if isinstance(result.get("checks"), list) else []
    checks: List[Dict[str, Any]] = [dict(item) for item in raw_checks if isinstance(item, dict)]
    target_rps_check = next(
        (dict(item) for item in checks if str(item.get("name") or "") == "target_rps"),
        None,
    )
    return {
        "verdict": str(result.get("verdict") or "Недостаточно данных"),
        "summary": str(result.get("summary") or ""),
        "test_mode": str(result.get("test_mode") or ""),
        "checks": checks,
        "target_rps_check": target_rps_check,
        "passed_checks": [str(item.get("name")) for item in checks if item.get("passed") is True],
        "failed_checks": [str(item.get("name")) for item in checks if item.get("passed") is False],
        "unknown_checks": [str(item.get("name")) for item in checks if item.get("passed") is None],
    }


def _reconcile_sla_for_test_profile(
    sla_result: Dict[str, Any] | None,
    test_profile: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Приводит SLA verdict к политике конкретного профиля теста.

    Для stability/soak тестов ресурсные превышения являются рисками, но не
    самостоятельным основанием для "Провал", если primary SLA не нарушены.
    """
    result = dict(sla_result or {})
    mode = str((test_profile or {}).get("mode") or result.get("test_mode") or "").strip().lower()
    checks = [dict(item) for item in (result.get("checks") or []) if isinstance(item, dict)]
    if mode != "stability" or not checks:
        return result

    primary_names = {"target_rps", "error_rate", "p95_latency", "p99_latency"}
    secondary_names = {"cpu_usage", "memory_usage"}
    failed_primary = [
        item for item in checks
        if item.get("passed") is False and str(item.get("name") or "") in primary_names
    ]
    failed_secondary = [
        item for item in checks
        if item.get("passed") is False and str(item.get("name") or "") in secondary_names
    ]
    if failed_primary:
        result["test_mode"] = "stability"
        return result
    if failed_secondary and str(result.get("verdict") or "") == "Провал":
        result["verdict"] = "Есть риски"
        result["test_mode"] = "stability"
        summary = str(result.get("summary") or "").strip()
        note = (
            "Режим stability: ресурсные превышения CPU/memory трактуются как риски, "
            "но без нарушений target_rps/error_rate/p95/p99 не переводят тест в «Провал»."
        )
        result["summary"] = f"{summary}; {note}" if summary else note
        for item in checks:
            if str(item.get("name") or "") in secondary_names:
                item.setdefault("category", "secondary")
            elif str(item.get("name") or "") in primary_names:
                item.setdefault("category", "primary")
        result["checks"] = checks
    else:
        result["test_mode"] = "stability"
    return result


def _test_profile_from_type(test_type: str | None) -> Dict[str, Any]:
    raw = str(test_type or "").strip()
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    stability_aliases = {
        "soak",
        "stability",
        "stable",
        "endurance",
        "long",
        "long_run",
        "longevity",
        "reliability",
        "стабильность",
        "стаб",
        "длительный",
        "длительная_нагрузка",
    }
    stability_markers = (
        "stability",
        "stable",
        "soak",
        "endurance",
        "longevity",
        "reliability",
        "стабил",
        "длительн",
    )
    capacity_aliases = {
        "step",
        "capacity",
        "max",
        "max_performance",
        "stress",
        "load",
        "ступенчатый",
        "поиск_максимума",
    }
    if normalized in stability_aliases or any(marker in normalized for marker in stability_markers):
        mode = "stability"
    elif normalized in capacity_aliases:
        mode = "capacity"
    else:
        mode = "capacity"
    return {
        "test_type": raw,
        "mode": mode,
        "peak_performance_applicable": mode == "capacity",
        "focus": (
            "Оценить удержание заданной нагрузки на всем интервале без накопления деградации."
            if mode == "stability"
            else "Определить максимальную устойчивую производительность и момент деградации."
        ),
    }


def _augment_prompt_with_test_profile(prompt: str, test_profile: Dict[str, Any]) -> str:
    if not isinstance(test_profile, dict):
        return prompt
    if test_profile.get("mode") == "stability":
        prefix = (
            "ПРОФИЛЬ ТЕСТА: STABILITY/SOAK.\n"
            "Это тест стабильности, а не поиск максимальной производительности. "
            "Не определяйте peak_performance и не делайте вывод о максимальном RPS. "
            "Оценивайте удержание заданной нагрузки на всем интервале, просадки RPS, latency, errors/checks, "
            "ресурсные тренды, backlog/lag и признаки накопления деградации. "
            "Если нужно упомянуть производительность, формулируйте это как стабильность под заданной нагрузкой, "
            "а не как максимальную производительность.\n\n"
        )
        return f"{prefix}{prompt or ''}"
    return prompt


def read_prompt_from_file(filename: str) -> str:
    """Читает текст промпта из файла в кодировке UTF-8.

    Параметры:
        filename (str): Абсолютный или относительный путь к шаблону.

    Возвращает:
        str: Содержимое файла.

    Побочные эффекты:
        Выполняет чтение с файловой системы.

    Исключения:
        OSError: если файл не найден или недоступен.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def parse_step_to_seconds(step: str) -> int:
    """Преобразует строковое значение шага агрегации PromQL в секунды.

    Параметры:
        step (str): Значение вида `30s`, `5m` или число секунд.

    Возвращает:
        int: Эквивалент в секундах.
    """
    if step.endswith('m'):
        return int(step[:-1]) * 60
    elif step.endswith('s'):
        return int(step[:-1])
    else:
        return int(step)


def fetch_prometheus_data(prometheus_url: str, start_ts: float, end_ts: float, promql_query: str, step: str) -> dict:
    """Выполняет PromQL-запрос напрямую к API Prometheus.

    Параметры:
        prometheus_url (str): Базовый URL сервера Prometheus.
        start_ts (float): Время начала окна (Unix, секунды).
        end_ts (float): Время окончания окна.
        promql_query (str): Запрос PromQL.
        step (str): Шаг агрегации (`30s`, `5m`, ...).

    Возвращает:
        dict: Сырые данные Prometheus (`status`, `data` и т.д.).

    Побочные эффекты:
        Сетевой HTTP-запрос к Prometheus.

    Исключения:
        requests.HTTPError: Если Prometheus вернул код ошибки.
    """
    step_in_seconds = parse_step_to_seconds(step)
    params = {
        'query': promql_query,
        'start': start_ts,
        'end':   end_ts,
        'step':  step_in_seconds
    }
    url = f'{prometheus_url}/api/v1/query_range'
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _resolve_grafana_prom_ds_id(g_cfg: dict) -> int:
    base_url = g_cfg["base_url"].rstrip("/")
    ds_cfg = g_cfg.get("prometheus_datasource", {})
    auth_cfg = g_cfg.get("auth", {})
    headers = {}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    verify = g_cfg.get("verify_ssl", True)
    if isinstance(ds_cfg.get("id"), int):
        return ds_cfg["id"]
    if ds_cfg.get("uid"):
        url = f"{base_url}/api/datasources/uid/{ds_cfg['uid']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]
    if ds_cfg.get("name"):
        url = f"{base_url}/api/datasources/name/{ds_cfg['name']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]
    url = f"{base_url}/api/datasources"
    resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
    resp.raise_for_status()
    for ds in resp.json():
        if ds.get("type") == "prometheus":
            return ds["id"]
    raise RuntimeError("Не найден Prometheus datasource в Grafana")


def _resolve_grafana_influx_ds_id(g_cfg: dict) -> int:
    base_url = g_cfg["base_url"].rstrip("/")
    ds_cfg = g_cfg.get("influxdb_datasource", {}) or g_cfg.get("prometheus_datasource", {}) or {}
    # допускаем переиспользование ключей uid/name, но по типу будем искать influxdb
    auth_cfg = g_cfg.get("auth", {})
    headers = {}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    verify = g_cfg.get("verify_ssl", True)
    # Если указан id — используем
    if isinstance(ds_cfg.get("id"), int):
        return ds_cfg["id"]
    # Если указан uid — попробуем его
    if ds_cfg.get("uid"):
        url = f"{base_url}/api/datasources/uid/{ds_cfg['uid']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]
    # Если указан name — попробуем его
    if ds_cfg.get("name"):
        url = f"{base_url}/api/datasources/name/{quote(str(ds_cfg['name']), safe='')}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]
    # Иначе постараемся найти любой influx datasource
    url = f"{base_url}/api/datasources"
    resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
    resp.raise_for_status()
    for ds in resp.json():
        if ds.get("type") in ("influxdb", "influxdb2"):
            return ds["id"]
    raise RuntimeError("Не найден InfluxDB datasource в Grafana")


def fetch_influx_data_via_grafana(g_cfg: dict, flux_query: str) -> str:
    """Выполняет Flux-запрос к InfluxDB через Grafana proxy.

    Параметры:
        g_cfg (dict): Конфигурация Grafana (base_url, auth, datasource).
        flux_query (str): Полный Flux-запрос с подстановками.

    Возвращает:
        str: CSV-ответ, возвращённый Grafana.

    Побочные эффекты:
        HTTP-запрос к Grafana с авторизацией.

    Исключения:
        requests.HTTPError: При ошибке ответа Grafana.
    """
    base_url = g_cfg["base_url"].rstrip("/")
    ds_id = _resolve_grafana_influx_ds_id(g_cfg)
    auth_cfg = g_cfg.get("auth", {})
    headers = {"Accept": "application/csv", "Content-Type": "application/json"}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    url = f"{base_url}/api/datasources/proxy/{ds_id}/api/v2/query"
    resp = requests.post(url, headers=headers, auth=auth, json={"query": flux_query}, timeout=60, verify=g_cfg.get("verify_ssl", True))
    resp.raise_for_status()
    return resp.text


def fetch_influx_and_aggregate_via_grafana(
    grafana_cfg: dict,
    influx_aux_cfg: dict,
    start_ts: float,
    end_ts: float,
    flux_queries: List[str],
    label_tag_keys_list: List[List[str]],
    labels: List[str],
    resample_interval: str
) -> List[pd.DataFrame]:
    """Получает Flux-метрики через Grafana proxy и формирует pivot-таблицы.

    Параметры:
        grafana_cfg (dict): Подключение к Grafana (URL, auth, datasource).
        influx_aux_cfg (dict): Параметры Influx (`bucket`, org и т.д.).
        start_ts (float): Начало интервала (Unix, секунды).
        end_ts (float): Конец интервала.
        flux_queries (list[str]): Список Flux-запросов с плейсхолдерами `{bucket}`, `{start}`, `{end}`.
        label_tag_keys_list (list[list[str]]): Теги, которые попадут в подпись серии.
        labels (list[str]): Человеко-читаемые подписи секций для Markdown.
        resample_interval (str): Период ресемплинга pandas (`5T`, `1H`, ...).

    Возвращает:
        list[pd.DataFrame]: Набор pivot-таблиц (по одному на запрос).

    Побочные эффекты:
        Делает HTTP-запросы к Grafana и хранит CSV в памяти.

    Исключения:
        Подавляет ошибки отдельных запросов, возвращая пустые DataFrame.
    """
    bucket = (influx_aux_cfg or {}).get("bucket", "")
    t_start = _iso8601_utc(start_ts)
    t_end = _iso8601_utc(end_ts)
    dfs: List[pd.DataFrame] = []
    for idx, flux in enumerate(flux_queries):
        try:
            q = (flux or "").replace("{bucket}", bucket).replace("{start}", t_start).replace("{end}", t_end)
            csv_text = fetch_influx_data_via_grafana(grafana_cfg, q)
            df = pd.read_csv(io.StringIO(csv_text))
            if "_time" not in df.columns or "_value" not in df.columns:
                dfs.append(pd.DataFrame())
                continue
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df = df.dropna(subset=["_time", "_value"])
            tag_keys = label_tag_keys_list[idx] if idx < len(label_tag_keys_list) else []
            tag_keys = list(tag_keys or [])
            def make_label(row):
                parts=[]
                for k in tag_keys:
                    if k in row and pd.notnull(row[k]):
                        parts.append(f"{k}={row[k]}")
                return "|".join(parts) if parts else "series"
            df["series"] = df.apply(make_label, axis=1)
            pivot = df.pivot_table(index="_time", columns="series", values="_value", aggfunc="mean")
            try:
                pivot = pivot.resample(resample_interval).mean()
            except Exception:
                pass
            pivot.index = pd.to_datetime(pivot.index, utc=True)
            dfs.append(pivot)
        except Exception:
            dfs.append(pd.DataFrame())
    return dfs


def _convert_pd_offset_to_influx_interval(s: str) -> str:
    if not isinstance(s, str) or not s:
        return "1m"
    s = s.strip()
    if s.endswith("T"):  # minutes
        try:
            return f"{int(s[:-1])}m"
        except Exception:
            return "1m"
    if s.endswith("S"):  # seconds
        try:
            return f"{int(s[:-1])}s"
        except Exception:
            return "60s"
    if s.endswith("H"):
        try:
            return f"{int(s[:-1])}h"
        except Exception:
            return "1h"
    return "1m"


def fetch_influxql_via_grafana(g_cfg: dict, q: str, database: str | None) -> dict:
    """Выполняет InfluxQL-запрос через Grafana proxy и возвращает JSON.

    Параметры:
        g_cfg (dict): Конфигурация Grafana.
        q (str): InfluxQL-запрос.
        database (str | None): Имя базы для параметра `db`.

    Возвращает:
        dict: JSON-ответ Grafana (совместим с API InfluxDB v1).
    """
    base_url = g_cfg["base_url"].rstrip("/")
    ds_id = _resolve_grafana_influx_ds_id(g_cfg)
    auth_cfg = g_cfg.get("auth", {})
    headers = {"Accept": "application/json"}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    params = {"q": q}
    if database:
        params["db"] = database
    url = f"{base_url}/api/datasources/proxy/{ds_id}/query"
    resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=60, verify=g_cfg.get("verify_ssl", True))
    resp.raise_for_status()
    return resp.json()


def _render_influxql_template(query: str, start_ts: float, end_ts: float, interval: str) -> str:
    """Подставляет поддерживаемые Grafana-макросы перед прямым запросом к InfluxDB."""
    t_start_ns = int(start_ts * 1_000_000_000)
    t_end_ns = int(end_ts * 1_000_000_000)
    rendered = (query or "")
    rendered = rendered.replace("$timeFilter", f"time >= {t_start_ns} AND time <= {t_end_ns}")
    rendered = rendered.replace("$__interval", interval)
    for name in ("Group", "Tag", "URL", "Measurement"):
        rendered = re.sub(r"\$\{" + re.escape(name) + r"(?::[^}]*)?\}", ".*", rendered)
        rendered = rendered.replace(f"${name}", ".*")
    return rendered


def _safe_query_excerpt(query: str, limit: int = 500) -> str:
    text = str(query or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def fetch_influxql_and_aggregate_via_grafana(
    grafana_cfg: dict,
    influx_aux_cfg: dict,
    start_ts: float,
    end_ts: float,
    influxql_queries: List[str],
    label_tag_keys_list: List[List[str]],
    labels: List[str],
    resample_interval: str
) -> List[pd.DataFrame]:
    """Получает InfluxQL-серии через Grafana proxy и строит DataFrame.

    Параметры:
        grafana_cfg (dict): Конфигурация Grafana/datasource.
        influx_aux_cfg (dict): Доп. параметры (`database`).
        start_ts (float): Начало интервала (Unix, секунды).
        end_ts (float): Конец интервала.
        influxql_queries (list[str]): Список InfluxQL-запросов с макросами (`$timeFilter`, `$__interval` и т.п.).
        label_tag_keys_list (list[list[str]]): Теги для подписи серий.
        labels (list[str]): Человеко-читаемые подписи (используются при Markdown).
        resample_interval (str): Интервал ресемплинга pandas.

    Возвращает:
        list[pd.DataFrame]: Pivot-таблицы по каждому запросу.

    Побочные эффекты:
        Выполняет HTTP-запросы к Grafana; подавляет ошибки, возвращая пустые DataFrame.
    """
    iv = _convert_pd_offset_to_influx_interval(resample_interval)
    database = (influx_aux_cfg or {}).get("database")
    dfs: List[pd.DataFrame] = []
    for idx, raw in enumerate(influxql_queries or []):
        try:
            q = _render_influxql_template(raw or "", start_ts=start_ts, end_ts=end_ts, interval=iv)
            logger.info("InfluxQL[%s] rendered query: %s", idx, _safe_query_excerpt(q))
            data = fetch_influxql_via_grafana(grafana_cfg, q, database)
            result_obj = ((data or {}).get("results") or [{}])[0]
            if isinstance(result_obj, dict) and result_obj.get("error"):
                logger.warning("InfluxQL[%s] response error: %s", idx, result_obj.get("error"))
            series_list = (result_obj.get("series") if isinstance(result_obj, dict) else None) or []
            logger.info("InfluxQL[%s] returned series_count=%s", idx, len(series_list))
            # Объединим все series в один pivot
            frames = []
            tag_keys = label_tag_keys_list[idx] if idx < len(label_tag_keys_list) else []
            tag_keys = list(tag_keys or [])
            for s in series_list:
                cols = s.get("columns") or []
                values = s.get("values") or []
                tags = s.get("tags") or {}
                if not values:
                    continue
                df = pd.DataFrame(values, columns=cols)
                if "time" not in df.columns:
                    # иногда колонка может называться "time"
                    continue
                df["time"] = pd.to_datetime(df["time"], utc=True)
                # label from tags
                parts = []
                for k in tag_keys:
                    if k in tags:
                        parts.append(f"{k}={tags[k]}")
                label = "|".join(parts) if parts else s.get("name") or "series"
                # value column: take first numeric column except time
                val_col = None
                for c in df.columns:
                    if c == "time":
                        continue
                    if pd.api.types.is_numeric_dtype(df[c]):
                        val_col = c
                        break
                if not val_col:
                    # fallback: try column named 'sum' or 'percentile'
                    for c in ("sum", "mean", "percentile", "value"):
                        if c in df.columns:
                            val_col = c
                            break
                if not val_col:
                    continue
                # Приведём значения к числу для устойчивого ресемплинга
                try:
                    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
                except Exception:
                    pass
                df = df.dropna(subset=[val_col])
                tmp = df[["time", val_col]].rename(columns={val_col: label}).set_index("time")
                frames.append(tmp)
            if not frames:
                dfs.append(pd.DataFrame())
                continue
            merged = pd.concat(frames, axis=1).sort_index()
            # опциональная ресемплинг
            try:
                merged = merged.resample(resample_interval).mean()
            except Exception:
                pass
            dfs.append(merged)
        except Exception as exc:
            logger.warning("InfluxQL[%s] failed: %s; query=%s", idx, exc, _safe_query_excerpt(q if 'q' in locals() else raw))
            dfs.append(pd.DataFrame())
    return dfs

def fetch_prometheus_data_via_grafana(g_cfg: dict, start_ts: float, end_ts: float, promql_query: str, step: str) -> dict:
    """Выполняет PromQL через Grafana `/api/datasources/proxy/...`.

    Параметры:
        g_cfg (dict): Конфигурация Grafana (включая datasource Prometheus).
        start_ts (float): Начало окна (Unix, секунды).
        end_ts (float): Конец окна.
        promql_query (str): Запрос PromQL.
        step (str): Шаг агрегации.

    Возвращает:
        dict: JSON-ответ Prometheus (через прокси Grafana).
    """
    step_in_seconds = parse_step_to_seconds(step)
    base_url = g_cfg["base_url"].rstrip("/")
    ds_id = _resolve_grafana_prom_ds_id(g_cfg)
    params = {
        'query': promql_query,
        'start': start_ts,
        'end':   end_ts,
        'step':  step_in_seconds
    }
    headers = {}
    auth = None
    auth_cfg = g_cfg.get("auth", {})
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    url = f"{base_url}/api/datasources/proxy/{ds_id}/api/v1/query_range"
    resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=30, verify=g_cfg.get("verify_ssl", True))
    resp.raise_for_status()
    return resp.json()


def fetch_metric_series(prometheus_url: str, start_ts: float, end_ts: float, promql_query: str, step: str, ef_config: dict | None = None) -> dict:
    """Абстрагирует выбор источника метрик (Prometheus напрямую или через Grafана).

    Параметры:
        prometheus_url (str): URL прямого Prometheus (используется в режиме `prometheus`).
        start_ts/end_ts (float): Интервал времени.
        promql_query (str): Запрос PromQL.
        step (str): Шаг агрегации.
        ef_config (dict | None): Эффективная конфигурация, чтобы определить тип источника.

    Возвращает:
        dict: Ответ Prometheus (напрямую или через Grafana).
    """
    cfg = ef_config or CONFIG
    src = (cfg.get("metrics_source", {}).get("type") or "prometheus").lower()
    if src == "grafana_proxy":
        g_cfg = cfg.get("metrics_source", {}).get("grafana", {})
        return fetch_prometheus_data_via_grafana(g_cfg, start_ts, end_ts, promql_query, step)
    else:
        prometheus_url_eff = (cfg.get("metrics_source", {}).get("prometheus", {}) or {}).get("url") or prometheus_url
        return fetch_prometheus_data(prometheus_url_eff, start_ts, end_ts, promql_query, step)


def fetch_and_aggregate_with_label_keys(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_queries: List[str],
    label_keys_list: List[List[str]],
    step: str,
    resample_interval: str,
    ef_config: dict | None = None
) -> List[pd.DataFrame]:
    """Выполняет набор PromQL-запросов и приводит данные к pivot-таблицам.

    Параметры:
        prometheus_url (str): URL Prometheus (используется при прямом доступе).
        start_ts/end_ts (float): Интервал времени.
        promql_queries (list[str]): Список PromQL-запросов.
        label_keys_list (list[list[str]]): Ключи лейблов для подписи серий (по порядку запросов).
        step (str): Шаг агрегации.
        resample_interval (str): Интервал ресемплинга pandas (например, `1T`).
        ef_config (dict | None): Конфиг для выбора источника метрик.

    Возвращает:
        list[pd.DataFrame]: Pivot-таблицы (пустые, если данных нет).

    Исключения:
        ValueError: Если длины списков запросов и лейблов не совпадают.
    """
    if len(promql_queries) != len(label_keys_list):
        raise ValueError("Количество запросов и количество списков лейблов не совпадает!")
    dfs = []
    for query, keys_for_this_query in zip(promql_queries, label_keys_list):
        data_json = fetch_metric_series(prometheus_url, start_ts, end_ts, query, step, ef_config=ef_config)
        records = []
        if data_json.get("status") == "success":
            result = data_json["data"].get("result", [])
            for series in result:
                lbls = series.get("metric", {})
                label_parts = []
                for key in keys_for_this_query:
                    val = lbls.get(key, "unknown")
                    label_parts.append(f"{key}={val}")
                label_str = "|".join(label_parts)
                for (ts_float, value_str) in series["values"]:
                    val = float(value_str)
                    records.append([ts_float, label_str, val])
        if not records:
            df = pd.DataFrame(columns=["ts", "series", "value"])  # пустая
        else:
            # Аггрегируем возможные дубликаты (ts, series), затем пивот
            tmp = pd.DataFrame(records, columns=["ts", "series", "value"]).groupby(["ts", "series"], as_index=False)["value"].mean()
            df = tmp.pivot(index="ts", columns="series", values="value")
            df.index = pd.to_datetime(df.index, unit='s')
            try:
                df = df.resample(resample_interval).mean()
            except Exception:
                pass
        dfs.append(df)
    return dfs


def _iso8601_utc(ts: float) -> str:
    try:
        return pd.to_datetime(ts, unit='s', utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_influx_and_aggregate(
    influx_cfg: dict,
    start_ts: float,
    end_ts: float,
    flux_queries: List[str],
    label_tag_keys_list: List[List[str]],
    labels: List[str],
    resample_interval: str
) -> List[pd.DataFrame]:
    """Получает данные напрямую из InfluxDB (API v2) и строит pivot-таблицы.

    Параметры:
        influx_cfg (dict): Конфигурация InfluxDB (`url`, `org`, `bucket`, `token`).
        start_ts/end_ts (float): Интервал времени.
        flux_queries (list[str]): Список Flux-запросов.
        label_tag_keys_list (list[list[str]]): Теги для формирования названий серий.
        labels (list[str]): Заголовки секций (используются при выводе).
        resample_interval (str): Интервал ресемплинга pandas.

    Возвращает:
        list[pd.DataFrame]: Pivot-таблицы по каждому запросу.

    Побочные эффекты:
        Выполняет HTTP-запросы к InfluxDB.
    """
    url = (influx_cfg or {}).get("url", "").rstrip("/")
    org = (influx_cfg or {}).get("org", "")
    bucket = (influx_cfg or {}).get("bucket", "")
    token = (influx_cfg or {}).get("token", "")
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/csv",
        "Content-Type": "application/json",
    }
    t_start = _iso8601_utc(start_ts)
    t_end = _iso8601_utc(end_ts)
    dfs: List[pd.DataFrame] = []
    for idx, flux in enumerate(flux_queries):
        try:
            q = (flux or "").replace("{bucket}", bucket).replace("{start}", t_start).replace("{end}", t_end)
            resp = requests.post(
                f"{url}/api/v2/query",
                params={"org": org},
                headers=headers,
                json={"query": q},
                timeout=60
            )
            resp.raise_for_status()
            csv_text = resp.text
            df = pd.read_csv(io.StringIO(csv_text))
            # ожидаемые колонки: _time, _value и теги для серии
            if "_time" not in df.columns or "_value" not in df.columns:
                dfs.append(pd.DataFrame())
                continue
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df = df.dropna(subset=["_time", "_value"])
            tag_keys = label_tag_keys_list[idx] if idx < len(label_tag_keys_list) else []
            tag_keys = list(tag_keys or [])
            def make_label(row):
                parts=[]
                for k in tag_keys:
                    if k in row and pd.notnull(row[k]):
                        parts.append(f"{k}={row[k]}")
                return "|".join(parts) if parts else "series"
            df["series"] = df.apply(make_label, axis=1)
            # Пивот по времени/серии
            pivot = df.pivot_table(index="_time", columns="series", values="_value", aggfunc="mean")
            try:
                pivot = pivot.resample(resample_interval).mean()
            except Exception:
                pass
            pivot.index = pd.to_datetime(pivot.index, utc=True)
            dfs.append(pivot)
        except Exception:
            dfs.append(pd.DataFrame())
    return dfs


def dataframes_to_markdown(labeled: List[Dict[str, object]]) -> str:
    """Генерирует Markdown-представление нескольких DataFrame для отчёта.

    Параметры:
        labeled (list[dict]): Список вида `{"label": str, "df": DataFrame}`.

    Возвращает:
        str: Markdown-текст со сводкой по каждой таблице.
    """
    lines = []
    shift_h = _time_shift_hours()
    for item in labeled:
        label = str(item.get("label") or "?")
        df = item.get("df")
        # Сдвигаем индекс времени для отображения (UTC -> локальное время) только в копии
        if shift_h and hasattr(df, "index") and isinstance(getattr(df, "index", None), pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = df.index + pd.to_timedelta(shift_h, unit="h")
            except Exception:
                pass
        lines.append(f"### {label}")
        try:
            md = (df.fillna("") if hasattr(df, 'fillna') else df).head(20).to_markdown() if df is not None else "(пусто)"
        except Exception:
            md = str(getattr(df, 'shape', None))
        lines.append(md)
        lines.append("")
    return "\n".join(lines)


def _find_stable_peak(
    col_series: pd.Series,
    min_stable_minutes: float = 5.0,
    max_cv: float = 0.20,
) -> Optional[Dict[str, object]]:
    s = col_series.dropna()
    try:
        s = s.sort_index()
    except Exception:
        pass
    if s.empty or len(s) < 3 or not isinstance(s.index, pd.DatetimeIndex):
        return None

    deltas = s.index.to_series().diff().dropna()
    if deltas.empty:
        return None
    step_sec = deltas.median().total_seconds()
    if step_sec <= 0:
        return None

    window_samples = max(3, int(round(min_stable_minutes * 60 / step_sec)))
    if len(s) < window_samples:
        return None

    rolling_mean = s.rolling(window=window_samples, min_periods=window_samples).mean()
    rolling_std = s.rolling(window=window_samples, min_periods=window_samples).std()
    cv = rolling_std / rolling_mean.clip(lower=1e-9)
    stable_mask = (cv <= max_cv) & rolling_mean.notna()
    stable_means = rolling_mean[stable_mask]

    if stable_means.empty:
        rm = rolling_mean.dropna()
        if rm.empty:
            return None
        return {
            "stable_max": float(rm.max()),
            "stable_max_time": str(rm.idxmax()),
            "stable_duration_min": min_stable_minutes,
            "method": "rolling_mean_fallback",
        }

    segments: List[Dict[str, object]] = []
    seg_start = None
    seg_end = None
    seg_vals: List[float] = []
    max_gap_sec = max(step_sec * 1.5, 1.0)
    level_jump_ratio = 0.15

    for ts, val in stable_means.items():
        try:
            v = float(val)
        except Exception:
            continue
        if seg_start is None:
            seg_start = ts
            seg_end = ts
            seg_vals = [v]
            continue
        gap_sec = float((ts - seg_end).total_seconds())
        prev_v = float(seg_vals[-1]) if seg_vals else v
        rel_jump = abs(v - prev_v) / max(abs(prev_v), 1e-9)
        if gap_sec <= max_gap_sec and rel_jump <= level_jump_ratio:
            seg_end = ts
            seg_vals.append(v)
            continue
        duration_min = ((seg_end - seg_start).total_seconds() + step_sec) / 60.0
        level = float(pd.Series(seg_vals).median()) if seg_vals else 0.0
        segments.append(
            {"start": seg_start, "end": seg_end, "level": level, "duration_min": float(max(duration_min, 0.0))}
        )
        seg_start = ts
        seg_end = ts
        seg_vals = [v]

    if seg_start is not None and seg_end is not None and seg_vals:
        duration_min = ((seg_end - seg_start).total_seconds() + step_sec) / 60.0
        level = float(pd.Series(seg_vals).median()) if seg_vals else 0.0
        segments.append(
            {"start": seg_start, "end": seg_end, "level": level, "duration_min": float(max(duration_min, 0.0))}
        )

    if not segments:
        return {
            "stable_max": float(stable_means.max()),
            "stable_max_time": str(stable_means.idxmax()),
            "stable_duration_min": min_stable_minutes,
            "method": "stable_window_fallback",
        }

    peak_level = max(float(seg.get("level", 0.0)) for seg in segments)
    top_segments = [seg for seg in segments if float(seg.get("level", 0.0)) >= peak_level * 0.90]
    candidate_pool = top_segments if top_segments else segments
    chosen = max(candidate_pool, key=lambda seg: (seg.get("end"), seg.get("level", 0.0)))

    return {
        "stable_max": float(chosen.get("level", 0.0)),
        "stable_max_time": str(chosen.get("end")),
        "stable_duration_min": float(chosen.get("duration_min", min_stable_minutes)),
        "stable_start_time": str(chosen.get("start")),
        "method": "last_stable_step",
    }


def _cfg_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(raw: Any, default: int) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


def _cfg_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return bool(raw)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _longest_true_run(values: List[bool]) -> int:
    best = 0
    cur = 0
    for v in values:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _first_true_run_start(values: List[bool], run_len: int) -> Optional[int]:
    need = max(1, int(run_len))
    cur = 0
    for idx, v in enumerate(values):
        if v:
            cur += 1
            if cur >= need:
                return idx - need + 1
        else:
            cur = 0
    return None


def _slope_rps_per_min(series_vals: pd.Series) -> float:
    if series_vals is None or len(series_vals) < 2:
        return 0.0
    try:
        y = [float(v) for v in series_vals.values]
    except Exception:
        return 0.0
    n = len(y)
    if n < 2:
        return 0.0
    if isinstance(getattr(series_vals, "index", None), pd.DatetimeIndex):
        t0 = series_vals.index[0]
        x = [float((ts - t0).total_seconds()) / 60.0 for ts in series_vals.index]
    else:
        x = [float(i) for i in range(n)]
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = 0.0
    den = 0.0
    for xi, yi in zip(x, y):
        dx = xi - x_mean
        num += dx * (yi - y_mean)
        den += dx * dx
    if den <= 0:
        return 0.0
    return float(num / den)


def _segment_end_sort_key(value: Any) -> float:
    try:
        return float(pd.Timestamp(value).value)
    except Exception:
        return float("-inf")


def _select_step_profile_candidate(segments: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    stable_segments = [seg for seg in segments if bool(seg.get("stable"))]
    if not stable_segments:
        return None
    terminal_stable_segments: List[Dict[str, object]] = []
    for idx, seg in enumerate(segments):
        if not bool(seg.get("stable")):
            continue
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
        if next_seg is not None and bool(next_seg.get("stable")):
            continue
        terminal_stable_segments.append(seg)
    if terminal_stable_segments:
        return max(
            terminal_stable_segments,
            key=lambda seg: (_segment_end_sort_key(seg.get("end")), float(seg.get("level", 0.0))),
        )
    return max(
        stable_segments,
        key=lambda seg: (float(seg.get("level", 0.0)), _segment_end_sort_key(seg.get("end"))),
    )


def _find_stable_peak_step_profile(
    col_series: pd.Series,
    min_stable_minutes: float,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, object]]:
    s = col_series.dropna()
    try:
        s = s.sort_index()
    except Exception:
        pass
    if s.empty or len(s) < 3 or not isinstance(s.index, pd.DatetimeIndex):
        return None

    resample_sec = max(1, _cfg_int(cfg.get("step_detection_resample_sec"), 15))
    smooth_sec = max(resample_sec, _cfg_int(cfg.get("step_detection_smooth_sec"), 60))
    confirm_hold_sec = max(resample_sec, _cfg_int(cfg.get("step_confirm_hold_sec"), 180))
    min_step_delta_rps = _cfg_float(cfg.get("step_min_step_delta_rps"), 15.0)
    min_step_delta_pct = _cfg_float(cfg.get("step_min_step_delta_pct"), 0.08)
    max_cv = _cfg_float(cfg.get("step_max_cv"), 0.10)
    max_slope = _cfg_float(cfg.get("step_max_slope_rps_per_min"), 0.5)
    max_drop_pct = _cfg_float(cfg.get("step_max_within_step_drop_pct"), 0.08)
    drop_hold_sec = max(resample_sec, _cfg_int(cfg.get("step_drop_hold_sec"), 120))

    rs = s.resample(f"{resample_sec}s").mean().dropna()
    if len(rs) < 3:
        return None
    rs_deltas = rs.index.to_series().diff().dropna()
    sample_sec = float(resample_sec)
    if not rs_deltas.empty:
        try:
            sample_sec = float(rs_deltas.median().total_seconds())
        except Exception:
            sample_sec = float(resample_sec)
    sample_sec = max(sample_sec, 1.0)

    smooth_samples = max(2, int(round(smooth_sec / sample_sec)))
    hold_samples = max(2, int(round(confirm_hold_sec / sample_sec)))
    smooth = rs.rolling(window=smooth_samples, min_periods=max(2, smooth_samples // 2)).median().dropna()
    if smooth.empty:
        smooth = rs

    values = smooth.values
    n = len(smooth)
    boundaries = [0]
    i = hold_samples
    while i < (n - hold_samples):
        seg_start = boundaries[-1]
        prev_left = max(seg_start, i - hold_samples)
        prev_vals = values[prev_left:i]
        if len(prev_vals) < max(2, hold_samples // 2):
            i += 1
            continue
        prev_level = float(pd.Series(prev_vals).median())
        next_vals = values[i:i + hold_samples]
        next_level = float(pd.Series(next_vals).median())
        delta = abs(next_level - prev_level)
        delta_thr = max(min_step_delta_rps, abs(prev_level) * min_step_delta_pct)
        if delta >= delta_thr:
            boundaries.append(i)
            i += hold_samples
            continue
        i += 1
    boundaries.append(n)

    segments: List[Dict[str, object]] = []
    for bi in range(len(boundaries) - 1):
        a = boundaries[bi]
        b = boundaries[bi + 1]
        if b - a < 2:
            continue
        seg = smooth.iloc[a:b]
        if seg.empty:
            continue
        st = seg.index[0]
        en = seg.index[-1]
        duration_min = ((en - st).total_seconds() + sample_sec) / 60.0
        level = float(seg.median())
        mean_v = float(seg.mean())
        std_v = float(seg.std(ddof=0))
        cv = float(std_v / max(abs(mean_v), 1e-9))
        slope = _slope_rps_per_min(seg)
        base_window = seg.iloc[:max(2, min(len(seg), hold_samples))]
        level_ref = float(base_window.median()) if not base_window.empty else level
        drop_threshold = level_ref * (1.0 - max_drop_pct)
        below = [bool(v < drop_threshold) for v in seg.values]
        longest_bad_min = _longest_true_run(below) * (sample_sec / 60.0)
        drop_run_samples = max(1, int(round(drop_hold_sec / sample_sec)))
        drop_start_idx = _first_true_run_start(below, drop_run_samples)
        has_drop = drop_start_idx is not None

        eval_seg = seg
        drop_time = None
        if has_drop and drop_start_idx is not None and drop_start_idx >= 2:
            eval_seg = seg.iloc[:drop_start_idx]
            drop_time = seg.index[drop_start_idx]
        if eval_seg is None or eval_seg.empty or len(eval_seg) < 2:
            continue

        eval_st = eval_seg.index[0]
        eval_en = eval_seg.index[-1]
        eval_duration_min = ((eval_en - eval_st).total_seconds() + sample_sec) / 60.0
        eval_level = float(eval_seg.median())
        eval_mean = float(eval_seg.mean())
        eval_std = float(eval_seg.std(ddof=0))
        eval_cv = float(eval_std / max(abs(eval_mean), 1e-9))
        eval_slope = _slope_rps_per_min(eval_seg)
        slope_limit = max(float(max_slope), abs(eval_level) * 0.01)
        is_stable = (
            eval_duration_min >= float(min_stable_minutes)
            and eval_cv <= max_cv
            and abs(eval_slope) <= slope_limit
        )
        segments.append({
            "start": eval_st,
            "end": eval_en,
            "level": eval_level,
            "duration_min": float(max(eval_duration_min, 0.0)),
            "cv": eval_cv,
            "slope_rps_per_min": eval_slope,
            "slope_limit_rps_per_min": float(slope_limit),
            "longest_drop_min": float(longest_bad_min),
            "drop_time": str(drop_time) if drop_time is not None else None,
            "had_drop": bool(has_drop),
            "stable": bool(is_stable),
        })

    chosen = _select_step_profile_candidate(segments)
    if not chosen:
        return None
    return {
        "stable_max": float(chosen.get("level", 0.0)),
        "stable_max_time": str(chosen.get("end")),
        "stable_duration_min": float(chosen.get("duration_min", min_stable_minutes)),
        "stable_start_time": str(chosen.get("start")),
        "method": "step_profile",
        "step_segments": [
            {
                "start": str(seg.get("start")),
                "end": str(seg.get("end")),
                "level": float(seg.get("level", 0.0)),
                "duration_min": float(seg.get("duration_min", 0.0)),
                "cv": float(seg.get("cv", 0.0)),
                "slope_rps_per_min": float(seg.get("slope_rps_per_min", 0.0)),
                "slope_limit_rps_per_min": float(seg.get("slope_limit_rps_per_min", 0.0)),
                "longest_drop_min": float(seg.get("longest_drop_min", 0.0)),
                "had_drop": bool(seg.get("had_drop", False)),
                "drop_time": seg.get("drop_time"),
                "stable": bool(seg.get("stable", False)),
            }
            for seg in segments
        ],
    }


def _summarize_time_series_dataframe(
    df: pd.DataFrame,
    top_n: int = 10,
    min_stable_minutes: float = 5.0,
    stable_detection_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    if df is None or getattr(df, 'empty', True):
        return summary
    if not isinstance(df.columns, pd.Index):
        return summary
    ranked_cols: List[tuple[int, str, float]] = []
    for idx, col in enumerate(list(df.columns)):
        try:
            col_series = df.iloc[:, idx]
            if col_series.dropna().empty:
                continue
            ranked_cols.append((idx, str(col), float(col_series.max(skipna=True))))
        except Exception:
            continue
    ranked_cols.sort(key=lambda x: x[2], reverse=True)
    selected = ranked_cols[: max(int(top_n), 0)] if top_n is not None else ranked_cols
    shift_h = _time_shift_hours()
    det_cfg = stable_detection_cfg if isinstance(stable_detection_cfg, dict) else {}
    for col_idx, col_name, _ in selected:
        col_series = df.iloc[:, col_idx]
        if col_series.dropna().empty:
            continue
        try:
            max_val = float(col_series.max(skipna=True))
            min_val = float(col_series.min(skipna=True))
            max_idx = col_series.idxmax()
            min_idx = col_series.idxmin()
            if shift_h:
                try:
                    if hasattr(max_idx, "to_pydatetime"):
                        max_idx = max_idx.to_pydatetime() + pd.to_timedelta(shift_h, unit="h")
                    if hasattr(min_idx, "to_pydatetime"):
                        min_idx = min_idx.to_pydatetime() + pd.to_timedelta(shift_h, unit="h")
                except Exception:
                    pass
            series_summary: Dict[str, object] = {
                "series": col_name,
                "mean": float(col_series.mean(skipna=True)),
                "min": min_val,
                "max": max_val,
                "last": float(col_series.dropna().iloc[-1]),
                "max_time": str(max_idx) if pd.notnull(max_idx) else None,
                "min_time": str(min_idx) if pd.notnull(min_idx) else None,
            }
            stable = None
            if bool(det_cfg.get("use_step_profile")):
                stable = _find_stable_peak_step_profile(
                    col_series,
                    min_stable_minutes=min_stable_minutes,
                    cfg=det_cfg,
                )
            if stable is None:
                stable = _find_stable_peak(col_series, min_stable_minutes=min_stable_minutes)
            if stable:
                series_summary["stable_max"] = stable["stable_max"]
                series_summary["stable_max_time"] = stable.get("stable_max_time")
                series_summary["stable_start_time"] = stable.get("stable_start_time")
                series_summary["stable_duration_min"] = stable.get("stable_duration_min")
                series_summary["stable_method"] = stable.get("method")
                if "step_segments" in stable:
                    series_summary["step_segments"] = stable.get("step_segments")
        except Exception:
            continue
        summary.append(series_summary)
    return summary


def build_context_pack(
    labeled_dfs: List[Dict[str, object]],
    top_n: int = 10,
    min_stable_minutes: float = 5.0,
    stable_detection_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """Формирует компактное описание топовых серий и окон аномалий по домену.

    Параметры:
        labeled_dfs (list[dict]): Список секций с DataFrame.
        top_n (int): Сколько серий на секцию включать в отчёт.

    Возвращает:
        dict: Структура `{"sections": [...]}` с топ-сериями и аномалиями.
    """
    def _detect_anomaly_windows(col_series: pd.Series, sigma: float = 2.0, max_windows: int = 2) -> List[Dict[str, object]]:
        windows: List[Dict[str, object]] = []
        shift_h = _time_shift_hours()
        try:
            s = col_series.dropna()
            if s.empty:
                return windows
            mu = float(s.mean())
            sd = float(s.std(ddof=0))
            if sd == 0 or not pd.notnull(sd):
                return windows
            thr = mu + sigma * sd
            mask = (col_series > thr).fillna(False)
            shifted = mask.astype(int).diff().fillna(int(mask.iloc[0]))
            starts = list(mask.index[shifted == 1])
            if mask.iloc[0]:
                starts = [mask.index[0]] + starts
            ends = list(mask.index[shifted == -1])
            if mask.iloc[-1]:
                ends = ends + [mask.index[-1]]
            for st, en in zip(starts, ends):
                window_slice = col_series.loc[st:en].dropna()
                if window_slice.empty:
                    continue
                peak_val = float(window_slice.max())
                peak_ts = window_slice.idxmax()
                def _fmt(ts_val):
                    try:
                        if shift_h and hasattr(ts_val, "to_pydatetime"):
                            ts_val = ts_val.to_pydatetime() + pd.to_timedelta(shift_h, unit="h")
                    except Exception:
                        pass
                    return str(ts_val)
                windows.append({
                    "start": _fmt(st),
                    "end": _fmt(en),
                    "peak_time": _fmt(peak_ts),
                    "peak": peak_val,
                    "mean": mu,
                    "threshold_high": thr
                })
            if len(windows) > max_windows:
                windows = sorted(windows, key=lambda w: w.get("peak", 0.0), reverse=True)[:max_windows]
        except Exception:
            return []
        return windows

    sections = []
    for item in labeled_dfs:
        label = item.get("label")
        df = item.get("df")
        section_summary = _summarize_time_series_dataframe(
            df,
            top_n=top_n,
            min_stable_minutes=min_stable_minutes,
            stable_detection_cfg=stable_detection_cfg,
        )
        anomalies: List[Dict[str, object]] = []
        if isinstance(df, pd.DataFrame) and not df.empty:
            for s in section_summary:
                series_name = s.get("series")
                if series_name in df.columns:
                    windows = _detect_anomaly_windows(df[series_name])
                    if windows:
                        anomalies.append({
                            "series": series_name,
                            "windows": windows
                        })
        sections.append({
            "label": label,
            "top_series": section_summary,
            "anomalies": anomalies
        })
    return {"sections": sections}


def uploadFromLLM(
    start_ts: float,
    end_ts: float,
    save_to_db: bool = False,
    run_meta: dict | None = None,
    only_collect: bool = False,
    ef_config: dict | None = None,
    prompts_override: dict | None = None,
    active_domains: List[str] | None = None,
    system_context: dict | None = None,
) -> Dict[str, object]:
    """Основной pipeline: сбор метрик, подготовка контекста и вызов LLM.

    Параметры:
        start_ts/end_ts (float): Границы интервала в секундах Unix.
        save_to_db (bool): Если True — сохраняет метрики и LLM-ответы в TimescaleDB.
        run_meta (dict | None): Служебные атрибуты запуска (`run_id`, `run_name`, `service`, `test_type`, `start_ms`, `end_ms`).
        only_collect (bool): При True собирает метрики и сохраняет их в БД, не вызывая LLM.
        ef_config (dict | None): Эффективная конфигурация (используется для override источников/LLM/запросов).
        prompts_override (dict | None): Пользовательские промпты по доменам.
        active_domains (list[str] | None): Ограничение списка доменов, которые нужно обрабатывать.
        system_context (dict | None): Снимок описания тестируемой системы на момент запуска отчета.

    Возвращает:
        dict: Структура с текстовыми блоками, JSON-парсами и оценками качества.

    Побочные эффекты:
        Выполняет сетевые обращения (Grafana/Prometheus/Influx/LLM), может писать в TimescaleDB.

    Исключения:
        Пробрасывает любые ошибки сбора данных или сохранения в БД.
    """
    _configure_logging()
    cfg = ef_config or CONFIG
    active_system_context = (
        copy.deepcopy(system_context)
        if isinstance(system_context, dict)
        else copy.deepcopy(cfg.get("system_context") or {})
    )
    system_context_brief = _system_context_prompt_summary(active_system_context)
    src_type = (cfg.get("metrics_source", {}) or {}).get("type", "prometheus").lower()
    ms_cfg = (cfg.get("metrics_source", {}) or {})
    prometheus_url = (ms_cfg.get("prometheus", {}) or {}).get("url", "")
    step = (cfg.get("default_params", {}) or {}).get("step") or CONFIG["default_params"]["step"]
    resample = (cfg.get("default_params", {}) or {}).get("resample_interval") or CONFIG["default_params"]["resample_interval"]

    sla_early = (cfg.get("sla") or CONFIG.get("sla") or {})
    try:
        min_stable_min = float(sla_early.get("min_stable_minutes", 5.0))
    except (TypeError, ValueError):
        min_stable_min = 5.0
    run_test_type = str((run_meta or {}).get("test_type") or "").strip().lower()
    test_profile = _test_profile_from_type(run_test_type)
    peak_performance_applicable = bool(test_profile.get("peak_performance_applicable", True))
    use_step_profile = _cfg_bool(
        sla_early.get("step_detection_enabled"),
        default=(run_test_type == "step" and peak_performance_applicable),
    )
    preset_raw = str(sla_early.get("step_detection_preset") or "balanced").strip().lower()
    preset_name = preset_raw if preset_raw in {"strict", "balanced", "lenient"} else "balanced"
    preset_map: Dict[str, Dict[str, Any]] = {
        "strict": {
            "step_detection_resample_sec": 10,
            "step_detection_smooth_sec": 45,
            "step_confirm_hold_sec": 240,
            "step_min_step_delta_rps": 20.0,
            "step_min_step_delta_pct": 0.10,
            "step_max_cv": 0.08,
            "step_max_slope_rps_per_min": 0.35,
            "step_max_within_step_drop_pct": 0.06,
            "step_drop_hold_sec": 150,
        },
        "balanced": {
            "step_detection_resample_sec": 15,
            "step_detection_smooth_sec": 60,
            "step_confirm_hold_sec": 180,
            "step_min_step_delta_rps": 8.0,
            "step_min_step_delta_pct": 0.08,
            "step_max_cv": 0.10,
            "step_max_slope_rps_per_min": 0.5,
            "step_max_within_step_drop_pct": 0.08,
            "step_drop_hold_sec": 120,
        },
        "lenient": {
            "step_detection_resample_sec": 20,
            "step_detection_smooth_sec": 90,
            "step_confirm_hold_sec": 120,
            "step_min_step_delta_rps": 10.0,
            "step_min_step_delta_pct": 0.06,
            "step_max_cv": 0.13,
            "step_max_slope_rps_per_min": 0.8,
            "step_max_within_step_drop_pct": 0.12,
            "step_drop_hold_sec": 90,
        },
    }
    preset = preset_map[preset_name]
    lt_stable_cfg: Dict[str, Any] = {
        "use_step_profile": bool(use_step_profile),
        "step_detection_preset": preset_name,
        "debug_peak_logging": _cfg_bool(sla_early.get("debug_peak_logging"), default=False),
        "debug_target_label": str(sla_early.get("max_performance_query") or "").strip(),
        "step_detection_resample_sec": _cfg_int(
            sla_early.get("step_detection_resample_sec", preset["step_detection_resample_sec"]),
            preset["step_detection_resample_sec"],
        ),
        "step_detection_smooth_sec": _cfg_int(
            sla_early.get("step_detection_smooth_sec", preset["step_detection_smooth_sec"]),
            preset["step_detection_smooth_sec"],
        ),
        "step_confirm_hold_sec": _cfg_int(
            sla_early.get("step_confirm_hold_sec", preset["step_confirm_hold_sec"]),
            preset["step_confirm_hold_sec"],
        ),
        "step_min_step_delta_rps": _cfg_float(
            sla_early.get("step_min_step_delta_rps", preset["step_min_step_delta_rps"]),
            preset["step_min_step_delta_rps"],
        ),
        "step_min_step_delta_pct": _cfg_float(
            sla_early.get("step_min_step_delta_pct", preset["step_min_step_delta_pct"]),
            preset["step_min_step_delta_pct"],
        ),
        "step_max_cv": _cfg_float(sla_early.get("step_max_cv", preset["step_max_cv"]), preset["step_max_cv"]),
        "step_max_slope_rps_per_min": _cfg_float(
            sla_early.get("step_max_slope_rps_per_min", preset["step_max_slope_rps_per_min"]),
            preset["step_max_slope_rps_per_min"],
        ),
        "step_max_within_step_drop_pct": _cfg_float(
            sla_early.get("step_max_within_step_drop_pct", preset["step_max_within_step_drop_pct"]),
            preset["step_max_within_step_drop_pct"],
        ),
        "step_drop_hold_sec": _cfg_int(
            sla_early.get("step_drop_hold_sec", preset["step_drop_hold_sec"]),
            preset["step_drop_hold_sec"],
        ),
    }

    queries = cfg.get("queries") or CONFIG.get("queries") or {}
    # Определяем доступные домены (включая lt_framework, если задан)
    domain_keys = ["jvm", "database", "kafka", "microservices", "hard_resources"]
    if isinstance(queries.get("lt_framework"), dict):
        domain_keys.append("lt_framework")
    enabled_domain_set = set(domain_keys if active_domains is None else [d for d in active_domains if d in domain_keys])

    def _is_enabled(key: str) -> bool:
        return active_domains is None or key in enabled_domain_set
    domain_data = {}
    def _empty_domain_payload(key: str) -> Dict[str, object]:
        return {"labeled": [], "markdown": "", "pack": {"sections": []}, "ctx": json.dumps({"domain": key, "sections": []}, ensure_ascii=False)}
    for key in domain_keys:
        try:
            if not _is_enabled(key):
                domain_data[key] = _empty_domain_payload(key)
                continue
            if key == "lt_framework":
                qcfg = queries.get("lt_framework") or {}
                # выбор источника lt: отдельный lt_metrics_source или общий
                lt_src_cfg = cfg.get("lt_metrics_source") or cfg.get("metrics_source") or {}
                lt_type = (lt_src_cfg.get("type") or "prometheus").lower()
                if lt_type in ("prometheus",):
                    dfs = fetch_and_aggregate_with_label_keys(
                        (lt_src_cfg.get("prometheus", {}) or {}).get("url", prometheus_url),
                        start_ts,
                        end_ts,
                        qcfg.get("promql_queries", []),
                        qcfg.get("label_keys_list", []),
                        step=step,
                        resample_interval=resample,
                        ef_config={"metrics_source": lt_src_cfg}
                    )
                elif lt_type == "grafana_proxy":
                    # Для LT-метрик через Grafana proxy InfluxQL приоритетнее PromQL,
                    # если он явно задан в эффективном конфиге области/сервиса.
                    if qcfg.get("influxql_queries"):
                        logger.info(
                            "lt_framework uses Grafana proxy InfluxQL: queries=%s datasource=%s database=%s",
                            len(qcfg.get("influxql_queries") or []),
                            ((lt_src_cfg.get("grafana", {}) or {}).get("influxdb_datasource")
                             or (lt_src_cfg.get("grafana", {}) or {}).get("prometheus_datasource")
                             or {}).get("name"),
                            (lt_src_cfg.get("influxdb", {}) or {}).get("database"),
                        )
                        influx_aux = lt_src_cfg.get("influxdb", {}) or {}
                        dfs = fetch_influxql_and_aggregate_via_grafana(
                            grafana_cfg=lt_src_cfg.get("grafana", {}) or {},
                            influx_aux_cfg=influx_aux,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            influxql_queries=qcfg.get("influxql_queries", []),
                            label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
                            labels=qcfg.get("labels", []),
                            resample_interval=resample
                        )
                    elif qcfg.get("promql_queries"):
                        logger.info(
                            "lt_framework uses Grafana proxy PromQL: queries=%s",
                            len(qcfg.get("promql_queries") or []),
                        )
                        dfs = fetch_and_aggregate_with_label_keys(
                            (lt_src_cfg.get("prometheus", {}) or {}).get("url", prometheus_url),
                            start_ts,
                            end_ts,
                            qcfg.get("promql_queries", []),
                            qcfg.get("label_keys_list", []),
                            step=step,
                            resample_interval=resample,
                            ef_config={"metrics_source": lt_src_cfg}
                        )
                    else:
                        logger.info(
                            "lt_framework uses Grafana proxy Flux: queries=%s",
                            len(qcfg.get("flux_queries") or []),
                        )
                        influx_aux = lt_src_cfg.get("influxdb", {}) or {}
                        dfs = fetch_influx_and_aggregate_via_grafana(
                            grafana_cfg=lt_src_cfg.get("grafana", {}) or {},
                            influx_aux_cfg=influx_aux,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            flux_queries=qcfg.get("flux_queries", []),
                            label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
                            labels=qcfg.get("labels", []),
                            resample_interval=resample
                        )
                elif lt_type == "influxdb":
                    influx_cfg = lt_src_cfg.get("influxdb", {}) or {}
                    dfs = fetch_influx_and_aggregate(
                        influx_cfg=influx_cfg,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        flux_queries=qcfg.get("flux_queries", []),
                        label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
                        labels=qcfg.get("labels", []),
                        resample_interval=resample
                    )
                else:
                    dfs = []
                labeled = label_dataframes(dfs, (queries.get(key, {}) or {}).get("labels", []))
            else:
                dfs = fetch_and_aggregate_with_label_keys(
                    prometheus_url,
                    start_ts,
                    end_ts,
                    queries[key]["promql_queries"],
                    queries[key]["label_keys_list"],
                    step=step,
                    resample_interval=resample,
                    ef_config=cfg
                )
                labeled = label_dataframes(dfs, queries[key]["labels"])
            markdown = dataframes_to_markdown(labeled)
            domain_stable_cfg = lt_stable_cfg if key == "lt_framework" else None
            pack = build_context_pack(
                labeled,
                top_n=15,
                min_stable_minutes=min_stable_min,
                stable_detection_cfg=domain_stable_cfg,
            )
            ctx_obj = {
                "domain": key,
                "time_range": {"start": start_ts, "end": end_ts},
                "test_profile": test_profile,
                **pack,
            }
            if _has_meaningful_system_context(active_system_context):
                ctx_obj["system_context"] = active_system_context
            ctx = json.dumps(ctx_obj, ensure_ascii=False)
            domain_data[key] = {"labeled": labeled, "markdown": markdown, "pack": pack, "ctx": ctx}
        except Exception as e:
            logger.error(f"Domain '{key}' build failed: {e}")
            domain_data[key] = {"labeled": [], "markdown": "", "pack": {"sections": []}, "ctx": json.dumps({"domain": key, "sections": []}, ensure_ascii=False)}

    storage_cfg = ((cfg.get("storage", {}) or {}).get("timescale") or (CONFIG.get("storage", {}) or {}).get("timescale") or {})

    # Сохранение метрик доменов в TimescaleDB
    if save_to_db:
        try:
            for key in domain_keys:
                dd = domain_data.get(key, {})
                labeled = dd.get("labeled") or []
                save_domain_labeled(
                    domain_key=key,
                    domain_conf=queries.get(key, {}),
                    labeled_dfs=labeled,
                    run_meta={
                        **(run_meta or {}),
                        "start_ms": int((run_meta or {}).get("start_ms") or int(start_ts * 1000)),
                        "end_ms": int((run_meta or {}).get("end_ms") or int(end_ts * 1000)),
                    },
                    storage_cfg=storage_cfg
                )
        except Exception as e:
            logger.error(f"Failed to save domain data to TimescaleDB: {e}")

    if only_collect:
        return {}

    prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    prompt_jvm = read_prompt_from_file(os.path.join(prompt_dir, "jvm_prompt.txt"))
    prompt_database = read_prompt_from_file(os.path.join(prompt_dir, "database_prompt.txt"))
    prompt_kafka = read_prompt_from_file(os.path.join(prompt_dir, "kafka_prompt.txt"))
    prompt_microservices = read_prompt_from_file(os.path.join(prompt_dir, "microservices_prompt.txt"))
    prompt_hard_resources = read_prompt_from_file(os.path.join(prompt_dir, "hard_resources_prompt.txt"))
    prompt_overall = read_prompt_from_file(os.path.join(prompt_dir, "overall_prompt.txt"))
    prompt_lt_framework = read_prompt_from_file(os.path.join(prompt_dir, "lt_framework_prompt.txt")) if os.path.exists(os.path.join(prompt_dir, "lt_framework_prompt.txt")) else "Проанализируйте метрики инструмента нагрузочного тестирования (lt_framework)."
    if isinstance(prompts_override, dict):
        prompt_jvm = prompts_override.get("jvm", prompt_jvm)
        prompt_database = prompts_override.get("database", prompt_database)
        prompt_kafka = prompts_override.get("kafka", prompt_kafka)
        prompt_microservices = prompts_override.get("microservices", prompt_microservices)
        prompt_hard_resources = prompts_override.get("hard_resources", prompt_hard_resources)
        prompt_overall = prompts_override.get("overall", prompt_overall)
        prompt_lt_framework = prompts_override.get("lt_framework", prompt_lt_framework)

    def _augment_prompt(prompt: str) -> str:
        return _augment_prompt_with_test_profile(
            _augment_prompt_with_system_context(prompt, system_context_brief),
            test_profile,
        )

    prompt_jvm = _augment_prompt(prompt_jvm)
    prompt_database = _augment_prompt(prompt_database)
    prompt_kafka = _augment_prompt(prompt_kafka)
    prompt_microservices = _augment_prompt(prompt_microservices)
    prompt_hard_resources = _augment_prompt(prompt_hard_resources)
    prompt_lt_framework = _augment_prompt(prompt_lt_framework)

    include_tables = bool(((CONFIG.get("llm", {}) or {}).get("include_markdown_tables_in_context", False)))
    jvm_full_data = domain_data["jvm"]["markdown"]; jvm_pack = domain_data["jvm"]["pack"]; jvm_ctx = domain_data["jvm"]["ctx"]
    database_full_data = domain_data["database"]["markdown"]; database_pack = domain_data["database"]["pack"]; database_ctx = domain_data["database"]["ctx"]
    kafka_full_data = domain_data["kafka"]["markdown"]; kafka_pack = domain_data["kafka"]["pack"]; kafka_ctx = domain_data["kafka"]["ctx"]
    ms_full_data = domain_data["microservices"]["markdown"]; ms_pack = domain_data["microservices"]["pack"]
    hr_full_data = domain_data["hard_resources"]["markdown"]; hr_pack = domain_data["hard_resources"]["pack"]; hr_ctx = domain_data["hard_resources"]["ctx"]
    lt_full_data = domain_data.get("lt_framework", {}).get("markdown", "")
    lt_pack = domain_data.get("lt_framework", {}).get("pack", {})

    cpu_sections = []
    mem_sections = []
    try:
        for sec in jvm_pack.get("sections", []):
            lbl = str(sec.get("label", ""))
            if "Process CPU usage" in lbl:
                cpu_sections.append(sec)
            if "Heap used" in lbl or "Heap max" in lbl:
                mem_sections.append(sec)
    except Exception:
        pass
    ms_ctx_obj = {
        "domain": "microservices",
        "time_range": {"start": start_ts, "end": end_ts},
        "test_profile": test_profile,
        **ms_pack,
        "aux_resources": {
            "cpu_sections": cpu_sections,
            "memory_sections": mem_sections
        }
    }
    if _has_meaningful_system_context(active_system_context):
        ms_ctx_obj["system_context"] = active_system_context
    ms_ctx = json.dumps(ms_ctx_obj, ensure_ascii=False)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    domains_jobs = []
    if "jvm" in domain_keys and _is_enabled("jvm"):
        domains_jobs.append(("jvm", prompt_jvm, jvm_ctx))
    if "database" in domain_keys and _is_enabled("database"):
        domains_jobs.append(("database", prompt_database, database_ctx))
    if "kafka" in domain_keys and _is_enabled("kafka"):
        domains_jobs.append(("kafka", prompt_kafka, kafka_ctx))
    if "microservices" in domain_keys and _is_enabled("microservices"):
        domains_jobs.append(("microservices", prompt_microservices, ms_ctx))
    if "hard_resources" in domain_keys and _is_enabled("hard_resources"):
        domains_jobs.append(("hard_resources", prompt_hard_resources, hr_ctx))
    if "lt_framework" in domain_keys and _is_enabled("lt_framework"):
        lt_ctx = domain_data.get("lt_framework", {}).get("ctx", json.dumps({"domain":"lt_framework","sections":[]}, ensure_ascii=False))
        domains_jobs.append(("lt_framework", prompt_lt_framework, lt_ctx))
    results_map: dict[str, tuple[str, object, dict]] = {}
    with ThreadPoolExecutor(max_workers=min(5, len(domains_jobs))) as executor:
        future_to_key = {
            executor.submit(llm_two_pass_self_consistency, p, c, 3, True, k): k
            for (k, p, c) in domains_jobs
        }
        for fut in as_completed(future_to_key):
            key = future_to_key[fut]
            try:
                text, parsed, score = fut.result()
                results_map[key] = (text, parsed, score)
            except Exception as e:
                logger.error(f"LLM {key} analysis failed: {e}")
                results_map[key] = ("{}", None, {})

    def _result_or_blank(key: str):
        if _is_enabled(key):
            return results_map.get(key, ("{}", None, {}))
        return ("", None, {})

    answer_jvm, jvm_parsed, jvm_score = _result_or_blank("jvm")
    answer_database, database_parsed, database_score = _result_or_blank("database")
    answer_kafka, kafka_parsed, kafka_score = _result_or_blank("kafka")
    answer_ms, ms_parsed, ms_score = _result_or_blank("microservices")
    answer_hr, hr_parsed, hr_score = _result_or_blank("hard_resources")
    if "lt_framework" in domain_keys:
        answer_lt, lt_parsed, lt_score = _result_or_blank("lt_framework")
    else:
        answer_lt, lt_parsed, lt_score = ("", None, {})

    merged_prompt_overall = (
        prompt_overall
        .replace("{answer_jvm}", answer_jvm)
        .replace("{answer_database}", answer_database)
        .replace("{answer_kafka}", answer_kafka)
        .replace("{answer_microservices}", answer_ms)
        .replace("{answer_hard_resources}", answer_hr)
        .replace("{answer_lt_framework}", answer_lt)
        .replace("{system_context_brief}", system_context_brief)
    )
    if "{system_context_brief}" not in prompt_overall:
        merged_prompt_overall = _augment_prompt_with_system_context(merged_prompt_overall, system_context_brief)
    merged_prompt_overall = _augment_prompt_with_test_profile(merged_prompt_overall, test_profile)

    perf_query_label = str(sla_early.get("max_performance_query") or "").strip()
    designated_peak: Optional[float] = None
    designated_series: Optional[str] = None
    designated_source_label: Optional[str] = None
    designated_method: Optional[str] = None
    if peak_performance_applicable and perf_query_label and "lt_framework" in domain_keys:
        rps_pick = extract_target_rps_from_pack(
            lt_pack,
            perf_query_label,
            allow_peak_fallback=_cfg_bool(sla_early.get("target_rps_allow_peak_fallback"), default=True),
        )
        if rps_pick.get("value") is not None:
            try:
                designated_peak = float(rps_pick.get("value"))
            except (TypeError, ValueError):
                designated_peak = None
            designated_series = str(rps_pick.get("series") or "") or None
            designated_source_label = str(rps_pick.get("source_label") or perf_query_label or "") or None
            designated_method = str(rps_pick.get("method") or "") or None

    sla_cfg = copy.deepcopy(cfg.get("sla") or CONFIG.get("sla") or {})
    if not peak_performance_applicable:
        stability_required_rps = sla_cfg.get("required_rps", sla_cfg.get("stability_required_rps"))
        if stability_required_rps is not None:
            sla_cfg["target_rps"] = stability_required_rps
        else:
            sla_cfg["target_rps"] = None
    sla_result = evaluate_sla(domain_data, sla_cfg, test_profile=test_profile)
    sla_result = _reconcile_sla_for_test_profile(sla_result, test_profile)
    logger.info(
        "SLA pipeline result: verdict=%s, mode=%s, failed=%s",
        sla_result.get("verdict"),
        sla_result.get("test_mode") or test_profile.get("mode"),
        [
            c.get("name")
            for c in (sla_result.get("checks") or [])
            if isinstance(c, dict) and c.get("passed") is False
        ],
    )
    deterministic_sla = _deterministic_sla_context(sla_result)

    base_ctx = {
        "time_range": {"start": start_ts, "end": end_ts},
        "test_profile": test_profile,
        "designated_peak_performance": {
            "source_domain": "lt_framework",
            "source_label": designated_source_label or perf_query_label or None,
            "stable_max": designated_peak,
            "not_applicable": not peak_performance_applicable,
            "reason": "not_applicable_for_stability_test" if not peak_performance_applicable else None,
            "value_type": (
                "stable_max" if str(designated_method or "").startswith("stable_max")
                else ("max" if str(designated_method or "").startswith("peak_max") else None)
            ),
            "series": designated_series,
            "method": designated_method,
            "note": (
                "Для stability/soak тестов peak_performance не применяется; оценивайте удержание нагрузки."
                if not peak_performance_applicable
                else (
                    "ЕДИНСТВЕННЫЙ источник peak_performance.max_rps. "
                    "НЕ использовать RPS из домена microservices для определения максимальной производительности системы."
                )
            ),
        },
        "deterministic_sla": deterministic_sla,
        "domains": {
            "jvm": jvm_pack,
            "database": database_pack,
            "kafka": kafka_pack,
            "microservices": ms_pack,
            "hard_resources": hr_pack
        }
    }
    if _has_meaningful_system_context(active_system_context):
        base_ctx["system_context"] = active_system_context
    if "lt_framework" in domain_keys:
        base_ctx["domains"]["lt_framework"] = lt_pack
    if include_tables:
        base_ctx["domains_tables_markdown"] = {
            "jvm": jvm_full_data,
            "database": database_full_data,
            "kafka": kafka_full_data,
            "microservices": ms_full_data,
            "hard_resources": hr_full_data,
        }
        if "lt_framework" in domain_keys:
            base_ctx["domains_tables_markdown"]["lt_framework"] = lt_full_data
    overall_ctx = json.dumps(base_ctx, ensure_ascii=False)
    final_answer, final_parsed, final_score = llm_two_pass_self_consistency(
        user_prompt=merged_prompt_overall,
        data_context=overall_ctx,
        k=3,
        return_scores=True,
        domain_key="final",
    )

    def _to_dict_maybe(obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return dict(obj)
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                return None
        return None

    def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    final_payload = _to_dict_maybe(final_parsed) or _extract_json_obj(final_answer)
    if isinstance(final_payload, dict):
        final_payload["test_profile"] = test_profile
        final_payload["deterministic_sla"] = deterministic_sla
        if not peak_performance_applicable:
            target_check = deterministic_sla.get("target_rps_check") if isinstance(deterministic_sla, dict) else None
            final_payload["peak_performance"] = {
                "not_applicable": True,
                "reason": "not_applicable_for_stability_test",
                "method": None,
                "note": "Для теста стабильности максимальная производительность не рассчитывается.",
            }
            final_payload["stability_under_load"] = {
                "mode": "stability",
                "target_rps": (target_check or {}).get("threshold") if isinstance(target_check, dict) else None,
                "actual_rps": (target_check or {}).get("actual") if isinstance(target_check, dict) else None,
                "sla_summary": deterministic_sla.get("summary") if isinstance(deterministic_sla, dict) else None,
                "focus": test_profile.get("focus"),
            }
        elif designated_peak is not None:
            peak_obj = final_payload.get("peak_performance")
            if not isinstance(peak_obj, dict):
                peak_obj = {}
            try:
                peak_obj["max_rps"] = round(float(designated_peak), 2)
            except (TypeError, ValueError):
                peak_obj["max_rps"] = designated_peak
            if designated_method:
                peak_obj["method"] = str(designated_method)
            if designated_series:
                peak_obj["series"] = str(designated_series)
            if designated_source_label:
                peak_obj["source_label"] = str(designated_source_label)
            final_payload["peak_performance"] = peak_obj
        if isinstance(sla_result, dict) and sla_result.get("verdict"):
            sla_verdict = str(sla_result["verdict"])
            original_verdict = str(final_payload.get("verdict") or "")
            final_payload["verdict"] = sla_verdict
            if sla_verdict == "Есть риски" and original_verdict == "Провал":
                rationale = str(final_payload.get("verdict_rationale") or "").strip()
                correction = (
                    "Итоговый вердикт скорректирован по deterministic_sla: для stability/soak теста "
                    "ресурсные превышения CPU/memory считаются рисками, а не самостоятельным провалом, "
                    "если primary SLA по target RPS, error rate и latency не нарушены."
                )
                final_payload["verdict_rationale"] = f"{correction}\n\n{rationale}" if rationale else correction
        final_parsed = final_payload
        try:
            trimmed = (final_answer or "").strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                json.loads(trimmed)
                final_answer = json.dumps(final_payload, ensure_ascii=False)
        except Exception:
            pass

    def _compose_text(full_md: str, header: str, analysis: str) -> str:
        if include_tables:
            return f"{full_md}\n\n{header}\n{analysis}"
        return analysis

    results = {
        "jvm": _compose_text(jvm_full_data, "Анализ JVM:", answer_jvm),
        "database": _compose_text(database_full_data, "Анализ Database:", answer_database),
        "kafka": _compose_text(kafka_full_data, "Анализ Kafka:", answer_kafka),
        "ms": _compose_text(ms_full_data, "Анализ микросервисов:", answer_ms),
        "hard_resources": _compose_text(hr_full_data, "Анализ ресурсов (CPU/MEM/Disk):", answer_hr),
        "lt_framework": answer_lt,
        "final": final_answer,
        "jvm_parsed": _to_dict_maybe(jvm_parsed),
        "database_parsed": _to_dict_maybe(database_parsed),
        "kafka_parsed": _to_dict_maybe(kafka_parsed),
        "ms_parsed": _to_dict_maybe(ms_parsed),
        "hard_resources_parsed": _to_dict_maybe(hr_parsed),
        "lt_framework_parsed": _to_dict_maybe(lt_parsed),
        "final_parsed": _to_dict_maybe(final_parsed),
        "scores": {
            "jvm": jvm_score,
            "database": database_score,
            "kafka": kafka_score,
            "microservices": ms_score,
            "hard_resources": hr_score,
            **({"lt_framework": lt_score} if "lt_framework" in domain_keys else {}),
            "final": final_score,
        },
        "sla_verdict": sla_result.get("verdict"),
        "sla_checks": sla_result.get("checks", []),
        "sla_summary": sla_result.get("summary", ""),
        "system_context": active_system_context,
    }

    # Сохранение LLM результатов в отдельную таблицу (если включено)
    if save_to_db:
        try:
            save_llm_results(
                results=results,
                run_meta={
                    **(run_meta or {}),
                    "start_ms": int((run_meta or {}).get("start_ms") or int(start_ts * 1000)),
                    "end_ms": int((run_meta or {}).get("end_ms") or int(end_ts * 1000)),
                },
                storage_cfg=storage_cfg
            )
        except Exception as e:
            logger.error(f"Failed to save LLM results: {e}")

    return results


def label_dataframes(dfs: List[pd.DataFrame], labels: List[str]) -> List[Dict[str, object]]:
    """Присваивает человекочитаемые подписи каждому DataFrame.

    Параметры:
        dfs (list[pd.DataFrame]): Набор таблиц.
        labels (list[str]): Подписи по порядку.

    Возвращает:
        list[dict]: Структуры вида `{"label": str, "df": DataFrame}`.

    Исключения:
        ValueError: Если количество таблиц и подписей различается.
    """
    if len(dfs) != len(labels):
        raise ValueError("Количество DataFrame и количество меток не совпадает!")
    labeled_list = []
    for df, label in zip(dfs, labels):
        labeled_list.append({
            "label": label,
            "df": df
        })
    return labeled_list


