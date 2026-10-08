import copy
import csv
import os
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, List, Dict, Optional
from urllib.parse import quote, urlsplit

import requests
import io
import pandas as pd

from settings import CONFIG
from AI.context_pack import (
    LoadStep,
    build_timeline,
    derive_load_steps,
    detect_step_anomalies,
    first_rps_drop,
    format_iso,
    is_load_step_segment,
    to_utc,
    load_step_report_from_frames,
    minutes_from,
    step_table,
)
from AI.prompt_assembly import assemble_prompt, build_analysis_guidance
from AI.providers import attach_usage_to_scores, reset_usage, usage_domain
from AI.scoring import LLMAnalysis, VerificationSummary, fill_missing_verdict_rationale, llm_two_pass_self_consistency, make_failed_llm_analysis, parse_llm_analysis_strict
from AI.verification import unverified_findings, verify_analysis
from AI.db_store import (
    BASELINE_MODE_PREVIOUS_SUCCESS,
    find_previous_run,
    load_run_metric_stats,
    save_domain_labeled,
    save_llm_results,
    _connect as _db_connect,
)
from AI.data_sources import ResolvedSource, domain_query_language, resolve_domain_source
from AI.opensearch_logs import APPLICATION_LOGS_DOMAIN, collect_application_logs
from AI.sla_evaluator import _find_section_by_label, evaluate_sla, extract_target_rps_from_pack


logger = logging.getLogger(__name__)

# Grafana answers these when /api/datasources/proxy is closed; /api/ds/query still works.
_PROXY_CLOSED = (403, 404, 405)

ProgressCallback = Callable[[str, Optional[int]], None]

# Human-readable domain names for progress messages shown to the user.
DOMAIN_TITLES: Dict[str, str] = {
    "jvm": "JVM",
    "database": "База данных",
    "kafka": "Kafka",
    "microservices": "Микросервисы",
    "hard_resources": "Ресурсы узлов",
    "lt_framework": "Нагрузочный инструмент",
    APPLICATION_LOGS_DOMAIN: "Логи приложений",
    "final": "Итог",
}

# Progress budget (percent) for each pipeline phase; update_page.py owns 0-5 and 98-100.
PROGRESS_COLLECT_START = 10
PROGRESS_COLLECT_END = 40
PROGRESS_SAVE_METRICS = 45
PROGRESS_LLM_START = 47
PROGRESS_LLM_END = 80
PROGRESS_SLA = 42
PROGRESS_FINAL_LLM = 83
PROGRESS_SAVE_RESULTS = 95


def _domain_title(key: str) -> str:
    return DOMAIN_TITLES.get(key, key)


def _phase_percent(start: int, end: int, done: int, total: int) -> int:
    if total <= 0:
        return end
    return start + int((end - start) * min(max(done, 0), total) / total)


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
        "stable_window": result.get("stable_window"),
        "target_rps_check": target_rps_check,
        "passed_checks": [str(item.get("name")) for item in checks if item.get("passed") is True],
        "failed_checks": [str(item.get("name")) for item in checks if item.get("passed") is False],
        "unknown_checks": [str(item.get("name")) for item in checks if item.get("passed") is None],
    }


def _sla_step_context(
    sla_result: Dict[str, Any] | None,
    load_steps: List[LoadStep],
    shift_hours: int,
) -> Optional[Dict[str, Any]]:
    """Step the SLA was checked on; shared by domain contexts and the domain verdict guard."""
    result = sla_result if isinstance(sla_result, dict) else {}
    window = result.get("stable_window")
    if not isinstance(window, dict) or not window.get("start") or not window.get("end"):
        return None
    start = to_utc(window["start"])
    end = to_utc(window["end"])
    step = next((item for item in reversed(load_steps) if to_utc(item.start) <= start <= to_utc(item.end)), None)
    checks = [
        {key: item.get(key) for key in ("name", "category", "threshold", "actual", "passed", "message")}
        for item in (result.get("checks") or [])
        if isinstance(item, dict)
    ]
    target = next((item for item in checks if item.get("name") == "target_rps"), None)
    primary_failed = any(
        item.get("passed") is False and str(item.get("category") or "primary") == "primary" for item in checks
    )
    return {
        "step_index": step.index if step else None,
        "step_label": step.label if step else None,
        "start_iso": format_iso(start, shift_hours),
        "end_iso": format_iso(end, shift_hours),
        "rps": window.get("level"),
        "verdict": result.get("verdict"),
        "primary_passed": (target is None or target.get("passed") is True) and not primary_failed,
        "checks": checks,
        "degraded_level": window.get("degraded_level"),
        "degraded_checks": list(window.get("degraded_checks") or []),
    }


_SEVERE_FINDINGS = frozenset({"critical", "high"})
_AFTER_STEP_TOLERANCE = pd.Timedelta(seconds=60)


def _align_domain_verdict_with_sla(
    text: str,
    parsed: Optional[Dict[str, Any]],
    sla_step: Optional[Dict[str, Any]],
) -> Optional[tuple[str, Dict[str, Any]]]:
    """Turns a domain «Провал» into «Есть риски» when every severe finding starts after the SLA step.

    Returns None when the verdict stays unchanged.
    """
    if not sla_step or sla_step.get("primary_passed") is not True or not sla_step.get("end_iso"):
        return None
    if not isinstance(parsed, dict) or parsed.get("verdict") != "Провал":
        return None
    raw = str(text or "").strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return None
    cutoff = to_utc(sla_step["end_iso"]) - _AFTER_STEP_TOLERANCE
    for finding in parsed.get("findings") or []:
        if not isinstance(finding, dict) or str(finding.get("severity") or "").lower() not in _SEVERE_FINDINGS:
            continue
        start_raw = str(finding.get("start_time") or "").strip()
        if not start_raw:
            return None
        try:
            started = to_utc(start_raw)
        except (TypeError, ValueError):
            return None
        if pd.isna(started) or started < cutoff:
            return None
    note = (
        f"Статус скорректирован по ступени SLA: на ступени «{sla_step.get('step_label') or 'SLA'}» пороги SLA "
        "выполнены, серьёзные проблемы домена начались после неё. Исходная оценка модели — «Провал»."
    )
    rationale = str(parsed.get("verdict_rationale") or "").strip().replace("Статус «Провал»", "Исходный статус «Провал»", 1)
    aligned = {**parsed, "verdict": "Есть риски", "verdict_rationale": f"{note}\n\n{rationale}" if rationale else note}
    payload = json.loads(raw)
    payload["verdict"] = aligned["verdict"]
    payload["verdict_rationale"] = aligned["verdict_rationale"]
    return json.dumps(payload, ensure_ascii=False), aligned


def _load_step_table_for_report(
    steps: List[LoadStep],
    labeled: List[Dict[str, Any]],
    sla_cfg: Dict[str, Any],
    window: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compact step table stored with the report: step RPS and plateau latency p95."""
    if not steps:
        return None
    cfg = sla_cfg if isinstance(sla_cfg, dict) else {}
    stable = window if isinstance(window, dict) else {}
    latency_query = str(cfg.get("p95_query") or "").strip() or str(cfg.get("p99_query") or "").strip()
    report = load_step_report_from_frames(
        steps,
        labeled,
        rps_query=str(cfg.get("max_performance_query") or ""),
        latency_query=latency_query,
        selected_level=stable.get("level"),
        selected_start=str(stable["start"]) if stable.get("start") else None,
    )
    return report if report.get("steps") else None


def _pin_series_to_sla_window(pack: Dict[str, Any], window: Dict[str, Any]) -> None:
    """Makes the designated series quote the SLA step, not a shorter detector plateau."""
    if not isinstance(pack, dict):
        return
    label = str(window.get("label") or "")
    series_name = str(window.get("series") or "")
    start = window.get("start")
    end = window.get("end")
    duration = None
    try:
        duration = round((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60.0, 1)
    except (TypeError, ValueError):
        duration = None
    for section in pack.get("sections") or []:
        if not isinstance(section, dict):
            continue
        if label and str(section.get("label") or "") != label:
            continue
        for series in section.get("top_series") or []:
            if not isinstance(series, dict):
                continue
            if series_name and str(series.get("series") or "") != series_name:
                continue
            series["stable_max"] = window.get("level")
            series["stable_window_start"] = start
            series["stable_window_end"] = end
            series["stable_start_time"] = start
            series["stable_max_time"] = end
            if duration is not None:
                series["stable_duration_min"] = duration
            series.pop("step_segments", None)
            return


def _designated_window_ctx(window: Optional[Dict[str, Any]], shift_hours: int) -> Optional[Dict[str, Any]]:
    """Stable step behind the designated peak, in report time, with the step above it that broke SLA."""
    if not window:
        return None
    return {
        "start_iso": format_iso(pd.Timestamp(window["start"]), shift_hours),
        "end_iso": format_iso(pd.Timestamp(window["end"]), shift_hours),
        "degraded_level": window.get("degraded_level"),
        "degraded_checks": window.get("degraded_checks") or [],
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


@dataclass(frozen=True)
class GrafanaDatasource:
    """Grafana datasource of a binding. Either field may be missing: old Grafana has no uid
    routes, newer Grafana drops the deprecated numeric-id endpoints."""

    id: Optional[int]
    uid: str


class GrafanaFallbackError(RuntimeError):
    """The datasource proxy was closed and the ``/api/ds/query`` fallback failed as well."""


def _datasource_cfg(g_cfg: dict, influx: bool) -> dict:
    if influx:
        return g_cfg.get("influxdb_datasource") or g_cfg.get("prometheus_datasource") or {}
    return g_cfg.get("prometheus_datasource") or {}


def _lookup_grafana_datasource(g_cfg: dict, influx: bool) -> GrafanaDatasource:
    """Finds the binding's datasource by uid, then by name, then by a legacy numeric id.

    The lookup answer carries both id and uid, so later requests can use uid routes even
    when the binding (for example one migrated from the old config) only names the datasource.
    Without uid, name and id the first datasource of the expected type is taken.
    """
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    ds_cfg = _datasource_cfg(g_cfg, influx)
    uid = str(ds_cfg.get("uid") or "").strip()
    name = str(ds_cfg.get("name") or "").strip()
    legacy_id = ds_cfg.get("id")
    if uid or name:
        path = f"uid/{quote(uid, safe='')}" if uid else f"name/{quote(name, safe='')}"
        resp = requests.get(f"{base_url}/api/datasources/{path}", headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return _grafana_datasource_from(resp.json(), uid)
    if isinstance(legacy_id, int) and not isinstance(legacy_id, bool):
        return GrafanaDatasource(id=legacy_id, uid="")
    resp = requests.get(f"{base_url}/api/datasources", headers=headers, auth=auth, timeout=30, verify=verify)
    resp.raise_for_status()
    wanted = ("influxdb", "influxdb2") if influx else ("prometheus",)
    for item in resp.json():
        if isinstance(item, dict) and item.get("type") in wanted:
            return _grafana_datasource_from(item, "")
    raise RuntimeError("Не найден InfluxDB datasource в Grafana" if influx else "Не найден Prometheus datasource в Grafana")


def _grafana_datasource_from(payload: Any, configured_uid: str) -> GrafanaDatasource:
    data = payload if isinstance(payload, dict) else {}
    raw_id = data.get("id")
    ds_id = raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else None
    uid = str(data.get("uid") or configured_uid or "").strip()
    if ds_id is None and not uid:
        raise RuntimeError("Grafana не вернула ни id, ни uid датасорса")
    return GrafanaDatasource(id=ds_id, uid=uid)


def _resolve_grafana_prom_ds_id(g_cfg: dict) -> Optional[int]:
    return _lookup_grafana_datasource(g_cfg, influx=False).id


def _resolve_grafana_influx_ds_id(g_cfg: dict) -> Optional[int]:
    return _lookup_grafana_datasource(g_cfg, influx=True).id


def _grafana_proxy_send(g_cfg: dict, ds: GrafanaDatasource, path: str, send: Callable[[str], Any]) -> Any:
    """Calls the datasource proxy: the uid route first, the deprecated id route for old Grafana.

    Returns the first response that is not a closed route, otherwise the last one.
    """
    base_url = str(g_cfg.get("base_url") or "").rstrip("/")
    prefixes = []
    if ds.uid:
        prefixes.append(f"{base_url}/api/datasources/proxy/uid/{quote(ds.uid, safe='')}")
    if ds.id is not None:
        prefixes.append(f"{base_url}/api/datasources/proxy/{ds.id}")
    resp = None
    for prefix in prefixes:
        resp = send(prefix + path)
        if resp.status_code not in _PROXY_CLOSED:
            return resp
    return resp


def _http_error_text(exc: Exception) -> str:
    """``404 GET /api/x: body`` for an HTTP error, the message for anything else."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        return str(exc).strip() or exc.__class__.__name__
    request = getattr(response, "request", None)
    method = str(getattr(request, "method", "") or "")
    path = urlsplit(str(getattr(response, "url", "") or "")).path
    where = " ".join(part for part in (method, path) if part)
    body = str(getattr(response, "text", "") or "").strip().replace("\n", " ")[:300]
    text = f"{status} {where}".strip()
    return f"{text}: {body}" if body else text


def _via_ds_query(kind: str, proxy_status: int, run: Callable[[], Any]) -> Any:
    """Runs the ``/api/ds/query`` fallback; its error also names the closed proxy, so both attempts are visible."""
    logger.warning("Grafana %s proxy HTTP %s, falling back to /api/ds/query", kind, proxy_status)
    try:
        return run()
    except Exception as exc:
        raise GrafanaFallbackError(
            f"Grafana: прокси датасорса ответил {proxy_status}, запасной запрос через /api/ds/query не прошёл: {_http_error_text(exc)}"
        ) from exc


def fetch_influx_data_via_grafana(
    g_cfg: dict,
    flux_query: str,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> str:
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
    _, base_headers, auth, verify = _grafana_influx_auth(g_cfg)
    headers = {**base_headers, "Accept": "application/csv", "Content-Type": "application/json"}
    ds = _lookup_grafana_datasource(g_cfg, influx=True)
    resp = _grafana_proxy_send(
        g_cfg, ds, "/api/v2/query",
        lambda url: requests.post(url, headers=headers, auth=auth, json={"query": flux_query}, timeout=60, verify=verify),
    )
    if resp.status_code in _PROXY_CLOSED and start_ts is not None and end_ts is not None:
        return _via_ds_query(
            "Flux", resp.status_code, lambda: _flux_csv_via_ds_query(g_cfg, ds, flux_query, float(start_ts), float(end_ts)),
        )
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
            csv_text = fetch_influx_data_via_grafana(grafana_cfg, q, start_ts=start_ts, end_ts=end_ts)
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
                logger.warning("Resample %r failed; storing the query step unchanged", resample_interval)
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


def _grafana_influx_auth(g_cfg: dict) -> tuple[str, dict, Any, bool]:
    base_url = str(g_cfg.get("base_url") or "").rstrip("/")
    auth_cfg = g_cfg.get("auth") or {}
    headers: Dict[str, str] = {}
    auth = None
    method = str(auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    return base_url, headers, auth, bool(g_cfg.get("verify_ssl", True))


def _grafana_ds_query_influxql(g_cfg: dict, ds: GrafanaDatasource, influxql: str, start_ts: float, end_ts: float) -> dict:
    """InfluxQL through Grafana ``/api/ds/query`` when the legacy datasource proxy is closed."""
    uid = _grafana_uid(g_cfg, ds)
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    url = f"{base_url}/api/ds/query"
    merged = dict(headers)
    merged["Content-Type"] = "application/json"
    merged["Accept"] = "application/json"
    resp = requests.post(
        url,
        json={
            "from": str(int(float(start_ts) * 1000)),
            "to": str(int(float(end_ts) * 1000)),
            "queries": [{
                "refId": "A",
                "datasource": {"type": "influxdb", "uid": uid},
                "query": influxql,
                "rawQuery": True,
                "queryType": "InfluxQL",
                "resultFormat": "time_series",
            }],
        },
        headers=merged,
        auth=auth,
        timeout=60,
        verify=verify,
    )
    resp.raise_for_status()
    body = resp.json()
    frames = (((body.get("results") or {}).get("A") or {}).get("frames") or []) if isinstance(body, dict) else []
    series: List[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        fields = ((frame.get("schema") or {}).get("fields") or [])
        data_values = ((frame.get("data") or {}).get("values") or [])
        columns: List[str] = []
        col_data: List[list] = []
        tags: Dict[str, Any] = {}
        for idx, field in enumerate(fields):
            if not isinstance(field, dict):
                continue
            labels = field.get("labels") if isinstance(field.get("labels"), dict) else {}
            tags.update(labels)
            name = str(field.get("name") or f"col{idx}")
            ftype = str(field.get("type") or "").lower()
            column = "time" if ftype == "time" or name.lower() in ("time", "timestamp") else name
            columns.append(column)
            col_data.append(list(data_values[idx]) if idx < len(data_values) else [])
        if not columns or not col_data:
            continue
        width = max((len(col) for col in col_data), default=0)
        series.append({
            "name": ((frame.get("schema") or {}).get("name") or "series"),
            "columns": columns,
            "values": [[col[row] if row < len(col) else None for col in col_data] for row in range(width)],
            "tags": tags,
        })
    return {"results": [{"series": series}]}


def fetch_influxql_via_grafana(
    g_cfg: dict,
    q: str,
    database: str | None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> dict:
    """Выполняет InfluxQL-запрос через Grafana proxy и возвращает JSON.

    Если старый ``/api/datasources/proxy/.../query`` закрыт, повторяет запрос через ``/api/ds/query``.
    """
    _, base_headers, auth, verify = _grafana_influx_auth(g_cfg)
    headers = {**base_headers, "Accept": "application/json"}
    ds = _lookup_grafana_datasource(g_cfg, influx=True)
    params = {"q": q}
    if database:
        params["db"] = database
    resp = _grafana_proxy_send(
        g_cfg, ds, "/query",
        lambda url: requests.get(url, headers=headers, auth=auth, params=params, timeout=60, verify=verify),
    )
    if resp.status_code in _PROXY_CLOSED and start_ts is not None and end_ts is not None:
        return _via_ds_query(
            "InfluxQL", resp.status_code, lambda: _grafana_ds_query_influxql(g_cfg, ds, q, float(start_ts), float(end_ts)),
        )
    resp.raise_for_status()
    return resp.json()


def _render_influxql_template(query: str, start_ts: float, end_ts: float, interval: str) -> str:
    """Подставляет поддерживаемые Grafana-макросы перед прямым запросом к InfluxDB."""
    t_start_ns = int(start_ts * 1_000_000_000)
    t_end_ns = int(end_ts * 1_000_000_000)
    rendered = (query or "")
    time_filter = f"time >= {t_start_ns} AND time <= {t_end_ns}"
    rendered = rendered.replace("$__timeFilter", time_filter)
    rendered = rendered.replace("$timeFilter", time_filter)
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
            data = fetch_influxql_via_grafana(grafana_cfg, q, database, start_ts=start_ts, end_ts=end_ts)
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
                logger.warning("Resample %r failed; storing the query step unchanged", resample_interval)
            dfs.append(merged)
        except Exception as exc:
            logger.warning("InfluxQL[%s] failed: %s; query=%s", idx, exc, _safe_query_excerpt(q if 'q' in locals() else raw))
            dfs.append(pd.DataFrame())
    return dfs

def _grafana_uid(g_cfg: dict, ds: GrafanaDatasource) -> str:
    """Datasource UID for ``/api/ds/query``; only a legacy id-only binding needs the numeric-id endpoint."""
    if ds.uid:
        return ds.uid
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    resp = requests.get(
        f"{base_url}/api/datasources/{ds.id}", headers=headers, auth=auth, timeout=30, verify=verify,
    )
    resp.raise_for_status()
    uid = str((resp.json() or {}).get("uid") or "").strip()
    if not uid:
        raise RuntimeError(f"У Grafana datasource id={ds.id} нет uid")
    return uid


def _post_ds_query(g_cfg: dict, payload: dict) -> dict:
    """POST /api/ds/query. Raises when Grafana reports an error for refId A."""
    base_url, headers, auth, verify = _grafana_influx_auth(g_cfg)
    merged = dict(headers)
    merged["Content-Type"] = "application/json"
    merged["Accept"] = "application/json"
    resp = requests.post(
        f"{base_url}/api/ds/query", json=payload, headers=merged, auth=auth, timeout=60, verify=verify,
    )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise RuntimeError("Grafana /api/ds/query вернул не объект")
    result = (body.get("results") or {}).get("A")
    error = result.get("error") if isinstance(result, dict) else None
    if error:
        raise RuntimeError(f"Grafana /api/ds/query: {error}")
    return body


def _frame_columns(frame: dict) -> tuple[list[dict], list[list]]:
    fields = ((frame.get("schema") or {}).get("fields") or [])
    values = ((frame.get("data") or {}).get("values") or [])
    columns = [field for field in fields if isinstance(field, dict)]
    series = [list(values[index]) if index < len(values) else [] for index in range(len(columns))]
    return columns, series


def _is_time_field(field: dict) -> bool:
    name = str(field.get("name") or "").lower()
    return str(field.get("type") or "").lower() == "time" or name in ("time", "timestamp", "_time")


def _epoch_seconds(raw: Any) -> Optional[float]:
    """Grafana frame timestamps are ms, µs or ns; Prometheus wants seconds."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            raw = float(text)
        except ValueError:
            return None
    if not isinstance(raw, (int, float)):
        return None
    number = float(raw)
    if number > 1e17:
        return number / 1e9
    if number > 1e14:
        return number / 1e6
    if number > 1e11:
        return number / 1e3
    return number


def _result_frames(body: dict) -> list[dict]:
    frames = (((body.get("results") or {}).get("A") or {}).get("frames") or [])
    return [frame for frame in frames if isinstance(frame, dict)]


def _prometheus_matrix(frames: list[dict]) -> dict:
    """Grafana time-series frames as a Prometheus range-query payload."""
    result: list[dict] = []
    for frame in frames:
        columns, series = _frame_columns(frame)
        time_index = next((index for index, field in enumerate(columns) if _is_time_field(field)), None)
        if time_index is None:
            continue
        times = series[time_index]
        for index, field in enumerate(columns):
            if index == time_index or str(field.get("type") or "").lower() not in ("number", "numeric", ""):
                continue
            if str(field.get("type") or "").lower() == "" and not _numeric_column(series[index]):
                continue
            labels = field.get("labels") if isinstance(field.get("labels"), dict) else {}
            points = _prometheus_points(times, series[index])
            if points:
                result.append({"metric": {str(key): str(value) for key, value in labels.items()}, "values": points})
    return {"status": "success", "data": {"resultType": "matrix", "result": result}}


def _numeric_column(values: list) -> bool:
    return any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)


def _prometheus_points(times: list, values: list) -> list[list]:
    points: list[list] = []
    for raw_time, raw_value in zip(times, values):
        if raw_value is None or isinstance(raw_value, bool):
            continue
        seconds = _epoch_seconds(raw_time)
        if seconds is None:
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            continue
        points.append([seconds, str(number)])
    return points


def _prometheus_via_ds_query(
    g_cfg: dict, ds: GrafanaDatasource, promql: str, start_ts: float, end_ts: float, step_seconds: int,
) -> dict:
    step_seconds = max(int(step_seconds), 1)
    points = max(1, int((float(end_ts) - float(start_ts)) / step_seconds) + 1)
    body = _post_ds_query(g_cfg, {
        "from": str(int(float(start_ts) * 1000)),
        "to": str(int(float(end_ts) * 1000)),
        "queries": [{
            "refId": "A",
            "datasource": {"type": "prometheus", "uid": _grafana_uid(g_cfg, ds)},
            "expr": promql,
            "instant": False,
            "range": True,
            "intervalMs": step_seconds * 1000,
            "maxDataPoints": points,
            "format": "time_series",
        }],
    })
    return _prometheus_matrix(_result_frames(body))


def _flux_name(field: dict) -> str:
    if _is_time_field(field):
        return "_time"
    name = str(field.get("name") or "")
    if name.lower() == "value" or (not name and str(field.get("type") or "").lower() in ("number", "numeric")):
        return "_value"
    return name or "column"


def _flux_rows(frame: dict) -> list[dict[str, Any]]:
    columns, series = _frame_columns(frame)
    if not columns:
        return []
    names = [_flux_name(field) for field in columns]
    labels: dict[str, Any] = {}
    for field in columns:
        raw = field.get("labels")
        if isinstance(raw, dict):
            labels.update(raw)
    width = max((len(column) for column in series), default=0)
    rows: list[dict[str, Any]] = []
    for index in range(width):
        row = dict(labels)
        for name, column in zip(names, series):
            value = column[index] if index < len(column) else None
            if name == "_time":
                seconds = _epoch_seconds(value)
                row[name] = _iso8601_utc(seconds) if seconds is not None else ""
            else:
                row[name] = "" if value is None else value
        rows.append(row)
    return rows


def _flux_csv_from_frames(frames: list[dict]) -> str:
    rows = [row for frame in frames for row in _flux_rows(frame)]
    if not rows:
        return "_time,_value\n"
    present: list[str] = []
    for row in rows:
        for key in row:
            if key not in present:
                present.append(key)
    headers = [key for key in ("_time", "_value") if key in present]
    headers.extend(key for key in present if key not in headers)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _flux_csv_via_ds_query(g_cfg: dict, ds: GrafanaDatasource, flux_query: str, start_ts: float, end_ts: float) -> str:
    body = _post_ds_query(g_cfg, {
        "from": str(int(float(start_ts) * 1000)),
        "to": str(int(float(end_ts) * 1000)),
        "queries": [{
            "refId": "A",
            "datasource": {"type": "influxdb", "uid": _grafana_uid(g_cfg, ds)},
            "query": flux_query,
            "rawQuery": True,
            "queryType": "flux",
            "resultFormat": "time_series",
        }],
    })
    return _flux_csv_from_frames(_result_frames(body))


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
    _, headers, auth, verify = _grafana_influx_auth(g_cfg)
    ds = _lookup_grafana_datasource(g_cfg, influx=False)
    params = {
        'query': promql_query,
        'start': start_ts,
        'end':   end_ts,
        'step':  step_in_seconds
    }
    resp = _grafana_proxy_send(
        g_cfg, ds, "/api/v1/query_range",
        lambda url: requests.get(url, headers=headers, auth=auth, params=params, timeout=30, verify=verify),
    )
    if resp.status_code in _PROXY_CLOSED:
        return _via_ds_query(
            "Prometheus", resp.status_code,
            lambda: _prometheus_via_ds_query(g_cfg, ds, promql_query, start_ts, end_ts, step_in_seconds),
        )
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
        try:
            data_json = fetch_metric_series(prometheus_url, start_ts, end_ts, query, step, ef_config=ef_config)
        except Exception as exc:
            logger.error("PromQL query failed and was skipped: %s", exc)
            dfs.append(pd.DataFrame(columns=["ts", "series", "value"]))
            continue
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
                logger.warning("Resample %r failed; storing the query step unchanged", resample_interval)
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
                logger.warning("Resample %r failed; storing the query step unchanged", resample_interval)
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


def _domain_worker_count(llm_cfg: Dict[str, Any]) -> int:
    """Parallel domain analyses, capped by the active provider's ``max_concurrent``.

    When the provider is not set, the GigaChat limit is used so a mono-provider
    config still stays inside the client semaphore.
    """
    requested = _cfg_int((llm_cfg or {}).get("max_domain_workers"), 4)
    provider = str((llm_cfg or {}).get("provider") or "").strip().lower()
    provider_cfg = (llm_cfg or {}).get(provider) if provider else None
    if not isinstance(provider_cfg, dict):
        provider_cfg = (llm_cfg or {}).get("gigachat") if isinstance((llm_cfg or {}).get("gigachat"), dict) else {}
    concurrent = _cfg_int((provider_cfg or {}).get("max_concurrent"), 4)
    return max(1, min(requested, concurrent))


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


def _persistent_drop_start(values: List[bool], run_len: int) -> Optional[int]:
    """Start of the first run of ``run_len`` low samples that is not followed by ``run_len`` recovered ones.

    A low run that RPS climbs back from inside the same segment is a transient dip, not a drop.
    """
    need = max(1, int(run_len))
    start: Optional[int] = None
    idx = 0
    while idx < len(values):
        end = idx
        while end < len(values) and values[end] == values[idx]:
            end += 1
        if values[idx] and end - idx >= need and start is None:
            start = idx
        elif not values[idx] and start is not None and end - idx >= need:
            start = None
        idx = end
    return start


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


def _step_delta_threshold(level: float, min_delta_rps: float, min_delta_pct: float) -> float:
    """Smallest RPS change that counts as a new load level near ``level``."""
    return max(float(min_delta_rps), abs(float(level)) * float(min_delta_pct))


def _plateau_core(seg: pd.Series, level: float, tolerance: float) -> pd.Series:
    """Segment without the leading/trailing samples farther than ``tolerance`` from ``level``.

    At coarse cadence the first samples of a plateau are still the ramp into it and would fail
    the slope check of an otherwise flat step. The whole segment is returned when trimming
    would leave less than half of it: then it is not a plateau with ramp edges.
    """
    values = seg.values
    lo, hi = 0, len(values)
    while lo < hi and abs(float(values[lo]) - level) > tolerance:
        lo += 1
    while hi > lo and abs(float(values[hi - 1]) - level) > tolerance:
        hi -= 1
    if hi - lo < max(2, (len(values) + 1) // 2):
        return seg
    return seg.iloc[lo:hi]


def _scan_step_boundaries(
    smooth: pd.Series,
    hold_samples: int,
    min_delta_rps: float,
    min_delta_pct: float,
) -> tuple[List[int], Dict[int, float]]:
    """Positions where the load level changes (with 0 and the length) and the level before every step down.

    A change is confirmed only when the level does not come back in the window after the next
    one, so a spike or dip of a few samples (GC pause, metrics gap) is not a load step. The
    level before a candidate is the median of up to two windows, so an excursion right before
    it does not shift the reference either.
    """
    values = smooth.values
    n = len(values)
    min_prev = max(2, hold_samples // 2)
    boundaries = [0]
    down_refs: Dict[int, float] = {}
    i = hold_samples
    while i < (n - hold_samples):
        prev_vals = values[max(boundaries[-1], i - 2 * hold_samples):i]
        if len(prev_vals) < min_prev:
            i += 1
            continue
        prev_level = float(pd.Series(prev_vals).median())
        next_level = float(pd.Series(values[i:i + hold_samples]).median())
        threshold = _step_delta_threshold(prev_level, min_delta_rps, min_delta_pct)
        if abs(next_level - prev_level) < threshold:
            i += 1
            continue
        after_vals = values[i + hold_samples:i + 2 * hold_samples]
        if len(after_vals) >= min_prev and abs(float(pd.Series(after_vals).median()) - prev_level) < threshold:
            i += 1
            continue
        boundaries.append(i)
        if next_level < prev_level:
            down_refs[i] = prev_level
        i += hold_samples
    boundaries.append(n)
    return boundaries, down_refs


def _mark_decline(members: List[Dict[str, object]], opener: Optional[Dict[str, object]], recovered: bool) -> None:
    """Flags a closed decline: ``dip`` when RPS came back to the level before it, ``after_drop`` otherwise.

    The drops of a recovered decline become ``dip_time``: RPS fell there but was not lost.
    """
    for seg in members:
        seg["dip" if recovered else "after_drop"] = True
    if not recovered:
        return
    for seg in members + ([opener] if opener is not None else []):
        if seg.get("drop_time") is not None:
            seg["dip_time"] = seg["drop_time"]
            seg["drop_time"] = None


def _classify_declines(segments: List[Dict[str, object]], min_delta_rps: float, min_delta_pct: float) -> None:
    """Sets ``after_drop`` / ``dip`` from the outcome of each RPS decline instead of a flag that never resets.

    A decline starts at a step down (reference: the level before it) or at a persistent drop
    inside a segment (reference: the segment's starting level). It is recovered when a later
    segment climbs back within the detector's step threshold of the reference. Segments of a
    recovered decline are ``dip``; segments of a decline RPS never recovered from are
    ``after_drop``. The segment where an in-segment drop starts keeps its plateau before the
    drop, so it is not part of the decline.
    """
    members: List[Dict[str, object]] = []
    opener: Optional[Dict[str, object]] = None
    reference: Optional[float] = None
    for seg in segments:
        down_ref = seg.get("_down_ref")
        if down_ref is not None:
            reference = float(down_ref) if reference is None else max(reference, float(down_ref))
            members.append(seg)
        elif reference is not None:
            if float(seg.get("level", 0.0)) >= reference - _step_delta_threshold(reference, min_delta_rps, min_delta_pct):
                _mark_decline(members, opener, recovered=True)
                members, opener, reference = [], None, None
            else:
                members.append(seg)
        if seg.get("drop_time") is not None:
            level_ref = float(seg.get("_level_ref", seg.get("level", 0.0)))
            if reference is None:
                reference, opener = level_ref, seg
            else:
                reference = max(reference, level_ref)
    if reference is not None:
        _mark_decline(members, opener, recovered=False)


def _select_step_profile_candidate(segments: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Last stable step before the load first dropped.

    A flat plateau after RPS fell (timeouts, ramp-down) is degradation, not a
    load step, so segments flagged ``after_drop`` are never the stable maximum.
    A recovered dip is not a load level either.
    """
    rising = [seg for seg in segments if not bool(seg.get("after_drop")) and not bool(seg.get("dip"))]
    stable_segments = [seg for seg in rising if bool(seg.get("stable"))]
    if not stable_segments:
        return None
    terminal_stable_segments: List[Dict[str, object]] = []
    for idx, seg in enumerate(rising):
        if not bool(seg.get("stable")):
            continue
        next_seg = rising[idx + 1] if idx + 1 < len(rising) else None
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
    """Load steps of an RPS series and the stable maximum among them.

    Each exported segment has ``stable`` (plateau held for ``min_stable_minutes`` with low cv
    and slope), ``drop_time`` (RPS fell inside it and did not recover), ``after_drop`` (it
    follows a decline RPS never recovered from) and ``dip`` / ``dip_time`` (a decline RPS
    recovered from: a transient dip, not a drop).
    """
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

    boundaries, down_refs = _scan_step_boundaries(smooth, hold_samples, min_step_delta_rps, min_step_delta_pct)
    drop_run_samples = max(1, int(round(drop_hold_sec / sample_sec)))

    segments: List[Dict[str, object]] = []
    # A step down whose segment is skipped below still opens the decline of the next kept segment.
    pending_down_ref: Optional[float] = None
    for bi in range(len(boundaries) - 1):
        a = boundaries[bi]
        b = boundaries[bi + 1]
        if a in down_refs:
            pending_down_ref = down_refs[a] if pending_down_ref is None else max(pending_down_ref, down_refs[a])
        if b - a < 2:
            continue
        seg = smooth.iloc[a:b]
        if seg.empty:
            continue
        base_window = seg.iloc[:max(2, min(len(seg), hold_samples))]
        level_ref = float(base_window.median()) if not base_window.empty else float(seg.median())
        drop_threshold = level_ref * (1.0 - max_drop_pct)
        below = [bool(v < drop_threshold) for v in seg.values]
        longest_bad_min = _longest_true_run(below) * (sample_sec / 60.0)
        has_drop = _first_true_run_start(below, drop_run_samples) is not None
        drop_start_idx = _persistent_drop_start(below, drop_run_samples)

        eval_seg = seg
        drop_time = None
        if drop_start_idx is not None and drop_start_idx >= 2:
            eval_seg = seg.iloc[:drop_start_idx]
            drop_time = seg.index[drop_start_idx]
        if eval_seg is None or eval_seg.empty or len(eval_seg) < 2:
            continue

        eval_st = eval_seg.index[0]
        eval_en = eval_seg.index[-1]
        eval_duration_min = ((eval_en - eval_st).total_seconds() + sample_sec) / 60.0
        eval_level = float(eval_seg.median())
        core = _plateau_core(
            eval_seg, eval_level, 0.5 * _step_delta_threshold(eval_level, min_step_delta_rps, min_step_delta_pct)
        )
        core_mean = float(core.mean())
        eval_cv = float(float(core.std(ddof=0)) / max(abs(core_mean), 1e-9))
        eval_slope = _slope_rps_per_min(core)
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
            "dip_time": None,
            "had_drop": bool(has_drop),
            "after_drop": False,
            "dip": False,
            "stable": bool(is_stable),
            "_down_ref": pending_down_ref,
            "_level_ref": level_ref,
        })
        pending_down_ref = None

    _classify_declines(segments, min_step_delta_rps, min_step_delta_pct)
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
                "dip_time": seg.get("dip_time"),
                "after_drop": bool(seg.get("after_drop", False)),
                "dip": bool(seg.get("dip", False)),
                "stable": bool(seg.get("stable", False)),
            }
            for seg in segments
        ],
    }


def _edge_change_pct(col_series: pd.Series) -> Optional[float]:
    """Relative change between the first and the last tenth of the samples."""
    values = col_series.dropna()
    if values.shape[0] < 4:
        return None
    edge = max(1, values.shape[0] // 10)
    first = float(values.iloc[:edge].mean())
    last = float(values.iloc[-edge:].mean())
    if abs(first) < 1e-9:
        return None
    return round((last / first - 1.0) * 100.0, 1)


def _summarize_time_series_dataframe(
    df: pd.DataFrame,
    top_n: int = 10,
    min_stable_minutes: float = 5.0,
    stable_detection_cfg: Optional[Dict[str, Any]] = None,
    section_label: Optional[str] = None,
    start_ts: Optional[float] = None,
    include_stable_max: bool = True,
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    if df is None or getattr(df, 'empty', True):
        return summary
    if not isinstance(df.columns, pd.Index):
        return summary
    ranked_cols: List[tuple[int, str, float, float]] = []
    for idx, col in enumerate(list(df.columns)):
        try:
            col_series = df.iloc[:, idx]
            if col_series.dropna().empty:
                continue
            col_max = float(col_series.max(skipna=True))
            change = _edge_change_pct(col_series)
            ranked_cols.append((idx, str(col), abs(change) if change is not None else -1.0, col_max))
        except Exception:
            continue
    ranked_cols.sort(key=lambda x: (x[2], x[3]), reverse=True)
    selected = ranked_cols[: max(int(top_n), 0)] if top_n is not None else ranked_cols
    shift_h = _time_shift_hours()
    det_cfg = stable_detection_cfg if isinstance(stable_detection_cfg, dict) else {}
    # RPS_DEBUG logs are limited to the SLA target label (debug_target_label) when it is set.
    debug_target_label = str(det_cfg.get("debug_target_label") or "").strip().lower()
    section_label_str = str(section_label or "")
    debug_this_section = _cfg_bool(det_cfg.get("debug_peak_logging"), default=False) and (
        not debug_target_label or debug_target_label in section_label_str.lower()
    )
    if debug_this_section:
        logger.info(
            "[RPS_DEBUG][summarize] section='%s' rows=%d cols=%d selected_top=%d min_stable_minutes=%s step_profile=%s",
            section_label_str, len(df.index), len(df.columns), len(selected), min_stable_minutes,
            bool(det_cfg.get("use_step_profile")),
        )

    def _shifted(ts_value: object) -> Optional[str]:
        if ts_value is None:
            return None
        if shift_h:
            try:
                return str(pd.Timestamp(ts_value).to_pydatetime() + pd.to_timedelta(shift_h, unit="h"))
            except Exception:
                return str(ts_value)
        return str(ts_value)

    for col_idx, col_name, _, _ in selected:
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
                "change_pct": _edge_change_pct(col_series),
                "max_time": str(max_idx) if pd.notnull(max_idx) else None,
                "min_time": str(min_idx) if pd.notnull(min_idx) else None,
            }
            raw_max = col_series.idxmax()
            raw_min = col_series.idxmin()
            if isinstance(df.index, pd.DatetimeIndex) and pd.notnull(raw_max) and pd.notnull(raw_min):
                series_summary["max_time_iso"] = format_iso(raw_max, shift_h)
                series_summary["min_time_iso"] = format_iso(raw_min, shift_h)
                if start_ts is not None:
                    series_summary["max_minute_from_start"] = minutes_from(raw_max, start_ts)
                    series_summary["min_minute_from_start"] = minutes_from(raw_min, start_ts)
            stable = None
            if include_stable_max and bool(det_cfg.get("use_step_profile")):
                stable = _find_stable_peak_step_profile(
                    col_series,
                    min_stable_minutes=min_stable_minutes,
                    cfg=det_cfg,
                )
                if stable is None and debug_this_section:
                    logger.info("[RPS_DEBUG][summarize] series='%s' step_profile returned None -> fallback generic", col_name)
            if stable is None and include_stable_max:
                stable = _find_stable_peak(
                    col_series,
                    min_stable_minutes=min_stable_minutes,
                    max_cv=_cfg_float(det_cfg.get("step_max_cv"), 0.20),
                )
            if stable:
                series_summary["stable_max"] = stable["stable_max"]
                # Unshifted bounds match the metric index. Display times below are shifted.
                series_summary["stable_window_start"] = stable.get("stable_start_time")
                series_summary["stable_window_end"] = stable.get("stable_max_time")
                series_summary["stable_max_time"] = _shifted(stable.get("stable_max_time"))
                series_summary["stable_start_time"] = _shifted(stable.get("stable_start_time"))
                series_summary["stable_duration_min"] = stable.get("stable_duration_min")
                series_summary["stable_method"] = stable.get("method")
                if "step_segments" in stable:
                    series_summary["step_segments"] = [
                        seg for seg in stable["step_segments"] if is_load_step_segment(seg, min_stable_minutes)
                    ]
                if debug_this_section:
                    logger.info(
                        "[RPS_DEBUG][summarize] series='%s' max=%s stable_max=%s stable_method=%s stable_duration_min=%s step_segments=%s",
                        col_name, max_val, stable.get("stable_max"), stable.get("method"),
                        stable.get("stable_duration_min"), stable.get("step_segments"),
                    )
            elif debug_this_section:
                logger.info("[RPS_DEBUG][summarize] series='%s' max=%s stable_max=None (no stable segment)", col_name, max_val)
        except Exception:
            continue
        summary.append(series_summary)
    return summary


def build_context_pack(
    labeled_dfs: List[Dict[str, object]],
    top_n: int = 10,
    min_stable_minutes: float = 5.0,
    stable_detection_cfg: Optional[Dict[str, Any]] = None,
    steps: Optional[List[LoadStep]] = None,
    start_ts: Optional[float] = None,
    anomaly_cfg: Optional[Dict[str, Any]] = None,
    include_stable_max: bool = True,
) -> Dict[str, object]:
    """Формирует компактное описание серий, таблицы по ступеням и аномалий по домену.

    При переданных ``steps`` аномалии считаются относительно ступеней
    (elasticity / drift / shift), иначе остаётся порог mean+2σ.
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
            section_label=str(label or ""),
            start_ts=start_ts,
            include_stable_max=include_stable_max,
        )
        anomalies: List[Dict[str, object]] = []
        if isinstance(df, pd.DataFrame) and not df.empty:
            for s in section_summary:
                series_name = s.get("series")
                if series_name not in df.columns:
                    continue
                if steps:
                    step_anomalies = detect_step_anomalies(df[series_name], steps, anomaly_cfg)
                    if step_anomalies:
                        anomalies.append({"series": series_name, "step_anomalies": step_anomalies})
                else:
                    windows = _detect_anomaly_windows(df[series_name])
                    if windows:
                        anomalies.append({"series": series_name, "windows": windows})
        sections.append({
            "label": label,
            "top_series": section_summary,
            "anomalies": anomalies
        })
    out: Dict[str, object] = {"sections": sections}
    if steps:
        out["load_steps"] = [step.to_dict() for step in steps]
        out["step_table"] = step_table(labeled_dfs, steps, top_n=min(int(top_n), 8))
    if isinstance(stable_detection_cfg, dict) and stable_detection_cfg:
        out["stable_detection"] = dict(stable_detection_cfg)
    return out


def _designated_rps_series(lt_labeled: List[Dict[str, object]], perf_label: str) -> Optional[pd.Series]:
    """Series with the highest peak inside the section designated by ``sla.max_performance_query``."""
    section = _find_section_by_label(lt_labeled, perf_label) if perf_label else None
    df = section.get("df") if isinstance(section, dict) else None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    best_col: Optional[str] = None
    best_max: Optional[float] = None
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        current = float(values.max())
        if best_max is None or current > best_max:
            best_col, best_max = col, current
    return df[best_col] if best_col is not None else None


def _labels_with_data(labeled: List[Dict[str, object]]) -> List[str]:
    """Query labels whose frames hold at least one value."""
    labels: List[str] = []
    for item in labeled or []:
        frame = item.get("df") if isinstance(item, dict) else None
        if isinstance(frame, pd.DataFrame) and not frame.dropna(how="all").empty:
            labels.append(str(item.get("label") or ""))
    return [label for label in labels if label]


@dataclass(frozen=True)
class RunSteps:
    """Load steps of a run and the first moment its designated RPS fell and did not recover."""
    steps: List[LoadStep]
    rps_drop: Optional[pd.Timestamp]


def _derive_steps_for_run(
    lt_labeled: List[Dict[str, object]],
    perf_label: str,
    stable_cfg: Optional[Dict[str, Any]],
    min_stable_minutes: float,
    start_ts: float,
    end_ts: float,
    shift_hours: int,
) -> RunSteps:
    """Load steps of the run: step detector on the designated RPS series, else equal time buckets."""
    segments = None
    rps_series = _designated_rps_series(lt_labeled, perf_label)
    if rps_series is not None and isinstance(stable_cfg, dict) and _cfg_bool(stable_cfg.get("use_step_profile"), default=False):
        profile = _find_stable_peak_step_profile(rps_series, min_stable_minutes=min_stable_minutes, cfg=stable_cfg)
        segments = (profile or {}).get("step_segments")
    steps = derive_load_steps(segments, start_ts, end_ts, shift_hours, min_step_minutes=min_stable_minutes)
    logger.info("Load steps for the run: %d (%s)", len(steps), steps[0].source if steps else "none")
    return RunSteps(steps=steps, rps_drop=first_rps_drop(segments))


def _baseline_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else (CONFIG.get("llm") or {})
    raw = llm_cfg.get("baseline") if isinstance(llm_cfg.get("baseline"), dict) else {}
    return {"enabled": _cfg_bool(raw.get("enabled"), default=True), "mode": str(raw.get("mode") or BASELINE_MODE_PREVIOUS_SUCCESS)}


def _load_baseline_context(
    cfg: Dict[str, Any],
    run_meta: Optional[Dict[str, object]],
    storage_cfg: Dict[str, Any],
    save_to_db: bool,
    start_ts: float,
) -> Dict[str, Any]:
    """Previous run of the same service (within the project area) and its series statistics."""
    settings = _baseline_settings(cfg)
    if not settings["enabled"]:
        return {"available": False, "reason": "baseline отключён в настройках (llm.baseline.enabled=false)"}
    if not save_to_db or not storage_cfg:
        return {"available": False, "reason": "хранилище метрик не используется для этого запуска"}
    service = str((run_meta or {}).get("service") or "").strip()
    run_name = str((run_meta or {}).get("run_name") or "").strip()
    project_area = str((run_meta or {}).get("project_area") or "").strip()
    area_services = (run_meta or {}).get("area_services")
    services = [str(item).strip() for item in area_services] if isinstance(area_services, list) else None
    if not service and not services:
        return {"available": False, "reason": "сервис запуска не задан"}
    schema = str(storage_cfg.get("schema", "public"))
    try:
        conn = _db_connect(storage_cfg)
        try:
            previous = find_previous_run(
                conn, schema, str(storage_cfg.get("llm_table", "llm_reports")),
                exclude_run_name=run_name, mode=settings["mode"], service=service or None,
                services=services, project_area=project_area or None,
                before=datetime.fromtimestamp(start_ts + 1, tz=timezone.utc),
            )
            if previous is None:
                label = "успешный прогон" if settings["mode"] == BASELINE_MODE_PREVIOUS_SUCCESS else "прогон"
                scope = f"области «{project_area}»" if project_area else f"сервиса «{service}»"
                return {"available": False, "reason": f"для {scope} нет более раннего {label}а"}
            stats = load_run_metric_stats(conn, schema, str(storage_cfg.get("table", "metrics")), previous.run_name)
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Baseline lookup failed for service '%s': %s", service, exc)
        return {"available": False, "reason": f"ошибка чтения baseline из БД: {exc}"}
    return {
        "available": True,
        "mode": settings["mode"],
        "run_name": previous.run_name,
        "verdict": previous.verdict,
        "created_at": previous.created_at.isoformat() if previous.created_at else None,
        "stats": stats,
    }


def _delta_pct(current: Optional[float], base: Optional[float]) -> Optional[float]:
    if current is None or base is None or abs(base) < 1e-9:
        return None
    return round((current / base - 1.0) * 100.0, 1)


def _baseline_for_domain(baseline_ctx: Dict[str, Any], domain: str, pack: Dict[str, Any]) -> Dict[str, Any]:
    """Baseline statistics for the series present in the current pack of one domain."""
    if not baseline_ctx.get("available"):
        return {"available": False, "reason": baseline_ctx.get("reason")}
    domain_stats = (baseline_ctx.get("stats") or {}).get(domain) or {}
    sections: Dict[str, Dict[str, Any]] = {}
    for section in pack.get("sections") or []:
        label = str(section.get("label") or "")
        base_series = domain_stats.get(label) or {}
        rows: Dict[str, Any] = {}
        for item in section.get("top_series") or []:
            name = str(item.get("series") or "")
            base = base_series.get(name)
            if base is None:
                continue
            rows[name] = {
                "baseline_mean": round(base.mean, 4),
                "baseline_p95": round(base.p95, 4),
                "baseline_max": round(base.max, 4),
                "delta_mean_pct": _delta_pct(item.get("mean"), base.mean),
                "delta_max_pct": _delta_pct(item.get("max"), base.max),
            }
        if rows:
            sections[label] = rows
    return {
        "available": True,
        "run_name": baseline_ctx.get("run_name"),
        "verdict": baseline_ctx.get("verdict"),
        "created_at": baseline_ctx.get("created_at"),
        "sections": sections,
        "note": "delta_*_pct — изменение текущего прогона относительно baseline в процентах.",
    }


def _baseline_for_overall(baseline_ctx: Dict[str, Any], domain_data: Dict[str, Dict[str, Any]], max_rows: int = 15) -> Dict[str, Any]:
    """Summary of the largest deltas against the baseline across all domains."""
    if not baseline_ctx.get("available"):
        return {"available": False, "reason": baseline_ctx.get("reason")}
    deltas: List[Dict[str, Any]] = []
    for domain, payload in domain_data.items():
        pack = payload.get("pack") if isinstance(payload, dict) else None
        if not isinstance(pack, dict):
            continue
        per_domain = _baseline_for_domain(baseline_ctx, domain, pack)
        for label, rows in (per_domain.get("sections") or {}).items():
            for series, row in rows.items():
                if row.get("delta_mean_pct") is None:
                    continue
                deltas.append({"domain": domain, "label": label, "series": series, **row})
    deltas.sort(key=lambda d: abs(d["delta_mean_pct"]), reverse=True)
    return {
        "available": True,
        "run_name": baseline_ctx.get("run_name"),
        "verdict": baseline_ctx.get("verdict"),
        "created_at": baseline_ctx.get("created_at"),
        "key_deltas": deltas[: max(int(max_rows), 1)],
    }


def _analysis_text(parsed: LLMAnalysis) -> str:
    return json.dumps(parsed.dict(exclude_none=True), ensure_ascii=False)


def _verify_result(text: str, parsed: Any, ctx_json: str, domain: str) -> tuple[str, Any, Optional[VerificationSummary]]:
    """Flags findings of one LLM answer as verified/unverified and rebuilds its text from the model."""
    if not isinstance(parsed, LLMAnalysis):
        return text, parsed, None
    try:
        ctx_obj = json.loads(ctx_json) if ctx_json else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Context of domain '{domain}' is not valid JSON: {exc}") from exc
    verified, summary = verify_analysis(parsed, ctx_obj if isinstance(ctx_obj, dict) else {})
    verified = fill_missing_verdict_rationale(verified)
    logger.info(
        "Verification '%s': findings=%d verified=%d unverified=%d qualitative=%d",
        domain, summary.total, summary.verified, summary.unverified, summary.qualitative,
    )
    return _analysis_text(verified), verified, summary


def _compact_overall_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Removes duplicate bulk from the final prompt so it fits a 200k context window.

    Load steps already sit at the top level, and ``step_segments`` is detector
    debug data. The numbers the model and the verifier need stay in
    ``top_series``, ``step_table``, ``timeline`` and ``deterministic_sla``.
    """
    packed = copy.deepcopy(ctx)
    packed.pop("domains_tables_markdown", None)
    domains = packed.get("domains")
    if isinstance(domains, dict):
        for pack in domains.values():
            if not isinstance(pack, dict):
                continue
            pack.pop("load_steps", None)
            pack.pop("stable_detection", None)
            for section in pack.get("sections") or []:
                if not isinstance(section, dict):
                    continue
                for series in section.get("top_series") or []:
                    if isinstance(series, dict):
                        series.pop("step_segments", None)
    return packed


def _compact_verification_context(base_ctx: Dict[str, Any]) -> Dict[str, Any]:
    baseline = base_ctx.get("baseline") if isinstance(base_ctx.get("baseline"), dict) else {}
    return {
        "time_range": base_ctx.get("time_range"),
        "load_steps": base_ctx.get("load_steps"),
        "timeline": base_ctx.get("timeline"),
        "deterministic_sla": base_ctx.get("deterministic_sla"),
        "designated_peak_performance": base_ctx.get("designated_peak_performance"),
        "baseline": {k: baseline.get(k) for k in ("available", "reason", "run_name", "verdict", "key_deltas") if k in baseline},
        "test_profile": base_ctx.get("test_profile"),
    }


def _run_verification_pass(final_parsed: LLMAnalysis, base_ctx: Dict[str, Any], prompt_dir: str) -> Optional[LLMAnalysis]:
    """Asks the model to re-evaluate the final verdict using only verified findings."""
    from AI.providers import ask_llm_with_text_data

    prompt = read_prompt_from_file(os.path.join(prompt_dir, "verify_prompt.txt"))
    payload = {
        "analysis": final_parsed.dict(exclude_none=True),
        "unverified_findings": unverified_findings(final_parsed),
        "context": _compact_verification_context(base_ctx),
    }
    with usage_domain("verify"):
        raw = ask_llm_with_text_data(prompt, json.dumps(payload, ensure_ascii=False))
    revised = parse_llm_analysis_strict(raw)
    if revised is None:
        logger.error("Verification pass returned an unparseable answer; keeping the flagged original")
        return None
    if not revised.findings and final_parsed.findings:
        logger.error("Verification pass dropped all findings; keeping the flagged original")
        return None
    re_verified, summary = verify_analysis(revised, base_ctx, revised_by_model=True)
    re_verified = fill_missing_verdict_rationale(re_verified)
    logger.info(
        "Verification pass: verdict %s -> %s, unverified=%d",
        final_parsed.verdict, re_verified.verdict, summary.unverified,
    )
    return re_verified


def _collect_domain_frames(
    resolved: ResolvedSource,
    qcfg: dict,
    language: str,
    start_ts: float,
    end_ts: float,
    step: str,
    resample: str,
) -> List[pd.DataFrame]:
    """Runs one domain's queries in the language ``domain_query_language`` selected."""
    source = resolved.config
    if language == "promql":
        return fetch_and_aggregate_with_label_keys(
            resolved.prometheus_url,
            start_ts,
            end_ts,
            qcfg.get("promql_queries", []),
            qcfg.get("label_keys_list", []),
            step=step,
            resample_interval=resample,
            ef_config={"metrics_source": source},
        )
    grafana = source.get("grafana") or {}
    influx = source.get("influxdb") or {}
    if language == "influxql":
        logger.info(
            "%s uses Grafana InfluxQL: queries=%s datasource=%s database=%s",
            resolved.domain,
            len(qcfg.get("influxql_queries") or []),
            ((grafana.get("influxdb_datasource") or grafana.get("prometheus_datasource") or {}).get("name")),
            influx.get("database"),
        )
        return fetch_influxql_and_aggregate_via_grafana(
            grafana_cfg=grafana,
            influx_aux_cfg=influx,
            start_ts=start_ts,
            end_ts=end_ts,
            influxql_queries=qcfg.get("influxql_queries", []),
            label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
            labels=qcfg.get("labels", []),
            resample_interval=resample,
        )
    if resolved.source_type == "influxdb":
        return fetch_influx_and_aggregate(
            influx_cfg=influx,
            start_ts=start_ts,
            end_ts=end_ts,
            flux_queries=qcfg.get("flux_queries", []),
            label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
            labels=qcfg.get("labels", []),
            resample_interval=resample,
        )
    logger.info("%s uses Grafana Flux: queries=%s", resolved.domain, len(qcfg.get("flux_queries") or []))
    return fetch_influx_and_aggregate_via_grafana(
        grafana_cfg=grafana,
        influx_aux_cfg=influx,
        start_ts=start_ts,
        end_ts=end_ts,
        flux_queries=qcfg.get("flux_queries", []),
        label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
        labels=qcfg.get("labels", []),
        resample_interval=resample,
    )


def _prepare_metric_domains(cfg: dict, queries: dict, collect_keys: List[str]) -> Dict[str, tuple]:
    """Resolves a source per enabled domain before collection starts.

    A broken binding fails the report here. A domain with no queries stays empty.
    """
    prepared: Dict[str, tuple] = {}
    for key in collect_keys:
        if key == APPLICATION_LOGS_DOMAIN:
            continue
        qcfg = queries.get(key)
        if not isinstance(qcfg, dict):
            continue
        resolved = resolve_domain_source(cfg, key)
        language = domain_query_language(resolved, qcfg)
        if language is None:
            continue
        prepared[key] = (resolved, language)
    return prepared


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
    progress_callback: ProgressCallback | None = None,
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
        progress_callback (callable | None): Принимает `(message, percent)` на каждой фазе pipeline.

    Возвращает:
        dict: Структура с текстовыми блоками, JSON-парсами и оценками качества.

    Побочные эффекты:
        Выполняет сетевые обращения (Grafana/Prometheus/Influx/OpenSearch/LLM), может писать в TimescaleDB.

    Исключения:
        Пробрасывает любые ошибки сбора данных или сохранения в БД.
    """
    _configure_logging()

    def _progress(message: str, percent: Optional[int]) -> None:
        if progress_callback is not None:
            progress_callback(message, percent)

    cfg = ef_config or CONFIG
    active_system_context = (
        copy.deepcopy(system_context)
        if isinstance(system_context, dict)
        else copy.deepcopy(cfg.get("system_context") or {})
    )
    system_context_brief = _system_context_prompt_summary(active_system_context)
    step = (cfg.get("default_params", {}) or {}).get("step") or CONFIG["default_params"]["step"]
    resample = (cfg.get("default_params", {}) or {}).get("resample_interval") or CONFIG["default_params"]["resample_interval"]
    logger.info("Metric sampling: step=%s resample_interval=%s", step, resample)

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
    if lt_stable_cfg["debug_peak_logging"]:
        logger.info(
            "[RPS_DEBUG][config] run='%s' service='%s' test_type='%s' target_rps=%s query='%s' min_stable_minutes=%s "
            "allow_peak_fallback=%s step_cfg=%s",
            str((run_meta or {}).get("run_name") or ""),
            str((run_meta or {}).get("service") or ""),
            run_test_type,
            sla_early.get("target_rps"),
            sla_early.get("max_performance_query"),
            min_stable_min,
            _cfg_bool(sla_early.get("target_rps_allow_peak_fallback"), default=True),
            lt_stable_cfg,
        )

    queries = cfg.get("queries") or CONFIG.get("queries") or {}
    # Определяем доступные домены (включая lt_framework и application_logs, если заданы)
    domain_keys = ["jvm", "database", "kafka", "microservices", "hard_resources"]
    if isinstance(queries.get("lt_framework"), dict):
        domain_keys.append("lt_framework")
    logs_cfg = cfg.get("logs_source") or CONFIG.get("logs_source") or {}
    if isinstance(logs_cfg, dict) and bool(logs_cfg.get("enabled")):
        domain_keys.append(APPLICATION_LOGS_DOMAIN)
    enabled_domain_set = set(domain_keys if active_domains is None else [d for d in active_domains if d in domain_keys])

    def _is_enabled(key: str) -> bool:
        return active_domains is None or key in enabled_domain_set
    domain_data = {}
    def _empty_domain_payload(key: str) -> Dict[str, object]:
        return {"labeled": [], "markdown": "", "pack": {"sections": []}, "ctx": json.dumps({"domain": key, "sections": []}, ensure_ascii=False)}
    collect_keys = [k for k in domain_keys if _is_enabled(k)]
    metric_sources = _prepare_metric_domains(cfg, queries, collect_keys)
    collected_count = 0
    for key in domain_keys:
        try:
            if not _is_enabled(key):
                domain_data[key] = _empty_domain_payload(key)
                continue
            _progress(
                f"Сбор метрик: {_domain_title(key)} ({collected_count + 1}/{len(collect_keys)})",
                _phase_percent(PROGRESS_COLLECT_START, PROGRESS_COLLECT_END, collected_count, len(collect_keys)),
            )
            collected_count += 1
            if key == APPLICATION_LOGS_DOMAIN:
                domain_data[key] = collect_application_logs(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    logs_cfg=logs_cfg,
                    resample_interval=resample,
                )
                continue
            prepared = metric_sources.get(key)
            if prepared is None:
                domain_data[key] = _empty_domain_payload(key)
                continue
            resolved, language = prepared
            qcfg = queries.get(key) or {}
            dfs = _collect_domain_frames(resolved, qcfg, language, start_ts, end_ts, step, resample)
            labeled = label_dataframes(dfs, qcfg.get("labels", []))
            markdown = dataframes_to_markdown(labeled)
            # Packs are built in a second pass, once load steps are known from lt_framework.
            domain_data[key] = {"labeled": labeled, "markdown": markdown, "pack": {"sections": []}, "ctx": ""}
        except Exception as e:
            logger.error(f"Domain '{key}' build failed: {e}")
            domain_data[key] = {"labeled": [], "markdown": "", "pack": {"sections": []}, "ctx": json.dumps({"domain": key, "sections": []}, ensure_ascii=False)}

    storage_cfg = ((cfg.get("storage", {}) or {}).get("timescale") or (CONFIG.get("storage", {}) or {}).get("timescale") or {})

    _progress("Подготовка контекста: ступени нагрузки и baseline", PROGRESS_COLLECT_END)
    shift_hours = _time_shift_hours()
    run_steps = _derive_steps_for_run(
        lt_labeled=domain_data.get("lt_framework", {}).get("labeled") or [],
        perf_label=str(sla_early.get("max_performance_query") or ""),
        stable_cfg=lt_stable_cfg,
        min_stable_minutes=min_stable_min,
        start_ts=start_ts,
        end_ts=end_ts,
        shift_hours=shift_hours,
    )
    load_steps = run_steps.steps
    anomaly_cfg = {
        "step_elasticity_threshold": _cfg_float(sla_early.get("step_elasticity_threshold"), 2.0),
        "step_drift_pct": _cfg_float(sla_early.get("step_drift_pct"), 20.0),
    }
    time_range_ctx = {
        "start": start_ts,
        "end": end_ts,
        "start_iso": format_iso(pd.Timestamp(start_ts, unit="s", tz="UTC"), shift_hours),
        "end_iso": format_iso(pd.Timestamp(end_ts, unit="s", tz="UTC"), shift_hours),
        "duration_min": round((end_ts - start_ts) / 60.0, 1),
    }
    baseline_ctx = _load_baseline_context(cfg, run_meta, storage_cfg, save_to_db, start_ts)
    for key in domain_keys:
        payload = domain_data.get(key) or {}
        if key == APPLICATION_LOGS_DOMAIN or not _is_enabled(key) or payload.get("ctx"):
            continue
        labeled = payload.get("labeled") or []
        pack = build_context_pack(
            labeled,
            top_n=15,
            min_stable_minutes=min_stable_min,
            stable_detection_cfg=lt_stable_cfg if key == "lt_framework" else None,
            include_stable_max=(key == "lt_framework"),
            steps=load_steps,
            start_ts=start_ts,
            anomaly_cfg=anomaly_cfg,
        )
        ctx_obj: Dict[str, Any] = {
            "domain": key,
            "time_range": time_range_ctx,
            "test_profile": test_profile,
            **pack,
        }
        ctx_obj["baseline"] = _baseline_for_domain(baseline_ctx, key, pack)
        if _has_meaningful_system_context(active_system_context):
            ctx_obj["system_context"] = active_system_context
        payload["pack"] = pack
        payload["ctx_obj"] = ctx_obj
        domain_data[key] = payload

    lt_pack = domain_data.get("lt_framework", {}).get("pack", {})
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
            debug=bool(lt_stable_cfg["debug_peak_logging"]),
        )
        if rps_pick.get("value") is not None:
            try:
                designated_peak = float(rps_pick.get("value"))
            except (TypeError, ValueError):
                designated_peak = None
            designated_series = str(rps_pick.get("source_series") or "") or None
            designated_source_label = str(rps_pick.get("source_label") or perf_query_label or "") or None
            designated_method = str(rps_pick.get("method") or "") or None
        if lt_stable_cfg["debug_peak_logging"]:
            logger.info(
                "[RPS_DEBUG][designated] value=%s label='%s' series='%s' method='%s' reason='%s'",
                designated_peak, designated_source_label, designated_series, designated_method, rps_pick.get("reason"),
            )

    _progress("Проверка SLA-критериев", PROGRESS_SLA)
    sla_cfg = copy.deepcopy(cfg.get("sla") or CONFIG.get("sla") or {})
    if not peak_performance_applicable:
        stability_required_rps = sla_cfg.get("required_rps", sla_cfg.get("stability_required_rps"))
        if stability_required_rps is not None:
            sla_cfg["target_rps"] = stability_required_rps
        else:
            sla_cfg["target_rps"] = None
    sla_result = evaluate_sla(domain_data, sla_cfg, test_profile=test_profile)
    sla_result = _reconcile_sla_for_test_profile(sla_result, test_profile)
    sla_window = sla_result.get("stable_window") if isinstance(sla_result.get("stable_window"), dict) else None
    if sla_window is not None and peak_performance_applicable:
        # SLA keeps the last report step where latency/errors still hold; the peak must be the same step.
        designated_peak = float(sla_window["level"])
        designated_series = str(sla_window.get("series") or "") or designated_series
        designated_source_label = str(sla_window.get("label") or "") or designated_source_label
        designated_method = f"stable_max (query: {designated_source_label})"
        _pin_series_to_sla_window(lt_pack, sla_window)
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
    load_step_table = _load_step_table_for_report(
        load_steps,
        (domain_data.get("lt_framework") or {}).get("labeled") or [],
        sla_cfg,
        sla_window,
    )
    sla_step_ctx = _sla_step_context(sla_result, load_steps, shift_hours)
    for key in domain_keys:
        payload = domain_data.get(key)
        if not isinstance(payload, dict):
            continue
        ctx_obj = payload.pop("ctx_obj", None)
        if ctx_obj is None and key == APPLICATION_LOGS_DOMAIN and payload.get("ctx"):
            ctx_obj = json.loads(payload["ctx"])
        if not isinstance(ctx_obj, dict):
            continue
        if sla_step_ctx is not None:
            ctx_obj["sla_step"] = sla_step_ctx
        payload["ctx"] = json.dumps(ctx_obj, ensure_ascii=False)

    # Сохранение метрик доменов в TimescaleDB
    if save_to_db:
        _progress("Сохранение метрик в базу данных", PROGRESS_SAVE_METRICS)
        try:
            for key in domain_keys:
                if key == APPLICATION_LOGS_DOMAIN:
                    continue
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
    prompt_application_logs = read_prompt_from_file(os.path.join(prompt_dir, "application_logs_prompt.txt")) if os.path.exists(os.path.join(prompt_dir, "application_logs_prompt.txt")) else "Проанализируйте агрегированные ERROR-логи приложений из OpenSearch."
    if isinstance(prompts_override, dict):
        prompt_jvm = prompts_override.get("jvm", prompt_jvm)
        prompt_database = prompts_override.get("database", prompt_database)
        prompt_kafka = prompts_override.get("kafka", prompt_kafka)
        prompt_microservices = prompts_override.get("microservices", prompt_microservices)
        prompt_hard_resources = prompts_override.get("hard_resources", prompt_hard_resources)
        prompt_overall = prompts_override.get("overall", prompt_overall)
        prompt_lt_framework = prompts_override.get("lt_framework", prompt_lt_framework)
        prompt_application_logs = prompts_override.get(APPLICATION_LOGS_DOMAIN, prompt_application_logs)

    context_guide = read_prompt_from_file(os.path.join(prompt_dir, "_context_guide.txt"))
    response_format = read_prompt_from_file(os.path.join(prompt_dir, "_response_format.txt"))
    analysis_guidance = build_analysis_guidance(sla_early, active_system_context, run_test_type)

    def _domain_prompt(base_text: str, domain: str) -> str:
        return assemble_prompt(
            _augment_prompt_with_system_context(base_text, system_context_brief),
            context_guide,
            build_analysis_guidance(sla_early, active_system_context, run_test_type, domain=domain),
            response_format,
        )

    prompt_jvm = _domain_prompt(prompt_jvm, "jvm")
    prompt_database = _domain_prompt(prompt_database, "database")
    prompt_kafka = _domain_prompt(prompt_kafka, "kafka")
    prompt_microservices = _domain_prompt(prompt_microservices, "microservices")
    prompt_hard_resources = _domain_prompt(prompt_hard_resources, "hard_resources")
    prompt_lt_framework = _domain_prompt(prompt_lt_framework, "lt_framework")
    prompt_application_logs = _domain_prompt(prompt_application_logs, APPLICATION_LOGS_DOMAIN)

    include_tables = bool(((CONFIG.get("llm", {}) or {}).get("include_markdown_tables_in_context", False)))
    jvm_full_data = domain_data["jvm"]["markdown"]; jvm_pack = domain_data["jvm"]["pack"]; jvm_ctx = domain_data["jvm"]["ctx"]
    database_full_data = domain_data["database"]["markdown"]; database_pack = domain_data["database"]["pack"]; database_ctx = domain_data["database"]["ctx"]
    kafka_full_data = domain_data["kafka"]["markdown"]; kafka_pack = domain_data["kafka"]["pack"]; kafka_ctx = domain_data["kafka"]["ctx"]
    ms_full_data = domain_data["microservices"]["markdown"]; ms_pack = domain_data["microservices"]["pack"]
    hr_full_data = domain_data["hard_resources"]["markdown"]; hr_pack = domain_data["hard_resources"]["pack"]; hr_ctx = domain_data["hard_resources"]["ctx"]
    lt_full_data = domain_data.get("lt_framework", {}).get("markdown", "")
    lt_pack = domain_data.get("lt_framework", {}).get("pack", {})
    application_logs_full_data = domain_data.get(APPLICATION_LOGS_DOMAIN, {}).get("markdown", "")
    application_logs_pack = domain_data.get(APPLICATION_LOGS_DOMAIN, {}).get("pack", {})
    application_logs_ctx = domain_data.get(APPLICATION_LOGS_DOMAIN, {}).get(
        "ctx",
        json.dumps({"domain": APPLICATION_LOGS_DOMAIN, "intervals": []}, ensure_ascii=False),
    )

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
    ms_ctx_raw = domain_data.get("microservices", {}).get("ctx") or ""
    try:
        ms_ctx_obj = json.loads(ms_ctx_raw) if ms_ctx_raw else {"domain": "microservices", "time_range": time_range_ctx, "test_profile": test_profile, **ms_pack}
    except json.JSONDecodeError:
        ms_ctx_obj = {"domain": "microservices", "time_range": time_range_ctx, "test_profile": test_profile, **ms_pack}
    ms_ctx_obj["aux_resources"] = {"cpu_sections": cpu_sections, "memory_sections": mem_sections}
    if _has_meaningful_system_context(active_system_context):
        ms_ctx_obj["system_context"] = active_system_context
    ms_ctx = json.dumps(ms_ctx_obj, ensure_ascii=False)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    llm_cfg = (cfg.get("llm", {}) or CONFIG.get("llm", {}) or {})
    domain_workers = _domain_worker_count(llm_cfg)
    self_consistency_k = max(1, _cfg_int(llm_cfg.get("self_consistency_k"), 3))
    reset_usage()
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
    if APPLICATION_LOGS_DOMAIN in domain_keys and _is_enabled(APPLICATION_LOGS_DOMAIN):
        domains_jobs.append((APPLICATION_LOGS_DOMAIN, prompt_application_logs, application_logs_ctx))
    results_map: dict[str, tuple[str, object, dict]] = {}
    jobs_by_key = {k: (p, c) for (k, p, c) in domains_jobs}
    if domains_jobs:
        _progress(f"Анализ ИИ: {len(domains_jobs)} доменов, ожидание ответов модели", PROGRESS_LLM_START)
        llm_done = 0
        with ThreadPoolExecutor(max_workers=min(domain_workers, len(domains_jobs))) as executor:
            future_to_key = {
                executor.submit(llm_two_pass_self_consistency, p, c, self_consistency_k, True, k): k
                for (k, p, c) in domains_jobs
            }
            for fut in as_completed(future_to_key):
                key = future_to_key[fut]
                try:
                    text, parsed, score = fut.result()
                    results_map[key] = (text, parsed, score)
                    logger.info(
                        "LLM domain '%s': text_len=%d, parsed=%s, verdict=%s",
                        key, len(text or ""), parsed is not None, getattr(parsed, "verdict", None) if parsed else None,
                    )
                except Exception as e:
                    logger.error(f"LLM {key} analysis failed: {e}")
                    results_map[key] = ("{}", None, {})
                llm_done += 1
                _progress(
                    f"Анализ ИИ: готово {llm_done}/{len(domains_jobs)} ({_domain_title(key)})",
                    _phase_percent(PROGRESS_LLM_START, PROGRESS_LLM_END, llm_done, len(domains_jobs)),
                )

    # Domains that failed in the parallel pass are retried one by one: a burst of
    # parallel requests is the usual reason for provider-side failures.
    failed_keys = [k for k, (txt, prs, _) in results_map.items() if prs is None and (not txt or txt in ("{}", "null"))]
    if failed_keys:
        import time as _time
        logger.info("Retrying %d failed domain(s) sequentially: %s", len(failed_keys), failed_keys)
        for retry_key in failed_keys:
            _progress(f"Повторный анализ ИИ: {_domain_title(retry_key)}", PROGRESS_LLM_END)
            _time.sleep(5)
            prompt_r, ctx_r = jobs_by_key[retry_key]
            try:
                results_map[retry_key] = llm_two_pass_self_consistency(prompt_r, ctx_r, self_consistency_k, True, retry_key)
                logger.info("LLM domain '%s' retry succeeded", retry_key)
            except Exception as e:
                logger.error("LLM %s retry also failed: %s", retry_key, e)
                results_map[retry_key] = ("{}", make_failed_llm_analysis(e), {})

    for key in list(results_map.keys()):
        txt, prs, scr = results_map[key]
        ctx_for_key = jobs_by_key.get(key, ("", ""))[1]
        txt, prs, _summary = _verify_result(txt, prs, ctx_for_key, key)
        results_map[key] = (txt, prs, scr)

    def _parsed_to_dict(obj: Any) -> Optional[Dict[str, Any]]:
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

    def _strip_peak_from_text(text: str) -> str:
        raw = str(text or "")
        stripped = raw.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return raw
        try:
            obj = json.loads(stripped)
        except Exception:
            return raw
        if isinstance(obj, dict) and "peak_performance" in obj:
            obj.pop("peak_performance", None)
            return json.dumps(obj, ensure_ascii=False)
        return raw

    # peak_performance is only meaningful for lt_framework (and later the overall
    # block); other domains must not report a "max RPS" of their own.
    for k in list(results_map.keys()):
        if k == "lt_framework":
            continue
        txt, prs, scr = results_map.get(k, ("{}", None, {}))
        parsed_dict = _parsed_to_dict(prs)
        if isinstance(parsed_dict, dict) and "peak_performance" in parsed_dict:
            parsed_dict.pop("peak_performance", None)
            prs = parsed_dict
        results_map[k] = (_strip_peak_from_text(txt), prs, scr)

    for k in list(results_map.keys()):
        txt, prs, scr = results_map[k]
        aligned = _align_domain_verdict_with_sla(txt, _parsed_to_dict(prs), sla_step_ctx)
        if aligned is not None:
            logger.info("Domain '%s' verdict aligned with SLA step: Провал -> Есть риски", k)
            results_map[k] = (aligned[0], aligned[1], scr)

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
    if APPLICATION_LOGS_DOMAIN in domain_keys:
        answer_application_logs, application_logs_parsed, application_logs_score = _result_or_blank(APPLICATION_LOGS_DOMAIN)
    else:
        answer_application_logs, application_logs_parsed, application_logs_score = ("", None, {})

    merged_prompt_overall = (
        prompt_overall
        .replace("{answer_jvm}", answer_jvm)
        .replace("{answer_database}", answer_database)
        .replace("{answer_kafka}", answer_kafka)
        .replace("{answer_microservices}", answer_ms)
        .replace("{answer_hard_resources}", answer_hr)
        .replace("{answer_lt_framework}", answer_lt)
        .replace("{answer_application_logs}", answer_application_logs)
        .replace("{system_context_brief}", system_context_brief)
    )
    if "{system_context_brief}" not in prompt_overall:
        merged_prompt_overall = _augment_prompt_with_system_context(merged_prompt_overall, system_context_brief)
    merged_prompt_overall = assemble_prompt(merged_prompt_overall, context_guide, analysis_guidance, response_format)

    base_ctx = {
        "time_range": time_range_ctx,
        "load_steps": [step.to_dict() for step in load_steps],
        "rps_drop_iso": format_iso(run_steps.rps_drop, shift_hours) if run_steps.rps_drop is not None else None,
        "lt_series_labels": _labels_with_data(domain_data.get("lt_framework", {}).get("labeled") or []),
        "timeline": build_timeline(
            {k: (v.get("pack") or {}) for k, v in domain_data.items() if isinstance(v, dict) and _is_enabled(k)},
            load_steps,
        ),
        "baseline": _baseline_for_overall(baseline_ctx, {k: v for k, v in domain_data.items() if isinstance(v, dict) and _is_enabled(k)}),
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
            "stable_window": _designated_window_ctx(sla_window, shift_hours),
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
    if APPLICATION_LOGS_DOMAIN in domain_keys:
        base_ctx["domains"][APPLICATION_LOGS_DOMAIN] = application_logs_pack
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
    if APPLICATION_LOGS_DOMAIN in domain_keys:
        base_ctx["domains_tables_markdown"][APPLICATION_LOGS_DOMAIN] = application_logs_full_data
    overall_ctx = json.dumps(_compact_overall_context(base_ctx), ensure_ascii=False)
    logger.info("Final LLM context size: %d chars", len(overall_ctx))
    _progress("Итоговый анализ ИИ по всем доменам", PROGRESS_FINAL_LLM)
    try:
        with usage_domain("final"):
            final_answer, final_parsed, final_score = llm_two_pass_self_consistency(
                user_prompt=merged_prompt_overall,
                data_context=overall_ctx,
                k=self_consistency_k,
                return_scores=True,
                domain_key="final",
            )
    except Exception as exc:
        logger.error("Final LLM analysis failed: %s", exc)
        final_parsed = make_failed_llm_analysis(exc)
        final_answer = json.dumps(final_parsed.dict(), ensure_ascii=False)
        final_score = {}
    final_answer, final_parsed, final_verification = _verify_result(final_answer, final_parsed, overall_ctx, "final")
    if (
        isinstance(final_parsed, LLMAnalysis)
        and final_verification is not None
        and final_verification.unverified > 0
        and _cfg_bool(llm_cfg.get("verification_pass"), default=True)
    ):
        _progress("Верификационный проход: вердикт по подтверждённым находкам", PROGRESS_FINAL_LLM)
        try:
            revised = _run_verification_pass(final_parsed, base_ctx, prompt_dir)
        except Exception as exc:
            logger.error("Verification pass failed, keeping the flagged final analysis: %s", exc)
            revised = None
        if revised is not None:
            final_parsed = revised
            final_answer = _analysis_text(revised)

    _to_dict_maybe = _parsed_to_dict

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
        if load_step_table:
            final_payload["load_step_table"] = load_step_table
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
        **({
            APPLICATION_LOGS_DOMAIN: _compose_text(application_logs_full_data, "Анализ логов приложений:", answer_application_logs),
        } if APPLICATION_LOGS_DOMAIN in domain_keys else {}),
        "final": final_answer,
        "jvm_parsed": _to_dict_maybe(jvm_parsed),
        "database_parsed": _to_dict_maybe(database_parsed),
        "kafka_parsed": _to_dict_maybe(kafka_parsed),
        "ms_parsed": _to_dict_maybe(ms_parsed),
        "hard_resources_parsed": _to_dict_maybe(hr_parsed),
        "lt_framework_parsed": _to_dict_maybe(lt_parsed),
        **({
            f"{APPLICATION_LOGS_DOMAIN}_parsed": _to_dict_maybe(application_logs_parsed),
        } if APPLICATION_LOGS_DOMAIN in domain_keys else {}),
        "final_parsed": _to_dict_maybe(final_parsed),
        "scores": attach_usage_to_scores({
            "jvm": jvm_score,
            "database": database_score,
            "kafka": kafka_score,
            "microservices": ms_score,
            "hard_resources": hr_score,
            **({"lt_framework": lt_score} if "lt_framework" in domain_keys else {}),
            **({APPLICATION_LOGS_DOMAIN: application_logs_score} if APPLICATION_LOGS_DOMAIN in domain_keys else {}),
            "final": final_score,
        }),
        "contexts": {
            **{key: ctx for key, (_prompt, ctx) in jobs_by_key.items()},
            "final": overall_ctx,
        },
        "sla_verdict": sla_result.get("verdict"),
        "sla_checks": sla_result.get("checks", []),
        "sla_summary": sla_result.get("summary", ""),
        "sla_window": sla_result.get("stable_window"),
        "load_step_table": load_step_table,
        "system_context": active_system_context,
    }

    # Сохранение LLM результатов в отдельную таблицу (если включено)
    if save_to_db:
        _progress("Сохранение результатов анализа", PROGRESS_SAVE_RESULTS)
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


