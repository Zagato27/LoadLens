"""Shared helpers and runtime utilities for the Flask app."""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import psycopg2
from psycopg2 import sql
from flask import request

from AI.db_store import _ensure_engineer_reports_table, _ensure_llm_reports_table, _ensure_schema_and_table
from metrics_config import METRICS_CONFIG
from settings import CONFIG

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_RUNTIME_PATH = ROOT_DIR / "settings_runtime.json"
METRICS_RUNTIME_PATH = ROOT_DIR / "metrics_config_runtime.json"
PROMPTS_DIR = ROOT_DIR / "AI" / "prompts"
logger = logging.getLogger(__name__)

PROMPT_DOMAIN_FILES = {
    "overall": "overall_prompt.txt",
    "database": "database_prompt.txt",
    "kafka": "kafka_prompt.txt",
    "microservices": "microservices_prompt.txt",
    "jvm": "jvm_prompt.txt",
    "hard_resources": "hard_resources_prompt.txt",
    "lt_framework": "lt_framework_prompt.txt",
    "judge": "judge_prompt.txt",
    "critic": "critic_prompt.txt",
}
LOCKED_PROMPT_DOMAINS = {"judge", "critic"}
BOOTSTRAP_SECTIONS = ("llm", "metrics_source", "lt_metrics_source", "default_params", "queries", "prompts", "sla")


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out.get(k) or {}, v)
        else:
            out[k] = v
    return out


def _clean_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, (int, float)):
            text = str(item).strip()
        else:
            continue
        if text:
            out.append(text)
    return out


def _coerce_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = _clean_text(value).lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_object_list(
    value,
    *,
    string_fields: tuple[str, ...] = (),
    list_fields: tuple[str, ...] = (),
) -> list[dict]:
    items = value if isinstance(value, list) else []
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = copy.deepcopy(item)
        for key in string_fields:
            row[key] = _clean_text(item.get(key))
        for key in list_fields:
            row[key] = _clean_string_list(item.get(key))
        normalized.append(row)
    return normalized


def _normalize_system_context(payload: dict | None) -> dict:
    raw = payload if isinstance(payload, dict) else {}
    base_context = CONFIG.get("system_context")
    if not isinstance(base_context, dict):
        base_context = {}
    merged = _deep_merge_dicts(copy.deepcopy(base_context), raw)

    try:
        merged["schema_version"] = int(merged.get("schema_version") or 1)
    except (TypeError, ValueError):
        merged["schema_version"] = 1
    merged["enabled"] = _coerce_bool(merged.get("enabled"), default=True)

    system = merged.get("system") if isinstance(merged.get("system"), dict) else {}
    merged["system"] = {
        **copy.deepcopy(system),
        "name": _clean_text(system.get("name")),
        "domain": _clean_text(system.get("domain")),
        "description": _clean_text(system.get("description")),
        "test_goal": _clean_text(system.get("test_goal")),
    }

    architecture = merged.get("architecture") if isinstance(merged.get("architecture"), dict) else {}
    merged["architecture"] = {
        **copy.deepcopy(architecture),
        "style": _clean_text(architecture.get("style")),
        "components": _normalize_object_list(
            architecture.get("components"),
            string_fields=("id", "name", "role", "criticality"),
            list_fields=("technologies",),
        ),
        "dependencies": _normalize_object_list(
            architecture.get("dependencies"),
            string_fields=("from", "to", "kind", "purpose"),
        ),
        "data_stores": _normalize_object_list(
            architecture.get("data_stores"),
            string_fields=("id", "type", "purpose"),
            list_fields=("used_by",),
        ),
    }

    load_model = merged.get("load_model") if isinstance(merged.get("load_model"), dict) else {}
    merged["load_model"] = {
        **copy.deepcopy(load_model),
        "entrypoints": _normalize_object_list(
            load_model.get("entrypoints"),
            string_fields=("id", "name", "kind", "business_priority"),
        ),
        "critical_user_flows": _normalize_object_list(
            load_model.get("critical_user_flows"),
            string_fields=("id", "name"),
            list_fields=("steps", "success_signals"),
        ),
        "expected_hotspots": _clean_string_list(load_model.get("expected_hotspots")),
    }

    operational_context = (
        merged.get("operational_context")
        if isinstance(merged.get("operational_context"), dict)
        else {}
    )
    merged["operational_context"] = {
        **copy.deepcopy(operational_context),
        "known_constraints": _clean_string_list(operational_context.get("known_constraints")),
        "known_risks": _clean_string_list(operational_context.get("known_risks")),
        "normal_degradation_rules": _clean_string_list(operational_context.get("normal_degradation_rules")),
        "analysis_focus": _clean_string_list(operational_context.get("analysis_focus")),
    }
    return merged


def _active_system_context() -> dict:
    return _normalize_system_context(CONFIG.get("system_context"))


try:
    if CONFIG_RUNTIME_PATH.exists():
        with CONFIG_RUNTIME_PATH.open("r", encoding="utf-8") as f:
            _rt = json.load(f)
        CONFIG.update(_deep_merge_dicts(CONFIG, _rt))
except Exception:
    pass


def _active_project_area() -> str | None:
    try:
        val = request.cookies.get("project_area")
        return val if isinstance(val, str) and val.strip() else None
    except Exception:
        return None


def _load_settings_runtime_data() -> dict:
    try:
        if CONFIG_RUNTIME_PATH.exists():
            with CONFIG_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_settings_runtime_data(payload: dict) -> None:
    try:
        with CONFIG_RUNTIME_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload or {}, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Не удалось сохранить runtime-конфигурацию в %s", CONFIG_RUNTIME_PATH)


def _per_area_config() -> dict:
    data = _load_settings_runtime_data()
    per_area = data.get("per_area") if isinstance(data, dict) else {}
    return per_area if isinstance(per_area, dict) else {}


def _area_entry(area_name: str | None) -> dict:
    if not area_name:
        return {}
    entry = _per_area_config().get(area_name)
    return entry if isinstance(entry, dict) else {}


def _services_map_for_area(area_name: str | None) -> dict:
    entry = _area_entry(area_name)
    services = entry.get("services")
    return services if isinstance(services, dict) else {}


def _service_meta(area_name: str | None, service_id: str | None) -> dict:
    if not area_name or not service_id:
        return {}
    services = _services_map_for_area(area_name)
    entry = services.get(service_id) if isinstance(services, dict) else {}
    return entry if isinstance(entry, dict) else {}


def _service_disabled_domains(area_name: str | None, service_id: str | None) -> list[str]:
    meta = _service_meta(area_name, service_id)
    disabled = meta.get("disabled_domains") if isinstance(meta, dict) else []
    return [d for d in disabled if isinstance(d, str)]


def _load_base_prompts() -> dict:
    prompts: dict[str, str] = {}
    for domain, fname in PROMPT_DOMAIN_FILES.items():
        try:
            path = PROMPTS_DIR / fname
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    prompts[domain] = f.read()
            else:
                prompts[domain] = ""
        except Exception:
            prompts[domain] = ""
    return prompts


def _deep_copy_prompts(section: str) -> dict:
    if section == "prompts":
        return copy.deepcopy(_load_base_prompts())
    return copy.deepcopy(CONFIG.get(section, {}) or {})


def _active_metrics_config() -> dict:
    base = METRICS_CONFIG if isinstance(METRICS_CONFIG, dict) else {}
    raw = copy.deepcopy(base)
    try:
        if METRICS_RUNTIME_PATH.exists():
            with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                override = json.load(f)
            if isinstance(override, dict):
                for key, value in override.items():
                    if not isinstance(value, dict):
                        continue
                    value_copy = copy.deepcopy(value)
                    replace = bool(value_copy.pop("__replace__", False))
                    if replace:
                        raw[key] = value_copy
                    else:
                        base_entry = raw.get(key, {})
                        raw[key] = _deep_merge_dicts(base_entry if isinstance(base_entry, dict) else {}, value_copy)
    except Exception:
        pass
    return _normalize_metrics_config(raw)


def _service_area_map() -> dict:
    mapping: dict[str, str] = {}
    per_area = _per_area_config()
    for area_name, cfg in per_area.items():
        services = cfg.get("services")
        if isinstance(services, dict):
            for sid in services.keys():
                mapping[sid] = area_name
    return mapping


def _normalize_metrics_config(raw: dict | None) -> dict:
    normalized: dict[str, dict] = {}
    service_to_area = _service_area_map()

    def ensure(area_name: str) -> dict:
        if area_name not in normalized or not isinstance(normalized[area_name], dict):
            normalized[area_name] = {"services": {}}
        if "services" not in normalized[area_name] or not isinstance(normalized[area_name]["services"], dict):
            normalized[area_name]["services"] = {}
        return normalized[area_name]

    for area_name in _per_area_config().keys():
        ensure(area_name)

    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("services"), dict):
                entry = ensure(key)
                entry_services = entry.get("services", {})
                entry_services.update(value.get("services") or {})
                entry["services"] = entry_services
                for meta_key, meta_val in value.items():
                    if meta_key != "services":
                        entry[meta_key] = meta_val
                continue
            target_area = service_to_area.get(key) or value.get("area") or key
            entry = ensure(target_area)
            entry["services"][key] = value
    return normalized


def _metrics_services_for_area(area_name: str | None) -> dict:
    metrics = _active_metrics_config() or {}
    if not area_name:
        return {}
    entry = metrics.get(area_name) or {}
    services = entry.get("services")
    return services if isinstance(services, dict) else {}


def _metrics_service_entry(service_id: str | None) -> tuple[str | None, dict]:
    if not service_id:
        return None, {}
    metrics = _active_metrics_config() or {}
    for area_name, cfg in metrics.items():
        services = cfg.get("services")
        if isinstance(services, dict) and service_id in services:
            return area_name, services.get(service_id) or {}
    legacy = metrics.get(service_id)
    if isinstance(legacy, dict):
        services = legacy.get("services")
        if isinstance(services, dict) and service_id in services:
            return service_id, services.get(service_id) or {}
    return None, {}


def _available_domain_keys(cfg: dict | None = None) -> list[str]:
    config = cfg or CONFIG
    queries = config.get("queries") or {}
    preferred_order = ["jvm", "database", "kafka", "microservices", "hard_resources", "lt_framework"]
    result = [d for d in preferred_order if d in queries]
    for key in queries.keys():
        if key not in result:
            result.append(key)
    return result


def _resolve_services_for_area(area_name: str | None) -> list[str]:
    if not area_name:
        return []
    services_map = _services_map_for_area(area_name)
    if services_map:
        return list(services_map.keys())
    metrics_services = _metrics_services_for_area(area_name)
    if metrics_services:
        return list(metrics_services.keys())
    if area_name:
        return [area_name]
    return []


def _resolve_services_filter(area_name: str | None) -> list[str]:
    services = _resolve_services_for_area(area_name)
    if services:
        return services
    if area_name:
        return [area_name]
    return []


def _find_area_for_service(service_id: str | None) -> str | None:
    if not service_id:
        return None
    per_area = _per_area_config()
    for area_name, cfg in per_area.items():
        services = cfg.get("services")
        if isinstance(services, dict) and service_id in services:
            return area_name
    metrics = _active_metrics_config() or {}
    for area_name, cfg in metrics.items():
        services = cfg.get("services")
        if isinstance(services, dict) and service_id in services:
            return area_name
    return None


def _default_section_template(section: str) -> dict:
    return _deep_copy_prompts(section)


def _ensure_area_runtime(area_name: str):
    runtime = _load_settings_runtime_data()
    if "per_area" not in runtime or not isinstance(runtime.get("per_area"), dict):
        runtime["per_area"] = {}
    if area_name not in runtime["per_area"] or not isinstance(runtime["per_area"].get(area_name), dict):
        runtime["per_area"][area_name] = {}
    return runtime, runtime["per_area"][area_name]


def _bootstrap_area_defaults(area_name: str | None) -> None:
    if not area_name:
        return
    runtime, area_entry = _ensure_area_runtime(area_name)
    changed = False
    for section in BOOTSTRAP_SECTIONS:
        if section in area_entry and isinstance(area_entry[section], dict) and area_entry[section]:
            continue
        template = _default_section_template(section)
        area_entry[section] = template
        changed = True
    if changed:
        _save_settings_runtime_data(runtime)


def _bootstrap_service_configs(area_name: str | None, service_id: str | None) -> None:
    if not area_name or not service_id:
        return
    _bootstrap_area_defaults(area_name)
    runtime, area_entry = _ensure_area_runtime(area_name)
    if "services" not in area_entry or not isinstance(area_entry.get("services"), dict):
        area_entry["services"] = {}
    if service_id not in area_entry["services"] or not isinstance(area_entry["services"].get(service_id), dict):
        area_entry["services"][service_id] = {}
    service_entry = area_entry["services"][service_id]
    changed = False
    for section in BOOTSTRAP_SECTIONS:
        section_value = service_entry.get(section)
        if isinstance(section_value, dict) and section_value:
            continue
        inherit = area_entry.get(section)
        if not isinstance(inherit, dict) or not inherit:
            inherit = _default_section_template(section)
        service_entry[section] = copy.deepcopy(inherit)
        changed = True
    if changed:
        _save_settings_runtime_data(runtime)
    _bootstrap_metrics_service_config(area_name, service_id)


def _default_metrics_service_template() -> dict:
    for cfg in METRICS_CONFIG.values():
        if isinstance(cfg, dict):
            return copy.deepcopy(cfg)
    return {"page_sample_id": "", "page_parent_id": "", "metrics": [], "logs": []}


def _bootstrap_metrics_service_config(area_name: str | None, service_id: str | None) -> None:
    if not area_name or not service_id:
        return
    base_metrics = _default_metrics_service_template()
    current = {}
    try:
        if METRICS_RUNTIME_PATH.exists():
            with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                current = json.load(f)
    except Exception:
        current = {}
    if not isinstance(current, dict):
        current = {}
    if area_name not in current or not isinstance(current.get(area_name), dict):
        current[area_name] = {"services": {}}
    if "services" not in current[area_name] or not isinstance(current[area_name].get("services"), dict):
        current[area_name]["services"] = {}
    if service_id not in current[area_name]["services"]:
        current[area_name]["services"][service_id] = copy.deepcopy(base_metrics)
        with METRICS_RUNTIME_PATH.open("w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)


def _list_project_areas() -> list[dict]:
    per_area = _per_area_config()
    if per_area:
        areas = []
        for name, cfg in per_area.items():
            title = ""
            if isinstance(cfg, dict):
                title = cfg.get("title") if isinstance(cfg.get("title"), str) else ""
            areas.append({"id": name, "title": title or name})
        return areas
    metrics = _active_metrics_config() or {}
    return [{"id": name, "title": name} for name in metrics.keys()]


def _prompt_templates_for_scope(area: str | None, service: str | None = None) -> dict:
    base = _load_base_prompts()
    area_entry = _area_entry(area)
    area_prompts = area_entry.get("prompts") if isinstance(area_entry, dict) else {}
    service_prompts = {}
    if service:
        meta = _service_meta(area, service)
        service_prompts = meta.get("prompts") if isinstance(meta, dict) else {}
    out: dict[str, str] = {}
    for domain in PROMPT_DOMAIN_FILES.keys():
        if isinstance(service_prompts, dict) and isinstance(service_prompts.get(domain), str) and service_prompts.get(domain).strip():
            out[domain] = service_prompts.get(domain, "")
        elif isinstance(area_prompts, dict) and isinstance(area_prompts.get(domain), str) and area_prompts.get(domain).strip():
            out[domain] = area_prompts.get(domain, "")
        else:
            out[domain] = base.get(domain, "")
    return out


def _active_area_prompts(area: str | None, service: str | None = None) -> dict:
    if service and not area:
        area = _find_area_for_service(service)
    return _prompt_templates_for_scope(area, service)


def _disabled_domains_payload(area: str | None, service: str | None) -> list[str]:
    if not service:
        return []
    return _service_disabled_domains(area, service)


def _ts_conn():
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    return psycopg2.connect(
        host=cfg.get("host"),
        port=cfg.get("port"),
        dbname=cfg.get("dbname"),
        user=cfg.get("user"),
        password=cfg.get("password"),
        sslmode=cfg.get("sslmode", "prefer"),
    )


def _series_key_for(domain: str, query_label: str, default_key: str = "application") -> str:
    try:
        q = (CONFIG.get("queries", {}) or {}).get(domain, {})
        labels = list(q.get("labels", []) or [])
        keys_list = list(q.get("label_keys_list", []) or [])
        if query_label in labels:
            idx = labels.index(query_label)
            if idx < len(keys_list):
                keys = list(keys_list[idx] or [])
                if keys:
                    return str(keys[0])
        if keys_list:
            keys = list(keys_list[0] or [])
            if keys:
                return str(keys[0])
    except Exception:
        pass
    return default_key


def convert_to_timestamp(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
    return int(dt.timestamp() * 1000)


def _delete_run_data(run_name: str) -> None:
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema = cfg.get("schema", "public")
    metrics_table = cfg.get("table", "metrics")
    llm_table = cfg.get("llm_table", "llm_reports")
    engineer_table = cfg.get("engineer_table", "engineer_reports")
    conn = _ts_conn()
    with conn, conn.cursor() as cur:
        for table in (llm_table, engineer_table, metrics_table):
            try:
                cur.execute(f"DELETE FROM {schema}.{table} WHERE run_name = %s", (run_name,))
            except Exception:
                logger.exception("Не удалось удалить данные запуска '%s' из %s.%s", run_name, schema, table)
    conn.close()


def _rename_run_data(old_run_name: str, new_run_name: str) -> dict:
    old_name = str(old_run_name or "").strip()
    new_name = str(new_run_name or "").strip()
    if not old_name:
        raise ValueError("Текущее имя отчёта не задано")
    if not new_name:
        raise ValueError("Новое имя отчёта не задано")
    if len(new_name) > 180:
        raise ValueError("Новое имя отчёта слишком длинное")
    if any(ch in new_name for ch in ("/", "\\", "<", ">", "\x00")):
        raise ValueError("Новое имя отчёта содержит недопустимые символы: / \\ < >")
    if old_name == new_name:
        return {"status": "ok", "renamed": 0, "run_name": new_name}

    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema = cfg.get("schema", "public")
    metrics_table = cfg.get("table", "metrics")
    llm_table = cfg.get("llm_table", "llm_reports")
    engineer_table = cfg.get("engineer_table", "engineer_reports")
    tables = (metrics_table, llm_table, engineer_table)
    conn = _ts_conn()
    renamed = 0
    try:
        try:
            _ensure_schema_and_table(conn, cfg, schema, metrics_table)
            _ensure_llm_reports_table(conn, cfg)
            _ensure_engineer_reports_table(conn, cfg)
        except Exception:
            logger.exception("Не удалось подготовить таблицы для переименования отчёта")
        with conn, conn.cursor() as cur:
            old_exists = False
            new_exists = False
            for table in tables:
                cur.execute(
                    sql.SQL("SELECT 1 FROM {}.{} WHERE run_name = %s LIMIT 1").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    ),
                    (old_name,),
                )
                old_exists = bool(cur.fetchone()) or old_exists
                cur.execute(
                    sql.SQL("SELECT 1 FROM {}.{} WHERE run_name = %s LIMIT 1").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    ),
                    (new_name,),
                )
                new_exists = bool(cur.fetchone()) or new_exists
            if not old_exists:
                raise LookupError(f"Отчёт '{old_name}' не найден")
            if new_exists:
                raise FileExistsError(f"Отчёт '{new_name}' уже существует")
            for table in tables:
                cur.execute(
                    sql.SQL("UPDATE {}.{} SET run_name = %s WHERE run_name = %s").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    ),
                    (new_name, old_name),
                )
                renamed += int(cur.rowcount or 0)
        return {"status": "ok", "renamed": renamed, "run_name": new_name, "old_run_name": old_name}
    finally:
        conn.close()


def _delete_service_data(area_name: str, service_name: str) -> None:
    runtime = _load_settings_runtime_data()
    changed = False
    if isinstance(runtime.get("per_area"), dict):
        area_entry = runtime["per_area"].get(area_name)
        if isinstance(area_entry, dict) and isinstance(area_entry.get("services"), dict):
            if service_name in area_entry["services"]:
                del area_entry["services"][service_name]
                changed = True
    if changed:
        _save_settings_runtime_data(runtime)

    metrics_rt = {}
    try:
        if METRICS_RUNTIME_PATH.exists():
            with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                metrics_rt = json.load(f)
    except Exception:
        metrics_rt = {}
    if isinstance(metrics_rt.get(area_name), dict):
        services_entry = metrics_rt[area_name].get("services")
        if isinstance(services_entry, dict) and service_name in services_entry:
            del services_entry[service_name]
            with METRICS_RUNTIME_PATH.open("w", encoding="utf-8") as f:
                json.dump(metrics_rt, f, ensure_ascii=False, indent=2)

    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema = cfg.get("schema", "public")
    metrics_table = cfg.get("table", "metrics")
    llm_table = cfg.get("llm_table", "llm_reports")
    engineer_table = cfg.get("engineer_table", "engineer_reports")
    conn = _ts_conn()
    with conn, conn.cursor() as cur:
        for table in (metrics_table,):
            try:
                cur.execute(f"DELETE FROM {schema}.{table} WHERE service = %s", (service_name,))
            except Exception:
                pass
        for table in (llm_table, engineer_table):
            try:
                cur.execute(f"DELETE FROM {schema}.{table} WHERE service = %s", (service_name,))
            except Exception:
                pass
    conn.close()


def _bootstrap_service_queries(area_name: str, service_name: str) -> None:
    """Ensures runtime config for service exists (backward compatibility helper)."""
    _bootstrap_service_configs(area_name, service_name)


__all__ = [
    "CONFIG_RUNTIME_PATH",
    "METRICS_RUNTIME_PATH",
    "PROMPTS_DIR",
    "PROMPT_DOMAIN_FILES",
    "LOCKED_PROMPT_DOMAINS",
    "_active_project_area",
    "_active_metrics_config",
    "_active_area_prompts",
    "_active_system_context",
    "_available_domain_keys",
    "_bootstrap_area_defaults",
    "_bootstrap_metrics_service_config",
    "_bootstrap_service_configs",
    "_bootstrap_service_queries",
    "_delete_run_data",
    "_delete_service_data",
    "_disabled_domains_payload",
    "_find_area_for_service",
    "_list_project_areas",
    "_load_base_prompts",
    "_load_settings_runtime_data",
    "_metrics_service_entry",
    "_metrics_services_for_area",
    "_normalize_metrics_config",
    "_per_area_config",
    "_prompt_templates_for_scope",
    "_rename_run_data",
    "_resolve_services_filter",
    "_resolve_services_for_area",
    "_save_settings_runtime_data",
    "_series_key_for",
    "_service_meta",
    "_service_disabled_domains",
    "_services_map_for_area",
    "_ts_conn",
    "convert_to_timestamp",
    "_deep_merge_dicts",
    "_normalize_system_context",
]


