"""Converts legacy ``metrics_source`` / ``lt_metrics_source`` into the catalog.

The conversion itself is pure. ``migrate_legacy_data_sources`` reads the runtime
file, writes the catalog once and keeps a copy of the previous file.
"""

from __future__ import annotations

import copy
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from AI.data_sources import DEFAULT_BINDING, LT_DOMAIN, SOURCE_TYPES, TITLE_MAX_LEN

logger = logging.getLogger(__name__)

LEGACY_METRICS = "metrics_source"
LEGACY_LOAD = "lt_metrics_source"
BACKUP_NAME = "settings_runtime.before-data-sources.json"
_TYPE_IDS = {"grafana_proxy": "grafana", "prometheus": "prometheus", "influxdb": "influxdb"}
_TYPE_TITLES = {"grafana_proxy": "Grafana", "prometheus": "Prometheus", "influxdb": "InfluxDB"}
_AUTH_KEYS = ("username", "password", "token")


@dataclass(frozen=True)
class AreaLegacy:
    """Overrides of one project area. ``None`` means the global section is inherited."""

    metrics: Optional[dict]
    load: Optional[dict]
    load_queries: Optional[dict]


@dataclass(frozen=True)
class MigrationResult:
    catalog: dict
    global_bindings: dict
    area_bindings: dict[str, dict]
    warnings: tuple[str, ...]


def convert_legacy_sources(
    global_metrics: Optional[dict],
    global_load: Optional[dict],
    global_load_queries: Optional[dict],
    areas: dict[str, AreaLegacy],
) -> MigrationResult:
    """Builds a catalog and bindings. Equal connections become one source."""
    catalog: dict = {}
    fingerprints: dict[tuple, str] = {}
    warnings: list[str] = []
    global_bindings: dict = {}
    _bind_scope("глобальные настройки", global_metrics, global_load, global_load_queries, global_bindings, catalog, fingerprints, warnings)
    area_bindings: dict[str, dict] = {}
    for name in sorted(areas):
        area = areas[name]
        if area.metrics is None and area.load is None:
            continue
        bindings: dict = {}
        _bind_scope(
            f"область {name}",
            _merged(global_metrics, area.metrics),
            _merged(global_load, area.load),
            _merged(global_load_queries, area.load_queries),
            bindings,
            catalog,
            fingerprints,
            warnings,
        )
        if DEFAULT_BINDING not in bindings and DEFAULT_BINDING in global_bindings:
            bindings[DEFAULT_BINDING] = copy.deepcopy(global_bindings[DEFAULT_BINDING])
        if bindings:
            area_bindings[name] = bindings
    return MigrationResult(catalog, global_bindings, area_bindings, tuple(warnings))


def migrate_legacy_data_sources() -> bool:
    """Writes the catalog into the runtime file. Returns whether this process migrated."""
    from loadlens_app.core import CONFIG, CONFIG_RUNTIME_PATH, _load_settings_runtime_data, _save_settings_runtime_data

    runtime = _load_settings_runtime_data()
    if _catalog_ready(runtime, CONFIG):
        _forget_legacy(CONFIG)
        return False
    if not _has_legacy(runtime, CONFIG):
        return False
    areas = _area_legacy(runtime)
    result = convert_legacy_sources(
        _as_dict(CONFIG.get(LEGACY_METRICS)),
        _as_dict(CONFIG.get(LEGACY_LOAD)),
        _load_queries(CONFIG.get("queries")),
        areas,
    )
    for warning in result.warnings:
        logger.warning("%s", warning)
    _backup(CONFIG_RUNTIME_PATH)
    _apply(runtime, result, CONFIG)
    _save_settings_runtime_data(runtime)
    logger.info("Источники метрик перенесены в data_sources и domain_sources")
    return True


def _bind_scope(
    place: str,
    metrics: Optional[dict],
    load: Optional[dict],
    load_queries: Optional[dict],
    bindings: dict,
    catalog: dict,
    fingerprints: dict[tuple, str],
    warnings: list[str],
) -> None:
    default_id = _ensure_source(place, metrics, "метрик", catalog, fingerprints, warnings)
    if default_id and isinstance(metrics, dict):
        bindings[DEFAULT_BINDING] = _binding(metrics, default_id, "prometheus_datasource", place, warnings)
    if _is_empty(load):
        return
    load_id = _ensure_source(place, load, "метрик нагрузки", catalog, fingerprints, warnings)
    if load_id and isinstance(load, dict):
        bindings[LT_DOMAIN] = _binding(load, load_id, _load_datasource_key(load, _only_promql(load_queries)), place, warnings)


def _ensure_source(
    place: str,
    section: Optional[dict],
    role: str,
    catalog: dict,
    fingerprints: dict[tuple, str],
    warnings: list[str],
) -> Optional[str]:
    if _is_empty(section):
        return None
    assert isinstance(section, dict)
    source_type = str(section.get("type") or "").strip().lower()
    if source_type not in SOURCE_TYPES:
        warnings.append(f"{place}: неизвестный тип источника {role} «{source_type or 'не задан'}», раздел не перенесён")
        return None
    key = _fingerprint(source_type, section)
    if key is None:
        warnings.append(f"{place}: у источника {role} нет адреса, раздел не перенесён")
        return None
    existing = fingerprints.get(key)
    if existing:
        return existing
    source_id = _alloc_id(source_type, set(catalog))
    catalog[source_id] = _catalog_entry(source_type, section, _title(source_type, _url_of(source_type, section)))
    fingerprints[key] = source_id
    return source_id


def _binding(section: dict, source_id: str, datasource_key: str, place: str, warnings: list[str]) -> dict:
    binding: dict[str, str] = {"source": source_id}
    source_type = str(section.get("type") or "").strip().lower()
    if source_type == "grafana_proxy":
        _copy_datasource(section, datasource_key, binding, place, warnings)
        _copy_influx_target(section, binding, include_database=True)
    elif source_type == "influxdb":
        _copy_influx_target(section, binding, include_database=False)
    return binding


def _copy_datasource(section: dict, datasource_key: str, binding: dict, place: str, warnings: list[str]) -> None:
    grafana = section.get("grafana") if isinstance(section.get("grafana"), dict) else {}
    datasource = grafana.get(datasource_key) if isinstance(grafana.get(datasource_key), dict) else {}
    uid = str(datasource.get("uid") or "").strip()
    name = str(datasource.get("name") or "").strip()
    if uid:
        binding["datasource_uid"] = uid
    if name:
        binding["datasource_name"] = name
    if uid or name:
        return
    if isinstance(datasource.get("id"), int):
        warnings.append(f"{place}: датасорс задан только числовым id={datasource.get('id')} и не перенесён")
        return
    warnings.append(f"{place}: у Grafana не задан датасорс, укажите UID или имя")


def _copy_influx_target(section: dict, binding: dict, *, include_database: bool) -> None:
    influx = section.get("influxdb") if isinstance(section.get("influxdb"), dict) else {}
    if include_database:
        database = str(influx.get("database") or "").strip()
        if database:
            binding["database"] = database
    bucket = str(influx.get("bucket") or "").strip()
    if bucket:
        binding["bucket"] = bucket


def _load_datasource_key(section: dict, promql_only: bool) -> str:
    if promql_only or str(section.get("type") or "").strip().lower() != "grafana_proxy":
        return "prometheus_datasource"
    grafana = section.get("grafana") if isinstance(section.get("grafana"), dict) else {}
    influx = grafana.get("influxdb_datasource") if isinstance(grafana.get("influxdb_datasource"), dict) else {}
    if str(influx.get("uid") or "").strip() or str(influx.get("name") or "").strip():
        return "influxdb_datasource"
    return "prometheus_datasource"


def _catalog_entry(source_type: str, section: dict, title: str) -> dict:
    if source_type == "prometheus":
        return {"title": title, "type": source_type, "prometheus": {"url": _nested_text(section, "prometheus", "url")}}
    if source_type == "influxdb":
        block: dict[str, str] = {"url": _nested_text(section, "influxdb", "url")}
        for field in ("org", "token"):
            value = _nested_text(section, "influxdb", field)
            if value:
                block[field] = value
        return {"title": title, "type": source_type, "influxdb": block}
    return {"title": title, "type": source_type, "grafana": _grafana_connection(section)}


def _grafana_connection(section: dict) -> dict:
    grafana = section.get("grafana") if isinstance(section.get("grafana"), dict) else {}
    auth_raw = grafana.get("auth") if isinstance(grafana.get("auth"), dict) else {}
    method = str(auth_raw.get("method") or "basic").strip().lower()
    if method not in {"basic", "bearer"}:
        method = "basic"
    auth: dict[str, str] = {"method": method}
    for key in _AUTH_KEYS:
        if isinstance(auth_raw.get(key), str):
            auth[key] = auth_raw[key]
    connection: dict = {"base_url": _nested_text(section, "grafana", "base_url"), "auth": auth}
    if isinstance(grafana.get("verify_ssl"), bool):
        connection["verify_ssl"] = grafana["verify_ssl"]
    return connection


def _fingerprint(source_type: str, section: dict) -> Optional[tuple]:
    if source_type == "prometheus":
        url = _nested_text(section, "prometheus", "url")
        return ("prometheus", url) if url else None
    if source_type == "influxdb":
        url = _nested_text(section, "influxdb", "url")
        if not url:
            return None
        return ("influxdb", url, _nested_text(section, "influxdb", "org"), _nested_text(section, "influxdb", "token"))
    url = _nested_text(section, "grafana", "base_url")
    if not url:
        return None
    grafana = section.get("grafana") if isinstance(section.get("grafana"), dict) else {}
    auth = grafana.get("auth") if isinstance(grafana.get("auth"), dict) else {}
    verify = grafana.get("verify_ssl", True)
    return (
        "grafana_proxy",
        url,
        bool(verify) if isinstance(verify, bool) else True,
        str(auth.get("method") or "basic").strip().lower(),
        str(auth.get("username") or ""),
        str(auth.get("password") or ""),
        str(auth.get("token") or ""),
    )


def _alloc_id(source_type: str, used: set[str]) -> str:
    base = _TYPE_IDS[source_type]
    if base not in used:
        return base
    number = 2
    while f"{base}-{number}" in used:
        number += 1
    return f"{base}-{number}"


def _title(source_type: str, url: str) -> str:
    return f"{_TYPE_TITLES[source_type]} {_host_label(url)}"[:TITLE_MAX_LEN].strip()


def _host_label(url: str) -> str:
    text = url.strip()
    if not text:
        return "без адреса"
    parsed = urlparse(text if "://" in text else f"//{text}")
    host = parsed.hostname or ""
    if host and parsed.port:
        return f"{host}:{parsed.port}"
    return host or text


def _only_promql(qcfg: Optional[dict]) -> bool:
    if not isinstance(qcfg, dict):
        return False
    return _filled(qcfg, "promql_queries") and not _filled(qcfg, "influxql_queries") and not _filled(qcfg, "flux_queries")


def _filled(qcfg: dict, key: str) -> bool:
    queries = qcfg.get(key)
    return isinstance(queries, list) and any(str(item or "").strip() for item in queries)


def _merged(base: Optional[dict], override: Optional[dict]) -> dict:
    if override is None:
        return copy.deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return copy.deepcopy(base) if isinstance(base, dict) else {}
    return _deep_merge(base if isinstance(base, dict) else {}, override)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _is_empty(section: Optional[dict]) -> bool:
    return not isinstance(section, dict) or not str(section.get("type") or "").strip()


def _nested_text(section: dict, block: str, field: str) -> str:
    raw = section.get(block)
    if not isinstance(raw, dict):
        return ""
    value = raw.get(field)
    return value.strip() if isinstance(value, str) else ""


def _url_of(source_type: str, section: dict) -> str:
    if source_type == "prometheus":
        return _nested_text(section, "prometheus", "url")
    if source_type == "influxdb":
        return _nested_text(section, "influxdb", "url")
    return _nested_text(section, "grafana", "base_url")


def _catalog_ready(runtime: dict, config: dict) -> bool:
    return isinstance(runtime.get("data_sources"), dict) or isinstance(config.get("data_sources"), dict)


def _has_legacy(runtime: dict, config: dict) -> bool:
    if isinstance(config.get(LEGACY_METRICS), dict) or isinstance(config.get(LEGACY_LOAD), dict):
        return True
    return any(LEGACY_METRICS in entry or LEGACY_LOAD in entry for entry in _file_areas(runtime).values())


def _forget_legacy(config: dict) -> None:
    had_legacy = LEGACY_METRICS in config or LEGACY_LOAD in config
    config.pop(LEGACY_METRICS, None)
    config.pop(LEGACY_LOAD, None)
    if had_legacy:
        logger.warning("Удалите metrics_source и lt_metrics_source из settings.py: используется каталог data_sources")


def _area_legacy(runtime: dict) -> dict[str, AreaLegacy]:
    areas: dict[str, AreaLegacy] = {}
    for name, entry in _file_areas(runtime).items():
        if LEGACY_METRICS not in entry and LEGACY_LOAD not in entry:
            continue
        areas[name] = AreaLegacy(_override(entry, LEGACY_METRICS), _override(entry, LEGACY_LOAD), _query_override(entry))
    return areas


def _file_areas(runtime: dict) -> dict[str, dict]:
    raw = runtime.get("per_area")
    if not isinstance(raw, dict):
        return {}
    return {str(name): entry for name, entry in raw.items() if isinstance(entry, dict)}


def _override(entry: dict, key: str) -> Optional[dict]:
    if key not in entry:
        return None
    value = entry.get(key)
    return value if isinstance(value, dict) else {}


def _query_override(entry: dict) -> Optional[dict]:
    queries = entry.get("queries")
    if not isinstance(queries, dict) or LT_DOMAIN not in queries:
        return None
    block = queries.get(LT_DOMAIN)
    return block if isinstance(block, dict) else {}


def _as_dict(value: object) -> Optional[dict]:
    return value if isinstance(value, dict) else None


def _load_queries(queries: object) -> Optional[dict]:
    if not isinstance(queries, dict):
        return None
    block = queries.get(LT_DOMAIN)
    return block if isinstance(block, dict) else None


def _backup(path: Path) -> None:
    if not path.is_file():
        return
    backup = path.with_name(BACKUP_NAME)
    if backup.exists():
        return
    shutil.copyfile(path, backup)


def _apply(runtime: dict, result: MigrationResult, config: dict) -> None:
    runtime["data_sources"] = result.catalog
    runtime["domain_sources"] = result.global_bindings
    runtime.pop(LEGACY_METRICS, None)
    runtime.pop(LEGACY_LOAD, None)
    per_area = runtime.get("per_area")
    if isinstance(per_area, dict):
        for name, entry in per_area.items():
            if not isinstance(entry, dict):
                continue
            if name in result.area_bindings:
                entry["domain_sources"] = result.area_bindings[name]
            entry.pop(LEGACY_METRICS, None)
            entry.pop(LEGACY_LOAD, None)
    config["data_sources"] = copy.deepcopy(result.catalog)
    config["domain_sources"] = copy.deepcopy(result.global_bindings)
    config.pop(LEGACY_METRICS, None)
    config.pop(LEGACY_LOAD, None)
    config_areas = config.get("per_area")
    if isinstance(config_areas, dict):
        for name, bindings in result.area_bindings.items():
            entry = config_areas.get(name)
            if isinstance(entry, dict):
                entry["domain_sources"] = copy.deepcopy(bindings)
                entry.pop(LEGACY_METRICS, None)
                entry.pop(LEGACY_LOAD, None)


__all__ = [
    "BACKUP_NAME",
    "AreaLegacy",
    "MigrationResult",
    "convert_legacy_sources",
    "migrate_legacy_data_sources",
]
