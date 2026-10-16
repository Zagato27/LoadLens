"""Shared helpers and runtime utilities for the Flask app."""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import psycopg2
from psycopg2 import sql
from flask import request

from AI import storage_state
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
    "lt_generators": "lt_generators_prompt.txt",
    "application_logs": "application_logs_prompt.txt",
    "judge": "judge_prompt.txt",
    "critic": "critic_prompt.txt",
}
LOCKED_PROMPT_DOMAINS = {"judge", "critic", "_context_guide", "_response_format", "verify"}
# Sections the settings UI may override per project area (and per service for queries/sla).
AREA_OVERRIDABLE_SECTIONS = frozenset({"llm", "domain_sources", "logs_source", "default_params", "queries", "sla"})
# Flat top-level keys of the legacy template-based Confluence flow, exposed as one config section.
CONFLUENCE_TEMPLATE_KEYS = ("url_basic", "space_conf", "user", "password", "grafana_base_url", "grafana_login", "grafana_pass", "loki_url")


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


# Serialises reads and writes of the runtime JSON files within the process.
_RUNTIME_FILES_LOCK = threading.RLock()

# A file read less than this long after its mtime may change again within the same timestamp tick
# and keep its size (coarse mtimes on FAT, network shares and some bind mounts, quick rewrites in
# tests). Such a cache entry is checked byte by byte until the file is older than the window.
_RACY_WINDOW_NS = 2_000_000_000


class RuntimeConfigError(ValueError):
    """A runtime configuration file exists but cannot be read; it is left untouched."""


class _CachedJson:
    """Parsed runtime JSON file with the stat signature and the bytes it was read from."""

    __slots__ = ("signature", "raw", "data", "verified_ns")

    def __init__(self, signature: tuple, raw: bytes, data: dict, verified_ns: int) -> None:
        self.signature = signature
        self.raw = raw
        self.data = data
        self.verified_ns = verified_ns

    def trusted(self, signature: tuple) -> bool:
        """Same signature as when read, and the file had last changed well before that read."""
        return self.signature == signature and signature[0] < self.verified_ns - _RACY_WINDOW_NS


# os.fspath(path) -> entry. Entries are replaced, never changed, so readers need no lock.
_JSON_CACHE: dict[str, _CachedJson] = {}


def _file_signature(st: os.stat_result) -> tuple:
    # st_mtime_ns comes first: _CachedJson.trusted reads it from there.
    return (st.st_mtime_ns, st.st_size, st.st_ino, st.st_dev, st.st_ctime_ns)


def _decode_runtime_json(path: Path, raw: bytes) -> dict:
    """Contents of a runtime JSON file already read as bytes; same rules as _read_json_file."""
    try:
        text = raw.decode("utf-8-sig")
        data = json.loads(text) if text.strip() else {}
    except ValueError as exc:
        raise RuntimeConfigError(f"{path.name}: файл не читается или содержит некорректный JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise RuntimeConfigError(f"{path.name}: ожидается JSON-объект")
    return data


def _refresh_cached_json(path: Path, force: bool = False) -> _CachedJson | None:
    """Reads ``path`` under the lock and stores its cache entry; None when the file is missing.

    When the bytes did not change, the new entry keeps the parsed object of the old one, so caches
    derived from it (_active_metrics_config) stay valid. ``force`` re-reads even a trusted entry.

    Raises:
        RuntimeConfigError: the file cannot be read or parsed. Its entry is dropped, so the next
            call reads the file again.
    """
    key = os.fspath(path)
    with _RUNTIME_FILES_LOCK:
        started = time.time_ns()
        entry = None
        try:
            signature = _file_signature(os.stat(key))
            entry = _JSON_CACHE.get(key)
            if not force and entry is not None and entry.trusted(signature):
                return entry  # another thread refreshed it while this one waited for the lock
            raw = path.read_bytes()
        except OSError as exc:
            _JSON_CACHE.pop(key, None)
            if not path.exists():
                return None
            raise RuntimeConfigError(f"{path.name}: файл не читается или содержит некорректный JSON ({exc})") from exc
        if entry is not None and entry.raw == raw:
            data = entry.data
        else:
            try:
                data = _decode_runtime_json(path, raw)
            except RuntimeConfigError:
                _JSON_CACHE.pop(key, None)
                raise
        fresh = _CachedJson(signature, raw, data, started)
        _JSON_CACHE[key] = fresh
        return fresh


def _cached_entry(path: Path) -> _CachedJson | None:
    """Cache entry of a runtime JSON file, None when it is missing; one ``os.stat`` while unchanged."""
    key = os.fspath(path)
    try:
        signature = _file_signature(os.stat(key))
    except FileNotFoundError:
        _JSON_CACHE.pop(key, None)
        return None
    except OSError:
        return _refresh_cached_json(path)
    entry = _JSON_CACHE.get(key)
    if entry is not None and entry.trusted(signature):
        return entry
    return _refresh_cached_json(path)


def _cached_json(path: Path) -> dict:
    """Parsed runtime JSON file, ``{}`` when missing or empty; read and parsed again only after a change.

    The dict is shared by every caller in the process: never modify it or anything inside it, and
    ``copy.deepcopy`` a part before changing it. Read-modify-write paths take a private copy from
    _load_settings_runtime_data() / _load_metrics_runtime_data() instead. A change is noticed by
    the file's stat signature (mtime, size, inode, ctime), byte by byte while the mtime is too
    recent to trust, and at once after _write_json_atomic; other processes are seen by the stat.

    Raises:
        RuntimeConfigError: as _read_json_file; a broken file is not cached.
    """
    entry = _cached_entry(path)
    return entry.data if entry is not None else {}


def _read_json_file(path: Path) -> dict:
    """Contents of a runtime JSON file, ``{}`` when it is missing or empty.

    Raises:
        RuntimeConfigError: the file cannot be parsed. Treating it as empty would let the next save
            silently drop every stored setting, so callers fail instead.
    """
    with _RUNTIME_FILES_LOCK:
        if not path.exists():
            return {}
        try:
            text = path.read_text(encoding="utf-8-sig")
            data = json.loads(text) if text.strip() else {}
        except (OSError, ValueError) as exc:
            raise RuntimeConfigError(f"{path.name}: файл не читается или содержит некорректный JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise RuntimeConfigError(f"{path.name}: ожидается JSON-объект")
    return data


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Writes JSON so that readers never see a half-written file.

    The text goes to a temporary file in the same directory, is flushed to disk and renamed over the
    target. A single-file bind mount (docker-compose) cannot be replaced by rename; then the file is
    rewritten in place, still under the process-wide lock. The cache of _cached_json is refreshed
    from the written file before the lock is released, whatever its mtime says.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with _RUNTIME_FILES_LOCK:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(tmp_name, path)
                tmp_name = ""
            except OSError:
                with path.open("w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        try:
            _refresh_cached_json(path, force=True)
        except RuntimeConfigError:
            pass  # the entry is dropped; the next read reports the problem


def public_base_url() -> str:
    """Address of LoadLens for links on Confluence pages.

    ``confluence.app_url`` when set; otherwise the address of the current request, which follows the
    Host header and should only be relied on behind a proxy that sets it.
    """
    configured = str(((CONFIG.get("confluence") or {}).get("app_url")) or "").strip().rstrip("/")
    if configured.lower().startswith(("http://", "https://")):
        return configured
    return request.host_url.rstrip("/")


def _load_settings_runtime_data() -> dict:
    """Private copy of settings_runtime.json, read from disk: for read-modify-write paths."""
    return _read_json_file(CONFIG_RUNTIME_PATH)


def _save_settings_runtime_data(payload: dict) -> None:
    _write_json_atomic(CONFIG_RUNTIME_PATH, payload or {})


def _load_metrics_runtime_data() -> dict:
    """Private copy of the runtime metrics_config overrides, read from disk: for read-modify-write paths.

    An unreadable file is an error, not an empty config.
    """
    return _read_json_file(METRICS_RUNTIME_PATH)


def _save_metrics_runtime_data(payload: dict) -> None:
    _write_json_atomic(METRICS_RUNTIME_PATH, payload or {})


def _settings_runtime_view() -> dict:
    """Shared cached settings_runtime.json for readers; do not modify (see _cached_json)."""
    return _cached_json(CONFIG_RUNTIME_PATH)


def _metrics_runtime_view() -> dict:
    """Shared cached metrics_config_runtime.json for readers; do not modify (see _cached_json)."""
    return _cached_json(METRICS_RUNTIME_PATH)


def _per_area_config() -> dict:
    """``per_area`` of the runtime settings. Shared and cached: do not modify (see _cached_json)."""
    per_area = _settings_runtime_view().get("per_area")
    return per_area if isinstance(per_area, dict) else {}


def _area_entry(area_name: str | None) -> dict:
    """Runtime entry of an area. Shared and cached: do not modify."""
    if not area_name:
        return {}
    entry = _per_area_config().get(area_name)
    return entry if isinstance(entry, dict) else {}


def _services_map_for_area(area_name: str | None) -> dict:
    """Runtime services of an area, by id. Shared and cached: do not modify."""
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


def _merged_metrics_config(base: dict, override: dict | None) -> dict:
    """Deep copy of ``base`` with the runtime overrides applied (``__replace__`` swaps a whole entry)."""
    raw = copy.deepcopy(base)
    try:
        for key, value in (override or {}).items():
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
    return raw


# (METRICS_CONFIG, settings data, metrics data, result) of the last computed _active_metrics_config().
_METRICS_VIEW: tuple | None = None


def _active_metrics_config() -> dict:
    """metrics_config with the runtime overrides, grouped by area (see _normalize_metrics_config).

    Computed again only when metrics_config_runtime.json or settings_runtime.json (whose per_area the
    grouping follows) changes. The result is shared by every caller: treat it as read-only and
    ``copy.deepcopy`` whatever is going to be changed. An unreadable metrics file counts as missing.
    """
    global _METRICS_VIEW
    base = METRICS_CONFIG if isinstance(METRICS_CONFIG, dict) else {}
    settings_entry = _cached_entry(CONFIG_RUNTIME_PATH)
    try:
        metrics_entry = _cached_entry(METRICS_RUNTIME_PATH)
    except RuntimeConfigError:
        metrics_entry = None
    settings_data = settings_entry.data if settings_entry is not None else None
    metrics_data = metrics_entry.data if metrics_entry is not None else None
    view = _METRICS_VIEW
    if view is not None and view[0] is base and view[1] is settings_data and view[2] is metrics_data:
        return view[3]
    result = _normalize_metrics_config(_merged_metrics_config(base, metrics_data))
    _METRICS_VIEW = (base, settings_data, metrics_data, result)
    return result


def _clear_runtime_cache() -> None:
    """Forgets every cached runtime file and the cached _active_metrics_config() (used by tests)."""
    global _METRICS_VIEW
    with _RUNTIME_FILES_LOCK:
        _JSON_CACHE.clear()
        _METRICS_VIEW = None


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
    """metrics_config services of an area, by id. Part of _active_metrics_config(): do not modify."""
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
    """Domains available for analysis: query-based ones plus application_logs when OpenSearch is enabled."""
    config = cfg or CONFIG
    queries = config.get("queries") or {}
    preferred_order = ["jvm", "database", "kafka", "microservices", "hard_resources", "lt_framework", "lt_generators"]
    result = [d for d in preferred_order if d in queries]
    logs_cfg = config.get("logs_source") if isinstance(config.get("logs_source"), dict) else {}
    if logs_cfg.get("enabled"):
        result.append("application_logs")
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


def _ensure_area_runtime(area_name: str):
    runtime = _load_settings_runtime_data()
    if "per_area" not in runtime or not isinstance(runtime.get("per_area"), dict):
        runtime["per_area"] = {}
    if area_name not in runtime["per_area"] or not isinstance(runtime["per_area"].get(area_name), dict):
        runtime["per_area"][area_name] = {}
    return runtime, runtime["per_area"][area_name]


def _bootstrap_area_defaults(area_name: str | None) -> None:
    """Registers the area in runtime settings; sections it does not set inherit the global values."""
    if not area_name or isinstance(_per_area_config().get(area_name), dict):
        return
    runtime, _area = _ensure_area_runtime(area_name)
    _save_settings_runtime_data(runtime)


def _bootstrap_service_configs(area_name: str | None, service_id: str | None) -> None:
    if not area_name or not service_id:
        return
    _bootstrap_area_defaults(area_name)
    # The cached view answers the common case (already registered) without reading the file again.
    if not isinstance(_services_map_for_area(area_name).get(service_id), dict):
        runtime, area_entry = _ensure_area_runtime(area_name)
        if "services" not in area_entry or not isinstance(area_entry.get("services"), dict):
            area_entry["services"] = {}
        changed = False
        if service_id not in area_entry["services"] or not isinstance(area_entry["services"].get(service_id), dict):
            area_entry["services"][service_id] = {}
            changed = True
        # Area settings are not copied onto the service. A copied snapshot would hide
        # later edits of the area. queries, sla and prompts land here only when the
        # user saves that service in the settings UI.
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
    cached_area = _metrics_runtime_view().get(area_name)
    cached_services = cached_area.get("services") if isinstance(cached_area, dict) else None
    if isinstance(cached_services, dict) and service_id in cached_services:
        return
    base_metrics = _default_metrics_service_template()
    current = _load_metrics_runtime_data()
    if area_name not in current or not isinstance(current.get(area_name), dict):
        current[area_name] = {"services": {}}
    if "services" not in current[area_name] or not isinstance(current[area_name].get("services"), dict):
        current[area_name]["services"] = {}
    if service_id not in current[area_name]["services"]:
        current[area_name]["services"][service_id] = copy.deepcopy(base_metrics)
        _save_metrics_runtime_data(current)


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
    """Connection to storage.timescale through the breaker: a short timeout and fast failures while it is down."""
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    params = {
        "host": cfg.get("host"),
        "port": cfg.get("port"),
        "dbname": cfg.get("dbname"),
        "user": cfg.get("user"),
        "password": cfg.get("password"),
        "sslmode": cfg.get("sslmode", "prefer"),
    }
    return storage_state.connect(params, cfg)


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
    """Converts an ISO 8601 datetime string to unix milliseconds.

    Accepts values with an explicit offset (``2025-01-15T10:00:00+03:00``,
    ``...Z``) and the legacy offset-less form ``2025-01-15T10:00``, which is
    interpreted in the server's local time zone.

    Raises:
        ValueError: if the string is not a valid ISO 8601 datetime.
    """
    raw = (date_str or "").strip()
    if not raw:
        raise ValueError("Пустое значение времени")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    return int(dt.timestamp() * 1000)


def _run_tables(cfg: dict) -> tuple[str, tuple[str, ...]]:
    """Schema and every table that stores rows keyed by run_name.

    Deleting and renaming a run walk this tuple, which keeps ``run_index`` (the run list) in step with
    ``metrics``; a rename to a name that any of these tables already has is refused before the UPDATEs,
    so the run_index primary key cannot collide.
    """
    from AI.db_store import RUN_INDEX_TABLE

    schema = cfg.get("schema", "public")
    tables = (
        cfg.get("table", "metrics"),
        cfg.get("llm_table", "llm_reports"),
        cfg.get("engineer_table", "engineer_reports"),
        cfg.get("confluence_table", "confluence_publications"),
        cfg.get("llm_feedback_table", "llm_feedback"),
        RUN_INDEX_TABLE,
    )
    return schema, tables


def _delete_run_data(run_name: str) -> None:
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema, tables = _run_tables(cfg)
    conn = _ts_conn()
    with conn, conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute("SAVEPOINT run_delete")
                cur.execute(f"DELETE FROM {schema}.{table} WHERE run_name = %s", (run_name,))
                cur.execute("RELEASE SAVEPOINT run_delete")
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT run_delete")
                logger.exception("Не удалось удалить данные запуска '%s' из %s.%s", run_name, schema, table)
    conn.close()


RUN_NAME_MAX_LEN = 180
# Run names appear in URLs, file names and Confluence pages.
RUN_NAME_FORBIDDEN_CHARS = ("/", "\\", "<", ">", "\x00")


def run_name_problem(name: str) -> str | None:
    """Why ``name`` cannot name a run, or None when it can."""
    if len(name) > RUN_NAME_MAX_LEN:
        return f"Имя отчёта: не длиннее {RUN_NAME_MAX_LEN} символов"
    if any(ch in name for ch in RUN_NAME_FORBIDDEN_CHARS):
        return "Имя отчёта содержит недопустимые символы: / \\ < >"
    return None


def _rename_run_data(old_run_name: str, new_run_name: str) -> dict:
    old_name = str(old_run_name or "").strip()
    new_name = str(new_run_name or "").strip()
    if not old_name:
        raise ValueError("Текущее имя отчёта не задано")
    if not new_name:
        raise ValueError("Новое имя отчёта не задано")
    if len(new_name) > RUN_NAME_MAX_LEN:
        raise ValueError("Новое имя отчёта слишком длинное")
    if any(ch in new_name for ch in RUN_NAME_FORBIDDEN_CHARS):
        raise ValueError("Новое имя отчёта содержит недопустимые символы: / \\ < >")
    if old_name == new_name:
        return {"status": "ok", "renamed": 0, "run_name": new_name}

    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema, tables = _run_tables(cfg)
    conn = _ts_conn()
    renamed = 0

    def _run_name_exists(cur, run_name: str) -> bool:
        # Each probe is isolated in a savepoint so a missing table does not abort the transaction.
        found = False
        for table in tables:
            try:
                cur.execute("SAVEPOINT run_name_probe")
                cur.execute(
                    sql.SQL("SELECT 1 FROM {}.{} WHERE run_name = %s LIMIT 1").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    ),
                    (run_name,),
                )
                found = bool(cur.fetchone()) or found
                cur.execute("RELEASE SAVEPOINT run_name_probe")
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT run_name_probe")
        return found

    try:
        try:
            _ensure_schema_and_table(conn, cfg, schema, cfg.get("table", "metrics"))
            _ensure_llm_reports_table(conn, cfg)
            _ensure_engineer_reports_table(conn, cfg)
        except Exception:
            logger.exception("Не удалось подготовить таблицы для переименования отчёта")
        with conn, conn.cursor() as cur:
            if not _run_name_exists(cur, old_name):
                raise LookupError(f"Отчёт '{old_name}' не найден")
            if _run_name_exists(cur, new_name):
                raise FileExistsError(f"Отчёт '{new_name}' уже существует")
            for table in tables:
                try:
                    cur.execute("SAVEPOINT run_name_rename")
                    cur.execute(
                        sql.SQL("UPDATE {}.{} SET run_name = %s WHERE run_name = %s").format(
                            sql.Identifier(schema), sql.Identifier(table)
                        ),
                        (new_name, old_name),
                    )
                    renamed += int(cur.rowcount or 0)
                    cur.execute("RELEASE SAVEPOINT run_name_rename")
                except Exception:
                    cur.execute("ROLLBACK TO SAVEPOINT run_name_rename")
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

    metrics_rt = _load_metrics_runtime_data()
    if isinstance(metrics_rt.get(area_name), dict):
        services_entry = metrics_rt[area_name].get("services")
        if isinstance(services_entry, dict) and service_name in services_entry:
            del services_entry[service_name]
            _save_metrics_runtime_data(metrics_rt)

    from AI.db_store import RUN_INDEX_TABLE

    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    schema = cfg.get("schema", "public")
    metrics_table = cfg.get("table", "metrics")
    llm_table = cfg.get("llm_table", "llm_reports")
    engineer_table = cfg.get("engineer_table", "engineer_reports")
    conn = _ts_conn()
    with conn, conn.cursor() as cur:
        # run_index goes with metrics, or the run list keeps showing the deleted runs. Each table has its
        # own savepoint: a missing table must not abort the transaction and silently undo the other deletes.
        for table in (metrics_table, llm_table, engineer_table, RUN_INDEX_TABLE):
            try:
                cur.execute("SAVEPOINT service_delete")
                cur.execute(f"DELETE FROM {schema}.{table} WHERE service = %s", (service_name,))
                cur.execute("RELEASE SAVEPOINT service_delete")
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT service_delete")
                logger.warning("Не удалось удалить данные сервиса '%s' из %s.%s: %s", service_name, schema, table, exc)
    conn.close()


def _bootstrap_service_queries(area_name: str, service_name: str) -> None:
    """Ensures runtime config for service exists (backward compatibility helper)."""
    _bootstrap_service_configs(area_name, service_name)


__all__ = [
    "AREA_OVERRIDABLE_SECTIONS",
    "CONFIG_RUNTIME_PATH",
    "CONFLUENCE_TEMPLATE_KEYS",
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
    "_load_metrics_runtime_data",
    "_load_settings_runtime_data",
    "_metrics_runtime_view",
    "_metrics_service_entry",
    "_metrics_services_for_area",
    "_normalize_metrics_config",
    "_per_area_config",
    "_prompt_templates_for_scope",
    "_rename_run_data",
    "public_base_url",
    "RuntimeConfigError",
    "RUN_NAME_MAX_LEN",
    "run_name_problem",
    "_resolve_services_filter",
    "_resolve_services_for_area",
    "_save_metrics_runtime_data",
    "_save_settings_runtime_data",
    "_series_key_for",
    "_service_meta",
    "_service_disabled_domains",
    "_services_map_for_area",
    "_settings_runtime_view",
    "_ts_conn",
    "convert_to_timestamp",
    "_deep_merge_dicts",
    "_normalize_system_context",
]


