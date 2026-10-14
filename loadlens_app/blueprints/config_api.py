"""Blueprint for configuration APIs."""

from __future__ import annotations

import copy
import json
from urllib.parse import urlsplit

from flask import Blueprint, g, jsonify, request

from AI.data_sources import (
    DEFAULT_BINDING,
    DataSourceConfigError,
    binding_errors,
    domain_query_language,
    resolve_domain_source,
    validate_data_sources,
    validate_domain_sources,
)
from settings import CONFIG

from loadlens_app.appearance import AppearanceError
from loadlens_app.errors import internal_error
from loadlens_app.config_schema import (
    QueryConfigError,
    SlaConfigError,
    validate_appearance,
    validate_queries,
    validate_sla,
    warnings_for,
)
from AI.providers import without_env_keys
from AI.query_packs import prune_queries
from loadlens_app.config_secrets import SecretPlaceholderError, destination_changed, mask_secrets, restore_secrets
from loadlens_app.connection_checks import (
    CHECKABLE_SECTIONS,
    check_data_source,
    check_resolved_source,
    list_grafana_datasources,
    run_check,
)
from loadlens_app.core import (
    AREA_OVERRIDABLE_SECTIONS,
    CONFLUENCE_TEMPLATE_KEYS,
    METRICS_RUNTIME_PATH,
    _active_metrics_config,
    _active_project_area,
    _available_domain_keys,
    _bootstrap_service_configs,
    _deep_merge_dicts,
    _list_project_areas,
    _load_metrics_runtime_data,
    _load_settings_runtime_data,
    _normalize_system_context,
    _save_metrics_runtime_data,
    _save_settings_runtime_data,
    _services_map_for_area,
)

config_bp = Blueprint("config_api", __name__)

# Section name under which the flat legacy Confluence/Grafana/Loki keys are exposed to the UI.
CONFLUENCE_TEMPLATE_SECTION = "confluence_template"
# Sections a service inside an area can override.
SERVICE_SCOPED_SECTIONS = frozenset({"queries", "sla", "system_context"})


def _config_node(section: str) -> dict:
    """Current value of a dotted section path in CONFIG (empty dict if missing)."""
    node = CONFIG
    for part in section.split("."):
        if not isinstance(node, dict):
            return {}
        node = node.get(part)
    return node if isinstance(node, dict) else {}


def _set_nested(target: dict, parts: list[str], value: dict) -> None:
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            target[part] = value
        else:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]


# Fields whose values name the host a section talks to. Only host and port reach the audit log.
_ADDRESS_KEYS = ("url", "base_url", "api_base_url", "url_basic", "grafana_base_url", "loki_url")
_AUDIT_TARGETS_LIMIT = 10


def _host_of(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname or ""
        return f"{host}:{parts.port}" if host and parts.port else host
    except ValueError:
        return ""


def _addresses(tree: object) -> list[str]:
    """``host[:port]`` of every address in a configuration payload: no paths, users or passwords."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            host = node.get("host")
            if isinstance(host, str) and host.strip():
                port = node.get("port")
                found.add(f"{host.strip()}:{port}" if port not in (None, "") else host.strip())
            for key, value in node.items():
                if key in _ADDRESS_KEYS and isinstance(value, str):
                    found.add(_host_of(value))
                elif key == "proxies" and isinstance(value, dict):
                    found.update(_host_of(item) for item in value.values() if isinstance(item, str))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(tree)
    return sorted(item for item in found if item)[:_AUDIT_TARGETS_LIMIT]


def _audit_addresses(tree: object) -> None:
    """Records where this request sends stored secrets, so a misuse shows up in the audit log."""
    addresses = _addresses(tree)
    if addresses:
        g.audit_extra = {"targets": addresses}


def _confluence_template_node() -> dict:
    return {key: CONFIG.get(key) for key in CONFLUENCE_TEMPLATE_KEYS}


def _load_metrics_runtime() -> dict:
    return _load_metrics_runtime_data()


def _save_metrics_runtime(payload: dict) -> None:
    _save_metrics_runtime_data(payload)


def _area_section(area: str, section_name: str) -> dict:
    """Effective value of a section for the area: runtime per_area override, else the global value."""
    runtime_per_area = _load_settings_runtime_data().get("per_area")
    if not isinstance(runtime_per_area, dict):
        runtime_per_area = {}
    per_area_entry = ((runtime_per_area.get(area, {}) or {}).get(section_name) if area else None)
    if not isinstance(per_area_entry, dict):
        per_area_entry = (((CONFIG.get("per_area", {}) or {}).get(area, {}) or {}).get(section_name) if area else None)
    if isinstance(per_area_entry, dict):
        return copy.deepcopy(per_area_entry)
    return copy.deepcopy(CONFIG.get(section_name, {}) or {})


def _service_entry(area: str, service: str) -> dict:
    meta = _services_map_for_area(area).get(service) if area and service else None
    return meta if isinstance(meta, dict) else {}


def _current_scope_value(section: str, area: str, service: str) -> dict:
    """Stored value the mask sentinel is restored from for the given section and scope."""
    if section == CONFLUENCE_TEMPLATE_SECTION:
        return _confluence_template_node()
    if section in SERVICE_SCOPED_SECTIONS and area and service:
        scoped = _service_entry(area, service).get(section)
        return scoped if isinstance(scoped, dict) else _area_section(area, section)
    if section in AREA_OVERRIDABLE_SECTIONS and area:
        return _area_section(area, section)
    return _config_node(section)


@config_bp.route("/config", methods=["GET"])
def get_config():
    """Возвращает текущий конфиг (с учётом области и сервисов) для UI; секреты замаскированы."""
    try:
        area = (request.args.get("area") or "")
        areas_meta = _list_project_areas()
        areas = [a["id"] for a in areas_meta]
        if not area:
            cookie_area = _active_project_area() or ""
            if cookie_area in areas:
                area = cookie_area
        active_area = area if area in areas else ""

        active_metrics = _active_metrics_config() or {}
        runtime_metrics = _load_metrics_runtime()
        area_runtime_entry = {}
        if area and isinstance(runtime_metrics.get(area), dict):
            area_runtime_entry = copy.deepcopy(runtime_metrics.get(area) or {})
            area_runtime_entry.pop("__replace__", None)
        area_metrics_cfg = area_runtime_entry if area_runtime_entry else (active_metrics.get(area, {}) if area else {})
        if not isinstance(area_metrics_cfg, dict):
            area_metrics_cfg = {}
        if "services" not in area_metrics_cfg or not isinstance(area_metrics_cfg.get("services"), dict):
            area_metrics_cfg["services"] = {}
        runtime_services_cfg = area_runtime_entry.get("services", {}) if isinstance(area_runtime_entry, dict) else {}

        out = {
            "areas": areas,
            "areas_meta": areas_meta,
            "active_area": active_area,
            "llm": _area_section(area, "llm"),
            "data_sources": copy.deepcopy(CONFIG.get("data_sources") or {}),
            "domain_sources": _area_section(area, "domain_sources"),
            "logs_source": _area_section(area, "logs_source"),
            "default_params": _area_section(area, "default_params"),
            "sla": _area_section(area, "sla"),
            "system_context": _normalize_system_context(_area_section(area, "system_context")),
            "storage": {"timescale": copy.deepcopy((CONFIG.get("storage", {}) or {}).get("timescale", {}))},
            "queries": _area_section(area, "queries"),
            "confluence": copy.deepcopy(CONFIG.get("confluence", {}) or {}),
            "appearance": copy.deepcopy(CONFIG.get("appearance", {}) or {}),
            CONFLUENCE_TEMPLATE_SECTION: _confluence_template_node(),
            "metrics_config": area_metrics_cfg,
        }
        services_meta = {}
        area_metrics_services = area_metrics_cfg.get("services", {})
        service_ids: set[str] = set()
        services_map_initial = _services_map_for_area(active_area) if active_area else {}
        service_ids.update(services_map_initial.keys())
        service_ids.update(area_metrics_services.keys())
        if active_area:
            for sid in service_ids:
                _bootstrap_service_configs(active_area, sid)
        services_map = _services_map_for_area(active_area) if active_area else {}
        for sid, meta in services_map.items():
            if not isinstance(meta, dict):
                meta = {}
            services_meta[sid] = {
                "title": (meta.get("title") if isinstance(meta.get("title"), str) else "") or sid,
                "disabled_domains": [d for d in (meta.get("disabled_domains") or []) if isinstance(d, str)],
            }
        for sid in service_ids:
            services_meta.setdefault(sid, {"title": sid, "disabled_domains": []})
        queries_map = {"": out["queries"]}
        service_sla = {}
        service_context = {}
        for sid in services_meta.keys():
            meta = services_map.get(sid) or {}
            svc_queries = meta.get("queries") if isinstance(meta, dict) else {}
            queries_map[sid] = svc_queries if isinstance(svc_queries, dict) else {}
            svc_sla = meta.get("sla") if isinstance(meta, dict) else {}
            service_sla[sid] = svc_sla if isinstance(svc_sla, dict) else {}
            svc_context = meta.get("system_context") if isinstance(meta, dict) else None
            if isinstance(svc_context, dict):
                # Only services with their own context are listed; the others use the area value.
                service_context[sid] = _normalize_system_context(svc_context)
        metrics_config_map = {"": area_metrics_cfg}
        for sid, cfg in area_metrics_services.items():
            svc_runtime = {}
            if isinstance(runtime_services_cfg, dict) and isinstance(runtime_services_cfg.get(sid), dict):
                svc_runtime = copy.deepcopy(runtime_services_cfg.get(sid))
            metrics_config_map[sid] = svc_runtime if svc_runtime else (cfg if isinstance(cfg, dict) else {})
        out["services_meta"] = services_meta
        out["services"] = [{"id": sid, **meta} for sid, meta in services_meta.items()]
        out["domain_list"] = _available_domain_keys(out)
        out["queries_map"] = queries_map
        out["service_sla"] = service_sla
        out["service_system_context"] = service_context
        out["metrics_config_map"] = metrics_config_map
        return jsonify(mask_secrets(out))
    except Exception as e:  # pragma: no cover
        return internal_error(e)


def _update_area_service_section(area: str, service: str, section: str, payload: dict) -> None:
    runtime = _load_settings_runtime_data()
    if "per_area" not in runtime or not isinstance(runtime.get("per_area"), dict):
        runtime["per_area"] = {}
    if area not in runtime["per_area"] or not isinstance(runtime["per_area"].get(area), dict):
        runtime["per_area"][area] = {}
    area_entry = runtime["per_area"][area]
    if "services" not in area_entry or not isinstance(area_entry.get("services"), dict):
        area_entry["services"] = {}
    if service not in area_entry["services"] or not isinstance(area_entry["services"].get(service), dict):
        area_entry["services"][service] = {}
    area_entry["services"][service][section] = payload
    _save_settings_runtime_data(runtime)


def _remove_area_service_section(area: str, service: str, section: str) -> bool:
    """Drops a service override so the area value applies again; returns whether it existed."""
    runtime = _load_settings_runtime_data()
    services = ((runtime.get("per_area") or {}).get(area) or {}).get("services") or {}
    entry = services.get(service) if isinstance(services, dict) else None
    if not isinstance(entry, dict) or section not in entry:
        return False
    entry.pop(section, None)
    _save_settings_runtime_data(runtime)
    return True


def _store_service_context(area: str, service: str, normalized: dict) -> dict:
    """Context of one service. A value equal to the area's is not stored, so later area edits still apply."""
    area_context = _normalize_system_context(_area_section(area, "system_context"))
    if normalized == area_context:
        _remove_area_service_section(area, service, "system_context")
        return {"scope": "service", "service": service, "inherited": True}
    _update_area_service_section(area, service, "system_context", normalized)
    return {"scope": "service", "service": service}


def _update_area_section(area: str, section: str, payload: dict) -> None:
    if "per_area" not in CONFIG or not isinstance(CONFIG.get("per_area"), dict):
        CONFIG["per_area"] = {}
    if area not in CONFIG["per_area"] or not isinstance(CONFIG["per_area"].get(area), dict):
        CONFIG["per_area"][area] = {}
    CONFIG["per_area"][area][section] = payload
    existing = _load_settings_runtime_data()
    if "per_area" not in existing or not isinstance(existing.get("per_area"), dict):
        existing["per_area"] = {}
    if area not in existing["per_area"] or not isinstance(existing["per_area"].get(area), dict):
        existing["per_area"][area] = {}
    existing["per_area"][area][section] = payload
    _save_settings_runtime_data(existing)


def _update_metrics_config(area: str, service: str, payload: dict) -> tuple[dict, int]:
    payload_copy = copy.deepcopy(payload)
    current = _load_metrics_runtime()
    if service:
        if area not in current or not isinstance(current.get(area), dict):
            current[area] = {"services": {}}
        if "services" not in current[area] or not isinstance(current[area].get("services"), dict):
            current[area]["services"] = {}
        current[area]["services"][service] = payload_copy
    else:
        if not isinstance(payload.get("services"), dict):
            return {"error": "metrics_config должен содержать ключ 'services' с объектом сервисов"}, 400
        current[area] = payload_copy
        current[area]["__replace__"] = True
    _save_metrics_runtime(current)
    return {"status": "ok"}, 200


@config_bp.route("/config", methods=["POST"])
def update_config():
    """Применяет изменения конфигурации (глобально, по области или по сервису).

    Поля с секретами могут приходить как маска ``***`` и восстанавливаются из сохранённых
    значений. В ответе ``warnings`` — ключи, которых нет в базовом settings.py (вероятные опечатки).
    """
    data = request.get_json(silent=True) or {}
    section = str(data.get("section") or "").strip()
    payload = data.get("data")
    area = (data.get("area") or "").strip()
    service = (data.get("service") or "").strip()
    if not section or not isinstance(payload, dict):
        return jsonify({"error": "section и data обязательны"}), 400
    if section.split(".")[0] == "auth":
        # Authentication settings (and the session key) must not be writable from the browser.
        return jsonify({"error": "Раздел auth не меняется через API: задайте его в settings.py или переменных окружения"}), 403
    if section != "metrics_config":
        try:
            payload = restore_secrets(payload, _current_scope_value(section, area, service))
        except SecretPlaceholderError as e:
            return jsonify({"error": str(e)}), 400
    if section == "queries":
        try:
            validate_queries(payload)
        except QueryConfigError as exc:
            return jsonify({"error": str(exc)}), 400
        # Roles of deleted rows and packs with no rows left are dropped, not reported as errors.
        payload = prune_queries(copy.deepcopy(payload))
    if section == "sla":
        try:
            validate_sla(payload)
        except SlaConfigError as exc:
            return jsonify({"error": str(exc)}), 400
    if section == "appearance":
        try:
            validate_appearance(payload)
        except AppearanceError as exc:
            return jsonify({"error": str(exc)}), 400
    if section == "data_sources":
        rejected = _reject_data_sources(payload)
        if rejected is not None:
            return rejected
    if section == "domain_sources":
        rejected = _reject_domain_sources(payload)
        if rejected is not None:
            return rejected
    warnings = warnings_for(section, payload)
    _audit_addresses(payload)
    try:
        if section == CONFLUENCE_TEMPLATE_SECTION:
            existing = _load_settings_runtime_data()
            for key, value in payload.items():
                if key in CONFLUENCE_TEMPLATE_KEYS:
                    CONFIG[key] = value
                    existing[key] = value
            _save_settings_runtime_data(existing)
            return jsonify({"status": "ok", "warnings": warnings})

        if section == "metrics_config":
            if not area:
                return jsonify({"error": "area обязательна для metrics_config"}), 400
            body, status = _update_metrics_config(area, service, payload)
            return jsonify(body), status

        if section == "system_context":
            normalized = _normalize_system_context(payload)
            if area and service:
                scope = _store_service_context(area, service, normalized)
                return jsonify({"status": "ok", **scope, "warnings": warnings})
            if area:
                _update_area_section(area, "system_context", normalized)
            else:
                CONFIG["system_context"] = normalized
                existing = _load_settings_runtime_data()
                existing["system_context"] = normalized
                _save_settings_runtime_data(existing)
            return jsonify({"status": "ok", "warnings": warnings})

        scope = store_section(section, payload, area, service)
        return jsonify({"status": "ok", **scope, "warnings": warnings})
    except Exception as e:  # pragma: no cover
        return internal_error(e)


def store_section(section: str, payload: dict, area: str, service: str) -> dict:
    """Writes a section at its scope: service (queries, sla), area override, or global.

    Returns the scope fields of the response. Shared by POST /config and the query packs.
    """
    if section in SERVICE_SCOPED_SECTIONS and area and service:
        _update_area_service_section(area, service, section, payload)
        return {"scope": "service", "service": service}
    if section in AREA_OVERRIDABLE_SECTIONS and area:
        _update_area_section(area, section, payload)
        return {"scope": "area", "area": area}
    parts = section.split(".")
    _set_nested(CONFIG, parts, payload)
    existing = _load_settings_runtime_data()
    _set_nested(existing, parts, payload)
    _save_settings_runtime_data(existing)
    return {}


def scope_view(section: str, area: str, service: str) -> dict:
    """Value the settings page shows for the scope: the area value merged with the service override."""
    base = _area_section(area, section)
    if section in SERVICE_SCOPED_SECTIONS and area and service:
        own = _service_entry(area, service).get(section)
        if isinstance(own, dict):
            return _deep_merge_dicts(base, copy.deepcopy(own))
    return base


def has_own_section(section: str, area: str, service: str) -> bool:
    """Whether the scope stores its own value (otherwise it inherits the wider scope)."""
    if area and service:
        return isinstance(_service_entry(area, service).get(section), dict)
    if area:
        runtime_area = ((_load_settings_runtime_data().get("per_area") or {}).get(area) or {})
        config_area = ((CONFIG.get("per_area") or {}).get(area) or {}) if isinstance(CONFIG.get("per_area"), dict) else {}
        return isinstance(runtime_area.get(section), dict) or isinstance(config_area.get(section), dict)
    return True


def _stored_bindings() -> tuple[dict, dict[str, dict]]:
    """Saved domain bindings: global ones and each area that has its own."""
    runtime = _load_settings_runtime_data()
    runtime_bindings = runtime.get("domain_sources") if isinstance(runtime.get("domain_sources"), dict) else None
    global_bindings = runtime_bindings if runtime_bindings is not None else (CONFIG.get("domain_sources") or {})
    if not isinstance(global_bindings, dict):
        global_bindings = {}
    file_areas = runtime.get("per_area") if isinstance(runtime.get("per_area"), dict) else {}
    config_areas = CONFIG.get("per_area") if isinstance(CONFIG.get("per_area"), dict) else {}
    areas: dict[str, dict] = {}
    for name in set(file_areas) | set(config_areas):
        file_entry = file_areas.get(name) if isinstance(file_areas.get(name), dict) else {}
        if isinstance(file_entry.get("domain_sources"), dict):
            areas[str(name)] = file_entry["domain_sources"]
            continue
        config_entry = config_areas.get(name) if isinstance(config_areas.get(name), dict) else {}
        if isinstance(config_entry.get("domain_sources"), dict):
            areas[str(name)] = config_entry["domain_sources"]
    return global_bindings, areas


def _reject_data_sources(payload: dict):
    try:
        validate_data_sources(payload)
    except DataSourceConfigError as exc:
        return jsonify({"error": str(exc)}), 400
    global_bindings, area_bindings = _stored_bindings()
    errors = binding_errors(payload, global_bindings, area_bindings)
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400
    return None


def _reject_domain_sources(payload: dict):
    try:
        validate_domain_sources(payload, _config_node("data_sources"))
    except DataSourceConfigError as exc:
        return jsonify({"error": str(exc)}), 400
    return None


def _restored_section(section: str, payload: object, area: str, service: str) -> dict | tuple:
    current = _current_scope_value(section, area, service)
    if not isinstance(payload, dict):
        payload = current
    try:
        return restore_secrets(payload, current)
    except SecretPlaceholderError as exc:
        return jsonify({"error": str(exc)}), 400


def _check_catalog_source(payload: dict, source_id: str):
    entry = payload.get(source_id)
    if not source_id or not isinstance(entry, dict):
        return jsonify({"error": f"Источник «{source_id or '—'}» не найден"}), 400
    return jsonify(check_data_source(entry).to_dict()), 200


def _check_domain_binding(payload: dict, domain: str, area: str):
    if not domain:
        return jsonify({"error": "Укажите домен"}), 400
    try:
        resolved = resolve_domain_source({"data_sources": _config_node("data_sources"), "domain_sources": payload}, domain)
    except DataSourceConfigError as exc:
        return jsonify({"error": str(exc)}), 400
    queries = _area_section(area, "queries") if area else (CONFIG.get("queries") or {})
    block = queries.get(domain) if isinstance(queries, dict) and domain != DEFAULT_BINDING else None
    qcfg = block if isinstance(block, dict) else {}
    try:
        language = domain_query_language(resolved, qcfg) if qcfg else None
    except DataSourceConfigError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(check_resolved_source(resolved, language).to_dict()), 200


@config_bp.route("/config/test_connection", methods=["POST"])
def test_connection():
    """Проверяет соединение по данным раздела из формы (ещё не сохранённым)."""
    data = request.get_json(silent=True) or {}
    section = str(data.get("section") or "").strip()
    payload = data.get("data")
    area = (data.get("area") or "").strip()
    service = (data.get("service") or "").strip()
    if section not in CHECKABLE_SECTIONS:
        return jsonify({"error": f"Проверка недоступна для раздела «{section}». Допустимо: {', '.join(CHECKABLE_SECTIONS)}"}), 400
    restored = _restored_section(section, payload, area, service)
    if isinstance(restored, tuple):
        return restored
    if section == "data_sources":
        source_id = str(data.get("source") or "").strip()
        _audit_addresses(restored.get(source_id))
        return _check_catalog_source(restored, source_id)
    _audit_addresses(restored)
    if section == "domain_sources":
        return _check_domain_binding(restored, str(data.get("domain") or "").strip(), area)
    if section == "llm" and destination_changed(restored, _current_scope_value(section, area, service)):
        # Server-wide provider keys belong to the saved address, not to one typed into the form.
        with without_env_keys():
            result = run_check(section, restored)
    else:
        result = run_check(section, restored)
    return jsonify(result.to_dict()), 200


@config_bp.route("/config/grafana_datasources", methods=["GET"])
def grafana_datasources():
    """Datasources of a saved Grafana source, for choosing one per domain binding."""
    source_id = (request.args.get("source") or "").strip()
    entry = _config_node("data_sources").get(source_id)
    if not isinstance(entry, dict) or entry.get("type") != "grafana_proxy":
        return jsonify({"error": f"Источник Grafana «{source_id or '—'}» не найден в сохранённом каталоге"}), 404
    grafana = entry.get("grafana") if isinstance(entry.get("grafana"), dict) else {}
    return jsonify(list_grafana_datasources(grafana).to_dict()), 200


@config_bp.route("/config/llm_models", methods=["POST"])
def llm_models():
    """Lists models for the LLM form, including values that are not saved yet."""
    from loadlens_app.connection_checks import list_llm_models

    data = request.get_json(silent=True) or {}
    payload = data.get("data")
    area = (data.get("area") or "").strip()
    service = (data.get("service") or "").strip()
    current = _current_scope_value("llm", area, service)
    if not isinstance(payload, dict):
        payload = current
    try:
        payload = restore_secrets(payload, current)
    except SecretPlaceholderError as e:
        return jsonify({"error": str(e)}), 400
    _audit_addresses(payload)
    return jsonify(list_llm_models(payload).to_dict()), 200


@config_bp.route("/config/query_preview", methods=["POST"])
def query_preview():
    """Runs one unsaved metrics query for the last 15 minutes and reports whether it returned series."""
    from loadlens_app.query_preview import parse_preview_ts, preview_metric_query
    from update_page import _effective_config_for_scope

    data = request.get_json(silent=True) or {}
    area = (data.get("area") or _active_project_area() or "").strip()
    service = (data.get("service") or "").strip()
    label_keys = data.get("label_keys") if isinstance(data.get("label_keys"), list) else []
    cfg = _effective_config_for_scope(area, service or None)
    try:
        start_ts = parse_preview_ts(data.get("start"))
        end_ts = parse_preview_ts(data.get("end"))
    except ValueError:
        return jsonify({"ok": False, "message": "Некорректное окно проверки"}), 400
    result = preview_metric_query(
        cfg,
        domain=str(data.get("domain") or ""),
        lang=str(data.get("lang") or ""),
        query=str(data.get("query") or ""),
        label_keys=[str(item) for item in label_keys],
        start_ts=start_ts,
        end_ts=end_ts,
    )
    rejected = result.message in {
        "Запрос пустой",
    } or result.message.startswith(("Неизвестный домен", "Укажите язык", "Этот источник", "Источник типа"))
    return jsonify(result.to_dict()), (400 if rejected else 200)


# ---- system context from a document ------------------------------------------------------------------

# Room for the multipart envelope and the form fields around the file.
_IMPORT_BODY_SLACK = 256 * 1024


def _import_owner() -> str:
    from loadlens_app.auth.web import current_auth

    user = current_auth().user
    return user.username if user is not None else "local"


def _scope_context(area: str, service: str) -> dict:
    """The context the form shows for the scope: the service's own one, else the area's (or the global one)."""
    own = _service_entry(area, service).get("system_context") if area and service else None
    if isinstance(own, dict):
        return _normalize_system_context(own)
    return _normalize_system_context(_area_section(area, "system_context") if area else _config_node("system_context"))


def _form_context(value: object) -> dict | None:
    """The context as the form shows it, unsaved edits included (an object, or JSON text in multipart)."""
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _structured_for_scope(area: str) -> bool:
    """GigaChat function calling reads the global settings, so it is used only when the area keeps them."""
    from AI.providers import structured_output_enabled

    if not structured_output_enabled():
        return False
    return not area or _area_section(area, "llm") == (CONFIG.get("llm") or {})


@config_bp.route("/config/system_context/import", methods=["POST"])
def import_system_context():
    """Fills the system context from a document or pasted text with one model call.

    Nothing is saved: the answer is the context for the form and the list of changed fields.
    Multipart: ``file``, ``area``, ``service``, ``mode``. JSON: ``text``, ``area``, ``service``, ``mode``.
    """
    from werkzeug.exceptions import RequestEntityTooLarge

    from loadlens_app.context_import import (
        MAX_FILE_BYTES,
        MODES,
        ContextImportError,
        ask_model,
        extract_text,
        llm_for_scope,
        merge_context,
        text_from_paste,
    )
    from loadlens_app.limits import LimitError, consume_llm, refund_llm

    too_large = jsonify({"error": f"Файл больше {MAX_FILE_BYTES // (1024 * 1024)} МБ", "code": "too_large"}), 413
    # The body size must be known before Werkzeug parses it; browsers always send Content-Length.
    if request.content_length is None:
        return jsonify({"error": "Нужен заголовок Content-Length", "code": "length_required"}), 411
    if request.content_length > MAX_FILE_BYTES + _IMPORT_BODY_SLACK:
        return too_large
    try:
        if request.is_json:
            body = request.get_json(silent=True) or {}
            upload = None
        else:
            body = request.form
            upload = request.files.get("file")
    except RequestEntityTooLarge:
        return too_large
    area = str(body.get("area") or "").strip()
    service = str(body.get("service") or "").strip()
    mode = str(body.get("mode") or "fill").strip()
    try:
        if mode not in MODES:
            raise ContextImportError(f"mode: допустимо {', '.join(MODES)}")
        if service and not area:
            raise ContextImportError("Для сервиса укажите область")
        if area and area not in {item["id"] for item in _list_project_areas()}:
            raise ContextImportError(f"Область «{area}» не найдена", 404, "not_found")
        if service and service not in _services_map_for_area(area):
            raise ContextImportError(f"Сервис «{service}» не найден в области «{area}»", 404, "not_found")
        # The model is checked first: without it there is no point in reading a page or a file.
        llm_section = _area_section(area, "llm") if area else _config_node("llm")
        provider, _settings = llm_for_scope(llm_section)
        page = None
        confluence_url = str(body.get("confluence_url") or "").strip()
        if upload is not None and upload.filename:
            name = upload.filename.replace("\\", "/").rsplit("/", 1)[-1][:120]
            data = upload.read(MAX_FILE_BYTES + 1)
            extracted = extract_text(name, data)
            size = len(data)
        elif confluence_url:
            from loadlens_app.confluence_page import read_page

            page = read_page(confluence_url)
            extracted = page.extracted
            name, size = (page.title or f"страница {page.page_id}")[:120], len(extracted.text.encode("utf-8"))
        else:
            text = body.get("text")
            extracted = text_from_paste(text)
            name, size = "", len(text.encode("utf-8")) if isinstance(text, str) else 0
        current = _form_context(body.get("current")) or _scope_context(area, service)
        owner = _import_owner()
        consume_llm(owner, "context_import")
        try:
            draft = ask_model(extracted.text, llm_section, use_structured=_structured_for_scope(area))
        except ContextImportError:
            refund_llm(owner, "context_import")
            raise
        context, changes = merge_context(current, draft, mode)
    except (ContextImportError, LimitError) as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    source = extracted.source(name, size)
    g.audit_extra = {
        "area": area, "service": service, "mode": mode, "format": source["format"],
        "file": name, "bytes": size, "chars": source["chars"], "changes": len(changes),
    }
    if page is not None:
        source["page_id"] = page.page_id
        g.audit_extra.update({"page_id": page.page_id, "targets": [page.host]})
    return jsonify({
        "context": context,
        "changes": changes,
        "source": source,
        "warnings": extracted.warnings,
        "provider": provider,
        "mode": mode,
        "area": area,
        "service": service,
    })
