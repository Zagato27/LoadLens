"""Blueprint for configuration APIs."""

from __future__ import annotations

import copy
import json

from flask import Blueprint, jsonify, request

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
from loadlens_app.config_schema import (
    QueryConfigError,
    SlaConfigError,
    validate_appearance,
    validate_queries,
    validate_sla,
    warnings_for,
)
from loadlens_app.config_secrets import SecretPlaceholderError, mask_secrets, restore_secrets
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
    _load_settings_runtime_data,
    _normalize_system_context,
    _save_settings_runtime_data,
    _services_map_for_area,
)

config_bp = Blueprint("config_api", __name__)

# Section name under which the flat legacy Confluence/Grafana/Loki keys are exposed to the UI.
CONFLUENCE_TEMPLATE_SECTION = "confluence_template"


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


def _confluence_template_node() -> dict:
    return {key: CONFIG.get(key) for key in CONFLUENCE_TEMPLATE_KEYS}


def _load_metrics_runtime() -> dict:
    try:
        if METRICS_RUNTIME_PATH.exists():
            with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_metrics_runtime(payload: dict) -> None:
    with METRICS_RUNTIME_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
    if section in {"queries", "sla"} and area and service:
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
        for sid in services_meta.keys():
            meta = services_map.get(sid) or {}
            svc_queries = meta.get("queries") if isinstance(meta, dict) else {}
            queries_map[sid] = svc_queries if isinstance(svc_queries, dict) else {}
            svc_sla = meta.get("sla") if isinstance(meta, dict) else {}
            service_sla[sid] = svc_sla if isinstance(svc_sla, dict) else {}
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
        out["metrics_config_map"] = metrics_config_map
        return jsonify(mask_secrets(out))
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


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
            if area:
                _update_area_section(area, "system_context", normalized)
            else:
                CONFIG["system_context"] = normalized
                existing = _load_settings_runtime_data()
                existing["system_context"] = normalized
                _save_settings_runtime_data(existing)
            return jsonify({"status": "ok", "warnings": warnings})

        if section in {"queries", "sla"} and area and service:
            _update_area_service_section(area, service, section, payload)
            return jsonify({"status": "ok", "scope": "service", "service": service, "warnings": warnings})

        if section in AREA_OVERRIDABLE_SECTIONS and area:
            _update_area_section(area, section, payload)
            return jsonify({"status": "ok", "scope": "area", "area": area, "warnings": warnings})

        parts = section.split(".")
        _set_nested(CONFIG, parts, payload)
        existing = _load_settings_runtime_data()
        _set_nested(existing, parts, payload)
        _save_settings_runtime_data(existing)
        return jsonify({"status": "ok", "warnings": warnings})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


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
        return _check_catalog_source(restored, str(data.get("source") or "").strip())
    if section == "domain_sources":
        return _check_domain_binding(restored, str(data.get("domain") or "").strip(), area)
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
