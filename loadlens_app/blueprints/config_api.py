"""Blueprint for configuration APIs."""

from __future__ import annotations

import json

import copy

from flask import Blueprint, jsonify, request

from settings import CONFIG

from loadlens_app.core import (
    CONFIG_RUNTIME_PATH,
    METRICS_RUNTIME_PATH,
    _available_domain_keys,
    _active_metrics_config,
    _active_project_area,
    _bootstrap_service_configs,
    _deep_merge_dicts,
    _list_project_areas,
    _load_settings_runtime_data,
    _metrics_service_entry,
    _save_settings_runtime_data,
    _services_map_for_area,
)

config_bp = Blueprint("config_api", __name__)


@config_bp.route("/config", methods=["GET"])
def get_config():
    """Возвращает текущий конфиг (с учётом области и сервисов) для UI."""
    try:
        area = (request.args.get("area") or "")
        areas_meta = _list_project_areas()
        areas = [a["id"] for a in areas_meta]
        if not area:
            cookie_area = _active_project_area() or ""
            if cookie_area in areas:
                area = cookie_area

        def merge_area_section(section_name: str) -> dict:
            base = CONFIG.get(section_name, {}) or {}
            per_area_entry = (
                ((CONFIG.get("per_area", {}) or {}).get(area, {}) or {}).get(section_name)
                if area else None
            )
            if isinstance(per_area_entry, dict):
                return copy.deepcopy(per_area_entry)
            return copy.deepcopy(base)

        active_area = area if area in areas else ""
        active_metrics = _active_metrics_config() or {}
        runtime_metrics = {}
        try:
            if METRICS_RUNTIME_PATH.exists():
                with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                    runtime_metrics = json.load(f)
        except Exception:
            runtime_metrics = {}

        area_runtime_entry = {}
        if area and isinstance(runtime_metrics.get(area), dict):
            area_runtime_entry = copy.deepcopy(runtime_metrics.get(area) or {})
            area_runtime_entry.pop("__replace__", None)

        area_metrics_cfg = (
            area_runtime_entry
            if area_runtime_entry
            else (active_metrics.get(area, {}) if area else {})
        )
        if not isinstance(area_metrics_cfg, dict):
            area_metrics_cfg = {}
        if "services" not in area_metrics_cfg or not isinstance(area_metrics_cfg.get("services"), dict):
            area_metrics_cfg["services"] = {}
        runtime_services_cfg = (
            area_runtime_entry.get("services", {}) if isinstance(area_runtime_entry, dict) else {}
        )
        out = {
            "areas": areas,
            "active_area": active_area,
            "llm": merge_area_section("llm"),
            "metrics_source": merge_area_section("metrics_source"),
            "lt_metrics_source": merge_area_section("lt_metrics_source"),
            "default_params": merge_area_section("default_params"),
            "storage": {"timescale": (CONFIG.get("storage", {}) or {}).get("timescale", {})},
            "queries": merge_area_section("queries"),
            "confluence": {
                "url_basic": CONFIG.get("url_basic"),
                "space_conf": CONFIG.get("space_conf"),
                "user": CONFIG.get("user"),
                "password": CONFIG.get("password"),
                "grafana_base_url": CONFIG.get("grafana_base_url"),
                "grafana_login": CONFIG.get("grafana_login"),
                "grafana_pass": CONFIG.get("grafana_pass"),
                "loki_url": CONFIG.get("loki_url"),
            },
            "metrics_config": area_metrics_cfg,
        }
        services_meta = {}
        area_metrics_services = area_metrics_cfg.get("services", {}) if isinstance(area_metrics_cfg, dict) else {}
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
            services_meta.setdefault(
                sid,
                {
                    "title": sid,
                    "disabled_domains": [],
                },
            )
        queries_map = {"": merge_area_section("queries")}
        for sid in services_meta.keys():
            meta = services_map.get(sid) or {}
            svc_queries = meta.get("queries") if isinstance(meta, dict) else {}
            queries_map[sid] = svc_queries if isinstance(svc_queries, dict) else {}
        metrics_config_map = {"": area_metrics_cfg}
        for sid, cfg in area_metrics_services.items():
            svc_runtime = {}
            if isinstance(runtime_services_cfg, dict) and isinstance(runtime_services_cfg.get(sid), dict):
                svc_runtime = copy.deepcopy(runtime_services_cfg.get(sid))
            metrics_config_map[sid] = (
                svc_runtime if svc_runtime else (cfg if isinstance(cfg, dict) else {})
            )
        out["services_meta"] = services_meta
        out["domain_list"] = _available_domain_keys()
        out["queries_map"] = queries_map
        out["metrics_config_map"] = metrics_config_map
        return jsonify(out)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@config_bp.route("/config", methods=["POST"])
def update_config():
    """Применяет изменения конфигурации (глобально, по области или по сервису)."""
    data = request.get_json(silent=True) or {}
    section = str(data.get("section") or "").strip()
    payload = data.get("data")
    area = (data.get("area") or "").strip()
    service = (data.get("service") or "").strip()
    if not section or not isinstance(payload, dict):
        return jsonify({"error": "section и data обязательны"}), 400
    try:
        if section == "confluence":
            allowed = {"url_basic", "space_conf", "user", "password", "grafana_base_url", "grafana_login", "grafana_pass", "loki_url"}
            for k, v in (payload or {}).items():
                if k in allowed:
                    CONFIG[k] = v
            existing = {}
            try:
                if CONFIG_RUNTIME_PATH.exists():
                    with CONFIG_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                        existing = json.load(f)
            except Exception:
                existing = {}
            for k, v in (payload or {}).items():
                if k in allowed:
                    existing[k] = v
            with CONFIG_RUNTIME_PATH.open("w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            return jsonify({"status": "ok"})

        if section == "metrics_config":
            if not area:
                return jsonify({"error": "area обязательна для metrics_config"}), 400
            if not isinstance(payload, dict):
                return jsonify({"error": "metrics_config должен быть объектом"}), 400
            payload_copy = copy.deepcopy(payload)
            current = {}
            try:
                if METRICS_RUNTIME_PATH.exists():
                    with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                        current = json.load(f)
            except Exception:
                current = {}
            if not isinstance(current, dict):
                current = {}
            if service:
                if area not in current or not isinstance(current.get(area), dict):
                    current[area] = {"services": {}}
                if "services" not in current[area] or not isinstance(current[area].get("services"), dict):
                    current[area]["services"] = {}
                current[area]["services"][service] = payload_copy
            else:
                if not isinstance(payload.get("services"), dict):
                    return jsonify({"error": "metrics_config должен содержать ключ 'services' с объектом сервисов"}), 400
                current[area] = payload_copy
                current[area]["__replace__"] = True
            with METRICS_RUNTIME_PATH.open("w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
            return jsonify({"status": "ok"})

        if section == "queries" and area and service:
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
            area_entry["services"][service]["queries"] = payload
            _save_settings_runtime_data(runtime)
            return jsonify({"status": "ok"})

        area_overridable = {"llm", "metrics_source", "lt_metrics_source", "default_params", "queries"}
        if section in area_overridable and area:
            if "per_area" not in CONFIG or not isinstance(CONFIG.get("per_area"), dict):
                CONFIG["per_area"] = {}
            if area not in CONFIG["per_area"] or not isinstance(CONFIG["per_area"].get(area), dict):
                CONFIG["per_area"][area] = {}
            CONFIG["per_area"][area][section] = payload

            existing = {}
            try:
                if CONFIG_RUNTIME_PATH.exists():
                    with CONFIG_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                        existing = json.load(f)
            except Exception:
                existing = {}
            if "per_area" not in existing or not isinstance(existing.get("per_area"), dict):
                existing["per_area"] = {}
            if area not in existing["per_area"] or not isinstance(existing["per_area"].get(area), dict):
                existing["per_area"][area] = {}
            existing["per_area"][area][section] = payload
            with CONFIG_RUNTIME_PATH.open("w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            return jsonify({"status": "ok"})

        target = CONFIG
        parts = section.split(".")
        for i, p in enumerate(parts):
            if i == len(parts) - 1:
                target[p] = payload
            else:
                if p not in target or not isinstance(target[p], dict):
                    target[p] = {}
                target = target[p]
        existing = {}
        try:
            if CONFIG_RUNTIME_PATH.exists():
                with CONFIG_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
        except Exception:
            existing = {}
        tgt = existing
        for i, p in enumerate(parts):
            if i == len(parts) - 1:
                tgt[p] = payload
            else:
                if p not in tgt or not isinstance(tgt[p], dict):
                    tgt[p] = {}
                tgt = tgt[p]
        with CONFIG_RUNTIME_PATH.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return jsonify({"status": "ok"})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


