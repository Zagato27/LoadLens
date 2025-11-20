"""Blueprint for settings UI and runtime overrides."""

from __future__ import annotations

import json
from flask import Blueprint, jsonify, render_template, request

from loadlens_app.core import (
    CONFIG_RUNTIME_PATH,
    LOCKED_PROMPT_DOMAINS,
    METRICS_RUNTIME_PATH,
    PROMPT_DOMAIN_FILES,
    _active_area_prompts,
    _active_metrics_config,
    _available_domain_keys,
    _delete_service_data,
    _find_area_for_service,
    _load_settings_runtime_data,
    _save_settings_runtime_data,
    _services_map_for_area,
)

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings", methods=["GET"])
def settings_page():
    """Отдаёт страницу UI для редактирования конфигураций."""
    return render_template("settings.html")


@settings_bp.route("/service", methods=["DELETE"])
def delete_service():
    """Удаляет сервис в выбранной области и полностью очищает его данные."""
    data = request.get_json(silent=True) or {}
    area = (data.get("area") or "").strip()
    service = (data.get("service") or "").strip()
    if not area or not service:
        return jsonify({"error": "area и service обязательны"}), 400
    _delete_service_data(area, service)
    return jsonify({"status": "ok"})


@settings_bp.route("/areas/<area_name>", methods=["DELETE"])
def delete_area(area_name: str):
    """Удаляет область, все её сервисы и соответствующие runtime-конфиги."""
    area_name = (area_name or "").strip()
    if not area_name:
        return jsonify({"error": "area_name обязателен"}), 400
    services = list(_services_map_for_area(area_name))
    for service in services:
        _delete_service_data(area_name, service)
    runtime = _load_settings_runtime_data()
    if isinstance(runtime.get("per_area"), dict) and area_name in runtime["per_area"]:
        del runtime["per_area"][area_name]
        _save_settings_runtime_data(runtime)
    metrics_rt = {}
    try:
        if METRICS_RUNTIME_PATH.exists():
            with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                metrics_rt = json.load(f)
    except Exception:
        metrics_rt = {}
    if isinstance(metrics_rt, dict) and area_name in metrics_rt:
        del metrics_rt[area_name]
        with METRICS_RUNTIME_PATH.open("w", encoding="utf-8") as f:
            json.dump(metrics_rt, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})


@settings_bp.route("/prompts", methods=["GET"])
def get_prompts():
    """Возвращает тексты промптов для выбранной области/сервиса."""
    try:
        area = (request.args.get("area") or "").strip()
        service = (request.args.get("service") or "").strip()
        areas_meta = _active_metrics_config().keys()
        area_ids = list(areas_meta)
        if service and not area:
            derived = _find_area_for_service(service)
            if derived:
                area = derived
        if not area:
            cookie_area = _find_area_for_service(service) or ""
            if cookie_area in area_ids:
                area = cookie_area
        active_area = area if area in area_ids else (area_ids[0] if area_ids else "")
        services_map = _services_map_for_area(active_area) if active_area else {}
        services_payload = []
        for sid, meta in services_map.items():
            title = sid
            if isinstance(meta, dict):
                title = (meta.get("title") if isinstance(meta.get("title"), str) else "") or title
            services_payload.append({"id": sid, "title": title})
        if service and service not in services_map:
            service = ""
        prompts = _active_area_prompts(active_area if active_area in area_ids else None, service or None)
        filtered_prompts = {k: v for k, v in prompts.items() if k not in LOCKED_PROMPT_DOMAINS}
        return jsonify(
            {
                "areas": area_ids,
                "active_area": active_area if active_area in area_ids else "",
                "services": services_payload,
                "active_service": service if service else "",
                "domains": filtered_prompts,
            }
        )
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/prompts", methods=["POST"])
def post_prompts():
    """Сохраняет пользовательский промпт для области или сервиса."""
    try:
        data = request.get_json(silent=True) or {}
        area = (data.get("area") or "").strip()
        service = (data.get("service") or "").strip()
        domain = (data.get("domain") or "").strip()
        text = data.get("text")
        if not area and service:
            area = _find_area_for_service(service) or ""
        if not area:
            return jsonify({"error": "area обязательна"}), 400
        if domain not in PROMPT_DOMAIN_FILES:
            return jsonify({"error": "неверный domain"}), 400
        if domain in LOCKED_PROMPT_DOMAINS:
            return jsonify({"error": "Редактирование домена запрещено"}), 400
        if not isinstance(text, str):
            return jsonify({"error": "text должен быть строкой"}), 400
        existing = _load_settings_runtime_data()
        if "per_area" not in existing or not isinstance(existing.get("per_area"), dict):
            existing["per_area"] = {}
        if area not in existing["per_area"] or not isinstance(existing["per_area"].get(area), dict):
            existing["per_area"][area] = {}
        target = existing["per_area"][area]
        if service:
            if "services" not in target or not isinstance(target.get("services"), dict):
                target["services"] = {}
            if service not in target["services"] or not isinstance(target["services"].get(service), dict):
                target["services"][service] = {}
            svc_entry = target["services"][service]
            if "prompts" not in svc_entry or not isinstance(svc_entry.get("prompts"), dict):
                svc_entry["prompts"] = {}
            svc_entry["prompts"][domain] = text
        else:
            if "prompts" not in target or not isinstance(target.get("prompts"), dict):
                target["prompts"] = {}
            target["prompts"][domain] = text
        _save_settings_runtime_data(existing)
        return jsonify({"status": "ok"})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/service_meta", methods=["POST"])
def update_service_meta():
    """Обновляет метаданные сервиса: заголовок и отключённые домены."""
    try:
        data = request.get_json(silent=True) or {}
        area = (data.get("area") or "").strip()
        service = (data.get("service") or "").strip()
        meta = data.get("data") if isinstance(data.get("data"), dict) else {}
        if service and not area:
            area = _find_area_for_service(service) or ""
        if not area or not service:
            return jsonify({"error": "area и service обязательны"}), 400
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
        svc_entry = area_entry["services"][service]
        if "title" in meta:
            title_val = meta.get("title")
            svc_entry["title"] = str(title_val).strip() if isinstance(title_val, str) else ""
        if "disabled_domains" in meta and isinstance(meta.get("disabled_domains"), list):
            allowed = set(_available_domain_keys())
            svc_entry["disabled_domains"] = [d for d in meta.get("disabled_domains") if isinstance(d, str) and d in allowed]
        _save_settings_runtime_data(runtime)
        return jsonify({"status": "ok", "service": service, "meta": svc_entry})
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/areas", methods=["POST"])
def create_area():
    """Создаёт новую область (перезаписи в runtime) без перезапуска приложения."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name обязателен"}), 400
    active = _active_metrics_config() or {}
    if name in active:
        return jsonify({"error": "область уже существует"}), 400

    runtime = {}
    try:
        if METRICS_RUNTIME_PATH.exists():
            with METRICS_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                runtime = json.load(f)
    except Exception:
        runtime = {}
    if not isinstance(runtime, dict):
        runtime = {}
    runtime[name] = {"services": {}}
    with METRICS_RUNTIME_PATH.open("w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)

    existing = {}
    try:
        if CONFIG_RUNTIME_PATH.exists():
            with CONFIG_RUNTIME_PATH.open("r", encoding="utf-8") as f:
                existing = json.load(f)
    except Exception:
        existing = {}
    if "per_area" not in existing or not isinstance(existing.get("per_area"), dict):
        existing["per_area"] = {}
    if name not in existing["per_area"]:
        existing["per_area"][name] = {}
    with CONFIG_RUNTIME_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})


