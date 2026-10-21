# SPDX-License-Identifier: Apache-2.0
"""Blueprint for settings UI and runtime overrides."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flask import Blueprint, jsonify, render_template, request

from AI.i18n import t as translate
from loadlens_app.errors import internal_error

from loadlens_app.core import (
    LOCKED_PROMPT_DOMAINS,
    METRICS_RUNTIME_PATH,
    PROMPT_DOMAIN_FILES,
    _active_area_prompt_scope,
    _available_domain_keys,
    _delete_service_data,
    _find_area_for_service,
    _list_project_areas,
    _load_base_prompts,
    _load_metrics_runtime_data,
    _load_settings_runtime_data,
    _save_metrics_runtime_data,
    _save_settings_runtime_data,
    _services_map_for_area,
)

settings_bp = Blueprint("settings", __name__)

PROMPT_HISTORY_LIMIT = 10


def _history_scope(area: str, service: str) -> str:
    """Key of the prompt history bucket: per area, optionally per service."""
    return f"{area}:{service}" if service else area


def _current_prompt_override(runtime: dict, area: str, service: str, domain: str) -> str:
    """Currently stored override text for the scope (empty if none)."""
    per_area = runtime.get("per_area") if isinstance(runtime.get("per_area"), dict) else {}
    area_entry = per_area.get(area) if isinstance(per_area.get(area), dict) else {}
    if service:
        services = area_entry.get("services") if isinstance(area_entry.get("services"), dict) else {}
        scope = services.get(service) if isinstance(services.get(service), dict) else {}
    else:
        scope = area_entry
    prompts = scope.get("prompts") if isinstance(scope.get("prompts"), dict) else {}
    value = prompts.get(domain)
    return value if isinstance(value, str) else ""


def _record_prompt_history(runtime: dict, area: str, service: str, domain: str, previous_text: str) -> None:
    """Keeps the last PROMPT_HISTORY_LIMIT replaced versions per scope and domain."""
    if not previous_text.strip():
        return
    history = runtime.get("prompt_history")
    if not isinstance(history, dict):
        history = {}
    scope_key = _history_scope(area, service)
    scope_bucket = history.get(scope_key)
    if not isinstance(scope_bucket, dict):
        scope_bucket = {}
    versions = scope_bucket.get(domain)
    if not isinstance(versions, list):
        versions = []
    if versions and isinstance(versions[0], dict) and versions[0].get("text") == previous_text:
        return
    versions.insert(0, {"saved_at": datetime.now(timezone.utc).isoformat(), "text": previous_text})
    scope_bucket[domain] = versions[:PROMPT_HISTORY_LIMIT]
    history[scope_key] = scope_bucket
    runtime["prompt_history"] = history


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
        return jsonify({"error": translate("api.required.area_service")}), 400
    _delete_service_data(area, service)
    return jsonify({"status": "ok"})


@settings_bp.route("/prompts", methods=["GET"])
def get_prompts():
    """Возвращает тексты промптов для выбранной области/сервиса.

    Без области отдаются базовые тексты из файлов (глобальный уровень);
    область берётся из параметра ``area`` или выводится из ``service``.
    """
    try:
        area = (request.args.get("area") or "").strip()
        service = (request.args.get("service") or "").strip()
        area_ids = [a["id"] for a in _list_project_areas()]
        if service and not area:
            derived = _find_area_for_service(service)
            if derived:
                area = derived
        active_area = area if area in area_ids else ""
        services_map = _services_map_for_area(active_area) if active_area else {}
        services_payload = []
        for sid, meta in services_map.items():
            title = sid
            if isinstance(meta, dict):
                title = (meta.get("title") if isinstance(meta.get("title"), str) else "") or title
            services_payload.append({"id": sid, "title": title})
        if service and service not in services_map:
            service = ""
        # Defaults are shown in the language of reports; an override keeps its own text (and language).
        scope = _active_area_prompt_scope(active_area or None, service or None)
        editable = [k for k in scope["texts"] if k not in LOCKED_PROMPT_DOMAINS]
        return jsonify(
            {
                "areas": area_ids,
                "active_area": active_area,
                "services": services_payload,
                "active_service": service if service else "",
                "domains": {k: scope["texts"][k] for k in editable},
                "language": scope["language"],
                "sources": {k: scope["sources"][k] for k in editable},
                "override_language": {k: v for k, v in scope["override_language"].items() if k in editable},
            }
        )
    except Exception as e:  # pragma: no cover
        return internal_error(e)


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
            return jsonify({"error": translate("api.required.area")}), 400
        if domain not in PROMPT_DOMAIN_FILES:
            return jsonify({"error": translate("api.settings.bad_domain")}), 400
        if domain in LOCKED_PROMPT_DOMAINS:
            return jsonify({"error": translate("server.prompts.locked_domain")}), 400
        if not isinstance(text, str):
            return jsonify({"error": translate("server.prompts.text_string")}), 400
        existing = _load_settings_runtime_data()
        previous = _current_prompt_override(existing, area, service, domain)
        if previous != text:
            _record_prompt_history(existing, area, service, domain, previous)
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
            if text.strip():
                svc_entry["prompts"][domain] = text
            else:
                svc_entry["prompts"].pop(domain, None)
        else:
            if "prompts" not in target or not isinstance(target.get("prompts"), dict):
                target["prompts"] = {}
            if text.strip():
                target["prompts"][domain] = text
            else:
                target["prompts"].pop(domain, None)
        _save_settings_runtime_data(existing)
        return jsonify({"status": "ok"})
    except Exception as e:  # pragma: no cover
        return internal_error(e)


@settings_bp.route("/prompts/defaults", methods=["GET"])
def get_prompt_defaults():
    """Базовые тексты промптов из AI/prompts/<lang>/*.txt (без runtime-переопределений).

    ``?lang=ru|en`` выбирает язык; по умолчанию — язык отчётов (``appearance.language``).
    """
    from loadlens_app.i18n import normalize_lang, report_language

    lang = normalize_lang(request.args.get("lang")) or report_language()
    base = _load_base_prompts(lang)
    return jsonify({"language": lang, "domains": {k: v for k, v in base.items() if k not in LOCKED_PROMPT_DOMAINS}})


@settings_bp.route("/prompts/history", methods=["GET"])
def get_prompt_history():
    """Последние сохранённые версии промпта для области (и, опционально, сервиса)."""
    domain = (request.args.get("domain") or "").strip()
    area = (request.args.get("area") or "").strip()
    service = (request.args.get("service") or "").strip()
    if domain not in PROMPT_DOMAIN_FILES or domain in LOCKED_PROMPT_DOMAINS:
        return jsonify({"error": translate("api.settings.bad_domain")}), 400
    if not area and service:
        area = _find_area_for_service(service) or ""
    if not area:
        return jsonify({"error": translate("api.required.area")}), 400
    runtime = _load_settings_runtime_data()
    history = runtime.get("prompt_history") if isinstance(runtime.get("prompt_history"), dict) else {}
    scope_bucket = history.get(_history_scope(area, service))
    versions = scope_bucket.get(domain) if isinstance(scope_bucket, dict) else None
    clean = [v for v in (versions or []) if isinstance(v, dict) and isinstance(v.get("text"), str)]
    return jsonify({"domain": domain, "area": area, "service": service, "versions": clean})


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
            return jsonify({"error": translate("api.required.area_service")}), 400
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
        return internal_error(e)


@settings_bp.route("/service_meta", methods=["DELETE"])
def delete_service_meta():
    """Удаляет runtime-метаданные и пользовательские оверрайды сервиса без очистки БД."""
    try:
        data = request.get_json(silent=True) or {}
        area = (data.get("area") or "").strip()
        service = (data.get("service") or "").strip()
        if service and not area:
            area = _find_area_for_service(service) or ""
        if not area or not service:
            return jsonify({"error": translate("api.required.area_service")}), 400

        runtime = _load_settings_runtime_data()
        changed = False
        if isinstance(runtime.get("per_area"), dict):
            area_entry = runtime["per_area"].get(area)
            if isinstance(area_entry, dict) and isinstance(area_entry.get("services"), dict):
                if service in area_entry["services"]:
                    del area_entry["services"][service]
                    changed = True
        if changed:
            _save_settings_runtime_data(runtime)

        metrics_rt = _load_metrics_runtime_data()
        if isinstance(metrics_rt.get(area), dict):
            services_entry = metrics_rt[area].get("services")
            if isinstance(services_entry, dict) and service in services_entry:
                del services_entry[service]
                _save_metrics_runtime_data(metrics_rt)

        return jsonify({"status": "ok", "service": service, "area": area})
    except Exception as e:  # pragma: no cover
        return internal_error(e)




