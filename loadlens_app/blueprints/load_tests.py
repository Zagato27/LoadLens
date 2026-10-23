# SPDX-License-Identifier: Apache-2.0
"""Pages and API for running load tests from LoadLens (see loadlens_app/load_tests)."""

from __future__ import annotations

from flask import Blueprint, g, jsonify, render_template, request

from loadlens_app import response_cache
from loadlens_app.auth import Role, current_auth
from loadlens_app.core import _active_project_area, _find_area_for_service, _resolve_services_filter
from loadlens_app.errors import internal_error
from loadlens_app.i18n import current_locale, translate
from loadlens_app.load_tests import service
from loadlens_app.load_tests.executors import TOKEN_HEADER
from loadlens_app.load_tests.live import cached_load_windows, live_snapshot
from loadlens_app.load_tests.store import RUN_ACTIVE, RUN_STATUSES, get_store
from loadlens_app.load_tests.watcher import ensure_watcher

load_tests_bp = Blueprint("load_tests", __name__)

RUNS_MAX_LIMIT = 200
LOG_MAX_BYTES = 256 * 1024
AUDIT_COMMAND_CHARS = 300


def _owner() -> str:
    user = current_auth().user
    return user.username if user is not None else "local"


def _error(exc: service.LaunchError):
    # str() renders the message in the language of this request.
    body = {"status": "error", "error": str(exc), "message": str(exc), "code": exc.code}
    # The launch page opens the card of the instance by this number (from 1), not by the message text.
    if getattr(exc, "instance", None) is not None:
        body["instance"] = exc.instance
    return jsonify(body), exc.status


def _run_or_404(run_id: str):
    run = get_store().get(run_id)
    if run is None:
        return None, (jsonify({"error": translate("lt.launch.run_not_found"), "code": "not_found"}), 404)
    return run, None


@load_tests_bp.route("/launch")
def launch_page():
    ensure_watcher()
    return render_template("launch.html")


@load_tests_bp.route("/launch/<run_id>")
def run_page(run_id: str):
    ensure_watcher()
    return render_template("launch_run.html", run_id=run_id)


@load_tests_bp.route("/load_tests/options", methods=["GET"])
def options():
    """Генераторы, инструменты и лимиты для формы запуска (без учётных данных)."""
    try:
        return jsonify(service.launch_options()), 200
    except Exception as exc:  # pragma: no cover
        return internal_error(exc)


@load_tests_bp.route("/load_tests/scripts", methods=["GET"])
def scripts():
    """Скрипты инструмента в каталоге генератора (поиск по маскам шаблона)."""
    try:
        return jsonify(service.list_scripts(request.args.get("generator", ""), request.args.get("tool", ""))), 200
    except service.LaunchError as exc:
        return _error(exc)


@load_tests_bp.route("/load_tests/preview", methods=["POST"])
def preview():
    """Проверяет форму и показывает команды инстансов; ничего не запускает."""
    data = request.get_json(silent=True)
    try:
        planned = service.plan_run(data or {}, _owner())
    except service.LaunchError as exc:
        return _error(exc)
    return jsonify({
        "run_name": planned.run.run_name,
        "run_tag": planned.run.run_tag,
        "commands": planned.commands,
        "warnings": planned.warnings,
    }), 200


@load_tests_bp.route("/load_tests", methods=["POST"])
def create():
    """Создаёт прогон; инстансы запускает фоновый наблюдатель."""
    data = request.get_json(silent=True)
    try:
        planned = service.create_run(data or {}, _owner())
    except service.LaunchError as exc:
        return _error(exc)
    response_cache.invalidate(response_cache.ACTIVE)
    run = planned.run
    g.audit_extra = {
        "run_id": run.run_id,
        "run_name": run.run_name,
        "instances": [
            f"{inst.generator}/{inst.tool or '-'}: {(inst.command or 'manual')[:AUDIT_COMMAND_CHARS]}" for inst in run.instances
        ][:20],
    }
    return jsonify({
        "status": "accepted",
        "run_id": run.run_id,
        "run_name": run.run_name,
        "run_tag": run.run_tag,
        "warnings": planned.warnings,
        "url": f"/launch/{run.run_id}",
    }), 200


@load_tests_bp.route("/load_tests/runs", methods=["GET"])
def list_runs():
    """Прогоны активной области, новые первыми; ``active=1`` — только идущие."""
    try:
        limit = int(request.args.get("limit", "50") or 50)
    except ValueError:
        return jsonify({"error": translate("lt.api.limit_int")}), 400
    limit = max(1, min(limit, RUNS_MAX_LIMIT))
    statuses = RUN_ACTIVE if request.args.get("active") == "1" else RUN_STATUSES
    services = _resolve_services_filter(_active_project_area())
    runs = get_store().list(limit=limit, statuses=statuses, services=services or None)
    if any(run.active for run in runs):
        ensure_watcher()
    catalog = service.load_catalog()
    return jsonify({"runs": [service.run_view(run, catalog, with_report=False) for run in runs]}), 200


def _active_cache_key() -> tuple | None:
    """Area services, the query, the user (``can_stop`` depends on who asks) and the language of the texts."""
    try:
        limit = int(request.args.get("limit", "5") or 0)
    except ValueError:
        return None
    return (
        tuple(_resolve_services_filter(_active_project_area()) or ()),
        limit,
        request.args.get("live") == "1",
        _owner(),
        current_auth().role >= Role.ADMIN,
        current_locale(),
    )


@load_tests_bp.route("/load_tests/active", methods=["GET"])
@response_cache.cached_json(response_cache.ACTIVE, response_cache.ACTIVE_TTL_SEC, _active_cache_key)
def active():
    """Идущие прогоны активной области для дэшборда и счётчика в меню.

    ``limit`` — сколько прогонов отдать подробно (0 — только число), ``live=1`` — RPS и ошибки за 15 минут.
    """
    try:
        limit = int(request.args.get("limit", "5") or 0)
    except ValueError:
        return jsonify({"error": translate("lt.api.limit_int")}), 400
    auth = current_auth()
    data = service.active_runs(
        _resolve_services_filter(_active_project_area()) or None,
        limit,
        _owner(),
        auth.role >= Role.ADMIN,
        request.args.get("live") == "1",
    )
    if data["total"]:
        ensure_watcher()
    return jsonify(data), 200


HISTORY_DEFAULT_LIMIT = 20
HISTORY_MAX_OFFSET = 1_000_000
HISTORY_QUERY_CHARS = 200
# Why a run was not deleted -> HTTP answer of the one-run DELETE.
_SKIP_STATUS = {"not_found": (404, "not_found"), "active": (409, "conflict"), "forbidden": (403, "forbidden")}


def _int_arg(name: str, default: int) -> int:
    raw = request.args.get(name, "")
    return int(raw) if str(raw).strip() else default


@load_tests_bp.route("/load_tests/history", methods=["GET"])
def history():
    """Runs of the active area for the archive's Runs tab: no commands or parameters, open to viewers.

    ``q`` is part of the name, ``service`` one service, ``status`` codes joined by commas (active, done,
    failed, cancelled), ``offset`` and ``limit`` (up to 100). The match count is in ``total`` and ``X-Total-Count``.
    """
    try:
        offset = max(0, min(_int_arg("offset", 0), HISTORY_MAX_OFFSET))
        limit = _int_arg("limit", HISTORY_DEFAULT_LIMIT)
    except ValueError:
        return jsonify({"error": translate("lt.api.limit_int"), "code": "bad_request"}), 400
    try:
        statuses = service.history_statuses(request.args.get("status", ""))
    except service.LaunchError as exc:
        return _error(exc)
    query = request.args.get("q", "").strip()[:HISTORY_QUERY_CHARS]
    services = _resolve_services_filter(_active_project_area()) or []
    wanted = request.args.get("service", "").strip()
    if wanted:
        if services and wanted not in services:
            return _history_response({"runs": [], "total": 0})
        services = [wanted]
    auth = current_auth()
    page = service.history_page(
        services, statuses, query, offset, limit,
        user=_owner(), can_write=auth.role >= Role.ENGINEER, is_admin=auth.role >= Role.ADMIN,
    )
    if any(run["active"] for run in page["runs"]):
        ensure_watcher()
    return _history_response(page)


def _history_response(page: dict):
    response = jsonify(page)
    response.headers["X-Total-Count"] = str(page["total"])
    return response, 200


def _after_delete(result: dict, with_reports: bool) -> None:
    if result["reports_deleted"]:
        response_cache.invalidate(response_cache.DASHBOARD)
    g.audit_extra = {
        "run_ids": ",".join(run["run_id"] for run in result["deleted"])[:500],
        "run_names": ", ".join(run["run_name"] for run in result["deleted"])[:500],
        "with_reports": "1" if with_reports else "0",
        "reports_deleted": ", ".join(result["reports_deleted"])[:500],
        "skipped": str(len(result["skipped"])),
    }


@load_tests_bp.route("/load_tests/runs/delete", methods=["POST"])
def delete_runs():
    """Deletes finished runs: ``{"run_ids": [...], "with_reports": false}``, up to 100 at once.

    Engineers delete their own runs, an administrator anyone's; active runs are skipped. ``with_reports``
    (administrators only) deletes their reports too. The answer lists deleted and skipped runs with the reason.
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": translate("lt.launch.json_object_expected"), "code": "bad_request"}), 400
    with_reports = body.get("with_reports") is True
    auth = current_auth()
    try:
        result = service.delete_runs(body.get("run_ids"), _owner(), auth.role >= Role.ADMIN, with_reports)
    except service.LaunchError as exc:
        return _error(exc)
    _after_delete(result, with_reports)
    return jsonify(result), 200


@load_tests_bp.route("/load_tests/runs/<run_id>", methods=["DELETE"])
def delete_run(run_id: str):
    """Deletes one finished run; ``with_report=1`` (administrators) deletes its report too."""
    with_report = request.args.get("with_report") == "1"
    auth = current_auth()
    try:
        result = service.delete_runs([run_id], _owner(), auth.role >= Role.ADMIN, with_report)
    except service.LaunchError as exc:
        return _error(exc)
    if result["skipped"]:
        skip = result["skipped"][0]
        status, code = _SKIP_STATUS.get(skip["code"], (409, "conflict"))
        return jsonify({"status": "error", "error": skip["message"], "message": skip["message"], "code": code}), status
    _after_delete(result, with_report)
    return jsonify({
        "status": "ok",
        "run_id": run_id,
        "report_deleted": bool(result["reports_deleted"]),
        "reports_failed": result["reports_failed"],
    }), 200


@load_tests_bp.route("/load_tests/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    run, missing = _run_or_404(run_id)
    if missing:
        return missing
    if run.active:
        ensure_watcher()
    return jsonify(service.run_view(run)), 200


@load_tests_bp.route("/load_tests/runs/<run_id>/stop", methods=["POST"])
def stop(run_id: str):
    """Останавливает прогон целиком или один инстанс (``{"index": N}``)."""
    body = request.get_json(silent=True) or {}
    index = body.get("index") if isinstance(body, dict) else None
    if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
        return jsonify({"error": translate("lt.api.index_int")}), 400
    try:
        run = service.request_stop(run_id, index)
    except service.LaunchError as exc:
        return _error(exc)
    response_cache.invalidate(response_cache.ACTIVE)
    g.audit_extra = {"run_id": run_id, "run_name": run.run_name, "instance": "all" if index is None else str(index)}
    return jsonify(service.run_view(run, with_report=False)), 200


@load_tests_bp.route("/load_tests/runs/<run_id>/logs", methods=["GET"])
def logs(run_id: str):
    """Хвост вывода инструмента одного инстанса."""
    run, missing = _run_or_404(run_id)
    if missing:
        return missing
    try:
        index = int(request.args.get("index", "0") or 0)
        max_bytes = int(request.args.get("bytes", "65536") or 65536)
    except ValueError:
        return jsonify({"error": translate("lt.api.index_bytes_int")}), 400
    max_bytes = max(1024, min(max_bytes, LOG_MAX_BYTES))
    try:
        text = service.instance_logs(run, index, max_bytes)
    except service.LaunchError as exc:
        return _error(exc)
    return jsonify({"index": index, "text": text, "truncated": len(text.encode("utf-8")) >= max_bytes}), 200


@load_tests_bp.route("/load_tests/hints", methods=["GET"])
def hints():
    """Флаги и переменные вывода метрик инструмента под источник домена «Нагрузка» сервиса."""
    try:
        data = service.output_hints(
            request.args.get("service", "").strip(), request.args.get("project_area", "").strip(), request.args.get("tool", "").strip(),
        )
    except service.LaunchError as exc:
        return _error(exc)
    return jsonify(data), 200


@load_tests_bp.route("/load_tests/presets", methods=["GET"])
def list_presets():
    """Пресеты формы запуска для сервисов активной области (``service=`` — одного сервиса)."""
    wanted = request.args.get("service", "").strip()
    services = [wanted] if wanted else (_resolve_services_filter(_active_project_area()) or None)
    return jsonify({"presets": service.list_presets(services)}), 200


@load_tests_bp.route("/load_tests/presets", methods=["POST"])
def save_preset():
    """Сохраняет форму запуска как пресет сервиса; одноимённый пресет заменяет его автор или администратор."""
    try:
        preset = service.save_preset(request.get_json(silent=True) or {}, _owner(), current_auth().role >= Role.ADMIN)
    except service.LaunchError as exc:
        return _error(exc)
    g.audit_extra = {"preset": preset["name"], "service": preset["service"]}
    return jsonify(preset), 200


@load_tests_bp.route("/load_tests/presets/<preset_id>", methods=["DELETE"])
def delete_preset(preset_id: str):
    try:
        service.delete_preset(preset_id, _owner(), current_auth().role >= Role.ADMIN)
    except service.LaunchError as exc:
        return _error(exc)
    return jsonify({"status": "ok"}), 200


@load_tests_bp.route("/load_tests/ci_params", methods=["GET"])
def ci_params():
    """Параметры пайплайна CI-генератора для формы запуска: ``generator=``, ``ref=`` — ветка (пусто — по умолчанию)."""
    try:
        return jsonify(service.ci_params_view(request.args.get("generator", "").strip(), request.args.get("ref", ""))), 200
    except service.LaunchError as exc:
        return _error(exc)


@load_tests_bp.route("/load_tests/ci_refs", methods=["GET"])
def ci_refs():
    """Ветки проекта GitLab или multibranch-задачи Jenkins: ``generator=``, ``search=``."""
    try:
        return jsonify(service.ci_refs_view(request.args.get("generator", "").strip(), request.args.get("search", ""))), 200
    except service.LaunchError as exc:
        return _error(exc)


@load_tests_bp.route("/load_tests/ci/<run_id>/<int:index>/events", methods=["POST"])
def ci_events(run_id: str, index: int):
    """События задачи CI: ``{"event": "started"}`` и ``{"event": "finished", "exit_code": N}``."""
    try:
        run, inst, _generator = service.ci_instance(run_id, index, request.headers.get(TOKEN_HEADER, ""))
        event = service.record_ci_event(run, inst, request.get_json(silent=True))
    except service.LaunchError as exc:
        return _error(exc)
    response_cache.invalidate(response_cache.ACTIVE)
    return jsonify({"status": "ok", "event": event["event"]}), 200


@load_tests_bp.route("/load_tests/runs/<run_id>/live", methods=["GET"])
def live(run_id: str):
    """RPS, ошибки, p95, пользователи и RPS инстансов прогона по запросам домена «Нагрузка».

    ``window`` — последние 15, 30 или 60 минут (0 — весь прогон); ``since`` — только точки с этого момента
    (мс), для страницы, у которой остальные уже есть; ``mode`` — ``filtered`` или ``plain`` из первого ответа.
    """
    run, missing = _run_or_404(run_id)
    if missing:
        return missing
    try:
        window = int(request.args.get("window") or 0)
        since_ms = request.args.get("since", "").strip()
        since = int(since_ms) / 1000 if since_ms else None
    except ValueError:
        return jsonify({"error": translate("lt.api.window_since"), "code": "bad_request"}), 400
    mode = request.args.get("mode", "").strip()
    if mode not in ("", "filtered", "plain"):
        return jsonify({"error": translate("lt.api.mode"), "code": "bad_request"}), 400
    return jsonify(live_snapshot(run, service.load_catalog().tools, window_min=window, since=since, mode=mode)), 200


@load_tests_bp.route("/lt_windows", methods=["GET"])
def lt_windows():
    """Недавние окна нагрузки сервиса для формы отчёта: отрезки с RPS > 0 и завершённые прогоны.

    ``service`` — сервис, ``hours`` — глубина поиска (по умолчанию 6, не больше 24). Ошибки источника
    метрик не дают ошибку ответа: ``ok`` = false, ``message`` объясняет причину.
    """
    service_id = request.args.get("service", "").strip()
    if not service_id:
        return jsonify({"error": translate("lt.api.service_required"), "code": "bad_request"}), 400
    raw_hours = request.args.get("hours", "").strip()
    try:
        hours = float(raw_hours) if raw_hours else None
    except ValueError:
        return jsonify({"error": translate("lt.api.hours_number"), "code": "bad_request"}), 400
    area = _find_area_for_service(service_id)
    if not area:
        return jsonify({"error": translate("lt.api.service_not_found", service=service_id), "code": "not_found"}), 404
    if hours is None:
        return jsonify(cached_load_windows(area, service_id)), 200
    return jsonify(cached_load_windows(area, service_id, hours)), 200


__all__ = ["load_tests_bp"]
