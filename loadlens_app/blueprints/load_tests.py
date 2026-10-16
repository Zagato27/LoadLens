"""Pages and API for running load tests from LoadLens (see loadlens_app/load_tests)."""

from __future__ import annotations

from flask import Blueprint, g, jsonify, render_template, request

from loadlens_app.auth import Role, current_auth
from loadlens_app.core import _active_project_area, _find_area_for_service, _resolve_services_filter
from loadlens_app.errors import internal_error
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
    return jsonify({"status": "error", "error": str(exc), "message": str(exc), "code": exc.code}), exc.status


def _run_or_404(run_id: str):
    run = get_store().get(run_id)
    if run is None:
        return None, (jsonify({"error": "Прогон не найден", "code": "not_found"}), 404)
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
        return jsonify({"error": "limit должен быть целым числом"}), 400
    limit = max(1, min(limit, RUNS_MAX_LIMIT))
    statuses = RUN_ACTIVE if request.args.get("active") == "1" else RUN_STATUSES
    services = _resolve_services_filter(_active_project_area())
    runs = get_store().list(limit=limit, statuses=statuses, services=services or None)
    if any(run.active for run in runs):
        ensure_watcher()
    catalog = service.load_catalog()
    return jsonify({"runs": [service.run_view(run, catalog, with_report=False) for run in runs]}), 200


@load_tests_bp.route("/load_tests/active", methods=["GET"])
def active():
    """Идущие прогоны активной области для дэшборда и счётчика в меню.

    ``limit`` — сколько прогонов отдать подробно (0 — только число), ``live=1`` — RPS и ошибки за 15 минут.
    """
    try:
        limit = int(request.args.get("limit", "5") or 0)
    except ValueError:
        return jsonify({"error": "limit должен быть целым числом"}), 400
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
        return jsonify({"error": "index — номер инстанса (целое число)"}), 400
    try:
        run = service.request_stop(run_id, index)
    except service.LaunchError as exc:
        return _error(exc)
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
        return jsonify({"error": "index и bytes — целые числа"}), 400
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
        return jsonify({"error": "window — минуты, since — время в мс", "code": "bad_request"}), 400
    mode = request.args.get("mode", "").strip()
    if mode not in ("", "filtered", "plain"):
        return jsonify({"error": "mode — filtered или plain", "code": "bad_request"}), 400
    return jsonify(live_snapshot(run, service.load_catalog().tools, window_min=window, since=since, mode=mode)), 200


@load_tests_bp.route("/lt_windows", methods=["GET"])
def lt_windows():
    """Недавние окна нагрузки сервиса для формы отчёта: отрезки с RPS > 0 и завершённые прогоны.

    ``service`` — сервис, ``hours`` — глубина поиска (по умолчанию 6, не больше 24). Ошибки источника
    метрик не дают ошибку ответа: ``ok`` = false, ``message`` объясняет причину.
    """
    service_id = request.args.get("service", "").strip()
    if not service_id:
        return jsonify({"error": "Укажите сервис", "code": "bad_request"}), 400
    raw_hours = request.args.get("hours", "").strip()
    try:
        hours = float(raw_hours) if raw_hours else None
    except ValueError:
        return jsonify({"error": "hours — число часов", "code": "bad_request"}), 400
    area = _find_area_for_service(service_id)
    if not area:
        return jsonify({"error": f"Сервис «{service_id}» не найден в настройках", "code": "not_found"}), 404
    if hours is None:
        return jsonify(cached_load_windows(area, service_id)), 200
    return jsonify(cached_load_windows(area, service_id, hours)), 200


__all__ = ["load_tests_bp"]
