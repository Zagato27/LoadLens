"""HTTP API of the capacity forecast pages."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Optional

import psycopg2
from flask import Blueprint, abort, jsonify, render_template, request

from AI.limit_explanation import ExplanationParseError
from loadlens_app.auth import current_auth
from loadlens_app.core import _active_project_area, _resolve_services_filter, public_base_url
from loadlens_app.forecast import (
    ForecastError,
    build_forecast,
    database_unavailable,
    forecast_status,
    list_forecast_reports,
)
# publish_forecast runs in loadlens_app.tasks, which looks it up here (tests replace it here).
from loadlens_app.forecast_confluence import load_forecast_publication, publish_forecast  # noqa: F401
from loadlens_app.forecast_explanation import ExplanationError, ExplanationView, explanation_view, generate_explanation
from loadlens_app import job_runner, tasks
from loadlens_app.jobs import JOB_KIND_FORECAST_CONFLUENCE, job_store
from loadlens_app.limits import LimitError, check_job_slot

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_TARGET_RPS = 10_000_000.0
DEFAULT_HEADROOM_PCT = 20.0
MAX_HEADROOM_PCT = 90.0
CPU_CEILING_RANGE_PCT = (10.0, 100.0)
MAX_RUN_NAME = 200

forecast_bp = Blueprint("forecast", __name__)


class InputError(ValueError):
    """A calculator parameter is outside its allowed range."""


@dataclass(frozen=True)
class ForecastInputs:
    target_rps: Optional[float]
    headroom_pct: float
    cpu_ceiling_pct: Optional[float]


def _number(source: Mapping[str, Any], name: str, message: str) -> Optional[float]:
    raw = source.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise InputError(message) from exc
    if not math.isfinite(value):
        raise InputError(message)
    return value


def _forecast_inputs(source: Mapping[str, Any]) -> ForecastInputs:
    """Target, headroom and CPU ceiling of the calculator, from query args or a JSON body."""
    target_message = "target_rps должен быть положительным числом не больше 10000000"
    target = _number(source, "target_rps", target_message)
    if target is not None and not 0 < target <= MAX_TARGET_RPS:
        raise InputError(target_message)
    headroom_message = "headroom_pct — от 0 до 90"
    headroom = _number(source, "headroom_pct", headroom_message)
    headroom = DEFAULT_HEADROOM_PCT if headroom is None else headroom
    if not 0 <= headroom <= MAX_HEADROOM_PCT:
        raise InputError(headroom_message)
    ceiling_message = "cpu_ceiling_pct — от 10 до 100"
    ceiling = _number(source, "cpu_ceiling_pct", ceiling_message)
    low, high = CPU_CEILING_RANGE_PCT
    if ceiling is not None and not low <= ceiling <= high:
        raise InputError(ceiling_message)
    return ForecastInputs(target, headroom, ceiling)


def _owner() -> str:
    user = current_auth().user
    return user.username if user is not None else "local"


def _bad_id(run_id: str):
    if RUN_ID_PATTERN.fullmatch(run_id or ""):
        return None
    return jsonify({"error": "Некорректный идентификатор отчёта", "code": "bad_request"}), 400


@forecast_bp.route("/forecasting")
def forecasting_page():
    """List of reports that can be forecast."""
    return render_template("forecasting.html")


@forecast_bp.route("/forecasting/<run_id>")
def forecasting_report_page(run_id: str):
    """Capacity forecast calculator of one report."""
    if not RUN_ID_PATTERN.fullmatch(run_id or ""):
        abort(404)
    return render_template("forecast_report.html")


@forecast_bp.route("/forecast_reports", methods=["GET"])
def get_forecast_reports():
    """Forecastable reports of the active project area."""
    try:
        catalog = list_forecast_reports(_resolve_services_filter(_active_project_area()) or [])
    except psycopg2.Error as exc:
        failure = database_unavailable(exc)
        return jsonify({"error": str(failure), "code": failure.code}), failure.status_code
    return jsonify(catalog.to_dict())


@forecast_bp.route("/forecast/<run_id>/status", methods=["GET"])
def get_forecast_status(run_id: str):
    """Whether the report page shows the «Прогноз мощностей» button."""
    rejected = _bad_id(run_id)
    if rejected is not None:
        return rejected
    try:
        status = forecast_status(run_id)
    except ForecastError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    except psycopg2.Error as exc:
        failure = database_unavailable(exc)
        return jsonify({"error": str(failure), "code": failure.code}), failure.status_code
    return jsonify(asdict(status))


@forecast_bp.route("/forecast/<run_id>", methods=["GET"])
def get_forecast(run_id: str):
    """Capacity forecast of a report; ``target_rps`` and ``headroom_pct`` drive the calculator."""
    rejected = _bad_id(run_id)
    if rejected is not None:
        return rejected
    try:
        inputs = _forecast_inputs(request.args)
    except InputError as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    try:
        report = build_forecast(run_id, inputs.target_rps, inputs.headroom_pct, inputs.cpu_ceiling_pct)
    except ForecastError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    except psycopg2.Error as exc:
        failure = database_unavailable(exc)
        return jsonify({"error": str(failure), "code": failure.code}), failure.status_code
    return jsonify(report.to_dict())


def _explanation_response(build: Callable[[], ExplanationView]):
    try:
        view = build()
    except (ForecastError, ExplanationError, LimitError) as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    except ExplanationParseError as exc:
        return jsonify({"error": str(exc), "code": "bad_answer"}), 502
    except psycopg2.Error as exc:
        failure = database_unavailable(exc)
        return jsonify({"error": str(failure), "code": failure.code}), failure.status_code
    return jsonify(view.to_dict())


@forecast_bp.route("/forecast/<run_id>/explanation", methods=["GET"])
def get_forecast_explanation(run_id: str):
    """Report findings related to the forecast limit and its cached LLM explanation."""
    rejected = _bad_id(run_id)
    if rejected is not None:
        return rejected
    return _explanation_response(lambda: explanation_view(run_id))


@forecast_bp.route("/forecast/<run_id>/explanation", methods=["POST"])
def post_forecast_explanation(run_id: str):
    """Asks the LLM to explain the limit; JSON only, so a cross-site form cannot start the call."""
    rejected = _bad_id(run_id)
    if rejected is not None:
        return rejected
    if not request.is_json:
        return jsonify({"error": "Нужен запрос с Content-Type: application/json", "code": "bad_request"}), 415
    owner = _owner()
    return _explanation_response(lambda: generate_explanation(run_id, budget_owner=owner))


@forecast_bp.route("/forecast/<run_id>/confluence", methods=["GET"])
def get_forecast_confluence(run_id: str):
    """Confluence page of the forecast report, if it was published."""
    rejected = _bad_id(run_id)
    if rejected is not None:
        return rejected
    try:
        publication = load_forecast_publication(run_id)
    except psycopg2.Error as exc:
        failure = database_unavailable(exc)
        return jsonify({"error": str(failure), "code": failure.code}), failure.status_code
    if publication is None:
        return jsonify({"run_id": run_id, "page_id": None, "page_url": None, "updated_at": None})
    return jsonify(asdict(publication))


def _failure_message(job_id: str) -> str:
    current = job_store.get(job_id)
    stage = (current.message or "").strip() if current else ""
    return f"Ошибка на этапе «{stage}»" if stage else "Ошибка публикации прогноза"


def _publish_in_background(job_id: str, run_id: str, inputs: ForecastInputs, source_url: str) -> None:
    """Starts loadlens_app.tasks.publish_forecast_task, which calls publish_forecast of this module.

    The job's runner decides where it runs: a job created with ``job_runner.create_job`` may go to the
    Celery worker, one created with ``job_store.create`` (a thread job) runs in a thread of this process.
    """
    job_runner.submit(
        tasks.publish_forecast_task, job_store.get(job_id) or job_id,
        run_id=run_id,
        target_rps=inputs.target_rps,
        headroom_pct=inputs.headroom_pct,
        cpu_ceiling_pct=inputs.cpu_ceiling_pct,
        source_url=source_url,
    )


@forecast_bp.route("/forecast/<run_id>/confluence", methods=["POST"])
def post_forecast_confluence(run_id: str):
    """Starts publishing the forecast report as its own Confluence page with the calculator settings."""
    rejected = _bad_id(run_id)
    if rejected is not None:
        return rejected
    if not request.is_json:
        return jsonify({"error": "Нужен запрос с Content-Type: application/json", "code": "bad_request"}), 415
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Тело запроса должно быть JSON-объектом", "code": "bad_request"}), 400
    try:
        inputs = _forecast_inputs(data)
    except InputError as exc:
        return jsonify({"error": str(exc), "code": "bad_request"}), 400
    run_name = str(data.get("run_name") or run_id).strip()[:MAX_RUN_NAME]
    owner = _owner()
    try:
        check_job_slot(*job_store.running_counts(owner))
    except LimitError as exc:
        return jsonify({"error": str(exc), "code": exc.code}), exc.status_code
    # The runner the job is registered for (Celery worker or a thread here) decides where it is published.
    job = job_runner.create_job(JOB_KIND_FORECAST_CONFLUENCE, run_name=run_name, message="Публикация прогноза в Confluence…", owner=owner)
    job_store.update(job.job_id, report_url=f"/forecasting/{run_id}")
    _publish_in_background(job.job_id, run_id, inputs, f"{public_base_url()}/forecasting/{run_id}")
    return jsonify({"status": "accepted", "job_id": job.job_id, "message": "Публикация прогноза в Confluence началась."}), 200
