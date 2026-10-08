"""LLM explanation of the forecast limit: built on request, cached per run until its inputs change."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from pydantic import ValidationError

from AI.context_pack import to_utc
from AI.db_store import RunAnalyses, StoredExplanation, StoredFinding, load_forecast_explanation, load_run_analyses, save_forecast_explanation
from AI.limit_explanation import (
    ExplanationCheck,
    LimitExplanation,
    LimitWindow,
    check_explanation,
    finding_payload,
    parse_explanation,
    related_findings,
)
from AI.providers import ask_llm_with_text_data, usage_domain
from loadlens_app.core import _ts_conn
from loadlens_app.forecast import ForecastReport, _storage, build_forecast
from settings import CONFIG

PROMPT_PATH = Path(__file__).resolve().parent.parent / "AI" / "prompts" / "forecast_limit_prompt.txt"
SYSTEM_PROMPT = "Вы инженер по производительности. Отвечайте строго в JSON по заданной схеме, все тексты — на русском языке."
DEFAULT_TABLE = "forecast_explanations"
DEFAULT_HEADROOM_PCT = 20.0
MIN_REGENERATE_SEC = 30.0
OLD_FORMAT_MESSAGE = "Разбор сделан в прежнем формате — обновите его"
TOP_SERVICES = 5
MAX_SYSTEM_ITEMS = 20

ViewStatus = Literal["unavailable", "missing", "ready", "stale"]


class ExplanationError(Exception):
    """The explanation cannot be built; ``code`` tells the page why."""

    def __init__(self, message: str, status_code: int, code: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class LlmChoice:
    provider: str
    settings: dict[str, Any]

    @property
    def model(self) -> str:
        return str(self.settings.get("model") or self.provider)


@dataclass(frozen=True)
class ExplanationContext:
    window: Optional[LimitWindow]
    findings: list[StoredFinding]
    payload: dict[str, Any]
    inputs_hash: str
    llm: Optional[LlmChoice]
    llm_problem: str


@dataclass(frozen=True)
class ExplanationView:
    status: ViewStatus
    message: str
    findings: list[StoredFinding]
    explanation: Optional[LimitExplanation]
    check: Optional[ExplanationCheck]
    model: str
    generated_at: Optional[str]
    llm_problem: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "findings": [asdict(finding) for finding in self.findings],
            "explanation": self.explanation.model_dump() if self.explanation is not None else None,
            "check": asdict(self.check) if self.check is not None else None,
            "model": self.model,
            "generated_at": self.generated_at,
            "llm_problem": self.llm_problem,
        }


_slots_lock = threading.Lock()
_running: set[str] = set()
_last_started: dict[str, float] = {}


@contextmanager
def _generation_slot(run_id: str) -> Iterator[None]:
    """One generation per run at a time and not more often than MIN_REGENERATE_SEC."""
    now = time.monotonic()
    with _slots_lock:
        if run_id in _running:
            raise ExplanationError("Разбор этого отчёта уже выполняется", 409, "busy")
        last = _last_started.get(run_id)
        if last is not None and now - last < MIN_REGENERATE_SEC:
            wait = int(MIN_REGENERATE_SEC - (now - last)) + 1
            raise ExplanationError(f"Повторный разбор возможен через {wait} с", 429, "too_often")
        _running.add(run_id)
        _last_started[run_id] = now
    try:
        yield
    finally:
        with _slots_lock:
            _running.discard(run_id)


def _hhmm(iso: str) -> str:
    return iso[11:16] if len(iso) >= 16 else iso


def _limit_window(forecast: ForecastReport) -> Optional[LimitWindow]:
    """From the step where stalls began (or the last step under a model ceiling) to the cutoff."""
    cutoff = to_utc(forecast.cutoff_iso)
    if forecast.instability is not None:
        onset = next((step for step in forecast.steps if step.number == forecast.instability.onset_step), None)
        return LimitWindow(to_utc(onset.start_iso if onset else forecast.instability.onset_iso), cutoff)
    steps = [step for step in forecast.steps if not step.after_drop]
    if forecast.limit.kind == "model" and steps:
        return LimitWindow(to_utc(steps[-1].start_iso), cutoff)
    return None


def _limit_facts(forecast: ForecastReport) -> dict[str, Any]:
    info = forecast.instability
    if info is not None:
        return {
            "kind": "провалы времени отклика",
            "rps": round(info.onset_rps),
            "onset_time": _hhmm(info.onset_iso),
            "onset_step": info.onset_label,
            "stall_points": info.stalls,
            "points": info.samples,
            "worst_mean_ms": round(info.worst_response_ms),
            "worst_p95_ms": round(info.worst_p95_ms) if info.worst_p95_ms is not None else None,
            "period_min": info.period_min,
        }
    return {"kind": "потолок модели USL", "rps": round(forecast.limit.rps) if forecast.limit.rps is not None else None}


def _cpu_facts(forecast: ForecastReport) -> dict[str, Any]:
    plan = forecast.resources
    if plan.status != "ok":
        return {"unavailable": plan.message}
    busiest = sorted(plan.services, key=lambda item: item.cpu_measured, reverse=True)[:TOP_SERVICES]
    return {
        "cause_service": plan.limit_cause,
        "busiest_at_limit": plan.limit_service,
        "busiest_cpu_at_limit_pct": round(plan.limit_cpu * 100, 1) if plan.limit_cpu is not None else None,
        "first_to_saturate": plan.bottleneck,
        "services_on_last_stable_step": [
            {"name": item.name, "instances": item.instances, "cpu_pct": round(item.cpu_measured * 100, 1)} for item in busiest
        ],
    }


def _step_fact(step: Any) -> dict[str, Any]:
    return {"step": step.label, "rps": round(step.rps), "mean_ms": round(step.response_ms, 1), "stall_points": step.stalls}


def _facts(forecast: ForecastReport) -> dict[str, Any]:
    steps = [step for step in forecast.steps if not step.after_drop]
    return {
        "test": {
            "run_name": forecast.run_name,
            "service": forecast.service,
            "window": f"{_hhmm(forecast.window_start_iso)}–{_hhmm(forecast.cutoff_iso)}",
            "point_minutes": forecast.bin_minutes,
        },
        "limit": _limit_facts(forecast),
        "steps": [_step_fact(step) for step in steps],
        "cpu": _cpu_facts(forecast),
        "usl": {"ceiling_rps": round(forecast.fit.x_max) if forecast.fit.x_max is not None else None, "r2": round(forecast.fit.r2, 2)},
    }


def _compact_system(snapshot: dict[str, Any]) -> dict[str, Any]:
    architecture = snapshot.get("architecture") if isinstance(snapshot.get("architecture"), dict) else {}
    operations = snapshot.get("operational_context") if isinstance(snapshot.get("operational_context"), dict) else {}
    compact = {
        "components": [
            {"id": item.get("id"), "role": item.get("role")}
            for item in architecture.get("components") or [] if isinstance(item, dict)
        ][:MAX_SYSTEM_ITEMS],
        "dependencies": [item for item in architecture.get("dependencies") or [] if isinstance(item, dict)][:MAX_SYSTEM_ITEMS],
        "data_stores": [item for item in architecture.get("data_stores") or [] if isinstance(item, dict)][:MAX_SYSTEM_ITEMS],
        "known_constraints": list(operations.get("known_constraints") or []),
        "known_risks": list(operations.get("known_risks") or []),
    }
    return {key: value for key, value in compact.items() if value}


def _llm_choice() -> tuple[Optional[LlmChoice], str]:
    """The provider the report pipeline calls (global ``llm`` settings), or the reason it is missing."""
    llm = CONFIG.get("llm") if isinstance(CONFIG.get("llm"), dict) else {}
    provider = str(llm.get("provider") or "").strip().lower()
    settings = llm.get(provider) if provider else None
    if not provider or not isinstance(settings, dict):
        return None, "Не настроен провайдер LLM: Настройки → LLM"
    return LlmChoice(provider=provider, settings=settings), ""


def _storage_names() -> tuple[str, str, str]:
    cfg = _storage()
    return str(cfg.get("schema") or "public"), str(cfg.get("llm_table") or "llm_reports"), str(cfg.get("forecast_table") or DEFAULT_TABLE)


def _analyses(run_id: str) -> RunAnalyses:
    schema, llm_table, _ = _storage_names()
    conn = _ts_conn()
    try:
        return load_run_analyses(conn, schema, llm_table, run_id)
    finally:
        conn.close()


def _context(run_id: str) -> ExplanationContext:
    forecast = build_forecast(run_id, None, DEFAULT_HEADROOM_PCT)
    window = _limit_window(forecast)
    analyses = _analyses(run_id)
    plan = forecast.resources
    services = [name for name in (plan.limit_cause, plan.bottleneck, plan.limit_service) if name]
    findings = related_findings(analyses.findings, window, services) if window is not None else []
    payload = {"facts": _facts(forecast), "findings": [finding_payload(item) for item in findings], "system": _compact_system(analyses.system_context)}
    digest = hashlib.sha256(json.dumps({"prompt": _prompt(), "payload": payload}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    llm, llm_problem = _llm_choice()
    return ExplanationContext(window, findings, payload, digest, llm, llm_problem)


def _prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _load_cached(run_id: str) -> Optional[StoredExplanation]:
    schema, _, table = _storage_names()
    conn = _ts_conn()
    try:
        return load_forecast_explanation(conn, schema, table, run_id)
    finally:
        conn.close()


def _save(run_id: str, inputs_hash: str, payload: dict[str, Any]) -> None:
    schema, _, table = _storage_names()
    conn = _ts_conn()
    try:
        save_forecast_explanation(conn, schema, table, run_id, inputs_hash, payload)
    finally:
        conn.close()


def _cached_view(status: ViewStatus, context: ExplanationContext, stored: StoredExplanation) -> ExplanationView:
    payload = stored.payload
    model = str(payload.get("model") or "")
    generated_at = str(payload.get("generated_at") or "") or None
    try:
        explanation = LimitExplanation.model_validate(payload["explanation"])
        check = ExplanationCheck(**payload["check"])
    except (KeyError, TypeError, ValidationError):
        return ExplanationView("stale", OLD_FORMAT_MESSAGE, context.findings, None, None, model, generated_at, context.llm_problem)
    return ExplanationView(
        status=status,
        message="Данные отчёта или настройки прогноза изменились после разбора — обновите его" if status == "stale" else "",
        findings=context.findings,
        explanation=explanation,
        check=check,
        model=model,
        generated_at=generated_at,
        llm_problem=context.llm_problem,
    )


def explanation_view(run_id: str) -> ExplanationView:
    """Related findings of the limit and the cached explanation, if any."""
    context = _context(run_id)
    if context.window is None:
        return ExplanationView("unavailable", "Предел по данным не виден — разбирать нечего", [], None, None, "", None, "")
    stored = _load_cached(run_id)
    if stored is None:
        model = context.llm.model if context.llm is not None else ""
        return ExplanationView("missing", "", context.findings, None, None, model, None, context.llm_problem)
    return _cached_view("ready" if stored.inputs_hash == context.inputs_hash else "stale", context, stored)


def _ask_model(context: ExplanationContext) -> str:
    llm = context.llm
    if llm is None:
        raise ExplanationError(context.llm_problem, 422, "missing_llm")
    try:
        with usage_domain("forecast"):
            return ask_llm_with_text_data(
                _prompt(),
                json.dumps(context.payload, ensure_ascii=False),
                llm_config={"force_json": True},
                system_prompt=SYSTEM_PROMPT,
            )
    except Exception as exc:
        raise ExplanationError(f"Модель {llm.provider} / {llm.model} не ответила: {exc}", 502, "llm_failed") from exc


def generate_explanation(run_id: str) -> ExplanationView:
    """Asks the model, checks its numbers and caches the result for the run."""
    with _generation_slot(run_id):
        context = _context(run_id)
        if context.window is None:
            raise ExplanationError("Предел по данным не виден — разбирать нечего", 422, "no_limit")
        raw = _ask_model(context)
        explanation = parse_explanation(raw)
        check = check_explanation(explanation, context.payload["facts"], context.findings)
        payload = {
            "explanation": explanation.model_dump(),
            "check": asdict(check),
            "model": context.llm.model if context.llm is not None else "",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(run_id, context.inputs_hash, payload)
        stored = StoredExplanation(run_id=run_id, inputs_hash=context.inputs_hash, payload=payload, created_at=None)
        return _cached_view("ready", context, stored)


__all__ = ["ExplanationError", "ExplanationView", "explanation_view", "generate_explanation"]
