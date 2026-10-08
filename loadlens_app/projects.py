"""Project areas for the settings page: overview, card edits, copies, service moves, overrides, deletion."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2 import sql

from AI.db_store import _ensure_llm_reports_table
from AI.pipeline import _has_meaningful_system_context
from loadlens_app.core import (
    _active_metrics_config,
    _deep_merge_dicts,
    _delete_service_data,
    _list_project_areas,
    _load_base_prompts,
    _load_metrics_runtime_data,
    _load_settings_runtime_data,
    _metrics_services_for_area,
    _normalize_system_context,
    _save_metrics_runtime_data,
    _save_settings_runtime_data,
    _ts_conn,
)
from settings import CONFIG

PROJECT_ID_MAX_LEN = 40
TITLE_MAX_LEN = 80
DESCRIPTION_MAX_LEN = 500
FORBIDDEN_ID_CHARS = re.compile(r"[\s/\\<>]")
AREA_SECTIONS: tuple[str, ...] = (
    "llm", "domain_sources", "logs_source",
    "default_params", "queries", "sla", "system_context", "prompts",
)
SERVICE_SECTIONS: tuple[str, ...] = ("queries", "sla", "prompts")
AREA_META_KEYS = frozenset({"title", "description", "services"})
STATUS_INHERITED = "inherited"
STATUS_SAME = "same"
STATUS_OWN = "own"
REPORT_ROWS_SQL = sql.SQL(
    "SELECT service, created_at, COALESCE(sla_verdict, verdict, 'Недостаточно данных') FROM ("
    "SELECT service, created_at, sla_verdict, verdict, "
    "ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn "
    "FROM {table} WHERE domain = 'final' AND service = ANY(%s)) latest WHERE rn = 1"
)
SET_REPORTS_AREA_SQL = sql.SQL("UPDATE {table} SET project_area = %s WHERE service = ANY(%s)")


class ProjectError(Exception):
    """Rejected project operation; ``status_code`` is the HTTP status for the API."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SectionOverride:
    section: str
    status: str
    own_domains: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Readiness:
    target_rps: Optional[float]
    performance_query: str
    has_system_context: bool
    query_counts: dict[str, int]


@dataclass(frozen=True)
class ServiceSummary:
    id: str
    title: str
    reports: Optional[int]
    last_report_at: Optional[str]
    own_sections: list[str]


@dataclass(frozen=True)
class ProjectSummary:
    id: str
    title: str
    description: str
    services: list[ServiceSummary]
    reports: Optional[int]
    last_report_at: Optional[str]
    last_verdict: Optional[str]
    readiness: Readiness
    overrides: list[SectionOverride]


@dataclass(frozen=True)
class ProjectsOverview:
    projects: list[ProjectSummary]
    active_project: str
    reports_error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReportStats:
    reports: int
    last_report_at: Optional[datetime]
    last_verdict: Optional[str]


def _area_entries() -> dict:
    per_area = _load_settings_runtime_data().get("per_area")
    return per_area if isinstance(per_area, dict) else {}


def _area_entry(project_id: str) -> dict:
    entry = _area_entries().get(project_id)
    return entry if isinstance(entry, dict) else {}


def _project_ids() -> list[str]:
    return [str(item["id"]) for item in _list_project_areas()]


def _require_project(project_id: str) -> None:
    if project_id not in _project_ids():
        raise ProjectError(f"Проект «{project_id}» не найден", 404)


def _service_titles(project_id: str) -> dict[str, str]:
    """Services of the project: runtime entries first, then services known only from metrics_config."""
    services = _area_entry(project_id).get("services")
    services = services if isinstance(services, dict) else {}
    titles: dict[str, str] = {}
    for service_id in [*services.keys(), *_metrics_services_for_area(project_id).keys()]:
        meta = services.get(service_id) if isinstance(services.get(service_id), dict) else {}
        titles.setdefault(str(service_id), str(meta.get("title") or service_id))
    return titles


def _llm_table() -> sql.Composed:
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {}
    return sql.SQL("{}.{}").format(
        sql.Identifier(cfg.get("schema", "public")), sql.Identifier(cfg.get("llm_table", "llm_reports"))
    )


def _report_stats(service_ids: list[str]) -> dict[str, ReportStats]:
    """Latest final row of every run of the given services, grouped by service."""
    if not service_ids:
        return {}
    conn = _ts_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(REPORT_ROWS_SQL.format(table=_llm_table()), (service_ids,))
            rows = cur.fetchall()
    finally:
        conn.close()
    stats: dict[str, ReportStats] = {}
    for service, created_at, verdict in rows:
        current = stats.get(service)
        newer = current is None or (
            created_at is not None and (current.last_report_at is None or created_at > current.last_report_at)
        )
        stats[service] = ReportStats(
            reports=(current.reports if current else 0) + 1,
            last_report_at=created_at if newer else current.last_report_at,
            last_verdict=verdict if newer else current.last_verdict,
        )
    return stats


def _merged_section(area_entry: dict, section: str) -> dict:
    """Area-level value the reports use: the global section merged with the area override."""
    base = CONFIG.get(section) if isinstance(CONFIG.get(section), dict) else {}
    override = area_entry.get(section) if isinstance(area_entry.get(section), dict) else {}
    return _deep_merge_dicts(base, override)


def _query_count(domain_cfg: object) -> int:
    labels = domain_cfg.get("labels") if isinstance(domain_cfg, dict) else None
    return len(labels) if isinstance(labels, list) else 0


def _readiness(area_entry: dict) -> Readiness:
    sla = _merged_section(area_entry, "sla")
    target = sla.get("target_rps")
    return Readiness(
        target_rps=float(target) if isinstance(target, (int, float)) and not isinstance(target, bool) else None,
        performance_query=str(sla.get("max_performance_query") or "").strip(),
        has_system_context=_has_meaningful_system_context(_merged_section(area_entry, "system_context")),
        query_counts={domain: _query_count(cfg) for domain, cfg in _merged_section(area_entry, "queries").items()},
    )


def _prompts_status(prompts: object) -> SectionOverride:
    base = _load_base_prompts()
    items = prompts.items() if isinstance(prompts, dict) else []
    own = sorted(
        domain for domain, text in items
        if isinstance(text, str) and text.strip() and text != base.get(domain, "")
    )
    return SectionOverride("prompts", STATUS_OWN if own else STATUS_SAME, own)


def _section_status(area_entry: dict, section: str) -> SectionOverride:
    """inherited: no value in the project; same: a copy equal to the global value; own: differs."""
    if section not in area_entry:
        return SectionOverride(section, STATUS_INHERITED)
    value = area_entry.get(section)
    if section == "prompts":
        return _prompts_status(value)
    if section == "system_context":
        same = _normalize_system_context(value) == _normalize_system_context({})
    else:
        same = value == (CONFIG.get(section) or {})
    return SectionOverride(section, STATUS_SAME if same else STATUS_OWN)


def _service_summary(
    service_id: str, title: str, area_entry: dict, stats: Optional[dict[str, ReportStats]]
) -> ServiceSummary:
    services = area_entry.get("services") if isinstance(area_entry.get("services"), dict) else {}
    entry = services.get(service_id) if isinstance(services.get(service_id), dict) else {}
    item = (stats or {}).get(service_id)
    return ServiceSummary(
        id=service_id,
        title=title,
        reports=None if stats is None else (item.reports if item else 0),
        last_report_at=item.last_report_at.isoformat() if item and item.last_report_at else None,
        own_sections=[section for section in SERVICE_SECTIONS if entry.get(section)],
    )


def _project_summary(
    project_id: str, area_entry: dict, services: dict[str, str], stats: Optional[dict[str, ReportStats]]
) -> ProjectSummary:
    own = [stats[sid] for sid in services if stats and sid in stats]
    latest = max(own, key=lambda item: item.last_report_at.timestamp() if item.last_report_at else 0.0, default=None)
    return ProjectSummary(
        id=project_id,
        title=str(area_entry.get("title") or project_id),
        description=str(area_entry.get("description") or ""),
        services=[_service_summary(sid, title, area_entry, stats) for sid, title in services.items()],
        reports=None if stats is None else sum(item.reports for item in own),
        last_report_at=latest.last_report_at.isoformat() if latest and latest.last_report_at else None,
        last_verdict=latest.last_verdict if latest else None,
        readiness=_readiness(area_entry),
        overrides=[_section_status(area_entry, section) for section in AREA_SECTIONS],
    )


def list_projects(active_project: str) -> ProjectsOverview:
    """All projects with services, report stats, readiness and overrides."""
    entries = _area_entries()
    services = {project_id: _service_titles(project_id) for project_id in _project_ids()}
    try:
        stats: Optional[dict[str, ReportStats]] = _report_stats(sorted({sid for titles in services.values() for sid in titles}))
        reports_error: Optional[str] = None
    except psycopg2.Error as exc:
        stats, reports_error = None, f"Отчёты недоступны: {exc}"
    projects = [
        _project_summary(pid, entries.get(pid) if isinstance(entries.get(pid), dict) else {}, titles, stats)
        for pid, titles in services.items()
    ]
    return ProjectsOverview(projects=projects, active_project=active_project, reports_error=reports_error)


def _validate_project_id(value: str) -> str:
    project_id = str(value or "").strip()
    if not project_id:
        raise ProjectError("Укажите идентификатор проекта")
    if len(project_id) > PROJECT_ID_MAX_LEN or FORBIDDEN_ID_CHARS.search(project_id):
        raise ProjectError(
            f"Идентификатор «{project_id}»: до {PROJECT_ID_MAX_LEN} символов, без пробелов, слэшей и знаков < >"
        )
    return project_id


def _clean_text_field(value: str, limit: int, label: str) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise ProjectError(f"{label}: не длиннее {limit} символов, сейчас {len(text)}")
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in text):
        raise ProjectError(f"{label}: недопустимые управляющие символы")
    return text


def create_project(project_id: str, title: str, description: str, copy_from: str) -> str:
    """New project; with ``copy_from`` its settings are copied, services and reports are not."""
    new_id = _validate_project_id(project_id)
    clean_title = _clean_text_field(title, TITLE_MAX_LEN, "Отображаемое имя")
    clean_description = _clean_text_field(description, DESCRIPTION_MAX_LEN, "Описание")
    existing = set(_project_ids()) | set(_active_metrics_config().keys())
    if new_id in existing:
        raise ProjectError(f"Проект «{new_id}» уже существует", 409)
    if copy_from and copy_from not in existing:
        raise ProjectError(f"Проект-источник «{copy_from}» не найден", 404)
    copied = (
        {key: copy.deepcopy(value) for key, value in _area_entry(copy_from).items() if key not in AREA_META_KEYS}
        if copy_from else {}
    )
    runtime = _load_settings_runtime_data()
    runtime.setdefault("per_area", {})[new_id] = {**copied, "title": clean_title, "description": clean_description}
    _save_settings_runtime_data(runtime)
    metrics = _load_metrics_runtime_data()
    metrics[new_id] = {"services": {}}
    _save_metrics_runtime_data(metrics)
    return new_id


def update_project(project_id: str, title: str, description: str) -> None:
    _require_project(project_id)
    clean_title = _clean_text_field(title, TITLE_MAX_LEN, "Отображаемое имя")
    clean_description = _clean_text_field(description, DESCRIPTION_MAX_LEN, "Описание")
    runtime = _load_settings_runtime_data()
    entry = runtime.setdefault("per_area", {}).setdefault(project_id, {})
    entry["title"] = clean_title
    entry["description"] = clean_description
    _save_settings_runtime_data(runtime)


def _set_reports_area(service_ids: list[str], project_id: Optional[str]) -> int:
    """The baseline lookup filters reports by project_area; None makes them eligible for any project."""
    conn = _ts_conn()
    try:
        _ensure_llm_reports_table(conn, (CONFIG.get("storage", {}) or {}).get("timescale", {}) or {})
        with conn, conn.cursor() as cur:
            cur.execute(SET_REPORTS_AREA_SQL.format(table=_llm_table()), (project_id, service_ids))
            return int(cur.rowcount or 0)
    finally:
        conn.close()


def _move_service_settings(project_id: str, service_id: str, target_id: str) -> None:
    runtime = _load_settings_runtime_data()
    per_area = runtime.setdefault("per_area", {})
    entry = per_area.setdefault(project_id, {}).setdefault("services", {}).pop(service_id, {})
    per_area.setdefault(target_id, {}).setdefault("services", {})[service_id] = entry if isinstance(entry, dict) else {}
    _save_settings_runtime_data(runtime)


def _move_service_metrics(project_id: str, service_id: str, target_id: str) -> None:
    metrics = _load_metrics_runtime_data()
    source = metrics.get(project_id) if isinstance(metrics.get(project_id), dict) else {}
    source_services = source.get("services") if isinstance(source.get("services"), dict) else {}
    entry = source_services.pop(service_id, None) or _metrics_services_for_area(project_id).get(service_id)
    if entry is None:
        return
    metrics.setdefault(target_id, {}).setdefault("services", {})[service_id] = entry
    _save_metrics_runtime_data(metrics)


def move_service(project_id: str, service_id: str, target_id: str) -> int:
    """Moves a service with its settings and reports; returns the number of updated report rows."""
    if not target_id:
        raise ProjectError("Укажите проект, в который перенести сервис")
    _require_project(project_id)
    _require_project(target_id)
    if target_id == project_id:
        raise ProjectError(f"Сервис «{service_id}» уже в проекте «{project_id}»")
    if service_id not in _service_titles(project_id):
        raise ProjectError(f"Сервиса «{service_id}» нет в проекте «{project_id}»", 404)
    if service_id in _service_titles(target_id):
        raise ProjectError(f"В проекте «{target_id}» уже есть сервис «{service_id}»", 409)
    updated = _set_reports_area([service_id], target_id)
    _move_service_settings(project_id, service_id, target_id)
    _move_service_metrics(project_id, service_id, target_id)
    return updated


def _forget_in_memory(project_id: str, section: Optional[str]) -> None:
    """config_api._area_section falls back to CONFIG['per_area']; a stale in-memory copy would outlive the file."""
    per_area = CONFIG.get("per_area")
    entry = per_area.get(project_id) if isinstance(per_area, dict) else None
    if not isinstance(entry, dict):
        return
    if section is None:
        per_area.pop(project_id, None)
    else:
        entry.pop(section, None)


def reset_area_section(project_id: str, section: str) -> bool:
    """Drops the project's own value of a section so the global one applies; returns whether it existed."""
    _require_project(project_id)
    if section not in AREA_SECTIONS:
        raise ProjectError(f"Раздел «{section}» нельзя сбросить. Допустимо: {', '.join(AREA_SECTIONS)}")
    runtime = _load_settings_runtime_data()
    entry = (runtime.get("per_area") or {}).get(project_id)
    changed = isinstance(entry, dict) and entry.pop(section, None) is not None
    if changed:
        _save_settings_runtime_data(runtime)
    _forget_in_memory(project_id, section)
    return changed


def reset_service_section(project_id: str, service_id: str, section: str) -> bool:
    """Drops a service override so the project value applies; returns whether it existed."""
    _require_project(project_id)
    if section not in SERVICE_SECTIONS:
        raise ProjectError(f"Раздел «{section}» сервиса нельзя сбросить. Допустимо: {', '.join(SERVICE_SECTIONS)}")
    runtime = _load_settings_runtime_data()
    services = ((runtime.get("per_area") or {}).get(project_id) or {}).get("services") or {}
    entry = services.get(service_id)
    if not isinstance(entry, dict):
        raise ProjectError(f"У сервиса «{service_id}» нет своих настроек в проекте «{project_id}»", 404)
    changed = entry.pop(section, None) is not None
    if changed:
        _save_settings_runtime_data(runtime)
    return changed


def _drop_project_settings(project_id: str) -> None:
    runtime = _load_settings_runtime_data()
    per_area = runtime.get("per_area")
    if isinstance(per_area, dict) and per_area.pop(project_id, None) is not None:
        _save_settings_runtime_data(runtime)
    metrics = _load_metrics_runtime_data()
    if metrics.pop(project_id, None) is not None:
        _save_metrics_runtime_data(metrics)
    _forget_in_memory(project_id, None)


def delete_project(project_id: str, with_data: bool) -> list[str]:
    """Deletes the project. Without data its reports stay and are shown under «Все области»."""
    _require_project(project_id)
    service_ids = list(_service_titles(project_id))
    if with_data:
        for service_id in service_ids:
            _delete_service_data(project_id, service_id)
    elif service_ids:
        _set_reports_area(service_ids, None)
    _drop_project_settings(project_id)
    return service_ids
