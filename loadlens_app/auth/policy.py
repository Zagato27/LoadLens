"""Who may call what: one table keyed by Flask endpoint ("blueprint.function").

Keeping the rules in one place makes the access model reviewable at a glance. An endpoint missing
from the table is treated as admin-only, and a test fails until every registered endpoint is listed.

Role guide:
  viewer    reads reports, archive, comparison and forecasts
  engineer  also runs analyses, publishes to Confluence, writes notes and votes
  admin     also changes settings, connections and secrets, manages users, deletes data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .models import Role

V, E, A = Role.VIEWER, Role.ENGINEER, Role.ADMIN


@dataclass(frozen=True)
class Rule:
    role: Optional[Role]  # None: reachable without signing in
    page: bool = False  # HTML page: unauthenticated users are redirected to the login form
    session_only: bool = False  # refuses API tokens (credential and user management)
    audit: str = ""  # audit action recorded after a successful state-changing request
    csrf: bool = True  # unsafe methods need the CSRF token unless the request uses an API token
    password_change_ok: bool = False  # reachable while a password change is being enforced


PUBLIC = Rule(None)
_DEFAULT = Rule(A)


def _read(role: Role = V) -> Rule:
    return Rule(role)


def _page(role: Role = V, **kwargs) -> Rule:
    return Rule(role, page=True, **kwargs)


def _write(role: Role, audit: str = "", **kwargs) -> Rule:
    return Rule(role, audit=audit, **kwargs)


POLICY: dict[str, Rule] = {
    # ---- public ----
    "static": PUBLIC,
    "appearance.logo": PUBLIC,
    "appearance.palette": PUBLIC,
    "auth.healthz": PUBLIC,
    "auth.login_page": PUBLIC,
    "auth.login_submit": PUBLIC,
    "auth.logout": Rule(None, csrf=False, password_change_ok=True),
    "auth.me": Rule(None, password_change_ok=True),
    # ---- account (any signed-in user) ----
    "auth.account_page": _page(password_change_ok=True),
    "auth.change_password": Rule(V, session_only=True, password_change_ok=True),
    "auth.list_tokens": Rule(V, session_only=True),
    "auth.create_token": Rule(V, session_only=True),
    "auth.revoke_token": Rule(V, session_only=True),
    # ---- user administration ----
    "auth.list_users": Rule(A, session_only=True),
    "auth.create_user": Rule(A, session_only=True),
    "auth.update_user": Rule(A, session_only=True),
    "auth.delete_user": Rule(A, session_only=True),
    "auth.reset_user_password": Rule(A, session_only=True),
    "auth.unlock_user": Rule(A, session_only=True),
    "auth.audit_log": Rule(A, session_only=True),
    # ---- dashboard / archive / reports ----
    "dashboard.home": _page(),
    "dashboard.reports_page": _page(),
    "dashboard.report_ref": _read(),
    "dashboard.reports_page_run": _page(),
    "dashboard.reports_page_service_run": _page(),
    "dashboard.new_report_page": _page(E),
    "dashboard.get_services": _read(),
    "dashboard.dashboard_data": _read(),
    "dashboard.demo_seed": _write(A, "demo.seed"),
    "dashboard.llm_reports": _read(),
    "dashboard.llm_context": _read(),
    "dashboard.llm_feedback_get": _read(),
    "dashboard.llm_feedback_put": _write(E),
    "dashboard.list_runs": _read(),
    "dashboard.create_report": _write(E, "report.create"),
    "dashboard.job_status": _read(),
    "dashboard.list_jobs": _read(),
    "dashboard.delete_run": _write(A, "run.delete"),
    "dashboard.rename_run": _write(E, "run.rename"),
    "dashboard.project_areas": _read(),
    "dashboard.current_project_area": _read(),
    "dashboard.set_project_area": _write(V),  # a UI preference stored in a cookie
    "dashboard.get_engineer_summary": _read(),
    "dashboard.post_engineer_summary": _write(E, "engineer_summary.save"),
    "dashboard.get_confluence_publication": _read(),
    "dashboard.publish_confluence": _write(E, "confluence.publish"),
    # ---- comparison ----
    "compare.compare_page": _page(),
    "compare.compare_summary": _read(),
    "compare.compare_metric_summary": _read(),
    "compare.compare_baseline": _read(),
    "compare.domains_schema": _read(),
    "compare.compare_series": _read(),
    "compare.run_series": _read(),
    # ---- forecasting ----
    "forecast.forecasting_page": _page(),
    "forecast.forecasting_report_page": _page(),
    "forecast.get_forecast_reports": _read(),
    "forecast.get_forecast_status": _read(),
    "forecast.get_forecast": _read(),
    "forecast.get_forecast_explanation": _read(),
    "forecast.post_forecast_explanation": _write(E, "forecast.explain"),
    "forecast.get_forecast_confluence": _read(),
    "forecast.post_forecast_confluence": _write(E, "forecast.publish"),
    # ---- projects ----
    "projects.get_projects": _read(),
    "projects.post_project": _write(A, "project.create"),
    "projects.patch_project": _write(A, "project.update"),
    "projects.remove_project": _write(A, "project.delete"),
    "projects.post_move_service": _write(A, "project.move_service"),
    "projects.delete_area_override": _write(A, "project.override_delete"),
    "projects.delete_service_override": _write(A, "project.override_delete"),
    # ---- settings: prompts and service metadata ----
    "settings.settings_page": _page(A),
    "settings.delete_service": _write(A, "service.delete"),
    "settings.get_prompts": _read(A),
    "settings.post_prompts": _write(A, "prompts.save"),
    "settings.get_prompt_defaults": _read(A),
    "settings.get_prompt_history": _read(A),
    "settings.update_service_meta": _write(A, "service.meta_update"),
    "settings.delete_service_meta": _write(A, "service.meta_delete"),
    # ---- configuration and connection checks (these use stored secrets) ----
    "config_api.get_config": _read(A),
    "config_api.update_config": _write(A, "config.update"),
    "config_api.test_connection": _write(A, "config.test_connection"),
    "config_api.grafana_datasources": _read(A),
    "config_api.llm_models": _write(A),
    "config_api.query_preview": _write(A, "config.query_preview"),
}


def rule_for(endpoint: Optional[str]) -> Rule:
    return POLICY.get(endpoint or "", _DEFAULT)


def unclassified(endpoints: Iterable[str]) -> list[str]:
    return sorted(name for name in endpoints if name not in POLICY)
