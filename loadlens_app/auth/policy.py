"""Who may call what: one table keyed by Flask endpoint ("blueprint.function").

Keeping the rules in one place makes the access model reviewable at a glance. An endpoint missing
from the table is treated as admin-only, and a test fails until every registered endpoint is listed.

Role guide:
  viewer    reads reports, archive, comparison and forecasts
  engineer  also runs analyses and load tests, publishes to Confluence, writes notes and votes
  admin     also changes settings, connections and secrets, manages users, deletes data

Settings, connections, projects, prompts and deletions are ``session_only``: an API token, even an
administrator's, is refused there, so a leaked token can neither reach stored secrets nor destroy data.
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
    recovery_ok: bool = False  # reachable by a recovery session while the storage database is down


PUBLIC = Rule(None)
_DEFAULT = Rule(A)


def _read(role: Role = V, **kwargs) -> Rule:
    return Rule(role, **kwargs)


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
    "dashboard.demo_seed": _write(A, "demo.seed", session_only=True),
    "dashboard.llm_reports": _read(),
    "dashboard.llm_context": _read(),
    "dashboard.llm_feedback_get": _read(),
    "dashboard.llm_feedback_put": _write(E),
    "dashboard.list_runs": _read(),
    "dashboard.create_report": _write(E, "report.create"),
    "dashboard.job_status": _read(),
    "dashboard.list_jobs": _read(),
    "dashboard.dismiss_jobs": _write(E, "jobs.dismiss"),  # hides finished jobs from the shared list
    "dashboard.delete_run": _write(A, "run.delete", session_only=True),
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
    "projects.post_project": _write(A, "project.create", session_only=True),
    "projects.patch_project": _write(A, "project.update", session_only=True),
    "projects.remove_project": _write(A, "project.delete", session_only=True),
    "projects.post_move_service": _write(A, "project.move_service", session_only=True),
    "projects.delete_area_override": _write(A, "project.override_delete", session_only=True),
    "projects.delete_service_override": _write(A, "project.override_delete", session_only=True),
    # ---- settings: prompts and service metadata ----
    "settings.settings_page": _page(A, session_only=True),
    "settings.delete_service": _write(A, "service.delete", session_only=True),
    "settings.get_prompts": _read(A, session_only=True),
    "settings.post_prompts": _write(A, "prompts.save", session_only=True),
    "settings.get_prompt_defaults": _read(A, session_only=True),
    "settings.get_prompt_history": _read(A, session_only=True),
    "settings.update_service_meta": _write(A, "service.meta_update", session_only=True),
    "settings.delete_service_meta": _write(A, "service.meta_delete", session_only=True),
    # ---- configuration and connection checks (these use stored secrets) ----
    "config_api.get_config": _read(A, session_only=True),
    "config_api.update_config": _write(A, "config.update", session_only=True),
    "config_api.test_connection": _write(A, "config.test_connection", session_only=True),
    "config_api.grafana_datasources": _read(A, session_only=True),
    "config_api.llm_models": _write(A, "config.llm_models", session_only=True),
    "config_api.query_preview": _write(A, "config.query_preview", session_only=True),
    # Sends a document to the LLM provider and returns a draft; nothing is saved.
    "config_api.import_system_context": _write(A, "config.context_import", session_only=True),
    # ---- recovery while the storage database is down (also open to administrators otherwise) ----
    "recovery.recovery_page": _page(A, session_only=True, recovery_ok=True),
    "recovery.recovery_status": _read(A, session_only=True, recovery_ok=True),
    "recovery.storage_check": _write(A, "recovery.storage_check", session_only=True, recovery_ok=True),
    "recovery.storage_save": _write(A, "config.update", session_only=True, recovery_ok=True),
    # ---- query packs (render and check run against stored connections) ----
    "query_packs.list_query_packs": _read(A, session_only=True),
    "query_packs.preview_query_pack": _read(A, session_only=True),
    "query_packs.check_query_pack": _write(A, "config.query_pack_check", session_only=True),
    "query_packs.detect_query_packs": _write(A, "config.query_pack_detect", session_only=True),
    "query_packs.apply_query_pack": _write(A, "config.query_pack_apply", session_only=True),
    "query_packs.summed_rps": _write(A, "config.summed_rps", session_only=True),
    # ---- load tests started from LoadLens (commands run on the configured generators) ----
    # Engineers only: commands and parameters may name internal hosts or carry test credentials.
    "load_tests.launch_page": _page(E),
    "load_tests.run_page": _page(E),
    "load_tests.options": _read(E),
    "load_tests.scripts": _read(E),
    "load_tests.ci_params": _read(E),
    "load_tests.ci_refs": _read(E),
    "load_tests.preview": _write(E),  # validates the form and renders commands, starts nothing
    "load_tests.create": _write(E, "load_test.start"),
    "load_tests.list_runs": _read(E),
    "load_tests.active": _read(E),
    "load_tests.get_run": _read(E),
    "load_tests.stop": _write(E, "load_test.stop"),
    "load_tests.logs": _read(E),
    "load_tests.live": _read(E),
    # Test window hint of the new report form: queries the service's load tool metrics like "live" does.
    "load_tests.lt_windows": _read(E),
    "load_tests.hints": _read(E),
    "load_tests.list_presets": _read(E),
    "load_tests.save_preset": _write(E, "load_test.preset_save"),
    "load_tests.delete_preset": _write(E, "load_test.preset_delete"),
    # CI pipelines report events with the secret of their instance, not a user session.
    "load_tests.ci_events": Rule(None, csrf=False),
}


def rule_for(endpoint: Optional[str]) -> Rule:
    return POLICY.get(endpoint or "", _DEFAULT)


def unclassified(endpoints: Iterable[str]) -> list[str]:
    return sorted(name for name in endpoints if name not in POLICY)
