# SPDX-License-Identifier: Apache-2.0
"""Connectivity checks for external systems, run from the settings UI.

Each check receives the configuration section as edited in the form (secrets
already restored), talks to the target with a short timeout and returns a
:class:`CheckResult` with a human-readable message in the locale of the request
(``server.connection.*`` keys). Checks never raise: failures are reported in the result so the UI can
show them inline.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict

import psycopg2
import requests

from AI.data_sources import ResolvedSource
from AI.i18n import t
from AI.pipeline import _resolve_grafana_influx_ds_id, _resolve_grafana_prom_ds_id
from AI.providers import SUPPORTED_PROVIDERS, _get_gigachat_client, _normalize_llm_base_url, ask_llm_with_text_data
from loadlens_app.confluence_export import ConfluenceClient

CHECK_TIMEOUT_SEC = 10
LLM_CHECK_TIMEOUT_SEC = 45

CHECKABLE_SECTIONS = (
    "storage", "storage.timescale", "data_sources", "domain_sources",
    "logs_source", "confluence", "confluence_template", "llm", "load_generators",
)


@dataclass
class CheckResult:
    ok: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _cfg(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _explain(exc: Exception) -> str:
    """Maps common transport/auth failures to actionable messages."""
    if isinstance(exc, requests.exceptions.Timeout):
        return t("server.connection.timeout", seconds=CHECK_TIMEOUT_SEC)
    if isinstance(exc, requests.exceptions.SSLError):
        return t("server.connection.tls", error=str(exc))
    if isinstance(exc, requests.exceptions.ConnectionError):
        return t("server.connection.connect_failed", error=str(exc))
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            return t("server.connection.access_denied", status=status)
        if status == 404:
            return t("server.connection.not_found")
        return t("server.connection.http_error", status=status, error=str(exc))
    if isinstance(exc, psycopg2.OperationalError):
        text = str(exc)
        if "password authentication failed" in text:
            return t("server.connection.db_auth")
        if "does not exist" in text:
            return t("server.connection.db_missing", error=text.strip())
        if "could not connect" in text or "Connection refused" in text or "timeout expired" in text:
            return t("server.connection.db_unreachable", error=text.strip())
        return text.strip()
    return str(exc) or exc.__class__.__name__


def _timed(fn: Callable[[], CheckResult]) -> CheckResult:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # checks report failures instead of raising
        elapsed = int((time.perf_counter() - started) * 1000)
        return CheckResult(ok=False, message=_explain(exc), details={"error": str(exc), "error_type": exc.__class__.__name__}, elapsed_ms=elapsed)
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


# ---- individual checks -----------------------------------------------------

def check_timescale(cfg: dict) -> CheckResult:
    def run() -> CheckResult:
        conn = psycopg2.connect(
            host=cfg.get("host"),
            port=int(cfg.get("port") or 5432),
            dbname=cfg.get("dbname"),
            user=cfg.get("user"),
            password=cfg.get("password"),
            sslmode=cfg.get("sslmode", "prefer"),
            connect_timeout=CHECK_TIMEOUT_SEC,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = str((cur.fetchone() or [""])[0]).split(" on ")[0]
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                ext_row = cur.fetchone()
                schema = str(cfg.get("schema") or "public")
                expected = [str(cfg.get("table") or "metrics"), str(cfg.get("llm_table") or "llm_reports"), str(cfg.get("engineer_table") or "engineer_reports")]
                cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_name = ANY(%s)",
                    (schema, expected),
                )
                present = sorted(str(r[0]) for r in cur.fetchall())
        finally:
            conn.close()
        missing = [table for table in expected if table not in present]
        timescale = f"TimescaleDB {ext_row[0]}" if ext_row else t("server.connection.no_timescale")
        message = (
            t("server.connection.storage_ok_missing", version=version, timescale=timescale, tables=", ".join(missing))
            if missing
            else t("server.connection.storage_ok", version=version, timescale=timescale)
        )
        return CheckResult(ok=True, message=message, details={"version": version, "timescaledb": ext_row[0] if ext_row else None, "tables_present": present, "tables_missing": missing})
    return _timed(run)


def _grafana_auth(g_cfg: dict) -> tuple[dict, tuple | None]:
    auth_cfg = _cfg(g_cfg.get("auth"))
    method = str(auth_cfg.get("method") or "basic").lower()
    headers: dict = {}
    auth = None
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    return headers, auth


def _grafana_login(g_cfg: dict) -> CheckResult | tuple[str, dict]:
    """Health and ``/api/user``. A ``CheckResult`` means the login itself failed."""
    base_url = str(g_cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        return CheckResult(ok=False, message=t("server.connection.grafana_url_missing"))
    verify = bool(g_cfg.get("verify_ssl", True))
    health = requests.get(f"{base_url}/api/health", timeout=CHECK_TIMEOUT_SEC, verify=verify)
    health.raise_for_status()
    version = str((health.json() or {}).get("version") or "?")
    headers, auth = _grafana_auth(g_cfg)
    me = requests.get(f"{base_url}/api/user", headers=headers, auth=auth, timeout=CHECK_TIMEOUT_SEC, verify=verify)
    me.raise_for_status()
    return version, {"version": version}


def check_grafana_auth(g_cfg: dict) -> CheckResult:
    """Grafana is reachable and the credentials are accepted. Does not look up a datasource."""
    def run() -> CheckResult:
        logged = _grafana_login(g_cfg)
        if isinstance(logged, CheckResult):
            return logged
        version, details = logged
        return CheckResult(ok=True, message=t("server.connection.grafana_auth_ok", version=version), details=details)
    return _timed(run)


def check_grafana(g_cfg: dict, datasource_kind: str) -> CheckResult:
    def run() -> CheckResult:
        logged = _grafana_login(g_cfg)
        if isinstance(logged, CheckResult):
            return logged
        version, details = logged
        resolver = _resolve_grafana_influx_ds_id if datasource_kind == "influxdb" else _resolve_grafana_prom_ds_id
        ds_id = resolver(g_cfg)
        ds_cfg = _cfg(g_cfg.get("prometheus_datasource")) or _cfg(g_cfg.get("influxdb_datasource"))
        ds_name = ds_cfg.get("name") or ds_cfg.get("uid") or f"id={ds_id}"
        id_note = f" (id={ds_id})" if ds_id is not None else ""
        return CheckResult(
            ok=True,
            message=t("server.connection.grafana_datasource_ok", version=version, datasource=ds_name, id_note=id_note),
            details={**details, "datasource_id": ds_id},
        )
    return _timed(run)


@dataclass(frozen=True)
class GrafanaDatasource:
    """One Prometheus or InfluxDB datasource of Grafana. ``mode`` is the query language it serves."""

    uid: str
    name: str
    type: str
    mode: str
    database: str
    bucket: str
    is_default: bool


@dataclass
class DatasourceListResult:
    ok: bool
    message: str
    datasources: list[GrafanaDatasource] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


_LISTED_DATASOURCE_TYPES = ("prometheus", "influxdb")


def _datasource_item(raw: dict) -> GrafanaDatasource:
    json_data = _cfg(raw.get("jsonData"))
    ds_type = str(raw.get("type") or "")
    if ds_type == "prometheus":
        mode = "promql"
    else:
        version = str(json_data.get("version") or "InfluxQL").strip().lower()
        mode = "flux" if version == "flux" else ("sql" if version == "sql" else "influxql")
    return GrafanaDatasource(
        uid=str(raw.get("uid") or ""),
        name=str(raw.get("name") or ""),
        type=ds_type,
        mode=mode,
        database=str(raw.get("database") or json_data.get("dbName") or ""),
        bucket=str(json_data.get("defaultBucket") or ""),
        is_default=bool(raw.get("isDefault")),
    )


def list_grafana_datasources(g_cfg: dict) -> DatasourceListResult:
    """Prometheus and InfluxDB datasources of a Grafana source, for picking one per domain."""
    started = time.perf_counter()
    base_url = str(g_cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        return DatasourceListResult(ok=False, message=t("server.connection.grafana_url_missing"))
    headers, auth = _grafana_auth(g_cfg)
    try:
        resp = requests.get(f"{base_url}/api/datasources", headers=headers, auth=auth, timeout=CHECK_TIMEOUT_SEC, verify=bool(g_cfg.get("verify_ssl", True)))
        if resp.status_code == 403:
            result = DatasourceListResult(ok=False, message=t("server.connection.grafana_datasources_forbidden"))
        else:
            resp.raise_for_status()
            payload = resp.json()
            raw_items = payload if isinstance(payload, list) else []
            items = [_datasource_item(item) for item in raw_items if isinstance(item, dict) and item.get("type") in _LISTED_DATASOURCE_TYPES]
            items.sort(key=lambda item: (item.type, item.name.lower()))
            result = DatasourceListResult(ok=True, message=t("server.connection.grafana_datasources", count=len(items)), datasources=items)
    except Exception as exc:
        result = DatasourceListResult(ok=False, message=_explain(exc))
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


def check_prometheus(url: str) -> CheckResult:
    def run() -> CheckResult:
        base = str(url or "").rstrip("/")
        if not base:
            return CheckResult(ok=False, message=t("server.connection.prometheus_url_missing"))
        resp = requests.get(f"{base}/api/v1/status/buildinfo", timeout=CHECK_TIMEOUT_SEC)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or {}
        return CheckResult(ok=True, message=t("server.connection.prometheus_ok", version=data.get("version", "?")), details={"version": data.get("version")})
    return _timed(run)


def check_influxdb(influx_cfg: dict) -> CheckResult:
    def run() -> CheckResult:
        base = str(influx_cfg.get("url") or "").rstrip("/")
        if not base:
            return CheckResult(ok=False, message=t("server.connection.influxdb_url_missing"))
        headers = {"Authorization": f"Token {influx_cfg.get('token')}"} if influx_cfg.get("token") else {}
        resp = requests.get(f"{base}/health", headers=headers, timeout=CHECK_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json() or {}
        status = str(data.get("status") or "unknown")
        ok = status == "pass"
        return CheckResult(ok=ok, message=t("server.connection.influxdb_status", version=data.get("version", "?"), status=status), details={"version": data.get("version"), "status": status})
    return _timed(run)


def check_data_source(entry: dict) -> CheckResult:
    """Connection only: Prometheus, Grafana login or InfluxDB health."""
    source_type = str(_cfg(entry).get("type") or "").lower()
    if source_type == "grafana_proxy":
        return check_grafana_auth(_cfg(entry.get("grafana")))
    if source_type == "prometheus":
        return check_prometheus(str(_cfg(entry.get("prometheus")).get("url") or ""))
    if source_type == "influxdb":
        return check_influxdb(_cfg(entry.get("influxdb")))
    return CheckResult(ok=False, message=t("server.connection.unknown_source_type", type=source_type or t("server.common.unset")))


def check_resolved_source(resolved: ResolvedSource, language: str | None) -> CheckResult:
    """Binding check. Grafana also resolves the datasource the domain reads."""
    if not isinstance(resolved, ResolvedSource):
        return CheckResult(ok=False, message=t("server.connection.domain_source_unknown"))
    if resolved.source_type == "prometheus":
        return check_prometheus(resolved.prometheus_url)
    if resolved.source_type == "influxdb":
        return check_influxdb(_cfg(resolved.config.get("influxdb")))
    if resolved.source_type == "grafana_proxy":
        return check_grafana(_cfg(resolved.config.get("grafana")), _grafana_binding_kind(resolved.config, language))
    return CheckResult(ok=False, message=t("server.connection.unknown_type", type=resolved.source_type))


def _grafana_binding_kind(config: dict, language: str | None) -> str:
    if language in ("influxql", "flux"):
        return "influxdb"
    if language == "promql":
        return "prometheus"
    influx = _cfg(config.get("influxdb"))
    if influx.get("database") or influx.get("bucket"):
        return "influxdb"
    return "prometheus"


def check_opensearch(logs_cfg: dict) -> CheckResult:
    def run() -> CheckResult:
        os_cfg = _cfg(logs_cfg.get("opensearch"))
        base = str(os_cfg.get("base_url") or "").rstrip("/")
        index = str(os_cfg.get("index_pattern") or "").strip()
        if not base or not index:
            return CheckResult(ok=False, message=t("server.connection.opensearch_missing"))
        username = str(os_cfg.get("username_env") or "").strip()
        password = str(os_cfg.get("password_env") or "").strip()
        resp = requests.post(
            f"{base}/api/console/proxy",
            params={"path": f"/{index}/_count", "method": "GET", "dataSourceId": ""},
            json={},
            headers={"osd-xsrf": "true", "Content-Type": "application/json"},
            auth=(username, password) if username else None,
            verify=bool(os_cfg.get("verify_ssl", True)),
            timeout=CHECK_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        count = (resp.json() or {}).get("count")
        return CheckResult(ok=True, message=t("server.connection.opensearch_ok", index=index, count=count), details={"count": count})
    return _timed(run)


def check_confluence(cfg: dict) -> CheckResult:
    def run() -> CheckResult:
        if not str(cfg.get("base_url") or "").strip():
            return CheckResult(ok=False, message=t("server.connection.confluence_url_missing"))
        client = ConfluenceClient({**cfg, "request_timeout_sec": CHECK_TIMEOUT_SEC})
        parent_id = str(cfg.get("parent_page_id") or "").strip()
        if parent_id:
            page = client.get_page(parent_id)
            if page is None:
                return CheckResult(ok=False, message=t("server.connection.confluence_parent_missing", page_id=parent_id))
            return CheckResult(ok=True, message=t("server.connection.confluence_parent_ok", title=page.get("title")), details={"page_title": page.get("title")})
        space_key = str(cfg.get("space_key") or "").strip()
        if not space_key:
            return CheckResult(ok=False, message=t("server.connection.confluence_space_required"))
        resp = client.session.get(client._url(f"/rest/api/space/{space_key}"), verify=client.verify_ssl, timeout=CHECK_TIMEOUT_SEC)
        resp.raise_for_status()
        space = resp.json() or {}
        return CheckResult(ok=True, message=t("server.connection.confluence_space_ok", space=space.get("name") or space_key), details={"space_name": space.get("name")})
    return _timed(run)


def check_confluence_template(cfg: dict) -> CheckResult:
    """Legacy template flow: Confluence space, Grafana render endpoint and Loki must be reachable."""
    def run() -> CheckResult:
        url_basic = str(cfg.get("url_basic") or "").rstrip("/")
        space = str(cfg.get("space_conf") or "").strip()
        if not url_basic or not space:
            return CheckResult(ok=False, message=t("server.connection.template_missing"))
        auth = (str(cfg.get("user") or ""), str(cfg.get("password") or ""))
        resp = requests.get(f"{url_basic}/rest/api/space/{space}", auth=auth, timeout=CHECK_TIMEOUT_SEC, verify=bool(cfg.get("verify_ssl", False)))
        resp.raise_for_status()
        space_name = str((resp.json() or {}).get("name") or space)
        details: Dict[str, Any] = {"space_name": space_name}
        notes = [t("server.connection.template_space_ok", space=space_name)]
        grafana_url = str(cfg.get("grafana_base_url") or "").rstrip("/")
        if grafana_url:
            health = requests.get(f"{grafana_url}/api/health", timeout=CHECK_TIMEOUT_SEC, verify=False)
            health.raise_for_status()
            details["grafana_version"] = str((health.json() or {}).get("version") or "?")
            notes.append(t("server.connection.template_grafana_ok", version=details["grafana_version"]))
        return CheckResult(ok=True, message="; ".join(notes), details=details)
    return _timed(run)


@dataclass
class ModelListResult:
    ok: bool
    message: str
    models: list = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _unsupported_models() -> str:
    return t("server.connection.models_unsupported")


def _model_names(payload: Any) -> list[str]:
    data = getattr(payload, "data", None)
    if data is None and isinstance(payload, dict):
        data = payload.get("data") or payload.get("models") or payload
    if not isinstance(data, list):
        return []
    names: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
            continue
        if isinstance(item, dict):
            name = item.get("id") or item.get("name") or item.get("id_")
        else:
            name = getattr(item, "id_", None) or getattr(item, "id", None) or getattr(item, "name", None)
        if name:
            names.append(str(name).strip())
    return sorted({name for name in names if name})


def _models_from_http(url: str, headers: dict, auth=None, verify: bool = True) -> ModelListResult:
    try:
        resp = requests.get(url, headers=headers, auth=auth, timeout=CHECK_TIMEOUT_SEC, verify=verify)
    except Exception as exc:
        return ModelListResult(ok=False, message=_explain(exc), models=[])
    if resp.status_code in (404, 405, 501):
        return ModelListResult(ok=False, message=_unsupported_models())
    if resp.status_code in (401, 403):
        return ModelListResult(ok=False, message=t("server.connection.models_access_denied", status=resp.status_code))
    if not resp.ok:
        return ModelListResult(ok=False, message=t("server.connection.models_http_error", status=resp.status_code))
    try:
        payload = resp.json()
    except ValueError:
        return ModelListResult(ok=False, message=_unsupported_models())
    names = _model_names(payload)
    if not names:
        return ModelListResult(ok=False, message=_unsupported_models())
    return ModelListResult(ok=True, message=t("server.connection.models_count", count=len(names)), models=names)


def _anthropic_compatible_models(pcfg: dict) -> ModelListResult:
    """Lists models for an Anthropic-compatible endpoint.

    Official Anthropic serves ``/v1/models`` on the same host as messages.
    Gateways such as MiMo and MiniMax keep messages under ``/anthropic`` and the
    catalog on the OpenAI-compatible ``/v1/models`` of the parent host.
    """
    base = str(pcfg.get("api_base_url") or pcfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
    verify = pcfg.get("verify", True) if isinstance(pcfg.get("verify"), bool) else True
    key = str(pcfg.get("api_key") or "")
    primary = _models_from_http(
        f"{base}/v1/models",
        {"x-api-key": key, "anthropic-version": "2023-06-01", "Accept": "application/json"},
        verify=verify,
    )
    if primary.ok or not base.endswith("/anthropic"):
        return primary
    catalog = _models_from_http(
        f"{base[: -len('/anthropic')]}/v1/models",
        {"Authorization": f"Bearer {key}", "Accept": "application/json"},
        verify=verify,
    )
    if catalog.ok or primary.message == _unsupported_models():
        return catalog
    return primary


def list_llm_models(llm_cfg: dict) -> ModelListResult:
    """Lists models for the provider configured in the (possibly unsaved) form."""
    started = time.perf_counter()
    provider = str((llm_cfg or {}).get("provider") or "").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        result = ModelListResult(
            ok=False, message=t("server.connection.unknown_provider", provider=provider, allowed=", ".join(SUPPORTED_PROVIDERS))
        )
    else:
        pcfg = _cfg((llm_cfg or {}).get(provider))
        if provider == "gigachat":
            try:
                client = _get_gigachat_client(pcfg)
                getter = getattr(client, "get_models", None)
                names = _model_names(getter()) if getter is not None else []
                message = t("server.connection.models_count", count=len(names)) if names else _unsupported_models()
                result = ModelListResult(ok=bool(names), message=message, models=names)
            except Exception as exc:
                result = ModelListResult(ok=False, message=_explain(exc))
        elif provider == "anthropic":
            result = _anthropic_compatible_models(pcfg)
        else:
            raw_base = str(pcfg.get("api_base_url") or pcfg.get("base_url") or "").strip()
            if not raw_base:
                result = ModelListResult(ok=False, message=t("server.connection.api_url_missing"))
            else:
                base = _normalize_llm_base_url(raw_base) if provider == "openai" else raw_base.rstrip("/")
                if base.endswith("/chat/completions"):
                    base = base[: -len("/chat/completions")]
                token = str(pcfg.get("api_key") or "")
                result = _models_from_http(f"{base}/models", {"Authorization": f"Bearer {token}", "Accept": "application/json"})
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


def check_llm(llm_cfg: dict) -> CheckResult:
    """Sends a one-word prompt to the configured provider using the (unsaved) form data."""
    def run() -> CheckResult:
        provider = str(llm_cfg.get("provider") or "").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            return CheckResult(
                ok=False, message=t("server.connection.unknown_provider", provider=provider, allowed=", ".join(SUPPORTED_PROVIDERS))
            )
        pcfg = dict(_cfg(llm_cfg.get(provider)))
        if not pcfg:
            return CheckResult(ok=False, message=t("server.connection.llm_section_empty", provider=provider))
        pcfg["request_timeout_sec"] = min(int(pcfg.get("request_timeout_sec") or LLM_CHECK_TIMEOUT_SEC), LLM_CHECK_TIMEOUT_SEC)
        pcfg["connect_timeout_sec"] = min(int(pcfg.get("connect_timeout_sec") or CHECK_TIMEOUT_SEC), CHECK_TIMEOUT_SEC)
        pcfg["generation"] = {**_cfg(pcfg.get("generation")), "max_tokens": 16}
        pcfg["request_min_interval_sec"] = 0
        text = ask_llm_with_text_data(
            t("server.connection.llm_probe_prompt"),
            "",
            llm_config={"provider": provider, provider: pcfg, "force_json": False, "max_attempts": 1},
            system_prompt=t("server.connection.llm_probe_system"),
        )
        reply = str(text or "").strip()
        return CheckResult(
            ok=True,
            message=t(
                "server.connection.llm_reply",
                provider=provider,
                model=pcfg.get("model") or t("server.connection.model_unset"),
                reply=reply[:60],
            ),
            details={"provider": provider, "model": pcfg.get("model"), "reply": reply[:200]},
        )
    return _timed(run)


# ---- dispatcher --------------------------------------------------------------

def run_check(section: str, data: dict) -> CheckResult:
    """Runs the check matching a configuration section name."""
    payload = _cfg(data)
    if section in ("storage", "storage.timescale"):
        timescale = _cfg(payload.get("timescale")) if section == "storage" else payload
        return check_timescale(timescale)
    if section == "data_sources":
        return CheckResult(ok=False, message=t("server.connection.source_required"))
    if section == "domain_sources":
        return CheckResult(ok=False, message=t("server.connection.domain_required"))
    if section == "logs_source":
        return check_opensearch(payload)
    if section == "confluence":
        return check_confluence(payload)
    if section == "confluence_template":
        return check_confluence_template(payload)
    if section == "llm":
        return check_llm(payload)
    raise ValueError(t("server.connection.section_unavailable", section=section, allowed=", ".join(CHECKABLE_SECTIONS)))


__all__ = [
    "CHECKABLE_SECTIONS", "CheckResult", "run_check", "check_timescale", "check_grafana", "check_prometheus",
    "check_influxdb", "check_data_source", "check_resolved_source", "check_grafana_auth",
    "check_opensearch", "check_confluence", "check_confluence_template", "check_llm",
    "list_grafana_datasources", "DatasourceListResult", "GrafanaDatasource",
    "list_llm_models", "ModelListResult",
]
