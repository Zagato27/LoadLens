"""Connectivity checks for external systems, run from the settings UI.

Each check receives the configuration section as edited in the form (secrets
already restored), talks to the target with a short timeout and returns a
:class:`CheckResult` with a human-readable Russian message. Checks never raise:
failures are reported in the result so the UI can show them inline.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict

import psycopg2
import requests

from AI.data_sources import ResolvedSource
from AI.pipeline import _resolve_grafana_influx_ds_id, _resolve_grafana_prom_ds_id
from AI.providers import SUPPORTED_PROVIDERS, _get_gigachat_client, _normalize_llm_base_url, ask_llm_with_text_data
from loadlens_app.confluence_export import ConfluenceClient

CHECK_TIMEOUT_SEC = 10
LLM_CHECK_TIMEOUT_SEC = 45

CHECKABLE_SECTIONS = (
    "storage", "storage.timescale", "data_sources", "domain_sources",
    "logs_source", "confluence", "confluence_template", "llm",
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
    """Maps common transport/auth failures to actionable Russian messages."""
    if isinstance(exc, requests.exceptions.Timeout):
        return f"Превышено время ожидания ответа ({CHECK_TIMEOUT_SEC} с): сервер не отвечает или адрес неверный"
    if isinstance(exc, requests.exceptions.SSLError):
        return f"Ошибка TLS-сертификата: включите доверенный сертификат или отключите проверку (verify_ssl). {exc}"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"Не удалось подключиться: сервер недоступен или адрес неверный. {exc}"
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            return f"Отказано в доступе (HTTP {status}): проверьте логин/пароль или токен"
        if status == 404:
            return "Ресурс не найден (HTTP 404): проверьте URL, датасорс или идентификатор"
        return f"Сервер вернул ошибку HTTP {status}: {exc}"
    if isinstance(exc, psycopg2.OperationalError):
        text = str(exc)
        if "password authentication failed" in text:
            return "Неверный логин или пароль базы данных"
        if "does not exist" in text:
            return f"База данных или пользователь не существует: {text.strip()}"
        if "could not connect" in text or "Connection refused" in text or "timeout expired" in text:
            return f"Сервер базы данных недоступен: {text.strip()}"
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
        missing = [t for t in expected if t not in present]
        timescale = f"TimescaleDB {ext_row[0]}" if ext_row else "расширение timescaledb не установлено"
        note = f"; таблицы будут созданы при первом отчёте: {', '.join(missing)}" if missing else "; все таблицы на месте"
        return CheckResult(ok=True, message=f"Подключение успешно: {version}, {timescale}{note}", details={"version": version, "timescaledb": ext_row[0] if ext_row else None, "tables_present": present, "tables_missing": missing})
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
        return CheckResult(ok=False, message="Не указан base_url Grafana")
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
        return CheckResult(ok=True, message=f"Grafana {version}: авторизация прошла", details=details)
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
            message=f"Grafana {version}: авторизация прошла, датасорс «{ds_name}» найден{id_note}",
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
        return DatasourceListResult(ok=False, message="Не указан base_url Grafana")
    headers, auth = _grafana_auth(g_cfg)
    try:
        resp = requests.get(f"{base_url}/api/datasources", headers=headers, auth=auth, timeout=CHECK_TIMEOUT_SEC, verify=bool(g_cfg.get("verify_ssl", True)))
        if resp.status_code == 403:
            result = DatasourceListResult(ok=False, message="Grafana не отдаёт список датасорсов (HTTP 403): у пользователя нет права на чтение датасорсов")
        else:
            resp.raise_for_status()
            payload = resp.json()
            raw_items = payload if isinstance(payload, list) else []
            items = [_datasource_item(item) for item in raw_items if isinstance(item, dict) and item.get("type") in _LISTED_DATASOURCE_TYPES]
            items.sort(key=lambda item: (item.type, item.name.lower()))
            result = DatasourceListResult(ok=True, message=f"Датасорсов Prometheus и InfluxDB: {len(items)}", datasources=items)
    except Exception as exc:
        result = DatasourceListResult(ok=False, message=_explain(exc))
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


def check_prometheus(url: str) -> CheckResult:
    def run() -> CheckResult:
        base = str(url or "").rstrip("/")
        if not base:
            return CheckResult(ok=False, message="Не указан URL Prometheus")
        resp = requests.get(f"{base}/api/v1/status/buildinfo", timeout=CHECK_TIMEOUT_SEC)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or {}
        return CheckResult(ok=True, message=f"Prometheus {data.get('version', '?')} доступен", details={"version": data.get("version")})
    return _timed(run)


def check_influxdb(influx_cfg: dict) -> CheckResult:
    def run() -> CheckResult:
        base = str(influx_cfg.get("url") or "").rstrip("/")
        if not base:
            return CheckResult(ok=False, message="Не указан URL InfluxDB")
        headers = {"Authorization": f"Token {influx_cfg.get('token')}"} if influx_cfg.get("token") else {}
        resp = requests.get(f"{base}/health", headers=headers, timeout=CHECK_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json() or {}
        status = str(data.get("status") or "unknown")
        ok = status == "pass"
        return CheckResult(ok=ok, message=f"InfluxDB {data.get('version', '?')}: статус {status}", details={"version": data.get("version"), "status": status})
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
    return CheckResult(ok=False, message=f"Неизвестный тип источника: {source_type or 'не задан'}. Допустимо: prometheus, grafana_proxy, influxdb")


def check_resolved_source(resolved: ResolvedSource, language: str | None) -> CheckResult:
    """Binding check. Grafana also resolves the datasource the domain reads."""
    if not isinstance(resolved, ResolvedSource):
        return CheckResult(ok=False, message="Не удалось определить источник домена")
    if resolved.source_type == "prometheus":
        return check_prometheus(resolved.prometheus_url)
    if resolved.source_type == "influxdb":
        return check_influxdb(_cfg(resolved.config.get("influxdb")))
    if resolved.source_type == "grafana_proxy":
        return check_grafana(_cfg(resolved.config.get("grafana")), _grafana_binding_kind(resolved.config, language))
    return CheckResult(ok=False, message=f"Неизвестный тип источника: {resolved.source_type}")


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
            return CheckResult(ok=False, message="Укажите base_url и index_pattern OpenSearch")
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
        return CheckResult(ok=True, message=f"OpenSearch доступен: индекс «{index}», документов: {count}", details={"count": count})
    return _timed(run)


def check_confluence(cfg: dict) -> CheckResult:
    def run() -> CheckResult:
        if not str(cfg.get("base_url") or "").strip():
            return CheckResult(ok=False, message="Не указан base_url Confluence")
        client = ConfluenceClient({**cfg, "request_timeout_sec": CHECK_TIMEOUT_SEC})
        parent_id = str(cfg.get("parent_page_id") or "").strip()
        if parent_id:
            page = client.get_page(parent_id)
            if page is None:
                return CheckResult(ok=False, message=f"Родительская страница {parent_id} не найдена: проверьте parent_page_id и права доступа")
            return CheckResult(ok=True, message=f"Confluence доступен: родительская страница «{page.get('title')}» найдена", details={"page_title": page.get("title")})
        space_key = str(cfg.get("space_key") or "").strip()
        if not space_key:
            return CheckResult(ok=False, message="Укажите space_key или parent_page_id")
        resp = client.session.get(client._url(f"/rest/api/space/{space_key}"), verify=client.verify_ssl, timeout=CHECK_TIMEOUT_SEC)
        resp.raise_for_status()
        space = resp.json() or {}
        return CheckResult(ok=True, message=f"Confluence доступен: пространство «{space.get('name') or space_key}» найдено", details={"space_name": space.get("name")})
    return _timed(run)


def check_confluence_template(cfg: dict) -> CheckResult:
    """Legacy template flow: Confluence space, Grafana render endpoint and Loki must be reachable."""
    def run() -> CheckResult:
        url_basic = str(cfg.get("url_basic") or "").rstrip("/")
        space = str(cfg.get("space_conf") or "").strip()
        if not url_basic or not space:
            return CheckResult(ok=False, message="Укажите url_basic и space_conf Confluence")
        auth = (str(cfg.get("user") or ""), str(cfg.get("password") or ""))
        resp = requests.get(f"{url_basic}/rest/api/space/{space}", auth=auth, timeout=CHECK_TIMEOUT_SEC, verify=bool(cfg.get("verify_ssl", False)))
        resp.raise_for_status()
        space_name = str((resp.json() or {}).get("name") or space)
        details: Dict[str, Any] = {"space_name": space_name}
        notes = [f"Confluence: пространство «{space_name}» доступно"]
        grafana_url = str(cfg.get("grafana_base_url") or "").rstrip("/")
        if grafana_url:
            health = requests.get(f"{grafana_url}/api/health", timeout=CHECK_TIMEOUT_SEC, verify=False)
            health.raise_for_status()
            details["grafana_version"] = str((health.json() or {}).get("version") or "?")
            notes.append(f"Grafana {details['grafana_version']} доступна")
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


_UNSUPPORTED_MODELS = "Провайдер не отдаёт список моделей"


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
        return ModelListResult(ok=False, message=_UNSUPPORTED_MODELS)
    if resp.status_code in (401, 403):
        return ModelListResult(ok=False, message=f"Отказано в доступе (HTTP {resp.status_code}): проверьте ключ")
    if not resp.ok:
        return ModelListResult(ok=False, message=f"Сервер вернул ошибку HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError:
        return ModelListResult(ok=False, message=_UNSUPPORTED_MODELS)
    names = _model_names(payload)
    if not names:
        return ModelListResult(ok=False, message=_UNSUPPORTED_MODELS)
    return ModelListResult(ok=True, message=f"Доступно моделей: {len(names)}", models=names)


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
    if catalog.ok or primary.message == _UNSUPPORTED_MODELS:
        return catalog
    return primary


def list_llm_models(llm_cfg: dict) -> ModelListResult:
    """Lists models for the provider configured in the (possibly unsaved) form."""
    started = time.perf_counter()
    provider = str((llm_cfg or {}).get("provider") or "").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        result = ModelListResult(ok=False, message=f"Неизвестный провайдер «{provider}». Допустимо: {', '.join(SUPPORTED_PROVIDERS)}")
    else:
        pcfg = _cfg((llm_cfg or {}).get(provider))
        if provider == "gigachat":
            try:
                client = _get_gigachat_client(pcfg)
                getter = getattr(client, "get_models", None)
                names = _model_names(getter()) if getter is not None else []
                result = ModelListResult(ok=bool(names), message=f"Доступно моделей: {len(names)}" if names else _UNSUPPORTED_MODELS, models=names)
            except Exception as exc:
                result = ModelListResult(ok=False, message=_explain(exc))
        elif provider == "anthropic":
            result = _anthropic_compatible_models(pcfg)
        else:
            raw_base = str(pcfg.get("api_base_url") or pcfg.get("base_url") or "").strip()
            if not raw_base:
                result = ModelListResult(ok=False, message="Не указан API URL")
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
            return CheckResult(ok=False, message=f"Неизвестный провайдер «{provider}». Допустимо: {', '.join(SUPPORTED_PROVIDERS)}")
        pcfg = dict(_cfg(llm_cfg.get(provider)))
        if not pcfg:
            return CheckResult(ok=False, message=f"Раздел llm.{provider} не заполнен")
        pcfg["request_timeout_sec"] = min(int(pcfg.get("request_timeout_sec") or LLM_CHECK_TIMEOUT_SEC), LLM_CHECK_TIMEOUT_SEC)
        pcfg["connect_timeout_sec"] = min(int(pcfg.get("connect_timeout_sec") or CHECK_TIMEOUT_SEC), CHECK_TIMEOUT_SEC)
        pcfg["generation"] = {**_cfg(pcfg.get("generation")), "max_tokens": 16}
        pcfg["request_min_interval_sec"] = 0
        text = ask_llm_with_text_data(
            "Ответь одним словом: OK.",
            "",
            llm_config={"provider": provider, provider: pcfg, "force_json": False, "max_attempts": 1},
            system_prompt="Отвечай одним словом.",
        )
        reply = str(text or "").strip()
        return CheckResult(
            ok=True,
            message=f"{provider} ({pcfg.get('model') or 'модель не указана'}) ответил: «{reply[:60]}»",
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
        return CheckResult(ok=False, message="Укажите источник для проверки")
    if section == "domain_sources":
        return CheckResult(ok=False, message="Укажите домен для проверки")
    if section == "logs_source":
        return check_opensearch(payload)
    if section == "confluence":
        return check_confluence(payload)
    if section == "confluence_template":
        return check_confluence_template(payload)
    if section == "llm":
        return check_llm(payload)
    raise ValueError(f"Проверка соединения недоступна для раздела «{section}». Допустимо: {', '.join(CHECKABLE_SECTIONS)}")


__all__ = [
    "CHECKABLE_SECTIONS", "CheckResult", "run_check", "check_timescale", "check_grafana", "check_prometheus",
    "check_influxdb", "check_data_source", "check_resolved_source", "check_grafana_auth",
    "check_opensearch", "check_confluence", "check_confluence_template", "check_llm",
    "list_grafana_datasources", "DatasourceListResult", "GrafanaDatasource",
    "list_llm_models", "ModelListResult",
]
