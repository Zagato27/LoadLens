"""Catalog of metric connections and per-domain bindings.

A source stores only how to connect (address, auth, TLS). A binding says which
source a domain reads and which datasource, database or bucket to use. Bindings
are replaced per project area; the catalog is global.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Mapping

SOURCE_TYPES: tuple[str, ...] = ("prometheus", "grafana_proxy", "influxdb")
APP_DOMAINS: tuple[str, ...] = ("jvm", "database", "kafka", "microservices", "hard_resources")
LT_DOMAIN = "lt_framework"
DEFAULT_BINDING = "default"
BINDING_DOMAINS: tuple[str, ...] = APP_DOMAINS + (LT_DOMAIN,)
BINDING_KEYS: frozenset[str] = frozenset((DEFAULT_BINDING, *BINDING_DOMAINS))
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
TITLE_MAX_LEN = 80
AUTH_METHODS: frozenset[str] = frozenset({"basic", "bearer"})
QUERY_LANG_FIELDS: dict[str, str] = {
    "promql": "promql_queries",
    "influxql": "influxql_queries",
    "flux": "flux_queries",
}
GRAFANA_LANG_PRIORITY: tuple[str, ...] = ("influxql", "promql", "flux")
_SOURCE_BLOCK: dict[str, str] = {
    "prometheus": "prometheus",
    "grafana_proxy": "grafana",
    "influxdb": "influxdb",
}
_PROMETHEUS_FIELDS: frozenset[str] = frozenset({"source"})
_INFLUX_FIELDS: frozenset[str] = frozenset({"source", "bucket"})
_GRAFANA_FIELDS: frozenset[str] = frozenset({
    "source", "datasource_uid", "datasource_name", "database", "bucket",
})
_BINDING_TEXT_FIELDS: tuple[str, ...] = ("source", "datasource_uid", "datasource_name", "database", "bucket")


class DataSourceConfigError(ValueError):
    """Invalid data source catalog or domain binding."""


@dataclass(frozen=True)
class DomainBinding:
    """What one domain reads. Empty strings mean the field is unset."""

    domain: str
    source_id: str
    datasource_uid: str
    datasource_name: str
    database: str
    bucket: str


@dataclass(frozen=True)
class ResolvedSource:
    """Connection for one domain, shaped like the legacy ``metrics_source`` section."""

    domain: str
    source_id: str
    source_type: str
    config: dict

    @property
    def prometheus_url(self) -> str:
        prometheus = self.config.get("prometheus")
        if isinstance(prometheus, dict):
            return str(prometheus.get("url") or "")
        return ""


def validate_data_sources(payload: dict) -> None:
    """Raises ``DataSourceConfigError`` when the catalog shape is wrong."""
    if not isinstance(payload, dict):
        raise DataSourceConfigError("Каталог источников должен быть объектом")
    for source_id, entry in payload.items():
        _validate_source(str(source_id), entry)


def validate_domain_sources(payload: dict, catalog: dict) -> None:
    """Raises ``DataSourceConfigError`` when bindings do not match ``catalog``."""
    errors = binding_messages(payload, catalog, require_default=True, missing_kind="missing")
    if errors:
        raise DataSourceConfigError(errors[0])


def binding_errors(catalog: dict, global_bindings: dict, area_bindings: Mapping[str, dict]) -> list[str]:
    """Bindings that the new catalog would break, including sources still in use."""
    errors = binding_messages(global_bindings, catalog, require_default=False, missing_kind="used", place="глобальные привязки")
    for area in sorted(area_bindings):
        place = f"область {area}"
        errors.extend(binding_messages(area_bindings[area], catalog, require_default=False, missing_kind="used", place=place))
    return errors


def binding_messages(
    payload: object,
    catalog: object,
    *,
    require_default: bool,
    missing_kind: str,
    place: str = "",
) -> list[str]:
    """Human-readable binding problems. Empty bindings are allowed when default is optional."""
    if not isinstance(payload, dict):
        return [_scoped(place, "привязки должны быть объектом")]
    if not payload and not require_default:
        return []
    if DEFAULT_BINDING not in payload:
        return [_scoped(place, "не задан источник по умолчанию")]
    catalog_map = catalog if isinstance(catalog, dict) else {}
    errors: list[str] = []
    for key, raw in payload.items():
        errors.extend(_one_binding(str(key), raw, catalog_map, place, missing_kind))
    return errors


def resolve_domain_source(cfg: dict, domain: str) -> ResolvedSource:
    """Binding of ``domain``, or the default binding when the domain has none."""
    if domain not in BINDING_KEYS:
        raise DataSourceConfigError(f"Неизвестный домен «{domain}»")
    catalog = cfg.get("data_sources")
    if not isinstance(catalog, dict):
        raise DataSourceConfigError("Каталог источников data_sources не задан")
    bindings = cfg.get("domain_sources")
    if not isinstance(bindings, dict) or DEFAULT_BINDING not in bindings:
        raise DataSourceConfigError("Не задан источник по умолчанию")
    raw = bindings.get(domain)
    if not isinstance(raw, dict):
        raw = bindings.get(DEFAULT_BINDING)
    if not isinstance(raw, dict):
        raise DataSourceConfigError(f"домен {domain}: не задан источник")
    binding = _parse_binding(domain, raw)
    entry = catalog.get(binding.source_id)
    if not isinstance(entry, dict):
        raise DataSourceConfigError(f"домен {domain}: источник «{binding.source_id}» не найден")
    source_type = str(entry.get("type") or "")
    if source_type not in SOURCE_TYPES:
        raise DataSourceConfigError(f"домен {domain}: источник «{binding.source_id}» имеет неизвестный тип «{source_type}»")
    if source_type == "grafana_proxy" and not binding.datasource_uid and not binding.datasource_name:
        raise DataSourceConfigError(f"домен {domain}: у источника «{binding.source_id}» не задан UID или имя датасорса")
    return ResolvedSource(domain, binding.source_id, source_type, _resolved_config(entry, source_type, binding))


def domain_query_language(resolved: ResolvedSource, qcfg: dict) -> str | None:
    """Language to run for this source. ``None`` when the domain has no queries.

    Grafana keeps the load-tool order: InfluxQL, then PromQL, then Flux.
    """
    present = [lang for lang in ("promql", "influxql", "flux") if _has_queries(qcfg, lang)]
    if not present:
        return None
    if resolved.source_type == "prometheus":
        allowed = ("promql",)
    elif resolved.source_type == "influxdb":
        allowed = ("flux",)
    else:
        for lang in GRAFANA_LANG_PRIORITY:
            if lang in present:
                return lang
        return None
    extra = [lang for lang in present if lang not in allowed]
    if extra or allowed[0] not in present:
        raise DataSourceConfigError(
            f"домен {resolved.domain}: источник «{resolved.source_id}» ({resolved.source_type}) "
            f"принимает {allowed[0]}, в запросах есть {', '.join(present)}"
        )
    return allowed[0]


def source_accepts_language(source_type: str, language: str) -> bool:
    """Whether one query of ``language`` can run against this source type."""
    if source_type == "prometheus":
        return language == "promql"
    if source_type == "influxdb":
        return language == "flux"
    return language in QUERY_LANG_FIELDS


def _validate_source(source_id: str, entry: object) -> None:
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise DataSourceConfigError(
            f"Идентификатор источника «{source_id}» недопустим: латиница в нижнем регистре, "
            "цифры, дефис и подчёркивание, до 40 символов"
        )
    if not isinstance(entry, dict):
        raise DataSourceConfigError(f"Источник «{source_id}» должен быть объектом")
    _validate_title(source_id, entry.get("title"))
    source_type = entry.get("type")
    if source_type not in SOURCE_TYPES:
        raise DataSourceConfigError(
            f"Источник «{source_id}»: неизвестный тип «{source_type}». Допустимо: {', '.join(SOURCE_TYPES)}"
        )
    block_key = _SOURCE_BLOCK[str(source_type)]
    for key in entry:
        if key not in {"title", "type", block_key}:
            raise DataSourceConfigError(f"Источник «{source_id}»: лишний ключ «{key}»")
    block = entry.get(block_key)
    if source_type == "prometheus":
        _require_url(source_id, block, "prometheus", "url", "prometheus.url")
        _reject_unknown(source_id, block, {"url"})
    elif source_type == "influxdb":
        _require_url(source_id, block, "influxdb", "url", "influxdb.url")
        _reject_unknown(source_id, block, {"url", "org", "token"})
        _optional_text(source_id, block, "org")
        _optional_text(source_id, block, "token")
    else:
        _validate_grafana(source_id, block)


def _validate_title(source_id: str, title: object) -> None:
    if not isinstance(title, str) or not title.strip():
        raise DataSourceConfigError(f"Источник «{source_id}»: укажите название")
    if len(title.strip()) > TITLE_MAX_LEN:
        raise DataSourceConfigError(f"Источник «{source_id}»: название длиннее {TITLE_MAX_LEN} символов")
    if any(ord(ch) < 32 for ch in title):
        raise DataSourceConfigError(f"Источник «{source_id}»: в названии есть управляющие символы")


def _validate_grafana(source_id: str, block: object) -> None:
    _require_url(source_id, block, "grafana", "base_url", "grafana.base_url")
    assert isinstance(block, dict)
    _reject_unknown(source_id, block, {"base_url", "verify_ssl", "auth"})
    if "verify_ssl" in block and not isinstance(block.get("verify_ssl"), bool):
        raise DataSourceConfigError(f"Источник «{source_id}»: grafana.verify_ssl должен быть true или false")
    auth = block.get("auth")
    if not isinstance(auth, dict):
        raise DataSourceConfigError(f"Источник «{source_id}»: укажите способ авторизации Grafana")
    _reject_unknown(source_id, auth, {"method", "username", "password", "token"})
    method = auth.get("method")
    if method not in AUTH_METHODS:
        raise DataSourceConfigError(f"Источник «{source_id}»: grafana.auth.method — допустимо basic, bearer")
    for key in ("username", "password", "token"):
        _optional_text(source_id, auth, key)


def _require_url(source_id: str, block: object, block_name: str, field: str, label: str) -> None:
    if not isinstance(block, dict):
        raise DataSourceConfigError(f"Источник «{source_id}»: блок {block_name} должен быть объектом")
    url = block.get(field)
    if not isinstance(url, str) or not url.strip():
        raise DataSourceConfigError(f"Источник «{source_id}»: укажите {label}")


def _reject_unknown(source_id: str, block: dict, allowed: set[str]) -> None:
    for key in block:
        if key not in allowed:
            raise DataSourceConfigError(f"Источник «{source_id}»: лишний ключ «{key}»")


def _optional_text(source_id: str, block: dict, field: str) -> None:
    if field in block and not isinstance(block.get(field), str):
        raise DataSourceConfigError(f"Источник «{source_id}»: поле «{field}» должно быть строкой")


def _one_binding(key: str, raw: object, catalog: dict, place: str, missing_kind: str) -> list[str]:
    if key not in BINDING_KEYS:
        return [_scoped(place, f"неизвестная привязка «{key}»")]
    if not isinstance(raw, dict):
        return [_scoped(place, f"привязка «{key}» должна быть объектом")]
    texts, text_error = _binding_texts(key, raw, place)
    if text_error:
        return [text_error]
    source_id = texts["source"]
    entry = catalog.get(source_id) if source_id else None
    if not source_id or not isinstance(entry, dict):
        return [_missing_source(source_id, key, place, missing_kind)]
    source_type = str(entry.get("type") or "")
    if source_type not in SOURCE_TYPES:
        return [_scoped(place, f"домен {key}: источник «{source_id}» имеет неизвестный тип «{source_type}»")]
    allowed = _fields_for(source_type)
    for field, value in texts.items():
        if field not in allowed and value:
            return [_scoped(place, f"привязка «{key}»: поле «{field}» не используется для типа {source_type}")]
    if source_type == "grafana_proxy" and not texts["datasource_uid"] and not texts["datasource_name"]:
        return [_scoped(place, f"привязка «{key}»: для Grafana укажите UID или имя датасорса")]
    return []


def _binding_texts(key: str, raw: dict, place: str) -> tuple[dict[str, str], str]:
    texts: dict[str, str] = {}
    for field in _BINDING_TEXT_FIELDS:
        value = raw.get(field, "")
        if field not in raw:
            texts[field] = ""
            continue
        if not isinstance(value, str):
            return {}, _scoped(place, f"привязка «{key}»: поле «{field}» должно быть строкой")
        texts[field] = value.strip()
    for field in raw:
        if field not in _BINDING_TEXT_FIELDS:
            return {}, _scoped(place, f"привязка «{key}»: лишний ключ «{field}»")
    return texts, ""


def _fields_for(source_type: str) -> frozenset[str]:
    if source_type == "prometheus":
        return _PROMETHEUS_FIELDS
    if source_type == "influxdb":
        return _INFLUX_FIELDS
    return _GRAFANA_FIELDS


def _missing_source(source_id: str, domain: str, place: str, missing_kind: str) -> str:
    if missing_kind == "used" and source_id:
        where = place or "привязки"
        return f"источник {source_id} используется: {where}, домен {domain}"
    if source_id:
        return _scoped(place, f"домен {domain}: источник «{source_id}» не найден")
    return _scoped(place, f"привязка «{domain}»: не выбран источник")


def _scoped(place: str, message: str) -> str:
    if not place or place == "привязки":
        return message[0].upper() + message[1:] if message else message
    return f"{place}: {message}"


def _parse_binding(domain: str, raw: dict) -> DomainBinding:
    def text(field: str) -> str:
        value = raw.get(field, "")
        return value.strip() if isinstance(value, str) else ""

    source_id = text("source")
    if not source_id:
        raise DataSourceConfigError(f"домен {domain}: не выбран источник")
    return DomainBinding(domain, source_id, text("datasource_uid"), text("datasource_name"), text("database"), text("bucket"))


def _resolved_config(entry: dict, source_type: str, binding: DomainBinding) -> dict:
    if source_type == "prometheus":
        prometheus = entry.get("prometheus") if isinstance(entry.get("prometheus"), dict) else {}
        return {"type": "prometheus", "prometheus": {"url": str(prometheus.get("url") or "")}}
    if source_type == "influxdb":
        return {"type": "influxdb", "influxdb": _influx_block(entry, binding.bucket)}
    return {"type": "grafana_proxy", "grafana": _grafana_block(entry, binding), "influxdb": _influx_aux(binding)}


def _influx_block(entry: dict, bucket: str) -> dict:
    raw = entry.get("influxdb") if isinstance(entry.get("influxdb"), dict) else {}
    block = {"url": str(raw.get("url") or "")}
    if isinstance(raw.get("org"), str):
        block["org"] = raw["org"]
    if isinstance(raw.get("token"), str):
        block["token"] = raw["token"]
    if bucket:
        block["bucket"] = bucket
    return block


def _grafana_block(entry: dict, binding: DomainBinding) -> dict:
    raw = entry.get("grafana") if isinstance(entry.get("grafana"), dict) else {}
    auth = raw.get("auth") if isinstance(raw.get("auth"), dict) else {}
    datasource = {"id": None, "uid": binding.datasource_uid, "name": binding.datasource_name}
    block: dict = {
        "base_url": str(raw.get("base_url") or ""),
        "auth": copy.deepcopy(auth),
        "prometheus_datasource": dict(datasource),
        "influxdb_datasource": dict(datasource),
    }
    if isinstance(raw.get("verify_ssl"), bool):
        block["verify_ssl"] = raw["verify_ssl"]
    return block


def _influx_aux(binding: DomainBinding) -> dict:
    aux: dict[str, str] = {}
    if binding.database:
        aux["database"] = binding.database
    if binding.bucket:
        aux["bucket"] = binding.bucket
    return aux


def _has_queries(qcfg: dict, language: str) -> bool:
    if not isinstance(qcfg, dict):
        return False
    queries = qcfg.get(QUERY_LANG_FIELDS[language]) or []
    if not isinstance(queries, list):
        return False
    return any(str(query or "").strip() for query in queries)


__all__ = [
    "APP_DOMAINS",
    "AUTH_METHODS",
    "BINDING_DOMAINS",
    "BINDING_KEYS",
    "DEFAULT_BINDING",
    "LT_DOMAIN",
    "QUERY_LANG_FIELDS",
    "SOURCE_ID_RE",
    "SOURCE_TYPES",
    "TITLE_MAX_LEN",
    "DataSourceConfigError",
    "DomainBinding",
    "ResolvedSource",
    "binding_errors",
    "binding_messages",
    "domain_query_language",
    "resolve_domain_source",
    "source_accepts_language",
    "validate_data_sources",
    "validate_domain_sources",
]
