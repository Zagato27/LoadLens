# SPDX-License-Identifier: Apache-2.0
"""Catalog of metric connections, per-domain bindings and per-row overrides.

A source stores only how to connect (address, auth, TLS). A binding says which
source a domain reads and which datasource, database or bucket to use. Bindings
are replaced per project area; the catalog is global.

A query row may read another source than its domain: ``queries.<domain>.sources`` maps the
row label to a binding of the same shape, or to ``None`` for "the domain binding" (an explicit
``None`` keeps a wider scope's override from coming back when the scopes are merged). The rows
of a domain are collected in groups, one per source and query language.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from AI.i18n import t

SOURCE_TYPES: tuple[str, ...] = ("prometheus", "grafana_proxy", "influxdb")
# lt_generators: resources of the load generator hosts (optional domain of the load test report).
APP_DOMAINS: tuple[str, ...] = ("jvm", "database", "kafka", "microservices", "hard_resources", "lt_generators")
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
# Series keys of a row by its language (parallel to ``labels`` like the query arrays).
QUERY_KEY_FIELDS: dict[str, str] = {
    "promql": "label_keys_list",
    "influxql": "label_tag_keys_list",
    "flux": "label_tag_keys_list",
}
SOURCES_KEY = "sources"
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
    """Connection for one domain (or one row of it), shaped like the legacy ``metrics_source`` section."""

    domain: str
    source_id: str
    source_type: str
    config: dict
    # The row label when a row override was resolved; "" for the domain binding.
    label: str = ""

    @property
    def prometheus_url(self) -> str:
        prometheus = self.config.get("prometheus")
        if isinstance(prometheus, dict):
            return str(prometheus.get("url") or "")
        return ""


def validate_data_sources(payload: dict) -> None:
    """Raises ``DataSourceConfigError`` when the catalog shape is wrong."""
    if not isinstance(payload, dict):
        raise DataSourceConfigError(t("server.data_sources.catalog_object"))
    for source_id, entry in payload.items():
        _validate_source(str(source_id), entry)


def validate_domain_sources(payload: dict, catalog: dict) -> None:
    """Raises ``DataSourceConfigError`` when bindings do not match ``catalog``."""
    errors = binding_messages(payload, catalog, require_default=True, missing_kind="missing")
    if errors:
        raise DataSourceConfigError(errors[0])


def binding_errors(catalog: dict, global_bindings: dict, area_bindings: Mapping[str, dict]) -> list[str]:
    """Bindings that the new catalog would break, including sources still in use."""
    errors = binding_messages(
        global_bindings, catalog, require_default=False, missing_kind="used", place=t("server.data_sources.place.global")
    )
    for area in sorted(area_bindings):
        place = t("server.data_sources.place.area", area=area)
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
        return [_scoped(place, t("server.data_sources.bindings_object"))]
    if not payload and not require_default:
        return []
    if DEFAULT_BINDING not in payload:
        return [_scoped(place, t("server.data_sources.no_default_lower"))]
    catalog_map = catalog if isinstance(catalog, dict) else {}
    errors: list[str] = []
    for key, raw in payload.items():
        errors.extend(_one_binding(str(key), raw, catalog_map, place, missing_kind))
    return errors


def resolve_domain_source(cfg: dict, domain: str) -> ResolvedSource:
    """Binding of ``domain``, or the default binding when the domain has none."""
    if domain not in BINDING_KEYS:
        raise DataSourceConfigError(t("server.data_sources.unknown_domain", domain=domain))
    catalog = cfg.get("data_sources")
    if not isinstance(catalog, dict):
        raise DataSourceConfigError(t("server.data_sources.catalog_missing"))
    raw = domain_binding(cfg, domain)
    if raw is None:
        bindings = cfg.get("domain_sources")
        if not isinstance(bindings, dict) or DEFAULT_BINDING not in bindings:
            raise DataSourceConfigError(t("server.data_sources.no_default"))
        raise DataSourceConfigError(t("server.data_sources.domain_no_source", domain=domain))
    return resolve_binding(cfg, domain, raw)


def domain_binding(cfg: Mapping[str, Any], domain: str) -> Optional[dict]:
    """The raw binding a domain reads: its own entry, else ``default``; ``None`` when there is none."""
    bindings = cfg.get("domain_sources")
    if not isinstance(bindings, dict) or DEFAULT_BINDING not in bindings:
        return None
    raw = bindings.get(domain)
    if not isinstance(raw, dict):
        raw = bindings.get(DEFAULT_BINDING)
    return raw if isinstance(raw, dict) else None


def resolve_binding(cfg: Mapping[str, Any], domain: str, raw: Mapping[str, Any], label: str = "") -> ResolvedSource:
    """Connection of one binding: a domain's, or the override of the row ``label`` of the domain."""
    name = row_name(domain, label) if label else domain
    catalog = cfg.get("data_sources")
    if not isinstance(catalog, dict):
        raise DataSourceConfigError(t("server.data_sources.catalog_missing"))
    binding = _parse_binding(name, dict(raw))
    entry = catalog.get(binding.source_id)
    if not isinstance(entry, dict):
        raise DataSourceConfigError(t("server.data_sources.domain_source_missing", domain=name, source=binding.source_id))
    source_type = str(entry.get("type") or "")
    if source_type not in SOURCE_TYPES:
        raise DataSourceConfigError(
            t("server.data_sources.domain_source_type", domain=name, source=binding.source_id, type=source_type)
        )
    if source_type == "grafana_proxy" and not binding.datasource_uid and not binding.datasource_name:
        raise DataSourceConfigError(t("server.data_sources.domain_no_datasource", domain=name, source=binding.source_id))
    return ResolvedSource(domain, binding.source_id, source_type, _resolved_config(entry, source_type, binding), label)


def row_name(domain: str, label: str) -> str:
    """How messages name one row of a domain."""
    return t("server.data_sources.row_name", domain=domain, label=label)


# ---- rows that read another source -------------------------------------------------------------------

def normalize_binding(raw: Mapping[str, Any]) -> dict[str, str]:
    """The binding with every text field present (empty when unused).

    Scopes are merged key by key (global, area, service): a full binding of a narrower scope then
    replaces every field of a wider one instead of mixing the datasource of one with the source of another.
    """
    return {name: (raw.get(name) or "").strip() if isinstance(raw.get(name), str) else "" for name in _BINDING_TEXT_FIELDS}


def binding_identity(raw: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    """Comparable form of a binding: equal tuples read the same data."""
    if not isinstance(raw, Mapping):
        return ()
    full = normalize_binding(raw)
    return tuple(full[name] for name in _BINDING_TEXT_FIELDS)


def row_overrides(block: Optional[Mapping[str, Any]]) -> dict[str, dict]:
    """{label: binding} of the rows of the block that read another source than the domain binding."""
    if not isinstance(block, Mapping):
        return {}
    raw = block.get(SOURCES_KEY)
    if not isinstance(raw, Mapping):
        return {}
    labels = {str(label) for label in (block.get("labels") or []) if isinstance(block.get("labels"), list)}
    out: dict[str, dict] = {}
    for label, binding in raw.items():
        if str(label) in labels and isinstance(binding, Mapping) and str(binding.get("source") or "").strip():
            out[str(label)] = dict(binding)
    return out


def normalize_row_sources(queries: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``queries`` whose row overrides carry every binding field; ``None`` entries stay."""
    out = copy.deepcopy(dict(queries or {}))
    for block in out.values():
        if not isinstance(block, dict) or not isinstance(block.get(SOURCES_KEY), dict):
            continue
        block[SOURCES_KEY] = {
            str(label): (normalize_binding(binding) if isinstance(binding, Mapping) else None)
            for label, binding in block[SOURCES_KEY].items()
        }
    return out


def own_row_sources(merged: Any, override: Any) -> Any:
    """``queries`` deep-merged from a wider and a narrower scope: a domain block the narrower scope lists
    rows for (it has ``labels``) keeps only its own ``sources``. Mutates and returns ``merged``.

    A narrower scope stores a copy of its rows; an override the wider scope adds later must not reach
    them unseen (the settings page of the narrower scope shows only its copy).
    """
    if not isinstance(merged, dict) or not isinstance(override, Mapping):
        return merged
    for domain, block in override.items():
        target = merged.get(domain)
        if not isinstance(block, Mapping) or not isinstance(block.get("labels"), list) or not isinstance(target, dict):
            continue
        own = block.get(SOURCES_KEY)
        if isinstance(own, Mapping):
            target[SOURCES_KEY] = copy.deepcopy(dict(own))
        else:
            target.pop(SOURCES_KEY, None)
    return merged


def row_source_messages(queries: object, catalog: object, *, place: str = "", missing_kind: str = "missing") -> list[str]:
    """Problems of the row overrides of ``queries`` against the ``catalog`` (unknown source, wrong fields)."""
    if not isinstance(queries, Mapping):
        return []
    catalog_map = catalog if isinstance(catalog, dict) else {}
    errors: list[str] = []
    for domain, block in queries.items():
        if not isinstance(block, Mapping) or domain not in BINDING_DOMAINS:
            continue
        labels = [str(label) for label in block.get("labels") or []] if isinstance(block.get("labels"), list) else []
        for label, raw in row_overrides(block).items():
            name = row_name(str(domain), label)
            problems = _one_binding(str(domain), raw, catalog_map, place, missing_kind, name=name)
            if not problems and missing_kind == "missing":
                problems = _row_language_problem(block, labels.index(label), name, raw, catalog_map, place)
            errors.extend(problems)
    return errors


def _row_language_problem(block: Mapping[str, Any], index: int, name: str, raw: Mapping[str, Any], catalog: dict, place: str) -> list[str]:
    """A row whose override does not run the language of its query (PromQL against InfluxDB)."""
    entry = catalog.get(str(raw.get("source") or "").strip())
    source_type = str(entry.get("type") or "") if isinstance(entry, dict) else ""
    present = row_languages(block, index)
    if not present or source_accepts_language(source_type, pick_row_language(source_type, present)):
        return []
    message = t(
        "server.data_sources.language_mismatch",
        domain=name,
        source=str(raw.get("source") or ""),
        type=source_type,
        accepted=_accepted_language(source_type),
        present=", ".join(present),
    )
    return [_scoped(place, message)]


def resolve_row_source(cfg: Mapping[str, Any], domain: str, block: Optional[Mapping[str, Any]], label: str) -> ResolvedSource:
    """Source of one row: its override when the block has one, else the domain binding."""
    override = row_overrides(block).get(str(label)) if label else None
    if override is None:
        return resolve_domain_source(dict(cfg), domain)
    return resolve_binding(cfg, domain, override, str(label))


@dataclass(frozen=True)
class SourceGroup:
    """Rows of one domain that run against one source in one language."""

    resolved: ResolvedSource
    language: str
    rows: tuple[int, ...]
    overridden: bool = False
    # Rows with text in several languages: the language picked for each (logged by the caller).
    mixed: tuple[int, ...] = field(default_factory=tuple)


def row_languages(block: Optional[Mapping[str, Any]], index: int) -> list[str]:
    """Languages with a non-empty query text in the row, in ``GRAFANA_LANG_PRIORITY`` order."""
    if not isinstance(block, Mapping):
        return []
    present = []
    for lang in GRAFANA_LANG_PRIORITY:
        values = block.get(QUERY_LANG_FIELDS[lang])
        if isinstance(values, list) and index < len(values) and str(values[index] or "").strip():
            present.append(lang)
    return present


def pick_row_language(source_type: str, present: list[str]) -> str:
    """Language a row runs in: the one its source accepts; Grafana keeps the order InfluxQL, PromQL, Flux."""
    if not present:
        return ""
    for lang in present:
        if source_accepts_language(source_type, lang):
            return lang
    return present[0]


def domain_source_groups(cfg: Mapping[str, Any], domain: str, block: Optional[Mapping[str, Any]]) -> list[SourceGroup]:
    """Rows of the domain grouped by source and language; rows without a query are left out.

    The domain binding is resolved only when some row reads it. A row whose source does not run its
    language raises ``DataSourceConfigError`` naming the row, like a broken binding.
    """
    if not isinstance(block, Mapping):
        return []
    labels = block.get("labels") if isinstance(block.get("labels"), list) else []
    # Rows are positions of labels: a query past the last label has no series name and is not run.
    count = len(labels)
    overrides = row_overrides(block)
    domain_resolved: Optional[ResolvedSource] = None
    order: list[tuple] = []
    rows: dict[tuple, list[int]] = {}
    mixed: dict[tuple, list[int]] = {}
    sources: dict[tuple, ResolvedSource] = {}
    for index in range(count):
        present = row_languages(block, index)
        if not present:
            continue
        label = str(labels[index]) if index < len(labels) else ""
        override = overrides.get(label)
        if override is not None:
            resolved = resolve_binding(cfg, domain, override, label)
            source_key = ("row",) + binding_identity(override)
        else:
            if domain_resolved is None:
                domain_resolved = resolve_domain_source(dict(cfg), domain)
            resolved = domain_resolved
            source_key = ("domain",)
        language = pick_row_language(resolved.source_type, present)
        if not source_accepts_language(resolved.source_type, language):
            name = row_name(domain, label) if label else domain
            raise DataSourceConfigError(
                t(
                    "server.data_sources.language_mismatch",
                    domain=name,
                    source=resolved.source_id,
                    type=resolved.source_type,
                    accepted=_accepted_language(resolved.source_type),
                    present=", ".join(present),
                )
            )
        key = source_key + (language,)
        if key not in rows:
            order.append(key)
            rows[key] = []
            mixed[key] = []
            sources[key] = resolved
        rows[key].append(index)
        if len(present) > 1:
            mixed[key].append(index)
    return [
        SourceGroup(
            resolved=sources[key],
            language=key[-1],
            rows=tuple(rows[key]),
            overridden=key[0] == "row",
            mixed=tuple(mixed[key]),
        )
        for key in order
    ]


def mixed_language_warnings(queries: object, cfg: Mapping[str, Any]) -> list[str]:
    """Rows with query texts in several languages: which one runs (by the row's source), for the settings page."""
    if not isinstance(queries, Mapping):
        return []
    catalog = cfg.get("data_sources") if isinstance(cfg.get("data_sources"), Mapping) else {}

    def source_type(raw: Optional[Mapping[str, Any]]) -> str:
        entry = catalog.get(str((raw or {}).get("source") or "").strip()) if raw else None
        return str(entry.get("type") or "") if isinstance(entry, Mapping) else ""

    out: list[str] = []
    for domain, block in queries.items():
        if not isinstance(block, Mapping):
            continue
        labels = block.get("labels") if isinstance(block.get("labels"), list) else []
        overrides = row_overrides(block)
        for index, raw_label in enumerate(labels):
            present = row_languages(block, index)
            if len(present) < 2:
                continue
            label = str(raw_label)
            kind = source_type(overrides.get(label) or domain_binding(cfg, str(domain)))
            out.append(t(
                "server.config.row_two_languages",
                domain=domain,
                label=label,
                languages=", ".join(present),
                language=pick_row_language(kind, present),
            ))
    return out


def domain_rows_language(resolved: ResolvedSource, block: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Language of the first row that reads the domain binding (rows with an override read their own source).

    ``None`` when no row reads the binding. A row in a language the binding's source does not run raises
    ``DataSourceConfigError`` naming the row.
    """
    if not isinstance(block, Mapping):
        return None
    labels = block.get("labels") if isinstance(block.get("labels"), list) else []
    overrides = row_overrides(block)
    first: Optional[str] = None
    for index, raw_label in enumerate(labels):
        label = str(raw_label)
        present = row_languages(block, index)
        if not present or label in overrides:
            continue
        language = pick_row_language(resolved.source_type, present)
        if not source_accepts_language(resolved.source_type, language):
            raise DataSourceConfigError(
                t(
                    "server.data_sources.language_mismatch",
                    domain=row_name(resolved.domain, label),
                    source=resolved.source_id,
                    type=resolved.source_type,
                    accepted=_accepted_language(resolved.source_type),
                    present=", ".join(present),
                )
            )
        first = first or language
    return first


def _accepted_language(source_type: str) -> str:
    if source_type == "prometheus":
        return "promql"
    if source_type == "influxdb":
        return "flux"
    return ", ".join(GRAFANA_LANG_PRIORITY)


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
            t(
                "server.data_sources.language_mismatch",
                domain=resolved.domain,
                source=resolved.source_id,
                type=resolved.source_type,
                accepted=allowed[0],
                present=", ".join(present),
            )
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
        raise DataSourceConfigError(t("server.data_sources.bad_id", source=source_id))
    if not isinstance(entry, dict):
        raise DataSourceConfigError(t("server.data_sources.source_object", source=source_id))
    _validate_title(source_id, entry.get("title"))
    source_type = entry.get("type")
    if source_type not in SOURCE_TYPES:
        raise DataSourceConfigError(
            t("server.data_sources.source_type", source=source_id, type=source_type, allowed=", ".join(SOURCE_TYPES))
        )
    block_key = _SOURCE_BLOCK[str(source_type)]
    for key in entry:
        if key not in {"title", "type", block_key}:
            raise DataSourceConfigError(t("server.data_sources.extra_key", source=source_id, key=key))
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
        raise DataSourceConfigError(t("server.data_sources.title_required", source=source_id))
    if len(title.strip()) > TITLE_MAX_LEN:
        raise DataSourceConfigError(t("server.data_sources.title_too_long", source=source_id, max=TITLE_MAX_LEN))
    if any(ord(ch) < 32 for ch in title):
        raise DataSourceConfigError(t("server.data_sources.title_control_chars", source=source_id))


def _validate_grafana(source_id: str, block: object) -> None:
    _require_url(source_id, block, "grafana", "base_url", "grafana.base_url")
    assert isinstance(block, dict)
    _reject_unknown(source_id, block, {"base_url", "verify_ssl", "auth"})
    if "verify_ssl" in block and not isinstance(block.get("verify_ssl"), bool):
        raise DataSourceConfigError(t("server.data_sources.verify_ssl", source=source_id))
    auth = block.get("auth")
    if not isinstance(auth, dict):
        raise DataSourceConfigError(t("server.data_sources.grafana_auth_required", source=source_id))
    _reject_unknown(source_id, auth, {"method", "username", "password", "token"})
    method = auth.get("method")
    if method not in AUTH_METHODS:
        raise DataSourceConfigError(t("server.data_sources.grafana_auth_method", source=source_id))
    for key in ("username", "password", "token"):
        _optional_text(source_id, auth, key)


def _require_url(source_id: str, block: object, block_name: str, field: str, label: str) -> None:
    if not isinstance(block, dict):
        raise DataSourceConfigError(t("server.data_sources.block_object", source=source_id, block=block_name))
    url = block.get(field)
    if not isinstance(url, str) or not url.strip():
        raise DataSourceConfigError(t("server.data_sources.field_required", source=source_id, field=label))


def _reject_unknown(source_id: str, block: dict, allowed: set[str]) -> None:
    for key in block:
        if key not in allowed:
            raise DataSourceConfigError(t("server.data_sources.extra_key", source=source_id, key=key))


def _optional_text(source_id: str, block: dict, field: str) -> None:
    if field in block and not isinstance(block.get(field), str):
        raise DataSourceConfigError(t("server.data_sources.field_string", source=source_id, field=field))


def _one_binding(key: str, raw: object, catalog: dict, place: str, missing_kind: str, name: str = "") -> list[str]:
    """Problems of one binding; ``name`` is how messages call it (a row of a domain), else the key."""
    if key not in BINDING_KEYS:
        return [_scoped(place, t("server.data_sources.unknown_binding", key=key))]
    title = name or key
    if not isinstance(raw, dict):
        return [_scoped(place, t("server.data_sources.binding_object", key=title))]
    texts, text_error = _binding_texts(title, raw, place)
    if text_error:
        return [text_error]
    source_id = texts["source"]
    entry = catalog.get(source_id) if source_id else None
    if not source_id or not isinstance(entry, dict):
        return [_missing_source(source_id, title, place, missing_kind)]
    source_type = str(entry.get("type") or "")
    if source_type not in SOURCE_TYPES:
        return [_scoped(place, t("server.data_sources.domain_source_type", domain=title, source=source_id, type=source_type))]
    allowed = _fields_for(source_type)
    for field_name, value in texts.items():
        if field_name not in allowed and value:
            return [_scoped(place, t("server.data_sources.binding_field_unused", key=title, field=field_name, type=source_type))]
    if source_type == "grafana_proxy" and not texts["datasource_uid"] and not texts["datasource_name"]:
        return [_scoped(place, t("server.data_sources.binding_datasource", key=title))]
    return []


def _binding_texts(key: str, raw: dict, place: str) -> tuple[dict[str, str], str]:
    texts: dict[str, str] = {}
    for field_name in _BINDING_TEXT_FIELDS:
        value = raw.get(field_name, "")
        if field_name not in raw:
            texts[field_name] = ""
            continue
        if not isinstance(value, str):
            return {}, _scoped(place, t("server.data_sources.binding_field_string", key=key, field=field_name))
        texts[field_name] = value.strip()
    for field_name in raw:
        if field_name not in _BINDING_TEXT_FIELDS:
            return {}, _scoped(place, t("server.data_sources.binding_extra_key", key=key, field=field_name))
    return texts, ""


def _fields_for(source_type: str) -> frozenset[str]:
    if source_type == "prometheus":
        return _PROMETHEUS_FIELDS
    if source_type == "influxdb":
        return _INFLUX_FIELDS
    return _GRAFANA_FIELDS


def _missing_source(source_id: str, domain: str, place: str, missing_kind: str) -> str:
    if missing_kind == "used" and source_id:
        where = place or t("server.data_sources.place.bindings")
        return t("server.data_sources.source_in_use", source=source_id, where=where, domain=domain)
    if source_id:
        return _scoped(place, t("server.data_sources.domain_source_missing", domain=domain, source=source_id))
    return _scoped(place, t("server.data_sources.binding_no_source", key=domain))


def _scoped(place: str, message: str) -> str:
    if not place or place == t("server.data_sources.place.bindings"):
        return message[0].upper() + message[1:] if message else message
    return f"{place}: {message}"


def _parse_binding(domain: str, raw: dict) -> DomainBinding:
    def text(field_name: str) -> str:
        value = raw.get(field_name, "")
        return value.strip() if isinstance(value, str) else ""

    source_id = text("source")
    if not source_id:
        raise DataSourceConfigError(t("server.data_sources.domain_not_selected", domain=domain))
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
    "GRAFANA_LANG_PRIORITY",
    "LT_DOMAIN",
    "QUERY_KEY_FIELDS",
    "QUERY_LANG_FIELDS",
    "SOURCES_KEY",
    "SOURCE_ID_RE",
    "SOURCE_TYPES",
    "TITLE_MAX_LEN",
    "DataSourceConfigError",
    "DomainBinding",
    "ResolvedSource",
    "SourceGroup",
    "binding_errors",
    "binding_identity",
    "binding_messages",
    "domain_binding",
    "domain_query_language",
    "domain_rows_language",
    "domain_source_groups",
    "mixed_language_warnings",
    "normalize_binding",
    "normalize_row_sources",
    "own_row_sources",
    "pick_row_language",
    "resolve_binding",
    "resolve_domain_source",
    "resolve_row_source",
    "row_languages",
    "row_name",
    "row_overrides",
    "row_source_messages",
    "source_accepts_language",
    "validate_data_sources",
    "validate_domain_sources",
]
