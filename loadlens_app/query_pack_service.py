# SPDX-License-Identifier: Apache-2.0
"""Query packs in the settings: catalog with compatibility, preview, check and apply for one scope.

The scope is the one the «Запросы» section edits: global defaults, a project area, or a service
inside an area. Reading and writing the scope is passed in by the blueprint, so this module holds
only pack logic and stays testable without Flask.

Texts (pack titles, descriptions, role and unit titles, errors) are in the current locale: the
request's language in a web request.
"""

from __future__ import annotations

import copy
import fnmatch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from AI.data_sources import (
    BINDING_DOMAINS,
    DataSourceConfigError,
    binding_identity,
    domain_binding,
    normalize_binding,
    resolve_binding,
    resolve_domain_source,
    row_overrides,
    row_source_messages,
    source_accepts_language,
)
from AI.i18n import get_locale, t, with_locale
from AI.query_packs import (
    CATEGORY_TITLES,
    HOST_DOMAINS,
    LANGUAGE_TITLES,
    LIVE_ONLY_ROLES,
    ROLES,
    UNIT_TITLES,
    BlockConflictError,
    Bundle,
    PackRenderError,
    QueryPack,
    RenderedPack,
    installed_instances,
    instance_key,
    load_catalog,
    localized,
    merge_conflict,
    merge_pack,
    render_pack,
    set_row_sources,
    sla_update,
    update_selection,
)
from loadlens_app.config_schema import QueryConfigError, SlaConfigError, validate_queries, validate_sla
from loadlens_app.query_preview import preview_metric_query

CHECK_WORKERS = 4
MODES = ("merge", "replace")

STATUS_OK = "ok"
STATUS_REPLACE_ONLY = "replace_only"
STATUS_INCOMPATIBLE = "incompatible"


class PackRequestError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Compatibility:
    status: str
    reason: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason, "source": self.source}


@dataclass
class ScopeState:
    """Stored values of the scope: what the settings page shows for ``queries`` and ``sla``."""

    queries: dict[str, Any]
    sla: dict[str, Any]
    effective_cfg: dict[str, Any]


@dataclass
class ApplyResult:
    pack: QueryPack
    rendered: RenderedPack
    domain: str
    mode: str
    queries: dict[str, Any]
    plan: dict[str, list[str]]
    sla: dict[str, Any]
    sla_report: dict[str, Any]
    sla_changed: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        everything = render_pack(self.pack, self.rendered.variables)
        return {
            "pack_id": self.pack.id,
            "version": self.pack.version,
            "domain": self.domain,
            "instance": instance_key(self.pack, self.rendered.variables),
            "mode": self.mode,
            "plan": self.plan,
            "sla": self.sla_report,
            "sla_changed": self.sla_changed,
            "warnings": self.warnings,
            "queries": [query.to_dict() for query in self.rendered.queries],
            # Every query of the pack with the request's variables, in pack order, for the page table.
            "all_queries": [query.to_dict() for query in everything.queries],
        }


def _sentence(text: str) -> str:
    """Upper-cases the first letter only; str.capitalize() would lower-case «InfluxQL»."""
    return text[:1].upper() + text[1:]


def _source_title(cfg: Mapping[str, Any], source_id: str) -> str:
    entry = (cfg.get("data_sources") or {}).get(source_id) if isinstance(cfg.get("data_sources"), Mapping) else None
    title = entry.get("title") if isinstance(entry, Mapping) else ""
    return str(title or source_id)


def binding_title(cfg: Mapping[str, Any], binding: Mapping[str, Any]) -> str:
    """How the page names a binding: the source title, plus the Grafana datasource, database or bucket."""
    parts = [_source_title(cfg, str(binding.get("source") or ""))]
    datasource = str(binding.get("datasource_name") or binding.get("datasource_uid") or "").strip()
    for value in (datasource, str(binding.get("database") or "").strip(), str(binding.get("bucket") or "").strip()):
        if value:
            parts.append(value)
    return " · ".join(parts)


def request_binding(state: "ScopeState", body: Mapping[str, Any], domain: str) -> tuple[bool, Optional[dict[str, str]]]:
    """(given, binding) of a request: the source the pack rows read.

    Absent: the rows keep the sources they have. ``null`` (or the domain binding itself): the domain
    binding. An object: that source, checked against the catalog.
    """
    if "binding" not in body:
        return False, None
    raw = body.get("binding")
    if raw is None or (isinstance(raw, Mapping) and not str(raw.get("source") or "").strip()):
        return True, None
    if not isinstance(raw, Mapping):
        raise PackRequestError(t("packs.error.binding_object"))
    binding = normalize_binding(raw)
    problems = row_source_messages({domain: {"labels": ["?"], "sources": {"?": binding}}}, state.effective_cfg.get("data_sources"))
    if problems:
        raise PackRequestError(problems[0])
    if binding_identity(binding) == binding_identity(domain_binding(state.effective_cfg, domain)):
        return True, None
    return True, binding


def _installed_binding(block: Any, rendered: RenderedPack) -> Optional[dict]:
    """The source the installed rows of the pack read, when they read another one than the domain."""
    overrides = row_overrides(block)
    for query in rendered.queries:
        if query.label in overrides:
            return normalize_binding(overrides[query.label])
    return None


def compatibility(
    cfg: Mapping[str, Any],
    queries: Mapping[str, Any],
    pack: QueryPack,
    domain: Optional[str] = None,
    binding: Optional[Mapping[str, Any]] = None,
    exclude: Optional[list[str]] = None,
) -> Compatibility:
    """Whether the source (the domain binding, or ``binding``) runs the pack language and the block can take it.

    ``exclude``: labels the pack rewrites (by default those of the pack with its default variables).
    """
    target = domain or pack.target_domains[0]
    try:
        resolved = resolve_binding(cfg, target, binding) if binding else resolve_domain_source(dict(cfg), target)
    except DataSourceConfigError as exc:
        return Compatibility(STATUS_INCOMPATIBLE, t("packs.error.domain_unbound", error=exc))
    source = _source_title(cfg, resolved.source_id)
    lang = LANGUAGE_TITLES[pack.language]
    if not source_accepts_language(resolved.source_type, pack.language):
        return Compatibility(STATUS_INCOMPATIBLE, t("packs.error.source_language", source=source, language=lang), source)
    if exclude is None:
        exclude = [query.label for query in render_pack(pack).queries]
    conflict = merge_conflict(queries.get(target), pack.language, binding, exclude=exclude, domain=domain_binding(cfg, target))
    if conflict:
        return Compatibility(STATUS_REPLACE_ONLY, t("packs.error.replace_only", reason=_sentence(conflict)), source)
    return Compatibility(STATUS_OK, "", source)


def target_domain(pack: QueryPack, raw: Any) -> str:
    """Domain the request installs the pack into: required for a pack whose domain is picked at install time."""
    value = str(raw or "").strip()
    if not pack.domains:
        if value and value != pack.domain:
            raise PackRequestError(t("packs.error.domain_not_allowed", domain=value, domains=pack.domain))
        return pack.domain
    if not value:
        raise PackRequestError(t("packs.error.domain_required", domains=", ".join(pack.domains)))
    if value not in pack.domains:
        raise PackRequestError(t("packs.error.domain_not_allowed", domain=value, domains=", ".join(pack.domains)))
    return value


def _variable_view(name: str, var) -> dict[str, Any]:
    return {
        "name": name,
        "title": localized(var, "title"),
        "default": var.default,
        "choices": list(var.choices),
        "help": localized(var, "help"),
        "pattern": var.pattern,
    }


def _query_view(query) -> dict[str, Any]:
    out = query.to_dict()
    role = ROLES.get(query.role or "")
    out["unit_title"] = UNIT_TITLES.get(query.unit, query.unit)
    out["role_title"] = role.title if role else ""
    out["sla_field"] = role.sla_field if role else None
    return out


def _installed_views(pack: QueryPack, block: Any, cfg: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """Each installation of the pack in the block: version, variables, the source its rows read, plus the
    queries to pre-select when it is opened again. A multi-instance pack has one entry per instance."""
    views = []
    for installed in installed_instances(block, pack):
        variables = {name: value for name, value in installed["variables"].items() if name in pack.variables}
        try:
            rendered = render_pack(pack, variables)
        except PackRenderError:
            rendered = render_pack(pack)
        binding = _installed_binding(block, rendered)
        views.append({
            **installed,
            "variables": variables,
            "outdated": installed["version"] < pack.version,
            "selected_indexes": update_selection(block, rendered, installed["version"]),
            "new_queries": sum(1 for query in rendered.queries if query.since > installed["version"]),
            # None: the rows read the domain binding.
            "binding": binding,
            "source": binding_title(cfg or {}, binding) if binding else "",
        })
    return views


def _target_view(state: "ScopeState", pack: QueryPack, domain: str) -> dict[str, Any]:
    instances = _installed_views(pack, state.queries.get(domain), state.effective_cfg)
    return {
        "compatibility": compatibility(state.effective_cfg, state.queries, pack, domain).to_dict(),
        "installed": instances[0] if instances else None,
        "instances": instances,
    }


def _bundle_view(bundle: Bundle) -> dict[str, Any]:
    return {
        "id": bundle.id,
        "title": localized(bundle, "title"),
        "description": localized(bundle, "description"),
        "packs": list(bundle.packs),
    }


def _ordered_packs(packs: list[QueryPack]) -> list[QueryPack]:
    """Catalog order (category, then title) by the titles of the current locale; in ru it is the catalog order."""
    categories = {name: index for index, name in enumerate(CATEGORY_TITLES)}
    return sorted(packs, key=lambda pack: (categories.get(pack.category, len(categories)), localized(pack, "title").lower()))


def catalog_view(state: ScopeState) -> dict[str, Any]:
    catalog = load_catalog()
    packs = []
    for pack in _ordered_packs(list(catalog.packs.values())):
        rendered = render_pack(pack)
        targets = {domain: _target_view(state, pack, domain) for domain in pack.target_domains}
        first = targets[pack.target_domains[0]]
        packs.append({
            "id": pack.id,
            "version": pack.version,
            "title": localized(pack, "title"),
            "category": pack.category,
            "stack": list(pack.stack),
            # The first target domain for a pack whose domain is picked; ``domains`` lists all of them.
            "domain": pack.target_domains[0],
            "domains": list(pack.target_domains),
            "domain_picked": bool(pack.domains),
            "max_instances": pack.max_instances,
            "instance_variable": pack.instance_variable,
            "targets": targets,
            "language": pack.language,
            "language_title": LANGUAGE_TITLES[pack.language],
            "description": " ".join(localized(pack, "description").split()),
            "requires_metrics": list(pack.requires_metrics),
            "variables": [_variable_view(name, var) for name, var in pack.variables.items()],
            "queries": [_query_view(query) for query in rendered.queries],
            "sla_fields": sorted(rendered.sla_refs),
            "compatibility": first["compatibility"],
            "installed": first["installed"],
            "domain_title": localized(pack, "domain_title"),
            "verified": pack.verified,
        })
    return {
        "packs": packs,
        "bundles": [_bundle_view(bundle) for bundle in catalog.bundles.values()],
        "categories": dict(CATEGORY_TITLES),
        "roles": {
            name: {
                "title": role.title,
                "unit": UNIT_TITLES.get(role.unit, role.unit),
                "domain": role.domain,
                "sla_field": role.sla_field,
                "live_only": name in LIVE_ONLY_ROLES,
            }
            for name, role in ROLES.items()
        },
        # Domains where roles without a domain (resources of a host) fit, as AI.query_packs.role_fits decides.
        "host_domains": list(HOST_DOMAINS),
    }


def _pack(pack_id: str) -> QueryPack:
    pack = load_catalog().pack(str(pack_id or "").strip())
    if pack is None:
        raise PackRequestError(t("packs.error.pack_not_found", pack=pack_id), 404)
    return pack


def _include(raw: Any) -> Optional[set[str]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise PackRequestError(t("packs.error.include_list"))
    return {str(item) for item in raw}


def _include_indexes(raw: Any, count: int) -> set[int]:
    if not isinstance(raw, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw):
        raise PackRequestError(t("packs.error.include_indexes_list"))
    if any(item < 0 or item >= count for item in raw):
        raise PackRequestError(t("packs.error.include_indexes_range"))
    return set(raw)


def _render(body: Mapping[str, Any]) -> RenderedPack:
    """Renders the pack with the request's variables and selection.

    The page selects queries by position (``include_indexes``): labels may contain variables and
    change while the form is edited. API clients may select by rendered label (``include``).
    """
    pack = _pack(body.get("pack_id"))
    variables = body.get("variables") or {}
    if not isinstance(variables, Mapping):
        raise PackRequestError(t("packs.error.variables_object"))
    try:
        include = _include(body.get("include"))
        if body.get("include_indexes") is not None:
            positions = _include_indexes(body.get("include_indexes"), len(pack.queries))
            full = render_pack(pack, variables)
            include = {query.label for index, query in enumerate(full.queries) if index in positions}
        rendered = render_pack(pack, variables, include=include)
    except PackRenderError as exc:
        raise PackRequestError(str(exc)) from exc
    if not rendered.queries:
        raise PackRequestError(t("packs.error.nothing_selected"))
    return rendered


def _check_instance_limit(block: Any, rendered: RenderedPack, domain: str) -> None:
    """A new instance of a multi-instance pack must fit ``max_instances``; updating an instance always fits."""
    pack = rendered.pack
    instances = installed_instances(block, pack)
    key = instance_key(pack, rendered.variables)
    if any(item["instance"] == key for item in instances):
        return
    if len(instances) >= pack.max_instances:
        raise PackRequestError(
            t("packs.error.too_many_instances", max=pack.max_instances, domain=t("common.domain." + domain)),
            409,
        )


def plan_apply(state: ScopeState, body: Mapping[str, Any]) -> ApplyResult:
    """What applying the pack would store; nothing is written here."""
    rendered = _render(body)
    pack = rendered.pack
    domain = target_domain(pack, body.get("domain"))
    mode = str(body.get("mode") or "merge")
    if mode not in MODES:
        raise PackRequestError(t("packs.error.mode", modes=", ".join(MODES)))
    given, binding = request_binding(state, body, domain)
    source = binding if given else _installed_binding(state.queries.get(domain), rendered)
    labels = [query.label for query in rendered.queries]
    compat = compatibility(state.effective_cfg, state.queries, pack, domain, source, exclude=labels)
    if compat.status == STATUS_INCOMPATIBLE:
        raise PackRequestError(compat.reason, 409)
    if mode == "merge":
        _check_instance_limit(state.queries.get(domain), rendered, domain)
    try:
        block, plan = merge_pack(
            state.queries.get(domain), rendered, mode, source, domain_binding(state.effective_cfg, domain),  # type: ignore[arg-type]
        )
    except BlockConflictError as exc:
        raise PackRequestError(t("packs.error.choose_replace", reason=_sentence(str(exc))), 409) from exc
    # Without a binding in the request the pack rows keep reading the source of the installation, new rows included.
    if given or source:
        set_row_sources(block, [query.label for query in rendered.queries], binding if given else source)
    queries = copy.deepcopy(state.queries)
    queries[domain] = block
    try:
        validate_queries(queries)
    except QueryConfigError as exc:
        raise PackRequestError(t("packs.error.queries_invalid", error=exc)) from exc
    sla = copy.deepcopy(state.sla)
    sla_report: dict[str, Any] = {"changes": {}, "kept": {}, "warnings": []}
    warnings: list[str] = []
    if bool(body.get("fill_sla", True)) and (rendered.sla_refs or rendered.sla_values):
        # The pack's queries replace SLA references by default; references to deleted queries always.
        update = sla_update(state.sla, rendered, overwrite=bool(body.get("overwrite_sla", True)), queries=queries)
        try:
            validate_sla(update.sla)
        except SlaConfigError as exc:
            raise PackRequestError(str(exc)) from exc
        sla = update.sla
        sla_report = update.to_dict()
        warnings.extend(update.warnings)
    return ApplyResult(
        pack=pack,
        rendered=rendered,
        domain=domain,
        mode=mode,
        queries=queries,
        plan=plan.to_dict(),
        sla=sla,
        sla_report=sla_report,
        sla_changed=bool(sla_report["changes"]),
        warnings=warnings,
    )


def check_pack(
    state: ScopeState,
    body: Mapping[str, Any],
    start_ts: Optional[float],
    end_ts: Optional[float],
    preview: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Runs every selected query of the pack against the domain source."""
    preview = preview or preview_metric_query
    rendered = _render(body)
    pack = rendered.pack
    domain = target_domain(pack, body.get("domain"))
    given, binding = request_binding(state, body, domain)
    source = binding if given else _installed_binding(state.queries.get(domain), rendered)
    compat = compatibility(state.effective_cfg, state.queries, pack, domain, source)
    if compat.status == STATUS_INCOMPATIBLE:
        raise PackRequestError(compat.reason, 409)
    # Passed only when the rows read another source: the domain binding is the preview's default.
    extra: dict[str, Any] = {"binding": source} if source else {}

    # Worker threads do not see the request: they answer in the locale of the request that started them.
    locale = get_locale()

    def run(query) -> dict[str, Any]:
        with with_locale(locale):
            result = preview(
                state.effective_cfg,
                domain=domain,
                lang=pack.language,
                query=query.query,
                label_keys=list(query.keys),
                start_ts=start_ts,
                end_ts=end_ts,
                **extra,
            )
        return {"label": query.label, "role": query.role, "required": query.required, **result.to_dict()}

    with ThreadPoolExecutor(max_workers=CHECK_WORKERS) as pool:
        results = list(pool.map(run, rendered.queries))
    missing_required = [item["label"] for item in results if item["required"] and not item["ok"]]
    return {
        "pack_id": pack.id,
        "results": results,
        "ok": not missing_required,
        "missing_required": missing_required,
        "source": compat.source,
    }


def _metric_present(required: str, names: frozenset[str]) -> bool:
    """An exact name, or a pattern with ``*`` for metric names that carry a variable part.

    Graphite paths stored in VictoriaMetrics hold the simulation name, e.g. ``gatling.*.allRequests.all.count``.
    """
    if "*" not in required:
        return required in names
    return any(fnmatch.fnmatchcase(name, required) for name in names)


def _score(pack: QueryPack, result: Any) -> dict[str, Any]:
    names = result.names if result is not None else frozenset()
    language_ok = result is not None and result.language in ("", pack.language)
    found = [name for name in pack.requires_metrics if _metric_present(name, names)] if language_ok else []
    required = len(pack.requires_metrics)
    return {
        "found": found,
        "missing": [name for name in pack.requires_metrics if name not in found],
        "score": round(len(found) / required, 2) if required else 0.0,
    }


EXTRA_PREFIX = "source:"


def _extra_sources(state: ScopeState) -> dict[str, dict[str, str]]:
    """Sources to read besides the domain bindings: the sources rows already read, and every direct
    Prometheus source of the catalog (it needs no datasource to be listed). Keyed ``source:<n>``."""
    cfg = state.effective_cfg
    bindings: list[dict[str, str]] = []
    for block in (state.queries or {}).values():
        for binding in row_overrides(block).values():
            bindings.append(normalize_binding(binding))
    catalog = cfg.get("data_sources") if isinstance(cfg.get("data_sources"), Mapping) else {}
    for source_id, entry in catalog.items():
        if isinstance(entry, Mapping) and entry.get("type") == "prometheus":
            bindings.append(normalize_binding({"source": str(source_id)}))
    bound = {binding_identity(domain_binding(cfg, domain)) for domain in BINDING_DOMAINS}
    out: dict[str, dict[str, str]] = {}
    seen: set[tuple] = set()
    for binding in bindings:
        identity = binding_identity(binding)
        if identity in seen or identity in bound:
            continue
        seen.add(identity)
        out[f"{EXTRA_PREFIX}{len(out) + 1}"] = binding
    return out


def detect_packs(state: ScopeState, discover: Optional[Callable[..., Any]] = None) -> dict[str, Any]:
    """Which packs fit the metrics the sources hold: share of ``requires_metrics`` found per pack.

    A pack scores against its own domain's source (the best of its domains when the domain is picked).
    Sources no domain is bound to (catalog Prometheus sources, the sources rows read) are read as well:
    a pack that fits one of them better gets it as ``alternative`` with the binding to install it with.
    A bundle fits as well as its packs on average.
    """
    if discover is None:
        from loadlens_app.metric_discovery import discover
    catalog = load_catalog()
    domains = sorted({domain for pack in catalog.packs.values() for domain in pack.target_domains})
    extra = _extra_sources(state)
    found_names = discover(state.effective_cfg, domains, extra) if extra else discover(state.effective_cfg, domains)
    packs: dict[str, dict[str, Any]] = {}
    for pack in catalog.packs.values():
        # A pack whose domain is picked scores against the best of its domains' sources.
        best: Optional[dict[str, Any]] = None
        for domain in pack.target_domains:
            scored = _score(pack, found_names.get(domain))
            scored["domain"] = domain
            if best is None or scored["score"] > best["score"]:
                best = scored
        result = best or _score(pack, None)
        alternative: Optional[dict[str, Any]] = None
        for key, binding in extra.items():
            scored = _score(pack, found_names.get(key))
            if scored["score"] > result["score"] and (alternative is None or scored["score"] > alternative["score"]):
                alternative = {**scored, "binding": binding, "title": binding_title(state.effective_cfg, binding)}
        if alternative is not None:
            result["alternative"] = alternative
        packs[pack.id] = result
    bundles = {}
    for bundle in catalog.bundles.values():
        # A bundle installs into the domain bindings: alternatives do not count.
        scores = [packs[pack_id]["score"] for pack_id in bundle.packs]
        bundles[bundle.id] = {
            "matched": sum(1 for score in scores if score >= 1.0),
            "total": len(scores),
            "score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        }
    return {
        "domains": {domain: item.to_dict() for domain, item in found_names.items() if not domain.startswith(EXTRA_PREFIX)},
        "sources": {
            key: {**found_names[key].to_dict(), "title": binding_title(state.effective_cfg, binding), "binding": binding}
            for key, binding in extra.items()
            if key in found_names
        },
        "packs": packs,
        "bundles": bundles,
    }


__all__ = [
    "ApplyResult",
    "detect_packs",
    "Compatibility",
    "PackRequestError",
    "ScopeState",
    "catalog_view",
    "check_pack",
    "compatibility",
    "binding_title",
    "plan_apply",
    "request_binding",
    "target_domain",
]
