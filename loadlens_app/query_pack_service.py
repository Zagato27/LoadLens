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

from AI.data_sources import DataSourceConfigError, resolve_domain_source, source_accepts_language
from AI.i18n import get_locale, t, with_locale
from AI.query_packs import (
    CATEGORY_TITLES,
    LANGUAGE_TITLES,
    ROLES,
    UNIT_TITLES,
    BlockConflictError,
    Bundle,
    PackRenderError,
    QueryPack,
    RenderedPack,
    installed_packs,
    load_catalog,
    localized,
    merge_conflict,
    merge_pack,
    render_pack,
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
            "domain": self.pack.domain,
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


def compatibility(cfg: Mapping[str, Any], queries: Mapping[str, Any], pack: QueryPack) -> Compatibility:
    """Whether the domain binding runs the pack language and the domain block can take it."""
    try:
        resolved = resolve_domain_source(dict(cfg), pack.domain)
    except DataSourceConfigError as exc:
        return Compatibility(STATUS_INCOMPATIBLE, t("packs.error.domain_unbound", error=exc))
    source = _source_title(cfg, resolved.source_id)
    lang = LANGUAGE_TITLES[pack.language]
    if not source_accepts_language(resolved.source_type, pack.language):
        return Compatibility(STATUS_INCOMPATIBLE, t("packs.error.source_language", source=source, language=lang), source)
    conflict = merge_conflict(queries.get(pack.domain), pack.language)
    if conflict:
        return Compatibility(STATUS_REPLACE_ONLY, t("packs.error.replace_only", reason=_sentence(conflict)), source)
    return Compatibility(STATUS_OK, "", source)


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


def _installed_view(pack: QueryPack, block: Any) -> Optional[dict[str, Any]]:
    """Installed version and variables, plus the queries to pre-select when the pack is opened again."""
    installed = next((entry for entry in installed_packs(block) if entry["id"] == pack.id), None)
    if installed is None:
        return None
    variables = {name: value for name, value in installed["variables"].items() if name in pack.variables}
    try:
        rendered = render_pack(pack, variables)
    except PackRenderError:
        rendered = render_pack(pack)
    return {
        **installed,
        "variables": variables,
        "outdated": installed["version"] < pack.version,
        "selected_indexes": update_selection(block, rendered, installed["version"]),
        "new_queries": sum(1 for query in rendered.queries if query.since > installed["version"]),
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
        installed = _installed_view(pack, state.queries.get(pack.domain))
        packs.append({
            "id": pack.id,
            "version": pack.version,
            "title": localized(pack, "title"),
            "category": pack.category,
            "stack": list(pack.stack),
            "domain": pack.domain,
            "language": pack.language,
            "language_title": LANGUAGE_TITLES[pack.language],
            "description": " ".join(localized(pack, "description").split()),
            "requires_metrics": list(pack.requires_metrics),
            "variables": [_variable_view(name, var) for name, var in pack.variables.items()],
            "queries": [_query_view(query) for query in rendered.queries],
            "sla_fields": sorted(rendered.sla_refs),
            "compatibility": compatibility(state.effective_cfg, state.queries, pack).to_dict(),
            "installed": installed,
            "domain_title": localized(pack, "domain_title"),
            "verified": pack.verified,
        })
    return {
        "packs": packs,
        "bundles": [_bundle_view(bundle) for bundle in catalog.bundles.values()],
        "categories": dict(CATEGORY_TITLES),
        "roles": {
            name: {"title": role.title, "unit": UNIT_TITLES.get(role.unit, role.unit), "domain": role.domain, "sla_field": role.sla_field}
            for name, role in ROLES.items()
        },
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


def plan_apply(state: ScopeState, body: Mapping[str, Any]) -> ApplyResult:
    """What applying the pack would store; nothing is written here."""
    rendered = _render(body)
    pack = rendered.pack
    mode = str(body.get("mode") or "merge")
    if mode not in MODES:
        raise PackRequestError(t("packs.error.mode", modes=", ".join(MODES)))
    compat = compatibility(state.effective_cfg, state.queries, pack)
    if compat.status == STATUS_INCOMPATIBLE:
        raise PackRequestError(compat.reason, 409)
    try:
        block, plan = merge_pack(state.queries.get(pack.domain), rendered, mode)  # type: ignore[arg-type]
    except BlockConflictError as exc:
        raise PackRequestError(t("packs.error.choose_replace", reason=_sentence(str(exc))), 409) from exc
    queries = copy.deepcopy(state.queries)
    queries[pack.domain] = block
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
    compat = compatibility(state.effective_cfg, state.queries, pack)
    if compat.status == STATUS_INCOMPATIBLE:
        raise PackRequestError(compat.reason, 409)

    # Worker threads do not see the request: they answer in the locale of the request that started them.
    locale = get_locale()

    def run(query) -> dict[str, Any]:
        with with_locale(locale):
            result = preview(
                state.effective_cfg,
                domain=pack.domain,
                lang=pack.language,
                query=query.query,
                label_keys=list(query.keys),
                start_ts=start_ts,
                end_ts=end_ts,
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


def detect_packs(state: ScopeState, discover: Optional[Callable[..., Any]] = None) -> dict[str, Any]:
    """Which packs fit the metrics the bound sources hold: share of ``requires_metrics`` found per pack.

    A pack scores only against its own domain's source; a bundle fits as well as its packs on average.
    """
    if discover is None:
        from loadlens_app.metric_discovery import discover
    catalog = load_catalog()
    domains = sorted({pack.domain for pack in catalog.packs.values()})
    found_names = discover(state.effective_cfg, domains)
    packs: dict[str, dict[str, Any]] = {}
    for pack in catalog.packs.values():
        result = found_names.get(pack.domain)
        names = result.names if result is not None else frozenset()
        language_ok = result is not None and result.language in ("", pack.language)
        found = [name for name in pack.requires_metrics if _metric_present(name, names)] if language_ok else []
        required = len(pack.requires_metrics)
        packs[pack.id] = {
            "found": found,
            "missing": [name for name in pack.requires_metrics if name not in found],
            "score": round(len(found) / required, 2) if required else 0.0,
        }
    bundles = {}
    for bundle in catalog.bundles.values():
        scores = [packs[pack_id]["score"] for pack_id in bundle.packs]
        bundles[bundle.id] = {
            "matched": sum(1 for score in scores if score >= 1.0),
            "total": len(scores),
            "score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        }
    return {
        "domains": {domain: item.to_dict() for domain, item in found_names.items()},
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
    "plan_apply",
]
