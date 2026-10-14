"""Query packs: ready-made metric queries for common stacks (exporters, runtimes, load tools).

A pack is a TOML file in ``packs/``: queries for one domain in one language, with roles, units and
variables. Applying a pack renders it and merges the rows into the ``queries`` block of the domain,
then points the SLA references at the pack queries. See ``docs/query-packs.md`` for the catalog.
"""

from .blocks import (
    BlockConflictError,
    MergePlan,
    SlaUpdate,
    block_labels,
    block_roles,
    filled_languages,
    installed_packs,
    labels_for_role,
    merge_conflict,
    merge_pack,
    missing_sla_references,
    prompt_addenda,
    prune_block,
    prune_queries,
    query_meta,
    resolve_sla_refs,
    series_annotations,
    sla_update,
    domain_title,
    domain_titles,
    update_selection,
)
from .catalog import Catalog, PackCatalogError, load_catalog
from .models import CATEGORY_TITLES, LANGUAGE_TITLES, Bundle, QueryPack
from .render import PackRenderError, RenderedPack, RenderedQuery, render_pack, resolve_variables
from .roles import READINESS_SLA_FIELDS, ROLES, SLA_FIELDS, UNIT_TITLES, Role, role_for_sla_field

__all__ = [
    "BlockConflictError",
    "Bundle",
    "CATEGORY_TITLES",
    "Catalog",
    "LANGUAGE_TITLES",
    "MergePlan",
    "PackCatalogError",
    "PackRenderError",
    "QueryPack",
    "READINESS_SLA_FIELDS",
    "ROLES",
    "RenderedPack",
    "RenderedQuery",
    "Role",
    "SLA_FIELDS",
    "SlaUpdate",
    "UNIT_TITLES",
    "block_labels",
    "block_roles",
    "filled_languages",
    "installed_packs",
    "labels_for_role",
    "load_catalog",
    "merge_conflict",
    "merge_pack",
    "missing_sla_references",
    "prompt_addenda",
    "prune_block",
    "prune_queries",
    "query_meta",
    "render_pack",
    "resolve_variables",
    "role_for_sla_field",
    "resolve_sla_refs",
    "series_annotations",
    "sla_update",
    "domain_title",
    "domain_titles",
    "update_selection",
]
