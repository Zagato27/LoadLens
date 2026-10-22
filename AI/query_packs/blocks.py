# SPDX-License-Identifier: Apache-2.0
"""Operations on one domain block of ``queries``: applying a pack, role lookups, SLA references.

A block keeps its rows in parallel arrays (``labels`` plus a query and series-key array per language).
Packs add two optional keys that the collection code ignores:

  ``roles``  {label: role} — what a series means; survives as long as the label exists
  ``packs``  [{"id", "version", "variables"}] — installed packs, enough to re-render their metadata

The settings page may also write ``descriptions`` {label: text} — what a row means, written by the user for
the model; it wins over the description of the pack query with the same label.

``sources`` {label: binding | None} — rows that read another source than the domain (see AI.data_sources).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional

from AI.data_sources import (
    QUERY_KEY_FIELDS,
    SOURCES_KEY,
    binding_identity,
    normalize_binding,
    row_languages,
    row_overrides,
)
from AI.i18n import t

from .catalog import Catalog, load_catalog
from .models import TEMPLATE_RE, QueryPack, localized
from .render import PackRenderError, RenderedPack, RenderedQuery, render_pack, resolve_variables
from .roles import LIVE_ONLY_ROLES, ROLES, SLA_FIELDS, UNIT_TITLES, Role

LANG_FIELDS: dict[str, tuple[str, str]] = {
    "promql": ("promql_queries", "label_keys_list"),
    "influxql": ("influxql_queries", "label_tag_keys_list"),
    "flux": ("flux_queries", "label_tag_keys_list"),
}
PARALLEL_KEYS: tuple[str, ...] = (
    "labels", "promql_queries", "label_keys_list", "influxql_queries", "flux_queries", "label_tag_keys_list",
)
ROLES_KEY = "roles"
PACKS_KEY = "packs"
DESCRIPTIONS_KEY = "descriptions"
DESCRIPTION_MAX_CHARS = 500
LATENCY_SLA_FIELDS: frozenset[str] = frozenset({"p95_query", "p99_query", "mean_latency_query"})
LOAD_MODEL_SLA_FIELDS: frozenset[str] = frozenset({"vus_query", "mean_latency_query"})
MergeMode = Literal["merge", "replace"]


class BlockConflictError(ValueError):
    """The pack cannot be merged into the block without replacing it."""


def _list(block: Mapping[str, Any], key: str) -> list:
    value = block.get(key)
    return list(value) if isinstance(value, list) else []


def block_labels(block: Optional[Mapping[str, Any]]) -> list[str]:
    if not isinstance(block, Mapping):
        return []
    return [str(label or "") for label in _list(block, "labels")]


def filled_languages(block: Optional[Mapping[str, Any]]) -> list[str]:
    """Languages with at least one non-empty query in the block."""
    if not isinstance(block, Mapping):
        return []
    return [lang for lang, (qkey, _) in LANG_FIELDS.items() if any(str(q or "").strip() for q in _list(block, qkey))]


def block_roles(block: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """{label: role} restricted to labels that still exist in the block."""
    if not isinstance(block, Mapping):
        return {}
    raw = block.get(ROLES_KEY)
    if not isinstance(raw, Mapping):
        return {}
    labels = set(block_labels(block))
    return {str(label): str(role) for label, role in raw.items() if str(label) in labels and isinstance(role, str)}


def block_descriptions(block: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """{label: description} the user wrote for the model, restricted to labels that still exist in the block."""
    if not isinstance(block, Mapping):
        return {}
    raw = block.get(DESCRIPTIONS_KEY)
    if not isinstance(raw, Mapping):
        return {}
    labels = set(block_labels(block))
    return {
        str(label): text.strip()
        for label, text in raw.items()
        if str(label) in labels and isinstance(text, str) and text.strip()
    }


def labels_for_role(block: Optional[Mapping[str, Any]], role: str) -> list[str]:
    roles = block_roles(block)
    return [label for label in block_labels(block) if roles.get(label) == role]


def installed_packs(block: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(block, Mapping):
        return []
    out = []
    for entry in _list(block, PACKS_KEY):
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            variables = entry.get("variables") if isinstance(entry.get("variables"), Mapping) else {}
            version = entry.get("version") if isinstance(entry.get("version"), int) else 0
            out.append({"id": entry["id"], "version": version, "variables": {str(k): str(v) for k, v in variables.items()}})
    return out


def instance_key(pack: QueryPack, variables: Optional[Mapping[str, Any]] = None) -> str:
    """Value of the pack's instance variable ("" for a pack installed once per domain)."""
    name = pack.instance_variable
    if not name or name not in pack.variables:
        return ""
    value = (variables or {}).get(name)
    if value is None:
        value = pack.variables[name].default
    return str(value).strip()


def installed_instances(block: Optional[Mapping[str, Any]], pack: QueryPack) -> list[dict[str, Any]]:
    """Installed entries of ``pack`` in the block, each with its ``instance`` key."""
    return [
        {**entry, "instance": instance_key(pack, entry["variables"])}
        for entry in installed_packs(block)
        if entry["id"] == pack.id
    ]


def _same_installation(entry: Mapping[str, Any], pack: QueryPack, variables: Mapping[str, Any]) -> bool:
    if entry.get("id") != pack.id:
        return False
    return instance_key(pack, entry.get("variables") or {}) == instance_key(pack, variables)


def _rendered_installed(block: Mapping[str, Any], catalog: Catalog) -> list[RenderedPack]:
    rendered = []
    for entry in installed_packs(block):
        pack = catalog.pack(entry["id"])
        if pack is None:
            continue
        variables = {name: value for name, value in entry["variables"].items() if name in pack.variables}
        try:
            rendered.append(render_pack(pack, variables))
        except PackRenderError:
            continue
    return rendered


def query_meta(block: Optional[Mapping[str, Any]], catalog: Optional[Catalog] = None) -> dict[str, RenderedQuery]:
    """{label: pack query} for block rows that come from installed packs (unit, description, role)."""
    if not isinstance(block, Mapping):
        return {}
    labels = set(block_labels(block))
    meta: dict[str, RenderedQuery] = {}
    for rendered in _rendered_installed(block, catalog or load_catalog()):
        for query in rendered.queries:
            if query.label in labels and query.label not in meta:
                meta[query.label] = query
    return meta


@dataclass(frozen=True)
class _RowNote:
    """What the block says about one row: its role, unit and description, pack or hand-written."""

    label: str
    index: int
    role: Optional[Role]
    unit: str
    description: str
    from_pack: bool


def _row_notes(block: Mapping[str, Any], catalog: Optional[Catalog]) -> list[_RowNote]:
    """One note per row in block order. The unit of a pack query wins over the unit of the role (it is the
    unit the query returns); a hand-written description wins over the pack's."""
    meta = query_meta(block, catalog)
    roles = block_roles(block)
    descriptions = block_descriptions(block)
    notes: list[_RowNote] = []
    for index, label in enumerate(block_labels(block)):
        role = ROLES.get(roles.get(label, ""))
        query = meta.get(label)
        unit = query.unit if query is not None and query.unit else (role.unit if role is not None else "")
        description = descriptions.get(label) or (query.text() if query is not None and query.description else "")
        notes.append(_RowNote(label, index, role, unit, description, query is not None))
    return notes


def series_annotations(block: Optional[Mapping[str, Any]], catalog: Optional[Catalog] = None) -> dict[str, dict[str, str]]:
    """{label: {"meaning", "unit", "description"}} for the analysis context of the domain.

    Text only: the numeric catalog the findings are verified against reads series statistics, not these.
    Written in the current locale: the pipeline runs inside ``with_locale(report_language)``.
    """
    if not isinstance(block, Mapping):
        return {}
    out: dict[str, dict[str, str]] = {}
    for note in _row_notes(block, catalog):
        item: dict[str, str] = {}
        if note.role is not None:
            item["meaning"] = note.role.title
        if note.unit:
            item["unit"] = UNIT_TITLES.get(note.unit, note.unit)
        if note.description:
            item["description"] = note.description
        if item:
            out[note.label] = item
    return out


def _row_keys(block: Mapping[str, Any], index: int) -> list[str]:
    """Series keys of the row in the first language it has a query text in."""
    for lang in row_languages(block, index):
        values = _list(block, QUERY_KEY_FIELDS[lang])
        keys = values[index] if index < len(values) else []
        return [str(key) for key in keys if str(key or "").strip()] if isinstance(keys, list) else []
    return []


def series_inventory(block: Optional[Mapping[str, Any]], catalog: Optional[Catalog] = None) -> list[dict[str, Any]]:
    """Every row of the block that has a query, as the model should know it, in block order.

    Items: ``label``, ``role`` (name, "" when none), ``meaning`` (role title), ``unit`` (code) and
    ``unit_title``, ``keys`` (series keys), ``description`` and ``origin`` (``pack`` for a row an installed
    pack renders, ``custom`` for a row added by hand). Text in the current locale.
    """
    if not isinstance(block, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for note in _row_notes(block, catalog):
        if not note.label or not row_languages(block, note.index):
            continue
        out.append({
            "label": note.label,
            "role": note.role.name if note.role is not None else "",
            "meaning": note.role.title if note.role is not None else "",
            "unit": note.unit,
            "unit_title": UNIT_TITLES.get(note.unit, note.unit) if note.unit else "",
            "keys": _row_keys(block, note.index),
            "description": note.description,
            "origin": "pack" if note.from_pack else "custom",
        })
    return out


def prompt_addenda(block: Optional[Mapping[str, Any]], catalog: Optional[Catalog] = None) -> list[tuple[str, str]]:
    """(pack title, text) of installed packs that describe their metrics for the model, in the current locale."""
    if not isinstance(block, Mapping):
        return []
    labels = set(block_labels(block))
    out = []
    seen: set[str] = set()
    for rendered in _rendered_installed(block, catalog or load_catalog()):
        pack = rendered.pack
        text = localized(pack, "prompt_addendum").strip()
        # Instances of one pack (two database hosts) share the note.
        if text and pack.id not in seen and any(query.label in labels for query in rendered.queries):
            seen.add(pack.id)
            out.append((localized(pack, "title"), text))
    return out


def run_filter_variables(pack: QueryPack, variables: Mapping[str, Any], run_tag: str) -> Optional[dict[str, str]]:
    """Full variable set with the pack's run filter applied, or ``None`` when the pack has no run filter."""
    rule = pack.run_filter
    if rule is None:
        return None
    values = resolve_variables(pack, {k: v for k, v in variables.items() if k in pack.variables})
    current = values.get(rule.variable, "").strip()
    added = TEMPLATE_RE.sub(lambda _match: run_tag, rule.value)
    values[rule.variable] = f"{current}{rule.join}{added}" if rule.join and current else added
    return values


def apply_run_filter(
    queries: Mapping[str, Any],
    run_tag: str,
    packs: Optional[Iterable[str]] = None,
    catalog: Optional[Catalog] = None,
) -> tuple[dict[str, Any], list[str]]:
    """Copy of ``queries`` where rows of load tool packs read only the metrics of one run.

    ``packs`` limits the filter to these pack ids: only packs whose tool really tagged the run. Rows keep
    their labels (stored series ids); only the query text changes. A row whose text differs from what its
    pack renders was edited by hand and stays as it is. Returns the copy and the ids of filtered packs.
    """
    out = copy.deepcopy(dict(queries or {}))
    tag = str(run_tag or "").strip()
    wanted = set(packs) if packs is not None else None
    if not tag or wanted == set():
        return out, []
    known = catalog or load_catalog()
    applied: list[str] = []
    for block in out.values():
        if not isinstance(block, dict):
            continue
        for entry in installed_packs(block):
            pack = known.pack(entry["id"])
            if pack is None or pack.run_filter is None or (wanted is not None and pack.id not in wanted):
                continue
            try:
                original = resolve_variables(pack, {k: v for k, v in entry["variables"].items() if k in pack.variables})
                filtered = run_filter_variables(pack, original, tag)
                before = render_pack(pack, original)
                after = render_pack(pack, filtered)
            except PackRenderError:
                continue
            qkey, _ = LANG_FIELDS[pack.language]
            rows = _list(block, qkey)
            positions = {label: index for index, label in enumerate(block_labels(block))}
            changed = False
            for old, new in zip(before.queries, after.queries):
                index = positions.get(old.label)
                if index is None or index >= len(rows):
                    continue
                if str(rows[index] or "").strip() != old.query.strip():
                    continue
                rows[index] = new.query
                changed = True
            if changed:
                block[qkey] = rows
                applied.append(pack.id)
        # A summed row follows its filtered components.
        refresh_derived(block)
    return out, applied


DERIVED_KEY = "derived"
SUMMED_RPS_LABEL = "LT: RPS total, all tools"
SUMMED_ROLE = "lt.rps_total"


class DerivedRowError(ValueError):
    """The summed row cannot be built for this block; the message is for the user."""


def strip_live_rows(queries: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """A copy of ``queries`` without the rows of live-only roles: the report does not read them."""
    out = copy.deepcopy(dict(queries or {}))
    for block in out.values():
        if not isinstance(block, dict):
            continue
        roles = block_roles(block)
        drop = {label for label, role in roles.items() if role in LIVE_ONLY_ROLES}
        if not drop:
            continue
        labels = block_labels(block)
        keep = [index for index, label in enumerate(labels) if label not in drop]
        for key in PARALLEL_KEYS:
            values = block.get(key)
            if isinstance(values, list) and len(values) == len(labels):
                block[key] = [values[index] for index in keep]
        kept_roles = {label: role for label, role in roles.items() if label not in drop}
        if kept_roles:
            block[ROLES_KEY] = kept_roles
        else:
            block.pop(ROLES_KEY, None)
        if DESCRIPTIONS_KEY in block:
            kept_notes = {label: text for label, text in block_descriptions(block).items() if label not in drop}
            if kept_notes:
                block[DESCRIPTIONS_KEY] = kept_notes
            else:
                block.pop(DESCRIPTIONS_KEY, None)
    return out


def derived_rows(block: Optional[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """{label: {"kind": "sum", "of": [labels]}} of rows computed from other rows of the block."""
    raw = block.get(DERIVED_KEY) if isinstance(block, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for label, meta in raw.items():
        if isinstance(meta, Mapping) and meta.get("kind") == "sum" and isinstance(meta.get("of"), list):
            out[str(label)] = {"kind": "sum", "of": [str(item) for item in meta["of"] if isinstance(item, str)]}
    return out


def summed_query(parts: list[str]) -> str:
    """PromQL sum of single-series queries: each part gets its own label so ``or`` keeps them all."""
    branches = [f'label_replace({part.strip()}, "ll_part", "{index}", "", "")' for index, part in enumerate(parts, start=1)]
    return f"sum by (series) ({' or '.join(branches)})"


def refresh_derived(block: dict[str, Any]) -> dict[str, Any]:
    """Recomputes derived rows from their components (after a run filter or a pack update). Mutates ``block``."""
    derived = derived_rows(block)
    if not derived or "promql" not in filled_languages(block):
        return block
    rows = _rows(block, "promql_queries", "label_keys_list")
    queries = _list(block, "promql_queries")
    positions = {label: index for index, label in enumerate(block_labels(block))}
    for label, meta in derived.items():
        parts = [rows[item][0] for item in meta["of"] if item in rows and item != label and rows[item][0].strip()]
        index = positions.get(label)
        if len(parts) >= 2 and index is not None and index < len(queries):
            queries[index] = summed_query(parts)
    block["promql_queries"] = queries
    return block


def add_summed_rps(
    block: Optional[Mapping[str, Any]],
    domain: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """Copy of the block with a row summing the RPS of every load tool; (block, label, component labels).

    For a complex run (k6 for REST and Gatling for Kafka writing to one PromQL source) the steps and SLA
    can then follow the whole load instead of one tool.
    """
    if not isinstance(block, Mapping):
        raise DerivedRowError(t("packs.error.domain_empty"))
    current = _aligned(block, ("labels", "promql_queries", "label_keys_list"))
    derived = derived_rows(current)
    candidates = [label for label in labels_for_role(current, SUMMED_ROLE) if label not in derived]
    if len(candidates) < 2:
        raise DerivedRowError(t("packs.error.summed_needs_two", role=ROLES[SUMMED_ROLE].title))
    # The sum is one PromQL query: its parts are PromQL rows that read one source.
    texts = _rows(current, "promql_queries", "label_keys_list")
    parts = [label for label in candidates if texts[label][0].strip()]
    if len(parts) < 2:
        raise DerivedRowError(t("packs.error.summed_promql_only"))
    overrides = row_overrides(current)
    # ``domain`` is the domain binding: a part whose own binding equals it reads the same source.
    if len({_source_identity(overrides.get(label), domain) for label in parts}) > 1:
        raise DerivedRowError(t("packs.error.summed_one_source"))
    labels = current["labels"]
    if SUMMED_RPS_LABEL not in labels:
        for key in PARALLEL_KEYS:
            if key in current:
                current[key].append(_blank(key))
        current["labels"][-1] = SUMMED_RPS_LABEL
    index = current["labels"].index(SUMMED_RPS_LABEL)
    current["label_keys_list"][index] = ["series"]
    # The row runs PromQL: a text of another language left in it from earlier is dropped.
    for lang, (other_key, _keys) in LANG_FIELDS.items():
        if lang != "promql" and isinstance(current.get(other_key), list) and index < len(current[other_key]):
            current[other_key][index] = ""
    derived[SUMMED_RPS_LABEL] = {"kind": "sum", "of": parts}
    current[DERIVED_KEY] = derived
    # The sum reads the source of its parts.
    set_row_sources(current, [SUMMED_RPS_LABEL], overrides.get(parts[0]))
    refresh_derived(current)
    return current, SUMMED_RPS_LABEL, parts


def set_row_sources(block: dict[str, Any], labels: Iterable[str], binding: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Points the rows ``labels`` at ``binding``; ``None`` returns them to the domain binding. Mutates ``block``.

    A row returned to the domain binding keeps an explicit ``None`` only when it had an entry: the entry
    may come from a wider scope, and ``None`` keeps it from coming back when the scopes are merged.
    """
    current = block.get(SOURCES_KEY)
    sources = dict(current) if isinstance(current, Mapping) else {}
    for label in labels:
        if binding is None:
            if label in sources:
                sources[label] = None
        else:
            sources[label] = normalize_binding(binding)
    if sources:
        block[SOURCES_KEY] = sources
    else:
        block.pop(SOURCES_KEY, None)
    return block


def prune_block(block: dict[str, Any], catalog: Optional[Catalog] = None) -> dict[str, Any]:
    """Drops roles, descriptions and sources of deleted rows and packs none of whose rows are left.

    Mutates and returns ``block``.
    """
    if not isinstance(block, dict):
        return block
    if SOURCES_KEY in block:
        labels = set(block_labels(block))
        raw = block.get(SOURCES_KEY)
        kept = {str(label): value for label, value in raw.items() if str(label) in labels} if isinstance(raw, Mapping) else {}
        if kept:
            block[SOURCES_KEY] = kept
        else:
            block.pop(SOURCES_KEY, None)
    if DERIVED_KEY in block:
        labels = set(block_labels(block))
        kept = {label: meta for label, meta in derived_rows(block).items()
                if label in labels and len([item for item in meta["of"] if item in labels]) >= 2}
        if kept:
            block[DERIVED_KEY] = kept
            refresh_derived(block)
        else:
            block.pop(DERIVED_KEY, None)
    if ROLES_KEY in block:
        roles = block_roles(block)
        if roles:
            block[ROLES_KEY] = roles
        else:
            block.pop(ROLES_KEY, None)
    if DESCRIPTIONS_KEY in block:
        descriptions = block_descriptions(block)
        if descriptions:
            block[DESCRIPTIONS_KEY] = descriptions
        else:
            block.pop(DESCRIPTIONS_KEY, None)
    if PACKS_KEY in block:
        labels = set(block_labels(block))
        known = catalog or load_catalog()
        kept = []
        for entry in installed_packs(block):
            pack = known.pack(entry["id"])
            if pack is None:
                kept.append(entry)
                continue
            try:
                rendered = render_pack(pack, {k: v for k, v in entry["variables"].items() if k in pack.variables})
            except PackRenderError:
                kept.append(entry)
                continue
            if any(query.label in labels for query in rendered.queries):
                kept.append(entry)
        if kept:
            block[PACKS_KEY] = kept
        else:
            block.pop(PACKS_KEY, None)
    return block


def prune_queries(queries: dict[str, Any], catalog: Optional[Catalog] = None) -> dict[str, Any]:
    for block in queries.values():
        if isinstance(block, dict):
            prune_block(block, catalog)
    return queries


@dataclass
class MergePlan:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def to_dict(self) -> dict[str, list[str]]:
        return {"added": self.added, "updated": self.updated, "unchanged": self.unchanged, "removed": self.removed}


def _row_count(block: Mapping[str, Any]) -> int:
    return max((len(_list(block, key)) for key in PARALLEL_KEYS), default=0)


def _blank(key: str) -> Any:
    return [] if key.endswith("_list") else ""


def _aligned(block: Mapping[str, Any], required: Iterable[str]) -> dict[str, Any]:
    """Copy of the block with every used parallel array padded to the same length."""
    out = copy.deepcopy(dict(block))
    count = _row_count(out)
    needed = set(required)
    for key in PARALLEL_KEYS:
        values = _list(out, key)
        if not values and key not in needed:
            continue
        while len(values) < count:
            values.append(_blank(key))
        out[key] = values
    return out


def _source_identity(binding: Optional[Mapping[str, Any]], domain: Optional[Mapping[str, Any]]) -> tuple:
    """What a row reads: its own binding, else the domain binding (``()`` when that is unknown)."""
    if binding:
        return binding_identity(binding)
    return binding_identity(domain) if domain else ()


def merge_conflict(
    block: Optional[Mapping[str, Any]],
    language: str,
    binding: Optional[Mapping[str, Any]] = None,
    exclude: Iterable[str] = (),
    domain: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Why rows in ``language`` cannot read the source ``binding`` (``None``: the domain binding) in the block.

    One source runs one language (a Grafana datasource is Prometheus or InfluxDB): rows in another
    language that read the same source would fail. Rows with another source do not count, nor the rows
    ``exclude`` names (the ones a pack is about to rewrite). ``domain`` is the domain binding: a row whose
    own binding equals it reads the same source as the rows without one.
    """
    if not isinstance(block, Mapping):
        return None
    overrides = row_overrides(block)
    target = _source_identity(binding, domain)
    skip = set(exclude)
    present: list[str] = []
    for index, label in enumerate(block_labels(block)):
        if label in skip:
            continue
        if _source_identity(overrides.get(label), domain) != target:
            continue
        for lang in row_languages(block, index):
            if lang != language and lang not in present:
                present.append(lang)
    if not present:
        return None
    names = ", ".join(lang_title(lang) for lang in present)
    return t("packs.error.one_language", languages=names)


def lang_title(lang: str) -> str:
    return {"promql": "PromQL", "influxql": "InfluxQL", "flux": "Flux"}.get(lang, lang)


def merge_pack(
    block: Optional[Mapping[str, Any]],
    rendered: RenderedPack,
    mode: MergeMode = "merge",
    source: Optional[Mapping[str, Any]] = None,
    domain: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, Any], MergePlan]:
    """Block with the rendered queries applied.

    ``merge`` keeps foreign rows and updates rows whose label the pack owns in place, so stored
    series keep their history. Rows of other languages stay: a domain may read several sources.
    It refuses only when rows of another language read the same source as the pack rows
    (``source``; ``None`` is the domain binding). ``replace`` leaves only the pack rows.
    """
    language = rendered.pack.language
    qkey, kkey = LANG_FIELDS[language]
    current = dict(block) if isinstance(block, Mapping) else {}
    plan = MergePlan()
    old_rows = _rows(current, qkey, kkey)
    if mode == "replace":
        out: dict[str, Any] = {"labels": [], qkey: [], kkey: []}
        for query in rendered.queries:
            out["labels"].append(query.label)
            out[qkey].append(query.query)
            out[kkey].append(list(query.keys))
            _classify(plan, query, old_rows.get(query.label))
        plan.removed = [label for label in old_rows if label not in rendered.by_label()]
        packs: list[dict[str, Any]] = []
        roles: dict[str, str] = {}
        # Rows the pack rewrites keep the source they read and the description the user wrote for them.
        old_sources = current.get(SOURCES_KEY)
        if isinstance(old_sources, Mapping):
            kept_sources = {label: copy.deepcopy(value) for label, value in old_sources.items() if label in rendered.by_label()}
            if kept_sources:
                out[SOURCES_KEY] = kept_sources
        kept_notes = {label: text for label, text in block_descriptions(current).items() if label in rendered.by_label()}
        if kept_notes:
            out[DESCRIPTIONS_KEY] = kept_notes
    else:
        conflict = merge_conflict(current, language, source, exclude=rendered.by_label(), domain=domain)
        if conflict:
            raise BlockConflictError(conflict)
        out = _aligned(current, ("labels", qkey, kkey))
        positions = {label: index for index, label in enumerate(block_labels(out))}
        for query in rendered.queries:
            index = positions.get(query.label)
            _classify(plan, query, old_rows.get(query.label))
            if index is None:
                for key in PARALLEL_KEYS:
                    if key in out:
                        out[key].append(_blank(key))
                index = len(out["labels"]) - 1
                out["labels"][index] = query.label
                positions[query.label] = index
            out[qkey][index] = query.query
            out[kkey][index] = list(query.keys)
            # A row runs one language: the pack's text replaces a text of another language in the row.
            for lang, (other_key, _keys) in LANG_FIELDS.items():
                if lang != language and isinstance(out.get(other_key), list) and index < len(out[other_key]):
                    out[other_key][index] = ""
        # The same installation is replaced; other instances of a multi-instance pack stay.
        packs = [entry for entry in installed_packs(current) if not _same_installation(entry, rendered.pack, rendered.variables)]
        roles = block_roles(current)
    for query in rendered.queries:
        if query.role:
            roles[query.label] = query.role
    packs.append({"id": rendered.pack.id, "version": rendered.pack.version, "variables": dict(rendered.variables)})
    out[ROLES_KEY] = roles
    out[PACKS_KEY] = packs
    return prune_block(out), plan


def _rows(block: Mapping[str, Any], qkey: str, kkey: str) -> dict[str, tuple[str, list[str]]]:
    labels = block_labels(block)
    queries = _list(block, qkey)
    keys = _list(block, kkey)
    rows = {}
    for index, label in enumerate(labels):
        query = str(queries[index]) if index < len(queries) else ""
        row_keys = [str(k) for k in keys[index]] if index < len(keys) and isinstance(keys[index], list) else []
        rows[label] = (query, row_keys)
    return rows


def _classify(plan: MergePlan, query: RenderedQuery, old: Optional[tuple[str, list[str]]]) -> None:
    if old is None:
        plan.added.append(query.label)
    elif old[0].strip() == query.query.strip() and old[1] == list(query.keys):
        plan.unchanged.append(query.label)
    else:
        plan.updated.append(query.label)


@dataclass
class SlaUpdate:
    sla: dict[str, Any]
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    kept: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"changes": self.changes, "kept": self.kept, "warnings": self.warnings}


def sla_update(
    current: Optional[Mapping[str, Any]],
    rendered: RenderedPack,
    overwrite: bool = False,
    pack_labels: Optional[Iterable[str]] = None,
    queries: Optional[Mapping[str, Any]] = None,
) -> SlaUpdate:
    """SLA with the pack's role queries filled in.

    A filled field keeps its value unless ``overwrite``. A reference to a query that is not in
    ``queries`` (the queries after the pack is applied) is treated as empty: it is replaced and the
    change is marked ``"reason": "missing"``. ``latency_unit`` follows the pack only when every latency
    reference then points at a query of this pack; otherwise the units would mix. ``load_model``
    follows when the pack set the VU or mean-latency reference.
    """
    from .roles import role_for_sla_field

    sla = copy.deepcopy(dict(current or {}))
    update = SlaUpdate(sla=sla)
    for field_name, label in rendered.sla_refs.items():
        old = str(sla.get(field_name) or "").strip()
        if old == label:
            continue
        role = role_for_sla_field(field_name)
        missing = bool(old) and queries is not None and role is not None and old not in block_labels(queries.get(role.domain))
        if old and not overwrite and not missing:
            update.kept[field_name] = old
            continue
        sla[field_name] = label
        update.changes[field_name] = {"from": old, "to": label}
        if missing:
            update.changes[field_name]["reason"] = "missing"
    own = set(pack_labels or ()) | {query.label for query in rendered.queries}
    written = set(update.changes)
    unit = rendered.sla_values.get("latency_unit")
    if unit and written & LATENCY_SLA_FIELDS:
        foreign = [f for f in sorted(LATENCY_SLA_FIELDS) if str(sla.get(f) or "").strip() and str(sla.get(f)).strip() not in own]
        if foreign:
            update.warnings.append(t("packs.warning.latency_unit_kept", fields=", ".join(foreign)))
        elif str(sla.get("latency_unit") or "") != unit:
            update.changes["latency_unit"] = {"from": sla.get("latency_unit"), "to": unit}
            sla["latency_unit"] = unit
    model = rendered.sla_values.get("load_model")
    if model and written & LOAD_MODEL_SLA_FIELDS and str(sla.get("load_model") or "") != model:
        update.changes["load_model"] = {"from": sla.get("load_model"), "to": model}
        sla["load_model"] = model
    return update


def update_selection(block: Optional[Mapping[str, Any]], rendered: RenderedPack, installed_version: int) -> list[int]:
    """Positions of pack queries to pre-select when the pack is opened again or updated.

    Rows still present in the block stay selected; rows the user removed stay unselected; queries
    added by pack versions newer than the installed one are offered.
    """
    labels = set(block_labels(block))
    return [
        index for index, query in enumerate(rendered.queries)
        if query.label in labels or query.since > installed_version
    ]


TITLE_KEY = "title"
TITLE_MAX_LEN = 60


def domain_title(block: Optional[Mapping[str, Any]], catalog: Optional[Catalog] = None) -> str:
    """Title of the domain in reports: the block's own ``title``, else the titles of its packs, else "".

    Pack titles are in the current locale (the report language during a run).
    """
    if not isinstance(block, Mapping):
        return ""
    own = block.get(TITLE_KEY)
    if isinstance(own, str) and own.strip():
        return own.strip()[:TITLE_MAX_LEN]
    labels = set(block_labels(block))
    titles: list[str] = []
    for rendered in _rendered_installed(block, catalog or load_catalog()):
        title = localized(rendered.pack, "domain_title").strip()
        if title and title not in titles and any(query.label in labels for query in rendered.queries):
            titles.append(title)
    return " · ".join(titles)


def domain_titles(queries: Optional[Mapping[str, Any]], defaults: Mapping[str, str], catalog: Optional[Catalog] = None) -> dict[str, str]:
    """{domain: title} for domains whose title differs from ``defaults`` (what reports should store)."""
    out: dict[str, str] = {}
    for domain, block in (queries or {}).items():
        title = domain_title(block, catalog)
        if title and title != defaults.get(domain, domain):
            out[str(domain)] = title
    return out


def _reference_alive(label: str, field_name: str, queries: Mapping[str, Any]) -> bool:
    """Whether an SLA reference names a query of its role's domain (fields without a role always are)."""
    from .roles import role_for_sla_field

    role = role_for_sla_field(field_name)
    if role is None:
        return True
    return label in block_labels(queries.get(role.domain))


def resolve_sla_refs(
    sla: Optional[Mapping[str, Any]],
    queries: Optional[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """SLA with empty query references taken from query roles, and {field: label} of what was filled.

    An explicit reference to an existing query always wins. A reference to a query that no longer
    exists (renamed, deleted, replaced by a pack) is taken from the role like an empty one; without a
    role query it stays as it was. ``latency_unit`` follows the role unit only when no live latency
    reference was set explicitly, so a hand-made seconds query never gets an "ms" unit.
    """
    from .roles import role_for_sla_field

    out = copy.deepcopy(dict(sla or {}))
    queries = queries or {}
    filled: dict[str, str] = {}
    explicit_latency = any(
        str(out.get(name) or "").strip() and _reference_alive(str(out.get(name)).strip(), name, queries)
        for name in LATENCY_SLA_FIELDS
    )
    latency_units: set[str] = set()
    for field_name in SLA_FIELDS:
        current = str(out.get(field_name) or "").strip()
        if current and _reference_alive(current, field_name, queries):
            continue
        role = role_for_sla_field(field_name)
        if role is None:
            continue
        labels = labels_for_role(queries.get(role.domain), role.name)
        if not labels:
            continue
        out[field_name] = labels[0]
        filled[field_name] = labels[0]
        if field_name in LATENCY_SLA_FIELDS:
            latency_units.add(role.unit)
    if latency_units and not explicit_latency and len(latency_units) == 1:
        unit = {"ms": "ms", "seconds": "s"}.get(latency_units.pop())
        if unit:
            out["latency_unit"] = unit
    return out, filled


def missing_sla_references(
    sla: Optional[Mapping[str, Any]],
    queries: Optional[Mapping[str, Any]],
    fields: Iterable[str],
) -> list[str]:
    """SLA fields that are empty or name a query absent from the domain of their role."""
    from .roles import role_for_sla_field

    sla = sla or {}
    queries = queries or {}
    missing = []
    for field_name in fields:
        if field_name not in SLA_FIELDS:
            continue
        label = str(sla.get(field_name) or "").strip()
        role = role_for_sla_field(field_name)
        block = queries.get(role.domain) if role else None
        if not label or label not in block_labels(block):
            missing.append(field_name)
    return missing


__all__ = [
    "DERIVED_KEY",
    "DerivedRowError",
    "SUMMED_RPS_LABEL",
    "BlockConflictError",
    "add_summed_rps",
    "derived_rows",
    "refresh_derived",
    "summed_query",
    "LANG_FIELDS",
    "MergePlan",
    "PACKS_KEY",
    "PARALLEL_KEYS",
    "ROLES_KEY",
    "DESCRIPTIONS_KEY",
    "DESCRIPTION_MAX_CHARS",
    "SlaUpdate",
    "apply_run_filter",
    "block_descriptions",
    "block_labels",
    "block_roles",
    "filled_languages",
    "installed_instances",
    "installed_packs",
    "instance_key",
    "labels_for_role",
    "lang_title",
    "merge_conflict",
    "merge_pack",
    "missing_sla_references",
    "prompt_addenda",
    "prune_block",
    "prune_queries",
    "query_meta",
    "resolve_sla_refs",
    "run_filter_variables",
    "series_annotations",
    "series_inventory",
    "set_row_sources",
    "sla_update",
    "update_selection",
    "domain_title",
    "domain_titles",
]
