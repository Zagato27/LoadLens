# SPDX-License-Identifier: Apache-2.0
"""Substitutes pack variables into labels, queries, series keys and SLA values."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from AI.i18n import t

from .models import TEMPLATE_RE, QueryPack, localized
from .roles import SLA_FIELDS, SLA_VALUE_CHOICES

_LEFTOVER_RE = re.compile(r"\{\{|\}\}")


class PackRenderError(ValueError):
    """Variable values that do not fit the pack, or a template that is left half-rendered."""


@dataclass(frozen=True)
class RenderedQuery:
    label: str
    query: str
    keys: tuple[str, ...]
    role: Optional[str]
    unit: str
    required: bool
    description: str
    since: int = 1
    description_en: str = ""

    def text(self, lang: Optional[str] = None) -> str:
        """The description in ``lang`` (default: the current locale)."""
        return localized(self, "description", lang)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "query": self.query,
            "keys": list(self.keys),
            "role": self.role,
            "unit": self.unit,
            "required": self.required,
            "description": self.text(),
            "since": self.since,
        }


@dataclass(frozen=True)
class RenderedPack:
    pack: QueryPack
    variables: dict[str, str]
    queries: tuple[RenderedQuery, ...]
    # SLA field -> label of the query that fills it (only roles present in ``queries``).
    sla_refs: dict[str, str]
    # latency_unit / load_model -> value.
    sla_values: dict[str, str]

    def by_label(self) -> dict[str, RenderedQuery]:
        return {query.label: query for query in self.queries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack.id,
            "version": self.pack.version,
            "domain": self.pack.domain,
            "language": self.pack.language,
            "variables": dict(self.variables),
            "queries": [query.to_dict() for query in self.queries],
            "sla_refs": dict(self.sla_refs),
            "sla_values": dict(self.sla_values),
        }


def resolve_variables(pack: QueryPack, given: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    """Pack defaults overridden by ``given``; each value is checked against its pattern and choices."""
    supplied = dict(given or {})
    unknown = sorted(set(supplied) - set(pack.variables))
    if unknown:
        raise PackRenderError(t("packs.error.unknown_variables", names=", ".join(unknown)))
    values: dict[str, str] = {}
    for name, var in pack.variables.items():
        raw = supplied.get(name, var.default)
        value = "" if raw is None else str(raw).strip()
        if var.choices and value not in var.choices:
            raise PackRenderError(t("packs.error.variable_choice", title=localized(var, "title"), choices=", ".join(var.choices), value=value))
        if "\n" in value or "\r" in value or not re.fullmatch(var.pattern, value):
            raise PackRenderError(t("packs.error.variable_value", title=localized(var, "title"), value=value))
        values[name] = value
    return values


def _substitute(text: str, values: Mapping[str, str], where: str) -> str:
    def replace(match: re.Match) -> str:
        return values[match.group(1)]

    rendered = TEMPLATE_RE.sub(replace, text)
    if _LEFTOVER_RE.search(rendered):
        raise PackRenderError(t("packs.error.leftover_braces", where=where))
    return rendered


def render_pack(
    pack: QueryPack,
    variables: Optional[Mapping[str, Any]] = None,
    include: Optional[set[str]] = None,
) -> RenderedPack:
    """Renders the pack. ``include`` keeps only queries whose rendered label is in it."""
    values = resolve_variables(pack, variables)
    queries: list[RenderedQuery] = []
    for query in pack.queries:
        label = _substitute(query.label, values, query.label).strip()
        if include is not None and label not in include:
            continue
        queries.append(RenderedQuery(
            label=label,
            query=_substitute(query.query, values, label).strip(),
            keys=tuple(_substitute(key, values, label).strip() for key in query.keys),
            role=query.role,
            unit=query.unit,
            required=query.required,
            description=query.description.strip(),
            since=query.since,
            description_en=query.description_en.strip(),
        ))
    by_role = {}
    for query in queries:
        if query.role and query.role not in by_role:
            by_role[query.role] = query.label
    sla_refs: dict[str, str] = {}
    sla_values: dict[str, str] = {}
    for field, ref in pack.sla.items():
        if field in SLA_FIELDS:
            label = by_role.get(ref[1:])
            if label:
                sla_refs[field] = label
        else:
            value = _substitute(ref, values, f"sla.{field}")
            if value not in SLA_VALUE_CHOICES.get(field, (value,)):
                raise PackRenderError(t("packs.error.sla_value_choice", field=field, choices=", ".join(SLA_VALUE_CHOICES[field]), value=value))
            sla_values[field] = value
    return RenderedPack(pack=pack, variables=values, queries=tuple(queries), sla_refs=sla_refs, sla_values=sla_values)


__all__ = ["PackRenderError", "RenderedPack", "RenderedQuery", "render_pack", "resolve_variables"]
