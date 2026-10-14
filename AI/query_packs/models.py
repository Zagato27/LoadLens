"""Shape of a query pack file (``packs/<id>.toml``) and of a bundle (``bundles.toml``)."""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from AI.data_sources import BINDING_DOMAINS

from .roles import ROLES, SLA_FIELDS, SLA_VALUE_CHOICES, UNITS

PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
VAR_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
# {{name}}: double braces never occur in PromQL, InfluxQL or Flux, so templates cannot clash with
# the pipeline's own placeholders ($timeFilter, $__interval, {bucket}, {start}, {end}).
TEMPLATE_RE = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
DEFAULT_VALUE_PATTERN = r"^[^\n\r{}]*$"

Category = Literal["infrastructure", "runtime", "http", "database", "messaging", "load_tool"]
Language = Literal["promql", "influxql", "flux"]
CATEGORY_TITLES: dict[str, str] = {
    "load_tool": "Нагрузочный инструмент",
    "runtime": "Рантайм приложения",
    "http": "HTTP и gRPC",
    "database": "Базы данных",
    "messaging": "Kafka",
    "infrastructure": "Узлы и Kubernetes",
}
LANGUAGE_TITLES: dict[str, str] = {"promql": "PromQL", "influxql": "InfluxQL", "flux": "Flux"}


def template_names(text: str) -> set[str]:
    return set(TEMPLATE_RE.findall(text or ""))


class PackVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    default: str = ""
    pattern: str = DEFAULT_VALUE_PATTERN
    choices: list[str] = Field(default_factory=list)
    help: str = ""

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        re.compile(value)
        return value

    @model_validator(mode="after")
    def _default_is_valid(self) -> "PackVariable":
        if self.choices and self.default not in self.choices:
            raise ValueError(f"default «{self.default}» is not one of the choices")
        if not re.fullmatch(self.pattern, self.default):
            raise ValueError(f"default «{self.default}» does not match the pattern")
        return self


class PackQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=3, max_length=160)
    query: str = Field(min_length=3)
    keys: list[str] = Field(default_factory=list)
    role: Optional[str] = None
    unit: str
    required: bool = False
    description: str = ""
    # Pack version that introduced the query: an update offers queries newer than the installed version.
    since: int = Field(default=1, ge=1)

    @field_validator("unit")
    @classmethod
    def _known_unit(cls, value: str) -> str:
        if value not in UNITS:
            raise ValueError(f"unknown unit «{value}»")
        return value


class QueryPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    id: str
    version: int = Field(ge=1)
    title: str
    category: Category
    stack: list[str] = Field(default_factory=list)
    domain: str
    language: Language
    description: str
    requires_metrics: list[str] = Field(default_factory=list)
    prompt_addendum: str = ""
    # Title of the domain in reports when this pack fills it (Go and Python runtimes live in "jvm").
    domain_title: str = Field(default="", max_length=40)
    # Every query returned data on docker/dev-stack (real exporters); see docker/dev-stack/README.md.
    verified: bool = False
    variables: dict[str, PackVariable] = Field(default_factory=dict)
    queries: list[PackQuery] = Field(min_length=1)
    # SLA field -> "@role" of one of the pack queries; latency_unit / load_model -> value or template.
    sla: dict[str, str] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not PACK_ID_RE.fullmatch(value):
            raise ValueError("id must be lowercase letters, digits and dashes")
        return value

    @field_validator("domain")
    @classmethod
    def _known_domain(cls, value: str) -> str:
        if value not in BINDING_DOMAINS:
            raise ValueError(f"unknown domain «{value}»")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "QueryPack":
        for name in self.variables:
            if not VAR_NAME_RE.fullmatch(name):
                raise ValueError(f"variable name «{name}» must be lowercase snake_case")
        labels: set[str] = set()
        used: set[str] = set()
        for query in self.queries:
            if query.label in labels:
                raise ValueError(f"label «{query.label}» is repeated")
            labels.add(query.label)
            if query.since > self.version:
                raise ValueError(f"{query.label}: since {query.since} is newer than the pack version {self.version}")
            if query.role is not None:
                role = ROLES.get(query.role)
                if role is None:
                    raise ValueError(f"{query.label}: unknown role «{query.role}»")
                if role.domain != self.domain:
                    raise ValueError(f"{query.label}: role {query.role} belongs to domain {role.domain}")
                if role.unit != query.unit:
                    raise ValueError(f"{query.label}: role {query.role} is in {role.unit}, query says {query.unit}")
            for text in (query.label, query.query, *query.keys):
                used |= template_names(text)
        roles = {query.role for query in self.queries if query.role}
        for field, value in self.sla.items():
            if field in SLA_FIELDS:
                if not value.startswith("@") or value[1:] not in roles:
                    raise ValueError(f"sla.{field} must reference a role of this pack as @role")
                if ROLES[value[1:]].sla_field != field:
                    raise ValueError(f"sla.{field}: role {value[1:]} fills {ROLES[value[1:]].sla_field}")
            elif field in SLA_VALUE_CHOICES:
                used |= template_names(value)
                if not template_names(value) and value not in SLA_VALUE_CHOICES[field]:
                    raise ValueError(f"sla.{field}: «{value}» is not one of {SLA_VALUE_CHOICES[field]}")
            else:
                raise ValueError(f"sla.{field} is not a field a pack can set")
        missing = used - set(self.variables)
        if missing:
            raise ValueError(f"undeclared variables: {', '.join(sorted(missing))}")
        unused = set(self.variables) - used
        if unused:
            raise ValueError(f"unused variables: {', '.join(sorted(unused))}")
        return self


class Bundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str
    packs: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not PACK_ID_RE.fullmatch(value):
            raise ValueError("id must be lowercase letters, digits and dashes")
        return value


class BundleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundles: list[Bundle] = Field(default_factory=list)


__all__ = [
    "Bundle",
    "BundleFile",
    "CATEGORY_TITLES",
    "Category",
    "DEFAULT_VALUE_PATTERN",
    "LANGUAGE_TITLES",
    "Language",
    "PackQuery",
    "PackVariable",
    "QueryPack",
    "TEMPLATE_RE",
    "template_names",
]
