# SPDX-License-Identifier: Apache-2.0
"""Shape of a query pack file (``packs/<id>.toml``) and of a bundle (``bundles.toml``)."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from AI.data_sources import BINDING_DOMAINS
from AI.i18n import DEFAULT, get_locale, normalize_lang, t

from .roles import HOST_DOMAINS, ROLES, SLA_FIELDS, SLA_VALUE_CHOICES, UNITS, LocalizedTitles, role_fits

PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
VAR_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
# {{name}}: double braces never occur in PromQL, InfluxQL or Flux, so templates cannot clash with
# the pipeline's own placeholders ($timeFilter, $__interval, {bucket}, {start}, {end}).
TEMPLATE_RE = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
DEFAULT_VALUE_PATTERN = r"^[^\n\r{}]*$"

Category = Literal["infrastructure", "runtime", "http", "database", "messaging", "load_tool"]
Language = Literal["promql", "influxql", "flux"]


def category_title(category: str) -> str:
    return t("packs.category." + category)


# In catalog order: packs are sorted by category in this order, the settings page groups them the same way.
CATEGORY_TITLES = LocalizedTitles(("load_tool", "runtime", "http", "database", "messaging", "infrastructure"), category_title)
LANGUAGE_TITLES: dict[str, str] = {"promql": "PromQL", "influxql": "InfluxQL", "flux": "Flux"}


def template_names(text: str) -> set[str]:
    return set(TEMPLATE_RE.findall(text or ""))


def localized(obj: Any, name: str, lang: Optional[str] = None) -> str:
    """Text field ``name`` of a pack object in ``lang`` (default: the current locale).

    Russian is the original; another language reads the twin ``<name>_<lang>`` (``title_en``) and falls
    back to the original while the twin is empty. Labels and queries have no twins: they are identifiers.
    """
    code = normalize_lang(lang) or get_locale()
    original = getattr(obj, name)
    if code != DEFAULT:
        twin = getattr(obj, f"{name}_{code}", "")
        if isinstance(twin, str) and twin.strip():
            return twin
    return original


class PackVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    title_en: str = ""
    default: str = ""
    pattern: str = DEFAULT_VALUE_PATTERN
    choices: list[str] = Field(default_factory=list)
    help: str = ""
    help_en: str = ""

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
    description_en: str = ""
    # Pack version that introduced the query: an update offers queries newer than the installed version.
    since: int = Field(default=1, ge=1)

    @field_validator("unit")
    @classmethod
    def _known_unit(cls, value: str) -> str:
        if value not in UNITS:
            raise ValueError(f"unknown unit «{value}»")
        return value


RUN_TAG_VARIABLE = "run_tag"
# Installations of one pack in one domain told apart by ``instance_variable`` (database primary and replica).
MAX_INSTANCES = 10


class RunFilter(BaseModel):
    """How a run started from LoadLens is told apart in the tool's metrics.

    LoadLens starts the tool with the run tag (k6 ``--tag testid=...``, JMeter ``-Japplication=...``,
    the Gatling Graphite prefix). For the report of that run, the pack variable ``variable`` gets ``value``
    with ``{{run_tag}}`` substituted. With ``join`` set, a non-empty installed value is kept and the filter
    is appended after the separator (extra PromQL matchers); otherwise the filter replaces the value.
    """

    model_config = ConfigDict(extra="forbid")

    variable: str
    value: str
    join: str = ""


class QueryPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    id: str
    version: int = Field(ge=1)
    title: str
    title_en: str = ""
    category: Category
    stack: list[str] = Field(default_factory=list)
    # One domain, or ``domains`` the user picks from at install time (resources of a database or broker host).
    domain: str = ""
    domains: list[str] = Field(default_factory=list)
    # How many times the pack may be installed in one domain; each installation has its own value of
    # ``instance_variable``, which every label uses, so the rows of two installations never collide.
    max_instances: int = Field(default=1, ge=1, le=MAX_INSTANCES)
    instance_variable: str = ""
    language: Language
    description: str
    description_en: str = ""
    # Metric (Prometheus) or measurement (InfluxDB) names detection looks for; ``*`` matches any part,
    # e.g. Graphite paths in VictoriaMetrics: "gatling.*.allRequests.all.count".
    requires_metrics: list[str] = Field(default_factory=list)
    # Notes on the stack's metrics for the model, in the report language (prompt_addendum_en for en reports).
    prompt_addendum: str = ""
    prompt_addendum_en: str = ""
    # Title of the domain in reports when this pack fills it (Go and Python runtimes live in "jvm").
    domain_title: str = Field(default="", max_length=40)
    domain_title_en: str = Field(default="", max_length=40)
    # Every query returned data on docker/dev-stack (real exporters); see docker/dev-stack/README.md.
    verified: bool = False
    variables: dict[str, PackVariable] = Field(default_factory=dict)
    queries: list[PackQuery] = Field(min_length=1)
    # SLA field -> "@role" of one of the pack queries; latency_unit / load_model -> value or template.
    sla: dict[str, str] = Field(default_factory=dict)
    # Load tool packs only: the variable that selects one run started from LoadLens.
    run_filter: Optional[RunFilter] = None

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not PACK_ID_RE.fullmatch(value):
            raise ValueError("id must be lowercase letters, digits and dashes")
        return value

    @field_validator("domain")
    @classmethod
    def _known_domain(cls, value: str) -> str:
        if value and value not in BINDING_DOMAINS:
            raise ValueError(f"unknown domain «{value}»")
        return value

    @field_validator("domains")
    @classmethod
    def _host_domains(cls, value: list[str]) -> list[str]:
        for name in value:
            if name not in HOST_DOMAINS:
                raise ValueError(f"domain «{name}» cannot be chosen at install time; allowed: {', '.join(HOST_DOMAINS)}")
        if len(set(value)) != len(value):
            raise ValueError("domains repeat")
        return value

    @property
    def target_domains(self) -> tuple[str, ...]:
        """Domains the pack installs into: its ``domain``, or the ``domains`` picked from at install time."""
        return tuple(self.domains) if self.domains else (self.domain,)

    @property
    def multi_instance(self) -> bool:
        return self.max_instances > 1

    def _check_targets(self) -> None:
        if bool(self.domain) == bool(self.domains):
            raise ValueError("set either domain or domains")
        if self.domains:
            if len(self.domains) < 2:
                raise ValueError("domains needs two domains or more; use domain for one")
            if self.sla or self.run_filter is not None or self.domain_title or self.domain_title_en:
                raise ValueError("a pack with domains cannot set sla, run_filter or domain_title")
        if self.max_instances > 1:
            var = self.instance_variable
            if var not in self.variables:
                raise ValueError("instance_variable must name a variable of the pack")
            if re.fullmatch(self.variables[var].pattern, ""):
                raise ValueError(f"instance variable «{var}» must not accept an empty value")
            if self.sla or self.run_filter is not None:
                raise ValueError("a pack with several instances cannot set sla or run_filter")
            for query in self.queries:
                if var not in template_names(query.label):
                    raise ValueError(f"{query.label}: the label must use {{{{{var}}}}}, it tells the instances apart")
        elif self.instance_variable:
            raise ValueError("instance_variable needs max_instances above 1")

    @model_validator(mode="after")
    def _consistent(self) -> "QueryPack":
        self._check_targets()
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
                wrong = [domain for domain in self.target_domains if not role_fits(role, domain)]
                if wrong and role.domain is not None:
                    raise ValueError(f"{query.label}: role {query.role} belongs to domain {role.domain}")
                if wrong:
                    raise ValueError(f"{query.label}: role {query.role} fits {', '.join(HOST_DOMAINS)}, not {wrong[0]}")
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
        if self.run_filter is not None:
            if self.category != "load_tool":
                raise ValueError("run_filter is for load tool packs only")
            if self.run_filter.variable not in self.variables:
                raise ValueError(f"run_filter.variable «{self.run_filter.variable}» is not a variable of the pack")
            if self.variables[self.run_filter.variable].choices:
                raise ValueError("run_filter.variable cannot be a variable with choices")
            if template_names(self.run_filter.value) != {RUN_TAG_VARIABLE}:
                raise ValueError("run_filter.value must use {{run_tag}} and nothing else")
        return self


class Bundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    title_en: str = ""
    description: str
    description_en: str = ""
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
    "RUN_TAG_VARIABLE",
    "RunFilter",
    "TEMPLATE_RE",
    "category_title",
    "localized",
    "template_names",
]
