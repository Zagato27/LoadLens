# SPDX-License-Identifier: Apache-2.0
"""Command line for the pack catalog.

    python -m AI.query_packs list
    python -m AI.query_packs render jvm-micrometer --var service_label=app
    python -m AI.query_packs bundle spring-k8s-k6-influxdb       # queries + SLA references as JSON
    python -m AI.query_packs docs docs/query-packs.md             # markdown catalog (Russian)
    python -m AI.query_packs docs --lang en docs/query-packs.en.md
    python -m AI.query_packs example                               # rewrites queries/SLA of settings.example.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from AI.i18n import normalize_lang, with_locale

from .blocks import merge_pack
from .catalog import Catalog, load_catalog
from .models import CATEGORY_TITLES, LANGUAGE_TITLES, localized
from .render import render_pack
from .roles import ROLES, UNIT_TITLES

EXAMPLE_BUNDLE = "spring-k8s-k6-influxdb"
EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "settings.example.py"
DOCS_FILES = {"ru": "docs/query-packs.md", "en": "docs/query-packs.en.md"}
# Texts of the generated catalog; category, role and unit titles come from the locale dictionaries.
DOCS_TEXT: dict[str, dict[str, Any]] = {
    "ru": {
        "title": "Пакеты запросов",
        "other_language": "[English version](query-packs.en.md)",
        "intro": [
            "Пакет — готовые запросы одного домена для типового источника метрик. Применение добавляет запросы",
            "в раздел «Запросы» выбранной области и заполняет ссылки SLA. Задержки нагрузочных инструментов",
            "приводятся к миллисекундам, CPU узлов — к процентам, CPU экземпляра сервиса — к доле 0..1.",
            "",
            "Отметка «проверен на стенде» значит, что каждый обязательный запрос пакета вернул данные на",
            "`docker/dev-stack` с живыми экспортерами. Остальные пакеты проверены синтаксически и на контракте",
            "рядов и единиц; перед использованием прогоните их кнопкой «Проверить».",
        ],
        "bundles": "Наборы",
        "bundle_columns": "| Набор | Пакеты |",
        "domain": "домен «{domain}»",
        "domains_pick": "домен на выбор: {domains}",
        "quote": "«{name}»",
        "instances": " · до {n} установок в одном домене",
        "verified": " · проверен на стенде `docker/dev-stack`",
        "variable_columns": "| Переменная | По умолчанию | Описание |",
        "empty": "пусто",
        "choices": "Варианты: {choices}",
        "query_columns": "| Запрос | Роль | Единица | SLA |",
        "required": " (обязательный)",
        "domains": {
            "lt_framework": "Нагрузочный инструмент",
            "jvm": "JVM и рантайм",
            "microservices": "Микросервисы",
            "database": "Базы данных",
            "kafka": "Kafka",
            "hard_resources": "Ресурсы узлов",
            "lt_generators": "Ресурсы генераторов нагрузки",
        },
        "sla_fields": {
            "max_performance_query": "запрос производительности",
            "error_rate_query": "доля ошибок",
            "p95_query": "p95",
            "p99_query": "p99",
            "cpu_query": "CPU узлов",
            "memory_query": "память узлов",
            "mean_latency_query": "среднее время (прогноз)",
            "vus_query": "VU (прогноз)",
            "service_cpu_query": "CPU сервисов (план ресурсов)",
        },
    },
    "en": {
        "title": "Query packs",
        "other_language": "[Русская версия](query-packs.md)",
        "intro": [
            "A pack is a set of ready queries of one domain for a typical metric source. Applying it adds the queries",
            "to the \"Queries\" section of the chosen area and fills the SLA references. Latencies of load tools are",
            "converted to milliseconds, node CPU to percent, the CPU of a service instance to a fraction 0..1.",
            "",
            "\"Verified on `docker/dev-stack`\" means that every required query of the pack returned data on",
            "`docker/dev-stack` with live exporters. The other packs are checked for syntax and for the contract of",
            "series and units; run them with the \"Check\" button before use.",
        ],
        "bundles": "Bundles",
        "bundle_columns": "| Bundle | Packs |",
        "domain": "domain \"{domain}\"",
        "domains_pick": "domain of your choice: {domains}",
        "quote": "\"{name}\"",
        "instances": " · up to {n} installations per domain",
        "verified": " · verified on `docker/dev-stack`",
        "variable_columns": "| Variable | Default | Description |",
        "empty": "empty",
        "choices": "Choices: {choices}",
        "query_columns": "| Query | Role | Unit | SLA |",
        "required": " (required)",
        "domains": {
            "lt_framework": "Load testing tool",
            "jvm": "JVM and runtime",
            "microservices": "Microservices",
            "database": "Databases",
            "kafka": "Kafka",
            "hard_resources": "Node resources",
            "lt_generators": "Load generator resources",
        },
        "sla_fields": {
            "max_performance_query": "performance query",
            "error_rate_query": "error rate",
            "p95_query": "p95",
            "p99_query": "p99",
            "cpu_query": "node CPU",
            "memory_query": "node memory",
            "mean_latency_query": "mean latency (forecast)",
            "vus_query": "VUs (forecast)",
            "service_cpu_query": "service CPU (resource plan)",
        },
    },
}


def bundle_config(bundle_id: str, catalog: Catalog | None = None) -> dict[str, Any]:
    """``queries`` and the SLA references a fresh install gets by applying every pack of the bundle."""
    catalog = catalog or load_catalog()
    bundle = catalog.bundles[bundle_id]
    queries: dict[str, Any] = {}
    sla_refs: dict[str, str] = {}
    sla_values: dict[str, str] = {}
    for pack_id in bundle.packs:
        rendered = render_pack(catalog.packs[pack_id])
        block, _ = merge_pack(queries.get(rendered.pack.domain), rendered, "merge")
        queries[rendered.pack.domain] = block
        for field_name, label in rendered.sla_refs.items():
            sla_refs.setdefault(field_name, label)
        for field_name, value in rendered.sla_values.items():
            sla_values.setdefault(field_name, value)
    return {"queries": queries, "sla_refs": sla_refs, "sla_values": sla_values}


def _literal(value: Any, indent: int) -> str:
    """Python literal in the style of settings.example.py: double quotes unless the text has them."""
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{pad}    {_literal(key, 0)}: {_literal(item, indent + 4)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value) and sum(len(item) for item in value) < 60:
            return "[" + ", ".join(_literal(item, 0) for item in value) + "]"
        items = [f"{pad}    {_literal(item, indent + 4)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    if isinstance(value, str):
        return repr(value) if '"' in value else json.dumps(value, ensure_ascii=False)
    return repr(value)


_QUERIES_BLOCK_RE = re.compile(r'(?ms)^    "queries": \{\n.*?^    \}\n(?=\}\n?\Z)')
_SLA_BLOCK_RE = re.compile(r'(?ms)^    "sla": \{\n.*?^    \},?\n')


def write_example(path: Path = EXAMPLE_PATH) -> None:
    """Replaces the ``queries`` block and the SLA references of the example config with the bundle."""
    config = bundle_config(EXAMPLE_BUNDLE)
    text = path.read_text(encoding="utf-8")
    block = '    "queries": ' + _literal(config["queries"], 4) + "\n"
    text, count = _QUERIES_BLOCK_RE.subn(lambda _m: block, text)
    if count != 1:
        raise SystemExit("settings.example.py: the queries block was not found at the end of CONFIG")
    sla_match = _SLA_BLOCK_RE.search(text)
    if sla_match is None:
        raise SystemExit("settings.example.py: the sla block was not found")
    sla_text = sla_match.group(0)
    for field_name, value in {**config["sla_refs"], **config["sla_values"]}.items():
        pattern = re.compile(rf'^( +"{field_name}": )"[^"\n]*"', re.M)
        sla_text, count = pattern.subn(lambda m: m.group(1) + json.dumps(value, ensure_ascii=False), sla_text)
        if count != 1:
            raise SystemExit(f"settings.example.py: sla.{field_name} was not found")
    text = text[:sla_match.start()] + sla_text + text[sla_match.end():]
    path.write_text(text, encoding="utf-8", newline="\n")


def docs_markdown(catalog: Catalog | None = None, lang: str = "ru") -> str:
    """Markdown catalog in ``lang``; role, unit and category titles otherwise follow the current locale."""
    code = normalize_lang(lang)
    if code not in DOCS_TEXT:
        raise ValueError(f"no catalog texts for language {lang!r}")
    with with_locale(code):
        return _docs_markdown(catalog or load_catalog(), code)


def _docs_markdown(catalog: Catalog, lang: str) -> str:
    text = DOCS_TEXT[lang]
    flag = "" if lang == "ru" else f" --lang {lang}"
    out = [
        f"# {text['title']}",
        "",
        f"<!-- Generated by `python -m AI.query_packs docs{flag} {DOCS_FILES[lang]}`; edit the TOML files in AI/query_packs/packs. -->",
        "",
        text["other_language"],
        "",
        *text["intro"],
        "",
        f"## {text['bundles']}",
        "",
        text["bundle_columns"],
        "|---|---|",
    ]
    for bundle in catalog.bundles.values():
        out.append(f"| {localized(bundle, 'title', lang)} | {', '.join(f'`{pack_id}`' for pack_id in bundle.packs)} |")
    for category, title in CATEGORY_TITLES.items():
        packs = [pack for pack in catalog.packs.values() if pack.category == category]
        if not packs:
            continue
        out += ["", f"## {title}"]
        for pack in packs:
            rendered = render_pack(pack)
            names = [text["domains"].get(name, name) for name in pack.target_domains]
            if pack.domains:
                domain = text["domains_pick"].format(domains=", ".join(text["quote"].format(name=name) for name in names))
            else:
                domain = text["domain"].format(domain=names[0])
            if pack.multi_instance:
                domain += text["instances"].format(n=pack.max_instances)
            out += [
                "",
                f"### {localized(pack, 'title', lang)}",
                "",
                f"`{pack.id}` v{pack.version} · {domain} · {LANGUAGE_TITLES[pack.language]}"
                + (text["verified"] if pack.verified else ""),
                "",
                " ".join(localized(pack, "description", lang).split()),
            ]
            if pack.variables:
                out += ["", text["variable_columns"], "|---|---|---|"]
                for name, var in pack.variables.items():
                    default = f"`{var.default}`" if var.default else text["empty"]
                    parts = [
                        value.strip().rstrip(".")
                        for value in (localized(var, "title", lang), localized(var, "help", lang))
                        if value.strip()
                    ]
                    if var.choices:
                        parts.append(text["choices"].format(choices=", ".join(var.choices)))
                    hint = ". ".join(parts)
                    out.append(f"| `{name}` | {default} | {_cell(hint)} |")
            out += ["", text["query_columns"], "|---|---|---|---|"]
            refs = {label: field for field, label in rendered.sla_refs.items()}
            for query in rendered.queries:
                role = ROLES.get(query.role or "")
                role_text = f"{role.title}{text['required'] if query.required else ''}" if role else ""
                sla = text["sla_fields"].get(refs.get(query.label, ""), "")
                out.append(f"| {_cell(query.label)} | {_cell(role_text)} | {UNIT_TITLES.get(query.unit, query.unit)} | {sla} |")
    return "\n".join(out) + "\n"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m AI.query_packs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="packs and bundles")
    render_cmd = sub.add_parser("render", help="render one pack as JSON")
    render_cmd.add_argument("pack_id")
    render_cmd.add_argument("--var", action="append", default=[], help="name=value")
    bundle_cmd = sub.add_parser("bundle", help="queries and SLA references of a bundle as JSON")
    bundle_cmd.add_argument("bundle_id")
    docs_cmd = sub.add_parser("docs", help="markdown catalog")
    docs_cmd.add_argument("output", nargs="?", help="file to write (default: stdout)")
    docs_cmd.add_argument("--lang", choices=sorted(DOCS_TEXT), default="ru", help="language of the catalog (default: ru)")
    sub.add_parser("example", help="rewrite queries and SLA references of settings.example.py")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Titles contain arrows and Cyrillic; a Windows console defaults to cp1251.
        sys.stdout.reconfigure(encoding="utf-8")
    catalog = load_catalog()
    if args.command == "list":
        for pack in catalog.packs.values():
            print(f"{pack.id:28} v{pack.version}  {','.join(pack.target_domains):15} {pack.language:9} {pack.title}")
        for bundle in catalog.bundles.values():
            print(f"bundle {bundle.id}: {', '.join(bundle.packs)}")
    elif args.command == "render":
        variables = dict(item.split("=", 1) for item in args.var)
        print(json.dumps(render_pack(catalog.packs[args.pack_id], variables).to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "bundle":
        print(json.dumps(bundle_config(args.bundle_id, catalog), ensure_ascii=False, indent=2))
    elif args.command == "docs":
        text = docs_markdown(catalog, args.lang)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stdout.write(text)
    elif args.command == "example":
        write_example()
        print(f"{EXAMPLE_PATH.name}: queries and SLA references of bundle {EXAMPLE_BUNDLE} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
