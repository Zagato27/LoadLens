# SPDX-License-Identifier: Apache-2.0
"""The «domain metrics» block of a domain prompt and the hand-written descriptions of query rows.

The block lists every configured query of the domain (pack rows and rows added by hand) with its meaning, unit,
series keys and description, and marks the rows that returned no data in the run. The domain prompt files no
longer list the sections by hand.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from AI.i18n import with_locale  # noqa: E402
from AI.prompt_store import PREVIOUS_DEFAULTS, effective_override, load_prompt  # noqa: E402
from AI import prompt_assembly  # noqa: E402  (its test_type_overlay is not a test)
from AI.prompt_assembly import build_analysis_guidance, metric_inventory_lines  # noqa: E402
from AI.query_packs import (  # noqa: E402
    DESCRIPTION_MAX_CHARS,
    ROLES,
    block_descriptions,
    load_catalog,
    merge_pack,
    prompt_addenda,
    prune_block,
    render_pack,
    series_annotations,
    series_inventory,
    strip_live_rows,
)
from loadlens_app import core  # noqa: E402
from loadlens_app.config_schema import QueryConfigError, validate_queries  # noqa: E402
from tests.js_harness import run_js  # noqa: E402
from tests.test_query_packs_api import env  # noqa: E402,F401  (fixture)

CATALOG = load_catalog()
PACK = CATALOG.packs["db-postgres-exporter"]
OUTBOX = "Outbox size"
RAW = "Custom raw"
EMPTY = "Not written yet"
TPS = "DB (PostgreSQL): transactions per second by database"
LAG = "DB (PostgreSQL): replication lag (s) by server"
CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def _block() -> dict:
    """The PostgreSQL pack after three rows added by hand: one with a role and a description, one bare, one empty."""
    custom = {
        "labels": [OUTBOX, RAW, EMPTY],
        "promql_queries": ["sum by (service) (outbox_rows)", "sum(foo)", ""],
        "label_keys_list": [["service"], [], []],
    }
    block, _ = merge_pack(custom, render_pack(PACK))
    block["roles"][OUTBOX] = "db.queue_length"
    block["descriptions"] = {OUTBOX: "  Rows waiting\n in the outbox table. ", "gone": "a row deleted earlier"}
    return block


# ---- the inventory of a block ---------------------------------------------------------------------------------


def test_inventory_lists_custom_and_pack_rows_in_block_order():
    block = _block()
    pack_labels = [query.label for query in render_pack(PACK).queries]
    with with_locale("ru"):
        items = series_inventory(block)
        queue_title = ROLES["db.queue_length"].title
    assert [item["label"] for item in items] == [OUTBOX, RAW, *pack_labels]  # the row without a query is left out
    assert items[0] == {
        "label": OUTBOX, "role": "db.queue_length", "meaning": queue_title, "unit": "count", "unit_title": "шт.",
        "keys": ["service"], "description": "Rows waiting\n in the outbox table.", "origin": "custom",
    }
    assert items[1] == {
        "label": RAW, "role": "", "meaning": "", "unit": "", "unit_title": "", "keys": [], "description": "",
        "origin": "custom",
    }
    tps = next(item for item in items if item["label"] == TPS)
    query = render_pack(PACK).by_label()[TPS]
    assert tps["origin"] == "pack" and tps["role"] == "db.tps" and tps["unit"] == "per_second"
    assert tps["keys"] == ["datname"] and tps["description"] == query.description


def test_inventory_reads_the_keys_of_the_row_language():
    block = {
        "labels": ["Influx row"],
        "promql_queries": [""],
        "label_keys_list": [[]],
        "influxql_queries": ['SELECT mean("value") FROM "cpu" GROUP BY "host"'],
        "label_tag_keys_list": [["host"]],
    }
    assert series_inventory(block)[0]["keys"] == ["host"]
    assert series_inventory(None) == [] and series_inventory({"labels": []}) == []


def test_a_hand_written_description_wins_over_the_pack_and_reaches_the_context():
    block = _block()
    block["descriptions"][TPS] = "TPS of the order database."
    with with_locale("ru"):
        notes = series_annotations(block)
        queue_title = ROLES["db.queue_length"].title
    assert notes[TPS]["description"] == "TPS of the order database." and notes[TPS]["meaning"] == "Транзакции в секунду"
    assert notes[OUTBOX] == {"meaning": queue_title, "unit": "шт.", "description": "Rows waiting\n in the outbox table."}
    assert RAW not in notes
    assert block_descriptions(block) == {OUTBOX: "Rows waiting\n in the outbox table.", TPS: "TPS of the order database."}


def test_descriptions_follow_their_rows():
    block = prune_block(_block())
    assert block["descriptions"] == {OUTBOX: "Rows waiting\n in the outbox table."}
    # Deleting the row deletes its description; the key goes when no description is left.
    index = block["labels"].index(OUTBOX)
    for key in ("labels", "promql_queries", "label_keys_list"):
        del block[key][index]
    assert "descriptions" not in prune_block(block)

    # A reinstalled pack keeps the description of its own row, in both merge modes.
    block = _block()
    block["descriptions"][TPS] = "TPS of the order database."
    merged, _ = merge_pack(block, render_pack(PACK))
    assert merged["descriptions"] == {OUTBOX: "Rows waiting\n in the outbox table.", TPS: "TPS of the order database."}
    replaced, _ = merge_pack(block, render_pack(PACK), mode="replace")
    assert replaced["descriptions"] == {TPS: "TPS of the order database."}


def test_live_rows_leave_with_their_descriptions():
    queries = {"lt_framework": {
        "labels": ["live", "kept"], "promql_queries": ["a", "b"], "label_keys_list": [[], []],
        "roles": {"live": "lt.rps_by_instance", "kept": "lt.rps_total"},
        "descriptions": {"live": "live chart only", "kept": "total RPS"},
    }}
    stripped = strip_live_rows(queries)["lt_framework"]
    assert stripped["labels"] == ["kept"] and stripped["descriptions"] == {"kept": "total RPS"}
    only_live = copy.deepcopy(queries)
    only_live["lt_framework"]["descriptions"] = {"live": "live chart only"}
    assert "descriptions" not in strip_live_rows(only_live)["lt_framework"]


# ---- the block of the prompt ----------------------------------------------------------------------------------


def _guidance(block: dict, lang: str, with_data=None) -> str:
    with with_locale(lang):
        items = series_inventory(block)
        notes = prompt_addenda(block)
    return build_analysis_guidance(
        {}, None, "step", domain="database", stack_notes=notes, lang=lang, metrics=items, labels_with_data=with_data,
    )


def test_guidance_lists_every_query_with_its_facts():
    block = _block()
    labels = [item["label"] for item in series_inventory(block)]
    guidance = _guidance(block, "ru", with_data=[label for label in labels if label != LAG])
    with with_locale("ru"):
        queue_title = ROLES["db.queue_length"].title
    lines = guidance.splitlines()
    outbox = f"- «{OUTBOX}» — {queue_title}; единица: шт.; серии по service; пояснение: Rows waiting in the outbox table."
    raw = f"- «{RAW}» — смысл не задан; единица не задана"
    assert outbox in lines and raw in lines
    lag = next(line for line in lines if line.startswith(f"- «{LAG}»"))
    tps = next(line for line in lines if line.startswith(f"- «{TPS}»"))
    assert "в этом прогоне данных нет" in lag and "данных нет" not in tps
    assert tps.startswith(f"- «{TPS}» — Транзакции в секунду; единица: в секунду; серии по datname; пояснение: ")
    assert lines.index(outbox) < lines.index(raw) < lines.index(tps)
    assert any("единицу не додумывай" in line for line in lines)
    # Order of the guidance: the metrics after the SLA, before the pack notes and the test profile.
    metrics_at = guidance.index("МЕТРИКИ ДОМЕНА")
    assert guidance.index("SLA сервиса:") < metrics_at < guidance.index("Особенности метрик стека")
    assert metrics_at < guidance.index(prompt_assembly.test_type_overlay("step", "ru")[:40])


def test_guidance_without_metrics_or_data_marks():
    assert "МЕТРИКИ ДОМЕНА" not in build_analysis_guidance({}, None, "step", domain="database", lang="ru")
    assert "МЕТРИКИ ДОМЕНА" not in build_analysis_guidance({}, None, "step", domain="database", lang="ru", metrics=[])
    # Unknown data (no labels passed): no row is marked as empty.
    assert "данных нет" not in _guidance(_block(), "ru")
    # Every row has a meaning and a unit: no rule about unknown units.
    pack_only, _ = merge_pack(None, render_pack(PACK))
    assert "додумывай" not in _guidance(pack_only, "ru")


def test_metric_lines_follow_the_report_language():
    block = _block()
    with with_locale("en"):
        items = series_inventory(block)
        queue_title = ROLES["db.queue_length"].title
    lines = metric_inventory_lines(items, [OUTBOX], "en")
    assert lines[0] == f'- "{OUTBOX}" — {queue_title}; unit: count; series by service; note: Rows waiting in the outbox table.'
    assert lines[1] == f'- "{RAW}" — meaning not set; unit not set; no data in this run — do not quote values'
    assert not CYRILLIC.search("\n".join(lines))
    assert "DOMAIN METRICS" in _guidance(block, "en")


@pytest.mark.parametrize("items", [None, [], [{"label": ""}], ["not a mapping"]])
def test_metric_lines_skip_unusable_items(items):
    assert metric_inventory_lines(items, None, "ru") == []


# ---- the prompt files ----------------------------------------------------------------------------------------

LISTED_DOMAINS = ("database", "jvm", "kafka", "microservices", "hard_resources", "lt_framework", "lt_generators")
OLD_DATABASE_LISTS = {
    "ru": (
        "ДАННЫЕ ДОМЕНА (какие секции могут присутствовать):\n"
        "- Объём запросов к БД (rate запросов) по сервисам/подам.\n"
        "- 95-й перцентиль времени выполнения запросов по сервисам/подам.\n"
        "- При наличии — соединения, ошибки, ожидания и другие серии, попавшие в контекст.\n"
    ),
    "en": (
        "DOMAIN DATA (sections that may be present):\n"
        "- Database query volume (query rate) by service/pod.\n"
        "- The 95th percentile of query execution time by service/pod.\n"
        "- If present — connections, errors, waits and other series included in the context.\n"
    ),
}


@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize("domain", LISTED_DOMAINS)
def test_domain_prompts_point_at_the_generated_block_instead_of_a_list(domain, lang):
    text = load_prompt(f"{domain}_prompt.txt", lang)
    assert "(какие секции могут присутствовать)" not in text and "(sections that may be present)" not in text
    assert ("«МЕТРИКИ ДОМЕНА»" if lang == "ru" else '"DOMAIN METRICS"') in text
    # Reading rules of the old lists stay: they are not a list of queries.
    if domain == "lt_framework":
        assert ("не считай любые checks ошибками" if lang == "ru" else "do not treat every check as an error") in text
    if domain == "microservices":
        assert "aux_resources" in text.split("\n\n")[1]


def test_copies_of_earlier_defaults_are_no_override():
    for lang, old_list in OLD_DATABASE_LISTS.items():
        current = load_prompt("database_prompt.txt", lang)
        head = current.split("\n\n")[1] + "\n"
        old = current.replace(head, old_list)
        assert old != current
        # The settings page may have stored the default it showed, with the line endings of a browser.
        assert effective_override(old.replace("\n", "\r\n"), "database") is None
        assert effective_override(old + "\nOwn rule.", "database") is not None
    assert set(PREVIOUS_DEFAULTS) == {f"{domain}_prompt.txt" for domain in LISTED_DOMAINS}
    assert all(len(hashes) == 2 for hashes in PREVIOUS_DEFAULTS.values())


# ---- descriptions in the settings: schema, saving, the role list ---------------------------------------------

ROW = {"labels": ["Outbox"], "promql_queries": ["sum(outbox_rows)"], "label_keys_list": [[]]}
SETTINGS_JS = ["static/js/common.js", "static/js/ui.js", "static/js/settings.js"]
INTERNALS = "LL.settings._internals"


def test_query_config_checks_the_shape_of_descriptions():
    validate_queries({"database": {**ROW, "descriptions": {"Outbox": "Rows in the outbox.", "Gone": "kept until saved"}}})
    validate_queries({"database": {**ROW, "descriptions": {"Outbox": "x" * 500}}})
    for bad in (["Outbox"], {"Outbox": 1}, {"Outbox": "x" * 501}):
        with pytest.raises(QueryConfigError, match="descriptions"):
            validate_queries({"database": {**ROW, "descriptions": bad}})


def test_saving_queries_keeps_the_descriptions_of_existing_rows(env):  # noqa: F811
    payload = {"database": {**ROW, "roles": {"Outbox": "db.queue_length"}, "descriptions": {"Outbox": "Rows.", "Gone": "x"}}}
    resp = env.client.post("/config", json={"section": "queries", "data": payload})
    assert resp.status_code == 200, resp.get_json()
    block = core.CONFIG["queries"]["database"]
    assert block["descriptions"] == {"Outbox": "Rows."} and block["roles"] == {"Outbox": "db.queue_length"}
    bad = {"database": {**ROW, "descriptions": {"Outbox": ["not text"]}}}
    assert env.client.post("/config", json={"section": "queries", "data": bad}).status_code == 400


def test_catalog_tells_the_page_where_host_roles_fit(env):  # noqa: F811
    data = env.client.get("/config/query_packs").get_json()
    assert data["host_domains"] == ["database", "kafka", "jvm", "microservices"]
    assert data["roles"]["lt.rps_by_instance"]["live_only"] is True
    assert data["roles"]["db.tps"]["live_only"] is False


def _catalog() -> dict:
    roles = {name: {"title": name.upper(), "unit": "u", "domain": role.domain, "sla_field": role.sla_field,
                    "live_only": name == "lt.rps_by_instance"} for name, role in ROLES.items()}
    return {"roles": roles, "host_domains": ["database", "kafka", "jvm", "microservices"]}


def test_role_list_follows_the_domain_of_the_row():
    catalog = _catalog()
    options = run_js(SETTINGS_JS, f"{INTERNALS}.roleOptions({json.dumps(catalog)}, 'database', '')")
    names = [name for name, _title in options]
    own = sorted(name for name, role in ROLES.items() if role.domain == "database")
    host = sorted(name for name, role in ROLES.items() if role.domain is None)
    # The roles of the domain by title, then the host roles by title (the test titles are the upper-cased names).
    assert options[0] == ["", "— не задана —"] and names[1:] == own + host
    assert options[names.index("db.tps")][1] == "DB.TPS (u)"
    nodes = [name for name, _ in run_js(SETTINGS_JS, f"{INTERNALS}.roleOptions({json.dumps(catalog)}, 'hard_resources', '')")]
    assert "node.cpu_pct" in nodes and "host.cpu_pct" not in nodes
    lt = [name for name, _ in run_js(SETTINGS_JS, f"{INTERNALS}.roleOptions({json.dumps(catalog)}, 'lt_framework', '')")]
    assert "lt.rps_total" in lt and "lt.rps_by_instance" not in lt
    kept = run_js(SETTINGS_JS, f"{INTERNALS}.roleOptions({json.dumps(catalog)}, 'lt_framework', 'lt.rps_by_instance')")
    assert "lt.rps_by_instance" in [name for name, _ in kept]
    # Without the catalog (not loaded yet) only "not set" is offered; the select adds the current role itself.
    assert run_js(SETTINGS_JS, f"{INTERNALS}.roleOptions(null, 'database', '')") == [["", "— не задана —"]]


def test_shared_sla_role_names_the_row_the_sla_takes():
    catalog = _catalog()
    block = {"labels": ["A", "B", "C"], "roles": {"A": "lt.rps_total", "B": "lt.rps_by_request", "C": "lt.rps_total"}}
    expr = f"{INTERNALS}.sharedRoleLabel({json.dumps(catalog)}, {json.dumps(block)}, '%s')"
    assert run_js(SETTINGS_JS, expr % "lt.rps_total") == "A"
    assert run_js(SETTINGS_JS, expr % "lt.rps_by_request") == ""  # one row only, and the role fills no SLA field
    assert run_js(SETTINGS_JS, expr % "") == ""


def test_the_page_and_the_server_share_the_description_limit():
    assert run_js(SETTINGS_JS, f"{INTERNALS}.QUERY_NOTE_MAX") == DESCRIPTION_MAX_CHARS
    source = (ROOT_DIR / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    # Renaming a row moves its description; deleting a row deletes it.
    assert "['roles', 'descriptions', 'sources'].forEach" in source
    assert "if (isPlainObject(block.descriptions)) delete block.descriptions[label];" in source
