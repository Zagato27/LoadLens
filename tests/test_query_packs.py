"""Query packs: catalog contract, query syntax, merging into a domain block, SLA references."""

import copy
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import loadlens_app  # noqa: F401  (imported before update_page users to avoid a circular import)
from AI.query_packs import (
    ROLES,
    BlockConflictError,
    PackRenderError,
    QueryPack,
    block_roles,
    labels_for_role,
    load_catalog,
    merge_pack,
    missing_sla_references,
    prompt_addenda,
    prune_block,
    query_meta,
    render_pack,
    series_annotations,
    sla_update,
)
from AI.query_packs.__main__ import EXAMPLE_BUNDLE, bundle_config, docs_markdown
from AI.query_packs.catalog import PackCatalogError, load_catalog_from
from loadlens_app.config_schema import QueryConfigError, validate_queries
from promql_lint import PromQLSyntaxError, check_promql

CATALOG = load_catalog()
PACKS = list(CATALOG.packs.values())
PACK_IDS = [pack.id for pack in PACKS]
# Values unlike the defaults: rendering with them must still give valid queries.
OTHER_VALUES = {
    "service_label": "app",
    "matchers": 'namespace="shop", env=~"perf|load"',
    "ksm_matchers": 'job="kube-state-metrics"',
    "window": "5m",
    "cpu_limit_cores": "2.5",
    "namespace": "shop|payments",
    "container": "orders",
    "datname": "orders",
    "broker_label": "pod",
    "metric_prefix": "myapp_http_request_duration_seconds",
    "route_label": "route",
    "status_label": "status",
    "measurement": "jmeter_perf",
    "application": "checkout.*",
    "send_interval_sec": "10",
    "simulation": "checkoutsimulation",
    "write_period_sec": "2",
    "prefix": "tank_",
    "uuid": "8f1c.*",
    "root_prefix": "perf_gatling",
    "window_sec": "60",
    "testid": "release-2.*",
}


def _other_variables(pack: QueryPack) -> dict:
    out = {}
    for name, var in pack.variables.items():
        if var.choices:
            out[name] = var.choices[-1]
        else:
            out[name] = OTHER_VALUES[name]
    return out


def _balanced(text: str, pairs: str = "()[]{}") -> bool:
    stack = []
    opening = {pairs[i + 1]: pairs[i] for i in range(0, len(pairs), 2)}
    quote = ""
    for char in text:
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char in pairs[::2]:
            stack.append(char)
        elif char in opening:
            if not stack or stack.pop() != opening[char]:
                return False
    return not stack and not quote


# ---- catalog --------------------------------------------------------------------------------

def test_catalog_covers_the_requested_stacks():
    ids = set(PACK_IDS)
    for required in (
        "node-exporter", "k8s-nodes", "k8s-workload-health", "k8s-workload-cpu",
        "jvm-micrometer", "jvm-jmx-exporter", "jvm-otel", "runtime-go", "runtime-python",
        "kafka-exporter", "kafka-broker-jmx", "db-postgres-exporter", "db-hikaricp",
        "lt-k6-influxql", "lt-k6-prometheus-rw", "lt-k6-influx2-flux", "lt-jmeter-influxql", "lt-gatling-influxql",
        "http-micrometer", "http-otel", "http-go", "http-python-fastapi", "http-python-flask", "http-python-django",
    ):
        assert required in ids, required
    assert EXAMPLE_BUNDLE in CATALOG.bundles
    for bundle in CATALOG.bundles.values():
        domains = [CATALOG.packs[pack_id].domain for pack_id in bundle.packs]
        assert "lt_framework" in domains, bundle.id


@pytest.mark.parametrize("pack_id", PACK_IDS)
def test_pack_renders_with_defaults_and_other_values(pack_id):
    pack = CATALOG.packs[pack_id]
    for variables in ({}, _other_variables(pack)):
        rendered = render_pack(pack, variables)
        assert rendered.queries, pack_id
        for query in rendered.queries:
            for text in (query.label, query.query, *query.keys):
                assert "{{" not in text and "}}" not in text, (pack_id, text)
            assert all(key.strip() for key in query.keys), (pack_id, query.label)


@pytest.mark.parametrize("pack_id", [pack.id for pack in PACKS if pack.language == "promql"])
def test_promql_pack_queries_parse(pack_id):
    pack = CATALOG.packs[pack_id]
    for variables in ({}, _other_variables(pack)):
        for query in render_pack(pack, variables).queries:
            try:
                check_promql(query.query)
            except PromQLSyntaxError as exc:  # pragma: no cover - the message names the query
                pytest.fail(f"{pack_id} / {query.label}: {exc}\n{query.query}")


@pytest.mark.parametrize("pack_id", [pack.id for pack in PACKS if pack.language == "influxql"])
def test_influxql_pack_queries_are_well_formed(pack_id):
    pack = CATALOG.packs[pack_id]
    for query in render_pack(pack, _other_variables(pack)).queries:
        text = query.query
        assert text.startswith("SELECT "), query.label
        assert " FROM " in text and "$timeFilter" in text, query.label
        assert re.search(r"GROUP BY .*time\((\d+[smh]|\$__interval)\)", text), query.label
        assert _balanced(text, "()"), query.label
        assert text.count("'") % 2 == 0 and text.count('"') % 2 == 0, query.label


@pytest.mark.parametrize("pack_id", [pack.id for pack in PACKS if pack.language == "flux"])
def test_flux_pack_queries_are_well_formed(pack_id):
    pack = CATALOG.packs[pack_id]
    for query in render_pack(pack).queries:
        assert 'from(bucket: "{bucket}")' in query.query, query.label
        assert "range(start: {start}, stop: {end})" in query.query, query.label
        assert _balanced(query.query.replace("{bucket}", "b").replace("{start}", "s").replace("{end}", "e")), query.label


def test_series_and_unit_conventions():
    """The forecast and the resource plan read these series without knowing which pack wrote them."""
    for pack in PACKS:
        for query in render_pack(pack).queries:
            role = ROLES.get(query.role or "")
            assert role is not None, (pack.id, query.label)
            where = (pack.id, query.label)
            if role.single_series:
                if pack.language == "promql":
                    # One named series: an aggregate without labels gets a name through label_replace.
                    assert query.keys == ("series",) and query.query.startswith("label_replace("), where
                else:
                    assert len(query.keys) <= 1, where
            if query.role == "runtime.cpu":
                assert query.unit == "fraction" and query.keys[-1] == "instance", where
            if query.role and query.role.startswith("node.") and query.role not in ("node.ready",):
                assert query.keys == ("node",), where
            if pack.domain == "lt_framework" and query.unit in ("ms", "seconds"):
                assert query.unit == "ms" and "(ms)" in query.label, where
            if pack.domain in ("database", "kafka"):
                # instance= is stripped when series are grouped by service; servers and brokers get own keys.
                assert "instance" not in query.keys, where


def test_pack_validation_rejects_broken_files(tmp_path):
    good = (CATALOG.packs["kafka-exporter"].model_dump())
    broken = copy.deepcopy(good)
    broken["queries"][0]["role"] = "runtime.cpu"
    with pytest.raises(ValueError, match="belongs to domain"):
        QueryPack.model_validate(broken)
    broken = copy.deepcopy(good)
    broken["queries"][0]["unit"] = "percent"
    with pytest.raises(ValueError, match="query says percent"):
        QueryPack.model_validate(broken)
    broken = copy.deepcopy(good)
    broken["queries"][0]["query"] = "sum(x{ {{missing}} })"
    with pytest.raises(ValueError, match="undeclared variables: missing"):
        QueryPack.model_validate(broken)
    broken = copy.deepcopy(good)
    broken["sla"] = {"cpu_query": "@kafka.consumer_lag"}
    with pytest.raises(ValueError, match="fills"):
        QueryPack.model_validate(broken)

    packs_dir = tmp_path / "packs"
    packs_dir.mkdir()
    source = ROOT_DIR / "AI" / "query_packs" / "packs" / "kafka-exporter.toml"
    (packs_dir / "wrong-name.toml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(PackCatalogError, match="must match the file name"):
        load_catalog_from(packs_dir, tmp_path / "bundles.toml")


# ---- rendering ------------------------------------------------------------------------------

def test_render_checks_variable_values():
    pack = CATALOG.packs["jvm-micrometer"]
    with pytest.raises(PackRenderError, match="Неизвестные"):
        render_pack(pack, {"nope": "1"})
    with pytest.raises(PackRenderError, match="не подходит"):
        render_pack(pack, {"service_label": "app}"})
    with pytest.raises(PackRenderError, match="не подходит"):
        render_pack(pack, {"matchers": 'a="1"\nb="2"'})
    with pytest.raises(PackRenderError, match="допустимо"):
        render_pack(CATALOG.packs["lt-k6-influxql"], {"load_model": "sometimes"})
    rendered = render_pack(pack, {"service_label": "app", "matchers": 'namespace="shop"'})
    cpu = next(q for q in rendered.queries if q.role == "runtime.cpu")
    assert cpu.label == "JVM: Process CPU usage by (app, instance)"
    assert cpu.keys == ("app", "instance")
    assert 'app!="", namespace="shop"' in cpu.query
    assert rendered.sla_refs == {"service_cpu_query": cpu.label}


def test_render_include_keeps_only_selected_queries():
    pack = CATALOG.packs["lt-k6-influxql"]
    rendered = render_pack(pack, include={"LT (InfluxQL): RPS sum by all groups", "LT (InfluxQL): VUs"})
    assert [q.label for q in rendered.queries] == ["LT (InfluxQL): RPS sum by all groups", "LT (InfluxQL): VUs"]
    assert set(rendered.sla_refs) == {"max_performance_query", "vus_query"}


# ---- merging into a domain block --------------------------------------------------------------

def test_merge_into_empty_block_then_again_is_unchanged():
    rendered = render_pack(CATALOG.packs["jvm-micrometer"])
    block, plan = merge_pack(None, rendered)
    assert plan.added == [q.label for q in rendered.queries] and not plan.updated
    assert len(block["labels"]) == len(block["promql_queries"]) == len(block["label_keys_list"])
    assert block["packs"] == [{"id": "jvm-micrometer", "version": 1, "variables": rendered.variables}]
    assert labels_for_role(block, "runtime.cpu") == ["JVM: Process CPU usage by (application, instance)"]
    validate_queries({"jvm": block})
    again, plan = merge_pack(block, rendered)
    assert again == block and not plan.changed and len(plan.unchanged) == len(rendered.queries)


def test_merge_keeps_foreign_rows_and_updates_own_rows_in_place():
    own = render_pack(CATALOG.packs["jvm-micrometer"])
    cpu_label = "JVM: Process CPU usage by (application, instance)"
    block = {
        "labels": ["Custom: threads", cpu_label],
        "promql_queries": ["sum(my_threads)", "old cpu query"],
        "label_keys_list": [["app"], ["application", "instance"]],
    }
    merged, plan = merge_pack(block, own)
    assert merged["labels"][:2] == ["Custom: threads", cpu_label]
    assert merged["promql_queries"][0] == "sum(my_threads)"
    assert merged["promql_queries"][1].startswith("sum by (application, instance) (process_cpu_usage")
    assert plan.updated == [cpu_label]
    assert "Custom: threads" not in block_roles(merged)


def test_merge_pads_other_language_arrays_and_refuses_mixed_languages():
    block = {"labels": ["Old"], "promql_queries": [""], "label_keys_list": [[]], "flux_queries": ["from(bucket: \"b\")"], "label_tag_keys_list": [[]]}
    with pytest.raises(BlockConflictError, match="Flux"):
        merge_pack(block, render_pack(CATALOG.packs["jvm-micrometer"]))
    replaced, plan = merge_pack(block, render_pack(CATALOG.packs["jvm-micrometer"]), mode="replace")
    assert plan.removed == ["Old"] and "flux_queries" not in replaced
    influx_only = {"labels": ["Old"], "influxql_queries": ["SELECT 1"], "label_tag_keys_list": [["a"]], "promql_queries": [""], "label_keys_list": [[]]}
    with pytest.raises(BlockConflictError):
        merge_pack(influx_only, render_pack(CATALOG.packs["lt-k6-prometheus-rw"]))
    lt_block, _ = merge_pack({"labels": ["Old"], "influxql_queries": ["SELECT 1"], "label_tag_keys_list": [["a"]]}, render_pack(CATALOG.packs["lt-k6-influxql"]))
    assert len(lt_block["labels"]) == len(lt_block["influxql_queries"]) == len(lt_block["label_tag_keys_list"])
    validate_queries({"lt_framework": lt_block})


def test_prune_drops_roles_of_deleted_rows_and_empty_packs():
    block, _ = merge_pack(None, render_pack(CATALOG.packs["kafka-exporter"]))
    keep = block["labels"][0]
    trimmed = {
        "labels": [keep],
        "promql_queries": block["promql_queries"][:1],
        "label_keys_list": block["label_keys_list"][:1],
        "roles": dict(block["roles"]),
        "packs": list(block["packs"]),
    }
    prune_block(trimmed)
    assert trimmed["roles"] == {keep: "kafka.consumer_lag"}
    assert trimmed["packs"][0]["id"] == "kafka-exporter"
    empty = {"labels": [], "promql_queries": [], "label_keys_list": [], "roles": {"x": "kafka.brokers"}, "packs": list(block["packs"])}
    prune_block(empty)
    assert "roles" not in empty and "packs" not in empty


def test_validate_queries_checks_pack_keys():
    with pytest.raises(QueryConfigError, match="roles"):
        validate_queries({"jvm": {"labels": [], "roles": ["runtime.cpu"]}})
    with pytest.raises(QueryConfigError, match="id"):
        validate_queries({"jvm": {"labels": [], "packs": [{"version": 1}]}})
    validate_queries({"jvm": {"labels": [], "roles": {"gone": "runtime.cpu"}, "packs": [{"id": "x", "variables": {}}]}})


# ---- SLA references -------------------------------------------------------------------------------

def test_sla_update_fills_empty_fields_and_keeps_filled_ones():
    rendered = render_pack(CATALOG.packs["lt-k6-influxql"])
    current = {"max_performance_query": "My RPS", "p95_query": "", "latency_unit": "s", "load_model": "closed", "target_rps": 100}
    update = sla_update(current, rendered)
    assert update.sla["max_performance_query"] == "My RPS"
    assert update.kept == {"max_performance_query": "My RPS"}
    assert update.sla["p95_query"] == "LT (InfluxQL): http_req_duration p95(ms) total"
    assert update.sla["mean_latency_query"] == "LT (InfluxQL): http_req_duration mean(ms)"
    assert update.sla["latency_unit"] == "ms" and update.sla["load_model"] == "open"
    assert update.sla["target_rps"] == 100 and current["latency_unit"] == "s"
    overwritten = sla_update(current, rendered, overwrite=True)
    assert overwritten.sla["max_performance_query"] == "LT (InfluxQL): RPS sum by all groups"


def test_sla_update_does_not_mix_latency_units():
    rendered = render_pack(CATALOG.packs["lt-k6-influxql"])
    current = {"p95_query": "", "mean_latency_query": "My mean (seconds)", "latency_unit": "s"}
    update = sla_update(current, rendered)
    assert update.sla["p95_query"].endswith("p95(ms) total")
    assert update.sla["latency_unit"] == "s"
    assert any("latency_unit" in warning for warning in update.warnings)
    assert "latency_unit" not in update.changes


def test_missing_sla_references():
    queries = bundle_config(EXAMPLE_BUNDLE)["queries"]
    sla = {"max_performance_query": "LT (InfluxQL): RPS sum by all groups", "mean_latency_query": "gone", "cpu_query": ""}
    assert missing_sla_references(sla, queries, ("max_performance_query", "mean_latency_query", "cpu_query")) == ["mean_latency_query", "cpu_query"]


# ---- metadata for the analysis ----------------------------------------------------------------------

def test_query_meta_annotations_and_prompt_notes():
    block, _ = merge_pack(None, render_pack(CATALOG.packs["runtime-go"], {"service_label": "service"}))
    meta = query_meta(block)
    label = "Go runtime: CPU (share of limit) by (service, instance)"
    assert meta[label].role == "runtime.cpu"
    notes = series_annotations(block)
    assert notes[label]["unit"] == "доля 0..1" and notes[label]["meaning"] == "CPU экземпляра сервиса"
    addenda = prompt_addenda(block)
    assert addenda and addenda[0][0] == CATALOG.packs["runtime-go"].title and "горутин" in addenda[0][1]
    assert series_annotations({"labels": ["x"], "promql_queries": ["up"], "label_keys_list": [[]]}) == {}


# ---- generated files -------------------------------------------------------------------------------

def _example_config() -> dict:
    spec = importlib.util.spec_from_file_location("settings_example_for_packs", ROOT_DIR / "settings.example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def test_example_settings_match_the_default_bundle():
    """`python -m AI.query_packs example` regenerates them after a pack changes."""
    example = _example_config()
    config = bundle_config(EXAMPLE_BUNDLE)
    assert example["queries"] == config["queries"]
    for field, label in config["sla_refs"].items():
        assert example["sla"][field] == label, field
    for field, value in config["sla_values"].items():
        assert example["sla"][field] == value, field
    validate_queries(example["queries"])


def test_docs_catalog_is_up_to_date():
    """`python -m AI.query_packs docs docs/query-packs.md` regenerates it."""
    assert (ROOT_DIR / "docs/query-packs.md").read_text(encoding="utf-8") == docs_markdown()


# ---- analysis pipeline and project readiness --------------------------------------------------------

def test_aux_resources_follow_roles_and_fall_back_to_jvm_names():
    from AI.pipeline import _aux_resource_sections

    block, _ = merge_pack(None, render_pack(CATALOG.packs["runtime-go"]))
    sections = [{"label": label} for label in block["labels"]] + [{"label": "JVM: Process CPU usage by (application, instance)"}]
    cpu, mem = _aux_resource_sections({"sections": sections}, block)
    assert [s["label"] for s in cpu] == ["Go runtime: CPU (share of limit) by (job, instance)"]
    assert {s["label"] for s in mem} == {
        "Go runtime: heap alloc (bytes) by (job, instance)",
        "Go runtime: resident memory (bytes) by (job, instance)",
    }
    legacy = [{"label": "JVM: Process CPU usage by (application, instance)"}, {"label": "JVM: Heap used (bytes) by (application, instance)"}]
    cpu, mem = _aux_resource_sections({"sections": legacy}, {"labels": []})
    assert len(cpu) == 1 and len(mem) == 1


def test_context_sections_get_units_and_prompt_gets_stack_notes():
    from AI.pipeline import _annotate_sections
    from AI.prompt_assembly import build_analysis_guidance

    block, _ = merge_pack(None, render_pack(CATALOG.packs["db-hikaricp"]))
    label = "DB pool (HikariCP): active connections by (application, pool)"
    pack = {"sections": [{"label": label, "top_series": []}, {"label": "Other"}]}
    _annotate_sections(pack, block)
    assert pack["sections"][0]["unit"] == "шт." and pack["sections"][0]["meaning"] == "Активные соединения пула"
    assert "unit" not in pack["sections"][1]
    guidance = build_analysis_guidance({}, {}, "step", domain="database", stack_notes=prompt_addenda(block))
    assert "Особенности метрик стека" in guidance and "HikariCP" in guidance
    assert "Особенности метрик стека" not in build_analysis_guidance({}, {}, "step", domain="database")


def test_project_readiness_lists_missing_sla_references_and_packs(monkeypatch):
    from loadlens_app import projects

    monkeypatch.setitem(projects.CONFIG, "queries", {})
    monkeypatch.setitem(projects.CONFIG, "sla", {})
    config = bundle_config(EXAMPLE_BUNDLE)
    # A dead reference is resolved from the role query at report time, so nothing is missing.
    area_entry = {"queries": config["queries"], "sla": {**config["sla_refs"], "mean_latency_query": "gone"}}
    readiness = projects._readiness(area_entry)
    assert readiness.missing_sla == {"report": [], "forecast": [], "resource_plan": []}
    # Without a mean-latency role query the dead reference cannot be replaced and is reported.
    queries = copy.deepcopy(config["queries"])
    lt_roles = queries["lt_framework"]["roles"]
    for label in [label for label, role in lt_roles.items() if role == "lt.mean_latency_total"]:
        del lt_roles[label]
    readiness = projects._readiness({"queries": queries, "sla": {**config["sla_refs"], "mean_latency_query": "gone"}})
    assert readiness.missing_sla == {"report": [], "forecast": ["mean_latency_query"], "resource_plan": []}
    assert {"lt-k6-influxql", "jvm-micrometer", "node-exporter"} <= {p["id"] for p in readiness.query_packs}
    assert all(p["outdated"] is False for p in readiness.query_packs)


def test_sla_update_replaces_references_to_missing_queries():
    rendered = render_pack(CATALOG.packs["lt-k6-influxql"])
    after = {"lt_framework": merge_pack(None, rendered)[0]}
    current = {"max_performance_query": "Old RPS", "p95_query": "LT (InfluxQL): http_req_duration p95(ms) total"}
    update = sla_update(current, rendered, queries=after)
    assert update.sla["max_performance_query"] == "LT (InfluxQL): RPS sum by all groups"
    assert update.changes["max_performance_query"] == {"from": "Old RPS", "to": "LT (InfluxQL): RPS sum by all groups", "reason": "missing"}
    assert "max_performance_query" not in update.kept
    # A live reference to a hand-made query is still kept without ``overwrite``.
    with_own = {"lt_framework": {**after["lt_framework"], "labels": after["lt_framework"]["labels"] + ["My RPS"]}}
    kept = sla_update({"max_performance_query": "My RPS"}, rendered, queries=with_own)
    assert kept.kept == {"max_performance_query": "My RPS"}
