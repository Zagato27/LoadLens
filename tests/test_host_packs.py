# SPDX-License-Identifier: Apache-2.0
"""Host packs: node_exporter of a database or broker host in that component's section, up to 10 installations."""

import copy
import json
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from AI.query_packs import (
    HOST_DOMAINS,
    ROLES,
    QueryPack,
    installed_instances,
    load_catalog,
    merge_pack,
    prompt_addenda,
    prune_block,
    render_pack,
    resolve_sla_refs,
    role_fits,
    role_for_sla_field,
    series_annotations,
)
from AI.query_packs.catalog import PackCatalogError, load_catalog_from
from loadlens_app import core
from loadlens_app.config_schema import validate_queries
from loadlens_app.metric_discovery import DomainNames

CATALOG = load_catalog()
HOST_PACK = "host-node-exporter"
SOURCES = {"prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}}}
BINDINGS = {"default": {"source": "prom"}}


def _base_pack(**changes) -> dict:
    data = {
        "schema_version": 1,
        "id": "test-host",
        "version": 1,
        "title": "Test host",
        "category": "infrastructure",
        "domains": ["database", "kafka"],
        "language": "promql",
        "description": "Test.",
        "max_instances": 3,
        "instance_variable": "host_title",
        "variables": {"host_title": {"title": "Host title", "default": "Host", "pattern": "^[A-Za-z ]{1,20}$"}},
        "queries": [{"label": "{{host_title}}: CPU", "query": "node_load1", "keys": ["node"], "role": "host.cpu_pct", "unit": "percent"}],
    }
    data.update(changes)
    return data


# ---- roles and the pack schema ------------------------------------------------------------------------

def test_host_roles_fit_component_domains_and_fill_no_sla():
    role = ROLES["host.cpu_pct"]
    assert role.domain is None and role.sla_field is None
    assert all(role_fits(role, domain) for domain in HOST_DOMAINS)
    assert not role_fits(role, "hard_resources") and not role_fits(role, "lt_framework")
    assert role_fits(ROLES["node.cpu_pct"], "hard_resources") and not role_fits(ROLES["node.cpu_pct"], "database")
    # The node SLA keeps reading node.* of hard_resources.
    assert role_for_sla_field("cpu_query").name == "node.cpu_pct"
    assert all(role.domain is not None for role in ROLES.values() if role.sla_field)


def test_pack_with_domains_validates():
    QueryPack.model_validate(_base_pack())
    for changes, message in (
        ({"domain": "database"}, "either domain or domains"),
        ({"domains": ["database"]}, "two domains or more"),
        ({"domains": ["database", "hard_resources"]}, "cannot be chosen at install time"),
        ({"domains": ["database", "database"]}, "repeat"),
        ({"sla": {"cpu_query": "@host.cpu_pct"}}, "cannot set sla"),
        ({"domain_title": "DB host"}, "cannot set sla, run_filter or domain_title"),
        ({"queries": [{"label": "{{host_title}}: CPU", "query": "node_load1", "role": "node.cpu_pct", "unit": "percent"}]}, "belongs to domain hard_resources"),
    ):
        with pytest.raises(ValueError, match=message):
            QueryPack.model_validate(_base_pack(**changes))


def test_a_single_domain_pack_cannot_use_host_roles_outside_component_domains():
    data = _base_pack(domains=[], domain="hard_resources", max_instances=1, instance_variable="")
    with pytest.raises(ValueError, match="fits database, kafka, jvm, microservices, not hard_resources"):
        QueryPack.model_validate(data)
    QueryPack.model_validate({**data, "domain": "database"})


def test_instances_need_a_variable_in_every_label():
    for changes, message in (
        ({"instance_variable": ""}, "instance_variable must name a variable"),
        ({"instance_variable": "other"}, "instance_variable must name a variable"),
        ({"max_instances": 1}, "needs max_instances above 1"),
        ({"max_instances": 11}, "less than or equal to 10"),
        ({"variables": {"host_title": {"title": "T", "default": "Host", "pattern": "^.*$"}}}, "must not accept an empty value"),
        ({"queries": [{"label": "Host: CPU", "query": "node_load1 {{host_title}}", "role": "host.cpu_pct", "unit": "percent"}]}, "the label must use"),
    ):
        with pytest.raises(ValueError, match=message):
            QueryPack.model_validate(_base_pack(**changes))


def test_bundle_cannot_name_a_pack_whose_domain_is_picked(tmp_path):
    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "test-host.toml").write_text(
        'schema_version = 1\nid = "test-host"\nversion = 1\ntitle = "T"\ncategory = "infrastructure"\n'
        'domains = ["database", "kafka"]\nlanguage = "promql"\ndescription = "T"\n'
        '[[queries]]\nlabel = "Host: CPU"\nquery = "node_load1"\nrole = "host.cpu_pct"\nunit = "percent"\n',
        encoding="utf-8",
    )
    bundles = tmp_path / "bundles.toml"
    bundles.write_text('[[bundles]]\nid = "b1"\ntitle = "B"\ndescription = "B"\npacks = ["test-host"]\n', encoding="utf-8")
    with pytest.raises(PackCatalogError, match="domain to pick"):
        load_catalog_from(packs, bundles)


def test_host_packs_in_the_catalog():
    for pack_id in ("host-node-exporter", "host-windows-exporter"):
        pack = CATALOG.packs[pack_id]
        assert pack.target_domains == ("database", "kafka", "jvm", "microservices")
        assert pack.max_instances == 10 and pack.instance_variable == "host_title"
        rendered = render_pack(pack, {"host_title": "DB primary"})
        assert all(query.label.startswith("DB primary: ") for query in rendered.queries)
        assert {query.role for query in rendered.queries} == {name for name in ROLES if name.startswith("host.")}
        assert not rendered.sla_refs
    assert all(not CATALOG.packs[pack_id].domains for bundle in CATALOG.bundles.values() for pack_id in bundle.packs)


# ---- installations in a block -------------------------------------------------------------------------

def _install(block, title, mode="merge"):
    rendered = render_pack(CATALOG.packs[HOST_PACK], {"host_title": title, "matchers": f'instance=~"{title.lower()}.*"'})
    return merge_pack(block, rendered, mode)


def test_two_installations_live_side_by_side_in_one_block():
    block, plan = _install(None, "DB primary")
    assert len(plan.added) == 7
    block, plan = _install(block, "DB replica")
    assert len(plan.added) == 7 and not plan.updated and not plan.removed
    assert len(block["labels"]) == 14
    pack = CATALOG.packs[HOST_PACK]
    assert [item["instance"] for item in installed_instances(block, pack)] == ["DB primary", "DB replica"]
    # The same title updates its installation in place.
    block, plan = _install(block, "DB primary")
    assert len(plan.unchanged) == 7 and len(block["labels"]) == 14
    assert [item["instance"] for item in installed_instances(block, pack)] == ["DB replica", "DB primary"]
    validate_queries({"database": block})
    # An installation whose rows were all deleted is dropped; the other one stays.
    keep = [index for index, label in enumerate(block["labels"]) if not label.startswith("DB replica")]
    for key in ("labels", "promql_queries", "label_keys_list"):
        block[key] = [block[key][index] for index in keep]
    pruned = prune_block(copy.deepcopy(block))
    assert [item["instance"] for item in installed_instances(pruned, pack)] == ["DB primary"]


def test_host_rows_get_meaning_and_one_prompt_note_per_pack():
    block, _ = _install(None, "DB primary")
    block, _ = _install(block, "DB replica")
    notes = prompt_addenda(block)
    assert len(notes) == 1 and "node_exporter" in notes[0][1]
    annotations = series_annotations(block)
    assert annotations["DB replica: CPU usage (%) by node"]["meaning"] == ROLES["host.cpu_pct"].title
    # SLA references never pick a host row: the node CPU field reads hard_resources.
    sla, filled = resolve_sla_refs({}, {"database": block})
    assert "cpu_query" not in filled and "cpu_query" not in sla


# ---- API ---------------------------------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    base = {"queries": {}, "sla": {"target_rps": 100}, "per_area": {}, "data_sources": SOURCES, "domain_sources": BINDINGS}
    for key, value in base.items():
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(value))

    from loadlens_app.blueprints import query_packs as blueprint

    monkeypatch.setattr(blueprint, "_effective_cfg", lambda area, service: {
        "data_sources": core.CONFIG["data_sources"],
        "domain_sources": core.CONFIG["domain_sources"],
        "queries": core.CONFIG["queries"],
        "default_params": {"step": "30s"},
    })
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), runtime=runtime)


def _catalog_pack(env, pack_id):
    data = env.client.get("/config/query_packs").get_json()
    return next(pack for pack in data["packs"] if pack["id"] == pack_id)


def _apply(env, title, domain="database"):
    return env.client.post("/config/query_packs/apply", json={"pack_id": HOST_PACK, "domain": domain, "variables": {"host_title": title}})


def test_catalog_lists_the_domains_to_pick(env):
    pack = _catalog_pack(env, HOST_PACK)
    assert pack["domain_picked"] is True and pack["domains"] == ["database", "kafka", "jvm", "microservices"]
    assert pack["max_instances"] == 10 and pack["instance_variable"] == "host_title"
    assert set(pack["targets"]) == set(pack["domains"])
    assert pack["targets"]["kafka"] == {"compatibility": {"status": "ok", "reason": "", "source": "Prometheus"}, "installed": None, "instances": []}
    plain = _catalog_pack(env, "node-exporter")
    assert plain["domain_picked"] is False and plain["domains"] == ["hard_resources"] and plain["max_instances"] == 1


def test_apply_needs_one_of_the_pack_domains(env):
    missing = env.client.post("/config/query_packs/apply", json={"pack_id": HOST_PACK})
    assert missing.status_code == 400 and "Выберите домен" in missing.get_json()["error"]
    wrong = _apply(env, "Host", domain="hard_resources")
    assert wrong.status_code == 400 and "hard_resources" in wrong.get_json()["error"]
    # A pack with its own domain accepts that domain (or none) and refuses another one.
    assert env.client.post("/config/query_packs/preview", json={"pack_id": "node-exporter", "domain": "hard_resources"}).status_code == 200
    assert env.client.post("/config/query_packs/preview", json={"pack_id": "node-exporter", "domain": "database"}).status_code == 400
    assert core.CONFIG["queries"] == {}


def test_apply_installs_into_the_picked_domain_and_counts_installations(env):
    first = _apply(env, "DB primary")
    assert first.status_code == 200, first.get_json()
    data = first.get_json()
    assert data["domain"] == "database" and data["instance"] == "DB primary" and data["sla_changed"] is False
    assert len(core.CONFIG["queries"]["database"]["labels"]) == 7
    assert core.CONFIG["sla"] == {"target_rps": 100}
    assert _apply(env, "DB replica").status_code == 200
    target = _catalog_pack(env, HOST_PACK)["targets"]["database"]
    assert [item["instance"] for item in target["instances"]] == ["DB primary", "DB replica"]
    assert target["instances"][1]["selected_indexes"] == list(range(7))
    assert _catalog_pack(env, HOST_PACK)["targets"]["kafka"]["instances"] == []


def test_eleventh_installation_is_refused_but_updates_still_apply(env):
    for index in range(10):
        assert _apply(env, f"Host {index}").status_code == 200
    refused = _apply(env, "Host 10")
    assert refused.status_code == 409 and "10" in refused.get_json()["error"]
    assert len(core.CONFIG["queries"]["database"]["labels"]) == 70
    assert _apply(env, "Host 3").status_code == 200
    # Replacing the whole domain is not limited: it leaves one installation.
    replaced = env.client.post("/config/query_packs/apply", json={"pack_id": HOST_PACK, "domain": "database", "mode": "replace", "variables": {"host_title": "Host 10"}})
    assert replaced.status_code == 200
    assert len(core.CONFIG["queries"]["database"]["packs"]) == 1


def test_check_runs_against_the_picked_domain(env, monkeypatch):
    from loadlens_app import query_pack_service
    from loadlens_app.query_preview import QueryPreview

    domains = set()

    def fake_preview(cfg, **kwargs):
        domains.add(kwargs["domain"])
        return QueryPreview(ok=True, message="ok", series_count=1, point_count=2, series=["node=db-1"])

    monkeypatch.setattr(query_pack_service, "preview_metric_query", fake_preview)
    resp = env.client.post("/config/query_packs/check", json={"pack_id": HOST_PACK, "domain": "kafka", "variables": {"host_title": "Brokers"}})
    assert resp.status_code == 200, resp.get_json()
    assert domains == {"kafka"} and resp.get_json()["ok"] is True


def test_detect_scores_a_host_pack_by_its_best_domain():
    from loadlens_app.query_pack_service import ScopeState, detect_packs

    def discover(cfg, domains):
        names = {"kafka": {"node_cpu_seconds_total", "node_memory_MemAvailable_bytes", "node_uname_info"}}
        return {domain: DomainNames(domain, "Prometheus", "promql", frozenset(names.get(domain, ()))) for domain in domains}

    result = detect_packs(ScopeState(queries={}, sla={}, effective_cfg={}), discover)
    assert result["packs"][HOST_PACK]["score"] == 1.0 and result["packs"][HOST_PACK]["domain"] == "kafka"
    assert set(HOST_DOMAINS) <= set(result["domains"])
