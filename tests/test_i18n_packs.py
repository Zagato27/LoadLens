# SPDX-License-Identifier: Apache-2.0
"""English query packs (plans/i18n-plan.md: 4.1 row "query packs", 4.2 class e, 4.6 task 11, 3.1 prompt_addendum).

Russian pack texts (title, description, help, domain_title, prompt_addendum) have English twins ``*_en``;
labels, queries and series keys are identifiers and stay the same in both languages. Role, unit and category
titles live in the ``packs`` dictionary. The settings API answers in the request locale; the texts for the
model follow the current locale, which the pipeline sets to the report language.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401,E402  (before update_page users)
from AI.i18n import get_locale, has_key, with_locale  # noqa: E402
from AI.query_packs import (  # noqa: E402
    CATEGORY_TITLES,
    ROLES,
    UNIT_TITLES,
    DerivedRowError,
    PackRenderError,
    add_summed_rps,
    domain_title,
    load_catalog,
    localized,
    merge_conflict,
    merge_pack,
    prompt_addenda,
    render_pack,
    series_annotations,
    sla_update,
)
from AI.query_packs.roles import UNITS  # noqa: E402
from loadlens_app import core  # noqa: E402
from tests.i18n_helpers import CYRILLIC, use_locale  # noqa: E402

CATALOG = load_catalog()
TEXT_FIELDS = ("title", "description", "help", "domain_title", "prompt_addendum")
CATEGORIES = ("load_tool", "runtime", "http", "database", "messaging", "infrastructure")


def _objects():
    """(where, model) of every pack object with text fields: packs, their variables and queries, bundles."""
    for pack in CATALOG.packs.values():
        yield pack.id, pack
        for name, var in pack.variables.items():
            yield f"{pack.id}.variables.{name}", var
        for index, query in enumerate(pack.queries):
            yield f"{pack.id}.queries.{index}", query
    for bundle in CATALOG.bundles.values():
        yield f"bundle {bundle.id}", bundle


# ---- catalog contract ----------------------------------------------------------------------------------

def test_every_russian_pack_text_has_an_english_twin():
    problems, checked = [], 0
    for where, obj in _objects():
        for field in TEXT_FIELDS:
            value = getattr(obj, field, None)
            twin = getattr(obj, f"{field}_en", None)
            if not isinstance(value, str):
                continue
            if twin and not value.strip():
                problems.append(f"{where}.{field}_en has no original")
            if twin and CYRILLIC.search(twin):
                problems.append(f"{where}.{field}_en has Cyrillic")
            if CYRILLIC.search(value):
                checked += 1
                if not (twin or "").strip():
                    problems.append(f"{where}.{field}_en is missing")
    assert not problems, "\n".join(problems)
    assert checked >= 500  # 546 Russian fields when the twins were written


def test_labels_queries_and_keys_are_the_same_in_both_languages():
    for pack in CATALOG.packs.values():
        with with_locale("ru"):
            ru = render_pack(pack).to_dict()
        with with_locale("en"):
            en = render_pack(pack).to_dict()
        for item in ru["queries"] + en["queries"]:
            assert not CYRILLIC.search(item["label"] + item["query"] + "".join(item["keys"])), (pack.id, item["label"])
        strip = lambda data: [{k: v for k, v in item.items() if k != "description"} for item in data["queries"]]  # noqa: E731
        assert strip(ru) == strip(en), pack.id
        assert ru["sla_refs"] == en["sla_refs"] and ru["sla_values"] == en["sla_values"], pack.id
        with with_locale("en"):
            assert not any(CYRILLIC.search(item["description"]) for item in en["queries"]), pack.id


def test_localized_falls_back_to_the_original():
    pack = CATALOG.packs["runtime-go"]  # a Latin title has no twin
    assert not pack.title_en
    with with_locale("en"):
        assert localized(pack, "title") == pack.title
        assert localized(pack, "description") == pack.description_en
    with with_locale("ru"):
        assert localized(pack, "description") == pack.description
    assert localized(pack, "description", "en") == pack.description_en
    assert localized(pack, "description", "ru") == pack.description


def test_role_unit_and_category_titles_come_from_the_dictionary():
    for lang in ("ru", "en"):
        assert all(has_key(f"packs.role.{name}", lang) for name in ROLES)
        assert all(has_key(f"packs.unit.{unit}", lang) for unit in UNITS)
        assert all(has_key(f"packs.category.{name}", lang) for name in CATEGORIES)
    # ru is the text that was hard-coded before.
    assert ROLES["runtime.cpu"].title == "CPU экземпляра сервиса"
    assert ROLES["lt.rps_total"].title == "Общий RPS теста"
    assert UNIT_TITLES["count"] == "шт." and UNIT_TITLES["fraction"] == "доля 0..1"
    assert UNIT_TITLES.get("parsecs", "parsecs") == "parsecs" and "parsecs" not in UNIT_TITLES
    assert list(CATEGORY_TITLES) == list(CATEGORIES)
    assert CATEGORY_TITLES["load_tool"] == "Нагрузочный инструмент"
    with with_locale("en"):
        assert ROLES["runtime.cpu"].title == "Service instance CPU"
        assert UNIT_TITLES["count"] == "count" and UNIT_TITLES["seconds"] == "s"
        assert dict(CATEGORY_TITLES)["infrastructure"] == "Nodes and Kubernetes"
        texts = [role.title for role in ROLES.values()] + list(UNIT_TITLES.values()) + list(CATEGORY_TITLES.values())
        assert not [text for text in texts if CYRILLIC.search(text)]


# ---- texts for the model (report language) ------------------------------------------------------------

GO_CPU = "Go runtime: CPU (share of limit) by (service, instance)"


def test_model_context_follows_the_report_language():
    pack = CATALOG.packs["runtime-go"]
    block, _ = merge_pack(None, render_pack(pack, {"service_label": "service"}))
    with with_locale("en"):
        notes = series_annotations(block)
        addenda = prompt_addenda(block)
    assert notes[GO_CPU]["meaning"] == "Service instance CPU" and notes[GO_CPU]["unit"] == "fraction 0..1"
    assert not [text for item in notes.values() for text in item.values() if CYRILLIC.search(text)]
    assert addenda == [(pack.title, pack.prompt_addendum_en.strip())]
    with with_locale("ru"):
        ru_notes = series_annotations(block)
        ru_addenda = prompt_addenda(block)
    assert ru_notes[GO_CPU]["meaning"] == "CPU экземпляра сервиса" and ru_notes[GO_CPU]["unit"] == "доля 0..1"
    assert ru_addenda == [(pack.title, pack.prompt_addendum.strip())] and "горутин" in ru_addenda[0][1]
    assert set(notes) == set(ru_notes)


def test_domain_title_follows_the_report_language():
    block, _ = merge_pack(None, render_pack(CATALOG.packs["k8s-workload-cpu"]))
    with with_locale("en"):
        assert domain_title(block) == "Containers"
    with with_locale("ru"):
        assert domain_title(block) == "Контейнеры"


def test_render_and_merge_messages_follow_the_locale():
    pack = CATALOG.packs["jvm-micrometer"]
    with with_locale("en"):
        with pytest.raises(PackRenderError) as bad_value:
            render_pack(pack, {"service_label": "a b"})
        with pytest.raises(PackRenderError) as unknown:
            render_pack(pack, {"nope": "1"})
        flux = {"labels": ["Old"], "flux_queries": ['from(bucket: "b")'], "label_tag_keys_list": [[]]}
        conflict = merge_conflict(flux, "promql")
        with pytest.raises(DerivedRowError) as empty:
            add_summed_rps(None)
        one_tool, _ = merge_pack(None, render_pack(CATALOG.packs["lt-k6-prometheus-rw"]))
        with pytest.raises(DerivedRowError) as single:
            add_summed_rps(one_tool)
        update = sla_update({"p95_query": "", "mean_latency_query": "My mean (seconds)", "latency_unit": "s"},
                            render_pack(CATALOG.packs["lt-k6-influxql"]))
    assert str(bad_value.value) == '"Label with the service name": the value "a b" does not fit this field'
    assert str(unknown.value) == "Unknown pack variables: nope"
    assert conflict == "the domain already has Flux queries, and a domain runs one language"
    assert str(empty.value) == "The domain has no queries"
    assert str(single.value) == 'At least two queries with the "Total test RPS" role from different tools are needed'
    assert update.warnings and not CYRILLIC.search(" ".join(update.warnings)) and "latency_unit" in update.warnings[0]
    with with_locale("ru"):
        with pytest.raises(DerivedRowError) as ru_single:
            add_summed_rps(one_tool)
        assert merge_conflict(flux, "promql") == "в домене уже есть запросы Flux, а домен выполняет один язык"
    assert str(ru_single.value) == "Нужны хотя бы два запроса с ролью «Общий RPS теста» от разных инструментов"


def test_project_readiness_shows_pack_titles_in_the_locale():
    from loadlens_app.projects import _installed_query_packs

    block, _ = merge_pack(None, render_pack(CATALOG.packs["db-hikaricp"]))
    with with_locale("en"):
        assert _installed_query_packs({"database": block})[0]["title"] == "HikariCP connection pool (Micrometer)"
    with with_locale("ru"):
        assert _installed_query_packs({"database": block})[0]["title"] == "Пул соединений HikariCP (Micrometer)"


# ---- settings API (request locale) ----------------------------------------------------------------------

SOURCES = {
    "prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}},
    "graf": {"title": "Grafana", "type": "grafana_proxy", "grafana": {"base_url": "http://grafana:3000", "auth": {"method": "basic", "username": "u", "password": "p"}}},
    "influx": {"title": "Influx", "type": "influxdb", "influxdb": {"url": "http://influx:8086", "org": "o", "token": "t"}},
}
BINDINGS = {
    "default": {"source": "prom"},
    "lt_framework": {"source": "graf", "datasource_uid": "influx-uid", "database": "k6"},
    "kafka": {"source": "influx", "bucket": "telegraf"},
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    import update_page

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"shop": {"title": "Shop", "services": {"checkout": {}}}}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", runtime)
    base = {"queries": {}, "sla": {"target_rps": 100}, "per_area": {}, "data_sources": SOURCES, "domain_sources": BINDINGS}
    for key, value in base.items():
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(value))

    from loadlens_app.blueprints import query_packs as blueprint

    monkeypatch.setattr(blueprint, "_effective_cfg", lambda area, service: {
        "data_sources": core.CONFIG["data_sources"],
        "domain_sources": core.CONFIG["domain_sources"],
        "default_params": {"step": "30s"},
    })
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), runtime=runtime)


def _text(data) -> str:
    return json.dumps(data, ensure_ascii=False)


def _pack(catalog: dict, pack_id: str) -> dict:
    return next(pack for pack in catalog["packs"] if pack["id"] == pack_id)


def test_catalog_in_english_has_no_russian(env):
    use_locale(env.client, "en")
    resp = env.client.get("/config/query_packs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert not CYRILLIC.search(_text(data)), CYRILLIC.findall(_text(data))[:20]
    hikari = _pack(data, "db-hikaricp")
    assert hikari["title"] == "HikariCP connection pool (Micrometer)"
    assert hikari["variables"][0]["title"] == "Label with the service name"
    assert hikari["queries"][0]["role_title"] == "Active pool connections" and hikari["queries"][0]["unit_title"] == "count"
    assert data["categories"]["infrastructure"] == "Nodes and Kubernetes"
    assert data["roles"]["runtime.cpu"]["title"] == "Service instance CPU"
    assert _pack(data, "k8s-workload-cpu")["domain_title"] == "Containers"
    assert _pack(data, "kafka-exporter")["compatibility"]["reason"] == 'The "Influx" source does not run PromQL'
    bundle = next(item for item in data["bundles"] if item["id"] == "spring-vm-jmeter")
    assert bundle["title"] == "Spring Boot on virtual machines + JMeter"


def test_catalog_in_russian_keeps_its_texts_and_shape(env):
    ru = env.client.get("/config/query_packs").get_json()
    use_locale(env.client, "en")
    en = env.client.get("/config/query_packs").get_json()
    hikari = _pack(ru, "db-hikaricp")
    assert hikari["title"] == "Пул соединений HikariCP (Micrometer)"
    assert hikari["description"].startswith("Метрики пула JDBC HikariCP из Micrometer")
    assert data_keys(ru) == data_keys(en)
    assert [pack["id"] for pack in ru["packs"]] == list(CATALOG.packs)
    assert sorted(pack["id"] for pack in en["packs"]) == sorted(CATALOG.packs)
    assert set(ru["bundles"][0]) == {"id", "title", "description", "packs"}
    assert ru["bundles"][0]["description"] == CATALOG.bundles["spring-k8s-k6-influxdb"].description
    for ru_pack in ru["packs"]:
        en_pack = _pack(en, ru_pack["id"])
        assert [q["label"] for q in ru_pack["queries"]] == [q["label"] for q in en_pack["queries"]]
        assert [q["query"] for q in ru_pack["queries"]] == [q["query"] for q in en_pack["queries"]]


def data_keys(data: dict) -> tuple:
    pack = data["packs"][0]
    return (sorted(data), sorted(pack), sorted(pack["queries"][0]), sorted(data["roles"]["runtime.cpu"]))


def test_replace_only_reason_in_english(env, monkeypatch):
    monkeypatch.setitem(core.CONFIG, "queries", {"jvm": {
        "labels": ["Old"], "influxql_queries": ["SELECT 1"], "label_tag_keys_list": [["a"]],
    }})
    use_locale(env.client, "en")
    compat = _pack(env.client.get("/config/query_packs").get_json(), "jvm-micrometer")["compatibility"]
    assert compat["status"] == "replace_only"
    assert compat["reason"] == ("The domain already has InfluxQL queries, and a domain runs one language: "
                                "the pack can only be applied by replacing the domain queries")


def test_preview_in_english_has_no_russian(env):
    use_locale(env.client, "en")
    resp = env.client.post("/config/query_packs/preview", json={"pack_id": "lt-k6-influxql"})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert not CYRILLIC.search(_text(data))
    assert any(item["description"] == "Total test RPS. Load steps and the stable maximum are found from it."
               for item in data["all_queries"])


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        ({"pack_id": "nope"}, 404, 'Pack "nope" not found'),
        ({"pack_id": "jvm-micrometer", "area": "missing"}, 404, 'Area "missing" not found'),
        ({"pack_id": "jvm-micrometer", "service": "checkout"}, 400, "A service is set together with an area"),
        ({"pack_id": "jvm-micrometer", "area": "shop", "service": "other"}, 404, 'Service "other" not found in area "shop"'),
        ({"pack_id": "jvm-micrometer", "variables": {"service_label": "a b"}}, 400,
         '"Label with the service name": the value "a b" does not fit this field'),
        ({"pack_id": "jvm-micrometer", "variables": "x"}, 400, "variables must be an object"),
        ({"pack_id": "jvm-micrometer", "include": "x"}, 400, "include must be a list of query names"),
        ({"pack_id": "jvm-micrometer", "include_indexes": "x"}, 400, "include_indexes must be a list of query numbers"),
        ({"pack_id": "jvm-micrometer", "include_indexes": [99]}, 400, "include_indexes: a query number is outside the pack"),
        ({"pack_id": "jvm-micrometer", "include_indexes": []}, 400, "No queries of the pack are selected"),
        ({"pack_id": "jvm-micrometer", "mode": "upsert"}, 400, "mode: allowed values are merge, replace"),
        ({"pack_id": "kafka-exporter"}, 409, 'The "Influx" source does not run PromQL'),
    ],
)
def test_apply_errors_in_english(env, body, status, message):
    use_locale(env.client, "en")
    resp = env.client.post("/config/query_packs/apply", json=body)
    assert resp.status_code == status
    assert resp.get_json()["error"] == message
    assert core.CONFIG["queries"] == {}


def test_request_errors_in_english(env):
    use_locale(env.client, "en")
    resp = env.client.post("/config/query_packs/preview", data="[]", content_type="application/json")
    assert resp.status_code == 400 and resp.get_json()["error"] == "A JSON object is expected"
    resp = env.client.post("/config/query_packs/summed_rps", json={})
    assert resp.status_code == 400 and resp.get_json()["error"] == "The domain has no queries"
    use_locale(env.client, None)
    resp = env.client.post("/config/query_packs/summed_rps", json={})
    assert resp.status_code == 400 and resp.get_json()["error"] == "В домене нет запросов"


def test_check_answers_in_the_request_locale_from_worker_threads(env, monkeypatch):
    from loadlens_app import query_pack_service
    from loadlens_app.query_preview import QueryPreview

    seen = []

    def fake_preview(cfg, *, domain, lang, query, label_keys, start_ts, end_ts):
        seen.append(get_locale())
        return QueryPreview(ok=True, message="1 series", series_count=1, point_count=10, series=["all"])

    monkeypatch.setattr(query_pack_service, "preview_metric_query", fake_preview)
    use_locale(env.client, "en")
    resp = env.client.post("/config/query_packs/check", json={"pack_id": "lt-k6-influxql", "include_indexes": [3, 9],
                                                               "start": "2024-11-01T10:00", "end": "2024-11-01T11:00"})
    assert resp.status_code == 200, resp.get_json()
    assert seen == ["en", "en"]
    # "window" is QueryPreview's own text (loadlens_app/query_preview.py), not a pack text.
    results = [{k: v for k, v in item.items() if k != "window"} for item in resp.get_json()["results"]]
    assert not CYRILLIC.search(_text(results))
