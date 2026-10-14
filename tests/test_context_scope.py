"""System context per service: stored under the service, used by reports, shown on the project card."""

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core  # noqa: E402  (before update_page)
from loadlens_app import projects  # noqa: E402
import update_page  # noqa: E402

AREA_CTX = {
    "system": {"name": "Shop", "description": "Интернет-магазин"},
    "architecture": {"components": [{"id": "api", "name": "API"}, {"id": "db", "name": "PostgreSQL"}]},
}
SERVICE_CTX = {
    "system": {"name": "Checkout", "description": "Оформление заказа"},
    "architecture": {"components": [{"id": "checkout", "name": "Checkout service"}]},
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"shop": {
        "title": "Shop",
        "system_context": core._normalize_system_context(AREA_CTX),
        "services": {"checkout": {}, "cart": {}},
    }}}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    monkeypatch.setitem(core.CONFIG, "system_context", {})
    monkeypatch.setitem(core.CONFIG, "per_area", {})

    def no_db():
        raise AssertionError("context scope tests must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)

    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), runtime


def _runtime(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(client, data, service=""):
    body = {"section": "system_context", "data": data, "area": "shop"}
    if service:
        body["service"] = service
    resp = client.post("/config", json=body)
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def test_service_context_is_stored_under_the_service(env):
    client, runtime = env
    answer = _save(client, SERVICE_CTX, "checkout")
    assert answer["scope"] == "service" and answer["service"] == "checkout" and "inherited" not in answer
    area = _runtime(runtime)["per_area"]["shop"]
    stored = area["services"]["checkout"]["system_context"]
    assert stored["system"]["name"] == "Checkout"
    assert [c["id"] for c in stored["architecture"]["components"]] == ["checkout"]
    assert area["system_context"]["system"]["name"] == "Shop"  # the area keeps its own value
    assert "system_context" not in area["services"]["cart"]


def test_config_lists_only_services_with_their_own_context(env):
    client, _ = env
    _save(client, SERVICE_CTX, "checkout")
    cfg = client.get("/config", query_string={"area": "shop"}).get_json()
    assert cfg["system_context"]["system"]["name"] == "Shop"
    assert set(cfg["service_system_context"]) == {"checkout"}
    assert cfg["service_system_context"]["checkout"]["system"]["description"] == "Оформление заказа"


def test_saving_the_area_value_for_a_service_removes_the_override(env):
    client, runtime = env
    _save(client, SERVICE_CTX, "checkout")
    area_value = client.get("/config", query_string={"area": "shop"}).get_json()["system_context"]
    answer = _save(client, area_value, "checkout")
    assert answer["inherited"] is True
    assert "system_context" not in _runtime(runtime)["per_area"]["shop"]["services"]["checkout"]
    # Saving the area value for a service without an override changes nothing.
    assert _save(client, area_value, "cart")["inherited"] is True


def test_reports_use_the_service_context_and_the_area_one_otherwise(env):
    client, _ = env
    _save(client, SERVICE_CTX, "checkout")
    checkout = update_page._effective_config_for_scope("shop", "checkout")["system_context"]
    cart = update_page._effective_config_for_scope("shop", "cart")["system_context"]
    assert checkout["system"]["name"] == "Checkout"
    # The service list replaces the area list as a whole.
    assert [c["id"] for c in checkout["architecture"]["components"]] == ["checkout"]
    assert cart["system"]["name"] == "Shop"
    assert [c["id"] for c in cart["architecture"]["components"]] == ["api", "db"]


def test_project_card_shows_and_resets_the_service_context(env):
    client, runtime = env
    _save(client, SERVICE_CTX, "checkout")
    area_entry = _runtime(runtime)["per_area"]["shop"]
    summary = projects._service_summary("checkout", "Checkout", area_entry, None)
    assert "system_context" in summary.own_sections
    assert projects.reset_service_section("shop", "checkout", "system_context") is True
    assert "system_context" not in _runtime(runtime)["per_area"]["shop"]["services"]["checkout"]
