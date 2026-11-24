# SPDX-License-Identifier: Apache-2.0
"""Project selection, credential isolation and LLM routing through saved connections."""

import base64
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import requests

from AI import pipeline, providers
from loadlens_app import core
from loadlens_app.llm_profiles import effective_llm


@pytest.fixture
def client(monkeypatch, tmp_path):
    from app import create_app

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({
        "per_area": {
            "shop": {"llm": {"openai": {"model": "legacy-shop"}}},
            "orders": {},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_runtime.json")
    monkeypatch.setitem(core.CONFIG, "per_area", {})
    monkeypatch.setitem(core.CONFIG, "llm", {
        "provider": "openai", "max_domain_workers": 2,
        "openai": {"api_base_url": "https://legacy.example/v1", "model": "legacy", "api_key": "legacy-key"},
    })
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-env-key")
    return create_app().test_client()


def request(client, action, area="", profile_id="", **data):
    return client.post("/config/llm_profiles", json={"action": action, "area": area, "id": profile_id, "data": data})


def save(client, name, area=""):
    result = request(client, "save", area, name=name, provider="openai", settings={
        "api_base_url": f"https://{name}.example/v1", "model": name, "api_key": f"{name}-key",
    })
    assert result.status_code == 200, result.get_json()
    return result.get_json()["id"]


def select(client, area, profile_id):
    result = request(client, "activate", area, profile_id)
    assert result.status_code == 200, result.get_json()
    return result.get_json()


def profiles_for(client, area):
    response = client.get("/config", query_string={"area": area})
    assert response.status_code == 200, response.get_json()
    return {profile["id"]: profile for profile in response.get_json()["llm_profiles"]["profiles"]}


def test_profiles_survive_reload_and_projects_choose_independently(client):
    assert effective_llm("shop")["openai"]["model"] == "legacy-shop"
    assert effective_llm("orders")["openai"]["model"] == "legacy"
    with providers.llm_scope(effective_llm("orders")):
        assert providers.provider_api_key("openai", providers.current_llm_config()["openai"]) == "legacy-env-key"
    first = save(client, "first")
    second = save(client, "second", "shop")
    assert request(client, "activate", "orders", second).status_code == 404
    select(client, "", first)
    select(client, "shop", second)
    assert effective_llm("orders")["_profile_id"] == first
    assert effective_llm("shop")["_profile_id"] == second
    shop_catalog = client.get("/config?area=shop").get_json()["llm_profiles"]
    selected = next(profile for profile in shop_catalog["profiles"] if profile["id"] == shop_catalog["active_id"])
    assert selected["settings"]["api_key"] == "***"
    selected["name"] = "Renamed"
    assert request(client, "save", "shop", second, **{k: selected[k] for k in ("name", "provider", "settings")}).status_code == 200
    core._clear_runtime_cache()
    assert effective_llm("shop")["openai"]["api_key"] == "second-key"
    assert effective_llm("shop")["max_domain_workers"] == 2
    assert request(client, "delete", "shop", second).status_code == 409
    assert select(client, "shop", "")["active_id"] == first
    assert request(client, "delete", "shop", second).status_code == 200
    select(client, "orders", first)
    select(client, "", "legacy-openai")
    assert request(client, "delete", "", first).status_code == 409
    select(client, "orders", "")
    assert request(client, "delete", "", first).status_code == 200


def test_analysis_settings_keep_a_legacy_provider_with_built_in_defaults(client, monkeypatch):
    monkeypatch.setitem(core.CONFIG, "llm", {"provider": "openai"})
    result = request(client, "analysis", max_domain_workers=3, time_shift_hours=1)
    assert result.status_code == 200
    llm = effective_llm()
    assert llm == {"provider": "openai", "max_domain_workers": 3, "time_shift_hours": 1}
    with providers.llm_scope(llm):
        assert pipeline._time_shift_hours() == 1


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "saved"])
def test_sharing_preserves_the_owner_selection_and_isolates_existing_project_copies(client, legacy):
    if legacy:
        previous_id = "legacy-project-openai"
    else:
        previous_id = save(client, "private", "shop")
        select(client, "shop", previous_id)
    copied = client.post("/projects", json={"id": "shop-copy", "copy_from": "shop"})
    assert copied.status_code == 201, copied.get_json()
    previous_copy = effective_llm("shop-copy")
    assert previous_copy["_profile_id"] == previous_id
    draft = profiles_for(client, "shop")[previous_id]
    assert draft["settings"]["api_key"] == "***"
    result = request(client, "save", "shop", previous_id, shared=True, name="Shared", provider=draft["provider"],
                     settings={**draft["settings"], "model": "shared-model"})
    assert result.status_code == 200, result.get_json()
    shared_id = result.get_json()["id"]
    assert shared_id != previous_id
    assert effective_llm("shop")["_profile_id"] == shared_id
    assert effective_llm("shop-copy") == previous_copy
    assert previous_id not in profiles_for(client, "shop")
    assert {previous_id, shared_id} <= profiles_for(client, "shop-copy").keys()
    select(client, "shop-copy", shared_id)
    assert effective_llm("shop-copy") == effective_llm("shop")


def test_shared_profile_updates_all_consumers_and_can_be_made_private_again(client):
    private_id = save(client, "team", "shop")
    masked = profiles_for(client, "shop")[private_id]["settings"]
    draft = {"name": "Team", "provider": "openai", "settings": masked}
    result = request(client, "save", "shop", private_id, **draft, shared=True)
    assert result.status_code == 200, result.get_json()
    shared_id = result.get_json()["id"]
    profile = profiles_for(client, "orders")[shared_id]
    assert profile["shared"] is True and profile["area"] == "shop"
    assert profile["settings"]["api_key"] == "***"
    assert effective_llm("orders")["_profile_id"] == "legacy-openai"
    assert effective_llm("shop")["_profile_id"] == "legacy-project-openai"
    assert request(client, "save", "orders", shared_id, **draft).status_code == 404
    assert request(client, "delete", "orders", shared_id).status_code == 404

    select(client, "orders", shared_id)
    draft["settings"] = {**masked, "model": "next"}
    updated = request(client, "save", "shop", shared_id, **draft)
    assert updated.status_code == 200, updated.get_json()
    assert updated.get_json()["id"] == shared_id
    assert profiles_for(client, "orders")[shared_id]["settings"]["model"] == "next"
    core._clear_runtime_cache()
    assert effective_llm("orders")["openai"] == {**draft["settings"], "api_key": "team-key"}
    assert request(client, "save", "shop", shared_id, **draft, shared=False).status_code == 409
    assert request(client, "delete", "shop", shared_id).status_code == 409
    select(client, "orders", "")
    select(client, "", shared_id)
    assert request(client, "save", "shop", shared_id, **draft, shared=False).status_code == 409
    select(client, "", "legacy-openai")

    select(client, "shop", shared_id)
    unshared = request(client, "save", "shop", shared_id, **draft, shared=False)
    assert unshared.status_code == 200, unshared.get_json()
    local_id = unshared.get_json()["id"]
    assert effective_llm("shop")["_profile_id"] == local_id
    assert effective_llm("shop")["openai"]["api_key"] == "team-key"
    assert profiles_for(client, "shop")[local_id]["shared"] is False
    assert shared_id not in profiles_for(client, "orders")
    assert request(client, "activate", "orders", local_id).status_code == 404


def test_new_project_connection_can_be_shared_without_setting_a_global_default(client, monkeypatch):
    # An installation can initially have only metrics-defined projects and no runtime settings.
    core.CONFIG_RUNTIME_PATH.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "METRICS_CONFIG", {"shop": {"services": {}}, "orders": {"services": {}}})
    monkeypatch.setitem(core.CONFIG, "llm", {})
    result = request(client, "save", "orders", name="Team", provider="openai", settings={"model": "team"}, shared=True)
    assert result.status_code == 200, result.get_json()
    shared_id = result.get_json()["id"]
    assert set(core._per_area_config()) == {"shop", "orders"}
    assert profiles_for(client, "shop")[shared_id]["area"] == "orders"
    assert "_profile_id" not in effective_llm()
    assert "_profile_id" not in effective_llm("orders")
    assert request(client, "delete", "orders", shared_id).status_code == 200
    assert shared_id not in profiles_for(client, "shop")


def test_shared_connection_survives_deletion_of_its_source_project(client, monkeypatch):
    from loadlens_app import projects

    monkeypatch.setattr(projects, "_service_titles", lambda _area: {})
    saved = request(client, "save", "shop", name="Team", provider="openai", shared=True,
                    settings={"model": "team", "api_key": "team-key"})
    assert saved.status_code == 200, saved.get_json()
    shared_id = saved.get_json()["id"]
    select(client, "orders", shared_id)
    removed = client.delete("/projects/shop?with_data=0")
    assert removed.status_code == 200, removed.get_json()
    profile = profiles_for(client, "orders")[shared_id]
    assert profile["area"] == ""
    assert profile["shared"] is True
    assert effective_llm("orders")["_profile_id"] == shared_id
    edited = request(client, "save", "", shared_id, name="Team", provider="openai",
                     settings={**profile["settings"], "model": "next"})
    assert edited.status_code == 200, edited.get_json()
    assert effective_llm("orders")["openai"] == {"model": "next", "api_key": "team-key"}


def test_checks_use_the_draft_credentials_and_reject_moving_masked_secrets(client, monkeypatch):
    first = save(client, "first")
    catalog = client.get("/config").get_json()["llm_profiles"]
    settings = next(profile["settings"] for profile in catalog["profiles"] if profile["id"] == first)
    calls = []

    def respond(url, headers, **kwargs):
        calls.append((url, headers["Authorization"], (kwargs.get("json") or {}).get("model")))
        return SimpleNamespace(
            ok=True, status_code=200, raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "OK"}}], "data": [{"id": "first"}]},
        )

    monkeypatch.setattr(providers.requests, "post", respond)
    monkeypatch.setattr(providers.requests, "get", respond)
    for action in ("test", "models"):
        response = request(client, action, "", first, provider="openai", settings=settings)
        assert response.get_json()["ok"], response.get_json()
    assert calls == [
        ("https://first.example/v1/chat/completions", "Bearer first-key", "first"),
        ("https://first.example/v1/models", "Bearer first-key", None),
    ]
    del settings["api_base_url"]
    response = request(client, "test", "", first, provider="openai", settings=settings)
    assert response.status_code == 400
    assert len(calls) == 2


def test_running_report_keeps_its_profile_after_switching(client, monkeypatch):
    first = save(client, "first", "shop")
    second = save(client, "second", "shop")
    select(client, "shop", first)
    reached, advance = Event(), Event()
    calls = []

    def collect(*args, **kwargs):
        if providers.current_llm_config()["_profile_id"] == first:
            reached.set()
            assert advance.wait(5)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(contextvars.copy_context().run, providers.ask_llm_with_text_data, "Check", "").result()

    def provider_call(provider, messages, settings, system):
        calls.append((settings["model"], providers.provider_api_key(provider, settings)))
        return settings["model"]

    monkeypatch.setattr(pipeline, "_upload_from_llm", collect)
    monkeypatch.setattr(providers, "_call_provider", provider_call)
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(pipeline.uploadFromLLM, 0, 30, ef_config={"llm": effective_llm("shop")})
        assert reached.wait(5)
        try:
            select(client, "shop", second)
            new = pool.submit(pipeline.uploadFromLLM, 0, 30, ef_config={"llm": effective_llm("shop")})
            assert new.result(timeout=5) == "second"
        finally:
            advance.set()
        assert old.result(timeout=5) == "first"
    assert calls == [("second", "second-key"), ("first", "first-key")]


def test_authenticated_proxy_is_used_by_checks_models_and_saved_connections(client):
    calls = []
    accept_auth = True

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            model = json.loads(data).get("model") if data else None
            calls.append((self.command, self.path, self.headers.get("Proxy-Authorization"), model))
            if not accept_auth:
                self.send_error(407)
                return
            response = {"data": [{"id": "test-model"}], "choices": [{"message": {"content": "OK"}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        do_GET = respond
        do_POST = respond
        do_CONNECT = respond

    with ThreadingHTTPServer(("127.0.0.1", 0), Proxy) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            proxy_host = f"127.0.0.1:{server.server_port}"
            base = "http://127.0.0.1:1/v1"
            profile = {"name": "Proxy", "provider": "openai", "settings": {
                "model": "test-model", "api_key": "model-key", "api_base_url": base,
                "proxy_url": f" http://demo:sample%40word@{proxy_host} ",
            }}
            saved = request(client, "save", "shop", **profile)
            assert saved.status_code == 200, saved.get_json()
            profile_id = saved.get_json()["id"]
            select(client, "shop", profile_id)
            catalog = client.get("/config?area=shop").get_json()["llm_profiles"]
            masked = next(item["settings"] for item in catalog["profiles"] if item["id"] == profile_id)
            assert masked["proxy_url"] == f"http://demo:***@{proxy_host}"
            assert "sample%40word" not in json.dumps(catalog)
            for action in ("test", "models"):
                checked = request(client, action, "shop", profile_id, provider="openai", settings=masked)
                assert checked.get_json()["ok"], checked.get_json()
            with providers.llm_scope(effective_llm("shop")):
                assert providers.ask_llm_with_text_data("Hello", "") == "OK"

            header = "Basic " + base64.b64encode(b"demo:sample@word").decode()
            post = ("POST", f"{base}/chat/completions", header, "test-model")
            assert calls == [post, ("GET", f"{base}/models", header, None), post]
            assert request(client, "save", "shop", profile_id, name="Proxy", provider="openai", settings={
                **masked, "proxy_url": "http://demo:***@other.example:8080",
            }).status_code == 400

            accept_auth = False
            https_draft = {**masked, "api_base_url": "https://127.0.0.1:1/v1", "api_key": "model-key"}
            checked = request(client, "test", "shop", profile_id, provider="openai", settings=https_draft)
            assert checked.status_code == 200 and not checked.get_json()["ok"]
            assert calls[-1] == ("CONNECT", "127.0.0.1:1", header, None)
            assert "sample" not in checked.get_data(as_text=True)
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("url", ["localhost:8080", "http://demo:sample@word@proxy.example", "http://proxy.example:65536"])
def test_invalid_proxy_url_is_rejected_without_echoing_credentials(client, url):
    result = request(client, "save", "shop", name="Proxy", provider="openai", settings={"proxy_url": url})
    assert result.status_code == 400
    assert url not in result.get_data(as_text=True)


def _moonshot_response(url, status, payload):
    response = requests.Response()
    response.status_code = status
    response.url = url
    response._content = json.dumps(payload).encode()
    return response


@pytest.mark.parametrize(("model", "budget_field", "probe_tokens", "probe_options"), [
    ("kimi-k2.5", "max_tokens", 16, {"thinking": {"type": "disabled"}}),
    ("kimi-k2.6", "max_completion_tokens", 16, {"thinking": {"type": "disabled"}}),
    ("kimi-k2.7-code", "max_completion_tokens", 512, {}),
    ("kimi-k3.1-pro", "max_completion_tokens", 512, {"reasoning_effort": "low"}),
])
def test_moonshot_uses_model_compatible_parameters_for_checks_and_analysis(
    client, monkeypatch, model, budget_field, probe_tokens, probe_options,
):
    payloads = []

    def respond(url, *, json, **_kwargs):
        assert url == "https://api.moonshot.ai/v1/chat/completions"
        payloads.append({key: value for key, value in json.items() if key != "messages"})
        if "temperature" in json or "top_p" in json:
            return _moonshot_response(url, 400, {"error": {"message": "Fixed sampling parameters"}})
        return _moonshot_response(url, 200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(providers.requests, "post", respond)
    settings = {
        "api_base_url": "https://api.moonshot.ai/v1", "api_key": "test-key", "model": model,
        "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1200},
    }
    check = request(client, "test", "shop", provider="openai", settings=settings)
    assert check.get_json()["ok"], check.get_json()
    assert payloads[0] == {"model": model, budget_field: probe_tokens, **probe_options}

    assert providers._openai_call([{"role": "user", "content": "Hello"}], settings) == "OK"
    assert payloads[1] == {"model": model, budget_field: 1200}


def test_moonshot_validation_error_shows_the_reason_without_credentials(client, monkeypatch):
    key = "test-api-key"
    proxy_password = "sample@word"
    proxy_auth = base64.b64encode(f"demo:{proxy_password}".encode()).decode()
    detail = f"Model is unavailable. {key} {proxy_password} sample%40word {proxy_auth}"
    monkeypatch.setattr(providers.requests, "post", lambda url, **_kwargs: _moonshot_response(
        url, 400, {"error": {"message": detail}},
    ))
    check = request(client, "test", "shop", provider="openai", settings={
        "api_base_url": "https://api.moonshot.ai/v1", "model": "kimi-unknown", "api_key": key,
        "proxy_url": "http://demo:sample%40word@proxy.example:8080",
    })
    result = check.get_json()
    assert not result["ok"]
    assert "HTTP 400" in result["message"] and "Model is unavailable." in result["message"]
    assert all(secret not in check.get_data(as_text=True) for secret in (key, proxy_password, "sample%40word", proxy_auth))


def test_forecast_uses_the_project_of_its_saved_report(client, monkeypatch):
    from loadlens_app import forecast_explanation as forecast

    own = save(client, "project", "shop")
    select(client, "shop", own)
    report = SimpleNamespace(project_area="shop", sla_details={})
    resources = SimpleNamespace(limit_cause=None, bottleneck=None, limit_service=None)
    monkeypatch.setattr(forecast, "_final_report", lambda _run_id: report)
    monkeypatch.setattr(forecast, "build_forecast", lambda *_args: SimpleNamespace(resources=resources, service="checkout"))
    monkeypatch.setattr(forecast, "_limit_window", lambda _forecast: None)
    monkeypatch.setattr(forecast, "_analyses", lambda _run_id: SimpleNamespace(system_context={}))
    monkeypatch.setattr(forecast, "_facts", lambda _forecast, _lang: {})
    monkeypatch.setattr(providers, "_call_provider", lambda provider, _messages, cfg, _prompt: providers.provider_api_key(provider, cfg))

    context = forecast._context("saved-report")
    assert context.llm.model == "project"
    assert forecast._ask_model(context) == "project-key"
