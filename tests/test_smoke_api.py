import threading
import types

import pytest

from loadlens_app import core


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return None

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return FakeCursor()

    def close(self):
        return None

    def rollback(self):
        return None


@pytest.fixture(autouse=True)
def patch_core(monkeypatch):
    monkeypatch.setattr(core, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(core, "_metrics_service_entry", lambda service: ("demo", {"page_sample_id": "1", "page_parent_id": "1", "metrics": [], "logs": []}))
    monkeypatch.setattr(core, "_find_area_for_service", lambda service: "demo")
    monkeypatch.setattr(core, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(core, "_resolve_services_filter", lambda area: [])
    yield


@pytest.fixture
def client(monkeypatch):
    from app import create_app
    from loadlens_app.blueprints import dashboard

    def _sync_thread(target, **kwargs):
        target()
        thread = types.SimpleNamespace(start=lambda: None)
        return thread

    monkeypatch.setattr(dashboard, "update_report", lambda *args, **kwargs: {"page_id": "1", "page_url": "/reports/demo/test-run", "run_name": "test-run"})
    monkeypatch.setattr(threading, "Thread", lambda target, daemon: _sync_thread(target=target))

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_get_config_endpoint(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "areas" in data


def test_runs_endpoint_smoke(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_create_report_smoke(client):
    payload = {
        "start": "2024-11-01T10:00",
        "end": "2024-11-01T11:00",
        "service": "demo",
        "project_area": "demo",
        "use_llm": False,
        "save_to_db": False,
        "web_only": True,
    }
    resp = client.post("/create_report", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "accepted"
    assert "job_id" in data


