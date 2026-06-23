import copy
import json
import sys
import threading
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core
from AI.pipeline import (
    _find_stable_peak_step_profile,
    _has_meaningful_system_context,
    _reconcile_sla_for_test_profile,
    _select_step_profile_candidate,
)
from AI.scoring import parse_llm_analysis_strict


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


class RowsCursor(FakeCursor):
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class RowsConnection(FakeConnection):
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return RowsCursor(self._rows)


@pytest.fixture(autouse=True)
def patch_core(monkeypatch, tmp_path):
    runtime_path = tmp_path / "settings_runtime.json"
    metrics_runtime_path = tmp_path / "metrics_config_runtime.json"
    original_system_context = copy.deepcopy(core.CONFIG.get("system_context"))
    monkeypatch.setattr(core, "_ts_conn", lambda: FakeConnection())
    monkeypatch.setattr(core, "_metrics_service_entry", lambda service: ("demo", {"page_sample_id": "1", "page_parent_id": "1", "metrics": [], "logs": []}))
    monkeypatch.setattr(core, "_find_area_for_service", lambda service: "demo")
    monkeypatch.setattr(core, "_bootstrap_service_configs", lambda area, service: None)
    monkeypatch.setattr(core, "_resolve_services_filter", lambda area: [])
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime_path)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", metrics_runtime_path)
    core.CONFIG["system_context"] = copy.deepcopy(original_system_context or {})
    yield
    core.CONFIG["system_context"] = original_system_context


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
    assert "system_context" in data


def test_runs_endpoint_smoke(client):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_rename_run_endpoint_updates_report_name(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    captured = {}

    def fake_rename(old_name, new_name):
        captured["old_name"] = old_name
        captured["new_name"] = new_name
        return {"status": "ok", "renamed": 3, "run_name": new_name, "old_run_name": old_name}

    monkeypatch.setattr(dashboard, "_rename_run_data", fake_rename)

    resp = client.patch("/runs/old-report", json={"new_run_name": "new-report"})

    assert resp.status_code == 200
    assert resp.get_json()["run_name"] == "new-report"
    assert captured == {"old_name": "old-report", "new_name": "new-report"}


def test_rename_run_endpoint_rejects_empty_name(client):
    resp = client.patch("/runs/old-report", json={"new_run_name": "   "})

    assert resp.status_code == 400
    assert "Новое имя" in resp.get_json()["error"]


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


def test_update_system_context_via_config(client):
    payload = {
        "schema_version": 1,
        "system": {
            "name": "Checkout Platform",
            "domain": "e-commerce",
            "description": "Обработка заказов и оплат",
            "test_goal": "Проверить стабильность checkout",
        },
        "architecture": {
            "style": "microservices",
            "components": [
                {
                    "id": "gateway",
                    "name": "API Gateway",
                    "role": "Внешняя точка входа",
                    "criticality": "high",
                    "technologies": ["nginx", "spring"],
                }
            ],
            "dependencies": [
                {
                    "from": "gateway",
                    "to": "orders",
                    "kind": "sync_http",
                    "purpose": "Маршрутизация запросов",
                }
            ],
            "data_stores": [],
        },
        "load_model": {
            "entrypoints": [{"id": "checkout", "name": "POST /checkout", "kind": "http", "business_priority": "high"}],
            "critical_user_flows": [{"id": "checkout", "name": "Оформление заказа", "steps": ["gateway", "orders"], "success_signals": ["2xx"]}],
            "expected_hotspots": ["gateway", "orders"],
        },
        "operational_context": {
            "known_constraints": ["Общий Redis"],
            "known_risks": ["Рост latency при shared DB"],
            "normal_degradation_rules": [],
            "analysis_focus": ["Проверить p95 checkout"],
        },
    }
    resp = client.post("/config", json={"section": "system_context", "data": payload})
    assert resp.status_code == 200

    cfg_resp = client.get("/config")
    assert cfg_resp.status_code == 200
    data = cfg_resp.get_json()
    assert data["system_context"]["system"]["name"] == "Checkout Platform"
    assert data["system_context"]["architecture"]["components"][0]["name"] == "API Gateway"


def test_system_context_can_be_disabled(client):
    payload = {
        "enabled": False,
        "schema_version": 1,
        "system": {
            "name": "Disabled Context Example",
        },
    }
    resp = client.post("/config", json={"section": "system_context", "data": payload})
    assert resp.status_code == 200

    cfg_resp = client.get("/config")
    assert cfg_resp.status_code == 200
    data = cfg_resp.get_json()
    assert data["system_context"]["enabled"] is False
    assert data["system_context"]["system"]["name"] == "Disabled Context Example"
    assert _has_meaningful_system_context(data["system_context"]) is False


def test_parse_llm_analysis_preserves_verdict_rationale():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "verdict_rationale": (
                "Вердикт снижен до уровня риска из-за локальных деградаций в пиковое окно.\n"
                "- p95 вышел за ориентир.\n"
                "- Ошибки выросли одновременно с ростом нагрузки."
            ),
            "confidence": 0.81,
            "findings": [],
            "recommended_actions": [],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.verdict == "Есть риски"
    assert "локальных деградаций" in (parsed.verdict_rationale or "")


def test_parse_llm_analysis_preserves_finding_links():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "verdict_rationale": "Есть локальные деградации.",
            "confidence": 0.77,
            "findings": [
                {
                    "id": "f1",
                    "summary": "P95 latency вырос до 420мс",
                    "severity": "high",
                    "component": "orders",
                    "evidence": "service=orders, 12:10-12:20, peak_time=12:16",
                },
                {
                    "summary": "Ошибки выросли до 1.2%",
                    "severity": "medium",
                    "component": "orders",
                    "evidence": "service=orders, 12:12-12:18, peak_time=12:17",
                },
            ],
            "recommended_actions": [
                {
                    "summary": "Проверить пул соединений и таймауты upstream.",
                    "details": "Сверить лимиты пула соединений с пиковым уровнем конкурентности и отдельно проверить таймауты upstream. После корректировки повторить тест и убедиться, что latency и доля timeout-ошибок снизились.",
                    "priority": "high",
                    "affected_components": ["orders"],
                    "for_finding_ids": ["f1"],
                },
                {
                    "summary": "Усилить retry budget и алерты.",
                    "details": "Ограничить агрессивные ретраи, чтобы они не усиливали деградацию, и добавить алерты на ранние признаки роста ошибок.",
                    "priority": "medium",
                    "affected_components": ["orders"],
                },
            ],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.findings[0].id == "f1"
    assert parsed.findings[1].id == "finding_2"
    assert parsed.recommended_actions[0].for_finding_ids == ["f1"]
    assert parsed.recommended_actions[1].for_finding_ids == ["f1", "finding_2"]
    assert "конкурентности" in (parsed.recommended_actions[0].details or "")


def test_parse_llm_analysis_derives_missing_verdict_rationale():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "findings": [
                {"id": "f1", "summary": "Kafka lag превышает порог", "severity": "high", "component": "kafka"},
            ],
            "recommended_actions": [],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.verdict_rationale
    assert "Kafka lag превышает порог" in parsed.verdict_rationale


def test_parse_llm_analysis_preserves_structured_finding_evidence():
    raw = json.dumps(
        {
            "verdict": "Есть риски",
            "verdict_rationale": "Есть локальные деградации.",
            "confidence": 0.79,
            "findings": [
                {
                    "id": "f1",
                    "summary": "P95 latency выросла в пиковое окно",
                    "severity": "high",
                    "component": "orders",
                    "start_time": "12:10",
                    "end_time": "12:18",
                    "peak_time": "12:16",
                    "evidence_summary": "Рост latency совпал с ростом нагрузки.",
                    "evidence_items": [
                        {
                            "metric": "P95 latency",
                            "observed_value": "420мс",
                            "threshold": "300мс",
                            "note": "service=orders",
                        }
                    ],
                }
            ],
            "recommended_actions": [
                {
                    "summary": "Проверить лимиты и таймауты upstream.",
                    "details": "Проверить лимиты downstream и таймауты клиентских вызовов в окне пика. Затем повторить прогон и убедиться, что p95 и error rate стабилизировались.",
                    "priority": "high",
                    "affected_components": ["orders"],
                    "for_finding_ids": ["f1"],
                }
            ],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_analysis_strict(raw)
    assert parsed is not None
    assert parsed.findings[0].start_time == "12:10"
    assert parsed.findings[0].end_time == "12:18"
    assert parsed.findings[0].peak_time == "12:16"
    assert parsed.findings[0].evidence_summary == "Рост latency совпал с ростом нагрузки."
    assert parsed.findings[0].evidence_items[0].metric == "P95 latency"
    assert parsed.findings[0].evidence_items[0].observed_value == "420мс"
    assert parsed.findings[0].evidence_items[0].threshold == "300мс"
    assert "error rate" in (parsed.recommended_actions[0].details or "")


def test_select_step_profile_candidate_prefers_last_stable_before_unstable():
    segments = [
        {"start": "2024-01-01T10:00:00", "end": "2024-01-01T10:10:00", "level": 240.0, "stable": True},
        {"start": "2024-01-01T10:10:00", "end": "2024-01-01T10:20:00", "level": 225.0, "stable": True},
        {"start": "2024-01-01T10:20:00", "end": "2024-01-01T10:30:00", "level": 260.0, "stable": False},
    ]
    chosen = _select_step_profile_candidate(segments)
    assert chosen is not None
    assert chosen["level"] == 225.0


def test_step_profile_uses_actual_series_cadence_for_sparse_points():
    idx = pd.date_range("2024-01-01T10:00:00Z", periods=8, freq="5min")
    series = pd.Series([190.0, 191.0, 225.0, 226.0, 225.5, 226.2, 160.0, 150.0], index=idx)
    cfg = {
        "step_detection_resample_sec": 20,
        "step_detection_smooth_sec": 90,
        "step_confirm_hold_sec": 120,
        "step_min_step_delta_rps": 8.0,
        "step_min_step_delta_pct": 0.05,
        "step_max_cv": 0.20,
        "step_max_slope_rps_per_min": 2.0,
        "step_max_within_step_drop_pct": 0.10,
        "step_drop_hold_sec": 90,
    }
    stable = _find_stable_peak_step_profile(series, min_stable_minutes=5.0, cfg=cfg)
    assert stable is not None
    assert stable["stable_max"] == pytest.approx(225.625)
    assert stable["method"] == "step_profile"


def test_reconcile_sla_downgrades_stability_resource_failures_to_risk():
    result = _reconcile_sla_for_test_profile(
        {
            "verdict": "Провал",
            "checks": [
                {"name": "p95_latency", "passed": True},
                {"name": "error_rate", "passed": True},
                {"name": "memory_usage", "passed": False},
            ],
            "summary": "SLA verdict: Провал. Нарушено: memory_usage",
        },
        {"mode": "stability"},
    )
    assert result["verdict"] == "Есть риски"
    assert result["test_mode"] == "stability"
    assert result["checks"][2]["category"] == "secondary"


def test_confluence_llm_renderer_matches_structured_ui_format():
    from confluence_manager.update_confluence_template import render_llm_markdown

    md = render_llm_markdown({
        "verdict": "Есть риски",
        "verdict_rationale": "Тест пройден с рисками.\n- Memory выше порога",
        "test_profile": {"mode": "stability", "test_type": "soak", "focus": "Удержание нагрузки"},
        "stability_under_load": {
            "target_rps": 200,
            "actual_rps": 210,
            "sla_summary": "SLA verdict: Есть риски",
        },
        "findings": [
            {
                "id": "f1",
                "summary": "Memory SLA превышен",
                "severity": "warning",
                "component": "k8s-arg-tr01",
                "evidence_summary": "ArangoDB резервирует память",
                "evidence_items": [
                    {"metric": "Memory", "observed_value": "93.4%", "threshold": "80%"},
                ],
            },
        ],
        "recommended_actions": [
            {
                "summary": "Проверить OOM/restarts",
                "details": "Если OOM/restarts нет, считать наблюдением.",
                "for_finding_ids": ["f1"],
            },
        ],
    })

    assert "| Вердикт по тесту | **Есть риски** |" in md
    assert "#### Обоснование вердикта" in md
    assert "#### Стабильность под нагрузкой" in md
    assert "#### Проблемы и рекомендации" in md
    assert "Memory SLA превышен" in md
    assert "Проверить OOM/restarts" in md
    assert "<table" not in md
    assert "<div" not in md
    assert "Доверие" not in md
    assert "Затронутые компоненты" not in md


def test_confluence_placeholder_replacement_uses_block_container():
    from confluence_manager.update_confluence_template import _replace_placeholder_storage

    html, replaced = _replace_placeholder_storage(
        "<p>Before</p><p>$$final_answer$$</p><p>After</p>",
        "$$final_answer$$",
        "<table><tbody><tr><td>OK</td></tr></tbody></table>",
    )
    assert replaced is True
    assert "<p><table" not in html
    assert "<table><tbody><tr><td>OK</td></tr></tbody></table>" in html


def test_perplexity_provider_uses_current_sonar_endpoint_and_models():
    from AI.providers import (
        _extract_perplexity_agent_text,
        _perplexity_api_type,
        _perplexity_model,
        _perplexity_url,
        _strip_think,
    )

    assert _perplexity_url({"api_base_url": "https://api.perplexity.ai"}) == "https://api.perplexity.ai/v1/sonar"
    assert _perplexity_url({"api_base_url": "https://api.perplexity.ai", "model": "openai/gpt-5.4"}) == "https://api.perplexity.ai/v1/agent"
    assert _perplexity_url({"api_base_url": "https://api.perplexity.ai/chat/completions"}) == "https://api.perplexity.ai/chat/completions"
    assert _perplexity_api_type({"model": "sonar-pro"}) == "sonar"
    assert _perplexity_api_type({"model": "openai/gpt-5.4"}) == "agent"
    assert _perplexity_model({"model": "sonar-pro"}) == "sonar-pro"
    assert _perplexity_model({"model": "openai/gpt-5.4"}) == "openai/gpt-5.4"
    assert _extract_perplexity_agent_text({
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "OK"},
                ],
            },
        ],
    }) == "OK"
    assert _strip_think("<think>partial reasoning") == ""


def test_llm_reports_returns_system_context(client, monkeypatch):
    from loadlens_app.blueprints import dashboard

    expected_context = {
        "schema_version": 1,
        "system": {
            "name": "Checkout Platform",
            "domain": "e-commerce",
            "description": "Обработка заказов",
            "test_goal": "Проверить checkout",
        },
        "architecture": {"style": "microservices", "components": [], "dependencies": [], "data_stores": []},
        "load_model": {"entrypoints": [], "critical_user_flows": [], "expected_hotspots": []},
        "operational_context": {"known_constraints": [], "known_risks": [], "normal_degradation_rules": [], "analysis_focus": []},
    }
    rows = [
        (
            "test-run",
            "demo",
            1730455200000,
            1730458800000,
            "final",
            "Успешно",
            '{"verdict":"Успешно","verdict_rationale":"Вердикт подтвержден стабильными метриками.\\n- Существенных отклонений не найдено.","findings":[],"recommended_actions":[]}',
            {
                "verdict": "Успешно",
                "verdict_rationale": "Вердикт подтвержден стабильными метриками.\n- Существенных отклонений не найдено.",
                "findings": [],
                "recommended_actions": [],
            },
            {"judge": {"overall": 0.9}},
            "Успешно",
            {"checks": [], "summary": "ok"},
            expected_context,
            datetime.now(timezone.utc),
        )
    ]
    monkeypatch.setattr(dashboard, "_ts_conn", lambda: RowsConnection(rows))
    monkeypatch.setattr(dashboard, "_ensure_llm_reports_table", lambda conn, cfg: None)
    monkeypatch.setattr(dashboard, "_resolve_services_filter", lambda area: [])

    resp = client.get("/llm_reports", query_string={"run_name": "test-run"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert data[0]["domain"] == "final"
    assert "стабильными метриками" in data[0]["parsed"]["verdict_rationale"]
    assert data[0]["system_context"]["system"]["name"] == "Checkout Platform"


