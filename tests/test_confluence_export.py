import xml.etree.ElementTree as ET

import pytest

from loadlens_app import confluence_export
from loadlens_app.confluence_export import (
    ConfluenceClient,
    _retry_delay_seconds,
    build_page_body,
    html_to_storage,
    load_step_expand,
    render_llm_domain_body,
    sla_criteria_block,
    wrap_ui_tabs,
)


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    """The page builder falls back to stored contexts; these tests must not reach a real database."""

    def no_db():
        raise AssertionError("confluence export tests must not open database connections")

    monkeypatch.setattr(confluence_export, "_ts_conn", no_db)


def _well_formed(storage: str) -> None:
    ET.fromstring(f'<root xmlns:ac="http://atlassian.com/content" xmlns:ri="http://atlassian.com/resource">{storage}</root>')


class _FakeResponse:
    def __init__(self, status_code: int, retry_after: str = "") -> None:
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after else {}
        self.text = "limited"
        self.content = b""


def _client() -> ConfluenceClient:
    return ConfluenceClient({
        "base_url": "https://wiki.local",
        "username": "user",
        "password": "secret",
        "space_key": "LT",
        "parent_page_id": "1",
    })


def test_retry_delay_uses_retry_after_header():
    assert _retry_delay_seconds({"Retry-After": "2.5"}, 1.0) == 2.5
    assert _retry_delay_seconds({}, 1.0) == 1.0


def test_confluence_request_retries_429(monkeypatch):
    client = _client()
    seen: list[int] = []

    def fake_request(method, url, **kwargs):
        seen.append(1)
        return _FakeResponse(429, "0") if len(seen) < 3 else _FakeResponse(200)

    monkeypatch.setattr(client.session, "request", fake_request)
    monkeypatch.setattr("loadlens_app.confluence_export.time.sleep", lambda _seconds: None)
    response = client._request("GET", "/rest/api/content/1", "get page")
    assert response.status_code == 200
    assert len(seen) == 3


def test_html_to_storage_keeps_safe_markup():
    html = '<p>Итог: <b>успех</b></p><a href="https://example.ru">ссылка</a><script>alert(1)</script>'
    converted = html_to_storage(html)
    assert "<b>успех</b>" in converted
    assert 'href="https://example.ru"' in converted
    assert "<script>" not in converted


def test_wrap_ui_tabs_uses_ui_tab_macro():
    xml = wrap_ui_tabs([("Итог", "<p>ok</p>"), ("jvm", "<p>heap</p>")])
    assert 'ac:name="ui-tabs"' in xml
    assert 'ac:name="ui-tab"' in xml
    assert "Итог" in xml
    assert "heap" in xml


def test_build_page_body_includes_report_sections():
    body = build_page_body(
        run_name="nt-run-1",
        service="NSI",
        source_url="http://localhost/reports/NSI/nt-run-1",
        llm_rows=[
            {
                "domain": "final",
                "start_ms": 1710000000000,
                "end_ms": 1710003600000,
                "verdict": "Успешно",
                "parsed": {
                    "verdict": "Успешно",
                    "verdict_rationale": "Целевой RPS достигнут",
                    "peak_performance": {"max_rps": 226.5, "max_time": "12:10", "drop_time": "12:40"},
                    "findings": [{"id": "f1", "summary": "CPU в норме", "severity": "low", "component": "app"}],
                    "recommended_actions": [
                        {
                            "summary": "Оставить текущие лимиты",
                            "details": "Проверить в следующем soak.",
                            "for_finding_ids": ["f1"],
                        }
                    ],
                },
                "system_context": {"system": {"name": "NSI", "description": "Реестр"}},
            },
            {
                "domain": "jvm",
                "parsed": {"verdict": "Есть риски", "findings": [{"summary": "Heap растет"}]},
            },
        ],
        engineer_html="<p>Комментарий инженера</p>",
        charts=[{"domain": "jvm", "title": "Heap used", "filename": "jvm-heap.png"}],
    )
    assert "LoadLens: nt-run-1" in body
    assert "Комментарий инженера" in body
    assert "Целевой RPS достигнут" in body
    assert "226.5" in body
    assert "Максимальный RPS" in body
    assert "Проблемы и рекомендации" in body
    assert "Рекомендации по устранению" in body
    assert "CPU в норме" in body
    assert "Оставить текущие лимиты" in body
    assert "ui-tabs" in body
    assert "jvm-heap.png" in body
    assert "Открыть отчёт в LoadLens" in body
    assert "Контекст тестируемой системы" in body
    assert body.index("SLA-критерии") < body.index("Итоги от инженера")
    assert "SLA-критерии для этого сервиса не заданы" in body


def test_render_llm_domain_keeps_numeric_rps_and_pairs_table():
    body = render_llm_domain_body({
        "domain": "final",
        "verdict": "Успешно",
        "parsed": {
            "verdict": "Успешно",
            "peak_performance": {"max_rps": 180, "max_time": "11:02"},
            "findings": [
                {"id": "f1", "summary": "Очередь Kafka растёт", "severity": "high", "component": "kafka"}
            ],
            "recommended_actions": [
                {"summary": "Увеличить партиции", "details": "С 12 до 24.", "for_finding_ids": ["f1"]}
            ],
        },
    })
    assert "<td>180</td>" in body
    assert "Пиковая производительность" in body
    assert "Очередь Kafka растёт" in body
    assert "Увеличить партиции" in body
    assert "Проблемы" in body
    assert "Ключевые находки" not in body
    other = render_llm_domain_body({
        "domain": "microservices",
        "parsed": {"verdict": "Есть риски", "findings": [{"summary": "Рост времени"}]},
    })
    assert "Пиковая производительность" not in other


def test_load_step_expand_flags_steps():
    body = load_step_expand({"steps": [
        {"index": 1, "rps": 100, "stable": True},
        {"index": 2, "rps": 180, "stable": False, "dip": True},
        {"index": 3, "rps": 150, "stable": False, "after_drop": True, "dip": True},
        {"index": 4, "rps": 400, "stable": False},
    ]})
    assert "<td>1</td>" in body
    assert "2, кратковременная просадка" in body
    assert "3, после просадки" in body
    assert "4, нестабильная" in body


SLA_FINAL = {
    "domain": "final",
    "sla_details": {
        "checks": [
            {"name": "target_rps", "passed": True, "threshold": 200, "actual": 245.678, "message": "RPS на стабильной ступени"},
            {"name": "p95_latency", "passed": False, "threshold": 500, "actual": 812.5, "message": "p95 выше порога <на 62%>"},
            {"name": "cpu_usage", "passed": None, "threshold": 80, "actual": None, "message": ""},
        ],
        "summary": "SLA verdict: Есть риски. p95 latency выше порога",
        "stable_window": {"start": "2024-03-09T16:10:00+00:00", "end": "2024-03-09T16:20:00+00:00", "level": 245.68},
    },
}
STEPS = {"steps": [
    {"index": 1, "rps": 100, "start_iso": "2024-03-09T18:50:00+03:00", "end_iso": "2024-03-09T19:05:00+03:00"},
    {"index": 2, "rps": 230, "start_iso": "2024-03-09T19:05:00+03:00", "end_iso": "2024-03-09T19:25:00+03:00", "selected": True},
]}


def test_sla_criteria_block_has_threshold_actual_and_status():
    body = sla_criteria_block(SLA_FINAL, STEPS)
    _well_formed(body)
    assert "<th>Критерий</th><th>Порог</th><th>Факт</th>" in body
    assert "<td>200</td><td>245.68</td>" in body and "<td>500</td><td>812.5</td>" in body
    assert "<td>80</td><td>—</td>" in body
    assert body.count('ac:name="status"') == 3
    assert "Соблюдён" in body and "Нарушен" in body and "Нет данных" in body
    assert "Green" in body and "Red" in body and "Grey" in body
    assert "Целевой RPS<br />RPS на стабильной ступени" in body
    assert "p95 выше порога &lt;на 62%&gt;" in body
    assert "SLA verdict: Есть риски" in body
    # Window times follow the offset of the step times, like the report page.
    assert "SLA проверено на ступени 2 (19:05–19:25), окно 19:10–19:20, RPS окна 245.68." in body
    assert "среднее всего интервала" in body


def test_sla_criteria_block_for_soak_and_spike_windows():
    hold = {"sla_details": {**SLA_FINAL["sla_details"], "stable_window": {**SLA_FINAL["sla_details"]["stable_window"], "kind": "hold"}}}
    assert "SLA проверено на окне удержания 19:10–19:20, средний RPS 245.68." in sla_criteria_block(hold, STEPS)
    spike = {"sla_details": {**SLA_FINAL["sla_details"], "stable_window": {"kind": "spike", "start": "", "end": "", "level": 900}}}
    assert "в окнах всплесков" in sla_criteria_block(spike, STEPS)


def test_sla_criteria_block_reads_older_reports_and_empty_sla():
    legacy = {"parsed": {"deterministic_sla": {"checks": [{"name": "error_rate", "passed": True, "threshold": 1, "actual": 0.25}]}}}
    body = sla_criteria_block(legacy)
    _well_formed(body)
    assert "Доля ошибок, %" in body and "<td>1</td><td>0.25</td>" in body
    assert "не заданы" in sla_criteria_block({"domain": "final"})

