from loadlens_app.confluence_export import (
    ConfluenceClient,
    _retry_delay_seconds,
    build_page_body,
    html_to_storage,
    load_step_expand,
    render_llm_domain_body,
    wrap_ui_tabs,
)


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
