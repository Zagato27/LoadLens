# SPDX-License-Identifier: Apache-2.0
"""Application logs from Loki: LogQL, answers, the domain pack, the transport, the check and the settings API."""

import copy
import json
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loadlens_app  # noqa: F401  (before update_page users)
from AI import loki_client, loki_logs
from AI import pipeline as pipeline_module
from AI.data_sources import DataSourceConfigError, logs_source_problems, resolve_logs_loki
from AI.loki_logs import (
    LokiConfigError,
    collect_loki_logs,
    count_query,
    counts_from_matrix,
    default_level_filter,
    lines_from_streams,
    detect_line_format,
    lines_query,
    logs_source_type,
    loki_settings,
    message_of,
    split_selector,
    text_level_regex,
    text_message,
)
from loadlens_app import connection_checks, core

CATALOG = {
    "prom": {"title": "Prometheus", "type": "prometheus", "prometheus": {"url": "http://prom:9090"}},
    "graf": {"title": "Grafana", "type": "grafana_proxy",
             "grafana": {"base_url": "http://grafana:3000", "verify_ssl": False, "auth": {"method": "basic", "username": "a", "password": "b"}}},
}
VIA_GRAFANA = {"source": "graf", "datasource_name": "LOKI-DEMO-PREPROD", "selector": '{namespace="apps"}'}
T0 = datetime(2026, 11, 10, 9, 30, tzinfo=timezone.utc).timestamp()


def _logs(**loki):
    return {"enabled": True, "type": "loki", "loki": {"selector": '{namespace="apps"}', **loki}}


# ---- settings and LogQL -----------------------------------------------------------------------------------

def test_type_defaults_to_opensearch():
    assert logs_source_type({}) == "opensearch"
    assert logs_source_type({"type": "OpenSearch"}) == "opensearch"
    assert logs_source_type({"type": " Loki "}) == "loki"
    assert logs_source_type(None) == "opensearch"


def test_selector_split_keeps_quoted_braces_and_line_filters():
    assert split_selector('{app="a}b", namespace=`x}`} |= "{" != "}"') == ('app="a}b", namespace=`x}`', '|= "{" != "}"')
    assert split_selector('{ namespace="apps", }') == ('namespace="apps"', "")
    for bad in ('namespace="apps"', '{namespace="apps"', '{app="}'):
        with pytest.raises(LokiConfigError):
            split_selector(bad)


def test_settings_require_a_selector_and_valid_names():
    with pytest.raises(LokiConfigError):
        loki_settings({"loki": {}})
    with pytest.raises(LokiConfigError):
        loki_settings({"loki": {"selector": "{a=\"b\"}", "service_label": "app-name"}})
    with pytest.raises(LokiConfigError):
        loki_settings({"loki": {"selector": "{a=\"b\"}", "format": "xml"}})
    settings = loki_settings({"loki": {"selector": "{a=\"b\"}", "lines_per_query": 100_000, "request_timeout_sec": True}})
    assert settings.lines_per_query == loki_logs.MAX_LINES_PER_QUERY
    assert settings.timeout_sec == loki_logs.DEFAULT_TIMEOUT_SEC
    assert settings.service_label == "app" and settings.line_format == "auto"
    # Until the lines are read, ``auto`` filters like text.
    assert settings.level_filter == default_level_filter("text", ["ERROR"], "level") and not settings.custom_filter


def test_default_error_filter_follows_format_levels_and_level_field():
    assert default_level_filter("json", ["ERROR"], "level") == '|~ "(?i)(error)" | json | level=~"(?i)(error)"'
    assert default_level_filter("logfmt", ["ERROR", "Fatal"], "lvl") == '|~ "(?i)(error|fatal)" | logfmt | lvl=~"(?i)(error|fatal)"'
    # Text: the regexp parser takes the leftmost level word; regex characters of a level are escaped once for RE2
    # and once more for the LogQL string.
    text = default_level_filter("text", ["C++"], "level")
    assert text.startswith('|~ "(?i)(c\\\\+\\\\+)" | regexp "(?i)(?:^|[\\\\s\\\\[|(]|\\\\x1b\\\\[[0-9;]*m)(?P<ll_level>')
    assert "|c\\\\+\\\\+|" in text and text.endswith(' | ll_level=~"(?i)(c\\\\+\\\\+)"')


def test_queries_use_the_selector_the_filter_and_the_service_label():
    settings = loki_settings({
        "filters": {"error_levels": ["ERROR"], "hostnames": ["user-task", "a.b"]},
        "loki": {"selector": '{namespace="apps"} != "health"', "service_label": "service_name", "level_filter": '| json | level="error"'},
    })
    assert count_query(settings, 60) == (
        'sum by (service_name) (count_over_time({namespace="apps", service_name=~"user-task|a\\\\.b"} != "health" | json | level="error" [60s]))'
    )
    # The sample of one service names it exactly; streams without the label are read with an empty value.
    assert lines_query(settings, 'quo"te') == '{namespace="apps", service_name="quo\\"te"} != "health" | json | level="error"'
    assert lines_query(settings, "").startswith('{namespace="apps", service_name=""}')


# ---- answers ----------------------------------------------------------------------------------------------

def test_matrix_points_count_the_interval_before_them():
    payload = {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": {"app": "orders"}, "values": [[T0 + 60, "3"], [T0 + 120, "0"], [T0 + 180.0, "2.0"]]},
        {"metric": {}, "values": [[T0 + 60, "1"]]},
    ]}}
    counts = counts_from_matrix(payload, "app", 60)
    start = datetime.fromtimestamp(T0, tz=timezone.utc)
    assert counts["orders"] == {start: 3, datetime.fromtimestamp(T0 + 120, tz=timezone.utc): 2}
    assert counts[""] == {start: 1}
    assert counts_from_matrix({"data": {"resultType": "streams", "result": []}}, "app", 60) == {}


def test_stream_lines_and_the_message_of_a_json_line():
    payload = {"data": {"resultType": "streams", "result": [
        {"stream": {"app": "orders"}, "values": [[str(int(T0 * 1e9)), '{"level":"ERROR","log":{"message":"boom 42"}}'], ["bad", "x"]]},
    ]}}
    lines = lines_from_streams(payload)
    assert len(lines) == 1 and lines[0][1] == {"app": "orders"}
    settings = loki_settings({"loki": {"selector": '{a="b"}', "message_field": "log.message"}})
    assert message_of(lines[0][2], settings) == "boom 42"
    assert message_of("plain ERROR text", settings) == "text"
    assert message_of('{"level":"ERROR"}', settings) == '{"level":"ERROR"}'
    # A JSON line gives its message field whatever the format says.
    text = loki_settings({"loki": {"selector": '{a="b"}', "format": "text"}})
    assert message_of('{"message":"m"}', text) == "m"


ESC = "\x1b"
# Synthetic Logback records: ANSI colours, nested exceptions and fabricated identifiers.
COLOURED_ERROR = (
    f"{ESC}[36m12:00:00.100{ESC}[0;39m [1] {ESC}[1;30m[pool-2-thread-41]{ESC}[0;39m {ESC}[1;31mERROR{ESC}[0;39m "
    f"{ESC}[35mcom.example.checkout.OrderListener{ESC}[0;39m - Order creation failed. Key: '00000000-0000-4000-8000-000000000001'.\n"
    "java.util.concurrent.CompletionException: com.example.checkout.HandlerException: Order already exists!\n"
    "\tat java.base/java.util.concurrent.CompletableFuture.encodeThrowable(CompletableFuture.java:315)"
)
WARN_WITH_ERROR_INSIDE = (
    f"{ESC}[36m12:00:02.300{ESC}[0;39m {ESC}[1;30m[processor-executor-thread-7]{ESC}[0;39m {ESC}[31mWARN {ESC}[0;39m "
    f"{ESC}[35mcom.example.checkout.MessageProcessor{ESC}[0;39m - Unhandled exception processing message id 123456\n"
    "org.postgresql.util.PSQLException: ERROR: could not serialize access\n\tat org.jdbi.v3.core.Handle.commit(Handle.java:556)"
)
PLAIN_ERROR = ("2026-11-10T12:00:03.400+03:00 ERROR [Thread-42] com.example.checkout.OrderService - Message processing error. "
               "orderEventId: 00000000-0000-4000-8000-000000000002")


@pytest.mark.parametrize("line, level", [
    (COLOURED_ERROR, "ERROR"),
    (WARN_WITH_ERROR_INSIDE, "WARN"),
    (PLAIN_ERROR, "ERROR"),
    ("2026/11/10 13:51:59 [error] 12#0: *5 connect() failed", "error"),
    ("INFO  c.e.ErrorHandler - request finished with error.", "INFO"),
    ("c.e.ErrorHandler - request finished with error.", None),
    ('{"id":"e5","body":{"task":{"status":"ERROR"}}}', None),
])
def test_text_level_is_the_leftmost_level_word(line, level):
    import re

    match = re.search(text_level_regex(["ERROR"]), line)
    assert (match.group("ll_level") if match else None) == level


def test_text_message_drops_time_thread_and_colours():
    assert text_message(COLOURED_ERROR, ["ERROR"]) == (
        "com.example.checkout.OrderListener - Order creation failed. Key: '00000000-0000-4000-8000-000000000001'. "
        "| HandlerException: Order already exists!"
    )
    # The cause of a wrapped exception, without its package; an exception without a message stays as it is.
    wrapped = ("ERROR x.Repo - failed\n"
               "java.util.concurrent.CompletionException: java.lang.RuntimeException: org.postgresql.util.PSQLException: ERROR: duplicate key")
    assert text_message(wrapped, ["ERROR"]) == "x.Repo - failed | PSQLException: ERROR: duplicate key"
    assert text_message("ERROR x.Repo - failed\njava.lang.NullPointerException", ["ERROR"]) == "x.Repo - failed | java.lang.NullPointerException"
    assert text_message(PLAIN_ERROR, ["ERROR"]).startswith("com.example.checkout.OrderService - Message processing error.")
    # The same error at another time and on another thread is one type.
    other = COLOURED_ERROR.replace("12:00:00.100", "12:00:01.200").replace("thread-41", "thread-7")
    assert text_message(other, ["ERROR"]) == text_message(COLOURED_ERROR, ["ERROR"])
    assert text_message('""" with parameters [1=a]\nnext', ["ERROR"]) == '""" with parameters [1=a]'


def test_line_format_is_what_most_error_lines_are():
    json_line = '{"level":"ERROR","message":"boom"}'
    payload = '{"id":"e5","body":{"task":{"status":"ERROR"}}}'
    assert detect_line_format([json_line, json_line, PLAIN_ERROR], "level") == "json"
    # A printed payload without the level field is not a JSON log record.
    assert detect_line_format([payload, payload, PLAIN_ERROR], "level") == "text"
    assert detect_line_format(['ts=1 level=error msg="x"', 'ts=2 level=error msg="y"'], "level") == "logfmt"
    assert detect_line_format([json_line, PLAIN_ERROR], "level") == "text"
    assert detect_line_format([], "level") == "text"


# ---- the domain ---------------------------------------------------------------------------------------------

class _FakeLoki:
    """Answers the count query with ``counts`` {service: [per minute]} and sample queries with ``lines``."""

    def __init__(self, counts, lines, fail_for=()):
        self.counts, self.lines, self.fail_for = counts, lines, set(fail_for)
        self.calls = []

    def __call__(self, *, query, start_ns, end_ns, step_seconds=None, limit=None, direction=None):
        self.calls.append({"query": query, "start_ns": start_ns, "end_ns": end_ns, "step": step_seconds, "limit": limit, "direction": direction})
        if not step_seconds and 'app="' not in query:
            # The format probe of ``auto``: error lines of every service.
            rows = [[str(start_ns + i), line] for i, line in enumerate(line for lines in self.lines.values() for line in lines)][:limit]
            return {"status": "success", "data": {"resultType": "streams", "result": [{"stream": {}, "values": rows}]}}
        if step_seconds:
            result = [{"metric": ({"app": name} if name else {}), "values": [[T0 + 60 * (i + 1), str(v)] for i, v in enumerate(per_minute)]}
                      for name, per_minute in self.counts.items()]
            return {"status": "success", "data": {"resultType": "matrix", "result": result}}
        service = query.split('app="', 1)[1].split('"', 1)[0]
        if service in self.fail_for:
            raise RuntimeError("Loki answered 500")
        minute = int((start_ns / 1e9 - T0) // 60)
        rows = [[str(start_ns + i), line] for i, line in enumerate(self.lines.get((service, minute), []))]
        return {"status": "success", "data": {"resultType": "streams", "result": [{"stream": {"app": service}, "values": rows}]}}


def _logs_cfg(**aggregation):
    cfg = _logs(lines_per_query=50)
    cfg["aggregation"] = {"context_top_intervals": 2, "context_top_services_per_interval": 1, **aggregation}
    return cfg


def test_totals_are_exact_and_types_come_from_the_busiest_cells():
    fake = _FakeLoki(
        counts={"graphql-federation": [0, 0, 40], "user-task": [5, 0, 3], "": [1, 0, 0]},
        lines={
            ("graphql-federation", 2): [json.dumps({"level": "ERROR", "message": f"upstream timeout, request {1_000_000 + i}"}) for i in range(30)],
            ("user-task", 0): [json.dumps({"level": "ERROR", "message": "Readiness lost"})] * 5,
        },
    )
    out = collect_loki_logs(T0, T0 + 180, _logs_cfg(), "1T", fake)
    probe, count_call, *samples = fake.calls
    # ``auto`` reads error lines of the window first: they are JSON with a level field.
    assert probe["limit"] == loki_logs.DETECT_LINES and probe["step"] is None and 'app="' not in probe["query"]
    assert "| json | level=~" in count_call["query"]
    assert count_call["step"] == 60 and count_call["start_ns"] == int(T0 + 60) * 10**9 and count_call["end_ns"] == int(T0 + 180) * 10**9
    # Two busiest intervals (minute 2: 43 errors, minute 0: 6), one busiest service in each.
    assert [(c["limit"], c["direction"]) for c in samples] == [(50, "backward"), (50, "backward")]
    assert 'app="graphql-federation"' in samples[0]["query"] and 'app="user-task"' in samples[1]["query"]

    pack = out["pack"]
    summary = pack["summary"]
    assert summary["total_errors"] == 49 and summary["source"] == "loki" and summary["sampled_lines"] == 35
    assert summary["error_types_from_sample"] is True and summary["affected_services_count"] == 3
    # Streams without the service label count under "unknown" (the context lists only the top services).
    assert "#### `unknown`" in out["markdown"]
    busiest = next(i for i in pack["intervals"] if i["total_errors"] == 43)
    federation = busiest["services"][0]
    assert federation["hostname"] == "graphql-federation" and federation["total_errors"] == 40
    assert federation["sampled_lines"] == 30 and federation["sample_complete"] is False
    assert federation["error_types"][0]["count"] == 30 and "<" in federation["error_types"][0]["type"]
    first = next(i for i in pack["intervals"] if i["total_errors"] == 6)
    task = first["services"][0]
    assert task["hostname"] == "user-task" and task["sample_complete"] is True and task["error_types"][0]["type"] == "Readiness lost"
    assert json.loads(out["ctx"])["domain"] == "application_logs"
    assert out["markdown"].startswith("# Loki Error Logs Report") and "Error lines (exact): `49`" in out["markdown"]
    assert summary["line_format"] == "json" and summary["line_format_detected"] is True
    assert "Line format: `json` (detected)" in out["markdown"]


@pytest.mark.parametrize("start_offset,end_offset", [(30.25, 150.5), (30.25, 40.75), (0, 30.25), (30.25, 60), (0, 180)])
def test_counts_and_samples_stay_inside_the_test_window(start_offset, end_offset):
    # Include records on internal bucket boundaries and immediately outside both test boundaries.
    offsets = [0, 10, 30.25, 31, 40.75, 59, 60, 61, 89, 90.75, 120, 130, 150.5, 151, 179, 180, 181]
    events = [(int(T0 * 10**9) + round(offset * 10**9), f"ERROR event at {offset}") for offset in offsets]
    calls = []

    def fetch(*, query, start_ns, end_ns, step_seconds=None, limit=None, direction=None):
        calls.append((query, start_ns, end_ns, step_seconds))
        if step_seconds:
            match = re.search(r"\[(\d+)(ms|s)\]", query)
            duration_ns = int(match[1]) * (10**6 if match[2] == "ms" else 10**9)
            values = []
            for point in range(start_ns, end_ns + 1, step_seconds * 10**9):
                value = sum(point - duration_ns < stamp <= point for stamp, _ in events)
                values.append([point / 10**9, str(value)])
            return {"data": {"resultType": "matrix", "result": [{"metric": {"app": "orders"}, "values": values}]}}
        values = [[str(stamp), line] for stamp, line in events if start_ns <= stamp <= end_ns]
        return {"data": {"resultType": "streams", "result": [{"stream": {"app": "orders"}, "values": values[:limit]}]}}

    cfg = _logs_cfg(context_top_intervals=20)
    cfg["loki"]["format"] = "text"
    result = collect_loki_logs(T0 + start_offset, T0 + end_offset, cfg, "1T", fetch)
    expected = sum(start_offset < offset <= end_offset for offset in offsets)
    assert result["pack"]["summary"]["total_errors"] == expected
    assert result["pack"]["summary"]["sampled_lines"] == expected
    test_start = int(T0 * 10**9) + round(start_offset * 10**9)
    test_end = int(T0 * 10**9) + round(end_offset * 10**9)
    for query, start_ns, end_ns, step in calls:
        assert end_ns <= test_end
        if step:
            match = re.search(r"\[(\d+)(ms|s)\]", query)
            duration_ns = int(match[1]) * (10**6 if match[2] == "ms" else 10**9)
            assert start_ns - duration_ns >= test_start
        else:
            assert start_ns > test_start


def test_text_logs_are_counted_by_their_header_level():
    fake = _FakeLoki(counts={"registry": [4]}, lines={("registry", 0): [COLOURED_ERROR, COLOURED_ERROR.replace("thread-41", "thread-9")]})
    out = collect_loki_logs(T0, T0 + 60, _logs_cfg(), "1T", fake)
    count_query_text = fake.calls[1]["query"]
    assert "| regexp " in count_query_text and '| ll_level=~"(?i)(error)"' in count_query_text
    service = out["pack"]["intervals"][0]["services"][0]
    assert out["pack"]["summary"]["line_format"] == "text"
    assert len(service["error_types"]) == 1 and service["error_types"][0]["count"] == 2
    assert service["error_types"][0]["type"].startswith("com.example.checkout.OrderListener - Order creation failed")


@pytest.mark.parametrize("loki", [{"format": "json"}, {"level_filter": '|= "ERROR"'}])
def test_a_set_format_or_filter_reads_no_probe(loki):
    fake = _FakeLoki(counts={"a": [1]}, lines={("a", 0): ["ERROR x"]})
    cfg = _logs_cfg()
    cfg["loki"].update(loki)
    out = collect_loki_logs(T0, T0 + 60, cfg, "1T", fake)
    assert fake.calls[0]["step"] == 60 and "line_format_detected" not in out["pack"]["summary"]


def test_a_failed_sample_query_does_not_drop_the_domain():
    fake = _FakeLoki(counts={"a": [9], "b": [4]}, lines={("b", 0): ["ERROR b"]}, fail_for={"a"})
    out = collect_loki_logs(T0, T0 + 60, _logs_cfg(context_top_services_per_interval=2), "1T", fake)
    summary = out["pack"]["summary"]
    assert summary["total_errors"] == 13 and summary["sample_queries_failed"] == 1 and summary["sampled_lines"] == 1
    services = {s["hostname"]: s for s in out["pack"]["intervals"][0]["services"]}
    assert services["a"]["sampled_lines"] == 0 and services["a"]["error_types"] == []
    assert "1 sample queries failed" in out["markdown"]


def test_no_errors_gives_an_empty_pack():
    out = collect_loki_logs(T0, T0 + 600, _logs_cfg(), "5T", _FakeLoki(counts={}, lines={}))
    assert out["pack"]["summary"]["total_errors"] == 0 and out["pack"]["intervals"] == []
    assert "No ERROR logs" in out["markdown"]


def test_pipeline_reads_the_configured_log_system(monkeypatch):
    seen = {}
    monkeypatch.setattr(loki_client, "loki_fetcher", lambda catalog, logs_cfg: (seen.setdefault("catalog", catalog) and None, "where"))
    monkeypatch.setattr(pipeline_module, "collect_loki_logs", lambda **kw: {"via": "loki", **kw})
    monkeypatch.setattr(pipeline_module, "collect_application_logs", lambda **kw: {"via": "opensearch"})
    out = pipeline_module.collect_logs_domain({"data_sources": CATALOG}, _logs(**VIA_GRAFANA), T0, T0 + 60, "1T")
    assert out["via"] == "loki" and out["resample_interval"] == "1T" and seen["catalog"] == CATALOG
    assert pipeline_module.collect_logs_domain({}, {"enabled": True}, T0, T0 + 60, "1T") == {"via": "opensearch"}


@pytest.mark.parametrize("tables", [False, True])
def test_a_report_with_loki_logs_reaches_the_final_analysis(monkeypatch, tables):
    """The final context takes the logs domain whether or not the Markdown tables are included (a KeyError before)."""
    import pandas as pd

    rps = "LT: RPS"
    cfg = {
        "llm": {"include_markdown_tables_in_context": tables, "self_consistency_k": 1, "max_domain_workers": 1},
        "default_params": {"step": "1m", "resample_interval": "1T"},
        "data_sources": copy.deepcopy(CATALOG),
        "domain_sources": {"default": {"source": "prom"}},
        "queries": {"lt_framework": {"promql_queries": ["sum(rate(x[1m]))"], "label_keys_list": [[]], "labels": [rps]}},
        "sla": {"target_rps": 1, "max_performance_query": rps},
        "logs_source": {**_logs(**VIA_GRAFANA), "aggregation": {"context_top_intervals": 1, "context_top_services_per_interval": 1}},
    }
    lt_df = pd.DataFrame({"sum_all": [10.0, 11.0, 12.0]}, index=pd.date_range(datetime.fromtimestamp(T0, tz=timezone.utc), periods=3, freq="1min"))
    # The pipeline reads this flag from the global settings, not from the report config.
    monkeypatch.setitem(pipeline_module.CONFIG["llm"], "include_markdown_tables_in_context", tables)
    monkeypatch.setattr(pipeline_module, "load_prompt", lambda name, lang=None: "prompt")
    monkeypatch.setattr(pipeline_module, "fetch_and_aggregate_with_label_keys", lambda *args, **kwargs: [lt_df])
    fake = _FakeLoki(counts={"user-task": [0, 7]}, lines={("user-task", 1): ['{"level":"ERROR","message":"Readiness lost"}'] * 7})
    monkeypatch.setattr(loki_client, "loki_fetcher", lambda catalog, logs_cfg: (fake, "graf / LOKI-DEMO-PREPROD"))
    captured = {}

    def fake_llm(user_prompt, data_context, k, return_scores, domain_key):
        captured[domain_key] = data_context
        payload = {"verdict": "pass", "verdict_rationale": "ok", "findings": [], "recommended_actions": []}
        return json.dumps(payload), dict(payload), {}

    monkeypatch.setattr(pipeline_module, "llm_two_pass_self_consistency", fake_llm)
    result = pipeline_module.uploadFromLLM(
        start_ts=T0, end_ts=T0 + 180, save_to_db=False, ef_config=cfg,
        active_domains=["lt_framework", "application_logs"],
    )
    logs_ctx = json.loads(captured["application_logs"])
    assert logs_ctx["summary"]["total_errors"] == 7 and logs_ctx["summary"]["source"] == "loki"
    assert "final" in captured and result["final_parsed"] is not None


def test_a_stored_copy_of_the_opensearch_only_prompt_is_no_override():
    from AI.prompt_store import effective_override, load_prompt

    for lang, (new_task, old_task) in {
        "ru": ("из системы логов (OpenSearch или Loki)", "из OpenSearch"),
        "en": ("from the log system (OpenSearch or Loki)", "from OpenSearch"),
    }.items():
        current = load_prompt("application_logs_prompt.txt", lang)
        lines = current.replace(new_task, old_task).split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith('- summary.source="loki"'))
        end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith("  "))
        old = "\n".join(lines[:start] + lines[end:])
        assert old != current and "Loki" not in old
        assert effective_override(old.replace("\n", "\r\n"), "application_logs") is None
        assert effective_override(old + "\nOwn rule.", "application_logs") is not None


# ---- connection --------------------------------------------------------------------------------------------

def test_resolve_through_grafana_or_directly():
    connection = resolve_logs_loki(CATALOG, _logs(**VIA_GRAFANA))
    assert connection["mode"] == "grafana" and connection["datasource"] == "LOKI-DEMO-PREPROD"
    assert connection["grafana"]["base_url"] == "http://grafana:3000"
    assert connection["grafana"]["loki_datasource"] == {"id": None, "uid": "", "name": "LOKI-DEMO-PREPROD"}
    direct = resolve_logs_loki(CATALOG, _logs(url="http://loki:3100", org_id="demo", username="u", password="p"))
    assert direct == {"mode": "direct", "url": "http://loki:3100", "org_id": "demo", "username": "u", "password": "p", "verify_ssl": True}
    for bad in (_logs(), _logs(source="nope", datasource_name="x"), _logs(source="prom", datasource_name="x"), _logs(source="graf")):
        with pytest.raises(DataSourceConfigError):
            resolve_logs_loki(CATALOG, bad)


def test_catalog_problems_only_for_loki_through_grafana():
    assert logs_source_problems({"type": "opensearch", "loki": {"source": "gone"}}, CATALOG) == []
    assert logs_source_problems(_logs(url="http://loki:3100"), CATALOG) == []
    assert logs_source_problems(_logs(**VIA_GRAFANA), CATALOG) == []
    missing = logs_source_problems(_logs(**VIA_GRAFANA), {}, place="area shop", missing_kind="used")
    assert len(missing) == 1 and "graf" in missing[0] and "area shop" in missing[0]


class _Response:
    def __init__(self, status, payload, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Http:
    """Fake requests: answers by URL path, records method, path and keyword arguments."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _answer(self, method, url, kwargs):
        path = url.split("://", 1)[1].split("/", 1)[1]
        self.calls.append((method, "/" + path, kwargs))
        status, payload = self.routes.get("/" + path, (404, {"message": "Not found"}))
        return _Response(status, payload, text=json.dumps(payload) if payload is not None else "")

    def get(self, url, **kwargs):
        return self._answer("GET", url, kwargs)

    def post(self, url, **kwargs):
        return self._answer("POST", url, kwargs)


def _install(monkeypatch, http):
    for module in (loki_client, pipeline_module):
        monkeypatch.setattr(module.requests, "get", http.get)
        monkeypatch.setattr(module.requests, "post", http.post)


_MATRIX = {"status": "success", "data": {"resultType": "matrix", "result": [{"metric": {"app": "a"}, "values": [[T0, "2"]]}]}}
_DS = {"/api/datasources/name/LOKI-DEMO-PREPROD": (200, {"id": 5, "uid": "loki-uid"})}


def test_grafana_proxy_route_is_read_first(monkeypatch):
    http = _Http({**_DS, "/api/datasources/proxy/uid/loki-uid/loki/api/v1/query_range": (200, _MATRIX)})
    _install(monkeypatch, http)
    fetch, where = loki_client.loki_fetcher(CATALOG, _logs(**VIA_GRAFANA))
    assert where == "graf / LOKI-DEMO-PREPROD"
    assert fetch(query="q", start_ns=1, end_ns=2, step_seconds=60) == _MATRIX
    assert fetch(query="q2", start_ns=1, end_ns=2, limit=10, direction="backward") == _MATRIX
    paths = [path for _, path, _ in http.calls]
    # The datasource is looked up once.
    assert paths.count("/api/datasources/name/LOKI-DEMO-PREPROD") == 1
    params = [kw["params"] for _, path, kw in http.calls if "query_range" in path]
    assert params[0] == {"query": "q", "start": "1", "end": "2", "step": "60"}
    assert params[1] == {"query": "q2", "start": "1", "end": "2", "limit": "10", "direction": "backward"}
    assert http.calls[-1][2]["auth"] == ("a", "b") and http.calls[-1][2]["verify"] is False


def test_closed_proxy_falls_back_to_ds_query_for_counts_and_lines(monkeypatch):
    metric_frames = {"results": {"A": {"frames": [{
        "schema": {"fields": [{"name": "Time", "type": "time"}, {"name": "Value", "type": "number", "labels": {"app": "a"}}]},
        "data": {"values": [[1_700_000_060_000], [3]]},
    }]}}}
    http = _Http({**_DS, "/api/ds/query": (200, metric_frames)})
    _install(monkeypatch, http)
    fetch, _ = loki_client.loki_fetcher(CATALOG, _logs(**VIA_GRAFANA))
    payload = fetch(query="sum by (app) (count_over_time({a=\"b\"} [60s]))", start_ns=1_700_000_000 * 10**9, end_ns=1_700_000_060 * 10**9, step_seconds=60)
    assert payload["data"]["result"] == [{"metric": {"app": "a"}, "values": [[1_700_000_060.0, "3.0"]]}]
    body = http.calls[-1][2]["json"]
    query = body["queries"][0]
    assert query["datasource"] == {"type": "loki", "uid": "loki-uid"} and query["step"] == "60s" and query["queryType"] == "range"
    assert body["from"] == "1700000000000" and body["to"] == "1700000060000"
    # Both proxy routes (uid, then numeric id) were tried before the fallback.
    assert [p for _, p, _ in http.calls if "proxy" in p] == [
        "/api/datasources/proxy/uid/loki-uid/loki/api/v1/query_range", "/api/datasources/proxy/5/loki/api/v1/query_range",
    ]


@pytest.mark.parametrize("frame", [
    # Grafana 8–10: one frame, labels column, Time/Line/tsNs.
    {"schema": {"fields": [{"name": "labels", "type": "other"}, {"name": "Time", "type": "time"}, {"name": "Line", "type": "string"},
                           {"name": "tsNs", "type": "string"}, {"name": "id", "type": "string"}]},
     "data": {"values": [[{"app": "a"}], [1_700_000_000_123], ["ERROR x"], ["1700000000123456789"], ["id1"]]}},
    # Grafana 11 data plane: timestamp/body, labels as JSON text.
    {"schema": {"fields": [{"name": "labels", "type": "other"}, {"name": "timestamp", "type": "time"}, {"name": "body", "type": "string"}]},
     "data": {"values": [['{"app": "a"}'], [1_700_000_000_123], ["ERROR x"]]}},
    # Older backends: labels on the line field.
    {"schema": {"fields": [{"name": "ts", "type": "time"}, {"name": "line", "type": "string", "labels": {"app": "a"}}]},
     "data": {"values": [[1_700_000_000_123], ["ERROR x"]]}},
])
def test_log_frames_become_streams(frame):
    payload = loki_client.streams_from_frames([frame])
    [item] = payload["data"]["result"]
    assert item["stream"] == {"app": "a"} and item["values"][0][1] == "ERROR x"
    assert abs(int(item["values"][0][0]) - 1_700_000_000_123_000_000) < 10**6


def test_direct_loki_sends_tenant_and_auth_and_shows_loki_errors(monkeypatch):
    http = _Http({"/loki/api/v1/query_range": (200, _MATRIX)})
    _install(monkeypatch, http)
    fetch, where = loki_client.loki_fetcher(CATALOG, _logs(url="http://loki:3100/loki/api/v1/query_range", org_id="demo", username="u", password="p", verify_ssl=False))
    assert where == "http://loki:3100"
    fetch(query="q", start_ns=1, end_ns=2, step_seconds=60)
    _, path, kwargs = http.calls[-1]
    assert path == "/loki/api/v1/query_range" and kwargs["headers"] == {"X-Scope-OrgID": "demo"}
    assert kwargs["auth"] == ("u", "p") and kwargs["verify"] is False and kwargs["timeout"] == loki_logs.DEFAULT_TIMEOUT_SEC

    # Loki answers a LogQL error as plain text: the message keeps it.
    monkeypatch.setattr(loki_client.requests, "get", lambda url, **kwargs: _Response(400, None, text="parse error at line 1, col 7: syntax error"))
    with pytest.raises(loki_client.LokiQueryError) as error:
        fetch(query="{", start_ns=1, end_ns=2, step_seconds=60)
    assert "400" in str(error.value) and "parse error at line 1" in str(error.value)


def test_check_dispatches_by_type(monkeypatch):
    monkeypatch.setattr(connection_checks, "check_opensearch", lambda cfg: connection_checks.CheckResult(ok=True, message="os"))
    assert connection_checks.run_check("logs_source", {"enabled": True}).message == "os"

    answers = {"total": 120.0, "per_service": [3.0, 0.0, 2.0]}

    def fake_fetch(*, query, **kwargs):
        if kwargs.get("limit"):
            return {"data": {"resultType": "streams", "result": [{"stream": {}, "values": [["1", PLAIN_ERROR]]}]}}
        if query.startswith("sum(count_over_time"):
            values = [answers["total"]]
        else:
            values = answers["per_service"]
        return {"data": {"result": [{"values": [[0, "0"], [1, str(v)]]} for v in values]}}

    seen = {}
    monkeypatch.setattr(connection_checks, "loki_fetcher", lambda catalog, cfg, timeout=None: (seen.update(catalog=catalog, timeout=timeout) or fake_fetch, "graf / LOKI"))
    ok = connection_checks.run_check("logs_source", _logs(**VIA_GRAFANA), catalog=CATALOG)
    assert ok.ok and "120" in ok.message and "5" in ok.message and ok.details["services"] == 2
    assert ok.details["format"] == "text" and "| regexp " in ok.details["filter"]
    assert seen == {"catalog": CATALOG, "timeout": connection_checks.LOKI_CHECK_TIMEOUT_SEC}
    answers["total"] = 0.0
    empty = connection_checks.run_check("logs_source", _logs(**VIA_GRAFANA), catalog=CATALOG)
    assert not empty.ok and '{namespace="apps"}' in empty.message
    broken = connection_checks.run_check("logs_source", {"type": "loki", "loki": {}}, catalog=CATALOG)
    assert not broken.ok


# ---- settings API ------------------------------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch, tmp_path):
    import update_page

    runtime = tmp_path / "settings_runtime.json"
    runtime.write_text(json.dumps({"per_area": {"shop": {"title": "Shop", "services": {}}}}), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", runtime)
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")
    monkeypatch.setattr(update_page, "_SETTINGS_RUNTIME_PATH", str(runtime))
    base = {"queries": {}, "sla": {}, "per_area": {}, "data_sources": copy.deepcopy(CATALOG),
            "domain_sources": {"default": {"source": "prom"}}, "logs_source": {"enabled": False}}
    for key, value in base.items():
        monkeypatch.setitem(core.CONFIG, key, copy.deepcopy(value))
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return types.SimpleNamespace(client=app.test_client(), runtime=runtime)


def _post(env, section, data, **scope):
    return env.client.post("/config", json={"section": section, "data": data, **scope})


def test_saving_logs_checks_the_grafana_source(env):
    bad = _post(env, "logs_source", _logs(source="nope", datasource_name="x"), area="shop")
    assert bad.status_code == 400 and "nope" in bad.get_json()["error"]
    no_ds = _post(env, "logs_source", _logs(source="graf"), area="shop")
    assert no_ds.status_code == 400 and "graf" in no_ds.get_json()["error"]
    assert _post(env, "logs_source", _logs(**VIA_GRAFANA), area="shop").status_code == 200
    # OpenSearch settings may keep an old Loki block.
    assert _post(env, "logs_source", {"enabled": True, "type": "opensearch", "loki": {"source": "nope"}}).status_code == 200


def test_a_grafana_source_read_by_the_logs_cannot_be_removed(env):
    assert _post(env, "logs_source", _logs(**VIA_GRAFANA), area="shop").status_code == 200
    catalog = copy.deepcopy(CATALOG)
    del catalog["graf"]
    resp = _post(env, "data_sources", catalog)
    assert resp.status_code == 400
    error = resp.get_json()["error"]
    assert "graf" in error and "shop" in error


def test_check_button_resolves_the_grafana_source_from_the_catalog(env, monkeypatch):
    seen = {}

    def fake_check(cfg, catalog=None):
        seen.update(cfg=cfg, catalog=catalog)
        return connection_checks.CheckResult(ok=True, message="loki ok")

    monkeypatch.setattr(connection_checks, "check_logs_source", fake_check)
    resp = env.client.post("/config/test_connection", json={"section": "logs_source", "data": _logs(**VIA_GRAFANA), "area": "shop"})
    assert resp.status_code == 200 and resp.get_json()["message"] == "loki ok"
    assert seen["catalog"]["graf"]["type"] == "grafana_proxy" and seen["cfg"]["loki"]["datasource_name"] == "LOKI-DEMO-PREPROD"
