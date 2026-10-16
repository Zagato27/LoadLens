"""OpenAI-compatible LLM stub for a LoadLens sandbox on 127.0.0.1:5090.

How to run (own console, or in the background; Ctrl+C stops it):

    C:/Python312/python.exe tools/stubs/llm_stub.py                    # 2 s per answer, like a fast model
    C:/Python312/python.exe tools/stubs/llm_stub.py --delay 0.2        # quick functional checks
    C:/Python312/python.exe tools/stubs/llm_stub.py --delay 5 --jitter 0.3 --log %TEMP%/ll_stubs/llm.jsonl

Point the sandbox at it in its settings_runtime.json (``tools/bench_endpoints.py --setup-stubs`` writes it):

    "llm": {"provider": "openai", "openai": {"api_base_url": "http://127.0.0.1:5090/v1", "api_key": "stub",
                                             "model": "stub-model", "verify": false}}

``POST /v1/chat/completions`` (any path ending in ``/chat/completions``) sleeps ``--delay`` seconds
(+- ``--jitter`` share, deterministic per request) and answers by the kind of request:

* report analysis of a domain (the context JSON carries ``"domain": "<key>"``): a valid ``LLMAnalysis``
  JSON (verdict, rationale, qualitative findings without numbers, actions) for that domain; the final
  analysis (context with ``designated_peak_performance``) and the verification pass (``unverified_findings``)
  get the overall answer with ``peak_performance``. Findings carry no numbers on purpose: the programmatic
  check marks them "qualitative", so no verification pass is triggered.
* judge (system prompt of AI/scoring.py ``JUDGE_SYSTEM_PROMPT``): scores for every candidate index found.
* context import (system prompt of loadlens_app/context_import.py): the fixed "Checkout" system context draft
  the e2e test of context import expects (this is what the original %TEMP%/ll_ctx_e2e/llm_stub.py answered).
* connection check of the settings page ("Отвечай одним словом."): ``OK``.

With the defaults of settings.example.py (``self_consistency_k`` 3, a judge per domain) one report with six
domains makes 28 calls: (6 domains + final) x (3 candidates + 1 judge); with ``llm.openai.max_concurrent`` 2
the LLM phase lasts about 14 x ``--delay``.

Other endpoints: ``GET /v1/models``; ``GET /_stub/stats`` (calls by kind and domain, delay settings) and
``POST /_stub/reset`` - ``tools/bench_endpoints.py --during-job`` reads them to report the number of calls.
``--fail-every N`` answers every N-th call with 500 (retry paths). ``--log FILE`` appends the request bodies
as JSON lines (off by default; the prompts are large). Listens on 127.0.0.1/localhost only. Standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Sequence
from urllib.parse import urlsplit

DEFAULT_PORT = 5090
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MODEL = "stub-model"

# Context import (loadlens_app/context_import.py): the draft the context-import e2e test expects.
CONTEXT_DRAFT = {
    "system": {"name": "Checkout", "domain": "интернет-магазин", "description": "Оформление и оплата заказов", "test_goal": "Подтвердить 300 RPS оформления заказов"},
    "architecture": {
        "style": "microservices",
        "components": [
            {"id": "checkout-api", "name": "checkout-api", "role": "оформление заказа", "criticality": "high", "technologies": ["Java", "Spring Boot"]},
            {"id": "payments", "name": "payments", "role": "платежи", "criticality": "high", "technologies": ["Go"]},
        ],
        "dependencies": [{"from": "checkout-api", "to": "payments", "kind": "sync", "purpose": "оплата заказа"}],
        "data_stores": [{"id": "orders-db", "type": "PostgreSQL", "purpose": "заказы", "used_by": ["checkout-api"]}],
    },
    "load_model": {
        "entrypoints": [{"id": "post-order", "name": "POST /orders", "kind": "http", "business_priority": "high"}],
        "critical_user_flows": [{"id": "buy", "name": "Покупка", "steps": ["корзина", "оформление", "оплата"], "success_signals": ["заказ создан"]}],
        "expected_hotspots": ["пул соединений к orders-db"],
    },
    "operational_context": {
        "known_constraints": ["Пул соединений к базе — 20"],
        "known_risks": ["Пул соединений к базе исчерпывается под нагрузкой"],
        "normal_degradation_rules": [],
        "analysis_focus": ["Время ответа payments"],
    },
}

# Per domain: verdict, component, what was looked at, the finding and the action. No digits anywhere.
DOMAINS: dict[str, dict[str, str]] = {
    "jvm": {"title": "JVM", "verdict": "Успешно", "metric": "jvm_memory_used_bytes",
            "finding": "Куча и паузы сборщика мусора стабильны на всех ступенях нагрузки",
            "action": "Оставить текущие настройки памяти JVM"},
    "database": {"title": "База данных", "verdict": "Успешно", "metric": "hikaricp_connections_active",
                 "finding": "Пул соединений не исчерпывается, ожидающих потоков нет",
                 "action": "Наблюдать за пулом соединений на следующих прогонах"},
    "kafka": {"title": "Kafka", "verdict": "Успешно", "metric": "kafka_consumergroup_lag",
              "finding": "Лаг потребителей не растёт, сообщения обрабатываются без задержек",
              "action": "Изменений не требуется"},
    "microservices": {"title": "Микросервисы", "verdict": "Успешно", "metric": "http_server_requests_seconds_count",
                      "finding": "Время ответа сервисов растёт вместе с нагрузкой без скачков ошибок",
                      "action": "Сравнить время ответа с прошлым прогоном"},
    "hard_resources": {"title": "Ресурсы", "verdict": "Есть риски", "metric": "node_cpu_seconds_total",
                       "finding": "Загрузка процессора узлов растёт ступенями вслед за нагрузкой",
                       "action": "Проверить запас по процессору перед повышением нагрузки"},
    "lt_framework": {"title": "Инструмент нагрузки", "verdict": "Успешно", "metric": "http_reqs",
                     "finding": "Нагрузка подавалась ступенями, ошибок генератора не видно",
                     "action": "Сохранить профиль нагрузки для следующего прогона"},
    "lt_generators": {"title": "Генераторы нагрузки", "verdict": "Успешно", "metric": "node_cpu_seconds_total",
                      "finding": "Генераторы не упирались в процессор и сеть",
                      "action": "Изменений не требуется"},
    "application_logs": {"title": "Логи приложений", "verdict": "Успешно", "metric": "error_count",
                         "finding": "Новых типов ошибок в логах не появилось",
                         "action": "Изменений не требуется"},
    "final": {"title": "Итог", "verdict": "Успешно", "metric": "http_reqs",
              "finding": "Система держала нагрузку на всех ступенях без деградации",
              "action": "Принять результат теста и сохранить его как базовый"},
}
_DOMAIN_RE = re.compile(r'"domain"\s*:\s*"([a-z_]+)"')
_INDEX_RE = re.compile(r'"index"\s*:\s*(\d+)')


def classify(body: dict) -> tuple[str, str]:
    """``(kind, domain)`` of a chat completion request: kind is judge, context_import, ping or analysis."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system = " ".join(str(m.get("content") or "") for m in messages if isinstance(m, dict) and m.get("role") == "system")
    user = "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict) and m.get("role") != "system")
    if "судьёй" in system or "судьей" in system:
        return "judge", ""
    if "переносите описание" in system:
        return "context_import", ""
    if system.strip().startswith("Отвечай одним словом"):
        return "ping", ""
    if "unverified_findings" in user:
        return "analysis", "final"
    if "designated_peak_performance" in user:
        return "analysis", "final"
    match = _DOMAIN_RE.search(user)
    return "analysis", (match.group(1) if match else "unknown")


def analysis_answer(domain: str) -> dict:
    """A valid LLMAnalysis payload for ``domain`` (AI/scoring.py), with qualitative findings only."""
    spec = DOMAINS.get(domain) or {**DOMAINS["final"], "title": domain or "Домен"}
    component = domain if domain in DOMAINS and domain != "final" else "system"
    answer: dict[str, Any] = {
        "verdict": spec["verdict"],
        "verdict_rationale": f"{spec['title']}: {spec['finding'].lower()}.",
        "confidence": 0.8,
        "findings": [{
            "id": "f1",
            "summary": spec["finding"],
            "severity": "medium" if spec["verdict"] == "Есть риски" else "low",
            "component": component,
            "start_time": "",
            "end_time": "",
            "peak_time": "",
            "evidence_summary": f"Ряды {spec['metric']} без резких скачков",
            "evidence_items": [{"metric": spec["metric"], "observed_value": "стабильно", "threshold": "", "note": "ответ заглушки"}],
        }],
        "recommended_actions": [{
            "summary": spec["action"],
            "details": "Ответ заглушки модели для песочницы LoadLens: действие не требует изменений в системе.",
            "priority": "low",
            "affected_components": [component],
            "for_finding_ids": ["f1"],
        }],
        "affected_components": [component],
    }
    if domain in ("lt_framework", "final"):
        answer["peak_performance"] = {"max_rps": 1000.0, "method": "stub", "note": "ответ заглушки"}
    return answer


def judge_answer(user_text: str) -> list[dict]:
    """Equal scores for every candidate index of the judge prompt (at least three)."""
    indexes = [int(i) for i in _INDEX_RE.findall(user_text or "")]
    count = max(indexes) + 1 if indexes else 3
    rubric = {"evidence_grounding": 0.8, "issue_coverage": 0.8, "specificity": 0.7, "sla_alignment": 0.8, "actionability": 0.7}
    return [{"index": i, "rubric": rubric, "overall": 0.78} for i in range(min(count, 20))]


def answer_text(body: dict) -> tuple[str, str, str]:
    """``(content, kind, domain)`` for a request body."""
    kind, domain = classify(body)
    if kind == "judge":
        messages = body.get("messages") or []
        user = "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict) and m.get("role") != "system")
        return json.dumps({"scores": judge_answer(user)}, ensure_ascii=False), kind, domain
    if kind == "context_import":
        return "```json\n" + json.dumps(CONTEXT_DRAFT, ensure_ascii=False) + "\n```", kind, domain
    if kind == "ping":
        return "OK", kind, domain
    return json.dumps(analysis_answer(domain), ensure_ascii=False), kind, domain


def call_delay(body_bytes: bytes, delay: float, jitter: float) -> float:
    """Seconds to wait for this request: ``delay`` +- ``jitter`` share, deterministic per request body."""
    if delay <= 0:
        return 0.0
    if jitter <= 0:
        return delay
    fraction = int.from_bytes(hashlib.sha256(body_bytes).digest()[:4], "big") / 0xFFFFFFFF
    return max(0.0, delay * (1.0 + jitter * (2.0 * fraction - 1.0)))


def completion(content: str, prompt_chars: int) -> dict:
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": max(1, prompt_chars // 4), "completion_tokens": max(1, len(content) // 4),
                  "total_tokens": max(1, prompt_chars // 4) + max(1, len(content) // 4)},
    }


# ---- HTTP -------------------------------------------------------------------------------------------------


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.calls = 0
            self.by_kind: dict[str, int] = {}
            self.by_domain: dict[str, int] = {}
            self.in_flight = 0
            self.max_in_flight = 0
            self.failed = 0
            self.started = time.time()

    def begin(self, kind: str, domain: str) -> int:
        with self.lock:
            self.calls += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
            if domain:
                self.by_domain[domain] = self.by_domain.get(domain, 0) + 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            return self.calls

    def end(self, failed: bool = False) -> None:
        with self.lock:
            self.in_flight -= 1
            self.failed += int(failed)

    def snapshot(self, options: argparse.Namespace) -> dict:
        with self.lock:
            return {
                "calls": self.calls, "by_kind": dict(self.by_kind), "by_domain": dict(self.by_domain),
                "in_flight": self.in_flight, "max_in_flight": self.max_in_flight, "failed": self.failed,
                "delay_sec": options.delay, "jitter": options.jitter, "since": self.started,
            }


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMStub/1"
    options: argparse.Namespace
    stats: Stats
    log_lock = threading.Lock()

    def _send(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (http.server API)
        path = urlsplit(self.path).path.rstrip("/")
        if path.endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "loadlens-stub"}]})
        if path == "/_stub/stats":
            return self._send(200, self.stats.snapshot(self.options))
        return self._send(404, {"error": {"message": f"unknown path {path}"}})

    def do_POST(self):  # noqa: N802 (http.server API)
        path = urlsplit(self.path).path.rstrip("/")
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if path == "/_stub/reset":
            self.stats.reset()
            return self._send(200, {"status": "reset"})
        if not path.endswith("/chat/completions"):
            return self._send(404, {"error": {"message": f"unknown path {path}"}})
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return self._send(400, {"error": {"message": "invalid JSON body"}})
        content, kind, domain = answer_text(body if isinstance(body, dict) else {})
        number = self.stats.begin(kind, domain)
        failed = False
        try:
            if self.options.log:
                with self.log_lock, open(self.options.log, "a", encoding="utf-8") as log:
                    log.write(json.dumps({"n": number, "kind": kind, "domain": domain, "body": body}, ensure_ascii=False) + "\n")
            time.sleep(call_delay(raw, self.options.delay, self.options.jitter))
            if self.options.fail_every and number % self.options.fail_every == 0:
                failed = True
                return self._send(500, {"error": {"message": "stub failure (--fail-every)", "type": "server_error"}})
            prompt_chars = sum(len(str(m.get("content") or "")) for m in (body.get("messages") or []) if isinstance(m, dict))
            return self._send(200, completion(content, prompt_chars))
        finally:
            self.stats.end(failed)

    def log_message(self, *args):
        pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible LLM stub for the LoadLens sandbox (see the module docstring).")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (local only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--delay", type=float, default=2.0, help="seconds per answer (model latency)")
    parser.add_argument("--jitter", type=float, default=0.0, help="+- share of --delay, deterministic per request")
    parser.add_argument("--fail-every", type=int, default=0, help="answer every N-th call with 500 (0: never)")
    parser.add_argument("--log", help="append the request bodies to this file as JSON lines")
    args = parser.parse_args(argv)
    if args.host not in LOCAL_HOSTS:
        parser.error("the stub listens on 127.0.0.1/localhost only")
    if args.delay < 0 or not 0 <= args.jitter <= 1 or args.fail_every < 0:
        parser.error("--delay and --fail-every must not be negative, --jitter is within 0..1")
    return args


def make_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"options": args, "stats": Stats()})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    return server


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    server = make_server(args)
    print(f"llm stub on http://{args.host}:{args.port}/v1 (delay {args.delay} s +- {args.jitter:.0%})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
