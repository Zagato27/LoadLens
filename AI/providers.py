# SPDX-License-Identifier: Apache-2.0
import contextvars
import os
import json
import random
import time
import threading
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Type, TypeVar
import requests
from urllib.parse import urlparse

StructuredT = TypeVar("StructuredT")

from settings import CONFIG
from AI.prompt_store import load_prompt, normalize_lang, resolve_lang
from AI.prompt_store import text as prompt_text
from AI.i18n import t


logger = logging.getLogger(__name__)

# Provider keys from the server environment (OPENAI_API_KEY ...) belong to the saved provider address.
# A connection check of an unsaved form that points elsewhere runs without them, see without_env_keys().
_ENV_KEYS_ALLOWED: contextvars.ContextVar[bool] = contextvars.ContextVar("llm_env_keys_allowed", default=True)


@contextmanager
def without_env_keys() -> Iterator[None]:
    token = _ENV_KEYS_ALLOWED.set(False)
    try:
        yield
    finally:
        _ENV_KEYS_ALLOWED.reset(token)


def _env_key(*names: str) -> str:
    if not _ENV_KEYS_ALLOWED.get():
        return ""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""
_llm_provider_name = (CONFIG.get("llm", {}) or {}).get("provider", "perplexity").lower()
_llm_provider_cfg = (CONFIG.get("llm", {}) or {}).get(_llm_provider_name, {})
_llm_semaphore = threading.Semaphore(int((_llm_provider_cfg or {}).get("max_concurrent") or 4))
_llm_rate_lock = threading.Lock()
_llm_last_request_ts = 0.0
_llm_next_allowed_ts = 0.0
_gigachat_lock = threading.Lock()
_gigachat_clients: dict[str, object] = {}
_usage_lock = threading.Lock()
_usage_by_domain: Dict[str, Dict[str, Any]] = {}
_usage_tls = threading.local()
# (configured language_preference, report language) pairs already warned about in this run.
_language_warnings: set[tuple[str, str]] = set()
SUPPORTED_PROVIDERS = ("perplexity", "openai", "anthropic", "gigachat")
DEFAULT_MAX_ATTEMPTS = 3
CONTEXT_MARGIN_TOKENS = 4_000
MIN_INPUT_TOKENS = 1_000
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})


class LLMOutputTruncated(RuntimeError):
    """The provider stopped because the output budget was exhausted."""

    def __init__(self, provider: str, max_tokens: int, output_tokens: Optional[int] = None):
        self.provider = provider
        self.max_tokens = int(max_tokens)
        self.output_tokens = output_tokens
        # Rendered in the current locale: the pipeline runs inside with_locale(report_language).
        if output_tokens is not None:
            text = t("llm.error.output_truncated_with_count", provider=provider,
                     max_tokens=self.max_tokens, output_tokens=output_tokens)
        else:
            text = t("llm.error.output_truncated", provider=provider, max_tokens=self.max_tokens)
        super().__init__(text)


class LLMEmptyAnswer(RuntimeError):
    """The provider returned no user-visible text (often only a thinking trace)."""

    def __init__(self, provider: str, max_tokens: int = 0):
        self.provider = provider
        self.max_tokens = int(max_tokens or 0)
        super().__init__(t("llm.error.empty_answer", provider=provider))


class LLMBudgetError(RuntimeError):
    """The requested output budget leaves no room for the prompt inside the context window."""
_PERPLEXITY_SONAR_MODELS = {
    "sonar",
    "sonar-pro",
    "sonar-deep-research",
    "sonar-reasoning-pro",
}
_PERPLEXITY_AGENT_MODEL_PREFIXES = (
    "perplexity/",
    "openai/",
    "anthropic/",
    "google/",
    "nvidia/",
    "xai/",
)


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None


def _extract_token_counts(response: Any) -> tuple[Optional[int], Optional[int]]:
    """Reads prompt/completion tokens from a provider payload; never invents numbers."""
    usage: Any = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage") or response.get("token_usage")
    if usage is None:
        meta = getattr(response, "response_metadata", None)
        if isinstance(meta, dict):
            usage = meta.get("token_usage") or meta.get("usage") or meta.get("tokenUsage")
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        prompt = _first_int(usage.get("input_tokens"), usage.get("prompt_tokens"), usage.get("promptTokens"))
        completion = _first_int(usage.get("output_tokens"), usage.get("completion_tokens"), usage.get("completionTokens"))
        return prompt, completion
    prompt = _first_int(getattr(usage, "input_tokens", None), getattr(usage, "prompt_tokens", None))
    completion = _first_int(getattr(usage, "output_tokens", None), getattr(usage, "completion_tokens", None))
    return prompt, completion


def _usage_cost(prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    pricing = ((CONFIG.get("llm") or {}).get("gigachat") or {}).get("pricing") or {}
    prompt_rate = float(pricing["prompt_per_1k"]) if pricing.get("prompt_per_1k") is not None else None
    completion_rate = float(pricing["completion_per_1k"]) if pricing.get("completion_per_1k") is not None else None
    if prompt_rate is None and completion_rate is None:
        return None
    if prompt_tokens is None and completion_tokens is None:
        return None
    total = 0.0
    if prompt_rate is not None and prompt_tokens is not None:
        total += prompt_rate * (prompt_tokens / 1000.0)
    if completion_rate is not None and completion_tokens is not None:
        total += completion_rate * (completion_tokens / 1000.0)
    return round(total, 6)


def _empty_usage_bucket() -> Dict[str, Any]:
    return {"calls": 0, "prompt_tokens": None, "completion_tokens": None, "elapsed_ms": 0, "cost": None}


def reset_usage() -> None:
    """Clears accumulated token usage and the per-run warnings (call at the start of a report run)."""
    with _usage_lock:
        _usage_by_domain.clear()
        _language_warnings.clear()


def current_usage_domain() -> str:
    return getattr(_usage_tls, "domain", None) or "unscoped"


@contextmanager
def usage_domain(domain_key: str) -> Iterator[str]:
    """Attributes LLM calls on this thread to ``domain_key`` until the block exits."""
    previous = getattr(_usage_tls, "domain", None)
    _usage_tls.domain = str(domain_key or "unscoped")
    try:
        yield _usage_tls.domain
    finally:
        _usage_tls.domain = previous


def record_llm_usage(
    *,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    elapsed_ms: int = 0,
    domain_key: Optional[str] = None,
) -> None:
    domain = str(domain_key or current_usage_domain())
    with _usage_lock:
        bucket = _usage_by_domain.setdefault(domain, _empty_usage_bucket())
        bucket["calls"] = int(bucket.get("calls") or 0) + 1
        bucket["elapsed_ms"] = int(bucket.get("elapsed_ms") or 0) + max(0, int(elapsed_ms or 0))
        if prompt_tokens is not None:
            bucket["prompt_tokens"] = int(bucket.get("prompt_tokens") or 0) + int(prompt_tokens)
        if completion_tokens is not None:
            bucket["completion_tokens"] = int(bucket.get("completion_tokens") or 0) + int(completion_tokens)
        bucket["cost"] = _usage_cost(bucket.get("prompt_tokens"), bucket.get("completion_tokens"))


def snapshot_usage() -> Dict[str, Dict[str, Any]]:
    """Copy of per-domain usage plus a ``run_total`` roll-up."""
    with _usage_lock:
        by_domain = {key: dict(value) for key, value in _usage_by_domain.items()}
    total = _empty_usage_bucket()
    known_prompt = False
    known_completion = False
    cost_sum = 0.0
    has_cost = False
    for bucket in by_domain.values():
        total["calls"] += int(bucket.get("calls") or 0)
        total["elapsed_ms"] += int(bucket.get("elapsed_ms") or 0)
        if bucket.get("prompt_tokens") is not None:
            total["prompt_tokens"] = int(total["prompt_tokens"] or 0) + int(bucket["prompt_tokens"])
            known_prompt = True
        if bucket.get("completion_tokens") is not None:
            total["completion_tokens"] = int(total["completion_tokens"] or 0) + int(bucket["completion_tokens"])
            known_completion = True
        if bucket.get("cost") is not None:
            cost_sum += float(bucket["cost"])
            has_cost = True
    if not known_prompt:
        total["prompt_tokens"] = None
    if not known_completion:
        total["completion_tokens"] = None
    total["cost"] = round(cost_sum, 6) if has_cost else None
    by_domain["run_total"] = total
    return by_domain


def attach_usage_to_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Writes ``usage`` onto each domain score dict and a run total onto ``final``."""
    usage = snapshot_usage()
    out: Dict[str, Any] = dict(scores or {})
    for domain, payload in list(out.items()):
        block = dict(payload) if isinstance(payload, dict) else {}
        block["usage"] = usage.get(domain) or block.get("usage") or _empty_usage_bucket()
        out[domain] = block
    final_block = dict(out.get("final") or {}) if isinstance(out.get("final"), dict) else {}
    final_block["usage"] = usage.get("final") or final_block.get("usage") or _empty_usage_bucket()
    final_block["run_usage"] = usage.get("run_total") or _empty_usage_bucket()
    out["final"] = final_block
    return out


def _normalize_llm_base_url(raw_url: str | None) -> str:
    base = (raw_url or "https://api.perplexity.ai").strip()
    if not base:
        return "https://api.perplexity.ai"
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _perplexity_api_type(pcfg: dict) -> str:
    explicit = str((pcfg or {}).get("api_type") or (pcfg or {}).get("api") or "").strip().lower()
    if explicit in {"agent", "sonar"}:
        return explicit
    endpoint = str((pcfg or {}).get("endpoint_path") or "").strip().lower()
    if endpoint.endswith("/v1/agent") or endpoint.endswith("/agent") or "agent" in endpoint:
        return "agent"
    model = str((pcfg or {}).get("model") or "").strip().lower()
    if model.startswith(_PERPLEXITY_AGENT_MODEL_PREFIXES):
        return "agent"
    return "sonar"


def _perplexity_url(pcfg: dict) -> str:
    raw_url = (pcfg.get("base_url") or pcfg.get("api_base_url") or "https://api.perplexity.ai").strip()
    base = raw_url.rstrip("/") or "https://api.perplexity.ai"
    if base.endswith("/chat/completions") or base.endswith("/v1/sonar") or base.endswith("/v1/agent"):
        return base
    default_endpoint = "/v1/agent" if _perplexity_api_type(pcfg) == "agent" else "/v1/sonar"
    endpoint = str(pcfg.get("endpoint_path") or default_endpoint).strip() or default_endpoint
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return f"{base}{endpoint}"


def _perplexity_model(pcfg: dict) -> str:
    api_type = _perplexity_api_type(pcfg)
    default_model = "perplexity/sonar" if api_type == "agent" else "sonar-reasoning-pro"
    model = str((pcfg or {}).get("model") or default_model).strip()
    if api_type == "agent":
        if model.startswith(_PERPLEXITY_AGENT_MODEL_PREFIXES):
            return model
        if model in _PERPLEXITY_SONAR_MODELS:
            return "perplexity/sonar"
        logger.warning(
            "Unsupported Perplexity Agent API model '%s'; using '%s'. "
            "Agent API models should use provider prefixes, e.g. openai/gpt-5.4.",
            model,
            default_model,
        )
        return default_model
    if model in _PERPLEXITY_SONAR_MODELS:
        return model
    fallback = "sonar-reasoning-pro"
    logger.warning(
        "Unsupported Perplexity model '%s'; using '%s'. Supported models: %s",
        model,
        fallback,
        ", ".join(sorted(_PERPLEXITY_SONAR_MODELS)),
    )
    return fallback


# Proxies and CA bundles are passed to every request explicitly. They used to be written into
# os.environ on the first call, which sent all later HTTP traffic of the process (Grafana,
# Confluence, OpenSearch) through whatever proxy that first, possibly unsaved, configuration named.
def _tls_verify(pcfg: dict) -> bool | str:
    """``verify`` for requests: ``False``, ``True`` or the path of a CA bundle."""
    pcfg = pcfg or {}
    if pcfg.get("insecure_skip_verify"):
        return False
    raw = pcfg.get("verify", True)
    if isinstance(raw, str):
        return raw.strip() or True
    if raw is False:
        return False
    return str(pcfg.get("ca_bundle") or "").strip() or True


def _request_proxies(pcfg: dict) -> dict | None:
    """Proxies of this provider for one request; ``None`` when none are set."""
    raw = (pcfg or {}).get("proxies") or {}
    proxies = {str(k).lower(): str(v).strip() for k, v in raw.items() if isinstance(v, str) and v.strip()} if isinstance(raw, dict) else {}
    return proxies or None


def _strip_think(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    try:
        import re
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        return re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)
    except Exception:
        return text


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default: int) -> int:
    """Integer setting; ``None`` (an empty number field of the settings form) or junk gives ``default``."""
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _effective_max_tokens(provider_cfg: dict, default: int, default_cap: int) -> int:
    override = _optional_int((provider_cfg or {}).get("_output_tokens_override"))
    if override is not None and override > 0:
        return override
    gen = (provider_cfg or {}).get("generation") or {}
    requested = _safe_int(gen.get("max_tokens"), default)
    if requested <= 0:
        requested = default
    cap = _first_int((provider_cfg or {}).get("max_tokens_cap"), os.getenv("LOADLENS_LLM_MAX_TOKENS_CAP"), default_cap)
    if cap is not None and cap > 0 and requested > cap:
        return cap
    return requested


def _wait_llm_slot(pcfg: dict) -> None:
    """Глобальный pacing между LLM-вызовами для снижения burst-нагрузки."""
    global _llm_last_request_ts
    min_interval = max(0.0, _safe_float((pcfg or {}).get("request_min_interval_sec", 0.0), 0.0))
    if min_interval <= 0:
        return
    while True:
        with _llm_rate_lock:
            now = time.monotonic()
            target = max(_llm_next_allowed_ts, _llm_last_request_ts + min_interval)
            wait_sec = target - now
            if wait_sec <= 0:
                _llm_last_request_ts = now
                return
        time.sleep(min(wait_sec, 1.0))


def _apply_llm_cooldown(seconds: float) -> None:
    """Сдвигает общий cooldown для всех потоков после 429/перегруза."""
    global _llm_next_allowed_ts
    sec = max(0.0, float(seconds or 0.0))
    if sec <= 0:
        return
    with _llm_rate_lock:
        _llm_next_allowed_ts = max(_llm_next_allowed_ts, time.monotonic() + sec)


def _messages_to_agent_input(messages: list[dict]) -> str:
    parts: list[str] = []
    role_labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        role = str(message.get("role") or "user").strip().lower()
        parts.append(f"{role_labels.get(role, role.title() or 'User')}:\n{content}")
    return "\n\n".join(parts)


def _extract_perplexity_agent_text(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    output_items = data.get("output")
    texts: list[str] = []
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if isinstance(content_items, list):
                for content in content_items:
                    if isinstance(content, dict):
                        text = content.get("text")
                        if isinstance(text, str) and text.strip():
                            texts.append(text)
                    elif isinstance(content, str) and content.strip():
                        texts.append(content)
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return "\n".join(texts)


def _perplexity_call(messages: list[dict], pcfg: dict) -> str:
    disable_web = bool(pcfg.get("disable_web_search", True))
    model = _perplexity_model(pcfg)
    api_type = _perplexity_api_type(pcfg)
    gen = (pcfg.get("generation") or {})
    url = _perplexity_url(pcfg)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {_env_key('PPLX_API_KEY', 'PERPLEXITY_API_KEY') or pcfg.get('api_key', '')}",
    }

    proxies = _request_proxies(pcfg)
    verify = _tls_verify(pcfg)

    req_max_tokens = _effective_max_tokens(pcfg, default=1200, default_cap=32768)
    if api_type == "agent":
        payload = {
            "model": model,
            "input": _messages_to_agent_input(messages),
            "max_output_tokens": req_max_tokens,
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": _safe_float(gen.get("temperature"), 0.2),
            "top_p": _safe_float(gen.get("top_p"), 0.9),
            "max_tokens": req_max_tokens,
        }
        if disable_web:
            payload["disable_search"] = True
        for key in (
            "disable_search",
            "enable_search_classifier",
            "search_mode",
            "search_domain_filter",
            "search_recency_filter",
            "return_images",
            "return_related_questions",
            "web_search_options",
            "reasoning_effort",
            "language_preference",
        ):
            if key in pcfg and pcfg.get(key) is not None:
                payload[key] = pcfg.get(key)

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=_safe_int(pcfg.get("request_timeout_sec"), 120),
        verify=verify,
        proxies=proxies,
    )
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") if isinstance(data, dict) else None
    output_tokens = _optional_int((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")) if isinstance(usage, dict) else None
    if api_type == "agent":
        return _ensure_text_complete("perplexity", _extract_perplexity_agent_text(data), _response_finish_reason(data), req_max_tokens, output_tokens)
    choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    return _ensure_text_complete("perplexity", str(content or ""), _response_finish_reason(data), req_max_tokens, output_tokens)


def _openai_call(messages: list[dict], pcfg: dict) -> str:
    base_url = (_normalize_llm_base_url(pcfg.get("api_base_url") or pcfg.get("base_url")))
    url = f"{base_url}/chat/completions"
    gen = (pcfg.get("generation") or {})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {_env_key('OPENAI_API_KEY') or pcfg.get('api_key', '')}",
    }
    proxies = _request_proxies(pcfg)
    verify = _tls_verify(pcfg)
    req_max_tokens = _effective_max_tokens(pcfg, default=1200, default_cap=32768)
    payload = {
        "model": pcfg.get("model", "gpt-4o-mini"),
        "messages": messages,
        "temperature": _safe_float(gen.get("temperature"), 0.2),
        "top_p": _safe_float(gen.get("top_p"), 0.9),
        "max_tokens": req_max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=_safe_int(pcfg.get("request_timeout_sec"), 120), verify=verify, proxies=proxies)
    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    usage = data.get("usage") if isinstance(data, dict) else None
    output_tokens = _optional_int((usage or {}).get("completion_tokens")) if isinstance(usage, dict) else None
    return _ensure_text_complete("openai", str(content or ""), _response_finish_reason(data), req_max_tokens, output_tokens)


def _estimate_tokens(text: str) -> int:
    """Upper bound on tokens. Cyrillic JSON is close to one token per character."""
    return max(1, len(text or ""))


def _context_window_tokens(pcfg: dict) -> int:
    """Input + max_tokens must stay inside this window or MiniMax returns HTTP 400."""
    raw = (pcfg or {}).get("context_window_tokens")
    if raw is None and "minimax" in str((pcfg or {}).get("api_base_url") or "").lower():
        return 204_800
    try:
        value = int(raw) if raw is not None else 200_000
    except (TypeError, ValueError):
        value = 200_000
    return max(4096, value)


def _shrink_context_json(context: str, max_chars: int) -> str:
    """Drops the heaviest optional blocks until the JSON context fits ``max_chars``."""
    try:
        obj = json.loads(context)
    except (TypeError, ValueError, json.JSONDecodeError):
        return context[:max_chars]
    if not isinstance(obj, dict):
        return context[:max_chars]
    domains = obj.get("domains")
    if isinstance(domains, dict):
        for pack in domains.values():
            if not isinstance(pack, dict):
                continue
            pack.pop("load_steps", None)
            pack.pop("stable_detection", None)
            for section in pack.get("sections") or []:
                if not isinstance(section, dict):
                    continue
                for series in section.get("top_series") or []:
                    if isinstance(series, dict):
                        series.pop("step_segments", None)
        text = json.dumps(obj, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        for pack in domains.values():
            if isinstance(pack, dict):
                pack.pop("sections", None)
        text = json.dumps(obj, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        obj.pop("domains", None)
    text = json.dumps(obj, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _trim_user_text(user_text: str, budget_chars: int) -> str:
    """Shrinks the user message so the reserved output budget still fits the window.

    The instruction stays ahead of the JSON context. Optional context blocks
    (detector debug, per-series sections) are dropped before the prompt itself.
    """
    budget = max(0, int(budget_chars))
    marker = "\n\n{"
    split_at = user_text.rfind(marker)
    if split_at < 0:
        return user_text[:budget]
    prompt = user_text[:split_at]
    context = user_text[split_at + 2:]
    prompt_budget = min(len(prompt), budget // 2)
    context_budget = max(0, budget - prompt_budget - 80)
    prompt = prompt[:prompt_budget]
    context = _shrink_context_json(context, context_budget) if context_budget else ""
    return prompt + "\n\n" + context


def _fit_to_window(
    system_text: str,
    user_text: str,
    output_tokens: int,
    window: int,
    count_tokens,
) -> tuple[str, int]:
    """Reserves ``output_tokens`` and trims the user text to the remaining input budget.

    ``count_tokens`` maps text to an integer token estimate. A budget that leaves
    fewer than ``MIN_INPUT_TOKENS`` for the prompt is an error, not a tiny answer.
    """
    reserved = max(1, int(output_tokens))
    input_budget = int(window) - CONTEXT_MARGIN_TOKENS - int(count_tokens(system_text or "")) - reserved
    if input_budget < MIN_INPUT_TOKENS:
        raise LLMBudgetError(
            f"max_tokens={reserved} leaves {input_budget} input tokens in window {window}"
        )
    if int(count_tokens(user_text or "")) <= input_budget:
        return user_text, reserved
    trimmed = _trim_user_text(user_text or "", input_budget)
    logger.warning(
        "LLM prompt reduced from %d to %d tokens to reserve %d output tokens in window %d",
        count_tokens(user_text or ""), count_tokens(trimmed), reserved, window,
    )
    return trimmed, reserved


def _fit_anthropic_payload(system_text: str, user_text: str, requested_max_tokens: int, pcfg: dict) -> tuple[str, int]:
    """Keeps input tokens + max_tokens inside the provider context window."""
    return _fit_to_window(
        system_text,
        user_text,
        int(requested_max_tokens),
        _context_window_tokens(pcfg),
        _estimate_tokens,
    )


def _count_gigachat_tokens(client: object, text: str) -> int:
    """Uses the GigaChat tokenizer when the client exposes it; otherwise the length bound."""
    counter = getattr(client, "tokens_count", None)
    if counter is None or not text:
        return _estimate_tokens(text)
    try:
        result = counter([text])
    except Exception as exc:
        logger.warning("GigaChat tokens_count failed, using the length estimate: %s", exc)
        return _estimate_tokens(text)
    item: Any = result[0] if isinstance(result, list) and result else result
    if isinstance(item, dict):
        counted = _optional_int(item.get("tokens") or item.get("token_count"))
    else:
        counted = _optional_int(getattr(item, "tokens", None) or getattr(item, "token_count", None))
    return counted if counted is not None and counted > 0 else _estimate_tokens(text)


def _response_finish_reason(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
            if reason:
                return str(reason)
        reason = response.get("stop_reason") or response.get("finish_reason")
        return str(reason) if reason else None
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        reason = meta.get("finish_reason") or meta.get("stop_reason")
        if reason:
            return str(reason)
    return None


def _ensure_text_complete(
    provider: str,
    text: str,
    finish: Optional[str],
    max_tokens: int,
    output_tokens: Optional[int] = None,
) -> str:
    """Rejects an answer the provider cut off or left empty."""
    if finish in _TRUNCATED_FINISH_REASONS:
        raise LLMOutputTruncated(provider, max_tokens, output_tokens)
    cleaned = _strip_think(text or "")
    if not cleaned.strip():
        raise LLMEmptyAnswer(provider, max_tokens)
    return cleaned


def _call_with_output_retry(pcfg: dict, call):
    """Runs ``call(cfg)``. A truncated or empty answer is retried once with a doubled budget."""
    try:
        return call(pcfg)
    except (LLMOutputTruncated, LLMEmptyAnswer) as exc:
        budget = int(getattr(exc, "max_tokens", 0) or _effective_max_tokens(pcfg, default=1200, default_cap=32768))
        doubled = max(budget * 2, budget + 1)
        logger.warning("LLM answer unusable (%s); one retry with max_tokens=%d", exc, doubled)
        return call({**pcfg, "_output_tokens_override": doubled})


def _raise_for_llm_status(resp: requests.Response) -> None:
    """Raises HTTPError whose message includes the provider body (MiniMax hides the reason in it)."""
    if resp.ok:
        return
    detail = (resp.text or "").strip().replace("\n", " ")
    if len(detail) > 800:
        detail = detail[:800] + "…"
    message = f"{resp.status_code} {resp.reason} for url: {resp.url}"
    if detail:
        message = f"{message}. {detail}"
    raise requests.HTTPError(message, response=resp)


def _anthropic_call(messages: list[dict], pcfg: dict, system_text: str) -> str:
    base_url = (pcfg.get("api_base_url") or pcfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
    url = f"{base_url}/v1/messages"
    gen = (pcfg.get("generation") or {})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": _env_key('ANTHROPIC_API_KEY') or pcfg.get('api_key', ''),
        "anthropic-version": "2023-06-01",
    }
    proxies = _request_proxies(pcfg)
    verify = _tls_verify(pcfg)

    user_parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            user_parts.append(str(content))
        elif role == "system":
            pass
        else:
            user_parts.append(str(content))
    user_combined = "\n\n".join(user_parts)
    req_max_tokens = _effective_max_tokens(pcfg, default=1200, default_cap=32768)
    user_combined, req_max_tokens = _fit_anthropic_payload(system_text or "", user_combined, req_max_tokens, pcfg)
    logger.info(
        "Anthropic request: chars=%d est_tokens=%d max_tokens=%d window=%d",
        len(user_combined), _estimate_tokens(user_combined), req_max_tokens, _context_window_tokens(pcfg),
    )
    payload = {
        "model": pcfg.get("model", "claude-3-5-sonnet-latest"),
        "system": system_text,
        "max_tokens": req_max_tokens,
        "temperature": _safe_float(gen.get("temperature"), 0.2),
        "messages": [
            {"role": "user", "content": user_combined}
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=_safe_int(pcfg.get("request_timeout_sec"), 120), verify=verify, proxies=proxies)
    _raise_for_llm_status(resp)
    data = resp.json()
    blocks = data.get("content") if isinstance(data, dict) else None
    texts: list[str] = []
    for block in blocks or []:
        if not isinstance(block, dict) or block.get("type") == "thinking":
            continue
        if block.get("type") in (None, "text") and block.get("text"):
            texts.append(str(block.get("text")))
    usage = data.get("usage") if isinstance(data, dict) else None
    output_tokens = _optional_int((usage or {}).get("output_tokens")) if isinstance(usage, dict) else None
    finish = str(data.get("stop_reason")) if isinstance(data, dict) and data.get("stop_reason") else None
    return _ensure_text_complete("anthropic", "\n".join(texts), finish, req_max_tokens, output_tokens)


def _normalize_gigachat_base_url(raw_url: str | None) -> str:
    base = (raw_url or "https://gigachat.devices.sberbank.ru/api/v1").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base or "https://gigachat.devices.sberbank.ru/api/v1"


def _get_gigachat_client(pcfg: dict | None = None):
    """Lazily creates a GigaChat client (langchain_gigachat) with mTLS; cached per config."""
    if not isinstance(pcfg, dict):
        pcfg = ((CONFIG.get("llm") or {}).get("gigachat") or {})
    from langchain_gigachat.chat_models import GigaChat as LC_GigaChat

    base_url = _normalize_gigachat_base_url(pcfg.get("base_url") or pcfg.get("api_base_url"))
    tls = _tls_verify(pcfg)
    verify_ssl_certs = tls is not False
    if _request_proxies(pcfg):
        logger.warning(
            "llm.gigachat.proxies is ignored: GigaChat uses HTTPS_PROXY/HTTP_PROXY of the server process "
            "(set them in the service environment, with NO_PROXY for internal hosts)"
        )
    gen = (pcfg.get("generation") or {})
    client_kwargs = {
        "model": str(pcfg.get("model") or "GigaChat-Pro"),
        "cert_file": pcfg.get("cert_file"),
        "key_file": pcfg.get("key_file"),
        "base_url": base_url,
        "verify_ssl_certs": verify_ssl_certs,
        **({"ca_bundle_file": tls} if isinstance(tls, str) else {}),
        "timeout": _safe_int(pcfg.get("request_timeout_sec"), 120),
        "max_tokens": _effective_max_tokens(pcfg, default=1200, default_cap=32768),
        "temperature": _safe_float(gen.get("temperature"), 0.2),
    }
    if pcfg.get("api_key") or pcfg.get("credentials"):
        client_kwargs["credentials"] = pcfg.get("api_key") or pcfg.get("credentials")
    cache_key = json.dumps(client_kwargs, sort_keys=True, default=str)
    with _gigachat_lock:
        client = _gigachat_clients.get(cache_key)
        if client is None:
            logger.info("Инициализация GigaChat: URL=%s Model=%s SSL=%s", base_url, client_kwargs["model"], verify_ssl_certs)
            client = LC_GigaChat(**client_kwargs)
            _gigachat_clients[cache_key] = client
    return client


def _gigachat_call(messages: list[dict], pcfg: dict, system_text: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    user_parts = [str(m.get("content", "")) for m in messages if m.get("role") != "system"]
    client = _get_gigachat_client(pcfg)
    output_tokens = _effective_max_tokens(pcfg, default=1200, default_cap=32768)
    user_text, _output = _fit_to_window(
        system_text or "",
        "\n\n".join(user_parts),
        output_tokens,
        _context_window_tokens(pcfg),
        lambda text: _count_gigachat_tokens(client, text),
    )
    response = client.invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=user_text),
    ])
    prompt_tokens, completion_tokens = _extract_token_counts(response)
    record_llm_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, elapsed_ms=0)
    return _ensure_text_complete(
        "gigachat",
        str(getattr(response, "content", "") or ""),
        _response_finish_reason(response),
        output_tokens,
        completion_tokens,
    )


def _call_provider(provider: str, messages: list[dict], pcfg: dict, system_text: str) -> str:
    if provider == "openai":
        return _openai_call(messages, pcfg)
    if provider == "anthropic":
        return _anthropic_call(messages, pcfg, system_text)
    if provider == "gigachat":
        return _gigachat_call(messages, pcfg, system_text)
    return _perplexity_call(messages, pcfg)


def _max_attempts(pcfg: dict, llm_config: Optional[dict]) -> int:
    raw = (llm_config or {}).get("max_attempts") if isinstance(llm_config, dict) else None
    if raw is None:
        raw = (pcfg or {}).get("max_attempts", DEFAULT_MAX_ATTEMPTS)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ATTEMPTS


def default_system_prompt(lang: Optional[str] = None, force_json: bool = True) -> str:
    """Default system prompt in ``lang``: ``_system.txt`` plus ``_system_json.txt`` when JSON is forced."""
    base = load_prompt("_system.txt", lang).strip()
    if not force_json:
        return base
    return f"{base} {load_prompt('_system_json.txt', lang).strip()}"


def _preference_lang(value: Any) -> Optional[str]:
    """``ru``/``en`` of a Perplexity ``language_preference`` (a code or a language name), else None."""
    raw = str(value or "").strip().lower()
    if raw.startswith("rus"):
        return "ru"
    if raw.startswith("eng"):
        return "en"
    return normalize_lang(raw)


def _check_language_preference(pcfg: dict, lang: Optional[str]) -> None:
    """Warns once per run when ``llm.perplexity.language_preference`` differs from the report language.

    The value is sent only when configured (it is not derived from the run): the prompts set the language.
    """
    configured = pcfg.get("language_preference") if isinstance(pcfg, dict) else None
    if not configured:
        return
    run_lang = resolve_lang(lang)
    preferred = _preference_lang(configured)
    if preferred == run_lang:
        return
    marker = (str(configured), run_lang)
    with _usage_lock:
        if marker in _language_warnings:
            return
        _language_warnings.add(marker)
    logger.warning(
        "llm.perplexity.language_preference=%r differs from the report language %r; the prompts ask for %r",
        configured, run_lang, run_lang,
    )


def ask_llm_with_text_data(
    user_prompt: str,
    data_context: str,
    llm_config: dict = None,
    api_key: str = None,
    model: str = None,
    base_url: str = None,
    system_prompt: Optional[str] = None,
    lang: Optional[str] = None,
) -> str:
    """Единая точка вызова LLM (Perplexity/OpenAI/Anthropic/GigaChat) с текстовым контекстом.

    Параметры:
        user_prompt (str): Инструкция пользователю.
        data_context (str): Дополнительные данные (обычно JSON).
        llm_config (dict | None): Переопределения: `provider`, `force_json`, `max_attempts`,
            а также полный конфиг провайдера под ключом с его именем
            (например ``{"provider": "openai", "openai": {...}}``) — используется
            проверкой соединения, чтобы вызвать модель с ещё не сохранёнными настройками.
        api_key (str | None): Персональный API-ключ (переопределяет конфиг).
        model (str | None): Имя модели.
        base_url (str | None): Альтернативный эндпойнт.
        system_prompt (str | None): Кастомный системный промпт.
        lang (str | None): Язык системного промпта по умолчанию (``AI/prompts/<lang>/_system*.txt``);
            по умолчанию текущая локаль — язык отчёта внутри прогона.

    Возвращает:
        str: Сырой ответ модели.

    Побочные эффекты:
        Выполняет HTTPS-запросы к соответствующему LLM-провайдеру.

    Исключения:
        Пробрасывает ошибки HTTP и таймауты после `max_attempts` попыток
        (по умолчанию три; настраивается в конфиге провайдера).
    """
    llm_root = CONFIG.get("llm", {}) or {}
    provider = (llm_config or {}).get("provider") if isinstance(llm_config, dict) else None
    provider = (provider or llm_root.get("provider") or "perplexity").lower()
    pcfg = llm_root.get(provider, {})
    if isinstance(llm_config, dict) and isinstance(llm_config.get(provider), dict):
        pcfg = llm_config[provider]
    gen = (pcfg.get("generation") or {})
    force_json = bool(gen.get("force_json_in_prompt", True))
    if isinstance(llm_config, dict) and "force_json" in llm_config:
        force_json = bool(llm_config.get("force_json"))
    if isinstance(system_prompt, str) and system_prompt.strip():
        system_text = system_prompt.strip()
    else:
        system_text = default_system_prompt(lang, force_json)

    user_content = user_prompt if not data_context else f"{user_prompt}\n\n{data_context}"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

    if isinstance(model, str) and model.strip():
        pcfg = {**pcfg, "model": model.strip()}
    if isinstance(base_url, str) and base_url.strip():
        pcfg = {**pcfg, "api_base_url": base_url.strip()}
    if isinstance(api_key, str) and api_key.strip():
        pcfg = {**pcfg, "api_key": api_key.strip()}
    if provider == "perplexity":
        _check_language_preference(pcfg, lang)

    max_attempts = _max_attempts(pcfg, llm_config)
    attempts = 0
    last_err = None
    while attempts < max_attempts:
        try:
            _wait_llm_slot(pcfg)
            started = time.perf_counter()
            with _llm_semaphore:
                text = _call_with_output_retry(
                    pcfg, lambda cfg: _call_provider(provider, messages, cfg, system_text),
                )
            if provider != "gigachat":
                record_llm_usage(elapsed_ms=int((time.perf_counter() - started) * 1000))
            return text
        except (LLMOutputTruncated, LLMEmptyAnswer, LLMBudgetError):
            raise
        except requests.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", 0) or 0
            # 400 from MiniMax is an invalid request (context window, bad params). Retrying it does not help.
            if status and status != 429 and status < 500:
                raise
            last_err = e
            attempts += 1
            if attempts >= max_attempts:
                break
            logger.warning(
                "LLM attempt %d/%d failed (provider=%s, status=%s): %s",
                attempts, max_attempts, provider, status, e,
            )
            time.sleep(min(2 ** attempts, 8))
        except Exception as e:
            last_err = e
            attempts += 1
            if attempts >= max_attempts:
                break
            is_rate_limit = "429" in str(e)
            base_delay = min(2 ** (attempts + 1), 60) if is_rate_limit else min(2 ** attempts, 8)
            jitter = random.uniform(0, base_delay * 0.5)
            delay = base_delay + jitter
            if is_rate_limit:
                global_cooldown = max(delay, _safe_float(pcfg.get("rate_limit_cooldown_sec", 15), 15))
                _apply_llm_cooldown(global_cooldown)
            logger.warning(
                "LLM attempt %d/%d failed (provider=%s, rate_limit=%s, retry in %.1fs): %s",
                attempts, max_attempts, provider, is_rate_limit, delay, e,
            )
            time.sleep(delay)
    raise last_err


def structured_output_enabled() -> bool:
    """True when ``llm.gigachat.structured_output`` asks for GigaChat function-calling output."""
    provider = str((CONFIG.get("llm") or {}).get("provider") or "").lower()
    gcfg = (CONFIG.get("llm") or {}).get("gigachat") or {}
    return provider == "gigachat" and bool(gcfg.get("structured_output", False))


def ask_llm_structured(
    user_prompt: str,
    data_context: str,
    schema: Type[StructuredT],
    system_prompt: Optional[str] = None,
    lang: Optional[str] = None,
) -> StructuredT:
    """Asks GigaChat for an answer that already matches ``schema``.

    Uses ``with_structured_output`` of langchain-gigachat. Raises RuntimeError
    when the installed client has no such method — there is no silent fallback.
    """
    gcfg = dict((CONFIG.get("llm") or {}).get("gigachat") or {})
    from langchain_core.messages import HumanMessage, SystemMessage

    system_text = (system_prompt or "").strip() or prompt_text("system.structured", lang)
    user_content = user_prompt if not data_context else f"{user_prompt}\n\n{data_context}"

    def _invoke(cfg: dict) -> StructuredT:
        client = _get_gigachat_client(cfg)
        if not hasattr(client, "with_structured_output"):
            raise RuntimeError(t("llm.error.gigachat_no_structured_output"))
        output_tokens = _effective_max_tokens(cfg, default=1200, default_cap=32768)
        fitted_user, _reserved = _fit_to_window(
            system_text,
            user_content,
            output_tokens,
            _context_window_tokens(cfg),
            lambda text: _count_gigachat_tokens(client, text),
        )
        messages = [SystemMessage(content=system_text), HumanMessage(content=fitted_user)]
        include_raw = False
        try:
            runnable = client.with_structured_output(schema, include_raw=True)
            include_raw = True
        except TypeError:
            runnable = client.with_structured_output(schema)
        started = time.perf_counter()
        result: Any = runnable.invoke(messages)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        parsed: Any = result
        raw_message = None
        if include_raw and isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw_message = result.get("raw")
        prompt_tokens, completion_tokens = _extract_token_counts(raw_message if raw_message is not None else result)
        record_llm_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, elapsed_ms=elapsed_ms)
        finish = _response_finish_reason(raw_message if raw_message is not None else result)
        if finish in _TRUNCATED_FINISH_REASONS:
            raise LLMOutputTruncated("gigachat", output_tokens, completion_tokens)
        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, dict):
            return schema(**parsed)
        raise TypeError(f"Structured output returned {type(result).__name__}, expected {schema.__name__}")

    max_attempts = _max_attempts(gcfg, None)
    attempts = 0
    last_err: Optional[Exception] = None
    while attempts < max_attempts:
        try:
            _wait_llm_slot(gcfg)
            with _llm_semaphore:
                return _call_with_output_retry(gcfg, _invoke)
        except (LLMOutputTruncated, LLMEmptyAnswer, LLMBudgetError, RuntimeError):
            raise
        except Exception as exc:
            last_err = exc
            attempts += 1
            if attempts >= max_attempts:
                break
            time.sleep(min(2 ** attempts, 8))
    raise last_err  # type: ignore[misc]


