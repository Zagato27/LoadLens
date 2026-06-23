import os
import json
import random
import time
import threading
import logging
from typing import Optional
import requests
from urllib.parse import urlparse

from settings import CONFIG


logger = logging.getLogger(__name__)
_llm_env_init_lock = threading.Lock()
_llm_env_applied = False
_llm_provider_name = (CONFIG.get("llm", {}) or {}).get("provider", "perplexity").lower()
_llm_provider_cfg = (CONFIG.get("llm", {}) or {}).get(_llm_provider_name, {})
_llm_semaphore = threading.Semaphore(int(_llm_provider_cfg.get("max_concurrent", 4)))
_llm_rate_lock = threading.Lock()
_llm_last_request_ts = 0.0
_llm_next_allowed_ts = 0.0
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


def _ensure_llm_network_env(gcfg: dict) -> None:
    proxies = (gcfg or {}).get("proxies", {}) or {}
    ca_bundle = (gcfg or {}).get("ca_bundle")
    insecure = bool((gcfg or {}).get("insecure_skip_verify", False))

    https_proxy = proxies.get("https") or proxies.get("HTTPS")
    http_proxy = proxies.get("http") or proxies.get("HTTP")

    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy

    if ca_bundle and not insecure:
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle

    if insecure:
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        os.environ.pop("REQUESTS_CA_BUNDLE", None)
        os.environ.pop("SSL_CERT_FILE", None)


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


def _effective_max_tokens(provider_cfg: dict, default: int, default_cap: int) -> int:
    gen = (provider_cfg or {}).get("generation") or {}
    requested = int(gen.get("max_tokens", default))
    try:
        cap = int((provider_cfg or {}).get("max_tokens_cap", os.getenv("LOADLENS_LLM_MAX_TOKENS_CAP", str(default_cap))))
        if cap > 0 and requested > cap:
            return cap
    except Exception:
        pass
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
        "Authorization": f"Bearer {os.getenv('PPLX_API_KEY') or os.getenv('PERPLEXITY_API_KEY') or pcfg.get('api_key', '')}",
    }

    proxies = (pcfg or {}).get("proxies", {}) or None
    verify_cfg = pcfg.get("verify", True)
    verify = True
    if isinstance(verify_cfg, bool):
        verify = verify_cfg
    elif isinstance(verify_cfg, str) and verify_cfg.strip():
        verify = verify_cfg.strip() if os.path.exists(verify_cfg.strip()) else True

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
            "temperature": float(gen.get("temperature", 0.2)),
            "top_p": float(gen.get("top_p", 0.9)),
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
        timeout=int(pcfg.get("request_timeout_sec", 120)),
        verify=verify,
        proxies=proxies,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        if api_type == "agent":
            agent_text = _extract_perplexity_agent_text(data)
            if agent_text:
                return _strip_think(agent_text)
        return _strip_think(data["choices"][0]["message"]["content"]) 
    except Exception:
        return _strip_think(json.dumps(data, ensure_ascii=False))


def _openai_call(messages: list[dict], pcfg: dict) -> str:
    base_url = (_normalize_llm_base_url(pcfg.get("api_base_url") or pcfg.get("base_url")))
    url = f"{base_url}/chat/completions"
    gen = (pcfg.get("generation") or {})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY') or pcfg.get('api_key', '')}",
    }
    proxies = (pcfg or {}).get("proxies", {}) or None
    verify_cfg = pcfg.get("verify", True)
    verify = True if isinstance(verify_cfg, bool) else (verify_cfg.strip() if isinstance(verify_cfg, str) and verify_cfg.strip() else True)
    req_max_tokens = _effective_max_tokens(pcfg, default=1200, default_cap=32768)
    payload = {
        "model": pcfg.get("model", "gpt-4o-mini"),
        "messages": messages,
        "temperature": float(gen.get("temperature", 0.2)),
        "top_p": float(gen.get("top_p", 0.9)),
        "max_tokens": req_max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=int(pcfg.get("request_timeout_sec", 120)), verify=verify, proxies=proxies)
    resp.raise_for_status()
    data = resp.json()
    try:
        return _strip_think(data["choices"][0]["message"]["content"]) 
    except Exception:
        return _strip_think(json.dumps(data, ensure_ascii=False))


def _anthropic_call(messages: list[dict], pcfg: dict, system_text: str) -> str:
    base_url = (pcfg.get("api_base_url") or pcfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
    url = f"{base_url}/v1/messages"
    gen = (pcfg.get("generation") or {})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": os.getenv('ANTHROPIC_API_KEY') or pcfg.get('api_key', ''),
        "anthropic-version": "2023-06-01",
    }
    proxies = (pcfg or {}).get("proxies", {}) or None
    verify_cfg = pcfg.get("verify", True)
    verify = True if isinstance(verify_cfg, bool) else (verify_cfg.strip() if isinstance(verify_cfg, str) and verify_cfg.strip() else True)

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
    payload = {
        "model": pcfg.get("model", "claude-3-5-sonnet-latest"),
        "system": system_text,
        "max_tokens": req_max_tokens,
        "temperature": float(gen.get("temperature", 0.2)),
        "messages": [
            {"role": "user", "content": user_combined}
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=int(pcfg.get("request_timeout_sec", 120)), verify=verify, proxies=proxies)
    resp.raise_for_status()
    data = resp.json()
    try:
        blocks = data.get("content", [])
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
        return _strip_think("\n".join([t for t in texts if t]))
    except Exception:
        return _strip_think(json.dumps(data, ensure_ascii=False))


def ask_llm_with_text_data(
    user_prompt: str,
    data_context: str,
    llm_config: dict = None,
    api_key: str = None,
    model: str = None,
    base_url: str = None,
    system_prompt: Optional[str] = None
) -> str:
    """Единая точка вызова LLM (Perplexity/OpenAI/Anthropic) с текстовым контекстом.

    Параметры:
        user_prompt (str): Инструкция пользователю.
        data_context (str): Дополнительные данные (обычно JSON).
        llm_config (dict | None): Переопределения (`provider`, `force_json` и др.).
        api_key (str | None): Персональный API-ключ (переопределяет конфиг).
        model (str | None): Имя модели.
        base_url (str | None): Альтернативный эндпойнт.
        system_prompt (str | None): Кастомный системный промпт.

    Возвращает:
        str: Сырой ответ модели.

    Побочные эффекты:
        Выполняет HTTPS-запросы к соответствующему LLM-провайдеру.

    Исключения:
        Пробрасывает ошибки HTTP и таймауты после трёх попыток.
    """
    llm_root = CONFIG.get("llm", {}) or {}
    provider = (llm_config or {}).get("provider") if isinstance(llm_config, dict) else None
    provider = (provider or llm_root.get("provider") or "perplexity").lower()
    pcfg = llm_root.get(provider, {})
    global _llm_env_applied
    if not _llm_env_applied:
        with _llm_env_init_lock:
            if not _llm_env_applied:
                _ensure_llm_network_env(pcfg)
                _llm_env_applied = True

    gen = (pcfg.get("generation") or {})
    force_json = bool(gen.get("force_json_in_prompt", True))
    if isinstance(llm_config, dict) and "force_json" in llm_config:
        force_json = bool(llm_config.get("force_json"))
    system_text = (
        "Вы инженер по нагрузочному тестированию. Должны проанализировать результаты ступенчатого нагрузочного теста поиска максимальной производительности."
        "Пользователь предоставит данные и вопрос. "
        "Используйте контекст этих данных, чтобы ответить на его вопрос. "
        "Отвечайте на русском языке. Все текстовые поля (verdict, findings.summary, findings.evidence_summary, findings.evidence_items.note, "
        "recommended_actions, affected_components) формулируйте по-русски; допускаются английские только ключи JSON, "
        "значения 'severity' и имена метрик/лейблов. " +
        (
            "Строго в JSON со схемой: {verdict, confidence, findings[], recommended_actions[]}. "
            "Каждый элемент findings обязан содержать: id, summary, severity (critical|high|medium|low), component, "
            "start_time, end_time, peak_time, evidence_summary, evidence_items[]. "
            "Каждый элемент evidence_items должен быть объектом {metric, observed_value, threshold, note}. "
            "id должен быть коротким ASCII-идентификатором вроде f1, f2. "
            "Каждый элемент recommended_actions должен быть объектом: {summary, details, priority (critical|high|medium|low), affected_components[], for_finding_ids[]}. "
            "Поле details должно содержать развернутое описание действия: что именно менять, зачем, и как понять, что проблема устранена. "
            "Поле for_finding_ids обязано ссылаться на один или несколько id из findings. "
            "Если component не указан — извлеките его из evidence_summary по лейблам application|service|job|pod|instance, иначе 'unknown'. "
            "Если severity не указана — используйте 'low'. "
            "Поле peak_performance допускается ТОЛЬКО для домена lt_framework "
            "или итогового overall (если в контексте есть designated_peak_performance). "
            "Для остальных доменов peak_performance не добавляйте."
            if force_json else ""
        )
    )
    if isinstance(system_prompt, str) and system_prompt.strip():
        system_text = system_prompt.strip()

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

    attempts = 0
    last_err = None
    while attempts < 3:
        try:
            _wait_llm_slot(pcfg)
            with _llm_semaphore:
                if provider == "perplexity":
                    return _perplexity_call(messages, pcfg)
                elif provider == "openai":
                    return _openai_call(messages, pcfg)
                elif provider == "anthropic":
                    return _anthropic_call(messages, pcfg, system_text)
                else:
                    return _perplexity_call(messages, pcfg)
        except Exception as e:
            last_err = e
            attempts += 1
            is_rate_limit = "429" in str(e)
            base_delay = min(2 ** (attempts + 1), 60) if is_rate_limit else min(2 ** attempts, 8)
            jitter = random.uniform(0, base_delay * 0.5)
            delay = base_delay + jitter
            if is_rate_limit:
                global_cooldown = max(delay, _safe_float(pcfg.get("rate_limit_cooldown_sec", 15), 15))
                _apply_llm_cooldown(global_cooldown)
            logger.warning(
                "LLM attempt %d/3 failed (provider=%s, rate_limit=%s, retry in %.1fs): %s",
                attempts, provider, is_rate_limit, delay, e,
            )
            time.sleep(delay)
    raise last_err


