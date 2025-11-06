import os
import json
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


def _normalize_llm_base_url(raw_url: str | None) -> str:
    base = (raw_url or "https://api.perplexity.ai").strip()
    if not base:
        return "https://api.perplexity.ai"
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


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
        return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    except Exception:
        return text


def _perplexity_call(messages: list[dict], pcfg: dict) -> str:
    base_url = _normalize_llm_base_url(pcfg.get("base_url") or pcfg.get("api_base_url"))
    disable_web = bool(pcfg.get("disable_web_search", True))
    default_model = "llama-3.1-70b-instruct"
    offline_default = "llama-3.1-70b-instruct"
    model = pcfg.get("model", offline_default if disable_web else default_model)
    gen = (pcfg.get("generation") or {})
    url = f"{base_url}/chat/completions"

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

    req_max_tokens = int(gen.get("max_tokens", 1200))
    try:
        cap = int(pcfg.get("max_tokens_cap", 8192))
        if req_max_tokens > cap:
            req_max_tokens = cap
    except Exception:
        pass
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(gen.get("temperature", 0.2)),
        "top_p": float(gen.get("top_p", 0.9)),
        "max_tokens": req_max_tokens,
    }
    if disable_web:
        payload["disable_search"] = True

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
    req_max_tokens = int(gen.get("max_tokens", 1200))
    try:
        cap = int(pcfg.get("max_tokens_cap", 8192))
        if req_max_tokens > cap:
            req_max_tokens = cap
    except Exception:
        pass
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
    req_max_tokens = int(gen.get("max_tokens", 1200))
    try:
        cap = int(pcfg.get("max_tokens_cap", 4096))
        if req_max_tokens > cap:
            req_max_tokens = cap
    except Exception:
        pass
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
    """Единая точка вызова LLM по текстовому интерфейсу."""
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
        "Отвечайте на русском языке. Все текстовые поля (verdict, findings.summary, findings.evidence, recommended_actions, affected_components) формулируйте по-русски; допускаются английские только ключи JSON, значения 'severity' и имена метрик/лейблов. " +
        (
            "Строго в JSON со схемой: {verdict, confidence, findings[], recommended_actions[]}. "
            "Каждый элемент findings обязан содержать: summary, severity (critical|high|medium|low), component, evidence. "
            "Если component не указан — извлеките его из evidence по лейблам application|service|job|pod|instance, иначе 'unknown'. "
            "Если severity не указана — используйте 'low'. "
            "Если данные содержат общий RPS на входной точке, включите поле peak_performance: {max_rps, max_time, drop_time, method='max_step_before_drop'}."
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
            time.sleep(min(2 ** attempts, 8))
    raise last_err


