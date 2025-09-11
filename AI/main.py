# main.py

import requests
import pandas as pd
from typing import List, Dict
import os
from datetime import datetime
from atlassian import Confluence
from datetime import datetime
from getpass import getpass
from bs4 import BeautifulSoup
from tabulate import tabulate
import logging
import threading
import socket
from urllib.parse import urlparse

from requests.auth import HTTPBasicAuth
from langchain_gigachat.chat_models import GigaChat as LC_GigaChat
try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    try:
        from langchain.schema import SystemMessage, HumanMessage  # старые версии langchain
    except Exception:
        SystemMessage = None
        HumanMessage = None


# Импортируем CONFIG из config.py
from AI.config import CONFIG

logger = logging.getLogger(__name__)
_gigachat_lock = threading.Lock()
_gigachat_client = None

# httpx/requests таймаут через окружение для некоторых клиентов
os.environ.setdefault("GIGACHAT_TIMEOUT", str(CONFIG.get("llm", {}).get("gigachat", {}).get("request_timeout_sec", 120)))


def _normalize_gigachat_base_url(raw_url: str | None) -> str:
    base = (raw_url or "https://gigachat.devices.sberbank.ru/api/v1").strip()
    if not base:
        return "https://gigachat.devices.sberbank.ru/api/v1"
    base = base.rstrip("/")
    # Убираем случайный суффикс /chat/completions, если администратор указал полноценный путь
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _ensure_gigachat_env(gcfg: dict) -> None:
    """Применяет сетевые настройки (прокси/CA/инsecure) через переменные окружения."""
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

    logger.info(
        f"GigaChat net env set: HTTPS_PROXY={'set' if https_proxy else 'unset'} "
        f"HTTP_PROXY={'set' if http_proxy else 'unset'} CA_BUNDLE={'set' if (ca_bundle and not insecure) else 'unset'} "
        f"INSECURE={'on' if insecure else 'off'}"
    )


def _gigachat_preflight(gcfg: dict) -> None:
    """Быстрая проверка доступности GigaChat API (mTLS)."""
    timeout = float((gcfg or {}).get("connect_timeout_sec") or 5)
    proxies = (gcfg or {}).get("proxies", {}) or {}
    has_proxy = bool(proxies.get("https") or proxies.get("http") or proxies.get("HTTPS") or proxies.get("HTTP"))

    api_base = _normalize_gigachat_base_url(gcfg.get("api_base_url") or gcfg.get("base_url"))
    parsed = urlparse(api_base)
    host = parsed.hostname or "gigachat.devices.sberbank.ru"
    port = parsed.port or (443 if (parsed.scheme or "https").lower() == "https" else 80)
    models_url = f"{api_base}/models"

    if not has_proxy:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                logger.info(f"Preflight ok: api_host {host}:{port}")
        except Exception as e:
            logger.error(f"Preflight FAIL: api_host {host}:{port} -> {e}")
    else:
        logger.info("Preflight: proxies detected, skipping direct TCP checks")

    verify = gcfg.get("verify") or gcfg.get("ca_bundle_file") or gcfg.get("ca_bundle") or True
    cert = None
    if gcfg.get("use_mtls") and gcfg.get("cert_file") and gcfg.get("key_file"):
        cert = (gcfg.get("cert_file"), gcfg.get("key_file"))

    try:
        resp = requests.get(
            models_url,
            headers={"Accept": "application/json"},
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies if proxies else None,
        )
        logger.info(f"Preflight GET {models_url} status={resp.status_code}")
    except Exception as e:
        logger.error(f"Preflight HTTP to API failed: {e}")


def _configure_logging():
    level_name = (CONFIG.get("logging", {}).get("level") if CONFIG.get("logging") else "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger().setLevel(level)


# Вспомогательная функция для конвертации времени
def convert_to_timestamp(date_str):
    """Конвертирует строку даты и времени в миллисекундный timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
    return int(dt.timestamp())

def parse_step_to_seconds(step: str) -> int:
    """Преобразует шаг '1m', '30s' в целые секунды."""
    if step.endswith('m'):
        return int(step[:-1]) * 60
    elif step.endswith('s'):
        return int(step[:-1])
    else:
        return int(step)


def fetch_prometheus_data(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    """
    Запрашивает PromQL-запрос у Prometheus за интервал [start_ts, end_ts].
    Возвращает сырые данные в формате JSON.
    """
    step_in_seconds = parse_step_to_seconds(step)

    params = {
        'query': promql_query,
        'start': start_ts,
        'end':   end_ts,
        'step':  step_in_seconds
    }

    url = f'{prometheus_url}/api/v1/query_range'
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _resolve_grafana_prom_ds_id(g_cfg: dict) -> int:
    base_url = g_cfg["base_url"].rstrip("/")
    ds_cfg = g_cfg.get("prometheus_datasource", {})
    auth_cfg = g_cfg.get("auth", {})

    headers = {}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg['token']}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg["username"], auth_cfg["password"])

    verify = g_cfg.get("verify_ssl", True)

    if isinstance(ds_cfg.get("id"), int):
        logger.info(f"Grafana datasource id (configured): {ds_cfg['id']}")
        return ds_cfg["id"]

    if ds_cfg.get("uid"):
        url = f"{base_url}/api/datasources/uid/{ds_cfg['uid']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]

    if ds_cfg.get("name"):
        url = f"{base_url}/api/datasources/name/{ds_cfg['name']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]

    url = f"{base_url}/api/datasources"
    resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
    resp.raise_for_status()
    for ds in resp.json():
        if ds.get("type") == "prometheus":
            return ds["id"]
    raise RuntimeError("Не найден Prometheus datasource в Grafana")


def fetch_prometheus_data_via_grafana(
    g_cfg: dict,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    step_in_seconds = parse_step_to_seconds(step)
    base_url = g_cfg["base_url"].rstrip("/")

    ds_id = _resolve_grafana_prom_ds_id(g_cfg)

    params = {
        'query': promql_query,
        'start': start_ts,
        'end':   end_ts,
        'step':  step_in_seconds
    }

    headers = {}
    auth = None
    method = (g_cfg.get("auth", {}).get("method") or "basic").lower()
    if method == "bearer" and g_cfg["auth"].get("token"):
        headers["Authorization"] = f"Bearer {g_cfg['auth']['token']}"
    elif method == "basic" and g_cfg["auth"].get("username") and g_cfg["auth"].get("password"):
        auth = (g_cfg["auth"]["username"], g_cfg["auth"]["password"])

    url = f"{base_url}/api/datasources/proxy/{ds_id}/api/v1/query_range"
    resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=30, verify=g_cfg.get("verify_ssl", True))
    resp.raise_for_status()
    return resp.json()


def fetch_metric_series(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    src = (CONFIG.get("metrics_source", {}).get("type") or "prometheus").lower()
    if src == "grafana_proxy":
        g_cfg = CONFIG.get("metrics_source", {}).get("grafana", {})
        return fetch_prometheus_data_via_grafana(g_cfg, start_ts, end_ts, promql_query, step)
    else:
        return fetch_prometheus_data(prometheus_url, start_ts, end_ts, promql_query, step)


def fetch_and_aggregate_with_label_keys(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_queries: List[str],
    label_keys_list: List[List[str]],
    step: str,
    resample_interval: str
) -> List[pd.DataFrame]:
    if len(promql_queries) != len(label_keys_list):
        raise ValueError(
            "Количество запросов (promql_queries) и количество списков лейблов (label_keys_list) не совпадает!"
        )

    dfs = []
    for query, keys_for_this_query in zip(promql_queries, label_keys_list):
        data_json = fetch_metric_series(
            prometheus_url,
            start_ts,
            end_ts,
            query,
            step
        )

        records = []
        if data_json.get("status") == "success":
            result = data_json["data"].get("result", [])
            for series in result:
                lbls = series.get("metric", {})

                label_parts = []
                for key in keys_for_this_query:
                    val = lbls.get(key, "unknown")
                    label_parts.append(f"{key}={val}")
                label_str = "|".join(label_parts)

                for (ts_float, value_str) in series["values"]:
                    val = float(value_str)
                    records.append([ts_float, label_str, val])

        if not records:
            dfs.append(pd.DataFrame())
            continue

        df = pd.DataFrame(records, columns=["timestamp", "label", "value"])
        df["time"] = pd.to_datetime(df["timestamp"], unit='s')
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-3')
        df.set_index("time", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)

        pivoted = df.pivot_table(
            index=df.index,
            columns="label",
            values="value",
            aggfunc="sum"
        )

        pivoted_resampled = pivoted.resample(resample_interval).mean()
        dfs.append(pivoted_resampled)

    return dfs


def dataframes_to_markdown(labeled_dfs):
    result = []
    for item in labeled_dfs:
        label = item['label']
        df = item['df'].copy()
        result.append(f"## {label}\n")
        result.append("### Топ-10 сервисов по среднему значению\n")
        if not df.empty and df.shape[0] > 0:
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                column_means = df[numeric_columns].mean()
                sorted_numeric_columns = column_means.sort_values(ascending=False).index.tolist()
                non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
                sorted_columns = sorted_numeric_columns + non_numeric_columns
            else:
                sorted_columns = df.columns.tolist()
            top_columns = sorted_columns[:min(10, len(sorted_columns))]
            df = df[top_columns]
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.replace('|', '/')
            for col in df.select_dtypes(include=['number']).columns:
                max_val = df[col].abs().max()
                if max_val >= 1e6:
                    df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
                elif max_val >= 1000:
                    df[col] = df[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) else "")
                else:
                    df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
        df_transposed = df.T
        df_transposed.index = df_transposed.index.map(lambda x: str(x).replace('|', '/'))
        if hasattr(df_transposed, 'columns'):
            df_transposed.columns = df_transposed.columns.map(lambda x: str(x).replace('|', '/'))
        result.append(df_transposed.to_markdown() + "\n\n")
    return "\n".join(result)


def label_dataframes(
    dfs: List[pd.DataFrame],
    labels: List[str]
) -> List[Dict[str, object]]:
    if len(dfs) != len(labels):
        raise ValueError("Количество DataFrame и количество меток не совпадает!")
    labeled_list = []
    for df, label in zip(dfs, labels):
        labeled_list.append({
            "label": label,
            "df": df
        })
    return labeled_list


def _get_gigachat_client() -> LC_GigaChat:
    global _gigachat_client
    if _gigachat_client is not None:
        return _gigachat_client
    gcfg = CONFIG.get("llm", {}).get("gigachat", {})
    base_url = _normalize_gigachat_base_url(gcfg.get("base_url") or gcfg.get("api_base_url"))
    model = gcfg.get("model", "GigaChat-Pro")
    cert_file = gcfg.get("cert_file")
    key_file = gcfg.get("key_file")
    verify_param = gcfg.get("verify")
    verify_ssl_certs = verify_param if isinstance(verify_param, bool) else True
    if isinstance(verify_param, str):
        os.environ["REQUESTS_CA_BUNDLE"] = verify_param
        os.environ["SSL_CERT_FILE"] = verify_param

    logger.info(
        """
Инициализация подключения к GigaChat
URL: %s
Model: %s
SSL: %s
Debug: %s
""",
        base_url,
        model,
        True,
        False,
    )
    _gigachat_client = LC_GigaChat(
        model=model,
        cert_file=cert_file,
        key_file=key_file,
        base_url=base_url,
        verify_ssl_certs=verify_ssl_certs,
        timeout=int(CONFIG.get("llm", {}).get("gigachat", {}).get("request_timeout_sec", 120)),
    )
    return _gigachat_client


def ask_llm_with_text_data(
    user_prompt: str,
    data_context: str,
    llm_config: dict = None,
    api_key: str = None,
    model: str = None,
    base_url: str = None
) -> str:
    """
    Отправляет запрос к GigaChat (через langchain_gigachat) с подготовленными текстовыми данными.
    """
    gcfg = CONFIG.get("llm", {}).get("gigachat", {})
    _ensure_gigachat_env(gcfg)
    _gigachat_preflight(gcfg)

    system_text = (
        "Вы инженер по нагрузочному тестированию. Должны проанализирвать результаты ступенчатого нагрузочного теста поиска максимальной производительности."
        "Пользователь предоставит данные и вопрос. "
        "Используйте контекст этих данных, чтобы ответить на его вопрос."
    )
    if SystemMessage and HumanMessage:
        lc_messages = [SystemMessage(content=system_text), HumanMessage(content=user_prompt + f"\n\n{data_context}")]
        with _gigachat_lock:
            # простые ретраи при read timeout
            attempts = 0
            last_err = None
            while attempts < 3:
                try:
                    result = _get_gigachat_client().invoke(lc_messages)
                    break
                except Exception as e:
                    last_err = e
                    attempts += 1
                    logger.warning(f"GigaChat invoke retry {attempts}/3 due to: {e}")
                    if attempts >= 3:
                        raise
            return getattr(result, "content", str(result))
    else:
        # Fallback: одним запросом
        with _gigachat_lock:
            attempts = 0
            last_err = None
            while attempts < 3:
                try:
                    result = _get_gigachat_client().invoke(user_prompt + f"\n\n{data_context}")
                    break
                except Exception as e:
                    last_err = e
                    attempts += 1
                    logger.warning(f"GigaChat invoke retry {attempts}/3 due to: {e}")
                    if attempts >= 3:
                        raise
        return getattr(result, "content", str(result))


def read_prompt_from_file(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def uploadFromLLM(start_ts, end_ts):
    _configure_logging()
    prometheus_url = CONFIG["prometheus"]["url"]
    src_type = (CONFIG.get("metrics_source", {}).get("type") or "prometheus").lower()
    if src_type == "grafana_proxy":
        g = CONFIG.get("metrics_source", {}).get("grafana", {})
        logger.info(f"Metrics source: Grafana proxy base_url={g.get('base_url')} auth_method={(g.get('auth', {}).get('method'))} verify_ssl={g.get('verify_ssl')} ds_hint={g.get('prometheus_datasource')}")
    else:
        logger.info(f"Metrics source: Prometheus url={prometheus_url}")
    
    step = CONFIG["default_params"]["step"]
    resample = CONFIG["default_params"]["resample_interval"]

    jvm_conf = CONFIG["queries"]["jvm"]
    dfs_jvm = fetch_and_aggregate_with_label_keys(
        prometheus_url,
        start_ts,
        end_ts,
        jvm_conf["promql_queries"],
        jvm_conf["label_keys_list"],
        step=step,
        resample_interval=resample
    )
    labeled_jvm = label_dataframes(dfs_jvm, jvm_conf["labels"])

    arango_conf = CONFIG["queries"]["arangodb"]
    dfs_arangodb = fetch_and_aggregate_with_label_keys(
        prometheus_url,
        start_ts,
        end_ts,
        arango_conf["promql_queries"],
        arango_conf["label_keys_list"],
        step=step,
        resample_interval=resample
    )
    labeled_arangodb = label_dataframes(dfs_arangodb, arango_conf["labels"])

    kafka_conf = CONFIG["queries"]["kafka"]
    dfs_kafka = fetch_and_aggregate_with_label_keys(
        prometheus_url,
        start_ts,
        end_ts,
        kafka_conf["promql_queries"],
        kafka_conf["label_keys_list"],
        step=step,
        resample_interval=resample
    )
    labeled_kafka = label_dataframes(dfs_kafka, kafka_conf["labels"])

    ms_conf = CONFIG["queries"]["microservices"]
    dfs_ms = fetch_and_aggregate_with_label_keys(
        prometheus_url,
        start_ts,
        end_ts,
        ms_conf["promql_queries"],
        ms_conf["label_keys_list"],
        step=step,
        resample_interval=resample
    )
    labeled_ms = label_dataframes(dfs_ms, ms_conf["labels"])

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(CURRENT_DIR, "prompts")
    
    prompt_jvm = read_prompt_from_file(os.path.join(prompts_dir, "jvm_prompt.txt"))
    prompt_arangodb = read_prompt_from_file(os.path.join(prompts_dir, "arangodb_prompt.txt"))
    prompt_kafka = read_prompt_from_file(os.path.join(prompts_dir, "kafka_prompt.txt"))
    prompt_microservices = read_prompt_from_file(os.path.join(prompts_dir, "microservices_prompt.txt"))
    prompt_overall = read_prompt_from_file(os.path.join(prompts_dir, "overall_prompt.txt"))

    jvm_full_data = dataframes_to_markdown(labeled_jvm)
    arangodb_full_data = dataframes_to_markdown(labeled_arangodb)
    kafka_full_data = dataframes_to_markdown(labeled_kafka)
    ms_full_data = dataframes_to_markdown(labeled_ms)

    answer_jvm = ask_llm_with_text_data(user_prompt=prompt_jvm, data_context=jvm_full_data)
    answer_arangodb = ask_llm_with_text_data(user_prompt=prompt_arangodb, data_context=arangodb_full_data)
    answer_kafka = ask_llm_with_text_data(user_prompt=prompt_kafka, data_context=kafka_full_data)
    answer_ms = ask_llm_with_text_data(user_prompt=prompt_microservices, data_context=ms_full_data)

    merged_prompt_overall = (
        prompt_overall
        .replace("{answer_jvm}", answer_jvm)
        .replace("{answer_arangodb}", answer_arangodb)
        .replace("{answer_kafka}", answer_kafka)
        .replace("{answer_microservices}", answer_ms)
    )

    final_answer = ask_llm_with_text_data(user_prompt=merged_prompt_overall, data_context="")

    return {
        "jvm": f"{jvm_full_data}\n\nАнализ JVM:\n{answer_jvm}",
        "arangodb": f"{arangodb_full_data}\n\nАнализ ArangoDB:\n{answer_arangodb}",
        "kafka": f"{kafka_full_data}\n\nАнализ Kafka:\n{answer_kafka}",
        "ms": f"{ms_full_data}\n\nАнализ микросервисов:\n{answer_ms}",
        "final": final_answer
    }


