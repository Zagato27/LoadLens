# main.py

import requests
import pandas as pd
from typing import List, Dict
from openai import OpenAI
import base64
import uuid
import os
from datetime import datetime
from atlassian import Confluence
from datetime import datetime
from getpass import getpass
from bs4 import BeautifulSoup
from tabulate import tabulate
import logging

from requests.auth import HTTPBasicAuth
from gigachat import GigaChat


# Импортируем CONFIG из config.py
from AI.config import CONFIG

logger = logging.getLogger(__name__)

def _configure_logging():
    level_name = (CONFIG.get("logging", {}).get("level") if CONFIG.get("logging") else "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger().setLevel(level)


def _safe_headers(headers: dict) -> dict:
    if not headers:
        return {}
    safe = dict(headers)
    if "Authorization" in safe:
        safe["Authorization"] = "***"
    return safe


def _safe_auth(auth):
    if auth is None:
        return None
    try:
        username, _ = auth
        return (username, "***")
    except Exception:
        return "***"


def _http_get(url: str, *, headers=None, auth=None, params=None, timeout=30, verify=True, log_context=""):
    logger.debug(f"HTTP GET {url} params={params} headers={_safe_headers(headers)} auth={_safe_auth(auth)} verify={verify} {log_context}")
    resp = None
    try:
        resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=timeout, verify=verify)
        logger.debug(f"HTTP GET {url} -> {resp.status_code}")
        resp.raise_for_status()
        return resp
    except Exception as e:
        body = None
        try:
            body = resp.text[:1000] if resp is not None else None
        except Exception:
            body = None
        logger.error(f"HTTP GET failed {url}: {e}; status={getattr(resp,'status_code',None)}; body_snippet={body}")
        raise


def _http_post(url: str, *, headers=None, data=None, json=None, timeout=30, verify=True, log_context=""):
    safe_headers = _safe_headers(headers)
    data_keys = list(data.keys()) if isinstance(data, dict) else None
    logger.debug(f"HTTP POST {url} data_keys={data_keys} json_keys={list(json.keys()) if isinstance(json, dict) else None} headers={safe_headers} verify={verify} {log_context}")
    resp = None
    try:
        resp = requests.post(url, headers=headers, data=data, json=json, timeout=timeout, verify=verify)
        logger.debug(f"HTTP POST {url} -> {resp.status_code}")
        resp.raise_for_status()
        return resp
    except Exception as e:
        body = None
        try:
            body = resp.text[:1000] if resp is not None else None
        except Exception:
            body = None
        logger.error(f"HTTP POST failed {url}: {e}; status={getattr(resp,'status_code',None)}; body_snippet={body}")
        raise


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
    resp = _http_get(url, params=params, timeout=30, verify=True, log_context="prometheus_query_range")
    return resp.json()


def _resolve_grafana_prom_ds_id(g_cfg: dict) -> int:
    """
    Определяет id Prometheus datasource в Grafana по id/uid/name.
    Возвращает целочисленный id.
    """
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

    # 1) Прямой id
    if isinstance(ds_cfg.get("id"), int):
        logger.info(f"Grafana datasource id (configured): {ds_cfg['id']}")
        return ds_cfg["id"]

    # 2) По uid
    if ds_cfg.get("uid"):
        url = f"{base_url}/api/datasources/uid/{ds_cfg['uid']}"
        resp = _http_get(url, headers=headers, auth=auth, timeout=30, verify=verify, log_context="grafana_get_ds_by_uid")
        ds_id = resp.json()["id"]
        logger.info(f"Grafana datasource resolved by uid={ds_cfg['uid']} -> id={ds_id}")
        return ds_id

    # 3) По name
    if ds_cfg.get("name"):
        url = f"{base_url}/api/datasources/name/{ds_cfg['name']}"
        resp = _http_get(url, headers=headers, auth=auth, timeout=30, verify=verify, log_context="grafana_get_ds_by_name")
        ds_id = resp.json()["id"]
        logger.info(f"Grafana datasource resolved by name={ds_cfg['name']} -> id={ds_id}")
        return ds_id

    # 4) Автовыбор первого Prometheus-датасорса
    url = f"{base_url}/api/datasources"
    resp = _http_get(url, headers=headers, auth=auth, timeout=30, verify=verify, log_context="grafana_list_ds")
    for ds in resp.json():
        if ds.get("type") == "prometheus":
            logger.info(f"Grafana datasource auto-selected: id={ds['id']} name={ds.get('name')}")
            return ds["id"]
    raise RuntimeError("Не найден Prometheus datasource в Grafana")


def fetch_prometheus_data_via_grafana(
    g_cfg: dict,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    """
    Выполняет PromQL-запрос через Grafana proxy: /api/datasources/proxy/{id}/api/v1/query_range.
    Возвращает ответ в формате Prometheus API.
    """
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
    resp = _http_get(url, headers=headers, auth=auth, params=params, timeout=30, verify=g_cfg.get("verify_ssl", True), log_context="grafana_proxy_query_range")
    return resp.json()


def fetch_metric_series(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    """Единая точка получения метрик: напрямую Prometheus или через Grafana proxy (по CONFIG)."""
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
    """
    Выполняет несколько PromQL-запросов, превращает их в DataFrame с учётом
    списка лейблов (label_keys) для каждого запроса.
    """
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
    """
    Преобразует DataFrame в читаемый формат Markdown с транспонированием таблицы
    и сортировкой значений от больших к меньшим, выводит только топ-10.
    Форматирует числа в обычном виде, без научной нотации.
    Заменяет символ '|' на '/' для корректного отображения таблицы в markdown.
    """
    result = []
    for item in labeled_dfs:
        label = item['label']
        df = item['df'].copy()  # Создаем копию, чтобы не модифицировать оригинал
        
        # Добавляем заголовок таблицы
        result.append(f"## {label}\n")
        
        # Проверяем структуру DataFrame
        is_time_in_index = 'time' in df.index.names
        
        # Вывод топ-10 данных (транспонированный и отсортированный)
        result.append("### Топ-10 сервисов по среднему значению\n")
        
        # Сортируем по среднему значению столбцов (от большего к меньшему)
        if not df.empty and df.shape[0] > 0:
            # Проверяем, содержит ли DataFrame числовые данные
            numeric_columns = df.select_dtypes(include=['number']).columns
            
            if len(numeric_columns) > 0:
                # Рассчитываем среднее значение для числовых столбцов
                column_means = df[numeric_columns].mean()
                # Сортируем числовые столбцы по среднему значению (от большего к меньшему)
                sorted_numeric_columns = column_means.sort_values(ascending=False).index.tolist()
                
                # Составляем полный список столбцов, сначала отсортированные числовые, затем остальные
                non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
                sorted_columns = sorted_numeric_columns + non_numeric_columns
            else:
                sorted_columns = df.columns.tolist()
            
            # Берем только топ-10 столбцов (или меньше, если столбцов меньше 10)
            top_columns = sorted_columns[:min(10, len(sorted_columns))]
            df = df[top_columns]
            
            # Заменяем символ '|' на '/' во всех строковых столбцах
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.replace('|', '/')
            
            # Форматируем числа, чтобы избавиться от научной нотации (e+)
            # Преобразуем все числовые столбцы к строкам с обычным форматом
            for col in df.select_dtypes(include=['number']).columns:
                # Определяем, содержит ли столбец большие числа
                max_val = df[col].abs().max()
                if max_val >= 1e6:  # Если значения превышают миллион
                    # Форматируем без десятичных знаков для больших чисел
                    df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
                elif max_val >= 1000:  # Для средних чисел
                    # Форматируем с одним десятичным знаком
                    df[col] = df[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) else "")
                else:  # Для малых чисел
                    # Форматируем с четырьмя десятичными знаками
                    df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
        
        # Транспонируем DataFrame перед выводом
        df_transposed = df.T
        
        # Заменяем символ '|' на '/' в названиях строк (индексе)
        df_transposed.index = df_transposed.index.map(lambda x: str(x).replace('|', '/'))
        
        # Если названия столбцов тоже могут содержать вертикальную черту
        if hasattr(df_transposed, 'columns'):
            df_transposed.columns = df_transposed.columns.map(lambda x: str(x).replace('|', '/'))
        
        # Используем to_markdown без дополнительного форматирования
        result.append(df_transposed.to_markdown() + "\n\n")
        
    return "\n".join(result)








def label_dataframes(
    dfs: List[pd.DataFrame],
    labels: List[str]
) -> List[Dict[str, object]]:
    """
    Принимает список датафреймов (dfs) и список строк-меток (labels).
    Возвращает список словарей [{ "label": <string>, "df": <DataFrame> }, ...].
    """
    if len(dfs) != len(labels):
        raise ValueError("Количество DataFrame и количество меток не совпадает!")

    labeled_list = []
    for df, label in zip(dfs, labels):
        labeled_list.append({
            "label": label,
            "df": df
        })
    return labeled_list





def _get_gigachat_access_token(client_id: str = None, client_secret: str = None, auth_url: str = None, scope: str = None, verify_ssl: bool = False, authorization_key: str = None) -> str:
    """Получает OAuth2 токен для Sber GigaChat API."""
    # Если передан готовый Basic Authorization key из личного кабинета, используем его напрямую
    if authorization_key:
        basic_header_value = authorization_key.strip()
        if not basic_header_value.lower().startswith("basic "):
            basic_header_value = f"Basic {basic_header_value}"
    else:
        # Формируем заголовок Authorization: Basic base64(client_id:client_secret)
        if not client_id or not client_secret:
            raise RuntimeError("Для GigaChat требуется либо authorization_key, либо client_id и client_secret")
        basic_token = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
        basic_header_value = f"Basic {basic_token}"
    headers = {
        "Authorization": basic_header_value,
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "scope": scope,
        "grant_type": "client_credentials"
    }
    logger.info(f"GigaChat OAuth: url={auth_url} verify_ssl={verify_ssl}")
    resp = _http_post(auth_url, headers=headers, data=data, timeout=30, verify=verify_ssl, log_context="gigachat_oauth")
    payload = resp.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("Не удалось получить access_token для GigaChat")
    return access_token


def _create_openai_compatible_client(llm_cfg: dict) -> (OpenAI, str):
    """Создаёт клиента OpenAI-совместимого API и возвращает пару (client, model)."""
    provider = llm_cfg.get("provider", "perplexity").lower()

    if provider in ("openai", "perplexity", "openai_compatible"):
        client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url"))
        logger.info(f"LLM client: provider={provider} base_url={llm_cfg.get('base_url')} model={llm_cfg.get('model')}")
        return client, llm_cfg["model"]

    if provider == "gigachat":
        gcfg = llm_cfg.get("gigachat", {})
        token = _get_gigachat_access_token(
            client_id=gcfg.get("client_id"),
            client_secret=gcfg.get("client_secret"),
            auth_url=gcfg.get("auth_url", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"),
            scope=gcfg.get("scope", "GIGACHAT_API_PERS"),
            verify_ssl=gcfg.get("verify_ssl", False),
            authorization_key=gcfg.get("authorization_key")
        )
        client = OpenAI(api_key=token, base_url=gcfg.get("api_base_url", "https://gigachat.devices.sberbank.ru/api/v1"))
        model_name = gcfg.get("model", llm_cfg.get("model", "GigaChat-Pro"))
        logger.info(f"LLM client: provider=gigachat base_url={gcfg.get('api_base_url')} model={model_name}")
        return client, model_name

    raise ValueError(f"Неизвестный провайдер LLM: {provider}")


def _normalize_gigachat_credentials(gcfg: dict) -> str:
    key = (gcfg or {}).get("authorization_key")
    if not key:
        client_id = (gcfg or {}).get("client_id")
        client_secret = (gcfg or {}).get("client_secret")
        if client_id and client_secret:
            # base64(client_id:client_secret)
            return base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
        raise RuntimeError("GigaChat: не указан authorization_key или client_id/client_secret")
    # допускаем, что ключ может быть с префиксом "Basic " — уберем его
    key = key.strip()
    if key.lower().startswith("basic "):
        key = key.split(" ", 1)[1].strip()
    return key


def _ask_gigachat(messages, llm_cfg: dict) -> str:
    gcfg = (llm_cfg or {}).get("gigachat", {})
    credentials = _normalize_gigachat_credentials(gcfg)
    scope = gcfg.get("scope", "GIGACHAT_API_PERS")
    model_name = gcfg.get("model", llm_cfg.get("model", "GigaChat-Pro"))

    # Логируем параметры (без чувствительных данных)
    logger.info(f"GigaChat chat: model={model_name} scope={scope}")

    # Инициализация клиента gigachat
    try:
        # Некоторые версии SDK принимают verify_ssl, некоторые — нет. Попробуем аккуратно.
        try:
            giga = GigaChat(credentials=credentials, scope=scope, verify_ssl=gcfg.get("verify_ssl", False))
        except TypeError:
            giga = GigaChat(credentials=credentials, scope=scope)
    except Exception as e:
        logger.error(f"GigaChat init failed: {e}")
        raise

    # Вызов chat. SDK может принимать либо строку, либо messages/model
    try:
        response = giga.chat({
            "model": model_name,
            "messages": messages,
            "stream": False,
        })
    except TypeError:
        # Попытка альтернативной сигнатуры: giga.chat(messages=..., model=...)
        response = giga.chat(messages=messages, model=model_name)
    except Exception as e:
        logger.error(f"GigaChat chat failed: {e}")
        raise

    # Извлекаем контент
    try:
        if isinstance(response, dict):
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Объект с атрибутами
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None) or first.get("message")
            if message:
                return getattr(message, "content", None) or message.get("content", "")
        # Fallback: преобразуем в строку
        return str(response)
    except Exception:
        return str(response)


def ask_llm_with_text_data(
    user_prompt: str,
    data_context: str,
    llm_config: dict = None,
    api_key: str = None,
    model: str = None,
    base_url: str = None
) -> str:
    """
    Отправляет запрос к LLM-провайдеру (OpenAI/Perplexity/GigaChat) с подготовленными текстовыми данными.
    Можно передать полный `llm_config` или совместимые параметры `api_key/model/base_url`.
    """
    system_message = {
        "role": "system",
        "content": (
            "Вы инженер по нагрузочному тестированию. Должны проанализирвать результаты ступенчатого нагрузочного теста поиска максимальной производительности."
            "Пользователь предоставит данные и вопрос. "
            "Используйте контекст этих данных, чтобы ответить на его вопрос."
        )
    }
    user_message = {
        "role": "user",
        "content": user_prompt + f"\n\n{data_context}"
    }
    messages = [system_message, user_message]

    if llm_config is None:
        # Совместимость со старым интерфейсом (OpenAI-совместимые API)
        llm_config = {
            "provider": "openai_compatible",
            "api_key": api_key,
            "model": model or "gpt-4o-mini",
            "base_url": base_url,
        }

    provider = (llm_config.get("provider") or "openai_compatible").lower()

    if provider == "gigachat":
        # Используем официальную библиотеку gigachat
        return _ask_gigachat(messages, llm_config)

    # Иначе используем OpenAI-совместимый клиент
    client, final_model = _create_openai_compatible_client(llm_config)

    base_for_log = (llm_config.get("gigachat", {}).get("api_base_url") if provider == "gigachat" else llm_config.get("base_url"))
    logger.info(f"LLM request: provider={provider} model={final_model} base_url={base_for_log} user_prompt_len={len(user_prompt)} data_context_len={len(data_context)}")

    try:
        response = client.chat.completions.create(
            model=final_model,
            messages=messages,
            stream=False,
            temperature=0,
            top_p=0.7
        )
    except Exception as e:
        logger.error(f"LLM request failed: provider={provider} model={final_model} base_url={base_for_log} error={e}")
        raise

    llm_answer = response.choices[0].message.content
    return llm_answer


def read_prompt_from_file(filename: str) -> str:
    """Считывает текст промта из файла с учетом кодировки UTF-8."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()




def uploadFromLLM(start_ts, end_ts):

    # 1. Читаем конфигурацию
    _configure_logging()
    prometheus_url = CONFIG["prometheus"]["url"]
    src_type = (CONFIG.get("metrics_source", {}).get("type") or "prometheus").lower()
    if src_type == "grafana_proxy":
        g = CONFIG.get("metrics_source", {}).get("grafana", {})
        logger.info(f"Metrics source: Grafana proxy base_url={g.get('base_url')} auth_method={(g.get('auth', {}).get('method'))} verify_ssl={g.get('verify_ssl')} ds_hint={g.get('prometheus_datasource')}")
    else:
        logger.info(f"Metrics source: Prometheus url={prometheus_url}")
    
    llm_config = CONFIG["llm"]

    step = CONFIG["default_params"]["step"]
    resample = CONFIG["default_params"]["resample_interval"]

    # 2. Получаем метрики JVM
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

    # 3. Аналогично для ArangoDB
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

    # 4. Аналогично для Kafka
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

    # 5. Аналогично для Microservices
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

    # 6. Читаем промты из файлов
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # путь к папке, где лежит main.py
    prompts_dir = os.path.join(CURRENT_DIR, "prompts")
    
    prompt_jvm = read_prompt_from_file(os.path.join(prompts_dir, "jvm_prompt.txt"))
    prompt_arangodb = read_prompt_from_file(os.path.join(prompts_dir, "arangodb_prompt.txt"))
    prompt_kafka = read_prompt_from_file(os.path.join(prompts_dir, "kafka_prompt.txt"))
    prompt_microservices = read_prompt_from_file(os.path.join(prompts_dir, "microservices_prompt.txt"))
    prompt_overall = read_prompt_from_file(os.path.join(prompts_dir, "overall_prompt.txt"))

    # Создаем полные текстовые представления данных
    jvm_full_data = dataframes_to_markdown(labeled_jvm)
    arangodb_full_data = dataframes_to_markdown(labeled_arangodb)
    kafka_full_data = dataframes_to_markdown(labeled_kafka)
    ms_full_data = dataframes_to_markdown(labeled_ms)

    # 7. Запрашиваем ответы у LLM (используя полные текстовые представления данных)
    answer_jvm = ask_llm_with_text_data(
        user_prompt=prompt_jvm,
        data_context=jvm_full_data,
        llm_config=llm_config
    )

    answer_arangodb = ask_llm_with_text_data(
        user_prompt=prompt_arangodb,
        data_context=arangodb_full_data,
        llm_config=llm_config
    )

    answer_kafka = ask_llm_with_text_data(
        user_prompt=prompt_kafka,
        data_context=kafka_full_data,
        llm_config=llm_config
    )

    answer_ms = ask_llm_with_text_data(
        user_prompt=prompt_microservices,
        data_context=ms_full_data,
        llm_config=llm_config
    )

    # 8. Формируем сводный отчёт (используем готовые ответы)
    merged_prompt_overall = (
        prompt_overall
        .replace("{answer_jvm}", answer_jvm)
        .replace("{answer_arangodb}", answer_arangodb)
        .replace("{answer_kafka}", answer_kafka)
        .replace("{answer_microservices}", answer_ms)
    )

    final_answer = ask_llm_with_text_data(
        user_prompt=merged_prompt_overall,
        data_context="",  # не передаём датафреймы, т.к. контекст уже в answer_*
        llm_config=llm_config
    )

    # Возвращаем результаты, включая полные данные таблиц
    return {
        "jvm": f"{jvm_full_data}\n\nАнализ JVM:\n{answer_jvm}",
        "arangodb": f"{arangodb_full_data}\n\nАнализ ArangoDB:\n{answer_arangodb}",
        "kafka": f"{kafka_full_data}\n\nАнализ Kafka:\n{answer_kafka}",
        "ms": f"{ms_full_data}\n\nАнализ микросервисов:\n{answer_ms}",
        "final": final_answer
    }


