# main.py

import requests
import pandas as pd
from typing import List, Dict
from openai import OpenAI
import os
from datetime import datetime
from atlassian import Confluence
from datetime import datetime
from getpass import getpass
from bs4 import BeautifulSoup
from tabulate import tabulate

from requests.auth import HTTPBasicAuth


# Импортируем CONFIG из config.py
from AI.config import CONFIG


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

    response = requests.get(
        f'{prometheus_url}/api/v1/query_range',
        params=params,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


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
        data_json = fetch_prometheus_data(
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





def ask_llm_with_text_data(
    user_prompt: str,
    data_context: str,
    api_key: str,
    model: str = "sonar-deep-research",
    base_url: str = "https://api.perplexity.ai"
) -> str:
    """
    Отправляет запрос к модели, используя предварительно форматированные текстовые данные.
    
    Args:
        user_prompt: Пользовательский запрос к модели
        data_context: Полное текстовое представление данных
        api_key: API ключ для доступа к модели
        model: Название модели
        base_url: Базовый URL для API запросов
        
    Returns:
        Ответ от модели
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

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=0,
        top_p=0.7

    )

    llm_answer = response.choices[0].message.content
    return llm_answer


def read_prompt_from_file(filename: str) -> str:
    """Считывает текст промта из файла с учетом кодировки UTF-8."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()




def uploadFromDeepSeek(start_ts, end_ts):

    # 1. Читаем конфигурацию
    prometheus_url = CONFIG["prometheus"]["url"]
    
    deepseek_api_key = CONFIG["llm"]["api_key"]
    llm_model = CONFIG["llm"]["model"]
    llm_base_url = CONFIG["llm"]["base_url"]

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
        api_key=deepseek_api_key,
        model=llm_model,
        base_url=llm_base_url
    )

    answer_arangodb = ask_llm_with_text_data(
        user_prompt=prompt_arangodb,
        data_context=arangodb_full_data,
        api_key=deepseek_api_key,
        model=llm_model,
        base_url=llm_base_url
    )

    answer_kafka = ask_llm_with_text_data(
        user_prompt=prompt_kafka,
        data_context=kafka_full_data,
        api_key=deepseek_api_key,
        model=llm_model,
        base_url=llm_base_url
    )

    answer_ms = ask_llm_with_text_data(
        user_prompt=prompt_microservices,
        data_context=ms_full_data,
        api_key=deepseek_api_key,
        model=llm_model,
        base_url=llm_base_url
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
        api_key=deepseek_api_key,
        model=llm_model,
        base_url=llm_base_url
    )

    # Возвращаем результаты, включая полные данные таблиц
    return {
        "jvm": f"{jvm_full_data}\n\nАнализ JVM:\n{answer_jvm}",
        "arangodb": f"{arangodb_full_data}\n\nАнализ ArangoDB:\n{answer_arangodb}",
        "kafka": f"{kafka_full_data}\n\nАнализ Kafka:\n{answer_kafka}",
        "ms": f"{ms_full_data}\n\nАнализ микросервисов:\n{answer_ms}",
        "final": final_answer
    }


