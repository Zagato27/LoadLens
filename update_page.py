from confluence_manager.update_confluence_template import copy_confluence_page, update_confluence_page, update_confluence_page_multi, render_llm_report_placeholders
from AI.main import uploadFromLLM

from concurrent.futures import ThreadPoolExecutor, as_completed
from data_collectors.grafana_collector import uploadFromGrafana
from data_collectors.loki_collector import uploadFromLoki
from config import CONFIG  # Импорт базовой конфигурации
from metrics_config import METRICS_CONFIG  # Импорт конфигурации метрик
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime  # если ещё не импортирован
import traceback  # для детального вывода ошибок (опционально)


def update_report(start, end, service):
    # Получение параметров из `config.py`
    user = CONFIG['user']
    password = CONFIG['password']
    grafana_login = CONFIG['grafana_login']
    grafana_pass = CONFIG['grafana_pass']
    url_basic = CONFIG['url_basic']
    space_conf = CONFIG['space_conf']
    grafana_base_url = CONFIG['grafana_base_url']
    loki_url = CONFIG['loki_url']

    
    # Получаем конфигурацию сервиса и проверяем наличие метрик
    service_config = METRICS_CONFIG.get(service)
    if not service_config:
        raise ValueError(f"Конфигурация для сервиса '{service}' не найдена.")
    
    # Получаем `page_sample_id` и `page_parent_id` из конфигурации сервиса
    page_parent_id = service_config["page_parent_id"]
    page_sample_id = service_config["page_sample_id"]
    copy_page_id = copy_confluence_page(url_basic, user, password, page_sample_id, page_parent_id)
   

    # Список задач для обновлений
    tasks = []
    
                
    # Добавим функцию обновления с повторными попытками
    def update_with_retry(url, username, password, page_id, data_to_find, replace_text, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                res = update_confluence_page(url, username, password, page_id, data_to_find, replace_text)
                # Обработка текстовых ошибок из update_confluence_page
                if isinstance(res, str) and (res.startswith("Ошибка") or res == "Плейсхолдер не найден"):
                    raise RuntimeError(res)
                return res
            except Exception as e:
                if ("Attempted to update stale data" in str(e) or "conflict" in str(e).lower()) and attempt < max_attempts-1:
                    print(f"Попытка {attempt+1} не удалась, повторяем через 1 секунду...")
                    time.sleep(1)
                elif attempt < max_attempts-1:
                    print(f"Попытка {attempt+1} не удалась: {e}. Повтор через 1 секунду...")
                    time.sleep(1)
                else:
                    raise e

    # Основной код с разделением параллельных и последовательных операций
    metrics_logs_tasks = []

    with ThreadPoolExecutor() as executor:
        # Добавляем задачи для каждой метрики, указанной для выбранного сервиса
        for metric in service_config["metrics"]:
            # Формируем полный URL для метрики с учетом базового URL Grafana и временного диапазона
            grafana_url = f"{grafana_base_url}{metric['grafana_url']}&from={start}&to={end}"
            name = metric['name']

            # Добавляем задачу для загрузки метрики и обновления Confluence
            metrics_logs_tasks.append(executor.submit(
                update_confluence_page,
                url_basic, user, password, copy_page_id,
                f"$${name}$$",
                uploadFromGrafana(user, password, url_basic, space_conf, copy_page_id, [[name, grafana_url]], service, grafana_login, grafana_pass)
            ))

        # Добавление задач для логов, если они есть в конфигурации сервиса
        for log in service_config.get("logs", []):
            placeholder = log["placeholder"]
            filter_query = log["filter_query"]

            metrics_logs_tasks.append(executor.submit(
                update_confluence_page,
                url_basic, user, password, copy_page_id,
                f"$${placeholder}$$",
                uploadFromLoki(loki_url, start, end, filter_query, user, password, url_basic, copy_page_id, service, placeholder)
            ))
        
        # Обработка результатов метрик и логов
        for future in as_completed(metrics_logs_tasks):
            try:
                result = future.result()
                print(f"Метрика/лог обновлены: {result}")
            except Exception as e:
                print(f"Ошибка при обновлении метрики/лога: {e}")

    # Получаем результаты LLM и обновляем их последовательно
    results = uploadFromLLM(start/1000, end/1000)

    # Мульти-обновление плейсхолдеров LLM за один проход
    try:
        print("Обновление данных LLM (одним проходом)...")
        llm_replacements = {
            "$$final_answer$$": results["final"],
            "$$answer_jvm$$": results["jvm"],
            "$$answer_arangodb$$": results["arangodb"],  # backward compat
            "$$answer_database$$": results.get("arangodb", ""),
            "$$answer_kafka$$": results["kafka"],
            "$$answer_ms$$": results["ms"],
        }

        # Добавляем структурированные плейсхолдеры, если есть JSON
        final_struct = results.get("final_parsed")
        if final_struct:
            llm_replacements.update(render_llm_report_placeholders(final_struct))

        update_confluence_page_multi(url_basic, user, password, copy_page_id, llm_replacements)
        print("✓ Плейсхолдеры LLM обновлены за один проход")
    except Exception as e:
        print(f"Ошибка при мульти-обновлении данных LLM: {e}")

