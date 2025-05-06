import requests
from datetime import datetime
import os
import shutil
from requests.auth import HTTPBasicAuth

# Функция для отправки логов как вложения на Confluence
def send_loki_file_to_attachment(url_basic, auth, page_id, file_path):
    """
    Отправка файла логов как вложения на страницу Confluence.
    """
    try:
        url = f"{url_basic}/rest/api/content/{page_id}/child/attachment"
        print(url)

        # Проверка существования файла перед отправкой
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден.")
            return None

        # Загрузка файла как вложения
        with open(file_path, 'rb') as file:
            files = {"file": file}
            response = requests.post(
                url,
                files=files,
                auth=auth,
                headers={'X-Atlassian-Token': 'nocheck'},
                verify=False
            )

        # Проверка ответа от Confluence
        if response.status_code == 200 or response.status_code == 201:
            print(f"Файл {file_path} успешно отправлен на Confluence.")
            return response
        else:
            print(f"Ошибка при отправке файла: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Произошла ошибка при отправке файла: {str(e)}")
        return None


# Функция для получения логов из Loki и их сохранения в файл
def fetch_loki_logs(loki_url, start_timestamp, end_timestamp, filter_query, filename):
    """
    Получение логов из Loki и сохранение их в файл.
    """
    # Конвертация меток времени в наносекунды
    start_ns = int(str(start_timestamp)[:-3])
    end_ns = int(str(end_timestamp)[:-3])

    # Параметры запроса
    params = {
        'query': filter_query,
        'start': start_ns,
        'end': end_ns,
        'limit': 1000
    }

    # Отправка запроса на Loki
    response = requests.get(loki_url, params=params)

    # Обработка ответа от Loki
    if response.status_code == 200:
        log_data = response.json()
        log_entries = []
        for stream in log_data['data']['result']:
            for entry in stream['values']:
                timestamp, log = entry
                log_entries.append(f"{datetime.fromtimestamp(int(timestamp) / 1e9)} - {log}")

        # Сохранение логов в файл
        file_path = f'data_collectors/temporary_files/{filename}.log'

        with open(file_path, 'w', encoding='utf-8') as file:
            for log_entry in log_entries:
                file.write(log_entry + "\n")
        print(f"Логи сохранены в {file_path}")
        return file_path
    else:
        print(f"Ошибка при получении логов из Loki: {response.status_code} - {response.text}")
        return None


# Основная функция для загрузки логов на Confluence
def uploadFromLoki(loki_url, start_timestamp, end_timestamp, filter_query, user, password, url_basic, page_id, service, microservice):
    """
    Загрузка логов из Loki и отправка их в виде вложения на Confluence.
    """
    try:
        # Получение логов из Loki и сохранение в файл
        filename = f"{service}_{microservice}_{page_id}"
        file_path = fetch_loki_logs(loki_url, start_timestamp, end_timestamp, filter_query, filename)
        
        if file_path is None:
            print("Логи не были получены.")
            return ""

        # Отправка файла на Confluence
        auth = HTTPBasicAuth(user, password)
        response = send_loki_file_to_attachment(url_basic, auth, page_id, file_path)

        if response:
            print("Вложение отправлено на страницу Confluence.")
            # Формирование разметки для отображения вложения
            utils = f'<ac:structured-macro ac:name="view-file" ac:schema-version="1"><ac:parameter ac:name="name"><ri:attachment ri:filename="{filename}.log" /></ac:parameter><ac:parameter ac:name="height">250</ac:parameter></ac:structured-macro>'
        else:
            utils = ""
        
        # Удаление временного файла после загрузки
        os.remove(file_path)
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        utils = ""

    return utils
