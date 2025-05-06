import base64
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from atlassian import Confluence
from datetime import datetime
from getpass import getpass

from requests.auth import HTTPBasicAuth



def copy_confluence_page(url, username, password ,page_id, page_parent_id):
    confluence = Confluence(
        url=url,
        username=username,
        password=password,
        verify_ssl=False
    )
    # Загружаем страницу
    try:
        page = confluence.get_page_by_id(page_id, expand='body.storage,history,space,version', status=None, version=None)
    except Exception as e:
        print(f"Ошибка при загрузке страницы: {e}")
        return

    date = datetime.now().strftime("%Y-%m-%d %H:%M")


    # Создаем новую страницу
    new_page = {
        "type": "page",
        "title": page["title"] + " - отчет " + str(date),
        "space": {
            "key": page["space"]["key"]
        },
        "body": {
            "storage": {
                "value": page["body"]["storage"]["value"],
                "representation": "storage"
            }
        },
        "version": {
            "number": 1
        }
    }

    # Пытаемся создать новую страницу
    try:
        new_page = confluence.create_page(space=new_page["space"]["key"], title=new_page["title"],
                                          body=new_page["body"]["storage"]["value"], parent_id=page_parent_id)
    except Exception as e:
        print(f"Ошибка при создании новой страницы: {e}")
        return

    print("Новая страница успешно создана.")
    return new_page["id"]





def update_confluence_page_old(url, username, password, page_id, data_to_find, replace_text):
    confluence = Confluence(
        url=url,
        username=username,
        password=password,
        verify_ssl=False
    )
    # Загружаем страницу
    try:
        page = confluence.get_page_by_id(page_id, expand='body.storage,history,space,version', status=None, version=None)
    except Exception as e:
        print(f"Ошибка при загрузке страницы: {e}")
        return

    # Если replace_text не строка, преобразуем в строку (например, для DataFrame)
    replace_content = str(replace_text) if not isinstance(replace_text, str) else replace_text
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Заменяем данные на странице
    page["body"]["storage"]["value"] = page["body"]["storage"]["value"].replace(str(data_to_find), replace_content)
    
    page["version"]["number"] += 1

    # Пытаемся загрузить обновленную страницу
    try:
        confluence.update_page(
            page_id=page["id"],
            title=page["title"],
            body=page["body"]["storage"]["value"],
            minor_edit=True
        )
    except Exception as e:
        print(f"Ошибка при обновлении страницы: {e}")
        return

    print("Страница успешно обновлена.")
  
def update_confluence_page(url, username, password, page_id, data_to_find, replace_text):
    confluence = Confluence(
        url=url,
        username=username,
        password=password,
        verify_ssl=False
    )
    
    try:
        page = confluence.get_page_by_id(page_id, expand='body.storage,history,space,version')
    except Exception as e:
        print(f"Ошибка при загрузке страницы: {e}")
        return
    
    # Преобразуем replace_text в строку, если это не строка
    replace_content = str(replace_text) if not isinstance(replace_text, str) else replace_text
    
    # Получаем исходное содержимое страницы
    original_html = page["body"]["storage"]["value"]
    
    # Проверка наличия плейсхолдера перед заменой
    if data_to_find not in original_html:
        print(f"ВНИМАНИЕ: Плейсхолдер '{data_to_find}' не найден на странице!")
        return "Плейсхолдер не найден"
    
    # Простая замена без использования BeautifulSoup
    modified_html = original_html.replace(str(data_to_find), replace_content)
    
    # Обновление версии
    page["version"]["number"] += 1
    
    # Обновление страницы с измененным содержимым
    try:
        confluence.update_page(
            page_id=page["id"],
            title=page["title"],
            body=modified_html,
            minor_edit=True
        )
        print("Страница успешно обновлена.")
        return "Успешно"
    except Exception as e:
        print(f"Ошибка при обновлении страницы: {e}")
        return f"Ошибка: {e}"





