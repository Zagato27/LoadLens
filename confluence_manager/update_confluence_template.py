import base64
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from atlassian import Confluence
from datetime import datetime
from getpass import getpass

from requests.auth import HTTPBasicAuth
import re



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
    
    # Обновление страницы с измененным содержимым
    try:
        confluence.update_page(
            page_id=page["id"],
            title=page["title"],
            body=modified_html,
            type='page',
            representation='storage',
            minor_edit=True
        )
        print("Страница успешно обновлена.")
        return "Успешно"
    except Exception as e:
        print(f"Ошибка при обновлении страницы: {e}")
        return f"Ошибка: {e}"


def render_llm_report_placeholders(report: dict) -> dict:
    """Формирует словарь {placeholder: html} с фолбэками из структурированного отчёта LLM.
    Ожидается структура {verdict, confidence, findings[], recommended_actions[]}.
    При отсутствии полей подставляются безопасные значения.
    """
    safe = lambda x: str(x).strip() if x is not None else ""

    verdict = safe((report or {}).get("verdict") or "нет данных")
    conf_val = (report or {}).get("confidence")
    confidence_str = f"{int(conf_val * 100)}%" if isinstance(conf_val, (int, float)) else "—"

    findings = (report or {}).get("findings") or []
    items = []
    for f in findings:
        if isinstance(f, dict):
            summary = safe(f.get("summary"))
            sev = safe(f.get("severity"))
            comp = safe(f.get("component"))
            ev = safe(f.get("evidence"))
            meta = []
            if sev:
                meta.append(f"<span style='color:#b00'><strong>{sev}</strong></span>")
            if comp:
                meta.append(f"<code>{comp}</code>")
            if ev:
                meta.append(f"<em>{ev}</em>")
            meta_str = (" &middot; ".join(meta)) if meta else ""
            if summary or meta_str:
                items.append(f"<li>{summary} {('— ' + meta_str) if meta_str else ''}</li>")
        else:
            s = safe(f)
            if s:
                items.append(f"<li>{s}</li>")
    findings_html = "<ul>" + "".join(items) + "</ul>" if items else "<em>Нет существенных находок</em>"

    actions = (report or {}).get("recommended_actions") or (report or {}).get("actions") or []
    aitems = []
    for a in actions:
        s = safe(a)
        if s:
            aitems.append(f"<li>{s}</li>")
    actions_html = "<ul>" + "".join(aitems) + "</ul>" if aitems else "<em>Нет рекомендаций</em>"

    affected = (report or {}).get("affected_components") or []
    affected_html = ""
    if affected:
        affected_html = "<p><strong>Затронутые компоненты:</strong> " + ", ".join([f"<code>{safe(a)}</code>" for a in affected]) + "</p>"

    return {
        "${LLM_VERDICT}": f"<strong>{verdict}</strong>",
        "${LLM_CONFIDENCE}": confidence_str,
        "${LLM_FINDINGS}": affected_html + findings_html,
        "${LLM_ACTIONS}": actions_html,
    }


def render_llm_markdown(report: dict) -> str:
    """Генерирует markdown для единого плейсхолдера `$$answer_llm$$`.
    Форматирует: вердикт/доверие, список находок с метаданными, список действий.
    """
    def safe(x):
        return str(x).strip() if x is not None else ""

    verdict = safe((report or {}).get("verdict") or "нет данных")
    conf_val = (report or {}).get("confidence")
    confidence_str = f"{int(conf_val*100)}%" if isinstance(conf_val, (int, float)) else "—"

    md_lines = []
    md_lines.append("### Итог LLM")
    md_lines.append(f"- Вердикт: {verdict}")
    md_lines.append(f"- Доверие: {confidence_str}")
    md_lines.append("")

    # Пиковая производительность (опционально)
    peak = (report or {}).get("peak_performance") or (report or {}).get("peak_perfomance")
    if isinstance(peak, dict):
        max_rps = safe(peak.get("max_rps"))
        max_time = safe(peak.get("max_time"))
        drop_time = safe(peak.get("drop_time"))
        method = safe(peak.get("method"))
        if any([max_rps, max_time, drop_time, method]):
            md_lines.append("#### Пиковая производительность")
            if max_rps:
                md_lines.append(f"- Максимальный RPS: {max_rps}")
            if max_time:
                md_lines.append(f"- Время пика: {max_time}")
            if drop_time:
                md_lines.append(f"- Время деградации: {drop_time}")
            if method:
                md_lines.append(f"- Метод оценки: {method}")
            md_lines.append("")

    findings = (report or {}).get("findings") or []
    md_lines.append("#### Ключевые находки")
    if not findings:
        md_lines.append("- Нет существенных находок")
    else:
        for f in findings:
            if isinstance(f, dict):
                summary = safe(f.get("summary"))
                sev = safe(f.get("severity"))
                comp = safe(f.get("component"))
                ev = safe(f.get("evidence"))
                meta = []
                if sev:
                    meta.append(f"severity: {sev}")
                if comp:
                    meta.append(f"component: {comp}")
                if ev:
                    meta.append(f"evidence: {ev}")
                meta_str = "; ".join(meta)
                if meta_str:
                    md_lines.append(f"- {summary} ({meta_str})")
                else:
                    md_lines.append(f"- {summary}")
            else:
                md_lines.append(f"- {safe(f)}")

    actions = (report or {}).get("recommended_actions") or (report or {}).get("actions") or []
    md_lines.append("")
    md_lines.append("#### Рекомендации")
    if not actions:
        md_lines.append("- Нет рекомендаций")
    else:
        for a in actions:
            s = safe(a)
            if s:
                md_lines.append(f"- {s}")

    affected = (report or {}).get("affected_components") or []
    if affected:
        md_lines.append("")
        md_lines.append("#### Затронутые компоненты")
        md_lines.append(", ".join([f"`{safe(x)}`" for x in affected]))

    return "\n".join(md_lines)


def render_llm_html(report: dict) -> str:
    """HTML-версия рендера LLM-ответа для вставки в Confluence (storage)."""
    def safe(x: object) -> str:
        return str(x).strip() if x is not None else ""

    verdict = safe((report or {}).get("verdict") or "нет данных")
    conf_val = (report or {}).get("confidence")
    confidence_str = f"{int(conf_val*100)}%" if isinstance(conf_val, (int, float)) else "—"

    parts: list[str] = []
    parts.append("<h3>Итог LLM</h3>")
    parts.append("<ul>")
    parts.append(f"<li><strong>Вердикт:</strong> {verdict}</li>")
    parts.append(f"<li><strong>Доверие:</strong> {confidence_str}</li>")
    parts.append("</ul>")

    peak = (report or {}).get("peak_performance") or (report or {}).get("peak_perfomance")
    if isinstance(peak, dict):
        max_rps = safe(peak.get("max_rps"))
        max_time = safe(peak.get("max_time"))
        drop_time = safe(peak.get("drop_time"))
        method = safe(peak.get("method"))
        if any([max_rps, max_time, drop_time, method]):
            parts.append("<h4>Пиковая производительность</h4>")
            parts.append("<ul>")
            if max_rps:
                parts.append(f"<li>Максимальный RPS: {max_rps}</li>")
            if max_time:
                parts.append(f"<li>Время пика: {max_time}</li>")
            if drop_time:
                parts.append(f"<li>Время деградации: {drop_time}</li>")
            if method:
                parts.append(f"<li>Метод оценки: {method}</li>")
            parts.append("</ul>")

    findings = (report or {}).get("findings") or []
    parts.append("<h4>Ключевые находки</h4>")
    if not findings:
        parts.append("<p><em>Нет существенных находок</em></p>")
    else:
        parts.append("<ul>")
        for f in findings:
            if isinstance(f, dict):
                summary = safe(f.get("summary"))
                sev = safe(f.get("severity"))
                comp = safe(f.get("component"))
                ev = safe(f.get("evidence"))
                meta = []
                if sev:
                    meta.append(f"severity: {sev}")
                if comp:
                    meta.append(f"component: {comp}")
                if ev:
                    meta.append(f"evidence: {ev}")
                meta_str = "; ".join(meta)
                text = f"{summary} ({meta_str})" if meta_str else summary
                parts.append(f"<li>{text}</li>")
            else:
                s = safe(f)
                if s:
                    parts.append(f"<li>{s}</li>")
        parts.append("</ul>")

    actions = (report or {}).get("recommended_actions") or (report or {}).get("actions") or []
    parts.append("<h4>Рекомендации</h4>")
    if not actions:
        parts.append("<p><em>Нет рекомендаций</em></p>")
    else:
        parts.append("<ul>")
        for a in actions:
            s = safe(a)
            if s:
                parts.append(f"<li>{s}</li>")
        parts.append("</ul>")

    affected = (report or {}).get("affected_components") or []
    if affected:
        parts.append("<h4>Затронутые компоненты</h4>")
        codes = ", ".join([f"<code>{safe(x)}</code>" for x in affected])
        parts.append(f"<p>{codes}</p>")

    return "\n".join(parts)


def update_confluence_page_multi(url, username, password, page_id, replacements: dict) -> str:
    """Один проход по странице: заменить несколько плейсхолдеров. Отсутствующие не считаем ошибкой."""
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
        return "Ошибка загрузки"

    html = page["body"]["storage"]["value"]
    replaced_any = False
    for placeholder, value in (replacements or {}).items():
        if not isinstance(value, str) or not value.strip():
            print(f"[warn] Пропускаю пустую замену для: {placeholder}")
            continue
        if placeholder in html:
            html = html.replace(str(placeholder), str(value))
            replaced_any = True
        else:
            # Попробуем более гибкую замену с допуском пробелов внутри $$...$$
            did_flexible = False
            try:
                ph = str(placeholder)
                if ph.startswith("$$") and ph.endswith("$$"):
                    inner = ph[2:-2].strip()
                    if inner:
                        pattern = r"\$\$\s*" + re.escape(inner) + r"\s*\$\$"
                        new_html, num = re.subn(pattern, str(value), html)
                        if num > 0:
                            html = new_html
                            replaced_any = True
                            did_flexible = True
            except Exception as e:
                print(f"[warn] Ошибка при гибкой замене '{placeholder}': {e}")
            if not did_flexible:
                print(f"[warn] Плейсхолдер '{placeholder}' не найден. Пропускаю.")

    if not replaced_any:
        print("Нет совпавших плейсхолдеров. Обновление не требуется.")
        return "Нет замен"

    try:
        confluence.update_page(
            page_id=page["id"],
            title=page["title"],
            body=html,
            type='page',
            representation='storage',
            minor_edit=True
        )
        print("Страница успешно обновлена (мульти-замена).")
        return "Успешно"
    except Exception as e:
        print(f"Ошибка при обновлении страницы: {e}")
        return f"Ошибка: {e}"





