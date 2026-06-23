import base64
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from atlassian import Confluence
from datetime import datetime
from getpass import getpass
from html import escape

from requests.auth import HTTPBasicAuth
import re



def copy_confluence_page(url, username, password ,page_id, page_parent_id):
    """Копирует страницу Confluence и возвращает идентификатор новой копии.

    Параметры:
        url (str): Базовый URL Confluence.
        username/password (str): Учётные данные.
        page_id (str): Идентификатор шаблонной страницы.
        page_parent_id (str): Родитель для новой страницы.
    """
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
    """Историческая версия обновления страницы (для обратной совместимости)."""
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
    """Заменяет одиночный плейсхолдер на странице Confluence.

    Параметры аналогичны `copy_confluence_page`, плюс:
        data_to_find (str): Плейсхолдер.
        replace_text (str): HTML/Storage, который нужно подставить.
    """
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
    
    modified_html, _ = _replace_placeholder_storage(original_html, str(data_to_find), replace_content)
    
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


def _safe_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _html(value: object) -> str:
    return escape(_safe_text(value), quote=True)


def _md(value: object) -> str:
    text = _safe_text(value)
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _md_cell(value: object) -> str:
    text = _md(value)
    text = re.sub(r"\s*\n+\s*", " / ", text)
    return text or "—"


def _md_rich_text(value: object) -> str:
    raw = _safe_text(value)
    if not raw:
        return ""
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+[.)]\s+", stripped):
            lines.append(_md(stripped))
        else:
            lines.append(_md(stripped))
    return "\n".join(lines).strip()


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value:
        return [value]
    return []


def _normalize_link_id(value: object, fallback: str = "") -> str:
    raw = _safe_text(value).lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or fallback


def _normalize_link_ids(value: object) -> list[str]:
    result: list[str] = []
    for item in _as_list(value):
        normalized = _normalize_link_id(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _render_rich_text(text: object) -> str:
    """Минимальный markdown-like рендер для Confluence storage."""
    raw = _safe_text(text)
    if not raw:
        return ""
    parts: list[str] = []
    list_items: list[str] = []

    def flush_list() -> None:
        if list_items:
            parts.append("<ul>" + "".join(list_items) + "</ul>")
            list_items.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_list()
            continue
        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        numbered_match = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if bullet_match or numbered_match:
            item_text = (bullet_match or numbered_match).group(1)
            list_items.append(f"<li>{_html(item_text)}</li>")
        else:
            flush_list()
            parts.append(f"<p>{_html(stripped)}</p>")
    flush_list()
    return "".join(parts)


def _verdict_pill(verdict: object) -> str:
    text = _safe_text(verdict) or "Недостаточно данных"
    lower = text.lower()
    if "усп" in lower:
        style = "background:#e3fcef;color:#006644;border:1px solid #57d9a3;"
    elif "риск" in lower:
        style = "background:#fff7d6;color:#974f0c;border:1px solid #ffab00;"
    elif "провал" in lower or "fail" in lower:
        style = "background:#ffebe6;color:#bf2600;border:1px solid #ff7452;"
    else:
        style = "background:#f4f5f7;color:#42526e;border:1px solid #dfe1e6;"
    return (
        f"<span style=\"display:inline-block;padding:4px 10px;border-radius:12px;"
        f"font-weight:600;{style}\">{_html(text)}</span>"
    )


def _render_key_value_table(title: str, rows: list[tuple[str, str]]) -> str:
    body = []
    for label, value_html in rows:
        value = value_html or "<span style=\"color:#6b778c;\">—</span>"
        body.append(
            "<tr>"
            f"<th style=\"width:42%;text-align:left;vertical-align:middle;background:#f4f5f7;"
            f"border:1px solid #dfe1e6;padding:8px;\">{_html(label)}</th>"
            f"<td style=\"text-align:right;vertical-align:middle;border:1px solid #dfe1e6;"
            f"padding:8px;\">{value}</td>"
            "</tr>"
        )
    heading = f"<h4>{_html(title)}</h4>" if title else ""
    return (
        f"{heading}<table style=\"width:100%;border-collapse:separate;border-spacing:0;"
        f"border:1px solid #dfe1e6;border-radius:8px;margin:8px 0 14px 0;\">"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _render_rationale_box(report: dict) -> str:
    text = (
        (report or {}).get("verdict_rationale")
        or (report or {}).get("verdict_reason")
        or (report or {}).get("rationale")
        or "Нет пояснения."
    )
    return (
        "<div style=\"border:1px solid #b3d4ff;border-left:4px solid #2684ff;"
        "background:#f4f9ff;border-radius:8px;padding:10px 12px;margin:8px 0 14px 0;\">"
        "<p style=\"margin:0 0 6px 0;\"><strong>Обоснование вердикта</strong></p>"
        f"{_render_rich_text(text)}"
        "</div>"
    )


def _normalize_finding_entry(item: object, idx: int) -> dict:
    raw = item if isinstance(item, dict) else {"summary": _safe_text(item)}
    summary = _safe_text(raw.get("summary") or raw.get("title") or raw.get("text"))
    components: list[str] = []
    component = _safe_text(raw.get("component")).lower()
    if component:
        components.append(component)
    for comp in _as_list(raw.get("affected_components")):
        normalized = _safe_text(comp).lower()
        if normalized and normalized not in components:
            components.append(normalized)
    fallback = f"finding_{idx + 1}"
    return {
        "idx": idx,
        "id": _normalize_link_id(raw.get("id") or raw.get("finding_id") or raw.get("key"), fallback),
        "summary": summary,
        "components": components,
        "item": raw,
    }


def _normalize_action_entry(item: object, idx: int) -> dict:
    raw = item if isinstance(item, dict) else {"summary": _safe_text(item)}
    summary = _safe_text(raw.get("summary") or raw.get("action") or raw.get("text"))
    components: list[str] = []
    component = _safe_text(raw.get("component")).lower()
    if component:
        components.append(component)
    for comp in _as_list(raw.get("affected_components")):
        normalized = _safe_text(comp).lower()
        if normalized and normalized not in components:
            components.append(normalized)
    link_ids = _normalize_link_ids(
        raw.get("for_finding_ids")
        or raw.get("for_findings")
        or raw.get("finding_ids")
        or raw.get("related_findings")
        or raw.get("for_finding_id")
        or raw.get("finding_id")
    )
    return {
        "idx": idx,
        "summary": summary,
        "components": components,
        "for_finding_ids": link_ids,
        "item": raw,
    }


def _match_legacy_action(finding: dict, actions: list[dict], unused_ids: set[int]) -> dict | None:
    legacy = [a for a in actions if not a["for_finding_ids"] and a["idx"] in unused_ids]
    matched = None
    if finding["components"]:
        matched = next(
            (a for a in legacy if any(c in finding["components"] for c in a["components"])),
            None,
        )
    if not matched and finding["idx"] in unused_ids:
        matched = next((a for a in legacy if a["idx"] == finding["idx"]), None)
    if not matched:
        matched = legacy[0] if legacy else None
    if matched:
        unused_ids.discard(matched["idx"])
    return matched


def _pair_findings_with_actions(findings: object, actions: object) -> list[dict]:
    finding_entries = [
        item for item in (
            _normalize_finding_entry(f, idx) for idx, f in enumerate(_as_list(findings))
        )
        if item["summary"]
    ]
    action_entries = [
        item for item in (
            _normalize_action_entry(a, idx) for idx, a in enumerate(_as_list(actions))
        )
        if item["summary"]
    ]
    linked_ids = {item["id"] for item in finding_entries}
    unused_legacy = {item["idx"] for item in action_entries if not item["for_finding_ids"]}
    rows: list[dict] = []
    for finding in finding_entries:
        explicit = [
            action["item"]
            for action in action_entries
            if finding["id"] in action["for_finding_ids"]
        ]
        legacy = None if explicit else _match_legacy_action(finding, action_entries, unused_legacy)
        rows.append({
            "problem": finding["item"],
            "actions": explicit or ([legacy["item"]] if legacy else []),
        })
    for action in action_entries:
        is_unmatched_legacy = not action["for_finding_ids"] and action["idx"] in unused_legacy
        is_unmatched_explicit = action["for_finding_ids"] and not any(
            finding_id in linked_ids for finding_id in action["for_finding_ids"]
        )
        if is_unmatched_legacy or is_unmatched_explicit:
            rows.append({"problem": None, "actions": [action["item"]]})
    if not rows:
        rows.append({"problem": None, "actions": [a["item"] for a in action_entries]})
    return rows


def _render_evidence(item: dict) -> str:
    evidence_summary = _safe_text(item.get("evidence_summary") or item.get("evidence"))
    evidence_items = _as_list(item.get("evidence_items") or item.get("evidence_list") or item.get("evidence_rows"))
    parts: list[str] = []
    if evidence_summary:
        parts.append(f"<p style=\"margin:6px 0;color:#42526e;\">{_html(evidence_summary)}</p>")
    bullets: list[str] = []
    for evidence in evidence_items:
        if isinstance(evidence, dict):
            bits = [
                _safe_text(evidence.get("metric") or evidence.get("name") or evidence.get("label")),
                f"значение: {_safe_text(evidence.get('observed_value') or evidence.get('value') or evidence.get('actual'))}"
                if _safe_text(evidence.get("observed_value") or evidence.get("value") or evidence.get("actual")) else "",
                f"порог: {_safe_text(evidence.get('threshold') or evidence.get('limit') or evidence.get('baseline'))}"
                if _safe_text(evidence.get("threshold") or evidence.get("limit") or evidence.get("baseline")) else "",
                _safe_text(evidence.get("note") or evidence.get("details") or evidence.get("evidence")),
            ]
            text = " | ".join([b for b in bits if b])
        else:
            text = _safe_text(evidence)
        if text:
            bullets.append(f"<li>{_html(text)}</li>")
    if bullets:
        parts.append("<ul style=\"margin-top:4px;\">" + "".join(bullets) + "</ul>")
    return "".join(parts)


def _render_problem(problem: object) -> str:
    if not isinstance(problem, dict):
        text = _safe_text(problem) or "Нет существенных проблем."
        return f"<div><strong>{_html(text)}</strong></div>"
    summary = _safe_text(problem.get("summary") or problem.get("title") or problem.get("text")) or "—"
    meta = []
    severity = _safe_text(problem.get("severity"))
    component = _safe_text(problem.get("component"))
    if severity:
        meta.append(("Критичность", severity))
    if component:
        meta.append(("Компонент", component))
    meta_html = ""
    if meta:
        chips = "".join(
            f"<span style=\"display:inline-block;background:#f4f5f7;border:1px solid #dfe1e6;"
            f"border-radius:10px;padding:2px 8px;margin:4px 4px 0 0;font-size:12px;\">"
            f"<strong>{_html(label)}:</strong> {_html(value)}</span>"
            for label, value in meta
        )
        meta_html = f"<div style=\"margin-top:6px;\">{chips}</div>"
    return (
        "<div>"
        f"<p style=\"margin:0;\"><strong>{_html(summary)}</strong></p>"
        f"{_render_evidence(problem)}"
        f"{meta_html}"
        "</div>"
    )


def _render_action(action: object) -> str:
    if not isinstance(action, dict):
        text = _safe_text(action)
        return f"<div>{_html(text) if text else '<em>Нет рекомендации.</em>'}</div>"
    summary = _safe_text(action.get("summary") or action.get("action") or action.get("text")) or "—"
    details = _render_rich_text(
        action.get("details") or action.get("description") or action.get("implementation_details")
    )
    return (
        "<div>"
        f"<p style=\"margin:0;\"><strong>{_html(summary)}</strong></p>"
        f"{details}"
        "</div>"
    )


def _render_actions(actions: list) -> str:
    if not actions:
        return "<em>Нет рекомендации.</em>"
    return "".join(
        f"<div style=\"margin-bottom:10px;\">{_render_action(action)}</div>"
        for action in actions
    )


def _is_stability_report(report: dict, peak: dict, test_profile: dict) -> bool:
    test_type = _safe_text(test_profile.get("test_type")).lower()
    return bool(
        peak.get("not_applicable")
        or peak.get("notApplicable")
        or _safe_text(test_profile.get("mode")).lower() == "stability"
        or test_type in {"soak", "stability", "endurance"}
    )


def _render_problem_recommendation_table(report: dict) -> str:
    rows = _pair_findings_with_actions(
        (report or {}).get("findings") or [],
        (report or {}).get("recommended_actions") or (report or {}).get("actions") or [],
    )
    body = []
    for row in rows:
        problem_html = _render_problem(row.get("problem")) if row.get("problem") else "<em>Нет существенных проблем.</em>"
        action_html = _render_actions(row.get("actions") or [])
        body.append(
            "<tr>"
            f"<td style=\"width:50%;vertical-align:top;border:1px solid #dfe1e6;padding:10px;\">{problem_html}</td>"
            f"<td style=\"width:50%;vertical-align:top;border:1px solid #dfe1e6;padding:10px;\">{action_html}</td>"
            "</tr>"
        )
    return (
        "<h4>Проблемы и рекомендации</h4>"
        "<table style=\"width:100%;border-collapse:separate;border-spacing:0;"
        "border:1px solid #dfe1e6;border-radius:8px;margin:8px 0 14px 0;\">"
        "<thead><tr>"
        "<th style=\"text-align:left;background:#f4f5f7;border:1px solid #dfe1e6;padding:8px;\">Проблемы</th>"
        "<th style=\"text-align:left;background:#f4f5f7;border:1px solid #dfe1e6;padding:8px;\">Рекомендации по устранению</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def render_llm_markdown(report: dict) -> str:
    """Возвращает Markdown для Confluence Markdown macro."""
    report = report if isinstance(report, dict) else {}
    peak = report.get("peak_performance") or report.get("peak_perfomance") or {}
    peak = peak if isinstance(peak, dict) else {}
    test_profile = report.get("test_profile") or {}
    test_profile = test_profile if isinstance(test_profile, dict) else {}
    stability = report.get("stability_under_load") or {}
    stability = stability if isinstance(stability, dict) else {}

    lines: list[str] = [
        "| Параметр | Значение |",
        "|---|---:|",
        f"| Вердикт по тесту | **{_md_cell(report.get('verdict') or 'Недостаточно данных')}** |",
        "",
        "#### Обоснование вердикта",
        _md_rich_text(
            report.get("verdict_rationale")
            or report.get("verdict_reason")
            or report.get("rationale")
            or "Нет пояснения."
        ),
        "",
    ]

    if _is_stability_report(report, peak, test_profile):
        lines.extend([
            "#### Стабильность под нагрузкой",
            "| Параметр | Значение |",
            "|---|---:|",
            f"| Тип теста | {_md_cell(test_profile.get('test_type') or 'stability/soak')} |",
            f"| Фокус оценки | {_md_cell(stability.get('focus') or test_profile.get('focus') or 'Удержание нагрузки без накопления деградации')} |",
            f"| Целевой RPS | {_md_cell(stability.get('target_rps') or '—')} |",
            f"| Фактический RPS | {_md_cell(stability.get('actual_rps') or '—')} |",
            f"| Итог SLA | {_md_cell(stability.get('sla_summary') or '—')} |",
            "",
        ])
    else:
        lines.extend([
            "#### Пиковая производительность",
            "| Параметр | Значение |",
            "|---|---:|",
            f"| Максимальный RPS | {_md_cell(peak.get('max_rps') or '—')} |",
            f"| Время пиковой производительности | {_md_cell(peak.get('max_time') or '—')} |",
            f"| Время деградации | {_md_cell(peak.get('drop_time') or '—')} |",
            "",
        ])

    lines.append("#### Проблемы и рекомендации")
    rows = _pair_findings_with_actions(
        report.get("findings") or [],
        report.get("recommended_actions") or report.get("actions") or [],
    )
    if not rows:
        lines.append("Существенных проблем и рекомендаций нет.")
    for idx, row in enumerate(rows, start=1):
        problem = row.get("problem")
        actions = row.get("actions") or []
        if isinstance(problem, dict):
            summary = _safe_text(problem.get("summary") or problem.get("title") or problem.get("text")) or "Нет существенных проблем."
            lines.extend(["", f"##### {idx}. {_md(summary)}"])
            evidence_summary = _md_rich_text(problem.get("evidence_summary") or problem.get("evidence"))
            if evidence_summary:
                lines.extend(["", evidence_summary])
            evidence_items = _as_list(problem.get("evidence_items") or problem.get("evidence_list") or problem.get("evidence_rows"))
            for evidence in evidence_items:
                if isinstance(evidence, dict):
                    bits = [
                        _safe_text(evidence.get("metric") or evidence.get("name") or evidence.get("label")),
                        f"значение: {_safe_text(evidence.get('observed_value') or evidence.get('value') or evidence.get('actual'))}"
                        if _safe_text(evidence.get("observed_value") or evidence.get("value") or evidence.get("actual")) else "",
                        f"порог: {_safe_text(evidence.get('threshold') or evidence.get('limit') or evidence.get('baseline'))}"
                        if _safe_text(evidence.get("threshold") or evidence.get("limit") or evidence.get("baseline")) else "",
                        _safe_text(evidence.get("note") or evidence.get("details") or evidence.get("evidence")),
                    ]
                    evidence_text = " | ".join([bit for bit in bits if bit])
                else:
                    evidence_text = _safe_text(evidence)
                if evidence_text:
                    lines.append(f"- {_md(evidence_text)}")
            meta = []
            if _safe_text(problem.get("severity")):
                meta.append(f"**Критичность:** `{_md(problem.get('severity'))}`")
            if _safe_text(problem.get("component")):
                meta.append(f"**Компонент:** `{_md(problem.get('component'))}`")
            if meta:
                lines.extend(["", "  ".join(meta)])
        else:
            lines.extend(["", f"##### {idx}. Нет существенных проблем."])

        lines.extend(["", "**Рекомендации по устранению:**"])
        if not actions:
            lines.append("- Нет рекомендации.")
        for action in actions:
            if isinstance(action, dict):
                summary = _safe_text(action.get("summary") or action.get("action") or action.get("text")) or "Рекомендация"
                lines.append(f"- **{_md(summary)}**")
                details = _md_rich_text(action.get("details") or action.get("description") or action.get("implementation_details"))
                if details:
                    for detail_line in details.splitlines():
                        lines.append(f"  {detail_line}" if detail_line else "")
            else:
                action_text = _safe_text(action)
                lines.append(f"- {_md(action_text)}" if action_text else "- Нет рекомендации.")

    return "\n".join(lines).strip()


def render_llm_html(report: dict) -> str:
    """HTML-версия рендера LLM-ответа для вставки в Confluence storage."""
    report = report if isinstance(report, dict) else {}
    peak = report.get("peak_performance") or report.get("peak_perfomance") or {}
    peak = peak if isinstance(peak, dict) else {}
    test_profile = report.get("test_profile") or {}
    test_profile = test_profile if isinstance(test_profile, dict) else {}
    stability = report.get("stability_under_load") or {}
    stability = stability if isinstance(stability, dict) else {}

    parts: list[str] = [
        _render_key_value_table(
            "",
            [("Вердикт по тесту", _verdict_pill(report.get("verdict")))],
        ),
        _render_rationale_box(report),
    ]

    if _is_stability_report(report, peak, test_profile):
        parts.append(_render_key_value_table(
            "Стабильность под нагрузкой",
            [
                ("Тип теста", _html(test_profile.get("test_type") or "stability/soak")),
                ("Фокус оценки", _html(stability.get("focus") or test_profile.get("focus") or "Удержание нагрузки без накопления деградации")),
                ("Целевой RPS", _html(stability.get("target_rps") or "—")),
                ("Фактический RPS", _html(stability.get("actual_rps") or "—")),
                ("Итог SLA", _html(stability.get("sla_summary") or "—")),
            ],
        ))
    else:
        parts.append(_render_key_value_table(
            "Пиковая производительность",
            [
                ("Максимальный RPS", _html(peak.get("max_rps") or "—")),
                ("Время пиковой производительности", _html(peak.get("max_time") or "—")),
                ("Время деградации", _html(peak.get("drop_time") or "—")),
            ],
        ))

    parts.append(_render_problem_recommendation_table(report))
    return "\n".join(parts)


def _replace_placeholder_storage(storage_html: str, placeholder: str, value: str) -> tuple[str, bool]:
    """Заменяет плейсхолдер в Confluence storage без вложения block HTML внутрь <p>.

    Если плейсхолдер стоит отдельным абзацем (`<p>$$...$$</p>`), заменяем весь
    абзац. Иначе блочная таблица может оказаться внутри `<p>`, и Confluence
    покажет HTML как обычный текст.
    """
    html = storage_html or ""
    ph = str(placeholder)
    replacement = str(value)
    patterns: list[str] = []
    if ph.startswith("$$") and ph.endswith("$$"):
        inner = ph[2:-2].strip()
        token = r"\$\$\s*" + re.escape(inner) + r"\s*\$\$"
        inline_wrappers = (
            r"(?:<span\b[^>]*>\s*)*"
            r"(?:<(?:strong|b|em|i)\b[^>]*>\s*)*"
            + token +
            r"(?:\s*</(?:strong|b|em|i)>)*"
            r"(?:\s*</span>)*"
        )
        patterns.extend([
            r"(?is)<p\b[^>]*>\s*" + inline_wrappers + r"\s*</p>",
            r"(?is)<div\b[^>]*>\s*" + inline_wrappers + r"\s*</div>",
            r"(?is)" + token,
        ])
    else:
        patterns.extend([
            r"(?is)<p\b[^>]*>\s*" + re.escape(ph) + r"\s*</p>",
            r"(?is)" + re.escape(ph),
        ])
    for pattern in patterns:
        new_html, count = re.subn(pattern, replacement, html, count=1)
        if count:
            return new_html, True
    if ph in html:
        return html.replace(ph, replacement, 1), True
    return html, False


def update_confluence_page_multi(url, username, password, page_id, replacements: dict) -> str:
    """Один проход по странице: заменить несколько плейсхолдеров.

    Параметры:
        url/username/password/page_id: доступ к Confluence.
        replacements (dict): Карта `{placeholder: html}`.

    Возвращает:
        str: Текст статуса.
    """
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
        new_html, did_replace = _replace_placeholder_storage(html, str(placeholder), str(value))
        if did_replace:
            html = new_html
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





