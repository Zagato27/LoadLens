from confluence_manager.update_confluence_template import copy_confluence_page, update_confluence_page, update_confluence_page_multi, render_llm_report_placeholders, render_llm_markdown
from AI.main import uploadFromLLM

from concurrent.futures import ThreadPoolExecutor, as_completed
from data_collectors.grafana_collector import downloadImagesLogin, send_file_to_attachment
from data_collectors.loki_collector import fetch_loki_logs, send_loki_file_to_attachment
from settings import CONFIG  # Импорт базовой конфигурации
from metrics_config import METRICS_CONFIG  # Импорт конфигурации метрик
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime  # если ещё не импортирован
import traceback  # для детального вывода ошибок (опционально)
import uuid
import os
from requests.auth import HTTPBasicAuth
import json
import ast


def update_report(start, end, service, use_llm: bool = True, save_to_db: bool = False, web_only: bool = False, run_name: str | None = None, progress_callback=None):
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

    # Режим «только веб»: не создаём страницу Confluence, не скачиваем изображения из Grafana
    if web_only:
        def _progress(msg: str, pct: int | None = None):
            try:
                if pct is not None:
                    print(f"[progress] {msg} {pct}%")
                else:
                    print(f"[progress] {msg}")
            except Exception:
                pass
            try:
                if callable(progress_callback):
                    progress_callback(msg, pct)
            except Exception:
                pass

        _progress("Создание веб-отчёта (без Confluence) начато…", 5)

        # Гарантируем сохранение данных в БД для веб-страниц /reports и графиков
        save_to_db_effective = True if web_only else bool(save_to_db)

        run_meta = {
            "run_id": uuid.uuid4().hex,
            "run_name": (run_name or "").strip() or datetime.now().strftime("run-%Y%m%d-%H%M%S"),
            "service": service,
            "start_ms": start,
            "end_ms": end,
        }

        _progress("Сбор метрик для веб-отчёта…", 30)
        results = None
        if use_llm or save_to_db_effective:
            results = uploadFromLLM(
                start/1000,
                end/1000,
                save_to_db=save_to_db_effective,
                run_meta=run_meta,
                only_collect=not use_llm
            )
        else:
            _progress("Пропускаем LLM-анализ и сбор доменных данных по запросу пользователя")

        _progress("Финализация веб-отчёта…", 95)
        page_url = f"/reports/{run_meta['run_name']}"
        _progress("Отчёт (веб) готов ✅", 100)
        return {"page_id": None, "page_url": page_url, "run_name": run_meta["run_name"]}

    # Получаем `page_sample_id` и `page_parent_id` из конфигурации сервиса
    page_parent_id = service_config["page_parent_id"]
    page_sample_id = service_config["page_sample_id"]
    copy_page_id = copy_confluence_page(url_basic, user, password, page_sample_id, page_parent_id)
    page_url = f"{url_basic.rstrip('/')}/pages/viewpage.action?pageId={copy_page_id}"

    # Прогресс
    def _progress(msg: str, pct: int | None = None):
        try:
            if pct is not None:
                print(f"[progress] {msg} {pct}%")
            else:
                print(f"[progress] {msg}")
        except Exception:
            pass
        try:
            if callable(progress_callback):
                progress_callback(msg, pct)
        except Exception:
            pass

    _progress("Создание отчёта начато…", 5)
   

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

    # Двухфазный процесс: 1) скачать всё 2) одним проходом вложить и заменить
    _progress("Скачивание графиков и логов…", 10)
    
    # Сформируем задания на скачивание графиков и логов
    metric_items = []  # элементы: {name, placeholder, file_basename, file_path}
    log_items = []     # элементы: {placeholder, file_basename, file_path}

    for metric in service_config["metrics"]:
        name = metric["name"]
        grafana_url = f"{grafana_base_url}{metric['grafana_url']}&from={start}&to={end}"
        file_basename = f"{name}_{service}_{copy_page_id}"
        file_path = f"data_collectors/temporary_files/{file_basename}.jpg"
        metric_items.append({
            "name": name,
            "placeholder": f"$${name}$$",
            "grafana_url": grafana_url,
            "file_basename": file_basename,
            "file_path": file_path,
        })

    for log in service_config.get("logs", []):
        placeholder = log["placeholder"]
        file_basename = f"{service}_{placeholder}_{copy_page_id}"
        file_path = f"data_collectors/temporary_files/{file_basename}.log"
        log_items.append({
            "placeholder": f"$${placeholder}$$",
            "filter_query": log["filter_query"],
            "file_basename": file_basename,
            "file_path": file_path,
        })

    # Вспомогательные обёртки с ретраями
    def _download_img_with_retry(image_url: str, file_basename: str, username: str, password: str, max_attempts: int = 3) -> bool:
        for attempt in range(max_attempts):
            try:
                downloadImagesLogin(image_url, file_basename, username, password)
                path = f"data_collectors/temporary_files/{file_basename}.jpg"
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return True
                else:
                    print(f"[warn] Файл не создан или пуст: {path}")
            except Exception as e:
                print(f"[warn] Попытка загрузки изображения не удалась ({attempt+1}/{max_attempts}): {e}")
            time.sleep(1 * (attempt + 1))
        return False

    def _download_log_with_retry(loki_url: str, start_ts: int, end_ts: int, filter_query: str, file_basename: str, max_attempts: int = 3) -> bool:
        for attempt in range(max_attempts):
            try:
                path = fetch_loki_logs(loki_url, start_ts, end_ts, filter_query, file_basename)
                if isinstance(path, str) and os.path.exists(path) and os.path.getsize(path) > 0:
                    return True
                else:
                    print(f"[warn] Лог-файл не создан или пуст: {file_basename}.log")
            except Exception as e:
                print(f"[warn] Попытка получения логов не удалась ({attempt+1}/{max_attempts}): {e}")
            time.sleep(1 * (attempt + 1))
        return False

    # Параллельно скачиваем все графики и получаем логи (ограниченный пул)
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(metric_items) + len(log_items)))) as executor:
        download_futures = []
        for m in metric_items:
            download_futures.append(executor.submit(
                _download_img_with_retry, m["grafana_url"], m["file_basename"], grafana_login, grafana_pass
            ))
        log_futures = []
        for l in log_items:
            log_futures.append(executor.submit(
                _download_log_with_retry, loki_url, start, end, l["filter_query"], l["file_basename"]
            ))

        # Дождёмся завершения загрузок
        for f in as_completed(download_futures + log_futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"Ошибка при скачивании/получении: {e}")

    _progress("Загрузка вложений и обновление страницы…", 50)

    # Одним проходом: прикрепляем все файлы и копим замены (ограниченный пул + ретраи)
    def _attach_with_retry(func, *args, max_attempts: int = 3, **kwargs) -> bool:
        for attempt in range(max_attempts):
            try:
                resp = func(*args, **kwargs)
                code = getattr(resp, "status_code", None) if resp is not None else None
                if code in (200, 201):
                    return True
                # 409 трактуем как успех (вложение с таким именем уже есть)
                if code == 409:
                    print("[info] Вложение уже существует, считаю успехом.")
                    return True
                print(f"[warn] Ошибка загрузки вложения (attempt {attempt+1}): status={code}")
            except Exception as e:
                print(f"[warn] Попытка загрузки вложения не удалась ({attempt+1}): {e}")
            time.sleep(1 * (attempt + 1))
        return False

    replacements_pending = {}
    attach_future_to_ph = {}
    success_placeholders: set[str] = set()

    auth = HTTPBasicAuth(user, password)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(metric_items) + len(log_items)))) as executor:
        # Графики
        for m in metric_items:
            if os.path.exists(m["file_path"]) and os.path.getsize(m["file_path"]) > 0:
                fut = executor.submit(_attach_with_retry, send_file_to_attachment, url_basic, auth, copy_page_id, m["file_path"])
                attach_future_to_ph[fut] = (m["placeholder"], f'<ac:image><ri:attachment ri:filename="{m["file_basename"]}.jpg" /></ac:image>')
            else:
                print(f"[warn] Не найден файл графика: {m['file_path']}")

        # Логи
        for l in log_items:
            if os.path.exists(l["file_path"]) and os.path.getsize(l["file_path"]) > 0:
                fut = executor.submit(_attach_with_retry, send_loki_file_to_attachment, url_basic, auth, copy_page_id, l["file_path"]) 
                attach_future_to_ph[fut] = (l["placeholder"], (
                    f'<ac:structured-macro ac:name="view-file" ac:schema-version="1">'
                    f'<ac:parameter ac:name="name">'
                    f'<ri:attachment ri:filename="{l["file_basename"]}.log" />'
                    f'</ac:parameter>'
                    f'<ac:parameter ac:name="height">250</ac:parameter>'
                    f'</ac:structured-macro>'
                ))
            else:
                print(f"[warn] Не найден файл логов: {l['file_path']}")

        # Дождёмся загрузки всех вложений и соберём успешные плейсхолдеры
        for f in as_completed(list(attach_future_to_ph.keys())):
            ph, html = attach_future_to_ph[f]
            ok = False
            try:
                ok = bool(f.result())
            except Exception as e:
                print(f"Ошибка при загрузке вложения: {e}")
                ok = False
            if ok:
                success_placeholders.add(ph)
                replacements_pending[ph] = html

    # Убираем временные файлы
    for m in metric_items:
        try:
            if os.path.exists(m["file_path"]):
                os.remove(m["file_path"])
        except Exception:
            pass
    for l in log_items:
        try:
            if os.path.exists(l["file_path"]):
                os.remove(l["file_path"])
        except Exception:
            pass

    # Одно мульти-обновление всех плейсхолдеров (только успешно загруженные)
    try:
        if replacements_pending:
            update_confluence_page_multi(url_basic, user, password, copy_page_id, replacements_pending)
        else:
            print("[warn] Нет успешных вложений для подстановки плейсхолдеров")
    except Exception as e:
        print(f"Ошибка при мульти-обновлении плейсхолдеров (графики/логи): {e}")

    _progress("Графики и логи добавлены и обновлены. Запуск анализа ИИ…", 70)

    # Получаем результаты LLM и обновляем их последовательно
    results = None
    # Сбор доменных данных и/или LLM анализ
    run_meta = None
    if save_to_db:
        run_meta = {
            "run_id": uuid.uuid4().hex,
            "run_name": (run_name or "").strip() or datetime.now().strftime("run-%Y%m%d-%H%M%S"),
            "service": service,
            "start_ms": start,
            "end_ms": end,
        }
    if use_llm or save_to_db:
        results = uploadFromLLM(
            start/1000,
            end/1000,
            save_to_db=save_to_db,
            run_meta=run_meta,
            only_collect=not use_llm
        )
    else:
        _progress("Пропускаем LLM-анализ и сбор доменных данных по запросу пользователя")

    # Мульти-обновление плейсхолдеров LLM за один проход
    try:
        _progress("Обновление данных LLM (одним проходом)...", 85)
        llm_replacements = {}
        # Подставляем только те плейсхолдеры, для которых есть данные
        def add_if_present(placeholder: str, key: str):
            if not results:
                return
            val = results.get(key)
            if isinstance(val, str) and val.strip():
                llm_replacements[placeholder] = val

        add_if_present("$$answer_jvm$$", "jvm")
        add_if_present("$$answer_database$$", "database")
        add_if_present("$$answer_kafka$$", "kafka")
        add_if_present("$$answer_ms$$", "ms")
        add_if_present("$$answer_hard_resources$$", "hard_resources")

        # Добавляем финальный плейсхолдер только как $$final_answer$$
        final_struct = (results or {}).get("final_parsed")
        if isinstance(final_struct, dict) and final_struct:
            md = render_llm_markdown(final_struct)
            if md.strip():
                llm_replacements["$$final_answer$$"] = md
        else:
            # Фолбэк: если нет структурированного ответа, отдаем текст как markdown-блок
            final_text = (results or {}).get("final")
            if isinstance(final_text, str) and final_text.strip():
                md_fallback = f"### Итог LLM\n\n{final_text}"
                llm_replacements["$$final_answer$$"] = md_fallback

        # Доменные секции в человекочитаемом markdown при наличии parsed
        def _inject_confidence(rep: dict | None, domain_key: str) -> dict | None:
            if not isinstance(rep, dict):
                return rep
            try:
                if rep.get("confidence") in (None, "", "null"):
                    c = ((results or {}).get("scores", {}) or {}).get(domain_key, {}) or {}
                    cval = c.get("confidence")
                    if isinstance(cval, (int, float)):
                        rep = {**rep, "confidence": float(cval)}
            except Exception:
                pass
            return rep

        jvm_struct = _inject_confidence((results or {}).get("jvm_parsed"), "jvm")
        if isinstance(jvm_struct, dict) and jvm_struct:
            md = render_llm_markdown(jvm_struct)
            if md.strip():
                llm_replacements["$$answer_jvm$$"] = md

        db_struct = _inject_confidence((results or {}).get("database_parsed"), "database")
        if isinstance(db_struct, dict) and db_struct:
            md = render_llm_markdown(db_struct)
            if md.strip():
                llm_replacements["$$answer_database$$"] = md

        kafka_struct = _inject_confidence((results or {}).get("kafka_parsed"), "kafka")
        if isinstance(kafka_struct, dict) and kafka_struct:
            md = render_llm_markdown(kafka_struct)
            if md.strip():
                llm_replacements["$$answer_kafka$$"] = md

        ms_struct = _inject_confidence((results or {}).get("ms_parsed"), "microservices")
        if isinstance(ms_struct, dict) and ms_struct:
            md = render_llm_markdown(ms_struct)
            if md.strip():
                llm_replacements["$$answer_ms$$"] = md

        hr_struct = _inject_confidence((results or {}).get("hard_resources_parsed"), "hard_resources")
        if isinstance(hr_struct, dict) and hr_struct:
            md = render_llm_markdown(hr_struct)
            if md.strip():
                llm_replacements["$$answer_hard_resources$$"] = md

        def _try_json_to_markdown(raw_text: str) -> str | None:
            if not isinstance(raw_text, str) or not raw_text.strip():
                return None
            try:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    return None
                candidate = raw_text[start:end + 1]
                parsed = None
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    try:
                        parsed = ast.literal_eval(candidate)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    md = render_llm_markdown(parsed)
                    return md.strip() or None
            except Exception:
                return None
            return None

        def _ensure_markdown_for_placeholder(ph: str):
            val = llm_replacements.get(ph)
            if not isinstance(val, str):
                return
            md = _try_json_to_markdown(val)
            if md:
                llm_replacements[ph] = md

        for ph in ("$$answer_jvm$$", "$$answer_database$$", "$$answer_kafka$$", "$$answer_ms$$", "$$answer_hard_resources$$"):
            _ensure_markdown_for_placeholder(ph)

        # Добавим данные судьи/программной оценки в раздел доверия (для всех доменов и финала)
        try:
            scores = (results or {}).get("scores", {})

            def _pct(x: float | None) -> str:
                try:
                    if x is None:
                        return "—"
                    return f"{int(float(x)*100)}%"
                except Exception:
                    return "—"

            def _append_judge(ph: str, domain_key: str):
                val = llm_replacements.get(ph)
                if not isinstance(val, str) or not val.strip():
                    return
                s = (scores or {}).get(domain_key) or {}
                judge = (s or {}).get("judge") or {}
                overall = judge.get("overall")
                factual = judge.get("factual")
                completeness = judge.get("completeness")
                specificity = judge.get("specificity")
                data_score = (s or {}).get("data_score")
                final_score = (s or {}).get("final_score")
                conf = (s or {}).get("confidence")
                lines = [
                    "\n\n#### Доверие (судья)",
                    f"- Итог: {_pct(overall)}",
                    f"- Согласованность (factual): {_pct(factual)}",
                    f"- Полнота (completeness): {_pct(completeness)}",
                    f"- Конкретика (specificity): {_pct(specificity)}",
                    f"- По данным: {_pct(data_score)}",
                    f"- Агрегат: {_pct(final_score)}",
                ]
                if isinstance(conf, (int, float)):
                    lines.append(f"- Доверие модели: {_pct(float(conf))}")
                lines.extend([
                    "",
                    "_Пояснения: итог = 0.6·factual + 0.3·completeness + 0.2·specificity._",
                ])
                llm_replacements[ph] = val + "\n".join(lines)

            _append_judge("$$answer_jvm$$", "jvm")
            _append_judge("$$answer_database$$", "database")
            _append_judge("$$answer_kafka$$", "kafka")
            _append_judge("$$answer_ms$$", "microservices")
            _append_judge("$$answer_hard_resources$$", "hard_resources")
            _append_judge("$$final_answer$$", "final")
        except Exception as e:
            print(f"[warn] Не удалось добавить оценки судьи: {e}")

        if llm_replacements:
            update_confluence_page_multi(url_basic, user, password, copy_page_id, llm_replacements)
        _progress("ИИ-анализ завершён. Финализация отчёта…", 95)
        print("✓ Плейсхолдеры LLM обновлены за один проход")
    except Exception as e:
        print(f"Ошибка при мульти-обновлении данных LLM: {e}")

    _progress("Отчёт готов ✅", 100)
    return {"page_id": copy_page_id, "page_url": page_url}
