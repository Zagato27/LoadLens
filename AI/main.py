"""Совместимостьный модуль для вызова LLM‑pipeline.

Все реальное поведение находится в `AI.pipeline.uploadFromLLM`, а здесь
сохраняется прежний импорт для стороннего кода и CLI‑примеров из README.
"""

from typing import Dict, List, Optional

from AI.pipeline import uploadFromLLM as _pipeline_upload

__all__ = ["uploadFromLLM"]


def uploadFromLLM(
    start_ts: float,
    end_ts: float,
    save_to_db: bool = False,
    run_meta: Optional[dict] = None,
    only_collect: bool = False,
    ef_config: Optional[dict] = None,
    prompts_override: Optional[dict] = None,
    active_domains: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Запускает полный цикл подготовки LLM‑отчёта.

    Назначение:
        Обёртка вокруг `_pipeline_upload`, чтобы не ломать внешние импорты.
        Включает сбор метрик, подготовку контекста, запросы к LLM и
        (опционально) сохранение результатов в TimescaleDB.

    Параметры:
        start_ts (float): Время начала интервала теста в секундах Unix.
        end_ts (float): Время окончания интервала теста в секундах Unix.
        save_to_db (bool): Если `True`, результаты и метрики фиксируются в БД.
        run_meta (dict | None): Служебные атрибуты запуска
            (`run_id`, `run_name`, `service`, `test_type`, `start_ms`, `end_ms`).
        only_collect (bool): При `True` собирает только метрики без вызова LLM.
        ef_config (dict | None): Эффективная конфигурация (источники метрик,
            запросы и т.д.), позволяющая переопределить `settings.py`.
        prompts_override (dict | None): Пользовательские тексты промптов
            в разрезе доменов (`jvm`, `database`, ...).
        active_domains (list[str] | None): Ограничение списка доменов,
            которые нужно анализировать (например, без `lt_framework`).

    Возвращаемое значение:
        dict: агрегат с текстовыми блоками, структурированными ответами
        (`*_parsed`) и метриками качества (`scores`) по каждому домену
        и итоговому разделу.

    Побочные эффекты:
        - выполняются сетевые обращения к источникам метрик и LLM‑провайдеру;
        - при `save_to_db=True` производится запись в TimescaleDB.

    Исключения:
        Пробрасывает исключения `_pipeline_upload`, если сбор данных или
        запись в БД завершаются с ошибкой.
    """

    return _pipeline_upload(
        start_ts=start_ts,
        end_ts=end_ts,
        save_to_db=save_to_db,
        run_meta=run_meta,
        only_collect=only_collect,
        ef_config=ef_config,
        prompts_override=prompts_override,
        active_domains=active_domains,
    )


