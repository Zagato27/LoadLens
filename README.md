# LoadLens - Автоматические отчёты по нагрузочному тестированию 
По вопросам можете писать в личку Telegram @pbekker


Веб‑приложение для автоматизации подготовки отчётов по нагрузочному тестированию: сбор метрик за выбранный интервал, анализ ИИ по доменам и итоговый вывод, хранение в TimescaleDB и интерактивный веб‑интерфейс. Поддерживаются источники: Prometheus (напрямую или через Grafana‑proxy) и InfluxDB (напрямую или через Grafana‑proxy) для метрик инструмента нагрузки.

## Возможности
- Сбор временных рядов метрик из Prometheus/InfluxDB (включая проксирование через Grafana).
- Доменные отчёты: JVM, базы данных, Kafka, микросервисы, инфраструктурные ресурсы; при необходимости — метрики инструмента нагрузки (`lt_framework`).
- Итоговый отчёт на основе доменных выводов с поддержкой детекции пиков производительности (peak_performance).
- Явное обоснование вердикта (`verdict_rationale`) в доменных и итоговом LLM-блоках: короткий вывод плюс ключевые факторы, повлиявшие на итог.
- G-Eval-lite judge для отбора лучшего кандидата LLM: статические rubric files по доменам + детерминированный `score_candidate_by_data()` как второй канал.
- Программная SLA-оценка результата теста: `target_rps`, пороги ошибок/latency/CPU/MEM, отдельный verdict и детали проверок.
- Сравнение запусков: сводные p95‑сводки по доменам и по сериям метрик.
- Области проекта (services): независимые настройки/промпты и фильтрация данных по области.
- Структурированный `system_context`: единый JSON-контекст тестируемой системы с описанием архитектуры, критичных потоков, зависимостей и фокуса анализа.
- «Итоги инженера»: хранение и редактирование пользовательского резюме по запуску.

## Быстрый старт
### Вариант A: Docker Compose
1) Скопируйте `docker/env.example` в `docker/.env` и заполните параметры TimescaleDB/Redis, например:
   ```
   POSTGRES_USER=ltar
   POSTGRES_PASSWORD=ltar
   POSTGRES_DB=loadtesting
   TSDB_PORT=5432
   APP_PORT=5000
   CELERY_BROKER_URL=redis://redis:6379/0
   CELERY_RESULT_BACKEND=redis://redis:6379/0
   CELERY_TASK_ALWAYS_EAGER=0
   ```
2) Запустите (из директории `docker/`):
   ```
   cd docker
   docker compose up -d --build
   ```
3) Откройте http://localhost:5000 и создайте первый отчёт на странице «Новый отчёт».
   Компоуз поднимет веб-приложение, TimescaleDB, Redis и отдельный Celery worker для фоновых задач (скачивание графиков, загрузка вложений, LLM).

### Вариант B: Локально (без контейнеров)
1) Python 3.12+.  
2) Установите зависимости:
   ```
   pip install -r requirements.txt
   ```
3) Настройте подключение к TimescaleDB и источникам метрик (см. раздел «Конфигурация»).  
4) Запустите приложение:
   ```
   python app.py
   ```

## Конфигурация
- `settings.py` — основной конфиг (можно начать с `settings.example.py`). Ключевые разделы:
  - `llm` — провайдер (`perplexity` | `openai` | `anthropic`), модели и лимиты токенов.
  - `sla` — критерии автоматической оценки: `target_rps`, `max_performance_query`, `max_error_rate_pct`, `max_p95_ms`, `max_p99_ms`, `max_cpu_pct`, `max_memory_pct`.
  - `metrics_source` — источник метрик системы под тестом: `type=prometheus|grafana_proxy`.
  - `lt_metrics_source` — источник метрик инструмента нагрузки: `type=prometheus|grafana_proxy|influxdb`.
  - `default_params` — шаг выборки (`step`) и ресемплирование (`resample_interval`).
  - `system_context` — структурированный контекст системы под тестом: описание продукта, архитектурный стиль, компоненты, зависимости, critical flows, hotspots, known risks и focus для AI-анализа.
  - `storage.timescale` — параметры TimescaleDB (хранение метрик, LLM‑результатов и «итогов инженера»).
  - `queries` — PromQL/Flux/InfluxQL по доменам и правила формирования подписей серий.
- `system_context` заполняется прямо в `CONFIG` как обычный JSON-блок. Это и есть единственный источник формата для UI, runtime-override и AI-пайплайна.
- Минимальный практический шаблон:
  ```json
  {
    "schema_version": 1,
    "system": {
      "name": "Checkout Platform",
      "domain": "e-commerce",
      "description": "Сервис оформления заказов и оплат",
      "test_goal": "Проверить устойчивость checkout под пиковой нагрузкой"
    },
    "architecture": {
      "style": "microservices",
      "components": [],
      "dependencies": [],
      "data_stores": []
    },
    "load_model": {
      "entrypoints": [],
      "critical_user_flows": [],
      "expected_hotspots": []
    },
    "operational_context": {
      "known_constraints": [],
      "known_risks": [],
      "normal_degradation_rules": [],
      "analysis_focus": []
    }
  }
  ```
- Что обычно заполнять:
  - `system` — что это за система и зачем запускается тест.
  - `architecture` — важные компоненты, зависимости и shared storage.
  - `load_model` — точки входа нагрузки, критичные user flows и ожидаемые hotspots.
  - `operational_context` — известные ограничения, риски и желаемый фокус анализа.
- Рантайм‑оверрайды без перезапуска:
  - `settings_runtime.json` — точечные изменения разделов (в т.ч. в разрезе областей: `per_area[<service>]`).
  - `metrics_config_runtime.json` — перезапись конфигурации визуализаций для областей.
- Для SLA можно задавать как area-level, так и service-level overrides через UI настроек.
- Конфигурация визуализаций для публикаций и ссылок: `metrics_config.py`.
- Judge-рубрики лежат в `AI/prompts/judge_rubrics/*.txt`; общий шаблон judge — в `AI/prompts/judge_prompt.txt`.

## Системные требования
- Python 3.12+ — выполнение Flask-приложения и ИИ-пайплайна.
- Redis 7+ (или совместимый брокер) — очередь для Celery. В режиме разработки можно оставить `CELERY_TASK_ALWAYS_EAGER=1`, чтобы задачи выполнялись синхронно.
- Celery 5+ — фоновые задачи (скачивание графиков, сбор логов, запросы к LLM).
- TimescaleDB 2.x (или PostgreSQL 14+ с расширением TimescaleDB) — хранилище метрик, отчётов LLM и заметок инженера.
- Grafana 9+ с Prometheus/Influx источниками — проксирование запросов и рендер графиков/скриншотов.

## Пользовательский интерфейс
- Главная (дашборд) — краткая сводка по последнему запуску и распределению результатов тестов.
- «Новый отчёт» — форма запуска: интервал времени, область (service), опции LLM/сохранения, произвольное имя запуска.
- «Архив» — список запусков; удаление; ссылки на страницу отчёта. Для итогового статуса используется SLA-verdict, если он рассчитан, иначе verdict LLM.
- «Сравнение» — p95‑сводки между двумя запусками.
- «Настройки» — редактирование конфигов, промптов, SLA, `system_context` и параметров для областей/сервисов.
- Страницы отчётов — доменные блоки, итоговый блок, явное обоснование verdict (`verdict_rationale`), snapshot `system_context` и «Итоги инженера».

## REST API (основные)
- `POST /create_report` — сформировать отчёт. Тело JSON: `start` (YYYY‑MM‑DDTHH:MM), `end`, `service`, `test_type?`, `use_llm?`, `save_to_db?`, `web_only?`, `run_name?`.
- `GET /job_status/<job_id>` — статус фоновой задачи формирования отчёта.
- `GET /runs` — список запусков (в т.ч. `verdict`, `test_type`).
- `DELETE /runs/<run_name>` — удалить данные запуска (метрики, LLM‑результаты, итоги инженера).
- `GET /llm_reports?run_name=...` — доменные и итоговые блоки отчёта.
- `DELETE /service_meta` — удалить service-level runtime overrides без очистки метрик и отчётов в БД.
- `GET /compare_summary`, `GET /compare_metric_summary` — p95‑сводки.
- `GET /run_series`, `GET /compare_series` — временные ряды (абсолют/смещение).
- `GET/POST /engineer_summary` — чтение/сохранение пользовательского резюме.
- `GET/POST /config`, `GET/POST /prompts` — конфигурация и промпты (с поддержкой областей); `/config` также отдает активный `system_context`.

## Хранилище данных
TimescaleDB (PostgreSQL):  
- `public.metrics` — временные ряды метрик.  
- `llm_reports` — результаты доменных/итогового анализов, включая `verdict`, `parsed.verdict_rationale`, `sla_verdict`, `sla_details` и snapshot `system_context`.  
- `engineer_reports` — версии раздела «Итоги инженера».


## Отладка
- Пустые блоки отчёта — проверьте доступность источников метрик, корректность интервала и параметров ресемплирования.
- Ошибки подключения к БД — проверьте сеть/порты/учётные данные и наличие расширений TimescaleDB.
- Медленный ответ LLM — уменьшите `max_tokens` или параллелизм, проверьте провайдера и сетевые настройки. Для нестабильных провайдеров доступны pacing/cooldown-параметры `request_min_interval_sec` и `rate_limit_cooldown_sec`.

Дополнительные подробности по подсистеме ИИ и промптам — в `AI/README.md`.