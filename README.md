# Автоматические отчёты по нагрузочному тестированию (LoadLens + Confluence)

Создание отчётов в два клика: сбор метрик/логов, AI‑аналитика по доменам, публикация в Confluence и/или локальный «LoadLens» (веб‑интерфейс). Поддерживаются Prometheus напрямую и через Grafana‑прокси, а также InfluxDB (напрямую и через Grafana‑прокси) для метрик генератора нагрузки (lt_framework).

## Оглавление
- [Что нового](#что-нового)
- [Архитектура](#архитектура)
- [Компоненты](#компоненты)
- [Плейсхолдеры шаблона Confluence](#плейсхолдеры-шаблона-confluence)
- [Конфигурация](#конфигурация)
- [Запуск и Docker](#запуск-и-docker)
- [REST API](#rest-api)
- [Особенности UI](#особенности-ui)
- [Безопасность и сеть](#безопасность-и-сеть)
- [Устранение неполадок](#устранение-неполадок)

## Что нового
- Две ветки публикации: Confluence и LoadLens (можно выбрать обе; LLM вызывается один раз, без дублирования для веб‑ветки).
- Пошаговая форма «Новый отчёт»: цель сохранения → время начала/окончания → проектная область → тип теста → LLM → название запуска. Кнопка «Создать отчёт» появляется только после ввода названия.
- Поддержка типов тестов: ступенчатый (step), soak, spike, stress — профиль автоматически внедряется в промпты общего и доменных анализов.
- Новая доменная область `lt_framework`: анализ метрик инструмента нагрузочного тестирования (RPS, checks, p95 по URL/транзакции и т.д.). Источники: Prometheus/InfluxDB напрямую или через Grafana‑прокси.
- Прямой доступ к Prometheus: `metrics_source.type="prometheus"`.
- Прямой и прокси‑доступ к InfluxDB для `lt_framework`: `lt_metrics_source.type="influxdb"` или `grafana_proxy`.
- Страница «Отчёт»: блок «Итоги от инженера» с редактором (по умолчанию read‑only, редактирование по кнопке), хранится в отдельной таблице TimescaleDB.
- Проектные области (service): фильтрация всего UI и данных по выбранной области; рантайм‑оверрайды конфигов и промптов по областям.
- Уникальность `run_name` в пределах выбранной области проверяется на бэкенде при создании отчёта.
- Архив: удаление отчётов, колонка «Тип теста», ссылки `/reports/{service}/{run_name}`.
- Сравнение тестов: сводные таблицы p95 по домену и отдельно «по каждой метрике» (расширяемые по клику) с сортировкой по колонкам.

## Архитектура
1. Веб‑UI (Flask + Jinja): дэшборд (`/`), «Новый отчёт» (`/new`), архив (`/reports`), сравнение (`/compare`), настройки (`/settings`).
2. Оркестратор (`update_page.update_report`): копирование шаблона Confluence, сбор метрик/логов, запуск AI конвейера, единичная массовая подстановка плейсхолдеров.
3. Источники данных:
   - Метрики тестируемой системы (SUT): `metrics_source` (Prometheus напрямую или Grafana‑прокси).
   - Метрики генератора нагрузки: `lt_metrics_source` (Prometheus/InfluxDB напрямую или через Grafana‑прокси).
   - Логи: Loki (по ссылкам из `metrics_config`).
4. Хранилище (TimescaleDB): метрики (`metrics`), LLM‑результаты (`llm_reports`), итоги инженера (`engineer_reports`).
5. Рантайм‑оверрайды: `settings_runtime.json`, `metrics_config_runtime.json` — применяются «на лету» и переживают перезапуск.

## Компоненты
- `app.py` — Flask‑маршруты и API:
  - Страницы: `/`, `/new`, `/reports`, `/reports/<service>/<run_name>`, `/compare`, `/settings`.
  - API: `/create_report`, `/job_status/<id>`, `/runs` (список архивов c `test_type`), `/runs/<run_name> [DELETE]`, `/llm_reports`, `/domains_schema`, `/run_series`, `/compare_series`, `/compare_summary`, `/compare_metric_summary`, `/engineer_summary [GET/POST]`, `/config [GET/POST]`, `/prompts [GET/POST]`, `/project_area [POST]`, `/current_project_area`, `/services`.
- `update_page.py` — оркестрация публикации; передаёт `test_type`, подставляет все домены (включая `lt_framework`) в Confluence.
- `AI/pipeline.py` — сбор данных из источников (Prometheus/InfluxDB, напрямую/через Grafana), формирование доменных контекстов, инъекция профиля `test_type`, запуск LLM, объединение результатов в итог.
- `AI/db_store.py` — сохранение LLM‑результатов и создание таблиц; поддержка колонки `test_type` (в `final`), отдельная таблица для «Итогов инженера».
- `confluence_manager/update_confluence_template.py` — массовое обновление плейсхолдеров и преобразование LLM‑JSON в markdown.
- `metrics_config.py` — ссылки на панели Grafana/фильтры Loki для Confluence‑отчётов по областям (service).
- `settings.py` / `settings.example.py` — основная конфигурация приложения (см. ниже).

## Плейсхолдеры шаблона Confluence
- Метрики: `$$<name>$$` — подставляются изображения панелей Grafana (имя берётся из `metrics_config`).
- Логи: `$$<placeholder>$$` — вставляется виджет просмотра вложенного `.log`.
- AI‑аналитика:
  - `$$answer_jvm$$`, `$$answer_database$$`, `$$answer_kafka$$`, `$$answer_ms$$`, `$$answer_hard_resources$$`, `$$answer_lt_framework$$`.
  - `$$final_answer$$` — общий сводный блок. Если LLM вернул `peak_performance`, раздел выводится автоматически [[memory:8657199]].

Отсутствующие данные пропускаются безопасно.

## Конфигурация
Основной файл: `settings.py` (создайте из `settings.example.py`).

- Базовые доступы: `user/password/url_basic/space_conf/grafana_base_url/loki_url`.
- `llm` — провайдер (`perplexity|openai|anthropic`), лимиты/ретраи, опция `include_markdown_tables_in_context`.
- `default_params` — `step` (шаг выборки), `resample_interval` (ресемплирование).
- Источники метрик:
  - `metrics_source` — метрики тестируемой системы:
    - `type`: `prometheus` | `grafana_proxy`.
    - `prometheus.url`: прямой PromQL.
    - `grafana.{base_url, verify_ssl, auth, prometheus_datasource}`: доступ к /api/datasources/proxy/.../api/v1/query_range.
  - `lt_metrics_source` — метрики генератора нагрузки:
    - `type`: `prometheus` | `grafana_proxy` | `influxdb`.
    - `prometheus.url`: прямой PromQL.
    - `grafana.{prometheus_datasource|influxdb_datasource}`: проксирование Prometheus/InfluxDB.
    - `influxdb.{url, org, bucket, database, token}`: прямые Flux/InfluxQL запросы.
- `storage.timescale` — параметры TimescaleDB (включая `engineer_table`).
- `queries` — список доменных запросов и подписи:
  - Для Prometheus: `promql_queries`, `label_keys_list`, `labels`.
  - Для InfluxDB (Flux): `flux_queries`, `label_tag_keys_list` (какие теги попадут в подпись серии), `labels`. Плейсхолдеры: `{bucket}`, `{start}`, `{end}` (ISO‑8601 UTC).
  - Для InfluxDB (InfluxQL): `influxql_queries`, `label_tag_keys_list`, `labels`. Плейсхолдеры Grafana: `$timeFilter`, `$__interval`, `$Group/$Tag/$URL/$Measurement` (в коде приводятся к разумным значениям).
- Рантайм‑оверрайды:
  - `settings_runtime.json` — перезаписывает секции `llm/metrics_source/lt_metrics_source/default_params/queries` точечно; поддерживает `per_area[<service>]` для областей.
  - `metrics_config_runtime.json` — перезаписывает `metrics_config` по областям.
- Проектные области: список доступных областей берётся из `metrics_config` (ключи верхнего уровня); активная область хранится в cookie `project_area`.

См. также `settings.example.py` с подробными комментариями к каждому полю.

## Запуск и Docker
- Локально:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
  Откройте `http://localhost:5000/`.

- Docker / Compose: см. подробную инструкцию в `DOCKER_SETUP.md` (поднятие `app` + `timescaledb`, пример `.env`, копирование `settings.example.py`).

## REST API
- `POST /create_report` — запуск отчёта (поля: `start`, `end`, `service`, `use_llm`, `run_name`, `test_type`; при выборе Confluence и LoadLens LLM выполняется один раз).
- `GET /runs` — список запусков (включая `test_type`, `verdict`, `report_created_at`).
- `DELETE /runs/<run_name>` — удалить отчёт (метрики, результаты LLM и «итоги инженера»).
- `GET /llm_reports?run_name=...&service=...` — доменные/финальные результаты.
- `GET /domains_schema` — доступные домены и их метрики.
- `GET /compare_summary` — p95 сводка по метрикам в домене.
- `GET /compare_metric_summary` — p95 сводка по всем сериям конкретной метрики.
- `GET /run_series`, `GET /compare_series` — временные ряды (абсолют/смещение).
- `GET/POST /engineer_summary` — чтение/сохранение блока «Итоги от инженера».
- `GET/POST /config` — чтение/изменение конфигурации (с поддержкой `per_area`).
- `GET/POST /prompts` — чтение/изменение промптов по доменам и областям.

## Особенности UI
- Новый отчёт: пошаговая форма, выбор цели публикации (Confluence/LoadLens), тип теста, LLM «Да/Нет», валидация уникальности `run_name` по области.
- Архив: удаление, колонка `Тип теста`, ссылки `/reports/{service}/{run_name}`.
- Страница отчёта: заголовок H1 «Отчёт по тесту {run_name}», ниже H1 «Время теста: ... – ...», блок «Итоги от инженера» под заголовком, редактор по кнопке «Редактировать».
- Сравнение: по каждому домену — сводная p95‑таблица; дополнительно — раскрывающиеся «сводные таблицы по метрикам» со значениями для всех серий; в обоих таблицах доступна сортировка по колонкам.
- Настройки: Accordion‑страница с редакторами JSON (Ace): разделы LoadLens (LLM, Источник метрик, Хранилище, Параметры, Запросы, Промпты по доменам, включая `lt_framework`) и Confluence (Confluence‑доступы и ссылки на графики/логи). Редактирование — по кнопке, с валидацией и возможностью отмены.

## Безопасность и сеть
- Не храните секреты в git. Пользуйтесь `settings.example.py` / `settings.py` локально, `.env` для compose, или секретами CI/CD.
- В Grafana/Loki используйте отдельные сервисные учётки и ограниченные токены.
- При прямом доступе к Prometheus/InfluxDB учитывайте требования к TLS/MTLS (см. параметры `verify` и сертификаты, если нужно).

## Устранение неполадок
- Пустые разделы отчёта: проверьте корректность источников (`metrics_source`, `lt_metrics_source`), доступность датасорсов Grafana и фильтров временного интервала.
- InfluxQL p95 пустой: убедитесь, что значения колонок — числа (в конвейере предусмотрено принудительное приведение).
- Ошибка TimescaleDB «current transaction is aborted»: бэкенд выполняет `rollback` после неудачных запросов; воспроизведите проблему, проверьте схему таблиц.
- LLM вернул «Недостаточно данных»: чаще всего не хватает метрик за интервал — проверьте настройки и окна агрегаций.

Подсистема AI и промпты подробно описаны в `AI/README.md`.