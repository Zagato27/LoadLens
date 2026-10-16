# План: ускорение LoadLens, этапы 0–3 в две волны

Цель: убрать главные источники задержек, найденные при разборе 2026-10-16, не меняя стек. Работа идёт двумя
волнами по шесть агентов, каждый в своём разделе и со строгим владением файлами:

- **Волна A** (разделы 3–10): этапы 0–2 — замер, пул соединений, кэш конфигов, SQL и индексы, таблица
  прогонов, опросы и статика. Все шесть агентов стартуют одновременно.
- **Волна B** (раздел 11): этап 3 — построение отчётов и публикации переезжают из веб-процесса в Celery-воркер.
  Стартует после слияния волны A: опирается на пул соединений и на бенч.

Стартовая точка: `main` после коммита текущих правок; офлайн-тесты: 1199 passed, 12 skipped за 85 с
(`C:/Python312/python.exe -m pytest tests -q -p no:cacheprovider`).

Что было найдено (подробности в разделах агентов):

| Проблема | Где | Кто закрывает |
|---|---|---|
| Нет замера: неизвестно, какие запросы медленные и насколько | gunicorn без времени ответа, нет бенча | Агент 1 |
| Каждый `_ts_conn()` открывает новое соединение psycopg2 с TLS; опросы с фронта делают это десятки раз в минуту | `AI/storage_state.py::connect` | Агент 2 |
| Runtime-конфиги (227 КБ и 54 КБ JSON) читаются с диска, парсятся и копируются при каждом вызове, под общим RLock | `loadlens_app/core.py` | Агент 3 |
| `llm_reports`: оконные функции по всей таблице и regex по JSONB в каждой строке, нет индексов под фильтры | `dashboard.py::dashboard_data`, `_runs_max_rps_sql` | Агент 4 |
| `/runs` делает `GROUP BY run_name` по всей гипертаблице `metrics`, дважды; `/dashboard_data` считает `COUNT(DISTINCT run_name)` по ней же | `dashboard.py::list_runs` | Агент 5 |
| Опросы `/jobs`, `/load_tests/active`, `/dashboard_data` множатся на число вкладок; watcher ходит в БД каждые 3 с даже без прогонов; статика без кэш-заголовков и 205 КБ блокирующего скрипта | `dashboard.py`, `load_tests.py`, `watcher.py`, `app.py`, шаблоны | Агент 6 |

## Статус

Заполняют агенты по мере сдачи. Числа — из `tools/bench_endpoints.py` агента 1 на посеянных данных
(`tools/seed_bench_data.py`, параметры по умолчанию), p50/p95 в мс.

| Агент | Задача | Состояние | До | После | Примечания |
|---|---|---|---|---|---|
| 1 | bench: замер, сид, Server-Timing | сдано, влито (90939ed) | база в `docs/perf-baseline.md`: `/runs` 552 / 1473 мс (300 / 1000 прогонов, c=1), `/dashboard_data` 146 мс, `/jobs` 40 мс, `/services` 18 мс | — | Сид через `execute_values` (в 4 раза быстрее); runtime-JSON песочницы добиваются до 227/54 КБ; индекс `idx_metrics_run_name` создаёт только initdb, без него счётчик 848 мс вместо 5 мс (передано агенту 5) |
| 2 | pool: пул соединений | сдано, влито (82c7b32) | c=4: `/jobs` 85 / 130, `/load_tests/active` 80 / 110, `/dashboard_data` 342 / 433 | c=4: `/jobs` 21 / 32, `/load_tests/active` 31 / 42, `/dashboard_data` 160 / 207 | A/B на одном коде с `pool_max=0`. 616 запросов: 4 физических соединения вместо 462. `/services` пул не ускоряет: в базу не ходит. Сверх плана: проверка сокета перед выдачей, TCP keepalive, `discard()`, обработка `fork`. Сессия, из которой вышло исключение, закрывается физически: иначе сессионный advisory-лок миграции утёк бы следующему пользователю |
| 3 | config-cache: кэш runtime-конфигов | сдано, влито (0fbfa75) | `/services` c=1 34 / 41, c=4 115 / 187 (33 запроса/с); `_resolve_services_filter` 1.9–8.9 мс | `/services` c=1 10 / 29, c=4 15 / 29 (232 запроса/с); `_resolve_services_filter` 0.03–0.1 мс | Подпись файла: mtime, размер, inode, ctime плюс побайтовая сверка 2 с после записи (грубый mtime на Windows). `_active_metrics_config` кэшируется по идентичности разобранных файлов. Читатели в `config_api.py` и `projects.py` тоже переведены на кэш: GET /config разбирал файл 238 КБ около 8 раз |
| 4 | reports-sql: колонка `max_rps`, индексы `llm_reports` | не начато | | | |
| 5 | run-index: таблица `run_index`, `/runs` одним запросом | не начато | | | |
| 6 | polling-static: TTL-кэш опросов, пауза watcher, кэш статики | сдано, влито (50b13ce) | c=4: `/dashboard_data` 864 / 1240, `/jobs` 98 / 206, `/load_tests/active` 92 / 141 (машина была перегружена) | c=4, повторные запросы: `/dashboard_data` 21 / 37, `/jobs` 13 / 25, `/load_tests/active` 22 / 35; второй визит страницы — 0 запросов статики | В ключе `/jobs` есть `is_admin`: сырые ошибки видит только администратор. Шаблоны используют `url_for('static')` с хуком `url_defaults`, который добавляет `?v=`. В тестах кэш выключен по умолчанию (`LOADLENS_RESPONSE_CACHE=0`). Сверх плана: single-flight и счётчик поколений. Первый, некэшированный запрос стоит как раньше |
| B1 | bench: заглушки Prometheus/LLM/Confluence, режим `--during-job` | не начато | | | |
| B2 | celery-infra: конфигурация Celery, воркер в compose, env | не начато | | | |
| B3 | job-store: таблица задач как источник правды, владелец, sweeper | не начато | | | |
| B4 | tasks-runner: задачи Celery и выбор режима thread/celery | не начато | | | |
| B5 | pipeline-worker: отчёт без вложенных подзадач, безопасен в воркере | не начато | | | |
| B6 | e2e-docs: сквозная проверка с воркером, документация | не начато | | | |

Для волны B «до/после» — это p95 эндпоинтов во время построения отчёта (`--during-job`) в режиме `thread`
против режима `celery`.

## 0. Общие правила для всех агентов

1. **Секреты.** Никогда не открывать и не печатать `settings.py`, `settings_runtime*.json`, `metrics_config.py`,
   `metrics_config_runtime.json`, `docker/.env`, `.loadlens_secret_key`. Структуру смотреть в `settings.example.py`
   и `metrics_config.example.py`. Если секрет всё же попал на экран, в отчёте только имя файла, ключ и длина значения.
2. **Запуск приложения** только в песочнице (раздел 1), никогда с настоящим `settings.py`: он указывает на
   боевые системы. Никаких обращений к внешним адресам; локальные docker-контейнеры разрешены.
3. **Python и тесты.** `C:/Python312/python.exe` (в нём pytest, Flask, Playwright; в `.venv` их нет). Тесты
   офлайн: `C:/Python312/python.exe -m pytest tests -q -p no:cacheprovider`. Перед сдачей полный прогон без падений.
   Тесты, которым нужны конфиги, патчат и `core.CONFIG_RUNTIME_PATH`, и `update_page._SETTINGS_RUNTIME_PATH`
   (см. `tests/test_context_scope.py`); `update_page` импортировать после `loadlens_app`, иначе циклический импорт.
   Тесты, которым не нужна БД, подменяют `_ts_conn` во всех модулях (образец: `tests/test_jobs_dismiss.py::_no_database`).
4. **Файлы со смешанными CRLF/LF.** Правки через инструмент Edit, не через heredoc с бэкслешами.
5. **Язык.** Комментарии, докстринги, сообщения коммитов — английский. Строки интерфейса, документация, ответы
   пользователю — русский.
6. **Владение файлами** (раздел 2) строгое. Нужна правка в чужом файле — добавить функцию у себя и написать в
   отчёте, что и где должен вызвать владелец. Новые тесты — только в своём файле `tests/test_perf_<имя>.py`;
   чужие тесты не править, кроме случаев, явно перечисленных в разделе агента.
7. **Контракты API не меняются.** Форма JSON, коды и заголовки `/runs` (включая `X-Total-Count`), `/dashboard_data`,
   `/load_tests/active`, `/jobs`, `/services` остаются прежними: `tests/test_smoke_api.py`, `tests/test_ui_archive.py`,
   `tests/test_ui_dashboard.py` это проверяют. Новых эндпоинтов не планируется; если появится, ему нужна запись в
   `loadlens_app/auth/policy.py` и тест.
8. **Сдача.** Код, тесты, числа «до/после» по бенчу агента 1 (пока бенча нет — свой микрозамер, потом
   пересчитать), строка в таблице «Статус» и краткий отчёт: что сделано, что отклонилось от плана и почему.
9. **Коммиты** в своей ветке, небольшими шагами, с осмысленным английским сообщением. Ничего не пушить.

## 1. Инфраструктура работы

### Ветки и слияние

Перед стартом в основном дереве коммитится текущее состояние (25 изменённых файлов). Каждому агенту свой worktree:

```
git worktree add ../ll-perf-<имя> -b perf/<имя>
```

Слияние в `main` в порядке: bench → pool → config-cache → polling-static → reports-sql → run-index. Ожидаемые
конфликты и как их решать:

- `loadlens_app/migrations.py::run_all` — агенты 4 и 5 добавляют по одной записи; оставить обе.
- `docker/initdb/02_schema.sql` — агент 4 правит блок `llm_reports`, агент 5 добавляет таблицу после блока `metrics`.
- `loadlens_app/blueprints/dashboard.py` — агент 4 только константы и `_runs_max_rps_sql`, агент 5 только
  `list_runs` и запрос `runs_total` в `dashboard_data`, агент 6 только декораторы/обёртки роутов и
  вызовы инвалидации в `delete_run`, `rename_run`, `demo_seed`. Хунки далеко друг от друга.
- `loadlens_app/core.py` — агент 3 правит чтение конфигов (верхняя треть файла), агент 5 только `_run_tables`.

Запасной вариант, если worktree не подходит: все в одном дереве, без `git stash`, `checkout` и форматирования
чужих файлов, строго по владению.

### Песочница с локальной базой

- Контейнер (поднимает агент 1 первым, остальные только создают свои базы):

  ```
  docker run -d --name ll-perf-db -p 127.0.0.1:55432:5432 -e POSTGRES_USER=ltar -e POSTGRES_PASSWORD=e2e-only-password timescale/timescaledb:latest-pg16
  docker exec ll-perf-db createdb -U ltar ll_s<N>
  ```

  Это локальный одноразовый контейнер, не боевая инфраструктура. В конце волны `docker rm -f ll-perf-db`.
- Скрипт `%TEMP%\ll_ui\make_sandbox.py <N>` копирует рабочее дерево (tracked + untracked, без игнорируемых файлов)
  в `%TEMP%\ll_ui_s<N>\app`, пишет `settings.py` из примера с адресами только `127.0.0.1`, базу `ll_s<N>` на
  `127.0.0.1:55432` и runner `%TEMP%\ll_ui_s<N>\run_app.py` на порт `5070+N` с выключенным входом
  (`LOADLENS_AUTH_ENABLED=0`). Константа `REPO` внутри скрипта указывает на основное дерево: при работе в worktree
  сделать копию скрипта и поменять `REPO` на свой worktree. После правок кода запускать заново: он сбрасывает
  песочницу (кроме базы).
- Номер `N` = номер агента: агент 1 → порт 5071 и база `ll_s1`, агент 6 → 5076 и `ll_s6`. Порты 5060/5061 Chromium
  блокирует.
- Таблицы приложение создаёт само при первом обращении; индексы из `docker/initdb/02_schema.sql` можно накатить
  вручную через `docker exec -i ll-perf-db psql -U ltar ll_s<N> < docker/initdb/02_schema.sql`.
- Данные: `tools/seed_bench_data.py` агента 1 (до его появления есть только `POST /demo/seed` с одним прогоном).
- Браузерная проверка: Playwright для `C:/Python312/python.exe`, скрипты запускать с `-X utf8`; CSP без
  `unsafe-eval`, поэтому `page.wait_for_function` не работает, опрашивать `locator.inner_text()`. Сервер
  останавливать по PID из `netstat -ano`, убедившись, что это python.exe. На пустой базе `/dashboard_data` отвечает
  500 до первого отчёта (известно); после работы агента 5 `/runs` на пустой базе должен отвечать `[]`.

### Дополнительно для волны B: брокер, воркер и заглушки

- Redis для песочниц: `docker run -d --name ll-perf-redis -p 127.0.0.1:56379:6379 redis:7.4-alpine` (поднимает
  агент B1, остальные используют базы `0..6` по своему номеру: `redis://127.0.0.1:56379/<N>`).
- Воркер из каталога своей песочницы, в отдельном окне, с теми же переменными, что у runner'а:

  ```
  set CELERY_BROKER_URL=redis://127.0.0.1:56379/<N>
  set CELERY_RESULT_BACKEND=redis://127.0.0.1:56379/<N>
  set CELERY_TASK_ALWAYS_EAGER=0
  set LOADLENS_JOB_RUNNER=celery
  C:/Python312/python.exe -m celery -A loadlens_app.celery_app worker --loglevel=info --pool=solo -Q reports,celery
  ```

  На Windows работает только `--pool=solo` (один процесс, одна задача за раз); этого достаточно для проверки.
  Переменные `CELERY_*` и `LOADLENS_JOB_RUNNER` выставить и в `run_app.py` песочницы (скопировать и дописать).
- Заглушки внешних систем: `%TEMP%\ll_ctx_e2e\llm_stub.py` (OpenAI-совместимый, 127.0.0.1:5090, фиксированный
  ответ) и `%TEMP%\ll_ctx_e2e\confluence_stub.py` (127.0.0.1:5091, basic auth reader/secret). Агент B1 копирует
  их в `tools/stubs/` и добавляет заглушку Prometheus; песочница указывает на них через свой
  `settings_runtime.json`: `llm.provider=openai`, `llm.openai.api_base_url=http://127.0.0.1:5090/v1`,
  `confluence.base_url=http://127.0.0.1:5091`, `data_sources.prometheus.prometheus.url=http://127.0.0.1:5092`.
- В конце волны: `docker rm -f ll-perf-redis`.

## 2. Владение файлами

| Агент | Файлы | Что именно |
|---|---|---|
| 1 bench | `tools/seed_bench_data.py`, `tools/bench_endpoints.py`, `docs/perf-baseline.md` (новые); `docker/Dockerfile` (только `GUNICORN_CMD_ARGS`); `loadlens_app/auth/web.py` (только `_gate` и `_after_request`); `tests/test_perf_bench.py` | Замер |
| 2 pool | `AI/storage_state.py`; `settings.example.py` (блок `storage.timescale`); `tests/conftest.py` (одна строка сброса пула в `_storage_breaker_closed`); `tests/test_perf_pool.py` | Пул соединений |
| 3 config-cache | `loadlens_app/core.py` (функции чтения конфигов, строки до `_ts_conn`); точечные `copy.deepcopy` в `loadlens_app/blueprints/config_api.py`, `loadlens_app/projects.py`, `loadlens_app/blueprints/settings.py`, `loadlens_app/query_pack_service.py` по результатам аудита; `tests/test_perf_config_cache.py` | Кэш конфигов |
| 4 reports-sql | `AI/db_store.py` (`_ensure_llm_reports_table`, `save_llm_results`, новые функции `max_rps_*`); `loadlens_app/migrations.py` (новая функция + строка в `run_all`); `loadlens_app/blueprints/dashboard.py` (только `DASHBOARD_MAX_RPS_SQL`, `RUNS_MAX_RPS_FIELDS`, `_runs_max_rps_sql`); `docker/initdb/02_schema.sql` (блок `llm_reports`); `tests/test_perf_reports_sql.py` | `max_rps`, индексы |
| 5 run-index | `AI/db_store.py` (`save_domain_labeled`, новые функции `*_run_index*`); `loadlens_app/migrations.py` (новая функция + строка в `run_all`); `loadlens_app/core.py` (только `_run_tables`); `loadlens_app/blueprints/dashboard.py` (только `list_runs` и запрос `runs_total` в `dashboard_data`); `docker/initdb/02_schema.sql` (новый блок после `metrics`); `tests/test_perf_run_index.py` | Таблица прогонов |
| 6 polling-static | `loadlens_app/response_cache.py`, `loadlens_app/static_assets.py` (новые); `app.py`; `loadlens_app/jobs.py`; `loadlens_app/load_tests/watcher.py`; `loadlens_app/blueprints/load_tests.py` (роуты `active`, `create`, `stop`, `ci_events`); `loadlens_app/blueprints/dashboard.py` (обёртки роутов `dashboard_data`, `list_jobs`; вызовы инвалидации в `delete_run`, `rename_run`, `demo_seed`); все `templates/*.html` (только ссылки на `/static/` и `defer`); правки ожиданий `/static/` в `tests/test_auth.py`, `tests/test_load_tests_api.py`, `tests/test_ui_forms.py`, `tests/test_ui_polish*.py`; `tests/test_perf_polling.py` | Опросы и статика |

Никто не трогает: `update_page.py`, `AI/pipeline.py`, `static/js/*`, `static/css/*`, `loadlens_app/auth/policy.py`.

### Владение в волне B

| Агент | Файлы | Что именно |
|---|---|---|
| B1 bench | `tools/stubs/prom_stub.py`, `tools/stubs/llm_stub.py`, `tools/stubs/confluence_stub.py` (новые); `tools/bench_endpoints.py`; `docs/perf-baseline.md`; `tests/test_perf_bench.py` | Заглушки и замер под нагрузкой |
| B2 celery-infra | `loadlens_app/celery_app.py`; `docker/docker-compose.yml`; `docker/env.example`; `tests/test_perf_celery_infra.py` | Конфигурация Celery и воркера |
| B3 job-store | `loadlens_app/jobs.py`; `docker/initdb/02_schema.sql` (блок `report_jobs`); `tests/test_perf_job_store.py` | Таблица задач как источник правды |
| B4 tasks-runner | `loadlens_app/tasks.py`, `loadlens_app/job_runner.py` (новые); `loadlens_app/blueprints/dashboard.py` (только `start_report_job`, `publish_confluence`, `_job_failure_message`); `loadlens_app/blueprints/forecast.py` (только `_publish_in_background`, `_failure_message`); `tests/test_perf_tasks.py` | Задачи и выбор режима |
| B5 pipeline-worker | `update_page.py`; `data_collectors/temp_files.py` (если потребуется); `tests/test_perf_pipeline_worker.py` | Отчёт без вложенных подзадач |
| B6 e2e-docs | `docs/jobs-runner.md` (новый); `README.md`; `docker/DOCKER_SETUP.md`; `tools/e2e_jobs/*` (новые); `tests/test_perf_ops_docs.py` | Сквозная проверка и документация |

Никто в волне B не трогает `AI/*`, `static/*`, `templates/*`, `loadlens_app/core.py`, `loadlens_app/limits.py`,
`loadlens_app/load_tests/*`.

## 3. Агент 1 — bench (этап 0)

Цель: за полдня дать остальным инструмент замера; затем снять базовые числа. Сначала сдаётся скрипт бенча и сид,
потом документ с базой.

### Что сделать

1. **Время ответа в логе gunicorn.** В `docker/Dockerfile` в `GUNICORN_CMD_ARGS` добавить
   `--access-logformat` с полем `%(M)s` (миллисекунды), остальное как у формата по умолчанию. На Windows gunicorn
   не запускается; проверка строки — при деплое, об этом написать в отчёте.
2. **Заголовок `Server-Timing`.** В `loadlens_app/auth/web.py`: в начале `_gate` запомнить `g.request_started =
   time.perf_counter()`; в `_after_request` добавить `Server-Timing: app;dur=<мс с одним знаком>`. Для всех ответов,
   стоимость — микросекунды.
3. **`tools/seed_bench_data.py`.** Пишет синтетические данные напрямую в базу песочницы через psycopg2, не через
   приложение. Аргументы: `--dsn` (или host/port/dbname/user/password), `--runs 300`, `--services 10`,
   `--minutes 60`, `--series 20`, `--seed 1`, `--reset`. Что пишет:
   - `public.metrics`: таблицу создаёт через `AI.db_store._ensure_schema_and_table`; для каждого прогона `minutes`
     минутных точек × `series` рядов домена `lt` (`query_label` вида `rps_total`, `errors_total`, `latency_p95`)
     плюс несколько рядов `jvm`, с `run_name` = `bench-<svc>-<k>`, `run_id` = uuid, `service` = `svc01..svcNN`,
     `start_ms`/`end_ms`. При параметрах по умолчанию это порядка 0.5–1 млн строк; `--runs 1000` даёт 2–3 млн,
     чтобы увидеть рост `/runs`. Вставка через `execute_batch` пачками по 5000.
   - `public.llm_reports`: таблицу создаёт через `_ensure_llm_reports_table`; на каждый прогон одна строка
     `domain='final'` с `verdict`/`sla_verdict` из набора вердиктов дашборда (`DASHBOARD_VERDICTS`), `test_type`
     из `step/soak/spike/stress`, `parsed` с `peak_performance.max_rps` (у части строк число строкой вида `~1200`,
     чтобы проверить разбор), `created_at` размазанное по последним 90 дням, плюс по 3 строки других доменов
     (`jvm`, `kafka`, `hard_resources`) с текстом, чтобы таблица была реалистично шире финальных строк. У 10 %
     прогонов две финальные строки (повторный анализ), как в проде.
   - `settings_runtime.json` **песочницы** (путь передаётся `--runtime-json`, это файл в `%TEMP%`, не секрет):
     область `bench` с сервисами `svc01..svcNN` в `per_area`, чтобы фильтр области их видел.
   - Идемпотентность: `--reset` чистит строки с `run_name LIKE 'bench-%'`.
4. **`tools/bench_endpoints.py`.** Аргументы: `--base http://127.0.0.1:5071`, `--area bench` (ставится cookie
   `project_area`), `--rounds 20`, `--concurrency 1`, `--json out.json`, `--compare before.json`. Эндпоинты:
   `/`, `/dashboard_data`, `/runs?limit=20`, `/runs?limit=20&q=svc0`, `/runs?limit=20&sort=max_rps`,
   `/load_tests/active?limit=5&live=1`, `/load_tests/active?limit=0`, `/jobs?status=running&limit=10`, `/services`,
   `/project_areas`, `/static/js/settings.js`, `/static/css/app.css`. Для каждого: p50, p95, max, значение
   `Server-Timing`, размер ответа. Печатает markdown-таблицу; с `--compare` добавляет колонку «было» и
   отношение. Режим `--concurrency 4` нужен, чтобы увидеть сериализацию на RLock конфигов и GIL.
5. **`docs/perf-baseline.md`.** Таблица бенча при `--concurrency 1` и `4` на данных по умолчанию и на `--runs 1000`;
   `EXPLAIN (ANALYZE, BUFFERS)` CTE `base` из `/runs` и двух самых тяжёлых запросов `/dashboard_data`
   (выполнить через psycopg2 в базе песочницы); `SELECT count(*)` по `metrics` и `llm_reports`; как повторить.
6. **Тесты `tests/test_perf_bench.py`:** заголовок `Server-Timing` есть и содержит число; чистые функции сида
   (построители строк) дают детерминированные ряды при одном `--seed`; функция статистики бенча (p50/p95) на
   известном массиве.

### Приёмка

- Бенч и сид запускаются одной командой каждый, описаны в шапке файла и в `docs/perf-baseline.md`.
- Остальные агенты могут повторить «до/после» в своей песочнице за 5 минут.

## 4. Агент 2 — pool (этап 1)

Цель: одно физическое соединение на поток вместо нового `psycopg2.connect` на каждый вызов. Все вызовы в коде
(`core._ts_conn`, `AI.db_store._connect`, `PgLoadTestStore`, `jobs.py`, auth-репозиторий) уже идут через
`AI/storage_state.py::connect`, поэтому меняется только он. `loadlens_app/connection_checks.py` открывает
соединение напрямую нарочно (проверка настроек) и остаётся как есть.

### Что сделать

1. **Пул.** В `storage_state` пул на процесс на каждую пару (адрес, параметры): `psycopg2.pool.ThreadedConnectionPool`
   или своя очередь из `queue.LifoQueue` (LIFO держит «тёплые» соединения). Размер из `storage.timescale.pool_max`
   (по умолчанию 12: 8 потоков gunicorn + watcher + задачи + запас), `pool_idle_sec` (по умолчанию 300: простоявшее
   дольше соединение закрывается при возврате). Ключи описать в `settings.example.py` рядом с
   `connect_timeout_sec`.
2. **Обёртка соединения.** `connect()` возвращает объект, который делегирует всё внутреннему соединению, а
   `close()` возвращает его в пул: если `closed` или `get_transaction_status()` не `IDLE`/`INTRANS` → физически
   закрыть и выбросить; иначе `rollback()` (дёшево и гарантирует, что транзакция не утечёт), `autocommit = False`,
   вернуть. Повторный `close()` — no-op. `__enter__`/`__exit__` делегируются внутреннему соединению: по коду
   `with conn, conn.cursor() as cur:` означает commit/rollback, а не закрытие. `_ensure_*`-функции переключают
   `autocommit` вручную — сброс при возврате обязателен.
3. **Выдача.** При получении из пула соединение с `closed != 0` выбрасывается и берётся следующее; живость
   проверять `SELECT 1` только если соединение простояло дольше `pool_idle_sec/2`, не на каждую выдачу. Пул пуст и
   лимит исчерпан → открыть обычное непуловое соединение (его `close()` закрывает физически), записать в лог на
   уровне debug; не блокировать и не бросать.
4. **Брейкер.** Открытие нового физического соединения проходит через `BREAKER.connect` как сейчас (учёт
   падений и восстановления не меняется). При записи падения пул очищается (`closeall`), чтобы после
   восстановления базы не раздавать мёртвые соединения. `OperationalError` во время работы ловит код вызова и
   делает `conn.close()` в `finally` — обёртка увидит `closed`/ошибочный статус и выбросит соединение.
5. **Диагностика.** `storage_state.pool_stats()` → `{"opened": n, "reused": m, "in_use": k, "idle": j}`; счётчик
   `opened` нужен для замера и для теста.
6. **Тесты `tests/test_perf_pool.py`** на фейковом `psycopg2.connect` (образец `tests/test_storage_state.py`):
   два последовательных `connect()/close()` → одно физическое открытие; `closed` соединение не переиспользуется;
   после `close()` `autocommit` сброшен и был вызван `rollback`; падение брейкера очищает пул; исчерпанный пул даёт
   непуловое соединение; `with conn:` не закрывает соединение; 8 потоков × 50 итераций без ошибок и не больше
   `pool_max` открытий. В `tests/conftest.py::_storage_breaker_closed` добавить сброс пулов между тестами.
   Существующий `tests/test_storage_state.py` проходит без правок.

### Приёмка

- Бенч агента 1: `/jobs`, `/load_tests/active`, `/services` при `--concurrency 4` быстрее минимум в 2 раза;
  `pool_stats()["opened"]` после 100 запросов ≤ `pool_max`.
- Полный прогон тестов зелёный.

## 5. Агент 3 — config-cache (этап 1)

Цель: чтение runtime-конфигов стоит микросекунды, а не 2.5–9 мс на вызов. Замер 2026-10-16 на реальных файлах:
`_per_area_config` 2.5 мс, `_active_metrics_config` 7 мс, `_resolve_services_filter` 9 мс; вызовов несколько на
запрос (config_api 13 мест, projects 12, load_tests/service 9), все под одним `_RUNTIME_FILES_LOCK`.

### Что сделать

1. **Кэш файла по сигнатуре.** В `loadlens_app/core.py` новая функция `_cached_json(path)`: один `os.stat`,
   ключ `(st_mtime_ns, st_size)`; при совпадении возвращает уже разобранный словарь, иначе читает под
   `_RUNTIME_FILES_LOCK`, разбирает и запоминает. `RuntimeConfigError` не кэшируется. `_write_json_atomic` после
   записи обновляет запись кэша сразу (защита от одинаковых mtime при быстрых перезаписях и от bind-mount, где
   файл переписывается на месте). Другие процессы (Celery-воркер) увидят изменение по `stat`.
2. **Читатели и писатели.** `_load_settings_runtime_data()` и `_load_metrics_runtime_data()` остаются
   «свежими» для путей записи (`runtime = _load...; runtime[...] = ...; _save...`). Читатели переводятся на кэш:
   `_per_area_config`, `_area_entry`, `_services_map_for_area`, `_service_meta`, `_service_disabled_domains`,
   `_service_area_map`, `_metrics_services_for_area`, `_metrics_service_entry`, `_available_domain_keys`,
   `_resolve_services_for_area`, `_find_area_for_service`, `_list_project_areas`, `_prompt_templates_for_scope`
   и всё, что найдёт аудит.
3. **`_active_metrics_config()`.** Нормализованный результат кэшируется по паре сигнатур (файл метрик + файл
   настроек, так как `_normalize_metrics_config` смотрит `per_area`). Возвращается общий объект без deepcopy.
4. **Аудит мутаций.** Пройти всех вызывающих `_active_metrics_config`, `_per_area_config`, `_services_map_for_area`,
   `_metrics_services_for_area` в `loadlens_app/**` и `AI/query_packs/*`: кто изменяет результат (`pop`,
   присваивание, `setdefault`, `update`). Таким вызовам добавить `copy.deepcopy(...)` на месте, в файлах из таблицы
   владения; список правок в отчёт. Читатели в `update_page.py` (`_effective_config_for_scope` и соседи) работают в
   задаче отчёта и в эту волну не трогаются.
5. **Тесты `tests/test_perf_config_cache.py`** с временными файлами (патчить `core.CONFIG_RUNTIME_PATH`,
   `core.METRICS_RUNTIME_PATH` и `update_page._SETTINGS_RUNTIME_PATH`): второй вызов не читает диск (считать
   через monkeypatch `Path.read_text`); изменение файла (другой размер или `os.utime`) → перечитано; `_save_*` →
   следующий вызов видит новые данные без изменения mtime; битый JSON → `RuntimeConfigError` и не застревает в
   кэше; 8 потоков читают параллельно без ошибок; результат `_active_metrics_config` стабилен между вызовами
   (одинаковый `id` при неизменных файлах).
6. **Микрозамер** в отчёте: те же три функции до/после на файлах того же размера (сгенерировать 227 КБ / 54 КБ
   JSON во временной папке; настоящие файлы не открывать).

### Приёмка

- `_resolve_services_filter` < 0.2 мс, `_active_metrics_config` < 1 мс на горячем пути.
- Бенч агента 1: `/services`, `/project_areas`, `/load_tests/active` при `--concurrency 4` заметно быстрее.
- Полный прогон тестов зелёный; тесты, читающие конфиги (`test_context_scope.py`, `test_context_import.py`,
  `test_query_packs_api.py`, `test_query_packs_phase2.py`, `test_smoke_api.py`), проходят без правок.

## 6. Агент 4 — reports-sql (этапы 1–2)

Цель: запросы `/dashboard_data` и финальная часть `/runs` перестают разбирать JSONB регулярными выражениями в
каждой строке и получают индексы под свои фильтры.

### Что сделать

1. **Одно место для правила `max_rps`.** В `AI/db_store.py`: кортеж полей (сейчас `RUNS_MAX_RPS_FIELDS` в
   `dashboard.py`: `peak_performance.max_rps`, `stability_under_load.actual_rps`, `spike_analysis.max_spike_rps`,
   в этом порядке) и две функции: `max_rps_from_parsed(parsed) -> float | None` (Python, то же правило, что и SQL:
   только «чистое» число, `~1200` → `None`, ограниченный показатель степени) и `max_rps_sql(parsed_column) -> str`
   (текущее выражение из `_runs_max_rps_sql`). `dashboard.py` импортирует их: `RUNS_MAX_RPS_FIELDS` становится
   реэкспортом, `_runs_max_rps_sql(parsed_column)` сохраняет сигнатуру (её вызывает `list_runs`, который
   переписывает агент 5) и возвращает `COALESCE(max_rps, <max_rps_sql(parsed_column)>)`; `DASHBOARD_MAX_RPS_SQL`
   — то же с колонкой `parsed`. `COALESCE` оставляется навсегда: для заполненных строк второе выражение не
   вычисляется, для строк старого кода остаётся запасной разбор.
2. **Колонка и индексы.** В `_ensure_llm_reports_table`: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS max_rps NUMERIC`;
   частичные индексы `idx_<table>_final_created ON (created_at DESC) WHERE domain = 'final'` и
   `idx_<table>_final_service_created ON (service, created_at DESC) WHERE domain = 'final'`. Каждая команда в
   своём `try`, как существующие. То же в `docker/initdb/02_schema.sql` в блоке `llm_reports`.
3. **Запись.** `save_llm_results`: для строки `domain='final'` считать `max_rps_from_parsed(parsed)` и писать в
   колонку (добавить в список колонок INSERT и в кортеж значений; у остальных доменов `NULL`).
4. **Миграция.** `loadlens_app/migrations.py::backfill_max_rps()`: под advisory lock (свой номер) `UPDATE
   <llm_table> SET max_rps = <max_rps_sql('parsed')> WHERE domain = 'final' AND max_rps IS NULL AND
   <max_rps_sql('parsed')> IS NOT NULL`, пачками по 5000 `id`, возвращает число строк; запись в `run_all`.
   Отсутствие колонки (таблица ещё не прошла `_ensure`) → сначала `_ensure_llm_reports_table`.
5. **Тесты `tests/test_perf_reports_sql.py`:** приоритет полей и разбор чисел в `max_rps_from_parsed` (включая
   `~1200`, `1.2e3`, `1e400`); `save_llm_results` передаёт `max_rps` в параметры INSERT (фейковый курсор,
   образец в `tests/test_smoke_api.py`); SQL миграции строится и выполняется на фейковом соединении пачками;
   `_runs_max_rps_sql("parsed")` начинается с `COALESCE(max_rps`; `_ensure_llm_reports_table` выполняет ALTER и
   оба индекса. Существующие `test_ui_dashboard.py`, `test_smoke_api.py` проходят без правок.

### Приёмка

- В песочнице после сида и `python -m loadlens_app.migrations` (из каталога песочницы) все финальные строки с
  числовым `max_rps` заполнены; `EXPLAIN` запросов `/dashboard_data` использует частичные индексы.
- Бенч агента 1: `/dashboard_data` и `/runs?sort=max_rps` быстрее; числа в «Статус».

## 7. Агент 5 — run-index (этап 2)

Цель: список прогонов и счётчик прогонов читают маленькую таблицу `run_index`, а не агрегируют гипертаблицу
`metrics`. Единственный писатель `metrics` — само приложение (`AI/db_store.py::save_domain_labeled`), поэтому
таблица поддерживается при записи, а миграция заполняет её для существующих данных и досчитывает на старте.

### Что сделать

1. **Таблица.** `AI/db_store.py::_ensure_run_index_table(conn, storage_cfg)` с guard'ом на процесс, как
   `_ENSURED_TABLES`: `<schema>.run_index (run_name TEXT PRIMARY KEY, service TEXT NOT NULL DEFAULT '',
   run_id TEXT NOT NULL DEFAULT '', start_time TIMESTAMPTZ NOT NULL, end_time TIMESTAMPTZ NOT NULL,
   updated_at TIMESTAMPTZ NOT NULL DEFAULT now())`, индекс `(service, end_time DESC)`. То же в
   `docker/initdb/02_schema.sql` после блока `metrics`.
2. **Upsert.** `upsert_run_index(conn, storage_cfg, run_name, service, run_id, start_time, end_time)`:
   `INSERT ... ON CONFLICT (run_name) DO UPDATE SET start_time = LEAST(...), end_time = GREATEST(...),
   service = COALESCE(NULLIF(EXCLUDED.service, ''), run_index.service), run_id = COALESCE(NULLIF(EXCLUDED.run_id,
   ''), run_index.run_id), updated_at = now()`. Пустой `run_name` — пропуск (такие строки `/runs` и так не видит).
3. **Запись.** В `save_domain_labeled` после вставки пачек, в той же транзакции: min/max колонки `time` по
   `long_df` всех рядов домена и один upsert на вызов.
4. **Досчёт.** `refresh_run_index(conn, storage_cfg, since=None) -> int`: `INSERT ... SELECT run_name,
   MAX(service), MAX(NULLIF(run_id, '')), MIN(time), MAX(time) FROM metrics WHERE run_name <> '' [AND time > %s]
   GROUP BY run_name ON CONFLICT ...` с теми же LEAST/GREATEST. Миграция `loadlens_app/migrations.py::
   backfill_run_index()` под advisory lock: таблица пуста → полный проход; иначе `since = MAX(end_time) − 1 день`
   (перекрытие на поздние данные; исключение чанков по времени делает это дешёвым). Запись в `run_all`, то есть
   выполняется при каждом старте с `LOADLENS_RUN_MIGRATIONS=1` и из CLI `python -m loadlens_app.migrations`.
5. **Удаление и переименование.** В `loadlens_app/core.py::_run_tables` добавить `run_index`: `_delete_run_data`
   и `_rename_run_data` перебирают этот кортеж и обработают её сами. Проверить, что `UPDATE ... SET run_name`
   не конфликтует с первичным ключом (переименование в существующее имя отсекается раньше).
6. **`/runs`.** В `dashboard.py::list_runs`: CTE `base` читает `run_index` (`start_time`, `end_time`, `service`,
   `run_id`) с фильтрами по сервисам и `q`; один SQL вместо двух, `COUNT(*) OVER () AS total` в выборке,
   `X-Total-Count` из него (0 строк → `total` = 0). Сортировки, `NULLS LAST`, tiebreak по `run_name`, `offset/limit`,
   элементы ответа (`_runs_item`) не меняются. Вызовы `_runs_max_rps_sql("parsed")` и `_runs_test_type_sql`
   остаются как есть (их внутренности правит агент 4). Перед запросом `_ensure_run_index_table`, как сейчас
   `_ensure_llm_reports_table`: пустая база отвечает `200` и `[]`, а не `500`.
7. **`/dashboard_data`.** Только запрос `runs_total`: `SELECT COUNT(*) FROM run_index` с тем же фильтром сервисов.
   Остальное тело функции не трогать.
8. **Проверка согласованности.** `tools/check_run_index.py` (небольшой, в своём владении) сравнивает `run_index` с
   `GROUP BY` по `metrics` в базе песочницы и печатает расхождения; результат в отчёт.
9. **Тесты `tests/test_perf_run_index.py`:** SQL upsert'а и LEAST/GREATEST (фейковый курсор); `save_domain_labeled`
   на маленьком DataFrame вызывает upsert с правильными min/max и не вызывает при пустом `run_name`; `refresh`
   строит полный и инкрементальный запросы; `_run_tables` содержит `run_index`; `/runs` выполняет один запрос,
   берёт `X-Total-Count` из колонки `total`, сохраняет порядок и поля ответа (фейковое соединение, образец
   `tests/test_ui_archive.py`); `/runs` при пустой таблице → `200` и `[]`; `/dashboard_data` считает `runs_total`
   по `run_index`.

### Приёмка

- В песочнице после сида и миграции `check_run_index.py` без расхождений; удаление и переименование прогона через
  UI обновляют список.
- Бенч агента 1 на `--runs 1000`: `/runs` быстрее минимум в 5 раз; `EXPLAIN` без `Seq Scan`/`GroupAggregate` по
  `metrics`.

## 8. Агент 6 — polling-static (этапы 1–2)

Цель: N открытых вкладок не превращаются в N одинаковых запросов к базе, watcher не ходит в базу вхолостую,
статика кэшируется браузером.

### Что сделать

1. **`loadlens_app/response_cache.py`.** `TTLCache`: словарь + `Lock`, `get_or_set(key, ttl, fn)`,
   `invalidate(prefix)`; не больше 256 записей, просроченные выбрасываются при вставке; кэш на процесс (TTL
   ограничивает рассинхрон с Celery-воркером, которому он и не нужен). Кэшируются только ответы `200`: обёртка
   роута вызывает исходную функцию, запоминает `response.get_data()` и отдаёт `Response(..., mimetype=
   "application/json")`; тело роутов не меняется.
2. **Применить.**
   - `/dashboard_data`: ключ `("dashboard", кортеж фильтра сервисов)`, TTL 10 с.
   - `/load_tests/active`: ключ `("active", сервисы, limit, live, пользователь, is_admin)` — `can_stop` зависит от
     пользователя; TTL 3 с. Инвалидация `active` в роутах `create`, `stop`, `ci_events`.
   - `/jobs`: ключ `("jobs", статусы, limit)`, TTL 3 с. Инвалидация в `JobStore.create/update/dismiss`; когда
     статус задачи становится `done`/`error`, инвалидировать и `dashboard` (отчёт сохранён).
   - `delete_run`, `rename_run`, `demo_seed` в `dashboard.py`: инвалидация `dashboard`.
3. **Пауза watcher.** `loadlens_app/load_tests/watcher.py`: если `tick()` вернул 0, ждать `IDLE_TICK_SEC = 15`,
   иначе `TICK_SEC`; ожидание на `threading.Event` `_wake`, метод `wake()` будит цикл; `ensure_watcher()` зовёт
   `wake()` и когда поток уже запущен (роуты вызывают его, увидев активный прогон, а `create` — сразу после
   создания). Остановка через `_stop` работает как раньше.
4. **Статика.** `loadlens_app/static_assets.py::static_url(path)` → `/static/<path>?v=<первые 10 символов sha1
   содержимого>`, с `lru_cache` по пути; при `LOADLENS_STATIC_NO_CACHE=1` пересчитывать по mtime. В `app.py`:
   `SEND_FILE_MAX_AGE_DEFAULT = 31536000` и регистрация `static_url` как Jinja-глобала. Во всех `templates/*.html`
   ссылки `/static/...` (30 штук) заменить на `{{ static_url('...') }}`; `chart.umd.js` в пяти шаблонах получает
   `defer` — проверить, что он стоит раньше `charts.js` и скриптов страниц (defer выполняется по порядку в
   документе) и что ни один inline-скрипт с nonce не обращается к `Chart` синхронно. Эндпоинт `static` в
   `policy.py` публичный, поэтому `Cache-Control: no-store` на него не ставится; убедиться тестом. Ссылка
   `/assets/logo.png?c=` не трогается.
5. **Правки чужих тестов** только там, где зашит точный `/static/...`: `tests/test_auth.py`,
   `tests/test_load_tests_api.py`, `tests/test_ui_forms.py`, `tests/test_ui_polish*.py` — принимать суффикс `?v=`.
6. **Тесты `tests/test_perf_polling.py`:** семантика `TTLCache` (TTL, prefix-инвалидация, лимит); второй
   `/dashboard_data` за 10 с не выполняет SQL (фейковое соединение считает `execute`); ключ зависит от области
   (cookie `project_area`); `/load_tests/active` для двух пользователей не делит кэш; `JobStore.update` сбрасывает
   `jobs`; watcher с фейковыми часами ждёт 15 с без прогонов и 3 с с прогонами, `wake()` прерывает ожидание;
   `static_url` стабилен и меняется при изменении файла; у `/static/css/app.css` заголовок
   `Cache-Control: public, max-age=31536000`; во всех шаблонах `chart.umd.js` с `defer`.
7. **Проверка в браузере.** Playwright в своей песочнице: дашборд, архив, новый отчёт, запуск теста, отчёт демо-
   прогона — без ошибок консоли; второй переход на страницу берёт статику из кэша (счётчик `page.on("request")`
   по `/static/`). Числа в отчёт.

### Приёмка

- Бенч агента 1 при `--concurrency 4`: `/jobs`, `/load_tests/active`, `/dashboard_data` быстрее минимум в 3 раза
  на втором и последующих запросах; `/static/js/settings.js` отдаётся с кэш-заголовком.
- Полный прогон тестов зелёный, браузерная проверка без ошибок консоли.

## 9. Приёмка волны A

1. Все шесть веток слиты в порядке из раздела 1, полный прогон тестов зелёный (не меньше 1199 старых плюс новые).
2. Таблица «Статус» заполнена числами из одного и того же бенча на одних параметрах сида; `docs/perf-baseline.md`
   дополнен колонкой «после».
3. Ориентиры: `/runs` ≥ 5× на `--runs 1000`; `/dashboard_data` ≥ 2×; `/jobs` и `/load_tests/active` при
   `--concurrency 4` ≥ 3×; физических соединений на 100 опросов ≤ `pool_max`; `/services` < 5 мс.
4. Браузерная проверка пяти страниц в песочнице без ошибок консоли; контракты API не изменились.
5. `settings.example.py` описывает `pool_max`/`pool_idle_sec`; `docs/perf-plan.md` содержит отклонения от плана.
6. Контейнер `ll-perf-db` удалён, worktree'и убраны (`git worktree prune`).

## 10. Риски и решения волны A

| Риск | Решение |
|---|---|
| Пул раздаёт соединение с незакрытой транзакцией или включённым `autocommit` | `rollback()` и сброс `autocommit` при каждом возврате; тест на оба случая |
| После падения базы пул полон мёртвых соединений | Очистка пула при записи падения в брейкере; проверка `closed` при выдаче |
| Кэшированный конфиг изменён читателем, и все запросы видят мусор до следующей записи файла | Аудит вызовов с `deepcopy` там, где мутируют; тест на стабильность объекта |
| `run_index` разошёлся с `metrics` (сбой между вставкой и upsert'ом) | Upsert в той же транзакции; досчёт на каждом старте; `tools/check_run_index.py` |
| TTL-кэш показывает «нет активных прогонов» через 3 с после запуска | Инвалидация в `create`/`stop`; TTL 3 с как верхняя граница |
| Хэш в URL статики ломает тесты и кэш CSP | Правки только в перечисленных тестах; `nonce` не затрагивается, `static` остаётся публичным |
| Конфликты слияния в `dashboard.py`, `db_store.py`, `migrations.py` | Владение по функциям из раздела 2; порядок слияния из раздела 1 |
| Эффект на проде меньше, чем в песочнице (другой объём, сеть, TLS) | Поле `%(M)s` в access-логе и `Server-Timing` остаются и дают честные числа после деплоя |

## 11. Волна B — этап 3: тяжёлые задачи вне веб-процесса

Сейчас `start_report_job`, `publish_confluence` и `_publish_in_background` запускают `threading.Thread` в
процессе gunicorn (`--workers 1 --threads 8`). Построение отчёта — pandas, matplotlib и вызовы LLM — держит GIL, и
на это время встают все потоки обработчиков запросов. Celery-воркер в compose уже есть (`celery_worker`, тот же
образ, те же настройки, eager-режим выключен), но ему отдают только скачивание картинок Grafana, логов Loki и
загрузку вложений.

### 11.0 Общие решения этапа (обязательны для всех агентов волны)

1. **Что переезжает.** Три фоновые задачи: отчёт (`update_report` из `start_report_job`, включая вызовы от
   watcher'а прогонов), публикация отчёта в Confluence (`publish_report`), публикация прогноза (`publish_forecast`).
   Объяснение прогноза и импорт контекста (синхронные вызовы LLM внутри запроса) не трогаем.
2. **Режим запуска** — `LOADLENS_JOB_RUNNER=auto|thread|celery`, по умолчанию `auto`: `celery`, если брокер задан
   и `CELERY_TASK_ALWAYS_EAGER` не `1`; иначе `thread` (как сейчас). Отправка в брокер не удалась
   (`kombu.exceptions.OperationalError`) → задача запускается потоком, в лог предупреждение. В тестах eager-режим
   включён по умолчанию в коде, поэтому тесты идут потоком и не меняются. Поток создаётся через
   `threading.Thread(target=..., daemon=True)` — `tests/test_smoke_api.py` подменяет именно этот атрибут.
3. **Источник правды о задачах** — таблица `report_jobs`. Новые колонки `owner` и `runner` (`thread`/`celery`).
   В памяти веб-процесса остаются только его thread-задачи; для celery-задач всегда читается таблица.
   `JobStore.update` работает из любого процесса (сейчас из воркера он молча возвращает `None`, и прогресс
   теряется). `running_counts` считается по таблице. Форма JSON `/jobs` и `/job_status` не меняется.
4. **Зависшие задачи.** Thread-задачи — как сейчас: при старте процесса строки `running` со старым `updated_at`
   становятся ошибками, но только с `runner='thread'`. Celery-задачи — sweeper: `running` и `updated_at` старше
   `LOADLENS_JOB_STALE_MIN` (по умолчанию 15 минут) → `error` «Задача прервана: воркер не отвечает». Чтобы долгий
   вызов LLM без прогресса не считался зависанием, задача в воркере раз в 30 с обновляет `updated_at`
   (heartbeat в отдельном потоке).
5. **Вложенные подзадачи** скачивания и загрузки внутри `update_report` (`download_*_task.delay()`,
   `upload_attachment_task.delay()` с ожиданием `_await_task`) заменяются локальным пулом потоков: внутри воркера
   ожидание подзадачи на том же пуле воркеров при concurrency 1 — взаимная блокировка, а после переезда отчёта в
   воркер эта индирекция ничего не даёт. Заодно уходит циклический импорт `update_page → celery_app → loadlens_app`.
6. **Лимиты** (`check_job_slot`, `consume_llm`, `DailyBudget`) остаются в веб-процессе: проверки идут до
   постановки задачи, процесс gunicorn один. `limits.py` не меняется.
7. **Celery.** Явный `include` модулей с задачами (сейчас задачи регистрируются побочным эффектом импорта пакета),
   отдельная очередь `reports`, `task_acks_late=False` (повтор отчёта после падения воркера не нужен: его пометит
   sweeper), `ignore_result=True` у задач отчётов, `worker_prefetch_multiplier=1`, короткая политика повторной
   отправки (`task_publish_retry_policy` с одной попыткой, `broker_connection_timeout` 2 с), лимиты времени для
   задачи отчёта из `LOADLENS_REPORT_TASK_LIMIT_SEC` (по умолчанию 3600 мягкий и +60 жёсткий).
8. **Воркер в compose:** `-Q reports,celery --concurrency ${WORKER_CONCURRENCY:-2} --max-tasks-per-child 20`
   (pandas не отдаёт память), память по умолчанию 2g, healthcheck через `celery inspect ping`.
9. **Что не меняется:** API `/create_report`, `/publish_confluence`, `/forecast/<id>/confluence`, `/jobs`,
   `/job_status`; сигнатура `start_report_job` (её зовёт watcher); `limits.py`; TTL-кэши веб-процесса из волны A
   не инвалидируются из воркера, рассинхрон не больше их TTL (3–10 с), это принято.
10. **Проверено заранее:** путь построения отчёта (`update_page.py`, `confluence_export.py`,
    `forecast_confluence.py`, `AI/*`) не использует `flask.request`, не пишет runtime-JSON (у воркера они смонтированы
    только на чтение) и получает `source_url` аргументом. Временные файлы — общий том
    `data_collectors/temporary_files` у app и воркера.

Порядок слияния волны B: job-store → celery-infra → pipeline-worker → tasks-runner → bench → e2e-docs. Агент B4
пишет код против API из 11.3 и 11.2 и в тестах подменяет их; интеграция при слиянии.

### 11.1 Агент B1 — bench: заглушки и замер под нагрузкой

Цель: честно измерить, как построение отчёта влияет на остальные запросы, в режимах `thread` и `celery`.

1. **`tools/stubs/prom_stub.py`.** HTTP-заглушка Prometheus на 127.0.0.1:5092: `/api/v1/query_range` и
   `/api/v1/query` отвечают синтетическими рядами на любой запрос (число рядов и точек из параметров запуска,
   детерминированно по seed), `/api/v1/status/buildinfo` для проверки подключения. `llm_stub.py` и
   `confluence_stub.py` скопировать из `%TEMP%\ll_ctx_e2e\` в `tools/stubs/` с шапкой «как запускать».
2. **Песочница с заглушками.** Описать в `docs/perf-baseline.md`, какие ключи дописать в `settings_runtime.json`
   песочницы (раздел 1), и убедиться, что `POST /create_report` с `web_only: true` на области `bench` доходит до
   `done` и отчёт открывается. Если конфигурация источников данных требует иного (см. `AI/data_sources.py`,
   `settings.example.py::data_sources`), описать найденный способ.
3. **Бенч `--during-job`.** Запускает `POST /create_report` (`web_only: true`, `use_llm: true` со
   стабом LLM, окно 60 минут из посеянных данных), пока задача не `done`, параллельно замеряет остальные
   эндпоинты и выводит таблицу «idle / during-job» p50/p95 плюс длительность задачи. Режим задаётся переменной
   окружения runner'а песочницы: прогнать `LOADLENS_JOB_RUNNER=thread` и `celery` (воркер поднят).
4. **`docs/perf-baseline.md`**: раздел «Этап 3» с обеими таблицами до изменений (сейчас оба режима фактически
   `thread`) — это базовая линия для B6.
5. **Тесты `tests/test_perf_bench.py`:** ряды заглушки детерминированы и в формате Prometheus; парсер таблицы
   `--during-job`.

### 11.2 Агент B2 — celery-infra

1. **`loadlens_app/celery_app.py`:** `include=["loadlens_app.tasks"]`; `task_routes` для `loadlens.*` → очередь
   `reports`; `task_acks_late=False`; `worker_prefetch_multiplier=1`; `broker_connection_timeout=2`;
   `task_publish_retry_policy={"max_retries": 1, "interval_start": 0, "interval_max": 0.5}`;
   `broker_connection_retry_on_startup=True`. Лимиты времени задач скачивания (600/660) оставить как общие,
   лимиты отчёта задаёт сама задача (11.4). Функция `celery_mode_enabled() -> bool`: брокер задан,
   `CELERY_TASK_ALWAYS_EAGER != "1"`, `LOADLENS_JOB_RUNNER` не `thread`; при `celery` — `True` всегда.
   Запасной класс `Celery` для окружений без пакета celery должен поддерживать `apply_async`.
2. **`docker/docker-compose.yml`:** у `celery_worker` команда с `-Q reports,celery --concurrency
   ${WORKER_CONCURRENCY:-2} --max-tasks-per-child 20`, `WORKER_MEMORY_LIMIT` по умолчанию `2g`, healthcheck
   `celery -A loadlens_app.celery_app inspect ping -d celery@$$HOSTNAME`; у `app` и воркера новые переменные
   `LOADLENS_JOB_RUNNER` (`auto`), `LOADLENS_JOB_STALE_MIN` (`15`), `LOADLENS_REPORT_TASK_LIMIT_SEC` (`3600`);
   `docker/env.example` с комментариями к ним.
3. **Тесты `tests/test_perf_celery_infra.py`:** значения конфигурации (`include`, маршруты, `acks_late`,
   политика отправки); `celery_mode_enabled()` на сочетаниях переменных окружения; compose разбирается PyYAML и
   команда воркера содержит `-Q reports`.

### 11.3 Агент B3 — job-store

Файл `loadlens_app/jobs.py`; образец для тестов — `tests/test_jobs_dismiss.py::FakeJobsTable`.

1. **Колонки** `owner TEXT NOT NULL DEFAULT ''` и `runner TEXT NOT NULL DEFAULT 'thread'` через
   `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` по образцу `_ensure_dismissed_column`; нет прав на ALTER →
   флаг `_columns_supported=False`, предупреждение в лог, счётчики из памяти как сейчас. То же в
   `docker/initdb/02_schema.sql`. `JobRecord.runner`; `owner` и `runner` в `_JOB_COLUMNS`, `from_row`, `_persist`;
   `to_dict` без новых ключей.
2. **API** (его ждёт B4):
   - `create(kind, *, run_name, service, message, owner, runner="thread") -> JobRecord`; в памяти остаются только
     `runner == "thread"`.
   - `update(job_id, **fields) -> JobRecord | None`: нет в памяти → `_load_one` из таблицы, применить изменения
     (`progress` не убывает), сохранить; в память не класть, если `runner != "thread"`.
   - `adopt_thread(job_id)`: запись переходит в `runner="thread"` и в память этого процесса (вызывается при
     откате на поток после неудачной отправки в брокер).
   - `heartbeat(job_id)`: только `updated_at = now()`.
   - `running_counts(owner) -> (total, for_owner)`: строки `running` с `updated_at` не старше порога sweeper'а
     плюс thread-задачи в памяти (объединение по `job_id`); таблица недоступна → только память.
   - `sweep_stale(max_age) -> list[str]`: `running AND runner='celery' AND updated_at < now - max_age` →
     `error`, сообщение «Задача прервана: воркер не отвечает», `error` по-английски; вызывается из `list()` не
     чаще раза в 30 с и доступен снаружи.
   - `_mark_interrupted` — только `runner='thread'`.
   - `list()`: для celery-задач строка таблицы побеждает память.
3. **Тесты `tests/test_perf_job_store.py`:** update «из другого процесса» доходит до таблицы и не кладёт запись в
   память; `running_counts` по таблице с порогом свежести и объединением с памятью; `sweep_stale` трогает только
   celery; `_mark_interrupted` только thread; `create` сохраняет `owner`/`runner`; `adopt_thread`; без прав на
   ALTER — поведение как сейчас. Существующие `test_jobs_dismiss.py` и `test_smoke_api.py` проходят без правок.

### 11.4 Агент B4 — tasks-runner

1. **`loadlens_app/tasks.py`.** Три задачи с именами `loadlens.report`, `loadlens.publish_confluence`,
   `loadlens.publish_forecast` (`bind=True`, `ignore_result=True`, `soft_time_limit`/`time_limit` из
   `LOADLENS_REPORT_TASK_LIMIT_SEC`). Аргументы только JSON-сериализуемые: `job_id` и именованные параметры текущих
   `_runner`-замыканий (`start`, `end`, `service`, `project_area`, `use_llm`, `save_to_db`, `web_only`, `run_name`,
   `test_type`, `run_tag`, `run_filter_packs`; `run_name`, `service`, `source_url`; `run_id`, `target_rps`,
   `headroom_pct`, `cpu_ceiling_pct`, `source_url`). Тело каждой задачи — то же, что сейчас в `_runner`:
   прогресс через `job_store.update`, по завершении `done` с теми же полями, при исключении `error` с
   `_job_failure_message`/`_failure_message`; `SoftTimeLimitExceeded` → `error` «превышено время задачи». Вокруг
   вызова — heartbeat-поток (`job_store.heartbeat` раз в 30 с), который останавливается в `finally`.
2. **`loadlens_app/job_runner.py`.** `mode() -> "thread" | "celery"` по `LOADLENS_JOB_RUNNER` и
   `celery_app.celery_mode_enabled()`; `submit(task, job_id, **kwargs) -> str` (режим, который реально сработал):
   `celery` → `task.apply_async(kwargs=..., queue="reports")`, при `kombu.exceptions.OperationalError` (и любом
   исключении из kombu) предупреждение в лог, `job_store.adopt_thread(job_id)` и запуск потоком; `thread` →
   `threading.Thread(target=lambda: task.run(job_id=job_id, **kwargs), daemon=True).start()`.
3. **Роуты.** `dashboard.py::start_report_job` и `publish_confluence`, `forecast.py::_publish_in_background`:
   `job_store.create(..., runner=job_runner.mode())`, затем `submit`; замыкания `_runner` удаляются; сигнатура
   `start_report_job`, проверки лимитов и `LimitError` до создания задачи — без изменений.
4. **Тесты `tests/test_perf_tasks.py`:** в режиме thread задача выполняется через подменённый `threading.Thread`
   (как в smoke-тестах) и доводит запись до `done`/`error`; в режиме celery вызывается `apply_async` с очередью
   `reports`; `OperationalError` → `adopt_thread` и поток; heartbeat останавливается после завершения;
   `SoftTimeLimitExceeded` даёт `error`; `start_report_job` по-прежнему бросает `LimitError` до создания задачи.
   `tests/test_smoke_api.py` и `tests/test_load_tests*.py` проходят без правок.

### 11.5 Агент B5 — pipeline-worker

1. **Локальный пул вместо подзадач.** В `update_page.py` скачивание картинок Grafana, логов Loki и загрузка
   вложений идут через `concurrent.futures.ThreadPoolExecutor` (размер из `CONFIG["llm"]["max_domain_workers"]`
   или 4) с тем же поведением, что у текущих задач: до 3 попыток с паузой 2 с, таймаут `_TASK_TIMEOUT` на элемент,
   ошибка логируется и элемент пропускается, порядок подстановок сохраняется. Определения `@celery_app.task`,
   `_await_task` и импорт `loadlens_app.celery_app` из `update_page.py` удаляются (цикл импорта исчезает: убедиться,
   что `import update_page` первым в чистом интерпретаторе работает, и обновить `tests/test_smoke_api.py` только
   если он опирался на этот импорт — по согласованию в отчёте).
2. **Аудит пути отчёта на состояние процесса.** Пройти `update_page.update_report`, `confluence_export.publish_report`,
   `forecast_confluence.publish_forecast`: обращения к `job_store` только через колбэк прогресса; никаких
   `flask.request`/`g`/`session`; временные файлы в `data_collectors/temporary_files` (общий том); логирование
   работает без Flask. Найденное — исправить или описать.
3. **Тесты `tests/test_perf_pipeline_worker.py`:** пул выполняет элементы параллельно, повторяет неудачные до
   трёх раз, пропускает окончательно неудачные без исключения, не превышает таймаут; подстановки и порядок
   вложений не изменились (фейковые функции скачивания); `update_page` не импортирует `loadlens_app.celery_app`;
   `python -c "import update_page"` в подпроцессе завершается без ошибки.

### 11.6 Агент B6 — e2e-docs

Стартует последним, на слитых ветках B1–B5.

1. **Сквозная проверка** в песочнице 6 с Redis, воркером (`--pool=solo`) и заглушками B1; скрипты в
   `tools/e2e_jobs/` (Playwright + requests):
   1. отчёт через UI «Новый отчёт» в режиме `celery`: прогресс виден, задача `done`, отчёт открывается;
   2. публикация в Confluence: заглушка получила страницу, задача `done`;
   3. публикация прогноза;
   4. воркер убит посреди задачи: в пределах `LOADLENS_JOB_STALE_MIN` (для теста выставить 1) задача становится
      `error` с сообщением «воркер не отвечает», список задач не опрашивается вечно;
   5. Redis остановлен: новая задача выполняется потоком, в логе предупреждение, результат `done`;
   6. веб-сервер перезапущен во время celery-задачи: задача не помечена прерванной и доходит до `done`;
   7. бенч B1 `--during-job` в режимах `thread` и `celery`; таблица в «Статус» и в `docs/perf-baseline.md`.
2. **Документация.** `docs/jobs-runner.md`: режимы, переменные окружения, что происходит при падении воркера и
   брокера, как смотреть очередь (`celery -A loadlens_app.celery_app inspect active`), разработка на Windows.
   `README.md`: абзацы про Redis/Celery (воркер теперь строит отчёты; `CELERY_TASK_ALWAYS_EAGER=1` для разработки
   означает режим `thread`), `LOADLENS_JOB_RUNNER` в списке переменных. `docker/DOCKER_SETUP.md` §4.
3. **Тест `tests/test_perf_ops_docs.py`:** каждая переменная `LOADLENS_JOB_*`/`LOADLENS_REPORT_*`, упомянутая в
   документации, читается в коде, и наоборот.

### 11.7 Приёмка волны B

1. Все ветки слиты в порядке из 11.0, полный прогон тестов зелёный.
2. Сценарии 1–6 из 11.6 проходят; в compose воркер проходит healthcheck.
3. Бенч `--during-job`: в режиме `celery` p95 `/dashboard_data`, `/runs`, `/jobs`, `/load_tests/active` во время
   построения отчёта не больше 1.3× от idle; в режиме `thread` зафиксирована деградация (для сравнения).
4. Длительность самой задачи в воркере не хуже, чем потоком, более чем на 10 %.
5. `/jobs`, `/job_status`, `/create_report`, `/publish_confluence`, `/forecast/<id>/confluence` отвечают прежним JSON.
6. Контейнеры `ll-perf-db`, `ll-perf-redis` удалены, worktree'и убраны.

### 11.8 Риски и решения волны B

| Риск | Решение |
|---|---|
| Прогресс из воркера не доходит до UI | `JobStore.update` читает строку из таблицы, когда её нет в памяти; тест B3 |
| Веб-рестарт помечает celery-задачи прерванными | `_mark_interrupted` только для `runner='thread'`; сценарий 6 |
| Воркер умер, задача «выполняется» вечно | sweeper по `updated_at` плюс heartbeat раз в 30 с; сценарий 4 |
| Брокер недоступен, запрос `/create_report` висит на повторных попытках | политика отправки с одной попыткой и таймаут 2 с, откат на поток; сценарий 5 |
| Взаимная блокировка на вложенных подзадачах при concurrency 1 | подзадач больше нет, локальный пул потоков (B5) |
| Долгий вызов LLM без прогресса сочтён зависанием | heartbeat независим от прогресса |
| Дубликат отчёта после падения воркера | `task_acks_late=False`: задача не переигрывается, её помечает sweeper |
| Память воркера на больших окнах | `--max-tasks-per-child 20`, лимит 2g, `WORKER_CONCURRENCY` |
| Версии кода app и воркера разошлись | один образ, перезапуск обоих сервисов при деплое; написано в `docs/jobs-runner.md` |
| Локальная разработка без Redis | `auto` даёт `thread`, поведение как сегодня |

## 12. Шаблон задания агенту

```
Ты агент <N> «<имя>» из docs/perf-plan.md (волна <A|B>). Прочитай план целиком, затем выполни свой раздел.
Соблюдай раздел 0 (секреты, песочница, тесты, язык), общие решения своей волны (для B — раздел 11.0) и владение
файлами из раздела 2; чужие файлы не правь. Работай в worktree ../ll-perf-<имя> на ветке perf/<имя>, коммить
небольшими шагами, ничего не пуши.
Сдай: код, тесты в tests/test_perf_<имя>.py, полный зелёный прогон pytest, числа «до/после» по бенчу агента 1,
строку в таблице «Статус» плана и отчёт: что сделано, что отклонилось от плана и почему.
```
