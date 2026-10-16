# Базовые замеры производительности LoadLens (этап 0)

Замер 2026-10-16 на ветке `perf/bench` (от `perf/integration`, коммит `c497eb4`), до работ волны A
(`docs/perf-plan.md`). Все числа — миллисекунды, если не сказано иное.

## Кратко

- `/runs` — самый медленный эндпоинт: 0.55 с на 300 прогонах и 1.4–2.0 с на 1000. Два почти одинаковых
  запроса (счётчик и страница) каждый раз агрегируют всю гипертаблицу `metrics` (`GROUP BY run_name`,
  19 тыс. буферов ≈ 150 МБ на 300 прогонах, 64 тыс. ≈ 500 МБ на 1000). Счётчик идёт без параллельных
  воркеров и в 1.3–1.6 раза медленнее страницы. На 1000 прогонах при 4 параллельных клиентах p95 — 5.3 с,
  с `q=` — 19 с.
- `/dashboard_data` — 140 мс (230 мс на 1000 прогонов). Восемь запросов, из них дорогие: «последний прогон»
  и «состояние по сервисам» разворачивают из TOAST `parsed` каждой финальной строки ради `max_rps`
  (3–10 тыс. буферов), `COUNT(DISTINCT run_name)` по `metrics` дешёв только благодаря индексу
  `idx_metrics_run_name` и SkipScan TimescaleDB (без индекса — 850 мс).
- Новое соединение на каждый запрос стоит дважды: 16–23 мс на `psycopg2.connect` и ещё ~40 мс на
  планирование первого запроса к гипертаблице в новой сессии (каталог 90 чанков не прогрет; повторный
  запрос в той же сессии планируется за 4 мс). Поэтому `/jobs` и `/load_tests/active`, у которых SQL
  выполняется за 0.1 мс, отвечают за 30–35 мс. Это задача агента 2 (пул).
- `/services` (13–14 мс без SQL) — чтение runtime-конфигов: пять разборов `settings_runtime.json` (238 КБ)
  и два `_active_metrics_config` с `deepcopy` на запрос. При 4 клиентах пропускная способность не растёт
  (52 → 50 запросов/с): всё сериализовано на GIL и `_RUNTIME_FILES_LOCK`. Это задача агента 3.
- Статика и `/` (рендер шаблона) — 1–2 мс в приложении; остальное — накладные расходы dev-сервера и
  нового TCP-соединения на каждый запрос (5–8 мс на Windows).

## Инструменты

| Что | Где | Зачем |
|---|---|---|
| Заголовок `Server-Timing: app;dur=<мс>` | `loadlens_app/auth/web.py` (`_gate`, `_after_request`) | Время внутри приложения на каждом ответе, виден в DevTools браузера и в бенче |
| Поле `%(M)s` в access-логе gunicorn | `docker/Dockerfile`, `GUNICORN_CMD_ARGS` | Время ответа в мс последним полем строки лога в проде |
| Сид | `tools/seed_bench_data.py` | Синтетические прогоны в базе песочницы и область `bench` в её runtime-конфиге |
| Бенч | `tools/bench_endpoints.py` | p50/p95/max, пропускная способность, `Server-Timing`, размер; JSON и сравнение |
| EXPLAIN | `tools/bench_endpoints.py --explain` | `EXPLAIN (ANALYZE, BUFFERS)` каждого SELECT, который выполняет эндпоинт |

Формат access-лога (проверен разбором `GUNICORN_CMD_ARGS` gunicorn'ом из образа `docker-app` без сети;
на Windows gunicorn не запускается, окончательная проверка — при деплое):

```
10.0.0.1 - - [16/Oct/2026:13:45:52 +0000] "GET /runs HTTP/1.1" 200 1234 "-" "-" 87
```

Время в мс — последнее поле: `docker logs <app> 2>&1 | awk '$9 ~ /^[0-9]+$/ {print $NF, $7}' | sort -n | tail`.

## Как повторить в своей песочнице (около 5 минут)

`N` — номер агента (порт `5070+N`, база `ll_s<N>` в контейнере `ll-perf-db`). Команды из корня своего
worktree, Python — `C:/Python312/python.exe`.

1. Песочница. Скопировать `%TEMP%\ll_ui\make_sandbox.py` к себе (например, `%TEMP%\ll_perf_<N>\make_sandbox.py`),
   поменять `REPO` на путь своего worktree и запустить с аргументом `N`. Повторять после каждой правки кода:
   скрипт пересобирает `%TEMP%\ll_ui_s<N>\app` и затирает её runtime-JSON (база остаётся).
2. Сид (~45 с; `--runs 1000` ~2.5 мин). Применяет `docker/initdb/02_schema.sql` своей ветки, создаёт таблицы
   функциями приложения, пишет данные и область `bench` в runtime-JSON песочницы:

   ```
   C:/Python312/python.exe tools/seed_bench_data.py --sandbox N --reset
   ```

   После повторного `make_sandbox.py` достаточно вернуть область, база не трогается:

   ```
   C:/Python312/python.exe tools/seed_bench_data.py --sandbox N --runtime-only
   ```

3. Сервер — в фоне: `C:/Python312/python.exe %TEMP%\ll_ui_s<N>\run_app.py`. Остановка — по PID из
   `netstat -ano | grep :507N`, предварительно убедившись (`tasklist /FI "PID eq <pid>"`), что это `python.exe`.
4. Бенч «до» (до своих правок, на той же базе):

   ```
   C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --json before-c1.json
   C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --concurrency 4 --json before-c4.json
   ```

   c=1 идёт ~50 с, c=4 ~2 мин. Только часть эндпоинтов: `--only runs` (подстрока пути) или
   `--endpoint "/runs?limit=20"` (повторяемый).
5. Правки → шаг 1 → `--runtime-only` → перезапуск сервера → бенч «после» со сравнением:

   ```
   C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --compare before-c1.json --json after-c1.json
   C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --concurrency 4 --compare before-c4.json --json after-c4.json
   ```

   Колонки «было p50/p95» и «×p50/×p95» = было/стало (больше 1 — быстрее).
6. Планы запросов (сервер не нужен: приложение песочницы импортируется в процесс, каждый SELECT выполняется
   ещё раз под `EXPLAIN (ANALYZE, BUFFERS)`; рабочее дерево с `.git` скрипт откажется импортировать):

   ```
   C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --explain --only runs --plans
   ```

   Первая строка вывода — стоимость `connect` и круга `SELECT 1`. Первый запрос к `metrics` в новой сессии
   включает ~40–60 мс холодного планирования (см. «Куда уходит время»), поэтому порядок эндпоинтов влияет
   на `Planning` первого из них.

Подводные камни:

- Git Bash переписывает аргументы, начинающиеся с `/`, в пути Windows (`/runs` → `C:/Program Files/Git/runs`):
  для `--endpoint` ставить `MSYS_NO_PATHCONV=1` перед командой или пользоваться `--only`.
- Числа сравнивать только внутри одной машины и одного запуска контейнера. Все агенты делят один
  `ll-perf-db` и один процессор: чужой бенч или сид в это время искажает результат. «До» и «после»
  снимать подряд; при сомнениях повторить «до».
- Шум. Повторный замер c=1 через 20 минут: p50 тяжёлых эндпоинтов совпал в пределах ±5 %, p95 — до ±35 %;
  у быстрых (< 20 мс) p95 при 20 запросах — это второй по величине замер, он скачет в 2 раза. Для них
  смотреть `app p50` (`Server-Timing`) или поднять `--rounds 50`; ускорение меньше 1.3× по p95 — не сигнал.
- Dev-сервер Werkzeug создаёт поток на запрос, бенч открывает новое TCP-соединение на каждый запрос.
  Разница «p50 − app p50» (5–8 мс) — эти накладные расходы, к коду не относятся.
- Бенч возвращает код 1, если были ответы ≥ 400 или обрывы (колонка «Ошибки», в JSON — `statuses`).
- На базе без таблиц `/dashboard_data` и `/runs` отвечают 500 — сначала сид.

## Среда

- Windows 11, 12 логических CPU; Python 3.12; Flask dev-сервер (`threaded=True`) из `run_app.py`
  песочницы, вход выключен (`LOADLENS_AUTH_ENABLED=0`). В проде gunicorn gthread `--threads 8`: абсолютные
  числа другие, сравнивать относительные.
- `timescale/timescaledb:latest-pg16` в Docker Desktop: PostgreSQL 16.10, TimescaleDB 2.23.1. Настройки
  `timescaledb-tune` образа: `shared_buffers` 3.9 ГБ, `work_mem` 3.3 МБ, `max_parallel_workers_per_gather` 6,
  `random_page_cost` 1.1, JIT выключен. Все данные в кэше (`shared hit`), чтений с диска нет.
- Соединение без TLS (`sslmode=disable`) через проброс порта Docker Desktop: `connect` 16–23 мс, круг
  `SELECT 1` 0.7–0.9 мс.

## Данные сида

| Параметр | По умолчанию | Вариант |
|---|---|---|
| `--runs` / `--services` | 300 / 10 (`svc01..svc10`, по кругу) | 1000 / 10 |
| `--minutes` × рядов | 60 × (20 `lt` + 16 `jvm`) | так же |
| `--days` (разброс `created_at`) | 90 | так же |
| Строк `public.metrics` | 648 000 (180 МБ, ~90 чанков по 1 дню) | 2 160 000 (581 МБ) |
| Строк `public.llm_reports` | 1 328 (300 прогонов: `final` + `jvm`, `kafka`, `hard_resources`; у 32 прогонов повторный анализ) | 4 440 (17 МБ) |
| `settings_runtime.json` / `metrics_config_runtime.json` песочницы | 238 КБ / 54 КБ (область `bench` + инертные `bench-pad-*` до размеров боевых файлов) | так же |

`SELECT count(*)`: по умолчанию `metrics` 648 000, `llm_reports` 1 328; при `--runs 1000` — 2 160 000 и 4 440.

Индексы после сида (как в docker-развёртывании: сид применяет `02_schema.sql`, затем `_ensure_*` приложения):
`metrics_time_idx (time DESC)`, `idx_metrics_run_time (run_id, time)`, `idx_metrics_run_name (run_name, time)`;
`llm_reports_pkey (id)`, `idx_llm_reports_run_created (run_name, created_at DESC)`. Приложение само создаёт
только `idx_metrics_run_time`: база, созданная без `02_schema.sql`, не имеет `idx_metrics_run_name`
(последствия — в разделе EXPLAIN). Финальные строки: `max_rps` в поле своего типа теста (step/stress —
`peak_performance.max_rps`, soak — `stability_under_load.actual_rps`, spike — `spike_analysis.max_spike_rps`),
70 % числом, 15 % числом-строкой, 15 % строкой `~1200` (SQL-разбор её пропускает).

## Базовые числа

`--rounds 20 --warmup 2`, область `bench`. c=1 — один клиент; c=4 — четыре клиента одновременно, по 20
запросов каждый (80 замеров). `app` — p50 `Server-Timing`. req/s — запросов в секунду за весь замер.

### По умолчанию (300 прогонов)

| Эндпоинт | c=1 p50 | c=1 p95 | c=1 app | c=1 req/s | c=4 p50 | c=4 p95 | c=4 app | c=4 req/s | Размер |
|---|---|---|---|---|---|---|---|---|---|
| `/` | 10.5 | 25.3 | 0.6 | 87.4 | 14.8 | 29.6 | 0.6 | 218.8 | 15.4 КБ |
| `/dashboard_data` | 145.7 | 184.3 | 138.4 | 6.7 | 257.5 | 325.1 | 244.3 | 15.1 | 20.7 КБ |
| `/runs?limit=20` | 551.7 | 622.3 | 543.7 | 1.8 | 1220.5 | 1622.9 | 1211.5 | 3.2 | 9.6 КБ |
| `/runs?limit=20&q=svc0` | 767.2 | 875.9 | 759.2 | 1.3 | 1551.1 | 5192.0 | 1542.8 | 1.7 | 9.6 КБ |
| `/runs?limit=20&sort=max_rps` | 543.1 | 645.3 | 537.1 | 1.7 | 1256.9 | 1461.8 | 1240.9 | 3.1 | 9.7 КБ |
| `/load_tests/active?limit=5&live=1` | 41.7 | 53.5 | 35.0 | 23.6 | 64.2 | 87.2 | 54.8 | 58.2 | 22 Б |
| `/load_tests/active?limit=0` | 36.8 | 42.1 | 32.4 | 26.2 | 74.9 | 138.2 | 61.1 | 43.8 | 22 Б |
| `/jobs?status=running&limit=10` | 39.7 | 52.0 | 33.8 | 24.5 | 64.2 | 313.7 | 52.1 | 39.9 | 12 Б |
| `/services` | 18.1 | 22.1 | 13.8 | 52.4 | 73.3 | 128.2 | 64.7 | 49.7 | 755 Б |
| `/project_areas` | 6.1 | 27.5 | 2.2 | 99.5 | 18.9 | 33.8 | 10.2 | 191.7 | 218 Б |
| `/static/js/settings.js` | 9.5 | 11.9 | 1.2 | 111.4 | 22.5 | 39.5 | 2.7 | 164.7 | 249.0 КБ |
| `/static/css/app.css` | 10.2 | 13.0 | 1.6 | 94.4 | 21.6 | 26.4 | 2.7 | 178.8 | 108.4 КБ |

### `--runs 1000`

| Эндпоинт | c=1 p50 | c=1 p95 | c=1 app | c=1 req/s | c=4 p50 | c=4 p95 | c=4 app | c=4 req/s | Размер |
|---|---|---|---|---|---|---|---|---|---|
| `/` | 7.9 | 9.9 | 0.7 | 117.2 | 13.7 | 30.6 | 0.6 | 241.8 | 15.4 КБ |
| `/dashboard_data` | 230.6 | 265.4 | 223.6 | 4.3 | 390.2 | 445.9 | 379.3 | 10.0 | 20.8 КБ |
| `/runs?limit=20` | 1473.3 | 1817.2 | 1463.8 | 0.7 | 2731.7 | 5304.4 | 2725.6 | 1.1 | 9.6 КБ |
| `/runs?limit=20&q=svc0` | 1983.2 | 2169.9 | 1975.9 | 0.5 | 4074.9 | 18952.7 | 4065.2 | 0.6 | 9.6 КБ |
| `/runs?limit=20&sort=max_rps` | 1399.2 | 1771.2 | 1391.0 | 0.7 | 2824.5 | 5495.8 | 2803.8 | 1.1 | 9.7 КБ |
| `/load_tests/active?limit=5&live=1` | 42.0 | 53.1 | 35.5 | 22.9 | 79.8 | 102.7 | 67.8 | 49.7 | 22 Б |
| `/load_tests/active?limit=0` | 37.7 | 41.9 | 33.8 | 25.5 | 60.9 | 78.6 | 52.3 | 62.1 | 22 Б |
| `/jobs?status=running&limit=10` | 34.5 | 54.8 | 30.1 | 27.2 | 56.0 | 75.9 | 45.2 | 64.6 | 12 Б |
| `/services` | 18.0 | 38.4 | 13.6 | 46.7 | 68.6 | 127.1 | 59.8 | 51.9 | 755 Б |
| `/project_areas` | 7.5 | 21.5 | 3.2 | 106.7 | 17.3 | 34.1 | 10.5 | 219.1 | 218 Б |
| `/static/js/settings.js` | 8.7 | 23.4 | 1.0 | 97.6 | 17.2 | 23.8 | 2.1 | 215.5 | 249.0 КБ |
| `/static/css/app.css` | 7.3 | 23.9 | 2.0 | 99.0 | 16.9 | 20.3 | 2.3 | 216.4 | 108.4 КБ |

Повторный замер c=1 на данных по умолчанию (после пересева, 20 минут спустя): `/dashboard_data` 143.1 / 161.9,
`/runs?limit=20` 540.8 / 604.9, `/runs?…&q=svc0` 794.1 / 1174.6, `/runs?…&sort=max_rps` 542.1 / 624.6,
`/jobs` 37.1 / 57.6, `/services` 18.5 / 40.0 (p50 / p95).

## Куда уходит время

Разбор `--explain` и профилировщика на данных по умолчанию:

| Эндпоинт | app | Из чего складывается |
|---|---|---|
| `/runs?limit=20` | 540 | 2 запроса: счётчик (план 8 + выполнение 300–330) и страница (12 + 190–250), `connect` ~20, холодное планирование первого запроса к `metrics` в новой сессии (до 57 мс) |
| `/dashboard_data` | 138 | 8 запросов ≈ 45 мс выполнения и 46 мс холодного планирования `COUNT(DISTINCT)` (на прогретой сессии 4–7), `connect` ~20, чтение конфигов ~2 |
| `/jobs`, `/load_tests/active` | 30–35 | `connect` ~20 и первый запрос новой сессии (~10, из них SQL 0.1), конфиги |
| `/services` | 14 | без SQL: 5 × `_read_json_file` (238 КБ), 2 × `_active_metrics_config` (`deepcopy` + нормализация) |
| `/project_areas` | 2 | один `_per_area_config` |

Холодное планирование (новое соединение на запрос): один и тот же `COUNT(DISTINCT run_name)` в свежей
сессии — Planning 43–46 мс, второй и третий раз в той же сессии — 4 мс. Пул соединений (агент 2) убирает
и `connect`, и это.

## EXPLAIN (ANALYZE, BUFFERS)

Сняты `tools/bench_endpoints.py --explain --plans` (соединение переиспользуется между запросами, поэтому
холодное планирование есть только у первого запроса к `metrics`). Списки чанков сокращены.

### `/runs`: CTE `base` отдельно

```sql
SELECT run_name, MIN("time") AS start_time, MAX("time") AS end_time,
       COALESCE(MAX(service), '') AS service, COALESCE(MAX(NULLIF(run_id, '')), '') AS run_id
FROM public.metrics
WHERE run_name IS NOT NULL AND run_name <> '' AND service = ANY(ARRAY['svc01', …, 'svc10'])
GROUP BY run_name
```

300 прогонов:

```
Finalize HashAggregate  (rows=300) (actual time=156.970..162.750 rows=300 loops=1)
  Group Key: metrics.run_name
  Buffers: shared hit=19117
  ->  Gather  (actual time=9.716..161.913 rows=338 loops=1)
        Workers Planned: 6
        Workers Launched: 6
        ->  Parallel Append  (actual time=15.142..116.010 rows=48 loops=7)
              ->  Partial HashAggregate  (actual time=24.491..24.497 rows=10 loops=1)
                    Group Key: _hyper_1_94_chunk.run_name
                    ->  Parallel Seq Scan on _hyper_1_94_chunk  (rows=19620)
              … по одному Partial HashAggregate + Parallel Seq Scan на каждый из ~90 чанков
Planning Time: 5.857 ms
Execution Time: 168.350 ms
```

1000 прогонов: тот же план, `Buffers: shared hit=63571`, Planning 8.5 мс, Execution 567.5 мс. Время растёт
линейно с размером `metrics`: индекса, из которого можно взять min/max по прогону, нет, читается вся таблица.

### `/runs`: счётчик и страница

`list_runs` выполняет CTE `base`+`final`+`joined` дважды: `SELECT COUNT(*) FROM joined` и страницу с
`ORDER BY … OFFSET/LIMIT`.

| Запрос | 300 прогонов: plan / exec / buffers | 1000 прогонов: plan / exec / buffers |
|---|---|---|
| Счётчик | 7.7 / 329.7 / 19 380 | 10.3 / 936.7 / 64 448 |
| Страница (`sort=end_time`) | 12.0 / 246.7 / 22 167 | 11.3 / 595.9 / 73 842 |

Счётчик (1000 прогонов) — без параллельных воркеров, поэтому дороже страницы:

```
Aggregate  (actual time=925.153..925.259 rows=1 loops=1)
  Buffers: shared hit=64454
  ->  Hash Left Join  (rows=1000)
        Hash Cond: (metrics.run_name = f.run_name)
        ->  Finalize HashAggregate  (actual time=919.491..919.768 rows=1000 loops=1)
              Group Key: metrics.run_name
              ->  Append  (actual time=9.806..918.475 rows=1045 loops=1)
                    ->  Partial HashAggregate … ->  Seq Scan on _hyper_1_183_chunk  (rows=19512)
                    … ~90 чанков
        ->  Hash  (rows=1000)
              ->  Subquery Scan on f  (rows=1000)  Filter: (f.rn = 1)
                    ->  WindowAgg … ->  Seq Scan on llm_reports  (rows=1110, Rows Removed by Filter: 3330)
Planning Time: 63.403 ms   (холодная сессия)
Execution Time: 933.542 ms
```

Страница (1000 прогонов): `base` параллельно (6 воркеров, 470 мс), `final` считает `max_rps` регулярными
выражениями по `parsed` каждой финальной строки (`Subquery Scan on f` 113 мс, 10 271 буфер — разворот TOAST):

```
Limit  (actual time=585.862..587.385 rows=20 loops=1)
  Buffers: shared hit=73842
  ->  Sort  Sort Key: (max(metrics."time")) DESC, metrics.run_name  (top-N heapsort)
        ->  Hash Right Join  (actual time=478.523..586.164 rows=1000 loops=1)
              Hash Cond: (f.run_name = metrics.run_name)
              ->  Subquery Scan on f  (actual time=8.884..113.224 rows=1000 loops=1)
                    Buffers: shared hit=10271
                    ->  WindowAgg … ->  Seq Scan on llm_reports  (rows=1110)
              ->  Hash
                    ->  Finalize HashAggregate  (actual time=468.192..470.304 rows=1000 loops=1)
                          ->  Gather  Workers Launched: 6
                                ->  Parallel Append … Parallel Seq Scan на каждом чанке
Planning Time: 11.300 ms
Execution Time: 595.885 ms
```

На 300 прогонах в странице планировщик ошибается в оценке `final` (rows=2 вместо 300) и выбирает
`Nested Loop Left Join` с `Materialize` (`Rows Removed by Join Filter: 89700`); на 1000 — `Hash Right Join`.

### `/dashboard_data`: два самых тяжёлых запроса

| # | Запрос | 300: plan / exec / buffers | 1000: plan / exec / buffers |
|---|---|---|---|
| 1 | последний прогон (`ORDER BY created_at DESC LIMIT 1` с `DASHBOARD_MAX_RPS_SQL`) | 5.4 / 17.1 / 3 066 | 3.7 / 54.4 / 10 166 |
| 2 | счётчик вердиктов (`ROW_NUMBER` по прогонам) | 0.4 / 2.7 / 266 | 0.4 / 6.4 / 880 |
| 3 | `runs_total`: `COUNT(DISTINCT run_name)` по `metrics` | 45.8 (холодная сессия) / 8.8 / 1 164 | 51.2 (холодная) / 27.3 / 3 474 |
| 4 | состояние по сервисам (две оконные функции, `max_rps`) | 0.4 / 13.7 / 2 045 | 0.5 / 49.0 / 6 811 |
| 5–8 | недавние прогоны, активность по неделям, предыдущий прогон | ≤ 0.4 / ≤ 1.6 | ≤ 0.4 / ≤ 7.7 |

Запрос 1 (1000 прогонов): выражение `max_rps` и `sla_details` вычисляются для всех финальных строк до
top-N сортировки, `parsed` каждой строки разворачивается из TOAST:

```
Limit  (actual time=55.825..55.827 rows=1 loops=1)
  Buffers: shared hit=10138
  ->  Sort  Sort Key: created_at DESC  Sort Method: top-N heapsort
        ->  Seq Scan on llm_reports  (actual time=0.093..54.804 rows=1110 loops=1)
              Filter: ((domain = 'final'::text) AND (service = ANY ('{svc01,…,svc10}'::text[])))
              Rows Removed by Filter: 3330
              Buffers: shared hit=10138
Planning Time: 0.311 ms
Execution Time: 55.874 ms
```

Для сравнения: тот же фильтр без `parsed` (запрос 2) читает 880 буферов. Запрос 4 устроен так же (6 811 буферов).

Запрос 3 (1000 прогонов): SkipScan TimescaleDB по `idx_metrics_run_name` в каждом чанке:

```
Aggregate  (actual time=13.878..13.934 rows=1 loops=1)
  Buffers: shared hit=3474
  ->  Merge Append  Sort Key: metrics.run_name  (rows=1045)
        ->  Custom Scan (SkipScan) on _hyper_1_183_chunk  (rows=10)
              ->  Index Scan using _hyper_1_183_chunk_idx_metrics_run_name on _hyper_1_183_chunk
                    Index Cond: (run_name IS NOT NULL)
                    Filter: ((run_name <> ''::text) AND (service = ANY (…)))
        … ~90 чанков
Planning Time: 51.2 ms (холодная сессия; в прогретой 4–7 мс)
```

Без `idx_metrics_run_name` (его создаёт только `02_schema.sql`, не приложение; проверено `DROP INDEX` в
транзакции с откатом) тот же запрос на 300 прогонах выполняется 847.8 мс вместо 5.4 мс — `/dashboard_data`
на такой базе стоит почти секунду.

## Перепроверка после волны A

Те же команды на тех же параметрах сида (по умолчанию, затем `--runs 1000`); «после» — из `--compare`
с JSON «до», снятым заново на той же машине непосредственно перед замером «после».
