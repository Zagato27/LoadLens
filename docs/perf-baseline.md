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

## Этап 3: запросы во время построения отчёта (волна B)

Замер 2026-10-16, агент B1, ветка `perf/b1-bench` от `perf/integration` (`6c5a282`, волна A влита). Это базовая
линия режима `thread` для агента B6: режима `celery` в коде ещё нет (его делают B2–B5), B6 снимет его тем же
инструментом и сравнит с таблицами ниже.

### Кратко

- Отчёт строится потоком внутри веб-процесса 43–49 с: сбор метрик и подготовка контекста 12–17 с (pandas,
  запись ~24 тыс. строк в `metrics`), анализ ИИ по доменам 24–25 с (ожидание модели), итог и сохранение 6–7 с.
- Пока идёт **сбор и подготовка**, все остальные запросы медленнее в 1.7–3.7 раза по p50 и до 12 раз по p95:
  `/dashboard_data` 6.5 → 22.4 мс (p95 59), `/runs` 17.5 → 30.2 (p95 76), `app.css` 7.6 → 28.4 (p95 219). Во
  время ожидания модели всё на уровне idle (p95 16–33 мс).
- По всей задаче p50 почти не меняется (0.97–1.37×), p95 растёт в 1.3–8 раз (c=1). Задача на 60 % состоит из
  ожидания LLM, поэтому общий p95 «размывает» вред: смотреть надо и таблицу по фазам.
- Время внутри приложения (`Server-Timing`, «app») во время задачи почти не растёт (в пределах 15 %), хотя
  клиентское растёт в разы: запрос ждёт GIL до и после обработчика (приём соединения, разбор, запись ответа),
  а не в нём. Сравнивать по клиентскому времени, не по app p50.
- С `LOADLENS_JOB_RUNNER=celery`, Redis и запущенным воркером сейчас всё то же самое: воркер не получил ни одной
  задачи, отчёт строит поток веб-процесса (проверено, таблица ниже). Переменную пока никто не читает.

### Заглушки (`tools/stubs/`)

Только стандартная библиотека, слушают только 127.0.0.1, запускаются из worktree или из копии песочницы
(`%TEMP%\ll_ui_s<N>\app\tools\stubs\`), каждая в своём окне или в фоне; остановка — Ctrl+C или по PID из
`netstat -ano | grep :509`. Подробности — в шапке каждого файла.

| Заглушка | Порт | Команда | Что отвечает | Служебное |
|---|---|---|---|---|
| `prom_stub.py` | 5092 | `C:/Python312/python.exe tools/stubs/prom_stub.py` | `/api/v1/query_range` и `/api/v1/query` на любой PromQL: `--series` рядов (6) с метками из `by (...)`/`label_replace` и общим набором (`application`, `instance`, `pool`, `node`, `pod`, …), значения — `--steps` ступеней нагрузки (4) с шумом 3 %, детерминированно по `--seed`; больше 11 000 точек — 400, как у Prometheus; `/api/v1/status/buildinfo`, `/api/v1/labels`, `/api/v1/label/<имя>/values` | `GET /_stub/stats`, `POST /_stub/reset`; `--delay-ms`, `--log` |
| `llm_stub.py` | 5090 | `C:/Python312/python.exe tools/stubs/llm_stub.py --delay 2` | OpenAI-совместимый `/v1/chat/completions`: анализ домена — валидный `LLMAnalysis` своего домена (находки без чисел, поэтому верификационного прохода нет), итог и проверка — общий ответ с `peak_performance`, судья — оценки всех кандидатов, импорт контекста — черновик «Checkout» (как прежняя заглушка `ll_ctx_e2e`), проверка подключения — `OK` | `GET /_stub/stats` (вызовы по видам и доменам, `max_in_flight`), `POST /_stub/reset`; `--delay`, `--jitter`, `--fail-every N` (каждый N-й — 500), `--log` |
| `confluence_stub.py` | 5091 | `C:/Python312/python.exe tools/stubs/confluence_stub.py` | basic auth `reader`/`secret`; страница 42 и короткая ссылка `/x/Ab42` (импорт контекста); поиск, создание, обновление с проверкой версии (409), удаление, вложения (multipart), `/rest/api/space/<key>` — всё, что зовут `confluence_export.py`, `confluence_page.py` и проверка подключения; страницы в памяти | `GET /_stub/pages`, `GET /_stub/pages/<id>`, `GET /_stub/stats`, `POST /_stub/reset`; `--delay-ms`, `--log` |

Журналы запросов по умолчанию выключены (`--log` пишет JSON-строки куда скажете; в каталог репозитория не писать).

### Рецепт: песочница, в которой настоящий отчёт строится до конца

`N` — номер песочницы (B1 — 1: порт 5071, база `ll_s1`, Redis-база 1). Команды из корня worktree.

1. Песочница: `%TEMP%\ll_ui\make_sandbox.py` скопировать (у B1 — `%TEMP%\ll_perf\make_sandbox_b1.py`), `REPO` —
   свой worktree, запустить с `N`. Песочница получает `settings.py` из примера (все адреса — 127.0.0.1,
   `default_params` = шаг `15s`, ресемплинг `1T`) и `run_app.py`.
2. Данные, один раз на базу (~45 с), затем миграции из каталога песочницы (заполняют `run_index`, `max_rps`):

   ```
   C:/Python312/python.exe tools/seed_bench_data.py --sandbox N --reset
   cd %TEMP%\ll_ui_s<N>\app && C:/Python312/python.exe -m loadlens_app.migrations
   ```

3. После **каждого** `make_sandbox` (он затирает runtime-JSON) — именно в таком порядке:

   ```
   C:/Python312/python.exe tools/seed_bench_data.py --sandbox N --runtime-only
   C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --setup-stubs
   ```

   Сид пересоздаёт область `bench` (сервисы `svc01..svc10`, пустые `metrics`/`logs` в
   `metrics_config_runtime.json`) и стирает всё, что было в ней раньше; `--setup-stubs` дописывает ключи заглушек.
4. Заглушки (три фоновых процесса, см. таблицу выше) и сервер: `C:/Python312/python.exe %TEMP%\ll_ui_s<N>\run_app.py`.
   Сервер перезапускать после `--setup-stubs`: глобальные ключи runtime-JSON вливаются в `CONFIG` при импорте.
5. Проверка: `C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --during-job --rounds 3` —
   код выхода 0, в сводке «done», «Отчёт … → 200, `/llm_reports` → 200, доменов 7».

Что пишет `--setup-stubs` в `settings_runtime.json` песочницы (идемпотентно, остальное не трогает; рабочее дерево
с `.git` отказывается менять):

```json
{
  "llm": {"provider": "openai", "openai": {"api_base_url": "http://127.0.0.1:5090/v1", "api_key": "stub",
          "model": "stub-model", "verify": false, "proxies": {"http": "", "https": ""}, "request_timeout_sec": 300}},
  "confluence": {"enabled": true, "base_url": "http://127.0.0.1:5091", "username": "reader", "password": "secret",
                 "space_key": "PERF", "parent_page_id": "42", "forecast_parent_page_id": "42",
                 "verify_ssl": false, "app_url": "http://127.0.0.1:507N"},
  "data_sources": {"prometheus": {"title": "Prometheus (заглушка)", "type": "prometheus",
                                  "prometheus": {"url": "http://127.0.0.1:5092"}}},
  "limits": {"llm_reports_per_user_per_day": null},
  "per_area": {"bench": {
    "domain_sources": {"default": {"source": "prometheus"}},
    "queries": {"lt_framework": {"influxql_queries": [], "promql_queries": ["…11 запросов…"],
                                 "label_keys_list": [["group", "name"], "…"]}}}}
}
```

Почему так:

- `data_sources` — каталог глобальный, а привязки области заменяют глобальные целиком
  (`update_page._effective_config_for_scope`), поэтому `domain_sources` пишется в область `bench`: все домены
  (`jvm`, `database`, `kafka`, `microservices`, `hard_resources`, `lt_framework`) читают заглушку Prometheus.
- `lt_framework` в примере — k6 из InfluxDB через Grafana (InfluxQL). Источник типа `prometheus` принимает только
  PromQL, а смесь языков `domain_query_language` отвергает и валит отчёт, поэтому в области InfluxQL-список
  пустой, а на месте каждого из 11 запросов — PromQL с теми же подписями (роли и ссылки SLA продолжают работать).
- `limits.llm_reports_per_user_per_day: null` — при выключенном входе все запросы идут от пользователя `local`,
  бюджет 20 отчётов в день кончился бы за вечер замеров.
- Данные окна: заглушка отвечает на любое окно, поэтому отчёт берёт последние `--window-min` минут (по умолчанию
  60) до текущей минуты. Сид даёт остальное: область и сервисы, архив прогонов для страниц и baseline —
  предыдущий успешный `step`-прогон области `bench` из `llm_reports`, его статистику отчёт читает из `metrics`.

Что делает один отчёт (`web_only`, `use_llm`, `svc01`, `step`, 60 минут): 69 запросов `query_range`
(~100 тыс. точек), 28 вызовов LLM (6 доменов + итог, по 3 кандидата и судье; при `max_concurrent` 2 из примера
и `--delay 2` фаза ИИ длится ~24 с), ~24 тыс. строк в `metrics`, 7 доменов в `llm_reports`, отчёт открывается по
`/reports/<run_id>`. Итоговый вердикт «Недостаточно данных» — у области нет порогов SLA, это ожидаемо.
Из UI то же самое: «Новый отчёт», область `bench`, сервис `svc01`, любое окно до 168 часов.

Публикация проверена тем же путём: `POST /publish_confluence {"run_name": …, "service": "svc01"}` для оставленного
отчёта (`--keep-report`) — задача `done` за ~40 с, в заглушке страница 1001 под родителем 42 (версия 2) и 69 PNG
(matplotlib). Это вторая тяжёлая задача для сценария 2 агента B6; заглушка хранит страницы в памяти, между
прогонами — `POST /_stub/reset` или перезапуск (иначе публикация того же отчёта обновит страницу, а не создаст).

### Бенч `--during-job`

```
C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --during-job --label thread --json thread-c1.json
C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --during-job --concurrency 4 --rounds 10 --label thread --json thread-c4.json
C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --during-job --label celery --compare thread-c1.json
C:/Python312/python.exe -X utf8 tools/bench_endpoints.py --sandbox N --during-job --label celery --compare docs/perf-baseline.md --compare-table 1
```

1. Idle-раунд: `--concurrency` клиентов `--rounds` раз обходят 8 эндпоинтов (`/`, `/dashboard_data`, два `/runs`,
   `/load_tests/active`, `/jobs`, `/services`, `app.css`; `--only`/`--endpoint` как обычно) с паузой `--pause-ms`
   после каждого запроса (по умолчанию 50 мс × клиентов: темп запросов одинаков при c=1 и c=4).
2. `POST /create_report` (CSRF-токен и cookie сессии берутся со страницы `/new`; `web_only`, `use_llm`, сервис
   `--service` области `--area`, окно `--window-min`, имя `bench-job-<время>`), и те же клиенты продолжают, пока
   опрос `/job_status/<id>` (раз в `--poll-sec`, отдельная строка таблицы) не покажет `done`/`error`.
3. После задачи: отчёт открывается (`/reports/<run_id>` и `/llm_reports`), прогон удаляется (`DELETE /runs/<имя>`;
   `--keep-report` оставляет), второй idle-раунд (`--no-idle-after` пропускает). Idle-колонки — оба раунда вместе.
4. Печать: таблица idle / during-job (p50, p95, `job/idle` — больше 1 значит медленнее во время задачи, app p50,
   число замеров, ошибки), сводка (длительность по опросу и по `created_at`/`updated_at` задачи, фазы по меткам
   прогресса 47 % и 83 %, вызовы заглушек LLM и Prometheus, проверка отчёта, сдвиг idle p50 после/до), таблица
   p50/p95 по фазам. `--json` сохраняет всё, включая хронологию прогресса; `--compare` берёт строки прошлого
   `--json` или K-ю таблицу markdown-файла (`--compare-table K`) и добавляет «было job p50/p95» и «было/стало».
5. Код выхода 1, если задача не дошла до `done` или отчёт не открылся. Перцентили — только по запросам с ответом;
   «без ответа» (статус 0) — ошибка, при доле больше 1 % сводка говорит, что числа не годятся.

Какой раннер построил отчёт — настройка сервера, не бенча: передавать в `--label`.

Режим `celery` для B6 (проверено с текущим кодом: воркер стартует и подключается к Redis): в каталоге песочницы
копия `run_app.py` → `run_app_celery.py` с переменными до `from app import app`:

```python
os.environ["CELERY_BROKER_URL"] = "redis://127.0.0.1:16379/<N>"
os.environ["CELERY_RESULT_BACKEND"] = "redis://127.0.0.1:16379/<N>"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "0"
os.environ["LOADLENS_JOB_RUNNER"] = "celery"
```

и воркер из `%TEMP%\ll_ui_s<N>\app` с теми же переменными плюс `LOADLENS_AUTH_ENABLED=0`, `LOADLENS_SECRET_KEY`:
`C:/Python312/python.exe -m celery -A loadlens_app.celery_app worker --loglevel=info --pool=solo -Q reports,celery`.
Что задача ушла в воркер, видно по `Received task` в его журнале (до B4 их 0).

### Базовая линия: режим `thread`

Параметры: сид по умолчанию (300 прогонов), заглушки по умолчанию (`prom_stub` 6 рядов × 4 ступени,
`llm_stub --delay 2`), окно 60 минут, `svc01`, `step`, сервер Werkzeug `threaded=True` из `run_app.py`, Windows 11,
12 логических CPU; на машине параллельно работали другие агенты, числа ориентировочные. Миллисекунды.

c=1 (`--rounds 20`, пауза 50 мс; idle — 40 замеров на эндпоинт, до и после задачи):

| Эндпоинт | idle p50 | idle p95 | job p50 | job p95 | job/idle p50 | job/idle p95 | app idle p50 | app job p50 | Замеров в job | Ошибки |
|---|---|---|---|---|---|---|---|---|---|---|
| `/` | 6.3 | 18.9 | 8.0 | 45.3 | 1.28× | 2.40× | 1.1 | 1.1 | 81 | 0 |
| `/dashboard_data` | 6.5 | 32.2 | 6.3 | 41.7 | 0.97× | 1.29× | 0.4 | 0.4 | 81 | 0 |
| `/runs?limit=20` | 17.5 | 30.7 | 19.1 | 56.1 | 1.09× | 1.83× | 11.8 | 11.8 | 80 | 0 |
| `/runs?limit=20&sort=max_rps` | 18.4 | 31.6 | 20.2 | 55.9 | 1.10× | 1.77× | 11.8 | 12.1 | 80 | 0 |
| `/load_tests/active?limit=5&live=1` | 6.4 | 22.7 | 8.8 | 55.0 | 1.37× | 2.42× | 0.5 | 0.5 | 80 | 0 |
| `/jobs?status=running&limit=10` | 6.7 | 18.6 | 7.4 | 48.4 | 1.09× | 2.60× | 0.2 | 0.2 | 80 | 0 |
| `/services` | 5.9 | 17.3 | 7.5 | 50.3 | 1.28× | 2.90× | 0.8 | 0.8 | 80 | 0 |
| `/static/css/app.css` | 7.6 | 18.2 | 7.4 | 144.5 | 0.97× | 7.96× | 1.0 | 1.0 | 80 | 0 |
| `/job_status/<id>` | — | — | 6.4 | 42.5 | — | — | — | 0.2 | 49 | 0 |

- Задача: done, 48.7 с по опросу `/job_status`, 48.0 с по `created_at`/`updated_at`
- Фазы: сбор метрик и подготовка 17.4 с · анализ ИИ по доменам 24.2 с · итоговый анализ и сохранение 7.1 с
- LLM-заглушка: 28 вызовов (analysis 21, judge 7), задержка 2.0 с на ответ, одновременно до 2
- Prometheus-заглушка: 69 запросов `query_range`, 99 774 точек; idle p50 после / до = 1.15×

По фазам задачи, c=1 (p50 / p95, в скобках число замеров):

| Эндпоинт | сбор метрик и подготовка | анализ ИИ по доменам | итоговый анализ и сохранение |
|---|---|---|---|
| `/` | 14.7 / 56.1 (25) | 6.5 / 19.0 (43) | 6.7 / 17.5 (13) |
| `/dashboard_data` | 22.4 / 59.3 (24) | 5.8 / 32.6 (44) | 5.8 / 25.2 (13) |
| `/runs?limit=20` | 30.2 / 76.2 (24) | 16.5 / 30.0 (44) | 18.3 / 27.6 (12) |
| `/runs?limit=20&sort=max_rps` | 32.2 / 75.1 (24) | 17.0 / 28.1 (44) | 17.3 / 25.6 (12) |
| `/load_tests/active?limit=5&live=1` | 15.9 / 60.4 (24) | 7.3 / 18.2 (44) | 6.9 / 69.7 (12) |
| `/jobs?status=running&limit=10` | 17.3 / 69.3 (24) | 5.9 / 17.7 (44) | 6.6 / 31.0 (12) |
| `/services` | 14.5 / 58.1 (24) | 6.8 / 18.9 (43) | 6.6 / 42.2 (13) |
| `/static/css/app.css` | 28.4 / 218.6 (24) | 6.5 / 18.8 (43) | 6.3 / 18.4 (13) |
| `/job_status/<id>` | 11.0 / 56.7 (18) | 5.7 / 16.5 (24) | 4.5 / 72.3 (7) |

c=4 (`--rounds 10`, пауза 200 мс на клиента; idle — 80 замеров на эндпоинт):

| Эндпоинт | idle p50 | idle p95 | job p50 | job p95 | job/idle p50 | job/idle p95 | app idle p50 | app job p50 | Замеров в job | Ошибки |
|---|---|---|---|---|---|---|---|---|---|---|
| `/` | 6.3 | 19.0 | 7.0 | 20.9 | 1.11× | 1.10× | 0.6 | 0.7 | 97 | 0 |
| `/dashboard_data` | 6.7 | 20.7 | 6.4 | 28.3 | 0.96× | 1.37× | 0.3 | 0.3 | 98 | 0 |
| `/runs?limit=20` | 17.4 | 32.2 | 20.2 | 43.0 | 1.16× | 1.33× | 10.2 | 10.0 | 99 | 0 |
| `/runs?limit=20&sort=max_rps` | 18.0 | 38.8 | 20.3 | 48.9 | 1.13× | 1.26× | 10.4 | 10.8 | 98 | 0 |
| `/load_tests/active?limit=5&live=1` | 7.7 | 18.7 | 7.4 | 30.5 | 0.96× | 1.63× | 0.3 | 0.4 | 98 | 0 |
| `/jobs?status=running&limit=10` | 7.4 | 17.6 | 8.6 | 25.1 | 1.16× | 1.43× | 0.1 | 0.1 | 96 | 0 |
| `/services` | 7.2 | 18.0 | 6.9 | 36.2 | 0.96× | 2.01× | 0.6 | 0.5 | 96 | 0 |
| `/static/css/app.css` | 7.1 | 20.3 | 7.7 | 79.9 | 1.09× | 3.93× | 0.7 | 0.7 | 96 | 0 |
| `/job_status/<id>` | — | — | 5.8 | 27.1 | — | — | — | 0.1 | 44 | 0 |

- Задача: done, 43.5 с (42.9 с по задаче); фазы 12.2 · 25.2 · 6.1 с; LLM 28 вызовов; idle p50 после / до = 1.25×
- В фазе сбора p50 / p95: `/dashboard_data` 10.1 / 29.5, `/runs` 29.5 / 45.2, `/services` 16.3 / 50.4,
  `app.css` 11.6 / 116.8; в фазе ИИ — 5.6–15.1 / 17.7–36.4

`LOADLENS_JOB_RUNNER=celery`, Redis-база 1 и воркер `--pool=solo` запущены, c=1 — до B4 это тот же поток
(воркер получил 0 задач):

| Эндпоинт | idle p50 | idle p95 | job p50 | job p95 | job/idle p50 | job/idle p95 | app idle p50 | app job p50 | Замеров в job | Ошибки |
|---|---|---|---|---|---|---|---|---|---|---|
| `/` | 5.9 | 18.2 | 6.2 | 27.2 | 1.06× | 1.50× | 0.7 | 0.8 | 77 | 0 |
| `/dashboard_data` | 5.0 | 36.4 | 6.1 | 36.7 | 1.21× | 1.01× | 0.3 | 0.3 | 77 | 0 |
| `/runs?limit=20` | 15.5 | 27.6 | 16.8 | 58.3 | 1.09× | 2.11× | 9.6 | 11.1 | 76 | 0 |
| `/runs?limit=20&sort=max_rps` | 15.2 | 29.6 | 19.0 | 45.9 | 1.25× | 1.55× | 9.6 | 10.8 | 76 | 0 |
| `/load_tests/active?limit=5&live=1` | 6.7 | 18.0 | 5.8 | 29.2 | 0.87× | 1.62× | 0.3 | 0.4 | 76 | 0 |
| `/jobs?status=running&limit=10` | 6.1 | 20.6 | 8.1 | 25.2 | 1.34× | 1.22× | 0.1 | 0.1 | 76 | 0 |
| `/services` | 5.7 | 17.8 | 6.2 | 50.2 | 1.08× | 2.82× | 0.5 | 0.5 | 76 | 0 |
| `/static/css/app.css` | 5.9 | 18.1 | 6.8 | 95.5 | 1.14× | 5.27× | 0.6 | 0.7 | 76 | 0 |
| `/job_status/<id>` | — | — | 5.5 | 31.6 | — | — | — | 0.1 | 45 | 0 |

- Задача: done, 44.5 с (43.9 с по задаче); фазы 13.3 · 24.2 · 7.1 с; idle p50 после / до = 1.08×
- В фазе сбора p50 / p95: `/` 20.8 / 45.0, `/runs` 34.1 / 76.9, `/load_tests/active` 16.8 / 92.4, `app.css` 19.9 / 163.3

Всего в тот вечер пять прогонов c=1 (два первых — до того, как бенч получил второй idle-раунд): длительность
задачи 44.5–58.2 с (фаза сбора 13–27 с — дольше, когда параллельно шёл чужой бенч), `job/idle p95` у `app.css`
4.5–12.4×, у JSON-эндпоинтов от 1.0 до 5.8×. Ориентир для B6: в режиме `celery`
фаза «сбор метрик и подготовка» должна выглядеть как фаза «анализ ИИ» здесь (p50 ≈ idle, p95 ≤ 1.3× idle),
а общий `job/idle p95` (критерий 11.7) — не больше 1.3.

### Подводные камни

- **Временные порты.** Werkzeug отвечает `Connection: close` на каждый запрос (keep-alive у dev-сервера
  выключен), на этой машине ~13.3 тыс. соединений держит Outlook, свободных временных портов остаётся ~3 тыс., а
  TIME_WAIT живёт 2 минуты. Прогон c=1 — около 1.2 тыс. соединений, c=4 с `--rounds 10` — около 1.6 тыс.
  c=4 с паузой 50 мс исчерпал порты: 620 запросов
  без ответа на эндпоинт, вызовы LLM из отчёта падали с `ConnectTimeout`, задача шла 123 с вместо 45. Поэтому
  пауза по умолчанию 50 мс × клиентов, для c=4 — `--rounds 10`, между прогонами — 1–2 минуты и
  `netstat -ano | grep -c TIME_WAIT`. Сводка предупреждает, если без ответа больше 1 % запросов.
- **Часы.** `time.monotonic()` на Windows тикает раз в 15.6 мс: последний опрос задачи и её конец попадали в
  один тик, и опрос выпадал из окна. Бенч использует `perf_counter`.
- **Шум.** Idle до и после задачи объединяются, их отношение p50 печатается: вне 0.77–1.3 замер повторить.
  p95 на 40–80 замерах — это 2–4 самых медленных запроса, ×1.3 по p95 у быстрых эндпоинтов не сигнал.
- **Кэш ответов** волны A включён: `/dashboard_data` (10 с), `/jobs` и `/load_tests/active` (3 с) чаще отдаются
  из кэша — как у браузерных опросов. Кэш `/jobs` сбрасывает каждое обновление прогресса задачи.
- **Порядок настройки.** `make_sandbox` → `seed --runtime-only` → `--setup-stubs` → перезапуск сервера. Сид после
  `--setup-stubs` стирает привязки и PromQL области — отчёт упадёт на «принимает promql, в запросах есть influxql».
- **Журнал сервера** забит `FutureWarning: 'T' is deprecated` от `resample("1T")` из настроек песочницы —
  безвредно.
- **Прогоны отчёта** называются `bench-job-<время>` и удаляются после проверки; если бенч прерван — их уберёт
  `DELETE /runs/<имя>` или `seed --reset` (`LIKE 'bench-%'`).
- **Порты заглушек общие для машины.** 5090–5092 — одна копия на все песочницы. Вторую копию запускать с
  `--port`, а песочницу направлять на неё через `--setup-stubs --llm-stub http://127.0.0.1:<p> --confluence-stub …
  --prom-stub …` (те же ключи бенч берёт для подсчёта вызовов). `--during-job` в начале сбрасывает счётчики
  заглушек: два бенча на одних заглушках одновременно путают число вызовов (сам отчёт от этого не страдает).
- **Параметры заглушек** определяют длительность фаз: `llm_stub --delay` — фазу ИИ (28 вызовов по 2 в параллель),
  `prom_stub --series` — объём данных фазы сбора. Для сравнения режимов они должны совпадать; `self_consistency_k`
  и `max_concurrent` берутся из `settings.example.py` (3 и 2).
