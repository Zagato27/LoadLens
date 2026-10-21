> English: [docs/deployment.md](../docs/deployment.md) (production deployment) and [docs/getting-started.md](../docs/getting-started.md) (demo and first steps).

## Развертывание в Docker (приложение + TimescaleDB)

Ниже — исчерпывающая инструкция: какие файлы добавлены, что в них писать, и как поднять всё в Docker на Windows (PowerShell) и Linux/macOS.

### Что добавлено

- `Dockerfile` — актуализирован: добавлены переменные окружения, `curl` для healthcheck и `HEALTHCHECK` (проверяет публичный `/healthz`: остальные страницы требуют входа).
- `docker-compose.yml` — поднимает сервисы: `app` (веб), `timescaledb` (TimescaleDB/PostgreSQL), `redis` (брокер) и `celery_worker` (фоновые задачи).
- `initdb/01_timescaledb.sql` — автоматическое включение расширения TimescaleDB в базе при первом старте.
- `initdb/03_app_role.sh` — непривилегированная роль приложения: владелец таблиц, без прав суперпользователя.
- `initdb/02_schema.sql` — таблицы `metrics` (гипертаблица), `llm_reports`, `engineer_reports`, `confluence_publications`, `report_jobs`, `load_test_runs`, `app_users`, `api_tokens`, `audit_log` и индексы; приложение создаёт их и само, скрипт лишь готовит схему заранее.
- Redis + Celery worker — через `docker-compose` теперь поднимаются брокер `redis` и отдельный контейнер `celery_worker`: он строит отчёты и публикует их в Confluence, веб-приложение только ставит задачи (§4).
- `env.example` (лежит в этой же папке) — пример файла окружения для Compose. Скопируйте его в `.env` и отредактируйте.
- `settings.example.py` — обезличенный пример `settings.py`. Скопируйте и заполните.
- `.gitignore` — добавлены правила для игнорирования действующих конфигов (`settings.py`, `settings_runtime.json`, `docker/.env` и прочих `.env`).
- `.dockerignore` — исключает из образа секреты и действующие конфиги (`settings.py`, `metrics_config.py`, `settings_runtime*.json`, `metrics_config_runtime.json`, все `.env`, `AI/GigaChat/`), временные файлы, артефакты и служебные директории. Конфиги монтируются в контейнеры при запуске.

### 0) Демо за пять минут

Чтобы посмотреть LoadLens без ключей API, Grafana, Prometheus и ручной настройки, есть отдельный стек `docker/docker-compose.demo.yml`. Одна команда из корня репозитория:

```bash
docker compose -f docker/docker-compose.demo.yml --env-file docker/demo.env --profile demo up --build
```

Откройте `http://localhost:5000` и войдите как `admin` / `loadlens-demo` (значения из `docker/demo.env`). Через минуту-две на дашборде появятся два прогона сервиса `demo-checkout`: синтетический `demo-checkout-step` и `demo-checkout-step-2`, который собран штатным конвейером отчётов по заглушкам Prometheus и модели. Архив, сравнение и прогноз мощностей заполнены; на дашборде висит баннер «Demo data». Остановить и удалить данные демо:

```bash
docker compose -f docker/docker-compose.demo.yml --env-file docker/demo.env --profile demo down -v
```

Что внутри и чем демо отличается от рабочей установки:

- Сервисы: `timescaledb` (порт наружу не публикуется), `llm-stub` (`tools/stubs/llm_stub.py --lang en`, OpenAI-совместимая заглушка модели), `prom-stub` (`tools/stubs/prom_stub.py`, синтетические ряды на любой PromQL-запрос) и `app` из того же образа, что и в продакшне. Отчёты выполняются потоком внутри `app` (`LOADLENS_JOB_RUNNER=thread`), поэтому Redis и воркер демо не нужны.
- `settings.demo.py` монтируется в контейнер как `/app/settings.py`, а `metrics_config.example.py` — как `/app/metrics_config.py`. Ваши `settings.py`, `docker/.env` и рантайм-файлы демо не читает и не меняет: у стека своё имя проекта `loadlens-demo`, свои контейнеры `loadlens_demo_*` и свой volume. `settings_runtime.json` демо пишет внутри контейнера.
- `--env-file docker/demo.env` обязателен: без него `docker compose` прочитал бы `docker/.env` рабочей установки. Все значения в `demo.env` — демонстрационные (пароль базы, администратор, ключ сессий); порт меняется переменной `APP_PORT`, скорость ответов заглушки — `DEMO_LLM_DELAY_SEC`.
- `LOADLENS_DEMO_SEED=1` в compose-файле включает автосев при старте `app`: демо-прогон записывается один раз, второй отчёт ставится в очередь один раз; повторные старты контейнера ничего не дублируют. Удалённый из архива `demo-checkout-step-2` пересобирается при следующем старте. В рабочем compose-файле переменная не задана, и автосев не работает.
- Сквозная проверка демо выполняется в CI (`.github/workflows/demo.yml`): по изменениям `docker/**`, `settings.demo.py`, `tools/stubs/**`, `loadlens_app/demo_data.py` и раз в неделю.

### 1) Подготовка конфигов

1. Скопируйте примеры и заполните их (обезличенные, без секретов):
   - Windows (PowerShell):
     ```powershell
     Copy-Item docker\env.example docker\.env -Force
     Copy-Item settings.example.py settings.py -Force
     if (-not (Test-Path metrics_config.py)) { Copy-Item metrics_config.example.py metrics_config.py }
     if (-not (Test-Path settings_runtime.json)) { Set-Content settings_runtime.json '{}' }
     if (-not (Test-Path metrics_config_runtime.json)) { Set-Content metrics_config_runtime.json '{}' }
     ```
   - Linux/macOS:
     ```bash
     cp docker/env.example docker/.env
     cp settings.example.py settings.py
     [ -f metrics_config.py ] || cp metrics_config.example.py metrics_config.py
     [ -f settings_runtime.json ] || echo '{}' > settings_runtime.json
     [ -f metrics_config_runtime.json ] || echo '{}' > metrics_config_runtime.json
     ```

   Все четыре файла (`settings.py`, `metrics_config.py`, `settings_runtime.json`, `metrics_config_runtime.json`) лежат в корне репозитория и монтируются в контейнеры. Если какого-то нет, `docker compose up` остановится с ошибкой `bind source path does not exist`.

2. Отредактируйте `docker/.env` (его читают контейнеры TimescaleDB, Redis и Celery):
   - `POSTGRES_USER` — имя пользователя БД (например, `app_user`)
   - `POSTGRES_PASSWORD` — пароль пользователя
   - `POSTGRES_DB` — имя базы (например, `loadtesting`)
   - `APP_PORT` — порт приложения на хосте (по умолчанию 5000)
   - `TSDB_PORT` — порт БД на хосте (по умолчанию 5432)
   - `TSDB_BIND` — интерфейс, на котором публикуется порт БД (по умолчанию `127.0.0.1`: только с этого хоста; приложение ходит в БД по внутренней сети compose)
   - `APP_DB_USER` / `APP_DB_PASSWORD` — непривилегированная роль БД для приложения (создаётся скриптом `initdb/03_app_role.sh`). Укажите те же значения в `settings.py` → `storage.timescale.user` / `password`. Пустые значения оставляют суперпользователя `POSTGRES_USER`
   - `APP_CPUS`, `APP_MEMORY_LIMIT`, `WORKER_CPUS`, `WORKER_MEMORY_LIMIT`, `REDIS_MEMORY_LIMIT` — ограничения контейнеров
   - `REDIS_PASSWORD` — **обязательный** пароль Redis (`openssl rand -hex 32`; только символы, допустимые в URL). Без него `docker compose up` не запустится
   - `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` — по умолчанию собираются из `REDIS_PASSWORD` (`redis://:<пароль>@redis:6379/0`). Задавайте их только для внешнего Redis и с паролем в URL
   - `CELERY_TASK_ALWAYS_EAGER` — `0`: отчёты и публикации строит воркер; `1`: они выполняются потоком внутри `app` (режим отладки)
   - `LOADLENS_JOB_RUNNER`, `LOADLENS_JOB_STALE_MIN`, `LOADLENS_JOB_QUEUE_MAX_MIN`, `LOADLENS_REPORT_TASK_LIMIT_SEC`, `WORKER_CONCURRENCY` — где выполняются фоновые задачи, когда зависшая задача считается прерванной, лимит времени задачи и сколько задач воркер выполняет одновременно (§4)
   - `LOADLENS_SECRET_KEY` — ключ подписи сессий: длинная случайная строка (`openssl rand -hex 32`), одна и та же между пересозданиями контейнера. Без неё ключ создаётся внутри контейнера и сессии сбрасываются при его пересоздании.
   - `LOADLENS_ADMIN_USER` / `LOADLENS_ADMIN_PASSWORD` — первый администратор; создаётся при первом открытии страницы входа, пока в базе нет пользователей. После первого входа удалите пароль из `.env`.
   - `LOADLENS_RECOVERY_USER` / `LOADLENS_RECOVERY_PASSWORD_HASH` — аварийный администратор: входит, только пока база недоступна, и может лишь исправить подключение к хранилищу на `/recovery`. Хеш выдаёт `docker compose exec app python -m loadlens_app.auth hash-password`; в `.env` заключите хеш в одинарные кавычки, иначе docker compose обрежет его на первом `$`. Подробнее — README, «Режим восстановления».
   - `LOADLENS_AUTH_ENABLED` — `1` (по умолчанию) вход обязателен; `0` отключает авторизацию (открытый доступ с правами администратора — только для отладки).
   - `LOADLENS_COOKIE_SECURE=1` и `LOADLENS_TRUSTED_PROXIES=1` — за HTTPS-прокси перед приложением (см. README, «Доступ и безопасность»).

3. Отредактируйте `settings.py` (используется приложением). Рекомендуется начать с `settings.example.py`:
   - Блок `storage.timescale`:
     - `host`: `timescaledb` (это имя сервиса в `docker-compose.yml`)
     - `port`: `5432`
     - `dbname`: совпадает с `POSTGRES_DB` из `.env`
     - `user`: совпадает с `POSTGRES_USER` из `.env`
     - `password`: совпадает с `POSTGRES_PASSWORD` из `.env`
     - `schema`: `public`
     - `table`: `metrics`
     - `llm_table`: `llm_reports`
     - `ensure_extension`: `True` (на всякий случай; расширение также создаётся скриптом инициализации)
   - Блок `data_sources` — каталог подключений (`prometheus`, `grafana_proxy`, `influxdb`): адрес, авторизация и TLS.
   - Блок `domain_sources` — какой источник читает домен. `default` задаёт источник доменов без своей строки. Для Grafana укажите `datasource_uid` или `datasource_name`, для InfluxQL — `database`, для Flux — `bucket`.
   - Блок `queries.lt_framework` — примеры запросов для Prometheus/InfluxDB и наборы ключей меток/тегов для подписи серий.

   При необходимости обновите блоки `llm`, `data_sources`, `domain_sources`, `grafana_base_url`, `loki_url` — используйте безопасные ключи/URL.

4. Секреты не попадают в образ: `.dockerignore` исключает `settings.py`, `metrics_config.py`, `settings_runtime*.json`, `metrics_config_runtime.json`, все `.env` и `AI/GigaChat/`, а `docker/docker-compose.yml` монтирует эти файлы в `app` и `celery_worker` (в воркер — только для чтения). Изменения из «Настроек» пишутся прямо в `settings_runtime.json` и `metrics_config_runtime.json` на хосте и переживают пересоздание контейнера. Сертификаты GigaChat кладите в `AI/GigaChat/` — каталог монтируется в `app` и `celery_worker` (отчёты с провайдером GigaChat строит воркер).

### 2) Поднятие в Docker

Все compose-команды выполняются из директории `docker/`:

- Windows (PowerShell):
  ```powershell
  cd docker
  docker compose up -d --build
  ```

- Linux/macOS:
  ```bash
  cd docker
  docker compose up -d --build
  ```

Проверка статуса:

```bash
cd docker
docker compose ps
docker logs ltar_app --tail=100
```

Приложение должно быть доступно на `http://localhost:5000/` (порт регулируется `APP_PORT` в `.env`); откроется страница входа. Войдите под `LOADLENS_ADMIN_USER` и добавьте остальных пользователей в «Настройки → Пользователи». Если администратор не задан в `.env`, создайте его командой:

```bash
cd docker
docker compose exec app python -m loadlens_app.auth create-user admin --role admin
```

Забытый пароль сбрасывается так же: `docker compose exec app python -m loadlens_app.auth set-password <логин>`.

Базы, созданные до появления авторизации, дополнительных действий не требуют: таблицы пользователей приложение создаёт само при первом обращении (пользователю БД из `settings.py` нужно право `CREATE` в схеме).

При каждом старте `app` в фоне проверяет миграции данных по таблице `app_migrations` и выполняет недостающие (`LOADLENS_RUN_MIGRATIONS=0` выключает): идентификаторы старых отчётов, таблица прогонов `run_index`, колонка `max_rps`, индекс `idx_metrics_run_name`. Так база, которую писала старая версия приложения, догоняет новую сразу после обновления. Выполненные одноразовые миграции не повторяются; если база в момент старта недоступна, попытки повторяются. Состояние: `docker compose exec app python -m loadlens_app.migrations --status`. На базе с уже сохранёнными прогонами архив пуст, пока первая миграция не заполнит `run_index` (около 2 с на 2 млн строк `metrics`); ход виден в `docker logs ltar_app`.

### HTTPS через Caddy

`docker/docker-compose.https.yml` ставит перед приложением [Caddy](https://caddyserver.com/): он занимает порты 80 и 443 и сам получает сертификат, а порт приложения публикуется только на `127.0.0.1`. Override также задаёт `LOADLENS_COOKIE_SECURE=1` и `LOADLENS_TRUSTED_PROXIES=1`. Нужен Docker Compose 2.24.4 или новее.

```bash
# docker/.env
LOADLENS_HOSTNAME=loadlens.example.com
# По желанию: тогда обычные команды docker compose (обновление, логи) тоже берут override; на Windows разделитель ;
COMPOSE_FILE=docker-compose.yml:docker-compose.https.yml
```

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.https.yml up -d --build
```

- **Публичное имя:** DNS указывает на хост, порты 80 и 443 доступны из интернета. Caddy получит сертификат Let's Encrypt при первом запросе и будет продлевать его сам. Сертификаты лежат в томе `caddy_data`, не удаляйте его.
- **Внутреннее имя:** раскомментируйте `tls internal` в `docker/Caddyfile`. Caddy выпустит сертификат своим CA, корневой сертификат (`docker compose exec caddy cat /data/caddy/pki/authorities/local/root.crt`) нужно добавить в доверенные на клиентах. Для сертификата корпоративного CA замените строку на `tls /certs/loadlens.crt /certs/loadlens.key` и смонтируйте файлы в сервис `caddy` в `/certs`.
- **Другой прокси** (nginx, Traefik, балансировщик): override не нужен. Задайте `LOADLENS_COOKIE_SECURE=1` и `LOADLENS_TRUSTED_PROXIES` по числу прокси, передавайте `X-Forwarded-For`, `-Proto` и `-Host` и не публикуйте порт 5000 дальше прокси.

### 3) Что делает TimescaleDB при старте

- Разворачивается образ `timescale/timescaledb:latest-pg16`.
- Применяется `docker/initdb/01_timescaledb.sql` — включает расширение TimescaleDB.
- Данные БД сохраняются в volume `tsdb_data` и сохраняются между перезапусками.
- Порт БД публикуется только на `127.0.0.1` (`TSDB_BIND`): снаружи хоста база недоступна.

Проверка расширения (опционально):

```bash
docker exec -it ltar_timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\dx"
```

### 4) Redis и Celery worker

- `redis` — официальный образ `redis:7.4-alpine` с volume `redis_data`. Используется как брокер и backend для Celery. Требует пароль `REDIS_PASSWORD` (`--requirepass`); healthcheck и `redis-cli` внутри контейнера авторизуются через `REDISCLI_AUTH`, например `docker exec ltar_redis redis-cli ping`.
- `celery_worker` — отдельный контейнер из того же образа, что и приложение. Он строит отчёты, публикует отчёты и прогнозы в Confluence (очередь `reports`) и выполняет прочие задачи (очередь `celery`). Команда запуска:  
  `celery -A loadlens_app.celery_app worker --loglevel=info -Q reports,celery --concurrency ${WORKER_CONCURRENCY:-2} --max-tasks-per-child 20`.  
  Процесс воркера заменяется после 20 задач (pandas не отдаёт память системе); лимит памяти контейнера — `WORKER_MEMORY_LIMIT` (2g).
- Переменные `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` по умолчанию указывают на `redis://:<REDIS_PASSWORD>@redis:6379/0`.
- `CELERY_TASK_ALWAYS_EAGER=0` и `LOADLENS_JOB_RUNNER=auto` (по умолчанию) — отчёты и публикации выполняет воркер, `app` только ставит задачи в очередь и показывает их ход из таблицы `report_jobs`; пока строится отчёт, страницы не замедляются. Графики Grafana, логи Loki и вложения Confluence отчёт скачивает и загружает сам, своим пулом потоков. Поставьте `CELERY_TASK_ALWAYS_EAGER=1` или `LOADLENS_JOB_RUNNER=thread`, чтобы выполнять задачи потоком внутри `app`, как раньше.
- Если Redis недоступен, задача не теряется: `app` выполняет её своим потоком и пишет в журнал `Could not send job … to the Celery broker`. Упавший воркер не повторяет задачу: через `LOADLENS_JOB_STALE_MIN` (15) минут без признаков жизни она становится ошибкой «Задача прервана: воркер не отвечает». Задача, которую воркер не взял за `LOADLENS_JOB_QUEUE_MAX_MIN` (120) минут, — ошибкой «Задача прервана: воркер не взял её из очереди». Лимит времени одной задачи — `LOADLENS_REPORT_TASK_LIMIT_SEC` (3600 с). Все эти переменные compose передаёт и `app`, и воркеру.
- Одновременно воркер выполняет `WORKER_CONCURRENCY` (2) задач, остальные ждут в очереди; лимит приложения `limits.max_parallel_jobs` по умолчанию 4. Поднимая `WORKER_CONCURRENCY`, поднимите `WORKER_MEMORY_LIMIT` и проверьте лимиты LLM-провайдера: `max_concurrent` действует на каждый процесс воркера. `docker compose up --scale celery_worker=2` не сработает: у сервиса задан `container_name`.
- Healthcheck воркера — `celery inspect ping` раз в минуту (`docker compose ps` показывает `healthy`). При остановке воркер ждёт до 2 минут (`stop_grace_period`), чтобы идущие задачи закончились.
- Логи воркера: `docker logs -f ltar_celery_worker` (строки `Task loadlens.report[<номер задачи>] received` и `… succeeded`).  
  Перезапуск: `cd docker && docker compose restart celery_worker`.
- Что выполняется сейчас и что ждёт:
  ```bash
  cd docker
  docker compose exec celery_worker celery -A loadlens_app.celery_app inspect active
  docker exec ltar_redis redis-cli llen reports
  ```
- Код у `app` и воркера должен совпадать: после обновления пересобирайте и перезапускайте оба сервиса. Часы хоста одни на оба контейнера; если воркер вынесен на другой хост, синхронизируйте часы (NTP).

Режимы, поведение при сбоях, просмотр очереди и запуск воркера на Windows — в [docs/jobs-runner.md](../docs/jobs-runner.md).

### 5) Тестовая запись в БД

После поднятия сервисов можно выполнить быстрый self-test записи в `public.metrics`:

```bash
docker exec -it ltar_app python AI/test_timescale_write.py
```

Если всё ок — увидите сообщение `OK: тестовые строки записаны. Проверьте public.metrics.`

### 6) Важное про секреты и гит

- В `.gitignore` уже добавлены правила для локальных конфигов: `settings.py`, `settings_runtime.json`, все `.env` (включая `docker/.env`).
- Примеры остаются в репозитории: `settings.example.py`, `docker/env.example`.
- Ключ сессий (`LOADLENS_SECRET_KEY` или файл `.loadlens_secret_key`) и пароль администратора — секреты: не коммитьте их; файл ключа уже в `.gitignore` и `.dockerignore`.

Если `settings.py` или `docker/.env` уже были закоммичены ранее, их нужно убрать из индекса git (файлы останутся локально):

```bash
git rm --cached settings.py docker/.env || true
git add .
git commit -m "chore: stop tracking local configs; add examples"
```

> Примечание: если у вас есть дополнительные приватные файлы (например, `AI/config.py`), добавьте их в `.gitignore` по аналогии и уберите из индекса с помощью `git rm --cached`.

#### Обновление установки, собранной до исключения секретов из образа

Образы, собранные раньше, содержат `settings.py`, `settings_runtime*.json`, `metrics_config*` и `docker/.env` в своих слоях. Считайте все секреты из них скомпрометированными и смените их.

1. В `docker/.env` добавьте `REDIS_PASSWORD` и удалите строки `CELERY_BROKER_URL=redis://redis:6379/0` и `CELERY_RESULT_BACKEND=redis://redis:6379/0`: без пароля в URL Celery не подключится к Redis.
2. Создайте недостающие файлы конфигурации (шаг 1 выше).
3. Пересоберите образы без кэша и перезапустите сервисы:
   ```bash
   cd docker
   docker compose build --pull --no-cache app celery_worker
   docker compose up -d
   ```
4. Убедитесь, что в новом образе секретов нет (имя образа покажет `docker compose config --images`):
   ```bash
   docker run --rm --entrypoint sh <образ app> -c 'ls /app/settings.py /app/settings_runtime.json /app/metrics_config.py /app/docker 2>&1'
   ```
   Для каждого пути должно быть `No such file or directory`.
5. Приложение в образе работает от пользователя `app` (UID 1000) под gunicorn. На Linux-хосте файлы, которые монтируются в контейнер (`settings_runtime.json`, `metrics_config_runtime.json`, `data_collectors/temporary_files`), должны быть доступны на запись UID 1000; другой UID задаётся аргументами сборки `APP_UID`/`APP_GID`.
6. Для уже инициализированной базы создайте роль приложения вручную и переключите на неё `settings.py` (`storage.timescale.user` / `password`):
   ```bash
   cd docker
   docker compose exec timescaledb sh /docker-entrypoint-initdb.d/03_app_role.sh
   ```
   Для базы на отдельном сервере скрипт запускается с любой машины, где есть `psql` и доступ к серверу:
   ```bash
   PGHOST=<адрес БД> PGPORT=5432 PGPASSWORD=<пароль администратора> \
   POSTGRES_USER=<администратор> POSTGRES_DB=<база> \
   APP_DB_USER=loadlens_app APP_DB_PASSWORD=<новый пароль> \
   sh docker/initdb/03_app_role.sh
   ```
   Что делает скрипт:
   - создаёт роль без прав суперпользователя, без CREATEDB, CREATEROLE и REPLICATION, а у существующей роли снимает эти права;
   - передаёт ей только таблицы LoadLens из схемы `public`, остальные таблицы базы не трогает (другие схема и список таблиц задаются `APP_DB_SCHEMA` и `APP_DB_TABLES`);
   - оставляет администратору и прежнему владельцу таблиц доступ через членство в роли, поэтому старая учётная запись в `settings.py` продолжает работать и к ней можно вернуться;
   - выполняет всё одной транзакцией с ожиданием блокировок не дольше 5 секунд: при любой ошибке ничего не меняется, повторный запуск безопасен.

   Если в таблице `metrics` уже есть данные, запускайте скрипт от суперпользователя: сменить владельца чанков TimescaleDB может только он, иначе скрипт остановится с понятной ошибкой. Пароль новой роли передаётся серверу в команде `CREATE ROLE`. При запуске от суперпользователя скрипт отключает журналирование запросов на время транзакции. При запуске от другой учётной записи и включённом `log_statement` пароль может попасть в журнал сервера.
7. Удалите старые образы и кэш сборки, в котором остались прежние слои: `docker image ls`, `docker image rm <id>`, `docker builder prune -a`. Если образы отправлялись в реестр, удалите там старые теги и дайджесты.

### 7) Типичные сценарии

- Пересобрать приложение после изменений кода (воркер собирается из того же образа и должен обновляться вместе с `app`):
  ```bash
  cd docker && docker compose up -d --build app celery_worker
  ```

- Обновить только БД (обычно не требуется):
  ```bash
  cd docker && docker compose up -d timescaledb
  ```

- Перезапустить только Celery worker:
  ```bash
  cd docker && docker compose up -d --build celery_worker
  ```

- Посмотреть логи Redis:
  ```bash
  docker logs ltar_redis --tail=100
  ```

- Перезапустить всё:
  ```bash
  cd docker && docker compose down
  cd docker && docker compose up -d --build
  ```

### 8) Подключение приложения к внешней БД (не из compose)

В `settings.py` укажите реальные `host`, `port`, `dbname`, `user`, `password` целевой TimescaleDB. В таком случае сервис `timescaledb` из compose можно не поднимать (или удалить из `docker-compose.yml`).

### 9) Рантайм‑оверрайды и области (per‑area)

- `settings_runtime.json` — позволяет изменить разделы `llm`, `data_sources`, `domain_sources`, `default_params`, `queries` без пересборки/перезапуска. Поддерживает блок `per_area`: можно задать отличающиеся настройки для разных проектных областей. Каталог `data_sources` общий, область заменяет только `domain_sources`.
- `metrics_config_runtime.json` — аналогично для ссылок на панели/логи Confluence в разрезе областей.
- Оба файла можно редактировать из вкладки «Настройки» в приложении. Изменения применяются динамически.

---

Готово. После выполнения шагов выше у вас будет развёрнуто приложение и TimescaleDB в Docker. Если что-то пойдёт не так — посмотрите логи контейнеров и сравните заполненные конфиги с примерами.


