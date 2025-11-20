## Развертывание в Docker (приложение + TimescaleDB)

Ниже — исчерпывающая инструкция: какие файлы добавлены, что в них писать, и как поднять всё в Docker на Windows (PowerShell) и Linux/macOS.

### Что добавлено

- `Dockerfile` — актуализирован: добавлены переменные окружения, `curl` для healthcheck и `HEALTHCHECK`.
- `docker-compose.yml` — поднимает сервисы: `app` (веб), `timescaledb` (TimescaleDB/PostgreSQL), `redis` (брокер) и `celery_worker` (фоновые задачи).
- `initdb/01_timescaledb.sql` — автоматическое включение расширения TimescaleDB в базе при первом старте.
- Redis + Celery worker — через `docker-compose` теперь поднимаются брокер `redis` и отдельный контейнер `celery_worker` для фоновых задач.
- `env.example` (лежит в этой же папке) — пример файла окружения для Compose. Скопируйте его в `.env` и отредактируйте.
- `settings.example.py` — обезличенный пример `settings.py`. Скопируйте и заполните.
- `.gitignore` — добавлены правила для игнорирования действующих конфигов (`settings.py`, `settings_runtime.json`, `docker/.env` и прочих `.env`).
- `.dockerignore` — исключает временные файлы, артефакты и служебные директории из образа.

### 1) Подготовка конфигов

1. Скопируйте примеры и заполните их (обезличенные, без секретов):
   - Windows (PowerShell):
     ```powershell
     Copy-Item docker\env.example docker\.env -Force
     Copy-Item settings.example.py settings.py -Force
     ```
   - Linux/macOS:
     ```bash
     cp docker/env.example docker/.env
     cp settings.example.py settings.py
     ```

2. Отредактируйте `docker/.env` (его читают контейнеры TimescaleDB, Redis и Celery):
   - `POSTGRES_USER` — имя пользователя БД (например, `app_user`)
   - `POSTGRES_PASSWORD` — пароль пользователя
   - `POSTGRES_DB` — имя базы (например, `loadtesting`)
   - `APP_PORT` — порт приложения на хосте (по умолчанию 5000)
   - `TSDB_PORT` — порт БД на хосте (по умолчанию 5432)
   - `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` — URL брокера/хранилища результатов (по умолчанию `redis://redis:6379/0`)
   - `CELERY_TASK_ALWAYS_EAGER` — `0` для настоящих фоновых задач, `1` чтобы выполнять их синхронно (режим отладки)

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
   - Блок `metrics_source` — источник метрик тестируемой системы (SUT):
     - Режим `type`: `prometheus` (прямой доступ) или `grafana_proxy` (через `/api/datasources/proxy/...`).
     - Для `prometheus`: укажите `prometheus.url`.
     - Для `grafana_proxy`: укажите `grafana.base_url`, `auth` и `prometheus_datasource` (uid/name/id).
   - Блок `lt_metrics_source` — источник метрик инструмента нагрузки (например, k6/JMeter):
     - Режим `type`: `prometheus` | `grafana_proxy` | `influxdb`.
     - `prometheus.url` — прямой PromQL.
     - `grafana.influxdb_datasource` — использование InfluxDB через Grafana‑прокси.
     - `influxdb.{url,org,bucket,database,token}` — прямой доступ к InfluxDB (Flux/InfluxQL).
   - Блок `queries.lt_framework` — примеры запросов для Prometheus/InfluxDB и наборы ключей меток/тегов для подписи серий.

   При необходимости обновите блоки `llm`, `metrics_source`, `grafana_base_url`, `loki_url` — используйте безопасные ключи/URL.

4. (Опционально) Если вы не хотите, чтобы секреты попадали в образ, можно смонтировать локальный `settings.py` в контейнер (см. комментарии в `docker/docker-compose.yml`).

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

Приложение должно быть доступно на `http://localhost:5000/` (порт регулируется `APP_PORT` в `.env`).

### 3) Что делает TimescaleDB при старте

- Разворачивается образ `timescale/timescaledb:latest-pg16`.
- Применяется `docker/initdb/01_timescaledb.sql` — включает расширение TimescaleDB.
- Данные БД сохраняются в volume `tsdb_data` и сохраняются между перезапусками.

Проверка расширения (опционально):

```bash
docker exec -it ltar_timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\dx"
```

### 4) Redis и Celery worker

- `redis` — официальный образ `redis:7.4-alpine` с volume `redis_data`. Используется как брокер и backend для Celery.
- `celery_worker` — отдельный контейнер из того же образа, что и приложение. Команда запуска:  
  `celery -A loadlens_app.celery_app worker --loglevel=info`.
- Переменные `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` по умолчанию указывают на `redis://redis:6379/0`.
- `CELERY_TASK_ALWAYS_EAGER=0` — фоновые задачи (скачивание графиков, загрузка вложений, LLM) выполняются асинхронно.
  Поставьте `1`, если нужно временно отключить Redis/Celery и выполнять всё внутри процесса `app`.
- Логи воркера: `docker logs -f ltar_celery_worker`.  
  Перезапуск: `cd docker && docker compose restart celery_worker`.
- Масштабирование (независимые воркеры): `cd docker && docker compose up -d --scale celery_worker=2`.

### 5) Тестовая запись в БД

После поднятия сервисов можно выполнить быстрый self-test записи в `public.metrics`:

```bash
docker exec -it ltar_app python AI/test_timescale_write.py
```

Если всё ок — увидите сообщение `OK: тестовые строки записаны. Проверьте public.metrics.`

### 6) Важное про секреты и гит

- В `.gitignore` уже добавлены правила для локальных конфигов: `settings.py`, `settings_runtime.json`, все `.env` (включая `docker/.env`).
- Примеры остаются в репозитории: `settings.example.py`, `docker/env.example`.

Если `settings.py` или `docker/.env` уже были закоммичены ранее, их нужно убрать из индекса git (файлы останутся локально):

```bash
git rm --cached settings.py docker/.env || true
git add .
git commit -m "chore: stop tracking local configs; add examples"
```

> Примечание: если у вас есть дополнительные приватные файлы (например, `AI/config.py`), добавьте их в `.gitignore` по аналогии и уберите из индекса с помощью `git rm --cached`.

### 7) Типичные сценарии

- Пересобрать приложение после изменений кода:
  ```bash
  cd docker && docker compose up -d --build app
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

- `settings_runtime.json` — позволяет изменить разделы `llm`, `metrics_source`, `lt_metrics_source`, `default_params`, `queries` без пересборки/перезапуска. Поддерживает блок `per_area`: можно задать отличающиеся настройки для разных проектных областей.
- `metrics_config_runtime.json` — аналогично для ссылок на панели/логи Confluence в разрезе областей.
- Оба файла можно редактировать из вкладки «Настройки» в приложении. Изменения применяются динамически.

---

Готово. После выполнения шагов выше у вас будет развёрнуто приложение и TimescaleDB в Docker. Если что-то пойдёт не так — посмотрите логи контейнеров и сравните заполненные конфиги с примерами.


