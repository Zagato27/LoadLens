## Развертывание в Docker (приложение + TimescaleDB)

Ниже — исчерпывающая инструкция: какие файлы добавлены, что в них писать, и как поднять всё в Docker на Windows (PowerShell) и Linux/macOS.

### Что добавлено

- `Dockerfile` — актуализирован: добавлены переменные окружения, `curl` для healthcheck и `HEALTHCHECK`.
- `docker-compose.yml` — поднимает два сервиса: `app` (это репозиторий) и `timescaledb` (TimescaleDB/PostgreSQL).
- `docker/initdb/01_timescaledb.sql` — автоматическое включение расширения TimescaleDB в базе при первом старте.
- `env.example` — пример файла окружения для Compose (порты и креды БД). Скопируйте его в `.env` и отредактируйте.
- `settings.example.py` — обезличенный пример `settings.py`. Скопируйте и заполните.
- `utils/config.example.yaml` — обезличенный пример `utils/config.yaml`.
- `.gitignore` — добавлены правила для игнорирования действующих конфигов (`settings.py`, `utils/config.yaml`, `.env`).
- `.dockerignore` — исключает временные файлы, артефакты и служебные директории из образа.

### 1) Подготовка конфигов

1. Скопируйте примеры и заполните их (обезличенные, без секретов):
   - Windows (PowerShell):
     ```powershell
     Copy-Item env.example .env -Force
     Copy-Item settings.example.py settings.py -Force
     Copy-Item utils\config.example.yaml utils\config.yaml -Force
     ```
   - Linux/macOS:
     ```bash
     cp env.example .env
     cp settings.example.py settings.py
     cp utils/config.example.yaml utils/config.yaml
     ```

2. Отредактируйте `.env` (используется только сервисом TimescaleDB):
   - `POSTGRES_USER` — имя пользователя БД (например, `app_user`)
   - `POSTGRES_PASSWORD` — пароль пользователя
   - `POSTGRES_DB` — имя базы (например, `loadtesting`)
   - `APP_PORT` — порт приложения на хосте (по умолчанию 5000)
   - `TSDB_PORT` — порт БД на хосте (по умолчанию 5432)

3. Отредактируйте `settings.py` (используется приложением):
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

   При необходимости обновите блоки `llm`, `metrics_source`, `grafana_base_url`, `loki_url` — используйте безопасные ключи/URL.

4. (Опционально) Если вы не хотите, чтобы секреты попадали в образ, можно смонтировать локальный `settings.py` в контейнер (см. комментарии в `docker-compose.yml`).

### 2) Поднятие в Docker

В корне репозитория выполните:

- Windows (PowerShell):
  ```powershell
  docker compose up -d --build
  ```

- Linux/macOS:
  ```bash
  docker compose up -d --build
  ```

Проверка статуса:

```bash
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

### 4) Тестовая запись в БД

После поднятия сервисов можно выполнить быстрый self-test записи в `public.metrics`:

```bash
docker exec -it ltar_app python AI/test_timescale_write.py
```

Если всё ок — увидите сообщение `OK: тестовые строки записаны. Проверьте public.metrics.`

### 5) Важное про секреты и гит

- В `.gitignore` уже добавлены правила для локальных конфигов: `settings.py`, `utils/config.yaml`, `.env`.
- Примеры остаются в репозитории: `settings.example.py`, `utils/config.example.yaml`, `env.example`.

Если `settings.py` и/или `utils/config.yaml` уже были закоммичены ранее, их нужно убрать из индекса git (файлы останутся локально):

```bash
git rm --cached settings.py utils/config.yaml || true
git add .
git commit -m "chore: stop tracking local configs; add examples"
```

> Примечание: если у вас есть дополнительные приватные файлы (например, `AI/config.py`), добавьте их в `.gitignore` по аналогии и уберите из индекса с помощью `git rm --cached`.

### 6) Типичные сценарии

- Пересобрать приложение после изменений кода:
  ```bash
  docker compose up -d --build app
  ```

- Обновить только БД (обычно не требуется):
  ```bash
  docker compose up -d timescaledb
  ```

- Перезапустить всё:
  ```bash
  docker compose down
  docker compose up -d --build
  ```

### 7) Подключение приложения к внешней БД (не из compose)

В `settings.py` укажите реальные `host`, `port`, `dbname`, `user`, `password` целевой TimescaleDB. В таком случае сервис `timescaledb` из compose можно не поднимать (или удалить из `docker-compose.yml`).

---

Готово. После выполнения шагов выше у вас будет развёрнуто приложение и TimescaleDB в Docker. Если что-то пойдёт не так — посмотрите логи контейнеров и сравните заполненные конфиги с примерами.


