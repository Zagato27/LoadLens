# Используем официальный образ Python
FROM python:3.12-slim

# Version and build id (date and short commit hash) of this image. .github/workflows/release.yml passes them
# as build args; GET /about and Settings → About show them. LICENSE, NOTICE and THIRD_PARTY_NOTICES.md ride
# along with the source copy below.
ARG LOADLENS_VERSION=""
ARG LOADLENS_BUILD=""
ENV LOADLENS_BUILD=$LOADLENS_BUILD
LABEL org.opencontainers.image.title="LoadLens" \
      org.opencontainers.image.description="Load test reports with SLA verdicts, verified AI analysis, forecasting and Confluence export" \
      org.opencontainers.image.source="https://github.com/Zagato27/LoadLens" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="$LOADLENS_VERSION"

# Принудительно ведём логирование в stdout/stderr и не пишем .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем системные зависимости (если необходимо)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user. UID/GID 1000 match the usual owner of the repository on a Linux host, so the
# bind-mounted runtime settings and temporary files stay writable; override with build args if needed.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app

# Копируем остальной код приложения в контейнер.
# Secrets and live configs (settings.py, settings_runtime*.json, metrics_config*, .env) are excluded by
# .dockerignore and mounted at run time by docker-compose.yml: an image must never contain them.
COPY --chown=app:app . .
RUN chown app:app /app && mkdir -p data_collectors/temporary_files && chown app:app data_collectors/temporary_files

USER app

# Открываем порт для взаимодействия (опционально, если нужно)
EXPOSE 5000

# Healthcheck: the public liveness endpoint (every other page requires a login)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://localhost:5000/healthz || exit 1

# Production server instead of the Flask development server. gunicorn reads GUNICORN_CMD_ARGS.
# One process with threads keeps the in-memory job registry, login throttling and limits in one
# place; gthread workers are not killed for long requests such as LLM calls.
# The access log is gunicorn's default format plus %(M)s, the request time in milliseconds.
ENV GUNICORN_CMD_ARGS="--bind 0.0.0.0:5000 --worker-class gthread --workers 1 --threads 8 --timeout 120 --graceful-timeout 30 --access-logfile - --access-logformat '%(h)s %(l)s %(u)s %(t)s \"%(r)s\" %(s)s %(b)s \"%(f)s\" \"%(a)s\" %(M)s'"
CMD ["gunicorn", "app:app"]
