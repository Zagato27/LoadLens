# Используем официальный образ Python
FROM python:3.12-slim

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

# Копируем остальной код приложения в контейнер.
# Secrets and live configs (settings.py, settings_runtime*.json, metrics_config*, .env) are excluded by
# .dockerignore and mounted at run time by docker-compose.yml: an image must never contain them.
COPY . .

# Открываем порт для взаимодействия (опционально, если нужно)
EXPOSE 5000

# Healthcheck: the public liveness endpoint (every other page requires a login)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://localhost:5000/healthz || exit 1

# Определяем команду для запуска приложения
CMD ["python", "app.py"]
