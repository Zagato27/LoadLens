# Используем официальный образ Python
FROM python:3.12-slim

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем системные зависимости (если необходимо)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код приложения в контейнер
COPY . .

# Открываем порт для взаимодействия (опционально, если нужно)
EXPOSE 5000

# Определяем команду для запуска приложения
CMD ["python", "app.py"]
