from flask import Flask, request, jsonify, render_template
from update_page import update_report  # Замените на имя файла, где находится `update_report`
from config import CONFIG  # Базовые настройки
from metrics_config import METRICS_CONFIG  # Конфигурация метрик
from datetime import datetime



app = Flask(__name__)

# Рендеринг главной страницы с формой
@app.route('/')
def home():
    return render_template('index.html')

# Вспомогательная функция для конвертации времени
def convert_to_timestamp(date_str):
    """Конвертирует строку даты и времени в миллисекундный timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
    return int(dt.timestamp() * 1000)

@app.route('/services', methods=['GET'])
def get_services():
    # Извлекаем все ключи словаря METRICS_CONFIG — это и есть названия сервисов
    services = list(METRICS_CONFIG.keys())
    return jsonify(services), 200

# Маршрут для создания отчета
@app.route('/create_report', methods=['POST'])
def create_report():
    data = request.json  # Получаем JSON-данные от формы
    start_str = data.get('start')
    end_str = data.get('end')
    service = data.get('service')

    # Проверка наличия необходимых параметров
    if not all([start_str, end_str, service]):
        return jsonify({"status": "error", "message": "Пожалуйста, укажите время начала, окончания и сервис"}), 400

    # Конвертация времени в timestamp
    try:
        start = convert_to_timestamp(start_str)
        end = convert_to_timestamp(end_str)
    except ValueError:
        return jsonify({"status": "error", "message": "Некорректный формат времени. Используйте формат YYYY-MM-DDTHH:MM"}), 400

    # Проверка наличия конфигурации для выбранного сервиса
    if service not in METRICS_CONFIG:
        return jsonify({"status": "error", "message": f"Конфигурация для сервиса '{service}' не найдена"}), 400

    try:
        # Вызываем update_report с параметрами из POST-запроса
        response = update_report(start, end, service)
        return jsonify({"status": "success", "message": "Отчет создан успешно"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0')
