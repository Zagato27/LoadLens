import os
import yaml


def load_config(config_file='config.yaml'):
    """Загружает конфигурацию приложения из YAML-файла и переменных окружения.

    Параметры:
        config_file (str): Путь к YAML с конфигурацией по умолчанию.

    Возвращает:
        dict: Словарь настроек (Confluence, Grafana и т.д.).

    Побочные эффекты:
        Читает файл и переменные окружения.
    """
    config = {}

    # Загрузка конфигурации из файла
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config.update(yaml.safe_load(file))

    # Загрузка конфигурации из переменных окружения
    config['confluence'] = {
        'url': os.environ.get('CONFLUENCE_URL', config.get('confluence', {}).get('url')),
        'username': os.environ.get('CONFLUENCE_USERNAME', config.get('confluence', {}).get('username')),
        'password': os.environ.get('CONFLUENCE_PASSWORD', config.get('confluence', {}).get('password')),
    }

    config['grafana'] = {
        'api_key': os.environ.get('GRAFANA_API_KEY', config.get('grafana', {}).get('api_key')),
        'base_url': os.environ.get('GRAFANA_BASE_URL', config.get('grafana', {}).get('base_url')),
    }



    # Добавьте другие конфигурации (InfluxDB, Dynatrace, Oracle) аналогично

    return config
