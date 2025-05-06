# Автоматическое создание отчетов в Confluence

Это приложение предназначено для автоматического сбора отчетов по результатам нагрузочного тестирования и обновления их в Confluence.

## Использование

### Параметры

- `start`: Время начала теста (в миллисекундах)
- `end`: Время окончания теста (в миллисекундах)
- `user`: Логин для доступа к Confluence
- `password`: Пароль для доступа к Confluence
- `url_basic`: URL вашей Confluence
- `space_conf`: Пространство в Confluence для хранения отчетов
- `page_parent_id`: ID родительской страницы в Confluence
- `page_sample_id`: ID шаблона страницы отчета в Confluence
- `service`: Сервис, для которого проводится тестирование
- `testType`: Тип тестирования (`Kafka`, `cofiguration`, `default`, `full`) для выбора шаблона и построения нужного отчета

### Запуск

1. Клонируйте репозиторий:
   ```sh
   git clone https://gitlab.com/your-repo.git
   cd your-repo

2. Установите зависимости:
   ```sh
   pip install -r requirements.txt

3. Запустите скрипт:
   ```sh
   python update_page.py --starttime <start> --endtime <end>

4. Пример
   ```sh
   python update_page.py --starttime 1720562420464 --endtime 1720591114582

5. Примечания
Убедитесь, что у вас есть доступ к указанным URL Grafana и Confluence.
Скрипт автоматически копирует шаблон страницы в Confluence и обновляет его данными из Grafana.
