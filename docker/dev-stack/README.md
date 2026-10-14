# Dev stack для проверки пакетов запросов

Стенд поднимает настоящие экспортеры и генераторы нагрузки, чтобы проверить запросы пакетов
(`AI/query_packs/packs/*.toml`) на живых метриках, а не только синтаксически. Все порты открыты только на
`127.0.0.1`; стенд не предназначен для продакшна и не нужен для работы LoadLens.

| Сервис | Что даёт | Порт на хосте |
|---|---|---|
| Prometheus 2.53 (remote write receiver, native histograms) | метрики всех сервисов, k6 remote write | 19090 |
| Grafana 11.2 | прокси InfluxQL для LoadLens (`admin` / `devstack-admin`) | 13000 |
| InfluxDB 1.8 (Flux включён) | k6 `--out influxdb`, Flux через `/api/v2/query` | 18086 |
| node_exporter | ресурсы узла (VM Docker Desktop) | — |
| PostgreSQL 16 + postgres_exporter | база Keycloak и pgbench | — |
| PgBouncer (transaction pooling, пул 4) + pgbouncer_exporter | очередь клиентов к пулу | — |
| Redis 7 (maxmemory 64 МБ) + redis_exporter | команды, память, вытеснения | — |
| Keycloak 26 (Quarkus, Micrometer) | JVM и HTTP-метрики Micrometer | 18080 |
| FastAPI-приложение (`pyapp/`) | Python runtime и HTTP-метрики FastAPI | 18000 |
| Kafka 3.8 KRaft + kafka_exporter (профиль `kafka`) | топики, consumer lag | — |
| Locust + locust_exporter (профиль `locust`) | метрики Locust | — |
| k6 0.54 (профиль `load`) | ступенчатая нагрузка: InfluxDB, Prometheus RW (trend stats и native histograms) | — |

## Запуск

```bash
cd docker/dev-stack
docker compose --profile kafka --profile locust up -d
docker compose --profile load run --rm -d k6           # ~8 минут ступеней 20 → 40 → 60 RPS
docker compose --profile load run --rm -d k6-native    # то же вдвое короче, native histograms

# Нагрузка на базы и Kafka (в Git Bash на Windows добавьте MSYS_NO_PATHCONV=1 перед docker compose exec)
docker compose exec -e PGPASSWORD=devstack postgres pgbench -i -s 5 -h pgbouncer -U devstack keycloak
docker compose exec -e PGPASSWORD=devstack postgres pgbench -c 12 -j 2 -T 420 -h pgbouncer -U devstack keycloak
docker compose exec redis redis-benchmark -q -n 3000000 -c 20 -P 8 -r 400000 -d 256 -t set,get,incr,lpush
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --create --topic devstack-orders --partitions 3
docker compose exec kafka /opt/kafka/bin/kafka-producer-perf-test.sh --topic devstack-orders --num-records 600000 --record-size 200 --throughput 1500 --producer-props bootstrap.servers=kafka:9092
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic devstack-orders --group devstack-consumer --max-messages 300000

# Проверка пакетов (из корня репозитория; окно — последние 15 минут)
python docker/dev-stack/check_packs.py --json dev-stack-results.json
```

`check_packs.py` выполняет каждый запрос пакета через проверку LoadLens (`loadlens_app.query_preview`) — тем же
путём, что кнопка «Проверить»: Prometheus напрямую, InfluxQL через прокси Grafana, Flux напрямую в InfluxDB.
Переменные пакетов подобраны под стенд: Keycloak не ставит тег `application`, поэтому для Micrometer
`service_label = job`. Код выхода 1 — обязательный запрос пакета вернул пусто.

**Корпоративный прокси с подменой TLS.** Если `pip` внутри контейнера падает с `CERTIFICATE_VERIFY_FAILED`,
скачайте колёса на хосте (там доверенный сертификат уже настроен) — `pyapp` установит их без сети:

```bash
python -m pip download --dest docker/dev-stack/pyapp/wheels --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp -r docker/dev-stack/pyapp/requirements.txt
```

Остановка: `docker compose --profile kafka --profile locust --profile load down -v`.

## Подключение LoadLens к стенду

Для ручной проверки в интерфейсе добавьте в каталог источников Prometheus `http://127.0.0.1:19090`,
Grafana `http://127.0.0.1:13000` (`admin` / `devstack-admin`, датасорс `InfluxDB-k6`) и InfluxDB
`http://127.0.0.1:18086` (bucket `k6/autogen`), затем «Запросы → Добавить из пакета → Определить по источникам».

## Результат прогона 2026-10-13

Все 16 пакетов, которые стенд может проверить, вернули данные по каждому обязательному запросу; в каталоге
они отмечены `verified = true`. Пустыми остались необязательные запросы, для которых на стенде нет источника:

| Пакет | Пустой запрос | Почему |
|---|---|---|
| jvm-micrometer | Tomcat busy threads | Keycloak работает на Quarkus/Vert.x, а не на Tomcat |
| http-micrometer | outgoing HTTP average time | Keycloak на стенде не делает исходящих HTTP-вызовов |
| http-go | три запроса gRPC | у Prometheus нет gRPC-сервера |
| kafka-exporter | under-replicated partitions | запрос показывает только ненулевые значения, а кластер здоров |

Прогон нашёл и исправил три ошибки пакетов:
- Flux-перцентили в `lt-k6-influx2-flux` падали на InfluxDB 1.8 без явного `method` у `quantile()`.
- Новые версии `pgbouncer_exporter` называют статистику `pgbouncer_stats_totals_*`: `db-pgbouncer` принимает обе схемы имён.
- Если в Prometheus включены native histograms, классические `_bucket`, `_count` и `_sum` у приложений, которые
  отдают native-гистограммы (сам Prometheus), не сохраняются. Пакеты HTTP читают классические ряды, поэтому
  в таком Prometheus включите `scrape_classic_histograms: true` (2.x) или `always_scrape_classic_histograms: true` (3.x).

Автоопределение на стенде для Prometheus, InfluxQL через Grafana и Flux поставило набор «Spring Boot на
Kubernetes + k6 (InfluxDB)» первым: подходят 6 из 9 пакетов, остальные три требуют Kubernetes и HikariCP.

## Что стенд не покрывает

Kubernetes (cAdvisor, kube-state-metrics), JMX exporter и брокерные метрики Kafka, OpenTelemetry, Istio,
ingress-nginx, Flask и Django, HikariCP (у Keycloak пул Agroal), MySQL, MongoDB, Windows, JMeter, Gatling и
Yandex.Tank. Пакеты для них проверены только синтаксически и на контракте рядов и единиц; в каталоге у них нет
отметки «проверен на стенде».
