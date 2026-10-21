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
| ArangoDB 3.12 + arangobench (профиль `arangodb`) | встроенные метрики `/_admin/metrics/v2`: HTTP, AQL, планировщик, транзакции | 18529 |
| Locust + locust_exporter (профиль `locust`) | метрики Locust | — |
| k6 0.54 (профиль `load`) | ступенчатая нагрузка: InfluxDB, Prometheus RW (trend stats и native histograms) | — |
| Gatling 3.10.5 + VictoriaMetrics 1.106 (профиль `gatling`) | Graphite-вывод Gatling, ступени 10 → 20 → 30 пользователей/с | 18428 |
| VictoriaMetrics с relabeling (профиль `gatling`) | те же метрики как `gatling_request_*` / `gatling_users_*` с метками (`gatling/vm-relabel.yml`) | 18429 |
| Генератор нагрузки (профиль `generator`) | SSH для страницы «Запуск теста»: k6, Locust и бандл Gatling | 12222 |
| docker-socket-proxy (профиль `generator`) | Docker API для Docker-генератора: контейнеры, образы, exec, сети, тома | 12375 |

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

# Gatling → Graphite → VictoriaMetrics (~6 минут; бандл Gatling — см. ниже)
docker compose --profile gatling up -d victoriametrics
docker compose --profile gatling run --rm gatling
# то же в VictoriaMetrics с relabeling; outputDirectoryBaseName — тег прогона вместо имени симуляции
docker compose --profile gatling up -d victoriametrics-labels
docker compose --profile gatling run --rm -e JAVA_OPTS="-Dgatling.data.graphite.host=victoriametrics-labels -Dgatling.core.outputDirectoryBaseName=devstack-labels" gatling

# ArangoDB: CRUD документов, AQL-вставки и AQL-чтения 10 минут (ARANGO_LOAD_SEC меняет длительность)
docker compose --profile arangodb up -d arangodb
docker compose --profile arangodb run --rm arangobench

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

**Бандл Gatling.** Нужен Gatling 3.10.5: он самодостаточен, а начиная с 3.11 бандл при запуске скачивает
зависимости через Maven, что внутри контейнера упирается в тот же TLS-прокси. Скачайте и распакуйте на хосте
(папка `gatling/bundle` в `.gitignore`), сверив SHA-1 с файлом `.sha1` рядом на Maven Central:

```bash
curl -o gatling.zip https://repo1.maven.org/maven2/io/gatling/highcharts/gatling-charts-highcharts-bundle/3.10.5/gatling-charts-highcharts-bundle-3.10.5-bundle.zip
sha1sum gatling.zip   # 8f8d02ba1828fc20e7500becbce56e5f1df40d81
unzip -q gatling.zip && mv gatling-charts-highcharts-bundle-3.10.5 docker/dev-stack/gatling/bundle
```

Симуляция `gatling/simulations/CheckoutSimulation.java` нагружает FastAPI-приложение: группа «Catalog», имена с
пробелами, запрос `/fail` с ошибкой и закрытый сценарий рядом с открытым. `gatling/gatling.conf` включает
Graphite-вывод в VictoriaMetrics.

Остановка: `docker compose --profile kafka --profile locust --profile load --profile gatling --profile arangodb --profile generator down -v`.

## Генератор нагрузки для страницы «Запуск теста»

Контейнер `generator` — SSH-точка с k6 0.54, Locust 2.31 и JDK 21 для бандла Gatling. Скрипты стенда смонтированы в
`/opt/loadlens/tests` (`k6/load.js`, `locust/locustfile.py`), симуляции Gatling — в бандл `/opt/gatling`. SSH-сервер —
`generator/devsshd.py` на paramiko: OpenSSH из apt за фильтрующим прокси не ставится, а LoadLens нужен только `exec`.
Вход — по ключу пользователя `loadlens`; ключ хоста лежит в томе `generator-hostkey` и переживает пересборку.

```bash
# ключ входа и колёса paramiko скачиваются на хосте (обе папки в .gitignore)
ssh-keygen -t ed25519 -N "" -C loadlens-dev -f docker/dev-stack/generator/keys/id_ed25519
cp docker/dev-stack/generator/keys/id_ed25519.pub docker/dev-stack/generator/keys/authorized_keys
python -m pip download --dest docker/dev-stack/generator/wheels --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 3.11 --implementation cp paramiko==5.0.0
cd docker/dev-stack
docker compose --profile gatling --profile generator up -d --build victoriametrics generator
```

В LoadLens добавьте генератор: адрес `127.0.0.1`, порт `12222`, пользователь `loadlens`, приватный ключ из
`generator/keys/id_ed25519`, рабочий каталог `/opt/loadlens/runs`, каталог скриптов `/opt/loadlens/tests`, для Gatling —
каталог скриптов `/opt/gatling/user-files/simulations`. Нажмите «Проверить» и сохраните, чтобы закрепить ключ хоста.
Инструменты внутри контейнера обращаются к сервисам стенда по именам: `http://pyapp:8000`,
`http://prometheus:9090/api/v1/write`, `http://influxdb:8086/k6`.

| Инстанс | Скрипт | Параметры и переменные |
|---|---|---|
| k6 в Prometheus | `k6/load.js` | `--out experimental-prometheus-rw`; `K6_PROMETHEUS_RW_SERVER_URL=http://prometheus:9090/api/v1/write`, `K6_PROMETHEUS_RW_TREND_STATS=p(95),p(99),avg`, `K6_DURATION_SCALE=0.5` |
| k6 с малым RPS | `k6/load.js` | `--vus 1 --duration 2m --rps 5 --out experimental-prometheus-rw` и те же переменные |
| Locust | `locust/locustfile.py` | `--host http://pyapp:8000 --users 5 --spawn-rate 5 --run-time 3m` |
| Gatling | `devstack.CheckoutSimulation` | `JAVA_OPTS=-DstepSeconds=40`: ступени по 40 с вместо 2 мин |

### Docker-генератор

`docker-proxy` — `tecnativa/docker-socket-proxy` перед Docker Desktop: LoadLens создаёт контейнеры инструментов через
него, а не через сокет. Скрипты стенда копирует в том `loadlens-dev-scripts` одноразовый сервис `scripts-volume`.

```bash
docker compose --profile generator up -d docker-proxy
docker compose --profile generator run --rm scripts-volume
```

Генератор в LoadLens: тип Docker, адрес `http://127.0.0.1:12375`, сеть `loadlens-dev-stack_default`, монтирование
`loadlens-dev-scripts:/scripts:ro`, каталог скриптов `/scripts`. Образы `grafana/k6:0.54.0` и `locustio/locust:2.31.8`
уже есть после запуска стенда; инструменты обращаются к сервисам по именам, как в таблице выше.

### Kubernetes-генератор на k3s

```bash
docker run -d --name ll-k3s --privileged --network loadlens-dev-stack_default -p 127.0.0.1:16443:6443 \
  rancher/k3s:v1.30.4-k3s1 server --disable traefik --disable metrics-server --tls-san 127.0.0.1
docker save grafana/k6:0.54.0 | docker exec -i ll-k3s ctr images import -
docker exec -i ll-k3s kubectl apply -f - < docker/dev-stack/k8s/loadlens.yaml   # пространство имён, Role, ConfigMap
docker exec ll-k3s kubectl -n loadtest create token loadlens --duration 24h          # токен для генератора
docker exec ll-k3s cat /etc/rancher/k3s/k3s.yaml                                       # certificate-authority-data — CA (base64)
```

Генератор в LoadLens: тип Kubernetes, адрес `https://127.0.0.1:16443`, токен и CA из команд выше, пространство имён
`loadtest`, ConfigMap `loadtest-scripts`, каталог скриптов `/scripts`. Поды не разрешают имена сервисов стенда
(DNS Docker им недоступен): скрипт `pyapp.js` берёт адрес из переменной `BASE_URL=http://<IP pyapp>:8000`, а remote
write k6 указывайте на IP Prometheus (`docker inspect` контейнеров стенда). Остановка: `docker rm -f ll-k3s`.

### GitLab CI и Jenkins

`ci/gitlab_stub.py` и `ci/jenkins_stub.py` отвечают на вызовы API, которые делает LoadLens: параметры пайплайна, ветки,
запуск, статус, журнал, отмена. «Пайплайн» — контейнер k6 стенда (`ci/stub_job.py`): 20 с подготовки, k6 с переданными
значениями (`VUS`/`users`, `DURATION`, `PROFILE` — 2, 5 или 10 RPS), по желанию тег прогона (`TAG_METRICS`) и события
(`SEND_EVENTS`), 15 с после теста. Подготовка и хвост нужны, чтобы видеть, как окно отчёта сужается по RPS.

```bash
python docker/dev-stack/ci/gitlab_stub.py                          # GitLab 17.11: GraphQL, spec:inputs
STUB_GITLAB_VERSION=15.4.0 python docker/dev-stack/ci/gitlab_stub.py  # старый GitLab: GraphQL по sha, без inputs
STUB_FORBID_VARIABLES=1 python docker/dev-stack/ci/gitlab_stub.py    # проект запрещает переменные пайплайна
python docker/dev-stack/ci/jenkins_stub.py
```

GitLab в LoadLens: адрес `http://127.0.0.1:5095`, проект `perf/load`, API-токен `api-secret` (без него — токен триггера
`trig-secret`), ветки `main` и `release`. Jenkins: адрес `http://127.0.0.1:5096`, задача `perf/load-test` (все типы
параметров, Active Choices) или multibranch `perf/load-mb`, пользователь `bot`, токен `JT`. Адрес LoadLens для событий —
`http://host.docker.internal:<порт LoadLens>`.

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

## Результат прогона 2026-10-14: Gatling → VictoriaMetrics

`lt-gatling-graphite-victoriametrics` вернул данные по всем девяти запросам на настоящем прогоне Gatling 3.10.5
(27 392 запроса, из них 7200 ошибочных на `/fail`) и отмечен `verified = true`. Общий RPS по минутам —
46 → 76 → 106 при расчётных 30 + 16, 60 + 16 и 90 + 16 (покупатели плюс закрытый сценарий); доля ошибок 21,8 / 26,4 /
28,4 % совпадает с 10 / 20 / 30 ошибочными запросами в секунду. Автоопределение по именам метрик VictoriaMetrics
нашло этот пакет единственным подходящим.

Что показал живой формат путей Graphite: имя симуляции в нижнем регистре, пробелы в именах запросов
заменяются на «_», группа — отдельный сегмент пути (`Catalog.Item_page`), счётчики `count` приходят каждую
секунду и в простое (нулями), а перцентили — только в секунды с ответами; у запроса без ошибок есть
`ko.count`, но нет перцентилей `ko`.

## Результат прогона 2026-10-17: Gatling → VictoriaMetrics с relabeling

`lt-gatling-graphite-victoriametrics-labels` вернул данные по всем девяти запросам на прогоне Gatling 3.10.5 со ступенями по
40 с (`-DstepSeconds=40`) в `victoriametrics-labels`. Общий RPS — 46 → 76 → 106, доля ошибок — 22 → 26 → 28 %, пользователи —
16 → 26 → 37, как у пакета без relabeling. С `-Dgatling.core.outputDirectoryBaseName=<тег>` метка `simulation` равна тегу:
фильтр прогона находит его ряды, а с чужим тегом рядов нет. Правило для групп даёт запросы `Catalog.Item_page`; без него
пути из шести частей остаются как есть. С правилом пользователей вида `gatling.users.*.*.*` (оно не срабатывает) пользователи
лежат в `gatling_request_active{request="users"}`, и пакет читает их оттуда.

## Результат прогона 2026-10-20: ArangoDB

`db-arangodb` вернул данные по всем 17 запросам на ArangoDB 3.12.12 (single server, лимит контейнера 2 CPU и 3 ГБ,
`ARANGODB_OVERRIDE_DETECTED_TOTAL_MEMORY` = 2 ГБ) под нагрузкой `arangobench`: около 1600 HTTP-запросов и 420 AQL-запросов
в секунду, из них около 9 медленнее порога 0,05 с. CPU процесса — 100 % от двух ядер, память — 32 % от 2 ГБ.
Пакет отмечен `verified = true`.

Что показал живой сервер: лимит контейнера виден только в `arangodb_server_statistics_effective_cpu_cores` и
`..._effective_physical_memory`, а `cpu_cores` и `physical_memory` показывают хост или значения
`ARANGODB_OVERRIDE_DETECTED_*`; поэтому пакет делит CPU и память на меньшее из двух. Без подмены памяти ArangoDB
рассчитывает кэши RocksDB от памяти хоста. Гистограмма `client_connection_statistics_total_time` начинается с 10 мс,
поэтому для HTTP-запросов в пакете только среднее время, а p95 считается по AQL (`aql_query_time`, бакеты от 95 мкс).
Нулями остались отклонённые запросы, прерванные транзакции и остановки записи RocksDB: стенд до них не доходит.

## Что стенд не покрывает

Kubernetes (cAdvisor, kube-state-metrics), JMX exporter и брокерные метрики Kafka, OpenTelemetry, Istio,
ingress-nginx, Flask и Django, HikariCP (у Keycloak пул Agroal), MySQL, MongoDB, Windows, JMeter, Gatling через
InfluxDB и Yandex.Tank. Пакеты для них проверены только синтаксически и на контракте рядов и единиц; в каталоге у них нет
отметки «проверен на стенде».
