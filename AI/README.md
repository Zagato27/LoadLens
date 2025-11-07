---

# Подсистема AI для отчётов по нагрузочному тестированию (домены + итог)

Модуль `AI/` выполняет доменный анализ метрик (JVM/Database/Kafka/Microservices/Hard Resources) и формирует строгий JSON‑ответ, который далее конвертируется в публикабельный markdown для Confluence. Итоговые блоки подставляются в плейсхолдеры доменов `$$answer_*$$` и финальный `$$final_answer$$`.

---

## Оглавление

- [Функциональные возможности](#функциональные-возможности)
- [Требования](#требования)
- [Конфигурация (`AI/config.py`)](#конфигурация-aiconfigpy)
- [Доменные промты (`AI/prompts/*.txt`)](#доменные-промты-aipromptstxt)
- [Как работает конвейер](#как-работает-конвейер)
- [Строгая схема JSON‑ответа](#строгая-схема-json-ответа)
- [Peak performance](#peak-performance)
- [Запуск анализа напрямую](#запуск-анализа-напрямую)
- [Отладка и типичные проблемы](#отладка-и-типичные-проблемы)

---

## Функциональные возможности

- **Доменный анализ**: `JVM`, `Database`, `Kafka`, `Microservices`, `Hard Resources`, `lt_framework` + общий итог.
- **Context pack**: компактные JSON‑сводки измерений (выбор top‑окон, пики, средние, пороги), вместо «сырых» временных рядов.
- **Двухпроходный инференс**: генерация нескольких кандидатов → критик приводит к строгому JSON → выбор лучшего с участием LLM‑судьи и программной проверки по данным (self‑consistency, k=3).
- **LLM‑судья + data‑score**: независимый судья оценивает `factual/completeness/specificity/overall`, а программная проверка сверяет численные утверждения с данными; итог — агрегатная метрика выбора. В веб‑интерфейсе «Доверие» для итогового блока отображается по `judge.overall`.
- **Строгая валидация**: JSON приводится к модели `LLMAnalysis` со всеми обязательными полями; русская локализация текстовых полей.
- **Markdown‑рендер**: генерация человекочитаемых блоков, включая «Пиковую производительность», при наличии `peak_performance` в JSON [[memory:8657199]].
- **Профили типа теста**: при создании отчёта профиль `test_type` (step/soak/spike/stress) автоматически внедряется в общий и доменные промпты.
- **Переопределение промптов по областям**: UI «Настройки» позволяет редактировать промпты по доменам в разрезе проектных областей (runtime‑оверрайды).

---

## Требования

- Python 3.12+
- Зависимости из корневого `requirements.txt` (включая `requests`, `pandas`, `atlassian-python-api`, `beautifulsoup4`).

> Убедитесь в наличии и заполненности `AI/config.py`.

---

## Конфигурация (`AI/config.py`)

- `prometheus.url` — источник метрик (или используйте `metrics_source.grafana_proxy`).
- `time_range.start_ts|end_ts` — примеры временных меток; в веб‑приложении берётся из POST‑тела.
- `llm.provider` — `perplexity` | `openai` | `anthropic`; настройте ключи/endpoint/модель в соответствующем разделе.
  - `generation`: `temperature`, `top_p`, `max_tokens`, `force_json_in_prompt`.
- `default_params`: `step`, `resample_interval` — управление плотностью данных и ресемплированием.
- `metrics_source.{type,grafana}` — получение метрик напрямую из Prometheus или через Grafana‑прокси.
- `lt_metrics_source.{type,prometheus,grafana,influxdb}` — источник метрик генератора нагрузки (Prometheus/InfluxDB напрямую или через Grafana‑прокси).
- `queries` — PromQL/Flux/InfluxQL по доменам: список запросов, ключи меток/тегов и человекочитаемые ярлыки.

---

## Доменные промты (`AI/prompts/*.txt`)

- `jvm_prompt.txt` — сбор и интерпретация heap/non‑heap, GC, CPU, threads (включая peak threads), classes. Требование указывать `peak_time` и интервалы `start–end` в findings.
- `database_prompt.txt` и `arangodb_prompt.txt` — интенсивности запросов, p95/99 задержек, SLA, закономерности при пиках.
- `kafka_prompt.txt` — throughput, lag по группам/топикам/клиентам; обязательны интервалы и `peak_time`.
- `microservices_prompt.txt` — RPS по сервисам, пиковые значения, сравнение с пропускной способностью; допускается `peak_performance`.
- `lt_framework_prompt.txt` — анализ метрик инструмента нагрузочного тестирования (RPS/checks/p95 и т.д.), допускается оценка `peak_performance`.
- `overall_prompt.txt` — агрегирует доменные выводы, учитывает `lt_framework` (плейсхолдер `{answer_lt_framework}`) и допускает `peak_performance` как часть общего итога.
- Во всех промтах и в fallback‑критике принудительно ограничено поле `verdict` значениями: `Успешно | Есть риски | Провал | Недостаточно данных`.

---

## Как работает конвейер

Высокоуровнево конвейер реализован в `AI/pipeline.py` (экспорт `uploadFromLLM` прокинут через `AI/main.py`):

1. Сбор данных: PromQL/Flux/InfluxQL по доменам → DataFrame → ресемплирование → выявление окон/пиков.
2. Формирование context pack по каждому домену (сжатая, но информативная сводка).
3. Инференс по доменам: `llm_two_pass_self_consistency(user_prompt, data_context, k=3)` возвращает `(text, parsed, score_info)`.
4. Общий итог: слияние доменных контекстов + общий промт → `(final_text, final_parsed, final_score)`.
5. Возврат результата в веб‑слой: тексты/структуры и `scores` для добавления раздела «Доверие (судья)».

Ключевые функции и модели:
- `LLMAnalysis` — pydantic‑модель строгого ответа: `{ verdict, confidence, findings[], recommended_actions[], affected_components?, peak_performance? }` (см. `AI/scoring.py`).
- `parse_llm_analysis_strict()` — извлечение/нормализация JSON в `LLMAnalysis`.
- `llm_two_pass_self_consistency()` — генерация кандидатов → строгий JSON критиком → судья + data‑score → выбор лучшего.

---

## Строгая схема JSON‑ответа

Обязательные поля:
- `verdict: string` — краткий вердикт по домену/в целом (на русском) из множества: `Успешно | Есть риски | Провал | Недостаточно данных`;
- `confidence: number [0..1]` — уверенность модели;
- `findings: (string | { summary, severity, component, evidence })[]` — список находок; для объектов обязательны `severity` и `component`, `evidence` должен содержать метрику и интервал `start–end` и `peak_time` при наличии;
- `recommended_actions: string[]` — конкретные действия;
- `affected_components?: string[]` — перечисление компонентов;
- `peak_performance?: { max_rps, max_time, drop_time, method }` — общий пик производительности, если применимо.

Все текстовые поля приводятся к русскому языку, ключи JSON остаются на английском. Отсутствующие не критичные поля нормализуются.

Примечание: хотя модель возвращает `confidence`, для UI используется итог судьи `scores.judge.overall` как показатель доверия.

---

## Peak performance

Если LLM возвращает блок `peak_performance`, итоговый рендер (`render_llm_markdown`) выводит подраздел «Пиковая производительность» с полями:
- `Максимальный RPS (max_rps)`
- `Время пика (max_time)`
- `Время деградации (drop_time)`
- `Метод оценки (method)`

Это касается как общего блока (`$$final_answer$$`), так и доменных, если они содержат такой раздел [[memory:8657199]].

---

## Запуск анализа напрямую

```python
from AI.main import uploadFromLLM

# UNIX‑время в секундах
start_ts = 1740126600
end_ts = 1740136200

results = uploadFromLLM(start_ts, end_ts)
print(results.keys())  # jvm, database, kafka, ms, hard_resources, final, *_parsed, scores
```

Возвращается словарь c текстовыми и структурированными полями. Веб‑слой затем вызывает рендер и массовую подстановку в Confluence или локальный рендер для LoadLens.

---

## Отладка и типичные проблемы

- Пустой `final_parsed`: проверьте доступность метрик и корректность интервала времени — конвейеру может не хватать данных для строгого JSON.
- Некорректные сертификаты GigaChat: задайте `verify` (`True`/`False`/путь к CA) и параметры mTLS (`cert_file`, `key_file`).
- Медленный ответ LLM: уменьшите `max_tokens`, понизьте `k` в `llm_two_pass_self_consistency`.
- Лишние/шумные findings: скорректируйте доменные промты и пороги аномалий в коде контекст‑паков.
- InfluxQL p95 пустой: конвейер выполняет принудительное приведение значений к числам (см. `AI/pipeline.py`), проверьте, что источник возвращает корректные типы.

Подробности по общей интеграции и REST‑слою смотрите в корневом `README.md`.

