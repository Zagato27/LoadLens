---

# Подсистема AI для отчётов по нагрузочному тестированию

Модуль `AI/` выполняет доменный анализ метрик (JVM/Database/Kafka/Microservices) и формирует строгий JSON‑ответ, который далее конвертируется в публикабельный markdown для Confluence. Итоговые блоки подставляются в плейсхолдеры `$$answer_*$$` и `$$final_answer$$/$$answer_llm$$`.

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

- **Доменный анализ**: `JVM`, `Database` (в т.ч. ArangoDB), `Kafka`, `Microservices` + общий итог.
- **Context pack**: компактные JSON‑сводки измерений (выбор top‑окон, пики, средние, пороги), вместо «сырых» временных рядов.
- **Двухпроходный инференс**: генерация нескольких кандидатов → критик приводит к строгому JSON → выбор лучшего по голосованию verdict и уверенности (self‑consistency, k=3).
- **Строгая валидация**: JSON приводится к модели `LLMAnalysis` со всеми обязательными полями; русская локализация текстовых полей.
- **Markdown‑рендер**: генерация человекочитаемых блоков, включая «Пиковую производительность», при наличии соответствующих данных [[memory:8657199]].

---

## Требования

- Python 3.12+
- Зависимости из корневого `requirements.txt` (включая `requests`, `pandas`, `atlassian-python-api`, `beautifulsoup4`, драйвер LLM `gigachat`).

> Убедитесь в наличии и заполненности `AI/config.py`.

---

## Конфигурация (`AI/config.py`)

- `prometheus.url` — источник метрик (или используйте `metrics_source.grafana_proxy`).
- `time_range.start_ts|end_ts` — примеры временных меток; в веб‑приложении берётся из POST‑тела.
- `llm.provider=gigachat` — прямые REST‑вызовы по mTLS:
  - `use_mtls`, `cert_file`, `key_file`, `verify`, `proxies`, таймауты.
  - `generation`: `temperature`, `top_p`, `max_tokens`, `force_json_in_prompt`.
- `default_params`: `step`, `resample_interval` — управление плотностью данных и ресемплированием.
- `metrics_source.{type,grafana}` — получение метрик напрямую из Prometheus или через Grafana‑прокси.
- `queries` — PromQL по доменам: список запросов, ключи меток и человекочитаемые ярлыки.

---

## Доменные промты (`AI/prompts/*.txt`)

- `jvm_prompt.txt` — сбор и интерпретация heap/non‑heap, GC, CPU, threads (включая peak threads), classes. Требование указывать `peak_time` и интервалы `start–end` в findings.
- `database_prompt.txt` и `arangodb_prompt.txt` — интенсивности запросов, p95/99 задержек, SLA, закономерности при пиках.
- `kafka_prompt.txt` — throughput, lag по группам/топикам/клиентам; обязательны интервалы и `peak_time`.
- `microservices_prompt.txt` — RPS по сервисам, пиковые значения, сравнение с пропускной способностью; допускается `peak_performance`.
- `overall_prompt.txt` — агрегирует доменные выводы, допускает `peak_performance` как часть общего итога.

---

## Как работает конвейер

Высокоуровнево конвейер реализован в `AI/main.py`:

1. Сбор данных: PromQL‑запросы по доменам → DataFrame → ресемплирование → выявление окон и пиков.
2. Формирование context pack по каждому домену (сжатая, но информативная сводка).
3. Инференс по доменам: `llm_two_pass_self_consistency(user_prompt, data_context, k=3)` возвращает `(text, parsed)`.
4. Общий итог: слияние доменных контекстов + общий промт → `(final_text, final_parsed)`.
5. Возврат результата в веб‑слой: сырые тексты и распарсенные структуры для `update_page.py`.

Ключевые функции и модели:
- `LLMAnalysis` — pydantic‑модель строгого ответа: `{ verdict, confidence, findings[], recommended_actions[], affected_components?, peak_performance? }`.
- `parse_llm_analysis_strict()` — извлечение/нормализация JSON в `LLMAnalysis`.
- `_format_parsed_as_text()` — человекочитаемый текст при необходимости фолбэка.
- `llm_two_pass_self_consistency()` — генерация кандидатов → строгий JSON критиком → выбор лучшего по majority verdict и confidence.

---

## Строгая схема JSON‑ответа

Обязательные поля:
- `verdict: string` — краткий вердикт по домену/в целом (на русском);
- `confidence: number [0..1]` — уверенность модели;
- `findings: (string | { summary, severity, component, evidence })[]` — список находок; для объектов обязательны `severity` и `component`, `evidence` должен содержать метрику и интервал `start–end` и `peak_time` при наличии;
- `recommended_actions: string[]` — конкретные действия;
- `affected_components?: string[]` — перечисление компонентов;
- `peak_performance?: { max_rps, max_time, drop_time, method }` — общий пик производительности, если применимо.

Все текстовые поля приводятся к русскому языку, ключи JSON остаются на английском. Отсутствующие не критичные поля нормализуются.

---

## Peak performance

Если LLM возвращает блок `peak_performance`, итоговый рендер (`render_llm_markdown`) выводит подраздел «Пиковая производительность» с полями:
- `Максимальный RPS (max_rps)`
- `Время пика (max_time)`
- `Время деградации (drop_time)`
- `Метод оценки (method)`

Это касается как общего блока (`$$answer_llm$$`/`$$final_answer$$`), так и доменных, если они содержат такой раздел [[memory:8657199]].

---

## Запуск анализа напрямую

```python
from AI.main import uploadFromLLM

# UNIX‑время в секундах
start_ts = 1740126600
end_ts = 1740136200

results = uploadFromLLM(start_ts, end_ts)
print(results.keys())  # jvm, database, kafka, ms, final, jvm_parsed, ..., final_parsed
```

Возвращается словарь c текстовыми и структурированными полями. Веб‑слой затем вызывает рендер и массовую подстановку в Confluence.

---

## Отладка и типичные проблемы

- Пустой `final_parsed`: проверьте доступность метрик и корректность интервала времени — конвейеру может не хватать данных для строгого JSON.
- Некорректные сертификаты GigaChat: задайте `verify` (`True`/`False`/путь к CA) и параметры mTLS (`cert_file`, `key_file`).
- Медленный ответ LLM: уменьшите `max_tokens`, понизьте `k` в `llm_two_pass_self_consistency`.
- Лишние/шумные findings: скорректируйте доменные промты и пороги аномалий в коде контекст‑паков.

Подробности по общей интеграции и REST‑слою смотрите в корневом `README.md`.

