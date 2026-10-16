# Подсистема ИИ (анализ результатов нагрузочного тестирования)

Модуль `AI/` собирает метрики по доменам, строит компактные “контекст‑пакеты” и выполняет многошаговый анализ LLM. Результаты возвращаются как текст (для отображения) и как строгие структуры (JSON) для программной обработки и сохранения.

## Возможности
- Домены анализа: JVM, базы данных, Kafka, микросервисы, инфраструктурные ресурсы; опционально — `lt_framework` (метрики инструмента нагрузки), `lt_generators` (ресурсы генераторов нагрузки) и `application_logs` (агрегированные ERROR‑логи из OpenSearch, включается `logs_source.enabled`).
- Контекст‑пакеты: резюме по сериям (средние/минимумы/максимумы/последние значения, время пиков, окна аномалий).
- Двухпроходный вывод: k кандидатов → приведение к строгому JSON → выбор лучшего по G-Eval-lite judge и верификации на данных.
- Вычисление “пиковой производительности” (peak_performance) в итоговом JSON при наличии соответствующих метрик; в остальных доменах `peak_performance` вырезается.
- Профиль типа теста (`AI/test_profile.py`): единый источник алиасов step/soak/spike/stress, режима анализа (`capacity | stability | spike | stress`), применимости peak_performance, детектора ступеней и прогноза. Его используют SLA, промпты (`AI/prompt_assembly.py`), прогноз, baseline и рендеринг.
- Детерминированный анализ по типу теста: `AI/stability.py` (окно удержания и тренды soak), `AI/spike.py` (окна всплесков, overshoot, восстановление), `AI/saturation.py` (точка насыщения stress). Секции для Confluence строит `AI/report_sections.py`, web-страница повторяет их в `static/js/reports.js`.
- Прогресс по фазам: `uploadFromLLM(..., progress_callback=fn)` вызывает `fn(message, percent)` при сборе каждого домена, после каждого ответа LLM, при проверке SLA, итоговом анализе и сохранении (диапазон 10–95 %).
- Домены, упавшие в параллельном прогоне, повторяются последовательно с паузой (частая причина — burst запросов к провайдеру).
- Пакеты запросов (`AI/query_packs`): каталог TOML-файлов с запросами, ролями, единицами и переменными, рендер и слияние в блок домена. Пайплайн использует их метаданные (см. ниже), а применение из настроек выполняет `loadlens_app/query_pack_service.py`.

## Пакеты запросов в анализе
- `roles` блока домена ({подпись → роль}) выбирают ряды CPU и памяти рантайма для `aux_resources` микросервисов: `runtime.cpu`, `runtime.heap_used`, `runtime.heap_max`, `runtime.rss`, `runtime.memory_limit`. Без ролей работают прежние подписи Micrometer («Process CPU usage», «Heap used/max»).
- Секции контекста получают `meaning`, `unit` и `description` запроса пакета; числа секции выражены в `unit`. Каталог чисел для проверки находок их не читает.
- `prompt_addendum` установленных пакетов добавляется в ориентиры анализа домена блоком «Особенности метрик стека»: так модель узнаёт, что в домене «JVM» лежат, например, метрики Go.
- Реестр ролей — `AI/query_packs/roles.py`: домен, единица, одна серия или много и поле SLA, которое роль заполняет. Пакеты нагрузочных инструментов отдают задержки в мс, CPU узлов — в %, CPU экземпляра — долей 0..1.
- Пустые ссылки SLA (`max_performance_query`, `p95_query`, `service_cpu_query` и другие) пайплайн и прогноз берут по ролям запросов (`resolve_sla_refs`); явная ссылка на существующий запрос важнее, ссылка на отсутствующий запрос заменяется ролью с предупреждением в логе. `latency_unit` следует единице роли, только если ни одна живая ссылка на задержки не задана явно.
- Итоговый блок отчёта получает `domain_titles` — заголовки доменов, отличные от стандартных: поле `title` блока домена или `domain_title` установленных пакетов («Go runtime», «JVM · Контейнеры»). Страница отчёта, сравнение и Confluence берут их из отчёта, поэтому старые отчёты не меняются.
- CLI: `python -m AI.query_packs list | render <id> | bundle <id> | docs <файл> | example`.

## Как это работает (поток)
1) Сбор данных за интервал (PromQL/Flux/InfluxQL; напрямую или через Grafana‑proxy).  
2) Построение `context pack` по каждому домену.  
3) Генерация нескольких кандидатов ответов LLM и приведение к строгому JSON.  
4) Оценка кандидатов G-Eval-lite judge (`AI/prompts/judge_prompt.txt` + `AI/prompts/judge_rubrics/*.txt`) и проверкой на данных; выбор лучшего.  
5) Возврат: доменные текстовые блоки, общий итог, разобранные структуры (`*_parsed`) и метрики качества (`scores`).

Основная функция: `AI.pipeline.uploadFromLLM(start_ts, end_ts, ...)` (есть прокси в `AI.main.uploadFromLLM`).

## Настройка
Источник конфигурации — `settings.py`:
- `llm` — провайдер (`perplexity|openai|anthropic|gigachat`), модель, лимиты токенов и таймауты; `max_domain_workers` и `self_consistency_k` задают параллельность доменов и число кандидатов, `self_consistency` — троттлинг вызовов внутри домена (последовательный режим с паузой для провайдеров с rate limit), `<provider>.max_attempts` — число попыток одного вызова;
- `data_sources` — каталог подключений (`prometheus|grafana_proxy|influxdb`);
- `domain_sources` — привязка домена к источнику (`default` или имя домена; датасорс, database, bucket);
- `logs_source` — OpenSearch Dashboards для домена `application_logs` (индекс, поля, фильтры уровней, лимиты агрегации);
- `default_params` — шаг выборки и ресемплирование;
- `queries` — список запросов/разметки серий для каждого домена (PromQL/Flux/InfluxQL).
GigaChat подключается через `langchain_gigachat` с mTLS (`cert_file`/`key_file`) или `api_key`; отдельный CLI `opensearch_error_report.py` строит markdown‑отчёт по ERROR‑логам тем же агрегатором, что и домен `application_logs`.
Для точечных изменений без перезапуска предназначен `settings_runtime.json` (в т.ч. по проектным областям: `per_area[<service>]`).  
Тексты промптов лежат в `AI/prompts/*.txt`, judge-рубрики — в `AI/prompts/judge_rubrics/*.txt`; `context_import_prompt.txt` переносит документ в `system_context` (`loadlens_app/context_import.py`, в интерфейсе не редактируется); всё это также может переопределяться через рантайм‑конфиг.

## Judge Layer
- Judge использует компактную form-filling схему: `scores[].rubric` плюс совместимые поля `factual`, `completeness`, `specificity`, `overall`.
- Доменный `domain_key` выбирает статическую rubric file (`jvm`, `database`, `kafka`, `microservices`, `hard_resources`, `lt_framework`, `lt_generators`, `application_logs`, `final`).
- Итоговый выбор остаётся гибридным: `0.6 * judge_overall + 0.35 * score_candidate_by_data() + 0.05 * confidence`.
- При одном кандидате judge не вызывается (выбирать не из чего): в `scores` сохраняются `data_score`, `data_score_details` и `confidence`, `judge_meta.skipped = true`; отчёт показывает уровень достоверности по проверке данных.
- Если judge упал или оценил не всех кандидатов, его отсутствие не считается нулём: кандидаты ранжируются по `(0.35 * data + 0.05 * confidence) / 0.4`, в `judge_meta` пишутся `unavailable = true` и `reason` (`error`, `no_scores`, `partial_scores`), а ключа `judge` нет. Разбор ответа judge принимает список в тексте или блоке кода, ключи `scores`/`results`/`candidates`, словарь по номерам и нумерацию с 1.
- Judge не урезает `data_context`; для наблюдаемости он только логирует размер prompt и judge latency.

## Формат результата (строгая часть)
Каждый домен и итоговый блок возвращают объект со следующими полями:
- `verdict` — один из: «Успешно», «Есть риски», «Провал», «Недостаточно данных»;
- `verdict_rationale?` — краткое объяснение, почему вынесен именно этот вердикт; рекомендуется формат: короткий вывод + 2-4 ключевых фактора;
- `confidence` — уверенность модели в диапазоне [0..1];
- `findings` — список строк или объектов `{summary, severity, component, evidence}`;
- `recommended_actions` — список конкретных действий;
- `affected_components?` — необязательный список компонент;
- `peak_performance?` — необязательный блок `{max_rps, max_time, drop_time, method}` при наличии входных метрик общего RPS; для soak и spike — `{not_applicable: true, reason}`.

Итоговый блок дополнительно содержит поля, которые пайплайн вычисляет сам, а не модель:
- `test_profile` — профиль типа теста (`test_type`, `normalized_test_type`, `mode`, `label_ru`, `focus`, …);
- `deterministic_sla` — программная проверка SLA (`verdict`, `checks[]`, `stable_window` с `kind`: ступень, `hold` или `spike`);
- `stability_under_load` (soak) — `target_rps`, `actual_rps`, `hold` (окно удержания, минимальный интервал), `trends[]`, `flagged_trends[]`;
- `spike_analysis` (spike) — `detected`, `baseline_rps`, `spikes[]` с `pre`/`during`, `overshoot_p95_pct`, `recovered`, `recovery_time_sec`;
- `saturation` (stress) — `saturation_rps`, ступень насыщения, `degraded_checks`, `failure_mode`, `limiting_resource`.

Политика вердикта по режимам описана в docstring `AI/sla_evaluator.py` и в разделе «Типы тестов» корневого README.

## Пример прямого вызова
```python
from AI.main import uploadFromLLM

start_ts = 1740126600  # UNIX-время (сек)
end_ts = 1740136200

res = uploadFromLLM(start_ts, end_ts, save_to_db=False)
print(res.keys())  # jvm, database, kafka, ms, hard_resources, lt_framework?, lt_generators?, final, *_parsed, scores
```

## Отладка
- Пустые доменные блоки — проверьте доступность источников метрик и корректность временного интервала.
- Значения p95 для InfluxQL пустые — проверьте типы колонок и корректность запросов.
- Медленный ответ LLM — уменьшите `max_tokens`/параллельность, проверьте сеть и параметры провайдера.

Подробности по REST‑эндпоинтам и UI см. в корневом `README.md`.

