# main.py

import requests
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
import json
import os
from datetime import datetime
import logging
import threading
import socket
import time
import re
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field, ValidationError, root_validator


# Unified settings
from settings import CONFIG
from AI.db_store import save_domain_labeled

logger = logging.getLogger(__name__)
_llm_env_init_lock = threading.Lock()
_llm_env_applied = False
_llm_provider_name = (CONFIG.get("llm", {}) or {}).get("provider", "perplexity").lower()
_llm_provider_cfg = (CONFIG.get("llm", {}) or {}).get(_llm_provider_name, {})
_llm_semaphore = threading.Semaphore(int(_llm_provider_cfg.get("max_concurrent", 4)))
_llm_preflight_last_ts = 0.0

# httpx/requests таймаут через окружение (общий для LLM)
os.environ.setdefault("PPLX_TIMEOUT", str(CONFIG.get("llm", {}).get("perplexity", {}).get("request_timeout_sec", 120)))


def _normalize_llm_base_url(raw_url: str | None) -> str:
    # Универсальная нормализация базового URL для LLM API
    base = (raw_url or "https://api.perplexity.ai").strip()
    if not base:
        return "https://api.perplexity.ai"
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _ensure_llm_network_env(gcfg: dict) -> None:
    """Применяет сетевые настройки (прокси/CA/инsecure) через переменные окружения."""
    proxies = (gcfg or {}).get("proxies", {}) or {}
    ca_bundle = (gcfg or {}).get("ca_bundle")
    insecure = bool((gcfg or {}).get("insecure_skip_verify", False))

    https_proxy = proxies.get("https") or proxies.get("HTTPS")
    http_proxy = proxies.get("http") or proxies.get("HTTP")

    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy

    if ca_bundle and not insecure:
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle

    if insecure:
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        os.environ.pop("REQUESTS_CA_BUNDLE", None)
        os.environ.pop("SSL_CERT_FILE", None)

    logger.info(
        f"LLM net env set: HTTPS_PROXY={'set' if https_proxy else 'unset'} "
        f"HTTP_PROXY={'set' if http_proxy else 'unset'} CA_BUNDLE={'set' if (ca_bundle and not insecure) else 'unset'} "
        f"INSECURE={'on' if insecure else 'off'}"
    )


def _llm_preflight(gcfg: dict) -> None:
    """Заглушка префлайта (для Perplexity не требуется отдельная проверка)."""
    try:
        api_base = _normalize_llm_base_url(gcfg.get("api_base_url") or gcfg.get("base_url"))
        parsed = urlparse(api_base)
        host = parsed.hostname or "api.perplexity.ai"
        port = parsed.port or 443
        with socket.create_connection((host, port), timeout=float((gcfg or {}).get("connect_timeout_sec") or 5)):
            logger.info(f"Preflight ok: api_host {host}:{port}")
    except Exception as e:
        logger.warning(f"Preflight skipped/failed: {e}")


def _strip_think(text: str) -> str:
    """Удаляет блоки <think>...</think> из ответа модели."""
    if not isinstance(text, str) or not text:
        return text
    try:
        return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    except Exception:
        return text


def _configure_logging():
    level_name = (CONFIG.get("logging", {}).get("level") if CONFIG.get("logging") else "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger().setLevel(level)


# Вспомогательная функция для конвертации времени
def convert_to_timestamp(date_str: str) -> int:
    """Конвертирует строку даты и времени в миллисекундный timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
    return int(dt.timestamp())

def parse_step_to_seconds(step: str) -> int:
    """Преобразует шаг '1m', '30s' в целые секунды."""
    if step.endswith('m'):
        return int(step[:-1]) * 60
    elif step.endswith('s'):
        return int(step[:-1])
    else:
        return int(step)


PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_PROMPT_CACHE: Dict[str, str] = {}


def read_prompt_from_file(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


CRITIC_PROMPT_FALLBACK = (
    "Вы выступаете как строгий валидатор отчёта. Отвечайте на русском языке. "
    "Перефразируйте все ТЕКСТОВЫЕ поля на русский язык (verdict, findings.summary, findings.evidence, recommended_actions, affected_components). "
    "Ключи JSON и значения поля severity оставьте на английском согласно схеме. "
    "Ниже дан проект ответа. Исправьте/нормализуйте его до СТРОГОГО JSON со схемой: "
    "{verdict, confidence, findings[], recommended_actions[]}. Каждый элемент findings обязан содержать severity (critical|high|medium|low) и component. "
    "Если component не указан — извлеките его из evidence по лейблам application|service|job|pod|instance, иначе 'unknown'. "
    "Если severity отсутствует — используйте 'low'. Дополнительно допускается поле peak_performance: {max_rps, max_time, drop_time, method}. "
    "Никакого текста вне JSON. Если данных недостаточно — верните verdict='insufficient_data'.\n\nПроект ответа:\n{{CANDIDATE}}"
)


JUDGE_PROMPT_FALLBACK = (
    "Вы выступаете как независимый арбитр отчётов по нагрузочному тестированию. "
    "У вас есть агрегированные данные теста и несколько кандидатов ответов модели (каждый в JSON). "
    "Для каждого кандидата оцените три аспекта (0..1) и общий балл: factual, completeness, specificity. "
    "Рассчитайте overall = 0.5*factual + 0.3*completeness + 0.2*specificity. "
    "Ответьте СТРОГО JSON формата {\"scores\": [{\"index\": int, \"factual\": float, \"completeness\": float, \"specificity\": float, \"overall\": float}, ...]}. "
    "Если данных недостаточно для оценки, укажите 0. Контекст приведён ниже.\n\nКонтекст:\n{{DATA_CONTEXT}}\n\nКандидаты:\n{{CANDIDATES_JSON}}"
)


JUDGE_SYSTEM_PROMPT = (
    "Вы опытный инженер по нагрузочному тестированию и выступаете независимым судьёй. "
    "Используйте предоставленный контекст метрик, чтобы беспристрастно оценить кандидатов. "
    "Верните только JSON согласно запросу."
)


def _get_prompt_template(filename: str, fallback: str) -> str:
    cache_key = filename
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        text = read_prompt_from_file(path)
    except Exception as e:
        logger.warning(f"Не удалось загрузить шаблон {filename}: {e}. Использую fallback.")
        text = fallback
    _PROMPT_CACHE[cache_key] = text
    return text


def fetch_prometheus_data(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    """
    Запрашивает PromQL-запрос у Prometheus за интервал [start_ts, end_ts].
    Возвращает сырые данные в формате JSON.
    """
    step_in_seconds = parse_step_to_seconds(step)

    params = {
        'query': promql_query,
        'start': start_ts,
        'end':   end_ts,
        'step':  step_in_seconds
    }

    url = f'{prometheus_url}/api/v1/query_range'
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _resolve_grafana_prom_ds_id(g_cfg: dict) -> int:
    base_url = g_cfg["base_url"].rstrip("/")
    ds_cfg = g_cfg.get("prometheus_datasource", {})
    auth_cfg = g_cfg.get("auth", {})

    headers = {}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))

    verify = g_cfg.get("verify_ssl", True)

    if isinstance(ds_cfg.get("id"), int):
        logger.info(f"Grafana datasource id (configured): {ds_cfg['id']}")
        return ds_cfg["id"]

    if ds_cfg.get("uid"):
        url = f"{base_url}/api/datasources/uid/{ds_cfg['uid']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]

    if ds_cfg.get("name"):
        url = f"{base_url}/api/datasources/name/{ds_cfg['name']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]

    url = f"{base_url}/api/datasources"
    resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
    resp.raise_for_status()
    for ds in resp.json():
        if ds.get("type") == "prometheus":
            return ds["id"]
    raise RuntimeError("Не найден Prometheus datasource в Grafana")


def fetch_prometheus_data_via_grafana(
    g_cfg: dict,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    step_in_seconds = parse_step_to_seconds(step)
    base_url = g_cfg["base_url"].rstrip("/")

    ds_id = _resolve_grafana_prom_ds_id(g_cfg)

    params = {
        'query': promql_query,
        'start': start_ts,
        'end':   end_ts,
        'step':  step_in_seconds
    }

    headers = {}
    auth = None
    auth_cfg = g_cfg.get("auth", {})
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))

    url = f"{base_url}/api/datasources/proxy/{ds_id}/api/v1/query_range"
    resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=30, verify=g_cfg.get("verify_ssl", True))
    resp.raise_for_status()
    return resp.json()


def fetch_metric_series(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_query: str,
    step: str
) -> dict:
    src = (CONFIG.get("metrics_source", {}).get("type") or "prometheus").lower()
    if src == "grafana_proxy":
        g_cfg = CONFIG.get("metrics_source", {}).get("grafana", {})
        return fetch_prometheus_data_via_grafana(g_cfg, start_ts, end_ts, promql_query, step)
    else:
        return fetch_prometheus_data(prometheus_url, start_ts, end_ts, promql_query, step)


def fetch_and_aggregate_with_label_keys(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_queries: List[str],
    label_keys_list: List[List[str]],
    step: str,
    resample_interval: str
) -> List[pd.DataFrame]:
    if len(promql_queries) != len(label_keys_list):
        raise ValueError(
            "Количество запросов (promql_queries) и количество списков лейблов (label_keys_list) не совпадает!"
        )

    dfs = []
    for query, keys_for_this_query in zip(promql_queries, label_keys_list):
        data_json = fetch_metric_series(
            prometheus_url,
            start_ts,
            end_ts,
            query,
            step
        )

        records = []
        if data_json.get("status") == "success":
            result = data_json["data"].get("result", [])
            for series in result:
                lbls = series.get("metric", {})

                label_parts = []
                for key in keys_for_this_query:
                    val = lbls.get(key, "unknown")
                    label_parts.append(f"{key}={val}")
                label_str = "|".join(label_parts)

                for (ts_float, value_str) in series["values"]:
                    val = float(value_str)
                    records.append([ts_float, label_str, val])

        if not records:
            dfs.append(pd.DataFrame())
            continue

        df = pd.DataFrame(records, columns=["timestamp", "label", "value"])
        df["time"] = pd.to_datetime(df["timestamp"], unit='s')
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-3')
        df.set_index("time", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)

        pivoted = df.pivot_table(
            index=df.index,
            columns="label",
            values="value",
            aggfunc="sum"
        )

        pivoted_resampled = pivoted.resample(resample_interval).mean()
        dfs.append(pivoted_resampled)

    return dfs


def dataframes_to_markdown(labeled_dfs: List[Dict[str, object]]) -> str:
    result = []
    for item in labeled_dfs:
        label = item['label']
        df = item['df'].copy()
        result.append(f"## {label}\n")
        result.append("### Топ-10 сервисов по среднему значению\n")
        if not df.empty and df.shape[0] > 0:
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                column_means = df[numeric_columns].mean()
                sorted_numeric_columns = column_means.sort_values(ascending=False).index.tolist()
                non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
                sorted_columns = sorted_numeric_columns + non_numeric_columns
            else:
                sorted_columns = df.columns.tolist()
            top_columns = sorted_columns[:min(10, len(sorted_columns))]
            df = df[top_columns]
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.replace('|', '/')
            for col in df.select_dtypes(include=['number']).columns:
                max_val = df[col].abs().max()
                if max_val >= 1e6:
                    df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
                elif max_val >= 1000:
                    df[col] = df[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) else "")
                else:
                    df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
        df_transposed = df.T
        df_transposed.index = df_transposed.index.map(lambda x: str(x).replace('|', '/'))
        if hasattr(df_transposed, 'columns'):
            df_transposed.columns = df_transposed.columns.map(lambda x: str(x).replace('|', '/'))
        result.append(df_transposed.to_markdown() + "\n\n")
    return "\n".join(result)


def _summarize_time_series_dataframe(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, object]]:
    """Возвращает компактное резюме по колонкам (сериям) DataFrame:
    - series: имя серии (лейблы)
    - mean/min/max/last: агрегаты по времени
    Рекомендуется подавать сюда уже ресемплированный по времени pivot DataFrame.
    """
    summary: List[Dict[str, object]] = []
    if df is None or df.empty:
        return summary

    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) == 0:
        return summary

    # сортируем по среднему за окно, берём Топ-N
    column_means = df[numeric_columns].mean(numeric_only=True)
    top_columns = (
        column_means.sort_values(ascending=False)
        .head(max(1, int(top_n)))
        .index
        .tolist()
    )

    for col in top_columns:
        col_series = df[col]
        if col_series.dropna().empty:
            continue
        try:
            # Вспомогательные экстремумы/времена
            max_val = float(col_series.max(skipna=True))
            min_val = float(col_series.min(skipna=True))
            max_idx = col_series.idxmax()
            min_idx = col_series.idxmin()
            series_summary = {
                "series": str(col),
                "mean": float(col_series.mean(skipna=True)),
                "min": min_val,
                "max": max_val,
                "last": float(col_series.dropna().iloc[-1]),
                "max_time": str(max_idx) if pd.notnull(max_idx) else None,
                "min_time": str(min_idx) if pd.notnull(min_idx) else None,
            }
        except Exception:
            # в редких случаях встречаются нечисловые значения/пустые ряды
            continue
        summary.append(series_summary)

    return summary


def build_context_pack(labeled_dfs: List[Dict[str, object]], top_n: int = 10) -> Dict[str, object]:
    """Строит компактный JSON-"context pack" по списку {label, df}.
    Формат:
    {
      "sections": [
        { "label": "...", "top_series": [{series, mean, min, max, last}, ...] },
        ...
      ]
    }
    """
    def _detect_anomaly_windows(col_series: pd.Series, sigma: float = 2.0, max_windows: int = 2) -> List[Dict[str, object]]:
        windows: List[Dict[str, object]] = []
        try:
            s = col_series.dropna()
            if s.empty:
                return windows
            mu = float(s.mean())
            sd = float(s.std(ddof=0))
            if sd == 0 or not pd.notnull(sd):
                return windows
            thr = mu + sigma * sd
            mask = (col_series > thr).fillna(False)
            # Поиск контуров True-участков
            shifted = mask.astype(int).diff().fillna(int(mask.iloc[0]))
            starts = list(mask.index[shifted == 1])
            # если начинается с True
            if mask.iloc[0]:
                starts = [mask.index[0]] + starts
            ends = list(mask.index[shifted == -1])
            if mask.iloc[-1]:
                ends = ends + [mask.index[-1]]
            for st, en in zip(starts, ends):
                window_slice = col_series.loc[st:en].dropna()
                if window_slice.empty:
                    continue
                peak_val = float(window_slice.max())
                peak_ts = window_slice.idxmax()
                windows.append({
                    "start": str(st),
                    "end": str(en),
                    "peak_time": str(peak_ts),
                    "peak": peak_val,
                    "mean": mu,
                    "threshold_high": thr
                })
            # ограничим количеством окон
            if len(windows) > max_windows:
                windows = sorted(windows, key=lambda w: w.get("peak", 0.0), reverse=True)[:max_windows]
        except Exception:
            return []
        return windows

    sections = []
    for item in labeled_dfs:
        label = item.get("label")
        df = item.get("df")
        section_summary = _summarize_time_series_dataframe(df, top_n=top_n)
        # Добавим окна аномалий для отобранных серий
        anomalies: List[Dict[str, object]] = []
        if isinstance(df, pd.DataFrame) and not df.empty:
            for s in section_summary:
                series_name = s.get("series")
                if series_name in df.columns:
                    windows = _detect_anomaly_windows(df[series_name])
                    if windows:
                        anomalies.append({
                            "series": series_name,
                            "windows": windows
                        })
        sections.append({
            "label": label,
            "top_series": section_summary,
            "anomalies": anomalies
        })
    return {"sections": sections}


# ======== Pydantic схемы ответа LLM (строгий парсинг с фолбэками) ========

class FindingItem(BaseModel):
    summary: str = Field(default="")
    severity: Optional[str] = Field(default=None)  # critical|high|medium|low
    component: Optional[str] = Field(default=None)
    evidence: Optional[str] = Field(default=None)  # факт/метрика/окно времени


class PeakPerformance(BaseModel):
    max_rps: Optional[float] = Field(default=None)
    max_time: Optional[str] = Field(default=None)
    drop_time: Optional[str] = Field(default=None)
    method: Optional[str] = Field(default=None)


class LLMAnalysis(BaseModel):
    verdict: str = Field(default="нет данных")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    findings: List[Union[str, FindingItem]] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    affected_components: Optional[List[str]] = Field(default=None)
    peak_performance: Optional[PeakPerformance] = Field(default=None)

    @root_validator(pre=True)
    def _normalize_fields(cls, values: Dict[str, object]) -> Dict[str, object]:
        # alias: actions -> recommended_actions
        actions = values.get("recommended_actions") or values.get("actions") or []
        values["recommended_actions"] = actions
        # findings может быть строкой, списком строк или списком объектов
        findings = values.get("findings")
        if findings is None:
            values["findings"] = []
        return values


def _extract_json_like(text: str) -> Optional[dict]:
    """Пытается вытащить JSON-объект из текста (включая случаи с пояснениями до/после)."""
    if not text:
        return None
    # Удаляем служебные блоки рассуждений (<think>..</think>) если присутствуют
    try:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    except Exception:
        pass
    # Быстрый поиск первого '{' и последней '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Поиск в кодовых блоках ```json ... ```
    fence = "```"
    if fence in text:
        parts = text.split(fence)
        for i in range(len(parts) - 1):
            block = parts[i + 1]
            if block.strip().startswith("json"):
                block_text = block.strip()[len("json"):].strip()
            else:
                block_text = block
            try:
                return json.loads(block_text)
            except Exception:
                continue
    return None


def parse_llm_analysis_strict(raw_text: str) -> Optional[LLMAnalysis]:
    """Строгий парсинг JSON-ответа модели в LLMAnalysis.
    1) Пробуем распарсить как JSON целиком
    2) Пробуем выделить JSON-блок из свободного текста
    3) Если не получилось — возвращаем None (в отчёт можно подставить фолбэк)
    """
    if not raw_text:
        return None
    try:
        maybe_json = json.loads(raw_text)
    except Exception:
        maybe_json = _extract_json_like(raw_text)

    if maybe_json is None:
        return None

    try:
        return LLMAnalysis.parse_obj(maybe_json)
    except ValidationError:
        return None


def _build_domain_data(
    domain_key: str,
    domain_conf: dict,
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    step: str,
    resample: str,
    top_n: int
) -> Dict[str, object]:
    dfs = fetch_and_aggregate_with_label_keys(
        prometheus_url,
        start_ts,
        end_ts,
        domain_conf["promql_queries"],
        domain_conf["label_keys_list"],
        step=step,
        resample_interval=resample
    )
    labeled = label_dataframes(dfs, domain_conf["labels"])
    markdown = dataframes_to_markdown(labeled)
    pack = build_context_pack(labeled, top_n=top_n)
    ctx = json.dumps({
        "domain": domain_key,
        "time_range": {"start": start_ts, "end": end_ts},
        **pack
    }, ensure_ascii=False)
    return {"labeled": labeled, "markdown": markdown, "pack": pack, "ctx": ctx}


def _ask_domain_analysis(prompt_text: str, data_context: str) -> tuple[str, Optional[LLMAnalysis], Dict[str, Any]]:
    return llm_two_pass_self_consistency(user_prompt=prompt_text, data_context=data_context, k=3, return_scores=True)


def _build_critic_prompt(candidate_text: str) -> str:
    """Строит промпт-критику для исправления ответа в строгий JSON по схеме."""
    template = _get_prompt_template("critic_prompt.txt", CRITIC_PROMPT_FALLBACK)
    return template.replace("{{CANDIDATE}}", candidate_text)


def _choose_best_candidate(candidates: list) -> tuple[str, Optional[LLMAnalysis]]:
    """Выбирает лучший из [(text, parsed)] по голосованию verdict и максимальной confidence."""
    if not candidates:
        return "", None
    # Счётчик по verdict
    from collections import Counter
    parsed_list = [p for (_, p) in candidates if p is not None]
    if not parsed_list:
        return candidates[0]
    verdicts = [p.verdict for p in parsed_list if p.verdict]
    majority_verdict = Counter(verdicts).most_common(1)[0][0] if verdicts else None

    def conf_val(p: Optional[LLMAnalysis]) -> float:
        if p is None or p.confidence is None:
            return 0.0
        try:
            return float(p.confidence)
        except Exception:
            return 0.0

    filtered = [(t, p) for (t, p) in candidates if p is not None and p.verdict == majority_verdict] if majority_verdict else []
    pool = filtered if filtered else candidates

    # Языковая эвристика: предпочесть кандидата с большей долей кириллицы в текстовых полях
    def _extract_text_for_lang_score(p: Optional[LLMAnalysis]) -> str:
        if p is None:
            return ""
        parts: list[str] = []
        try:
            if getattr(p, "verdict", None):
                parts.append(str(p.verdict))
            for f in (p.findings or []):
                if isinstance(f, dict):
                    for key in ("summary", "evidence", "component"):
                        val = f.get(key)
                        if isinstance(val, str) and val.strip():
                            parts.append(val)
                else:
                    s = str(f).strip()
                    if s:
                        parts.append(s)
            for a in (p.recommended_actions or []):
                s = str(a).strip()
                if s:
                    parts.append(s)
            if getattr(p, "affected_components", None):
                parts.extend([str(x) for x in p.affected_components if str(x).strip()])
        except Exception:
            pass
        return " \n".join(parts)

    def _russian_ratio(text: str) -> float:
        if not isinstance(text, str) or not text:
            return 0.0
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
        if not letters:
            return 0.0
        cyr = re.findall(r"[А-Яа-яЁё]", text)
        return float(len(cyr)) / float(len(letters))

    def lang_score(p: Optional[LLMAnalysis]) -> float:
        try:
            blob = _extract_text_for_lang_score(p)
            return _russian_ratio(blob)
        except Exception:
            return 0.0

    best = max(pool, key=lambda tp: (lang_score(tp[1]), conf_val(tp[1])))
    return best


def judge_candidates_with_llm(candidates_texts: List[str], data_context: str) -> Dict[int, Dict[str, float]]:
    if not candidates_texts:
        return {}
    template = _get_prompt_template("judge_prompt.txt", JUDGE_PROMPT_FALLBACK)
    candidates_payload = [{"index": idx, "text": text} for idx, text in enumerate(candidates_texts)]
    prompt_text = template.replace("{{CANDIDATES_JSON}}", json.dumps(candidates_payload, ensure_ascii=False))
    prompt_text = prompt_text.replace("{{DATA_CONTEXT}}", data_context or "нет данных")

    try:
        raw = ask_llm_with_text_data(
            user_prompt=prompt_text,
            data_context="",
            llm_config={"force_json": True},
            system_prompt=JUDGE_SYSTEM_PROMPT
        )
    except Exception as e:
        logger.warning(f"Не удалось получить оценку судьи LLM: {e}")
        return {}

    parsed = None
    try:
        parsed = _extract_json_like(raw) or json.loads(raw)
    except Exception:
        parsed = None

    if not isinstance(parsed, dict):
        return {}
    scores = parsed.get("scores")
    if not isinstance(scores, list):
        return {}

    result: Dict[int, Dict[str, float]] = {}
    for item in scores:
        if not isinstance(item, dict):
            continue
        idx_raw = item.get("index")
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        result[idx] = {
            "factual": float(item.get("factual", 0.0) or 0.0),
            "completeness": float(item.get("completeness", 0.0) or 0.0),
            "specificity": float(item.get("specificity", 0.0) or 0.0),
            "overall": float(item.get("overall", 0.0) or 0.0),
        }
    return result


def _extract_sections_from_context(ctx_obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(ctx_obj, dict):
        return []
    sections: List[Dict[str, Any]] = []
    if isinstance(ctx_obj.get("sections"), list):
        sections.extend(ctx_obj["sections"])
    domains = ctx_obj.get("domains")
    if isinstance(domains, dict):
        for val in domains.values():
            if isinstance(val, dict) and isinstance(val.get("sections"), list):
                sections.extend(val["sections"])
    return sections


def _collect_label_vocab(sections: List[Dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for section in sections:
        label = section.get("label")
        if label:
            labels.add(str(label).lower())
        for series in section.get("top_series", []) or []:
            series_name = series.get("series")
            if series_name:
                labels.add(str(series_name).lower())
    return labels


def _extract_peak_estimate(sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for section in sections:
        for series in section.get("top_series", []) or []:
            max_val = series.get("max")
            if max_val is None:
                continue
            try:
                max_float = float(max_val)
            except Exception:
                continue
            if best is None or max_float > best.get("max", float("-inf")):
                best = {
                    "max": max_float,
                    "max_time": series.get("max_time"),
                    "series": series.get("series")
                }
    return best


def _finding_matches_labels(finding: Any, labels: set[str]) -> bool:
    if not labels:
        return False
    text_parts: List[str] = []
    try:
        if isinstance(finding, FindingItem):
            text_parts.extend([
                getattr(finding, "summary", ""),
                getattr(finding, "component", ""),
                getattr(finding, "evidence", ""),
            ])
        elif isinstance(finding, dict):
            text_parts.extend([
                str(finding.get("summary", "")),
                str(finding.get("component", "")),
                str(finding.get("evidence", "")),
            ])
        else:
            text_parts.append(str(finding))
    except Exception:
        text_parts.append(str(finding))

    blob = " ".join([part for part in text_parts if isinstance(part, str)])
    blob_lower = blob.lower()
    return any(label in blob_lower for label in labels if label)


def score_candidate_by_data(parsed: Optional[LLMAnalysis], context_obj: Dict[str, Any]) -> float:
    if not isinstance(parsed, LLMAnalysis):
        return 0.0

    sections = _extract_sections_from_context(context_obj)
    labels = _collect_label_vocab(sections)
    score = 0.15  # базовый балл за структурированный ответ

    findings = parsed.findings or []
    if findings:
        matches = sum(1 for f in findings if _finding_matches_labels(f, labels))
        coverage = matches / max(len(findings), 1)
        score += 0.35 * max(0.0, min(coverage, 1.0))

    peak_estimate = _extract_peak_estimate(sections)
    peak = getattr(parsed, "peak_performance", None)
    if peak_estimate and peak and getattr(peak, "max_rps", None) is not None:
        try:
            claimed = float(peak.max_rps)
            actual = float(peak_estimate.get("max", 0.0))
            if actual > 0:
                rel_error = abs(claimed - actual) / max(actual, 1e-9)
                score += 0.35 * max(0.0, 1.0 - min(rel_error, 1.0))
        except Exception:
            pass

    actions = parsed.recommended_actions or []
    if actions:
        score += 0.15 * max(0.0, min(len(actions) / 3.0, 1.0))

    return max(0.0, min(score, 1.0))


def _select_best_candidate(
    candidates: List[Tuple[str, Optional[LLMAnalysis]]],
    data_context: str
) -> Tuple[str, Optional[LLMAnalysis], Dict[str, Any]]:
    if not candidates:
        return "", None, {}

    try:
        context_obj = json.loads(data_context) if data_context else {}
    except Exception:
        context_obj = {}

    try:
        judge_scores = judge_candidates_with_llm([text for (text, _) in candidates], data_context)
    except Exception as e:
        logger.warning(f"Judge scoring failed: {e}")
        judge_scores = {}

    scored: List[Tuple[float, int]] = []
    for idx, (text, parsed) in enumerate(candidates):
        judge_entry = judge_scores.get(idx) or judge_scores.get(str(idx)) or {}
        judge_overall = float(judge_entry.get("overall", 0.0) or 0.0)
        data_score = score_candidate_by_data(parsed, context_obj)
        conf = 0.0
        if isinstance(parsed, LLMAnalysis) and parsed.confidence is not None:
            try:
                conf = float(parsed.confidence)
            except Exception:
                conf = 0.0
        final_score = 0.6 * judge_overall + 0.35 * data_score + 0.05 * max(0.0, min(conf, 1.0))
        scored.append((final_score, idx))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_idx = scored[0][1]
        # Восстановим метрики для лучшего
        best_text, best_parsed = candidates[best_idx]
        judge_entry = judge_scores.get(best_idx) or judge_scores.get(str(best_idx)) or {}
        data_score_best = score_candidate_by_data(best_parsed, context_obj)
        conf_best = 0.0
        if isinstance(best_parsed, LLMAnalysis) and best_parsed.confidence is not None:
            try:
                conf_best = float(best_parsed.confidence)
            except Exception:
                conf_best = 0.0
        final_score_best = [s for s in scored if s[1] == best_idx][0][0]
        score_info = {
            "selected_index": best_idx,
            "judge": {
                "overall": float(judge_entry.get("overall", 0.0) or 0.0),
                "factual": float(judge_entry.get("factual", 0.0) or 0.0),
                "completeness": float(judge_entry.get("completeness", 0.0) or 0.0),
                "specificity": float(judge_entry.get("specificity", 0.0) or 0.0),
            },
            "data_score": float(data_score_best),
            "confidence": float(max(0.0, min(conf_best, 1.0))),
            "final_score": float(final_score_best),
        }
        return best_text, best_parsed, score_info

    # Фолбэк на старую стратегию выбора
    best_text, best_parsed = _choose_best_candidate(candidates)
    return best_text, best_parsed, {}


def _format_parsed_as_text(p: LLMAnalysis) -> str:
    """Простое текстовое представление LLMAnalysis для человека."""
    if p is None:
        return "нет данных"
    parts = []
    parts.append(f"Вердикт: {p.verdict}")
    parts.append(f"Доверие: {int(p.confidence*100)}%" if p.confidence is not None else "Доверие: —")
    if p.findings:
        lines = []
        for f in p.findings:
            if isinstance(f, dict):
                s = str(f.get("summary", "")).strip()
                sev = f.get("severity")
                comp = f.get("component")
                ev = f.get("evidence")
                frag = [s]
                if sev:
                    frag.append(f"[severity: {sev}]")
                if comp:
                    frag.append(f"[component: {comp}]")
                if ev:
                    frag.append(f"[evidence: {ev}]")
                s = " ".join([x for x in frag if x])
            else:
                s = str(f).strip()
            if s:
                lines.append(f"- {s}")
        if lines:
            parts.append("Выводы:\n" + "\n".join(lines))
    if p.recommended_actions:
        acts = [str(a).strip() for a in p.recommended_actions if str(a).strip()]
        if acts:
            parts.append("Рекомендации:\n" + "\n".join([f"- {a}" for a in acts]))
    # Пиковая производительность, если модель её вернула
    try:
        pp = getattr(p, "peak_performance", None)
        if pp:
            max_rps = getattr(pp, "max_rps", None)
            max_time = getattr(pp, "max_time", None) or "—"
            drop_time = getattr(pp, "drop_time", None) or "—"
            method = getattr(pp, "method", None)
            if isinstance(max_rps, (int, float)):
                rps_str = f"{max_rps:.0f}"
            else:
                rps_str = "—"
            line = f"Максимальная производительность: {rps_str} rps, время пика: {max_time}, падение: {drop_time}"
            if isinstance(method, str) and method.strip():
                line += f", метод: {method.strip()}"
            parts.append(line)
    except Exception:
        pass
    return "\n".join(parts)


def llm_two_pass_self_consistency(user_prompt: str, data_context: str, k: int = 3, return_scores: bool = False) -> tuple:
    """Двухпроходный режим: генерируем k кандидатов, критик исправляет до строгого JSON, выбираем лучший.
    Возвращает (best_text, best_parsed). Текст — отформатированный JSON.
    """
    candidates: list[tuple[str, Optional[LLMAnalysis]]] = []
    gen_count = max(1, int(k))
    # Параллельная генерация k кандидатов
    with ThreadPoolExecutor(max_workers=gen_count) as executor:
        futures = [executor.submit(ask_llm_with_text_data, user_prompt, data_context) for _ in range(gen_count)]
        raw_results = [f.result() for f in futures]

    # Пытаемся распарсить, для неуспешных — параллельный критик
    need_critics = []
    parsed_or_raw: list[tuple[Optional[LLMAnalysis], str]] = []
    for raw in raw_results:
        p = parse_llm_analysis_strict(raw)
        if p is None:
            need_critics.append(raw)
            parsed_or_raw.append((None, raw))
        else:
            parsed_or_raw.append((p, raw))

    if need_critics:
        with ThreadPoolExecutor(max_workers=len(need_critics)) as executor:
            critic_prompts = [_build_critic_prompt(r) for r in need_critics]
            critic_futs = [executor.submit(ask_llm_with_text_data, cp, data_context) for cp in critic_prompts]
            critic_results = [f.result() for f in critic_futs]
        # заместим соответствующие None на результаты критика (в порядке обхода)
        ci = 0
        for p, raw in parsed_or_raw:
            if p is None:
                crit = critic_results[ci]
                ci += 1
                p2 = parse_llm_analysis_strict(crit)
                if p2 is not None:
                    candidates.append((json.dumps(p2.dict(), ensure_ascii=False, indent=2), p2))
                else:
                    candidates.append((raw, None))
            else:
                candidates.append((json.dumps(p.dict(), ensure_ascii=False, indent=2), p))
    else:
        for p, _raw in parsed_or_raw:
            if p is not None:
                candidates.append((json.dumps(p.dict(), ensure_ascii=False, indent=2), p))

    best_text, best_parsed, score_info = _select_best_candidate(candidates, data_context)
    # Если лучший без парсинга — сделаем мягкий фолбэк текстом без изменения
    if best_parsed is None and best_text:
        try:
            mj = _extract_json_like(best_text)
            if mj:
                best_parsed = LLMAnalysis.parse_obj(mj)
                best_text = json.dumps(best_parsed.dict(), ensure_ascii=False, indent=2)
        except Exception:
            pass
    if return_scores:
        return best_text, best_parsed, score_info
    return best_text, best_parsed


def label_dataframes(
    dfs: List[pd.DataFrame],
    labels: List[str]
) -> List[Dict[str, object]]:
    if len(dfs) != len(labels):
        raise ValueError("Количество DataFrame и количество меток не совпадает!")
    labeled_list = []
    for df, label in zip(dfs, labels):
        labeled_list.append({
            "label": label,
            "df": df
        })
    return labeled_list


def _perplexity_call(messages: list[dict], pcfg: dict) -> str:
    base_url = _normalize_llm_base_url(pcfg.get("base_url") or pcfg.get("api_base_url"))
    # Если запрещён веб-поиск, используем оффлайн-инструкционную модель по умолчанию
    disable_web = bool(pcfg.get("disable_web_search", True))
    default_model = "llama-3.1-70b-instruct"
    offline_default = "llama-3.1-70b-instruct"  # та же, без web-search
    model = pcfg.get("model", offline_default if disable_web else default_model)
    gen = (pcfg.get("generation") or {})
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('PPLX_API_KEY') or os.getenv('PERPLEXITY_API_KEY') or pcfg.get('api_key', '')}",
    }

    proxies = (pcfg or {}).get("proxies", {}) or None
    verify_cfg = pcfg.get("verify", True)
    verify = True
    if isinstance(verify_cfg, bool):
        verify = verify_cfg
    elif isinstance(verify_cfg, str) and verify_cfg.strip():
        verify = verify_cfg.strip() if os.path.exists(verify_cfg.strip()) else True

    req_max_tokens = int(gen.get("max_tokens", 1200))
    try:
        cap = int(pcfg.get("max_tokens_cap", 8192))
        if req_max_tokens > cap:
            logger.warning(f"perplexity.max_tokens={req_max_tokens} > cap={cap}, снижаю до cap")
            req_max_tokens = cap
    except Exception:
        pass
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(gen.get("temperature", 0.2)),
        "top_p": float(gen.get("top_p", 0.9)),
        "max_tokens": req_max_tokens,
    }
    if disable_web:
        # Явно отключаем веб-поиск в Perplexity API (см. docs)
        payload["disable_search"] = True

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=int(pcfg.get("request_timeout_sec", 120)),
        verify=verify,
        proxies=proxies,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            logger.warning(f"Perplexity HTTP {resp.status_code}: {resp.text[:500]}")
        except Exception:
            pass
        raise e
    data = resp.json()
    try:
        return _strip_think(data["choices"][0]["message"]["content"]) 
    except Exception:
        # вернём сырой json как текст
        return _strip_think(json.dumps(data, ensure_ascii=False))


def _openai_call(messages: list[dict], pcfg: dict) -> str:
    base_url = (_normalize_llm_base_url(pcfg.get("api_base_url") or pcfg.get("base_url")))
    url = f"{base_url}/chat/completions"
    gen = (pcfg.get("generation") or {})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY') or pcfg.get('api_key', '')}",
    }
    proxies = (pcfg or {}).get("proxies", {}) or None
    verify_cfg = pcfg.get("verify", True)
    verify = True if isinstance(verify_cfg, bool) else (verify_cfg.strip() if isinstance(verify_cfg, str) and verify_cfg.strip() else True)
    req_max_tokens = int(gen.get("max_tokens", 1200))
    try:
        cap = int(pcfg.get("max_tokens_cap", 8192))
        if req_max_tokens > cap:
            logger.warning(f"openai.max_tokens={req_max_tokens} > cap={cap}, снижаю до cap")
            req_max_tokens = cap
    except Exception:
        pass
    payload = {
        "model": pcfg.get("model", "gpt-4o-mini"),
        "messages": messages,
        "temperature": float(gen.get("temperature", 0.2)),
        "top_p": float(gen.get("top_p", 0.9)),
        "max_tokens": req_max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=int(pcfg.get("request_timeout_sec", 120)), verify=verify, proxies=proxies)
    resp.raise_for_status()
    data = resp.json()
    try:
        return _strip_think(data["choices"][0]["message"]["content"]) 
    except Exception:
        return _strip_think(json.dumps(data, ensure_ascii=False))


def _anthropic_call(messages: list[dict], pcfg: dict, system_text: str) -> str:
    base_url = (pcfg.get("api_base_url") or pcfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
    url = f"{base_url}/v1/messages"
    gen = (pcfg.get("generation") or {})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": os.getenv('ANTHROPIC_API_KEY') or pcfg.get('api_key', ''),
        "anthropic-version": "2023-06-01",
    }
    proxies = (pcfg or {}).get("proxies", {}) or None
    verify_cfg = pcfg.get("verify", True)
    verify = True if isinstance(verify_cfg, bool) else (verify_cfg.strip() if isinstance(verify_cfg, str) and verify_cfg.strip() else True)
    # Схема сообщений Anthropic: system + messages (user)
    # Объединяем наши сообщения в один блок user
    user_content_parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            user_content_parts.append(str(content))
        elif role == "system":
            # system_text уже отдельно передаём
            pass
        else:
            user_content_parts.append(str(content))
    user_combined = "\n\n".join(user_content_parts)
    req_max_tokens = int(gen.get("max_tokens", 1200))
    try:
        cap = int(pcfg.get("max_tokens_cap", 4096))
        if req_max_tokens > cap:
            logger.warning(f"anthropic.max_tokens={req_max_tokens} > cap={cap}, снижаю до cap")
            req_max_tokens = cap
    except Exception:
        pass
    payload = {
        "model": pcfg.get("model", "claude-3-5-sonnet-latest"),
        "system": system_text,
        "max_tokens": req_max_tokens,
        "temperature": float(gen.get("temperature", 0.2)),
        "messages": [
            {"role": "user", "content": user_combined}
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=int(pcfg.get("request_timeout_sec", 120)), verify=verify, proxies=proxies)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            logger.warning(f"Anthropic-compatible HTTP {resp.status_code}: {resp.text[:500]}")
        except Exception:
            pass
        raise e
    data = resp.json()
    try:
        blocks = data.get("content", [])
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
        return _strip_think("\n".join([t for t in texts if t]))
    except Exception:
        return _strip_think(json.dumps(data, ensure_ascii=False))


def ask_llm_with_text_data(
    user_prompt: str,
    data_context: str,
    llm_config: dict = None,
    api_key: str = None,
    model: str = None,
    base_url: str = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Отправляет запрос к Perplexity Chat Completions API с подготовленными текстовыми данными.
    """
    llm_root = CONFIG.get("llm", {}) or {}
    provider = (llm_config or {}).get("provider") if isinstance(llm_config, dict) else None
    provider = (provider or llm_root.get("provider") or "perplexity").lower()
    pcfg = llm_root.get(provider, {})
    global _llm_env_applied
    if not _llm_env_applied:
        with _llm_env_init_lock:
            if not _llm_env_applied:
                _ensure_llm_network_env(pcfg)
                _llm_env_applied = True

    gen = (pcfg.get("generation") or {})
    force_json = bool(gen.get("force_json_in_prompt", True))
    if isinstance(llm_config, dict) and "force_json" in llm_config:
        force_json = bool(llm_config.get("force_json"))
    system_text = (
        "Вы инженер по нагрузочному тестированию. Должны проанализировать результаты ступенчатого нагрузочного теста поиска максимальной производительности."
        "Пользователь предоставит данные и вопрос. "
        "Используйте контекст этих данных, чтобы ответить на его вопрос. "
        "Отвечайте на русском языке. Все текстовые поля (verdict, findings.summary, findings.evidence, recommended_actions, affected_components) формулируйте по-русски; допускаются английские только ключи JSON, значения 'severity' и имена метрик/лейблов. " +
        (
            "Строго в JSON со схемой: {verdict, confidence, findings[], recommended_actions[]}. "
            "Каждый элемент findings обязан содержать: summary, severity (critical|high|medium|low), component, evidence. "
            "Если component не указан — извлеките его из evidence по лейблам application|service|job|pod|instance, иначе 'unknown'. "
            "Если severity не указана — используйте 'low'. "
            "Если данные содержат общий RPS на входной точке, включите поле peak_performance: {max_rps, max_time, drop_time, method='max_step_before_drop'}."
            if force_json else ""
        )
    )
    if isinstance(system_prompt, str) and system_prompt.strip():
        system_text = system_prompt.strip()

    user_content = user_prompt if not data_context else f"{user_prompt}\n\n{data_context}"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

    # Переопределения на уровне аргументов
    if isinstance(model, str) and model.strip():
        pcfg = {**pcfg, "model": model.strip()}
    if isinstance(base_url, str) and base_url.strip():
        pcfg = {**pcfg, "api_base_url": base_url.strip()}
    if isinstance(api_key, str) and api_key.strip():
        pcfg = {**pcfg, "api_key": api_key.strip()}

    attempts = 0
    last_err = None
    while attempts < 3:
        try:
            with _llm_semaphore:
                if provider == "perplexity":
                    return _perplexity_call(messages, pcfg)
                elif provider == "openai":
                    return _openai_call(messages, pcfg)
                elif provider == "anthropic":
                    return _anthropic_call(messages, pcfg, system_text)
                else:
                    return _perplexity_call(messages, pcfg)
        except Exception as e:
            last_err = e
            attempts += 1
            logger.warning(f"LLM call retry {attempts}/3 due to: {e}")
            if attempts < 3:
                time.sleep(min(2 ** attempts, 8))
            else:
                raise last_err


def uploadFromLLM(start_ts: float, end_ts: float, save_to_db: bool = False, run_meta: dict | None = None, only_collect: bool = False) -> Dict[str, object]:
	from AI.pipeline import uploadFromLLM as _pipeline_upload
	return _pipeline_upload(start_ts, end_ts, save_to_db=save_to_db, run_meta=run_meta, only_collect=only_collect)


