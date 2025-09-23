# main.py

import requests
import pandas as pd
from typing import List, Dict, Optional, Union
import json
import os
from datetime import datetime
import logging
import threading
import socket
import time
from urllib.parse import urlparse

from langchain_gigachat.chat_models import GigaChat as LC_GigaChat
from pydantic import BaseModel, Field, ValidationError, root_validator
try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    try:
        from langchain.schema import SystemMessage, HumanMessage  # старые версии langchain
    except Exception:
        SystemMessage = None
        HumanMessage = None


# Импортируем CONFIG из config.py
from AI.config import CONFIG

logger = logging.getLogger(__name__)
_gigachat_lock = threading.Lock()
_gigachat_client = None
_gigachat_env_applied = False
_gigachat_preflight_last_ts = 0.0

# httpx/requests таймаут через окружение для некоторых клиентов
os.environ.setdefault("GIGACHAT_TIMEOUT", str(CONFIG.get("llm", {}).get("gigachat", {}).get("request_timeout_sec", 120)))


def _normalize_gigachat_base_url(raw_url: str | None) -> str:
    base = (raw_url or "https://gigachat.devices.sberbank.ru/api/v1").strip()
    if not base:
        return "https://gigachat.devices.sberbank.ru/api/v1"
    base = base.rstrip("/")
    # Убираем случайный суффикс /chat/completions, если администратор указал полноценный путь
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _ensure_gigachat_env(gcfg: dict) -> None:
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
        f"GigaChat net env set: HTTPS_PROXY={'set' if https_proxy else 'unset'} "
        f"HTTP_PROXY={'set' if http_proxy else 'unset'} CA_BUNDLE={'set' if (ca_bundle and not insecure) else 'unset'} "
        f"INSECURE={'on' if insecure else 'off'}"
    )


def _gigachat_preflight(gcfg: dict) -> None:
    """Быстрая проверка доступности GigaChat API (mTLS)."""
    timeout = float((gcfg or {}).get("connect_timeout_sec") or 5)
    proxies = (gcfg or {}).get("proxies", {}) or {}
    has_proxy = bool(proxies.get("https") or proxies.get("http") or proxies.get("HTTPS") or proxies.get("HTTP"))

    api_base = _normalize_gigachat_base_url(gcfg.get("api_base_url") or gcfg.get("base_url"))
    parsed = urlparse(api_base)
    host = parsed.hostname or "gigachat.devices.sberbank.ru"
    port = parsed.port or (443 if (parsed.scheme or "https").lower() == "https" else 80)
    models_url = f"{api_base}/models"

    if not has_proxy:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                logger.info(f"Preflight ok: api_host {host}:{port}")
        except Exception as e:
            logger.error(f"Preflight FAIL: api_host {host}:{port} -> {e}")
    else:
        logger.info("Preflight: proxies detected, skipping direct TCP checks")

    if not bool(gcfg.get("enable_preflight_models", True)):
        logger.info("Preflight: /models check disabled by config")
        return

    verify_cfg = gcfg.get("verify") or gcfg.get("ca_bundle_file") or gcfg.get("ca_bundle") or True
    verify = True
    if isinstance(verify_cfg, bool):
        verify = verify_cfg
    elif isinstance(verify_cfg, str) and verify_cfg.strip():
        verify = verify_cfg.strip() if os.path.exists(verify_cfg.strip()) else True

    cert = None
    cfile = gcfg.get("cert_file"); kfile = gcfg.get("key_file")
    if gcfg.get("use_mtls") and isinstance(cfile, str) and isinstance(kfile, str):
        cfile = cfile.strip(); kfile = kfile.strip()
        if cfile and kfile and os.path.exists(cfile) and os.path.exists(kfile):
            cert = (cfile, kfile)
        else:
            logger.warning("mTLS включён, но cert/key файл(ы) не найдены. Пропускаю клиентский сертификат.")

    try:
        resp = requests.get(
            models_url,
            headers={"Accept": "application/json"},
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies if proxies else None,
        )
        logger.info(f"Preflight GET {models_url} status={resp.status_code}")
    except Exception as e:
        logger.error(f"Preflight HTTP to API failed: {e}")


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


def _ask_domain_analysis(prompt_text: str, data_context: str) -> tuple[str, Optional[LLMAnalysis]]:
    return llm_two_pass_self_consistency(user_prompt=prompt_text, data_context=data_context, k=3)


def _build_critic_prompt(candidate_text: str) -> str:
    """Строит промпт-критику для исправления ответа в строгий JSON по схеме.
    Кандидат может содержать пояснения; задача критика — выдать ЧИСТЫЙ JSON.
    """
    return (
        "Вы выступаете как строгий валидатор отчёта. Отвечайте на русском языке. "
        "Ниже дан проект ответа модели. Исправьте/нормализуйте его до СТРОГОГО JSON со схемой: "
        "{verdict, confidence, findings[], recommended_actions[]}. "
        "Каждый элемент findings ДОЛЖЕН содержать поля severity (critical|high|medium|low) и component (имя сервиса/процесса/инстанса). "
        "Если component явно не указан, извлеките его из evidence по лейблам application|service|job|pod|instance; если не удаётся — поставьте 'unknown'. "
        "Если severity не указана или не из списка — установите 'low'. "
        "Дополнительно допускается поле peak_performance: {max_rps, max_time, drop_time, method}, если оно упомянуто в проекте ответа. "
        "Никакого текста вне JSON. Если данных недостаточно — верните verdict='insufficient_data'.\n\n"
        f"Проект ответа:\n{candidate_text}"
    )


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
    best = max(pool, key=lambda tp: conf_val(tp[1]))
    return best


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


def llm_two_pass_self_consistency(user_prompt: str, data_context: str, k: int = 3) -> tuple[str, Optional[LLMAnalysis]]:
    """Двухпроходный режим: генерируем k кандидатов, критик исправляет до строгого JSON, выбираем лучший.
    Возвращает (best_text, best_parsed). Текст — отформатированный JSON.
    """
    candidates: list[tuple[str, Optional[LLMAnalysis]]] = []
    for _ in range(max(1, int(k))):
        raw = ask_llm_with_text_data(user_prompt=user_prompt, data_context=data_context)
        parsed = parse_llm_analysis_strict(raw)
        if parsed is None:
            # Критик с попыткой нормализовать
            critic_prompt = _build_critic_prompt(raw)
            crit = ask_llm_with_text_data(user_prompt=critic_prompt, data_context=data_context)
            parsed = parse_llm_analysis_strict(crit)
            if parsed is not None:
                json_text = json.dumps(parsed.dict(), ensure_ascii=False, indent=2)
                candidates.append((json_text, parsed))
            else:
                candidates.append((raw, None))
        else:
            json_text = json.dumps(parsed.dict(), ensure_ascii=False, indent=2)
            candidates.append((json_text, parsed))

    best_text, best_parsed = _choose_best_candidate(candidates)
    # Если лучший без парсинга — сделаем мягкий фолбэк текстом без изменения
    if best_parsed is None and best_text:
        try:
            mj = _extract_json_like(best_text)
            if mj:
                best_parsed = LLMAnalysis.parse_obj(mj)
                best_text = json.dumps(best_parsed.dict(), ensure_ascii=False, indent=2)
        except Exception:
            pass
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


def _get_gigachat_client() -> LC_GigaChat:
    global _gigachat_client
    if _gigachat_client is not None:
        return _gigachat_client
    gcfg = CONFIG.get("llm", {}).get("gigachat", {})
    base_url = _normalize_gigachat_base_url(gcfg.get("base_url") or gcfg.get("api_base_url"))
    model = gcfg.get("model", "GigaChat-Pro")
    # приводим пустые строки к None, чтобы не пытаться открыть несуществующие файлы
    raw_cert_file = gcfg.get("cert_file")
    raw_key_file = gcfg.get("key_file")
    cert_file = raw_cert_file if (isinstance(raw_cert_file, str) and raw_cert_file.strip()) else None
    key_file = raw_key_file if (isinstance(raw_key_file, str) and raw_key_file.strip()) else None

    verify_param = gcfg.get("verify")
    verify_ssl_certs = verify_param if isinstance(verify_param, bool) else True
    if isinstance(verify_param, str):
        verify_path = verify_param.strip()
        if verify_path:
            if os.path.exists(verify_path):
                os.environ["REQUESTS_CA_BUNDLE"] = verify_path
                os.environ["SSL_CERT_FILE"] = verify_path
            else:
                logger.warning(f"CA bundle path not found: '{verify_path}'. Skipping SSL env vars.")

    logger.info(
        """
Инициализация подключения к GigaChat
URL: %s
Model: %s
SSL: %s
Debug: %s
""",
        base_url,
        model,
        True,
        False,
    )
    gen = (CONFIG.get("llm", {}).get("gigachat", {}).get("generation") or {})
    _gigachat_client = LC_GigaChat(
        model=model,
        cert_file=cert_file,
        key_file=key_file,
        base_url=base_url,
        verify_ssl_certs=verify_ssl_certs,
        timeout=int(CONFIG.get("llm", {}).get("gigachat", {}).get("request_timeout_sec", 120)),
        # Дополнительные параметры генерации (если клиент поддерживает)
        temperature=float(gen.get("temperature", 0.2)),
        top_p=float(gen.get("top_p", 0.9)),
        max_tokens=int(gen.get("max_tokens", 1200)),
    )
    return _gigachat_client


def ask_llm_with_text_data(
    user_prompt: str,
    data_context: str,
    llm_config: dict = None,
    api_key: str = None,
    model: str = None,
    base_url: str = None
) -> str:
    """
    Отправляет запрос к GigaChat (через langchain_gigachat) с подготовленными текстовыми данными.
    """
    gcfg = CONFIG.get("llm", {}).get("gigachat", {})
    global _gigachat_env_applied, _gigachat_preflight_last_ts
    # Кэшируем применение окружения на процесс
    if not _gigachat_env_applied:
        _ensure_gigachat_env(gcfg)
        _gigachat_env_applied = True
    # Префлайт выполняем не чаще, чем раз в 10 минут
    now_ts = time.time()
    if now_ts - _gigachat_preflight_last_ts > 600:
        try:
            _gigachat_preflight(gcfg)
        finally:
            _gigachat_preflight_last_ts = now_ts

    gen = (CONFIG.get("llm", {}).get("gigachat", {}).get("generation") or {})
    # Можно переопределить требование JSON на уровне вызова (для overall)
    force_json = bool(gen.get("force_json_in_prompt", True))
    if isinstance(llm_config, dict) and "force_json" in llm_config:
        force_json = bool(llm_config.get("force_json"))
    system_text = (
        "Вы инженер по нагрузочному тестированию. Должны проанализировать результаты ступенчатого нагрузочного теста поиска максимальной производительности."
        "Пользователь предоставит данные и вопрос. "
        "Используйте контекст этих данных, чтобы ответить на его вопрос. "
        "Отвечайте на русском языке. " +
        (
            "Строго в JSON со схемой: {verdict, confidence, findings[], recommended_actions[]}. "
            "Каждый элемент findings обязан содержать: summary, severity (critical|high|medium|low), component, evidence. "
            "Если component не указан — извлеките его из evidence по лейблам application|service|job|pod|instance, иначе 'unknown'. "
            "Если severity не указана — используйте 'low'. "
            "Если данные содержат общий RPS на входной точке, включите поле peak_performance: {max_rps, max_time, drop_time, method='max_step_before_drop'}."
            if force_json else ""
        )
    )
    if SystemMessage and HumanMessage:
        lc_messages = [SystemMessage(content=system_text), HumanMessage(content=user_prompt + f"\n\n{data_context}")]
        with _gigachat_lock:
            # простые ретраи при read timeout
            attempts = 0
            last_err = None
            while attempts < 3:
                try:
                    result = _get_gigachat_client().invoke(lc_messages)
                    break
                except Exception as e:
                    last_err = e
                    attempts += 1
                    logger.warning(f"GigaChat invoke retry {attempts}/3 due to: {e}")
                    if attempts < 3:
                        time.sleep(min(2 ** attempts, 8))
                    else:
                        raise last_err
            return getattr(result, "content", str(result))
    else:
        # Fallback: одним запросом
        with _gigachat_lock:
            attempts = 0
            last_err = None
            while attempts < 3:
                try:
                    result = _get_gigachat_client().invoke(system_text + "\n\n" + user_prompt + f"\n\n{data_context}")
                    break
                except Exception as e:
                    last_err = e
                    attempts += 1
                    logger.warning(f"GigaChat invoke retry {attempts}/3 due to: {e}")
                    if attempts < 3:
                        time.sleep(min(2 ** attempts, 8))
                    else:
                        raise last_err
        return getattr(result, "content", str(result))


def read_prompt_from_file(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def uploadFromLLM(start_ts: float, end_ts: float) -> Dict[str, object]:
    _configure_logging()
    prometheus_url = CONFIG["prometheus"]["url"]
    src_type = (CONFIG.get("metrics_source", {}).get("type") or "prometheus").lower()
    if src_type == "grafana_proxy":
        g = CONFIG.get("metrics_source", {}).get("grafana", {})
        logger.info(f"Metrics source: Grafana proxy base_url={g.get('base_url')} auth_method={(g.get('auth', {}).get('method'))} verify_ssl={g.get('verify_ssl')} ds_hint={g.get('prometheus_datasource')}")
    else:
        logger.info(f"Metrics source: Prometheus url={prometheus_url}")
    
    step = CONFIG["default_params"]["step"]
    resample = CONFIG["default_params"]["resample_interval"]

    queries = CONFIG["queries"]
    domain_keys = ["jvm", "database", "kafka", "microservices"]
    domain_data = {}
    for key in domain_keys:
        try:
            domain_data[key] = _build_domain_data(
                domain_key=key,
                domain_conf=queries[key],
                prometheus_url=prometheus_url,
                start_ts=start_ts,
                end_ts=end_ts,
                step=step,
                resample=resample,
                top_n=15
            )
        except Exception as e:
            logger.error(f"Domain '{key}' build failed: {e}")
            # продвигаемся дальше, формируя пустые структуры
            domain_data[key] = {"labeled": [], "markdown": "", "pack": {"sections": []}, "ctx": json.dumps({"domain": key, "sections": []}, ensure_ascii=False)}

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(CURRENT_DIR, "prompts")
    
    prompt_jvm = read_prompt_from_file(os.path.join(prompts_dir, "jvm_prompt.txt"))
    # Промпт для домена Database (без обратной совместимости)
    prompt_database = read_prompt_from_file(os.path.join(prompts_dir, "database_prompt.txt"))
    prompt_kafka = read_prompt_from_file(os.path.join(prompts_dir, "kafka_prompt.txt"))
    prompt_microservices = read_prompt_from_file(os.path.join(prompts_dir, "microservices_prompt.txt"))
    prompt_overall = read_prompt_from_file(os.path.join(prompts_dir, "overall_prompt.txt"))

    # Витрины и контексты для LLM
    jvm_full_data = domain_data["jvm"]["markdown"]; jvm_pack = domain_data["jvm"]["pack"]; jvm_ctx = domain_data["jvm"]["ctx"]
    database_full_data = domain_data["database"]["markdown"]; database_pack = domain_data["database"]["pack"]; database_ctx = domain_data["database"]["ctx"]
    kafka_full_data = domain_data["kafka"]["markdown"]; kafka_pack = domain_data["kafka"]["pack"]; kafka_ctx = domain_data["kafka"]["ctx"]
    ms_full_data = domain_data["microservices"]["markdown"]; ms_pack = domain_data["microservices"]["pack"]
    # Обогащаем контекст микросервисов ресурсными метриками из JVM (CPU/Memory)
    cpu_sections = []
    mem_sections = []
    try:
        for sec in jvm_pack.get("sections", []):
            lbl = str(sec.get("label", ""))
            if "Process CPU usage" in lbl:
                cpu_sections.append(sec)
            if "Heap used" in lbl or "Heap max" in lbl:
                mem_sections.append(sec)
    except Exception:
        pass
    ms_ctx_obj = {
        "domain": "microservices",
        "time_range": {"start": start_ts, "end": end_ts},
        **ms_pack,
        "aux_resources": {
            "cpu_sections": cpu_sections,
            "memory_sections": mem_sections
        }
    }
    ms_ctx = json.dumps(ms_ctx_obj, ensure_ascii=False)

    # Two-pass + self-consistency (k=3)
    try:
        answer_jvm, jvm_parsed = _ask_domain_analysis(prompt_jvm, jvm_ctx)
    except Exception as e:
        logger.error(f"LLM jvm analysis failed: {e}")
        answer_jvm, jvm_parsed = ("{}", None)
    try:
        answer_database, database_parsed = _ask_domain_analysis(prompt_database, database_ctx)
    except Exception as e:
        logger.error(f"LLM database analysis failed: {e}")
        answer_database, database_parsed = ("{}", None)
    try:
        answer_kafka, kafka_parsed = _ask_domain_analysis(prompt_kafka, kafka_ctx)
    except Exception as e:
        logger.error(f"LLM kafka analysis failed: {e}")
        answer_kafka, kafka_parsed = ("{}", None)
    try:
        answer_ms, ms_parsed = _ask_domain_analysis(prompt_microservices, ms_ctx)
    except Exception as e:
        logger.error(f"LLM microservices analysis failed: {e}")
        answer_ms, ms_parsed = ("{}", None)

    merged_prompt_overall = (
        prompt_overall
        .replace("{answer_jvm}", answer_jvm)
        .replace("{answer_database}", answer_database)
        .replace("{answer_kafka}", answer_kafka)
        .replace("{answer_microservices}", answer_ms)
    )

    overall_ctx = json.dumps({
        "time_range": {"start": start_ts, "end": end_ts},
        "domains": {
            "jvm": jvm_pack,
            "database": database_pack,
            "kafka": kafka_pack,
            "microservices": ms_pack
        }
    }, ensure_ascii=False)
    final_answer, final_parsed = llm_two_pass_self_consistency(user_prompt=merged_prompt_overall, data_context=overall_ctx, k=3)

    return {
        "jvm": f"{jvm_full_data}\n\nАнализ JVM:\n{answer_jvm}",
        "database": f"{database_full_data}\n\nАнализ Database:\n{answer_database}",
        "kafka": f"{kafka_full_data}\n\nАнализ Kafka:\n{answer_kafka}",
        "ms": f"{ms_full_data}\n\nАнализ микросервисов:\n{answer_ms}",
        "final": final_answer,
        "jvm_parsed": (jvm_parsed.dict() if jvm_parsed else None),
        "database_parsed": (database_parsed.dict() if database_parsed else None),
        "kafka_parsed": (kafka_parsed.dict() if kafka_parsed else None),
        "ms_parsed": (ms_parsed.dict() if ms_parsed else None),
        "final_parsed": (final_parsed.dict() if final_parsed else None),
    }


