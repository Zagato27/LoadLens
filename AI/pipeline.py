import os
import json
import logging
from datetime import datetime
from typing import List, Dict

import requests
import io
import pandas as pd

from settings import CONFIG
from AI.scoring import llm_two_pass_self_consistency
from AI.db_store import save_domain_labeled, save_llm_results


logger = logging.getLogger(__name__)


def _configure_logging():
    level_name = (CONFIG.get("logging", {}).get("level") if CONFIG.get("logging") else "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger().setLevel(level)


def read_prompt_from_file(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def parse_step_to_seconds(step: str) -> int:
    if step.endswith('m'):
        return int(step[:-1]) * 60
    elif step.endswith('s'):
        return int(step[:-1])
    else:
        return int(step)


def fetch_prometheus_data(prometheus_url: str, start_ts: float, end_ts: float, promql_query: str, step: str) -> dict:
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


def _resolve_grafana_influx_ds_id(g_cfg: dict) -> int:
    base_url = g_cfg["base_url"].rstrip("/")
    ds_cfg = g_cfg.get("influxdb_datasource", {}) or g_cfg.get("prometheus_datasource", {}) or {}
    # допускаем переиспользование ключей uid/name, но по типу будем искать influxdb
    auth_cfg = g_cfg.get("auth", {})
    headers = {}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    verify = g_cfg.get("verify_ssl", True)
    # Если указан id — используем
    if isinstance(ds_cfg.get("id"), int):
        return ds_cfg["id"]
    # Если указан uid — попробуем его
    if ds_cfg.get("uid"):
        url = f"{base_url}/api/datasources/uid/{ds_cfg['uid']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]
    # Если указан name — попробуем его
    if ds_cfg.get("name"):
        url = f"{base_url}/api/datasources/name/{ds_cfg['name']}"
        resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
        resp.raise_for_status()
        return resp.json()["id"]
    # Иначе постараемся найти любой influx datasource
    url = f"{base_url}/api/datasources"
    resp = requests.get(url, headers=headers, auth=auth, timeout=30, verify=verify)
    resp.raise_for_status()
    for ds in resp.json():
        if ds.get("type") in ("influxdb", "influxdb2"):
            return ds["id"]
    raise RuntimeError("Не найден InfluxDB datasource в Grafana")


def fetch_influx_data_via_grafana(g_cfg: dict, flux_query: str) -> str:
    base_url = g_cfg["base_url"].rstrip("/")
    ds_id = _resolve_grafana_influx_ds_id(g_cfg)
    auth_cfg = g_cfg.get("auth", {})
    headers = {"Accept": "application/csv", "Content-Type": "application/json"}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    url = f"{base_url}/api/datasources/proxy/{ds_id}/api/v2/query"
    resp = requests.post(url, headers=headers, auth=auth, json={"query": flux_query}, timeout=60, verify=g_cfg.get("verify_ssl", True))
    resp.raise_for_status()
    return resp.text


def fetch_influx_and_aggregate_via_grafana(
    grafana_cfg: dict,
    influx_aux_cfg: dict,
    start_ts: float,
    end_ts: float,
    flux_queries: List[str],
    label_tag_keys_list: List[List[str]],
    labels: List[str],
    resample_interval: str
) -> List[pd.DataFrame]:
    bucket = (influx_aux_cfg or {}).get("bucket", "")
    t_start = _iso8601_utc(start_ts)
    t_end = _iso8601_utc(end_ts)
    dfs: List[pd.DataFrame] = []
    for idx, flux in enumerate(flux_queries):
        try:
            q = (flux or "").replace("{bucket}", bucket).replace("{start}", t_start).replace("{end}", t_end)
            csv_text = fetch_influx_data_via_grafana(grafana_cfg, q)
            df = pd.read_csv(io.StringIO(csv_text))
            if "_time" not in df.columns or "_value" not in df.columns:
                dfs.append(pd.DataFrame())
                continue
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df = df.dropna(subset=["_time", "_value"])
            tag_keys = label_tag_keys_list[idx] if idx < len(label_tag_keys_list) else []
            tag_keys = list(tag_keys or [])
            def make_label(row):
                parts=[]
                for k in tag_keys:
                    if k in row and pd.notnull(row[k]):
                        parts.append(f"{k}={row[k]}")
                return "|".join(parts) if parts else "series"
            df["series"] = df.apply(make_label, axis=1)
            pivot = df.pivot_table(index="_time", columns="series", values="_value", aggfunc="mean")
            try:
                pivot = pivot.resample(resample_interval).mean()
            except Exception:
                pass
            pivot.index = pd.to_datetime(pivot.index, utc=True)
            dfs.append(pivot)
        except Exception:
            dfs.append(pd.DataFrame())
    return dfs


def _convert_pd_offset_to_influx_interval(s: str) -> str:
    if not isinstance(s, str) or not s:
        return "1m"
    s = s.strip()
    if s.endswith("T"):  # minutes
        try:
            return f"{int(s[:-1])}m"
        except Exception:
            return "1m"
    if s.endswith("S"):  # seconds
        try:
            return f"{int(s[:-1])}s"
        except Exception:
            return "60s"
    if s.endswith("H"):
        try:
            return f"{int(s[:-1])}h"
        except Exception:
            return "1h"
    return "1m"


def fetch_influxql_via_grafana(g_cfg: dict, q: str, database: str | None) -> dict:
    base_url = g_cfg["base_url"].rstrip("/")
    ds_id = _resolve_grafana_influx_ds_id(g_cfg)
    auth_cfg = g_cfg.get("auth", {})
    headers = {"Accept": "application/json"}
    auth = None
    method = (auth_cfg.get("method") or "basic").lower()
    if method == "bearer" and auth_cfg.get("token"):
        headers["Authorization"] = f"Bearer {auth_cfg.get('token')}"
    elif method == "basic" and auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (auth_cfg.get("username"), auth_cfg.get("password"))
    params = {"q": q}
    if database:
        params["db"] = database
    url = f"{base_url}/api/datasources/proxy/{ds_id}/query"
    resp = requests.get(url, headers=headers, auth=auth, params=params, timeout=60, verify=g_cfg.get("verify_ssl", True))
    resp.raise_for_status()
    return resp.json()


def fetch_influxql_and_aggregate_via_grafana(
    grafana_cfg: dict,
    influx_aux_cfg: dict,
    start_ts: float,
    end_ts: float,
    influxql_queries: List[str],
    label_tag_keys_list: List[List[str]],
    labels: List[str],
    resample_interval: str
) -> List[pd.DataFrame]:
    # Подстановка макросов $timeFilter и $__interval; шаблонные переменные ($Group/$Tag/$URL/$Measurement) заменяем на .* (все)
    t_start_ns = int(start_ts * 1_000_000_000)
    t_end_ns = int(end_ts * 1_000_000_000)
    iv = _convert_pd_offset_to_influx_interval(resample_interval)
    database = (influx_aux_cfg or {}).get("database")
    dfs: List[pd.DataFrame] = []
    for idx, raw in enumerate(influxql_queries or []):
        try:
            q = (raw or "")
            q = q.replace("$timeFilter", f"time >= {t_start_ns} AND time <= {t_end_ns}")
            q = q.replace("$__interval", iv)
            # шаблонные переменные -> .* (regex match any)
            for var in ("$Group", "$Tag", "$URL", "$Measurement"):
                q = q.replace(var, ".*")
            data = fetch_influxql_via_grafana(grafana_cfg, q, database)
            series_list = (((data or {}).get("results") or [{}])[0].get("series")) or []
            # Объединим все series в один pivot
            frames = []
            tag_keys = label_tag_keys_list[idx] if idx < len(label_tag_keys_list) else []
            tag_keys = list(tag_keys or [])
            for s in series_list:
                cols = s.get("columns") or []
                values = s.get("values") or []
                tags = s.get("tags") or {}
                if not values:
                    continue
                df = pd.DataFrame(values, columns=cols)
                if "time" not in df.columns:
                    # иногда колонка может называться "time"
                    continue
                df["time"] = pd.to_datetime(df["time"], utc=True)
                # label from tags
                parts = []
                for k in tag_keys:
                    if k in tags:
                        parts.append(f"{k}={tags[k]}")
                label = "|".join(parts) if parts else s.get("name") or "series"
                # value column: take first numeric column except time
                val_col = None
                for c in df.columns:
                    if c == "time":
                        continue
                    if pd.api.types.is_numeric_dtype(df[c]):
                        val_col = c
                        break
                if not val_col:
                    # fallback: try column named 'sum' or 'percentile'
                    for c in ("sum", "mean", "percentile", "value"):
                        if c in df.columns:
                            val_col = c
                            break
                if not val_col:
                    continue
                # Приведём значения к числу для устойчивого ресемплинга
                try:
                    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
                except Exception:
                    pass
                df = df.dropna(subset=[val_col])
                tmp = df[["time", val_col]].rename(columns={val_col: label}).set_index("time")
                frames.append(tmp)
            if not frames:
                dfs.append(pd.DataFrame())
                continue
            merged = pd.concat(frames, axis=1).sort_index()
            # опциональная ресемплинг
            try:
                merged = merged.resample(resample_interval).mean()
            except Exception:
                pass
            dfs.append(merged)
        except Exception:
            dfs.append(pd.DataFrame())
    return dfs

def fetch_prometheus_data_via_grafana(g_cfg: dict, start_ts: float, end_ts: float, promql_query: str, step: str) -> dict:
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


def fetch_metric_series(prometheus_url: str, start_ts: float, end_ts: float, promql_query: str, step: str, ef_config: dict | None = None) -> dict:
    cfg = ef_config or CONFIG
    src = (cfg.get("metrics_source", {}).get("type") or "prometheus").lower()
    if src == "grafana_proxy":
        g_cfg = cfg.get("metrics_source", {}).get("grafana", {})
        return fetch_prometheus_data_via_grafana(g_cfg, start_ts, end_ts, promql_query, step)
    else:
        prometheus_url_eff = (cfg.get("metrics_source", {}).get("prometheus", {}) or {}).get("url") or prometheus_url
        return fetch_prometheus_data(prometheus_url_eff, start_ts, end_ts, promql_query, step)


def fetch_and_aggregate_with_label_keys(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    promql_queries: List[str],
    label_keys_list: List[List[str]],
    step: str,
    resample_interval: str,
    ef_config: dict | None = None
) -> List[pd.DataFrame]:
    if len(promql_queries) != len(label_keys_list):
        raise ValueError("Количество запросов и количество списков лейблов не совпадает!")
    dfs = []
    for query, keys_for_this_query in zip(promql_queries, label_keys_list):
        data_json = fetch_metric_series(prometheus_url, start_ts, end_ts, query, step, ef_config=ef_config)
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
            df = pd.DataFrame(columns=["ts", "series", "value"])  # пустая
        else:
            # Аггрегируем возможные дубликаты (ts, series), затем пивот
            tmp = pd.DataFrame(records, columns=["ts", "series", "value"]).groupby(["ts", "series"], as_index=False)["value"].mean()
            df = tmp.pivot(index="ts", columns="series", values="value")
            df.index = pd.to_datetime(df.index, unit='s')
            try:
                df = df.resample(resample_interval).mean()
            except Exception:
                pass
        dfs.append(df)
    return dfs


def _iso8601_utc(ts: float) -> str:
    try:
        return pd.to_datetime(ts, unit='s', utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_influx_and_aggregate(
    influx_cfg: dict,
    start_ts: float,
    end_ts: float,
    flux_queries: List[str],
    label_tag_keys_list: List[List[str]],
    labels: List[str],
    resample_interval: str
) -> List[pd.DataFrame]:
    url = (influx_cfg or {}).get("url", "").rstrip("/")
    org = (influx_cfg or {}).get("org", "")
    bucket = (influx_cfg or {}).get("bucket", "")
    token = (influx_cfg or {}).get("token", "")
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/csv",
        "Content-Type": "application/json",
    }
    t_start = _iso8601_utc(start_ts)
    t_end = _iso8601_utc(end_ts)
    dfs: List[pd.DataFrame] = []
    for idx, flux in enumerate(flux_queries):
        try:
            q = (flux or "").replace("{bucket}", bucket).replace("{start}", t_start).replace("{end}", t_end)
            resp = requests.post(
                f"{url}/api/v2/query",
                params={"org": org},
                headers=headers,
                json={"query": q},
                timeout=60
            )
            resp.raise_for_status()
            csv_text = resp.text
            df = pd.read_csv(io.StringIO(csv_text))
            # ожидаемые колонки: _time, _value и теги для серии
            if "_time" not in df.columns or "_value" not in df.columns:
                dfs.append(pd.DataFrame())
                continue
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df = df.dropna(subset=["_time", "_value"])
            tag_keys = label_tag_keys_list[idx] if idx < len(label_tag_keys_list) else []
            tag_keys = list(tag_keys or [])
            def make_label(row):
                parts=[]
                for k in tag_keys:
                    if k in row and pd.notnull(row[k]):
                        parts.append(f"{k}={row[k]}")
                return "|".join(parts) if parts else "series"
            df["series"] = df.apply(make_label, axis=1)
            # Пивот по времени/серии
            pivot = df.pivot_table(index="_time", columns="series", values="_value", aggfunc="mean")
            try:
                pivot = pivot.resample(resample_interval).mean()
            except Exception:
                pass
            pivot.index = pd.to_datetime(pivot.index, utc=True)
            dfs.append(pivot)
        except Exception:
            dfs.append(pd.DataFrame())
    return dfs


def dataframes_to_markdown(labeled: List[Dict[str, object]]) -> str:
    lines = []
    for item in labeled:
        label = str(item.get("label") or "?")
        df = item.get("df")
        lines.append(f"### {label}")
        try:
            md = (df.fillna("") if hasattr(df, 'fillna') else df).head(20).to_markdown() if df is not None else "(пусто)"
        except Exception:
            md = str(getattr(df, 'shape', None))
        lines.append(md)
        lines.append("")
    return "\n".join(lines)


def _summarize_time_series_dataframe(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    if df is None or getattr(df, 'empty', True):
        return summary
    if not isinstance(df.columns, pd.Index):
        return summary
    top_columns = list(df.columns)[:top_n]
    for col in top_columns:
        col_series = df[col]
        if col_series.dropna().empty:
            continue
        try:
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
            continue
        summary.append(series_summary)
    return summary


def build_context_pack(labeled_dfs: List[Dict[str, object]], top_n: int = 10) -> Dict[str, object]:
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
            shifted = mask.astype(int).diff().fillna(int(mask.iloc[0]))
            starts = list(mask.index[shifted == 1])
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


def uploadFromLLM(start_ts: float, end_ts: float, save_to_db: bool = False, run_meta: dict | None = None, only_collect: bool = False, ef_config: dict | None = None, prompts_override: dict | None = None, active_domains: List[str] | None = None) -> Dict[str, object]:
    _configure_logging()
    cfg = ef_config or CONFIG
    src_type = (cfg.get("metrics_source", {}) or {}).get("type", "prometheus").lower()
    ms_cfg = (cfg.get("metrics_source", {}) or {})
    prometheus_url = (ms_cfg.get("prometheus", {}) or {}).get("url", "")
    step = (cfg.get("default_params", {}) or {}).get("step") or CONFIG["default_params"]["step"]
    resample = (cfg.get("default_params", {}) or {}).get("resample_interval") or CONFIG["default_params"]["resample_interval"]

    queries = cfg.get("queries") or CONFIG.get("queries") or {}
    # Определяем доступные домены (включая lt_framework, если задан)
    domain_keys = ["jvm", "database", "kafka", "microservices", "hard_resources"]
    if isinstance(queries.get("lt_framework"), dict):
        domain_keys.append("lt_framework")
    enabled_domain_set = set(domain_keys if active_domains is None else [d for d in active_domains if d in domain_keys])

    def _is_enabled(key: str) -> bool:
        return active_domains is None or key in enabled_domain_set
    domain_data = {}
    def _empty_domain_payload(key: str) -> Dict[str, object]:
        return {"labeled": [], "markdown": "", "pack": {"sections": []}, "ctx": json.dumps({"domain": key, "sections": []}, ensure_ascii=False)}
    for key in domain_keys:
        try:
            if not _is_enabled(key):
                domain_data[key] = _empty_domain_payload(key)
                continue
            if key == "lt_framework":
                qcfg = queries.get("lt_framework") or {}
                # выбор источника lt: отдельный lt_metrics_source или общий
                lt_src_cfg = cfg.get("lt_metrics_source") or cfg.get("metrics_source") or {}
                lt_type = (lt_src_cfg.get("type") or "prometheus").lower()
                if lt_type in ("prometheus",):
                    dfs = fetch_and_aggregate_with_label_keys(
                        (lt_src_cfg.get("prometheus", {}) or {}).get("url", prometheus_url),
                        start_ts,
                        end_ts,
                        qcfg.get("promql_queries", []),
                        qcfg.get("label_keys_list", []),
                        step=step,
                        resample_interval=resample,
                        ef_config={"metrics_source": lt_src_cfg}
                    )
                elif lt_type == "grafana_proxy":
                    # Через Grafana proxy: если заданы promql_queries — используем прометеус-путь; если заданы flux_queries — Influx-путь
                    if qcfg.get("promql_queries"):
                        dfs = fetch_and_aggregate_with_label_keys(
                            (lt_src_cfg.get("prometheus", {}) or {}).get("url", prometheus_url),
                            start_ts,
                            end_ts,
                            qcfg.get("promql_queries", []),
                            qcfg.get("label_keys_list", []),
                            step=step,
                            resample_interval=resample,
                            ef_config={"metrics_source": lt_src_cfg}
                        )
                    elif qcfg.get("influxql_queries"):
                        influx_aux = lt_src_cfg.get("influxdb", {}) or {}
                        dfs = fetch_influxql_and_aggregate_via_grafana(
                            grafana_cfg=lt_src_cfg.get("grafana", {}) or {},
                            influx_aux_cfg=influx_aux,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            influxql_queries=qcfg.get("influxql_queries", []),
                            label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
                            labels=qcfg.get("labels", []),
                            resample_interval=resample
                        )
                    else:
                        influx_aux = lt_src_cfg.get("influxdb", {}) or {}
                        dfs = fetch_influx_and_aggregate_via_grafana(
                            grafana_cfg=lt_src_cfg.get("grafana", {}) or {},
                            influx_aux_cfg=influx_aux,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            flux_queries=qcfg.get("flux_queries", []),
                            label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
                            labels=qcfg.get("labels", []),
                            resample_interval=resample
                        )
                elif lt_type == "influxdb":
                    influx_cfg = lt_src_cfg.get("influxdb", {}) or {}
                    dfs = fetch_influx_and_aggregate(
                        influx_cfg=influx_cfg,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        flux_queries=qcfg.get("flux_queries", []),
                        label_tag_keys_list=qcfg.get("label_tag_keys_list", []),
                        labels=qcfg.get("labels", []),
                        resample_interval=resample
                    )
                else:
                    dfs = []
                labeled = label_dataframes(dfs, (queries.get(key, {}) or {}).get("labels", []))
            else:
                dfs = fetch_and_aggregate_with_label_keys(
                    prometheus_url,
                    start_ts,
                    end_ts,
                    queries[key]["promql_queries"],
                    queries[key]["label_keys_list"],
                    step=step,
                    resample_interval=resample,
                    ef_config=cfg
                )
                labeled = label_dataframes(dfs, queries[key]["labels"])
            markdown = dataframes_to_markdown(labeled)
            pack = build_context_pack(labeled, top_n=15)
            ctx = json.dumps({
                "domain": key,
                "time_range": {"start": start_ts, "end": end_ts},
                **pack
            }, ensure_ascii=False)
            domain_data[key] = {"labeled": labeled, "markdown": markdown, "pack": pack, "ctx": ctx}
        except Exception as e:
            logger.error(f"Domain '{key}' build failed: {e}")
            domain_data[key] = {"labeled": [], "markdown": "", "pack": {"sections": []}, "ctx": json.dumps({"domain": key, "sections": []}, ensure_ascii=False)}

    storage_cfg = ((cfg.get("storage", {}) or {}).get("timescale") or (CONFIG.get("storage", {}) or {}).get("timescale") or {})

    # Сохранение метрик доменов в TimescaleDB
    if save_to_db:
        try:
            for key in domain_keys:
                dd = domain_data.get(key, {})
                labeled = dd.get("labeled") or []
                save_domain_labeled(
                    domain_key=key,
                    domain_conf=queries.get(key, {}),
                    labeled_dfs=labeled,
                    run_meta={
                        **(run_meta or {}),
                        "start_ms": int((run_meta or {}).get("start_ms") or int(start_ts * 1000)),
                        "end_ms": int((run_meta or {}).get("end_ms") or int(end_ts * 1000)),
                    },
                    storage_cfg=storage_cfg
                )
        except Exception as e:
            logger.error(f"Failed to save domain data to TimescaleDB: {e}")

    if only_collect:
        return {}

    prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    prompt_jvm = read_prompt_from_file(os.path.join(prompt_dir, "jvm_prompt.txt"))
    prompt_database = read_prompt_from_file(os.path.join(prompt_dir, "database_prompt.txt"))
    prompt_kafka = read_prompt_from_file(os.path.join(prompt_dir, "kafka_prompt.txt"))
    prompt_microservices = read_prompt_from_file(os.path.join(prompt_dir, "microservices_prompt.txt"))
    prompt_hard_resources = read_prompt_from_file(os.path.join(prompt_dir, "hard_resources_prompt.txt"))
    prompt_overall = read_prompt_from_file(os.path.join(prompt_dir, "overall_prompt.txt"))
    prompt_lt_framework = read_prompt_from_file(os.path.join(prompt_dir, "lt_framework_prompt.txt")) if os.path.exists(os.path.join(prompt_dir, "lt_framework_prompt.txt")) else "Проанализируйте метрики инструмента нагрузочного тестирования (lt_framework)."
    if isinstance(prompts_override, dict):
        prompt_jvm = prompts_override.get("jvm", prompt_jvm)
        prompt_database = prompts_override.get("database", prompt_database)
        prompt_kafka = prompts_override.get("kafka", prompt_kafka)
        prompt_microservices = prompts_override.get("microservices", prompt_microservices)
        prompt_hard_resources = prompts_override.get("hard_resources", prompt_hard_resources)
        prompt_overall = prompts_override.get("overall", prompt_overall)
        prompt_lt_framework = prompts_override.get("lt_framework", prompt_lt_framework)

    include_tables = bool(((CONFIG.get("llm", {}) or {}).get("include_markdown_tables_in_context", False)))
    jvm_full_data = domain_data["jvm"]["markdown"]; jvm_pack = domain_data["jvm"]["pack"]; jvm_ctx = domain_data["jvm"]["ctx"]
    database_full_data = domain_data["database"]["markdown"]; database_pack = domain_data["database"]["pack"]; database_ctx = domain_data["database"]["ctx"]
    kafka_full_data = domain_data["kafka"]["markdown"]; kafka_pack = domain_data["kafka"]["pack"]; kafka_ctx = domain_data["kafka"]["ctx"]
    ms_full_data = domain_data["microservices"]["markdown"]; ms_pack = domain_data["microservices"]["pack"]
    hr_full_data = domain_data["hard_resources"]["markdown"]; hr_pack = domain_data["hard_resources"]["pack"]; hr_ctx = domain_data["hard_resources"]["ctx"]
    lt_full_data = domain_data.get("lt_framework", {}).get("markdown", "")
    lt_pack = domain_data.get("lt_framework", {}).get("pack", {})

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

    from concurrent.futures import ThreadPoolExecutor, as_completed
    domains_jobs = []
    if "jvm" in domain_keys and _is_enabled("jvm"):
        domains_jobs.append(("jvm", prompt_jvm, jvm_ctx))
    if "database" in domain_keys and _is_enabled("database"):
        domains_jobs.append(("database", prompt_database, database_ctx))
    if "kafka" in domain_keys and _is_enabled("kafka"):
        domains_jobs.append(("kafka", prompt_kafka, kafka_ctx))
    if "microservices" in domain_keys and _is_enabled("microservices"):
        domains_jobs.append(("microservices", prompt_microservices, ms_ctx))
    if "hard_resources" in domain_keys and _is_enabled("hard_resources"):
        domains_jobs.append(("hard_resources", prompt_hard_resources, hr_ctx))
    if "lt_framework" in domain_keys and _is_enabled("lt_framework"):
        lt_ctx = domain_data.get("lt_framework", {}).get("ctx", json.dumps({"domain":"lt_framework","sections":[]}, ensure_ascii=False))
        domains_jobs.append(("lt_framework", prompt_lt_framework, lt_ctx))
    results_map: dict[str, tuple[str, object, dict]] = {}
    with ThreadPoolExecutor(max_workers=min(5, len(domains_jobs))) as executor:
        future_to_key = {executor.submit(llm_two_pass_self_consistency, p, c, 3, True): k for (k, p, c) in domains_jobs}
        for fut in as_completed(future_to_key):
            key = future_to_key[fut]
            try:
                text, parsed, score = fut.result()
                results_map[key] = (text, parsed, score)
            except Exception as e:
                logger.error(f"LLM {key} analysis failed: {e}")
                results_map[key] = ("{}", None, {})

    def _result_or_blank(key: str):
        if _is_enabled(key):
            return results_map.get(key, ("{}", None, {}))
        return ("", None, {})

    answer_jvm, jvm_parsed, jvm_score = _result_or_blank("jvm")
    answer_database, database_parsed, database_score = _result_or_blank("database")
    answer_kafka, kafka_parsed, kafka_score = _result_or_blank("kafka")
    answer_ms, ms_parsed, ms_score = _result_or_blank("microservices")
    answer_hr, hr_parsed, hr_score = _result_or_blank("hard_resources")
    if "lt_framework" in domain_keys:
        answer_lt, lt_parsed, lt_score = _result_or_blank("lt_framework")
    else:
        answer_lt, lt_parsed, lt_score = ("", None, {})

    merged_prompt_overall = (
        prompt_overall
        .replace("{answer_jvm}", answer_jvm)
        .replace("{answer_database}", answer_database)
        .replace("{answer_kafka}", answer_kafka)
        .replace("{answer_microservices}", answer_ms)
        .replace("{answer_hard_resources}", answer_hr)
        .replace("{answer_lt_framework}", answer_lt)
    )

    base_ctx = {
        "time_range": {"start": start_ts, "end": end_ts},
        "domains": {
            "jvm": jvm_pack,
            "database": database_pack,
            "kafka": kafka_pack,
            "microservices": ms_pack,
            "hard_resources": hr_pack
        }
    }
    if "lt_framework" in domain_keys:
        base_ctx["domains"]["lt_framework"] = lt_pack
    if include_tables:
        base_ctx["domains_tables_markdown"] = {
            "jvm": jvm_full_data,
            "database": database_full_data,
            "kafka": kafka_full_data,
            "microservices": ms_full_data,
            "hard_resources": hr_full_data,
        }
        if "lt_framework" in domain_keys:
            base_ctx["domains_tables_markdown"]["lt_framework"] = lt_full_data
    overall_ctx = json.dumps(base_ctx, ensure_ascii=False)
    final_answer, final_parsed, final_score = llm_two_pass_self_consistency(user_prompt=merged_prompt_overall, data_context=overall_ctx, k=3, return_scores=True)

    def _compose_text(full_md: str, header: str, analysis: str) -> str:
        if include_tables:
            return f"{full_md}\n\n{header}\n{analysis}"
        return analysis

    results = {
        "jvm": _compose_text(jvm_full_data, "Анализ JVM:", answer_jvm),
        "database": _compose_text(database_full_data, "Анализ Database:", answer_database),
        "kafka": _compose_text(kafka_full_data, "Анализ Kafka:", answer_kafka),
        "ms": _compose_text(ms_full_data, "Анализ микросервисов:", answer_ms),
        "hard_resources": _compose_text(hr_full_data, "Анализ ресурсов (CPU/MEM/Disk):", answer_hr),
        "lt_framework": answer_lt,
        "final": final_answer,
        "jvm_parsed": (jvm_parsed.dict() if jvm_parsed else None),
        "database_parsed": (database_parsed.dict() if database_parsed else None),
        "kafka_parsed": (kafka_parsed.dict() if kafka_parsed else None),
        "ms_parsed": (ms_parsed.dict() if ms_parsed else None),
        "hard_resources_parsed": (hr_parsed.dict() if hr_parsed else None),
        "lt_framework_parsed": (lt_parsed.dict() if lt_parsed else None) if lt_parsed else None,
        "final_parsed": (final_parsed.dict() if final_parsed else None),
        "scores": {
            "jvm": jvm_score,
            "database": database_score,
            "kafka": kafka_score,
            "microservices": ms_score,
            "hard_resources": hr_score,
            **({"lt_framework": lt_score} if "lt_framework" in domain_keys else {}),
            "final": final_score,
        }
    }

    # Сохранение LLM результатов в отдельную таблицу (если включено)
    if save_to_db:
        try:
            save_llm_results(
                results=results,
                run_meta={
                    **(run_meta or {}),
                    "start_ms": int((run_meta or {}).get("start_ms") or int(start_ts * 1000)),
                    "end_ms": int((run_meta or {}).get("end_ms") or int(end_ts * 1000)),
                },
                storage_cfg=storage_cfg
            )
        except Exception as e:
            logger.error(f"Failed to save LLM results: {e}")

    return results


def label_dataframes(dfs: List[pd.DataFrame], labels: List[str]) -> List[Dict[str, object]]:
    if len(dfs) != len(labels):
        raise ValueError("Количество DataFrame и количество меток не совпадает!")
    labeled_list = []
    for df, label in zip(dfs, labels):
        labeled_list.append({
            "label": label,
            "df": df
        })
    return labeled_list


