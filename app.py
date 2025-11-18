from flask import Flask, request, jsonify, render_template, send_file, make_response
from update_page import update_report  # Замените на имя файла, где находится `update_report`
from settings import CONFIG  # Базовые настройки (унифицированные)
from metrics_config import METRICS_CONFIG  # Конфигурация метрик
from datetime import datetime
import threading
import uuid
import psycopg2
import os
import json
from AI.db_store import _ensure_llm_reports_table, _ensure_engineer_reports_table  # для гарантий наличия таблиц



app = Flask(__name__)

# -------------------- Project area helpers (service-based) --------------------
def _active_project_area() -> str | None:
    try:
        val = request.cookies.get('project_area')
        return val if isinstance(val, str) and val.strip() else None
    except Exception:
        return None

# -------------------- Runtime settings overrides --------------------
CONFIG_RUNTIME_PATH = os.path.join(os.path.dirname(__file__), 'settings_runtime.json')
METRICS_RUNTIME_PATH = os.path.join(os.path.dirname(__file__), 'metrics_config_runtime.json')

def _deep_merge_dicts(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out.get(k) or {}, v)
        else:
            out[k] = v
    return out

try:
    if os.path.exists(CONFIG_RUNTIME_PATH):
        with open(CONFIG_RUNTIME_PATH, 'r', encoding='utf-8') as f:
            _rt = json.load(f)
        # Мягкое объединение в память (для маршрутов этого файла)
        CONFIG.update(_deep_merge_dicts(CONFIG, _rt))
except Exception:
    pass

# Активный metrics_config с учётом runtime-оверрайда
def _active_metrics_config() -> dict:
    raw = {}
    try:
        from metrics_config import METRICS_CONFIG as BASE
    except Exception:
        BASE = {}
    raw = BASE
    try:
        if os.path.exists(METRICS_RUNTIME_PATH):
            with open(METRICS_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                override = json.load(f)
            if isinstance(override, dict):
                raw = _deep_merge_dicts(BASE, override)
    except Exception:
        pass
    return _normalize_metrics_config(raw)


def _service_area_map() -> dict:
    mapping: dict[str, str] = {}
    per_area = _per_area_config()
    for area_name, cfg in per_area.items():
        services = cfg.get('services')
        if isinstance(services, dict):
            for sid in services.keys():
                mapping[sid] = area_name
    return mapping


def _normalize_metrics_config(raw: dict | None) -> dict:
    normalized: dict[str, dict] = {}
    service_to_area = _service_area_map()

    def ensure(area_name: str) -> dict:
        if area_name not in normalized or not isinstance(normalized[area_name], dict):
            normalized[area_name] = {"services": {}}
        if "services" not in normalized[area_name] or not isinstance(normalized[area_name]["services"], dict):
            normalized[area_name]["services"] = {}
        return normalized[area_name]

    for area_name in _per_area_config().keys():
        ensure(area_name)

    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("services"), dict):
                entry = ensure(key)
                entry_services = entry.get("services", {})
                entry_services.update(value.get("services") or {})
                entry["services"] = entry_services
                for meta_key, meta_val in value.items():
                    if meta_key != "services":
                        entry[meta_key] = meta_val
                continue
            target_area = service_to_area.get(key) or value.get("area") or key
            entry = ensure(target_area)
            entry["services"][key] = value
    return normalized


def _metrics_services_for_area(area_name: str | None) -> dict:
    metrics = _active_metrics_config() or {}
    if not area_name:
        return {}
    entry = metrics.get(area_name) or {}
    services = entry.get('services')
    return services if isinstance(services, dict) else {}


def _metrics_service_entry(service_id: str | None) -> tuple[str | None, dict]:
    if not service_id:
        return None, {}
    metrics = _active_metrics_config() or {}
    for area_name, cfg in metrics.items():
        services = cfg.get('services')
        if isinstance(services, dict) and service_id in services:
            return area_name, services.get(service_id) or {}
    # наследуем старый формат: ключ совпадает с сервисом
    legacy = metrics.get(service_id)
    if isinstance(legacy, dict):
        services = legacy.get('services')
        if isinstance(services, dict) and service_id in services:
            return service_id, services.get(service_id) or {}
    return None, {}

# -------------------- Runtime helpers for per-area/service data --------------------
def _load_settings_runtime_data() -> dict:
    try:
        if os.path.exists(CONFIG_RUNTIME_PATH):
            with open(CONFIG_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_settings_runtime_data(payload: dict) -> None:
    try:
        with open(CONFIG_RUNTIME_PATH, 'w', encoding='utf-8') as f:
            json.dump(payload or {}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _per_area_config() -> dict:
    data = _load_settings_runtime_data()
    per_area = data.get('per_area') if isinstance(data, dict) else {}
    return per_area if isinstance(per_area, dict) else {}


def _area_entry(area_name: str | None) -> dict:
    if not area_name:
        return {}
    entry = _per_area_config().get(area_name)
    return entry if isinstance(entry, dict) else {}


def _services_map_for_area(area_name: str | None) -> dict:
    entry = _area_entry(area_name)
    services = entry.get('services')
    return services if isinstance(services, dict) else {}


def _available_domain_keys(cfg: dict | None = None) -> list[str]:
    config = cfg or CONFIG
    queries = config.get('queries') or {}
    preferred_order = ['jvm', 'database', 'kafka', 'microservices', 'hard_resources', 'lt_framework']
    result = [d for d in preferred_order if d in queries]
    for key in queries.keys():
        if key not in result:
            result.append(key)
    return result


def _resolve_services_for_area(area_name: str | None) -> list[str]:
    if not area_name:
        return []
    services_map = _services_map_for_area(area_name)
    if services_map:
        return list(services_map.keys())
    metrics_services = _metrics_services_for_area(area_name)
    if metrics_services:
        return list(metrics_services.keys())
    # fallback: treat area as сервис напрямую
    if area_name:
        return [area_name]
    return []


def _resolve_services_filter(area_name: str | None) -> list[str]:
    services = _resolve_services_for_area(area_name)
    if services:
        return services
    if area_name:
        return [area_name]
    return []


def _find_area_for_service(service_id: str | None) -> str | None:
    if not service_id:
        return None
    per_area = _per_area_config()
    for area_name, cfg in per_area.items():
        services = cfg.get('services')
        if isinstance(services, dict) and service_id in services:
            return area_name
    metrics = _active_metrics_config() or {}
    for area_name, cfg in metrics.items():
        services = cfg.get('services')
        if isinstance(services, dict) and service_id in services:
            return area_name
    return None


def _service_meta(area_name: str | None, service_id: str | None) -> dict:
    if not area_name or not service_id:
        return {}
    services = _services_map_for_area(area_name)
    entry = services.get(service_id) if isinstance(services, dict) else {}
    return entry if isinstance(entry, dict) else {}


def _service_disabled_domains(area_name: str | None, service_id: str | None) -> list[str]:
    meta = _service_meta(area_name, service_id)
    disabled = meta.get('disabled_domains') if isinstance(meta, dict) else []
    return [d for d in disabled if isinstance(d, str)]


def _list_project_areas() -> list[dict]:
    per_area = _per_area_config()
    if per_area:
        areas = []
        for name, cfg in per_area.items():
            title = ''
            if isinstance(cfg, dict):
                title = cfg.get('title') if isinstance(cfg.get('title'), str) else ''
            areas.append({"id": name, "title": title or name})
        return areas
    metrics = _active_metrics_config() or {}
    return [{"id": name, "title": name} for name in metrics.keys()]


def _prompt_templates_for_scope(area: str | None, service: str | None = None) -> dict:
    base = _load_base_prompts()
    area_entry = _area_entry(area)
    area_prompts = area_entry.get('prompts') if isinstance(area_entry, dict) else {}
    service_prompts = {}
    if service:
        meta = _service_meta(area, service)
        service_prompts = meta.get('prompts') if isinstance(meta, dict) else {}
    out: dict[str, str] = {}
    for domain in PROMPT_DOMAIN_FILES.keys():
        if isinstance(service_prompts, dict) and isinstance(service_prompts.get(domain), str) and service_prompts.get(domain).strip():
            out[domain] = service_prompts.get(domain, '')
        elif isinstance(area_prompts, dict) and isinstance(area_prompts.get(domain), str) and area_prompts.get(domain).strip():
            out[domain] = area_prompts.get(domain, '')
        else:
            out[domain] = base.get(domain, '')
    return out


def _disabled_domains_payload(area: str | None, service: str | None) -> list[str]:
    if not service:
        return []
    return _service_disabled_domains(area, service)

# -------------------- LLM prompts (base + per-area overrides) --------------------
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'AI', 'prompts')
PROMPT_DOMAIN_FILES = {
    'overall': 'overall_prompt.txt',
    'database': 'database_prompt.txt',
    'kafka': 'kafka_prompt.txt',
    'microservices': 'microservices_prompt.txt',
    'jvm': 'jvm_prompt.txt',
    'hard_resources': 'hard_resources_prompt.txt',
    'lt_framework': 'lt_framework_prompt.txt',
    'judge': 'judge_prompt.txt',
    'critic': 'critic_prompt.txt',
}
LOCKED_PROMPT_DOMAINS = {'judge', 'critic'}

def _load_base_prompts() -> dict:
    prompts: dict[str, str] = {}
    for domain, fname in PROMPT_DOMAIN_FILES.items():
        try:
            p = os.path.join(PROMPTS_DIR, fname)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    prompts[domain] = f.read()
            else:
                prompts[domain] = ''
        except Exception:
            prompts[domain] = ''
    return prompts

def _active_area_prompts(area: str | None, service: str | None = None) -> dict:
    if service and not area:
        area = _find_area_for_service(service)
    return _prompt_templates_for_scope(area, service)

# Простенький менеджер задач для прогресса и ссылки на отчёт
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# Рендеринг главной страницы с формой
@app.route('/')
def home():
    return render_template('dashboard.html')

# Страница сравнения релизов
@app.route('/compare')
def compare_page():
    return render_template('compare.html')

# Страница архива отчётов (список запусков)
@app.route('/reports')
def reports_page():
    return render_template('archive.html')

@app.route('/reports/<run_name>')
def reports_page_run(run_name: str):
    # Рендерим ту же страницу; выбор запуска произойдёт на клиенте по части URL
    return render_template('reports.html')

@app.route('/reports/<service>/<run_name>')
def reports_page_service_run(service: str, run_name: str):
    # Устанавливаем область в cookie и отдаём страницу отчёта
    resp = make_response(render_template('reports.html'))
    try:
        area = _find_area_for_service(service) or service
        resp.set_cookie('project_area', area, max_age=60*60*24*365, samesite='Lax')
    except Exception:
        pass
    return resp

@app.route('/new')
def new_report_page():
    return render_template('index.html')

# Отдача логотипа из templates (временное решение, чтобы не переносить файл)
@app.route('/assets/logo.png')
def asset_logo():
    logo_path = os.path.join(app.root_path, 'templates', 'logo.png')
    return send_file(logo_path, mimetype='image/png')

# Вспомогательная функция для конвертации времени
def convert_to_timestamp(date_str):
    """Конвертирует строку даты и времени в миллисекундный timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
    return int(dt.timestamp() * 1000)

@app.route('/services', methods=['GET'])
def get_services():
    area = (request.args.get('area') or '').strip()
    if not area:
        area = _active_project_area() or ''
    services_map = _services_map_for_area(area)
    metrics = _active_metrics_config() or {}
    area_metrics_services = _metrics_services_for_area(area)
    payload = []
    if services_map:
        for sid, meta in services_map.items():
            title = sid
            if isinstance(meta, dict):
                title = (meta.get('title') if isinstance(meta.get('title'), str) else '') or title
                disabled = [d for d in (meta.get('disabled_domains') or []) if isinstance(d, str)]
            else:
                disabled = []
            payload.append({
                "id": sid,
                "title": title,
                "disabled_domains": disabled,
            })
    elif area_metrics_services:
        for sid in area_metrics_services.keys():
            payload.append({
                "id": sid,
                "title": sid,
                "disabled_domains": [],
            })
    else:
        for area_name, cfg in metrics.items():
            services = cfg.get('services')
            if isinstance(services, dict):
                for sid in services.keys():
                    payload.append({"id": sid, "title": sid, "disabled_domains": []})
    return jsonify({
        "area": area,
        "services": payload,
        "domains": _available_domain_keys()
    }), 200

# -------------------- TimescaleDB helpers --------------------
def _ts_conn():
    cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
    return psycopg2.connect(
        host=cfg.get("host"),
        port=cfg.get("port"),
        dbname=cfg.get("dbname"),
        user=cfg.get("user"),
        password=cfg.get("password"),
        sslmode=cfg.get("sslmode", "prefer"),
    )

# -------------------- Helpers --------------------
def _series_key_for(domain: str, query_label: str, default_key: str = "application") -> str:
    try:
        q = (CONFIG.get("queries", {}) or {}).get(domain, {})
        labels = list(q.get("labels", []) or [])
        keys_list = list(q.get("label_keys_list", []) or [])
        if query_label in labels:
            idx = labels.index(query_label)
            if idx < len(keys_list):
                keys = list(keys_list[idx] or [])
                if keys:
                    return str(keys[0])
        # fallback: если в домене одна группа ключей — берём первый
        if keys_list:
            keys = list(keys_list[0] or [])
            if keys:
                return str(keys[0])
    except Exception:
        pass
    return default_key
# -------------------- Dashboard API --------------------
@app.route('/dashboard_data', methods=['GET'])
def dashboard_data():
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        services_filter = _resolve_services_filter(pa)
        last_run = None
        verdict_counts = {"Успешно": 0, "Есть риски": 0, "Провал": 0, "Недостаточно данных": 0}
        with conn, conn.cursor() as cur:
            # Последний запуск по времени создания итогового отчёта (final)
            if services_filter:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, verdict, created_at
                    FROM {schema}.{table}
                    WHERE domain = 'final' AND service = ANY(%s)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (services_filter,)
                )
            else:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, verdict, created_at
                    FROM {schema}.{table}
                    WHERE domain = 'final'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
            r = cur.fetchone()
            if r:
                last_run = {
                    "run_name": r[0],
                    "service": r[1],
                    "start_ms": int(r[2]) if r[2] is not None else None,
                    "end_ms": int(r[3]) if r[3] is not None else None,
                    "verdict": r[4],
                    "created_at": r[5].isoformat() if r[5] else None,
                }

            # Распределение статусов по последнему финальному вердикту каждого запуска
            if services_filter:
                cur.execute(
                    f"""
                    WITH ranked AS (
                      SELECT run_name, verdict, created_at,
                             ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                      FROM {schema}.{table}
                      WHERE domain = 'final' AND service = ANY(%s)
                    )
                    SELECT COALESCE(verdict, 'Недостаточно данных') AS v, COUNT(*) AS cnt
                    FROM ranked
                    WHERE rn = 1
                    GROUP BY v
                    """,
                    (services_filter,)
                )
            else:
                cur.execute(
                    f"""
                    WITH ranked AS (
                      SELECT run_name, verdict, created_at,
                             ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                      FROM {schema}.{table}
                      WHERE domain = 'final'
                    )
                    SELECT COALESCE(verdict, 'Недостаточно данных') AS v, COUNT(*) AS cnt
                    FROM ranked
                    WHERE rn = 1
                    GROUP BY v
                    """
                )
            rows = cur.fetchall()
            for v, cnt in rows:
                if v in verdict_counts:
                    verdict_counts[v] = int(cnt)
                else:
                    # неизвестные значения складываем в "Недостаточно данных"
                    verdict_counts["Недостаточно данных"] += int(cnt)
        conn.close()
        return jsonify({"last_run": last_run, "verdict_counts": verdict_counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Compare summary (P95) --------------------
@app.route('/compare_summary', methods=['GET'])
def compare_summary():
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    if not all([run_a, run_b, domain]):
        return jsonify({"error": "run_a, run_b, domain обязательны"}), 400
    try:
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    """
                    WITH raw AS (
                      SELECT query_label, run_name, value
                      FROM public.metrics
                      WHERE run_name IN (%s, %s)
                        AND domain = %s
                        AND service = ANY(%s)
                    ), p AS (
                      SELECT query_label,
                             run_name,
                             percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
                      FROM raw
                      GROUP BY query_label, run_name
                    )
                    SELECT query_label,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_a,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_b
                    FROM p
                    GROUP BY query_label
                    ORDER BY query_label
                    """,
                    (run_a, run_b, domain, services_filter, run_a, run_b)
                )
            else:
                cur.execute(
                    """
                    WITH raw AS (
                      SELECT query_label, run_name, value
                      FROM public.metrics
                      WHERE run_name IN (%s, %s)
                        AND domain = %s
                    ), p AS (
                      SELECT query_label,
                             run_name,
                             percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
                      FROM raw
                      GROUP BY query_label, run_name
                    )
                    SELECT query_label,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_a,
                           MAX(p95) FILTER (WHERE run_name = %s) AS p95_b
                    FROM p
                    GROUP BY query_label
                    ORDER BY query_label
                    """,
                    (run_a, run_b, domain, run_a, run_b)
                )
            rows = cur.fetchall()
        conn.close()

        def _safe_float(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        out = []
        for ql, a, b in rows:
            a_f = _safe_float(a)
            b_f = _safe_float(b)
            trend = None
            if a_f is not None and b_f is not None:
                denom = abs(a_f) if abs(a_f) > 1e-9 else 1.0
                trend = ((b_f - a_f) / denom) * 100.0
            out.append({
                "query_label": ql,
                "p95_a": a_f,
                "p95_b": b_f,
                "trend_pct": trend
            })
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#
# -------------------- Compare metric series summary (P95 by series) --------------------
@app.route('/compare_metric_summary', methods=['GET'])
def compare_metric_summary():
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    if not all([run_a, run_b, domain, ql]):
        return jsonify({"error": "run_a, run_b, domain, query_label обязательны"}), 400
    try:
        # Автоматически подобрать ключ серии, если не задан
        if not series_key or series_key == "auto":
            series_key = _series_key_for(domain, ql, default_key="application")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        svc_clause = " AND m.service = ANY(%s)" if services_filter else ""
        sql_common = f"""
          SELECT
            regexp_replace(m.series, '.*{series_key}=([^|]+).*', '\\1') AS series_name,
            m.run_name,
            m.value
          FROM public.metrics m
          WHERE m.run_name IN (%s, %s)
            AND m.domain = %s
            AND m.query_label = %s
            {svc_clause}
        """
        base_params = [run_a, run_b, domain, ql]
        if services_filter:
            base_params.append(services_filter)
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    f"""
                      WITH base AS ({sql_common}),
                      p AS (
                        SELECT
                          series_name,
                          run_name,
                          percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
                        FROM base
                        GROUP BY series_name, run_name
                      )
                      SELECT
                        series_name,
                        MAX(p95) FILTER (WHERE run_name = %s) AS p95_a,
                        MAX(p95) FILTER (WHERE run_name = %s) AS p95_b
                      FROM p
                      GROUP BY series_name
                      ORDER BY series_name
                    """,
                    (*base_params, run_a, run_b)
                )
            else:
                cur.execute(
                    f"""
                      WITH base AS ({sql_common}),
                      p AS (
                        SELECT
                          series_name,
                          run_name,
                          percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95
                        FROM base
                        GROUP BY series_name, run_name
                      )
                      SELECT
                        series_name,
                        MAX(p95) FILTER (WHERE run_name = %s) AS p95_a,
                        MAX(p95) FILTER (WHERE run_name = %s) AS p95_b
                      FROM p
                      GROUP BY series_name
                      ORDER BY series_name
                    """,
                    (*base_params, run_a, run_b)
                )
            rows = cur.fetchall()
        conn.close()

        def _safe_float(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        out = []
        for s, a, b in rows:
            a_f = _safe_float(a)
            b_f = _safe_float(b)
            trend = None
            if a_f is not None and b_f is not None:
                denom = abs(a_f) if abs(a_f) > 1e-9 else 1.0
                trend = ((b_f - a_f) / denom) * 100.0
            out.append({
                "series": s,
                "p95_a": a_f,
                "p95_b": b_f,
                "trend_pct": trend
            })
        return jsonify({"query_label": ql, "series_key": series_key, "rows": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# -------------------- Готовые отчёты (LLM) --------------------
@app.route('/llm_reports', methods=['GET'])
def llm_reports():
    run_name = request.args.get("run_name")
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, domain, verdict, text, parsed, scores, created_at
                    FROM {schema}.{table}
                    WHERE run_name = %s AND domain <> 'engineer' AND service = ANY(%s)
                    ORDER BY created_at DESC, domain
                    """,
                    (run_name, services_filter)
                )
            else:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, domain, verdict, text, parsed, scores, created_at
                    FROM {schema}.{table}
                    WHERE run_name = %s AND domain <> 'engineer'
                    ORDER BY created_at DESC, domain
                    """,
                    (run_name,)
                )
            rows = cur.fetchall()
        conn.close()
        data = []
        for r in rows:
            data.append({
                "run_name": r[0],
                "service": r[1],
                "start_ms": int(r[2]) if r[2] is not None else None,
                "end_ms": int(r[3]) if r[3] is not None else None,
                "domain": r[4],
                "verdict": r[5],
                "text": r[6],
                "parsed": r[7],
                "scores": r[8],
                "created_at": r[9].isoformat() if r[9] else None,
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- Compare API (TimescaleDB) --------------------

@app.route('/runs', methods=['GET'])
def list_runs():
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        llm_table = cfg.get("llm_table", "llm_reports")
        pa = _active_project_area()
        q = request.args.get('q', '').strip()
        offset = int(request.args.get('offset', '0') or 0)
        limit = int(request.args.get('limit', '20') or 20)
        sort = (request.args.get('sort', 'end_time') or 'end_time').lower()
        direction = (request.args.get('dir', 'desc') or 'desc').lower()
        # Безопасная сортировка
        sort_map = {
            'run_name': 'run_name',
            'service': 'service',
            'start_time': 'start_time',
            'end_time': 'end_time',
        }
        sort_sql = sort_map.get(sort, 'end_time')
        dir_sql = 'ASC' if direction == 'asc' else 'DESC'

        params = []
        where_q = ''
        if q:
            where_q = " AND (run_name ILIKE %s OR service ILIKE %s)"
            like = f"%{q}%"
            params.extend([like, like])

        conn = _ts_conn()
        test_type_supported = True
        with conn, conn.cursor() as cur:
            try:
                if services_filter:
                    cur.execute(
                        f"""
                        WITH base AS (
                          SELECT run_name,
                                 MIN("time") AS start_time,
                                 MAX("time") AS end_time,
                                 COALESCE(MAX(service), '') AS service
                          FROM public.metrics
                          WHERE run_name IS NOT NULL AND run_name <> '' AND service = ANY(%s)
                          {where_q}
                          GROUP BY run_name
                        ), final AS (
                          SELECT run_name, verdict, created_at, test_type,
                                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                          FROM {schema}.{llm_table}
                          WHERE domain = 'final' AND service = ANY(%s)
                        )
                        SELECT b.run_name, b.start_time, b.end_time, b.service,
                               f.verdict, f.created_at AS report_created_at, f.test_type
                        FROM base b
                        LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                        ORDER BY {sort_sql} {dir_sql}
                        OFFSET %s LIMIT %s
                        """,
                        (services_filter, *params, services_filter, offset, limit)
                    )
                else:
                    cur.execute(
                        f"""
                        WITH base AS (
                          SELECT run_name,
                                 MIN("time") AS start_time,
                                 MAX("time") AS end_time,
                                 COALESCE(MAX(service), '') AS service
                          FROM public.metrics
                          WHERE run_name IS NOT NULL AND run_name <> ''
                          {where_q}
                          GROUP BY run_name
                        ), final AS (
                          SELECT run_name, verdict, created_at, test_type,
                                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                          FROM {schema}.{llm_table}
                          WHERE domain = 'final'
                        )
                        SELECT b.run_name, b.start_time, b.end_time, b.service,
                               f.verdict, f.created_at AS report_created_at, f.test_type
                        FROM base b
                        LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                        ORDER BY {sort_sql} {dir_sql}
                        OFFSET %s LIMIT %s
                        """,
                        (*params, offset, limit)
                    )
                rows = cur.fetchall()
            except Exception:
                # Fallback для старых таблиц без test_type
                test_type_supported = False
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    if services_filter:
                        cur.execute(
                            f"""
                            WITH base AS (
                              SELECT run_name,
                                     MIN("time") AS start_time,
                                     MAX("time") AS end_time,
                                     COALESCE(MAX(service), '') AS service
                              FROM public.metrics
                              WHERE run_name IS NOT NULL AND run_name <> '' AND service = ANY(%s)
                              {where_q}
                              GROUP BY run_name
                            ), final AS (
                              SELECT run_name, verdict, created_at,
                                     ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                              FROM {schema}.{llm_table}
                              WHERE domain = 'final' AND service = ANY(%s)
                            )
                            SELECT b.run_name, b.start_time, b.end_time, b.service,
                                   f.verdict, f.created_at AS report_created_at
                            FROM base b
                            LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                            ORDER BY {sort_sql} {dir_sql}
                            OFFSET %s LIMIT %s
                            """,
                            (services_filter, *params, services_filter, offset, limit)
                        )
                    else:
                        cur.execute(
                            f"""
                            WITH base AS (
                              SELECT run_name,
                                     MIN("time") AS start_time,
                                     MAX("time") AS end_time,
                                     COALESCE(MAX(service), '') AS service
                              FROM public.metrics
                              WHERE run_name IS NOT NULL AND run_name <> ''
                              {where_q}
                              GROUP BY run_name
                            ), final AS (
                              SELECT run_name, verdict, created_at,
                                     ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                              FROM {schema}.{llm_table}
                              WHERE domain = 'final'
                            )
                            SELECT b.run_name, b.start_time, b.end_time, b.service,
                                   f.verdict, f.created_at AS report_created_at
                            FROM base b
                            LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                            ORDER BY {sort_sql} {dir_sql}
                            OFFSET %s LIMIT %s
                            """,
                            (*params, offset, limit)
                        )
                    rows = cur.fetchall()
                except Exception:
                    # Минимальный фолбэк без verdict/test_type
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    if services_filter:
                        cur.execute(
                            f"""
                            WITH base AS (
                              SELECT run_name,
                                     MIN("time") AS start_time,
                                     MAX("time") AS end_time,
                                     COALESCE(MAX(service), '') AS service
                              FROM public.metrics
                              WHERE run_name IS NOT NULL AND run_name <> '' AND service = ANY(%s)
                              {where_q}
                              GROUP BY run_name
                            ), final AS (
                              SELECT run_name, created_at,
                                     ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                              FROM {schema}.{llm_table}
                              WHERE domain = 'final' AND service = ANY(%s)
                            )
                            SELECT b.run_name, b.start_time, b.end_time, b.service,
                                   NULL::TEXT AS verdict, f.created_at AS report_created_at
                            FROM base b
                            LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                            ORDER BY {sort_sql} {dir_sql}
                            OFFSET %s LIMIT %s
                            """,
                            (services_filter, *params, services_filter, offset, limit)
                        )
                    else:
                        cur.execute(
                            f"""
                            WITH base AS (
                              SELECT run_name,
                                     MIN("time") AS start_time,
                                     MAX("time") AS end_time,
                                     COALESCE(MAX(service), '') AS service
                              FROM public.metrics
                              WHERE run_name IS NOT NULL AND run_name <> ''
                              {where_q}
                              GROUP BY run_name
                            ), final AS (
                              SELECT run_name, created_at,
                                     ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                              FROM {schema}.{llm_table}
                              WHERE domain = 'final'
                            )
                            SELECT b.run_name, b.start_time, b.end_time, b.service,
                                   NULL::TEXT AS verdict, f.created_at AS report_created_at
                            FROM base b
                            LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                            ORDER BY {sort_sql} {dir_sql}
                            OFFSET %s LIMIT %s
                            """,
                            (*params, offset, limit)
                        )
                    rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            item = {
                "run_name": r[0],
                "start_time": r[1].isoformat() if r[1] else None,
                "end_time": r[2].isoformat() if r[2] else None,
                "service": r[3],
                "verdict": r[4],
                "report_created_at": (r[5].isoformat() if r[5] else None),
            }
            if test_type_supported:
                item["test_type"] = r[6] or ""
            else:
                item["test_type"] = ""
            out.append(item)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/domains_schema', methods=['GET'])
def domains_schema():
    try:
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        with conn, conn.cursor() as cur:
            if services_filter:
                cur.execute(
                    """
                    SELECT domain, query_label, COUNT(*) AS cnt
                    FROM public.metrics
                    WHERE service = ANY(%s)
                    GROUP BY domain, query_label
                    ORDER BY domain, query_label
                    """,
                    (services_filter,)
                )
            else:
                cur.execute(
                    """
                    SELECT domain, query_label, COUNT(*) AS cnt
                    FROM public.metrics
                    GROUP BY domain, query_label
                    ORDER BY domain, query_label
                    """
                )
            rows = cur.fetchall()
        conn.close()
        out = {}
        for d, ql, cnt in rows:
            out.setdefault(d, []).append({"query_label": ql, "count": int(cnt)})
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compare_series', methods=['GET'])
def compare_series():
    run_a = request.args.get("run_a")
    run_b = request.args.get("run_b")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    align = request.args.get("align", "offset")  # absolute|offset
    if not all([run_a, run_b, domain, ql]):
        return jsonify({"error": "run_a, run_b, domain, query_label обязательны"}), 400
    try:
        # Автоматически подобрать ключ серии, если не задан или задан как 'auto'
        if not series_key or series_key == "auto":
            series_key = _series_key_for(domain, ql, default_key="application")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        svc_clause = " AND m.service = ANY(%s)" if services_filter else ""
        sql_common = f"""
          SELECT
            m."time",
            m.run_name,
            regexp_replace(m.series, '.*{series_key}=([^|]+).*', '\\1') AS series_name,
            m.value
          FROM public.metrics m
          WHERE m.run_name IN (%s, %s)
            AND m.domain = %s
            AND m.query_label = %s
            {svc_clause}
        """
        base_params = [run_a, run_b, domain, ql]
        if services_filter:
            base_params.append(services_filter)
        with conn, conn.cursor() as cur:
            if align == "absolute":
                cur.execute(
                    f"""
                      WITH base AS ({sql_common})
                      SELECT
                        time_bucket('1 minute'::interval, base."time") AS t,
                        base.run_name,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base
                      GROUP BY 1,2,3
                      ORDER BY 1,2,3
                    """,
                    tuple(base_params)
                )
                rows = cur.fetchall()
                data = [{"t": r[0].isoformat(), "run_name": r[1], "series": r[2], "value": float(r[3])} for r in rows]
            else:
                # Фиксированная дискретизация смещения: 60 секунд
                bucket_secs = 60

                cur.execute(
                    f"""
                      WITH base AS (
                        {sql_common}
                      ), start_ts AS (
                        SELECT run_name, MIN("time") AS t0
                        FROM base GROUP BY run_name
                      )
                      SELECT
                        FLOOR(EXTRACT(EPOCH FROM (base."time" - st.t0)) / %s)::bigint * %s AS t_offset_sec,
                        base.run_name,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base
                      JOIN start_ts st USING (run_name)
                      GROUP BY 1,2,3
                      ORDER BY 2,1,3
                    """,
                    tuple(base_params + [bucket_secs, bucket_secs])
                )
                rows = cur.fetchall()
                data = [{"t_offset_sec": int(r[0]), "run_name": r[1], "series": r[2], "value": float(r[3])} for r in rows]
        conn.close()
        return jsonify({"align": align, "points": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/run_series', methods=['GET'])
def run_series():
    run_name = request.args.get("run_name")
    domain = request.args.get("domain")
    ql = request.args.get("query_label")
    series_key = request.args.get("series_key")
    align = request.args.get("align", "absolute")
    if not all([run_name, domain, ql]):
        return jsonify({"error": "run_name, domain, query_label обязательны"}), 400
    try:
        # Автоматически подобрать ключ серии, если не задан или задан как 'auto'
        if not series_key or series_key == "auto":
            series_key = _series_key_for(domain, ql, default_key="application")
        conn = _ts_conn()
        pa = _active_project_area()
        services_filter = _resolve_services_filter(pa)
        svc_clause = " AND m.service = ANY(%s)" if services_filter else ""
        sql_common = f"""
          SELECT
            m."time",
            m.run_name,
            regexp_replace(m.series, '.*{series_key}=([^|]+).*', '\\1') AS series_name,
            m.value
          FROM public.metrics m
          WHERE m.run_name = %s
            AND m.domain = %s
            AND m.query_label = %s
            {svc_clause}
        """.replace("{series_key}", series_key)
        base_params = [run_name, domain, ql]
        if services_filter:
            base_params.append(services_filter)
        with conn, conn.cursor() as cur:
            if align == "absolute":
                cur.execute(
                    f"""
                      WITH base AS ({sql_common})
                      SELECT
                        time_bucket('1 minute'::interval, base."time") AS t,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base
                      GROUP BY 1,2
                      ORDER BY 1,2
                    """,
                    tuple(base_params)
                )
                rows = cur.fetchall()
                data = [{"t": r[0].isoformat(), "series": r[1], "value": float(r[2])} for r in rows]
            else:
                bucket_secs = 60
                cur.execute(
                    f"""
                      WITH base AS ({sql_common}),
                      start_ts AS (
                        SELECT MIN("time") AS t0 FROM base
                      )
                      SELECT
                        FLOOR(EXTRACT(EPOCH FROM (base."time" - st.t0)) / %s)::bigint * %s AS t_offset_sec,
                        base.series_name,
                        avg(base.value) AS v
                      FROM base, start_ts st
                      GROUP BY 1,2
                      ORDER BY 1,2
                    """,
                    tuple(base_params + [bucket_secs, bucket_secs])
                )
                rows = cur.fetchall()
                data = [{"t_offset_sec": int(r[0]), "series": r[1], "value": float(r[2])} for r in rows]
        conn.close()
        return jsonify({"align": align, "points": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Маршрут для создания отчета
@app.route('/create_report', methods=['POST'])
def create_report():
    data = request.json  # Получаем JSON-данные от формы
    start_str = data.get('start')
    end_str = data.get('end')
    service = data.get('service')
    project_area = (data.get('project_area') or data.get('area') or data.get('projectArea') or '').strip()
    test_type = (data.get('test_type') or '').strip()
    use_llm = bool(data.get('use_llm', True))
    save_to_db = bool(data.get('save_to_db', False))
    web_only = bool(data.get('web_only', False))
    run_name = data.get('run_name') if isinstance(data.get('run_name'), str) else None

    # Проверка наличия необходимых параметров
    if not all([start_str, end_str, service]):
        return jsonify({"status": "error", "message": "Пожалуйста, укажите время начала, окончания и сервис"}), 400

    # Конвертация времени в timestamp
    try:
        start = convert_to_timestamp(start_str)
        end = convert_to_timestamp(end_str)
    except ValueError:
        return jsonify({"status": "error", "message": "Некорректный формат времени. Используйте формат YYYY-MM-DDTHH:MM"}), 400

    # Проверка наличия конфигурации для выбранного сервиса (по активному metrics_config)
    metrics_area, service_metrics_cfg = _metrics_service_entry(service)
    if not service_metrics_cfg:
        return jsonify({"status": "error", "message": f"Конфигурация для сервиса '{service}' не найдена"}), 400

    service_area = _find_area_for_service(service)
    if not service_area:
        service_area = metrics_area
    if project_area:
        if not service_area:
            return jsonify({"status": "error", "message": f"Сервис '{service}' не привязан к области '{project_area}'"}), 400
        if service_area != project_area:
            return jsonify({"status": "error", "message": f"Сервис '{service}' принадлежит другой области"}), 400
    else:
        project_area = service_area or ''

    # Проверка уникальности имени запуска (в пределах выбранной области/сервиса)
    run_name = (run_name or '').strip() if isinstance(run_name, str) else ''
    if run_name:
        try:
            cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
            schema = cfg.get("schema", "public")
            llm_table = cfg.get("llm_table", "llm_reports")
            conn = _ts_conn()
            exists = False
            with conn, conn.cursor() as cur:
                try:
                    cur.execute("SELECT 1 FROM public.metrics WHERE run_name = %s AND service = %s LIMIT 1", (run_name, service))
                    if cur.fetchone():
                        exists = True
                except Exception:
                    exists = False if exists is False else True
                if not exists:
                    try:
                        cur.execute(f"SELECT 1 FROM {schema}.{llm_table} WHERE run_name = %s AND service = %s LIMIT 1", (run_name, service))
                        if cur.fetchone():
                            exists = True
                    except Exception:
                        pass
            conn.close()
            if exists:
                return jsonify({"status": "error", "message": f"Запуск с именем '{run_name}' уже существует для области '{service}'. Выберите другое имя."}), 400
        except Exception:
            # В случае ошибки проверки не блокируем, но логичнее отклонить — оставим мягко
            pass

    # Регистрируем задачу
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Инициализация…",
            "report_url": None,
            "error": None,
            "service": service,
            "project_area": project_area,
        }

    def _progress_cb(msg: str, pct: int | None = None):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            if isinstance(pct, int):
                job["progress"] = max(job.get("progress", 0), pct)
            job["message"] = str(msg)

    def _runner():
        try:
            res = update_report(start, end, service, use_llm=use_llm, save_to_db=save_to_db, web_only=web_only, run_name=run_name, test_type=test_type, progress_callback=_progress_cb)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "done"
                    job["progress"] = max(job.get("progress", 0), 100)
                    job["message"] = "Готово"
                    if isinstance(res, dict) and res.get("page_url"):
                        job["report_url"] = res["page_url"]
        except Exception as e:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(e)

    threading.Thread(target=_runner, daemon=True).start()
    return jsonify({"status": "accepted", "job_id": job_id, "service": service, "message": "Задача принята. Формирование отчёта началось."}), 200


@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "not_found"}), 404
        return jsonify(job), 200

# -------------------- Delete run and all related data --------------------
@app.route('/runs/<run_name>', methods=['DELETE'])
def delete_run(run_name: str):
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        metrics_table = cfg.get("table", "metrics")
        llm_table = cfg.get("llm_table", "llm_reports")
        engineer_table = cfg.get("engineer_table", "engineer_reports")
        conn = _ts_conn()
        with conn, conn.cursor() as cur:
            # Удаляем LLM-результаты
            try:
                cur.execute(f"DELETE FROM {schema}.{llm_table} WHERE run_name = %s", (run_name,))
            except Exception:
                pass
            # Удаляем итоги инженера
            try:
                cur.execute(f"DELETE FROM {schema}.{engineer_table} WHERE run_name = %s", (run_name,))
            except Exception:
                pass
            # Удаляем метрики
            try:
                cur.execute(f"DELETE FROM {schema}.{metrics_table} WHERE run_name = %s", (run_name,))
            except Exception:
                pass
        conn.close()
        return jsonify({"status": "ok", "message": "Отчёт удалён"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Project areas API --------------------
@app.route('/project_areas', methods=['GET'])
def project_areas():
    try:
        return jsonify(_list_project_areas())
    except Exception:
        return jsonify([])

@app.route('/current_project_area', methods=['GET'])
def current_project_area():
    return jsonify({"project_area": _active_project_area()})

@app.route('/project_area', methods=['POST'])
def set_project_area():
    data = request.get_json(silent=True) or {}
    name = (data.get('project_area') or data.get('service') or data.get('name') or '').strip()
    resp = jsonify({"status": "ok", "project_area": name})
    try:
        resp.set_cookie('project_area', name, max_age=60*60*24*365, samesite='Lax')
    except Exception:
        pass
    return resp

# -------------------- Engineer summary (manual conclusions) --------------------
@app.route('/engineer_summary', methods=['GET'])
def get_engineer_summary():
    run_name = request.args.get('run_name')
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        table = cfg.get("engineer_table", "engineer_reports")
        conn = _ts_conn()
        try:
            _ensure_engineer_reports_table(conn, cfg)
        except Exception:
            pass
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT content_html, created_at
                FROM {schema}.{table}
                WHERE run_name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_name,)
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({"run_name": run_name, "content_html": "", "created_at": None})
        return jsonify({"run_name": run_name, "content_html": row[0] or "", "created_at": row[1].isoformat() if row[1] else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Settings UI and API --------------------
@app.route('/settings', methods=['GET'])
def settings_page():
    return render_template('settings.html')

@app.route('/config', methods=['GET'])
def get_config():
    # Возвращаем конфиг с учётом проектной области (area)
    try:
        area = (request.args.get('area') or '')
        areas_meta = _list_project_areas()
        areas = [a['id'] for a in areas_meta]
        if not area:
            # По умолчанию используем активную область из cookie, если она есть
            cookie_area = _active_project_area() or ''
            if cookie_area in areas:
                area = cookie_area
        def merge_area_section(section_name: str) -> dict:
            base = CONFIG.get(section_name, {}) or {}
            per_area = ((CONFIG.get('per_area', {}) or {}).get(area, {}) or {}).get(section_name, {}) if area else {}
            return _deep_merge_dicts(base, per_area)
        active_area = area if area in areas else ""
        active_metrics = _active_metrics_config() or {}
        area_metrics_cfg = active_metrics.get(area, {}) if area else {}
        if not isinstance(area_metrics_cfg, dict):
            area_metrics_cfg = {}
        if 'services' not in area_metrics_cfg:
            area_metrics_cfg['services'] = {}
        out = {
            "areas": areas,
            "active_area": active_area,
            "llm": merge_area_section("llm"),
            "metrics_source": merge_area_section("metrics_source"),
            "lt_metrics_source": merge_area_section("lt_metrics_source"),
            "default_params": merge_area_section("default_params"),
            "storage": {"timescale": (CONFIG.get("storage", {}) or {}).get("timescale", {})},
            "queries": merge_area_section("queries"),
            "confluence": {
                "url_basic": CONFIG.get("url_basic"),
                "space_conf": CONFIG.get("space_conf"),
                "user": CONFIG.get("user"),
                "password": CONFIG.get("password"),
                "grafana_base_url": CONFIG.get("grafana_base_url"),
                "grafana_login": CONFIG.get("grafana_login"),
                "grafana_pass": CONFIG.get("grafana_pass"),
                "loki_url": CONFIG.get("loki_url"),
            },
            # Отдаём metrics_config только для выбранной области, чтобы редактировать точечно
            "metrics_config": area_metrics_cfg,
        }
        services_meta = {}
        services_map = _services_map_for_area(active_area) if active_area else {}
        if active_area:
            for sid, meta in services_map.items():
                if not isinstance(meta, dict):
                    meta = {}
                services_meta[sid] = {
                    "title": (meta.get('title') if isinstance(meta.get('title'), str) else '') or sid,
                    "disabled_domains": [d for d in (meta.get('disabled_domains') or []) if isinstance(d, str)],
                }
        area_metrics_services = area_metrics_cfg.get('services', {}) if isinstance(area_metrics_cfg, dict) else {}
        for sid in area_metrics_services.keys():
            services_meta.setdefault(sid, {
                "title": sid,
                "disabled_domains": [],
            })
        queries_map = {"": merge_area_section("queries")}
        for sid in services_meta.keys():
            meta = services_map.get(sid) or {}
            svc_queries = meta.get('queries') if isinstance(meta, dict) else {}
            queries_map[sid] = svc_queries if isinstance(svc_queries, dict) else {}
        metrics_config_map = {"": area_metrics_cfg}
        for sid, cfg in area_metrics_services.items():
            metrics_config_map[sid] = cfg if isinstance(cfg, dict) else {}
        out["services_meta"] = services_meta
        out["domain_list"] = _available_domain_keys()
        out["queries_map"] = queries_map
        out["metrics_config_map"] = metrics_config_map
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['POST'])
def update_config():
    # Принимает { section: "llm"|"metrics_source"|"storage.timescale"|..., data: { ... }, area?: string }
    data = request.get_json(silent=True) or {}
    section = str(data.get('section') or '').strip()
    payload = data.get('data')
    area = (data.get('area') or '').strip()
    service = (data.get('service') or '').strip()
    if not section or not isinstance(payload, dict):
        return jsonify({"error": "section и data обязательны"}), 400
    # Применяем в память/файлы
    try:
        # Глобальные разделы
        if section == 'confluence':
            allowed = {"url_basic", "space_conf", "user", "password", "grafana_base_url", "grafana_login", "grafana_pass", "loki_url"}
            # Обновление в памяти
            for k, v in (payload or {}).items():
                if k in allowed:
                    CONFIG[k] = v
            # Сохранение в runtime файл на верхнем уровне
            existing = {}
            try:
                if os.path.exists(CONFIG_RUNTIME_PATH):
                    with open(CONFIG_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
            except Exception:
                existing = {}
            for k, v in (payload or {}).items():
                if k in allowed:
                    existing[k] = v
            with open(CONFIG_RUNTIME_PATH, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            return jsonify({"status": "ok"})

        # Персервисный metrics_config — сохраняем секцию только для выбранной области
        if section == 'metrics_config':
            if not area:
                return jsonify({"error": "area обязательна для metrics_config"}), 400
            if not isinstance(payload, dict):
                return jsonify({"error": "metrics_config должен быть объектом"}), 400
            current = {}
            try:
                if os.path.exists(METRICS_RUNTIME_PATH):
                    with open(METRICS_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                        current = json.load(f)
            except Exception:
                current = {}
            if not isinstance(current, dict):
                current = {}
            if service:
                if area not in current or not isinstance(current.get(area), dict):
                    current[area] = {"services": {}}
                if 'services' not in current[area] or not isinstance(current[area].get('services'), dict):
                    current[area]['services'] = {}
                current[area]['services'][service] = payload
            else:
                if not isinstance(payload.get('services'), dict):
                    return jsonify({"error": "metrics_config должен содержать ключ 'services' с объектом сервисов"}), 400
                current[area] = payload
            with open(METRICS_RUNTIME_PATH, 'w', encoding='utf-8') as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
            return jsonify({"status": "ok"})

        if section == 'queries' and area and service:
            runtime = _load_settings_runtime_data()
            if 'per_area' not in runtime or not isinstance(runtime.get('per_area'), dict):
                runtime['per_area'] = {}
            if area not in runtime['per_area'] or not isinstance(runtime['per_area'].get(area), dict):
                runtime['per_area'][area] = {}
            area_entry = runtime['per_area'][area]
            if 'services' not in area_entry or not isinstance(area_entry.get('services'), dict):
                area_entry['services'] = {}
            if service not in area_entry['services'] or not isinstance(area_entry['services'].get(service), dict):
                area_entry['services'][service] = {}
            area_entry['services'][service]['queries'] = payload
            _save_settings_runtime_data(runtime)
            return jsonify({"status": "ok"})

        # Разделы, поддерживающие override по области
        area_overridable = {"llm", "metrics_source", "lt_metrics_source", "default_params", "queries"}
        if section in area_overridable and area:
            # Пишем в CONFIG['per_area'][area][section]
            if 'per_area' not in CONFIG or not isinstance(CONFIG.get('per_area'), dict):
                CONFIG['per_area'] = {}
            if area not in CONFIG['per_area'] or not isinstance(CONFIG['per_area'].get(area), dict):
                CONFIG['per_area'][area] = {}
            base = CONFIG['per_area'][area].get(section, {}) if isinstance(CONFIG['per_area'][area].get(section), dict) else {}
            CONFIG['per_area'][area][section] = _deep_merge_dicts(base, payload)

            # Сохраняем в settings_runtime.json
            existing = {}
            try:
                if os.path.exists(CONFIG_RUNTIME_PATH):
                    with open(CONFIG_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
            except Exception:
                existing = {}
            if 'per_area' not in existing or not isinstance(existing.get('per_area'), dict):
                existing['per_area'] = {}
            if area not in existing['per_area'] or not isinstance(existing['per_area'].get(area), dict):
                existing['per_area'][area] = {}
            base2 = existing['per_area'][area].get(section, {}) if isinstance(existing['per_area'][area].get(section), dict) else {}
            existing['per_area'][area][section] = _deep_merge_dicts(base2, payload)
            with open(CONFIG_RUNTIME_PATH, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            return jsonify({"status": "ok"})

        # Обычные (глобальные) разделы через точку
        target = CONFIG
        parts = section.split('.')
        for i, p in enumerate(parts):
            if i == len(parts)-1:
                base = target.get(p, {}) if isinstance(target.get(p), dict) else {}
                target[p] = _deep_merge_dicts(base, payload)
            else:
                if p not in target or not isinstance(target[p], dict):
                    target[p] = {}
                target = target[p]
        # Сохраняем runtime-оверрайд на диск
        existing = {}
        try:
            if os.path.exists(CONFIG_RUNTIME_PATH):
                with open(CONFIG_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
        except Exception:
            existing = {}
        tgt = existing
        for i, p in enumerate(parts):
            if i == len(parts)-1:
                base = tgt.get(p, {}) if isinstance(tgt.get(p), dict) else {}
                tgt[p] = _deep_merge_dicts(base, payload)
            else:
                if p not in tgt or not isinstance(tgt[p], dict):
                    tgt[p] = {}
                tgt = tgt[p]
        with open(CONFIG_RUNTIME_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Prompts API --------------------
@app.route('/prompts', methods=['GET'])
def get_prompts():
    try:
        area = (request.args.get('area') or '').strip()
        service = (request.args.get('service') or '').strip()
        areas_meta = _list_project_areas()
        area_ids = [a['id'] for a in areas_meta]
        if service and not area:
            derived = _find_area_for_service(service)
            if derived:
                area = derived
        if not area:
            cookie_area = _active_project_area() or ''
            if cookie_area in area_ids:
                area = cookie_area
        active_area = area if area in area_ids else (area_ids[0] if area_ids else '')
        services_map = _services_map_for_area(active_area) if active_area else {}
        services_payload = []
        for sid, meta in services_map.items():
            title = sid
            if isinstance(meta, dict):
                title = (meta.get('title') if isinstance(meta.get('title'), str) else '') or title
            services_payload.append({"id": sid, "title": title})
        if service and service not in services_map:
            service = ''
        prompts = _active_area_prompts(active_area if active_area in area_ids else None, service or None)
        filtered_prompts = {k: v for k, v in prompts.items() if k not in LOCKED_PROMPT_DOMAINS}
        return jsonify({
            "areas": area_ids,
            "active_area": active_area if active_area in area_ids else "",
            "services": services_payload,
            "active_service": service if service else "",
            "domains": filtered_prompts,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prompts', methods=['POST'])
def post_prompts():
    try:
        data = request.get_json(silent=True) or {}
        area = (data.get('area') or '').strip()
        service = (data.get('service') or '').strip()
        domain = (data.get('domain') or '').strip()
        text = data.get('text')
        if not area and service:
            area = _find_area_for_service(service) or ''
        if not area:
            return jsonify({"error": "area обязательна"}), 400
        if domain not in PROMPT_DOMAIN_FILES:
            return jsonify({"error": "неверный domain"}), 400
        if domain in LOCKED_PROMPT_DOMAINS:
            return jsonify({"error": "Редактирование домена запрещено"}), 400
        if not isinstance(text, str):
            return jsonify({"error": "text должен быть строкой"}), 400
        existing = _load_settings_runtime_data()
        if 'per_area' not in existing or not isinstance(existing.get('per_area'), dict):
            existing['per_area'] = {}
        if area not in existing['per_area'] or not isinstance(existing['per_area'].get(area), dict):
            existing['per_area'][area] = {}
        target = existing['per_area'][area]
        if service:
            if 'services' not in target or not isinstance(target.get('services'), dict):
                target['services'] = {}
            if service not in target['services'] or not isinstance(target['services'].get(service), dict):
                target['services'][service] = {}
            svc_entry = target['services'][service]
            if 'prompts' not in svc_entry or not isinstance(svc_entry.get('prompts'), dict):
                svc_entry['prompts'] = {}
            svc_entry['prompts'][domain] = text
        else:
            if 'prompts' not in target or not isinstance(target.get('prompts'), dict):
                target['prompts'] = {}
            target['prompts'][domain] = text
        _save_settings_runtime_data(existing)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/service_meta', methods=['POST'])
def update_service_meta():
    try:
        data = request.get_json(silent=True) or {}
        area = (data.get('area') or '').strip()
        service = (data.get('service') or '').strip()
        meta = data.get('data') if isinstance(data.get('data'), dict) else {}
        if service and not area:
            area = _find_area_for_service(service) or ''
        if not area or not service:
            return jsonify({"error": "area и service обязательны"}), 400
        runtime = _load_settings_runtime_data()
        if 'per_area' not in runtime or not isinstance(runtime.get('per_area'), dict):
            runtime['per_area'] = {}
        if area not in runtime['per_area'] or not isinstance(runtime['per_area'].get(area), dict):
            runtime['per_area'][area] = {}
        area_entry = runtime['per_area'][area]
        if 'services' not in area_entry or not isinstance(area_entry.get('services'), dict):
            area_entry['services'] = {}
        if service not in area_entry['services'] or not isinstance(area_entry['services'].get(service), dict):
            area_entry['services'][service] = {}
        svc_entry = area_entry['services'][service]
        if 'title' in meta:
            title_val = meta.get('title')
            svc_entry['title'] = str(title_val).strip() if isinstance(title_val, str) else ''
        if 'disabled_domains' in meta and isinstance(meta.get('disabled_domains'), list):
            allowed = set(_available_domain_keys())
            svc_entry['disabled_domains'] = [d for d in meta.get('disabled_domains') if isinstance(d, str) and d in allowed]
        _save_settings_runtime_data(runtime)
        return jsonify({"status": "ok", "service": service, "meta": svc_entry})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/areas', methods=['POST'])
def create_area():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({"error": "name обязателен"}), 400
    active = _active_metrics_config() or {}
    if name in active:
        return jsonify({"error": "область уже существует"}), 400
    # Добавляем запись в metrics_config_runtime.json
    runtime = {}
    try:
        if os.path.exists(METRICS_RUNTIME_PATH):
            with open(METRICS_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                runtime = json.load(f)
    except Exception:
        runtime = {}
    if not isinstance(runtime, dict):
        runtime = {}
    runtime[name] = {
        "services": {}
    }
    with open(METRICS_RUNTIME_PATH, 'w', encoding='utf-8') as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)
    # Создадим каркас в settings_runtime.json -> per_area[name]
    existing = {}
    try:
        if os.path.exists(CONFIG_RUNTIME_PATH):
            with open(CONFIG_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                existing = json.load(f)
    except Exception:
        existing = {}
    if 'per_area' not in existing or not isinstance(existing.get('per_area'), dict):
        existing['per_area'] = {}
    if name not in existing['per_area']:
        existing['per_area'][name] = {}
    with open(CONFIG_RUNTIME_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})

@app.route('/engineer_summary', methods=['POST'])
def post_engineer_summary():
    data = request.get_json(silent=True) or {}
    run_name = (data.get('run_name') or '').strip()
    content_html = data.get('content_html') if isinstance(data.get('content_html'), str) else data.get('content')
    if not run_name:
        return jsonify({"error": "run_name обязателен"}), 400
    if not isinstance(content_html, str):
        content_html = ""
    try:
        cfg = (CONFIG.get("storage", {}) or {}).get("timescale", {})
        schema = cfg.get("schema", "public")
        engineer_table = cfg.get("engineer_table", "engineer_reports")
        llm_table = cfg.get("llm_table", "llm_reports")
        conn = _ts_conn()
        try:
            _ensure_engineer_reports_table(conn, cfg)
        except Exception:
            pass
        service_val = ""
        with conn, conn.cursor() as cur:
            # Определим сервис по финальному домену из llm_reports, если есть
            try:
                cur.execute(
                    f"SELECT service FROM {schema}.{llm_table} WHERE run_name=%s AND domain='final' ORDER BY created_at DESC LIMIT 1",
                    (run_name,)
                )
                r = cur.fetchone()
                if r and r[0]:
                    service_val = str(r[0])
            except Exception:
                service_val = ""
            # Запишем новую версию заметки инженера в отдельную таблицу
            cur.execute(
                f"""
                INSERT INTO {schema}.{engineer_table}
                  (run_id, run_name, service, content_html)
                VALUES
                  (%s, %s, %s, %s)
                """,
                (None, run_name, service_val, content_html)
            )
        conn.close()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0')
