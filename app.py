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
    try:
        from metrics_config import METRICS_CONFIG as BASE
    except Exception:
        BASE = {}
    try:
        if os.path.exists(METRICS_RUNTIME_PATH):
            with open(METRICS_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                override = json.load(f)
            if isinstance(override, dict):
                return _deep_merge_dicts(BASE, override)
    except Exception:
        return BASE
    return BASE

# -------------------- LLM prompts (base + per-area overrides) --------------------
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'AI', 'prompts')
PROMPT_DOMAIN_FILES = {
    'overall': 'overall_prompt.txt',
    'judge': 'judge_prompt.txt',
    'critic': 'critic_prompt.txt',
    'database': 'database_prompt.txt',
    'kafka': 'kafka_prompt.txt',
    'microservices': 'microservices_prompt.txt',
    'jvm': 'jvm_prompt.txt',
    'hard_resources': 'hard_resources_prompt.txt',
    'lt_framework': 'lt_framework_prompt.txt',
}

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

def _active_area_prompts(area: str | None) -> dict:
    base = _load_base_prompts()
    if not area:
        return base
    # прочитаем overrides из settings_runtime.json: per_area[area].prompts
    try:
        if os.path.exists(CONFIG_RUNTIME_PATH):
            with open(CONFIG_RUNTIME_PATH, 'r', encoding='utf-8') as f:
                rt = json.load(f)
            per_area = (rt.get('per_area') or {}) if isinstance(rt, dict) else {}
            area_cfg = (per_area.get(area) or {}) if isinstance(per_area, dict) else {}
            overrides = (area_cfg.get('prompts') or {}) if isinstance(area_cfg, dict) else {}
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    if isinstance(v, str) and k in base:
                        base[k] = v
    except Exception:
        pass
    return base

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
        resp.set_cookie('project_area', service, max_age=60*60*24*365, samesite='Lax')
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
    # Извлекаем все ключи активного metrics_config — это и есть названия сервисов
    services = list((_active_metrics_config() or {}).keys())
    return jsonify(services), 200

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
        last_run = None
        verdict_counts = {"Успешно": 0, "Есть риски": 0, "Провал": 0, "Недостаточно данных": 0}
        with conn, conn.cursor() as cur:
            # Последний запуск по времени создания итогового отчёта (final)
            if pa:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, verdict, created_at
                    FROM {schema}.{table}
                    WHERE domain = 'final' AND service = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (pa,)
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
            if pa:
                cur.execute(
                    f"""
                    WITH ranked AS (
                      SELECT run_name, verdict, created_at,
                             ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                      FROM {schema}.{table}
                      WHERE domain = 'final' AND service = %s
                    )
                    SELECT COALESCE(verdict, 'Недостаточно данных') AS v, COUNT(*) AS cnt
                    FROM ranked
                    WHERE rn = 1
                    GROUP BY v
                    """,
                    (pa,)
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
        with conn, conn.cursor() as cur:
            if pa:
                cur.execute(
                    """
                    WITH raw AS (
                      SELECT query_label, run_name, value
                      FROM public.metrics
                      WHERE run_name IN (%s, %s)
                        AND domain = %s
                        AND service = %s
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
                    (run_a, run_b, domain, pa, run_a, run_b)
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
        sql_common = f"""
          SELECT
            regexp_replace(m.series, '.*{series_key}=([^|]+).*', '\\1') AS series_name,
            m.run_name,
            m.value
          FROM public.metrics m
          WHERE m.run_name IN (%s, %s)
            AND m.domain = %s
            AND m.query_label = %s
            { 'AND m.service = %s' if pa else '' }
        """
        with conn, conn.cursor() as cur:
            if pa:
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
                    (run_a, run_b, domain, ql, pa, run_a, run_b)
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
                    (run_a, run_b, domain, ql, run_a, run_b)
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
        with conn, conn.cursor() as cur:
            if pa:
                cur.execute(
                    f"""
                    SELECT run_name, service, start_ms, end_ms, domain, verdict, text, parsed, scores, created_at
                    FROM {schema}.{table}
                    WHERE run_name = %s AND domain <> 'engineer' AND service = %s
                    ORDER BY created_at DESC, domain
                    """,
                    (run_name, pa)
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
                if pa:
                    cur.execute(
                        f"""
                        WITH base AS (
                          SELECT run_name,
                                 MIN("time") AS start_time,
                                 MAX("time") AS end_time,
                                 COALESCE(MAX(service), '') AS service
                          FROM public.metrics
                          WHERE run_name IS NOT NULL AND run_name <> '' AND service = %s
                          {where_q}
                          GROUP BY run_name
                        ), final AS (
                          SELECT run_name, verdict, created_at, test_type,
                                 ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                          FROM {schema}.{llm_table}
                          WHERE domain = 'final' AND service = %s
                        )
                        SELECT b.run_name, b.start_time, b.end_time, b.service,
                               f.verdict, f.created_at AS report_created_at, f.test_type
                        FROM base b
                        LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                        ORDER BY {sort_sql} {dir_sql}
                        OFFSET %s LIMIT %s
                        """,
                        (pa, *params, pa, offset, limit)
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
                    if pa:
                        cur.execute(
                            f"""
                            WITH base AS (
                              SELECT run_name,
                                     MIN("time") AS start_time,
                                     MAX("time") AS end_time,
                                     COALESCE(MAX(service), '') AS service
                              FROM public.metrics
                              WHERE run_name IS NOT NULL AND run_name <> '' AND service = %s
                              {where_q}
                              GROUP BY run_name
                            ), final AS (
                              SELECT run_name, verdict, created_at,
                                     ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                              FROM {schema}.{llm_table}
                              WHERE domain = 'final' AND service = %s
                            )
                            SELECT b.run_name, b.start_time, b.end_time, b.service,
                                   f.verdict, f.created_at AS report_created_at
                            FROM base b
                            LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                            ORDER BY {sort_sql} {dir_sql}
                            OFFSET %s LIMIT %s
                            """,
                            (pa, *params, pa, offset, limit)
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
                    if pa:
                        cur.execute(
                            f"""
                            WITH base AS (
                              SELECT run_name,
                                     MIN("time") AS start_time,
                                     MAX("time") AS end_time,
                                     COALESCE(MAX(service), '') AS service
                              FROM public.metrics
                              WHERE run_name IS NOT NULL AND run_name <> '' AND service = %s
                              {where_q}
                              GROUP BY run_name
                            ), final AS (
                              SELECT run_name, created_at,
                                     ROW_NUMBER() OVER (PARTITION BY run_name ORDER BY created_at DESC) AS rn
                              FROM {schema}.{llm_table}
                              WHERE domain = 'final' AND service = %s
                            )
                            SELECT b.run_name, b.start_time, b.end_time, b.service,
                                   NULL::TEXT AS verdict, f.created_at AS report_created_at
                            FROM base b
                            LEFT JOIN final f ON f.run_name = b.run_name AND f.rn = 1
                            ORDER BY {sort_sql} {dir_sql}
                            OFFSET %s LIMIT %s
                            """,
                            (pa, *params, pa, offset, limit)
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
        with conn, conn.cursor() as cur:
            if pa:
                cur.execute(
                    """
                    SELECT domain, query_label, COUNT(*) AS cnt
                    FROM public.metrics
                    WHERE service = %s
                    GROUP BY domain, query_label
                    ORDER BY domain, query_label
                    """,
                    (pa,)
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
            { 'AND m.service = %s' if pa else '' }
        """
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
                    ((run_a, run_b, domain, ql, pa) if pa else (run_a, run_b, domain, ql))
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
                    ((run_a, run_b, domain, ql, pa, bucket_secs, bucket_secs) if pa else (run_a, run_b, domain, ql, bucket_secs, bucket_secs))
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
            { 'AND m.service = %s' if pa else '' }
        """.replace("{series_key}", series_key)
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
                    ((run_name, domain, ql, pa) if pa else (run_name, domain, ql))
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
                    ((run_name, domain, ql, pa, bucket_secs, bucket_secs) if pa else (run_name, domain, ql, bucket_secs, bucket_secs))
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
    if service not in (_active_metrics_config() or {}):
        return jsonify({"status": "error", "message": f"Конфигурация для сервиса '{service}' не найдена"}), 400

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
        services = list(METRICS_CONFIG.keys())
        return jsonify(services)
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
        areas = list((_active_metrics_config() or {}).keys())
        if not area:
            # По умолчанию используем активную область из cookie, если она есть
            cookie_area = _active_project_area() or ''
            if cookie_area in areas:
                area = cookie_area
        def merge_area_section(section_name: str) -> dict:
            base = CONFIG.get(section_name, {}) or {}
            per_area = ((CONFIG.get('per_area', {}) or {}).get(area, {}) or {}).get(section_name, {}) if area else {}
            return _deep_merge_dicts(base, per_area)
        out = {
            "areas": areas,
            "active_area": area if area in areas else "",
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
            "metrics_config": ((_active_metrics_config() or {}).get(area, {}) if area else {}),
        }
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
            current[area] = payload
            with open(METRICS_RUNTIME_PATH, 'w', encoding='utf-8') as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
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
        # если не передано, используем активную из cookie
        if not area:
            cookie_area = _active_project_area() or ''
            area = cookie_area
        areas = list((_active_metrics_config() or {}).keys())
        prompts = _active_area_prompts(area if area in areas else None)
        return jsonify({
            "areas": areas,
            "active_area": area if area in areas else "",
            "domains": prompts,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prompts', methods=['POST'])
def post_prompts():
    try:
        data = request.get_json(silent=True) or {}
        area = (data.get('area') or '').strip()
        domain = (data.get('domain') or '').strip()
        text = data.get('text')
        if not area:
            return jsonify({"error": "area обязательна"}), 400
        if domain not in PROMPT_DOMAIN_FILES:
            return jsonify({"error": "неверный domain"}), 400
        if not isinstance(text, str):
            return jsonify({"error": "text должен быть строкой"}), 400
        # читаем существующий settings_runtime.json
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
        if 'prompts' not in existing['per_area'][area] or not isinstance(existing['per_area'][area].get('prompts'), dict):
            existing['per_area'][area]['prompts'] = {}
        existing['per_area'][area]['prompts'][domain] = text
        with open(CONFIG_RUNTIME_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return jsonify({"status": "ok"})
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
        "page_sample_id": "",
        "page_parent_id": "",
        "metrics": [],
        "logs": []
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
