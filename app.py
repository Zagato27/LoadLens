from flask import Flask, request, jsonify, render_template, send_file
from update_page import update_report  # Замените на имя файла, где находится `update_report`
from settings import CONFIG  # Базовые настройки (унифицированные)
from metrics_config import METRICS_CONFIG  # Конфигурация метрик
from datetime import datetime
import threading
import uuid
import psycopg2
import os



app = Flask(__name__)

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
    # Извлекаем все ключи словаря METRICS_CONFIG — это и есть названия сервисов
    services = list(METRICS_CONFIG.keys())
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
        last_run = None
        verdict_counts = {"Успешно": 0, "Есть риски": 0, "Провал": 0, "Недостаточно данных": 0}
        with conn, conn.cursor() as cur:
            # Последний запуск по времени создания итогового отчёта (final)
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
        with conn, conn.cursor() as cur:
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
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT run_name, service, start_ms, end_ms, domain, verdict, text, parsed, scores, created_at
                FROM {schema}.{table}
                WHERE run_name = %s
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
        with conn, conn.cursor() as cur:
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
        conn.close()
        return jsonify([
            {
                "run_name": r[0],
                "start_time": r[1].isoformat() if r[1] else None,
                "end_time": r[2].isoformat() if r[2] else None,
                "service": r[3],
                "verdict": r[4],
                "report_created_at": (r[5].isoformat() if r[5] else None),
            }
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/domains_schema', methods=['GET'])
def domains_schema():
    try:
        conn = _ts_conn()
        with conn, conn.cursor() as cur:
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
                    (run_a, run_b, domain, ql)
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
                    (run_a, run_b, domain, ql, bucket_secs, bucket_secs)
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
                    (run_name, domain, ql)
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
                    (run_name, domain, ql, bucket_secs, bucket_secs)
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

    # Проверка наличия конфигурации для выбранного сервиса
    if service not in METRICS_CONFIG:
        return jsonify({"status": "error", "message": f"Конфигурация для сервиса '{service}' не найдена"}), 400

    # Регистрируем задачу
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Инициализация…",
            "report_url": None,
            "error": None,
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
            res = update_report(start, end, service, use_llm=use_llm, save_to_db=save_to_db, web_only=web_only, run_name=run_name, progress_callback=_progress_cb)
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
    return jsonify({"status": "accepted", "job_id": job_id, "message": "Задача принята. Формирование отчёта началось."}), 200


@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "not_found"}), 404
        return jsonify(job), 200

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0')
