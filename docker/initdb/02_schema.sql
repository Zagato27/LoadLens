-- Схема LoadLens. Приложение создаёт таблицы само при первом обращении,
-- этот скрипт лишь готовит их заранее при инициализации контейнера.

-- ============================================================
-- Метрики (временные ряды)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.metrics (
    time        TIMESTAMPTZ      NOT NULL,
    domain      TEXT             NOT NULL,
    query_label TEXT             NOT NULL,
    run_id      TEXT,
    run_name    TEXT,
    service     TEXT,
    series      TEXT,
    value       DOUBLE PRECISION,
    promql      TEXT,
    start_ms    BIGINT,
    end_ms      BIGINT
);

SELECT create_hypertable(
    'public.metrics', 'time',
    if_not_exists       => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_time ON public.metrics (run_id, time);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name ON public.metrics (run_name, time);

-- ============================================================
-- LLM-отчёты
-- ============================================================
CREATE TABLE IF NOT EXISTS public.llm_reports (
    id             BIGSERIAL   PRIMARY KEY,
    created_at     TIMESTAMPTZ DEFAULT now(),
    run_id         TEXT,
    run_name       TEXT,
    service        TEXT,
    test_type      TEXT,
    start_ms       BIGINT,
    end_ms         BIGINT,
    domain         TEXT        NOT NULL,
    verdict        TEXT,
    text           TEXT,
    parsed         JSONB,
    scores         JSONB,
    sla_verdict    TEXT,
    sla_details    JSONB,
    system_context JSONB,
    context        JSONB,
    project_area   TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_reports_run_created ON public.llm_reports (run_name, created_at DESC);

CREATE TABLE IF NOT EXISTS public.llm_feedback (
    id          BIGSERIAL   PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    run_name    TEXT        NOT NULL,
    domain      TEXT        NOT NULL,
    target      TEXT        NOT NULL,
    finding_id  TEXT        NOT NULL DEFAULT '',
    vote        TEXT        NOT NULL,
    comment     TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_feedback_unique_target
    ON public.llm_feedback (run_name, domain, target, finding_id);
CREATE INDEX IF NOT EXISTS idx_llm_feedback_run_created
    ON public.llm_feedback (run_name, created_at DESC);

-- ============================================================
-- Итоги инженера
-- ============================================================
CREATE TABLE IF NOT EXISTS public.engineer_reports (
    id           BIGSERIAL   PRIMARY KEY,
    created_at   TIMESTAMPTZ DEFAULT now(),
    run_id       TEXT,
    run_name     TEXT        NOT NULL,
    service      TEXT,
    content_html TEXT
);

CREATE INDEX IF NOT EXISTS idx_engineer_reports_run_created ON public.engineer_reports (run_name, created_at DESC);

-- ============================================================
-- Публикации отчётов в Confluence
-- ============================================================
CREATE TABLE IF NOT EXISTS public.confluence_publications (
    id             BIGSERIAL   PRIMARY KEY,
    created_at     TIMESTAMPTZ DEFAULT now(),
    updated_at     TIMESTAMPTZ DEFAULT now(),
    run_name       TEXT        NOT NULL UNIQUE,
    service        TEXT,
    page_id        TEXT        NOT NULL,
    page_url       TEXT,
    space_key      TEXT,
    parent_page_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_confluence_publications_run ON public.confluence_publications (run_name);

-- ============================================================
-- Фоновые задачи (генерация отчётов, публикация в Confluence)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.report_jobs (
    job_id      TEXT        PRIMARY KEY,
    kind        TEXT        NOT NULL,
    run_name    TEXT,
    service     TEXT,
    status      TEXT        NOT NULL,
    progress    INTEGER     NOT NULL DEFAULT 0,
    message     TEXT,
    error       TEXT,
    report_url  TEXT,
    page_url    TEXT,
    page_id     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_report_jobs_status_updated ON public.report_jobs (status, updated_at DESC);

-- ============================================================
-- Разбор предела прогноза мощностей моделью (кэш на отчёт)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.forecast_explanations (
    run_id      TEXT        PRIMARY KEY,
    inputs_hash TEXT        NOT NULL,
    payload     JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- Authentication: users, API tokens, audit log
-- (mirrors loadlens_app/auth/pg_repository.py)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.app_users (
    id                   BIGSERIAL   PRIMARY KEY,
    username             TEXT        NOT NULL,
    display_name         TEXT        NOT NULL DEFAULT '',
    email                TEXT        NOT NULL DEFAULT '',
    role                 TEXT        NOT NULL DEFAULT 'viewer' CHECK (role IN ('viewer', 'engineer', 'admin')),
    provider             TEXT        NOT NULL DEFAULT 'local',
    external_id          TEXT,
    password_hash        TEXT,
    is_active            BOOLEAN     NOT NULL DEFAULT TRUE,
    must_change_password BOOLEAN     NOT NULL DEFAULT FALSE,
    session_epoch        INTEGER     NOT NULL DEFAULT 0,
    failed_logins        INTEGER     NOT NULL DEFAULT 0,
    locked_until         TIMESTAMPTZ,
    last_login_at        TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS app_users_username_lower_uq ON public.app_users (lower(username));
CREATE UNIQUE INDEX IF NOT EXISTS app_users_external_uq ON public.app_users (provider, external_id) WHERE external_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS public.api_tokens (
    id           BIGSERIAL   PRIMARY KEY,
    user_id      BIGINT      NOT NULL REFERENCES public.app_users (id) ON DELETE CASCADE,
    name         TEXT        NOT NULL,
    token_prefix TEXT        NOT NULL,
    token_hash   TEXT        NOT NULL UNIQUE,
    role         TEXT        NOT NULL DEFAULT 'viewer' CHECK (role IN ('viewer', 'engineer', 'admin')),
    expires_at   TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS api_tokens_user_idx ON public.api_tokens (user_id);

CREATE TABLE IF NOT EXISTS public.audit_log (
    id         BIGSERIAL   PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor_id   BIGINT,
    actor_name TEXT        NOT NULL DEFAULT '',
    action     TEXT        NOT NULL,
    target     TEXT        NOT NULL DEFAULT '',
    success    BOOLEAN     NOT NULL DEFAULT TRUE,
    ip         TEXT        NOT NULL DEFAULT '',
    via        TEXT        NOT NULL DEFAULT 'session',
    details    JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS audit_log_created_idx ON public.audit_log (created_at DESC);
