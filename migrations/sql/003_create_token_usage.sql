-- 003_create_token_usage.sql
-- Multi-stage token accounting and audit storage

CREATE TABLE IF NOT EXISTS token_usage (
    id            BIGSERIAL PRIMARY KEY,
    thread_id     TEXT NOT NULL,
    tenant_id     TEXT NOT NULL,
    user_id       TEXT,
    node          TEXT NOT NULL,
    model         TEXT NOT NULL,
    provider      TEXT NOT NULL DEFAULT '',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      NUMERIC(12, 6) NOT NULL DEFAULT 0,
    estimated     BOOLEAN NOT NULL DEFAULT FALSE,
    trace_id      TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_usage_tenant_day ON token_usage (tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_thread     ON token_usage (thread_id);
