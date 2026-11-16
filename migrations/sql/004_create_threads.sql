-- 004_create_threads.sql
-- Queryable index table for conversation threads, titles, and cumulative token/cost rollups

CREATE TABLE IF NOT EXISTS threads (
    thread_id       TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    owner_user_id   TEXT,
    title           TEXT NOT NULL,
    last_activity   TIMESTAMPTZ NOT NULL DEFAULT now(),
    input_tokens    BIGINT NOT NULL DEFAULT 0,
    output_tokens   BIGINT NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(12,6) NOT NULL DEFAULT 0,
    message_count   INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_threads_tenant_owner
    ON threads (tenant_id, owner_user_id, last_activity DESC);

CREATE INDEX IF NOT EXISTS idx_threads_tenant_activity
    ON threads (tenant_id, last_activity DESC);
