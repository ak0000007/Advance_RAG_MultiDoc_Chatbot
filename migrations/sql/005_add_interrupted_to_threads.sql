-- 005_add_interrupted_to_threads.sql
-- Add interrupted flag to threads table and partial index for pending approval queues

ALTER TABLE threads ADD COLUMN IF NOT EXISTS interrupted BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_threads_tenant_interrupted
    ON threads (tenant_id, last_activity DESC)
    WHERE interrupted = TRUE;
