-- Migration 001: Initial write_audit_log table and indexes
CREATE TABLE IF NOT EXISTS write_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id TEXT NOT NULL,
    sf_username TEXT NOT NULL,
    action_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    record_name TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    proposed_payload JSONB NOT NULL,
    decision TEXT,
    status TEXT NOT NULL,
    result JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_thread ON write_audit_log(thread_id);
CREATE INDEX IF NOT EXISTS idx_audit_user ON write_audit_log(sf_username);
