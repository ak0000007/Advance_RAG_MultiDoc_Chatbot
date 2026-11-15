-- Migration 002: Add approver column to write_audit_log
ALTER TABLE write_audit_log ADD COLUMN IF NOT EXISTS approver TEXT;
