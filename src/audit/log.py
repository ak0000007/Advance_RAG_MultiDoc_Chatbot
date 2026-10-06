"""
Enterprise Write Audit Logging for LangGraph Approval Lifecycle.

Persists proposal, decision, and outcome states in PostgreSQL write_audit_log table.
Zero logging of sensitive payloads or usernames in plaintext logs.
"""

from contextlib import asynccontextmanager
from typing import Any, Optional
from psycopg.types.json import Jsonb


@asynccontextmanager
async def _get_conn(pool_or_conn):
    """Context manager supporting AsyncConnectionPool or individual AsyncConnection."""
    if hasattr(pool_or_conn, "connection"):
        async with pool_or_conn.connection() as conn:
            yield conn
    else:
        yield pool_or_conn


async def init_audit_table(pool) -> None:
    """Initialize write_audit_log table and indexes if they do not exist."""
    async with _get_conn(pool) as conn:
        await conn.execute("""
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
        """)


async def record_proposed(
    pool,
    thread_id: str,
    sf_username: str,
    action_type: str,
    record_id: str,
    record_name: Optional[str],
    idempotency_key: str,
    proposed_payload: dict[str, Any],
    status: str = "proposed",
) -> None:
    """Record initial proposal before human approval interrupt."""
    query = """
        INSERT INTO write_audit_log (
            thread_id, sf_username, action_type, record_id, record_name,
            idempotency_key, proposed_payload, status, updated_at
        ) VALUES (
            %(thread_id)s, %(sf_username)s, %(action_type)s, %(record_id)s, %(record_name)s,
            %(idempotency_key)s, %(proposed_payload)s, %(status)s, now()
        )
        ON CONFLICT (idempotency_key) DO UPDATE SET
            thread_id = EXCLUDED.thread_id,
            sf_username = EXCLUDED.sf_username,
            action_type = EXCLUDED.action_type,
            record_id = EXCLUDED.record_id,
            record_name = EXCLUDED.record_name,
            proposed_payload = EXCLUDED.proposed_payload,
            status = EXCLUDED.status,
            updated_at = now();
    """
    params = {
        "thread_id": thread_id,
        "sf_username": sf_username,
        "action_type": action_type,
        "record_id": record_id,
        "record_name": record_name,
        "idempotency_key": idempotency_key,
        "proposed_payload": Jsonb(proposed_payload),
        "status": status,
    }
    async with _get_conn(pool) as conn:
        await conn.execute(query, params)


async def record_decision(
    pool,
    idempotency_key: str,
    decision: str,
    status: str,
) -> None:
    """Record human approval decision (Approve / Reject)."""
    query = """
        UPDATE write_audit_log
        SET decision = %(decision)s,
            status = %(status)s,
            updated_at = now()
        WHERE idempotency_key = %(idempotency_key)s;
    """
    params = {
        "idempotency_key": idempotency_key,
        "decision": decision,
        "status": status,
    }
    async with _get_conn(pool) as conn:
        await conn.execute(query, params)


async def record_outcome(
    pool,
    idempotency_key: str,
    result: Any,
    status: str,
    action_type: Optional[str] = None,
) -> None:
    """Record final execution outcome (executed / failed)."""
    jsonb_result = Jsonb(result) if isinstance(result, (dict, list)) else Jsonb({"output": result})
    query = """
        UPDATE write_audit_log
        SET result = %(result)s,
            status = %(status)s,
            updated_at = now()
        WHERE idempotency_key = %(idempotency_key)s;
    """
    params = {
        "idempotency_key": idempotency_key,
        "result": jsonb_result,
        "status": status,
    }
    async with _get_conn(pool) as conn:
        await conn.execute(query, params)

    # Increment Prometheus metric counter
    if action_type:
        try:
            from src.telemetry_metrics import record_write_attempt
            record_write_attempt(action_type=action_type, status=status)
        except Exception:
            pass


async def get_audit_records_by_thread(
    pool,
    thread_id: str,
    sf_username: str,
) -> list[dict[str, Any]]:
    """Fetch all audit log rows for a thread scoped to authenticated sf_username."""
    query = """
        SELECT id, thread_id, sf_username, action_type, record_id, record_name,
               idempotency_key, proposed_payload, decision, status, result,
               created_at, updated_at
        FROM write_audit_log
        WHERE thread_id = %(thread_id)s AND sf_username = %(sf_username)s
        ORDER BY created_at ASC;
    """
    params = {"thread_id": thread_id, "sf_username": sf_username}
    async with _get_conn(pool) as conn:
        res = await conn.execute(query, params)
        rows = await res.fetchall()

        cols = [desc[0] for desc in res.description] if res.description else []
        records = []
        for row in rows:
            record = {}
            for col, val in zip(cols, row):
                if hasattr(val, "isoformat"):
                    record[col] = val.isoformat()
                elif hasattr(val, "hex"):
                    record[col] = str(val)
                else:
                    record[col] = val
            records.append(record)
        return records
