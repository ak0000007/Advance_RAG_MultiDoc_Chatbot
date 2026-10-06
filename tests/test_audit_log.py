"""
Tests for Enterprise Write Audit Logging and Telemetry Metrics.

Verifies:
1. Full propose -> approve -> execute flow in PostgreSQL write_audit_log.
2. Propose -> reject flow.
3. Idempotency deduplication (ON CONFLICT DO UPDATE).
4. Tenant user isolation: users only see their own audit records.
5. Integration with human_approval_node.
6. /audit/{thread_id} and /metrics HTTP endpoints.
"""

import uuid
from contextlib import asynccontextmanager
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from psycopg_pool import AsyncConnectionPool

from src.config import settings
from src.audit.log import (
    init_audit_table,
    record_proposed,
    record_decision,
    record_outcome,
    get_audit_records_by_thread,
)
from src.telemetry_metrics import (
    record_write_attempt,
    record_tool_call,
)


@asynccontextmanager
async def get_test_pool():
    """Create test AsyncConnectionPool and ensure audit table is initialized."""
    pool = AsyncConnectionPool(
        conninfo=settings.postgres_url,
        min_size=1,
        max_size=5,
        open=False,
        kwargs={"autocommit": True},
    )
    await pool.open()
    await init_audit_table(pool)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_full_propose_approve_execute_flow():
    """Verify propose -> approve -> execute updates the single row to status=executed."""
    async with get_test_pool() as db_pool:
        thread_id = f"test-thread-{uuid.uuid4()}"
        idemp_key = f"key-{uuid.uuid4()}"
        sf_user = "agent_test@example.com"
        payload = {"booking_id": "BK-101", "travelers": 4}

        # 1. Propose
        await record_proposed(
            pool=db_pool,
            thread_id=thread_id,
            sf_username=sf_user,
            action_type="booking_update",
            record_id="BK-101",
            record_name="Bali Explorer",
            idempotency_key=idemp_key,
            proposed_payload=payload,
            status="proposed",
        )

        records = await get_audit_records_by_thread(db_pool, thread_id, sf_user)
        assert len(records) == 1
        assert records[0]["status"] == "proposed"
        assert records[0]["decision"] is None
        assert records[0]["result"] is None
        assert records[0]["proposed_payload"] == payload

        # 2. Approve
        await record_decision(
            pool=db_pool,
            idempotency_key=idemp_key,
            decision="Approve",
            status="approved",
        )

        records = await get_audit_records_by_thread(db_pool, thread_id, sf_user)
        assert len(records) == 1
        assert records[0]["status"] == "approved"
        assert records[0]["decision"] == "Approve"

        # 3. Execute
        exec_result = {"isSuccess": True, "message": "Booking updated successfully"}
        await record_outcome(
            pool=db_pool,
            idempotency_key=idemp_key,
            result=exec_result,
            status="executed",
            action_type="booking_update",
        )

        records = await get_audit_records_by_thread(db_pool, thread_id, sf_user)
        assert len(records) == 1
        assert records[0]["status"] == "executed"
        assert records[0]["result"] == exec_result


@pytest.mark.asyncio
async def test_propose_reject_flow():
    """Verify propose -> reject flow marks status=rejected and stores decision=Reject."""
    async with get_test_pool() as db_pool:
        thread_id = f"test-thread-{uuid.uuid4()}"
        idemp_key = f"key-{uuid.uuid4()}"
        sf_user = "agent_test@example.com"

        await record_proposed(
            pool=db_pool,
            thread_id=thread_id,
            sf_username=sf_user,
            action_type="payment_update",
            record_id="PAY-001",
            record_name="Deposit Payment",
            idempotency_key=idemp_key,
            proposed_payload={"status": "Refunded"},
            status="proposed",
        )

        await record_decision(
            pool=db_pool,
            idempotency_key=idemp_key,
            decision="Reject",
            status="rejected",
        )

        records = await get_audit_records_by_thread(db_pool, thread_id, sf_user)
        assert len(records) == 1
        assert records[0]["status"] == "rejected"
        assert records[0]["decision"] == "Reject"
        assert records[0]["result"] is None


@pytest.mark.asyncio
async def test_idempotency_deduplication_on_conflict():
    """Verify same idempotency_key upserts existing row instead of duplicate insertion."""
    async with get_test_pool() as db_pool:
        thread_id = f"test-thread-{uuid.uuid4()}"
        idemp_key = f"key-{uuid.uuid4()}"
        sf_user = "agent_test@example.com"

        # Initial write
        await record_proposed(
            pool=db_pool,
            thread_id=thread_id,
            sf_username=sf_user,
            action_type="opportunity_update",
            record_id="OPP-001",
            record_name="Acme Corp",
            idempotency_key=idemp_key,
            proposed_payload={"stage": "Proposal/Price Quote"},
            status="proposed",
        )

        # Re-submission / resume retry with same idempotency key
        await record_proposed(
            pool=db_pool,
            thread_id=thread_id,
            sf_username=sf_user,
            action_type="opportunity_update",
            record_id="OPP-001",
            record_name="Acme Corp",
            idempotency_key=idemp_key,
            proposed_payload={"stage": "Negotiation/Review"},
            status="proposed",
        )

        records = await get_audit_records_by_thread(db_pool, thread_id, sf_user)
        assert len(records) == 1
        assert records[0]["proposed_payload"] == {"stage": "Negotiation/Review"}


@pytest.mark.asyncio
async def test_tenant_user_isolation():
    """Verify users cannot see audit rows belonging to other users on same thread."""
    async with get_test_pool() as db_pool:
        thread_id = f"shared-thread-{uuid.uuid4()}"
        user_alice = f"alice-{uuid.uuid4()}@example.com"
        user_bob = f"bob-{uuid.uuid4()}@example.com"

        # Alice proposes
        await record_proposed(
            pool=db_pool,
            thread_id=thread_id,
            sf_username=user_alice,
            action_type="booking_update",
            record_id="BK-ALICE",
            record_name="Alice Trip",
            idempotency_key=f"key-alice-{uuid.uuid4()}",
            proposed_payload={"travelers": 1},
        )

        # Bob proposes on same thread
        await record_proposed(
            pool=db_pool,
            thread_id=thread_id,
            sf_username=user_bob,
            action_type="booking_update",
            record_id="BK-BOB",
            record_name="Bob Trip",
            idempotency_key=f"key-bob-{uuid.uuid4()}",
            proposed_payload={"travelers": 2},
        )

        alice_records = await get_audit_records_by_thread(db_pool, thread_id, user_alice)
        assert len(alice_records) == 1
        assert alice_records[0]["record_name"] == "Alice Trip"

        bob_records = await get_audit_records_by_thread(db_pool, thread_id, user_bob)
        assert len(bob_records) == 1
        assert bob_records[0]["record_name"] == "Bob Trip"


@pytest.mark.asyncio
async def test_human_approval_node_records_audit():
    """Verify build_human_approval_node calls record_proposed, record_decision, and record_outcome."""
    async with get_test_pool() as db_pool:
        from langchain_core.messages import ToolMessage
        from src.graph.approval import build_human_approval_node

        mock_client = AsyncMock()
        mock_client.update_booking.return_value = {"isSuccess": True, "message": "Updated"}

        node = build_human_approval_node(salesforce_client=mock_client, pool=db_pool)

        tool_call_id = f"tc-{uuid.uuid4()}"
        thread_id = f"thread-{uuid.uuid4()}"
        sf_user = "agent@travel.com"

        approval_payload = {
            "__requires_approval__": True,
            "action_type": "booking_update",
            "record_id": "BK-999",
            "record_name": "Safari Express",
            "update_fields": {"status": "Confirmed"},
            "sf_username": sf_user,
        }

        import json
        tool_msg = ToolMessage(
            content=json.dumps(approval_payload),
            tool_call_id=tool_call_id,
            id="msg-1",
        )

        state = {
            "messages": [tool_msg],
            "sf_username": sf_user,
            "write_count": 0,
        }
        config = {
            "configurable": {
                "thread_id": thread_id,
                "sf_username": sf_user,
            }
        }

        # Mock interrupt to simulate user choosing 'Approve'
        with patch("src.graph.approval.interrupt", return_value="Approve"):
            result = await node(state, config)

        assert "messages" in result
        assert "Success" in result["messages"][0].content
        assert result.get("write_count") == 1

        # Verify audit row in database
        records = await get_audit_records_by_thread(db_pool, thread_id, sf_user)
        assert len(records) == 1
        assert records[0]["status"] == "executed"
        assert records[0]["decision"] == "Approve"
        assert records[0]["action_type"] == "booking_update"


def test_metrics_endpoint():
    """Verify /metrics endpoint returns Prometheus metrics."""
    from src.api.app import app

    record_tool_call("test_tool")
    record_write_attempt("booking_update", "executed")

    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    content = response.text
    assert "write_attempts_total" in content
    assert "tool_calls_total" in content


@pytest.mark.asyncio
async def test_audit_api_endpoint():
    """Verify GET /audit/{thread_id} endpoint returns scoped audit logs and enforces user isolation."""
    from src.api.app import app
    from src.api.dependencies import get_db_pool
    from src.api.routes import _verify_auth_and_get_user

    async with get_test_pool() as db_pool:
        thread_id = f"api-thread-{uuid.uuid4()}"
        sf_user = "auth_tester@example.com"
        other_user = "other_tester@example.com"

        await record_proposed(
            pool=db_pool,
            thread_id=thread_id,
            sf_username=sf_user,
            action_type="travel_package_update",
            record_id="PKG-01",
            record_name="Package Gold",
            idempotency_key=f"api-key-{uuid.uuid4()}",
            proposed_payload={"discount": 10},
        )

        app.dependency_overrides[get_db_pool] = lambda: db_pool
        app.dependency_overrides[_verify_auth_and_get_user] = lambda: sf_user
        try:
            client = TestClient(app)
            # 1. Authenticated user sees their own audit records
            response = client.get(f"/audit/{thread_id}")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["record_name"] == "Package Gold"
            assert data[0]["sf_username"] == sf_user

            # 2. Another authenticated user on same thread sees nothing (isolation)
            app.dependency_overrides[_verify_auth_and_get_user] = lambda: other_user
            response_other = client.get(f"/audit/{thread_id}")
            assert response_other.status_code == 200
            assert response_other.json() == []
        finally:
            app.dependency_overrides.pop(get_db_pool, None)
            app.dependency_overrides.pop(_verify_auth_and_get_user, None)
