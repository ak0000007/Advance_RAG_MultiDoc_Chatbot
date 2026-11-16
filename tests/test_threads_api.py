"""
Unit and integration tests for Persistent Chat History & Resume API (Phase 3).

Validates:
1. Cross-tenant isolation on GET /threads (Tenant A cannot see Tenant B's threads).
2. Uniform IDOR defense on GET /threads/{thread_id} (cross-tenant request returns HTTP 404).
3. Message scrubbing: Strips system messages, tool calls, tool messages, and CRAG JSONs.
4. Thread record upserting & turn accumulation: message_count increment, token & cost accumulation.
5. Open unauthenticated mode scopes to 'default' tenant.
6. Paused thread returns interrupted=True and approval_request payload in ThreadDetail.
"""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.api.app import app
from src.api.routes import (
    AuthUser,
    _verify_auth_and_get_user,
    _extract_clean_thread_messages,
    upsert_thread_record,
    get_db_pool,
)
from src.api.dependencies import get_active_graph
from src.llm.usage import TurnUsage, StageUsage


class FakeInterrupt:
    def __init__(self, value):
        self.value = value


class FakeTask:
    def __init__(self, interrupts=None):
        self.interrupts = interrupts or []


class FakeSnapshot:
    def __init__(self, values=None, next_tasks=None, tasks=None):
        self.values = values or {}
        self.next = next_tasks or ()
        self.tasks = tasks or []


class FakeGraph:
    def __init__(self, state_store=None):
        self.state_store = state_store or {}
        self.aget_state_calls = []

    async def aget_state(self, config):
        thread_id = config.get("configurable", {}).get("thread_id")
        self.aget_state_calls.append(config)
        return self.state_store.get(thread_id, FakeSnapshot())

    async def astream(self, inputs, config, stream_mode="values"):
        thread_id = config.get("configurable", {}).get("thread_id")
        current = self.state_store.get(thread_id, FakeSnapshot())
        new_values = dict(current.values)
        if isinstance(inputs, dict):
            new_values.update(inputs)
        # Clear interrupts on resume completion
        new_snapshot = FakeSnapshot(values=new_values, next_tasks=(), tasks=[])
        self.state_store[thread_id] = new_snapshot
        yield new_values

    async def astream_events(self, inputs, config, version="v2"):
        thread_id = config.get("configurable", {}).get("thread_id")
        current = self.state_store.get(thread_id, FakeSnapshot())
        new_values = dict(current.values)
        if isinstance(inputs, dict):
            new_values.update(inputs)
        new_snapshot = FakeSnapshot(values=new_values, next_tasks=(), tasks=[])
        self.state_store[thread_id] = new_snapshot
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": "Approved and executed."},
        }


class FakeCursor:
    def __init__(self, rows=None):
        self._rows = rows or []

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, rows=None):
        self.cursor_mock = FakeCursor(rows)
        self.executed_queries = []

    async def execute(self, query, params=None):
        self.executed_queries.append((query, params))
        return self.cursor_mock

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class FakePool:
    def __init__(self, rows=None):
        self.connection = FakeConnection(rows)

    def connection(self):
        return self.connection


@pytest.fixture
def fake_graph():
    return FakeGraph()


@pytest.fixture
def client(fake_graph):
    app.dependency_overrides[get_active_graph] = lambda: fake_graph
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()


def test_extract_clean_thread_messages_scrubbing():
    """Verify internal tool calls, system messages, and CRAG JSONs are completely scrubbed."""
    raw_messages = [
        SystemMessage(content="You are an enterprise assistant."),
        HumanMessage(content="Can you check booking BK-001?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_booking", "args": {"name": "BK-001"}, "id": "tc-1"}],
        ),
        ToolMessage(content='{"id": "a01", "name": "BK-001", "status": "Confirmed"}', tool_call_id="tc-1"),
        AIMessage(content='{"is_relevant": true, "reason": "Accurate booking info"}'),
        AIMessage(content="Booking BK-001 is confirmed with 2 travelers."),
    ]

    cleaned = _extract_clean_thread_messages(raw_messages)
    assert len(cleaned) == 2
    assert cleaned[0].role == "user"
    assert cleaned[0].content == "Can you check booking BK-001?"
    assert cleaned[1].role == "assistant"
    assert cleaned[1].content == "Booking BK-001 is confirmed with 2 travelers."


def test_list_threads_cross_tenant_isolation(client):
    """GET /threads only queries and returns records for the caller's tenant."""
    now = datetime.now(timezone.utc)
    mock_rows = [
        ("th-1", "tenant_a", "user_1", "Trip to Paris", now, 100, 50, 0.0012, 2, False),
        ("th-2", "tenant_a", "user_1", "Policy inquiry", now, 80, 40, 0.0009, 2, False),
    ]
    fake_pool = MagicMock()
    fake_conn = FakeConnection(mock_rows)
    fake_pool.connection.return_value = fake_conn

    user_a = AuthUser("user_1", "tenant_a")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_a
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    resp = client.get("/threads")
    assert resp.status_code == 200
    threads = resp.json()
    assert len(threads) == 2
    assert threads[0]["thread_id"] == "th-1"
    assert threads[0]["tenant_id"] == "tenant_a"
    assert threads[1]["thread_id"] == "th-2"

    # Verify query strictly filtered by tenant_a
    assert len(fake_conn.executed_queries) >= 1
    query, params = fake_conn.executed_queries[0]
    assert "tenant_id = %(tenant_id)s" in query
    assert params["tenant_id"] == "tenant_a"
    assert params["owner_user_id"] == "user_1"


def test_get_thread_cross_tenant_idor_returns_404(client, fake_graph):
    """GET /threads/{id} returns uniform 404 when requested by another tenant."""
    thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    # Thread belongs to tenant_a
    fake_graph.state_store[thread_id] = FakeSnapshot(
        values={
            "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")],
            "owner_tenant_id": "tenant_a",
            "owner_user_id": "user_a",
        }
    )

    # Caller is from tenant_b
    user_b = AuthUser("user_b", "tenant_b")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_b

    resp = client.get(f"/threads/{thread_id}")
    # Must be 404, never 403 (to avoid leaking existence)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


def test_get_thread_returns_detail_and_approval_state(client, fake_graph):
    """GET /threads/{id} returns clean detail and detects pending interrupt."""
    thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    user_a = AuthUser("user_a", "tenant_a")

    approval_payload = {
        "__requires_approval__": True,
        "action": "update_booking",
        "record_id": "a01xx001",
        "message": "Confirm update to 4 travelers",
    }

    # Snapshot paused at human approval
    fake_graph.state_store[thread_id] = FakeSnapshot(
        values={
            "messages": [
                HumanMessage(content="Update booking BK-100"),
                AIMessage(content="Requesting approval for booking update."),
            ],
            "owner_tenant_id": "tenant_a",
            "owner_user_id": "user_a",
        },
        next_tasks=("human_approval",),
        tasks=[FakeTask(interrupts=[FakeInterrupt(approval_payload)])],
    )

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_a

    resp = client.get(f"/threads/{thread_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["thread_id"] == thread_id
    assert data["tenant_id"] == "tenant_a"
    assert data["interrupted"] is True
    assert data["approval_request"] is not None
    assert data["approval_request"]["action"] == "update_booking"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_upsert_thread_record_accumulates_usage():
    """upsert_thread_record correctly inserts title, increments count, and accumulates tokens/cost."""
    fake_conn = FakeConnection()
    fake_pool = MagicMock()
    fake_pool.connection.return_value = fake_conn

    turn_usage = TurnUsage(
        input_tokens=150,
        output_tokens=75,
        total_tokens=225,
        cost_usd=0.0015,
        by_stage=[
            StageUsage("synthesis", "gemini-2.5-pro", "google_genai", 150, 75, 0.0015, False)
        ],
        estimated=False,
    )

    thread_id = "thread-acc-123"
    tenant_id = "tenant_test"
    user_id = "user_test"
    question = "How much is the flight to Tokyo?"

    await upsert_thread_record(
        fake_pool,
        thread_id=thread_id,
        tenant_id=tenant_id,
        owner_user_id=user_id,
        question=question,
        turn_usage=turn_usage,
    )

    assert len(fake_conn.executed_queries) == 1
    query, params = fake_conn.executed_queries[0]
    assert "INSERT INTO threads" in query
    assert "ON CONFLICT (thread_id) DO UPDATE" in query
    assert params["thread_id"] == thread_id
    assert params["tenant_id"] == tenant_id
    assert params["title"] == "How much is the flight to Tokyo?"
    assert params["input_tokens"] == 150
    assert params["output_tokens"] == 75
    assert params["cost_usd"] == 0.0015
    assert "message_count = threads.message_count + 1" in query


def test_open_unauthenticated_mode_scopes_to_default_tenant(client):
    """When no auth is active, GET /threads defaults to tenant_id='default' and owner_user_id IS NULL."""
    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: None
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    resp = client.get("/threads")
    assert resp.status_code == 200
    assert fake_conn.executed_queries
    query, params = fake_conn.executed_queries[0]
    assert params["tenant_id"] == "default"
    assert "owner_user_id IS NULL" in query


def test_intra_tenant_anonymous_caller_scoped_to_null_owner_only(client):
    """
    Observation 1 fix:
    Caller with tenant but no user_id (caller_user_id=None) MUST be scoped to
    owner_user_id IS NULL only. It must NEVER broaden to tenant-wide query.
    """
    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    # Caller has tenant_id="tenant_corp", but no username (e.g. partial token or unassigned key)
    anon_user = AuthUser(username="", tenant_id="tenant_corp")
    anon_user.username = None
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: anon_user
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    resp = client.get("/threads")
    assert resp.status_code == 200
    assert fake_conn.executed_queries
    query, params = fake_conn.executed_queries[0]

    assert params["tenant_id"] == "tenant_corp"
    # Strict anti-IDOR guard: must explicitly filter by owner_user_id IS NULL
    assert "owner_user_id IS NULL" in query
    # Must NOT run the broad tenant query
    assert "owner_user_id = %(owner_user_id)s" not in query


def test_approver_list_scoping_sees_only_own_and_interrupted(client):
    """
    Observation 2 fix (list scoping):
    Approvers see their own threads plus colleagues' interrupted threads (interrupted=TRUE).
    They must NOT see colleagues' normal non-interrupted threads (no privacy over-grant).
    """
    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    approver = AuthUser(username="mgr_1", tenant_id="tenant_corp", sf_role="Manager")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: approver
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    with patch("src.config.settings.approver_roles", ["Manager"]):
        resp = client.get("/threads")
        assert resp.status_code == 200
        assert fake_conn.executed_queries
        query, params = fake_conn.executed_queries[0]

        assert params["tenant_id"] == "tenant_corp"
        assert params["owner_user_id"] == "mgr_1"
        # Must check owner OR interrupted = TRUE
        assert "interrupted = TRUE" in query


def test_approver_detail_parity_interrupted_vs_normal(client, fake_graph):
    """
    Observation 2 fix (detail parity):
    Approver calling GET /threads/{id} on colleague's thread:
    - 200 OK when thread has a pending approval interrupt.
    - 404 Not Found when thread is normal/completed (cannot snoop on colleagues' chats).
    """
    th_interrupted = f"thread-intr-{uuid.uuid4().hex[:8]}"
    th_normal = f"thread-norm-{uuid.uuid4().hex[:8]}"

    approval_payload = {
        "__requires_approval__": True,
        "action": "update_booking",
        "record_id": "BK-100",
    }

    # 1. Colleague Alice's thread with pending interrupt
    fake_graph.state_store[th_interrupted] = FakeSnapshot(
        values={
            "messages": [HumanMessage(content="Update booking")],
            "owner_tenant_id": "tenant_corp",
            "owner_user_id": "alice",
        },
        next_tasks=("human_approval",),
        tasks=[FakeTask(interrupts=[FakeInterrupt(approval_payload)])],
    )

    # 2. Colleague Alice's normal thread without interrupt
    fake_graph.state_store[th_normal] = FakeSnapshot(
        values={
            "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi Alice")],
            "owner_tenant_id": "tenant_corp",
            "owner_user_id": "alice",
        },
        next_tasks=(),
        tasks=[],
    )

    approver = AuthUser(username="mgr_1", tenant_id="tenant_corp", sf_role="Manager")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: approver

    with patch("src.config.settings.approver_roles", ["Manager"]):
        # Manager accessing interrupted thread -> GRANTED (200)
        resp_intr = client.get(f"/threads/{th_interrupted}")
        assert resp_intr.status_code == 200
        data = resp_intr.json()
        assert data["interrupted"] is True
        assert data["approval_request"]["action"] == "update_booking"

        # Manager accessing normal thread -> DENIED (404, masks existence & prevents snooping)
        resp_norm = client.get(f"/threads/{th_normal}")
        assert resp_norm.status_code == status.HTTP_404_NOT_FOUND


def test_approver_cannot_chat_on_colleague_interrupted_thread(client, fake_graph):
    """
    Approver can view (get_thread) and resume (POST /chat/resume) an interrupted thread,
    but CANNOT post new chat messages (POST /chat) into colleague's thread (no impersonation).
    """
    th_interrupted = f"thread-intr-{uuid.uuid4().hex[:8]}"
    fake_graph.state_store[th_interrupted] = FakeSnapshot(
        values={
            "messages": [HumanMessage(content="Update booking")],
            "owner_tenant_id": "tenant_corp",
            "owner_user_id": "alice",
        },
        next_tasks=("human_approval",),
        tasks=[FakeTask(interrupts=[FakeInterrupt({"__requires_approval__": True})])],
    )

    approver = AuthUser(username="mgr_1", tenant_id="tenant_corp", sf_role="Manager")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: approver

    with patch("src.config.settings.approver_roles", ["Manager"]):
        # POST /chat into Alice's thread must return 404
        resp = client.post("/chat", json={"question": "I am the manager chatting here", "thread_id": th_interrupted})
        assert resp.status_code == status.HTTP_404_NOT_FOUND


def test_resume_chat_clears_interrupted_flag(client, fake_graph):
    """
    POST /chat/resume executes and clears threads.interrupted to FALSE in Postgres.
    """
    th_id = f"thread-res-{uuid.uuid4().hex[:8]}"
    approval_payload = {"__requires_approval__": True, "action": "update_booking", "record_id": "BK-200"}
    fake_graph.state_store[th_id] = FakeSnapshot(
        values={"messages": [HumanMessage(content="Update booking")], "owner_tenant_id": "tenant_corp", "owner_user_id": "alice"},
        next_tasks=("human_approval",),
        tasks=[FakeTask(interrupts=[FakeInterrupt(approval_payload)])],
    )

    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    approver = AuthUser(username="mgr_1", tenant_id="tenant_corp", sf_role="Manager")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: approver
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    with patch("src.config.settings.approver_roles", ["Manager"]):
        resp = client.post("/chat/resume", json={"thread_id": th_id, "decision": "Approve"})
        assert resp.status_code == 200

        # Check executed queries for UPDATE threads SET interrupted = %(interrupted)s
        sync_queries = [
            (q, p) for q, p in fake_conn.executed_queries
            if "UPDATE threads" in q and "interrupted = %(interrupted)s" in q
        ]
        assert len(sync_queries) > 0
        assert sync_queries[-1][1]["interrupted"] is False
        assert sync_queries[-1][1]["thread_id"] == th_id


def test_resume_stream_clears_interrupted_flag(client, fake_graph):
    """
    POST /chat/resume/stream runs SSE generator and guarantees threads.interrupted = FALSE in finally.
    """
    th_id = f"thread-stream-res-{uuid.uuid4().hex[:8]}"
    approval_payload = {"__requires_approval__": True, "action": "update_booking", "record_id": "BK-300"}
    fake_graph.state_store[th_id] = FakeSnapshot(
        values={"messages": [HumanMessage(content="Update booking")], "owner_tenant_id": "tenant_corp", "owner_user_id": "alice"},
        next_tasks=("human_approval",),
        tasks=[FakeTask(interrupts=[FakeInterrupt(approval_payload)])],
    )

    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    approver = AuthUser(username="mgr_1", tenant_id="tenant_corp", sf_role="Manager")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: approver
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    with patch("src.config.settings.approver_roles", ["Manager"]):
        resp = client.post("/chat/resume/stream", json={"thread_id": th_id, "decision": "Approve"})
        assert resp.status_code == 200

        sync_queries = [
            (q, p) for q, p in fake_conn.executed_queries
            if "UPDATE threads" in q and "interrupted = %(interrupted)s" in q
        ]
        assert len(sync_queries) > 0
        assert sync_queries[-1][1]["interrupted"] is False
        assert sync_queries[-1][1]["thread_id"] == th_id


def test_resume_error_still_syncs_interrupted_in_finally(client, fake_graph):
    """
    If graph execution raises an error during resume, finally block still syncs threads.interrupted.
    """
    th_id = f"thread-err-res-{uuid.uuid4().hex[:8]}"
    approval_payload = {"__requires_approval__": True, "action": "update_booking", "record_id": "BK-400"}
    fake_graph.state_store[th_id] = FakeSnapshot(
        values={"messages": [HumanMessage(content="Update booking")], "owner_tenant_id": "tenant_corp", "owner_user_id": "alice"},
        next_tasks=("human_approval",),
        tasks=[FakeTask(interrupts=[FakeInterrupt(approval_payload)])],
    )

    async def failing_astream(inputs, config, stream_mode="values"):
        # Clear interrupts in checkpointer to simulate task finishing but downstream crashing
        fake_graph.state_store[th_id] = FakeSnapshot(
            values={"owner_tenant_id": "tenant_corp", "owner_user_id": "alice"},
            next_tasks=(),
            tasks=[]
        )
        raise RuntimeError("Salesforce network connection timed out")
        yield {}

    fake_graph.astream = failing_astream

    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    approver = AuthUser(username="mgr_1", tenant_id="tenant_corp", sf_role="Manager")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: approver
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    with patch("src.config.settings.approver_roles", ["Manager"]):
        resp = client.post("/chat/resume", json={"thread_id": th_id, "decision": "Approve"})
        assert resp.status_code == 500

        # Finally must have updated threads.interrupted = False
        sync_queries = [
            (q, p) for q, p in fake_conn.executed_queries
            if "UPDATE threads" in q and "interrupted = %(interrupted)s" in q
        ]
        assert len(sync_queries) > 0
        assert sync_queries[-1][1]["interrupted"] is False


def test_get_thread_self_heals_stale_interrupted_db_flag(client, fake_graph):
    """
    Self-healing: DB has interrupted = TRUE, but checkpointer has no interrupt.
    GET /threads/{id} detects discrepancy and reconciles threads.interrupted = FALSE.
    """
    th_id = f"thread-heal-{uuid.uuid4().hex[:8]}"
    fake_graph.state_store[th_id] = FakeSnapshot(
        values={"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")], "owner_tenant_id": "tenant_corp", "owner_user_id": "alice"},
        next_tasks=(),
        tasks=[],
    )

    now = datetime.now(timezone.utc)
    # Stale DB row has interrupted = True
    stale_db_row = ("Hello title", now, 100, 50, 0.002, 2, True)

    fake_pool = MagicMock()
    fake_conn = FakeConnection([stale_db_row])
    fake_pool.connection.return_value = fake_conn

    owner = AuthUser(username="alice", tenant_id="tenant_corp")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: owner
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    resp = client.get(f"/threads/{th_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["interrupted"] is False

    # Verify self-healing UPDATE executed
    sync_queries = [
        (q, p) for q, p in fake_conn.executed_queries
        if "UPDATE threads" in q and "interrupted = %(interrupted)s" in q
    ]
    assert len(sync_queries) > 0
    assert sync_queries[-1][1]["interrupted"] is False
    assert sync_queries[-1][1]["thread_id"] == th_id


def test_unowned_threads_hidden_when_allow_unowned_false(client):
    """
    When ALLOW_UNOWNED_THREADS=False (default), unowned rows (owner_user_id IS NULL)
    are strictly filtered out from list queries so they don't leak or 404 on click.
    """
    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    alice = AuthUser(username="alice", tenant_id="tenant_corp")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: alice
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    with patch("src.config.settings.allow_unowned_threads", False):
        resp = client.get("/threads")
        assert resp.status_code == 200
        assert fake_conn.executed_queries
        query, params = fake_conn.executed_queries[0]

        assert params["allow_unowned"] is False
        assert "(%(allow_unowned)s OR owner_user_id IS NOT NULL)" in query


def test_unowned_threads_permitted_when_allow_unowned_true(client):
    """
    When ALLOW_UNOWNED_THREADS=True, allow_unowned parameter evaluates to True,
    allowing unowned rows to be queried.
    """
    fake_pool = MagicMock()
    fake_conn = FakeConnection([])
    fake_pool.connection.return_value = fake_conn

    alice = AuthUser(username="alice", tenant_id="tenant_corp")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: alice
    app.dependency_overrides[get_db_pool] = lambda: fake_pool

    with patch("src.config.settings.allow_unowned_threads", True):
        resp = client.get("/threads")
        assert resp.status_code == 200
        assert fake_conn.executed_queries
        query, params = fake_conn.executed_queries[0]

        assert params["allow_unowned"] is True
        assert "(%(allow_unowned)s OR owner_user_id IS NOT NULL)" in query

