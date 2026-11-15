"""
Unit and integration tests for Thread Ownership & IDOR Protection (P0-1).

Tests:
1. User A creates a thread; user B (same tenant, different user) gets the uniform 404
   on /chat, /chat/stream, /chat/resume and /chat/resume/stream, and the fake graph
   receives NO execution call.
2. Different tenant, same username -> denied with uniform 404.
3. Owner A can continue and resume normally; resume config contains A's identity
   from auth_user, not from the snapshot.
4. Attacker B cannot cause a Salesforce write to execute under A's identity.
5. Unowned legacy thread is denied by default; allowed only when ALLOW_UNOWNED_THREADS=true.
6. Denial response is byte-identical to the not-found response.
7. Client cannot overwrite owner fields via the request body.
8. thread_id pattern: bad IDs get 422; generated IDs pass.
9. Shared-key mode: two different API keys/mapped users in the same tenant cannot access each other's threads.
10. SF-auth mode: ownership keys off sf_user_id.
"""

import pytest
import uuid
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import status

from src.api.app import app
from src.api.routes import AuthUser, _verify_auth_and_get_user
from src.api.dependencies import get_active_graph
from src.api.schemas import ChatRequest, ResumeRequest


class FakeSnapshot:
    def __init__(self, values=None, next_tasks=None):
        self.values = values or {}
        self.next = next_tasks or ()
        self.tasks = []


class FakeGraph:
    def __init__(self, state_store=None):
        self.state_store = state_store or {}
        self.astream_calls = []
        self.astream_events_calls = []
        self.aget_state_calls = []

    async def aget_state(self, config):
        thread_id = config.get("configurable", {}).get("thread_id")
        self.aget_state_calls.append(config)
        return self.state_store.get(thread_id, FakeSnapshot())

    async def astream(self, inputs, config, stream_mode="values"):
        self.astream_calls.append((inputs, config))
        thread_id = config.get("configurable", {}).get("thread_id")
        current = self.state_store.get(thread_id, FakeSnapshot())
        new_values = dict(current.values)
        if isinstance(inputs, dict):
            new_values.update(inputs)
        new_snapshot = FakeSnapshot(values=new_values, next_tasks=current.next)
        self.state_store[thread_id] = new_snapshot
        yield new_values

    async def astream_events(self, inputs, config, version="v2"):
        self.astream_events_calls.append((inputs, config))
        thread_id = config.get("configurable", {}).get("thread_id")
        current = self.state_store.get(thread_id, FakeSnapshot())
        new_values = dict(current.values)
        if isinstance(inputs, dict):
            new_values.update(inputs)
        new_snapshot = FakeSnapshot(values=new_values, next_tasks=current.next)
        self.state_store[thread_id] = new_snapshot
        yield {"event": "done", "metadata": {}, "data": {}}


@pytest.fixture
def fake_graph():
    return FakeGraph()


@pytest.fixture
def client(fake_graph):
    app.dependency_overrides[get_active_graph] = lambda: fake_graph
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()


def test_user_a_creates_thread_user_b_denied(client, fake_graph):
    """
    1. User A creates a thread; user B (same tenant, different user) gets uniform 404
       on all 4 endpoints, and the fake graph receives NO execution side-effect.
    """
    thread_id = f"thread-sec-{uuid.uuid4().hex[:12]}"
    user_a = AuthUser("user_a", "tenant_corp")
    user_b = AuthUser("user_b", "tenant_corp")

    # Step 1: User A creates the thread via /chat
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_a
    resp_a = client.post("/chat", json={"question": "Initial question", "thread_id": thread_id})
    assert resp_a.status_code == 200

    # Verify owner stamped
    stored = fake_graph.state_store[thread_id]
    assert stored.values["owner_tenant_id"] == "tenant_corp"
    assert stored.values["owner_user_id"] == "user_a"

    # Step 2: User B tries /chat, /chat/stream, /chat/resume, /chat/resume/stream
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_b
    execution_calls_before = len(fake_graph.astream_calls)

    # /chat
    resp_b_chat = client.post("/chat", json={"question": "Sneaky question", "thread_id": thread_id})
    assert resp_b_chat.status_code == 404
    assert resp_b_chat.json() == {"detail": f"Thread '{thread_id}' not found."}
    assert len(fake_graph.astream_calls) == execution_calls_before

    # /chat/stream
    resp_b_stream = client.post("/chat/stream", json={"question": "Sneaky stream", "thread_id": thread_id})
    assert resp_b_stream.status_code == 404
    assert resp_b_stream.json() == {"detail": f"Thread '{thread_id}' not found."}

    # Set up thread as interrupted for resume tests
    stored.next = ("human_approval",)

    # /chat/resume
    resp_b_resume = client.post("/chat/resume", json={"thread_id": thread_id, "decision": "Approve"})
    assert resp_b_resume.status_code == 404
    assert resp_b_resume.json() == {"detail": f"Thread '{thread_id}' is not paused or does not exist."}

    # /chat/resume/stream
    resp_b_rstream = client.post("/chat/resume/stream", json={"thread_id": thread_id, "decision": "Approve"})
    assert resp_b_rstream.status_code == 404
    assert resp_b_rstream.json() == {"detail": f"Thread '{thread_id}' is not paused or does not exist."}


def test_different_tenant_same_username_denied(client, fake_graph):
    """
    2. User with same username in Tenant B cannot access Tenant A's thread.
    """
    thread_id = f"thread-tenant-{uuid.uuid4().hex[:12]}"
    user_tenant_a = AuthUser("agent_smith", "tenant_alpha")
    user_tenant_b = AuthUser("agent_smith", "tenant_beta")

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_tenant_a
    resp_create = client.post("/chat", json={"question": "Confidential alpha doc", "thread_id": thread_id})
    assert resp_create.status_code == 200

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_tenant_b
    resp_denied = client.post("/chat", json={"question": "Read alpha doc", "thread_id": thread_id})
    assert resp_denied.status_code == 404
    assert resp_denied.json() == {"detail": f"Thread '{thread_id}' not found."}


def test_owner_resume_uses_caller_identity_not_stored(client, fake_graph):
    """
    3. Owner A can continue and resume normally; resume config contains A's identity
       from auth_user, not from the snapshot (stored identity mismatch test).
    """
    thread_id = f"thread-resume-{uuid.uuid4().hex[:12]}"
    user_a = AuthUser("alice@corp.com", "tenant_a")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_a

    # Craft snapshot whose stored sf_username was spoofed / poisoned to victim@sf.com
    # but owner is Alice
    fake_graph.state_store[thread_id] = FakeSnapshot(
        values={
            "owner_tenant_id": "tenant_a",
            "owner_user_id": "alice@corp.com",
            "sf_username": "poisoned_admin@sf.com",
            "tenant_id": "tenant_a",
        },
        next_tasks=("human_approval",),
    )

    resp_resume = client.post("/chat/resume", json={"thread_id": thread_id, "decision": "Approve"})
    assert resp_resume.status_code == 200

    # Assert execution config received Alice's verified identity, NOT the poisoned snapshot username
    assert len(fake_graph.astream_calls) > 0
    _, config = fake_graph.astream_calls[-1]
    assert config["configurable"]["sf_username"] == "alice@corp.com"
    assert config["configurable"]["sf_username"] != "poisoned_admin@sf.com"


def test_attacker_cannot_execute_salesforce_write_under_victim_identity(client, fake_graph):
    """
    4. Attacker B cannot resume A's write, and write config identity strictly matches caller.
    """
    thread_id = f"thread-write-{uuid.uuid4().hex[:12]}"
    user_victim = AuthUser("victim_agent", "travel_org")
    user_attacker = AuthUser("attacker_agent", "travel_org")

    fake_graph.state_store[thread_id] = FakeSnapshot(
        values={
            "owner_tenant_id": "travel_org",
            "owner_user_id": "victim_agent",
            "sf_username": "victim_agent",
        },
        next_tasks=("human_approval",),
    )

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_attacker
    resp = client.post("/chat/resume", json={"thread_id": thread_id, "decision": "Approve"})
    assert resp.status_code == 404
    # Ensure no execution took place
    assert len(fake_graph.astream_calls) == 0


def test_unowned_legacy_thread_access_control(client, fake_graph):
    """
    5. Unowned legacy thread is denied by default; allowed only when ALLOW_UNOWNED_THREADS=true.
    """
    thread_id = f"thread-legacy-{uuid.uuid4().hex[:12]}"
    user = AuthUser("modern_user", "corp")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user

    # Legacy snapshot without owner_tenant_id / owner_user_id
    fake_graph.state_store[thread_id] = FakeSnapshot(
        values={"messages": [], "answer": "old answer"},
        next_tasks=(),
    )

    # By default: ALLOW_UNOWNED_THREADS=false -> Denied with 404
    with patch("src.api.routes.settings.allow_unowned_threads", False):
        resp_denied = client.post("/chat", json={"question": "Access legacy", "thread_id": thread_id})
        assert resp_denied.status_code == 404

    # With ALLOW_UNOWNED_THREADS=true -> Allowed
    with patch("src.api.routes.settings.allow_unowned_threads", True):
        resp_allowed = client.post("/chat", json={"question": "Access legacy", "thread_id": thread_id})
        assert resp_allowed.status_code == 200


def test_denial_response_byte_identical_to_not_found(client, fake_graph):
    """
    6. Denial response is byte-identical to the not-found response (prevent thread enumeration).
    """
    target_thread_id = f"thread-probe-{uuid.uuid4().hex[:12]}"

    user_owner = AuthUser("owner_1", "t1")
    user_intruder = AuthUser("intruder_2", "t1")

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_intruder

    # Step 1: Thread does not exist yet -> probe resume
    resp_not_found = client.post("/chat/resume", json={"thread_id": target_thread_id, "decision": "Approve"})
    assert resp_not_found.status_code == 404

    # Step 2: Thread exists and belongs to owner_1
    fake_graph.state_store[target_thread_id] = FakeSnapshot(
        values={"owner_tenant_id": "t1", "owner_user_id": "owner_1"},
        next_tasks=("human_approval",),
    )

    # Intruder tries to resume owner's thread
    resp_denied = client.post("/chat/resume", json={"thread_id": target_thread_id, "decision": "Approve"})
    assert resp_denied.status_code == 404

    # Crucial security assertion: response is byte-identical (prevent timing/oracle attacks)
    assert resp_denied.content == resp_not_found.content
    assert resp_denied.json() == resp_not_found.json()


def test_client_cannot_overwrite_owner_fields_via_request(client, fake_graph):
    """
    7. Client cannot overwrite owner fields via the request body.
    """
    thread_id = f"thread-tamper-{uuid.uuid4().hex[:12]}"
    user_a = AuthUser("legit_user", "legit_tenant")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_a

    # Send client payload attempting to inject owner fields
    resp = client.post(
        "/chat",
        json={
            "question": "Can I tamper owner?",
            "thread_id": thread_id,
            "owner_user_id": "hacked_user",
            "owner_tenant_id": "hacked_tenant",
        },
    )
    assert resp.status_code == 200

    stored = fake_graph.state_store[thread_id]
    assert stored.values["owner_user_id"] == "legit_user"
    assert stored.values["owner_tenant_id"] == "legit_tenant"


def test_thread_id_pattern_validation(client):
    """
    8. thread_id pattern: bad IDs get 422; valid generated IDs pass.
    """
    # Bad IDs: too short (<8), invalid chars (slashes, symbols)
    bad_ids = ["t1", "short", "bad/id!", "thread with space", "a" * 65]
    for bad_id in bad_ids:
        resp_chat = client.post("/chat", json={"question": "Test", "thread_id": bad_id})
        assert resp_chat.status_code == 422, f"Expected 422 for bad thread_id: '{bad_id}'"

        resp_resume = client.post("/chat/resume", json={"thread_id": bad_id, "decision": "Approve"})
        assert resp_resume.status_code == 422, f"Expected 422 for bad resume thread_id: '{bad_id}'"

    # Valid ID: 8 to 64 chars, alphanumeric, hyphens, underscores
    valid_id = f"thread-valid-{uuid.uuid4().hex[:8]}"
    resp_valid = client.post("/chat", json={"question": "Valid thread test", "thread_id": valid_id})
    assert resp_valid.status_code == 200


def test_shared_key_mode_isolation(client, fake_graph):
    """
    9. Shared-key mode: two different API keys/mapped users in the same tenant
       cannot access each other's threads.
    """
    thread_id = f"thread-apikey-{uuid.uuid4().hex[:12]}"
    user_key_1 = AuthUser("agent_1", "tenant_shared", api_key_id="hash_key_1")
    user_key_2 = AuthUser("agent_2", "tenant_shared", api_key_id="hash_key_2")

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_key_1
    resp_init = client.post("/chat", json={"question": "Agent 1 thread", "thread_id": thread_id})
    assert resp_init.status_code == 200

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_key_2
    resp_access = client.post("/chat", json={"question": "Agent 2 trying to read", "thread_id": thread_id})
    assert resp_access.status_code == 404
    assert resp_access.json() == {"detail": f"Thread '{thread_id}' not found."}


def test_sf_auth_mode_ownership_keys_off_sf_user_id(client, fake_graph):
    """
    10. SF-auth mode: ownership keys off sf_user_id.
    """
    thread_id = f"thread-sfauth-{uuid.uuid4().hex[:12]}"
    user_sf_1 = AuthUser("agent_name", "sf_org_001", sf_user_id="005xx000001")
    user_sf_2 = AuthUser("agent_name", "sf_org_001", sf_user_id="005xx000002")

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_sf_1
    resp_init = client.post("/chat", json={"question": "SF User 1 turn", "thread_id": thread_id})
    assert resp_init.status_code == 200

    stored = fake_graph.state_store[thread_id]
    assert stored.values["owner_user_id"] == "005xx000001"

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_sf_2
    resp_denied = client.post("/chat", json={"question": "SF User 2 turn", "thread_id": thread_id})
    assert resp_denied.status_code == 404
