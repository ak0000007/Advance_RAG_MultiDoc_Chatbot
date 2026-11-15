"""
Unit and integration tests for Identity Resolution & Server Identity Enforcement (P0-2).

Tests:
1. DEV_MODE=false: body sf_username ignored in all four configurations
   (mapped key, shared key + default user, shared key without default user, SF-auth),
   and the identity passed into the graph config equals the server-derived one.
2. Shared key without sf_default_username: behaves as specified, never picks up body username.
3. DEV_MODE=true: body username honoured, control characters stripped, length capped,
   and a WARNING is logged.
4. Open mode (no key) with DEV_MODE=false: body username ignored.
5. SF-auth active: body identity ignored (regression test).
"""

import pytest
import uuid
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes import AuthUser, _verify_auth_and_get_user, _prepare_chat_context, _resolve_identity
from src.api.dependencies import get_active_graph
from src.api.schemas import ChatRequest
from src.config import Settings


class CaptureGraph:
    """Mock graph that captures inputs and execution config."""
    def __init__(self):
        self.captured_configs = []
        self.captured_inputs = []

    async def aget_state(self, config):
        class Snap:
            values = {}
            next = ()
            tasks = []
        return Snap()

    async def astream(self, inputs, config, stream_mode="values"):
        self.captured_inputs.append(inputs)
        self.captured_configs.append(config)
        yield {"answer": "ok"}


@pytest.fixture
def capture_graph():
    return CaptureGraph()


@pytest.fixture
def client(capture_graph):
    app.dependency_overrides[get_active_graph] = lambda: capture_graph
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()


def test_dev_mode_false_body_username_ignored_all_configs(client, capture_graph, monkeypatch):
    """
    1. DEV_MODE=false: body sf_username ignored across:
       - Mapped key
       - Shared key + default user
       - Shared key without default user
       - SF-auth
    """
    test_settings = Settings(
        dev_mode=False,
        api_secret_key="shared-secret-key-123",
        sf_default_username="global_default_sf",
        api_key_mapping={
            "token-mapped": {"username": "mapped_user@sf.com", "tenant_id": "tenant_mapped"},
        },
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    # Config A: Mapped key
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: AuthUser(
        "mapped_user@sf.com", "tenant_mapped", api_key_id="hash1"
    )
    tid_a = f"thread-map-{uuid.uuid4().hex[:12]}"
    resp = client.post(
        "/chat",
        json={"question": "Test", "thread_id": tid_a, "sf_username": "attacker_spoofed@sf.com"},
    )
    assert resp.status_code == 200
    cfg_a = capture_graph.captured_configs[-1]
    assert cfg_a["configurable"]["sf_username"] == "mapped_user@sf.com"
    assert cfg_a["configurable"]["sf_username"] != "attacker_spoofed@sf.com"

    # Config B: Shared key + default user
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: AuthUser(
        "global_default_sf", "default", api_key_id="hash2"
    )
    tid_b = f"thread-shared-{uuid.uuid4().hex[:12]}"
    resp = client.post(
        "/chat",
        json={"question": "Test", "thread_id": tid_b, "sf_username": "attacker_spoofed@sf.com"},
    )
    assert resp.status_code == 200
    cfg_b = capture_graph.captured_configs[-1]
    assert cfg_b["configurable"]["sf_username"] == "global_default_sf"
    assert cfg_b["configurable"]["sf_username"] != "attacker_spoofed@sf.com"

    # Config C: Shared key without default user
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: AuthUser(
        "key_abc12345", "default", api_key_id="abc12345"
    )
    tid_c = f"thread-nokey-{uuid.uuid4().hex[:12]}"
    resp = client.post(
        "/chat",
        json={"question": "Test", "thread_id": tid_c, "sf_username": "attacker_spoofed@sf.com"},
    )
    assert resp.status_code == 200
    cfg_c = capture_graph.captured_configs[-1]
    assert cfg_c["configurable"]["sf_username"] == "key_abc12345"
    assert cfg_c["configurable"]["sf_username"] != "attacker_spoofed@sf.com"

    # Config D: SF-auth active
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: AuthUser(
        "005xx0000001", "sf_org", sf_user_id="005xx0000001"
    )
    tid_d = f"thread-sf-{uuid.uuid4().hex[:12]}"
    resp = client.post(
        "/chat",
        json={"question": "Test", "thread_id": tid_d, "sf_username": "attacker_spoofed@sf.com"},
    )
    assert resp.status_code == 200
    cfg_d = capture_graph.captured_configs[-1]
    assert cfg_d["configurable"]["sf_username"] == "005xx0000001"
    assert cfg_d["configurable"]["sf_username"] != "attacker_spoofed@sf.com"


def test_shared_key_without_sf_default_username_never_picks_body(client, capture_graph, monkeypatch):
    """
    2. Shared key without sf_default_username maps to key service identity and NEVER picks body username.
    """
    test_settings = Settings(
        dev_mode=False,
        api_secret_key="secret-shared-key-999",
        sf_default_username=None,
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    auth = _verify_auth_and_get_user(api_key="secret-shared-key-999")
    assert auth is not None
    assert auth.username.startswith("key_")

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: auth
    tid = f"thread-sk-{uuid.uuid4().hex[:12]}"
    resp = client.post(
        "/chat",
        json={"question": "Check key", "thread_id": tid, "sf_username": "ceo@target-corp.com"},
    )
    assert resp.status_code == 200
    cfg = capture_graph.captured_configs[-1]
    assert cfg["configurable"]["sf_username"] == auth.username
    assert cfg["configurable"]["sf_username"] != "ceo@target-corp.com"


def test_dev_mode_true_body_username_honoured_and_sanitized(client, capture_graph, monkeypatch, caplog):
    """
    3. DEV_MODE=true: body username honoured, control characters stripped, length capped,
       and a WARNING is logged.
    """
    test_settings = Settings(
        dev_mode=True,
        sf_default_username="default_dev_user",
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    # Empty auth user in dev mode
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: None

    raw_malicious = "\x00\x1fadmin_user@dev.com\x7f" + ("x" * 200)
    tid = f"thread-dev-{uuid.uuid4().hex[:12]}"

    with caplog.at_level("WARNING"):
        resp = client.post(
            "/chat",
            json={"question": "Dev test", "thread_id": tid, "sf_username": raw_malicious},
        )
    assert resp.status_code == 200

    cfg = capture_graph.captured_configs[-1]
    resolved_username = cfg["configurable"]["sf_username"]
    assert "\x00" not in resolved_username
    assert "\x1f" not in resolved_username
    assert "\x7f" not in resolved_username
    assert len(resolved_username) <= 100
    assert resolved_username.startswith("admin_user@dev.com")
    assert any("DEV_MODE" in r.message for r in caplog.records)


def test_open_mode_dev_mode_false_ignores_body(client, capture_graph, monkeypatch):
    """
    4. Open mode (no key) with DEV_MODE=false: body username ignored.
    """
    test_settings = Settings(
        dev_mode=False,
        api_secret_key=None,
        sf_default_username="system_open_user",
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    app.dependency_overrides[_verify_auth_and_get_user] = lambda: None
    tid = f"thread-open-{uuid.uuid4().hex[:12]}"

    resp = client.post(
        "/chat",
        json={"question": "Open query", "thread_id": tid, "sf_username": "injected_user@sf.com"},
    )
    assert resp.status_code == 200

    cfg = capture_graph.captured_configs[-1]
    assert cfg["configurable"]["sf_username"] == "system_open_user"
    assert cfg["configurable"]["sf_username"] != "injected_user@sf.com"


def test_sf_auth_active_body_identity_strictly_ignored(client, capture_graph, monkeypatch):
    """
    5. SF-auth active: body identity ignored (regression test).
    """
    test_settings = Settings(
        dev_mode=True,  # Even in dev mode, SF-auth active MUST never allow body identity
        sf_auth_enabled=True,
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    user_sf = AuthUser("sf_user_123", "sf_tenant", sf_user_id="005xxREALUSER")
    app.dependency_overrides[_verify_auth_and_get_user] = lambda: user_sf

    tid = f"thread-sfauth-{uuid.uuid4().hex[:12]}"
    resp = client.post(
        "/chat",
        json={"question": "SF Auth query", "thread_id": tid, "sf_username": "fake_admin@sf.com"},
    )
    assert resp.status_code == 200

    cfg = capture_graph.captured_configs[-1]
    assert cfg["configurable"]["sf_username"] == "sf_user_123"
    assert cfg["configurable"]["sf_user_id"] == "005xxREALUSER"
    assert cfg["configurable"]["sf_username"] != "fake_admin@sf.com"
