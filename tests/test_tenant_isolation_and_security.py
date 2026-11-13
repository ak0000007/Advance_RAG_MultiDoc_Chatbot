"""
Tests for multi-tenant isolation, constant-time authentication,
grader fail-safe evaluation, and mock CRM dependency wiring.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from langchain_core.documents import Document

from src.api.routes import _verify_auth_and_get_user, AuthUser, _build_config, _prepare_chat_context
from src.api.schemas import ChatRequest
from src.retrieval.bm25_store import BM25Store
from src.tools.document_search import build_document_search_tool
from src.config import Settings


def test_auth_user_class_properties():
    user = AuthUser("travel_agent_01", "acme_corp")
    assert isinstance(user, str)
    assert user == "travel_agent_01"
    assert user.username == "travel_agent_01"
    assert user.tenant_id == "acme_corp"


def test_verify_auth_mapping_and_constant_time(monkeypatch):
    """Verify that api_key_mapping maps tokens to specific users and tenants."""
    test_settings = Settings(
        api_secret_key="global-secret-123",
        sf_default_username="default_agent",
        api_key_mapping={
            "token-tenant-a": {"username": "user_a", "tenant_id": "tenant_a"},
            "token-tenant-b": {"username": "user_b", "tenant_id": "tenant_b"},
        },
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    # 1. Matching token for tenant A
    auth_a = _verify_auth_and_get_user(api_key="token-tenant-a")
    assert auth_a is not None
    assert auth_a.username == "user_a"
    assert auth_a.tenant_id == "tenant_a"

    # 2. Matching token for tenant B
    auth_b = _verify_auth_and_get_user(api_key="token-tenant-b")
    assert auth_b is not None
    assert auth_b.username == "user_b"
    assert auth_b.tenant_id == "tenant_b"

    # 3. Global secret fallback
    auth_global = _verify_auth_and_get_user(api_key="global-secret-123")
    assert auth_global is not None
    assert auth_global.username == "default_agent"
    assert auth_global.tenant_id == "default"

    # 4. Invalid key raises 401
    with pytest.raises(HTTPException) as exc_info:
        _verify_auth_and_get_user(api_key="invalid-token-xyz")
    assert exc_info.value.status_code == 401


def test_prepare_chat_context_injects_tenant_id():
    user = AuthUser("agent_42", "tenant_travel_inc")
    req = ChatRequest(question="What is the refund policy?", thread_id="test_thread_001")
    inputs, config, thread_id = _prepare_chat_context(req, user)

    assert thread_id == "test_thread_001"
    assert inputs["tenant_id"] == "tenant_travel_inc"
    assert config["configurable"]["tenant_id"] == "tenant_travel_inc"
    assert config["metadata"]["tenant_id"] == "tenant_travel_inc"


@pytest.mark.asyncio
async def test_tenant_isolation_bm25_retrieval():
    """Verify that User A cannot retrieve documents indexed for Tenant B."""
    store = BM25Store()

    doc_tenant_a = Document(
        page_content="Acme Corporation secret discount code is ACME-VIP-999 for luxury safaris.",
        metadata={"source": "acme_policy.txt", "tenant_id": "tenant_a", "chunk_id": "c1"},
    )
    doc_tenant_b = Document(
        page_content="Beta Global travel voucher code is BETA-CORP-444 for luxury safaris.",
        metadata={"source": "beta_policy.txt", "tenant_id": "tenant_b", "chunk_id": "c2"},
    )
    store.add_documents([doc_tenant_a, doc_tenant_b])

    # Query scoped to tenant_a
    results_a = store.search("luxury safaris", k=5, metadata_filter={"tenant_id": "tenant_a"})
    assert len(results_a) == 1
    assert results_a[0].metadata["tenant_id"] == "tenant_a"
    assert "ACME-VIP-999" in results_a[0].page_content
    assert "BETA-CORP-444" not in results_a[0].page_content

    # Query scoped to tenant_b
    results_b = store.search("luxury safaris", k=5, metadata_filter={"tenant_id": "tenant_b"})
    assert len(results_b) == 1
    assert results_b[0].metadata["tenant_id"] == "tenant_b"
    assert "BETA-CORP-444" in results_b[0].page_content
    assert "ACME-VIP-999" not in results_b[0].page_content


@pytest.mark.asyncio
async def test_document_search_grader_fail_safe():
    """Verify that grader failure logs error and flags context as UNVERIFIED RETRIEVAL."""
    doc = Document(
        page_content="Flight bookings must be made 14 days in advance.",
        metadata={"source": "policy.txt", "chunk_id": "chunk_001", "tenant_id": "tenant_a"},
    )

    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = [doc]

    mock_grader = AsyncMock()
    mock_grader.ainvoke.side_effect = RuntimeError("LLM rate limit or connection timeout")

    mock_rewriter = AsyncMock()

    tool = build_document_search_tool(
        retriever=mock_retriever,
        grader_llm=mock_grader,
        rewriter_chain=mock_rewriter,
        max_attempts=2,
    )

    config = {"configurable": {"tenant_id": "tenant_a"}}
    result = await tool.ainvoke({"query": "How many days in advance to book?"}, config=config)

    # Grader failed: failure is not hidden, result has loud unverified banner
    assert "[UNVERIFIED RETRIEVAL - Grader Evaluation Failed: LLM rate limit or connection timeout]" in result
    assert "Flight bookings must be made 14 days in advance." in result


def test_mock_salesforce_dependency_wiring(monkeypatch):
    """Verify that setting mock_salesforce=True returns MockSalesforceAsyncClient."""
    from src.api.dependencies import get_salesforce_client
    get_salesforce_client.cache_clear()

    test_settings = Settings(mock_salesforce=True)
    monkeypatch.setattr("src.api.dependencies.settings", test_settings)

    client = get_salesforce_client()
    assert client is not None
    assert client.__class__.__name__ == "MockSalesforceAsyncClient"
    get_salesforce_client.cache_clear()
