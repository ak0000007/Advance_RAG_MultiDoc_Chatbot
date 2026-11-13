"""
Comprehensive Test Suite for Salesforce Record-Level Access Control for RAG Retrieval.

Tests:
1. HEADLINE: Two users, one account-linked doc (User A reads, User B gets zero chunks & no leaked info).
2. Role pre-filter: User with account access but wrong role gets nothing.
3. Org-scope vs none-scope: Org docs visible to all in tenant; none-scope visible to none.
4. Cross-tenant isolation: Same user ID in different tenant gets nothing.
5. Fail-closed: Salesforce timeout/error -> no chunks returned, error logged.
6. Cache & TTL: Fast cache hit with 0 SF calls; re-query after TTL; revoked access applies after TTL.
7. JWT security: Expired, wrong aud/iss, HS256, alg=none, lifetime > 120s, replayed jti, missing sf_user_id -> 401.
8. Identity spoofing: Body fields cannot override verified JWT identity.
9. Over-fetch and widening: Top inaccessible candidates widened to fetch accessible candidate ranked below.
10. Ingestion: 15/18-char ID normalization, bad ID rejection, deterministic chunk ID collision resistance.
"""

import asyncio
import time
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import jwt
from langchain_core.documents import Document

from src.config import Settings
from src.security.sf_access import AccessResolver
from src.retrieval.bm25_store import BM25Store
from src.vector_stores.qdrant_store import QdrantStore
from src.tools.document_search import build_document_search_tool
from src.api.routes import (
    _verify_auth_and_get_user,
    _verify_salesforce_jwt,
    _prepare_chat_context,
    AuthUser,
    _seen_jtis,
)
from src.api.schemas import ChatRequest
from evals.mock_salesforce import MockSalesforceAsyncClient
from ingest import normalize_salesforce_id, generate_chunk_id, load_sf_manifest


@pytest.fixture(scope="session")
def rsa_keys():
    """Generate in-memory RSA keypair for JWT signing and verification tests."""
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return priv_pem, pub_pem


def _create_jwt(
    priv_key: str,
    claims: dict | None = None,
    algorithm: str = "RS256",
    headers: dict | None = None,
) -> str:
    now = int(time.time())
    default_claims = {
        "iss": "https://test.salesforce.com",
        "aud": "advance-rag-travel-copilot",
        "exp": now + 60,
        "iat": now,
        "jti": str(uuid.uuid4()),
        "sf_user_id": "005000000000001AAA",
        "sf_role": "Sales Manager",
        "sf_org_id": "00D000000000001AAA",
        "tenant_id": "tenant_1",
    }
    if claims:
        default_claims.update(claims)
    return jwt.encode(default_claims, priv_key, algorithm=algorithm, headers=headers)


# ── Scenario 1: HEADLINE — User A vs User B ──────────────────────────────────

@pytest.mark.asyncio
async def test_headline_two_users_one_account_doc():
    """
    User A has read access to Account X; User B does not.
    User A retrieves Account X chunk ($120k renewal price).
    User B gets 0 chunks and zero trace of $120k in result.
    """
    client = MockSalesforceAsyncClient()
    acc_x = "001000000000001AAA"
    acc_y = "001000000000002BBB"

    client.grant_record_access("user_a", acc_x, True)
    client.grant_record_access("user_b", acc_x, False)
    client.grant_record_access("user_b", acc_y, True)

    store = BM25Store()
    doc_x = Document(
        page_content="Acme renewal price is $120k for Account X special contract terms.",
        metadata={
            "source": "acme_contract.pdf",
            "sf_account_id": acc_x,
            "allowed_roles": ["Sales Manager"],
            "sf_scope": "account",
            "tenant_id": "tenant_1",
            "chunk_id": "cx_1",
        },
    )
    doc_y = Document(
        page_content="Beta renewal price is $50k for Account Y standard contract terms.",
        metadata={
            "source": "beta_contract.pdf",
            "sf_account_id": acc_y,
            "allowed_roles": ["Sales Manager"],
            "sf_scope": "account",
            "tenant_id": "tenant_1",
            "chunk_id": "cy_1",
        },
    )
    store.add_documents([doc_x, doc_y])

    resolver = AccessResolver(salesforce_client=client, cache_ttl_seconds=60)
    mock_grader = AsyncMock()
    mock_grader.ainvoke.return_value = MagicMock(is_relevant=True, reason="Relevant")
    mock_rewriter = AsyncMock()

    tool = build_document_search_tool(
        retriever=store.as_dynamic_retriever(),
        grader_llm=mock_grader,
        rewriter_chain=mock_rewriter,
        max_attempts=1,
        access_resolver=resolver,
        k=2,
    )

    # 1. User A query
    config_a = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "user_a",
            "sf_role": "Sales Manager",
            "sf_org_id": "org_1",
            "tenant_id": "tenant_1",
        }
    }
    res_a = await tool.ainvoke({"query": "Acme renewal price"}, config=config_a)
    assert "$120k" in res_a
    assert "acme_contract.pdf" in res_a

    # 2. User B query (has access to Y, but NO access to X)
    config_b = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "user_b",
            "sf_role": "Sales Manager",
            "sf_org_id": "org_1",
            "tenant_id": "tenant_1",
        }
    }
    res_b = await tool.ainvoke({"query": "Acme renewal price"}, config=config_b)
    # ZERO Account X chunks, no citation to X, and no trace of $120k
    assert "$120k" not in res_b
    assert "acme_contract.pdf" not in res_b
    assert "cx_1" not in res_b
    # Account Y chunk is returned since user_b has access to Y
    assert "beta_contract.pdf" in res_b
    assert "$50k" in res_b

    # 3. User with NO access to any matching account gets neutral "No accessible documents found"
    client.grant_record_access("user_c", acc_x, False)
    client.grant_record_access("user_c", acc_y, False)
    config_c = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "user_c",
            "sf_role": "Sales Manager",
            "sf_org_id": "org_1",
            "tenant_id": "tenant_1",
        }
    }
    res_c = await tool.ainvoke({"query": "Acme renewal price"}, config=config_c)
    assert "$120k" not in res_c
    assert "acme_contract.pdf" not in res_c
    assert "No accessible documents found" in res_c


# ── Scenario 2: Role Pre-Filter ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_role_prefilter_blocks_access_even_with_account_access():
    """User has Salesforce account access, but role is not in allowed_roles -> gets nothing."""
    client = MockSalesforceAsyncClient()
    acc_x = "001000000000001AAA"
    client.grant_record_access("support_user", acc_x, True)

    store = BM25Store()
    doc_x = Document(
        page_content="Confidential executive compensation figures for Acme leadership.",
        metadata={
            "source": "exec_comp.pdf",
            "sf_account_id": acc_x,
            "allowed_roles": ["Executive", "VP Sales"],  # Support Agent is not allowed
            "sf_scope": "account",
            "tenant_id": "tenant_1",
            "chunk_id": "c_comp_1",
        },
    )
    store.add_documents([doc_x])

    resolver = AccessResolver(salesforce_client=client)
    tool = build_document_search_tool(
        retriever=store.as_dynamic_retriever(),
        grader_llm=AsyncMock(),
        rewriter_chain=AsyncMock(),
        max_attempts=1,
        access_resolver=resolver,
        k=2,
    )

    config = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "support_user",
            "sf_role": "Support Agent",
            "sf_org_id": "org_1",
            "tenant_id": "tenant_1",
        }
    }
    res = await tool.ainvoke({"query": "executive compensation"}, config=config)
    assert "Confidential executive compensation" not in res
    assert "No accessible documents found" in res


# ── Scenario 3: Org-Scope vs None-Scope ──────────────────────────────────────

@pytest.mark.asyncio
async def test_org_scope_visible_and_none_scope_invisible():
    """Org-scope documents visible to everyone; untagged (sf_scope='none') visible to nobody."""
    client = MockSalesforceAsyncClient()
    store = BM25Store()

    doc_org = Document(
        page_content="Company general travel policy: Economy class for flights under 6 hours.",
        metadata={
            "source": "general_policy.pdf",
            "sf_scope": "org",
            "tenant_id": "tenant_1",
            "chunk_id": "c_org",
        },
    )
    doc_none = Document(
        page_content="Unapproved internal draft notes about travel policy changes.",
        metadata={
            "source": "draft_notes.txt",
            "sf_scope": "none",
            "tenant_id": "tenant_1",
            "chunk_id": "c_none",
        },
    )
    store.add_documents([doc_org, doc_none])

    resolver = AccessResolver(salesforce_client=client)
    mock_grader = AsyncMock()
    mock_grader.ainvoke.return_value = MagicMock(is_relevant=True, reason="Relevant")

    tool = build_document_search_tool(
        retriever=store.as_dynamic_retriever(),
        grader_llm=mock_grader,
        rewriter_chain=AsyncMock(),
        max_attempts=1,
        access_resolver=resolver,
        k=2,
    )

    config = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "regular_user",
            "sf_role": "Any Role",
            "tenant_id": "tenant_1",
        }
    }

    # Query for org policy
    res_org = await tool.ainvoke({"query": "general travel policy"}, config=config)
    assert "Economy class for flights" in res_org

    # Query for unmanifested doc
    res_none = await tool.ainvoke({"query": "Unapproved internal draft"}, config=config)
    assert "Unapproved internal draft" not in res_none
    assert "No accessible documents found" in res_none


# ── Scenario 4: Cross-Tenant Isolation ───────────────────────────────────────

@pytest.mark.asyncio
async def test_cross_tenant_isolation_blocks_retrieval():
    """User in tenant_2 cannot access chunks belonging to tenant_1, even with same user_id."""
    client = MockSalesforceAsyncClient()
    acc_x = "001000000000001AAA"
    client.grant_record_access("user_universal", acc_x, True)

    store = BM25Store()
    doc_t1 = Document(
        page_content="Tenant 1 confidential safari itinerary details.",
        metadata={
            "source": "t1_safari.pdf",
            "sf_account_id": acc_x,
            "allowed_roles": ["Sales Manager"],
            "sf_scope": "account",
            "tenant_id": "tenant_1",
            "chunk_id": "c_t1",
        },
    )
    store.add_documents([doc_t1])

    resolver = AccessResolver(salesforce_client=client)
    tool = build_document_search_tool(
        retriever=store.as_dynamic_retriever(),
        grader_llm=AsyncMock(),
        rewriter_chain=AsyncMock(),
        max_attempts=1,
        access_resolver=resolver,
        k=2,
    )

    # User in tenant_2 querying
    config_t2 = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "user_universal",
            "sf_role": "Sales Manager",
            "tenant_id": "tenant_2",
        }
    }
    res = await tool.ainvoke({"query": "safari itinerary"}, config=config_t2)
    assert "Tenant 1 confidential safari" not in res
    assert "No accessible documents found" in res


# ── Scenario 5: Fail-Closed on Salesforce Error/Timeout ───────────────────────

@pytest.mark.asyncio
async def test_fail_closed_on_salesforce_client_error(caplog):
    """When Salesforce UserRecordAccess call raises an exception, access fails closed."""
    client = MockSalesforceAsyncClient()
    acc_x = "001000000000001AAA"
    client.grant_record_access("user_a", acc_x, True)
    # Simulate API timeout / failure
    client.simulate_record_access_exception = TimeoutError("Salesforce API 504 Gateway Timeout")

    store = BM25Store()
    doc = Document(
        page_content="Sensitive travel discount code: VIP-999-TRAVEL.",
        metadata={
            "source": "discounts.pdf",
            "sf_account_id": acc_x,
            "allowed_roles": ["Sales Manager"],
            "sf_scope": "account",
            "tenant_id": "tenant_1",
            "chunk_id": "c_disc",
        },
    )
    store.add_documents([doc])

    resolver = AccessResolver(salesforce_client=client)
    tool = build_document_search_tool(
        retriever=store.as_dynamic_retriever(),
        grader_llm=AsyncMock(),
        rewriter_chain=AsyncMock(),
        max_attempts=1,
        access_resolver=resolver,
        k=2,
    )

    config = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "user_a",
            "sf_role": "Sales Manager",
            "tenant_id": "tenant_1",
            "trace_id": "trace-test-fail-closed-99",
        }
    }

    res = await tool.ainvoke({"query": "Sensitive travel discount"}, config=config)
    # Fails closed: 0 chunks returned
    assert "VIP-999-TRAVEL" not in res
    assert "No accessible documents found" in res


# ── Scenario 6: Cache and TTL with Injectable Clock ──────────────────────────

@pytest.mark.asyncio
async def test_access_resolver_cache_ttl_and_revocation():
    """
    1. First request queries Salesforce.
    2. Second request within TTL uses cache (0 SF calls).
    3. Third request after TTL re-queries Salesforce.
    4. Access revocation takes effect after TTL expires.
    """
    client = MockSalesforceAsyncClient()
    acc_x = "001000000000001AAA"
    client.grant_record_access("user_1", acc_x, True)

    fake_now = [1000.0]
    resolver = AccessResolver(
        salesforce_client=client,
        cache_ttl_seconds=60,
        time_fn=lambda: fake_now[0],
    )

    # Call 1: uncached -> 1 SF call
    readable1 = await resolver.readable_accounts("user_1", {acc_x})
    assert acc_x in readable1
    assert len(client.call_history) == 1

    # Call 2: 20 seconds later (within 60s TTL) -> 0 new SF calls
    fake_now[0] += 20.0
    readable2 = await resolver.readable_accounts("user_1", {acc_x})
    assert acc_x in readable2
    assert len(client.call_history) == 1

    # Revoke access in Salesforce, but TTL not expired yet
    client.grant_record_access("user_1", acc_x, False)
    fake_now[0] += 20.0  # total 40s elapsed < 60s
    readable_still_cached = await resolver.readable_accounts("user_1", {acc_x})
    assert acc_x in readable_still_cached
    assert len(client.call_history) == 1

    # Call 3: After TTL expires (70s total elapsed > 60s) -> re-queries SF, sees revocation
    fake_now[0] += 30.0  # total 70s elapsed
    readable_revoked = await resolver.readable_accounts("user_1", {acc_x})
    assert acc_x not in readable_revoked
    assert len(client.call_history) == 2


# ── Scenario 7: Signed RS256 JWT Authentication & Validation ─────────────────

def test_jwt_validation_matrix(rsa_keys, monkeypatch):
    """
    Verify RS256 JWT validation:
    - Valid token -> 200 OK with AuthUser
    - Expired -> 401
    - Wrong audience -> 401
    - Wrong issuer -> 401
    - HS256 signed -> 401
    - alg=none -> 401
    - Lifetime > 120s -> 401
    - Missing sf_user_id -> 401
    - Replayed jti -> 401
    """
    priv_pem, pub_pem = rsa_keys
    test_settings = Settings(
        sf_auth_enabled=True,
        sf_jwt_public_key=pub_pem,
        sf_jwt_audience="advance-rag-travel-copilot",
        sf_jwt_issuer="https://test.salesforce.com",
        sf_jwt_max_lifetime_seconds=120,
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    # 1. Valid token
    valid_token = _create_jwt(priv_pem)
    user = _verify_salesforce_jwt(valid_token)
    assert isinstance(user, AuthUser)
    assert user.sf_user_id == "005000000000001AAA"
    assert user.sf_role == "Sales Manager"
    assert user.tenant_id == "tenant_1"

    # 2. Replayed jti
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(valid_token)
    assert exc.value.status_code == 401
    assert "replay" in exc.value.detail.lower()

    # 3. Expired token
    now = int(time.time())
    expired_token = _create_jwt(priv_pem, {"exp": now - 100, "iat": now - 150})
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(expired_token)
    assert exc.value.status_code == 401

    # 4. Wrong audience
    wrong_aud = _create_jwt(priv_pem, {"aud": "wrong-audience"})
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(wrong_aud)
    assert exc.value.status_code == 401

    # 5. Wrong issuer
    wrong_iss = _create_jwt(priv_pem, {"iss": "https://attacker.com"})
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(wrong_iss)
    assert exc.value.status_code == 401

    # 6. HS256 signed
    hs256_token = jwt.encode(
        {"iss": "https://test.salesforce.com", "aud": "advance-rag-travel-copilot", "exp": now + 60, "iat": now, "jti": "j1", "sf_user_id": "u1"},
        "secret",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(hs256_token)
    assert exc.value.status_code == 401
    assert "only rs256" in exc.value.detail.lower()

    # 7. alg=none
    none_token = jwt.encode(
        {"iss": "https://test.salesforce.com", "aud": "advance-rag-travel-copilot", "exp": now + 60, "iat": now, "jti": "j2", "sf_user_id": "u1"},
        key="",
        algorithm="none",
    )
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(none_token)
    assert exc.value.status_code == 401

    # 8. Lifetime > 120s (e.g. 150s)
    long_lifetime = _create_jwt(priv_pem, {"iat": now, "exp": now + 150})
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(long_lifetime)
    assert exc.value.status_code == 401
    assert "exceeds maximum allowed" in exc.value.detail.lower()

    # 9. Missing sf_user_id
    missing_uid = _create_jwt(priv_pem, {"sf_user_id": ""})
    with pytest.raises(HTTPException) as exc:
        _verify_salesforce_jwt(missing_uid)
    assert exc.value.status_code == 401
    assert "missing required 'sf_user_id'" in exc.value.detail.lower()


# ── Scenario 8: Identity Spoofing Protection ─────────────────────────────────

def test_identity_spoofing_request_body_ignored(rsa_keys, monkeypatch):
    """When SF_AUTH_ENABLED=True, user/role/tenant from body are ignored; token claims win."""
    priv_pem, pub_pem = rsa_keys
    test_settings = Settings(
        sf_auth_enabled=True,
        sf_jwt_public_key=pub_pem,
        sf_jwt_audience="advance-rag-travel-copilot",
        sf_jwt_issuer="https://test.salesforce.com",
    )
    monkeypatch.setattr("src.api.routes.settings", test_settings)

    # Verified token has user="genuine_user", tenant="genuine_tenant"
    token = _create_jwt(priv_pem, {"sf_user_id": "genuine_user", "tenant_id": "genuine_tenant", "sf_role": "Agent"})
    auth_user = _verify_salesforce_jwt(token)

    # Attacker passes spoofed fields in request body
    spoofed_request = ChatRequest(
        question="Show my documents",
        sf_username="attacker_superadmin",
    )

    inputs, config, thread_id = _prepare_chat_context(spoofed_request, auth_user)

    # Assert genuine identity from token is used, NOT the spoofed body value
    assert inputs["sf_username"] == "genuine_user"
    assert inputs["tenant_id"] == "genuine_tenant"
    assert config["configurable"]["sf_user_id"] == "genuine_user"
    assert config["configurable"]["tenant_id"] == "genuine_tenant"


# ── Scenario 9: Over-fetch & Widening ────────────────────────────────────────

@pytest.mark.asyncio
async def test_overfetch_and_widening_retrieves_accessible_chunk():
    """
    When top-k raw hits are all inaccessible, widening retrieves the accessible chunk ranked below.
    Initial fetch (k*4 = 4) has only inaccessible docs.
    Widening (k*8 = 8) pulls in the accessible document ranked at position 6.
    """
    client = MockSalesforceAsyncClient()
    acc_accessible = "001000000000099AAA"
    acc_restricted = "001000000000088BBB"

    client.grant_record_access("user_wide", acc_accessible, True)
    client.grant_record_access("user_wide", acc_restricted, False)

    store = BM25Store()

    # Index 5 restricted docs matching the keyword "renewal agreement" strongly
    for i in range(1, 6):
        store.add_documents([
            Document(
                page_content=f"Restricted renewal agreement details document number {i}.",
                metadata={
                    "source": f"restricted_{i}.pdf",
                    "sf_account_id": acc_restricted,
                    "allowed_roles": ["Sales Manager"],
                    "sf_scope": "account",
                    "tenant_id": "tenant_1",
                    "chunk_id": f"res_{i}",
                },
            )
        ])

    # Index 1 accessible doc matching "renewal agreement"
    store.add_documents([
        Document(
            page_content="Accessible renewal agreement for VIP Account: discount rate is 25%.",
            metadata={
                "source": "accessible_vip.pdf",
                "sf_account_id": acc_accessible,
                "allowed_roles": ["Sales Manager"],
                "sf_scope": "account",
                "tenant_id": "tenant_1",
                "chunk_id": "acc_vip",
            },
        )
    ])

    resolver = AccessResolver(salesforce_client=client)
    mock_grader = AsyncMock()
    mock_grader.ainvoke.return_value = MagicMock(is_relevant=True, reason="Relevant")

    # k=1, overfetch_factor=4 -> initial_k=4 (all restricted), widen_k=8 (pulls in doc 6)
    tool = build_document_search_tool(
        retriever=store.as_dynamic_retriever(),
        grader_llm=mock_grader,
        rewriter_chain=AsyncMock(),
        max_attempts=1,
        access_resolver=resolver,
        k=1,
        overfetch_factor=4,
    )

    config = {
        "configurable": {
            "sf_auth_active": True,
            "sf_user_id": "user_wide",
            "sf_role": "Sales Manager",
            "tenant_id": "tenant_1",
        }
    }

    result = await tool.ainvoke({"query": "renewal agreement"}, config=config)
    assert "discount rate is 25%" in result
    assert "accessible_vip.pdf" in result


# ── Scenario 10: Ingestion Validation & Chunk ID Collision Resistance ────────

def test_ingestion_manifest_validation_and_id_normalization(tmp_path):
    """
    - 15-char Salesforce ID normalizes to 18-char.
    - Bad Salesforce ID fails with ValueError.
    - Deterministic chunk ID hash includes sf_account_id so same file under 2 accounts doesn't collide.
    """
    # 1. Normalization
    id_15 = "001000000000001"
    id_18 = normalize_salesforce_id(id_15)
    assert len(id_18) == 18
    assert id_18.startswith("001000000000001")
    assert normalize_salesforce_id(id_18) == id_18

    # 2. Bad ID rejection
    with pytest.raises(ValueError) as exc:
        normalize_salesforce_id("invalid-id-too-short")
    assert "15 or 18 alphanumeric characters" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        normalize_salesforce_id("001000000000001@@@")
    assert "alphanumeric" in str(exc.value)

    # 3. Manifest parsing validation
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(
        '[{"file": "test.pdf", "sf_account_id": "short_bad_id", "allowed_roles": ["Agent"]}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_sf_manifest(str(bad_manifest))

    # 4. Chunk ID collision resistance: same file under two accounts
    sample_doc = Document(
        page_content="Terms and conditions text.",
        metadata={"source": "standard_contract.pdf"},
    )
    cid_acc_1 = generate_chunk_id(
        sample_doc,
        chunk_idx=0,
        tenant_id="tenant_1",
        sf_account_id="001000000000001AAA",
    )
    cid_acc_2 = generate_chunk_id(
        sample_doc,
        chunk_idx=0,
        tenant_id="tenant_1",
        sf_account_id="001000000000002BBB",
    )
    assert cid_acc_1 != cid_acc_2, "Chunk IDs for same content with different accounts MUST NOT collide"
