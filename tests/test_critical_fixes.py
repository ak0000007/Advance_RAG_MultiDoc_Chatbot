import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.clients.salesforce_client import SalesforceAsyncClient, _ALLOWED_SOBJECTS
from src.rag.grading import build_retrieval_grader, RetrievalGrade
from src.write_guards import bulk_intent_guard
from src.graph.state import RAGState


def test_soql_injection_rejection():
    """Verify that disallowed sobject strings raise ValueError."""
    with patch.object(SalesforceAsyncClient, '_load_private_key', return_value='fake'):
        client = SalesforceAsyncClient('id', 'https://login.salesforce.com', 'k.pem', 'https://example.com')
        with pytest.raises(ValueError, match="Invalid sobject"):
            import asyncio
            asyncio.run(client._resolve_id_by_name("tok", "Custom_Bad_Object__c; DROP TABLE", "Booking1"))


def test_allowed_sobjects_contain_core_types():
    """Verify that allowed sobjects contain all operational CRM entities."""
    for expected in ["Booking__c", "Travel_Booking__c", "Travel_Package__c", "Payment__c", "Opportunity"]:
        assert expected in _ALLOWED_SOBJECTS


def test_grader_prompt_formatting():
    """Verify C4 fix: grader prompt can format with context and question without KeyError."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_llm
    grader_chain = build_retrieval_grader(mock_llm)
    
    prompt = grader_chain.first
    prompt_val = prompt.invoke({"question": "When is trip?", "context": "Trip is tomorrow."})
    assert len(prompt_val.messages) == 2
    # Verify format instructions were rendered
    assert "relevant" in prompt_val.messages[0].content


def test_bulk_intent_guard_no_false_positives():
    """Verify H7 fix: 'Small group' and 'Call' do not trigger false positive bulk detection."""
    assert bulk_intent_guard("Small group trip to Paris") is None
    assert bulk_intent_guard("Call client regarding booking") is None
    assert bulk_intent_guard("Install equipment") is None
    
    # Genuine bulk intent is still blocked
    assert bulk_intent_guard("Close all deals") is not None
    assert bulk_intent_guard("Update every booking") is not None


@pytest.mark.asyncio
async def test_routes_auth_and_resume_username():
    """Verify C2 & C3 fixes in routes."""
    from src.api.routes import _verify_auth_and_get_user, _build_config
    from src.config import settings

    # Without API secret key (dev mode) -> allows access
    with patch.object(settings, "api_secret_key", None):
        user = _verify_auth_and_get_user(None, None)
        assert user == settings.sf_default_username

    # With API secret key -> rejects missing credentials
    with patch.object(settings, "api_secret_key", "secret123"):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _verify_auth_and_get_user(None, None)
        assert exc.value.status_code == 401

        # Passes with correct API key
        user = _verify_auth_and_get_user(None, "secret123")
        assert user == settings.sf_default_username

    # Verify config builder includes sf_username
    config = _build_config("thread_1", "agent@salesforce.com")
    assert config["configurable"]["thread_id"] == "thread_1"
    assert config["configurable"]["sf_username"] == "agent@salesforce.com"
