"""
Unit and integration tests for the Source Citation Pipeline (P1).

Verifies:
- Document citation formatting (chunk_id, source, title, page).
- Citation extraction and deduplication from XML text and messages.
- Pydantic schema validation for Citation and ChatResponse.
- Preservation and propagation of citations through document_search, tool_node, and API routes.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from src.retrieval.citations import (
    format_document_citation,
    extract_citations_from_text,
    extract_citations_from_messages,
)
from src.api.schemas import Citation, ChatResponse
from src.tools.document_search import build_document_search_tool


# ----------------------------------------------------------------------
# 1. format_document_citation Tests
# ----------------------------------------------------------------------

def test_format_document_citation_complete_metadata():
    """Verify complete metadata formatting into XML tag and dict."""
    doc = Document(
        page_content="Flight cancellation requires 24 hours advance notice.\x00",
        metadata={
            "chunk_id": "c1-uuid-1234",
            "source": "data/travel_policy.pdf",
            "file_name": "travel_policy.pdf",
            "title": "Corporate Travel Cancellation Policy",
            "page": 4,
        },
    )
    xml_tag, citation = format_document_citation(doc, index=1)

    assert 'id="c1-uuid-1234"' in xml_tag
    assert 'source="data/travel_policy.pdf"' in xml_tag
    assert 'title="Corporate Travel Cancellation Policy"' in xml_tag
    assert 'page="4"' in xml_tag
    assert "\x00" not in xml_tag
    assert "Flight cancellation requires 24 hours advance notice." in xml_tag

    assert citation["chunk_id"] == "c1-uuid-1234"
    assert citation["source"] == "data/travel_policy.pdf"
    assert citation["title"] == "Corporate Travel Cancellation Policy"
    assert citation["page"] == 4
    assert citation["snippet"].startswith("Flight cancellation")


def test_format_document_citation_minimal_metadata():
    """Verify minimal metadata produces clean, backwards-compatible XML tag."""
    doc = Document(
        page_content="Generic policy without source or title.",
        metadata={"id": "chunk-minimal-99"},
    )
    xml_tag, citation = format_document_citation(doc, index=2)

    assert xml_tag == '<document id="chunk-minimal-99" source="unknown">\nGeneric policy without source or title.\n</document>'
    assert citation["chunk_id"] == "chunk-minimal-99"
    assert citation["source"] == "unknown"
    assert citation["title"] is None
    assert citation["page"] is None


# ----------------------------------------------------------------------
# 2. extract_citations_from_text Tests
# ----------------------------------------------------------------------

def test_extract_citations_from_text_multi_doc():
    """Verify multi-document XML parsing and field extraction."""
    raw_text = (
        '<document id="uuid-1" source="policies/hotels.pdf" title="Hotel Guidelines" page="2">\n'
        'Standard hotel rate cap is $250/night.\n'
        '</document>\n\n'
        '<document id="uuid-2" source="policies/flights.pdf" title="Flight Booking Policy" page="7">\n'
        'Economy class must be booked for domestic trips.\n'
        '</document>'
    )
    citations = extract_citations_from_text(raw_text)

    assert len(citations) == 2
    assert citations[0]["chunk_id"] == "uuid-1"
    assert citations[0]["title"] == "Hotel Guidelines"
    assert citations[0]["page"] == 2
    assert "$250/night" in citations[0]["snippet"]

    assert citations[1]["chunk_id"] == "uuid-2"
    assert citations[1]["title"] == "Flight Booking Policy"
    assert citations[1]["page"] == 7


def test_extract_citations_deduplication():
    """Verify that multiple occurrences of identical chunk_id are deduplicated."""
    raw_text = (
        '<document id="uuid-duplicate" source="doc.pdf" title="Duplicate Doc" page="1">\n'
        'First appearance.\n'
        '</document>\n\n'
        '<document id="uuid-duplicate" source="doc.pdf" title="Duplicate Doc" page="1">\n'
        'Second appearance.\n'
        '</document>'
    )
    citations = extract_citations_from_text(raw_text)
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "uuid-duplicate"


def test_extract_citations_fallback_title():
    """Verify fallback title derivation from source filename when title attribute omitted."""
    raw_text = (
        '<document id="uuid-3" source="handbooks/employee_benefits_v2.docx">\n'
        'Benefits guidelines.\n'
        '</document>'
    )
    citations = extract_citations_from_text(raw_text)
    assert len(citations) == 1
    assert citations[0]["title"] == "employee_benefits_v2.docx"


# ----------------------------------------------------------------------
# 3. extract_citations_from_messages Tests
# ----------------------------------------------------------------------

def test_extract_citations_from_messages():
    """Verify extraction across ToolMessage and AIMessage sequences."""
    messages = [
        HumanMessage(content="What is the hotel allowance?"),
        ToolMessage(
            content='<document id="c-101" source="travel_policy.pdf" title="Travel Policy" page="12">\nHotel limit $300.\n</document>',
            tool_call_id="call-1",
        ),
        AIMessage(content="According to the [Travel Policy, Page 12], the limit is $300."),
    ]
    citations = extract_citations_from_messages(messages)
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "c-101"
    assert citations[0]["title"] == "Travel Policy"
    assert citations[0]["page"] == 12


# ----------------------------------------------------------------------
# 4. Pydantic Citation & ChatResponse Schema Tests
# ----------------------------------------------------------------------

def test_citation_pydantic_schema():
    """Verify Citation and ChatResponse Pydantic validation and JSON serialization."""
    citation = Citation(
        chunk_id="chk-abc-001",
        source="policies/reimbursement.pdf",
        title="Reimbursement Policy",
        page=3,
        snippet="Submit receipts within 30 days.",
    )
    response = ChatResponse(
        answer="You must submit receipts within 30 days.",
        citations=[citation],
    )

    data = response.model_dump()
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "chk-abc-001"
    assert data["citations"][0]["title"] == "Reimbursement Policy"
    assert data["citations"][0]["page"] == 3


# ----------------------------------------------------------------------
# 5. Document Search Tool Integration
# ----------------------------------------------------------------------

class MockSimpleRetriever:
    async def ainvoke(self, payload, config=None):
        return [
            Document(
                page_content="Meals up to $75/day eligible for reimbursement.",
                metadata={
                    "chunk_id": "meal-policy-chunk-5",
                    "source": "policies/meals.pdf",
                    "file_name": "meals.pdf",
                    "title": "Per Diem Meal Policy",
                    "page": 5,
                },
            )
        ]


class MockSimpleGrader:
    class Grade:
        relevant = True
        reason = "Relevant"
    async def ainvoke(self, payload, config=None):
        return self.Grade()


@pytest.mark.asyncio
async def test_document_search_tool_outputs_formatted_citations():
    """Verify search_documents tool returns XML enriched with citations."""
    retriever = MockSimpleRetriever()
    grader = MockSimpleGrader()
    tool = build_document_search_tool(retriever, grader, None, max_attempts=1)

    result = await tool.ainvoke({"query": "What is meal allowance?"})
    assert '<document id="meal-policy-chunk-5" source="policies/meals.pdf" title="Per Diem Meal Policy" page="5">' in result
    assert "Meals up to $75/day eligible for reimbursement." in result

    # Extract back from result
    extracted = extract_citations_from_text(result)
    assert len(extracted) == 1
    assert extracted[0]["chunk_id"] == "meal-policy-chunk-5"
    assert extracted[0]["title"] == "Per Diem Meal Policy"
    assert extracted[0]["page"] == 5


# ----------------------------------------------------------------------
# 6. Salesforce Action & Cross-Turn Isolation Tests
# ----------------------------------------------------------------------

def test_extract_citations_cross_turn_isolation():
    """Verify that document citations from Turn 1 do NOT leak into Turn 2 (Salesforce query)."""
    messages = [
        # Turn 1: Document search
        HumanMessage(content="What is the cancellation policy?"),
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "cancellation"}, "id": "call-1"}]),
        ToolMessage(
            name="search_documents",
            content='<document id="doc-turn1" source="policy.pdf" title="Policy" page="1">\nCancel rules.\n</document>',
            tool_call_id="call-1",
        ),
        AIMessage(content="According to the policy, cancellation must be 24h prior."),

        # Turn 2: Salesforce CRM query
        HumanMessage(content="Now look up booking BK-000001 in Salesforce"),
        AIMessage(content="", tool_calls=[{"name": "get_booking", "args": {"booking_id_or_name": "BK-000001"}, "id": "call-2"}]),
        ToolMessage(
            name="get_booking",
            content='Found 1 booking: BK-000001, status: Confirmed, travelers: 2',
            tool_call_id="call-2",
        ),
        AIMessage(content="Your booking BK-000001 is confirmed for 2 travelers."),
    ]

    # Current turn only should extract ZERO citations because Turn 2 is a Salesforce CRM query
    turn2_citations = extract_citations_from_messages(messages, current_turn_only=True)
    assert turn2_citations == []


def test_salesforce_update_action_has_no_citations():
    """Verify that a Salesforce write/action turn produces zero document citations."""
    messages = [
        HumanMessage(content="Update booking BK-000001 to 4 travelers"),
        AIMessage(content="", tool_calls=[{"name": "update_booking", "args": {"record_id": "a001", "travelers": 4}, "id": "call-update"}]),
        ToolMessage(
            name="update_booking",
            content='{"__requires_approval__": true, "action": "update_booking", "record_id": "a001"}',
            tool_call_id="call-update",
        ),
        AIMessage(content="I have prepared the update for booking BK-000001. Please confirm approval."),
    ]

    citations = extract_citations_from_messages(messages, current_turn_only=True)
    assert citations == []


def test_xml_attribute_injection_sanitization():
    """Verify attribute breakout payload is escaped safely in XML and extracted correctly."""
    doc = Document(
        page_content="Policy text.",
        metadata={
            "chunk_id": 'c-123" onclick="alert(1)',
            "source": 'travel_policy.pdf" onload="malicious()',
            "title": 'Policy "Breakout" Title',
            "page": 1,
        },
    )
    xml_tag, citation = format_document_citation(doc)
    # Attribute quotes are escaped, preventing tag attribute breakout
    assert 'onload="malicious()"' not in xml_tag
    assert 'onclick="alert(1)"' not in xml_tag
    assert '&quot; onload=&quot;malicious()' in xml_tag

    # Extracted citations roundtrip the original string safely
    extracted = extract_citations_from_text(xml_tag)
    assert len(extracted) == 1
    assert extracted[0]["source"] == 'travel_policy.pdf" onload="malicious()'
    assert extracted[0]["title"] == 'Policy "Breakout" Title'
    assert extracted[0]["chunk_id"] == 'c-123" onclick="alert(1)'



def test_salesforce_resume_turn_has_no_citations():
    """Verify that resuming a Salesforce write execution produces zero document citations."""
    messages = [
        HumanMessage(content="Update opportunity 0065g000001AAAAAA1 stage to Closed Won"),
        AIMessage(content="", tool_calls=[{"name": "update_salesforce_opportunity_status", "args": {"opportunity_id": "0065g000001AAAAAA1", "new_status": "Closed Won"}, "id": "call-opp"}]),
        ToolMessage(
            name="update_salesforce_opportunity_status",
            content='{"__requires_approval__": true, "action": "update_salesforce_opportunity_status"}',
            tool_call_id="call-opp",
        ),
        # Resumed turn message
        AIMessage(content="Opportunity 0065g000001AAAAAA1 status updated to Closed Won."),
    ]

    citations = extract_citations_from_messages(messages, current_turn_only=True)
    assert citations == []


def test_xml_attribute_injection_sanitization():
    """Verify attribute breakout payload is escaped safely in XML and extracted correctly."""
    doc = Document(
        page_content="Policy text.",
        metadata={
            "chunk_id": 'c-123" onclick="alert(1)',
            "source": 'travel_policy.pdf" onload="malicious()',
            "title": 'Policy "Breakout" Title',
            "page": 1,
        },
    )
    xml_tag, citation = format_document_citation(doc)
    # Attribute quotes are escaped, preventing tag attribute breakout
    assert 'onload="malicious()"' not in xml_tag
    assert 'onclick="alert(1)"' not in xml_tag
    assert '&quot; onload=&quot;malicious()' in xml_tag

    # Extracted citations roundtrip the original string safely
    extracted = extract_citations_from_text(xml_tag)
    assert len(extracted) == 1
    assert extracted[0]["source"] == 'travel_policy.pdf" onload="malicious()'
    assert extracted[0]["title"] == 'Policy "Breakout" Title'
    assert extracted[0]["chunk_id"] == 'c-123" onclick="alert(1)'


