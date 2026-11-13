"""
Tests for Automated Inline Per-Claim Citation Verification and Filtering.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from src.retrieval.citations import verify_and_filter_inline_citations
from src.api.schemas import ChatResponse, Citation


def test_inline_citation_verification_all_valid():
    """
    Verify that when claims cite documents and pages that were actually retrieved,
    the citations are marked verified and preserved in the text.
    """
    text = (
        "Gold members receive a full refund if cancelled 48 hours prior [cancellation_policy.pdf, page 3]. "
        "Standard hotel reimbursement cap is $250 per night [hotel_allowance.pdf, page 1]."
    )
    retrieved = [
        {"chunk_id": "c-1", "source": "data/cancellation_policy.pdf", "title": "cancellation_policy.pdf", "page": 3},
        {"chunk_id": "c-2", "source": "data/hotel_allowance.pdf", "title": "hotel_allowance.pdf", "page": 1},
    ]

    verified_text, verified_cites, summary = verify_and_filter_inline_citations(
        text,
        retrieved,
        mode="flag",
    )

    assert "[cancellation_policy.pdf, page 3]" in verified_text
    assert "[hotel_allowance.pdf, page 1]" in verified_text
    assert "UNVERIFIED" not in verified_text
    assert summary["total_inline_citations"] == 2
    assert summary["verified_count"] == 2
    assert summary["unverified_count"] == 0
    assert len(verified_cites) == 2


def test_inline_citation_verification_flags_hallucinated_source():
    """
    Verify that citations pointing to non-retrieved documents are flagged
    as [UNVERIFIED CITATION: ...] in 'flag' mode.
    """
    text = (
        "Complimentary airport limousine is provided [vip_perks_fake.pdf, page 99]. "
        "Cancellation fee is $50 [cancellation_policy.pdf, page 3]."
    )
    retrieved = [
        {"chunk_id": "c-1", "source": "data/cancellation_policy.pdf", "title": "cancellation_policy.pdf", "page": 3},
    ]

    verified_text, verified_cites, summary = verify_and_filter_inline_citations(
        text,
        retrieved,
        mode="flag",
    )

    # Valid citation preserved
    assert "[cancellation_policy.pdf, page 3]" in verified_text
    # Fake citation flagged
    assert "[UNVERIFIED CITATION: vip_perks_fake.pdf, page 99]" in verified_text
    assert summary["total_inline_citations"] == 2
    assert summary["verified_count"] == 1
    assert summary["unverified_count"] == 1
    assert summary["unverified_citations"][0]["source"] == "vip_perks_fake.pdf"


def test_inline_citation_verification_drops_hallucinated_source():
    """
    Verify that in 'drop' mode, hallucinated citations are stripped out cleanly.
    """
    text = (
        "Free spa access is included [phantom_brochure.pdf, page 5]. "
        "Normal check-in time is 3 PM [hotel_policy.pdf, page 2]."
    )
    retrieved = [
        {"chunk_id": "c-2", "source": "policies/hotel_policy.pdf", "title": "hotel_policy.pdf", "page": 2},
    ]

    verified_text, verified_cites, summary = verify_and_filter_inline_citations(
        text,
        retrieved,
        mode="drop",
    )

    assert "[hotel_policy.pdf, page 2]" in verified_text
    assert "phantom_brochure" not in verified_text
    assert "[UNVERIFIED" not in verified_text
    assert "Free spa access is included" in verified_text
    assert summary["total_inline_citations"] == 2
    assert summary["verified_count"] == 1
    assert summary["unverified_count"] == 1


def test_inline_citation_verification_page_mismatch():
    """
    Verify that if a document exists but the cited page does not match the retrieved chunk,
    the citation is flagged.
    """
    text = "Baggage fee waiver applies [baggage.pdf, page 14]."
    retrieved = [
        {"chunk_id": "c-bag-1", "source": "baggage.pdf", "title": "baggage.pdf", "page": 2},
    ]

    verified_text, _, summary = verify_and_filter_inline_citations(
        text,
        retrieved,
        mode="flag",
    )

    assert "[UNVERIFIED CITATION: baggage.pdf, page 14]" in verified_text
    assert summary["unverified_count"] == 1
    assert summary["verified_count"] == 0


def test_empty_retrieved_citations_flags_all_citations():
    """
    If no documents were retrieved in this turn, any inline citation must be flagged.
    """
    text = "The speed of light is 300,000 km/s [physics.pdf, page 1]."
    verified_text, _, summary = verify_and_filter_inline_citations(
        text,
        [],
        mode="flag",
    )
    assert "[UNVERIFIED CITATION: physics.pdf, page 1]" in verified_text
    assert summary["unverified_count"] == 1


def test_chat_response_schema_with_citation_verification():
    """
    Verify ChatResponse accepts and validates citation_verification report.
    """
    response = ChatResponse(
        answer="Cancellation is free within 24h [cancellation.pdf, page 1].",
        citations=[Citation(chunk_id="c-1", source="cancellation.pdf", page=1)],
        citation_verification={
            "total_inline_citations": 1,
            "verified_count": 1,
            "unverified_count": 0,
            "mode": "flag",
        },
    )
    dumped = response.model_dump()
    assert dumped["citation_verification"]["verified_count"] == 1
    assert dumped["citation_verification"]["unverified_count"] == 0
