"""
Unit tests for RAG evaluation metrics (evals/rag/metrics.py).

Tests mathematical correctness of:
- Hit@k
- Recall@k
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)
- Source filename normalization and substring matching
"""

import pytest
from langchain_core.documents import Document

from evals.rag.metrics import (
    evaluate_retrieval_case,
    hit_at_k,
    is_chunk_relevant,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Standard travel cancellation allows 100% refund up to 7 days before departure.",
            metadata={"source": "data/pdf/travel_policy.pdf", "chunk_id": "c1"},
        ),
        Document(
            page_content="Luxury safari packages require a 20% non-refundable deposit upon booking.",
            metadata={"source": "data/docx/safari_terms.docx", "chunk_id": "c2"},
        ),
        Document(
            page_content="Corporate travel expense caps are set at $250 per night for domestic lodging.",
            metadata={"source": "data/pdf/expense_guidelines.pdf", "chunk_id": "c3"},
        ),
        Document(
            page_content="Emergency medical evacuation is included in all premium tier bookings.",
            metadata={"source": "data/pdf/insurance_coverage.pdf", "chunk_id": "c4"},
        ),
    ]


def test_hit_at_k(sample_documents):
    """Test Hit@k returns 1.0 when relevant chunk is within top-k and 0.0 otherwise."""
    # Target is in rank 1 (index 0) -> hit@1 is 1.0
    assert hit_at_k(sample_documents, ["travel_policy.pdf"], k=1) == 1.0
    assert hit_at_k(sample_documents, ["travel_policy.pdf"], k=3) == 1.0

    # Target is in rank 3 (index 2) -> hit@1 is 0.0, hit@3 is 1.0
    assert hit_at_k(sample_documents, ["expense_guidelines.pdf"], k=1) == 0.0
    assert hit_at_k(sample_documents, ["expense_guidelines.pdf"], k=2) == 0.0
    assert hit_at_k(sample_documents, ["expense_guidelines.pdf"], k=3) == 1.0

    # Non-existent target
    assert hit_at_k(sample_documents, ["non_existent_file.pdf"], k=4) == 0.0


def test_recall_at_k(sample_documents):
    """Test Recall@k computes proportion of ground-truth target sources retrieved."""
    # Query expects 2 sources: travel_policy.pdf and safari_terms.docx
    targets = ["travel_policy.pdf", "safari_terms.docx"]

    # At k=1, only travel_policy is found -> 1/2 = 0.5
    assert recall_at_k(sample_documents, targets, k=1) == 0.5

    # At k=2, both are found -> 2/2 = 1.0
    assert recall_at_k(sample_documents, targets, k=2) == 1.0

    # Empty targets return 0.0
    assert recall_at_k(sample_documents, [], k=2) == 0.0


def test_mrr(sample_documents):
    """Test Mean Reciprocal Rank (1 / rank) correctly decays with rank."""
    # Relevant doc at rank 1 -> 1 / 1 = 1.0
    assert mrr(sample_documents, ["travel_policy.pdf"]) == 1.0

    # Relevant doc at rank 2 -> 1 / 2 = 0.5
    assert mrr(sample_documents, ["safari_terms.docx"]) == 0.5

    # Relevant doc at rank 3 -> 1 / 3 ~= 0.3333
    assert pytest.approx(mrr(sample_documents, ["expense_guidelines.pdf"]), 0.001) == 0.3333

    # Relevant doc not found within k
    assert mrr(sample_documents, ["insurance_coverage.pdf"], k=2) == 0.0
    assert mrr(sample_documents, ["unknown.pdf"], k=4) == 0.0


def test_ndcg_at_k(sample_documents):
    """Test Normalized Discounted Cumulative Gain against theoretical values."""
    # Single target at rank 1: DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> nDCG = 1.0
    assert ndcg_at_k(sample_documents, ["travel_policy.pdf"], k=1) == 1.0

    # Single target at rank 2: DCG = 1/log2(3) = 0.6309, IDCG = 1/log2(2) = 1.0 -> nDCG = 0.6309
    expected_ndcg = 1.0 / 1.58496  # 1 / log2(3)
    assert pytest.approx(ndcg_at_k(sample_documents, ["safari_terms.docx"], k=2), 0.001) == expected_ndcg

    # No relevant docs
    assert ndcg_at_k(sample_documents, ["missing.pdf"], k=3) == 0.0


def test_contains_text_matching(sample_documents):
    """Test that contains phrase filter requires textual evidence in chunk content."""
    # Matches both source and contains
    assert is_chunk_relevant(
        sample_documents[0],
        ["travel_policy.pdf"],
        contains=["100% refund"],
    ) is True

    # Source matches, but contains phrase is absent -> NOT relevant
    assert is_chunk_relevant(
        sample_documents[0],
        ["travel_policy.pdf"],
        contains=["flight delays"],
    ) is False

    # Path normalization matches case-insensitive basename
    assert is_chunk_relevant(
        sample_documents[1],
        ["SAFARI_TERMS.DOCX"],
        contains=["safari"],
    ) is True

    # Full case evaluation
    results = evaluate_retrieval_case(
        sample_documents,
        relevant_sources=["travel_policy.pdf"],
        contains=["100% refund"],
        k_values=(1, 3, 5),
    )
    assert results["hit@1"] == 1.0
    assert results["mrr"] == 1.0
    assert results["ndcg@1"] == 1.0
