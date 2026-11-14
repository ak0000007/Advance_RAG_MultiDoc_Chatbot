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
    aggregate_category_metrics,
    evaluate_retrieval_case,
    format_category_markdown_table,
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


def test_multi_source_and_alternate_phrase_matching(sample_documents):
    """Test matching when multiple authoritative sources and alternate phrases are specified."""
    # Matches second source in list and alternate contains phrase
    assert is_chunk_relevant(
        sample_documents[1],
        target_sources=["other_policy.pdf", "safari_terms.docx"],
        contains=["non-existent phrase", "20% non-refundable deposit"],
    ) is True

    # Neither source matches
    assert is_chunk_relevant(
        sample_documents[1],
        target_sources=["other_policy.pdf", "unrelated.pdf"],
        contains=["20% non-refundable deposit"],
    ) is False


def test_aggregate_category_metrics_arithmetic():
    """Test mathematical accuracy of per-category metric aggregation."""
    fake_case_details = [
        {
            "id": "c1",
            "category": "single_doc",
            "is_unanswerable": False,
            "results": {
                "Dense Only": {
                    "metrics": {
                        "hit@1": 1.0,
                        "hit@3": 1.0,
                        "hit@5": 1.0,
                        "recall@5": 1.0,
                        "mrr": 1.0,
                        "ndcg@5": 1.0,
                    }
                }
            },
        },
        {
            "id": "c2",
            "category": "single_doc",
            "is_unanswerable": False,
            "results": {
                "Dense Only": {
                    "metrics": {
                        "hit@1": 0.0,
                        "hit@3": 1.0,
                        "hit@5": 1.0,
                        "recall@5": 0.5,
                        "mrr": 0.5,
                        "ndcg@5": 0.6309,
                    }
                }
            },
        },
        {
            "id": "c3",
            "category": "unanswerable",
            "is_unanswerable": True,
            "results": {
                "Dense Only": {
                    "judge": {"abstention": 1.0, "faithfulness": 1.0}
                }
            },
        },
    ]

    summary = aggregate_category_metrics(fake_case_details, pipelines=["Dense Only"])

    # Category single_doc assertions
    assert "single_doc" in summary
    sd = summary["single_doc"]["Dense Only"]
    assert sd["N"] == 2
    assert sd["note"] == "low sample - indicative only"
    assert sd["hit@1"] == 0.5
    assert sd["hit@3"] == 1.0
    assert sd["hit@5"] == 1.0
    assert sd["recall@5"] == 0.75
    assert sd["mrr"] == 0.75
    assert sd["ndcg@5"] == 0.8155

    # Category unanswerable assertions
    assert "unanswerable" in summary
    ua = summary["unanswerable"]["Dense Only"]
    assert ua["N"] == 1
    assert ua["is_unanswerable"] is True
    assert ua["abstention_rate"] == 1.0
    assert ua["note"] == "low sample - indicative only"

    # Markdown formatting assertion
    md_table = format_category_markdown_table(summary)
    assert "| **single_doc** | Dense Only | 2 | 0.500 | 1.000 | 1.000 | 0.750 | 0.750 | 0.816 | low sample - indicative only |" in md_table
    assert "| **unanswerable** | Dense Only | 1 | N/A | N/A | N/A | N/A | N/A | N/A |" in md_table


def test_bootstrap_confidence_interval_math():
    """Verify bootstrap CI properties on deterministic values."""
    from evals.rag.metrics import bootstrap_confidence_interval

    # All identical values: CI lower and upper match exactly
    mean, low, high = bootstrap_confidence_interval([1.0, 1.0, 1.0, 1.0])
    assert mean == 1.0
    assert low == 1.0
    assert high == 1.0

    # Binary split: mean = 0.5, lower < 0.5 < upper
    mean, low, high = bootstrap_confidence_interval([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], seed=42)
    assert mean == 0.5
    assert 0.0 < low <= 0.5 <= high < 1.0


def test_aggregate_split_metrics():
    """Verify split filtering and aggregation across dev and heldout."""
    from evals.rag.metrics import aggregate_split_metrics

    case_details = [
        {
            "id": "c1",
            "split": "dev",
            "is_unanswerable": False,
            "results": {"Dense": {"metrics": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "recall@5": 1.0, "mrr": 1.0, "ndcg@5": 1.0}}},
        },
        {
            "id": "c2",
            "split": "heldout",
            "is_unanswerable": False,
            "results": {"Dense": {"metrics": {"hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "recall@5": 0.5, "mrr": 0.5, "ndcg@5": 0.7}}},
        },
        {
            "id": "c3",
            "split": "dev",
            "is_unanswerable": True,
            "results": {"Dense": {"judge": {"abstention": 1.0}}},
        },
    ]

    dev_summary = aggregate_split_metrics(case_details, pipelines=["Dense"], split_name="dev")
    assert dev_summary["Dense"]["N"] == 1
    assert dev_summary["Dense"]["hit@1"] == 1.0

    heldout_summary = aggregate_split_metrics(case_details, pipelines=["Dense"], split_name="heldout")
    assert heldout_summary["Dense"]["N"] == 1
    assert heldout_summary["Dense"]["hit@1"] == 0.0

    all_summary = aggregate_split_metrics(case_details, pipelines=["Dense"], split_name=None)
    assert all_summary["Dense"]["N"] == 2
    assert all_summary["Dense"]["hit@1"] == 0.5


def test_distractor_source_rejection():
    """Verify that chunks from distractor sources in data/manifest.json are strictly rejected as hits."""
    from evals.rag.metrics import assert_not_mock_on_official_path, is_chunk_relevant

    # Document originating from a known distractor file
    distractor_doc = Document(
        page_content="Per diem in the archived 2021 expense guidelines was $50 USD per calendar day.",
        metadata={"source": "legacy_2021_expense_and_lodging_policy.txt"},
    )
    # Even if target_sources includes the distractor file and phrase matches, must return False
    assert is_chunk_relevant(
        distractor_doc,
        target_sources=["legacy_2021_expense_and_lodging_policy.txt"],
        contains=["$50 USD per calendar day"],
    ) is False

    # Authoritative doc with same phrase must be accepted
    authoritative_doc = Document(
        page_content="Domestic per diem is $75 per calendar day under active policy.",
        metadata={"source": "business_meals_and_entertainment_rules.txt"},
    )
    assert is_chunk_relevant(
        authoritative_doc,
        target_sources=["business_meals_and_entertainment_rules.txt"],
        contains=["$75 per calendar day"],
    ) is True


def test_assert_not_mock_on_official_path(tmp_path):
    """Verify that mock runs are refused when directed to official paths."""
    from evals.rag.metrics import assert_not_mock_on_official_path

    # Official report path with mode="mock" must raise ValueError
    with pytest.raises(ValueError, match="Refusing to write mock report to official path"):
        assert_not_mock_on_official_path(tmp_path / "rag_report.json", mode="mock")

    with pytest.raises(ValueError, match="Refusing to write mock report to official path"):
        assert_not_mock_on_official_path(tmp_path / "rrf_tuning_report.json", mode="mock")

    # Mock report path ending in .mock.json is allowed
    assert_not_mock_on_official_path(tmp_path / "rag_report.mock.json", mode="mock")

    # Live mode is allowed everywhere
    assert_not_mock_on_official_path(tmp_path / "rag_report.json", mode="live")


def test_paired_bootstrap_difference():
    """Verify paired bootstrap difference arithmetic and significance testing."""
    from evals.rag.metrics import paired_bootstrap_difference

    # Identical performance -> mean diff 0.0, not significant
    scores_a = [1.0, 1.0, 0.0, 1.0, 0.0]
    res_ident = paired_bootstrap_difference(scores_a, scores_a, n_resamples=500, seed=42)
    assert res_ident["mean_diff"] == 0.0
    assert res_ident["significant"] is False
    assert res_ident["conclusion"] == "not significant"

    # Superior performance across 20 cases -> significant positive difference
    scores_low = [0.0] * 20
    scores_high = [1.0] * 20
    res_diff = paired_bootstrap_difference(scores_low, scores_high, n_resamples=500, seed=42)
    assert res_diff["mean_diff"] == 1.0
    assert res_diff["ci_lower_95"] == 1.0
    assert res_diff["ci_upper_95"] == 1.0
    assert res_diff["significant"] is True
    assert res_diff["conclusion"] == "significant"


def test_wilson_score_interval():
    """Verify Wilson score confidence interval computation."""
    from evals.rag.metrics import compute_wilson_score_interval

    p, low, high = compute_wilson_score_interval(successes=10, total=10)
    assert p == 1.0
    assert 0.69 <= low <= 1.0
    assert high == 1.0

    p, low, high = compute_wilson_score_interval(successes=0, total=10)
    assert p == 0.0
    assert low == 0.0
    assert 0.0 <= high <= 0.31


