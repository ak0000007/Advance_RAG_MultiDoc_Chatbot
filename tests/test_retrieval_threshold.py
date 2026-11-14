"""Unit tests for retrieval relevance thresholding in RRF and HybridRetriever.
Ensures low-relevance or unanswerable candidates can be cleanly filtered out
before generation.
"""
from langchain_core.documents import Document
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


def test_rrf_min_score_filtering():
    doc1 = Document(page_content="Document 1 content")
    doc2 = Document(page_content="Document 2 content")
    doc3 = Document(page_content="Document 3 content")

    # Ranked in list 1 only:
    # rank 0 score = 1 / (60 + 0 + 1) = 1/61 = ~0.01639
    # rank 1 score = 1 / (60 + 1 + 1) = 1/62 = ~0.01613
    # rank 2 score = 1 / (60 + 2 + 1) = 1/63 = ~0.01587
    ranked_lists = [[doc1, doc2, doc3]]

    # No filter (default)
    res_all = reciprocal_rank_fusion(ranked_lists, k=60, min_score=None)
    assert len(res_all) == 3

    # Filter with min_score between rank 0 and rank 1
    # 1/61 = 0.0163934..., 1/62 = 0.0161290...
    res_top1 = reciprocal_rank_fusion(ranked_lists, k=60, min_score=0.0162)
    assert len(res_top1) == 1
    assert res_top1[0].page_content == "Document 1 content"

    # Filter with high min_score -> returns empty list
    res_none = reciprocal_rank_fusion(ranked_lists, k=60, min_score=0.05)
    assert len(res_none) == 0


def test_hybrid_retriever_min_score_integration():
    doc1 = Document(page_content="Active corporate travel policy")
    doc2 = Document(page_content="Unrelated noise document")

    retriever1 = lambda q: [doc1]
    retriever2 = lambda q: [doc2]

    # When doc1 appears in both retrievers:
    # score = 1/61 + 1/61 = 2/61 = ~0.03278
    # doc2 appears in list 2 at rank 0: score = 1/61 = ~0.01639
    hybrid = HybridRetriever(
        retrievers=[lambda q: [doc1], lambda q: [doc1, doc2]],
        min_score=0.025,
    )
    results = hybrid.search("travel policy")
    assert len(results) == 1
    assert results[0].page_content == "Active corporate travel policy"

    # High threshold filters everything
    hybrid_strict = HybridRetriever(
        retrievers=[lambda q: [doc1], lambda q: [doc1, doc2]],
        min_score=0.10,
    )
    empty_results = hybrid_strict.search("travel policy")
    assert empty_results == []


def test_calibrate_threshold_optimization_and_guards(tmp_path):
    """Verify calibrate_threshold DEV sweep constraint and refusal guard."""
    from scripts.calibrate_threshold import calibrate_threshold
    import pytest

    # Create dummy JSONL dataset
    fake_dataset = tmp_path / "test_dataset.jsonl"
    cases = [
        {"id": "c1", "split": "dev", "query": "q1", "is_unanswerable": False, "category": "single_doc", "relevant_sources": ["a.txt"], "contains": ["x"]},
        {"id": "c2", "split": "dev", "query": "q2", "is_unanswerable": True, "category": "unanswerable", "relevant_sources": [], "contains": []},
        {"id": "c3", "split": "heldout", "query": "q3", "is_unanswerable": False, "category": "single_doc", "relevant_sources": ["a.txt"], "contains": ["x"]},
        {"id": "c4", "split": "heldout", "query": "q4", "is_unanswerable": True, "category": "unanswerable", "relevant_sources": [], "contains": []},
    ]
    with fake_dataset.open("w", encoding="utf-8") as f:
        for c in cases:
            import json
            f.write(json.dumps(c) + "\n")

    # Run calibration to mock destination
    mock_out = tmp_path / "threshold.mock.json"
    rep = calibrate_threshold(
        dataset_path=fake_dataset,
        output_path=mock_out,
        signal="dense_cosine",
        max_false_abstention=0.05,
        mock_mode=True,
    )

    assert "optimal_threshold" in rep
    assert "true_abstention_wilson_95" in rep["heldout_performance"]
    assert "false_abstention_wilson_95" in rep["heldout_performance"]
    assert rep["mode"] == "mock"

    # Official path rejection
    official_out = tmp_path / "threshold_calibration_report.json"
    with pytest.raises(ValueError, match="Refusing to write mock report to official path"):
        calibrate_threshold(
            dataset_path=fake_dataset,
            output_path=official_out,
            signal="dense_cosine",
            mock_mode=True,
        )

