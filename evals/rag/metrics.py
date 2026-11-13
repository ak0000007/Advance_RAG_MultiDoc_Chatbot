"""
Retrieval metrics for RAG evaluation.

Evaluates retrieval quality without requiring hardcoded chunk IDs:
A retrieved chunk is considered relevant if:
1. Its source path, title, or filename matches one of the expected relevant sources.
2. (Optional) Any of the specified `contains` substrings appears in the chunk text.

Metrics implemented:
- Hit@k: 1.0 if at least one relevant document is in top-k, else 0.0.
- Recall@k: Proportion of target sources captured in top-k.
- MRR (Mean Reciprocal Rank): 1 / rank of first relevant document.
- nDCG@k (Normalized Discounted Cumulative Gain): Graded relevance ranking quality.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence
from langchain_core.documents import Document


def _normalize_source_name(source: str) -> str:
    """Normalize file name or path for comparison (case-insensitive basename)."""
    if not source:
        return ""
    # Strip paths and whitespace, lowercase
    p = Path(source.strip()).name.lower()
    return p


def doc_matches_source(doc: Document, target_source: str) -> bool:
    """Check if document metadata matches target source name."""
    norm_target = _normalize_source_name(target_source)
    if not norm_target:
        return False

    meta = getattr(doc, "metadata", {}) or {}
    candidates = [
        meta.get("source"),
        meta.get("file_name"),
        meta.get("title"),
        meta.get("document_name"),
    ]

    for cand in candidates:
        if cand and norm_target == _normalize_source_name(str(cand)):
            return True

    return False


def doc_matches_contains(doc: Document, contains: Optional[Sequence[str]]) -> bool:
    """Check if document content contains any of the required phrases."""
    if not contains:
        return True

    text = (getattr(doc, "page_content", "") or "").lower()
    for phrase in contains:
        if phrase and phrase.strip().lower() in text:
            return True

    return False


def is_chunk_relevant(
    doc: Document,
    target_sources: Sequence[str],
    contains: Optional[Sequence[str]] = None,
) -> bool:
    """
    Check if a retrieved document chunk is relevant to the query.
    Requires source match AND contains phrase match (if contains specified).
    """
    source_match = any(doc_matches_source(doc, src) for src in target_sources)
    if not source_match:
        return False

    return doc_matches_contains(doc, contains)


def hit_at_k(
    retrieved_docs: Sequence[Document],
    relevant_sources: Sequence[str],
    contains: Optional[Sequence[str]] = None,
    k: int = 5,
) -> float:
    """
    Compute Hit@k: returns 1.0 if at least one relevant chunk is in top-k, else 0.0.
    """
    if not relevant_sources:
        return 0.0

    top_k = retrieved_docs[:k]
    for doc in top_k:
        if is_chunk_relevant(doc, relevant_sources, contains):
            return 1.0
    return 0.0


def recall_at_k(
    retrieved_docs: Sequence[Document],
    relevant_sources: Sequence[str],
    contains: Optional[Sequence[str]] = None,
    k: int = 5,
) -> float:
    """
    Compute Recall@k: proportion of expected relevant sources captured in top-k.
    """
    if not relevant_sources:
        return 0.0

    norm_targets = {_normalize_source_name(s) for s in relevant_sources if s}
    if not norm_targets:
        return 0.0

    found_sources = set()
    top_k = retrieved_docs[:k]

    for doc in top_k:
        for target in norm_targets:
            if doc_matches_source(doc, target) and doc_matches_contains(doc, contains):
                found_sources.add(target)

    return len(found_sources) / len(norm_targets)


def mrr(
    retrieved_docs: Sequence[Document],
    relevant_sources: Sequence[str],
    contains: Optional[Sequence[str]] = None,
    k: int = 10,
) -> float:
    """
    Compute Reciprocal Rank: 1 / (rank of first relevant chunk).
    Returns 0.0 if no relevant chunk found in top-k.
    """
    if not relevant_sources:
        return 0.0

    top_k = retrieved_docs[:k]
    for idx, doc in enumerate(top_k):
        if is_chunk_relevant(doc, relevant_sources, contains):
            return 1.0 / (idx + 1)

    return 0.0


def ndcg_at_k(
    retrieved_docs: Sequence[Document],
    relevant_sources: Sequence[str],
    contains: Optional[Sequence[str]] = None,
    k: int = 5,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k (nDCG@k).
    Binary relevance: 1 if relevant, 0 otherwise.
    """
    if not relevant_sources:
        return 0.0

    top_k = retrieved_docs[:k]
    dcg = 0.0
    for idx, doc in enumerate(top_k):
        rel = 1.0 if is_chunk_relevant(doc, relevant_sources, contains) else 0.0
        if rel > 0:
            dcg += rel / math.log2(idx + 2)  # idx 0 -> log2(2) = 1.0

    # Ideal DCG: all relevant documents placed at the top
    total_relevant = len(relevant_sources)
    ideal_hits = min(total_relevant, k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_case(
    retrieved_docs: Sequence[Document],
    relevant_sources: Sequence[str],
    contains: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Evaluate all retrieval metrics for a single query case.
    """
    results: dict[str, float] = {
        "mrr": mrr(retrieved_docs, relevant_sources, contains, k=max(k_values)),
    }
    for k in k_values:
        results[f"hit@{k}"] = hit_at_k(retrieved_docs, relevant_sources, contains, k=k)
        results[f"recall@{k}"] = recall_at_k(retrieved_docs, relevant_sources, contains, k=k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_docs, relevant_sources, contains, k=k)

    return results
