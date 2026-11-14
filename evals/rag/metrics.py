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
import numpy as np
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


def aggregate_category_metrics(
    case_details: Sequence[dict[str, Any]],
    pipelines: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """
    Compute per-category aggregated retrieval metrics across pipelines.
    Flags categories with N < 10 as 'low sample - indicative only'.
    """
    # Group cases by category
    categories: list[str] = []
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in case_details:
        cat = case.get("category", "unclassified")
        if cat not in by_category:
            categories.append(cat)
            by_category[cat] = []
        by_category[cat].append(case)

    category_summary: dict[str, dict[str, Any]] = {}
    metric_keys = ("hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@5")

    for cat in categories:
        cases = by_category[cat]
        n_cases = len(cases)
        note = "low sample - indicative only" if n_cases < 10 else "statistically representative"
        category_summary[cat] = {}

        for p_name in pipelines:
            is_unanswerable_cat = all(c.get("is_unanswerable", False) for c in cases)
            if is_unanswerable_cat:
                abstentions = [
                    c.get("results", {}).get(p_name, {}).get("judge", {}).get("abstention")
                    for c in cases
                    if c.get("results", {}).get(p_name, {}).get("judge", {}).get("abstention") is not None
                ]
                rate = round(sum(abstentions) / len(abstentions), 4) if abstentions else None
                category_summary[cat][p_name] = {
                    "N": n_cases,
                    "is_unanswerable": True,
                    "abstention_rate": rate,
                    "note": note,
                }
                continue

            # Answerable category: aggregate metrics
            cat_metrics: dict[str, list[float]] = {k: [] for k in metric_keys}
            for c in cases:
                if c.get("is_unanswerable", False):
                    continue
                p_res = c.get("results", {}).get(p_name, {})
                m = p_res.get("metrics") or {}
                for k in metric_keys:
                    val = m.get(k)
                    if val is not None:
                        cat_metrics[k].append(float(val))

            aggregated = {
                "N": n_cases,
                "note": note,
            }
            for k in metric_keys:
                vals = cat_metrics[k]
                aggregated[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

            category_summary[cat][p_name] = aggregated

    return category_summary


def format_category_markdown_table(category_summary: dict[str, dict[str, Any]]) -> str:
    """Format category summary dict into a clean Markdown table."""
    lines = [
        "| Category | Retrieval Pipeline | N | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@5 | Note |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for cat, p_dict in category_summary.items():
        for p_name, metrics in p_dict.items():
            n = metrics.get("N", 0)
            note = metrics.get("note", "")
            if metrics.get("is_unanswerable", False):
                abstention = metrics.get("abstention_rate")
                abst_str = f"Abstention: {abstention:.3f}" if abstention is not None else "Unanswerable test"
                lines.append(
                    f"| **{cat}** | {p_name} | {n} | N/A | N/A | N/A | N/A | N/A | N/A | {abst_str} ({note}) |"
                )
            else:
                lines.append(
                    f"| **{cat}** | {p_name} | {n} | "
                    f"{metrics.get('hit@1', 0.0):.3f} | {metrics.get('hit@3', 0.0):.3f} | {metrics.get('hit@5', 0.0):.3f} | "
                    f"{metrics.get('recall@5', 0.0):.3f} | {metrics.get('mrr', 0.0):.3f} | {metrics.get('ndcg@5', 0.0):.3f} | "
                    f"{note} |"
                )

    return "\n".join(lines)


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute mean and empirical bootstrap confidence interval [lower, upper].
    Returns (mean, ci_lower, ci_upper).
    """
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return (round(mean_val, 4), round(mean_val, 4), round(mean_val, 4))

    rng = np.random.default_rng(seed)
    resamples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
    resample_means = np.mean(resamples, axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(resample_means, 100.0 * alpha))
    upper = float(np.percentile(resample_means, 100.0 * (1.0 - alpha)))
    return (round(mean_val, 4), round(lower, 4), round(upper, 4))


def aggregate_split_metrics(
    case_details: Sequence[dict[str, Any]],
    pipelines: Sequence[str],
    split_name: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """
    Aggregate metrics across pipelines filtered to a specific split ('dev' or 'heldout').
    If split_name is None, computes across all cases.
    Includes 95% bootstrap confidence intervals for key metrics.
    """
    filtered = [
        c for c in case_details
        if not c.get("is_unanswerable", False) and (split_name is None or c.get("split") == split_name)
    ]
    n_cases = len(filtered)
    metric_keys = ("hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@5")
    summary: dict[str, dict[str, Any]] = {}

    for p_name in pipelines:
        cat_metrics: dict[str, list[float]] = {k: [] for k in metric_keys}
        for c in filtered:
            m = c.get("results", {}).get(p_name, {}).get("metrics") or {}
            for k in metric_keys:
                val = m.get(k)
                if val is not None:
                    cat_metrics[k].append(float(val))

        p_summary: dict[str, Any] = {"N": n_cases}
        ci_dict: dict[str, dict[str, float]] = {}
        for k in metric_keys:
            vals = cat_metrics[k]
            mean_val, ci_low, ci_high = bootstrap_confidence_interval(vals, n_resamples=1000)
            p_summary[k] = mean_val
            ci_dict[k] = {"mean": mean_val, "ci_lower": ci_low, "ci_upper": ci_high}

        p_summary["bootstrap_ci_95"] = ci_dict
        summary[p_name] = p_summary

    return summary

