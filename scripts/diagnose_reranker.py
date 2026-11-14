"""
Reranker Diagnostic Tool
Analyzes ranking behavior of gold chunks before and after cross-encoder reranking.
Inspects candidate pool depth (top 10 vs top 20), contains phrase matching,
and score direction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.rag.metrics import is_chunk_relevant
from evals.rag.runner import build_live_retrievers

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose_reranker")


def diagnose_case(
    case: dict,
    hybrid_retriever,
    rerank_retriever,
    candidate_limit: int = 10,
) -> dict:
    query = case["query"]
    target_sources = case.get("relevant_sources", [])
    contains = case.get("contains", [])

    # 1. Retrieve Hybrid candidates (full 20)
    all_hybrid_docs = hybrid_retriever(query)
    candidate_pool = all_hybrid_docs[:candidate_limit]

    # Find gold rank in hybrid
    hybrid_rank = None
    for idx, doc in enumerate(all_hybrid_docs, start=1):
        if is_chunk_relevant(doc, target_sources, contains):
            hybrid_rank = idx
            break

    # Check if gold chunk is inside the candidate pool (top 10)
    in_candidate_pool = hybrid_rank is not None and hybrid_rank <= candidate_limit

    # 2. Retrieve Reranked docs
    reranked_docs = rerank_retriever(query)
    rerank_rank = None
    rerank_scores = []
    for idx, doc in enumerate(reranked_docs, start=1):
        score = doc.metadata.get("rerank_score")
        rerank_scores.append(score)
        if is_chunk_relevant(doc, target_sources, contains) and rerank_rank is None:
            rerank_rank = idx

    # 3. Check score direction (should be monotonically decreasing)
    scores_valid = all(
        s1 is None or s2 is None or s1 >= s2
        for s1, s2 in zip(rerank_scores, rerank_scores[1:])
    )

    top_hybrid_source = Path(all_hybrid_docs[0].metadata.get("source", "unknown")).name if all_hybrid_docs else "none"
    top_rerank_source = Path(reranked_docs[0].metadata.get("source", "unknown")).name if reranked_docs else "none"

    return {
        "id": case.get("id"),
        "query": query,
        "hybrid_rank": hybrid_rank,
        "rerank_rank": rerank_rank,
        "in_candidate_pool": in_candidate_pool,
        "scores_decreasing": scores_valid,
        "top_hybrid_source": top_hybrid_source,
        "top_rerank_source": top_rerank_source,
        "top_rerank_score": rerank_scores[0] if rerank_scores else None,
        "top_rerank_preview": reranked_docs[0].page_content[:100].replace("\n", " ") if reranked_docs else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose Reranker ranking discrepancies.")
    parser.add_argument("--dataset", default="evals/rag/rag_dataset.jsonl", help="Dataset path")
    parser.add_argument("--candidate-pool", type=int, default=10, help="Candidate pool size passed to reranker")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    answerable = [c for c in cases if not c.get("is_unanswerable", False)]
    logger.info("Loaded %d answerable cases from %s", len(answerable), dataset_path)

    retrievers = build_live_retrievers()
    hybrid_retriever = retrievers["Hybrid (RRF)"]
    rerank_retriever = retrievers["Hybrid + Rerank"]

    logger.info("\n%-8s | %-12s | %-12s | %-10s | %-30s -> %-30s", "ID", "Hybrid Rank", "Rerank Rank", "Pool (<=10)", "Top Hybrid Source", "Top Rerank Source")
    logger.info("-" * 115)

    degraded = []
    improved = []
    same = []

    for c in answerable:
        diag = diagnose_case(c, hybrid_retriever, rerank_retriever, candidate_limit=args.candidate_pool)
        h_rank = diag["hybrid_rank"]
        r_rank = diag["rerank_rank"]

        h_str = f"Rank {h_rank}" if h_rank else "Not in top 20"
        r_str = f"Rank {r_rank}" if r_rank else "Not in top 5"

        logger.info(
            "%-8s | %-12s | %-12s | %-10s | %-30s -> %-30s",
            diag["id"],
            h_str,
            r_str,
            "Yes" if diag["in_candidate_pool"] else "NO",
            diag["top_hybrid_source"][:30],
            diag["top_rerank_source"][:30],
        )

        if h_rank == 1 and (r_rank is None or r_rank > 1):
            degraded.append(diag)
        elif (h_rank is None or h_rank > 1) and r_rank == 1:
            improved.append(diag)
        else:
            same.append(diag)

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSIS SUMMARY:")
    logger.info("Total Answerable Cases: %d", len(answerable))
    logger.info("Hit@1 Maintained/Same: %d", len(same))
    logger.info("Hit@1 Degraded by Rerank: %d", len(degraded))
    logger.info("Hit@1 Improved by Rerank: %d", len(improved))
    logger.info("=" * 60)

    if degraded:
        logger.info("\nDetailed breakdown of degraded cases:")
        for d in degraded:
            logger.info("  [%s] %s", d["id"], d["query"])
            logger.info("    Hybrid Top: %s | Rerank Top: %s (Score: %s)", d["top_hybrid_source"], d["top_rerank_source"], d["top_rerank_score"])
            logger.info("    Preview: %s...", d["top_rerank_preview"])


if __name__ == "__main__":
    main()
