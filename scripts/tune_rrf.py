"""Grid search tuning for Reciprocal Rank Fusion (RRF) parameters.

Sweeps dense weight w_dense in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] (w_sparse = 1.0)
and rrf_k in [10, 30, 60] strictly on the DEV split.
Evaluates the optimal parameter set on the independent HELD-OUT split
with 95% bootstrap confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.rag.metrics import (
    assert_not_mock_on_official_path,
    bootstrap_confidence_interval,
    compute_file_sha256,
    evaluate_retrieval_case,
    get_git_commit,
    paired_bootstrap_difference,
)
from evals.rag.runner import build_live_retrievers, build_mock_retrievers, load_dataset
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tune_rrf(
    dataset_path: Path,
    output_path: Path,
    mock_mode: bool = False,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    cases = load_dataset(dataset_path)
    dev_cases = [c for c in cases if c.get("split") == "dev" and not c.get("is_unanswerable", False)]
    heldout_cases = [c for c in cases if c.get("split") == "heldout" and not c.get("is_unanswerable", False)]

    logger.info("Loaded %d DEV answerable cases, %d HELDOUT answerable cases", len(dev_cases), len(heldout_cases))

    if mock_mode:
        retrievers = build_mock_retrievers()
    else:
        retrievers = build_live_retrievers(device=device)

    # Pre-retrieve candidate pools per query to avoid redundant compute during grid search
    def _fetch_candidates(query: str):
        dense_docs = retrievers["Dense Only"](query)
        sparse_docs = retrievers["BM25 Only"](query)
        return dense_docs, sparse_docs

    logger.info("Caching dense and sparse candidate pools for DEV queries...")
    dev_pools = []
    for c in dev_cases:
        dense_docs, sparse_docs = _fetch_candidates(c["query"])
        dev_pools.append((c, dense_docs, sparse_docs))

    # Grid search parameters (Task C: dense weights 0.5 to 2.0, k in 10, 30, 60)
    weights_candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0]
    k_candidates = [10, 30, 60]

    grid_results = []
    best_config = None
    best_dev_ndcg = -1.0

    logger.info("Sweeping %d combinations on DEV split...", len(weights_candidates) * len(k_candidates))

    for w_dense in weights_candidates:
        for rrf_k in k_candidates:
            ndcg_scores = []
            mrr_scores = []
            hit1_scores = []

            for c, dense_docs, sparse_docs in dev_pools:
                fused = reciprocal_rank_fusion(
                    [dense_docs, sparse_docs],
                    k=rrf_k,
                    weights=[w_dense, 1.0],
                )
                m = evaluate_retrieval_case(
                    fused[:20],
                    relevant_sources=c.get("relevant_sources", []),
                    contains=c.get("contains", []),
                    k_values=(1, 3, 5),
                )
                ndcg_scores.append(m["ndcg@5"])
                mrr_scores.append(m["mrr"])
                hit1_scores.append(m["hit@1"])

            mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
            mean_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
            mean_hit1 = sum(hit1_scores) / len(hit1_scores) if hit1_scores else 0.0

            result = {
                "w_dense": w_dense,
                "w_sparse": 1.0,
                "rrf_k": rrf_k,
                "dev_ndcg@5": round(mean_ndcg, 4),
                "dev_mrr": round(mean_mrr, 4),
                "dev_hit@1": round(mean_hit1, 4),
            }
            grid_results.append(result)

            if mean_ndcg > best_dev_ndcg:
                best_dev_ndcg = mean_ndcg
                best_config = (w_dense, rrf_k, result)

    assert best_config is not None
    best_w_dense, best_rrf_k, best_dev_metrics = best_config
    logger.info(
        "Optimal DEV Configuration: w_dense=%.2f, rrf_k=%d (DEV nDCG@5: %.4f, MRR: %.4f)",
        best_w_dense,
        best_rrf_k,
        best_dev_ndcg,
        best_dev_metrics["dev_mrr"],
    )

    # Evaluate Tuned, Default (1.0, 1.0, 60), and Dense-Only on HELDOUT split
    logger.info("Evaluating Tuned, Default, and Dense-Only configs on HELDOUT split (%d cases)...", len(heldout_cases))

    tuned_hit1, tuned_hit3, tuned_hit5, tuned_recall5, tuned_mrr, tuned_ndcg5 = [], [], [], [], [], []
    default_hit1, default_ndcg5 = [], []
    dense_hit1, dense_ndcg5 = [], []

    for c in heldout_cases:
        dense_docs, sparse_docs = _fetch_candidates(c["query"])

        # 1. Tuned config
        fused_tuned = reciprocal_rank_fusion(
            [dense_docs, sparse_docs],
            k=best_rrf_k,
            weights=[best_w_dense, 1.0],
        )
        m_tuned = evaluate_retrieval_case(
            fused_tuned[:20],
            relevant_sources=c.get("relevant_sources", []),
            contains=c.get("contains", []),
            k_values=(1, 3, 5),
        )
        tuned_hit1.append(m_tuned["hit@1"])
        tuned_hit3.append(m_tuned["hit@3"])
        tuned_hit5.append(m_tuned["hit@5"])
        tuned_recall5.append(m_tuned["recall@5"])
        tuned_mrr.append(m_tuned["mrr"])
        tuned_ndcg5.append(m_tuned["ndcg@5"])

        # 2. Default baseline config (w_dense=1.0, w_sparse=1.0, rrf_k=60)
        fused_default = reciprocal_rank_fusion(
            [dense_docs, sparse_docs],
            k=60,
            weights=[1.0, 1.0],
        )
        m_default = evaluate_retrieval_case(
            fused_default[:20],
            relevant_sources=c.get("relevant_sources", []),
            contains=c.get("contains", []),
            k_values=(1, 3, 5),
        )
        default_hit1.append(m_default["hit@1"])
        default_ndcg5.append(m_default["ndcg@5"])

        # 3. Dense-only reference
        m_dense = evaluate_retrieval_case(
            dense_docs[:20],
            relevant_sources=c.get("relevant_sources", []),
            contains=c.get("contains", []),
            k_values=(1, 3, 5),
        )
        dense_hit1.append(m_dense["hit@1"])
        dense_ndcg5.append(m_dense["ndcg@5"])

    # Compute 95% bootstrap confidence intervals for heldout evaluation
    heldout_summary = {}
    metric_map = {
        "hit@1": tuned_hit1,
        "hit@3": tuned_hit3,
        "hit@5": tuned_hit5,
        "recall@5": tuned_recall5,
        "mrr": tuned_mrr,
        "ndcg@5": tuned_ndcg5,
    }
    for k_name, vals in metric_map.items():
        mean_v, low, high = bootstrap_confidence_interval(vals, n_resamples=1000, seed=seed)
        heldout_summary[k_name] = {
            "mean": mean_v,
            "ci_lower_95": low,
            "ci_upper_95": high,
        }

    # Paired bootstrap differences on heldout
    diff_tuned_vs_default_hit1 = paired_bootstrap_difference(default_hit1, tuned_hit1, seed=seed)
    diff_tuned_vs_default_ndcg = paired_bootstrap_difference(default_ndcg5, tuned_ndcg5, seed=seed)
    diff_tuned_vs_dense_hit1 = paired_bootstrap_difference(dense_hit1, tuned_hit1, seed=seed)
    diff_tuned_vs_dense_ndcg = paired_bootstrap_difference(dense_ndcg5, tuned_ndcg5, seed=seed)

    if diff_tuned_vs_default_ndcg["significant"] and diff_tuned_vs_default_ndcg["mean_diff"] > 0:
        conclusion = "adopt tuned config: statistically significant improvement over default on held-out split"
        recommend_adoption = True
    else:
        conclusion = "keep default (w_dense=1.0, w_sparse=1.0, rrf_k=60): tuning difference on held-out split is within statistical noise"
        recommend_adoption = False

    dataset_sha256 = compute_file_sha256(dataset_path)
    corpus_manifest_sha256 = compute_file_sha256("data/manifest.json")
    git_commit = get_git_commit()
    mode_str = "mock" if mock_mode else "live"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode_str,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "dataset_total_cases": len(cases),
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "git_commit": git_commit,
        "seed": seed,
        "dev_cases_count": len(dev_cases),
        "heldout_cases_count": len(heldout_cases),
        "default_config": {
            "w_dense": 1.0,
            "w_sparse": 1.0,
            "rrf_k": 60,
        },
        "optimal_config": {
            "w_dense": best_w_dense,
            "w_sparse": 1.0,
            "rrf_k": best_rrf_k,
        },
        "dev_performance": best_dev_metrics,
        "heldout_performance": heldout_summary,
        "heldout_comparisons": {
            "tuned_vs_default": {
                "hit@1": diff_tuned_vs_default_hit1,
                "ndcg@5": diff_tuned_vs_default_ndcg,
            },
            "tuned_vs_dense": {
                "hit@1": diff_tuned_vs_dense_hit1,
                "ndcg@5": diff_tuned_vs_dense_ndcg,
            },
        },
        "recommendation": conclusion,
        "recommend_adoption": recommend_adoption,
        "grid_search_history": grid_results,
    }

    assert_not_mock_on_official_path(output_path, mode_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Tuning report written to %s", output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Tune RRF weights on DEV split and report on HELDOUT split")
    parser.add_argument("--dataset", type=str, default="evals/rag/rag_dataset.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Run with mock retrievers")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    dest = Path(args.output) if args.output else (
        Path("evals/results/rrf_tuning_report.mock.json") if args.mock else Path("evals/results/rrf_tuning_report.json")
    )
    tune_rrf(
        dataset_path=Path(args.dataset),
        output_path=dest,
        mock_mode=args.mock,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
