"""Calibrate retrieval relevance threshold on the DEV split.

Sweeps thresholds to maximize F1 (answerable vs unanswerable),
picks the optimal threshold, and evaluates it on the HELD-OUT split.
Reports True Abstention, False Abstention, Precision, Recall, and F1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.rag.runner import build_live_retrievers, build_mock_retrievers, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_max_rrf_score(ranked_lists, k=60, weights=None) -> float:
    scores = {}
    for list_idx, ranked_docs in enumerate(ranked_lists):
        w = float(weights[list_idx]) if weights and list_idx < len(weights) else 1.0
        for rank, doc in enumerate(ranked_docs):
            doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()
            scores[doc_key] = scores.get(doc_key, 0.0) + (w / (k + rank + 1))
    return max(scores.values()) if scores else 0.0


def evaluate_threshold(items: list[tuple[float, bool]], threshold: float) -> dict[str, float]:
    """
    items: list of (max_score, is_answerable)
    Prediction: answerable if max_score >= threshold, else abstain
    """
    tp = fp = fn = tn = 0
    for score, is_answerable in items:
        pred_answerable = score >= threshold
        if is_answerable and pred_answerable:
            tp += 1
        elif (not is_answerable) and pred_answerable:
            fp += 1
        elif is_answerable and (not pred_answerable):
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    true_abstention = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_abstention = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "threshold": round(threshold, 6),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_abstention_rate": round(true_abstention, 4),
        "false_abstention_rate": round(false_abstention, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibrate_threshold(
    dataset_path: Path,
    output_path: Path,
    mock_mode: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    cases = load_dataset(dataset_path)
    dev_cases = [c for c in cases if c.get("split") == "dev"]
    heldout_cases = [c for c in cases if c.get("split") == "heldout"]

    logger.info("Loaded %d DEV cases (%d answerable, %d unanswerable)",
                len(dev_cases),
                sum(1 for c in dev_cases if not c.get("is_unanswerable")),
                sum(1 for c in dev_cases if c.get("is_unanswerable")))

    if mock_mode:
        retrievers = build_mock_retrievers()
    else:
        retrievers = build_live_retrievers(device=device)

    def _get_max_score(query: str) -> float:
        dense_docs = retrievers["Dense Only"](query)
        sparse_docs = retrievers["BM25 Only"](query)
        return compute_max_rrf_score([dense_docs, sparse_docs])

    logger.info("Computing retrieval fusion scores for DEV cases...")
    dev_items = []
    for c in dev_cases:
        score = _get_max_score(c["query"])
        is_ans = not c.get("is_unanswerable", False)
        dev_items.append((score, is_ans))

    scores = [s for s, _ in dev_items]
    min_s = min(scores) if scores else 0.0
    max_s = max(scores) if scores else 0.0
    logger.info("DEV score range: min=%.6f, max=%.6f", min_s, max_s)

    # Sweep 50 candidate thresholds across score range
    step = (max_s - min_s) / 50 if max_s > min_s else 0.01
    candidate_thresholds = [min_s + i * step for i in range(51)]

    best_thresh_result = None
    best_f1 = -1.0
    sweep_history = []

    for t in candidate_thresholds:
        res = evaluate_threshold(dev_items, t)
        sweep_history.append(res)
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_thresh_result = res

    assert best_thresh_result is not None
    best_thresh = best_thresh_result["threshold"]
    logger.info(
        "Optimal DEV Threshold: %.6f (DEV F1: %.4f, Precision: %.4f, Recall: %.4f, Abstention: %.4f)",
        best_thresh,
        best_thresh_result["f1"],
        best_thresh_result["precision"],
        best_thresh_result["recall"],
        best_thresh_result["true_abstention_rate"],
    )

    # Evaluate optimal threshold on HELDOUT cases
    logger.info("Evaluating optimal threshold on HELDOUT cases (%d cases)...", len(heldout_cases))
    heldout_items = []
    for c in heldout_cases:
        score = _get_max_score(c["query"])
        is_ans = not c.get("is_unanswerable", False)
        heldout_items.append((score, is_ans))

    heldout_result = evaluate_threshold(heldout_items, best_thresh)
    logger.info(
        "HELDOUT Evaluation at threshold %.6f: F1=%.4f, Precision=%.4f, Recall=%.4f, True Abstention=%.4f, False Abstention=%.4f",
        best_thresh,
        heldout_result["f1"],
        heldout_result["precision"],
        heldout_result["recall"],
        heldout_result["true_abstention_rate"],
        heldout_result["false_abstention_rate"],
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "mock" if mock_mode else "live",
        "optimal_threshold": best_thresh,
        "dev_performance": best_thresh_result,
        "heldout_performance": heldout_result,
        "sweep_history": sweep_history,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Calibration report written to %s", output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Calibrate retrieval relevance threshold on DEV split")
    parser.add_argument("--dataset", type=str, default="evals/rag/rag_dataset.jsonl")
    parser.add_argument("--output", type=str, default="evals/results/threshold_calibration_report.json")
    parser.add_argument("--mock", action="store_true", help="Run with mock retrievers")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    calibrate_threshold(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        mock_mode=args.mock,
        device=args.device,
    )


if __name__ == "__main__":
    main()
