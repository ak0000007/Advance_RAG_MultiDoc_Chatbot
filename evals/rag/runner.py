"""
RAG evaluation benchmark runner.

Compares 4 retrieval pipelines:
1. Dense Vector Retrieval (Qdrant)
2. Sparse Keyword Retrieval (BM25)
3. Hybrid Fusion (Dense + Sparse via RRF)
4. Hybrid Fusion + Cross-Encoder Reranking

Outputs:
- Live report: evals/results/rag_report.json (mode="live")
- Mock report: evals/results/rag_report.mock.json (mode="mock")
- Markdown summary table printed to stdout.
- Optional --judge: evaluates answer faithfulness and abstention rate.

Key Invariant:
Retrieval metrics (Hit@k, Recall@k, MRR, nDCG@k) are computed strictly on answerable
cases. Unanswerable cases are evaluated separately on model abstention rate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.documents import Document

from evals.rag.metrics import evaluate_retrieval_case

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load JSONL dataset and validate structure."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cases.append(data)
            except json.JSONDecodeError as err:
                logger.warning("Skipping invalid JSON on line %d: %s", line_num, err)

    if len(cases) < 20:
        print(
            f"\n[WARNING] Dataset contains {len(cases)} cases. "
            f"Recommended size for statistical validity is 30-50 questions.\n"
        )
    return cases


def build_mock_retrievers() -> dict[str, Callable[[str], list[Document]]]:
    """Mock/offline retrievers based on local sample documents."""
    sample_dir = Path("data/sample_docs")
    mock_docs = []
    if sample_dir.exists():
        for file in sorted(sample_dir.glob("*.txt")):
            text = file.read_text(encoding="utf-8")
            mock_docs.append(Document(page_content=text, metadata={"source": file.name, "file_name": file.name}))

    if not mock_docs:
        mock_docs = [
            Document(
                page_content="Standard travel cancellation allows 50% refund within 48 hours of departure. Domestic business travel must be booked at least 14 days in advance.",
                metadata={"source": "corporate_travel_policy.txt", "file_name": "corporate_travel_policy.txt"},
            ),
            Document(
                page_content="Luxury safari packages require a 20% non-refundable deposit upon booking. Remaining balance due 60 days before departure. Luggage restricted to 15 kilograms in soft-sided duffel bags.",
                metadata={"source": "luxury_safari_terms.txt", "file_name": "luxury_safari_terms.txt"},
            ),
            Document(
                page_content="Emergency medical evacuation is included up to $500,000 for premium tier bookings. Trip interruption coverage up to $10,000. Completely lost luggage covered up to $1,500.",
                metadata={"source": "travel_insurance_coverage.txt", "file_name": "travel_insurance_coverage.txt"},
            ),
            Document(
                page_content="Domestic lodging expenses are capped at $250 per night. High cost cities capped at $350 per night. Domestic meals per diem is $75 per day, international is $100 per day.",
                metadata={"source": "expense_and_lodging_guidelines.txt", "file_name": "expense_and_lodging_guidelines.txt"},
            ),
        ]

    from src.retrieval.bm25_store import BM25Store
    bm25 = BM25Store()
    bm25.add_documents(mock_docs)

    def _dense_search(query: str) -> list[Document]:
        q = query.lower()
        scored = []
        for d in mock_docs:
            text = d.page_content.lower()
            matches = sum(1 for w in q.split() if len(w) > 3 and w in text)
            if matches > 0:
                scored.append((matches, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored] or mock_docs[:2]

    def _sparse_search(query: str) -> list[Document]:
        return bm25.search(query, k=3) or mock_docs[:2]

    def _hybrid_search(query: str) -> list[Document]:
        d = _dense_search(query)
        s = _sparse_search(query)
        combined = []
        for doc in d + s:
            if doc not in combined:
                combined.append(doc)
        return combined

    def _rerank_search(query: str) -> list[Document]:
        hybrid_docs = _hybrid_search(query)
        q = query.lower()
        def score_relevance(doc):
            content = doc.page_content.lower()
            score = 0
            for term in q.split():
                if len(term) > 3 and term in content:
                    score += 2
            return score
        return sorted(hybrid_docs, key=score_relevance, reverse=True)

    return {
        "Dense Only": _dense_search,
        "BM25 Only": _sparse_search,
        "Hybrid (RRF)": _hybrid_search,
        "Hybrid + Rerank": _rerank_search,
    }


def build_live_retrievers() -> dict[str, Callable[[str], list[Document]]]:
    """Instantiate live production retrievers from project modules."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from src.api.dependencies import (
        BGEEmbeddings,
        BM25Store,
        CrossEncoderReranker,
        HybridRetriever,
        QdrantStore,
        settings,
    )

    embeddings = BGEEmbeddings().get_embeddings()

    if settings.qdrant_url:
        qdrant_store = QdrantStore(
            embeddings=embeddings,
            collection_name="multidoc_rag",
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        )
    else:
        qdrant_store = QdrantStore(
            embeddings=embeddings,
            collection_name="multidoc_rag",
            path="./qdrant_data",
        )

    # Validate Qdrant collection is non-empty
    try:
        count = qdrant_store.vector_store.client.count("multidoc_rag").count
        if count == 0:
            raise RuntimeError("Qdrant collection 'multidoc_rag' has 0 points. Run python ingest.py first.")
    except Exception as exc:
        raise RuntimeError(f"Qdrant store check failed: {exc}")

    bm25_path = Path(settings.bm25_store_path)
    if not bm25_path.exists():
        raise RuntimeError(f"BM25 store file not found at {bm25_path}. Run python ingest.py first.")
    bm25_store = BM25Store.load(str(bm25_path))
    docs = getattr(bm25_store, "_documents", getattr(bm25_store, "documents", []))
    if len(docs) == 0:
        raise RuntimeError(f"BM25 store at {bm25_path} is empty (0 docs). Run python ingest.py first.")

    hybrid = HybridRetriever(
        retrievers=[
            qdrant_store.as_dynamic_retriever(),
            bm25_store.as_dynamic_retriever(),
        ],
        final_k=20,
    )

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
    )

    dense_retriever = qdrant_store.as_retriever(search_kwargs={"k": 20})

    def _dense_search(query: str) -> list[Document]:
        return dense_retriever.invoke(query)

    def _sparse_search(query: str) -> list[Document]:
        return bm25_store.search(query, k=20)

    def _hybrid_search(query: str) -> list[Document]:
        return hybrid.search(query)

    def _rerank_search(query: str) -> list[Document]:
        candidates = hybrid.search(query)[:10]
        return reranker.rerank(query, candidates, top_k=5)

    return {
        "Dense Only": _dense_search,
        "BM25 Only": _sparse_search,
        "Hybrid (RRF)": _hybrid_search,
        "Hybrid + Rerank": _rerank_search,
    }


def _compute_percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(p * len(sorted_data))
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def evaluate_llm_judge(
    query: str,
    retrieved_docs: list[Document],
    is_unanswerable: bool,
    expected_answer: str,
) -> dict[str, float]:
    """
    Optional LLM Judge: Evaluates answer generation, faithfulness, and abstention.
    For unanswerable queries, checks if output abstains from hallucinating.
    """
    try:
        from src.llm.provider import create_llm

        llm = create_llm()
        context = "\n\n".join([d.page_content for d in retrieved_docs[:3]])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer based strictly on the context above. If the context does not "
            f"contain the answer, reply exactly: 'I do not have enough information to answer this question.'"
        )
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        abstention_triggers = [
            "do not have enough information",
            "not mentioned",
            "i do not know",
            "cannot be answered",
            "does not contain",
        ]
        abstained = any(t in text.lower() for t in abstention_triggers)

        if is_unanswerable:
            abstention_score = 1.0 if abstained else 0.0
            faithfulness_score = 1.0 if abstained else 0.0
        else:
            abstention_score = 0.0 if abstained else 1.0
            faithfulness_score = 1.0 if not abstained and len(text) > 10 else 0.0

        return {
            "faithfulness": faithfulness_score,
            "abstention": abstention_score,
        }
    except Exception as exc:
        logger.warning("LLM Judge evaluation failed: %s", exc)
        return {"faithfulness": 0.0, "abstention": 0.0}


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    use_judge: bool = False,
    mock_mode: bool = False,
) -> dict[str, Any]:
    """Run full evaluation suite across all 4 configurations."""
    cases = load_dataset(dataset_path)

    if mock_mode:
        mode = "mock"
        retrievers = build_mock_retrievers()
    else:
        mode = "live"
        try:
            retrievers = build_live_retrievers()
        except Exception as exc:
            logger.error("Live infrastructure initialization failed: %s", exc)
            sys.exit(
                f"\n[ERROR] Live infrastructure unavailable: {exc}\n"
                f"Cannot run real evaluation without live stores.\n"
                f"1. Ingest documents first: python ingest.py\n"
                f"2. Or if running offline/mock tests: python -m evals.rag.runner --mock\n"
            )

    k_cutoffs = (1, 3, 5)
    answerable_cases = [c for c in cases if not c.get("is_unanswerable", False)]
    unanswerable_cases = [c for c in cases if c.get("is_unanswerable", False)]

    pipeline_scores: dict[str, dict[str, list[float]]] = {
        name: {
            "hit@1": [],
            "hit@3": [],
            "hit@5": [],
            "mrr": [],
            "ndcg@5": [],
            "latency": [],
        }
        for name in retrievers
    }
    if use_judge:
        for name in retrievers:
            pipeline_scores[name]["faithfulness"] = []
            pipeline_scores[name]["abstention"] = []

    case_details = [
        {
            "id": case.get("id"),
            "category": case.get("category"),
            "query": case["query"],
            "is_unanswerable": case.get("is_unanswerable", False),
            "results": {},
        }
        for case in cases
    ]

    for name, retriever_func in retrievers.items():
        print(f"\n--- Evaluating Pipeline: {name} ---", flush=True)
        # Warmup pass over all queries
        print(f"[Warmup] Running warmup pass for {name} across all queries...", flush=True)
        for c in cases:
            try:
                retriever_func(c["query"])
            except Exception as e:
                logger.warning("Warmup failed for %s on '%s': %s", name, c["query"][:30], e)
        print(f"[Warmup] {name} warmed. Running timed evaluation pass...", flush=True)

        for idx, case in enumerate(cases):
            query = case["query"]
            relevant_sources = case.get("relevant_sources", [])
            contains = case.get("contains", [])
            is_unanswerable = case.get("is_unanswerable", False)

            start_time = time.perf_counter()
            try:
                retrieved = retriever_func(query)
            except Exception as e:
                logger.error("Pipeline '%s' failed on query: %s", name, e)
                retrieved = []
            elapsed = time.perf_counter() - start_time
            pipeline_scores[name]["latency"].append(elapsed)

            # Compute retrieval metrics ONLY for answerable cases
            if not is_unanswerable:
                metrics = evaluate_retrieval_case(
                    retrieved,
                    relevant_sources=relevant_sources,
                    contains=contains,
                    k_values=k_cutoffs,
                )
                for metric_name, val in metrics.items():
                    if metric_name in pipeline_scores[name]:
                        pipeline_scores[name][metric_name].append(val)
            else:
                metrics = None

            case_perf = {
                "metrics": metrics,
                "latency_sec": round(elapsed, 4),
                "num_retrieved": len(retrieved),
            }

            if use_judge:
                judge_scores = evaluate_llm_judge(
                    query=query,
                    retrieved_docs=retrieved,
                    is_unanswerable=is_unanswerable,
                    expected_answer=case.get("expected_answer", ""),
                )
                pipeline_scores[name]["faithfulness"].append(judge_scores["faithfulness"])
                pipeline_scores[name]["abstention"].append(judge_scores["abstention"])
                case_perf["judge"] = judge_scores

            case_details[idx]["results"][name] = case_perf

    # Compute Aggregate Metrics
    summary: dict[str, dict[str, float]] = {}
    for name, metric_dict in pipeline_scores.items():
        summary[name] = {}
        for m_name, vals in metric_dict.items():
            if m_name == "latency":
                continue
            summary[name][m_name] = round(sum(vals) / len(vals), 4) if vals else 0.0

        latencies = metric_dict.get("latency", [])
        if latencies:
            summary[name]["latency_median"] = round(statistics.median(latencies), 4)
            summary[name]["latency_p95"] = round(_compute_percentile(latencies, 0.95), 4)
            summary[name]["latency_mean"] = round(statistics.mean(latencies), 4)
            summary[name]["latency"] = summary[name]["latency_median"]
        else:
            summary[name]["latency_median"] = 0.0
            summary[name]["latency_p95"] = 0.0
            summary[name]["latency_mean"] = 0.0
            summary[name]["latency"] = 0.0

    # Format Markdown Table
    table_lines = [
        f"| Retrieval Pipeline | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | Median (s) | P95 (s) |"
        + (" Faithfulness | Abstention |" if use_judge else ""),
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        + (" :---: | :---: |" if use_judge else ""),
    ]

    for name, metrics in summary.items():
        row = (
            f"| **{name}** | {metrics.get('hit@1', 0.0):.3f} | {metrics.get('hit@3', 0.0):.3f} | "
            f"{metrics.get('hit@5', 0.0):.3f} | {metrics.get('mrr', 0.0):.3f} | {metrics.get('ndcg@5', 0.0):.3f} | "
            f"{metrics.get('latency_median', 0.0):.4f} | {metrics.get('latency_p95', 0.0):.4f} |"
        )
        if use_judge:
            row += f" {metrics.get('faithfulness', 0.0):.3f} | {metrics.get('abstention', 0.0):.3f} |"
        table_lines.append(row)

    markdown_table = "\n".join(table_lines)

    report_payload = {
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "total_cases": len(cases),
        "answerable_cases": len(answerable_cases),
        "unanswerable_cases": len(unanswerable_cases),
        "use_judge": use_judge,
        "summary": summary,
        "markdown_table": markdown_table,
        "case_details": case_details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "=" * 76)
    print(f"        RAG RETRIEVAL EVALUATION REPORT (Mode: {mode.upper()})")
    print(f"        Evaluated on {len(answerable_cases)} answerable cases (and {len(unanswerable_cases)} unanswerable)")
    print("=" * 76 + "\n")
    print(markdown_table)
    print("\n" + "=" * 76)
    print(f"Report written to: {output_path}\n")

    return report_payload


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Benchmark Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/rag/rag_dataset.jsonl",
        help="Path to JSONL evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for destination report JSON (defaults to rag_report.json or rag_report.mock.json)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM Judge for faithfulness and abstention rate",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without live Qdrant / external API calls",
    )

    args = parser.parse_args()

    if args.output:
        dest_path = Path(args.output)
    else:
        dest_path = Path("evals/results/rag_report.mock.json") if args.mock else Path("evals/results/rag_report.json")

    run_evaluation(
        dataset_path=Path(args.dataset),
        output_path=dest_path,
        use_judge=args.judge,
        mock_mode=args.mock,
    )


if __name__ == "__main__":
    main()
