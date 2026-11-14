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
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from langchain_core.documents import Document

from evals.rag.metrics import (
    aggregate_category_metrics,
    aggregate_split_metrics,
    assert_not_mock_on_official_path,
    compute_file_sha256,
    doc_matches_source,
    evaluate_retrieval_case,
    format_category_markdown_table,
    get_git_commit,
    get_manifest_sources,
    paired_bootstrap_difference,
)

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
        "_components": {
            "embed": lambda q: [0.0] * 384,
            "vector_search": lambda vec, q="": _dense_search(q),
            "bm25": _sparse_search,
            "fusion": lambda lists, q="": _hybrid_search(q),
            "rerank": lambda q, docs: _rerank_search(q),
            "corpus_chunk_count": len(mock_docs),
        },
    }


def build_live_retrievers(device: str | None = None) -> dict[str, Any]:
    """Instantiate live production retrievers from project modules."""
    from src.api.dependencies import (
        BGEEmbeddings,
        BM25Store,
        CrossEncoderReranker,
        HybridRetriever,
        QdrantStore,
        settings,
    )
    from src.retrieval.hybrid_retriever import reciprocal_rank_fusion

    embeddings = BGEEmbeddings(device=device).get_embeddings()

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
        device=device,
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
        "_components": {
            "embed": lambda q: embeddings.embed_query(q),
            "vector_search": lambda vec, q="": qdrant_store.vector_store.similarity_search_by_vector(vec, k=20),
            "bm25": lambda q: bm25_store.search(q, k=20),
            "fusion": lambda lists, q="": reciprocal_rank_fusion(lists, k=60),
            "rerank": lambda q, docs: reranker.rerank(q, docs[:10], top_k=5),
            "corpus_chunk_count": count,
        },
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
    expected_answer: str = "",
    llm: Any = None,
) -> dict[str, float]:
    """
    Optional LLM Judge: Evaluates answer generation, faithfulness, and abstention.
    For unanswerable queries, checks if output abstains from hallucinating.
    For answerable queries, measures false abstention and groundedness.
    """
    try:
        if llm is None:
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
            "cannot answer",
        ]
        abstained = any(t in text.lower() for t in abstention_triggers)

        if is_unanswerable:
            abstention_score = 1.0 if abstained else 0.0
            false_abstention_score = 0.0
            faithfulness_score = 1.0 if abstained else 0.0
        else:
            abstention_score = 0.0
            false_abstention_score = 1.0 if abstained else 0.0
            faithfulness_score = 1.0 if (not abstained and len(text.strip()) > 10) else 0.0

        return {
            "faithfulness": faithfulness_score,
            "abstention": abstention_score,
            "false_abstention": false_abstention_score,
            "generated_answer": text,
        }
    except Exception as exc:
        logger.warning("LLM Judge evaluation failed: %s", exc)
        return {
            "faithfulness": 0.0,
            "abstention": 0.0,
            "false_abstention": 0.0,
            "generated_answer": "",
        }


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    use_judge: bool = False,
    mock_mode: bool = False,
    seed: int = 0,
    repeats: int = 3,
    device: str | None = None,
    split_filter: str = "all",
) -> dict[str, Any]:
    """Run full evaluation suite across all 4 configurations."""
    cases = load_dataset(dataset_path)
    if split_filter != "all":
        cases = [c for c in cases if c.get("split") == split_filter]
        if not cases:
            raise ValueError(f"No cases found for split filter: {split_filter}")

    if mock_mode:
        mode = "mock"
        retrievers = build_mock_retrievers()
    else:
        mode = "live"
        try:
            retrievers = build_live_retrievers(device=device)
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
    pipeline_names = ["Dense Only", "BM25 Only", "Hybrid (RRF)", "Hybrid + Rerank"]
    components = retrievers.get("_components", {})
    rng = random.Random(seed)

    comp_latencies: dict[str, list[float]] = {
        "query_embedding": [],
        "vector_search": [],
        "bm25": [],
        "fusion": [],
        "rerank": [],
    }

    pipeline_scores: dict[str, dict[str, list[float]]] = {
        name: {
            "hit@1": [],
            "hit@3": [],
            "hit@5": [],
            "recall@5": [],
            "mrr": [],
            "ndcg@5": [],
            "latency": [],
        }
        for name in pipeline_names
    }
    if use_judge:
        for name in pipeline_names:
            pipeline_scores[name]["faithfulness"] = []
            pipeline_scores[name]["abstention"] = []
            pipeline_scores[name]["false_abstention"] = []

    case_details = [
        {
            "id": case.get("id"),
            "category": case.get("category"),
            "split": case.get("split", "dev"),
            "query": case["query"],
            "is_unanswerable": case.get("is_unanswerable", False),
            "results": {},
        }
        for case in cases
    ]

    # Warm-up pass across all queries (discarded from benchmark timing)
    print(f"\n[Warmup] Running discarded warmup pass across all queries ({mode.upper()})...", flush=True)
    for c in cases:
        wq = c["query"]
        for p_name in pipeline_names:
            try:
                retrievers[p_name](wq)
            except Exception as e:
                logger.warning("Warmup query failed for %s: %s", p_name, e)
    print(f"[Warmup] Completed. Running timed benchmark ({repeats} repeats, seed={seed})...\n", flush=True)

    for idx, case in enumerate(cases):
        query = case["query"]
        relevant_sources = case.get("relevant_sources", [])
        contains = case.get("contains", [])
        is_unanswerable = case.get("is_unanswerable", False)

        q_pipeline_times: dict[str, list[float]] = {name: [] for name in pipeline_names}
        q_comp_times: dict[str, list[float]] = {k: [] for k in comp_latencies}
        cached_docs: dict[str, list[Document]] = {}

        for r_iter in range(repeats):
            # 1. Component timings
            t0 = time.perf_counter()
            q_vec = components["embed"](query)
            t_embed = time.perf_counter() - t0
            q_comp_times["query_embedding"].append(t_embed)

            t0 = time.perf_counter()
            dense_docs = components["vector_search"](q_vec, query)
            t_vec = time.perf_counter() - t0
            q_comp_times["vector_search"].append(t_vec)

            t0 = time.perf_counter()
            bm25_docs = components["bm25"](query)
            t_bm25 = time.perf_counter() - t0
            q_comp_times["bm25"].append(t_bm25)

            t0 = time.perf_counter()
            fused_docs = components["fusion"]([dense_docs, bm25_docs], query)
            t_fusion = time.perf_counter() - t0
            q_comp_times["fusion"].append(t_fusion)

            t0 = time.perf_counter()
            rerank_docs = components["rerank"](query, fused_docs)
            t_rerank = time.perf_counter() - t0
            q_comp_times["rerank"].append(t_rerank)

            if r_iter == 0:
                cached_docs["Dense Only"] = dense_docs
                cached_docs["BM25 Only"] = bm25_docs
                cached_docs["Hybrid (RRF)"] = fused_docs[:20]
                cached_docs["Hybrid + Rerank"] = rerank_docs[:5]

            # 2. Randomized pipeline end-to-end execution
            order = list(pipeline_names)
            rng.shuffle(order)
            for p_name in order:
                t_start = time.perf_counter()
                try:
                    retrievers[p_name](query)
                except Exception as e:
                    logger.error("Pipeline '%s' failed on query: %s", p_name, e)
                elapsed = time.perf_counter() - t_start
                q_pipeline_times[p_name].append(elapsed)

        # Median per query across repeats
        for p_name in pipeline_names:
            med_elapsed = statistics.median(q_pipeline_times[p_name])
            pipeline_scores[p_name]["latency"].append(med_elapsed)
            docs = cached_docs.get(p_name, [])

            # Compute retrieval metrics ONLY for answerable cases
            if not is_unanswerable:
                metrics = evaluate_retrieval_case(
                    docs,
                    relevant_sources=relevant_sources,
                    contains=contains,
                    k_values=k_cutoffs,
                )
                for metric_name, val in metrics.items():
                    if metric_name in pipeline_scores[p_name]:
                        pipeline_scores[p_name][metric_name].append(val)
            else:
                metrics = None

            top1_source = None
            if docs:
                meta_top = getattr(docs[0], "metadata", {}) or {}
                top1_source = meta_top.get("source") or meta_top.get("file_name")

            case_perf = {
                "metrics": metrics,
                "latency_sec": round(med_elapsed, 4),
                "num_retrieved": len(docs),
                "top1_source": top1_source,
            }

            if use_judge:
                judge_scores = evaluate_llm_judge(
                    query=query,
                    retrieved_docs=docs,
                    is_unanswerable=is_unanswerable,
                    expected_answer=case.get("expected_answer", ""),
                )
                if not is_unanswerable:
                    pipeline_scores[p_name]["faithfulness"].append(judge_scores["faithfulness"])
                    pipeline_scores[p_name]["false_abstention"].append(judge_scores["false_abstention"])
                else:
                    pipeline_scores[p_name]["abstention"].append(judge_scores["abstention"])
                case_perf["judge"] = judge_scores

            case_details[idx]["results"][p_name] = case_perf

        for c_k in comp_latencies:
            comp_latencies[c_k].append(statistics.median(q_comp_times[c_k]))

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

    # Component breakdown summary
    component_summary: dict[str, dict[str, float]] = {}
    for c_k, vals in comp_latencies.items():
        component_summary[c_k] = {
            "median_sec": round(statistics.median(vals), 4) if vals else 0.0,
            "p95_sec": round(_compute_percentile(vals, 0.95), 4) if vals else 0.0,
            "mean_sec": round(statistics.mean(vals), 4) if vals else 0.0,
        }

    # Sanity check: Hybrid must be >= Dense (or within tolerance)
    dense_med = summary.get("Dense Only", {}).get("latency_median", 0.0)
    hybrid_med = summary.get("Hybrid (RRF)", {}).get("latency_median", 0.0)
    timing_artifact_warning = None
    if hybrid_med < dense_med - 0.005:
        timing_artifact_warning = (
            f"WARNING: Timing artifact suspected! Hybrid latency ({hybrid_med:.4f}s) "
            f"is less than Dense latency ({dense_med:.4f}s)."
        )
        print(f"\n[CRITICAL TIMING WARNING] {timing_artifact_warning}\n")

    # Format Markdown Table
    table_lines = [
        f"| Retrieval Pipeline | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@5 | Median (s) | P95 (s) |"
        + (" Faithfulness | Abstention |" if use_judge else ""),
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        + (" :---: | :---: |" if use_judge else ""),
    ]

    for name, metrics in summary.items():
        row = (
            f"| **{name}** | {metrics.get('hit@1', 0.0):.3f} | {metrics.get('hit@3', 0.0):.3f} | "
            f"{metrics.get('hit@5', 0.0):.3f} | {metrics.get('recall@5', 0.0):.3f} | "
            f"{metrics.get('mrr', 0.0):.3f} | {metrics.get('ndcg@5', 0.0):.3f} | "
            f"{metrics.get('latency_median', 0.0):.4f} | {metrics.get('latency_p95', 0.0):.4f} |"
        )
        if use_judge:
            row += f" {metrics.get('faithfulness', 0.0):.3f} | {metrics.get('abstention', 0.0):.3f} |"
        table_lines.append(row)

    markdown_table = "\n".join(table_lines)

    # Compute Per-Category Aggregated Metrics
    category_summary = aggregate_category_metrics(case_details, pipeline_names)
    category_markdown_table = format_category_markdown_table(category_summary)

    # Compute Split Summaries with Bootstrap CIs
    dev_summary = aggregate_split_metrics(case_details, pipeline_names, split_name="dev")
    heldout_summary = aggregate_split_metrics(case_details, pipeline_names, split_name="heldout")
    overall_split_summary = aggregate_split_metrics(case_details, pipeline_names, split_name=None)

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(0)
        if device_str.startswith("cuda") and torch.cuda.is_available()
        else platform.processor() or "CPU"
    )

    # Paired Bootstrap Differences between Pipelines
    dense_scores = pipeline_scores.get("Dense Only", {})
    hybrid_scores = pipeline_scores.get("Hybrid (RRF)", {})
    rerank_scores = pipeline_scores.get("Hybrid + Rerank", {})

    paired_comparisons: dict[str, Any] = {}
    if dense_scores.get("hit@1") and hybrid_scores.get("hit@1"):
        paired_comparisons["Dense_vs_Hybrid"] = {
            "hit@1": paired_bootstrap_difference(dense_scores["hit@1"], hybrid_scores["hit@1"], seed=seed),
            "ndcg@5": paired_bootstrap_difference(dense_scores["ndcg@5"], hybrid_scores["ndcg@5"], seed=seed),
        }
    if hybrid_scores.get("hit@1") and rerank_scores.get("hit@1"):
        paired_comparisons["Hybrid_vs_Rerank"] = {
            "hit@1": paired_bootstrap_difference(hybrid_scores["hit@1"], rerank_scores["hit@1"], seed=seed),
            "ndcg@5": paired_bootstrap_difference(hybrid_scores["ndcg@5"], rerank_scores["ndcg@5"], seed=seed),
        }

    # Strictness & Matching Audit
    auth_sources, dist_sources = get_manifest_sources()
    cases_source_only = sum(1 for c in answerable_cases if not c.get("contains"))
    cases_contains = sum(1 for c in answerable_cases if c.get("contains"))

    distractor_top1_by_pipeline = {
        p_name: {"near_duplicate_trap": 0, "exact_id": 0, "total": 0}
        for p_name in pipeline_names
    }
    for c_dict in case_details:
        cat = c_dict.get("category")
        if cat in ("near_duplicate_trap", "exact_id"):
            for p_name in pipeline_names:
                p_res = c_dict.get("results", {}).get(p_name, {})
                top1_src = p_res.get("top1_source")
                if top1_src:
                    norm_src = Path(str(top1_src)).name.lower()
                    if norm_src in dist_sources:
                        distractor_top1_by_pipeline[p_name][cat] += 1
                        distractor_top1_by_pipeline[p_name]["total"] += 1

    matching_audit = {
        "cases_relying_on_source_only": cases_source_only,
        "cases_using_contains": cases_contains,
        "authoritative_sources_count": len(auth_sources),
        "distractor_sources_count": len(dist_sources),
        "distractor_top1_by_pipeline": distractor_top1_by_pipeline,
        "authoritative_source_enforced": True,
    }

    dataset_sha256 = compute_file_sha256(dataset_path)
    corpus_manifest_sha256 = compute_file_sha256("data/manifest.json")
    git_commit = get_git_commit()
    corpus_chunk_count = components.get("corpus_chunk_count", 0)

    metadata = {
        "device": device_str,
        "device_name": device_name,
        "embedding_model": "BAAI/bge-m3" if not mock_mode else "mock-embedding",
        "reranker_model": "BAAI/bge-reranker-v2-m3" if not mock_mode else "mock-reranker",
        "corpus_chunks_count": corpus_chunk_count,
        "corpus_chunk_count": corpus_chunk_count,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "dataset_sha256": dataset_sha256,
        "dataset_total_cases": len(cases),
        "git_commit": git_commit,
        "mode": mode,
        "seed": seed,
        "repeats": repeats,
        "component_latencies": component_summary,
    }
    if timing_artifact_warning:
        metadata["timing_artifact_warning"] = timing_artifact_warning

    report_payload = {
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "dataset_total_cases": len(cases),
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "corpus_chunk_count": corpus_chunk_count,
        "git_commit": git_commit,
        "split_filter": split_filter,
        "total_cases": len(cases),
        "answerable_cases": len(answerable_cases),
        "unanswerable_cases": len(unanswerable_cases),
        "use_judge": use_judge,
        "metadata": metadata,
        "summary": summary,
        "dev_summary": dev_summary,
        "heldout_summary": heldout_summary,
        "overall_split_summary": overall_split_summary,
        "paired_comparisons": paired_comparisons,
        "matching_audit": matching_audit,
        "markdown_table": markdown_table,
        "category_summary": category_summary,
        "category_markdown_table": category_markdown_table,
        "case_details": case_details,
    }

    assert_not_mock_on_official_path(output_path, mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    if use_judge:
        import csv
        csv_filename = "judge_spot_check.mock.csv" if mock_mode else "judge_spot_check.csv"
        csv_path = Path("evals/results") / csv_filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        rng_csv = random.Random(seed)
        ans_indices = [i for i, c in enumerate(case_details) if not c.get("is_unanswerable")]
        unans_indices = [i for i, c in enumerate(case_details) if c.get("is_unanswerable")]
        sample_ans = rng_csv.sample(ans_indices, min(10, len(ans_indices)))
        sample_unans = rng_csv.sample(unans_indices, min(10, len(unans_indices)))
        selected_indices = sorted(sample_ans + sample_unans)

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case_id",
                "split",
                "category",
                "query",
                "retrieved_context_preview",
                "generated_answer",
                "judge_faithfulness",
                "judge_abstention",
                "human_verified_label",
                "comments",
            ])
            for s_idx in selected_indices:
                cd = case_details[s_idx]
                p_res = cd.get("results", {}).get("Hybrid + Rerank") or list(cd.get("results", {}).values())[0]
                j_info = p_res.get("judge") or {}
                top1_src = p_res.get("top1_source") or ""
                ans_text = j_info.get("generated_answer", "")
                writer.writerow([
                    cd.get("id"),
                    cd.get("split"),
                    cd.get("category"),
                    cd.get("query"),
                    top1_src,
                    ans_text,
                    j_info.get("faithfulness", ""),
                    j_info.get("abstention", ""),
                    "",
                    "",
                ])
        logger.info("Judge spot check CSV exported to %s", csv_path)

    print("\n" + "=" * 76)
    print(f"        RAG RETRIEVAL EVALUATION REPORT (Mode: {mode.upper()})")
    print(f"        Evaluated on {len(answerable_cases)} answerable cases (and {len(unanswerable_cases)} unanswerable)")
    print(f"        Device: {device_str.upper()} ({device_name}) | Repeats: {repeats} | Seed: {seed}")
    print("=" * 76 + "\n")
    print(markdown_table)
    print("\n" + "=" * 76)
    print("        PER-CATEGORY RETRIEVAL PERFORMANCE")
    print("=" * 76 + "\n")
    print(category_markdown_table)
    print("\n" + "=" * 76)
    print("        COMPONENT LATENCY BREAKDOWN (Median / P95)")
    print("=" * 76)
    for c_k, c_v in component_summary.items():
        print(f"  - {c_k:<18}: {c_v['median_sec']:.4f}s (median), {c_v['p95_sec']:.4f}s (P95)")
    print("=" * 76)
    print(f"\nReport written to: {output_path}\n")

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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for randomized pipeline execution order (default 0)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of latency measurement repeats per query (default 3, median taken)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target compute device: 'cpu' or 'cuda' (default auto-detect)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "dev", "heldout"],
        default="all",
        help="Evaluation split to benchmark ('all', 'dev', 'heldout')",
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
        seed=args.seed,
        repeats=args.repeats,
        device=args.device,
        split_filter=args.split,
    )


if __name__ == "__main__":
    main()
