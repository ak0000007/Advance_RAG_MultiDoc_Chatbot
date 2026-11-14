"""
Diagnostic script for failure analysis on specific dataset queries.
Examines top retrieved chunks, source matching, phrase matching, and rank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.rag.metrics import doc_matches_source, doc_matches_contains, is_chunk_relevant
from evals.rag.runner import build_live_retrievers, build_mock_retrievers


def diagnose_failures(case_ids: list[str], dataset_path: Path, mock: bool = False):
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    selected_cases = [c for c in cases if c.get("id") in case_ids]
    if not selected_cases:
        print(f"No cases found matching IDs: {case_ids}")
        return

    try:
        if mock:
            retrievers = build_mock_retrievers()
        else:
            retrievers = build_live_retrievers()
    except Exception as exc:
        print(f"Live retrievers unavailable ({exc}), falling back to mock mode.")
        retrievers = build_mock_retrievers()

    dense = retrievers["Dense Only"]

    for case in selected_cases:
        cid = case.get("id")
        query = case.get("query")
        gold_sources = case.get("relevant_sources", [])
        gold_contains = case.get("contains", [])

        print("=" * 80)
        print(f"CASE ID: {cid}")
        print(f"QUERY: {query}")
        print(f"GOLD SOURCES: {gold_sources}")
        print(f"GOLD CONTAINS: {gold_contains}")
        print("-" * 80)

        docs = dense(query)
        top5 = docs[:5]

        gold_rank = None
        for r, doc in enumerate(docs, start=1):
            if is_chunk_relevant(doc, gold_sources, gold_contains):
                gold_rank = r
                break

        print(f"GOLD CRITERIA FIRST SATISFIED AT RANK: {gold_rank if gold_rank else '> ' + str(len(docs))}")
        print("\nTOP-5 RETRIEVED CHUNKS:")
        for idx, doc in enumerate(top5, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
            norm_source = Path(source).name
            src_match = any(doc_matches_source(doc, s) for s in gold_sources)
            contains_matches = {
                phrase: (phrase.lower() in doc.page_content.lower())
                for phrase in gold_contains
            }
            overall_relevant = is_chunk_relevant(doc, gold_sources, gold_contains)

            print(f"  [Rank {idx}] Source: {norm_source}")
            print(f"    Source Match: {src_match}")
            print(f"    Contains Matches: {contains_matches} (Any match: {any(contains_matches.values()) if contains_matches else True})")
            print(f"    Is Gold Relevant: {overall_relevant}")
            preview = doc.page_content.replace("\n", " ")[:200]
            print(f"    Content Preview: {preview}...")
            print()
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose retrieval failures.")
    parser.add_argument("--cases", nargs="+", default=["rag_009", "rag_010", "rag_023", "rag_028"], help="Case IDs to diagnose")
    parser.add_argument("--dataset", default="evals/rag/rag_dataset.jsonl", help="Dataset path")
    parser.add_argument("--mock", action="store_true", help="Use mock retrievers")
    args = parser.parse_args()

    diagnose_failures(args.cases, Path(args.dataset), mock=args.mock)
