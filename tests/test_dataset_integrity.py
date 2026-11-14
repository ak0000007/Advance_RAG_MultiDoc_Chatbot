"""Unit and integrity tests for the RAG evaluation dataset.
Verifies unique IDs, schema validity, corpus source existence,
verbatim phrase matches, unanswerable criteria, and split balance.
"""
import json
from pathlib import Path
import pytest

DATASET_PATH = Path("evals/rag/rag_dataset.jsonl")
DATA_DIR = Path("data/txt")

VALID_CATEGORIES = {
    "single_doc",
    "multi_doc",
    "paraphrased",
    "exact_id",
    "exact_figure",
    "negation_edge",
    "near_duplicate_trap",
    "unanswerable",
}

VALID_SPLITS = {"dev", "heldout"}


def load_dataset():
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"
    cases = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def test_dataset_exists_and_non_empty():
    cases = load_dataset()
    assert len(cases) >= 120, f"Expected at least 120 cases, got {len(cases)}"


def test_unique_case_ids():
    cases = load_dataset()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found in dataset: {len(ids)} total, {len(set(ids))} unique"


def test_required_fields():
    cases = load_dataset()
    required = {"id", "category", "split", "query", "relevant_sources", "contains"}
    for c in cases:
        missing = required - set(c.keys())
        assert not missing, f"Case {c.get('id')} missing fields: {missing}"


def test_category_and_split_validity():
    cases = load_dataset()
    for c in cases:
        assert c["category"] in VALID_CATEGORIES, f"Invalid category {c['category']} in {c['id']}"
        assert c["split"] in VALID_SPLITS, f"Invalid split {c['split']} in {c['id']}"


def test_stratified_split_balance():
    from collections import defaultdict
    cases = load_dataset()
    counts = defaultdict(lambda: {"dev": 0, "heldout": 0})
    for c in cases:
        counts[c["category"]][c["split"]] += 1

    for cat, splits in counts.items():
        total = splits["dev"] + splits["heldout"]
        assert total > 0, f"No cases for category {cat}"
        # Dev and heldout should each be roughly 50% (+/- 1 case difference)
        diff = abs(splits["dev"] - splits["heldout"])
        assert diff <= 1, f"Category {cat} unbalanced: dev={splits['dev']}, heldout={splits['heldout']}"


def test_corpus_sources_exist():
    cases = load_dataset()
    assert DATA_DIR.exists()
    for c in cases:
        for src in c["relevant_sources"]:
            fpath = DATA_DIR / src
            assert fpath.is_file(), f"Case {c['id']} cites non-existent file: {src}"


def test_contains_phrases_verbatim_in_sources():
    cases = load_dataset()
    for c in cases:
        if c["category"] == "unanswerable" or c.get("is_unanswerable"):
            continue
        assert len(c["contains"]) >= 1, f"Answerable case {c['id']} has empty contains list"
        for phrase in c["contains"]:
            matched = False
            for src in c["relevant_sources"]:
                fpath = DATA_DIR / src
                if phrase in fpath.read_text(encoding="utf-8"):
                    matched = True
                    break
            assert matched, f"Phrase '{phrase}' for case {c['id']} not found in cited sources {c['relevant_sources']}"


def test_unanswerable_invariants():
    cases = load_dataset()
    unanswerables = [c for c in cases if c["category"] == "unanswerable"]
    assert len(unanswerables) >= 20, f"Expected >= 20 unanswerable cases, got {len(unanswerables)}"
    for c in unanswerables:
        assert c["relevant_sources"] == [], f"Unanswerable {c['id']} must have empty relevant_sources"
        assert c["contains"] == [], f"Unanswerable {c['id']} must have empty contains"


def test_answerable_counts():
    cases = load_dataset()
    answerables = [c for c in cases if c["category"] != "unanswerable"]
    assert len(answerables) >= 100, f"Expected >= 100 answerable cases, got {len(answerables)}"
