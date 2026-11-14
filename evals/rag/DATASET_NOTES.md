# RAG Evaluation Dataset Notes

## Overview
The evaluation dataset in [`evals/rag/rag_dataset.jsonl`](rag_dataset.jsonl) comprises **162 test cases** (114 answerable, 48 unanswerable) designed to rigorously benchmark hybrid retrieval, dense vector search, sparse BM25, and cross-encoder reranking over enterprise travel policy documents.

Corpus files are cataloged in [`data/manifest.json`](../../data/manifest.json), comprising:
- **16 Authoritative Documents**: Active travel policies, expense caps, booking rules, executive protocols, insurance coverage, and safari guidelines.
- **8 Distractor Documents**: Topically similar historical policies (2020 COVID voucher policy, 2021 legacy lodging caps, 2022 flight advance directive), regional variants (APAC addendum, EMEA contractor addendum), budget excursion terms, and internal motor pool directives designed to challenge semantic and lexical retrieval discrimination.

---

## Split Methodology
The dataset is split into **dev** (50%) and **heldout** (50%) partitions using a deterministic stratified split per category.
- **Tuning on Dev Only**: Any parameter search or optimization (such as Reciprocal Rank Fusion dense/sparse weights or retrieval similarity thresholds) is conducted exclusively on the `dev` partition.
- **Reporting on Heldout**: Final reported benchmarks reflect the independent, unseen `heldout` partition, eliminating data snooping and overfitting risks.

---

## Category Specifications

| Category | Count (Dev / Heldout) | Description & Generation Methodology |
|:---|:---:|:---|
| `single_doc` | 18 (9 / 9) | Direct factual inquiries targeting explicit rules within one authoritative policy document. |
| `multi_doc` | 16 (8 / 8) | Inquiries spanning multiple distinct documents requiring joint retrieval (e.g., flight booking rules + emergency evacuation limits). |
| `paraphrased` | 16 (8 / 8) | Natural colloquial queries authored intentionally *without* reusing key technical terms or sentence structures from the source policy. |
| `exact_id` | 16 (8 / 8) | Specific identifier, form, and policy code lookups (e.g., PIR form, ICVP vaccine certificate, CDW/LDW waivers, `POL-APAC-2024-R1`, `SAF-OVR-2023-B`, `FLEET-DIR-2023-A`, pump #4). Every ID is verified to exist in the corpus. |
| `exact_figure` | 16 (8 / 8) | Queries focused on exact numerical values, per diem breakdowns ($15/$25/$35), hotel caps (320 SGD, 180 GBP), and timeline windows. |
| `negation_edge` | 16 (8 / 8) | Queries assessing negative constraints, prohibitions, and exclusions (e.g., prohibition of alcohol during solo meals, hard-shell suitcase bans, C-suite flight separation). |
| `near_duplicate_trap` | 16 (8 / 8) | Queries comparing current 2024 policies against superseded legacy policies (2021 expense cap vs 2024 cap) or adjacent tiers (budget vs luxury safari). Tests whether the retriever distinguishes the authoritative document from topically similar distractors. |
| `unanswerable` | 48 (24 / 24) | Out-of-scope inquiries (e.g., lunar shuttle codes, pet pythons, cryptocurrency reimbursements, personal acupuncture, scuba diving, pet relocation, IT password reset, corporate CPQ discounts) testing system abstention and hallucination prevention. |

---

## Methodological Caveats & Lexical Bias
- **Lexical Overlap Risk**: Questions generated directly from corpus text inherently risk lexical overlap with the source sentences, which artificially inflates sparse and hybrid retrieval scores.
- **Mitigation Strategy**: The dataset includes deliberately varied categories (`paraphrased`, `negation_edge`, `near_duplicate_trap`) where vocabulary diverges significantly from the corpus sentences.
- **Verbatim Ground Truth Verification**: Every answerable case's `contains` phrases are automatedly verified to exist verbatim in at least one cited source in `relevant_sources` via [`tests/test_dataset_integrity.py`](../../tests/test_dataset_integrity.py).
