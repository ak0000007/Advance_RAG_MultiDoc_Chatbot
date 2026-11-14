# RAG Evaluation Dataset Changelog

## 2026-11-14: Task 1 Diagnostic Fixes (Shared Failures rag_009, rag_010, rag_023, rag_028)

| Case ID | Cause | Evidence | Applied Fix |
| :--- | :--- | :--- | :--- |
| **`rag_009`** | **(b) contains phrase too strict & (c) relevant_sources incomplete** | Retrieved Rank 1 is `business_meals_and_entertainment_rules.txt`, which contains the authoritative per diem rule: "Domestic business travel meal per diem allowance is strictly capped at $75 per calendar day ($15 breakfast, $25 lunch, $35 dinner)". The existing contains check strictly looked for "$75 per day" rather than "$75 per calendar day". | Added `"business_meals_and_entertainment_rules.txt"` to `relevant_sources` and added `"$75 per calendar day"` to `contains`. |
| **`rag_010`** | **(c) relevant_sources list incomplete** | Retrieved Rank 1 is `business_meals_and_entertainment_rules.txt`, which contains: "International business travel meal per diem allowance is capped at $100 per day ($20 breakfast, $35 lunch, $45 dinner)". Contains matched "$100 per day", but source check failed. | Added `"business_meals_and_entertainment_rules.txt"` to `relevant_sources` alongside `expense_and_lodging_guidelines.txt`. |
| **`rag_023`** | **(c) relevant_sources list incomplete** | Retrieved Rank 1 is `african_safari_bush_flight_and_luggage.txt`, which is the primary dedicated policy for bush flights; Section 1 explicitly bans hard-shell suitcases and mandates soft-sided duffel bags. | Added `"african_safari_bush_flight_and_luggage.txt"` to `relevant_sources` alongside `luxury_safari_terms.txt`. |
| **`rag_028`** | **(c) relevant_sources list incomplete** | Retrieved Rank 1 is `baggage_delay_loss_and_interruption.txt`, which is the primary dedicated policy for baggage delay and loss; Section 2 details the $1,500 maximum lost luggage reimbursement upon PIR submission. | Added `"baggage_delay_loss_and_interruption.txt"` to `relevant_sources` alongside `travel_insurance_coverage.txt`. |

---

## 2026-11-14: Audit Verification, Distractor Filtering, and Integrity Guards

1. **Distractor Source Rejection in `metrics.py`**:
   - Chunks originating from distractor files listed in `data/manifest.json` (`distractor_sources`) are strictly barred from counting as retrieval hits, even if their content contains verbatim matching terms.
   - Enforces that retrieval benchmarks reflect strictly authoritative policy grounding.

2. **Dataset Expansion (`evals/rag/rag_dataset.jsonl`)**:
   - Expanded dataset to 162 total cases (114 answerable, 48 unanswerable) with 50/50 stratified Dev/Heldout balance.
   - Added 24 new unanswerable cases (`rag_139`–`rag_162`) spanning travel policy near-misses (pet travel, scuba certification, airline lounge clubs, EU261 claims) and out-of-scope enterprise queries (Active Directory password reset, Salesforce CPQ discounts, Workday payroll, Okta MFA). Verified via corpus grep that zero answerable sentences exist in `data/txt/`.

3. **Freshness & Integrity Guards**:
   - Embedded SHA-256 hashes (`dataset_sha256`, `corpus_manifest_sha256`), git commit, and mode checks across runner, tuning, and calibration reports.
   - Refuse mock output writes to official paths (`rag_report.json`, `rrf_tuning_report.json`, `threshold_calibration_report.json`).
   - Updated `scripts/check_readme_sync.py` to assert SHA-256 match, case count match, and live report mode.

4. **Statistical Rigor & Metric Upgrades**:
   - Replaced RRF rank-sum thresholding with semantic similarity signals (`dense_cosine`, `reranker`).
   - Implemented constrained DEV sweep (maximizing true abstention subject to false abstention $\le 0.05$) and Wilson 95% score intervals on Heldout.
   - Added paired bootstrap differences and statistical significance testing across pipeline pairs.

