# RAG Evaluation Dataset Changelog

## 2026-11-14: Task 1 Diagnostic Fixes (Shared Failures rag_009, rag_010, rag_023, rag_028)

| Case ID | Cause | Evidence | Applied Fix |
| :--- | :--- | :--- | :--- |
| **`rag_009`** | **(b) contains phrase too strict & (c) relevant_sources incomplete** | Retrieved Rank 1 is `business_meals_and_entertainment_rules.txt`, which contains the authoritative per diem rule: "Domestic business travel meal per diem allowance is strictly capped at $75 per calendar day ($15 breakfast, $25 lunch, $35 dinner)". The existing contains check strictly looked for "$75 per day" rather than "$75 per calendar day". | Added `"business_meals_and_entertainment_rules.txt"` to `relevant_sources` and added `"$75 per calendar day"` to `contains`. |
| **`rag_010`** | **(c) relevant_sources list incomplete** | Retrieved Rank 1 is `business_meals_and_entertainment_rules.txt`, which contains: "International business travel meal per diem allowance is capped at $100 per day ($20 breakfast, $35 lunch, $45 dinner)". Contains matched "$100 per day", but source check failed. | Added `"business_meals_and_entertainment_rules.txt"` to `relevant_sources` alongside `expense_and_lodging_guidelines.txt`. |
| **`rag_023`** | **(c) relevant_sources list incomplete** | Retrieved Rank 1 is `african_safari_bush_flight_and_luggage.txt`, which is the primary dedicated policy for bush flights; Section 1 explicitly bans hard-shell suitcases and mandates soft-sided duffel bags. | Added `"african_safari_bush_flight_and_luggage.txt"` to `relevant_sources` alongside `luxury_safari_terms.txt`. |
| **`rag_028`** | **(c) relevant_sources list incomplete** | Retrieved Rank 1 is `baggage_delay_loss_and_interruption.txt`, which is the primary dedicated policy for baggage delay and loss; Section 2 details the $1,500 maximum lost luggage reimbursement upon PIR submission. | Added `"baggage_delay_loss_and_interruption.txt"` to `relevant_sources` alongside `travel_insurance_coverage.txt`. |
