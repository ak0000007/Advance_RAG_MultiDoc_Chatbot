# Project Progress Tracker

> **Protocol**: Autonomous agents update this file at start and end of multi-step sessions.
> Keep ultra-lean (< 50 lines). Only record actionable state, verified items, and blockers.

---

## Active Goal
Remediate Tier 1 (P0) security vulnerabilities and critical pipeline crashes from [`PROJECT_AUDIT_REPORT.md`](file:///home/akhilsaini/Advance_RAG_Chatbot/PROJECT_AUDIT_REPORT.md).

---

## Done (Verified)
- [x] **Agent Harness**: Added [`AGENTS.md`](file:///home/akhilsaini/Advance_RAG_Chatbot/AGENTS.md) with 8 behavioral rules, architecture map, roadmap, and design guidelines.
- [x] **Verification Suite**: Added Section 6 to [`AGENTS.md`](file:///home/akhilsaini/Advance_RAG_Chatbot/AGENTS.md#section-6--verification-commands-harness-automation) with exact CLI verification commands.
- [x] **Test Suite Baseline**: Verified [`tests/test_document_search.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_document_search.py) passes 5/5 (`PYTHONPATH=. pytest tests/ -v`).
- [x] **Core ReAct + HITL Architecture**: LangGraph workflow with tool router and interrupt-driven human approval wired in [`src/graph/graph.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/graph/graph.py).
- [x] **Write Safety Guards**: Session write cap, bulk intent detection, and single-record lock implemented in [`src/write_guards.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/write_guards.py).
- [x] **C1: SOQL Injection Fixed**: Validated `sobject` allowlist and escaped query string in [`src/clients/salesforce_client.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/clients/salesforce_client.py#L150-L163).
- [x] **C2: API Authentication**: Added bearer/API-key auth dependency in [`src/api/routes.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/api/routes.py) with dev fallback and user context extraction.
- [x] **C3: Resume State Retention**: Persisted `sf_username` in [`src/graph/state.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/graph/state.py) and restored on `/chat/resume`.
- [x] **C4: Grading Prompt Crash Fixed**: Added `.partial(format_instructions=...)` in [`src/rag/grading.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/rag/grading.py).
- [x] **H7: Bulk Guard Regex Fixed**: Word-boundary regex `\b(token)\b` applied in [`src/write_guards.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/write_guards.py).
- [x] **Evaluation Benchmark Harness**: Built [`evals/mock_salesforce.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/evals/mock_salesforce.py), 16-case golden dataset in [`evals/golden_dataset.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/evals/golden_dataset.py), and runner [`evals/runner.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/evals/runner.py) achieving 100.0% accuracy across Opportunities, Bookings, Packages, Payments, and Safety Guards.
- [x] **CI Pipeline Restored**: Fixed missing `pythonpath` in [`pyproject.toml`](file:///home/akhilsaini/Advance_RAG_Chatbot/pyproject.toml) and workflow runner invocation in [`.github/workflows/ci.yml`](file:///home/akhilsaini/Advance_RAG_Chatbot/.github/workflows/ci.yml); GitHub Actions run `37436740796` all green (11/11 tests pass).
- [x] **Agent Routing Evaluation Harness**: Implemented [`evals/routing/`](file:///home/akhilsaini/Advance_RAG_Chatbot/evals/routing/) (dataset, judge, runner) with 24 natural-language cases against live LLM; achieved **75.0% empirical accuracy**, with `@pytest.mark.routing` verified passing in [`tests/test_routing_eval.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_routing_eval.py).

---

## In Progress / Pending (P0 Backlog)
- [ ] **H2 + H3: Ingestion Sync**: Dual-index Qdrant + [`BM25Store`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/retrieval/bm25_store.py) with deterministic content-hash IDs in [`ingest.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/ingest.py).
- [ ] **H10: Postgres Pool Lifecycle**: Add `await pool.open()` in FastAPI lifespan / [`src/graph/memory.py:26`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/graph/memory.py#L26-L31).

---

## Next Immediate Action
1. Populate BM25 store during `ingest.py` (H2/H3).
2. Wire `await pool.open()` and graceful shutdown for Postgres pool in FastAPI lifespan (H10).
