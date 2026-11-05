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
- [x] **Agent Routing Evaluation Harness**: Implemented [`evals/routing/`](file:///home/akhilsaini/Advance_RAG_Chatbot/evals/routing/) (dataset, judge, runner) with 24 natural-language cases against live LLM; achieved **95.83% empirical accuracy** (23/24 passing), with `@pytest.mark.routing` verified passing in [`tests/test_routing_eval.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_routing_eval.py).
- [x] **Enterprise Write Audit Logging & Telemetry (H10 Fixed)**: Built [`src/audit/log.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/audit/log.py), wired proposal/decision/outcome audits into [`src/graph/approval.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/graph/approval.py), shared pool, exposed `/audit/{thread_id}` & `/metrics`; verified 7/7 tests in [`tests/test_audit_log.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_audit_log.py).
- [x] **H2 + H3: Ingestion Sync & Deduplication**: Added deterministic UUIDv5 ID generation in [`ingest.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/ingest.py); added pickle persistence & dedup to [`src/retrieval/bm25_store.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/retrieval/bm25_store.py); verified 4/4 tests in [`tests/test_ingest_sync.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_ingest_sync.py).
- [x] **Multi-Agent Specialist Architecture**: Decomposed monolithic agent into 3 specialist subgraphs (`crm_agent`, `travel_agent`, `docs_agent`) with intent classification orchestrator in [`src/graph/orchestrator.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/graph/orchestrator.py), prompts in [`src/graph/prompts.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/graph/prompts.py), feature toggle `use_multi_agent_architecture` in [`src/config.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/config.py); verified 6/6 unit tests in [`tests/test_orchestrator.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_orchestrator.py); benchmarked 95.83% accuracy in both modes.

- [x] **H4: Parallel Hybrid Retrieval**: Implemented async fan-out/fan-in via `asyncio.gather` & thread pool in [`src/retrieval/hybrid_retriever.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/retrieval/hybrid_retriever.py) & [`src/reranking/reranker.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/reranking/reranker.py); verified 4/4 tests in [`tests/test_parallel_retrieval.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_parallel_retrieval.py).
- [x] **H9: CORS Middleware**: Added configurable `cors_origins` in [`src/config.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/config.py) and `CORSMiddleware` in [`src/api/app.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/api/app.py); verified 3/3 tests in [`tests/test_cors.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_cors.py).
- [x] **P1: Token Streaming (SSE)**: Added `/chat/stream` & `/chat/resume/stream` via `graph.astream_events(version="v2")` in [`src/api/routes.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/api/routes.py); wired live streaming & tool status into [`src/ui/static/app.js`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/ui/static/app.js); verified 5/5 unit tests in [`tests/test_streaming.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_streaming.py).
- [x] **H1: Salesforce Client Connection Pooling**: Persistent `httpx.AsyncClient` with keep-alive limits, auto-recovery, ownership tracking, and lifespan teardown in [`src/clients/salesforce_client.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/clients/salesforce_client.py) and [`src/api/app.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/api/app.py); verified 7/7 tests in [`tests/test_salesforce_client_pooling.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_salesforce_client_pooling.py).
- [x] **H14: Thread-Safe CrossEncoder Lazy Init**: Double-checked locking with `threading.Lock` and `asyncio.Lock` in [`src/reranking/reranker.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/reranking/reranker.py); verified 4/4 concurrency tests in [`tests/test_reranker_thread_safety.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/tests/test_reranker_thread_safety.py).

---

## In Progress / Pending (P1 Scale & Enterprise Backlog)
- [ ] **P1: SecretStr for API Keys & Passwords**: Prevent key leaks in tracebacks/repr (debt H8).
- [ ] **P1: Path Traversal Sandboxing**: Secure `load_file()` directory jail in [`src/ingestion/loader.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/ingestion/loader.py) (debt H13).

---

## Next Immediate Action
1. Implement P1 item: Wrap sensitive credentials with `SecretStr` in [`src/config.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/config.py) (H8).
