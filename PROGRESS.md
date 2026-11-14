# Project Progress Tracker

> **Protocol**: Autonomous agents update this file at start and end of multi-step sessions.
> Keep ultra-lean (< 50 lines). Only record actionable state, verified items, and blockers.

---

## Active Goal
Execute Tier 1 (P1) scale and enterprise architecture backlog from [`AGENTS.md`](AGENTS.md).

---

## Done (Verified)
- [x] **Core ReAct + HITL Architecture**: LangGraph workflow with tool router and interrupt approval in [`src/graph/graph.py`](src/graph/graph.py).
- [x] **C1-C4 Critical Vulnerabilities**: SOQL injection allowlist, bearer auth middleware, resume sf_username state retention, and grading format instructions fixed and verified.
- [x] **H1-H15 Audit Remediations**: Pooling in [`src/clients/salesforce_client.py`](src/clients/salesforce_client.py), ingest sync & deduplication in [`ingest.py`](ingest.py), parallel retrieval in [`src/retrieval/hybrid_retriever.py`](src/retrieval/hybrid_retriever.py), CORS in [`src/api/app.py`](src/api/app.py), audit tables in [`src/audit/log.py`](src/audit/log.py), and thread-safe CrossEncoder init in [`src/reranking/reranker.py`](src/reranking/reranker.py).
- [x] **Multi-Agent Specialist Architecture**: 3 specialist subgraphs with intent classification orchestrator in [`src/graph/orchestrator.py`](src/graph/orchestrator.py); 6/6 tests passing.
- [x] **Token Streaming & Citations**: SSE streaming (`/chat/stream`, `/chat/resume/stream`), citation extraction, and UI chips in [`src/ui/static/app.js`](src/ui/static/app.js); verified across 13 tests.
- [x] **P1: Observability Suite (LangSmith, Langfuse, OpenTelemetry & In-Process Telemetry)**: Configured `SecretStr` settings in [`src/config.py`](src/config.py); built `ObservabilityCallbackHandler` recording node and tool latency histograms in [`src/telemetry_metrics.py`](src/telemetry_metrics.py); added multi-tracer fallback in [`src/telemetry.py`](src/telemetry.py); propagated `trace_id` through routes, schemas, and UI; verified 11/11 in [`tests/test_telemetry.py`](tests/test_telemetry.py).
- [x] **RAG Retrieval Evaluation Harness & 35-Case Benchmark**: Built [`evals/rag/metrics.py`](evals/rag/metrics.py) (Hit@k, Recall@k, MRR, nDCG@k without chunk ID churn), benchmark runner in [`evals/rag/runner.py`](evals/rag/runner.py) (Dense vs BM25 vs Hybrid vs Rerank), 35-question dataset in [`evals/rag/rag_dataset.jsonl`](evals/rag/rag_dataset.jsonl), report in [`evals/results/rag_report.json`](evals/results/rag_report.json); verified in [`tests/test_rag_metrics.py`](tests/test_rag_metrics.py).
- [x] **Multi-Tenant Document Isolation & Ingestion Stamping**: Implemented `--tenant-id` CLI flag in [`ingest.py`](ingest.py); injected `tenant_id` into `RAGState` and `RunnableConfig` in [`src/api/routes.py`](src/api/routes.py); verified User A cannot retrieve User B's chunks in [`tests/test_tenant_isolation_and_security.py`](tests/test_tenant_isolation_and_security.py).
- [x] **Constant-Time Authentication & API Key Mapping**: Implemented `secrets.compare_digest` and `api_key_mapping` user/tenant resolver in [`src/api/routes.py`](src/api/routes.py); verified constant-time auth in [`tests/test_tenant_isolation_and_security.py`](tests/test_tenant_isolation_and_security.py).
- [x] **Salesforce Record-Level Access Control for RAG Retrieval**: Implemented permission-aware retrieval: tagged chunks via `--sf-manifest`, verified RS256 JWTs with JTI replay prevention, added `AccessResolver` querying `UserRecordAccess` with TTL caching, overfetch/widen filtering in `document_search.py`; verified 10/10 tests in [`tests/test_sf_record_access.py`](tests/test_sf_record_access.py) (225 passed total).
- [x] **Fail-Safe Relevance Grader**: Swapped silent exception swallowing for loud error logging and tagged context with `[UNVERIFIED RETRIEVAL]` flag in [`src/tools/document_search.py`](src/tools/document_search.py); verified in unit tests.
- [x] **Repo Tidiness, Demoability & Mock CRM**: Created MIT [`LICENSE`](LICENSE); moved legacy root test scripts to `scripts/legacy_dev/`; cleaned [`requirements.txt`](requirements.txt) with [`requirements.lock`](requirements.lock); added [`Dockerfile`](Dockerfile) and `app` service to [`docker-compose.yml`](docker-compose.yml); added sample docs in `data/sample_docs/`; wired `MockSalesforceAsyncClient` in [`src/api/dependencies.py`](src/api/dependencies.py); generated 11-section documentation in [`Explaination/rag-eval-harness-documentation.html`](Explaination/rag-eval-harness-documentation.html).

---

## In Progress / Pending (P1 Scale & Enterprise Backlog)
- [ ] **P1: Multi-Model Routing**: Lightweight model for grading & intent classification, frontier for synthesis.

---

## Next Immediate Action
1. Implement P1 item: Multi-Model Routing (lightweight LLM for intent routing and grading, frontier LLM for final synthesis in [`src/llm/provider.py`](src/llm/provider.py) and [`src/graph/orchestrator.py`](src/graph/orchestrator.py)).
