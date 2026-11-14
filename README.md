# Enterprise Travel RAG & CRM Copilot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.8+-red.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade autonomous AI copilot for travel operations. Combines hybrid Retrieval-Augmented Generation (RAG) with transactional Salesforce CRM integrations, deterministic safety guards, and Human-in-the-Loop (HITL) execution controls.

---

## Architecture Overview

```mermaid
flowchart TD
    Client["Client / Web UI"] -->|"POST /chat\nPOST /chat/resume"| API["FastAPI (src/api/routes.py)"]
    API --> Graph["LangGraph Engine (src/graph/graph.py)"]

    subgraph "ReAct Loop"
        Graph --> Agent["Agent Node (ReAct LLM)"]
        Agent -->|"tool_calls"| Tools["ToolNode (src/tools/)"]
        Tools --> Guard{"__requires_approval__?"}
        Guard -->|No| Agent
        Guard -->|Yes| HITL["human_approval (interrupt)"]
        HITL -->|"POST /chat/resume"| Agent
    end

    subgraph "Retrieval Stack"
        Tools -->|"search_documents"| Hybrid["HybridRetriever (RRF)"]
        Hybrid --> Qdrant[("Qdrant\n(Dense Vectors)")]
        Hybrid --> BM25[("BM25Store\n(Sparse Keyword)")]
        Hybrid --> XEncoder["CrossEncoder\n(Reranker)"]
    end

    subgraph "Salesforce CRM"
        Tools -->|"CRM Operations"| SFClient["SalesforceAsyncClient"]
        SFClient --> Apex[("Apex REST Endpoints\n(JWT Bearer OAuth2)")]
    end

    Graph <-->|"Checkpointing"| PG[("PostgreSQL\n(Conversation Memory)")]
```

### Core Design Invariants
1. **Self-Correcting RAG**: Dense semantic search (Qdrant) and sparse keyword retrieval (BM25) fused via Reciprocal Rank Fusion (RRF), verified by a relevance grader with corrective query rewriting and cross-encoder reranking.
2. **Deterministic CRM Action**: Every mutating tool (`update_booking`, `update_travel_package`, `update_payment`, `update_salesforce_opportunity_status`) triggers a LangGraph `interrupt()`, awaiting user approval via `/chat/resume`.
3. **Hard-Capped Safety Guards**: Pre-execution gates prevent runaway writes, bulk updates, and invalid Salesforce IDs before network requests execute.
4. **Resilient State Persistence**: Multi-turn conversation state and authenticated user context (`sf_username`) persist across approvals via PostgreSQL checkpointing.

---

## Interaction Models

| Flow | Trigger | Execution Path | Safety Level |
|---|---|---|---|
| **Document Q&A** | Policy questions, FAQs, travel guidelines | `search_documents` → HybridRetriever → Grader/Rewriter → LLM Synthesis | Read-Only |
| **CRM Read** | Listing bookings, viewing payment status, opportunity queries | GET tools (`get_booking`, `get_payments`, etc.) → Apex REST GET | Read-Only |
| **CRM Write (HITL)** | Modifying bookings, updating packages, stage updates | GET tool (fetch ID) → UPDATE tool → `interrupt()` → User Review → DML execution | Write with Approval Gate |

---

## Safety & Security Architecture

### 1. Multi-Tiered Write Guards
Before any mutating Salesforce tool issues an approval request or API call, it passes through three hard coded guards in [`src/write_guards.py`](src/write_guards.py):
- **`single_record_guard`**: Enforces that exactly one valid Salesforce 15- or 18-character ID is targeted.
- **`bulk_intent_guard`**: Evaluates arguments using word-boundary regex (`\b(all|every|bulk|each|batch)\b`) to prevent mass modifications.
- **`session_write_cap`**: Hard limit on approved write operations per conversation session (default: 5).

### 2. Injection Defenses
- **SOQL Injection Defense**: [`src/clients/salesforce_client.py`](src/clients/salesforce_client.py) enforces a strict SObject allowlist (`Booking__c`, `Travel_Package__c`, `Payment__c`, etc.), strips ASCII control characters, bounds length to 255 chars, and escapes SOQL delimiters.
- **Prompt Injection Defense**: External CRM fields and document chunks are sanitized, bounded, and fenced inside structured tags (`<context>`, `<record>`).

### 3. API Authentication
Endpoints `/chat` and `/chat/resume` support standard HTTP Bearer token and `X-API-Key` headers configured via `API_SECRET_KEY`. When enabled, requests lacking valid credentials return `401 Unauthorized`. In local development without configured keys, requests run in permissive dev mode.

### 4. Record-Level Access Control (Salesforce Permission-Aware RAG)
When `SF_AUTH_ENABLED=true`, retrieval enforces Salesforce Account-level access control:
- **Chunk Metadata Tagging**: At ingestion via `--sf-manifest <PATH>` (JSON/CSV), every chunk is tagged with `sf_account_id` (normalized 18-char), `allowed_roles`, and `sf_scope` (`"account"`, `"org"`, or `"none"`). Untagged files default to `sf_scope="none"` and fail closed.
- **Short-Lived Signed JWT**: Identity is authenticated via RS256-signed JWTs (`iss`, `aud`, `exp`, `iat`, `jti`, `sf_user_id`, `sf_role`, `sf_org_id`, `tenant_id`). The system rejects `alg=none`, symmetric algorithms, expired tokens, tokens with lifetime > 120s, and replayed `jti`s.
- **Permission Filtering Strategy**:
  1. *Static Pre-Filter*: Qdrant and BM25 filter candidates by `tenant_id` AND (`sf_scope == "org"` OR `allowed_roles` contains caller's `sf_role`).
  2. *Over-Fetch & Access Resolution*: Candidates are over-fetched ($k \times 4$); distinct Account IDs are checked against Salesforce `UserRecordAccess` (`SELECT RecordId, HasReadAccess FROM UserRecordAccess`).
  3. *Widening*: If fewer than $k$ candidates survive, the search widens once to $k \times 8$ before final reranking.
  4. *Fail-Closed & Leak Prevention*: Inaccessible chunks are dropped before grading or rewriting; if zero chunks survive, the tool returns a neutral "No accessible documents found" without leaking the existence of restricted documents.
- **Caching**: Access resolution results are cached in-memory per `(sf_org_id, user_id, account_id)` with configurable TTL (default 60s via `SF_ACCESS_CACHE_TTL_SECONDS`) and bounded LRU eviction (10,000 entries).

---

## Known Limitations
- **Salesforce Sharing Complexity**: Record access is evaluated directly via Salesforce `UserRecordAccess` for explicit Account read grants. Deep implicit sharing hierarchies depend on what `UserRecordAccess` reports.
- **In-Memory JTI Replay Cache**: The JTI replay protection set is kept in process memory. Multi-worker or distributed deployments require an external distributed key-value store (such as Redis) with atomic `SETNX` and TTL.
- **60-Second Revocation Delay**: Account permissions are cached for `SF_ACCESS_CACHE_TTL_SECONDS` (default: 60s) to balance SOQL query limits with responsiveness. Permissions revoked in Salesforce take up to 60 seconds to reflect in retrieval.

---

## Repository Map

```
Advance_RAG_Chatbot/
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application, lifespan, middleware
│   │   ├── routes.py           # /chat, /chat/resume, /health endpoints
│   │   ├── dependencies.py     # DI container: LLM, stores, tools, compiled graph
│   │   └── schemas.py          # Pydantic v2 request/response schemas
│   ├── graph/
│   │   ├── graph.py            # LangGraph ReAct workflow, nodes, routing
│   │   ├── state.py            # RAGState TypedDict definition
│   │   ├── approval.py         # human_approval node & interrupt handling
│   │   └── memory.py           # PostgreSQL checkpointer factory
│   ├── tools/
│   │   ├── document_search.py  # search_documents tool (RAG pipeline)
│   │   ├── travel_tools.py     # Booking, Package, and Payment CRM tools
│   │   └── salesforce_tools.py # Opportunity search and update tools
│   ├── clients/
│   │   └── salesforce_client.py# Async HTTP client: JWT Bearer + Apex REST
│   ├── retrieval/
│   │   ├── hybrid_retriever.py # Reciprocal Rank Fusion (dense + sparse)
│   │   └── bm25_store.py       # BM25Okapi keyword store
│   ├── reranking/
│   │   └── reranker.py         # CrossEncoder (bge-reranker-v2-m3)
│   ├── rag/
│   │   ├── grading.py          # Document relevance grader
│   │   ├── rewriter.py         # Conversational query rewriter
│   │   └── chain.py            # Answer generation chain
│   ├── config.py               # Pydantic Settings & environment variables
│   └── write_guards.py         # Single-record, bulk intent, and session caps
├── tests/
│   ├── test_document_search.py # RAG tool pipeline tests
│   └── test_critical_fixes.py  # SOQL, auth, state, and write guard tests
├── docker-compose.yml          # Local PostgreSQL + Qdrant services
├── ingest.py                   # Document ingestion CLI
├── requirements.txt            # Python dependencies
├── AGENTS.md                   # Agent harness rules and operational guidelines
├── PROJECT_AUDIT_REPORT.md     # Security, architecture, and tech debt audit
└── PROGRESS.md                 # Real-time task tracking ledger
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for local PostgreSQL and Qdrant)
- Access to an LLM provider (Google Gemini, DeepSeek, or OpenAI)
- Salesforce Connected App with digital certificates (for CRM features)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ak0000007/Advance_RAG_MultiDoc_Chatbot.git Advance_RAG_Chatbot
cd Advance_RAG_Chatbot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Infrastructure Services

Launch PostgreSQL (conversation checkpointing) and Qdrant (vector database):

```bash
docker-compose up -d
```

### 3. Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

Configure your `.env` with required credentials:

```dotenv
# LLM Providers
GOOGLE_API_KEY="your-google-api-key"
# DEEPSEEK_API_KEY="your-deepseek-api-key"
# OPENAI_API_KEY="your-openai-api-key"

# Storage
QDRANT_URL="http://localhost:6333"
POSTGRES_URL="postgresql://admin:your_secure_password_here@localhost:5432/rag_db"

# API Security (Optional for local dev)
API_SECRET_KEY="your-secret-api-key"

# Salesforce Connected App (Optional for pure document RAG)
SF_CLIENT_ID="your_connected_app_client_id"
SF_LOGIN_URL="https://login.salesforce.com"
SF_PRIVATE_KEY_PATH="./secrets/server.key"
SF_DOMAIN="https://your-domain.my.salesforce.com"
```

### 4. Ingest Documents (Reproducible Corpus)

The repository includes a reproducible operational corpus of 16 corporate travel policies in [`data/txt/`](data/txt/). To build or refresh the vector store (Qdrant) and sparse index (BM25) with deterministic IDs and chunk metadata:

```bash
python ingest.py --data-dir data/txt --recreate
```

For tenant-isolated ingestion or Salesforce permission tagging:
```bash
python ingest.py --data-dir data/txt --tenant-id acme_corp --sf-manifest data/manifest.json --recreate
```

### 5. Start Application Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive web UI at `http://localhost:8000/`.

### 6. API Interaction Examples

#### Document Q&A (Read-Only)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the cancellation refund policy if canceled within 48 hours?", "session_id": "sess_01"}'
```

#### Mutating CRM Action (HITL Approval Loop)
```bash
# 1. Initiate booking update (triggers interrupt approval)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Update booking BK-201 to 4 travelers", "session_id": "sess_02"}'

# 2. Resume and approve the transaction
curl -X POST http://localhost:8000/chat/resume \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_02", "approved": true}'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

---

## Verification & Testing

### 1. Automated Test Suite

Run the full pytest suite:

```bash
pytest tests/ -v
```

Validates:
- Golden dataset accuracy (16/16 test cases, 100% accuracy) across Opportunities, Bookings, Packages, Payments, and Write Guards
- Mock Salesforce in-memory Apex REST transactions and state retention
- Grader prompt parsing with `.partial(format_instructions=...)`
- Disallowed SOQL SObject string rejection via `ValueError`
- Safe identifier escaping and length bounds
- Regex word-boundary protection in `bulk_intent_guard`
- API key authentication and unauthorized access rejection (`401`)
- State retention of `sf_username` across `/chat/resume`
- Multi-attempt retrieval and corrective rewriter routing
- Restricted CORS origins and credential handling

### 2. CRM & Tool Evaluation Harness

Run the CRM transaction accuracy benchmark:

```bash
python -m evals.runner
```

### 3. RAG Retrieval Benchmark (35 Evaluation Cases)

Run the retrieval pipeline evaluation comparing Dense (Qdrant), Sparse (BM25), Hybrid (RRF), and Cross-Encoder Reranking:

```bash
# Run against live Qdrant + BM25 indexes with warm-up latency calibration
python -m evals.rag.runner

# Or run in offline/mock mode
python -m evals.rag.runner --mock
```

#### Empirical Retrieval Benchmark Results (Live Corpus, 28 Answerable Cases)

| Retrieval Pipeline | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | Latency (s) | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense Only (BGE-M3)** | 0.821 | 1.000 | 1.000 | 0.911 | 0.913 | 3.64s | High semantic recall; includes remote/local embedding overhead |
| **BM25 Only (Okapi)** | 0.750 | 0.893 | 0.964 | 0.839 | 0.805 | 0.0007s | Ultra-low latency (~0.7ms); excels on exact IDs, figures & per diems |
| **Hybrid (RRF)** | **0.750** | **1.000** | **1.000** | **0.875** | **0.893** | **0.37s** | **Recommended Default**: 100% Top-3 Recall, low latency, robust fusion |
| **Hybrid + Rerank (Cross-Encoder)** | **0.750** | **1.000** | **1.000** | **0.875** | **0.901** | 17.17s | **GPU Opt-In**: +0.008 nDCG@5 gain; CPU inference cost (~17s) prohibitive for real-time |

> [!NOTE]
> **Reranking Caveat & CPU vs. GPU Deployment**:
> `Hybrid + Rerank` uses `BAAI/bge-reranker-v2-m3` (568M params). On CPU environments, scoring 10 candidate document pairs takes ~17s per query, yielding parity Hit@1 (0.750) and a marginal nDCG@5 increase (0.893 -> 0.901).
> Therefore, **`USE_RERANKER=false` is the default** for low-latency CPU production environments. On GPU/CUDA infrastructure or when maximum ranking precision is required, set `USE_RERANKER=true` in `.env`.

### 4. Reranker Diagnostic Inspection

To diagnose candidate pool depths, score direction, and rank shifts before and after cross-encoder reranking:

```bash
python scripts/diagnose_reranker.py
```

---

## Agent Guidelines & Standards

AI coding agents and autonomous contributors must follow [`AGENTS.md`](AGENTS.md):
- **Rule 1**: Never fabricate Salesforce IDs or values. Always call a GET tool before an UPDATE tool.
- **Rule 2**: Never bypass the LangGraph `interrupt()` lifecycle.
- **Rule 3**: Use async HTTP (`httpx.AsyncClient`) inside tools; never block the FastAPI event loop.
- **Rule 4**: Server-side auth context takes precedence over client-supplied `sf_username`.
- **Rule 5**: All mutating tools must pass write safety guards before issuing network calls.
- **Rule 6**: Always sanitize external strings before prompt interpolation.
- **Rule 7**: Wire dependencies exclusively in [`src/api/dependencies.py`](src/api/dependencies.py).
- **Rule 8**: Secrets remain in `.env` and are loaded via [`src/config.py`](src/config.py).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
