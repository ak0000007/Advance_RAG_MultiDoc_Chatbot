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
git clone https://github.com/ak0000007/Advance_RAG_MultiDoc_Chatbot.git
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

### 4. Ingest Documents

Place your operational documents (PDF, DOCX, CSV, XLSX) into `data/` and run the ingestion pipeline:

```bash
python ingest.py
```

### 5. Start the Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

- API Base URL: `http://localhost:8000`
- Interactive Swagger Docs: `http://localhost:8000/docs`
- Built-in Web UI: `http://localhost:8000/ui`

---

## API Reference

### 1. `POST /chat`
Sends a query to the copilot.

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "question": "Show all bookings for this user",
    "sf_username": "agent@travelcorp.com"
  }'
```

**Response (Normal Completion):**
```json
{
  "answer": "You currently have 2 active bookings:\n1. BK-000001: 5-Day Bali Retreat (Confirmed)\n2. BK-000002: Alpine Ski Pass (Pending)",
  "interrupted": false,
  "thread_id": "8f8b3c3c-d3f3-4e3e-bf63-c19eb1234567",
  "approval_request": null
}
```

**Response (Action Requires Approval):**
```json
{
  "answer": "",
  "interrupted": true,
  "thread_id": "8f8b3c3c-d3f3-4e3e-bf63-c19eb1234567",
  "approval_request": {
    "action": "human_approval",
    "message": "Update booking 'BK-000001': status='Confirmed'?",
    "record_id": "a005g00003ABCDE",
    "record_name": "BK-000001",
    "options": ["Approve", "Reject"]
  }
}
```

### 2. `POST /chat/resume`
Resumes an interrupted session with the human operator's decision.

**Request:**
```bash
curl -X POST http://localhost:8000/chat/resume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "thread_id": "8f8b3c3c-d3f3-4e3e-bf63-c19eb1234567",
    "decision": "Approve"
  }'
```

**Response:**
```json
{
  "answer": "Booking BK-000001 has been successfully updated to 'Confirmed'.",
  "interrupted": false,
  "thread_id": "8f8b3c3c-d3f3-4e3e-bf63-c19eb1234567",
  "approval_request": null
}
```

### 3. `GET /health`
Returns system liveness status.

---

## Verification & Testing

Run the automated test suite:

```bash
pytest tests/ -v
```

Run the standalone evaluation benchmark harness (OpenAI/Anthropic benchmark format):

```bash
python -m evals.runner
```

Current test suite validates:
- Golden dataset accuracy (16/16 test cases, 100% accuracy) across Opportunities, Bookings, Packages, Payments, and Guards
- Mock Salesforce in-memory Apex REST transactions and state retention
- Grader prompt parsing with `.partial(format_instructions=...)`
- Disallowed SOQL SObject string rejection via `ValueError`
- Safe identifier escaping and length bounds
- Regex word-boundary protection in `bulk_intent_guard`
- API key authentication and unauthorized access rejection (`401`)
- State retention of `sf_username` across `/chat/resume`
- Multi-attempt retrieval and corrective rewriter routing

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
