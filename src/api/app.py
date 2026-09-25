"""
FastAPI application — the single wiring point.

This file ONLY connects pieces. It does not contain:
- Business logic  (→ graph/nodes)
- Object construction  (→ dependencies)
- Data shapes  (→ schemas)
- Route handlers  (→ routes)

SOLID summary:
    SRP — each module has one reason to change.
    OCP — add routes via new routers, add deps via new factories.
    LSP — graph is a compiled StateGraph; any LangGraph works.
    ISP — schemas are small, purpose-built.
    DIP — routes depend on get_compiled_graph(), not concrete classes.

To extend:
    New endpoint?  → add router in routes.py (or new file), include here.
    New provider?  → change dependencies.py.
    New schema?    → add to schemas.py.

To minimize:
    Remove a router include line. Nothing else breaks.

Run:
    uvicorn src.api.app:app --reload
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from src.api.routes import router
from src.api.dependencies import get_compiled_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eagerly build graph at startup so first request is fast.
    """
    get_compiled_graph()
    yield


app = FastAPI(
    title="Advanced RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
