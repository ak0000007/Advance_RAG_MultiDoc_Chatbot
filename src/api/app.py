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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

from src.config import settings
from src.telemetry import setup_langsmith
setup_langsmith()

from src.api.routes import router
from src.api.dependencies import get_compiled_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Validate runtime configuration, eagerly build graph, and initialize shared DB pool + audit tables.
    Cleanly shuts down DB connection pool and Salesforce HTTP connection pool on exit.
    """
    import logging
    app_logger = logging.getLogger(__name__)

    from src.startup_validation import validate_runtime_config
    fatals, warnings = validate_runtime_config(settings)
    for w in warnings:
        app_logger.warning("Startup Configuration Warning: %s", w)

    if fatals:
        if settings.dev_mode:
            for f in fatals:
                app_logger.warning("DEV_MODE: ignoring fatal configuration error: %s", f)
        else:
            msg = "Application startup aborted due to configuration errors:\n  - " + "\n  - ".join(fatals)
            app_logger.error(msg)
            raise RuntimeError(msg)

    from src.api.dependencies import get_db_pool, get_salesforce_client
    from src.audit.log import init_audit_table

    pool = get_db_pool()
    if pool is not None:
        await pool.open()
        await init_audit_table(pool)

    sf_client = get_salesforce_client()

    graph = get_compiled_graph()
    if hasattr(graph, "checkpointer") and hasattr(graph.checkpointer, "setup"):
        import inspect
        if inspect.iscoroutinefunction(graph.checkpointer.setup):
            await graph.checkpointer.setup()
        else:
            graph.checkpointer.setup()
    yield

    if pool is not None:
        await pool.close()

    if sf_client is not None:
        await sf_client.aclose()



app = FastAPI(
    title="Advanced RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for cross-origin frontend support (Debt H9)
# Per W3C CORS spec: allow_credentials MUST NOT be True when allow_origins contains "*"
has_wildcard = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not has_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/ui", StaticFiles(directory="src/ui/static", html=True), name="ui")
