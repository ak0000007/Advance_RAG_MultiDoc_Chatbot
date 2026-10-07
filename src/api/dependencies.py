"""
Dependency Injection — builds heavy objects once at startup.

DIP: routes depend on abstractions (compiled graph), not on
concrete store/LLM/retriever construction details.

SRP: this module's only job is object construction.

OCP: swap providers, add retrievers, change reranker — routes
never change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.llm.provider import create_llm
from src.embeddings.embedding import BGEEmbeddings
from src.vector_stores.qdrant_store import QdrantStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker
from src.rag.chain import build_generation_chain
from src.tools.document_search import build_document_search_tool
from src.tools.salesforce_tools import build_salesforce_opportunities_tool, build_salesforce_update_tool
from src.tools.travel_tools import (
    build_get_booking_tool,
    build_get_travel_packages_tool,
    build_get_payments_tool,
    build_update_booking_tool,
    build_update_travel_package_tool,
    build_update_payment_tool,
)
from src.graph.approval import build_human_approval_node
from src.graph.graph import build_rag_graph
from src.config import settings


from src.clients.salesforce_client import SalesforceAsyncClient

@lru_cache
def get_salesforce_client() -> SalesforceAsyncClient | None:
    if not settings.sf_client_id or not settings.sf_private_key_path:
        return None
        
    return SalesforceAsyncClient(
        client_id=settings.sf_client_id,
        login_url=settings.sf_login_url,
        private_key_path=settings.sf_private_key_path,
        domain=settings.sf_domain
    )

_DEFAULT_CLIENT = object()
_DEFAULT_POOL = object()


def build_graph_with_client(sf_client=_DEFAULT_CLIENT, checkpointer=None, pool=_DEFAULT_POOL):
    """
    Build and return compiled LangGraph with injected or default Salesforce client.
    Reuses all construction logic so production and evals share a single code path.
    """
    # ------------------------------------------------
    # 1. LLM (Gemini + DeepSeek fallback)
    # ------------------------------------------------

    from src.rag.grading import RetrievalGrade
    
    fallback = "openai" if settings.openai_api_key else None
    
    # Standard LLM (for generation and query rewriting)
    llm = create_llm(
        provider="google",
        fallback_provider=fallback,
        temperature=0.0,
        max_tokens=2048,
    )
    
    # Pre-configured structured LLMs for nodes
    retrieval_grader_llm = create_llm(
        provider="google",
        fallback_provider=fallback,
        temperature=0.0,
        max_tokens=2048,
        structured_output=RetrievalGrade,
    )

    # ------------------------------------------------
    # 2. Embeddings
    # ------------------------------------------------

    bge = BGEEmbeddings()
    embeddings = bge.get_embeddings()

    # ------------------------------------------------
    # 3. Vector store (Qdrant — local persistent)
    # ------------------------------------------------

    if settings.qdrant_url:
        qdrant_store = QdrantStore(
            embeddings=embeddings,
            collection_name="multidoc_rag",
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    else:
        qdrant_store = QdrantStore(
            embeddings=embeddings,
            collection_name="multidoc_rag",
            path="./qdrant_data",
        )

    # ------------------------------------------------
    # 4. BM25 sparse retriever
    # ------------------------------------------------

    if Path(settings.bm25_store_path).exists():
        bm25_store = BM25Store.load(settings.bm25_store_path)
    else:
        bm25_store = BM25Store()

    # ------------------------------------------------
    # 5. Hybrid retriever (RRF fusion)
    # ------------------------------------------------

    hybrid = HybridRetriever(
        retrievers=[
            qdrant_store.as_dynamic_retriever(),
            bm25_store.as_dynamic_retriever(),
        ],
        final_k=20,
    )

    # ------------------------------------------------
    # 6. Cross-encoder reranker
    # ------------------------------------------------

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
    )

    retriever = reranker.wrap_retriever(
        hybrid.as_dynamic_retriever(),
        top_k=5,
    )

    # ------------------------------------------------
    # 7. Generation chain (docs already retrieved)
    # ------------------------------------------------

    generation_chain = build_generation_chain(llm)

    if checkpointer is None:
        try:
            from src.graph.memory import get_checkpointer
            checkpointer = get_checkpointer(settings.postgres_url)
        except Exception:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()

    # ------------------------------------------------
    # 8. Build Tools and Compile Graph
    # ------------------------------------------------
    
    from src.rag.grading import build_retrieval_grader
    from src.rag.rewriter import build_corrective_rewriter
    
    grader_chain = build_retrieval_grader(retrieval_grader_llm)
    rewriter_chain = build_corrective_rewriter(llm)
    
    qdrant_tool = build_document_search_tool(
        retriever, 
        grader_chain, 
        rewriter_chain, 
        settings.max_retrieval_attempts
    )

    if sf_client is _DEFAULT_CLIENT:
        sf_client = get_salesforce_client()

    tools = [qdrant_tool]
    human_approval_node = None

    if pool is _DEFAULT_POOL:
        if settings.postgres_url:
            try:
                from src.graph.memory import get_postgres_pool
                pool = get_postgres_pool(settings.postgres_url)
            except Exception:
                pool = None
        else:
            pool = None

    if sf_client:
        # ── Read-only tools: always available when SF is configured ───────────
        tools.append(build_salesforce_opportunities_tool(sf_client))
        tools.append(build_get_booking_tool(sf_client))
        tools.append(build_get_travel_packages_tool(sf_client))
        tools.append(build_get_payments_tool(sf_client))

        # ── Write tools: only when WRITES_ENABLED=true (default) ─────────────
        # Set WRITES_ENABLED=false in .env to drop all write capability instantly.
        if settings.writes_enabled:
            tools.append(
                build_salesforce_update_tool(
                    sf_client,
                    max_writes_per_session=settings.max_writes_per_session,
                )
            )
            tools.append(build_update_booking_tool(sf_client, settings.max_writes_per_session))
            tools.append(build_update_travel_package_tool(sf_client, settings.max_writes_per_session))
            tools.append(build_update_payment_tool(sf_client, settings.max_writes_per_session))
            human_approval_node = build_human_approval_node(sf_client, pool=pool)

    graph = build_rag_graph(
        llm=llm,
        tools=tools,
        checkpointer=checkpointer,
        human_approval_node=human_approval_node,
    )

    return graph


@lru_cache
def get_compiled_graph():
    """
    Build and return the compiled LangGraph once.

    Everything is wired here:
        LLM → Embeddings → Stores → Retrievers →
        Reranker → Generation chain → Graph

    Cached — subsequent calls return the same instance.
    """
    return build_graph_with_client()


def get_db_pool():
    """
    Return the shared PostgreSQL connection pool singleton.
    Uses settings.postgres_url.
    """
    if not settings.postgres_url:
        return None
    try:
        from src.graph.memory import get_postgres_pool
        return get_postgres_pool(settings.postgres_url)
    except Exception:
        return None


