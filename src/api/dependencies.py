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


def _prepare_core_dependencies(sf_client=_DEFAULT_CLIENT, pool=_DEFAULT_POOL, checkpointer=None):
    """Internal helper to construct shared LLM, RAG tools, and checkpointer."""
    from src.rag.grading import RetrievalGrade, build_retrieval_grader
    from src.rag.rewriter import build_corrective_rewriter

    if hasattr(pool, "get_tuple") or hasattr(pool, "put"):
        checkpointer = pool
        pool = _DEFAULT_POOL

    fallback = "openai" if settings.openai_api_key else None
    llm = create_llm(
        provider="google",
        fallback_provider=fallback,
        temperature=0.0,
        max_tokens=2048,
    )
    retrieval_grader_llm = create_llm(
        provider="google",
        fallback_provider=fallback,
        temperature=0.0,
        max_tokens=2048,
        structured_output=RetrievalGrade,
    )

    bge = BGEEmbeddings()
    embeddings = bge.get_embeddings()

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

    if Path(settings.bm25_store_path).exists():
        bm25_store = BM25Store.load(settings.bm25_store_path)
    else:
        bm25_store = BM25Store()

    hybrid = HybridRetriever(
        retrievers=[
            qdrant_store.as_dynamic_retriever(),
            bm25_store.as_dynamic_retriever(),
        ],
        final_k=20,
    )

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
    )

    retriever = reranker.wrap_retriever(
        hybrid.as_dynamic_retriever(),
        top_k=5,
    )

    grader_chain = build_retrieval_grader(retrieval_grader_llm)
    rewriter_chain = build_corrective_rewriter(llm)

    qdrant_tool = build_document_search_tool(
        retriever,
        grader_chain,
        rewriter_chain,
        settings.max_retrieval_attempts,
    )

    if sf_client is _DEFAULT_CLIENT:
        sf_client = get_salesforce_client()

    if pool is _DEFAULT_POOL:
        if settings.postgres_url:
            try:
                from src.graph.memory import get_postgres_pool
                pool = get_postgres_pool(settings.postgres_url)
            except Exception:
                pool = None
        else:
            pool = None

    if checkpointer is None:
        try:
            from src.graph.memory import get_checkpointer
            checkpointer = get_checkpointer(settings.postgres_url)
        except Exception:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()

    return llm, qdrant_tool, sf_client, pool, checkpointer


def build_graph_with_client(
    sf_client=_DEFAULT_CLIENT,
    checkpointer=None,
    pool=_DEFAULT_POOL,
    force_single_agent: bool = False,
):
    """
    Build and return compiled LangGraph with injected or default Salesforce client.
    Reuses all construction logic so production and evals share a single code path.
    """
    if not force_single_agent and settings.use_multi_agent_architecture:
        return build_multi_agent_graph(
            sf_client=sf_client,
            pool=pool,
            checkpointer=checkpointer,
        )

    llm, qdrant_tool, sf_client, pool, checkpointer = _prepare_core_dependencies(
        sf_client=sf_client,
        pool=pool,
        checkpointer=checkpointer,
    )

    tools = [qdrant_tool]
    human_approval_node = None

    if sf_client:
        tools.append(build_salesforce_opportunities_tool(sf_client))
        tools.append(build_get_booking_tool(sf_client))
        tools.append(build_get_travel_packages_tool(sf_client))
        tools.append(build_get_payments_tool(sf_client))

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


def build_multi_agent_graph(
    sf_client=_DEFAULT_CLIENT,
    pool=_DEFAULT_POOL,
    checkpointer=None,
    **kwargs,
):
    """
    Build and return compiled multi-agent orchestrator workflow with three domain specialists:
      - crm_agent: search_salesforce_opportunities, update_salesforce_opportunity_status
      - travel_agent: get_booking, update_booking, get_travel_packages, update_travel_package, get_payments, update_payment
      - docs_agent: search_documents (no write tools, no approval node)
    """
    from src.graph.prompts import CRM_SYSTEM_PROMPT, TRAVEL_SYSTEM_PROMPT, DOCS_SYSTEM_PROMPT
    from src.graph.orchestrator import build_orchestrator_graph

    # Handle if checkpointer was passed as 2nd positional argument
    if hasattr(pool, "get_tuple") or hasattr(pool, "put"):
        checkpointer = pool
        pool = _DEFAULT_POOL

    llm, qdrant_tool, sf_client, pool, checkpointer = _prepare_core_dependencies(
        sf_client=sf_client,
        pool=pool,
        checkpointer=checkpointer,
    )

    # 1. CRM Specialist Tools
    crm_tools = []
    crm_approval_node = None
    if sf_client:
        crm_tools.append(build_salesforce_opportunities_tool(sf_client))
        if settings.writes_enabled:
            crm_tools.append(
                build_salesforce_update_tool(
                    sf_client,
                    max_writes_per_session=settings.max_writes_per_session,
                )
            )
            crm_approval_node = build_human_approval_node(sf_client, pool=pool)

    # 2. Travel Specialist Tools
    travel_tools = []
    travel_approval_node = None
    if sf_client:
        travel_tools.append(build_get_booking_tool(sf_client))
        travel_tools.append(build_get_travel_packages_tool(sf_client))
        travel_tools.append(build_get_payments_tool(sf_client))
        if settings.writes_enabled:
            travel_tools.append(build_update_booking_tool(sf_client, settings.max_writes_per_session))
            travel_tools.append(build_update_travel_package_tool(sf_client, settings.max_writes_per_session))
            travel_tools.append(build_update_payment_tool(sf_client, settings.max_writes_per_session))
            travel_approval_node = build_human_approval_node(sf_client, pool=pool)

    # 3. Docs Specialist Tools
    docs_tools = [qdrant_tool]

    # Build the 3 specialist subgraphs using build_rag_graph
    crm_graph = build_rag_graph(
        llm=llm,
        tools=crm_tools,
        checkpointer=checkpointer,
        human_approval_node=crm_approval_node,
        system_prompt=CRM_SYSTEM_PROMPT,
    )

    travel_graph = build_rag_graph(
        llm=llm,
        tools=travel_tools,
        checkpointer=checkpointer,
        human_approval_node=travel_approval_node,
        system_prompt=TRAVEL_SYSTEM_PROMPT,
    )

    docs_graph = build_rag_graph(
        llm=llm,
        tools=docs_tools,
        checkpointer=checkpointer,
        human_approval_node=None,
        system_prompt=DOCS_SYSTEM_PROMPT,
    )

    return build_orchestrator_graph(
        crm_graph=crm_graph,
        travel_graph=travel_graph,
        docs_graph=docs_graph,
        llm=llm,
        checkpointer=checkpointer,
    )


@lru_cache
def get_compiled_graph():
    """
    Build and return the single-agent compiled LangGraph once.
    Cached — subsequent calls return the same instance.
    """
    return build_graph_with_client(force_single_agent=True)


@lru_cache
def get_compiled_multi_agent_graph():
    """
    Build and return the multi-agent orchestrator LangGraph once.
    Cached — subsequent calls return the same instance.
    """
    return build_multi_agent_graph()


def get_active_graph():
    """
    Return active graph according to settings.use_multi_agent_architecture.
    Provides dependency injection for FastAPI routes.
    """
    if settings.use_multi_agent_architecture:
        return get_compiled_multi_agent_graph()
    return get_compiled_graph()


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


