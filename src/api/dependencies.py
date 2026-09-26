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

from src.llm.provider import create_llm
from src.embeddings.embedding import BGEEmbeddings
from src.vector_stores.qdrant_store import QdrantStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker
from src.rag.chain import build_generation_chain
from src.graph.graph import build_rag_graph
from src.config import settings


@lru_cache
def get_compiled_graph():
    """
    Build and return the compiled LangGraph once.

    Everything is wired here:
        LLM → Embeddings → Stores → Retrievers →
        Reranker → Generation chain → Graph

    Cached — subsequent calls return the same instance.
    """

    # ------------------------------------------------
    # 1. LLM (Gemini + DeepSeek fallback)
    # ------------------------------------------------

    fallback = "openai" if settings.openai_api_key else None
    llm = create_llm(
        provider="google",
        fallback_provider=fallback,
        temperature=0.0,
    )

    # ------------------------------------------------
    # 2. Embeddings
    # ------------------------------------------------

    bge = BGEEmbeddings()
    embeddings = bge.get_embeddings()

    # ------------------------------------------------
    # 3. Vector store (Qdrant — local persistent)
    # ------------------------------------------------

    qdrant_store = QdrantStore(
        embeddings=embeddings,
        collection_name="multidoc_rag",
        path="./qdrant_data",
    )

    # ------------------------------------------------
    # 4. BM25 sparse retriever
    #    ponytail: BM25 starts empty at server boot.
    #    If you need pre-loaded BM25, call
    #    bm25.add_documents() here with your corpus.
    # ------------------------------------------------

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

    from src.graph.memory import get_checkpointer
    checkpointer = get_checkpointer(settings.postgres_url)

    # ------------------------------------------------
    # 8. Compile graph
    # ------------------------------------------------

    graph = build_rag_graph(
        llm=llm,
        retriever=retriever,
        generation_chain=generation_chain,
        score_threshold=settings.retrieval_confidence_threshold,
        min_confident_docs=settings.retrieval_min_confident_docs,
        checkpointer=checkpointer,
    )

    return graph
