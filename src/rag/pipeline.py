from typing import Optional, List, Any, Iterator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker
from src.rag.chain import build_rag_chain


class RAGPipeline:
    """
    End-to-end RAG facade.
    Production ready: supports async, streaming, LangSmith configs, and generic retriever injection.
    """

    def __init__(
        self,
        qdrant_retriever=None,
        bm25_retriever=None,
        llm=None,
        reranker_model_name="BAAI/bge-reranker-v2-m3",
        retrieval_k: int = 20,
        final_k: int = 5,
        retrievers: Optional[List[Any]] = None,
        reranker: Optional[Any] = None,
    ):
        self.llm = llm

        # --------------------------------------------------
        # 1. Backward Compatibility & DI
        # --------------------------------------------------
        if retrievers is None:
            # Fallback to legacy arguments
            retrievers = [r for r in [qdrant_retriever, bm25_retriever] if r is not None]

        if reranker is None and reranker_model_name:
            reranker = CrossEncoderReranker(model_name=reranker_model_name)

        # --------------------------------------------------
        # 2. Hybrid Retrieval
        # --------------------------------------------------
        hybrid = HybridRetriever(retrievers=retrievers, final_k=retrieval_k)
        base_retriever = hybrid.as_dynamic_retriever()

        # --------------------------------------------------
        # 3. Cross-encoder reranking
        # --------------------------------------------------
        if reranker:
            self.retriever = reranker.wrap_retriever(base_retriever, top_k=final_k)
        else:
            self.retriever = base_retriever

        # --------------------------------------------------
        # 4. Complete LCEL RAG chain
        # --------------------------------------------------
        self.chain = build_rag_chain(
            retriever_runnable=self.retriever,
            llm=self.llm,
        )

    def get_chain(self):
        """
        Return the underlying LCEL RAG chain.

        This allows higher-level application layers such as
        conversational RAG and LangGraph to compose the existing
        RAG pipeline without duplicating its internals.
        """
        return self.chain    

    def invoke(self, question: str, metadata_filter: Optional[dict] = None, config: Optional[dict] = None):
        """Standard synchronous execution."""
        return self.chain.invoke({"question": question, "metadata_filter": metadata_filter}, config=config)

    async def ainvoke(self, question: str, metadata_filter: Optional[dict] = None, config: Optional[dict] = None):
        """Async execution for concurrent API servers."""
        return await self.chain.ainvoke({"question": question, "metadata_filter": metadata_filter}, config=config)

    def stream(self, question: str, metadata_filter: Optional[dict] = None, config: Optional[dict] = None) -> Iterator[str]:
        """Token streaming for UI/chatbots."""
        return self.chain.stream({"question": question, "metadata_filter": metadata_filter}, config=config)

    def get_chain(self):
        """Return the underlying LCEL RAG chain."""
        return self.chain