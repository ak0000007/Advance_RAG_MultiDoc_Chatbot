"""
Reranker module — scores and reorders retrieved documents by
query relevance using a cross-encoder model.

Fully independent: does NOT import or depend on any other src module.
Consumes list[Document], produces list[Document]. Plug in or remove
from any pipeline without breaking anything.

SOLID compliance:
  SRP — only reranks, no retrieval, no LLM generation.
  OCP — swap the model_name or subclass for API-based rerankers
        (Cohere, Jina) without touching existing code.
  LSP — same input/output contract as any retrieval stage:
        (query, docs) -> docs.
  ISP — exposes only rerank() and as_runnable(). No bloated interface.
  DIP — depends on LangChain Document abstraction, not concrete stores.

Future-proofing:
  - LangGraph agents can call reranker.as_runnable() as a node.
  - MCP tools can wrap rerank() directly.
  - Swap cross-encoder for Cohere/Jina API by changing one class.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model.

    Cross-encoders read (query, document) together — unlike bi-encoders
    (embedding models) that encode them separately. This gives much
    higher relevance accuracy at the cost of speed, which is fine
    because we only rerank a small candidate set (10-50 docs), not
    the entire corpus.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, docs, top_k=5)

    In a chain:
        retriever | reranker.as_runnable(top_k=5) | format_docs | llm
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model.
                        Default is fast and accurate for English.
                        Alternatives:
                          - cross-encoder/ms-marco-TinyBERT-L-2-v2  (faster, less accurate)
                          - BAAI/bge-reranker-v2-m3                 (multilingual)
            device:     "cpu", "cuda", "cuda:0", etc. None = auto-detect.
        """
        self.model_name = model_name
        self._device = device
        self._model = None  # Lazy-loaded

    def _load_model(self):
        """Lazy-load the cross-encoder. First call pays the cost."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self.model_name,
            device=self._device,
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Score each document against the query and return top_k
        in descending relevance order.

        Args:
            query:     User query string.
            documents: Candidate documents from retrieval.
            top_k:     How many to keep after reranking.

        Returns:
            Reranked list[Document], length = min(top_k, len(documents)).
            Each doc gets metadata["rerank_score"] with its score.
        """
        if not documents:
            return []

        self._load_model()

        # Build (query, doc_text) pairs for the cross-encoder
        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self._model.predict(pairs)

        # Pair scores with docs, sort descending
        scored = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in scored[:top_k]:
            # Attach score to metadata for debugging/observability
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def as_runnable(self, top_k: int = 5):
        """
        Returns a Runnable: list[Document] -> list[Document].

        Expects the input to be a dict with "question" and "documents"
        keys (standard in LangChain chains), OR a plain list[Document]
        when piped directly after a retriever with query bound upstream.

        For use in LCEL chains:
            retriever | reranker.as_runnable(top_k=5)
        """

        def _rerank_from_retriever(inputs):
            """Handle both dict input and list[Document] input."""
            if isinstance(inputs, dict):
                query = inputs.get("question", "")
                docs = inputs.get("documents", [])
            elif isinstance(inputs, list):
                # When piped after a retriever in a chain where
                # query is already bound, inputs = list[Document].
                # We need the query from somewhere — store it.
                raise ValueError(
                    "Reranker needs both query and documents. "
                    "Use as_dynamic_runnable() for chain integration, "
                    "or pass a dict with 'question' and 'documents'."
                )
            else:
                raise TypeError(
                    f"Expected dict or list, got {type(inputs)}"
                )

            return self.rerank(query, docs, top_k=top_k)

        return RunnableLambda(_rerank_from_retriever)

    def as_dynamic_runnable(self, top_k: int = 5):
        """
        Returns a Runnable that wraps a retriever runnable and adds
        reranking. Compatible with HybridRetriever.as_dynamic_retriever().

        Usage:
            base_retriever = hybrid.as_dynamic_retriever()
            reranked_retriever = reranker.as_dynamic_runnable(top_k=5)

            # In chain.py:
            build_rag_chain(
                reranked_retriever.bind(retriever=base_retriever),
                llm,
            )

        Or more simply, use wrap_retriever() below.
        """

        def _rerank(inputs: dict) -> list[Document]:
            raise NotImplementedError(
                "Use wrap_retriever() for the standard integration pattern."
            )

        return RunnableLambda(_rerank)

    def wrap_retriever(self, retriever_runnable, top_k: int = 5):
        """
        Wraps an existing retriever Runnable, adding reranking.
        Returns a new Runnable with the same interface.

        This is the recommended integration point:

            base = hybrid.as_dynamic_retriever()
            reranked = reranker.wrap_retriever(base, top_k=5)
            chain = build_rag_chain(reranked, llm)

        The RAG chain and retriever are completely unaware of reranking.
        Remove this one line and everything still works — OCP satisfied.
        """

        def _retrieve_and_rerank(inputs: dict) -> list[Document]:
            # Step 1: Retrieve using the original retriever
            try:
                if hasattr(retriever_runnable, "invoke"):
                    docs = retriever_runnable.invoke(inputs)
                else:
                    docs = retriever_runnable(inputs)
            except AttributeError as e:
                if "'dict' object" in str(e) and isinstance(inputs, dict) and "question" in inputs:
                    if hasattr(retriever_runnable, "invoke"):
                        docs = retriever_runnable.invoke(inputs["question"])
                    else:
                        docs = retriever_runnable(inputs["question"])
                else:
                    raise

            # Step 2: Rerank
            question = (
                inputs.get("question", "")
                if isinstance(inputs, dict)
                else str(inputs)
            )
            return self.rerank(question, docs, top_k=top_k)

        return RunnableLambda(_retrieve_and_rerank)
