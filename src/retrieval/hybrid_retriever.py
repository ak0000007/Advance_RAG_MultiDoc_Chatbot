"""
Hybrid retriever — merges results from multiple retrieval sources.

Combines vector (dense) and keyword (sparse) retrieval via
Reciprocal Rank Fusion (RRF). Depends on abstractions
(LangChain Runnables), not concrete stores — DIP satisfied.

Adding/removing retrieval sources requires no code change to
existing stores — OCP satisfied.
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Merge multiple ranked document lists using RRF.

    RRF score = sum( 1 / (k + rank) ) across all lists.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            # Use page_content hash as dedup key since
            # docs from different sources won't share IDs.
            doc_key = str(hash(doc.page_content))

            if doc_key not in doc_map:
                doc_map[doc_key] = doc
                scores[doc_key] = 0.0

            scores[doc_key] += 1.0 / (k + rank + 1)

    sorted_keys = sorted(
        scores.keys(), key=lambda x: scores[x], reverse=True
    )
    return [doc_map[key] for key in sorted_keys]


class HybridRetriever:
    """
    Combines N retrieval sources via RRF.

    Constructor receives retrieval callables — any function
    (query: str) -> list[Document] works. This means:
    - QdrantStore.as_retriever().invoke
    - BM25Store.search
    - Any future MCP/agentic source
    ...all plug in without modifying this class (OCP).

    Usage:
        hybrid = HybridRetriever(
            retrievers=[
                qdrant_store.as_retriever(search_kwargs={"k": 30}),
                bm25_store.as_retriever(search_kwargs={"k": 30}),
            ],
            final_k=10,
        )
        chain_retriever = hybrid.as_dynamic_retriever()
    """

    def __init__(
        self,
        retrievers: list,
        final_k: int = 10,
        rrf_k: int = 60,
    ):
        """
        Args:
            retrievers: List of LangChain Runnables or callables
                        that accept a query string and return list[Document].
            final_k:    Number of documents to return after fusion.
            rrf_k:      RRF constant (default 60, per original paper).
        """
        if not retrievers:
            raise ValueError("At least one retriever required.")
        self.retrievers = retrievers
        self.final_k = final_k
        self.rrf_k = rrf_k

    def search(self, query: str) -> list[Document]:
        """Run all retrievers, fuse results via RRF, return top final_k."""
        all_results: list[list[Document]] = []
        for retriever in self.retrievers:
            # Support both Runnable.invoke() and plain callables
            if hasattr(retriever, "invoke"):
                docs = retriever.invoke(query)
            else:
                docs = retriever(query)
            all_results.append(docs)

        fused = reciprocal_rank_fusion(all_results, k=self.rrf_k)
        return fused[: self.final_k]

    # ------------------------------------------------------------------
    # LangChain integration — same interface as QdrantStore / BM25Store
    # ------------------------------------------------------------------

    def as_retriever(self):
        """Return a Runnable that takes a query string."""

        def _retrieve(query: str) -> list[Document]:
            return self.search(query)

        return RunnableLambda(_retrieve)

    def as_dynamic_retriever(self):
        """
        Return a Runnable that accepts {"question": ..., "metadata_filter": ...}
        Same interface as QdrantStore.as_dynamic_retriever — DIP satisfied.

        Note: metadata_filter is handled inside each individual retriever's
        dynamic retriever. HybridRetriever simply delegates.
        """

        def _retrieve(inputs: dict) -> list[Document]:
            question = inputs.get("question", "")
            all_results: list[list[Document]] = []
            for retriever in self.retrievers:
                if hasattr(retriever, "invoke"):
                    docs = retriever.invoke(inputs)
                else:
                    docs = retriever(inputs)
                all_results.append(docs)

            fused = reciprocal_rank_fusion(all_results, k=self.rrf_k)
            return fused[: self.final_k]

        return RunnableLambda(_retrieve)
