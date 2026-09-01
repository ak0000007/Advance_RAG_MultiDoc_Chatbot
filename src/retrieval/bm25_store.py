"""
BM25 keyword retriever.

Uses BM25 (Okapi) scoring over pre-chunked LangChain Documents.
Independent of any vector store — satisfies SRP and DIP.
"""

import math
from collections import Counter

from langchain_core.documents import Document


class BM25Store:
    """
    In-memory BM25 index over LangChain Documents.

    Operates independently of the vector store.
    Consumers interact via `as_retriever()` which returns
    a LangChain-compatible Runnable.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self._documents: list[Document] = []
        self._doc_freqs: list[Counter] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Index documents. Can be called multiple times (appends)."""
        self._documents.extend(documents)
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase whitespace tokenizer. Good enough for BM25."""
        return text.lower().split()

    def _build_index(self) -> None:
        self._doc_freqs = [
            Counter(self._tokenize(doc.page_content))
            for doc in self._documents
        ]

        total_tokens = sum(sum(tf.values()) for tf in self._doc_freqs)
        n_docs = len(self._documents)
        self._avg_dl = total_tokens / n_docs if n_docs else 1.0

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df_counts: Counter = Counter()
        for tf in self._doc_freqs:
            df_counts.update(tf.keys())

        self._idf = {}
        for term, df in df_counts.items():
            self._idf[term] = math.log(
                (n_docs - df + 0.5) / (df + 0.5) + 1
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        tf = self._doc_freqs[doc_idx]
        dl = sum(tf.values())
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            term_freq = tf[token]
            idf = self._idf.get(token, 0.0)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * dl / self._avg_dl
            )
            score += idf * (numerator / denominator)
        return score

    def search(
        self,
        query: str,
        k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        """
        Return top-k documents by BM25 score.
        Optional metadata_filter for exact-match filtering.
        """
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)

        scored: list[tuple[float, int]] = []
        for idx in range(len(self._documents)):
            # Apply metadata filter
            if metadata_filter:
                doc_meta = self._documents[idx].metadata
                if not all(
                    doc_meta.get(key) == value
                    for key, value in metadata_filter.items()
                ):
                    continue
            scored.append((self._score(query_tokens, idx), idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            self._documents[idx]
            for _, idx in scored[:k]
        ]

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def as_retriever(self, search_kwargs: dict | None = None):
        """Return a LangChain Runnable for use in chains."""
        from langchain_core.runnables import RunnableLambda

        k = (search_kwargs or {}).get("k", 20)

        def _retrieve(query: str) -> list[Document]:
            return self.search(query=query, k=k)

        return RunnableLambda(_retrieve)

    def as_dynamic_retriever(self, base_search_kwargs: dict | None = None):
        """
        Returns a Runnable that accepts {"question": ..., "metadata_filter": ...}
        Same interface as QdrantStore.as_dynamic_retriever — DIP satisfied.
        """
        from langchain_core.runnables import RunnableLambda

        k = (base_search_kwargs or {}).get("k", 20)

        def _retrieve(inputs: dict) -> list[Document]:
            question = inputs.get("question", "")
            metadata_filter = inputs.get("metadata_filter", None)
            return self.search(
                query=question, k=k, metadata_filter=metadata_filter
            )

        return RunnableLambda(_retrieve)
