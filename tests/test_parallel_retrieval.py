import asyncio
import time
import pytest
from langchain_core.documents import Document
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.reranking.reranker import CrossEncoderReranker


class AsyncMockRetriever:
    def __init__(self, name: str, delay: float, docs: list[Document]):
        self.name = name
        self.delay = delay
        self.docs = docs
        self.call_count = 0

    async def ainvoke(self, query):
        self.call_count += 1
        await asyncio.sleep(self.delay)
        return self.docs

    def invoke(self, query):
        self.call_count += 1
        time.sleep(self.delay)
        return self.docs


class SyncMockRetriever:
    def __init__(self, name: str, delay: float, docs: list[Document]):
        self.name = name
        self.delay = delay
        self.docs = docs
        self.call_count = 0

    def invoke(self, query):
        self.call_count += 1
        time.sleep(self.delay)
        return self.docs


@pytest.mark.asyncio
async def test_async_parallel_speedup():
    """
    Verify that 2 retrievers each taking 0.15s execute concurrently
    in ~0.15s rather than serially taking 0.30s.
    """
    doc1 = Document(page_content="Doc from dense", metadata={"source": "qdrant", "id": "1"})
    doc2 = Document(page_content="Doc from sparse", metadata={"source": "bm25", "id": "2"})

    r1 = AsyncMockRetriever(name="dense", delay=0.15, docs=[doc1])
    r2 = SyncMockRetriever(name="sparse", delay=0.15, docs=[doc2])

    hybrid = HybridRetriever(retrievers=[r1, r2], final_k=5)

    start = time.perf_counter()
    results = await hybrid.asearch("test query")
    duration = time.perf_counter() - start

    assert len(results) == 2
    assert r1.call_count == 1
    assert r2.call_count == 1
    # Serial execution would be >= 0.30s. Parallel should be < 0.25s.
    assert duration < 0.25, f"Expected parallel duration < 0.25s, got {duration:.3f}s"


def test_sync_parallel_speedup():
    """
    Verify that sync .search() also executes via ThreadPoolExecutor concurrently.
    """
    doc1 = Document(page_content="Doc from dense", metadata={"id": "1"})
    doc2 = Document(page_content="Doc from sparse", metadata={"id": "2"})

    r1 = SyncMockRetriever(name="dense", delay=0.15, docs=[doc1])
    r2 = SyncMockRetriever(name="sparse", delay=0.15, docs=[doc2])

    hybrid = HybridRetriever(retrievers=[r1, r2], final_k=5)

    start = time.perf_counter()
    results = hybrid.search("test query")
    duration = time.perf_counter() - start

    assert len(results) == 2
    assert r1.call_count == 1
    assert r2.call_count == 1
    assert duration < 0.25, f"Expected parallel sync duration < 0.25s, got {duration:.3f}s"


@pytest.mark.asyncio
async def test_dynamic_retriever_ainvoke_with_dict():
    """
    Verify that dynamic retriever handles {"question": ..., "metadata_filter": ...}
    via .ainvoke().
    """
    doc = Document(page_content="Filtered result", metadata={"category": "travel"})
    r = AsyncMockRetriever(name="mock", delay=0.01, docs=[doc])

    hybrid = HybridRetriever(retrievers=[r], final_k=5)
    dynamic_runnable = hybrid.as_dynamic_retriever()

    results = await dynamic_runnable.ainvoke({"question": "cancellation policy", "metadata_filter": None})
    assert len(results) == 1
    assert results[0].page_content == "Filtered result"


@pytest.mark.asyncio
async def test_reranker_wrap_retriever_async(monkeypatch):
    """
    Verify that CrossEncoderReranker.wrap_retriever properly delegates
    async calls with ainvoke and returns reranked results.
    """
    doc1 = Document(page_content="Doc 1", metadata={"id": "1"})
    doc2 = Document(page_content="Doc 2", metadata={"id": "2"})
    r = AsyncMockRetriever(name="mock", delay=0.01, docs=[doc1, doc2])

    reranker = CrossEncoderReranker()
    # Mock rerank to avoid downloading/loading CrossEncoder model during test
    monkeypatch.setattr(
        reranker,
        "rerank",
        lambda query, docs, top_k=5: [docs[1], docs[0]][:top_k]
    )

    wrapped = reranker.wrap_retriever(r, top_k=2)

    # Test async path
    results = await wrapped.ainvoke({"question": "test"})
    assert len(results) == 2
    assert results[0].page_content == "Doc 2"
    assert results[1].page_content == "Doc 1"

    # Test sync path
    sync_results = wrapped.invoke({"question": "test"})
    assert len(sync_results) == 2
    assert sync_results[0].page_content == "Doc 2"
