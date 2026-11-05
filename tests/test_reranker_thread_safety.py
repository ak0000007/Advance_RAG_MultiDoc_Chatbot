import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document

from src.reranking.reranker import CrossEncoderReranker


class MockCrossEncoder:
    init_counter = 0

    def __init__(self, model_name, device=None):
        MockCrossEncoder.init_counter += 1
        # Add latency to simulate heavy model weight loading & widen race condition window
        time.sleep(0.05)
        self.model_name = model_name
        self.device = device

    def predict(self, pairs):
        # Return arbitrary scores matching pair length
        return [0.95 - (i * 0.1) for i in range(len(pairs))]


@pytest.fixture(autouse=True)
def reset_mock_counter():
    MockCrossEncoder.init_counter = 0
    yield
    MockCrossEncoder.init_counter = 0


@pytest.mark.asyncio
async def test_concurrent_async_model_loading_single_instance():
    """Verify H14 fix: 10 concurrent coroutines calling _load_model_async instantiate CrossEncoder exactly once."""
    reranker = CrossEncoderReranker()

    with patch("sentence_transformers.CrossEncoder", side_effect=MockCrossEncoder):
        # Launch 10 concurrent coroutines simultaneously
        tasks = [reranker._load_model_async() for _ in range(10)]
        await asyncio.gather(*tasks)

        assert MockCrossEncoder.init_counter == 1
        assert reranker._model is not None
        assert isinstance(reranker._model, MockCrossEncoder)


def test_concurrent_threaded_model_loading_single_instance():
    """Verify H14 fix: 10 concurrent OS threads calling _load_model instantiate CrossEncoder exactly once."""
    reranker = CrossEncoderReranker()

    with patch("sentence_transformers.CrossEncoder", side_effect=MockCrossEncoder):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(reranker._load_model) for _ in range(10)]
            for f in futures:
                f.result()

        assert MockCrossEncoder.init_counter == 1
        assert reranker._model is not None
        assert isinstance(reranker._model, MockCrossEncoder)


@pytest.mark.asyncio
async def test_subsequent_calls_bypass_initialization():
    """Verify that once loaded, subsequent calls do not re-run constructor."""
    reranker = CrossEncoderReranker()

    with patch("sentence_transformers.CrossEncoder", side_effect=MockCrossEncoder):
        await reranker._load_model_async()
        assert MockCrossEncoder.init_counter == 1

        # Second async call
        await reranker._load_model_async()
        assert MockCrossEncoder.init_counter == 1

        # Third sync call
        reranker._load_model()
        assert MockCrossEncoder.init_counter == 1


@pytest.mark.asyncio
async def test_wrap_retriever_concurrent_ainvoke():
    """Verify that wrap_retriever ainvoke handles concurrent cold-start requests safely."""
    reranker = CrossEncoderReranker()

    mock_docs = [
        Document(page_content="Policy doc A", metadata={"id": "A"}),
        Document(page_content="Policy doc B", metadata={"id": "B"}),
    ]

    mock_retriever = MagicMock()
    async def mock_ainvoke(inputs):
        return mock_docs
    mock_retriever.ainvoke = mock_ainvoke

    with patch("sentence_transformers.CrossEncoder", side_effect=MockCrossEncoder):
        wrapped = reranker.wrap_retriever(mock_retriever, top_k=2)

        # Run 5 concurrent user queries hitting un-warmed reranker
        tasks = [wrapped.ainvoke({"question": f"Query {i}"}) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert MockCrossEncoder.init_counter == 1
        assert len(results) == 5
        for res in results:
            assert len(res) == 2
            assert res[0].metadata["rerank_score"] > res[1].metadata["rerank_score"]
