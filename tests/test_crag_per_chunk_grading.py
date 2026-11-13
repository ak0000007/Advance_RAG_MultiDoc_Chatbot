"""
Tests for CRAG Parallel Per-Chunk Grading, Filtering, and Tunable Retry Thresholds.
"""

import pytest
from langchain_core.documents import Document
from src.tools.document_search import build_document_search_tool
from src.rag.grading import ChunkGrade, build_chunk_grader


class FakeChunkRetriever:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.queries = []

    async def ainvoke(self, input_dict, config=None):
        query = input_dict["question"]
        self.queries.append(query)
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class FakeChunkGrader:
    def __init__(self, grade_mapping: dict[str, bool]):
        self.grade_mapping = grade_mapping
        self.calls = []

    async def ainvoke(self, input_dict, config=None):
        chunk_text = input_dict.get("chunk") or input_dict.get("context") or ""
        self.calls.append(chunk_text)
        is_rel = self.grade_mapping.get(chunk_text, True)
        reason = "Matches question" if is_rel else "Irrelevant off-topic text"
        return ChunkGrade(is_relevant=is_rel, reason=reason)


class FakeChunkRewriter:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    async def ainvoke(self, input_dict, config=None):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


@pytest.mark.asyncio
async def test_parallel_chunk_grading_filters_irrelevant_chunks():
    """
    Verify that 5 candidate chunks are graded in parallel and only the
    relevant chunks are returned to generation. Irrelevant chunks are dropped.
    """
    docs = [
        Document(page_content="Policy: Gold members get 100% refund.", metadata={"source": "refunds.pdf", "page": 1}),
        Document(page_content="Unrelated lunch menu for staff.", metadata={"source": "menu.pdf", "page": 2}),
        Document(page_content="Cancellation within 24h incurs $50 fee.", metadata={"source": "refunds.pdf", "page": 3}),
        Document(page_content="Company holiday schedule 2026.", metadata={"source": "holidays.pdf", "page": 1}),
        Document(page_content="Full credit given if flight cancelled by airline.", metadata={"source": "flights.pdf", "page": 5}),
    ]

    retriever = FakeChunkRetriever([docs])
    grader = FakeChunkGrader({
        "Policy: Gold members get 100% refund.": True,
        "Unrelated lunch menu for staff.": False,
        "Cancellation within 24h incurs $50 fee.": True,
        "Company holiday schedule 2026.": False,
        "Full credit given if flight cancelled by airline.": True,
    })
    rewriter = FakeChunkRewriter([])

    # 2 of 5 irrelevant (0.4) < threshold 0.8 -> no rewrite needed
    tool = build_document_search_tool(
        retriever,
        grader,
        rewriter,
        max_attempts=2,
        irrelevance_threshold=0.8,
    )
    result = await tool.ainvoke({"query": "What is the cancellation refund policy?"})

    assert "Gold members get 100% refund." in result
    assert "Cancellation within 24h incurs $50 fee." in result
    assert "Full credit given if flight cancelled by airline." in result

    # Irrelevant chunks MUST be filtered out
    assert "Unrelated lunch menu for staff." not in result
    assert "Company holiday schedule 2026." not in result

    # Grader called for all 5 chunks in parallel
    assert len(grader.calls) == 5
    assert retriever.call_count == 1
    assert rewriter.call_count == 0


@pytest.mark.asyncio
async def test_tunable_retry_threshold_triggers_rewrite_when_exceeded():
    """
    Verify that when 4 out of 5 chunks are irrelevant (ratio = 0.8 >= threshold 0.8),
    a corrective query rewrite is triggered and retrieval re-runs.
    """
    attempt1_docs = [
        Document(page_content="Bad chunk 1", metadata={"source": "a.pdf"}),
        Document(page_content="Bad chunk 2", metadata={"source": "b.pdf"}),
        Document(page_content="Bad chunk 3", metadata={"source": "c.pdf"}),
        Document(page_content="Bad chunk 4", metadata={"source": "d.pdf"}),
        Document(page_content="Sole good chunk in attempt 1", metadata={"source": "e.pdf"}),
    ]
    attempt2_docs = [
        Document(page_content="Target policy chunk A", metadata={"source": "target.pdf", "page": 1}),
        Document(page_content="Target policy chunk B", metadata={"source": "target.pdf", "page": 2}),
    ]

    retriever = FakeChunkRetriever([attempt1_docs, attempt2_docs])
    grader = FakeChunkGrader({
        "Bad chunk 1": False,
        "Bad chunk 2": False,
        "Bad chunk 3": False,
        "Bad chunk 4": False,
        "Sole good chunk in attempt 1": True,
        "Target policy chunk A": True,
        "Target policy chunk B": True,
    })
    rewriter = FakeChunkRewriter(["better targeted policy query"])

    tool = build_document_search_tool(
        retriever,
        grader,
        rewriter,
        max_attempts=2,
        irrelevance_threshold=0.8,
    )
    result = await tool.ainvoke({"query": "vague query"})

    # Attempt 1 had 4/5 irrelevant (0.8 >= 0.8) -> triggered rewriter
    assert retriever.call_count == 2
    assert retriever.queries == ["vague query", "better targeted policy query"]
    assert rewriter.call_count == 1

    # Final result has attempt 2's verified chunks
    assert "Target policy chunk A" in result
    assert "Target policy chunk B" in result
    assert "Bad chunk" not in result


@pytest.mark.asyncio
async def test_tunable_threshold_respects_custom_setting():
    """
    Verify that if the threshold is set to 0.9 (e.g. 90%), 4 out of 5 irrelevant
    chunks (0.8) will NOT trigger a retry, returning the single good chunk immediately.
    """
    attempt1_docs = [
        Document(page_content="Bad chunk 1", metadata={"source": "a.pdf"}),
        Document(page_content="Bad chunk 2", metadata={"source": "b.pdf"}),
        Document(page_content="Bad chunk 3", metadata={"source": "c.pdf"}),
        Document(page_content="Bad chunk 4", metadata={"source": "d.pdf"}),
        Document(page_content="Keep this solitary good chunk", metadata={"source": "e.pdf"}),
    ]

    retriever = FakeChunkRetriever([attempt1_docs])
    grader = FakeChunkGrader({
        "Bad chunk 1": False,
        "Bad chunk 2": False,
        "Bad chunk 3": False,
        "Bad chunk 4": False,
        "Keep this solitary good chunk": True,
    })
    rewriter = FakeChunkRewriter(["should not be called"])

    # Threshold set to 0.9: 4/5 = 0.8 < 0.9 -> accepts single good chunk without retry
    tool = build_document_search_tool(
        retriever,
        grader,
        rewriter,
        max_attempts=2,
        irrelevance_threshold=0.9,
    )
    result = await tool.ainvoke({"query": "my query"})

    assert retriever.call_count == 1
    assert rewriter.call_count == 0
    assert "Keep this solitary good chunk" in result
    assert "Bad chunk" not in result


@pytest.mark.asyncio
async def test_all_chunks_irrelevant_retries_and_graceful_fallback():
    """
    Verify that if all chunks are irrelevant across all attempts, a safe warning note
    is returned with the unconfirmed content without crashing the pipeline.
    """
    docs1 = [Document(page_content="Irrelevant 1", metadata={"source": "x.pdf"})]
    docs2 = [Document(page_content="Irrelevant 2", metadata={"source": "y.pdf"})]

    retriever = FakeChunkRetriever([docs1, docs2])
    grader = FakeChunkGrader({
        "Irrelevant 1": False,
        "Irrelevant 2": False,
    })
    rewriter = FakeChunkRewriter(["query attempt 2"])

    tool = build_document_search_tool(
        retriever,
        grader,
        rewriter,
        max_attempts=2,
        irrelevance_threshold=0.8,
    )
    result = await tool.ainvoke({"query": "impossible query"})

    assert retriever.call_count == 2
    assert rewriter.call_count == 1
    assert "Note: Relevance of these documents could not be fully confirmed." in result
    assert "Irrelevant 2" in result


def test_chunk_grade_model():
    """Verify ChunkGrade schema validation and default fields."""
    grade = ChunkGrade(is_relevant=True, reason="Direct policy statement")
    assert grade.is_relevant is True
    assert grade.reason == "Direct policy statement"

    grade_default = ChunkGrade(is_relevant=False)
    assert grade_default.is_relevant is False
    assert grade_default.reason == ""
