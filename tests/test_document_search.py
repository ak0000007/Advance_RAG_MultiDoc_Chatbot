import pytest
from langchain_core.documents import Document
from src.tools.document_search import build_document_search_tool

class FakeRetriever:
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


class FakeGrader:
    def __init__(self, responses, raises_exception=False):
        self.responses = responses
        self.call_count = 0
        self.raises_exception = raises_exception

    async def ainvoke(self, input_dict, config=None):
        if self.raises_exception:
            raise ValueError("Grader failed")
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

class FakeRewriter:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    async def ainvoke(self, input_dict, config=None):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class GradeResponse:
    def __init__(self, relevant, reason):
        self.relevant = relevant
        self.reason = reason


@pytest.mark.asyncio
async def test_relevant_first_try():
    docs = [Document(page_content="test doc", metadata={"source": "test"})]
    retriever = FakeRetriever([docs])
    grader = FakeGrader([GradeResponse(True, "Relevant")])
    rewriter = FakeRewriter([])
    
    tool = build_document_search_tool(retriever, grader, rewriter, max_attempts=2)
    result = await tool.ainvoke({"query": "my query"})
    
    assert "test doc" in result
    assert "Note:" not in result
    assert retriever.call_count == 1
    assert grader.call_count == 1
    assert rewriter.call_count == 0


@pytest.mark.asyncio
async def test_irrelevant_first_relevant_second():
    docs1 = [Document(page_content="bad doc", metadata={"source": "test"})]
    docs2 = [Document(page_content="good doc", metadata={"source": "test"})]
    retriever = FakeRetriever([docs1, docs2])
    grader = FakeGrader([GradeResponse(False, "Not relevant"), GradeResponse(True, "Relevant")])
    rewriter = FakeRewriter(["rewritten query"])
    
    tool = build_document_search_tool(retriever, grader, rewriter, max_attempts=2)
    result = await tool.ainvoke({"query": "my query"})
    
    assert "good doc" in result
    assert "bad doc" not in result
    assert retriever.call_count == 2
    assert retriever.queries == ["my query", "rewritten query"]
    assert grader.call_count == 2
    assert rewriter.call_count == 1


@pytest.mark.asyncio
async def test_never_relevant():
    docs1 = [Document(page_content="bad doc 1", metadata={"source": "test"})]
    docs2 = [Document(page_content="bad doc 2", metadata={"source": "test"})]
    retriever = FakeRetriever([docs1, docs2])
    grader = FakeGrader([GradeResponse(False, "No"), GradeResponse(False, "No")])
    rewriter = FakeRewriter(["rewritten query"])
    
    tool = build_document_search_tool(retriever, grader, rewriter, max_attempts=2)
    result = await tool.ainvoke({"query": "my query"})
    
    assert "Note: Relevance of these documents could not be fully confirmed." in result
    assert "bad doc 2" in result
    assert retriever.call_count == 2
    assert grader.call_count == 2
    assert rewriter.call_count == 1


@pytest.mark.asyncio
async def test_empty_retrieval():
    retriever = FakeRetriever([[]])
    grader = FakeGrader([])
    rewriter = FakeRewriter([])
    
    tool = build_document_search_tool(retriever, grader, rewriter, max_attempts=2)
    result = await tool.ainvoke({"query": "my query"})
    
    assert "No relevant documents were found" in result
    assert retriever.call_count == 1
    assert grader.call_count == 0


@pytest.mark.asyncio
async def test_grader_exception():
    docs = [Document(page_content="test doc", metadata={"source": "test"})]
    retriever = FakeRetriever([docs])
    grader = FakeGrader([], raises_exception=True)
    rewriter = FakeRewriter([])
    
    tool = build_document_search_tool(retriever, grader, rewriter, max_attempts=2)
    result = await tool.ainvoke({"query": "my query"})
    
    assert "test doc" in result
    assert "Note:" not in result
    assert retriever.call_count == 1
    assert grader.call_count == 0  # exception caught immediately
    assert rewriter.call_count == 0

