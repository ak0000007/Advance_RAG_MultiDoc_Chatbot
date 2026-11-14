"""
Unit and integration tests for Server-Sent Events (SSE) token streaming.

Tests:
1. Token chunk streaming and completion (event: token, event: done).
2. Intermediate tool start and end status events (event: status).
3. Interrupt / HITL pause during streaming (event: interrupt).
4. Resume chat streaming (POST /chat/resume/stream).
5. Error handling during streaming (event: error).
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from src.api.app import app
from src.api.dependencies import get_active_graph


class MockGraphStreaming:
    """Mock LangGraph workflow supporting astream_events and aget_state."""

    def __init__(self, events_to_emit=None, final_values=None, interrupts=None):
        self.events_to_emit = events_to_emit or []
        self.final_values = final_values or {"answer": "Final synthesized answer"}
        self.interrupts = interrupts or []

    async def astream_events(self, inputs, config=None, version="v2"):
        for ev in self.events_to_emit:
            yield ev

    async def aget_state(self, config):
        task_mock = MagicMock()
        task_mock.interrupts = [MagicMock(value=intr) for intr in self.interrupts]
        snapshot_mock = MagicMock()
        snapshot_mock.values = self.final_values
        snapshot_mock.tasks = [task_mock] if self.interrupts else []
        snapshot_mock.next = ["human_approval"] if self.interrupts else []
        return snapshot_mock


def _parse_sse_response(text: str) -> list[dict]:
    """Parse raw text/event-stream into a list of {event: ..., data: ...} dicts."""
    events = []
    blocks = text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        event_type = "message"
        data_str = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:].strip()
        if data_str:
            try:
                payload = json.loads(data_str)
            except Exception:
                payload = data_str
            events.append({"event": event_type, "data": payload})
    return events


def test_chat_stream_tokens_and_done():
    """Verify /chat/stream emits incremental tokens and final done event."""
    words = ["Travel ", "policy ", "allows ", "premium ", "economy."]
    events = [
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content=w)},
        }
        for w in words
    ]

    mock_graph = MockGraphStreaming(
        events_to_emit=events,
        final_values={"answer": "".join(words)},
    )
    app.dependency_overrides[get_active_graph] = lambda: mock_graph

    try:
        client = TestClient(app)
        response = client.post(
            "/chat/stream",
            json={"question": "What is flight policy?", "thread_id": "thread-stream-1"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        parsed = _parse_sse_response(response.text)
        token_events = [e for e in parsed if e["event"] == "token"]
        assert len(token_events) == 5
        assert [e["data"]["chunk"] for e in token_events] == words

        done_events = [e for e in parsed if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["thread_id"] == "thread-stream-1"
        assert done_events[0]["data"]["answer"] == "".join(words)
        assert done_events[0]["data"]["interrupted"] is False
    finally:
        app.dependency_overrides.pop(get_active_graph, None)


def test_chat_stream_tool_status_events():
    """Verify tool start and end events are emitted as status progress."""
    events = [
        {
            "event": "on_tool_start",
            "name": "search_documents",
            "data": {"input": {"query": "hotel policy"}},
        },
        {
            "event": "on_tool_end",
            "name": "search_documents",
            "data": {"output": "Found 3 results"},
        },
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content="Found hotel policy.")},
        },
    ]

    mock_graph = MockGraphStreaming(
        events_to_emit=events,
        final_values={"answer": "Found hotel policy."},
    )
    app.dependency_overrides[get_active_graph] = lambda: mock_graph

    try:
        client = TestClient(app)
        response = client.post(
            "/chat/stream",
            json={"question": "Search hotels", "thread_id": "thread-tools-1"},
        )
        assert response.status_code == 200

        parsed = _parse_sse_response(response.text)
        status_events = [e for e in parsed if e["event"] == "status"]
        assert len(status_events) == 2
        assert status_events[0]["data"]["stage"] == "tool_start"
        assert status_events[0]["data"]["tool"] == "search_documents"
        assert status_events[1]["data"]["stage"] == "tool_end"
        assert status_events[1]["data"]["tool"] == "search_documents"

        token_events = [e for e in parsed if e["event"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["data"]["chunk"] == "Found hotel policy."
    finally:
        app.dependency_overrides.pop(get_active_graph, None)


def test_chat_stream_interrupt_approval():
    """Verify /chat/stream emits interrupt event when human approval is triggered."""
    approval_payload = {
        "__requires_approval__": True,
        "action": "update_booking",
        "record_id": "BK-0099",
        "changes": {"travelers": 4},
    }
    mock_graph = MockGraphStreaming(
        events_to_emit=[],
        final_values={},
        interrupts=[approval_payload],
    )
    app.dependency_overrides[get_active_graph] = lambda: mock_graph

    try:
        client = TestClient(app)
        response = client.post(
            "/chat/stream",
            json={"question": "Update booking BK-0099 to 4 travelers", "thread_id": "thread-intr-1"},
        )
        assert response.status_code == 200
        parsed = _parse_sse_response(response.text)
        interrupt_events = [e for e in parsed if e["event"] == "interrupt"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0]["data"]["interrupted"] is True
        assert interrupt_events[0]["data"]["thread_id"] == "thread-intr-1"
        assert interrupt_events[0]["data"]["approval_request"]["record_id"] == "BK-0099"
    finally:
        app.dependency_overrides.pop(get_active_graph, None)


def test_resume_stream():
    """Verify /chat/resume/stream resumes paused graph and streams response."""
    words = ["Booking ", "BK-0099 ", "successfully ", "updated."]
    events = [
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content=w)},
        }
        for w in words
    ]

    mock_graph = MockGraphStreaming(
        events_to_emit=events,
        final_values={"answer": "".join(words)},
        interrupts=[{"action": "update_booking"}],  # snapshot is in interrupted state
    )
    app.dependency_overrides[get_active_graph] = lambda: mock_graph

    try:
        client = TestClient(app)
        response = client.post(
            "/chat/resume/stream",
            json={"thread_id": "thread-resume-1", "decision": "Approve"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        parsed = _parse_sse_response(response.text)
        token_events = [e for e in parsed if e["event"] == "token"]
        assert len(token_events) == 4
        assert [e["data"]["chunk"] for e in token_events] == words
    finally:
        app.dependency_overrides.pop(get_active_graph, None)


def test_chat_stream_error_handling():
    """Verify /chat/stream yields error event when an unexpected exception occurs."""
    class FailingGraph:
        async def astream_events(self, inputs, config=None, version="v2"):
            raise RuntimeError("Database connection timed out during execution")
            yield  # pragma: no cover

    app.dependency_overrides[get_active_graph] = lambda: FailingGraph()
    try:
        client = TestClient(app)
        response = client.post(
            "/chat/stream",
            json={"question": "Trigger error", "thread_id": "thread-err-1"},
        )
        assert response.status_code == 200
        parsed = _parse_sse_response(response.text)
        err_events = [e for e in parsed if e["event"] == "error"]
        assert len(err_events) == 1
        assert "Database connection timed out" in err_events[0]["data"]["detail"]
    finally:
        app.dependency_overrides.pop(get_active_graph, None)


def test_chat_stream_skips_tool_internal_llm_and_grading_tokens():
    """
    Verify /chat/stream strictly suppresses internal LLM tokens from tools,
    such as CRAG chunk relevance graders and query rewriters, emitting only
    the final agent tokens.
    """
    events = [
        # 1. Intent classifier (should be skipped)
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "classify_intent"},
            "data": {"chunk": AIMessageChunk(content='{"domain": "docs"}')},
        },
        # 2. Agent decides to call search_documents with tool_call_chunks (should be skipped)
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "chunk": AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": "search_documents", "args": '{"query": "leave policy"}'}],
                )
            },
        },
        # 3. Tool execution starts
        {
            "event": "on_tool_start",
            "name": "search_documents",
            "data": {"input": {"query": "leave policy"}},
        },
        # 4. Internal CRAG chunk grader output inside tools (MUST be skipped)
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "tools"},
            "data": {"chunk": AIMessageChunk(content='{"relevant": false, "reason": "unrelated travel policy"}')},
        },
        # 5. Internal rewriter chain output inside tools (MUST be skipped)
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "tools"},
            "data": {"chunk": AIMessageChunk(content="leave policy time off vacation PTO guidelines")},
        },
        # 6. Tool execution ends
        {
            "event": "on_tool_end",
            "name": "search_documents",
            "data": {"output": "No relevant documents found"},
        },
        # 7. Final agent response to user (MUST be streamed)
        {
            "event": "on_chat_model_stream",
            "name": "ChatModel",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content="I couldn't find specific information regarding leave policies.")},
        },
    ]

    mock_graph = MockGraphStreaming(
        events_to_emit=events,
        final_values={"answer": "I couldn't find specific information regarding leave policies."},
    )
    app.dependency_overrides[get_active_graph] = lambda: mock_graph

    try:
        client = TestClient(app)
        response = client.post(
            "/chat/stream",
            json={"question": "Whats is the leave policy", "thread_id": "thread-stream-crag-1"},
        )
        assert response.status_code == 200

        parsed = _parse_sse_response(response.text)
        token_events = [e for e in parsed if e["event"] == "token"]

        # Only exactly ONE token event from the agent should be emitted!
        assert len(token_events) == 1
        assert token_events[0]["data"]["chunk"] == "I couldn't find specific information regarding leave policies."

        # Verify NO JSON grader strings or query rewriter queries were leaked into tokens
        for te in token_events:
            chunk = te["data"]["chunk"]
            assert "relevant" not in chunk
            assert "classify_intent" not in chunk
            assert "vacation PTO" not in chunk

        done_events = [e for e in parsed if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["answer"] == "I couldn't find specific information regarding leave policies."
    finally:
        app.dependency_overrides.pop(get_active_graph, None)


def test_unconfirmed_documents_not_emitted_as_citations():
    """
    Verify that when document retrieval cannot confirm relevance, fallback chunks
    are NOT extracted as valid citations in extract_citations_from_messages.
    """
    from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
    from src.retrieval.citations import extract_citations_from_messages

    messages = [
        HumanMessage(content="Whats is the leave policy"),
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "leave policy"}, "id": "tc-1"}]),
        ToolMessage(
            name="search_documents",
            content=(
                "Note: Relevance of these documents could not be fully confirmed.\n\n"
                '<document id="doc-unrelated-1" source="travel_policy.txt" title="Travel Policy" page="1">\n'
                "Airline tickets and travel vouchers...\n"
                "</document>"
            ),
            tool_call_id="tc-1",
        ),
        AIMessage(content="I couldn't find specific information regarding leave policies in the internal knowledge base."),
    ]

    citations = extract_citations_from_messages(messages, current_turn_only=True)
    assert citations == []


