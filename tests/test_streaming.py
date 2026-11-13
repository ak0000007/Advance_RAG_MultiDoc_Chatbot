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

